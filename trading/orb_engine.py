"""ORB (Opening Range Breakout) live trading engine.

Third strategy alongside bull flag + MACD wave in main.py process.
Fires at 9:35 ET on gap-up movers breaking the first 5-minute range high.

Lifecycle (per trading day):
  08:30 ET  build_universe() — seed candidates from bull flag cache
  09:30 ET  market open; bars start streaming via shared StopMonitor
  09:35 ET  5-min range closes; on_bar_close() computes range_high/low per symbol
  09:35+    check_entries() — features → filter → rank → dedup → plan → submit
  09:35+    check_exits()   — drain StopMonitor queue (lock/stop/force_close)
  15:45 ET  force_close_all()
  16:00+    reset_daily()

Config: orb.yaml. Feature flag: strategy.enabled (master kill switch).

Trade tagging: every DB row has strategy='orb' so per-strategy P&L filtering works.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import os
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

from trading.buy_stop_guard import (
    BuyStopAction, BuyStopDecision, evaluate_buy_stop,
)
from trading.orb_correlation import dedup_candidates, symbol_family, symbol_super_group
from trading.orb_filter import (
    FeatureParam, assign_quintile, composite_score, load_feature_params,
)
from trading.orb_conviction import apply_adaptive_mult, load_adaptive_mults
from trading.orb_asset_class import (
    UNKNOWN as CLASS_UNKNOWN, classify_asset, effective_has_news,
    load_class_map, underlying_anchor,
)
from trading.orb_catalyst_veto import (
    DEFAULT_MIN_COHORT, anchor_cohort_counts, catalyst_veto_applies,
)
from trading.orb_pm_mult import (
    DEFAULT_HIGH_CUT_USD, DEFAULT_HIGH_MULT, DEFAULT_HIGH_MULT_NEWS,
    compute_pm_dollar_vol, pm_size_multiplier,
)
from trading.orb_pdr_veto import (
    DEFAULT_MIN_PDR_PCT, compute_prev_day_range_pct, pdr_veto_applies,
)
from trading.orb_planner import OrbTradePlan, OrbTradePlanner, PlannerReject
from trading.orb_touchgo_filter import (
    TouchgoConfig, evaluate_rule_d, evaluate_rule_m, find_breakout_bar_ts,
    load_touchgo_config,
)
from trading.exit_reasons import ExitReason
from trading import touchgo_audit as _tg_audit
from trading.orphan_reconciler import (
    ReconcilerConfig, reconcile_strategy_orphans,
)
from trading.stop_monitor import build_exit_update


logger = logging.getLogger(__name__)


# Max calendar-day age tolerated for a cached prev-day daily bar before the
# ORB feature path force-refetches it from Alpaca. The prior *trading* day is
# at most 4 calendar days behind "today" (Tue after a Monday market holiday:
# prev trading day = Friday). A newest cached bar older than that means the
# nightly broad-universe refresh skipped this symbol — and silently using its
# stale close corrupts gap_pct / prev_day_close_position, which flips the
# composite quintile and changes which symbols ORB trades (ASTN/PLTG on
# 2026-05-29: prod's daily_bars were frozen at 05-22).
_PREV_BAR_STALENESS_MAX_DAYS = 4


def _newest_bar_date(bars) -> Optional[date]:
    """Most-recent bar date from a list of daily-bar dicts/rows, or None.

    Normalizes the 'date' field, which is a 'YYYY-MM-DD' string from the DB
    cache but a datetime.date from the Alpaca range fetch.
    """
    latest: Optional[date] = None
    for b in bars:
        raw = b.get('date') if isinstance(b, dict) else getattr(b, 'date', None)
        if raw is None:
            continue
        if isinstance(raw, datetime):
            d = raw.date()
        elif isinstance(raw, date):
            d = raw
        else:
            try:
                d = datetime.strptime(str(raw)[:10], '%Y-%m-%d').date()
            except (ValueError, TypeError):
                continue
        if latest is None or d > latest:
            latest = d
    return latest

STRATEGY_NAME = 'orb'

# Market hours (ET)
ET_OFFSET_EDT_HOURS = 4   # Mar-Nov (EDT: ET+4 = UTC)
ET_OFFSET_EST_HOURS = 5   # Nov-Mar (EST: ET+5 = UTC)


@dataclass
class RangeData:
    """5-min opening range for one symbol (9:30-9:35 ET).

    range_open: price at the open of the 9:30 bar — used as the REFERENCE
    price for gap/size/high_20d features (matches BT's study_orb_features.py,
    where `open_p = rb['open'].iloc[0]` is the denominator for these features).
    Do NOT use range_high as the reference — on breakout days this systematically
    shifts gap_pct and range_size_pct.
    """
    symbol: str
    range_high: float
    range_low: float
    range_volume: int        # total volume across the 5 range bars
    range_avg_bar_range_pct: float  # avg (high-low)/close across bars
    range_close: float       # close of the 5th bar (9:34 bar)
    range_start_ts: pd.Timestamp
    bars_green: int = 0      # count of green bars in range
    last_bar_green: bool = False
    range_open: float = 0.0  # open price of 9:30 bar (reference for gap features)


@dataclass
class CandidateState:
    """Per-symbol state for one trading day. State is ADDITIVE across ticks —
    build_universe is idempotent and preserves fields already populated."""
    symbol: str
    # Range data populated when 5-min range closes (once, then immutable)
    range_data: Optional[RangeData] = None
    # Features computed at 9:35 (one-shot)
    features: Optional[Dict[str, float]] = None
    composite: Optional[float] = None
    quintile: Optional[str] = None
    # Plan / order tracking
    plan_submitted: bool = False
    order_id: Optional[str] = None        # Alpaca order_id for the stop-buy entry
    order_submitted_at: Optional[datetime] = None  # for 10:35 ET time-stop cancellation
    rejected_reason: Optional[str] = None
    bars_subscribed: bool = False         # tracks whether we've asked StopMonitor to stream bars


@dataclass
class OpenPosition:
    """Tracked ORB position.

    While order_id is non-empty: entry order is PENDING (no fill yet).
    Once order_id is cleared: position is filled + StopMonitor watching.
    Matches the MACD wave pattern.
    """
    symbol: str
    entry_price: float             # fill_price when filled; limit_price while pending
    stop_price: float
    shares: int
    trade_id: int
    order_id: str                  # cleared ('') once fill confirmed + stop_monitor armed
    entry_time: datetime           # submit time while pending; fill time once filled
    range_high: float
    range_low: float
    lock_arm_at_r: float
    lock_stop_r: float
    composite_score: float
    quintile: str
    # Safety-net leg IDs for cleanup on exit (populated from submit response)
    tp_leg_id: str = ''
    sl_leg_id: str = ''
    # Slippage attribution — set at entry submit, consumed at fill time.
    # bar_close_price: BT reference level (range_high — what BT uses as the
    #   fill trigger; anything above it is real slippage on top of the 30bps
    #   planner buffer). drift_bar_to_fill_bps = (fill - this) / this × 10000.
    # order_submitted_at: submit timestamp (UTC) — enables submit_to_fill_ms.
    # entry_quote_ask: NBBO ask at submit — enables drift_ask_to_fill_bps.
    bar_close_price: Optional[float] = None
    order_submitted_at: Optional[datetime] = None
    entry_quote_ask: Optional[float] = None
    # Touchgo filter state (Rule M / Rule D).
    # breakout_bar_ts = the MARKET breakout bar (first 1-min bar with
    # high > range_high), captured during the pending/early phase by
    # _ensure_breakout_bar_ts — what BT keys touchgo to (parity). Rule M is
    # evaluated at the bar event with timestamp == breakout_bar_ts, Rule D at
    # +1min. Under the legacy breakout_bar_source='fill' rollback it is instead
    # the minute-floor of the fill time (pre-2026-06 behaviour).
    breakout_bar_ts: Optional[datetime] = None
    rule_m_evaluated: bool = False
    rule_d_evaluated: bool = False
    # FABC 2026-06-09 fix: track when we first saw a partial fill, so we can
    # tell the difference between "broker is still working the order" and
    # "broker stalled mid-fill, time to cancel + accept the partial". None
    # until first partial observed; reset on terminal-state transition.
    first_partial_at: Optional[datetime] = None
    # KOLD/PLTU/TSDD 2026-07-03 fix: the position's own range-end timestamp
    # (range_start + range_minutes), captured at entry submit from the
    # candidate's range_data. Breakout-bar keying uses THIS as the search
    # anchor instead of re-deriving a "session open" from the current bars
    # window — streamed windows usually lack the 9:30 bar (Alpaca WS delivers
    # from subscribe-time forward), which made the old anchor heuristic fall
    # through to the 14:30Z bar (10:30 ET during EDT) and key Rule M/D one
    # hour late. None on rehydrated positions → capture declines (touchgo
    # correctly inert for positions recovered mid-session).
    range_end_ts: Optional[datetime] = None
    # Highest filled_qty observed so far on this order. Used by the
    # partial-fill polling loop to suppress duplicate INFO logs when the
    # broker reports the same qty across multiple polls.
    last_observed_filled_qty: int = 0


class ORBEngine:
    """Opening Range Breakout trading engine.

    Thread model:
      * Main thread calls build_universe / check_entries / check_exits / force_close.
      * StopMonitor daemon thread emits bar-close events (via register_on_stop_monitor)
        and stop-exit events (via drain_exit_events filtered by strategy).
      * Bounded queue buffers bar events; main loop drains each tick.
    """

    STRATEGY_NAME = STRATEGY_NAME

    def __init__(
        self,
        alpaca_client,
        db,
        stop_monitor,
        notifier=None,
        config: Optional[dict] = None,
        order_stream=None,
        position_manager=None,
        dry_run: bool = False,
    ):
        """Initialize.

        Args:
            alpaca_client: AlpacaClient authenticated to ORB's account (paper in Phase 1).
            db: Database for trade persistence (shared across strategies).
            stop_monitor: Shared StopMonitor (market-data WS on main account, exits routed
                by strategy via alpaca_clients_by_strategy dict — see main.py wiring).
            notifier: Optional TelegramNotifier for alerts (prefixes with [ORB]).
            config: Parsed orb.yaml dict.
            order_stream: OrderStreamWatcher for this account's fills (optional but
                strongly recommended for fast fill detection).
            position_manager: Optional PositionManager for cross-strategy risk gates.
            dry_run: If True, logs plans but does NOT submit orders (for paper sanity).
        """
        cfg = config or {}
        self.alpaca = alpaca_client
        self.db = db
        self.stop_monitor = stop_monitor
        self.order_stream = order_stream
        self.notifier = notifier
        self.position_manager = position_manager
        self.dry_run = dry_run

        # Master kill switch — matches orb.yaml `strategy.enabled`
        self.enabled = bool(cfg.get('strategy', {}).get('enabled', False))

        # Config sections
        uni = cfg.get('universe', {})
        entry_cfg = cfg.get('entry', {})
        exit_cfg = cfg.get('exit', {})
        sizing_cfg = cfg.get('sizing', {})
        risk_cfg = cfg.get('risk', {})
        dedup_cfg = cfg.get('dedup', {})
        conflict_cfg = cfg.get('conflict', {})
        notifications_cfg = cfg.get('notifications', {}).get('telegram', {})
        safety_cfg = cfg.get('safety_net', {})
        # FABC 2026-06-09: how long to keep polling a partially-filled order
        # before giving up and accepting whatever filled. Set high enough that
        # a normal multi-fill order ("partial 1438 → partial 3188 → filled
        # 3188" inside a couple seconds) always reaches terminal `filled`
        # before timing out. Set low enough that a truly stuck broker order
        # doesn't strand the engine in a never-confirms state.
        fill_handling_cfg = cfg.get('fill_handling', {})
        self.partial_fill_stall_seconds_max = int(
            fill_handling_cfg.get('partial_fill_stall_seconds_max', 60)
        )
        # 2026-07-03 selection-race fix (CRCD/AVEX/FABC/RGNX): the 9:35:01
        # first-ranking event must see the FULL candidate field, not just the
        # subset whose 9:34 bars happened to have consolidated. Two layers:
        #   sweep_retry_delay_s  — post-open sweep re-fetches still-rangeless
        #                          candidates once after this delay (0=off).
        #   first_rank_grace_s   — check_entries defers the day's FIRST
        #                          placement burst while pool candidates are
        #                          still rangeless within this many seconds
        #                          after the range end (backstop; 0=off).
        self.sweep_retry_delay_s = float(entry_cfg.get('sweep_retry_delay_s', 4.0))
        self.first_rank_grace_s = float(entry_cfg.get('first_rank_grace_s', 25.0))

        self.range_minutes = int(entry_cfg.get('range_minutes', 5))
        self.entry_slip_bps = float(entry_cfg.get('entry_slip_bps', 30))
        self.time_stop_minutes = int(entry_cfg.get('time_stop_minutes', 60))
        self.max_spread_bps = float(entry_cfg.get('max_spread_bps', 300))

        # Universe criteria (BT-parity: gap >= 5%, vol >= 500K, price $3-30)
        self.universe_source = uni.get('source', 'bull_flag_cache')
        self.universe_min_price = float(uni.get('min_price', 3.0))
        self.universe_max_price = float(uni.get('max_price', 30.0))
        self.universe_min_gap_pct = float(uni.get('min_gap_pct', 5.0))
        self.universe_min_prev_volume = int(uni.get('min_prev_volume', 500_000))

        self.max_concurrent = int(sizing_cfg.get('max_concurrent', 4))

        self.daily_loss_limit_usd = float(risk_cfg.get('daily_loss_limit_usd', -5000))
        self.safety_sl_pct = float(safety_cfg.get('sl_pct', 0.10))

        self.dedup_by_family = bool(dedup_cfg.get('by_family', True))
        self.dedup_by_super_group = bool(dedup_cfg.get('by_super_group', True))

        self.skip_if_any_strategy_has_symbol = bool(
            conflict_cfg.get('skip_if_any_strategy_has_symbol', True))

        # Filter + conviction params (loaded once, used repeatedly)
        filter_cfg = cfg.get('filter', {})
        self.filter_threshold = float(filter_cfg.get('threshold', 0.0))
        # Q1 filter — drop bottom-quintile candidates at ranking time. Default on.
        # Q1 was TRAIN-positive (+$6K) but OOS-negative (VAL -$5.1K, HOQ1+ -$3.4K).
        # BT validation: study_orb_q1q2_filter.py.
        self.skip_q1 = bool(filter_cfg.get('skip_q1', True))
        # PDR veto — skip picks whose PREV day range was quiet (<= min pct).
        # NO-REFILL: the vetoed pick's slot stays empty (backfill tested
        # toxic — see trading/orb_pdr_veto.py docstring for evidence).
        # Replica-validated +$55K / MDD −31% over Jan'25–Jul'26. Default on.
        pdr_cfg = filter_cfg.get('prev_day_range_veto', {}) or {}
        self.pdr_veto_enabled = bool(pdr_cfg.get('enabled', True))
        self.pdr_veto_min_pct = float(
            pdr_cfg.get('min_prev_day_range_pct', DEFAULT_MIN_PDR_PCT))
        _pdr_env = os.environ.get('ORB_PDR_VETO')
        if _pdr_env is not None:
            self.pdr_veto_enabled = _pdr_env.strip().lower() not in (
                '0', 'false', 'no', 'off', '')
        _pdr_min_env = os.environ.get('ORB_PDR_VETO_MIN_PCT')
        if _pdr_min_env:
            self.pdr_veto_min_pct = float(_pdr_min_env)
        self._pdr_vetoed_today: Set[str] = set()
        # Premarket dollar-volume sizing mult (2026-07-04, upsize-only).
        # Shared math: trading/orb_pm_mult.py. Cut frozen from H1-2025 TRAIN.
        sizing_pm_cfg = (cfg.get('sizing', {}) or {}).get('pm_dollar_vol_mult', {}) or {}
        self.pm_mult_enabled = bool(sizing_pm_cfg.get('enabled', True))
        self.pm_mult_high_cut = float(sizing_pm_cfg.get('high_cut_usd', DEFAULT_HIGH_CUT_USD))
        self.pm_mult_high = float(sizing_pm_cfg.get('high_mult', DEFAULT_HIGH_MULT))
        _pm_env = os.environ.get('ORB_PM_MULT')
        if _pm_env is not None:
            self.pm_mult_enabled = _pm_env.strip().lower() not in (
                '0', 'false', 'no', 'off', '')
        self._pm_dollar_vols: Dict[str, Optional[float]] = {}
        self._pm_fetch_done_day: Optional[date] = None
        # News gate on the PM mult (2026-07-10 A2 ship,
        # research/orb_news_catalyst_jul2026.md): above-cut trades WITH
        # premarket news → high_mult_news (2.0); WITHOUT news → high_mult
        # (de-boosted to 1.0 — the flat bucket). Fail-open: news fetch
        # failure → has_news None → no news boost. Rollback:
        # news_gate: false + high_mult: 1.5 (legacy), or ORB_PM_NEWS_GATE=0.
        self.pm_news_gate = bool(sizing_pm_cfg.get('news_gate', True))
        self.pm_mult_high_news = float(
            sizing_pm_cfg.get('high_mult_news', DEFAULT_HIGH_MULT_NEWS))
        _ng_env = os.environ.get('ORB_PM_NEWS_GATE')
        if _ng_env is not None:
            self.pm_news_gate = _ng_env.strip().lower() not in (
                '0', 'false', 'no', 'off', '')
        # symbol -> {'n_articles': int, 'headline': str} | None (fetch failed)
        self._news_flags: Dict[str, Optional[Dict]] = {}
        self._news_fetch_done_day: Optional[date] = None
        # 9:33 indexing-lag second pass (once/day, upgrade-only)
        self._news_refresh_done_day: Optional[date] = None
        # EoD lag-audit snapshot target (cwd-independent; tests override)
        self._news_snapshot_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'logs')
        # Asset-class rule (2026-07-11): news boost requires POSITIVE
        # identification as a common stock — wrappers/unknown are
        # structurally ineligible (trading/orb_asset_class.py).
        self._asset_class: Dict[str, str] = {}
        self._class_map: Optional[Dict[str, str]] = None
        # Catalyst-required veto (2026-07-18, owner-approved −$36K):
        # newsless-and-alone picks vetoed, slot consumed. Rollback:
        # filter.catalyst_veto.enabled: false or ORB_CATALYST_VETO=0.
        _cv_cfg = (filter_cfg.get('catalyst_veto', {}) or {})
        self.catalyst_veto_enabled = bool(_cv_cfg.get('enabled', True))
        self.catalyst_min_cohort = int(
            _cv_cfg.get('min_cohort', DEFAULT_MIN_COHORT))
        _cv_env = os.environ.get('ORB_CATALYST_VETO')
        if _cv_env is not None:
            self.catalyst_veto_enabled = _cv_env.strip().lower() not in (
                '0', 'false', 'no', 'off', '')
        self._anchor_cache: Dict[str, Optional[str]] = {}
        self._class_names: Dict[str, str] = {}
        # Touchgo filter (Rule M = entry-bar close-pos < 0.5; Rule D = bar-1
        # revert ≥ 0.75R → exit at entry -0.5R). Shared module
        # trading/orb_touchgo_filter.py imported by both LIVE and BT for parity.
        # Walk-forward validated +$27K OOS / +$44K full-timeline. Default on.
        self.touchgo_cfg: TouchgoConfig = load_touchgo_config(
            filter_cfg.get('touchgo', {})
        )
        # Buy-stop rejection guard (shared with bull flag via
        # trading/buy_stop_guard.py — parity by construction). Reads
        # config.yaml::trading.marketable_limit_fallback for enabled flag +
        # rebump_buffer. Defaults to enabled with $0.02 buffer.
        # See docs/irez_ttgt_paper_vs_prod_divergence.md.
        try:
            from config import Config as _Config
            self._buy_stop_guard_cfg = _Config().marketable_limit_fallback_cfg
        except Exception as e:
            logger.warning(
                f"ORBEngine: failed to load marketable_limit_fallback_cfg: {e} "
                f"— defaulting to enabled with $0.02 buffer"
            )
            self._buy_stop_guard_cfg = {"enabled": True, "rebump_buffer": 0.02}
        self.z_params: Dict[str, FeatureParam] = (
            load_feature_params(filter_cfg) if filter_cfg.get('features') else {}
        )
        self.quintile_cutoffs: List[float] = list(cfg.get('quintile_cutoffs', []))
        if len(self.quintile_cutoffs) != 4:
            raise ValueError(
                f"orb.yaml quintile_cutoffs must be length 4, got {self.quintile_cutoffs}"
            )
        self.adaptive_mults = load_adaptive_mults(cfg.get('adaptive_mults', {}))

        # Ranking order
        ranking_cfg = cfg.get('ranking', {})
        self.ranking_order = ranking_cfg.get('order', ['Q4', 'Q5', 'Q3', 'Q2', 'Q1'])

        # Force close time (ET)
        fc_str = exit_cfg.get('force_close_time_et', '15:45')
        hour, minute = [int(x) for x in fc_str.split(':')]
        self.force_close_hour_et = hour
        self.force_close_minute_et = minute

        # Last-entry cutoff (ET). BT picks top-K once at 9:35 ET and never
        # submits new entries afterward. Live allows a short window for
        # bar-stream latency / backfill / candidates arriving slowly, but
        # hard-blocks new submits past this time. Default 10:00 ET = 25-min
        # window post range-close. Raise to be more permissive; lower to
        # tighten to BT-exact behavior. See `check_entries` for the guard.
        le_str = entry_cfg.get('last_entry_submit_time_et', '10:00')
        le_hour, le_minute = [int(x) for x in le_str.split(':')]
        self.last_entry_hour_et = le_hour
        self.last_entry_minute_et = le_minute
        self._late_entry_cutoff_logged = False
        # Post-open range-sweep done flag (one-shot per day). See _ensure_ranges_post_open.
        self._post_open_range_sweep_done = False
        # The 4s consolidation-lag retry inside the sweep runs at most once
        # per day, even when the grace gate re-arms the sweep (2026-07-04).
        self._sweep_retry_used_today = False

        # Planner (stateless, built once)
        self.planner = OrbTradePlanner(cfg)

        # Universe (refreshed daily at 8:30 ET)
        self.universe: Set[str] = set()
        self.universe_date: Optional[str] = None  # 'YYYY-MM-DD'

        # Per-day state
        self.candidates: Dict[str, CandidateState] = {}   # symbol -> state
        self.open_positions: Dict[str, OpenPosition] = {}  # symbol -> position
        self.daily_pnl: float = 0.0
        self.daily_loss_limit_logged: bool = False

        # Fill-rate telemetry. Backtest expects ~73% overall fill rate on
        # top-K picks (see study_orb_fill_rate.py validation). Alerts if
        # live numbers diverge meaningfully from expectation.
        self.daily_n_placed: int = 0      # orders submitted today
        self.daily_n_filled: int = 0      # orders that actually filled today
        self.daily_n_time_stop_canceled: int = 0  # unfilled past 10:35 ET
        self.daily_summary_sent: bool = False

        # Graceful shutdown flag (set by signal handler in main.py).
        # Checked in check_entries / check_exits to bail early.
        self.shutdown_requested: bool = False

        # Bar event queue (fed by StopMonitor bar handler, drained by scanner tick)
        self._bar_event_queue: "queue.Queue" = queue.Queue(maxsize=1000)
        self._bar_queue_full_logged = False
        self._bar_handler_registered = False

        # Rolling 1-min bars per symbol (kept in-memory; used to build range data)
        self._bar_windows: Dict[str, List[dict]] = {}

        # Notification toggles
        self.notify_on_startup = bool(notifications_cfg.get('notify_on_startup', True))
        self.notify_on_entry = bool(notifications_cfg.get('notify_on_entry', True))
        self.notify_on_exit = bool(notifications_cfg.get('notify_on_exit', True))
        self.notify_on_spread_skip = bool(notifications_cfg.get('notify_on_spread_skip', True))
        self.notify_on_capital_exhausted = bool(notifications_cfg.get('notify_on_capital_exhausted', True))
        self.notify_on_daily_loss_limit = bool(notifications_cfg.get('notify_on_daily_loss_limit', True))
        self.notify_on_force_close = bool(notifications_cfg.get('notify_on_force_close', True))

        # FC verify/retry tunables (2026-05-07 hardening). Defaults matched to
        # Alpaca paper ACK latency; tests can shrink these for fast execution.
        fc_cfg = notifications_cfg.get('force_close_verify', {}) if isinstance(
            notifications_cfg.get('force_close_verify', {}), dict) else {}
        self.fc_verify_max_wait_s = float(fc_cfg.get('max_wait_s', 10.0))
        self.fc_verify_poll_interval_s = float(fc_cfg.get('poll_interval_s', 1.0))
        self.fc_retry_count = int(fc_cfg.get('retry_count', 3))
        self.fc_retry_backoffs_s = list(fc_cfg.get('retry_backoffs_s', [5, 10, 15]))

        tg_prefix = notifications_cfg.get('prefix', '[ORB]')
        self.tg_prefix = tg_prefix

        logger.info(
            f"ORBEngine init: enabled={self.enabled}, dry_run={self.dry_run}, "
            f"max_concurrent={self.max_concurrent}, "
            f"risk_per_trade=${self.planner.risk_per_trade_usd:,.0f}, "
            f"per_pos_cap=${self.planner.per_pos_cap_usd:,.0f}, "
            f"filter_features={list(self.z_params.keys())}"
        )
        # veto arming is grep-verifiable at boot (deliberate-rules
        # doctrine: a silently-disarmed gate must be visible)
        logger.info(
            f"ORBEngine gates: catalyst_veto={self.catalyst_veto_enabled} "
            f"(min_cohort={self.catalyst_min_cohort}), "
            f"pdr_veto={self.pdr_veto_enabled}, "
            f"touchgo={self.touchgo_cfg.master_enabled}, "
            f"skip_q1={self.skip_q1}, "
            f"pm_mult={self.pm_mult_enabled} "
            f"(news_gate={self.pm_news_gate})"
        )

    # =====================================================================
    # Universe management
    # =====================================================================

    def build_universe(self, source_loader=None) -> int:
        """Seed or EXTEND today's candidate symbols.

        IDEMPOTENT: can be called repeatedly; preserves range_data, composite,
        etc. for symbols already in the universe. New symbols get a fresh
        CandidateState + bar-stream subscription. Day-boundary reset happens
        via reset_daily().

        Args:
            source_loader: callable returning List[str] of candidate symbols.

        Returns:
            Total number of symbols in the universe after this call.
        """
        if source_loader is None:
            logger.warning("ORBEngine.build_universe: no source_loader — universe unchanged")
            return len(self.universe)
        try:
            symbols = list(source_loader())
        except Exception as e:
            logger.error(f"ORBEngine.build_universe: loader failed: {e}")
            return len(self.universe)

        new_syms = set(symbols) - self.universe
        if not new_syms:
            return len(self.universe)

        for sym in new_syms:
            self.candidates[sym] = CandidateState(symbol=sym)
            self._subscribe_bars(sym)
        self.universe |= new_syms
        self.universe_date = datetime.now(timezone.utc).date().isoformat()
        logger.info(
            f"ORBEngine.build_universe: +{len(new_syms)} new candidates "
            f"(universe now {len(self.universe)}) for {self.universe_date}"
        )
        return len(self.universe)

    def build_orb_universe_from_snapshots(self, candidate_symbols: Optional[List[str]] = None) -> List[str]:
        """Build ORB universe by querying Alpaca snapshots matching BT criteria.

        BT validated on: gap >= 5%, prev-day volume >= 500K, price $3-30.
        (study_orb_broad.py `load_broad_universe`)

        Args:
            candidate_symbols: optional pre-filter (e.g., scanner's active list).
                If None, queries all active US equities (slow — ~8000 symbols).

        Returns:
            List of symbols passing ORB's universe criteria today.
        """
        # Fetch snapshots for candidates — single API call, up to a few hundred symbols.
        if candidate_symbols is None or len(candidate_symbols) == 0:
            logger.info("ORB: build_orb_universe_from_snapshots called with no candidates")
            return []
        try:
            snapshots = self.alpaca.get_snapshots(list(candidate_symbols)) \
                if hasattr(self.alpaca, 'get_snapshots') else {}
        except Exception as e:
            logger.warning(f"ORB: get_snapshots failed: {e}")
            return []

        keep: List[str] = []
        for sym, snap in (snapshots or {}).items():
            try:
                # AlpacaClient.get_snapshots returns a FLAT dict per symbol:
                #   {'open', 'high', 'low', 'close', 'volume',           # today's daily bar
                #    'prev_close', 'prev_volume',                         # yesterday's daily bar
                #    'latest_price', 'bid_price', 'ask_price', ...}       # latest trade/quote
                # Prior code tried `snap.daily_bar.open` nested access — that's how
                # Alpaca's raw SDK object looks, but alpaca_client.py flattens it.
                open_price = float(snap.get('open', 0) or 0) if isinstance(snap, dict) else 0
                if open_price <= 0:
                    # Fall back to latest trade price (used when today's daily bar
                    # hasn't been ticked yet — rare but seen right at 9:30 ET).
                    lp = snap.get('latest_price', 0) if isinstance(snap, dict) else 0
                    open_price = float(lp or 0)
                if open_price <= 0:
                    continue
                prev_close = float(snap.get('prev_close', 0) or 0) if isinstance(snap, dict) else 0
                prev_volume = int(snap.get('prev_volume', 0) or 0) if isinstance(snap, dict) else 0
                # Vendor-corpse gate (2026-07-23: ORIS/CUK/MIGI entered on
                # snapshots whose daily bar was WEEKS old — dead symbols
                # can't trade, but they consumed news/PM/anchor prefetch
                # work and false-alarmed the selection observer). A daily
                # bar dated before today = no market data today = skip.
                # Fail-open: missing date field passes (older client).
                bar_date = snap.get('daily_bar_date') if isinstance(snap, dict) else None
                if bar_date:
                    try:
                        from zoneinfo import ZoneInfo
                        today_et = datetime.now(timezone.utc).astimezone(
                            ZoneInfo('America/New_York')).date().isoformat()
                    except Exception:
                        _n = datetime.now(timezone.utc)
                        today_et = (_n - timedelta(
                            hours=_et_offset_hours(_n))).date().isoformat()
                    if bar_date < today_et:
                        logger.info(
                            f"ORB: {sym} stale-snapshot reject — daily bar "
                            f"dated {bar_date} (vendor corpse, no data today)")
                        continue
                # Apply BT criteria
                if not (self.universe_min_price <= open_price <= self.universe_max_price):
                    continue
                if prev_close <= 0:
                    continue
                gap_pct = (open_price - prev_close) / prev_close * 100.0
                if gap_pct < self.universe_min_gap_pct:
                    continue
                if prev_volume < self.universe_min_prev_volume:
                    continue
                keep.append(sym)
            except Exception as e:
                logger.debug(f"ORB: snapshot parse failed for {sym}: {e}")
                continue
        logger.info(
            f"ORB: snapshot universe — {len(keep)}/{len(snapshots)} symbols pass "
            f"(gap>={self.universe_min_gap_pct}%, vol>={self.universe_min_prev_volume:,}, "
            f"${self.universe_min_price}-${self.universe_max_price})"
        )
        return keep

    def _subscribe_bars(self, symbol: str) -> None:
        """Subscribe a symbol to StopMonitor's 1-min bar stream.

        Required so _on_bar_close fires for this symbol and range_data can be
        computed when the first 5 bars close. MACD wave uses the same API.

        If we're already past 9:35 ET when subscribing, we've missed the live
        bar stream for the range window. Attempt to backfill historical 1-min
        bars 9:30 → now so `_ingest_bars` can synthesize range_data for this
        late-added candidate.
        """
        if self.stop_monitor is None:
            return
        try:
            if hasattr(self.stop_monitor, 'subscribe_bars'):
                self.stop_monitor.subscribe_bars(symbol)
                cand = self.candidates.get(symbol)
                if cand:
                    cand.bars_subscribed = True
        except Exception as e:
            logger.warning(f"ORB: subscribe_bars({symbol}) failed: {e}")

        # Backfill historical bars if we're past 9:35 (range already closed).
        self._backfill_range_if_needed(symbol)

    def _backfill_range_if_needed(self, symbol: str) -> None:
        """Fetch historical 1-min bars to synthesize range_data for a symbol
        added AFTER the 9:30-9:35 ET range window closed.

        No-op if: already have range_data, or we're before 9:35 ET (live stream
        will handle it), or past 11:00 ET (late enough that ORB wouldn't trade
        this day — stale universe addition).
        """
        cand = self.candidates.get(symbol)
        if cand is None or cand.range_data is not None:
            return

        now_utc = datetime.now(timezone.utc)
        try:
            from zoneinfo import ZoneInfo
            et_now = now_utc.astimezone(ZoneInfo('America/New_York'))
        except Exception:
            et_now = now_utc - timedelta(hours=_et_offset_hours(now_utc))

        # Only backfill if we're past 9:35 ET AND before 11:00 ET (ORB time-stop is 10:35;
        # give 25 min buffer for post-close reconciliation).
        if et_now.time() < dtime(9, 35) or et_now.time() > dtime(11, 0):
            return

        # Fetch historical bars from Alpaca
        try:
            # Compute UTC start = 9:30 ET today. Convert via zoneinfo or offset.
            try:
                from zoneinfo import ZoneInfo
                et_930 = datetime.combine(et_now.date(), dtime(9, 30), tzinfo=ZoneInfo('America/New_York'))
                start_utc = et_930.astimezone(timezone.utc)
            except Exception:
                offset = _et_offset_hours(now_utc)
                start_utc = datetime(et_now.year, et_now.month, et_now.day,
                                     9 + offset, 30, tzinfo=timezone.utc)
            end_utc = now_utc
            if not hasattr(self.alpaca, 'get_historical_1min_bars') \
                    and not hasattr(self.alpaca, 'get_bars'):
                return
            # Prefer purpose-built helper (bull flag uses get_historical_1min_bars)
            bars = None
            if hasattr(self.alpaca, 'get_historical_1min_bars'):
                try:
                    bars = self.alpaca.get_historical_1min_bars(symbol, start_utc, end_utc)
                except Exception:
                    bars = None
            if bars is None and hasattr(self.alpaca, 'get_bars'):
                try:
                    bars = self.alpaca.get_bars(symbol, '1Min', start_utc, end_utc)
                except Exception:
                    bars = None
            if bars is None:
                return
            # Normalize to DataFrame
            if isinstance(bars, pd.DataFrame):
                bars_df = bars
            else:
                try:
                    bars_df = pd.DataFrame([
                        {
                            'timestamp': getattr(b, 'timestamp', None) or b.get('timestamp'),
                            'open': getattr(b, 'open', None) or b.get('open'),
                            'high': getattr(b, 'high', None) or b.get('high'),
                            'low': getattr(b, 'low', None) or b.get('low'),
                            'close': getattr(b, 'close', None) or b.get('close'),
                            'volume': getattr(b, 'volume', None) or b.get('volume', 0),
                        }
                        for b in bars
                    ])
                except Exception:
                    return
            if bars_df.empty or 'timestamp' not in bars_df.columns:
                return
            bars_df['timestamp'] = pd.to_datetime(bars_df['timestamp'], utc=True)
            bars_df = bars_df.sort_values('timestamp').reset_index(drop=True)
            logger.info(f"ORB: {symbol} backfilling {len(bars_df)} historical bars")
            self._ingest_bars(symbol, bars_df)
        except Exception as e:
            logger.warning(f"ORB: backfill for {symbol} failed: {e}")

    def _ensure_ranges_post_open(self) -> Set[str]:
        """One-shot post-9:35 ET sweep — backfill range_data for any candidate
        that didn't get it via the WebSocket bar stream.

        Context: universe_build runs at ~9:31 ET and subscribes symbols to the
        WS bar stream. But Alpaca's WS only delivers bars from subscribe-time
        forward — the 9:30 bar (closed at 9:31:00) is already gone. Late-arriving
        or low-volume bars can also delay the 5-bar range from completing. Result
        on 2026-04-21: 7 of 11 candidates had INCOMPLETE ranges 13 minutes after
        range close, even though all 5 bars existed in Alpaca's historical feed.

        Fix: on the first check_entries call after 9:35 ET, batch-fetch historical
        1-min bars for every candidate still missing range_data. One API call via
        get_1min_bars_multi. Then ingest each — which computes range_data.

        Idempotent via self._post_open_range_sweep_done; resets daily.

        Returns:
            Set of symbols whose range_data was just filled by this sweep.
            Callers (check_entries) should widen their eligible set with these
            so a subset-scoped check_entries(symbols=subset) doesn't skip them
            — was the 2026-04-22 bug: WS drain fired check_entries with the
            Q5 subset {CRMX,BKKT,RGTX,BITU}, sweep filled ETHT/RDTL/NBIG/VNCE
            (all Q4 — should rank FIRST per orb.yaml), but those never scored
            because cand_pool=symbols didn't include them.
        """
        if self._post_open_range_sweep_done:
            return set()
        now_utc = datetime.now(timezone.utc)
        try:
            from zoneinfo import ZoneInfo
            et_now = now_utc.astimezone(ZoneInfo('America/New_York'))
        except Exception:
            et_now = now_utc - timedelta(hours=_et_offset_hours(now_utc))
        # Gate: only run between 9:35 and 11:00 ET (same window as
        # _backfill_range_if_needed — past 11:00, ORB shouldn't trade anyway).
        if et_now.time() < dtime(9, 35) or et_now.time() > dtime(11, 0):
            return set()

        missing = [
            sym for sym, cand in self.candidates.items()
            if cand.range_data is None
        ]
        if not missing:
            self._post_open_range_sweep_done = True
            return set()

        logger.info(
            f"ORB: post-open range sweep — backfilling {len(missing)} candidates "
            f"missing range_data at {et_now.time().strftime('%H:%M:%S')} ET: "
            f"{','.join(missing[:10])}{'...' if len(missing) > 10 else ''}"
        )

        # Batch fetch: 1 API call for all missing symbols. Lookback covers
        # 9:30 ET → now with headroom. Generous (60 min) is fine — we only
        # look at the 9:30-9:34 slice for range computation; extra bars are
        # just stored in the rolling window.
        minutes_since_open = 60
        bars_by_sym: Dict[str, pd.DataFrame] = {}
        try:
            if hasattr(self.alpaca, 'get_1min_bars_multi'):
                bars_by_sym = self.alpaca.get_1min_bars_multi(
                    missing, lookback_minutes=minutes_since_open
                ) or {}
            else:
                for sym in missing:
                    try:
                        b = self.alpaca.get_1min_bars(sym, lookback_minutes=minutes_since_open)
                        if b is not None and not b.empty:
                            bars_by_sym[sym] = b
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"ORB: post-open sweep bar fetch failed: {e}")

        filled_syms: Set[str] = set()

        def _ingest_batch(sym_list, bars_map):
            for sym in sym_list:
                bars = bars_map.get(sym)
                if bars is None or bars.empty:
                    continue
                try:
                    b = bars.copy()
                    b['timestamp'] = pd.to_datetime(b['timestamp'], utc=True)
                    b = b.sort_values('timestamp').reset_index(drop=True)
                    self._ingest_bars(sym, b)
                    if self.candidates.get(sym) and self.candidates[sym].range_data is not None:
                        filled_syms.add(sym)
                except Exception as e:
                    logger.warning(f"ORB: post-open sweep ingest({sym}) failed: {e}")

        _ingest_batch(missing, bars_by_sym)

        # 2026-07-03 selection-race fix (CRCD/AVEX/FABC/RGNX incidents): the
        # 9:34 bar consolidates at the vendor with 2-10s lag, so the first
        # sweep routinely leaves ~20% of candidates rangeless (7/2: 21/27
        # filled). check_entries then ranked the READY subset and burned all
        # max_concurrent daily slots within 3s — late-consolidating names
        # (BT-selected winners: CRCD +$15.8K on 6/30) were locked out for the
        # day. Retry the unfilled remainder ONCE after a short delay so the
        # first ranking sees the full field. Bounded: adds <= retry_delay+
        # fetch time (~5s) to the opening tick, only when stragglers exist.
        still_missing = [s for s in missing if s not in filled_syms]
        if still_missing and self.sweep_retry_delay_s > 0 \
                and not self._sweep_retry_used_today:
            self._sweep_retry_used_today = True
            logger.info(
                f"ORB: post-open sweep — {len(still_missing)} candidates still "
                f"rangeless ({','.join(still_missing[:8])}"
                f"{'...' if len(still_missing) > 8 else ''}); retrying once in "
                f"{self.sweep_retry_delay_s:.0f}s (bar-consolidation lag)"
            )
            time.sleep(self.sweep_retry_delay_s)
            retry_bars: Dict[str, pd.DataFrame] = {}
            try:
                if hasattr(self.alpaca, 'get_1min_bars_multi'):
                    retry_bars = self.alpaca.get_1min_bars_multi(
                        still_missing, lookback_minutes=minutes_since_open
                    ) or {}
                else:
                    for sym in still_missing:
                        try:
                            b = self.alpaca.get_1min_bars(sym, lookback_minutes=minutes_since_open)
                            if b is not None and not b.empty:
                                retry_bars[sym] = b
                        except Exception:
                            continue
            except Exception as e:
                logger.warning(f"ORB: post-open sweep retry fetch failed: {e}")
            _ingest_batch(still_missing, retry_bars)

        logger.info(
            f"ORB: post-open range sweep — filled {len(filled_syms)}/{len(missing)} ranges"
        )
        # Mark done even if some failed — next tick's WS bars will handle
        # any stragglers through the normal flow (and the first-rank grace
        # gate in check_entries holds the daily cap until they arrive or
        # the grace window expires).
        self._post_open_range_sweep_done = True
        return filled_syms

    # =====================================================================
    # Bar stream integration
    # =====================================================================

    def register_on_stop_monitor(self) -> None:
        """Register bar-close handler on the shared StopMonitor.

        StopMonitor emits bar-close events for subscribed symbols via
        registered handlers. ORB's handler enqueues (symbol, bars_df) for
        main-thread drain (queue is thread-safe; no DB/Alpaca from WS thread).
        """
        if self._bar_handler_registered:
            return
        if self.stop_monitor is None:
            logger.warning("ORBEngine.register_on_stop_monitor: no stop_monitor — skipping")
            return
        if getattr(self.stop_monitor, 'polling_mode', False):
            logger.warning("ORBEngine: stop_monitor in polling_mode — bar events disabled")
            return
        try:
            self.stop_monitor.register_bar_handler('orb', self._on_bar_close)
            self._bar_handler_registered = True
            logger.info("ORBEngine: bar handler registered on shared StopMonitor")
        except Exception as e:
            logger.error(f"ORBEngine.register_on_stop_monitor failed: {e}")

    def _on_bar_close(self, symbol: str, bars_df: pd.DataFrame) -> None:
        """Bar-close handler (runs in StopMonitor's daemon thread).

        Enqueues event for main thread; does NO heavy work here (no DB, no Alpaca).
        """
        if not self.enabled:
            return
        if symbol not in self.universe:
            return
        try:
            self._bar_event_queue.put_nowait((symbol, bars_df))
        except queue.Full:
            if not self._bar_queue_full_logged:
                logger.error(
                    "ORBEngine: bar event queue FULL — main loop may be stuck. "
                    "Dropping events."
                )
                self._bar_queue_full_logged = True

    def drain_bar_events(self) -> Set[str]:
        """Drain queued bar events; update range data + candidate state.

        Returns:
            Set of symbols with fresh bar data (caller may use for targeted re-eval).
        """
        touched: Set[str] = set()
        while True:
            try:
                symbol, bars_df = self._bar_event_queue.get_nowait()
            except queue.Empty:
                break
            self._ingest_bars(symbol, bars_df)
            touched.add(symbol)
        return touched

    def _ingest_bars(self, symbol: str, bars_df: pd.DataFrame) -> None:
        """Process fresh bars for a symbol: maintain rolling window, detect range close."""
        if bars_df is None or len(bars_df) == 0:
            return
        # Keep rolling window (latest ~60 bars is enough for ORB)
        self._bar_windows[symbol] = bars_df.tail(60).to_dict('records')

        # Capture the MARKET breakout bar (first bar high>range_high) for any
        # position on this symbol — pending OR filled — so touchgo Rule M/D
        # evaluate the BT-parity bar regardless of fill latency. Runs every bar
        # event from the pending phase onward, so the breakout bar is captured
        # while still in the streamed window (robust to late fills).
        try:
            self._ensure_breakout_bar_ts(symbol, bars_df)
        except Exception as e:
            logger.warning(f"ORB: _ensure_breakout_bar_ts({symbol}) failed: {e}")

        # Evaluate touchgo filter on any open position for this symbol.
        # Runs before range-complete check so it fires even for already-filled
        # positions (range was completed at fill time).
        try:
            self._evaluate_touchgo(symbol, bars_df)
        except Exception as e:
            logger.warning(f"ORB: _evaluate_touchgo({symbol}) failed: {e}")

        cand = self.candidates.get(symbol)
        if cand is None or cand.range_data is not None:
            return  # already processed or not in universe

        # Check if the first `range_minutes` bars (from session open) are complete
        session_open_ts = _first_session_open_ts_utc(bars_df)
        if session_open_ts is None:
            return
        range_end_ts = session_open_ts + timedelta(minutes=self.range_minutes)
        range_mask = (bars_df['timestamp'] >= session_open_ts) & (bars_df['timestamp'] < range_end_ts)
        range_bars = bars_df.loc[range_mask]
        if len(range_bars) < self.range_minutes:
            return  # range incomplete

        rh = float(range_bars['high'].max())
        rl = float(range_bars['low'].min())
        if rh <= 0 or rl <= 0 or rh <= rl:
            return  # degenerate
        total_vol = int(range_bars['volume'].sum())
        avg_bar_range_pct = float(
            ((range_bars['high'] - range_bars['low']) / range_bars['close']).mean() * 100.0
        ) if len(range_bars) > 0 else 0.0
        range_close = float(range_bars.iloc[-1]['close'])
        range_open = float(range_bars.iloc[0]['open'])  # 9:30 bar open — BT-parity
        green_bars = int((range_bars['close'] > range_bars['open']).sum())
        last_green = bool(range_bars.iloc[-1]['close'] > range_bars.iloc[-1]['open'])

        cand.range_data = RangeData(
            symbol=symbol,
            range_high=rh, range_low=rl,
            range_volume=total_vol,
            range_avg_bar_range_pct=avg_bar_range_pct,
            range_close=range_close,
            range_start_ts=session_open_ts,
            bars_green=green_bars,
            last_bar_green=last_green,
            range_open=range_open,
        )
        logger.info(
            f"ORB: {symbol} range complete — "
            f"H=${rh:.2f} L=${rl:.2f} range%={(rh-rl)/rl*100:.2f}% vol={total_vol:,}"
        )

    # =====================================================================
    # Touchgo filter (Rule M + Rule D)
    # =====================================================================

    def _market_breakout_bar_ts(self, bars_df: pd.DataFrame, range_high: float):
        """Market breakout bar timestamp from a bars DataFrame.

        The first 1-min bar (at/after the 9:35 ET range end) whose high exceeds
        range_high — identical definition to BT (study_orb_pipeline_static_lock).
        Delegates to the shared trading.orb_touchgo_filter.find_breakout_bar_ts.
        Returns a tz-aware pd.Timestamp or None.

        Requires the 9:30 session-open bar to be present so the search can be
        anchored at the range end. If it is absent — e.g. a position rehydrated
        after a mid-session restart, where the fresh StopMonitor window starts
        mid-day — we DECLINE (return None) rather than do an unanchored search
        that could match a random later bar trading above range_high. Declining
        leaves breakout_bar_ts None, so _evaluate_touchgo stays inert for that
        position (touchgo is a first-1-2-min filter; a position recovered hours
        in is past it anyway). Mirrors the same 9:30 anchor that range detection
        in _ingest_bars already depends on.
        """
        session_open_ts = _first_session_open_ts_utc(bars_df)
        if session_open_ts is None:
            return None
        range_end_ts = session_open_ts + timedelta(minutes=self.range_minutes)
        return find_breakout_bar_ts(bars_df, range_high, range_end_ts)

    def _ensure_breakout_bar_ts(self, symbol: str, bars_df: pd.DataFrame) -> None:
        """Set pos.breakout_bar_ts to the MARKET breakout bar (BT-parity).

        Runs on every bar event for pending or filled positions. No-op in
        legacy breakout_bar_source='fill' mode, once breakout_bar_ts is set, or
        when there's no position.

        2026-07-03 fix (KOLD/PLTU/TSDD false tag exits): the search anchor is
        the POSITION'S OWN range_end_ts (stored at entry submit), passed
        straight to the shared find_breakout_bar_ts. The old path re-derived a
        "session open" from the current bars window — but streamed windows
        usually lack the 9:30 bar (Alpaca WS delivers from subscribe-time
        forward), so the anchor heuristic fell through to the first :30 bar it
        saw (14:30Z = 10:30 ET during EDT) and keyed Rule M/D to a bar an hour
        after the real breakout (tag_bb firings at 14:36:2xZ on KOLD 6/29,
        PLTU 7/1, TSDD 6/23). The window-anchor path remains only as a
        fallback for positions without range_end_ts (rehydrated after a
        restart), where declining is the correct conservative outcome.
        """
        if self.touchgo_cfg.breakout_bar_source != 'market':
            return
        pos = self.open_positions.get(symbol)
        if pos is None or pos.breakout_bar_ts is not None:
            return
        if pos.range_end_ts is not None:
            bb_ts = find_breakout_bar_ts(bars_df, pos.range_high, pos.range_end_ts)
            source = 'pos_range'
        else:
            bb_ts = self._market_breakout_bar_ts(bars_df, pos.range_high)
            source = 'window_anchor'
        if bb_ts is not None:
            pos.breakout_bar_ts = bb_ts
            # Confirmation log — one line per position, greppable:
            # journalctl -u onemil-trader | grep "breakout bar keyed"
            logger.info(
                f"ORB: {symbol} breakout bar keyed {bb_ts} "
                f"(range_end={pos.range_end_ts}, source={source})"
            )

    def _evaluate_touchgo(self, symbol: str, bars_df: pd.DataFrame) -> None:
        """Evaluate Rule M and Rule D for any open position on this symbol.

        Called from _ingest_bars on every bar event. The bar event's last row
        is the just-closed 1-min bar.

        Rule M: at the close of the breakout bar (the bar containing the fill
            timestamp), if bb_close_pos < threshold, force-exit.
        Rule D: at the close of the next bar (breakout_bar + 1min), if bar's
            low reverted ≥ revert_R below entry, force-exit at entry + exit_R*R.

        Both rules dedup via rule_*_evaluated flags so the same bar can't
        trigger twice.
        """
        pos = self.open_positions.get(symbol)
        if pos is None or pos.order_id != '':
            return  # no position or not yet filled
        if pos.breakout_bar_ts is None:
            return
        if pos.rule_m_evaluated and pos.rule_d_evaluated:
            return  # both already evaluated
        if bars_df is None or len(bars_df) == 0:
            return

        # The latest bar is the just-closed one.
        last_bar = bars_df.iloc[-1]
        last_ts = last_bar['timestamp']
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize('UTC')

        bb_ts = pos.breakout_bar_ts
        if bb_ts.tzinfo is None:
            bb_ts = bb_ts.replace(tzinfo=timezone.utc)
        b1_ts = bb_ts + timedelta(minutes=1)

        # Late-fill guard (market mode only): if we filled long after the
        # breakout bar, the entry is no longer a clean opening-range breakout.
        # Rule M/D are a first-1-2-minute failed-breakout detector — do NOT fire
        # a retroactive tag exit on a bar that closed long before we held the
        # position. BT never sees this (instant fill at the breakout bar), so
        # this guards live's pathological late stop-limit fills only.
        if (self.touchgo_cfg.breakout_bar_source == 'market'
                and pos.entry_time is not None):
            et = pos.entry_time
            if getattr(et, 'tzinfo', None) is None:
                et = et.replace(tzinfo=timezone.utc)
            age_min = (et - bb_ts).total_seconds() / 60.0
            if age_min > self.touchgo_cfg.max_breakout_age_min:
                if not (pos.rule_m_evaluated and pos.rule_d_evaluated):
                    logger.info(
                        f"ORB: {symbol} touchgo SKIP — fill lagged breakout bar "
                        f"by {age_min:.0f}min (> {self.touchgo_cfg.max_breakout_age_min:.0f}min "
                        f"cap); stale entry, no retroactive tag exit"
                    )
                pos.rule_m_evaluated = True
                pos.rule_d_evaluated = True
                return
            # Negative-age tripwire (2026-07-03): a breakout bar can never
            # meaningfully POSTDATE the fill — our stop-limit fill IS in (or
            # within a few minutes of) the bar that broke the range. A large
            # negative age means the keying is insane (the KOLD/PLTU/TSDD
            # incidents keyed bars 53-84min after the fill via the false
            # 14:30Z anchor). Decline touchgo loudly rather than fire a
            # first-2-minute rule on a random mid-session bar. Tolerance
            # -5min allows the rare touch-without-exceed fill where the true
            # breakout bar legitimately closes a few minutes after entry.
            if age_min < -5.0:
                logger.warning(
                    f"ORB: {symbol} touchgo DECLINED — breakout_bar_ts "
                    f"{bb_ts} postdates fill {et} by {-age_min:.0f}min; "
                    f"keying is invalid (see 2026-07-03 false-anchor "
                    f"incident). No tag exit for this position."
                )
                pos.rule_m_evaluated = True
                pos.rule_d_evaluated = True
                return

        range_size = max(pos.range_high - pos.range_low, 0.0)

        # Rule M: evaluated as soon as the breakout bar is in our window.
        # We do NOT gate on last_ts < b1_ts — if bars arrive late, batched,
        # or the engine restarts mid-trade after both bars have closed, the
        # rolling bars_df can have last_ts >= b1_ts on the very first event
        # we see. The bb_row lookup below is robust to this (it searches
        # bars_df for the breakout bar by timestamp range, regardless of
        # last_ts). The rule_m_evaluated flag dedups against re-firing.
        if not pos.rule_m_evaluated and last_ts >= bb_ts:
            try:
                bb_row = bars_df[bars_df['timestamp'].between(bb_ts, b1_ts, inclusive='left')]
                if len(bb_row) > 0:
                    bb_bar = bb_row.iloc[-1]
                    bb_o = float(bb_bar['open']); bb_high = float(bb_bar['high'])
                    bb_low = float(bb_bar['low']); bb_close = float(bb_bar['close'])
                    fire, exit_p = evaluate_rule_m(
                        bb_o, bb_high, bb_low, bb_close, self.touchgo_cfg,
                    )
                    pos.rule_m_evaluated = True
                    close_pos = ((bb_close - bb_low) / (bb_high - bb_low)
                                 if bb_high > bb_low else 0.0)
                    # Audit EVERY Rule-M eval (fire + no-fire) for BT↔live
                    # divergence diagnosis — see trading/touchgo_audit.py.
                    self._audit_touchgo(
                        pos, 'M', fire, exit_p, last_ts, bb_ts, range_size,
                        bb_ohlc=(bb_o, bb_high, bb_low, bb_close),
                        bb_close_pos=close_pos,
                    )
                    if fire and exit_p is not None:
                        self._fire_touchgo_exit(
                            pos, reason=ExitReason.TAG_BB.value, exit_price=exit_p,
                            detail=f"bb_close_pos={close_pos:.3f} (threshold {self.touchgo_cfg.rule_m_threshold})",
                        )
                        return  # position is being exited; don't evaluate D
            except Exception as e:
                logger.warning(f"ORB: Rule M eval failed for {symbol}: {e}")

        # Rule D: triggered when the bar event for b1_ts arrives.
        if not pos.rule_d_evaluated and last_ts >= b1_ts:
            try:
                b1_end = b1_ts + timedelta(minutes=1)
                b1_row = bars_df[bars_df['timestamp'].between(b1_ts, b1_end, inclusive='left')]
                if len(b1_row) > 0:
                    b1_bar = b1_row.iloc[-1]
                    b1_low = float(b1_bar['low'])
                    fire, exit_p = evaluate_rule_d(
                        pos.entry_price, b1_low, range_size, self.touchgo_cfg,
                    )
                    pos.rule_d_evaluated = True
                    revert_R = ((pos.entry_price - b1_low) / range_size
                                if range_size > 0 else 0.0)
                    self._audit_touchgo(
                        pos, 'D', fire, exit_p, last_ts, bb_ts, range_size,
                        b1_ts=b1_ts, b1_low=b1_low, b1_revert_R=revert_R,
                    )
                    if fire and exit_p is not None:
                        self._fire_touchgo_exit(
                            pos, reason=ExitReason.TAG_B1.value, exit_price=exit_p,
                            detail=f"b1_revert={revert_R:.2f}R (threshold ≥{self.touchgo_cfg.rule_d_revert_R}R)",
                        )
                # Even if no bar found, mark evaluated so we don't keep checking.
                # We'll catch it on the NEXT event if we mis-aligned.
                elif last_ts >= b1_end:
                    pos.rule_d_evaluated = True
            except Exception as e:
                logger.warning(f"ORB: Rule D eval failed for {symbol}: {e}")

    def _audit_touchgo(self, pos: 'OpenPosition', rule: str, fired: bool,
                       exit_p, last_ts, bb_ts, range_size: float,
                       bb_ohlc=None, bb_close_pos=None,
                       b1_ts=None, b1_low=None, b1_revert_R=None) -> None:
        """Emit a diagnostic record for one Rule M / Rule D evaluation.

        Captures the exact bar live keyed to + the computed decision so a
        divergence vs the consolidated market bars (the EIDO/OSCR/TSDD
        false-positive class) can be diagnosed post-hoc by
        scripts/audit_touchgo_live_vs_consolidated.py. Best-effort: never
        raises into the trade path.
        """
        try:
            def _iso(t):
                if t is None:
                    return None
                return t.isoformat() if hasattr(t, 'isoformat') else str(t)

            entry_time = getattr(pos, 'entry_time', None)
            bb_age_min = None
            if entry_time is not None and bb_ts is not None:
                try:
                    et = entry_time
                    if getattr(et, 'tzinfo', None) is None:
                        et = et.replace(tzinfo=timezone.utc)
                    _bb = bb_ts
                    if getattr(_bb, 'tzinfo', None) is None:
                        _bb = _bb.replace(tzinfo=timezone.utc)
                    bb_age_min = (et - _bb).total_seconds() / 60.0
                except Exception:
                    bb_age_min = None

            rec = {
                'ts_utc': datetime.now(timezone.utc).isoformat(),
                'trade_date': (_iso(bb_ts) or '')[:10],
                'symbol': pos.symbol,
                'trade_id': getattr(pos, 'trade_id', None),
                'rule': rule,
                'fired': bool(fired),
                'exit_price': float(exit_p) if exit_p is not None else None,
                'breakout_bar_ts': _iso(bb_ts),
                'breakout_bar_source': getattr(self.touchgo_cfg, 'breakout_bar_source', None),
                'bb_age_min': round(bb_age_min, 2) if bb_age_min is not None else None,
                'max_breakout_age_min': getattr(self.touchgo_cfg, 'max_breakout_age_min', None),
                'range_high': float(pos.range_high),
                'range_low': float(pos.range_low),
                'range_size': float(range_size),
                'entry_price': float(pos.entry_price),
                'entry_time': _iso(entry_time),
                'last_bar_ts': _iso(last_ts),
            }
            if rule == 'M':
                o, h, l, c = (bb_ohlc if bb_ohlc else (None, None, None, None))
                rec.update({
                    'bb_open': o, 'bb_high': h, 'bb_low': l, 'bb_close': c,
                    'bb_close_pos': (round(bb_close_pos, 4)
                                     if bb_close_pos is not None else None),
                    'rule_m_threshold': getattr(self.touchgo_cfg, 'rule_m_threshold', None),
                })
            else:  # 'D'
                rec.update({
                    'b1_ts': _iso(b1_ts),
                    'b1_low': b1_low,
                    'b1_revert_R': (round(b1_revert_R, 4)
                                    if b1_revert_R is not None else None),
                    'rule_d_revert_R': getattr(self.touchgo_cfg, 'rule_d_revert_R', None),
                })
            _tg_audit.record(rec)
        except Exception as e:  # diagnostic must never break the trade path
            logger.debug(f"ORB: touchgo audit record failed for {pos.symbol}: {e}")

    def _fire_touchgo_exit(self, pos: 'OpenPosition', reason: str,
                            exit_price: float, detail: str) -> None:
        """Route a touchgo exit through StopMonitor.force_exit + Telegram alert.

        reason: 'tag_bb' or 'tag_b1'.
        exit_price: target limit price from the shared helper.
        detail: human-readable diagnostic string (e.g., 'bb_close_pos=0.32').
        """
        # Estimate $-impact vs holding to full -1R stop, for Telegram message.
        full_stop_pnl = (pos.stop_price - pos.entry_price) * pos.shares
        est_exit_pnl = (exit_price - pos.entry_price) * pos.shares
        saved_vs_stop = est_exit_pnl - full_stop_pnl  # +ve = filter helped

        logger.info(
            f"ORB: {pos.symbol} TOUCHGO {reason.upper()} fired — "
            f"{detail}, entry=${pos.entry_price:.2f} exit≈${exit_price:.2f} "
            f"est_pnl=${est_exit_pnl:+,.2f} saved_vs_stop=${saved_vs_stop:+,.2f}"
        )

        # Route exit through StopMonitor's force_exit (new public wrapper).
        if self.stop_monitor is not None:
            try:
                self.stop_monitor.force_exit(
                    symbol=pos.symbol,
                    reason=reason,
                    limit_price=exit_price,
                )
            except Exception as e:
                logger.error(
                    f"ORB: {pos.symbol} touchgo exit force_exit failed: {e}"
                )
                return

        # Telegram alert (fire-and-forget; failure is non-fatal).
        # Use self._notify which handles both sync TelegramNotifier and the
        # async TelegramNotifier.send_message coroutine correctly. Calling
        # send_message directly (without await) was the 2026-05-19 bug —
        # coroutine was created and discarded, alert never delivered.
        if self.notifier is not None:
            try:
                rule_label = 'TAG_BB EXIT' if reason == ExitReason.TAG_BB.value else 'TAG_B1 EXIT'
                rule_desc = (
                    'Breakout bar closed weak'
                    if reason == ExitReason.TAG_BB.value
                    else 'Bar-1 reverted deep'
                )
                # 2026-07-09 copy fix (PLTZ/AAOX): the old message headlined
                # "Est P&L: $-20" and the fill then printed +$87 — one exit,
                # two contradicting numbers (the estimate is next-bar-open
                # based and systematically pessimistic: the sell limit fills
                # at market-or-better). Frame this message as a CUT IN
                # PROGRESS; the real P&L arrives with the fill message.
                msg = (
                    f"[ORB] {rule_label}: {pos.symbol} — cutting failed "
                    f"breakout\n"
                    f"{rule_desc} ({detail})\n"
                    f"Entry ${pos.entry_price:.2f} → exit order placed "
                    f"(limit ≈${exit_price:.2f}; fills at market-or-better)\n"
                    f"Avoids riding to stop ${pos.stop_price:.2f} "
                    f"(≈${saved_vs_stop:+,.0f} protected)\n"
                    f"Final P&L follows on fill confirmation."
                )
                self._notify(msg)
            except Exception as e:
                logger.warning(
                    f"ORB: {pos.symbol} touchgo Telegram send failed: {e}"
                )

    # =====================================================================
    # Feature extraction
    # =====================================================================

    def _get_feature_context(self, symbol: str) -> Dict[str, dict]:
        """Fetch prev_day_bar + 20d daily stats from DB for one candidate.

        Cached per-symbol-per-day via self._feature_context_cache to avoid N
        DB queries per tick. Cache cleared by reset_daily.

        Returns empty dict on error — composite_score will then reject the
        candidate via feature_missing path.
        """
        if not hasattr(self, '_feature_context_cache'):
            self._feature_context_cache = {}
        if symbol in self._feature_context_cache:
            return self._feature_context_cache[symbol]

        ctx: Dict[str, dict] = {}
        try:
            # Fetch ~25 daily bars (need 20d + T-1 + buffer).
            # Preferred: DB cache (populated by nightly universe build — fast + free).
            # Fallback: alpaca.get_daily_bars_range (NB: plain get_daily_bars returns
            # only a summary dict per symbol, NOT a bar list — don't use here).
            daily_bars = None
            today = datetime.now(timezone.utc).date()
            start_d = today - timedelta(days=40)
            end_d = today - timedelta(days=1)
            if hasattr(self.db, 'get_daily_bars_cached'):
                try:
                    bulk = self.db.get_daily_bars_cached(
                        [symbol], str(start_d), str(end_d))
                    daily_bars = bulk.get(symbol) if isinstance(bulk, dict) else None
                except Exception as e:
                    logger.debug(f"ORB: db.get_daily_bars_cached({symbol}) failed: {e}")

            # Freshness guard. The old code only refetched when the symbol was
            # ENTIRELY absent from cache, so a STALE most-recent bar (nightly
            # broad-universe refresh dropped this symbol) slipped through and
            # silently fed a days-old prev_close into the gap/feature math.
            # Refetch on demand when the newest cached bar predates the last
            # trading day. (See ASTN/PLTG 2026-05-29 prod incident.)
            stale = False
            if daily_bars:
                newest = _newest_bar_date(daily_bars)
                if newest is not None and (today - newest).days > _PREV_BAR_STALENESS_MAX_DAYS:
                    logger.warning(
                        f"ORB: {symbol} cached prev-day bar is stale "
                        f"(newest={newest}, today={today}, "
                        f">{_PREV_BAR_STALENESS_MAX_DAYS}d) — refetching daily "
                        f"bars from Alpaca to avoid a stale-gap mispick"
                    )
                    stale = True

            if (not daily_bars or stale) and hasattr(self.alpaca, 'get_daily_bars_range'):
                try:
                    bulk = self.alpaca.get_daily_bars_range(
                        [symbol], start_d, end_d)
                    fresh = bulk.get(symbol) if isinstance(bulk, dict) else None
                    if fresh:
                        daily_bars = fresh
                    elif stale:
                        logger.warning(
                            f"ORB: {symbol} stale-bar Alpaca refetch returned no "
                            f"data — falling back to stale cache"
                        )
                except Exception as e:
                    # On the stale path, keep the stale cache as last resort but
                    # log loudly: a silent stale-gap pick is worse than a logged
                    # refetch failure.
                    (logger.warning if stale else logger.debug)(
                        f"ORB: alpaca.get_daily_bars_range({symbol}) failed: {e}"
                    )

            if not daily_bars:
                self._feature_context_cache[symbol] = ctx
                return ctx

            # Normalize: accept list-of-dicts or list-of-rows
            # Expected keys on each bar: open/high/low/close/volume, sorted by date asc
            bars_list = list(daily_bars) if not isinstance(daily_bars, list) else daily_bars
            if not bars_list:
                self._feature_context_cache[symbol] = ctx
                return ctx

            def _get(b, key, default=0.0):
                if isinstance(b, dict):
                    return float(b.get(key, default) or default)
                return float(getattr(b, key, default) or default)

            # T-1 bar (last bar before today)
            prev_bar = bars_list[-1]  # most recent trading day
            ctx['prev_day_bar'] = {
                'open':   _get(prev_bar, 'open'),
                'high':   _get(prev_bar, 'high'),
                'low':    _get(prev_bar, 'low'),
                'close':  _get(prev_bar, 'close'),
                'volume': _get(prev_bar, 'volume'),
            }
            # 20d high + avg volume over the most recent 20 bars (excluding today)
            window = bars_list[-20:] if len(bars_list) >= 20 else bars_list
            highs = [_get(b, 'high') for b in window]
            ctx['daily_stats_20d'] = {
                'high_20d': max(highs) if highs else 0.0,
                'volume_20d': (sum(_get(b, 'volume') for b in window) / len(window))
                    if window else 0.0,
            }
        except Exception as e:
            logger.warning(f"ORB: _get_feature_context({symbol}) failed: {e}")

        self._feature_context_cache[symbol] = ctx
        return ctx

    def _compute_features(self, cand: CandidateState,
                          prev_day_bar: Optional[dict] = None,
                          daily_stats_20d: Optional[dict] = None) -> Dict[str, float]:
        """Compute filter features from range data + (optional) prior-day context.

        Args:
            cand: CandidateState with range_data populated.
            prev_day_bar: dict with 'open', 'high', 'low', 'close', 'volume' — T-1 daily bar.
            daily_stats_20d: dict with 'high_20d', 'range_pct_20d', 'volume_20d' — 20d stats.

        Returns:
            dict of feature_name -> float. Missing features default to 0.0 but return
            None from composite_score (which rejects the candidate).
        """
        rd = cand.range_data
        if rd is None:
            return {}
        # BT parity: denominator for gap/size/high20 features is the 9:30 bar open
        # (NOT range_high). This matches study_orb_features.py:
        #   open_p = rb['open'].iloc[0]
        #   feat['range_size_pct'] = range_size / open_p * 100
        #   feat['gap_pct'] = (open_p - prev_close) / prev_close * 100
        #   feat['price_vs_20d_high_pct'] = (open_p - high_20d) / high_20d * 100
        ref_open = rd.range_open if rd.range_open > 0 else rd.range_high
        features: Dict[str, float] = {
            'range_size_pct': (rd.range_high - rd.range_low) / ref_open * 100.0,
            'range_total_volume': float(rd.range_volume),
            'range_avg_bar_range_pct': rd.range_avg_bar_range_pct,
            'range_close_position': (
                (rd.range_close - rd.range_low) / (rd.range_high - rd.range_low)
                if rd.range_high > rd.range_low else 0.5
            ),
        }
        # Prior-day features
        if prev_day_bar:
            pc = float(prev_day_bar.get('close', 0.0))
            ph = float(prev_day_bar.get('high', 0.0))
            pl = float(prev_day_bar.get('low', 0.0))
            if pc > 0:
                features['gap_pct'] = (ref_open - pc) / pc * 100.0
            if ph > pl > 0:
                features['prev_day_close_position'] = (pc - pl) / (ph - pl)
            pdr = compute_prev_day_range_pct(ph, pl, pc)
            if pdr is not None:
                features['prev_day_range_pct'] = pdr
        # 20d features
        if daily_stats_20d:
            h20 = float(daily_stats_20d.get('high_20d', 0.0))
            if h20 > 0:
                features['price_vs_20d_high_pct'] = (ref_open - h20) / h20 * 100.0
        return features

    # =====================================================================
    # Entry flow
    # =====================================================================

    def check_entries(self, symbols: Optional[Iterable[str]] = None,
                      feature_providers: Optional[Dict[str, dict]] = None) -> List[str]:
        """Evaluate candidates for entry.

        Args:
            symbols: Subset to check; if None, evaluates all candidates with
                range_data populated and not yet plan_submitted.
            feature_providers: optional override dict sym -> {prev_day_bar,
                daily_stats_20d}. When None, ORBEngine fetches from DB/alpaca
                on demand (with per-day caching).

        Returns:
            list of symbols for which entries were SUBMITTED.
        """
        if not self.enabled or self.shutdown_requested:
            return []
        if not self.z_params:
            logger.warning("ORBEngine.check_entries: no filter params loaded — skipping")
            return []

        # PM-mult prefetch (2026-07-06): warm the premarket dollar-volume
        # cache on early ticks (>=9:31 ET) so the 9:35 burst never blocks.
        try:
            self._maybe_prefetch_pm()
        except Exception as e:
            logger.warning(f"ORB: PM prefetch failed: {e} — lazy path remains")

        # Catalyst-veto anchor pre-warm — deliberately INDEPENDENT of the
        # PM prefetch above (the veto must not degrade if pm sizing or the
        # news gate is ever disabled).
        try:
            self._prewarm_anchors()
        except Exception as e:
            logger.warning(f"ORB: anchor pre-warm failed: {e} — "
                           f"submit loop resolves offline (no-anchor)")

        # Process pending fills BEFORE evaluating new entries. This transitions
        # pending_new trades to filled state + adds StopMonitor watches.
        self._process_pending_fills()

        # Cancel any unfilled stop-buy orders past the 10:35 ET time stop.
        self._cancel_stale_pending_orders()

        # Post-9:35 range sweep: any candidate without range_data gets
        # backfilled via REST. Runs ONCE per day, triggered on the first
        # check_entries call after 9:35 ET. Prevents the 2026-04-21 bug
        # where bars subscribed at 9:31 via WS missed the 9:30/9:31 bars
        # (WS only delivers from subscribe-time forward) and ranges stayed
        # incomplete for 10-15 minutes until a late scanner tick noticed.
        # Direct REST fetch with a single batch call is instant + reliable.
        #
        # 2026-04-22 bug fix: when called via scanner's WS drain path,
        # check_entries gets `symbols=<subset>` — only WS-touched symbols.
        # The sweep populates ranges for symbols that DIDN'T arrive via WS
        # (e.g. 9:30 bars already closed by subscribe-time). Without this
        # widening, those sweep-filled candidates are never scored and top-K
        # collapses to whatever the WS happened to deliver. On 4/22 that
        # meant Q4 winners (ETHT/RDTL/NBIG/VNCE) were skipped and four Q5s
        # filled all 4 slots — against orb.yaml ranking.order=[Q4,Q5,...].
        sweep_filled = self._ensure_ranges_post_open()
        if sweep_filled and symbols is not None:
            symbols = set(symbols) | sweep_filled

        if self._daily_loss_limit_hit():
            return []

        # Hard time-of-day cutoff on NEW entry submissions. BT picks top-K
        # once at 9:35 ET; live needs a short window for bar-stream latency
        # + backfill but must block late-afternoon entries (see 2026-04-20:
        # QBTZ submitted 12:30 ET, BATL 1:11 PM ET — well outside BT's
        # time_stop window of 10:35 ET). Defaults to 10:00 ET (25-min window).
        if self._past_last_entry_time():
            if not self._late_entry_cutoff_logged:
                logger.info(
                    f"ORB: past last_entry_submit_time "
                    f"{self.last_entry_hour_et:02d}:{self.last_entry_minute_et:02d} ET "
                    f"— no new entries for the rest of the day"
                )
                self._late_entry_cutoff_logged = True
            return []

        # BT parity: pick top-K ONCE at 9:35, then no new entries ever.
        #   (a) at most max_concurrent TOTAL entries per day
        #   (b) at most 1 entry per symbol per day — even after an exit
        # The in-memory counters (daily_n_placed, CandidateState.plan_submitted)
        # are lost on restart, so rely on DB state as the source of truth.
        # This was the bug on 2026-04-20: two mid-morning restarts wiped the
        # in-memory flags → BMNZ/SKYQ re-entered after stopping out, and
        # QBTZ/BATL slipped in as brand-new entries at 12:30 / 1:11 PM ET.
        symbols_entered_today = self._symbols_entered_today_db()
        # Daily cap counts ENTERED plus PDR-VETOED picks: BT's selection is
        # one-shot top-K, and a vetoed pick's slot is dead for the day
        # (2026-07-07 IREZ fix). The vetoed set is in-memory; after a
        # mid-day restart it self-heals — the same names re-rank, re-veto,
        # and re-consume their slots on the first post-restart burst.
        _slots_used = len(symbols_entered_today
                          | getattr(self, '_pdr_vetoed_today', set()))
        if _slots_used >= self.max_concurrent:
            return []  # daily cap exhausted (restart-safe via DB + re-veto)

        # 2026-07-03 first-rank grace gate (selection-race fix): before the
        # day's FIRST placement burst, if any pool candidate is still
        # rangeless (its 9:34 bar hadn't consolidated when the sweep ran),
        # DEFER ranking until the field completes or the grace window
        # expires. Without this, the 9:35:01 ranking burned all
        # max_concurrent slots on the ready subset and late-consolidating
        # BT-winners (CRCD +$15.8K 6/30, RGNX 6/22, FABC 6/11) were locked
        # out for the day. Only gates when NOTHING has been placed yet —
        # once the first burst happens the day's selection is committed.
        # NOTE (2026-07-04 review fix): the gate inspects the FULL candidate
        # pool internally — NOT the `symbols` subset. The day's first burst
        # usually arrives via the WS drain with symbols=<ready subset>, whose
        # members by construction all have ranges; scoping the rangeless
        # check to that subset made the gate blind on exactly the racing
        # path. When deferring, re-arm the post-open sweep so stragglers are
        # actively re-fetched (WS alone can't complete them — it lacks the
        # 9:30 anchor bar), otherwise the defer is pure entry delay.
        if not symbols_entered_today and self._should_defer_first_rank():
            self._post_open_range_sweep_done = False
            return []

        # 1. Build candidate set
        eligible: List[CandidateState] = []
        cand_pool = symbols if symbols is not None else self.candidates.keys()
        current_positions = len(self.open_positions)
        if current_positions >= self.max_concurrent:
            return []
        for sym in cand_pool:
            cand = self.candidates.get(sym)
            if cand is None or cand.plan_submitted:
                continue
            if cand.range_data is None:
                continue
            if sym in self.open_positions:
                continue  # already have ORB position (pending or filled)
            # DB-backed per-symbol dedup: if this symbol has ANY ORB trade row
            # today (any status — even closed), skip. Prevents re-entry after
            # exit and survives restart.
            if sym in symbols_entered_today:
                cand.rejected_reason = 'already_entered_today'
                continue
            # Cross-strategy FCFS
            if self.skip_if_any_strategy_has_symbol and self._symbol_has_any_open_trade(sym):
                logger.info(f"ORB: {sym} FCFS skip — already open in another strategy")
                cand.rejected_reason = 'fcfs_other_strategy'
                continue
            eligible.append(cand)

        if not eligible:
            return []

        # 2. Compute composite + quintile for each
        scored: List[CandidateState] = []
        for cand in eligible:
            # Feature-provider precedence: explicit override > DB fetch.
            providers = (feature_providers or {}).get(cand.symbol)
            if providers is None:
                providers = self._get_feature_context(cand.symbol)
            feats = self._compute_features(
                cand,
                prev_day_bar=providers.get('prev_day_bar'),
                daily_stats_20d=providers.get('daily_stats_20d'),
            )
            cand.features = feats
            # 2026-05-08 fix: phantom-gap guard at scoring time.
            # Universe filter at 9:30 used snapshot.open (line 392) which can
            # be a phantom-print or pre-open quote. We now have the real
            # bar1.open via range_open, so re-validate gap_pct here.
            # Without this guard, symbols whose real bar1.open == prev_close
            # (no gap) get scored — and because gap_pct sign=-1 in the
            # composite, low-gap entries SCORE HIGHER, ranking phantom-gap
            # symbols above legit ones. LYG 5/7 was a textbook case:
            # snapshot.open showed +5% gap, real bar1.open = prev_close,
            # yet LYG ranked Q5 in LIVE.
            real_gap_pct = feats.get('gap_pct', None)
            if real_gap_pct is not None and real_gap_pct < self.universe_min_gap_pct:
                logger.info(
                    f"ORB: {cand.symbol} phantom-gap reject — "
                    f"real gap {real_gap_pct:.2f}% < {self.universe_min_gap_pct}% "
                    f"(snapshot universe was satisfied via phantom snap.open)"
                )
                cand.rejected_reason = 'phantom_gap'
                continue
            score = composite_score(feats, self.z_params)
            if score is None:
                logger.debug(f"ORB: {cand.symbol} dropped — missing feature")
                cand.rejected_reason = 'feature_missing'
                continue
            if score < self.filter_threshold:
                # 2026-07-07: was SILENT — TDTH (+65% gap, composite-penalized)
                # vanished without a line and cost an hour of forensics. Every
                # candidate exit must leave a trace. DEBUG: re-scored per tick.
                if cand.rejected_reason != 'below_filter_threshold':
                    logger.info(
                        f"ORB: {cand.symbol} below filter threshold — "
                        f"comp={score:.4f} < {self.filter_threshold} "
                        f"(gap={feats.get('gap_pct', float('nan')):.1f}%)")
                cand.rejected_reason = 'below_filter_threshold'
                continue
            cand.composite = score
            cand.quintile = assign_quintile(score, self.quintile_cutoffs)
            scored.append(cand)
            # Telemetry for live-vs-BT parity debugging. 2026-04-22: live
            # composites ran ~0.09 below BT on same symbols/data, cause
            # unknown. Dump each feature so next session we can diff against
            # analysis_results/orb_features_*.csv to identify the divergent
            # input (likely a stale T-1 daily bar or 20d-high at 9:35 ET).
            # Abbrev: rtv=range_total_volume, rabr=range_avg_bar_range_pct,
            # rs=range_size_pct, p20h=price_vs_20d_high_pct,
            # pdcp=prev_day_close_position, rcp=range_close_position.
            # Names match column headers in orb_features_*.csv.
            logger.info(
                "ORB SCORED: %s comp=%.4f %s | "
                "gap=%.3f rtv=%.0f rabr=%.3f rs=%.3f p20h=%.3f pdcp=%.3f rcp=%.3f "
                "| prev_close=%.4f range_open=%.4f",
                cand.symbol, score, cand.quintile,
                feats.get('gap_pct', float('nan')),
                feats.get('range_total_volume', float('nan')),
                feats.get('range_avg_bar_range_pct', float('nan')),
                feats.get('range_size_pct', float('nan')),
                feats.get('price_vs_20d_high_pct', float('nan')),
                feats.get('prev_day_close_position', float('nan')),
                feats.get('range_close_position', float('nan')),
                float((providers.get('prev_day_bar') or {}).get('close') or 0.0),
                cand.range_data.range_open if cand.range_data else 0.0,
            )

        if not scored:
            return []

        # 3a. Q1 filter — drop bottom-quintile candidates if configured.
        # Q1 is net-negative OOS (see filter.skip_q1 comment in orb.yaml).
        # Single-pass partition keeps semantics identical to the prior two-pass
        # version while making the data flow obvious.
        if self.skip_q1:
            kept_scored = []
            q1_dropped = []
            for c in scored:
                if c.quintile == 'Q1':
                    c.rejected_reason = 'q1_filter'
                    q1_dropped.append(c)
                else:
                    kept_scored.append(c)
            if q1_dropped:
                logger.info(
                    "[ORB] Q1 filter dropped %d candidate(s): %s",
                    len(q1_dropped),
                    ', '.join(f"{c.symbol}(comp={c.composite:.3f})" for c in q1_dropped)
                )
            scored = kept_scored
            if not scored:
                return []

        # 3b. Rank — quintile order first, then composite DESC inside bucket
        q_rank = {q: i for i, q in enumerate(self.ranking_order)}
        scored.sort(key=lambda c: (q_rank.get(c.quintile, 99), -c.composite))
        ranked_symbols = [c.symbol for c in scored]

        # 4. Dedup by family + super-group.
        # 2026-07-07 review fix: the pick budget must subtract ENTERED and
        # VETOED slots too, not just open positions — otherwise a mid-window
        # restart (vetoed set is in-memory) or a multi-burst morning can
        # overshoot the daily slot invariant and replay the IREZ backfill
        # within a single burst.
        top_syms = dedup_candidates(
            ranked_symbols,
            max_keep=self.max_concurrent - len(
                symbols_entered_today | self._pdr_vetoed_today
                | set(self.open_positions)),
            by_family=self.dedup_by_family,
            by_super_group=self.dedup_by_super_group,
        )

        # 2026-07-03 selection audit: persist the full ranked field at every
        # placement-burst so any future BT↔live selection divergence
        # (CRCD/AVEX/FABC/RGNX class) is post-mortemable from disk instead of
        # racing journald rotation. One JSON line per burst; ~1-3/day.
        self._audit_selection(ranked_symbols, top_syms, scored)

        # 5. For each kept candidate, build plan + submit
        submitted: List[str] = []
        for sym in top_syms:
            cand = self.candidates[sym]
            # PDR veto — post-ranking, NO backfill: this pick already
            # consumed its slot/dedup place; the slot stays empty (refill
            # form is toxic — trading/orb_pdr_veto.py docstring).
            if self._pdr_veto_reject(cand):
                continue
            # Catalyst-required veto (2026-07-18): newsless AND alone
            # (no same-morning complex confirmation) — no catalyst, no
            # trade. Same no-refill slot semantics as PDR.
            if self._catalyst_veto_reject(cand):
                continue
            spread_bps = self._get_spread_bps(sym)
            plan = self.planner.build(
                symbol=sym,
                range_high=cand.range_data.range_high,
                range_low=cand.range_data.range_low,
                range_open=cand.range_data.range_open,  # BT-parity sizing denominator
                composite_score=cand.composite,
                quintile=cand.quintile,
                adaptive_mult=apply_adaptive_mult(cand.quintile, self.adaptive_mults),
                spread_bps=spread_bps,
                pm_mult=self._get_pm_mult(sym),
            )
            if isinstance(plan, PlannerReject):
                cand.rejected_reason = plan.reason
                self._handle_reject(sym, plan)
                continue
            # Check buying power live
            if not self._has_buying_power(plan.position_dollars):
                logger.warning(
                    f"ORB: {sym} insufficient buying power for ${plan.position_dollars:,.0f}"
                )
                cand.rejected_reason = 'insufficient_bp'
                if self.notify_on_capital_exhausted and self.notifier:
                    self._notify(f"{self.tg_prefix} {sym} skipped — insufficient buying power")
                continue
            order_id = self._submit_entry(plan)
            if order_id:
                cand.plan_submitted = True
                submitted.append(sym)
        return submitted

    def _get_pm_mult(self, symbol: str) -> float:
        """Premarket dollar-volume sizing mult for a pick (2026-07-04).

        Lazily batch-fetches premarket 1-min bars (4:00-9:29 ET) for ALL
        current candidates once per day on first use — one API call.
        Fail-open: fetch failure or no premarket prints → 1.0 + WARNING
        (never blocks or blindly boosts a trade).
        """
        if not self.pm_mult_enabled:
            return 1.0
        today = datetime.now(timezone.utc).date()
        if self._pm_fetch_done_day != today:
            self._fetch_pm_dollar_vols()
        if self.pm_news_gate and self._news_fetch_done_day != today:
            self._fetch_news_flags()
        pm = self._pm_dollar_vols.get(symbol)
        has_news_raw = self._get_has_news(symbol) if self.pm_news_gate else None
        # Deliberate class rule (2026-07-11): news counts only for
        # identified common stocks — wrappers have no company events and
        # their newsy days are the crowding cell (negative all 3 eras).
        cls = self._classify_symbol(symbol) if has_news_raw else CLASS_UNKNOWN
        has_news = effective_has_news(has_news_raw, cls) \
            if self.pm_news_gate else None
        mult = pm_size_multiplier(
            pm, self.pm_mult_high_cut, self.pm_mult_high,
            has_news=has_news, high_mult_news=self.pm_mult_high_news,
            news_gate=self.pm_news_gate)
        if mult != 1.0:
            logger.info(
                f"[ORB] PM MULT: {symbol} premarket ${pm:,.0f} > "
                f"${self.pm_mult_high_cut:,.0f}, news={has_news} "
                f"class={cls} — sizing x{mult}")
        elif has_news_raw and has_news is False:
            logger.info(
                f"[ORB] PM MULT: {symbol} newsy but class={cls} — news "
                f"boost structurally out of scope (deliberate wrapper rule)")
        elif pm is None:
            logger.warning(
                f"[ORB] PM MULT: {symbol} premarket volume unavailable — "
                f"fail-open x1.0")
        elif (self.pm_news_gate and has_news is None
              and pm > self.pm_mult_high_cut):
            logger.warning(
                f"[ORB] PM MULT: {symbol} above PM$ cut but news "
                f"UNAVAILABLE — fail-open, no news boost (x{mult})")
        return mult

    def _classify_symbol(self, symbol: str) -> str:
        """Asset class for the news-gate rule: 'stock' | 'wrapper' |
        'unknown'. Resolution order: cache → lev-family sets / offline
        class map (no I/O) → asset-name API fetch (8s bounded, only for
        symbols the 14K map doesn't know — new listings). Failure →
        'unknown' (not news-boost eligible — never boost blind)."""
        cached = self._asset_class.get(symbol)
        if cached is not None:
            return cached
        # lev-family fast path first (curated sets, zero I/O)
        if classify_asset(symbol, None) == 'wrapper':
            self._asset_class[symbol] = 'wrapper'
            return 'wrapper'
        if self._class_map is None:
            self._class_map = load_class_map()
        cls = self._class_map.get(symbol)
        if cls is None:
            name = None
            if hasattr(self.alpaca, 'get_asset_name'):
                name = self.alpaca.get_asset_name(symbol)
            cls = classify_asset(symbol, name)
            if cls == CLASS_UNKNOWN:
                logger.warning(
                    f"[ORB] CLASS: {symbol} unresolvable (no name) — "
                    f"'unknown', news boost ineligible")
        self._asset_class[symbol] = cls
        return cls

    def _get_has_news(self, symbol: str) -> Optional[bool]:
        """Tri-state news presence for a symbol: True/False, or None when
        the news fetch failed for it (fail-open — no boost)."""
        flags = self._news_flags.get(symbol)
        if flags is None:
            return None
        try:
            return int(flags.get('n_articles') or 0) > 0
        except (TypeError, ValueError, AttributeError) as e:
            logger.warning(
                f"[ORB] NEWS: malformed flags for {symbol} ({e}) — "
                f"treating as unknown (no boost)")
            return None

    def _fetch_news_flags(self) -> None:
        """Batch-fetch premarket news flags for all candidates (2026-07-10).

        Mirrors _fetch_pm_dollar_vols exactly: per-batch merge for late
        universe batches, empty-pool no-stamp, and failure poisoning with
        None (fail-open = no news boost) so a persistent API failure never
        re-blocks the 9:35 submit burst.
        """
        today = datetime.now(timezone.utc).date()
        if self._news_fetch_done_day != today:
            self._news_flags = {}
        syms = [s for s in self.candidates.keys()
                if s not in self._news_flags]
        if not syms:
            if self._news_flags:
                self._news_fetch_done_day = today
            return
        self._news_fetch_done_day = today
        try:
            if hasattr(self.alpaca, 'get_premarket_news_multi'):
                news_map = self.alpaca.get_premarket_news_multi(syms) or {}
                for sym in syms:
                    self._news_flags[sym] = news_map.get(
                        sym, {'n_articles': 0, 'headline': ''})
                n_newsy = sum(
                    1 for s2 in syms
                    if (self._news_flags.get(s2) or {}).get('n_articles', 0))
                logger.info(
                    f"[ORB] NEWS prefetch: batch {n_newsy}/{len(syms)} with "
                    f"premarket news (total covered "
                    f"{sum(1 for v in self._news_flags.values() if v is not None)}"
                    f"/{len(self._news_flags)})")
                # Pre-warm the class rule for newsy symbols (9:31, off the
                # 9:35 hot path) — only they need classification
                for s2 in syms:
                    if (self._news_flags.get(s2) or {}).get('n_articles', 0):
                        self._classify_symbol(s2)
            else:
                logger.warning(
                    "ORB: alpaca client lacks get_premarket_news_multi — "
                    "news gate fail-open (no boost) for today")
                for sym in syms:
                    self._news_flags.setdefault(sym, None)
        except Exception as e:
            logger.warning(
                f"ORB: premarket news fetch failed ({e}) — news gate "
                f"fail-open (no boost) for today")
            for sym in syms:
                self._news_flags.setdefault(sym, None)

    def _fetch_pm_dollar_vols(self) -> None:
        """Batch-fetch premarket dollar volumes for all candidates (once/day).

        2026-07-06 fixes:
        - Uses get_premarket_1min_bars_multi (get_1min_bars_multi clamps to
          the 9:30 session open BY DESIGN for range math, which silently
          starved the PM mult — pm_dollar_vol was None on every symbol).
        - Called EARLY via _maybe_prefetch_pm (any tick >= 9:31 ET) so the
          latency-critical 9:35 submit burst never blocks on this fetch;
          the lazy call inside _get_pm_mult remains as a fallback.
        """
        # 2026-07-07 fix (BEZ/IREZ fail-opens): the universe builds in
        # BATCHES (31 at 9:30:36, +21 at 9:31:43 today) and the old
        # once-per-day fetch missed every late-batch symbol. Fetch only
        # the MISSING symbols and merge, so each batch gets covered.
        today = datetime.now(timezone.utc).date()
        if self._pm_fetch_done_day != today:
            self._pm_dollar_vols = {}
        syms = [s for s in self.candidates.keys()
                if s not in self._pm_dollar_vols]
        if not syms:
            # Do NOT stamp the day on an empty pool: a pre-universe tick
            # would kill PM mult for the whole day (review 2026-07-06).
            if self._pm_dollar_vols:
                self._pm_fetch_done_day = today
            return
        self._pm_fetch_done_day = today
        try:
            if hasattr(self.alpaca, 'get_premarket_1min_bars_multi'):
                bars_map = self.alpaca.get_premarket_1min_bars_multi(syms) or {}
                for sym, b in bars_map.items():
                    self._pm_dollar_vols[sym] = compute_pm_dollar_vol(b)
                for sym in syms:
                    self._pm_dollar_vols.setdefault(sym, None)
                n_batch = sum(1 for s2 in syms if self._pm_dollar_vols.get(s2))
                logger.info(
                    f"[ORB] PM MULT prefetch: batch {n_batch}/{len(syms)} "
                    f"with premarket data (total covered "
                    f"{sum(1 for v in self._pm_dollar_vols.values() if v)}"
                    f"/{len(self._pm_dollar_vols)})")
            else:
                logger.warning(
                    "ORB: alpaca client lacks get_premarket_1min_bars_multi "
                    "— PM mult fail-open at 1.0 for today")
                for sym in syms:
                    self._pm_dollar_vols.setdefault(sym, None)
        except Exception as e:
            logger.warning(
                f"ORB: premarket bars fetch failed ({e}) — PM mult "
                f"fail-open at 1.0 for today")
            # Poison the attempted symbols with None (fail-open) so a
            # persistent API failure doesn't re-trigger this BLOCKING call
            # on every tick in the entry hot path (review 2026-07-07).
            for sym in syms:
                self._pm_dollar_vols.setdefault(sym, None)

    def _maybe_prefetch_pm(self) -> None:
        """Prefetch PM dollar volumes on any tick at/after 9:31 ET so the
        9:35 burst doesn't pay the fetch latency. No-op when disabled,
        already fetched today, or before 9:31.

        The NEWS side also feeds the catalyst veto (2026-07-19 review):
        news_needed below keeps the fetch + 9:33 lag pass alive when the
        veto is on even with pm sizing (or its news gate) disabled —
        otherwise _news_flags would stay empty, _get_has_news would
        return None for everything, and the veto would silently fail
        open into a no-op (accidental behavior; forbidden)."""
        news_needed = ((self.pm_mult_enabled and self.pm_news_gate)
                       or self.catalyst_veto_enabled)
        if not self.pm_mult_enabled and not news_needed:
            return
        today = datetime.now(timezone.utc).date()
        cands = set(self.candidates.keys())
        pm_done = ((not self.pm_mult_enabled)
                   or (self._pm_fetch_done_day == today
                       and not (cands - set(self._pm_dollar_vols.keys()))))
        news_done = ((not news_needed)
                     or (self._news_fetch_done_day == today
                         and not (cands - set(self._news_flags.keys()))))
        # 9:33 lag pass still owed? (must not be short-circuited by the
        # fetched-everything early return — review bug 2026-07-10)
        refresh_pending = (news_needed
                           and self._news_fetch_done_day == today
                           and self._news_refresh_done_day != today)
        if pm_done and news_done and not refresh_pending:
            return  # fetched today, no late-batch symbols, refresh done
        now_utc = datetime.now(timezone.utc)
        try:
            from zoneinfo import ZoneInfo
            et = now_utc.astimezone(ZoneInfo('America/New_York'))
        except Exception:
            et = now_utc - timedelta(hours=_et_offset_hours(now_utc))
        if et.time() >= dtime(9, 31):
            if not pm_done:
                self._fetch_pm_dollar_vols()
            if not news_done:
                self._fetch_news_flags()
        # Indexing-lag second pass (2026-07-10 review): Benzinga articles can
        # index minutes after publication; a symbol wrongly flagged no-news at
        # 9:31 would trade UNboosted (worst grid row: -$62K/18mo vs baseline
        # if systematic). One bounded re-fetch at >=9:33 for no-news/failed
        # symbols, UPGRADE-ONLY (an article cannot be untagged), still ahead
        # of the 9:35 submit burst and off its latency-critical path.
        if (news_needed and et.time() >= dtime(9, 33)
                and self._news_fetch_done_day == today
                and self._news_refresh_done_day != today):
            self._news_refresh_done_day = today
            stale = [s for s, f in self._news_flags.items()
                     if f is None or not (f or {}).get('n_articles')]
            if stale:
                try:
                    news_map = self.alpaca.get_premarket_news_multi(stale) or {}
                    n_new = 0
                    for sym in stale:
                        nf = news_map.get(sym)
                        if nf and nf.get('n_articles'):
                            self._news_flags[sym] = nf
                            n_new += 1
                    logger.info(
                        f"[ORB] NEWS refresh (9:33 lag pass): {n_new}/"
                        f"{len(stale)} no-news symbols flipped to newsy")
                except Exception as e:
                    logger.warning(
                        f"ORB: news refresh failed ({e}) — keeping 9:31 flags")
            self._dump_news_snapshot()

    def _dump_news_snapshot(self) -> None:
        """Persist the live news/PM view for the EoD indexing-lag audit
        (2026-07-10). The audit re-queries the news API hours later (fully
        indexed) and diffs against THIS file: any article with a premarket
        created_at that live never saw = a lag event — the failure mode
        that would silently ship the worst sizing row. Fail-soft: a write
        failure only degrades the audit, never trading."""
        try:
            import json as _json
            day = datetime.now(timezone.utc).date().isoformat()
            path = os.path.join(self._news_snapshot_dir,
                                f'orb_news_flags_{day}.json')
            snap = {
                'day': day,
                'written_at_utc': datetime.now(timezone.utc).isoformat(),
                'flags': {s: (None if f is None else int(f.get('n_articles')
                                                         or 0))
                          for s, f in self._news_flags.items()},
                'pm_dollar_vols': {s: v for s, v
                                   in self._pm_dollar_vols.items()},
            }
            with open(path, 'w') as fh:
                _json.dump(snap, fh)
        except Exception as e:
            logger.warning(
                f"ORB: news snapshot write failed ({e}) — EoD lag audit "
                f"will be unavailable for today")

    def _prewarm_anchors(self) -> None:
        """Warm underlying anchors for every candidate ahead of the 9:35
        submit burst so the catalyst veto's cohort computation is pure
        cache lookups (allow_api=False there). 20s budget per tick;
        unwarmed symbols retry next tick — each symbol costs at most ONE
        bounded API name lookup ever (results, including failures, are
        memoized in _anchor_cache / _class_names)."""
        if not self.catalyst_veto_enabled:
            return
        todo = [s for s in self.candidates.keys()
                if s not in self._anchor_cache]
        if not todo:
            return
        import time as _t
        _t0 = _t.monotonic()
        warmed = 0
        for s2 in todo:
            if _t.monotonic() - _t0 > 20.0:
                logger.warning(
                    f"[ORB] anchor pre-warm budget hit "
                    f"({warmed}/{len(todo)} this tick) — rest next tick")
                break
            try:
                self._anchor_for(s2)
            except Exception as e:
                logger.warning(f"[ORB] anchor pre-warm {s2} failed ({e}) "
                               f"— treated as no-anchor")
                self._anchor_cache[s2] = None
            warmed += 1

    def _anchor_for(self, symbol: str,
                    allow_api: bool = True) -> Optional[str]:
        """Underlying-complex anchor for the catalyst veto. Cached per
        symbol; resolves via the class map name (offline) or the asset-
        name API fallback. None = no anchor (index/commodity wrapper or
        unresolvable) -> can never be complex-confirmed.

        allow_api=False (submit-loop path): NEVER hits the network — an
        unwarmed unmapped symbol resolves to None with a WARNING. The
        prefetch pre-warms all candidates so this is a rare fallback."""
        if symbol in self._anchor_cache:
            return self._anchor_cache[symbol]
        if self._class_map is None:
            self._class_map = load_class_map()
        name = self._class_names.get(symbol)
        if name is None:
            if allow_api:
                name = self._load_class_name(symbol)
            else:
                if not self._class_names:
                    self._load_class_names_offline()
                name = self._class_names.get(symbol)
                if name is None:
                    logger.warning(
                        f"[ORB] anchor: {symbol} unwarmed in submit loop "
                        f"— treated as no-anchor (no API in hot path)")
                    self._anchor_cache[symbol] = None
                    return None
        a = underlying_anchor(symbol, name, self._class_map)
        self._anchor_cache[symbol] = a
        return a

    def _load_class_names_offline(self) -> None:
        """One-time load of the class-map CSV's names column (no API)."""
        try:
            import csv as _csv
            from trading.orb_asset_class import DEFAULT_CLASS_MAP
            with open(DEFAULT_CLASS_MAP, newline='') as fh:
                for row in _csv.DictReader(fh):
                    self._class_names[row['symbol']] = row.get('name', '')
        except Exception as e:
            logger.warning(f"[ORB] anchor: class-map names unavailable "
                           f"({e}) — API fallback per symbol")
            self._class_names['__load_failed__'] = ''

    def _load_class_name(self, symbol: str) -> Optional[str]:
        """Asset name for anchor parsing: class-map CSV names first (no
        I/O beyond first load), then the get_asset_name API fallback."""
        if not self._class_names:
            self._load_class_names_offline()
        nm = self._class_names.get(symbol)
        if nm:
            return nm
        if hasattr(self.alpaca, 'get_asset_name'):
            nm = self.alpaca.get_asset_name(symbol)
            self._class_names[symbol] = nm or ''
            return nm
        return None

    def _catalyst_veto_reject(self, cand: CandidateState) -> bool:
        """Catalyst-required veto (2026-07-18, owner-approved −$36K):
        newsless-and-alone picks are vetoed; slot consumed, no refill.
        Fail-open on UNKNOWN news (None). Evidence + semantics:
        trading/orb_catalyst_veto.py."""
        if not self.catalyst_veto_enabled:
            return False
        has_news = self._get_has_news(cand.symbol)
        anchor = self._anchor_for(cand.symbol, allow_api=False)
        cohort = anchor_cohort_counts(
            self._anchor_for(s, allow_api=False)
            for s in self.candidates.keys())
        if catalyst_veto_applies(has_news, anchor, cohort,
                                 self.catalyst_min_cohort):
            logger.info(
                f"[ORB] CATALYST VETO: {cand.symbol} newsless and alone "
                f"(anchor={anchor}, cohort="
                f"{cohort.get(anchor, 0) if anchor else 0}) — no catalyst, "
                f"slot left empty (no backfill)")
            cand.rejected_reason = 'catalyst_veto'
            self._pdr_vetoed_today.add(cand.symbol)   # same slot accounting
            cand.plan_submitted = True
            return True
        return False

    def _pdr_veto_reject(self, cand: CandidateState) -> bool:
        """PDR veto decision for one SELECTED pick (2026-07-04 ship).

        True -> skip submission; the slot stays EMPTY (no backfill — the
        refill form is toxic, see trading/orb_pdr_veto.py). Fail-open with
        a WARNING when the feature is unavailable: missing prev-day data
        drops the candidate at BT's feature stage, so fail-open cannot
        diverge from BT on any candidate BT actually traded.
        """
        if not self.pdr_veto_enabled:
            return False
        pdr = (cand.features or {}).get('prev_day_range_pct')
        if pdr is None:
            logger.warning(
                f"[ORB] PDR VETO: {cand.symbol} prev_day_range_pct "
                f"unavailable — fail-open (no veto)")
            return False
        if pdr_veto_applies(pdr, self.pdr_veto_min_pct):
            logger.info(
                f"[ORB] PDR VETO: {cand.symbol} prev-day range {pdr:.2f}% "
                f"<= {self.pdr_veto_min_pct:.1f}% — quiet prev day, "
                f"slot left empty (no backfill)")
            cand.rejected_reason = 'pdr_veto'
            # 2026-07-07 fix (IREZ incident): a vetoed pick must CONSUME its
            # daily slot like BT's one-shot top-K. Pre-fix, the slot math
            # (max_keep = cap - open_positions) recounted after an exit and
            # backfilled the NEXT-ranked name across ticks (IREZ entered 94s
            # after BEZ stopped) — the refill form the veto study proved
            # toxic. plan_submitted also stops per-tick rescoring spam.
            self._pdr_vetoed_today.add(cand.symbol)
            cand.plan_submitted = True
            return True
        return False

    def _should_defer_first_rank(self) -> bool:
        """First-rank grace gate (2026-07-03 selection-race fix).

        True iff the day's FIRST placement burst should be deferred because
        candidates in the FULL universe pool are still rangeless (their
        9:34 bar hadn't consolidated when the sweep ran) AND we are within
        `entry.first_rank_grace_s` seconds after the range end. Callers only
        invoke this when nothing has been placed today — once the first
        burst happens the day's selection is committed.

        2026-07-04 review fix: the rangeless check runs over
        `self.candidates` (the full pool), NEVER a caller-supplied subset.
        The racing path is the WS drain calling
        check_entries(symbols=<ready subset>) — every member of that subset
        has a range by construction, so a subset-scoped check was always
        empty and the gate never fired where it mattered.

        Pre-fix, the 9:35:01 ranking burned all max_concurrent daily slots
        on the ready subset; late-consolidating BT-winners (CRCD +$15.8K
        model 6/30, RGNX 6/22, FABC 6/11, AVEX 6/30) were locked out for
        the day. BT ranks the full field, so this was pure live drift.
        """
        if self.first_rank_grace_s <= 0:
            return False
        rangeless = [s for s, cand in self.candidates.items()
                     if cand.range_data is None]
        if not rangeless:
            return False
        now_utc = datetime.now(timezone.utc)
        try:
            from zoneinfo import ZoneInfo
            et_now = now_utc.astimezone(ZoneInfo('America/New_York'))
        except Exception:
            et_now = now_utc - timedelta(hours=_et_offset_hours(now_utc))
        range_end_et = (
            datetime.combine(et_now.date(), dtime(9, 30))
            .replace(tzinfo=et_now.tzinfo)
            + timedelta(minutes=int(self.range_minutes))
        )
        grace_end = range_end_et + timedelta(seconds=self.first_rank_grace_s)
        if range_end_et <= et_now < grace_end:
            logger.info(
                f"ORB: first-rank GRACE — {len(rangeless)} pool candidate(s) "
                f"still rangeless ({','.join(rangeless[:6])}"
                f"{'...' if len(rangeless) > 6 else ''}); deferring ranking "
                f"until field completes or {grace_end.strftime('%H:%M:%S')} ET"
            )
            return True
        return False

    def _audit_selection(self, ranked_symbols, top_syms, scored) -> None:
        """Append one JSON line describing this placement-burst's full field.

        Written to logs/orb_selection_audit.jsonl (gitignored runtime data).
        Captures: every scored candidate (composite/quintile/range), the
        post-dedup picks, and which pool candidates were still rangeless —
        the exact evidence that was unrecoverable for the 6/11, 6/22 and
        6/30 missed-winner incidents. Best-effort: never raises.
        """
        try:
            import json as _json
            from pathlib import Path as _Path
            rangeless = sorted(
                s for s, c in self.candidates.items()
                if c is not None and c.range_data is None
            )
            rec = {
                'ts_utc': datetime.now(timezone.utc).isoformat(),
                'ranked': [
                    {'sym': c.symbol, 'comp': round(float(c.composite), 4),
                     'q': c.quintile}
                    for c in scored
                ],
                'picks': list(top_syms),
                'rangeless_pool': rangeless,
                'open_positions': sorted(self.open_positions.keys()),
                'universe_n': len(self.universe),
            }
            p = _Path(__file__).resolve().parent.parent / 'logs' / 'orb_selection_audit.jsonl'
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'a', encoding='utf-8') as fh:
                fh.write(_json.dumps(rec, separators=(',', ':')) + '\n')
        except Exception as e:
            logger.debug(f"ORB: selection audit write failed: {e}")

    def _submit_entry(self, plan: OrbTradePlan) -> Optional[str]:
        """Submit stop-limit buy bracket for a plan.

        On success:
          - Persists pending_new DB trade record
          - Adds OpenPosition with order_id (indicates still pending)
          - Stores order_id on CandidateState for time-stop cancel
          - _process_pending_fills (next tick) transitions to filled state +
            calls stop_monitor.add_watch with lock_arm_at_r / lock_stop_r.

        Returns order_id on success, None on failure.
        """
        if self.dry_run:
            logger.info(
                f"ORB DRY-RUN: would submit {plan.symbol} "
                f"qty={plan.shares} @ stop-limit ${plan.entry_price:.2f} "
                f"(stop={plan.stop_price:.2f}, R=${plan.range_size:.2f})"
            )
            return 'dry-run-' + plan.symbol
        # Account-level halt check (set by the scanner's account-state monitor
        # on margin call / blocked / status-leaves-ACTIVE). Existing positions
        # keep being managed; only NEW entries are refused.
        from trading import system_state as _system_state
        if _system_state.is_account_halted():
            _det = _system_state.get_halt_details()
            logger.warning(
                f"ORB: {plan.symbol} entry refused — account halt active "
                f"(event={_det.get('event_type')}, since {_det.get('halted_at')})"
            )
            return None
        # Safety-net bracket legs (REAL exit is monitored client-side by StopMonitor):
        #   SL: 10% below entry (wide — belt-and-suspenders if StopMonitor WS dies)
        #   TP: 300% of entry (unreachable — ORB has no fixed target; legally must set)
        safety_sl = round(plan.entry_price * (1.0 - self.safety_sl_pct), 2)
        safety_tp = round(plan.entry_price * 3.0, 2)  # unreachable safety-net
        # BT parity: stop triggers at range_high (BT: `if row['high'] > range_high`),
        # limit at range_high × (1 + entry_slip_bps/10000) = plan.entry_price
        # (30bps slippage budget baked in the planner — DO NOT double-apply here).
        stop_trigger = round(plan.range_high, 2)
        limit_price = round(plan.entry_price, 2)

        # Fetch the NBBO at submit BEFORE we decide how to submit. Used for
        # both the buy-stop rejection guard AND entry-slippage telemetry —
        # one quote, two purposes. Done early so the guard can pick the
        # right submission path (marketable / rebump / skip / as-is). Quote
        # fields persist to DB via _save_pending_trade so they survive a
        # restart (previously lost — reconstructed OpenPosition had
        # entry_quote_ask=None after sync_positions).
        entry_quote_ask: Optional[float] = None
        submit_bid, submit_ask = 0.0, 0.0
        submit_bid_size, submit_ask_size = 0, 0
        try:
            quote = self.alpaca.get_latest_quote(plan.symbol) if hasattr(self.alpaca, 'get_latest_quote') else None
            if quote:
                submit_bid = float(quote.get('bid_price', 0.0) or 0.0)
                submit_ask = float(quote.get('ask_price', 0.0) or 0.0)
                submit_bid_size = int(quote.get('bid_size', 0) or 0)
                submit_ask_size = int(quote.get('ask_size', 0) or 0)
                if submit_ask > 0:
                    entry_quote_ask = submit_ask
        except Exception as e:
            logger.debug(f"ORB: get_latest_quote({plan.symbol}) failed at submit: {e}")

        # Pre-flight buy-stop rejection guard (shared with bull flag via
        # trading/buy_stop_guard.py — parity by construction). Alpaca LIVE
        # rejects buy stop-limit with stop_price <= ask. The guard returns
        # one of: SUBMIT_AS_IS, MARKETABLE_LIMIT, REBUMP_STOP, SKIP.
        # 2026-05-18 ORB rejections (BTCZ/YSS/BMNZ) all match this pattern.
        _guard_cfg = self._buy_stop_guard_cfg or {}
        _guard_enabled = _guard_cfg.get('enabled', True)
        _rebump_buffer = _guard_cfg.get('rebump_buffer', 0.02)
        if _guard_enabled:
            decision = evaluate_buy_stop(
                bid=submit_bid, ask=submit_ask,
                stop_price=stop_trigger, limit_price=limit_price,
                rebump_buffer=_rebump_buffer,
            )
        else:
            decision = BuyStopDecision(
                action=BuyStopAction.SUBMIT_AS_IS,
                reason="guard disabled by config",
            )

        try:
            if decision.action == BuyStopAction.MARKETABLE_LIMIT:
                logger.info(
                    f"ORB: {plan.symbol} STOP ALREADY TRIGGERED "
                    f"({decision.reason}) — submitting as marketable "
                    f"LIMIT-BRACKET @ ${limit_price:.2f} (SL ${safety_sl:.2f}, "
                    f"TP ${safety_tp:.2f}) to avoid Alpaca live rejection"
                )
                result = self.alpaca.submit_bracket_order(
                    symbol=plan.symbol,
                    qty=plan.shares,
                    side='buy',
                    limit_price=limit_price,
                    tp_price=safety_tp,
                    sl_price=safety_sl,
                )
            elif decision.action == BuyStopAction.SKIP:
                logger.warning(
                    f"ORB: {plan.symbol} ENTRY SKIPPED — {decision.reason}. "
                    f"Not chasing."
                )
                return None
            elif decision.action == BuyStopAction.REBUMP_STOP:
                _new_stop = decision.new_stop_price
                logger.info(
                    f"ORB: {plan.symbol} STOP STRADDLED BY SPREAD "
                    f"({decision.reason}) — limit ${limit_price:.2f} unchanged"
                )
                result = self.alpaca.submit_stop_bracket_order(
                    symbol=plan.symbol,
                    qty=plan.shares,
                    side='buy',
                    stop_price=_new_stop,
                    limit_price=limit_price,
                    tp_price=safety_tp,
                    sl_price=safety_sl,
                )
            else:  # SUBMIT_AS_IS
                result = self.alpaca.submit_stop_bracket_order(
                    symbol=plan.symbol,
                    qty=plan.shares,
                    side='buy',
                    stop_price=stop_trigger,
                    limit_price=limit_price,
                    tp_price=safety_tp,
                    sl_price=safety_sl,
                )
            if not result:
                logger.error(f"ORB: {plan.symbol} alpaca submit returned empty")
                return None
            order_id = result.get('id') or ''
            # 2026-06-09 (FABC fix): extract bracket leg IDs so StopMonitor's
            # BRANCH_SL_LEG_RACE recovery path can query the SL leg fill.
            # alpaca_client.submit_bracket_order now returns 'legs' list
            # (mirroring get_order's leg shape). Pick TP by limit_price,
            # SL by stop_price.
            _legs = result.get('legs') or []
            tp_leg_id = next(
                (str(leg.get('id', '')) for leg in _legs
                 if leg.get('limit_price') is not None and leg.get('side') == 'sell'),
                '',
            )
            sl_leg_id = next(
                (str(leg.get('id', '')) for leg in _legs
                 if leg.get('stop_price') is not None and leg.get('side') == 'sell'),
                '',
            )
            if _legs and not (tp_leg_id and sl_leg_id):
                logger.warning(
                    f"ORB: {plan.symbol} bracket leg extraction incomplete "
                    f"(tp={bool(tp_leg_id)}, sl={bool(sl_leg_id)}, "
                    f"n_legs={len(_legs)}) — SL race recovery degraded"
                )
        except Exception as e:
            logger.error(f"ORB: {plan.symbol} submit_entry failed: {e}")
            return None

        # Single submit_time used for both DB persistence AND the in-memory position.
        # This ensures sync_positions on restart can recover the exact original
        # submit time from DB (vs the prior bug where rehydration used now()).
        submit_time = datetime.now(timezone.utc)
        trade_id = self._save_pending_trade(
            plan, order_id,
            submit_time=submit_time,
            submit_bid=submit_bid, submit_ask=submit_ask,
            submit_bid_size=submit_bid_size, submit_ask_size=submit_ask_size,
        )
        if trade_id is None:
            # Order was accepted by Alpaca but we failed to persist tracking.
            # Operator must decide: cancel the orphan order or manually record.
            self._notify_error(
                f"{plan.symbol}: DB save failed AFTER Alpaca accepted order "
                f"{order_id}. Order is LIVE on Alpaca but ORB engine will not "
                f"track it. CANCEL MANUALLY or insert DB row before next fill."
            )
            return None

        # Fill-rate telemetry
        self.daily_n_placed += 1

        # Track pending position (cleared once fill confirmed in _process_pending_fills)

        # Range end for breakout-bar keying (2026-07-03 fix): derived from the
        # candidate's range_data, which was computed at range-completion time
        # from a df that HAD the 9:30 anchor (WS-early or backfill). Entries
        # gate at 10:00 ET, so any entered position's range is true-anchored.
        _range_end_ts = None
        _cand = self.candidates.get(plan.symbol)
        if _cand is not None and _cand.range_data is not None:
            try:
                _rst = _cand.range_data.range_start_ts
                if _rst is not None:
                    _range_end_ts = (pd.Timestamp(_rst).tz_localize('UTC')
                                     if pd.Timestamp(_rst).tzinfo is None
                                     else pd.Timestamp(_rst)) \
                        + timedelta(minutes=self.range_minutes)
            except Exception as e:
                logger.warning(
                    f"ORB: {plan.symbol} range_end_ts derivation failed ({e}) — "
                    f"breakout-bar keying will use window-anchor fallback"
                )

        self.open_positions[plan.symbol] = OpenPosition(
            symbol=plan.symbol,
            entry_price=plan.entry_price,  # placeholder — overwritten on fill
            stop_price=plan.stop_price,
            shares=plan.shares,
            trade_id=trade_id,
            order_id=order_id,
            entry_time=submit_time,
            range_high=plan.range_high,
            range_low=plan.range_low,
            lock_arm_at_r=plan.lock_arm_at_r,
            lock_stop_r=plan.lock_stop_r,
            composite_score=plan.composite_score,
            quintile=plan.quintile,
            bar_close_price=plan.range_high,
            order_submitted_at=submit_time,
            entry_quote_ask=entry_quote_ask,
            range_end_ts=_range_end_ts,
            tp_leg_id=tp_leg_id,
            sl_leg_id=sl_leg_id,
        )
        # Track on CandidateState too for clean time-stop cancellation
        cand = _cand
        if cand is not None:
            cand.order_id = order_id
            cand.order_submitted_at = submit_time

        # Start passive quote watch for entry microstructure (bid/ask/depth/OFI).
        # Harmless if already watched; safe no-op if StopMonitor is None.
        if self.stop_monitor is not None:
            try:
                self.stop_monitor.add_quote_watch(
                    plan.symbol,
                    submit_bid=submit_bid, submit_ask=submit_ask,
                    submit_bid_size=submit_bid_size, submit_ask_size=submit_ask_size,
                )
            except Exception as e:
                logger.debug(f"ORB: add_quote_watch({plan.symbol}) failed: {e}")

        if self.notify_on_entry and self.notifier:
            self._notify(
                f"{self.tg_prefix} ENTRY SUBMITTED {plan.symbol} x{plan.shares} "
                f"@ stop-limit ${plan.entry_price:.2f} (stop ${plan.stop_price:.2f}, "
                f"{plan.quintile} mult={plan.adaptive_mult:.2f}x, "
                f"composite={plan.composite_score:+.2f})"
            )
        logger.info(
            f"ORB ENTRY SUBMITTED: {plan.symbol} order={order_id} "
            f"qty={plan.shares} entry=${plan.entry_price:.2f} "
            f"stop=${plan.stop_price:.2f} trade_id={trade_id}"
        )
        return order_id

    def _process_pending_fills(self) -> None:
        """For each open position with non-empty order_id (pending), poll order
        status; on fill, update DB + add StopMonitor watch.

        Matches MACD wave pattern (trading/macd_wave_engine.py:1400+). Uses
        OrderStreamWatcher fast path when available, falls back to REST.
        """
        for sym, pos in list(self.open_positions.items()):
            if not pos.order_id:
                continue  # already filled
            try:
                order_status = None
                if self.order_stream is not None:
                    order_status = self.order_stream.get_status(pos.order_id)
                if order_status is None:
                    # First few seconds post-submit: OrderStream may be empty.
                    # After 5s, always fall back to REST.
                    age_s = (datetime.now(timezone.utc) - pos.entry_time).total_seconds()
                    if age_s >= 5.0 or self.order_stream is None:
                        try:
                            order_status = self.alpaca.get_order(pos.order_id)
                        except Exception as e:
                            logger.debug(f"ORB: get_order({pos.order_id}) failed: {e}")
                            continue
                    else:
                        continue
                if order_status is None:
                    continue
                status = order_status.get('status', '') if isinstance(order_status, dict) \
                         else getattr(order_status, 'status', '')
                status = str(status).lower()

                if status == 'filled':
                    self._confirm_fill(pos, order_status)
                elif status == 'partially_filled':
                    # FABC 2026-06-09 fix: previously this branch accepted
                    # the FIRST observed partial as terminal and called
                    # _confirm_fill, which clears pos.order_id → polling
                    # stops → subsequent partials (and the final 'filled'
                    # transition) were never recorded. FABC ordered 3,188 sh
                    # in a multi-fill sequence (~1438 → 3188 within seconds);
                    # pre-fix engine recorded shares=1438 and reported a
                    # $561 loss while the broker filled all 3,188 and the
                    # real account loss was $1,243.
                    #
                    # New behavior (mirrors bull-flag's _manage_pending_orders
                    # at trading/trading_engine.py): on partial, just log and
                    # keep polling. Terminal 'filled' status drives the
                    # _confirm_fill below. The stall-timeout branch above
                    # is a safety net for orders the broker never marks
                    # 'filled' (rare; we still capture the observed qty).
                    if isinstance(order_status, dict):
                        _filled_qty = order_status.get('filled_qty', 0)
                        _req_qty = order_status.get('qty', 0)
                    else:
                        _filled_qty = getattr(order_status, 'filled_qty', 0)
                        _req_qty = getattr(order_status, 'qty', 0)
                    try:
                        _filled_int = int(_filled_qty) if _filled_qty else 0
                        _req_int = int(_req_qty) if _req_qty else 0
                    except (ValueError, TypeError):
                        _filled_int = 0
                        _req_int = 0
                    now_utc = datetime.now(timezone.utc)
                    if pos.first_partial_at is None:
                        pos.first_partial_at = now_utc
                        logger.info(
                            f"ORB: {sym} partial — filled {_filled_int}/"
                            f"{_req_int} sh, continuing to poll (stall-"
                            f"timeout={self.partial_fill_stall_seconds_max}s)"
                        )
                    elif _filled_int > pos.last_observed_filled_qty:
                        # Progress — re-log so we have telemetry on multi-bar
                        # fills without spamming on flat polls.
                        logger.info(
                            f"ORB: {sym} partial fill progress — "
                            f"{pos.last_observed_filled_qty} → {_filled_int}"
                            f"/{_req_int} sh"
                        )
                    pos.last_observed_filled_qty = max(
                        pos.last_observed_filled_qty, _filled_int
                    )
                    # Stall timeout: broker reported partials but never
                    # transitioned to terminal 'filled'. Cancel remainder +
                    # accept observed qty. _confirm_fill writes the actual
                    # filled qty (the order_status payload at this point
                    # carries the latest broker-reported fill).
                    elapsed = (now_utc - pos.first_partial_at).total_seconds()
                    if elapsed > self.partial_fill_stall_seconds_max:
                        _remaining = max(_req_int - _filled_int, 0)
                        logger.warning(
                            f"ORB: {sym} partial-fill stall — "
                            f"{_filled_int}/{_req_int} sh after {elapsed:.0f}s; "
                            f"cancelling remaining {_remaining} sh + "
                            f"accepting observed qty as final"
                        )
                        if _remaining > 0 and pos.order_id:
                            try:
                                self.alpaca.cancel_order(pos.order_id)
                            except Exception as e:
                                logger.error(
                                    f"ORB: {sym} stall-cancel FAILED: {e} — "
                                    f"position may continue growing; "
                                    f"sync_positions orphan-detect is the "
                                    f"only backstop"
                                )
                        # 2026-07-04 review fix: shares can fill between the
                        # poll above and the cancel-ack. Confirming from the
                        # stale snapshot would leave those shares outside the
                        # StopMonitor watch (naked overnight on a gap). Re-
                        # fetch once and confirm from the freshest payload.
                        final_status = order_status
                        try:
                            refetched = self.alpaca.get_order(pos.order_id)
                            if refetched:
                                final_status = refetched
                        except Exception as e:
                            logger.warning(
                                f"ORB: {sym} post-cancel re-fetch failed: {e} "
                                f"— confirming from pre-cancel snapshot"
                            )
                        self._confirm_fill(pos, final_status)
                elif status in ('canceled', 'cancelled', 'expired', 'rejected',
                                 'done_for_day', 'suspended'):
                    # 2026-07-04 review fix: since the FABC partial-fill fix,
                    # order_id stays set during partials — so a broker-side
                    # cancel/expiry CAN now carry filled_qty > 0. Those shares
                    # are held; dropping tracking would leave them unmanaged
                    # (no watch, no open DB row). Route through _confirm_fill.
                    try:
                        _fq = int(order_status.get('filled_qty', 0) or 0)                             if isinstance(order_status, dict)                             else int(getattr(order_status, 'filled_qty', 0) or 0)
                    except (TypeError, ValueError):
                        _fq = 0
                    if _fq > 0:
                        logger.warning(
                            f"ORB: {sym} order {pos.order_id} terminal "
                            f"'{status}' WITH {_fq} sh filled — confirming "
                            f"partial as final position (not dropping)"
                        )
                        self._confirm_fill(pos, order_status)
                        continue
                    logger.warning(
                        f"ORB: {sym} order {pos.order_id} terminal status '{status}' — "
                        f"clearing tracking"
                    )
                    self.open_positions.pop(sym, None)
                    cand = self.candidates.get(sym)
                    if cand:
                        cand.rejected_reason = f'order_{status}'
                        cand.order_id = None
                    try:
                        self.db.update_trade(pos.trade_id, {'order_status': status})
                    except Exception:
                        pass
                # else: pending_new / accepted / new / pending_cancel → keep polling
            except Exception as e:
                logger.error(f"ORB: _process_pending_fills({sym}) error: {e}")

    def _confirm_fill(self, pos: OpenPosition, order_status) -> None:
        """Transition a pending OpenPosition to filled + arm StopMonitor watch."""
        if isinstance(order_status, dict):
            fill_price = order_status.get('filled_avg_price') or order_status.get('fill_price')
            filled_qty = order_status.get('filled_qty', pos.shares)
        else:
            fill_price = getattr(order_status, 'filled_avg_price', None)
            filled_qty = getattr(order_status, 'filled_qty', pos.shares)

        fill_price = float(fill_price) if fill_price else pos.entry_price
        try:
            shares = int(filled_qty) if filled_qty else pos.shares
        except (TypeError, ValueError):
            shares = pos.shares
        # 2026-05-08 fix: prefer Alpaca's actual filled_at timestamp instead
        # of NOW(). Pre-fix, polling-based fill confirmation (after restart)
        # wrote sync-time as filled_at, clobbering real fill times by minutes
        # to hours (ASTX 5/6 was -5h50m off; OKLS 5/7 was -13min off).
        if isinstance(order_status, dict):
            alp_fill_at = order_status.get('filled_at')
        else:
            alp_fill_at = getattr(order_status, 'filled_at', None)
        if alp_fill_at is not None:
            try:
                if isinstance(alp_fill_at, str):
                    from dateutil.parser import isoparse as _iso
                    alp_fill_at = _iso(alp_fill_at)
                if alp_fill_at.tzinfo is None:
                    alp_fill_at = alp_fill_at.replace(tzinfo=timezone.utc)
                fill_at = alp_fill_at
            except Exception:
                fill_at = datetime.now(timezone.utc)
        else:
            fill_at = datetime.now(timezone.utc)

        pos.entry_price = fill_price
        pos.shares = shares
        pos.entry_time = fill_at
        pos.order_id = ''  # cleared = filled
        # Touchgo breakout bar (Rule M / Rule D reference). See _evaluate_touchgo.
        if self.touchgo_cfg.breakout_bar_source == 'fill':
            # Legacy rollback path: minute-floor of the fill time (pre-2026-06).
            pos.breakout_bar_ts = fill_at.replace(second=0, microsecond=0)
        elif pos.breakout_bar_ts is None:
            # Market mode (BT-parity, default): the breakout bar is normally
            # captured during the pending phase by _ensure_breakout_bar_ts. If a
            # very fast fill beat the first post-fill bar event, best-effort from
            # the rolling window now; _ingest_bars sets it on the next event
            # otherwise. If neither resolves it, touchgo stays inert for this
            # position (logged in _ensure path) — acceptable for a stale entry.
            window = self._bar_windows.get(pos.symbol)
            if window:
                try:
                    bb = self._market_breakout_bar_ts(
                        pd.DataFrame(window), pos.range_high
                    )
                    if bb is not None:
                        pos.breakout_bar_ts = bb
                except Exception as e:
                    logger.warning(
                        f"ORB: breakout-bar capture at fill failed for {pos.symbol}: {e}"
                    )
        pos.rule_m_evaluated = False
        pos.rule_d_evaluated = False
        # Fill-rate telemetry
        self.daily_n_filled += 1

        # Risk-per-share recomputed from ACTUAL fill
        risk_per_share = max(pos.entry_price - pos.stop_price, 0.0)

        # DB update — include entry slippage attribution (parity with bull flag + MACD wave).
        # 2026-06-05 fix: also persist the ACTUAL filled qty. Without this,
        # `shares` in the DB stays as the original order qty (e.g., RPGL 6-04
        # showed shares=3596 in the DB but only 1 share actually filled), and
        # any downstream notional/exposure query computed off `shares` is wrong.
        # Both `shares` and `filled_qty` get the actual filled count — the
        # original order qty is recoverable from Alpaca order history if needed.
        fill_update: Dict[str, object] = {
            'order_status': 'filled',
            'fill_price': fill_price,
            'filled_at': fill_at,
            'order_filled_at': fill_at,
            'shares': shares,
            'filled_qty': shares,
        }
        if pos.order_submitted_at is not None:
            fill_update['order_submitted_at'] = pos.order_submitted_at
            try:
                fill_update['submit_to_fill_ms'] = int(
                    (fill_at - pos.order_submitted_at).total_seconds() * 1000
                )
            except (TypeError, ValueError) as e:
                logger.debug(f"ORB: submit_to_fill_ms skip ({e})")
        if pos.bar_close_price and pos.bar_close_price > 0:
            fill_update['bar_close_price'] = pos.bar_close_price
            fill_update['drift_bar_to_fill_bps'] = (
                (fill_price - pos.bar_close_price) / pos.bar_close_price * 10000.0
            )
        if pos.entry_quote_ask and pos.entry_quote_ask > 0:
            fill_update['drift_ask_to_fill_bps'] = (
                (fill_price - pos.entry_quote_ask) / pos.entry_quote_ask * 10000.0
            )
        try:
            self.db.update_trade(pos.trade_id, fill_update)
        except Exception as e:
            # DB missed the fill transition. Alpaca says filled but our DB
            # still shows pending_new. Next sync will re-try; meanwhile exit
            # paths that query DB may misbehave. Operator should verify.
            self._notify_error(
                f"{pos.symbol}: DB update failed on fill confirm "
                f"(fill=${fill_price:.2f} x{shares}). Row {pos.trade_id} may "
                f"show wrong state until next sync.",
                exc=e,
            )

        # Entry microstructure from quote watch (bid/ask/depth/OFI at submit + fill).
        if self.stop_monitor is not None and hasattr(self.stop_monitor, 'get_quote_watch_snapshot'):
            try:
                qsnap = self.stop_monitor.get_quote_watch_snapshot(pos.symbol)
                if qsnap:
                    entry_micro = {
                        'entry_quote_bid': qsnap.get('submit_bid'),
                        'entry_quote_ask': qsnap.get('submit_ask'),
                        'entry_quote_bid_size': qsnap.get('submit_bid_size'),
                        'entry_quote_ask_size': qsnap.get('submit_ask_size'),
                        'entry_quote_spread': (
                            qsnap['submit_ask'] - qsnap['submit_bid']
                            if qsnap.get('submit_ask', 0) > 0 else None
                        ),
                        'entry_quote_ofi': qsnap.get('ofi_cumulative'),
                        'entry_fill_quote_bid': qsnap.get('latest_bid'),
                        'entry_fill_quote_ask': qsnap.get('latest_ask'),
                    }
                    self.db.update_trade(pos.trade_id, entry_micro)
                    logger.info(
                        f"ORB: {pos.symbol} entry microstructure — "
                        f"submit bid=${qsnap.get('submit_bid', 0):.2f} "
                        f"ask=${qsnap.get('submit_ask', 0):.2f}, "
                        f"fill bid=${qsnap.get('latest_bid', 0):.2f} "
                        f"ask=${qsnap.get('latest_ask', 0):.2f}"
                    )
            except Exception as e:
                logger.debug(f"ORB: entry quote-watch snapshot failed: {e}")
            try:
                self.stop_monitor.remove_quote_watch(pos.symbol)
            except Exception as e:
                logger.debug(f"ORB: remove_quote_watch({pos.symbol}) failed: {e}")

        # Register StopMonitor watch with static lock.
        # BT parity: lock math uses range_size (range_high - range_low) as 1R.
        # This is distinct from risk_per_share (entry_price - range_low), which
        # differs by the slippage buffer baked into entry_price.
        range_size = max(pos.range_high - pos.range_low, 0.0)
        # BT parity: skip exits during the entry bar (BT's sim_bars.iloc[1:]
        # skips the entry bar entirely). Compute end-of-current-minute as the
        # skip boundary — all ticks before that timestamp are ignored for
        # stop/lock checks (but peak tracking continues so arming fires
        # correctly as soon as the skip window closes).
        next_bar_start = fill_at.replace(second=0, microsecond=0) + timedelta(minutes=1)
        skip_until_ts = next_bar_start.timestamp()
        if self.stop_monitor is not None:
            try:
                self.stop_monitor.add_watch(
                    symbol=pos.symbol,
                    stop_price=pos.stop_price,
                    shares=pos.shares,
                    tp_leg_id=pos.tp_leg_id,
                    sl_leg_id=pos.sl_leg_id,
                    trade_db_id=pos.trade_id,
                    entry_price=pos.entry_price,
                    risk_per_share=risk_per_share,
                    strategy=STRATEGY_NAME,
                    lock_arm_at_r=pos.lock_arm_at_r,
                    lock_stop_r=pos.lock_stop_r,
                    lock_r_unit=range_size,  # BT-parity: 1R = range_size, not risk_per_share
                    skip_exits_until_ts=skip_until_ts,  # BT-parity: skip entry bar
                )
                # 1R for the displayed lock levels is range_size (matches the
                # lock_r_unit passed to StopMonitor), not risk_per_share.
                # risk_per_share differs by the slippage buffer baked into
                # entry_price — using it here misreports the actual lock levels
                # by a couple cents on wide-range setups.
                lock_arm_px = pos.entry_price + pos.lock_arm_at_r * range_size
                lock_stop_px = pos.entry_price + pos.lock_stop_r * range_size
                logger.info(
                    f"ORB FILL: {pos.symbol} @ ${fill_price:.2f} x{shares} — "
                    f"StopMonitor watching (stop=${pos.stop_price:.2f}, "
                    f"lock arms at ${lock_arm_px:.2f}, lock stop ${lock_stop_px:.2f})"
                )
            except Exception as e:
                # CRITICAL: position filled but StopMonitor isn't watching it.
                # Only the safety-net bracket SL (10% below entry) will exit
                # this trade. Operator must manually add a tighter stop or
                # rebuild the watch.
                self._notify_error(
                    f"{pos.symbol}: filled @ ${fill_price:.2f} x{shares} but "
                    f"StopMonitor.add_watch FAILED. Safety-net SL at "
                    f"${pos.entry_price * (1 - self.safety_sl_pct):.2f} is the "
                    f"ONLY active stop. Rebuild watch or cancel position.",
                    exc=e,
                )

        # Clear candidate order tracking (position is now live)
        cand = self.candidates.get(pos.symbol)
        if cand:
            cand.order_id = None

    def _cancel_stale_pending_orders(self) -> None:
        """Cancel any stop-buy orders still pending past the 10:35 ET time stop
        (range_end + time_stop_minutes). Matches BT behavior — BT's simulator
        skips entries that don't trigger within time_stop_minutes.
        """
        now_utc = datetime.now(timezone.utc)
        for sym, cand in list(self.candidates.items()):
            if not cand.order_id or not cand.order_submitted_at:
                continue
            pos = self.open_positions.get(sym)
            if pos is None or not pos.order_id:
                continue  # already filled (pos.order_id cleared in _confirm_fill)
            age_min = (now_utc - cand.order_submitted_at).total_seconds() / 60.0
            if age_min >= self.time_stop_minutes:
                try:
                    self.alpaca.cancel_order(pos.order_id)
                    logger.info(
                        f"ORB TIME-STOP CANCEL: {sym} unfilled after "
                        f"{age_min:.0f}min — cancelling order {pos.order_id}"
                    )
                    # 2026-07-04 review fix: since the FABC partial-fill fix
                    # order_id stays set during partials, so a time-stopped
                    # order can carry filled shares. Check once post-cancel;
                    # if any filled, keep the position (confirm at final qty)
                    # instead of dropping held shares from management.
                    try:
                        final = self.alpaca.get_order(pos.order_id)
                        _fq = int((final or {}).get('filled_qty', 0) or 0)                             if isinstance(final, dict)                             else int(getattr(final, 'filled_qty', 0) or 0)
                    except Exception:
                        _fq = 0
                    if _fq > 0:
                        logger.warning(
                            f"ORB TIME-STOP CANCEL: {sym} had {_fq} sh "
                            f"already filled — confirming partial as final "
                            f"position (not dropping)"
                        )
                        self._confirm_fill(pos, final)
                        cand.order_id = None
                        self.daily_n_time_stop_canceled += 1
                        continue
                    self.db.update_trade(pos.trade_id, {'order_status': 'time_stop_canceled'})
                    self.open_positions.pop(sym, None)
                    cand.rejected_reason = 'time_stop'
                    cand.order_id = None
                    self.daily_n_time_stop_canceled += 1
                    # Free the quote-watch subscription — we're no longer
                    # chasing a fill for this symbol.
                    if self.stop_monitor is not None:
                        try:
                            self.stop_monitor.remove_quote_watch(sym)
                        except Exception:
                            pass
                except Exception as e:
                    logger.error(f"ORB: time-stop cancel({sym}) failed: {e}")

    # =====================================================================
    # Exit flow
    # =====================================================================

    def check_exits(self) -> List[str]:
        """Drain StopMonitor exit events tagged strategy='orb'.

        Updates DB + in-memory state for each exit. Orphan reconciliation
        is NOT triggered here — the L7 periodic intraday hook
        (2026-06-05) introduced race A with the sync_positions recovery
        loop and was bypassed in practice. Reconciler is startup-only:
        cross-day orphans get caught at the next reset_daily.

        Returns:
            list of symbols that exited on this tick.
        """
        if self.stop_monitor is None:
            return []
        exited: List[str] = []
        try:
            events = self.stop_monitor.drain_exit_events(strategy=STRATEGY_NAME)
        except TypeError:
            # Back-compat: older drain_exit_events doesn't accept strategy filter
            events = [e for e in self.stop_monitor.drain_exit_events()
                      if getattr(e, 'strategy', 'bull_flag') == STRATEGY_NAME]
        for ev in events:
            self._handle_exit_event(ev)
            exited.append(ev.symbol)
        return exited

    def _handle_exit_event(self, ev) -> None:
        """Update DB + open_positions from a StopMonitor exit event."""
        symbol = ev.symbol
        pos = self.open_positions.pop(symbol, None)
        if pos is None:
            logger.warning(f"ORB: exit event for {symbol} but no tracked position — orphan?")
            return
        confirmed = getattr(ev, 'confirmed', True)
        exit_price = float(ev.exit_price)
        pnl = (exit_price - pos.entry_price) * pos.shares
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100.0

        def _numf(v):
            """Coerce to float if real numeric (not MagicMock), else None."""
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None
            f = float(v)
            return f if f > 0 else None

        # Shared build_exit_update gives us the confirmed/unconfirmed-correct
        # base payload; ORB adds its own exit_slippage + qty fields on top.
        exit_update = build_exit_update(ev)
        exit_limit_price = exit_update.get('exit_limit_price')
        if confirmed:
            self.daily_pnl += pnl
            exit_update['pnl'] = pnl
            exit_update['pnl_pct'] = pnl_pct
            exit_update['exit_slippage'] = (
                exit_limit_price - exit_price if exit_limit_price is not None else None
            )
        else:
            # Unconfirmed: no daily_pnl mutation, no exit_price/pnl write.
            # The orphan reconciler will retry. Keep the position visible to
            # ORB's per-day caps (exit_pending_verification stays "open").
            logger.error(
                f"ORB: {symbol} UNCONFIRMED EXIT (BRANCH_LAST_RESORT) — "
                f"order_status=exit_pending_verification, "
                f"exit_reason={ev.exit_reason}. Reconciler will retry."
            )
        try:
            self.db.update_trade(pos.trade_id, exit_update)
        except Exception as e:
            logger.error(f"ORB: {symbol} DB update failed: {e}")

        if self.notify_on_exit and self.notifier:
            self._notify(
                f"{self.tg_prefix} EXIT {symbol} @ ${exit_price:.2f} "
                f"({ev.exit_reason}) — PnL ${pnl:+,.2f} ({pnl_pct:+.2f}%)"
            )
        logger.info(
            f"ORB EXIT: {symbol} {ev.exit_reason} @ ${exit_price:.2f} "
            f"pnl=${pnl:+,.2f} daily_pnl=${self.daily_pnl:+,.2f}"
        )

    # GLWG 2026-05-11 FC: bracket cancel ACK can take 1-5s+ on busy Alpaca
    # paper accounts. Hardcoded sleep(0.8) before close was insufficient —
    # close_position fired with held_for_orders=1431, triggering false-alarm
    # CRITICAL leaks. Backoff schedule mirrors StopMonitor's
    # _HELD_QTY_RETRY_BACKOFFS_S but with longer values (FC has more time
    # headroom than a stop exit). Cumulative ~6.5s.
    _FC_HELD_QTY_BACKOFFS_S = (0.5, 1.0, 2.0, 3.0)
    # Phase-1b pre-close sleep — gives Alpaca a head-start on cancel-ACK
    # propagation so attempt 0 of the retry helper doesn't always hit
    # held_for_orders (Bug 6: pre-fix code-review found every FC produced
    # at least one "race attempt 1/5" warning even on the happy path).
    # 200ms covers the vast majority of cancel-ACK latencies.
    _FC_PHASE1B_PRE_CLOSE_SLEEP_S = 0.2

    @staticmethod
    def _is_retryable_close_error(e: Exception) -> bool:
        """Identify transient errors worth retrying on close_position.

        - 40310000 / 'insufficient qty available' — bracket-cancel race
          (Alpaca's specific code surfaced in GLWG/APT/MLTX 2026-05-11)
        - 'rate limit' / 'too many requests' — 429 throttling
        - 5xx phrasing — server-side hiccups (avoid raw '500' substring
          which false-positives on prices like 5153)
        """
        s = str(e)
        sl = s.lower()
        if '40310000' in s or 'insufficient qty available' in sl:
            return True
        if 'rate limit' in sl or 'too many requests' in sl:
            return True
        if any(p in sl for p in (
            'internal server error', 'bad gateway',
            'service unavailable', 'gateway timeout',
        )):
            return True
        return False

    def _close_position_with_held_qty_retry(self, sym: str):
        """close_position with backoff retry on transient errors.

        Retries on: 40310000 / insufficient-qty-available (bracket-cancel
        race), 429 rate-limit, and 5xx server errors. Re-raises any other
        exception immediately so the caller's existing error paths still
        trigger.

        Bracket-cancel ACK propagation on Alpaca is async — the qty stays
        `held_for_orders` for hundreds of ms to several seconds after the
        cancel ACK lands. Bug-9 fix (post-code-review): broadened from the
        pre-fix narrow '40310000-only' check.

        Returns the order submission dict on success (NOT a fill confirmation
        — caller must verify fill separately).
        """
        import time as _time
        attempts = 1 + len(self._FC_HELD_QTY_BACKOFFS_S)
        last_err = None
        for attempt in range(attempts):
            if attempt > 0:
                _time.sleep(self._FC_HELD_QTY_BACKOFFS_S[attempt - 1])
            try:
                return self.alpaca.close_position(sym)
            except Exception as e:
                last_err = e
                if not self._is_retryable_close_error(e):
                    raise
                logger.warning(
                    f"ORB FC: {sym} close_position retryable error "
                    f"(attempt {attempt + 1}/{attempts}): {e} — "
                    f"{'retrying' if attempt + 1 < attempts else 'giving up'}"
                )
        if last_err is not None:
            raise last_err
        return None

    def _is_close_order_still_pending(self, order_id: str) -> bool:
        """True if a specific close order is still working (not terminal).

        Bug-2 fix (post-code-review): replaces the pre-fix
        `_has_pending_sell(symbol)` which conflated Phase-1's full-qty
        close with partial-qty bracket legs that survived a failed cancel.
        Order-specific lookup is precise — we only wait on OUR pending
        close, not on a stranded TP/SL leg that won't close the position.

        Returns False on API error (fail-open: FC VERIFY then proceeds
        with its cancel+re-close, which is the safer side of the race).
        """
        try:
            o = self.alpaca.get_order(order_id)
        except Exception as e:
            logger.warning(
                f"ORB FC: get_order({order_id[:8]}) check failed "
                f"(assume not pending): {e}"
            )
            return False
        if o is None:
            return False
        status = (
            o.get('status') if isinstance(o, dict)
            else getattr(o, 'status', '')
        )
        status_val = getattr(status, 'value', status)
        status_str = str(status_val).lower()
        return status_str not in (
            'filled', 'canceled', 'cancelled', 'rejected',
            'expired', 'done_for_day', 'suspended', 'replaced',
        )

    def _cancel_symbol_open_orders(self, sym: str) -> int:
        """Cancel ALL open Alpaca orders for a symbol (SL/TP bracket legs, etc).

        Required before `close_position` — if any sell order is live for the
        symbol, Alpaca reports available=0 / held_for_orders=N and rejects
        the close with "insufficient qty available". On 2026-04-20 this
        caused ANNA to leak overnight (-$2.5K) because the unreachable TP
        leg held all 11,682 shares.

        Returns:
            Number of orders canceled.
        """
        canceled = 0
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            open_orders = self.alpaca.trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[sym])
            ) or []
        except Exception as e:
            logger.warning(f"ORB: get_orders({sym}) failed pre-close: {e}")
            return 0
        for o in open_orders:
            oid = str(getattr(o, 'id', '') or '')
            if not oid:
                continue
            try:
                # Prefer direct trading_client call — cancel_order_by_id is the
                # canonical path. Our AlpacaClient.cancel_order wrapper also works.
                self.alpaca.trading_client.cancel_order_by_id(oid)
                canceled += 1
            except Exception as e:
                logger.warning(f"ORB: cancel_order_by_id({oid}/{sym}) failed: {e}")
        return canceled

    def force_close_all(self) -> int:
        """15:45 ET: cancel pending ORB orders + market-close all ORB positions.

        Bracket legs (OCO SL + safety-net TP) hold shares as 'held_for_orders'
        on Alpaca. They MUST be canceled before close_position or Alpaca
        refuses the close. See _cancel_symbol_open_orders for the history.

        Returns:
            Number of positions closed.
        """
        closed = 0
        failed = []
        # Track which symbols got close orders submitted, for the post-FC
        # DB sync pass (recovers exit_price from Alpaca order history when
        # the async TradingStream watcher misses the sell-fill event).
        closed_symbols: List[str] = []
        # Bug-2 fix (post-code-review): track Phase-1/SWEEP close order_ids
        # by symbol so FC VERIFY can wait on OUR specific close rather than
        # any pending sell (which previously conflated with a partial-cancel
        # bracket leg). Maps sym → most-recent close order_id.
        phase1_close_orders: Dict[str, str] = {}
        # Bug-3 fix (post-code-review): defer per-phase alerts. Pre-fix,
        # each phase fired its own CRITICAL telegram on transient errors,
        # producing 2-3 alerts per FC even when VERIFY recovered cleanly.
        # Now: log warnings per phase, fire ONE summary alert at end iff
        # positions are STILL open after Phase1+SWEEP+VERIFY.
        helper_exhausted: List[str] = []
        sweep_close_failed: List[str] = []
        import time as _time

        # Phase 1a: cancel any unfilled pending buy-stop entry orders.
        # GLWG 2026-05-11 fix (Bug 5): query DB pending list ONCE and filter,
        # don't iterate-then-query-per-candidate. Pre-fix, every candidate
        # with `cand.plan_submitted=True` triggered a "cancelling unfilled
        # pending order for X" log line even when X had filled+exited hours
        # ago (plan_submitted stays True for the session). SONY/APT/MLTX
        # spammed the log at 19:45:01 on 2026-05-11 even though all three
        # had already exited via stop_loss / lock_stop.
        #
        # Bug-1 fix (post-code-review): on batch DB query failure, fall back
        # to per-candidate cancel using cand.order_id (the in-memory pending
        # order id). Pre-fix, a single DB hiccup at FC start would silently
        # skip cancelling ALL unfilled orders — they'd carry over to the
        # next day. SQLite contention can happen mid-day (CLAUDE.md).
        db_pending_now: Dict[str, List[Dict]] = {}
        batch_db_ok = False
        try:
            pending_today = self.db.get_open_trades(
                datetime.now(timezone.utc).date(), strategy=STRATEGY_NAME
            ) if hasattr(self.db, 'get_open_trades') else []
            for t in pending_today:
                if t.get('order_status') == 'pending_new':
                    db_pending_now.setdefault(t.get('symbol'), []).append(t)
            batch_db_ok = True
        except Exception as e:
            logger.error(
                f"ORB FC: batch pending-orders DB query failed: {e} — "
                f"falling back to per-candidate in-memory cancel "
                f"(cand.order_id is the source of truth in this path)"
            )
        for sym, cand in self.candidates.items():
            if not cand.plan_submitted or sym in self.open_positions:
                continue
            if batch_db_ok:
                # Normal path: trust the DB query
                actual_pending = db_pending_now.get(sym, [])
                if not actual_pending:
                    # plan_submitted sticks after fill+exit — suppress noise
                    continue
                logger.info(
                    f"ORB FORCE-CLOSE: cancelling {len(actual_pending)} "
                    f"unfilled pending order(s) for {sym}"
                )
                for t in actual_pending:
                    try:
                        self.alpaca.cancel_order(t['order_id'])
                    except Exception as e:
                        logger.warning(
                            f"ORB: cancel pending for {sym} failed: {e}"
                        )
            else:
                # Fallback: DB unavailable, use cand.order_id directly.
                # Cancelling a non-existent/already-terminal order is a
                # cheap no-op on Alpaca, so over-cancelling is safe.
                if not cand.order_id:
                    continue
                logger.info(
                    f"ORB FORCE-CLOSE (DB-fallback): cancelling order "
                    f"{cand.order_id[:8]} for {sym}"
                )
                try:
                    self.alpaca.cancel_order(cand.order_id)
                except Exception as e:
                    logger.warning(
                        f"ORB: cancel pending {sym} (DB-fallback) failed: {e}"
                    )

        # Phase 1b: market-close all ORB open positions. Cancel bracket legs
        # FIRST so close_position doesn't fail with held_for_orders. Use the
        # retry helper (Bug 1 fix) — pre-fix code used a hardcoded 0.8s sleep
        # then ONE retry at 2.0s; insufficient on busy paper accounts (GLWG
        # 2026-05-11: held_for_orders=1431 fired after 0.8s wait). Bug 3
        # fix: log "submitted" not "closed" — close_position returns the
        # order submission, NOT a fill confirmation.
        for sym, pos in list(self.open_positions.items()):
            try:
                n_legs = self._cancel_symbol_open_orders(sym)
                if n_legs > 0:
                    logger.info(
                        f"ORB FORCE-CLOSE: {sym} canceled {n_legs} bracket/safety legs"
                    )
                    # Bug-6 fix (post-code-review): brief pre-sleep gives
                    # Alpaca a head-start propagating the cancel-ACK so the
                    # retry helper's attempt 0 doesn't always burn on
                    # held_for_orders. Without this every FC produced at
                    # least one "race attempt 1/5" warning.
                    _time.sleep(self._FC_PHASE1B_PRE_CLOSE_SLEEP_S)
                result = self._close_position_with_held_qty_retry(sym)
                close_oid = (result or {}).get('id', '?')
                if result and result.get('id'):
                    phase1_close_orders[sym] = result['id']
                # Bug-8 fix (post-code-review): clearer log phrasing —
                # "fill confirmation deferred to post-Phase-1 verify"
                # accurately conveys that VERIFY runs LATER, not now.
                logger.info(
                    f"ORB FORCE-CLOSE: {sym} submitted close order="
                    f"{close_oid} — fill confirmation deferred to "
                    f"post-Phase-1 verify"
                )
                closed += 1
                closed_symbols.append(sym)
            except Exception as e:
                # Helper exhausted retries OR a non-retryable exception.
                # FC VERIFY gets the next bite — defer the alert to the
                # consolidated end-of-FC summary (Bug-3 fix). Log warning
                # so it's observable in journalctl.
                failed.append(sym)
                helper_exhausted.append(sym)
                logger.warning(
                    f"ORB FC Phase 1b: {sym} close_position helper "
                    f"exhausted retries (qty={pos.shares}, "
                    f"entry=${pos.entry_price:.2f}): {e} — "
                    f"will retry in SWEEP/VERIFY"
                )

        # 2026-04-29: Alpaca-driven orphan sweep. Engine state can drift from
        # Alpaca via crashes / cross-day persistence gaps (4/28 OPRA: BUY
        # filled during the 13:34/13:37 UTC crash window, fill never reached
        # DB, sync_positions classified as orphan, FC ignored it because the
        # iterator above is engine-state-driven). Belt-and-suspenders: after
        # the engine-state pass, query Alpaca for ORB's remaining positions
        # and close them.
        #
        # 2026-05-22: STRATEGY-SCOPED. ORB shares its Alpaca account with
        # bull flag (and other projects on some nodes). The sweep MUST only
        # close positions ORB owns (`orb_owned`) or it flattens another
        # strategy's position — a divergence CLF short was one force-close
        # away from being covered here.
        orb_owned = self._orb_owned_symbols()
        try:
            alp_positions = self.alpaca.get_open_positions() or []
        except Exception as e:
            self._notify_error(
                f"FC SWEEP: Alpaca query failed — cannot verify flat: {e}"
            )
            alp_positions = []
        for p in alp_positions:
            sym = (
                getattr(p, 'symbol', None)
                or (p.get('symbol') if isinstance(p, dict) else None)
            )
            if not sym:
                continue
            # Bug 2 fix (GLWG 2026-05-11): Phase 1 may have already submitted
            # a close order for this symbol that's still PENDING fill on
            # Alpaca. Pre-fix, SWEEP saw the pending position, declared it
            # "orphan from prior crash/drift", tried to close again, FAILED
            # with held_for_orders (held by Phase 1's pending close), and
            # fired CRITICAL "WILL leak overnight" alert — all false alarms.
            # FC VERIFY (next phase) is the right place to wait + retry.
            if sym in closed_symbols:
                logger.info(
                    f"ORB FC SWEEP: {sym} close submitted in Phase 1 — "
                    f"awaiting fill via VERIFY (not a SWEEP orphan)"
                )
                continue
            if sym not in orb_owned:
                logger.warning(
                    f"ORB FC SWEEP: {sym} is on the account but is NOT an "
                    f"ORB position (no open ORB trade row) — leaving it "
                    f"untouched. Shared account: {sym} belongs to another "
                    f"strategy/project."
                )
                continue
            logger.warning(
                f"ORB FC SWEEP: Alpaca position {sym} survived engine-state "
                f"close pass — orphan from prior crash/drift, closing now"
            )
            try:
                self._cancel_symbol_open_orders(sym)
                _time.sleep(self._FC_PHASE1B_PRE_CLOSE_SLEEP_S)
                result = self._close_position_with_held_qty_retry(sym)
                close_oid = (result or {}).get('id', '?')
                if result and result.get('id'):
                    phase1_close_orders[sym] = result['id']
                logger.warning(
                    f"ORB FC SWEEP: closed orphan {sym} (order={close_oid})"
                )
                self._notify(
                    f"{self.tg_prefix} ⚠️ FC SWEEP: closed orphan {sym} "
                    f"(not tracked in engine state)"
                )
                closed += 1
                closed_symbols.append(sym)
            except Exception as e:
                # Bug-3 fix (post-code-review): defer alert to end-of-FC
                # summary. VERIFY may still recover this orphan.
                failed.append(sym)
                sweep_close_failed.append(sym)
                logger.warning(
                    f"ORB FC SWEEP: {sym} orphan close failed: {e} — "
                    f"will retry in VERIFY"
                )

        # Post-FC verification: confirm Alpaca shows zero positions.
        # 2026-05-07 hardening (ASTX 5/6 post-mortem):
        #   - Verify with grace: poll Alpaca up to 10s, succeed as soon as
        #     position is gone. The previous 1s sleep was below typical
        #     Alpaca ACK latency and false-positive'd FC failures.
        #   - Retry on failure: re-close + re-verify up to 3 times with
        #     5s/10s/15s backoff. Total max ~45s, well within the 15-min
        #     FC window. Loud CRITICAL alert only if all 3 retries fail.
        #   - DB sync: on success, fetch each closed symbol's actual sell
        #     order from Alpaca and write exit_price/pnl to DB. Closes the
        #     async-watcher gap that left phantom-open rows for ASTX 5/6,
        #     NVOX 5/1, CRML 5/4.
        max_retries = self.fc_retry_count
        backoffs_s = self.fc_retry_backoffs_s
        verify_passed = False
        last_still_open: List = []
        for attempt in range(max_retries):
            still_open = self._verify_flat_with_grace(
                max_wait_s=self.fc_verify_max_wait_s,
                poll_interval_s=self.fc_verify_poll_interval_s,
                orb_owned=orb_owned,
            )
            last_still_open = still_open
            if not still_open:
                verify_passed = True
                if attempt > 0:
                    logger.info(
                        f"FC VERIFY: passed on retry {attempt+1}/{max_retries}"
                    )
                break
            still_syms = [
                getattr(p, 'symbol', None)
                or (p.get('symbol') if isinstance(p, dict) else '?')
                for p in still_open
            ]
            still_syms = [s for s in still_syms if s]
            logger.warning(
                f"FC VERIFY attempt {attempt+1}/{max_retries}: "
                f"{len(still_open)} position(s) STILL open: "
                f"{','.join(still_syms)} — re-closing"
            )
            # Bug 4 fix (GLWG 2026-05-11): on non-final attempts, check if
            # a Phase-1/SWEEP close order is still in flight. If yes, let
            # it work — don't cancel + duplicate. Pre-fix, FC VERIFY blindly
            # cancelled any open order and submitted a fresh close, which
            # killed Phase-1's d086167a 12s after submission (it may have
            # been seconds from filling) and submitted duplicate d315539c.
            # Final attempt (attempt == max_retries - 1): force cancel +
            # close to break a wedged state (prevents leaking past FC).
            #
            # Bug-2 fix (post-code-review): check OUR specific close
            # order_id, not "any pending sell". Prevents waiting on a
            # stranded bracket leg (e.g., partial cancel failure left
            # the SL leg alive — that's NOT a close-in-progress).
            is_final_attempt = (attempt == max_retries - 1)
            for sym in still_syms:
                try:
                    prior_close_id = phase1_close_orders.get(sym)
                    if (
                        not is_final_attempt
                        and prior_close_id
                        and self._is_close_order_still_pending(prior_close_id)
                    ):
                        logger.info(
                            f"FC VERIFY attempt {attempt+1}/{max_retries}: "
                            f"{sym} close {prior_close_id[:8]} still "
                            f"pending — letting it work; recheck after backoff"
                        )
                        continue
                    self._cancel_symbol_open_orders(sym)
                    _time.sleep(self._FC_PHASE1B_PRE_CLOSE_SLEEP_S)
                    result = self._close_position_with_held_qty_retry(sym)
                    if result and result.get('id'):
                        phase1_close_orders[sym] = result['id']
                    if sym not in closed_symbols:
                        closed_symbols.append(sym)
                except Exception as e:
                    logger.warning(
                        f"FC VERIFY retry {attempt+1}: close {sym} failed: {e}"
                    )
            # Backoff before next verify (except after the last attempt)
            if attempt < max_retries - 1:
                backoff = (backoffs_s[attempt] if attempt < len(backoffs_s)
                           else (backoffs_s[-1] if backoffs_s else 5))
                _time.sleep(backoff)

        # Bug-3 fix (post-code-review): consolidated end-of-FC alert.
        # Pre-fix each phase fired its own CRITICAL on transient errors,
        # producing 2-3 alerts per FC even when VERIFY recovered cleanly
        # (GLWG 2026-05-11: false-alarm "WILL leak overnight" fired while
        # the close was 30s from filling). Now: phases log warnings, this
        # is the ONE alert iff positions truly remain open after all
        # phases. Diagnostic context (helper_exhausted / sweep_close_failed)
        # is included so the operator can see which phase each failure
        # came from.
        if not verify_passed:
            still_syms = [
                getattr(p, 'symbol', None)
                or (p.get('symbol') if isinstance(p, dict) else '?')
                for p in last_still_open
            ]
            still_syms = [s for s in still_syms if s]
            self._notify_error(
                f"ORB FC FINAL FAILURE: {len(still_syms)} position(s) STILL "
                f"open after Phase1+SWEEP+VERIFY({max_retries} retries): "
                f"{','.join(still_syms)}. "
                f"helper_exhausted={helper_exhausted or 'none'}, "
                f"sweep_close_failed={sweep_close_failed or 'none'}. "
                f"MANUAL ACTION REQUIRED before market close."
            )
        elif helper_exhausted or sweep_close_failed:
            # Transient phase failures, all resolved by later phases —
            # log for observability but no operator alert needed.
            logger.info(
                f"ORB FC: VERIFY recovered transient failures — "
                f"helper_exhausted={helper_exhausted}, "
                f"sweep_close_failed={sweep_close_failed}"
            )

        # DB sync: even on partial success, sync the symbols we know we closed.
        # This is the path that recovers exit_price for the trades whose
        # async sell-fill events were missed by the TradingStream watcher.
        if closed_symbols:
            try:
                self._sync_db_after_fc(closed_symbols)
            except Exception as e:
                logger.warning(f"FC DB SYNC: failed (non-fatal): {e}")

        if self.notify_on_force_close and self.notifier and closed > 0:
            self._notify(
                f"{self.tg_prefix} FORCE-CLOSE at "
                f"{self.force_close_hour_et:02d}:{self.force_close_minute_et:02d} — "
                f"{closed} positions closed"
                + (f" / {len(failed)} FAILED ({','.join(failed)})" if failed else "")
            )

        # Daily summary — fires once per day at force-close. Telegram + log.
        if not self.daily_summary_sent:
            self.send_daily_report()
            self.daily_summary_sent = True

        return closed

    def send_daily_report(self) -> None:
        """Log + (optionally) telegram-send today's ORB telemetry.

        Key metrics:
          - Fill rate (filled / placed). BT expected ~73% overall, 75% median/day.
            Paper-phase divergence from this range is an early-warning signal.
          - Daily P&L
          - Time-stop cancellations (target ~27% of placements)
        """
        placed = self.daily_n_placed
        filled = self.daily_n_filled
        canceled = self.daily_n_time_stop_canceled
        fill_rate = (filled / placed) if placed else 0.0

        summary = (
            f"{self.tg_prefix} DAILY: "
            f"placed={placed} filled={filled} canceled={canceled} "
            f"fill_rate={fill_rate:.0%} P&L=${self.daily_pnl:+,.0f}"
        )
        logger.info(f"ORB DAILY REPORT: {summary}")

        if self.notifier and placed > 0:
            # Only Telegram if we actually placed orders — skip quiet days
            extra = ""
            if fill_rate < 0.50 and placed >= 3:
                extra = " ⚠️ LOW FILL RATE"
            elif fill_rate == 1.0 and placed >= 3:
                extra = " ✅ all filled"
            self._notify(summary + extra)

    # =====================================================================
    # Startup recovery
    # =====================================================================

    def sync_positions(self) -> None:
        """On startup: reconcile Alpaca state with DB for strategy='orb'.

        Restart-safe. Handles three states per DB trade:
          A. FILLED on Alpaca + in DB → rehydrate as open, call add_watch
          B. PENDING on Alpaca (stop-buy not yet triggered) + in DB as pending_new
             → rehydrate with order_id set → _process_pending_fills polls it
          C. DB open but Alpaca has no position + no pending order → stale; mark closed
        Does NOT touch bull flag / MACD wave positions (different AlpacaClient).
        """
        try:
            alpaca_positions = self.alpaca.get_open_positions()
        except Exception as e:
            logger.error(f"ORB.sync_positions: alpaca fetch failed: {e}")
            return

        # Fetch Alpaca pending orders (state B)
        alp_pending_orders = []
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            alp_pending_orders = self.alpaca.trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN)
            ) or []
        except Exception as e:
            logger.warning(f"ORB.sync_positions: get_orders(OPEN) failed: {e}")

        try:
            today_trades = self.db.get_open_trades(
                datetime.now(timezone.utc).date(), strategy=STRATEGY_NAME
            ) if hasattr(self.db, 'get_open_trades') else []
        except Exception as e:
            logger.error(f"ORB.sync_positions: db query failed: {e}")
            today_trades = []

        # Build lookups
        alp_pos_by_sym: Dict[str, dict] = {}
        for p in alpaca_positions:
            sym = getattr(p, 'symbol', None) or (p.get('symbol') if isinstance(p, dict) else None)
            if sym:
                alp_pos_by_sym[sym] = p
        alp_pending_by_id: Dict[str, object] = {}
        for o in alp_pending_orders:
            oid = str(getattr(o, 'id', '') or (o.get('id', '') if isinstance(o, dict) else ''))
            if oid:
                alp_pending_by_id[oid] = o

        # Match DB open trades against Alpaca state
        recovered = 0
        recovered_pending = 0
        for t in today_trades:
            sym = t.get('symbol')
            order_id = str(t.get('order_id') or '')
            db_status = t.get('order_status') or ''
            alp_pos = alp_pos_by_sym.get(sym)

            # Race-A fix (2026-06-06): an exit_pending_verification row was
            # emitted by StopMonitor.BRANCH_LAST_RESORT — the orphan_reconciler
            # owns it from this point. Skip the State A/B/C rehydrate paths;
            # re-creating a StopMonitor watch with a stale hard_stop would
            # bypass the reconciler (which would then skip the symbol because
            # it's back in open_positions).
            if db_status == 'exit_pending_verification':
                continue

            # State B: order still pending on Alpaca + DB shows pending
            if alp_pos is None and order_id and order_id in alp_pending_by_id:
                # Parse pattern_data for lock params
                import json as _json
                pdata = t.get('pattern_data') or {}
                if isinstance(pdata, str):
                    try:
                        pdata = _json.loads(pdata)
                    except (ValueError, TypeError):
                        pdata = {}
                # Recover ORIGINAL submit time from DB so the time-stop cancel
                # clock doesn't get reset by restart. Prefer the explicit
                # order_submitted_at column (written at submit); fall back to
                # created_at (always present) for legacy trades that predate
                # the explicit column.
                submit_ts = (
                    t.get('order_submitted_at')
                    or t.get('created_at')
                    or datetime.now(timezone.utc)
                )
                # DB may return naive strings — normalize to tz-aware UTC
                if isinstance(submit_ts, str):
                    try:
                        from dateutil.parser import isoparse as _iso
                        submit_ts = _iso(submit_ts)
                    except Exception:
                        submit_ts = datetime.now(timezone.utc)
                if submit_ts.tzinfo is None:
                    submit_ts = submit_ts.replace(tzinfo=timezone.utc)
                # Recover slippage-attribution fields persisted at submit
                bar_close = t.get('bar_close_price')
                entry_ask = t.get('entry_quote_ask')
                pos = OpenPosition(
                    symbol=sym,
                    entry_price=float(t.get('entry_price') or 0.0),
                    stop_price=float(t.get('stop_loss_price') or 0.0),
                    shares=int(t.get('shares') or 0),
                    trade_id=int(t['id']),
                    order_id=order_id,  # CRITICAL: populated = pending; _process_pending_fills will poll
                    entry_time=submit_ts,
                    range_high=float(pdata.get('range_high', 0.0)),
                    range_low=float(pdata.get('range_low', 0.0)),
                    lock_arm_at_r=float(pdata.get('lock_arm_at_r', 1.5)),
                    lock_stop_r=float(pdata.get('lock_stop_r', 1.0)),
                    composite_score=float(pdata.get('composite_score', 0.0)),
                    quintile=str(pdata.get('quintile', 'Q3')),
                    bar_close_price=(float(bar_close) if bar_close else float(pdata.get('range_high', 0.0)) or None),
                    order_submitted_at=submit_ts,
                    entry_quote_ask=(float(entry_ask) if entry_ask else None),
                )
                self.open_positions[sym] = pos
                # 2026-05-08 fix: ALWAYS register the recovered pending order
                # on self.candidates, creating a stub CandidateState if needed.
                # Pre-fix, the conditional `if sym in self.candidates` made
                # restart-recovered pending orders invisible to
                # _cancel_stale_pending_orders (which iterates self.candidates),
                # so the 60-min time_stop never fired post-restart.
                # CORD 5/7 was the symptom: my 13:53 ET restart killed the
                # old process before time_stop's 10:35 deadline; the new
                # process recovered CORD into open_positions but NOT into
                # candidates → CORD's buy-stop stayed live until 11:22 ET,
                # 47 min past the cancel deadline.
                if sym not in self.candidates:
                    self.candidates[sym] = CandidateState(symbol=sym)
                    self.candidates[sym].plan_submitted = True
                self.candidates[sym].order_id = order_id
                self.candidates[sym].order_submitted_at = submit_ts
                recovered_pending += 1
                logger.info(
                    f"ORB.sync: {sym} rehydrated as PENDING "
                    f"(order {order_id}, submitted {submit_ts.isoformat()}, "
                    f"age {(datetime.now(timezone.utc) - submit_ts).total_seconds()/60:.1f}min, "
                    f"will poll for fill)"
                )
                continue

            # State C: DB open but nothing on Alpaca
            if alp_pos is None:
                logger.warning(f"ORB.sync: {sym} DB open but not on Alpaca — marking stale")
                try:
                    self.db.update_trade(t['id'], {'order_status': 'stale_closed'})
                except Exception:
                    pass
                continue
            # Parse pattern_data (stored as JSON string in DB)
            import json as _json
            pdata = t.get('pattern_data') or {}
            if isinstance(pdata, str):
                try:
                    pdata = _json.loads(pdata)
                except (ValueError, TypeError):
                    pdata = {}
            if not isinstance(pdata, dict):
                pdata = {}
            # Rehydrate
            pos = OpenPosition(
                symbol=sym,
                entry_price=float(t.get('fill_price') or t.get('entry_price') or 0.0),
                stop_price=float(t.get('stop_loss_price') or 0.0),
                shares=int(t.get('shares') or 0),
                trade_id=int(t['id']),
                order_id=str(t.get('order_id') or ''),
                entry_time=t.get('filled_at') or datetime.now(timezone.utc),
                range_high=float(pdata.get('range_high', 0.0)),
                range_low=float(pdata.get('range_low', 0.0)),
                lock_arm_at_r=float(pdata.get('lock_arm_at_r', 1.5)),
                lock_stop_r=float(pdata.get('lock_stop_r', 1.0)),
                composite_score=float(pdata.get('composite_score', 0.0)),
                quintile=str(pdata.get('quintile', 'Q3')),
            )
            self.open_positions[sym] = pos
            # Re-register StopMonitor watch
            if self.stop_monitor is not None:
                try:
                    risk_per_share = max(pos.entry_price - pos.stop_price, 0.0)
                    self.stop_monitor.add_watch(
                        symbol=sym,
                        stop_price=pos.stop_price,
                        shares=pos.shares,
                        tp_leg_id='',
                        sl_leg_id='',
                        trade_db_id=pos.trade_id,
                        entry_price=pos.entry_price,
                        risk_per_share=risk_per_share,
                        strategy=STRATEGY_NAME,
                        lock_arm_at_r=pos.lock_arm_at_r,
                        lock_stop_r=pos.lock_stop_r,
                    )
                except Exception as e:
                    logger.error(f"ORB.sync: re-watch {sym} failed: {e}")
            recovered += 1

        # Orphan detection: Alpaca has positions we don't track.
        # Cause (seen 2026-04-20→21): force-close failed → position carried
        # overnight → today's sync_positions queries get_open_trades(today)
        # which returns 0 rows → ORB engine has no tracker for a live
        # Alpaca position. Without this alert the position drifts unmanaged.
        orphans = [
            sym for sym in alp_pos_by_sym
            if sym not in self.open_positions
        ]
        if orphans:
            # Summary log (no Telegram) — the reconciler's per-orphan
            # alerts below ARE rate-limited (alert_cooldown_minutes).
            # Pre-2026-06-05 we Telegram'd this summary unconditionally,
            # which produced a duplicate alert per sync cycle for any
            # long-lived orphan (e.g., 10 days × 1 alert/min = ~14k spam).
            details = []
            for sym in orphans:
                p = alp_pos_by_sym[sym]
                try:
                    qty = int(getattr(p, 'qty', 0) or (p.get('qty', 0) if isinstance(p, dict) else 0))
                    avg = float(getattr(p, 'avg_entry_price', 0) or
                                (p.get('avg_entry_price', 0) if isinstance(p, dict) else 0))
                    upl = float(getattr(p, 'unrealized_pl', 0) or
                                (p.get('unrealized_pl', 0) if isinstance(p, dict) else 0))
                    details.append(f"{sym} qty={qty} avg=${avg:.2f} upl=${upl:+.0f}")
                except Exception:
                    details.append(f"{sym} (parse-failed)")
            logger.error(
                f"ORB: orphan(s) detected — {'; '.join(details)} — "
                f"deferring to reconciler for classification + action"
            )

            # 2026-06-05: orphan reconciler replaces the old in-engine
            # ownership check (_orb_owned_symbols → get_open_trades), which
            # excluded the exact rows we need to reconcile (rows where
            # exit_reason='stop_loss_unconfirmed' had a fake exit_price
            # written). That deadlock hid QBTZ for 4 days. The hardened
            # multi-signal predicate (strategy + stale + avg-entry + qty)
            # is safe to run any time of day — same-day fresh entries are
            # excluded by the cross-day stale-signal check.
            try:
                cfg = getattr(self, 'orphan_reconciler_cfg', None) or ReconcilerConfig()
                # alp_pos_by_sym was built at the top of sync_positions —
                # pass it in instead of re-hitting Alpaca.
                broker_snapshot = []
                for p in alp_pos_by_sym.values():
                    try:
                        broker_snapshot.append({
                            'symbol': getattr(p, 'symbol', None) or p.get('symbol'),
                            'qty': int(getattr(p, 'qty', 0) or
                                       (p.get('qty', 0) if isinstance(p, dict) else 0)),
                            'avg_entry_price': float(
                                getattr(p, 'avg_entry_price', 0) or
                                (p.get('avg_entry_price', 0) if isinstance(p, dict) else 0)
                            ),
                            'unrealized_pl': float(
                                getattr(p, 'unrealized_pl', 0) or
                                (p.get('unrealized_pl', 0) if isinstance(p, dict) else 0)
                            ),
                        })
                    except Exception:
                        continue
                reconcile_strategy_orphans(
                    strategy=STRATEGY_NAME, alpaca=self.alpaca, db=self.db,
                    notifier=self.notifier,
                    tracked_symbols=set(self.open_positions.keys()),
                    cfg=cfg,
                    broker_positions=broker_snapshot,
                )
            except Exception as e:
                logger.error(
                    f"ORB: orphan reconciler raised: {e} — sync continues"
                )

        logger.info(
            f"ORB.sync_positions: {recovered} filled + {recovered_pending} pending rehydrated; "
            f"{len(self.open_positions)} tracked; "
            f"{len(alp_pos_by_sym)} Alpaca positions, "
            f"{len(alp_pending_by_id)} Alpaca pending orders"
            + (f"; {len(orphans)} ORPHAN(s): {','.join(orphans)}" if orphans else "")
        )

    def reset_daily(self) -> None:
        """Clear day-scoped state. Called at day boundary BEFORE building the
        next day's universe. DOES NOT touch open_positions — those persist
        across day boundary if somehow not closed at 15:45.
        """
        self.candidates.clear()
        self.universe.clear()
        self.universe_date = None
        self._bar_windows.clear()
        self._feature_context_cache = {}  # re-fetched next day with T-1 = today
        self.daily_pnl = 0.0
        self.daily_loss_limit_logged = False
        self.daily_n_placed = 0
        self.daily_n_filled = 0
        self.daily_n_time_stop_canceled = 0
        self.daily_summary_sent = False
        self._late_entry_cutoff_logged = False
        self._post_open_range_sweep_done = False
        self._sweep_retry_used_today = False
        self._pdr_vetoed_today = set()
        # Drain any leftover bar events from yesterday
        try:
            while True:
                self._bar_event_queue.get_nowait()
        except queue.Empty:
            pass
        logger.info("ORBEngine.reset_daily: day state cleared")

    # =====================================================================
    # Helpers
    # =====================================================================

    def _daily_loss_limit_hit(self) -> bool:
        if self.daily_pnl <= self.daily_loss_limit_usd:
            if not self.daily_loss_limit_logged:
                logger.warning(
                    f"ORB: daily loss limit hit (pnl=${self.daily_pnl:+,.2f} "
                    f"<= limit=${self.daily_loss_limit_usd:+,.0f}) — blocking new entries"
                )
                if self.notify_on_daily_loss_limit and self.notifier:
                    self._notify(
                        f"{self.tg_prefix} DAILY LOSS LIMIT HIT "
                        f"(${self.daily_pnl:+,.2f}) — no new entries today"
                    )
                self.daily_loss_limit_logged = True
            return True
        return False

    def _symbols_entered_today_db(self) -> Set[str]:
        """Return the set of symbols that have ANY ORB trade today in DB —
        regardless of current status (pending, filled, closed, canceled, etc).

        Used by check_entries to enforce BT-parity daily caps:
          * max_concurrent entries total per day
          * 1 entry per symbol per day (no re-entry after stop-out)

        DB-backed so it survives restarts (in-memory CandidateState.plan_submitted
        and daily_n_placed counter are lost on every service restart — the
        source of the 2026-04-20 re-entry + new-entry-late bugs).
        """
        try:
            today = datetime.now(timezone.utc).date().isoformat()
            if hasattr(self.db, 'get_trades_by_date'):
                rows = self.db.get_trades_by_date(today) or []
                return {
                    t.get('symbol') for t in rows
                    if t.get('strategy') == STRATEGY_NAME and t.get('symbol')
                }
            return set()
        except Exception as e:
            logger.warning(f"ORB: _symbols_entered_today_db failed: {e}")
            return set()

    def _symbol_has_any_open_trade(self, symbol: str) -> bool:
        """FCFS cross-strategy check — if any strategy has symbol open today, skip."""
        try:
            today = datetime.now(timezone.utc).date()
            if not hasattr(self.db, 'get_open_trades'):
                return False
            # Check across all strategies (no strategy filter)
            all_open = self.db.get_open_trades(today)
            return any(t.get('symbol') == symbol for t in all_open)
        except Exception as e:
            logger.warning(f"ORB: FCFS check for {symbol} failed: {e}")
            return False

    @staticmethod
    def _position_symbol(p) -> Optional[str]:
        """Extract the ticker from an Alpaca position object or dict."""
        return (
            getattr(p, 'symbol', None)
            or (p.get('symbol') if isinstance(p, dict) else None)
        )

    def _orb_owned_symbols(self, lookback_days: int = 0) -> Set[str]:
        """Symbols ORB currently has an OPEN trade row for (strategy='orb').

        The authoritative "positions ORB owns" set. ORB shares its Alpaca
        account with bull flag (and, on some nodes, other projects), so
        every position-closing action — the force-close sweep, post-FC
        verify, and orphan auto-close — MUST scope to this set or it will
        flatten another strategy's position (2026-05-22: a divergence CLF
        short was one force-close away from being covered by ORB's sweep).

        Only OPEN ORB rows count (exit_price NULL, active order_status) —
        a closed row does NOT, because the symbol may since have been
        re-opened by another strategy on the shared account.

        Args:
            lookback_days: also scan open ORB rows from this many prior
                calendar days (0 = today only). Covers a position carried
                past a day boundary by a failed force-close.

        Returns:
            Set of symbols. Empty on DB error — the safe default: every
            caller treats "not ORB-owned" as "do not close".
        """
        if not hasattr(self.db, 'get_open_trades'):
            return set()
        owned: Set[str] = set()
        try:
            today = datetime.now(timezone.utc).date()
            for d in range(lookback_days + 1):
                day = (today - timedelta(days=d)).isoformat()
                for t in (self.db.get_open_trades(day, strategy=STRATEGY_NAME) or []):
                    sym = t.get('symbol')
                    if sym:
                        owned.add(sym)
        except Exception as e:
            logger.error(
                f"ORB: _orb_owned_symbols DB query failed: {e} — treating "
                f"ALL Alpaca positions as non-ORB (close nothing). Manual "
                f"review needed."
            )
            return set()
        return owned

    def _has_buying_power(self, required_usd: float) -> bool:
        try:
            info = self.alpaca.get_account_info() if hasattr(self.alpaca, 'get_account_info') else None
            if not info:
                return True  # optimistic if can't query
            bp = float(info.get('buying_power', 0.0))
            return bp >= required_usd
        except Exception as e:
            logger.warning(f"ORB: buying power check failed: {e}")
            return True

    def _get_spread_bps(self, symbol: str) -> Optional[float]:
        """Fetch current spread in bps via latest quote. None if unavailable."""
        try:
            if not hasattr(self.alpaca, 'get_latest_quote'):
                return None
            q = self.alpaca.get_latest_quote(symbol)
            bid = float(q.get('bid_price', 0.0))
            ask = float(q.get('ask_price', 0.0))
            if bid <= 0 or ask <= 0 or ask < bid:
                return None
            mid = (bid + ask) / 2.0
            return (ask - bid) / mid * 10000.0
        except Exception:
            return None

    def _save_pending_trade(
        self,
        plan: OrbTradePlan,
        order_id: str,
        submit_time: Optional[datetime] = None,
        submit_bid: float = 0.0,
        submit_ask: float = 0.0,
        submit_bid_size: int = 0,
        submit_ask_size: int = 0,
    ) -> Optional[int]:
        """Insert pending_new trade record with strategy='orb'.

        Persists slippage-attribution fields at submit time so that:
          1. `_confirm_fill` can compute drift_bar_to_fill_bps + submit_to_fill_ms
             correctly even after a restart wipes the in-memory OpenPosition.
          2. `sync_positions` rehydration can restore the submit-time reference
             fields onto the OpenPosition (previously lost on restart).
          3. Time-stop cancel uses the ORIGINAL submit_time from DB, not the
             post-restart rehydration time.
        """
        import json as _json
        try:
            # Sizing attribution (2026-07-10): persist the PM$/news inputs
            # + final pm_mult so the EoD green-check can recompute the
            # expected mult and flag drift (scripts/daily_green_check.py).
            _pm_dv = self._pm_dollar_vols.get(plan.symbol)
            _news = self._news_flags.get(plan.symbol)
            pattern_data = _json.dumps({
                'range_high': plan.range_high,
                'range_low': plan.range_low,
                'range_size': plan.range_size,
                'composite_score': plan.composite_score,
                'quintile': plan.quintile,
                'adaptive_mult': plan.adaptive_mult,
                'lock_arm_at_r': plan.lock_arm_at_r,
                'lock_stop_r': plan.lock_stop_r,
                'pm_dollar_vol': _pm_dv,
                'has_news': (None if _news is None
                             else _news.get('n_articles', 0) > 0),
                'n_articles': (None if _news is None
                               else _news.get('n_articles', 0)),
                'news_headline': ('' if _news is None
                                  else _news.get('headline', '')),
                'asset_class': self._asset_class.get(plan.symbol),
                'anchor': self._anchor_for(plan.symbol, allow_api=False),
                'anchor_cohort': (anchor_cohort_counts(
                    self._anchor_for(s, allow_api=False)
                    for s in self.candidates.keys())
                    .get(self._anchor_cache.get(plan.symbol) or '', 0)),
                # the mult that ACTUALLY sized this plan (carried on the
                # plan — never recomputed here, so recorded == applied)
                'pm_mult': plan.pm_mult,
            })
            now_utc = submit_time or datetime.now(timezone.utc)
            record = {
                'trade_date': now_utc.date().isoformat(),
                'symbol': plan.symbol,
                'side': 'buy',
                'entry_price': plan.entry_price,
                'stop_loss_price': plan.stop_price,
                'take_profit_price': 0,  # ORB has no fixed target
                'shares': plan.shares,
                'risk_per_share': plan.risk_per_share,
                'total_risk': plan.total_risk,
                'risk_reward_ratio': 0,
                'order_id': order_id,
                'order_status': 'pending_new',
                'fill_price': None,
                'filled_at': None,
                'exit_price': None,
                'exit_reason': None,
                'exited_at': None,
                'pnl': None,
                'pnl_pct': None,
                'pattern_data': pattern_data,
                'strategy': STRATEGY_NAME,
                # Slippage attribution baseline (set ONCE at submit; consumed at fill)
                'order_submitted_at': now_utc,
                'bar_close_price': plan.range_high,  # BT reference = range_high
                'entry_quote_bid': submit_bid if submit_bid > 0 else None,
                'entry_quote_ask': submit_ask if submit_ask > 0 else None,
                'entry_quote_bid_size': submit_bid_size if submit_bid_size > 0 else None,
                'entry_quote_ask_size': submit_ask_size if submit_ask_size > 0 else None,
                'entry_quote_spread': (
                    (submit_ask - submit_bid)
                    if submit_bid > 0 and submit_ask > 0 else None
                ),
            }
            trade_id = self.db.save_trade(record)
            return int(trade_id) if trade_id else None
        except Exception as e:
            logger.error(f"ORB: save_trade failed for {plan.symbol}: {e}")
            return None

    def _handle_reject(self, symbol: str, reject: PlannerReject) -> None:
        if reject.reason == 'spread_gate' and self.notify_on_spread_skip and self.notifier:
            self._notify(
                f"{self.tg_prefix} {symbol} spread skip — "
                f"{reject.details.get('spread_bps', 0):.0f}bps > {self.max_spread_bps:.0f}bps"
            )
        logger.info(f"ORB: {symbol} rejected ({reject.reason}) — {reject.details}")

    def _verify_flat_with_grace(
        self, max_wait_s: int = 10, poll_interval_s: float = 1.0,
        orb_owned: Optional[Set[str]] = None,
    ) -> List:
        """Poll Alpaca for open positions until flat or timeout.

        Replaces the previous single-shot post-close check that ran 1s after
        submission — too fast for typical Alpaca ACK latency, generating
        false-positive FC failures (ASTX 5/6 post-mortem).

        Returns the list of remaining positions (empty list = flat). Polling
        stops as soon as Alpaca reports zero. Returns empty on transient
        query errors so the caller can decide whether to retry.

        Args:
            orb_owned: when provided, positions whose symbol is NOT in this
                set are ignored — they belong to another strategy on a
                shared account, are not ORB's to close, and must not trip
                an FC-failure alert or a VERIFY re-close.
        """
        import time as _time
        deadline = _time.time() + max_wait_s
        last = []
        while _time.time() < deadline:
            try:
                last = self.alpaca.get_open_positions() or []
            except Exception as e:
                logger.warning(f"FC VERIFY poll: Alpaca query failed: {e}")
                _time.sleep(poll_interval_s)
                continue
            if orb_owned is not None:
                last = [
                    p for p in last
                    if self._position_symbol(p) in orb_owned
                ]
            if not last:
                return []
            _time.sleep(poll_interval_s)
        return last

    def _sync_db_after_fc(self, symbols: List[str]) -> None:
        """For each symbol whose FC close order was submitted, fetch the
        actual sell-fill from Alpaca and update DB exit_price/pnl.

        Closes the gap where the async TradingStream watcher misses sell-fill
        events, leaving phantom-open DB rows (ASTX 5/6, NVOX 5/1, CRML 5/4).
        Idempotent: rows that already have exit_price are skipped.
        """
        if not symbols:
            return
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
        except Exception as e:
            logger.warning(f"FC DB SYNC: alpaca import failed: {e}")
            return

        today = datetime.now(timezone.utc).date()
        try:
            db_open = self.db.get_open_trades(today, strategy=STRATEGY_NAME) or []
        except Exception as e:
            logger.warning(f"FC DB SYNC: db query failed: {e}")
            return
        # Index by symbol — one open trade per symbol per day in normal flow.
        db_by_sym: Dict[str, Dict] = {}
        for t in db_open:
            sym = t.get('symbol')
            if not sym:
                continue
            if t.get('fill_price') in (None, 0, 0.0):
                continue  # not actually filled
            if t.get('exit_price') is not None:
                continue  # already updated by stream watcher
            db_by_sym[sym] = t

        for sym in symbols:
            t = db_by_sym.get(sym)
            if t is None:
                continue
            try:
                orders = self.alpaca.trading_client.get_orders(
                    GetOrdersRequest(
                        status=QueryOrderStatus.CLOSED,
                        symbols=[sym], limit=10,
                    )
                ) or []
            except Exception as e:
                logger.warning(f"FC DB SYNC {sym}: get_orders failed: {e}")
                continue
            sell = None
            for o in orders:
                if o.side.value != 'sell':
                    continue
                if o.status.value != 'filled':
                    continue
                if not o.filled_avg_price:
                    continue
                # Most recent first
                if sell is None or (o.filled_at and sell.filled_at
                                     and o.filled_at > sell.filled_at):
                    sell = o
            if sell is None:
                logger.warning(
                    f"FC DB SYNC {sym}: no FILLED sell order found at Alpaca"
                )
                continue
            entry_price = float(t.get('fill_price') or 0)
            shares = int(t.get('filled_qty') or t.get('shares') or 0)
            if entry_price <= 0 or shares <= 0:
                continue
            exit_price = float(sell.filled_avg_price)
            pnl = (exit_price - entry_price) * shares
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            try:
                self.db.update_trade(t['id'], {
                    'exit_price': exit_price,
                    'exit_reason': ExitReason.FORCE_CLOSE.value,
                    'exited_at': sell.filled_at or datetime.now(timezone.utc),
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                })
                logger.info(
                    f"FC DB SYNC: {sym} updated exit=${exit_price:.2f} "
                    f"pnl=${pnl:+,.0f} (order={str(sell.id)[:8]})"
                )
            except Exception as e:
                logger.warning(f"FC DB SYNC {sym}: db update failed: {e}")

    def _notify(self, msg: str) -> None:
        """Send Telegram message (never raises — notifier failure shouldn't break trading)."""
        if not self.notifier:
            return
        try:
            # Support both sync and async notifiers
            send = getattr(self.notifier, 'send_message', None)
            if send is None:
                return
            import asyncio as _asyncio
            result = send(msg)
            if _asyncio.iscoroutine(result):
                # Fire and forget
                try:
                    loop = _asyncio.get_event_loop()
                    loop.run_until_complete(result)
                except RuntimeError:
                    _asyncio.run(result)
        except Exception as e:
            logger.debug(f"ORB: notifier failed (non-critical): {e}")

    def _notify_error(self, msg: str, exc: Optional[Exception] = None) -> None:
        """Log + Telegram critical error.

        Use for state-drift / unrecoverable-position / failed-exit scenarios
        where a human may need to intervene manually. Lower-stakes errors
        should use `logger.error` or `logger.warning` directly without
        telegram noise.

        Policy (from 2026-04-21 post-mortem on the ANNA force-close failure
        that leaked overnight): anything that could leave Alpaca and DB in
        an inconsistent state, or leave a position unmanaged, MUST alert
        via telegram so we catch it before market close.
        """
        full = f"{self.tg_prefix} ❌ {msg}"
        if exc is not None:
            full += f" — {type(exc).__name__}: {exc}"
        logger.error(f"ORB CRITICAL: {msg}" + (f" ({exc})" if exc else ""))
        self._notify(full)

    def is_force_close_time(self, now_utc: Optional[datetime] = None) -> bool:
        """True if current ET time is past the force-close threshold (15:45 ET default).

        Uses zoneinfo for proper DST (Mar 2nd-Sunday / Nov 1st-Sunday transitions).
        Falls back to month-based approximation on systems without zoneinfo data.
        """
        now_utc = now_utc or datetime.now(timezone.utc)
        fc = dtime(hour=self.force_close_hour_et, minute=self.force_close_minute_et)
        try:
            from zoneinfo import ZoneInfo
            et = now_utc.astimezone(ZoneInfo('America/New_York'))
            return et.time() >= fc
        except Exception:
            # Fallback: month-based offset (close enough on non-transition weeks)
            et_offset_hours = _et_offset_hours(now_utc)
            et_naive = now_utc - timedelta(hours=et_offset_hours)
            return et_naive.time() >= fc

    def _past_last_entry_time(self, now_utc: Optional[datetime] = None) -> bool:
        """True if current ET time is >= last_entry_submit_time_et.

        BT parity hard-cutoff for NEW order submissions. Paired with
        time_stop_minutes for existing-order cancellation. See check_entries.
        """
        now_utc = now_utc or datetime.now(timezone.utc)
        le = dtime(hour=self.last_entry_hour_et, minute=self.last_entry_minute_et)
        try:
            from zoneinfo import ZoneInfo
            et = now_utc.astimezone(ZoneInfo('America/New_York'))
            return et.time() >= le
        except Exception:
            et_offset_hours = _et_offset_hours(now_utc)
            et_naive = now_utc - timedelta(hours=et_offset_hours)
            return et_naive.time() >= le


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _first_session_open_ts_utc(bars_df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Return the timestamp of the first regular-session bar (9:30 ET).

    Resilient to DataFrames where the 'timestamp' column is object-dtype
    (e.g. when bars arrive from the StopMonitor WebSocket with raw python
    datetime objects, which happens in the scanner's drain_bar_events path).
    Coerce via pd.to_datetime before using the .dt accessor.

    2026-07-03 fix: the 9:30 ET bar is 13:30Z during EDT and 14:30Z during
    EST — ONE of those per date, never both. The previous
    `hour.isin([13, 14])` tolerance accepted a 14:30Z bar (10:30 ET!) as the
    session open during EDT whenever the window was missing the true 13:30Z
    bar (Alpaca WS delivers from subscribe-time forward, so post-9:31
    subscriptions never see it). That false anchor shifted the breakout-bar
    search one hour late and mis-keyed touchgo (KOLD/PLTU/TSDD tag_bb at
    14:36:2xZ). Derive the single correct hour from the bars' own date via
    _et_offset_hours.
    """
    if bars_df is None or len(bars_df) == 0:
        return None
    if 'timestamp' not in bars_df.columns:
        return None
    ts_col = bars_df['timestamp']
    try:
        if not pd.api.types.is_datetime64_any_dtype(ts_col):
            ts_col = pd.to_datetime(ts_col, utc=True, errors='coerce')
        first_valid = ts_col.dropna()
        if len(first_valid) == 0:
            return None
        # 9:30 ET in UTC for THIS date: 13:30Z (EDT) / 14:30Z (EST).
        open_hour_utc = 9 + _et_offset_hours(first_valid.iloc[0].to_pydatetime())
        mask = (ts_col.dt.minute == 30) & (ts_col.dt.hour == open_hour_utc)
    except (AttributeError, TypeError, ValueError) as e:
        logger.debug(f"ORB: _first_session_open_ts_utc timestamp parse failed: {e}")
        return None
    if not mask.any():
        return None
    return ts_col.loc[mask].iloc[0]


def _et_offset_hours(now_utc: datetime) -> int:
    """ET offset from UTC for the GIVEN instant — DST-transition-accurate.

    2026-07-04 review fix: the previous month-granularity approximation
    (Mar-Oct = EDT) was wrong for ~2 weeks/year — Mar 1 to the 2nd Sunday
    (still EST) and Nov 1 to the 1st Sunday (still EDT). Callers use this
    to locate the 9:30 ET session-open bar; a wrong offset made the mask
    match NOTHING, so ranges never completed and ORB silently placed zero
    entries on those days. ZoneInfo is authoritative; the month heuristic
    survives only as the fallback if tzdata is unavailable.
    """
    try:
        from zoneinfo import ZoneInfo
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        off = now_utc.astimezone(ZoneInfo('America/New_York')).utcoffset()
        return int(-off.total_seconds() // 3600)
    except Exception as e:
        logger.warning(
            f"ORB: _et_offset_hours ZoneInfo failed ({e}) — falling back to "
            f"month-granularity DST approximation (wrong near transitions)")
        m = now_utc.month
        if 3 <= m <= 10:
            return ET_OFFSET_EDT_HOURS
        return ET_OFFSET_EST_HOURS
