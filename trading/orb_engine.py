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
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

from trading.orb_correlation import dedup_candidates, symbol_family, symbol_super_group
from trading.orb_filter import (
    FeatureParam, assign_quintile, composite_score, load_feature_params,
)
from trading.orb_conviction import apply_adaptive_mult, load_adaptive_mults
from trading.orb_planner import OrbTradePlan, OrbTradePlanner, PlannerReject


logger = logging.getLogger(__name__)

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

        self.range_minutes = int(entry_cfg.get('range_minutes', 5))
        self.entry_slip_bps = float(entry_cfg.get('entry_slip_bps', 30))
        self.time_stop_minutes = int(entry_cfg.get('time_stop_minutes', 60))
        self.max_spread_bps = float(entry_cfg.get('max_spread_bps', 150))

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

        tg_prefix = notifications_cfg.get('prefix', '[ORB]')
        self.tg_prefix = tg_prefix

        logger.info(
            f"ORBEngine init: enabled={self.enabled}, dry_run={self.dry_run}, "
            f"max_concurrent={self.max_concurrent}, "
            f"risk_per_trade=${self.planner.risk_per_trade_usd:,.0f}, "
            f"per_pos_cap=${self.planner.per_pos_cap_usd:,.0f}, "
            f"filter_features={list(self.z_params.keys())}"
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
        for sym in missing:
            bars = bars_by_sym.get(sym)
            if bars is None or bars.empty:
                continue
            try:
                bars = bars.copy()
                bars['timestamp'] = pd.to_datetime(bars['timestamp'], utc=True)
                bars = bars.sort_values('timestamp').reset_index(drop=True)
                self._ingest_bars(sym, bars)
                if self.candidates.get(sym) and self.candidates[sym].range_data is not None:
                    filled_syms.add(sym)
            except Exception as e:
                logger.warning(f"ORB: post-open sweep ingest({sym}) failed: {e}")

        logger.info(
            f"ORB: post-open range sweep — filled {len(filled_syms)}/{len(missing)} ranges"
        )
        # Mark done even if some failed — next tick's WS bars will handle
        # any stragglers through the normal flow.
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
            start_d = (datetime.now(timezone.utc) - timedelta(days=40)).date()
            end_d = (datetime.now(timezone.utc) - timedelta(days=1)).date()
            if hasattr(self.db, 'get_daily_bars_cached'):
                try:
                    bulk = self.db.get_daily_bars_cached(
                        [symbol], str(start_d), str(end_d))
                    daily_bars = bulk.get(symbol) if isinstance(bulk, dict) else None
                except Exception as e:
                    logger.debug(f"ORB: db.get_daily_bars_cached({symbol}) failed: {e}")
            if not daily_bars and hasattr(self.alpaca, 'get_daily_bars_range'):
                try:
                    bulk = self.alpaca.get_daily_bars_range(
                        [symbol], start_d, end_d)
                    daily_bars = bulk.get(symbol) if isinstance(bulk, dict) else None
                except Exception as e:
                    logger.debug(f"ORB: alpaca.get_daily_bars_range({symbol}) failed: {e}")

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
        if len(symbols_entered_today) >= self.max_concurrent:
            return []  # daily cap exhausted (restart-safe via DB)

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
            score = composite_score(feats, self.z_params)
            if score is None:
                logger.debug(f"ORB: {cand.symbol} dropped — missing feature")
                cand.rejected_reason = 'feature_missing'
                continue
            if score < self.filter_threshold:
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

        # 4. Dedup by family + super-group
        top_syms = dedup_candidates(
            ranked_symbols,
            max_keep=self.max_concurrent - len(self.open_positions),
            by_family=self.dedup_by_family,
            by_super_group=self.dedup_by_super_group,
        )

        # 5. For each kept candidate, build plan + submit
        submitted: List[str] = []
        for sym in top_syms:
            cand = self.candidates[sym]
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
        try:
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
        except Exception as e:
            logger.error(f"ORB: {plan.symbol} submit_entry failed: {e}")
            return None

        # Capture NBBO at submit for entry slippage attribution (mirrors bull flag).
        # Done BEFORE _save_pending_trade so the quote fields persist to DB and
        # survive a restart (previously lost — reconstructed OpenPosition had
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
        )
        # Track on CandidateState too for clean time-stop cancellation
        cand = self.candidates.get(plan.symbol)
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
                    # Accept the partial as a real fill. Subsequent fills on
                    # same order would be rare (day stop-limit) but if they
                    # happen, the bracket SL/TP cap the position anyway.
                    # Use filled_qty from status.
                    logger.warning(
                        f"ORB: {sym} partial fill — treating as complete "
                        f"(filled_qty={order_status.get('filled_qty') if isinstance(order_status, dict) else getattr(order_status, 'filled_qty', 0)})"
                    )
                    self._confirm_fill(pos, order_status)
                elif status in ('canceled', 'cancelled', 'expired', 'rejected',
                                 'done_for_day', 'suspended'):
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
        fill_at = datetime.now(timezone.utc)

        pos.entry_price = fill_price
        pos.shares = shares
        pos.entry_time = fill_at
        pos.order_id = ''  # cleared = filled
        # Fill-rate telemetry
        self.daily_n_filled += 1

        # Risk-per-share recomputed from ACTUAL fill
        risk_per_share = max(pos.entry_price - pos.stop_price, 0.0)

        # DB update — include entry slippage attribution (parity with bull flag + MACD wave).
        fill_update: Dict[str, object] = {
            'order_status': 'filled',
            'fill_price': fill_price,
            'filled_at': fill_at,
            'order_filled_at': fill_at,
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

        Updates DB + in-memory state for each exit.

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
        exit_price = float(ev.exit_price)
        pnl = (exit_price - pos.entry_price) * pos.shares
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100.0
        self.daily_pnl += pnl

        # Exit microstructure from StopMonitor event (parity with bull flag + MACD wave).
        # exit_slippage convention: exit_limit_price - actual_exit_price.
        # Positive = we got a better price than our limit; negative = we paid slip.
        def _numf(v):
            """Coerce to float if real numeric (not MagicMock), else None."""
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None
            f = float(v)
            return f if f > 0 else None

        def _numi(v):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None
            i = int(v)
            return i if i > 0 else None

        def _strs(v):
            if isinstance(v, str) and v:
                return v
            return None

        exit_limit_price = _numf(getattr(ev, 'exit_limit_price', None))
        exit_slippage = (
            exit_limit_price - exit_price if exit_limit_price is not None else None
        )
        exit_update = {
            'exit_price': exit_price,
            'exit_reason': ev.exit_reason,
            'exited_at': datetime.now(timezone.utc),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'order_status': 'closed',
            'exit_trigger_price': _numf(getattr(ev, 'exit_trigger_price', None)),
            'exit_quote_bid': _numf(getattr(ev, 'exit_quote_bid', None)),
            'exit_quote_ask': _numf(getattr(ev, 'exit_quote_ask', None)),
            'exit_quote_bid_size': _numi(getattr(ev, 'exit_quote_bid_size', None)),
            'exit_quote_ask_size': _numi(getattr(ev, 'exit_quote_ask_size', None)),
            'exit_limit_price': exit_limit_price,
            'exit_pricing_method': _strs(getattr(ev, 'pricing_method', None)),
            'exit_slippage': exit_slippage,
        }
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
        # Cancel pending buy-stop orders (candidates with plan submitted but not filled)
        for sym, cand in self.candidates.items():
            if cand.plan_submitted and sym not in self.open_positions:
                logger.info(f"ORB FORCE-CLOSE: cancelling unfilled pending order for {sym}")
                try:
                    pending = self.db.get_open_trades(
                        datetime.now(timezone.utc).date(), strategy=STRATEGY_NAME
                    ) if hasattr(self.db, 'get_open_trades') else []
                    for t in pending:
                        if t.get('symbol') == sym and t.get('order_status') == 'pending_new':
                            self.alpaca.cancel_order(t['order_id'])
                except Exception as e:
                    logger.warning(f"ORB: cancel pending for {sym} failed: {e}")

        # Market-close all ORB open positions. Cancel bracket legs FIRST.
        import time as _time
        for sym, pos in list(self.open_positions.items()):
            try:
                n_legs = self._cancel_symbol_open_orders(sym)
                if n_legs > 0:
                    logger.info(
                        f"ORB FORCE-CLOSE: {sym} canceled {n_legs} bracket/safety legs "
                        f"— waiting for ACK before close_position"
                    )
                    # Alpaca needs a moment to process cancels before the shares
                    # become 'available' again. 0.8s empirically sufficient on paper.
                    _time.sleep(0.8)
                result = self.alpaca.close_position(sym)
                logger.info(
                    f"ORB FORCE-CLOSE: {sym} market close order="
                    f"{(result or {}).get('id', '?')}"
                )
                closed += 1
            except Exception as e:
                failed.append(sym)
                # Retry ONCE with longer wait — Alpaca may still be ACK'ing cancels.
                try:
                    _time.sleep(2.0)
                    result = self.alpaca.close_position(sym)
                    logger.info(f"ORB FORCE-CLOSE: {sym} closed on retry "
                                f"(order={(result or {}).get('id', '?')})")
                    closed += 1
                    failed.remove(sym)
                except Exception as e2:
                    # CRITICAL: position will leak overnight unless operator acts.
                    # Telegram + DB flag so it shows in tomorrow's sync drift scan.
                    self._notify_error(
                        f"FORCE-CLOSE FAILED for {sym} — position will leak overnight "
                        f"(qty={pos.shares}, entry=${pos.entry_price:.2f}). "
                        f"MANUAL ACTION REQUIRED before next market open.",
                        exc=e2,
                    )

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
                # Also track on candidate so time-stop cancel logic works with
                # the ORIGINAL submit clock (not now()).
                if sym in self.candidates:
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
            # Gather per-orphan info for the alert
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
            self._notify_error(
                f"ORPHAN ALPACA POSITIONS ({len(orphans)}) not tracked by ORB — "
                f"{'; '.join(details)}. Likely from a failed force-close or "
                f"cross-day state drift. Manual review required."
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
            pattern_data = _json.dumps({
                'range_high': plan.range_high,
                'range_low': plan.range_low,
                'range_size': plan.range_size,
                'composite_score': plan.composite_score,
                'quintile': plan.quintile,
                'adaptive_mult': plan.adaptive_mult,
                'lock_arm_at_r': plan.lock_arm_at_r,
                'lock_stop_r': plan.lock_stop_r,
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
    """
    if bars_df is None or len(bars_df) == 0:
        return None
    if 'timestamp' not in bars_df.columns:
        return None
    ts_col = bars_df['timestamp']
    try:
        if not pd.api.types.is_datetime64_any_dtype(ts_col):
            ts_col = pd.to_datetime(ts_col, utc=True, errors='coerce')
        mask = (ts_col.dt.minute == 30) & (ts_col.dt.hour.isin([13, 14]))
    except (AttributeError, TypeError, ValueError) as e:
        logger.debug(f"ORB: _first_session_open_ts_utc timestamp parse failed: {e}")
        return None
    if not mask.any():
        return None
    return ts_col.loc[mask].iloc[0]


def _et_offset_hours(now_utc: datetime) -> int:
    """Approximate ET offset from UTC. EDT (Mar-Nov) = 4h. EST (Nov-Mar) = 5h."""
    m = now_utc.month
    if 3 <= m <= 10:
        return ET_OFFSET_EDT_HOURS
    return ET_OFFSET_EST_HOURS
