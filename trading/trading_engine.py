"""
Trading engine — orchestrator for the automated trading pipeline.

Flow:
1. Scanner qualifies a stock → on_stock_qualified(symbol)
2. Fetch 1-min bars for qualified symbols
3. Run bull flag detection
4. Create trade plan if pattern detected
5. Check position manager limits
6. Submit bracket order
7. Track positions
"""

import logging
import os
import queue
import threading
import time as time_mod
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Set, Optional, Dict, Any, List, Tuple

import pandas as pd
import pytz

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.exit_reasons import ExitReason
from trading.pattern_detector import BullFlagDetector
from trading.trade_planner import TradePlanner, TradePlan
from trading.news_kill_guard import news_kill_decision
from trading.bf_vwap_gate import load_vwap_gate_config, passes_vwap_gate
from trading.bf_risk_cap import load_risk_cap_config, cap_usd as _risk_cap_usd, capped_shares
from trading.order_executor import OrderExecutor
from trading.orphan_reconciler import (
    ReconcilerConfig, reconcile_strategy_orphans,
)
from trading.stop_monitor import build_exit_update
from trading.position_manager import PositionManager
from notifications.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)

ET = pytz.timezone('US/Eastern')


# Order types that have NO broker-level bracket legs (no pre-attached SL/TP).
# Both `stop_simple` (production default w/ self_managed_stops=True) and
# `marketable_limit_fallback` (IREZ+TTGT post-mortem 2026-05-08 — submitted
# when bid >= stop, since Alpaca live rejects stop-limit BUY in that case)
# go through the SAME post-fill flow:
#   * gap-fill target adjustment is SKIPPED (no TP leg to replace)
#   * safety-net SL is submitted as a STANDALONE stop-sell (no bracket leg)
# Adding 'marketable_limit_fallback' to this set was missed in the original
# ship — caused fills to take the bracket gap-adjust path, fail to find a
# TP leg, and get emergency-closed (the exact failure shape this fix was
# trying to prevent). Use this set going forward; do not duplicate the
# string check inline.
_SIMPLE_ORDER_TYPES = frozenset(('stop_simple', 'marketable_limit_fallback'))


def _end_of_minute_epoch(ts) -> float:
    """Unix epoch at the END of the minute containing `ts` (aware datetime,
    naive-UTC datetime, or ISO string). Used as `skip_exits_until_ts` on
    bull-flag watches so the fill minute is excluded from trail state and
    stop checks — BT parity (simulate loop starts at entry+1), 2026-09-05.

    Returns 0.0 (= no exclusion) when `ts` is missing or unparseable; that
    fallback is logged because it silently reverts the watch to the
    pre-fix semantics for one minute.
    """
    import math
    if ts is None:
        logger.warning("_end_of_minute_epoch: no fill timestamp — entry-bar "
                       "exclusion disabled for this watch")
        return 0.0
    try:
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        if getattr(ts, 'tzinfo', None) is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return float(math.floor(ts.timestamp() / 60.0) * 60 + 60)
    except Exception as e:
        logger.warning(f"_end_of_minute_epoch: unparseable fill timestamp {ts!r}: {e} "
                       f"— entry-bar exclusion disabled for this watch")
        return 0.0


def _is_simple_order(pending: dict) -> bool:
    """True if the pending order has no broker-level bracket legs.

    Centralized check for the post-fill flow's simple-vs-bracket dispatch.
    See `_SIMPLE_ORDER_TYPES` for rationale.
    """
    return pending.get('order_type') in _SIMPLE_ORDER_TYPES


class TradingEngine:
    """
    Orchestrates the automated trading pipeline.

    Receives qualified stocks from the scanner, detects patterns,
    creates trade plans, and executes bracket orders.
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        db: Database,
        detector: BullFlagDetector,
        planner: TradePlanner,
        executor: OrderExecutor,
        position_manager: PositionManager,
        pattern_poll_interval: int = 60,
        enabled: bool = False,
        notifier: Optional['TelegramNotifier'] = None,
        last_entry_time_et: str = "15:00",
        force_close_time_et: str = "15:45",
        setup_expiry_seconds: int = 600,
        market_regime: Optional['MarketRegimeFilter'] = None,
        stop_monitor: Optional[Any] = None,
        safety_net_sl_pct: float = 0.05,
        order_stream: Optional[Any] = None,
    ):
        """
        Initialize TradingEngine.

        Args:
            alpaca_client: Alpaca API client
            db: Database instance
            detector: Bull flag pattern detector
            planner: Trade planner
            executor: Order executor
            position_manager: Position manager
            pattern_poll_interval: Seconds between pattern checks
            enabled: Master kill switch
            notifier: Optional Telegram notifier for trading alerts
            last_entry_time_et: No new entries after this ET time (HH:MM)
            force_close_time_et: Force close all positions at this ET time (HH:MM)
            setup_expiry_seconds: Cancel pending buy-stop after this many seconds
            market_regime: Optional MarketRegimeFilter for SPY regime check
            stop_monitor: Optional StopMonitor for self-managed stops
            safety_net_sl_pct: Safety-net SL percentage for bracket when
                using self-managed stops (default 5%)
        """
        self.alpaca = alpaca_client
        self.db = db
        self.detector = detector
        self.planner = planner
        self.executor = executor
        self.position_manager = position_manager
        self.pattern_poll_interval = pattern_poll_interval
        self.enabled = enabled
        self.notifier = notifier

        # Time controls
        last_h, last_m = last_entry_time_et.split(':')
        self.last_entry_hour = int(last_h)
        self.last_entry_minute = int(last_m)
        fc_h, fc_m = force_close_time_et.split(':')
        self.force_close_hour = int(fc_h)
        self.force_close_minute = int(fc_m)

        self.setup_expiry_seconds = setup_expiry_seconds

        self.market_regime = market_regime

        # Self-managed stops
        self.stop_monitor = stop_monitor
        self.safety_net_sl_pct = safety_net_sl_pct

        # S1: OrderStreamWatcher for push-delivered order status. When present
        # and healthy, the hybrid helper below prefers its cached status over
        # a REST get_order() round-trip on hot-path fill detection.
        self.order_stream = order_stream

        # Load trailing stop + skip_fridays from config
        from config import Config
        _cfg = Config._load_yaml_only()
        self.skip_fridays = bool(_cfg.get("trading", {}).get("skip_fridays", False))
        self.min_stop_distance = float(_cfg.get("trading", {}).get("min_stop_distance", 0.0))
        self.min_price = float(_cfg.get("trading", {}).get("min_price", 2.0))
        ud_cfg = _cfg.get("trading", {}).get("ud_risk_scaling", {})
        self.ud_risk_scaling_enabled = bool(ud_cfg.get("enabled", False))
        self.ud_threshold = float(ud_cfg.get("ud_threshold", 1.2))
        self.ud_scale_factor = float(ud_cfg.get("scale_factor", 0.5))
        trail_cfg = _cfg.get("trading", {}).get("trailing_stop", {})
        self.trailing_stop_enabled = bool(trail_cfg.get("enabled", False))
        self.trailing_stop_r = float(trail_cfg.get("trail_r", 1.0))
        self.trailing_activate_at_r = float(trail_cfg.get("activate_at_r", 2.0))
        # 2026-09-05 BF trail unification: ONE R-basis knob shared with the
        # BT simulator (trading/bf_trail.py). Invalid value raises — must
        # not silently pick a side of the parity contract.
        from trading.bf_trail import normalize_r_basis
        self.trail_r_basis = normalize_r_basis(trail_cfg.get("r_basis"))
        # Experiment D: volume-confirmed trail exit. When enabled, StopMonitor
        # skips trail exits on bars where last-closed-bar volume is below
        # `flag_avg_volume × min_vol_ratio`. See trading/trail_vol_guard.py.
        _vol_conf_cfg = trail_cfg.get("vol_confirmed_exit", {}) or {}
        self.vol_confirmed_trail_enabled = bool(_vol_conf_cfg.get("enabled", False))
        self.vol_confirmed_trail_min_ratio = float(_vol_conf_cfg.get("min_vol_ratio", 1.0))

        # Exhaustion exit config
        exhaust_cfg = _cfg.get("trading", {}).get("exhaustion_exit", {})
        self.exhaustion_exit_enabled = bool(exhaust_cfg.get("enabled", False))
        self.exhaustion_partial_fraction = float(exhaust_cfg.get("partial_fraction", 0.5))
        self.exhaustion_tighter_trail_r = float(exhaust_cfg.get("tighter_trail_r", 0.5))
        self.exhaustion_min_profit_r = float(exhaust_cfg.get("min_profit_r", 3.0))
        # 2026-09-06 profit partial — ONE spec with the BT simulator
        # (trading/bf_profit_partial.py); default off.
        from trading.bf_profit_partial import load_profit_partial_config
        self.profit_partial = load_profit_partial_config(_cfg.get("trading", {}))
        if self.profit_partial.enabled:
            logger.info(
                f"Profit partial ARMED: {self.profit_partial.fraction:.0%} at "
                f"+{self.profit_partial.r_multiple}R, breakeven={self.profit_partial.move_to_breakeven}"
            )
        self.exhaustion_signals = exhaust_cfg.get("signals", {
            'volume_divergence': False,
            'climax_candle': True,
            'shrinking_bodies': False,
            'shooting_star': True,
        })

        # Minimum daily volume filter — skip illiquid stocks
        self.min_daily_volume = int(_cfg.get("scanner", {}).get("min_daily_volume", 0))

        # Two-tier filter config (surgical drop + composite gate on Extras only).
        # Uses the Config.two_tier_filter_cfg helper which returns a dict compatible
        # with trading.two_tier_filter.should_keep. Enabled=false is a no-op at runtime.
        from config import Config as _ConfigCls
        self._two_tier_cfg = _ConfigCls().two_tier_filter_cfg

        # Post-fill gate thresholds (IREZ post-mortem 2026-05-08).
        # Defaults 0.5 / 0.5 — see docs/post_fill_gate_variant_analysis.md.
        # Override via `trading.post_fill_gate.{spy_3d_threshold,bk_ratio_threshold}`
        # in config.yaml. `enabled=false` short-circuits the entire kill switch.
        self._post_fill_gate_cfg = _ConfigCls().post_fill_gate_cfg

        # Risk tiers: scale risk on high-conviction setups
        tier_cfg = _cfg.get("trading", {}).get("risk_tiers", {})
        self.risk_tiers_enabled = bool(tier_cfg.get("enabled", False))
        self.risk_tiers = []
        if self.risk_tiers_enabled:
            for prefix in ['tier1', 'tier2', 'tier3']:
                mult = float(tier_cfg.get(f"{prefix}_multiplier", 0))
                if mult > 0:
                    self.risk_tiers.append({
                        'min_price': float(tier_cfg.get(f"{prefix}_min_price", 0)),
                        'max_price': float(tier_cfg.get(f"{prefix}_max_price", 999)),
                        'min_volume': int(tier_cfg.get(f"{prefix}_min_volume", 0)),
                        'max_volume': int(tier_cfg.get(f"{prefix}_max_volume", 999999999)),
                        'multiplier': mult,
                    })

        self._qualified_symbols: Set[str] = set()
        self._traded_symbols: Set[str] = set()
        self._patterns_detected: int = 0
        self._patterns_traded: int = 0
        self._pattern_details: list = []
        self._pending_orders: Dict[str, Dict] = {}  # symbol -> {order_id, plan, setup, placed_at}
        self._invalidated_levels: Dict[str, float] = {}  # symbol -> breakout_level (skip re-detection)
        self._daily_trade_count: int = 0
        self._notified_setups: Dict[str, float] = {}  # symbol -> breakout_level (dedup Telegram)
        self._macd_warmup_cache: Dict[str, Optional[pd.Series]] = {}  # symbol -> prev-day closes
        self._pending_stop_exits: Dict[str, Any] = {}  # symbol -> StopExitEvent awaiting fill
        self._news_data: Dict[str, Dict] = {}  # symbol -> {news_catalyst, news_headline, news_reason}
        # Two-tier filter: running max(gap_pct, range_pct) per symbol as seen
        # by scanner at qualification time. Used at _check_symbol to classify
        # A-tier (>=20%) vs Extras (10-19%) for the surgical+composite gate.
        self._qualified_max_intraday: Dict[str, float] = {}

        # MACD zone filter config
        macd_zones_cfg = _cfg.get("trading", {}).get("macd_zones", {})
        self.macd_zones_enabled = bool(macd_zones_cfg.get("enabled", False))
        self.macd_dead_zone_min = float(macd_zones_cfg.get("dead_zone_min_pct", -0.2))
        self.macd_dead_zone_max = float(macd_zones_cfg.get("dead_zone_max_pct", 0.1))
        self.macd_strong_neg_threshold = float(macd_zones_cfg.get("strong_neg_threshold_pct", -0.5))
        self.macd_strong_neg_multiplier = float(macd_zones_cfg.get("strong_neg_multiplier", 1.5))
        self.macd_strong_pos_threshold = float(macd_zones_cfg.get("strong_pos_threshold_pct", 0.5))
        self.macd_strong_pos_multiplier = float(macd_zones_cfg.get("strong_pos_multiplier", 1.5))
        self.macd_normal_multiplier = float(macd_zones_cfg.get("normal_multiplier", 1.0))
        # Per-tier MACD zone multipliers for Extras tier (10% ≤ intraday < 20%).
        # Falls back to A-tier defaults if extras_tier block absent.
        _extras_cfg = macd_zones_cfg.get("extras_tier", {}) or {}
        self.macd_extras_strong_pos_multiplier = float(
            _extras_cfg.get("strong_pos_multiplier", self.macd_strong_pos_multiplier))
        self.macd_extras_strong_neg_multiplier = float(
            _extras_cfg.get("strong_neg_multiplier", self.macd_strong_neg_multiplier))
        self.macd_extras_normal_multiplier = float(
            _extras_cfg.get("normal_multiplier", self.macd_normal_multiplier))

        # Per-trade risk cap (2026-09-06): ONE clamp with BT Stage-2
        # (trading/bf_risk_cap.py). Default off until the joint ship call.
        self.risk_cap = load_risk_cap_config(_cfg.get("trading", {}))
        self.risk_cap_usd = _risk_cap_usd(
            self.risk_cap, float(_cfg.get("trading", {}).get("risk_per_trade", 0) or 0))
        if self.risk_cap_usd:
            logger.info(f"Risk cap: ON — max ${self.risk_cap_usd:,.0f} per trade "
                        f"({self.risk_cap.max_risk_mult}x risk_per_trade)")

        # BF entry-price cap (2026-09-06, P1): `bull_flag.max_entry_price` — a
        # TRADE-TIME rule on the breakout level, deliberately separate from the
        # universe band `scanner.price_max` (which the ignition shadow and the
        # BF watchlist share; the BT cache is built on that band and Stage-2
        # applies this same knob to entry_price). 0 = off.
        self.trade_price_max = float(
            _cfg.get("trading", {}).get("bull_flag", {}).get("max_entry_price", 0) or 0)
        if self.trade_price_max > 0:
            logger.info(f"BF entry-price cap: breakouts above ${self.trade_price_max:.2f} skipped")

        # Above-VWAP gate (2026-09-06): ONE decision with BT Stage-2
        # (trading/bf_vwap_gate.py). Default off until the joint ship call.
        self.vwap_gate = load_vwap_gate_config(
            _cfg.get("trading", {}).get("bull_flag", {}))
        if self.vwap_gate.enabled:
            logger.info(f"VWAP gate: ON — long only when breakout > VWAP"
                        f"{self.vwap_gate.min_dist_pct:+.2f}%")

        # Quality filter: skip low-probability setups (validated on 15mo data)
        qf_cfg = _cfg.get("trading", {}).get("quality_filter", {})
        self.quality_filter_enabled = bool(qf_cfg.get("enabled", False))
        self.qf_max_vwap_dist = float(qf_cfg.get("max_vwap_distance_pct", 4.0))
        self.qf_gap_fade_threshold = float(qf_cfg.get("gap_fade_threshold_pct", 15.0))
        self.qf_min_spy_return = float(qf_cfg.get("min_spy_return_pct", -0.3))
        self.qf_slow_pole_max_bars = int(qf_cfg.get("slow_pole_max_bars", 15))
        self.qf_slow_pole_min_gain = float(qf_cfg.get("slow_pole_min_gain_pct", 5.0))
        self._spy_bars_cache: Optional[pd.DataFrame] = None  # cached SPY 1-min bars for quality filter
        self._spy_bars_cache_date: Optional[str] = None

        if self.quality_filter_enabled:
            logger.info(
                f"Quality filter: VWAP>{self.qf_max_vwap_dist}%, "
                f"gap_fade>{self.qf_gap_fade_threshold}%, "
                f"SPY<{self.qf_min_spy_return}%, "
                f"slow_pole>{self.qf_slow_pole_max_bars}bars/<{self.qf_slow_pole_min_gain}%"
            )

        # Regime-aware sizing (Phase 1.4b ship, 2026-04-18). Classifies the
        # current trading day A/B/C1/C2 from SPY T-1 features; applies per-regime
        # multiplier ON TOP of macd_zone * conviction sizing. C2 regime (shallow
        # dip in uptrend) → multiplier 0 → skip trade. Shared classifier with
        # backtest.py via trading/regime_helpers.py (parity by construction).
        _regime_cfg_raw = _cfg.get("trading", {}).get("regime_sizing", {}) or {}
        _regime_mults_raw = _regime_cfg_raw.get("multipliers", {}) or {}
        self.regime_sizing_enabled = bool(_regime_cfg_raw.get("enabled", False))
        self.regime_vol_threshold = float(_regime_cfg_raw.get("vol_threshold_pct", 22.0))
        self.regime_slope_threshold = float(_regime_cfg_raw.get("slope_threshold_pct", 0.15))
        self.regime_multipliers: Dict[str, float] = {
            "A":  float(_regime_mults_raw.get("A",  1.0)),
            "B":  float(_regime_mults_raw.get("B",  1.0)),
            "C1": float(_regime_mults_raw.get("C1", 1.0)),
            "C2": float(_regime_mults_raw.get("C2", 1.0)),
        }
        # Per-trading-day cache — classify at first use each day, reuse all day.
        self._regime_cache_date: Optional[date] = None
        self._regime_cache_value: str = 'unknown'
        if self.regime_sizing_enabled:
            logger.info(
                f"Regime sizing: vol>={self.regime_vol_threshold}%=B, "
                f"slope>{self.regime_slope_threshold}%→C2, "
                f"mults A={self.regime_multipliers['A']} "
                f"B={self.regime_multipliers['B']} "
                f"C1={self.regime_multipliers['C1']} "
                f"C2={self.regime_multipliers['C2']} (0=skip)"
            )

        # Conviction scoring: scale position size by setup quality
        conv_cfg = _cfg.get("trading", {}).get("conviction_scoring", {})
        self.conviction_enabled = bool(conv_cfg.get("enabled", False))
        # Conviction filter (skip trades below threshold). 0.0 = disabled.
        # 1.2 = walk-forward validated (+$10-18K/period OOS).
        # COUPLED to current 5 conviction rules — re-validate if rules change.
        self.conviction_min_threshold = float(conv_cfg.get("min_threshold", 0.0))
        # Sanity-check threshold at startup — catch misconfig loudly,
        # not at first trade tomorrow morning.
        if self.conviction_min_threshold > 3.0:
            logger.warning(
                f"conviction_min_threshold={self.conviction_min_threshold:.2f} > 3.0 "
                f"(max possible conviction score). ALL trades will be blocked. "
                f"Did you mean {self.conviction_min_threshold/10:.2f}?"
            )
        elif self.conviction_min_threshold < 0:
            logger.warning(
                f"conviction_min_threshold={self.conviction_min_threshold:.2f} < 0 "
                f"— filter is INACTIVE (threshold must be > 0)."
            )
        if not self.conviction_enabled and self.conviction_min_threshold > 0:
            logger.warning(
                f"conviction_min_threshold={self.conviction_min_threshold:.2f} set but "
                f"conviction_scoring.enabled=false — filter is INACTIVE."
            )
        if self.conviction_enabled:
            msg = "Conviction scoring: ENABLED (matches backtest V4 model)"
            if self.conviction_min_threshold > 0:
                msg += f" — filter trades with conv < {self.conviction_min_threshold:.2f}"
            logger.info(msg)
        # Marginal-conviction defensive scaling (Experiment H, 2026-04-17).
        # Feature-flagged. When enabled, trades with conviction in
        # [min_threshold, upper_bound) have SIZING scaled by scale_factor.
        _marg_cfg = conv_cfg.get("marginal_scaling", {}) or {}
        _marg_enabled = bool(_marg_cfg.get("enabled", False))
        self.conviction_marginal_scale_factor = (
            float(_marg_cfg.get("scale_factor", 0.5)) if _marg_enabled else 1.0
        )
        self.conviction_marginal_upper = float(_marg_cfg.get("upper_bound", 1.7))
        # V-reversal bonus (Experiment V, 2026-04-17). Feature-flagged.
        _vrev_cfg = conv_cfg.get("v_reversal_bonus", {}) or {}
        self.v_reversal_enabled = bool(_vrev_cfg.get("enabled", False))
        self.v_reversal_bonus = float(_vrev_cfg.get("bonus", 0.4))
        self.v_reversal_gap_pct_max = float(_vrev_cfg.get("gap_pct_max", 0.0))
        self.v_reversal_intraday_range_min = float(
            _vrev_cfg.get("intraday_range_min", 20.0))
        self.v_reversal_pole_gain_min = float(
            _vrev_cfg.get("pole_gain_min", 5.0))

        # News gate: require real catalyst before trading
        news_gate_cfg = _cfg.get("trading", {}).get("news_gate", {})
        self.news_gate_enabled = bool(news_gate_cfg.get("enabled", False))
        if self.news_gate_enabled:
            logger.info("News gate: ENABLED — no catalyst = no trade")

        # News kill rules: block trades in specific loser segments
        nkr_cfg = _cfg.get("trading", {}).get("news_kill_rules", {})
        self.news_kill_enabled = bool(nkr_cfg.get("enabled", False))
        self.nkr_max_avg_vol = float(nkr_cfg.get("max_avg_vol_no_news", 3_000_000))
        self.nkr_min_price = float(nkr_cfg.get("min_price_no_news", 3.0))
        self.nkr_max_float = float(nkr_cfg.get("max_float_no_news", 30_000_000))
        # Catalyst exemption — default OFF (2026-05 A/B: value-destroying;
        # segment rules now apply to every trade). See trading/news_kill_guard.py.
        self.news_kill_catalyst_exemption = bool(
            nkr_cfg.get("catalyst_exemption", False))
        if self.news_kill_enabled:
            logger.info(
                f"News kill rules: ENABLED — "
                f"vol>={self.nkr_max_avg_vol/1e6:.0f}M, "
                f"price<${self.nkr_min_price:.0f}, float>={self.nkr_max_float/1e6:.0f}M "
                f"(catalyst_exemption={self.news_kill_catalyst_exemption})"
            )

        # EOD summary tracking
        self._eod_traded: list = []    # [(symbol, category, headline, pnl)]
        self._eod_skipped: list = []   # [(symbol, category, headline)]

        # SPY MACD afternoon cutoff
        spy_cutoff_cfg = _cfg.get("trading", {}).get("spy_macd_cutoff", {})
        self._spy_macd_cutoff_enabled = bool(spy_cutoff_cfg.get("enabled", False))
        _cutoff_str = spy_cutoff_cfg.get("cutoff_time", "11:30")
        _ch, _cm = _cutoff_str.split(':')
        self._spy_macd_cutoff_time = (int(_ch), int(_cm))
        self._spy_macd_cache: Optional[float] = None  # latest SPY MACD histogram value

        # BF kill rails (Discipline Program Phase 1, docs/
        # bf_discipline_program_aug2026.md, 2026-08-22). Mirrors the ORB
        # kill-rail pattern (trading/orb_engine.py): DB-derived (restart-safe),
        # ET-dated, realized bull_flag P&L only, fail-closed on query error.
        #   daily  <= daily_usd   -> no NEW entries rest of day
        #   weekly <= weekly_usd  -> flatten BF + no entries rest of ISO week
        #   month  <= month_pause_usd -> PAUSE latch + persistent flag file
        #     (data/bf_month_pause.flag, honored at boot, cleared only by
        #     owner removing the file) + [BF] ABANDON-GATE telegram.
        # Env kill: BF_KILL_RAILS=0. Grep: journalctl | grep "BF RAIL".
        _bf_kr_cfg = (_cfg.get("trading", {}).get("bull_flag", {}) or {}) \
            .get("kill_rails", {}) or {}
        self.kill_rails_enabled = bool(_bf_kr_cfg.get("enabled", True))
        self.kill_daily_usd = float(_bf_kr_cfg.get("daily_usd", -800))
        self.kill_weekly_usd = float(_bf_kr_cfg.get("weekly_usd", -1200))
        self.kill_month_pause_usd = float(
            _bf_kr_cfg.get("month_pause_usd", -2500))
        _bf_kr_env = os.environ.get("BF_KILL_RAILS")
        if _bf_kr_env is not None:
            self.kill_rails_enabled = _bf_kr_env.strip().lower() not in (
                '0', 'false', 'no', 'off', '')
            if not self.kill_rails_enabled:
                logger.warning(
                    "BF RAIL: kill rails DISABLED via BF_KILL_RAILS env")
        # Notify/act-once latches (rolled in reset_daily; reset on restart =>
        # at most one re-notify per still-breached period, like ORB).
        self._kill_daily_notified = False
        self._kill_weekly_notified = False
        self._kill_weekly_flattened = False
        self._kill_query_fail_notified = False
        self._kill_pause_logged = False
        self._kill_week_key: Optional[str] = None
        self._kill_month_key: Optional[str] = None
        # Month pause: runtime latch + persistent marker file. Path override
        # via BF_MONTH_PAUSE_FLAG env (tests/drills) or config key.
        _flag_default = str(
            Path(__file__).resolve().parent.parent / 'data'
            / 'bf_month_pause.flag')
        self.month_pause_flag_path = Path(
            os.environ.get('BF_MONTH_PAUSE_FLAG')
            or _bf_kr_cfg.get('month_pause_flag_path', _flag_default))
        self._bf_month_paused = False
        self._sync_month_pause_from_flag(at_boot=True)
        if self.kill_rails_enabled:
            logger.info(
                f"BF RAIL: kill rails ENABLED — daily ${self.kill_daily_usd:,.0f} "
                f"/ weekly ${self.kill_weekly_usd:,.0f} "
                f"/ month-pause ${self.kill_month_pause_usd:,.0f} "
                f"(flag: {self.month_pause_flag_path})")

        self.shutdown_event = None  # Set by caller for graceful shutdown
        # Bar event queue: WebSocket thread enqueues (symbol, bars_df), main thread drains
        self._bar_event_queue: queue.Queue = queue.Queue()

    # ------------------------------------------------------------------
    # S1 — Hybrid order-status helper (stream-first, REST fallback)
    # ------------------------------------------------------------------

    def _get_order_hybrid(
        self,
        order_id: str,
        submitted_at: Optional[datetime] = None,
        fallback_after_s: float = 5.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Return the latest known order status, preferring push-delivered data
        from OrderStreamWatcher (if attached and healthy).

        Semantics mirror AlpacaClient.get_order() return shape so call sites
        can swap with zero downstream changes. Returns None when the stream
        has nothing yet AND the order is too fresh to justify a REST fallback
        (caller should treat as "still pending, try next tick").

        Behavior matrix:
          stream=None                     -> REST always (previous behavior)
          stream set, cache hit           -> cached dict, no network
          stream set, cache miss, aged>=N -> REST (reconcile missed push)
          stream set, cache miss, age<N   -> None  (fresh order; caller retries)

        Args:
            order_id: Alpaca order id to look up
            submitted_at: when the order was submitted (UTC). Enables the
                age gate; if None we skip the gate and fall through to REST.
            fallback_after_s: REST fallback kicks in at or after this age.
        """
        if self.order_stream is not None:
            try:
                cached = self.order_stream.get_status(order_id)
                if cached is not None:
                    return cached
            except Exception as e:
                logger.debug(
                    f"_get_order_hybrid: order_stream.get_status({order_id}) "
                    f"raised ({e}), falling back to REST"
                )

        # Cache miss (or no stream). Decide whether to hit REST now or wait.
        if self.order_stream is not None and submitted_at is not None:
            age = (datetime.now(timezone.utc) - submitted_at).total_seconds()
            if age < fallback_after_s:
                return None  # too fresh; let the stream deliver

        # Fall through to REST.
        try:
            return self.alpaca.get_order(order_id)
        except Exception as e:
            logger.warning(f"_get_order_hybrid: REST get_order({order_id}) failed: {e}")
            return None

    def _get_risk_tier(self, entry_price: float, avg_volume: int) -> float:
        """
        Determine risk multiplier based on entry price and daily volume.

        Returns 1.0 (default) if no tier matches.
        """
        for tier in self.risk_tiers:
            if (tier['min_price'] <= entry_price < tier['max_price'] and
                    tier['min_volume'] <= avg_volume <= tier['max_volume']):
                return tier['multiplier']
        return 1.0

    def _apply_risk_cap(self, plan: 'TradePlan', symbol: str) -> 'TradePlan':
        """Clamp shares so total risk <= risk_per_trade × max_risk_mult.
        Runs AFTER every multiplier (tier×conviction, MACD zone, regime, UD)
        and BEFORE the BP ceiling — same order as BT Stage-2."""
        cap = getattr(self, 'risk_cap_usd', None)
        if not cap:
            return plan
        new_shares, scale = capped_shares(plan.shares, plan.risk_per_share, cap)
        if scale >= 1.0:
            return plan
        logger.info(
            f"{symbol}: RISK CAP ${cap:,.0f} — risk ${plan.shares * plan.risk_per_share:,.0f} "
            f"→ shares {plan.shares} → {new_shares}"
        )
        return TradePlan(
            symbol=plan.symbol,
            entry_price=plan.entry_price,
            stop_loss_price=plan.stop_loss_price,
            take_profit_price=plan.take_profit_price,
            risk_per_share=plan.risk_per_share,
            reward_per_share=plan.reward_per_share,
            risk_reward_ratio=plan.risk_reward_ratio,
            shares=new_shares,
            total_risk=plan.risk_per_share * new_shares,
            pattern=plan.pattern,
        )

    def _apply_bp_ceiling(self, plan: 'TradePlan', symbol: str) -> Optional['TradePlan']:
        """Clamp a plan's share count to what buying power allows.

        2026-05-14: this MUST run LAST in the sizing chain — after
        tier×conviction, MACD-zone, regime, and UD scaling. Pre-2026-05-14
        the BP check ran right after create_plan, BEFORE regime/UD scaling,
        with two defects:
          1. Ordering — regime's up-to-1.5x boost could push the position
             back OVER buying power after the cap had reduced it, producing
             orders the account can't hold (→ rejects / partial fills).
          2. The re-plan used `risk_multiplier` (bare tier mult) instead of
             `combined_mult` (tier × conviction), silently dropping the
             conviction factor and under-sizing BP-capped trades ~33-50%
             (TRT 2026-05-14: BP allowed 15,071 sh, buggy re-plan gave
             ~10,047).
        As a pure post-hoc hard ceiling on the FINAL share count there is no
        re-plan and no multiplier math — it cannot be re-violated by a later
        scaler and it cannot drop a multiplier.

        Returns the (possibly clamped) plan, or None if BP can't afford even
        one share. On any API error it returns the plan unchanged (fail
        open — never block a trade on a transient BP-query failure).
        """
        # Wrap the whole BP computation (API call + comparisons + arithmetic)
        # — fail open on ANY error, never block a trade on a transient BP
        # issue. Matches the pre-2026-05-14 block's broad try/except.
        try:
            buying_power = self.alpaca.get_buying_power()
            if buying_power <= 0:
                return plan
            position_cost = plan.entry_price * plan.shares
            if position_cost <= buying_power:
                return plan
            affordable_shares = int(buying_power / plan.entry_price)
        except Exception as e:
            logger.warning(
                f"{symbol}: Buying power check failed: {e} — proceeding with plan"
            )
            return plan
        if affordable_shares < 1:
            logger.warning(
                f"{symbol}: No buying power for even 1 share "
                f"(BP ${buying_power:,.0f}, entry ${plan.entry_price:.2f}) — skipping"
            )
            return None
        logger.info(
            f"{symbol}: BP ceiling — {plan.shares} → {affordable_shares} shares "
            f"(cost ${position_cost:,.0f} > BP ${buying_power:,.0f})"
        )
        return TradePlan(
            symbol=plan.symbol,
            entry_price=plan.entry_price,
            stop_loss_price=plan.stop_loss_price,
            take_profit_price=plan.take_profit_price,
            risk_per_share=plan.risk_per_share,
            reward_per_share=plan.reward_per_share,
            risk_reward_ratio=plan.risk_reward_ratio,
            shares=affordable_shares,
            total_risk=plan.risk_per_share * affordable_shares,
            pattern=plan.pattern,
        )

    def _get_macd_zone_multiplier(self, symbol: str, bars: pd.DataFrame,
                                    entry_price: float,
                                    intraday_change_pct: float = 0.0) -> float:
        """
        Compute MACD zone risk multiplier for live trading (tier-aware).

        Uses warmed-up MACD histogram to determine zone, then selects the
        multiplier from the per-tier bucket based on intraday_change_pct
        (A-tier ≥20%, Extras 10-20%, edge <10%).

        Args:
            symbol: Stock symbol
            bars: Current day's 1-min bars (from market open)
            entry_price: Planned entry price
            intraday_change_pct: max intraday % gain at entry time (0.0 means
                "unknown" → defaults to A-tier multipliers for back-compat)

        Returns:
            0.0 = skip (dead zone OR Extras-tier normal), else tier-specific
            multiplier for the identified zone.
        """
        from trading.indicators import macd_histogram
        from trading.macd_tier_helpers import select_tier_multipliers

        closes = bars['close'].copy()

        # Prepend warm-up from previous day
        if symbol in self._macd_warmup_cache and self._macd_warmup_cache[symbol] is not None:
            closes = pd.concat([self._macd_warmup_cache[symbol], closes], ignore_index=True)

        if len(closes) < 35:
            return 1.0

        hist = macd_histogram(closes)
        hist_val = float(hist.iloc[-1])
        macd_pct = (hist_val / entry_price) * 100

        # Tier-aware multiplier selection — single source of truth shared
        # with backtest.py via trading.macd_tier_helpers.
        strong_pos_mult, strong_neg_mult, normal_mult, tier = \
            select_tier_multipliers(
                intraday_change_pct,
                self.macd_strong_pos_multiplier,
                self.macd_strong_neg_multiplier,
                self.macd_normal_multiplier,
                self.macd_extras_strong_pos_multiplier,
                self.macd_extras_strong_neg_multiplier,
                self.macd_extras_normal_multiplier,
            )

        if self.macd_dead_zone_min <= macd_pct <= self.macd_dead_zone_max:
            logger.info(f"{symbol}: MACD ZONE SKIP (dead) ({macd_pct:.2f}%)")
            return 0.0
        elif macd_pct < self.macd_strong_neg_threshold:
            logger.info(f"{symbol}: MACD zone strong neg ({macd_pct:.2f}%) tier={tier} → {strong_neg_mult}x")
            return strong_neg_mult
        elif macd_pct > self.macd_strong_pos_threshold:
            logger.info(f"{symbol}: MACD zone strong pos ({macd_pct:.2f}%) tier={tier} → {strong_pos_mult}x")
            return strong_pos_mult
        else:
            # Normal zone. For Extras tier under S2-max, normal_mult=0.0 →
            # trade is skipped by caller. Log at INFO for parity with dead zone.
            if normal_mult == 0.0:
                logger.info(
                    f"{symbol}: MACD ZONE SKIP (Extras-tier normal) "
                    f"({macd_pct:.2f}%) → 0.0x (Extras MACD-neutral filter per S2-max)"
                )
            else:
                logger.info(
                    f"{symbol}: MACD zone normal ({macd_pct:.2f}%) tier={tier} → {normal_mult}x"
                )
            return normal_mult

    def _fetch_macd_warmup(self, symbol: str) -> None:
        """
        Fetch previous trading day's 1-min bars for MACD warm-up.

        Caches the result per symbol so we only fetch once per day.
        Uses the last 60 bars (1 hour) of the previous session.

        Args:
            symbol: Stock symbol to fetch warm-up data for
        """
        import pytz as _pytz
        _et = _pytz.timezone('US/Eastern')
        today = datetime.now(_et).date()

        # Find previous trading day (skip weekends)
        prev_date = today - timedelta(days=1)
        while prev_date.weekday() >= 5:
            prev_date -= timedelta(days=1)

        try:
            # Previous day's market hours in UTC
            prev_open = _et.localize(
                datetime(prev_date.year, prev_date.month, prev_date.day, 9, 30)
            ).astimezone(timezone.utc)
            prev_close = _et.localize(
                datetime(prev_date.year, prev_date.month, prev_date.day, 16, 0)
            ).astimezone(timezone.utc)

            prev_bars = self.alpaca.get_historical_1min_bars(symbol, prev_open, prev_close)
            if prev_bars is not None and not prev_bars.empty:
                warmup_closes = prev_bars['close'].tail(60).reset_index(drop=True)
                self._macd_warmup_cache[symbol] = warmup_closes
                logger.debug(
                    f"{symbol}: MACD warm-up loaded ({len(warmup_closes)} bars "
                    f"from {prev_date})"
                )
            else:
                self._macd_warmup_cache[symbol] = None
                logger.debug(f"{symbol}: No prev-day bars for MACD warm-up")
        except Exception as e:
            self._macd_warmup_cache[symbol] = None
            logger.warning(f"{symbol}: Failed to fetch MACD warm-up: {e}")

    def _compute_vwap(self, bars: pd.DataFrame, up_to_idx: int = None) -> Optional[float]:
        """Compute VWAP from bars[0:up_to_idx+1]. Point-in-time correct.

        VWAP = Σ(typical_price × volume) / Σ(volume)
        typical_price = (H + L + C) / 3
        """
        if bars is None or len(bars) < 1:
            return None
        try:
            slice_end = min(up_to_idx + 1, len(bars)) if up_to_idx is not None else len(bars)
            highs = bars['high'].iloc[:slice_end].values
            lows = bars['low'].iloc[:slice_end].values
            closes = bars['close'].iloc[:slice_end].values
            volumes = bars['volume'].iloc[:slice_end].values
            cum_vol = volumes.sum()
            if cum_vol <= 0:
                return None
            typical_prices = (highs + lows + closes) / 3.0
            return float((typical_prices * volumes).sum() / cum_vol)
        except (KeyError, TypeError):
            return None

    def _get_spy_return(self, at_timestamp=None) -> Optional[float]:
        """Get SPY return from open to a specific time. Point-in-time correct.

        Uses the SPY bars already fetched by _refresh_spy_macd().
        If at_timestamp provided, uses SPY close at-or-before that time (matches BT).
        Otherwise uses latest bar.
        Returns SPY return as percentage, or None if unavailable.
        """
        if self._spy_bars_cache is None or len(self._spy_bars_cache) < 2:
            return None
        spy_open = float(self._spy_bars_cache.iloc[0]['open'])
        if spy_open <= 0:
            return None
        if at_timestamp is not None:
            try:
                ts_str = str(at_timestamp)[:19]
                spy_ts = self._spy_bars_cache['timestamp'].astype(str).str[:19]
                mask = spy_ts <= ts_str
                if mask.any():
                    spy_close = float(self._spy_bars_cache.loc[mask, 'close'].iloc[-1])
                else:
                    spy_close = float(self._spy_bars_cache.iloc[0]['close'])
            except Exception:
                spy_close = float(self._spy_bars_cache.iloc[-1]['close'])
        else:
            spy_close = float(self._spy_bars_cache.iloc[-1]['close'])
        return (spy_close - spy_open) / spy_open * 100

    def _check_quality_filter(
        self, symbol: str, bars: pd.DataFrame, setup, plan,
        prev_close: Optional[float] = None,
        bar_idx: int = None,
    ) -> tuple:
        """Check quality filter conditions. All features known at setup detection time.

        Args:
            symbol: Stock ticker
            bars: 1-min bars from market open to now (completed bars)
            setup: BullFlagSetup with pattern measurements
            plan: TradePlan with entry/stop/target
            prev_close: Previous day's close price (from scanner tracked data)

        Returns:
            (pass: bool, reason: str). If pass=False, skip this setup.
        """
        breakout_level = setup.breakout_level

        # 1. VWAP overextension: breakout level too far above VWAP
        vwap = self._compute_vwap(bars, up_to_idx=bar_idx)
        if vwap and vwap > 0:
            vwap_dist_pct = (breakout_level - vwap) / vwap * 100
            if vwap_dist_pct > self.qf_max_vwap_dist:
                return (False, f"VWAP +{vwap_dist_pct:.1f}% > {self.qf_max_vwap_dist}% (overextended)")

        # 2. Gap fading: stock gapped up big but breakout is below open
        if prev_close and prev_close > 0 and len(bars) > 0:
            try:
                open_price = float(bars.iloc[0]['open'])
                gap_pct = (open_price - prev_close) / prev_close * 100
                if gap_pct >= self.qf_gap_fade_threshold and breakout_level < open_price:
                    return (False,
                            f"gap_fade: gap +{gap_pct:.1f}% but breakout "
                            f"${breakout_level:.2f} < open ${open_price:.2f}")
            except (KeyError, TypeError):
                pass  # bars missing 'open' column (test mocks)

        # 3. SPY down: risk-off environment (point-in-time, matches BT)
        _setup_ts = bars.iloc[bar_idx].name if bar_idx is not None and bar_idx < len(bars) else None
        spy_return = self._get_spy_return(at_timestamp=_setup_ts)
        if spy_return is not None and spy_return < self.qf_min_spy_return:
            return (False, f"SPY {spy_return:+.2f}% < {self.qf_min_spy_return}% (risk-off)")

        # 4. Slow weak pole: pattern took too long with too little gain
        pole_bars = setup.pole_end_idx - setup.pole_start_idx
        pole_gain = setup.pole_gain_pct
        if pole_bars > self.qf_slow_pole_max_bars and pole_gain < self.qf_slow_pole_min_gain:
            return (False,
                    f"slow_pole: {pole_bars} bars, {pole_gain:.1f}% gain (weak momentum)")

        return (True, "")

    def _compute_conviction_score_setup(
        self, setup, spy_3d_range: Optional[float], *,
        vwap_dist_pct: float = 0.0,
        gap_fading: bool = False,
        gap_pct: float = 0.0,
        intraday_range_pct: float = 0.0,
        v_reversal_enabled: bool = False,
        v_reversal_bonus: float = 0.4,
        v_reversal_gap_pct_max: float = 0.0,
        v_reversal_intraday_range_min: float = 20.0,
        v_reversal_pole_gain_min: float = 5.0,
        return_breakdown: bool = False,
    ):
        """Compute conviction score at setup detection time.

        Returns a multiplier (0.25 to 3.0) that scales position size.
        Matches backtest.py exactly — 8 pattern rules (V2_clean + V-reversal
        rule 9, 2026-04-17).

        Args:
            setup: BullFlagSetup object
            spy_3d_range: SPY 3-day average daily range (%)
            vwap_dist_pct: (breakout_level - vwap)/vwap * 100, computed up to
                setup bar. Defaults to 0.0 (rule 7 silent) for back-compat.
                NEW in V2_clean.
            gap_fading: True if today gapped up >=qf_gap_fade_threshold% from
                prev_close AND breakout_level is below today's open. Defaults
                to False (rule 8 silent) for back-compat. NEW in V2_clean.
            return_breakdown: If True, return (final_score, breakdown_dict).
                breakdown_dict has per-rule contributions plus 'raw_score'
                (pre-clamp) and 'final_score' (post-clamp). For trace logging.

        Returns:
            float (when return_breakdown=False) — the position multiplier
            tuple (float, dict) — when return_breakdown=True

        V2_clean rules 7+8 added 2026-04-15 from walk-forward research:
        canonical 16mo +$52K (+15.5%), mean OOS test +$28K, robust on
        all 3 splits. Rule 6 (daily_range_pct) was rejected — look-ahead.
        """
        score = 1.0
        breakdown = {}

        # 1. Pole gain sweet spot (4.5-9%)
        pg = setup.pole_gain_pct
        pg_contrib = 0.3 if 4.5 <= pg <= 9.0 else 0.0
        score += pg_contrib
        breakdown['pole_gain'] = pg_contrib

        # 2. Flag tightness (tight < 30% = good, loose > 50% = bad)
        ft_contrib = 0.0
        pole_height = setup.pole_high - setup.pole_low
        if pole_height > 0:
            flag_range = setup.flag_high - setup.flag_low
            tightness = flag_range / pole_height * 100
            if tightness < 30:
                ft_contrib = 0.3
            elif tightness > 50:
                ft_contrib = -0.3
        score += ft_contrib
        breakdown['flag_tightness'] = ft_contrib

        # 3. Volume ratio pole/flag (>1.7x = buying conviction)
        vr_contrib = 0.0
        if setup.avg_flag_volume > 0:
            vol_ratio = setup.avg_pole_volume / setup.avg_flag_volume
            if vol_ratio > 1.7:
                vr_contrib = 0.3
        score += vr_contrib
        breakdown['vol_ratio'] = vr_contrib

        # 4. SPY 3d range regime.
        # None = data missing/stale per spy_regime helper. Treat as worst case
        # (same penalty as low-vol regime) — matches BT exactly and degrades
        # gracefully when SPY refresh fails. Was a 1.0 sentinel pre-2026-05-02
        # which silently inflated conviction by +0.5 (EAF post-mortem).
        if spy_3d_range is None:
            sr_contrib = -0.5
        elif spy_3d_range > 1.2:
            sr_contrib = 0.3
        elif spy_3d_range < 0.8:
            sr_contrib = -0.5
        else:
            sr_contrib = 0.0
        score += sr_contrib
        breakdown['spy_regime'] = sr_contrib

        # 5. Shallow retracement (< 30%)
        rt_contrib = 0.2 if setup.retracement_pct < 30 else 0.0
        score += rt_contrib
        breakdown['retracement'] = rt_contrib

        # 6. (Rule 6 reserved — daily_range_pct was rejected as look-ahead.)

        # 7. VWAP distance — extension above VWAP signals momentum quality.
        # Walk-forward bucket EV: vwap_dist >= 2 → mean +$1.5K/trade vs <0 → -$1K/tr.
        vw_contrib = 0.2 if vwap_dist_pct >= 2.0 else 0.0
        score += vw_contrib
        breakdown['vwap_dist'] = vw_contrib

        # 8. Gap fading penalty — gap-up that broke down before entry is bearish.
        # Walk-forward: gap_fading=True → -$612/trade test, =False → +$535/tr.
        gf_contrib = -0.3 if gap_fading else 0.0
        score += gf_contrib
        breakdown['gap_fading'] = gf_contrib

        # 9. V-reversal bonus — gap-down + high intraday range + meaningful pole.
        # Feature-flagged; default OFF. Mirrors backtest.py for BT/PROD parity.
        vr_contrib = 0.0
        if v_reversal_enabled and (
            gap_pct < v_reversal_gap_pct_max
            and intraday_range_pct >= v_reversal_intraday_range_min
            and setup.pole_gain_pct >= v_reversal_pole_gain_min
        ):
            vr_contrib = v_reversal_bonus
        score += vr_contrib
        breakdown['v_reversal'] = vr_contrib

        final = max(0.25, min(3.0, score))
        if return_breakdown:
            breakdown['raw_score'] = score
            breakdown['final_score'] = final
            return final, breakdown
        return final

    def _get_spy_3d_range_live(self) -> Optional[float]:
        """SPY 3-day avg daily range from MarketRegimeFilter, or None if missing.

        Calls into `trading.spy_regime.compute_spy_3d_range` for the math —
        same helper used by BT — so live and BT produce identical outputs
        for identical bars (parity by construction).

        Uses bars STRICTLY BEFORE today (T-1, T-2, T-3) — same window as
        BT's `_get_spy_3d_range` (`WHERE bar_date < trade_date`). Today's
        intraday-updating partial daily bar is excluded so the value is
        stable across the entire trading day.

        IREZ post-mortem 2026-05-08: prior to this fix, live re-fetched
        SPY daily bars per cycle including today's partial bar; the 3-day
        mean drifted within seconds (0.80% at conviction → 0.77% at
        post-fill 9 sec later), tripping the post-fill gate kill switch
        on a $37K-shape winner that BT (which always uses T-1/T-2/T-3)
        would have happily ridden. This mismatch was a silent BT-live
        parity bug, separate from any threshold tuning.

        Returns:
            Average daily range %, or `None` if SPY data is missing or stale.
            Callers MUST treat `None` as 'regime unknown' (the conviction
            rule maps None to the worst-case -0.5 penalty). NEVER substitute
            a numeric default — historical bug 2026-05-01 was a `1.0`
            sentinel that landed in the rule's neutral band and inflated
            conviction by +0.5, firing a live order that should have been
            filtered.
        """
        from trading.spy_regime import (
            compute_spy_3d_range,
            is_spy_data_stale,
        )

        today = date.today()

        # Path 1 (preferred): MarketRegimeFilter daily bars, populated by
        # `_refresh_spy_data` at startup + per-cycle. Use the public
        # `get_recent_bars` API rather than reaching for private internals.
        # `before_date=today` restricts to T-1 / T-2 / T-3 — must match
        # BT's `WHERE bar_date < trade_date` filter (parity guarantee).
        if self.market_regime is not None:
            try:
                recent = self.market_regime.get_recent_bars(n=3, before_date=today)
            except Exception as e:
                logger.warning(
                    "_get_spy_3d_range_live: market_regime accessor raised "
                    "%s — skipping path 1", e,
                )
                recent = None
            if recent and len(recent) >= 3:
                # Staleness is gauged from the freshest bar IN OUR 3-DAY
                # WINDOW (T-1 typically). If T-1 is too old (e.g., we've
                # missed a week of data), fall through to None — same
                # behavior as BT.
                latest_in_window = recent[-1].get('date')
                if latest_in_window is None or not is_spy_data_stale(latest_in_window, today):
                    result = compute_spy_3d_range(recent)
                    if result is not None:
                        return result
                    # compute_spy_3d_range already logged WARNING
            else:
                logger.warning(
                    "_get_spy_3d_range_live: market_regime has fewer than 3 "
                    "bars before today (got %d) — _refresh_spy_data may not have run",
                    len(recent) if recent else 0,
                )
        else:
            logger.warning(
                "_get_spy_3d_range_live: no market_regime configured — "
                "regime scoring unavailable",
            )

        # Path 2 (fallback): today's intraday SPY bars give a 1-day range,
        # not 3-day, but it's a directional signal when daily bars are missing.
        # Logged as WARNING so this fallback is observable in journalctl.
        if self._spy_bars_cache is not None and len(self._spy_bars_cache) > 1:
            day_high = float(self._spy_bars_cache['high'].max())
            day_low = float(self._spy_bars_cache['low'].min())
            if day_low > 0:
                logger.warning(
                    "_get_spy_3d_range_live: using today's 1-min range as "
                    "3d-range proxy (market_regime data unavailable)",
                )
                return (day_high - day_low) / day_low * 100.0

        # Both paths failed — fail closed. Caller (conviction rule) must
        # treat None as the worst-case penalty per CLAUDE.md "All Errors
        # must be reported and execution should break" (here: refuse to
        # score regime, downstream conviction may still allow a trade if
        # other rules are strong enough).
        logger.error(
            "_get_spy_3d_range_live: NO SPY data available from any source "
            "— returning None (regime contribution will be max penalty)",
        )
        return None

    def _refresh_spy_macd(self) -> None:
        """
        Fetch SPY 1-min bars and compute current MACD histogram.

        Called each run_pattern_check() cycle when spy_macd_cutoff is enabled.
        Reuses _macd_warmup_cache['SPY'] for prev-day warmup.
        """
        if not self._spy_macd_cutoff_enabled and not self.quality_filter_enabled:
            return
        try:
            import pytz as _pytz
            _et = _pytz.timezone('US/Eastern')
            now_et = datetime.now(_et)
            minutes_since_open = max(
                int((now_et - now_et.replace(hour=9, minute=30, second=0)).total_seconds() / 60), 30
            )
            spy_bars = self.alpaca.get_1min_bars('SPY', lookback_minutes=minutes_since_open)
            if spy_bars is None or spy_bars.empty:
                self._spy_macd_cache = None
                return

            # Cache SPY bars for quality filter (no extra API call)
            self._spy_bars_cache = spy_bars

            # Warmup: fetch prev day bars for SPY (cache once per day)
            if 'SPY' not in self._macd_warmup_cache:
                self._fetch_macd_warmup('SPY')
            warmup = self._macd_warmup_cache.get('SPY')

            from trading.indicators import macd_histogram
            closes = spy_bars['close'].copy()
            if warmup is not None:
                closes = pd.concat([warmup, closes], ignore_index=True)

            if len(closes) < 35:
                self._spy_macd_cache = None
                return

            hist = macd_histogram(closes)
            self._spy_macd_cache = float(hist.iloc[-1])
            logger.debug(f"SPY MACD histogram: {self._spy_macd_cache:.6f}")
        except Exception as e:
            logger.warning(f"Failed to refresh SPY MACD: {e}")
            self._spy_macd_cache = None

    def _get_today_regime(self) -> str:
        """Return the Phase 1.4b regime label ('A'/'B'/'C1'/'C2'/'unknown')
        for today's trading session, using SPY features from YESTERDAY's close.

        Cached once per ET calendar date — first call of the day triggers
        an Alpaca fetch of ~100 calendar days of SPY daily bars (enough
        warmup for the 50-day SMA), subsequent calls reuse the cached label.

        Safe behavior on any failure: return 'unknown' (downstream mult=1.0,
        no trade effect). Never silently amps size or skips everything.
        """
        if not self.regime_sizing_enabled:
            return 'disabled'
        today_et = datetime.now(ET).date()
        if self._regime_cache_date == today_et:
            return self._regime_cache_value

        try:
            from trading.regime_helpers import (
                compute_regime_features, classify_regime)
            # ~100 calendar days ≈ 69 trading days; need 50 for SMA warmup +
            # 10 for slope + 20 for vol. Extra buffer absorbs holidays.
            start = today_et - timedelta(days=100)
            end = today_et - timedelta(days=1)
            bars_by_symbol = self.alpaca.get_daily_bars_range(['SPY'], start, end)
            rows = bars_by_symbol.get('SPY') or []
            if not rows:
                raise RuntimeError("Alpaca returned zero SPY daily bars")
            spy = pd.DataFrame(rows)
            # get_daily_bars_range returns column 'date'; helper expects 'bar_date'.
            if 'bar_date' not in spy.columns and 'date' in spy.columns:
                spy = spy.rename(columns={'date': 'bar_date'})
            spy['bar_date'] = pd.to_datetime(spy['bar_date'])
            feats = compute_regime_features(spy)
            last = feats.iloc[-1]
            above = last['above_sma_50']
            regime = classify_regime(
                float(last['vol_20_ann']) if not pd.isna(last['vol_20_ann']) else None,
                None if pd.isna(above) else bool(above),
                float(last['sma_50_slope_10d']) if not pd.isna(last['sma_50_slope_10d']) else None,
                self.regime_vol_threshold, self.regime_slope_threshold,
            )
        except Exception as exc:
            logger.error(
                f"Regime classification failed for {today_et} ({exc!r}) — "
                f"defaulting to 'unknown' (multiplier 1.0, no trade effect)."
            )
            regime = 'unknown'

        self._regime_cache_date = today_et
        self._regime_cache_value = regime
        logger.info(f"REGIME today={today_et} classified as {regime}")
        return regime

    def _is_spy_macd_cutoff_blocked(self) -> bool:
        """
        Check if SPY MACD afternoon cutoff is blocking new entries.

        Returns True when: enabled AND past cutoff_time AND SPY MACD > 0.
        """
        if not self._spy_macd_cutoff_enabled:
            return False
        now_et = datetime.now(ET)
        current_time = (now_et.hour, now_et.minute)
        if current_time < self._spy_macd_cutoff_time:
            return False
        if self._spy_macd_cache is None:
            return False  # No data → don't block
        return self._spy_macd_cache > 0

    def _refresh_spy_data(self) -> None:
        """Fetch recent SPY daily bars for regime filter."""
        if not self.market_regime:
            return
        try:
            end = date.today()
            # Need enough history for SMA period + buffer
            sma_period = getattr(self.market_regime, 'sma_period', 50)
            lookback_days = int(sma_period * 1.5) + 14  # trading days -> calendar days
            start = end - timedelta(days=lookback_days)
            bars = self.alpaca.get_daily_bars_range(['SPY'], start, end)
            spy_bars = bars.get('SPY', [])
            self.market_regime.load_spy_bars(spy_bars)
            info = self.market_regime.get_regime_info(end)
            vol_str = f"{info['vol_5d']:.2f}%" if info['vol_5d'] is not None else "N/A"
            sma_str = f"{info['sma']:.2f}" if info['sma'] is not None else "N/A"
            below_str = info['is_below_sma']
            logger.info(
                f"SPY regime refreshed: {len(spy_bars)} bars, "
                f"vol_5d={vol_str}, SMA={sma_str}, below_SMA={below_str}, "
                f"regime_ok={info['is_ok']}"
            )
        except Exception as e:
            logger.error(f"Failed to refresh SPY regime data: {e}")

    def on_stock_qualified(self, symbol: str, news_catalyst: bool = None,
                           news_headline: str = None, news_reason: str = None,
                           news_category: str = None,
                           max_intraday_change_pct: Optional[float] = None) -> None:
        """
        Handle a stock qualified by the scanner.

        Adds to the qualified symbols set for pattern monitoring.
        Stores news classification for gate check + persistence.

        Args:
            symbol: Qualified stock symbol
            news_catalyst: LLM classification (True=real catalyst, False=noise, None=unknown)
            news_headline: Top news headline
            news_reason: LLM's reason for classification
            news_category: News category (FDA_CLINICAL, EARNINGS, GARBAGE_RECAP, etc.)
            max_intraday_change_pct: Running max of max(gap_pct, range_pct) so far
                today. Used by the two-tier filter to classify A-tier vs Extras
                at entry time. None = scanner didn't provide (older callers).
        """
        # Two-tier filter: keep running max across re-qualifications.
        if max_intraday_change_pct is not None:
            existing = self._qualified_max_intraday.get(symbol)
            if existing is None or max_intraday_change_pct > existing:
                self._qualified_max_intraday[symbol] = float(max_intraday_change_pct)
        # Store news data for later persistence with trade record.
        # Never downgrade: once a real catalyst is found, keep it.
        # Scanner re-qualifies stocks each cycle — LLM may flip on re-classification.
        if news_catalyst is not None:
            existing = self._news_data.get(symbol, {})
            existing_is_real = existing.get('news_catalyst') is True
            if not existing_is_real or news_catalyst is True:
                self._news_data[symbol] = {
                    'news_catalyst': news_catalyst,
                    'news_headline': (news_headline or '')[:200],
                    'news_reason': (news_reason or '')[:100],
                    'news_category': news_category or 'OTHER',
                }
        if not self.enabled:
            logger.debug(f"{symbol}: Trading engine disabled, ignoring qualified stock")
            return

        if symbol in self._traded_symbols:
            logger.debug(f"{symbol}: Already traded today, skipping")
            return

        if symbol not in self._qualified_symbols:
            self._qualified_symbols.add(symbol)
            logger.info(f"{symbol}: Added to qualified symbols for pattern monitoring")
            # Fix 4 (2026-05-01): pre-warm marginability cache in background.
            # `is_marginable` is the only Alpaca API call on the entry-path
            # critical section (gated on risk_multiplier > 1.0). Pre-warming
            # at qualification time means the result is cached well before
            # the pattern fires (worst case ~30 min later, best case ~30 sec).
            # Fail-safe: if pre-warm errors, the entry path's sync fallback
            # still works (line 2810-2812).
            if (self.risk_tiers_enabled
                    and not hasattr(self, '_margin_cache')):
                self._margin_cache = {}
            if (self.risk_tiers_enabled
                    and symbol not in self._margin_cache):
                threading.Thread(
                    target=self._prewarm_marginability,
                    args=(symbol,),
                    name=f'prewarm-margin-{symbol}',
                    daemon=True,
                ).start()
            # Subscribe to real-time 1-min bars — skip for sub-ADV stocks
            # (saves WebSocket bandwidth + RT callback cycles)
            if self.stop_monitor and hasattr(self.stop_monitor, 'subscribe_bars'):
                _uni = self.db.get_universe_stock(symbol) if self.db else None
                _adv = int((_uni.get('avg_volume_daily') or 0)) if _uni else 0
                if self.min_daily_volume > 0 and 0 < _adv < self.min_daily_volume:
                    logger.debug(f"{symbol}: Skipping bar subscription (ADV {_adv:,} < {self.min_daily_volume:,})")
                else:
                    self.stop_monitor.subscribe_bars(symbol)
                # Seed bar window with historical bars from market open
                try:
                    import pytz as _pytz
                    _et = _pytz.timezone('US/Eastern')
                    _now_et = datetime.now(_et)
                    _market_open = _now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                    _mins = max(int((_now_et - _market_open).total_seconds() / 60), 30)
                    hist = self.alpaca.get_1min_bars(symbol, lookback_minutes=_mins)
                    if hist is not None and not hist.empty:
                        self.stop_monitor._bar_windows[symbol] = hist.to_dict('records')
                        logger.info(f"{symbol}: Seeded bar window with {len(hist)} historical bars")
                except Exception as e:
                    logger.warning(f"{symbol}: Failed to seed bar window: {e}")

    def _persist_marginability_if_needed(
        self, symbol: str, is_marginable: bool,
    ) -> None:
        """Persist marginability observation to universe table (main thread).

        Idempotent within a session via `_margin_persisted` set — writes
        DB at most once per (symbol, day). MUST be called from the main
        thread; the pre-warm daemon thread MUST NOT call this (SQLite
        connections are not thread-safe).
        """
        if not self.db:
            return
        if not hasattr(self, '_margin_persisted'):
            self._margin_persisted = set()
        if symbol in self._margin_persisted:
            return
        try:
            self.db.set_marginability(symbol, is_marginable)
            self._margin_persisted.add(symbol)
        except Exception as e:
            logger.warning(
                f"{symbol}: marginability persist failed: {e}"
            )

    def _prewarm_marginability(self, symbol: str) -> None:
        """Background task: fetch is_marginable and cache it in-memory (Fix 4).

        Runs in a daemon thread spawned from `on_stock_qualified`. Errors
        are swallowed at WARNING — the entry path retries synchronously
        if the cache is still empty when needed (line 2810-2812).

        IMPORTANT: this thread does NOT touch SQLite. Cross-thread DB
        writes on a main-thread connection cause SQLite segfaults under
        pytest (tests/test_trading_engine.py crashed on this in 5/1
        pre-deploy). The DB persistence is handled on the main thread
        in `_persist_marginability_if_needed` invoked from the entry
        path, which sees the cache populated and writes once.
        """
        try:
            if symbol in self._margin_cache:
                return
            result = self.alpaca.is_marginable(symbol)
            self._margin_cache[symbol] = bool(result)
            logger.debug(
                f"{symbol}: marginability pre-warmed = {result}"
            )
        except Exception as e:
            logger.warning(
                f"{symbol}: marginability pre-warm failed: {e} — "
                f"entry-path will retry synchronously if needed"
            )

    def _on_bar_close(self, symbol: str, bars_df) -> None:
        """Handle real-time 1-min bar close from WebSocket.

        Called by StopMonitor in the WebSocket daemon thread.
        Enqueues the event for the main thread to process (avoids SQLite cross-thread errors).
        """
        if not self.enabled:
            return
        if symbol not in self._qualified_symbols:
            return
        if symbol in self._traded_symbols:
            return
        if symbol in self._pending_orders:
            return

        # Enqueue for main thread — WebSocket thread can't touch SQLite
        try:
            self._bar_event_queue.put_nowait((symbol, bars_df))
        except queue.Full:
            logger.warning(f"{symbol}: Bar event queue full, dropping")

    def _drain_bar_events(self) -> Optional[Dict[str, Any]]:
        """Process queued bar events from WebSocket thread. Called from main thread."""
        last_result = None

        # Same guards as run_pattern_check() — RT events must respect all limits
        def _flush_queue():
            while not self._bar_event_queue.empty():
                try:
                    self._bar_event_queue.get_nowait()
                except queue.Empty:
                    break

        if self.skip_fridays and date.today().weekday() == 4:
            _flush_queue()
            return None
        if self.market_regime and not self.market_regime.is_regime_ok(date.today()):
            _flush_queue()
            return None
        if self.market_regime and self.market_regime.max_trades_per_day > 0 and self._daily_trade_count >= self.market_regime.max_trades_per_day:
            _flush_queue()
            return None
        # BF kill rails — RT bar entries must respect the same gates as the
        # polling path (Discipline Program Phase 1).
        if self._kill_rails_blocked():
            _flush_queue()
            return None

        while not self._bar_event_queue.empty():
            try:
                symbol, bars_df = self._bar_event_queue.get_nowait()
            except queue.Empty:
                break

            if symbol in self._traded_symbols or symbol in self._pending_orders:
                continue
            if self._is_past_last_entry_time():
                continue

            logger.info(f"{symbol}: RT bar close — instant pattern check")
            try:
                result = self._check_symbol(symbol, prefetched_bars=bars_df)
                if result:
                    logger.info(f"{symbol}: RT pattern detection → order placed!")
                    last_result = result
            except Exception as e:
                logger.error(f"{symbol}: RT bar check error: {e}")
        return last_result

    def clear_qualified_symbols(self) -> None:
        """Clear qualified symbols for fresh scanner cycle.

        Called by scanner before each 1-min intraday cycle. Symbols that
        already qualified are KEPT — once qualified, stay qualified for
        the day. This prevents dollar-volume bucket rotation from
        de-qualifying stocks mid-session. Only news_data for NEW symbols
        is refreshed; existing qualified symbols retain their data.
        """
        # Don't clear _qualified_symbols — once qualified, always qualified
        # Only clear news_data for symbols not yet qualified (fresh scan picks up new ones)
        # The scanner will re-call on_stock_qualified() which is idempotent (checks set membership)
        pass

    @staticmethod
    def _bar_start_to_close(bar_timestamp_raw) -> datetime:
        """Convert an Alpaca 1-min bar timestamp (bar START) to actual close time.

        Alpaca convention: a bar with `timestamp = 14:04:00` represents trades
        from 14:04:00 to 14:05:00, closing at 14:05:00. Add 60s to get the
        actual close. Mirrors trading.macd_wave_engine.MACDWaveEngine._bar_start_to_close
        — kept as an inline duplicate to avoid cross-engine coupling.
        """
        if isinstance(bar_timestamp_raw, pd.Timestamp):
            bar_start = bar_timestamp_raw.to_pydatetime()
        elif isinstance(bar_timestamp_raw, datetime):
            bar_start = bar_timestamp_raw
        else:
            bar_start = pd.to_datetime(bar_timestamp_raw, utc=True).to_pydatetime()
        if bar_start.tzinfo is None:
            bar_start = bar_start.replace(tzinfo=timezone.utc)
        return bar_start + timedelta(seconds=60)

    def _is_past_last_entry_time(self) -> bool:
        """Check if current ET time is past last_entry_time."""
        now_et = datetime.now(ET)
        return (now_et.hour > self.last_entry_hour or
                (now_et.hour == self.last_entry_hour and now_et.minute >= self.last_entry_minute))

    def send_eod_summary(self) -> None:
        """Send EOD summary of traded and skipped stocks to Telegram."""
        if not self.notifier:
            return
        if not self._eod_traded and not self._eod_skipped:
            return

        lines = ["📊 EOD News Gate Summary:"]

        if self._eod_traded:
            lines.append("\nTRADED (with catalyst):")
            for sym, cat, hl, pnl in self._eod_traded:
                lines.append(f"  {sym} — {cat}: {hl[:60]}")

        if self._eod_skipped:
            lines.append("\nSKIPPED (no catalyst):")
            for sym, cat, hl in self._eod_skipped:
                reason = hl[:60] if hl else "no news found"
                lines.append(f"  {sym} — {cat}: {reason}")

        lines.append(f"\nStats: {len(self._eod_traded)} traded, {len(self._eod_skipped)} skipped")

        msg = "\n".join(lines)
        logger.info(msg)
        try:
            self.notifier.send_message_sync(msg)
        except Exception as e:
            logger.error(f"Failed to send EOD summary: {e}")

        # Reset for next day
        self._eod_traded.clear()
        self._eod_skipped.clear()

    def _is_past_force_close_time(self) -> bool:
        """Check if current ET time is past force_close_time."""
        now_et = datetime.now(ET)
        return (now_et.hour > self.force_close_hour or
                (now_et.hour == self.force_close_hour and now_et.minute >= self.force_close_minute))

    def _identify_bracket_legs(
        self, legs: List[Dict], expected_sl: float = None, expected_tp: float = None
    ) -> tuple:
        """
        Identify stop-loss and take-profit legs from bracket order legs.

        Args:
            legs: List of leg dicts from Alpaca order
            expected_sl: Expected stop loss price (for disambiguation)
            expected_tp: Expected take profit price (for disambiguation)

        Returns:
            Tuple of (sl_leg, tp_leg) — either may be None if not found
        """
        sl_leg = None
        tp_leg = None
        for leg in legs:
            if leg.get('side') != 'sell':
                continue
            has_stop = leg.get('stop_price') is not None
            has_limit = leg.get('limit_price') is not None
            if has_stop and not has_limit:
                sl_leg = leg
            elif has_limit and not has_stop:
                tp_leg = leg
            elif has_stop and has_limit:
                # Both present — match by proximity to expected prices
                if expected_sl and abs(leg['stop_price'] - expected_sl) < abs(leg['limit_price'] - expected_sl):
                    sl_leg = leg
                else:
                    tp_leg = leg
        return sl_leg, tp_leg

    def _manage_pending_orders(self) -> Optional[Dict[str, Any]]:
        """
        Check status of all pending buy-stop orders.

        Processes ALL pending orders each cycle (does not return early on
        first fill). This ensures no fills are missed when multiple orders
        fill simultaneously.

        For each pending order:
        - If filled → mark traded, send notification
        - If price dropped below flag_low → cancel order (setup invalidated)
        - If cancelled/expired → remove from tracking

        Returns:
            Dict with last fill details if any order was filled, None otherwise
        """
        if not self._pending_orders:
            return None

        symbols_to_remove = []
        last_fill_result = None

        for symbol, pending in list(self._pending_orders.items()):
            order_id = pending['order_id']

            # S1: stream-first, REST fallback after 5s of order age. When the
            # stream has nothing yet for a fresh order, hybrid returns None —
            # we continue and re-check next tick (the stream will deliver).
            order_status = self._get_order_hybrid(
                order_id, submitted_at=pending.get('placed_at')
            )
            if order_status is None:
                # Only notify if the order is old enough that REST should've
                # answered — fresh-order None is just "stream hasn't fired yet".
                placed = pending.get('placed_at')
                aged = (
                    placed is not None
                    and (datetime.now(timezone.utc) - placed).total_seconds() >= 10.0
                )
                if aged and self.notifier:
                    self.notifier.notify_error(
                        f"{symbol}: order status unavailable for >10s "
                        "(stream+REST both empty)",
                        component="OrderTracking",
                    )
                continue

            status = order_status.get('status', 'unknown')

            if status == 'filled':
                fill_price = order_status.get('filled_avg_price')
                filled_qty = order_status.get('filled_qty', 0)

                # Fix 1: Retry if fill data missing (Alpaca can lag on fill price).
                # S1: hybrid prefers the stream (no RTT when push has delivered);
                # fallback_after_s=0 disables the age gate for this tight retry
                # loop — we want REST IMMEDIATELY on a stream miss, not wait for
                # the default 5s gate when the order is already known to be filled.
                if fill_price is None:
                    for attempt in range(5):
                        time_mod.sleep(0.5)
                        refreshed = self._get_order_hybrid(
                            order_id,
                            submitted_at=pending.get('placed_at'),
                            fallback_after_s=0.0,
                        )
                        if refreshed is None:
                            continue
                        fill_price = refreshed.get('filled_avg_price')
                        filled_qty = refreshed.get('filled_qty', filled_qty)
                        if fill_price is not None:
                            logger.info(f"{symbol}: Fill price resolved on retry {attempt + 1}")
                            break

                    # Position fallback
                    if fill_price is None:
                        try:
                            positions = self.alpaca.get_open_positions()
                            for pos in positions:
                                if pos['symbol'] == symbol:
                                    fill_price = float(pos['avg_entry_price'])
                                    filled_qty = int(pos['qty'])
                                    logger.warning(f"{symbol}: Using position fallback — ${fill_price}")
                                    break
                        except Exception as e:
                            error_msg = f"{symbol}: Position fallback failed: {e}"
                            logger.error(error_msg)
                            if self.notifier:
                                self.notifier.notify_error(error_msg, component="FillTracking")

                    if fill_price is None:
                        error_msg = (
                            f"{symbol}: Fill price unavailable after retries — "
                            f"UNTRACKED FILLED POSITION (order {order_id})"
                        )
                        logger.error(error_msg)
                        if self.notifier:
                            self.notifier.notify_error(error_msg, component="FillTracking")
                        continue

                # Fix 2: Partial fill detection
                plan = pending['plan']
                requested_qty = plan.shares if plan else 0
                if filled_qty and requested_qty and filled_qty < requested_qty:
                    logger.warning(
                        f"{symbol}: PARTIAL FILL — {filled_qty}/{requested_qty} shares @ ${fill_price}"
                    )
                actual_qty = filled_qty if filled_qty and filled_qty > 0 else requested_qty

                logger.info(
                    f"{symbol}: Buy-stop order FILLED at ${fill_price} — "
                    f"{actual_qty} shares, ID: {order_id}"
                )

                # IREZ 2026-05-08 fix: idempotency guard against OrderStream
                # replay re-firing the post-fill kill switch on stale fills.
                # If the symbol is already in _traded_symbols, a prior live
                # entry already processed (and either kept open or closed
                # this fill). Subsequent "fills" detected here are replays
                # from OrderStream reconnect after restart — silently discard.
                # Without this, the kill switch can re-trigger and try to
                # close a separate live position (today's IREZ #237 was
                # saved only because broker SL was holding the qty).
                if symbol in self._traded_symbols:
                    logger.warning(
                        f"{symbol}: ignoring stale fill replay for {order_id} — "
                        f"symbol already in _traded_symbols (live position managed elsewhere)"
                    )
                    symbols_to_remove.append(symbol)
                    continue

                # IREZ 2026-05-08 fix (Bugs 4b + 5): book the fill state FIRST,
                # BEFORE any early-exit rule (gap-over / post-fill-exit) can
                # short-circuit via continue. Without this, the rule paths
                # bypassed _traded_symbols.add + mark_traded → pattern
                # detector kept firing → Alpaca wash-trade-rejected the
                # re-buys all the way to last-entry cutoff. The DB row
                # also stayed in pre-fill state with no fill_price/exit_*,
                # then got marked 'cancelled' by downstream cleanup
                # (today's IREZ row 113 looked cancelled despite a real
                # round-trip on Alpaca).
                self._traded_symbols.add(symbol)
                self.position_manager.mark_traded(symbol)
                symbols_to_remove.append(symbol)

                fill_at = datetime.now(timezone.utc)
                trade_record = self.db.get_trade_by_order_id(order_id)
                if trade_record:
                    update = {
                        'order_status': 'filled',
                        'fill_price': fill_price,
                        'filled_qty': actual_qty,
                        'filled_at': fill_at,
                    }
                    self.db.update_trade(trade_record['id'], update)

                # Gap-over rejection: if fill is >2% above breakout, close immediately.
                # 15-month BT data: >2% gap-overs have 23% WR, net losers.
                # Matches backtest.py:1587 logic.
                setup = pending.get('setup')
                if setup and setup.breakout_level > 0:
                    gap_over_pct = (fill_price - setup.breakout_level) / setup.breakout_level
                    if gap_over_pct > 0.02:
                        logger.warning(
                            f"{symbol}: GAP-OVER REJECTION — fill ${fill_price:.2f} is "
                            f"{gap_over_pct:.1%} above breakout ${setup.breakout_level:.2f} "
                            f"(max 2%). Closing position immediately."
                        )
                        try:
                            self.alpaca.close_position(symbol)
                            if self.notifier:
                                self.notifier.notify_error(
                                    f"{symbol}: Gap-over rejection — sold at market "
                                    f"(fill ${fill_price:.2f} vs breakout ${setup.breakout_level:.2f})",
                                    component="GapOver"
                                )
                        except Exception as e:
                            logger.error(f"{symbol}: Failed to close gap-over position: {e}")
                        # Stamp exit_reason + exited_at on the trade row so
                        # the round-trip record is complete. exit_price + pnl
                        # are filled in by the sell-fill handler downstream.
                        if trade_record:
                            self.db.update_trade(trade_record['id'], {
                                'exit_reason': ExitReason.GAP_OVER_REJECTION.value,
                                'exited_at': datetime.now(timezone.utc),
                            })
                        continue

                # Post-fill exit: in calm markets + weak breakout vol → close immediately.
                # Matches backtest.py post-fill exit logic. Treat missing/stale SPY
                # data (None) the same as low-vol regime — defensive default.
                # Thresholds + enable flag are config-driven (IREZ post-mortem
                # 2026-05-08): defaults tightened to 0.5/0.5 from 0.8/1.0.
                _pfg_enabled = self._post_fill_gate_cfg.get('enabled', True)
                _pfg_spy_thresh = self._post_fill_gate_cfg.get('spy_3d_threshold', 0.5)
                _pfg_bk_thresh = self._post_fill_gate_cfg.get('bk_ratio_threshold', 0.5)
                if _pfg_enabled and self.conviction_enabled and setup:
                    _afv = float(setup.avg_flag_volume) if hasattr(setup, 'avg_flag_volume') else 0
                    _spy_3d = self._get_spy_3d_range_live()
                    _spy_hostile = _spy_3d is None or _spy_3d < _pfg_spy_thresh
                    if _afv > 0 and _spy_hostile:
                        # Get recent bar volume (the fill bar)
                        try:
                            _recent_bars = self.alpaca.get_1min_bars(symbol, lookback_minutes=2)
                            _bk_vol = float(_recent_bars.iloc[-1]['volume']) if _recent_bars is not None and len(_recent_bars) > 0 else 0
                        except Exception:
                            _bk_vol = 0
                        _bk_ratio = _bk_vol / _afv if _afv > 0 else 99
                        if _bk_ratio < _pfg_bk_thresh:
                            _spy_str = f"{_spy_3d:.2f}%" if _spy_3d is not None else "MISSING"
                            logger.warning(
                                f"{symbol}: POST-FILL EXIT — SPY 3d {_spy_str} + "
                                f"bk_vol {_bk_ratio:.1f}x → closing immediately")
                            try:
                                self.alpaca.close_position(symbol)
                                if self.notifier:
                                    self.notifier.notify_error(
                                        f"{symbol}: Post-fill exit (calm market + weak breakout vol)",
                                        component="PostFillExit")
                            except Exception as e:
                                logger.error(f"{symbol}: Failed to close post-fill exit: {e}")
                            # Stamp exit_reason + exited_at so the round-trip
                            # record is complete — exit_price + pnl come in
                            # from the sell-fill handler.
                            if trade_record:
                                self.db.update_trade(trade_record['id'], {
                                    'exit_reason': ExitReason.POST_FILL_EXIT.value,
                                    'exited_at': datetime.now(timezone.utc),
                                })
                            continue

                # Phase 2: Update trade record with fill data
                trade_record = self.db.get_trade_by_order_id(order_id)
                if trade_record:
                    fill_at = datetime.now(timezone.utc)
                    update = {
                        'order_status': 'filled',
                        'fill_price': fill_price,
                        'filled_qty': actual_qty,
                        'filled_at': fill_at,
                    }
                    # Slippage instrumentation (Migration 10) for bull flag —
                    # parity with MACD wave's analyze_slippage.py output. For
                    # bull flag the BT reference price IS the breakout level
                    # (plan.entry_price is set to pattern.breakout_level by
                    # the planner), so we reuse bar_close_price for that role.
                    # loop_processed_at / quote_fetched_at / bar_close_at have
                    # no natural bull-flag analog and stay NULL.
                    placed_at = pending.get('placed_at')
                    _plan = pending.get('plan')
                    update['order_filled_at'] = fill_at
                    if placed_at:
                        update['order_submitted_at'] = placed_at
                        # Guard against tz-naive placed_at from historic DB
                        # rows (startup recovery path: _sync_startup_state
                        # reads created_at which may lack a tz suffix). Skip
                        # this derived field rather than raising — the rest
                        # of the update still persists.
                        try:
                            update['submit_to_fill_ms'] = int(
                                (fill_at - placed_at).total_seconds() * 1000
                            )
                        except (TypeError, ValueError) as e:
                            logger.debug(
                                f"{symbol}: submit_to_fill_ms skipped "
                                f"(tz/value mismatch): {e}"
                            )
                    if _plan and getattr(_plan, 'entry_price', 0) > 0:
                        ref = float(_plan.entry_price)
                        update['bar_close_price'] = ref
                        if fill_price:
                            update['drift_bar_to_fill_bps'] = (
                                (float(fill_price) - ref) / ref * 10000
                            )
                    # Persist news classification with trade for future analysis
                    # 1=catalyst, 0=noise, NULL=no articles or unknown
                    news = pending.get('news_data')
                    if news:
                        cat = news.get('news_catalyst')
                        if cat is True:
                            update['news_catalyst'] = 1
                        elif cat is False:
                            update['news_catalyst'] = 0
                        # else: None → don't set (stays NULL in DB)
                        update['news_headline'] = news.get('news_headline', '')
                        update['news_reason'] = news.get('news_reason', '')
                        logger.info(
                            f"{symbol}: News: catalyst={news['news_catalyst']}, "
                            f"reason={news.get('news_reason', 'N/A')}"
                        )
                    self.db.update_trade(trade_record['id'], update)
                    logger.info(f"{symbol}: Trade DB updated — fill ${fill_price}, qty {actual_qty}")

                    # Persist entry microstructure from quote watch
                    if self.stop_monitor:
                        qsnap = self.stop_monitor.get_quote_watch_snapshot(symbol)
                        if qsnap:
                            entry_micro = {
                                'entry_quote_bid': qsnap['submit_bid'],
                                'entry_quote_ask': qsnap['submit_ask'],
                                'entry_quote_bid_size': qsnap['submit_bid_size'],
                                'entry_quote_ask_size': qsnap['submit_ask_size'],
                                'entry_quote_spread': (
                                    qsnap['submit_ask'] - qsnap['submit_bid']
                                    if qsnap['submit_ask'] > 0 else None
                                ),
                                'entry_quote_ofi': qsnap['ofi_cumulative'],
                                'entry_fill_quote_bid': qsnap['latest_bid'],
                                'entry_fill_quote_ask': qsnap['latest_ask'],
                            }
                            self.db.update_trade(trade_record['id'], entry_micro)
                            logger.info(
                                f"{symbol}: Entry microstructure — "
                                f"submit bid=${qsnap['submit_bid']:.2f} ask=${qsnap['submit_ask']:.2f}, "
                                f"fill bid=${qsnap['latest_bid']:.2f} ask=${qsnap['latest_ask']:.2f}, "
                                f"OFI={qsnap['ofi_cumulative']:.0f}"
                            )
                        # NOTE: do NOT remove_quote_watch here — that schedules
                        # an _unsubscribe_symbol coroutine that races against
                        # the add_watch / upgrade_quote_to_stop_watch call
                        # below (TTGT 2026-05-08 root cause). The atomic
                        # upgrade later in this method handles the swap.

                    # Log L2 order book depth at fill time (async, never blocks trading)
                    try:
                        from data_sources.l2_depth import log_l2_async
                        log_l2_async(symbol, update['filled_at'], trade_record['id'],
                                     self.db.update_trade, column='entry_l2_depth')
                    except Exception as e:
                        logger.debug(f"{symbol}: L2 async launch failed: {e}")
                else:
                    error_msg = f"{symbol}: No trade record for order {order_id} — DB integrity issue"
                    logger.error(error_msg)
                    if self.notifier:
                        self.notifier.notify_error(error_msg, component="DBIntegrity")

                # H5 OR: check breakout volume on thin-liquidity days
                if pending.get('thin_liquidity', False):
                    if not self._check_breakout_volume(symbol, pending):
                        self._emergency_close_position(
                            symbol, order_id, fill_price, actual_qty, trade_record
                        )
                        last_fill_result = {
                            'status': 'thin_liquidity_rejected',
                            'symbol': symbol,
                            'fill_price': fill_price,
                            'reason': 'weak_breakout_volume',
                        }
                        continue

                # Phase 3: Gap-fill TARGET adjustment only — BRACKET orders only.
                # Stop stays at the original technical level (flag low region).
                # Moving stop above the technical level puts it in no-man's land
                # where normal price noise triggers it. Dollar risk increases
                # but the stop is at a price that has structural meaning.
                # When trailing stop is enabled, skip TP adjustment — trail handles exits.
                # Simple (non-bracket) orders have no TP leg; gap-fill adjustment
                # is a no-op and its leg-identification would fire a spurious
                # 'No TP leg found → emergency close' — skip entirely. StopMonitor
                # trail + standalone safety-net SL (submitted below) protect the
                # position on the downside.
                setup = pending.get('setup')
                trail_active = self.trailing_stop_enabled and self.stop_monitor
                # Use the shared helper (covers stop_simple AND
                # marketable_limit_fallback). Simple orders have no broker
                # bracket legs, so the gap-fill TP-replacement path is a no-op
                # and would emergency-close the position trying to find a leg
                # that doesn't exist. See `_SIMPLE_ORDER_TYPES`.
                is_simple_order = _is_simple_order(pending)
                if (not is_simple_order
                        and fill_price and plan and setup
                        and fill_price > setup.breakout_level):
                    entry_gap = fill_price - setup.breakout_level
                    actual_risk = round(fill_price - plan.stop_loss_price, 2)
                    adjusted_target = round(fill_price + plan.risk_per_share * plan.risk_reward_ratio, 2)
                    logger.info(
                        f"{symbol}: Gap fill +${entry_gap:.2f} — "
                        f"stop KEPT at ${plan.stop_loss_price:.2f} (technical level), "
                        f"risk ${plan.risk_per_share:.2f} → ${actual_risk:.2f}/sh"
                        f"{' (trail handles TP)' if trail_active else f', target ${plan.take_profit_price:.2f} → ${adjusted_target:.2f}'}"
                    )

                    # When trailing stop is active, skip TP adjustment — trail handles exits.
                    # The TP leg will be cancelled after StopMonitor registration below.
                    if not trail_active:
                        gap_adjust_failed = False
                        try:
                            order_detail = self.alpaca.get_order(order_id)
                            sl_leg, tp_leg = self._identify_bracket_legs(
                                order_detail.get('legs', []),
                                expected_sl=plan.stop_loss_price,
                                expected_tp=plan.take_profit_price,
                            )

                            # Stop stays at original — no replacement needed
                            # Only adjust target upward
                            if tp_leg:
                                self.alpaca.replace_order_limit_price(tp_leg['id'], adjusted_target)
                                logger.info(f"{symbol}: Target adjusted to ${adjusted_target:.2f}")
                            else:
                                logger.error(f"{symbol}: No TP leg found — cannot adjust target")
                                gap_adjust_failed = True

                            if not gap_adjust_failed and trade_record:
                                self.db.update_trade(trade_record['id'], {
                                    'take_profit_price': adjusted_target,
                                })
                        except Exception as e:
                            logger.error(f"{symbol}: Failed to adjust target after gap fill: {e}")
                            gap_adjust_failed = True

                        if gap_adjust_failed:
                            error_msg = (
                                f"{symbol}: GAP FILL TARGET ADJUSTMENT FAILED — "
                                f"entry gap +${entry_gap:.2f}, actual risk "
                                f"${actual_risk:.2f}/sh. Target not updated."
                            )
                            logger.error(error_msg)
                            if self.notifier:
                                self.notifier.notify_error(error_msg, component="GapFill")
                            self._emergency_close_position(
                                symbol, order_id, fill_price, actual_qty, trade_record,
                                exit_reason=ExitReason.GAP_ADJUST_FAILED.value,
                            )
                            last_fill_result = {
                                'status': 'gap_adjust_failed',
                                'symbol': symbol,
                                'fill_price': fill_price,
                                'reason': 'leg_replacement_failed',
                            }
                            continue

                # Register with StopMonitor for real-time stop watching
                if self.stop_monitor and pending.get('real_stop_level'):
                    if not self.stop_monitor._running:
                        logger.error(
                            f"{symbol}: StopMonitor NOT RUNNING! "
                            f"Call stop_monitor.start() before trading. "
                            f"Position has NO real-time stop protection."
                        )
                        if self.notifier:
                            self.notifier.notify_error(
                                f"{symbol}: CRITICAL — StopMonitor not started! "
                                f"Position unprotected. Only safety-net SL active.",
                                component="StopMonitor",
                            )
                    real_stop = pending['real_stop_level']
                    try:
                        # Trailing stop params (0 = disabled)
                        trail_r = self.trailing_stop_r if self.trailing_stop_enabled else 0.0
                        activate_r = self.trailing_activate_at_r if self.trailing_stop_enabled else 0.0

                        # Shared helper — both stop_simple AND
                        # marketable_limit_fallback have no bracket legs and
                        # need a standalone safety-net SL. See _SIMPLE_ORDER_TYPES.
                        is_simple_order = _is_simple_order(pending)

                        if is_simple_order:
                            # Simple order path: submit standalone safety-net SL
                            safety_net_price = round(fill_price * (1 - self.safety_net_sl_pct), 2)
                            try:
                                sl_result = self.alpaca.submit_stop_sell_order(
                                    symbol=symbol,
                                    qty=actual_qty,
                                    stop_price=safety_net_price,
                                )
                                sl_leg_id = sl_result.get('id', '') if sl_result else ''
                                logger.info(
                                    f"{symbol}: Safety-net SL submitted — "
                                    f"${safety_net_price:.2f} ({self.safety_net_sl_pct:.0%}), "
                                    f"ID: {sl_leg_id}"
                                )
                            except Exception as sl_err:
                                logger.error(
                                    f"{symbol}: Safety-net SL submission FAILED: {sl_err} — "
                                    f"position has NO crash protection, StopMonitor only"
                                )
                                sl_leg_id = ''
                            tp_leg_id = ''  # No TP leg — trailing stop handles it
                        else:
                            # Bracket order path: identify existing bracket legs
                            order_detail = self.alpaca.get_order(order_id)
                            sl_leg, tp_leg = self._identify_bracket_legs(
                                order_detail.get('legs', []),
                                expected_sl=plan.entry_price * (1 - self.safety_net_sl_pct) if plan else None,
                                expected_tp=plan.take_profit_price if plan else None,
                            )
                            tp_leg_id = tp_leg['id'] if tp_leg else ''
                            sl_leg_id = sl_leg['id'] if sl_leg else ''

                        # Save real_stop_loss_price to DB
                        if trade_record:
                            self.db.update_trade(trade_record['id'], {
                                'real_stop_loss_price': real_stop,
                            })

                        _flag_vol = (plan.pattern.avg_flag_volume
                                     if plan and plan.pattern else 0.0)
                        # plan-R fix (2026-05-08): pass planned breakout level
                        # and planned R as well — these decouple the trail's
                        # activation/ratchet math from entry slippage. BT proves
                        # +$62K-$98K HOLDOUT lift at LIVE-realistic slippage.
                        # Hard stop and broker SL stay at fill-based levels.
                        _planned_entry = plan.entry_price if plan else 0.0
                        _planned_R = (plan.entry_price - plan.stop_loss_price
                                      if plan else 0.0)
                        # TTGT 2026-05-08 root-cause fix: atomic quote→stop
                        # upgrade. Replaces the racy remove_quote_watch +
                        # add_watch pair. See StopMonitor.upgrade_quote_to_stop_watch
                        # docstring for why the prior approach silently lost
                        # WS handlers, causing 14+ days of zero R-trail
                        # activations across all bull flag trades.
                        self.stop_monitor.upgrade_quote_to_stop_watch(
                            symbol=symbol,
                            stop_price=real_stop,
                            shares=actual_qty,
                            tp_leg_id=tp_leg_id,
                            sl_leg_id=sl_leg_id,
                            trade_db_id=trade_record['id'] if trade_record else None,
                            entry_price=fill_price,
                            risk_per_share=fill_price - real_stop,
                            trail_r=trail_r,
                            activate_at_r=activate_r,
                            avg_flag_volume=_flag_vol,
                            vol_confirmed_trail_enabled=self.vol_confirmed_trail_enabled,
                            vol_confirmed_trail_min_ratio=self.vol_confirmed_trail_min_ratio,
                            planned_entry_price=_planned_entry,
                            planned_risk_per_share=_planned_R,
                            r_basis=self.trail_r_basis,
                            # 2026-09-06 profit partial (shared spec) — 0 = off
                            pp_r_multiple=(self.profit_partial.r_multiple
                                           if self.profit_partial.enabled else 0.0),
                            pp_fraction=self.profit_partial.fraction,
                            pp_breakeven=self.profit_partial.move_to_breakeven,
                            # BT parity (2026-09-05): the fill minute is
                            # excluded from trail state AND stop checks —
                            # the simulate loop starts at entry+1. Broker
                            # safety-net SL still covers the minute.
                            skip_exits_until_ts=_end_of_minute_epoch(
                                datetime.now(timezone.utc)),
                        )

                        if self.profit_partial.enabled:
                            self.stop_monitor.arm_profit_partial(
                                symbol, self.profit_partial.r_multiple)

                        # Cancel TP leg when trailing stop is active (bracket path only)
                        if trail_r > 0 and tp_leg_id:
                            try:
                                self.alpaca.cancel_order(tp_leg_id)
                                with self.stop_monitor._watch_lock:
                                    w = self.stop_monitor._watches.get(symbol)
                                    if w:
                                        w.tp_leg_id = ''
                                logger.info(
                                    f"{symbol}: Cancelled TP leg — "
                                    f"trailing stop ({trail_r:.1f}R, +{activate_r:.1f}R) "
                                    f"replaces fixed TP"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"{symbol}: TP leg cancel failed: {e} — "
                                    f"TP may still fill before trail activates"
                                )

                        logger.info(
                            f"{symbol}: StopMonitor watching — "
                            f"real stop ${real_stop:.2f}, "
                            f"SL leg {sl_leg_id}"
                            f"{f', trail={trail_r:.1f}R' if trail_r > 0 else ''}"
                        )
                    except Exception as e:
                        error_msg = (
                            f"{symbol}: Failed to register with StopMonitor: {e} — "
                            f"safety-net SL on Alpaca is active"
                        )
                        logger.error(error_msg)
                        if self.notifier:
                            self.notifier.notify_error(error_msg, component="StopMonitor")

                self._daily_trade_count += 1
                self._patterns_traded += 1

                if self.notifier:
                    self.notifier.notify_order_submitted(
                        symbol=symbol,
                        order_id=order_id,
                        shares=actual_qty,
                        entry=fill_price or (plan.entry_price if plan else 0),
                    )

                last_fill_result = {
                    'order_id': order_id,
                    'status': 'filled',
                    'symbol': symbol,
                    'fill_price': fill_price,
                    'filled_qty': actual_qty,
                }

            elif status in ('cancelled', 'expired', 'rejected'):
                # Capture reject_reason from OrderStreamWatcher (only the
                # WebSocket trade-update event carries it; REST GET /orders
                # does NOT). See db Migration 13 + trading/order_stream.py
                # _order_to_status. NULL is acceptable — older rejections
                # without a captured reason still get logged.
                reject_reason = None
                try:
                    if (self.executor is not None
                            and getattr(self.executor, 'order_stream', None) is not None):
                        _ws_status = self.executor.order_stream.get_status(order_id)
                        if _ws_status:
                            reject_reason = _ws_status.get('reject_reason')
                except Exception as _rr_err:
                    logger.warning(
                        f"{symbol}: failed to read reject_reason from OrderStream: "
                        f"{_rr_err}"
                    )
                _rr_str = (
                    f" reason={reject_reason!r}" if reject_reason else ""
                )
                logger.info(
                    f"{symbol}: Pending order {status} — ID: {order_id}"
                    f"{_rr_str}"
                )
                # Persist status + reject_reason to the trade DB row so SQL
                # post-mortems work without REST archaeology.
                try:
                    trade_row = self.db.get_trade_by_order_id(order_id)
                    if trade_row:
                        update = {'order_status': status}
                        if reject_reason:
                            update['reject_reason'] = reject_reason
                        self.db.update_trade(trade_row['id'], update)
                except Exception as _db_err:
                    logger.warning(
                        f"{symbol}: failed to persist {status} status to DB: "
                        f"{_db_err}"
                    )
                symbols_to_remove.append(symbol)

            else:
                # Cancel pending orders in midday (11:30-14:00 ET).
                # Matches position_manager.can_open_position() and backtest.
                now_et = datetime.now(ET)
                current_minutes = now_et.hour * 60 + now_et.minute
                if 11 * 60 + 30 <= current_minutes < 14 * 60:
                    logger.info(f"{symbol}: Cancelling pending buy-stop — midday dead zone")
                    try:
                        self.alpaca.cancel_order(order_id)
                    except Exception as e:
                        logger.error(f"{symbol}: Failed to cancel midday order: {e}")
                    symbols_to_remove.append(symbol)
                    continue

                # Phase 5: Setup expiry — cancel stale buy-stops
                placed_at = pending.get('placed_at')
                if placed_at:
                    age = (datetime.now(timezone.utc) - placed_at).total_seconds()
                    if age > self.setup_expiry_seconds:
                        logger.info(f"{symbol}: Buy-stop EXPIRED after {age:.0f}s, cancelling")
                        # Fix 7: Refresh status before cancel — order may have filled.
                        # S1: hybrid uses the stream cache first (zero RTT when
                        # we have a fresh push) and only falls back to REST when
                        # the cache is cold.
                        refreshed = self._get_order_hybrid(order_id, submitted_at=placed_at)
                        if refreshed is not None:
                            if refreshed.get('status') == 'filled':
                                logger.info(f"{symbol}: Order filled while checking expiry — handling next cycle")
                                continue
                            elif refreshed.get('status') in ('cancelled', 'expired'):
                                logger.info(f"{symbol}: Order already {refreshed['status']}")
                                symbols_to_remove.append(symbol)
                                continue
                        try:
                            self.alpaca.cancel_order(order_id)
                        except Exception as e:
                            logger.error(f"{symbol}: Failed to cancel expired order: {e}")
                        symbols_to_remove.append(symbol)
                        # Remember expired breakout level to prevent re-detection
                        setup = pending.get('setup')
                        if setup:
                            self._invalidated_levels[symbol] = setup.breakout_level
                        continue

                # Still pending — check if setup invalidated
                setup = pending.get('setup')
                if setup:
                    try:
                        bars = self.alpaca.get_1min_bars(symbol, lookback_minutes=5)
                        if bars is not None and not bars.empty:
                            latest_low = bars.iloc[-1]['low']
                            if latest_low < setup.flag_low:
                                logger.info(
                                    f"{symbol}: Setup INVALIDATED — "
                                    f"low ${latest_low:.2f} < flag_low ${setup.flag_low:.2f}, "
                                    f"cancelling order {order_id}"
                                )
                                # Fix 7: Refresh status before cancel.
                                # S1: hybrid uses the stream cache first.
                                refreshed = self._get_order_hybrid(
                                    order_id, submitted_at=pending.get('placed_at')
                                )
                                if refreshed is not None:
                                    if refreshed.get('status') == 'filled':
                                        logger.info(f"{symbol}: Order filled while checking invalidation — handling next cycle")
                                        continue
                                    elif refreshed.get('status') in ('cancelled', 'expired'):
                                        logger.info(f"{symbol}: Order already {refreshed['status']}")
                                        symbols_to_remove.append(symbol)
                                        continue
                                self.alpaca.cancel_order(order_id)
                                symbols_to_remove.append(symbol)
                                # Remember invalidated breakout level to prevent re-detection loop
                                self._invalidated_levels[symbol] = setup.breakout_level
                    except Exception as e:
                        logger.error(f"{symbol}: Failed to check invalidation: {e}")

        for symbol in symbols_to_remove:
            pending = self._pending_orders.pop(symbol, None)
            # Update DB record so cancelled orders don't show as "open positions"
            if pending:
                order_id = pending.get('order_id')
                if order_id:
                    trade_record = self.db.get_trade_by_order_id(order_id)
                    if trade_record and trade_record.get('fill_price') is None:
                        # Don't clobber a more specific terminal status
                        # ('rejected'/'expired') that the elif branch above
                        # already persisted with reject_reason. Only mark
                        # 'cancelled' when status is still in a non-terminal
                        # state (e.g., we're cleaning up an invalidation,
                        # midday-pause cancel, or expiry-by-our-cancel).
                        current_status = (trade_record.get('order_status')
                                          or '').lower()
                        if current_status not in ('rejected', 'expired'):
                            self.db.update_trade(trade_record['id'], {
                                'order_status': 'cancelled',
                            })
                            logger.debug(f"{symbol}: DB trade record marked cancelled")

            # Clean up quote watch on cancellation
            if self.stop_monitor:
                self.stop_monitor.remove_quote_watch(symbol)

        return last_fill_result

    def _try_get_fill(self, event, max_polls: int = 2) -> Optional[float]:
        """
        Poll exit order for fill price. Updates event.filled_qty.

        Checks filled_qty against expected shares. If partial fill detected,
        emergency-closes remaining via close_position().

        Args:
            event: StopExitEvent with order_id (mutated: filled_qty set)
            max_polls: Number of poll attempts (0.5s apart)

        Returns:
            Actual fill price, or None if not yet filled
        """
        if not event.order_id:
            return None
        # Convert StopExitEvent.submitted_at (Unix float) to a tz-aware datetime
        # for the hybrid age gate. Missing/zero → None → hybrid skips the gate.
        submitted_dt: Optional[datetime] = None
        evt_ts = getattr(event, 'submitted_at', 0.0)
        if evt_ts:
            try:
                submitted_dt = datetime.fromtimestamp(float(evt_ts), tz=timezone.utc)
            except Exception:
                submitted_dt = None
        for _ in range(max_polls):
            time_mod.sleep(0.5)
            # S1: stream-first (no network RTT when push has delivered).
            # fallback_after_s=0 disables the age gate — this is a tight poll
            # loop (max_polls × 0.5s = 1-5s total) where we NEED REST on every
            # stream miss, not a wait for the default 5s gate.
            exit_order = self._get_order_hybrid(
                event.order_id, submitted_at=submitted_dt, fallback_after_s=0.0
            )
            if exit_order is None:
                continue
            if exit_order.get('status') == 'filled':
                fill = exit_order.get('filled_avg_price')
                filled_qty = int(exit_order.get('filled_qty', 0) or 0)

                if fill is not None:
                    # Check for partial fill
                    if filled_qty > 0 and filled_qty < event.shares:
                        remaining = event.shares - filled_qty
                        logger.error(
                            f"{event.symbol}: PARTIAL FILL on exit — "
                            f"{filled_qty}/{event.shares} filled, "
                            f"{remaining} shares UNPROTECTED"
                        )
                        blended = self._handle_exit_partial_fill(
                            event.symbol, fill, filled_qty,
                            event.shares, remaining
                        )
                        event.filled_qty = event.shares  # all shares now closed
                        return blended
                    else:
                        event.filled_qty = filled_qty or event.shares
                        logger.info(
                            f"{event.symbol}: exit filled at ${fill:.2f} "
                            f"({event.filled_qty}sh, {event.pricing_method})"
                        )
                        return fill
        return None

    def _handle_exit_partial_fill(
        self, symbol: str, first_fill_price: float, first_qty: int,
        total_expected: int, remaining: int,
    ) -> float:
        """
        Handle partial fill on exit: emergency-close remaining shares.

        Args:
            symbol: Stock symbol
            first_fill_price: Fill price from the partial fill
            first_qty: Shares filled in the first order
            total_expected: Total shares we expected to sell
            remaining: Shares still open

        Returns:
            Blended average fill price across all fills
        """
        logger.warning(
            f"{symbol}: emergency closing {remaining} remaining shares"
        )
        try:
            close_result = self.alpaca.close_position(symbol)
            close_id = close_result.get('id', '')
            close_submitted_at = datetime.now(timezone.utc)
            for _ in range(10):
                time_mod.sleep(0.5)
                # S1: hybrid — stream-first, REST on every miss.
                # fallback_after_s=0 disables the default age gate (this is a
                # 5-second tight loop where stream-miss must go straight to REST).
                close_order = self._get_order_hybrid(
                    close_id, submitted_at=close_submitted_at, fallback_after_s=0.0
                )
                if close_order is None:
                    continue
                if close_order.get('status') == 'filled':
                    close_price = float(
                        close_order.get('filled_avg_price', 0) or 0
                    )
                    close_qty = int(
                        close_order.get('filled_qty', 0) or 0
                    )
                    total = first_qty + close_qty
                    blended = (
                        (first_fill_price * first_qty + close_price * close_qty)
                        / total
                    ) if total > 0 else first_fill_price
                    logger.info(
                        f"{symbol}: partial fill resolved — "
                        f"{first_qty}@${first_fill_price:.2f} + "
                        f"{close_qty}@${close_price:.2f} = "
                        f"${blended:.2f} blended"
                    )
                    return blended

            logger.error(
                f"{symbol}: emergency close fill unknown — "
                f"using first fill ${first_fill_price:.2f}"
            )
        except Exception as e:
            logger.error(
                f"{symbol}: emergency close FAILED: {e} — "
                f"{remaining} shares may be orphaned!"
            )
            if self.notifier:
                self.notifier.notify_error(
                    f"{symbol}: PARTIAL FILL — {remaining} shares ORPHANED! "
                    f"Manual intervention required.",
                    component="PartialFill",
                )

        return first_fill_price

    def _cancel_and_market_sell(self, event) -> None:
        """
        Cancel unfilled limit exit and force market sell after timeout.

        Args:
            event: StopExitEvent whose limit order timed out
        """
        symbol = event.symbol
        logger.warning(
            f"{symbol}: exit limit order UNFILLED after 30s — "
            f"cancelling and market-selling"
        )

        # Cancel the limit order (may have filled in the meantime)
        actual_fill = None
        try:
            self.alpaca.cancel_order(event.order_id)
        except Exception:
            pass  # 422 = already filled/cancelled

        # Check if it filled (fully or partially) during cancel race.
        # S1: hybrid is ideal here — push-delivered status ends the race cleanly.
        evt_ts = getattr(event, 'submitted_at', 0.0)
        evt_submitted_dt: Optional[datetime] = None
        if evt_ts:
            try:
                evt_submitted_dt = datetime.fromtimestamp(float(evt_ts), tz=timezone.utc)
            except Exception:
                evt_submitted_dt = None
        order = self._get_order_hybrid(event.order_id, submitted_at=evt_submitted_dt)
        if order is not None and order.get('status') == 'filled':
            actual_fill = order.get('filled_avg_price')
            filled_qty = int(order.get('filled_qty', 0) or 0)
            if actual_fill:
                # Check for partial fill during cancel
                if filled_qty > 0 and filled_qty < event.shares:
                    remaining = event.shares - filled_qty
                    logger.warning(
                        f"{symbol}: partial fill during cancel — "
                        f"{filled_qty}/{event.shares}, emergency closing {remaining}"
                    )
                    actual_fill = self._handle_exit_partial_fill(
                        symbol, actual_fill, filled_qty,
                        event.shares, remaining
                    )
                event.filled_qty = event.shares  # all closed now
                logger.info(
                    f"{symbol}: limit order filled during cancel — "
                    f"${actual_fill:.2f}"
                )

        if actual_fill is None:
            # Market sell via close_position (closes entire remaining position)
            try:
                fallback = self.alpaca.close_position(symbol)
                fallback_id = fallback.get('id', '')
                fallback_submitted_at = datetime.now(timezone.utc)
                # Poll for market fill (S1: hybrid — stream-first).
                # fallback_after_s=0 disables age gate for this tight loop —
                # market fills happen in <1s, we want REST on any stream miss.
                for _ in range(10):
                    time_mod.sleep(0.5)
                    fb_order = self._get_order_hybrid(
                        fallback_id,
                        submitted_at=fallback_submitted_at,
                        fallback_after_s=0.0,
                    )
                    if fb_order is None:
                        continue
                    if fb_order.get('status') == 'filled':
                        actual_fill = fb_order.get('filled_avg_price')
                        fb_filled = int(fb_order.get('filled_qty', 0) or 0)
                        if fb_filled > 0:
                            event.filled_qty = fb_filled
                        break
                if actual_fill is None:
                    actual_fill = event.exit_price  # last resort
                    logger.error(
                        f"{symbol}: market sell fill unknown — using limit "
                        f"${event.exit_price:.2f} as estimate"
                    )
                else:
                    logger.info(
                        f"{symbol}: market sell filled at ${actual_fill:.2f} "
                        f"({event.filled_qty}sh)"
                    )
                event.pricing_method = f"{event.pricing_method}+market_fallback"
            except Exception as e:
                logger.error(f"{symbol}: market sell also failed: {e}")
                actual_fill = event.exit_price
                if self.notifier:
                    self.notifier.notify_error(
                        f"{symbol}: EXIT FAILED — position may still be open! "
                        f"Manual intervention required.",
                        component="ExitFailure",
                    )

        # Finalize with the fill we got
        self._finalize_stop_exit(event, actual_fill)

    STOP_EXIT_TIMEOUT_SECONDS = 10  # 10s: speed > price improvement on stop exits

    def _process_stop_monitor_exits(self) -> None:
        """Drain and process exit events from StopMonitor."""
        if not self.stop_monitor:
            return

        # 1. Drain new events from queue — filter to bull-flag-tagged events
        # so MACD wave's events stay queued for its own engine to consume.
        events = self.stop_monitor.drain_exit_events(strategy='bull_flag')
        for event in events:
            # Partials (exhaustion / profit) are processed by their own
            # checkers, which receive the event directly — never as a full exit.
            if event.exit_reason in ('exhaustion_partial', 'profit_partial'):
                continue

            # Unconfirmed exits (BRANCH_LAST_RESORT — no broker fill report)
            # short-circuit to the pending-verification state. Trying to
            # poll fill on these is useless: the StopMonitor already
            # exhausted its retries. The orphan reconciler handles them.
            if not getattr(event, 'confirmed', True):
                # 2026-07-04 review fix: if the trade is ALREADY closed
                # (sync's Layer-2 order-history recovery beat this event
                # during a slow escalation), do NOT clobber the correct
                # row with pending-verification state.
                already_closed = False
                if event.trade_db_id:
                    try:
                        open_ids = {t.get('id') for t in
                                    self.db.get_open_trades(
                                        date.today().isoformat())}
                        already_closed = event.trade_db_id not in open_ids
                    except Exception as e:
                        logger.warning(
                            f"{event.symbol}: open-trade check before "
                            f"unconfirmed write failed: {e} — writing anyway")
                if already_closed:
                    logger.warning(
                        f"{event.symbol}: UNCONFIRMED EXIT event arrived but "
                        f"trade {event.trade_db_id} is already closed "
                        f"(recovered via order history) — skipping "
                        f"pending-verification write")
                    continue
                logger.error(
                    f"{event.symbol}: UNCONFIRMED EXIT — writing "
                    f"order_status=exit_pending_verification, "
                    f"exit_reason={event.exit_reason}. Reconciler will retry."
                )
                if event.trade_db_id:
                    try:
                        self.db.update_trade(event.trade_db_id,
                                             build_exit_update(event))
                    except Exception as e:
                        logger.error(
                            f"{event.symbol}: pending-verification DB "
                            f"write failed: {e}"
                        )
                continue

            logger.info(
                f"{event.symbol}: StopMonitor exit — "
                f"stop=${event.stop_price:.2f}, exit=${event.exit_price:.2f}, "
                f"reason={event.exit_reason}, {event.pricing_method}, "
                f"order={event.order_id}"
            )

            # Quick fill check (2 polls × 0.5s = 1s)
            actual_fill = self._try_get_fill(event, max_polls=2)
            if actual_fill:
                self._finalize_stop_exit(event, actual_fill)
            else:
                self._pending_stop_exits[event.symbol] = event
                logger.info(
                    f"{event.symbol}: exit order pending fill — "
                    f"will timeout after {self.STOP_EXIT_TIMEOUT_SECONDS}s"
                )

        # 2. Check pending orders for fill or timeout
        if self._pending_stop_exits:
            self._check_pending_stop_exit_timeouts()

    def _check_pending_stop_exit_timeouts(self) -> None:
        """Check pending stop exit orders for fill or timeout, blocking until resolved."""
        start = time_mod.time()
        while self._pending_stop_exits and (time_mod.time() - start < 35):
            for symbol in list(self._pending_stop_exits.keys()):
                event = self._pending_stop_exits[symbol]
                actual_fill = self._try_get_fill(event, max_polls=1)
                if actual_fill:
                    del self._pending_stop_exits[symbol]
                    self._finalize_stop_exit(event, actual_fill)
                elif time_mod.time() - event.submitted_at > self.STOP_EXIT_TIMEOUT_SECONDS:
                    del self._pending_stop_exits[symbol]
                    self._cancel_and_market_sell(event)

            if self._pending_stop_exits:
                time_mod.sleep(3)

    def _finalize_stop_exit(self, event, actual_exit_price: float) -> None:
        """Finalize a stop exit: update DB, record P&L, notify Telegram."""
        if not event.trade_db_id:
            return

        try:
            trades_today = self.db.get_open_trades(date.today().isoformat(), strategy='bull_flag')
            trade_record = None
            for t in trades_today:
                if t['id'] == event.trade_db_id:
                    trade_record = t
                    break

            if trade_record and trade_record.get('fill_price'):
                # Use actual filled_qty from broker, not expected shares
                exit_qty = event.filled_qty if event.filled_qty > 0 else event.shares

                # Remainder P&L: actual shares sold in this exit × price diff
                remainder_pnl = (actual_exit_price - trade_record['fill_price']) * exit_qty

                # Combine with partial exit P&L if exhaustion partial was taken
                partial_pnl = trade_record.get('partial_exit_pnl') or 0.0
                partial_shares = trade_record.get('partial_exit_shares') or 0
                pnl = remainder_pnl + partial_pnl

                # P&L % based on total capital deployed (entry_price × total_shares)
                total_shares = trade_record.get('filled_qty') or trade_record['shares']
                pnl_pct = (pnl / (trade_record['fill_price'] * total_shares)) * 100

                exit_reason = event.exit_reason
                if partial_pnl != 0.0:
                    exit_reason = f"exhaust+{event.exit_reason}"

                # Compute exit microstructure metrics
                exit_spread = (event.exit_quote_ask - event.exit_quote_bid
                               if event.exit_quote_ask > 0 else None)
                exit_slippage = (event.exit_limit_price - actual_exit_price
                                 if event.exit_limit_price > 0 else None)
                exit_latency = ((time_mod.time() - event.submitted_at) * 1000
                                if event.submitted_at > 0 else None)
                exit_submitted = (datetime.fromtimestamp(event.submitted_at, tz=timezone.utc)
                                  if event.submitted_at > 0 else None)

                self.db.update_trade(event.trade_db_id, {
                    'exit_price': actual_exit_price,
                    'exit_reason': exit_reason,
                    'exited_at': datetime.now(timezone.utc),
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    # Exit microstructure
                    'exit_trigger_price': event.exit_trigger_price or None,
                    'exit_quote_bid': event.exit_quote_bid or None,
                    'exit_quote_ask': event.exit_quote_ask or None,
                    'exit_quote_bid_size': event.exit_quote_bid_size or None,
                    'exit_quote_ask_size': event.exit_quote_ask_size or None,
                    'exit_quote_spread': exit_spread,
                    'exit_limit_price': event.exit_limit_price or None,
                    'exit_pricing_method': event.pricing_method,
                    'exit_submitted_at': exit_submitted,
                    'exit_fill_latency_ms': exit_latency,
                    'exit_slippage': exit_slippage,
                    'exit_ofi': event.exit_ofi if hasattr(event, 'exit_ofi') and event.exit_ofi != 0 else None,
                })
                self.position_manager.record_trade_pnl(pnl)
                logger.info(
                    f"{event.symbol}: exit finalized — "
                    f"P&L ${pnl:+,.2f} ({pnl_pct:+.1f}%)"
                    f"{f' (partial ${partial_pnl:+,.2f} + remainder ${remainder_pnl:+,.2f})' if partial_pnl != 0 else ''}"
                )

                # Log L2 order book depth at stop trigger time (non-blocking)
                # Log exit L2 async (use trigger time, not post-fill)
                try:
                    from data_sources.l2_depth import log_l2_async
                    trigger_dt = (datetime.fromtimestamp(event.submitted_at, tz=timezone.utc)
                                  if event.submitted_at > 0 else datetime.now(timezone.utc))
                    log_l2_async(event.symbol, trigger_dt, event.trade_db_id,
                                 self.db.update_trade, column='exit_l2_depth')
                except Exception as e:
                    logger.debug(f"{event.symbol}: Exit L2 async launch failed: {e}")

                if self.notifier:
                    self.notifier.notify_position_closed(
                        symbol=event.symbol,
                        entry_price=trade_record['fill_price'],
                        exit_price=actual_exit_price,
                        shares=event.shares,
                        pnl=pnl,
                        exit_reason=exit_reason,
                    )
            else:
                logger.warning(
                    f"{event.symbol}: StopMonitor exit — "
                    f"no matching open trade for DB id {event.trade_db_id}"
                )
        except Exception as e:
            logger.error(
                f"{event.symbol}: Failed to finalize stop exit: {e}"
            )

    def _check_profit_partials(self) -> None:
        """Execute armed profit partials whose closed-bar high reached the
        level (trading/bf_profit_partial.py — the BT simulator applies the
        identical rule). Uses the same partial-sell executor as the
        exhaustion rule; the stop then moves to the fill (breakeven).
        Cadence: this cycle (~60s) after the bar closed — BT fills at that
        bar's close, live at the next quote; the DECISION is identical."""
        if not self.stop_monitor or not self.profit_partial.enabled:
            return
        for symbol in self.stop_monitor.pending_profit_partials('bull_flag'):
            snapshot = self.stop_monitor.get_watch_snapshot(symbol)
            if not snapshot:
                continue
            if self.profit_partial.shadow:
                logger.info(
                    f"{symbol}: PROFIT PARTIAL [SHADOW] would sell "
                    f"{self.profit_partial.fraction:.0%} — closed-bar high "
                    f"${snapshot.get('highest_since_entry', 0):.2f} reached +"
                    f"{self.profit_partial.r_multiple}R (no order; shadow window)"
                )
                self.stop_monitor.mark_profit_partial_shadow(symbol)
                continue
            logger.info(
                f"{symbol}: PROFIT PARTIAL fired — closed-bar high "
                f"${snapshot.get('highest_since_entry', 0):.2f} >= level, "
                f"selling {self.profit_partial.fraction:.0%}"
            )
            event = self.stop_monitor.execute_partial_exit(
                symbol,
                fraction=self.profit_partial.fraction,
                tighter_trail_r=float(snapshot.get('trail_r') or 0.0),  # keep the trail
                reason='profit_partial',
            )
            if event:
                self._process_exhaustion_partial_event(event)

    def _check_exhaustion_exits(self) -> None:
        """
        Check active positions for exhaustion exit signals.

        Called every 60s from run_pattern_check(). For each watched symbol:
        1. Get snapshot → skip if partial already taken
        2. Compute current R from latest bar close
        3. Skip if current_r < min_profit_r (3.0)
        4. Fetch last 10 bars, drop last (in-progress)
        5. Run check_exhaustion() on the completed bar
        6. If fired: execute_partial_exit() via StopMonitor
        7. Process partial exit event (poll fill, update DB, notify)
        """
        if not self.exhaustion_exit_enabled or not self.stop_monitor:
            return

        from trading.exhaustion_signals import check_exhaustion

        # Scope to bull-flag watches ONLY. The shared StopMonitor also holds
        # watches added by ORB and MACD Wave; iterating those here would
        # make bull flag try to exit ORB positions on the main Alpaca
        # account and log 42210000 rejections (the 2026-04-24 SMCX bug).
        watched = self.stop_monitor.watched_symbols_for('bull_flag')
        if not watched:
            return

        for symbol in watched:
            snapshot = self.stop_monitor.get_watch_snapshot(symbol)
            if not snapshot:
                continue

            if snapshot['exhaustion_partial_taken']:
                continue
            if snapshot.get('pp_taken'):
                # BT parity: after the +2R profit partial the remainder keeps
                # the unified trail only — the simulator skips the exhaustion
                # partial once pp_taken (backtest.py main walk).
                continue

            entry_price = snapshot['entry_price']
            risk = snapshot['risk_per_share']
            if risk <= 0 or entry_price <= 0:
                continue

            # Fetch recent bars
            try:
                bars = self.alpaca.get_1min_bars(symbol, lookback_minutes=10)
            except Exception as e:
                logger.error(f"{symbol}: exhaustion check — bar fetch failed: {e}")
                continue

            if bars is None or len(bars) < 3:
                continue

            # Drop last bar (in-progress, incomplete)
            bars = bars.iloc[:-1]
            if bars.empty:
                continue

            # Compute R from latest completed bar close
            latest_close = float(bars.iloc[-1]['close'])
            current_r = (latest_close - entry_price) / risk

            if current_r < self.exhaustion_min_profit_r:
                continue

            # Run signal detection on last completed bar
            check_idx = len(bars) - 1
            if check_exhaustion(bars, check_idx, self.exhaustion_signals):
                logger.info(
                    f"{symbol}: EXHAUSTION signal at +{current_r:.1f}R "
                    f"(close=${latest_close:.2f}) — executing partial exit"
                )
                event = self.stop_monitor.execute_partial_exit(
                    symbol=symbol,
                    fraction=self.exhaustion_partial_fraction,
                    tighter_trail_r=self.exhaustion_tighter_trail_r,
                )
                if event:
                    self._process_exhaustion_partial_event(event)

    def _process_exhaustion_partial_event(self, event) -> None:
        """
        Process an exhaustion partial exit event: update DB, notify.

        Fill is already confirmed by StopMonitor.execute_partial_exit()
        (waits up to 30s for fill before emitting event). event.exit_price
        is the actual fill price.

        Args:
            event: StopExitEvent from execute_partial_exit()
        """
        actual_exit_price = event.exit_price

        # Update DB with partial exit details
        if event.trade_db_id:
            try:
                trade_record = None
                trades_today = self.db.get_open_trades(date.today().isoformat(), strategy='bull_flag')
                for t in trades_today:
                    if t['id'] == event.trade_db_id:
                        trade_record = t
                        break

                if trade_record and trade_record.get('fill_price'):
                    partial_pnl = (
                        (actual_exit_price - trade_record['fill_price'])
                        * event.shares
                    )
                    self.db.update_trade(event.trade_db_id, {
                        'partial_exit_price': actual_exit_price,
                        'partial_exit_shares': event.shares,
                        'partial_exit_pnl': partial_pnl,
                        'partial_exit_reason': ('profit_partial' if event.exit_reason == 'profit_partial'
                                            else 'exhaustion'),  # partial-exit reason has its own column; ExitReason enum tracks full-exit only
                        'partial_exited_at': datetime.now(timezone.utc),
                    })
                    logger.info(
                        f"{event.symbol}: exhaustion partial DB updated — "
                        f"{event.shares}sh @ ${actual_exit_price:.2f}, "
                        f"partial P&L ${partial_pnl:+,.2f}"
                    )

                    # Telegram notification
                    if self.notifier:
                        self.notifier.notify_position_closed(
                            symbol=event.symbol,
                            entry_price=trade_record['fill_price'],
                            exit_price=actual_exit_price,
                            shares=event.shares,
                            pnl=partial_pnl,
                            exit_reason=ExitReason.EXHAUSTION_PARTIAL.value,
                        )
                else:
                    logger.warning(
                        f"{event.symbol}: exhaustion partial — "
                        f"no matching open trade for DB id {event.trade_db_id}"
                    )
            except Exception as e:
                logger.error(
                    f"{event.symbol}: Failed to process exhaustion partial: {e}"
                )

    def _recover_exit_from_order_history(
        self, symbol: str, fill_price: float,
        planned_stop: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[str]]:
        """Query Alpaca's recent closed orders for ``symbol`` and return
        ``(exit_price, exit_reason)`` for the actual sell that closed the
        position.

        Mirrors the proven MACD wave pattern
        (``macd_wave_engine.py:339-381``). Used by
        ``_sync_closed_positions`` when the bracket-leg lookup can't find
        a fill (e.g., StopMonitor's market close fired AFTER we cancelled
        both legs — GLXG 2026-06-11 case). Pre-fix this path wrote
        ``exit_reason='unknown_exit'`` with ``pnl=$0`` placeholder,
        silently losing the real P&L.

        Classification of ``exit_reason``:
          - ``order_class in {bracket, oto, oco}`` → ``'bracket_sl_tp'``
          - solo sell within 0.5% of planned_stop (or below) → ``'stop_loss'``
          - solo sell ≥ fill_price (small win / breakeven) → ``'trail_stop'``
          - solo sell between fill_price and stop → ``'stop_loss_market_fallback'``
          - planned_stop unknown → ``'stopmonitor_exit'``

        Returns ``(None, None)`` when no filled sell can be recovered
        (API error, empty list, or only cancelled/expired orders).
        Caller should fall through to the ``unknown_exit`` placeholder
        only in that case.
        """
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            orders = self.alpaca.trading_client.get_orders(
                GetOrdersRequest(
                    status=QueryOrderStatus.CLOSED,
                    symbols=[symbol], limit=10,
                )
            )
        except Exception as e:
            logger.warning(
                f"{symbol}: order-history fetch failed: {e}"
            )
            return None, None

        classified_sell = None
        for o in orders or []:
            try:
                if (o.side.value == 'sell'
                        and o.status.value == 'filled'
                        and o.filled_avg_price):
                    classified_sell = o
                    break
            except Exception:
                continue

        if classified_sell is None:
            return None, None

        try:
            exit_price = float(classified_sell.filled_avg_price)
        except Exception:
            return None, None

        # Classify exit_reason — same shape as MACD wave's discriminator.
        try:
            oc = (classified_sell.order_class.value
                   if classified_sell.order_class else None)
        except Exception:
            oc = None

        if oc in ('bracket', 'oto', 'oco'):
            exit_reason = 'bracket_sl_tp'
        elif (planned_stop and planned_stop > 0
                and fill_price and fill_price > 0):
            # Solo sell — discriminate by where exit landed relative to
            # the planned stop. Tolerance: 0.5% of stop covers normal
            # slippage on a marketable limit fill.
            tolerance = planned_stop * 0.005
            if exit_price <= planned_stop + tolerance:
                exit_reason = 'stop_loss'
            elif exit_price >= fill_price:
                # Sold at or above entry — trail caught a small win or
                # breakeven exit.
                exit_reason = 'trail_stop'
            else:
                # Between planned stop and entry: limit didn't fill,
                # escalate-to-market caught at a worse price than entry
                # but better than the planned stop.
                exit_reason = 'stop_loss_market_fallback'
        else:
            exit_reason = 'stopmonitor_exit'

        return exit_price, exit_reason

    def _sync_closed_positions(self) -> None:
        """Detect bracket exits (SL/TP hit) and update DB + circuit breaker."""
        # Process StopMonitor exits first — updates DB with exit_price.
        # Must happen BEFORE we fetch open_trades, otherwise trades just
        # closed by StopMonitor still appear as "open" and get double-processed.
        self._process_stop_monitor_exits()

        # Check exhaustion exit signals on active positions (every 60s cycle)
        self._check_profit_partials()
        self._check_exhaustion_exits()

        today = date.today().isoformat()
        open_trades = self.db.get_open_trades(today, strategy='bull_flag')
        if not open_trades:
            return

        try:
            alpaca_positions = {p['symbol'] for p in self.alpaca.get_open_positions()}
        except Exception as e:
            error_msg = f"Failed to sync positions: {e}"
            logger.error(error_msg)
            if self.notifier:
                self.notifier.notify_error(error_msg, component="PositionSync")
            return

        for trade in open_trades:
            symbol = trade['symbol']
            if symbol not in alpaca_positions and trade.get('fill_price'):
                try:
                    order_id = trade.get('order_id')
                    exit_price = None
                    exit_reason = None
                    if order_id:
                        order_detail = self.alpaca.get_order(order_id)
                        sl_leg, tp_leg = self._identify_bracket_legs(
                            order_detail.get('legs', []),
                            expected_sl=trade.get('stop_loss_price'),
                            expected_tp=trade.get('take_profit_price'),
                        )
                        # Check SL leg
                        if sl_leg and sl_leg.get('status') == 'filled':
                            fill = sl_leg.get('filled_avg_price')
                            exit_price = fill or sl_leg['stop_price']
                            exit_reason = ExitReason.STOP_LOSS.value
                        # Check TP leg
                        elif tp_leg and tp_leg.get('status') == 'filled':
                            fill = tp_leg.get('filled_avg_price')
                            exit_price = fill or tp_leg['limit_price']
                            exit_reason = ExitReason.TAKE_PROFIT.value

                    if exit_price:
                        # Remaining shares after any partial exit
                        total_shares = trade.get('filled_qty') or trade['shares']
                        partial_shares = trade.get('partial_exit_shares') or 0
                        remainder_shares = total_shares - partial_shares if partial_shares else total_shares
                        remainder_pnl = (exit_price - trade['fill_price']) * remainder_shares

                        # Combine with partial P&L
                        partial_pnl = trade.get('partial_exit_pnl') or 0.0
                        pnl = remainder_pnl + partial_pnl
                        pnl_pct = (pnl / (trade['fill_price'] * total_shares)) * 100

                        if partial_pnl != 0.0:
                            exit_reason = f"exhaust+{exit_reason}"

                        self.db.update_trade(trade['id'], {
                            'exit_price': exit_price,
                            'exit_reason': exit_reason,
                            'exited_at': datetime.now(timezone.utc),
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                        })
                        self.position_manager.record_trade_pnl(pnl)
                        # Remove StopMonitor watch — position is gone (TP or
                        # safety-net SL filled on Alpaca side)
                        if self.stop_monitor:
                            self.stop_monitor.remove_watch(symbol)
                        logger.info(
                            f"{symbol}: {exit_reason} — exit ${exit_price:.2f}, "
                            f"P&L ${pnl:+,.2f} ({pnl_pct:+.1f}%)"
                        )
                    else:
                        # GLXG 2026-06-11 — two complementary layers, merged
                        # 2026-07-03 from parallel fixes to the same incident:
                        #
                        # Layer 1 (race guard): if StopMonitor is currently
                        # mid-exit on this symbol (limit poll, market-close
                        # escalation, SL-leg recovery), DEFER entirely. The
                        # real exit event lands with the ACTUAL fill price —
                        # better than any reconstruction. 60s staleness cutoff
                        # inside is_exit_in_progress keeps a stuck flow from
                        # suppressing reconciliation forever.
                        if (self.stop_monitor is not None
                                and self.stop_monitor.is_exit_in_progress(symbol)):
                            logger.info(
                                f"{symbol}: Position closed but StopMonitor exit "
                                f"in-progress — deferring reconcile "
                                f"(retry next sync iteration)"
                            )
                            continue
                        # Layer 2 (order-history recovery): the position
                        # closed via a path we didn't attribute (StopMonitor
                        # flow already finished, manual sell, external close).
                        # Mirror MACD wave's sync external-close pattern
                        # (macd_wave_engine.py:339-381) and query Alpaca's
                        # recent closed orders for the actual sell fill.
                        # Pre-fix this branch wrote exit_reason='unknown_exit'
                        # / pnl=$0, losing the real P&L (GLXG 2026-06-11).
                        recovered_price, recovered_reason = (
                            self._recover_exit_from_order_history(
                                symbol,
                                trade['fill_price'],
                                planned_stop=trade.get('stop_loss_price'),
                            )
                        )
                        if recovered_price is not None:
                            total_shares = (trade.get('filled_qty')
                                              or trade['shares'])
                            partial_shares = trade.get('partial_exit_shares') or 0
                            remainder_shares = (
                                total_shares - partial_shares
                                if partial_shares else total_shares
                            )
                            remainder_pnl = (
                                (recovered_price - trade['fill_price'])
                                * remainder_shares
                            )
                            partial_pnl = trade.get('partial_exit_pnl') or 0.0
                            pnl_recovered = remainder_pnl + partial_pnl
                            pnl_pct = (
                                (pnl_recovered
                                  / (trade['fill_price'] * total_shares))
                                * 100
                            )
                            final_reason = (
                                f"exhaust+{recovered_reason}"
                                if partial_pnl != 0.0 else recovered_reason
                            )
                            self.db.update_trade(trade['id'], {
                                'exit_price': recovered_price,
                                'exit_reason': final_reason,
                                'exited_at': datetime.now(timezone.utc),
                                'pnl': pnl_recovered,
                                'pnl_pct': pnl_pct,
                            })
                            self.position_manager.record_trade_pnl(
                                pnl_recovered)
                            if self.stop_monitor:
                                self.stop_monitor.remove_watch(symbol)
                            logger.info(
                                f"{symbol}: {final_reason} — exit "
                                f"${recovered_price:.2f} (recovered from "
                                f"order history), P&L ${pnl_recovered:+,.2f} "
                                f"({pnl_pct:+.1f}%)"
                            )
                            continue

                        # Truly unrecoverable. Use fill_price as fallback
                        # exit to prevent infinite re-check (exit_price IS
                        # NULL keeps this trade in get_open_trades forever).
                        fallback_exit = trade['fill_price']
                        pnl_est = 0.0  # Assume breakeven if unknown
                        error_msg = (
                            f"{symbol}: Position closed but exit price unknown — "
                            f"using fill_price ${fallback_exit:.2f} as estimate "
                            f"(order-history recovery also failed)"
                        )
                        logger.warning(error_msg)
                        if self.notifier:
                            self.notifier.notify_error(error_msg, component="PositionSync")
                        # UNKNOWN_EXIT is the documented leak signal — every
                        # row in DB with this value means a code path failed
                        # to attribute the close. See trading/exit_reasons.py
                        # and needs_reconcile().
                        self.db.update_trade(trade['id'], {
                            'exit_price': fallback_exit,
                            'exit_reason': ExitReason.UNKNOWN_EXIT.value,
                            'exited_at': datetime.now(timezone.utc),
                            'pnl': pnl_est,
                            'pnl_pct': 0.0,
                        })
                except Exception as e:
                    error_msg = f"{symbol}: Failed to process closed position: {e}"
                    logger.error(error_msg)
                    if self.notifier:
                        self.notifier.notify_error(error_msg, component="PositionSync")

    def run_pattern_check(self) -> Optional[Dict[str, Any]]:
        """
        Run one pattern detection cycle on all qualified symbols.

        Flow:
        1. Sync closed positions (detect bracket exits)
        2. Manage pending buy-stop orders (check fills, invalidations)
        3. For each qualified symbol without a pending/filled order:
           a. Fetch 1-min bars
           b. Run bull flag setup detection
           c. If setup found, create plan and submit buy-stop bracket order
        4. If past last_entry_time, skip new order placement

        Returns:
            Dict with order details if a trade was executed, None otherwise
        """
        if not self.enabled:
            return None

        # Drain real-time bar events FIRST (from WebSocket thread, via queue)
        rt_result = self._drain_bar_events()

        # Marginability cache is per-day (cleared in reset_daily). Marginability
        # is stable within a session — clearing per-cycle was wasteful, costing
        # ~150ms per qualified symbol per cycle (60 cycles/hour = 60 redundant
        # API calls per symbol per hour). Pre-warm via on_stock_qualified
        # (Fix 4, 2026-05-01) means the cache hit rate at the entry path is
        # ~100% in normal operation.

        # ALWAYS sync positions and manage pending orders — these must run
        # regardless of regime filter or max trades. Skipping them means
        # SL/TP exits go unrecorded, PnL is wrong, and circuit breaker is deaf.
        self._sync_closed_positions()
        fill_result = self._manage_pending_orders()

        # BF kill rails (Discipline Program Phase 1): DB-derived realized-P&L
        # gates. Daily/weekly/month breach or a fail-closed pnl-query error
        # blocks all NEW entries (weekly also flattens, month pauses).
        # Placed AFTER sync/fill management so exits are never blocked.
        if self._kill_rails_blocked():
            return fill_result

        # Friday filter — blocks NEW order placement only
        if self.skip_fridays and date.today().weekday() == 4:
            logger.info("FRIDAY FILTER: skipping new trades (30% WR on Fridays)")
            return fill_result

        # Market regime filter — blocks NEW order placement only
        if self.market_regime and not self.market_regime.is_regime_ok(date.today()):
            info = self.market_regime.get_regime_info(date.today())
            vol_str = f"{info['vol_5d']:.2f}%" if info['vol_5d'] is not None else "N/A"
            logger.warning(
                f"REGIME FILTER: vol_5d={vol_str} > {self.market_regime.vol_threshold}% "
                f"AND below SMA{self.market_regime.sma_period} — skipping new trades"
            )
            return fill_result

        # Thin liquidity: log warning for awareness (H5 OR filter)
        # Production enforcement: buy-stop submitted on all days; post-fill volume check on thin days
        if self.market_regime and self.market_regime.is_thin_liquidity(date.today()):
            info = self.market_regime.get_regime_info(date.today())
            svr = info.get('spy_volume_ratio')
            svr_str = f"{svr:.2f}" if svr is not None else "N/A"
            logger.warning(
                f"THIN LIQUIDITY: SPY vol ratio {svr_str} "
                f"< {self.market_regime.min_spy_volume_ratio} — "
                f"breakout vol threshold raised to "
                f"{self.market_regime.thin_liquidity_breakout_vol_ratio:.1f}x"
            )

        # Max trades per day — blocks NEW order placement only
        if self.market_regime and self.market_regime.max_trades_per_day > 0 and self._daily_trade_count >= self.market_regime.max_trades_per_day:
            logger.warning(
                f"MAX TRADES PER DAY reached ({self._daily_trade_count}) — skipping new trades"
            )
            return fill_result

        if not self._qualified_symbols:
            logger.debug("No qualified symbols to check")
            return fill_result

        # Background news refresh: re-check ONE no-news symbol per cycle (round-robin).
        # Catches news that breaks after scanner qualification. Updates _news_data cache
        # so the instant bar callback and news kill rules see fresh classifications.
        # NOT in the order path — runs here in the 5s polling loop.
        if hasattr(self, 'news_provider') and self.news_provider:
            no_news_syms = [
                s for s in self._qualified_symbols - self._traded_symbols
                if self._news_data.get(s, {}).get('news_category', 'NO_NEWS')
                   in ('NO_NEWS', 'OTHER', 'GARBAGE_RECAP', None)
            ]
            if no_news_syms:
                # Round-robin: pick one symbol per cycle
                if not hasattr(self, '_news_refresh_idx'):
                    self._news_refresh_idx = 0
                sym = no_news_syms[self._news_refresh_idx % len(no_news_syms)]
                self._news_refresh_idx += 1
                try:
                    uni = self.db.get_universe_stock(sym) if self.db else None
                    _ctx = {'float_shares': (uni.get('float_shares') or 0) if uni else 0,
                            'price': 0}
                    fresh = self.news_provider.classify_news(sym, stock_context=_ctx)
                    fresh_cat = fresh.get('category', 'NO_NEWS')
                    old_cat = self._news_data.get(sym, {}).get('news_category', 'NO_NEWS')
                    if fresh_cat != old_cat and fresh_cat not in ('NO_NEWS', 'OTHER', 'GARBAGE_RECAP'):
                        logger.info(f"{sym}: Background news refresh: {old_cat} → {fresh_cat}")
                        self._news_data[sym] = {
                            'news_catalyst': fresh.get('catalyst'),
                            'news_headline': (fresh.get('headline') or '')[:200],
                            'news_reason': (fresh.get('reason') or '')[:100],
                            'news_category': fresh_cat,
                        }
                except Exception as e:
                    logger.debug(f"{sym}: Background news refresh failed: {e}")

        # Skip new orders after last_entry_time
        if self._is_past_last_entry_time():
            logger.debug("Past last entry time, not placing new orders")
            return fill_result

        # SPY MACD afternoon cutoff — refresh and check
        self._refresh_spy_macd()
        if self._is_spy_macd_cutoff_blocked():
            logger.info(
                f"SPY MACD CUTOFF: histogram={self._spy_macd_cache:.6f} > 0 "
                f"after {self._spy_macd_cutoff_time[0]:02d}:{self._spy_macd_cutoff_time[1]:02d} ET "
                f"— skipping new trades"
            )
            return fill_result

        symbols_to_check = (
            self._qualified_symbols - self._traded_symbols
            - set(self._pending_orders.keys())
        )
        if not symbols_to_check:
            logger.debug("All qualified symbols already traded or have pending orders")
            return fill_result

        logger.info(f"Pattern check: {len(symbols_to_check)} symbols — {sorted(symbols_to_check)}")

        # Batch-fetch 1-min bars for ALL symbols in a single API call.
        # Eliminates N sequential REST calls (1-2s each) → single ~1s call.
        import pytz as _pytz
        _et = _pytz.timezone('US/Eastern')
        _now_et = datetime.now(_et)
        _market_open = _now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        _minutes_since_open = max(int((_now_et - _market_open).total_seconds() / 60), 30)
        try:
            _bars_batch = self.alpaca.get_1min_bars_multi(
                list(symbols_to_check), lookback_minutes=_minutes_since_open)
        except Exception as e:
            logger.error(f"Batch bar fetch failed: {e}, falling back to sequential")
            _bars_batch = {}

        last_order_result = None
        for symbol in sorted(symbols_to_check):
            prefetched = _bars_batch.get(symbol)
            result = self._check_symbol(symbol, prefetched_bars=prefetched)
            if result is not None:
                last_order_result = result

        return rt_result or fill_result or last_order_result

    def _check_symbol(self, symbol: str,
                      prefetched_bars: 'pd.DataFrame' = None) -> Optional[Dict[str, Any]]:
        """
        Check a single symbol for bull flag setup and place buy-stop order.

        Uses detect_setup() instead of detect() to find setups BEFORE breakout,
        then submits a buy-stop bracket order at breakout_level.

        Args:
            symbol: Stock symbol to check
            prefetched_bars: Pre-fetched 1-min bars from batch call (skips individual API call)

        Returns:
            Dict with order details if buy-stop placed, None otherwise
        """
        # Pipeline timing — captured at the very top so it represents
        # "we started looking at this symbol". Bar-close is derived later
        # once we have the bars dataframe (only meaningful when prefetched
        # bars came from a real-time WebSocket bar event).
        loop_processed_at = datetime.now(timezone.utc)

        # Fetch universe stock data once (used for volume filter + risk tier)
        uni_stock = self.db.get_universe_stock(symbol) if self.db else None

        # Volume filter: skip illiquid stocks before wasting API calls
        # Stocks NOT in universe (uni_stock=None) are blocked — no volume history.
        # Stocks in universe with avg_vol=0 are also blocked (no data = untradeable).
        if self.min_daily_volume > 0:
            if uni_stock is None:
                logger.info(
                    f"{symbol}: Skipping — not in universe (no volume data)"
                )
                return None
            avg_vol = (uni_stock.get('avg_volume_daily') or 0)
            if avg_vol < self.min_daily_volume:
                logger.info(
                    f"{symbol}: Skipping — avg daily vol {avg_vol:,.0f} "
                    f"< {self.min_daily_volume:,.0f} minimum"
                )
                return None

        # Use pre-fetched bars if available (batch call), otherwise fetch individually
        import pytz as _pytz
        _et = _pytz.timezone('US/Eastern')
        if prefetched_bars is not None and not prefetched_bars.empty:
            bars = prefetched_bars
        else:
            _now_et = datetime.now(_et)
            _market_open = _now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            _minutes_since_open = max(int((_now_et - _market_open).total_seconds() / 60), 30)
            try:
                bars = self.alpaca.get_1min_bars(symbol, lookback_minutes=_minutes_since_open)
            except Exception as e:
                logger.error(f"{symbol}: Failed to fetch 1-min bars: {e}")
                return None

        if bars is None or bars.empty:
            logger.debug(f"{symbol}: No 1-min bars available")
            return None

        # MACD warm-up: fetch previous trading day's bars (once per symbol per day)
        # Needed for both require_macd_positive detector filter AND macd_zones risk scaling
        need_warmup = getattr(self.detector, 'require_macd_positive', False) or self.macd_zones_enabled
        if need_warmup:
            if symbol not in self._macd_warmup_cache:
                self._fetch_macd_warmup(symbol)
            warmup = self._macd_warmup_cache.get(symbol)
            if hasattr(self.detector, 'set_macd_warmup'):
                self.detector.set_macd_warmup(warmup)

        # Detect setup (before breakout)
        setup = self.detector.detect_setup(symbol, bars)
        if setup is None:
            return None

        # Skip if this breakout level was already invalidated (flag_low broken).
        # Prevents re-detection loop where same historical pattern is found every cycle.
        # New pattern at a different level will still trade.
        invalidated = self._invalidated_levels.get(symbol)
        if invalidated is not None and abs(setup.breakout_level - invalidated) < 0.02:
            logger.debug(
                f"{symbol}: Skipping invalidated breakout ${setup.breakout_level:.2f} "
                f"(invalidated at ${invalidated:.2f})"
            )
            return None

        self._patterns_detected += 1
        self._pattern_details.append({
            'symbol': symbol,
            'pole_gain_pct': setup.pole_gain_pct,
            'retracement_pct': setup.retracement_pct,
            'breakout_level': setup.breakout_level,
        })

        # Deduplicate notifications — don't spam Telegram with the same
        # setup every 60s when position manager blocks (e.g., midday).
        # Only notify if breakout_level changed (new setup) or first time.
        already_notified = (
            self._notified_setups.get(symbol) == setup.breakout_level
        )

        # News gate: require real catalyst before trading
        if self.news_gate_enabled:
            news_data = self._news_data.get(symbol, {})
            news_cat = news_data.get('news_category', 'NO_NEWS')
            news_catalyst = news_data.get('news_catalyst')
            news_hl = news_data.get('news_headline', '')[:80]

            real_catalysts = {'FDA_CLINICAL', 'EARNINGS', 'CONTRACT_DEAL', 'MA',
                            'ANALYST', 'PRODUCT_LAUNCH', 'MANAGEMENT', 'SEC_FILING'}

            if news_cat not in real_catalysts:
                skip_reason = f"no catalyst ({news_cat})" if news_cat != 'NO_NEWS' else "no news found"
                logger.info(f"{symbol}: NEWS GATE SKIP — {skip_reason}: {news_hl}")
                self._eod_skipped.append((symbol, news_cat, news_hl))
                return None
            else:
                logger.info(f"{symbol}: NEWS GATE PASS — [{news_cat}]: {news_hl}")
                self._eod_traded.append((symbol, news_cat, news_hl, 0))  # pnl filled later

        # ============================================================
        # Filter ordering (cheap → expensive):
        #   1. News kill        (no API)
        #   2. Conviction       (no API)
        #   3. Risk tier + marginability check (Alpaca API call)
        # Conviction MUST come before marginability so we don't waste an
        # API call on trades we'll skip. Filters run sequentially — the
        # first to fire attributes the skip in logs/_eod_skipped. For
        # multi-reason post-hoc analysis, parse INFO logs (each filter
        # logs its own SKIP line independently).
        # ============================================================

        # News classification: use scanner's cached result (from on_stock_qualified).
        # LLM re-check removed — was 2-5s in critical order path. News kill rules
        # handle no-news risk, scanner classification is sufficient.

        # News kill rules: block trades in specific loser segments.
        # Shared decision (trading/news_kill_guard.py) — BT + live parity.
        if self.news_kill_enabled:
            _ndata = self._news_data.get(symbol, {})
            _ncat = _ndata.get('news_category', 'NO_NEWS')
            # has_catalyst is only consulted when the exemption is ON. A real-
            # catalyst category (or PENDING/ERROR) counts as catalyst so the
            # legacy exemption keeps giving them benefit of the doubt.
            _no_news_cats = {'NO_NEWS', 'GARBAGE_RECAP', 'OTHER'}
            _has_cat = (self.news_kill_catalyst_exemption
                        and _ncat not in _no_news_cats)
            _avg_vol = (uni_stock.get('avg_volume_daily') or 0) if uni_stock else 0
            _float = (uni_stock.get('float_shares') or 0) if uni_stock else 0
            _trade, _reason = news_kill_decision(
                has_catalyst=_has_cat,
                catalyst_exemption=self.news_kill_catalyst_exemption,
                avg_vol=_avg_vol, entry_price=setup.breakout_level,
                float_shares=_float, pole_gain=setup.pole_gain_pct,
                max_avg_vol=self.nkr_max_avg_vol,
                min_price=self.nkr_min_price, max_float=self.nkr_max_float)
            if not _trade:
                logger.info(f"{symbol}: NEWS KILL: {_reason}")
                self._eod_skipped.append((symbol, _ncat, _reason))
                return None

        # BF entry-price cap — Stage-2's `entry_price <= bull_flag.max_entry_price`.
        if self.trade_price_max > 0 and float(setup.breakout_level) > self.trade_price_max:
            logger.info(f"{symbol}: PRICE CAP skip — breakout ${setup.breakout_level:.2f} "
                        f"> ${self.trade_price_max:.2f}")
            self._eod_skipped.append((symbol, "PRICE_CAP", f"${setup.breakout_level:.2f}"))
            return None

        # Above-VWAP gate — same feature the BT caches as qf_vwap_dist_pct
        # (breakout_level vs cumulative VWAP through the setup bar).
        if self.vwap_gate.enabled or self.vwap_gate.shadow:
            _vg_vwap = self._compute_vwap(bars, up_to_idx=setup.flag_end_idx)
            _vg_dist = (
                (setup.breakout_level - _vg_vwap) / _vg_vwap * 100
                if _vg_vwap and _vg_vwap > 0 else None
            )
            _vg_keep, _vg_reason = passes_vwap_gate(_vg_dist, self.vwap_gate)
            if not _vg_keep and not self.vwap_gate.enabled:
                # shadow: say what the gate would do, keep going
                logger.info(f"{symbol}: VWAP GATE [SHADOW] would skip — {_vg_reason}")
            elif not _vg_keep:
                logger.info(f"{symbol}: VWAP GATE skip — {_vg_reason}")
                self._eod_skipped.append((symbol, "BELOW_VWAP", _vg_reason))
                return None
            # Consistency-profile shadow (2026-09-06, README §6): the two raw
            # rules that live applies through EXISTING knobs (detector
            # min_pole_gain_pct, scanner price_max) get a would-skip line too,
            # so the shadow window measures the whole P1 rule set.
            if self.vwap_gate.shadow:
                _pole = float(getattr(setup, 'pole_gain_pct', 0.0) or 0.0)
                _px = float(setup.breakout_level)
                _would = []
                if _pole < 5.0:
                    _would.append(f"pole {_pole:.1f}% < 5%")
                if _px > 20.0:
                    _would.append(f"price ${_px:.2f} > $20")
                if _would:
                    logger.info(f"{symbol}: CONSISTENCY RULES [SHADOW] would skip — "
                                + '; '.join(_would))

        # Conviction scoring: combine with risk tier, cap at 3x
        # Always compute breakdown (cheap, pure arithmetic) so we have it
        # ready for the skip-log without recomputing.
        conviction_mult = 1.0
        if self.conviction_enabled:
            spy_3d = self._get_spy_3d_range_live()
            # V2_clean rule 7+8 inputs — computed inline (mirrors
            # backtest._compute_qf_features:1166-1183 EXACTLY for parity).
            # NB: prev_close is available here from the universe (uni_stock
            # was loaded at line 2340); compute up-front so conviction has it.
            _v7_dist = 0.0
            _g8_fade = False
            _gap_pct_for_v9 = 0.0
            _vwap_at_setup = self._compute_vwap(bars, up_to_idx=setup.flag_end_idx)
            if _vwap_at_setup and _vwap_at_setup > 0:
                _v7_dist = (setup.breakout_level - _vwap_at_setup) / _vwap_at_setup * 100
            _conv_prev_close = (uni_stock.get('price_close') or 0) if uni_stock else 0
            if _conv_prev_close > 0 and len(bars) > 0:
                try:
                    _today_open = float(bars.iloc[0]['open'])
                    _gap_pct = (_today_open - _conv_prev_close) / _conv_prev_close * 100
                    _gap_pct_for_v9 = _gap_pct
                    _g8_fade = bool(
                        _gap_pct >= self.qf_gap_fade_threshold
                        and setup.breakout_level < _today_open
                    )
                except (KeyError, TypeError):
                    pass  # bars missing 'open' column — leave _g8_fade=False
            # Rule 9 intraday range (V-reversal). Uses bars up to setup bar.
            _intraday_range_pct = 0.0
            if len(bars) > 0 and setup.flag_end_idx >= 0:
                try:
                    _end = min(setup.flag_end_idx + 1, len(bars))
                    _hi = float(bars.iloc[:_end]['high'].max())
                    _lo = float(bars.iloc[:_end]['low'].min())
                    if _lo > 0:
                        _intraday_range_pct = (_hi - _lo) / _lo * 100
                except (KeyError, ValueError):
                    pass
            conviction_mult, _conv_brkdn = self._compute_conviction_score_setup(
                setup, spy_3d,
                vwap_dist_pct=_v7_dist,
                gap_fading=_g8_fade,
                gap_pct=_gap_pct_for_v9,
                intraday_range_pct=_intraday_range_pct,
                v_reversal_enabled=self.v_reversal_enabled,
                v_reversal_bonus=self.v_reversal_bonus,
                v_reversal_gap_pct_max=self.v_reversal_gap_pct_max,
                v_reversal_intraday_range_min=self.v_reversal_intraday_range_min,
                v_reversal_pole_gain_min=self.v_reversal_pole_gain_min,
                return_breakdown=True)
            if abs(conviction_mult - 1.0) > 0.05:
                _spy3d_str = f"{spy_3d:.1f}%" if spy_3d is not None else "MISSING"
                logger.info(
                    f"{symbol}: Conviction {conviction_mult:.2f}x "
                    f"(pole={setup.pole_gain_pct:.1f}%, "
                    f"retr={setup.retracement_pct:.0f}%, SPY3d={_spy3d_str}, "
                    f"vwap_dist={_v7_dist:+.1f}%, gap_fade={_g8_fade})")

            # Conviction filter: skip trades below quality threshold.
            # V2_clean (2026-04-15): 7-rule formula at threshold 1.4 (was 1.2).
            # Walk-forward: mean OOS +$28K, robust on all 3 splits.
            # Placed before risk_tier+marginability so we save the is_marginable API
            # call (~100-200ms) on every conviction-skipped trade.
            if (self.conviction_min_threshold > 0
                    and conviction_mult < self.conviction_min_threshold):
                breakdown_str = (
                    f"pole={_conv_brkdn['pole_gain']:+.1f} "
                    f"flag={_conv_brkdn['flag_tightness']:+.1f} "
                    f"vol={_conv_brkdn['vol_ratio']:+.1f} "
                    f"spy={_conv_brkdn['spy_regime']:+.1f} "
                    f"retr={_conv_brkdn['retracement']:+.1f} "
                    f"vwap={_conv_brkdn['vwap_dist']:+.1f} "
                    f"gap={_conv_brkdn['gap_fading']:+.1f}"
                )
                logger.info(
                    f"{symbol}: CONVICTION SKIP: {conviction_mult:.2f} < "
                    f"{self.conviction_min_threshold:.2f} "
                    f"({breakdown_str}; raw={_conv_brkdn['raw_score']:.2f})"
                )
                self._eod_skipped.append((
                    symbol, "LOW_CONVICTION",
                    f"conv {conviction_mult:.2f} ({breakdown_str})"
                ))
                return None

        # Marginal-conviction defensive scaling: applied to the SIZING multiplier
        # only. conviction_mult is left as the raw value so telemetry + downstream
        # filters see the true quality; sizing_conviction is used in combined_mult.
        sizing_conviction = conviction_mult
        if (self.conviction_marginal_scale_factor < 1.0
                and conviction_mult < self.conviction_marginal_upper):
            sizing_conviction = conviction_mult * self.conviction_marginal_scale_factor
            logger.info(
                f"{symbol}: MARGINAL CONV SCALE — conv {conviction_mult:.2f} × "
                f"{self.conviction_marginal_scale_factor} = "
                f"sizing {sizing_conviction:.2f} (below {self.conviction_marginal_upper})"
            )

        # Risk tier: scale risk on high-conviction setups.
        # Marginability API call is gated by risk_multiplier > 1.0,
        # AND only happens after news_kill + conviction filters pass.
        risk_multiplier = 1.0
        if self.risk_tiers_enabled:
            avg_vol = (uni_stock.get('avg_volume_daily') or 0) if uni_stock else 0
            risk_multiplier = self._get_risk_tier(setup.breakout_level, avg_vol)

            # Check marginability for leveraged trades (cached per-day).
            if risk_multiplier > 1.0:
                if not hasattr(self, '_margin_cache'):
                    self._margin_cache = {}
                if symbol not in self._margin_cache:
                    self._margin_cache[symbol] = bool(
                        self.alpaca.is_marginable(symbol)
                    )
                # Persist to universe table for BT-LIVE parity. Always
                # called on the main thread (this code path) — never from
                # the pre-warm daemon thread, which would trigger SQLite
                # cross-thread errors.
                self._persist_marginability_if_needed(
                    symbol, self._margin_cache[symbol]
                )
                if not self._margin_cache[symbol]:
                    logger.info(
                        f"{symbol}: Not marginable — falling back to 1x "
                        f"(wanted {risk_multiplier:.1f}x)"
                    )
                    risk_multiplier = 1.0

        # Use sizing_conviction (may be halved for marginal trades) for sizing,
        # but conviction_mult (raw) for telemetry / TTF composite features.
        combined_mult = min(3.0, risk_multiplier * sizing_conviction)

        # Two-tier filter gate (2026-04-17). When enabled, classify the trade
        # by max intraday change seen by scanner, then apply surgical drop +
        # composite z-score to Extras only (10-19%). A-tier (>=20%) and edge
        # (<10%) unfiltered. See trading/two_tier_filter.py for logic.
        if self._two_tier_cfg.get("enabled", False):
            from trading.two_tier_filter import (
                classify_tier,
                should_keep,
                TIER_EXTRAS,
            )
            _max_ic = self._qualified_max_intraday.get(symbol)
            tier_label = classify_tier(
                _max_ic,
                a_tier_lower=self._two_tier_cfg.get("a_tier_lower", 20.0),
                extras_lower=self._two_tier_cfg.get("extras_lower", 10.0),
            )
            # Only compute the MACD zone mult ONCE if we're about to gate on
            # Extras — it's used later anyway for sizing when zone filter fires.
            # None = signal unavailable (macd_zones disabled) -> should_keep
            # will skip the surgical check. When enabled, value in [0, 2.0].
            _ttf_macd_mult = None
            if tier_label == TIER_EXTRAS and self.macd_zones_enabled:
                _ttf_macd_mult = self._get_macd_zone_multiplier(
                    symbol, bars, setup.breakout_level,
                    intraday_change_pct=float(_max_ic or 0.0))
            # entry_minute = minutes-since-midnight ET of the current bar
            import pytz as _pytz
            _et = _pytz.timezone('US/Eastern')
            _now_et = datetime.now(_et)
            _feat_entry_min = _now_et.hour * 60 + _now_et.minute
            # vwap distance %: breakout_level vs cumulative VWAP through setup bar
            _vwap = self._compute_vwap(bars, setup.flag_end_idx)
            _vwap_dist = (
                (setup.breakout_level - _vwap) / _vwap * 100
                if _vwap and _vwap > 0 else None
            )
            _features = {
                "conviction_mult": float(conviction_mult),
                "qf_vwap_dist_pct": _vwap_dist,
                "qf_fill_vwap_dist_pct": _vwap_dist,  # fill ~= setup in live scanner moment
                "entry_minute": float(_feat_entry_min),
            }
            keep, reason = should_keep(
                tier=tier_label,
                macd_zone_mult=_ttf_macd_mult,
                features=_features,
                cfg=self._two_tier_cfg,
            )
            if not keep:
                _mi_str = f"{_max_ic:.1f}%" if _max_ic is not None else "?"
                _mz_str = f"{_ttf_macd_mult:.2f}" if _ttf_macd_mult is not None else "?"
                logger.info(
                    f"{symbol}: TWO-TIER FILTER SKIP (tier={tier_label}, "
                    f"max_intraday={_mi_str}, macd_mult={_mz_str}): {reason}"
                )
                return None

        # Create trade plan (pass ADV for liquidity cap — matches BT)
        _adv = int(uni_stock.get('avg_volume_daily') or 0) if uni_stock else 0
        plan = self.planner.create_plan(setup, avg_daily_volume=_adv, risk_multiplier=combined_mult)
        if plan is None:
            return None

        # Buying-power ceiling moved (2026-05-14) — it used to run HERE,
        # right after create_plan, BEFORE MACD-zone / regime / UD scaling.
        # That let regime's up-to-1.5x boost re-violate buying power after
        # the cap, and the re-plan dropped the conviction multiplier. It is
        # now a true last-step hard ceiling — see `_apply_bp_ceiling` call
        # after UD scaling below.

        # Min stop distance filter: reject tick-noise setups
        if self.min_stop_distance > 0:
            stop_dist = plan.entry_price - plan.stop_loss_price
            if stop_dist < self.min_stop_distance:
                logger.info(
                    f"{symbol}: SKIP — stop dist ${stop_dist:.2f} "
                    f"< min ${self.min_stop_distance:.2f} (tick noise)"
                )
                return None

        # Min price filter: reject sub-$2 entries (matches backtest.py:1737)
        if self.min_price > 0 and plan.entry_price < self.min_price:
            logger.info(
                f"{symbol}: SKIP — entry ${plan.entry_price:.2f} "
                f"< min ${self.min_price:.2f}"
            )
            return None

        # Check position limits (includes midday check)
        if not self.position_manager.can_open_position(symbol):
            return None

        # Quality filter: skip low-probability setups (VWAP, gap fade, SPY, slow pole)
        if self.quality_filter_enabled:
            _prev_close = (uni_stock.get('price_close') or 0) if uni_stock else 0
            qf_pass, qf_reason = self._check_quality_filter(
                symbol, bars, setup, plan,
                prev_close=_prev_close if _prev_close > 0 else None,
                bar_idx=setup.flag_end_idx,
            )
            if not qf_pass:
                logger.info(f"{symbol}: QUALITY FILTER SKIP: {qf_reason}")
                return None

        # MACD zone filter: dead zone always rejects, scaling only when no risk tier
        # Dead zone = garbage setups (30% WR) → reject regardless of tier
        # Scaling skipped when risk tier active (don't compound 3x * 1.5x = 4.5x)
        # Pass intraday_change_pct for tier-aware multiplier (A-tier vs Extras).
        _applied_macd_zone = 1.0
        if self.macd_zones_enabled:
            _max_ic_for_macd = self._qualified_max_intraday.get(symbol) or 0.0
            zone_mult = self._get_macd_zone_multiplier(
                symbol, bars, plan.entry_price,
                intraday_change_pct=float(_max_ic_for_macd))
            if zone_mult == 0.0:
                logger.info(f"{symbol}: MACD zone SKIP (dead zone)")
                return None
            elif zone_mult != 1.0 and risk_multiplier <= 1.0:
                _applied_macd_zone = zone_mult
                max_sh = int(self.planner.max_shares * zone_mult)
                scaled_shares = min(max_sh, max(1, int(plan.shares * zone_mult)))
                logger.info(f"{symbol}: MACD zone {zone_mult}x → shares {plan.shares} → {scaled_shares}")
                plan = TradePlan(
                    symbol=plan.symbol,
                    entry_price=plan.entry_price,
                    stop_loss_price=plan.stop_loss_price,
                    take_profit_price=plan.take_profit_price,
                    risk_per_share=plan.risk_per_share,
                    reward_per_share=plan.reward_per_share,
                    risk_reward_ratio=plan.risk_reward_ratio,
                    shares=scaled_shares,
                    total_risk=plan.risk_per_share * scaled_shares,
                    pattern=plan.pattern,
                )

        # Regime-aware sizing (Phase 1.4b): C2 regime skips; A/C1 boost.
        # Stacks multiplicatively on top of MACD zone scaling (above).
        if self.regime_sizing_enabled:
            from trading.regime_helpers import get_regime_multiplier
            _regime = self._get_today_regime()
            _regime_mult = get_regime_multiplier(_regime, self.regime_multipliers)
            if _regime_mult == 0.0:
                logger.info(f"{symbol}: REGIME {_regime} skip — no trade")
                return None
            if _regime_mult != 1.0:
                _reg_max = int(
                    self.planner.max_shares * _applied_macd_zone * _regime_mult
                )
                _reg_shares = min(_reg_max, max(1, int(plan.shares * _regime_mult)))
                logger.info(
                    f"{symbol}: REGIME {_regime} mult={_regime_mult:.2f} "
                    f"→ shares {plan.shares} → {_reg_shares}"
                )
                plan = TradePlan(
                    symbol=plan.symbol,
                    entry_price=plan.entry_price,
                    stop_loss_price=plan.stop_loss_price,
                    take_profit_price=plan.take_profit_price,
                    risk_per_share=plan.risk_per_share,
                    reward_per_share=plan.reward_per_share,
                    risk_reward_ratio=plan.risk_reward_ratio,
                    shares=_reg_shares,
                    total_risk=plan.risk_per_share * _reg_shares,
                    pattern=plan.pattern,
                )

        # UD risk scaling: reduce size when SPY up/down volume ratio is euphoric
        if self.ud_risk_scaling_enabled:
            ud = self.market_regime.get_spy_ud_volume_ratio(date.today()) if self.market_regime else None
            if ud is not None and isinstance(ud, (int, float)) and ud > self.ud_threshold:
                ud_shares = max(1, int(plan.shares * self.ud_scale_factor))
                logger.info(
                    f"{symbol}: UD scaling {self.ud_scale_factor}x (UD={ud:.2f}>{self.ud_threshold}) "
                    f"→ shares {plan.shares} → {ud_shares}"
                )
                plan = TradePlan(
                    symbol=plan.symbol,
                    entry_price=plan.entry_price,
                    stop_loss_price=plan.stop_loss_price,
                    take_profit_price=plan.take_profit_price,
                    risk_per_share=plan.risk_per_share,
                    reward_per_share=plan.reward_per_share,
                    risk_reward_ratio=plan.risk_reward_ratio,
                    shares=ud_shares,
                    total_risk=plan.risk_per_share * ud_shares,
                    pattern=plan.pattern,
                )

        # Buying-power ceiling — LAST sizing step, after tier×conviction +
        # MACD-zone + regime + UD scaling. A pure post-hoc hard ceiling that
        # no later multiplier can re-violate. See _apply_bp_ceiling docstring
        # for the 2026-05-14 reorder rationale (TRT BP-replan bug + regime
        # re-violation).
        plan = self._apply_risk_cap(plan, symbol)
        plan = self._apply_bp_ceiling(plan, symbol)
        if plan is None:
            return None

        # Production enforcement: buy-stop submitted on all days; post-fill volume check on thin days
        is_thin = self.market_regime and self.market_regime.is_thin_liquidity(date.today())

        # Self-managed stops: widen bracket SL to safety-net level,
        # real stop is monitored by StopMonitor via WebSocket.
        # Pass safety-net SL as override — DON'T mutate plan, so DB records
        # correct risk_per_share and stop_loss_price.
        # All filters passed — NOW notify (pattern + plan)
        if not already_notified:
            self._notified_setups[symbol] = setup.breakout_level
            if self.notifier:
                self.notifier.notify_pattern_detected(
                    symbol=symbol,
                    pole_gain_pct=setup.pole_gain_pct,
                    retracement_pct=setup.retracement_pct,
                    breakout_level=setup.breakout_level,
                )
                self.notifier.notify_trade_planned(
                    symbol=symbol,
                    entry=plan.entry_price,
                    stop=plan.stop_loss_price,
                    target=plan.take_profit_price,
                    shares=plan.shares,
                    risk_reward=plan.risk_reward_ratio,
                )

        # Build pipeline_timing once for whichever submit path runs. Only
        # populate bar_close_at when we have prefetched bars from a RT
        # WebSocket event (the periodic poll fetches bars itself, where
        # the "last bar close" isn't a meaningful trigger).
        _bar_close_at = None
        if prefetched_bars is not None and not prefetched_bars.empty:
            try:
                _ts = prefetched_bars.iloc[-1].get('timestamp')
                if _ts is None:
                    _ts = prefetched_bars.iloc[-1].name
                _bar_close_at = self._bar_start_to_close(_ts)
            except Exception:
                _bar_close_at = None  # not fatal — derived metrics just stay NULL
        _pipeline_timing = {
            'loop_processed_at': loop_processed_at,
            'bar_close_at': _bar_close_at,
        }

        # Account-level halt check (margin call / blocked / status). Refuses
        # NEW entries only; existing position management is unaffected.
        from trading import system_state as _system_state
        if _system_state.is_account_halted():
            _det = _system_state.get_halt_details()
            logger.warning(
                f"{symbol}: BF entry refused — account halt active "
                f"(event={_det.get('event_type')}, since {_det.get('halted_at')})"
            )
            return None

        real_stop_level = plan.stop_loss_price
        if self.stop_monitor:
            # Simple stop-limit (no bracket) — avoids 3x margin reservation
            # Safety-net SL submitted separately after fill detection
            logger.info(
                f"{symbol}: Self-managed stops — real stop ${real_stop_level:.2f}, "
                f"safety-net SL after fill ({self.safety_net_sl_pct:.0%})"
            )
            result = self.executor.submit_buy_stop_order(
                plan, pipeline_timing=_pipeline_timing)
        else:
            # Bracket order — SL/TP legs provide protection without StopMonitor
            # No sl_override: bracket SL = plan.stop_loss_price (real stop)
            result = self.executor.submit_buy_stop_bracket_order(
                plan, pipeline_timing=_pipeline_timing)

        if result is not None:
            # NOTE: _daily_trade_count and mark_traded are deferred to fill
            # time (_manage_pending_orders status=='filled'). This allows
            # re-entry after cancel/expire and accurate trade counting.
            pending = {
                'order_id': result['order_id'],
                'plan': plan,
                'setup': setup,
                'placed_at': datetime.now(timezone.utc),
                'news_data': self._news_data.get(symbol),
                'order_type': result.get('order_type', 'stop_bracket'),
            }
            # Store real stop for StopMonitor registration on fill
            if self.stop_monitor:
                pending['real_stop_level'] = real_stop_level
            if is_thin:
                pending['thin_liquidity'] = True
                pending['min_breakout_vol_ratio'] = self.market_regime.get_min_breakout_volume_ratio(date.today())
                logger.info(
                    f"{symbol}: BUY-STOP ORDER PLACED (thin liquidity) — "
                    f"min BVR {pending['min_breakout_vol_ratio']:.1f}x, {result}"
                )
            else:
                logger.info(f"{symbol}: BUY-STOP ORDER PLACED — {result}")
            self._pending_orders[symbol] = pending

            # Start passive quote monitoring for entry slippage analysis
            if self.stop_monitor:
                try:
                    quote = self.alpaca.get_latest_quote(symbol)
                    self.stop_monitor.add_quote_watch(
                        symbol,
                        submit_bid=quote.get('bid_price', 0.0),
                        submit_ask=quote.get('ask_price', 0.0),
                        submit_bid_size=quote.get('bid_size', 0),
                        submit_ask_size=quote.get('ask_size', 0),
                    )
                except Exception as e:
                    logger.warning(f"{symbol}: quote-watch start failed: {e}")
                    # Still start quote watch with zeros — will capture live quotes
                    self.stop_monitor.add_quote_watch(symbol)

            # Notify order submitted
            if self.notifier:
                self.notifier.notify_order_submitted(
                    symbol=symbol,
                    order_id=result.get('order_id', ''),
                    shares=plan.shares,
                    entry=plan.entry_price,
                )

        return result

    def _check_breakout_volume(self, symbol: str, pending: Dict) -> bool:
        """
        Check if the breakout bar had sufficient volume on a thin-liquidity day.

        Lookback window is computed from order placement time to now (the buy-stop
        could have filled anytime in that window), ensuring the breakout bar is
        captured even if fill detection is delayed by the poll interval.

        Finds the first bar where high >= breakout_level and computes
        BVR (breakout volume ratio) = bar_volume / avg_flag_volume.

        Fails open: returns True if bars unavailable or no breakout bar found.
        Fails safe: returns False if avg_flag_volume <= 0.

        Args:
            symbol: Stock symbol
            pending: Pending order dict with setup and min_breakout_vol_ratio

        Returns:
            True if volume is adequate (keep trade), False if weak (reject trade)
        """
        setup = pending['setup']
        min_bvr = pending.get('min_breakout_vol_ratio', 2.0)

        # Lookback must cover from order placement to now (fill could happen anytime)
        # Add 2-min buffer for bar completion lag and poll delay
        placed_at = pending.get('placed_at')
        if placed_at:
            elapsed_minutes = (datetime.now(timezone.utc) - placed_at).total_seconds() / 60.0
            lookback = int(elapsed_minutes) + 2
        else:
            lookback = 15  # fallback: conservative wide window
        lookback = max(lookback, 5)  # minimum 5 minutes
        lookback = min(lookback, 30)  # cap at 30 minutes (same as detection window)

        # Fetch recent 1-min bars
        try:
            bars = self.alpaca.get_1min_bars(symbol, lookback_minutes=lookback)
        except Exception as e:
            logger.warning(
                f"{symbol}: Failed to fetch bars for breakout volume check: {e} — "
                f"failing open (keeping trade)"
            )
            return True

        if bars is None or bars.empty:
            logger.warning(
                f"{symbol}: No bars available for breakout volume check — "
                f"failing open (keeping trade)"
            )
            return True

        # Find first bar where high >= breakout_level
        breakout_bar = None
        for _, bar in bars.iterrows():
            if bar['high'] >= setup.breakout_level:
                breakout_bar = bar
                break

        if breakout_bar is None:
            logger.warning(
                f"{symbol}: No breakout bar found in recent bars — "
                f"failing open (keeping trade)"
            )
            return True

        # Compute BVR
        avg_flag_vol = setup.avg_flag_volume
        if avg_flag_vol <= 0:
            logger.warning(
                f"{symbol}: avg_flag_volume <= 0 — failing safe (rejecting trade)"
            )
            return False

        bar_volume = breakout_bar['volume']
        bvr = bar_volume / avg_flag_vol

        if bvr >= min_bvr:
            logger.info(
                f"{symbol}: Breakout volume CONFIRMED — "
                f"BVR {bvr:.1f}x >= {min_bvr:.1f}x (keeping trade)"
            )
            return True
        else:
            logger.info(
                f"{symbol}: Breakout volume WEAK — "
                f"BVR {bvr:.1f}x < {min_bvr:.1f}x (rejecting trade)"
            )
            return False

    def _emergency_close_position(
        self, symbol: str, order_id: str, fill_price: float,
        actual_qty: int, trade_record: dict,
        exit_reason: str = 'thin_liquidity_reject'
    ) -> None:
        """
        Close a position immediately after fill.

        Used when a post-fill check fails: weak breakout volume on thin days,
        gap-fill leg replacement failure, etc.

        Handles the full lifecycle: close position, poll for exit price,
        compute PnL, update DB, record in circuit breaker, notify via Telegram.

        Note: _sync_closed_positions() filters by exit_price IS NULL,
        so once we set exit_price here, it won't double-process.

        Args:
            symbol: Stock symbol
            order_id: Original buy-stop order ID
            fill_price: Entry fill price
            actual_qty: Number of shares filled
            trade_record: DB trade record dict (may be None)
            exit_reason: Reason string for DB (e.g. 'thin_liquidity_reject',
                'gap_adjust_failed')
        """
        logger.info(
            f"{symbol}: EMERGENCY CLOSE ({exit_reason}) — closing position immediately"
        )

        # Close the position
        close_order_id = None
        try:
            close_result = self.alpaca.close_position(symbol)
            close_order_id = close_result.get('id', '')
            logger.info(f"{symbol}: Close order submitted — ID: {close_order_id}")
        except Exception as e:
            error_msg = f"{symbol}: Failed to close position ({exit_reason}): {e}"
            logger.error(error_msg)
            if self.notifier:
                self.notifier.notify_error(error_msg, component="EmergencyClose")
            return

        # Poll for exit price (reuse fill-price retry pattern)
        exit_price = None
        if close_order_id:
            for attempt in range(5):
                time_mod.sleep(0.5)
                try:
                    close_order = self.alpaca.get_order(close_order_id)
                    if close_order.get('status') == 'filled':
                        exit_price = close_order.get('filled_avg_price')
                        if exit_price is not None:
                            logger.info(
                                f"{symbol}: Close filled at ${exit_price:.2f} "
                                f"(attempt {attempt + 1})"
                            )
                            break
                except Exception:
                    pass

        if exit_price is None:
            logger.warning(
                f"{symbol}: Could not get exit price for {exit_reason} — "
                f"using fill_price as estimate"
            )
            exit_price = fill_price

        # Compute PnL
        pnl = (exit_price - fill_price) * actual_qty
        pnl_pct = (exit_price / fill_price - 1) * 100 if fill_price > 0 else 0.0

        # Update DB
        if trade_record:
            self.db.update_trade(trade_record['id'], {
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'exited_at': datetime.now(timezone.utc),
                'pnl': pnl,
                'pnl_pct': pnl_pct,
            })
            logger.info(
                f"{symbol}: {exit_reason} DB updated — "
                f"exit ${exit_price:.2f}, P&L ${pnl:+,.2f} ({pnl_pct:+.1f}%)"
            )
        else:
            logger.error(
                f"{symbol}: No trade record to update for {exit_reason}"
            )

        # Circuit breaker
        self.position_manager.record_trade_pnl(pnl)

        # Notify
        if self.notifier:
            self.notifier.notify_position_closed(
                symbol=symbol,
                entry_price=fill_price,
                exit_price=exit_price,
                shares=actual_qty,
                pnl=pnl,
                exit_reason=exit_reason,
            )

    # =====================================================================
    # BF kill rails (Discipline Program Phase 1, 2026-08-22) — DB-derived,
    # ET-dated, fail-closed. Mirrors trading/orb_engine kill rails.
    # =====================================================================

    def _et_now(self) -> datetime:
        """Market-clock now (ET). Kill-rail windows key on the ET session
        date, not the box's UTC wall clock (mirrors ORBEngine._et_now)."""
        return datetime.now(ET)

    def _sync_month_pause_from_flag(self, at_boot: bool = False) -> None:
        """Sync the month-pause runtime latch from the persistent flag file.

        Called at boot (__init__) and every reset_daily. The flag is set by a
        month-rail breach and cleared ONLY by the owner removing the file —
        a month roll does NOT clear it. Owner may also pause BF manually by
        creating the file. Logs every transition (no silent state changes)."""
        exists = self.month_pause_flag_path.exists()
        if exists and not self._bf_month_paused:
            self._bf_month_paused = True
            logger.error(
                f"BF RAIL: MONTH-PAUSE flag present "
                f"({self.month_pause_flag_path})"
                f"{' at boot' if at_boot else ''} — bull flag entries DISABLED "
                f"until owner removes the file")
        elif not exists and self._bf_month_paused:
            self._bf_month_paused = False
            logger.warning(
                f"BF RAIL: month-pause flag removed "
                f"({self.month_pause_flag_path}) — owner cleared the pause; "
                f"bull flag entries re-enabled")

    def _realized_bf_pnl(self, since_date: str) -> float:
        """Realized bull_flag P&L (closed rows) with trade_date >= since_date.

        DB-derived so it survives restarts (zero extra state). FAIL-CLOSED:
        any query error returns the -1e9 sentinel so the caller blocks new
        entries rather than trading blind. Mirrors
        ORBEngine._realized_orb_pnl."""
        path = getattr(self.db, '_trades_path', None)
        if path is None:
            # No real trades DB configured (e.g. a bare mock) — nothing to
            # check; stay INERT rather than read some default file
            # (fail-open, logged).
            logger.warning("BF RAIL: realized-pnl: db has no _trades_path — "
                           "kill rail inert (no block)")
            return 0.0
        try:
            import sqlite3
            conn = sqlite3.connect(str(path), timeout=10)
            cur = conn.execute(
                "SELECT COALESCE(SUM(pnl),0) FROM trades WHERE "
                "strategy='bull_flag' AND trade_date>=? AND pnl IS NOT NULL",
                (since_date,))
            v = float(cur.fetchone()[0] or 0.0)
            conn.close()
            return v
        except Exception as e:
            logger.error(f"BF RAIL: realized-pnl query failed ({e}) — "
                         f"FAIL-SAFE: treating as kill-breached")
            return -1e9

    def _bf_month_trades(self, month_start: str) -> List[Tuple[str, str, float]]:
        """Closed bull_flag trades (trade_date, symbol, pnl) for the month —
        the evidence list attached to the ABANDON-GATE telegram. Best-effort:
        a query error returns [] with an ERROR log (the pause itself is
        decided by _realized_bf_pnl, never by this list)."""
        path = getattr(self.db, '_trades_path', None)
        if path is None:
            logger.error("BF RAIL: month trade list unavailable "
                         "(db has no _trades_path)")
            return []
        try:
            import sqlite3
            conn = sqlite3.connect(str(path), timeout=10)
            rows = conn.execute(
                "SELECT trade_date, symbol, pnl FROM trades WHERE "
                "strategy='bull_flag' AND trade_date>=? AND pnl IS NOT NULL "
                "ORDER BY trade_date, symbol",
                (month_start,)).fetchall()
            conn.close()
            return [(str(d), str(s), float(p)) for d, s, p in rows]
        except Exception as e:
            logger.error(f"BF RAIL: month trade list query failed ({e}) — "
                         f"sending ABANDON-GATE without the list")
            return []

    def _bf_notify(self, msg: str) -> None:
        """Send a rail Telegram (never raises — notifier failure must not
        break the trading loop; failure is logged per fallback rule)."""
        if not self.notifier:
            return
        try:
            self.notifier.send_message_sync(msg)
        except Exception as e:
            logger.error(f"BF RAIL: telegram send failed ({e}) — "
                         f"rail action already applied, notification lost")

    def _fire_month_pause(self, mo: float, month_start: str) -> None:
        """Month rail breach: set the runtime pause latch, write the
        persistent flag file, and send the [BF] ABANDON-GATE telegram with
        the month's closed-trade list attached (owner decision evidence)."""
        self._bf_month_paused = True
        logger.error(
            f"BF RAIL: MONTH PAUSE — realized ${mo:+,.0f} breached "
            f"${self.kill_month_pause_usd:,.0f}; bull flag PAUSED "
            f"(no new entries) until owner clears "
            f"{self.month_pause_flag_path}")
        try:
            self.month_pause_flag_path.parent.mkdir(parents=True,
                                                    exist_ok=True)
            self.month_pause_flag_path.write_text(
                f"paused_at={self._et_now().isoformat()}\n"
                f"month_start={month_start}\n"
                f"month_realized_usd={mo:.2f}\n"
                f"threshold_usd={self.kill_month_pause_usd:.2f}\n"
                f"clear: owner deletes this file (docs/"
                f"bf_discipline_program_aug2026.md Phase 1)\n")
        except Exception as e:
            logger.error(
                f"BF RAIL: could not write month-pause flag file ({e}) — "
                f"runtime latch still set; pause will NOT survive restart "
                f"unless P&L still breaches")
        trades = self._bf_month_trades(month_start)
        lines = [f"{d[5:]} {s} ${p:+,.0f}" for d, s, p in trades[:40]]
        if len(trades) > 40:
            lines.append(f"...and {len(trades) - 40} more")
        trade_block = "\n".join(lines) if lines else "(trade list unavailable)"
        self._bf_notify(
            f"[BF] ABANDON-GATE — month realized ${mo:+,.0f} breached "
            f"${self.kill_month_pause_usd:,.0f}.\n"
            f"Bull flag PAUSED (no new entries; open positions keep their "
            f"stops). Clear: remove {self.month_pause_flag_path.name} + "
            f"restart or wait for daily reset.\n"
            f"Month trades ({len(trades)}):\n{trade_block}")

    def _kill_rails_blocked(self) -> Optional[str]:
        """Return a kill-rail reason blocking NEW bull_flag entries, or None.

        Order of severity: month pause > weekly > daily (mirrors ORB). Each
        rail Telegram-notifies once per still-breached period. Weekly
        additionally FLATTENS open BF positions once (real
        _force_close_all path). Month breach latches the pause + writes the
        persistent flag file. A pnl-query failure fails CLOSED (blocks)
        WITHOUT escalating to pause/flatten (a transient DB error must not
        fire the ABANDON-GATE). Rails only gate/flatten — they never place
        orders."""
        if not self.kill_rails_enabled:
            return None
        if self._bf_month_paused:
            if not self._kill_pause_logged:
                self._kill_pause_logged = True
                logger.warning(
                    f"BF RAIL: month-pause latch active "
                    f"({self.month_pause_flag_path}) — no new entries")
            return 'month_pause'
        et = self._et_now()
        today = et.strftime('%Y-%m-%d')
        week_start = (et - timedelta(days=et.weekday())).strftime('%Y-%m-%d')
        month_start = et.strftime('%Y-%m-01')
        dy = self._realized_bf_pnl(today)
        wk = self._realized_bf_pnl(week_start)
        mo = self._realized_bf_pnl(month_start)
        _SENTINEL = -1e8   # -1e9 fail-closed marker (well below any real loss)
        if min(dy, wk, mo) <= _SENTINEL:
            if not self._kill_query_fail_notified:
                self._kill_query_fail_notified = True
                logger.error("BF RAIL: P&L query failed — FAIL-SAFE blocking "
                             "new entries until it recovers")
            return 'pnl_query_failed'
        if mo <= self.kill_month_pause_usd:
            self._fire_month_pause(mo, month_start)
            return 'month_pause'
        if wk <= self.kill_weekly_usd:
            if not self._kill_weekly_notified:
                self._kill_weekly_notified = True
                logger.warning(
                    f"BF RAIL: WEEKLY KILL — realized ${wk:+,.0f} breached "
                    f"${self.kill_weekly_usd:,.0f}; flat + no new entries "
                    f"this ISO week")
                self._bf_notify(
                    f"[BF] WEEKLY KILL — realized ${wk:+,.0f} breached "
                    f"${self.kill_weekly_usd:,.0f}. Flattening + no new "
                    f"entries this week.")
            if not self._kill_weekly_flattened:
                self._kill_weekly_flattened = True
                try:
                    self._force_close_all()
                    logger.warning("BF RAIL: WEEKLY KILL flat — "
                                   "_force_close_all completed")
                except Exception as e:
                    logger.error(f"BF RAIL: WEEKLY KILL flat failed ({e}) — "
                                 f"positions keep their stops")
            return 'weekly_kill'
        if dy <= self.kill_daily_usd:
            if not self._kill_daily_notified:
                self._kill_daily_notified = True
                logger.warning(
                    f"BF RAIL: DAILY KILL — realized ${dy:+,.0f} breached "
                    f"${self.kill_daily_usd:,.0f}; no new entries today")
                self._bf_notify(
                    f"[BF] DAILY KILL — realized ${dy:+,.0f} breached "
                    f"${self.kill_daily_usd:,.0f}. No new entries today "
                    f"(open positions keep their stops).")
            return 'daily_kill'
        return None

    def _force_close_all(self) -> None:
        """
        Cancel all pending orders and close all open positions.

        Called at force_close_time to ensure we're flat before market close.
        Syncs closed positions first so any SL/TP exits that already happened
        are recorded before we attempt to close remaining positions.
        """
        # Stop StopMonitor before force-closing — prevents race conditions
        # where monitor tries to exit while we're also closing.
        # SCOPED to bull_flag watches so we don't remove ORB / MACD Wave
        # watches from the shared monitor (would disable their own exit
        # paths). Each strategy runs its own force_close_all.
        if self.stop_monitor:
            for symbol in list(self.stop_monitor.watched_symbols_for('bull_flag')):
                self.stop_monitor.remove_watch(symbol)
            self._process_stop_monitor_exits()

        # Sync first — record any SL/TP exits that happened before force close
        self._sync_closed_positions()
        # Process any pending order fills (e.g., late fills just before force close)
        self._manage_pending_orders()

        # Cancel pending orders
        for symbol, pending in list(self._pending_orders.items()):
            try:
                self.alpaca.cancel_order(pending['order_id'])
                logger.info(f"{symbol}: Force-close — cancelled pending order {pending['order_id']}")
            except Exception as e:
                error_msg = f"{symbol}: Failed to cancel pending order during force-close: {e}"
                logger.error(error_msg)
                if self.notifier:
                    self.notifier.notify_error(error_msg, component="ForceClose")
            if self.stop_monitor:
                self.stop_monitor.remove_quote_watch(symbol)
        self._pending_orders.clear()

        # Close open positions and update DB
        try:
            positions = self.alpaca.get_open_positions()
            today = date.today().isoformat()
            open_trades = self.db.get_open_trades(today, strategy='bull_flag')
            # Index open trades by symbol for fast lookup
            trades_by_symbol = {}
            for t in open_trades:
                trades_by_symbol[t['symbol']] = t

            FORCE_CLOSE_RETRIES = 3
            FORCE_CLOSE_BACKOFF = [2, 5, 10]

            for pos in positions:
                symbol = pos['symbol']

                # Only close positions owned by bull_flag. MACD wave manages
                # its own positions via MACDWaveEngine.force_close_all().
                # Without this filter, a systemctl restart mid-session would
                # liquidate macd_wave winners (see 2026-04-16 CDNA incident).
                if symbol not in trades_by_symbol:
                    logger.info(
                        f"{symbol}: Skipping force-close — no open bull_flag "
                        f"trade (likely macd_wave or external position)"
                    )
                    continue

                close_succeeded = False

                # Cancel any open sell orders (TP/SL legs) holding shares
                # before attempting close_position — otherwise Alpaca rejects
                # with "insufficient qty available" (shares held by orders).
                try:
                    from alpaca.trading.requests import GetOrdersRequest
                    from alpaca.trading.enums import QueryOrderStatus
                    open_orders = self.alpaca.trading_client.get_orders(
                        GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
                    )
                    for oo in open_orders:
                        try:
                            self.alpaca.cancel_order(str(oo.id))
                        except Exception:
                            pass
                    if open_orders:
                        time_mod.sleep(1)  # Let cancels settle
                except Exception as e:
                    logger.warning(f"{symbol}: Failed to cancel open orders before force-close: {e}")

                close_order_id = None
                for attempt in range(FORCE_CLOSE_RETRIES):
                    try:
                        close_result = self.alpaca.close_position(symbol)
                        close_order_id = close_result.get('id', '') if close_result else ''
                        close_succeeded = True
                        break
                    except Exception as e:
                        if attempt < FORCE_CLOSE_RETRIES - 1:
                            wait = FORCE_CLOSE_BACKOFF[attempt]
                            logger.warning(
                                f"{symbol}: Force close attempt {attempt + 1} failed: {e}, "
                                f"retry in {wait}s"
                            )
                            time_mod.sleep(wait)
                        else:
                            logger.error(f"{symbol}: ALL force close attempts failed: {e}")
                            if self.notifier:
                                self.notifier.notify_error(
                                    f"MANUAL INTERVENTION: {symbol} force close failed "
                                    f"after {FORCE_CLOSE_RETRIES} attempts",
                                    component="ForceClose",
                                )

                if not close_succeeded:
                    continue

                # Poll for actual fill price (don't use stale position snapshot)
                exit_price = None
                if close_order_id:
                    for poll in range(5):
                        time_mod.sleep(0.5)
                        try:
                            close_order = self.alpaca.get_order(close_order_id)
                            if close_order.get('status') == 'filled':
                                exit_price = close_order.get('filled_avg_price')
                                if exit_price is not None:
                                    break
                        except Exception:
                            pass

                # Fallback to position snapshot if poll fails
                if exit_price is None:
                    qty = pos.get('qty', 0)
                    if qty > 0 and pos.get('market_value'):
                        exit_price = float(pos['market_value']) / qty
                    else:
                        exit_price = pos.get('avg_entry_price', 0)
                    logger.warning(
                        f"{symbol}: Force-close fill price unavailable, "
                        f"using snapshot ${exit_price:.2f}"
                    )

                logger.info(f"{symbol}: Force-close — position closed at ${exit_price:.2f}")

                # Update DB trade record with exit details
                trade = trades_by_symbol.get(symbol)
                if trade and trade.get('fill_price'):
                    qty_for_pnl = trade.get('filled_qty') or trade['shares']
                    pnl = (exit_price - trade['fill_price']) * qty_for_pnl
                    pnl_pct = (exit_price / trade['fill_price'] - 1) * 100
                    self.db.update_trade(trade['id'], {
                        'exit_price': exit_price,
                        'exit_reason': ExitReason.FORCE_CLOSE.value,
                        'exited_at': datetime.now(timezone.utc),
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                    })
                    self.position_manager.record_trade_pnl(pnl)
                    logger.info(
                        f"{symbol}: Force-close DB updated — "
                        f"P&L ${pnl:+,.2f} ({pnl_pct:+.1f}%)"
                    )
                elif trade:
                    logger.warning(
                        f"{symbol}: Force-close — trade has no fill_price, "
                        f"cannot compute P&L"
                    )

                if self.notifier:
                    entry = trade['fill_price'] if trade and trade.get('fill_price') else 0
                    fc_shares = trade.get('filled_qty') or trade.get('shares', 0) if trade else 0
                    fc_pnl = pnl if trade and trade.get('fill_price') else 0
                    self.notifier.notify_position_closed(
                        symbol=symbol,
                        entry_price=entry,
                        exit_price=exit_price,
                        shares=fc_shares,
                        pnl=fc_pnl,
                        exit_reason=ExitReason.FORCE_CLOSE.value,
                    )
        except Exception as e:
            error_msg = f"Failed to get open positions for force-close: {e}"
            logger.error(error_msg)
            if self.notifier:
                self.notifier.notify_error(error_msg, component="ForceClose")

    def run_monitoring_loop(self) -> None:
        """
        Run the pattern monitoring loop.

        Polls qualified symbols every pattern_poll_interval seconds.
        Stops placing new orders after last_entry_time.
        Force-closes all positions at force_close_time.
        Stops at market close (16:00 ET).
        """
        if not self.enabled:
            logger.info("Trading engine disabled, skipping monitoring loop")
            return

        # Start StopMonitor WebSocket thread if configured
        if self.stop_monitor:
            self.stop_monitor.start()
            logger.info("StopMonitor started for self-managed stops")

        logger.info(
            f"Trading engine monitoring loop started — "
            f"interval: {self.pattern_poll_interval}s, "
            f"symbols: {len(self._qualified_symbols)}, "
            f"last entry: {self.last_entry_hour}:{self.last_entry_minute:02d} ET, "
            f"force close: {self.force_close_hour}:{self.force_close_minute:02d} ET, "
            f"self_managed_stops: {self.stop_monitor is not None}"
        )

        force_closed = False

        while not (self.shutdown_event and self.shutdown_event.is_set()):
            now_et = datetime.now(ET)
            if now_et.hour >= 16:
                logger.info("Market closed, stopping monitoring loop")
                break

            # Force close check
            if not force_closed and self._is_past_force_close_time():
                logger.info("Force close time reached — closing all positions")
                self._force_close_all()
                force_closed = True

            if not force_closed:
                self.run_pattern_check()

            # Use shutdown_event.wait() instead of time.sleep() for interruptible sleep
            if self.shutdown_event:
                self.shutdown_event.wait(self.pattern_poll_interval)
            else:
                time_mod.sleep(self.pattern_poll_interval)

        # Stop StopMonitor regardless of exit reason (market close or SIGTERM)
        if self.stop_monitor:
            self.stop_monitor.stop()

        # Graceful shutdown: only force-close if past EOD time.
        # Mid-session SIGTERM (code deploy) should NOT liquidate positions —
        # they survive the restart and get recovered by startup sync.
        # See 2026-04-16 CDNA incident: a deploy at 12:20 ET force-closed
        # a winning MACD wave position via this path.
        if self.shutdown_event and self.shutdown_event.is_set():
            if self._is_past_force_close_time():
                logger.info("Shutdown at EOD — force-closing bull_flag positions...")
                self._force_close_all()
            else:
                logger.info(
                    "Shutdown mid-session — skipping force-close "
                    "(positions survive restart via startup sync)"
                )
            self.save_daily_summary()
            logger.info("Graceful shutdown complete")

    def get_daily_stats(self) -> Dict[str, Any]:
        """Get daily trading statistics."""
        today = date.today().isoformat()
        trades = self.db.get_trades_by_date(today)
        daily_pnl = self.db.get_daily_pnl(today)
        open_trades = self.db.get_open_trades(today, strategy='bull_flag')

        winning = sum(1 for t in trades if t.get('pnl') and t['pnl'] > 0)
        losing = sum(1 for t in trades if t.get('pnl') and t['pnl'] < 0)

        return {
            'trade_date': today,
            'total_trades': len(trades),
            'winning_trades': winning,
            'losing_trades': losing,
            'gross_pnl': daily_pnl,
            'open_positions': len(open_trades),
            'patterns_detected': self._patterns_detected,
            'patterns_traded': self._patterns_traded,
            'qualified_symbols': len(self._qualified_symbols),
            'patterns_detected_details': list(self._pattern_details),
            'trades': [dict(t) for t in trades] if trades else [],
        }

    def generate_daily_report(self, premarket_gaps: list = None,
                               qualified_stocks: list = None,
                               universe_size: int = 0) -> Dict[str, Any]:
        """
        Generate the full daily report data for Telegram.

        Args:
            premarket_gaps: List of pre-market gap dicts from scanner
            qualified_stocks: List of qualified stock dicts from scanner
            universe_size: Size of the stock universe

        Returns:
            Complete report dict for TelegramNotifier.send_daily_report()
        """
        stats = self.get_daily_stats()
        return {
            'trade_date': stats['trade_date'],
            'universe_size': universe_size,
            'premarket_gaps': premarket_gaps or [],
            'qualified_stocks': qualified_stocks or [],
            'patterns_detected': stats['patterns_detected'],
            'patterns_detected_details': stats['patterns_detected_details'],
            'trades': stats['trades'],
            'total_trades': stats['total_trades'],
            'winning_trades': stats['winning_trades'],
            'losing_trades': stats['losing_trades'],
            'gross_pnl': stats['gross_pnl'],
            'open_positions': stats['open_positions'],
        }

    def send_daily_report(self, premarket_gaps: list = None,
                           qualified_stocks: list = None,
                           universe_size: int = 0) -> None:
        """Generate and send the end-of-day Telegram report."""
        if not self.notifier:
            logger.debug("No notifier configured, skipping daily report")
            return

        report = self.generate_daily_report(
            premarket_gaps=premarket_gaps,
            qualified_stocks=qualified_stocks,
            universe_size=universe_size,
        )
        self.notifier.send_daily_report(report)
        logger.info("End-of-day Telegram report sent")

    def save_daily_summary(self) -> None:
        """Save daily trading summary to database."""
        stats = self.get_daily_stats()
        self.db.save_daily_summary({
            'trade_date': stats['trade_date'],
            'total_trades': stats['total_trades'],
            'winning_trades': stats['winning_trades'],
            'losing_trades': stats['losing_trades'],
            'gross_pnl': stats['gross_pnl'],
            'patterns_detected': stats['patterns_detected'],
            'patterns_traded': stats['patterns_traded'],
        })
        logger.info(f"Daily summary saved: {stats}")

    def reset_daily(self) -> None:
        """Reset daily state for a new trading day, then sync from DB/Alpaca.

        After clearing in-memory state, rebuilds _traded_symbols,
        _pending_orders, and _daily_trade_count from today's DB trades
        and Alpaca open orders. This ensures crash recovery doesn't
        orphan live positions or allow double-entry.
        """
        self._qualified_symbols.clear()
        self._traded_symbols.clear()
        self._patterns_detected = 0
        self._patterns_traded = 0
        self._pattern_details.clear()
        self._pending_orders.clear()
        self._daily_trade_count = 0
        self._notified_setups.clear()
        self._margin_cache = {}  # marginability cache is per-day (Fix 4)
        self._margin_persisted = set()  # DB-write tracker, per-day
        self.position_manager.reset_daily()
        # Roll BF kill-rail notify latches: daily every day; weekly on
        # ISO-week change; month key on month change. P&L itself is
        # DB-derived, so the latches only gate re-notification (a breach
        # persists across the roll until P&L recovers). The month-pause
        # latch is NOT rolled — it re-syncs from the persistent flag file
        # (cleared only by the owner removing it).
        et = self._et_now()
        self._kill_daily_notified = False
        self._kill_query_fail_notified = False
        self._kill_pause_logged = False
        _wk = (et - timedelta(days=et.weekday())).strftime('%Y-%m-%d')
        _mo = et.strftime('%Y-%m')
        if _wk != self._kill_week_key:
            self._kill_week_key = _wk
            self._kill_weekly_notified = False
            self._kill_weekly_flattened = False
        if _mo != self._kill_month_key:
            self._kill_month_key = _mo
        self._sync_month_pause_from_flag()
        self._refresh_spy_data()
        self._sync_startup_state()
        logger.info("Trading engine: daily state reset")

    def _sync_startup_state(self) -> None:
        """Rebuild in-memory state from DB trades and Alpaca for today.

        Prevents crash recovery from:
        - Allowing double-entry on symbols already traded today
        - Losing track of pending buy-stop orders still live on Alpaca
        - Miscounting daily trades
        - Leaving orphan positions from prior days open
        """
        today = date.today().isoformat()
        try:
            all_trades_today = self.db.get_trades_by_date(today)
        except Exception as e:
            logger.error(f"Startup sync: failed to load today's trades: {e}")
            return

        # CRITICAL: filter to BULL FLAG trades only. Before this filter, ORB
        # (and any other strategy running in the same service) had its pending
        # order_ids swept into bull flag's _pending_orders — bull flag then
        # polled them against the MAIN Alpaca account (wrong account for ORB
        # paper orders) and logged "order not found" every cycle. Legacy trades
        # from before the `strategy` column existed have strategy=None — treat
        # those as bull flag for backwards compatibility.
        trades_today = [
            t for t in all_trades_today
            if (t.get('strategy') or 'bull_flag') == 'bull_flag'
        ]
        skipped = len(all_trades_today) - len(trades_today)
        if skipped:
            logger.info(
                f"Startup sync: skipping {skipped} non-bull_flag trades "
                f"(other strategies manage their own state)"
            )

        # Rebuild _traded_symbols and _daily_trade_count from DB
        # Only count FILLED trades — cancelled orders should not block re-entry
        filled_count = 0
        for trade in trades_today:
            symbol = trade['symbol']
            if trade.get('fill_price') is not None:
                self._traded_symbols.add(symbol)
                self.position_manager.mark_traded(symbol)
                filled_count += 1

        self._daily_trade_count = filled_count

        # Rebuild _pending_orders from DB trades that have order_id but no fill.
        # IREZ 2026-05-08 fix: must EXCLUDE terminal statuses (cancelled,
        # time_stop_canceled, etc.) — otherwise dead orders get re-added to
        # pending. Their fills are then "rediscovered" via OrderStream replay
        # at the next restart, re-firing the post-fill kill switch on what
        # is now a SEPARATE live position. Today's IREZ #237 was preserved
        # only because the broker SL was holding the qty (close failed). Don't
        # rely on luck — filter by status here at the source.
        _terminal_pending_statuses = {
            'cancelled', 'canceled', 'expired', 'rejected', 'time_stop_canceled',
        }
        for trade in trades_today:
            symbol = trade['symbol']
            order_id = trade.get('order_id')
            order_status = (trade.get('order_status') or '').lower()
            if (order_id
                    and trade.get('fill_price') is None
                    and trade.get('exit_price') is None
                    and order_status not in _terminal_pending_statuses):
                plan = self._reconstruct_plan(trade)
                setup = self._reconstruct_setup(trade)
                self._pending_orders[symbol] = {
                    'order_id': order_id,
                    'plan': plan,
                    'setup': setup,
                    'placed_at': trade.get('created_at', datetime.now(timezone.utc)),
                }
                logger.info(f"{symbol}: Recovered pending order {order_id} from DB (plan={'yes' if plan else 'no'})")

        # Re-register filled-but-open positions with StopMonitor
        # (crash recovery: service restarted with live positions)
        if self.stop_monitor:
            for trade in trades_today:
                symbol = trade['symbol']
                if (trade.get('fill_price') is not None
                        and trade.get('exit_price') is None
                        and trade.get('real_stop_loss_price') is not None):
                    order_id = trade.get('order_id')
                    try:
                        order_detail = self.alpaca.get_order(order_id)
                        sl_leg, tp_leg = self._identify_bracket_legs(
                            order_detail.get('legs', []),
                            expected_sl=trade.get('entry_price', 0) * (1 - self.safety_net_sl_pct),
                            expected_tp=trade.get('take_profit_price'),
                        )
                        tp_leg_id = tp_leg['id'] if tp_leg else ''
                        sl_leg_id = sl_leg['id'] if sl_leg else ''
                        trail_r = self.trailing_stop_r if self.trailing_stop_enabled else 0.0
                        activate_r = self.trailing_activate_at_r if self.trailing_stop_enabled else 0.0
                        fill = trade['fill_price']
                        real_sl = trade['real_stop_loss_price']
                        # plan-R fix: pass planned values from DB.
                        # entry_price column = planned breakout level (set
                        # at trade-plan time, before fill). stop_loss_price
                        # is the planned stop (also unchanged by fill).
                        _planned_entry_db = float(trade.get('entry_price') or 0)
                        _planned_stop_db = float(trade.get('stop_loss_price') or 0)
                        _planned_R_db = (
                            (_planned_entry_db - _planned_stop_db)
                            if (_planned_entry_db > 0 and _planned_stop_db > 0)
                            else 0.0
                        )
                        self.stop_monitor.add_watch(
                            symbol=symbol,
                            stop_price=real_sl,
                            shares=trade.get('filled_qty') or trade['shares'],
                            tp_leg_id=tp_leg_id,
                            sl_leg_id=sl_leg_id,
                            trade_db_id=trade['id'],
                            entry_price=fill,
                            risk_per_share=fill - real_sl,
                            trail_r=trail_r,
                            activate_at_r=activate_r,
                            # Crash recovery: avg_flag_volume not in DB; passing
                            # 0.0 → vol-confirm helper falls back to always-fire.
                            # Worst case: recovered watches miss the vol-conf
                            # hold benefit until next fresh trade.
                            avg_flag_volume=0.0,
                            vol_confirmed_trail_enabled=self.vol_confirmed_trail_enabled,
                            vol_confirmed_trail_min_ratio=self.vol_confirmed_trail_min_ratio,
                            planned_entry_price=_planned_entry_db,
                            planned_risk_per_share=_planned_R_db,
                            r_basis=self.trail_r_basis,
                            # 2026-09-06 profit partial (shared spec) — 0 = off
                            pp_r_multiple=(self.profit_partial.r_multiple
                                           if self.profit_partial.enabled else 0.0),
                            pp_fraction=self.profit_partial.fraction,
                            pp_breakeven=self.profit_partial.move_to_breakeven,
                            # Crash recovery: exclude the original fill
                            # minute (usually long past → no-op).
                            skip_exits_until_ts=_end_of_minute_epoch(
                                trade.get('filled_at')),
                        )
                        if self.profit_partial.enabled:
                            self.stop_monitor.arm_profit_partial(
                                symbol, self.profit_partial.r_multiple)
                        logger.info(
                            f"{symbol}: Crash recovery — re-registered StopMonitor watch "
                            f"stop=${real_sl:.2f}"
                            f"{f', trail={trail_r:.1f}R' if trail_r > 0 else ''}"
                        )
                    except Exception as e:
                        logger.error(
                            f"{symbol}: Crash recovery — failed to re-register "
                            f"StopMonitor watch: {e} (safety-net SL active)"
                        )

        # 2026-06-05: cross-strategy orphan reconciler. Run BEFORE the
        # _close_orphan_positions registration step so the reconciler
        # sees prior-day broker positions as candidates (not as already-
        # tracked). _close_orphan_positions then registers any survivors
        # in _traded_symbols to prevent same-day re-entry.
        try:
            # tracked = symbols the engine is ACTIVELY managing right now.
            # At startup we have no in-flight bull flag positions, so any
            # broker position belongs in the orphan candidates pool. Past
            # this point, anything still on the broker is either ours and
            # got close-submitted, or foreign and got alerted.
            today_symbols = {t['symbol'] for t in trades_today
                             if t.get('fill_price') is not None}
            # Shared-account exclusion (2026-08-14): ignition trades the
            # SAME Alpaca account. Its open positions are tracked by its
            # own engine/reconciler — without this, a mid-session restart
            # while ignition holds positions fires a spurious 'foreign
            # orphan' alert per ignition symbol.
            try:
                today_symbols |= {
                    t['symbol'] for t in self.db.get_open_trades(
                        date.today().isoformat(), strategy='ignition')
                    if t.get('exit_price') is None}
            except Exception as e:
                logger.warning(
                    f"bull_flag reconciler: ignition sibling query "
                    f"failed ({e}) — foreign-alert noise possible")
            reconcile_strategy_orphans(
                strategy='bull_flag', alpaca=self.alpaca, db=self.db,
                notifier=self.notifier, tracked_symbols=today_symbols,
                cfg=getattr(self, 'orphan_reconciler_cfg',
                             None) or ReconcilerConfig(),
            )
        except Exception as e:
            logger.error(f"bull_flag orphan reconciler raised: {e}")

        # Detect orphan positions from prior days (registration only —
        # close decisions are the reconciler's job, see above).
        self._close_orphan_positions(trades_today)

        logger.info(
            f"Startup sync: {len(self._traded_symbols)} traded symbols, "
            f"{self._daily_trade_count} filled trades, "
            f"{len(self._pending_orders)} pending orders recovered"
        )

    def _reconstruct_plan(self, trade: Dict[str, Any]) -> Optional['TradePlan']:
        """Reconstruct a TradePlan from DB trade fields.

        Args:
            trade: Trade dict from database

        Returns:
            TradePlan if enough data exists, None otherwise
        """
        try:
            entry = trade.get('entry_price')
            sl = trade.get('stop_loss_price')
            tp = trade.get('take_profit_price')
            shares = trade.get('shares')
            if not all([entry, sl, tp, shares]):
                return None

            risk = trade.get('risk_per_share', entry - sl)
            reward = tp - entry
            rr = trade.get('risk_reward_ratio', reward / risk if risk > 0 else 0)

            return TradePlan(
                symbol=trade['symbol'],
                entry_price=entry,
                stop_loss_price=sl,
                take_profit_price=tp,
                risk_per_share=risk,
                reward_per_share=reward,
                risk_reward_ratio=rr,
                shares=shares,
                total_risk=trade.get('total_risk', risk * shares),
                pattern=None,
            )
        except Exception as e:
            logger.warning(f"{trade.get('symbol')}: Failed to reconstruct plan: {e}")
            return None

    def _reconstruct_setup(self, trade: Dict[str, Any]) -> Optional[Any]:
        """Reconstruct a BullFlagSetup from trade's pattern_data JSON.

        Args:
            trade: Trade dict from database (with pattern_data JSON field)

        Returns:
            BullFlagSetup if pattern_data is parseable, None otherwise
        """
        import json
        from trading.pattern_detector import BullFlagSetup

        pattern_data = trade.get('pattern_data')
        if not pattern_data:
            return None

        try:
            data = json.loads(pattern_data) if isinstance(pattern_data, str) else pattern_data
            if not data or not isinstance(data, dict):
                return None

            breakout = data.get('breakout_level')
            if breakout is None:
                return None

            return BullFlagSetup(
                symbol=trade['symbol'],
                pole_start_idx=data.get('pole_start_idx', 0),
                pole_end_idx=data.get('pole_end_idx', 0),
                flag_start_idx=data.get('flag_start_idx', 0),
                flag_end_idx=data.get('flag_end_idx', 0),
                pole_low=data.get('pole_low', 0),
                pole_high=data.get('pole_high', 0),
                pole_height=data.get('pole_height', 0),
                pole_gain_pct=data.get('pole_gain_pct', 0),
                flag_low=data.get('flag_low', 0),
                flag_high=data.get('flag_high', 0),
                retracement_pct=data.get('retracement_pct', 0),
                pullback_candle_count=data.get('pullback_candle_count', 0),
                avg_pole_volume=data.get('avg_pole_volume', 0),
                avg_flag_volume=data.get('avg_flag_volume', 0),
                breakout_level=breakout,
            )
        except Exception as e:
            logger.warning(f"{trade.get('symbol')}: Failed to reconstruct setup: {e}")
            return None

    def _cancel_open_orders_for_symbol(self, symbol: str) -> int:
        """Cancel any OPEN orders for a symbol, returns count cancelled.

        Used before close_position on an orphan: Alpaca holds the shares
        for any open bracket legs (held_for_orders == qty), so a plain
        close_position fails with "insufficient qty available". Cancelling
        the legs releases the shares so close can proceed.

        Idempotent: returns 0 if there are no open orders. Per-order cancel
        failures are logged WARN but don't raise — best-effort.
        """
        import time as _time
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(
                status=QueryOrderStatus.OPEN, symbols=[symbol]
            )
            open_orders = self.alpaca.trading_client.get_orders(filter=req)
        except Exception as e:
            logger.warning(
                f"{symbol}: failed to query open orders before close: {e}"
            )
            return 0
        cancelled = 0
        for o in (open_orders or []):
            try:
                self.alpaca.cancel_order(str(o.id))
                cancelled += 1
                logger.info(
                    f"{symbol}: Cancelled open order {o.id} "
                    f"({str(o.side).split('.')[-1]} "
                    f"{str(o.order_type).split('.')[-1]} qty={o.qty}) "
                    f"before close"
                )
            except Exception as ce:
                logger.warning(
                    f"{symbol}: Failed to cancel open order {o.id}: {ce}"
                )
        if cancelled > 0:
            # Brief pause so cancels propagate and held_for_orders releases
            # before the close_position call. Alpaca's eventual consistency
            # window is typically <500ms; 1s is safe.
            _time.sleep(1.0)
        return cancelled

    def _close_orphan_positions(self, trades_today: List[Dict]) -> None:
        """Register prior-day positions in `_traded_symbols` so we don't
        re-trade them today. ACTUAL close is delegated to the shared
        orphan_reconciler (called separately from _sync_startup_state),
        which applies the hardened predicate so we never accidentally
        flatten another strategy's identical-symbol position.

        Before 2026-06-05 this method ALSO called close_position. That
        path had the same vulnerability the SMU/QBTZ post-mortem
        exposed in ORB: any DB row tagged strategy='bull_flag' within
        the lookback window claimed ownership regardless of avg-entry
        match or stale signal. An unknown-strategy or manual position
        could match by symbol coincidence and get flattened. The
        reconciler does not have this hole.

        Args:
            trades_today: Today's trades from DB (already fetched)
        """
        try:
            positions = self.alpaca.get_open_positions()
        except Exception as e:
            logger.error(f"Startup sync: failed to get Alpaca positions: {e}")
            return

        if not positions:
            return

        today_symbols = {t['symbol'] for t in trades_today}

        for pos in positions:
            symbol = pos['symbol']
            if symbol in today_symbols:
                continue  # Known today — handled by startup sync
            # Register the symbol either way so bull flag doesn't try to
            # take a fresh setup on a symbol the broker already holds —
            # whether it's ours or not. The reconciler handles the
            # close decision separately.
            self._traded_symbols.add(symbol)
            logger.info(
                f"{symbol}: pre-existing broker position registered with "
                f"_traded_symbols (close decision deferred to reconciler)"
            )
