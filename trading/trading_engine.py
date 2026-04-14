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
import queue
import time as time_mod
from datetime import date, datetime, timedelta, timezone
from typing import Set, Optional, Dict, Any, List

import pandas as pd
import pytz

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.pattern_detector import BullFlagDetector
from trading.trade_planner import TradePlanner, TradePlan
from trading.order_executor import OrderExecutor
from trading.position_manager import PositionManager
from notifications.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)

ET = pytz.timezone('US/Eastern')


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

        # Exhaustion exit config
        exhaust_cfg = _cfg.get("trading", {}).get("exhaustion_exit", {})
        self.exhaustion_exit_enabled = bool(exhaust_cfg.get("enabled", False))
        self.exhaustion_partial_fraction = float(exhaust_cfg.get("partial_fraction", 0.5))
        self.exhaustion_tighter_trail_r = float(exhaust_cfg.get("tighter_trail_r", 0.5))
        self.exhaustion_min_profit_r = float(exhaust_cfg.get("min_profit_r", 3.0))
        self.exhaustion_signals = exhaust_cfg.get("signals", {
            'volume_divergence': False,
            'climax_candle': True,
            'shrinking_bodies': False,
            'shooting_star': True,
        })

        # Minimum daily volume filter — skip illiquid stocks
        self.min_daily_volume = int(_cfg.get("scanner", {}).get("min_daily_volume", 0))

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

        # Conviction scoring: scale position size by setup quality
        conv_cfg = _cfg.get("trading", {}).get("conviction_scoring", {})
        self.conviction_enabled = bool(conv_cfg.get("enabled", False))
        # Conviction filter (skip trades below threshold). 0.0 = disabled.
        # 1.2 = walk-forward validated (+$10-18K/period OOS).
        # COUPLED to current 5 conviction rules — re-validate if rules change.
        self.conviction_min_threshold = float(conv_cfg.get("min_threshold", 0.0))
        if self.conviction_enabled:
            msg = "Conviction scoring: ENABLED (matches backtest V4 model)"
            if self.conviction_min_threshold > 0:
                msg += f" — filter trades with conv < {self.conviction_min_threshold:.2f}"
            logger.info(msg)

        # News gate: require real catalyst before trading
        news_gate_cfg = _cfg.get("trading", {}).get("news_gate", {})
        self.news_gate_enabled = bool(news_gate_cfg.get("enabled", False))
        if self.news_gate_enabled:
            logger.info("News gate: ENABLED — no catalyst = no trade")

        # News kill rules: block no-news trades in specific loser segments
        nkr_cfg = _cfg.get("trading", {}).get("news_kill_rules", {})
        self.news_kill_enabled = bool(nkr_cfg.get("enabled", False))
        self.nkr_max_avg_vol = float(nkr_cfg.get("max_avg_vol_no_news", 3_000_000))
        self.nkr_min_price = float(nkr_cfg.get("min_price_no_news", 3.0))
        self.nkr_max_float = float(nkr_cfg.get("max_float_no_news", 30_000_000))
        if self.news_kill_enabled:
            logger.info(
                f"News kill rules: ENABLED — "
                f"vol>={self.nkr_max_avg_vol/1e6:.0f}M, "
                f"price<${self.nkr_min_price:.0f}, float>={self.nkr_max_float/1e6:.0f}M "
                f"(no-news only)"
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

    def _get_macd_zone_multiplier(self, symbol: str, bars: pd.DataFrame, entry_price: float) -> float:
        """
        Compute MACD zone risk multiplier for live trading.

        Uses warmed-up MACD histogram to determine zone.

        Args:
            symbol: Stock symbol
            bars: Current day's 1-min bars (from market open)
            entry_price: Planned entry price

        Returns:
            0.0 = skip (dead zone), 1.0 = normal, >1.0 = boosted
        """
        from trading.indicators import macd_histogram

        closes = bars['close'].copy()

        # Prepend warm-up from previous day
        if symbol in self._macd_warmup_cache and self._macd_warmup_cache[symbol] is not None:
            closes = pd.concat([self._macd_warmup_cache[symbol], closes], ignore_index=True)

        if len(closes) < 35:
            return 1.0

        hist = macd_histogram(closes)
        hist_val = float(hist.iloc[-1])
        macd_pct = (hist_val / entry_price) * 100

        if self.macd_dead_zone_min <= macd_pct <= self.macd_dead_zone_max:
            logger.info(f"{symbol}: MACD zone dead ({macd_pct:.2f}%)")
            return 0.0
        elif macd_pct < self.macd_strong_neg_threshold:
            logger.info(f"{symbol}: MACD zone strong neg ({macd_pct:.2f}%) → {self.macd_strong_neg_multiplier}x")
            return self.macd_strong_neg_multiplier
        elif macd_pct > self.macd_strong_pos_threshold:
            logger.info(f"{symbol}: MACD zone strong pos ({macd_pct:.2f}%) → {self.macd_strong_pos_multiplier}x")
            return self.macd_strong_pos_multiplier
        else:
            return self.macd_normal_multiplier

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
        self, setup, spy_3d_range: float, return_breakdown: bool = False,
    ):
        """Compute conviction score at setup detection time.

        Returns a multiplier (0.25 to 3.0) that scales position size.
        Matches backtest.py exactly — 5 pattern rules, no news scoring.

        Args:
            setup: BullFlagSetup object
            spy_3d_range: SPY 3-day average daily range (%)
            return_breakdown: If True, return (final_score, breakdown_dict).
                breakdown_dict has per-rule contributions plus 'raw_score'
                (pre-clamp) and 'final_score' (post-clamp). For trace logging.

        Returns:
            float (when return_breakdown=False) — the position multiplier
            tuple (float, dict) — when return_breakdown=True
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

        # 4. SPY 3d range regime
        if spy_3d_range > 1.2:
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

        final = max(0.25, min(3.0, score))
        if return_breakdown:
            breakdown['raw_score'] = score
            breakdown['final_score'] = final
            return final, breakdown
        return final

    def _get_spy_3d_range_live(self) -> float:
        """Get SPY 3-day avg daily range. Uses market_regime's SPY daily bars if available."""
        # Try market_regime first (has SPY daily bars from _refresh_spy_data)
        if hasattr(self, 'market_regime') and self.market_regime:
            try:
                spy_bars = self.market_regime._spy_bars
                if spy_bars and len(spy_bars) >= 3:
                    ranges = []
                    for b in spy_bars[-3:]:
                        h = b.get('high', 0); l = b.get('low', 0)
                        if l > 0:
                            ranges.append((h - l) / l * 100)
                    if ranges:
                        return sum(ranges) / len(ranges)
            except Exception:
                pass
        # Fallback: today's range from cached 1-min SPY bars
        if self._spy_bars_cache is not None and len(self._spy_bars_cache) > 1:
            day_high = float(self._spy_bars_cache['high'].max())
            day_low = float(self._spy_bars_cache['low'].min())
            if day_low > 0:
                return (day_high - day_low) / day_low * 100
        return 1.0  # default neutral

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
                           news_category: str = None) -> None:
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
        """
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
                        symbols_to_remove.append(symbol)
                        continue

                # Post-fill exit: in calm markets + weak breakout vol → close immediately
                # Matches backtest.py post-fill exit logic
                if self.conviction_enabled and setup:
                    _afv = float(setup.avg_flag_volume) if hasattr(setup, 'avg_flag_volume') else 0
                    _spy_3d = self._get_spy_3d_range_live()
                    if _afv > 0 and _spy_3d < 0.8:
                        # Get recent bar volume (the fill bar)
                        try:
                            _recent_bars = self.alpaca.get_1min_bars(symbol, lookback_minutes=2)
                            _bk_vol = float(_recent_bars.iloc[-1]['volume']) if _recent_bars is not None and len(_recent_bars) > 0 else 0
                        except Exception:
                            _bk_vol = 0
                        _bk_ratio = _bk_vol / _afv if _afv > 0 else 99
                        if _bk_ratio < 1.0:
                            logger.warning(
                                f"{symbol}: POST-FILL EXIT — SPY 3d {_spy_3d:.2f}% + "
                                f"bk_vol {_bk_ratio:.1f}x → closing immediately")
                            try:
                                self.alpaca.close_position(symbol)
                                if self.notifier:
                                    self.notifier.notify_error(
                                        f"{symbol}: Post-fill exit (calm market + weak breakout vol)",
                                        component="PostFillExit")
                            except Exception as e:
                                logger.error(f"{symbol}: Failed to close post-fill exit: {e}")
                            symbols_to_remove.append(symbol)
                            continue

                self._traded_symbols.add(symbol)
                self.position_manager.mark_traded(symbol)
                symbols_to_remove.append(symbol)

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
                        self.stop_monitor.remove_quote_watch(symbol)

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
                is_simple_order = pending.get('order_type') == 'stop_simple'
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
                                exit_reason='gap_adjust_failed',
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

                        is_simple_order = pending.get('order_type') == 'stop_simple'

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

                        self.stop_monitor.add_watch(
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
                        )

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
                logger.info(f"{symbol}: Pending order {status} — ID: {order_id}")
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
            # Exhaustion partials are processed separately
            if event.exit_reason == 'exhaustion_partial':
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

        watched = self.stop_monitor.watched_symbols
        if not watched:
            return

        for symbol in watched:
            snapshot = self.stop_monitor.get_watch_snapshot(symbol)
            if not snapshot:
                continue

            if snapshot['exhaustion_partial_taken']:
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
                        'partial_exit_reason': 'exhaustion',
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
                            exit_reason='exhaustion_partial',
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

    def _sync_closed_positions(self) -> None:
        """Detect bracket exits (SL/TP hit) and update DB + circuit breaker."""
        # Process StopMonitor exits first — updates DB with exit_price.
        # Must happen BEFORE we fetch open_trades, otherwise trades just
        # closed by StopMonitor still appear as "open" and get double-processed.
        self._process_stop_monitor_exits()

        # Check exhaustion exit signals on active positions (every 60s cycle)
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
                            exit_reason = 'stop_loss'
                        # Check TP leg
                        elif tp_leg and tp_leg.get('status') == 'filled':
                            fill = tp_leg.get('filled_avg_price')
                            exit_price = fill or tp_leg['limit_price']
                            exit_reason = 'take_profit'

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
                        # Use fill_price as fallback exit to prevent infinite re-check
                        # (exit_price IS NULL keeps this trade in get_open_trades forever)
                        fallback_exit = trade['fill_price']
                        pnl_est = 0.0  # Assume breakeven if unknown
                        error_msg = (
                            f"{symbol}: Position closed but exit price unknown — "
                            f"using fill_price ${fallback_exit:.2f} as estimate"
                        )
                        logger.warning(error_msg)
                        if self.notifier:
                            self.notifier.notify_error(error_msg, component="PositionSync")
                        self.db.update_trade(trade['id'], {
                            'exit_price': fallback_exit,
                            'exit_reason': 'unknown_exit',
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

        # Clear per-cycle caches (marginability)
        self._margin_cache = {}

        # ALWAYS sync positions and manage pending orders — these must run
        # regardless of regime filter or max trades. Skipping them means
        # SL/TP exits go unrecorded, PnL is wrong, and circuit breaker is deaf.
        self._sync_closed_positions()
        fill_result = self._manage_pending_orders()

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

        # Risk tier: scale risk on high-conviction setups
        risk_multiplier = 1.0
        if self.risk_tiers_enabled:
            avg_vol = (uni_stock.get('avg_volume_daily') or 0) if uni_stock else 0
            risk_multiplier = self._get_risk_tier(setup.breakout_level, avg_vol)

            # Check marginability for leveraged trades (real-time, cached per cycle)
            if risk_multiplier > 1.0:
                if not hasattr(self, '_margin_cache'):
                    self._margin_cache = {}
                if symbol not in self._margin_cache:
                    self._margin_cache[symbol] = self.alpaca.is_marginable(symbol)
                if not self._margin_cache[symbol]:
                    logger.info(
                        f"{symbol}: Not marginable — falling back to 1x "
                        f"(wanted {risk_multiplier:.1f}x)"
                    )
                    risk_multiplier = 1.0

        # News classification: use scanner's cached result (from on_stock_qualified).
        # LLM re-check removed — was 2-5s in critical order path. News kill rules
        # handle no-news risk, scanner classification is sufficient.

        # News kill rules: block no-news trades in specific loser segments
        # Matches backtest.py _check_news_kill() logic exactly
        if self.news_kill_enabled:
            _ndata = self._news_data.get(symbol, {})
            _ncat = _ndata.get('news_category', 'NO_NEWS')
            # Only kill confirmed no-news. PENDING/ERROR get benefit of the doubt.
            _no_news_cats = {'NO_NEWS', 'GARBAGE_RECAP', 'OTHER'}
            if _ncat in _no_news_cats:
                _avg_vol = (uni_stock.get('avg_volume_daily') or 0) if uni_stock else 0
                _float = (uni_stock.get('float_shares') or 0) if uni_stock else 0
                _ep = setup.breakout_level
                _pg = setup.pole_gain_pct
                _kill_reason = None
                if _avg_vol >= self.nkr_max_avg_vol:
                    _kill_reason = f"no_news + avg_vol {_avg_vol/1e6:.1f}M"
                elif _ep < self.nkr_min_price:
                    _kill_reason = f"no_news + price ${_ep:.2f}"
                elif _float >= self.nkr_max_float:
                    _kill_reason = f"no_news + float {_float/1e6:.0f}M"
                elif 5 <= _ep < 12 and 8 <= _pg < 15:
                    _kill_reason = f"no_news + ${_ep:.0f} + pole {_pg:.1f}%"
                if _kill_reason:
                    logger.info(f"{symbol}: NEWS KILL: {_kill_reason}")
                    self._eod_skipped.append((symbol, _ncat, _kill_reason))
                    return None

        # Conviction scoring: combine with risk tier, cap at 3x
        conviction_mult = 1.0
        if self.conviction_enabled:
            spy_3d = self._get_spy_3d_range_live()
            conviction_mult = self._compute_conviction_score_setup(setup, spy_3d)
            if abs(conviction_mult - 1.0) > 0.05:
                logger.info(
                    f"{symbol}: Conviction {conviction_mult:.2f}x "
                    f"(pole={setup.pole_gain_pct:.1f}%, "
                    f"retr={setup.retracement_pct:.0f}%, SPY3d={spy_3d:.1f}%)")

            # Conviction filter: skip trades below quality threshold.
            # Walk-forward validated: conv<1.2 setups = 28% WR, -0.18R avg (negative EV).
            if (self.conviction_min_threshold > 0
                    and conviction_mult < self.conviction_min_threshold):
                _, brkdn = self._compute_conviction_score_setup(
                    setup, spy_3d, return_breakdown=True)
                breakdown_str = (
                    f"pole={brkdn['pole_gain']:+.1f} "
                    f"flag={brkdn['flag_tightness']:+.1f} "
                    f"vol={brkdn['vol_ratio']:+.1f} "
                    f"spy={brkdn['spy_regime']:+.1f} "
                    f"retr={brkdn['retracement']:+.1f}"
                )
                logger.info(
                    f"{symbol}: CONVICTION SKIP: {conviction_mult:.2f} < "
                    f"{self.conviction_min_threshold:.2f} "
                    f"({breakdown_str}; raw={brkdn['raw_score']:.2f})"
                )
                self._eod_skipped.append((
                    symbol, "LOW_CONVICTION",
                    f"conv {conviction_mult:.2f} ({breakdown_str})"
                ))
                return None

        combined_mult = min(3.0, risk_multiplier * conviction_mult)

        # Create trade plan (pass ADV for liquidity cap — matches BT)
        _adv = int(uni_stock.get('avg_volume_daily') or 0) if uni_stock else 0
        plan = self.planner.create_plan(setup, avg_daily_volume=_adv, risk_multiplier=combined_mult)
        if plan is None:
            return None

        # Buying power check: reduce size if needed, never skip
        # Query real-time buying power (reflects pending orders already reserved)
        try:
            buying_power = self.alpaca.get_buying_power()
            position_cost = plan.entry_price * plan.shares
            if position_cost > buying_power and buying_power > 0:
                affordable_shares = int(buying_power / plan.entry_price)
                if affordable_shares >= 1:
                    logger.info(
                        f"{symbol}: Reducing {plan.shares} → {affordable_shares} shares "
                        f"(buying power ${buying_power:,.0f} < cost ${position_cost:,.0f})"
                    )
                    # Recreate plan with reduced multiplier
                    reduced_mult = risk_multiplier * (affordable_shares / plan.shares)
                    plan = self.planner.create_plan(setup, avg_daily_volume=_adv, risk_multiplier=max(reduced_mult, 0.1))
                    if plan is None:
                        return None
                else:
                    logger.warning(f"{symbol}: No buying power for even 1 share, skipping")
                    return None
        except Exception as e:
            logger.warning(f"{symbol}: Buying power check failed: {e} — proceeding with plan")

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
        if self.macd_zones_enabled:
            zone_mult = self._get_macd_zone_multiplier(symbol, bars, plan.entry_price)
            if zone_mult == 0.0:
                logger.info(f"{symbol}: MACD zone SKIP (dead zone)")
                return None
            elif zone_mult != 1.0 and risk_multiplier <= 1.0:
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

        real_stop_level = plan.stop_loss_price
        if self.stop_monitor:
            # Simple stop-limit (no bracket) — avoids 3x margin reservation
            # Safety-net SL submitted separately after fill detection
            logger.info(
                f"{symbol}: Self-managed stops — real stop ${real_stop_level:.2f}, "
                f"safety-net SL after fill ({self.safety_net_sl_pct:.0%})"
            )
            result = self.executor.submit_buy_stop_order(plan)
        else:
            # Bracket order — SL/TP legs provide protection without StopMonitor
            # No sl_override: bracket SL = plan.stop_loss_price (real stop)
            result = self.executor.submit_buy_stop_bracket_order(plan)

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

    def _force_close_all(self) -> None:
        """
        Cancel all pending orders and close all open positions.

        Called at force_close_time to ensure we're flat before market close.
        Syncs closed positions first so any SL/TP exits that already happened
        are recorded before we attempt to close remaining positions.
        """
        # Stop StopMonitor before force-closing — prevents race conditions
        # where monitor tries to exit while we're also closing
        if self.stop_monitor:
            for symbol in list(self.stop_monitor.watched_symbols):
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
                        'exit_reason': 'force_close',
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
                        exit_reason='force_close',
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

        # Graceful shutdown: force-close all positions
        if self.shutdown_event and self.shutdown_event.is_set():
            logger.info("Shutdown signal received — force-closing all positions...")
            self._force_close_all()
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
        self.position_manager.reset_daily()
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
            trades_today = self.db.get_trades_by_date(today)
        except Exception as e:
            logger.error(f"Startup sync: failed to load today's trades: {e}")
            return

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

        # Rebuild _pending_orders from DB trades that have order_id but no fill
        for trade in trades_today:
            symbol = trade['symbol']
            order_id = trade.get('order_id')
            if order_id and trade.get('fill_price') is None and trade.get('exit_price') is None:
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
                        )
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

        # Detect orphan positions from prior days
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

    def _close_orphan_positions(self, trades_today: List[Dict]) -> None:
        """Detect and close positions from prior days opened by THIS node.

        An orphan is an Alpaca position that THIS node opened (has a record
        in our trades DB) but failed to close (e.g., service crashed after
        market close without running force_close).

        Positions with NO record in our DB are assumed to belong to another
        node/strategy sharing the same Alpaca account — leave them alone.

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

            # Check if THIS node has any record of this position in our DB
            # (open trade from today or any prior day)
            our_trade = self.db.get_trade_by_symbol(symbol) if self.db and hasattr(self.db, 'get_trade_by_symbol') else None
            if our_trade is None and self.db:
                # Fallback: check all recent trades
                from datetime import timedelta
                for days_back in range(5):
                    _check_date = (date.today() - timedelta(days=days_back)).isoformat()
                    _trades = self.db.get_trades_by_date(_check_date)
                    if any(t['symbol'] == symbol for t in _trades):
                        our_trade = True
                        break

            if our_trade:
                # We opened it but didn't close — orphan from prior day
                logger.warning(f"{symbol}: Orphan position from prior day (ours) — closing")
                try:
                    self.alpaca.close_position(symbol)
                    self._traded_symbols.add(symbol)
                    logger.info(f"{symbol}: Orphan position closed")
                except Exception as e:
                    logger.error(f"{symbol}: Failed to close orphan position: {e}")
            else:
                # Not ours — belongs to another node/strategy
                logger.info(f"{symbol}: Position exists but not ours — skipping (other node/strategy)")
                self._traded_symbols.add(symbol)  # Prevent this node from trading it
