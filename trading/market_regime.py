"""
Market regime filter — blocks trading under hostile SPY conditions.

Two independent filtering mechanisms:
1. Volatility + downtrend: SPY 5-day avg daily range > threshold AND below SMA.
   When BOTH conditions are true, the market is hostile to momentum trading.

2. Thin liquidity (H5 OR filter): On thin-liquidity days (SPY T-1 volume ratio
   < threshold), the minimum breakout volume ratio is raised. This doesn't block
   all trades — only trades where the breakout also lacks conviction. Trades with
   strong breakout volume are still allowed.

Also exposes max_trades_per_day for callers (TradingEngine, batch backtest).

Shared by backtest and production. Stateless after loading bars.
"""

import logging
from datetime import date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MarketRegimeFilter:
    """
    Filters trading days based on SPY volatility + trend regime and liquidity.

    Blocking (entire date skipped) when BOTH:
    - 5-day average daily range > vol_threshold (high volatility)
    - T-1 close < SMA(sma_period) (downtrend)

    Tightening (stricter breakout volume required) when:
    - SPY T-1 volume / SMA20(volume) < min_spy_volume_ratio (thin liquidity)

    Lookahead prevention: all indicators use data strictly BEFORE trade_date (T).
    For live trading at 9:30 AM on day T, T-1's close is yesterday's close — fully known.
    """

    def __init__(
        self,
        enabled: bool = True,
        vol_threshold: float = 1.5,
        sma_period: int = 50,
        max_trades_per_day: int = 5,
        min_spy_volume_ratio: float = 0.70,
        thin_liquidity_breakout_vol_ratio: float = 2.0,
        sma_slope_filter: bool = False,
        sma_slope_threshold: float = -0.5,
        euphoria_filter: bool = False,
        euphoria_ud_threshold: float = 1.2,
        euphoria_rsi_threshold: float = 60.0,
    ):
        """
        Initialize MarketRegimeFilter.

        Args:
            enabled: Whether the filter is active. When False, all days pass.
            vol_threshold: 5-day avg daily range % threshold. When exceeded
                AND below SMA, trading is blocked.
            sma_period: SMA period for trend detection (default 50).
            max_trades_per_day: Maximum trades allowed per day (used by callers).
            min_spy_volume_ratio: SPY T-1 volume / SMA20(volume) threshold.
                Below this = thin liquidity day — breakout volume requirement
                is raised to thin_liquidity_breakout_vol_ratio.
            thin_liquidity_breakout_vol_ratio: Minimum breakout volume ratio
                required on thin liquidity days (default 2.0).
            sma_slope_filter: Also block when SMA50 slope < 0 (dead-cat-bounce
                filter). Catches periods where price is above SMA50 but the
                moving average itself is still declining.
            euphoria_filter: Block when SPY up/down volume ratio > threshold
                AND RSI > threshold. Bullish euphoria crowds breakout trades —
                FOMO buyers dump, stops get tagged. Uses T-1 data (no look-ahead).
            euphoria_ud_threshold: Up/Down volume ratio threshold (10-day window).
            euphoria_rsi_threshold: RSI(14) threshold.
        """
        self.enabled = enabled
        self.vol_threshold = vol_threshold
        self.sma_period = sma_period
        self.max_trades_per_day = max_trades_per_day
        self.min_spy_volume_ratio = min_spy_volume_ratio
        self.thin_liquidity_breakout_vol_ratio = thin_liquidity_breakout_vol_ratio
        self.sma_slope_filter = sma_slope_filter
        self.sma_slope_threshold = sma_slope_threshold
        self.euphoria_filter = euphoria_filter
        self.euphoria_ud_threshold = euphoria_ud_threshold
        self.euphoria_rsi_threshold = euphoria_rsi_threshold
        self._bars_by_date: Dict[date, Dict[str, float]] = {}  # date -> {open, high, low, close, volume}
        self._sorted_dates: List[date] = []

    def load_spy_bars(self, spy_bars: List[Dict]) -> None:
        """
        Load SPY daily bars (OHLCV).

        Args:
            spy_bars: List of bar dicts with 'date' (date obj or str),
                'open', 'high', 'low', 'close' (floats), and optionally
                'volume' (int/float).
                Backwards compatible: bars with only 'close' also work.
        """
        self._bars_by_date.clear()
        for bar in spy_bars:
            bar_date = bar['date']
            if isinstance(bar_date, str):
                bar_date = date.fromisoformat(bar_date)
            self._bars_by_date[bar_date] = {
                'open': float(bar.get('open', 0)),
                'high': float(bar.get('high', 0)),
                'low': float(bar.get('low', 0)),
                'close': float(bar['close']),
                'volume': float(bar.get('volume', 0)),
            }

        self._sorted_dates = sorted(self._bars_by_date.keys())
        logger.info(
            f"MarketRegimeFilter loaded {len(self._sorted_dates)} SPY bars "
            f"({self._sorted_dates[0]} to {self._sorted_dates[-1]})"
            if self._sorted_dates else
            "MarketRegimeFilter loaded 0 SPY bars"
        )

    def _get_prior_dates(self, trade_date: date) -> List[date]:
        """Get all trading dates strictly before trade_date, sorted ascending."""
        return [d for d in self._sorted_dates if d < trade_date]

    def get_recent_bars(
        self, n: int = 3, before_date: Optional[date] = None
    ) -> List[Dict]:
        """Return the most-recent up-to-`n` SPY bars in ascending date order.

        Public API for consumers that need raw bars (e.g., the SPY-3d-range
        helper used by conviction scoring). Replaces direct access to the
        internal `_bars_by_date` / `_sorted_dates` attributes — see post-mortem
        on EAF 2026-05-01 where the live conviction reader reached for a
        non-existent `_spy_bars` attribute and silently fell back to a 1.0
        sentinel.

        Args:
            n: Maximum number of bars to return.
            before_date: If provided, only consider bars strictly before this
                date (look-ahead prevention for backtests). If None, return
                up to `n` most recent bars regardless of date.

        Returns:
            List of bar dicts each containing 'date', 'open', 'high', 'low',
            'close', 'volume'. Ascending date order. Empty list if no matching
            bars exist.
        """
        if before_date is not None:
            candidates = [d for d in self._sorted_dates if d < before_date]
        else:
            candidates = list(self._sorted_dates)
        if not candidates:
            return []
        selected = candidates[-n:] if n > 0 else []
        return [
            {**self._bars_by_date[d], 'date': d} for d in selected
        ]

    def get_latest_bar_date(self) -> Optional[date]:
        """Most-recent SPY bar date, or None if none loaded.

        Used by callers running a staleness check (e.g.,
        `trading.spy_regime.is_spy_data_stale`) without exposing the internal
        sorted-dates list.
        """
        return self._sorted_dates[-1] if self._sorted_dates else None

    def get_spy_vol_5d(self, trade_date: date) -> Optional[float]:
        """
        Average daily range % over 5 trading days BEFORE trade_date.

        daily_range = (high - low) / close * 100
        Returns mean of last 5 completed days, or None if < 5 bars.

        Args:
            trade_date: The date we want to trade on.

        Returns:
            Average daily range percentage, or None if insufficient data.
        """
        prior_dates = self._get_prior_dates(trade_date)
        if len(prior_dates) < 5:
            return None

        recent_5 = prior_dates[-5:]
        ranges = []
        for d in recent_5:
            bar = self._bars_by_date[d]
            close = bar['close']
            if close == 0:
                continue
            daily_range = (bar['high'] - bar['low']) / close * 100
            ranges.append(daily_range)

        if not ranges:
            return None

        return sum(ranges) / len(ranges)

    def get_spy_sma(self, trade_date: date, period: Optional[int] = None) -> Optional[float]:
        """
        SMA of closes for `period` trading days BEFORE trade_date.

        Args:
            trade_date: The date we want to trade on.
            period: Number of days for SMA. Defaults to self.sma_period.

        Returns:
            SMA value, or None if fewer than `period` prior bars.
        """
        if period is None:
            period = self.sma_period

        prior_dates = self._get_prior_dates(trade_date)
        if len(prior_dates) < period:
            return None

        recent = prior_dates[-period:]
        total = sum(self._bars_by_date[d]['close'] for d in recent)
        return total / period

    def is_below_sma(self, trade_date: date) -> Optional[bool]:
        """
        Check if T-1 close is below SMA(sma_period).

        Args:
            trade_date: The date we want to trade on.

        Returns:
            True if T-1 close < SMA, False if >= SMA, None if insufficient data.
        """
        prior_dates = self._get_prior_dates(trade_date)
        if not prior_dates:
            return None

        sma = self.get_spy_sma(trade_date)
        if sma is None:
            return None

        t1_close = self._bars_by_date[prior_dates[-1]]['close']
        return t1_close < sma

    def get_spy_sma_slope(self, trade_date: date, lookback: int = 5) -> Optional[float]:
        """
        5-day change in SMA50 value — detects dead-cat-bounce conditions.

        When SMA50 is declining, price crossing above it is a trap, not
        a real recovery. Momentum breakouts fail in this regime.

        Uses data strictly before trade_date (no look-ahead).

        Args:
            trade_date: The date to check.
            lookback: Number of days to measure SMA change over.

        Returns:
            SMA50 slope (change over lookback days), or None if insufficient data.
        """
        sma_today = self.get_spy_sma(trade_date)
        if sma_today is None:
            return None

        # Get SMA from `lookback` trading days earlier
        prior_dates = self._get_prior_dates(trade_date)
        if len(prior_dates) < lookback:
            return None

        # Find the date `lookback` days before the last prior date
        earlier_date_idx = len(prior_dates) - lookback
        if earlier_date_idx < self.sma_period:
            return None

        # Compute SMA at that earlier point
        earlier_dates = prior_dates[:earlier_date_idx + 1]
        if len(earlier_dates) < self.sma_period:
            return None
        recent = earlier_dates[-self.sma_period:]
        sma_earlier = sum(self._bars_by_date[d]['close'] for d in recent) / self.sma_period

        return sma_today - sma_earlier

    def is_sma_slope_negative(self, trade_date: date, lookback: int = 5) -> Optional[bool]:
        """
        Check if SMA50 slope is below threshold (declining trend).

        Args:
            trade_date: The date to check.
            lookback: Number of days to measure slope over.

        Returns:
            True if slope < threshold, False otherwise, None if insufficient data.
        """
        slope = self.get_spy_sma_slope(trade_date, lookback)
        if slope is None:
            return None
        return slope < self.sma_slope_threshold

    def get_spy_rsi(self, trade_date: date, period: int = 14) -> Optional[float]:
        """
        RSI(14) of SPY closes using data strictly before trade_date.

        Args:
            trade_date: The date to check.
            period: RSI lookback period.

        Returns:
            RSI value (0-100), or None if insufficient data.
        """
        prior_dates = self._get_prior_dates(trade_date)
        if len(prior_dates) < period + 1:
            return None

        closes = [self._bars_by_date[d]['close'] for d in prior_dates[-(period + 1):]]
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [max(d, 0) for d in deltas]
        losses = [max(-d, 0) for d in deltas]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def get_spy_ud_volume_ratio(self, trade_date: date, window: int = 10) -> Optional[float]:
        """
        Up/Down volume ratio over window days before trade_date.

        Up volume = volume on days where close > prev_close.
        Down volume = volume on days where close <= prev_close.
        Ratio > 1.0 = bullish volume dominance (euphoria risk).

        Args:
            trade_date: The date to check.
            window: Number of trading days to look back.

        Returns:
            UD ratio, or None if insufficient data.
        """
        prior_dates = self._get_prior_dates(trade_date)
        if len(prior_dates) < window + 1:
            return None

        dates = prior_dates[-(window + 1):]
        up_vol = 0
        down_vol = 0
        for i in range(1, len(dates)):
            bar = self._bars_by_date[dates[i]]
            prev_bar = self._bars_by_date[dates[i - 1]]
            vol = bar.get('volume', 0) or 0
            if bar['close'] > prev_bar['close']:
                up_vol += vol
            else:
                down_vol += vol
        if down_vol == 0:
            return 10.0  # effectively infinite
        return up_vol / down_vol

    def is_euphoria(self, trade_date: date) -> bool:
        """
        Check if market is in bullish euphoria — up/down volume ratio AND
        RSI both elevated. When both are high, momentum breakouts get
        crowded and fail (FOMO chasers dump).

        Uses T-1 data only (no look-ahead).

        Returns:
            True if euphoria detected, False otherwise.
        """
        if not self.euphoria_filter:
            return False

        rsi = self.get_spy_rsi(trade_date)
        if rsi is None:
            return False

        ud = self.get_spy_ud_volume_ratio(trade_date)
        if ud is None:
            return False

        return ud > self.euphoria_ud_threshold and rsi > self.euphoria_rsi_threshold

    def get_spy_volume_ratio(self, trade_date: date) -> Optional[float]:
        """
        SPY T-1 volume / SMA20(volume) — market-wide liquidity proxy.

        Holidays and thin-liquidity days show SPY volume 30-40% below normal.
        These days produce structurally worse momentum trades (immediate stop-outs).

        Uses T-1 data to avoid lookahead bias.

        Args:
            trade_date: The date we want to trade on.

        Returns:
            Volume ratio (1.0 = normal), or None if insufficient data.
        """
        prior_dates = self._get_prior_dates(trade_date)
        if len(prior_dates) < 20:
            return None

        t1 = prior_dates[-1]
        t1_volume = self._bars_by_date[t1]['volume']
        if t1_volume <= 0:
            return None

        # SMA20 of volume ending at T-1
        recent_20 = prior_dates[-20:]
        volumes = [self._bars_by_date[d]['volume'] for d in recent_20]
        volumes = [v for v in volumes if v > 0]
        if not volumes:
            return None

        sma20_vol = sum(volumes) / len(volumes)
        if sma20_vol <= 0:
            return None

        return t1_volume / sma20_vol

    def is_thin_liquidity(self, trade_date: date) -> bool:
        """
        Check if trade_date falls on a thin liquidity day.

        Thin liquidity = SPY T-1 volume / SMA20(volume) < min_spy_volume_ratio.
        On these days, breakout volume requirements are tightened (H5 OR filter).

        Returns False (not thin) when filter is disabled or data is insufficient.

        Args:
            trade_date: The date to check.

        Returns:
            True if thin liquidity, False otherwise.
        """
        if not self.enabled:
            return False

        spy_vol_ratio = self.get_spy_volume_ratio(trade_date)
        if spy_vol_ratio is None:
            return False

        return spy_vol_ratio < self.min_spy_volume_ratio

    def get_min_breakout_volume_ratio(self, trade_date: date, default: float = 1.5) -> float:
        """
        Return the effective minimum breakout volume ratio for trade_date.

        On thin liquidity days, returns thin_liquidity_breakout_vol_ratio (default 2.0).
        On normal days, returns the caller-provided default (typically 1.5).

        This implements the H5 OR filter: block only when BOTH SPY volume is thin
        AND breakout volume is weak. Strong breakout volume overrides thin liquidity.

        Args:
            trade_date: The date to check.
            default: Breakout volume ratio for normal days (from config).

        Returns:
            Effective minimum breakout volume ratio.
        """
        if self.is_thin_liquidity(trade_date):
            logger.info(
                f"Thin liquidity on {trade_date}: raising min breakout vol ratio "
                f"from {default:.1f} to {self.thin_liquidity_breakout_vol_ratio:.1f}"
            )
            return self.thin_liquidity_breakout_vol_ratio

        return default

    def is_regime_ok(self, trade_date: date) -> bool:
        """
        Check if trading is allowed on trade_date.

        Blocks when SPY 5-day avg daily range > vol_threshold.
        Extreme volatility (5%+ range) indicates regime chaos where
        momentum breakouts fail. Validated on 16mo:
        - Threshold 5.0%: +$21K P&L, -30% Max DD, 9 trades blocked
        - Touches only Feb 2026 (chaos) + 1 Apr 2025 day; 14 other months untouched
        - SMA-below condition removed: trendy down-months (Mar 2025) still had
          big winners. Only chaos (not direction) kills our setups.

        Returns True (trading allowed) when:
        - Filter is disabled
        - Insufficient data (safe default — better to trade than miss)
        - vol <= threshold

        Args:
            trade_date: The date to check.

        Returns:
            True if trading is allowed, False if blocked.
        """
        if not self.enabled:
            return True

        vol = self.get_spy_vol_5d(trade_date)
        if vol is None:
            return True

        # Block on extreme volatility regardless of direction
        if vol > self.vol_threshold:
            return False

        # SMA slope filter: block when SMA50 is declining (dead-cat-bounce)
        if self.sma_slope_filter:
            slope_neg = self.is_sma_slope_negative(trade_date)
            if slope_neg is True:
                return False

        # Euphoria filter: block when UD volume ratio AND RSI both elevated
        if self.is_euphoria(trade_date):
            return False

        return True

    def get_regime_info(self, trade_date: date) -> dict:
        """
        Return structured regime info for logging.

        Args:
            trade_date: The date to check.

        Returns:
            Dict with vol_5d, sma, is_below_sma, spy_volume_ratio,
            is_thin_liquidity, is_ok keys.
        """
        vol = self.get_spy_vol_5d(trade_date)
        sma = self.get_spy_sma(trade_date)
        below = self.is_below_sma(trade_date)
        spy_vol_ratio = self.get_spy_volume_ratio(trade_date)
        thin = self.is_thin_liquidity(trade_date)
        ok = self.is_regime_ok(trade_date)

        return {
            'vol_5d': vol,
            'sma': sma,
            'is_below_sma': below,
            'spy_volume_ratio': spy_vol_ratio,
            'is_thin_liquidity': thin,
            'is_ok': ok,
        }


# ---------------------------------------------------------------------------
# SPY MACD Afternoon Cutoff
# ---------------------------------------------------------------------------


def compute_spy_macd_for_day(
    spy_1min_bars,
    prev_day_bars=None,
) -> Dict[tuple, float]:
    """
    Compute SPY 1-min MACD histogram for each minute of the trading day.

    Uses prev_day_bars tail(60) for warm-up (continuous MACD, same as
    per-stock MACD warmup pattern in pattern_detector.py).

    Args:
        spy_1min_bars: DataFrame with SPY 1-min bars (timestamp, OHLCV)
        prev_day_bars: Previous day's 1-min bars for warmup (optional)

    Returns:
        Dict mapping (hour, minute) ET tuple → MACD histogram float value
    """
    import pandas as pd
    import pytz
    from datetime import timezone
    from trading.indicators import macd_histogram

    if spy_1min_bars is None or spy_1min_bars.empty:
        return {}

    closes = spy_1min_bars['close'].copy().reset_index(drop=True)
    warmup_len = 0

    if prev_day_bars is not None and not prev_day_bars.empty:
        warmup = prev_day_bars['close'].tail(60).reset_index(drop=True)
        closes = pd.concat([warmup, closes], ignore_index=True)
        warmup_len = len(warmup)

    if len(closes) < 35:
        return {}

    hist = macd_histogram(closes)

    ET = pytz.timezone('US/Eastern')
    result = {}
    for i in range(len(spy_1min_bars)):
        ts = spy_1min_bars.iloc[i]['timestamp']
        if hasattr(ts, 'astimezone'):
            et_time = ts.astimezone(ET)
        elif hasattr(ts, 'replace'):
            et_time = ts.replace(tzinfo=timezone.utc).astimezone(ET)
        else:
            continue
        key = (et_time.hour, et_time.minute)
        result[key] = float(hist.iloc[warmup_len + i])

    return result


class SpyMacdCutoff:
    """
    Blocks new trade entries when SPY 1-min MACD histogram is positive
    AND time is after cutoff_time ET.

    When SPY is in a short-term uptrend (MACD > 0), small-cap momentum
    stocks lose their edge in the afternoon — 30.4% WR, -$36K over 15 months.
    Removing these trades improves PnL by +13.5% (p=0.0026).

    The check is dynamic: if SPY MACD flips negative after cutoff, trading
    resumes. Only blocks NEW setup detection — pending buy-stops still fill.
    """

    def __init__(self, enabled: bool = True, cutoff_time: tuple = (11, 30)):
        """
        Initialize SPY MACD cutoff filter.

        Args:
            enabled: Whether the filter is active
            cutoff_time: (hour, minute) ET tuple — block entries after this time
        """
        self.enabled = enabled
        self.cutoff_time = cutoff_time
        self._macd_by_time: Dict[tuple, float] = {}

    def load_spy_macd(self, macd_by_time: Dict[tuple, float]) -> None:
        """
        Load pre-computed SPY MACD histogram indexed by (hour, minute) ET.

        Called once per trading day (by batch_backtest or trading_engine).

        Args:
            macd_by_time: Dict mapping (hour, minute) → MACD histogram value
        """
        self._macd_by_time = macd_by_time

    def is_blocked(self, bar_time_et: tuple) -> bool:
        """
        Check if new entries should be blocked at this bar time.

        Returns True (blocked) when ALL conditions are met:
        1. Filter is enabled
        2. bar_time_et >= cutoff_time
        3. SPY MACD histogram at this time > 0

        Args:
            bar_time_et: (hour, minute) tuple in ET

        Returns:
            True if entries should be blocked, False if allowed
        """
        if not self.enabled:
            return False
        if bar_time_et < self.cutoff_time:
            return False
        macd_val = self._macd_by_time.get(bar_time_et)
        if macd_val is None:
            return False  # No data → don't block (safe default)
        return macd_val > 0
