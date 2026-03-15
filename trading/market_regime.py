"""
Market regime filter — blocks trading when SPY is volatile AND in a downtrend.

Uses a composite signal: SPY 5-day average daily range (volatility) AND
SPY closing below its 50-day SMA (downtrend). When BOTH conditions are true,
the market is hostile to momentum day trading and all trades are skipped.

Also exposes max_trades_per_day for callers (TradingEngine, batch backtest).

Shared by backtest and production. Stateless after loading bars.
"""

import logging
from datetime import date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MarketRegimeFilter:
    """
    Filters trading days based on SPY volatility + trend regime.

    Blocks trading when BOTH:
    - 5-day average daily range > vol_threshold (high volatility)
    - T-1 close < SMA(sma_period) (downtrend)

    Lookahead prevention: all indicators use data strictly BEFORE trade_date (T).
    For live trading at 9:30 AM on day T, T-1's close is yesterday's close — fully known.
    """

    def __init__(
        self,
        enabled: bool = True,
        vol_threshold: float = 1.5,
        sma_period: int = 50,
        max_trades_per_day: int = 5,
    ):
        """
        Initialize MarketRegimeFilter.

        Args:
            enabled: Whether the filter is active. When False, all days pass.
            vol_threshold: 5-day avg daily range % threshold. When exceeded
                AND below SMA, trading is blocked.
            sma_period: SMA period for trend detection (default 50).
            max_trades_per_day: Maximum trades allowed per day (used by callers).
        """
        self.enabled = enabled
        self.vol_threshold = vol_threshold
        self.sma_period = sma_period
        self.max_trades_per_day = max_trades_per_day
        self._bars_by_date: Dict[date, Dict[str, float]] = {}  # date -> {open, high, low, close}
        self._sorted_dates: List[date] = []

    def load_spy_bars(self, spy_bars: List[Dict]) -> None:
        """
        Load SPY daily bars (OHLC).

        Args:
            spy_bars: List of bar dicts with 'date' (date obj or str),
                'open', 'high', 'low', 'close' (floats).
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

    def is_regime_ok(self, trade_date: date) -> bool:
        """
        Check if trading is allowed on trade_date.

        Blocks when BOTH conditions are true:
        - vol_5d > vol_threshold (high volatility)
        - T-1 close < SMA(sma_period) (downtrend)

        Returns True (trading allowed) when:
        - Filter is disabled
        - Insufficient data (safe default — better to trade than miss)
        - Either volatility is low OR SPY is above SMA

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

        below = self.is_below_sma(trade_date)
        if below is None:
            return True

        # Block only when BOTH: high vol AND downtrend
        if vol > self.vol_threshold and below:
            return False

        return True

    def get_regime_info(self, trade_date: date) -> dict:
        """
        Return structured regime info for logging.

        Args:
            trade_date: The date to check.

        Returns:
            Dict with vol_5d, sma, is_below_sma, is_ok keys.
        """
        vol = self.get_spy_vol_5d(trade_date)
        sma = self.get_spy_sma(trade_date)
        below = self.is_below_sma(trade_date)
        ok = self.is_regime_ok(trade_date)

        return {
            'vol_5d': vol,
            'sma': sma,
            'is_below_sma': below,
            'is_ok': ok,
        }
