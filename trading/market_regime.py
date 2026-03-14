"""
Market regime filter — blocks trading when SPY trailing return is bearish.

Uses SPY's 5-day return (T-1 close vs T-6 close) to detect hostile market
conditions. When the return falls below a configurable threshold (default -2%),
all trading is skipped for that day.

Shared by backtest and production. Stateless after loading bars.
"""

import logging
from datetime import date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MarketRegimeFilter:
    """
    Filters trading days based on SPY trailing 5-day return.

    Lookahead prevention: is_regime_ok(T) uses only closes strictly before T.
    For live trading at 9:30 AM on day T, the latest close is T-1 (yesterday).
    The backtest uses identical logic — no future data.
    """

    def __init__(self, enabled: bool = True, spy_5d_return_min: float = -2.0):
        """
        Initialize MarketRegimeFilter.

        Args:
            enabled: Whether the filter is active. When False, all days pass.
            spy_5d_return_min: Minimum SPY 5-day return percentage to allow
                trading. E.g., -2.0 means skip when SPY is down >2% over 5 days.
        """
        self.enabled = enabled
        self.spy_5d_return_min = spy_5d_return_min
        self._bars_by_date: Dict[date, float] = {}  # date -> close
        self._sorted_dates: List[date] = []

    def load_spy_bars(self, spy_bars: List[Dict]) -> None:
        """
        Load SPY daily bars.

        Args:
            spy_bars: List of bar dicts with 'date' (date obj) and 'close' (float).
        """
        self._bars_by_date.clear()
        for bar in spy_bars:
            bar_date = bar['date']
            if isinstance(bar_date, str):
                bar_date = date.fromisoformat(bar_date)
            self._bars_by_date[bar_date] = float(bar['close'])

        self._sorted_dates = sorted(self._bars_by_date.keys())
        logger.info(
            f"MarketRegimeFilter loaded {len(self._sorted_dates)} SPY bars "
            f"({self._sorted_dates[0]} to {self._sorted_dates[-1]})"
            if self._sorted_dates else
            "MarketRegimeFilter loaded 0 SPY bars"
        )

    def get_spy_5d_return(self, trade_date: date) -> Optional[float]:
        """
        SPY 5-day return as of T-1 close (no lookahead).

        Finds the 6 most recent closes BEFORE trade_date.
        Returns (close[T-1] / close[T-6] - 1) * 100 as percentage.
        Returns None if fewer than 6 prior bars available.

        Args:
            trade_date: The date we want to trade on.

        Returns:
            Percentage return (e.g., -2.5 means down 2.5%), or None.
        """
        # Get all dates strictly before trade_date
        prior_dates = [d for d in self._sorted_dates if d < trade_date]
        if len(prior_dates) < 6:
            return None

        # Most recent 6 closes before trade_date
        recent_6 = prior_dates[-6:]
        close_t1 = self._bars_by_date[recent_6[-1]]   # T-1
        close_t6 = self._bars_by_date[recent_6[0]]     # T-6

        if close_t6 == 0:
            return None

        return (close_t1 / close_t6 - 1) * 100

    def is_regime_ok(self, trade_date: date) -> bool:
        """
        Check if trading is allowed on trade_date.

        Returns True (trading allowed) when:
        - Filter is disabled
        - Insufficient data (safe default — better to trade than miss)
        - SPY 5-day return >= threshold

        Args:
            trade_date: The date to check.

        Returns:
            True if trading is allowed, False if blocked.
        """
        if not self.enabled:
            return True

        ret = self.get_spy_5d_return(trade_date)
        if ret is None:
            return True

        return ret >= self.spy_5d_return_min
