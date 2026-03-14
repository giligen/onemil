"""
Technical indicators for pattern detection and trade filtering.

Provides pure functions for calculating standard indicators on price series.
"""

import pandas as pd


def macd_histogram(
    closes: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    """
    Calculate MACD histogram (MACD line - signal line).

    Positive histogram = short-term momentum stronger than long-term.
    Negative histogram = momentum fading or bearish.

    Args:
        closes: Series of close prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal EMA period (default 9)

    Returns:
        Series of MACD histogram values (same index as closes)
    """
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line
