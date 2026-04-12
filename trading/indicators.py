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


def rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    Measures momentum on a 0-100 scale.
    >70 = overbought, <30 = oversold.

    Args:
        closes: Series of close prices
        period: Lookback period (default 14)

    Returns:
        Series of RSI values (0-100)
    """
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def obv(closes: pd.Series, volumes: pd.Series) -> pd.Series:
    """
    On-Balance Volume.

    Cumulative volume: adds volume on up-closes, subtracts on down-closes.
    Rising OBV = buying pressure. Falling OBV = selling pressure.

    Args:
        closes: Series of close prices
        volumes: Series of volumes (same index as closes)

    Returns:
        Series of OBV values
    """
    direction = closes.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (volumes * direction).cumsum()


def ema(closes: pd.Series, period: int = 21) -> pd.Series:
    """
    Exponential Moving Average.

    Args:
        closes: Series of close prices
        period: EMA period (default 21)

    Returns:
        Series of EMA values
    """
    return closes.ewm(span=period, adjust=False).mean()


def stochastic(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic oscillator %K and %D (both 0-100).

    %K = 100 * (close - min(low, k_period)) / (max(high, k_period) - min(low, k_period))
    %D = SMA(%K, d_period)

    >80 = overbought, <20 = oversold. %K crossing above %D = bullish trigger.

    Args:
        highs: Series of bar highs
        lows: Series of bar lows
        closes: Series of bar closes
        k_period: Lookback for %K (default 14)
        d_period: Smoothing for %D (default 3)

    Returns:
        (percent_k, percent_d) tuple of Series (same index as input)
    """
    low_min = lows.rolling(k_period).min()
    high_max = highs.rolling(k_period).max()
    denom = (high_max - low_min).replace(0, 1e-9)
    percent_k = 100.0 * (closes - low_min) / denom
    percent_d = percent_k.rolling(d_period).mean()
    return percent_k, percent_d


def vwap(highs: pd.Series, lows: pd.Series, closes: pd.Series, volumes: pd.Series) -> pd.Series:
    """
    Volume-Weighted Average Price (cumulative from start of series).

    VWAP[i] = sum(typical_price[0..i] * volume[0..i]) / sum(volume[0..i])
    where typical_price = (high + low + close) / 3

    Above VWAP = buyers in control. Below VWAP = sellers in control.

    Args:
        highs: Series of bar highs
        lows: Series of bar lows
        closes: Series of bar closes
        volumes: Series of bar volumes

    Returns:
        Series of VWAP values (same index as input)
    """
    typical = (highs + lows + closes) / 3.0
    cum_pv = (typical * volumes).cumsum()
    cum_v = volumes.cumsum().replace(0, 1e-9)  # avoid div by zero
    return cum_pv / cum_v
