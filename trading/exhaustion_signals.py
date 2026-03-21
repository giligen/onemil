"""
Shared exhaustion signal detectors for both backtest and production.

Exhaustion signals detect when a strong move is running out of steam —
selling partial into strength (while buyers are still aggressive) reduces
slippage vs. waiting for the trailing stop to trigger on the reversal.

Signals:
- climax_candle: Body AND volume both > 2× average of prior 5 bars
- shooting_star: Long upper wick (>2× body) with close near the low
- volume_divergence: Volume declining over 3 bars while price makes higher highs
- shrinking_bodies: Current body < 50% of body 3 bars ago, price still near highs
"""

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def check_exhaustion(bars: pd.DataFrame, idx: int,
                     signals: Dict[str, bool]) -> bool:
    """
    Check if any enabled exhaustion signal fires at bar idx.

    Args:
        bars: DataFrame with OHLCV columns
        idx: Bar index to check
        signals: Dict of signal_name -> enabled flag

    Returns:
        True if any enabled signal fires
    """
    if signals.get('volume_divergence') and sig_volume_divergence(bars, idx):
        return True
    if signals.get('climax_candle') and sig_climax_candle(bars, idx):
        return True
    if signals.get('shrinking_bodies') and sig_shrinking_bodies(bars, idx):
        return True
    if signals.get('shooting_star') and sig_shooting_star(bars, idx):
        return True
    return False


def sig_volume_divergence(bars: pd.DataFrame, idx: int,
                          lookback: int = 3) -> bool:
    """
    Volume declining over lookback bars while price makes higher highs.

    Args:
        bars: DataFrame with OHLCV columns
        idx: Bar index to check
        lookback: Number of consecutive declining-volume bars required

    Returns:
        True if volume divergence detected
    """
    if idx < lookback:
        return False
    for j in range(1, lookback + 1):
        curr = idx - lookback + j
        prev = curr - 1
        if prev < 0:
            return False
        if bars.iloc[curr]['volume'] >= bars.iloc[prev]['volume']:
            return False
        if bars.iloc[curr]['high'] < bars.iloc[prev]['high'] - 0.01:
            return False
    return True


def sig_climax_candle(bars: pd.DataFrame, idx: int,
                      lookback: int = 5, body_mult: float = 2.0,
                      vol_mult: float = 2.0) -> bool:
    """
    Body AND volume both > mult x average of previous lookback bars.

    A climax candle signals exhaustion — the last burst of buying
    before sellers take over.

    Args:
        bars: DataFrame with OHLCV columns
        idx: Bar index to check
        lookback: Number of prior bars for average calculation
        body_mult: Current body must exceed avg × this multiplier
        vol_mult: Current volume must exceed avg × this multiplier

    Returns:
        True if climax candle detected
    """
    if idx < lookback:
        return False
    curr_body = abs(bars.iloc[idx]['close'] - bars.iloc[idx]['open'])
    curr_vol = bars.iloc[idx]['volume']
    avg_body = sum(
        abs(bars.iloc[idx - j]['close'] - bars.iloc[idx - j]['open'])
        for j in range(1, lookback + 1)
    ) / lookback
    avg_vol = sum(
        bars.iloc[idx - j]['volume'] for j in range(1, lookback + 1)
    ) / lookback
    if avg_body <= 0 or avg_vol <= 0:
        return False
    return curr_body >= avg_body * body_mult and curr_vol >= avg_vol * vol_mult


def sig_shrinking_bodies(bars: pd.DataFrame, idx: int,
                         lookback: int = 3,
                         shrink_ratio: float = 0.5) -> bool:
    """
    Current body < shrink_ratio x body from lookback bars ago, price still near highs.

    Shrinking bodies with price holding = buyers exhausted but not yet selling.

    Args:
        bars: DataFrame with OHLCV columns
        idx: Bar index to check
        lookback: Compare current body vs this many bars ago
        shrink_ratio: Current body must be below this fraction of prior body

    Returns:
        True if shrinking bodies detected
    """
    if idx < lookback:
        return False
    curr_body = abs(bars.iloc[idx]['close'] - bars.iloc[idx]['open'])
    prev_body = abs(
        bars.iloc[idx - lookback]['close'] - bars.iloc[idx - lookback]['open']
    )
    if prev_body <= 0:
        return False
    if curr_body >= prev_body * shrink_ratio:
        return False
    if bars.iloc[idx]['close'] < bars.iloc[idx - lookback]['close']:
        return False
    return True


def sig_shooting_star(bars: pd.DataFrame, idx: int,
                      wick_ratio: float = 2.0) -> bool:
    """
    Long upper wick (> wick_ratio x body) with close near the low.

    Classic reversal candle — buyers pushed high but sellers took over.

    Args:
        bars: DataFrame with OHLCV columns
        idx: Bar index to check
        wick_ratio: Upper wick must be at least this × body size

    Returns:
        True if shooting star detected
    """
    bar = bars.iloc[idx]
    body = abs(bar['close'] - bar['open'])
    upper_wick = bar['high'] - max(bar['open'], bar['close'])
    if body <= 0.001:
        return False
    if upper_wick < body * wick_ratio:
        return False
    bar_range = bar['high'] - bar['low']
    if bar_range <= 0:
        return False
    close_position = (bar['close'] - bar['low']) / bar_range
    return close_position <= 0.4
