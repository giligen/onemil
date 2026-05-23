"""MACD wave signal detector — shared between BT and live trading engine.

Pure functions over bar DataFrames. No I/O, no engine state, no logging.
Mirrors the established parity-by-construction pattern used by
trading/buy_stop_guard.py, trading/orb_touchgo_filter.py, and
trading/two_tier_filter.py.

The MACD wave strategy has three computational stages:

  1. compute_macd_histogram(close, fast, slow, signal) → histogram series
     EMAs of the closing price; the histogram is `macd_line - signal_line`.

  2. find_wave_onset(bars, intraday_pct) → WaveOnset | None
     First bar whose high crosses `bars[0].open * (1 + intraday_pct/100)`.
     Marks the start of the "wave"; cum-volume through that bar is captured.

  3. find_first_confirmed_entry(histogram, start_idx, confirm_bars) → int | None
     Walk forward from start_idx; return the first index where the last
     `confirm_bars` histogram values are all > 0. This is the bar at which
     the BT enters and the live engine's pos_count >= confirm_bars check
     fires.

Both consumers (macd_wave_backtest.py::generate_signals,
trading/macd_wave_engine.py::check_entries) must import from THIS module
so they cannot drift. Parity is enforced by source-code inspection in
tests/test_macd_cross_detector_parity.py (the buy_stop_guard pattern).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import pandas as pd


# Default MACD parameters — match macd_wave.yaml validated values.
DEFAULT_FAST = 12
DEFAULT_SLOW = 26
DEFAULT_SIGNAL = 9
DEFAULT_CONFIRM_BARS = 3


@dataclass(frozen=True)
class WaveOnset:
    """The +N% intraday-threshold crossing — when the wave starts.

    bar_index:           0-indexed position within the regular-session bars
                         passed to find_wave_onset.
    bar_close_minute:    bar_index + 1 (1-indexed; equals minutes-after-09:30
                         when bars[0] is the 09:30 bar).
    cumulative_volume:   total volume through and including the cross bar.
    """
    bar_index: int
    bar_close_minute: int
    cumulative_volume: int


@dataclass(frozen=True)
class EntrySignal:
    """A `confirm_bars`-bar-confirmed MACD entry signal.

    bar_index:     0-indexed position of the confirming bar.
    entry_minute:  bar_index + 1 (minutes-after-09:30 when bars are
                   regular-session-aligned).
    entry_price:   bar.close at bar_index.
    hist_pct:      histogram[bar_index] / entry_price * 100.
    """
    bar_index: int
    entry_minute: int
    entry_price: float
    hist_pct: float


def compute_macd_histogram(
    close: pd.Series,
    fast_period: int = DEFAULT_FAST,
    slow_period: int = DEFAULT_SLOW,
    signal_period: int = DEFAULT_SIGNAL,
) -> pd.Series:
    """Compute the MACD histogram (macd_line - signal_line) from closing prices.

    Uses exponential-weighted moving averages with `adjust=False` to match
    classic MACD. Pure function — no side effects.

    Args:
        close: closing prices indexed in chronological order.
        fast_period / slow_period / signal_period: EMA spans.

    Returns:
        pd.Series of the same index as `close`, holding macd_line - signal_line.
    """
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line - signal_line


def find_wave_onset(
    bars: pd.DataFrame,
    intraday_pct: float,
) -> Optional[WaveOnset]:
    """Find the first bar whose high crosses the +intraday_pct threshold.

    Threshold = bars.iloc[0]['open'] * (1 + intraday_pct/100). `bars` must be
    regular-session bars in chronological order with 'open', 'high', 'volume'
    columns; bars[0] is the session-anchor (09:30 bar) so the open is the
    day's reference price.

    Returns None on empty bars, non-positive open, or no cross.
    """
    if bars is None or len(bars) == 0:
        return None
    try:
        op = float(bars.iloc[0]['open'])
    except (KeyError, ValueError, TypeError):
        return None
    if op <= 0:
        return None
    threshold = op * (1.0 + intraday_pct / 100.0)
    for i in range(len(bars)):
        try:
            bar_high = float(bars.iloc[i]['high'])
        except (TypeError, ValueError):
            continue
        if bar_high >= threshold:
            cum_vol = int(bars.iloc[:i + 1]['volume'].sum())
            return WaveOnset(
                bar_index=i,
                bar_close_minute=i + 1,
                cumulative_volume=cum_vol,
            )
    return None


def count_consecutive_positive_ending_at(
    histogram: pd.Series,
    end_idx: int,
) -> int:
    """How many consecutive positive histogram values END at end_idx.

    Returns 0 if histogram[end_idx] <= 0 or NaN. Walks backward from
    end_idx; stops at the first non-positive value.

    Used by the live engine to answer "is the latest bar confirmed?" —
    it tracks pos_count incrementally from this value's evolution.

    Args:
        histogram: MACD histogram series (output of compute_macd_histogram).
        end_idx:   0-indexed position to count back from.

    Returns:
        int >= 0.
    """
    if histogram is None or len(histogram) == 0 or end_idx < 0:
        return 0
    if end_idx >= len(histogram):
        end_idx = len(histogram) - 1
    count = 0
    for i in range(end_idx, -1, -1):
        v = histogram.iloc[i]
        if pd.notna(v) and v > 0:
            count += 1
        else:
            break
    return count


def find_first_confirmed_entry(
    histogram: pd.Series,
    bars: pd.DataFrame,
    start_idx: int = 0,
    confirm_bars: int = DEFAULT_CONFIRM_BARS,
) -> Optional[EntrySignal]:
    """Find the first bar at-or-after start_idx where the last `confirm_bars`
    histogram values are all > 0 — i.e., the BT's confirmed-entry bar.

    Walks bars forward from `max(start_idx, 1)` (we never enter on the very
    first bar of the day — same convention as macd_wave_backtest.py:469).
    Tracks pos_count incrementally; resets to 0 on any non-positive bar.

    Args:
        histogram: MACD histogram series aligned with `bars`.
        bars:      DataFrame with at least a 'close' column.
        start_idx: earliest bar index to consider (typically the wave-onset
                   bar index from find_wave_onset).
        confirm_bars: number of consecutive positive histogram bars required.

    Returns:
        EntrySignal at the FIRST confirming bar, or None if none in range.
    """
    if histogram is None or bars is None or len(bars) == 0:
        return None
    if confirm_bars <= 0:
        confirm_bars = 1
    pos_count = 0
    for i in range(max(start_idx, 1), len(bars)):
        if i >= len(histogram):
            break
        h = histogram.iloc[i]
        if pd.notna(h) and h > 0:
            pos_count += 1
        else:
            pos_count = 0
        if pos_count >= confirm_bars:
            try:
                raw_price = float(bars.iloc[i]['close'])
            except (KeyError, TypeError, ValueError):
                continue
            hist_pct = (h / raw_price * 100.0) if raw_price > 0 else 0.0
            return EntrySignal(
                bar_index=i,
                entry_minute=i + 1,
                entry_price=raw_price,
                hist_pct=float(hist_pct),
            )
    return None
