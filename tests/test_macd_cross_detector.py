"""Unit tests for trading.macd_cross_detector.

Mirrors the pure-function-test pattern established by
test_buy_stop_guard.py and test_orb_touchgo_filter.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.macd_cross_detector import (
    DEFAULT_CONFIRM_BARS,
    DEFAULT_FAST,
    DEFAULT_SIGNAL,
    DEFAULT_SLOW,
    EntrySignal,
    WaveOnset,
    compute_macd_histogram,
    count_consecutive_positive_ending_at,
    find_first_confirmed_entry,
    find_wave_onset,
)


def _bars(highs, lows=None, closes=None, opens=None, vols=None):
    """Build a minimal regular-session bars DataFrame.

    Defaults: low = high - 0.5, close = high - 0.25, open = high - 0.5,
    volume = 10000.
    """
    n = len(highs)
    return pd.DataFrame({
        'open': opens if opens is not None else [h - 0.5 for h in highs],
        'high': highs,
        'low': lows if lows is not None else [h - 0.5 for h in highs],
        'close': closes if closes is not None else [h - 0.25 for h in highs],
        'volume': vols if vols is not None else [10000] * n,
    })


# =========================================================================
# compute_macd_histogram
# =========================================================================

class TestComputeMACDHistogram:

    def test_returns_series_same_length(self):
        close = pd.Series([10.0, 10.1, 10.2, 10.3, 10.4] * 20)
        h = compute_macd_histogram(close)
        assert isinstance(h, pd.Series)
        assert len(h) == len(close)

    def test_uses_default_params(self):
        close = pd.Series(np.linspace(10, 12, 60))
        h_default = compute_macd_histogram(close)
        h_explicit = compute_macd_histogram(close, DEFAULT_FAST, DEFAULT_SLOW,
                                              DEFAULT_SIGNAL)
        pd.testing.assert_series_equal(h_default, h_explicit)

    def test_rising_trend_positive_histogram(self):
        """A steady upward trend should drive the histogram positive
        once the EMAs cross."""
        close = pd.Series(np.linspace(10, 20, 80))
        h = compute_macd_histogram(close)
        # Late bars should be positive after the EMAs cross
        assert h.iloc[-1] > 0
        assert h.iloc[-10] > 0

    def test_falling_trend_negative_histogram(self):
        close = pd.Series(np.linspace(20, 10, 80))
        h = compute_macd_histogram(close)
        assert h.iloc[-1] < 0

    def test_constant_price_zero_histogram(self):
        close = pd.Series([10.0] * 50)
        h = compute_macd_histogram(close)
        # All EMAs converge to 10.0 → histogram == 0
        assert abs(h.iloc[-1]) < 1e-9


# =========================================================================
# find_wave_onset
# =========================================================================

class TestFindWaveOnset:

    def test_finds_first_cross(self):
        # open=10.0, threshold at 10% = 11.0. Cross at index 3.
        bars = _bars(
            highs=[10.5, 10.7, 10.9, 11.2, 11.5],
            opens=[10.0] * 5,
        )
        onset = find_wave_onset(bars, intraday_pct=10.0)
        assert onset is not None
        assert onset.bar_index == 3
        assert onset.bar_close_minute == 4
        assert onset.cumulative_volume == 40000  # 4 bars × 10000

    def test_returns_none_when_no_cross(self):
        bars = _bars(highs=[10.5, 10.7, 10.9], opens=[10.0] * 3)
        onset = find_wave_onset(bars, intraday_pct=10.0)
        assert onset is None

    def test_returns_none_on_empty(self):
        bars = pd.DataFrame({'open': [], 'high': [], 'volume': []})
        assert find_wave_onset(bars, 10.0) is None

    def test_returns_none_on_zero_open(self):
        bars = _bars(highs=[11.0, 12.0], opens=[0.0, 11.0])
        assert find_wave_onset(bars, 10.0) is None

    def test_returns_none_on_negative_open(self):
        bars = _bars(highs=[11.0], opens=[-1.0])
        assert find_wave_onset(bars, 10.0) is None

    def test_first_bar_already_crosses(self):
        bars = _bars(highs=[15.0, 16.0], opens=[10.0, 10.0])
        onset = find_wave_onset(bars, 10.0)
        assert onset.bar_index == 0
        assert onset.bar_close_minute == 1

    def test_cumulative_volume_through_cross(self):
        bars = _bars(
            highs=[10.5, 11.5],
            opens=[10.0, 10.0],
            vols=[3000, 7000],
        )
        onset = find_wave_onset(bars, 10.0)
        assert onset.cumulative_volume == 10000


# =========================================================================
# count_consecutive_positive_ending_at
# =========================================================================

class TestCountConsecutivePositiveEndingAt:

    def test_all_positive(self):
        h = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        assert count_consecutive_positive_ending_at(h, 4) == 5

    def test_zero_breaks_count(self):
        # ending value is 0 → not positive → count 0
        h = pd.Series([0.1, 0.2, 0.0])
        assert count_consecutive_positive_ending_at(h, 2) == 0

    def test_break_in_middle(self):
        h = pd.Series([0.1, -0.1, 0.2, 0.3, 0.4])
        assert count_consecutive_positive_ending_at(h, 4) == 3

    def test_negative_at_end(self):
        h = pd.Series([0.1, 0.2, 0.3, -0.1])
        assert count_consecutive_positive_ending_at(h, 3) == 0

    def test_empty_series(self):
        h = pd.Series([], dtype=float)
        assert count_consecutive_positive_ending_at(h, 0) == 0

    def test_end_idx_out_of_range_clamps(self):
        h = pd.Series([0.1, 0.2, 0.3])
        # end_idx=99 → clamped to len-1=2
        assert count_consecutive_positive_ending_at(h, 99) == 3

    def test_negative_end_idx(self):
        h = pd.Series([0.1, 0.2])
        assert count_consecutive_positive_ending_at(h, -1) == 0

    def test_nan_breaks_count(self):
        h = pd.Series([0.1, np.nan, 0.2, 0.3])
        assert count_consecutive_positive_ending_at(h, 3) == 2


# =========================================================================
# find_first_confirmed_entry
# =========================================================================

class TestFindFirstConfirmedEntry:

    def test_entry_at_third_consecutive_positive(self):
        # confirm_bars=3, all positive from index 1
        h = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        bars = pd.DataFrame({'close': [10.0, 10.1, 10.2, 10.3, 10.4, 10.5]})
        entry = find_first_confirmed_entry(h, bars, start_idx=1, confirm_bars=3)
        assert entry is not None
        # pos_count: i=1→1, i=2→2, i=3→3 (entry)
        assert entry.bar_index == 3
        assert entry.entry_price == 10.3
        assert entry.entry_minute == 4
        # hist_pct = 0.3 / 10.3 * 100 ≈ 2.91
        assert entry.hist_pct == pytest.approx(0.3 / 10.3 * 100, abs=0.001)

    def test_reset_on_negative_then_recover(self):
        # First 2 positive, then a negative resets, then 3 positives confirm
        h = pd.Series([0.0, 0.1, 0.2, -0.1, 0.1, 0.2, 0.3])
        bars = pd.DataFrame({'close': [10.0] * 7})
        entry = find_first_confirmed_entry(h, bars, start_idx=0, confirm_bars=3)
        assert entry is not None
        # pos_count: 1→1, 2→2, 3→0(reset), 4→1, 5→2, 6→3 (entry)
        assert entry.bar_index == 6

    def test_returns_none_when_no_confirmation(self):
        h = pd.Series([0.1, -0.1, 0.1, -0.1, 0.1])
        bars = pd.DataFrame({'close': [10.0] * 5})
        entry = find_first_confirmed_entry(h, bars, start_idx=0, confirm_bars=3)
        assert entry is None

    def test_start_idx_respected(self):
        # confirm at index 3 regardless, but if start_idx is 5, never enter
        h = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4])
        bars = pd.DataFrame({'close': [10.0] * 5})
        entry = find_first_confirmed_entry(h, bars, start_idx=5, confirm_bars=3)
        assert entry is None

    def test_default_confirm_bars_is_three(self):
        h = pd.Series([0.0, 0.1, 0.2, 0.3])
        bars = pd.DataFrame({'close': [10.0] * 4})
        entry_explicit = find_first_confirmed_entry(h, bars, 0, 3)
        entry_default = find_first_confirmed_entry(h, bars, 0)
        assert entry_explicit == entry_default

    def test_nan_treated_as_negative(self):
        # NaN in histogram should reset pos_count
        h = pd.Series([0.0, 0.1, np.nan, 0.2, 0.3, 0.4])
        bars = pd.DataFrame({'close': [10.0] * 6})
        entry = find_first_confirmed_entry(h, bars, 0, 3)
        # pos: 1→1, 2→0(NaN), 3→1, 4→2, 5→3 (entry)
        assert entry.bar_index == 5

    def test_never_enters_on_bar_zero(self):
        # max(start_idx, 1) prevents entry at index 0 even if all positive
        h = pd.Series([0.1, 0.2, 0.3])
        bars = pd.DataFrame({'close': [10.0, 10.1, 10.2]})
        entry = find_first_confirmed_entry(h, bars, 0, 1)
        # start_idx=0 → max(0,1)=1 → first checked bar is index 1
        # pos at i=1 = 1, confirm_bars=1 → entry at 1
        assert entry.bar_index == 1

    def test_returns_first_not_subsequent(self):
        """If multiple confirmation runs exist, return the first."""
        # Two separate runs of 3+ positives separated by negatives
        h = pd.Series([0.0, 0.1, 0.2, 0.3, -0.1, 0.1, 0.2, 0.3])
        bars = pd.DataFrame({'close': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                                        10.0, 10.0]})
        entry = find_first_confirmed_entry(h, bars, 0, 3)
        # First confirmation at index 3 (1→2→3)
        assert entry.bar_index == 3

    def test_returns_none_on_empty(self):
        h = pd.Series([], dtype=float)
        bars = pd.DataFrame({'close': []})
        assert find_first_confirmed_entry(h, bars, 0, 3) is None


# =========================================================================
# Live-vs-BT semantic equivalence
# =========================================================================

class TestLiveBTEquivalence:
    """The live engine uses `count_consecutive_positive_ending_at(hist, last_idx)`
    each bar event; the BT walks bars forward incrementing pos_count.

    For a fully-formed bar series, BOTH should agree on whether the LATEST
    bar is a confirmed entry. This test pins that contract.
    """

    def test_live_count_at_end_matches_bt_forward_walk(self):
        # Hand-built histogram with a clear confirmation at the end
        h = pd.Series([-0.1, 0.1, -0.1, 0.1, 0.2, 0.3])
        bars = pd.DataFrame({'close': [10.0] * 6})

        # Live's check at the latest bar
        live_pos = count_consecutive_positive_ending_at(h, len(h) - 1)
        live_confirmed = live_pos >= 3

        # BT's forward walk — would it have entered at the latest bar?
        bt_entry = find_first_confirmed_entry(h, bars, 0, 3)
        bt_confirmed_at_end = (bt_entry is not None
                                and bt_entry.bar_index == len(h) - 1)

        assert live_confirmed == bt_confirmed_at_end
        assert live_pos == 3
        assert bt_entry.bar_index == 5
