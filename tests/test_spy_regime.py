"""Unit tests for trading/spy_regime.py — shared SPY-regime helper.

Covers:
    - compute_spy_3d_range happy path with ascending bars
    - returns None on < 3 bars
    - returns None on bars with bad shape (missing high/low, non-numeric)
    - returns None on non-positive high/low
    - uses ONLY the last 3 bars when more are passed in
    - is_spy_data_stale boundary (exact threshold, edge of weekend)
    - is_spy_data_stale on None / missing data
"""
from __future__ import annotations

import logging
from datetime import date

import pytest

from trading.spy_regime import (
    DEFAULT_STALENESS_MAX_CALENDAR_DAYS,
    compute_spy_3d_range,
    is_spy_data_stale,
)


# ---------------------------------------------------------------------------
# compute_spy_3d_range
# ---------------------------------------------------------------------------


class TestComputeSpy3dRange:
    def test_three_bars_simple_average(self):
        bars = [
            {'high': 11.0, 'low': 10.0},   # 10.0%
            {'high': 22.0, 'low': 20.0},   # 10.0%
            {'high': 33.0, 'low': 30.0},   # 10.0%
        ]
        assert compute_spy_3d_range(bars) == pytest.approx(10.0)

    def test_three_bars_realistic_spy(self):
        # Apr 28/29/30 SPY values used in EAF post-mortem
        bars = [
            {'high': 712.88, 'low': 709.25},
            {'high': 712.20, 'low': 708.37},
            {'high': 719.79, 'low': 710.445},
        ]
        # 0.512 + 0.541 + 1.315 = 2.368, avg = 0.789
        assert compute_spy_3d_range(bars) == pytest.approx(0.789, abs=0.01)

    def test_uses_only_last_three_bars(self):
        # 5 bars; first 2 must be ignored
        bars = [
            {'high': 100.0, 'low': 1.0},   # huge range, should not affect
            {'high': 100.0, 'low': 1.0},
            {'high': 11.0, 'low': 10.0},   # 10%
            {'high': 22.0, 'low': 20.0},   # 10%
            {'high': 33.0, 'low': 30.0},   # 10%
        ]
        assert compute_spy_3d_range(bars) == pytest.approx(10.0)

    def test_returns_none_for_empty(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert compute_spy_3d_range([]) is None
        assert "insufficient SPY bars" in caplog.text

    def test_returns_none_for_two_bars(self, caplog):
        bars = [{'high': 10.0, 'low': 9.0}, {'high': 11.0, 'low': 10.0}]
        with caplog.at_level(logging.WARNING):
            assert compute_spy_3d_range(bars) is None
        assert "got 2" in caplog.text

    def test_returns_none_for_none_input(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert compute_spy_3d_range(None) is None  # type: ignore[arg-type]
        assert "got 0" in caplog.text

    def test_returns_none_when_low_is_zero(self, caplog):
        bars = [
            {'high': 10.0, 'low': 9.0},
            {'high': 10.0, 'low': 0.0},   # invalid
            {'high': 10.0, 'low': 9.0},
        ]
        with caplog.at_level(logging.WARNING):
            assert compute_spy_3d_range(bars) is None
        assert "non-positive" in caplog.text

    def test_returns_none_when_high_is_zero(self, caplog):
        bars = [
            {'high': 10.0, 'low': 9.0},
            {'high': 10.0, 'low': 9.0},
            {'high': 0.0, 'low': 9.0},
        ]
        with caplog.at_level(logging.WARNING):
            assert compute_spy_3d_range(bars) is None

    def test_returns_none_when_low_is_negative(self, caplog):
        bars = [
            {'high': 10.0, 'low': 9.0},
            {'high': 10.0, 'low': -1.0},
            {'high': 10.0, 'low': 9.0},
        ]
        with caplog.at_level(logging.WARNING):
            assert compute_spy_3d_range(bars) is None

    def test_returns_none_when_keys_missing(self, caplog):
        bars = [
            {'high': 10.0, 'low': 9.0},
            {'open': 1.0, 'close': 2.0},   # no high/low
            {'high': 10.0, 'low': 9.0},
        ]
        with caplog.at_level(logging.WARNING):
            # bar.get('high', 0) → 0 → non-positive guard fires; logs about 0/0
            assert compute_spy_3d_range(bars) is None

    def test_returns_none_when_bar_is_not_mapping(self, caplog):
        # tuple bars (no .get) should fail gracefully via except path
        bars = [
            {'high': 10.0, 'low': 9.0},
            ("not", "a", "dict"),
            {'high': 10.0, 'low': 9.0},
        ]
        with caplog.at_level(logging.WARNING):
            assert compute_spy_3d_range(bars) is None  # type: ignore[arg-type]

    def test_handles_string_numeric_values(self):
        # YAML/CSV loads sometimes leave strings; .get + float() handles it
        bars = [
            {'high': '11.0', 'low': '10.0'},
            {'high': '22.0', 'low': '20.0'},
            {'high': '33.0', 'low': '30.0'},
        ]
        assert compute_spy_3d_range(bars) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# is_spy_data_stale
# ---------------------------------------------------------------------------


class TestIsSpyDataStale:
    def test_none_input_is_stale(self, caplog):
        with caplog.at_level(logging.ERROR):
            assert is_spy_data_stale(None, date(2026, 5, 1)) is True
        assert "no SPY bars at all" in caplog.text

    def test_same_day_is_fresh(self):
        ref = date(2026, 5, 1)
        assert is_spy_data_stale(ref, ref) is False

    def test_one_day_old_is_fresh(self):
        assert is_spy_data_stale(date(2026, 4, 30), date(2026, 5, 1)) is False

    def test_at_threshold_is_fresh(self):
        # default threshold = 5 calendar days; age == 5 is still fresh
        assert is_spy_data_stale(date(2026, 4, 26), date(2026, 5, 1)) is False

    def test_one_day_past_threshold_is_stale(self, caplog):
        # age = 6 > 5 → stale
        with caplog.at_level(logging.ERROR):
            assert is_spy_data_stale(date(2026, 4, 25), date(2026, 5, 1)) is True
        assert "refusing to score regime" in caplog.text

    def test_eaf_post_mortem_scenario(self, caplog):
        # Prod's actual scenario on 2026-05-01: latest SPY bar = Apr 17
        latest = date(2026, 4, 17)
        ref = date(2026, 5, 1)
        with caplog.at_level(logging.ERROR):
            assert is_spy_data_stale(latest, ref) is True
        assert "is 14 calendar days before" in caplog.text

    def test_friday_to_monday_with_holiday_is_fresh(self):
        # Latest = Friday; reference = following Tuesday after Mon holiday
        # 4 calendar days < threshold 5 → fresh
        assert is_spy_data_stale(date(2026, 5, 1), date(2026, 5, 5)) is False

    def test_custom_threshold(self):
        # tighten threshold to 1 day
        assert is_spy_data_stale(
            date(2026, 4, 30), date(2026, 5, 1), max_calendar_days=1
        ) is False
        assert is_spy_data_stale(
            date(2026, 4, 29), date(2026, 5, 1), max_calendar_days=1
        ) is True

    def test_default_threshold_constant_is_5(self):
        assert DEFAULT_STALENESS_MAX_CALENDAR_DAYS == 5
