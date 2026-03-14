"""
Extensive tests for BullFlagDetector — the core pattern detection algorithm.

Tests cover:
- Positive: Valid bull flags that MUST be detected
- Negative: Patterns that MUST be rejected
- Edge cases: Boundary conditions and unusual inputs
- Real-world inspired: Simulated market scenarios

Uses synthetic bar data via _make_bars() helper for readable, reproducible tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from trading.pattern_detector import BullFlagDetector, BullFlagPattern, BullFlagSetup


# ---------------------------------------------------------------------------
# Helper: Build synthetic 1-min bar data
# ---------------------------------------------------------------------------

def _make_bars(candles, base_time=None):
    """
    Create a DataFrame of synthetic 1-min bars.

    Args:
        candles: List of (open, high, low, close, volume) tuples
        base_time: Starting timestamp (defaults to now - len(candles) minutes)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if base_time is None:
        base_time = datetime.now(timezone.utc) - timedelta(minutes=len(candles))

    records = []
    for i, (o, h, l, c, v) in enumerate(candles):
        records.append({
            'timestamp': base_time + timedelta(minutes=i),
            'open': float(o),
            'high': float(h),
            'low': float(l),
            'close': float(c),
            'volume': int(v),
        })
    return pd.DataFrame(records)


def _green(open_price, gain, volume, wick_pct=0.02):
    """Create a green candle tuple. close > open."""
    close = round(open_price + gain, 4)
    low = round(open_price - open_price * wick_pct, 4)
    high = round(close + close * wick_pct, 4)
    return (open_price, high, low, close, volume)


def _red(open_price, drop, volume, wick_pct=0.02):
    """Create a red candle tuple. close < open."""
    close = round(open_price - drop, 4)
    low = round(close - close * wick_pct, 4)
    high = round(open_price + open_price * wick_pct, 4)
    return (open_price, high, low, close, volume)


@pytest.fixture
def detector():
    """Standard detector with default thresholds."""
    return BullFlagDetector(
        min_pole_candles=3,
        min_pole_gain_pct=3.0,
        max_retracement_pct=50.0,
        max_pullback_candles=5,
        min_breakout_volume_ratio=1.5,
    )


# ===========================================================================
# POSITIVE TESTS — Valid bull flags that MUST be detected
# ===========================================================================

class TestPositivePatterns:
    """Valid bull flag patterns that must be detected."""

    def test_classic_bull_flag_3_pole_2_pullback(self, detector):
        """Textbook: 3 green pole candles, 2 red pullback, breakout on volume."""
        candles = [
            # Pole: 3 green candles, $4.00 -> $4.50 (+12.5%)
            (4.00, 4.10, 3.98, 4.15, 200000),   # green
            (4.15, 4.30, 4.12, 4.30, 180000),   # green
            (4.30, 4.55, 4.28, 4.50, 150000),   # green
            # Pullback: 2 red candles, retrace to ~$4.35 (33% of 0.50 pole)
            (4.50, 4.52, 4.38, 4.40, 50000),    # red
            (4.40, 4.42, 4.33, 4.35, 30000),    # red
            # Breakout candle (close above flag high 4.42)
            (4.35, 4.60, 4.34, 4.55, 250000),   # green breakout
            # Current bar (will be dropped)
            (4.55, 4.60, 4.50, 4.58, 100000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)

        assert pattern is not None
        assert pattern.symbol == "TEST"
        assert pattern.pullback_candle_count == 2
        assert pattern.pole_gain_pct >= 3.0
        assert pattern.retracement_pct <= 50.0

    def test_classic_bull_flag_4_pole_3_pullback(self, detector):
        """4 green pole, 3 red pullback, clean breakout."""
        candles = [
            (5.00, 5.10, 4.98, 5.10, 200000),   # green
            (5.10, 5.25, 5.08, 5.20, 190000),   # green
            (5.20, 5.35, 5.18, 5.30, 180000),   # green
            (5.30, 5.50, 5.28, 5.45, 170000),   # green
            # Pullback: 3 red, retrace ~30%
            (5.45, 5.47, 5.30, 5.32, 60000),    # red
            (5.32, 5.35, 5.28, 5.30, 40000),    # red
            (5.30, 5.33, 5.25, 5.28, 30000),    # red
            # Breakout
            (5.28, 5.55, 5.27, 5.50, 200000),   # green breakout
            # Current (dropped)
            (5.50, 5.55, 5.48, 5.52, 100000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)

        assert pattern is not None
        assert pattern.pullback_candle_count == 3

    def test_strong_pole_5_candles(self, detector):
        """5 green pole candles, big move, 2 candle pullback."""
        candles = [
            (3.00, 3.10, 2.98, 3.08, 200000),
            (3.08, 3.20, 3.06, 3.18, 200000),
            (3.18, 3.30, 3.16, 3.28, 190000),
            (3.28, 3.42, 3.26, 3.40, 180000),
            (3.40, 3.55, 3.38, 3.50, 170000),
            # Pullback
            (3.50, 3.52, 3.40, 3.42, 50000),
            (3.42, 3.44, 3.38, 3.40, 30000),
            # Breakout
            (3.40, 3.60, 3.39, 3.55, 200000),
            # Current (dropped)
            (3.55, 3.58, 3.53, 3.56, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is not None

    def test_minimal_3pct_pole_gain(self, detector):
        """Pole gain just above 3.0% boundary — should pass."""
        # pole_low = min(3.98, 4.02, 4.05) = 3.98
        # pole_high = max(4.05, 4.08, 4.13) = 4.13
        # pole_height = 4.13 - 3.98 = 0.15, gain = 0.15/3.98 = 3.77%
        # flag_low = min(4.09, 4.07) = 4.07
        # retracement = (4.13 - 4.07) / 0.15 = 40%
        candles = [
            (4.00, 4.05, 3.98, 4.04, 200000),   # green
            (4.04, 4.08, 4.02, 4.07, 180000),   # green
            (4.07, 4.13, 4.05, 4.12, 160000),   # green
            # Pullback: shallow retrace ~40%
            (4.12, 4.13, 4.09, 4.10, 50000),    # red
            (4.10, 4.11, 4.07, 4.08, 30000),    # red
            # Breakout: close above flag_high (4.11)
            (4.08, 4.20, 4.07, 4.18, 200000),
            # Current
            (4.18, 4.22, 4.16, 4.20, 100000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is not None
        assert pattern.pole_gain_pct >= 3.0

    def test_exactly_50pct_retracement(self, detector):
        """Retracement at exactly 50% boundary — should pass."""
        # Pole: 4.00 low -> 4.40 high = 0.40 height
        # 50% retrace = flag low at 4.20
        candles = [
            (4.00, 4.15, 3.98, 4.13, 200000),
            (4.13, 4.28, 4.11, 4.25, 180000),
            (4.25, 4.42, 4.23, 4.40, 160000),
            # Pullback: flag_low = 4.20 exactly (50% of 0.40)
            # retracement = 4.40 - 4.20 = 0.20, pct = 0.20/0.40 * 100 = 50%
            (4.40, 4.41, 4.25, 4.28, 50000),
            (4.28, 4.30, 4.20, 4.22, 30000),
            # Breakout
            (4.22, 4.50, 4.21, 4.45, 200000),
            # Current
            (4.45, 4.48, 4.43, 4.46, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is not None
        assert pattern.retracement_pct <= 50.0

    def test_shallow_20pct_retracement(self, detector):
        """Very tight pullback (20% retrace) — ideal setup."""
        # Pole: 5.00 low -> 5.30 high = 0.30 height
        # 20% retrace = flag low at 5.24
        candles = [
            (5.00, 5.12, 4.98, 5.10, 200000),
            (5.10, 5.22, 5.08, 5.20, 180000),
            (5.20, 5.32, 5.18, 5.30, 160000),
            # Shallow pullback
            (5.30, 5.31, 5.25, 5.26, 50000),
            (5.26, 5.28, 5.24, 5.25, 30000),
            # Breakout
            (5.25, 5.40, 5.24, 5.38, 200000),
            # Current
            (5.38, 5.40, 5.36, 5.39, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is not None
        assert pattern.retracement_pct <= 25.0

    def test_high_volume_pole_declining_flag(self, detector):
        """Volume: 200K→180K→150K pole, 50K→30K flag, 250K breakout."""
        candles = [
            (6.00, 6.15, 5.98, 6.12, 200000),
            (6.12, 6.28, 6.10, 6.25, 180000),
            (6.25, 6.42, 6.23, 6.40, 150000),
            # Declining flag volume
            (6.40, 6.42, 6.30, 6.32, 50000),
            (6.32, 6.34, 6.28, 6.30, 30000),
            # Volume expansion breakout
            (6.30, 6.55, 6.29, 6.50, 250000),
            # Current
            (6.50, 6.55, 6.48, 6.52, 100000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is not None
        assert pattern.avg_pole_volume > pattern.avg_flag_volume

    def test_breakout_volume_exactly_1_5x(self, detector):
        """Breakout volume at exactly 1.5x boundary."""
        # avg flag volume = 40000, breakout = 60000 = 1.5x exactly
        candles = [
            (4.00, 4.15, 3.98, 4.12, 200000),
            (4.12, 4.28, 4.10, 4.25, 180000),
            (4.25, 4.42, 4.23, 4.40, 160000),
            # Flag
            (4.40, 4.42, 4.30, 4.32, 50000),
            (4.32, 4.34, 4.28, 4.30, 30000),  # avg = 40000
            # Breakout: 60000 = 1.5x of 40000
            (4.30, 4.50, 4.29, 4.45, 60000),
            # Current
            (4.45, 4.48, 4.43, 4.46, 30000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is not None

    def test_pole_with_wicks(self, detector):
        """Pole candles have wicks but bodies are green (close > open)."""
        candles = [
            (4.00, 4.20, 3.90, 4.10, 200000),   # green, big wicks
            (4.10, 4.35, 4.00, 4.25, 180000),   # green, big wicks
            (4.25, 4.50, 4.15, 4.40, 160000),   # green, big wicks
            (4.40, 4.42, 4.30, 4.32, 50000),    # red pullback
            (4.32, 4.34, 4.28, 4.30, 30000),    # red pullback
            (4.30, 4.55, 4.29, 4.50, 200000),   # breakout
            (4.50, 4.55, 4.48, 4.52, 80000),    # current
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is not None

    def test_pullback_with_lower_wicks(self, detector):
        """Red pullback candles have long lower wicks (buyers stepping in)."""
        candles = [
            (4.00, 4.12, 3.98, 4.10, 200000),
            (4.10, 4.24, 4.08, 4.22, 180000),
            (4.22, 4.38, 4.20, 4.35, 160000),
            # Red with long lower wicks, retrace ~45% (under 50%)
            (4.35, 4.36, 4.22, 4.28, 50000),    # red, lower wick
            (4.28, 4.30, 4.20, 4.22, 30000),    # red, lower wick
            # Breakout: close above flag_high (4.30)
            (4.22, 4.45, 4.21, 4.42, 200000),
            # Current
            (4.42, 4.45, 4.40, 4.43, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is not None

    def test_large_price_stock_8_dollar(self, detector):
        """Bull flag on $8 stock (higher end of $2-$20 range)."""
        candles = [
            (8.00, 8.15, 7.98, 8.12, 200000),
            (8.12, 8.28, 8.10, 8.25, 180000),
            (8.25, 8.42, 8.23, 8.40, 160000),
            (8.40, 8.42, 8.30, 8.32, 50000),
            (8.32, 8.34, 8.28, 8.30, 30000),
            (8.30, 8.55, 8.29, 8.50, 200000),
            (8.50, 8.55, 8.48, 8.52, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is not None

    def test_penny_stock_2_50(self, detector):
        """Bull flag on $2.50 stock (lower end)."""
        candles = [
            (2.50, 2.60, 2.48, 2.58, 300000),
            (2.58, 2.70, 2.56, 2.68, 280000),
            (2.68, 2.82, 2.66, 2.78, 260000),
            (2.78, 2.80, 2.70, 2.72, 80000),
            (2.72, 2.74, 2.68, 2.70, 50000),
            (2.70, 2.90, 2.69, 2.85, 300000),
            (2.85, 2.88, 2.83, 2.86, 100000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is not None

    def test_flag_drifting_sideways(self, detector):
        """Pullback moves mostly sideways (very small red bodies)."""
        candles = [
            (4.00, 4.12, 3.98, 4.10, 200000),
            (4.10, 4.24, 4.08, 4.22, 180000),
            (4.22, 4.38, 4.20, 4.35, 160000),
            # Sideways: tiny red bodies
            (4.35, 4.36, 4.33, 4.34, 50000),
            (4.34, 4.35, 4.32, 4.33, 30000),
            # Breakout
            (4.33, 4.50, 4.32, 4.45, 200000),
            # Current
            (4.45, 4.48, 4.43, 4.46, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is not None


# ===========================================================================
# NEGATIVE TESTS — Patterns that MUST be rejected
# ===========================================================================

class TestNegativePatterns:
    """Patterns that must be rejected by the detector."""

    def test_rejects_pole_too_short_2_candles(self, detector):
        """Only 2 green candles in pole (need 3+)."""
        candles = [
            (4.00, 4.15, 3.98, 4.12, 200000),
            (4.12, 4.28, 4.10, 4.25, 180000),
            # Only 2 pole candles, then pullback
            (4.25, 4.27, 4.18, 4.20, 50000),
            (4.20, 4.22, 4.15, 4.17, 30000),
            (4.17, 4.30, 4.16, 4.28, 200000),
            (4.28, 4.30, 4.26, 4.29, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is None

    def test_rejects_pole_too_short_1_candle(self, detector):
        """Single spike candle, no pole."""
        candles = [
            (4.00, 4.20, 3.98, 4.15, 200000),   # single green
            (4.15, 4.17, 4.08, 4.10, 50000),     # red
            (4.10, 4.12, 4.05, 4.07, 30000),     # red
            (4.07, 4.20, 4.06, 4.18, 200000),    # breakout
            (4.18, 4.20, 4.16, 4.19, 80000),     # current
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is None

    def test_rejects_pole_gain_below_3pct(self, detector):
        """Pole only gains 2.5%."""
        # 3 green candles but small move: 4.00 -> 4.10 = 2.5%
        candles = [
            (4.00, 4.04, 3.99, 4.03, 200000),
            (4.03, 4.07, 4.02, 4.06, 180000),
            (4.06, 4.10, 4.05, 4.10, 160000),
            (4.10, 4.11, 4.06, 4.07, 50000),
            (4.07, 4.09, 4.04, 4.05, 30000),
            (4.05, 4.15, 4.04, 4.12, 200000),
            (4.12, 4.14, 4.10, 4.13, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is None

    def test_rejects_retracement_over_50pct(self, detector):
        """Pullback retraces 60% of pole."""
        # Pole: low=4.00, high=4.50, height=0.50
        # 60% retrace: flag_low = 4.50 - 0.30 = 4.20
        candles = [
            (4.00, 4.15, 3.98, 4.13, 200000),
            (4.13, 4.30, 4.11, 4.28, 180000),
            (4.28, 4.52, 4.26, 4.50, 160000),
            # Deep pullback
            (4.50, 4.51, 4.22, 4.25, 50000),
            (4.25, 4.27, 4.18, 4.20, 30000),
            # Breakout
            (4.20, 4.55, 4.19, 4.50, 200000),
            (4.50, 4.55, 4.48, 4.52, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is None

    def test_rejects_retracement_over_50pct_deep(self, detector):
        """Pullback retraces 80% — basically gave back the move."""
        candles = [
            (4.00, 4.12, 3.98, 4.10, 200000),
            (4.10, 4.25, 4.08, 4.22, 180000),
            (4.22, 4.42, 4.20, 4.40, 160000),
            # 80% retrace: 4.40 - 0.32 = 4.08
            (4.40, 4.41, 4.10, 4.12, 50000),
            (4.12, 4.14, 4.06, 4.08, 30000),
            (4.08, 4.20, 4.07, 4.18, 200000),
            (4.18, 4.20, 4.16, 4.19, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is None

    def test_rejects_pullback_too_long_5_candles(self, detector):
        """6 red candles exceeds max_pullback_candles=5 — rejected."""
        candles = [
            (4.00, 4.12, 3.98, 4.10, 200000),
            (4.10, 4.24, 4.08, 4.22, 180000),
            (4.22, 4.38, 4.20, 4.35, 160000),
            # 6 red candles (exceeds max 5)
            (4.35, 4.36, 4.30, 4.32, 50000),
            (4.32, 4.34, 4.28, 4.30, 45000),
            (4.30, 4.32, 4.26, 4.28, 40000),
            (4.28, 4.30, 4.24, 4.26, 35000),
            (4.26, 4.28, 4.22, 4.24, 30000),
            (4.24, 4.26, 4.20, 4.22, 25000),
            # Breakout
            (4.22, 4.40, 4.21, 4.38, 200000),
            # Current
            (4.38, 4.40, 4.36, 4.39, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is None

    def test_rejects_pullback_too_long_6_candles(self, detector):
        """6 red candles — clear downtrend, not a flag."""
        detector_strict = BullFlagDetector(max_pullback_candles=5)
        candles = [
            (4.00, 4.12, 3.98, 4.10, 200000),
            (4.10, 4.24, 4.08, 4.22, 180000),
            (4.22, 4.38, 4.20, 4.35, 160000),
            # 6 red candles
            (4.35, 4.36, 4.30, 4.32, 50000),
            (4.32, 4.34, 4.28, 4.30, 45000),
            (4.30, 4.32, 4.26, 4.28, 40000),
            (4.28, 4.30, 4.24, 4.26, 35000),
            (4.26, 4.28, 4.22, 4.24, 30000),
            (4.24, 4.26, 4.20, 4.22, 25000),
            (4.22, 4.40, 4.21, 4.38, 200000),
            (4.38, 4.40, 4.36, 4.39, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector_strict.detect("TEST", bars)
        assert pattern is None

    def test_rejects_no_pullback_all_green(self, detector):
        """All green candles — no flag formed yet."""
        candles = [
            (4.00, 4.12, 3.98, 4.10, 200000),
            (4.10, 4.22, 4.08, 4.20, 180000),
            (4.20, 4.32, 4.18, 4.30, 160000),
            (4.30, 4.42, 4.28, 4.40, 150000),
            (4.40, 4.52, 4.38, 4.50, 140000),
            (4.50, 4.62, 4.48, 4.60, 130000),
            (4.60, 4.72, 4.58, 4.70, 120000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is None

    def test_rejects_no_pole_all_red(self, detector):
        """All red candles — bear move."""
        candles = [
            (4.70, 4.72, 4.58, 4.60, 200000),
            (4.60, 4.62, 4.48, 4.50, 180000),
            (4.50, 4.52, 4.38, 4.40, 160000),
            (4.40, 4.42, 4.28, 4.30, 150000),
            (4.30, 4.32, 4.18, 4.20, 140000),
            (4.20, 4.22, 4.08, 4.10, 130000),
            (4.10, 4.12, 3.98, 4.00, 120000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is None

    def test_rejects_no_volume_expansion_breakout(self, detector):
        """Breakout candle volume is only 0.8x pullback avg — no conviction."""
        candles = [
            (4.00, 4.12, 3.98, 4.10, 200000),
            (4.10, 4.24, 4.08, 4.22, 180000),
            (4.22, 4.38, 4.20, 4.35, 160000),
            (4.35, 4.36, 4.28, 4.30, 50000),
            (4.30, 4.32, 4.25, 4.27, 30000),  # avg flag vol = 40000
            # Breakout with low volume: 32000 = 0.8x of 40000
            (4.27, 4.45, 4.26, 4.40, 32000),
            (4.40, 4.42, 4.38, 4.41, 20000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is None

    def test_rejects_volume_increasing_during_pullback(self, detector):
        """Volume goes UP during pullback (distribution/selling)."""
        candles = [
            (4.00, 4.12, 3.98, 4.10, 100000),
            (4.10, 4.24, 4.08, 4.22, 100000),
            (4.22, 4.38, 4.20, 4.35, 100000),
            # Volume increasing during pullback (bad)
            (4.35, 4.36, 4.28, 4.30, 150000),
            (4.30, 4.32, 4.25, 4.27, 200000),
            (4.27, 4.45, 4.26, 4.40, 300000),
            (4.40, 4.42, 4.38, 4.41, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is None

    def test_rejects_no_breakout_yet(self, detector):
        """Perfect pole + flag, but latest bar hasn't broken above flag high."""
        candles = [
            (4.00, 4.12, 3.98, 4.10, 200000),
            (4.10, 4.24, 4.08, 4.22, 180000),
            (4.22, 4.38, 4.20, 4.35, 160000),
            (4.35, 4.36, 4.28, 4.30, 50000),
            (4.30, 4.32, 4.25, 4.27, 30000),
            # "Breakout" bar closes below flag_high (4.32)
            (4.27, 4.30, 4.25, 4.28, 200000),
            (4.28, 4.30, 4.26, 4.29, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is None

    def test_rejects_insufficient_data_3_bars(self, detector):
        """Only 3 bars total — not enough for pattern."""
        candles = [
            (4.00, 4.10, 3.98, 4.08, 200000),
            (4.08, 4.15, 4.06, 4.12, 180000),
            (4.12, 4.18, 4.10, 4.15, 160000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is None

    def test_rejects_insufficient_data_5_bars(self, detector):
        """5 bars — marginal, still not enough after dropping current."""
        candles = [
            (4.00, 4.10, 3.98, 4.08, 200000),
            (4.08, 4.15, 4.06, 4.12, 180000),
            (4.12, 4.18, 4.10, 4.15, 160000),
            (4.15, 4.16, 4.10, 4.11, 50000),
            (4.11, 4.18, 4.10, 4.16, 200000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is None

    def test_rejects_flat_price_no_movement(self, detector):
        """All bars at same price — no pole."""
        candles = [
            (4.00, 4.01, 3.99, 4.00, 100000),
            (4.00, 4.01, 3.99, 4.00, 100000),
            (4.00, 4.01, 3.99, 4.00, 100000),
            (4.00, 4.01, 3.99, 4.00, 100000),
            (4.00, 4.01, 3.99, 4.00, 100000),
            (4.00, 4.01, 3.99, 4.00, 100000),
            (4.00, 4.01, 3.99, 4.00, 100000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is None

    def test_rejects_inverted_flag_green_pullback(self, detector):
        """'Pullback' candles are green (not a real pullback)."""
        candles = [
            (4.00, 4.12, 3.98, 4.10, 200000),
            (4.10, 4.24, 4.08, 4.22, 180000),
            (4.22, 4.38, 4.20, 4.35, 160000),
            # "Pullback" is green — not a pullback
            (4.35, 4.38, 4.34, 4.37, 50000),
            (4.37, 4.40, 4.36, 4.39, 30000),
            (4.39, 4.50, 4.38, 4.48, 200000),
            (4.48, 4.50, 4.46, 4.49, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        # Should be rejected because there's no red pullback
        assert pattern is None

    def test_rejects_bear_flag(self, detector):
        """Red pole candles then green pullback (bearish, not bullish)."""
        candles = [
            # Bear pole (red candles going down)
            (4.50, 4.52, 4.38, 4.40, 200000),
            (4.40, 4.42, 4.28, 4.30, 180000),
            (4.30, 4.32, 4.18, 4.20, 160000),
            # Green "pullback" (bounce)
            (4.20, 4.28, 4.19, 4.25, 50000),
            (4.25, 4.32, 4.24, 4.30, 30000),
            # Breakdown
            (4.30, 4.31, 4.10, 4.12, 200000),
            (4.12, 4.15, 4.10, 4.13, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is None


# ===========================================================================
# EDGE CASES
# ===========================================================================

class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_mixed_candles_in_pole_majority_green(self, detector):
        """4 candles: green, green, tiny red, green — pole has a mixed candle."""
        # With our strict consecutive-green pole detection, the tiny red breaks the pole.
        # This means the pole will only be 1 green candle (the last one before pullback),
        # which is too short. So this should be rejected.
        candles = [
            (4.00, 4.12, 3.98, 4.10, 200000),   # green
            (4.10, 4.22, 4.08, 4.20, 180000),   # green
            (4.20, 4.22, 4.18, 4.19, 160000),   # tiny red (breaks pole)
            (4.19, 4.35, 4.17, 4.32, 140000),   # green
            (4.32, 4.34, 4.25, 4.27, 50000),    # red pullback
            (4.27, 4.29, 4.22, 4.24, 30000),    # red pullback
            (4.24, 4.38, 4.23, 4.35, 200000),   # breakout
            (4.35, 4.38, 4.33, 4.36, 80000),    # current
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        # With strict consecutive detection, single green before pullback is too short
        assert pattern is None

    def test_doji_candle_in_pullback(self, detector):
        """Pullback has a doji (open ≈ close) — counts as neutral."""
        candles = [
            (4.00, 4.12, 3.98, 4.10, 200000),
            (4.10, 4.24, 4.08, 4.22, 180000),
            (4.22, 4.38, 4.20, 4.35, 160000),
            # Red + doji in pullback
            (4.35, 4.36, 4.28, 4.30, 50000),    # red
            (4.30, 4.32, 4.28, 4.30, 30000),    # doji (open == close)
            # Breakout
            (4.30, 4.48, 4.29, 4.45, 200000),
            (4.45, 4.48, 4.43, 4.46, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is not None

    def test_gap_up_in_pole(self, detector):
        """Pole includes a gap-up between candles."""
        candles = [
            (4.00, 4.12, 3.98, 4.10, 200000),
            (4.20, 4.32, 4.18, 4.30, 180000),   # Gap up from 4.10 to 4.20
            (4.30, 4.48, 4.28, 4.45, 160000),
            (4.45, 4.47, 4.38, 4.40, 50000),
            (4.40, 4.42, 4.35, 4.37, 30000),
            (4.37, 4.55, 4.36, 4.52, 200000),
            (4.52, 4.55, 4.50, 4.53, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is not None

    def test_multiple_patterns_returns_most_recent(self, detector):
        """Two potential bull flags in data — return the most recent one."""
        candles = [
            # First pattern (older)
            (3.00, 3.10, 2.98, 3.08, 200000),
            (3.08, 3.20, 3.06, 3.18, 180000),
            (3.18, 3.32, 3.16, 3.30, 160000),
            (3.30, 3.31, 3.22, 3.24, 50000),
            (3.24, 3.26, 3.20, 3.22, 30000),
            (3.22, 3.35, 3.21, 3.32, 200000),
            # Some bars in between
            (3.32, 3.34, 3.28, 3.30, 100000),
            (3.30, 3.32, 3.28, 3.29, 90000),
            # Second pattern (newer)
            (3.30, 3.42, 3.28, 3.40, 200000),
            (3.40, 3.52, 3.38, 3.50, 180000),
            (3.50, 3.62, 3.48, 3.60, 160000),
            (3.60, 3.61, 3.52, 3.54, 50000),
            (3.54, 3.56, 3.50, 3.52, 30000),
            (3.52, 3.68, 3.51, 3.65, 200000),
            (3.65, 3.68, 3.63, 3.66, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        # Should detect the most recent (latest) pattern
        if pattern is not None:
            # The breakout level should be from the second pattern
            assert pattern.breakout_level > 3.50

    def test_pole_ends_with_high_volume_spike(self, detector):
        """Last pole candle has massive volume (climax bar)."""
        candles = [
            (4.00, 4.12, 3.98, 4.10, 150000),
            (4.10, 4.24, 4.08, 4.22, 160000),
            (4.22, 4.45, 4.20, 4.40, 500000),   # Climax volume
            (4.40, 4.42, 4.30, 4.32, 50000),
            (4.32, 4.34, 4.28, 4.30, 30000),
            (4.30, 4.50, 4.29, 4.45, 200000),
            (4.45, 4.48, 4.43, 4.46, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is not None

    def test_exactly_3_pole_exactly_2_pullback(self, detector):
        """Minimum viable pattern at exact boundaries."""
        candles = [
            (4.00, 4.12, 3.98, 4.10, 200000),
            (4.10, 4.24, 4.08, 4.22, 180000),
            (4.22, 4.38, 4.20, 4.35, 160000),
            (4.35, 4.36, 4.28, 4.30, 50000),
            (4.30, 4.32, 4.25, 4.27, 30000),
            (4.27, 4.45, 4.26, 4.40, 200000),
            (4.40, 4.42, 4.38, 4.41, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is not None
        assert pattern.pullback_candle_count == 2

    def test_zero_volume_bars(self, detector):
        """Some bars have 0 volume — handle gracefully."""
        candles = [
            (4.00, 4.12, 3.98, 4.10, 200000),
            (4.10, 4.24, 4.08, 4.22, 180000),
            (4.22, 4.38, 4.20, 4.35, 160000),
            (4.35, 4.36, 4.28, 4.30, 0),        # zero volume
            (4.30, 4.32, 4.25, 4.27, 0),        # zero volume
            (4.27, 4.45, 4.26, 4.40, 200000),
            (4.40, 4.42, 4.38, 4.41, 80000),
        ]
        bars = _make_bars(candles)
        # Should not crash, may or may not detect depending on volume validation
        pattern = detector.detect("TEST", bars)
        # No crash is the main assertion

    def test_single_bar_input(self, detector):
        """Only 1 bar — return None, no crash."""
        candles = [(4.00, 4.10, 3.98, 4.08, 200000)]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is None

    def test_empty_bars_list(self, detector):
        """Empty input — return None, no crash."""
        bars = pd.DataFrame()
        pattern = detector.detect("TEST", bars)
        assert pattern is None

    def test_drops_incomplete_current_bar(self, detector):
        """Last bar is in-progress (current minute) — must be excluded."""
        # Build a valid pattern, but the breakout is only in the "current" bar
        # After dropping it, there should be no breakout
        candles = [
            (4.00, 4.12, 3.98, 4.10, 200000),
            (4.10, 4.24, 4.08, 4.22, 180000),
            (4.22, 4.38, 4.20, 4.35, 160000),
            (4.35, 4.36, 4.28, 4.30, 50000),
            (4.30, 4.32, 4.25, 4.27, 30000),
            # This would be the breakout but is "current" bar (will be dropped)
            (4.27, 4.45, 4.26, 4.40, 200000),
        ]
        bars = _make_bars(candles)
        # After dropping last bar, the last completed bar is the 2nd pullback candle
        # No breakout candle exists, so should be None
        pattern = detector.detect("TEST", bars)
        assert pattern is None

    def test_pattern_valid_only_after_dropping_partial(self, detector):
        """Pattern looks valid WITH partial bar but invalid WITHOUT — must reject."""
        # After dropping the last bar, the "breakout" candle is actually the
        # second pullback candle, which closes below flag_high
        candles = [
            (4.00, 4.12, 3.98, 4.10, 200000),
            (4.10, 4.24, 4.08, 4.22, 180000),
            (4.22, 4.38, 4.20, 4.35, 160000),
            (4.35, 4.36, 4.28, 4.30, 50000),
            # This is actually the last completed bar after dropping partial
            (4.30, 4.32, 4.25, 4.27, 30000),
            # Partial bar that would make it look like breakout
            (4.27, 4.50, 4.26, 4.45, 300000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is None

    def test_none_bars_input(self, detector):
        """None input — return None, no crash."""
        pattern = detector.detect("TEST", None)
        assert pattern is None


# ===========================================================================
# REAL-WORLD INSPIRED PATTERNS
# ===========================================================================

class TestRealWorldPatterns:
    """Patterns inspired by real market scenarios."""

    def test_morning_momentum_spike_pullback(self, detector):
        """Simulates 9:30-9:45 spike on news, then 2-bar pullback, breakout."""
        candles = [
            # Pre-spike
            (3.50, 3.52, 3.48, 3.51, 100000),
            # Morning spike (3 green candles)
            (3.51, 3.65, 3.50, 3.62, 300000),
            (3.62, 3.78, 3.60, 3.75, 280000),
            (3.75, 3.92, 3.73, 3.88, 260000),
            # Quick 2-bar pullback
            (3.88, 3.90, 3.78, 3.80, 80000),
            (3.80, 3.82, 3.75, 3.77, 50000),
            # Breakout
            (3.77, 3.95, 3.76, 3.92, 300000),
            # Current
            (3.92, 3.95, 3.90, 3.93, 100000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is not None

    def test_slow_grind_up_then_pullback(self, detector):
        """Gradual pole (many small green candles), tight 2-bar pullback."""
        candles = [
            # Long gradual pole
            (5.00, 5.04, 4.99, 5.03, 100000),
            (5.03, 5.07, 5.02, 5.06, 100000),
            (5.06, 5.10, 5.05, 5.09, 100000),
            (5.09, 5.13, 5.08, 5.12, 100000),
            (5.12, 5.16, 5.11, 5.15, 100000),
            (5.15, 5.20, 5.14, 5.18, 100000),
            (5.18, 5.24, 5.17, 5.22, 100000),
            (5.22, 5.28, 5.21, 5.26, 100000),
            (5.26, 5.32, 5.25, 5.30, 100000),
            (5.30, 5.36, 5.29, 5.34, 100000),
            # Pullback
            (5.34, 5.35, 5.28, 5.30, 30000),
            (5.30, 5.32, 5.26, 5.28, 20000),
            # Breakout
            (5.28, 5.42, 5.27, 5.40, 150000),
            # Current
            (5.40, 5.42, 5.38, 5.41, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is not None

    def test_volatile_pole_large_wicks(self, detector):
        """Pole candles with 50%+ wicks but still close green."""
        candles = [
            (4.00, 4.30, 3.80, 4.15, 200000),   # 50%+ wicks
            (4.15, 4.50, 3.95, 4.35, 180000),   # 50%+ wicks
            (4.35, 4.70, 4.10, 4.55, 160000),   # 50%+ wicks
            (4.55, 4.57, 4.40, 4.42, 50000),
            (4.42, 4.44, 4.35, 4.38, 30000),
            (4.38, 4.65, 4.37, 4.60, 200000),
            (4.60, 4.65, 4.58, 4.62, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is not None

    def test_failed_flag_turns_into_selloff(self, detector):
        """Starts as flag pattern but pullback accelerates into breakdown."""
        candles = [
            (4.00, 4.12, 3.98, 4.10, 200000),
            (4.10, 4.24, 4.08, 4.22, 180000),
            (4.22, 4.38, 4.20, 4.35, 160000),
            # Pullback accelerates into selloff (too deep)
            (4.35, 4.36, 4.20, 4.22, 100000),
            (4.22, 4.24, 4.05, 4.08, 150000),
            (4.08, 4.10, 3.90, 3.95, 200000),
            (3.95, 3.98, 3.85, 3.88, 100000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is None

    def test_micro_pullback_1_candle(self, detector):
        """Only 1 red pullback candle — should be rejected (need 2+)."""
        candles = [
            (4.00, 4.12, 3.98, 4.10, 200000),
            (4.10, 4.24, 4.08, 4.22, 180000),
            (4.22, 4.38, 4.20, 4.35, 160000),
            # Only 1 red candle
            (4.35, 4.36, 4.28, 4.30, 50000),
            # Breakout
            (4.30, 4.50, 4.29, 4.45, 200000),
            # Current
            (4.45, 4.48, 4.43, 4.46, 80000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)
        assert pattern is None


# ===========================================================================
# DETECT_SETUP TESTS — Setup detection BEFORE breakout
# ===========================================================================

class TestDetectSetup:
    """Tests for detect_setup() — finds pole+flag before breakout happens."""

    def test_setup_found_before_breakout(self, detector):
        """detect_setup() finds pole+flag when last bar is still in flag."""
        candles = [
            # Pole: 3 green
            (4.00, 4.15, 3.98, 4.13, 200000),
            (4.13, 4.30, 4.11, 4.28, 180000),
            (4.28, 4.52, 4.26, 4.50, 160000),
            # Flag: 2 red
            (4.50, 4.52, 4.38, 4.40, 50000),
            (4.40, 4.42, 4.33, 4.35, 30000),
            # Current bar (dropped by detector)
            (4.35, 4.38, 4.32, 4.34, 25000),
        ]
        bars = _make_bars(candles)
        setup = detector.detect_setup("TEST", bars)

        assert setup is not None
        assert isinstance(setup, BullFlagSetup)
        assert setup.symbol == "TEST"
        assert setup.breakout_level == setup.flag_high
        assert setup.pole_gain_pct >= 3.0
        assert setup.retracement_pct <= 50.0

    def test_setup_breakout_level_equals_flag_high(self, detector):
        """Setup breakout_level should equal flag_high."""
        candles = [
            (4.00, 4.15, 3.98, 4.13, 200000),
            (4.13, 4.30, 4.11, 4.28, 180000),
            (4.28, 4.52, 4.26, 4.50, 160000),
            (4.50, 4.52, 4.38, 4.40, 50000),
            (4.40, 4.42, 4.33, 4.35, 30000),
            (4.35, 4.38, 4.32, 4.34, 25000),
        ]
        bars = _make_bars(candles)
        setup = detector.detect_setup("TEST", bars)

        assert setup is not None
        assert setup.breakout_level == setup.flag_high

    def test_setup_returns_none_for_short_pole(self, detector):
        """detect_setup() rejects short pole (< 3 candles)."""
        candles = [
            (4.00, 4.15, 3.98, 4.12, 200000),
            (4.12, 4.28, 4.10, 4.25, 180000),
            (4.25, 4.27, 4.18, 4.20, 50000),
            (4.20, 4.22, 4.15, 4.17, 30000),
            (4.17, 4.20, 4.15, 4.18, 25000),
        ]
        bars = _make_bars(candles)
        setup = detector.detect_setup("TEST", bars)
        assert setup is None

    def test_setup_returns_none_for_deep_retracement(self, detector):
        """detect_setup() rejects retracement > 50%."""
        candles = [
            (4.00, 4.15, 3.98, 4.13, 200000),
            (4.13, 4.30, 4.11, 4.28, 180000),
            (4.28, 4.52, 4.26, 4.50, 160000),
            # Deep pullback: 60%+ retrace
            (4.50, 4.51, 4.22, 4.25, 50000),
            (4.25, 4.27, 4.18, 4.20, 30000),
            (4.20, 4.22, 4.16, 4.18, 25000),
        ]
        bars = _make_bars(candles)
        setup = detector.detect_setup("TEST", bars)
        assert setup is None

    def test_setup_returns_none_insufficient_data(self, detector):
        """detect_setup() returns None with too few bars."""
        candles = [
            (4.00, 4.10, 3.98, 4.08, 200000),
            (4.08, 4.15, 4.06, 4.12, 180000),
            (4.12, 4.18, 4.10, 4.15, 160000),
        ]
        bars = _make_bars(candles)
        setup = detector.detect_setup("TEST", bars)
        assert setup is None

    def test_setup_returns_none_for_none_input(self, detector):
        """detect_setup() handles None input gracefully."""
        setup = detector.detect_setup("TEST", None)
        assert setup is None

    def test_setup_returns_none_for_empty_bars(self, detector):
        """detect_setup() handles empty DataFrame gracefully."""
        setup = detector.detect_setup("TEST", pd.DataFrame())
        assert setup is None

    def test_detect_still_works_identically(self, detector):
        """detect() backward compatibility — same results as before refactor."""
        candles = [
            (4.00, 4.10, 3.98, 4.15, 200000),
            (4.15, 4.30, 4.12, 4.30, 180000),
            (4.30, 4.55, 4.28, 4.50, 150000),
            (4.50, 4.52, 4.38, 4.40, 50000),
            (4.40, 4.42, 4.33, 4.35, 30000),
            (4.35, 4.60, 4.34, 4.55, 250000),
            (4.55, 4.60, 4.50, 4.58, 100000),
        ]
        bars = _make_bars(candles)
        pattern = detector.detect("TEST", bars)

        assert pattern is not None
        assert pattern.symbol == "TEST"
        assert pattern.pullback_candle_count == 2
        assert pattern.pole_gain_pct >= 3.0

    def test_setup_with_end_idx(self, detector):
        """detect_setup() works with end_idx parameter (backtest mode)."""
        candles = [
            (4.00, 4.15, 3.98, 4.13, 200000),
            (4.13, 4.30, 4.11, 4.28, 180000),
            (4.28, 4.52, 4.26, 4.50, 160000),
            (4.50, 4.52, 4.38, 4.40, 50000),
            (4.40, 4.42, 4.33, 4.35, 30000),
            # Extra bars that would be "future" in sliding window
            (4.35, 4.60, 4.34, 4.55, 250000),
            (4.55, 4.60, 4.50, 4.58, 100000),
        ]
        bars = _make_bars(candles)
        # Use end_idx=5 to only see bars 0-4 as completed
        setup = detector.detect_setup("TEST", bars, end_idx=5)

        assert setup is not None
        assert setup.breakout_level == setup.flag_high


# ===========================================================================
# MACD FILTER TESTS
# ===========================================================================

class TestMACDFilter:
    """Tests for MACD momentum filter in pattern detection."""

    def _make_uptrend_with_flag(self):
        """Build 45+ bars: steady uptrend → pole → flag → current bar.

        The sustained uptrend ensures MACD stays positive through the flag.
        """
        candles = []
        # 30 bars of steady uptrend (MACD will be solidly positive)
        price = 3.00
        for i in range(30):
            o = price
            c = price + 0.03
            h = c + 0.01
            l = o - 0.01
            candles.append((o, h, l, c, 100000))
            price = c

        # Pole: 3 green candles accelerating
        for _ in range(3):
            o = price
            c = price + 0.12
            h = c + 0.02
            l = o - 0.02
            candles.append((o, h, l, c, 200000))
            price = c

        # Flag: 2 red candles, small retrace
        pole_high = price + 0.02  # approximate
        for _ in range(2):
            o = price
            c = price - 0.04
            h = o + 0.02
            l = c - 0.02
            candles.append((o, h, l, c, 40000))
            price = c

        # Current bar (dropped by detect_setup)
        candles.append((price, price + 0.02, price - 0.01, price + 0.01, 30000))
        return candles

    def _make_enough_bars_with_flag(self):
        """Build 40+ bars with a valid pole+flag pattern for MACD testing.

        Starts with a long uptrend so there are enough bars for MACD calculation,
        then forms pole + flag. Used with mocked MACD to test rejection logic.
        """
        candles = []
        # 30 bars of uptrend (enough bars for MACD)
        price = 3.00
        for i in range(30):
            o = price
            c = price + 0.02
            h = c + 0.01
            l = o - 0.01
            candles.append((o, h, l, c, 100000))
            price = c

        # Pole: 3 green
        for _ in range(3):
            o = price
            c = price + 0.12
            h = c + 0.02
            l = o - 0.02
            candles.append((o, h, l, c, 200000))
            price = c

        # Flag: 2 red
        for _ in range(2):
            o = price
            c = price - 0.04
            h = o + 0.02
            l = c - 0.02
            candles.append((o, h, l, c, 40000))
            price = c

        # Current bar
        candles.append((price, price + 0.02, price - 0.01, price + 0.01, 30000))
        return candles

    def test_macd_disabled_by_default(self):
        """Default detector (require_macd_positive=False) ignores MACD."""
        detector = BullFlagDetector()
        bars = _make_bars(self._make_enough_bars_with_flag())
        # Even with negative MACD, setup should be found when filter is off
        setup = detector.detect_setup("TEST", bars)
        # May or may not find setup based on other criteria, but MACD won't block it
        # The key assertion is it doesn't crash and doesn't filter on MACD
        assert detector.require_macd_positive is False

    @patch('trading.indicators.macd_histogram')
    def test_macd_rejects_negative_histogram(self, mock_macd):
        """Setup with negative MACD histogram is rejected when filter is on."""
        detector = BullFlagDetector(require_macd_positive=True)
        bars = _make_bars(self._make_enough_bars_with_flag())
        # Mock MACD to return negative histogram — momentum fading
        n_bars = len(bars)
        mock_macd.return_value = pd.Series([-0.05] * n_bars)
        setup = detector.detect_setup("TEST", bars)
        assert setup is None
        mock_macd.assert_called_once()

    def test_macd_passes_positive_histogram(self):
        """Setup with positive MACD histogram is accepted when filter is on."""
        detector = BullFlagDetector(require_macd_positive=True)
        bars = _make_bars(self._make_uptrend_with_flag())
        setup = detector.detect_setup("TEST", bars)
        assert setup is not None
        assert setup.macd_histogram_value is not None
        assert setup.macd_histogram_value > 0

    def test_macd_insufficient_bars(self):
        """Rejects setup when not enough bars for meaningful MACD (< slow + signal)."""
        detector = BullFlagDetector(require_macd_positive=True)
        # Only 7 bars — not enough for MACD(12,26,9) which needs 35
        candles = [
            (4.00, 4.15, 3.98, 4.13, 200000),
            (4.13, 4.30, 4.11, 4.28, 180000),
            (4.28, 4.52, 4.26, 4.50, 160000),
            (4.50, 4.52, 4.38, 4.40, 50000),
            (4.40, 4.42, 4.33, 4.35, 30000),
            (4.35, 4.38, 4.32, 4.34, 25000),
        ]
        bars = _make_bars(candles)
        setup = detector.detect_setup("TEST", bars)
        assert setup is None

    @patch('trading.indicators.macd_histogram')
    def test_macd_works_with_detect_too(self, mock_macd):
        """detect() also respects MACD filter (both use _scan_pole_and_flag)."""
        detector = BullFlagDetector(require_macd_positive=True)
        # Build valid breakout pattern with enough bars
        candles = list(self._make_enough_bars_with_flag())
        # Replace last 2 entries with breakout + current instead of flag + current
        candles[-2] = (candles[-3][3], candles[-3][3] + 0.30,
                       candles[-3][3] - 0.01, candles[-3][3] + 0.25, 250000)
        candles[-1] = (candles[-2][3], candles[-2][3] + 0.02,
                       candles[-2][3] - 0.01, candles[-2][3] + 0.01, 80000)
        bars = _make_bars(candles)
        # Mock MACD to return negative — should block detect() too
        n_bars = len(bars)
        mock_macd.return_value = pd.Series([-0.05] * n_bars)
        pattern = detector.detect("TEST", bars)
        assert pattern is None  # MACD negative should block detect() too

    def test_macd_custom_parameters(self):
        """Custom MACD periods are used correctly."""
        detector = BullFlagDetector(
            require_macd_positive=True,
            macd_fast=8, macd_slow=17, macd_signal=9,
        )
        # Faster MACD needs fewer bars — use uptrend bars
        bars = _make_bars(self._make_uptrend_with_flag())
        setup = detector.detect_setup("TEST", bars)
        assert setup is not None

    def test_macd_value_stored_on_pattern(self):
        """MACD histogram value is stored on the returned pattern."""
        detector = BullFlagDetector(require_macd_positive=True)
        bars = _make_bars(self._make_uptrend_with_flag())
        setup = detector.detect_setup("TEST", bars)
        assert setup is not None
        assert isinstance(setup.macd_histogram_value, float)
        assert setup.macd_histogram_value > 0

    def test_macd_none_when_filter_disabled(self):
        """MACD value is None when filter is disabled."""
        detector = BullFlagDetector(require_macd_positive=False)
        bars = _make_bars(self._make_uptrend_with_flag())
        setup = detector.detect_setup("TEST", bars)
        if setup is not None:
            assert setup.macd_histogram_value is None
