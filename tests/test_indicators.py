"""
Unit tests for technical indicators.
"""

import pytest
import pandas as pd
import numpy as np

from trading.indicators import macd_histogram


class TestMACDHistogram:
    """Tests for MACD histogram calculation."""

    def test_trending_up_positive_histogram(self):
        """Monotonically increasing prices produce positive histogram."""
        closes = pd.Series([float(i) for i in range(1, 51)])
        hist = macd_histogram(closes)
        # After warmup, histogram should be positive for uptrend
        assert hist.iloc[-1] > 0

    def test_trending_down_negative_histogram(self):
        """Monotonically decreasing prices produce negative histogram."""
        closes = pd.Series([float(50 - i) for i in range(50)])
        hist = macd_histogram(closes)
        assert hist.iloc[-1] < 0

    def test_flat_prices_near_zero(self):
        """Flat prices produce histogram near zero."""
        closes = pd.Series([10.0] * 50)
        hist = macd_histogram(closes)
        assert abs(hist.iloc[-1]) < 0.001

    def test_returns_series_same_length(self):
        """Output series has same length as input."""
        closes = pd.Series([float(i) for i in range(40)])
        hist = macd_histogram(closes)
        assert len(hist) == len(closes)

    def test_custom_parameters(self):
        """Custom fast/slow/signal periods work."""
        closes = pd.Series([float(i) for i in range(1, 51)])
        hist = macd_histogram(closes, fast=8, slow=17, signal=9)
        assert hist.iloc[-1] > 0
        # Faster params should give different (larger) values
        hist_default = macd_histogram(closes)
        assert hist.iloc[-1] != hist_default.iloc[-1]

    def test_momentum_fade_after_spike(self):
        """Simulates pole+flag: spike up then flat → histogram should decrease."""
        # 30 bars flat, then 5-bar spike, then 10 bars flat at top
        flat_start = [10.0] * 30
        spike = [10.5, 11.0, 11.5, 12.0, 12.5]
        flat_top = [12.5] * 10
        closes = pd.Series(flat_start + spike + flat_top)
        hist = macd_histogram(closes)

        # During spike, histogram should increase
        spike_end = 35
        # After flat consolidation, histogram should decrease toward zero
        assert hist.iloc[-1] < hist.iloc[spike_end]

    def test_short_series(self):
        """Short series still returns values (EWM handles warmup)."""
        closes = pd.Series([1.0, 2.0, 3.0])
        hist = macd_histogram(closes)
        assert len(hist) == 3
        assert not hist.isna().any()
