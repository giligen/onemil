"""
Unit tests for technical indicators.
"""

import pytest
import pandas as pd
import numpy as np

from trading.indicators import macd_histogram, rsi, obv, ema, vwap, stochastic


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


class TestRSI:
    """Tests for RSI calculation."""

    def test_strong_uptrend_high_rsi(self):
        """Strong uptrend produces RSI > 70."""
        closes = pd.Series([float(i) for i in range(1, 31)])
        r = rsi(closes)
        assert r.iloc[-1] > 70

    def test_strong_downtrend_low_rsi(self):
        """Strong downtrend produces RSI < 30."""
        closes = pd.Series([float(30 - i) for i in range(30)])
        r = rsi(closes)
        assert r.iloc[-1] < 30

    def test_flat_prices_near_50(self):
        """Flat prices produce RSI near 50."""
        closes = pd.Series([10.0] * 30)
        r = rsi(closes)
        # With no movement, RSI is NaN or 50-ish
        # ewm on all zeros gives NaN, which is expected
        assert pd.isna(r.iloc[-1]) or abs(r.iloc[-1] - 50) < 10

    def test_range_0_100(self):
        """RSI values should be between 0 and 100."""
        np.random.seed(42)
        closes = pd.Series(np.cumsum(np.random.randn(100)) + 50)
        r = rsi(closes)
        valid = r.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_same_length(self):
        """Output same length as input."""
        closes = pd.Series([float(i) for i in range(30)])
        r = rsi(closes)
        assert len(r) == len(closes)


class TestOBV:
    """Tests for On-Balance Volume."""

    def test_rising_prices_positive_obv(self):
        """Rising prices with volume produce positive OBV."""
        closes = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
        volumes = pd.Series([100, 200, 300, 200, 100])
        o = obv(closes, volumes)
        assert o.iloc[-1] > 0

    def test_falling_prices_negative_obv(self):
        """Falling prices produce negative OBV."""
        closes = pd.Series([14.0, 13.0, 12.0, 11.0, 10.0])
        volumes = pd.Series([100, 200, 300, 200, 100])
        o = obv(closes, volumes)
        assert o.iloc[-1] < 0

    def test_same_length(self):
        """Output same length as input."""
        closes = pd.Series([10.0, 11.0, 10.5])
        volumes = pd.Series([100, 200, 150])
        o = obv(closes, volumes)
        assert len(o) == len(closes)


class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_uptrend_ema_below_price(self):
        """In uptrend, EMA lags below current price."""
        closes = pd.Series([float(i) for i in range(1, 31)])
        e = ema(closes, period=9)
        assert e.iloc[-1] < closes.iloc[-1]

    def test_same_length(self):
        """Output same length as input."""
        closes = pd.Series([10.0] * 20)
        e = ema(closes, period=9)
        assert len(e) == len(closes)

    def test_flat_prices_equals_price(self):
        """Flat prices produce EMA equal to price."""
        closes = pd.Series([10.0] * 30)
        e = ema(closes, period=9)
        assert abs(e.iloc[-1] - 10.0) < 0.001


class TestVWAP:
    """Tests for Volume-Weighted Average Price."""

    def test_constant_price(self):
        """Constant price produces VWAP equal to that price."""
        bars = pd.DataFrame({
            'high': [10.0] * 5,
            'low': [10.0] * 5,
            'close': [10.0] * 5,
            'volume': [100, 200, 300, 400, 500],
        })
        v = vwap(bars['high'], bars['low'], bars['close'], bars['volume'])
        assert abs(v.iloc[-1] - 10.0) < 1e-9

    def test_volume_weighting(self):
        """Higher volume bars pull VWAP toward their typical price."""
        bars = pd.DataFrame({
            'high': [10.0, 12.0],
            'low': [10.0, 12.0],
            'close': [10.0, 12.0],
            'volume': [100, 900],  # 90% of volume at $12
        })
        v = vwap(bars['high'], bars['low'], bars['close'], bars['volume'])
        # VWAP should be much closer to 12 than to 11 (simple avg)
        assert v.iloc[-1] > 11.5

    def test_same_length(self):
        """Output same length as input."""
        n = 20
        bars = pd.DataFrame({
            'high': [10.0] * n, 'low': [10.0] * n,
            'close': [10.0] * n, 'volume': [100] * n,
        })
        v = vwap(bars['high'], bars['low'], bars['close'], bars['volume'])
        assert len(v) == n

    def test_zero_volume_safe(self):
        """Zero volume doesn't crash (divide-by-zero protection)."""
        bars = pd.DataFrame({
            'high': [10.0, 11.0],
            'low': [10.0, 11.0],
            'close': [10.0, 11.0],
            'volume': [0, 0],
        })
        v = vwap(bars['high'], bars['low'], bars['close'], bars['volume'])
        assert len(v) == 2  # no exception


class TestStochastic:
    """Tests for Stochastic oscillator."""

    def test_strong_uptrend_overbought(self):
        """Strong uptrend produces %K > 80."""
        highs = pd.Series([float(i) + 0.5 for i in range(1, 31)])
        lows = pd.Series([float(i) - 0.5 for i in range(1, 31)])
        closes = pd.Series([float(i) for i in range(1, 31)])
        k, d = stochastic(highs, lows, closes)
        assert k.iloc[-1] > 80

    def test_strong_downtrend_oversold(self):
        """Strong downtrend produces %K < 20."""
        highs = pd.Series([float(30 - i) + 0.5 for i in range(30)])
        lows = pd.Series([float(30 - i) - 0.5 for i in range(30)])
        closes = pd.Series([float(30 - i) for i in range(30)])
        k, d = stochastic(highs, lows, closes)
        assert k.iloc[-1] < 20

    def test_k_crosses_above_d_on_reversal(self):
        """%K crosses above %D when price reverses up."""
        # 20 bars declining, then 5 bars rising sharply
        down = [float(30 - i) for i in range(20)]  # 30 → 11
        up = [12.0, 14.0, 16.0, 18.0, 20.0]
        closes = pd.Series(down + up)
        highs = closes + 0.5
        lows = closes - 0.5
        k, d = stochastic(highs, lows, closes)
        # By end of up-move, %K should be above %D
        assert k.iloc[-1] > d.iloc[-1]

    def test_same_length(self):
        """Output same length as input."""
        n = 30
        highs = pd.Series([float(i) + 0.5 for i in range(n)])
        lows = pd.Series([float(i) - 0.5 for i in range(n)])
        closes = pd.Series([float(i) for i in range(n)])
        k, d = stochastic(highs, lows, closes)
        assert len(k) == n
        assert len(d) == n

    def test_range_0_100(self):
        """%K and %D values (once warm) are bounded to [0, 100]."""
        np.random.seed(7)
        closes = pd.Series(np.cumsum(np.random.randn(100)) + 50)
        highs = closes + np.abs(np.random.randn(100)) * 0.3
        lows = closes - np.abs(np.random.randn(100)) * 0.3
        k, d = stochastic(highs, lows, closes)
        kv = k.dropna()
        dv = d.dropna()
        assert (kv >= 0).all() and (kv <= 100).all()
        assert (dv >= 0).all() and (dv <= 100).all()

    def test_flat_range_safe(self):
        """Flat high/low range doesn't crash (divide-by-zero protection)."""
        highs = pd.Series([10.0] * 20)
        lows = pd.Series([10.0] * 20)
        closes = pd.Series([10.0] * 20)
        k, d = stochastic(highs, lows, closes)
        assert len(k) == 20  # no exception
