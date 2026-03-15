"""
Unit tests for MarketRegimeFilter — volatility + trend regime filter.

Tests cover:
- Regime allowed when SPY is calm or in uptrend
- Regime blocked when SPY is volatile AND in downtrend
- Individual condition checks (high vol only, below SMA only)
- Exact boundary behavior
- Disabled filter always allows
- Insufficient data allows (safe default)
- No lookahead — bars on trade_date are NOT used
- Mathematical accuracy of vol_5d and SMA calculations
- max_trades_per_day property
"""

import pytest
from datetime import date

from trading.market_regime import MarketRegimeFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlc_bars(dates_and_data):
    """Build SPY bar list from (date, open, high, low, close) tuples."""
    return [
        {'date': d, 'open': o, 'high': h, 'low': lo, 'close': c}
        for d, o, h, lo, c in dates_and_data
    ]


def _make_close_only_bars(dates_and_closes):
    """Build SPY bar list from (date, close) pairs — backwards compat."""
    return [{'date': d, 'close': c} for d, c in dates_and_closes]


def _build_50_sma_bars(base_close=500.0, sma_period=50):
    """
    Build 55 bars where close is constant at base_close.
    SMA will equal base_close. Use to test below/above SMA.
    All bars have 1% daily range for predictable vol.
    """
    from datetime import timedelta
    bars = []
    d = date(2025, 1, 1)
    for i in range(55):
        # Skip weekends
        while d.weekday() >= 5:
            d += timedelta(days=1)
        bars.append({
            'date': d,
            'open': base_close,
            'high': base_close * 1.005,  # 0.5% above
            'low': base_close * 0.995,   # 0.5% below
            'close': base_close,
        })
        d += timedelta(days=1)
    return bars


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMarketRegimeFilter:
    """Unit tests for MarketRegimeFilter vol+trend regime."""

    def test_regime_ok_low_vol_above_sma(self):
        """Normal market: low vol + above SMA — trading allowed."""
        # Build 55 bars with stable prices (low vol, close above SMA)
        bars = _build_50_sma_bars(base_close=500.0)
        # Last 5 bars: slightly higher to be above SMA
        for i in range(-5, 0):
            bars[i]['close'] = 505.0
            bars[i]['high'] = 506.0
            bars[i]['low'] = 504.0

        mrf = MarketRegimeFilter(enabled=True, vol_threshold=1.5, sma_period=50)
        mrf.load_spy_bars(bars)

        trade_date = bars[-1]['date']
        from datetime import timedelta
        next_day = trade_date + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)

        assert mrf.is_regime_ok(next_day) is True

    def test_regime_blocked_high_vol_below_sma(self):
        """Volatile downtrend: high vol + below SMA — trading blocked."""
        bars = _build_50_sma_bars(base_close=500.0)
        # Make last 5 bars volatile AND below SMA
        for i in range(-5, 0):
            bars[i]['close'] = 480.0  # well below 500 SMA
            bars[i]['high'] = 495.0   # high range
            bars[i]['low'] = 470.0    # (495-470)/480 = 5.2%
            bars[i]['open'] = 485.0

        mrf = MarketRegimeFilter(enabled=True, vol_threshold=1.5, sma_period=50)
        mrf.load_spy_bars(bars)

        trade_date = bars[-1]['date']
        from datetime import timedelta
        next_day = trade_date + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)

        assert mrf.is_regime_ok(next_day) is False

    def test_high_vol_above_sma_allowed(self):
        """High vol but above SMA (uptrend) — trading allowed."""
        bars = _build_50_sma_bars(base_close=500.0)
        # Make last 5 bars volatile but ABOVE SMA
        for i in range(-5, 0):
            bars[i]['close'] = 520.0  # above 500 SMA
            bars[i]['high'] = 535.0   # high range
            bars[i]['low'] = 510.0    # (535-510)/520 = 4.8%
            bars[i]['open'] = 515.0

        mrf = MarketRegimeFilter(enabled=True, vol_threshold=1.5, sma_period=50)
        mrf.load_spy_bars(bars)

        trade_date = bars[-1]['date']
        from datetime import timedelta
        next_day = trade_date + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)

        assert mrf.is_regime_ok(next_day) is True

    def test_low_vol_below_sma_allowed(self):
        """Below SMA but low volatility (calm downtrend) — trading allowed."""
        bars = _build_50_sma_bars(base_close=500.0)
        # Make last 5 bars below SMA but very low vol
        for i in range(-5, 0):
            bars[i]['close'] = 498.0  # slightly below 500 SMA
            bars[i]['high'] = 499.0   # tiny range
            bars[i]['low'] = 497.0    # (499-497)/498 = 0.4%
            bars[i]['open'] = 498.0

        mrf = MarketRegimeFilter(enabled=True, vol_threshold=1.5, sma_period=50)
        mrf.load_spy_bars(bars)

        trade_date = bars[-1]['date']
        from datetime import timedelta
        next_day = trade_date + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)

        assert mrf.is_regime_ok(next_day) is True

    def test_boundary_exact_threshold(self):
        """Vol just below threshold and below SMA — should NOT block."""
        bars = _build_50_sma_bars(base_close=500.0)
        # Set last 5 bars to have < 1.5% daily range and below SMA
        for i in range(-5, 0):
            close = 490.0
            # daily_range = (high - low) / close * 100 = 1.4%
            # so high - low = 1.4 * 490 / 100 = 6.86
            bars[i]['close'] = close
            bars[i]['low'] = close - 3.43
            bars[i]['high'] = close + 3.43

        mrf = MarketRegimeFilter(enabled=True, vol_threshold=1.5, sma_period=50)
        mrf.load_spy_bars(bars)

        trade_date = bars[-1]['date']
        from datetime import timedelta
        next_day = trade_date + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)

        # vol ~1.4% < 1.5% threshold, so should be allowed even though below SMA
        vol = mrf.get_spy_vol_5d(next_day)
        assert vol is not None
        assert vol < 1.5
        assert mrf.is_regime_ok(next_day) is True

    def test_disabled_always_allows(self):
        """When enabled=False, is_regime_ok always returns True."""
        bars = _build_50_sma_bars(base_close=500.0)
        # Make conditions that would block
        for i in range(-5, 0):
            bars[i]['close'] = 480.0
            bars[i]['high'] = 495.0
            bars[i]['low'] = 470.0

        mrf = MarketRegimeFilter(enabled=False, vol_threshold=1.5, sma_period=50)
        mrf.load_spy_bars(bars)

        trade_date = bars[-1]['date']
        from datetime import timedelta
        next_day = trade_date + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)

        assert mrf.is_regime_ok(next_day) is True

    def test_insufficient_data_allows(self):
        """Fewer than required bars — returns True (safe default)."""
        bars = _make_ohlc_bars([
            (date(2025, 3, 3), 500, 505, 498, 502),
            (date(2025, 3, 4), 502, 507, 500, 504),
            (date(2025, 3, 5), 504, 509, 502, 506),
        ])
        mrf = MarketRegimeFilter(enabled=True, vol_threshold=1.5, sma_period=50)
        mrf.load_spy_bars(bars)

        assert mrf.is_regime_ok(date(2025, 3, 6)) is True

    def test_no_lookahead(self):
        """Bars on trade_date are NOT used — only T-1 and earlier."""
        bars = _build_50_sma_bars(base_close=500.0)
        # Last bar is "normal"
        trade_date = bars[-1]['date']
        from datetime import timedelta
        next_day = trade_date + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)

        # Add a bar ON trade_date that would be very volatile and below SMA
        bars.append({
            'date': next_day,
            'open': 450.0,
            'high': 480.0,
            'low': 420.0,  # massive range
            'close': 430.0,  # way below SMA
        })

        mrf = MarketRegimeFilter(enabled=True, vol_threshold=1.5, sma_period=50)
        mrf.load_spy_bars(bars)

        # Should use data BEFORE next_day only, ignoring the crash bar
        assert mrf.is_regime_ok(next_day) is True

    def test_get_spy_vol_5d_calculation(self):
        """Verify vol_5d math: average daily range % over 5 prior days."""
        bars = _make_ohlc_bars([
            (date(2025, 3, 3), 500, 510, 495, 500),  # range = (510-495)/500 = 3.0%
            (date(2025, 3, 4), 500, 508, 496, 500),  # range = (508-496)/500 = 2.4%
            (date(2025, 3, 5), 500, 506, 498, 500),  # range = (506-498)/500 = 1.6%
            (date(2025, 3, 6), 500, 504, 497, 500),  # range = (504-497)/500 = 1.4%
            (date(2025, 3, 7), 500, 512, 494, 500),  # range = (512-494)/500 = 3.6%
        ])
        mrf = MarketRegimeFilter(enabled=True)
        mrf.load_spy_bars(bars)

        vol = mrf.get_spy_vol_5d(date(2025, 3, 10))
        expected = (3.0 + 2.4 + 1.6 + 1.4 + 3.6) / 5  # = 2.4%
        assert vol == pytest.approx(expected, abs=0.01)

    def test_get_spy_sma_calculation(self):
        """Verify SMA math with small period."""
        bars = _make_ohlc_bars([
            (date(2025, 3, 3), 500, 510, 495, 500),
            (date(2025, 3, 4), 500, 508, 496, 502),
            (date(2025, 3, 5), 500, 506, 498, 504),
            (date(2025, 3, 6), 500, 504, 497, 506),
            (date(2025, 3, 7), 500, 512, 494, 508),
        ])
        mrf = MarketRegimeFilter(enabled=True, sma_period=5)
        mrf.load_spy_bars(bars)

        sma = mrf.get_spy_sma(date(2025, 3, 10), period=5)
        expected = (500 + 502 + 504 + 506 + 508) / 5  # = 504.0
        assert sma == pytest.approx(expected, abs=0.01)

    def test_max_trades_per_day_property(self):
        """max_trades_per_day is stored and accessible."""
        mrf = MarketRegimeFilter(max_trades_per_day=7)
        assert mrf.max_trades_per_day == 7

        mrf2 = MarketRegimeFilter()
        assert mrf2.max_trades_per_day == 5  # default

    def test_get_regime_info(self):
        """get_regime_info returns structured dict."""
        bars = _build_50_sma_bars(base_close=500.0)
        mrf = MarketRegimeFilter(enabled=True, vol_threshold=1.5, sma_period=50)
        mrf.load_spy_bars(bars)

        trade_date = bars[-1]['date']
        from datetime import timedelta
        next_day = trade_date + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)

        info = mrf.get_regime_info(next_day)
        assert 'vol_5d' in info
        assert 'sma' in info
        assert 'is_below_sma' in info
        assert 'is_ok' in info
        assert isinstance(info['is_ok'], bool)

    def test_load_spy_bars_with_string_dates(self):
        """load_spy_bars handles string dates (from DB cache)."""
        bars = [
            {'date': '2025-03-03', 'open': 500, 'high': 505, 'low': 498, 'close': 502},
            {'date': '2025-03-04', 'open': 502, 'high': 507, 'low': 500, 'close': 504},
        ]
        mrf = MarketRegimeFilter(enabled=True)
        mrf.load_spy_bars(bars)
        assert len(mrf._sorted_dates) == 2

    def test_no_bars_loaded(self):
        """No bars loaded — allows trading (safe default)."""
        mrf = MarketRegimeFilter(enabled=True)
        assert mrf.is_regime_ok(date(2025, 3, 11)) is True
        assert mrf.get_spy_vol_5d(date(2025, 3, 11)) is None
        assert mrf.get_spy_sma(date(2025, 3, 11)) is None

    def test_backwards_compat_close_only_bars(self):
        """Bars with only 'close' still load (backwards compat)."""
        bars = _make_close_only_bars([
            (date(2025, 3, 3), 500.0),
            (date(2025, 3, 4), 501.0),
        ])
        mrf = MarketRegimeFilter(enabled=True)
        mrf.load_spy_bars(bars)
        assert len(mrf._sorted_dates) == 2
