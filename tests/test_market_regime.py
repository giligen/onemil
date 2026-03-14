"""
Unit tests for MarketRegimeFilter.

Tests cover:
- Regime allowed when SPY is above threshold
- Regime blocked when SPY is below threshold
- Exact boundary behavior (>= threshold)
- Disabled filter always allows
- Insufficient data allows (safe default)
- No lookahead — bars on trade_date are NOT used
- Weekend handling — Friday close used for Monday check
"""

import pytest
from datetime import date

from trading.market_regime import MarketRegimeFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(dates_and_closes):
    """Build SPY bar list from (date, close) pairs."""
    return [{'date': d, 'close': c} for d, c in dates_and_closes]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMarketRegimeFilter:
    """Unit tests for MarketRegimeFilter."""

    def test_regime_ok_above_threshold(self):
        """SPY up over 5 days — trading allowed."""
        bars = _make_bars([
            (date(2025, 3, 3), 500.0),   # T-6
            (date(2025, 3, 4), 501.0),
            (date(2025, 3, 5), 502.0),
            (date(2025, 3, 6), 503.0),
            (date(2025, 3, 7), 504.0),
            (date(2025, 3, 10), 510.0),  # T-1
        ])
        mrf = MarketRegimeFilter(enabled=True, spy_5d_return_min=-2.0)
        mrf.load_spy_bars(bars)

        # Trade on Mar 11 — uses closes up to Mar 10
        assert mrf.is_regime_ok(date(2025, 3, 11)) is True
        ret = mrf.get_spy_5d_return(date(2025, 3, 11))
        assert ret is not None
        assert ret > 0  # SPY went up

    def test_regime_blocked_below_threshold(self):
        """SPY down >2% over 5 days — trading blocked."""
        bars = _make_bars([
            (date(2025, 3, 3), 500.0),   # T-6
            (date(2025, 3, 4), 498.0),
            (date(2025, 3, 5), 496.0),
            (date(2025, 3, 6), 494.0),
            (date(2025, 3, 7), 490.0),
            (date(2025, 3, 10), 485.0),  # T-1: (485/500-1)*100 = -3.0%
        ])
        mrf = MarketRegimeFilter(enabled=True, spy_5d_return_min=-2.0)
        mrf.load_spy_bars(bars)

        assert mrf.is_regime_ok(date(2025, 3, 11)) is False
        ret = mrf.get_spy_5d_return(date(2025, 3, 11))
        assert ret is not None
        assert ret == pytest.approx(-3.0, abs=0.01)

    def test_exact_threshold_boundary(self):
        """SPY 5d return just at threshold — allowed (>= threshold)."""
        # Use -1.99% (just above -2.0) to avoid floating point edge
        # 500 * 0.9801 = 490.05 → (490.05/500 - 1)*100 = -1.99%
        bars = _make_bars([
            (date(2025, 3, 3), 500.0),   # T-6
            (date(2025, 3, 4), 499.0),
            (date(2025, 3, 5), 497.0),
            (date(2025, 3, 6), 495.0),
            (date(2025, 3, 7), 493.0),
            (date(2025, 3, 10), 490.05),  # T-1: (490.05/500-1)*100 = -1.99%
        ])
        mrf = MarketRegimeFilter(enabled=True, spy_5d_return_min=-2.0)
        mrf.load_spy_bars(bars)

        # -1.99% is above -2.0% threshold, should be allowed
        assert mrf.is_regime_ok(date(2025, 3, 11)) is True
        ret = mrf.get_spy_5d_return(date(2025, 3, 11))
        assert ret == pytest.approx(-1.99, abs=0.01)

    def test_disabled_always_allows(self):
        """When enabled=False, is_regime_ok always returns True."""
        bars = _make_bars([
            (date(2025, 3, 3), 500.0),
            (date(2025, 3, 4), 490.0),
            (date(2025, 3, 5), 480.0),
            (date(2025, 3, 6), 470.0),
            (date(2025, 3, 7), 460.0),
            (date(2025, 3, 10), 440.0),  # -12% — extreme crash
        ])
        mrf = MarketRegimeFilter(enabled=False, spy_5d_return_min=-2.0)
        mrf.load_spy_bars(bars)

        assert mrf.is_regime_ok(date(2025, 3, 11)) is True

    def test_insufficient_data_allows(self):
        """Fewer than 6 prior bars — returns True (safe default)."""
        bars = _make_bars([
            (date(2025, 3, 3), 500.0),
            (date(2025, 3, 4), 490.0),
            (date(2025, 3, 5), 480.0),
        ])
        mrf = MarketRegimeFilter(enabled=True, spy_5d_return_min=-2.0)
        mrf.load_spy_bars(bars)

        assert mrf.is_regime_ok(date(2025, 3, 6)) is True
        assert mrf.get_spy_5d_return(date(2025, 3, 6)) is None

    def test_no_lookahead(self):
        """Bars on trade_date are NOT used — only T-1 and earlier."""
        bars = _make_bars([
            (date(2025, 3, 3), 500.0),   # T-6
            (date(2025, 3, 4), 499.0),
            (date(2025, 3, 5), 498.0),
            (date(2025, 3, 6), 497.0),
            (date(2025, 3, 7), 496.0),
            (date(2025, 3, 10), 495.0),  # T-1: (495/500-1)*100 = -1.0%
            # Trade date bar — crash that should NOT be used
            (date(2025, 3, 11), 450.0),  # If used, would give much worse return
        ])
        mrf = MarketRegimeFilter(enabled=True, spy_5d_return_min=-2.0)
        mrf.load_spy_bars(bars)

        # Should use T-1=Mar10 and T-6=Mar3, NOT Mar11
        ret = mrf.get_spy_5d_return(date(2025, 3, 11))
        assert ret == pytest.approx(-1.0, abs=0.01)  # Not the crash value
        assert mrf.is_regime_ok(date(2025, 3, 11)) is True  # -1.0% > -2.0% threshold

    def test_weekend_handling(self):
        """Friday close is used for Monday check (no Sat/Sun bars)."""
        bars = _make_bars([
            # Previous week provides enough history
            (date(2025, 2, 24), 495.0),  # Mon
            (date(2025, 2, 25), 496.0),  # Tue
            (date(2025, 2, 26), 497.0),  # Wed
            (date(2025, 2, 27), 498.0),  # Thu
            (date(2025, 2, 28), 499.0),  # Fri
            (date(2025, 3, 3), 500.0),   # Mon
            (date(2025, 3, 4), 502.0),   # Tue
            (date(2025, 3, 5), 504.0),   # Wed
            (date(2025, 3, 6), 506.0),   # Thu
            (date(2025, 3, 7), 508.0),   # Fri
            # No Sat/Sun bars — just like real markets
            (date(2025, 3, 10), 510.0),  # Mon
        ])
        mrf = MarketRegimeFilter(enabled=True, spy_5d_return_min=-2.0)
        mrf.load_spy_bars(bars)

        # Monday Mar 10: prior dates=[Feb24..Mar7] (10 dates), last 6=[Feb28..Mar7]
        # T-1=Mar7=508, T-6=Feb28=499 → (508/499-1)*100
        ret_monday = mrf.get_spy_5d_return(date(2025, 3, 10))
        assert ret_monday == pytest.approx((508.0 / 499.0 - 1) * 100, abs=0.01)

        # Tuesday Mar 11: prior dates=[Feb24..Mar10] (11 dates), last 6=[Mar3..Mar10]
        # T-1=Mar10=510, T-6=Mar3=500 → (510/500-1)*100 = 2.0%
        ret_tuesday = mrf.get_spy_5d_return(date(2025, 3, 11))
        assert ret_tuesday == pytest.approx((510.0 / 500.0 - 1) * 100, abs=0.01)

    def test_load_spy_bars_with_string_dates(self):
        """load_spy_bars handles string dates (from DB cache)."""
        bars = [
            {'date': '2025-03-03', 'close': 500.0},
            {'date': '2025-03-04', 'close': 501.0},
            {'date': '2025-03-05', 'close': 502.0},
            {'date': '2025-03-06', 'close': 503.0},
            {'date': '2025-03-07', 'close': 504.0},
            {'date': '2025-03-10', 'close': 505.0},
        ]
        mrf = MarketRegimeFilter(enabled=True, spy_5d_return_min=-2.0)
        mrf.load_spy_bars(bars)

        assert mrf.is_regime_ok(date(2025, 3, 11)) is True

    def test_no_bars_loaded(self):
        """No bars loaded — allows trading (safe default)."""
        mrf = MarketRegimeFilter(enabled=True, spy_5d_return_min=-2.0)
        assert mrf.is_regime_ok(date(2025, 3, 11)) is True
        assert mrf.get_spy_5d_return(date(2025, 3, 11)) is None
