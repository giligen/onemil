"""Direct tests for trading/macd_tier_helpers.select_tier_multipliers.

This is the single source of truth for per-tier MACD multiplier selection,
called by BOTH backtest.py and trading/trading_engine.py. Testing the helper
directly guarantees identical tier routing in both paths without the overhead
of instantiating a live TradingEngine.

Also tests max_intraday_change_pre_entry for tier-classification input
parity between BT and PROD.
"""
from __future__ import annotations

import pytest

from trading.macd_tier_helpers import select_tier_multipliers
from trading.two_tier_filter import (
    TIER_A, TIER_EXTRAS, TIER_EDGE,
    max_intraday_change_pre_entry,
)


# -------------------------------------------------------------------------
# select_tier_multipliers — tier routing
# -------------------------------------------------------------------------

class TestTierMultiplierSelection:
    A = (1.8, 1.8, 1.0)          # A-tier (strong_pos, strong_neg, normal)
    E = (2.0, 2.0, 0.0)          # Extras-tier overrides

    def test_a_tier_intraday_25(self):
        """Intraday ≥ 20% → A-tier multipliers."""
        sp, sn, n, tier = select_tier_multipliers(
            25.0, *self.A, *self.E)
        assert (sp, sn, n) == self.A
        assert tier == TIER_A

    def test_extras_tier_intraday_15(self):
        """Intraday in [10, 20) → Extras multipliers."""
        sp, sn, n, tier = select_tier_multipliers(
            15.0, *self.A, *self.E)
        assert (sp, sn, n) == self.E
        assert tier == TIER_EXTRAS

    def test_edge_tier_falls_back_to_a_tier(self):
        """Intraday < 10% → edge → A-tier multipliers."""
        sp, sn, n, tier = select_tier_multipliers(
            5.0, *self.A, *self.E)
        assert (sp, sn, n) == self.A
        assert tier == TIER_EDGE

    def test_zero_intraday_is_edge(self):
        """intraday=0.0 → edge → A-tier fallback (back-compat)."""
        sp, sn, n, tier = select_tier_multipliers(
            0.0, *self.A, *self.E)
        assert (sp, sn, n) == self.A
        assert tier == TIER_EDGE

    def test_boundary_20_0_is_a_tier(self):
        """Boundary: intraday = 20.0 is A-tier (inclusive)."""
        _, _, _, tier = select_tier_multipliers(20.0, *self.A, *self.E)
        assert tier == TIER_A

    def test_boundary_10_0_is_extras(self):
        """Boundary: intraday = 10.0 is Extras (inclusive)."""
        _, _, _, tier = select_tier_multipliers(10.0, *self.A, *self.E)
        assert tier == TIER_EXTRAS

    def test_boundary_19_99_is_extras(self):
        """Boundary: intraday = 19.99 still Extras (strict < 20)."""
        _, _, _, tier = select_tier_multipliers(19.99, *self.A, *self.E)
        assert tier == TIER_EXTRAS

    def test_boundary_9_99_is_edge(self):
        _, _, _, tier = select_tier_multipliers(9.99, *self.A, *self.E)
        assert tier == TIER_EDGE

    def test_returns_extras_normal_zero_when_s2max(self):
        """Critical S2-max behavior: Extras normal returns 0.0 → skip."""
        sp, sn, n, tier = select_tier_multipliers(
            15.0,
            1.8, 1.8, 1.0,  # A-tier
            2.0, 2.0, 0.0,  # S2-max Extras
        )
        assert n == 0.0
        assert tier == TIER_EXTRAS

    def test_returns_extras_strong_positive_when_s2max(self):
        """Critical S2-max behavior: Extras strong returns 2.0."""
        sp, sn, _, _ = select_tier_multipliers(
            15.0, 1.8, 1.8, 1.0, 2.0, 2.0, 0.0)
        assert sp == 2.0
        assert sn == 2.0

    @pytest.mark.parametrize("ic,expected_tier", [
        (100.0, TIER_A),
        (30.0, TIER_A),
        (20.001, TIER_A),
        (20.0, TIER_A),
        (19.999, TIER_EXTRAS),
        (15.0, TIER_EXTRAS),
        (10.5, TIER_EXTRAS),
        (10.001, TIER_EXTRAS),
        (10.0, TIER_EXTRAS),
        (9.999, TIER_EDGE),
        (5.0, TIER_EDGE),
        (0.0, TIER_EDGE),
        (-1.0, TIER_EDGE),
    ])
    def test_tier_classification_matrix(self, ic, expected_tier):
        """Parametrized tier classification across the input range."""
        _, _, _, tier = select_tier_multipliers(
            ic, 1.8, 1.8, 1.0, 2.0, 2.0, 0.0)
        assert tier == expected_tier

    def test_no_extras_block_fallback(self):
        """If extras_* multipliers default to A-tier values (config fallback),
        Extras trades behave identically to A-tier trades."""
        # Simulate config.yaml missing extras_tier block → defaults = A-tier
        sp_a, sn_a, n_a, _ = select_tier_multipliers(
            15.0, 1.8, 1.8, 1.0, 1.8, 1.8, 1.0)
        sp_x, sn_x, n_x, _ = select_tier_multipliers(
            25.0, 1.8, 1.8, 1.0, 1.8, 1.8, 1.0)
        assert (sp_a, sn_a, n_a) == (sp_x, sn_x, n_x)


# -------------------------------------------------------------------------
# max_intraday_change_pre_entry — input parity for tier classifier
# -------------------------------------------------------------------------

class TestMaxIntradayChangePreEntry:
    """Ensures the function used by BT for tier classification produces
    predictable values. PROD uses `_qualified_max_intraday` which the scanner
    populates with an equivalent `max(gap_pct, range_pct)` running max —
    both paths should yield the same number for the same bar sequence."""

    def test_single_bar_gap_up_10pct(self):
        """Bar 1: close = prev_close × 1.10 → gap_pct = 10%."""
        bars = [
            ("09:30", 10.0, 11.0, 10.0, 11.0),
        ]
        # prev_close = 10.0, bar_high = 11.0 → range_pct = (11-10)/10*100 = 10%
        # gap_pct (from close=11.0) = (11-10)/10*100 = 10%
        # max = 10%
        result = max_intraday_change_pre_entry(bars, prev_close=10.0,
                                                 entry_ts_utc="09:31")
        assert result == pytest.approx(10.0, rel=1e-3)

    def test_two_bars_high_then_retrace(self):
        """Range is based on day_high/day_low, captures highest point even
        if stock retraces back. max_intraday_change should equal the max."""
        bars = [
            ("09:30", 10.0, 12.5, 10.0, 12.0),  # high 12.5, low 10.0
            ("09:31", 12.0, 12.0, 11.0, 11.0),  # low drops, but day_low stays 10
        ]
        # After bar 1: range_pct = 25%, gap_pct (from close 12) = 20% → max 25%
        # After bar 2: day_high still 12.5, day_low still 10.0 → range 25%,
        # gap_pct (from close 11) = 10% → max stays 25%
        result = max_intraday_change_pre_entry(bars, prev_close=10.0,
                                                 entry_ts_utc="09:32")
        assert result == pytest.approx(25.0, rel=1e-3)

    def test_entry_ts_filter(self):
        """Bars at/after entry_ts_utc are excluded from max computation."""
        bars = [
            ("09:30", 10.0, 11.0, 10.0, 11.0),
            ("09:31", 11.0, 20.0, 11.0, 20.0),  # big spike AFTER entry
        ]
        # entry_ts = 09:31 → bar at 09:31 excluded → only bar 1 counted
        result = max_intraday_change_pre_entry(bars, prev_close=10.0,
                                                 entry_ts_utc="09:31")
        assert result == pytest.approx(10.0, rel=1e-3)

    def test_no_bars_returns_none(self):
        result = max_intraday_change_pre_entry([], prev_close=10.0,
                                                 entry_ts_utc="09:31")
        assert result is None

    def test_none_prev_close_returns_range_only(self):
        """When prev_close is None, only range_pct is used (gap_pct skipped)."""
        bars = [("09:30", 10.0, 12.5, 9.5, 11.0)]
        # range_pct = (12.5 - 9.5) / 9.5 * 100 ≈ 31.58%
        result = max_intraday_change_pre_entry(bars, prev_close=None,
                                                 entry_ts_utc="09:31")
        assert result == pytest.approx(31.58, rel=1e-2)
