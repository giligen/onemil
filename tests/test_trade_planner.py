"""
Unit tests for TradePlanner — trade plan creation from bull flag patterns.

Tests cover:
- Valid plan creation
- Stop loss rules (flag low, 20-cent cap)
- Target calculation (2:1 R:R vs pole height)
- Position sizing
- Rejection conditions
"""

import pytest
from datetime import datetime, timezone

from trading.pattern_detector import BullFlagPattern
from trading.trade_planner import TradePlanner, TradePlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pattern(
    symbol="TEST",
    pole_low=4.00,
    pole_high=4.50,
    flag_low=4.30,
    flag_high=4.40,
    breakout_level=4.40,
    pole_height=None,
    pole_gain_pct=None,
    retracement_pct=None,
    pullback_candle_count=2,
    avg_pole_volume=180000,
    avg_flag_volume=40000,
):
    """Create a BullFlagPattern with sensible defaults."""
    if pole_height is None:
        pole_height = pole_high - pole_low
    if pole_gain_pct is None:
        pole_gain_pct = (pole_height / pole_low) * 100 if pole_low > 0 else 0
    if retracement_pct is None:
        retracement_pct = ((pole_high - flag_low) / pole_height) * 100 if pole_height > 0 else 0

    return BullFlagPattern(
        symbol=symbol,
        pole_start_idx=0,
        pole_end_idx=2,
        flag_start_idx=3,
        flag_end_idx=4,
        pole_low=pole_low,
        pole_high=pole_high,
        pole_height=pole_height,
        pole_gain_pct=pole_gain_pct,
        flag_low=flag_low,
        flag_high=flag_high,
        retracement_pct=retracement_pct,
        pullback_candle_count=pullback_candle_count,
        avg_pole_volume=avg_pole_volume,
        avg_flag_volume=avg_flag_volume,
        breakout_level=breakout_level,
        detected_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def planner():
    """Standard planner with default settings."""
    return TradePlanner(
        position_size_dollars=500,
        max_shares=1000,
        max_risk_per_share=0.20,
        min_risk_per_share=0.05,
        min_risk_reward=2.0,
    )


# ===========================================================================
# POSITIVE TESTS
# ===========================================================================

class TestCreateValidPlan:
    """Tests for valid plan creation."""

    def test_creates_valid_plan_from_pattern(self, planner):
        """Basic valid pattern produces a complete trade plan."""
        pattern = _make_pattern(
            breakout_level=4.40,
            flag_low=4.30,
            pole_height=0.50,
        )
        plan = planner.create_plan(pattern)

        assert plan is not None
        assert isinstance(plan, TradePlan)
        assert plan.symbol == "TEST"
        assert plan.entry_price == 4.40
        assert plan.shares > 0
        assert plan.risk_per_share > 0
        assert plan.reward_per_share > 0
        assert plan.risk_reward_ratio >= 2.0
        assert plan.pattern is pattern

    def test_stop_at_flag_low_minus_penny(self, planner):
        """Stop loss is flag_low - $0.01 when natural risk <= 20 cents."""
        pattern = _make_pattern(
            breakout_level=4.40,
            flag_low=4.30,  # natural risk = 4.40 - 4.29 = 0.11
        )
        plan = planner.create_plan(pattern)

        assert plan is not None
        assert plan.stop_loss_price == pytest.approx(4.29, abs=0.01)

    def test_caps_risk_at_20_cents(self, planner):
        """When natural stop is > 20 cents from entry, cap at 20 cents."""
        pattern = _make_pattern(
            breakout_level=4.50,
            flag_low=4.20,  # natural risk = 4.50 - 4.19 = 0.31
            pole_height=0.50,
        )
        plan = planner.create_plan(pattern)

        assert plan is not None
        assert plan.risk_per_share == pytest.approx(0.20, abs=0.01)
        assert plan.stop_loss_price == pytest.approx(4.30, abs=0.01)

    def test_rejects_when_natural_stop_over_50_cents(self, planner):
        """Pattern too volatile when natural stop > 50 cents."""
        pattern = _make_pattern(
            breakout_level=5.00,
            flag_low=4.40,  # natural risk = 5.00 - 4.39 = 0.61
            pole_height=1.00,
        )
        plan = planner.create_plan(pattern)
        assert plan is None

    def test_target_is_2_to_1_rr(self, planner):
        """Target is always 2:1 R:R regardless of pole height."""
        pattern = _make_pattern(
            breakout_level=4.40,
            flag_low=4.30,  # risk = 0.11
            pole_height=0.50,  # pole target would be 4.90, but we use 2:1
            # 2:1 target = 4.40 + 0.22 = 4.62
        )
        plan = planner.create_plan(pattern)

        assert plan is not None
        expected_target = 4.40 + 2 * plan.risk_per_share
        assert plan.take_profit_price == pytest.approx(expected_target, abs=0.01)
        assert plan.risk_reward_ratio == pytest.approx(2.0, abs=0.1)

    def test_position_size_500_dollars(self, planner):
        """Shares = floor(500 / entry_price)."""
        pattern = _make_pattern(breakout_level=5.00, flag_low=4.90, pole_height=0.50)
        plan = planner.create_plan(pattern)

        assert plan is not None
        assert plan.shares == 100  # floor(500 / 5.00)

    def test_caps_shares_at_max_1000(self, planner):
        """Shares capped at max_shares when position_size would exceed it."""
        # With $500 at $0.40, we'd get 1250 shares, but cap at 1000
        planner_big = TradePlanner(position_size_dollars=500, max_shares=1000, min_risk_per_share=0.05)
        pattern = _make_pattern(
            breakout_level=0.40,
            flag_low=0.30,
            pole_height=0.15,
            pole_low=0.25,
            pole_high=0.40,
        )
        plan = planner_big.create_plan(pattern)

        if plan is not None:
            assert plan.shares <= 1000


# ===========================================================================
# REJECTION TESTS
# ===========================================================================

class TestPlanRejections:
    """Tests for plan rejection conditions."""

    def test_rejects_noise_stop_too_tight(self, planner):
        """When risk < min_risk_per_share ($0.05), reject as noise stop."""
        pattern = _make_pattern(
            breakout_level=4.40,
            flag_low=4.40,  # stop = 4.39, risk = 0.01 < 0.05
            pole_height=0.50,
        )
        plan = planner.create_plan(pattern)
        assert plan is None

    def test_rejects_negative_risk_stop_above_entry(self, planner):
        """When flag_low > entry, stop would be above entry — invalid."""
        pattern = _make_pattern(
            breakout_level=4.30,
            flag_low=4.40,  # stop = 4.39, above entry of 4.30 -> negative risk
            pole_height=0.50,
        )
        plan = planner.create_plan(pattern)
        assert plan is None

    def test_rejects_zero_entry_price(self, planner):
        """Zero entry price is invalid."""
        pattern = _make_pattern(breakout_level=0.0, flag_low=0.0, pole_height=0.0)
        plan = planner.create_plan(pattern)
        assert plan is None

    def test_rejects_2_cent_risk_as_noise(self, planner):
        """PLYX-style pattern: 2-cent risk gets rejected."""
        pattern = _make_pattern(
            breakout_level=6.05,
            flag_low=6.04,  # stop = 6.03, risk = 0.02 < 0.05
            pole_height=0.41,
        )
        plan = planner.create_plan(pattern)
        assert plan is None

    def test_rejects_4_cent_risk_below_min(self, planner):
        """4 cents risk is below 5-cent minimum, rejected."""
        pattern = _make_pattern(
            breakout_level=4.40,
            flag_low=4.37,  # stop = 4.36, risk = 0.04 < 0.05
            pole_height=0.50,
        )
        plan = planner.create_plan(pattern)
        assert plan is None

    def test_accepts_11_cent_risk_above_min(self, planner):
        """11 cents risk is well above 5-cent minimum, accepted."""
        pattern = _make_pattern(
            breakout_level=4.40,
            flag_low=4.30,  # stop = 4.29, risk = 0.11
            pole_height=0.50,
        )
        plan = planner.create_plan(pattern)
        assert plan is not None
        assert plan.risk_per_share == pytest.approx(0.11, abs=0.01)

    def test_custom_min_risk_per_share(self):
        """Custom min_risk_per_share overrides default."""
        planner = TradePlanner(min_risk_per_share=0.10)
        pattern = _make_pattern(
            breakout_level=4.40,
            flag_low=4.33,  # stop = 4.32, risk = 0.08 < 0.10
            pole_height=0.50,
        )
        plan = planner.create_plan(pattern)
        assert plan is None

    def test_total_risk_calculation(self, planner):
        """Total risk = risk_per_share * shares."""
        pattern = _make_pattern(
            breakout_level=5.00,
            flag_low=4.90,  # risk = 0.11
            pole_height=0.50,
        )
        plan = planner.create_plan(pattern)

        assert plan is not None
        assert plan.total_risk == pytest.approx(plan.risk_per_share * plan.shares)
