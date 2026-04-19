"""Unit tests for trading/orb_planner.py."""
import pytest

from trading.orb_planner import (
    OrbTradePlanner, OrbTradePlan, PlannerReject,
    SKIP_SPREAD_GATE, SKIP_ZERO_RANGE, SKIP_TOO_SMALL,
)


@pytest.fixture
def default_cfg():
    """Minimal cfg matching orb.yaml structure."""
    return {
        'entry': {
            'entry_slip_bps': 30,
            'max_spread_bps': 150,
        },
        'exit': {
            'lock_arm_at_r': 1.5,
            'lock_stop_r': 1.0,
        },
        'sizing': {
            'account_budget_usd': 100_000,
            'max_concurrent': 4,
            'risk_per_trade_usd': 3_000,
            'min_stop_pct': 1.0,
        },
    }


@pytest.fixture
def planner(default_cfg):
    return OrbTradePlanner(default_cfg)


# =========================================================================
# Init validation
# =========================================================================

class TestInit:
    def test_default_init_computes_per_pos_cap(self, planner):
        # $100K / 4 = $25K
        assert planner.per_pos_cap_usd == 25_000

    def test_invalid_max_concurrent(self, default_cfg):
        default_cfg['sizing']['max_concurrent'] = 0
        with pytest.raises(ValueError, match="max_concurrent"):
            OrbTradePlanner(default_cfg)

    def test_invalid_risk_per_trade(self, default_cfg):
        default_cfg['sizing']['risk_per_trade_usd'] = -100
        with pytest.raises(ValueError, match="risk_per_trade"):
            OrbTradePlanner(default_cfg)


# =========================================================================
# Gate: spread
# =========================================================================

class TestSpreadGate:
    def test_spread_within_gate_passes(self, planner):
        p = planner.build(
            symbol='TSLA', range_high=100.0, range_low=95.0,
            composite_score=0.5, quintile='Q4', adaptive_mult=0.95,
            spread_bps=100.0,  # under 150
        )
        assert isinstance(p, OrbTradePlan)

    def test_spread_exceeds_gate_rejects(self, planner):
        r = planner.build(
            symbol='TSLA', range_high=100.0, range_low=95.0,
            composite_score=0.5, quintile='Q4', adaptive_mult=0.95,
            spread_bps=200.0,  # > 150
        )
        assert isinstance(r, PlannerReject)
        assert r.reason == SKIP_SPREAD_GATE
        assert r.details['spread_bps'] == 200.0

    def test_spread_exactly_at_gate_passes(self, planner):
        r = planner.build(
            symbol='TSLA', range_high=100.0, range_low=95.0,
            composite_score=0.5, quintile='Q4', adaptive_mult=0.95,
            spread_bps=150.0,  # exactly at limit
        )
        # > 150 rejects, so exactly 150 passes
        assert isinstance(r, OrbTradePlan)

    def test_spread_none_skips_gate(self, planner):
        """If spread unknown, don't reject on spread (would break live when quote unavailable)."""
        p = planner.build(
            symbol='TSLA', range_high=100.0, range_low=95.0,
            composite_score=0.5, quintile='Q4', adaptive_mult=0.95,
            spread_bps=None,
        )
        assert isinstance(p, OrbTradePlan)


# =========================================================================
# Gate: zero range
# =========================================================================

class TestZeroRange:
    def test_rejects_zero_range(self, planner):
        r = planner.build(
            symbol='X', range_high=10.0, range_low=10.0,  # identical = zero range
            composite_score=1.0, quintile='Q5', adaptive_mult=1.5,
        )
        assert isinstance(r, PlannerReject)
        assert r.reason == SKIP_ZERO_RANGE

    def test_rejects_inverted_range(self, planner):
        r = planner.build(
            symbol='X', range_high=5.0, range_low=10.0,  # low > high
            composite_score=1.0, quintile='Q4', adaptive_mult=1.0,
        )
        assert isinstance(r, PlannerReject)
        assert r.reason == SKIP_ZERO_RANGE


# =========================================================================
# Sizing math
# =========================================================================

class TestSizingMath:
    def test_entry_price_includes_slippage(self, planner):
        p = planner.build(
            symbol='X', range_high=100.0, range_low=95.0,
            composite_score=0.5, quintile='Q3', adaptive_mult=1.0,
        )
        # 100 × (1 + 30/10000) = 100.30
        assert p.entry_price == pytest.approx(100.30, abs=0.01)

    def test_stop_equals_range_low(self, planner):
        p = planner.build(
            symbol='X', range_high=100.0, range_low=95.0,
            composite_score=0.5, quintile='Q4', adaptive_mult=1.0,
        )
        assert p.stop_price == 95.0

    def test_risk_per_share_is_stop_distance(self, planner):
        p = planner.build(
            symbol='X', range_high=100.0, range_low=95.0,
            composite_score=0.5, quintile='Q4', adaptive_mult=1.0,
        )
        # Entry 100.30, stop 95 → 5.30
        assert p.risk_per_share == pytest.approx(5.30, abs=0.01)

    def test_risk_parity_uncapped_wide_stop(self, planner):
        """Stock with 5% stop: uncapped position = $3K / 5% = $60K → capped to $25K."""
        p = planner.build(
            symbol='X', range_high=100.0, range_low=95.0,  # 5% range
            composite_score=0.5, quintile='Q3', adaptive_mult=1.0,
        )
        # position_before_mult = min($60K, $25K) = $25K
        # adaptive_mult=1.0 → position_dollars = $25K
        assert p.position_dollars == pytest.approx(25_000, abs=100)

    def test_risk_parity_uncapped_narrow_stop(self, planner):
        """Stock with 2% stop: uncapped = $3K / 2% = $150K → capped to $25K.
        Position cap strictly binds on narrow stops."""
        p = planner.build(
            symbol='X', range_high=100.0, range_low=98.0,  # 2% range
            composite_score=0.5, quintile='Q3', adaptive_mult=1.0,
        )
        assert p.position_dollars == pytest.approx(25_000, abs=100)

    def test_risk_parity_uncapped_wide_stop_no_cap(self, planner):
        """Stock with 15% stop: uncapped = $3K / 15% = $20K, below $25K cap."""
        p = planner.build(
            symbol='X', range_high=100.0, range_low=85.0,  # 15% range
            composite_score=0.5, quintile='Q3', adaptive_mult=1.0,
        )
        # Uncapped $20K < $25K cap → no cap
        assert p.position_dollars == pytest.approx(20_000, abs=500)

    def test_min_stop_pct_floors_sizing(self, planner):
        """Stock with 0.5% stop (below 1% floor): sizing uses 1% not 0.5%."""
        p = planner.build(
            symbol='X', range_high=100.0, range_low=99.5,  # 0.5% range
            composite_score=0.5, quintile='Q3', adaptive_mult=1.0,
        )
        # Without floor: uncapped = $3K / 0.5% = $600K → capped to $25K
        # With 1% floor: uncapped = $3K / 1% = $300K → still capped to $25K
        # Either way capped, so this shows floor works WITH cap
        assert p.position_dollars == pytest.approx(25_000, abs=100)
        # Real stop is still range_low, not floored
        assert p.stop_price == 99.5

    def test_adaptive_mult_scales_position(self, planner):
        """Q5 with 1.5x mult on $25K-capped position → effective $37.5K."""
        p = planner.build(
            symbol='X', range_high=100.0, range_low=95.0,  # 5% — will be capped
            composite_score=1.0, quintile='Q5', adaptive_mult=1.5,
        )
        assert p.position_dollars == pytest.approx(37_500, abs=200)
        assert p.adaptive_mult == 1.5
        assert p.quintile == 'Q5'

    def test_adaptive_mult_reduces_position_q2(self, planner):
        """Q2 with 0.266x mult on $25K-capped → $6.65K."""
        p = planner.build(
            symbol='X', range_high=100.0, range_low=95.0,
            composite_score=0.15, quintile='Q2', adaptive_mult=0.266,
        )
        assert p.position_dollars == pytest.approx(6_650, abs=50)


# =========================================================================
# Shares calculation
# =========================================================================

class TestSharesCalc:
    def test_shares_from_position_and_entry(self, planner):
        """Position $25K / entry $100 = 250 shares (floor)."""
        p = planner.build(
            symbol='X', range_high=100.0, range_low=95.0,
            composite_score=0.5, quintile='Q3', adaptive_mult=1.0,
        )
        # Entry = 100.30 → 25000 / 100.30 = 249.25 → floor(249.25) = 249
        assert p.shares == 249

    def test_shares_rounded_down(self, planner):
        """Never round up shares (conservative sizing)."""
        p = planner.build(
            symbol='X', range_high=10.0, range_low=9.5,  # 5% stop
            composite_score=0.5, quintile='Q3', adaptive_mult=1.0,
        )
        # Entry=10.03 (rounded), position=$25K, 25000/10.03=2492.52 → floor 2492
        assert p.shares == 2492

    def test_sub_one_share_rejected(self, planner):
        """Position tiny → shares < 1 → reject."""
        p = planner.build(
            symbol='X', range_high=1000.0, range_low=995.0,  # high-priced, narrow %
            composite_score=0.1, quintile='Q1', adaptive_mult=0.25,  # tiny mult
        )
        # position_dollars very small relative to $1000+ entry → maybe fractional share
        # $25K × 0.25 = $6.25K; 6250 / 1000 = 6 shares → OK, not rejected
        # Construct a case that DOES reject
        p = planner.build(
            symbol='X', range_high=100_000.0, range_low=99_000.0,
            composite_score=0.1, quintile='Q1', adaptive_mult=0.25,
        )
        # $25K × 0.25 = $6.25K position, entry ~$100,300 → 0.0623 shares → floor 0 → reject
        assert isinstance(p, PlannerReject)
        assert p.reason == SKIP_TOO_SMALL

    def test_total_risk_equals_risk_per_share_times_shares(self, planner):
        p = planner.build(
            symbol='X', range_high=100.0, range_low=95.0,
            composite_score=0.5, quintile='Q4', adaptive_mult=0.95,
        )
        assert p.total_risk == pytest.approx(p.risk_per_share * p.shares, abs=0.01)


# =========================================================================
# Plan fields / metadata
# =========================================================================

class TestPlanFields:
    def test_plan_carries_lock_params(self, planner):
        p = planner.build(
            symbol='X', range_high=100.0, range_low=95.0,
            composite_score=0.5, quintile='Q4', adaptive_mult=1.0,
        )
        assert p.lock_arm_at_r == 1.5
        assert p.lock_stop_r == 1.0

    def test_plan_carries_range_size(self, planner):
        p = planner.build(
            symbol='X', range_high=100.0, range_low=95.0,
            composite_score=0.5, quintile='Q4', adaptive_mult=1.0,
        )
        assert p.range_size == pytest.approx(5.0)

    def test_plan_carries_conviction_metadata(self, planner):
        p = planner.build(
            symbol='TSLA', range_high=100.0, range_low=95.0,
            composite_score=0.73, quintile='Q5', adaptive_mult=1.5,
        )
        assert p.composite_score == 0.73
        assert p.quintile == 'Q5'
        assert p.adaptive_mult == 1.5

    def test_plan_symbol_propagates(self, planner):
        p = planner.build(
            symbol='NVDA', range_high=200.0, range_low=195.0,
            composite_score=0.5, quintile='Q3', adaptive_mult=0.5,
        )
        assert p.symbol == 'NVDA'
