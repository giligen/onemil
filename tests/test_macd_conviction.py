"""Unit tests for the shared MACD wave conviction scoring module.

Guards the V4 formula and boundary semantics that both BT and PROD depend on.
See trading/macd_conviction.py for research provenance.
"""
import pytest

from trading.macd_conviction import compute_conviction_score


class TestScoreTopAndBottomTiers:
    """Edge-of-range sanity — max and min scores."""

    def test_score_both_top_tier(self):
        """Fast cross + tiny vol → score = 1.8 (max, both rules fire top-tier)."""
        score, brk = compute_conviction_score(cross_time_min=1, vol_at_cross=10_000)
        assert score == pytest.approx(1.8, abs=1e-9)
        assert brk['cross_speed'] == pytest.approx(0.4, abs=1e-9)
        assert brk['vol_at_cross'] == pytest.approx(0.4, abs=1e-9)
        assert brk['final_score'] == pytest.approx(1.8, abs=1e-9)

    def test_score_both_bottom_tier(self):
        """Slow cross + huge vol → score = 1.0 (no rule fires)."""
        score, brk = compute_conviction_score(cross_time_min=10, vol_at_cross=300_000)
        assert score == pytest.approx(1.0, abs=1e-9)
        assert brk['cross_speed'] == 0.0
        assert brk['vol_at_cross'] == 0.0
        assert brk['final_score'] == pytest.approx(1.0, abs=1e-9)


class TestTierBoundaries:
    """Boundary semantics — tier edges use `<=`, not `<`."""

    def test_cross_time_tier1_upper_boundary_is_3(self):
        """cross_time_min = 3 → still top-tier (+0.4)."""
        _, brk = compute_conviction_score(cross_time_min=3, vol_at_cross=500_000)
        assert brk['cross_speed'] == pytest.approx(0.4)

    def test_cross_time_tier2_upper_boundary_is_5(self):
        """cross_time_min = 5 → second-tier (+0.2), not top."""
        _, brk = compute_conviction_score(cross_time_min=5, vol_at_cross=500_000)
        assert brk['cross_speed'] == pytest.approx(0.2)

    def test_cross_time_tier3_upper_boundary_is_7(self):
        """cross_time_min = 7 → third-tier (+0.1), not second."""
        _, brk = compute_conviction_score(cross_time_min=7, vol_at_cross=500_000)
        assert brk['cross_speed'] == pytest.approx(0.1)

    def test_cross_time_8_is_zero(self):
        """cross_time_min = 8 → no contribution (outside all 3 tiers)."""
        _, brk = compute_conviction_score(cross_time_min=8, vol_at_cross=500_000)
        assert brk['cross_speed'] == 0.0

    def test_vol_tier1_upper_boundary_is_27k(self):
        """vol_at_cross = 27_000 → top-tier (+0.4)."""
        _, brk = compute_conviction_score(cross_time_min=100, vol_at_cross=27_000)
        assert brk['vol_at_cross'] == pytest.approx(0.4)

    def test_vol_tier2_upper_boundary_is_79k(self):
        """vol_at_cross = 79_000 → second-tier (+0.2)."""
        _, brk = compute_conviction_score(cross_time_min=100, vol_at_cross=79_000)
        assert brk['vol_at_cross'] == pytest.approx(0.2)

    def test_vol_tier3_upper_boundary_is_165k(self):
        """vol_at_cross = 165_000 → third-tier (+0.1)."""
        _, brk = compute_conviction_score(cross_time_min=100, vol_at_cross=165_000)
        assert brk['vol_at_cross'] == pytest.approx(0.1)

    def test_vol_over_165k_is_zero(self):
        """vol_at_cross = 165_001 → no contribution."""
        _, brk = compute_conviction_score(cross_time_min=100, vol_at_cross=165_001)
        assert brk['vol_at_cross'] == 0.0


class TestMixedTiers:
    """Mixed signals — one rule fires, the other doesn't."""

    def test_cross_top_vol_bottom(self):
        """Top-tier cross + out-of-range vol → 1.0 + 0.4 + 0 = 1.4."""
        score, brk = compute_conviction_score(cross_time_min=3, vol_at_cross=300_000)
        assert score == pytest.approx(1.4)
        assert brk['cross_speed'] == pytest.approx(0.4)
        assert brk['vol_at_cross'] == 0.0

    def test_cross_bottom_vol_top(self):
        """Out-of-range cross + top-tier vol → 1.0 + 0 + 0.4 = 1.4."""
        score, brk = compute_conviction_score(cross_time_min=10, vol_at_cross=10_000)
        assert score == pytest.approx(1.4)
        assert brk['cross_speed'] == 0.0
        assert brk['vol_at_cross'] == pytest.approx(0.4)

    def test_cross_third_vol_second(self):
        """Third-tier cross (+0.1) + second-tier vol (+0.2) → 1.3."""
        score, _ = compute_conviction_score(cross_time_min=7, vol_at_cross=79_000)
        assert score == pytest.approx(1.3)


class TestBreakdownStructure:
    """Breakdown dict must have all expected keys for downstream logging."""

    def test_breakdown_dict_has_all_keys(self):
        """Emitted breakdown must contain per-rule contribs + raw + final."""
        _, brk = compute_conviction_score(cross_time_min=3, vol_at_cross=27_000)
        assert set(brk.keys()) == {'cross_speed', 'vol_at_cross', 'raw_score', 'final_score'}

    def test_raw_and_final_agree_when_within_range(self):
        """When score is inside [0.5, 2.0], raw == final."""
        _, brk = compute_conviction_score(cross_time_min=5, vol_at_cross=79_000)
        assert brk['raw_score'] == pytest.approx(brk['final_score'])
        assert brk['raw_score'] == pytest.approx(1.4)  # 1.0 + 0.2 + 0.2


class TestClampingSafety:
    """Safety clamp [0.5, 2.0] protects against future formula bugs."""

    def test_score_never_below_floor(self):
        """Even with no rules firing, score floors at 1.0 (naturally above 0.5)."""
        score, _ = compute_conviction_score(cross_time_min=999, vol_at_cross=999_999_999)
        assert score >= 0.5
        assert score == 1.0

    def test_score_never_above_ceiling_on_current_rules(self):
        """Current max (both top tier) = 1.8, below the 2.0 ceiling — not clamped."""
        score, brk = compute_conviction_score(cross_time_min=1, vol_at_cross=1)
        assert score <= 2.0
        assert brk['raw_score'] == pytest.approx(1.8)


class TestBTProdParity:
    """Smoke test — both entry points should import and produce identical outputs.

    The BT imports from this module as its single source of truth.
    If this test breaks, the import chain is broken somewhere.
    """

    def test_bt_imports_same_function(self):
        """macd_wave_backtest exposes compute_conviction_score via the import."""
        from macd_wave_backtest import compute_conviction_score as bt_csc
        from trading.macd_conviction import compute_conviction_score as shared_csc
        # Same object — single source of truth
        assert bt_csc is shared_csc

    def test_outputs_match_across_scenarios(self):
        """Call both paths with same inputs; outputs must be identical."""
        from macd_wave_backtest import compute_conviction_score as bt_csc
        from trading.macd_conviction import compute_conviction_score as shared_csc
        scenarios = [(1, 10_000), (3, 27_000), (5, 79_000), (7, 165_000), (10, 300_000)]
        for ct, v in scenarios:
            bt_out = bt_csc(ct, v)
            shared_out = shared_csc(ct, v)
            assert bt_out == shared_out, f"Drift at ({ct}, {v}): bt={bt_out} shared={shared_out}"
