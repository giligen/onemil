"""Unit tests for trading/orb_rvol_tilt.py (sizing.rvol_tilt, owner 2026-10-01,
cell 1,694 Part B — research/orb_freq/RESULT_1694.md)."""
import math

import pytest

from trading.orb_rvol_tilt import (
    DEFAULT_EDGES, DEFAULT_MULTS, TOTAL_MULT_CAP,
    clamp_rvol_tilt_to_cap, resolve_rel_volume_0935, resolve_rvol_tilt_mult,
    tercile_for_rvol,
)


# =========================================================================
# Tercile mapping
# =========================================================================

class TestTercileForRvol:
    def test_below_e1_is_low(self):
        assert tercile_for_rvol(1.0, edges=(3.0, 6.0)) == 'low'

    def test_between_edges_is_mid(self):
        assert tercile_for_rvol(4.0, edges=(3.0, 6.0)) == 'mid'

    def test_above_e2_is_high(self):
        assert tercile_for_rvol(10.0, edges=(3.0, 6.0)) == 'high'

    def test_exactly_e1_is_mid(self):
        """Edges are [e1, e2) / [e2, inf) half-open, matching the research
        tercile_edges binning (research/orb_freq/1693_pool_exits.py)."""
        assert tercile_for_rvol(3.0, edges=(3.0, 6.0)) == 'mid'

    def test_exactly_e2_is_high(self):
        assert tercile_for_rvol(6.0, edges=(3.0, 6.0)) == 'high'

    def test_none_is_none(self):
        assert tercile_for_rvol(None) is None

    def test_nan_is_none(self):
        assert tercile_for_rvol(float('nan')) is None

    def test_non_numeric_is_none(self):
        assert tercile_for_rvol('x') is None

    def test_default_edges_are_live_scale(self):
        """DEFAULT_EDGES are the TRAIN2025 research-ratio tercile edges
        (0.03716, 0.07649) rescaled x78 (SESSION_MINUTES/RANGE_MINUTES) to
        match rel_volume_0935's own scale (trading/orb_addon_gates.py)."""
        e1, e2 = DEFAULT_EDGES
        assert e1 == pytest.approx(0.03716 * 78.0, abs=1e-6)
        assert e2 == pytest.approx(0.07649 * 78.0, abs=1e-6)
        assert 2.5 < e1 < 3.2
        assert 5.5 < e2 < 6.5


# =========================================================================
# Mult resolution
# =========================================================================

class TestResolveRvolTiltMult:
    def test_low_tercile_gets_upsize_mult(self):
        mult, tercile = resolve_rvol_tilt_mult(1.0, edges=(3.0, 6.0), mults=(1.5, 1.0, 0.5))
        assert (mult, tercile) == (1.5, 'low')

    def test_mid_tercile_gets_neutral_mult(self):
        mult, tercile = resolve_rvol_tilt_mult(4.0, edges=(3.0, 6.0), mults=(1.5, 1.0, 0.5))
        assert (mult, tercile) == (1.0, 'mid')

    def test_high_tercile_gets_downsize_mult(self):
        mult, tercile = resolve_rvol_tilt_mult(10.0, edges=(3.0, 6.0), mults=(1.5, 1.0, 0.5))
        assert (mult, tercile) == (0.5, 'high')

    def test_unresolvable_rvol_fails_open(self):
        """Missing RVOL -> (1.0, None): never tilts blind."""
        assert resolve_rvol_tilt_mult(None) == (1.0, None)

    def test_default_mults_match_owner_spec(self):
        assert DEFAULT_MULTS == (1.5, 1.0, 0.5)


# =========================================================================
# rel_volume_0935 formula (parity twin of ORBEngine._build_pool_gate_inputs)
# =========================================================================

class TestResolveRelVolume0935:
    def test_basic_ratio(self):
        # adv20=100_000 -> 5-min expectation = 100_000 * 5/390 = 1282.05
        # range_total_volume=2000 -> ratio ~1.56
        got = resolve_rel_volume_0935(2000.0, 100_000.0)
        expected = 2000.0 / (100_000.0 * 5.0 / 390.0)
        assert got == pytest.approx(expected)

    def test_zero_adv20_is_none(self):
        assert resolve_rel_volume_0935(2000.0, 0.0) is None

    def test_none_adv20_is_none(self):
        assert resolve_rel_volume_0935(2000.0, None) is None

    def test_none_range_volume_is_none(self):
        assert resolve_rel_volume_0935(None, 100_000.0) is None


# =========================================================================
# Total-multiplier cap clamp (the "never-rule")
# =========================================================================

class TestClampRvolTiltToCap:
    def test_disabled_tilt_is_byte_identical_noop(self):
        """tilt_mult=1.0 (disabled, or the hook resolved mid/no tercile) never
        changes the stack, REGARDLESS of how large the pre-existing stack
        already is — this is what makes `enabled: false` byte-identical."""
        effective, clamped = clamp_rvol_tilt_to_cap(3.0, 1.0)
        assert (effective, clamped) == (1.0, False)

    def test_q5_low_tercile_clamps_to_cap(self):
        """Q5 adaptive_mult=1.5 stacked with the low-tercile upsize (1.5x)
        would be 2.25x -- over the 1.5 cap -- so the tilt is reduced to 1.0,
        landing the total EXACTLY at the cap."""
        effective, clamped = clamp_rvol_tilt_to_cap(1.5, 1.5)
        assert clamped is True
        assert effective == pytest.approx(1.0)
        assert 1.5 * effective == pytest.approx(TOTAL_MULT_CAP)

    def test_low_tercile_under_cap_is_unclamped(self):
        """Q3 adaptive_mult=1.0 + low tercile (1.5x) = 1.5 total -- exactly
        at the cap, not over it -- passes through unclamped."""
        effective, clamped = clamp_rvol_tilt_to_cap(1.0, 1.5)
        assert (effective, clamped) == (1.5, False)

    def test_reducing_tilt_never_clamped_even_on_high_pre_existing_stack(self):
        """A high-tercile downsize (0.5x) can only shrink exposure, so it is
        never clamped even when pm_mult x adaptive_mult alone is already
        large (an independent, pre-existing condition this hook does not
        own or correct)."""
        effective, clamped = clamp_rvol_tilt_to_cap(3.0, 0.5)
        assert (effective, clamped) == (0.5, False)

    def test_clamp_never_touches_the_pre_existing_stack(self):
        """The function returns only the tilt factor -- callers must apply
        it on top of (never instead of) adaptive_mult x pm_mult, so those
        two are never adjusted by this hook (CLAUDE.md 'never touching
        adaptive_mults')."""
        stacked_before = 1.5
        effective, clamped = clamp_rvol_tilt_to_cap(stacked_before, 1.5)
        total = stacked_before * effective
        assert total <= TOTAL_MULT_CAP + 1e-9

    def test_zero_stack_returns_tilt_unclamped(self):
        """Degenerate stacked_mult_before_tilt<=0 (should not happen in
        practice) fails safe -- returns the tilt as-is, never divides by 0."""
        effective, clamped = clamp_rvol_tilt_to_cap(0.0, 1.5)
        assert (effective, clamped) == (1.5, False)
