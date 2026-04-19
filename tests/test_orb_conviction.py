"""Unit tests for trading/orb_conviction.py."""
import pytest

from trading.orb_conviction import (
    MIN_MULT, DEFAULT_MAX_MULT, Q5_MAX_MULT,
    load_adaptive_mults, apply_adaptive_mult,
    compute_adaptive_mults_from_averages,
)


# =========================================================================
# load_adaptive_mults
# =========================================================================

def test_load_adaptive_mults_happy_path():
    raw = {'Q1': 0.83, 'Q2': 0.27, 'Q3': 0.50, 'Q4': 0.95, 'Q5': 1.5}
    mults = load_adaptive_mults(raw)
    assert mults == {'Q1': 0.83, 'Q2': 0.27, 'Q3': 0.50, 'Q4': 0.95, 'Q5': 1.5}


def test_load_adaptive_mults_missing_quintile():
    raw = {'Q1': 0.83, 'Q2': 0.27, 'Q3': 0.50, 'Q4': 0.95}  # no Q5
    with pytest.raises(ValueError, match="missing quintile 'Q5'"):
        load_adaptive_mults(raw)


def test_load_adaptive_mults_below_min():
    raw = {'Q1': 0.10, 'Q2': 0.5, 'Q3': 0.5, 'Q4': 0.5, 'Q5': 0.5}
    with pytest.raises(ValueError, match="< min"):
        load_adaptive_mults(raw)


def test_load_adaptive_mults_q5_exceeds_cap():
    """Q5 cap at 1.5 is the anti-overfit guard — should reject 2.0."""
    raw = {'Q1': 1, 'Q2': 1, 'Q3': 1, 'Q4': 1, 'Q5': 2.0}
    with pytest.raises(ValueError, match="Q5 is capped at 1.5"):
        load_adaptive_mults(raw)


def test_load_adaptive_mults_q4_exceeds_cap():
    raw = {'Q1': 1, 'Q2': 1, 'Q3': 1, 'Q4': 3.5, 'Q5': 1.0}
    with pytest.raises(ValueError, match="> cap 3.0"):
        load_adaptive_mults(raw)


def test_load_adaptive_mults_q5_at_cap_ok():
    raw = {'Q1': 1, 'Q2': 1, 'Q3': 1, 'Q4': 1, 'Q5': Q5_MAX_MULT}
    # Exactly at cap should be accepted
    mults = load_adaptive_mults(raw)
    assert mults['Q5'] == Q5_MAX_MULT


def test_load_adaptive_mults_q1_at_min_ok():
    raw = {'Q1': MIN_MULT, 'Q2': 1, 'Q3': 1, 'Q4': 1, 'Q5': 1}
    mults = load_adaptive_mults(raw)
    assert mults['Q1'] == MIN_MULT


def test_load_adaptive_mults_invalid_type():
    with pytest.raises(ValueError, match="must be a dict"):
        load_adaptive_mults([0.5, 1.0, 1.5])  # list not dict


def test_load_adaptive_mults_h1_2025_values():
    """Smoke test with actual orb.yaml values."""
    raw = {'Q1': 0.830, 'Q2': 0.266, 'Q3': 0.496, 'Q4': 0.946, 'Q5': 1.500}
    mults = load_adaptive_mults(raw)
    assert mults['Q4'] == 0.946
    assert mults['Q5'] == 1.500


# =========================================================================
# apply_adaptive_mult
# =========================================================================

def test_apply_adaptive_mult_lookup():
    mults = {'Q1': 0.5, 'Q2': 0.7, 'Q3': 1.0, 'Q4': 1.2, 'Q5': 1.5}
    assert apply_adaptive_mult('Q4', mults) == 1.2


def test_apply_adaptive_mult_missing_quintile_fallback():
    """If somehow a quintile missing, fallback to 1.0 (safety)."""
    mults = {'Q1': 0.5, 'Q2': 0.7, 'Q3': 1.0, 'Q4': 1.2}  # no Q5
    assert apply_adaptive_mult('Q5', mults) == 1.0


def test_apply_adaptive_mult_invalid_label():
    with pytest.raises(ValueError, match="Invalid quintile"):
        apply_adaptive_mult('Q6', {})
    with pytest.raises(ValueError, match="Invalid quintile"):
        apply_adaptive_mult('q1', {})  # lowercase not accepted


# =========================================================================
# compute_adaptive_mults_from_averages (refit formula)
# =========================================================================

def test_compute_from_averages_simple():
    # Overall avg=100, Q5 avg=200 → raw ratio 2.0 → clipped to Q5_MAX_MULT=1.5
    # Other quintiles: avg=100 → ratio 1.0 → mult 1.0
    q_avgs = {'Q1': 100, 'Q2': 100, 'Q3': 100, 'Q4': 100, 'Q5': 200}
    mults = compute_adaptive_mults_from_averages(q_avgs, overall_avg=100.0)
    assert mults['Q1'] == 1.0
    assert mults['Q5'] == Q5_MAX_MULT  # capped from 2.0


def test_compute_from_averages_applies_min_floor():
    # Q1 avg 10 / overall 100 = 0.1 → clipped up to MIN_MULT=0.25
    q_avgs = {'Q1': 10, 'Q2': 100, 'Q3': 100, 'Q4': 100, 'Q5': 100}
    mults = compute_adaptive_mults_from_averages(q_avgs, overall_avg=100.0)
    assert mults['Q1'] == MIN_MULT


def test_compute_from_averages_q4_cap_at_3x():
    # Q4 avg 400 / overall 100 = 4.0 → clipped to DEFAULT_MAX_MULT=3.0
    q_avgs = {'Q1': 100, 'Q2': 100, 'Q3': 100, 'Q4': 400, 'Q5': 100}
    mults = compute_adaptive_mults_from_averages(q_avgs, overall_avg=100.0)
    assert mults['Q4'] == DEFAULT_MAX_MULT


def test_compute_from_averages_negative_overall_rejects():
    """If the strategy is net-negative, refit is meaningless."""
    q_avgs = {'Q1': -100, 'Q2': -100, 'Q3': -100, 'Q4': -100, 'Q5': -100}
    with pytest.raises(ValueError, match="do not refit"):
        compute_adaptive_mults_from_averages(q_avgs, overall_avg=-10.0)


def test_compute_from_averages_zero_overall_rejects():
    q_avgs = {'Q1': 10, 'Q2': 10, 'Q3': 10, 'Q4': 10, 'Q5': 10}
    with pytest.raises(ValueError, match="must be > 0"):
        compute_adaptive_mults_from_averages(q_avgs, overall_avg=0.0)


def test_compute_from_averages_missing_quintile():
    q_avgs = {'Q1': 100, 'Q2': 100, 'Q3': 100, 'Q4': 100}
    with pytest.raises(ValueError, match="missing 'Q5'"):
        compute_adaptive_mults_from_averages(q_avgs, overall_avg=100.0)


def test_compute_from_averages_matches_h1_2025_fit():
    """Reproduce the actual orb.yaml fit from H1 2025 TRAIN stats."""
    # From Bash script output earlier:
    q_avgs = {'Q1': 102.81, 'Q2': 32.98, 'Q3': 61.43, 'Q4': 117.29, 'Q5': 305.14}
    overall = 123.93
    mults = compute_adaptive_mults_from_averages(q_avgs, overall_avg=overall)
    # Q1-Q4 under cap, Q5 above 1.5 → capped
    assert mults['Q1'] == pytest.approx(0.830, abs=0.01)
    assert mults['Q2'] == pytest.approx(0.266, abs=0.01)
    assert mults['Q3'] == pytest.approx(0.496, abs=0.01)
    assert mults['Q4'] == pytest.approx(0.946, abs=0.01)
    assert mults['Q5'] == Q5_MAX_MULT  # would be 2.462, clipped
