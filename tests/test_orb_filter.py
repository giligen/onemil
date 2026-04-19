"""Unit tests for trading/orb_filter.py."""
import math
import pytest

from trading.orb_filter import (
    FeatureParam, load_feature_params, composite_score, assign_quintile,
)


# =========================================================================
# load_feature_params
# =========================================================================

def test_load_feature_params_basic():
    yaml_dict = {
        'features': {
            'feat_a': {'sign': +1, 'mean': 10.0, 'std': 2.0},
            'feat_b': {'sign': -1, 'mean': 5.0, 'std': 1.5},
        }
    }
    params = load_feature_params(yaml_dict)
    assert set(params.keys()) == {'feat_a', 'feat_b'}
    assert params['feat_a'].sign == 1
    assert params['feat_a'].mean == 10.0
    assert params['feat_a'].std == 2.0
    assert params['feat_b'].sign == -1


def test_load_feature_params_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        load_feature_params({'features': {}})
    with pytest.raises(ValueError, match="empty"):
        load_feature_params({})


def test_load_feature_params_invalid_sign():
    with pytest.raises(ValueError, match="invalid sign"):
        load_feature_params({'features': {'f': {'sign': 0, 'mean': 1, 'std': 1}}})
    with pytest.raises(ValueError, match="invalid sign"):
        load_feature_params({'features': {'f': {'sign': 2, 'mean': 1, 'std': 1}}})


def test_load_feature_params_zero_std():
    with pytest.raises(ValueError, match="std=0"):
        load_feature_params({'features': {'f': {'sign': 1, 'mean': 1, 'std': 0}}})


def test_load_feature_params_negative_std():
    with pytest.raises(ValueError, match="must be > 0"):
        load_feature_params({'features': {'f': {'sign': 1, 'mean': 1, 'std': -1}}})


# =========================================================================
# composite_score
# =========================================================================

def _params(*specs):
    """Build param dict from (name, sign, mean, std) tuples."""
    return {name: FeatureParam(sign=s, mean=m, std=sd) for (name, s, m, sd) in specs}


def test_composite_score_single_feature_at_mean():
    # Value exactly at mean → z = 0 → composite = 0
    params = _params(('x', +1, 10.0, 2.0))
    assert composite_score({'x': 10.0}, params) == pytest.approx(0.0)


def test_composite_score_single_feature_positive_sign():
    # Value +1 std above mean with sign=+1 → composite = +1
    params = _params(('x', +1, 10.0, 2.0))
    assert composite_score({'x': 12.0}, params) == pytest.approx(1.0)


def test_composite_score_single_feature_negative_sign():
    # Value +1 std above mean with sign=-1 (lower is better) → composite = -1
    params = _params(('x', -1, 10.0, 2.0))
    assert composite_score({'x': 12.0}, params) == pytest.approx(-1.0)


def test_composite_score_negative_sign_below_mean():
    # Value -1 std below mean with sign=-1 (lower is better) → composite = +1 (good)
    params = _params(('x', -1, 10.0, 2.0))
    assert composite_score({'x': 8.0}, params) == pytest.approx(1.0)


def test_composite_score_multi_feature_average():
    # Two features: both +2 std, signs mixed → composite = (2 + (-2))/2 = 0
    params = _params(('a', +1, 0, 1), ('b', -1, 0, 1))
    assert composite_score({'a': 2.0, 'b': 2.0}, params) == pytest.approx(0.0)


def test_composite_score_three_features_avg():
    # Three features +1 z each → composite = 1.0
    params = _params(('a', +1, 0, 1), ('b', +1, 0, 1), ('c', +1, 0, 1))
    assert composite_score({'a': 1, 'b': 1, 'c': 1}, params) == pytest.approx(1.0)


def test_composite_score_missing_feature_returns_none():
    params = _params(('a', +1, 0, 1), ('b', +1, 0, 1))
    assert composite_score({'a': 1.0}, params) is None


def test_composite_score_nan_feature_returns_none():
    params = _params(('a', +1, 0, 1))
    assert composite_score({'a': float('nan')}, params) is None


def test_composite_score_empty_params_raises():
    with pytest.raises(ValueError, match="empty params"):
        composite_score({'a': 1.0}, {})


def test_composite_score_matches_validated_values():
    """Sanity check using actual H1 2025 TRAIN params from orb.yaml."""
    params = _params(
        ('gap_pct',                    -1, 187.905162, 1749.630103),
        ('range_total_volume',         -1, 1032481.678899, 2236111.090560),
        ('range_avg_bar_range_pct',    -1, 2.481485, 1.721792),
        ('range_size_pct',             -1, 6.304430, 4.482915),
        ('price_vs_20d_high_pct',      -1, 26.814918, 339.841711),
        ('prev_day_close_position',    -1, 0.488915, 0.303208),
        ('range_close_position',       +1, 0.580629, 0.303179),
    )
    # A "median" candidate should yield a composite near 0
    features = {
        'gap_pct': 187.905162,  # at mean
        'range_total_volume': 1032481.678899,
        'range_avg_bar_range_pct': 2.481485,
        'range_size_pct': 6.304430,
        'price_vs_20d_high_pct': 26.814918,
        'prev_day_close_position': 0.488915,
        'range_close_position': 0.580629,
    }
    assert composite_score(features, params) == pytest.approx(0.0, abs=1e-9)


# =========================================================================
# assign_quintile
# =========================================================================

def test_assign_quintile_below_first_cutoff():
    cutoffs = [0.1, 0.2, 0.3, 0.4]
    assert assign_quintile(-0.5, cutoffs) == 'Q1'
    assert assign_quintile(0.099, cutoffs) == 'Q1'


def test_assign_quintile_in_q2():
    cutoffs = [0.1, 0.2, 0.3, 0.4]
    assert assign_quintile(0.1, cutoffs) == 'Q2'
    assert assign_quintile(0.15, cutoffs) == 'Q2'
    assert assign_quintile(0.199, cutoffs) == 'Q2'


def test_assign_quintile_in_q3():
    cutoffs = [0.1, 0.2, 0.3, 0.4]
    assert assign_quintile(0.2, cutoffs) == 'Q3'
    assert assign_quintile(0.25, cutoffs) == 'Q3'


def test_assign_quintile_in_q4():
    cutoffs = [0.1, 0.2, 0.3, 0.4]
    assert assign_quintile(0.3, cutoffs) == 'Q4'
    assert assign_quintile(0.35, cutoffs) == 'Q4'


def test_assign_quintile_in_q5():
    cutoffs = [0.1, 0.2, 0.3, 0.4]
    assert assign_quintile(0.4, cutoffs) == 'Q5'
    assert assign_quintile(1.0, cutoffs) == 'Q5'


def test_assign_quintile_wrong_cutoffs_length():
    with pytest.raises(ValueError, match="length 4"):
        assign_quintile(0.5, [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="length 4"):
        assign_quintile(0.5, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_assign_quintile_non_ascending_cutoffs():
    with pytest.raises(ValueError, match="ascending"):
        assign_quintile(0.5, [0.1, 0.3, 0.2, 0.4])


def test_assign_quintile_h1_2025_cutoffs():
    """Smoke test with real cutoffs from orb.yaml."""
    cutoffs = [0.1082, 0.1959, 0.2893, 0.4081]
    assert assign_quintile(0.05, cutoffs) == 'Q1'
    assert assign_quintile(0.15, cutoffs) == 'Q2'
    assert assign_quintile(0.25, cutoffs) == 'Q3'
    assert assign_quintile(0.35, cutoffs) == 'Q4'
    assert assign_quintile(0.50, cutoffs) == 'Q5'
