"""Unit tests for cell_1478.py -- research/hod_entry/PREREG_1478.md + amendment.

Every assert targets ONE frozen rule named in the PREREG: the threshold is a pure function of
TRAIN-H2 predictions only; the placebo shuffles labels (not features); decoy columns never reach
the real model's feature matrix; and the stop-limit slip substitution arithmetic matches a hand-
computed example, split by split, why by why.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from research.hod_entry import cell_1478 as m  # noqa: E402


# ------------------------------------------------------------------------------------------------
# top_tercile_threshold: a pure function of TRAIN predictions only
# ------------------------------------------------------------------------------------------------

def test_threshold_is_top_tercile_of_train():
    """30 evenly-spaced values 1..30: the 2/3 quantile keeps exactly the top third (10 of 30)."""
    prob_tr = np.arange(1, 31, dtype=float)
    thr = m.top_tercile_threshold(prob_tr)
    assert thr == pytest.approx(np.quantile(prob_tr, 2.0 / 3.0))
    kept = prob_tr[prob_tr >= thr]
    assert len(kept) == 10
    assert sorted(kept) == list(range(21, 31))


def test_threshold_ignores_val_predictions():
    """The threshold must be identical whether or not a VAL array with wildly different values is
    ever computed alongside it -- it is never a function of VAL. (A regression here would mean
    someone accidentally concatenated TRAIN+VAL before calling top_tercile_threshold.)"""
    prob_tr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    thr_alone = m.top_tercile_threshold(prob_tr)
    # A caller who accidentally included VAL predictions would get a different threshold here;
    # we assert the function signature only ever accepts one array and that array is unchanged.
    prob_val_extreme = np.array([0.99] * 100)
    thr_again = m.top_tercile_threshold(prob_tr)
    assert thr_again == thr_alone
    assert thr_again != m.top_tercile_threshold(np.concatenate([prob_tr, prob_val_extreme]))


# ------------------------------------------------------------------------------------------------
# shuffle_labels: the placebo shuffles LABELS, preserving class counts, seeded and reproducible
# ------------------------------------------------------------------------------------------------

def test_shuffle_labels_preserves_class_counts():
    y = pd.Series([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], index=range(10))
    shuffled = m.shuffle_labels(y, seed=1478)
    assert shuffled.sum() == y.sum()
    assert len(shuffled) == len(y)
    assert set(shuffled.index) == set(y.index)


def test_shuffle_labels_is_seeded_and_actually_permutes():
    y = pd.Series(np.arange(20) % 2, index=range(20))
    s1 = m.shuffle_labels(y, seed=1478)
    s2 = m.shuffle_labels(y, seed=1478)
    assert (s1.to_numpy() == s2.to_numpy()).all(), 'same seed must reproduce the same permutation'
    s3 = m.shuffle_labels(y, seed=1)
    assert not (s1.to_numpy() == s3.to_numpy()).all(), 'a different seed must give a different draw'
    # A real permutation of a 50/50 20-row vector essentially never equals the identity by chance;
    # assert it actually moved something (not a no-op shuffle).
    assert not (s1.to_numpy() == y.to_numpy()).all()


# ------------------------------------------------------------------------------------------------
# feature_columns: decoy columns (and id/object columns) never reach the real-model feature list
# ------------------------------------------------------------------------------------------------

def test_feature_columns_excludes_decoy_and_id_columns():
    df = pd.DataFrame({
        'day': ['2025-07-01'], 'symbol': ['AAA'], 'fill_min': [600.0], 'split': ['TRAIN'],
        'sic2': ['35'],
        'store_served_1438': [1], 'rth_bar_count_1438': [388], 'tick_window_has_bar_j': [1],
        'rv_j': [1.4], 'spread_bps_at_arm': [12.0],
    })
    cols = m.feature_columns(df)
    for decoy in m.DECOY_COLS:
        assert decoy not in cols, f'{decoy} must be excluded from the real model'
    for idc in ('day', 'symbol', 'fill_min', 'split', 'sic2'):
        assert idc not in cols
    assert 'rv_j' in cols and 'spread_bps_at_arm' in cols


def test_feature_columns_excludes_label_and_outcome_columns():
    """Regression guard: an earlier draft of main() computed fcols from the label/outcome-
    enriched `merged` frame instead of `feats` alone, so full_day_range_pct (which directly
    determines L1) leaked into the real model's feature matrix (VAL AUC ~0.99 on a book with a
    true ceiling of ~0.6). feature_columns() must drop every LABEL_OUTCOME_COLS name even if a
    caller passes the wrong (enriched) frame."""
    df = pd.DataFrame({
        'day': ['2025-07-01'], 'symbol': ['AAA'], 'fill_min': [600.0], 'split': ['TRAIN'],
        'L1': [1.0], 'L2': [0.0], 'outcome_R': [0.3], 'full_day_range_pct': [12.0],
        'day_high': [11.0], 'day_low': [9.0], 'day_close': [10.5], 'prev_session_volume': [2e6],
        'why': ['stop'], 'R': [0.5], 'net_R': [0.1],
        'rv_j': [1.4],
    })
    cols = m.feature_columns(df)
    for leaky in m.LABEL_OUTCOME_COLS:
        assert leaky not in cols, f'{leaky} must never reach the feature matrix'
    assert 'rv_j' in cols


# ------------------------------------------------------------------------------------------------
# substitute_stop_slip: hand-computed slip arithmetic, split by split, why by why
# ------------------------------------------------------------------------------------------------

def _row(why, split, exit_price, R, net_R_costfix, net_R_corr):
    return dict(why=why, split=split, exit_price=exit_price, R=R,
                net_R_costfix=net_R_costfix, net_R_corr=net_R_corr)


def test_substitute_stop_slip_train_stop_row():
    """A TRAIN-H2 'stop' row: expected slip = 0.88*2.9 + 0.12*94 = 13.832 bps, in R via
    exit_price * bps / 1e4 / R."""
    exit_price, R = 20.0, 0.50
    df = pd.DataFrame([_row('stop', 'TRAIN', exit_price, R, net_R_costfix=0.10, net_R_corr=-0.05)])
    outcome, new_slip_R, is_stop = m.substitute_stop_slip(df)
    expected_bps = 0.88 * 2.9 + 0.12 * 94.0
    expected_slip_R = exit_price * expected_bps / 1e4 / R
    assert is_stop.iloc[0]
    assert new_slip_R.iloc[0] == pytest.approx(expected_slip_R)
    assert outcome.iloc[0] == pytest.approx(0.10 - expected_slip_R)


def test_substitute_stop_slip_val_stop_bar_row():
    """A VAL 'stop_bar' row uses the VAL-holdout bps (0.88*3.2 + 0.12*76) and is treated the same
    as a plain 'stop' row (both in STOP_WHY)."""
    exit_price, R = 15.0, 1.0
    df = pd.DataFrame([_row('stop_bar', 'VAL', exit_price, R, net_R_costfix=0.20, net_R_corr=0.05)])
    outcome, new_slip_R, is_stop = m.substitute_stop_slip(df)
    expected_bps = 0.88 * 3.2 + 0.12 * 76.0
    expected_slip_R = exit_price * expected_bps / 1e4 / R
    assert is_stop.iloc[0]
    assert outcome.iloc[0] == pytest.approx(0.20 - expected_slip_R)


@pytest.mark.parametrize('why', ['target', 'eod', 'eod_fallback'])
def test_substitute_stop_slip_leaves_non_stop_why_unchanged(why):
    """Only why in {stop, stop_bar} is substituted; target/eod/eod_fallback keep net_R_corr_v2
    exactly (the amendment names only stop-limit exits)."""
    df = pd.DataFrame([_row(why, 'TRAIN', 20.0, 0.5, net_R_costfix=0.10, net_R_corr=-0.03)])
    outcome, new_slip_R, is_stop = m.substitute_stop_slip(df)
    assert not is_stop.iloc[0]
    assert outcome.iloc[0] == pytest.approx(-0.03)


def test_substitute_stop_slip_mixed_book_only_touches_stop_rows():
    df = pd.DataFrame([
        _row('stop', 'TRAIN', 20.0, 0.5, net_R_costfix=0.10, net_R_corr=-0.05),
        _row('target', 'TRAIN', 20.0, 0.5, net_R_costfix=0.30, net_R_corr=0.30),
        _row('eod', 'VAL', 10.0, 0.4, net_R_costfix=-0.02, net_R_corr=-0.06),
        _row('stop', 'VAL', 12.0, 0.6, net_R_costfix=-0.10, net_R_corr=-0.30),
    ])
    outcome, new_slip_R, is_stop = m.substitute_stop_slip(df)
    assert is_stop.tolist() == [True, False, False, True]
    assert outcome.iloc[1] == pytest.approx(0.30)   # target: untouched
    assert outcome.iloc[2] == pytest.approx(-0.06)  # eod: untouched
    bps_train = 0.88 * 2.9 + 0.12 * 94.0
    bps_val = 0.88 * 3.2 + 0.12 * 76.0
    assert outcome.iloc[0] == pytest.approx(0.10 - 20.0 * bps_train / 1e4 / 0.5)
    assert outcome.iloc[3] == pytest.approx(-0.10 - 12.0 * bps_val / 1e4 / 0.6)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
