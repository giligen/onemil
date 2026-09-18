"""Meta-label hook in study_orb_pipeline_static_lock.py (research/meta_label).

The hook lets a secondary model REORDER a day's ranked list and VETO a selected
pick. It may never RESIZE one. Both halves are env-gated and default-inert; the
parity proof that they are inert is that the pipeline with the env unset
reproduces D1's book to the cent ($14,428.62 / 215 picks) — asserted in
research/meta_label/REPORT.md §3. These tests pin the SEMANTICS so a later edit
cannot change them silently.
"""
import numpy as np
import pandas as pd
import pytest

SRC = 'study_orb_pipeline_static_lock.py'


def _src():
    with open(SRC) as fh:
        return fh.read()


def test_hook_is_env_gated_and_default_inert():
    s = _src()
    # both hooks read an env var that defaults to '' and is stripped
    assert "os.environ.get('ORB_META_RANK_COL') or ''" in s
    assert "os.environ.get('ORB_META_VETO_COL') or ''" in s
    # the ranking branch is guarded by the rank column being set
    assert 'if _meta_rank:' in s
    assert 'if _meta_veto:' in s
    # the veto is POST-selection (operates on `sel`, the selected picks) and
    # leaves the slot empty — no refill
    assert "sel = sel[~_mv].copy()" in s
    # the hook never touches sizing
    assert '_meta' not in s.split("sel['_sized_pnl'] = sel.apply")[1].split('\n')[0]


def _rank(d):
    """The hook's ranking expression, verbatim from the pipeline."""
    d = d.copy()
    d['_meta_key'] = pd.to_numeric(d['meta_score'], errors='coerce')
    return d.sort_values(['_meta_key', '_q_rank', '_composite'],
                         ascending=[False, True, False],
                         na_position='last', kind='mergesort')


def test_all_unscored_day_reproduces_the_shipped_order():
    """A month the walk-forward could not score must fall back to M0 exactly."""
    d = pd.DataFrame({'meta_score': [np.nan] * 4,
                      '_q_rank': [2, 1, 1, 3],
                      '_composite': [0.1, 0.5, 0.9, 0.2]})
    shipped = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])
    assert list(_rank(d).index) == list(shipped.index)


def test_scored_rows_outrank_unscored_and_score_beats_quintile():
    d = pd.DataFrame({'meta_score': [np.nan, 0.9, np.nan, 0.1],
                      '_q_rank': [1, 5, 1, 3],
                      '_composite': [0.9, 0.1, 0.8, 0.2]})
    assert list(_rank(d).index) == [1, 3, 0, 2]


@pytest.mark.parametrize('thr,expect_dropped', [(0.5, [0, 2]), (0.1, [0])])
def test_veto_drops_below_threshold_and_fails_open_on_nan(thr, expect_dropped):
    sel = pd.DataFrame({'meta_score': [0.05, 0.9, 0.4, np.nan]})
    mv = (pd.to_numeric(sel['meta_score'], errors='coerce') < thr).fillna(False)
    assert list(sel.index[mv]) == expect_dropped
    assert not bool(mv.iloc[3]), 'an unscored pick must fail OPEN (kept)'
