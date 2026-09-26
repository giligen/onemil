"""Unit tests for cell_1480.py's pure short-leg logic (amendment stop floor, walk_short, cap
sampling). No network, no DB.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell_1480 as c1480
import cell_1479 as c1479


def bar(m, o, h, l, c):
    return dict(m=m, o=o, h=h, l=l, c=c, v=100)


def test_amendment_short_stop_floor_kicks_in_on_a_doji_break_bar():
    """Amendment item 2: a doji break bar (high ~= level) made the pre-amendment R ~= 0; the 1%
    floor must dominate when brk_high+0.01 would give a near-zero or negative risk."""
    row = pd.Series(dict(day='2025-07-01', symbol='ABC', fill=10.0, stop=9.5, R=0.5, level=10.0,
                          base_why='stop', base_net_R=-0.3, half_entry=0.01, exit_half=0.02,
                          m_candidate=600, brk_high=9.995, print_px=9.98, bid=9.98))
    legs = c1480.price_leg(row, bars_cache={})
    # brk_high+0.01 = 10.005 < entry*1.01 = 10.0798 -> the 1% floor wins
    assert legs['short_stop'] == pytest.approx(9.98 * 1.01, rel=1e-9)
    assert legs['short_stop'] > row.brk_high + 0.01


def test_amendment_floor_does_not_override_a_wide_break_bar():
    """When the break bar's high is genuinely far above the print, brk_high+0.01 should win (the
    floor is a FLOOR, not an override) and stay strictly the max of the two."""
    row = pd.Series(dict(day='2025-07-01', symbol='ABC', fill=10.0, stop=9.5, R=0.5, level=10.0,
                          base_why='stop', base_net_R=-0.3, half_entry=0.01, exit_half=0.02,
                          m_candidate=600, brk_high=11.5, print_px=9.98, bid=9.98))
    legs = c1480.price_leg(row, bars_cache={})
    assert legs['short_stop'] == pytest.approx(max(11.5 + 0.01, 9.98 * 1.01))
    assert legs['short_stop'] == pytest.approx(11.51)


def test_walk_short_stop_priority_when_both_touch_same_bar():
    entry, stop, target = 10.0, 10.5, 9.0
    path = pd.DataFrame([bar(600, 10.0, 10.6, 9.0, 10.5)])   # touches both stop and target in one bar
    m, px, why = c1480.walk_short(entry, stop, target, path)
    assert why == 'stop'


def test_walk_short_gap_through_open_on_stop():
    entry, stop, target = 10.0, 10.5, 9.0
    path = pd.DataFrame([bar(600, 10.8, 11.0, 10.7, 10.9)])  # opens already above the stop
    m, px, why = c1480.walk_short(entry, stop, target, path)
    assert why == 'stop'
    assert px == pytest.approx(10.8)


def test_walk_short_eod_exits_at_open():
    entry, stop, target = 10.0, 10.5, 9.0
    path = pd.DataFrame([bar(c1479.EOD_M, 10.1, 10.2, 10.0, 10.15)])
    m, px, why = c1480.walk_short(entry, stop, target, path)
    assert why == 'eod'
    assert px == pytest.approx(10.1)


def test_cap_candidates_is_a_noop_under_the_real_cap():
    cands = pd.DataFrame({'holdout': ['TRAIN-H2'] * 5 + ['VAL'] * 3})
    out = c1480.cap_candidates(cands)
    assert len(out) == 5 + 3  # under the real 1500 cap nothing is dropped


def test_cap_candidates_actually_caps_when_over_limit():
    cands = pd.DataFrame({'holdout': ['TRAIN-H2'] * 5 + ['VAL'] * 3})
    orig = c1480.FETCH_CAP_PER_HOLDOUT
    c1480.FETCH_CAP_PER_HOLDOUT = 2
    try:
        out = c1480.cap_candidates(cands)
    finally:
        c1480.FETCH_CAP_PER_HOLDOUT = orig
    assert (out.holdout == 'TRAIN-H2').sum() == 2
    assert (out.holdout == 'VAL').sum() == 2


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
