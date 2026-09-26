"""Unit tests for research/hod_entry/cell_1548.py -- PREREG_1548.md mechanics only (no I/O, no DB):
the through-print bid-fill rule (incl. the fill bar itself), the +5% limit target / stop-first walk,
the drawdown measurement stopping at the touch (not counting bars after it), and d/s/W being derived
from TRAIN-only rows.

Run: python3 -m pytest research/hod_entry/test_cell_1548.py -v
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from research.hod_entry import cell_1548 as c1548  # noqa: E402


def bars(rows):
    """rows: list of (m, o, h, l, c) -> the DataFrame shape cell_1548 expects."""
    return pd.DataFrame(rows, columns=['m', 'o', 'h', 'l', 'c'])


# ------------------------------------------------------------------------------------------- SWEEP entry
def test_sweep_fills_on_the_fill_bar_itself_through_print():
    """Through-print rule: the fill bar's own low, if strictly below the limit, fills it -- the
    resting order need not wait for a later bar."""
    level = 100.0
    d, s, W = 1.0, 2.0, 60.0                      # limit=99, stop=98, target=105
    b = bars([(600, 100.0, 100.5, 98.5, 99.8),    # fill bar itself: low 98.5 < limit 99 -> fills here
              (601, 99.8, 106.0, 99.5, 105.5)])   # target touch, would also work if not filled at 600
    row = SimpleNamespace(symbol='AAA', day='2026-01-01', fill_min=600.4, level=level)
    t = c1548.sweep_trade(row, {('AAA', '2026-01-01'): b}, d, s, W)
    assert t['status'] == 'filled'
    assert t['entry_m'] == 600
    assert t['entry_price'] == level * (1 - d / 100.0)


def test_sweep_unfilled_when_no_print_below_limit_in_window():
    level = 100.0
    d, s, W = 1.0, 2.0, 5.0
    b = bars([(600, 100.0, 100.2, 99.5, 100.0), (601, 100.0, 100.2, 99.5, 100.0)])   # low never < 99
    row = SimpleNamespace(symbol='AAA', day='2026-01-01', fill_min=600.0, level=level)
    t = c1548.sweep_trade(row, {('AAA', '2026-01-01'): b}, d, s, W)
    assert t['status'] == 'unfilled'


# ------------------------------------------------------------------------------------------- walk_path (target / stop-first)
def test_walk_path_target_touch():
    from research.hod_entry.sip_rebuild import walk_path
    p = bars([(600, 99.0, 99.5, 98.8, 99.4), (601, 99.4, 105.5, 99.0, 105.0)])
    exit_m, exit_price, why = walk_path(99.0, 98.0, 105.0, p)
    assert why == 'target' and exit_m == 601 and exit_price == 105.0


def test_walk_path_stop_first_on_bar_touching_both():
    """A bar whose low<=stop AND high>=target resolves to the STOP (stop-first rule), never target."""
    from research.hod_entry.sip_rebuild import walk_path
    p = bars([(600, 100.0, 106.0, 97.0, 99.0)])     # touches stop(98) and target(105) in the same bar
    exit_m, exit_price, why = walk_path(100.0, 98.0, 105.0, p)
    assert why == 'stop'
    assert exit_price == 98.0


def test_walk_path_gap_through_at_open():
    from research.hod_entry.sip_rebuild import walk_path
    p = bars([(600, 97.0, 97.5, 96.5, 97.0)])        # opens BELOW the stop -> exit at the open, not the stop
    exit_m, exit_price, why = walk_path(100.0, 98.0, 105.0, p)
    assert why == 'stop' and exit_price == 97.0


# ------------------------------------------------------------------------------------------- anatomy / drawdown-until-touch
def test_anatomy_drawdown_stops_counting_at_the_touch():
    """A lower low AFTER the +5% touch bar must NOT lower the reported drawdown -- the window is
    [fill bar, touch bar] inclusive, nothing after."""
    level = 100.0
    b = bars([
        (600, 100.0, 100.2, 99.5, 100.0),     # fill bar, small dip (0.5%)
        (601, 100.0, 105.5, 99.9, 105.0),     # touches +5% here -> touch_m = 601
        (602, 105.0, 105.2, 90.0, 91.0),      # deep dip AFTER the touch -- must be ignored
    ])
    row = SimpleNamespace(symbol='AAA', day='2026-01-01', fill_min=600.0, level=level, stop=90.0,
                           is_extender=True, why='target', exit_m=650)
    a = c1548.anatomy_one(row, {('AAA', '2026-01-01'): b})
    assert a['touch_m'] == 601
    # drawdown should reflect the min low up to and incl. the touch bar (99.5), NOT the post-touch 90.0
    assert abs(a['dd_pct'] - 0.5) < 1e-6, a['dd_pct']


def test_anatomy_no_touch_found_flags_missing_touch_for_extenders():
    level = 100.0
    b = bars([(600, 100.0, 100.2, 99.8, 100.0), (601, 100.0, 100.3, 99.7, 100.0)])   # never reaches 105
    row = SimpleNamespace(symbol='AAA', day='2026-01-01', fill_min=600.0, level=level, stop=90.0,
                           is_extender=True, why='eod', exit_m=650)
    a = c1548.anatomy_one(row, {('AAA', '2026-01-01'): b})
    assert a == {'missing_touch': True}


def test_anatomy_non_extender_window_is_until_1555():
    """Non-extenders never touch +5%; the drawdown window runs to EOD_M (15:55), and a dip AFTER
    that (which cannot exist inside RTH bars provided) is irrelevant -- here we just confirm the
    window includes bars up to EOD_M and minutes_to_touch is NaN."""
    level = 100.0
    b = bars([(600, 100.0, 100.2, 95.0, 96.0), (955, 96.0, 96.5, 94.0, 95.0), (960, 95.0, 95.5, 80.0, 81.0)])
    row = SimpleNamespace(symbol='AAA', day='2026-01-01', fill_min=600.0, level=level, stop=50.0,
                           is_extender=False, why='stop', exit_m=650)
    a = c1548.anatomy_one(row, {('AAA', '2026-01-01'): b})
    assert np.isnan(a['minutes_to_touch'])
    # the m=960 bar (after EOD_M=955) must not be counted -- min low should be 94.0, not 80.0
    assert abs(a['dd_pct'] - 6.0) < 1e-6, a['dd_pct']


# ------------------------------------------------------------------------------------------- d, s, W from TRAIN only
def test_dsw_computed_from_train_only():
    """VAL rows with wildly different drawdown/minutes must not move d, s or W at all."""
    train_rows = pd.DataFrame({
        'split': ['TRAIN'] * 4,
        'is_extender': [True] * 4,
        'dd_pct': [0.5, 1.0, 1.5, 2.0],
        'minutes_to_touch': [10.0, 20.0, 30.0, 40.0],
    })
    val_rows = pd.DataFrame({
        'split': ['VAL'] * 4,
        'is_extender': [True] * 4,
        'dd_pct': [50.0, 60.0, 70.0, 80.0],           # if these leaked in, d/s would be huge
        'minutes_to_touch': [500.0, 500.0, 500.0, 500.0],
    })
    an = pd.concat([train_rows, val_rows], ignore_index=True)
    d, s, W = c1548.compute_dsw(an)
    assert d < 5.0 and s < 5.0 and W < 60.0     # nowhere near the VAL-contaminated values
    # exact TRAIN-only quantiles: median([0.5,1,1.5,2])=1.25, p75=1.625, minutes p75=32.5 (capped @120)
    assert abs(d - 1.25) < 1e-9
    assert abs(s - 1.625) < 1e-9
    assert abs(W - 32.5) < 1e-9


def test_d_is_floored_at_020_pct():
    train_rows = pd.DataFrame({'split': ['TRAIN'] * 3, 'is_extender': [True] * 3,
                                'dd_pct': [0.01, 0.02, 0.03], 'minutes_to_touch': [1.0, 2.0, 3.0]})
    d, s, W = c1548.compute_dsw(train_rows)
    assert d == c1548.D_FLOOR_PCT


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
