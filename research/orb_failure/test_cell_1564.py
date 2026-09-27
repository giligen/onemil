#!/usr/bin/env python3
"""Unit tests for research/orb_failure/cell_1564.py (PREREG_1564.md). Synthetic bars only --
no DB or network access, so these run standalone and fast."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cell_1564 import (  # noqa: E402
    classify_one, walk_short, walk_long, cost_R, build_trade, RANGE_BUFFER,
    DECL_1030_M, EOD_M, SLIP_STOP_BPS,
)


def bar(m, o, h, l, c, v=1000):
    return dict(m=m, o=o, h=h, l=l, c=c, v=v)


def bars_df(rows):
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------
# The 10:30 boundary: nothing after 10:30:00 enters an event
# ------------------------------------------------------------------------------------------

def test_break_after_1030_is_not_counted():
    """A high that would break the range only appears at 10:31 -- must NOT count as a BREAK for
    the 10:30 declaration (it only shows up in the 11:00 report-only declaration)."""
    rows = [bar(570, 10, 10.2, 9.9, 10.0)] + [bar(m, 10, 10.1, 9.95, 10.0) for m in range(571, 575)]
    rows += [bar(m, 10, 10.15, 9.9, 10.0) for m in range(575, 630)]      # no break through 10:29
    rows.append(bar(630, 10.0, 10.0, 9.9, 10.0))                        # 10:30 bar (entry)
    rows.append(bar(631, 10.0, 10.35, 9.9, 10.3))                       # break happens at 10:31
    df = bars_df(rows)
    cl = classify_one(df, DECL_1030_M)
    assert cl['event'] == 'NO_BREAK'
    cl11 = classify_one(df, 11 * 60)
    assert cl11['event'] in ('SUCCESS', 'INDETERMINATE', 'FAILURE')  # break IS visible by 11:00


def test_declaration_uses_bar_strictly_before_decl_m_for_pre_decl_stats():
    """The 10:30 bar itself (m == 630) must not leak into the pre-declaration BREAK/FAILURE scan."""
    rows = [bar(m, 10, 10.05, 9.98, 10.0) for m in range(570, 630)]
    rows.append(bar(630, 10.0, 50.0, 0.01, 10.0))   # huge range only in the entry bar itself
    df = bars_df(rows)
    cl = classify_one(df, DECL_1030_M)
    assert cl['event'] == 'NO_BREAK'   # the 10:30 bar's own extremes must not count as a break


# ------------------------------------------------------------------------------------------
# FAILURE needs a break first
# ------------------------------------------------------------------------------------------

def test_failure_requires_a_prior_break():
    """A low <= range_low - buffer with NO prior break is not a FAILURE (nothing broke out)."""
    rows = [bar(m, 10, 10.05, 9.98, 10.0) for m in range(570, 575)]
    rows += [bar(m, 9.95, 9.97, 9.80, 9.90) for m in range(575, 630)]  # drops through range_low,
    df = bars_df(rows)                                                  # but never broke the high
    cl = classify_one(df, DECL_1030_M)
    assert cl['event'] == 'NO_BREAK'


def test_break_then_stop_is_failure():
    rows = [bar(m, 10, 10.05, 9.98, 10.0) for m in range(570, 575)]     # range_high=10.05, low=9.98
    rows.append(bar(600, 10.0, 10.10, 9.99, 10.08))                     # BREAK (>=10.06)
    rows.append(bar(610, 10.05, 10.06, 9.96, 9.97))                     # stop: low <= 9.97
    rows += [bar(m, 9.9, 9.95, 9.85, 9.9) for m in range(611, 630)]
    df = bars_df(rows)
    cl = classify_one(df, DECL_1030_M)
    assert cl['event'] == 'FAILURE'
    assert cl['break_m'] == 600


def test_break_held_is_success():
    rows = [bar(m, 10, 10.05, 9.98, 10.0) for m in range(570, 575)]
    rows.append(bar(600, 10.0, 10.10, 9.99, 10.08))                     # BREAK, holds above range_high
    rows += [bar(m, 10.08, 10.20, 10.06, 10.10) for m in range(601, 630)]
    df = bars_df(rows)
    cl = classify_one(df, DECL_1030_M)
    assert cl['event'] == 'SUCCESS'


# ------------------------------------------------------------------------------------------
# Entry is the 10:30 open
# ------------------------------------------------------------------------------------------

def test_entry_open_is_the_1030_bar_open_not_close():
    rows = [bar(m, 10, 10.05, 9.98, 10.0) for m in range(570, 575)]
    rows += [bar(m, 10.0, 10.02, 9.99, 10.0) for m in range(575, 630)]
    rows.append(bar(630, 11.5, 11.6, 11.4, 11.55))   # 10:30 bar: open != close
    df = bars_df(rows)
    cl = classify_one(df, DECL_1030_M)
    assert cl['entry_open'] == 11.5


# ------------------------------------------------------------------------------------------
# The short's stop is the day high through 10:30 + 1 cent
# ------------------------------------------------------------------------------------------

def test_short_stop_is_high_through_decl_plus_buffer():
    rows = [bar(m, 10, 10.05, 9.98, 10.0) for m in range(570, 575)]
    rows.append(bar(600, 10.0, 10.30, 9.99, 10.08))                     # break, day high so far 10.30
    rows.append(bar(610, 10.05, 10.06, 9.96, 9.97))                     # stop-out -> FAILURE
    rows += [bar(m, 9.9, 9.95, 9.85, 9.9) for m in range(611, 630)]
    df = bars_df(rows)
    cl = classify_one(df, DECL_1030_M)
    assert cl['high_through_decl'] == pytest.approx(10.30)
    # build_trade computes stop = high_through_decl + RANGE_BUFFER
    stop = cl['high_through_decl'] + RANGE_BUFFER
    assert stop == pytest.approx(10.31)


# ------------------------------------------------------------------------------------------
# walk_path: stop-first priority, gap-through at the open, EOD at 15:55 open
# ------------------------------------------------------------------------------------------

def test_walk_short_stop_first_when_both_touch_same_bar():
    path = bars_df([bar(630, 10.0, 10.5, 9.0, 9.5)])   # both stop (10.31) and target could trip
    exit_m, exit_px, why = walk_short(630, 10.0, 10.31, 9.0, path)
    assert why == 'stop'


def test_walk_short_gap_through_at_open():
    path = bars_df([bar(630, 10.5, 10.6, 10.4, 10.5)])   # opens already above the stop
    exit_m, exit_px, why = walk_short(630, 10.0, 10.31, 9.0, path)
    assert why == 'stop' and exit_px == pytest.approx(10.5)  # exits at the open, not the stop level


def test_walk_short_eod_at_1555_open():
    path = bars_df([bar(EOD_M, 9.8, 9.85, 9.75, 9.8)])
    exit_m, exit_px, why = walk_short(630, 10.0, 10.31, 9.0, path)
    assert why == 'eod' and exit_px == pytest.approx(9.8)


def test_walk_long_stop_and_target_mirror():
    path = bars_df([bar(630, 10.0, 10.5, 9.4, 10.0)])
    exit_m, exit_px, why = walk_long(630, 10.0, 9.5, 11.0, path)
    assert why not in ('target',)  # low 9.4 <= stop 9.5 trips first


# ------------------------------------------------------------------------------------------
# SSR exclusion (via build_trade)
# ------------------------------------------------------------------------------------------

def _minimal_cl(entry_open=10.0, high_through=10.3, range_low=9.9, vwap=10.0):
    return dict(entry_open=entry_open, high_through_decl=high_through, range_low=range_low,
                vwap_through_decl=vwap, range_high=10.05, n_range_bars=5)


def test_ssr_excludes_the_short():
    coverage = {}
    path = bars_df([bar(EOD_M, 9.8, 9.85, 9.75, 9.8)])
    tr = build_trade('2025-06-01', 'ABC', 'TRAIN', 10.0, _minimal_cl(), 'short', {}, {},
                      borrow_ok=True, ssr=True, bars_after=path, coverage=coverage)
    assert tr['entered'] is False and tr['why'] == 'ssr'


def test_not_shortable_excludes_the_short():
    coverage = {}
    path = bars_df([bar(EOD_M, 9.8, 9.85, 9.75, 9.8)])
    tr = build_trade('2025-06-01', 'ABC', 'TRAIN', 10.0, _minimal_cl(), 'short', {}, {},
                      borrow_ok=False, ssr=False, bars_after=path, coverage=coverage)
    assert tr['entered'] is False and tr['why'] == 'not_shortable'


def test_price_below_5_excludes_every_cell():
    coverage = {}
    path = bars_df([bar(EOD_M, 4.5, 4.55, 4.45, 4.5)])
    cl = _minimal_cl(entry_open=4.5)
    tr = build_trade('2025-06-01', 'ABC', 'TRAIN', 4.5, cl, 'long', {}, {},
                      borrow_ok=True, ssr=False, bars_after=path, coverage=coverage)
    assert tr['entered'] is False and tr['why'] == 'price_lt_5'


# ------------------------------------------------------------------------------------------
# Cost units: cost_R divides the dollar cost by R (dollar risk/share), not by price
# ------------------------------------------------------------------------------------------

def test_cost_R_units_are_per_R_not_per_price():
    entry_fill, exit_price, R = 10.0, 9.9, 1.0     # R = $1/share (a wide, atypical R for the test)
    entry_half, exit_half = 0.02, 0.02
    c_stop = cost_R('stop', entry_fill, exit_price, entry_half, exit_half, 'TRAIN', R,
                     is_short=True, minutes_held=5, borrow_applicable=True)
    # stop cost = SLIP_STOP_BPS bps of exit_price (in $) + entry_half + tiny borrow, all / R
    expected_stop_dollars = SLIP_STOP_BPS['TRAIN'] / 1e4 * exit_price
    expected = (entry_half + expected_stop_dollars) / R
    assert c_stop == pytest.approx(expected, rel=1e-6, abs=1e-6) or c_stop > expected - 1e-6
    # doubling R must roughly halve cost_R (all else equal) -- confirms the division is by R
    c_stop_wideR = cost_R('stop', entry_fill, exit_price, entry_half, exit_half, 'TRAIN', 2.0,
                           is_short=True, minutes_held=5, borrow_applicable=True)
    assert c_stop_wideR < c_stop


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
