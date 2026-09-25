"""Unit tests for cell_1431.py's short-side walk, entry chase cap, and cost mechanics, on synthetic
bars (frozen PREREG in research/hod_entry/PREREG_WEEKEND.md, section 1,431)."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell_1431 as c1431  # noqa: E402


def bars(*rows):
    """rows = (m, o, h, l, c) tuples -> itertuples()-ready DataFrame."""
    df = pd.DataFrame(rows, columns=['m', 'o', 'h', 'l', 'c'])
    return df


ENTRY, STOP = 10.0, 10.5     # R = 0.5, target = entry - 2R = 9.0
R = STOP - ENTRY
TARGET = ENTRY - c1431.TARGET_R * R
EOD_M = 955


# --------------------------------------------------------------------------------------------
# walk_short: stop/target mirror of B0's long-side walk
# --------------------------------------------------------------------------------------------

def test_walk_short_target_hit_first():
    b = bars((700, 10.0, 10.1, 8.9, 9.0))   # low 8.9 <= target 9.0, high 10.1 < stop 10.5
    exit_m, px, why = c1431.walk_short(ENTRY, STOP, TARGET, b.itertuples(), EOD_M)
    assert (exit_m, px, why) == (700, TARGET, 'target')


def test_walk_short_stop_hit_first():
    b = bars((700, 10.0, 10.6, 9.5, 10.4))  # high 10.6 >= stop 10.5, low 9.5 <= target too
    exit_m, px, why = c1431.walk_short(ENTRY, STOP, TARGET, b.itertuples(), EOD_M)
    assert (exit_m, px, why) == (700, STOP, 'stop')     # stop wins the same-bar tie, not gapped


def test_walk_short_stop_priority_over_target_on_double_touch():
    """A bar whose high clears stop AND low clears target must exit at the stop (mirror of B0's
    own stop-before-target rule on a bar that touches both)."""
    b = bars((700, 10.0, 11.0, 8.0, 9.0))
    exit_m, px, why = c1431.walk_short(ENTRY, STOP, TARGET, b.itertuples(), EOD_M)
    assert why == 'stop'
    assert px == STOP


def test_walk_short_gap_through_open_above_stop():
    b = bars((700, 10.7, 10.9, 10.6, 10.8))  # open already above stop -> covered at the open
    exit_m, px, why = c1431.walk_short(ENTRY, STOP, TARGET, b.itertuples(), EOD_M)
    assert (exit_m, px, why) == (700, 10.7, 'stop')


def test_walk_short_physics_apply_on_the_entry_bar_itself():
    """The FIRST bar passed in (m == entry_m) is the entry bar; stop/target checked on it too,
    matching sip_rebuild.walk_path's 'path rows m >= entry_m'."""
    b = bars((700, 10.0, 10.1, 8.9, 9.0))   # target hit on the very first (entry) bar
    exit_m, px, why = c1431.walk_short(ENTRY, STOP, TARGET, b.itertuples(), EOD_M)
    assert exit_m == 700 and why == 'target'


def test_walk_short_covers_at_eod_open_when_neither_triggers():
    b = bars((700, 10.0, 10.2, 9.8, 10.1),
             (955, 9.9, 10.0, 9.8, 9.9))
    exit_m, px, why = c1431.walk_short(ENTRY, STOP, TARGET, b.itertuples(), EOD_M)
    assert (exit_m, px, why) == (955, 9.9, 'eod')


def test_walk_short_returns_none_if_path_ends_before_eod():
    b = bars((700, 10.0, 10.2, 9.8, 10.1))
    assert c1431.walk_short(ENTRY, STOP, TARGET, b.itertuples(), EOD_M) is None


# --------------------------------------------------------------------------------------------
# chase cap: skip if the entry bar's open is more than 60 bps BELOW the level
# --------------------------------------------------------------------------------------------

def _row(level, entry_m=700, day='2025-06-02', symbol='ABC', break_m=699, bb_high=10.5, wk='w1'):
    return pd.Series(dict(day=day, symbol=symbol, entry_m=entry_m, break_m=break_m, level=level,
                           bb_high=bb_high, wk=wk))


def test_chase_cap_skips_when_open_too_far_below_level():
    row = _row(level=10.0)
    idx = pd.DataFrame({'day': ['2025-06-02'], 'symbol': ['ABC'], 'm': [700],
                         'o': [9.4], 'h': [9.5], 'l': [9.3], 'c': [9.4]}).set_index(['day', 'symbol'])
    out = c1431.simulate_signal(row, idx, {}, exit_half_const=0.01)
    assert out is not None and out['skipped'] == 'chase_cap'    # 9.4 < 10.0*(1-0.006)=9.94


def test_chase_cap_allows_open_within_60bps_below_level():
    row = _row(level=10.0)
    idx = pd.DataFrame({'day': ['2025-06-02', '2025-06-02'], 'symbol': ['ABC', 'ABC'], 'm': [700, 955],
                         'o': [9.95, 9.9], 'h': [9.96, 10.0], 'l': [9.9, 9.8],
                         'c': [9.95, 9.9]}).set_index(['day', 'symbol'])
    out = c1431.simulate_signal(row, idx, {}, exit_half_const=0.01)
    assert out is None    # entry ok (9.95 >= 9.94) but no cached quote at S_ns -> dropped, not skipped


# --------------------------------------------------------------------------------------------
# cost: entry half-spread from the quote at the open + fixed exit constant + slip on stops
# --------------------------------------------------------------------------------------------

def test_simulate_signal_cost_and_slip_on_a_stop_exit():
    row = _row(level=10.0, entry_m=700, break_m=699, bb_high=10.5)   # stop = 10.51
    idx = pd.DataFrame({'day': ['2025-06-02'], 'symbol': ['ABC'], 'm': [700],
                         'o': [10.0], 'h': [10.6], 'l': [9.9],
                         'c': [10.3]}).set_index(['day', 'symbol'])
    quotes = pd.DataFrame({'ts': [0], 'bid': [9.99], 'ask': [10.01]})   # half-spread = 0.01
    tapes_by_day = {'2025-06-02': {c1431.sr.sig_key('ABC', 699): (None, quotes)}}

    out = c1431.simulate_signal(row, idx, tapes_by_day, exit_half_const=0.02)
    assert out is not None and out['skipped'] is None
    R = 10.51 - 10.0
    raw_R = (10.0 - 10.51) / R          # stop exit, entry=10.0
    cost_R = (0.01 + 0.02 + c1431.sr.SLIP_BP * 10.51) / R
    assert out['net_R_noslip'] == pytest.approx(raw_R - cost_R, abs=1e-9)
    expected_slip = out['net_R_noslip'] - c1431.c1430.STOP_SLIP_BP * 10.51 / R
    assert out['net_R_slip'] == pytest.approx(expected_slip, abs=1e-9)
    assert out['why'] == 'stop'


def test_simulate_signal_drops_degenerate_r_below_the_floor():
    """entry == bb_high -> R = 1 tick, far below the 0.5%-of-price floor -> dropped, not scored."""
    row = _row(level=10.0, entry_m=700, break_m=699, bb_high=10.0)   # stop = 10.01, R = 0.01
    idx = pd.DataFrame({'day': ['2025-06-02'], 'symbol': ['ABC'], 'm': [700],
                         'o': [10.0], 'h': [10.02], 'l': [9.98],
                         'c': [10.0]}).set_index(['day', 'symbol'])
    quotes = pd.DataFrame({'ts': [0], 'bid': [9.99], 'ask': [10.01]})
    tapes_by_day = {'2025-06-02': {c1431.sr.sig_key('ABC', 699): (None, quotes)}}
    out = c1431.simulate_signal(row, idx, tapes_by_day, exit_half_const=0.02)
    assert out is None


def test_simulate_signal_returns_none_without_a_cached_quote():
    row = _row(level=10.0, entry_m=700, break_m=699, bb_high=10.5)
    idx = pd.DataFrame({'day': ['2025-06-02'], 'symbol': ['ABC'], 'm': [700],
                         'o': [10.0], 'h': [10.05], 'l': [9.95],
                         'c': [10.0]}).set_index(['day', 'symbol'])
    out = c1431.simulate_signal(row, idx, {}, exit_half_const=0.02)
    assert out is None
