"""Unit tests for cell_1441.py (prior-day-high break) on synthetic minute bars. Mirrors
test_causal_arming.py's fixture style with the level FIXED at a PDH argument instead of the running HOD."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell_1441 as c1441  # noqa: E402
from trading.hod_break import HodBreakParams  # noqa: E402

P = HodBreakParams(max_per_day=12, last_entry_minute=840)
PDH = 10.6


def bars(o0=10.0, low_min=10.3, start_m=570, n_after=3, next_high=10.7, pre_break=None):
    """Session: bar0 open o0, bars 1-6 consolidate under PDH (highs <= PDH, lows >= low_min), bar 7 = next bar
    that may cross PDH. `pre_break`, if given, overrides bar index `pre_break`'s high to exceed PDH (to test
    that a prior print through PDH disarms every later bar)."""
    h = [10.2, 10.55, 10.55, 10.55, 10.55, 10.55, 10.55, next_high] + [10.5] * n_after
    lo = [9.9, 10.3, 10.35, low_min, 10.35, 10.35, 10.35, 10.4] + [10.4] * n_after
    n = len(h)
    if pre_break is not None:
        h[pre_break] = PDH + 0.05
    return pd.DataFrame({'m': np.arange(start_m, start_m + n), 'o': [o0] + [10.4] * (n - 1), 'h': h, 'l': lo,
                         'c': [10.4] * n, 'v': [1000.0] * n})


def arm(b, j=6, adv=175000.0, use_rv=True, floor=0.0, pdh=PDH):
    return c1441.arm_state_pdh(b.o.values, b.h.values, b.l.values, b.v.values, b.m.values, j, adv, P, floor,
                                pdh, use_rv)


def test_armed_baseline_uses_fixed_pdh_as_level_and_stop_is_consolidation_low():
    """PDH (not the running HOD) is the level; the mirrored consolidation low is the stop."""
    a = arm(bars())
    assert a['level'] == PDH and a['stop'] == 10.3 and a['trigger'] == pytest.approx(10.61)
    assert a['limit'] == pytest.approx(PDH * 1.0015)


def test_disarmed_once_the_running_high_reaches_pdh_first_cross_only():
    """A bar that already printed at/through PDH before bar j permanently disarms every later bar — this IS
    the 'one entry per symbol-day, first cross only' rule (no separate flag needed)."""
    assert arm(bars()) is not None
    assert arm(bars(pre_break=2)) is None                          # bar 2's high already reached PDH
    assert arm(bars(pre_break=6)) is None                          # even the arming bar itself must stay below


def test_consolidation_must_hold_under_the_level_not_the_running_hod():
    """Bars j-K+1..j must have highs <= PDH and lows within consol_pct of PDH (mirror of consolidation_low with
    the level swapped in) — a low too far below PDH disarms."""
    assert arm(bars(low_min=PDH * 0.96 + 0.001)) is not None
    assert arm(bars(low_min=PDH * 0.96 - 0.001)) is None
    # a high inside the consolidation window that exceeds PDH also disarms (covered by pre_break tests above
    # for the running-high guard; consolidation_low_pdh's own highs<=level check is redundant-but-explicit).
    b = bars(); b.loc[4, 'h'] = PDH + 0.01
    assert arm(b) is None


def test_min_dist_and_floor_boundaries():
    """PDH must be >= open x (1 + min_dist%) and >= floor + 1c."""
    assert arm(bars(o0=PDH / 1.0501)) is not None
    assert arm(bars(o0=PDH / 1.0499)) is None
    assert arm(bars(), floor=PDH - 0.02) is not None
    assert arm(bars(), floor=PDH) is None


def test_rv_band_boundary_uses_volume_through_j_only():
    """Same rv mechanism as cell 1,438 — volume/rv logic is level-agnostic."""
    assert arm(bars(), adv=7000 / 0.02 / 1.001) is not None
    assert arm(bars(), adv=7000 / 0.02 / 0.999) is None


def test_crossing_filter_needs_next_bar_at_trigger():
    """Only a symbol-day whose next bar reaches the (constant) trigger produces a candidate."""
    assert [c['j'] for c in c1441.armed_crossing_bars_pdh(bars(next_high=10.65), 175000.0, PDH, P, 0.0)] == [6]
    assert c1441.armed_crossing_bars_pdh(bars(next_high=10.605), 175000.0, PDH, P, 0.0) == []


def test_armed_crossing_bars_pdh_stops_after_the_first_cross():
    """Even if later bars' highs would otherwise re-qualify, no candidate appears once PDH has printed."""
    b = bars(next_high=10.65)
    b = pd.concat([b, pd.DataFrame({'m': [b.m.max() + 1], 'o': [10.6], 'h': [10.65], 'l': [10.55], 'c': [10.6],
                                     'v': [1000.0]})], ignore_index=True)
    cands = c1441.armed_crossing_bars_pdh(b, 175000.0, PDH, P, 0.0)
    assert [c['j'] for c in cands] == [6]                            # not also the later bar


def test_consolidation_low_pdh_boundary_matches_mirror_shape():
    h = np.array([10.4, 10.5, 10.55, 10.55, 10.55])
    l = np.array([10.1, 10.2, 10.2, 10.2, 10.2])
    assert c1441.consolidation_low_pdh(l, h, 4, HodBreakParams(consol_bars=4), PDH) == pytest.approx(10.2)
    h2 = np.array([10.4, 10.5, 10.55, 10.61, 10.55])                 # a high above PDH inside the window
    assert c1441.consolidation_low_pdh(l, h2, 4, HodBreakParams(consol_bars=4), PDH) is None


def test_prior_trading_day_and_calendar():
    cal = ['2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08']
    assert c1441.prior_trading_day(cal, '2026-01-07') == '2026-01-06'
    assert c1441.prior_trading_day(cal, '2026-01-05') is None       # first day in the calendar -> no prior
    assert c1441.prior_trading_day(cal, '2026-01-09') == '2026-01-08'  # day not in calendar -> last one before it


def test_load_pdh_table_marks_missing_prior_session_as_absent(tmp_path):
    """A symbol with no rows on the prior day is simply absent from the returned dict (no_pdh), never
    backfilled from another source."""
    import sqlite3
    db = tmp_path / 'sip.db'
    con = sqlite3.connect(str(db))
    con.execute('create table bars (day text, symbol text, t text, o real, h real, l real, c real, v real)')
    day0, day1 = '2026-01-06', '2026-01-07'
    for m in range(570, 576):                                       # 09:30-09:35 ET = 14:30-14:35 UTC (EST, Jan)
        ts = f'{day0}T{m // 60 + 5:02d}:{m % 60:02d}:00Z'
        con.execute('insert into bars values (?,?,?,?,?,?,?,?)', (day0, 'AAA', ts, 10, 10.7, 9.9, 10.1, 100))
    con.commit()
    calendar = [day0, day1]
    pdh = c1441.load_pdh_table(con, [day1], {day1: ['AAA', 'BBB']}, calendar)
    assert pdh[(day1, 'AAA')] == pytest.approx(10.7)
    assert (day1, 'BBB') not in pdh                                  # BBB never traded on day0 -> no_pdh
    con.close()
