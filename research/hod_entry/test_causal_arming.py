"""Unit tests for causal_arming.py (cell 1,438) on synthetic minute bars and ticks."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_arming as ca  # noqa: E402
import sip_rebuild as sr  # noqa: E402
from trading.hod_break import HodBreakParams  # noqa: E402

P = HodBreakParams(max_per_day=12, last_entry_minute=840)
DAY = '2026-02-02'


def bars(o0=10.0, low_min=10.3, start_m=570, n_after=3, next_high=10.7):
    """Session: bar0 open o0, bar1 sets HOD 10.6, bars 2-6 consolidate (lows >= low_min), bar 7 = next bar."""
    h = [10.2, 10.6, 10.55, 10.55, 10.55, 10.55, 10.55, next_high] + [10.5] * n_after
    lo = [9.9, 10.3, 10.35, low_min, 10.35, 10.35, 10.35, 10.4] + [10.4] * n_after
    n = len(h)
    return pd.DataFrame({'m': np.arange(start_m, start_m + n), 'o': [o0] + [10.4] * (n - 1), 'h': h, 'l': lo,
                         'c': [10.4] * n, 'v': [1000.0] * n})


def arm(b, j=6, adv=175000.0, use_rv=True, floor=0.0):
    """arm_state on a bars frame."""
    return ca.arm_state(b.o.values, b.h.values, b.l.values, b.v.values, b.m.values, j, adv, P, floor, use_rv)


def test_armed_baseline_and_level_stop():
    """HOD through j is the level, the consolidation low the stop."""
    a = arm(bars())
    assert a['level'] == 10.6 and a['stop'] == 10.3 and a['trigger'] == pytest.approx(10.61)
    assert a['limit'] == pytest.approx(10.6 * 1.0015)


def test_consolidation_boundary():
    """A low below HOD x (1 - 4 %) inside bars j-4..j disarms."""
    assert arm(bars(low_min=10.6 * 0.96 + 0.001)) is not None
    assert arm(bars(low_min=10.6 * 0.96 - 0.001)) is None


def test_min_dist_boundary():
    """Level must be >= open x 1.05."""
    assert arm(bars(o0=10.6 / 1.0501)) is not None
    assert arm(bars(o0=10.6 / 1.0499)) is None


def test_rv_band_boundary_uses_volume_through_j_only():
    """rv = cumv[j] / (adv20 x fraction(m[j])): 7,000 / (adv x 0.02); band [1, 5)."""
    assert arm(bars(), adv=7000 / 0.02 / 1.001) is not None       # rv 1.001
    assert arm(bars(), adv=7000 / 0.02 / 0.999) is None           # rv 0.999
    assert arm(bars(), adv=7000 / 0.02 / 4.999) is not None       # rv 4.999
    assert arm(bars(), adv=7000 / 0.02 / 5.001) is None           # rv 5.001
    b = bars(); b.loc[7, 'v'] = 1e9                               # the next bar's volume never matters
    assert arm(b) is not None
    assert arm(bars(), adv=7000 / 0.02 / 0.5, use_rv=False) is not None


def test_last_entry_minute_boundary():
    """m[j+1] <= last_entry_minute (840) arms, 841 does not."""
    from trading.hod_break import profile_fraction
    adv = 7000 / (profile_fraction(839) * 2)                        # rv 2 at the clock m[j] = 839
    assert arm(bars(start_m=840 - 7), adv=adv) is not None
    assert arm(bars(start_m=841 - 7), adv=adv) is None


def test_price_floor_and_min_r():
    """trigger >= min_price, and (trigger - stop) / trigger >= 1 %."""
    assert arm(bars(), floor=10.61) is not None and arm(bars(), floor=10.62) is None
    b = bars(low_min=10.55); b.loc[[1, 2, 4, 5, 6], 'l'] = 10.55
    assert arm(b) is None                                          # R = 0.06 / 10.61 < 1 %


def test_crossing_filter_needs_next_bar_at_trigger():
    """Only armed bars whose next bar's high reaches level + 1c are candidates."""
    assert [c['j'] for c in ca.armed_crossing_bars(bars(next_high=10.61), 175000.0, P, 0.0)] == [6]
    assert ca.armed_crossing_bars(bars(next_high=10.605), 175000.0, P, 0.0) == []


def ticks(day, m, rows):
    """Trades df from (second-in-minute, price, size) inside minute m."""
    base = sr.et_ns(day, m * 60)
    return pd.DataFrame({'ts': [base + int(s * 1e9) for s, _, _ in rows], 'price': [p for _, p, _ in rows],
                         'size': [float(z) for _, _, z in rows]}).astype({'ts': 'int64'})


def quotes(day, m, rows):
    """Quotes df from (second-in-minute, bid, ask) inside minute m."""
    base = sr.et_ns(day, m * 60)
    return pd.DataFrame({'ts': [base + int(s * 1e9) for s, _, _ in rows], 'bid': [b for _, b, _ in rows],
                         'ask': [a for _, _, a in rows]}).astype({'ts': 'int64'})


ARM = dict(level=10.6, trigger=10.61, limit=10.6 * 1.0015, stop=10.3)


def test_rearm_after_no_cross_and_one_fill_per_day():
    """First armed window has no print >= trigger -> next armed window fills; later windows are never used."""
    c1 = dict(ARM, m_lo=600, m_hi=601, cumv_j=0.0)
    c2 = dict(ARM, m_lo=610, m_hi=611, cumv_j=0.0)
    c3 = dict(ARM, m_lo=620, m_hi=621, cumv_j=0.0)
    tapes = {601: (ticks(DAY, 601, [(10, 10.60, 100)]), quotes(DAY, 600, [(50, 10.59, 10.60)])),
             611: (ticks(DAY, 611, [(10, 10.62, 100)]), quotes(DAY, 610, [(50, 10.60, 10.61)])),
             621: (ticks(DAY, 621, [(10, 10.62, 100)]), quotes(DAY, 620, [(50, 10.60, 10.61)]))}
    status, a, e = ca.resolve_day([c1, c2, c3], lambda a: tapes[a['m_hi']], DAY, 1e6, P)
    assert status == 'fill' and a['m_hi'] == 611 and e['fill'] == pytest.approx(10.61)


def test_unusable_window_stops_the_day_and_ask_above_limit_is_nofill():
    """No print in a crossing window -> no_tape; ask above limit -> nofill then re-arm."""
    c1 = dict(ARM, m_lo=600, m_hi=601, cumv_j=0.0)
    empty = (ticks(DAY, 601, []), quotes(DAY, 600, [(50, 10.59, 10.60)]))
    assert ca.resolve_day([c1], lambda a: empty, DAY, 1e6, P)[0] == 'no_tape'
    wide = (ticks(DAY, 601, [(10, 10.62, 100)]), quotes(DAY, 600, [(50, 10.60, 10.63)]))
    assert ca.resolve_day([c1], lambda a: wide, DAY, 1e6, P)[0] == 'nofill'


def test_window_bounds_and_tick_rv_variant():
    """Prints before the end of minute m_lo are outside the window; tick_rv fills at the first print whose running
    volume puts rv in band."""
    c = dict(ARM, m_lo=600, m_hi=601, cumv_j=0.0)
    s, e = ca.window_ns(DAY, 600, 601)
    t = ticks(DAY, 600, [(30, 10.7, 100)])                         # inside minute 600 = before the window
    q = quotes(DAY, 600, [(1, 10.60, 10.61)])
    assert ca.resolve_window(t, q, s, e, c)['status'] == 'no_tape'  # no print inside the window = unusable
    t = ticks(DAY, 601, [(5, 10.61, 100), (20, 10.62, 5000)])
    frac = 0.097                                                    # profile fraction at minute 601 (600 checkpoint)
    adv = 5100 / frac / 1.5                                         # rv 1.5 only after the second print
    r = ca.resolve_window(t, q, s, e, c, rv_ctx=(0.0, adv, 601, P))
    assert r['status'] == 'fill' and r['fill_ts'] == t.ts.iloc[1]
