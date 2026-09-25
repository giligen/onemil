"""Unit tests for cell_1439.py (low-of-day mirror) on synthetic minute bars and ticks. Mirrors
test_causal_arming.py's fixture style for the SELL side."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell_1439 as c1439  # noqa: E402
import sip_rebuild as sr  # noqa: E402
from trading.hod_break import HodBreakParams  # noqa: E402

P = HodBreakParams(max_per_day=12, last_entry_minute=840)
DAY = '2026-02-02'


def bars(o0=10.6, high_max=10.3, start_m=570, n_after=3, next_low=9.9):
    """Session: bar0 open o0, bar1 sets LOD 10.0, bars 2-6 consolidate (highs <= high_max), bar 7 = next bar
    (mirror of test_causal_arming.bars)."""
    lo = [10.2, 10.0, 10.05, 10.05, 10.05, 10.05, 10.05, next_low] + [10.1] * n_after
    hi = [10.5, 10.3, 10.25, high_max, 10.25, 10.25, 10.25, 10.15] + [10.1] * n_after
    n = len(lo)
    return pd.DataFrame({'m': np.arange(start_m, start_m + n), 'o': [o0] + [10.1] * (n - 1), 'h': hi, 'l': lo,
                         'c': [10.1] * n, 'v': [1000.0] * n})


def arm(b, j=6, adv=175000.0, use_rv=True, floor=0.0):
    return c1439.arm_state_low(b.o.values, b.h.values, b.l.values, b.v.values, b.m.values, j, adv, P, floor, use_rv)


def test_armed_baseline_and_level_stop():
    """LOD through j is the level, the mirrored consolidation high the stop."""
    a = arm(bars())
    assert a['level'] == 10.0 and a['stop'] == 10.3 and a['trigger'] == pytest.approx(9.99)
    assert a['limit'] == pytest.approx(10.0 * 0.9985)


def test_consolidation_boundary():
    """A high above LOD x (1 + 4 %) inside bars j-4..j disarms (mirror: high, not low)."""
    assert arm(bars(high_max=10.0 * 1.04 - 0.001)) is not None
    assert arm(bars(high_max=10.0 * 1.04 + 0.001)) is None


def test_min_dist_boundary():
    """Level must be <= open x 0.95 (mirror of >= open x 1.05)."""
    assert arm(bars(o0=10.0 / 0.9499)) is not None
    assert arm(bars(o0=10.0 / 0.9501)) is None


def test_rv_band_boundary_uses_volume_through_j_only():
    """Same rv mechanism as the long side — volume/rv logic is side-agnostic."""
    assert arm(bars(), adv=7000 / 0.02 / 1.001) is not None
    assert arm(bars(), adv=7000 / 0.02 / 0.999) is None
    assert arm(bars(), adv=7000 / 0.02 / 4.999) is not None
    assert arm(bars(), adv=7000 / 0.02 / 5.001) is None
    b = bars(); b.loc[7, 'v'] = 1e9
    assert arm(b) is not None


def test_price_floor_and_min_r():
    """trigger >= floor still gates (admission is about price level, not direction); R = stop - trigger."""
    assert arm(bars(), floor=9.99) is not None and arm(bars(), floor=10.0) is None
    b = bars(high_max=10.05); b.loc[[1, 2, 4, 5, 6], 'h'] = 10.05
    assert arm(b) is None                                          # R = 0.05 / 9.99 < 1 %


def test_crossing_filter_needs_next_bar_at_trigger():
    """Only armed bars whose next bar's LOW reaches level - 1c are candidates (mirror of high)."""
    assert [c['j'] for c in c1439.armed_crossing_bars_low(bars(next_low=9.99), 175000.0, P, 0.0)] == [6]
    assert c1439.armed_crossing_bars_low(bars(next_low=9.995), 175000.0, P, 0.0) == []


def ticks(day, m, rows):
    base = sr.et_ns(day, m * 60)
    return pd.DataFrame({'ts': [base + int(s * 1e9) for s, _, _ in rows], 'price': [p for _, p, _ in rows],
                         'size': [float(z) for _, _, z in rows]}).astype({'ts': 'int64'})


def quotes(day, m, rows):
    base = sr.et_ns(day, m * 60)
    return pd.DataFrame({'ts': [base + int(s * 1e9) for s, _, _ in rows], 'bid': [b for _, b, _ in rows],
                         'ask': [a for _, _, a in rows]}).astype({'ts': 'int64'})


ARM = dict(level=10.0, trigger=9.99, limit=10.0 * 0.9985, stop=10.3)


def test_rearm_after_no_cross_and_one_fill_per_day():
    """First armed window has no print <= trigger -> next armed window fills (mirror)."""
    c1 = dict(ARM, m_lo=600, m_hi=601, cumv_j=0.0)
    c2 = dict(ARM, m_lo=610, m_hi=611, cumv_j=0.0)
    tapes = {601: (ticks(DAY, 601, [(10, 10.00, 100)]), quotes(DAY, 600, [(50, 10.00, 10.01)])),
             611: (ticks(DAY, 611, [(10, 9.98, 100)]), quotes(DAY, 610, [(50, 9.99, 9.995)]))}
    status, a, e = c1439.resolve_day_low([c1, c2], lambda a: tapes[a['m_hi']], DAY)
    assert status == 'fill' and a['m_hi'] == 611 and e['fill'] == pytest.approx(9.99)


def test_bid_below_limit_is_nofill_and_unusable_window_stops_the_day():
    """bid < limit -> nofill (mirror of ask > limit); no print in a crossing window -> no_tape."""
    c1 = dict(ARM, m_lo=600, m_hi=601, cumv_j=0.0)
    low_bid = (ticks(DAY, 601, [(10, 9.98, 100)]), quotes(DAY, 600, [(50, 9.97, 9.98)]))
    assert c1439.resolve_day_low([c1], lambda a: low_bid, DAY)[0] == 'nofill'
    empty = (ticks(DAY, 601, []), quotes(DAY, 600, [(50, 9.98, 9.99)]))
    assert c1439.resolve_day_low([c1], lambda a: empty, DAY)[0] == 'no_tape'


def test_fill_records_bid_as_price_and_stopped_flag():
    """Fill price is the BID (not ask); a later print >= stop sets stopped=True."""
    c1 = dict(ARM, m_lo=600, m_hi=601, cumv_j=0.0)
    s, e_ns = c1439.ca.window_ns(DAY, 600, 601)
    t = ticks(DAY, 601, [(5, 9.98, 100), (20, 10.31, 100)])
    q = quotes(DAY, 600, [(1, 9.99, 9.995)])
    r = c1439.resolve_window_low(t, q, s, e_ns, c1)
    assert r['status'] == 'fill' and r['fill'] == pytest.approx(9.99) and r['stopped'] is True


def test_walk_path_short_stop_first_and_target():
    """Mirror of sip_rebuild.walk_path: a bar touching both stop and target is STOP-first; gap-through at open."""
    path = pd.DataFrame({'m': [601, 602], 'o': [10.0, 10.0], 'h': [10.31, 10.0], 'l': [9.5, 9.4], 'c': [9.6, 9.5]})
    m, px, why = c1439.walk_path_short(fill=9.98, stop=10.3, target=9.38, path=path)
    assert (m, why) == (601, 'stop') and px == pytest.approx(10.3)  # h>=stop and l<=target both true on bar 601
    path2 = pd.DataFrame({'m': [601], 'o': [10.0], 'h': [10.1], 'l': [9.3], 'c': [9.5]})
    m2, px2, why2 = c1439.walk_path_short(fill=9.98, stop=10.3, target=9.38, path=path2)
    assert (m2, why2, px2) == (601, 'target', pytest.approx(9.38))


def test_trade_result_short_sign():
    """Short: R = stop - fill; raw R positive when price falls below fill."""
    raw, cost, net, R = c1439.trade_result_short(fill=9.98, stop=10.3, half_entry=0.005, exit_px=9.6, exit_half=0.005)
    assert R == pytest.approx(0.32) and raw == pytest.approx((9.98 - 9.6) / 0.32) and net < raw


def test_population_filter_never_conditions_on_a_sub_floor_low(tmp_path, monkeypatch):
    """9/25 VOID: `low <= floor - 1c` selected days that fell below the floor AFTER the entry (look-ahead).
    The only admissible daily-bar floor condition is the necessary one: day high >= floor + 1c."""
    import pandas as pd
    import cell_1439 as c
    rows = [dict(symbol='KEEP1', bar_date='2025-08-01', open=30.0, high=31.0, low=15.0, adv20=1e6),   # fell through $20: KEPT (necessary cond. holds)
            dict(symbol='KEEP2', bar_date='2025-08-01', open=30.0, high=31.0, low=25.0, adv20=1e6),   # never near the floor: KEPT (was wrongly dropped)
            dict(symbol='DROP1', bar_date='2025-08-01', open=18.0, high=19.5, low=15.0, adv20=1e6),   # high < floor: no trigger >= $20 possible
            dict(symbol='DROP2', bar_date='2025-08-01', open=30.0, high=31.0, low=29.0, adv20=1e6)]   # never 5 % below the open
    f = tmp_path / 'u.csv'; pd.DataFrame(rows).to_csv(f, index=False)
    monkeypatch.setattr(c, 'UNIVERSE_CSV', str(f))
    p, floor, min_adv = c.ca.live_params()
    got = set(c.load_population_low(p, 20.0, 1e5).symbol)
    assert got == {'KEEP1', 'KEEP2'}, got
