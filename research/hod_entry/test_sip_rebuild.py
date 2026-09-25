"""Unit tests for sip_rebuild.py on synthetic ticks (cell 1,427 independent rebuild)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sip_rebuild as sr  # noqa: E402

S = sr.et_ns('2026-02-02', 600 * 60)         # entry bar 10:00 ET -> break bar 09:59
SEC = 10**9


def tr(*rows):
    """Trades frame from (seconds-before-S, price) pairs."""
    return pd.DataFrame({'ts': [S - int(s * SEC) for s, _ in rows], 'price': [p for _, p in rows],
                         'size': [100.0] * len(rows)}).astype({'ts': 'int64'})


def qu(*rows):
    """Quotes frame from (seconds-before-S, bid, ask) triples."""
    return pd.DataFrame({'ts': [S - int(s * SEC) for s, _, _ in rows], 'bid': [b for _, b, _ in rows],
                         'ask': [a for _, _, a in rows]}).astype({'ts': 'int64'})


def test_et_ns_is_utc_of_et_clock():
    """10:00 ET on a winter day is 15:00 UTC."""
    assert pd.Timestamp(S, unit='ns', tz='UTC') == pd.Timestamp('2026-02-02 15:00', tz='UTC')
    assert sr.ns_to_et_minutes(S, '2026-02-02') == pytest.approx(600.0)


def test_hod_level_uses_rth_bars_before_break_bar_only():
    """Pre-market and the break bar itself are excluded from the level."""
    bars = pd.DataFrame({'m': [500, 570, 571, 572, 573], 'high': [99.0, 10.0, 10.5, 10.2, 12.0]})
    assert sr.hod_level(bars, 573) == 10.5
    assert np.isnan(sr.hod_level(bars, 570))


def test_fill_at_prevailing_ask_when_within_limit():
    """First print >= trigger fills at the last quote's ask at or before it."""
    t = tr((50, 9.99), (40, 10.01), (30, 10.05))
    q = qu((62, 9.98, 10.00), (45, 9.99, 10.01), (35, 10.04, 10.06))
    e = sr.simulate_entry(t, q, S, level=10.00, stop=9.80)
    assert e['status'] == 'fill' and e['fill'] == pytest.approx(10.01)
    assert e['half_spread'] == pytest.approx(0.01)
    assert e['fill_ts'] == S - 40 * SEC and not e['stopped_bb']


def test_no_fill_when_ask_above_limit():
    """Ask beyond level x 1.0015 at the trigger print -> no position (no later fill)."""
    t = tr((40, 10.02), (20, 10.01))
    q = qu((50, 10.00, 10.03), (21, 10.00, 10.01))
    e = sr.simulate_entry(t, q, S, level=10.00, stop=9.80)
    assert e['status'] == 'nofill' and np.isnan(e['fill'])


def test_no_fill_when_no_print_reaches_trigger():
    """Prints at the level (not level + 1 tick) do not trigger."""
    e = sr.simulate_entry(tr((40, 10.00)), qu((50, 9.99, 10.00)), S, level=10.00, stop=9.80)
    assert e['status'] == 'nofill'


def test_prints_outside_break_bar_are_ignored():
    """A print in the 5 s quote look-back or at/after S never triggers."""
    t = tr((62, 10.50), (0, 10.50), (30, 9.90))
    q = qu((63, 10.00, 10.01))
    t.loc[1, 'ts'] = S                         # exactly S is outside [S-60, S)
    assert sr.simulate_entry(t, q, S, level=10.00, stop=9.80)['status'] == 'nofill'


def test_unusable_tape_without_prints_or_quotes():
    """No break-bar print, or no valid quote -> no_tape (availability rail)."""
    assert sr.simulate_entry(tr((62, 10.2)), qu((50, 10, 10.01)), S, 10.0, 9.8)['status'] == 'no_tape'
    assert sr.simulate_entry(tr((30, 10.2)), qu((50, 0, 0)), S, 10.0, 9.8)['status'] == 'no_tape'
    empty_q = qu()
    assert sr.simulate_entry(tr((30, 10.2)), empty_q, S, 10.0, 9.8)['status'] == 'no_tape'


def test_trigger_without_prevailing_quote_is_unusable():
    """Quotes exist only after the triggering print -> no_tape, not a fill."""
    assert sr.simulate_entry(tr((40, 10.05)), qu((20, 10.0, 10.01)), S, 10.0, 9.8)['status'] == 'no_tape'


def test_stop_inside_break_bar_after_fill():
    """A print <= stop after the fill inside the break bar marks stopped_bb."""
    t = tr((40, 10.01), (10, 9.79))
    e = sr.simulate_entry(t, qu((50, 10.0, 10.01)), S, level=10.00, stop=9.80)
    assert e['status'] == 'fill' and e['stopped_bb']


def test_extra_tick_and_limit_variants():
    """ask + 1 tick moves the price; a 5 bps limit rejects an ask the 15 bps limit accepts."""
    t, q = tr((40, 10.01)), qu((50, 10.0, 10.01))
    assert sr.simulate_entry(t, q, S, 10.0, 9.8, extra_ticks=1)['fill'] == pytest.approx(10.02)
    assert sr.simulate_entry(t, qu((50, 10.0, 10.012)), S, 10.0, 9.8, limit_bps=0.0005)['status'] == 'nofill'
    assert sr.simulate_entry(t, qu((50, 10.0, 10.012)), S, 10.0, 9.8)['status'] == 'fill'


def path(rows):
    """Path frame from (m, o, h, l, c) tuples."""
    return pd.DataFrame(rows, columns=['m', 'o', 'h', 'l', 'c'])


def test_walk_path_rules():
    """Gap-through, stop, target, stop-first on both, 15:55 open."""
    assert sr.walk_path(10, 9, 12, path([(600, 8.5, 9.5, 8.4, 9)])) == (600, 8.5, 'stop')
    assert sr.walk_path(10, 9, 12, path([(600, 10, 10.5, 8.9, 9)])) == (600, 9.0, 'stop')
    assert sr.walk_path(10, 9, 12, path([(600, 10, 12.5, 8.9, 9)])) == (600, 9.0, 'stop')
    assert sr.walk_path(10, 9, 12, path([(600, 10, 12.5, 9.5, 12)])) == (600, 12.0, 'target')
    assert sr.walk_path(10, 9, 12, path([(600, 10, 11, 9.5, 10), (955, 10.7, 11, 10.5, 10.8)])) == \
        (955, 10.7, 'eod')


def test_trade_result_cost_convention():
    """Entry half-spread + exit half-spread + 2 bp of exit, all in the new R."""
    raw, cost, net, R = sr.trade_result(fill=10.0, stop=9.8, half_entry=0.01, exit_px=10.4, exit_half=0.02)
    assert R == pytest.approx(0.2) and raw == pytest.approx(2.0)
    assert cost == pytest.approx((0.01 + 0.02 + 0.0002 * 10.4) / 0.2)
    assert net == pytest.approx(raw - cost)


def test_needs_quote_history_only_without_quote_before_break_bar():
    """History is needed iff no valid NBBO at/before S-60 s in the cached window."""
    assert not sr.needs_quote_history(qu((62, 10.0, 10.01)), S)
    assert sr.needs_quote_history(qu((50, 10.0, 10.01)), S)
    assert sr.needs_quote_history(qu((62, 0.0, 10.01)), S)          # invalid quote does not count


def test_history_quote_becomes_prevailing_for_thin_name():
    """A prepended older quote makes a trigger print with no in-window quote usable, at the old ask."""
    t = tr((40, 10.01))
    q_old = qu((20, 10.0, 10.01))
    assert sr.simulate_entry(t, q_old, S, 10.0, 9.8)['status'] == 'no_tape'
    hist = pd.DataFrame({'ts': [S - 300 * SEC], 'bid': [9.99], 'ask': [10.012]})
    q = pd.concat([hist, q_old], ignore_index=True)
    e = sr.simulate_entry(t, q, S, 10.0, 9.8)
    assert e['status'] == 'fill' and e['fill'] == pytest.approx(10.012)


def test_history_never_changes_a_previously_usable_signal():
    """An older quote never supersedes an in-window quote at or before the fill instant."""
    t, q = tr((40, 10.01)), qu((50, 9.99, 10.01))
    hist = pd.DataFrame({'ts': [S - 300 * SEC], 'bid': [9.0], 'ask': [9.5]})
    a = sr.simulate_entry(t, q, S, 10.0, 9.8)
    b = sr.simulate_entry(t, pd.concat([hist, q], ignore_index=True), S, 10.0, 9.8)
    assert a == b
