"""Unit tests for cell_1481.py -- synthetic bars/tape, no network.

Covers: the strictly-below fill rule (a print exactly at the limit does not fill), the window
boundary at the Nth minute (a candidate bar beyond the cutoff is never scanned), stop/target
resolution inside the retest minute from the tape, target computed from the NEW entry (not the
base fill), and no-fill when no print ever trades below the limit."""
import os
import sys
import pickle

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell_1481 as c1481


def bars(rows):
    """rows: list of (m, o, h, l, c)."""
    return pd.DataFrame(rows, columns=['m', 'o', 'h', 'l', 'c']).astype(float).assign(
        m=lambda d: d.m.astype(int))


def tape(rows):
    """rows: list of (ts, price)."""
    df = pd.DataFrame(rows, columns=['ts', 'price'])
    df['size'] = 100.0
    return df.astype({'ts': 'int64', 'price': 'float64'})


# ------------------------------------------------------------------------------------------- candidate window
def test_find_candidate_minutes_includes_m_break_and_respects_cutoff():
    b = bars([(100, 10, 10.2, 9.9, 10.0),     # m_break itself: low 9.9 <= limit 9.95 -> candidate
              (101, 10, 10.1, 10.0, 10.05),   # no dip
              (115, 10, 10.1, 9.8, 9.9),      # exactly at the 15-min cutoff -> candidate
              (116, 10, 10.1, 9.7, 9.8)])     # one past the cutoff -> excluded even though it dips
    cands = c1481.find_candidate_minutes(m_break=100, bars=b, window_min=15, limit=9.95, eod_m=955)
    assert cands == [100, 115]


def test_find_candidate_minutes_no_dip_returns_empty():
    b = bars([(100, 10, 10.2, 9.96, 10.0), (101, 10, 10.1, 9.97, 10.05)])
    cands = c1481.find_candidate_minutes(m_break=100, bars=b, window_min=15, limit=9.95, eod_m=955)
    assert cands == []


# ------------------------------------------------------------------------------------------- fill rule (resolve_retest)
def test_strictly_below_fills_at_the_print_price():
    """A print strictly below the limit fills; an earlier print exactly AT the limit must not."""
    limit = 9.95
    t_hit_ns = 1_000_000_000

    def fetch_fn(symbol, day, m):
        return tape([(t_hit_ns + 10, 9.95),      # at the limit -- not a fill
                     (t_hit_ns + 20, 9.94)]), None  # strictly below -- fills here
    res = c1481.resolve_retest('AAPL', '2026-01-05', 100, t_hit_ns, [100], limit, fetch_fn=fetch_fn)
    assert res['status'] == 'fill'
    assert res['fill_px'] == pytest.approx(9.94)
    assert res['fill_ts'] == t_hit_ns + 20
    assert res['at_or_below_ts'] == t_hit_ns + 10          # report-only at-or-below tracked separately


def test_at_limit_only_never_fills_primary_book():
    limit = 9.95
    t_hit_ns = 1_000_000_000

    def fetch_fn(symbol, day, m):
        return tape([(t_hit_ns + 10, 9.95)]), None         # exactly at the limit, no lower print

    res = c1481.resolve_retest('AAPL', '2026-01-05', 100, t_hit_ns, [100], limit, fetch_fn=fetch_fn)
    assert res['status'] == 'bar_tick_disagree'
    assert res['at_or_below_ts'] == t_hit_ns + 10


def test_prints_before_the_fill_instant_are_ignored_in_the_fill_bar():
    """A dip that happened BEFORE the base fill instant (earlier in the same m_break bar) must not
    count -- only ts > t_hit_ns is eligible."""
    limit = 9.95
    t_hit_ns = 1_000_000_000

    def fetch_fn(symbol, day, m):
        return tape([(t_hit_ns - 500, 9.80),               # before the fill -- ignored
                     (t_hit_ns + 500, 9.99)]), None         # after, but not below the limit

    res = c1481.resolve_retest('AAPL', '2026-01-05', 100, t_hit_ns, [100], limit, fetch_fn=fetch_fn)
    assert res['status'] == 'bar_tick_disagree'


def test_no_candidates_is_never_retest():
    res = c1481.resolve_retest('AAPL', '2026-01-05', 100, 0, [], 9.95)
    assert res['status'] == 'never_retest'
    assert res['n_fetched'] == 0


def test_falls_through_to_next_candidate_minute_on_disagreement():
    limit = 9.95
    t_hit_ns = 1_000_000_000
    calls = []

    def fetch_fn(symbol, day, m):
        calls.append(m)
        if m == 100:
            return tape([(t_hit_ns + 5, 9.95)]), None       # at the limit only -- no fill here
        return tape([(t_hit_ns + 1000, 9.90)]), None        # next candidate minute: a real fill

    res = c1481.resolve_retest('AAPL', '2026-01-05', 100, t_hit_ns, [100, 102], limit, fetch_fn=fetch_fn)
    assert res['status'] == 'fill'
    assert res['m_retest'] == 102
    assert calls == [100, 102]


def test_fetch_error_is_reported_not_silently_dropped():
    def fetch_fn(symbol, day, m):
        raise RuntimeError('rate limited')
    res = c1481.resolve_retest('AAPL', '2026-01-05', 100, 0, [100], 9.95, fetch_fn=fetch_fn)
    assert res['status'] == 'fetch_error'


# ------------------------------------------------------------------------------------------- path pricing
def test_target_is_computed_from_the_new_entry_not_the_base_fill():
    """entry=9.94 (the retest fill, NOT the base's original higher fill), stop=9.50 (base
    consolidation low, unchanged) -> R'=0.44, target = 9.94 + 2*0.44 = 10.82."""
    retest = dict(fill_px=9.94, fill_ts=100, m_retest=100, dip_low=9.90)

    def fetch_fn(symbol, day, m):
        return tape([(200, 10.00)]), None                    # no stop/target touch in-minute

    b = bars([(101, 10.0, 10.0, 9.99, 9.995)])                # walk_path: closes flat, eod fallback path
    res = c1481.walk_retest_path(stop=9.50, retest=retest, bars=b, split='VAL', eod_m=101, limit=9.94,
                                  fetch_fn=fetch_fn, symbol='AAPL', day='2026-01-05')
    assert res is not None
    assert res['target'] == pytest.approx(9.94 + 2 * 0.44)
    assert res['Rp'] == pytest.approx(0.44)


def test_stop_hit_inside_the_retest_minute_from_the_tape():
    retest = dict(fill_px=10.00, fill_ts=1000, m_retest=100, dip_low=9.95)

    def fetch_fn(symbol, day, m):
        return tape([(1000, 10.00), (1500, 9.40), (2000, 12.0)]), None  # stop print comes first

    res = c1481.walk_retest_path(stop=9.50, retest=retest, bars=bars([]), split='VAL', eod_m=200,
                                  limit=10.00, fetch_fn=fetch_fn, symbol='AAPL', day='2026-01-05')
    assert res['why'] == 'stop'
    assert res['exit_price'] == pytest.approx(9.50)
    assert res['exit_m'] == 100


def test_target_hit_inside_the_retest_minute_from_the_tape():
    retest = dict(fill_px=10.00, fill_ts=1000, m_retest=100, dip_low=9.95)
    target = 10.00 + 2 * (10.00 - 9.50)                       # = 11.00

    def fetch_fn(symbol, day, m):
        return tape([(1000, 10.00), (1500, 11.05), (2000, 8.0)]), None  # target print first

    res = c1481.walk_retest_path(stop=9.50, retest=retest, bars=bars([]), split='VAL', eod_m=200,
                                  limit=10.00, fetch_fn=fetch_fn, symbol='AAPL', day='2026-01-05')
    assert res['why'] == 'target'
    assert res['exit_price'] == pytest.approx(target)


def test_zero_or_negative_risk_returns_none():
    """limit at/below the consolidation low -- a degenerate retest, must be dropped, not costed."""
    retest = dict(fill_px=9.40, fill_ts=1000, m_retest=100, dip_low=9.35)

    def fetch_fn(symbol, day, m):
        return tape([]), None
    res = c1481.walk_retest_path(stop=9.50, retest=retest, bars=bars([]), split='VAL', eod_m=200,
                                  limit=9.40, fetch_fn=fetch_fn, symbol='AAPL', day='2026-01-05')
    assert res is None


def test_stop_takes_priority_on_a_tie():
    retest = dict(fill_px=10.00, fill_ts=1000, m_retest=100, dip_low=9.95)

    def fetch_fn(symbol, day, m):
        return tape([(1000, 10.00), (1500, 9.40)]), None      # a single print both <= stop; no separate target print
    res = c1481.walk_retest_path(stop=9.50, retest=retest, bars=bars([]), split='VAL', eod_m=200,
                                  limit=10.00, fetch_fn=fetch_fn, symbol='AAPL', day='2026-01-05')
    assert res['why'] == 'stop'


# ------------------------------------------------------------------------------------------- limit-price convention
def test_entry_books_at_the_limit_not_the_print():
    """PREREG obtainability: the resting buy limit fills AT THE LIMIT on a strict-below print, not
    at the print itself. A print at 9.97 with limit 9.99 must book entry 9.99, not 9.97 (the print
    is kept only as the report-only `print_px`)."""
    limit = 9.99
    t_hit_ns = 1_000_000_000

    def fetch_fn(symbol, day, m):
        return tape([(t_hit_ns + 10, 9.97)]), None            # strictly below the limit

    retest = c1481.resolve_retest('AAPL', '2026-01-05', 100, t_hit_ns, [100], limit, fetch_fn=fetch_fn)
    assert retest['status'] == 'fill'
    assert retest['fill_px'] == pytest.approx(9.97)            # the raw print

    def path_fetch_fn(symbol, day, m):
        return tape([(retest['fill_ts'] + 10, 10.50)]), None   # no stop/target touch in-minute

    res = c1481.walk_retest_path(stop=9.00, retest=retest, bars=bars([]), split='VAL', eod_m=200,
                                  limit=limit, fetch_fn=path_fetch_fn, symbol='AAPL', day='2026-01-05')
    assert res['entry'] == pytest.approx(9.99)                 # the LIMIT, not the 9.97 print
    assert res['print_px'] == pytest.approx(9.97)
    assert res['Rp'] == pytest.approx(9.99 - 9.00)
    assert res['target'] == pytest.approx(9.99 + 2 * (9.99 - 9.00))


# ------------------------------------------------------------------------------------------- cache crash-tolerance
def test_corrupt_pickle_falls_back_to_refetch_and_rewrites_atomically(tmp_path, monkeypatch):
    """A concurrent writer (rebuild_1481.py, same cache dir) can leave a zero-size/partial pickle
    mid-write. The reader must not crash with EOFError -- it logs a WARNING, treats the key as not
    cached, re-fetches via the injected fetch, and the rewritten file must load cleanly."""
    monkeypatch.setattr(c1481, 'CACHE_DIR_1481', str(tmp_path))
    monkeypatch.setattr(c1481, 'CACHE_DIR_1480', str(tmp_path / 'nonexistent_1480'))
    cp = c1481._cache_path('AAPL', '2026-01-05', 100)
    with open(cp, 'wb'):
        pass                                                   # zero-size -- simulates a mid-write race
    assert os.path.getsize(cp) == 0

    calls = []

    def stub_fetch_window(symbol, day, m, m_end):
        calls.append((symbol, day, m, m_end))
        return tape([(1, 9.99)]), tape([(1, 9.99)])

    monkeypatch.setattr(c1481.ca, 'fetch_window', stub_fetch_window)
    t, q = c1481.fetch_minute_tape('AAPL', '2026-01-05', 100)
    assert calls == [('AAPL', '2026-01-05', 100, 101)]          # re-fetched, not silently empty
    assert len(t) == 1 and t.price.iloc[0] == pytest.approx(9.99)

    assert os.path.getsize(cp) > 0                              # rewritten atomically, no .tmp left behind
    assert not any(str(p).endswith(tuple(['.tmp'])) or '.tmp.' in str(p) for p in tmp_path.iterdir())
    with open(cp, 'rb') as f:
        t2, _q2 = pickle.load(f)
    assert len(t2) == 1 and t2.price.iloc[0] == pytest.approx(9.99)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
