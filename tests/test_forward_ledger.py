"""Tests for research/day_breadth/forward_ledger.py (FORWARD_SPEC.md)."""
import sqlite3
import sys

import pandas as pd
import pytest

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, f'{ROOT}/research/day_breadth')
sys.path.insert(0, f'{ROOT}/research/hod_consol')
import forward_ledger as fl  # noqa: E402


def _make_daily_bars_db(path, rows):
    """rows: (symbol, bar_date, open, high, low, close, volume); seeds a matching
    intraday_bars_1min row per (symbol, bar_date) so load_broad_universe's EXISTS join passes."""
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE daily_bars (symbol TEXT, bar_date TEXT, open REAL, high REAL, '
                'low REAL, close REAL, volume REAL)')
    con.execute('CREATE TABLE intraday_bars_1min (symbol TEXT, bar_date TEXT, timestamp TEXT, volume REAL)')
    con.executemany('INSERT INTO daily_bars VALUES (?,?,?,?,?,?,?)', rows)
    for sym, day, *_ in rows:
        # five 09:30-09:34 ET bars (EDT dates -> 13:3x UTC): the range-computability condition
        # (study_orb_broad, 2026-11-11) replaced the old "any bar" EXISTS join / 15K volume proxy
        for minute in range(30, 35):
            con.execute('INSERT INTO intraday_bars_1min VALUES (?,?,?,?)',
                        (sym, day, f"{day}T13:{minute:02d}:00+00:00", 4000))
    con.commit()
    con.close()


def test_universe_filter_synthetic(tmp_path):
    """gap>=3%, open $3-50, prev_vol>=500K -- symbols failing any one gate are excluded."""
    db = str(tmp_path / 'cache.db')
    day, prev = '2026-09-16', '2026-09-15'
    rows = [
        ('AAAA', prev, 10.0, 10.5, 9.5, 10.0, 1_000_000),
        ('AAAA', day, 10.5, 11.0, 10.0, 10.8, 900_000),   # gap +5% -- qualifies
        ('BBBB', prev, 10.0, 10.5, 9.5, 10.0, 1_000_000),
        ('BBBB', day, 10.1, 10.5, 10.0, 10.3, 900_000),   # gap +1% -- fails gap gate
        ('CCCC', prev, 10.0, 10.5, 9.5, 10.0, 1_000_000),
        ('CCCC', day, 60.0, 61.0, 59.0, 60.5, 900_000),   # huge gap but open > $50 -- fails price
        ('DDDD', prev, 10.0, 10.5, 9.5, 10.0, 400_000),
        ('DDDD', day, 10.6, 11.0, 10.0, 10.8, 900_000),   # prev_vol < 500K -- fails volume gate
    ]
    _make_daily_bars_db(db, rows)
    assert fl.daily_bar_exists(day, db_path=db)
    assert not fl.daily_bar_exists('2026-09-17', db_path=db)
    uni = fl.universe_for_day(day, db_path=db)
    assert uni == ['AAAA']


def test_universe_for_day_restores_study_orb_broad_globals(tmp_path):
    """Regression 11/11: the wide-seed patch must not leak (MIN_GAP_PCT 5.0 -> 3.0 broke the ORB builder tests)."""
    db = str(tmp_path / 'cache.db')
    _make_daily_bars_db(db, [('AAAA', '2026-09-15', 10.0, 10.5, 9.5, 10.0, 1_000_000),
                             ('AAAA', '2026-09-16', 10.5, 11.0, 10.0, 10.8, 900_000)])
    names = ('MIN_GAP_PCT', 'MAX_OPEN_PRICE', 'DATE_START', 'DATE_END')
    before = {n: getattr(fl.broad, n) for n in names}
    fl.universe_for_day('2026-09-16', db_path=db)
    assert {n: getattr(fl.broad, n) for n in names} == before


def test_kept_rule_edge_boundary():
    """kept = BR(signal_m) >= EDGE (0.6115); boundary is inclusive."""
    w = pd.DataFrame(dict(entry_m=[100, 101, 102], BR=[fl.EDGE - 1e-9, fl.EDGE, fl.EDGE + 0.1]))
    out = fl.apply_kept_rule(w)
    assert list(out.kept) == [False, True, True]
    assert list(out.loc[out.kept, 'order_in_day']) == [1, 2]


def test_idempotent_rerun(tmp_path):
    path = str(tmp_path / 'ledger.csv')
    d1 = pd.DataFrame([dict(date='2026-09-16', symbol='AAAA', val=1)])
    d2 = pd.DataFrame([dict(date='2026-09-16', symbol='BBBB', val=2)])
    other = pd.DataFrame([dict(date='2026-09-17', symbol='CCCC', val=3)])
    fl._replace_date_rows(path, '2026-09-16', d1)
    fl._replace_date_rows(path, '2026-09-17', other)
    fl._replace_date_rows(path, '2026-09-16', d2)   # re-run of the same date replaces it
    out = pd.read_csv(path)
    assert sorted(out.date.unique()) == ['2026-09-16', '2026-09-17']
    assert list(out[out.date == '2026-09-16'].symbol) == ['BBBB']
