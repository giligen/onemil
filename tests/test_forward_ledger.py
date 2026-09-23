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
        con.execute('INSERT INTO intraday_bars_1min VALUES (?,?,?,?)',
                    (sym, day, f"{day}T09:00:00+00:00", 20000))
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


@pytest.mark.integration
def test_process_day_cached_test_day():
    """One cached TEST day (bars already in bars_sip.db/bars_rth.db) -- real Alpaca NBBO calls."""
    ok = fl.process_day('2026-06-01', send_telegram=False)
    assert ok is True
    summary = pd.read_csv(fl.SUMMARY_CSV)
    row = summary[summary.date == '2026-06-01']
    assert len(row) == 1
    assert row.n_signals.iloc[0] >= 1
