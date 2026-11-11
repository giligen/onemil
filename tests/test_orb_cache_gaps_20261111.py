"""Opening-range bar-gap gate + idempotent SIP refill (spec 2026-11-11)."""
import logging
import os
import sqlite3

import pandas as pd
import pytest

from batch.orb_bar_gap_gate import gapped_pairs, refill_pairs, run_gate, window_utc

DAY = '2026-11-04'  # EST (UTC-5): 09:30 ET = 14:30 UTC


def _conn(minutes_by_sym):
    c = sqlite3.connect(':memory:')
    c.execute("CREATE TABLE intraday_bars_1min (symbol TEXT NOT NULL, bar_date TEXT NOT NULL, timestamp TEXT NOT NULL,"
              " open REAL, high REAL, low REAL, close REAL, volume INTEGER, PRIMARY KEY (symbol, timestamp))")
    for sym, mins in minutes_by_sym.items():
        for m in mins:
            c.execute("INSERT INTO intraday_bars_1min VALUES (?,?,?,?,?,?,?,?)",
                      (sym, DAY, f"{DAY}T14:{30 + m:02d}:00+00:00", 1, 1, 1, 1, 100))
    c.commit()
    return c


class _FakeAlpaca:
    """Returns the full 5 minutes for every symbol except those listed as no-trade."""
    def __init__(self, no_trade=()):
        self.no_trade, self.calls = set(no_trade), []

    def get_historical_1min_bars(self, sym, lo, hi):
        self.calls.append((sym, lo, hi))
        if sym in self.no_trade:
            return pd.DataFrame()
        ts = [pd.Timestamp(lo) + pd.Timedelta(minutes=i) for i in range(5)]
        return pd.DataFrame({'timestamp': ts, 'open': 2.0, 'high': 2.0, 'low': 2.0, 'close': 2.0, 'volume': 50})


def test_window_is_dst_correct():
    assert window_utc('2026-10-26')[0].hour == 13 and window_utc(DAY)[0].hour == 14


def test_gate_counts_complete_partial_empty():
    c = _conn({'FULL': range(5), 'PART': [0, 2, 4], 'ONE': [0]})
    got = gapped_pairs(c, [('FULL', DAY), ('PART', DAY), ('ONE', DAY), ('NONE', DAY)])
    assert sorted(got) == [('NONE', DAY, 0), ('ONE', DAY, 1), ('PART', DAY, 3)]


def test_refill_inserts_only_missing_minutes_and_is_idempotent():
    c = _conn({'PART': [0, 2, 4]})
    fa = _FakeAlpaca()
    r1 = refill_pairs(c, fa, [('PART', DAY, 3)])
    assert r1['bars_added'] == 2 and r1['refilled_pairs'] == 1
    assert c.execute("select count(*) from intraday_bars_1min").fetchone()[0] == 5
    # existing rows untouched (volume 100, not the fake's 50)
    assert c.execute("select volume from intraday_bars_1min where timestamp like '%14:30:00%'").fetchone()[0] == 100
    assert fa.calls[0][2] - fa.calls[0][1] == pd.Timedelta(minutes=5).to_pytimedelta() - pd.Timedelta(seconds=1).to_pytimedelta()
    assert refill_pairs(c, fa, [('PART', DAY, 5)])['bars_added'] == 0


def test_no_trade_minutes_are_not_a_vendor_gap(caplog):
    c = _conn({'THIN': [0, 3]})
    with caplog.at_level(logging.WARNING):
        res = run_gate(c, _FakeAlpaca(no_trade=['THIN']), [('THIN', DAY)])
    assert res['gapped'] == 1 and res['refilled_pairs'] == 0 and res['real_gap_share_latest'] == 0
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


def test_error_only_when_over_ten_percent_really_missing(caplog):
    syms = {f'S{i}': range(5) for i in range(18)}
    syms.update({'A': [0, 1], 'B': [0, 1]})  # 2/20 = 10 % -> not > 10 %
    pairs = [(s, DAY) for s in syms]
    with caplog.at_level(logging.WARNING):
        run_gate(_conn(syms), _FakeAlpaca(), pairs)
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)
    caplog.clear()
    syms['C'] = [0]  # 3/21 = 14 %
    with caplog.at_level(logging.WARNING):
        run_gate(_conn(syms), _FakeAlpaca(), [(s, DAY) for s in syms])
    assert any(r.levelno == logging.ERROR and 'VENDOR DEGRADATION' in r.getMessage() for r in caplog.records)


def test_warning_lists_first_20_names(caplog):
    syms = {f'G{i:02d}': [0] for i in range(25)}
    with caplog.at_level(logging.WARNING):
        run_gate(_conn(syms), _FakeAlpaca(no_trade=list(syms)), [(s, DAY) for s in syms])
    msg = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING][0]
    assert '25/25' in msg and 'G19@' in msg and 'G20@' not in msg


@pytest.mark.skipif(not os.environ.get('ALPACA_API_KEY') and not os.path.exists('/home/ec2-user/onemil/.env'),
                    reason='needs Alpaca keys / network')
def test_integration_real_sip_refill_two_symbol_days():
    """Real SIP: copy ATYR 2026-10-29 / DPC 2026-10-27 opening rows from the cache, delete one minute each, refill."""
    from dotenv import load_dotenv
    from data_sources.alpaca_client import AlpacaClient
    load_dotenv('/home/ec2-user/onemil/.env')
    if not os.environ.get('ALPACA_API_KEY'):
        pytest.skip('no keys')
    prod = sqlite3.connect('file:/home/ec2-user/onemil/data/cache.db?mode=ro', uri=True)
    t = sqlite3.connect(':memory:')
    t.execute("CREATE TABLE intraday_bars_1min (symbol TEXT NOT NULL, bar_date TEXT NOT NULL, timestamp TEXT NOT NULL,"
              " open REAL, high REAL, low REAL, close REAL, volume INTEGER, PRIMARY KEY (symbol, timestamp))")
    pairs = [('ATYR', '2026-10-29'), ('DPC', '2026-10-27')]
    for s, d in pairs:
        lo, hi = window_utc(d)
        rows = prod.execute("select * from intraday_bars_1min where symbol=? and bar_date=? and timestamp>=? and timestamp<?",
                            (s, d, lo.strftime('%Y-%m-%dT%H:%M:%S+00:00'), hi.strftime('%Y-%m-%dT%H:%M:%S+00:00'))).fetchall()
        if len(rows) < 2:
            pytest.skip('cache rows unavailable')
        t.executemany("insert into intraday_bars_1min values (?,?,?,?,?,?,?,?)", rows)
        t.execute("delete from intraday_bars_1min where symbol=? and timestamp=?", (s, rows[0][2]))
    t.commit()
    n_before = t.execute("select count(*) from intraday_bars_1min").fetchone()[0]
    cl = AlpacaClient(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_API_SECRET'], paper=True)
    res = run_gate(t, cl, pairs)
    assert res['refilled_pairs'] == 2 and res['bars_added'] == 2
    assert t.execute("select count(*) from intraday_bars_1min").fetchone()[0] == n_before + 2
    assert run_gate(t, cl, pairs)['bars_added'] == 0
