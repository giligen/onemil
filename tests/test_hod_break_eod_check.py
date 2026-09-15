"""REPORT §12 live measurables in scripts/hod_break_eod_check.py: synthetic journal lines + a temp trades DB with two
closed rows + a bars dict → every row's number and verdict."""
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('hod_break_eod_check', ROOT / 'scripts' / 'hod_break_eod_check.py')
eod = importlib.util.module_from_spec(spec); sys.modules['hod_break_eod_check'] = eod; spec.loader.exec_module(eod)

DAY = '2026-09-16'
TS = f'{DAY}T14:31:02+0000 host onemil-trader[1]: {DAY} 14:31:02 | INFO | trading.hod_break_engine:1 | '
LINES = [
    TS + '[HOD] ENTRY SUBMITTED AAA level 25.00 limit 25.15 stop 24.50 target 26.20 R 0.60 (2.4%) x166 | +6.1% from open, rv 1.8, spread 12 bps, ask 25.10 order o1',
    TS + '[HOD] ENTRY SUBMITTED BBB level 40.00 limit 40.24 stop 39.20 target 41.80 R 0.90 (2.2%) x111 | +7.0% from open, rv 2.1, spread 20 bps, ask 40.10 order o2',
    TS + '[HOD] CCC: spread 60 bps = 31% of R 0.40 > 15% — skip',
    TS + '[HOD] DDD: ask 30.40 above cap 30.18 — NO CHASE, skip',
    TS + '[HOD] EEE: stop 21.90 within 1.0% of the ask 22.05 — skip',
    TS + '[HOD] FFF: spread 140 bps > 100 — skip',
    TS + '[HOD] GGG: MISSED the spec\'s break at bar 40 (level 33.10, now 3 bars old) — no later break is taken',
    TS + '[HOD] FILLED AAA x166 @ 25.10 after 1.2s (limit 25.15, slip 40 bps vs level, risk $100 vs $100 planned)',
    TS + '[HOD] EXIT AAA target @ 26.30 pnl 199.2 (+2.00R) day 199',
]


def _bars(minutes_opens: dict):
    """(o,h,l,c,v,m) arrays for one symbol from {minute: open}."""
    m = np.array(sorted(minutes_opens)); o = np.array([minutes_opens[k] for k in m], dtype=float)
    return (o, o, o, o, np.ones_like(o), m.astype(int))


@pytest.fixture
def trades_db(tmp_path):
    p = tmp_path / 'trades.db'; con = sqlite3.connect(p)
    con.execute('create table trades (id integer primary key, strategy text, trade_date text, symbol text, fill_price real, entry_price real, stop_loss_price real, '
                'take_profit_price real, exit_price real, exit_reason text, pnl real, shares integer, filled_qty integer, filled_at text, pattern_data text, order_status text)')
    # AAA: fill 25.10 at 10:31 ET (14:31Z), stop 24.50, target hit at 26.30 → +2.0R on 166 shares = +199.2
    con.execute('insert into trades(strategy, trade_date, symbol, fill_price, entry_price, stop_loss_price, take_profit_price, exit_price, exit_reason, pnl, shares, filled_qty, filled_at, pattern_data, order_status) '
                'values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', ('hod_break', DAY, 'AAA', 25.10, 25.15, 24.50, 26.30, 26.30, 'target', 199.2, 166, 166, f'{DAY}T14:31:03+00:00', json.dumps({'quote_ask': 25.10, 'quote_bid': 25.07}), 'closed'))
    # BBB: fill 40.10, stop 39.20, stopped at 39.00 → −1.222R on 111 shares = −122.1; stop fill −51 bps vs the stop
    con.execute('insert into trades(strategy, trade_date, symbol, fill_price, entry_price, stop_loss_price, take_profit_price, exit_price, exit_reason, pnl, shares, filled_qty, filled_at, pattern_data, order_status) '
                'values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', ('hod_break', DAY, 'BBB', 40.10, 40.24, 39.20, 41.90, 39.00, 'stop', -122.1, 111, 111, f'{DAY}T14:31:03+00:00', json.dumps({'quote_ask': 40.10, 'quote_bid': 40.02}), 'closed'))
    # a third row: submitted, never filled (counts against the fill rate)
    con.execute("insert into trades(strategy, trade_date, symbol, entry_price, stop_loss_price, order_status) values ('hod_break', ?, 'ZZZ', 10.06, 9.90, 'time_stop_canceled')", (DAY,))
    con.commit(); con.close(); return p


BARS = {'AAA': _bars({630: 25.00, 631: 25.05, 955: 25.90}), 'BBB': _bars({630: 40.00, 631: 40.00, 955: 39.50})}


def _rows(lines, dry, db):
    return {r[0]: r for r in eod.live_measurables(lines, DAY, dry, db, BARS, 955, since=DAY)}


def test_journal_rows_live(trades_db):
    r = _rows(LINES, False, trades_db)
    assert r[1][2] == '2/3 = 67%' and r[1][4] == 'WATCH n<20'
    assert r[2][2] == '6 today' and r[2][4] == 'OK'          # 2 ordered + gate + nochase + rmin + spread100
    assert r[14][2].startswith('1 MISSED') and r[14][4] == 'ESCALATE'
    assert r[15][2] == 'median 16 bps over 2' and r[15][4] == 'OK'
    assert r[16][2] == '1/6 = 17%' and r[16][4] == 'OK'


def test_db_rows_live(trades_db):
    r = _rows(LINES, False, trades_db)
    assert r[3][2].startswith('2 today · rolling 2.0/day over 1 sessions') and r[3][4] == 'WATCH'
    assert r[4][2] == '2/3 today = 67%' and r[4][4] == 'ESCALATE'
    # entry fill vs the 10:31 open: AAA 25.10/25.05 = +20 bps, BBB 40.10/40.00 = +25 bps → mean +22 → WATCH
    assert r[5][2] == 'median +22 mean +22 bps n=2' and r[5][4] == 'WATCH'
    assert r[6][2] == 'median +22 mean +22 bps n=2' and r[6][4] == 'WATCH'
    assert r[7][2] == '50% of 2 closed' and r[7][4] == 'WATCH n<20'
    assert r[8][2] == '50% / 0%'
    assert r[9][2] == 'median -51 mean -51 bps n=1' and r[9][4] == 'ESCALATE'
    assert r[10][2] == 'no rows' and r[10][4] == 'n/a'
    assert r[11][2] == '+0.389R over 2 trades' and r[11][4] == 'WATCH n<30'   # (+2.000 − 1.222) / 2
    assert r[12][2] == '50% over 2' and r[12][4] == 'WATCH n<30'
    assert r[13][2] == 'this week +0.8R · worst week +0.8R' and r[13][4] == 'OK'


def test_dry_mode_rows_na(trades_db):
    r = _rows(LINES, True, trades_db)
    assert all(r[k][2] == 'n/a dry' and r[k][4] == 'n/a' for k in range(3, 14))
    assert r[1][4] == 'WATCH n<20' and r[2][4] == 'OK' and r[16][4] == 'OK'


def test_empty_day(trades_db):
    r = _rows([], False, trades_db)
    assert r[1][2] == 'no signal reached the gate' and r[2][4] == 'ESCALATE none reached'
    assert r[16][2] == 'no signal reached' and r[16][4] == 'n/a' and r[15][4] == 'n/a'


def test_no_parentheses_in_output(trades_db, capsys):
    eod.print_measurables(eod.live_measurables(LINES, DAY, False, trades_db, BARS, 955, since=DAY))
    out = capsys.readouterr().out
    assert 'LIVE MEASURABLES vs SPEC' in out and '(' not in out and out.count('\n') == 18
