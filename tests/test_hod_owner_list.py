"""Tests for scripts/hod_owner_list.py and scripts/hod_owner_list_score.py (spec docs/hod_owner_list_spec_20261112.md).

No network: the Alpaca HTTP session is a MagicMock(spec=requests.Session); trades.db / cache.db are real temp
SQLite files so the DB -> rows boundary is exercised for real.
"""
import logging
import os
import sqlite3
import sys
from unittest.mock import MagicMock

import pytest
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import hod_owner_list as hl            # noqa: E402
import hod_owner_list_score as hs      # noqa: E402

DAY = '2026-11-12'


def _dry(ts, sym, level, trig, limit, stop):
    return f'{DAY} {ts} | INFO     | trading.hod_break_engine:1022 | [HOD DRY] ARMED {sym} level {level} trigger {trig} limit {limit} stop {stop}'


def _live(ts, sym, trig, limit):
    return f'{DAY} {ts} | INFO     | trading.hod_break_engine:1871 | [HOD] {sym}: LIVE ARMED stop {trig} limit {limit} qty 67 order abc'


JOURNAL = [
    f'2026-11-12T14:36:00+0000 host onemil-trader[1]: {_dry("14:36:00", "MBGL", "22.63", "22.64", "22.66", "21.90")}',
    f'2026-11-12T14:36:00+0000 host onemil-trader[1]: {_live("14:36:00", "MBGL", "22.64", "22.66")}',
    f'2026-11-12T14:37:00+0000 host onemil-trader[1]: {_dry("14:37:00", "APH", "80.24", "80.25", "80.36", "77.69")}',
    f'2026-11-12T14:50:00+0000 host onemil-trader[1]: {_dry("14:50:00", "MBGL", "22.80", "22.81", "22.84", "22.10")}',
    f'2026-11-12T14:51:00+0000 host onemil-trader[1]: {_live("14:51:00", "ONLYLIVE", "5.10", "5.12")}',
    f'2026-11-12T14:52:00+0000 host onemil-trader[1]: {DAY} 14:52:00 | INFO | x | [R2G DRY] ARMED R2GSYM level 1.00 trigger 1.01 limit 1.02 stop 0.9',
]


# ---------------------------------------------------------------- parsing
def test_parse_arms_latest_wins_and_rearms():
    arms = hl.parse_arms(JOURNAL)
    assert arms['MBGL']['level'] == 22.80 and arms['MBGL']['stop'] == 22.10 and arms['MBGL']['rearms'] == 1
    assert arms['APH']['rearms'] == 0
    assert 'R2GSYM' not in arms                       # other book's tag is not ours


def test_parse_arms_live_only_fallback_and_dry_tag_and_cutoff():
    arms = hl.parse_arms(JOURNAL)
    assert arms['ONLYLIVE']['trigger'] == 5.10 and arms['ONLYLIVE']['level'] is None
    early = hl.parse_arms(JOURNAL, cutoff_utc='2026-11-12 14:40:00')
    assert early['MBGL']['level'] == 22.63 and early['MBGL']['rearms'] == 0 and 'ONLYLIVE' not in early


# ---------------------------------------------------------------- arithmetic
def test_rvol_and_spread_arithmetic():
    # 195 minutes into a 390-minute session = half: expected volume = adv20 * 0.5
    assert hl.rvol(1_000_000, 500_000, 195) == pytest.approx(4.0)
    assert hl.rvol(1000, 0, 195) is None and hl.rvol(None, 5, 195) is None
    assert hl.rvol(100, 100, 0) == pytest.approx(100 / (100 / 390))      # floor of one minute
    assert hl.rvol(100, 100, 9999) == pytest.approx(1.0)                 # capped at the full session
    assert hl.spread_bps(9.99, 10.01) == pytest.approx(20.0, rel=1e-3)
    assert hl.spread_bps(0, 10) is None


# ---------------------------------------------------------------- trades.db boundary
def _make_trades_db(path):
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, trade_date DATE, symbol TEXT, strategy TEXT, '
                'account TEXT, order_status TEXT, fill_price REAL, exit_price REAL, exit_reason TEXT, pnl REAL, total_risk REAL)')
    rows = [
        (DAY, 'MBGL', 'hod_break', 'paper', 'closed', 22.81, 22.10, 'stop_loss', -50.0, 50.0),
        (DAY, 'APH', 'hod_break', 'paper', 'filled', 80.30, None, None, None, 40.0),
        (DAY, 'TGT', 'hod_break', 'paper', 'closed', 10.0, 11.0, 'target', 100.0, 50.0),
        (DAY, 'EODX', 'hod_break', 'paper', 'closed', 10.0, 10.2, 'eod', 20.0, 50.0),
        (DAY, 'CXL', 'hod_break', 'paper', 'canceled', None, None, None, None, 50.0),
        (DAY, 'ORBX', 'orb', 'paper', 'closed', 10.0, 11.0, 'target', 99.0, 50.0),       # other strategy
        ('2026-11-11', 'OLD', 'hod_break', 'paper', 'closed', 10.0, 11.0, 'target', 10.0, 50.0),
    ]
    con.executemany('INSERT INTO trades (trade_date,symbol,strategy,account,order_status,fill_price,exit_price,exit_reason,pnl,total_risk) '
                    'VALUES (?,?,?,?,?,?,?,?,?,?)', rows)
    con.commit(); con.close()


def test_trade_status_from_real_db(tmp_path):
    p = str(tmp_path / 'trades.db'); _make_trades_db(p)
    st = hl.load_trade_status(DAY, p)
    assert st['MBGL']['status'] == 'STOPPED' and st['MBGL']['r'] == pytest.approx(-1.0)
    assert st['APH']['status'] == 'FILLED @80.30' and st['APH']['r'] is None
    assert st['TGT']['status'] == 'TARGET' and st['TGT']['r'] == pytest.approx(2.0)
    assert st['EODX']['status'] == 'EOD'
    assert st['CXL']['status'] == 'ARMED'
    assert 'ORBX' not in st and 'OLD' not in st


def test_adv20_from_real_cache_db(tmp_path):
    p = str(tmp_path / 'cache.db')
    con = sqlite3.connect(p)
    con.execute('CREATE TABLE daily_bars (symbol TEXT, bar_date DATE, open REAL, high REAL, low REAL, close REAL, volume INTEGER, fetched_at TEXT)')
    for i in range(25):
        d = f'2026-10-{i + 1:02d}' if i < 31 else DAY
        con.execute('INSERT INTO daily_bars VALUES (?,?,?,?,?,?,?,?)', ('AAA', d, 1, 1, 1, 1, 1000 + i, 'x'))
    con.execute('INSERT INTO daily_bars VALUES (?,?,?,?,?,?,?,?)', ('AAA', DAY, 1, 1, 1, 1, 9_999_999, 'x'))   # today excluded
    con.commit(); con.close()
    adv = hl.load_adv20(['AAA', 'NONE'], DAY, p)
    assert adv['AAA'] == pytest.approx(sum(1000 + i for i in range(5, 25)) / 20)
    assert 'NONE' not in adv


# ---------------------------------------------------------------- snapshots
def test_fetch_snapshots_parses_with_mocked_session():
    sess = MagicMock(spec=requests.Session)
    resp = MagicMock(); resp.status_code = 200
    resp.json.return_value = {'MBGL': {'latestTrade': {'p': 22.7}, 'latestQuote': {'bp': 22.69, 'ap': 22.71},
                                       'dailyBar': {'o': 21.0, 'v': 123456}, 'prevDailyBar': {'c': 20.0}}}
    sess.get.return_value = resp
    snaps = hl.fetch_snapshots(['MBGL', 'ZZZ'], 'k', 's', session=sess)
    assert sess.get.call_count == 1 and 'symbols=' not in sess.get.call_args[0][0]
    assert sess.get.call_args[1]['params']['symbols'] == 'MBGL,ZZZ'
    assert snaps['MBGL'] == {'last': 22.7, 'bid': 22.69, 'ask': 22.71, 'day_open': 21.0, 'prev_close': 20.0, 'day_volume': 123456}
    assert 'ZZZ' not in snaps


# ---------------------------------------------------------------- rows, message, ledger
def _rows():
    arms = hl.parse_arms(JOURNAL)
    trades = {'MBGL': {'status': 'STOPPED', 'r': -1.0, 'pnl': -50.0}, 'APH': {'status': 'FILLED @80.30', 'r': None, 'pnl': None}}
    snaps = {'MBGL': {'last': 22.60, 'bid': 22.59, 'ask': 22.61, 'day_open': 21.0, 'prev_close': 20.0, 'day_volume': 2_000_000},
             'APH': {'last': 80.0, 'bid': 79.9, 'ask': 80.1, 'day_open': 79, 'prev_close': 78.0, 'day_volume': 100}}
    return hl.build_rows(DAY, '1400', arms, trades, {'MBGL': {'qualified': '1', 'cf_floor_stop_px': '21.5'}}, snaps,
                         {'MBGL': 500_000.0}, minutes_elapsed=195)


def test_build_rows_fields():
    rows = {r['symbol']: r for r in _rows()}
    m = rows['MBGL']
    assert m['dist_pct'] == pytest.approx((22.60 - 22.80) / 22.80 * 100)
    assert m['r_pct'] == pytest.approx((22.81 - 22.10) / 22.81 * 100)
    assert m['day_pct'] == pytest.approx(13.0) and m['rvol'] == pytest.approx(8.0)
    assert m['qualified'] == '1' and m['rearms'] == 1 and m['status'] == 'STOPPED' and m['machine_r'] == -1.0
    assert rows['APH']['rvol'] is None and rows['APH']['qualified'] == ''


def test_message_sorted_limited_and_header():
    rows = []
    for i in range(20):
        rows.append({'symbol': f'S{i:02d}', 'level': 10.0, 'last': 10.0, 'dist_pct': (20 - i) / 10.0 * (-1 if i % 2 else 1), 'trigger': 10.1,
                     'stop': 9.5, 'r_pct': 5.0, 'day_pct': 1.0, 'rvol': 2.0, 'spread_bps': 10.0, 'rearms': 0, 'qualified': '1',
                     'status': 'ARMED', 'machine_r': None, 'pnl': None})
    rows[0].update(status='STOPPED', pnl=-50.0, machine_r=-1.0)
    rows[1].update(status='FILLED @10.10')
    msg = hl.format_message(rows, '0940')
    lines = msg.split('\n')
    assert lines[0] == '[HOD-LIST 09:40 ET] 20 armed · 2 filled · 1 closed -$50'
    body = [ln for ln in lines if ln.startswith('S')]
    assert len(body) == 14
    dists = [abs(float(ln.split('dist ')[1].split('%')[0])) for ln in body]
    assert dists == sorted(dists) and dists[0] == 0.1
    assert lines[-1] == 'pick by symbol on the live account; the ledger scores you Friday'
    assert '<' not in msg and '&' not in msg


def test_zero_rows_no_message_and_one_info(caplog):
    assert hl.format_message([], '1030') is None
    notifier = MagicMock(spec=hl.TelegramNotifier)
    caplog.set_level(logging.INFO)
    sent = hl.publish([], '1030', notifier=notifier)
    assert sent is False and notifier.send_message_sync.call_count == 0
    assert len([r for r in caplog.records if r.levelno == logging.INFO and 'no armed' in r.getMessage()]) == 1


def test_publish_sends_one_plain_message():
    notifier = MagicMock(spec=hl.TelegramNotifier); notifier.send_message_sync.return_value = True
    assert hl.publish(_rows(), '1400', notifier=notifier) is True
    notifier.send_message_sync.assert_called_once()
    assert notifier.send_message_sync.call_args[1].get('parse_mode') is None


def test_ledger_append_and_roundtrip(tmp_path):
    p = str(tmp_path / 'hod_owner_list.csv')
    rows = _rows()
    hl.append_ledger(rows, DAY, '1400', p)
    hl.append_ledger(rows[:1], DAY, '1030', p)
    back = hl.read_ledger(p)
    assert len(back) == len(rows) + 1
    assert back[0]['date'] == DAY and back[0]['at'] == '1400' and {r['symbol'] for r in back} >= {'MBGL', 'APH'}
    assert float(next(r for r in back if r['symbol'] == 'MBGL')['level']) == 22.80
    assert open(p).read().count('date,at,symbol') == 1          # header written once


def test_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hl, 'LEDGER', str(tmp_path / 'l.csv'))
    monkeypatch.setattr(hl, 'gather_rows', lambda date, at, now=None: _rows())
    monkeypatch.setattr(hl, 'TelegramNotifier', MagicMock(side_effect=AssertionError('must not send')))
    assert hl.main(['--at', '1400', '--dry-run', '--date', DAY]) == 0
    assert '[HOD-LIST 14:00 ET]' in capsys.readouterr().out
    assert not os.path.exists(str(tmp_path / 'l.csv'))


# ---------------------------------------------------------------- scorer
def test_prefix_exclusion_every_prefix():
    for pre in hs.OUR_PREFIXES:
        assert not hs.is_owner_pick({'client_order_id': pre + 'ABC-1', 'side': 'buy', 'status': 'filled', 'filled_qty': '10'})
    for pre in ('hod-', 'orb-', 'bf-', 'moml-', 'tom-', 'micha-', 'micha-hf-', 'micha-vid-'):
        assert pre in hs.OUR_PREFIXES
    assert hs.is_owner_pick({'client_order_id': '3f2a9c1e-0000-4000-8000-123456789abc', 'side': 'buy', 'status': 'filled', 'filled_qty': '10'})
    assert not hs.is_owner_pick({'client_order_id': 'x1', 'side': 'sell', 'status': 'filled', 'filled_qty': '10'})
    assert not hs.is_owner_pick({'client_order_id': 'x1', 'side': 'buy', 'status': 'canceled', 'filled_qty': '0'})


def test_in_list_vs_off_list_by_time():
    ledger = [{'date': DAY, 'at': '1030', 'symbol': 'AAA'}, {'date': DAY, 'at': '1400', 'symbol': 'BBB'}]
    after_list = '2026-11-12T16:00:00Z'          # 11:00 ET
    assert hs.classify_pick({'symbol': 'AAA', 'ts': after_list}, ledger) == 'IN-LIST'
    assert hs.classify_pick({'symbol': 'BBB', 'ts': after_list}, ledger) == 'OFF-LIST'        # listed at 14:00, bought 11:00
    assert hs.classify_pick({'symbol': 'CCC', 'ts': after_list}, ledger) == 'OFF-LIST'
    assert hs.classify_pick({'symbol': 'BBB', 'ts': '2026-11-12T19:30:00Z'}, ledger) == 'IN-LIST'


def test_fifo_pnl_buy_sell_pair_partial_and_open():
    fills = [{'side': 'buy', 'qty': 100, 'price': 10.0}, {'side': 'buy', 'qty': 100, 'price': 11.0},
             {'side': 'sell', 'qty': 150, 'price': 12.0}]
    out = hs.fifo_pnl(fills, mark=13.0)
    assert out['realized'] == pytest.approx(100 * 2.0 + 50 * 1.0)
    assert out['open_qty'] == 50 and out['unrealized'] == pytest.approx(50 * 2.0)
    assert out['pnl'] == pytest.approx(350.0) and out['bought_notional'] == pytest.approx(2100.0)
    flat = hs.fifo_pnl([{'side': 'buy', 'qty': 10, 'price': 5.0}, {'side': 'sell', 'qty': 10, 'price': 4.0}], mark=None)
    assert flat['pnl'] == pytest.approx(-10.0) and flat['open_qty'] == 0


def test_fetch_orders_paginates_and_filters():
    sess = MagicMock(spec=requests.Session)
    def page(orders):
        r = MagicMock(); r.status_code = 200; r.json.return_value = orders; return r
    o1 = {'id': '1', 'created_at': '2026-11-09T15:00:00Z', 'client_order_id': 'u1', 'side': 'buy', 'status': 'filled', 'filled_qty': '5'}
    o2 = {'id': '2', 'created_at': '2026-11-09T16:00:00Z', 'client_order_id': 'u2', 'side': 'buy', 'status': 'filled', 'filled_qty': '5'}
    sess.get.side_effect = [page([o1, o2]), page([o2])]
    out = hs.fetch_orders('2026-11-09', 'k', 's', session=sess, limit=2)
    assert [o['id'] for o in out] == ['1', '2'] and sess.get.call_count == 2
    assert sess.get.call_args_list[1][1]['params']['after'] == '2026-11-09T16:00:00Z'


def _picks_fixture():
    ledger = [{'date': DAY, 'at': '1030', 'symbol': s} for s in ('AAA', 'BBB', 'CCC')]
    machine = {(DAY, 'AAA'): 2.0, (DAY, 'BBB'): -1.0, (DAY, 'CCC'): -1.0}
    return ledger, machine


def _pick(sym, kind, pnl, notional=1000.0):
    return {'date': DAY, 'symbol': sym, 'kind': kind, 'pnl': pnl, 'bought_notional': notional}


def test_report_n_lt_5_no_verdict_and_tables():
    ledger, machine = _picks_fixture()
    picks = [_pick('AAA', 'IN-LIST', 50.0), _pick('ZZZ', 'OFF-LIST', -20.0)]
    rep = hs.build_report(picks, ledger, machine, since='2026-11-09')
    assert 'n < 5, no verdict' in rep['text']
    assert rep['n_picks'] == 2 and rep['n_in'] == 1 and rep['n_off'] == 1
    assert rep['owner_pnl'] == pytest.approx(30.0) and rep['owner_ret_pct'] == pytest.approx(1.5)
    assert rep['machine_r_picked'] == pytest.approx(2.0)             # AAA only (ZZZ not traded by the machine)
    assert rep['machine_r_not_picked'] == pytest.approx(-1.0)        # BBB, CCC: the list's base rate
    assert rep['owner_hit_rate'] == pytest.approx(0.5)


def test_report_with_verdict_at_five():
    ledger, machine = _picks_fixture()
    picks = [_pick('AAA', 'IN-LIST', 50.0)] + [_pick(f'X{i}', 'OFF-LIST', -10.0) for i in range(4)]
    assert 'no verdict' not in hs.build_report(picks, ledger, machine, since='2026-11-09')['text']


# ---------------------------------------------------------------- guard
@pytest.mark.parametrize('mod,name', [(hl, 'hod_owner_list'), (hs, 'hod_owner_list_score')])
def test_run_guarded_wraps_main(mod, name, caplog, monkeypatch):
    from monitoring import cron_guard
    monkeypatch.setattr(cron_guard, 'attach_telegram_error_handler', lambda: False)
    monkeypatch.setattr(cron_guard, '_wait_for_telegram_sends', lambda t: None)
    def boom():
        raise RuntimeError('kaput')
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError):
            cron_guard.run_guarded(boom, name)
    errs = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errs) == 1 and name in errs[0].getMessage()
    src = open(mod.__file__).read()
    assert f"run_guarded(main, '{name}')" in src and callable(mod.main)
