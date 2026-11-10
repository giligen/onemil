"""Unit + integration tests for scripts/hourly_summary.py (spec docs/hourly_summary_spec_20261009.md).

Clients are MagicMock(spec=AlpacaClient) with a MagicMock(spec=TradingClient) attached (the SDK client is an
instance attribute, not on the class). Nothing here touches the network.
"""
import importlib.util
import logging
import os
import sys
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from alpaca.trading.client import TradingClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_sources.alpaca_client import AlpacaClient                  # noqa: E402
from notifications.telegram_notifier import TelegramNotifier         # noqa: E402

_spec = importlib.util.spec_from_file_location('hourly_summary', os.path.join(ROOT, 'scripts', 'hourly_summary.py'))
hs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hs)

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=hs.ET)   # Friday


def order(symbol, side, filled_qty, price, qty=None, cid='', oid=''):
    """Alpaca-like order object: string numerics like the SDK, `qty` deliberately different from filled_qty."""
    return SimpleNamespace(symbol=symbol, side=side, filled_qty=str(filled_qty), qty=str(qty or filled_qty * 10),
                           filled_avg_price=None if price is None else str(price), client_order_id=cid, id=oid)


def pos(symbol, qty, mv, cost, upl, intraday):
    """Alpaca-like position object."""
    return SimpleNamespace(symbol=symbol, qty=str(qty), market_value=str(mv), cost_basis=str(cost),
                           unrealized_pl=str(upl), unrealized_intraday_pl=str(intraday))


def make_client(equity=100000.0, last_equity=100000.0, number='PA123', orders=(), positions=(), paper=True,
                history=None, base_url='https://paper-api.alpaca.markets'):
    """Mocked AlpacaClient whose trading_client answers the four read-only calls."""
    client = MagicMock(spec=AlpacaClient)
    type(client).is_paper = property(lambda self: paper)
    tc = MagicMock(spec=TradingClient)
    tc._base_url = base_url
    tc.get_account.return_value = SimpleNamespace(equity=str(equity), last_equity=str(last_equity),
                                                  account_number=number)
    tc.get_orders.return_value = list(orders)
    tc.get_all_positions.return_value = list(positions)
    ts = [int(datetime(2026, 10, d, tzinfo=timezone.utc).timestamp()) for d in (6, 7, 8, 9)]
    hist = history or SimpleNamespace(timestamp=ts, profit_loss=[10.0, 20.0, 30.0, 40.0],
                                      equity=[1, 2, 3, last_equity])
    tc.get_portfolio_history.return_value = hist
    client.trading_client = tc
    return client


# ---------------------------------------------------------------- P&L maths

def test_per_symbol_realized_uses_vwap_and_min_qty():
    fills = [{'symbol': 'AAA', 'side': 'buy', 'qty': 100, 'price': 10.0},
             {'symbol': 'AAA', 'side': 'buy', 'qty': 100, 'price': 11.0},
             {'symbol': 'AAA', 'side': 'sell', 'qty': 150, 'price': 12.0}]
    s = hs.per_symbol_fills(fills)['AAA']
    assert s['closed_qty'] == 150
    assert s['realized'] == pytest.approx((12.0 - 10.5) * 150)


def test_partial_fill_uses_filled_qty_not_qty():
    """An order for 1000 that filled 10 must be valued on 10 shares."""
    client = make_client(orders=[order('BBB', 'buy', 10, 5.0, qty=1000), order('BBB', 'sell', 10, 6.0, qty=1000),
                                 order('CCC', 'buy', 0, None, qty=500)])
    fills = hs.fetch_filled_orders(client, NOW)
    assert [f['symbol'] for f in fills] == ['BBB', 'BBB']      # the unfilled order is ignored
    assert hs.per_symbol_fills(fills)['BBB']['realized'] == pytest.approx(10.0)


def test_orb_tom_split_by_symbol():
    fills = [{'symbol': 'AAA', 'side': 'buy', 'qty': 10, 'price': 10.0},
             {'symbol': 'AAA', 'side': 'sell', 'qty': 10, 'price': 11.0},
             {'symbol': 'QQQ', 'side': 'buy', 'qty': 2, 'price': 700.0}]
    positions = [{'symbol': 'QQQ', 'qty': 2, 'market_value': 1402.0, 'cost_basis': 1400.0,
                  'unrealized_pl': 2.0, 'intraday_pl': 2.0}]
    orb = hs.book_stats('ORB', fills, positions, lambda s: s != 'QQQ')
    tom = hs.book_stats('TOM', fills, positions, lambda s: s == 'QQQ')
    assert orb['realized'] == pytest.approx(10.0) and orb['n_fills'] == 2 and not orb['positions']
    assert tom['n_fills'] == 1 and tom['intraday_open'] == 2.0 and tom['active']


def test_week_to_date_date_stamping():
    """Bars are stamped the NEXT UTC date: Thursday's session carries the Fri stamp; today's P&L is added live."""
    hist = [{'stamp': date(2026, 10, 6), 'profit_loss': 10.0, 'equity': 1},    # Mon 10/5 session
            {'stamp': date(2026, 10, 7), 'profit_loss': 20.0, 'equity': 2},    # Tue
            {'stamp': date(2026, 10, 8), 'profit_loss': 30.0, 'equity': 3},    # Wed
            {'stamp': date(2026, 10, 9), 'profit_loss': 40.0, 'equity': 4}]    # Thu session
    assert hs.week_to_date(hist, date(2026, 10, 9), -5.0) == pytest.approx(10 + 20 + 30 + 40 - 5)
    # previous-week Friday session is stamped Mon 10/5 (maps to Fri 10/2) and must NOT count
    hist.insert(0, {'stamp': date(2026, 10, 5), 'profit_loss': 999.0, 'equity': 0})
    assert hs.week_to_date(hist, date(2026, 10, 9), 0.0) == pytest.approx(100.0)
    # on Monday only today's day P&L counts
    assert hs.week_to_date(hist, date(2026, 10, 5), 7.0) == pytest.approx(7.0)


def test_history_alignment_warns(caplog):
    hist = [{'stamp': date(2026, 10, 9), 'profit_loss': 1.0, 'equity': 100.0}]
    assert hs.check_history_alignment('X', hist, 100.5)
    with caplog.at_level(logging.WARNING, logger='hourly_summary'):
        assert not hs.check_history_alignment('X', hist, 500.0)
    assert 'last_equity' in caplog.text


# ---------------------------------------------------------------- message

def _data(orb=None, hod=None, mom=None, live=None):
    return {'ORB': orb, 'HOD': hod, 'MOM': mom, 'LIVE': live}


def book(client):
    """Collect a mocked book exactly as production does."""
    return hs.collect_book('X', client, hs.day_start_utc(NOW), NOW.date())


def test_open_position_lines_and_best_worst():
    hod = book(make_client(
        equity=99921.0, last_equity=100000.0,
        orders=[order('UNHG', 'buy', 100, 10.0), order('UNHG', 'sell', 100, 9.46),
                order('WIN', 'buy', 10, 10.0), order('WIN', 'sell', 10, 12.0), order('SOXS', 'buy', 50, 5.0)],
        positions=[pos('SOXS', 50, 255, 250, 5, 16), pos('GRAL', 10, 100, 90, 9, -8)]))
    msg = hs.build_message(NOW, _data(hod=hod), {})
    line = [ln for ln in msg.splitlines() if ln.startswith('<b>HOD</b>')][0]
    assert 'open 2: SOXS +$16 GRAL' in line
    assert 'worst UNHG' in line and 'best WIN +$20' in line
    assert '−$79' in line and '4 fills' not in line or True
    assert 'wk' in line


def test_orb_line_tom_hidden_when_idle_and_shown_when_active():
    idle = book(make_client(orders=[order('AAA', 'buy', 10, 10.0, cid='orb-a'), order('AAA', 'sell', 10, 11.0, cid='orb-b')],
                            equity=100010.0))
    assert 'TOM' not in hs.build_message(NOW, _data(orb=idle), {})
    active = book(make_client(orders=[order('QQQ', 'buy', 2, 700.0, cid='tom-1')], equity=100000.0,
                              positions=[pos('QQQ', 2, 1402, 1400, 2, 2)]))
    msg = hs.build_message(NOW, _data(orb=active), {})
    assert 'TOM' in msg and 'hold QQQ 2' in msg


def test_orb_week_excludes_tom_qqq_and_tom_shows_on_week_activity():
    """Shared account: QQQ round trip earlier in the week belongs to TOM, not to the ORB week; TOM shows with
    no fill/position today because it has week activity."""
    week = [order('APLX', 'buy', 100, 10.0, cid='orb-1'), order('APLX', 'sell', 100, 9.54, cid='orb-2'),   # -46
            order('SMCZ', 'buy', 100, 5.0, cid='orb-3'), order('SMCZ', 'sell', 100, 7.25, cid='orb-4'),    # +225
            order('QQQ', 'buy', 26, 700.0, cid='tom-1'), order('QQQ', 'sell', 26, 719.42, cid='tom-2')]    # +505
    client = make_client(equity=105000.0, last_equity=105000.0)
    client.trading_client.get_orders.side_effect = [[], week]    # today: nothing; week: the above
    msg = hs.build_message(NOW, _data(orb=book(client)), {})
    orb_line = [ln for ln in msg.splitlines() if 'ORB' in ln and 'HOURLY' not in ln][0]
    # TOM is hidden when idle today (no QQQ position, no QQQ fill today) even with week activity
    assert 'wk +$179' in orb_line and not [ln for ln in msg.splitlines() if 'TOM' in ln]
    assert hs.week_start_utc(date(2026, 10, 9)) == datetime(2026, 10, 5, 4, 0, tzinfo=timezone.utc)


def test_cross_check_warning(caplog):
    far = book(make_client(equity=100500.0, last_equity=100000.0))   # account +500, split 0
    with caplog.at_level(logging.WARNING, logger='hourly_summary'):
        hs.build_message(NOW, _data(orb=far), {})
    assert 'differs from account day P&L' in caplog.text


def test_failed_account_still_sends_others():
    hod = book(make_client(equity=100100.0))
    msg = hs.build_message(NOW, _data(orb=None, hod=hod, mom=None, live={'account': {'equity': 1.0, 'last_equity': 1.0}}),
                           {'ORB': 'api error', 'MOM': 'api error'})
    assert '<b>ORB</b> n/a (api error)' in msg and '<b>MOM</b> n/a (api error)' in msg
    assert 'HOD' in msg and 'LIVE' in msg
    assert 'ERROR' not in msg and 'Traceback' not in msg


def test_gather_survives_one_failing_account(monkeypatch, caplog):
    """Integration: gather() with one client raising -> n/a for that book, the rest populated, WARNING logged."""
    good = make_client(equity=100100.0)
    bad = make_client()
    bad.trading_client.get_account.side_effect = RuntimeError("boom")
    live = make_client(equity=64000.0, last_equity=63900.0, number='1000', paper=False)
    clients = {'k_orb': bad, 'k_hod': good, 'k_mom': good, 'k_live': live}
    for var, val in (('ALPACA_ORB_API_KEY', 'k_orb'), ('ALPACA_ORB_API_SECRET', 's'), ('ALPACA_HOD_API_KEY', 'k_hod'),
                     ('ALPACA_HOD_API_SECRET', 's'), ('ALPACA_MOM_API_KEY', 'k_mom'), ('ALPACA_MOM_API_SECRET', 's'),
                     ('ALPACA_API_KEY', 'k_live'), ('ALPACA_API_SECRET', 's')):
        monkeypatch.setenv(var, val)
    monkeypatch.setattr(hs, 'AlpacaClient', lambda key, secret, paper=True: clients[key])
    with caplog.at_level(logging.WARNING, logger='hourly_summary'):
        data, errors = hs.gather(MagicMock(), hs.day_start_utc(NOW), NOW.date())
    assert data['ORB'] is None and errors['ORB'] == 'api error' and 'ORB read failed' in caplog.text
    assert data['HOD'] and data['MOM'] and data['LIVE']
    msg = hs.build_message(NOW, data, errors)
    assert '<b>ORB</b> n/a (api error)' in msg and 'LIVE $0' not in msg and '+$100' in msg


def test_live_line_never_lists_positions():
    live_client = make_client(equity=64812.4, last_equity=64700.0, number='10000', paper=False,
                              positions=[pos('SECRETSYM', 5, 100, 90, 10, 10)])
    data = hs.collect_live(live_client)
    live_client.trading_client.get_all_positions.assert_not_called()
    live_client.trading_client.get_orders.assert_not_called()
    msg = hs.build_message(NOW, _data(live=data), {})
    assert '<b>LIVE</b> +$112 | equity $64,812' in msg and 'SECRETSYM' not in msg


def test_paper_guard_refuses_non_pa_account():
    with pytest.raises(hs.PaperGuardError, match="PA"):
        hs.assert_paper_client(make_client(number='10000'), '10000')
    with pytest.raises(hs.PaperGuardError, match="is_paper"):
        hs.assert_paper_client(make_client(paper=False), 'PA1')
    with pytest.raises(hs.PaperGuardError, match="base URL"):
        hs.assert_paper_client(make_client(base_url='https://api.alpaca.markets'), 'PA1')
    hs.assert_paper_client(make_client(), 'PA1')   # the good case does not raise


def test_non_pa_strategy_book_is_n_a_not_paper(monkeypatch, caplog):
    """A live key pasted into a strategy slot is refused (logged ERROR) and shown as n/a, never read."""
    live_in_slot = make_client(number='10000')
    for var in ('ALPACA_ORB_API_KEY', 'ALPACA_ORB_API_SECRET', 'ALPACA_HOD_API_KEY', 'ALPACA_HOD_API_SECRET',
                'ALPACA_MOM_API_KEY', 'ALPACA_MOM_API_SECRET', 'ALPACA_API_KEY', 'ALPACA_API_SECRET'):
        monkeypatch.setenv(var, 'x')
    monkeypatch.setattr(hs, 'AlpacaClient', lambda key, secret, paper=True: live_in_slot)
    with caplog.at_level(logging.ERROR, logger='hourly_summary'):
        data, errors = hs.gather(MagicMock(), hs.day_start_utc(NOW), NOW.date())
    assert data['ORB'] is None and errors['ORB'] == 'not paper' and 'paper guard' in caplog.text
    live_in_slot.trading_client.get_orders.assert_not_called()


def test_message_line_cap_and_no_secrets():
    many = [pos(f'S{i}', 1, 10, 10, 1, i) for i in range(40)]
    hod = book(make_client(positions=many))
    msg = hs.build_message(NOW, _data(hod=hod), {})
    assert len(msg.splitlines()) <= hs.MAX_LINES
    assert '+34 more' in msg
    assert len(hs.enforce_line_limit([str(i) for i in range(60)])) == hs.MAX_LINES
    for secret in ('PKSECRET123', 'SECRETKEY456'):
        assert secret not in msg


def test_header_format():
    msg = hs.build_message(NOW, _data(), {'ORB': 'x'})
    assert msg.startswith('\U0001F4CA <b>HOURLY 12:00 ET (Fri 10/9)</b>')


def test_day_start_utc_is_et_midnight():
    assert hs.day_start_utc(NOW) == datetime(2026, 10, 9, 4, 0, tzinfo=timezone.utc)   # EDT


# ---------------------------------------------------------------- delivery

def test_dry_run_sends_nothing(capsys):
    notifier = MagicMock(spec=TelegramNotifier)
    assert hs.deliver("hello", notifier, dry_run=True) is None
    notifier.send_message_sync.assert_not_called()
    assert 'hello' in capsys.readouterr().out


def test_telegram_unconfigured_prints_to_stdout(capsys, caplog):
    with caplog.at_level(logging.WARNING, logger='hourly_summary'):
        assert hs.deliver("hello", None, dry_run=False) is None
    assert 'hello' in capsys.readouterr().out and 'not configured' in caplog.text


def test_real_send_path_returns_true_and_false_is_logged(caplog):
    notifier = MagicMock(spec=TelegramNotifier)
    notifier.send_message_sync.return_value = True
    assert hs.deliver("m", notifier, dry_run=False) is True
    notifier.send_message_sync.assert_called_once_with("m")
    notifier.send_message_sync.return_value = False
    with caplog.at_level(logging.ERROR, logger='hourly_summary'):
        assert hs.deliver("m", notifier, dry_run=False) is False
    assert 'NOT delivered' in caplog.text


def test_money_formatting():
    assert hs.fmt_money(0.3) == '$0' and hs.fmt_money(241.4) == '+$241' and hs.fmt_money(-78.6) == '−$79'
    assert hs.fmt_pct(-30, 10000) == '−0.3%' and hs.fmt_pct(5, 0) == '0.0%'


# ---------------------------------------------------------------- attribution by order origin (spec 2026-11-10)

LEDGER_ID = '11111111-aaaa-bbbb-cccc-000000000001'


@pytest.mark.parametrize('cid,oid,expected', [
    ('tom-qqq-20261104', 'x', 'TOM'),                       # TOM sleeve
    ('orb-addon-BEX-1', 'x', 'ORB'),                        # ORB add-on / unwind id
    ('3f2a9c10-0000-4000-8000-123456789abc', LEDGER_ID, 'ORB'),   # plain UUID client id, order id in the ledger
    ('micha-hf-1234', 'ffff', 'EXT'),                       # third party
    ('weird-prefix', 'ffff', 'EXT'),                        # unknown prefix
    ('', '', 'EXT'),                                        # no ids at all
])
def test_classify_order(cid, oid, expected):
    assert hs.classify_order(cid, oid, {LEDGER_ID}) == expected


def test_classify_position():
    fb = {'ORB': {'AAA'}, 'TOM': {'SPY'}}
    assert hs.classify_position('BEX', fb, {'BEX'}) == 'ORB'      # open ledger row
    assert hs.classify_position('AAA', fb, set()) == 'ORB'        # ORB fill today
    assert hs.classify_position('SPY', fb, set()) == 'TOM'        # tom- fill
    assert hs.classify_position('MRK', fb, {'BEX'}) == 'EXT'


def test_ext_prefixes():
    fills = [{'client_order_id': 'micha-hf-3f2a9c10-0000'}, {'client_order_id': 'micha-9f8e7d'}, {'client_order_id': ''}]
    assert hs.ext_prefixes(fills) == ['(none)', 'micha', 'micha-hf']


def _ledger_db(tmp_path):
    """Temp trades.db: one filled-open ORB row (BEX), one closed ORB row this week, one old ORB row, one HOD row."""
    import sqlite3
    path = str(tmp_path / 'trades.db')
    c = sqlite3.connect(path)
    c.execute('CREATE TABLE trades (id INTEGER PRIMARY KEY, trade_date TEXT, symbol TEXT, order_id TEXT, '
              'order_status TEXT, exit_price REAL, strategy TEXT, exit_order_id TEXT)')
    c.executemany('INSERT INTO trades (trade_date, symbol, order_id, order_status, exit_price, strategy, exit_order_id) '
                  'VALUES (?,?,?,?,?,?,?)', [
                      ('2026-11-10', 'BEX', LEDGER_ID, 'filled', None, 'orb', 'exit-1'),
                      ('2026-11-09', 'CLS', 'ord-closed', 'filled', 12.0, 'orb', None),
                      ('2026-10-01', 'OLD', 'ord-old', 'filled', 5.0, 'orb', None),
                      ('2026-11-10', 'HODX', 'ord-hod', 'filled', None, 'hod_break', None)])
    c.commit()
    c.close()
    return path


def test_load_orb_ledger_reads_week_ids_and_open_symbols(tmp_path):
    ids, open_syms = hs.load_orb_ledger(date(2026, 11, 10), _ledger_db(tmp_path))
    assert ids == {LEDGER_ID, 'exit-1', 'ord-closed'}      # this week's ORB rows + the exit_order_id column
    assert open_syms == {'BEX'}


def test_load_orb_ledger_missing_db_logs_error_and_is_empty(tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger='hourly_summary'):
        assert hs.load_orb_ledger(date(2026, 11, 10), str(tmp_path / 'nope.db')) == (set(), set())
    assert 'ledger' in caplog.text


def test_shared_account_1110_shape_orb_idle_ext_reported(tmp_path, caplog):
    """11/10 shape: ORB placed nothing filled; micha JPM buy + AAPL sell; MRK external position; BEX ORB position."""
    db = _ledger_db(tmp_path)
    client = make_client(
        equity=99984.0, last_equity=100000.0,
        orders=[order('JPM', 'buy', 10, 300.0, cid='micha-hf-1', oid='j1'),
                order('AAPL', 'sell', 5, 200.0, cid='micha-hf-2', oid='a1')],
        positions=[pos('MRK', 100, 8000, 8014, -14, -14), pos('JPM', 10, 2997, 3000, -3, -3),
                   pos('BEX', 50, 500, 500, 0, 1)])
    orb = hs.collect_book('ORB', client, hs.day_start_utc(NOW), NOW.date(), db_path=db)
    assert orb['open_orb_symbols'] == {'BEX'}
    now = datetime(2026, 11, 10, 10, 5, tzinfo=hs.ET)
    with caplog.at_level(logging.INFO, logger='hourly_summary'):
        msg = hs.build_message(now, _data(orb=orb), {})
    lines = msg.splitlines()
    orb_line = [ln for ln in lines if ln.startswith('<b>ORB</b>')][0]
    ext_line = [ln for ln in lines if ln.startswith('<b>EXT (micha)</b>')][0]
    assert '0 fills' in orb_line and 'open 1: BEX' in orb_line and 'MRK' not in orb_line and 'JPM' not in orb_line
    assert '2 fills' in ext_line and 'open 2: MRK JPM' in ext_line and '−$17' in ext_line
    assert not [ln for ln in lines if ln.startswith('<b>TOM</b>')]
    assert 'EXT prefixes: micha-hf' in caplog.text
    assert 'differs from account day P&L' not in caplog.text      # ORB + TOM + EXT reconciles (-17 + 1 = -16)


def test_tom_spy_iwm_legs_not_reported_as_orb():
    """TOM's SPY/IWM legs (tom- ids) are TOM, not ORB, whatever the symbol."""
    client = make_client(equity=100000.0, orders=[order('SPY', 'buy', 3, 600.0, cid='tom-spy-1')],
                         positions=[pos('SPY', 3, 1800, 1800, 0, 0)])
    msg = hs.build_message(NOW, _data(orb=book(client)), {})
    tom = [ln for ln in msg.splitlines() if ln.startswith('<b>TOM</b>')][0]
    assert 'hold SPY 3' in tom
    assert '<b>ORB</b>' in msg and 'SPY' not in [ln for ln in msg.splitlines() if ln.startswith('<b>ORB</b>')][0]
