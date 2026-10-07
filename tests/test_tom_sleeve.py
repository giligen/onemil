"""Unit tests for scripts/tom_sleeve.py — the turn-of-month PAPER sleeve runner
(research/index_overnight/RESULT_1649.md consequence, cell 1,651). Loaded via importlib since
scripts/ is not a package (same pattern as tests/test_orb_ramp_check.py).
"""
import csv
import datetime as dt
import importlib.util
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from unittest.mock import MagicMock  # noqa: E402
from data_sources.alpaca_client import AlpacaClient, AlpacaAPIError  # noqa: E402
from notifications.telegram_notifier import TelegramNotifier  # noqa: E402

spec = importlib.util.spec_from_file_location('tom_sleeve', os.path.join(ROOT, 'scripts', 'tom_sleeve.py'))
tom = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tom
spec.loader.exec_module(tom)


# ---------------------------------------------------------------------------
# Fake calendar (weekday-only + a couple of real NYSE holidays), independent
# of tom_sleeve's own logic — used as AlpacaClient.get_market_calendar's mock.
# ---------------------------------------------------------------------------

HOLIDAYS = {dt.date(2026, 12, 25), dt.date(2027, 1, 1)}


def fake_calendar(start_date, end_date):
    days = []
    d = start_date
    while d <= end_date:
        if d.weekday() < 5 and d not in HOLIDAYS:
            days.append({'date': d, 'open': dt.time(9, 30), 'close': dt.time(16, 0)})
        d += dt.timedelta(days=1)
    return days


class _FakeStatus:
    def __init__(self, value):
        self.value = value


class _FakeOrder:
    """Stand-in for the alpaca-py Order object returned by trading_client.submit_order."""
    def __init__(self, id_, status='accepted'):
        self.id = id_
        self.status = _FakeStatus(status)


def make_alpaca(is_paper=True, base_url='https://paper-api.alpaca.markets',
                 buying_power=100_000.0, prices=None, open_orders=None, open_positions=None,
                 short_day=False):
    """A MagicMock(spec=AlpacaClient) wired with sane defaults for a clean entry/exit run."""
    a = MagicMock(spec=AlpacaClient)
    a.is_paper = is_paper
    a.trading_client = MagicMock()
    a.trading_client._base_url = types.SimpleNamespace(value=base_url)
    a.trading_client.submit_order.side_effect = lambda req: _FakeOrder(f"buy-{req.symbol}")
    a.get_market_calendar.side_effect = fake_calendar
    a.is_short_trading_day.return_value = short_day
    a.get_buying_power.return_value = buying_power
    a.get_latest_trades.return_value = prices or {
        'SPY': {'price': 500.0}, 'QQQ': {'price': 400.0}, 'IWM': {'price': 200.0},
    }
    a.get_open_orders.return_value = open_orders or []
    a.get_open_positions.return_value = open_positions or []
    a.submit_moc_sell_order.side_effect = lambda symbol, qty, client_order_id=None: {
        'id': f'sell-{symbol}', 'status': 'accepted', 'symbol': symbol, 'qty': qty,
    }
    a.get_order.return_value = {'status': 'filled', 'filled_avg_price': 500.0}
    return a


def make_notifier():
    n = MagicMock(spec=TelegramNotifier)
    n.send_message_sync.return_value = True
    return n


def et_dt(y, m, d, hh, mm):
    """A UTC datetime whose ET wall-clock is exactly (y, m, d, hh:mm) — EDT (UTC-4) for
    Mar-Nov dates used here, EST (UTC-5) for the December fixture."""
    naive = dt.datetime(y, m, d, hh, mm)
    offset = 4 if m in range(3, 12) else 5
    return (naive + dt.timedelta(hours=offset)).replace(tzinfo=dt.timezone.utc)


# ---------------------------------------------------------------------------
# Calendar: entry/exit day detection
# ---------------------------------------------------------------------------

def test_classify_day_last_session_is_entry():
    a = make_alpaca()
    assert tom.classify_day(dt.date(2026, 9, 30), a) == 'entry'


def test_classify_day_third_session_is_exit():
    a = make_alpaca()
    assert tom.classify_day(dt.date(2026, 10, 5), a) == 'exit'


def test_classify_day_mid_month_is_none():
    a = make_alpaca()
    assert tom.classify_day(dt.date(2026, 9, 15), a) == 'none'


def test_classify_day_december_january_boundary_with_holidays():
    """Dec 25 2026 (Fri) and Jan 1 2027 (Fri) are NYSE holidays: Dec's last session is unaffected
    (Dec 31, a Thursday) but Jan's third session shifts from Jan 5 to Jan 6 because Jan 1 is not a
    session at all."""
    a = make_alpaca()
    assert tom.classify_day(dt.date(2026, 12, 31), a) == 'entry'
    assert tom.classify_day(dt.date(2027, 1, 6), a) == 'exit'
    assert tom.classify_day(dt.date(2027, 1, 5), a) == 'none'
    assert tom.classify_day(dt.date(2026, 12, 24), a) == 'none'


def test_in_action_window():
    assert tom.in_action_window(et_dt(2026, 9, 30, 15, 45).astimezone(tom.ET))
    assert not tom.in_action_window(et_dt(2026, 9, 30, 15, 39).astimezone(tom.ET))
    assert not tom.in_action_window(et_dt(2026, 9, 30, 15, 56).astimezone(tom.ET))


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

def test_shares_for_notional():
    assert tom.shares_for_notional(500.0, 20_000.0) == 40
    assert tom.shares_for_notional(300.0, 20_000.0) == 66          # floored, not rounded
    assert tom.shares_for_notional(0.0, 20_000.0) == 0
    assert tom.shares_for_notional(None, 20_000.0) == 0


def test_client_order_id_format():
    assert tom._client_order_id(dt.date(2026, 9, 30), 'SPY', 'in') == 'tom-202609-SPY-in'
    assert tom._client_order_id(dt.date(2026, 10, 5), 'QQQ', 'out') == 'tom-202610-QQQ-out'
    with pytest.raises(ValueError):
        tom._client_order_id(dt.date(2026, 9, 30), 'TSLA', 'in')
    with pytest.raises(ValueError):
        tom._client_order_id(dt.date(2026, 9, 30), 'SPY', 'sideways')


# ---------------------------------------------------------------------------
# Safety guards
# ---------------------------------------------------------------------------

def test_assert_paper_account_passes_on_paper():
    tom.assert_paper_account(make_alpaca())  # no raise


def test_assert_paper_account_rejects_is_paper_false():
    a = make_alpaca(is_paper=False)
    with pytest.raises(RuntimeError, match='is_paper'):
        tom.assert_paper_account(a)


def test_assert_paper_account_rejects_live_base_url():
    a = make_alpaca(is_paper=True, base_url='https://api.alpaca.markets')
    with pytest.raises(RuntimeError, match='base URL'):
        tom.assert_paper_account(a)


def test_assert_buying_power_blocks_below_floor():
    a = make_alpaca(buying_power=50_000.0)
    with pytest.raises(RuntimeError, match='buying power'):
        tom.assert_buying_power(a)


def test_assert_buying_power_passes_above_floor():
    a = make_alpaca(buying_power=100_000.0)
    assert tom.assert_buying_power(a) == 100_000.0


# ---------------------------------------------------------------------------
# run(): entry / exit / idempotency / ledger, against a tmp ledger path
# ---------------------------------------------------------------------------

def test_entry_day_submits_moc_for_qqq_in_main_window(tmp_path):
    """15:45 ET (MOC window): only QQQ (MOC_RELIABLE_SYMBOLS) submits here — SPY/IWM wait for the
    late fallback window (module docstring, 2026-09-30 incident: their MOC legs expired unfilled)."""
    ledger = str(tmp_path / 'ledger.csv')
    a = make_alpaca()
    n = make_notifier()
    summary = tom.run(a, n, et_dt(2026, 9, 30, 15, 45), dry_run=False, ledger_path=ledger)
    assert a.trading_client.submit_order.call_count == 1
    assert n.send_message_sync.call_count == 1
    submitted_syms = {call.args[0].symbol for call in a.trading_client.submit_order.call_args_list}
    assert submitted_syms == {'QQQ'}
    rows = tom.read_ledger(ledger)
    assert len(rows) == 1
    qqq_row = rows[0]
    assert qqq_row['action'] == 'entry'
    assert qqq_row['qty'] == '50'          # 20000 // 400.0
    assert qqq_row['client_order_id'] == 'tom-202609-QQQ-in'
    assert qqq_row['status'] == 'submitted'
    assert 'BUY MOC' in summary


def test_entry_day_submits_fallback_limit_for_arca_symbols_in_late_window(tmp_path):
    """15:58 ET (late fallback window): SPY/IWM (FALLBACK_LIMIT_SYMBOLS) submit a marketable DAY
    limit — not CLS — so they do not depend on the closing auction that expired them 2026-09-30."""
    ledger = str(tmp_path / 'ledger.csv')
    a = make_alpaca()
    n = make_notifier()
    summary = tom.run(a, n, et_dt(2026, 9, 30, 15, 58), dry_run=False, ledger_path=ledger)
    assert a.trading_client.submit_order.call_count == 2
    submitted_syms = {call.args[0].symbol for call in a.trading_client.submit_order.call_args_list}
    assert submitted_syms == {'SPY', 'IWM'}
    ref_price = {'SPY': 500.0, 'IWM': 200.0}
    for call in a.trading_client.submit_order.call_args_list:
        req = call.args[0]
        assert req.time_in_force.value == 'day'
        assert req.limit_price > ref_price[req.symbol]  # marketable: above the latest trade
    rows = tom.read_ledger(ledger)
    assert len(rows) == 2
    assert 'BUY LIMIT' in summary


def test_entry_day_skips_symbol_with_existing_open_order(tmp_path):
    ledger = str(tmp_path / 'ledger.csv')
    existing_coid = tom._client_order_id(dt.date(2026, 9, 30), 'SPY', 'in')
    a = make_alpaca(open_orders=[{'client_order_id': existing_coid, 'symbol': 'SPY'}])
    n = make_notifier()
    tom.run(a, n, et_dt(2026, 9, 30, 15, 58), dry_run=False, ledger_path=ledger)
    # SPY must not have been (re)submitted; only IWM goes out (QQQ is not in the late window).
    submitted_syms = {call.args[0].symbol for call in a.trading_client.submit_order.call_args_list}
    assert 'SPY' not in submitted_syms
    assert submitted_syms == {'IWM'}


def test_entry_day_skips_symbol_with_existing_open_position(tmp_path):
    """A tom-tagged SPY position already logged as held (no exit yet) must not be re-entered even
    though nothing about today's order IDs collides — this is the ledger-based guard, distinct from
    the broker open-order check above."""
    ledger = str(tmp_path / 'ledger.csv')
    tom.append_ledger_row({
        'date': '2026-08-31', 'action': 'entry', 'symbol': 'SPY', 'qty': 39,
        'ref_price': '510.0', 'order_id': 'old-buy-spy', 'status': 'filled',
        'client_order_id': 'tom-202608-SPY-in', 'timestamp_utc': '2026-08-31T19:45:00+00:00',
        'partial_entry': '',
    }, ledger)
    a = make_alpaca()
    n = make_notifier()
    tom.run(a, n, et_dt(2026, 9, 30, 15, 58), dry_run=False, ledger_path=ledger)
    submitted_syms = {call.args[0].symbol for call in a.trading_client.submit_order.call_args_list}
    assert 'SPY' not in submitted_syms
    assert submitted_syms == {'IWM'}


def test_buying_power_guard_blocks_entry_run(tmp_path):
    ledger = str(tmp_path / 'ledger.csv')
    a = make_alpaca(buying_power=50_000.0)
    n = make_notifier()
    with pytest.raises(RuntimeError, match='buying power'):
        tom.run(a, n, et_dt(2026, 9, 30, 15, 45), dry_run=False, ledger_path=ledger)
    assert a.trading_client.submit_order.call_count == 0


def test_exit_day_sells_held_qty_from_broker_and_writes_ledger(tmp_path):
    ledger = str(tmp_path / 'ledger.csv')
    tom.append_ledger_row({
        'date': '2026-09-30', 'action': 'entry', 'symbol': 'SPY', 'qty': 40,
        'ref_price': '500.0', 'order_id': 'buy-spy', 'status': 'filled',
        'client_order_id': 'tom-202609-SPY-in', 'timestamp_utc': '2026-09-30T19:45:00+00:00',
    }, ledger)
    a = make_alpaca(open_positions=[{'symbol': 'SPY', 'qty': 40}])
    n = make_notifier()
    summary = tom.run(a, n, et_dt(2026, 10, 5, 15, 45), dry_run=False, ledger_path=ledger)
    a.submit_moc_sell_order.assert_called_once_with('SPY', 40, client_order_id='tom-202610-SPY-out')
    rows = tom.read_ledger(ledger)
    exit_row = next(r for r in rows if r['action'] == 'exit')
    assert exit_row['qty'] == '40'
    assert exit_row['client_order_id'] == 'tom-202610-SPY-out'
    assert 'SELL MOC' in summary


def test_exit_day_skips_symbol_with_no_open_ledger_position(tmp_path):
    """No entry row logged for QQQ/IWM -> exit day must not try to sell them."""
    ledger = str(tmp_path / 'ledger.csv')
    tom.append_ledger_row({
        'date': '2026-09-30', 'action': 'entry', 'symbol': 'SPY', 'qty': 40,
        'ref_price': '500.0', 'order_id': 'buy-spy', 'status': 'filled',
        'client_order_id': 'tom-202609-SPY-in', 'timestamp_utc': '2026-09-30T19:45:00+00:00',
    }, ledger)
    a = make_alpaca(open_positions=[{'symbol': 'SPY', 'qty': 40}])
    n = make_notifier()
    tom.run(a, n, et_dt(2026, 10, 5, 15, 45), dry_run=False, ledger_path=ledger)
    assert a.submit_moc_sell_order.call_count == 1  # SPY only


# ---------------------------------------------------------------------------
# dry-run / window / short-day behaviour
# ---------------------------------------------------------------------------

def test_dry_run_submits_nothing_and_writes_no_ledger(tmp_path):
    ledger = str(tmp_path / 'ledger.csv')
    a = make_alpaca()
    n = make_notifier()
    summary = tom.run(a, n, et_dt(2026, 9, 30, 15, 45), dry_run=True, ledger_path=ledger)
    assert a.trading_client.submit_order.call_count == 0
    assert n.send_message_sync.call_count == 0
    assert not os.path.exists(ledger)
    assert 'DRY-RUN' in summary


def test_outside_window_is_noop_and_skips_calendar_call(tmp_path):
    ledger = str(tmp_path / 'ledger.csv')
    a = make_alpaca()
    n = make_notifier()
    summary = tom.run(a, n, et_dt(2026, 9, 30, 12, 0), dry_run=False, ledger_path=ledger)
    a.get_market_calendar.assert_not_called()
    assert 'outside' in summary
    assert a.trading_client.submit_order.call_count == 0


def test_short_trading_day_skips_without_submitting(tmp_path):
    ledger = str(tmp_path / 'ledger.csv')
    a = make_alpaca(short_day=True)
    n = make_notifier()
    summary = tom.run(a, n, et_dt(2026, 9, 30, 15, 45), dry_run=False, ledger_path=ledger)
    assert 'short trading day' in summary
    assert a.trading_client.submit_order.call_count == 0


# ---------------------------------------------------------------------------
# Ledger read/write and open_tom_symbols bookkeeping
# ---------------------------------------------------------------------------

def test_ledger_round_trip_and_header(tmp_path):
    ledger = str(tmp_path / 'ledger.csv')
    tom.append_ledger_row({
        'date': '2026-09-30', 'action': 'entry', 'symbol': 'IWM', 'qty': 90,
        'ref_price': '220.0', 'order_id': 'buy-iwm', 'status': 'submitted',
        'client_order_id': 'tom-202609-IWM-in', 'timestamp_utc': '2026-09-30T19:45:00+00:00',
    }, ledger)
    with open(ledger, newline='') as f:
        header = next(csv.reader(f))
    assert header == tom.LEDGER_FIELDS
    rows = tom.read_ledger(ledger)
    assert len(rows) == 1 and rows[0]['symbol'] == 'IWM'


def test_open_tom_symbols_tracks_entry_then_exit():
    rows = [
        {'action': 'entry', 'symbol': 'SPY', 'status': 'filled'},
        {'action': 'entry', 'symbol': 'QQQ', 'status': 'filled'},
        {'action': 'exit', 'symbol': 'SPY', 'status': 'filled'},
    ]
    assert tom.open_tom_symbols(rows) == {'QQQ'}


def test_open_tom_symbols_clears_symbol_whose_entry_expired_unfilled():
    """2026-09-30 bug: an entry row starts 'submitted' (tentatively held); if it resolves EXPIRED
    rather than filled, the symbol must NOT stay 'held' forever — else it is locked out of every
    future month's entry with nothing to exit. entry_fillcheck status='filled' must still count."""
    rows = [
        {'action': 'entry', 'symbol': 'SPY', 'status': 'submitted'},
        {'action': 'entry', 'symbol': 'QQQ', 'status': 'submitted'},
        {'action': 'entry_fillcheck', 'symbol': 'SPY', 'status': 'expired'},
        {'action': 'entry_fillcheck', 'symbol': 'QQQ', 'status': 'filled'},
    ]
    assert tom.open_tom_symbols(rows) == {'QQQ'}


# ---------------------------------------------------------------------------
# check_pending_fills: status + reason logging and the partial_entry flag
# ---------------------------------------------------------------------------

def test_check_pending_fills_flags_expired_entry_as_partial_with_reason(tmp_path):
    ledger = str(tmp_path / 'ledger.csv')
    tom.append_ledger_row({
        'date': '2026-09-30', 'action': 'entry', 'symbol': 'SPY', 'qty': 26,
        'ref_price': '765.80', 'order_id': 'spy-order-1', 'status': 'submitted',
        'client_order_id': 'tom-202609-SPY-in', 'timestamp_utc': '2026-09-30T19:45:04+00:00',
        'partial_entry': '',
    }, ledger)
    a = make_alpaca()
    a.get_order.return_value = {'status': 'expired', 'filled_avg_price': None}
    n = make_notifier()
    messages = tom.check_pending_fills(a, n, ledger_path=ledger)
    assert len(messages) == 1
    assert 'expired' in messages[0] and 'PARTIAL ENTRY' in messages[0]
    rows = tom.read_ledger(ledger)
    fillcheck = next(r for r in rows if r['action'] == 'entry_fillcheck')
    assert fillcheck['status'] == 'expired'
    assert fillcheck['partial_entry'] == 'true'
    # Ledger honesty flows straight into held-symbol tracking: SPY was never really entered.
    assert 'SPY' not in tom.open_tom_symbols(rows)


def test_check_pending_fills_marks_filled_entry_not_partial(tmp_path):
    ledger = str(tmp_path / 'ledger.csv')
    tom.append_ledger_row({
        'date': '2026-09-30', 'action': 'entry', 'symbol': 'QQQ', 'qty': 26,
        'ref_price': '743.29', 'order_id': 'qqq-order-1', 'status': 'submitted',
        'client_order_id': 'tom-202609-QQQ-in', 'timestamp_utc': '2026-09-30T19:45:05+00:00',
        'partial_entry': '',
    }, ledger)
    a = make_alpaca()
    a.get_order.return_value = {'status': 'filled', 'filled_avg_price': 739.55}
    n = make_notifier()
    tom.check_pending_fills(a, n, ledger_path=ledger)
    rows = tom.read_ledger(ledger)
    fillcheck = next(r for r in rows if r['action'] == 'entry_fillcheck')
    assert fillcheck['status'] == 'filled'
    assert fillcheck['partial_entry'] == 'false'
    assert 'QQQ' in tom.open_tom_symbols(rows)


def test_in_late_window_boundaries():
    assert tom.in_late_window(et_dt(2026, 9, 30, 15, 57).astimezone(tom.ET))
    assert tom.in_late_window(et_dt(2026, 9, 30, 15, 59).astimezone(tom.ET))
    assert not tom.in_late_window(et_dt(2026, 9, 30, 16, 0).astimezone(tom.ET))
    assert not tom.in_action_window(et_dt(2026, 9, 30, 15, 57).astimezone(tom.ET))


# ---------------------------------------------------------------------------
# 2026-10-05 fix: fill-aware idempotency, late market fallback, next-session catch-up
# ---------------------------------------------------------------------------

def _row(date, action, sym, coid, status, order_id=None, qty=26):
    return {'date': date, 'action': action, 'symbol': sym, 'qty': qty, 'ref_price': '',
            'order_id': order_id or f'id-{coid}', 'status': status, 'client_order_id': coid,
            'timestamp_utc': f'{date}T19:45:00+00:00', 'partial_entry': '', 'deviation': ''}


def _qqq_held_ledger(tmp_path, exit_status=None):
    """QQQ entered (filled) 9/30; optionally an exit submitted 10/5 that resolved `exit_status`."""
    ledger = str(tmp_path / 'ledger.csv')
    tom.append_ledger_row(_row('2026-09-30', 'entry', 'QQQ', 'tom-202609-QQQ-in', 'submitted'), ledger)
    tom.append_ledger_row(_row('2026-09-30', 'entry_fillcheck', 'QQQ', 'tom-202609-QQQ-in', 'filled'), ledger)
    if exit_status:
        tom.append_ledger_row(_row('2026-10-05', 'exit', 'QQQ', 'tom-202610-QQQ-out', 'submitted'), ledger)
        if exit_status != 'pending':
            tom.append_ledger_row(_row('2026-10-05', 'exit_fillcheck', 'QQQ', 'tom-202610-QQQ-out', exit_status), ledger)
    return ledger


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(tom, '_sleep', lambda s: None)


def test_expired_exit_id_is_resubmitted_as_r1(tmp_path):
    ledger = _qqq_held_ledger(tmp_path, 'expired')
    a = make_alpaca(open_positions=[{'symbol': 'QQQ', 'qty': 26}])
    n = make_notifier()
    # 10/5 itself, same session, MOC window: expired id is still owed -> -r1
    tom.run(a, n, et_dt(2026, 10, 5, 15, 45), ledger_path=ledger)
    a.submit_moc_sell_order.assert_called_once_with('QQQ', 26, client_order_id='tom-202610-QQQ-out-r1')


def test_filled_exit_id_is_skipped(tmp_path):
    ledger = _qqq_held_ledger(tmp_path, 'filled')
    a = make_alpaca(open_positions=[{'symbol': 'QQQ', 'qty': 26}])
    tom.run(a, make_notifier(), et_dt(2026, 10, 5, 15, 45), ledger_path=ledger)
    a.submit_moc_sell_order.assert_not_called()


def test_live_unfilled_exit_in_late_window_cancels_then_market(tmp_path):
    ledger = _qqq_held_ledger(tmp_path, 'pending')
    a = make_alpaca(open_positions=[{'symbol': 'QQQ', 'qty': 26}],
                    open_orders=[{'client_order_id': 'tom-202610-QQQ-out', 'id': 'id-tom-202610-QQQ-out'}])
    a.get_order.return_value = {'status': 'canceled'}
    a.submit_market_sell_order = MagicMock(return_value={'id': 'mkt1', 'status': 'accepted'})
    n = make_notifier()
    tom.run(a, n, et_dt(2026, 10, 5, 15, 58), ledger_path=ledger)
    a.cancel_order.assert_called_once_with('id-tom-202610-QQQ-out')
    a.submit_market_sell_order.assert_called_once_with('QQQ', 26, client_order_id='tom-202610-QQQ-out-mkt')
    rows = tom.read_ledger(ledger)
    assert rows[-1]['action'] == 'late_fallback_replace'
    assert 'QQQ' not in tom.open_tom_symbols(rows)
    assert any('[TOM]' in c.args[0] for c in n.send_message_sync.call_args_list)


def test_late_fallback_unconfirmed_cancel_does_not_double_submit(tmp_path):
    ledger = _qqq_held_ledger(tmp_path, 'pending')
    a = make_alpaca(open_positions=[{'symbol': 'QQQ', 'qty': 26}],
                    open_orders=[{'client_order_id': 'tom-202610-QQQ-out', 'id': 'id-tom-202610-QQQ-out'}])
    a.get_order.return_value = {'status': 'pending_cancel'}
    a.submit_market_sell_order = MagicMock()
    tom.run(a, make_notifier(), et_dt(2026, 10, 5, 15, 58), ledger_path=ledger)
    a.submit_market_sell_order.assert_not_called()


def test_owed_exit_next_session_exits_with_deviation_tag(tmp_path):
    ledger = _qqq_held_ledger(tmp_path, 'expired')
    a = make_alpaca(open_positions=[{'symbol': 'QQQ', 'qty': 26}])
    n = make_notifier()
    summary = tom.run(a, n, et_dt(2026, 10, 6, 15, 45), ledger_path=ledger)
    a.submit_moc_sell_order.assert_called_once_with('QQQ', 26, client_order_id='tom-202610-QQQ-out-r1')
    last = tom.read_ledger(ledger)[-1]
    assert last['deviation'] == 'late_exit_1_sessions' and last['action'] == 'exit'
    assert 'late_exit_1_sessions' in summary


def test_owed_exit_dry_run_submits_nothing(tmp_path):
    ledger = _qqq_held_ledger(tmp_path, 'expired')
    a = make_alpaca(open_positions=[{'symbol': 'QQQ', 'qty': 26}])
    summary = tom.run(a, None, et_dt(2026, 10, 6, 15, 45), dry_run=True, ledger_path=ledger)
    assert 'DRY-RUN would SELL MOC 26 QQQ id=tom-202610-QQQ-out-r1' in summary
    a.submit_moc_sell_order.assert_not_called()


def test_midmonth_noop_unchanged_with_flat_ledger(tmp_path):
    a = make_alpaca()
    summary = tom.run(a, make_notifier(), et_dt(2026, 10, 14, 15, 45), ledger_path=str(tmp_path / 'l.csv'))
    assert 'not a turn-of-month' in summary


# ---------------------------------------------------------------------------
# 2026-10-07 fix: partial fills, catch-up-day late fallback, owed qty from the broker
# (docs/tom_sleeve_partial_fill_fix_20261007.md; incident = 10/6 QQQ MOC r1 filled 25 of 26)
# ---------------------------------------------------------------------------

R1 = 'tom-202610-QQQ-out-r1'


def _incident_ledger(tmp_path, r1_fillcheck=None, partial_row=False):
    """QQQ entered 26 on 9/30; exit r0 (10/5) expired; exit r1 (10/6) submitted. `r1_fillcheck` appends
    an r1 fill check with that status (the 20:45 UTC tick); `partial_row` adds the 25 @ 759.76 row."""
    ledger = _qqq_held_ledger(tmp_path, 'expired')
    tom.append_ledger_row(_row('2026-10-06', 'exit', 'QQQ', R1, 'submitted'), ledger)
    if r1_fillcheck:
        tom.append_ledger_row(_row('2026-10-06', 'exit_fillcheck', 'QQQ', R1, r1_fillcheck), ledger)
    if partial_row:
        tom.append_ledger_row(_row('2026-10-06', 'exit_fillcheck', 'QQQ', R1, 'partial_fill', qty=25), ledger)
    return ledger


def _orders_by_id(a, mapping):
    """get_order side effect: broker order dict per order id."""
    a.get_order.side_effect = lambda oid: mapping[oid]


def test_fillcheck_partial_exit_writes_partial_row_and_leaves_remainder_owed(tmp_path):
    ledger = _incident_ledger(tmp_path)
    a = make_alpaca()
    a.get_order.return_value = {'status': 'expired', 'qty': 26, 'filled_qty': 25, 'filled_avg_price': 759.76}
    n = make_notifier()
    msgs = tom.check_pending_fills(a, n, ledger_path=ledger)
    assert len(msgs) == 1 and '25/26' in msgs[0] and '759.76' in msgs[0] and 'owed' in msgs[0]
    assert '25/26' in n.send_message_sync.call_args.args[0]
    rows = tom.read_ledger(ledger)
    fc = rows[-1]
    assert (fc['action'], fc['status'], fc['qty'], fc['ref_price']) == ('exit_fillcheck', 'partial_fill', '25', '759.7600')
    assert fc['client_order_id'] == R1 and fc['partial_entry'] == 'false'
    assert 'QQQ' in tom.open_tom_symbols(rows)              # remainder still held / owed
    assert tom.ledger_open_qty(rows, 'QQQ') == 1


def test_fillcheck_full_fill_and_zero_fill_unchanged(tmp_path):
    ledger = _incident_ledger(tmp_path)
    a = make_alpaca()
    a.get_order.return_value = {'status': 'filled', 'qty': 26, 'filled_qty': 26, 'filled_avg_price': 759.76}
    tom.check_pending_fills(a, make_notifier(), ledger_path=ledger)
    assert tom.read_ledger(ledger)[-1]['status'] == 'filled'
    (tmp_path / 'z').mkdir()
    ledger2 = _incident_ledger(tmp_path / 'z')
    a.get_order.return_value = {'status': 'expired', 'qty': 26, 'filled_qty': 0, 'filled_avg_price': None}
    tom.check_pending_fills(a, make_notifier(), ledger_path=ledger2)
    assert tom.read_ledger(ledger2)[-1]['status'] == 'expired'


def test_partial_entry_fill_keeps_symbol_held_with_filled_qty(tmp_path):
    ledger = str(tmp_path / 'ledger.csv')
    tom.append_ledger_row(_row('2026-09-30', 'entry', 'SPY', 'tom-202609-SPY-in', 'submitted'), ledger)
    a = make_alpaca()
    a.get_order.return_value = {'status': 'expired', 'qty': 26, 'filled_qty': 10, 'filled_avg_price': 765.8}
    msgs = tom.check_pending_fills(a, make_notifier(), ledger_path=ledger)
    assert '10/26' in msgs[0]
    rows = tom.read_ledger(ledger)
    assert rows[-1]['status'] == 'partial_fill' and rows[-1]['partial_entry'] == 'true'
    assert 'SPY' in tom.open_tom_symbols(rows) and tom.ledger_open_qty(rows, 'SPY') == 10


def test_ledger_open_qty_counts_each_partial_order_once(tmp_path):
    ledger = _incident_ledger(tmp_path, partial_row=True)
    tom.append_ledger_row(_row('2026-10-06', 'exit_fillcheck', 'QQQ', R1, 'partial_fill', qty=25), ledger)  # duplicate
    assert tom.ledger_open_qty(tom.read_ledger(ledger), 'QQQ') == 1
    assert tom.ledger_open_qty([], 'QQQ') is None


def test_owed_qty_comes_from_the_broker_not_the_ledger(tmp_path, caplog):
    ledger = _incident_ledger(tmp_path, r1_fillcheck='expired')           # ledger still says 26
    a = make_alpaca(open_positions=[{'symbol': 'QQQ', 'qty': 1}])
    with caplog.at_level('WARNING', logger='tom_sleeve'):
        tom.run(a, make_notifier(), et_dt(2026, 10, 7, 15, 45), ledger_path=ledger)
    a.submit_moc_sell_order.assert_called_once_with('QQQ', 1, client_order_id='tom-202610-QQQ-out-r2')
    assert any('broker qty 1' in r.getMessage() and 'ledger' in r.getMessage() and '26' in r.getMessage()
               for r in caplog.records)


def test_owed_qty_never_exceeds_ledger_open_qty(tmp_path):
    ledger = _incident_ledger(tmp_path, partial_row=True)                 # ledger open = 1
    a = make_alpaca(open_positions=[{'symbol': 'QQQ', 'qty': 40}])        # broker holds more than ours
    tom.run(a, make_notifier(), et_dt(2026, 10, 7, 15, 45), ledger_path=ledger)
    a.submit_moc_sell_order.assert_called_once_with('QQQ', 1, client_order_id='tom-202610-QQQ-out-r2')


def test_dry_run_after_partial_sells_one_share_r2_with_two_sessions(tmp_path):
    ledger = _incident_ledger(tmp_path, partial_row=True)
    a = make_alpaca(open_positions=[{'symbol': 'QQQ', 'qty': 1}])
    summary = tom.run(a, None, et_dt(2026, 10, 7, 15, 45), dry_run=True, ledger_path=ledger)
    assert ('DRY-RUN would SELL MOC 1 QQQ id=tom-202610-QQQ-out-r2 deviation=late_exit_2_sessions') in summary
    a.submit_moc_sell_order.assert_not_called()


def test_late_fallback_fires_on_catch_up_day_with_live_order(tmp_path):
    """10/6 19:58 UTC incident: r1 submitted 15:45 ET, still live at 15:58, day is NOT a TOM session."""
    ledger = _incident_ledger(tmp_path)                                    # r1 submitted, no fill check yet
    a = make_alpaca(open_positions=[{'symbol': 'QQQ', 'qty': 26}],
                    open_orders=[{'client_order_id': R1, 'id': f'id-{R1}'}])
    a.get_order.return_value = {'status': 'canceled'}
    a.submit_market_sell_order = MagicMock(return_value={'id': 'mkt1', 'status': 'accepted'})
    summary = tom.run(a, make_notifier(), et_dt(2026, 10, 6, 15, 58), ledger_path=ledger)
    a.cancel_order.assert_called_once_with(f'id-{R1}')
    a.submit_market_sell_order.assert_called_once_with('QQQ', 26, client_order_id='tom-202610-QQQ-out-mkt')
    assert 'not a turn-of-month' not in summary
    assert tom.read_ledger(ledger)[-1]['action'] == 'late_fallback_replace'


def test_catch_up_day_live_order_is_left_alone_in_moc_window(tmp_path):
    ledger = _incident_ledger(tmp_path)
    a = make_alpaca(open_positions=[{'symbol': 'QQQ', 'qty': 26}],
                    open_orders=[{'client_order_id': R1, 'id': f'id-{R1}'}])
    tom.run(a, make_notifier(), et_dt(2026, 10, 6, 15, 50), ledger_path=ledger)
    a.submit_moc_sell_order.assert_not_called()
    a.cancel_order.assert_not_called()


def test_partial_fill_flow_end_to_end_through_the_csv(tmp_path):
    """Integration: fill check writes the partial row to the CSV, the next session reads it back and
    sells exactly the 1 owed share as -r2 with the right session count."""
    ledger = _incident_ledger(tmp_path)
    a = make_alpaca(open_positions=[{'symbol': 'QQQ', 'qty': 1}])
    a.get_order.return_value = {'status': 'expired', 'qty': 26, 'filled_qty': 25, 'filled_avg_price': 759.76}
    tom.check_pending_fills(a, make_notifier(), ledger_path=ledger)
    tom.run(a, make_notifier(), et_dt(2026, 10, 7, 15, 45), ledger_path=ledger)
    a.submit_moc_sell_order.assert_called_once_with('QQQ', 1, client_order_id='tom-202610-QQQ-out-r2')
    last = tom.read_ledger(ledger)[-1]
    assert (last['action'], last['qty'], last['deviation']) == ('exit', '1', 'late_exit_2_sessions')


def _reconcile_broker(a):
    _orders_by_id(a, {
        f'id-{R1}': {'status': 'expired', 'qty': 26, 'filled_qty': 25, 'filled_avg_price': 759.76},
        'id-tom-202610-QQQ-out': {'status': 'expired', 'qty': 26, 'filled_qty': 0, 'filled_avg_price': None},
        'id-tom-202609-QQQ-in': {'status': 'filled', 'qty': 26, 'filled_qty': 26, 'filled_avg_price': 739.55},
    })


def test_reconcile_appends_exactly_one_partial_row_and_is_idempotent(tmp_path):
    ledger = _incident_ledger(tmp_path, r1_fillcheck='expired')
    a = make_alpaca()
    _reconcile_broker(a)
    before = len(tom.read_ledger(ledger))
    msgs = tom.reconcile_ledger(a, None, ledger_path=ledger)
    rows = tom.read_ledger(ledger)
    assert len(msgs) == 1 and len(rows) == before + 1
    new = rows[-1]
    assert (new['action'], new['status'], new['qty'], new['ref_price'], new['client_order_id'], new['date']) == \
           ('exit_fillcheck', 'partial_fill', '25', '759.7600', R1, '2026-10-06')
    assert tom.ledger_open_qty(rows, 'QQQ') == 1 and 'QQQ' in tom.open_tom_symbols(rows)
    assert tom.reconcile_ledger(a, None, ledger_path=ledger) == []        # second run: nothing
    assert len(tom.read_ledger(ledger)) == before + 1
    a.trading_client.submit_order.assert_not_called()
    a.submit_moc_sell_order.assert_not_called()
    a.cancel_order.assert_not_called()


def test_reconcile_dry_run_writes_nothing(tmp_path):
    ledger = _incident_ledger(tmp_path, r1_fillcheck='expired')
    a = make_alpaca()
    _reconcile_broker(a)
    before = tom.read_ledger(ledger)
    msgs = tom.reconcile_ledger(a, None, ledger_path=ledger, dry_run=True)
    assert len(msgs) == 1 and 'DRY-RUN' in msgs[0] and tom.read_ledger(ledger) == before


def test_reconcile_refuses_when_a_later_exit_row_exists(tmp_path, caplog):
    """A partial row appended after a newer exit row would re-add the symbol to the held set."""
    ledger = _incident_ledger(tmp_path, r1_fillcheck='expired')
    tom.append_ledger_row(_row('2026-10-07', 'exit', 'QQQ', 'tom-202610-QQQ-out-r2', 'submitted'), ledger)
    a = make_alpaca()
    _reconcile_broker(a)
    before = len(tom.read_ledger(ledger))
    with caplog.at_level('ERROR', logger='tom_sleeve'):
        assert tom.reconcile_ledger(a, None, ledger_path=ledger) == []
    assert len(tom.read_ledger(ledger)) == before and any('LATER' in r.getMessage() for r in caplog.records)


def test_main_reconcile_flag_calls_reconcile_and_places_no_orders(monkeypatch, capsys):
    calls = {}
    fake_client = make_alpaca()
    monkeypatch.setattr(tom, 'Config', lambda: types.SimpleNamespace(
        alpaca_orb_api_key='k', alpaca_orb_api_secret='s', alpaca_orb_paper=True,
        telegram_bot_token='', telegram_chat_id=''))
    monkeypatch.setattr(tom, 'AlpacaClient', lambda *a, **k: fake_client)

    def fake_reconcile(client, notifier, dry_run=False):
        calls['dry'] = dry_run
        return []

    monkeypatch.setattr(tom, 'reconcile_ledger', fake_reconcile)
    monkeypatch.setattr(tom, 'run', lambda *a, **k: pytest.fail('run() must not execute on --reconcile'))
    monkeypatch.setattr(sys, 'argv', ['tom_sleeve.py', '--reconcile'])
    assert tom.main() == 0
    assert calls == {'dry': False}
    fake_client.submit_moc_sell_order.assert_not_called()
