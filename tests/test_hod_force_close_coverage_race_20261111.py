"""2026-11-11 AXSM: force close vs exit-coverage race (docs/hod_force_close_coverage_race_spec_20261111.md).
(a) the StopMonitor yields to a close in flight, (b) a force-close retry cancels OUR open sells first,
(c) a carried-over row is closed at the open, plus an integration replay of the AXSM sequence."""
import logging
import time
from unittest.mock import MagicMock, patch

from data_sources.alpaca_client import AlpacaClient
from trading.hod_break_engine import HodBreakEngine
from trading.stop_monitor import StopMonitor
from tests.test_hod_break_engine import (  # noqa: F401  (fixtures)
    trades_db, mock_alpaca, mock_db, mock_sm, engine, bars_df, admit,
)
from tests.test_hod_break import drive_then_consolidate

HELD = Exception('{"code":40310000,"message":"insufficient qty available for order (requested: 5, available: 0)"}')


def _open_pos(engine, mock_alpaca):
    admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1]))
    pos = engine.positions['ABC']; pos.status = 'open'; pos.fill_price = 11.05
    pos.tp_leg_id, pos.sl_leg_id = 'tp1', 'sl1'
    mock_alpaca.get_open_positions.return_value = [{'symbol': 'ABC', 'qty': pos.shares}]
    mock_alpaca.get_order.return_value = {'status': 'canceled', 'filled_qty': 0}
    engine._notify = MagicMock(); engine.fc_leg_poll_s = 0
    return pos


def _pass(engine, minute=955, **kw):
    with patch.object(HodBreakEngine, '_minute_of_day', return_value=minute), patch('trading.hod_break_engine.time.sleep'):
        return engine.force_close_all(**kw)


def _sell(oid, coid='', symbol='ABC'):
    return {'id': oid, 'client_order_id': coid, 'symbol': symbol, 'side': 'sell', 'status': 'new', 'qty': 9}


def _monitor(client, shares=9):
    m = StopMonitor(api_key='k', api_secret='s', alpaca_client=client)
    m.add_watch(symbol='ABC', stop_price=10.5, shares=shares, tp_leg_id='', sl_leg_id='sl1', entry_price=11.05,
                risk_per_share=0.5, strategy='hod_break')
    m._watches['ABC'].added_at = time.time() - 600
    return m


# ---- (a) coverage yields to a close in flight --------------------------------------------------------------
def test_coverage_skip_true_while_marked_false_after_clear_and_after_expiry(caplog):
    client = MagicMock(spec=AlpacaClient)
    m = _monitor(client); w = m._watches['ABC']; now = time.time()
    assert m._coverage_skip('ABC', w, now) is False
    m.mark_closing('hod_break', 'ABC')
    assert m._coverage_skip('ABC', w, now) is True
    m.clear_closing('ABC')
    assert m._coverage_skip('ABC', w, now) is False
    m.mark_closing('hod_break', 'ABC')
    with caplog.at_level(logging.WARNING):
        assert m._coverage_skip('ABC', w, now + StopMonitor._CLOSING_MARK_TTL_S + 1) is False
    assert any(r.levelname == 'WARNING' and 'close mark expired' in r.message for r in caplog.records)
    assert 'ABC' not in m._closing_marks


def test_mark_of_another_strategy_does_not_skip():
    m = _monitor(MagicMock(spec=AlpacaClient)); m.mark_closing('orb', 'ABC')
    assert m._coverage_skip('ABC', m._watches['ABC'], time.time()) is False


def test_remove_watch_clears_the_mark():
    m = _monitor(MagicMock(spec=AlpacaClient)); m.mark_closing('hod_break', 'ABC'); m.remove_watch('ABC')
    assert 'ABC' not in m._closing_marks


def test_force_close_marks_before_cancelling_any_leg(engine, mock_alpaca, mock_sm):
    _open_pos(engine, mock_alpaca)
    order = []
    mock_sm.mark_closing.side_effect = lambda *a: order.append('mark')
    mock_alpaca.cancel_order.side_effect = lambda *a, **k: order.append('cancel') or True
    mock_alpaca.submit_limit_sell_order.return_value = {'id': 'c1', 'status': 'accepted'}
    _pass(engine)
    assert order[0] == 'mark' and 'cancel' in order
    mock_sm.mark_closing.assert_called_once_with('hod_break', 'ABC')


# ---- (b) retry cancels what holds the shares -----------------------------------------------------------------
def test_attempt2_cancels_own_sells_then_waits_then_submits_foreign_untouched(engine, mock_alpaca, mock_sm):
    pos = _open_pos(engine, mock_alpaca); pos.fc_attempts = 1
    mock_sm.coverage_order_ids.return_value = {'cov1'}
    mock_alpaca.get_open_orders.return_value = [_sell('cov1', 'uuid-xyz'), _sell('hodx', 'hod-fc-ABC-09-aa'), _sell('own1', 'owner-manual')]
    mock_alpaca.submit_limit_sell_order.return_value = {'id': 'c1', 'status': 'accepted'}
    _pass(engine)
    names = [(c[0], c[1][0] if c[1] else None) for c in mock_alpaca.mock_calls
             if c[0] in ('cancel_order', 'get_order', 'submit_limit_sell_order')]
    cancelled = [a for n, a in names if n == 'cancel_order']
    assert 'cov1' in cancelled and 'hodx' in cancelled and 'own1' not in cancelled
    last_cancel = max(i for i, (n, a) in enumerate(names) if n == 'cancel_order' and a in ('cov1', 'hodx'))
    polled_after = [i for i, (n, a) in enumerate(names) if n == 'get_order' and a in ('cov1', 'hodx') and i > last_cancel]
    submit_at = next(i for i, (n, _) in enumerate(names) if n == 'submit_limit_sell_order')
    assert polled_after and max(polled_after) < submit_at     # waited for terminal BEFORE selling
    assert pos.close_order_id == 'c1'


def test_held_error_on_attempt1_is_warning_attempt2_after_the_sweep_is_error(engine, mock_alpaca, caplog):
    pos = _open_pos(engine, mock_alpaca)
    mock_alpaca.submit_limit_sell_order.side_effect = HELD
    with caplog.at_level(logging.WARNING):
        _pass(engine)                                          # attempt 1, legs terminal: no sweep -> WARNING
    assert not [r for r in caplog.records if r.levelname == 'ERROR'] and pos.fc_attempts == 1
    with caplog.at_level(logging.WARNING):
        _pass(engine)                                          # attempt 2: swept, still held -> genuine ERROR
    assert any(r.levelname == 'ERROR' and 'FORCE CLOSE ABC FAILED' in r.message for r in caplog.records)


def test_attempt1_with_leg_poll_timeout_also_sweeps(engine, mock_alpaca):
    _open_pos(engine, mock_alpaca)
    mock_alpaca.get_order.return_value = {'status': 'pending_cancel', 'filled_qty': 0}      # leg never settles
    mock_alpaca.get_open_orders.return_value = [_sell('sl1', 'x')]
    mock_alpaca.submit_limit_sell_order.return_value = {'id': 'c1', 'status': 'accepted'}
    _pass(engine)
    mock_alpaca.get_open_orders.assert_called_once()


# ---- (c) overnight leftover closes at the open -------------------------------------------------------------------
def test_overnight_leftover_closed_at_0931_not_before_and_not_waiting_for_1555(engine, mock_alpaca):
    pos = _open_pos(engine, mock_alpaca); pos.overnight_from = '2020-01-01'
    mock_alpaca.submit_limit_sell_order.return_value = {'id': 'c1', 'status': 'accepted'}
    with patch.object(HodBreakEngine, '_minute_of_day', return_value=570):
        assert engine._close_overnight_leftovers() == 0
    mock_alpaca.submit_limit_sell_order.assert_not_called()
    with patch.object(HodBreakEngine, '_minute_of_day', return_value=571), patch('trading.hod_break_engine.time.sleep'):
        assert engine._close_overnight_leftovers() == 1
    mock_alpaca.submit_limit_sell_order.assert_called_once()
    assert pos.close_order_id == 'c1' and engine._flattened is False       # `only=` never marks the book flat


def test_todays_position_is_not_an_overnight_leftover(engine, mock_alpaca):
    _open_pos(engine, mock_alpaca)
    with patch.object(HodBreakEngine, '_minute_of_day', return_value=600):
        assert engine._close_overnight_leftovers() == 0


def test_adoption_flags_the_row_and_warns(engine, mock_alpaca, mock_db, caplog):
    mock_alpaca.get_open_positions.return_value = [{'symbol': 'AXSM', 'qty': 9}]
    mock_db.get_open_trades_between.return_value = [{
        'symbol': 'AXSM', 'id': 5, 'order_status': 'filled', 'trade_date': '2020-01-01', 'shares': 9, 'entry_price': 145.5,
        'stop_loss_price': 144.9, 'take_profit_price': 147.0, 'fill_price': 145.5, 'pattern_data': '{}', 'order_id': 'o'}]
    with patch.object(engine, '_cancel_and_settle_safety_net', return_value=None), \
         patch.object(engine, '_place_safety_net', return_value=('tp9', 'sl9')), caplog.at_level(logging.WARNING):
        engine._adopt_prior_day_open_rows('2026-11-12', {'AXSM': 9})
    assert engine.positions['AXSM'].overnight_from == '2020-01-01'
    assert any('AXSM: overnight leftover from 2020-01-01 — closing at the open' in r.message for r in caplog.records)


# ---- integration: AXSM replay with a REAL StopMonitor ---------------------------------------------------------
def test_axsm_sequence_no_new_stop_no_error_position_closed(engine, mock_alpaca, caplog):
    pos = _open_pos(engine, mock_alpaca)
    m = _monitor(mock_alpaca, shares=pos.shares); engine.stop_monitor = m
    mock_alpaca.get_open_orders.return_value = [_sell('sl1', 'x')]
    mock_alpaca.get_order.return_value = {'status': 'pending_cancel', 'filled_qty': 0}      # the leg is still not terminal
    mock_alpaca.submit_stop_sell_order.return_value = {'id': 'covstop-12345678'}
    cancelled = []
    mock_alpaca.cancel_order.side_effect = lambda oid, *a, **k: cancelled.append(oid) or True

    def _sell_ok(*a, **k):
        assert 'sl1' in cancelled                                  # the broker releases the shares only after our cancel
        return {'id': 'c1', 'status': 'accepted'}
    mock_alpaca.submit_limit_sell_order.side_effect = _sell_ok
    with caplog.at_level(logging.INFO):
        _pass(engine)                                              # 20:55:00 force close: cancel, leg pending, sweep, sell
        mock_alpaca.get_open_orders.return_value = []              # the broker shows no live exit order -> a naked-looking position
        counts = m._exit_coverage_cycle()                          # 20:55:33 coverage cycle
        assert counts['checked'] == 0
        mock_alpaca.submit_stop_sell_order.assert_not_called()
        mock_alpaca.get_order.return_value = {'status': 'filled', 'filled_qty': pos.shares, 'filled_avg_price': 11.0}
        _pass(engine)                                              # next pass books the fill
    assert 'ABC' not in engine.positions
    assert not [r for r in caplog.records if r.levelname in ('ERROR', 'CRITICAL')]
    engine._notify.assert_not_called()


def test_control_without_the_mark_coverage_would_have_placed_the_stop(engine, mock_alpaca):
    pos = _open_pos(engine, mock_alpaca)
    m = _monitor(mock_alpaca, shares=pos.shares)
    mock_alpaca.get_open_orders.return_value = []
    mock_alpaca.submit_stop_sell_order.return_value = {'id': 'covstop-12345678'}
    assert m._exit_coverage_cycle()['repaired'] == 1
