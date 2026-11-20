"""2026-11-20 NBIZ: the 60-s broker reconcile adopted our OWN resting-order fill before the fill poll registered it.
Part 1: periodic adoption defers to / registers through `_on_live_fill` while the resting order is still tracked.
Part 2: a safety net blocked by an exit-coverage stop adopts that stop instead of alarming UNPROTECTED."""
import logging
from unittest.mock import MagicMock

import pytest

from data_sources.alpaca_client import AlpacaClient, AlpacaAPIError
from trading.stop_monitor import StopMonitor
from trading.hod_break_engine import HodBreakEngine, Candidate
from tests.hod_fills_helper import set_our_fills
from tests.test_hod_break_engine import cfg as hod_cfg
from tests.test_hod_break_engine_boot_defects import real_db, mock_alpaca, mock_sm, TODAY  # noqa: F401  (fixtures)

LOG = 'trading.hod_break_engine'


def _engine(alpaca, db, sm, tmp_path):
    """Live resting-mode engine with every ledger/state path in tmp_path."""
    c = hod_cfg(dry_run=False, entry_mode='resting_stop_limit',
                live_parity_ledger_path=str(tmp_path / 'parity.csv'),
                live_orders_state_path=str(tmp_path / 'state.json'),
                dry_ledger_path=str(tmp_path / 'dry.csv'))
    eng = HodBreakEngine(alpaca, db, sm, cfg=c)
    eng._roll_session()
    return eng


def _nbiz(eng, order_id='eb5ba207'):
    """NBIZ candidate with a tracked, unbooked resting order (the engine's real live_order keys)."""
    cand = Candidate(symbol='NBIZ', day_open=70.0, adv20=1e6)
    cand.live_order = {'order_id': order_id, 'coid': 'hod-rest-NBIZ-11-20-ab12cd34', 'qty': 18, 'stop': 75.21 - 2.0,
                       'limit': 75.31, 'level': 75.0, 'trigger': 75.21, 'booked_qty': 0, 'tp_leg_id': '', 'sl_leg_id': '',
                       'trade_id': None}
    eng.candidates['NBIZ'] = cand
    return cand


def _setup(mock_alpaca, order_status):
    """Broker long 18 NBIZ @ 75.30, OCO that succeeds, get_order -> order_status."""
    mock_alpaca.get_open_positions.return_value = [{'symbol': 'NBIZ', 'qty': 18, 'avg_entry_price': 75.30}]
    set_our_fills(mock_alpaca, {'NBIZ': 18})
    mock_alpaca.submit_oco_sell_order.return_value = {'id': 'tp-parent', 'legs': [{'id': 'sl-leg', 'type': 'stop'}]}
    mock_alpaca.get_order.return_value = order_status
    mock_alpaca.get_signed_position_qty = MagicMock(return_value=18)


FILLED = {'status': 'filled', 'filled_qty': 18, 'filled_avg_price': 75.30}


def test_1_nbiz_replay_registers_through_on_live_fill(real_db, mock_alpaca, mock_sm, tmp_path, caplog):
    _setup(mock_alpaca, FILLED)
    eng = _engine(mock_alpaca, real_db, mock_sm, tmp_path)
    cand = _nbiz(eng)
    with caplog.at_level(logging.INFO, logger=LOG):
        eng._adopt_unregistered_positions_on_boot(periodic=True)
    assert cand.live_filled is True and cand.live_order is None
    assert eng.positions['NBIZ'].sl_leg_id == 'sl-leg'
    rows = real_db.get_open_trades(TODAY, strategy='hod_break')
    assert len(rows) == 1 and rows[0]['order_status'] == 'filled' and rows[0]['shares'] == 18 and rows[0]['fill_price'] == 75.30
    assert mock_sm.add_watch.call_count == 1 and mock_sm.add_watch.call_args.kwargs['sl_leg_id']
    assert any('registering through _on_live_fill' in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
    assert not any('ADOPTED' in r.getMessage() for r in caplog.records)


def test_2_no_new_fill_defers_then_backstop_adopts(real_db, mock_alpaca, mock_sm, tmp_path, caplog):
    _setup(mock_alpaca, {'status': 'new', 'filled_qty': 0})
    eng = _engine(mock_alpaca, real_db, mock_sm, tmp_path)
    cand = _nbiz(eng)
    with caplog.at_level(logging.INFO, logger=LOG):
        eng._adopt_unregistered_positions_on_boot(periodic=True)
    assert real_db.get_open_trades(TODAY, strategy='hod_break') == []
    assert 'NBIZ' not in eng.positions and mock_sm.add_watch.call_count == 0
    assert any('waiting for the fill poll' in r.getMessage() and r.levelno == logging.INFO for r in caplog.records)
    caplog.clear()
    eng._adopt_deferred_since['NBIZ'] -= 200
    with caplog.at_level(logging.INFO, logger=LOG):
        eng._adopt_unregistered_positions_on_boot(periodic=True)
    assert len(real_db.get_open_trades(TODAY, strategy='hod_break')) == 1
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any('adopted after' in m and 'still unresolved' in m for m in msgs)
    assert not any('on boot' in m for m in msgs)
    assert 'NBIZ' not in eng._adopt_deferred_since


def test_3_get_order_raises_defers(real_db, mock_alpaca, mock_sm, tmp_path):
    _setup(mock_alpaca, None)
    mock_alpaca.get_order.side_effect = AlpacaAPIError('boom')
    eng = _engine(mock_alpaca, real_db, mock_sm, tmp_path)
    _nbiz(eng)
    eng._adopt_unregistered_positions_on_boot(periodic=True)
    assert real_db.get_open_trades(TODAY, strategy='hod_break') == [] and 'NBIZ' not in eng.positions
    assert 'NBIZ' in eng._adopt_deferred_since


def test_4_virt_style_adopts_with_reconcile_wording(real_db, mock_alpaca, mock_sm, tmp_path, caplog):
    _setup(mock_alpaca, None)
    eng = _engine(mock_alpaca, real_db, mock_sm, tmp_path)
    cand = Candidate(symbol='NBIZ', day_open=70.0, adv20=1e6)
    eng.candidates['NBIZ'] = cand
    eng._live_armed_stops['NBIZ'] = 73.0
    with caplog.at_level(logging.INFO, logger=LOG):
        eng._adopt_unregistered_positions_on_boot(periodic=True)
    assert len(real_db.get_open_trades(TODAY, strategy='hod_break')) == 1 and 'NBIZ' in eng.positions
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and 'ADOPTED' in r.getMessage()]
    assert msgs and all('on reconcile' in m and 'on boot' not in m for m in msgs)


def test_5_boot_mode_unchanged(real_db, mock_alpaca, mock_sm, tmp_path):
    _setup(mock_alpaca, FILLED)
    eng = _engine(mock_alpaca, real_db, mock_sm, tmp_path)
    cand = _nbiz(eng)
    eng._adopt_unregistered_positions_on_boot()
    rows = real_db.get_open_trades(TODAY, strategy='hod_break')
    assert len(rows) == 1 and rows[0]['shares'] == 18
    assert cand.live_order['trade_id'] == rows[0]['id']
    mock_alpaca.get_order.assert_not_called()


def test_6_rest_fallback_after_registration_books_nothing_new(real_db, mock_alpaca, mock_sm, tmp_path):
    _setup(mock_alpaca, FILLED)
    eng = _engine(mock_alpaca, real_db, mock_sm, tmp_path)
    _nbiz(eng)
    eng._adopt_unregistered_positions_on_boot(periodic=True)
    eng._last_rest_fallback_ts = 0.0
    eng._poll_live_fills_rest_fallback()
    assert len(real_db.get_open_trades(TODAY, strategy='hod_break')) == 1
    assert list(eng.positions) == ['NBIZ'] and mock_sm.add_watch.call_count == 1


def _net_engine(mock_alpaca, mock_sm, real_db, tmp_path, order, snap):
    mock_alpaca.submit_oco_sell_order.side_effect = AlpacaAPIError('40310000 insufficient qty available')
    mock_alpaca.submit_stop_sell_order.side_effect = AlpacaAPIError('40310000 insufficient qty available')
    mock_sm.get_watch_snapshot.return_value = snap
    mock_alpaca.get_order.return_value = order
    eng = _engine(mock_alpaca, real_db, mock_sm, tmp_path)
    eng._notify = MagicMock()
    return eng


def test_7_safety_net_adopts_exit_coverage_stop(real_db, mock_alpaca, mock_sm, tmp_path, caplog):
    eng = _net_engine(mock_alpaca, mock_sm, real_db, tmp_path,
                      {'status': 'accepted', 'type': 'stop', 'stop_price': 72.5}, {'sl_leg_id': '58096ff1', 'tp_leg_id': ''})
    with caplog.at_level(logging.INFO, logger=LOG):
        assert eng._place_safety_net('NBIZ', 18, 80.0, 70.0) == ('', '58096ff1')
    assert any('exit-coverage stop' in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
    assert not any(r.levelno >= logging.ERROR and 'FAILED twice' in r.getMessage() for r in caplog.records)
    assert not any('UNPROTECTED' in str(c) for c in eng._notify.call_args_list)


def test_7b_canceled_coverage_stop_keeps_error(real_db, mock_alpaca, mock_sm, tmp_path, caplog):
    eng = _net_engine(mock_alpaca, mock_sm, real_db, tmp_path,
                      {'status': 'canceled', 'type': 'stop', 'stop_price': 72.5}, {'sl_leg_id': '58096ff1', 'tp_leg_id': ''})
    with caplog.at_level(logging.INFO, logger=LOG):
        assert eng._place_safety_net('NBIZ', 18, 80.0, 70.0) == ('', '')
    assert any('FAILED twice' in r.getMessage() for r in caplog.records)
    assert any('UNPROTECTED' in str(c) for c in eng._notify.call_args_list)


def test_7c_no_watch_keeps_error(real_db, mock_alpaca, mock_sm, tmp_path, caplog):
    eng = _net_engine(mock_alpaca, mock_sm, real_db, tmp_path, {'status': 'accepted', 'type': 'stop'}, None)
    with caplog.at_level(logging.INFO, logger=LOG):
        assert eng._place_safety_net('NBIZ', 18, 80.0, 70.0) == ('', '')
    assert any('FAILED twice' in r.getMessage() for r in caplog.records)
    assert any('UNPROTECTED' in str(c) for c in eng._notify.call_args_list)


def test_8_watch_snapshot_carries_leg_ids():
    sm = StopMonitor(api_key='k', api_secret='s', alpaca_client=MagicMock(spec=AlpacaClient))
    sm.add_watch(symbol='NBIZ', stop_price=72.0, shares=18, tp_leg_id='tp-1', sl_leg_id='sl-1', trade_db_id=1,
                 entry_price=75.3, risk_per_share=3.3, strategy='hod_break')
    snap = sm.get_watch_snapshot('NBIZ')
    assert snap['sl_leg_id'] == 'sl-1' and snap['tp_leg_id'] == 'tp-1'
