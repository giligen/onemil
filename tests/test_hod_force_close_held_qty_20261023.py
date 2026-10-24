"""2026-10-22 CELC/PHM: attempt-1 held_for_orders race on force close is a WARNING with no Telegram."""
import logging
from unittest.mock import MagicMock, patch

from trading.hod_break_engine import HodBreakEngine
from tests.test_hod_break_engine import (  # noqa: F401  (fixtures)
    trades_db, mock_alpaca, mock_db, mock_sm, engine, bars_df, admit,
)
from tests.test_hod_break import drive_then_consolidate

HELD = Exception('{"code":40310000,"message":"insufficient qty available for order (requested: 5, available: 0)"}')


def _open_pos(engine, mock_alpaca):
    admit(engine); engine._ingest_bars('ABC', bars_df(drive_then_consolidate()[:-1]))
    pos = engine.positions['ABC']; pos.status = 'open'; pos.fill_price = 11.05
    mock_alpaca.get_open_positions.return_value = [{'symbol': 'ABC', 'qty': pos.shares}]
    engine._notify = MagicMock()
    return pos


def _pass(engine):
    with patch.object(HodBreakEngine, '_minute_of_day', return_value=955), patch('trading.hod_break_engine.time.sleep'):
        return engine.force_close_all()


def test_leg_poll_is_longer_than_paper_cancel_propagation():
    assert HodBreakEngine.FC_LEG_POLL_S >= 10.0


def test_attempt1_held_qty_warns_no_notify_and_next_pass_resubmits(engine, mock_alpaca, caplog):
    pos = _open_pos(engine, mock_alpaca)
    mock_alpaca.submit_limit_sell_order.side_effect = HELD
    with caplog.at_level(logging.WARNING):
        _pass(engine)
    assert 'ABC' in engine.positions and pos.fc_attempts == 1 and pos.close_order_id is None
    assert not [r for r in caplog.records if r.levelname == 'ERROR']
    assert any('still held by a cancelling leg (attempt 1)' in r.message and r.levelname == 'WARNING' for r in caplog.records)
    engine._notify.assert_not_called()
    mock_alpaca.submit_limit_sell_order.side_effect = None
    mock_alpaca.submit_limit_sell_order.return_value = {'id': 'c1', 'status': 'accepted'}
    _pass(engine)
    assert pos.close_order_id == 'c1' and pos.fc_attempts == 2


def test_attempt2_held_qty_is_error_and_notifies(engine, mock_alpaca, caplog):
    pos = _open_pos(engine, mock_alpaca); pos.fc_attempts = 1
    mock_alpaca.submit_limit_sell_order.side_effect = HELD
    with caplog.at_level(logging.WARNING):
        _pass(engine)
    assert any(r.levelname == 'ERROR' and 'FORCE CLOSE ABC FAILED' in r.message for r in caplog.records)
    engine._notify.assert_called_once()


def test_other_exception_attempt1_is_error_and_notifies(engine, mock_alpaca, caplog):
    _open_pos(engine, mock_alpaca)
    mock_alpaca.submit_limit_sell_order.side_effect = Exception('boom: connection reset')
    with caplog.at_level(logging.WARNING):
        _pass(engine)
    assert any(r.levelname == 'ERROR' and 'FORCE CLOSE ABC FAILED' in r.message for r in caplog.records)
    engine._notify.assert_called_once()
