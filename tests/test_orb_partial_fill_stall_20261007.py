"""
AAOZ 2026-10-07 regression: the partial-fill stall branch cancelled "0 sh".

Incident: stop-limit buy 200 sh, 170 filled, the entry poll logged
``partial-fill stall — 170/0 sh ... cancelling remaining 0 sh`` and sent NO
cancel; the broker order stayed live for 30 more shares while StopMonitor
watched 170.

Real key: the REST payload (``AlpacaClient.get_order``) carries ``qty`` but the
OrderStreamWatcher payload (``trading/order_stream._order_to_dict``, the first
source ``_process_pending_fills`` reads) has NO ``qty`` key at all, so
``order_status.get('qty', 0)`` was 0 and ``_remaining = max(0 - 170, 0) = 0``.
Fix: the requested size is the engine's own record (``pos.shares``); the
payload qty is only a cross-check (WARNING on a mismatch), the cancel is
confirmed by a re-fetch (canceled/expired/filled) before the fill qty is
confirmed, and an unconfirmed cancel past the stall timeout logs ERROR.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.order_stream import OrderStreamWatcher
from trading.orb_engine import ORBEngine, OpenPosition
from trading.stop_monitor import StopMonitor

ORDER_ID = 'aaoz-parent'


@pytest.fixture
def orb_cfg():
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    cfg.setdefault('fill_handling', {})['partial_fill_stall_seconds_max'] = 1
    return cfg


@pytest.fixture
def mock_alpaca():
    c = MagicMock(spec=AlpacaClient)
    c.get_open_positions.return_value = []
    c.get_account_info.return_value = {'buying_power': 100_000.0}
    c.cancel_order.return_value = True
    c.get_daily_bars.return_value = {}
    return c


@pytest.fixture
def mock_db():
    db = MagicMock(spec=Database)
    db.save_trade.return_value = 1
    db.update_trade.return_value = True
    return db


@pytest.fixture
def mock_sm():
    sm = MagicMock(spec=StopMonitor)
    sm.polling_mode = False
    sm.drain_exit_events.return_value = []
    return sm


@pytest.fixture
def engine(orb_cfg, mock_alpaca, mock_db, mock_sm):
    return ORBEngine(alpaca_client=mock_alpaca, db=mock_db,
                     stop_monitor=mock_sm, config=orb_cfg)


def _pos(shares: int = 200) -> OpenPosition:
    """Pending position, stall clock already past the 1 s test timeout."""
    return OpenPosition(
        symbol='AAOZ', entry_price=10.03, stop_price=9.50, shares=shares,
        trade_id=1, order_id=ORDER_ID,
        entry_time=datetime.now(timezone.utc) - timedelta(seconds=90),
        range_high=10.0, range_low=9.5,
        lock_arm_at_r=1.5, lock_stop_r=1.0,
        composite_score=0.5, quintile='Q4',
        first_partial_at=datetime.now(timezone.utc) - timedelta(seconds=84),
    )


def _stream_payload(filled: int, status: str = 'partially_filled') -> dict:
    """Exact key set of ``trading/order_stream._order_to_dict`` - no 'qty'."""
    return {
        'id': ORDER_ID, 'client_order_id': 'orb-AAOZ', 'symbol': 'AAOZ',
        'status': status, 'filled_avg_price': 10.04, 'filled_qty': filled,
        'submitted_at': None, 'filled_at': None,
        'updated_at': datetime.now(timezone.utc), 'event': 'partial_fill',
        'reject_reason': None,
    }


def _rest_payload(filled: int, status: str, qty: int = 200) -> dict:
    """Key set of ``AlpacaClient.get_order`` (carries 'qty')."""
    return {
        'id': ORDER_ID, 'status': status, 'symbol': 'AAOZ', 'qty': qty,
        'filled_qty': filled, 'filled_avg_price': 10.04,
    }


def test_stream_payload_has_no_qty_key():
    """Pins the root cause: the stream payload shape carries no 'qty'."""
    assert 'qty' not in _stream_payload(170)
    assert 'qty' in _rest_payload(170, 'partially_filled')


def test_payload_without_qty_cancels_real_remainder(engine, mock_alpaca, mock_sm, caplog):
    """The AAOZ path: stream payload (no qty), 170/200 -> cancel of 30 sh."""
    caplog.set_level(logging.INFO, logger='trading.orb_engine')
    pos = _pos(200)
    engine.open_positions['AAOZ'] = pos
    engine.order_stream = MagicMock(spec=OrderStreamWatcher)
    engine.order_stream.get_status.return_value = _stream_payload(170)
    mock_alpaca.get_order.return_value = _rest_payload(170, 'canceled')

    engine._process_pending_fills()

    mock_alpaca.cancel_order.assert_called_once_with(ORDER_ID)
    text = ' '.join(r.getMessage() for r in caplog.records)
    assert '170/200 sh' in text            # the true requested qty is logged
    assert 'cancelling remaining 30 sh' in text
    assert '/0 sh' not in text
    mock_sm.add_watch.assert_called_once()
    assert pos.shares == 170 and pos.order_id == ''


def test_payload_with_qty_cancels_same_remainder(engine, mock_alpaca, mock_sm):
    """REST payload carrying qty=200: identical behaviour."""
    pos = _pos(200)
    engine.open_positions['AAOZ'] = pos
    mock_alpaca.get_order.side_effect = [
        _rest_payload(170, 'partially_filled'),   # poll
        _rest_payload(170, 'canceled'),           # cancel confirmation
    ]

    engine._process_pending_fills()

    mock_alpaca.cancel_order.assert_called_once_with(ORDER_ID)
    mock_sm.add_watch.assert_called_once()
    assert pos.shares == 170 and pos.order_id == ''


def test_payload_qty_mismatch_warns_and_engine_record_wins(engine, mock_alpaca, caplog):
    """Payload qty != pos.shares: WARNING, and the cancel still uses pos.shares."""
    caplog.set_level(logging.WARNING, logger='trading.orb_engine')
    pos = _pos(200)
    engine.open_positions['AAOZ'] = pos
    mock_alpaca.get_order.side_effect = [
        _rest_payload(170, 'partially_filled', qty=150),
        _rest_payload(170, 'canceled', qty=150),
    ]

    engine._process_pending_fills()

    mock_alpaca.cancel_order.assert_called_once_with(ORDER_ID)
    assert any('qty mismatch' in r.getMessage() and r.levelno == logging.WARNING
               for r in caplog.records)


def test_filled_equals_requested_sends_no_cancel(engine, mock_alpaca, mock_sm):
    """200/200 still reported partially_filled: nothing to cancel."""
    pos = _pos(200)
    engine.open_positions['AAOZ'] = pos
    engine.order_stream = MagicMock(spec=OrderStreamWatcher)
    engine.order_stream.get_status.return_value = _stream_payload(200)
    mock_alpaca.get_order.return_value = _rest_payload(200, 'partially_filled')

    engine._process_pending_fills()

    mock_alpaca.cancel_order.assert_not_called()
    mock_sm.add_watch.assert_called_once()
    assert pos.shares == 200


def test_shares_filled_between_poll_and_cancel_ack_are_confirmed(engine, mock_alpaca, mock_sm):
    """Poll saw 170; the cancel-ack re-fetch shows 200 -> confirmed qty 200."""
    pos = _pos(200)
    engine.open_positions['AAOZ'] = pos
    engine.order_stream = MagicMock(spec=OrderStreamWatcher)
    engine.order_stream.get_status.return_value = _stream_payload(170)
    mock_alpaca.get_order.return_value = _rest_payload(200, 'filled')

    engine._process_pending_fills()

    mock_alpaca.cancel_order.assert_called_once_with(ORDER_ID)
    assert pos.shares == 200
    assert pos.order_id == ''
    mock_sm.add_watch.assert_called_once()


def test_unconfirmed_cancel_is_not_confirmed_until_timeout_then_errors(
        engine, mock_alpaca, mock_sm, caplog):
    """Re-fetch still shows the order live: do NOT confirm yet (the position
    would grow unwatched); after the stall timeout log ERROR and confirm the
    observed qty so the filled shares get a stop."""
    caplog.set_level(logging.ERROR, logger='trading.orb_engine')
    pos = _pos(200)
    engine.open_positions['AAOZ'] = pos
    mock_alpaca.get_order.return_value = _rest_payload(170, 'partially_filled')

    engine._process_pending_fills()          # cancel sent, not confirmed
    mock_alpaca.cancel_order.assert_called_once_with(ORDER_ID)
    mock_sm.add_watch.assert_not_called()
    assert pos.order_id == ORDER_ID
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)

    pos.stall_cancel_sent_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    engine._process_pending_fills()          # past the timeout -> ERROR + confirm
    assert any('cancel NOT confirmed' in r.getMessage() and r.levelno == logging.ERROR
               for r in caplog.records)
    mock_sm.add_watch.assert_called_once()
    assert pos.shares == 170 and pos.order_id == ''
    mock_alpaca.cancel_order.assert_called_once()   # acked once, not re-sent


def test_cancel_exception_is_retried_then_errors(engine, mock_alpaca, mock_sm, caplog):
    """cancel_order raising: ERROR, retried on the next tick, not confirmed
    inside the window."""
    caplog.set_level(logging.ERROR, logger='trading.orb_engine')
    pos = _pos(200)
    engine.open_positions['AAOZ'] = pos
    mock_alpaca.get_order.return_value = _rest_payload(170, 'partially_filled')
    mock_alpaca.cancel_order.side_effect = Exception('broker race')

    engine._process_pending_fills()
    engine._process_pending_fills()

    assert mock_alpaca.cancel_order.call_count == 2
    assert any('stall-cancel FAILED' in r.getMessage() for r in caplog.records)
    mock_sm.add_watch.assert_not_called()
