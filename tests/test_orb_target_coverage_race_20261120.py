"""SPCG 2026-11-20 (docs/orb_target_coverage_race_20261120.md): our own rested target fill was booked
as `external` by exit coverage, then the late target poll raised a false ERROR."""
import logging
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from persistence.database import Database
from trading.exit_reasons import ExitReason
from trading.order_stream import OrderStreamWatcher
from trading.orb_engine import ORBEngine
from trading.stop_monitor import StopMonitor
from tests.test_orb_add_on import _make_pos, orb_cfg  # noqa: F401  (fixture)
from tests.test_orb_scaleout_coverage_conflict_20261113 import client  # noqa: F401  (fixture)
from tests.test_orb_target_limit import (  # noqa: F401  (fixtures)
    _seed_position, engine, mock_alpaca, mock_db, mock_stop_monitor,
)
import tests.test_orb_target_limit as _tl

FILL_PX = 16.910983


def _mon(client, resting=True):
    """A StopMonitor holding the SPCG watch, target rested like the incident."""
    m = StopMonitor(api_key='k', api_secret='s', alpaca_client=client)
    m.add_watch(symbol='SPCG', stop_price=16.42, shares=295, tp_leg_id='b46723bf', sl_leg_id='sl-1',
                trade_db_id=848, entry_price=16.8837, risk_per_share=0.46, strategy='orb')
    if resting:
        m.mark_target_resting('SPCG', 'a682c8f9', 16.89)
    m._watches['SPCG'].added_at = time.time() - 600
    return m


def _flat(client, fill_id='a682c8f9'):
    client.get_open_positions.return_value = []
    client.get_last_filled_sell.return_value = {
        'id': fill_id, 'qty': 295.0, 'price': FILL_PX, 'filled_at': datetime.now(timezone.utc)}


def test_spcg_replay_books_target_rested(client, caplog):
    m = _mon(client)
    _flat(client)
    with caplog.at_level(logging.INFO):
        assert m._book_external_flat(client, 'SPCG', m._watches['SPCG']) is True
    ev = m.drain_exit_events('orb')[0]
    assert ev.exit_reason == ExitReason.TARGET_RESTED.value
    assert ev.pricing_method == 'target_rested_limit'
    assert ev.exit_limit_price == 16.89 and ev.shares == 295 and ev.confirmed is True
    assert ev.exit_price == pytest.approx(FILL_PX) and ev.trade_db_id == 848
    assert any('(exit coverage)' in r.message for r in caplog.records)
    assert not any('closed externally' in r.message for r in caplog.records)
    assert 'SPCG' not in m._watches


def test_other_sell_id_stays_external(client):
    m = _mon(client)
    _flat(client, fill_id='zzz')
    assert m._book_external_flat(client, 'SPCG', m._watches['SPCG']) is True
    assert m.drain_exit_events('orb')[0].exit_reason == ExitReason.EXTERNAL.value


def test_target_not_resting_stays_external(client):
    m = _mon(client, resting=False)
    _flat(client)
    assert m._book_external_flat(client, 'SPCG', m._watches['SPCG']) is True
    assert m.drain_exit_events('orb')[0].exit_reason == ExitReason.EXTERNAL.value


def test_late_poll_after_coverage_is_not_an_error(client, caplog):
    m = _mon(client)
    _flat(client)
    m._book_external_flat(client, 'SPCG', m._watches['SPCG'])
    with caplog.at_level(logging.INFO):
        assert m.book_target_rested_fill('SPCG', 295, FILL_PX, 16.89, datetime.now(timezone.utc)) is True
    assert any('already booked' in r.message and r.levelno == logging.INFO for r in caplog.records)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_no_record_or_stop_record_warns_and_returns_false(client, caplog):
    m = StopMonitor(api_key='k', api_secret='s', alpaca_client=client)
    with caplog.at_level(logging.WARNING):
        assert m.book_target_rested_fill('SPCG', 295, FILL_PX, 16.89, None) is False
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    caplog.clear()
    m._booked_exits['SPCG'] = (ExitReason.STOP_LOSS.value, 295, 16.4, time.time())
    with caplog.at_level(logging.WARNING):
        assert m.book_target_rested_fill('SPCG', 295, FILL_PX, 16.89, None) is False
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_stale_record_warns_and_returns_false(client, caplog):
    m = StopMonitor(api_key='k', api_secret='s', alpaca_client=client)
    m._booked_exits['SPCG'] = (ExitReason.TARGET_RESTED.value, 295, FILL_PX, time.time() - 601)
    with caplog.at_level(logging.WARNING):
        assert m.book_target_rested_fill('SPCG', 295, FILL_PX, 16.89, None) is False
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def _poll_engine(engine):
    pos = _seed_position(engine)
    pos.target_resting = True
    pos.target_price = 9.10
    engine.order_stream = MagicMock(spec=OrderStreamWatcher)
    engine.order_stream.get_status.return_value = {
        'status': 'filled', 'filled_qty': 200, 'filled_avg_price': 9.11}
    return pos


def test_engine_poll_booked_true_no_error(engine, mock_stop_monitor, caplog):
    pos = _poll_engine(engine)
    mock_stop_monitor.book_target_rested_fill.return_value = True
    with caplog.at_level(logging.ERROR):
        engine._poll_target_fills()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert pos.target_last_booked_qty == 200


def test_engine_poll_booked_false_one_error_with_new_text(engine, mock_stop_monitor, caplog):
    _poll_engine(engine)
    mock_stop_monitor.book_target_rested_fill.return_value = False
    with caplog.at_level(logging.ERROR):
        engine._poll_target_fills()
    errs = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errs) == 1 and 'sync_positions runs at boot only' in errs[0].message


def test_integration_coverage_target_drains_into_db_row(client, orb_cfg, mock_stop_monitor):  # noqa: F811
    """Real StopMonitor + real ORBEngine drain: the row closes as target_rested at the broker price."""
    m = _mon(client)
    _flat(client)
    m._book_external_flat(client, 'SPCG', m._watches['SPCG'])
    ev = m.drain_exit_events('orb')[0]
    db = MagicMock(spec=Database)
    orb_cfg['strategy']['enabled'] = True
    eng = ORBEngine(alpaca_client=MagicMock(spec=_tl.AlpacaClient), db=db,
                    stop_monitor=mock_stop_monitor, config=orb_cfg)
    eng.open_positions['SPCG'] = _make_pos(symbol='SPCG', shares=295, trade_id=848,
                                           entry_price=16.8837, combined_avg_price=16.8837)
    eng._handle_exit_event(ev)
    upd = [c.args[1] for c in db.update_trade.call_args_list if c.args[0] == 848][-1]
    assert upd['exit_reason'] == 'target_rested' and upd['exit_price'] == pytest.approx(FILL_PX)


def test_watch_claimed_once_when_both_paths_race(client, caplog):
    """Review 2026-11-20: coverage and the order-stream poll both hold the SPCG watch; whichever
    pops it books the exit, the other queues NOTHING (one fill must never produce two events)."""
    m = _mon(client)
    watch = m._watches['SPCG']
    _flat(client)
    assert m._book_external_flat(client, 'SPCG', watch) is True
    with caplog.at_level(logging.WARNING):
        assert m._queue_target_rested_event('SPCG', watch, 295, FILL_PX, 16.89, '') is False
    assert len(m.drain_exit_events('orb')) == 1
    assert any('already retired while booking the target fill' in r.message for r in caplog.records)
    # the late poll itself still reads the booking and stays quiet
    assert m.book_target_rested_fill('SPCG', 295, FILL_PX, 16.89, time.time()) is True
