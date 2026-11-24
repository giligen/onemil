"""RAMZ 2026-11-24: Alpaca refuses a qty change on a bracket leg (42210000), so the add-on places a
dedicated plain stop-sell for the added shares instead of calling replace_order_qty."""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading import exit_coverage, orb_add_on
from trading.orb_engine import ORBEngine, OpenPosition
from trading.stop_monitor import StopMonitor, WatchEntry
from tests.test_orb_add_on import orb_cfg, mock_db, _make_pos  # noqa: F401  (fixtures)

SL, TP = 'f25ac28e-0000', 'e388daee-0000'


def _build(orb_cfg, mock_db, *, watch_stop=22.13, sl_class='bracket', add_stop_leg_ids=None,
           real_monitor=False):
    """Real engine + OpenPosition (216 sh RAMZ bracket); Alpaca mocked with spec=."""
    alpaca = MagicMock(spec=AlpacaClient)
    legs = {SL: {'id': SL, 'qty': 216, 'order_class': sl_class, 'stop_price': 20.76, 'status': 'held'},
            TP: {'id': TP, 'qty': 216, 'order_class': 'bracket', 'limit_price': 69.21, 'status': 'new'}}
    alpaca.get_order.side_effect = lambda oid: legs[oid]
    alpaca.get_open_positions.return_value = [{'symbol': 'RAMZ', 'qty': 432}]
    alpaca.submit_stop_sell_order.return_value = {'id': 'add-stop-1234', 'status': 'accepted'}
    alpaca.replace_order_qty.return_value = {'id': 'replaced-1'}
    if real_monitor:
        sm = StopMonitor(api_key='k', api_secret='s', alpaca_client=alpaca)
    else:
        sm = StopMonitor.__new__(StopMonitor)
        sm._watch_lock = threading.Lock()
        sm._watches = {}
    sm._watches['RAMZ'] = WatchEntry(symbol='RAMZ', stop_price=watch_stop, shares=216,
                                     tp_leg_id=TP, sl_leg_id=SL)
    orb_cfg['strategy']['enabled'] = True
    eng = ORBEngine(alpaca_client=alpaca, db=mock_db, stop_monitor=sm, config=orb_cfg)
    pos = _make_pos(symbol='RAMZ', shares=216, tp_leg_id=TP, sl_leg_id=SL,
                    add_stop_leg_ids=list(add_stop_leg_ids or []))
    return eng, alpaca, sm, pos


def test_ramz_replay_places_add_stop_no_replace_no_error(orb_cfg, mock_db, caplog):
    eng, alpaca, sm, pos = _build(orb_cfg, mock_db)
    with caplog.at_level(logging.INFO):
        assert eng._resize_exit_legs_for_add(pos, 432) == 432
    alpaca.replace_order_qty.assert_not_called()
    alpaca.submit_stop_sell_order.assert_called_once_with('RAMZ', 216, 22.13)
    assert pos.add_stop_leg_ids == ['add-stop-1234']
    assert pos.sl_leg_id == SL and pos.tp_leg_id == TP
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    text = caplog.text
    assert 'bracket SL leg f25ac28e keeps 216sh' in text and 'add stop add-stop placed for 216sh @ $22.13' in text
    assert 'bracket TP leg e388daee keeps 216sh' in text
    assert sm._watches['RAMZ'].shares == 432


def test_no_watch_stop_uses_leg_stop(orb_cfg, mock_db):
    eng, alpaca, sm, pos = _build(orb_cfg, mock_db, watch_stop=0.0)
    eng._resize_exit_legs_for_add(pos, 432)
    alpaca.submit_stop_sell_order.assert_called_once_with('RAMZ', 216, 20.76)
    sm._watches.clear()
    pos.add_stop_leg_ids = []
    alpaca.submit_stop_sell_order.reset_mock()
    eng._resize_exit_legs_for_add(pos, 432)
    alpaca.submit_stop_sell_order.assert_called_once_with('RAMZ', 216, 20.76)


def test_stop_submit_failure_one_error_and_watch_tracks_real_qty(orb_cfg, mock_db, caplog):
    eng, alpaca, sm, pos = _build(orb_cfg, mock_db)
    alpaca.submit_stop_sell_order.side_effect = RuntimeError('boom')
    with caplog.at_level(logging.INFO):
        assert eng._resize_exit_legs_for_add(pos, 432) == 432
    errs = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errs) == 1 and 'NAKED until exit coverage' in errs[0].getMessage()
    assert pos.add_stop_leg_ids == []
    assert sm._watches['RAMZ'].shares == 432


def test_simple_leg_still_replaced(orb_cfg, mock_db):
    eng, alpaca, sm, pos = _build(orb_cfg, mock_db, sl_class='simple')
    eng._resize_exit_legs_for_add(pos, 432)
    assert (SL, 432) in [c.args for c in alpaca.replace_order_qty.call_args_list]
    assert pos.sl_leg_id == 'replaced-1'
    alpaca.submit_stop_sell_order.assert_not_called()


def test_leg_already_at_applied_skips(orb_cfg, mock_db, caplog):
    eng, alpaca, sm, pos = _build(orb_cfg, mock_db)
    alpaca.get_order.side_effect = lambda oid: {'id': oid, 'qty': 432, 'order_class': 'bracket',
                                                'stop_price': 20.76}
    with caplog.at_level(logging.INFO):
        eng._resize_exit_legs_for_add(pos, 432)
    alpaca.replace_order_qty.assert_not_called()
    alpaca.submit_stop_sell_order.assert_not_called()
    assert 'already covers 432sh' in caplog.text


def test_add_be_shrink_cancels_add_stops(orb_cfg, mock_db, caplog):
    eng, alpaca, sm, pos = _build(orb_cfg, mock_db, add_stop_leg_ids=['x'])
    pos.shares = 432
    alpaca.get_open_positions.return_value = [{'symbol': 'RAMZ', 'qty': 216}]
    with caplog.at_level(logging.INFO):
        eng._resize_exit_legs_for_add(pos, 216)
    alpaca.cancel_order.assert_called_once_with('x')
    assert pos.add_stop_leg_ids == []
    alpaca.replace_order_qty.assert_not_called()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_add_stop_ids_persist_and_rehydrate(tmp_path, orb_cfg):
    db = Database(db_path=str(tmp_path / 'c.db'), cache_path=str(tmp_path / 'c.db'),
                  trades_path=str(tmp_path / 't.db'))
    tid = db.save_trade({
        'trade_date': '2026-11-24', 'symbol': 'RAMZ', 'side': 'buy', 'entry_price': 23.07,
        'stop_loss_price': 20.76, 'take_profit_price': 69.21, 'shares': 216, 'risk_per_share': 2.3,
        'total_risk': 500.0, 'risk_reward_ratio': 2.0, 'order_id': '', 'order_status': 'filled',
        'fill_price': 23.07, 'filled_at': datetime.now(timezone.utc), 'exit_price': None,
        'exit_reason': None, 'exited_at': None, 'pnl': None, 'pnl_pct': None,
        'pattern_data': json.dumps({'pool_id': 'production'}), 'strategy': 'orb', 'account': 'paper'})
    orb_cfg['strategy']['enabled'] = True
    eng = ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient), db=db,
                    stop_monitor=MagicMock(spec=StopMonitor), config=orb_cfg)
    pos = _make_pos(symbol='RAMZ', trade_id=tid, shares=432, add_stop_leg_ids=['a1', 'a2'],
                    adds=[{'at_r': 1.0, 'px': 24.0, 'qty': 216, 'ts': 'x'}])
    eng._persist_add_on_state(pos)
    blob = json.loads(db.get_trade_by_id(tid)['pattern_data'])
    assert blob['pool_id'] == 'production'
    assert blob['add_on']['add_stop_leg_ids'] == ['a1', 'a2']
    from trading.orb_engine import _add_stop_ids_from_pdata
    assert _add_stop_ids_from_pdata(blob) == ['a1', 'a2']
    assert _add_stop_ids_from_pdata({}) == []


def test_exit_coverage_sees_no_naked_after_resize(orb_cfg, mock_db):
    eng, alpaca, sm, pos = _build(orb_cfg, mock_db, real_monitor=True)
    eng._resize_exit_legs_for_add(pos, 432)
    sells = [
        {'symbol': 'RAMZ', 'side': 'sell', 'status': 'held', 'order_class': 'bracket', 'qty': 216, 'filled_qty': 0},
        {'symbol': 'RAMZ', 'side': 'sell', 'status': 'new', 'order_class': 'bracket', 'qty': 216, 'filled_qty': 0},
        {'symbol': 'RAMZ', 'side': 'sell', 'status': 'accepted', 'order_class': 'simple', 'qty': 216, 'filled_qty': 0},
    ]
    snap = exit_coverage.BrokerSnapshot(positions={'RAMZ': 432.0}, orders=sells)
    res = exit_coverage.check(alpaca, 'RAMZ', 432, snap=snap)
    assert res.naked_qty == 0
