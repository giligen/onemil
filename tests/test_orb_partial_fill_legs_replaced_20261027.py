"""LABX 2026-10-27: the partial-fill stall cancel cancelled the whole bracket with the parent and
the engine kept the dead leg ids. Fix: read the legs, rest ONE OCO for the filled qty, register the
StopMonitor watch with the NEW ids (empty ids + one ERROR if the OCO fails)."""
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, OpenPosition
from trading.stop_monitor import StopMonitor

PARENT = 'bdb74ee6-parent'


@pytest.fixture
def orb_cfg():
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    cfg.setdefault('fill_handling', {})['partial_fill_stall_seconds_max'] = 1
    return cfg


@pytest.fixture
def alpaca():
    c = MagicMock(spec=AlpacaClient)
    c.get_open_positions.return_value = []
    c.get_account_info.return_value = {'buying_power': 100_000.0}
    c.cancel_order.return_value = True
    c.get_daily_bars.return_value = {}
    return c


@pytest.fixture
def db():
    d = MagicMock(spec=Database)
    d.save_trade.return_value = 1
    d.update_trade.return_value = True
    d.get_trade_by_id.return_value = {'pattern_data': '{}'}
    return d


@pytest.fixture
def sm():
    m = MagicMock(spec=StopMonitor)
    m.polling_mode = False
    m.drain_exit_events.return_value = []
    return m


@pytest.fixture
def engine(orb_cfg, alpaca, db, sm):
    return ORBEngine(alpaca_client=alpaca, db=db, stop_monitor=sm, config=orb_cfg)


def _pos():
    return OpenPosition(
        symbol='LABX', entry_price=8.03, stop_price=7.76, shares=622, trade_id=1, order_id=PARENT,
        entry_time=datetime.now(timezone.utc) - timedelta(seconds=90),
        range_high=8.01, range_low=7.76, lock_arm_at_r=1.5, lock_stop_r=1.0,
        composite_score=0.5, quintile='Q4', tp_leg_id='tp-old', sl_leg_id='sl-old',
        first_partial_at=datetime.now(timezone.utc) - timedelta(seconds=84))


def _parent(status):
    return {'id': PARENT, 'status': status, 'symbol': 'LABX', 'qty': 622, 'filled_qty': 423,
            'filled_avg_price': 8.03,
            'legs': [{'id': 'tp-old', 'side': 'sell', 'type': 'limit', 'limit_price': 24.09, 'stop_price': None},
                     {'id': 'sl-old', 'side': 'sell', 'type': 'stop', 'limit_price': None, 'stop_price': 7.23}]}


def _get_order(leg_status):
    def f(oid):
        return _parent('canceled') if oid == PARENT else {'id': oid, 'status': leg_status}
    return f


def test_dead_legs_replaced_by_one_oco_and_watch_gets_new_ids(engine, alpaca, sm, db, caplog):
    caplog.set_level(logging.INFO, logger='trading.orb_engine')
    pos = _pos()
    engine.open_positions['LABX'] = pos
    alpaca.get_order.side_effect = _get_order('canceled')
    alpaca.submit_oco_sell_order.return_value = {'id': 'tp-new-id', 'legs': [{'id': 'sl-new-id', 'type': 'stop'}]}

    engine._handle_partial_fill_stall('LABX', pos, _parent('partially_filled'), 423, 90.0)

    alpaca.submit_oco_sell_order.assert_called_once_with(
        symbol='LABX', qty=423, limit_price=24.09, stop_price=round(pos.stop_price, 2))
    assert (pos.sl_leg_id, pos.tp_leg_id) == ('sl-new-id', 'tp-new-id')
    kw = sm.add_watch.call_args.kwargs
    assert kw['sl_leg_id'] == 'sl-new-id' and kw['tp_leg_id'] == 'tp-new-id' and kw['shares'] == 423
    assert any('replaced with OCO sl=sl-new-i tp=tp-new-i for 423 sh' in r.getMessage() for r in caplog.records)
    persisted = [c.args[1] for c in db.update_trade.call_args_list if 'replacement_oco' in str(c.args[1].get('pattern_data', ''))]
    assert len(persisted) == 1
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_live_legs_are_left_alone(engine, alpaca, sm):
    pos = _pos()
    engine.open_positions['LABX'] = pos
    alpaca.get_order.side_effect = _get_order('held')
    engine._handle_partial_fill_stall('LABX', pos, _parent('partially_filled'), 423, 90.0)
    alpaca.submit_oco_sell_order.assert_not_called()
    assert sm.add_watch.call_args.kwargs['sl_leg_id'] == 'sl-old'


def test_oco_failure_logs_one_error_and_registers_empty_ids(engine, alpaca, sm, caplog):
    caplog.set_level(logging.INFO, logger='trading.orb_engine')
    pos = _pos()
    engine.open_positions['LABX'] = pos
    alpaca.get_order.side_effect = _get_order('canceled')
    alpaca.submit_oco_sell_order.side_effect = RuntimeError('insufficient qty')

    engine._handle_partial_fill_stall('LABX', pos, _parent('partially_filled'), 423, 90.0)

    errs = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errs) == 1 and 'NAKED' in errs[0].getMessage()
    kw = sm.add_watch.call_args.kwargs
    assert kw['sl_leg_id'] == '' and kw['tp_leg_id'] == ''
