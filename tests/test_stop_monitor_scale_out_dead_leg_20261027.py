"""LABX 2026-10-27: a scale-out whose safety leg is dead at the broker (cancelled with its
parent bracket) retried replace_order_qty on every quote tick (1,316 ERRORs). The dead leg id
is dropped once; any other resize failure is disarmed after 3 attempts."""
import asyncio
import logging
from unittest.mock import MagicMock

import pytest

from data_sources.alpaca_client import AlpacaAPIError, AlpacaClient
from trading.stop_monitor import StopMonitor

DEAD = AlpacaAPIError('Failed to replace order qty sl-1: {"code":42210000,"message":"order is not open"}')


@pytest.fixture
def mock_alpaca():
    c = MagicMock(spec=AlpacaClient)
    c.cancel_order.return_value = True
    c.replace_order_qty.side_effect = lambda oid, q: {'id': f'new-{oid}', 'status': 'accepted'}
    c.submit_limit_sell_order.return_value = {'id': 'scale-1', 'status': 'accepted'}
    c.get_order.return_value = {'id': 'scale-1', 'status': 'new', 'filled_qty': 0, 'filled_avg_price': None}
    c.get_open_positions.return_value = [{'symbol': 'TEST', 'qty': 1000}]
    return c


@pytest.fixture
def monitor(mock_alpaca):
    mon = StopMonitor(api_key='k', api_secret='s', alpaca_client=mock_alpaca,
                      marketable_limit_offset=0.03, marketable_limit_offset_pct=0.005)
    mon._STOP_EXIT_FILL_TIMEOUT_S = 0.2
    mon._STOP_EXIT_POLL_INTERVAL_S = 0.05
    return mon


def _watch(mon):
    mon.add_watch(symbol='TEST', stop_price=9.5, shares=1000, tp_leg_id='tp-1', sl_leg_id='sl-1',
                  trade_db_id=42, entry_price=10.0, risk_per_share=0.5, strategy='orb',
                  lock_arm_at_r=1.75, lock_stop_r=0.5, lock_r_unit=1.0)
    w = mon._watches['TEST']
    mon.arm_scale_out('TEST', 13.0, 400)
    return w


def _tick(mon, price):
    t = MagicMock()
    t.symbol, t.price = 'TEST', price
    asyncio.run(mon._on_trade(t))


def test_dead_sl_leg_warns_once_clears_id_and_places_scale(monitor, mock_alpaca, caplog):
    caplog.set_level(logging.INFO, logger='trading.stop_monitor')
    w = _watch(monitor)
    mock_alpaca.replace_order_qty.side_effect = (
        lambda oid, q: (_ for _ in ()).throw(DEAD) if oid == 'sl-1' else {'id': f'new-{oid}', 'status': 'accepted'})
    _tick(monitor, 13.05)
    assert w.sl_leg_id == ''
    mock_alpaca.submit_limit_sell_order.assert_called_once()
    assert mock_alpaca.submit_limit_sell_order.call_args.kwargs['qty'] == 400
    dead = [r for r in caplog.records if 'is not open' in r.getMessage()]
    assert len(dead) == 1 and dead[0].levelno == logging.WARNING
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_second_tick_does_not_replace_dead_leg_again(monitor, mock_alpaca):
    w = _watch(monitor)
    calls = []

    def repl(oid, q):
        calls.append(oid)
        if oid == 'sl-1':
            raise DEAD
        return {'id': f'new-{oid}', 'status': 'accepted'}
    mock_alpaca.replace_order_qty.side_effect = repl
    mock_alpaca.submit_limit_sell_order.side_effect = RuntimeError('boom')  # first cycle fails after the resize
    _tick(monitor, 13.05)
    n_sl = calls.count('sl-1')
    _tick(monitor, 13.10)
    assert n_sl == 1 and calls.count('sl-1') == 1
    assert w.sl_leg_id == ''


def test_generic_failure_x3_disarms_with_one_error(monitor, mock_alpaca, caplog):
    caplog.set_level(logging.INFO, logger='trading.stop_monitor')
    w = _watch(monitor)
    mock_alpaca.replace_order_qty.side_effect = AlpacaAPIError('503 upstream timeout')
    for px in (13.05, 13.06, 13.07, 13.08, 13.09):
        _tick(monitor, px)
    assert mock_alpaca.replace_order_qty.call_count == 3
    assert w.scale_qty == 0 and w.scale_fail_count == 3
    errs = [r for r in caplog.records if r.levelno >= logging.ERROR and 'DISARMED' in r.getMessage()]
    assert len(errs) == 1 and 'TEST' in errs[0].getMessage()
    mock_alpaca.submit_limit_sell_order.assert_not_called()


def test_replace_order_qty_failure_logs_warning_not_error(caplog):
    client = AlpacaClient.__new__(AlpacaClient)
    client.trading_client = MagicMock()
    client.trading_client.replace_order_by_id.side_effect = RuntimeError('order is not open')
    client._call_with_timeout = lambda fn, desc: fn()
    caplog.set_level(logging.DEBUG, logger='data_sources.alpaca_client')
    with pytest.raises(AlpacaAPIError):
        client.replace_order_qty('abc', 5)
    with pytest.raises(AlpacaAPIError):
        client.replace_order_stop_price('abc', 5.0)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2
