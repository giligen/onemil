"""2026-11-16 CLS: bracket SL leg executing at the broker while the exit path fights it.

Covers S1 (wait for an executing leg), S2 (re-check leg/position before ERROR + emergency stop),
S3 (no emergency stop on a flat position; cancel a dangling one).
"""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock

from trading.stop_monitor import StopMonitor, WatchEntry

HELD = Exception('insufficient qty available, code=40310000, held_for_orders=12')


def _watch():
    return WatchEntry(symbol='CLS', stop_price=237.89, shares=12, tp_leg_id='tp-1',
                      sl_leg_id='leg-1', trade_db_id=1, entry_price=240.0)


def _monitor():
    m = StopMonitor.__new__(StopMonitor)
    m.notifier = MagicMock()
    m._HELD_QTY_RETRY_BACKOFFS_S = (0.001, 0.001)
    m._LEG_EXECUTING_POLL_S = 0.005
    m._LEG_EXECUTING_POLL_INTERVAL_S = 0.001
    m._EMERGENCY_RECHECK_DELAY_S = 0.001
    return m


def _client(positions):
    c = MagicMock()
    c.get_open_positions = MagicMock(return_value=positions)
    c.submit_stop_sell_order = MagicMock(return_value={'id': 'em-1'})
    c.cancel_order = MagicMock(return_value=False)
    return c


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _emergency(m, c):
    async def go():
        return await m._place_emergency_stop_fallback(
            asyncio.get_event_loop(), c, 'CLS', 237.61, watch=_watch(), reason='x')
    return _run(go())


def _escalate(m, c):
    return _run(m._escalate_to_market_close(
        c, 'CLS', 'leg-1', 237.61, sl_leg_id='leg-1', watch=_watch()))


def test_not_cancelable_leg_fills_on_third_poll():
    m = _monitor()
    m._LEG_EXECUTING_POLL_S = 0.1
    c = _client([{'symbol': 'CLS', 'qty': '12'}])
    answers = iter([{'status': 'new'}, {'status': 'new'}, {'status': 'new'},
                    {'status': 'filled', 'filled_avg_price': '237.19'}])
    c.get_order = MagicMock(side_effect=lambda oid: next(answers))
    price, oid, branch = _escalate(m, c)
    assert (price, oid, branch) == (237.19, 'leg-1', StopMonitor.BRANCH_LIMIT_RACE)
    c.close_position.assert_not_called()
    c.submit_stop_sell_order.assert_not_called()


def test_leg_confirms_only_at_recheck_no_error_no_emergency(caplog):
    m = _monitor()
    c = _client([{'symbol': 'CLS', 'qty': '12'}])
    c.close_position = MagicMock(side_effect=HELD)
    calls = {'n': 0}

    def get_order(oid):
        calls['n'] += 1
        # not filled inside the poll budget + race check; filled afterwards
        return {'status': 'new'} if calls["n"] <= 4 else {'status': 'filled', 'filled_avg_price': 237.0}
    c.get_order = MagicMock(side_effect=get_order)
    # make the poll budget cheap: count-based, so force the re-check to be the first filled answer
    m._LEG_EXECUTING_POLL_S = 0.003
    with caplog.at_level(logging.WARNING):
        price, oid, branch = _escalate(m, c)
    assert branch == StopMonitor.BRANCH_SL_LEG_RACE
    assert price == 237.0
    c.submit_stop_sell_order.assert_not_called()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_emergency_fallback_flat_position_no_submit_no_notify():
    m = _monitor()
    c = _client([])
    res = _emergency(m, c)
    assert res is None
    c.submit_stop_sell_order.assert_not_called()
    m.notifier.notify_error.assert_not_called()


def test_emergency_placed_then_flat_is_cancelled(caplog):
    m = _monitor()
    c = _client([])
    states = iter([[{'symbol': 'CLS', 'qty': '12'}], []])
    c.get_open_positions = MagicMock(side_effect=lambda: next(states))
    c.cancel_order = MagicMock(return_value=True)
    with caplog.at_level(logging.WARNING):
        res = _emergency(m, c)
    assert res is None
    c.cancel_order.assert_called_once_with('em-1')
    assert 'cancelled dangling emergency stop' in caplog.text


def test_position_present_close_failing_keeps_error_and_emergency(caplog):
    m = _monitor()
    c = _client([{'symbol': 'CLS', 'qty': '12'}])
    c.close_position = MagicMock(side_effect=HELD)
    c.get_order = MagicMock(return_value={'status': 'new'})
    with caplog.at_level(logging.ERROR):
        price, oid, branch = _escalate(m, c)
    assert branch == StopMonitor.BRANCH_LAST_RESORT
    c.submit_stop_sell_order.assert_called_once()
    assert 'market close failed after retry' in caplog.text


def test_position_query_error_degrades_to_placing_stop(caplog):
    m = _monitor()
    c = _client([])
    c.get_open_positions = MagicMock(side_effect=RuntimeError('boom'))
    with caplog.at_level(logging.WARNING):
        res = _emergency(m, c)
    assert res == 'em-1'
    assert 'position query failed' in caplog.text
