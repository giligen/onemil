"""Exit-coverage invariant (docs/exit_coverage_invariant_spec_20261029.md): LABX 10/27 shape —
423 shares with no resting broker exit. Real StopMonitor + spec'd AlpacaClient doubles."""
import asyncio
import logging
import time
from unittest.mock import MagicMock

import pytest

from data_sources.alpaca_client import AlpacaAPIError, AlpacaClient
from trading import exit_coverage
from trading.stop_monitor import StopMonitor


def _sell(oid, qty, status='held', cls='oco', filled=0, symbol='LABX'):
    return {'id': oid, 'symbol': symbol, 'side': 'sell', 'status': status, 'qty': qty,
            'filled_qty': filled, 'order_class': cls}


@pytest.fixture
def client():
    c = MagicMock(spec=AlpacaClient)
    c.get_open_positions.return_value = [{'symbol': 'LABX', 'qty': 423, 'side': 'long'}]
    c.get_open_orders.return_value = []
    c.submit_stop_sell_order.return_value = {'id': 'stopsell-12345678', 'status': 'accepted'}
    c.get_order.return_value = {'id': 'sl-old', 'status': 'canceled'}
    return c


@pytest.fixture
def mon(client):
    m = StopMonitor(api_key='k', api_secret='s', alpaca_client=client)
    m.add_watch(symbol='LABX', stop_price=9.50, shares=423, tp_leg_id='tp-old', sl_leg_id='sl-old',
                entry_price=10.0, risk_per_share=0.5, strategy='orb',
                lock_arm_at_r=1.75, lock_stop_r=0.5, lock_r_unit=1.0)
    m._watches['LABX'].added_at = time.time() - 600   # old enough
    return m


def test_covered_no_action(mon, client):
    client.get_open_orders.return_value = [_sell('a', 423, cls='simple', status='new')]
    assert mon._exit_coverage_cycle() == {'checked': 1, 'naked': 0, 'repaired': 0}
    client.submit_stop_sell_order.assert_not_called()


def test_dead_sl_leg_live_tp_full_qty_is_covered(mon, client):
    # SL leg cancelled (not in the open list), TP leg still open for the full qty
    client.get_open_orders.return_value = [_sell('tp', 423, cls='oco', status='new')]
    mon._exit_coverage_cycle()
    client.submit_stop_sell_order.assert_not_called()


def test_no_open_sells_places_one_stop_at_stop_price_and_sets_leg(mon, client, caplog):
    caplog.set_level(logging.DEBUG)
    counts = mon._exit_coverage_cycle()
    assert counts == {'checked': 1, 'naked': 1, 'repaired': 1}
    client.submit_stop_sell_order.assert_called_once_with(symbol='LABX', qty=423, stop_price=9.50)
    assert mon._watches['LABX'].sl_leg_id == 'stopsell-12345678'
    errs = [r for r in caplog.records if r.levelno == logging.ERROR and 'EXIT COVERAGE' in r.message]
    assert len(errs) == 1 and 'NAKED 423/423' in errs[0].message and 'stopsell' in errs[0].message


def test_lock_stop_used_when_armed(mon, client):
    mon._watches['LABX'].stop_price = 10.5      # StopMonitor writes the lock stop into stop_price
    mon._watches['LABX'].lock_armed = True
    mon._exit_coverage_cycle()
    assert client.submit_stop_sell_order.call_args.kwargs['stop_price'] == 10.5


def test_partial_coverage_labx_shape_places_223(mon, client):
    # OCO pair (SL leg + TP leg) for 200 of 423: legs are alternatives, counted once
    client.get_open_orders.return_value = [_sell('sl', 200), _sell('tp', 200, status='new')]
    mon._exit_coverage_cycle()
    client.submit_stop_sell_order.assert_called_once_with(symbol='LABX', qty=223, stop_price=9.50)


def test_live_sl_leg_id_not_replaced(mon, client):
    client.get_open_orders.return_value = [_sell('x', 100, cls='simple', status='new')]
    client.get_order.return_value = {'id': 'sl-old', 'status': 'new'}
    mon._exit_coverage_cycle()
    assert mon._watches['LABX'].sl_leg_id == 'sl-old'


def test_placement_failure_critical_no_retry_in_cycle_then_retry_after_cooldown(mon, client, caplog):
    client.submit_stop_sell_order.side_effect = AlpacaAPIError('boom')
    mon._exit_coverage_cycle()
    assert client.submit_stop_sell_order.call_count == 1
    crit = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert len(crit) == 1 and 'repair failed: boom' in crit[0].message and 'VERIFY ON ALPACA' in crit[0].message
    mon._exit_coverage_cycle()                      # inside the 120 s cooldown: still naked, no retry
    assert client.submit_stop_sell_order.call_count == 1
    mon._coverage_last_repair['LABX'] -= 121        # cooldown elapsed
    mon._exit_coverage_cycle()
    assert client.submit_stop_sell_order.call_count == 2


def test_broker_qty_below_watch_one_warning_nothing_placed(mon, client, caplog):
    client.get_open_positions.return_value = [{'symbol': 'LABX', 'qty': 100, 'side': 'long'}]
    mon._exit_coverage_cycle()
    mon._exit_coverage_cycle()
    client.submit_stop_sell_order.assert_not_called()
    warns = [r for r in caplog.records if 'broker qty 100 < watch qty 423' in r.message]
    assert len(warns) == 1


def test_broker_flat_not_this_specs_business(mon, client):
    client.get_open_positions.return_value = []
    mon._exit_coverage_cycle()
    client.submit_stop_sell_order.assert_not_called()


def test_excess_broker_qty_never_covered_one_warning(mon, client, caplog):
    """Owner's manual shares on the shared account: repair is capped at the watch qty."""
    mon._watches['LABX'].shares = 100
    client.get_open_positions.return_value = [{'symbol': 'LABX', 'qty': 150, 'side': 'long'}]
    mon._exit_coverage_cycle()
    client.submit_stop_sell_order.assert_called_once_with(symbol='LABX', qty=100, stop_price=9.50)
    mon._coverage_last_repair.clear()
    mon._exit_coverage_cycle()
    warns = [r for r in caplog.records if r.levelno == logging.WARNING and 'excess 50 sh not ours, not touched' in r.message]
    assert len(warns) == 1
    assert 'broker holds 150 sh, watch claims 100' in warns[0].message


def test_excess_with_full_cover_places_nothing(mon, client):
    client.get_open_positions.return_value = [{'symbol': 'LABX', 'qty': 500, 'side': 'long'}]
    client.get_open_orders.return_value = [_sell('a', 423, cls='simple', status='new')]
    mon._exit_coverage_cycle()
    client.submit_stop_sell_order.assert_not_called()


def test_young_watch_skipped(mon, client):
    mon._watches['LABX'].added_at = time.time() - 5
    assert mon._exit_coverage_cycle()['checked'] == 0
    client.get_open_positions.assert_not_called()


def test_exit_in_progress_skipped(mon, client):
    mon._exit_in_progress['LABX'] = True
    assert mon._exit_coverage_cycle()['checked'] == 0
    client.submit_stop_sell_order.assert_not_called()


def test_loop_survives_exception_on_one_watch(mon, client, caplog):
    mon.add_watch(symbol='ZZZ', stop_price=5.0, shares=10, tp_leg_id='', sl_leg_id='',
                  entry_price=5.5, risk_per_share=0.5, strategy='orb')
    mon._watches['ZZZ'].added_at = time.time() - 600
    mon._watches['ZZZ'].added_at = time.time() - 700     # ZZZ is checked first (oldest)
    real = exit_coverage.check

    def flaky(c, sym, qty, snap=None):
        if sym == 'ZZZ':
            raise RuntimeError('bad watch')
        return real(c, sym, qty, snap)
    exit_coverage.check = flaky
    try:
        counts = mon._exit_coverage_cycle()
    finally:
        exit_coverage.check = real
    assert counts['repaired'] == 1
    assert any('ZZZ check failed' in r.message for r in caplog.records if r.levelno == logging.WARNING)


def test_snapshot_failure_is_warning_not_crash(mon, client, caplog):
    client.get_open_positions.side_effect = AlpacaAPIError('down')
    assert mon._exit_coverage_cycle()['checked'] == 0
    assert any('LABX check failed' in r.message for r in caplog.records)


def test_disabled_flag_loop_exits_immediately(client):
    m = StopMonitor(api_key='k', api_secret='s', alpaca_client=client, exit_coverage_enabled=False)
    m._running = True
    asyncio.run(asyncio.wait_for(m._exit_coverage_loop(), timeout=2))
    m._start_exit_coverage_thread()
    assert m._coverage_thread is None


def test_loop_runs_cycle_after_delay(mon, client):
    mon._EXIT_COVERAGE_FIRST_RUN_DELAY_S = 0.0
    mon._EXIT_COVERAGE_PERIOD_S = 0.05
    mon._running = True

    async def run():
        task = asyncio.create_task(mon._exit_coverage_loop())
        await asyncio.sleep(0.4)
        mon._running = False
        await asyncio.wait_for(task, timeout=3)
    asyncio.run(run())
    assert client.submit_stop_sell_order.call_count == 1   # cooldown: ONE order across many cycles


def test_two_cycles_exactly_one_order(mon, client):
    """Integration-style: naked at cycle 1; the order then shows up at the broker; cycle 2 is quiet."""
    mon._exit_coverage_cycle()
    assert client.submit_stop_sell_order.call_count == 1
    client.get_open_orders.return_value = [_sell('stopsell-12345678', 423, cls='simple', status='new')]
    assert mon._exit_coverage_cycle()['naked'] == 0
    assert client.submit_stop_sell_order.call_count == 1


def test_still_naked_inside_cooldown_not_doubled(mon, client):
    mon._exit_coverage_cycle()
    mon._exit_coverage_cycle()      # broker has not shown the order yet
    assert client.submit_stop_sell_order.call_count == 1


def test_two_accounts_routed_per_strategy():
    live, paper = MagicMock(spec=AlpacaClient), MagicMock(spec=AlpacaClient)
    for c in (live, paper):
        c.get_open_orders.return_value = []
        c.submit_stop_sell_order.return_value = {'id': 'o-1234567890'}
        c.get_order.return_value = {'status': 'canceled'}
    live.get_open_positions.return_value = []
    paper.get_open_positions.return_value = [{'symbol': 'HOD1', 'qty': 50, 'side': 'long'}]
    m = StopMonitor(api_key='k', api_secret='s', alpaca_client=live, alpaca_clients_by_strategy={'hod_break': paper})
    m.add_watch(symbol='HOD1', stop_price=3.0, shares=50, tp_leg_id='', sl_leg_id='', entry_price=3.3,
                risk_per_share=0.3, strategy='hod_break')
    m._watches['HOD1'].added_at = time.time() - 600
    m._exit_coverage_cycle()
    paper.submit_stop_sell_order.assert_called_once_with(symbol='HOD1', qty=50, stop_price=3.0)
    live.submit_stop_sell_order.assert_not_called()


def test_check_pure_function_shapes(client):
    client.get_open_orders.return_value = [
        _sell('a', 100, cls='simple', status='new'),
        _sell('b', 100, cls='simple', status='canceled'),               # terminal: ignored
        {**_sell('c', 50, cls='simple', status='partially_filled', filled=20)},   # 30 remaining
        {**_sell('d', 99, cls='simple', status='new'), 'side': 'buy'},  # buy: ignored
    ]
    res = exit_coverage.check(client, 'LABX', 423)
    assert res.covered_qty == 130 and res.naked_qty == 293 and len(res.open_sell_orders) == 2
