"""StopMonitor scale-out mechanics (ORB winner stack, 2026-08-22).

Covers the P0-5 collision protocol: leg resize with NEW-id capture,
submit-failure compensation, async fill booking (latch idempotence),
partial-fill adoption, cancel-race at stop time, and the
no-_exit_in_progress-across-the-wait rule.
"""
import asyncio
import time as time_mod
from unittest.mock import MagicMock

import pytest

from data_sources.alpaca_client import AlpacaClient
from trading.exit_reasons import ExitReason
from trading.stop_monitor import StopMonitor


@pytest.fixture
def mock_alpaca():
    client = MagicMock(spec=AlpacaClient)
    client.cancel_order.return_value = True
    client.replace_order_qty.side_effect = (
        lambda oid, qty: {'id': f'new-{oid}', 'status': 'accepted'})
    client.submit_limit_sell_order.return_value = {
        'id': 'scale-order-1', 'status': 'accepted'}
    # Default: scale order resting (tests override for fills).
    client.get_order.return_value = {
        'id': 'scale-order-1', 'status': 'new',
        'filled_qty': 0, 'filled_avg_price': None}
    client.get_open_positions.return_value = [
        {'symbol': 'TEST', 'qty': 600}]
    return client


@pytest.fixture
def monitor(mock_alpaca):
    mon = StopMonitor(
        api_key='k', api_secret='s', alpaca_client=mock_alpaca,
        marketable_limit_offset=0.03, marketable_limit_offset_pct=0.005,
    )
    mon._STOP_EXIT_FILL_TIMEOUT_S = 0.2
    mon._STOP_EXIT_POLL_INTERVAL_S = 0.05
    return mon


def _add_orb_watch(mon, shares=1000, stop=9.5, entry=10.0):
    mon.add_watch(
        symbol='TEST', stop_price=stop, shares=shares,
        tp_leg_id='tp-1', sl_leg_id='sl-1', trade_db_id=42,
        entry_price=entry, risk_per_share=entry - stop, strategy='orb',
        lock_arm_at_r=1.75, lock_stop_r=0.5, lock_r_unit=1.0,
    )
    return mon._watches['TEST']


def _tick(mon, price):
    trade = MagicMock()
    trade.symbol = 'TEST'
    trade.price = price
    asyncio.run(mon._on_trade(trade))


class TestArmScaleOut:
    def test_arm_success(self, monitor):
        w = _add_orb_watch(monitor)
        assert monitor.arm_scale_out('TEST', 13.0, 400) is True
        assert w.scale_at_px == 13.0
        assert w.scale_qty == 400

    def test_arm_not_watched(self, monitor):
        assert monitor.arm_scale_out('NOPE', 13.0, 400) is False

    def test_arm_tiny_qty_rejected(self, monitor):
        _add_orb_watch(monitor)
        assert monitor.arm_scale_out('TEST', 13.0, 0) is False

    def test_arm_after_done_rejected(self, monitor):
        w = _add_orb_watch(monitor)
        w.scale_done = True
        assert monitor.arm_scale_out('TEST', 13.0, 400) is False


class TestScaleTrigger:
    def test_tick_at_level_submits(self, monitor, mock_alpaca):
        w = _add_orb_watch(monitor)
        monitor.arm_scale_out('TEST', 13.0, 400)
        _tick(monitor, 13.05)
        # Legs resized to runner qty (600) with NEW ids captured (P0-5.3)
        mock_alpaca.replace_order_qty.assert_any_call('sl-1', 600)
        mock_alpaca.replace_order_qty.assert_any_call('tp-1', 600)
        assert w.sl_leg_id == 'new-sl-1'
        assert w.tp_leg_id == 'new-tp-1'
        # Independent limit sell at the level (never an OCO conversion)
        mock_alpaca.submit_limit_sell_order.assert_called_once_with(
            symbol='TEST', qty=400, limit_price=13.0)
        assert w.scale_order_id == 'scale-order-1'
        # P0-5.1: the latch is NOT held while the order rests
        assert monitor.is_exit_in_progress('TEST') is False
        # Shares untouched until the FILL books
        assert w.shares == 1000

    def test_tick_below_level_no_submit(self, monitor, mock_alpaca):
        _add_orb_watch(monitor)
        monitor.arm_scale_out('TEST', 13.0, 400)
        _tick(monitor, 12.99)
        mock_alpaca.submit_limit_sell_order.assert_not_called()

    def test_stop_wins_over_scale_same_tick(self, monitor, mock_alpaca):
        """price <= stop fires the stop path; the scale check never runs
        (live conservative ordering — P0-1 documented deviation)."""
        w = _add_orb_watch(monitor, stop=9.5)
        monitor.arm_scale_out('TEST', 13.0, 400)
        mock_alpaca.get_order.return_value = {
            'id': 'x', 'status': 'filled', 'filled_avg_price': 9.45,
            'filled_qty': 1000}
        mock_alpaca.get_open_positions.return_value = [
            {'symbol': 'TEST', 'qty': 1000}]
        _tick(monitor, 9.4)
        assert w.scale_order_id == ''
        # the sell submitted was the STOP exit, not the scale limit
        _, kwargs = mock_alpaca.submit_limit_sell_order.call_args
        assert kwargs['qty'] == 1000

    def test_unarmed_watch_never_scales(self, monitor, mock_alpaca):
        _add_orb_watch(monitor)
        _tick(monitor, 99.0)
        mock_alpaca.submit_limit_sell_order.assert_not_called()

    def test_no_double_submit_while_resting(self, monitor, mock_alpaca):
        w = _add_orb_watch(monitor)
        monitor.arm_scale_out('TEST', 13.0, 400)
        _tick(monitor, 13.05)
        _tick(monitor, 13.10)
        assert mock_alpaca.submit_limit_sell_order.call_count == 1
        assert w.scale_order_id == 'scale-order-1'


class TestLegFailureCompensation:
    def test_first_leg_resize_failure_skips_cycle(self, monitor, mock_alpaca):
        w = _add_orb_watch(monitor)
        monitor.arm_scale_out('TEST', 13.0, 400)
        mock_alpaca.replace_order_qty.side_effect = RuntimeError('api down')
        _tick(monitor, 13.05)
        mock_alpaca.submit_limit_sell_order.assert_not_called()
        assert w.scale_order_id == ''
        assert monitor.is_exit_in_progress('TEST') is False
        # retry allowed on a later tick
        mock_alpaca.replace_order_qty.side_effect = (
            lambda oid, qty: {'id': f'new-{oid}', 'status': 'accepted'})
        _tick(monitor, 13.06)
        assert w.scale_order_id == 'scale-order-1'

    def test_second_leg_failure_restores_first(self, monitor, mock_alpaca):
        """sl resize ok, tp resize fails -> sl restored to full qty."""
        w = _add_orb_watch(monitor)
        monitor.arm_scale_out('TEST', 13.0, 400)
        calls = []

        def _replace(oid, qty):
            calls.append((oid, qty))
            if oid == 'tp-1':
                raise RuntimeError('tp resize rejected')
            return {'id': f'new-{oid}', 'status': 'accepted'}
        mock_alpaca.replace_order_qty.side_effect = _replace
        _tick(monitor, 13.05)
        mock_alpaca.submit_limit_sell_order.assert_not_called()
        # compensation: the resized SL leg (now new-sl-1) restored to 1000
        assert ('new-sl-1', 1000) in calls
        assert w.scale_order_id == ''

    def test_submit_failure_restores_both_legs(self, monitor, mock_alpaca):
        """P0-5.2: legs resized OK but the scale submit fails -> both legs
        restored to the full position qty."""
        w = _add_orb_watch(monitor)
        monitor.arm_scale_out('TEST', 13.0, 400)
        mock_alpaca.submit_limit_sell_order.side_effect = RuntimeError('rej')
        _tick(monitor, 13.05)
        restore_calls = [c for c in mock_alpaca.replace_order_qty.call_args_list
                         if c.args[1] == 1000]
        assert len(restore_calls) == 2
        assert w.scale_order_id == ''
        assert monitor.is_exit_in_progress('TEST') is False


class TestFillBooking:
    def test_full_fill_books_once(self, monitor, mock_alpaca):
        w = _add_orb_watch(monitor)
        monitor.arm_scale_out('TEST', 13.0, 400)
        _tick(monitor, 13.05)
        mock_alpaca.get_order.return_value = {
            'id': 'scale-order-1', 'status': 'filled',
            'filled_qty': 400, 'filled_avg_price': 13.01}
        assert monitor._check_scale_fill_once('TEST') == 'filled'
        assert w.scale_done is True
        assert w.shares == 600
        assert w.scale_order_id == ''
        evs = monitor.drain_exit_events(strategy='orb')
        assert len(evs) == 1
        ev = evs[0]
        assert ev.exit_reason == ExitReason.SCALE_OUT.value
        assert ev.filled_qty == 400
        assert ev.exit_price == pytest.approx(13.01)
        assert ev.trade_db_id == 42
        # idempotent: a second booking attempt is a no-op
        assert monitor._check_scale_fill_once('TEST') == 'gone'
        assert w.shares == 600

    def test_partial_fill_of_scale_order_adopted(self, monitor, mock_alpaca):
        """Scale order partially filled then cancelled -> the partial leg
        books (never 'booked nowhere' — P0-5.4 class)."""
        w = _add_orb_watch(monitor)
        monitor.arm_scale_out('TEST', 13.0, 400)
        _tick(monitor, 13.05)
        mock_alpaca.get_order.return_value = {
            'id': 'scale-order-1', 'status': 'canceled',
            'filled_qty': 150, 'filled_avg_price': 13.0}
        assert monitor._check_scale_fill_once('TEST') == 'partial_adopted'
        assert w.scale_done is True
        assert w.shares == 850
        ev = monitor.drain_exit_events(strategy='orb')[0]
        assert ev.filled_qty == 150

    def test_cancelled_unfilled_clears_for_retry(self, monitor, mock_alpaca):
        w = _add_orb_watch(monitor)
        monitor.arm_scale_out('TEST', 13.0, 400)
        _tick(monitor, 13.05)
        mock_alpaca.get_order.return_value = {
            'id': 'scale-order-1', 'status': 'rejected',
            'filled_qty': 0, 'filled_avg_price': None}
        assert monitor._check_scale_fill_once('TEST') == 'aborted'
        assert w.scale_done is False
        assert w.scale_order_id == ''
        assert w.shares == 1000
        assert not monitor.drain_exit_events(strategy='orb')


class TestAdoptRestingScale:
    def test_adopt_filled(self, monitor, mock_alpaca):
        w = _add_orb_watch(monitor)
        monitor.arm_scale_out('TEST', 13.0, 400)
        _tick(monitor, 13.05)
        mock_alpaca.get_order.return_value = {
            'id': 'scale-order-1', 'status': 'canceled',
            'filled_qty': 400, 'filled_avg_price': 13.02}
        monitor.adopt_resting_scale('TEST')
        mock_alpaca.cancel_order.assert_any_call('scale-order-1')
        assert w.scale_done is True
        assert w.shares == 600

    def test_adopt_unfilled_clears(self, monitor, mock_alpaca):
        w = _add_orb_watch(monitor)
        monitor.arm_scale_out('TEST', 13.0, 400)
        _tick(monitor, 13.05)
        monitor.adopt_resting_scale('TEST')
        assert w.scale_order_id == ''
        assert w.scale_done is False
        assert w.shares == 1000

    def test_adopt_noop_without_order(self, monitor, mock_alpaca):
        _add_orb_watch(monitor)
        mock_alpaca.cancel_order.reset_mock()
        monitor.adopt_resting_scale('TEST')
        mock_alpaca.cancel_order.assert_not_called()


class TestStopWhileScaleResting:
    def test_stop_exit_adopts_partial_then_sells_runner(self, monitor,
                                                        mock_alpaca):
        """Cancel race (P0-5.4): stop fires while the scale order rests with
        a partial fill — the exit books the scale leg FIRST (event ordering)
        then sells the runner qty."""
        w = _add_orb_watch(monitor)
        monitor.arm_scale_out('TEST', 13.0, 400)
        _tick(monitor, 13.05)
        assert w.scale_order_id == 'scale-order-1'

        def _get_order(oid):
            if oid == 'scale-order-1':
                return {'id': oid, 'status': 'canceled',
                        'filled_qty': 150, 'filled_avg_price': 13.0}
            return {'id': oid, 'status': 'filled',
                    'filled_avg_price': 9.44, 'filled_qty': 850}
        mock_alpaca.get_order.side_effect = _get_order
        mock_alpaca.get_open_positions.return_value = [
            {'symbol': 'TEST', 'qty': 850}]
        mock_alpaca.submit_limit_sell_order.reset_mock()
        mock_alpaca.submit_limit_sell_order.return_value = {
            'id': 'stop-sell-1', 'status': 'accepted'}
        _tick(monitor, 9.4)
        evs = monitor.drain_exit_events(strategy='orb')
        # the 13.05 trigger tick armed the static lock (>= entry+1.75R), so
        # the final exit classifies as lock_stop — the ordering contract is
        # what matters: scale event FIRST, then the terminal exit.
        assert [e.exit_reason for e in evs] == [
            ExitReason.SCALE_OUT.value, ExitReason.LOCK_STOP.value]
        assert evs[0].filled_qty == 150
        assert evs[1].shares == 850      # runner qty, not the entry qty
        _, kwargs = mock_alpaca.submit_limit_sell_order.call_args
        assert kwargs['qty'] == 850

    def test_scale_watcher_stops_after_watch_removed(self, monitor,
                                                     mock_alpaca):
        """The async fill watcher terminates when the watch disappears
        (bounded-by-construction)."""
        w = _add_orb_watch(monitor)
        monitor.arm_scale_out('TEST', 13.0, 400)
        w.scale_order_id = 'scale-order-1'
        monitor._SCALE_FILL_POLL_INTERVAL_S = 0.01

        async def run():
            task = asyncio.ensure_future(monitor._scale_fill_watcher('TEST'))
            await asyncio.sleep(0.03)
            monitor.remove_watch('TEST')
            await asyncio.wait_for(task, timeout=1.0)
        asyncio.run(run())


class TestRehydratedWatch:
    def test_add_watch_with_scale_done(self, monitor):
        """P0-4: restart mid-scale re-adds the watch scale-done so the leg
        can never double-sell, with runner shares."""
        monitor.add_watch(
            symbol='TEST', stop_price=9.6, shares=600,
            tp_leg_id='', sl_leg_id='', trade_db_id=42,
            entry_price=10.0, risk_per_share=0.4, strategy='orb',
            scale_at_px=13.0, scale_qty=400, scale_done=True,
        )
        w = monitor._watches['TEST']
        assert w.scale_done is True
        assert w.shares == 600
        # a tick at the level never re-submits
        _tick(monitor, 13.5)
        assert w.scale_order_id == ''

    def test_snapshot_exposes_scale_fields(self, monitor):
        _add_orb_watch(monitor)
        monitor.arm_scale_out('TEST', 13.0, 400)
        snap = monitor.get_watch_snapshot('TEST')
        assert snap['scale_at_px'] == 13.0
        assert snap['scale_qty'] == 400
        assert snap['scale_done'] is False
