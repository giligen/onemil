"""
Unit tests for MACDWaveEngine helpers added in the execution-latency refactor.

Covers:
- _has_entry_capacity (hoisted capacity check)
- _has_conflicting_alpaca_orders (wash-trade pre-check)
- Bar event queue: normal put, Full handling, drain_bar_events, reset_daily
- check_entries(symbols=...) targeted path
"""

import queue as _q
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest

from trading.macd_wave_engine import MACDWaveEngine, OpenPosition, CrossedStock


def _make_engine(**overrides):
    """Minimal engine with mocked dependencies. No StopMonitor unless provided."""
    cfg = {
        'universe': {}, 'entry': {}, 'macd': {},
        'sizing': {'position_size': 50000, 'max_concurrent': 3},
        'risk': {'daily_loss_limit': -5000},
        'slippage': {}, 'waves': {},
    }
    defaults = dict(
        alpaca_client=MagicMock(),
        db=MagicMock(),
        config=cfg,
    )
    defaults.update(overrides)
    return MACDWaveEngine(**defaults)


def _make_pos(order_id='', entry_time=None):
    return OpenPosition(
        symbol='X', entry_price=10.0, shares=1000, hard_stop=9.8,
        trade_id=1, order_id=order_id,
        entry_time=entry_time or datetime.now(timezone.utc),
    )


class TestHasEntryCapacity:
    def test_empty_has_capacity(self):
        e = _make_engine()
        assert e._has_entry_capacity() is True

    def test_at_max_concurrent_no_capacity(self):
        e = _make_engine()
        e.max_concurrent = 2
        e.open_positions = {
            'A': _make_pos(order_id=''),  # filled
            'B': _make_pos(order_id=''),  # filled
        }
        assert e._has_entry_capacity() is False

    def test_stale_pending_counted_until_gc(self):
        """After split: stale-pending stays counted until _gc_stale_pending runs."""
        e = _make_engine()
        e.max_concurrent = 1
        stale = _make_pos(
            order_id='pending-xyz',
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=3),
        )
        e.open_positions = {'STALE': stale}
        # Pure predicate: stale order was >120s, so NOT counted as active.
        # With max=1 and 0 active, capacity is True.
        assert e._has_entry_capacity() is True
        # But the stale entry is still in open_positions until GC runs.
        assert 'STALE' in e.open_positions

    def test_gc_stale_pending_removes_old(self):
        e = _make_engine()
        stale = _make_pos(
            order_id='pending-xyz',
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=3),
        )
        fresh = _make_pos(
            order_id='pending-abc',
            entry_time=datetime.now(timezone.utc) - timedelta(seconds=30),
        )
        e.open_positions = {'STALE': stale, 'FRESH': fresh}
        e._gc_stale_pending()
        assert 'STALE' not in e.open_positions
        assert 'STALE' in e.invalidated
        assert 'FRESH' in e.open_positions  # untouched

    def test_has_entry_capacity_is_pure(self):
        """_has_entry_capacity must not mutate state."""
        e = _make_engine()
        stale = _make_pos(
            order_id='pending-xyz',
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=3),
        )
        e.open_positions = {'STALE': stale}
        invalidated_before = set(e.invalidated)
        positions_before = dict(e.open_positions)
        _ = e._has_entry_capacity()
        assert e.open_positions == positions_before
        assert e.invalidated == invalidated_before

    def test_fresh_pending_counts_as_active(self):
        e = _make_engine()
        e.max_concurrent = 1
        fresh = _make_pos(
            order_id='pending-abc',
            entry_time=datetime.now(timezone.utc) - timedelta(seconds=30),
        )
        e.open_positions = {'FRESH': fresh}
        assert e._has_entry_capacity() is False

    def test_daily_loss_limit_blocks(self):
        e = _make_engine()
        e.daily_pnl = -6000
        assert e._has_entry_capacity() is False


class TestConflictingOrdersCheck:
    """Covers both the fast (stream cache) and slow (REST) paths."""

    # --- Slow path: no stream or unhealthy stream → hit REST ---

    def test_no_existing_orders_rest(self):
        e = _make_engine()  # no order_stream attached
        e.alpaca.trading_client.get_orders.return_value = []
        assert e._has_conflicting_alpaca_orders('AAPL') is False

    def test_existing_order_blocks_rest(self):
        e = _make_engine()
        fake_order = SimpleNamespace(side=SimpleNamespace(value='buy'))
        e.alpaca.trading_client.get_orders.return_value = [fake_order]
        assert e._has_conflicting_alpaca_orders('AAPL') is True

    def test_fail_open_on_exception_rest(self):
        e = _make_engine()
        e.alpaca.trading_client.get_orders.side_effect = RuntimeError("api down")
        # Should NOT raise; returns False so Alpaca is the final gate
        assert e._has_conflicting_alpaca_orders('AAPL') is False

    # --- Fast path: healthy stream → no REST call ---

    def test_fast_path_conflict_via_stream(self):
        from trading.order_stream import OrderStreamWatcher
        e = _make_engine()
        stream = MagicMock(spec=OrderStreamWatcher)
        stream.is_healthy.return_value = True
        stream.get_open_order_symbols.return_value = {'AAPL'}
        e.order_stream = stream
        assert e._has_conflicting_alpaca_orders('AAPL') is True
        # REST must NOT be called when fast path is healthy
        e.alpaca.trading_client.get_orders.assert_not_called()

    def test_fast_path_no_conflict_via_stream(self):
        from trading.order_stream import OrderStreamWatcher
        e = _make_engine()
        stream = MagicMock(spec=OrderStreamWatcher)
        stream.is_healthy.return_value = True
        stream.get_open_order_symbols.return_value = {'MSFT', 'NVDA'}
        e.order_stream = stream
        assert e._has_conflicting_alpaca_orders('AAPL') is False
        e.alpaca.trading_client.get_orders.assert_not_called()

    def test_unhealthy_stream_falls_back_to_rest(self):
        from trading.order_stream import OrderStreamWatcher
        e = _make_engine()
        stream = MagicMock(spec=OrderStreamWatcher)
        stream.is_healthy.return_value = False  # unhealthy
        e.order_stream = stream
        e.alpaca.trading_client.get_orders.return_value = []
        assert e._has_conflicting_alpaca_orders('AAPL') is False
        # REST path WAS invoked
        e.alpaca.trading_client.get_orders.assert_called_once()


class TestBarEventQueue:
    def test_register_and_drain(self):
        e = _make_engine()
        from trading.stop_monitor import StopMonitor
        sm = MagicMock(spec=StopMonitor)
        e.stop_monitor = sm
        e.register_on_stop_monitor()
        # Capture the handler registered
        cb = sm.register_bar_handler.call_args[0][1]
        cb('AAPL', None)
        cb('MSFT', None)
        assert e.drain_bar_events() == {'AAPL', 'MSFT'}
        # Second drain is empty
        assert e.drain_bar_events() == set()

    def test_queue_full_logs_and_drops(self, caplog):
        e = _make_engine()
        from trading.stop_monitor import StopMonitor
        sm = MagicMock(spec=StopMonitor)
        e.stop_monitor = sm
        e.register_on_stop_monitor()
        cb = sm.register_bar_handler.call_args[0][1]
        # Fill the queue to cap
        for i in range(1000):
            cb(f'SYM{i}', None)
        # Next put should be dropped + logged
        import logging
        with caplog.at_level(logging.ERROR, logger='trading.macd_wave_engine'):
            cb('OVERFLOW', None)
        assert any('queue FULL' in r.message for r in caplog.records)
        assert e._bar_queue_full_logged is True

    def test_reset_daily_drains_queue(self):
        e = _make_engine()
        from trading.stop_monitor import StopMonitor
        sm = MagicMock(spec=StopMonitor)
        e.stop_monitor = sm
        e.register_on_stop_monitor()
        cb = sm.register_bar_handler.call_args[0][1]
        cb('STALE1', None)
        cb('STALE2', None)
        assert e._bar_event_queue.qsize() == 2
        e.reset_daily()
        assert e._bar_event_queue.qsize() == 0
