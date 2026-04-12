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

    def test_stale_pending_reclaims_slot(self):
        e = _make_engine()
        e.max_concurrent = 1
        stale = _make_pos(
            order_id='pending-xyz',
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=3),
        )
        e.open_positions = {'STALE': stale}
        assert e._has_entry_capacity() is True  # stale evicted, slot freed
        assert 'STALE' not in e.open_positions
        assert 'STALE' in e.invalidated

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
    def test_no_existing_orders(self):
        e = _make_engine()
        e.alpaca.trading_client.get_orders.return_value = []
        assert e._has_conflicting_alpaca_orders('AAPL') is False

    def test_existing_order_blocks(self):
        e = _make_engine()
        fake_order = SimpleNamespace(side=SimpleNamespace(value='buy'))
        e.alpaca.trading_client.get_orders.return_value = [fake_order]
        assert e._has_conflicting_alpaca_orders('AAPL') is True

    def test_fail_open_on_exception(self):
        e = _make_engine()
        e.alpaca.trading_client.get_orders.side_effect = RuntimeError("api down")
        # Should NOT raise; returns False so Alpaca is the final gate
        assert e._has_conflicting_alpaca_orders('AAPL') is False


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
