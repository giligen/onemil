"""
Unit tests for OrderStreamWatcher — thread-safe status map + fallback semantics.

Tests avoid touching the real Alpaca TradingStream; callback is invoked
directly with fake payloads to simulate push events.
"""

import asyncio
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from trading.order_stream import OrderStreamWatcher, _order_to_status


def _fake_trade_update(order_id: str, status: str, event: str = 'fill',
                      filled_avg_price: float = 10.0, filled_qty: int = 100):
    """Construct a SimpleNamespace shaped like Alpaca's TradeUpdate event."""
    order = SimpleNamespace(
        id=order_id,
        status=SimpleNamespace(value=status),
        filled_avg_price=filled_avg_price,
        filled_qty=filled_qty,
        submitted_at=datetime.now(timezone.utc),
        filled_at=datetime.now(timezone.utc) if status == 'filled' else None,
    )
    return SimpleNamespace(order=order, event=SimpleNamespace(value=event))


class TestGetStatus:
    @pytest.mark.asyncio
    async def test_callback_populates_dict(self):
        w = OrderStreamWatcher(api_key='k', api_secret='s')
        assert w.get_status('abc') is None
        await w._on_trade_update(_fake_trade_update('abc', 'filled'))
        st = w.get_status('abc')
        assert st is not None
        assert st['status'] == 'filled'
        assert st['filled_avg_price'] == 10.0
        assert st['filled_qty'] == 100
        assert st['event'] == 'fill'

    @pytest.mark.asyncio
    async def test_status_updates_overwrite(self):
        w = OrderStreamWatcher(api_key='k', api_secret='s')
        await w._on_trade_update(_fake_trade_update('x', 'partial_fill', event='partial_fill', filled_qty=50))
        await w._on_trade_update(_fake_trade_update('x', 'filled', filled_qty=100))
        st = w.get_status('x')
        assert st['status'] == 'filled'
        assert st['filled_qty'] == 100

    @pytest.mark.asyncio
    async def test_missing_order_returns_none(self):
        w = OrderStreamWatcher(api_key='k', api_secret='s')
        assert w.get_status('never-seen') is None


class TestIsHealthy:
    def test_not_running(self):
        w = OrderStreamWatcher(api_key='k', api_secret='s')
        assert w.is_healthy() is False

    def test_running_no_events_yet_is_healthy(self):
        w = OrderStreamWatcher(api_key='k', api_secret='s')
        # Simulate running state without actually starting the thread
        w._running = True
        w._thread = threading.current_thread()  # alive
        assert w.is_healthy() is True  # zero events tolerated pre-first-event

    def test_stale_after_event(self):
        w = OrderStreamWatcher(api_key='k', api_secret='s')
        w._running = True
        w._thread = threading.current_thread()
        w._last_event_ts = time.time() - 120  # 2 min ago
        assert w.is_healthy(max_stale_s=30) is False


class TestThreadSafety:
    @pytest.mark.asyncio
    async def test_concurrent_writes_reads(self):
        """Many writer threads + many readers — no dropped keys, no exceptions."""
        w = OrderStreamWatcher(api_key='k', api_secret='s')

        # Writer does async updates
        async def writer(n):
            for i in range(n):
                await w._on_trade_update(_fake_trade_update(f'ord-{i}', 'filled'))

        async def reader(n):
            for i in range(n):
                _ = w.get_status(f'ord-{i % 50}')

        await asyncio.gather(writer(200), reader(500), reader(500))
        # All 200 orders should be present
        for i in range(200):
            assert w.get_status(f'ord-{i}') is not None


class TestGetOpenOrderSymbols:
    """Symbol-level view used by MACD's fast conflict check."""

    @pytest.mark.asyncio
    async def test_open_orders_only(self):
        w = OrderStreamWatcher(api_key='k', api_secret='s')
        await w._on_trade_update(_fake_trade_update('o1', 'new', event='new'))
        await w._on_trade_update(_fake_trade_update('o2', 'accepted', event='accepted'))
        await w._on_trade_update(_fake_trade_update('o3', 'filled'))
        # Manually set symbols so the fake update's missing symbol doesn't leak
        # (SimpleNamespace in _fake_trade_update doesn't set symbol by default)
        with w._lock:
            w._statuses['o1']['symbol'] = 'AAA'
            w._statuses['o2']['symbol'] = 'BBB'
            w._statuses['o3']['symbol'] = 'CCC'
        syms = w.get_open_order_symbols()
        assert syms == {'AAA', 'BBB'}  # 'CCC' is filled (terminal)

    @pytest.mark.asyncio
    async def test_empty_when_all_terminal(self):
        w = OrderStreamWatcher(api_key='k', api_secret='s')
        await w._on_trade_update(_fake_trade_update('o1', 'filled'))
        await w._on_trade_update(_fake_trade_update('o2', 'canceled'))
        with w._lock:
            w._statuses['o1']['symbol'] = 'A'
            w._statuses['o2']['symbol'] = 'B'
        assert w.get_open_order_symbols() == set()

    @pytest.mark.asyncio
    async def test_missing_symbol_excluded(self):
        """Orders without a symbol field are defensively excluded."""
        w = OrderStreamWatcher(api_key='k', api_secret='s')
        await w._on_trade_update(_fake_trade_update('o1', 'new'))
        with w._lock:
            w._statuses['o1']['symbol'] = ''  # simulate missing
        assert w.get_open_order_symbols() == set()


class TestStatusPruning:
    """Terminal-order eviction bounds memory under long-running use."""

    @pytest.mark.asyncio
    async def test_open_orders_never_pruned(self):
        from trading.order_stream import _MAX_TERMINAL_ENTRIES
        w = OrderStreamWatcher(api_key='k', api_secret='s')
        # 5 open (non-terminal) orders
        for i in range(5):
            await w._on_trade_update(_fake_trade_update(f'open-{i}', 'new', event='new'))
        # Overflow: 1.5x cap of terminal orders
        for i in range(_MAX_TERMINAL_ENTRIES + 100):
            await w._on_trade_update(_fake_trade_update(f'filled-{i}', 'filled'))
        # Open ones still there
        for i in range(5):
            assert w.get_status(f'open-{i}') is not None

    @pytest.mark.asyncio
    async def test_terminal_eviction_order(self):
        from trading.order_stream import _MAX_TERMINAL_ENTRIES
        w = OrderStreamWatcher(api_key='k', api_secret='s')
        # Insert cap + 3 terminal orders; oldest 3 should be evicted
        for i in range(_MAX_TERMINAL_ENTRIES + 3):
            await w._on_trade_update(_fake_trade_update(f'f-{i}', 'filled'))
        # First 3 evicted, last N kept
        assert w.get_status('f-0') is None
        assert w.get_status('f-1') is None
        assert w.get_status('f-2') is None
        assert w.get_status(f'f-{_MAX_TERMINAL_ENTRIES + 2}') is not None


class TestOrderToStatus:
    def test_plain_object(self):
        order = SimpleNamespace(
            id='o1', status='filled', filled_avg_price=12.5, filled_qty=50,
            submitted_at=datetime.now(timezone.utc), filled_at=None,
        )
        st = _order_to_status(order, event='fill')
        assert st['id'] == 'o1'
        assert st['status'] == 'filled'
        assert st['filled_avg_price'] == 12.5
        assert st['filled_qty'] == 50
        assert st['event'] == 'fill'

    def test_enum_status(self):
        order = SimpleNamespace(
            id='o2', status=SimpleNamespace(value='canceled'),
            filled_avg_price=None, filled_qty=0,
            submitted_at=None, filled_at=None,
        )
        st = _order_to_status(order)
        assert st['status'] == 'canceled'

    def test_none_fields_survive(self):
        order = SimpleNamespace(
            id='o3', status='', filled_avg_price=None, filled_qty=None,
            submitted_at=None, filled_at=None,
        )
        st = _order_to_status(order)
        assert st['filled_avg_price'] is None
        assert st['filled_qty'] == 0
