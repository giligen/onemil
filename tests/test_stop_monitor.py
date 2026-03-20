"""
Unit tests for StopMonitor — self-managed stop monitoring via WebSocket.

Tests cover:
- add_watch / remove_watch state management
- Price trigger detection (_on_trade callback)
- Limit price computation (fixed vs percentage offset)
- Double-fire prevention (_exit_in_progress flag)
- Exit event emission and draining
- Stop monitor start/stop lifecycle
- Thread safety of watch operations
"""

import asyncio
import pytest
import queue
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

from data_sources.alpaca_client import AlpacaClient
from trading.stop_monitor import (
    StopMonitor,
    StopExitEvent,
    WatchEntry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_alpaca():
    """Mocked AlpacaClient."""
    client = MagicMock(spec=AlpacaClient)
    client.cancel_order.return_value = True
    client.submit_limit_sell_order.return_value = {
        'id': 'sell-order-123',
        'status': 'accepted',
        'symbol': 'TEST',
    }
    client.close_position.return_value = {
        'id': 'close-order-456',
        'status': 'accepted',
        'symbol': 'TEST',
    }
    return client


@pytest.fixture
def monitor(mock_alpaca):
    """StopMonitor instance (not started — no WebSocket thread)."""
    return StopMonitor(
        api_key='test-key',
        api_secret='test-secret',
        alpaca_client=mock_alpaca,
        marketable_limit_offset=0.03,
        marketable_limit_offset_pct=0.005,
    )


# ---------------------------------------------------------------------------
# Limit price computation
# ---------------------------------------------------------------------------

class TestComputeLimitPrice:
    """Test marketable limit price calculation."""

    def test_fixed_offset_dominates_low_price(self, monitor):
        """For low-priced stocks, fixed $0.03 offset is larger than 0.5%."""
        # $4.00 * 0.005 = $0.02, but fixed = $0.03 → use $0.03
        limit = monitor.compute_limit_price(4.00)
        assert limit == 3.97  # 4.00 - 0.03

    def test_pct_offset_dominates_high_price(self, monitor):
        """For higher-priced stocks, 0.5% offset exceeds $0.03."""
        # $10.00 * 0.005 = $0.05 > $0.03 → use $0.05
        limit = monitor.compute_limit_price(10.00)
        assert limit == 9.95  # 10.00 - 0.05

    def test_exact_crossover_at_six_dollars(self, monitor):
        """At $6.00: fixed=$0.03, pct=$0.03 — equal, pick max = $0.03."""
        limit = monitor.compute_limit_price(6.00)
        assert limit == 5.97

    def test_floor_at_one_cent(self, monitor):
        """Limit price floors at $0.01."""
        limit = monitor.compute_limit_price(0.02)
        assert limit == 0.01

    def test_custom_offsets(self):
        """Custom offset values work correctly."""
        client = MagicMock(spec=AlpacaClient)
        mon = StopMonitor(
            api_key='k', api_secret='s', alpaca_client=client,
            marketable_limit_offset=0.10,
            marketable_limit_offset_pct=0.01,
        )
        # $8.00 * 0.01 = $0.08, but fixed = $0.10 → use $0.10
        assert mon.compute_limit_price(8.00) == 7.90


# ---------------------------------------------------------------------------
# Watch management
# ---------------------------------------------------------------------------

class TestWatchManagement:
    """Test add_watch / remove_watch / watched_symbols."""

    def test_add_watch(self, monitor):
        """add_watch registers a symbol."""
        monitor.add_watch(
            symbol='PLYX', stop_price=4.29, shares=500,
            tp_leg_id='tp-1', sl_leg_id='sl-1', trade_db_id=42,
        )
        assert 'PLYX' in monitor.watched_symbols
        assert len(monitor.watched_symbols) == 1

    def test_add_multiple_watches(self, monitor):
        """Multiple symbols can be watched simultaneously."""
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')
        monitor.add_watch('SVCO', 3.10, 300, 'tp-2', 'sl-2')
        assert set(monitor.watched_symbols) == {'PLYX', 'SVCO'}

    def test_remove_watch(self, monitor):
        """remove_watch removes the symbol from watch list."""
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')
        monitor.remove_watch('PLYX')
        assert 'PLYX' not in monitor.watched_symbols

    def test_remove_nonexistent_watch(self, monitor):
        """Removing a non-existent watch doesn't raise."""
        monitor.remove_watch('NOPE')  # Should not raise

    def test_replace_watch(self, monitor):
        """Adding same symbol replaces the existing watch."""
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')
        monitor.add_watch('PLYX', 4.15, 500, 'tp-2', 'sl-2')
        assert len(monitor.watched_symbols) == 1
        # Verify new stop price
        with monitor._watch_lock:
            entry = monitor._watches['PLYX']
        assert entry.stop_price == 4.15
        assert entry.tp_leg_id == 'tp-2'

    def test_watched_symbols_returns_copy(self, monitor):
        """watched_symbols returns a copy, not a reference."""
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')
        syms = monitor.watched_symbols
        syms.append('FAKE')
        assert 'FAKE' not in monitor.watched_symbols


# ---------------------------------------------------------------------------
# Trade callback — price trigger
# ---------------------------------------------------------------------------

class TestOnTrade:
    """Test the _on_trade WebSocket callback."""

    @pytest.mark.asyncio
    async def test_price_at_stop_triggers_exit(self, monitor, mock_alpaca):
        """Price exactly at stop level triggers exit."""
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1', trade_db_id=42)

        trade = MagicMock()
        trade.symbol = 'PLYX'
        trade.price = 4.29  # Exactly at stop

        await monitor._on_trade(trade)

        # Should have cancelled TP and SL legs
        assert mock_alpaca.cancel_order.call_count == 2
        # Should have submitted limit sell
        mock_alpaca.submit_limit_sell_order.assert_called_once()
        call_kwargs = mock_alpaca.submit_limit_sell_order.call_args
        assert call_kwargs[1]['symbol'] == 'PLYX'
        assert call_kwargs[1]['qty'] == 500

    @pytest.mark.asyncio
    async def test_price_below_stop_triggers_exit(self, monitor, mock_alpaca):
        """Price below stop level triggers exit."""
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')

        trade = MagicMock()
        trade.symbol = 'PLYX'
        trade.price = 4.20  # Below stop

        await monitor._on_trade(trade)

        mock_alpaca.submit_limit_sell_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_price_above_stop_no_exit(self, monitor, mock_alpaca):
        """Price above stop level does not trigger exit."""
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')

        trade = MagicMock()
        trade.symbol = 'PLYX'
        trade.price = 4.50  # Above stop

        await monitor._on_trade(trade)

        mock_alpaca.submit_limit_sell_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_unwatched_symbol_ignored(self, monitor, mock_alpaca):
        """Trades for unwatched symbols are ignored."""
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')

        trade = MagicMock()
        trade.symbol = 'NOPE'  # Not watched
        trade.price = 1.00

        await monitor._on_trade(trade)

        mock_alpaca.submit_limit_sell_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_exit_emits_event(self, monitor, mock_alpaca):
        """Triggered exit puts a StopExitEvent on the queue."""
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1', trade_db_id=42)

        trade = MagicMock()
        trade.symbol = 'PLYX'
        trade.price = 4.25

        await monitor._on_trade(trade)

        events = monitor.drain_exit_events()
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, StopExitEvent)
        assert event.symbol == 'PLYX'
        assert event.stop_price == 4.29
        assert event.shares == 500
        assert event.order_id == 'sell-order-123'
        assert event.exit_reason == 'stop_loss'
        assert event.trade_db_id == 42

    @pytest.mark.asyncio
    async def test_exit_removes_watch(self, monitor, mock_alpaca):
        """After exit, the symbol is removed from watch list."""
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')

        trade = MagicMock()
        trade.symbol = 'PLYX'
        trade.price = 4.20

        await monitor._on_trade(trade)

        assert 'PLYX' not in monitor.watched_symbols


# ---------------------------------------------------------------------------
# Double-fire prevention
# ---------------------------------------------------------------------------

class TestDoubleFire:
    """Test _exit_in_progress prevents duplicate exit attempts."""

    @pytest.mark.asyncio
    async def test_double_fire_prevented(self, monitor, mock_alpaca):
        """Rapid second tick for same symbol is blocked by _exit_in_progress."""
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')

        # First trigger fires and sets _exit_in_progress
        trade1 = MagicMock()
        trade1.symbol = 'PLYX'
        trade1.price = 4.25
        await monitor._on_trade(trade1)

        assert mock_alpaca.submit_limit_sell_order.call_count == 1

        # Re-add watch to simulate a re-entry. The _exit_in_progress flag
        # from the first fire is still set (only cleared on exception or
        # via remove_watch). This correctly blocks a rapid second exit.
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')

        trade2 = MagicMock()
        trade2.symbol = 'PLYX'
        trade2.price = 4.23
        await monitor._on_trade(trade2)

        # Second exit should be blocked
        assert mock_alpaca.submit_limit_sell_order.call_count == 1

        # After explicit remove_watch, the flag is cleared and re-entry works
        monitor.remove_watch('PLYX')
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')

        trade3 = MagicMock()
        trade3.symbol = 'PLYX'
        trade3.price = 4.20
        await monitor._on_trade(trade3)

        # Now it should fire again
        assert mock_alpaca.submit_limit_sell_order.call_count == 2

    @pytest.mark.asyncio
    async def test_exit_in_progress_blocks_duplicate(self, monitor, mock_alpaca):
        """Direct test: if _exit_in_progress is set, exit is skipped."""
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')

        # Manually set exit_in_progress
        with monitor._exit_lock:
            monitor._exit_in_progress['PLYX'] = True

        trade = MagicMock()
        trade.symbol = 'PLYX'
        trade.price = 4.20

        # Create watch entry for _execute_stop_exit
        with monitor._watch_lock:
            watch = monitor._watches.get('PLYX')

        await monitor._execute_stop_exit('PLYX', 4.20, watch)

        # Should NOT have submitted any sell order
        mock_alpaca.submit_limit_sell_order.assert_not_called()


# ---------------------------------------------------------------------------
# Fallback to close_position
# ---------------------------------------------------------------------------

class TestFallback:
    """Test fallback to market close when limit sell fails."""

    @pytest.mark.asyncio
    async def test_limit_sell_failure_falls_back(self, monitor, mock_alpaca):
        """When limit sell fails, falls back to close_position."""
        mock_alpaca.submit_limit_sell_order.side_effect = Exception("Order rejected")

        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')

        trade = MagicMock()
        trade.symbol = 'PLYX'
        trade.price = 4.20

        await monitor._on_trade(trade)

        # Should have tried limit sell, then fallen back to close_position
        mock_alpaca.submit_limit_sell_order.assert_called_once()
        mock_alpaca.close_position.assert_called_once_with('PLYX')

        events = monitor.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_reason == 'stop_loss_fallback'
        assert events[0].order_id == 'close-order-456'

    @pytest.mark.asyncio
    async def test_both_fail_no_event(self, monitor, mock_alpaca):
        """When both limit sell and close_position fail, no event emitted."""
        mock_alpaca.submit_limit_sell_order.side_effect = Exception("Rejected")
        mock_alpaca.close_position.side_effect = Exception("Also failed")

        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')

        trade = MagicMock()
        trade.symbol = 'PLYX'
        trade.price = 4.20

        await monitor._on_trade(trade)

        events = monitor.drain_exit_events()
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Drain exit events
# ---------------------------------------------------------------------------

class TestDrainEvents:
    """Test drain_exit_events queue behavior."""

    def test_drain_empty_queue(self, monitor):
        """Draining empty queue returns empty list."""
        events = monitor.drain_exit_events()
        assert events == []

    def test_drain_returns_all_events(self, monitor):
        """All queued events are returned and queue is emptied."""
        for i in range(3):
            monitor._exit_events.put(StopExitEvent(
                symbol=f'SYM{i}', stop_price=1.0, exit_price=0.97,
                shares=100, order_id=f'ord-{i}', exit_reason='stop_loss',
            ))

        events = monitor.drain_exit_events()
        assert len(events) == 3

        # Queue should be empty now
        more = monitor.drain_exit_events()
        assert len(more) == 0


# ---------------------------------------------------------------------------
# Leg cancellation
# ---------------------------------------------------------------------------

class TestLegCancellation:
    """Test bracket leg cancellation during exit."""

    @pytest.mark.asyncio
    async def test_cancels_both_legs(self, monitor, mock_alpaca):
        """Both TP and SL legs are cancelled."""
        monitor.add_watch('PLYX', 4.29, 500, 'tp-leg-abc', 'sl-leg-xyz')

        trade = MagicMock()
        trade.symbol = 'PLYX'
        trade.price = 4.20

        await monitor._on_trade(trade)

        cancel_calls = mock_alpaca.cancel_order.call_args_list
        cancelled_ids = [call[0][0] for call in cancel_calls]
        assert 'tp-leg-abc' in cancelled_ids
        assert 'sl-leg-xyz' in cancelled_ids

    @pytest.mark.asyncio
    async def test_cancel_failure_continues(self, monitor, mock_alpaca):
        """Cancel failure (e.g., 422 already filled) doesn't block exit."""
        mock_alpaca.cancel_order.side_effect = Exception("422 not cancelable")

        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')

        trade = MagicMock()
        trade.symbol = 'PLYX'
        trade.price = 4.20

        await monitor._on_trade(trade)

        # Limit sell should still be submitted despite cancel failures
        mock_alpaca.submit_limit_sell_order.assert_called_once()


# ---------------------------------------------------------------------------
# Start / Stop lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    """Test start/stop of StopMonitor."""

    def test_start_creates_thread(self, monitor):
        """start() creates a daemon thread."""
        with patch.object(monitor, '_run_stream_loop'):
            monitor.start()
            assert monitor._thread is not None
            assert monitor._thread.daemon is True
            monitor._running = False
            monitor._stop_event.set()
            monitor._thread.join(timeout=2)

    def test_stop_sets_running_false(self, monitor):
        """stop() sets _running to False."""
        monitor._running = True
        monitor._thread = MagicMock()
        monitor._thread.is_alive.return_value = False
        monitor.stop()
        assert monitor._running is False

    def test_double_start_warns(self, monitor):
        """Starting when already running logs warning."""
        monitor._thread = MagicMock()
        monitor._thread.is_alive.return_value = True
        monitor.start()  # Should warn but not crash


# ---------------------------------------------------------------------------
# WatchEntry dataclass
# ---------------------------------------------------------------------------

class TestWatchEntry:
    """Test WatchEntry dataclass."""

    def test_creation(self):
        """WatchEntry fields are set correctly."""
        entry = WatchEntry(
            symbol='PLYX', stop_price=4.29, shares=500,
            tp_leg_id='tp-1', sl_leg_id='sl-1', trade_db_id=42,
        )
        assert entry.symbol == 'PLYX'
        assert entry.stop_price == 4.29
        assert entry.shares == 500
        assert entry.trade_db_id == 42

    def test_default_trade_db_id(self):
        """trade_db_id defaults to None."""
        entry = WatchEntry(
            symbol='PLYX', stop_price=4.29, shares=500,
            tp_leg_id='tp-1', sl_leg_id='sl-1',
        )
        assert entry.trade_db_id is None


# ---------------------------------------------------------------------------
# StopExitEvent dataclass
# ---------------------------------------------------------------------------

class TestStopExitEvent:
    """Test StopExitEvent dataclass."""

    def test_creation(self):
        """StopExitEvent fields are set correctly."""
        event = StopExitEvent(
            symbol='PLYX', stop_price=4.29, exit_price=4.26,
            shares=500, order_id='ord-123', exit_reason='stop_loss',
            trade_db_id=42,
        )
        assert event.symbol == 'PLYX'
        assert event.exit_price == 4.26
        assert event.exit_reason == 'stop_loss'


# ---------------------------------------------------------------------------
# Trailing stop
# ---------------------------------------------------------------------------

class TestTrailingStop:
    """Test trailing stop logic in StopMonitor._on_trade."""

    @pytest.fixture
    def trail_monitor(self, mock_alpaca):
        """StopMonitor with a trailing stop watch."""
        mon = StopMonitor(
            api_key='k', api_secret='s', alpaca_client=mock_alpaca,
            marketable_limit_offset=0.03, marketable_limit_offset_pct=0.005,
        )
        mon.add_watch(
            'PLYX', stop_price=4.29, shares=500,
            tp_leg_id='tp-1', sl_leg_id='sl-1',
            entry_price=4.40, risk_per_share=0.11,
            trail_r=1.0, activate_at_r=2.0,
        )
        return mon

    def test_watch_stores_trailing_fields(self, trail_monitor):
        """add_watch stores trailing stop params in WatchEntry."""
        with trail_monitor._watch_lock:
            w = trail_monitor._watches['PLYX']
        assert w.entry_price == 4.40
        assert w.risk_per_share == 0.11
        assert w.trail_r == 1.0
        assert w.activate_at_r == 2.0
        assert w.highest_since_entry == 4.40
        assert w.trailing_active is False

    @pytest.mark.asyncio
    async def test_trail_not_active_before_threshold(self, trail_monitor):
        """Trail doesn't activate before +2R."""
        # +1R = 4.40 + 0.11 = 4.51 — not enough
        trade = MagicMock()
        trade.symbol = 'PLYX'
        trade.price = 4.51
        await trail_monitor._on_trade(trade)

        with trail_monitor._watch_lock:
            w = trail_monitor._watches['PLYX']
        assert w.trailing_active is False
        assert w.stop_price == 4.29  # unchanged

    @pytest.mark.asyncio
    async def test_trail_activates_at_threshold(self, trail_monitor):
        """Trail activates at +2R."""
        # +2R = 4.40 + 0.22 = 4.62, use 4.63 to clear float precision
        trade = MagicMock()
        trade.symbol = 'PLYX'
        trade.price = 4.63
        await trail_monitor._on_trade(trade)

        with trail_monitor._watch_lock:
            w = trail_monitor._watches['PLYX']
        assert w.trailing_active is True
        # New stop = 4.62 - 0.11 * 1.0 = 4.51
        assert w.stop_price == pytest.approx(4.51, abs=0.01)

    @pytest.mark.asyncio
    async def test_trail_ratchets_up(self, trail_monitor):
        """Trail ratchets stop up as price climbs."""
        # Activate at +2R
        t1 = MagicMock(); t1.symbol = 'PLYX'; t1.price = 4.62
        await trail_monitor._on_trade(t1)

        # Price climbs to 4.80 — stop should follow
        t2 = MagicMock(); t2.symbol = 'PLYX'; t2.price = 4.80
        await trail_monitor._on_trade(t2)

        with trail_monitor._watch_lock:
            w = trail_monitor._watches['PLYX']
        # New stop = 4.80 - 0.11 = 4.69
        assert w.stop_price == pytest.approx(4.69, abs=0.01)
        assert w.highest_since_entry == 4.80

    @pytest.mark.asyncio
    async def test_trail_never_ratchets_down(self, trail_monitor):
        """Trail stop never moves down when price drops."""
        # Activate and ratchet up
        t1 = MagicMock(); t1.symbol = 'PLYX'; t1.price = 4.80
        await trail_monitor._on_trade(t1)

        with trail_monitor._watch_lock:
            stop_after_up = trail_monitor._watches['PLYX'].stop_price

        # Price drops — stop should NOT move down
        t2 = MagicMock(); t2.symbol = 'PLYX'; t2.price = 4.72
        await trail_monitor._on_trade(t2)

        with trail_monitor._watch_lock:
            w = trail_monitor._watches['PLYX']
        assert w.stop_price == stop_after_up  # unchanged

    @pytest.mark.asyncio
    async def test_trail_exit_reason_is_trail_stop(self, trail_monitor, mock_alpaca):
        """Exit reason is 'trail_stop' when trailing is active."""
        # Activate trailing
        t1 = MagicMock(); t1.symbol = 'PLYX'; t1.price = 4.80
        await trail_monitor._on_trade(t1)

        # Price drops below trail stop level (4.80 - 0.11 = ~4.69)
        t2 = MagicMock(); t2.symbol = 'PLYX'; t2.price = 4.68
        await trail_monitor._on_trade(t2)

        events = trail_monitor.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_reason == 'trail_stop'

    @pytest.mark.asyncio
    async def test_fixed_stop_exit_reason_without_trail(self, monitor, mock_alpaca):
        """Without trailing, exit reason is 'stop_loss'."""
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')
        trade = MagicMock(); trade.symbol = 'PLYX'; trade.price = 4.20
        await monitor._on_trade(trade)

        events = monitor.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_reason == 'stop_loss'

    @pytest.mark.asyncio
    async def test_trail_disabled_when_trail_r_zero(self, mock_alpaca):
        """trail_r=0 means no trailing, behaves as fixed stop."""
        mon = StopMonitor(
            api_key='k', api_secret='s', alpaca_client=mock_alpaca,
        )
        mon.add_watch(
            'PLYX', stop_price=4.29, shares=500,
            tp_leg_id='tp-1', sl_leg_id='sl-1',
            entry_price=4.40, risk_per_share=0.11,
            trail_r=0.0, activate_at_r=0.0,
        )
        # Price goes way up — stop should NOT change
        t1 = MagicMock(); t1.symbol = 'PLYX'; t1.price = 5.00
        await mon._on_trade(t1)

        with mon._watch_lock:
            assert mon._watches['PLYX'].stop_price == 4.29

    def test_update_stop_moves_up(self, trail_monitor):
        """update_stop moves stop up."""
        result = trail_monitor.update_stop('PLYX', 4.50)
        assert result is True
        with trail_monitor._watch_lock:
            assert trail_monitor._watches['PLYX'].stop_price == 4.50

    def test_update_stop_rejects_lower(self, trail_monitor):
        """update_stop rejects lower stop price."""
        result = trail_monitor.update_stop('PLYX', 4.00)
        assert result is False
        with trail_monitor._watch_lock:
            assert trail_monitor._watches['PLYX'].stop_price == 4.29


# ---------------------------------------------------------------------------
# Backtest slippage cap integration
# ---------------------------------------------------------------------------

class TestBacktestSlippageCap:
    """Test that TradeSimulator._compute_stop_fill caps slippage correctly."""

    def test_no_cap_uses_raw_slippage(self):
        """Without marketable limit offset, raw slippage applies."""
        from backtest import TradeSimulator
        sim = TradeSimulator(exit_slippage_pct=0.001)
        # $10.00 stop, 0.1% slippage = $0.01
        fill = sim._compute_stop_fill(10.00)
        assert fill == pytest.approx(9.99, abs=0.001)

    def test_cap_limits_slippage(self):
        """Marketable limit offset caps the slippage."""
        from backtest import TradeSimulator
        sim = TradeSimulator(
            exit_slippage_pct=0.02,  # 2% raw slippage
            marketable_limit_offset=0.03,
            marketable_limit_offset_pct=0.005,
        )
        # $5.00 stop: raw_slip = $0.10 (2%)
        # fixed_cap = $0.03, pct_cap = $0.025 → cap = $0.03
        # Capped slip = min($0.10, $0.03) = $0.03
        fill = sim._compute_stop_fill(5.00)
        assert fill == pytest.approx(4.97, abs=0.001)

    def test_pct_cap_when_larger(self):
        """Percentage cap used when it exceeds fixed cap."""
        from backtest import TradeSimulator
        sim = TradeSimulator(
            exit_slippage_pct=0.02,  # 2% raw slippage
            marketable_limit_offset=0.03,
            marketable_limit_offset_pct=0.01,  # 1%
        )
        # $10.00 stop: raw_slip = $0.20 (2%)
        # fixed_cap = $0.03, pct_cap = $0.10 → cap = $0.10
        # Capped slip = min($0.20, $0.10) = $0.10
        fill = sim._compute_stop_fill(10.00)
        assert fill == pytest.approx(9.90, abs=0.001)

    def test_raw_slippage_smaller_than_cap(self):
        """When raw slippage is smaller than cap, raw slippage is used."""
        from backtest import TradeSimulator
        sim = TradeSimulator(
            exit_slippage_pct=0.001,  # 0.1% raw slippage
            marketable_limit_offset=0.03,
            marketable_limit_offset_pct=0.005,
        )
        # $10.00 stop: raw_slip = $0.01
        # cap = max($0.03, $0.05) = $0.05
        # min($0.01, $0.05) = $0.01 (raw < cap, so raw is used)
        fill = sim._compute_stop_fill(10.00)
        assert fill == pytest.approx(9.99, abs=0.001)

    def test_zero_slippage(self):
        """Zero slippage returns exact stop price."""
        from backtest import TradeSimulator
        sim = TradeSimulator(exit_slippage_pct=0.0)
        fill = sim._compute_stop_fill(5.00)
        assert fill == 5.00
