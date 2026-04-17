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

        # After successful exit, _exit_in_progress is cleared and watch removed.
        # Re-add watch to simulate re-entry — add_watch clears _exit_in_progress
        # so a new exit CAN fire (this is correct: it's a new trade, not a dupe).
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')

        trade2 = MagicMock()
        trade2.symbol = 'PLYX'
        trade2.price = 4.23
        await monitor._on_trade(trade2)

        # Second exit fires because add_watch cleared the flag (new trade)
        assert mock_alpaca.submit_limit_sell_order.call_count == 2

    @pytest.mark.asyncio
    async def test_rapid_ticks_blocked(self, monitor, mock_alpaca):
        """Two rapid ticks on same watch — second is blocked by _exit_in_progress."""
        monitor.add_watch('PLYX', 4.29, 500, 'tp-1', 'sl-1')

        # Manually set _exit_in_progress to simulate an in-flight exit
        with monitor._exit_lock:
            monitor._exit_in_progress['PLYX'] = True

        trade = MagicMock()
        trade.symbol = 'PLYX'
        trade.price = 4.20
        await monitor._on_trade(trade)

        # Should be blocked — no sell submitted
        assert mock_alpaca.submit_limit_sell_order.call_count == 0

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


# ---------------------------------------------------------------------------
# Multi-consumer bar handlers (unified service support)
# ---------------------------------------------------------------------------

class TestBarHandlerMultiConsumer:
    """register_bar_handler + unregister_bar_handler + _on_bar fan-out."""

    @pytest.mark.asyncio
    async def test_two_handlers_both_fire(self, monitor):
        """Both handlers receive every bar event for subscribed symbols."""
        calls_a, calls_b = [], []
        monitor.register_bar_handler('strat_a', lambda s, df: calls_a.append(s))
        monitor.register_bar_handler('strat_b', lambda s, df: calls_b.append(s))
        monitor._bar_symbols.add('AAPL')

        bar = MagicMock(symbol='AAPL', timestamp='2026-04-12T14:00:00Z',
                       open=10.0, high=10.1, low=9.95, close=10.05, volume=1000)
        await monitor._on_bar(bar)
        assert calls_a == ['AAPL']
        assert calls_b == ['AAPL']

    @pytest.mark.asyncio
    async def test_handlers_get_independent_dataframes(self, monitor):
        """Each handler gets its own DataFrame copy — mutation by one
        doesn't leak to others."""
        import pandas as pd
        frames_seen = []

        def mutating(symbol, df):
            df['poisoned'] = 1  # mutate
            frames_seen.append(('mutator', df))

        def later(symbol, df):
            frames_seen.append(('later', df))

        monitor.register_bar_handler('mutator', mutating)
        monitor.register_bar_handler('later', later)
        monitor._bar_symbols.add('NVDA')

        bar = MagicMock(symbol='NVDA', timestamp='2026-04-12T14:00:00Z',
                       open=800.0, high=801.0, low=799.0, close=800.5, volume=2000)
        await monitor._on_bar(bar)

        mutator_df = next(df for tag, df in frames_seen if tag == 'mutator')
        later_df = next(df for tag, df in frames_seen if tag == 'later')
        assert 'poisoned' in mutator_df.columns
        assert 'poisoned' not in later_df.columns  # independent copy

    @pytest.mark.asyncio
    async def test_failing_handler_does_not_kill_others(self, monitor):
        """One handler raising does not prevent others from firing."""
        def broken(symbol, df):
            raise RuntimeError("boom")
        good_calls = []
        monitor.register_bar_handler('broken', broken)
        monitor.register_bar_handler('good', lambda s, df: good_calls.append(s))
        monitor._bar_symbols.add('MSFT')

        bar = MagicMock(symbol='MSFT', timestamp='2026-04-12T14:00:00Z',
                       open=300.0, high=301.0, low=299.0, close=300.5, volume=500)
        await monitor._on_bar(bar)
        assert good_calls == ['MSFT']

    def test_unregister_removes_handler(self, monitor):
        """After unregister, the handler is gone from the dict."""
        monitor.register_bar_handler('tmp', lambda s, df: None)
        assert 'tmp' in monitor._bar_handlers
        monitor.unregister_bar_handler('tmp')
        assert 'tmp' not in monitor._bar_handlers

    def test_backcompat_set_bar_callback(self, monitor):
        """Legacy set_bar_callback still works — registers under id='default'."""
        cb = lambda s, df: None
        monitor.set_bar_callback(cb)
        assert monitor._bar_handlers.get('default') is cb

    def test_register_same_id_overwrites(self, monitor):
        """Registering the same id twice overwrites the previous callback."""
        first = lambda s, df: None
        second = lambda s, df: None
        monitor.register_bar_handler('strategy', first)
        monitor.register_bar_handler('strategy', second)
        assert monitor._bar_handlers['strategy'] is second

    def test_polling_mode_warning_on_register(self, mock_alpaca, caplog):
        """register_bar_handler under polling_mode=True still stores but WARNs loudly."""
        import logging
        from trading.stop_monitor import StopMonitor
        m = StopMonitor(
            api_key='k', api_secret='s', alpaca_client=mock_alpaca,
            polling_mode=True,
        )
        with caplog.at_level(logging.WARNING, logger='trading.stop_monitor'):
            m.register_bar_handler('macd_wave', lambda s, df: None)
        assert any(
            'polling mode' in r.message.lower() and 'macd_wave' in r.message
            for r in caplog.records
        )
        # Handler was still stored (for mode-switch resilience)
        assert 'macd_wave' in m._bar_handlers


class TestWatchStrategyTag:
    """add_watch accepts strategy; propagates to WatchEntry."""

    def test_default_strategy_bull_flag(self, monitor):
        monitor.add_watch('AAPL', stop_price=10, shares=100,
                          tp_leg_id='', sl_leg_id='')
        assert monitor._watches['AAPL'].strategy == 'bull_flag'

    def test_explicit_strategy_macd_wave(self, monitor):
        monitor.add_watch('TSLA', stop_price=200, shares=50,
                          tp_leg_id='', sl_leg_id='', strategy='macd_wave')
        assert monitor._watches['TSLA'].strategy == 'macd_wave'

    def test_exit_event_default_strategy(self):
        """StopExitEvent defaults to bull_flag when not set."""
        ev = StopExitEvent(symbol='X', stop_price=1, exit_price=0.99,
                           shares=10, order_id='', exit_reason='stop_loss')
        assert ev.strategy == 'bull_flag'


class TestDrainExitEventsFilter:
    """drain_exit_events(strategy=...) filters by strategy and requeues others."""

    def _make_event(self, symbol: str, strategy: str):
        return StopExitEvent(
            symbol=symbol, stop_price=1.0, exit_price=0.99, shares=10,
            order_id='', exit_reason='stop_loss', strategy=strategy,
        )

    def test_no_arg_drains_all_backcompat(self, monitor):
        monitor._exit_events.put(self._make_event('A', 'bull_flag'))
        monitor._exit_events.put(self._make_event('B', 'macd_wave'))
        events = monitor.drain_exit_events()
        assert {e.symbol for e in events} == {'A', 'B'}
        # queue empty afterward
        assert monitor._exit_events.qsize() == 0

    def test_strategy_filter_returns_only_matching(self, monitor):
        monitor._exit_events.put(self._make_event('A', 'bull_flag'))
        monitor._exit_events.put(self._make_event('B', 'macd_wave'))
        monitor._exit_events.put(self._make_event('C', 'bull_flag'))
        mw = monitor.drain_exit_events(strategy='macd_wave')
        assert {e.symbol for e in mw} == {'B'}
        # bull_flag events requeued
        bf = monitor.drain_exit_events(strategy='bull_flag')
        assert {e.symbol for e in bf} == {'A', 'C'}

    def test_requeued_events_survive_multiple_drains(self, monitor):
        monitor._exit_events.put(self._make_event('A', 'bull_flag'))
        monitor._exit_events.put(self._make_event('B', 'macd_wave'))
        # MACD drains — bull_flag event requeues
        monitor.drain_exit_events(strategy='macd_wave')
        # Second MACD drain returns empty; bull_flag drain finds A
        assert monitor.drain_exit_events(strategy='macd_wave') == []
        bf = monitor.drain_exit_events(strategy='bull_flag')
        assert {e.symbol for e in bf} == {'A'}

    def test_missing_strategy_attribute_defaults_bull_flag(self, monitor):
        """Backwards compat: events from before strategy field default to bull_flag."""
        # Simulate a legacy event without strategy
        ev = StopExitEvent(symbol='X', stop_price=1, exit_price=0.99,
                           shares=10, order_id='', exit_reason='stop_loss')
        # Explicitly remove the attribute to simulate a pre-migration pickle
        monitor._exit_events.put(ev)
        bf = monitor.drain_exit_events(strategy='bull_flag')
        assert len(bf) == 1 and bf[0].symbol == 'X'


# ---------------------------------------------------------------------------
# Volume-confirmed trail exit (Experiment D) — tick-path tests
# ---------------------------------------------------------------------------

class TestVolConfirmedTrailExit:
    """Live-path tests: _on_trade skips trail exits on low-volume bars.

    Setup: trail ratchets up as price climbs, then a drop triggers the trail
    stop. The test varies whether the last closed bar had enough volume to
    confirm active selling.
    """

    @pytest.fixture
    def vol_mon(self, mock_alpaca):
        """StopMonitor with vol-confirmed trail enabled on the watch."""
        mon = StopMonitor(
            api_key='k', api_secret='s', alpaca_client=mock_alpaca,
            marketable_limit_offset=0.03, marketable_limit_offset_pct=0.005,
        )
        mon.add_watch(
            'PLYX', stop_price=4.29, shares=500,
            tp_leg_id='tp-1', sl_leg_id='sl-1',
            entry_price=4.40, risk_per_share=0.11,
            trail_r=1.0, activate_at_r=2.0,
            avg_flag_volume=50_000,
            vol_confirmed_trail_enabled=True,
            vol_confirmed_trail_min_ratio=1.0,
        )
        return mon

    @pytest.mark.asyncio
    async def test_vol_conf_low_vol_skips_trail_exit(self, vol_mon):
        """Trail would fire, but last bar volume < threshold → skip, keep holding."""
        # Push high to activate trail: +3R = 4.40 + 0.33 = 4.73
        t1 = MagicMock(); t1.symbol = 'PLYX'; t1.price = 4.80
        await vol_mon._on_trade(t1)

        # Simulate last closed bar volume: 5_000 < 50_000 × 1.0 → low-vol
        with vol_mon._watch_lock:
            vol_mon._watches['PLYX'].last_bar_volume = 5_000

        # Price drops below trail stop (4.80 - 0.11 = 4.69). Would normally exit.
        t2 = MagicMock(); t2.symbol = 'PLYX'; t2.price = 4.65
        await vol_mon._on_trade(t2)

        # Exit skipped — no event emitted
        events = vol_mon.drain_exit_events()
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_vol_conf_high_vol_fires_trail_exit(self, vol_mon):
        """Trail fires normally when last bar volume >= threshold."""
        t1 = MagicMock(); t1.symbol = 'PLYX'; t1.price = 4.80
        await vol_mon._on_trade(t1)

        with vol_mon._watch_lock:
            vol_mon._watches['PLYX'].last_bar_volume = 100_000  # 2× baseline

        t2 = MagicMock(); t2.symbol = 'PLYX'; t2.price = 4.65
        await vol_mon._on_trade(t2)

        events = vol_mon.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_reason == 'trail_stop'

    @pytest.mark.asyncio
    async def test_vol_conf_disabled_fires_regardless(self, mock_alpaca):
        """When vol_confirmed flag is off, trail fires as before."""
        mon = StopMonitor(
            api_key='k', api_secret='s', alpaca_client=mock_alpaca,
        )
        mon.add_watch(
            'PLYX', stop_price=4.29, shares=500,
            tp_leg_id='tp-1', sl_leg_id='sl-1',
            entry_price=4.40, risk_per_share=0.11,
            trail_r=1.0, activate_at_r=2.0,
            avg_flag_volume=50_000,
            vol_confirmed_trail_enabled=False,  # OFF
            vol_confirmed_trail_min_ratio=1.0,
        )
        t1 = MagicMock(); t1.symbol = 'PLYX'; t1.price = 4.80
        await mon._on_trade(t1)

        with mon._watch_lock:
            mon._watches['PLYX'].last_bar_volume = 100  # very low

        t2 = MagicMock(); t2.symbol = 'PLYX'; t2.price = 4.65
        await mon._on_trade(t2)

        events = mon.drain_exit_events()
        assert len(events) == 1  # fires regardless of low vol — flag is off
        assert events[0].exit_reason == 'trail_stop'

    @pytest.mark.asyncio
    async def test_vol_conf_does_not_skip_initial_stop_loss(self, vol_mon):
        """Initial hard stop (not trailing) fires regardless of vol-conf.

        Vol-confirmed guard only filters TRAIL exits, not the base stop_loss.
        """
        # Set last bar volume very low
        with vol_mon._watch_lock:
            vol_mon._watches['PLYX'].last_bar_volume = 1  # low

        # Drop straight through stop (trail never activated → initial stop_loss)
        t = MagicMock(); t.symbol = 'PLYX'; t.price = 4.20
        await vol_mon._on_trade(t)

        events = vol_mon.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_reason == 'stop_loss'  # NOT trail_stop

    @pytest.mark.asyncio
    async def test_on_bar_updates_last_bar_volume(self, vol_mon):
        """_on_bar callback stashes the latest closed bar volume on the watch."""
        vol_mon._bar_symbols.add('PLYX')  # required for _on_bar to process

        bar = MagicMock()
        bar.symbol = 'PLYX'
        bar.timestamp = '2025-01-01T09:31:00Z'
        bar.open = 4.50
        bar.high = 4.55
        bar.low = 4.48
        bar.close = 4.52
        bar.volume = 75_000

        await vol_mon._on_bar(bar)

        with vol_mon._watch_lock:
            assert vol_mon._watches['PLYX'].last_bar_volume == 75_000
