"""
Unit tests for production exhaustion exit implementation.

Tests cover:
- StopMonitor.get_watch_snapshot() — thread-safe state reading
- StopMonitor.execute_partial_exit() — partial sell + trail tightening
- WatchEntry.exhaustion_partial_taken field
- TradingEngine._check_exhaustion_exits() — signal detection → partial exit
- TradingEngine._process_stop_monitor_exits() — combined P&L calculation
- Database migration for partial exit columns
"""

import asyncio
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from data_sources.alpaca_client import AlpacaClient
from trading.stop_monitor import StopMonitor, StopExitEvent, WatchEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_alpaca():
    """Mocked AlpacaClient."""
    client = MagicMock(spec=AlpacaClient)
    client.cancel_order.return_value = True
    client.submit_limit_sell_order.return_value = {
        'id': 'partial-sell-001',
        'status': 'accepted',
        'symbol': 'TEST',
    }
    # Quote for spread-based exit pricing
    client.get_latest_quote.return_value = {
        'bid_price': 8.48, 'ask_price': 8.50,
        'bid_size': 100, 'ask_size': 200,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    # Order fill confirmation (for partial exit fill wait)
    client.get_order.return_value = {
        'status': 'filled', 'filled_avg_price': 8.49,
    }
    client.replace_order_qty.return_value = {'id': 'sl-456', 'status': 'accepted'}
    return client


@pytest.fixture
def monitor(mock_alpaca):
    """StopMonitor with a watched position."""
    mon = StopMonitor(
        api_key='test-key',
        api_secret='test-secret',
        alpaca_client=mock_alpaca,
        marketable_limit_offset=0.03,
        marketable_limit_offset_pct=0.005,
    )
    mon.add_watch(
        symbol='TEST',
        stop_price=4.50,
        shares=1000,
        tp_leg_id='tp-123',
        sl_leg_id='sl-456',
        trade_db_id=42,
        entry_price=5.00,
        risk_per_share=0.50,
        trail_r=1.0,
        activate_at_r=2.0,
    )
    return mon


# ---------------------------------------------------------------------------
# WatchEntry — exhaustion_partial_taken field
# ---------------------------------------------------------------------------

class TestWatchEntryExhaustionField:
    """Test the new exhaustion_partial_taken field on WatchEntry."""

    def test_defaults_to_false(self):
        """New WatchEntry has exhaustion_partial_taken=False."""
        entry = WatchEntry(
            symbol='TEST', stop_price=4.50, shares=1000,
            tp_leg_id='tp', sl_leg_id='sl',
        )
        assert entry.exhaustion_partial_taken is False

    def test_can_set_to_true(self):
        """Field can be toggled."""
        entry = WatchEntry(
            symbol='TEST', stop_price=4.50, shares=1000,
            tp_leg_id='tp', sl_leg_id='sl',
        )
        entry.exhaustion_partial_taken = True
        assert entry.exhaustion_partial_taken is True


# ---------------------------------------------------------------------------
# get_watch_snapshot
# ---------------------------------------------------------------------------

class TestGetWatchSnapshot:
    """Test thread-safe snapshot reading."""

    def test_returns_snapshot_for_watched_symbol(self, monitor):
        """Snapshot contains all expected fields."""
        snap = monitor.get_watch_snapshot('TEST')
        assert snap is not None
        assert snap['entry_price'] == 5.00
        assert snap['risk_per_share'] == 0.50
        assert snap['shares'] == 1000
        assert snap['exhaustion_partial_taken'] is False
        assert snap['trade_db_id'] == 42
        assert snap['stop_price'] == 4.50
        assert snap['trail_r'] == 1.0

    def test_returns_none_for_unwatched(self, monitor):
        """Unwatched symbol → None."""
        assert monitor.get_watch_snapshot('NOPE') is None

    def test_snapshot_is_detached_copy(self, monitor):
        """Mutating snapshot doesn't affect internal watch."""
        snap = monitor.get_watch_snapshot('TEST')
        snap['shares'] = 0
        snap2 = monitor.get_watch_snapshot('TEST')
        assert snap2['shares'] == 1000


# ---------------------------------------------------------------------------
# execute_partial_exit
# ---------------------------------------------------------------------------

class TestExecutePartialExit:
    """Test partial exit execution."""

    def test_sells_fraction_and_tightens_trail(self, monitor, mock_alpaca):
        """execute_partial_exit sells 50% and tightens trail to 0.5R."""
        event = monitor.execute_partial_exit(
            symbol='TEST',
            fraction=0.5,
            tighter_trail_r=0.5,
        )

        assert event is not None
        assert event.shares == 500
        assert event.exit_reason == 'exhaustion_partial'
        assert event.trade_db_id == 42
        assert event.order_id == 'partial-sell-001'

        # Verify internal state updated
        snap = monitor.get_watch_snapshot('TEST')
        assert snap['shares'] == 500
        assert snap['exhaustion_partial_taken'] is True
        # Trail should be tightened
        assert snap['trail_r'] == 0.5

    def test_emits_exit_event_on_queue(self, monitor, mock_alpaca):
        """Partial exit puts event on exit_events queue."""
        monitor.execute_partial_exit('TEST', 0.5, 0.5)
        events = monitor.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_reason == 'exhaustion_partial'

    def test_no_double_partial(self, monitor, mock_alpaca):
        """Second call after partial already taken → None."""
        monitor.execute_partial_exit('TEST', 0.5, 0.5)
        result = monitor.execute_partial_exit('TEST', 0.5, 0.5)
        assert result is None

    def test_returns_none_for_unwatched(self, monitor):
        """Unwatched symbol → None."""
        result = monitor.execute_partial_exit('NOPE', 0.5, 0.5)
        assert result is None

    def test_ratchets_stop_with_tighter_trail(self, monitor, mock_alpaca):
        """After partial, stop ratchets up using tighter trail distance."""
        # Set highest high to trigger ratchet
        with monitor._watch_lock:
            monitor._watches['TEST'].highest_since_entry = 7.00

        monitor.execute_partial_exit('TEST', 0.5, 0.5)
        snap = monitor.get_watch_snapshot('TEST')
        # new_stop = 7.00 - 0.50 * 0.5 = 6.75
        assert snap['stop_price'] == 6.75

    def test_api_failure_returns_none(self, monitor, mock_alpaca):
        """If limit sell fails, returns None without updating state."""
        mock_alpaca.submit_limit_sell_order.side_effect = Exception("API down")
        result = monitor.execute_partial_exit('TEST', 0.5, 0.5)
        assert result is None
        # State unchanged
        snap = monitor.get_watch_snapshot('TEST')
        assert snap['exhaustion_partial_taken'] is False
        assert snap['shares'] == 1000

    def test_tiny_position_no_shares_to_sell(self):
        """Single share × 0.5 = 0 → no partial."""
        mock_client = MagicMock(spec=AlpacaClient)
        mon = StopMonitor('k', 's', mock_client)
        mon.add_watch('TINY', 4.0, 1, 'tp', 'sl',
                       entry_price=5.0, risk_per_share=1.0)
        result = mon.execute_partial_exit('TINY', 0.5, 0.5)
        assert result is None


# ---------------------------------------------------------------------------
# TradingEngine._check_exhaustion_exits
# ---------------------------------------------------------------------------

class TestCheckExhaustionExits:
    """Test the 60s polling exhaustion check in TradingEngine."""

    def _make_engine(self, mock_alpaca, mock_db, mock_stop_monitor,
                     exhaustion_enabled=True):
        """Build a minimal TradingEngine with exhaustion config."""
        from trading.pattern_detector import BullFlagDetector
        from trading.trade_planner import TradePlanner
        from trading.order_executor import OrderExecutor
        from trading.position_manager import PositionManager

        with patch('config.Config') as MockConfig:
            MockConfig._load_yaml_only.return_value = {
                'trading': {
                    'trailing_stop': {'enabled': True, 'trail_r': 1.0, 'activate_at_r': 2.0},
                    'skip_fridays': False,
                    'macd_zones': {'enabled': False},
                    'exhaustion_exit': {
                        'enabled': exhaustion_enabled,
                        'partial_fraction': 0.5,
                        'tighter_trail_r': 0.5,
                        'min_profit_r': 3.0,
                        'signals': {
                            'climax_candle': True,
                            'shooting_star': True,
                            'volume_divergence': False,
                            'shrinking_bodies': False,
                        },
                    },
                }
            }
            from trading.trading_engine import TradingEngine
            engine = TradingEngine(
                alpaca_client=mock_alpaca,
                db=mock_db,
                detector=MagicMock(spec=BullFlagDetector),
                planner=MagicMock(spec=TradePlanner),
                executor=MagicMock(spec=OrderExecutor),
                position_manager=MagicMock(spec=PositionManager),
                enabled=True,
                stop_monitor=mock_stop_monitor,
            )
        return engine

    def test_skips_when_disabled(self):
        """Exhaustion disabled → no API calls."""
        mock_alpaca = MagicMock(spec=AlpacaClient)
        mock_db = MagicMock()
        mock_sm = MagicMock()
        engine = self._make_engine(mock_alpaca, mock_db, mock_sm,
                                   exhaustion_enabled=False)
        engine._check_exhaustion_exits()
        mock_sm.get_watch_snapshot.assert_not_called()

    def test_skips_when_no_stop_monitor(self):
        """No stop_monitor → no-op."""
        mock_alpaca = MagicMock(spec=AlpacaClient)
        mock_db = MagicMock()
        engine = self._make_engine(mock_alpaca, mock_db, None)
        engine.stop_monitor = None
        engine._check_exhaustion_exits()  # Should not raise

    def test_skips_partial_already_taken(self):
        """If exhaustion_partial_taken, skip symbol."""
        mock_alpaca = MagicMock(spec=AlpacaClient)
        mock_db = MagicMock()
        mock_sm = MagicMock()
        mock_sm.watched_symbols = ['TEST']
        mock_sm.get_watch_snapshot.return_value = {
            'entry_price': 5.0, 'risk_per_share': 0.5,
            'highest_since_entry': 8.0, 'shares': 500,
            'exhaustion_partial_taken': True,
            'trade_db_id': 42, 'stop_price': 6.0,
            'trail_r': 0.5, 'trailing_active': True,
        }
        engine = self._make_engine(mock_alpaca, mock_db, mock_sm)
        engine._check_exhaustion_exits()
        mock_alpaca.get_1min_bars.assert_not_called()

    def test_skips_below_min_profit_r(self):
        """Below min_profit_r (3.0) → no signal check."""
        mock_alpaca = MagicMock(spec=AlpacaClient)
        mock_db = MagicMock()
        mock_sm = MagicMock()
        mock_sm.watched_symbols = ['TEST']
        mock_sm.get_watch_snapshot.return_value = {
            'entry_price': 5.0, 'risk_per_share': 0.5,
            'highest_since_entry': 6.0, 'shares': 1000,
            'exhaustion_partial_taken': False,
            'trade_db_id': 42, 'stop_price': 4.5,
            'trail_r': 1.0, 'trailing_active': True,
        }
        # Bars with close at 6.0 → R = (6.0-5.0)/0.5 = 2.0 < 3.0
        bars_data = [
            {'timestamp': datetime.now(timezone.utc), 'open': 5.5, 'close': 5.8, 'high': 5.9, 'low': 5.4, 'volume': 1000},
            {'timestamp': datetime.now(timezone.utc), 'open': 5.8, 'close': 6.0, 'high': 6.1, 'low': 5.7, 'volume': 1000},
            {'timestamp': datetime.now(timezone.utc), 'open': 6.0, 'close': 6.0, 'high': 6.1, 'low': 5.9, 'volume': 1000},  # in-progress
        ]
        mock_alpaca.get_1min_bars.return_value = pd.DataFrame(bars_data)
        engine = self._make_engine(mock_alpaca, mock_db, mock_sm)
        engine._check_exhaustion_exits()
        mock_sm.execute_partial_exit.assert_not_called()

    def test_fires_when_signal_detected_above_min_r(self):
        """At +4R with climax candle → execute_partial_exit called."""
        mock_alpaca = MagicMock(spec=AlpacaClient)
        mock_db = MagicMock()
        mock_db.get_open_trades.return_value = []
        mock_sm = MagicMock()
        mock_sm.watched_symbols = ['TEST']
        mock_sm.get_watch_snapshot.return_value = {
            'entry_price': 5.0, 'risk_per_share': 0.5,
            'highest_since_entry': 8.0, 'shares': 1000,
            'exhaustion_partial_taken': False,
            'trade_db_id': 42, 'stop_price': 6.0,
            'trail_r': 1.0, 'trailing_active': True,
        }
        mock_sm.execute_partial_exit.return_value = StopExitEvent(
            symbol='TEST', stop_price=0.0, exit_price=7.45,
            shares=500, order_id='partial-001',
            exit_reason='exhaustion_partial', trade_db_id=42,
        )

        # Build bars where last completed bar is a climax
        bars = []
        for i in range(8):
            bars.append({
                'timestamp': datetime.now(timezone.utc),
                'open': 6.0 + i * 0.1,
                'close': 6.1 + i * 0.1,
                'high': 6.2 + i * 0.1,
                'low': 5.9 + i * 0.1,
                'volume': 1000,
            })
        # Climax bar: huge body + volume
        bars.append({
            'timestamp': datetime.now(timezone.utc),
            'open': 6.8, 'close': 7.5, 'high': 7.6, 'low': 6.7,
            'volume': 5000,
        })
        # In-progress bar (will be dropped)
        bars.append({
            'timestamp': datetime.now(timezone.utc),
            'open': 7.5, 'close': 7.5, 'high': 7.6, 'low': 7.4,
            'volume': 100,
        })
        mock_alpaca.get_1min_bars.return_value = pd.DataFrame(bars)
        mock_alpaca.get_order.return_value = {'status': 'filled', 'filled_avg_price': 7.50}

        engine = self._make_engine(mock_alpaca, mock_db, mock_sm)
        engine._check_exhaustion_exits()
        mock_sm.execute_partial_exit.assert_called_once_with(
            symbol='TEST', fraction=0.5, tighter_trail_r=0.5,
        )


# ---------------------------------------------------------------------------
# Combined P&L in _process_stop_monitor_exits
# ---------------------------------------------------------------------------

class TestCombinedPnL:
    """Test that final exit P&L combines partial + remainder."""

    def _make_engine_with_event(self, partial_pnl=None, partial_shares=None):
        """Build engine and simulate a stop monitor exit with/without prior partial."""
        from trading.pattern_detector import BullFlagDetector
        from trading.trade_planner import TradePlanner
        from trading.order_executor import OrderExecutor
        from trading.position_manager import PositionManager

        mock_alpaca = MagicMock(spec=AlpacaClient)
        mock_db = MagicMock()
        mock_pm = MagicMock(spec=PositionManager)

        # Trade record with potential partial exit data
        trade_record = {
            'id': 42,
            'symbol': 'TEST',
            'fill_price': 5.0,
            'shares': 1000,
            'filled_qty': 1000,
            'partial_exit_pnl': partial_pnl,
            'partial_exit_shares': partial_shares,
        }
        mock_db.get_open_trades.return_value = [trade_record]
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 7.0,
        }

        mock_sm = MagicMock()
        # Simulate a trail stop exit event
        exit_event = StopExitEvent(
            symbol='TEST', stop_price=6.5, exit_price=6.45,
            shares=500 if partial_shares else 1000,
            order_id='exit-001',
            exit_reason='trail_stop',
            trade_db_id=42,
        )
        mock_sm.drain_exit_events.return_value = [exit_event]

        with patch('config.Config') as MockConfig:
            MockConfig._load_yaml_only.return_value = {
                'trading': {
                    'trailing_stop': {'enabled': True, 'trail_r': 1.0, 'activate_at_r': 2.0},
                    'skip_fridays': False,
                    'macd_zones': {'enabled': False},
                    'exhaustion_exit': {'enabled': False},
                }
            }
            from trading.trading_engine import TradingEngine
            engine = TradingEngine(
                alpaca_client=mock_alpaca,
                db=mock_db,
                detector=MagicMock(spec=BullFlagDetector),
                planner=MagicMock(spec=TradePlanner),
                executor=MagicMock(spec=OrderExecutor),
                position_manager=mock_pm,
                enabled=True,
                stop_monitor=mock_sm,
            )
        return engine, mock_db, mock_pm

    def test_pnl_without_partial(self):
        """Without partial, P&L = (exit-entry) × shares."""
        engine, mock_db, mock_pm = self._make_engine_with_event()
        engine._process_stop_monitor_exits()

        # exit at $7.0 (filled), entry $5.0, 1000 shares → P&L = $2000
        update_call = mock_db.update_trade.call_args
        assert update_call[0][0] == 42
        pnl = update_call[0][1]['pnl']
        assert pnl == 2000.0

    def test_pnl_with_partial_combines(self):
        """With partial, total = partial_pnl + remainder_pnl."""
        # Partial: sold 500sh at $8.0 → partial_pnl = (8-5)*500 = $1500
        # Remainder: 500sh exit at $7.0 → (7-5)*500 = $1000
        # Total: $2500
        engine, mock_db, mock_pm = self._make_engine_with_event(
            partial_pnl=1500.0, partial_shares=500,
        )
        engine._process_stop_monitor_exits()

        update_call = mock_db.update_trade.call_args
        pnl = update_call[0][1]['pnl']
        assert pnl == 2500.0
        # exit_reason should be prefixed with exhaust+
        assert 'exhaust+' in update_call[0][1]['exit_reason']


# ---------------------------------------------------------------------------
# Database migration
# ---------------------------------------------------------------------------

class TestDatabaseMigration:
    """Test that partial exit columns are added by migration."""

    def test_migration_adds_partial_columns(self, tmp_path):
        """Migration 4 adds 5 partial exit columns."""
        from persistence.database import Database
        db = Database(str(tmp_path / 'test.db'))
        columns = [row[1] for row in db.conn.execute("PRAGMA table_info(trades)").fetchall()]
        assert 'partial_exit_price' in columns
        assert 'partial_exit_shares' in columns
        assert 'partial_exit_pnl' in columns
        assert 'partial_exit_reason' in columns
        assert 'partial_exited_at' in columns

    def test_migration_idempotent(self, tmp_path):
        """Running migration twice doesn't fail."""
        from persistence.database import Database
        db1 = Database(str(tmp_path / 'test.db'))
        db1.conn.close()
        db2 = Database(str(tmp_path / 'test.db'))
        columns = [row[1] for row in db2.conn.execute("PRAGMA table_info(trades)").fetchall()]
        assert 'partial_exit_price' in columns
