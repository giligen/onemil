"""
Unit tests for PositionManager — risk limit enforcement.

Tests cover:
- Max positions limit
- Daily loss limit
- Duplicate symbol prevention
- Market close proximity check
- Midday dead zone filter (11:30-14:00 ET)
"""

import pytest
from datetime import datetime, date
from unittest.mock import MagicMock, patch

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.position_manager import PositionManager


@pytest.fixture
def db(tmp_path):
    """Real database with temp file."""
    database = Database(db_path=str(tmp_path / "test.db"))
    yield database
    database.close()


@pytest.fixture
def mock_alpaca():
    """Mocked AlpacaClient."""
    return MagicMock(spec=AlpacaClient)


@pytest.fixture
def manager(mock_alpaca, db):
    """PositionManager with default limits."""
    return PositionManager(
        alpaca_client=mock_alpaca,
        db=db,
        max_positions=3,
        daily_loss_limit=-100.0,
        stop_trading_before_close_min=15,
    )


def _save_open_trade(db, symbol, entry_price=4.40):
    """Helper to save an open trade record to DB."""
    from datetime import timezone
    now = datetime.now(timezone.utc)
    db.save_trade({
        'trade_date': date.today().isoformat(),
        'symbol': symbol,
        'side': 'buy',
        'entry_price': entry_price,
        'stop_loss_price': entry_price - 0.20,
        'take_profit_price': entry_price + 0.50,
        'shares': 100,
        'risk_per_share': 0.20,
        'total_risk': 20.0,
        'risk_reward_ratio': 2.5,
        'order_id': f'order-{symbol}',
        'order_status': 'filled',
        'fill_price': entry_price,
        'filled_at': now,
        'exit_price': None,
        'exit_reason': None,
        'exited_at': None,
        'pnl': None,
        'pnl_pct': None,
        'pattern_data': '{}',
        'created_at': now,
        'updated_at': now,
    })


class TestCanOpenPosition:
    """Tests for position opening checks."""

    @patch('trading.position_manager.datetime')
    def test_allows_when_all_limits_ok(self, mock_dt, manager):
        """Allows position when no limits are breached."""
        # Mock time to be mid-day (not near close)
        mock_now = MagicMock()
        mock_now.hour = 10
        mock_now.minute = 30
        mock_dt.now.return_value = mock_now

        assert manager.can_open_position("AAPL") is True

    @patch('trading.position_manager.datetime')
    def test_rejects_near_close(self, mock_dt, manager):
        """Rejects when within 15 min of market close."""
        mock_now = MagicMock()
        mock_now.hour = 15
        mock_now.minute = 50  # 10 min to close
        mock_dt.now.return_value = mock_now

        assert manager.can_open_position("AAPL") is False

    @patch('trading.position_manager.datetime')
    def test_rejects_duplicate_symbol(self, mock_dt, manager):
        """Rejects symbol already traded today."""
        mock_now = MagicMock()
        mock_now.hour = 10
        mock_now.minute = 30
        mock_dt.now.return_value = mock_now

        manager.mark_traded("AAPL")
        assert manager.can_open_position("AAPL") is False

    @patch('trading.position_manager.datetime')
    def test_rejects_max_positions_reached(self, mock_dt, manager, db):
        """Rejects when max concurrent positions reached."""
        mock_now = MagicMock()
        mock_now.hour = 10
        mock_now.minute = 30
        mock_dt.now.return_value = mock_now

        # Fill up positions
        _save_open_trade(db, "POS1")
        _save_open_trade(db, "POS2")
        _save_open_trade(db, "POS3")

        assert manager.can_open_position("NEW") is False

    @patch('trading.position_manager.datetime')
    def test_rejects_daily_loss_limit(self, mock_dt, manager, db):
        """Rejects when daily loss limit breached."""
        from datetime import timezone
        mock_now = MagicMock()
        mock_now.hour = 10
        mock_now.minute = 30
        mock_dt.now.return_value = mock_now

        # Save a closed trade with big loss
        now = datetime.now(timezone.utc)
        db.save_trade({
            'trade_date': date.today().isoformat(),
            'symbol': 'LOSER',
            'side': 'buy',
            'entry_price': 5.00,
            'stop_loss_price': 4.80,
            'take_profit_price': 5.40,
            'shares': 500,
            'risk_per_share': 0.20,
            'total_risk': 100.0,
            'risk_reward_ratio': 2.0,
            'order_id': 'order-loss',
            'order_status': 'filled',
            'fill_price': 5.00,
            'filled_at': now,
            'exit_price': 4.80,
            'exit_reason': 'stop_loss',
            'exited_at': now,
            'pnl': -100.0,
            'pnl_pct': -4.0,
            'pattern_data': '{}',
            'created_at': now,
            'updated_at': now,
        })

        assert manager.can_open_position("NEW") is False

    @patch('trading.position_manager.datetime')
    def test_allows_different_symbol(self, mock_dt, manager):
        """Allows different symbol even if one is already traded."""
        mock_now = MagicMock()
        mock_now.hour = 10
        mock_now.minute = 30
        mock_dt.now.return_value = mock_now

        manager.mark_traded("AAPL")
        assert manager.can_open_position("TSLA") is True

    @patch('trading.position_manager.datetime')
    def test_rejects_symbol_with_open_position(self, mock_dt, manager, db):
        """Rejects when symbol already has an open position."""
        mock_now = MagicMock()
        mock_now.hour = 10
        mock_now.minute = 30
        mock_dt.now.return_value = mock_now

        _save_open_trade(db, "AAPL")
        assert manager.can_open_position("AAPL") is False


class TestMarkTraded:
    """Tests for mark_traded."""

    def test_mark_traded_adds_to_set(self, manager):
        """Marking a symbol adds it to traded set."""
        manager.mark_traded("AAPL")
        assert "AAPL" in manager._traded_symbols

    def test_mark_traded_idempotent(self, manager):
        """Marking same symbol twice is fine."""
        manager.mark_traded("AAPL")
        manager.mark_traded("AAPL")
        assert "AAPL" in manager._traded_symbols


class TestResetDaily:
    """Tests for daily reset."""

    def test_reset_clears_traded_symbols(self, manager):
        """Reset clears the traded symbols set."""
        manager.mark_traded("AAPL")
        manager.mark_traded("TSLA")
        manager.reset_daily()
        assert len(manager._traded_symbols) == 0


class TestGetOpenPositions:
    """Tests for getting open positions from Alpaca."""

    def test_returns_positions(self, manager, mock_alpaca):
        """Returns positions from Alpaca."""
        mock_alpaca.get_open_positions.return_value = [
            {'symbol': 'AAPL', 'qty': 100},
        ]
        positions = manager.get_open_positions()
        assert len(positions) == 1
        assert positions[0]['symbol'] == 'AAPL'

    def test_handles_api_error(self, manager, mock_alpaca):
        """Returns empty list on API error."""
        mock_alpaca.get_open_positions.side_effect = Exception("API down")
        positions = manager.get_open_positions()
        assert positions == []


class TestMiddayFilter:
    """Tests for midday dead zone filter (11:30-14:00 ET)."""

    @patch('trading.position_manager.datetime')
    def test_rejects_during_midday(self, mock_dt, mock_alpaca, db):
        """Rejects position during 11:30-14:00 ET dead zone."""
        manager = PositionManager(
            alpaca_client=mock_alpaca, db=db, skip_midday=True
        )
        mock_now = MagicMock()
        mock_now.hour = 12
        mock_now.minute = 30
        mock_dt.now.return_value = mock_now

        assert manager.can_open_position("AAPL") is False

    @patch('trading.position_manager.datetime')
    def test_allows_before_midday(self, mock_dt, mock_alpaca, db):
        """Allows position at 11:29 (just before dead zone)."""
        manager = PositionManager(
            alpaca_client=mock_alpaca, db=db, skip_midday=True
        )
        mock_now = MagicMock()
        mock_now.hour = 11
        mock_now.minute = 29
        mock_dt.now.return_value = mock_now

        assert manager.can_open_position("AAPL") is True

    @patch('trading.position_manager.datetime')
    def test_allows_after_midday(self, mock_dt, mock_alpaca, db):
        """Allows position at 14:00 (end of dead zone)."""
        manager = PositionManager(
            alpaca_client=mock_alpaca, db=db, skip_midday=True
        )
        mock_now = MagicMock()
        mock_now.hour = 14
        mock_now.minute = 0
        mock_dt.now.return_value = mock_now

        assert manager.can_open_position("AAPL") is True

    @patch('trading.position_manager.datetime')
    def test_rejects_at_midday_start_boundary(self, mock_dt, mock_alpaca, db):
        """Rejects at exactly 11:30 ET (start of dead zone)."""
        manager = PositionManager(
            alpaca_client=mock_alpaca, db=db, skip_midday=True
        )
        mock_now = MagicMock()
        mock_now.hour = 11
        mock_now.minute = 30
        mock_dt.now.return_value = mock_now

        assert manager.can_open_position("AAPL") is False

    @patch('trading.position_manager.datetime')
    def test_rejects_at_1359(self, mock_dt, mock_alpaca, db):
        """Rejects at 13:59 ET (last minute of dead zone)."""
        manager = PositionManager(
            alpaca_client=mock_alpaca, db=db, skip_midday=True
        )
        mock_now = MagicMock()
        mock_now.hour = 13
        mock_now.minute = 59
        mock_dt.now.return_value = mock_now

        assert manager.can_open_position("AAPL") is False

    @patch('trading.position_manager.datetime')
    def test_allows_midday_when_disabled(self, mock_dt, mock_alpaca, db):
        """Allows midday position when skip_midday=False."""
        manager = PositionManager(
            alpaca_client=mock_alpaca, db=db, skip_midday=False
        )
        mock_now = MagicMock()
        mock_now.hour = 12
        mock_now.minute = 30
        mock_dt.now.return_value = mock_now

        assert manager.can_open_position("AAPL") is True

    @patch('trading.position_manager.datetime')
    def test_morning_trading_allowed(self, mock_dt, mock_alpaca, db):
        """Morning hours (9:30-11:30 ET) are fully open."""
        manager = PositionManager(
            alpaca_client=mock_alpaca, db=db, skip_midday=True
        )
        mock_now = MagicMock()
        mock_now.hour = 10
        mock_now.minute = 15
        mock_dt.now.return_value = mock_now

        assert manager.can_open_position("AAPL") is True

    @patch('trading.position_manager.datetime')
    def test_afternoon_trading_allowed(self, mock_dt, mock_alpaca, db):
        """Afternoon hours (14:00-15:45 ET) are fully open."""
        manager = PositionManager(
            alpaca_client=mock_alpaca, db=db, skip_midday=True
        )
        mock_now = MagicMock()
        mock_now.hour = 15
        mock_now.minute = 0
        mock_dt.now.return_value = mock_now

        assert manager.can_open_position("AAPL") is True


class TestCircuitBreaker:
    """Tests for circuit breaker based on drawdown from peak P&L."""

    def test_record_trade_pnl_triggers_cb(self, mock_alpaca, db):
        """3 losses totaling >$3K drawdown triggers circuit breaker."""
        manager = PositionManager(
            alpaca_client=mock_alpaca, db=db,
            circuit_breaker_dd=3000.0, circuit_breaker_pause=2,
        )
        # Simulate losses: -1000, -1000, -1500 = -3500 total
        manager.record_trade_pnl(-1000.0)
        assert manager._cb_skips_remaining == 0  # Not yet triggered (dd=1000)
        manager.record_trade_pnl(-1000.0)
        assert manager._cb_skips_remaining == 0  # dd=2000, still under
        manager.record_trade_pnl(-1500.0)
        assert manager._cb_skips_remaining == 2  # dd=3500, triggered

    def test_record_trade_pnl_peak_tracking(self, mock_alpaca, db):
        """Peak tracks cumulative high, dd measured from peak."""
        manager = PositionManager(
            alpaca_client=mock_alpaca, db=db,
            circuit_breaker_dd=3000.0, circuit_breaker_pause=2,
        )
        # Win, then lose from peak
        manager.record_trade_pnl(2000.0)  # cum=2000, peak=2000
        assert manager._peak_pnl == 2000.0
        manager.record_trade_pnl(-2000.0)  # cum=0, dd=2000
        assert manager._cb_skips_remaining == 0  # Under threshold
        manager.record_trade_pnl(-1500.0)  # cum=-1500, dd=3500
        assert manager._cb_skips_remaining == 2  # Triggered

    @patch('trading.position_manager.datetime')
    def test_cb_blocks_trades(self, mock_dt, mock_alpaca, db):
        """When skips > 0, can_open_position returns False and decrements."""
        manager = PositionManager(
            alpaca_client=mock_alpaca, db=db,
            circuit_breaker_dd=3000.0, circuit_breaker_pause=2,
        )
        mock_now = MagicMock()
        mock_now.hour = 10
        mock_now.minute = 30
        mock_dt.now.return_value = mock_now

        manager._cb_skips_remaining = 2
        assert manager.can_open_position("AAPL") is False
        assert manager._cb_skips_remaining == 1
        assert manager.can_open_position("TSLA") is False
        assert manager._cb_skips_remaining == 0
        # Now should allow (all other limits OK)
        assert manager.can_open_position("GOOG") is True

    def test_cb_resets_daily(self, mock_alpaca, db):
        """reset_daily clears all CB state."""
        manager = PositionManager(
            alpaca_client=mock_alpaca, db=db,
            circuit_breaker_dd=3000.0, circuit_breaker_pause=2,
        )
        manager._cumulative_pnl = -5000.0
        manager._peak_pnl = 1000.0
        manager._cb_skips_remaining = 2

        manager.reset_daily()

        assert manager._cumulative_pnl == 0.0
        assert manager._peak_pnl == 0.0
        assert manager._cb_skips_remaining == 0
