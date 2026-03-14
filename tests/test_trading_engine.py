"""
Unit tests for TradingEngine — orchestration of the trading pipeline.

Tests cover:
- Stock qualification handling
- Pattern detection cycle
- Trade execution flow
- Daily stats and summary
- Enable/disable behavior
"""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.pattern_detector import BullFlagDetector, BullFlagPattern
from trading.trade_planner import TradePlanner, TradePlan
from trading.order_executor import OrderExecutor
from trading.position_manager import PositionManager
from trading.trading_engine import TradingEngine
from notifications.telegram_notifier import TelegramNotifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
def mock_detector():
    """Mocked BullFlagDetector."""
    return MagicMock(spec=BullFlagDetector)


@pytest.fixture
def mock_planner():
    """Mocked TradePlanner."""
    return MagicMock(spec=TradePlanner)


@pytest.fixture
def mock_executor():
    """Mocked OrderExecutor."""
    return MagicMock(spec=OrderExecutor)


@pytest.fixture
def mock_position_manager():
    """Mocked PositionManager."""
    return MagicMock(spec=PositionManager)


def _make_pattern(symbol="TEST"):
    """Create a BullFlagPattern."""
    return BullFlagPattern(
        symbol=symbol,
        pole_start_idx=0, pole_end_idx=2,
        flag_start_idx=3, flag_end_idx=4,
        pole_low=4.00, pole_high=4.50,
        pole_height=0.50, pole_gain_pct=12.5,
        flag_low=4.30, flag_high=4.40,
        retracement_pct=40.0, pullback_candle_count=2,
        avg_pole_volume=180000, avg_flag_volume=40000,
        breakout_level=4.40,
    )


def _make_plan(symbol="TEST"):
    """Create a TradePlan."""
    return TradePlan(
        symbol=symbol,
        entry_price=4.40,
        stop_loss_price=4.29,
        take_profit_price=4.90,
        risk_per_share=0.11,
        reward_per_share=0.50,
        risk_reward_ratio=4.5,
        shares=113,
        total_risk=12.43,
        pattern=_make_pattern(symbol),
    )


@pytest.fixture
def engine(mock_alpaca, db, mock_detector, mock_planner, mock_executor, mock_position_manager):
    """TradingEngine with all mocked dependencies."""
    return TradingEngine(
        alpaca_client=mock_alpaca,
        db=db,
        detector=mock_detector,
        planner=mock_planner,
        executor=mock_executor,
        position_manager=mock_position_manager,
        pattern_poll_interval=60,
        enabled=True,
    )


# ===========================================================================
# TESTS
# ===========================================================================

class TestOnStockQualified:
    """Tests for stock qualification handling."""

    def test_adds_symbol_to_qualified(self, engine):
        """Qualified symbol is added to monitoring set."""
        engine.on_stock_qualified("AAPL")
        assert "AAPL" in engine._qualified_symbols

    def test_ignores_when_disabled(self, engine):
        """Does nothing when engine is disabled."""
        engine.enabled = False
        engine.on_stock_qualified("AAPL")
        assert "AAPL" not in engine._qualified_symbols

    def test_ignores_already_traded(self, engine):
        """Skips symbols already traded today."""
        engine._traded_symbols.add("AAPL")
        engine.on_stock_qualified("AAPL")
        assert "AAPL" not in engine._qualified_symbols

    def test_no_duplicate_qualified(self, engine):
        """Same symbol isn't added twice."""
        engine.on_stock_qualified("AAPL")
        engine.on_stock_qualified("AAPL")
        assert len(engine._qualified_symbols) == 1


class TestRunPatternCheck:
    """Tests for pattern detection cycle."""

    def test_returns_none_when_disabled(self, engine):
        """Returns None when engine disabled."""
        engine.enabled = False
        assert engine.run_pattern_check() is None

    def test_returns_none_when_no_qualified(self, engine):
        """Returns None when no qualified symbols."""
        assert engine.run_pattern_check() is None

    def test_full_successful_trade_flow(self, engine, mock_alpaca, mock_detector,
                                        mock_planner, mock_executor, mock_position_manager):
        """Complete flow: qualify → detect_setup → plan → submit buy-stop."""
        # Setup
        bars = pd.DataFrame({
            'open': [4.0], 'high': [4.1], 'low': [3.9],
            'close': [4.05], 'volume': [100000],
        })
        mock_alpaca.get_1min_bars.return_value = bars
        mock_detector.detect_setup.return_value = _make_pattern("AAPL")
        mock_planner.create_plan.return_value = _make_plan("AAPL")
        mock_position_manager.can_open_position.return_value = True
        mock_executor.submit_buy_stop_bracket_order.return_value = {
            'order_id': 'order-123', 'status': 'accepted',
            'symbol': 'AAPL', 'shares': 113,
        }

        engine.on_stock_qualified("AAPL")
        result = engine.run_pattern_check()

        assert result is not None
        assert result['symbol'] == 'AAPL'
        # Symbol should be in pending_orders (not traded until filled)
        assert "AAPL" in engine._pending_orders
        mock_position_manager.mark_traded.assert_called_with("AAPL")

    def test_no_trade_when_no_pattern(self, engine, mock_alpaca, mock_detector):
        """No trade when pattern not detected."""
        bars = pd.DataFrame({'open': [4.0], 'high': [4.1], 'low': [3.9],
                            'close': [4.05], 'volume': [100000]})
        mock_alpaca.get_1min_bars.return_value = bars
        mock_detector.detect_setup.return_value = None

        engine.on_stock_qualified("AAPL")
        result = engine.run_pattern_check()
        assert result is None

    def test_no_trade_when_plan_rejected(self, engine, mock_alpaca, mock_detector, mock_planner):
        """No trade when planner rejects pattern."""
        bars = pd.DataFrame({'open': [4.0], 'high': [4.1], 'low': [3.9],
                            'close': [4.05], 'volume': [100000]})
        mock_alpaca.get_1min_bars.return_value = bars
        mock_detector.detect_setup.return_value = _make_pattern("AAPL")
        mock_planner.create_plan.return_value = None

        engine.on_stock_qualified("AAPL")
        result = engine.run_pattern_check()
        assert result is None

    def test_no_trade_when_position_blocked(self, engine, mock_alpaca, mock_detector,
                                             mock_planner, mock_position_manager):
        """No trade when position manager rejects."""
        bars = pd.DataFrame({'open': [4.0], 'high': [4.1], 'low': [3.9],
                            'close': [4.05], 'volume': [100000]})
        mock_alpaca.get_1min_bars.return_value = bars
        mock_detector.detect_setup.return_value = _make_pattern("AAPL")
        mock_planner.create_plan.return_value = _make_plan("AAPL")
        mock_position_manager.can_open_position.return_value = False

        engine.on_stock_qualified("AAPL")
        result = engine.run_pattern_check()
        assert result is None

    def test_handles_bars_fetch_error(self, engine, mock_alpaca):
        """Handles error when fetching 1-min bars."""
        mock_alpaca.get_1min_bars.side_effect = Exception("API error")

        engine.on_stock_qualified("AAPL")
        result = engine.run_pattern_check()
        assert result is None

    def test_skips_already_traded_symbols(self, engine, mock_alpaca):
        """Doesn't check symbols already traded today."""
        engine.on_stock_qualified("AAPL")
        engine._traded_symbols.add("AAPL")

        result = engine.run_pattern_check()
        assert result is None
        mock_alpaca.get_1min_bars.assert_not_called()


class TestDailyStats:
    """Tests for daily statistics."""

    def test_get_daily_stats_empty(self, engine):
        """Stats are zero when no trades."""
        stats = engine.get_daily_stats()
        assert stats['total_trades'] == 0
        assert stats['gross_pnl'] == 0.0

    def test_patterns_detected_counter(self, engine, mock_alpaca, mock_detector,
                                        mock_planner, mock_position_manager, mock_executor):
        """Patterns detected counter increments on detection."""
        bars = pd.DataFrame({'open': [4.0], 'high': [4.1], 'low': [3.9],
                            'close': [4.05], 'volume': [100000]})
        mock_alpaca.get_1min_bars.return_value = bars
        mock_detector.detect_setup.return_value = _make_pattern("AAPL")
        mock_planner.create_plan.return_value = _make_plan("AAPL")
        mock_position_manager.can_open_position.return_value = True
        mock_executor.submit_buy_stop_bracket_order.return_value = {
            'order_id': 'x', 'status': 'accepted', 'symbol': 'AAPL', 'shares': 100
        }

        engine.on_stock_qualified("AAPL")
        engine.run_pattern_check()

        stats = engine.get_daily_stats()
        assert stats['patterns_detected'] == 1
        # patterns_traded increments on fill, not on order placement
        assert stats['patterns_traded'] == 0


class TestResetDaily:
    """Tests for daily state reset."""

    def test_reset_clears_all_state(self, engine, mock_position_manager):
        """Reset clears qualified, traded, pending, and counters."""
        engine._qualified_symbols.add("AAPL")
        engine._traded_symbols.add("TSLA")
        engine._patterns_detected = 5
        engine._patterns_traded = 2
        engine._pending_orders['AAPL'] = {'order_id': 'x'}

        engine.reset_daily()

        assert len(engine._qualified_symbols) == 0
        assert len(engine._traded_symbols) == 0
        assert engine._patterns_detected == 0
        assert engine._patterns_traded == 0
        assert len(engine._pending_orders) == 0
        mock_position_manager.reset_daily.assert_called_once()
        assert engine._pattern_details == []


class TestNotifierIntegration:
    """Tests for Telegram notifier integration with TradingEngine."""

    def test_engine_with_notifier(self, mock_alpaca, db, mock_detector,
                                   mock_planner, mock_executor, mock_position_manager):
        """Engine accepts and stores notifier."""
        mock_notifier = MagicMock(spec=TelegramNotifier)
        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db,
            detector=mock_detector, planner=mock_planner,
            executor=mock_executor, position_manager=mock_position_manager,
            notifier=mock_notifier, enabled=True,
        )
        assert engine.notifier is mock_notifier

    def test_engine_without_notifier(self, engine):
        """Engine works without notifier."""
        assert engine.notifier is None

    def test_pattern_detected_notifies(self, mock_alpaca, db, mock_detector,
                                        mock_planner, mock_executor, mock_position_manager):
        """Pattern detection triggers notifier.notify_pattern_detected."""
        mock_notifier = MagicMock(spec=TelegramNotifier)
        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db,
            detector=mock_detector, planner=mock_planner,
            executor=mock_executor, position_manager=mock_position_manager,
            notifier=mock_notifier, enabled=True,
        )
        bars = pd.DataFrame({'open': [4.0], 'high': [4.1], 'low': [3.9],
                            'close': [4.05], 'volume': [100000]})
        mock_alpaca.get_1min_bars.return_value = bars
        mock_detector.detect_setup.return_value = _make_pattern("AAPL")
        mock_planner.create_plan.return_value = None  # Plan rejected

        engine.on_stock_qualified("AAPL")
        engine.run_pattern_check()

        mock_notifier.notify_pattern_detected.assert_called_once()

    def test_trade_executed_notifies_order(self, mock_alpaca, db, mock_detector,
                                            mock_planner, mock_executor, mock_position_manager):
        """Successful trade triggers notifier.notify_order_submitted."""
        mock_notifier = MagicMock(spec=TelegramNotifier)
        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db,
            detector=mock_detector, planner=mock_planner,
            executor=mock_executor, position_manager=mock_position_manager,
            notifier=mock_notifier, enabled=True,
        )
        bars = pd.DataFrame({'open': [4.0], 'high': [4.1], 'low': [3.9],
                            'close': [4.05], 'volume': [100000]})
        mock_alpaca.get_1min_bars.return_value = bars
        mock_detector.detect_setup.return_value = _make_pattern("AAPL")
        mock_planner.create_plan.return_value = _make_plan("AAPL")
        mock_position_manager.can_open_position.return_value = True
        mock_executor.submit_buy_stop_bracket_order.return_value = {
            'order_id': 'ord-1', 'status': 'accepted', 'symbol': 'AAPL', 'shares': 113
        }

        engine.on_stock_qualified("AAPL")
        engine.run_pattern_check()

        mock_notifier.notify_pattern_detected.assert_called_once()
        mock_notifier.notify_trade_planned.assert_called_once()
        mock_notifier.notify_order_submitted.assert_called_once()

    def test_generate_daily_report(self, engine):
        """generate_daily_report creates properly structured dict."""
        engine._qualified_symbols = {"AAPL", "TSLA"}
        engine._patterns_detected = 1
        engine._pattern_details = [{'symbol': 'AAPL', 'pole_gain_pct': 5.0}]

        report = engine.generate_daily_report(
            premarket_gaps=[{'symbol': 'AAPL', 'gap_pct': 10.0}],
            qualified_stocks=[{'symbol': 'AAPL'}],
            universe_size=100,
        )

        assert report['universe_size'] == 100
        assert report['patterns_detected'] == 1
        assert len(report['premarket_gaps']) == 1
        assert len(report['qualified_stocks']) == 1

    def test_send_daily_report_with_notifier(self, mock_alpaca, db, mock_detector,
                                              mock_planner, mock_executor, mock_position_manager):
        """send_daily_report calls notifier.send_daily_report."""
        mock_notifier = MagicMock(spec=TelegramNotifier)
        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db,
            detector=mock_detector, planner=mock_planner,
            executor=mock_executor, position_manager=mock_position_manager,
            notifier=mock_notifier, enabled=True,
        )
        engine.send_daily_report(universe_size=100)
        mock_notifier.send_daily_report.assert_called_once()

    def test_send_daily_report_without_notifier(self, engine):
        """send_daily_report is a no-op without notifier."""
        engine.send_daily_report(universe_size=100)  # Should not crash
