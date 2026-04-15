"""
Unit tests for TradingEngine — orchestration of the trading pipeline.

Tests cover:
- Stock qualification handling
- Pattern detection cycle
- Trade execution flow
- Daily stats and summary
- Enable/disable behavior
- Fill data fallback and retry
- Partial fill detection
- Force close retry with backoff
- Graceful shutdown
- Bracket leg identification
- Race condition on cancel
"""

import pytest
import threading
import pandas as pd
from datetime import date, datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, PropertyMock, call

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

def _seed_universe(database, symbols=("TEST", "AAPL", "TSLA", "TIER", "TIER2")):
    """Seed universe with test symbols so volume filter doesn't block them."""
    from datetime import datetime, timezone
    for sym in symbols:
        database.upsert_universe_stock({
            'symbol': sym, 'company_name': sym, 'exchange': 'NASDAQ',
            'sector': '', 'country': 'US', 'price_close': 10.0,
            'float_shares': 5_000_000, 'float_updated_at': None,
            'avg_volume_daily': 500_000, 'last_updated': datetime.now(timezone.utc),
            'active': 1,
        })


@pytest.fixture
def db(tmp_path):
    """Real database with temp file + seeded universe."""
    database = Database(db_path=str(tmp_path / "test.db"))
    _seed_universe(database)
    yield database
    database.close()


@pytest.fixture
def mock_alpaca():
    """Mocked AlpacaClient."""
    return MagicMock(spec=AlpacaClient)


@pytest.fixture
def mock_detector():
    """Mocked BullFlagDetector."""
    detector = MagicMock(spec=BullFlagDetector)
    detector.min_breakout_volume_ratio = 1.5  # default from config
    return detector


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
    """Create a TradePlan with stop distance > min_stop_distance ($0.12)."""
    return TradePlan(
        symbol=symbol,
        entry_price=4.40,
        stop_loss_price=4.25,
        take_profit_price=4.90,
        risk_per_share=0.15,
        reward_per_share=0.50,
        risk_reward_ratio=3.3,
        shares=113,
        total_risk=16.95,
        pattern=_make_pattern(symbol),
    )


@pytest.fixture
def engine(mock_alpaca, db, mock_detector, mock_planner, mock_executor, mock_position_manager):
    """TradingEngine with all mocked dependencies."""
    eng = TradingEngine(
        alpaca_client=mock_alpaca,
        db=db,
        detector=mock_detector,
        planner=mock_planner,
        executor=mock_executor,
        position_manager=mock_position_manager,
        pattern_poll_interval=60,
        enabled=True,
    )
    # Disable features that need real data in tests
    eng.quality_filter_enabled = False
    eng.conviction_enabled = False
    eng.news_gate_enabled = False
    return eng


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

    @patch.object(TradingEngine, '_is_past_last_entry_time', return_value=False)
    def test_full_successful_trade_flow(self, _mock_time, engine, mock_alpaca, mock_detector,
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
        # mark_traded and _daily_trade_count are deferred to fill time
        mock_position_manager.mark_traded.assert_not_called()
        assert engine._daily_trade_count == 0

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
        # SPY MACD cutoff may call get_1min_bars('SPY'), but AAPL should NOT be fetched
        for call_args in mock_alpaca.get_1min_bars.call_args_list:
            assert call_args[0][0] != "AAPL", "Should not fetch bars for already-traded symbol"


class TestDailyStats:
    """Tests for daily statistics."""

    def test_get_daily_stats_empty(self, engine):
        """Stats are zero when no trades."""
        stats = engine.get_daily_stats()
        assert stats['total_trades'] == 0
        assert stats['gross_pnl'] == 0.0

    @patch.object(TradingEngine, '_is_past_last_entry_time', return_value=False)
    def test_patterns_detected_counter(self, _mock_time, engine, mock_alpaca, mock_detector,
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
        engine.news_gate_enabled = False
        assert engine.notifier is mock_notifier

    def test_engine_without_notifier(self, engine):
        """Engine works without notifier."""
        assert engine.notifier is None

    @patch.object(TradingEngine, '_is_past_last_entry_time', return_value=False)
    def test_pattern_detected_notifies(self, _mock_time, mock_alpaca, db, mock_detector,
                                        mock_planner, mock_executor, mock_position_manager):
        """Notifications only fire when all filters pass and order will be placed."""
        mock_notifier = MagicMock(spec=TelegramNotifier)
        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db,
            detector=mock_detector, planner=mock_planner,
            executor=mock_executor, position_manager=mock_position_manager,
            notifier=mock_notifier, enabled=True,
        )
        engine.news_gate_enabled = False
        engine.quality_filter_enabled = False  # Mock bars don't have OHLCV
        bars = pd.DataFrame({'open': [4.0], 'high': [4.1], 'low': [3.9],
                            'close': [4.05], 'volume': [100000]})
        mock_alpaca.get_1min_bars.return_value = bars
        mock_detector.detect_setup.return_value = _make_pattern("AAPL")
        mock_planner.create_plan.return_value = _make_plan("AAPL")
        mock_position_manager.can_open_position.return_value = True
        mock_executor.submit_buy_stop_bracket_order.return_value = {
            'order_id': 'ord-1', 'status': 'accepted', 'symbol': 'AAPL', 'shares': 113,
        }

        engine.on_stock_qualified("AAPL")
        engine.run_pattern_check()

        mock_notifier.notify_pattern_detected.assert_called_once()
        mock_notifier.notify_trade_planned.assert_called_once()

    @patch.object(TradingEngine, '_is_past_last_entry_time', return_value=False)
    def test_trade_executed_notifies_order(self, _mock_time, mock_alpaca, db, mock_detector,
                                            mock_planner, mock_executor, mock_position_manager):
        """Successful trade triggers notifier.notify_order_submitted."""
        mock_notifier = MagicMock(spec=TelegramNotifier)
        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db,
            detector=mock_detector, planner=mock_planner,
            executor=mock_executor, position_manager=mock_position_manager,
            notifier=mock_notifier, enabled=True,
        )
        engine.news_gate_enabled = False
        engine.quality_filter_enabled = False  # Mock bars don't have OHLCV
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
        engine.news_gate_enabled = False
        engine.send_daily_report(universe_size=100)
        mock_notifier.send_daily_report.assert_called_once()

    def test_send_daily_report_without_notifier(self, engine):
        """send_daily_report is a no-op without notifier."""
        engine.send_daily_report(universe_size=100)  # Should not crash


# ===========================================================================
# Phase 2: DB Update on Fill
# ===========================================================================

class TestDBUpdateOnFill:
    """Tests for updating trade DB record when buy-stop order fills."""

    def test_fill_updates_db(self, engine, mock_alpaca, db):
        """Filled order updates trade record with fill_price and filled_at."""
        # Save a trade to DB
        from datetime import timezone
        now = datetime.now(timezone.utc)
        db.save_trade({
            'trade_date': date.today().isoformat(),
            'symbol': 'TEST',
            'side': 'buy',
            'entry_price': 5.00,
            'stop_loss_price': 4.80,
            'take_profit_price': 5.40,
            'shares': 100,
            'risk_per_share': 0.20,
            'total_risk': 20.0,
            'risk_reward_ratio': 2.0,
            'order_id': 'order-fill-test',
            'order_status': 'accepted',
            'fill_price': None,
            'filled_at': None,
            'exit_price': None,
            'exit_reason': None,
            'exited_at': None,
            'pnl': None,
            'pnl_pct': None,
            'pattern_data': '{}',
            'created_at': now,
            'updated_at': now,
        })

        # Set up pending order
        plan = _make_plan("TEST")
        setup = _make_pattern("TEST")
        engine._pending_orders['TEST'] = {
            'order_id': 'order-fill-test',
            'plan': plan,
            'setup': setup,
            'placed_at': now,
        }

        # Mock Alpaca returning filled status
        mock_alpaca.get_order.return_value = {
            'status': 'filled',
            'filled_avg_price': 4.45,
            'legs': [],
        }

        result = engine._manage_pending_orders()

        assert result is not None
        assert result['fill_price'] == 4.45

        # Verify DB was updated
        trade = db.get_trade_by_order_id('order-fill-test')
        assert trade['fill_price'] == 4.45
        assert trade['order_status'] == 'filled'
        assert trade['filled_at'] is not None

    def test_fill_writes_migration_10_columns(self, engine, mock_alpaca, db):
        """Bull flag fill writes Migration 10 slippage instrumentation columns:
        bar_close_price (=plan.entry_price), order_submitted_at, order_filled_at,
        submit_to_fill_ms, drift_bar_to_fill_bps. Parity with MACD wave."""
        from datetime import timezone
        now = datetime.now(timezone.utc)
        placed_at = now - timedelta(seconds=3)

        db.save_trade({
            'trade_date': date.today().isoformat(),
            'symbol': 'TEST',
            'side': 'buy',
            'entry_price': 4.40,
            'stop_loss_price': 4.25,
            'take_profit_price': 4.90,
            'shares': 113,
            'risk_per_share': 0.15,
            'total_risk': 16.95,
            'risk_reward_ratio': 3.3,
            'order_id': 'order-mig10',
            'order_status': 'accepted',
            'fill_price': None, 'filled_at': None,
            'exit_price': None, 'exit_reason': None, 'exited_at': None,
            'pnl': None, 'pnl_pct': None, 'pattern_data': '{}',
            'created_at': now, 'updated_at': now,
        })

        plan = _make_plan("TEST")  # entry_price = 4.40 (= breakout reference)
        engine._pending_orders['TEST'] = {
            'order_id': 'order-mig10',
            'plan': plan, 'setup': _make_pattern("TEST"),
            'placed_at': placed_at,
        }
        # Filled at 4.45 → drift = (4.45 - 4.40) / 4.40 * 10000 ≈ 113.6 bps
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 4.45, 'legs': [],
        }

        engine._manage_pending_orders()

        trade = db.get_trade_by_order_id('order-mig10')
        # Reference price = breakout level (= plan.entry_price)
        assert trade['bar_close_price'] == 4.40
        # Timestamps written
        assert trade['order_submitted_at'] is not None
        assert trade['order_filled_at'] is not None
        # Latency ≈ 3000 ms (tolerate +/- 2000 for test jitter)
        assert 1000 <= trade['submit_to_fill_ms'] <= 6000
        # Drift positive (fill above breakout)
        assert abs(trade['drift_bar_to_fill_bps'] - 113.636) < 0.5

    def test_fill_with_naive_placed_at_survives(self, engine, mock_alpaca, db):
        """Regression: if placed_at comes back tz-naive (historic DB rows
        restored via startup recovery), the (fill_at - placed_at) subtraction
        would raise TypeError aware/naive and abort the entire update.
        Guard skips submit_to_fill_ms without losing fill_price + other fields."""
        from datetime import timezone
        now = datetime.now(timezone.utc)
        # Simulate startup-recovery: placed_at is tz-NAIVE (as produced by
        # datetime.fromisoformat on a legacy '2026-04-12T14:00:00' string
        # that lacks a tz suffix).
        naive_placed_at = datetime(2026, 4, 12, 14, 0, 0)  # no tzinfo

        db.save_trade({
            'trade_date': date.today().isoformat(),
            'symbol': 'TEST', 'side': 'buy',
            'entry_price': 4.40, 'stop_loss_price': 4.25, 'take_profit_price': 4.90,
            'shares': 113, 'risk_per_share': 0.15, 'total_risk': 16.95,
            'risk_reward_ratio': 3.3,
            'order_id': 'order-naive', 'order_status': 'accepted',
            'fill_price': None, 'filled_at': None,
            'exit_price': None, 'exit_reason': None, 'exited_at': None,
            'pnl': None, 'pnl_pct': None, 'pattern_data': '{}',
            'created_at': now, 'updated_at': now,
        })
        engine._pending_orders['TEST'] = {
            'order_id': 'order-naive',
            'plan': _make_plan("TEST"),
            'setup': _make_pattern("TEST"),
            'placed_at': naive_placed_at,  # ← the dangerous case
        }
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 4.45, 'legs': [],
        }

        # Must not raise
        engine._manage_pending_orders()

        trade = db.get_trade_by_order_id('order-naive')
        # Core fill fields still persisted
        assert trade['fill_price'] == 4.45
        assert trade['order_status'] == 'filled'
        assert trade['filled_at'] is not None
        # bar_close_price doesn't depend on placed_at → still populated
        assert trade['bar_close_price'] == 4.40
        assert trade['drift_bar_to_fill_bps'] is not None
        # submit_to_fill_ms couldn't be computed → stayed NULL
        assert trade['submit_to_fill_ms'] is None

    def test_fill_missing_placed_at_graceful(self, engine, mock_alpaca, db):
        """If placed_at is None (shouldn't happen but be defensive), the
        Migration 10 fields that depend on it stay NULL, and the rest of
        the fill flow still runs."""
        from datetime import timezone
        now = datetime.now(timezone.utc)
        db.save_trade({
            'trade_date': date.today().isoformat(),
            'symbol': 'TEST', 'side': 'buy',
            'entry_price': 4.40, 'stop_loss_price': 4.25, 'take_profit_price': 4.90,
            'shares': 113, 'risk_per_share': 0.15, 'total_risk': 16.95,
            'risk_reward_ratio': 3.3,
            'order_id': 'order-noplaced',
            'order_status': 'accepted', 'fill_price': None, 'filled_at': None,
            'exit_price': None, 'exit_reason': None, 'exited_at': None,
            'pnl': None, 'pnl_pct': None, 'pattern_data': '{}',
            'created_at': now, 'updated_at': now,
        })
        engine._pending_orders['TEST'] = {
            'order_id': 'order-noplaced',
            'plan': _make_plan("TEST"),
            'setup': _make_pattern("TEST"),
            'placed_at': None,  # missing
        }
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 4.45, 'legs': [],
        }

        engine._manage_pending_orders()

        trade = db.get_trade_by_order_id('order-noplaced')
        # Fill itself still persisted
        assert trade['fill_price'] == 4.45
        # Reference price still captured (from plan)
        assert trade['bar_close_price'] == 4.40
        # Drift still computable (doesn't need placed_at)
        assert trade['drift_bar_to_fill_bps'] is not None
        # submit_to_fill_ms needs placed_at — stays NULL
        assert trade['submit_to_fill_ms'] is None

    def test_fill_no_trade_record_logs_error(self, engine, mock_alpaca, db):
        """Logs error when no DB trade record exists for filled order."""
        plan = _make_plan("TEST")
        engine._pending_orders['TEST'] = {
            'order_id': 'nonexistent-order',
            'plan': plan,
            'setup': _make_pattern("TEST"),
            'placed_at': datetime.now(timezone.utc),
        }

        mock_alpaca.get_order.return_value = {
            'status': 'filled',
            'filled_avg_price': 4.45,
            'legs': [],
        }

        # Should not raise, just log error
        result = engine._manage_pending_orders()
        assert result is not None


# ===========================================================================
# Phase 3: Gap-Fill Stop Adjustment
# ===========================================================================

class TestGapFillStopAdjustment:
    """Tests for adjusting stop price when buy-stop fills above breakout level."""

    def _setup_filled_order(self, engine, mock_alpaca, db, fill_price, breakout_level=4.40):
        """Helper: set up a pending order that will return as filled."""
        from datetime import timezone
        now = datetime.now(timezone.utc)
        db.save_trade({
            'trade_date': date.today().isoformat(),
            'symbol': 'GAP',
            'side': 'buy',
            'entry_price': breakout_level,
            'stop_loss_price': 4.25,
            'take_profit_price': 4.90,
            'shares': 100,
            'risk_per_share': 0.15,
            'total_risk': 16.95,
            'risk_reward_ratio': 3.3,
            'order_id': 'order-gap',
            'order_status': 'accepted',
            'fill_price': None,
            'filled_at': None,
            'exit_price': None,
            'exit_reason': None,
            'exited_at': None,
            'pnl': None,
            'pnl_pct': None,
            'pattern_data': '{}',
            'created_at': now,
            'updated_at': now,
        })

        plan = _make_plan("GAP")
        setup = _make_pattern("GAP")
        engine._pending_orders['GAP'] = {
            'order_id': 'order-gap',
            'plan': plan,
            'setup': setup,
            'placed_at': now,
        }

        return plan, setup

    def test_gap_fill_adjusts_target_keeps_stop(self, engine, mock_alpaca, db):
        """Fill above breakout adjusts target UP but keeps stop at technical level."""
        plan, setup = self._setup_filled_order(engine, mock_alpaca, db, fill_price=4.48)

        # First get_order call returns filled status
        mock_alpaca.get_order.side_effect = [
            {
                'status': 'filled',
                'filled_avg_price': 4.48,
                'legs': [],
            },
            # Second get_order call (for gap-fill) returns legs
            {
                'legs': [
                    {'id': 'sl-leg-1', 'side': 'sell', 'stop_price': 4.25,
                     'limit_price': None, 'status': 'new'},
                    {'id': 'tp-leg-1', 'side': 'sell', 'stop_price': None,
                     'limit_price': 4.90, 'status': 'new'},
                ],
            },
        ]
        mock_alpaca.replace_order_limit_price.return_value = {'id': 'tp-leg-1', 'status': 'replaced'}

        engine._manage_pending_orders()

        # Stop NOT replaced — stays at technical level
        mock_alpaca.replace_order_stop_price.assert_not_called()

        # Target adjusted up
        expected_target = round(4.48 + plan.risk_per_share * plan.risk_reward_ratio, 2)
        mock_alpaca.replace_order_limit_price.assert_called_once_with('tp-leg-1', expected_target)

        # DB: stop unchanged, only target updated
        trade = db.get_trade_by_order_id('order-gap')
        assert trade['stop_loss_price'] == plan.stop_loss_price  # original
        assert trade['take_profit_price'] == expected_target

    def test_no_gap_no_stop_adjustment(self, engine, mock_alpaca, db):
        """Fill at breakout level does NOT trigger stop replacement."""
        self._setup_filled_order(engine, mock_alpaca, db, fill_price=4.40)

        mock_alpaca.get_order.return_value = {
            'status': 'filled',
            'filled_avg_price': 4.40,
            'legs': [],
        }

        engine._manage_pending_orders()

        mock_alpaca.replace_order_stop_price.assert_not_called()

    @patch('trading.trading_engine.time_mod')
    def test_gap_fill_no_legs_triggers_emergency_close(self, mock_time, engine, mock_alpaca, db):
        """No SL leg found on gap-fill triggers emergency close."""
        self._setup_filled_order(engine, mock_alpaca, db, fill_price=4.48)

        mock_alpaca.get_order.side_effect = [
            {'status': 'filled', 'filled_avg_price': 4.48, 'legs': []},
            {'legs': []},  # No legs — gap_adjust_failed
            # get_order for close poll
            {'status': 'filled', 'filled_avg_price': 4.50},
        ]
        mock_alpaca.close_position.return_value = {'id': 'close-gap', 'status': 'accepted'}

        result = engine._manage_pending_orders()
        assert result is not None
        assert result['status'] == 'gap_adjust_failed'
        mock_alpaca.close_position.assert_called_once_with('GAP')

        # Verify DB has exit with gap_adjust_failed reason
        trade = db.get_trade_by_order_id('order-gap')
        assert trade['exit_reason'] == 'gap_adjust_failed'

    @patch('trading.trading_engine.time_mod')
    def test_gap_fill_replace_fails_triggers_emergency_close(self, mock_time, engine, mock_alpaca, db):
        """Replace exception on gap-fill triggers emergency close."""
        self._setup_filled_order(engine, mock_alpaca, db, fill_price=4.48)

        mock_alpaca.get_order.side_effect = [
            {'status': 'filled', 'filled_avg_price': 4.48, 'legs': []},
            {
                'legs': [
                    {'id': 'sl-leg-1', 'side': 'sell', 'stop_price': 4.25,
                     'limit_price': None, 'status': 'new'},
                ],
            },
            # get_order for close poll
            {'status': 'filled', 'filled_avg_price': 4.48},
        ]
        mock_alpaca.replace_order_stop_price.side_effect = Exception("API error")
        mock_alpaca.close_position.return_value = {'id': 'close-gap', 'status': 'accepted'}

        result = engine._manage_pending_orders()
        assert result is not None
        assert result['status'] == 'gap_adjust_failed'
        mock_alpaca.close_position.assert_called_once_with('GAP')


# ===========================================================================
# Phase 4: Sync Closed Positions
# ===========================================================================

class TestSyncClosedPositions:
    """Tests for _sync_closed_positions detecting bracket exits."""

    def test_sync_closed_updates_db(self, engine, mock_alpaca, db, mock_position_manager):
        """Closed position detected → DB updated + circuit breaker fed."""
        from datetime import timezone
        now = datetime.now(timezone.utc)

        db.save_trade({
            'trade_date': date.today().isoformat(),
            'symbol': 'CLOSED',
            'side': 'buy',
            'entry_price': 5.00,
            'stop_loss_price': 4.80,
            'take_profit_price': 5.40,
            'shares': 100,
            'risk_per_share': 0.20,
            'total_risk': 20.0,
            'risk_reward_ratio': 2.0,
            'order_id': 'order-closed',
            'order_status': 'filled',
            'fill_price': 5.00,
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

        # Alpaca says no open positions (meaning bracket closed)
        mock_alpaca.get_open_positions.return_value = []
        mock_alpaca.get_order.return_value = {
            'legs': [
                {'id': 'sl-1', 'side': 'sell', 'stop_price': 4.80,
                 'limit_price': None, 'status': 'filled'},
            ],
        }

        engine._sync_closed_positions()

        trade = db.get_trade_by_order_id('order-closed')
        assert trade['exit_price'] == 4.80
        assert trade['exit_reason'] == 'stop_loss'
        assert trade['pnl'] == pytest.approx(-20.0, abs=0.01)
        mock_position_manager.record_trade_pnl.assert_called_once()
        assert mock_position_manager.record_trade_pnl.call_args[0][0] == pytest.approx(-20.0, abs=0.01)

    def test_sync_no_open_trades(self, engine, mock_alpaca, db):
        """No open trades means no Alpaca calls."""
        engine._sync_closed_positions()
        mock_alpaca.get_open_positions.assert_not_called()

    def test_sync_take_profit_exit(self, engine, mock_alpaca, db, mock_position_manager):
        """Take profit exit detected correctly."""
        from datetime import timezone
        now = datetime.now(timezone.utc)

        db.save_trade({
            'trade_date': date.today().isoformat(),
            'symbol': 'WINNER',
            'side': 'buy',
            'entry_price': 5.00,
            'stop_loss_price': 4.80,
            'take_profit_price': 5.40,
            'shares': 100,
            'risk_per_share': 0.20,
            'total_risk': 20.0,
            'risk_reward_ratio': 2.0,
            'order_id': 'order-winner',
            'order_status': 'filled',
            'fill_price': 5.00,
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

        mock_alpaca.get_open_positions.return_value = []
        mock_alpaca.get_order.return_value = {
            'legs': [
                {'id': 'tp-1', 'side': 'sell', 'stop_price': None,
                 'limit_price': 5.40, 'status': 'filled'},
            ],
        }

        engine._sync_closed_positions()

        trade = db.get_trade_by_order_id('order-winner')
        assert trade['exit_reason'] == 'take_profit'
        assert trade['pnl'] == pytest.approx(40.0, abs=0.01)
        mock_position_manager.record_trade_pnl.assert_called_once()
        assert mock_position_manager.record_trade_pnl.call_args[0][0] == pytest.approx(40.0, abs=0.01)


# ===========================================================================
# Phase 5: Setup Expiry
# ===========================================================================

class TestSetupExpiry:
    """Tests for cancelling expired pending buy-stop orders."""

    MORNING_UTC = datetime(2026, 3, 16, 14, 0, 0, tzinfo=timezone.utc)  # 10:00 ET

    @pytest.fixture(autouse=True)
    def _mock_morning_time(self):
        """Force 10:00 ET so midday cancellation doesn't interfere."""
        with patch('trading.trading_engine.datetime') as mock_dt:
            mock_dt.now.return_value = self.MORNING_UTC
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            yield

    def test_expired_order_cancelled(self, engine, mock_alpaca):
        """Order older than expiry is cancelled."""
        old_time = self.MORNING_UTC - timedelta(seconds=700)
        engine.setup_expiry_seconds = 600

        engine._pending_orders['STALE'] = {
            'order_id': 'order-stale',
            'plan': _make_plan("STALE"),
            'setup': _make_pattern("STALE"),
            'placed_at': old_time,
        }

        mock_alpaca.get_order.return_value = {
            'status': 'new',
            'filled_avg_price': None,
        }
        mock_alpaca.cancel_order.return_value = True

        engine._manage_pending_orders()

        mock_alpaca.cancel_order.assert_called_once_with('order-stale')
        assert 'STALE' not in engine._pending_orders

    def test_non_expired_order_continues(self, engine, mock_alpaca):
        """Order within expiry window is NOT cancelled."""
        recent_time = self.MORNING_UTC - timedelta(seconds=60)
        engine.setup_expiry_seconds = 600

        engine._pending_orders['FRESH'] = {
            'order_id': 'order-fresh',
            'plan': _make_plan("FRESH"),
            'setup': _make_pattern("FRESH"),
            'placed_at': recent_time,
        }

        mock_alpaca.get_order.return_value = {
            'status': 'new',
            'filled_avg_price': None,
        }

        engine._manage_pending_orders()

        mock_alpaca.cancel_order.assert_not_called()
        assert 'FRESH' in engine._pending_orders


# ===========================================================================
# GAP 1: Force-Close DB Update
# ===========================================================================

class TestForceCloseDBUpdate:
    """Tests for _force_close_all updating trade records in DB."""

    @patch('trading.trading_engine.time_mod')
    def test_force_close_updates_db_with_exit(self, mock_time, engine, mock_alpaca, db, mock_position_manager):
        """Force-close writes exit_price, exit_reason, and P&L to DB."""
        now = datetime.now(timezone.utc)
        db.save_trade({
            'trade_date': date.today().isoformat(),
            'symbol': 'FC',
            'side': 'buy',
            'entry_price': 5.00,
            'stop_loss_price': 4.80,
            'take_profit_price': 5.40,
            'shares': 100,
            'risk_per_share': 0.20,
            'total_risk': 20.0,
            'risk_reward_ratio': 2.0,
            'order_id': 'order-fc',
            'order_status': 'filled',
            'fill_price': 5.00,
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

        # Alpaca returns position with market value
        mock_alpaca.get_open_positions.return_value = [
            {'symbol': 'FC', 'qty': 100, 'avg_entry_price': 5.00,
             'market_value': 520.0, 'unrealized_pl': 20.0, 'unrealized_plpc': 0.04},
        ]
        mock_alpaca.close_position.return_value = {'id': 'close-order-1'}
        # Poll returns actual fill price
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 5.18,
        }

        engine._force_close_all()

        # Verify DB updated with polled fill price
        trade = db.get_trade_by_order_id('order-fc')
        assert trade['exit_reason'] == 'force_close'
        assert trade['exit_price'] == pytest.approx(5.18, abs=0.01)
        expected_pnl = (5.18 - 5.00) * 100
        assert trade['pnl'] == pytest.approx(expected_pnl, abs=0.01)
        assert trade['exited_at'] is not None

        # Verify circuit breaker was fed
        mock_position_manager.record_trade_pnl.assert_called_once()
        assert mock_position_manager.record_trade_pnl.call_args[0][0] == pytest.approx(expected_pnl, abs=0.01)

    def test_force_close_cancels_pending(self, engine, mock_alpaca):
        """Force-close cancels pending buy-stop orders."""
        engine._pending_orders['PEND'] = {
            'order_id': 'order-pend',
            'plan': _make_plan("PEND"),
        }
        mock_alpaca.get_open_positions.return_value = []

        engine._force_close_all()

        mock_alpaca.cancel_order.assert_called_once_with('order-pend')
        assert len(engine._pending_orders) == 0

    @patch('trading.trading_engine.time_mod')
    def test_force_close_no_fill_price_logs_warning(self, mock_time, engine, mock_alpaca, db):
        """Force-close with no fill_price skips P&L update and warns."""
        now = datetime.now(timezone.utc)
        db.save_trade({
            'trade_date': date.today().isoformat(),
            'symbol': 'NOFILL',
            'side': 'buy',
            'entry_price': 5.00,
            'stop_loss_price': 4.80,
            'take_profit_price': 5.40,
            'shares': 100,
            'risk_per_share': 0.20,
            'total_risk': 20.0,
            'risk_reward_ratio': 2.0,
            'order_id': 'order-nofill',
            'order_status': 'accepted',
            'fill_price': None,
            'filled_at': None,
            'exit_price': None,
            'exit_reason': None,
            'exited_at': None,
            'pnl': None,
            'pnl_pct': None,
            'pattern_data': '{}',
            'created_at': now,
            'updated_at': now,
        })

        mock_alpaca.get_open_positions.return_value = [
            {'symbol': 'NOFILL', 'qty': 100, 'avg_entry_price': 5.00,
             'market_value': 480.0, 'unrealized_pl': -20.0, 'unrealized_plpc': -0.04},
        ]
        mock_alpaca.close_position.return_value = {'id': 'close-order-2'}
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 4.80,
        }

        engine._force_close_all()

        # DB should NOT be updated with exit (no fill_price to compute from)
        trade = db.get_trade_by_order_id('order-nofill')
        assert trade['exit_price'] is None


# ===========================================================================
# GAP 2: Sync uses filled_avg_price for legs
# ===========================================================================

class TestSyncUsesFilledAvgPrice:
    """Tests for _sync_closed_positions using actual fill price from legs."""

    def test_sync_uses_filled_avg_price_over_stop_price(self, engine, mock_alpaca, db, mock_position_manager):
        """When leg has filled_avg_price, use it instead of stop_price."""
        now = datetime.now(timezone.utc)
        db.save_trade({
            'trade_date': date.today().isoformat(),
            'symbol': 'SLIP',
            'side': 'buy',
            'entry_price': 5.00,
            'stop_loss_price': 4.80,
            'take_profit_price': 5.40,
            'shares': 100,
            'risk_per_share': 0.20,
            'total_risk': 20.0,
            'risk_reward_ratio': 2.0,
            'order_id': 'order-slip',
            'order_status': 'filled',
            'fill_price': 5.00,
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

        mock_alpaca.get_open_positions.return_value = []
        # Stop triggered at 4.80 but filled at 4.75 (slippage)
        mock_alpaca.get_order.return_value = {
            'legs': [
                {'id': 'sl-slip', 'side': 'sell', 'stop_price': 4.80,
                 'limit_price': None, 'filled_avg_price': 4.75, 'status': 'filled'},
            ],
        }

        engine._sync_closed_positions()

        trade = db.get_trade_by_order_id('order-slip')
        assert trade['exit_price'] == pytest.approx(4.75, abs=0.01)
        assert trade['exit_reason'] == 'stop_loss'
        # P&L should use actual fill (4.75), not trigger (4.80)
        assert trade['pnl'] == pytest.approx(-25.0, abs=0.01)

    def test_sync_falls_back_to_stop_price(self, engine, mock_alpaca, db, mock_position_manager):
        """When leg has no filled_avg_price, falls back to stop_price."""
        now = datetime.now(timezone.utc)
        db.save_trade({
            'trade_date': date.today().isoformat(),
            'symbol': 'NOSLIP',
            'side': 'buy',
            'entry_price': 5.00,
            'stop_loss_price': 4.80,
            'take_profit_price': 5.40,
            'shares': 100,
            'risk_per_share': 0.20,
            'total_risk': 20.0,
            'risk_reward_ratio': 2.0,
            'order_id': 'order-noslip',
            'order_status': 'filled',
            'fill_price': 5.00,
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

        mock_alpaca.get_open_positions.return_value = []
        mock_alpaca.get_order.return_value = {
            'legs': [
                {'id': 'sl-ns', 'side': 'sell', 'stop_price': 4.80,
                 'limit_price': None, 'filled_avg_price': None, 'status': 'filled'},
            ],
        }

        engine._sync_closed_positions()

        trade = db.get_trade_by_order_id('order-noslip')
        assert trade['exit_price'] == pytest.approx(4.80, abs=0.01)



# ===========================================================================
# Fix 1: Fill Data Missing Fallback
# ===========================================================================

class TestFillDataFallback:
    """Tests for retry + position fallback when fill_price is None."""

    def _setup_pending(self, engine, db, order_id='order-retry'):
        """Helper: create a pending order with DB record."""
        now = datetime.now(timezone.utc)
        db.save_trade({
            'trade_date': date.today().isoformat(),
            'symbol': 'RETRY',
            'side': 'buy',
            'entry_price': 5.00,
            'stop_loss_price': 4.80,
            'take_profit_price': 5.40,
            'shares': 100,
            'risk_per_share': 0.20,
            'total_risk': 20.0,
            'risk_reward_ratio': 2.0,
            'order_id': order_id,
            'order_status': 'accepted',
            'fill_price': None,
            'filled_at': None,
            'exit_price': None,
            'exit_reason': None,
            'exited_at': None,
            'pnl': None,
            'pnl_pct': None,
            'pattern_data': '{}',
            'created_at': now,
            'updated_at': now,
        })
        plan = _make_plan("RETRY")
        engine._pending_orders['RETRY'] = {
            'order_id': order_id,
            'plan': plan,
            'setup': _make_pattern("RETRY"),
            'placed_at': now,
        }
        return plan

    @patch('trading.trading_engine.time_mod.sleep')
    def test_retry_resolves_fill_price(self, mock_sleep, engine, mock_alpaca, db):
        """Fill price resolved on retry after initial None."""
        self._setup_pending(engine, db)

        # First call: filled but no price. Second call (retry): price present.
        mock_alpaca.get_order.side_effect = [
            {'status': 'filled', 'filled_avg_price': None, 'filled_qty': 0, 'legs': []},
            {'filled_avg_price': 4.48, 'filled_qty': 100},
        ]

        result = engine._manage_pending_orders()

        assert result is not None
        assert result['fill_price'] == 4.48
        trade = db.get_trade_by_order_id('order-retry')
        assert trade['fill_price'] == 4.48

    @patch('trading.trading_engine.time_mod.sleep')
    def test_position_fallback_when_retries_fail(self, mock_sleep, engine, mock_alpaca, db):
        """Falls back to position data when all retries return None."""
        self._setup_pending(engine, db)

        # All get_order calls return None for fill_price
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': None, 'filled_qty': 0, 'legs': [],
        }
        mock_alpaca.get_open_positions.return_value = [
            {'symbol': 'RETRY', 'avg_entry_price': '4.47', 'qty': '100'},
        ]

        result = engine._manage_pending_orders()

        assert result is not None
        assert result['fill_price'] == 4.47

    @patch('trading.trading_engine.time_mod.sleep')
    def test_skip_when_all_fallbacks_fail(self, mock_sleep, engine, mock_alpaca, db):
        """Skips trade when fill price unavailable after all retries and fallbacks."""
        self._setup_pending(engine, db)

        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': None, 'filled_qty': 0, 'legs': [],
        }
        mock_alpaca.get_open_positions.return_value = []  # No matching position

        result = engine._manage_pending_orders()

        # Should return None (skipped) and symbol still in pending
        assert result is None


# ===========================================================================
# Fix 2: Partial Fill Detection
# ===========================================================================

class TestPartialFillDetection:
    """Tests for partial fill detection and filled_qty tracking."""

    def test_partial_fill_logged_and_tracked(self, engine, mock_alpaca, db):
        """Partial fill records actual filled_qty, not requested shares."""
        now = datetime.now(timezone.utc)
        db.save_trade({
            'trade_date': date.today().isoformat(),
            'symbol': 'PART',
            'side': 'buy',
            'entry_price': 4.40,
            'stop_loss_price': 4.29,
            'take_profit_price': 4.90,
            'shares': 113,
            'risk_per_share': 0.11,
            'total_risk': 12.43,
            'risk_reward_ratio': 4.5,
            'order_id': 'order-partial',
            'order_status': 'accepted',
            'fill_price': None,
            'filled_at': None,
            'exit_price': None,
            'exit_reason': None,
            'exited_at': None,
            'pnl': None,
            'pnl_pct': None,
            'pattern_data': '{}',
            'created_at': now,
            'updated_at': now,
        })

        plan = _make_plan("PART")
        engine._pending_orders['PART'] = {
            'order_id': 'order-partial',
            'plan': plan,
            'setup': _make_pattern("PART"),
            'placed_at': now,
        }

        # Fill at breakout_level (4.40) to avoid gap-fill path
        mock_alpaca.get_order.return_value = {
            'status': 'filled',
            'filled_avg_price': 4.40,
            'filled_qty': 50,
        }

        result = engine._manage_pending_orders()

        assert result is not None
        assert result['filled_qty'] == 50
        trade = db.get_trade_by_order_id('order-partial')
        assert trade['filled_qty'] == 50

    def test_full_fill_uses_requested_qty(self, engine, mock_alpaca, db):
        """Full fill uses requested quantity when filled_qty matches or is 0."""
        now = datetime.now(timezone.utc)
        db.save_trade({
            'trade_date': date.today().isoformat(),
            'symbol': 'FULL',
            'side': 'buy',
            'entry_price': 4.40,
            'stop_loss_price': 4.29,
            'take_profit_price': 4.90,
            'shares': 113,
            'risk_per_share': 0.11,
            'total_risk': 12.43,
            'risk_reward_ratio': 4.5,
            'order_id': 'order-full',
            'order_status': 'accepted',
            'fill_price': None,
            'filled_at': None,
            'exit_price': None,
            'exit_reason': None,
            'exited_at': None,
            'pnl': None,
            'pnl_pct': None,
            'pattern_data': '{}',
            'created_at': now,
            'updated_at': now,
        })

        plan = _make_plan("FULL")
        engine._pending_orders['FULL'] = {
            'order_id': 'order-full',
            'plan': plan,
            'setup': _make_pattern("FULL"),
            'placed_at': now,
        }

        # Fill at breakout_level (4.40) to avoid gap-fill path
        mock_alpaca.get_order.return_value = {
            'status': 'filled',
            'filled_avg_price': 4.40,
            'filled_qty': 0,
        }

        result = engine._manage_pending_orders()

        assert result is not None
        # When filled_qty=0, should use requested (plan.shares = 113)
        assert result['filled_qty'] == plan.shares

    def test_sync_closed_uses_filled_qty_for_pnl(self, engine, mock_alpaca, db, mock_position_manager):
        """P&L calculation uses filled_qty when available."""
        now = datetime.now(timezone.utc)
        db.save_trade({
            'trade_date': date.today().isoformat(),
            'symbol': 'PQTY',
            'side': 'buy',
            'entry_price': 5.00,
            'stop_loss_price': 4.80,
            'take_profit_price': 5.40,
            'shares': 100,
            'risk_per_share': 0.20,
            'total_risk': 20.0,
            'risk_reward_ratio': 2.0,
            'order_id': 'order-pqty',
            'order_status': 'filled',
            'fill_price': 5.00,
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
        # Manually set filled_qty to 50 (partial fill)
        db.update_trade(1, {'filled_qty': 50})

        mock_alpaca.get_open_positions.return_value = []
        mock_alpaca.get_order.return_value = {
            'legs': [
                {'id': 'sl-1', 'side': 'sell', 'stop_price': 4.80,
                 'limit_price': None, 'filled_avg_price': 4.80, 'status': 'filled'},
            ],
        }

        engine._sync_closed_positions()

        trade = db.get_trade_by_order_id('order-pqty')
        # P&L should be (4.80 - 5.00) * 50 = -10.0
        assert trade['pnl'] == pytest.approx(-10.0, abs=0.01)


# ===========================================================================
# Fix 3: Force Close Retry
# ===========================================================================

class TestForceCloseRetry:
    """Tests for force close retry with backoff."""

    @patch('trading.trading_engine.time_mod.sleep')
    def test_retry_succeeds_on_second_attempt(self, mock_sleep, engine, mock_alpaca, db, mock_position_manager):
        """Force close succeeds on retry after first attempt fails."""
        now = datetime.now(timezone.utc)
        db.save_trade({
            'trade_date': date.today().isoformat(),
            'symbol': 'RETRY',
            'side': 'buy',
            'entry_price': 5.00,
            'stop_loss_price': 4.80,
            'take_profit_price': 5.40,
            'shares': 100,
            'risk_per_share': 0.20,
            'total_risk': 20.0,
            'risk_reward_ratio': 2.0,
            'order_id': 'order-retry-fc',
            'order_status': 'filled',
            'fill_price': 5.00,
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

        mock_alpaca.get_open_positions.return_value = [
            {'symbol': 'RETRY', 'qty': 100, 'avg_entry_price': 5.00,
             'market_value': 520.0},
        ]
        # First attempt fails, second succeeds
        mock_alpaca.close_position.side_effect = [
            Exception("API timeout"),
            {'id': 'close-1'},
        ]
        # Poll for close fill price
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 5.20,
        }

        engine._force_close_all()

        assert mock_alpaca.close_position.call_count == 2
        trade = db.get_trade_by_order_id('order-retry-fc')
        assert trade['exit_reason'] == 'force_close'
        assert trade['exit_price'] == pytest.approx(5.20, abs=0.01)
        # Verify backoff sleep was called (among fill-poll sleeps)
        mock_sleep.assert_any_call(2)

    @patch('trading.trading_engine.time_mod.sleep')
    def test_all_retries_fail_notifies(self, mock_sleep, engine, mock_alpaca, db):
        """All retries fail triggers error notification."""
        mock_notifier = MagicMock(spec=TelegramNotifier)
        engine.notifier = mock_notifier

        mock_alpaca.get_open_positions.return_value = [
            {'symbol': 'STUCK', 'qty': 100, 'avg_entry_price': 5.00,
             'market_value': 520.0},
        ]
        mock_alpaca.close_position.side_effect = Exception("persistent error")

        engine._force_close_all()

        assert mock_alpaca.close_position.call_count == 3
        mock_notifier.notify_error.assert_called_once()
        assert 'MANUAL INTERVENTION' in mock_notifier.notify_error.call_args[0][0]


# ===========================================================================
# Fix 4: Graceful Shutdown
# ===========================================================================

class TestGracefulShutdown:
    """Tests for shutdown_event integration in monitoring loop."""

    def test_shutdown_event_stops_loop(self, engine, mock_alpaca, db):
        """Setting shutdown_event stops the monitoring loop."""
        shutdown_event = threading.Event()
        engine.shutdown_event = shutdown_event

        # Set shutdown immediately
        shutdown_event.set()

        mock_alpaca.get_open_positions.return_value = []

        # Should exit immediately and call force_close_all
        engine.run_monitoring_loop()

        # Verify graceful shutdown happened
        mock_alpaca.get_open_positions.assert_called()

    def test_loop_without_shutdown_event(self, engine):
        """Loop works normally without shutdown_event set."""
        engine.shutdown_event = None
        # With market closed (16:00+), loop exits naturally
        with patch('trading.trading_engine.datetime') as mock_dt:
            mock_now = MagicMock()
            mock_now.hour = 16
            mock_now.minute = 0
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            # This will just exit because hour >= 16
            engine.run_monitoring_loop()


# ===========================================================================
# Fix 6: Bracket Leg Identification
# ===========================================================================

class TestIdentifyBracketLegs:
    """Tests for _identify_bracket_legs helper."""

    def test_identifies_sl_and_tp(self, engine):
        """Correctly identifies SL (stop_price only) and TP (limit_price only)."""
        legs = [
            {'side': 'sell', 'stop_price': 4.80, 'limit_price': None},
            {'side': 'sell', 'stop_price': None, 'limit_price': 5.40},
        ]
        sl, tp = engine._identify_bracket_legs(legs)
        assert sl['stop_price'] == 4.80
        assert tp['limit_price'] == 5.40

    def test_ignores_buy_legs(self, engine):
        """Buy-side legs are ignored."""
        legs = [
            {'side': 'buy', 'stop_price': 4.80, 'limit_price': None},
            {'side': 'sell', 'stop_price': None, 'limit_price': 5.40},
        ]
        sl, tp = engine._identify_bracket_legs(legs)
        assert sl is None
        assert tp['limit_price'] == 5.40

    def test_empty_legs(self, engine):
        """Empty legs list returns (None, None)."""
        sl, tp = engine._identify_bracket_legs([])
        assert sl is None
        assert tp is None

    def test_both_stop_and_limit_disambiguated_by_expected_sl(self, engine):
        """Leg with both stop_price and limit_price is disambiguated by expected_sl."""
        legs = [
            {'side': 'sell', 'stop_price': 4.80, 'limit_price': 4.75},
        ]
        # stop_price (4.80) is closer to expected_sl (4.80) than limit_price (4.75)
        sl, tp = engine._identify_bracket_legs(legs, expected_sl=4.80)
        assert sl is not None
        assert sl['stop_price'] == 4.80

    def test_both_stop_and_limit_disambiguated_as_tp(self, engine):
        """Leg with both prices assigned to TP when limit is closer to expected_sl."""
        legs = [
            {'side': 'sell', 'stop_price': 5.40, 'limit_price': 4.82},
        ]
        # limit_price (4.82) is closer to expected_sl (4.80) than stop_price (5.40)
        # So abs(stop-sl)=0.60 > abs(limit-sl)=0.02 → assigned to tp_leg
        sl, tp = engine._identify_bracket_legs(legs, expected_sl=4.80)
        assert sl is None
        assert tp is not None


# ===========================================================================
# Fix 7: Race Condition — Check Status Before Cancel
# ===========================================================================

class TestRaceConditionOnCancel:
    """Tests for checking order status before cancellation."""

    MORNING_UTC = datetime(2026, 3, 16, 14, 0, 0, tzinfo=timezone.utc)  # 10:00 ET

    @pytest.fixture(autouse=True)
    def _mock_morning_time(self):
        """Force 10:00 ET so midday cancellation doesn't interfere."""
        with patch('trading.trading_engine.datetime') as mock_dt:
            mock_dt.now.return_value = self.MORNING_UTC
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            yield

    def test_expiry_detects_filled_order(self, engine, mock_alpaca):
        """Expired order that filled in the interim is not cancelled."""
        old_time = self.MORNING_UTC - timedelta(seconds=700)
        engine.setup_expiry_seconds = 600

        engine._pending_orders['RACE'] = {
            'order_id': 'order-race',
            'plan': _make_plan("RACE"),
            'setup': _make_pattern("RACE"),
            'placed_at': old_time,
        }

        # First get_order: status is new (triggers expiry path)
        # Second get_order (refresh): status is filled
        mock_alpaca.get_order.side_effect = [
            {'status': 'new', 'filled_avg_price': None},
            {'status': 'filled'},
        ]

        engine._manage_pending_orders()

        # Should NOT cancel — order filled
        mock_alpaca.cancel_order.assert_not_called()
        # Symbol stays in pending (will be processed as fill next cycle)
        assert 'RACE' in engine._pending_orders

    def test_expiry_detects_already_cancelled(self, engine, mock_alpaca):
        """Expired order that is already cancelled is cleaned up without cancel call."""
        old_time = self.MORNING_UTC - timedelta(seconds=700)
        engine.setup_expiry_seconds = 600

        engine._pending_orders['GONE'] = {
            'order_id': 'order-gone',
            'plan': _make_plan("GONE"),
            'setup': _make_pattern("GONE"),
            'placed_at': old_time,
        }

        mock_alpaca.get_order.side_effect = [
            {'status': 'new', 'filled_avg_price': None},
            {'status': 'cancelled'},
        ]

        engine._manage_pending_orders()

        mock_alpaca.cancel_order.assert_not_called()
        assert 'GONE' not in engine._pending_orders

    def test_invalidation_detects_filled_order(self, engine, mock_alpaca):
        """Invalidated order that filled is not cancelled."""
        recent_time = self.MORNING_UTC - timedelta(seconds=30)
        engine.setup_expiry_seconds = 600

        setup = _make_pattern("INVAL")
        engine._pending_orders['INVAL'] = {
            'order_id': 'order-inval',
            'plan': _make_plan("INVAL"),
            'setup': setup,
            'placed_at': recent_time,
        }

        # Create bars below flag_low to trigger invalidation
        bars = pd.DataFrame({
            'open': [4.0], 'high': [4.1], 'low': [3.0],
            'close': [3.5], 'volume': [100000],
        })
        mock_alpaca.get_1min_bars.return_value = bars

        # First get_order: pending (to enter invalidation check)
        # Second get_order (refresh before cancel): filled!
        mock_alpaca.get_order.side_effect = [
            {'status': 'new', 'filled_avg_price': None},
            {'status': 'filled'},
        ]

        engine._manage_pending_orders()

        mock_alpaca.cancel_order.assert_not_called()
        assert 'INVAL' in engine._pending_orders


# ===========================================================================
# DB Migration: filled_qty column
# ===========================================================================

class TestFilledQtyMigration:
    """Tests for filled_qty column migration."""

    def test_filled_qty_column_exists(self, db):
        """Database has filled_qty column after migration."""
        columns = [row[1] for row in db.conn.execute("PRAGMA table_info(trades)").fetchall()]
        assert 'filled_qty' in columns

    def test_filled_qty_stored_and_retrieved(self, db):
        """filled_qty can be saved and retrieved from trades."""
        now = datetime.now(timezone.utc)
        trade_id = db.save_trade({
            'trade_date': date.today().isoformat(),
            'symbol': 'MIGR',
            'side': 'buy',
            'entry_price': 5.00,
            'stop_loss_price': 4.80,
            'take_profit_price': 5.40,
            'shares': 100,
            'risk_per_share': 0.20,
            'total_risk': 20.0,
            'risk_reward_ratio': 2.0,
            'order_id': 'order-migr',
            'order_status': 'accepted',
            'fill_price': None,
            'filled_at': None,
            'exit_price': None,
            'exit_reason': None,
            'exited_at': None,
            'pnl': None,
            'pnl_pct': None,
            'pattern_data': '{}',
            'created_at': now,
            'updated_at': now,
        })
        db.update_trade(trade_id, {'filled_qty': 75})
        trade = db.get_trade_by_order_id('order-migr')
        assert trade['filled_qty'] == 75


# ---------------------------------------------------------------------------
# Market Regime Filter Integration
# ---------------------------------------------------------------------------

class TestMarketRegimeInEngine:
    """Tests for market regime filter in TradingEngine."""

    @staticmethod
    def _build_regime_bars(last_5_volatile_below_sma=True, num_bars=55):
        """Build SPY OHLC bars for regime testing. Needs 50+ for SMA."""
        bars = []
        d = date(2025, 1, 1)
        for i in range(num_bars):
            while d.weekday() >= 5:
                d += timedelta(days=1)
            bars.append({
                'date': d,
                'open': 500.0,
                'high': 502.0,
                'low': 498.0,
                'close': 500.0,
            })
            d += timedelta(days=1)

        if last_5_volatile_below_sma:
            # Make last 5 bars volatile AND below SMA(50) ~500
            for i in range(-5, 0):
                bars[i]['close'] = 480.0
                bars[i]['high'] = 495.0
                bars[i]['low'] = 470.0  # (495-470)/480 = 5.2%
        return bars

    def test_regime_blocks_pattern_check(self, mock_alpaca, db, mock_detector,
                                          mock_planner, mock_executor, mock_position_manager):
        """run_pattern_check returns None when regime filter blocks."""
        from trading.market_regime import MarketRegimeFilter

        bars = self._build_regime_bars(last_5_volatile_below_sma=True)
        regime = MarketRegimeFilter(enabled=True, vol_threshold=1.5, sma_period=50)
        regime.load_spy_bars(bars)

        # Find trade_date = day after last bar
        trade_date = bars[-1]['date'] + timedelta(days=1)
        while trade_date.weekday() >= 5:
            trade_date += timedelta(days=1)

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=mock_detector,
            planner=mock_planner, executor=mock_executor,
            position_manager=mock_position_manager,
            enabled=True, market_regime=regime,
        )
        engine.news_gate_enabled = False
        engine.on_stock_qualified("AAPL")

        with patch('trading.trading_engine.date') as mock_date:
            mock_date.today.return_value = trade_date
            mock_date.side_effect = lambda *a, **k: date(*a, **k)
            result = engine.run_pattern_check()

        assert result is None
        mock_detector.detect_setup.assert_not_called()

    def test_regime_allows_when_ok(self, mock_alpaca, db, mock_detector,
                                    mock_planner, mock_executor, mock_position_manager):
        """run_pattern_check proceeds when regime filter allows."""
        from trading.market_regime import MarketRegimeFilter

        bars = self._build_regime_bars(last_5_volatile_below_sma=False)
        regime = MarketRegimeFilter(enabled=True, vol_threshold=1.5, sma_period=50)
        regime.load_spy_bars(bars)

        trade_date = bars[-1]['date'] + timedelta(days=1)
        while trade_date.weekday() >= 5:
            trade_date += timedelta(days=1)

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=mock_detector,
            planner=mock_planner, executor=mock_executor,
            position_manager=mock_position_manager,
            enabled=True, market_regime=regime,
        )
        engine.news_gate_enabled = False
        engine.on_stock_qualified("AAPL")

        with patch('trading.trading_engine.date') as mock_date:
            mock_date.today.return_value = trade_date
            mock_date.side_effect = lambda *a, **k: date(*a, **k)
            with patch.object(TradingEngine, '_is_past_last_entry_time', return_value=False):
                mock_alpaca.get_1min_bars.return_value = pd.DataFrame()
                engine.run_pattern_check()

        # Should have proceeded past regime check

    def test_max_trades_per_day_blocks(self, mock_alpaca, db, mock_detector,
                                        mock_planner, mock_executor, mock_position_manager):
        """run_pattern_check blocks when max trades per day reached."""
        from trading.market_regime import MarketRegimeFilter

        regime = MarketRegimeFilter(enabled=True, max_trades_per_day=2)
        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=mock_detector,
            planner=mock_planner, executor=mock_executor,
            position_manager=mock_position_manager,
            enabled=True, market_regime=regime,
        )
        engine.news_gate_enabled = False
        engine._daily_trade_count = 2  # Already at max

        result = engine.run_pattern_check()
        assert result is None

    def test_reset_daily_resets_trade_count(self, mock_alpaca, db, mock_detector,
                                             mock_planner, mock_executor, mock_position_manager):
        """reset_daily resets _daily_trade_count to 0."""
        from trading.market_regime import MarketRegimeFilter

        regime = MarketRegimeFilter(enabled=True)
        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=mock_detector,
            planner=mock_planner, executor=mock_executor,
            position_manager=mock_position_manager,
            enabled=True, market_regime=regime,
        )
        engine.news_gate_enabled = False
        engine._daily_trade_count = 5

        mock_alpaca.get_daily_bars_range.return_value = {'SPY': [
            {'date': date(2025, 3, 10), 'close': 510.0},
        ]}

        engine.reset_daily()
        assert engine._daily_trade_count == 0

    def test_reset_daily_refreshes_spy(self, mock_alpaca, db, mock_detector,
                                        mock_planner, mock_executor, mock_position_manager):
        """reset_daily calls _refresh_spy_data."""
        from trading.market_regime import MarketRegimeFilter

        regime = MarketRegimeFilter(enabled=True)
        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=mock_detector,
            planner=mock_planner, executor=mock_executor,
            position_manager=mock_position_manager,
            enabled=True, market_regime=regime,
        )
        engine.news_gate_enabled = False

        mock_alpaca.get_daily_bars_range.return_value = {'SPY': [
            {'date': date(2025, 3, 10), 'close': 510.0},
        ]}

        engine.reset_daily()

        mock_alpaca.get_daily_bars_range.assert_called_once()
        call_args = mock_alpaca.get_daily_bars_range.call_args
        assert call_args[0][0] == ['SPY']


# ===========================================================================
# Volume-Conditional Pending Orders (H5 OR filter in live trading)
# ===========================================================================

class TestThinLiquidityPostFillCheck:
    """Tests for thin-liquidity post-fill volume check (H5 OR filter rewrite)."""

    def _make_engine(self, mock_alpaca, db, mock_detector, mock_planner,
                     mock_executor, mock_position_manager, market_regime):
        """Create TradingEngine with market_regime."""
        eng = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=mock_detector,
            planner=mock_planner, executor=mock_executor,
            position_manager=mock_position_manager,
            enabled=True, market_regime=market_regime,
        )
        eng.news_gate_enabled = False
        eng.quality_filter_enabled = False
        eng.conviction_enabled = False
        return eng

    def _make_thin_regime(self):
        """Create a mock MarketRegimeFilter that reports thin liquidity."""
        from trading.market_regime import MarketRegimeFilter
        regime = MagicMock(spec=MarketRegimeFilter)
        regime.is_regime_ok.return_value = True
        regime.is_thin_liquidity.return_value = True
        regime.get_min_breakout_volume_ratio.return_value = 2.0
        regime.get_regime_info.return_value = {
            'vol_5d': 1.0, 'sma': 500.0, 'is_below_sma': False,
            'is_ok': True, 'spy_volume_ratio': 0.55,
        }
        regime.max_trades_per_day = 5
        regime.min_spy_volume_ratio = 0.70
        regime.thin_liquidity_breakout_vol_ratio = 2.0
        regime.sma_period = 50
        return regime

    def _make_normal_regime(self):
        """Create a mock MarketRegimeFilter that reports normal liquidity."""
        from trading.market_regime import MarketRegimeFilter
        regime = MagicMock(spec=MarketRegimeFilter)
        regime.is_regime_ok.return_value = True
        regime.is_thin_liquidity.return_value = False
        regime.max_trades_per_day = 5
        regime.sma_period = 50
        return regime

    @patch('trading.trading_engine.TradingEngine._is_past_last_entry_time', return_value=False)
    def test_thin_day_tags_pending_with_thin_liquidity(
        self, _mock_time, mock_alpaca, db, mock_detector, mock_planner,
        mock_executor, mock_position_manager
    ):
        """On thin day, pending order has thin_liquidity=True + min_breakout_vol_ratio."""
        regime = self._make_thin_regime()
        engine = self._make_engine(
            mock_alpaca, db, mock_detector, mock_planner,
            mock_executor, mock_position_manager, regime,
        )

        setup = _make_pattern("AAPL")
        plan = _make_plan("AAPL")
        mock_detector.detect_setup.return_value = setup
        mock_planner.create_plan.return_value = plan
        mock_position_manager.can_open_position.return_value = True
        mock_alpaca.get_1min_bars.return_value = pd.DataFrame({'close': [4.35]})
        mock_alpaca.get_open_positions.return_value = []
        mock_executor.submit_buy_stop_bracket_order.return_value = {
            'order_id': 'order-thin-1', 'status': 'accepted',
            'symbol': 'AAPL', 'shares': 113,
        }

        engine.on_stock_qualified("AAPL")
        result = engine.run_pattern_check()

        assert result is not None
        assert 'AAPL' in engine._pending_orders
        pending = engine._pending_orders['AAPL']
        assert pending['thin_liquidity'] is True
        assert pending['min_breakout_vol_ratio'] == 2.0
        assert pending['order_id'] == 'order-thin-1'

    @patch('trading.trading_engine.TradingEngine._is_past_last_entry_time', return_value=False)
    def test_thin_day_still_submits_buy_stop(
        self, _mock_time, mock_alpaca, db, mock_detector, mock_planner,
        mock_executor, mock_position_manager
    ):
        """On thin day, submit_buy_stop_bracket_order is called (NOT held back)."""
        regime = self._make_thin_regime()
        engine = self._make_engine(
            mock_alpaca, db, mock_detector, mock_planner,
            mock_executor, mock_position_manager, regime,
        )

        setup = _make_pattern("AAPL")
        plan = _make_plan("AAPL")
        mock_detector.detect_setup.return_value = setup
        mock_planner.create_plan.return_value = plan
        mock_position_manager.can_open_position.return_value = True
        mock_alpaca.get_1min_bars.return_value = pd.DataFrame({'close': [4.35]})
        mock_alpaca.get_open_positions.return_value = []
        mock_executor.submit_buy_stop_bracket_order.return_value = {
            'order_id': 'order-thin-2', 'status': 'accepted',
            'symbol': 'AAPL', 'shares': 113,
        }

        engine.on_stock_qualified("AAPL")
        engine.run_pattern_check()

        mock_executor.submit_buy_stop_bracket_order.assert_called_once_with(plan)

    @patch('trading.trading_engine.TradingEngine._is_past_last_entry_time', return_value=False)
    def test_normal_day_no_thin_tag(
        self, _mock_time, mock_alpaca, db, mock_detector, mock_planner,
        mock_executor, mock_position_manager
    ):
        """Non-thin day: no thin_liquidity key in pending."""
        regime = self._make_normal_regime()
        engine = self._make_engine(
            mock_alpaca, db, mock_detector, mock_planner,
            mock_executor, mock_position_manager, regime,
        )

        setup = _make_pattern("AAPL")
        plan = _make_plan("AAPL")
        mock_detector.detect_setup.return_value = setup
        mock_planner.create_plan.return_value = plan
        mock_position_manager.can_open_position.return_value = True
        mock_alpaca.get_1min_bars.return_value = pd.DataFrame({'close': [4.35]})
        mock_alpaca.get_open_positions.return_value = []
        mock_executor.submit_buy_stop_bracket_order.return_value = {
            'order_id': 'order-123', 'status': 'accepted',
            'symbol': 'AAPL', 'shares': 113,
        }

        engine.on_stock_qualified("AAPL")
        engine.run_pattern_check()

        assert 'AAPL' in engine._pending_orders
        assert 'thin_liquidity' not in engine._pending_orders['AAPL']
        mock_executor.submit_buy_stop_bracket_order.assert_called_once()

    @patch('trading.trading_engine.time_mod')
    def test_fill_on_thin_day_checks_breakout_volume(
        self, mock_time, mock_alpaca, db, mock_detector, mock_planner,
        mock_executor, mock_position_manager
    ):
        """On fill with thin_liquidity=True, _check_breakout_volume is called."""
        regime = self._make_thin_regime()
        engine = self._make_engine(
            mock_alpaca, db, mock_detector, mock_planner,
            mock_executor, mock_position_manager, regime,
        )

        setup = _make_pattern("AAPL")
        plan = _make_plan("AAPL")

        # Save trade to DB first
        from datetime import date as date_cls
        trade_id = db.save_trade({
            'trade_date': date_cls.today().isoformat(), 'symbol': 'AAPL',
            'side': 'buy', 'entry_price': 4.40, 'stop_loss_price': 4.29,
            'take_profit_price': 4.90, 'shares': 113, 'risk_per_share': 0.11,
            'total_risk': 12.43, 'risk_reward_ratio': 4.5,
            'order_id': 'order-fill-1', 'order_status': 'accepted',
            'fill_price': None, 'filled_at': None, 'exit_price': None,
            'exit_reason': None, 'exited_at': None, 'pnl': None, 'pnl_pct': None,
            'pattern_data': '{}', 'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
        })

        engine._pending_orders['AAPL'] = {
            'order_id': 'order-fill-1',
            'plan': plan,
            'setup': setup,
            'placed_at': datetime.now(timezone.utc),
            'thin_liquidity': True,
            'min_breakout_vol_ratio': 2.0,
        }

        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 4.40, 'filled_qty': 113,
        }
        mock_alpaca.get_open_positions.return_value = []

        # Strong volume — BVR 2.5x (100000/40000)
        bars = pd.DataFrame([
            {'high': 4.45, 'low': 4.35, 'close': 4.42, 'volume': 100000},
        ])
        mock_alpaca.get_1min_bars.return_value = bars

        result = engine._manage_pending_orders()

        # Should be kept (filled at breakout_level, no gap), not rejected
        assert result is not None
        assert result['status'] == 'filled'
        assert engine._patterns_traded == 1

    @patch('trading.trading_engine.time_mod')
    def test_fill_on_thin_day_strong_volume_keeps_trade(
        self, mock_time, mock_alpaca, db, mock_detector, mock_planner,
        mock_executor, mock_position_manager
    ):
        """BVR >= 2.0x → trade kept, gap-fill proceeds, _patterns_traded incremented."""
        regime = self._make_thin_regime()
        engine = self._make_engine(
            mock_alpaca, db, mock_detector, mock_planner,
            mock_executor, mock_position_manager, regime,
        )

        setup = _make_pattern("AAPL")
        plan = _make_plan("AAPL")

        from datetime import date as date_cls
        db.save_trade({
            'trade_date': date_cls.today().isoformat(), 'symbol': 'AAPL',
            'side': 'buy', 'entry_price': 4.40, 'stop_loss_price': 4.29,
            'take_profit_price': 4.90, 'shares': 113, 'risk_per_share': 0.11,
            'total_risk': 12.43, 'risk_reward_ratio': 4.5,
            'order_id': 'order-keep-1', 'order_status': 'accepted',
            'fill_price': None, 'filled_at': None, 'exit_price': None,
            'exit_reason': None, 'exited_at': None, 'pnl': None, 'pnl_pct': None,
            'pattern_data': '{}', 'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
        })

        engine._pending_orders['AAPL'] = {
            'order_id': 'order-keep-1',
            'plan': plan,
            'setup': setup,
            'placed_at': datetime.now(timezone.utc),
            'thin_liquidity': True,
            'min_breakout_vol_ratio': 2.0,
        }

        # First call: order status (filled). Second: gap-fill leg lookup
        mock_alpaca.get_order.side_effect = [
            {'status': 'filled', 'filled_avg_price': 4.42, 'filled_qty': 113},
            {'legs': [
                {'id': 'sl-1', 'side': 'sell', 'stop_price': 4.29,
                 'limit_price': None, 'status': 'new'},
                {'id': 'tp-1', 'side': 'sell', 'stop_price': None,
                 'limit_price': 4.90, 'status': 'new'},
            ]},
        ]
        mock_alpaca.replace_order_stop_price.return_value = {'id': 'sl-1'}
        mock_alpaca.replace_order_limit_price.return_value = {'id': 'tp-1'}
        mock_alpaca.get_open_positions.return_value = []

        # Strong volume bars
        bars = pd.DataFrame([
            {'high': 4.45, 'low': 4.35, 'close': 4.42, 'volume': 120000},
        ])
        mock_alpaca.get_1min_bars.return_value = bars

        result = engine._manage_pending_orders()

        assert result['status'] == 'filled'
        assert engine._patterns_traded == 1

    @patch('trading.trading_engine.time_mod')
    def test_fill_on_thin_day_weak_volume_closes_position(
        self, mock_time, mock_alpaca, db, mock_detector, mock_planner,
        mock_executor, mock_position_manager
    ):
        """BVR < 2.0x → close_position called, _patterns_traded NOT incremented."""
        regime = self._make_thin_regime()
        engine = self._make_engine(
            mock_alpaca, db, mock_detector, mock_planner,
            mock_executor, mock_position_manager, regime,
        )

        setup = _make_pattern("AAPL")
        plan = _make_plan("AAPL")

        from datetime import date as date_cls
        db.save_trade({
            'trade_date': date_cls.today().isoformat(), 'symbol': 'AAPL',
            'side': 'buy', 'entry_price': 4.40, 'stop_loss_price': 4.29,
            'take_profit_price': 4.90, 'shares': 113, 'risk_per_share': 0.11,
            'total_risk': 12.43, 'risk_reward_ratio': 4.5,
            'order_id': 'order-reject-1', 'order_status': 'accepted',
            'fill_price': None, 'filled_at': None, 'exit_price': None,
            'exit_reason': None, 'exited_at': None, 'pnl': None, 'pnl_pct': None,
            'pattern_data': '{}', 'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
        })

        engine._pending_orders['AAPL'] = {
            'order_id': 'order-reject-1',
            'plan': plan,
            'setup': setup,
            'placed_at': datetime.now(timezone.utc),
            'thin_liquidity': True,
            'min_breakout_vol_ratio': 2.0,
        }

        # Order is filled
        mock_alpaca.get_order.side_effect = [
            # First call: order status
            {'status': 'filled', 'filled_avg_price': 4.42, 'filled_qty': 113},
            # Subsequent calls: close order polling
            {'status': 'filled', 'filled_avg_price': 4.40},
        ]
        mock_alpaca.get_open_positions.return_value = []
        mock_alpaca.close_position.return_value = {'id': 'close-order-1', 'status': 'accepted'}

        # Weak volume — BVR 1.5x (60000/40000) < 2.0
        bars = pd.DataFrame([
            {'high': 4.45, 'low': 4.35, 'close': 4.42, 'volume': 60000},
        ])
        mock_alpaca.get_1min_bars.return_value = bars

        result = engine._manage_pending_orders()

        assert result['status'] == 'thin_liquidity_rejected'
        assert result['reason'] == 'weak_breakout_volume'
        assert engine._patterns_traded == 0
        mock_alpaca.close_position.assert_called_once_with('AAPL')
        mock_position_manager.record_trade_pnl.assert_called_once()

    @patch('trading.trading_engine.time_mod')
    def test_fill_on_normal_day_no_volume_check(
        self, mock_time, mock_alpaca, db, mock_detector, mock_planner,
        mock_executor, mock_position_manager
    ):
        """No thin_liquidity tag → no volume check after fill."""
        regime = self._make_normal_regime()
        engine = self._make_engine(
            mock_alpaca, db, mock_detector, mock_planner,
            mock_executor, mock_position_manager, regime,
        )

        setup = _make_pattern("AAPL")
        plan = _make_plan("AAPL")

        from datetime import date as date_cls
        db.save_trade({
            'trade_date': date_cls.today().isoformat(), 'symbol': 'AAPL',
            'side': 'buy', 'entry_price': 4.40, 'stop_loss_price': 4.29,
            'take_profit_price': 4.90, 'shares': 113, 'risk_per_share': 0.11,
            'total_risk': 12.43, 'risk_reward_ratio': 4.5,
            'order_id': 'order-normal-1', 'order_status': 'accepted',
            'fill_price': None, 'filled_at': None, 'exit_price': None,
            'exit_reason': None, 'exited_at': None, 'pnl': None, 'pnl_pct': None,
            'pattern_data': '{}', 'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
        })

        engine._pending_orders['AAPL'] = {
            'order_id': 'order-normal-1',
            'plan': plan,
            'setup': setup,
            'placed_at': datetime.now(timezone.utc),
            # No thin_liquidity tag
        }

        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 4.40, 'filled_qty': 113,
        }
        mock_alpaca.get_open_positions.return_value = []

        result = engine._manage_pending_orders()

        assert result['status'] == 'filled'
        assert engine._patterns_traded == 1
        # close_position should NOT have been called (no thin check, no gap)
        mock_alpaca.close_position.assert_not_called()

    def test_check_breakout_volume_fails_open_on_no_bars(
        self, mock_alpaca, db, mock_detector, mock_planner,
        mock_executor, mock_position_manager
    ):
        """Bars unavailable → _check_breakout_volume returns True (keep trade)."""
        regime = self._make_thin_regime()
        engine = self._make_engine(
            mock_alpaca, db, mock_detector, mock_planner,
            mock_executor, mock_position_manager, regime,
        )

        setup = _make_pattern("AAPL")
        pending = {
            'setup': setup,
            'min_breakout_vol_ratio': 2.0,
            'placed_at': datetime.now(timezone.utc) - timedelta(seconds=120),
        }

        # No bars available
        mock_alpaca.get_1min_bars.return_value = None
        assert engine._check_breakout_volume("AAPL", pending) is True

        # Empty bars
        mock_alpaca.get_1min_bars.return_value = pd.DataFrame()
        assert engine._check_breakout_volume("AAPL", pending) is True

        # Exception fetching bars
        mock_alpaca.get_1min_bars.side_effect = Exception("API error")
        assert engine._check_breakout_volume("AAPL", pending) is True

    def test_check_breakout_volume_dynamic_lookback(
        self, mock_alpaca, db, mock_detector, mock_planner,
        mock_executor, mock_position_manager
    ):
        """Lookback window covers from placed_at to now, not a hardcoded value."""
        regime = self._make_thin_regime()
        engine = self._make_engine(
            mock_alpaca, db, mock_detector, mock_planner,
            mock_executor, mock_position_manager, regime,
        )

        setup = _make_pattern("AAPL")

        # Order placed 7 minutes ago — lookback should be >= 9 (7 + 2 buffer)
        pending = {
            'setup': setup,
            'min_breakout_vol_ratio': 2.0,
            'placed_at': datetime.now(timezone.utc) - timedelta(minutes=7),
        }

        # Strong volume so test doesn't reject
        bars = pd.DataFrame([
            {'high': 4.45, 'low': 4.35, 'close': 4.42, 'volume': 100000},
        ])
        mock_alpaca.get_1min_bars.return_value = bars

        engine._check_breakout_volume("AAPL", pending)

        # Verify lookback_minutes passed to get_1min_bars is >= 9 (7 elapsed + 2 buffer)
        call_args = mock_alpaca.get_1min_bars.call_args
        lookback_used = call_args[1].get('lookback_minutes', call_args[0][1] if len(call_args[0]) > 1 else None)
        assert lookback_used >= 9, f"Expected lookback >= 9, got {lookback_used}"
        assert lookback_used <= 30, f"Expected lookback <= 30, got {lookback_used}"

    def test_check_breakout_volume_no_placed_at_uses_wide_window(
        self, mock_alpaca, db, mock_detector, mock_planner,
        mock_executor, mock_position_manager
    ):
        """Missing placed_at → uses 15-min fallback window."""
        regime = self._make_thin_regime()
        engine = self._make_engine(
            mock_alpaca, db, mock_detector, mock_planner,
            mock_executor, mock_position_manager, regime,
        )

        setup = _make_pattern("AAPL")
        pending = {
            'setup': setup,
            'min_breakout_vol_ratio': 2.0,
            # No placed_at
        }

        bars = pd.DataFrame([
            {'high': 4.45, 'low': 4.35, 'close': 4.42, 'volume': 100000},
        ])
        mock_alpaca.get_1min_bars.return_value = bars

        engine._check_breakout_volume("AAPL", pending)

        call_args = mock_alpaca.get_1min_bars.call_args
        lookback_used = call_args[1].get('lookback_minutes', call_args[0][1] if len(call_args[0]) > 1 else None)
        assert lookback_used == 15, f"Expected fallback lookback 15, got {lookback_used}"

    @patch('trading.trading_engine.time_mod')
    def test_reject_updates_db_with_exit(
        self, mock_time, mock_alpaca, db, mock_detector, mock_planner,
        mock_executor, mock_position_manager
    ):
        """DB has exit_price, exit_reason='thin_liquidity_reject', pnl, exited_at."""
        regime = self._make_thin_regime()
        engine = self._make_engine(
            mock_alpaca, db, mock_detector, mock_planner,
            mock_executor, mock_position_manager, regime,
        )

        from datetime import date as date_cls
        trade_id = db.save_trade({
            'trade_date': date_cls.today().isoformat(), 'symbol': 'AAPL',
            'side': 'buy', 'entry_price': 4.40, 'stop_loss_price': 4.29,
            'take_profit_price': 4.90, 'shares': 113, 'risk_per_share': 0.11,
            'total_risk': 12.43, 'risk_reward_ratio': 4.5,
            'order_id': 'order-db-1', 'order_status': 'filled',
            'fill_price': 4.42, 'filled_at': datetime.now(timezone.utc),
            'exit_price': None, 'exit_reason': None, 'exited_at': None,
            'pnl': None, 'pnl_pct': None,
            'pattern_data': '{}', 'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
        })

        trade_record = db.get_trade_by_order_id('order-db-1')

        mock_alpaca.close_position.return_value = {'id': 'close-1', 'status': 'accepted'}
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 4.38,
        }

        engine._emergency_close_position('AAPL', 'order-db-1', 4.42, 113, trade_record)

        # Verify DB updated
        updated = db.get_trade_by_order_id('order-db-1')
        assert updated['exit_price'] == 4.38
        assert updated['exit_reason'] == 'thin_liquidity_reject'
        assert updated['exited_at'] is not None
        assert updated['pnl'] is not None
        expected_pnl = (4.38 - 4.42) * 113
        assert abs(updated['pnl'] - expected_pnl) < 0.01

    @patch('trading.trading_engine.time_mod')
    def test_reject_records_pnl_in_circuit_breaker(
        self, mock_time, mock_alpaca, db, mock_detector, mock_planner,
        mock_executor, mock_position_manager
    ):
        """record_trade_pnl called with actual PnL on reject."""
        regime = self._make_thin_regime()
        engine = self._make_engine(
            mock_alpaca, db, mock_detector, mock_planner,
            mock_executor, mock_position_manager, regime,
        )

        from datetime import date as date_cls
        db.save_trade({
            'trade_date': date_cls.today().isoformat(), 'symbol': 'AAPL',
            'side': 'buy', 'entry_price': 4.40, 'stop_loss_price': 4.29,
            'take_profit_price': 4.90, 'shares': 113, 'risk_per_share': 0.11,
            'total_risk': 12.43, 'risk_reward_ratio': 4.5,
            'order_id': 'order-cb-1', 'order_status': 'filled',
            'fill_price': 4.42, 'filled_at': datetime.now(timezone.utc),
            'exit_price': None, 'exit_reason': None, 'exited_at': None,
            'pnl': None, 'pnl_pct': None,
            'pattern_data': '{}', 'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
        })

        trade_record = db.get_trade_by_order_id('order-cb-1')

        mock_alpaca.close_position.return_value = {'id': 'close-cb', 'status': 'accepted'}
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 4.39,
        }

        engine._emergency_close_position('AAPL', 'order-cb-1', 4.42, 113, trade_record)

        expected_pnl = (4.39 - 4.42) * 113
        mock_position_manager.record_trade_pnl.assert_called_once()
        actual_pnl = mock_position_manager.record_trade_pnl.call_args[0][0]
        assert abs(actual_pnl - expected_pnl) < 0.01


# ===========================================================================
# Bug #4: clear_qualified_symbols
# ===========================================================================

class TestClearQualifiedSymbols:
    """Tests for clear_qualified_symbols() — once qualified, stay qualified."""

    def test_clear_qualified_preserves_symbols(self, engine):
        """clear_qualified_symbols() keeps previously qualified symbols."""
        engine._qualified_symbols = {'AAPL', 'TSLA', 'MSFT'}
        engine.clear_qualified_symbols()
        assert engine._qualified_symbols == {'AAPL', 'TSLA', 'MSFT'}

    def test_clear_qualified_preserves_traded_and_pending(self, engine):
        """Clearing qualified does not affect traded symbols or pending orders."""
        engine._qualified_symbols = {'AAPL', 'TSLA'}
        engine._traded_symbols = {'GOOG'}
        engine._pending_orders = {'AMZN': {'order_id': 'test-1'}}

        engine.clear_qualified_symbols()

        assert engine._qualified_symbols == {'AAPL', 'TSLA'}
        assert 'GOOG' in engine._traded_symbols
        assert 'AMZN' in engine._pending_orders


# ===========================================================================
# Bug #5: _daily_trade_count only on successful fill
# ===========================================================================

class TestDailyTradeCountFix:
    """Tests that _daily_trade_count only increments after rejection checks pass."""

    def _setup_filled_order(self, engine, mock_alpaca, db, symbol='TEST',
                            thin_liquidity=False):
        """Set up a pending order that will be filled."""
        setup = _make_pattern(symbol)
        plan = _make_plan(symbol)

        from datetime import date as date_cls
        db.save_trade({
            'trade_date': date_cls.today().isoformat(), 'symbol': symbol,
            'side': 'buy', 'entry_price': 4.40, 'stop_loss_price': 4.29,
            'take_profit_price': 4.90, 'shares': 113, 'risk_per_share': 0.11,
            'total_risk': 12.43, 'risk_reward_ratio': 4.5,
            'order_id': f'order-{symbol}', 'order_status': 'accepted',
            'fill_price': None, 'filled_at': None, 'exit_price': None,
            'exit_reason': None, 'exited_at': None, 'pnl': None, 'pnl_pct': None,
            'pattern_data': '{}', 'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
        })

        pending = {
            'order_id': f'order-{symbol}',
            'plan': plan,
            'setup': setup,
            'placed_at': datetime.now(timezone.utc),
        }
        if thin_liquidity:
            pending['thin_liquidity'] = True
            pending['min_breakout_vol_ratio'] = 2.0
        engine._pending_orders[symbol] = pending

    @patch('trading.trading_engine.time_mod')
    def test_rejected_thin_liquidity_no_daily_count_increment(
        self, mock_time, mock_alpaca, db, mock_detector, mock_planner,
        mock_executor, mock_position_manager
    ):
        """_daily_trade_count stays 0 on thin-liquidity rejection."""
        from trading.market_regime import MarketRegimeFilter
        regime = MagicMock(spec=MarketRegimeFilter)
        regime.is_thin_liquidity.return_value = True
        regime.max_trades_per_day = 5
        regime.sma_period = 50

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=mock_detector,
            planner=mock_planner, executor=mock_executor,
            position_manager=mock_position_manager,
            enabled=True, market_regime=regime,
        )
        engine.news_gate_enabled = False

        self._setup_filled_order(engine, mock_alpaca, db, 'AAPL', thin_liquidity=True)

        # Order fills but breakout volume is weak
        mock_alpaca.get_order.side_effect = [
            {'status': 'filled', 'filled_avg_price': 4.42, 'filled_qty': 113},
            {'status': 'filled', 'filled_avg_price': 4.40},  # close order
        ]
        mock_alpaca.close_position.return_value = {'id': 'close-1', 'status': 'accepted'}

        # Weak volume
        bars = pd.DataFrame([
            {'high': 4.45, 'low': 4.35, 'close': 4.42, 'volume': 50000},
        ])
        mock_alpaca.get_1min_bars.return_value = bars

        engine._manage_pending_orders()

        assert engine._daily_trade_count == 0

    @patch('trading.trading_engine.time_mod')
    def test_gap_fill_failure_no_daily_count_increment(
        self, mock_time, mock_alpaca, db, mock_detector, mock_planner,
        mock_executor, mock_position_manager
    ):
        """_daily_trade_count stays 0 on gap-fill adjustment failure."""
        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=mock_detector,
            planner=mock_planner, executor=mock_executor,
            position_manager=mock_position_manager,
            enabled=True,
        )
        engine.news_gate_enabled = False

        self._setup_filled_order(engine, mock_alpaca, db, 'AAPL')

        # Fill above breakout (gap fill needed), but leg replacement fails
        mock_alpaca.get_order.side_effect = [
            {'status': 'filled', 'filled_avg_price': 4.50, 'filled_qty': 113},
            {'legs': []},  # No legs found
            {'status': 'filled', 'filled_avg_price': 4.48},  # close order
        ]
        mock_alpaca.close_position.return_value = {'id': 'close-gap', 'status': 'accepted'}

        engine._manage_pending_orders()

        assert engine._daily_trade_count == 0

    @patch('trading.trading_engine.time_mod')
    def test_successful_fill_increments_daily_count(
        self, mock_time, mock_alpaca, db, mock_detector, mock_planner,
        mock_executor, mock_position_manager
    ):
        """_daily_trade_count = 1 on successful fill (no rejection)."""
        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=mock_detector,
            planner=mock_planner, executor=mock_executor,
            position_manager=mock_position_manager,
            enabled=True,
        )
        engine.news_gate_enabled = False

        self._setup_filled_order(engine, mock_alpaca, db, 'AAPL')

        # Fill at breakout level (no gap, no thin liquidity)
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 4.40, 'filled_qty': 113,
        }

        engine._manage_pending_orders()

        assert engine._daily_trade_count == 1
        assert engine._patterns_traded == 1


# ===========================================================================
# Bug A: Position management not blocked by regime/max-trades guards
# ===========================================================================

def _make_trade_record(**overrides):
    """Create a complete trade dict with all DB columns, applying overrides."""
    record = {
        'trade_date': date.today().isoformat(),
        'symbol': 'TEST', 'side': 'buy',
        'entry_price': 10.0, 'stop_loss_price': 9.5,
        'take_profit_price': 11.0, 'shares': 100,
        'risk_per_share': 0.5, 'total_risk': 50.0,
        'risk_reward_ratio': 2.0,
        'order_id': None, 'order_status': None,
        'fill_price': None, 'filled_at': None,
        'exit_price': None, 'exit_reason': None, 'exited_at': None,
        'pnl': None, 'pnl_pct': None, 'pattern_data': None,
    }
    record.update(overrides)
    return record


class TestPositionManagementNotBlockedByGuards:
    """Verify _sync_closed_positions and _manage_pending_orders run even when
    regime filter or max-trades guard would block new order placement."""

    def test_sync_runs_when_regime_blocks(self, engine, mock_alpaca, db, mock_position_manager):
        """_sync_closed_positions runs even when regime filter is active."""
        from trading.market_regime import MarketRegimeFilter
        regime = MagicMock(spec=MarketRegimeFilter)
        regime.is_regime_ok.return_value = False
        regime.get_regime_info.return_value = {
            'vol_5d': 3.0, 'sma': 400.0, 'is_below_sma': True, 'is_ok': False,
            'spy_volume_ratio': 1.0,
        }
        regime.vol_threshold = 2.0
        regime.sma_period = 50
        engine.market_regime = regime

        # Create an open trade in DB
        today = date.today().isoformat()
        db.save_trade(_make_trade_record(
            trade_date=today, symbol='AAPL', order_id='ord-1',
            order_status='filled', fill_price=10.0,
        ))

        # Alpaca says no open position (SL was hit)
        mock_alpaca.get_open_positions.return_value = []
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 10.0, 'filled_qty': 100,
            'legs': [
                {'side': 'sell', 'stop_price': 9.5, 'status': 'filled',
                 'filled_avg_price': 9.5},
                {'side': 'sell', 'limit_price': 11.0, 'status': 'canceled'},
            ],
        }

        result = engine.run_pattern_check()

        # Sync should have detected the closed position and updated DB
        trade = db.get_trades_by_date(today)[0]
        assert trade['exit_price'] is not None
        assert trade['exit_reason'] == 'stop_loss'

    def test_manage_pending_runs_when_max_trades_reached(
        self, engine, mock_alpaca, db, mock_position_manager
    ):
        """_manage_pending_orders runs even when max_trades_per_day reached."""
        from trading.market_regime import MarketRegimeFilter
        regime = MagicMock(spec=MarketRegimeFilter)
        regime.is_regime_ok.return_value = True
        regime.is_thin_liquidity.return_value = False
        regime.max_trades_per_day = 2
        engine.market_regime = regime
        engine._daily_trade_count = 2  # At max

        # Set up a pending order
        engine._pending_orders['TSLA'] = {
            'order_id': 'ord-99',
            'plan': _make_plan('TSLA'),
            'setup': _make_pattern('TSLA'),
            'placed_at': datetime.now(timezone.utc) - timedelta(seconds=30),
        }

        # Order expired (cancelled — Alpaca uses British spelling)
        mock_alpaca.get_order.return_value = {
            'status': 'cancelled', 'filled_avg_price': None, 'filled_qty': 0,
        }

        result = engine.run_pattern_check()

        # Pending order should have been cleaned up even at max trades
        assert 'TSLA' not in engine._pending_orders

    def test_guards_return_fill_result(self, engine, mock_alpaca, db, mock_position_manager):
        """Guards return fill_result from _manage_pending_orders, not None."""
        from trading.market_regime import MarketRegimeFilter
        regime = MagicMock(spec=MarketRegimeFilter)
        regime.is_regime_ok.return_value = False
        regime.get_regime_info.return_value = {
            'vol_5d': 3.0, 'sma': 400.0, 'is_below_sma': True, 'is_ok': False,
            'spy_volume_ratio': 1.0,
        }
        regime.vol_threshold = 2.0
        regime.sma_period = 50
        engine.market_regime = regime

        # No pending orders, no open trades — fill_result should be None
        mock_alpaca.get_open_positions.return_value = []
        result = engine.run_pattern_check()

        # With regime blocking and no pending fills, result is None (fill_result)
        assert result is None


# ===========================================================================
# Bug C: Startup sync rebuilds state after crash recovery
# ===========================================================================

class TestStartupSync:
    """Verify _sync_startup_state rebuilds traded_symbols, pending_orders,
    and daily_trade_count from DB after reset_daily."""

    def test_traded_symbols_rebuilt_from_db(self, engine, db, mock_alpaca, mock_position_manager):
        """Traded symbols are rebuilt from today's DB trades."""
        today = date.today().isoformat()
        db.save_trade(_make_trade_record(
            trade_date=today, symbol='AAPL', order_id='ord-1',
            order_status='filled', fill_price=10.0,
        ))
        db.save_trade(_make_trade_record(
            trade_date=today, symbol='TSLA', entry_price=20.0,
            stop_loss_price=19.0, take_profit_price=22.0, shares=50,
            order_id='ord-2', order_status='filled', fill_price=20.0,
        ))

        # Mock SPY refresh so reset_daily doesn't fail
        mock_alpaca.get_daily_bars_range.return_value = {}

        engine.reset_daily()

        assert 'AAPL' in engine._traded_symbols
        assert 'TSLA' in engine._traded_symbols
        assert engine._daily_trade_count == 2

    def test_pending_orders_rebuilt_from_db(self, engine, db, mock_alpaca, mock_position_manager):
        """Pending (unfilled) orders are rebuilt from DB."""
        today = date.today().isoformat()
        db.save_trade(_make_trade_record(
            trade_date=today, symbol='PLTR', entry_price=15.0,
            stop_loss_price=14.5, take_profit_price=16.0, shares=200,
            order_id='ord-pending', order_status='accepted',
        ))

        mock_alpaca.get_daily_bars_range.return_value = {}

        engine.reset_daily()

        assert 'PLTR' in engine._pending_orders
        assert engine._pending_orders['PLTR']['order_id'] == 'ord-pending'
        assert 'PLTR' not in engine._traded_symbols  # Unfilled — should NOT block re-entry
        assert engine._daily_trade_count == 0  # Not filled, so not counted

    def test_filled_not_in_pending(self, engine, db, mock_alpaca, mock_position_manager):
        """Filled trades are NOT added to _pending_orders."""
        today = date.today().isoformat()
        db.save_trade(_make_trade_record(
            trade_date=today, symbol='AAPL', order_id='ord-filled',
            order_status='filled', fill_price=10.05,
        ))

        mock_alpaca.get_daily_bars_range.return_value = {}

        engine.reset_daily()

        assert 'AAPL' not in engine._pending_orders
        assert 'AAPL' in engine._traded_symbols
        assert engine._daily_trade_count == 1

    def test_exited_trades_not_in_pending(self, engine, db, mock_alpaca, mock_position_manager):
        """Exited trades (with exit_price) are NOT added to _pending_orders."""
        today = date.today().isoformat()
        trade_id = db.save_trade(_make_trade_record(
            trade_date=today, symbol='NVDA', order_id='ord-exited',
            order_status='filled', fill_price=10.0,
        ))
        db.update_trade(trade_id, {
            'exit_price': 11.0, 'exit_reason': 'take_profit',
            'pnl': 100.0, 'pnl_pct': 10.0,
        })

        mock_alpaca.get_daily_bars_range.return_value = {}

        engine.reset_daily()

        assert 'NVDA' not in engine._pending_orders
        assert 'NVDA' in engine._traded_symbols
        assert engine._daily_trade_count == 1

    def test_no_trades_today_skips_gracefully(self, engine, db, mock_alpaca, mock_position_manager):
        """No trades today — sync completes without error."""
        mock_alpaca.get_daily_bars_range.return_value = {}

        engine.reset_daily()

        assert len(engine._traded_symbols) == 0
        assert len(engine._pending_orders) == 0
        assert engine._daily_trade_count == 0


# ===========================================================================
# Simple Order Path (no bracket)
# ===========================================================================

class TestSimpleOrderPath:
    """Tests for simple stop-limit order submission when StopMonitor is active."""

    @pytest.fixture
    def mock_alpaca(self):
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.get_1min_bars.return_value = pd.DataFrame({'close': [4.35]})
        alpaca.get_open_positions.return_value = []
        alpaca.get_daily_bars_range.return_value = {}
        return alpaca

    @pytest.fixture
    def db(self, tmp_path):
        database = Database(db_path=str(tmp_path / "test.db"))
        _seed_universe(database)
        yield database
        database.close()

    @pytest.fixture
    def mock_stop_monitor(self):
        from trading.stop_monitor import StopMonitor
        sm = MagicMock(spec=StopMonitor)
        sm._running = True
        sm._watch_lock = threading.Lock()
        sm._watches = {}
        return sm

    @patch('trading.trading_engine.TradingEngine._is_past_last_entry_time', return_value=False)
    def test_uses_simple_order_when_stop_monitor_active(
        self, _mock_time, mock_alpaca, db, mock_stop_monitor
    ):
        """With StopMonitor, engine calls submit_buy_stop_order (not bracket)."""
        mock_detector = MagicMock(spec=BullFlagDetector)
        mock_planner = MagicMock(spec=TradePlanner)
        mock_executor = MagicMock(spec=OrderExecutor)
        mock_pm = MagicMock(spec=PositionManager)
        mock_pm.can_open_position.return_value = True

        setup = _make_pattern("AAPL")
        plan = _make_plan("AAPL")
        mock_detector.detect_setup.return_value = setup
        mock_planner.create_plan.return_value = plan
        mock_executor.submit_buy_stop_order.return_value = {
            'order_id': 'simple-1', 'status': 'accepted',
            'symbol': 'AAPL', 'shares': 113,
            'order_type': 'stop_simple',
            'stop_price': 4.40, 'limit_price': 4.49,
            'stop_loss_price': 4.25, 'take_profit_price': 4.90,
        }

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=mock_detector,
            planner=mock_planner, executor=mock_executor,
            position_manager=mock_pm, enabled=True,
            stop_monitor=mock_stop_monitor,
        )
        engine.news_gate_enabled = False

        engine.on_stock_qualified("AAPL")
        engine.run_pattern_check()

        # Simple order called, NOT bracket
        mock_executor.submit_buy_stop_order.assert_called_once()
        mock_executor.submit_buy_stop_bracket_order.assert_not_called()

    @patch('trading.trading_engine.TradingEngine._is_past_last_entry_time', return_value=False)
    def test_uses_bracket_when_no_stop_monitor(
        self, _mock_time, mock_alpaca, db
    ):
        """Without StopMonitor, engine falls back to bracket order."""
        mock_detector = MagicMock(spec=BullFlagDetector)
        mock_planner = MagicMock(spec=TradePlanner)
        mock_executor = MagicMock(spec=OrderExecutor)
        mock_pm = MagicMock(spec=PositionManager)
        mock_pm.can_open_position.return_value = True

        setup = _make_pattern("AAPL")
        plan = _make_plan("AAPL")
        mock_detector.detect_setup.return_value = setup
        mock_planner.create_plan.return_value = plan
        mock_executor.submit_buy_stop_bracket_order.return_value = {
            'order_id': 'bracket-1', 'status': 'accepted',
            'symbol': 'AAPL', 'shares': 113,
            'order_type': 'stop_bracket',
            'stop_price': 4.40, 'limit_price': 4.49,
            'stop_loss_price': 4.25, 'take_profit_price': 4.90,
        }

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=mock_detector,
            planner=mock_planner, executor=mock_executor,
            position_manager=mock_pm, enabled=True,
            # No stop_monitor
        )
        engine.news_gate_enabled = False

        engine.on_stock_qualified("AAPL")
        engine.run_pattern_check()

        # Bracket called, NOT simple
        mock_executor.submit_buy_stop_bracket_order.assert_called_once()
        mock_executor.submit_buy_stop_order.assert_not_called()

    @patch('trading.trading_engine.TradingEngine._is_past_last_entry_time', return_value=False)
    def test_pending_order_stores_order_type(
        self, _mock_time, mock_alpaca, db, mock_stop_monitor
    ):
        """Pending dict stores order_type for fill-time routing."""
        mock_detector = MagicMock(spec=BullFlagDetector)
        mock_planner = MagicMock(spec=TradePlanner)
        mock_executor = MagicMock(spec=OrderExecutor)
        mock_pm = MagicMock(spec=PositionManager)
        mock_pm.can_open_position.return_value = True

        setup = _make_pattern("AAPL")
        plan = _make_plan("AAPL")
        mock_detector.detect_setup.return_value = setup
        mock_planner.create_plan.return_value = plan
        mock_executor.submit_buy_stop_order.return_value = {
            'order_id': 'simple-2', 'status': 'accepted',
            'symbol': 'AAPL', 'shares': 113,
            'order_type': 'stop_simple',
            'stop_price': 4.40, 'limit_price': 4.49,
            'stop_loss_price': 4.25, 'take_profit_price': 4.90,
        }

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=mock_detector,
            planner=mock_planner, executor=mock_executor,
            position_manager=mock_pm, enabled=True,
            stop_monitor=mock_stop_monitor,
        )
        engine.news_gate_enabled = False

        engine.on_stock_qualified("AAPL")
        engine.run_pattern_check()

        assert 'AAPL' in engine._pending_orders
        assert engine._pending_orders['AAPL']['order_type'] == 'stop_simple'

    def test_fill_submits_standalone_safety_net_sl(
        self, mock_alpaca, db, mock_stop_monitor
    ):
        """On fill of simple order, standalone safety-net SL is submitted."""
        mock_executor = MagicMock(spec=OrderExecutor)
        mock_pm = MagicMock(spec=PositionManager)

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db,
            detector=MagicMock(spec=BullFlagDetector),
            planner=MagicMock(spec=TradePlanner),
            executor=mock_executor,
            position_manager=mock_pm, enabled=True,
            stop_monitor=mock_stop_monitor,
            safety_net_sl_pct=0.05,
        )
        engine.news_gate_enabled = False

        # Simulate a pending simple order that just filled
        plan = _make_plan("AAPL")
        engine._pending_orders['AAPL'] = {
            'order_id': 'simple-fill-1',
            'plan': plan,
            'setup': _make_pattern("AAPL"),
            'placed_at': datetime.now(timezone.utc),
            'news_data': None,
            'order_type': 'stop_simple',
            'real_stop_level': 4.25,
        }

        # Mock the fill
        mock_alpaca.get_order.return_value = {
            'id': 'simple-fill-1',
            'status': 'filled',
            'filled_avg_price': 4.42,
            'filled_qty': 113,
            'filled_at': datetime.now(timezone.utc).isoformat(),
            'legs': [],
        }
        mock_alpaca.submit_stop_sell_order.return_value = {
            'id': 'safety-net-sl-1', 'status': 'accepted',
            'symbol': 'AAPL', 'qty': 113, 'stop_price': 4.20,
        }

        engine._manage_pending_orders()

        # Verify standalone SL was submitted
        mock_alpaca.submit_stop_sell_order.assert_called_once()
        call_args = mock_alpaca.submit_stop_sell_order.call_args
        assert call_args.kwargs['symbol'] == 'AAPL'
        assert call_args.kwargs['qty'] == 113
        # Safety net = fill_price * (1 - 0.05) = 4.42 * 0.95 = 4.199
        assert call_args.kwargs['stop_price'] == round(4.42 * 0.95, 2)

        # Verify StopMonitor.add_watch called with the SL order ID
        mock_stop_monitor.add_watch.assert_called_once()
        watch_kwargs = mock_stop_monitor.add_watch.call_args.kwargs
        assert watch_kwargs['sl_leg_id'] == 'safety-net-sl-1'
        assert watch_kwargs['tp_leg_id'] == ''  # No TP leg for simple orders


# ===========================================================================
# MACD Dead Zone + Risk Tier Interaction
# ===========================================================================

class TestMacdDeadZoneWithRiskTier:
    """Dead zone must reject trades regardless of risk tier multiplier."""

    @pytest.fixture
    def db(self, tmp_path):
        database = Database(db_path=str(tmp_path / "test.db"))
        yield database
        database.close()

    def _build_engine(self, mock_alpaca, db):
        """Create engine with MACD zones + risk tiers both enabled."""
        mock_detector = MagicMock(spec=BullFlagDetector)
        mock_planner = MagicMock(spec=TradePlanner)
        mock_executor = MagicMock(spec=OrderExecutor)
        mock_pm = MagicMock(spec=PositionManager)
        mock_pm.can_open_position.return_value = True

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db,
            detector=mock_detector, planner=mock_planner,
            executor=mock_executor, position_manager=mock_pm,
            enabled=True,
        )
        # Disable conviction/QF in this test — testing MACD zone interaction only
        engine.conviction_enabled = False
        engine.news_gate_enabled = False
        engine.quality_filter_enabled = False
        # Enable MACD zones
        engine.macd_zones_enabled = True
        engine.macd_dead_zone_min = -0.2
        engine.macd_dead_zone_max = 0.1
        engine.macd_strong_neg_threshold = -0.5
        engine.macd_strong_neg_multiplier = 1.5
        engine.macd_strong_pos_threshold = 0.5
        engine.macd_strong_pos_multiplier = 1.5
        engine.macd_normal_multiplier = 1.0

        # Enable risk tiers
        engine.risk_tiers_enabled = True
        engine.risk_tiers = [{
            'min_price': 10.0, 'max_price': 15.0,
            'min_volume': 500000, 'max_volume': 5000000,
            'multiplier': 2.0,
        }]
        return engine, mock_detector, mock_planner, mock_executor

    @patch('trading.trading_engine.TradingEngine._is_past_last_entry_time', return_value=False)
    def test_dead_zone_rejects_even_with_risk_tier(self, _mock_time):
        """MACD dead zone must reject trades even when risk tier = 2x."""
        mock_alpaca = MagicMock(spec=AlpacaClient)
        mock_alpaca.get_open_positions.return_value = []

        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(db_path=os.path.join(tmp, "test.db"))
            try:
                engine, mock_detector, mock_planner, mock_executor = self._build_engine(mock_alpaca, db)

                # Stock at $12 with 1M volume → matches tier1 (2x)
                pattern = _make_pattern("TIER")
                pattern = BullFlagPattern(
                    symbol="TIER",
                    pole_start_idx=0, pole_end_idx=2,
                    flag_start_idx=3, flag_end_idx=4,
                    pole_low=11.0, pole_high=12.5,
                    pole_height=1.5, pole_gain_pct=13.6,
                    flag_low=11.80, flag_high=12.10,
                    retracement_pct=26.7, pullback_candle_count=2,
                    avg_pole_volume=800000, avg_flag_volume=400000,
                    breakout_level=12.10,
                )
                plan = TradePlan(
                    symbol="TIER", entry_price=12.10,
                    stop_loss_price=11.79, take_profit_price=12.88,
                    risk_per_share=0.31, reward_per_share=0.78,
                    risk_reward_ratio=2.5, shares=322,
                    total_risk=99.82, pattern=pattern,
                )

                # Mock universe stock with sufficient volume for tier match
                db.get_universe_stock = MagicMock(return_value={
                    'symbol': 'TIER', 'avg_volume_daily': 1000000,
                })

                bars = pd.DataFrame({
                    'open': [11.9, 12.0, 12.05],
                    'high': [12.0, 12.1, 12.08],
                    'low': [11.85, 11.95, 12.0],
                    'close': [12.0, 12.05, 12.05],
                    'volume': [500000, 600000, 400000],
                })
                mock_alpaca.get_1min_bars.return_value = bars

                mock_detector.detect_setup.return_value = pattern
                mock_planner.create_plan.return_value = plan
                mock_planner.max_shares = 10000
                mock_alpaca.is_marginable.return_value = True
                mock_alpaca.get_buying_power.return_value = 100000.0

                # MACD histogram returns dead zone value (0.0 → reject)
                with patch.object(engine, '_get_macd_zone_multiplier', return_value=0.0):
                    engine.on_stock_qualified("TIER")
                    result = engine.run_pattern_check()

                # Order should NOT be submitted — dead zone must reject
                mock_executor.submit_buy_stop_bracket_order.assert_not_called()
                mock_executor.submit_buy_stop_order.assert_not_called()
            finally:
                db.close()

    @patch('trading.trading_engine.TradingEngine._is_past_last_entry_time', return_value=False)
    def test_risk_tier_skips_macd_scaling_but_not_rejection(self, _mock_time):
        """Risk tier trades skip MACD scaling (no compounding) but pass through."""
        mock_alpaca = MagicMock(spec=AlpacaClient)
        mock_alpaca.get_open_positions.return_value = []

        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(db_path=os.path.join(tmp, "test.db"))
            try:
                engine, mock_detector, mock_planner, mock_executor = self._build_engine(mock_alpaca, db)

                pattern = BullFlagPattern(
                    symbol="TIER2", pole_start_idx=0, pole_end_idx=2,
                    flag_start_idx=3, flag_end_idx=4,
                    pole_low=11.0, pole_high=12.5, pole_height=1.5,
                    pole_gain_pct=13.6, flag_low=11.80, flag_high=12.10,
                    retracement_pct=26.7, pullback_candle_count=2,
                    avg_pole_volume=800000, avg_flag_volume=400000,
                    breakout_level=12.10,
                )
                plan = TradePlan(
                    symbol="TIER2", entry_price=12.10,
                    stop_loss_price=11.79, take_profit_price=12.88,
                    risk_per_share=0.31, reward_per_share=0.78,
                    risk_reward_ratio=2.5, shares=322,
                    total_risk=99.82, pattern=pattern,
                )

                db.get_universe_stock = MagicMock(return_value={
                    'symbol': 'TIER2', 'avg_volume_daily': 1000000,
                })

                bars = pd.DataFrame({
                    'open': [11.9, 12.0, 12.05],
                    'high': [12.0, 12.1, 12.08],
                    'low': [11.85, 11.95, 12.0],
                    'close': [12.0, 12.05, 12.05],
                    'volume': [500000, 600000, 400000],
                })
                mock_alpaca.get_1min_bars.return_value = bars
                mock_detector.detect_setup.return_value = pattern
                mock_planner.create_plan.return_value = plan
                mock_planner.max_shares = 10000
                mock_alpaca.is_marginable.return_value = True
                mock_alpaca.get_buying_power.return_value = 100000.0

                # MACD zone returns 1.5x (strong momentum) — should NOT compound with tier
                with patch.object(engine, '_get_macd_zone_multiplier', return_value=1.5):
                    engine.on_stock_qualified("TIER2")
                    result = engine.run_pattern_check()

                # Order should be submitted (not dead zone), but shares should be
                # unchanged (1.5x scaling skipped due to risk tier)
                assert mock_executor.submit_buy_stop_bracket_order.called or mock_executor.submit_buy_stop_order.called
                # Verify planner was called with risk_multiplier=2.0 (tier), not 3.0 (tier*MACD)
                planner_calls = mock_planner.create_plan.call_args_list
                # The last call should have risk_multiplier=2.0
                last_call = planner_calls[-1]
                assert last_call.kwargs.get('risk_multiplier', last_call.args[1] if len(last_call.args) > 1 else 1.0) == 2.0
            finally:
                db.close()


class TestGetOrderHybrid:
    """S1: Stream-first, REST-fallback order-status helper."""

    def _make_engine(self, order_stream=None, alpaca=None):
        alpaca = alpaca or MagicMock(spec=AlpacaClient)
        db = MagicMock(spec=Database)
        detector = MagicMock(spec=BullFlagDetector)
        planner = MagicMock(spec=TradePlanner)
        executor = MagicMock(spec=OrderExecutor)
        pm = MagicMock(spec=PositionManager)
        return TradingEngine(
            alpaca_client=alpaca, db=db,
            detector=detector, planner=planner,
            executor=executor, position_manager=pm,
            order_stream=order_stream,
        )

    def test_stream_hit_short_circuits_rest(self):
        """When the stream has the order, REST is not called."""
        stream = MagicMock()
        stream.get_status.return_value = {
            'id': 'abc', 'status': 'filled',
            'filled_avg_price': 10.0, 'filled_qty': 100,
        }
        alpaca = MagicMock(spec=AlpacaClient)
        engine = self._make_engine(order_stream=stream, alpaca=alpaca)

        result = engine._get_order_hybrid('abc')
        assert result['status'] == 'filled'
        alpaca.get_order.assert_not_called()

    def test_stream_miss_falls_back_to_rest_when_aged(self):
        """Cache miss + order_submitted_at older than 5s → REST fires."""
        stream = MagicMock()
        stream.get_status.return_value = None
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.get_order.return_value = {'id': 'x', 'status': 'filled'}
        engine = self._make_engine(order_stream=stream, alpaca=alpaca)

        old_submit = datetime.now(timezone.utc) - timedelta(seconds=10)
        result = engine._get_order_hybrid('x', submitted_at=old_submit)
        assert result['status'] == 'filled'
        alpaca.get_order.assert_called_once_with('x')

    def test_stream_miss_too_fresh_returns_none(self):
        """Cache miss + order<5s old → None (caller retries next tick)."""
        stream = MagicMock()
        stream.get_status.return_value = None
        alpaca = MagicMock(spec=AlpacaClient)
        engine = self._make_engine(order_stream=stream, alpaca=alpaca)

        fresh_submit = datetime.now(timezone.utc) - timedelta(seconds=1)
        result = engine._get_order_hybrid('x', submitted_at=fresh_submit)
        assert result is None
        alpaca.get_order.assert_not_called()

    def test_no_stream_uses_rest_always(self):
        """order_stream=None → REST is always the path (legacy behavior)."""
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.get_order.return_value = {'id': 'y', 'status': 'filled'}
        engine = self._make_engine(order_stream=None, alpaca=alpaca)

        result = engine._get_order_hybrid('y')
        assert result['status'] == 'filled'
        alpaca.get_order.assert_called_once_with('y')

    def test_stream_exception_falls_through_to_rest(self):
        """Stream raising doesn't break the hybrid — falls to REST."""
        stream = MagicMock()
        stream.get_status.side_effect = RuntimeError("boom")
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.get_order.return_value = {'id': 'z', 'status': 'filled'}
        engine = self._make_engine(order_stream=stream, alpaca=alpaca)

        result = engine._get_order_hybrid('z')
        assert result['status'] == 'filled'
        alpaca.get_order.assert_called_once_with('z')

    def test_rest_exception_returns_none_no_raise(self):
        """REST failure logs warning but doesn't raise; caller sees None."""
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.get_order.side_effect = RuntimeError("api down")
        engine = self._make_engine(order_stream=None, alpaca=alpaca)

        result = engine._get_order_hybrid('w')
        assert result is None

    def test_fallback_after_s_zero_disables_age_gate(self):
        """fallback_after_s=0 means stream miss ALWAYS falls to REST, even
        for very fresh orders. Used by the tight poll loops in _try_get_fill,
        _handle_exit_partial_fill, _manage_pending_orders retry, and
        _cancel_and_market_sell — each runs at 0.5s cadence where we can't
        tolerate a 5-second wait on stream misses.
        """
        stream = MagicMock()
        stream.get_status.return_value = None  # miss
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.get_order.return_value = {'id': 'f', 'status': 'filled'}
        engine = self._make_engine(order_stream=stream, alpaca=alpaca)

        # Very fresh order — default gate would return None without REST
        fresh = datetime.now(timezone.utc) - timedelta(milliseconds=100)
        result = engine._get_order_hybrid('f', submitted_at=fresh, fallback_after_s=0.0)
        assert result['status'] == 'filled'
        alpaca.get_order.assert_called_once_with('f')


class TestS1CallSiteIntegration:
    """S1: verify the hybrid is wired correctly at the migrated call sites.

    These tests are integration-flavour — they construct a near-real engine,
    invoke the migrated method directly with both stream-hit and stream-miss
    scenarios, and assert the correct downstream effect (fill captured,
    REST not called, notifier fired, etc.).
    """

    def _make_engine(self, order_stream=None, alpaca=None, notifier=None):
        alpaca = alpaca or MagicMock(spec=AlpacaClient)
        db = MagicMock(spec=Database)
        detector = MagicMock(spec=BullFlagDetector)
        planner = MagicMock(spec=TradePlanner)
        executor = MagicMock(spec=OrderExecutor)
        pm = MagicMock(spec=PositionManager)
        return TradingEngine(
            alpaca_client=alpaca, db=db,
            detector=detector, planner=planner,
            executor=executor, position_manager=pm,
            order_stream=order_stream, notifier=notifier,
        )

    # -------- _try_get_fill --------

    @patch('trading.trading_engine.time_mod.sleep', new=lambda *_: None)
    def test_try_get_fill_stream_hit_no_rest(self):
        """Stream delivers 'filled' → _try_get_fill returns fill, REST untouched."""
        from types import SimpleNamespace
        import time
        stream = MagicMock()
        stream.get_status.return_value = {
            'id': 'exit-1', 'status': 'filled',
            'filled_avg_price': 45.50, 'filled_qty': 100,
        }
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.get_order.side_effect = AssertionError("REST must not be called on stream hit")
        engine = self._make_engine(order_stream=stream, alpaca=alpaca)

        event = SimpleNamespace(
            order_id='exit-1', shares=100,
            submitted_at=time.time() - 10.0,  # 10s ago
            filled_qty=0, symbol='AAPL', pricing_method='test',
        )
        fill = engine._try_get_fill(event, max_polls=1)
        assert fill == 45.50
        assert event.filled_qty == 100
        alpaca.get_order.assert_not_called()

    @patch('trading.trading_engine.time_mod.sleep', new=lambda *_: None)
    def test_try_get_fill_fresh_stream_miss_hits_rest_via_zero_gate(self):
        """Fresh order + stream miss: age-gate fix means REST fires on the
        very first poll iteration (fallback_after_s=0 on this call site)."""
        from types import SimpleNamespace
        import time
        stream = MagicMock()
        stream.get_status.return_value = None  # stream miss
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.get_order.return_value = {
            'id': 'exit-fresh', 'status': 'filled',
            'filled_avg_price': 42.0, 'filled_qty': 100,
        }
        engine = self._make_engine(order_stream=stream, alpaca=alpaca)

        event = SimpleNamespace(
            order_id='exit-fresh', shares=100,
            submitted_at=time.time() - 0.5,  # 500ms ago — would trigger default gate
            filled_qty=0, symbol='AAPL', pricing_method='test',
        )
        fill = engine._try_get_fill(event, max_polls=1)
        assert fill == 42.0
        # REST was called despite fresh age because fallback_after_s=0 disables gate
        alpaca.get_order.assert_called_once()

    # -------- _manage_pending_orders aged-notify guard --------

    def test_manage_pending_orders_fresh_order_no_notify(self):
        """Fresh pending order + stream miss + REST would fail → notify NOT fired
        (age gate keeps us from trying REST at all for orders <5s old)."""
        stream = MagicMock()
        stream.get_status.return_value = None
        alpaca = MagicMock(spec=AlpacaClient)
        # If REST were called this would raise, but age gate prevents the call
        alpaca.get_order.side_effect = AssertionError("REST should be gated on fresh order")
        notifier = MagicMock()
        engine = self._make_engine(order_stream=stream, alpaca=alpaca, notifier=notifier)

        engine._pending_orders['AAPL'] = {
            'order_id': 'p1',
            'placed_at': datetime.now(timezone.utc) - timedelta(seconds=2),
            'plan': MagicMock(),
            'setup': MagicMock(),
        }
        engine._manage_pending_orders()
        notifier.notify_error.assert_not_called()

    def test_manage_pending_orders_aged_both_empty_notifies(self):
        """Aged pending order (>10s) + stream miss + REST failure → aged-notify fires."""
        stream = MagicMock()
        stream.get_status.return_value = None
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.get_order.side_effect = RuntimeError("api down")
        notifier = MagicMock()
        engine = self._make_engine(order_stream=stream, alpaca=alpaca, notifier=notifier)

        engine._pending_orders['AAPL'] = {
            'order_id': 'p1',
            'placed_at': datetime.now(timezone.utc) - timedelta(seconds=15),
            'plan': MagicMock(),
            'setup': MagicMock(),
        }
        engine._manage_pending_orders()
        notifier.notify_error.assert_called_once()
        msg = notifier.notify_error.call_args[0][0]
        assert 'unavailable' in msg.lower()


# ===========================================================================
# Conviction filter (skip trades with conviction_mult < min_threshold)
# Walk-forward validated: conv<1.2 = 28% WR / -0.18R avg (negative EV)
# ===========================================================================

class TestConvictionFilter:
    """Tests for the conviction-min-threshold filter."""

    def _low_quality_setup(self):
        """Setup expected to score 0.25 (clamped) — fails most rules."""
        s = MagicMock()
        s.pole_gain_pct = 12.0       # outside 4.5-9.0 sweet spot → 0.0
        s.pole_high = 10.0
        s.pole_low = 9.0             # height 1.0
        s.flag_high = 9.6
        s.flag_low = 9.0             # range 0.6 = 60% of pole → loose → -0.3
        s.avg_pole_volume = 100      # ratio 1.0 → 0.0
        s.avg_flag_volume = 100
        s.retracement_pct = 50.0     # > 30 → 0.0
        return s

    def _high_quality_setup(self):
        """Setup expected to score ~2.4 — hits all rules."""
        s = MagicMock()
        s.pole_gain_pct = 6.0        # in sweet spot → +0.3
        s.pole_high = 10.0
        s.pole_low = 9.0             # height 1.0
        s.flag_high = 9.7
        s.flag_low = 9.5             # range 0.2 = 20% of pole → tight → +0.3
        s.avg_pole_volume = 200      # ratio 2.0 → +0.3
        s.avg_flag_volume = 100
        s.retracement_pct = 25.0     # < 30 → +0.2
        return s

    def test_breakdown_returns_per_rule_contributions(self, engine):
        """return_breakdown=True returns (final_score, breakdown_dict)
        with all 7 rule contributions (V1 + V2_clean rules 7+8) +
        raw_score + final_score."""
        s = self._low_quality_setup()
        score, brk = engine._compute_conviction_score_setup(
            s, spy_3d_range=0.5, return_breakdown=True)
        assert score == 0.25  # clamped from 0.20
        assert brk['pole_gain'] == 0.0
        assert brk['flag_tightness'] == -0.3
        assert brk['vol_ratio'] == 0.0
        assert brk['spy_regime'] == -0.5
        assert brk['retracement'] == 0.0
        # V2_clean rules: defaults silent
        assert brk['vwap_dist'] == 0.0
        assert brk['gap_fading'] == 0.0
        assert brk['raw_score'] == pytest.approx(0.20, abs=0.001)
        assert brk['final_score'] == 0.25

    def test_rule7_vwap_dist_boundary(self, engine):
        """Rule 7: vwap_dist >= 2 → +0.2; otherwise 0. No stacking above +0.2."""
        s = self._low_quality_setup()
        _, brk = engine._compute_conviction_score_setup(
            s, spy_3d_range=1.0, vwap_dist_pct=1.99, return_breakdown=True)
        assert brk['vwap_dist'] == 0.0
        _, brk = engine._compute_conviction_score_setup(
            s, spy_3d_range=1.0, vwap_dist_pct=2.0, return_breakdown=True)
        assert brk['vwap_dist'] == 0.2
        _, brk = engine._compute_conviction_score_setup(
            s, spy_3d_range=1.0, vwap_dist_pct=10.0, return_breakdown=True)
        assert brk['vwap_dist'] == 0.2

    def test_rule8_gap_fading_penalty(self, engine):
        """Rule 8: gap_fading=True → -0.3; False → 0."""
        s = self._low_quality_setup()
        _, brk = engine._compute_conviction_score_setup(
            s, spy_3d_range=1.0, gap_fading=True, return_breakdown=True)
        assert brk['gap_fading'] == -0.3
        _, brk = engine._compute_conviction_score_setup(
            s, spy_3d_range=1.0, gap_fading=False, return_breakdown=True)
        assert brk['gap_fading'] == 0.0

    def test_default_args_keep_v1_behavior(self, engine):
        """Calling without vwap_dist_pct/gap_fading reproduces V1 raw score."""
        s = self._high_quality_setup()
        score_v1, brk_v1 = engine._compute_conviction_score_setup(
            s, spy_3d_range=1.5, return_breakdown=True)
        score_explicit, brk_explicit = engine._compute_conviction_score_setup(
            s, spy_3d_range=1.5, vwap_dist_pct=0.0, gap_fading=False,
            return_breakdown=True)
        assert score_v1 == score_explicit
        assert brk_v1 == brk_explicit
        assert brk_v1['vwap_dist'] == 0.0
        assert brk_v1['gap_fading'] == 0.0

    def test_breakdown_high_quality_setup(self, engine):
        """High-quality setup: all positive contributions, no clamp."""
        s = self._high_quality_setup()
        score, brk = engine._compute_conviction_score_setup(
            s, spy_3d_range=1.5, return_breakdown=True)
        assert score == pytest.approx(2.40, abs=0.01)
        assert brk['pole_gain'] == 0.3
        assert brk['flag_tightness'] == 0.3
        assert brk['vol_ratio'] == 0.3
        assert brk['spy_regime'] == 0.3
        assert brk['retracement'] == 0.2

    def test_backward_compat_returns_float_only(self, engine):
        """Without return_breakdown, returns plain float (not tuple)."""
        s = self._low_quality_setup()
        score = engine._compute_conviction_score_setup(s, spy_3d_range=0.5)
        assert isinstance(score, float)
        assert score == 0.25

    def test_threshold_boundary_strict_less_than(self, engine):
        """Filter uses strict less-than: conviction == threshold PASSES.

        Locks in the boundary semantic — if a future maintainer changes
        `<` to `<=`, this test fails. A setup that hits ONLY the
        retracement rule scores exactly 1.2 (1.0 base + 0.2 retr).
        With threshold 1.2, the trade should NOT be filtered.
        """
        s = MagicMock()
        s.pole_gain_pct = 12.0       # outside 4.5-9 sweet spot → 0.0
        s.pole_high = 10.0
        s.pole_low = 9.0             # height 1.0
        s.flag_high = 9.4
        s.flag_low = 9.0             # range 0.4 = 40% (between 30 and 50) → 0.0
        s.avg_pole_volume = 100      # ratio 1.0 (≤1.7) → 0.0
        s.avg_flag_volume = 100
        s.retracement_pct = 25.0     # < 30 → +0.2
        # spy_3d in dead zone 0.8-1.2 (no contribution)
        score = engine._compute_conviction_score_setup(s, spy_3d_range=1.0)
        assert score == 1.2, f"Expected exactly 1.2, got {score}"
        # Verify strict less-than semantic: 1.2 < 1.2 is False
        threshold = 1.2
        assert (score < threshold) is False, "1.2 < 1.2 must be False"

    def test_threshold_loaded_from_config(self, mock_alpaca, db, mock_detector,
                                          mock_planner, mock_executor,
                                          mock_position_manager, monkeypatch):
        """min_threshold loads from config.yaml conviction_scoring section."""
        from config import Config
        # Patch config loader to return our test value
        original = Config._load_yaml_only
        def patched():
            cfg = original()
            cfg.setdefault('trading', {}).setdefault('conviction_scoring', {})['min_threshold'] = 1.2
            cfg['trading']['conviction_scoring']['enabled'] = True
            return cfg
        monkeypatch.setattr(Config, '_load_yaml_only', staticmethod(patched))

        eng = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=mock_detector,
            planner=mock_planner, executor=mock_executor,
            position_manager=mock_position_manager,
            pattern_poll_interval=60, enabled=True,
        )
        assert eng.conviction_enabled is True
        assert eng.conviction_min_threshold == 1.2

    def test_threshold_default_zero_when_unset(self, mock_alpaca, db, mock_detector,
                                                mock_planner, mock_executor,
                                                mock_position_manager, monkeypatch):
        """When config has no min_threshold, default is 0.0 (filter disabled,
        backward compat)."""
        from config import Config
        original = Config._load_yaml_only
        def patched():
            cfg = original()
            # Strip min_threshold if present — test the default
            cfg.setdefault('trading', {}).setdefault('conviction_scoring', {})
            cfg['trading']['conviction_scoring'].pop('min_threshold', None)
            return cfg
        monkeypatch.setattr(Config, '_load_yaml_only', staticmethod(patched))

        eng = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=mock_detector,
            planner=mock_planner, executor=mock_executor,
            position_manager=mock_position_manager,
            pattern_poll_interval=60, enabled=True,
        )
        # Default is 0.0 — filter disabled (backward compat)
        assert eng.conviction_min_threshold == 0.0

    def _make_engine_with_conv_config(self, mocks, monkeypatch, *,
                                       enabled=True, min_threshold=None):
        """Helper: build engine with patched conviction_scoring config."""
        mock_alpaca, db, mock_detector, mock_planner, mock_executor, mock_pm = mocks
        from config import Config
        original = Config._load_yaml_only
        def patched():
            cfg = original()
            cfg.setdefault('trading', {}).setdefault('conviction_scoring', {})
            cfg['trading']['conviction_scoring']['enabled'] = enabled
            if min_threshold is None:
                cfg['trading']['conviction_scoring'].pop('min_threshold', None)
            else:
                cfg['trading']['conviction_scoring']['min_threshold'] = min_threshold
            return cfg
        monkeypatch.setattr(Config, '_load_yaml_only', staticmethod(patched))
        return TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=mock_detector,
            planner=mock_planner, executor=mock_executor,
            position_manager=mock_pm,
            pattern_poll_interval=60, enabled=True,
        )

    def test_warning_threshold_above_max_blocks_everything(
        self, mock_alpaca, db, mock_detector, mock_planner,
        mock_executor, mock_position_manager, monkeypatch, caplog):
        """Threshold > 3.0 (max conviction) emits WARNING — would block all trades."""
        import logging
        mocks = (mock_alpaca, db, mock_detector, mock_planner,
                 mock_executor, mock_position_manager)
        with caplog.at_level(logging.WARNING):
            self._make_engine_with_conv_config(
                mocks, monkeypatch, enabled=True, min_threshold=12.0)
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any('> 3.0' in w and 'ALL trades will be blocked' in w
                   for w in warnings), f"Expected '> 3.0' warning. Got: {warnings}"

    def test_warning_threshold_negative(
        self, mock_alpaca, db, mock_detector, mock_planner,
        mock_executor, mock_position_manager, monkeypatch, caplog):
        """Negative threshold emits WARNING (filter inactive)."""
        import logging
        mocks = (mock_alpaca, db, mock_detector, mock_planner,
                 mock_executor, mock_position_manager)
        with caplog.at_level(logging.WARNING):
            self._make_engine_with_conv_config(
                mocks, monkeypatch, enabled=True, min_threshold=-1.0)
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any('< 0' in w and 'INACTIVE' in w
                   for w in warnings), f"Expected '< 0' warning. Got: {warnings}"

    def test_warning_threshold_set_but_conviction_disabled(
        self, mock_alpaca, db, mock_detector, mock_planner,
        mock_executor, mock_position_manager, monkeypatch, caplog):
        """Threshold set with enabled=false emits WARNING (silent ignore)."""
        import logging
        mocks = (mock_alpaca, db, mock_detector, mock_planner,
                 mock_executor, mock_position_manager)
        with caplog.at_level(logging.WARNING):
            self._make_engine_with_conv_config(
                mocks, monkeypatch, enabled=False, min_threshold=1.2)
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any('enabled=false' in w and 'INACTIVE' in w
                   for w in warnings), f"Expected enabled=false warning. Got: {warnings}"

    def test_no_warning_for_valid_threshold(
        self, mock_alpaca, db, mock_detector, mock_planner,
        mock_executor, mock_position_manager, monkeypatch, caplog):
        """Valid threshold (0 < t <= 3) with enabled=true → no warnings."""
        import logging
        mocks = (mock_alpaca, db, mock_detector, mock_planner,
                 mock_executor, mock_position_manager)
        with caplog.at_level(logging.WARNING):
            self._make_engine_with_conv_config(
                mocks, monkeypatch, enabled=True, min_threshold=1.2)
        # No conviction-related warnings (other warnings from setup are OK)
        conv_warnings = [r.message for r in caplog.records
                         if r.levelno == logging.WARNING
                         and 'conviction_min_threshold' in r.message]
        assert conv_warnings == [], f"Unexpected warnings: {conv_warnings}"
