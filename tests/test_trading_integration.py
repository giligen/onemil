"""
Integration tests for the trading pipeline.

Tests the full flow: pattern detection → trade planning → order execution → DB persistence.
Uses real database (temp file) and real component instances (mocked Alpaca API only).
"""

import pytest
import json
import pandas as pd
from datetime import datetime, timezone, timedelta, date
from unittest.mock import MagicMock, patch

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.pattern_detector import BullFlagDetector, BullFlagPattern
from trading.trade_planner import TradePlanner, TradePlan
from trading.order_executor import OrderExecutor
from trading.position_manager import PositionManager
from trading.trading_engine import TradingEngine


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
    """Mocked AlpacaClient (only external dependency)."""
    client = MagicMock(spec=AlpacaClient)
    client.get_open_positions.return_value = []
    return client


def _make_bull_flag_bars():
    """Create synthetic 1-min bars with a valid bull flag pattern (includes breakout)."""
    base_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    records = []
    candles = [
        # Pole: 3 green candles, ~10% gain
        (4.00, 4.12, 3.98, 4.10, 200000),
        (4.10, 4.24, 4.08, 4.22, 180000),
        (4.22, 4.42, 4.20, 4.40, 160000),
        # Flag: 2 red candles, ~35% retrace
        (4.40, 4.42, 4.30, 4.32, 50000),
        (4.32, 4.34, 4.28, 4.30, 30000),
        # Breakout candle
        (4.30, 4.55, 4.29, 4.50, 200000),
        # Current bar (will be dropped by detector)
        (4.50, 4.55, 4.48, 4.52, 100000),
    ]
    for i, (o, h, l, c, v) in enumerate(candles):
        records.append({
            'timestamp': base_time + timedelta(minutes=i),
            'open': float(o), 'high': float(h),
            'low': float(l), 'close': float(c),
            'volume': int(v),
        })
    return pd.DataFrame(records)


def _make_bull_flag_setup_bars():
    """Create bars for detect_setup() — pole + flag, NO breakout yet.

    detect_setup() drops the last bar as 'current', so completed bars
    end at the last flag bar. This lets it find the setup before breakout.
    """
    base_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    records = []
    candles = [
        # Pole: 3 green candles, ~10% gain
        (4.00, 4.12, 3.98, 4.10, 200000),
        (4.10, 4.24, 4.08, 4.22, 180000),
        (4.22, 4.42, 4.20, 4.40, 160000),
        # Flag: 2 red candles, ~35% retrace
        (4.40, 4.42, 4.30, 4.32, 50000),
        (4.32, 4.34, 4.28, 4.30, 30000),
        # Current bar (will be dropped by detect_setup) — still in flag
        (4.30, 4.35, 4.28, 4.31, 25000),
    ]
    for i, (o, h, l, c, v) in enumerate(candles):
        records.append({
            'timestamp': base_time + timedelta(minutes=i),
            'open': float(o), 'high': float(h),
            'low': float(l), 'close': float(c),
            'volume': int(v),
        })
    return pd.DataFrame(records)


# ===========================================================================
# INTEGRATION TESTS
# ===========================================================================

@pytest.mark.integration
class TestFullTradingPipeline:
    """End-to-end integration tests for the trading pipeline."""

    def test_detect_plan_execute_persist(self, mock_alpaca, db):
        """Full pipeline: detect pattern → create plan → execute → persist to DB."""
        # Setup real components
        detector = BullFlagDetector()
        planner = TradePlanner(position_size_dollars=500, max_shares=1000, min_risk_per_share=0.05)
        executor = OrderExecutor(alpaca_client=mock_alpaca, db=db)

        # Mock Alpaca order submission (submit_bracket_order used directly here)
        mock_alpaca.submit_bracket_order.return_value = {
            'id': 'bracket-order-123',
            'status': 'accepted',
        }

        # Step 1: Detect pattern
        bars = _make_bull_flag_bars()
        pattern = detector.detect("AAPL", bars)
        assert pattern is not None, "Pattern should be detected from valid bars"
        assert pattern.symbol == "AAPL"
        assert pattern.pole_gain_pct >= 3.0

        # Step 2: Create trade plan
        plan = planner.create_plan(pattern)
        assert plan is not None, "Plan should be created from valid pattern"
        assert plan.risk_reward_ratio >= 2.0
        assert plan.shares > 0

        # Step 3: Execute order
        result = executor.submit_bracket_order(plan)
        assert result is not None
        assert result['order_id'] == 'bracket-order-123'

        # Step 4: Verify DB persistence
        trades = db.get_trades_by_date(date.today().isoformat())
        assert len(trades) == 1

        trade = trades[0]
        assert trade['symbol'] == 'AAPL'
        assert trade['entry_price'] == plan.entry_price
        assert trade['stop_loss_price'] == plan.stop_loss_price
        assert trade['take_profit_price'] == plan.take_profit_price
        assert trade['shares'] == plan.shares
        assert trade['order_id'] == 'bracket-order-123'

        # Verify pattern data JSON
        pattern_data = json.loads(trade['pattern_data'])
        assert pattern_data['pole_height'] == pattern.pole_height
        assert pattern_data['breakout_level'] == pattern.breakout_level

    @patch('trading.trading_engine.TradingEngine._is_past_last_entry_time', return_value=False)
    @patch('trading.position_manager.datetime')
    def test_full_engine_cycle(self, mock_dt, _mock_time, mock_alpaca, db):
        """Full TradingEngine cycle: qualify → detect_setup → buy-stop → pending order."""
        # Mock time to mid-day
        mock_now = MagicMock()
        mock_now.hour = 10
        mock_now.minute = 30
        mock_dt.now.return_value = mock_now

        # Setup real components
        detector = BullFlagDetector()
        planner = TradePlanner(position_size_dollars=500, max_shares=1000, min_risk_per_share=0.05)
        executor = OrderExecutor(alpaca_client=mock_alpaca, db=db)
        position_manager = PositionManager(
            alpaca_client=mock_alpaca, db=db,
            max_positions=3, daily_loss_limit=-100.0,
        )

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db,
            detector=detector, planner=planner,
            executor=executor, position_manager=position_manager,
            enabled=True,
        )

        # Mock Alpaca responses — setup bars (no breakout yet)
        mock_alpaca.get_1min_bars.return_value = _make_bull_flag_setup_bars()
        mock_alpaca.submit_stop_bracket_order.return_value = {
            'id': 'engine-order-456', 'status': 'accepted',
        }

        # Run the cycle
        engine.on_stock_qualified("AAPL")
        result = engine.run_pattern_check()

        # Verify buy-stop order placed (goes to pending_orders)
        assert result is not None
        assert result['symbol'] == 'AAPL'
        assert result['order_type'] == 'stop_bracket'
        assert 'AAPL' in engine._pending_orders

        # patterns_detected increments on detection, patterns_traded on fill
        stats = engine.get_daily_stats()
        assert stats['patterns_detected'] >= 1
        assert stats['patterns_traded'] == 0  # not filled yet

        # Verify DB trade record
        trades = db.get_trades_by_date(date.today().isoformat())
        assert len(trades) == 1
        assert trades[0]['order_id'] == 'engine-order-456'

    @patch('trading.trading_engine.TradingEngine._is_past_last_entry_time', return_value=False)
    @patch('trading.position_manager.datetime')
    def test_position_limits_enforced(self, mock_dt, _mock_time, mock_alpaca, db):
        """Position manager correctly blocks when limits reached."""
        mock_now = MagicMock()
        mock_now.hour = 10
        mock_now.minute = 30
        mock_dt.now.return_value = mock_now

        detector = BullFlagDetector()
        planner = TradePlanner(position_size_dollars=500, max_shares=1000, min_risk_per_share=0.05)
        executor = OrderExecutor(alpaca_client=mock_alpaca, db=db)
        position_manager = PositionManager(
            alpaca_client=mock_alpaca, db=db,
            max_positions=1, daily_loss_limit=-100.0,
        )

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db,
            detector=detector, planner=planner,
            executor=executor, position_manager=position_manager,
            enabled=True,
        )

        mock_alpaca.get_1min_bars.return_value = _make_bull_flag_setup_bars()
        mock_alpaca.submit_stop_bracket_order.return_value = {
            'id': 'order-1', 'status': 'accepted',
        }

        # First trade should succeed (buy-stop placed)
        engine.on_stock_qualified("AAPL")
        result1 = engine.run_pattern_check()
        assert result1 is not None

        # Second trade should be blocked (max_positions=1, AAPL already marked traded)
        engine.on_stock_qualified("TSLA")
        mock_alpaca.get_1min_bars.return_value = _make_bull_flag_setup_bars()
        result2 = engine.run_pattern_check()
        assert result2 is None

    def test_database_trade_crud(self, db):
        """Test trade CRUD operations on the database."""
        now = datetime.now(timezone.utc)
        trade_data = {
            'trade_date': '2026-03-13',
            'symbol': 'AAPL',
            'side': 'buy',
            'entry_price': 4.40,
            'stop_loss_price': 4.20,
            'take_profit_price': 5.00,
            'shares': 100,
            'risk_per_share': 0.20,
            'total_risk': 20.0,
            'risk_reward_ratio': 3.0,
            'order_id': 'test-order-1',
            'order_status': 'accepted',
            'fill_price': None,
            'filled_at': None,
            'exit_price': None,
            'exit_reason': None,
            'exited_at': None,
            'pnl': None,
            'pnl_pct': None,
            'pattern_data': '{"pole_height": 0.50}',
            'created_at': now,
            'updated_at': now,
        }

        # Save
        trade_id = db.save_trade(trade_data)
        assert trade_id > 0

        # Retrieve
        trades = db.get_trades_by_date('2026-03-13')
        assert len(trades) == 1
        assert trades[0]['symbol'] == 'AAPL'

        # Update
        db.update_trade(trade_id, {
            'order_status': 'filled',
            'fill_price': 4.40,
            'filled_at': now,
        })
        trade = db.get_trade_by_order_id('test-order-1')
        assert trade['order_status'] == 'filled'
        assert trade['fill_price'] == 4.40

        # Open trades
        open_trades = db.get_open_trades('2026-03-13')
        assert len(open_trades) == 1

        # Close trade
        db.update_trade(trade_id, {
            'exit_price': 5.00,
            'exit_reason': 'take_profit',
            'exited_at': now,
            'pnl': 60.0,
            'pnl_pct': 13.6,
        })

        # Verify closed
        open_trades = db.get_open_trades('2026-03-13')
        assert len(open_trades) == 0

        # Verify PnL
        pnl = db.get_daily_pnl('2026-03-13')
        assert pnl == 60.0

    def test_daily_summary_crud(self, db):
        """Test daily summary save and retrieve."""
        summary = {
            'trade_date': '2026-03-13',
            'total_trades': 5,
            'winning_trades': 3,
            'losing_trades': 2,
            'gross_pnl': 45.50,
            'patterns_detected': 10,
            'patterns_traded': 5,
        }

        db.save_daily_summary(summary)
        result = db.get_daily_summary('2026-03-13')

        assert result is not None
        assert result['total_trades'] == 5
        assert result['winning_trades'] == 3
        assert result['gross_pnl'] == 45.50

        # Update
        summary['gross_pnl'] = 55.00
        db.save_daily_summary(summary)
        result = db.get_daily_summary('2026-03-13')
        assert result['gross_pnl'] == 55.00

    def test_pattern_detector_to_planner_data_integrity(self, db):
        """Verify data flows correctly from detector to planner."""
        detector = BullFlagDetector()
        planner = TradePlanner(position_size_dollars=500, min_risk_per_share=0.05)

        bars = _make_bull_flag_bars()
        pattern = detector.detect("TEST", bars)
        assert pattern is not None

        plan = planner.create_plan(pattern)
        assert plan is not None

        # Verify plan uses pattern data correctly
        assert plan.entry_price == pattern.breakout_level
        assert plan.pattern is pattern
        assert plan.shares > 0
        assert plan.risk_per_share > 0


# ===========================================================================
# THIN LIQUIDITY POST-FILL INTEGRATION TESTS
# ===========================================================================

@pytest.mark.integration
class TestThinLiquidityIntegration:
    """Integration tests for thin-liquidity post-fill volume check flow."""

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
        """Create a mock MarketRegimeFilter with normal liquidity."""
        from trading.market_regime import MarketRegimeFilter
        regime = MagicMock(spec=MarketRegimeFilter)
        regime.is_regime_ok.return_value = True
        regime.is_thin_liquidity.return_value = False
        regime.max_trades_per_day = 5
        regime.sma_period = 50
        return regime

    @patch('trading.trading_engine.TradingEngine._is_past_last_entry_time', return_value=False)
    @patch('trading.position_manager.datetime')
    @patch('trading.trading_engine.time_mod')
    def test_integration_thin_day_full_accept_flow(
        self, mock_time, mock_dt, _mock_entry, mock_alpaca, db
    ):
        """Setup → buy-stop → fill → volume OK → DB has fill, NO exit."""
        mock_now = MagicMock()
        mock_now.hour = 10
        mock_now.minute = 30
        mock_dt.now.return_value = mock_now

        regime = self._make_thin_regime()
        detector = BullFlagDetector()
        planner = TradePlanner(position_size_dollars=500, max_shares=1000, min_risk_per_share=0.05)
        executor = OrderExecutor(alpaca_client=mock_alpaca, db=db)
        position_manager = PositionManager(
            alpaca_client=mock_alpaca, db=db,
            max_positions=3, daily_loss_limit=-100.0,
        )

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db,
            detector=detector, planner=planner,
            executor=executor, position_manager=position_manager,
            enabled=True, market_regime=regime,
        )

        # Phase 1: Setup detection + buy-stop submission
        mock_alpaca.get_1min_bars.return_value = _make_bull_flag_setup_bars()
        mock_alpaca.submit_stop_bracket_order.return_value = {
            'id': 'thin-accept-1', 'status': 'accepted',
        }
        mock_alpaca.get_open_positions.return_value = []

        engine.on_stock_qualified("AAPL")
        result = engine.run_pattern_check()

        assert result is not None
        assert 'AAPL' in engine._pending_orders
        assert engine._pending_orders['AAPL']['thin_liquidity'] is True

        # Phase 2: Order fills + strong breakout volume
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 4.42, 'filled_qty': 113,
            'legs': [],
        }
        # Strong volume bars — BVR = 100000/40000 = 2.5x
        bars_with_volume = pd.DataFrame([
            {'high': 4.45, 'low': 4.35, 'close': 4.42, 'volume': 100000},
        ])
        mock_alpaca.get_1min_bars.return_value = bars_with_volume

        fill_result = engine._manage_pending_orders()

        assert fill_result is not None
        assert fill_result['status'] == 'filled'

        # Verify DB: has fill, NO exit
        trade = db.get_trade_by_order_id('thin-accept-1')
        assert trade['fill_price'] == 4.42
        assert trade['exit_price'] is None
        assert trade['exit_reason'] is None

    @patch('trading.trading_engine.TradingEngine._is_past_last_entry_time', return_value=False)
    @patch('trading.position_manager.datetime')
    @patch('trading.trading_engine.time_mod')
    def test_integration_thin_day_full_reject_flow(
        self, mock_time, mock_dt, _mock_entry, mock_alpaca, db
    ):
        """Setup → buy-stop → fill → volume weak → close → DB has fill AND exit."""
        mock_now = MagicMock()
        mock_now.hour = 10
        mock_now.minute = 30
        mock_dt.now.return_value = mock_now

        regime = self._make_thin_regime()
        detector = BullFlagDetector()
        planner = TradePlanner(position_size_dollars=500, max_shares=1000, min_risk_per_share=0.05)
        executor = OrderExecutor(alpaca_client=mock_alpaca, db=db)
        position_manager = PositionManager(
            alpaca_client=mock_alpaca, db=db,
            max_positions=3, daily_loss_limit=-100.0,
        )

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db,
            detector=detector, planner=planner,
            executor=executor, position_manager=position_manager,
            enabled=True, market_regime=regime,
        )

        # Phase 1: Setup detection + buy-stop submission
        mock_alpaca.get_1min_bars.return_value = _make_bull_flag_setup_bars()
        mock_alpaca.submit_stop_bracket_order.return_value = {
            'id': 'thin-reject-1', 'status': 'accepted',
        }
        mock_alpaca.get_open_positions.return_value = []

        engine.on_stock_qualified("AAPL")
        result = engine.run_pattern_check()
        assert result is not None

        # Phase 2: Order fills + weak breakout volume
        call_count = [0]
        def get_order_side_effect(order_id):
            """Return filled for first call (order status), then filled close order."""
            call_count[0] += 1
            if call_count[0] == 1:
                return {'status': 'filled', 'filled_avg_price': 4.42, 'filled_qty': 113}
            else:
                return {'status': 'filled', 'filled_avg_price': 4.39}

        mock_alpaca.get_order.side_effect = get_order_side_effect
        mock_alpaca.close_position.return_value = {'id': 'close-reject-1', 'status': 'accepted'}

        # Weak volume bars — BVR = 50000/40000 = 1.25x < 2.0x
        bars_weak = pd.DataFrame([
            {'high': 4.45, 'low': 4.35, 'close': 4.42, 'volume': 50000},
        ])
        mock_alpaca.get_1min_bars.return_value = bars_weak

        fill_result = engine._manage_pending_orders()

        assert fill_result is not None
        assert fill_result['status'] == 'thin_liquidity_rejected'

        # Verify DB: has fill AND exit with thin_liquidity_reject
        trade = db.get_trade_by_order_id('thin-reject-1')
        assert trade['fill_price'] == 4.42
        assert trade['exit_price'] == 4.39
        assert trade['exit_reason'] == 'thin_liquidity_reject'
        assert trade['pnl'] is not None
        expected_pnl = (4.39 - 4.42) * 113
        assert abs(trade['pnl'] - expected_pnl) < 0.01

    @patch('trading.trading_engine.TradingEngine._is_past_last_entry_time', return_value=False)
    @patch('trading.position_manager.datetime')
    @patch('trading.trading_engine.time_mod')
    def test_integration_normal_day_no_volume_check(
        self, mock_time, mock_dt, _mock_entry, mock_alpaca, db
    ):
        """Non-thin day → no bars fetched for volume check after fill."""
        mock_now = MagicMock()
        mock_now.hour = 10
        mock_now.minute = 30
        mock_dt.now.return_value = mock_now

        regime = self._make_normal_regime()
        detector = BullFlagDetector()
        planner = TradePlanner(position_size_dollars=500, max_shares=1000, min_risk_per_share=0.05)
        executor = OrderExecutor(alpaca_client=mock_alpaca, db=db)
        position_manager = PositionManager(
            alpaca_client=mock_alpaca, db=db,
            max_positions=3, daily_loss_limit=-100.0,
        )

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db,
            detector=detector, planner=planner,
            executor=executor, position_manager=position_manager,
            enabled=True, market_regime=regime,
        )

        # Phase 1: Setup + buy-stop
        mock_alpaca.get_1min_bars.return_value = _make_bull_flag_setup_bars()
        mock_alpaca.submit_stop_bracket_order.return_value = {
            'id': 'normal-1', 'status': 'accepted',
        }
        mock_alpaca.get_open_positions.return_value = []

        engine.on_stock_qualified("AAPL")
        engine.run_pattern_check()

        # Phase 2: Fill — no volume check on normal day
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 4.42, 'filled_qty': 113,
            'legs': [],
        }

        # Reset get_1min_bars call count
        mock_alpaca.get_1min_bars.reset_mock()

        fill_result = engine._manage_pending_orders()

        assert fill_result['status'] == 'filled'
        # get_1min_bars should NOT have been called for volume check
        mock_alpaca.get_1min_bars.assert_not_called()
        mock_alpaca.close_position.assert_not_called()

    @patch('trading.trading_engine.time_mod')
    def test_integration_thin_reject_force_close_no_double_close(
        self, mock_time, mock_alpaca, db
    ):
        """Rejected trade not in Alpaca positions → force_close skips it."""
        regime = self._make_thin_regime()
        detector = BullFlagDetector()
        planner = TradePlanner(position_size_dollars=500, max_shares=1000, min_risk_per_share=0.05)
        executor = OrderExecutor(alpaca_client=mock_alpaca, db=db)
        position_manager = PositionManager(
            alpaca_client=mock_alpaca, db=db,
            max_positions=3, daily_loss_limit=-100.0,
        )

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db,
            detector=detector, planner=planner,
            executor=executor, position_manager=position_manager,
            enabled=True, market_regime=regime,
        )

        # Create a trade that was already rejected (has exit_price set)
        now = datetime.now(timezone.utc)
        db.save_trade({
            'trade_date': date.today().isoformat(), 'symbol': 'AAPL',
            'side': 'buy', 'entry_price': 4.40, 'stop_loss_price': 4.29,
            'take_profit_price': 4.90, 'shares': 113, 'risk_per_share': 0.11,
            'total_risk': 12.43, 'risk_reward_ratio': 4.5,
            'order_id': 'rejected-order', 'order_status': 'filled',
            'fill_price': 4.42, 'filled_at': now,
            'exit_price': 4.39, 'exit_reason': 'thin_liquidity_reject',
            'exited_at': now, 'pnl': -3.39, 'pnl_pct': -0.68,
            'pattern_data': '{}', 'created_at': now, 'updated_at': now,
        })

        # No Alpaca positions (trade was already closed)
        mock_alpaca.get_open_positions.return_value = []

        engine._force_close_all()

        # close_position should NOT be called (no positions)
        mock_alpaca.close_position.assert_not_called()

        # get_open_trades should NOT return this trade (exit_price IS NOT NULL)
        open_trades = db.get_open_trades(date.today().isoformat())
        assert len(open_trades) == 0

    @patch('trading.trading_engine.time_mod')
    def test_integration_thin_reject_db_complete_lifecycle(
        self, mock_time, mock_alpaca, db
    ):
        """All fields populated: trade_date, fill_price, filled_at, exit_price, exit_reason, pnl, exited_at."""
        regime = self._make_thin_regime()
        executor = OrderExecutor(alpaca_client=mock_alpaca, db=db)
        position_manager = PositionManager(
            alpaca_client=mock_alpaca, db=db,
            max_positions=3, daily_loss_limit=-100.0,
        )

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db,
            detector=BullFlagDetector(), planner=TradePlanner(position_size_dollars=500, min_risk_per_share=0.05),
            executor=executor, position_manager=position_manager,
            enabled=True, market_regime=regime,
        )

        now = datetime.now(timezone.utc)
        trade_id = db.save_trade({
            'trade_date': date.today().isoformat(), 'symbol': 'AAPL',
            'side': 'buy', 'entry_price': 4.40, 'stop_loss_price': 4.29,
            'take_profit_price': 4.90, 'shares': 113, 'risk_per_share': 0.11,
            'total_risk': 12.43, 'risk_reward_ratio': 4.5,
            'order_id': 'lifecycle-1', 'order_status': 'filled',
            'fill_price': 4.42, 'filled_at': now,
            'exit_price': None, 'exit_reason': None, 'exited_at': None,
            'pnl': None, 'pnl_pct': None,
            'pattern_data': '{}', 'created_at': now, 'updated_at': now,
        })
        trade_record = db.get_trade_by_order_id('lifecycle-1')

        mock_alpaca.close_position.return_value = {'id': 'close-lc', 'status': 'accepted'}
        mock_alpaca.get_order.return_value = {'status': 'filled', 'filled_avg_price': 4.38}

        engine._reject_thin_liquidity_trade('AAPL', 'lifecycle-1', 4.42, 113, trade_record)

        # Verify complete lifecycle fields
        trade = db.get_trade_by_order_id('lifecycle-1')
        assert str(trade['trade_date']) == date.today().isoformat()
        assert trade['fill_price'] == 4.42
        assert trade['filled_at'] is not None
        assert trade['exit_price'] == 4.38
        assert trade['exit_reason'] == 'thin_liquidity_reject'
        assert trade['exited_at'] is not None
        expected_pnl = (4.38 - 4.42) * 113
        assert abs(trade['pnl'] - expected_pnl) < 0.01
        expected_pnl_pct = (4.38 / 4.42 - 1) * 100
        assert abs(trade['pnl_pct'] - expected_pnl_pct) < 0.01
