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
    client.get_buying_power.return_value = 1_000_000.0  # ample buying power
    client.is_marginable.return_value = True
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
        engine.quality_filter_enabled = False

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
        engine.quality_filter_enabled = False

        mock_alpaca.get_1min_bars.return_value = _make_bull_flag_setup_bars()
        mock_alpaca.submit_stop_bracket_order.return_value = {
            'id': 'order-1', 'status': 'accepted',
        }

        # First trade should succeed (buy-stop placed)
        engine.on_stock_qualified("AAPL")
        result1 = engine.run_pattern_check()
        assert result1 is not None

        # Second trade should be blocked (max_positions=1, AAPL pending)
        engine.on_stock_qualified("TSLA")
        mock_alpaca.get_1min_bars.return_value = _make_bull_flag_setup_bars()
        # AAPL's pending order is still 'new' (not filled)
        mock_alpaca.get_order.return_value = {'status': 'new', 'filled_avg_price': None, 'filled_qty': 0}
        result2 = engine.run_pattern_check()
        # TSLA should not have a pending order (position manager blocks it)
        assert "TSLA" not in engine._pending_orders

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
        engine.quality_filter_enabled = False

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
        # Get the actual plan to know stop/target for legs
        pending = engine._pending_orders['AAPL']
        plan = pending['plan']
        setup = pending['setup']
        breakout = setup.breakout_level

        # Use side_effect: first call = order status, second = legs for gap-fill
        mock_alpaca.get_order.side_effect = [
            {'status': 'filled', 'filled_avg_price': 4.42, 'filled_qty': plan.shares},
            {'legs': [
                {'id': 'sl-1', 'side': 'sell', 'stop_price': plan.stop_loss_price,
                 'limit_price': None, 'status': 'new'},
                {'id': 'tp-1', 'side': 'sell', 'stop_price': None,
                 'limit_price': plan.take_profit_price, 'status': 'new'},
            ]},
        ]
        mock_alpaca.replace_order_stop_price.return_value = {'id': 'sl-1'}
        mock_alpaca.replace_order_limit_price.return_value = {'id': 'tp-1'}

        # Strong volume bars — BVR = 100000/avg_flag_volume
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
        engine.quality_filter_enabled = False

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
        engine.quality_filter_enabled = False

        # Phase 1: Setup + buy-stop
        mock_alpaca.get_1min_bars.return_value = _make_bull_flag_setup_bars()
        mock_alpaca.submit_stop_bracket_order.return_value = {
            'id': 'normal-1', 'status': 'accepted',
        }
        mock_alpaca.get_open_positions.return_value = []

        engine.on_stock_qualified("AAPL")
        engine.run_pattern_check()

        # Phase 2: Fill — no volume check on normal day
        pending = engine._pending_orders['AAPL']
        plan = pending['plan']

        # Provide legs for gap-fill adjustment (fill may be above breakout)
        mock_alpaca.get_order.side_effect = [
            {'status': 'filled', 'filled_avg_price': 4.42, 'filled_qty': plan.shares},
            {'legs': [
                {'id': 'sl-1', 'side': 'sell', 'stop_price': plan.stop_loss_price,
                 'limit_price': None, 'status': 'new'},
                {'id': 'tp-1', 'side': 'sell', 'stop_price': None,
                 'limit_price': plan.take_profit_price, 'status': 'new'},
            ]},
        ]
        mock_alpaca.replace_order_stop_price.return_value = {'id': 'sl-1'}
        mock_alpaca.replace_order_limit_price.return_value = {'id': 'tp-1'}

        # Reset get_1min_bars call count
        mock_alpaca.get_1min_bars.reset_mock()

        fill_result = engine._manage_pending_orders()

        assert fill_result['status'] == 'filled'
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

        engine._emergency_close_position('AAPL', 'lifecycle-1', 4.42, 113, trade_record)

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


# ===========================================================================
# SCANNER → ENGINE LOOP INTEGRATION TESTS (Bug #1, #2, #3, #4 fixes)
# ===========================================================================

@pytest.mark.integration
class TestScannerEngineLoopIntegration:
    """Integration tests for the scanner's main loop driving the trading engine."""

    def _make_engine(self, mock_alpaca, db):
        """Create a real TradingEngine with mocked Alpaca."""
        from trading.market_regime import MarketRegimeFilter

        detector = BullFlagDetector()
        planner = TradePlanner(position_size_dollars=500, max_shares=1000, min_risk_per_share=0.05)
        executor = OrderExecutor(alpaca_client=mock_alpaca, db=db)
        position_manager = PositionManager(
            alpaca_client=mock_alpaca, db=db,
            max_positions=3, daily_loss_limit=-100.0,
        )
        regime = MagicMock(spec=MarketRegimeFilter)
        regime.is_regime_ok.return_value = True
        regime.is_thin_liquidity.return_value = False
        regime.max_trades_per_day = 5
        regime.sma_period = 50

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db,
            detector=detector, planner=planner,
            executor=executor, position_manager=position_manager,
            enabled=True, market_regime=regime,
            force_close_time_et="15:45",
        )
        return engine

    @patch('scanner.realtime_scanner.datetime')
    @patch('trading.trading_engine.time_mod')
    def test_scanner_forces_close_at_configured_time(
        self, mock_time, mock_dt, mock_alpaca, db
    ):
        """Full flow: scanner loop → force close at 15:45 → positions closed."""
        import pytz
        import threading
        from datetime import datetime as real_datetime
        from scanner.criteria import ScannerCriteria
        from scanner.realtime_scanner import RealtimeScanner
        from data_sources.news_provider import NewsProvider

        ET = pytz.timezone('US/Eastern')
        shutdown = threading.Event()

        engine = self._make_engine(mock_alpaca, db)
        mock_news = MagicMock(spec=NewsProvider)

        scanner = RealtimeScanner(
            alpaca_client=mock_alpaca,
            news_provider=mock_news,
            db=db,
            criteria=ScannerCriteria(),
            trading_engine=engine,
            shutdown_event=shutdown,
        )

        # Setup universe
        mock_alpaca.is_trading_day.return_value = True
        mock_alpaca.is_short_trading_day.return_value = False
        scanner._universe = [{'symbol': 'AAA', 'price_close': 5.0, 'float_shares': 1_000_000}]
        scanner._premarket_gap_symbols = {'AAA'}  # Skip gap scan

        # Time: 15:46 → force close fires, then 16:00 → exit
        tick = [0]
        def fake_now(*args, **kwargs):
            tick[0] += 1
            if tick[0] <= 4:
                return real_datetime(2026, 3, 16, 15, 46, 0, tzinfo=ET)
            return real_datetime(2026, 3, 16, 16, 0, 0, tzinfo=ET)
        mock_dt.now.side_effect = fake_now
        mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

        mock_alpaca.get_open_positions.return_value = []
        mock_alpaca.get_latest_trades.return_value = {}
        mock_alpaca.get_current_bars.return_value = {}

        scanner._interruptible_sleep = MagicMock(return_value=False)

        # Bypass _is_trading_day and _load_universe
        with patch.object(scanner, '_is_trading_day', return_value=True), \
             patch.object(scanner, '_load_universe'):
            scanner.run()

        # Verify force close was triggered (via _force_close_all)
        # The engine should have attempted force close
        assert mock_alpaca.get_open_positions.called

    @patch('scanner.realtime_scanner.datetime')
    def test_shutdown_triggers_force_close(self, mock_dt, mock_alpaca, db):
        """Shutdown event → force close → positions closed."""
        import pytz
        import threading
        from datetime import datetime as real_datetime
        from scanner.criteria import ScannerCriteria
        from scanner.realtime_scanner import RealtimeScanner
        from data_sources.news_provider import NewsProvider

        ET = pytz.timezone('US/Eastern')
        shutdown = threading.Event()

        engine = self._make_engine(mock_alpaca, db)
        mock_news = MagicMock(spec=NewsProvider)

        scanner = RealtimeScanner(
            alpaca_client=mock_alpaca,
            news_provider=mock_news,
            db=db,
            criteria=ScannerCriteria(),
            trading_engine=engine,
            shutdown_event=shutdown,
        )

        mock_alpaca.is_trading_day.return_value = True
        mock_alpaca.is_short_trading_day.return_value = False
        scanner._universe = [{'symbol': 'AAA', 'price_close': 5.0, 'float_shares': 1_000_000}]
        scanner._premarket_gap_symbols = {'AAA'}

        fake_now = real_datetime(2026, 3, 16, 10, 0, 0, tzinfo=ET)
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

        mock_alpaca.get_open_positions.return_value = []

        # Set shutdown before first iteration
        shutdown.set()

        with patch.object(scanner, '_is_trading_day', return_value=True), \
             patch.object(scanner, '_load_universe'):
            scanner.run()

        # Post-loop safety net should call _force_close_all
        mock_alpaca.get_open_positions.assert_called()

    @patch('scanner.realtime_scanner.datetime')
    def test_qualified_symbols_cleared_each_cycle(self, mock_dt, mock_alpaca, db):
        """Stale symbols purged at bucket boundary."""
        import pytz
        import threading
        from datetime import datetime as real_datetime
        from scanner.criteria import ScannerCriteria
        from scanner.realtime_scanner import RealtimeScanner
        from data_sources.news_provider import NewsProvider

        ET = pytz.timezone('US/Eastern')
        shutdown = threading.Event()

        engine = self._make_engine(mock_alpaca, db)
        mock_news = MagicMock(spec=NewsProvider)

        scanner = RealtimeScanner(
            alpaca_client=mock_alpaca,
            news_provider=mock_news,
            db=db,
            criteria=ScannerCriteria(),
            trading_engine=engine,
            shutdown_event=shutdown,
        )

        mock_alpaca.is_trading_day.return_value = True
        mock_alpaca.is_short_trading_day.return_value = False
        scanner._universe = [{'symbol': 'AAA', 'price_close': 5.0, 'float_shares': 1_000_000}]
        scanner._premarket_gap_symbols = {'AAA'}

        mock_alpaca.get_latest_trades.return_value = {}
        mock_alpaca.get_current_bars.return_value = {}
        mock_alpaca.get_open_positions.return_value = []

        # Pre-populate stale symbols
        engine._qualified_symbols = {'STALE1', 'STALE2'}

        # One tick at 10:00, then 16:00 to exit
        tick = [0]
        def fake_now(*args, **kwargs):
            tick[0] += 1
            if tick[0] <= 3:
                return real_datetime(2026, 3, 16, 10, 0, 0, tzinfo=ET)
            return real_datetime(2026, 3, 16, 16, 0, 0, tzinfo=ET)
        mock_dt.now.side_effect = fake_now
        mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

        scanner._interruptible_sleep = MagicMock(return_value=False)

        with patch.object(scanner, '_is_trading_day', return_value=True), \
             patch.object(scanner, '_load_universe'):
            scanner.run()

        # Qualified symbols should have been cleared before intraday cycle
        # After the cycle, only newly qualified symbols should be present (none in this case)
        assert 'STALE1' not in engine._qualified_symbols
        assert 'STALE2' not in engine._qualified_symbols

    @patch('scanner.realtime_scanner.datetime')
    def test_end_of_day_safety_net_forces_close(self, mock_dt, mock_alpaca, db):
        """16:00 exit → post-loop force close fires."""
        import pytz
        import threading
        from datetime import datetime as real_datetime
        from scanner.criteria import ScannerCriteria
        from scanner.realtime_scanner import RealtimeScanner
        from data_sources.news_provider import NewsProvider

        ET = pytz.timezone('US/Eastern')
        shutdown = threading.Event()

        engine = self._make_engine(mock_alpaca, db)
        mock_news = MagicMock(spec=NewsProvider)

        scanner = RealtimeScanner(
            alpaca_client=mock_alpaca,
            news_provider=mock_news,
            db=db,
            criteria=ScannerCriteria(),
            trading_engine=engine,
            shutdown_event=shutdown,
        )

        mock_alpaca.is_trading_day.return_value = True
        mock_alpaca.is_short_trading_day.return_value = False
        scanner._universe = [{'symbol': 'AAA', 'price_close': 5.0, 'float_shares': 1_000_000}]
        scanner._premarket_gap_symbols = {'AAA'}

        mock_alpaca.get_open_positions.return_value = []
        mock_alpaca.get_latest_trades.return_value = {}
        mock_alpaca.get_current_bars.return_value = {}

        # First call at 15:59 (enters loop), second call at 16:00 (exits loop)
        tick = [0]
        def fake_now(*args, **kwargs):
            tick[0] += 1
            if tick[0] <= 2:
                return real_datetime(2026, 3, 16, 15, 59, 0, tzinfo=ET)
            return real_datetime(2026, 3, 16, 16, 0, 0, tzinfo=ET)
        mock_dt.now.side_effect = fake_now
        mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

        scanner._interruptible_sleep = MagicMock(return_value=False)

        with patch.object(scanner, '_is_trading_day', return_value=True), \
             patch.object(scanner, '_load_universe'):
            scanner.run()

        # Post-loop safety net should have called _force_close_all
        mock_alpaca.get_open_positions.assert_called()


# ===========================================================================
# Bug A: Position management not blocked by regime/max-trades guards
# ===========================================================================

@pytest.mark.integration
class TestPositionSyncUnderGuards:
    """Verify that SL/TP exits are recorded even when regime filter or
    max-trades guard blocks new order placement."""

    def test_sl_exit_recorded_under_regime_filter(self, db, mock_alpaca):
        """A stop-loss exit is detected and recorded even when regime blocks trading."""
        from trading.market_regime import MarketRegimeFilter

        detector = BullFlagDetector()
        planner = TradePlanner()
        executor = OrderExecutor(mock_alpaca, db)
        pm = PositionManager(mock_alpaca, db)

        regime = MagicMock(spec=MarketRegimeFilter)
        regime.is_regime_ok.return_value = False
        regime.get_regime_info.return_value = {
            'vol_5d': 3.0, 'sma': 400.0, 'is_below_sma': True,
            'is_ok': False, 'spy_volume_ratio': 1.0,
        }
        regime.vol_threshold = 2.0
        regime.sma_period = 50

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=detector,
            planner=planner, executor=executor, position_manager=pm,
            enabled=True, market_regime=regime,
        )

        # Pre-existing filled trade in DB
        today = date.today().isoformat()
        trade_id = db.save_trade(_make_trade_record(
            trade_date=today, symbol='AAPL', order_id='ord-1',
            order_status='filled', fill_price=10.0,
        ))

        # Alpaca: position is GONE (SL hit)
        mock_alpaca.get_open_positions.return_value = []
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 10.0, 'filled_qty': 100,
            'legs': [
                {'side': 'sell', 'stop_price': 9.5, 'status': 'filled',
                 'filled_avg_price': 9.5},
                {'side': 'sell', 'limit_price': 11.0, 'status': 'canceled'},
            ],
        }

        engine.run_pattern_check()

        # Exit should be recorded despite regime blocking new trades
        trade = db.get_trades_by_date(today)[0]
        assert trade['exit_price'] == 9.5
        assert trade['exit_reason'] == 'stop_loss'
        assert trade['pnl'] == pytest.approx(-50.0)


# ===========================================================================
# Bug C: Startup sync rebuilds state after crash
# ===========================================================================

@pytest.mark.integration
class TestStartupSyncIntegration:
    """Verify reset_daily rebuilds traded_symbols, pending_orders, and
    daily_trade_count from real DB state — prevents crash recovery issues."""

    def test_crash_recovery_prevents_double_entry(self, db, mock_alpaca):
        """After crash + restart, engine won't re-trade a symbol it already traded."""
        detector = BullFlagDetector()
        planner = TradePlanner()
        executor = OrderExecutor(mock_alpaca, db)
        pm = PositionManager(mock_alpaca, db)

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=detector,
            planner=planner, executor=executor, position_manager=pm,
            enabled=True,
        )

        # Simulate: previous run traded AAPL, then crashed
        today = date.today().isoformat()
        db.save_trade(_make_trade_record(
            trade_date=today, symbol='AAPL', order_id='ord-1',
            order_status='filled', fill_price=10.0,
        ))

        # Mock SPY data for regime refresh
        mock_alpaca.get_daily_bars_range.return_value = {}

        # Simulate restart: reset_daily should rebuild state
        engine.reset_daily()

        # AAPL should be in traded_symbols — engine won't try it again
        assert 'AAPL' in engine._traded_symbols
        assert engine._daily_trade_count == 1

        # Try to qualify AAPL — should be rejected
        engine.on_stock_qualified('AAPL')
        assert 'AAPL' not in engine._qualified_symbols

    def test_pending_order_recovered_after_crash(self, db, mock_alpaca):
        """After crash, a pending buy-stop still live on Alpaca is tracked."""
        detector = BullFlagDetector()
        planner = TradePlanner()
        executor = OrderExecutor(mock_alpaca, db)
        pm = PositionManager(mock_alpaca, db)

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=detector,
            planner=planner, executor=executor, position_manager=pm,
            enabled=True,
        )

        today = date.today().isoformat()
        db.save_trade(_make_trade_record(
            trade_date=today, symbol='TSLA', entry_price=15.0,
            stop_loss_price=14.5, take_profit_price=16.0, shares=200,
            order_id='ord-pending', order_status='accepted',
        ))

        mock_alpaca.get_daily_bars_range.return_value = {}

        engine.reset_daily()

        # Pending order should be recovered with reconstructed plan
        assert 'TSLA' in engine._pending_orders
        assert engine._pending_orders['TSLA']['order_id'] == 'ord-pending'
        assert engine._pending_orders['TSLA']['plan'] is not None
        assert engine._pending_orders['TSLA']['plan'].shares == 200
        assert engine._pending_orders['TSLA']['plan'].entry_price == 15.0
        # Not counted as filled
        assert engine._daily_trade_count == 0


# ===========================================================================
# Bug #1 + #2: Crash Recovery Integration
# ===========================================================================

@pytest.mark.integration
class TestCrashRecoveryIntegration:
    """Full crash recovery cycle: save → crash → reset → fill → DB correct."""

    @patch('trading.trading_engine.time_mod.sleep')
    def test_full_crash_recovery_cycle(self, mock_sleep, db, mock_alpaca):
        """Save trade → crash (clear in-memory) → reset → pending fills → DB updated."""
        import json

        detector = BullFlagDetector()
        planner = TradePlanner()
        executor = OrderExecutor(mock_alpaca, db)
        pm = PositionManager(mock_alpaca, db)

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=detector,
            planner=planner, executor=executor, position_manager=pm,
            enabled=True,
        )

        today = date.today().isoformat()
        pattern_data = json.dumps({
            'breakout_level': 10.0, 'flag_low': 9.5,
            'avg_flag_volume': 50000,
            'pole_start_idx': 0, 'pole_end_idx': 2,
            'flag_start_idx': 3, 'flag_end_idx': 4,
        })
        db.save_trade(_make_trade_record(
            trade_date=today, symbol='CRASH', entry_price=10.0,
            stop_loss_price=9.5, take_profit_price=11.0, shares=100,
            risk_per_share=0.5, risk_reward_ratio=2.0,
            order_id='ord-crash', order_status='accepted',
            pattern_data=pattern_data,
        ))

        # Simulate crash: clear all in-memory state
        mock_alpaca.get_daily_bars_range.return_value = {}
        engine.reset_daily()

        # Verify recovery
        assert 'CRASH' in engine._pending_orders
        pending = engine._pending_orders['CRASH']
        assert pending['plan'] is not None
        assert pending['plan'].shares == 100
        assert pending['setup'] is not None
        assert pending['setup'].breakout_level == 10.0

        # Now simulate the order filling
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 10.05, 'filled_qty': 100,
            'legs': [],
        }

        result = engine._manage_pending_orders()

        # Verify DB updated correctly
        assert 'CRASH' in engine._traded_symbols
        trade = db.get_trade_by_order_id('ord-crash')
        assert trade['fill_price'] == pytest.approx(10.05, abs=0.01)
        assert trade['order_status'] == 'filled'
        assert trade['filled_qty'] == 100

    @patch('trading.trading_engine.time_mod.sleep')
    def test_force_close_after_sl_exit(self, mock_sleep, db, mock_alpaca):
        """SL fills → force_close_all → SL exit recorded with correct reason."""
        detector = BullFlagDetector()
        planner = TradePlanner()
        executor = OrderExecutor(mock_alpaca, db)
        pm = PositionManager(mock_alpaca, db)

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=detector,
            planner=planner, executor=executor, position_manager=pm,
            enabled=True,
        )

        today = date.today().isoformat()
        db.save_trade(_make_trade_record(
            trade_date=today, symbol='SLFC', entry_price=10.0,
            stop_loss_price=9.5, take_profit_price=11.0, shares=100,
            risk_per_share=0.5, risk_reward_ratio=2.0,
            order_id='ord-slfc', order_status='filled', fill_price=10.0,
        ))

        # SL was hit — Alpaca shows no open position
        mock_alpaca.get_open_positions.return_value = []
        # Bracket order shows SL leg filled
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 10.0, 'filled_qty': 100,
            'legs': [
                {'side': 'sell', 'stop_price': 9.5, 'limit_price': None,
                 'status': 'filled', 'filled_avg_price': 9.50},
                {'side': 'sell', 'stop_price': None, 'limit_price': 11.0,
                 'status': 'canceled', 'filled_avg_price': None},
            ],
        }

        # Force close — should sync first, detect SL exit, then find nothing to close
        engine._force_close_all()

        trade = db.get_trade_by_order_id('ord-slfc')
        assert trade['exit_reason'] == 'stop_loss'
        assert trade['exit_price'] == pytest.approx(9.50, abs=0.01)

    @patch('trading.trading_engine.time_mod.sleep')
    def test_late_fill_before_force_close_has_pnl(self, mock_sleep, db, mock_alpaca):
        """Pending order fills at 15:44, force-close at 15:45 — P&L computed correctly."""
        detector = BullFlagDetector()
        planner = TradePlanner()
        executor = OrderExecutor(mock_alpaca, db)
        pm = PositionManager(mock_alpaca, db)

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=detector,
            planner=planner, executor=executor, position_manager=pm,
            enabled=True,
        )

        today = date.today().isoformat()
        pattern_data = json.dumps({'breakout_level': 5.0, 'flag_low': 4.5})
        db.save_trade(_make_trade_record(
            trade_date=today, symbol='LATE', entry_price=5.0,
            stop_loss_price=4.50, take_profit_price=6.0, shares=200,
            risk_per_share=0.50, risk_reward_ratio=2.0,
            order_id='ord-late', order_status='accepted',
            pattern_data=pattern_data,
        ))

        # Simulate crash recovery to rebuild pending orders
        mock_alpaca.get_daily_bars_range.return_value = {}
        mock_alpaca.get_open_positions.return_value = []
        engine.reset_daily()

        assert 'LATE' in engine._pending_orders

        # Now the order fills at breakout_level (no gap → no gap-fill adjustment)
        call_count = [0]
        def get_order_side_effect(order_id):
            call_count[0] += 1
            if order_id == 'ord-late':
                return {
                    'status': 'filled', 'filled_avg_price': 5.00,
                    'filled_qty': 200, 'legs': [],
                }
            # For close order polling
            return {'status': 'filled', 'filled_avg_price': 5.20}

        mock_alpaca.get_order.side_effect = get_order_side_effect

        # Alpaca shows the position after fill
        mock_alpaca.get_open_positions.return_value = [
            {'symbol': 'LATE', 'qty': 200, 'avg_entry_price': 5.00,
             'market_value': 1000.0},
        ]
        mock_alpaca.close_position.return_value = {'id': 'close-late'}
        mock_alpaca.cancel_order.side_effect = Exception("order already filled")

        # Force close — should process fill FIRST, then close
        engine._force_close_all()

        trade = db.get_trade_by_order_id('ord-late')
        # fill_price must be set (from _manage_pending_orders)
        assert trade['fill_price'] == pytest.approx(5.00, abs=0.01)
        # P&L must be computed (force_close used fill_price)
        assert trade['exit_reason'] == 'force_close'
        assert trade['pnl'] is not None


@pytest.mark.integration
class TestOrphanPositionIntegration:
    """Integration test: orphan positions from prior days are detected and closed."""

    @patch('trading.trading_engine.time_mod.sleep')
    def test_orphan_detected_on_startup(self, mock_sleep, db, mock_alpaca):
        """Orphan position from prior day is auto-closed on startup."""
        detector = BullFlagDetector()
        planner = TradePlanner()
        executor = OrderExecutor(mock_alpaca, db)
        pm = PositionManager(mock_alpaca, db)

        engine = TradingEngine(
            alpaca_client=mock_alpaca, db=db, detector=detector,
            planner=planner, executor=executor, position_manager=pm,
            enabled=True,
        )

        # No trades today in DB
        mock_alpaca.get_daily_bars_range.return_value = {}
        # But Alpaca has a position from yesterday
        mock_alpaca.get_open_positions.return_value = [
            {'symbol': 'YEST', 'qty': 100, 'avg_entry_price': 8.00,
             'market_value': 780.0},
        ]
        mock_alpaca.close_position.return_value = {'id': 'close-yest'}
        mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_avg_price': 7.80,
        }

        engine.reset_daily()

        # Orphan closed
        mock_alpaca.close_position.assert_called_once_with('YEST')
        # Marked traded to prevent re-entry
        assert 'YEST' in engine._traded_symbols
        assert engine._daily_trade_count == 0  # Orphan is not counted as today's trade


class TestFlagHighMaxFlowThrough:
    """Integration: max flag high flows through detector → planner → entry_price."""

    def test_flag_high_max_flows_through_to_entry_price(self, db, mock_alpaca):
        """pattern.breakout_level (max flag high) → plan.entry_price."""
        detector = BullFlagDetector()
        planner = TradePlanner()

        # Build bars where flag bar highs are descending: 4.55, 4.45, 4.42
        # Max flag high = 4.55
        base_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        candles = [
            # Pole: 3 green candles, ~10% gain
            (4.00, 4.15, 3.98, 4.13, 200000),
            (4.13, 4.30, 4.11, 4.28, 180000),
            (4.28, 4.52, 4.26, 4.50, 160000),
            # Flag: descending highs — first bar has highest high
            (4.50, 4.55, 4.38, 4.40, 50000),
            (4.40, 4.45, 4.33, 4.35, 30000),
            (4.35, 4.42, 4.32, 4.34, 25000),
            # Current bar (dropped by detect_setup)
            (4.34, 4.38, 4.32, 4.33, 20000),
        ]
        records = []
        for i, (o, h, l, c, v) in enumerate(candles):
            records.append({
                'timestamp': base_time + timedelta(minutes=i),
                'open': float(o), 'high': float(h),
                'low': float(l), 'close': float(c),
                'volume': int(v),
            })
        bars = pd.DataFrame(records)

        # Detect setup
        setup = detector.detect_setup("TEST", bars)
        assert setup is not None
        assert setup.flag_high == 4.55  # Max of all flag bar highs
        assert setup.breakout_level == 4.55

        # Create trade plan
        plan = planner.create_plan(setup)
        assert plan is not None
        assert plan.entry_price == 4.55  # Flows through from breakout_level
