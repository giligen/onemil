"""
Tests for the backtesting engine.

Covers:
- TradeSimulator: stop, target, EOD, ambiguity scenarios
- BacktestRunner: sliding window, one-trade-per-day, edge cases
- Integration: full pipeline with synthetic bars that form a bull flag
"""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from backtest import (
    SimulatedTrade,
    BacktestResult,
    PatternDetection,
    PendingBuyStop,
    TradeSimulator,
    BacktestRunner,
    print_report,
)
from trading.pattern_detector import BullFlagDetector, BullFlagPattern, BullFlagSetup
from trading.trade_planner import TradePlanner, TradePlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2026, 3, 13, 14, 0, 0, tzinfo=timezone.utc)


def _ts(minute_offset: int) -> datetime:
    """Create a timestamp offset by minutes from BASE_TIME."""
    return BASE_TIME + timedelta(minutes=minute_offset)


def _make_bars(candles: list, start_minute: int = 0) -> pd.DataFrame:
    """
    Build a bars DataFrame from (open, high, low, close, volume) tuples.

    Args:
        candles: List of (open, high, low, close, volume) tuples
        start_minute: Starting minute offset from BASE_TIME

    Returns:
        DataFrame with timestamp, open, high, low, close, volume
    """
    records = []
    for i, (o, h, l, c, v) in enumerate(candles):
        records.append({
            'timestamp': _ts(start_minute + i),
            'open': o, 'high': h, 'low': l, 'close': c, 'volume': v,
        })
    return pd.DataFrame(records)


def _make_pattern(
    symbol: str = "TEST",
    breakout_level: float = 5.50,
    flag_low: float = 5.10,
    pole_height: float = 0.80,
    pole_gain_pct: float = 10.0,
    retracement_pct: float = 30.0,
) -> BullFlagPattern:
    """Create a BullFlagPattern with sensible defaults."""
    return BullFlagPattern(
        symbol=symbol,
        pole_start_idx=0,
        pole_end_idx=3,
        flag_start_idx=4,
        flag_end_idx=5,
        pole_low=4.70,
        pole_high=5.50,
        pole_height=pole_height,
        pole_gain_pct=pole_gain_pct,
        flag_low=flag_low,
        flag_high=5.40,
        retracement_pct=retracement_pct,
        pullback_candle_count=2,
        avg_pole_volume=10000,
        avg_flag_volume=5000,
        breakout_level=breakout_level,
    )


def _make_plan(
    symbol: str = "TEST",
    entry_price: float = 5.50,
    stop_loss_price: float = 5.30,
    take_profit_price: float = 5.90,
    shares: int = 90,
) -> TradePlan:
    """Create a TradePlan with sensible defaults."""
    pattern = _make_pattern(symbol=symbol, breakout_level=entry_price)
    risk = entry_price - stop_loss_price
    reward = take_profit_price - entry_price
    return TradePlan(
        symbol=symbol,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        risk_per_share=risk,
        reward_per_share=reward,
        risk_reward_ratio=reward / risk if risk > 0 else 0,
        shares=shares,
        total_risk=risk * shares,
        pattern=pattern,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simulator():
    """Fresh TradeSimulator instance."""
    return TradeSimulator()


@pytest.fixture
def detector():
    """Fresh BullFlagDetector with defaults."""
    return BullFlagDetector()


@pytest.fixture
def planner():
    """Fresh TradePlanner with defaults."""
    return TradePlanner(min_risk_per_share=0.05)


@pytest.fixture
def runner():
    """Fresh BacktestRunner with real detector/planner/simulator."""
    return BacktestRunner()


# ===========================================================================
# TradeSimulator Tests
# ===========================================================================


class TestTradeSimulatorTargetHit:
    """Test cases where the target price is reached."""

    def test_target_hit_on_next_bar(self, simulator):
        """Target hit immediately on the bar after entry."""
        plan = _make_plan(entry_price=5.50, stop_loss_price=5.30, take_profit_price=5.90)
        bars = _make_bars([
            (5.40, 5.55, 5.35, 5.50, 1000),  # bar 0: entry bar
            (5.50, 5.95, 5.45, 5.85, 2000),  # bar 1: target hit (high >= 5.90)
        ])
        trade = simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.exit_reason == 'target'
        assert trade.exit_price == 5.90
        assert trade.pnl == pytest.approx((5.90 - 5.50) * 90, abs=0.01)
        assert trade.pnl_pct == pytest.approx(7.27, abs=0.1)

    def test_target_hit_several_bars_later(self, simulator):
        """Target hit after several bars of consolidation."""
        plan = _make_plan(entry_price=5.50, stop_loss_price=5.30, take_profit_price=5.90)
        bars = _make_bars([
            (5.50, 5.55, 5.45, 5.52, 1000),  # entry
            (5.52, 5.60, 5.48, 5.55, 1000),  # consolidation
            (5.55, 5.65, 5.50, 5.60, 1000),  # consolidation
            (5.60, 5.92, 5.58, 5.88, 2000),  # target hit
        ])
        trade = simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.exit_reason == 'target'
        assert trade.exit_price == 5.90
        assert trade.exit_time == _ts(3)


class TestTradeSimulatorStopHit:
    """Test cases where the stop loss is triggered."""

    def test_stop_hit_on_next_bar(self, simulator):
        """Stop triggered immediately on the bar after entry."""
        plan = _make_plan(entry_price=5.50, stop_loss_price=5.30, take_profit_price=5.90)
        bars = _make_bars([
            (5.50, 5.55, 5.45, 5.50, 1000),  # entry
            (5.45, 5.48, 5.25, 5.30, 2000),  # stop hit (low <= 5.30)
        ])
        trade = simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.exit_reason == 'stop'
        assert trade.exit_price == 5.30
        assert trade.pnl == pytest.approx((5.30 - 5.50) * 90, abs=0.01)
        assert trade.pnl < 0

    def test_stop_hit_after_consolidation(self, simulator):
        """Stop hit after several neutral bars."""
        plan = _make_plan(entry_price=5.50, stop_loss_price=5.30, take_profit_price=5.90)
        bars = _make_bars([
            (5.50, 5.55, 5.45, 5.52, 1000),  # entry
            (5.52, 5.56, 5.48, 5.50, 1000),  # neutral
            (5.50, 5.52, 5.28, 5.32, 1500),  # stop hit
        ])
        trade = simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.exit_reason == 'stop'
        assert trade.exit_price == 5.30


class TestTradeSimulatorEOD:
    """Test cases where position is held until end of day."""

    def test_eod_exit_no_stop_no_target(self, simulator):
        """No stop or target hit — exits at last bar's close."""
        plan = _make_plan(entry_price=5.50, stop_loss_price=5.30, take_profit_price=5.90)
        bars = _make_bars([
            (5.50, 5.55, 5.45, 5.52, 1000),  # entry
            (5.52, 5.60, 5.48, 5.55, 1000),  # neutral
            (5.55, 5.65, 5.50, 5.60, 1000),  # neutral (last bar)
        ])
        trade = simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.exit_reason == 'eod'
        assert trade.exit_price == 5.60
        assert trade.pnl == pytest.approx((5.60 - 5.50) * 90, abs=0.01)

    def test_eod_exit_with_loss(self, simulator):
        """EOD exit at a loss when price drifts down but doesn't hit stop."""
        plan = _make_plan(entry_price=5.50, stop_loss_price=5.30, take_profit_price=5.90)
        bars = _make_bars([
            (5.50, 5.55, 5.45, 5.50, 1000),
            (5.50, 5.52, 5.35, 5.38, 1000),  # down but above stop
        ])
        trade = simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.exit_reason == 'eod'
        assert trade.exit_price == 5.38
        assert trade.pnl < 0


class TestTradeSimulatorAmbiguity:
    """Test same-bar stop+target ambiguity (conservative = stop wins)."""

    def test_both_stop_and_target_hit_same_bar(self, simulator):
        """When both levels are crossed in the same bar, stop wins."""
        plan = _make_plan(entry_price=5.50, stop_loss_price=5.30, take_profit_price=5.90)
        bars = _make_bars([
            (5.50, 5.55, 5.45, 5.50, 1000),  # entry
            (5.50, 5.95, 5.25, 5.60, 3000),  # both hit
        ])
        trade = simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.exit_reason == 'stop'
        assert trade.exit_price == 5.30


class TestSimulatedTradeDataclass:
    """Test SimulatedTrade dataclass defaults."""

    def test_defaults(self):
        """Default values are sensible."""
        trade = SimulatedTrade(
            symbol="TEST",
            entry_time=_ts(0),
            entry_price=5.50,
            stop_loss=5.30,
            take_profit=5.90,
            shares=100,
        )
        assert trade.exit_time is None
        assert trade.exit_price is None
        assert trade.exit_reason is None
        assert trade.pnl == 0.0
        assert trade.pnl_pct == 0.0


# ===========================================================================
# BacktestResult Tests
# ===========================================================================


class TestBacktestResult:
    """Test BacktestResult dataclass."""

    def test_summary_pnl_no_trades(self):
        """Summary P&L is zero when no trades."""
        result = BacktestResult(
            symbol="TEST", trade_date="2026-03-13", total_bars=100, patterns_detected=0
        )
        assert result.summary_pnl == 0.0

    def test_summary_pnl_multiple_trades(self):
        """Summary P&L sums across multiple trades."""
        t1 = SimulatedTrade(
            symbol="TEST", entry_time=_ts(0), entry_price=5.0,
            stop_loss=4.8, take_profit=5.4, shares=100, pnl=40.0,
        )
        t2 = SimulatedTrade(
            symbol="TEST", entry_time=_ts(10), entry_price=6.0,
            stop_loss=5.8, take_profit=6.4, shares=100, pnl=-20.0,
        )
        result = BacktestResult(
            symbol="TEST", trade_date="2026-03-13", total_bars=100,
            patterns_detected=2, trades_simulated=[t1, t2],
        )
        assert result.summary_pnl == pytest.approx(20.0)


# ===========================================================================
# BacktestRunner Tests
# ===========================================================================


class TestBacktestRunnerEdgeCases:
    """Test BacktestRunner edge cases."""

    def test_too_few_bars(self):
        """Runner returns empty result when not enough bars."""
        runner = BacktestRunner()
        bars = _make_bars([(5.0, 5.1, 4.9, 5.0, 100)] * 3)
        result = runner.run("TEST", bars, "2026-03-13")

        assert result.total_bars == 3
        assert result.patterns_detected == 0
        assert len(result.trades_simulated) == 0

    def test_no_patterns_found(self):
        """Runner returns zero patterns when bars don't form a flag."""
        runner = BacktestRunner()
        # Flat bars — no pattern
        flat_bars = _make_bars([(5.0, 5.05, 4.95, 5.0, 1000)] * 20)
        result = runner.run("TEST", flat_bars, "2026-03-13")

        assert result.patterns_detected == 0
        assert len(result.trades_simulated) == 0

    def test_one_trade_per_day_limit(self):
        """Runner only takes first valid trade, counts remaining patterns."""
        mock_detector = MagicMock(spec=BullFlagDetector)
        mock_planner = MagicMock(spec=TradePlanner)
        mock_simulator = MagicMock(spec=TradeSimulator)

        pattern = _make_pattern()
        plan = _make_plan()
        trade = SimulatedTrade(
            symbol="TEST", entry_time=_ts(7), entry_price=5.50,
            stop_loss=5.30, take_profit=5.90, shares=90,
            exit_time=_ts(10), exit_price=5.90, exit_reason='target',
            pnl=36.0, pnl_pct=7.27,
        )

        # Detector finds pattern on every bar from 7 onward
        mock_detector.detect.return_value = pattern
        mock_planner.create_plan.return_value = plan
        mock_simulator.simulate.return_value = trade

        # Disable early exit to test that patterns are still counted after trade
        runner = BacktestRunner(
            detector=mock_detector, planner=mock_planner, simulator=mock_simulator,
            early_exit_after_trade=False,
        )
        bars = _make_bars([(5.0, 5.1, 4.9, 5.0, 1000)] * 20)
        result = runner.run("TEST", bars, "2026-03-13")

        # Should have many pattern detections but only 1 trade
        assert result.patterns_detected > 1
        assert len(result.trades_simulated) == 1
        mock_simulator.simulate.assert_called_once()


class TestBacktestRunnerWithMocks:
    """Test runner behavior with controlled mock components."""

    def test_pattern_detected_but_plan_rejected(self):
        """Pattern found but planner rejects it — no trade taken."""
        mock_detector = MagicMock(spec=BullFlagDetector)
        mock_planner = MagicMock(spec=TradePlanner)

        pattern = _make_pattern()
        mock_detector.detect.return_value = pattern
        mock_planner.create_plan.return_value = None  # Rejected

        runner = BacktestRunner(detector=mock_detector, planner=mock_planner)
        bars = _make_bars([(5.0, 5.1, 4.9, 5.0, 1000)] * 20)
        result = runner.run("TEST", bars, "2026-03-13")

        assert result.patterns_detected > 0
        assert len(result.trades_simulated) == 0

    def test_no_detection_no_trade(self):
        """No patterns detected → no trades."""
        mock_detector = MagicMock(spec=BullFlagDetector)
        mock_detector.detect.return_value = None

        runner = BacktestRunner(detector=mock_detector)
        bars = _make_bars([(5.0, 5.1, 4.9, 5.0, 1000)] * 20)
        result = runner.run("TEST", bars, "2026-03-13")

        assert result.patterns_detected == 0
        assert len(result.trades_simulated) == 0


# ===========================================================================
# Integration Test: Full pipeline with synthetic bull flag bars
# ===========================================================================


class TestBacktestIntegration:
    """
    End-to-end integration test using real detector, planner, and simulator
    with synthetic bars that form a valid bull flag pattern.
    """

    def _build_bull_flag_bars(self) -> pd.DataFrame:
        """
        Build synthetic 1-min bars forming a bull flag pattern.

        Structure:
        - Bars 0-4:  Pole (strong uptrend with green candles, high volume)
        - Bars 5-6:  Pullback/flag (red candles, lower volume)
        - Bar 7:     Breakout candle (green, high volume, closes above flag high)
        - Bar 8:     Dummy bar (dropped by detector as "in-progress")
        - Bars 9-14: Post-breakout bars for trade simulation
        """
        candles = [
            # Pole candles (strong green, high volume)
            (4.50, 4.65, 4.48, 4.60, 50000),   # bar 0: +2.2%
            (4.60, 4.80, 4.58, 4.75, 60000),   # bar 1: +3.3%
            (4.75, 5.00, 4.73, 4.95, 70000),   # bar 2: +4.2%
            (4.95, 5.20, 4.93, 5.15, 65000),   # bar 3: +4.0%
            (5.15, 5.40, 5.13, 5.35, 55000),   # bar 4: +3.9%, pole top

            # Flag/pullback candles (red, declining volume)
            (5.35, 5.38, 5.18, 5.20, 20000),   # bar 5: pullback
            (5.20, 5.25, 5.10, 5.12, 15000),   # bar 6: pullback continues

            # Breakout candle (green, volume surge, closes above flag high)
            (5.15, 5.50, 5.12, 5.45, 80000),   # bar 7: breakout!

            # "In-progress" bar — detector will drop this
            (5.45, 5.55, 5.40, 5.50, 30000),   # bar 8

            # Post-breakout bars for trade simulation
            (5.50, 5.60, 5.45, 5.55, 25000),   # bar 9
            (5.55, 5.70, 5.50, 5.65, 30000),   # bar 10
            (5.65, 5.80, 5.60, 5.75, 28000),   # bar 11
            (5.75, 5.90, 5.70, 5.85, 32000),   # bar 12
            (5.85, 6.10, 5.82, 6.05, 40000),   # bar 13: target likely hit
            (6.00, 6.10, 5.95, 6.00, 20000),   # bar 14
        ]
        return _make_bars(candles)

    def test_full_backtest_with_real_components(self):
        """
        Integration test: real detector + planner + simulator with synthetic
        bull flag bars. Validates the complete data flow.
        """
        bars = self._build_bull_flag_bars()

        runner = BacktestRunner(
            detector=BullFlagDetector(),
            planner=TradePlanner(min_risk_per_share=0.05),
            simulator=TradeSimulator(),
        )

        result = runner.run("TEST", bars, "2026-03-13")

        # The bull flag should be detected
        assert result.patterns_detected >= 1, (
            f"Expected at least 1 pattern, got {result.patterns_detected}"
        )

        # Verify pattern details recorded
        assert len(result.pattern_details) >= 1

        # If a trade was taken, validate its structure
        if result.trades_simulated:
            trade = result.trades_simulated[0]
            assert trade.symbol == "TEST"
            assert trade.entry_price > 0
            assert trade.exit_price is not None
            assert trade.exit_reason in ('target', 'stop', 'eod')
            assert trade.shares > 0
            # P&L should be consistent
            expected_pnl = (trade.exit_price - trade.entry_price) * trade.shares
            assert trade.pnl == pytest.approx(expected_pnl, abs=0.01)

    def test_backtest_result_consistency(self):
        """Validate BacktestResult fields are internally consistent."""
        bars = self._build_bull_flag_bars()
        runner = BacktestRunner()
        result = runner.run("TEST", bars, "2026-03-13")

        assert result.symbol == "TEST"
        assert result.trade_date == "2026-03-13"
        assert result.total_bars == len(bars)
        assert result.patterns_detected == len(result.pattern_details)
        assert len(result.trades_simulated) <= 1  # One trade per day


# ===========================================================================
# print_report Tests
# ===========================================================================


class TestPrintReport:
    """Test report printing doesn't crash."""

    def test_print_report_no_trades(self, capsys):
        """Report prints cleanly when no trades."""
        result = BacktestResult(
            symbol="TEST", trade_date="2026-03-13", total_bars=100, patterns_detected=0
        )
        print_report(result)
        output = capsys.readouterr().out
        assert "TEST" in output
        assert "No trades taken" in output

    def test_print_report_with_trade(self, capsys):
        """Report prints trade details."""
        trade = SimulatedTrade(
            symbol="TEST", entry_time=_ts(0), entry_price=5.50,
            stop_loss=5.30, take_profit=5.90, shares=90,
            exit_time=_ts(5), exit_price=5.90, exit_reason='target',
            pnl=36.0, pnl_pct=7.27,
        )
        detection = PatternDetection(
            bar_index=7, timestamp=_ts(7), pattern=_make_pattern()
        )
        result = BacktestResult(
            symbol="TEST", trade_date="2026-03-13", total_bars=100,
            patterns_detected=1, trades_simulated=[trade],
            pattern_details=[detection],
        )
        print_report(result)
        output = capsys.readouterr().out
        assert "target" in output
        assert "$36.00" in output


# ===========================================================================
# AlpacaClient.get_historical_1min_bars Tests
# ===========================================================================


class TestGetHistorical1MinBars:
    """Unit tests for the new AlpacaClient.get_historical_1min_bars method."""

    @pytest.fixture
    def mock_sdk_clients(self):
        """Patch all Alpaca SDK client constructors."""
        with patch("data_sources.alpaca_client.StockHistoricalDataClient") as mock_data_cls, \
             patch("data_sources.alpaca_client.TradingClient") as mock_trading_cls, \
             patch("data_sources.alpaca_client.NewsClient") as mock_news_cls:
            mock_data_inst = MagicMock()
            mock_trading_inst = MagicMock()
            mock_news_inst = MagicMock()
            mock_data_cls.return_value = mock_data_inst
            mock_trading_cls.return_value = mock_trading_inst
            mock_news_cls.return_value = mock_news_inst
            yield {
                "data_client": mock_data_inst,
                "trading_client": mock_trading_inst,
                "news_client": mock_news_inst,
            }

    @pytest.fixture
    def client(self, mock_sdk_clients):
        """AlpacaClient with mocked SDK clients."""
        from data_sources.alpaca_client import AlpacaClient
        c = AlpacaClient(api_key="test-key", api_secret="test-secret")
        c._api_timeout = 5
        return c

    def test_returns_dataframe_with_correct_columns(self, client, mock_sdk_clients):
        """Returned DataFrame has expected columns."""
        mock_bar = MagicMock()
        mock_bar.timestamp = _ts(0)
        mock_bar.open = 5.0
        mock_bar.high = 5.1
        mock_bar.low = 4.9
        mock_bar.close = 5.05
        mock_bar.volume = 1000

        mock_sdk_clients["data_client"].get_stock_bars.return_value = {"TEST": [mock_bar]}

        start = _ts(0)
        end = _ts(30)
        df = client.get_historical_1min_bars("TEST", start, end)

        assert not df.empty
        assert list(df.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        assert len(df) == 1
        assert df.iloc[0]['open'] == 5.0
        assert df.iloc[0]['volume'] == 1000

    def test_returns_empty_dataframe_when_no_data(self, client, mock_sdk_clients):
        """Returns empty DataFrame when API returns no bars."""
        mock_sdk_clients["data_client"].get_stock_bars.return_value = {}

        start = _ts(0)
        end = _ts(30)
        df = client.get_historical_1min_bars("NONE", start, end)

        assert df.empty

    def test_raises_on_api_failure(self, client, mock_sdk_clients):
        """Raises AlpacaAPIError on exception."""
        from data_sources.alpaca_client import AlpacaAPIError

        mock_sdk_clients["data_client"].get_stock_bars.side_effect = Exception("API down")

        start = _ts(0)
        end = _ts(30)
        with pytest.raises(AlpacaAPIError, match="Failed to get historical"):
            client.get_historical_1min_bars("TEST", start, end)


# ===========================================================================
# TradeSimulator Force Close Tests
# ===========================================================================


class TestTradeSimulatorForceClose:
    """Test force_close_time_utc exits trades at configured time."""

    def test_force_close_exits_at_bar_open(self):
        """Force close exits at bar open when timestamp >= force_close_time."""
        simulator = TradeSimulator(force_close_time_utc=14.25)  # 14:15 UTC
        plan = _make_plan(entry_price=5.50, stop_loss_price=5.30, take_profit_price=5.90)

        bars = _make_bars([
            (5.50, 5.55, 5.45, 5.50, 1000),   # bar 0 at 14:00 — entry
            (5.52, 5.58, 5.48, 5.55, 1000),   # bar 1 at 14:01 — hold
        ], start_minute=0)

        # Add a bar at 14:15 (minute 15) that triggers force close
        force_close_bar = {
            'timestamp': _ts(15),
            'open': 5.55, 'high': 5.58, 'low': 5.52, 'close': 5.56, 'volume': 1000,
        }
        bars_with_fc = pd.concat([bars, pd.DataFrame([force_close_bar])], ignore_index=True)

        trade = simulator.simulate(plan, bars_with_fc, entry_bar_idx=0)

        assert trade.exit_reason == 'force_close'
        assert trade.exit_price == 5.55  # bar open

    def test_no_force_close_when_disabled(self):
        """Without force_close_time, trade runs until stop/target/eod."""
        simulator = TradeSimulator(force_close_time_utc=None)
        plan = _make_plan(entry_price=5.50, stop_loss_price=5.30, take_profit_price=5.90)

        bars = _make_bars([
            (5.50, 5.55, 5.45, 5.50, 1000),
            (5.52, 5.58, 5.48, 5.55, 1000),
            (5.55, 5.60, 5.52, 5.58, 1000),
        ])

        trade = simulator.simulate(plan, bars, entry_bar_idx=0)
        assert trade.exit_reason == 'eod'  # Not force_close


class TestTradeSimulatorEntryOverride:
    """Test entry_price_override for realistic buy-stop fills."""

    def test_entry_override_uses_custom_price(self):
        """When entry_price_override is set, trade uses that price."""
        simulator = TradeSimulator()
        plan = _make_plan(entry_price=5.50, stop_loss_price=5.30, take_profit_price=5.90)

        bars = _make_bars([
            (5.55, 5.60, 5.45, 5.58, 1000),   # entry bar
            (5.58, 5.95, 5.55, 5.88, 2000),   # target hit
        ])

        trade = simulator.simulate(plan, bars, entry_bar_idx=0, entry_price_override=5.55)

        assert trade.entry_price == 5.55
        assert trade.planned_entry == 5.50
        assert trade.entry_gap == pytest.approx(0.05, abs=0.001)
        assert trade.pnl == pytest.approx((5.90 - 5.55) * 90, abs=0.01)

    def test_no_override_uses_plan_price(self):
        """Without override, trade uses plan.entry_price."""
        simulator = TradeSimulator()
        plan = _make_plan(entry_price=5.50, stop_loss_price=5.30, take_profit_price=5.90)

        bars = _make_bars([
            (5.50, 5.55, 5.45, 5.52, 1000),
            (5.52, 5.95, 5.50, 5.88, 2000),
        ])

        trade = simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.entry_price == 5.50
        assert trade.planned_entry == 5.50
        assert trade.entry_gap == 0.0


# ===========================================================================
# BacktestRunner Realistic Mode Tests
# ===========================================================================


class TestBacktestRunnerRealistic:
    """Test BacktestRunner in realistic mode with pending buy-stops."""

    def test_pending_buystop_triggers_on_breakout(self):
        """Pending buy-stop triggers when bar reaches breakout_level."""
        candles = [
            # Pole: 3 green
            (4.00, 4.15, 3.98, 4.13, 200000),
            (4.13, 4.30, 4.11, 4.28, 180000),
            (4.28, 4.52, 4.26, 4.50, 160000),
            # Flag: 2 red
            (4.50, 4.52, 4.38, 4.40, 50000),
            (4.40, 4.42, 4.33, 4.35, 30000),
            # Calm bar (setup detectable after this)
            (4.35, 4.38, 4.32, 4.34, 25000),
            # Breakout bar
            (4.38, 4.60, 4.36, 4.55, 250000),
            # Post-breakout
            (4.55, 4.65, 4.52, 4.62, 120000),
            (4.62, 4.72, 4.58, 4.70, 110000),
            (4.70, 4.85, 4.68, 4.82, 100000),
            (4.82, 4.95, 4.78, 4.92, 95000),
            (4.92, 5.10, 4.90, 5.05, 90000),
            # Dummy
            (5.05, 5.10, 5.00, 5.08, 50000),
        ]
        bars = _make_bars(candles)

        runner = BacktestRunner(
            planner=TradePlanner(min_risk_per_share=0.01),
            realistic=True,
            min_price=0.0,
        )
        result = runner.run("TEST", bars, "2026-03-13")

        assert len(result.trades_simulated) >= 1
        if result.trades_simulated:
            trade = result.trades_simulated[0]
            assert trade.planned_entry is not None
            assert trade.entry_gap >= 0

    def test_realistic_entry_is_max_open_breakout(self):
        """Realistic entry = max(bar_open, breakout_level)."""
        # When bar opens above breakout_level, fill at open
        candles = [
            (4.00, 4.15, 3.98, 4.13, 200000),
            (4.13, 4.30, 4.11, 4.28, 180000),
            (4.28, 4.52, 4.26, 4.50, 160000),
            (4.50, 4.52, 4.38, 4.40, 50000),
            (4.40, 4.42, 4.33, 4.35, 30000),
            (4.35, 4.38, 4.32, 4.34, 25000),
            # Gap-over breakout: opens at 4.60, above flag_high (~4.52)
            (4.60, 4.75, 4.58, 4.72, 300000),
            (4.72, 4.85, 4.70, 4.82, 120000),
            (4.82, 4.95, 4.80, 4.92, 110000),
            (4.92, 5.10, 4.90, 5.05, 100000),
            (5.05, 5.20, 5.00, 5.15, 90000),
            # Dummy
            (5.15, 5.20, 5.10, 5.18, 50000),
        ]
        bars = _make_bars(candles)

        runner = BacktestRunner(
            planner=TradePlanner(min_risk_per_share=0.01),
            realistic=True,
            min_price=0.0,
        )
        result = runner.run("TEST", bars, "2026-03-13")

        if result.trades_simulated:
            trade = result.trades_simulated[0]
            # Entry should be at bar open (4.60), not breakout_level
            assert trade.entry_price >= 4.55
            assert trade.entry_gap > 0

    def test_min_price_default_is_2(self):
        """Default min_price is now $2.00."""
        runner = BacktestRunner()
        assert runner.min_price == 2.0

    def test_pending_buystop_dataclass(self):
        """PendingBuyStop dataclass stores correct fields."""
        setup = BullFlagSetup(
            symbol="TEST", pole_start_idx=0, pole_end_idx=2,
            flag_start_idx=3, flag_end_idx=4,
            pole_low=4.00, pole_high=4.50, pole_height=0.50,
            pole_gain_pct=12.5, flag_low=4.33, flag_high=4.42,
            retracement_pct=30.0, pullback_candle_count=2,
            avg_pole_volume=180000, avg_flag_volume=40000,
            breakout_level=4.42,
        )
        plan = _make_plan()
        pending = PendingBuyStop(
            setup=setup, plan=plan,
            placed_at_bar_idx=5, breakout_level=4.42,
        )

        assert pending.breakout_level == 4.42
        assert pending.placed_at_bar_idx == 5
        assert pending.setup.symbol == "TEST"

    def test_simulated_trade_new_fields(self):
        """SimulatedTrade has planned_entry and entry_gap fields."""
        trade = SimulatedTrade(
            symbol="TEST",
            entry_time=_ts(0),
            entry_price=5.55,
            stop_loss=5.30,
            take_profit=5.90,
            shares=100,
            planned_entry=5.50,
            entry_gap=0.05,
        )
        assert trade.planned_entry == 5.50
        assert trade.entry_gap == 0.05

    def test_simulated_trade_default_new_fields(self):
        """New fields default to None/0.0 for backward compat."""
        trade = SimulatedTrade(
            symbol="TEST",
            entry_time=_ts(0),
            entry_price=5.50,
            stop_loss=5.30,
            take_profit=5.90,
            shares=100,
        )
        assert trade.planned_entry is None
        assert trade.entry_gap == 0.0

    def test_gap_fill_adjusts_stop_loss(self):
        """When entry gaps above breakout, stop moves up to maintain planned risk."""
        # Pattern: pole 3 green, flag 2 red, then gap-over breakout
        candles = [
            (4.00, 4.15, 3.98, 4.13, 200000),  # pole
            (4.13, 4.30, 4.11, 4.28, 180000),
            (4.28, 4.52, 4.26, 4.50, 160000),
            (4.50, 4.52, 4.38, 4.40, 50000),    # flag
            (4.40, 4.42, 4.33, 4.35, 30000),
            (4.35, 4.38, 4.32, 4.34, 25000),    # calm (setup detected)
            # Gap-over: opens at 4.70, breakout was ~4.42 (flag_high)
            # Gap = $0.28. Planned risk ~$0.09 (4.42-4.33).
            # Without fix: stop stays at 4.33, risk = 4.70-4.33 = $0.37/sh
            # With fix: stop = 4.70-0.09 = 4.61, risk = $0.09/sh (same as planned)
            (4.70, 4.75, 4.58, 4.72, 300000),
            # Reversal hits new tighter stop but not old stop
            (4.72, 4.73, 4.59, 4.60, 120000),
            # Dummy trailing bars
            (4.60, 4.62, 4.55, 4.58, 100000),
        ]
        bars = _make_bars(candles)

        runner = BacktestRunner(
            planner=TradePlanner(min_risk_per_share=0.01),
            realistic=True,
            min_price=0.0,
        )
        result = runner.run("TEST", bars, "2026-03-13")

        if result.trades_simulated:
            trade = result.trades_simulated[0]
            # Entry should be at gap open (~4.70)
            assert trade.entry_price >= 4.65
            assert trade.entry_gap > 0.2  # Significant gap
            # Stop should have been adjusted UP, not at original flag_low (4.33)
            assert trade.stop_loss > 4.40, (
                f"Stop should be adjusted up for gap fill, got {trade.stop_loss}"
            )

    def test_no_gap_no_stop_adjustment(self):
        """When entry is at breakout level (no gap), stop stays at flag_low."""
        candles = [
            (4.00, 4.15, 3.98, 4.13, 200000),
            (4.13, 4.30, 4.11, 4.28, 180000),
            (4.28, 4.52, 4.26, 4.50, 160000),
            (4.50, 4.52, 4.38, 4.40, 50000),
            (4.40, 4.42, 4.33, 4.35, 30000),
            (4.35, 4.38, 4.32, 4.34, 25000),
            # Clean breakout at flag_high, no gap
            (4.42, 4.60, 4.40, 4.55, 250000),
            (4.55, 4.65, 4.52, 4.62, 120000),
            (4.62, 4.75, 4.58, 4.70, 110000),
            (4.70, 4.85, 4.68, 4.82, 100000),
            (4.82, 4.95, 4.78, 4.92, 95000),
            (4.92, 5.10, 4.90, 5.05, 90000),
        ]
        bars = _make_bars(candles)

        runner = BacktestRunner(
            planner=TradePlanner(min_risk_per_share=0.01),
            realistic=True,
            min_price=0.0,
        )
        result = runner.run("TEST", bars, "2026-03-13")

        if result.trades_simulated:
            trade = result.trades_simulated[0]
            # No gap → stop stays at original flag_low area
            assert trade.entry_gap < 0.01 or trade.stop_loss <= 4.40


# ===========================================================================
# TradeSimulator Partial Profit Tests
# ===========================================================================


@pytest.fixture
def partial_simulator():
    """TradeSimulator with partial profit enabled."""
    return TradeSimulator(partial_profit_enabled=True)


class TestTradeSimulatorPartialProfit:
    """Test partial profit exit strategy (sell half at +1R, move stop to breakeven)."""

    def test_partial_disabled_by_default(self, simulator):
        """Default simulator does not take partial profits."""
        plan = _make_plan(entry_price=5.00, stop_loss_price=4.90, take_profit_price=5.25, shares=100)
        bars = _make_bars([
            (5.00, 5.05, 4.95, 5.00, 1000),  # entry bar
            (5.00, 5.12, 4.98, 5.10, 1000),   # +1R hit (5.10) but not target
            (5.10, 5.30, 5.08, 5.28, 1000),   # target hit
        ])
        trade = simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.partial_exit_taken is False
        assert trade.exit_reason == 'target'
        assert trade.pnl == pytest.approx((5.25 - 5.00) * 100, abs=0.01)

    def test_partial_then_breakeven_stop(self, partial_simulator):
        """Price hits +1R, sells half, then reverses to breakeven stop."""
        # entry=5.00, stop=4.90, risk=0.10, partial_target=5.10, target=5.25
        plan = _make_plan(entry_price=5.00, stop_loss_price=4.90, take_profit_price=5.25, shares=100)
        bars = _make_bars([
            (5.00, 5.05, 4.95, 5.02, 1000),   # bar 0: entry
            (5.02, 5.12, 5.00, 5.10, 1000),    # bar 1: +1R hit (5.10)
            (5.10, 5.12, 5.05, 5.08, 1000),    # bar 2: drifts down
            (5.08, 5.09, 4.98, 5.00, 1000),    # bar 3: hits breakeven stop (5.00)
        ])
        trade = partial_simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.partial_exit_taken is True
        assert trade.partial_shares == 50
        assert trade.partial_exit_price == pytest.approx(5.10, abs=0.01)
        assert trade.partial_pnl == pytest.approx((5.10 - 5.00) * 50, abs=0.01)
        assert trade.breakeven_stop_active is True
        assert trade.exit_reason == 'partial+breakeven'
        # Remaining 50 shares exit at breakeven (5.00), P&L = 0
        # Total P&L = partial_pnl + 0 = 5.00
        assert trade.pnl == pytest.approx(5.00, abs=0.01)

    def test_partial_then_target(self, partial_simulator):
        """Price hits +1R, sells half, then hits full target."""
        plan = _make_plan(entry_price=5.00, stop_loss_price=4.90, take_profit_price=5.25, shares=100)
        bars = _make_bars([
            (5.00, 5.05, 4.95, 5.02, 1000),   # entry
            (5.02, 5.12, 5.00, 5.10, 1000),    # +1R hit
            (5.10, 5.20, 5.08, 5.18, 1000),    # approaching target
            (5.18, 5.30, 5.16, 5.28, 1000),    # target hit (5.25)
        ])
        trade = partial_simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.partial_exit_taken is True
        assert trade.exit_reason == 'partial+target'
        # 50 shares at +1R (5.10): pnl = 50 * 0.10 = 5.00
        # 50 shares at target (5.25): pnl = 50 * 0.25 = 12.50
        # Total = 17.50
        assert trade.pnl == pytest.approx(17.50, abs=0.01)

    def test_straight_to_stop_no_partial(self, partial_simulator):
        """Price goes straight to stop without hitting +1R."""
        plan = _make_plan(entry_price=5.00, stop_loss_price=4.90, take_profit_price=5.25, shares=100)
        bars = _make_bars([
            (5.00, 5.05, 4.95, 5.00, 1000),   # entry
            (5.00, 5.02, 4.88, 4.90, 1000),    # stop hit
        ])
        trade = partial_simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.partial_exit_taken is False
        assert trade.exit_reason == 'stop'
        assert trade.pnl == pytest.approx((4.90 - 5.00) * 100, abs=0.01)

    def test_partial_then_eod(self, partial_simulator):
        """Price hits +1R, sells half, then EOD exit."""
        plan = _make_plan(entry_price=5.00, stop_loss_price=4.90, take_profit_price=5.25, shares=100)
        bars = _make_bars([
            (5.00, 5.05, 4.95, 5.02, 1000),   # entry
            (5.02, 5.12, 5.00, 5.10, 1000),    # +1R hit
            (5.10, 5.15, 5.05, 5.12, 1000),    # last bar — EOD
        ])
        trade = partial_simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.partial_exit_taken is True
        assert trade.exit_reason == 'partial+eod'
        # 50 shares at 5.10: pnl = 5.00
        # 50 shares at 5.12 (EOD close): pnl = 50 * 0.12 = 6.00
        # Total = 11.00
        assert trade.pnl == pytest.approx(11.00, abs=0.01)

    def test_partial_then_force_close(self):
        """Price hits +1R, sells half, then force close triggers."""
        sim = TradeSimulator(
            force_close_time_utc=14.25,  # 14:15 UTC
            partial_profit_enabled=True,
        )
        plan = _make_plan(entry_price=5.00, stop_loss_price=4.90, take_profit_price=5.25, shares=100)

        # Bar at 14:00 (entry), 14:01 (+1R), 14:15 (force close)
        bars = _make_bars([
            (5.00, 5.05, 4.95, 5.02, 1000),   # bar 0 at 14:00 — entry
            (5.02, 5.12, 5.00, 5.10, 1000),    # bar 1 at 14:01 — +1R hit
        ], start_minute=0)
        # Add a bar at 14:15
        fc_bar = {
            'timestamp': _ts(15),
            'open': 5.08, 'high': 5.10, 'low': 5.05, 'close': 5.07, 'volume': 1000,
        }
        bars_with_fc = pd.concat([bars, pd.DataFrame([fc_bar])], ignore_index=True)

        trade = sim.simulate(plan, bars_with_fc, entry_bar_idx=0)

        assert trade.partial_exit_taken is True
        assert trade.exit_reason == 'partial+force_close'
        # 50 shares at 5.10: pnl = 5.00
        # 50 shares at 5.08 (force close open): pnl = 50 * 0.08 = 4.00
        # Total = 9.00
        assert trade.pnl == pytest.approx(9.00, abs=0.01)

    def test_partial_and_target_same_bar(self, partial_simulator):
        """Same bar hits both +1R and full target."""
        plan = _make_plan(entry_price=5.00, stop_loss_price=4.90, take_profit_price=5.25, shares=100)
        bars = _make_bars([
            (5.00, 5.05, 4.95, 5.02, 1000),   # entry
            (5.02, 5.30, 5.00, 5.28, 2000),    # both +1R (5.10) and target (5.25) hit
        ])
        trade = partial_simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.partial_exit_taken is True
        assert trade.exit_reason == 'partial+target'
        # 50 shares at 5.10: pnl = 5.00
        # 50 shares at 5.25: pnl = 50 * 0.25 = 12.50
        # Total = 17.50
        assert trade.pnl == pytest.approx(17.50, abs=0.01)

    def test_stop_and_partial_same_bar(self, partial_simulator):
        """Same bar touches both stop and +1R — conservative: stop wins."""
        plan = _make_plan(entry_price=5.00, stop_loss_price=4.90, take_profit_price=5.25, shares=100)
        bars = _make_bars([
            (5.00, 5.05, 4.95, 5.02, 1000),   # entry
            (5.00, 5.12, 4.88, 4.95, 2000),    # both stop (4.90) and +1R (5.10)
        ])
        trade = partial_simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.partial_exit_taken is False
        assert trade.exit_reason == 'stop'
        assert trade.pnl == pytest.approx((4.90 - 5.00) * 100, abs=0.01)

    def test_partial_shares_rounding(self, partial_simulator):
        """Odd share count rounds down for partial, remaining gets the extra."""
        plan = _make_plan(entry_price=5.00, stop_loss_price=4.90, take_profit_price=5.25, shares=91)
        bars = _make_bars([
            (5.00, 5.05, 4.95, 5.02, 1000),   # entry
            (5.02, 5.12, 5.00, 5.10, 1000),    # +1R hit
            (5.10, 5.30, 5.08, 5.28, 1000),    # target hit
        ])
        trade = partial_simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.partial_shares == 45  # 91 // 2
        assert trade.remaining_shares == 46  # 91 - 45
        assert trade.exit_reason == 'partial+target'

    def test_pnl_pct_on_full_position(self, partial_simulator):
        """P&L percent is calculated relative to total entry cost, not remaining."""
        plan = _make_plan(entry_price=5.00, stop_loss_price=4.90, take_profit_price=5.25, shares=100)
        bars = _make_bars([
            (5.00, 5.05, 4.95, 5.02, 1000),   # entry
            (5.02, 5.12, 5.00, 5.10, 1000),    # +1R hit
            (5.10, 5.12, 4.98, 5.00, 1000),    # breakeven stop hit
        ])
        trade = partial_simulator.simulate(plan, bars, entry_bar_idx=0)

        # Total position value = 5.00 * 100 = 500
        # Total P&L = 5.00 (from partial)
        # P&L % = 5.00 / 500 * 100 = 1.0%
        assert trade.pnl_pct == pytest.approx(1.0, abs=0.01)


# ===========================================================================
# TradeSimulator Breakeven-Stop-Only Tests (fraction=0.0)
# ===========================================================================


@pytest.fixture
def breakeven_simulator():
    """TradeSimulator with breakeven stop at +1R, no partial sell."""
    return TradeSimulator(
        partial_profit_enabled=True,
        partial_profit_fraction=0.0,
    )


class TestTradeSimulatorBreakevenStopOnly:
    """Test breakeven stop move without selling any shares (fraction=0)."""

    def test_breakeven_then_stop_at_entry(self, breakeven_simulator):
        """Price hits +1R, stop moves to breakeven, then reverses to entry → $0 P&L."""
        plan = _make_plan(entry_price=5.00, stop_loss_price=4.90, take_profit_price=5.25, shares=100)
        bars = _make_bars([
            (5.00, 5.05, 4.95, 5.02, 1000),   # entry
            (5.02, 5.12, 5.00, 5.10, 1000),    # +1R hit → stop moves to 5.00
            (5.10, 5.12, 5.05, 5.08, 1000),    # drifts
            (5.08, 5.09, 4.98, 5.00, 1000),    # hits breakeven stop
        ])
        trade = breakeven_simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.partial_exit_taken is True
        assert trade.partial_shares == 0  # No shares sold at partial
        assert trade.partial_pnl == 0.0
        assert trade.remaining_shares == 100  # All shares remain
        assert trade.breakeven_stop_active is True
        assert trade.exit_reason == 'partial+breakeven'
        assert trade.pnl == pytest.approx(0.0, abs=0.01)  # Breakeven!
        assert trade.exit_price == 5.00

    def test_breakeven_then_target_full_profit(self, breakeven_simulator):
        """Price hits +1R, then hits target → full profit on all shares."""
        plan = _make_plan(entry_price=5.00, stop_loss_price=4.90, take_profit_price=5.25, shares=100)
        bars = _make_bars([
            (5.00, 5.05, 4.95, 5.02, 1000),   # entry
            (5.02, 5.12, 5.00, 5.10, 1000),    # +1R hit
            (5.10, 5.30, 5.08, 5.28, 1000),    # target hit
        ])
        trade = breakeven_simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.exit_reason == 'partial+target'
        # Full 100 shares at target: (5.25 - 5.00) * 100 = 25.00
        assert trade.pnl == pytest.approx(25.00, abs=0.01)

    def test_straight_to_stop_unchanged(self, breakeven_simulator):
        """Price never reaches +1R, hits original stop → same as no-partial."""
        plan = _make_plan(entry_price=5.00, stop_loss_price=4.90, take_profit_price=5.25, shares=100)
        bars = _make_bars([
            (5.00, 5.05, 4.95, 5.00, 1000),   # entry
            (5.00, 5.03, 4.88, 4.90, 1000),    # stop hit
        ])
        trade = breakeven_simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.partial_exit_taken is False
        assert trade.exit_reason == 'stop'
        assert trade.pnl == pytest.approx(-10.00, abs=0.01)  # Same as H10a

    def test_breakeven_preserves_all_shares_for_target(self, breakeven_simulator):
        """Verify no shares are sold at +1R — full position hits target."""
        plan = _make_plan(entry_price=5.00, stop_loss_price=4.90, take_profit_price=5.25, shares=200)
        bars = _make_bars([
            (5.00, 5.05, 4.95, 5.02, 1000),   # entry
            (5.02, 5.12, 5.00, 5.10, 1000),    # +1R hit
            (5.10, 5.15, 5.08, 5.12, 1000),    # hold
            (5.12, 5.30, 5.10, 5.28, 1000),    # target hit
        ])
        trade = breakeven_simulator.simulate(plan, bars, entry_bar_idx=0)

        assert trade.partial_shares == 0
        assert trade.remaining_shares == 200
        # Full 200 shares at target
        assert trade.pnl == pytest.approx((5.25 - 5.00) * 200, abs=0.01)
