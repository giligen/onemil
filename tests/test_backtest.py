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
    TradeSimulator,
    BacktestRunner,
    print_report,
)
from trading.pattern_detector import BullFlagDetector, BullFlagPattern
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
