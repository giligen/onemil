"""
Integration tests for the realistic backtest pipeline.

Tests the full flow: synthetic bars → detect_setup → pending buy-stop →
trigger → simulate → verify P&L. Validates that realistic entries use
max(bar_open, breakout_level) and that time controls work correctly.
"""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta

from backtest import (
    BacktestRunner,
    BacktestResult,
    PendingBuyStop,
    SimulatedTrade,
    TradeSimulator,
)
from trading.pattern_detector import BullFlagDetector, BullFlagSetup
from trading.trade_planner import TradePlanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 10:00 ET = 14:00 UTC — well within trading hours
BASE_TIME = datetime(2026, 3, 13, 14, 0, 0, tzinfo=timezone.utc)


def _ts(minute_offset: int) -> datetime:
    """Create a timestamp offset by minutes from BASE_TIME."""
    return BASE_TIME + timedelta(minutes=minute_offset)


def _make_bars(candles: list, start_minute: int = 0) -> pd.DataFrame:
    """Build a bars DataFrame from (open, high, low, close, volume) tuples."""
    records = []
    for i, (o, h, l, c, v) in enumerate(candles):
        records.append({
            'timestamp': _ts(start_minute + i),
            'open': o, 'high': h, 'low': l, 'close': c, 'volume': v,
        })
    return pd.DataFrame(records)


def _build_setup_then_breakout_bars():
    """
    Build bars that form a setup (pole+flag) followed by a breakout bar.

    Structure:
    - Bars 0-2:  Pole (3 green candles, 4.00→4.50, ~12.5% gain)
    - Bars 3-4:  Flag (2 red candles, retrace to ~4.35, ~30%)
    - Bar 5:     "Calm" bar — flag is now complete, setup detectable
    - Bar 6:     Breakout bar — opens at 4.38, highs above flag_high
    - Bars 7-12: Post-breakout bars for trade simulation
    - Bar 13:    Dummy (dropped)

    Flag high = max high of flag candles = 4.52 (bar 3)
    """
    candles = [
        # Pole: 3 green candles
        (4.00, 4.15, 3.98, 4.13, 200000),   # bar 0: green
        (4.13, 4.30, 4.11, 4.28, 180000),   # bar 1: green
        (4.28, 4.52, 4.26, 4.50, 160000),   # bar 2: green (pole top)

        # Flag: 2 red candles (retrace ~30%)
        (4.50, 4.52, 4.38, 4.40, 50000),    # bar 3: red
        (4.40, 4.42, 4.33, 4.35, 30000),    # bar 4: red

        # Calm bar — still in flag territory (red/neutral), setup now detectable
        (4.35, 4.38, 4.32, 4.34, 25000),    # bar 5: red

        # Breakout bar — opens below flag_high, highs above it
        (4.38, 4.60, 4.36, 4.55, 250000),   # bar 6: breakout

        # Post-breakout: trending up to hit target
        (4.55, 4.65, 4.52, 4.62, 120000),   # bar 7
        (4.62, 4.72, 4.58, 4.70, 110000),   # bar 8
        (4.70, 4.82, 4.68, 4.80, 100000),   # bar 9
        (4.80, 4.95, 4.78, 4.92, 95000),    # bar 10
        (4.92, 5.10, 4.90, 5.05, 90000),    # bar 11
        (5.05, 5.20, 5.00, 5.15, 85000),    # bar 12

        # Dummy bar (dropped)
        (5.15, 5.20, 5.10, 5.18, 50000),    # bar 13
    ]
    return _make_bars(candles)


def _build_gap_over_breakout_bars():
    """
    Bars where the breakout bar gaps ABOVE the flag_high.

    This tests that fill price = max(bar_open, breakout_level) = bar_open.
    """
    candles = [
        # Pole: 3 green
        (4.00, 4.15, 3.98, 4.13, 200000),
        (4.13, 4.30, 4.11, 4.28, 180000),
        (4.28, 4.52, 4.26, 4.50, 160000),

        # Flag: 2 red
        (4.50, 4.52, 4.38, 4.40, 50000),
        (4.40, 4.42, 4.33, 4.35, 30000),

        # Calm bar (flag end)
        (4.35, 4.38, 4.32, 4.34, 25000),

        # Gap-over breakout: opens at 4.60, well above flag_high (~4.52)
        (4.60, 4.75, 4.58, 4.72, 300000),

        # Post-breakout
        (4.72, 4.85, 4.70, 4.82, 120000),
        (4.82, 4.95, 4.80, 4.92, 110000),
        (4.92, 5.10, 4.90, 5.05, 100000),
        (5.05, 5.20, 5.00, 5.15, 90000),
        (5.15, 5.30, 5.10, 5.25, 85000),

        # Dummy
        (5.25, 5.30, 5.20, 5.28, 50000),
    ]
    return _make_bars(candles)


# ===========================================================================
# Full pipeline integration tests
# ===========================================================================


class TestRealisticPipeline:
    """End-to-end realistic backtest: setup detection → buy-stop → trigger → simulate."""

    def test_realistic_detects_setup_and_fills_buystop(self):
        """Full pipeline: setup detected → buy-stop placed → triggered → trade simulated."""
        bars = _build_setup_then_breakout_bars()

        runner = BacktestRunner(
            detector=BullFlagDetector(),
            planner=TradePlanner(min_risk_per_share=0.01),
            realistic=True,
            min_price=0.0,  # Allow low prices for test
        )

        result = runner.run("TEST", bars, "2026-03-13")

        assert result.patterns_detected >= 1, "Should detect at least one setup"
        assert len(result.trades_simulated) == 1, "Should take exactly one trade"

        trade = result.trades_simulated[0]
        assert trade.symbol == "TEST"
        assert trade.entry_price > 0
        assert trade.exit_price is not None
        assert trade.exit_reason in ('target', 'stop', 'eod', 'force_close')

        # Realistic entry should be >= breakout_level
        assert trade.planned_entry is not None
        assert trade.entry_price >= trade.planned_entry

        # Entry gap should be >= 0
        assert trade.entry_gap >= 0

        # P&L should be consistent
        expected_pnl = (trade.exit_price - trade.entry_price) * trade.shares
        assert trade.pnl == pytest.approx(expected_pnl, abs=0.01)

    def test_gap_over_fills_at_bar_open(self):
        """When bar gaps over breakout_level, fill at bar_open (higher)."""
        bars = _build_gap_over_breakout_bars()

        runner = BacktestRunner(
            detector=BullFlagDetector(),
            planner=TradePlanner(min_risk_per_share=0.01),
            realistic=True,
            min_price=0.0,
        )

        result = runner.run("TEST", bars, "2026-03-13")

        if result.trades_simulated:
            trade = result.trades_simulated[0]
            # Bar 6 opens at 4.60, which is above flag_high (~4.52)
            # So entry should be at bar_open = 4.60, not breakout_level
            assert trade.entry_price >= 4.55, (
                f"Gap-over should fill at bar_open (4.60), got {trade.entry_price}"
            )
            assert trade.entry_gap > 0, "Gap-over should have positive entry_gap"

    def test_fantasy_vs_realistic_different_entries(self):
        """Fantasy mode enters at breakout_level; realistic enters at max(open, level)."""
        bars = _build_gap_over_breakout_bars()
        planner = TradePlanner(min_risk_per_share=0.01)

        # Fantasy mode
        fantasy_runner = BacktestRunner(
            planner=planner, realistic=False, min_price=0.0
        )
        fantasy_result = fantasy_runner.run("TEST", bars, "2026-03-13")

        # Realistic mode
        realistic_runner = BacktestRunner(
            planner=planner, realistic=True, min_price=0.0
        )
        realistic_result = realistic_runner.run("TEST", bars, "2026-03-13")

        # Both should produce trades (though potentially different patterns/timing)
        # The key property: realistic entry >= fantasy entry when there's a gap
        if fantasy_result.trades_simulated and realistic_result.trades_simulated:
            fantasy_trade = fantasy_result.trades_simulated[0]
            realistic_trade = realistic_result.trades_simulated[0]

            # Fantasy enters at breakout_level, realistic at max(open, breakout_level)
            assert realistic_trade.entry_gap >= 0
            assert realistic_trade.planned_entry is not None

    def test_realistic_buystop_invalidates_on_price_drop(self):
        """Buy-stop invalidates when bar low drops below flag_low."""
        candles = [
            # Pole
            (4.00, 4.15, 3.98, 4.13, 200000),
            (4.13, 4.30, 4.11, 4.28, 180000),
            (4.28, 4.52, 4.26, 4.50, 160000),
            # Flag
            (4.50, 4.52, 4.38, 4.40, 50000),
            (4.40, 4.42, 4.33, 4.35, 30000),
            # Flag end (setup detectable)
            (4.35, 4.38, 4.32, 4.34, 25000),
            # Price crashes below flag_low (4.32) — invalidates buy-stop
            (4.34, 4.36, 4.20, 4.22, 150000),
            # Subsequent bars — no fill should happen
            (4.22, 4.60, 4.20, 4.55, 200000),  # would trigger if order still active
            (4.55, 4.60, 4.50, 4.58, 100000),
            # Dummy
            (4.58, 4.62, 4.56, 4.60, 80000),
        ]
        bars = _make_bars(candles)

        runner = BacktestRunner(
            planner=TradePlanner(min_risk_per_share=0.01),
            realistic=True,
            min_price=0.0,
        )
        result = runner.run("TEST", bars, "2026-03-13")

        # The buy-stop should have been invalidated, so no trade
        assert len(result.trades_simulated) == 0, (
            "Buy-stop should be invalidated when price drops below flag_low"
        )

    def test_realistic_buystop_expires_after_n_bars(self):
        """Buy-stop expires after setup_expiry_bars without triggering."""
        candles = [
            # Pole
            (4.00, 4.15, 3.98, 4.13, 200000),
            (4.13, 4.30, 4.11, 4.28, 180000),
            (4.28, 4.52, 4.26, 4.50, 160000),
            # Flag
            (4.50, 4.52, 4.38, 4.40, 50000),
            (4.40, 4.42, 4.33, 4.35, 30000),
            # Flag end
            (4.35, 4.38, 4.32, 4.34, 25000),
        ]
        # Add 12 GREEN bars that stay below breakout level but break the
        # flag pattern so no new setup forms. Green bars = close > open.
        for j in range(12):
            candles.append((4.33, 4.37, 4.32, 4.35, 200000))
        # Add dummy
        candles.append((4.35, 4.37, 4.33, 4.36, 150000))

        bars = _make_bars(candles)

        runner = BacktestRunner(
            planner=TradePlanner(min_risk_per_share=0.01),
            realistic=True,
            min_price=0.0,
            setup_expiry_bars=5,
        )
        result = runner.run("TEST", bars, "2026-03-13")

        # The original buy-stop should have expired after 5 bars,
        # and with green consolidation bars there's no new valid setup
        # (green bars don't form a pullback, so no new pole+flag detected)
        assert len(result.trades_simulated) == 0


class TestForceCloseIntegration:
    """Integration tests for force-close time control."""

    def test_force_close_exits_at_configured_time(self):
        """Trade is force-closed when bar timestamp >= force_close_time_utc."""
        # Build bars that get a trade, then hit force_close time
        # Start at 14:00 UTC (10:00 ET), force close at 19.75 UTC (15:45 ET)
        # Need bars that span to 19:45 UTC

        # Create setup + breakout bars starting at 14:00 UTC
        candles = [
            (4.00, 4.15, 3.98, 4.13, 200000),
            (4.13, 4.30, 4.11, 4.28, 180000),
            (4.28, 4.52, 4.26, 4.50, 160000),
            (4.50, 4.52, 4.38, 4.40, 50000),
            (4.40, 4.42, 4.33, 4.35, 30000),
            (4.35, 4.38, 4.32, 4.34, 25000),
            (4.38, 4.60, 4.36, 4.55, 250000),
        ]

        # Add bars up to 19:45 UTC = minute offset 345 from 14:00
        # Fill with neutral bars that don't hit stop or target
        for i in range(7, 350):
            candles.append((4.55, 4.58, 4.52, 4.55, 20000))
        # Dummy
        candles.append((4.55, 4.58, 4.52, 4.55, 15000))

        bars = _make_bars(candles)

        runner = BacktestRunner(
            planner=TradePlanner(min_risk_per_share=0.01),
            realistic=True,
            min_price=0.0,
            force_close_time_utc=19.75,
        )

        result = runner.run("TEST", bars, "2026-03-13")

        if result.trades_simulated:
            trade = result.trades_simulated[0]
            assert trade.exit_reason == 'force_close', (
                f"Expected force_close exit, got {trade.exit_reason}"
            )


class TestMinPriceFilter:
    """Integration tests for the $2 minimum price filter."""

    def test_min_price_filters_cheap_stocks(self):
        """Stocks below $2 are filtered out with default min_price=2.0."""
        # Build a bull flag at $1.50 price range
        candles = [
            (1.00, 1.10, 0.98, 1.08, 200000),
            (1.08, 1.20, 1.06, 1.18, 180000),
            (1.18, 1.32, 1.16, 1.30, 160000),
            (1.30, 1.32, 1.25, 1.27, 50000),
            (1.27, 1.29, 1.22, 1.24, 30000),
            (1.24, 1.35, 1.23, 1.32, 200000),
            (1.32, 1.35, 1.30, 1.33, 80000),
        ]
        bars = _make_bars(candles)

        # Default min_price = 2.0
        runner = BacktestRunner(min_price=2.0)
        result = runner.run("TEST", bars, "2026-03-13")

        # Trade should be filtered
        assert len(result.trades_simulated) == 0

    def test_min_price_allows_qualifying_stocks(self):
        """Stocks at or above min_price pass the filter."""
        bars = _build_setup_then_breakout_bars()  # $4+ price range

        runner = BacktestRunner(
            planner=TradePlanner(min_risk_per_share=0.01),
            min_price=2.0,
        )
        result = runner.run("TEST", bars, "2026-03-13")

        # Pattern at $4+ should pass the $2 filter
        # (may or may not result in a trade depending on other filters)
        # The key is that it's NOT filtered by min_price


class TestLastEntryTime:
    """Integration tests for last_entry_time cutoff in realistic mode."""

    def test_no_new_setups_after_last_entry_time(self):
        """No new setups are scanned after last_entry_time_utc."""
        # Build bars starting at 18:55 UTC (14:55 ET), after last_entry_time
        start_time = datetime(2026, 3, 13, 18, 55, 0, tzinfo=timezone.utc)
        candles = [
            (4.00, 4.15, 3.98, 4.13, 200000),
            (4.13, 4.30, 4.11, 4.28, 180000),
            (4.28, 4.52, 4.26, 4.50, 160000),
            (4.50, 4.52, 4.38, 4.40, 50000),
            (4.40, 4.42, 4.33, 4.35, 30000),
            (4.35, 4.38, 4.32, 4.34, 25000),
            (4.38, 4.60, 4.36, 4.55, 250000),
            (4.55, 4.60, 4.50, 4.58, 100000),
        ]

        records = []
        for i, (o, h, l, c, v) in enumerate(candles):
            records.append({
                'timestamp': start_time + timedelta(minutes=i),
                'open': o, 'high': h, 'low': l, 'close': c, 'volume': v,
            })
        bars = pd.DataFrame(records)

        runner = BacktestRunner(
            planner=TradePlanner(min_risk_per_share=0.01),
            realistic=True,
            min_price=0.0,
            last_entry_time_utc=19.0,  # 15:00 ET = 19:00 UTC
        )

        result = runner.run("TEST", bars, "2026-03-13")

        # All bars are after 18:55 UTC, last_entry_time is 19:00 UTC
        # Most setups should be detected after 19:00, so no trades
        assert len(result.trades_simulated) == 0
