"""
Tests for risk model research — hypothesis registry, metrics, and output.

Tests cover:
- Hypothesis registry: all 10 hypotheses create valid planners
- Metrics computation: correct aggregate metrics from trades
- Price bucket analysis: correct breakdown by entry price
- Comparison table: formatting and output
- Integration: H0 baseline produces identical behavior to default TradePlanner
"""

import math
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from backtest import SimulatedTrade
from risk_research import (
    HYPOTHESES,
    build_planner,
    compute_metrics,
    compute_price_bucket_metrics,
    print_comparison_table,
    print_price_bucket_analysis,
    write_comparison_csv,
    write_trades_csv,
)
from trading.pattern_detector import BullFlagPattern
from trading.trade_planner import TradePlanner, TradePlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pattern(
    symbol="TEST",
    breakout_level=4.40,
    flag_low=4.30,
    pole_low=4.00,
    pole_high=4.50,
):
    """Create a BullFlagPattern with sensible defaults."""
    pole_height = pole_high - pole_low
    pole_gain_pct = (pole_height / pole_low) * 100 if pole_low > 0 else 0
    retracement_pct = ((pole_high - flag_low) / pole_height) * 100 if pole_height > 0 else 0

    return BullFlagPattern(
        symbol=symbol,
        pole_start_idx=0,
        pole_end_idx=2,
        flag_start_idx=3,
        flag_end_idx=4,
        pole_low=pole_low,
        pole_high=pole_high,
        pole_height=pole_height,
        pole_gain_pct=pole_gain_pct,
        flag_low=flag_low,
        flag_high=breakout_level - 0.05,
        retracement_pct=retracement_pct,
        pullback_candle_count=2,
        avg_pole_volume=180000,
        avg_flag_volume=40000,
        breakout_level=breakout_level,
        detected_at=datetime.now(timezone.utc),
    )


def _make_trade(symbol="TEST", entry_price=10.0, stop_loss=9.80, take_profit=10.40,
                shares=500, exit_price=10.40, exit_reason="target", pnl=200.0):
    """Create a SimulatedTrade for testing metrics."""
    return SimulatedTrade(
        symbol=symbol,
        entry_time=datetime(2026, 3, 13, 14, 0, tzinfo=timezone.utc),
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        shares=shares,
        exit_time=datetime(2026, 3, 13, 15, 0, tzinfo=timezone.utc),
        exit_price=exit_price,
        exit_reason=exit_reason,
        pnl=pnl,
        pnl_pct=(exit_price - entry_price) / entry_price * 100,
    )


# ===========================================================================
# HYPOTHESIS REGISTRY TESTS
# ===========================================================================

class TestHypothesisRegistry:
    """Tests for the hypothesis registry and planner builder."""

    def test_all_hypotheses_exist(self):
        """Registry contains H0-H9 plus H9a, H9b, and H10."""
        expected = {f"H{i}" for i in range(11)} | {"H9a", "H9b"}
        assert set(HYPOTHESES.keys()) == expected

    def test_all_hypotheses_create_valid_planners(self):
        """Every hypothesis in the registry produces a valid TradePlanner."""
        for h_id in HYPOTHESES:
            planner = build_planner(h_id)
            assert isinstance(planner, TradePlanner), f"{h_id} didn't produce TradePlanner"

    def test_h0_is_fixed_investment(self):
        """H0 uses fixed_investment sizing (baseline)."""
        planner = build_planner("H0")
        assert planner.sizing_mode == "fixed_investment"
        assert planner.position_size_dollars == 50000

    def test_h1_is_fixed_risk_500(self):
        """H1 uses fixed_risk with $500 budget."""
        planner = build_planner("H1")
        assert planner.sizing_mode == "fixed_risk"
        assert planner.risk_per_trade == 500

    def test_h2_conservative_250(self):
        """H2 uses $250 risk budget."""
        planner = build_planner("H2")
        assert planner.risk_per_trade == 250

    def test_h3_aggressive_1000(self):
        """H3 uses $1000 risk budget."""
        planner = build_planner("H3")
        assert planner.risk_per_trade == 1000

    def test_h4_pct_stops(self):
        """H4 uses percentage-based stops (1%-5%)."""
        planner = build_planner("H4")
        assert planner.min_risk_pct == 0.01
        assert planner.max_risk_pct == 0.05

    def test_h5_tight_pct_stops(self):
        """H5 uses tight pct stops (0.5%-3%)."""
        planner = build_planner("H5")
        assert planner.min_risk_pct == 0.005
        assert planner.max_risk_pct == 0.03

    def test_h7_lower_rr(self):
        """H7 uses 1.5:1 R:R."""
        planner = build_planner("H7")
        assert planner.min_risk_reward == 1.5

    def test_h8_higher_rr(self):
        """H8 uses 3.0:1 R:R."""
        planner = build_planner("H8")
        assert planner.min_risk_reward == 3.0

    def test_unknown_hypothesis_raises(self):
        """Unknown hypothesis ID raises KeyError."""
        with pytest.raises(KeyError, match="Unknown hypothesis"):
            build_planner("H99")

    def test_h0_through_h3_use_flat_stops(self):
        """H0-H3 all use flat dollar stops (no pct)."""
        for h_id in ["H0", "H1", "H2", "H3"]:
            planner = build_planner(h_id)
            assert planner.min_risk_pct is None, f"{h_id} should have no min_risk_pct"
            assert planner.max_risk_pct is None, f"{h_id} should have no max_risk_pct"

    def test_h4_through_h9_use_pct_stops(self):
        """H4-H9 all use percentage-based stops."""
        for h_id in ["H4", "H5", "H6", "H7", "H8", "H9"]:
            planner = build_planner(h_id)
            assert planner.min_risk_pct is not None, f"{h_id} should have min_risk_pct"
            assert planner.max_risk_pct is not None, f"{h_id} should have max_risk_pct"


# ===========================================================================
# METRICS COMPUTATION TESTS
# ===========================================================================

class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_empty_trades_returns_zeros(self):
        """No trades produces all-zero metrics."""
        metrics = compute_metrics([])
        assert metrics["trade_count"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["total_pnl"] == 0.0

    def test_all_wins(self):
        """All winning trades = 100% WR."""
        trades = [
            _make_trade(pnl=100, exit_price=10.20),
            _make_trade(pnl=200, exit_price=10.40),
        ]
        metrics = compute_metrics(trades)
        assert metrics["trade_count"] == 2
        assert metrics["win_rate"] == 100.0
        assert metrics["total_pnl"] == 300.0
        assert metrics["avg_win"] == 150.0

    def test_all_losses(self):
        """All losing trades = 0% WR."""
        trades = [
            _make_trade(pnl=-100, exit_price=9.80, exit_reason="stop"),
            _make_trade(pnl=-50, exit_price=9.90, exit_reason="stop"),
        ]
        metrics = compute_metrics(trades)
        assert metrics["win_rate"] == 0.0
        assert metrics["total_pnl"] == -150.0

    def test_mixed_trades_correct_rr(self):
        """Mixed wins/losses compute correct actual R:R."""
        trades = [
            _make_trade(pnl=200, exit_price=10.40),
            _make_trade(pnl=-100, exit_price=9.80, exit_reason="stop"),
        ]
        metrics = compute_metrics(trades)
        assert metrics["trade_count"] == 2
        assert metrics["win_rate"] == 50.0
        assert metrics["actual_rr"] == pytest.approx(2.0, abs=0.01)

    def test_profit_factor(self):
        """Profit factor = gross wins / gross losses."""
        trades = [
            _make_trade(pnl=300, exit_price=10.60),
            _make_trade(pnl=-100, exit_price=9.80, exit_reason="stop"),
        ]
        metrics = compute_metrics(trades)
        assert metrics["profit_factor"] == pytest.approx(3.0, abs=0.01)

    def test_max_drawdown(self):
        """Max drawdown tracks peak-to-trough in cumulative P&L."""
        trades = [
            _make_trade(pnl=500, exit_price=11.00),    # cum = 500, peak = 500
            _make_trade(pnl=-200, exit_price=9.60, exit_reason="stop"),  # cum = 300
            _make_trade(pnl=-300, exit_price=9.40, exit_reason="stop"),  # cum = 0, dd = 500
            _make_trade(pnl=100, exit_price=10.20),     # cum = 100
        ]
        metrics = compute_metrics(trades)
        assert metrics["max_drawdown"] == pytest.approx(-500, abs=0.01)


# ===========================================================================
# PRICE BUCKET TESTS
# ===========================================================================

class TestPriceBucketAnalysis:
    """Tests for price bucket breakdown."""

    def test_buckets_by_price(self):
        """Trades are correctly bucketed by entry price."""
        trades = [
            _make_trade(entry_price=3.00, pnl=50),
            _make_trade(entry_price=4.50, pnl=-20, exit_reason="stop"),
            _make_trade(entry_price=7.00, pnl=100),
            _make_trade(entry_price=15.00, pnl=200),
        ]
        buckets = compute_price_bucket_metrics(trades)

        bucket_names = [b["bucket"] for b in buckets]
        assert "$2-5" in bucket_names
        assert "$5-10" in bucket_names
        assert "$10-20" in bucket_names

        cheap = next(b for b in buckets if b["bucket"] == "$2-5")
        assert cheap["trade_count"] == 2

    def test_empty_trades_no_buckets(self):
        """No trades produces empty bucket list."""
        buckets = compute_price_bucket_metrics([])
        assert buckets == []

    def test_print_bucket_no_crash(self, capsys):
        """print_price_bucket_analysis doesn't crash on valid data."""
        trades = [_make_trade(entry_price=5.00, pnl=100)]
        print_price_bucket_analysis("H0", trades)
        captured = capsys.readouterr()
        assert "H0 by price" in captured.out


# ===========================================================================
# OUTPUT TESTS
# ===========================================================================

class TestOutput:
    """Tests for CSV and table output."""

    def test_comparison_table_prints(self, capsys):
        """print_comparison_table outputs formatted table without crashing."""
        results = {
            "H0": {
                "trade_count": 32, "win_rate": 62.5, "total_pnl": 21212,
                "avg_win": 1668, "avg_loss": -1012, "actual_rr": 1.65,
                "max_drawdown": -3024, "profit_factor": 2.87,
            },
        }
        print_comparison_table(results)
        captured = capsys.readouterr()
        assert "H0" in captured.out
        assert "HYPOTHESIS COMPARISON" in captured.out

    def test_write_comparison_csv(self):
        """write_comparison_csv creates valid CSV file."""
        results = {
            "H0": {
                "trade_count": 32, "win_rate": 62.5, "total_pnl": 21212,
                "avg_win": 1668, "avg_loss": -1012, "actual_rr": 1.65,
                "max_drawdown": -3024, "profit_factor": 2.87,
                "description": "Baseline",
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w') as f:
            path = f.name
        try:
            write_comparison_csv(results, path)
            with open(path, 'r') as f:
                content = f.read()
            assert "hypothesis" in content
            assert "H0" in content
            assert "21212" in content
        finally:
            os.unlink(path)

    def test_write_trades_csv(self):
        """write_trades_csv creates per-hypothesis trade CSV."""
        trades = [_make_trade(pnl=200)]
        params = {"sizing_mode": "fixed_risk", "risk_per_trade": 500}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_trades_csv("H1", trades, params, tmpdir)
            assert os.path.exists(path)
            with open(path, 'r') as f:
                content = f.read()
            assert "H1" in content
            assert "fixed_risk" in content


# ===========================================================================
# INTEGRATION: H0 MATCHES BASELINE
# ===========================================================================

class TestH0Baseline:
    """Integration test: H0 planner produces identical plans to original defaults."""

    def test_h0_plan_matches_default_planner(self):
        """H0 hypothesis produces same plan as original TradePlanner defaults."""
        h0_planner = build_planner("H0")

        # Original-style planner with same dollar amount
        original_planner = TradePlanner(
            position_size_dollars=50000,
            max_shares=10000,
            max_risk_per_share=0.20,
            min_risk_per_share=0.05,
            min_risk_reward=2.0,
        )

        pattern = _make_pattern(breakout_level=8.00, flag_low=7.90)

        h0_plan = h0_planner.create_plan(pattern)
        orig_plan = original_planner.create_plan(pattern)

        assert h0_plan is not None
        assert orig_plan is not None

        assert h0_plan.entry_price == orig_plan.entry_price
        assert h0_plan.stop_loss_price == pytest.approx(orig_plan.stop_loss_price, abs=0.01)
        assert h0_plan.take_profit_price == pytest.approx(orig_plan.take_profit_price, abs=0.01)
        assert h0_plan.risk_per_share == pytest.approx(orig_plan.risk_per_share, abs=0.01)
        assert h0_plan.shares == orig_plan.shares
        assert h0_plan.risk_reward_ratio == pytest.approx(orig_plan.risk_reward_ratio, abs=0.01)

    def test_h0_rejects_same_patterns_as_default(self):
        """H0 rejects the same volatile patterns as original."""
        h0_planner = build_planner("H0")

        # Too volatile — should be rejected
        volatile = _make_pattern(
            breakout_level=5.00,
            flag_low=4.40,  # risk = 0.61 > 0.50
        )
        assert h0_planner.create_plan(volatile) is None

        # Noise stop — should be rejected
        noise = _make_pattern(
            breakout_level=5.00,
            flag_low=5.00,  # risk = 0.01 < 0.05
        )
        assert h0_planner.create_plan(noise) is None
