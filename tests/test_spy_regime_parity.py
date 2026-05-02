"""Integration test: None-SPY -> -0.5 penalty parity between BT and live.

Pinned regression test for the EAF 2026-05-01 false-positive: missing SPY
data must produce IDENTICAL conviction output across BT and live conviction
scorers, both treating None as the worst-case -0.5 penalty (NOT a 1.0 sentinel
that lands in the rule's neutral 0.8-1.2 band).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from trading.pattern_detector import BullFlagSetup


def _make_setup(pole_gain=4.1, retr=7.0):
    """Build a BullFlagSetup matching EAF 2026-05-01 conviction breakdown.

    Calibrated so the rule contributions match what dev's debug log captured:
        pole=+0.0 flag=+0.0 vol=+0.3 retr=+0.2 vwap=+0.0 gap=+0.0
    Sum of non-SPY contributions = +0.5. With SPY missing (None → -0.5)
    score = 1.0 (matches dev). With SPY=1.0 sentinel score = 1.5 (matches
    prod's bug pre-fix).
    """
    pole_low, pole_high = 6.85, 7.13                    # +4.1% (out of sweet spot 4.5-9)
    return BullFlagSetup(
        symbol="EAF",
        pole_start_idx=53, pole_end_idx=56,
        flag_start_idx=57, flag_end_idx=58,
        pole_low=pole_low, pole_high=pole_high,
        pole_height=pole_high - pole_low,
        pole_gain_pct=pole_gain,
        # Flag range = 0.10, pole_height = 0.28, tightness = 35.7%
        # → in (30, 50) → flag_tightness contrib = 0.0 (matches EAF)
        flag_low=6.95, flag_high=7.05,
        retracement_pct=retr,
        pullback_candle_count=2,
        # Vol ratio 1000/500 = 2.0 > 1.7 → +0.3 (matches EAF)
        avg_pole_volume=1000.0, avg_flag_volume=500.0,
        breakout_level=7.09,
    )


def _bt_conviction(spy_3d, *, return_breakdown=False):
    """Call BT's conviction scorer with neutral non-SPY inputs."""
    from backtest import BacktestRunner
    runner = BacktestRunner.__new__(BacktestRunner)  # bypass __init__
    return runner._compute_conviction_score_setup(
        _make_setup(),
        spy_3d,
        vwap_dist_pct=0.0,
        gap_fading=False,
        gap_pct=0.0,
        intraday_range_pct=0.0,
        v_reversal_enabled=False,
        return_breakdown=return_breakdown,
    )


def _live_conviction(spy_3d, *, return_breakdown=False):
    """Call live's conviction scorer with neutral non-SPY inputs."""
    from trading.trading_engine import TradingEngine
    engine = TradingEngine.__new__(TradingEngine)
    return engine._compute_conviction_score_setup(
        _make_setup(),
        spy_3d,
        vwap_dist_pct=0.0,
        gap_fading=False,
        gap_pct=0.0,
        intraday_range_pct=0.0,
        v_reversal_enabled=False,
        return_breakdown=return_breakdown,
    )


class TestSpyNoneParity:
    def test_none_produces_minus_05_in_bt(self):
        _, bd = _bt_conviction(None, return_breakdown=True)
        assert bd['spy_regime'] == -0.5

    def test_none_produces_minus_05_in_live(self):
        _, bd = _live_conviction(None, return_breakdown=True)
        assert bd['spy_regime'] == -0.5

    def test_bt_and_live_match_on_none(self):
        bt_score, bt_bd = _bt_conviction(None, return_breakdown=True)
        live_score, live_bd = _live_conviction(None, return_breakdown=True)
        assert bt_score == live_score
        assert bt_bd == live_bd


class TestSpyValueParity:
    """Parity across BT and live for several spy_3d_range values."""

    @pytest.mark.parametrize("spy_3d", [0.5, 0.79, 0.8, 1.0, 1.2, 1.21, 2.0])
    def test_parity_at_value(self, spy_3d):
        bt_score, bt_bd = _bt_conviction(spy_3d, return_breakdown=True)
        live_score, live_bd = _live_conviction(spy_3d, return_breakdown=True)
        assert bt_score == live_score, f"score mismatch at SPY 3d={spy_3d}"
        assert bt_bd['spy_regime'] == live_bd['spy_regime'], (
            f"spy_regime contrib mismatch at SPY 3d={spy_3d}: "
            f"BT={bt_bd['spy_regime']} vs live={live_bd['spy_regime']}"
        )


class TestEafFalsePositiveRegression:
    """The EAF 2026-05-01 case — assert the bug stays fixed.

    Before fix:  prod live computed SPY3d=1.0 (sentinel), conviction=1.50,
                 trade fired (rejected by Alpaca for buying-power, not by
                 our filter — would have leaked through otherwise).
    After fix:   None propagates, conviction includes -0.5 SPY penalty,
                 lands at 1.00 — below threshold 1.40, no trade.
    """

    def test_none_lands_below_default_threshold(self):
        score = _live_conviction(None)
        # EAF setup w/ vol_ratio=+0.3, retr<30=+0.2, all else 0
        # spy_regime=-0.5 → raw_score = 1.0 + 0.3 + 0.2 - 0.5 = 1.0
        assert score == pytest.approx(1.0)
        assert score < 1.4, "missing SPY must fail the 1.4 threshold gate"

    def test_old_sentinel_value_would_have_passed_threshold(self):
        # Demonstrates WHY the old return-1.0 was harmful:
        # spy_3d=1.0 → sr_contrib=0.0 → score = 1.0 + 0.3 + 0.2 = 1.5 > 1.4 = trade
        score = _live_conviction(1.0)
        assert score == pytest.approx(1.5)
        assert score >= 1.4

# ---------------------------------------------------------------------------
# CSV-write regression — added 2026-05-02 during code review.
#
# When `_get_spy_3d_range` returns None (missing/stale SPY), the BT writes
# the trade to `bull_flag_cache_e50_x30.csv` with a {:.3f} format spec on
# `trade.spy_3d_range`. If None propagated through the pending_order →
# trade chain, the format would raise:
#
#     TypeError: unsupported format string passed to NoneType.__format__
#
# The fix in backtest.py:2474/2500 collapses None → 0.0 at the trade-storage
# point so the dataclass field stays float. This test pins the contract.
# ---------------------------------------------------------------------------


class TestNoneSpyDoesNotCrashCsvWrite:
    """Pinned regression for the trade.spy_3d_range = None → CSV crash."""

    def _make_simulated_trade(self):
        """Minimal SimulatedTrade for cache-row conversion."""
        from datetime import datetime, timezone
        from backtest import SimulatedTrade
        return SimulatedTrade(
            symbol="EAF",
            entry_time=datetime(2026, 5, 1, 14, 36, tzinfo=timezone.utc),
            entry_price=7.17,
            stop_loss=6.99,
            take_profit=7.42,
            shares=1620,
            exit_time=datetime(2026, 5, 1, 15, 19, tzinfo=timezone.utc),
            exit_price=7.73,
            exit_reason="exhaust+trail_stop",
            pnl=955.55,
            pnl_pct=8.22,
        )

    def test_storage_collapses_none_to_zero(self):
        """Simulating the assignment in backtest.py:2474/2500."""

        class _Pending:
            pass

        pending = _Pending()
        pending._spy_3d_range = None  # missing data path

        trade = self._make_simulated_trade()
        # Mirror exactly the line in backtest.py
        trade.spy_3d_range = getattr(pending, '_spy_3d_range', 0.0) or 0.0

        assert trade.spy_3d_range == 0.0
        # Verify the CSV format spec works without crashing
        formatted = f"{trade.spy_3d_range:.3f}"
        assert formatted == "0.000"

    def test_real_value_passes_through_unchanged(self):
        """`or 0.0` must NOT clobber a legitimate spy_3d_range value."""

        class _Pending:
            pass

        pending = _Pending()
        pending._spy_3d_range = 0.789

        trade = self._make_simulated_trade()
        trade.spy_3d_range = getattr(pending, '_spy_3d_range', 0.0) or 0.0
        assert trade.spy_3d_range == 0.789

    def test_csv_row_serializes_with_none_input_at_format(self):
        """End-to-end: pending_order with None _spy_3d_range → cache row OK."""
        from batch_backtest import _trade_to_cache_row

        class _Pending:
            pass

        pending = _Pending()
        pending._spy_3d_range = None

        trade = self._make_simulated_trade()
        trade.spy_3d_range = getattr(pending, '_spy_3d_range', 0.0) or 0.0
        trade.partial_exit_taken = False
        trade.partial_exit_price = None
        trade.partial_shares = 0
        trade.partial_pnl = 0.0
        # Defaults the dataclass would have set to 0.0 / None already.

        row = _trade_to_cache_row(trade)
        assert row is not None, "_trade_to_cache_row must not crash on None spy"
        assert row['spy_3d_range'] == "0.000"
