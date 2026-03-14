"""
Tests for the monthly backtest runner.

Covers:
- split_into_months: date range splitting logic
- build_rich_row: rich CSV row construction
- write_rich_csv_report: CSV writing with extended columns
- MonthlyBacktestRunner.aggregate_csvs: CSV aggregation
- MonthlyBacktestRunner.run_month: single month processing
- Integration: monthly splitting + aggregation end-to-end
"""

import csv
import os
from datetime import date, datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from backtest import BacktestResult, SimulatedTrade
from batch.monthly_runner import (
    RICH_CSV_HEADERS,
    MonthlyBacktestRunner,
    MonthResult,
    build_rich_row,
    split_into_months,
    write_rich_csv_report,
    _find_daily_bar,
)
from trading.pattern_detector import BullFlagPattern
from trading.trade_planner import TradePlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pattern(**kwargs) -> BullFlagPattern:
    """Create a BullFlagPattern with sensible defaults."""
    defaults = dict(
        symbol="TEST",
        pole_start_idx=0,
        pole_end_idx=3,
        flag_start_idx=4,
        flag_end_idx=5,
        pole_low=4.70,
        pole_high=5.50,
        pole_height=0.80,
        pole_gain_pct=10.0,
        flag_low=5.10,
        flag_high=5.40,
        retracement_pct=30.0,
        pullback_candle_count=2,
        avg_pole_volume=10000,
        avg_flag_volume=5000,
        breakout_level=5.50,
    )
    defaults.update(kwargs)
    return BullFlagPattern(**defaults)


def _make_plan(pattern=None, **kwargs) -> TradePlan:
    """Create a TradePlan with sensible defaults."""
    if pattern is None:
        pattern = _make_pattern()
    defaults = dict(
        symbol="TEST",
        entry_price=5.50,
        stop_loss_price=5.30,
        take_profit_price=5.90,
        risk_per_share=0.20,
        reward_per_share=0.40,
        risk_reward_ratio=2.0,
        shares=90,
        total_risk=18.0,
        pattern=pattern,
    )
    defaults.update(kwargs)
    return TradePlan(**defaults)


def _make_trade(plan=None, **kwargs) -> SimulatedTrade:
    """Create a SimulatedTrade with sensible defaults."""
    if plan is None:
        plan = _make_plan()
    defaults = dict(
        symbol="TEST",
        entry_time=datetime(2026, 3, 5, 14, 30, 0, tzinfo=timezone.utc),
        entry_price=5.50,
        stop_loss=5.30,
        take_profit=5.90,
        shares=90,
        exit_time=datetime(2026, 3, 5, 15, 45, 0, tzinfo=timezone.utc),
        exit_price=5.90,
        exit_reason="target",
        pnl=36.0,
        pnl_pct=7.27,
        bars_held=15,
        plan=plan,
    )
    defaults.update(kwargs)
    return SimulatedTrade(**defaults)


def _make_result(trades=None, **kwargs) -> BacktestResult:
    """Create a BacktestResult with sensible defaults."""
    defaults = dict(
        symbol="TEST",
        trade_date="2026-03-05",
        total_bars=390,
        patterns_detected=2,
    )
    defaults.update(kwargs)
    result = BacktestResult(**defaults)
    if trades:
        result.trades_simulated = trades
    return result


# ===========================================================================
# Tests: split_into_months
# ===========================================================================


class TestSplitIntoMonths:
    """Tests for monthly date range splitting."""

    def test_single_month(self):
        """Date range within a single month."""
        chunks = split_into_months(date(2025, 3, 1), date(2025, 3, 15))
        assert len(chunks) == 1
        assert chunks[0] == (date(2025, 3, 1), date(2025, 3, 15))

    def test_full_single_month(self):
        """Full month range."""
        chunks = split_into_months(date(2025, 3, 1), date(2025, 3, 31))
        assert len(chunks) == 1
        assert chunks[0] == (date(2025, 3, 1), date(2025, 3, 31))

    def test_two_months(self):
        """Range spanning two months."""
        chunks = split_into_months(date(2025, 3, 15), date(2025, 4, 15))
        assert len(chunks) == 2
        assert chunks[0] == (date(2025, 3, 15), date(2025, 3, 31))
        assert chunks[1] == (date(2025, 4, 1), date(2025, 4, 15))

    def test_twelve_months(self):
        """Full year range."""
        chunks = split_into_months(date(2025, 1, 1), date(2025, 12, 31))
        assert len(chunks) == 12
        assert chunks[0] == (date(2025, 1, 1), date(2025, 1, 31))
        assert chunks[11] == (date(2025, 12, 1), date(2025, 12, 31))

    def test_cross_year(self):
        """Range crossing year boundary."""
        chunks = split_into_months(date(2025, 11, 1), date(2026, 2, 28))
        assert len(chunks) == 4
        assert chunks[0] == (date(2025, 11, 1), date(2025, 11, 30))
        assert chunks[1] == (date(2025, 12, 1), date(2025, 12, 31))
        assert chunks[2] == (date(2026, 1, 1), date(2026, 1, 31))
        assert chunks[3] == (date(2026, 2, 1), date(2026, 2, 28))

    def test_same_day(self):
        """Start == end should return single chunk."""
        chunks = split_into_months(date(2025, 3, 15), date(2025, 3, 15))
        assert len(chunks) == 1
        assert chunks[0] == (date(2025, 3, 15), date(2025, 3, 15))

    def test_start_after_end_returns_empty(self):
        """Start > end should return empty list."""
        chunks = split_into_months(date(2025, 4, 1), date(2025, 3, 1))
        assert chunks == []

    def test_february_leap_year(self):
        """February in a leap year (2028 is leap year)."""
        chunks = split_into_months(date(2028, 2, 1), date(2028, 2, 29))
        assert len(chunks) == 1
        assert chunks[0] == (date(2028, 2, 1), date(2028, 2, 29))

    def test_mid_month_to_mid_month(self):
        """Start and end mid-month across 3 months."""
        chunks = split_into_months(date(2025, 1, 15), date(2025, 3, 14))
        assert len(chunks) == 3
        assert chunks[0] == (date(2025, 1, 15), date(2025, 1, 31))
        assert chunks[1] == (date(2025, 2, 1), date(2025, 2, 28))
        assert chunks[2] == (date(2025, 3, 1), date(2025, 3, 14))

    def test_fifteen_months(self):
        """Range from Jan 2025 to Mar 2026 = 15 months."""
        chunks = split_into_months(date(2025, 1, 1), date(2026, 3, 14))
        assert len(chunks) == 15
        assert chunks[0][0] == date(2025, 1, 1)
        assert chunks[-1][1] == date(2026, 3, 14)


# ===========================================================================
# Tests: build_rich_row
# ===========================================================================


class TestBuildRichRow:
    """Tests for rich CSV row construction."""

    def test_full_row_with_all_data(self):
        """Row with complete trade, pattern, plan, daily, and universe data."""
        trade = _make_trade()
        result = _make_result(trades=[trade])
        daily_bars = {
            "TEST": [
                {"date": date(2026, 3, 5), "open": 5.0, "high": 6.0,
                 "low": 4.5, "close": 5.8, "volume": 500000},
            ]
        }
        universe = {
            "TEST": {"sector": "Technology", "float_shares": 1000000,
                     "avg_volume_daily": 200000},
        }

        row = build_rich_row(trade, result, daily_bars, universe)

        assert len(row) == len(RICH_CSV_HEADERS)
        assert row[0] == "TEST"  # symbol
        assert row[1] == "2026-03-05"  # date
        assert row[3] == "5.50"  # entry_price
        assert row[10] == "36.00"  # pnl
        assert row[12] == 15  # bars_held
        assert row[17] == "10.00"  # pole_gain_pct
        assert row[25] == "5.00"  # day_open
        assert row[31] == "Technology"  # sector
        assert row[34] == 60  # entry_minutes_from_open (14:30 - 13:30 = 60 min)
        assert row[35] == 2  # patterns_detected_that_day

    def test_row_with_missing_daily_bars(self):
        """Row handles missing daily bar data gracefully."""
        trade = _make_trade()
        result = _make_result(trades=[trade])

        row = build_rich_row(trade, result, {}, {})

        assert len(row) == len(RICH_CSV_HEADERS)
        assert row[25] == ""  # day_open
        assert row[30] == ""  # intraday_move_pct
        assert row[31] == ""  # sector

    def test_row_with_no_plan(self):
        """Row handles trade without plan."""
        trade = SimulatedTrade(
            symbol="TEST",
            entry_time=datetime(2026, 3, 5, 14, 30, 0, tzinfo=timezone.utc),
            entry_price=5.50,
            stop_loss=5.30,
            take_profit=5.90,
            shares=90,
            plan=None,
        )
        result = _make_result(trades=[trade])

        row = build_rich_row(trade, result, {}, {})

        assert row[13] == ""  # risk_per_share
        assert row[17] == ""  # pole_gain_pct


# ===========================================================================
# Tests: _find_daily_bar
# ===========================================================================


class TestFindDailyBar:
    """Tests for daily bar lookup."""

    def test_finds_matching_date(self):
        """Should find bar matching the date string."""
        bars = {
            "TEST": [
                {"date": date(2026, 3, 5), "open": 5.0, "high": 6.0,
                 "low": 4.5, "close": 5.8},
            ]
        }
        bar = _find_daily_bar("TEST", "2026-03-05", bars)
        assert bar is not None
        assert bar["open"] == 5.0

    def test_returns_none_for_missing_symbol(self):
        """Should return None for unknown symbol."""
        bar = _find_daily_bar("MISSING", "2026-03-05", {})
        assert bar is None

    def test_returns_none_for_wrong_date(self):
        """Should return None when date doesn't match."""
        bars = {
            "TEST": [
                {"date": date(2026, 3, 5), "open": 5.0, "high": 6.0,
                 "low": 4.5, "close": 5.8},
            ]
        }
        bar = _find_daily_bar("TEST", "2026-03-06", bars)
        assert bar is None

    def test_handles_string_dates(self):
        """Should match when daily bar date is a string."""
        bars = {
            "TEST": [
                {"date": "2026-03-05", "open": 5.0, "high": 6.0,
                 "low": 4.5, "close": 5.8},
            ]
        }
        bar = _find_daily_bar("TEST", "2026-03-05", bars)
        assert bar is not None


# ===========================================================================
# Tests: write_rich_csv_report
# ===========================================================================


class TestWriteRichCsvReport:
    """Tests for rich CSV writing."""

    def test_writes_correct_headers(self, tmp_path):
        """Rich CSV should have extended headers."""
        output = str(tmp_path / "rich.csv")
        write_rich_csv_report([], output, {}, {})

        with open(output, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert headers == RICH_CSV_HEADERS

    def test_writes_trade_data(self, tmp_path):
        """Trade data should appear in rich CSV."""
        trade = _make_trade()
        result = _make_result(trades=[trade])
        output = str(tmp_path / "rich.csv")

        count = write_rich_csv_report([result], output, {}, {})
        assert count == 1

        with open(output, 'r') as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["symbol"] == "TEST"
        assert row["entry_price"] == "5.50"
        assert row["pnl"] == "36.00"
        assert row["bars_held"] == "15"
        assert row["risk_per_share"] == "0.2000"

    def test_empty_results(self, tmp_path):
        """Empty results should produce header-only CSV."""
        output = str(tmp_path / "rich.csv")
        count = write_rich_csv_report([], output, {}, {})
        assert count == 0


# ===========================================================================
# Tests: MonthlyBacktestRunner.aggregate_csvs
# ===========================================================================


class TestAggregateCsvs:
    """Tests for CSV aggregation."""

    def test_aggregates_multiple_csvs(self, tmp_path):
        """Should combine multiple CSVs into one master file."""
        # Create two monthly CSVs
        headers = ["symbol", "date", "pnl"]
        csv1 = str(tmp_path / "month1.csv")
        csv2 = str(tmp_path / "month2.csv")

        for path, rows in [(csv1, [["A", "2025-01-05", "100"]]),
                           (csv2, [["B", "2025-02-05", "200"]])]:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)

        master = str(tmp_path / "master.csv")
        runner = MonthlyBacktestRunner()
        total = runner.aggregate_csvs([csv1, csv2], master)

        assert total == 2
        with open(master, 'r') as f:
            reader = csv.reader(f)
            all_rows = list(reader)
        assert len(all_rows) == 3  # header + 2 data rows
        assert all_rows[0] == headers
        assert all_rows[1][0] == "A"
        assert all_rows[2][0] == "B"

    def test_skips_missing_files(self, tmp_path):
        """Should skip non-existent files."""
        master = str(tmp_path / "master.csv")
        runner = MonthlyBacktestRunner()
        total = runner.aggregate_csvs(
            ["/nonexistent/path.csv", ""], master
        )
        assert total == 0

    def test_empty_csv_list(self, tmp_path):
        """Should handle empty input list."""
        master = str(tmp_path / "master.csv")
        runner = MonthlyBacktestRunner()
        total = runner.aggregate_csvs([], master)
        assert total == 0

    def test_single_csv(self, tmp_path):
        """Single CSV should produce identical master."""
        headers = ["col1", "col2"]
        csv1 = str(tmp_path / "only.csv")
        with open(csv1, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(["val1", "val2"])

        master = str(tmp_path / "master.csv")
        runner = MonthlyBacktestRunner()
        total = runner.aggregate_csvs([csv1], master)
        assert total == 1


# ===========================================================================
# Tests: MonthlyBacktestRunner.run_month (mocked)
# ===========================================================================


class TestRunMonth:
    """Tests for single month processing with mocked dependencies."""

    @patch("batch.monthly_runner.get_database")
    @patch("batch.monthly_runner.AlpacaClient")
    @patch("batch.monthly_runner.run_batch_backtest")
    @patch("batch.monthly_runner.find_big_movers")
    @patch("batch.monthly_runner.fetch_daily_bars_cached")
    @patch("batch.monthly_runner.load_dotenv")
    @patch.dict(os.environ, {"ALPACA_API_KEY": "test", "ALPACA_API_SECRET": "test"})
    def test_run_month_with_trades(
        self, mock_dotenv, mock_fetch, mock_find, mock_batch,
        mock_client_cls, mock_get_db, tmp_path
    ):
        """Should process a month and return MonthResult with trades."""
        # Setup mocks
        mock_db = MagicMock()
        mock_db.get_active_universe.return_value = [
            {"symbol": "TEST", "sector": "Tech", "float_shares": 100000,
             "avg_volume_daily": 50000},
        ]
        mock_get_db.return_value = mock_db

        mock_fetch.return_value = {
            "TEST": [{"date": date(2025, 1, 5), "open": 5.0, "high": 6.0,
                      "low": 5.0, "close": 5.8, "volume": 500000}]
        }
        mock_find.return_value = [("TEST", date(2025, 1, 5))]

        trade = _make_trade()
        result = _make_result(trades=[trade])
        mock_batch.return_value = [result]

        runner = MonthlyBacktestRunner(max_workers=1)
        output_dir = str(tmp_path)
        month_result = runner.run_month(
            date(2025, 1, 1), date(2025, 1, 31), output_dir, 1, 1
        )

        assert month_result.month_label == "2025-01"
        assert month_result.num_movers == 1
        assert month_result.num_trades == 1
        assert month_result.total_pnl == pytest.approx(36.0)
        assert month_result.csv_path.endswith("backtest_2025_01.csv")
        assert os.path.exists(month_result.csv_path)

    @patch("batch.monthly_runner.get_database")
    @patch("batch.monthly_runner.AlpacaClient")
    @patch("batch.monthly_runner.find_big_movers")
    @patch("batch.monthly_runner.fetch_daily_bars_cached")
    @patch("batch.monthly_runner.load_dotenv")
    @patch.dict(os.environ, {"ALPACA_API_KEY": "test", "ALPACA_API_SECRET": "test"})
    def test_run_month_no_movers(
        self, mock_dotenv, mock_fetch, mock_find,
        mock_client_cls, mock_get_db, tmp_path
    ):
        """Should handle months with no qualifying movers."""
        mock_db = MagicMock()
        mock_db.get_active_universe.return_value = [{"symbol": "TEST"}]
        mock_get_db.return_value = mock_db

        mock_fetch.return_value = {"TEST": []}
        mock_find.return_value = []

        runner = MonthlyBacktestRunner(max_workers=1)
        month_result = runner.run_month(
            date(2025, 6, 1), date(2025, 6, 30), str(tmp_path), 1, 1
        )

        assert month_result.num_movers == 0
        assert month_result.num_trades == 0
        assert month_result.csv_path == ""


# ===========================================================================
# Tests: End-to-end monthly runner (mocked APIs)
# ===========================================================================


class TestMonthlyRunnerEndToEnd:
    """Integration test for the full monthly pipeline with mocked APIs."""

    @patch("batch.monthly_runner.get_database")
    @patch("batch.monthly_runner.AlpacaClient")
    @patch("batch.monthly_runner.run_batch_backtest")
    @patch("batch.monthly_runner.find_big_movers")
    @patch("batch.monthly_runner.fetch_daily_bars_cached")
    @patch("batch.monthly_runner.load_dotenv")
    @patch.dict(os.environ, {"ALPACA_API_KEY": "test", "ALPACA_API_SECRET": "test"})
    def test_run_all_sequential(
        self, mock_dotenv, mock_fetch, mock_find, mock_batch,
        mock_client_cls, mock_get_db, tmp_path
    ):
        """Sequential run_all should produce master CSV from monthly chunks."""
        mock_db = MagicMock()
        mock_db.get_active_universe.return_value = [
            {"symbol": "TEST", "sector": "Tech", "float_shares": 100000,
             "avg_volume_daily": 50000},
        ]
        mock_get_db.return_value = mock_db

        mock_fetch.return_value = {
            "TEST": [{"date": date(2025, 1, 5), "open": 5.0, "high": 6.0,
                      "low": 5.0, "close": 5.8, "volume": 500000}]
        }
        mock_find.return_value = [("TEST", date(2025, 1, 5))]

        trade = _make_trade()
        result = _make_result(trades=[trade])
        mock_batch.return_value = [result]

        output_dir = str(tmp_path / "results")
        runner = MonthlyBacktestRunner(max_workers=1)
        master_csv = runner.run_all(
            date(2025, 1, 1), date(2025, 2, 28), output_dir=output_dir
        )

        # Master CSV should exist
        assert os.path.exists(master_csv)
        assert "backtest_full_2025_01_to_2025_02.csv" in master_csv

        # Should have data rows
        with open(master_csv, 'r') as f:
            reader = csv.reader(f)
            all_rows = list(reader)
        # Header + 2 months of data (each month has 1 trade)
        assert len(all_rows) >= 2  # at least header + 1 trade

    @patch("batch.monthly_runner.get_database")
    @patch("batch.monthly_runner.AlpacaClient")
    @patch("batch.monthly_runner.run_batch_backtest")
    @patch("batch.monthly_runner.find_big_movers")
    @patch("batch.monthly_runner.fetch_daily_bars_cached")
    @patch("batch.monthly_runner.load_dotenv")
    @patch.dict(os.environ, {"ALPACA_API_KEY": "test", "ALPACA_API_SECRET": "test"})
    def test_run_all_parallel(
        self, mock_dotenv, mock_fetch, mock_find, mock_batch,
        mock_client_cls, mock_get_db, tmp_path
    ):
        """Parallel run_all with 2 workers should produce same result."""
        mock_db = MagicMock()
        mock_db.get_active_universe.return_value = [
            {"symbol": "TEST", "sector": "Tech", "float_shares": 100000,
             "avg_volume_daily": 50000},
        ]
        mock_get_db.return_value = mock_db

        mock_fetch.return_value = {
            "TEST": [{"date": date(2025, 1, 5), "open": 5.0, "high": 6.0,
                      "low": 5.0, "close": 5.8, "volume": 500000}]
        }
        mock_find.return_value = [("TEST", date(2025, 1, 5))]

        trade = _make_trade()
        result = _make_result(trades=[trade])
        mock_batch.return_value = [result]

        output_dir = str(tmp_path / "results")
        runner = MonthlyBacktestRunner(max_workers=2)
        master_csv = runner.run_all(
            date(2025, 1, 1), date(2025, 3, 31), output_dir=output_dir
        )

        assert os.path.exists(master_csv)


# ===========================================================================
# Tests: Early exit after trade (Part A3)
# ===========================================================================


class TestEarlyExitAfterTrade:
    """Tests for the early_exit_after_trade flag."""

    def test_early_exit_default_is_true(self):
        """BacktestRunner defaults to early_exit_after_trade=True."""
        from backtest import BacktestRunner
        runner = BacktestRunner()
        assert runner.early_exit_after_trade is True

    def test_early_exit_can_be_disabled(self):
        """BacktestRunner can disable early exit."""
        from backtest import BacktestRunner
        runner = BacktestRunner(early_exit_after_trade=False)
        assert runner.early_exit_after_trade is False


# ===========================================================================
# Tests: end_idx parameter for detector (Part A1)
# ===========================================================================


class TestDetectorEndIdx:
    """Tests for the end_idx parameter added to BullFlagDetector.detect()."""

    def test_end_idx_equivalent_to_slicing(self):
        """detect(bars[:N]) dropping last internally == detect(bars, end_idx=N-1).

        When detect() is called without end_idx, it drops the last bar as
        "in-progress". So detect(bars[:7]) sees 6 completed bars (0-5).
        With end_idx=6, completed = bars[:6] = same 6 bars.
        """
        from trading.pattern_detector import BullFlagDetector
        import pandas as pd

        detector = BullFlagDetector()
        candles = [
            (4.00, 4.10, 3.98, 4.15, 200000),
            (4.15, 4.30, 4.12, 4.30, 180000),
            (4.30, 4.55, 4.28, 4.50, 150000),
            (4.50, 4.52, 4.38, 4.40, 50000),
            (4.40, 4.42, 4.33, 4.35, 30000),
            (4.35, 4.60, 4.34, 4.55, 250000),
            (4.55, 4.60, 4.50, 4.58, 100000),
            # Extra bars that should be ignored
            (4.58, 4.65, 4.55, 4.60, 80000),
            (4.60, 4.70, 4.58, 4.65, 90000),
        ]

        base_time = datetime(2026, 3, 5, 14, 0, 0, tzinfo=timezone.utc)
        records = []
        for i, (o, h, l, c, v) in enumerate(candles):
            records.append({
                'timestamp': base_time + timedelta(minutes=i),
                'open': float(o), 'high': float(h),
                'low': float(l), 'close': float(c), 'volume': int(v),
            })
        bars = pd.DataFrame(records)

        # Old way: slice to 7 bars, detect() drops last → 6 completed bars
        sliced = bars.iloc[:7].copy().reset_index(drop=True)
        pattern_old = detector.detect("TEST", sliced)

        # New way: end_idx=6 → completed = bars[:6] → 6 completed bars (same data)
        pattern_new = detector.detect("TEST", bars, end_idx=6)

        # Both should detect (or not detect) the same pattern
        if pattern_old is None:
            assert pattern_new is None
        else:
            assert pattern_new is not None
            assert pattern_old.pole_gain_pct == pytest.approx(pattern_new.pole_gain_pct, abs=0.01)
            assert pattern_old.retracement_pct == pytest.approx(pattern_new.retracement_pct, abs=0.01)
            assert pattern_old.breakout_level == pytest.approx(pattern_new.breakout_level, abs=0.01)

    def test_end_idx_none_drops_last_bar(self):
        """detect(bars) without end_idx should still drop last bar."""
        from trading.pattern_detector import BullFlagDetector
        import pandas as pd

        detector = BullFlagDetector()
        candles = [
            (4.00, 4.10, 3.98, 4.15, 200000),
            (4.15, 4.30, 4.12, 4.30, 180000),
            (4.30, 4.55, 4.28, 4.50, 150000),
            (4.50, 4.52, 4.38, 4.40, 50000),
            (4.40, 4.42, 4.33, 4.35, 30000),
            (4.35, 4.60, 4.34, 4.55, 250000),
            (4.55, 4.60, 4.50, 4.58, 100000),
        ]

        base_time = datetime(2026, 3, 5, 14, 0, 0, tzinfo=timezone.utc)
        records = []
        for i, (o, h, l, c, v) in enumerate(candles):
            records.append({
                'timestamp': base_time + timedelta(minutes=i),
                'open': float(o), 'high': float(h),
                'low': float(l), 'close': float(c), 'volume': int(v),
            })
        bars = pd.DataFrame(records)

        # Without end_idx, last bar is dropped (bar 6 is "in-progress")
        pattern = detector.detect("TEST", bars)
        assert pattern is not None  # The classic bull flag should be detected
