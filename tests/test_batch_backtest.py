"""
Unit tests for batch_backtest module.

Covers:
- find_big_movers: filtering logic, edge cases, sorting
- write_csv_report: CSV format, headers, data accuracy
- run_batch_backtest: progress, error handling, result collection
- get_daily_bars_range: chunking, date handling
- print_summary: output formatting, edge cases
- fetch_daily_bars_cached: cache hit/miss, API fetch + store
- get_1min_bars_cached: cache hit/miss, API fetch + store
- Database daily_bars and intraday_bars_1min cache methods
"""

import csv
import os
from datetime import date, datetime, timezone
from io import StringIO
from unittest.mock import MagicMock, patch, AsyncMock

import pandas as pd
import pytest

from backtest import BacktestResult, SimulatedTrade, BacktestRunner
from batch_backtest import (
    find_big_movers,
    write_csv_report,
    run_batch_backtest,
    print_summary,
    utc_to_et_str,
    fetch_daily_bars_cached,
    get_1min_bars_cached,
    INTRADAY_MOVE_THRESHOLD,
    CSV_HEADERS,
)
from persistence.database import Database
from data_sources.alpaca_client import AlpacaClient, AlpacaAPIError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_daily_bars():
    """Daily bars with a mix of qualifying and non-qualifying moves."""
    return {
        "PLYX": [
            {"date": date(2026, 3, 5), "open": 5.0, "high": 6.0, "low": 5.0, "close": 5.8, "volume": 500000},
            # (6-5)/5 = 20% — qualifies
            {"date": date(2026, 3, 6), "open": 5.5, "high": 5.7, "low": 5.4, "close": 5.6, "volume": 200000},
            # (5.7-5.4)/5.4 = 5.6% — does NOT qualify
        ],
        "SVCO": [
            {"date": date(2026, 3, 5), "open": 10.0, "high": 12.0, "low": 10.0, "close": 11.5, "volume": 1000000},
            # (12-10)/10 = 20% — qualifies
        ],
        "FLAT": [
            {"date": date(2026, 3, 5), "open": 8.0, "high": 8.2, "low": 8.0, "close": 8.1, "volume": 50000},
            # (8.2-8.0)/8.0 = 2.5% — does NOT qualify
        ],
    }


@pytest.fixture
def sample_trade():
    """A SimulatedTrade for CSV testing."""
    return SimulatedTrade(
        symbol="PLYX",
        entry_time=datetime(2026, 3, 5, 14, 30, 0, tzinfo=timezone.utc),
        entry_price=5.50,
        stop_loss=5.30,
        take_profit=5.90,
        shares=1000,
        exit_time=datetime(2026, 3, 5, 15, 45, 0, tzinfo=timezone.utc),
        exit_price=5.90,
        exit_reason="target",
        pnl=400.0,
        pnl_pct=7.27,
        bars_held=15,
    )


@pytest.fixture
def sample_backtest_result(sample_trade):
    """A BacktestResult with one trade."""
    return BacktestResult(
        symbol="PLYX",
        trade_date="2026-03-05",
        total_bars=390,
        patterns_detected=2,
        trades_simulated=[sample_trade],
    )


@pytest.fixture
def mock_sdk_clients():
    """Patch all Alpaca SDK client constructors so no real connection is made."""
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
        }


@pytest.fixture
def client(mock_sdk_clients):
    """AlpacaClient with mocked SDK backends."""
    c = AlpacaClient(api_key="test-key", api_secret="test-secret")
    c._api_timeout = 5
    return c


# ---------------------------------------------------------------------------
# Tests: find_big_movers
# ---------------------------------------------------------------------------


class TestFindBigMovers:
    """Tests for find_big_movers filtering logic."""

    def test_filters_qualifying_moves(self, sample_daily_bars):
        """Only symbol/date pairs with >= 10% move should be returned."""
        movers = find_big_movers(sample_daily_bars)
        assert len(movers) == 2
        symbols = {m[0] for m in movers}
        assert "PLYX" in symbols
        assert "SVCO" in symbols
        assert "FLAT" not in symbols

    def test_correct_dates_returned(self, sample_daily_bars):
        """Each qualifying pair should have the correct date."""
        movers = find_big_movers(sample_daily_bars)
        mover_dict = {m[0]: m[1] for m in movers}
        assert mover_dict["PLYX"] == date(2026, 3, 5)
        assert mover_dict["SVCO"] == date(2026, 3, 5)

    def test_excludes_non_qualifying_day_same_symbol(self, sample_daily_bars):
        """PLYX on 3/6 (5.6% move) should not appear."""
        movers = find_big_movers(sample_daily_bars)
        plyx_dates = [m[1] for m in movers if m[0] == "PLYX"]
        assert date(2026, 3, 6) not in plyx_dates

    def test_empty_input(self):
        """Empty dict should return empty list."""
        assert find_big_movers({}) == []

    def test_zero_low_skipped(self):
        """Bars with low=0 should be skipped (avoid division by zero)."""
        bars = {"ZERO": [{"date": date(2026, 3, 5), "open": 0, "high": 1, "low": 0, "close": 0.5, "volume": 100}]}
        movers = find_big_movers(bars)
        assert len(movers) == 0

    def test_custom_threshold(self, sample_daily_bars):
        """Custom threshold should change which pairs qualify."""
        # 5% threshold should also include PLYX on 3/6 (5.6%)
        movers = find_big_movers(sample_daily_bars, threshold=0.05)
        assert len(movers) == 3  # PLYX 3/5, PLYX 3/6, SVCO 3/5

    def test_sorted_by_date_then_symbol(self):
        """Results should be sorted by (date, symbol)."""
        bars = {
            "ZZZ": [{"date": date(2026, 3, 1), "open": 5, "high": 6, "low": 5, "close": 5.5, "volume": 100}],
            "AAA": [{"date": date(2026, 3, 1), "open": 5, "high": 6, "low": 5, "close": 5.5, "volume": 100}],
            "MMM": [{"date": date(2026, 3, 2), "open": 5, "high": 6, "low": 5, "close": 5.5, "volume": 100}],
        }
        movers = find_big_movers(bars)
        assert movers == [
            ("AAA", date(2026, 3, 1)),
            ("ZZZ", date(2026, 3, 1)),
            ("MMM", date(2026, 3, 2)),
        ]

    def test_exactly_at_threshold(self):
        """Move exactly at 10% should qualify."""
        bars = {"EDGE": [{"date": date(2026, 3, 5), "open": 10, "high": 11, "low": 10, "close": 10.5, "volume": 100}]}
        movers = find_big_movers(bars)
        assert len(movers) == 1

    def test_just_below_threshold(self):
        """Move at 9.99% should not qualify."""
        bars = {"EDGE": [{"date": date(2026, 3, 5), "open": 10, "high": 10.999, "low": 10, "close": 10.5, "volume": 100}]}
        movers = find_big_movers(bars)
        assert len(movers) == 0


# ---------------------------------------------------------------------------
# Tests: write_csv_report
# ---------------------------------------------------------------------------


class TestWriteCsvReport:
    """Tests for CSV report generation."""

    def test_writes_correct_headers(self, sample_backtest_result, tmp_path):
        """CSV file should have the expected headers."""
        output = str(tmp_path / "test.csv")
        write_csv_report([sample_backtest_result], output)

        with open(output, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert headers == CSV_HEADERS

    def test_writes_trade_data(self, sample_backtest_result, tmp_path):
        """Trade data should be correctly written to CSV."""
        output = str(tmp_path / "test.csv")
        count = write_csv_report([sample_backtest_result], output)

        assert count == 1
        with open(output, 'r') as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["symbol"] == "PLYX"
        assert row["date"] == "2026-03-05"
        assert row["entry_price"] == "5.50"
        assert row["stop_loss"] == "5.30"
        assert row["target"] == "5.90"
        assert row["shares"] == "1000"
        assert row["exit_price"] == "5.90"
        assert row["exit_reason"] == "target"
        assert row["pnl"] == "400.00"

    def test_empty_results_writes_header_only(self, tmp_path):
        """No results should produce CSV with only headers."""
        output = str(tmp_path / "test.csv")
        count = write_csv_report([], output)

        assert count == 0
        with open(output, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 1  # Header only

    def test_multiple_results(self, sample_backtest_result, tmp_path):
        """Multiple results should each contribute their trades."""
        trade2 = SimulatedTrade(
            symbol="SVCO", entry_time=datetime(2026, 3, 5, 14, 0, tzinfo=timezone.utc),
            entry_price=10.0, stop_loss=9.80, take_profit=10.40, shares=500,
            exit_time=datetime(2026, 3, 5, 16, 0, tzinfo=timezone.utc),
            exit_price=9.80, exit_reason="stop", pnl=-100.0, pnl_pct=-2.0,
        )
        result2 = BacktestResult(
            symbol="SVCO", trade_date="2026-03-05", total_bars=390,
            patterns_detected=1, trades_simulated=[trade2],
        )

        output = str(tmp_path / "test.csv")
        count = write_csv_report([sample_backtest_result, result2], output)
        assert count == 2

    def test_result_with_no_trades(self, tmp_path):
        """Results with no trades should not produce rows."""
        result = BacktestResult(
            symbol="NONE", trade_date="2026-03-05", total_bars=390,
            patterns_detected=0,
        )
        output = str(tmp_path / "test.csv")
        count = write_csv_report([result], output)
        assert count == 0

    def test_returns_trade_count(self, sample_backtest_result, tmp_path):
        """Return value should be the number of trades written."""
        output = str(tmp_path / "test.csv")
        count = write_csv_report([sample_backtest_result], output)
        assert count == 1


# ---------------------------------------------------------------------------
# Tests: utc_to_et_str
# ---------------------------------------------------------------------------


class TestUtcToEtStr:
    """Tests for UTC to ET time conversion."""

    def test_converts_utc_to_et(self):
        """UTC 14:30 should become ET 10:30 (EDT, -4h)."""
        ts = datetime(2026, 3, 5, 14, 30, 0, tzinfo=timezone.utc)
        assert utc_to_et_str(ts) == "10:30:00"

    def test_none_returns_empty(self):
        """None timestamp should return empty string."""
        assert utc_to_et_str(None) == ""


# ---------------------------------------------------------------------------
# Tests: run_batch_backtest
# ---------------------------------------------------------------------------


class TestRunBatchBacktest:
    """Tests for batch backtest runner."""

    def test_runs_all_movers(self, client, mock_sdk_clients):
        """Should attempt to backtest each (symbol, date) pair."""
        movers = [
            ("PLYX", date(2026, 3, 5)),
            ("SVCO", date(2026, 3, 5)),
        ]

        # Mock 1-min bars response
        bars_df = pd.DataFrame({
            'timestamp': [datetime(2026, 3, 5, 13, 30 + i, tzinfo=timezone.utc) for i in range(20)],
            'open': [5.0 + i * 0.1 for i in range(20)],
            'high': [5.1 + i * 0.1 for i in range(20)],
            'low': [4.9 + i * 0.1 for i in range(20)],
            'close': [5.05 + i * 0.1 for i in range(20)],
            'volume': [10000] * 20,
        })

        runner = MagicMock(spec=BacktestRunner)
        runner.run.return_value = BacktestResult(
            symbol="TEST", trade_date="2026-03-05", total_bars=20, patterns_detected=0,
        )

        with patch.object(client, 'get_historical_1min_bars', return_value=bars_df):
            results = run_batch_backtest(movers, client, runner)

        assert len(results) == 2
        assert runner.run.call_count == 2

    def test_skips_on_api_error(self, client, mock_sdk_clients):
        """API errors should be logged and skipped, not abort batch."""
        movers = [("FAIL", date(2026, 3, 5)), ("OK", date(2026, 3, 5))]

        bars_df = pd.DataFrame({
            'timestamp': [datetime(2026, 3, 5, 13, 30 + i, tzinfo=timezone.utc) for i in range(20)],
            'open': [5.0] * 20, 'high': [5.1] * 20, 'low': [4.9] * 20,
            'close': [5.0] * 20, 'volume': [10000] * 20,
        })

        call_count = 0

        def mock_bars(symbol, start, end):
            nonlocal call_count
            call_count += 1
            if symbol == "FAIL":
                raise AlpacaAPIError("Rate limit exceeded")
            return bars_df

        runner = MagicMock(spec=BacktestRunner)
        runner.run.return_value = BacktestResult(
            symbol="OK", trade_date="2026-03-05", total_bars=20, patterns_detected=0,
        )

        with patch.object(client, 'get_historical_1min_bars', side_effect=mock_bars):
            results = run_batch_backtest(movers, client, runner)

        assert len(results) == 1
        assert results[0].symbol == "OK"

    def test_skips_empty_bars(self, client, mock_sdk_clients):
        """Symbol/dates with no bars should be skipped."""
        movers = [("NODATA", date(2026, 3, 5))]

        runner = MagicMock(spec=BacktestRunner)
        with patch.object(client, 'get_historical_1min_bars', return_value=pd.DataFrame()):
            results = run_batch_backtest(movers, client, runner)

        assert len(results) == 0
        runner.run.assert_not_called()

    def test_empty_movers_list(self, client, mock_sdk_clients):
        """Empty movers list should return empty results."""
        runner = MagicMock(spec=BacktestRunner)
        results = run_batch_backtest([], client, runner)
        assert results == []


# ---------------------------------------------------------------------------
# Tests: print_summary
# ---------------------------------------------------------------------------


class TestPrintSummary:
    """Tests for console summary output."""

    def test_prints_without_error(self, sample_backtest_result, capsys):
        """Summary should print without raising errors."""
        movers = [("PLYX", date(2026, 3, 5))]
        print_summary(100, movers, [sample_backtest_result])
        captured = capsys.readouterr()
        assert "BATCH BACKTEST SUMMARY" in captured.out
        assert "100 symbols" in captured.out
        assert "1" in captured.out  # 1 trade

    def test_handles_no_trades(self, capsys):
        """Should handle zero trades gracefully (no division by zero)."""
        result = BacktestResult(
            symbol="NONE", trade_date="2026-03-05", total_bars=390, patterns_detected=0,
        )
        print_summary(50, [], [result])
        captured = capsys.readouterr()
        assert "0.0%" in captured.out  # win rate

    def test_handles_empty_results(self, capsys):
        """Should handle empty results list."""
        print_summary(50, [], [])
        captured = capsys.readouterr()
        assert "BATCH BACKTEST SUMMARY" in captured.out


# ---------------------------------------------------------------------------
# Tests: get_daily_bars_range (AlpacaClient)
# ---------------------------------------------------------------------------


class TestGetDailyBarsRange:
    """Tests for AlpacaClient.get_daily_bars_range."""

    def test_returns_bars_for_symbols(self, client, mock_sdk_clients):
        """Should return daily bars dict for each symbol with data."""
        bar1 = MagicMock()
        bar1.timestamp = datetime(2026, 3, 5, 20, 0, tzinfo=timezone.utc)
        bar1.open = 5.0
        bar1.high = 6.0
        bar1.low = 5.0
        bar1.close = 5.8
        bar1.volume = 500000

        mock_response = MagicMock()
        mock_response.data = {"PLYX": [bar1]}
        mock_sdk_clients["data_client"].get_stock_bars.return_value = mock_response

        result = client.get_daily_bars_range(
            ["PLYX"], date(2026, 3, 1), date(2026, 3, 13)
        )

        assert "PLYX" in result
        assert len(result["PLYX"]) == 1
        assert result["PLYX"][0]["high"] == 6.0
        assert result["PLYX"][0]["low"] == 5.0
        assert result["PLYX"][0]["volume"] == 500000

    def test_empty_symbols_returns_empty(self, client, mock_sdk_clients):
        """Empty symbol list should return empty dict without API call."""
        result = client.get_daily_bars_range([], date(2026, 3, 1), date(2026, 3, 13))
        assert result == {}
        mock_sdk_clients["data_client"].get_stock_bars.assert_not_called()

    def test_chunks_large_symbol_lists(self, client, mock_sdk_clients):
        """Should chunk symbols by 200."""
        symbols = [f"SYM{i}" for i in range(450)]

        mock_response = MagicMock()
        mock_response.data = {}
        mock_sdk_clients["data_client"].get_stock_bars.return_value = mock_response

        client.get_daily_bars_range(symbols, date(2026, 3, 1), date(2026, 3, 13))

        # 450 symbols / 200 chunk = 3 API calls
        assert mock_sdk_clients["data_client"].get_stock_bars.call_count == 3

    def test_skips_symbols_with_no_bars(self, client, mock_sdk_clients):
        """Symbols with no data should not appear in results."""
        mock_response = MagicMock()
        mock_response.data = {"PLYX": []}  # Empty bars
        mock_sdk_clients["data_client"].get_stock_bars.return_value = mock_response

        result = client.get_daily_bars_range(
            ["PLYX", "MISSING"], date(2026, 3, 1), date(2026, 3, 13)
        )
        assert "PLYX" not in result
        assert "MISSING" not in result

    def test_api_error_propagates(self, client, mock_sdk_clients):
        """AlpacaAPIError should propagate up."""
        mock_sdk_clients["data_client"].get_stock_bars.side_effect = Exception("Network error")

        with pytest.raises(AlpacaAPIError, match="Network error"):
            client.get_daily_bars_range(["PLYX"], date(2026, 3, 1), date(2026, 3, 13))

    def test_multiple_bars_per_symbol(self, client, mock_sdk_clients):
        """Should return all daily bars for each symbol."""
        bar1 = MagicMock()
        bar1.timestamp = datetime(2026, 3, 5, 20, 0, tzinfo=timezone.utc)
        bar1.open, bar1.high, bar1.low, bar1.close, bar1.volume = 5.0, 6.0, 5.0, 5.8, 500000

        bar2 = MagicMock()
        bar2.timestamp = datetime(2026, 3, 6, 20, 0, tzinfo=timezone.utc)
        bar2.open, bar2.high, bar2.low, bar2.close, bar2.volume = 5.5, 5.7, 5.4, 5.6, 200000

        mock_response = MagicMock()
        mock_response.data = {"PLYX": [bar1, bar2]}
        mock_sdk_clients["data_client"].get_stock_bars.return_value = mock_response

        result = client.get_daily_bars_range(
            ["PLYX"], date(2026, 3, 1), date(2026, 3, 13)
        )
        assert len(result["PLYX"]) == 2


# ---------------------------------------------------------------------------
# Tests: Database daily_bars cache
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    """Fresh test database."""
    db_path = str(tmp_path / "test_onemil.db")
    database = Database(db_path=db_path)
    yield database
    database.close()


class TestDailyBarsCache:
    """Tests for Database daily bars caching."""

    def test_save_and_retrieve(self, db):
        """Saved daily bars should be retrievable."""
        bars = [
            {"symbol": "PLYX", "date": "2026-03-05", "open": 5.0, "high": 6.0,
             "low": 5.0, "close": 5.8, "volume": 500000},
            {"symbol": "PLYX", "date": "2026-03-06", "open": 5.5, "high": 5.7,
             "low": 5.4, "close": 5.6, "volume": 200000},
        ]
        count = db.save_daily_bars(bars)
        assert count == 2

        result = db.get_daily_bars_cached(["PLYX"], "2026-03-01", "2026-03-13")
        assert "PLYX" in result
        assert len(result["PLYX"]) == 2
        assert result["PLYX"][0]["high"] == 6.0

    def test_upsert_overwrites(self, db):
        """Saving same symbol/date again should update, not duplicate."""
        bars = [{"symbol": "PLYX", "date": "2026-03-05", "open": 5.0,
                 "high": 6.0, "low": 5.0, "close": 5.8, "volume": 500000}]
        db.save_daily_bars(bars)

        # Update with different close
        bars[0]["close"] = 6.2
        db.save_daily_bars(bars)

        result = db.get_daily_bars_cached(["PLYX"], "2026-03-01", "2026-03-13")
        assert len(result["PLYX"]) == 1
        assert result["PLYX"][0]["close"] == 6.2

    def test_get_cached_symbols(self, db):
        """Should return set of symbols with cached data."""
        bars = [
            {"symbol": "PLYX", "date": "2026-03-05", "open": 5, "high": 6,
             "low": 5, "close": 5.5, "volume": 100},
            {"symbol": "SVCO", "date": "2026-03-05", "open": 10, "high": 12,
             "low": 10, "close": 11, "volume": 200},
        ]
        db.save_daily_bars(bars)

        cached = db.get_cached_daily_bar_symbols("2026-03-01", "2026-03-13")
        assert cached == {"PLYX", "SVCO"}

    def test_empty_save(self, db):
        """Saving empty list should return 0."""
        assert db.save_daily_bars([]) == 0

    def test_date_range_filter(self, db):
        """Should only return bars within the requested range."""
        bars = [
            {"symbol": "PLYX", "date": "2026-02-28", "open": 5, "high": 6,
             "low": 5, "close": 5.5, "volume": 100},
            {"symbol": "PLYX", "date": "2026-03-05", "open": 5, "high": 6,
             "low": 5, "close": 5.5, "volume": 100},
        ]
        db.save_daily_bars(bars)

        result = db.get_daily_bars_cached(["PLYX"], "2026-03-01", "2026-03-13")
        assert len(result["PLYX"]) == 1  # Only March bar


# ---------------------------------------------------------------------------
# Tests: Database intraday 1-min bars cache
# ---------------------------------------------------------------------------


class TestIntradayBarsCache:
    """Tests for Database intraday 1-min bars caching."""

    def test_save_and_retrieve(self, db):
        """Saved intraday bars should be retrievable."""
        bars = [
            {"timestamp": datetime(2026, 3, 5, 13, 30, tzinfo=timezone.utc),
             "open": 5.0, "high": 5.1, "low": 4.9, "close": 5.05, "volume": 10000},
            {"timestamp": datetime(2026, 3, 5, 13, 31, tzinfo=timezone.utc),
             "open": 5.05, "high": 5.15, "low": 5.0, "close": 5.10, "volume": 8000},
        ]
        count = db.save_intraday_bars("PLYX", "2026-03-05", bars)
        assert count == 2

        result = db.get_intraday_bars_cached("PLYX", "2026-03-05")
        assert len(result) == 2
        assert result[0]["open"] == 5.0
        assert result[1]["close"] == 5.10

    def test_cache_miss_returns_empty(self, db):
        """No cached data should return empty list."""
        result = db.get_intraday_bars_cached("MISSING", "2026-03-05")
        assert result == []

    def test_save_empty_returns_zero(self, db):
        """Saving empty list should return 0."""
        assert db.save_intraday_bars("PLYX", "2026-03-05", []) == 0

    def test_different_dates_isolated(self, db):
        """Bars from different dates should not mix."""
        bars_day1 = [
            {"timestamp": datetime(2026, 3, 5, 13, 30, tzinfo=timezone.utc),
             "open": 5.0, "high": 5.1, "low": 4.9, "close": 5.05, "volume": 10000},
        ]
        bars_day2 = [
            {"timestamp": datetime(2026, 3, 6, 13, 30, tzinfo=timezone.utc),
             "open": 6.0, "high": 6.1, "low": 5.9, "close": 6.05, "volume": 12000},
        ]
        db.save_intraday_bars("PLYX", "2026-03-05", bars_day1)
        db.save_intraday_bars("PLYX", "2026-03-06", bars_day2)

        result = db.get_intraday_bars_cached("PLYX", "2026-03-05")
        assert len(result) == 1
        assert result[0]["open"] == 5.0


# ---------------------------------------------------------------------------
# Tests: fetch_daily_bars_cached (integration of cache + API)
# ---------------------------------------------------------------------------


class TestFetchDailyBarsCached:
    """Tests for fetch_daily_bars_cached function."""

    def test_fetches_from_api_when_cache_empty(self, client, mock_sdk_clients, db):
        """Should call API when DB cache is empty, then store results."""
        bar1 = MagicMock()
        bar1.timestamp = datetime(2026, 3, 5, 20, 0, tzinfo=timezone.utc)
        bar1.open, bar1.high, bar1.low, bar1.close, bar1.volume = 5.0, 6.0, 5.0, 5.8, 500000

        mock_response = MagicMock()
        mock_response.data = {"PLYX": [bar1]}
        mock_sdk_clients["data_client"].get_stock_bars.return_value = mock_response

        result = fetch_daily_bars_cached(
            ["PLYX"], date(2026, 3, 1), date(2026, 3, 13), client, db
        )

        assert "PLYX" in result
        # Verify it was stored in cache
        cached = db.get_cached_daily_bar_symbols("2026-03-01", "2026-03-13")
        assert "PLYX" in cached

    def test_uses_cache_when_available(self, client, mock_sdk_clients, db):
        """Should NOT call API when all symbols are cached."""
        # Pre-populate cache
        db.save_daily_bars([
            {"symbol": "PLYX", "date": "2026-03-05", "open": 5.0, "high": 6.0,
             "low": 5.0, "close": 5.8, "volume": 500000},
        ])

        result = fetch_daily_bars_cached(
            ["PLYX"], date(2026, 3, 1), date(2026, 3, 13), client, db
        )

        assert "PLYX" in result
        # API should NOT have been called
        mock_sdk_clients["data_client"].get_stock_bars.assert_not_called()

    def test_mixed_cached_and_uncached(self, client, mock_sdk_clients, db):
        """Should only fetch uncached symbols from API."""
        # Pre-populate PLYX in cache
        db.save_daily_bars([
            {"symbol": "PLYX", "date": "2026-03-05", "open": 5.0, "high": 6.0,
             "low": 5.0, "close": 5.8, "volume": 500000},
        ])

        # SVCO not cached — mock API response for it
        bar1 = MagicMock()
        bar1.timestamp = datetime(2026, 3, 5, 20, 0, tzinfo=timezone.utc)
        bar1.open, bar1.high, bar1.low, bar1.close, bar1.volume = 10.0, 12.0, 10.0, 11.5, 1000000

        mock_response = MagicMock()
        mock_response.data = {"SVCO": [bar1]}
        mock_sdk_clients["data_client"].get_stock_bars.return_value = mock_response

        result = fetch_daily_bars_cached(
            ["PLYX", "SVCO"], date(2026, 3, 1), date(2026, 3, 13), client, db
        )

        assert "PLYX" in result
        assert "SVCO" in result
        # API was called only once (for SVCO chunk)
        assert mock_sdk_clients["data_client"].get_stock_bars.call_count == 1


# ---------------------------------------------------------------------------
# Tests: get_1min_bars_cached
# ---------------------------------------------------------------------------


class TestGet1minBarsCached:
    """Tests for get_1min_bars_cached function."""

    def test_returns_cached_bars(self, client, mock_sdk_clients, db):
        """Should return cached bars without API call."""
        bars = [
            {"timestamp": datetime(2026, 3, 5, 13, 30, tzinfo=timezone.utc),
             "open": 5.0, "high": 5.1, "low": 4.9, "close": 5.05, "volume": 10000},
        ]
        db.save_intraday_bars("PLYX", "2026-03-05", bars)

        result = get_1min_bars_cached("PLYX", date(2026, 3, 5), client, db)

        assert len(result) == 1
        mock_sdk_clients["data_client"].get_stock_bars.assert_not_called()

    def test_fetches_and_caches_on_miss(self, client, mock_sdk_clients, db):
        """Should fetch from API and cache when not in DB."""
        bar1 = MagicMock()
        bar1.timestamp = datetime(2026, 3, 5, 13, 30, tzinfo=timezone.utc)
        bar1.open, bar1.high, bar1.low, bar1.close, bar1.volume = 5.0, 5.1, 4.9, 5.05, 10000

        mock_response = MagicMock()
        mock_response.data = {"PLYX": [bar1]}
        mock_sdk_clients["data_client"].get_stock_bars.return_value = mock_response

        result = get_1min_bars_cached("PLYX", date(2026, 3, 5), client, db)

        assert len(result) == 1
        # Verify it was cached
        cached = db.get_intraday_bars_cached("PLYX", "2026-03-05")
        assert len(cached) == 1


# ---------------------------------------------------------------------------
# Market Regime + Circuit Breaker in Batch Backtest
# ---------------------------------------------------------------------------


class TestBatchBacktestRegimeAndCB:
    """Tests for market regime filter and circuit breaker in run_batch_backtest."""

    def test_regime_skips_bad_dates(self):
        """run_batch_backtest skips entire dates when regime is bearish."""
        from trading.market_regime import MarketRegimeFilter

        # Create regime that blocks Mar 5 but allows Mar 6
        regime = MarketRegimeFilter(enabled=True, spy_5d_return_min=-2.0)
        spy_bars = [
            # For Mar 5: prior 6 = [Feb25..Mar4], T-1=Mar4=480, T-6=Feb25=500 → -4%
            {'date': date(2026, 2, 25), 'close': 500.0},
            {'date': date(2026, 2, 26), 'close': 498.0},
            {'date': date(2026, 2, 27), 'close': 496.0},
            {'date': date(2026, 2, 28), 'close': 494.0},
            {'date': date(2026, 3, 3), 'close': 490.0},
            {'date': date(2026, 3, 4), 'close': 480.0},  # T-1 for Mar 5: -4%
            # For Mar 6: prior 6 = [Feb26..Mar5], T-1=Mar5=510, T-6=Feb26=498 → +2.4%
            {'date': date(2026, 3, 5), 'close': 510.0},
        ]
        regime.load_spy_bars(spy_bars)

        movers = [
            ("PLYX", date(2026, 3, 5)),  # Should be skipped (regime blocks)
            ("SVCO", date(2026, 3, 5)),  # Should be skipped (same date)
            ("AAPL", date(2026, 3, 6)),  # Should be allowed (regime OK)
        ]

        # Mock client and runner
        client = MagicMock(spec=AlpacaClient)
        runner = MagicMock(spec=BacktestRunner)

        # Create a result for AAPL
        aapl_result = BacktestResult(
            symbol="AAPL", trade_date="2026-03-06",
            total_bars=100, patterns_detected=1,
            trades_simulated=[],
        )
        runner.run.return_value = aapl_result

        # Mock 1-min bars
        bars_df = pd.DataFrame({
            'timestamp': [datetime(2026, 3, 6, 14, 0, tzinfo=timezone.utc)],
            'open': [5.0], 'high': [5.5], 'low': [4.8],
            'close': [5.3], 'volume': [100000],
        })
        client.get_historical_1min_bars.return_value = bars_df

        results = run_batch_backtest(
            movers, client, runner,
            market_regime=regime,
        )

        # Only AAPL should have been backtested (Mar 6 is allowed)
        assert len(results) == 1
        assert results[0].symbol == "AAPL"
        # Runner should have been called only once (for AAPL)
        assert runner.run.call_count == 1

    def test_cb_skips_after_drawdown(self):
        """Circuit breaker skips trades after drawdown threshold is hit."""
        movers = [
            ("SYM1", date(2026, 3, 5)),
            ("SYM2", date(2026, 3, 5)),
            ("SYM3", date(2026, 3, 5)),
        ]

        client = MagicMock(spec=AlpacaClient)
        runner = MagicMock(spec=BacktestRunner)

        # SYM1 loses $2000 (triggers CB at $1500 DD)
        trade1 = MagicMock()
        trade1.pnl = -2000.0
        result1 = BacktestResult(
            symbol="SYM1", trade_date="2026-03-05",
            total_bars=100, patterns_detected=1,
            trades_simulated=[trade1],
        )

        # SYM3 would be a winning trade but should not be reached if CB pause=1
        trade3 = MagicMock()
        trade3.pnl = 500.0
        result3 = BacktestResult(
            symbol="SYM3", trade_date="2026-03-05",
            total_bars=100, patterns_detected=1,
            trades_simulated=[trade3],
        )

        # Runner returns different results for each call
        runner.run.side_effect = [result1, result3]

        bars_df = pd.DataFrame({
            'timestamp': [datetime(2026, 3, 5, 14, 0, tzinfo=timezone.utc)],
            'open': [5.0], 'high': [5.5], 'low': [4.8],
            'close': [5.3], 'volume': [100000],
        })
        client.get_historical_1min_bars.return_value = bars_df

        results = run_batch_backtest(
            movers, client, runner,
            circuit_breaker_dd=1500.0,
            circuit_breaker_pause=1,
        )

        # SYM1 runs (loses $2000, triggers CB), SYM2 is skipped (CB), SYM3 runs
        assert len(results) == 2
        assert results[0].symbol == "SYM1"
        assert results[1].symbol == "SYM3"
        assert runner.run.call_count == 2  # SYM1 and SYM3, not SYM2
