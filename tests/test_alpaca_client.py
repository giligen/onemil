"""
Unit tests for AlpacaClient.

Covers:
- Initialization with valid/missing credentials
- Connection testing (success/failure)
- Asset retrieval
- Daily bars (including chunking)
- Intraday bars returning DataFrame
- Latest trades
- Current bars
- News fetching
- _call_with_timeout: timeout, rate limit retry, success
- Empty symbol lists returning empty dicts
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone

import pandas as pd

from data_sources.alpaca_client import (
    AlpacaClient,
    AlpacaAPIError,
    AlpacaAPITimeoutError,
    DEFAULT_API_TIMEOUT,
    MAX_RATE_LIMIT_RETRIES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
            "data_cls": mock_data_cls,
            "trading_cls": mock_trading_cls,
            "news_cls": mock_news_cls,
            "data_client": mock_data_inst,
            "trading_client": mock_trading_inst,
            "news_client": mock_news_inst,
        }


@pytest.fixture
def client(mock_sdk_clients):
    """Return an AlpacaClient with mocked SDK backends."""
    c = AlpacaClient(api_key="test-key", api_secret="test-secret")
    # Short timeout so tests don't hang
    c._api_timeout = 5
    return c


def _make_bar(**overrides):
    """Create a mock bar object (external SDK, no spec needed)."""
    defaults = {
        "open": 10.0,
        "high": 11.0,
        "low": 9.5,
        "close": 10.5,
        "volume": 100_000,
        "timestamp": datetime(2026, 3, 12, 20, 0, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    bar = MagicMock()
    for k, v in defaults.items():
        setattr(bar, k, v)
    return bar


def _make_trade(**overrides):
    """Create a mock trade object (external SDK, no spec needed)."""
    defaults = {
        "price": 12.50,
        "size": 200,
        "timestamp": datetime(2026, 3, 13, 14, 30, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    trade = MagicMock()
    for k, v in defaults.items():
        setattr(trade, k, v)
    return trade


def _make_asset(symbol="AAPL", name="Apple Inc", exchange_value="NASDAQ",
                tradable=True, fractionable=True):
    """Create a mock asset object (external SDK, no spec needed)."""
    asset = MagicMock()
    asset.symbol = symbol
    asset.name = name
    asset.tradable = tradable
    asset.fractionable = fractionable
    asset.exchange.value = exchange_value
    return asset


# ===================================================================
# Initialization
# ===================================================================

class TestInit:
    """Tests for AlpacaClient.__init__."""

    def test_init_with_valid_credentials(self, mock_sdk_clients):
        """AlpacaClient initializes successfully with valid key and secret."""
        client = AlpacaClient(api_key="key", api_secret="secret")
        mock_sdk_clients["data_cls"].assert_called_once_with("key", "secret")
        mock_sdk_clients["trading_cls"].assert_called_once_with("key", "secret", paper=True)
        assert client._api_timeout == DEFAULT_API_TIMEOUT

    def test_init_missing_api_key(self, mock_sdk_clients):
        """AlpacaClient raises AlpacaAPIError when api_key is empty."""
        with pytest.raises(AlpacaAPIError, match="ALPACA_API_KEY required"):
            AlpacaClient(api_key="", api_secret="secret")

    def test_init_missing_api_secret(self, mock_sdk_clients):
        """AlpacaClient raises AlpacaAPIError when api_secret is empty."""
        with pytest.raises(AlpacaAPIError, match="ALPACA_API_SECRET required"):
            AlpacaClient(api_key="key", api_secret="")

    def test_init_both_missing(self, mock_sdk_clients):
        """AlpacaClient raises AlpacaAPIError when both credentials are empty."""
        with pytest.raises(AlpacaAPIError, match="ALPACA_API_KEY required"):
            AlpacaClient(api_key="", api_secret="")


# ===================================================================
# test_connection
# ===================================================================

class TestTestConnection:
    """Tests for AlpacaClient.test_connection."""

    def test_connection_success(self, client, mock_sdk_clients):
        """test_connection returns True when account fetch succeeds."""
        mock_account = MagicMock()
        mock_account.account_number = "PA12345"
        mock_sdk_clients["trading_client"].get_account.return_value = mock_account

        assert client.test_connection() is True

    def test_connection_failure(self, client, mock_sdk_clients):
        """test_connection returns False when API raises exception."""
        mock_sdk_clients["trading_client"].get_account.side_effect = Exception("auth failed")

        assert client.test_connection() is False


# ===================================================================
# get_all_tradeable_assets
# ===================================================================

class TestGetAllTradeableAssets:
    """Tests for AlpacaClient.get_all_tradeable_assets."""

    def test_returns_tradeable_assets(self, client, mock_sdk_clients):
        """Returns list of dicts for tradeable assets with fractionable set."""
        assets = [
            _make_asset("AAPL", "Apple Inc", "NASDAQ", tradable=True, fractionable=True),
            _make_asset("GOOG", "Alphabet", "NASDAQ", tradable=True, fractionable=False),
        ]
        mock_sdk_clients["trading_client"].get_all_assets.return_value = assets

        result = client.get_all_tradeable_assets()

        assert len(result) == 2
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["company_name"] == "Apple Inc"
        assert result[0]["exchange"] == "NASDAQ"

    def test_filters_non_tradable(self, client, mock_sdk_clients):
        """Excludes assets where tradable is False."""
        assets = [
            _make_asset("AAPL", tradable=True, fractionable=True),
            _make_asset("DELISTED", tradable=False, fractionable=True),
        ]
        mock_sdk_clients["trading_client"].get_all_assets.return_value = assets

        result = client.get_all_tradeable_assets()
        symbols = [a["symbol"] for a in result]
        assert "AAPL" in symbols
        assert "DELISTED" not in symbols

    def test_filters_warrants_and_preferred(self, client, mock_sdk_clients):
        """Excludes warrants, preferred shares, and rights from results."""
        assets = [
            _make_asset("AAPL", name="Apple Inc", tradable=True),
            _make_asset("ARBEW", name="Arbe Robotics Ltd. Warrant", tradable=True),
            _make_asset("BAC.PRE", name="Bank of America Preferred E", tradable=True),
            _make_asset("BIORR", name="Biora Therapeutics Rights", tradable=True),
        ]
        mock_sdk_clients["trading_client"].get_all_assets.return_value = assets

        result = client.get_all_tradeable_assets()
        symbols = [a["symbol"] for a in result]
        assert "AAPL" in symbols
        assert "ARBEW" not in symbols, "Warrants should be excluded"
        assert "BAC.PRE" not in symbols, "Preferred shares should be excluded"
        assert "BIORR" not in symbols, "Rights should be excluded"

    def test_handles_missing_name(self, client, mock_sdk_clients):
        """Handles asset with name=None gracefully."""
        asset = _make_asset("XYZ", name=None)
        mock_sdk_clients["trading_client"].get_all_assets.return_value = [asset]

        result = client.get_all_tradeable_assets()
        assert result[0]["company_name"] == ""

    def test_api_error_propagates(self, client, mock_sdk_clients):
        """Non-AlpacaAPIError is wrapped in AlpacaAPIError."""
        mock_sdk_clients["trading_client"].get_all_assets.side_effect = RuntimeError("boom")

        with pytest.raises(AlpacaAPIError, match="Failed to get tradeable assets"):
            client.get_all_tradeable_assets()


# ===================================================================
# get_daily_bars
# ===================================================================

class TestGetDailyBars:
    """Tests for AlpacaClient.get_daily_bars."""

    def test_returns_bars_for_symbols(self, client, mock_sdk_clients):
        """Returns dict with close, volume, timestamp for each symbol."""
        bar = _make_bar(close=150.0, volume=5_000_000)
        mock_sdk_clients["data_client"].get_stock_bars.return_value = {
            "AAPL": [bar],
        }

        result = client.get_daily_bars(["AAPL"])

        assert "AAPL" in result
        assert result["AAPL"]["close"] == 150.0
        assert result["AAPL"]["volume"] == 5_000_000

    def test_empty_symbols_returns_empty(self, client):
        """Empty symbol list returns empty dict without API call."""
        result = client.get_daily_bars([])
        assert result == {}

    def test_missing_symbol_in_response(self, client, mock_sdk_clients):
        """Symbols not in API response are omitted from results."""
        mock_sdk_clients["data_client"].get_stock_bars.return_value = {}

        result = client.get_daily_bars(["MISSING"])
        assert result == {}

    def test_empty_bars_for_symbol(self, client, mock_sdk_clients):
        """Symbol present but with empty bar list is omitted."""
        mock_sdk_clients["data_client"].get_stock_bars.return_value = {"AAPL": []}

        result = client.get_daily_bars(["AAPL"])
        assert result == {}

    def test_chunking_large_symbol_list(self, client, mock_sdk_clients):
        """Symbols are chunked into groups of 200; multiple API calls made."""
        symbols = [f"SYM{i}" for i in range(450)]
        bar = _make_bar()

        # Return bar for every requested symbol
        def fake_get_bars(req):
            return {s: [bar] for s in req.symbol_or_symbols}

        mock_sdk_clients["data_client"].get_stock_bars.side_effect = fake_get_bars

        result = client.get_daily_bars(symbols)

        assert len(result) == 450
        # 450 symbols / 200 chunk = 3 calls
        assert mock_sdk_clients["data_client"].get_stock_bars.call_count == 3

    def test_api_error_propagates(self, client, mock_sdk_clients):
        """RuntimeError from SDK is wrapped in AlpacaAPIError."""
        mock_sdk_clients["data_client"].get_stock_bars.side_effect = RuntimeError("fail")

        with pytest.raises(AlpacaAPIError, match="Failed to get daily bars"):
            client.get_daily_bars(["AAPL"])


# ===================================================================
# get_intraday_bars
# ===================================================================

class TestGetIntradayBars:
    """Tests for AlpacaClient.get_intraday_bars."""

    def test_returns_dataframe(self, client, mock_sdk_clients):
        """Returns a DataFrame with expected columns."""
        bar = _make_bar()
        mock_sdk_clients["data_client"].get_stock_bars.return_value = {"AAPL": [bar]}

        df = client.get_intraday_bars("AAPL")

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert len(df) == 1
        assert df.iloc[0]["close"] == 10.5

    def test_no_bars_returns_empty_df(self, client, mock_sdk_clients):
        """Returns empty DataFrame when symbol has no data."""
        mock_sdk_clients["data_client"].get_stock_bars.return_value = {"AAPL": []}

        df = client.get_intraday_bars("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_symbol_missing_returns_empty_df(self, client, mock_sdk_clients):
        """Returns empty DataFrame when symbol not in response."""
        mock_sdk_clients["data_client"].get_stock_bars.return_value = {}

        df = client.get_intraday_bars("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_api_error_propagates(self, client, mock_sdk_clients):
        """RuntimeError is wrapped in AlpacaAPIError."""
        mock_sdk_clients["data_client"].get_stock_bars.side_effect = RuntimeError("fail")

        with pytest.raises(AlpacaAPIError, match="Failed to get intraday bars for AAPL"):
            client.get_intraday_bars("AAPL")


# ===================================================================
# get_latest_trades
# ===================================================================

class TestGetLatestTrades:
    """Tests for AlpacaClient.get_latest_trades."""

    def test_returns_trade_data(self, client, mock_sdk_clients):
        """Returns dict with price, size, timestamp for each symbol."""
        trade = _make_trade(price=15.0, size=300)
        mock_sdk_clients["data_client"].get_stock_latest_trade.return_value = {"AAPL": trade}

        result = client.get_latest_trades(["AAPL"])

        assert result["AAPL"]["price"] == 15.0
        assert result["AAPL"]["size"] == 300
        assert result["AAPL"]["timestamp"] is not None

    def test_empty_symbols_returns_empty(self, client):
        """Empty symbol list returns empty dict without API call."""
        result = client.get_latest_trades([])
        assert result == {}

    def test_missing_symbol_omitted(self, client, mock_sdk_clients):
        """Symbols not in response are omitted."""
        mock_sdk_clients["data_client"].get_stock_latest_trade.return_value = {}

        result = client.get_latest_trades(["MISSING"])
        assert result == {}

    def test_handles_none_price(self, client, mock_sdk_clients):
        """Trade with price=None yields 0."""
        trade = _make_trade(price=None, size=None, timestamp=None)
        mock_sdk_clients["data_client"].get_stock_latest_trade.return_value = {"X": trade}

        result = client.get_latest_trades(["X"])
        assert result["X"]["price"] == 0
        assert result["X"]["size"] == 0
        assert result["X"]["timestamp"] is None

    def test_chunking(self, client, mock_sdk_clients):
        """Large symbol lists are chunked into groups of 200."""
        symbols = [f"S{i}" for i in range(250)]
        trade = _make_trade()

        def fake_latest(req):
            return {s: trade for s in req.symbol_or_symbols}

        mock_sdk_clients["data_client"].get_stock_latest_trade.side_effect = fake_latest

        result = client.get_latest_trades(symbols)
        assert len(result) == 250
        assert mock_sdk_clients["data_client"].get_stock_latest_trade.call_count == 2

    def test_api_error_propagates(self, client, mock_sdk_clients):
        """RuntimeError is wrapped in AlpacaAPIError."""
        mock_sdk_clients["data_client"].get_stock_latest_trade.side_effect = RuntimeError("fail")

        with pytest.raises(AlpacaAPIError, match="Failed to get latest trades"):
            client.get_latest_trades(["AAPL"])


# ===================================================================
# get_current_bars
# ===================================================================

class TestGetCurrentBars:
    """Tests for AlpacaClient.get_current_bars."""

    def test_returns_bar_data(self, client, mock_sdk_clients):
        """Returns dict with OHLCV and timestamp for each symbol."""
        bar = _make_bar(open=9.0, high=12.0, low=8.5, close=11.0, volume=200_000)
        mock_sdk_clients["data_client"].get_stock_latest_bar.return_value = {"TSLA": bar}

        result = client.get_current_bars(["TSLA"])

        assert result["TSLA"]["open"] == 9.0
        assert result["TSLA"]["high"] == 12.0
        assert result["TSLA"]["low"] == 8.5
        assert result["TSLA"]["close"] == 11.0
        assert result["TSLA"]["volume"] == 200_000

    def test_empty_symbols_returns_empty(self, client):
        """Empty symbol list returns empty dict without API call."""
        result = client.get_current_bars([])
        assert result == {}

    def test_missing_symbol_omitted(self, client, mock_sdk_clients):
        """Symbols not in response are omitted from result."""
        mock_sdk_clients["data_client"].get_stock_latest_bar.return_value = {}

        result = client.get_current_bars(["MISSING"])
        assert result == {}

    def test_api_error_propagates(self, client, mock_sdk_clients):
        """RuntimeError is wrapped in AlpacaAPIError."""
        mock_sdk_clients["data_client"].get_stock_latest_bar.side_effect = RuntimeError("fail")

        with pytest.raises(AlpacaAPIError, match="Failed to get current bars"):
            client.get_current_bars(["AAPL"])


# ===================================================================
# get_news
# ===================================================================

class TestGetNews:
    """Tests for AlpacaClient.get_news."""

    def test_returns_articles(self, client, mock_sdk_clients):
        """Returns list of article dicts parsed from NewsClient response."""
        mock_article = MagicMock()
        mock_article.headline = "AAPL hits new high"
        mock_article.summary = "Apple stock surges"
        mock_article.source = "Reuters"
        mock_article.created_at = "2026-03-13T10:00:00Z"
        mock_article.url = "https://example.com/article"

        mock_news_set = MagicMock()
        mock_news_set.data = {"news": [mock_article]}
        mock_sdk_clients["news_client"].get_news.return_value = mock_news_set

        result = client.get_news("AAPL", limit=5)

        assert len(result) == 1
        assert result[0]["headline"] == "AAPL hits new high"
        assert result[0]["source"] == "Reuters"

    def test_empty_response_returns_empty(self, client, mock_sdk_clients):
        """Empty news list returns empty list."""
        mock_news_set = MagicMock()
        mock_news_set.data = {"news": []}
        mock_sdk_clients["news_client"].get_news.return_value = mock_news_set

        result = client.get_news("AAPL")
        assert result == []

    def test_api_error_returns_empty_with_warning(self, client, mock_sdk_clients):
        """API failure returns empty list (news is non-critical)."""
        mock_sdk_clients["news_client"].get_news.side_effect = RuntimeError("news api down")

        result = client.get_news("AAPL")
        assert result == []

    def test_missing_fields_default_to_empty(self, client, mock_sdk_clients):
        """Missing article attributes default to empty string via getattr."""
        mock_article = MagicMock(spec=[])  # Empty spec = no attributes
        mock_news_set = MagicMock()
        mock_news_set.data = {"news": [mock_article]}
        mock_sdk_clients["news_client"].get_news.return_value = mock_news_set

        result = client.get_news("AAPL")
        assert result[0]["headline"] == ""
        assert result[0]["summary"] == ""
        assert result[0]["url"] == ""


# ===================================================================
# _call_with_timeout
# ===================================================================

class TestCallWithTimeout:
    """Tests for AlpacaClient._call_with_timeout."""

    def test_success(self, client):
        """Successful function call returns its result."""
        result = client._call_with_timeout(lambda: 42, "test_op")
        assert result == 42

    def test_timeout_raises_timeout_error(self, client):
        """FuturesTimeoutError is converted to AlpacaAPITimeoutError."""
        def slow_func():
            import time
            time.sleep(60)

        client._api_timeout = 0.1

        with pytest.raises(AlpacaAPITimeoutError, match="timed out"):
            client._call_with_timeout(slow_func, "slow_op")

    @patch("data_sources.alpaca_client.time_mod.sleep")
    def test_rate_limit_retry_then_success(self, mock_sleep, client):
        """Rate-limited call retries with backoff and succeeds."""
        call_count = 0

        def rate_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("429 Too Many Requests")
            return "ok"

        result = client._call_with_timeout(rate_then_ok, "rate_op")

        assert result == "ok"
        assert mock_sleep.call_count == 2
        # Exponential backoff: 1.0, 2.0
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)

    @patch("data_sources.alpaca_client.time_mod.sleep")
    def test_rate_limit_exhausted(self, mock_sleep, client):
        """Raises AlpacaAPIError after exhausting all rate limit retries."""
        def always_rate_limited():
            raise Exception("rate limit exceeded")

        with pytest.raises(AlpacaAPIError, match="Rate limit exhausted"):
            client._call_with_timeout(always_rate_limited, "exhaust_op")

        assert mock_sleep.call_count == MAX_RATE_LIMIT_RETRIES

    def test_non_rate_limit_error_raises_immediately(self, client):
        """Non-rate-limit exceptions are re-raised immediately without retry."""
        def bad_func():
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            client._call_with_timeout(bad_func, "bad_op")

    @patch("data_sources.alpaca_client.time_mod.sleep")
    def test_rate_limit_detection_429(self, mock_sleep, client):
        """Detects rate limit from '429' in exception message."""
        calls = []

        def once_429():
            calls.append(1)
            if len(calls) == 1:
                raise Exception("HTTP 429")
            return "done"

        result = client._call_with_timeout(once_429, "op")
        assert result == "done"

    @patch("data_sources.alpaca_client.time_mod.sleep")
    def test_rate_limit_detection_too_many_requests(self, mock_sleep, client):
        """Detects rate limit from 'too many requests' in exception message."""
        calls = []

        def once_tmr():
            calls.append(1)
            if len(calls) == 1:
                raise Exception("Too Many Requests")
            return "done"

        result = client._call_with_timeout(once_tmr, "op")
        assert result == "done"
