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
    """Tests for AlpacaClient.get_current_bars (uses 15-min StockBarsRequest)."""

    def test_returns_bar_data(self, client, mock_sdk_clients):
        """Returns dict with OHLCV and timestamp for each symbol."""
        bar = _make_bar(open=9.0, high=12.0, low=8.5, close=11.0, volume=200_000)
        mock_response = MagicMock()
        mock_response.data = {"TSLA": [bar]}
        mock_sdk_clients["data_client"].get_stock_bars.return_value = mock_response

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
        mock_response = MagicMock()
        mock_response.data = {}
        mock_sdk_clients["data_client"].get_stock_bars.return_value = mock_response

        result = client.get_current_bars(["MISSING"])
        assert result == {}

    def test_uses_latest_bar_from_multiple(self, client, mock_sdk_clients):
        """When multiple 15-min bars returned, uses the most recent one."""
        bar_old = _make_bar(open=8.0, high=9.0, low=7.5, close=8.5, volume=100_000)
        bar_new = _make_bar(open=9.0, high=12.0, low=8.5, close=11.0, volume=200_000)
        mock_response = MagicMock()
        mock_response.data = {"TSLA": [bar_old, bar_new]}
        mock_sdk_clients["data_client"].get_stock_bars.return_value = mock_response

        result = client.get_current_bars(["TSLA"])

        assert result["TSLA"]["close"] == 11.0
        assert result["TSLA"]["volume"] == 200_000

    def test_api_error_propagates(self, client, mock_sdk_clients):
        """RuntimeError is wrapped in AlpacaAPIError."""
        mock_sdk_clients["data_client"].get_stock_bars.side_effect = RuntimeError("fail")

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

    def test_api_error_raises_alpaca_error(self, client, mock_sdk_clients):
        """API failure raises AlpacaAPIError (not silently swallowed)."""
        mock_sdk_clients["news_client"].get_news.side_effect = RuntimeError("news api down")

        with pytest.raises(AlpacaAPIError, match="News API call failed"):
            client.get_news("AAPL")

    def test_parse_error_returns_empty_with_warning(self, client, mock_sdk_clients):
        """Response parsing failure returns empty list (non-critical)."""
        mock_news_set = MagicMock()
        # .data is a property that raises when accessed during parsing
        type(mock_news_set).data = property(lambda self: (_ for _ in ()).throw(ValueError("bad data")))
        mock_sdk_clients["news_client"].get_news.return_value = mock_news_set

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
            time.sleep(1)

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


# ===================================================================
# get_1min_bars (Phase 2)
# ===================================================================

class TestGet1MinBars:
    """Tests for AlpacaClient.get_1min_bars."""

    def test_returns_dataframe(self, client, mock_sdk_clients):
        """Returns DataFrame with OHLCV columns."""
        bar = _make_bar(open=4.0, high=4.1, low=3.9, close=4.05, volume=100_000)
        mock_response = MagicMock()
        mock_response.data = {"AAPL": [bar]}
        mock_sdk_clients["data_client"].get_stock_bars.return_value = mock_response

        result = client.get_1min_bars("AAPL", lookback_minutes=30)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["open"] == 4.0
        assert result.iloc[0]["close"] == 4.05
        assert result.iloc[0]["volume"] == 100_000

    def test_empty_response_returns_empty_df(self, client, mock_sdk_clients):
        """Empty response returns empty DataFrame."""
        mock_response = MagicMock()
        mock_response.data = {}
        mock_sdk_clients["data_client"].get_stock_bars.return_value = mock_response

        result = client.get_1min_bars("AAPL")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_api_error_propagates(self, client, mock_sdk_clients):
        """API error is wrapped in AlpacaAPIError."""
        mock_sdk_clients["data_client"].get_stock_bars.side_effect = RuntimeError("fail")

        with pytest.raises(AlpacaAPIError, match="Failed to get 1-min bars"):
            client.get_1min_bars("AAPL")

    def test_multiple_bars_returned(self, client, mock_sdk_clients):
        """Multiple bars are returned in DataFrame."""
        bars = [
            _make_bar(open=4.0, close=4.05, volume=100_000),
            _make_bar(open=4.05, close=4.10, volume=120_000),
            _make_bar(open=4.10, close=4.15, volume=90_000),
        ]
        mock_response = MagicMock()
        mock_response.data = {"AAPL": bars}
        mock_sdk_clients["data_client"].get_stock_bars.return_value = mock_response

        result = client.get_1min_bars("AAPL")
        assert len(result) == 3


# ===================================================================
# submit_bracket_order (Phase 2)
# ===================================================================

class TestSubmitBracketOrder:
    """Tests for AlpacaClient.submit_bracket_order."""

    def test_submits_order_successfully(self, client, mock_sdk_clients):
        """Successful bracket order returns dict with id and status."""
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_order.status = MagicMock()
        mock_order.status.value = "accepted"
        mock_sdk_clients["trading_client"].submit_order.return_value = mock_order

        result = client.submit_bracket_order(
            symbol="AAPL", qty=100, side="buy",
            limit_price=4.40, tp_price=4.90, sl_price=4.20,
        )

        assert result["id"] == "order-123"
        assert result["status"] == "accepted"
        assert result["symbol"] == "AAPL"

    def test_api_error_propagates(self, client, mock_sdk_clients):
        """API error raises AlpacaAPIError."""
        mock_sdk_clients["trading_client"].submit_order.side_effect = RuntimeError("rejected")

        with pytest.raises(AlpacaAPIError, match="Failed to submit bracket order"):
            client.submit_bracket_order("AAPL", 100, "buy", 4.40, 4.90, 4.20)


# ===================================================================
# get_open_positions (Phase 2)
# ===================================================================

class TestGetOpenPositions:
    """Tests for AlpacaClient.get_open_positions."""

    def test_returns_positions(self, client, mock_sdk_clients):
        """Returns list of position dicts."""
        mock_pos = MagicMock()
        mock_pos.symbol = "AAPL"
        mock_pos.qty = "100"
        mock_pos.side = "long"
        mock_pos.avg_entry_price = "4.40"
        mock_pos.market_value = "450.00"
        mock_pos.unrealized_pl = "10.00"
        mock_pos.unrealized_plpc = "0.0227"
        mock_sdk_clients["trading_client"].get_all_positions.return_value = [mock_pos]

        result = client.get_open_positions()

        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["qty"] == 100

    def test_empty_positions(self, client, mock_sdk_clients):
        """Empty positions returns empty list."""
        mock_sdk_clients["trading_client"].get_all_positions.return_value = []

        result = client.get_open_positions()
        assert result == []

    def test_api_error_propagates(self, client, mock_sdk_clients):
        """API error raises AlpacaAPIError."""
        mock_sdk_clients["trading_client"].get_all_positions.side_effect = RuntimeError("fail")

        with pytest.raises(AlpacaAPIError, match="Failed to get open positions"):
            client.get_open_positions()


# ===================================================================
# get_account_info (Phase 2)
# ===================================================================

class TestGetAccountInfo:
    """Tests for AlpacaClient.get_account_info."""

    def test_returns_account_data(self, client, mock_sdk_clients):
        """Returns dict with account equity, buying power, etc."""
        mock_account = MagicMock()
        mock_account.equity = "25000.00"
        mock_account.buying_power = "50000.00"
        mock_account.cash = "25000.00"
        mock_account.daytrade_count = "2"
        mock_account.pattern_day_trader = False
        mock_sdk_clients["trading_client"].get_account.return_value = mock_account

        result = client.get_account_info()

        assert result["equity"] == 25000.0
        assert result["buying_power"] == 50000.0
        assert result["daytrade_count"] == 2

    def test_api_error_propagates(self, client, mock_sdk_clients):
        """API error raises AlpacaAPIError."""
        mock_sdk_clients["trading_client"].get_account.side_effect = RuntimeError("fail")

        with pytest.raises(AlpacaAPIError, match="Failed to get account info"):
            client.get_account_info()


# ===================================================================
# cancel_order (Phase 2)
# ===================================================================

class TestCancelOrder:
    """Tests for AlpacaClient.cancel_order."""

    def test_cancels_successfully(self, client, mock_sdk_clients):
        """Successful cancellation returns True."""
        mock_sdk_clients["trading_client"].cancel_order_by_id.return_value = None

        result = client.cancel_order("order-123")
        assert result is True

    def test_api_error_propagates(self, client, mock_sdk_clients):
        """API error raises AlpacaAPIError."""
        mock_sdk_clients["trading_client"].cancel_order_by_id.side_effect = RuntimeError("fail")

        with pytest.raises(AlpacaAPIError, match="Failed to cancel order"):
            client.cancel_order("order-123")


# ===================================================================
# get_order (Phase 2)
# ===================================================================

class TestGetOrder:
    """Tests for AlpacaClient.get_order."""

    def test_returns_order_data(self, client, mock_sdk_clients):
        """Returns dict with order details."""
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_order.status = MagicMock()
        mock_order.status.value = "filled"
        mock_order.symbol = "AAPL"
        mock_order.qty = "100"
        mock_order.filled_qty = "100"
        mock_order.filled_avg_price = "4.40"
        mock_order.side = MagicMock()
        mock_order.side.value = "buy"
        mock_order.type = MagicMock()
        mock_order.type.value = "limit"
        mock_sdk_clients["trading_client"].get_order_by_id.return_value = mock_order

        result = client.get_order("order-123")

        assert result["id"] == "order-123"
        assert result["status"] == "filled"
        assert result["symbol"] == "AAPL"
        assert result["filled_avg_price"] == 4.40

    def test_api_error_propagates(self, client, mock_sdk_clients):
        """API error raises AlpacaAPIError."""
        mock_sdk_clients["trading_client"].get_order_by_id.side_effect = RuntimeError("fail")

        with pytest.raises(AlpacaAPIError, match="Failed to get order"):
            client.get_order("order-123")

    def test_returns_legs(self, client, mock_sdk_clients):
        """get_order returns bracket child order legs."""
        mock_sl_leg = MagicMock()
        mock_sl_leg.id = "sl-leg-123"
        mock_sl_leg.side = MagicMock()
        mock_sl_leg.side.value = "sell"
        mock_sl_leg.type = MagicMock()
        mock_sl_leg.type.value = "stop"
        mock_sl_leg.stop_price = 4.20
        mock_sl_leg.limit_price = None
        mock_sl_leg.status = MagicMock()
        mock_sl_leg.status.value = "new"

        mock_tp_leg = MagicMock()
        mock_tp_leg.id = "tp-leg-456"
        mock_tp_leg.side = MagicMock()
        mock_tp_leg.side.value = "sell"
        mock_tp_leg.type = MagicMock()
        mock_tp_leg.type.value = "limit"
        mock_tp_leg.stop_price = None
        mock_tp_leg.limit_price = 5.50
        mock_tp_leg.status = MagicMock()
        mock_tp_leg.status.value = "new"

        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_order.status = MagicMock()
        mock_order.status.value = "filled"
        mock_order.symbol = "AAPL"
        mock_order.qty = "100"
        mock_order.filled_qty = "100"
        mock_order.filled_avg_price = "4.40"
        mock_order.side = MagicMock()
        mock_order.side.value = "buy"
        mock_order.type = MagicMock()
        mock_order.type.value = "stop_limit"
        mock_order.legs = [mock_sl_leg, mock_tp_leg]

        mock_sdk_clients["trading_client"].get_order_by_id.return_value = mock_order

        result = client.get_order("order-123")

        assert len(result['legs']) == 2
        sl = result['legs'][0]
        assert sl['id'] == "sl-leg-123"
        assert sl['side'] == "sell"
        assert sl['stop_price'] == 4.20
        assert sl['limit_price'] is None
        tp = result['legs'][1]
        assert tp['limit_price'] == 5.50
        assert tp['stop_price'] is None

    def test_no_legs_returns_empty_list(self, client, mock_sdk_clients):
        """get_order returns empty legs list when order has no legs."""
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_order.status = MagicMock()
        mock_order.status.value = "filled"
        mock_order.symbol = "AAPL"
        mock_order.qty = "100"
        mock_order.filled_qty = "100"
        mock_order.filled_avg_price = "4.40"
        mock_order.side = MagicMock()
        mock_order.side.value = "buy"
        mock_order.type = MagicMock()
        mock_order.type.value = "limit"
        mock_order.legs = None

        mock_sdk_clients["trading_client"].get_order_by_id.return_value = mock_order

        result = client.get_order("order-123")
        assert result['legs'] == []


# ===================================================================
# replace_order_stop_price (Phase 3)
# ===================================================================

class TestReplaceOrderStopPrice:
    """Tests for AlpacaClient.replace_order_stop_price."""

    def test_replaces_stop_successfully(self, client, mock_sdk_clients):
        """Successful stop replacement returns id and status."""
        mock_order = MagicMock()
        mock_order.id = "sl-leg-123"
        mock_order.status = MagicMock()
        mock_order.status.value = "replaced"
        mock_sdk_clients["trading_client"].replace_order_by_id.return_value = mock_order

        result = client.replace_order_stop_price("sl-leg-123", 4.35)

        assert result['id'] == "sl-leg-123"
        assert result['status'] == "replaced"
        mock_sdk_clients["trading_client"].replace_order_by_id.assert_called_once()

    def test_api_error_propagates(self, client, mock_sdk_clients):
        """API error raises AlpacaAPIError."""
        mock_sdk_clients["trading_client"].replace_order_by_id.side_effect = RuntimeError("fail")

        with pytest.raises(AlpacaAPIError, match="Failed to replace order stop price"):
            client.replace_order_stop_price("sl-leg-123", 4.35)


# ===================================================================
# replace_order_limit_price (gap-fill target adjustment)
# ===================================================================

class TestReplaceOrderLimitPrice:
    """Tests for AlpacaClient.replace_order_limit_price."""

    def test_replaces_limit_successfully(self, client, mock_sdk_clients):
        """Successful limit replacement returns id and status."""
        mock_order = MagicMock()
        mock_order.id = "tp-leg-456"
        mock_order.status = MagicMock()
        mock_order.status.value = "replaced"
        mock_sdk_clients["trading_client"].replace_order_by_id.return_value = mock_order

        result = client.replace_order_limit_price("tp-leg-456", 5.05)

        assert result['id'] == "tp-leg-456"
        assert result['status'] == "replaced"
        mock_sdk_clients["trading_client"].replace_order_by_id.assert_called_once()

    def test_api_error_propagates(self, client, mock_sdk_clients):
        """API error raises AlpacaAPIError."""
        mock_sdk_clients["trading_client"].replace_order_by_id.side_effect = RuntimeError("fail")

        with pytest.raises(AlpacaAPIError, match="Failed to replace order limit price"):
            client.replace_order_limit_price("tp-leg-456", 5.05)


class TestMarketCalendar:
    """Tests for calendar methods: get_market_calendar, is_trading_day, is_short_trading_day."""

    def test_get_market_calendar_returns_days(self, client, mock_sdk_clients):
        """Returns list of trading day dicts."""
        from datetime import date as date_cls

        mock_day = MagicMock()
        mock_day.date = date_cls(2026, 3, 13)
        mock_day.open = MagicMock(hour=9, minute=30)
        mock_day.close = MagicMock(hour=16, minute=0)

        mock_sdk_clients["trading_client"].get_calendar.return_value = [mock_day]

        result = client.get_market_calendar(date_cls(2026, 3, 13), date_cls(2026, 3, 13))
        assert len(result) == 1
        assert result[0]['date'] == date_cls(2026, 3, 13)

    def test_get_market_calendar_api_error(self, client, mock_sdk_clients):
        """API error raises AlpacaAPIError."""
        from datetime import date as date_cls
        mock_sdk_clients["trading_client"].get_calendar.side_effect = RuntimeError("fail")

        with pytest.raises(AlpacaAPIError, match="Failed to get market calendar"):
            client.get_market_calendar(date_cls(2026, 3, 13), date_cls(2026, 3, 13))

    def test_is_trading_day_true(self, client, mock_sdk_clients):
        """Returns True when calendar has an entry for the date."""
        mock_day = MagicMock()
        mock_sdk_clients["trading_client"].get_calendar.return_value = [mock_day]

        assert client.is_trading_day() is True

    def test_is_trading_day_false_on_weekend(self, client, mock_sdk_clients):
        """Returns False when calendar is empty (weekend/holiday)."""
        mock_sdk_clients["trading_client"].get_calendar.return_value = []

        assert client.is_trading_day() is False

    def test_is_short_trading_day_true(self, client, mock_sdk_clients):
        """Returns True when close is before 16:00."""
        mock_day = MagicMock()
        mock_day.date = MagicMock()
        mock_day.open = MagicMock(hour=9, minute=30)
        mock_day.close = MagicMock(hour=13, minute=0)  # 1pm close = short day

        mock_sdk_clients["trading_client"].get_calendar.return_value = [mock_day]

        assert client.is_short_trading_day() is True

    def test_is_short_trading_day_false_normal(self, client, mock_sdk_clients):
        """Returns False on normal day (closes at 16:00)."""
        mock_day = MagicMock()
        mock_day.date = MagicMock()
        mock_day.open = MagicMock(hour=9, minute=30)
        mock_day.close = MagicMock(hour=16, minute=0)

        mock_sdk_clients["trading_client"].get_calendar.return_value = [mock_day]

        assert client.is_short_trading_day() is False

    def test_is_short_trading_day_not_trading_day(self, client, mock_sdk_clients):
        """Returns False when not a trading day at all."""
        mock_sdk_clients["trading_client"].get_calendar.return_value = []

        assert client.is_short_trading_day() is False
