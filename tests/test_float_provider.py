"""
Tests for data_sources/float_provider.py - FloatProvider.

Covers:
- get_float success (returns int)
- get_float when floatShares is missing (returns None)
- get_float exception handling
- get_float_batch with progress logging
- get_stock_info success and failure
- Exponential backoff retry on rate limits
- Request throttling between batch fetches
"""

import pytest
from unittest.mock import patch, MagicMock, call

from data_sources.float_provider import FloatProvider


@pytest.fixture
def provider():
    """Create a FloatProvider instance with no delay for testing."""
    return FloatProvider(max_retries=3, initial_backoff=0.01, request_delay=0)


# =============================================================================
# get_float
# =============================================================================

class TestGetFloat:
    """Tests for FloatProvider.get_float."""

    @patch('data_sources.float_provider.yf')
    def test_success_returns_int(self, mock_yf, provider):
        """get_float returns an integer when floatShares is available."""
        mock_ticker = MagicMock()  # External SDK object, OK without spec
        mock_ticker.info = {'floatShares': 5_500_000.0}
        mock_yf.Ticker.return_value = mock_ticker

        result = provider.get_float("AAPL")

        assert result == 5_500_000
        assert isinstance(result, int)
        mock_yf.Ticker.assert_called_once_with("AAPL")

    @patch('data_sources.float_provider.yf')
    def test_returns_none_when_float_missing(self, mock_yf, provider):
        """get_float returns None when floatShares key is missing from info."""
        mock_ticker = MagicMock()
        mock_ticker.info = {'sector': 'Technology'}  # No floatShares
        mock_yf.Ticker.return_value = mock_ticker

        result = provider.get_float("NODATA")

        assert result is None

    @patch('data_sources.float_provider.yf')
    def test_returns_none_when_float_is_none(self, mock_yf, provider):
        """get_float returns None when floatShares value is explicitly None."""
        mock_ticker = MagicMock()
        mock_ticker.info = {'floatShares': None}
        mock_yf.Ticker.return_value = mock_ticker

        result = provider.get_float("NULLFLOAT")

        assert result is None

    @patch('data_sources.float_provider.yf')
    def test_exception_returns_none_after_retries(self, mock_yf, provider):
        """get_float returns None after exhausting retries on persistent error."""
        mock_yf.Ticker.side_effect = Exception("Network error")

        result = provider.get_float("FAIL")

        assert result is None


# =============================================================================
# Retry / Backoff
# =============================================================================

class TestRetryBackoff:
    """Tests for exponential backoff retry logic."""

    @patch('data_sources.float_provider.time')
    @patch('data_sources.float_provider.yf')
    def test_retries_on_rate_limit_then_succeeds(self, mock_yf, mock_time, provider):
        """get_float retries on 429 rate limit and succeeds on retry."""
        mock_ticker_fail = MagicMock()
        mock_ticker_success = MagicMock()
        mock_ticker_success.info = {'floatShares': 3_000_000.0}

        # First call raises rate limit, second succeeds
        mock_yf.Ticker.side_effect = [
            Exception("429 Too Many Requests"),
            mock_ticker_success,
        ]

        result = provider.get_float("RETRY")

        assert result == 3_000_000
        assert mock_yf.Ticker.call_count == 2
        mock_time.sleep.assert_called_once()  # Backoff sleep

    @patch('data_sources.float_provider.time')
    @patch('data_sources.float_provider.yf')
    def test_retries_on_connection_error(self, mock_yf, mock_time, provider):
        """get_float retries on connection errors."""
        mock_ticker_success = MagicMock()
        mock_ticker_success.info = {'floatShares': 1_000_000.0}

        mock_yf.Ticker.side_effect = [
            Exception("Connection timeout"),
            mock_ticker_success,
        ]

        result = provider.get_float("CONNFAIL")

        assert result == 1_000_000
        assert mock_yf.Ticker.call_count == 2

    @patch('data_sources.float_provider.time')
    @patch('data_sources.float_provider.yf')
    def test_exponential_backoff_doubles(self, mock_yf, mock_time):
        """Backoff doubles on each retry attempt."""
        provider = FloatProvider(max_retries=3, initial_backoff=1.0, request_delay=0)

        mock_yf.Ticker.side_effect = Exception("429 rate limit")

        provider.get_float("EXHAUST")

        # Should have slept 3 times: 1.0s, 2.0s, 4.0s
        sleep_calls = mock_time.sleep.call_args_list
        assert len(sleep_calls) == 3
        assert sleep_calls[0] == call(1.0)
        assert sleep_calls[1] == call(2.0)
        assert sleep_calls[2] == call(4.0)

    @patch('data_sources.float_provider.yf')
    def test_non_rate_limit_error_no_retry(self, mock_yf):
        """Non-rate-limit errors don't trigger retries."""
        provider = FloatProvider(max_retries=3, initial_backoff=0.01, request_delay=0)

        mock_yf.Ticker.side_effect = KeyError("unexpected_key")

        result = provider.get_float("NORETRY")

        assert result is None
        assert mock_yf.Ticker.call_count == 1  # No retries

    @patch('data_sources.float_provider.time')
    @patch('data_sources.float_provider.yf')
    def test_rate_limit_exhausted_returns_none(self, mock_yf, mock_time, provider):
        """get_float returns None after exhausting all retry attempts."""
        mock_yf.Ticker.side_effect = Exception("429 Too Many Requests")

        result = provider.get_float("EXHAUST")

        assert result is None
        # max_retries=3, so 4 total attempts (initial + 3 retries)
        assert mock_yf.Ticker.call_count == 4


# =============================================================================
# get_float_batch
# =============================================================================

class TestGetFloatBatch:
    """Tests for FloatProvider.get_float_batch."""

    @patch('data_sources.float_provider.yf')
    def test_batch_returns_dict_for_all_symbols(self, mock_yf, provider):
        """get_float_batch returns a dict with an entry for every input symbol."""
        def make_ticker(symbol):
            """Create a mock ticker with float data based on symbol."""
            ticker = MagicMock()
            floats = {'AAA': 1_000_000.0, 'BBB': 2_000_000.0, 'CCC': None}
            ticker.info = {'floatShares': floats.get(symbol)}
            return ticker

        mock_yf.Ticker.side_effect = make_ticker

        symbols = ['AAA', 'BBB', 'CCC']
        result = provider.get_float_batch(symbols, progress_interval=2)

        assert len(result) == 3
        assert result['AAA'] == 1_000_000
        assert result['BBB'] == 2_000_000
        assert result['CCC'] is None

    @patch('data_sources.float_provider.yf')
    def test_batch_empty_list(self, mock_yf, provider):
        """get_float_batch returns empty dict for empty input."""
        result = provider.get_float_batch([])
        assert result == {}

    @patch('data_sources.float_provider.yf')
    def test_batch_handles_exceptions(self, mock_yf, provider):
        """get_float_batch continues processing after an exception for one symbol."""
        def make_ticker(symbol):
            """Raise on second symbol, succeed on others."""
            if symbol == 'BAD':
                raise Exception("API down")
            ticker = MagicMock()
            ticker.info = {'floatShares': 500_000.0}
            return ticker

        mock_yf.Ticker.side_effect = make_ticker

        result = provider.get_float_batch(['OK1', 'BAD', 'OK2'], progress_interval=1)

        assert result['OK1'] == 500_000
        assert result['BAD'] is None
        assert result['OK2'] == 500_000

    @patch('data_sources.float_provider.time')
    @patch('data_sources.float_provider.yf')
    def test_batch_throttles_requests(self, mock_yf, mock_time):
        """get_float_batch pauses between requests when request_delay > 0."""
        provider = FloatProvider(max_retries=0, initial_backoff=0.01, request_delay=0.5)

        def make_ticker(symbol):
            """Create a mock ticker."""
            ticker = MagicMock()
            ticker.info = {'floatShares': 1_000_000.0}
            return ticker

        mock_yf.Ticker.side_effect = make_ticker

        provider.get_float_batch(['A', 'B', 'C'], progress_interval=10)

        # Should sleep between requests (2 sleeps for 3 symbols: after A and B, not after C)
        sleep_calls = [c for c in mock_time.sleep.call_args_list if c == call(0.5)]
        assert len(sleep_calls) == 2


# =============================================================================
# get_stock_info
# =============================================================================

class TestGetStockInfo:
    """Tests for FloatProvider.get_stock_info."""

    @patch('data_sources.float_provider.yf')
    def test_success(self, mock_yf, provider):
        """get_stock_info returns sector, country, and float_shares when available."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            'sector': 'Technology',
            'country': 'United States',
            'floatShares': 8_000_000.0,
        }
        mock_yf.Ticker.return_value = mock_ticker

        result = provider.get_stock_info("AAPL")

        assert result['sector'] == 'Technology'
        assert result['country'] == 'United States'
        assert result['float_shares'] == 8_000_000
        assert isinstance(result['float_shares'], int)

    @patch('data_sources.float_provider.yf')
    def test_missing_fields(self, mock_yf, provider):
        """get_stock_info returns None for missing fields."""
        mock_ticker = MagicMock()
        mock_ticker.info = {}  # No sector, country, or floatShares
        mock_yf.Ticker.return_value = mock_ticker

        result = provider.get_stock_info("EMPTY")

        assert result['sector'] is None
        assert result['country'] is None
        assert result['float_shares'] is None

    @patch('data_sources.float_provider.yf')
    def test_exception_returns_none_dict(self, mock_yf, provider):
        """get_stock_info returns dict with None values on persistent error."""
        mock_yf.Ticker.side_effect = Exception("API error")

        result = provider.get_stock_info("FAIL")

        assert result == {'sector': None, 'country': None, 'float_shares': None}

    @patch('data_sources.float_provider.time')
    @patch('data_sources.float_provider.yf')
    def test_stock_info_retries_on_rate_limit(self, mock_yf, mock_time, provider):
        """get_stock_info retries on rate limit errors."""
        mock_ticker = MagicMock()
        mock_ticker.info = {'sector': 'Finance', 'country': 'US', 'floatShares': 5_000_000.0}

        mock_yf.Ticker.side_effect = [
            Exception("429 rate limit"),
            mock_ticker,
        ]

        result = provider.get_stock_info("RETRY")

        assert result['sector'] == 'Finance'
        assert result['float_shares'] == 5_000_000
