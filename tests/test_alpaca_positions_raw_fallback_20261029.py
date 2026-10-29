"""Positions read survives an SDK enum rejection (TXSE incident 2026-10-28): raw REST fallback."""
import logging
import uuid
from unittest.mock import patch, MagicMock

import pytest
from pydantic import ValidationError
from alpaca.trading.models import Position

from data_sources.alpaca_client import AlpacaClient, AlpacaAPIError

WARN_TEXT = "SDK model rejected the broker payload"


def _row(symbol, exchange, qty="100", side="long"):
    """One raw /positions JSON row as the broker sends it."""
    return {
        'asset_id': str(uuid.uuid4()), 'symbol': symbol, 'exchange': exchange, 'asset_class': 'us_equity',
        'avg_entry_price': '4.40', 'qty': qty, 'side': side, 'market_value': '450.00',
        'cost_basis': '440.00', 'unrealized_pl': '10.00', 'unrealized_plpc': '0.0227',
    }


def _real_validation_error():
    """A genuine pydantic error from the real Position model rejecting exchange='TXSE'."""
    with pytest.raises(ValidationError) as ei:
        Position.model_validate(_row('TXG', 'TXSE'))
    return ei.value


@pytest.fixture
def trading():
    """Patched SDK constructors; yields the trading client double."""
    with patch("data_sources.alpaca_client.StockHistoricalDataClient"), \
         patch("data_sources.alpaca_client.NewsClient"), \
         patch("data_sources.alpaca_client.TradingClient") as cls:
        inst = MagicMock()
        cls.return_value = inst
        yield inst


@pytest.fixture
def client(trading):
    c = AlpacaClient(api_key="k", api_secret="s")
    c._api_timeout = 5
    return c


def test_validation_error_falls_back_to_raw_rest_with_exchange(client, trading):
    """SDK ValidationError -> raw /positions list incl. the TXSE row; same dict shape + exchange."""
    trading.get_all_positions.side_effect = _real_validation_error()
    trading.get.return_value = [_row('TXG', 'TXSE', qty="25"), _row('AAPL', 'NASDAQ')]
    result = client.get_open_positions()
    trading.get.assert_called_once_with('/positions')
    assert [p['symbol'] for p in result] == ['TXG', 'AAPL']
    assert result[0]['exchange'] == 'TXSE'
    assert result[0]['qty'] == 25 and isinstance(result[0]['qty'], int)
    assert result[0]['side'] == 'long'
    assert result[0]['avg_entry_price'] == 4.40
    assert set(result[0]) == {'symbol', 'qty', 'side', 'avg_entry_price', 'market_value',
                              'unrealized_pl', 'unrealized_plpc', 'exchange'}


def test_value_error_with_validation_text_also_falls_back(client, trading):
    """A ValueError whose text says 'validation error' is treated the same."""
    trading.get_all_positions.side_effect = ValueError("1 validation error for list[Position]")
    trading.get.return_value = [_row('TXG', 'TXSE')]
    assert client.get_open_positions()[0]['exchange'] == 'TXSE'


def test_other_value_error_is_not_swallowed(client, trading):
    """An unrelated ValueError keeps the existing ERROR path (no raw call)."""
    trading.get_all_positions.side_effect = ValueError("boom")
    with pytest.raises(AlpacaAPIError, match="Failed to get open positions"):
        client.get_open_positions()
    trading.get.assert_not_called()


def test_warning_once_per_hour(client, trading, caplog):
    """Second fallback within the hour logs no second WARNING; never an ERROR."""
    trading.get_all_positions.side_effect = _real_validation_error()
    trading.get.return_value = [_row('TXG', 'TXSE')]
    with caplog.at_level(logging.WARNING, logger="data_sources.alpaca_client"):
        client.get_open_positions()
        client.get_open_positions()
    assert sum(WARN_TEXT in r.getMessage() for r in caplog.records) == 1
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    # after an hour it warns again
    client._positions_raw_warn_ts -= 3601
    with caplog.at_level(logging.WARNING, logger="data_sources.alpaca_client"):
        client.get_open_positions()
    assert sum(WARN_TEXT in r.getMessage() for r in caplog.records) == 2


def test_raw_path_failure_takes_error_path(client, trading, caplog):
    """If the raw REST call fails too, the existing ERROR + AlpacaAPIError path runs."""
    trading.get_all_positions.side_effect = _real_validation_error()
    trading.get.side_effect = RuntimeError("raw down")
    with caplog.at_level(logging.ERROR, logger="data_sources.alpaca_client"):
        with pytest.raises(AlpacaAPIError, match="Failed to get open positions"):
            client.get_open_positions()
    assert any("Failed to get open positions" in r.getMessage() and r.levelno == logging.ERROR
               for r in caplog.records)


def test_normal_sdk_path_unchanged_and_has_exchange(client, trading):
    """Healthy SDK payload: no raw call; dict carries the exchange string."""
    pos = MagicMock()
    pos.symbol, pos.qty, pos.side = "AAPL", "100", "long"
    pos.avg_entry_price, pos.market_value = "4.40", "450.00"
    pos.unrealized_pl, pos.unrealized_plpc = "10.00", "0.0227"
    pos.exchange = "NASDAQ"
    trading.get_all_positions.return_value = [pos]
    result = client.get_open_positions()
    trading.get.assert_not_called()
    assert result[0]['exchange'] == 'NASDAQ' and result[0]['qty'] == 100
