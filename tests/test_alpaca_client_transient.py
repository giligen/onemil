"""Transient Alpaca disconnects: one retry, WARNING-only wrapper (spec 2026-11-05).

The wrapper methods used to log ERROR and raise, and the scanner cycle logged ERROR again:
two Telegram pings for one self-healing RemoteDisconnected.
"""
import logging
from http.client import RemoteDisconnected
from unittest.mock import MagicMock, patch

import pytest
import requests

from data_sources.alpaca_client import (
    AlpacaAPIError, AlpacaAPITimeoutError, AlpacaClient, _is_transient)


@pytest.fixture
def client():
    """AlpacaClient with SDK constructors patched (no network) and sleeps disabled."""
    with patch("data_sources.alpaca_client.StockHistoricalDataClient") as d, \
         patch("data_sources.alpaca_client.TradingClient") as t, \
         patch("data_sources.alpaca_client.NewsClient"), \
         patch("data_sources.alpaca_client.time_mod.sleep"):
        d.return_value = MagicMock()
        t.return_value = MagicMock()
        c = AlpacaClient(api_key="k", api_secret="s")
        c._api_timeout = 5
        yield c


def _trade(price=10.0):
    """External SDK trade object (no spec needed)."""
    tr = MagicMock()
    tr.price, tr.size = price, 100
    tr.timestamp.isoformat.return_value = "2026-11-05T15:30:00+00:00"
    return tr


class TestIsTransient:
    @pytest.mark.parametrize("exc", [
        requests.exceptions.ConnectionError("x"), RemoteDisconnected("x")])
    def test_transient_types(self, exc):
        assert _is_transient(exc)

    def test_timeout_is_not_transient(self):
        # _call_with_timeout already retries a timeout once; a third attempt would overrun the cycle budget
        assert not _is_transient(AlpacaAPITimeoutError("x"))

    def test_wrapped_cause_is_found(self):
        try:
            try:
                raise RemoteDisconnected("gone")
            except Exception as inner:
                raise AlpacaAPIError("wrapped") from inner
        except AlpacaAPIError as e:
            assert _is_transient(e)

    def test_value_error_is_not(self):
        assert not _is_transient(ValueError("bad"))


class TestLatestTradesRetry:
    def test_one_disconnect_then_success(self, client, caplog):
        client.data_client.get_stock_latest_trade.side_effect = [
            RemoteDisconnected("gone"), {"AAPL": _trade(12.5)}]
        with caplog.at_level(logging.DEBUG):
            out = client.get_latest_trades(["AAPL"])
        assert out["AAPL"]["price"] == 12.5
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warns) == 1 and "retrying once" in warns[0].getMessage()
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    def test_two_disconnects_raise_with_warning_only(self, client, caplog):
        client.data_client.get_stock_latest_trade.side_effect = RemoteDisconnected("gone")
        with caplog.at_level(logging.DEBUG), pytest.raises(AlpacaAPIError):
            client.get_latest_trades(["AAPL"])
        assert client.data_client.get_stock_latest_trade.call_count == 2
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_value_error_raises_without_retry(self, client, caplog):
        client.data_client.get_stock_latest_trade.side_effect = ValueError("bad")
        with caplog.at_level(logging.DEBUG), pytest.raises(AlpacaAPIError):
            client.get_latest_trades(["AAPL"])
        assert client.data_client.get_stock_latest_trade.call_count == 1
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


class TestOtherMethodsShareTheHelper:
    @pytest.mark.parametrize("method,args,attr", [
        ("get_snapshots", (["AAPL"],), "get_stock_snapshot"),
        ("get_current_bars", (["AAPL"],), "get_stock_bars"),
    ])
    def test_data_methods_retry_then_raise_as_warning(self, client, caplog, method, args, attr):
        getattr(client.data_client, attr).side_effect = requests.exceptions.ConnectionError("drop")
        with caplog.at_level(logging.DEBUG), pytest.raises(AlpacaAPIError):
            getattr(client, method)(*args)
        assert getattr(client.data_client, attr).call_count == 2
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    def test_account_info_retry_then_raise_as_warning(self, client, caplog):
        client.trading_client.get_account.side_effect = requests.exceptions.ConnectionError("drop")
        with caplog.at_level(logging.DEBUG), pytest.raises(AlpacaAPIError):
            client.get_account_info()
        assert client.trading_client.get_account.call_count == 2
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
