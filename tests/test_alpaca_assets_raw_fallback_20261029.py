"""Asset reads survive an SDK enum rejection (TXSE, 2026-10-28/29): raw REST fallback."""
import logging
import uuid
from typing import List
from unittest.mock import patch, MagicMock

import pytest
from pydantic import TypeAdapter, ValidationError
from alpaca.trading.models import Asset

from data_sources.alpaca_client import AlpacaClient, AlpacaAPIError, RawAsset, _asset_from_json
from scripts import momentum_sleeve as runner

WARN_TEXT = "SDK rejected the payload"


def _row(symbol, exchange="NASDAQ", name=None, tradable=True, shortable=False, etb=False):
    """One raw /assets JSON row as the broker sends it."""
    return {
        'id': str(uuid.uuid4()), 'class': 'us_equity', 'exchange': exchange, 'symbol': symbol,
        'name': name or f'{symbol} Inc. Common Stock', 'status': 'active', 'tradable': tradable,
        'marginable': True, 'shortable': shortable, 'easy_to_borrow': etb, 'fractionable': True,
        'maintenance_margin_requirement': 30, 'some_future_key': 1,
    }


def _real_validation_error():
    """A genuine pydantic error from the real Asset model rejecting exchange='TXSE'."""
    with pytest.raises(ValidationError) as ei:
        TypeAdapter(List[Asset]).validate_python([_row('TXG', 'TXSE')])
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


def test_asset_from_json_maps_fields_and_ignores_unknown():
    a = _asset_from_json(_row('TXG', 'TXSE', shortable=True, etb=True))
    assert isinstance(a, RawAsset)
    assert (a.symbol, a.exchange, a.asset_class, a.status) == ('TXG', 'TXSE', 'us_equity', 'active')
    assert a.tradable and a.marginable and a.shortable and a.easy_to_borrow and a.fractionable
    assert _asset_from_json({}).tradable is False and _asset_from_json({}).exchange == ''


def test_get_assets_tolerant_falls_back_with_txse_and_warns_once(client, trading, caplog):
    """Real pydantic error -> one raw /assets call, RawAssets with exchange str kept, WARNING once per hour."""
    trading.get_all_assets.side_effect = _real_validation_error()
    trading.get.return_value = [_row('TXG', 'TXSE'), _row('AAPL'), _row('OFF', tradable=False)]
    with caplog.at_level(logging.WARNING, logger="data_sources.alpaca_client"):
        out = client.get_assets_tolerant()
        client.get_assets_tolerant()
    assert trading.get.call_args_list[0][0][0] == '/assets'
    params = trading.get.call_args_list[0][0][1]
    assert params['status'] == 'active' and params['asset_class'] == 'us_equity'
    assert [a.symbol for a in out] == ['TXG', 'AAPL', 'OFF']
    assert out[0].exchange == 'TXSE' and out[0].tradable and out[2].tradable is False
    msgs = [r.getMessage() for r in caplog.records if WARN_TEXT in r.getMessage()]
    assert len(msgs) == 1 and 'TXSE' in msgs[0] and 'returned 3 assets' in msgs[0]
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    client._assets_raw_warned_at -= 3601
    with caplog.at_level(logging.WARNING, logger="data_sources.alpaca_client"):
        client.get_assets_tolerant()
    assert sum(WARN_TEXT in r.getMessage() for r in caplog.records) == 2


def test_get_all_tradeable_assets_on_fallback_keeps_txse_and_filters(client, trading):
    """Dicts carry exchange 'TXSE'; _is_common_stock still drops the leveraged wrapper; non-tradable skipped."""
    trading.get_all_assets.side_effect = _real_validation_error()
    trading.get.return_value = [
        _row('TXG', 'TXSE'), _row('AAPL', name='Apple Inc. Common Stock'),
        _row('TQQQ', name='ProShares UltraPro QQQ 3x Shares'), _row('OFF', tradable=False)]
    out = client.get_all_tradeable_assets()
    assert {d['symbol'] for d in out} == {'TXG', 'AAPL'}
    assert next(d for d in out if d['symbol'] == 'TXG')['exchange'] == 'TXSE'
    assert set(out[0]) == {'symbol', 'company_name', 'exchange', 'marginable'}


def test_get_all_tradeable_assets_healthy_sdk_path_unchanged(client, trading):
    """Healthy SDK payload: no raw call; enum exchange -> its value."""
    a = MagicMock()
    a.tradable, a.symbol, a.name, a.marginable = True, 'AAPL', 'Apple Inc. Common Stock', True
    a.exchange.value = 'NASDAQ'
    trading.get_all_assets.return_value = [a]
    out = client.get_all_tradeable_assets()
    trading.get.assert_not_called()
    assert out[0]['exchange'] == 'NASDAQ'


def test_get_asset_tolerant_and_shortability_fallback(client, trading):
    """get_asset ValidationError -> /assets/{symbol}; real flags returned, WARNING once for many symbols."""
    trading.get_asset.side_effect = _real_validation_error()
    trading.get.return_value = _row('TXG', 'TXSE', shortable=True, etb=True)
    assert client.get_shortability('TXG') == {'shortable': True, 'easy_to_borrow': True}
    trading.get.assert_called_once_with('/assets/TXG')
    assert client.is_marginable('TXG') is True
    assert client.get_asset_name('TXG') == 'TXG Inc. Common Stock'


def test_asset_warning_once_per_hour_across_symbols(client, trading, caplog):
    """Per-symbol fallbacks share the hourly WARNING."""
    trading.get_asset.side_effect = _real_validation_error()
    trading.get.side_effect = lambda path, *a: _row(path.rsplit('/', 1)[1], 'TXSE')
    with caplog.at_level(logging.WARNING, logger="data_sources.alpaca_client"):
        for s in ('AAA', 'BBB', 'CCC'):
            client.get_asset_tolerant(s)
    assert sum(WARN_TEXT in r.getMessage() for r in caplog.records) == 1


def test_non_validation_exception_propagates_no_fallback(client, trading):
    """A timeout/API/other error keeps the existing behaviour: no raw call, tradeable wrapper raises."""
    trading.get_all_assets.side_effect = RuntimeError("boom")
    with pytest.raises(AlpacaAPIError, match="Failed to get tradeable assets"):
        client.get_all_tradeable_assets()
    trading.get.assert_not_called()
    trading.get_asset.side_effect = ValueError("unrelated")
    with pytest.raises(ValueError):
        client.get_asset_tolerant('AAA')
    trading.get.assert_not_called()
    assert client.get_shortability('AAA') == {'shortable': False, 'easy_to_borrow': False}


def test_raw_path_failure_is_not_swallowed(client, trading):
    """If the raw REST call fails too, the error propagates (no silent empty list)."""
    trading.get_all_assets.side_effect = _real_validation_error()
    trading.get.side_effect = RuntimeError("raw down")
    with pytest.raises(RuntimeError, match="raw down"):
        client.get_assets_tolerant()


def test_momentum_sleeve_fetch_assets_on_fallback_path(client, trading):
    """Integration (real AlpacaClient, mocked network): fetch_assets returns the tradable-filtered frame."""
    trading.get_all_assets.side_effect = _real_validation_error()
    trading.get.return_value = [_row('TXG', 'TXSE'), _row('AAPL'), _row('OFF', tradable=False)]
    df = runner.fetch_assets(client)
    assert list(df.columns) == ['symbol', 'name']
    assert list(df['symbol']) == ['TXG', 'AAPL']
    assert trading.get.call_args_list[0][0][0] == '/assets'
