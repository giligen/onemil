"""The 2x-wrapper universe rule (2026-09-05, owner GO).

Pins:
  1. AlpacaClient._is_common_stock keeps leveraged/inverse wrappers when
     exclude_leveraged=False and STILL drops warrants / units / preferred /
     rights; the default (bull-flag universe) is unchanged.
  2. get_all_tradeable_assets(exclude_leveraged=...) threads the flag.
  3. UniverseBuilder Step 9 (ORB/ignition seed pool for daily_bars) unions
     the wrappers inside the close band with the common-stock pool, and a
     wrapper-fetch failure logs ERROR and falls back to the common pool
     without aborting the build.
Evidence: research/orb_entered_inclusive/wrapper_rule/summary.csv.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from batch.universe_builder import BROAD_BARS_MAX_CLOSE, UniverseBuilder
from data_sources.alpaca_client import AlpacaClient
from data_sources.float_provider import FloatProvider
from persistence.database import Database

WRAPPERS = [
    ('AXTX', 'Tradr 2X Long AXTI Daily ETF'),
    ('LITZ', 'Tradr 2X Short LITE Daily ETF'),
    ('SNDG', 'Leverage Shares 2X Long SNDK Daily ETF'),
    ('TQQQ', 'ProShares UltraPro QQQ'),
    ('MSTU', 'T-Rex 2X Long MSTR Daily Target ETF'),   # symbol-list member
]
NON_COMMON = [
    ('USARW', 'USA Rare Earth Warrant'),
    ('CCCXU', 'Churchill Capital Corp X Units'),
    ('BAC.PRE', 'Bank of America Preferred'),
    ('XYZR', 'Something Rights'),
]


class TestIsCommonStockFlag:
    @pytest.mark.parametrize('symbol,name', WRAPPERS)
    def test_default_drops_wrappers(self, symbol, name):
        assert AlpacaClient._is_common_stock(symbol, name) is False

    @pytest.mark.parametrize('symbol,name', WRAPPERS)
    def test_flag_keeps_wrappers(self, symbol, name):
        assert AlpacaClient._is_common_stock(symbol, name, exclude_leveraged=False) is True

    @pytest.mark.parametrize('symbol,name', NON_COMMON)
    def test_flag_still_drops_warrants_units_preferred_rights(self, symbol, name):
        assert AlpacaClient._is_common_stock(symbol, name, exclude_leveraged=False) is False
        assert AlpacaClient._is_common_stock(symbol, name) is False

    def test_plain_common_stock_both_ways(self):
        assert AlpacaClient._is_common_stock('AAPL', 'Apple Inc. Common Stock') is True
        assert AlpacaClient._is_common_stock('AAPL', 'Apple Inc. Common Stock', exclude_leveraged=False) is True


class TestGetAllTradeableAssetsFlag:
    def _client_with_assets(self, monkeypatch):
        client = AlpacaClient.__new__(AlpacaClient)
        assets = []
        for sym, name in WRAPPERS + NON_COMMON + [('AAPL', 'Apple Inc. Common Stock')]:
            a = MagicMock()
            a.symbol, a.name, a.tradable = sym, name, True
            a.exchange = None
            a.marginable = True
            assets.append(a)
        client.trading_client = MagicMock()
        client._call_with_timeout = lambda fn, label: assets
        return client

    def test_default_excludes_wrappers(self, monkeypatch):
        c = self._client_with_assets(monkeypatch)
        syms = {a['symbol'] for a in c.get_all_tradeable_assets()}
        assert syms == {'AAPL'}

    def test_flag_includes_wrappers_only(self, monkeypatch):
        c = self._client_with_assets(monkeypatch)
        syms = {a['symbol'] for a in c.get_all_tradeable_assets(exclude_leveraged=False)}
        assert syms == {'AAPL'} | {s for s, _ in WRAPPERS}


@pytest.fixture
def builder():
    return UniverseBuilder(
        alpaca_client=MagicMock(spec=AlpacaClient),
        float_provider=MagicMock(spec=FloatProvider),
        db=MagicMock(spec=Database),
    )


class TestStep9WrapperPool:
    def test_wrappers_in_band_join_the_seed_pool(self, builder):
        common = {'KEEP': {'close': 10.0, 'volume': 750_000}}
        builder.alpaca.get_all_tradeable_assets.return_value = [
            {'symbol': 'KEEP'}, {'symbol': 'AXTX'}, {'symbol': 'TQQQ'}, {'symbol': 'PRICEY'}]
        builder.alpaca.get_daily_bars.return_value = {
            'AXTX': {'close': 12.5, 'volume': 900_000},
            'TQQQ': {'close': 0.5, 'volume': 5_000_000},            # below band
            'PRICEY': {'close': BROAD_BARS_MAX_CLOSE + 1, 'volume': 1}, # above band
        }
        pool = builder._orb_broad_symbols(common)
        assert pool == ['AXTX', 'KEEP']
        builder.alpaca.get_all_tradeable_assets.assert_called_once_with(exclude_leveraged=False)
        builder.alpaca.get_daily_bars.assert_called_once_with(['AXTX', 'PRICEY', 'TQQQ'])

    def test_no_new_wrappers_returns_common_pool(self, builder):
        common = {'KEEP': {'close': 10.0, 'volume': 1}}
        builder.alpaca.get_all_tradeable_assets.return_value = [{'symbol': 'KEEP'}]
        assert builder._orb_broad_symbols(common) == ['KEEP']
        builder.alpaca.get_daily_bars.assert_not_called()

    def test_wrapper_fetch_failure_logs_error_and_falls_back(self, builder, caplog):
        common = {'KEEP': {'close': 10.0, 'volume': 1}}
        builder.alpaca.get_all_tradeable_assets.side_effect = RuntimeError('api down')
        with caplog.at_level(logging.ERROR):
            pool = builder._orb_broad_symbols(common)
        assert pool == ['KEEP']
        assert any('wrapper pool fetch FAILED' in r.message for r in caplog.records)

    def test_build_passes_wrappers_to_step9(self, builder):
        """End to end through build(): the broad refresh call carries the wrapper."""
        daily = {'KEEP': {'close': 10.0, 'volume': 750_000}}
        builder.alpaca.get_all_tradeable_assets.side_effect = lambda exclude_leveraged=True: (
            [{'symbol': 'KEEP', 'company_name': '', 'exchange': '', 'asset_class': 'us_equity'}]
            if exclude_leveraged else [{'symbol': 'KEEP'}, {'symbol': 'AXTX'}])
        builder.alpaca.get_daily_bars.side_effect = lambda syms: (
            daily if 'KEEP' in syms else {'AXTX': {'close': 12.5, 'volume': 900_000}})
        builder.alpaca.get_daily_bars_range.return_value = {}
        builder.db.get_active_universe.return_value = []
        builder.db.get_symbols_needing_float_update.return_value = []
        builder.db.get_volume_profile_count.return_value = 0
        builder._cache_volume_profiles = MagicMock(return_value=None)
        builder.build()
        broad_calls = [c for c in builder.alpaca.get_daily_bars_range.call_args_list
                       if c.args and isinstance(c.args[0], list) and 'SPY' not in c.args[0]]
        assert broad_calls, "Step 9 never called get_daily_bars_range"
        assert set(broad_calls[0].args[0]) == {'AXTX', 'KEEP'}
