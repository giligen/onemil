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
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from batch.universe_builder import BROAD_BARS_MAX_CLOSE, UniverseBuilder
from data_sources.alpaca_client import AlpacaClient
from data_sources.float_provider import FloatProvider
from persistence.database import Database
from trading.bf_universe_filter import is_bf_eligible
from trading.pattern_detector import BullFlagPattern
from trading.trade_planner import TradePlan
from trading.trading_engine import TradingEngine

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


def _bf_pattern(symbol="TEST"):
    return BullFlagPattern(
        symbol=symbol,
        pole_start_idx=0, pole_end_idx=2,
        flag_start_idx=3, flag_end_idx=4,
        pole_low=4.00, pole_high=4.50,
        pole_height=0.50, pole_gain_pct=12.5,
        flag_low=4.30, flag_high=4.40,
        retracement_pct=40.0, pullback_candle_count=2,
        avg_pole_volume=180000, avg_flag_volume=40000,
        breakout_level=4.40,
    )


def _bf_plan(symbol="TEST"):
    return TradePlan(
        symbol=symbol,
        entry_price=4.40,
        stop_loss_price=4.25,
        take_profit_price=4.90,
        risk_per_share=0.15,
        reward_per_share=0.50,
        risk_reward_ratio=3.3,
        shares=113,
        total_risk=16.95,
        pattern=_bf_pattern(symbol),
    )


class TestLiveBFWrapperGuard:
    """2026-09-24 parity defect (docs/bf_wrapper_guard_spec_20260924.md):
    live BF bought RGTZ ('Tidal Trust II Defiance Daily Target 2x Short
    RGTI ETF') on 9/23 and lost 4R because `_qualified_symbols` had no
    name guard, even though BT Stage-2 always excluded it by name. A
    qualified wrapper must never reach the pattern check or order
    submission; a common stock must (regression + BT/live parity)."""

    RGTZ_NAME = 'Tidal Trust II Defiance Daily Target 2x Short RGTI ETF'
    JAGX_NAME = 'Jaguar Health, Inc. Common Stock'

    def _bars(self):
        return pd.DataFrame({'open': [4.0], 'high': [4.1], 'low': [3.9],
                              'close': [4.05], 'volume': [100000]})

    def test_wrapper_never_qualifies_or_reaches_order_submission(self, bf_engine, bf_mock_alpaca):
        bf_mock_alpaca.get_asset_name.return_value = self.RGTZ_NAME
        bf_engine.on_stock_qualified('RGTZ')
        assert 'RGTZ' not in bf_engine._qualified_symbols

        result = bf_engine.run_pattern_check()
        assert result is None
        bf_engine.executor.submit_buy_stop_bracket_order.assert_not_called()

    def test_failed_live_name_lookup_falls_back_to_the_bt_offline_dump(self, bf_engine, bf_mock_alpaca):
        """Live lookup returns None (timeout/API error): the guard must use the BT's own offline name dumps
        before the legacy symbol list — RGTZ is in the dump (not the legacy list) and must still be excluded."""
        bf_mock_alpaca.get_asset_name.return_value = None
        bf_engine.on_stock_qualified('RGTZ')
        assert 'RGTZ' not in bf_engine._qualified_symbols

    def test_common_stock_qualifies_and_reaches_order_submission(self, bf_engine, bf_mock_alpaca):
        """Positive control: proves this harness would have caught 9/23 RGTZ."""
        # Seed the universe row — run_pattern_check's volume filter blocks
        # any symbol with no universe/volume data (line ~3470).
        bf_engine.db.upsert_universe_stock({
            'symbol': 'JAGX', 'company_name': 'Jaguar Health', 'exchange': 'NASDAQ',
            'sector': '', 'country': 'US', 'price_close': 10.0,
            'float_shares': 5_000_000, 'float_updated_at': None,
            'avg_volume_daily': 500_000, 'last_updated': datetime.now(timezone.utc),
            'active': 1,
        })
        bf_mock_alpaca.get_asset_name.return_value = self.JAGX_NAME
        bf_mock_alpaca.get_1min_bars.return_value = self._bars()
        bf_engine.detector.detect_setup.return_value = _bf_pattern('JAGX')
        bf_engine.planner.create_plan.return_value = _bf_plan('JAGX')
        bf_engine.position_manager.can_open_position.return_value = True
        bf_engine.executor.submit_buy_stop_bracket_order.return_value = {
            'order_id': 'order-1', 'status': 'accepted', 'symbol': 'JAGX', 'shares': 100,
        }

        bf_engine.on_stock_qualified('JAGX')
        assert 'JAGX' in bf_engine._qualified_symbols

        with patch.object(TradingEngine, '_is_past_last_entry_time', return_value=False):
            result = bf_engine.run_pattern_check()
        assert result is not None
        bf_engine.executor.submit_buy_stop_bracket_order.assert_called_once()

    def test_excluded_symbol_logged_once_and_name_cached(self, bf_engine, bf_mock_alpaca, caplog):
        bf_mock_alpaca.get_asset_name.return_value = self.RGTZ_NAME
        with caplog.at_level(logging.INFO):
            bf_engine.on_stock_qualified('RGTZ')
            bf_engine.on_stock_qualified('RGTZ')
        infos = [r for r in caplog.records if r.levelname == 'INFO'
                 and 'BF UNIVERSE' in r.message and 'RGTZ' in r.message]
        assert len(infos) == 1
        assert 'leveraged/inverse wrapper' in infos[0].message
        bf_mock_alpaca.get_asset_name.assert_called_once_with('RGTZ')  # cached, not refetched

    @pytest.mark.parametrize('symbol,name,expected_eligible', [
        ('RGTZ', RGTZ_NAME, False),
        ('JAGX', JAGX_NAME, True),
        ('AAPL', 'Apple Inc. Common Stock', True),
        ('TQQQ', 'ProShares UltraPro QQQ', False),
        ('MSTU', 'T-Rex 2X Long MSTR Daily Target ETF', False),
    ])
    def test_bt_and_live_guard_agree(self, bf_engine, bf_mock_alpaca, symbol, name, expected_eligible):
        """Parity: BT's predicate (is_bf_eligible, what filter_trades calls)
        and the live guard (TradingEngine._bf_wrapper_excluded) are the SAME
        function by construction. Drive both call sites on a fixed
        (symbol, name) list and check they agree."""
        bt_verdict = is_bf_eligible(symbol, {symbol: name})
        bf_mock_alpaca.get_asset_name.return_value = name
        live_verdict = not bf_engine._bf_wrapper_excluded(symbol)
        assert bt_verdict == expected_eligible
        assert live_verdict == expected_eligible
        assert bt_verdict == live_verdict
