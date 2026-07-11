"""Asset-class rule for the ORB news gate (2026-07-11 deliberate-rules
mandate): the news boost requires POSITIVE identification as a common
stock. Wrappers have no company events (their newsy days are the
crowding cell — negative all 3 eras); unknown never boosts blind.
"""
from __future__ import annotations

import pytest

from trading.orb_asset_class import (
    STOCK, UNKNOWN, WRAPPER, classify_asset, effective_has_news,
    load_class_map,
)


class TestClassifyAsset:
    @pytest.mark.parametrize('sym,name', [
        ('BEZ', 'Tradr 2X Short BE Daily ETF'),
        ('PLTZ', 'Tidal Trust II Defiance Daily Target 2x Short PLTR ETF'),
        ('CRCA', 'ProShares Ultra CRCL'),
        ('MSTZ', 'T-Rex 2X Inverse MSTR Daily Target ETF'),
        ('TSDD', 'GraniteShares 2x Short TSLA Daily ETF'),
        ('KOLD', 'ProShares UltraShort Bloomberg Natural Gas'),
        ('PLTU', 'Direxion Daily PLTR Bull 2X ETF'),
        ('SOLT', 'Volatility Shares Trust 2x Solana ETF'),
        ('FNGD', 'MicroSectors FANG+ Index -3X Inverse Leveraged ETN'),
    ])
    def test_wrappers_by_name(self, sym, name):
        assert classify_asset(sym, name) == WRAPPER

    @pytest.mark.parametrize('sym,name', [
        ('AMCI', 'AMC Robotics Corporation Common Stock'),
        ('ANNA', 'AleAnna, Inc. Class A Common Stock'),
        ('SLDP', 'Solid Power, Inc. Class A Common Stock'),
        ('CCXI', 'Churchill Capital Corp XI Class A Ordinary Shares'),
    ])
    def test_stocks_by_name(self, sym, name):
        assert classify_asset(sym, name) == STOCK

    def test_missing_name_is_unknown(self):
        assert classify_asset('ZZZZ', None) == UNKNOWN
        assert classify_asset('ZZZZ', '') == UNKNOWN
        assert classify_asset('ZZZZ', '   ') == UNKNOWN

    def test_lev_family_fast_path_needs_no_name(self):
        """orb_correlation's curated leveraged sets classify without I/O
        or a name — covers the MSTU/MSTZ complex on day one."""
        from trading.orb_correlation import LEVERAGED_LONG_ALL
        sym = LEVERAGED_LONG_ALL[0]
        assert classify_asset(sym, None) == WRAPPER

    def test_x_multiple_variants(self):
        assert classify_asset('XX', 'Foo 3X Leveraged Something') == WRAPPER
        assert classify_asset('XX', 'Foo 1.5X Daily Thing') == WRAPPER
        # 'X' inside a word must not trigger
        assert classify_asset('XX', 'Xerox Holdings Corporation') == STOCK


class TestEffectiveHasNews:
    def test_stock_news_passes(self):
        assert effective_has_news(True, STOCK) is True

    def test_wrapper_news_blocked(self):
        assert effective_has_news(True, WRAPPER) is False

    def test_unknown_news_blocked_never_boost_blind(self):
        assert effective_has_news(True, UNKNOWN) is False

    def test_no_news_passthrough_any_class(self):
        for cls in (STOCK, WRAPPER, UNKNOWN):
            assert effective_has_news(False, cls) is False

    def test_tristate_none_passthrough(self):
        """Fetch-failed stays None (fail-open handled downstream)."""
        for cls in (STOCK, WRAPPER, UNKNOWN):
            assert effective_has_news(None, cls) is None


class TestClassMap:
    def test_ships_with_repo_and_covers_the_week(self):
        m = load_class_map()
        assert m.get('BEZ') == 'wrapper'
        assert m.get('CRCA') == 'wrapper'
        assert m.get('ANNA') == 'stock'
        # verified manual overrides for delisted stocks (BT parity)
        assert m.get('KZR') == 'stock'
        assert m.get('UGRO') == 'stock'
        assert len(m) > 30_000

    def test_missing_file_returns_empty_with_warning(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            assert load_class_map('/nonexistent/map.csv') == {}
        assert any('class map' in r.message for r in caplog.records)
