"""Tests for the point-in-time listings helper (research/scripts/pit_listings.py).

Unit tests run on a synthetic parquet directory (no data dependency). The
integration tests run on the REAL bought file and skip if it is absent, so the
suite stays green on a node that has not pulled it.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.scripts.pit_listings import DEFAULT_DIR, PitListings, is_test_ticker  # noqa: E402


# ---------------------------------------------------------------- unit: names

@pytest.mark.parametrize('sym', ['ZVZZT', 'ZAZZT', 'ZBZZT', 'ZCZZT', 'ZJZZT', 'ZWZZT'])
def test_test_tickers_detected(sym):
    """The six NASDAQ test symbols — one of them WAS the whole F6 TEST profit."""
    assert is_test_ticker(sym) is True


@pytest.mark.parametrize('sym', ['AAPL', 'ZM', 'ZS', 'ZIM', 'ZTS', '', 'ZZZ'])
def test_real_symbols_not_flagged(sym):
    assert is_test_ticker(sym) is False


# ----------------------------------------------------- unit: synthetic dir

@pytest.fixture
def fake_dir(tmp_path):
    """Two months: SIEB delists after 2025-01, ZVZZT present in both."""
    jan = pd.DataFrame({
        'raw_symbol': ['AAPL', 'SIEB', 'ZVZZT', 'SPY'],
        'security_type': ['C', 'C', 'C', 'Q'],
        'exchange': ['XNAS', 'XNAS', 'XNAS', 'ARCX'],
    })
    feb = pd.DataFrame({
        'raw_symbol': ['AAPL', 'ZVZZT', 'SPY'],
        'security_type': ['C', 'C', 'Q'],
        'exchange': ['XNAS', 'XNAS', 'ARCX'],
    })
    jan.to_parquet(tmp_path / 'def_202501.parquet')
    feb.to_parquet(tmp_path / 'def_202502.parquet')
    return tmp_path


def test_listed_symbols_excludes_test_tickers(fake_dir):
    pit = PitListings(fake_dir)
    assert pit.listed_symbols('2025-01-15') == frozenset({'AAPL', 'SIEB', 'SPY'})


def test_delisted_name_is_listed_before_and_not_after(fake_dir):
    """The survivorship guarantee: SIEB existed in January, not in February."""
    pit = PitListings(fake_dir)
    assert pit.was_listed('SIEB', '2025-01-15') is True
    assert pit.was_listed('SIEB', '2025-02-15') is False


def test_common_stock_filter(fake_dir):
    pit = PitListings(fake_dir)
    assert pit.common_stock_symbols('2025-01-15') == frozenset({'AAPL', 'SIEB'})


def test_listing_exchange_and_security_type(fake_dir):
    pit = PitListings(fake_dir)
    assert pit.listing_exchange('SPY', '2025-01-15') == 'ARCX'
    assert pit.security_type('AAPL', '2025-01-15') == 'C'
    assert pit.listing_exchange('NOPE', '2025-01-15') is None


def test_screen_reports_every_drop_with_a_reason(fake_dir):
    pit = PitListings(fake_dir)
    kept, dropped = pit.screen(['AAPL', 'ZVZZT', 'SIEB', 'NOPE'], '2025-02-15')
    assert kept == frozenset({'AAPL'})
    assert dropped == {'ZVZZT': 'test_ticker',
                       'SIEB': 'not_listed_on_date',
                       'NOPE': 'not_listed_on_date'}


def test_date_outside_window_raises_rather_than_guessing(fake_dir):
    """A silent fallback to today's symbol list IS the survivorship bug."""
    pit = PitListings(fake_dir)
    with pytest.raises(KeyError, match='outside the bought point-in-time window'):
        pit.listed_symbols('2024-06-15')


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PitListings(tmp_path / 'nope')


def test_empty_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PitListings(tmp_path)


# ------------------------------------------------- integration: the real file

real = pytest.mark.skipif(not DEFAULT_DIR.is_dir(),
                          reason='point-in-time listings not pulled on this node')


@real
def test_real_file_flags_the_zvzzt_family():
    pit = PitListings()
    assert 'ZVZZT' not in pit.listed_symbols('2026-09-01')


@real
def test_real_file_carries_every_major_listing_venue():
    """Venue coverage is what caps a primary-venue feed study (see O_halt)."""
    pit = PitListings()
    venues = {pit.listing_exchange(s, '2026-09-01')
              for s in ('AAPL', 'BAC', 'SPY')}
    assert 'XNAS' in venues and 'XNYS' in venues


@real
def test_real_coverage_bounds_are_honest():
    pit = PitListings()
    first, last = pit.coverage
    assert first == '202407'
    assert last >= '202609'
