"""
Integration tests for the OneMil scanner system.

Tests components working TOGETHER with REAL instances (Database, ScannerCriteria,
UniverseBuilder, RealtimeScanner). External APIs (Alpaca, Yahoo Finance) are mocked.

Each test uses a fresh temporary SQLite database via tmp_path fixture.
"""

import os
from datetime import datetime, timezone, timedelta, date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from batch.universe_builder import UniverseBuilder
from data_sources.alpaca_client import AlpacaClient
from data_sources.float_provider import FloatProvider
from data_sources.news_provider import NewsProvider, NewsAnalyzer
from persistence.database import Database
from scanner.criteria import ScannerCriteria, ScanCandidate
from scanner.realtime_scanner import RealtimeScanner


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def db(tmp_path):
    """Create a real Database backed by a temp SQLite file."""
    db_path = str(tmp_path / "test_onemil.db")
    database = Database(db_path=db_path)
    yield database
    database.close()


@pytest.fixture
def mock_alpaca():
    """Create an AlpacaClient mock with spec to catch interface violations."""
    return MagicMock(spec=AlpacaClient)


@pytest.fixture
def mock_float_provider():
    """Create a FloatProvider mock with spec."""
    return MagicMock(spec=FloatProvider)


@pytest.fixture
def mock_news_provider():
    """Create a NewsProvider mock with spec."""
    return MagicMock(spec=NewsProvider)


@pytest.fixture
def criteria():
    """Create a real ScannerCriteria with default thresholds."""
    return ScannerCriteria()


def _seed_universe(db, stocks):
    """
    Helper to insert stock dicts into the universe table.

    Args:
        db: Real Database instance
        stocks: List of minimal stock dicts (symbol required, others defaulted)
    """
    now = datetime.now(timezone.utc)
    for s in stocks:
        db.upsert_universe_stock({
            'symbol': s['symbol'],
            'company_name': s.get('company_name', f"{s['symbol']} Inc"),
            'exchange': s.get('exchange', 'NASDAQ'),
            'sector': s.get('sector', None),
            'country': s.get('country', None),
            'price_close': s.get('price_close', 5.0),
            'float_shares': s.get('float_shares', 3_000_000),
            'float_updated_at': s.get('float_updated_at', now),
            'avg_volume_daily': s.get('avg_volume_daily', 500_000),
            'last_updated': now,
            'active': s.get('active', 1),
        })


# =============================================================================
# 1. Universe build -> scan candidate flow
# =============================================================================


class TestUniverseBuildToScanFlow:
    """Integration: Alpaca assets + bars -> DB -> ScannerCriteria evaluation."""

    def test_universe_build_to_scan_flow(self, db, mock_alpaca, mock_float_provider):
        """
        Build universe from mocked Alpaca data, verify DB persistence,
        then create ScanCandidates from DB rows and evaluate with real criteria.
        """
        # --- Arrange: mock Alpaca to return fake assets + bars ---
        fake_assets = [
            {'symbol': 'AAA', 'company_name': 'Alpha Corp', 'exchange': 'NASDAQ'},
            {'symbol': 'BBB', 'company_name': 'Beta Corp', 'exchange': 'NYSE'},
            {'symbol': 'CCC', 'company_name': 'Gamma Corp', 'exchange': 'NASDAQ'},
            {'symbol': 'ZZZ', 'company_name': 'Expensive Corp', 'exchange': 'NYSE'},
        ]
        mock_alpaca.get_all_tradeable_assets.return_value = fake_assets

        fake_bars = {
            'AAA': {'close': 5.50, 'volume': 1_200_000, 'timestamp': datetime.now(timezone.utc)},
            'BBB': {'close': 8.00, 'volume': 800_000, 'timestamp': datetime.now(timezone.utc)},
            'CCC': {'close': 15.00, 'volume': 2_000_000, 'timestamp': datetime.now(timezone.utc)},
            'ZZZ': {'close': 55.00, 'volume': 500_000, 'timestamp': datetime.now(timezone.utc)},
        }
        mock_alpaca.get_daily_bars.return_value = fake_bars

        # Intraday bars for volume profile - realistic 15-min data
        intraday_records = []
        base_date = datetime(2026, 3, 10, tzinfo=timezone.utc)
        for day_offset in range(5):
            day = base_date + timedelta(days=day_offset)
            for h in range(9, 16):
                for m in (0, 15, 30, 45):
                    if (h == 9 and m < 30) or (h >= 16):
                        continue
                    ts = day.replace(hour=h, minute=m, second=0)
                    intraday_records.append({
                        'timestamp': ts,
                        'open': 5.0, 'high': 5.5, 'low': 4.9,
                        'close': 5.2, 'volume': 10_000 + h * 1_000,
                    })
        mock_alpaca.get_intraday_bars.return_value = pd.DataFrame(intraday_records)

        # Float provider returns float + sector/country in single pass
        mock_float_provider.get_stock_info_batch.return_value = {
            'AAA': {'float_shares': 2_000_000, 'sector': 'Technology', 'country': 'US'},
            'BBB': {'float_shares': 5_000_000, 'sector': 'Healthcare', 'country': 'US'},
            'CCC': {'float_shares': 50_000_000, 'sector': 'Finance', 'country': 'US'},
        }

        # --- Act ---
        builder = UniverseBuilder(
            alpaca_client=mock_alpaca,
            float_provider=mock_float_provider,
            db=db,
            price_min=2.0,
            price_max=20.0,
            float_max=10_000_000,
        )
        result = builder.build()

        # --- Assert: DB has the right stocks active ---
        active = db.get_active_universe()
        active_symbols = {s['symbol'] for s in active}

        # ZZZ ($55) excluded by price, CCC (50M float) excluded by float
        assert 'AAA' in active_symbols, "AAA should be active (price $5.50, float 2M)"
        assert 'BBB' in active_symbols, "BBB should be active (price $8.00, float 5M)"
        assert 'CCC' not in active_symbols, "CCC should be excluded (float 50M > 10M)"
        assert 'ZZZ' not in active_symbols, "ZZZ should be excluded (price $55 > $20)"
        assert result['total_stocks'] == 2

        # Verify volume profiles were cached
        profile_aaa = db.get_volume_profile('AAA')
        assert len(profile_aaa) > 0, "AAA should have volume profile buckets"
        assert '09:30' in profile_aaa, "09:30 bucket should be present"
        assert profile_aaa['09:30'] > 0, "09:30 avg_volume should be positive"

        # --- Now evaluate candidates from DB data with real criteria ---
        criteria = ScannerCriteria()
        for stock in active:
            candidate = ScanCandidate(
                symbol=stock['symbol'],
                company_name=stock['company_name'],
                prev_close=stock['price_close'],
                current_price=stock['price_close'] * 1.05,  # Simulate 5% gap
                float_shares=stock['float_shares'],
                gap_pct=5.0,
            )
            qualified = criteria.evaluate_premarket(candidate)
            assert qualified is True, (
                f"{stock['symbol']} should qualify premarket with 5% gap"
            )


# =============================================================================
# 2. Volume profile calculation and storage
# =============================================================================


class TestVolumeProfileCalculationAndStorage:
    """Integration: intraday bars -> _calculate_volume_profile -> DB -> retrieval."""

    def test_volume_profile_calculation_and_storage(self, db, mock_alpaca, mock_float_provider):
        """
        Mock realistic 15-min intraday bars, run _calculate_volume_profile,
        store in real DB, retrieve and verify bucket values.
        """
        # Seed universe so foreign-key-like expectations are met
        _seed_universe(db, [{'symbol': 'VTEST', 'price_close': 7.0, 'float_shares': 1_000_000}])

        # Build realistic intraday data: 10 trading days, varying volume by bucket
        records = []
        base_date = datetime(2026, 2, 20, tzinfo=timezone.utc)
        for day_offset in range(10):
            day = base_date + timedelta(days=day_offset)
            if day.weekday() >= 5:
                continue
            for h in range(9, 16):
                for m in (0, 15, 30, 45):
                    if (h == 9 and m < 30) or h >= 16:
                        continue
                    ts = day.replace(hour=h, minute=m, second=0)
                    # Morning volume higher than afternoon
                    volume = 50_000 if h < 12 else 20_000
                    records.append({
                        'timestamp': ts,
                        'open': 7.0, 'high': 7.2, 'low': 6.9,
                        'close': 7.1, 'volume': volume,
                    })

        mock_alpaca.get_intraday_bars.return_value = pd.DataFrame(records)

        builder = UniverseBuilder(
            alpaca_client=mock_alpaca,
            float_provider=mock_float_provider,
            db=db,
        )

        # --- Act: calculate and store ---
        profiles = builder._calculate_volume_profile('VTEST')
        assert len(profiles) > 0, "Should produce profile buckets"
        db.upsert_volume_profiles(profiles)

        # --- Assert: retrieve from DB ---
        stored = db.get_volume_profile('VTEST')
        assert '09:30' in stored, "09:30 bucket should be stored"
        assert '14:00' in stored, "14:00 bucket should be stored"

        # Morning volume should be higher than afternoon
        assert stored['09:30'] > stored['14:00'], (
            f"Morning volume ({stored['09:30']}) should exceed "
            f"afternoon volume ({stored['14:00']})"
        )

        # Verify all bucket values are positive integers
        for bucket, vol in stored.items():
            assert isinstance(vol, int), f"Volume for {bucket} should be int"
            assert vol > 0, f"Volume for {bucket} should be positive"


# =============================================================================
# 3. Pre-market scan saves to DB
# =============================================================================


class TestPremarketScanSavesToDb:
    """Integration: universe in DB -> mocked trades -> premarket gap scan -> DB results.

    Premarket is pure quantitative (gap detection). No news/LLM calls.
    """

    def test_premarket_scan_saves_to_db(self, db, mock_alpaca, criteria):
        """
        Set up universe in real DB, mock Alpaca latest trades for gap-up prices,
        run one premarket cycle, verify gap-up scan_results saved in DB.
        """
        # Seed universe with two stocks
        _seed_universe(db, [
            {'symbol': 'GAP1', 'price_close': 5.00, 'float_shares': 2_000_000},
            {'symbol': 'GAP2', 'price_close': 10.00, 'float_shares': 4_000_000},
            {'symbol': 'FLAT', 'price_close': 8.00, 'float_shares': 3_000_000},
        ])

        # Mock latest trades: GAP1 up 10%, GAP2 up 5%, FLAT unchanged
        mock_alpaca.get_latest_trades.return_value = {
            'GAP1': {'price': 5.50, 'size': 100, 'timestamp': None},   # +10%
            'GAP2': {'price': 10.50, 'size': 200, 'timestamp': None},  # +5%
            'FLAT': {'price': 8.00, 'size': 50, 'timestamp': None},    # 0%
        }

        news_provider = NewsProvider(alpaca_client=mock_alpaca)

        scanner = RealtimeScanner(
            alpaca_client=mock_alpaca,
            news_provider=news_provider,
            db=db,
            criteria=criteria,
        )
        scanner._load_universe()

        # --- Act ---
        scanner._run_premarket_cycle()

        # --- Assert ---
        today = date.today().isoformat()
        results = db.get_scan_results(today, phase='premarket')
        result_symbols = {r['symbol'] for r in results}

        assert 'GAP1' in result_symbols, "GAP1 (10% gap) should be saved"
        assert 'GAP2' in result_symbols, "GAP2 (5% gap) should be saved"
        assert 'FLAT' not in result_symbols, "FLAT (0% gap) should NOT be saved"

        # Verify data integrity — no news in premarket phase
        gap1_result = next(r for r in results if r['symbol'] == 'GAP1')
        assert gap1_result['phase'] == 'premarket'
        assert gap1_result['prev_close'] == 5.00
        assert gap1_result['current_price'] == 5.50
        assert abs(gap1_result['gap_pct'] - 10.0) < 0.1
        assert gap1_result['has_news'] == 0, "Premarket does not check news"
        assert gap1_result['qualified'] == 1


# =============================================================================
# 4. Intraday scan qualification flow
# =============================================================================


class TestIntradayScanQualificationFlow:
    """Integration: universe + volume profiles in DB -> intraday cycle -> qualified saves."""

    def test_intraday_scan_qualification_flow(self, db, mock_alpaca, criteria):
        """
        Set up universe and volume profiles in real DB, mock current bars
        (high volume) and trades (high price), mock news, run one intraday cycle,
        verify qualified stocks saved to DB.
        """
        # Seed universe
        _seed_universe(db, [
            {'symbol': 'HOT', 'price_close': 5.00, 'float_shares': 2_000_000},
            {'symbol': 'WARM', 'price_close': 8.00, 'float_shares': 3_000_000},
            {'symbol': 'COLD', 'price_close': 6.00, 'float_shares': 4_000_000},
        ])

        # Set up volume profiles: avg_volume per bucket
        now = datetime.now(timezone.utc)
        db.upsert_volume_profiles([
            {'symbol': 'HOT', 'time_bucket': '10:00', 'avg_volume': 10_000, 'last_updated': now},
            {'symbol': 'WARM', 'time_bucket': '10:00', 'avg_volume': 10_000, 'last_updated': now},
            {'symbol': 'COLD', 'time_bucket': '10:00', 'avg_volume': 10_000, 'last_updated': now},
        ])

        # Mock current bars: HOT has 60K volume (6x), WARM 30K (3x), COLD 5K (0.5x)
        mock_alpaca.get_current_bars.return_value = {
            'HOT':  {'open': 5.5, 'high': 6.0, 'low': 5.4, 'close': 5.8, 'volume': 60_000, 'timestamp': now},
            'WARM': {'open': 8.5, 'high': 9.0, 'low': 8.4, 'close': 8.8, 'volume': 30_000, 'timestamp': now},
            'COLD': {'open': 6.0, 'high': 6.1, 'low': 5.9, 'close': 6.0, 'volume': 5_000, 'timestamp': now},
        }

        # Mock latest trades: HOT up 20%, WARM up 12%, COLD up 1%
        mock_alpaca.get_latest_trades.return_value = {
            'HOT':  {'price': 6.00, 'size': 100, 'timestamp': None},  # +20%
            'WARM': {'price': 8.96, 'size': 100, 'timestamp': None},  # +12%
            'COLD': {'price': 6.06, 'size': 100, 'timestamp': None},  # +1%
        }

        # Mock news: HOT and WARM have news
        mock_alpaca.get_news.return_value = [
            {'headline': 'Breaking catalyst', 'summary': 'Details',
             'source': 'Reuters', 'created_at': '2026-03-13T10:00:00Z', 'url': 'http://example.com'}
        ]

        news_provider = NewsProvider(alpaca_client=mock_alpaca)

        scanner = RealtimeScanner(
            alpaca_client=mock_alpaca,
            news_provider=news_provider,
            db=db,
            criteria=criteria,
        )
        scanner._load_universe()

        # Patch ET time so bucket = "10:00"
        with patch('scanner.realtime_scanner.datetime') as mock_dt:
            import pytz
            ET = pytz.timezone('US/Eastern')
            fake_now = ET.localize(datetime(2026, 3, 13, 10, 0, 0))
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            scanner._run_intraday_cycle()

        # --- Assert ---
        today = date.today().isoformat()
        results = db.get_scan_results(today, phase='intraday')
        result_symbols = {r['symbol'] for r in results}

        # HOT: 20% move, 6x relVol, news => QUALIFIED
        assert 'HOT' in result_symbols, "HOT should qualify (20% move, 6x vol, news)"

        # WARM: 12% move but only 3x relVol => NOT qualified (relVol < 5x)
        assert 'WARM' not in result_symbols, "WARM should NOT qualify (relVol 3x < 5x)"

        # COLD: 1% move, 0.5x vol => NOT qualified
        assert 'COLD' not in result_symbols, "COLD should NOT qualify (1% move, 0.5x vol)"

        # Verify HOT result data integrity
        hot_result = next(r for r in results if r['symbol'] == 'HOT')
        assert hot_result['current_price'] == 6.00
        assert hot_result['relative_volume'] == pytest.approx(6.0, abs=0.1)
        assert hot_result['time_bucket'] == '10:00'
        assert hot_result['qualified'] == 1


# =============================================================================
# 5. Float cache freshness skip
# =============================================================================


class TestFloatCacheSkipFresh:
    """Integration: DB float_updated_at -> get_symbols_needing_float_update logic."""

    def test_float_cache_skip_fresh(self, db):
        """
        Insert stock with recent float_updated_at and verify it's skipped.
        Insert stock with old float_updated_at and verify it's included.
        """
        now = datetime.now(timezone.utc)
        fresh_date = now - timedelta(days=2)   # 2 days old: within 7-day window
        stale_date = now - timedelta(days=10)  # 10 days old: outside 7-day window

        _seed_universe(db, [
            {'symbol': 'FRESH', 'float_shares': 3_000_000, 'float_updated_at': fresh_date},
            {'symbol': 'STALE', 'float_shares': 4_000_000, 'float_updated_at': stale_date},
            {'symbol': 'NOFLOAT', 'float_shares': None, 'float_updated_at': None},
        ])

        # --- Act ---
        needing_update = db.get_symbols_needing_float_update(max_age_days=7)

        # --- Assert ---
        assert 'FRESH' not in needing_update, "FRESH (2 days old) should be skipped"
        assert 'STALE' in needing_update, "STALE (10 days old) should need update"
        assert 'NOFLOAT' in needing_update, "NOFLOAT (no float_updated_at) should need update"

    def test_float_cache_boundary(self, db):
        """
        Verify that a stock updated exactly at the boundary is handled correctly.
        A stock updated 7 days ago should be considered stale.
        """
        now = datetime.now(timezone.utc)
        boundary_date = now - timedelta(days=7, seconds=1)  # Just past the boundary

        _seed_universe(db, [
            {'symbol': 'BOUNDARY', 'float_shares': 2_000_000, 'float_updated_at': boundary_date},
        ])

        needing_update = db.get_symbols_needing_float_update(max_age_days=7)
        assert 'BOUNDARY' in needing_update, "Stock at boundary should need update"

    def test_null_float_with_recent_check_skipped(self, db):
        """
        Stocks where float is None but float_updated_at is recent should be skipped.

        This prevents re-fetching ETFs/CEFs that will never have float data.
        """
        now = datetime.now(timezone.utc)
        recent = now - timedelta(days=2)

        _seed_universe(db, [
            # Checked 2 days ago, no float available — should be SKIPPED
            {'symbol': 'NOFLOAT_RECENT', 'float_shares': None, 'float_updated_at': recent},
            # Never checked — should need update
            {'symbol': 'NOFLOAT_NEVER', 'float_shares': None, 'float_updated_at': None},
        ])

        needing_update = db.get_symbols_needing_float_update(max_age_days=7)
        assert 'NOFLOAT_RECENT' not in needing_update, \
            "Recently checked null-float should be skipped"
        assert 'NOFLOAT_NEVER' in needing_update, \
            "Never-checked null-float should need update"


# =============================================================================
# 6. Deactivation on rebuild
# =============================================================================


class TestDeactivationOnRebuild:
    """Integration: build -> rebuild with fewer stocks -> verify deactivation."""

    def test_deactivation_on_rebuild(self, db, mock_alpaca, mock_float_provider):
        """
        Build universe with stocks A, B, C. Rebuild where C's float has
        increased beyond the threshold (e.g., secondary offering dilution).
        Verify C is deactivated in the DB while A and B remain active.
        """
        # --- First build: A, B, C all qualify ---
        fake_assets = [
            {'symbol': 'AAAA', 'company_name': 'Alpha', 'exchange': 'NASDAQ'},
            {'symbol': 'BBBB', 'company_name': 'Beta', 'exchange': 'NYSE'},
            {'symbol': 'CCCC', 'company_name': 'Gamma', 'exchange': 'NASDAQ'},
        ]
        mock_alpaca.get_all_tradeable_assets.return_value = fake_assets
        mock_alpaca.get_daily_bars.return_value = {
            'AAAA': {'close': 5.0, 'volume': 500_000, 'timestamp': datetime.now(timezone.utc)},
            'BBBB': {'close': 8.0, 'volume': 600_000, 'timestamp': datetime.now(timezone.utc)},
            'CCCC': {'close': 12.0, 'volume': 700_000, 'timestamp': datetime.now(timezone.utc)},
        }
        mock_alpaca.get_intraday_bars.return_value = pd.DataFrame()  # Skip vol profiles

        mock_float_provider.get_stock_info_batch.return_value = {
            'AAAA': {'float_shares': 2_000_000, 'sector': 'Tech', 'country': 'US'},
            'BBBB': {'float_shares': 3_000_000, 'sector': 'Tech', 'country': 'US'},
            'CCCC': {'float_shares': 4_000_000, 'sector': 'Tech', 'country': 'US'},
        }

        builder = UniverseBuilder(
            alpaca_client=mock_alpaca,
            float_provider=mock_float_provider,
            db=db,
        )
        result_v1 = builder.build()
        assert result_v1['total_stocks'] == 3

        active_v1 = {s['symbol'] for s in db.get_active_universe()}
        assert active_v1 == {'AAAA', 'BBBB', 'CCCC'}

        # --- Second build: C's float diluted to 20M (above 10M threshold) ---
        # Force float_updated_at to be stale so the builder re-fetches float
        stale_date = datetime.now(timezone.utc) - timedelta(days=10)
        db.conn.execute(
            "UPDATE universe SET float_updated_at = ? WHERE symbol = 'CCCC'",
            (stale_date,)
        )
        db.conn.commit()

        mock_alpaca.get_all_tradeable_assets.return_value = fake_assets
        mock_alpaca.get_daily_bars.return_value = {
            'AAAA': {'close': 5.5, 'volume': 500_000, 'timestamp': datetime.now(timezone.utc)},
            'BBBB': {'close': 8.5, 'volume': 600_000, 'timestamp': datetime.now(timezone.utc)},
            'CCCC': {'close': 11.0, 'volume': 700_000, 'timestamp': datetime.now(timezone.utc)},
        }

        # On second build, CCCC's float is now 20M (secondary offering dilution)
        mock_float_provider.get_stock_info_batch.return_value = {
            'CCCC': {'float_shares': 20_000_000, 'sector': 'Tech', 'country': 'US'},
        }

        result_v2 = builder.build()

        # --- Assert: C deactivated ---
        active_v2 = {s['symbol'] for s in db.get_active_universe()}
        assert 'AAAA' in active_v2, "AAAA should remain active"
        assert 'BBBB' in active_v2, "BBBB should remain active"
        assert 'CCCC' not in active_v2, "CCCC should be deactivated (float 20M > 10M)"

        assert result_v2['total_stocks'] == 2
        assert 'CCCC' in result_v2['removed'], "CCCC should appear in removed list"

        # Verify the stock record still exists but is inactive
        cccc = db.get_universe_stock('CCCC')
        assert cccc is not None, "CCCC record should still exist in DB"
        assert cccc['active'] == 0, "CCCC should be marked inactive"


# =============================================================================
# 7. LLM News Analyzer integration (real Anthropic API)
# =============================================================================


@pytest.mark.integration
class TestLLMNewsAnalyzerIntegration:
    """Integration tests for LLMNewsAnalyzer using real Anthropic API.

    Requires ANTHROPIC_API_KEY in environment.
    Run with: pytest -m integration tests/test_integration.py -v
    """

    @pytest.fixture(autouse=True)
    def _require_api_key(self):
        """Skip if ANTHROPIC_API_KEY is not set."""
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set — skipping LLM integration tests")
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)

    def test_real_catalyst_classified_true(self):
        """Known catalyst headline should be classified as True."""
        from data_sources.news_provider import LLMNewsAnalyzer

        analyzer = LLMNewsAnalyzer(self.client)
        article = {
            'headline': 'XYZ Pharma Receives FDA Approval for Cancer Drug',
            'summary': 'XYZ Pharma announced today that the FDA has granted '
                       'full approval for its novel cancer treatment drug.',
        }
        result = analyzer.is_interesting(article, symbol='XYZ')
        assert result is True, "FDA approval should be classified as a real catalyst"

    def test_listicle_classified_false(self):
        """Known listicle/noise headline should be classified as False."""
        from data_sources.news_provider import LLMNewsAnalyzer

        analyzer = LLMNewsAnalyzer(self.client)
        article = {
            'headline': '12 Stocks Moving In Thursday\'s Pre-Market Session',
            'summary': 'Here are the stocks making moves in pre-market trading today.',
        }
        result = analyzer.is_interesting(article, symbol='RAND')
        assert result is False, "Generic listicle should be classified as noise"
