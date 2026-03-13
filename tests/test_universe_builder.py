"""
Tests for batch/universe_builder.py - UniverseBuilder.

Covers:
- build() full pipeline with mocked data
- _filter_by_price
- _calculate_volume_profile
- TIME_BUCKETS constant validation (26 buckets, 09:30 to 15:45)
"""

import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone

import pandas as pd

from data_sources.alpaca_client import AlpacaClient
from data_sources.float_provider import FloatProvider
from persistence.database import Database
from batch.universe_builder import UniverseBuilder, TIME_BUCKETS


@pytest.fixture
def mock_alpaca():
    """Create a mock AlpacaClient with spec."""
    return MagicMock(spec=AlpacaClient)


@pytest.fixture
def mock_float_provider():
    """Create a mock FloatProvider with spec."""
    return MagicMock(spec=FloatProvider)


@pytest.fixture
def mock_db():
    """Create a mock Database with spec."""
    return MagicMock(spec=Database)


@pytest.fixture
def builder(mock_alpaca, mock_float_provider, mock_db):
    """Create a UniverseBuilder with mocked dependencies."""
    return UniverseBuilder(
        alpaca_client=mock_alpaca,
        float_provider=mock_float_provider,
        db=mock_db,
    )


# =============================================================================
# TIME_BUCKETS constant
# =============================================================================

class TestTimeBuckets:
    """Tests for the TIME_BUCKETS constant."""

    def test_bucket_count(self):
        """TIME_BUCKETS should have exactly 26 entries (09:30 to 15:45)."""
        assert len(TIME_BUCKETS) == 26

    def test_first_bucket(self):
        """First bucket should be 09:30."""
        assert TIME_BUCKETS[0] == '09:30'

    def test_last_bucket(self):
        """Last bucket should be 15:45."""
        assert TIME_BUCKETS[-1] == '15:45'

    def test_no_pre_market_buckets(self):
        """No buckets before 09:30."""
        for bucket in TIME_BUCKETS:
            assert bucket >= '09:30'

    def test_no_after_hours_buckets(self):
        """No buckets after 15:45 (16:00 excluded)."""
        for bucket in TIME_BUCKETS:
            assert bucket <= '15:45'

    def test_15_min_intervals(self):
        """All buckets are at 15-minute intervals (:00, :15, :30, :45)."""
        for bucket in TIME_BUCKETS:
            minutes = int(bucket.split(':')[1])
            assert minutes in (0, 15, 30, 45)

    def test_buckets_sorted(self):
        """TIME_BUCKETS are in chronological order."""
        assert TIME_BUCKETS == sorted(TIME_BUCKETS)


# =============================================================================
# _filter_by_price
# =============================================================================

class TestFilterByPrice:
    """Tests for UniverseBuilder._filter_by_price."""

    def test_filters_in_range(self, builder):
        """Stocks with close price in $2-$20 pass the filter."""
        assets = [
            {'symbol': 'CHEAP', 'company_name': 'Cheap Inc'},
            {'symbol': 'MID', 'company_name': 'Mid Inc'},
            {'symbol': 'PRICEY', 'company_name': 'Pricey Inc'},
        ]
        daily_bars = {
            'CHEAP': {'close': 1.50},   # Too cheap
            'MID': {'close': 10.0},     # In range
            'PRICEY': {'close': 25.0},  # Too expensive
        }

        result = builder._filter_by_price(assets, daily_bars)

        assert len(result) == 1
        assert result[0]['symbol'] == 'MID'

    def test_boundary_values(self, builder):
        """Stocks at exactly $2 and $20 pass the filter (inclusive)."""
        assets = [
            {'symbol': 'LOW', 'company_name': ''},
            {'symbol': 'HIGH', 'company_name': ''},
        ]
        daily_bars = {
            'LOW': {'close': 2.0},
            'HIGH': {'close': 20.0},
        }

        result = builder._filter_by_price(assets, daily_bars)

        assert len(result) == 2

    def test_missing_bar_data(self, builder):
        """Stocks without bar data are excluded."""
        assets = [
            {'symbol': 'NOBAR', 'company_name': 'No Bar Inc'},
            {'symbol': 'HASBAR', 'company_name': 'Has Bar Inc'},
        ]
        daily_bars = {
            'HASBAR': {'close': 5.0},
        }

        result = builder._filter_by_price(assets, daily_bars)

        assert len(result) == 1
        assert result[0]['symbol'] == 'HASBAR'

    def test_empty_assets(self, builder):
        """Empty assets list returns empty result."""
        result = builder._filter_by_price([], {})
        assert result == []


# =============================================================================
# _calculate_volume_profile
# =============================================================================

class TestCalculateVolumeProfile:
    """Tests for UniverseBuilder._calculate_volume_profile."""

    def test_calculates_averages(self, builder, mock_alpaca):
        """Calculates average volume per 15-min bucket from intraday bars."""
        # Create 2 days of data for the 09:30 bucket
        timestamps = [
            datetime(2026, 3, 11, 9, 30, tzinfo=timezone.utc),
            datetime(2026, 3, 12, 9, 30, tzinfo=timezone.utc),
        ]
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': [5.0, 5.1],
            'high': [5.2, 5.3],
            'low': [4.9, 5.0],
            'close': [5.1, 5.2],
            'volume': [10000, 20000],
        })
        mock_alpaca.get_intraday_bars.return_value = df

        result = builder._calculate_volume_profile("TEST")

        assert len(result) == 1
        assert result[0]['symbol'] == 'TEST'
        assert result[0]['time_bucket'] == '09:30'
        assert result[0]['avg_volume'] == 15000  # avg of 10000 and 20000

    def test_empty_dataframe(self, builder, mock_alpaca):
        """Returns empty list when no intraday data is available."""
        mock_alpaca.get_intraday_bars.return_value = pd.DataFrame()

        result = builder._calculate_volume_profile("EMPTY")

        assert result == []

    def test_filters_non_market_hours(self, builder, mock_alpaca):
        """Filters out bars outside market hours (before 09:30 or after 15:45)."""
        timestamps = [
            datetime(2026, 3, 11, 8, 0, tzinfo=timezone.utc),   # Pre-market
            datetime(2026, 3, 11, 10, 0, tzinfo=timezone.utc),  # Market hours
            datetime(2026, 3, 11, 16, 15, tzinfo=timezone.utc), # After hours
        ]
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': [5.0, 5.1, 5.2],
            'high': [5.2, 5.3, 5.4],
            'low': [4.9, 5.0, 5.1],
            'close': [5.1, 5.2, 5.3],
            'volume': [1000, 5000, 2000],
        })
        mock_alpaca.get_intraday_bars.return_value = df

        result = builder._calculate_volume_profile("FILTERED")

        # Only 10:00 bucket should remain
        assert len(result) == 1
        assert result[0]['time_bucket'] == '10:00'
        assert result[0]['avg_volume'] == 5000


# =============================================================================
# build() full pipeline
# =============================================================================

class TestBuild:
    """Tests for UniverseBuilder.build full pipeline."""

    def test_full_pipeline(self, builder, mock_alpaca, mock_float_provider, mock_db):
        """build() orchestrates the full pipeline: assets -> price -> float -> volume."""
        # Step 1: Previous universe
        mock_db.get_active_universe.side_effect = [
            # First call: previous universe (for comparison)
            [{'symbol': 'OLD', 'float_shares': 1_000_000}],
            # Second call: after upserts (for float filtering)
            [
                {'symbol': 'GOOD', 'float_shares': 2_000_000, 'price_close': 5.0},
                {'symbol': 'BIGFLOAT', 'float_shares': 50_000_000, 'price_close': 8.0},
            ],
        ]

        # Step 1: Tradeable assets
        mock_alpaca.get_all_tradeable_assets.return_value = [
            {'symbol': 'GOOD', 'company_name': 'Good Co', 'exchange': 'NASDAQ'},
            {'symbol': 'BIGFLOAT', 'company_name': 'Big Float Co', 'exchange': 'NYSE'},
            {'symbol': 'EXPENSIVE', 'company_name': 'Expensive Co', 'exchange': 'NYSE'},
        ]

        # Step 2: Daily bars
        mock_alpaca.get_daily_bars.return_value = {
            'GOOD': {'close': 5.0, 'volume': 100000},
            'BIGFLOAT': {'close': 8.0, 'volume': 200000},
            'EXPENSIVE': {'close': 50.0, 'volume': 500000},  # Filtered by price
        }

        # Step 4: Float update
        mock_db.get_symbols_needing_float_update.return_value = ['GOOD', 'BIGFLOAT']
        mock_float_provider.get_float_batch.return_value = {
            'GOOD': 2_000_000,
            'BIGFLOAT': 50_000_000,
        }
        mock_float_provider.get_stock_info.return_value = {
            'sector': 'Technology',
            'country': 'US',
            'float_shares': None,
        }

        # Step 6: Volume profiles
        mock_alpaca.get_intraday_bars.return_value = pd.DataFrame()
        mock_db.get_volume_profile_count.return_value = 1

        result = builder.build()

        # Verify pipeline was called
        mock_alpaca.get_all_tradeable_assets.assert_called_once()
        mock_alpaca.get_daily_bars.assert_called_once()
        mock_float_provider.get_float_batch.assert_called_once()

        # GOOD passes float filter (2M <= 10M), BIGFLOAT does not (50M > 10M)
        assert result['total_stocks'] >= 1

        # Result has expected keys
        assert 'total_stocks' in result
        assert 'volume_profiles' in result
        assert 'added' in result
        assert 'removed' in result

    def test_deactivates_removed_stocks(self, builder, mock_alpaca, mock_float_provider, mock_db):
        """build() deactivates stocks that no longer qualify."""
        # Previous universe had OLD stock
        mock_db.get_active_universe.side_effect = [
            [{'symbol': 'OLD', 'float_shares': 1_000_000}],
            # After upserts: only NEW passes float
            [{'symbol': 'NEW', 'float_shares': 2_000_000, 'price_close': 5.0}],
        ]

        mock_alpaca.get_all_tradeable_assets.return_value = [
            {'symbol': 'NEW', 'company_name': 'New Co', 'exchange': 'NASDAQ'},
        ]
        mock_alpaca.get_daily_bars.return_value = {
            'NEW': {'close': 5.0, 'volume': 100000},
        }
        mock_db.get_symbols_needing_float_update.return_value = []
        mock_alpaca.get_intraday_bars.return_value = pd.DataFrame()
        mock_db.get_volume_profile_count.return_value = 0

        builder.build()

        # OLD should be deactivated (was in previous, not in current)
        mock_db.deactivate_stocks.assert_called()
        all_deactivated = set()
        for c in mock_db.deactivate_stocks.call_args_list:
            all_deactivated.update(c[0][0])
        assert 'OLD' in all_deactivated
