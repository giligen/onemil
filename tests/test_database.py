"""
Unit tests for persistence.database.Database class.

Uses a temporary SQLite file per test to guarantee isolation.
No external services or mocks needed -- Database talks directly to SQLite.
"""

import os
import pytest
from datetime import datetime, timezone, timedelta

from persistence.database import (
    Database, get_database, reset_database,
    _adapt_datetime_iso, _convert_timestamp,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path):
    """Create a Database backed by a temp file and tear it down after the test."""
    db_path = str(tmp_path / "test_onemil.db")
    database = Database(db_path=db_path)
    yield database
    database.close()


def _make_stock(symbol="AAPL", active=1, float_shares=5_000_000,
                float_updated_at=None, price_close=10.0):
    """Return a minimal universe stock dict."""
    return {
        "symbol": symbol,
        "company_name": f"{symbol} Inc.",
        "exchange": "NASDAQ",
        "sector": "Technology",
        "country": "US",
        "price_close": price_close,
        "float_shares": float_shares,
        "float_updated_at": float_updated_at,
        "avg_volume_daily": 1_000_000,
        "last_updated": datetime.now(timezone.utc),
        "active": active,
    }


def _make_scan_result(symbol="AAPL", scan_date="2026-03-13", phase="premarket",
                      gap_pct=5.0, qualified=1):
    """Return a minimal scan_results dict."""
    return {
        "scan_date": scan_date,
        "symbol": symbol,
        "detected_at": datetime.now(timezone.utc),
        "phase": phase,
        "prev_close": 9.0,
        "current_price": 10.0,
        "gap_pct": gap_pct,
        "intraday_change_pct": 11.0,
        "relative_volume": 6.0,
        "current_volume": 500_000,
        "time_bucket": "09:30",
        "float_shares": 4_000_000,
        "has_news": 1,
        "news_headline": "Big news",
        "qualified": qualified,
    }


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

class TestTableCreation:
    """Verify that schema is created on initialisation."""

    def test_tables_exist(self, db):
        """All expected tables are created on Database init."""
        cursor = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row["name"] for row in cursor.fetchall()}
        assert "universe" in tables
        assert "volume_profiles" in tables
        assert "scan_results" in tables

    def test_indexes_exist(self, db):
        """Expected indexes are created on Database init."""
        cursor = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
        )
        indexes = {row["name"] for row in cursor.fetchall()}
        assert "idx_scan_results_date" in indexes
        assert "idx_scan_results_symbol" in indexes
        assert "idx_volume_profiles_symbol" in indexes


# ---------------------------------------------------------------------------
# Universe operations
# ---------------------------------------------------------------------------

class TestUpsertUniverseStock:
    """Tests for upsert_universe_stock (single row)."""

    def test_insert_new_stock(self, db):
        """Inserting a stock that does not exist creates a new row."""
        stock = _make_stock("TSLA")
        db.upsert_universe_stock(stock)

        row = db.get_universe_stock("TSLA")
        assert row is not None
        assert row["symbol"] == "TSLA"
        assert row["company_name"] == "TSLA Inc."

    def test_update_existing_stock(self, db):
        """Re-upserting the same symbol updates the row."""
        db.upsert_universe_stock(_make_stock("TSLA", price_close=10.0))
        db.upsert_universe_stock(_make_stock("TSLA", price_close=15.0))

        row = db.get_universe_stock("TSLA")
        assert row["price_close"] == 15.0

    def test_upsert_preserves_float_when_null(self, db):
        """Upserting with NULL float_shares keeps the existing float value."""
        db.upsert_universe_stock(_make_stock("TSLA", float_shares=3_000_000,
                                             float_updated_at=datetime.now(timezone.utc)))
        db.upsert_universe_stock(_make_stock("TSLA", float_shares=None, float_updated_at=None))

        row = db.get_universe_stock("TSLA")
        assert row["float_shares"] == 3_000_000


class TestUpsertUniverseStocksBatch:
    """Tests for upsert_universe_stocks_batch."""

    def test_batch_insert(self, db):
        """Batch insert creates multiple rows."""
        stocks = [_make_stock("A"), _make_stock("B"), _make_stock("C")]
        count = db.upsert_universe_stocks_batch(stocks)

        assert count == 3
        assert db.get_universe_count() == 3

    def test_batch_empty_list(self, db):
        """Empty list returns 0 and inserts nothing."""
        assert db.upsert_universe_stocks_batch([]) == 0

    def test_batch_update_existing(self, db):
        """Batch upsert updates rows that already exist."""
        db.upsert_universe_stock(_make_stock("X", price_close=5.0))
        db.upsert_universe_stocks_batch([_make_stock("X", price_close=8.0)])

        row = db.get_universe_stock("X")
        assert row["price_close"] == 8.0


class TestGetActiveUniverse:
    """Tests for get_active_universe."""

    def test_returns_only_active(self, db):
        """Only stocks with active=1 are returned."""
        db.upsert_universe_stock(_make_stock("ACTIVE", active=1))
        db.upsert_universe_stock(_make_stock("DEAD", active=0))

        active = db.get_active_universe()
        symbols = [s["symbol"] for s in active]
        assert "ACTIVE" in symbols
        assert "DEAD" not in symbols

    def test_empty_universe(self, db):
        """Returns empty list when no stocks exist."""
        assert db.get_active_universe() == []


class TestDeactivateStocks:
    """Tests for deactivate_stocks."""

    def test_deactivate(self, db):
        """Deactivated stocks are no longer in the active universe."""
        db.upsert_universe_stocks_batch([_make_stock("A"), _make_stock("B")])
        count = db.deactivate_stocks(["A"])

        assert count == 1
        active = db.get_active_universe()
        symbols = [s["symbol"] for s in active]
        assert "A" not in symbols
        assert "B" in symbols

    def test_deactivate_empty_list(self, db):
        """Deactivating an empty list is a no-op."""
        assert db.deactivate_stocks([]) == 0

    def test_deactivate_nonexistent_symbol(self, db):
        """Deactivating a symbol not in universe returns 0."""
        assert db.deactivate_stocks(["ZZZZZ"]) == 0


class TestGetSymbolsNeedingFloatUpdate:
    """Tests for get_symbols_needing_float_update."""

    def test_null_float_needs_update(self, db):
        """Stocks with no float_updated_at are returned."""
        db.upsert_universe_stock(_make_stock("STALE", float_updated_at=None))
        needing = db.get_symbols_needing_float_update(max_age_days=7)
        assert "STALE" in needing

    def test_old_float_needs_update(self, db):
        """Stocks with float_updated_at older than max_age_days are returned."""
        old_date = datetime.now(timezone.utc) - timedelta(days=10)
        db.upsert_universe_stock(_make_stock("OLD", float_updated_at=old_date))
        assert "OLD" in db.get_symbols_needing_float_update(max_age_days=7)

    def test_fresh_float_excluded(self, db):
        """Stocks with recent float_updated_at are excluded."""
        fresh = datetime.now(timezone.utc) - timedelta(days=1)
        db.upsert_universe_stock(_make_stock("FRESH", float_updated_at=fresh))
        assert "FRESH" not in db.get_symbols_needing_float_update(max_age_days=7)

    def test_inactive_excluded(self, db):
        """Inactive stocks are excluded even if float is stale."""
        db.upsert_universe_stock(_make_stock("GONE", active=0, float_updated_at=None))
        assert "GONE" not in db.get_symbols_needing_float_update(max_age_days=7)


class TestUpdateFloat:
    """Tests for update_float."""

    def test_float_is_updated(self, db):
        """update_float sets float_shares and float_updated_at."""
        db.upsert_universe_stock(_make_stock("UPD", float_shares=None))
        db.update_float("UPD", 2_500_000)

        row = db.get_universe_stock("UPD")
        assert row["float_shares"] == 2_500_000
        assert row["float_updated_at"] is not None


# ---------------------------------------------------------------------------
# Volume profile operations
# ---------------------------------------------------------------------------

class TestVolumeProfiles:
    """Tests for volume-profile CRUD."""

    def _seed_stock(self, db, symbol="AAPL"):
        """Insert a universe row so FK is satisfied."""
        db.upsert_universe_stock(_make_stock(symbol))

    def test_upsert_volume_profiles(self, db):
        """Upserting profiles creates rows and returns the count."""
        self._seed_stock(db, "AAPL")
        profiles = [
            {"symbol": "AAPL", "time_bucket": "09:30", "avg_volume": 50000,
             "last_updated": datetime.now(timezone.utc)},
            {"symbol": "AAPL", "time_bucket": "09:45", "avg_volume": 30000,
             "last_updated": datetime.now(timezone.utc)},
        ]
        assert db.upsert_volume_profiles(profiles) == 2

    def test_upsert_empty_list(self, db):
        """Empty list returns 0."""
        assert db.upsert_volume_profiles([]) == 0

    def test_get_volume_profile(self, db):
        """get_volume_profile returns bucket->volume mapping for one symbol."""
        self._seed_stock(db, "AAPL")
        db.upsert_volume_profiles([
            {"symbol": "AAPL", "time_bucket": "09:30", "avg_volume": 50000,
             "last_updated": datetime.now(timezone.utc)},
            {"symbol": "AAPL", "time_bucket": "09:45", "avg_volume": 30000,
             "last_updated": datetime.now(timezone.utc)},
        ])

        profile = db.get_volume_profile("AAPL")
        assert profile == {"09:30": 50000, "09:45": 30000}

    def test_get_volume_profile_empty(self, db):
        """Returns empty dict for symbol with no profile."""
        assert db.get_volume_profile("NONE") == {}

    def test_get_all_volume_profiles(self, db):
        """get_all_volume_profiles returns symbol->bucket->volume nested dict."""
        self._seed_stock(db, "AAPL")
        self._seed_stock(db, "TSLA")
        now = datetime.now(timezone.utc)
        db.upsert_volume_profiles([
            {"symbol": "AAPL", "time_bucket": "09:30", "avg_volume": 50000, "last_updated": now},
            {"symbol": "TSLA", "time_bucket": "09:30", "avg_volume": 70000, "last_updated": now},
        ])

        all_profiles = db.get_all_volume_profiles()
        assert "AAPL" in all_profiles
        assert "TSLA" in all_profiles
        assert all_profiles["AAPL"]["09:30"] == 50000
        assert all_profiles["TSLA"]["09:30"] == 70000

    def test_upsert_updates_existing_profile(self, db):
        """Re-upserting a (symbol, time_bucket) pair updates avg_volume."""
        self._seed_stock(db, "AAPL")
        now = datetime.now(timezone.utc)
        db.upsert_volume_profiles([
            {"symbol": "AAPL", "time_bucket": "09:30", "avg_volume": 50000, "last_updated": now},
        ])
        db.upsert_volume_profiles([
            {"symbol": "AAPL", "time_bucket": "09:30", "avg_volume": 99999, "last_updated": now},
        ])

        profile = db.get_volume_profile("AAPL")
        assert profile["09:30"] == 99999


# ---------------------------------------------------------------------------
# Scan results
# ---------------------------------------------------------------------------

class TestScanResults:
    """Tests for scan result CRUD."""

    def test_save_and_retrieve(self, db):
        """save_scan_result stores a row retrievable by get_scan_results."""
        result = _make_scan_result("AAPL")
        row_id = db.save_scan_result(result)
        assert isinstance(row_id, int)

        results = db.get_scan_results("2026-03-13")
        assert len(results) == 1
        assert results[0]["symbol"] == "AAPL"

    def test_get_scan_results_filter_by_phase(self, db):
        """Phase filter returns only matching rows."""
        db.save_scan_result(_make_scan_result("A", phase="premarket"))
        db.save_scan_result(_make_scan_result("B", phase="intraday"))

        pre = db.get_scan_results("2026-03-13", phase="premarket")
        intra = db.get_scan_results("2026-03-13", phase="intraday")
        assert len(pre) == 1
        assert pre[0]["symbol"] == "A"
        assert len(intra) == 1
        assert intra[0]["symbol"] == "B"

    def test_get_scan_results_no_match(self, db):
        """Returns empty list when no results match the date."""
        assert db.get_scan_results("1999-01-01") == []


class TestPremarketGapSymbols:
    """Tests for get_premarket_gap_symbols."""

    def test_returns_qualifying_symbols(self, db):
        """Symbols with premarket phase and gap_pct >= 2.0 are returned."""
        db.save_scan_result(_make_scan_result("GAP", phase="premarket", gap_pct=5.0))
        db.save_scan_result(_make_scan_result("LOW", phase="premarket", gap_pct=1.0))
        db.save_scan_result(_make_scan_result("INTRA", phase="intraday", gap_pct=10.0))

        symbols = db.get_premarket_gap_symbols("2026-03-13")
        assert "GAP" in symbols
        assert "LOW" not in symbols
        assert "INTRA" not in symbols

    def test_no_duplicates(self, db):
        """Multiple scan results for the same symbol produce a single entry."""
        db.save_scan_result(_make_scan_result("DUP", phase="premarket", gap_pct=3.0))
        db.save_scan_result(_make_scan_result("DUP", phase="premarket", gap_pct=4.0))

        symbols = db.get_premarket_gap_symbols("2026-03-13")
        assert symbols.count("DUP") == 1

    def test_upsert_updates_existing_row(self, db):
        """INSERT OR REPLACE updates the row instead of creating duplicates."""
        db.save_scan_result(_make_scan_result("UPD", phase="premarket", gap_pct=3.0))
        db.save_scan_result(_make_scan_result("UPD", phase="premarket", gap_pct=7.0))

        results = db.get_scan_results("2026-03-13", phase="premarket")
        upd_rows = [r for r in results if r["symbol"] == "UPD"]
        assert len(upd_rows) == 1, "Should have exactly 1 row after upsert"
        assert upd_rows[0]["gap_pct"] == 7.0, "Should have the latest gap_pct"

    def test_different_buckets_are_separate_rows(self, db):
        """Same symbol in different time buckets creates separate rows."""
        r1 = _make_scan_result("MULTI", phase="intraday")
        r1["time_bucket"] = "09:30"
        r2 = _make_scan_result("MULTI", phase="intraday")
        r2["time_bucket"] = "09:45"
        db.save_scan_result(r1)
        db.save_scan_result(r2)

        results = db.get_scan_results("2026-03-13", phase="intraday")
        multi_rows = [r for r in results if r["symbol"] == "MULTI"]
        assert len(multi_rows) == 2, "Different buckets should create separate rows"


# ---------------------------------------------------------------------------
# Counts / Utility
# ---------------------------------------------------------------------------

class TestCounts:
    """Tests for utility count methods."""

    def test_get_universe_count(self, db):
        """get_universe_count returns the number of active stocks."""
        db.upsert_universe_stocks_batch([
            _make_stock("A", active=1),
            _make_stock("B", active=1),
            _make_stock("C", active=0),
        ])
        assert db.get_universe_count() == 2

    def test_get_universe_count_empty(self, db):
        """get_universe_count returns 0 on an empty table."""
        assert db.get_universe_count() == 0

    def test_get_volume_profile_count(self, db):
        """get_volume_profile_count returns distinct symbols with profiles."""
        db.upsert_universe_stock(_make_stock("AAPL"))
        db.upsert_universe_stock(_make_stock("TSLA"))
        now = datetime.now(timezone.utc)
        db.upsert_volume_profiles([
            {"symbol": "AAPL", "time_bucket": "09:30", "avg_volume": 100, "last_updated": now},
            {"symbol": "AAPL", "time_bucket": "09:45", "avg_volume": 200, "last_updated": now},
            {"symbol": "TSLA", "time_bucket": "09:30", "avg_volume": 300, "last_updated": now},
        ])
        assert db.get_volume_profile_count() == 2

    def test_get_volume_profile_count_empty(self, db):
        """get_volume_profile_count returns 0 when no profiles exist."""
        assert db.get_volume_profile_count() == 0


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

class TestDatabaseSingleton:
    """Tests for get_database() / reset_database() singleton management."""

    def test_get_database_returns_same_instance(self, tmp_path):
        """Consecutive get_database() calls return the same object."""
        db_path = str(tmp_path / "singleton.db")
        reset_database()
        try:
            first = get_database(db_path=db_path)
            second = get_database(db_path=db_path)
            assert first is second
        finally:
            reset_database()

    def test_reset_database_clears_singleton(self, tmp_path):
        """reset_database() forces a fresh instance on next get_database()."""
        db_path = str(tmp_path / "singleton.db")
        reset_database()
        try:
            first = get_database(db_path=db_path)
            reset_database()
            second = get_database(db_path=db_path)
            assert first is not second
        finally:
            reset_database()


# ---------------------------------------------------------------------------
# Timestamp adapters / converters
# ---------------------------------------------------------------------------

class TestTimestampHelpers:
    """Tests for _adapt_datetime_iso and _convert_timestamp."""

    def test_adapt_datetime_iso_normal(self):
        """_adapt_datetime_iso returns ISO string for a datetime."""
        dt = datetime(2026, 3, 13, 12, 0, 0, tzinfo=timezone.utc)
        assert _adapt_datetime_iso(dt) == "2026-03-13T12:00:00+00:00"

    def test_adapt_datetime_iso_none(self):
        """_adapt_datetime_iso returns None for None input."""
        assert _adapt_datetime_iso(None) is None

    def test_convert_timestamp_valid(self):
        """_convert_timestamp parses valid ISO bytes to datetime."""
        val = b"2026-03-13T12:00:00+00:00"
        result = _convert_timestamp(val)
        assert isinstance(result, datetime)
        assert result.year == 2026

    def test_convert_timestamp_none(self):
        """_convert_timestamp returns None for None input."""
        assert _convert_timestamp(None) is None

    def test_convert_timestamp_invalid(self):
        """_convert_timestamp returns None for unparseable input."""
        assert _convert_timestamp(b"not-a-date") is None


# ---------------------------------------------------------------------------
# Delete volume profiles
# ---------------------------------------------------------------------------

class TestDeleteVolumeProfiles:
    """Tests for delete_volume_profiles."""

    def test_delete_removes_profiles(self, db):
        """delete_volume_profiles removes all rows for that symbol."""
        db.upsert_universe_stock(_make_stock("DEL"))
        now = datetime.now(timezone.utc)
        db.upsert_volume_profiles([
            {"symbol": "DEL", "time_bucket": "09:30", "avg_volume": 100, "last_updated": now},
        ])
        assert db.get_volume_profile("DEL") != {}

        db.delete_volume_profiles("DEL")
        assert db.get_volume_profile("DEL") == {}

    def test_delete_nonexistent_is_noop(self, db):
        """Deleting profiles for a symbol with none is a no-op."""
        db.delete_volume_profiles("NOPE")  # should not raise
