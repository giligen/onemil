"""
SQLite database for OneMil scanner persistence.

Tables:
- universe: Stock universe with price, float, volume filters
- volume_profiles: 15-min bucketed average volumes (50-day)
- scan_results: Scanner output (pre-market + intraday hits)

Handles concurrent access via WAL mode and busy timeouts.
"""

import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Singleton instance
_db_instance: Optional['Database'] = None


# =============================================================================
# Custom SQLite adapters/converters for timezone-aware timestamps
# =============================================================================

def _adapt_datetime_iso(dt: datetime) -> str:
    """Adapt datetime to ISO format string for storage."""
    if dt is None:
        return None
    return dt.isoformat()


def _convert_timestamp(val: bytes) -> Optional[datetime]:
    """Convert stored timestamp bytes to timezone-aware datetime."""
    if val is None:
        return None

    text = val.decode('utf-8')
    try:
        return datetime.fromisoformat(text)
    except (ValueError, AttributeError):
        logger.warning(f"Could not parse timestamp: {text}")
        return None


sqlite3.register_converter("TIMESTAMP", _convert_timestamp)
sqlite3.register_adapter(datetime, _adapt_datetime_iso)


class Database:
    """
    SQLite database for scanner data persistence.

    Uses WAL mode for concurrent read access and busy timeout
    for handling locked DB from parallel processes.
    """

    def __init__(self, db_path: str = "data/onemil.db"):
        """
        Initialize database connection and create tables.

        Args:
            db_path: Path to SQLite database file
        """
        # Re-register converters to override any third-party (e.g., peewee) interference
        sqlite3.register_converter("TIMESTAMP", _convert_timestamp)
        sqlite3.register_adapter(datetime, _adapt_datetime_iso)

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(
            str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            timeout=30  # Wait up to 30s for locked DB
        )
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.row_factory = sqlite3.Row

        self._create_tables()
        logger.info(f"Database initialized: {self.db_path}")

    def _create_tables(self) -> None:
        """Create all tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS universe (
                symbol VARCHAR(10) PRIMARY KEY,
                company_name TEXT,
                exchange VARCHAR(10),
                sector TEXT,
                country TEXT,
                price_close REAL,
                float_shares INTEGER,
                float_updated_at TIMESTAMP,
                avg_volume_daily INTEGER,
                last_updated TIMESTAMP,
                active INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS volume_profiles (
                symbol VARCHAR(10),
                time_bucket VARCHAR(5),
                avg_volume INTEGER,
                last_updated TIMESTAMP,
                PRIMARY KEY (symbol, time_bucket),
                FOREIGN KEY (symbol) REFERENCES universe(symbol)
            );

            CREATE TABLE IF NOT EXISTS scan_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date DATE,
                symbol VARCHAR(10),
                detected_at TIMESTAMP,
                phase VARCHAR(10),
                prev_close REAL,
                current_price REAL,
                gap_pct REAL,
                intraday_change_pct REAL,
                relative_volume REAL,
                current_volume INTEGER,
                time_bucket VARCHAR(5),
                float_shares INTEGER,
                has_news INTEGER,
                news_headline TEXT,
                qualified INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_scan_results_date
                ON scan_results(scan_date);
            CREATE INDEX IF NOT EXISTS idx_scan_results_symbol
                ON scan_results(symbol, scan_date);
            CREATE INDEX IF NOT EXISTS idx_volume_profiles_symbol
                ON volume_profiles(symbol);
        """)
        self.conn.commit()
        logger.debug("Database tables verified/created")

    # =========================================================================
    # Universe operations
    # =========================================================================

    def upsert_universe_stock(self, stock: Dict[str, Any]) -> None:
        """
        Insert or update a stock in the universe table.

        Args:
            stock: Dict with keys matching universe columns
        """
        self.conn.execute("""
            INSERT INTO universe (symbol, company_name, exchange, sector, country,
                                  price_close, float_shares, float_updated_at,
                                  avg_volume_daily, last_updated, active)
            VALUES (:symbol, :company_name, :exchange, :sector, :country,
                    :price_close, :float_shares, :float_updated_at,
                    :avg_volume_daily, :last_updated, :active)
            ON CONFLICT(symbol) DO UPDATE SET
                company_name = excluded.company_name,
                exchange = excluded.exchange,
                sector = excluded.sector,
                country = excluded.country,
                price_close = excluded.price_close,
                float_shares = COALESCE(excluded.float_shares, universe.float_shares),
                float_updated_at = COALESCE(excluded.float_updated_at, universe.float_updated_at),
                avg_volume_daily = excluded.avg_volume_daily,
                last_updated = excluded.last_updated,
                active = excluded.active
        """, stock)
        self.conn.commit()

    def upsert_universe_stocks_batch(self, stocks: List[Dict[str, Any]]) -> int:
        """
        Batch insert/update stocks in the universe table.

        Args:
            stocks: List of dicts with keys matching universe columns

        Returns:
            Number of stocks upserted
        """
        if not stocks:
            return 0

        self.conn.executemany("""
            INSERT INTO universe (symbol, company_name, exchange, sector, country,
                                  price_close, float_shares, float_updated_at,
                                  avg_volume_daily, last_updated, active)
            VALUES (:symbol, :company_name, :exchange, :sector, :country,
                    :price_close, :float_shares, :float_updated_at,
                    :avg_volume_daily, :last_updated, :active)
            ON CONFLICT(symbol) DO UPDATE SET
                company_name = excluded.company_name,
                exchange = excluded.exchange,
                sector = excluded.sector,
                country = excluded.country,
                price_close = excluded.price_close,
                float_shares = COALESCE(excluded.float_shares, universe.float_shares),
                float_updated_at = COALESCE(excluded.float_updated_at, universe.float_updated_at),
                avg_volume_daily = excluded.avg_volume_daily,
                last_updated = excluded.last_updated,
                active = excluded.active
        """, stocks)
        self.conn.commit()
        return len(stocks)

    def get_active_universe(self) -> List[Dict[str, Any]]:
        """Get all active stocks in the universe."""
        cursor = self.conn.execute(
            "SELECT * FROM universe WHERE active = 1 ORDER BY symbol"
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_universe_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get a single stock from the universe."""
        cursor = self.conn.execute(
            "SELECT * FROM universe WHERE symbol = ?", (symbol,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def deactivate_stocks(self, symbols: List[str]) -> int:
        """
        Mark stocks as inactive (removed from universe).

        Args:
            symbols: List of symbols to deactivate

        Returns:
            Number of stocks deactivated
        """
        if not symbols:
            return 0

        placeholders = ','.join('?' for _ in symbols)
        cursor = self.conn.execute(
            f"UPDATE universe SET active = 0, last_updated = ? WHERE symbol IN ({placeholders})",
            [datetime.now(timezone.utc)] + symbols
        )
        self.conn.commit()
        return cursor.rowcount

    def get_symbols_needing_float_update(self, max_age_days: int = 7) -> List[str]:
        """
        Get active symbols whose float data is stale or missing.

        Args:
            max_age_days: Consider float stale after this many days

        Returns:
            List of symbols needing float update
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        cursor = self.conn.execute("""
            SELECT symbol FROM universe
            WHERE active = 1
              AND (float_updated_at IS NULL OR float_updated_at < ?)
            ORDER BY symbol
        """, (cutoff,))
        return [row['symbol'] for row in cursor.fetchall()]

    def update_float(self, symbol: str, float_shares: Optional[int]) -> None:
        """
        Update float data for a symbol.

        Also updates float_updated_at to mark it as recently checked,
        even when float_shares is None (prevents re-fetching unavailable data).
        """
        now = datetime.now(timezone.utc)
        self.conn.execute("""
            UPDATE universe SET float_shares = ?, float_updated_at = ?, last_updated = ?
            WHERE symbol = ?
        """, (float_shares, now, now, symbol))
        self.conn.commit()

    # =========================================================================
    # Volume profile operations
    # =========================================================================

    def upsert_volume_profiles(self, profiles: List[Dict[str, Any]]) -> int:
        """
        Batch upsert volume profiles.

        Args:
            profiles: List of dicts with symbol, time_bucket, avg_volume, last_updated

        Returns:
            Number of profiles upserted
        """
        if not profiles:
            return 0

        self.conn.executemany("""
            INSERT INTO volume_profiles (symbol, time_bucket, avg_volume, last_updated)
            VALUES (:symbol, :time_bucket, :avg_volume, :last_updated)
            ON CONFLICT(symbol, time_bucket) DO UPDATE SET
                avg_volume = excluded.avg_volume,
                last_updated = excluded.last_updated
        """, profiles)
        self.conn.commit()
        return len(profiles)

    def get_volume_profile(self, symbol: str) -> Dict[str, int]:
        """
        Get volume profile for a symbol.

        Returns:
            Dict mapping time_bucket -> avg_volume (e.g., {'09:30': 50000, ...})
        """
        cursor = self.conn.execute(
            "SELECT time_bucket, avg_volume FROM volume_profiles WHERE symbol = ?",
            (symbol,)
        )
        return {row['time_bucket']: row['avg_volume'] for row in cursor.fetchall()}

    def get_all_volume_profiles(self) -> Dict[str, Dict[str, int]]:
        """
        Get all volume profiles keyed by symbol.

        Returns:
            Dict mapping symbol -> {time_bucket: avg_volume}
        """
        cursor = self.conn.execute(
            "SELECT symbol, time_bucket, avg_volume FROM volume_profiles ORDER BY symbol"
        )
        profiles: Dict[str, Dict[str, int]] = {}
        for row in cursor.fetchall():
            sym = row['symbol']
            if sym not in profiles:
                profiles[sym] = {}
            profiles[sym][row['time_bucket']] = row['avg_volume']
        return profiles

    def delete_volume_profiles(self, symbol: str) -> None:
        """Delete all volume profiles for a symbol."""
        self.conn.execute("DELETE FROM volume_profiles WHERE symbol = ?", (symbol,))
        self.conn.commit()

    # =========================================================================
    # Scan results operations
    # =========================================================================

    def save_scan_result(self, result: Dict[str, Any]) -> int:
        """
        Save a scan result.

        Args:
            result: Dict with scan result data

        Returns:
            ID of the inserted row
        """
        cursor = self.conn.execute("""
            INSERT INTO scan_results (scan_date, symbol, detected_at, phase,
                                      prev_close, current_price, gap_pct,
                                      intraday_change_pct, relative_volume,
                                      current_volume, time_bucket, float_shares,
                                      has_news, news_headline, qualified)
            VALUES (:scan_date, :symbol, :detected_at, :phase,
                    :prev_close, :current_price, :gap_pct,
                    :intraday_change_pct, :relative_volume,
                    :current_volume, :time_bucket, :float_shares,
                    :has_news, :news_headline, :qualified)
        """, result)
        self.conn.commit()
        return cursor.lastrowid

    def get_scan_results(self, scan_date: str, phase: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get scan results for a date, optionally filtered by phase.

        Args:
            scan_date: Date string (YYYY-MM-DD)
            phase: Optional phase filter ('premarket' or 'intraday')

        Returns:
            List of scan result dicts
        """
        if phase:
            cursor = self.conn.execute(
                "SELECT * FROM scan_results WHERE scan_date = ? AND phase = ? ORDER BY detected_at",
                (scan_date, phase)
            )
        else:
            cursor = self.conn.execute(
                "SELECT * FROM scan_results WHERE scan_date = ? ORDER BY detected_at",
                (scan_date,)
            )
        return [dict(row) for row in cursor.fetchall()]

    def get_premarket_gap_symbols(self, scan_date: str) -> List[str]:
        """Get symbols that had a pre-market gap on the given date."""
        cursor = self.conn.execute("""
            SELECT DISTINCT symbol FROM scan_results
            WHERE scan_date = ? AND phase = 'premarket' AND gap_pct >= 2.0
            ORDER BY symbol
        """, (scan_date,))
        return [row['symbol'] for row in cursor.fetchall()]

    # =========================================================================
    # Utility
    # =========================================================================

    def get_universe_count(self) -> int:
        """Get count of active stocks in universe."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM universe WHERE active = 1")
        return cursor.fetchone()[0]

    def get_volume_profile_count(self) -> int:
        """Get count of unique symbols with volume profiles."""
        cursor = self.conn.execute("SELECT COUNT(DISTINCT symbol) FROM volume_profiles")
        return cursor.fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def get_database(db_path: str = "data/onemil.db") -> Database:
    """
    Get or create the singleton Database instance.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Database singleton instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(db_path=db_path)
    return _db_instance


def reset_database() -> None:
    """Reset the singleton (for testing)."""
    global _db_instance
    if _db_instance is not None:
        _db_instance.close()
    _db_instance = None
