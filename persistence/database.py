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
from datetime import datetime, timezone, timedelta, date as _date
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# Central source of truth for "is the regular session over?". Save-side
# and read-side daily-bar guards both consult this so their decision can
# never drift.
from trading.trading_hours import is_regular_session_closed, today_et

logger = logging.getLogger(__name__)

# Singleton instance
_db_instance: Optional['Database'] = None


def _drop_today_provisional_bars(
    bars: List[Dict[str, Any]],
    now_et: Optional[datetime] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Strip rows whose `date == today_et` when the regular session hasn't
    closed yet (using a 16:15 ET grace window for settlement).

    Alpaca's daily-bar endpoint, queried during market hours, returns a bar
    whose `close` is the last trade price at that moment — NOT the
    end-of-session close. Persisting such a row would pollute `prev_close`
    reads the next morning (bug seen 2026-04-22 → 2026-04-23: BMNZ's 4/22
    close came back as $13.66 mid-day vs the real final of $13.43, which
    shifted live's 4/23 gap_pct from BT's 5.96% to 4.17% and mis-quintiled
    the trade).

    Weekends and post-16:15-ET hours: all rows pass through unchanged.

    Returns (kept_bars, n_dropped). Pass `now_et` for deterministic tests.
    """
    if not bars:
        return bars, 0
    if is_regular_session_closed(now_et=now_et):
        return bars, 0
    today_str = today_et(now_et=now_et).isoformat()
    out: List[Dict[str, Any]] = []
    dropped = 0
    for b in bars:
        bdate = b.get('date')
        if isinstance(bdate, datetime):
            bdate_str = bdate.date().isoformat()
        elif isinstance(bdate, _date):
            bdate_str = bdate.isoformat()
        else:
            bdate_str = str(bdate)
        if bdate_str == today_str:
            dropped += 1
            continue
        out.append(b)
    return out, dropped


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

    Split into two physical databases:
    - cache_conn: Disposable data (bars, universe, volume profiles) — shareable
    - trades_conn: Precious data (trades, scan results, summaries) — per-node

    The public API is unchanged — all methods work transparently.
    For backward compatibility, passing a single db_path uses it for both.
    """

    def __init__(self, db_path: str = None, cache_path: str = None, trades_path: str = None):
        """
        Initialize database connections and create tables.

        Args:
            db_path: Legacy single-DB path (used for both if cache/trades paths not set)
            cache_path: Path to cache database (bars, universe, profiles)
            trades_path: Path to trades database (trades, scans, summaries)
        """
        # Re-register converters to override any third-party (e.g., peewee) interference
        sqlite3.register_converter("TIMESTAMP", _convert_timestamp)
        sqlite3.register_adapter(datetime, _adapt_datetime_iso)

        # Resolve paths: explicit split > legacy single path > defaults
        if cache_path and trades_path:
            self._cache_path = Path(cache_path)
            self._trades_path = Path(trades_path)
        elif db_path:
            # Legacy: single DB for both (backward compatible)
            self._cache_path = Path(db_path)
            self._trades_path = Path(db_path)
        else:
            self._cache_path = Path("data/cache.db")
            self._trades_path = Path("data/trades.db")

        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._trades_path.parent.mkdir(parents=True, exist_ok=True)

        # Primary connection reference (for code that accesses self.conn directly)
        self.db_path = self._cache_path

        self._cache_conn = self._open_connection(self._cache_path)
        if self._cache_path == self._trades_path:
            # Same file — share the connection
            self._trades_conn = self._cache_conn
            self._split = False
        else:
            self._trades_conn = self._open_connection(self._trades_path)
            self._split = True

        # Legacy alias — some code accesses self.conn directly (e.g., batch_backtest)
        self.conn = self._cache_conn

        self._create_tables()
        self._migrate()

        if self._split:
            logger.info(f"Database initialized (split): cache={self._cache_path}, trades={self._trades_path}")
        else:
            logger.info(f"Database initialized: {self._cache_path}")

    @staticmethod
    def _open_connection(path: Path) -> sqlite3.Connection:
        """Open a SQLite connection with WAL mode and busy timeout."""
        conn = sqlite3.connect(
            str(path),
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            timeout=30,
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self) -> None:
        """Create all tables if they don't exist."""
        self._create_cache_tables()
        self._create_trades_tables()

    def _create_cache_tables(self) -> None:
        """Create cache tables (universe, bars, profiles)."""
        self._cache_conn.executescript("""
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

            CREATE INDEX IF NOT EXISTS idx_volume_profiles_symbol
                ON volume_profiles(symbol);
        """)
        self._cache_conn.commit()
        logger.debug("Cache tables verified/created")

    def _create_trades_tables(self) -> None:
        """Create trades tables (trades, scan_results, summaries)."""
        self._trades_conn.executescript("""
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

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date DATE NOT NULL,
                symbol VARCHAR(10) NOT NULL,
                side VARCHAR(4) NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss_price REAL NOT NULL,
                take_profit_price REAL NOT NULL,
                shares INTEGER NOT NULL,
                risk_per_share REAL NOT NULL,
                total_risk REAL NOT NULL,
                risk_reward_ratio REAL NOT NULL,
                order_id VARCHAR(64),
                order_status VARCHAR(20),
                fill_price REAL,
                filled_at TIMESTAMP,
                exit_price REAL,
                exit_reason VARCHAR(20),
                exited_at TIMESTAMP,
                pnl REAL,
                pnl_pct REAL,
                pattern_data TEXT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            );

            CREATE TABLE IF NOT EXISTS daily_trading_summary (
                trade_date DATE PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                gross_pnl REAL DEFAULT 0.0,
                patterns_detected INTEGER DEFAULT 0,
                patterns_traded INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_trades_date
                ON trades(trade_date);
            CREATE INDEX IF NOT EXISTS idx_trades_symbol
                ON trades(symbol, trade_date);
            CREATE INDEX IF NOT EXISTS idx_trades_order_id
                ON trades(order_id);
        """)
        self._trades_conn.commit()

        # Cache tables continued (daily_bars, intraday, news)
        self._cache_conn.executescript("""
            CREATE TABLE IF NOT EXISTS daily_bars (
                symbol VARCHAR(10) NOT NULL,
                bar_date DATE NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                fetched_at TIMESTAMP NOT NULL,
                PRIMARY KEY (symbol, bar_date)
            );

            CREATE INDEX IF NOT EXISTS idx_daily_bars_date
                ON daily_bars(bar_date);

            -- Sidecar for MID-DAY provisional daily bars. Only written by
            -- BT runs that opt in to `--include-today-provisional`. Live
            -- engines NEVER read this table. See save_daily_bars_provisional
            -- / get_daily_bars_cached(include_provisional=True).
            CREATE TABLE IF NOT EXISTS daily_bars_provisional (
                symbol VARCHAR(10) NOT NULL,
                bar_date DATE NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                fetched_at TIMESTAMP NOT NULL,
                PRIMARY KEY (symbol, bar_date)
            );

            CREATE TABLE IF NOT EXISTS intraday_bars_1min (
                symbol VARCHAR(10) NOT NULL,
                bar_date DATE NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            );

            CREATE INDEX IF NOT EXISTS idx_intraday_bars_symbol_date
                ON intraday_bars_1min(symbol, bar_date);

            CREATE TABLE IF NOT EXISTS news_cache (
                symbol VARCHAR(10) NOT NULL,
                news_date DATE NOT NULL,
                headline TEXT NOT NULL,
                catalyst INTEGER,
                reason VARCHAR(100),
                classified_at TIMESTAMP,
                halt INTEGER DEFAULT 0,
                UNIQUE(symbol, news_date, headline)
            );

            CREATE INDEX IF NOT EXISTS idx_news_cache_symbol_date
                ON news_cache(symbol, news_date);
        """)
        self._cache_conn.commit()

    def _migrate(self) -> None:
        """Run database migrations for schema changes on existing DBs."""
        # Migration 1: Add unique index on scan_results to prevent duplicate rows.
        try:
            if not self._trades_conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_scan_results_unique'"
            ).fetchone():
                # Remove duplicates: keep the row with the latest detected_at per group
                self._trades_conn.execute("""
                    DELETE FROM scan_results WHERE id NOT IN (
                        SELECT MAX(id) FROM scan_results
                        GROUP BY scan_date, symbol, phase, COALESCE(time_bucket, '')
                    )
                """)
                deleted = self._trades_conn.execute("SELECT changes()").fetchone()[0]
                if deleted > 0:
                    logger.info(f"Migration: removed {deleted} duplicate scan_results rows")

                self._trades_conn.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_scan_results_unique
                        ON scan_results(scan_date, symbol, phase, COALESCE(time_bucket, ''))
                """)
                self._trades_conn.commit()
                logger.info("Migration: added unique index on scan_results")
        except Exception as e:
            logger.warning(f"Migration 1 (scan_results unique index) failed (non-fatal): {e}")

        # Migration 2: Add filled_qty column to trades table for partial fill tracking.
        try:
            columns = [row[1] for row in self._trades_conn.execute("PRAGMA table_info(trades)").fetchall()]
            if 'filled_qty' not in columns:
                self._trades_conn.execute("ALTER TABLE trades ADD COLUMN filled_qty INTEGER")
                self._trades_conn.commit()
                logger.info("Migration: added filled_qty column to trades table")
        except Exception as e:
            logger.warning(f"Migration 2 (filled_qty column) failed (non-fatal): {e}")

        # Migration 3: Add real_stop_loss_price column for self-managed stops.
        # Stores the actual stop level monitored by StopMonitor (flag_low region),
        # distinct from the safety-net SL on the Alpaca bracket (entry * 0.95).
        try:
            columns = [row[1] for row in self._trades_conn.execute("PRAGMA table_info(trades)").fetchall()]
            if 'real_stop_loss_price' not in columns:
                self._trades_conn.execute("ALTER TABLE trades ADD COLUMN real_stop_loss_price REAL")
                self._trades_conn.commit()
                logger.info("Migration: added real_stop_loss_price column to trades table")
        except Exception as e:
            logger.warning(f"Migration 3 (real_stop_loss_price column) failed (non-fatal): {e}")

        # Migration 4: Add partial exit columns for exhaustion exits.
        # Tracks partial sell into strength before final trail stop exit.
        partial_cols = {
            'partial_exit_price': 'REAL',
            'partial_exit_shares': 'INTEGER',
            'partial_exit_pnl': 'REAL',
            'partial_exit_reason': 'VARCHAR(20)',
            'partial_exited_at': 'TIMESTAMP',
        }
        try:
            columns = [row[1] for row in self._trades_conn.execute("PRAGMA table_info(trades)").fetchall()]
            added = []
            for col_name, col_type in partial_cols.items():
                if col_name not in columns:
                    self._trades_conn.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
                    added.append(col_name)
            if added:
                self._trades_conn.commit()
                logger.info(f"Migration 4: added partial exit columns to trades: {added}")
        except Exception as e:
            logger.warning(f"Migration 4 (partial exit columns) failed (non-fatal): {e}")

        # Migration 5: Add news classification columns for catalyst analysis.
        # Persists LLM classification with each trade for future backtesting
        # of news-based filters (data collection — no filtering yet).
        news_cols = {
            'news_catalyst': 'INTEGER',       # 1=catalyst, 0=noise, NULL=unknown
            'news_headline': 'TEXT',           # top headline at trade time
            'news_reason': 'VARCHAR(100)',     # LLM's reason for classification
        }
        try:
            columns = [row[1] for row in self._trades_conn.execute("PRAGMA table_info(trades)").fetchall()]
            added = []
            for col_name, col_type in news_cols.items():
                if col_name not in columns:
                    self._trades_conn.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
                    added.append(col_name)
            if added:
                self._trades_conn.commit()
                logger.info(f"Migration 5: added news columns to trades: {added}")
        except Exception as e:
            logger.warning(f"Migration 5 (news columns) failed (non-fatal): {e}")

        # Migration 6: Add exit microstructure columns for execution analysis.
        # Records bid/ask/depth at exit time, limit vs fill, latency, slippage.
        exit_micro_cols = {
            'exit_trigger_price': 'REAL',
            'exit_quote_bid': 'REAL',
            'exit_quote_ask': 'REAL',
            'exit_quote_bid_size': 'INTEGER',
            'exit_quote_ask_size': 'INTEGER',
            'exit_quote_spread': 'REAL',
            'exit_limit_price': 'REAL',
            'exit_pricing_method': 'VARCHAR(20)',
            'exit_submitted_at': 'TIMESTAMP',
            'exit_fill_latency_ms': 'REAL',
            'exit_slippage': 'REAL',
            'exit_ofi': 'REAL',
        }
        try:
            columns = [row[1] for row in self._trades_conn.execute("PRAGMA table_info(trades)").fetchall()]
            added = []
            for col_name, col_type in exit_micro_cols.items():
                if col_name not in columns:
                    self._trades_conn.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
                    added.append(col_name)
            if added:
                self._trades_conn.commit()
                logger.info(f"Migration 6: added exit microstructure columns: {added}")
        except Exception as e:
            logger.warning(f"Migration 6 (exit microstructure) failed (non-fatal): {e}")

        # Migration 7: Add entry microstructure columns for slippage analysis.
        # Records bid/ask/depth at buy-stop submission AND at fill time.
        entry_micro_cols = {
            'entry_quote_bid': 'REAL',
            'entry_quote_ask': 'REAL',
            'entry_quote_bid_size': 'INTEGER',
            'entry_quote_ask_size': 'INTEGER',
            'entry_quote_spread': 'REAL',
            'entry_quote_ofi': 'REAL',
            'entry_fill_quote_bid': 'REAL',
            'entry_fill_quote_ask': 'REAL',
        }
        try:
            columns = [row[1] for row in self._trades_conn.execute("PRAGMA table_info(trades)").fetchall()]
            added = []
            for col_name, col_type in entry_micro_cols.items():
                if col_name not in columns:
                    self._trades_conn.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
                    added.append(col_name)
            if added:
                self._trades_conn.commit()
                logger.info(f"Migration 7: added entry microstructure columns: {added}")
        except Exception as e:
            logger.warning(f"Migration 7 (entry microstructure) failed (non-fatal): {e}")

        # Migration 8: Add strategy column to trades table.
        # Distinguishes bull_flag trades from macd_wave trades.
        try:
            columns = [row[1] for row in self._trades_conn.execute("PRAGMA table_info(trades)").fetchall()]
            if 'strategy' not in columns:
                self._trades_conn.execute("ALTER TABLE trades ADD COLUMN strategy VARCHAR(20) DEFAULT 'bull_flag'")
                self._trades_conn.commit()
                logger.info("Migration 8: added strategy column to trades (default='bull_flag')")
        except Exception as e:
            logger.warning(f"Migration 8 (strategy column) failed (non-fatal): {e}")

        # Migration 9: Add L2 order book depth snapshot columns.
        # JSON blobs with 10 levels of bid/ask price+size at entry/exit time.
        # Source: Databento ITCH/PITCH feed across 4 exchanges.
        try:
            columns = [row[1] for row in self._trades_conn.execute("PRAGMA table_info(trades)").fetchall()]
            added = []
            for col in ['entry_l2_depth', 'exit_l2_depth']:
                if col not in columns:
                    self._trades_conn.execute(f"ALTER TABLE trades ADD COLUMN {col} TEXT")
                    added.append(col)
            if added:
                self._trades_conn.commit()
                logger.info(f"Migration 9: added L2 depth columns: {added}")
        except Exception as e:
            logger.warning(f"Migration 9 (L2 depth columns) failed (non-fatal): {e}")

        # Migration 10: Entry pipeline timing + reference price for slippage breakdown.
        # Decomposes slippage into: wait (bar close → loop processes) + quote RTT +
        # submit RTT + fill RTT. Reference price = last 1-min bar close when signal fired.
        slippage_cols = {
            'bar_close_price': 'REAL',      # close of the confirmation bar (ideal reference)
            'bar_close_at': 'TIMESTAMP',    # timestamp of that bar
            'loop_processed_at': 'TIMESTAMP',   # when engine finished MACD compute
            'quote_fetched_at': 'TIMESTAMP',    # when L1 quote came back
            'order_submitted_at': 'TIMESTAMP',  # when submit_bracket_order returned
            'order_filled_at': 'TIMESTAMP',     # when fill status first observed
            'bar_close_to_loop_ms': 'INTEGER',
            'loop_to_quote_ms': 'INTEGER',
            'quote_to_submit_ms': 'INTEGER',
            'submit_to_fill_ms': 'INTEGER',
            'drift_bar_to_ask_bps': 'REAL',   # (ask - bar_close) / bar_close * 10000
            'drift_bar_to_fill_bps': 'REAL',  # (fill - bar_close) / bar_close * 10000
            'drift_ask_to_fill_bps': 'REAL',  # (fill - ask) / ask * 10000 (should be ≤ 0)
        }
        try:
            columns = [row[1] for row in self._trades_conn.execute("PRAGMA table_info(trades)").fetchall()]
            added = []
            for col_name, col_type in slippage_cols.items():
                if col_name not in columns:
                    self._trades_conn.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
                    added.append(col_name)
            if added:
                self._trades_conn.commit()
                logger.info(f"Migration 10: added slippage timing columns: {added}")
        except Exception as e:
            logger.warning(f"Migration 10 (slippage timing) failed (non-fatal): {e}")

        # Migration 12: Add is_marginable + marginability_checked_at to universe.
        # Populated by trading_engine.TradingEngine when LIVE calls
        # alpaca.is_marginable(symbol) for the first time per symbol per day.
        # Read by backtest.BacktestRunner during risk-tier sizing — when LIVE has
        # observed the symbol is non-marginable, BT also downgrades to 1.0x.
        # Without this, BT systematically overstates P&L on non-marginable
        # micro-caps (OPTX 4/13: BT applied 2.0x risk → 10,303 shares; LIVE
        # downgraded to 1.0x → 5,799 shares; same trade, 2x P&L gap).
        # Default NULL means "unknown" — BT fails open (full risk_tier) until
        # LIVE has populated the field. Over time the universe table converges.
        try:
            cols = [row[1] for row in self._cache_conn.execute("PRAGMA table_info(universe)").fetchall()]
            added = []
            if 'is_marginable' not in cols:
                self._cache_conn.execute("ALTER TABLE universe ADD COLUMN is_marginable INTEGER DEFAULT NULL")
                added.append('is_marginable')
            if 'marginability_checked_at' not in cols:
                self._cache_conn.execute("ALTER TABLE universe ADD COLUMN marginability_checked_at TIMESTAMP DEFAULT NULL")
                added.append('marginability_checked_at')
            if added:
                self._cache_conn.commit()
                logger.info(f"Migration 12: added marginability columns to universe: {added}")
        except Exception as e:
            logger.warning(f"Migration 12 (universe marginability) failed (non-fatal): {e}")

        # Migration 11: Add halt column to news_cache for halt-aware entry filter.
        # Populated by NewsProvider.classify_news as a side effect of regex-matching
        # halt vocabulary in headlines. Read by NewsProvider.is_halted_today via
        # an indexed point-query — sub-ms at entry path. Drives the MACD wave
        # halt-aware filter (4/28 HTCO incident: Stock Now Up 360.18%, Halted
        # On Circuit Breaker To The Upside → entered after collapse, lost $3.5K).
        try:
            cols = [row[1] for row in self._cache_conn.execute("PRAGMA table_info(news_cache)").fetchall()]
            if 'halt' not in cols:
                self._cache_conn.execute("ALTER TABLE news_cache ADD COLUMN halt INTEGER DEFAULT 0")
                self._cache_conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_news_cache_halt "
                    "ON news_cache(symbol, news_date, halt)"
                )
                self._cache_conn.commit()
                logger.info("Migration 11: added halt column + index to news_cache")
        except Exception as e:
            logger.warning(f"Migration 11 (news_cache halt column) failed (non-fatal): {e}")

        # Migration 13: Add reject_reason column to trades table.
        # Captures Alpaca's reject reason (when present) on rejected/cancelled
        # OrderStream events so post-mortems don't have to grep journalctl or
        # query Alpaca's REST API to discover why an order failed. The TTGT
        # rejection on 2026-05-08 took ~30 minutes of REST archaeology because
        # this column didn't exist. With it: SELECT reject_reason FROM trades
        # WHERE order_status='rejected' AND trade_date=?.
        # NOTE: Alpaca's REST GET /orders/{id} does NOT include reject_reason
        # — it's only delivered via the OrderStream trade-update event. So
        # capture happens in trading/order_stream.py:_on_trade_update.
        try:
            columns = [row[1] for row in self._trades_conn.execute(
                "PRAGMA table_info(trades)").fetchall()]
            if 'reject_reason' not in columns:
                self._trades_conn.execute(
                    "ALTER TABLE trades ADD COLUMN reject_reason VARCHAR(100)"
                )
                self._trades_conn.commit()
                logger.info("Migration 13: added reject_reason column to trades")
        except Exception as e:
            logger.warning(f"Migration 13 (reject_reason column) failed (non-fatal): {e}")

        # Migration 14: ORB winner-stack scale-out columns (2026-08-22).
        # One row = one trade; a 40%@+3R scale-out books its leg into these
        # nullable columns while the row STAYS OPEN (order_status unchanged,
        # pnl stays NULL); the final exit writes `pnl` exactly ONCE as
        # scale_pnl + runner-leg P&L (P0-2/P0-3 single-writer rule in
        # docs/orb_winner_stack_design_aug2026.md). `filled_qty` keeps the
        # ENTRY qty forever. absent/NULL scaled_at = no scale happened.
        #
        # Deliberately a SEPARATE family from Migration 4's partial_exit_*
        # (review P1-4, justified): the exhaustion columns carry bull-flag
        # semantics — the partial books into daily P&L at partial time and
        # exit_reason composes as 'exhaust+<reason>' — while the ORB scale is
        # realized-at-close with the runner's plain exit_reason, and its
        # presence drives flag-INDEPENDENT rehydration (shares -= scale_qty,
        # scale_done restore — review P1-7). Overloading partial_exit_reason
        # to discriminate two lifecycles would force every report consumer to
        # branch on the reason string; a dedicated family keeps each
        # convention single-meaning. Report consumers of scale_*:
        # scripts/report_common.py (_row_pnl, cumulative_orb_since,
        # sizing_attribution) + orb_engine (_handle_exit_event,
        # _sync_db_after_fc, sync_positions rehydration).
        scale_cols = {
            'scale_qty': 'INTEGER',    # shares sold in the scale leg
            'scale_price': 'REAL',     # avg fill price of the scale leg
            'scale_pnl': 'REAL',       # (scale_price - fill_price) * scale_qty
            'scaled_at': 'TIMESTAMP',  # scale-leg fill time (NULL = no scale)
        }
        try:
            columns = [row[1] for row in self._trades_conn.execute(
                "PRAGMA table_info(trades)").fetchall()]
            added = []
            for col_name, col_type in scale_cols.items():
                if col_name not in columns:
                    self._trades_conn.execute(
                        f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
                    added.append(col_name)
            if added:
                self._trades_conn.commit()
                logger.info(f"Migration 14: added ORB scale-out columns to "
                            f"trades: {added}")
        except Exception as e:
            logger.warning(f"Migration 14 (scale-out columns) failed "
                           f"(non-fatal): {e}")

    # =========================================================================
    # News cache (halt detection + per-article classification)
    # =========================================================================

    def save_news_classification(
        self, symbol: str, news_date: str, headline: str,
        catalyst: Optional[bool], reason: str,
        halt: bool = False,
    ) -> None:
        """Upsert a per-article classification row in news_cache.

        Args:
            symbol: Stock symbol.
            news_date: ISO date string (YYYY-MM-DD) for the article publication date.
            headline: Article headline (truncated to 500 chars to match TEXT capacity guidance).
            catalyst: LLM classification — True (catalyst), False (noise), None (unknown).
            reason: LLM's reason / category short-label.
            halt: True if headline matched halt-vocabulary regex (separate from catalyst).
        """
        try:
            self._cache_conn.execute(
                "INSERT OR REPLACE INTO news_cache "
                "(symbol, news_date, headline, catalyst, reason, classified_at, halt) "
                "VALUES (?, ?, ?, ?, ?, datetime('now'), ?)",
                (
                    symbol, news_date, (headline or '')[:500],
                    (1 if catalyst else 0) if catalyst is not None else None,
                    (reason or '')[:100],
                    1 if halt else 0,
                ),
            )
            self._cache_conn.commit()
        except Exception as e:
            logger.warning(f"save_news_classification({symbol}, {news_date}) failed: {e}")

    def is_halted_today(self, symbol: str, today: Optional[str] = None) -> Tuple[bool, str]:
        """Return (halt_detected, matched_headline) for the symbol on `today`.

        Per-day semantics: a halt headline persists in news for the rest of the
        day. Sub-ms point-query on (symbol, news_date, halt) index.

        Fail-open: any DB error returns (False, "") with a warning logged —
        we never block legitimate trades on infrastructure issues.

        Args:
            symbol: Stock symbol.
            today: ISO date string (YYYY-MM-DD). Defaults to UTC today.
                   Callers may pass an ET-derived date for boundary clarity.
        """
        if today is None:
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        try:
            row = self._cache_conn.execute(
                "SELECT headline FROM news_cache "
                "WHERE symbol = ? AND news_date = ? AND halt = 1 LIMIT 1",
                (symbol, today),
            ).fetchone()
            if row:
                return True, row[0] or ""
            return False, ""
        except Exception as e:
            logger.warning(f"is_halted_today({symbol}, {today}) failed (fail-open): {e}")
            return False, ""

    # =========================================================================
    # Universe operations
    # =========================================================================

    def upsert_universe_stock(self, stock: Dict[str, Any]) -> None:
        """
        Insert or update a stock in the universe table.

        Args:
            stock: Dict with keys matching universe columns
        """
        self._cache_conn.execute("""
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
        self._cache_conn.commit()

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

        self._cache_conn.executemany("""
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
        self._cache_conn.commit()
        return len(stocks)

    def get_active_universe(self) -> List[Dict[str, Any]]:
        """Get all active stocks in the universe."""
        cursor = self._cache_conn.execute(
            "SELECT * FROM universe WHERE active = 1 ORDER BY symbol"
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_universe_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get a single stock from the universe."""
        cursor = self._cache_conn.execute(
            "SELECT * FROM universe WHERE symbol = ?", (symbol,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_marginability(self, symbol: str) -> Optional[bool]:
        """Read cached marginability for a symbol (None = unknown).

        Populated by LIVE's `alpaca.is_marginable` call (see
        trading.trading_engine._check_marginability_with_persist). BT reads
        this to mirror LIVE's risk-tier downgrade on non-marginable stocks.

        Returns:
            True/False if LIVE has observed the symbol's marginability,
            None if never checked (BT should fail open = use full risk_tier).
        """
        try:
            row = self._cache_conn.execute(
                "SELECT is_marginable FROM universe WHERE symbol = ?",
                (symbol,),
            ).fetchone()
            if row is None or row[0] is None:
                return None
            return bool(row[0])
        except Exception as e:
            logger.warning(
                f"get_marginability({symbol}) failed: {e} — returning None"
            )
            return None

    def set_marginability(self, symbol: str, is_marginable: bool) -> None:
        """Persist a marginability observation for a symbol.

        Called from LIVE on first successful is_marginable() return per
        symbol per day. The universe row must already exist (universe is
        prebuilt via main.py --rebuild-universe). On a missing row we
        upsert just the marginability columns to avoid losing the
        observation, but log a warning.
        """
        now = datetime.now(timezone.utc)
        try:
            cursor = self._cache_conn.execute(
                "UPDATE universe SET is_marginable = ?, "
                "marginability_checked_at = ? WHERE symbol = ?",
                (1 if is_marginable else 0, now, symbol),
            )
            if cursor.rowcount == 0:
                logger.warning(
                    f"set_marginability({symbol}, {is_marginable}) — "
                    f"symbol not in universe; inserting minimal row"
                )
                self._cache_conn.execute(
                    "INSERT INTO universe (symbol, is_marginable, "
                    "marginability_checked_at, active) VALUES (?, ?, ?, 1)",
                    (symbol, 1 if is_marginable else 0, now),
                )
            self._cache_conn.commit()
        except Exception as e:
            logger.error(
                f"set_marginability({symbol}, {is_marginable}) failed: {e}"
            )

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
        cursor = self._cache_conn.execute(
            f"UPDATE universe SET active = 0, last_updated = ? WHERE symbol IN ({placeholders})",
            [datetime.now(timezone.utc)] + symbols
        )
        self._cache_conn.commit()
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
        cursor = self._cache_conn.execute("""
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
        self._cache_conn.execute("""
            UPDATE universe SET float_shares = ?, float_updated_at = ?, last_updated = ?
            WHERE symbol = ?
        """, (float_shares, now, now, symbol))
        self._cache_conn.commit()

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

        self._cache_conn.executemany("""
            INSERT INTO volume_profiles (symbol, time_bucket, avg_volume, last_updated)
            VALUES (:symbol, :time_bucket, :avg_volume, :last_updated)
            ON CONFLICT(symbol, time_bucket) DO UPDATE SET
                avg_volume = excluded.avg_volume,
                last_updated = excluded.last_updated
        """, profiles)
        self._cache_conn.commit()
        return len(profiles)

    def get_volume_profile(self, symbol: str) -> Dict[str, int]:
        """
        Get volume profile for a symbol.

        Returns:
            Dict mapping time_bucket -> avg_volume (e.g., {'09:30': 50000, ...})
        """
        cursor = self._cache_conn.execute(
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
        cursor = self._cache_conn.execute(
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
        self._cache_conn.execute("DELETE FROM volume_profiles WHERE symbol = ?", (symbol,))
        self._cache_conn.commit()

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
        cursor = self._trades_conn.execute("""
            INSERT OR REPLACE INTO scan_results (scan_date, symbol, detected_at, phase,
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
        self._trades_conn.commit()
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
            cursor = self._trades_conn.execute(
                "SELECT * FROM scan_results WHERE scan_date = ? AND phase = ? ORDER BY detected_at",
                (scan_date, phase)
            )
        else:
            cursor = self._trades_conn.execute(
                "SELECT * FROM scan_results WHERE scan_date = ? ORDER BY detected_at",
                (scan_date,)
            )
        return [dict(row) for row in cursor.fetchall()]

    def get_premarket_gap_symbols(self, scan_date: str) -> List[str]:
        """Get symbols that had a pre-market gap on the given date."""
        cursor = self._trades_conn.execute("""
            SELECT DISTINCT symbol FROM scan_results
            WHERE scan_date = ? AND phase = 'premarket' AND gap_pct >= 2.0
            ORDER BY symbol
        """, (scan_date,))
        return [row['symbol'] for row in cursor.fetchall()]

    # =========================================================================
    # Trade operations
    # =========================================================================

    def save_trade(self, trade: Dict[str, Any]) -> int:
        """
        Save a trade record.

        Args:
            trade: Dict with trade data matching trades table columns

        Returns:
            ID of the inserted row
        """
        now = datetime.now(timezone.utc)
        if 'strategy' not in trade:
            logger.warning("save_trade called without explicit strategy — defaulting to 'bull_flag'")
        trade.setdefault('strategy', 'bull_flag')
        trade.setdefault('created_at', now)
        trade.setdefault('updated_at', now)
        cursor = self._trades_conn.execute("""
            INSERT INTO trades (trade_date, symbol, side, entry_price,
                               stop_loss_price, take_profit_price, shares,
                               risk_per_share, total_risk, risk_reward_ratio,
                               order_id, order_status, fill_price, filled_at,
                               exit_price, exit_reason, exited_at,
                               pnl, pnl_pct, pattern_data,
                               strategy, created_at, updated_at)
            VALUES (:trade_date, :symbol, :side, :entry_price,
                    :stop_loss_price, :take_profit_price, :shares,
                    :risk_per_share, :total_risk, :risk_reward_ratio,
                    :order_id, :order_status, :fill_price, :filled_at,
                    :exit_price, :exit_reason, :exited_at,
                    :pnl, :pnl_pct, :pattern_data,
                    :strategy, :created_at, :updated_at)
        """, trade)
        self._trades_conn.commit()
        logger.info(f"Saved trade: {trade['symbol']} {trade['side']} "
                     f"{trade['shares']} shares @ ${trade['entry_price']:.2f}")
        return cursor.lastrowid

    def update_trade(self, trade_id: int, updates: Dict[str, Any]) -> None:
        """
        Update a trade record.

        Args:
            trade_id: ID of the trade to update
            updates: Dict of column->value pairs to update
        """
        updates['updated_at'] = datetime.now(timezone.utc)
        set_clause = ', '.join(f"{k} = :{k}" for k in updates)
        updates['id'] = trade_id
        self._trades_conn.execute(
            f"UPDATE trades SET {set_clause} WHERE id = :id", updates
        )
        self._trades_conn.commit()
        logger.debug(f"Updated trade {trade_id}: {list(updates.keys())}")

    def get_trade_by_order_id(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get a trade by its Alpaca order ID."""
        cursor = self._trades_conn.execute(
            "SELECT * FROM trades WHERE order_id = ?", (order_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_trades_by_date(self, trade_date: str) -> List[Dict[str, Any]]:
        """
        Get all trades for a given date.

        Args:
            trade_date: Date string (YYYY-MM-DD)

        Returns:
            List of trade dicts
        """
        cursor = self._trades_conn.execute(
            "SELECT * FROM trades WHERE trade_date = ? ORDER BY created_at",
            (trade_date,)
        )
        return [dict(row) for row in cursor.fetchall()]

    # Statuses that indicate an order is actively in flight or holding a live
    # position. Anything not in this set (cancelled, canceled, rejected,
    # expired, time_stop_canceled, closed, stale_closed, …) is treated as
    # terminal — see TRT cross-strategy lock incident 2026-05-15 where an
    # ORB time_stop_canceled row blocked bull_flag re-entry because the
    # old denylist only excluded 'cancelled'.
    #
    # 'exit_pending_verification' (2026-06-05): added after the SMU/QBTZ
    # orphan incident — when StopMonitor's BRANCH_LAST_RESORT cannot
    # confirm an exit fill, the trade is left in this state instead of
    # being marked 'closed' with a fake exit_price. Stays in
    # get_open_trades() so the orphan reconciler can retry.
    _ACTIVE_ORDER_STATUSES = (
        'filled', 'partially_filled', 'pending_new', 'accepted', 'new',
        'exit_pending_verification',
    )

    def get_strategy_trades_in_window(
        self, strategy: str, since_date: str,
        symbols: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """All rows for ``strategy`` on/after ``since_date``, optionally
        filtered to ``symbols``.

        Returns rows regardless of order_status or exit state — used by the
        orphan reconciler, which needs to see poisoned-but-marked-exited
        rows (exit_price set, exit_reason='stop_loss_unconfirmed') that
        get_open_trades would filter out.

        Public alternative to the private _trades_conn.execute pattern;
        gives tests a clean seam to mock.
        """
        if not symbols:
            cursor = self._trades_conn.execute(
                "SELECT * FROM trades "
                "WHERE strategy = ? AND trade_date >= ? "
                "ORDER BY trade_date DESC, id DESC",
                (strategy, since_date),
            )
        else:
            placeholders = ','.join('?' for _ in symbols)
            cursor = self._trades_conn.execute(
                f"SELECT * FROM trades "
                f"WHERE strategy = ? AND trade_date >= ? "
                f"AND symbol IN ({placeholders}) "
                f"ORDER BY trade_date DESC, id DESC",
                (strategy, since_date, *symbols),
            )
        return [dict(row) for row in cursor.fetchall()]

    def get_open_trades(self, trade_date: str, strategy: str = None) -> List[Dict[str, Any]]:
        """
        Get trades that are still open (no exit) for a given date.

        "Open" means: exit_price not yet set AND order_status is one of the
        active in-flight states (see ``_ACTIVE_ORDER_STATUSES``). Terminal
        statuses such as 'cancelled', 'rejected', 'time_stop_canceled' are
        excluded, even when the row carries no exit_price.

        Args:
            trade_date: Date string (YYYY-MM-DD)
            strategy: Optional strategy filter ('bull_flag', 'macd_wave', 'orb').
                      If None, returns all strategies (backward compatible).

        Returns:
            List of open trade dicts
        """
        placeholders = ','.join('?' for _ in self._ACTIVE_ORDER_STATUSES)
        if strategy:
            cursor = self._trades_conn.execute(
                f"SELECT * FROM trades WHERE trade_date = ? AND exit_price IS NULL "
                f"AND order_status IN ({placeholders}) AND strategy = ? "
                f"ORDER BY created_at",
                (trade_date, *self._ACTIVE_ORDER_STATUSES, strategy)
            )
        else:
            cursor = self._trades_conn.execute(
                f"SELECT * FROM trades WHERE trade_date = ? AND exit_price IS NULL "
                f"AND order_status IN ({placeholders}) ORDER BY created_at",
                (trade_date, *self._ACTIVE_ORDER_STATUSES)
            )
        return [dict(row) for row in cursor.fetchall()]

    def get_daily_pnl(self, trade_date: str) -> float:
        """
        Get total realized P&L for a given date.

        Args:
            trade_date: Date string (YYYY-MM-DD)

        Returns:
            Total P&L in dollars
        """
        cursor = self._trades_conn.execute(
            "SELECT COALESCE(SUM(pnl), 0.0) FROM trades WHERE trade_date = ? AND pnl IS NOT NULL",
            (trade_date,)
        )
        return float(cursor.fetchone()[0])

    def save_daily_summary(self, summary: Dict[str, Any]) -> None:
        """
        Save or update daily trading summary.

        Args:
            summary: Dict with trade_date and summary stats
        """
        self._trades_conn.execute("""
            INSERT INTO daily_trading_summary
                (trade_date, total_trades, winning_trades, losing_trades,
                 gross_pnl, patterns_detected, patterns_traded)
            VALUES (:trade_date, :total_trades, :winning_trades, :losing_trades,
                    :gross_pnl, :patterns_detected, :patterns_traded)
            ON CONFLICT(trade_date) DO UPDATE SET
                total_trades = excluded.total_trades,
                winning_trades = excluded.winning_trades,
                losing_trades = excluded.losing_trades,
                gross_pnl = excluded.gross_pnl,
                patterns_detected = excluded.patterns_detected,
                patterns_traded = excluded.patterns_traded
        """, summary)
        self._trades_conn.commit()
        logger.info(f"Saved daily summary for {summary['trade_date']}: "
                     f"{summary['total_trades']} trades, P&L: ${summary['gross_pnl']:.2f}")

    def get_daily_summary(self, trade_date: str) -> Optional[Dict[str, Any]]:
        """Get daily trading summary for a date."""
        cursor = self._trades_conn.execute(
            "SELECT * FROM daily_trading_summary WHERE trade_date = ?",
            (trade_date,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    # =========================================================================
    # Utility
    # =========================================================================

    def get_universe_count(self) -> int:
        """Get count of active stocks in universe."""
        cursor = self._cache_conn.execute("SELECT COUNT(*) FROM universe WHERE active = 1")
        return cursor.fetchone()[0]

    def get_volume_profile_count(self) -> int:
        """Get count of unique symbols with volume profiles."""
        cursor = self._cache_conn.execute("SELECT COUNT(DISTINCT symbol) FROM volume_profiles")
        return cursor.fetchone()[0]

    # =========================================================================
    # Daily bars cache
    # =========================================================================

    def save_daily_bars(
        self, bars: List[Dict[str, Any]],
        now_et: Optional[datetime] = None,
    ) -> int:
        """
        Cache daily bars to avoid re-fetching from API. INSERT OR REPLACE
        semantics — a previous row for (symbol, date) is overwritten.

        Silently drops rows whose `date == today_et` when the regular
        session hasn't ended (before 16:15 ET). See
        `_drop_today_provisional_bars` for background. Pass `now_et` to
        override the clock for tests.

        Args:
            bars: List of dicts with keys: symbol, date, open, high, low, close, volume
            now_et: Override 'now' for tests. Defaults to real ET wall clock.

        Returns:
            Number of bars actually saved (may be < len(bars) if some were
            dropped by the provisional-today guard).
        """
        if not bars:
            return 0

        bars, n_dropped = _drop_today_provisional_bars(bars, now_et=now_et)
        if n_dropped:
            logger.warning(
                f"save_daily_bars: dropped {n_dropped} provisional row(s) "
                f"for today — market still open (write post-16:15 ET to "
                f"persist today's bar). See _drop_today_provisional_bars."
            )
        if not bars:
            return 0

        now = datetime.now(timezone.utc)
        self._cache_conn.executemany("""
            INSERT OR REPLACE INTO daily_bars
                (symbol, bar_date, open, high, low, close, volume, fetched_at)
            VALUES (:symbol, :date, :open, :high, :low, :close, :volume, :fetched_at)
        """, [{**b, 'fetched_at': now} for b in bars])
        self._cache_conn.commit()
        logger.info(f"Cached {len(bars)} daily bars")
        return len(bars)

    def save_daily_bars_provisional(self, bars: List[Dict[str, Any]]) -> int:
        """Write mid-day provisional daily bars to the sidecar table.

        This is the escape hatch for BT runs that want to see today's
        trades without polluting the main `daily_bars` cache that live
        reads. No time-of-day guard — callers opt in explicitly.

        Auto-cleanup: BEFORE each write we DELETE any provisional rows
        whose `bar_date < today_et`. Rationale: this table is only ever
        meant to carry TODAY's still-open bar; anything older is stale
        from a prior mid-day run that forgot (or couldn't) call
        `clear_provisional_daily_bars()` on its way out. Deleting here
        removes the caller-contract footgun and keeps the table small.
        `clear_provisional_daily_bars()` still exists for a full wipe
        (BT always calls it at run start to clear today's leftover too).

        Args:
            bars: list of dicts with keys symbol, date, open, high, low, close, volume

        Returns:
            number of rows written
        """
        if not bars:
            return 0
        # Sweep stale (pre-today) rows so a previous run's leftover can't
        # linger after the next day's boundary.
        today_str = today_et().isoformat()
        cur = self._cache_conn.execute(
            "DELETE FROM daily_bars_provisional WHERE bar_date < ?",
            (today_str,),
        )
        swept = cur.rowcount or 0
        if swept:
            logger.info(
                f"save_daily_bars_provisional: swept {swept} stale pre-today "
                f"row(s) (< {today_str}) before insert"
            )

        now = datetime.now(timezone.utc)
        self._cache_conn.executemany("""
            INSERT OR REPLACE INTO daily_bars_provisional
                (symbol, bar_date, open, high, low, close, volume, fetched_at)
            VALUES (:symbol, :date, :open, :high, :low, :close, :volume, :fetched_at)
        """, [{**b, 'fetched_at': now} for b in bars])
        self._cache_conn.commit()
        logger.info(f"Cached {len(bars)} PROVISIONAL daily bars (sidecar)")
        return len(bars)

    def clear_provisional_daily_bars(self) -> int:
        """Wipe the provisional sidecar. Call at the start of a BT run so
        stale mid-day data from a previous run doesn't leak into today's
        features extraction.
        """
        cur = self._cache_conn.execute("DELETE FROM daily_bars_provisional")
        n = cur.rowcount or 0
        self._cache_conn.commit()
        if n:
            logger.info(f"Cleared {n} provisional daily bars")
        return n

    def get_daily_bars_cached(
        self, symbols: List[str], start_date: str, end_date: str,
        include_provisional: bool = False,
        now_et: Optional[datetime] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve cached daily bars for symbols in a date range.

        Args:
            symbols: List of stock symbols
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            include_provisional: If True, also union in today's mid-day
                provisional rows from `daily_bars_provisional`. Finals
                always win over provisional for the same (sym, date) —
                a FINAL row hides any leftover provisional for the same
                key. Live engines should NEVER set this to True; only BT
                flows that explicitly opt in via `--include-today-provisional`.
            now_et: override clock for tests.

        Defensive read-side guard: during market hours (before 16:15 ET)
        we treat ANY today-row in the main `daily_bars` table as
        non-final and drop it from the result. This protects against
        legacy polluted rows that existed before the save-side guard
        shipped, or rows written via paths that bypass
        `save_daily_bars`. Cost: live queries use `end_date = today - 1`
        and are never affected by this filter; BT queries that want
        today's data opt in via `include_provisional=True` and get the
        sidecar row instead.

        Returns:
            Dict mapping symbol -> list of bar dicts {date, open, high, low, close, volume}
        """
        if not symbols:
            return {}

        # Compute the "treat this date as provisional" predicate for the
        # read-side guard. Only applies when the regular session hasn't
        # closed yet.
        today_drop = (
            today_et(now_et=now_et).isoformat()
            if not is_regular_session_closed(now_et=now_et)
            else None
        )

        # Step 1: collect final bars into a per-symbol dict keyed by bar_date
        # so it's easy to layer provisional rows on top.
        per_sym: Dict[str, Dict[Any, Dict[str, Any]]] = {}
        chunk_size = 500  # SQLite parameter limit
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            placeholders = ','.join('?' * len(chunk))
            cursor = self._cache_conn.execute(f"""
                SELECT symbol, bar_date, open, high, low, close, volume
                FROM daily_bars
                WHERE symbol IN ({placeholders})
                  AND bar_date >= ? AND bar_date <= ?
                ORDER BY symbol, bar_date
            """, chunk + [start_date, end_date])
            for row in cursor.fetchall():
                rd = dict(row)
                # Defensive: skip today-rows during market hours. Legacy
                # polluted rows or bypassed-guard writes won't leak out.
                if today_drop and str(rd['bar_date']) == today_drop:
                    continue
                per_sym.setdefault(rd['symbol'], {})[rd['bar_date']] = {
                    'date': rd['bar_date'],
                    'open': rd['open'], 'high': rd['high'],
                    'low': rd['low'], 'close': rd['close'],
                    'volume': rd['volume'],
                }

        # Step 2: if opted-in, layer provisional rows but DO NOT overwrite a
        # final row. Final always wins.
        if include_provisional:
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i:i + chunk_size]
                placeholders = ','.join('?' * len(chunk))
                cursor = self._cache_conn.execute(f"""
                    SELECT symbol, bar_date, open, high, low, close, volume
                    FROM daily_bars_provisional
                    WHERE symbol IN ({placeholders})
                      AND bar_date >= ? AND bar_date <= ?
                    ORDER BY symbol, bar_date
                """, chunk + [start_date, end_date])
                for row in cursor.fetchall():
                    rd = dict(row)
                    bucket = per_sym.setdefault(rd['symbol'], {})
                    if rd['bar_date'] in bucket:
                        continue  # final already present — skip provisional
                    bucket[rd['bar_date']] = {
                        'date': rd['bar_date'],
                        'open': rd['open'], 'high': rd['high'],
                        'low': rd['low'], 'close': rd['close'],
                        'volume': rd['volume'],
                    }

        # Step 3: collapse to list-per-symbol ordered by date.
        results: Dict[str, List[Dict[str, Any]]] = {
            sym: [bars[d] for d in sorted(bars.keys())]
            for sym, bars in per_sym.items()
        }
        if results:
            suffix = " (+ provisional overlay)" if include_provisional else ""
            logger.info(
                f"Cache hit: {len(results)} symbols with daily bars "
                f"({start_date} to {end_date}){suffix}"
            )
        return results

    def get_cached_daily_bar_symbols(self, start_date: str, end_date: str) -> set:
        """
        Get set of symbols that have cached daily bars in the date range.

        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)

        Returns:
            Set of symbol strings with cached data
        """
        cursor = self._cache_conn.execute("""
            SELECT DISTINCT symbol FROM daily_bars
            WHERE bar_date >= ? AND bar_date <= ?
        """, (start_date, end_date))
        return {row['symbol'] for row in cursor.fetchall()}

    # =========================================================================
    # Intraday 1-min bars cache
    # =========================================================================

    def save_intraday_bars(self, symbol: str, bar_date: str, bars: List[Dict]) -> int:
        """
        Cache 1-min intraday bars for a symbol/date.

        Args:
            symbol: Stock symbol
            bar_date: Date string (YYYY-MM-DD)
            bars: List of dicts with keys: timestamp, open, high, low, close, volume

        Returns:
            Number of bars saved
        """
        if not bars:
            return 0

        rows = []
        for b in bars:
            ts = b['timestamp']
            # Convert pandas Timestamp to Python datetime if needed
            if hasattr(ts, 'to_pydatetime'):
                ts = ts.to_pydatetime()
            rows.append({
                'symbol': symbol,
                'bar_date': bar_date,
                'timestamp': ts,
                'open': float(b['open']),
                'high': float(b['high']),
                'low': float(b['low']),
                'close': float(b['close']),
                'volume': int(b['volume']),
            })

        self._cache_conn.executemany("""
            INSERT OR REPLACE INTO intraday_bars_1min
                (symbol, bar_date, timestamp, open, high, low, close, volume)
            VALUES (:symbol, :bar_date, :timestamp, :open, :high, :low, :close, :volume)
        """, rows)
        self._cache_conn.commit()
        logger.debug(f"Cached {len(rows)} intraday 1-min bars for {symbol} on {bar_date}")
        return len(rows)

    def get_intraday_bars_cached(self, symbol: str, bar_date: str) -> List[Dict]:
        """
        Retrieve cached 1-min intraday bars for a symbol/date.

        Args:
            symbol: Stock symbol
            bar_date: Date string (YYYY-MM-DD)

        Returns:
            List of bar dicts with keys: timestamp, open, high, low, close, volume
            Empty list if no cached data
        """
        cursor = self._cache_conn.execute("""
            SELECT timestamp, open, high, low, close, volume
            FROM intraday_bars_1min
            WHERE symbol = ? AND bar_date = ?
            ORDER BY timestamp
        """, (symbol, bar_date))

        rows = cursor.fetchall()
        if not rows:
            return []

        bars = [dict(row) for row in rows]
        logger.debug(f"Cache hit: {len(bars)} intraday bars for {symbol} on {bar_date}")
        return bars

    def get_intraday_bars_bulk(self, symbol_dates: list) -> Dict[tuple, List[Dict]]:
        """
        Bulk-load 1-min bars for multiple (symbol, date) pairs.

        Uses a single full-table scan with date range filter, then filters
        to requested pairs in Python. Much faster than N individual queries
        or N IN-clause queries for large batches.

        Args:
            symbol_dates: List of (symbol, date_str) tuples

        Returns:
            Dict mapping (symbol, date_str) -> list of bar dicts
        """
        if not symbol_dates:
            return {}

        # Build set of requested pairs for fast lookup
        requested = set(symbol_dates)

        # Find date range for efficient SQL filtering
        dates = [d for _, d in symbol_dates]
        min_date = min(dates)
        max_date = max(dates)

        cursor = self._cache_conn.execute("""
            SELECT symbol, bar_date, timestamp, open, high, low, close, volume
            FROM intraday_bars_1min
            WHERE bar_date >= ? AND bar_date <= ?
            ORDER BY symbol, bar_date, timestamp
        """, (min_date, max_date))

        result = {}
        current_key = None
        current_bars = []

        for row in cursor:
            bar_date_val = row['bar_date']
            if hasattr(bar_date_val, 'isoformat'):
                bar_date_val = bar_date_val.isoformat()
            key = (row['symbol'], str(bar_date_val))

            if key != current_key:
                # Flush previous group
                if current_key is not None and current_bars and current_key in requested:
                    result[current_key] = current_bars
                current_key = key
                current_bars = []

            # Only accumulate bars for requested pairs
            if key in requested:
                current_bars.append({
                    'timestamp': row['timestamp'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                })

        # Flush last group
        if current_key is not None and current_bars and current_key in requested:
            result[current_key] = current_bars

        logger.debug(f"Bulk loaded {len(result)} bar sets")
        return result

    def close(self) -> None:
        """Close all database connections."""
        if self._cache_conn:
            self._cache_conn.close()
        if self._split and self._trades_conn:
            self._trades_conn.close()
        logger.info("Database connections closed")


def get_database(
    db_path: str = None,
    cache_path: str = None,
    trades_path: str = None,
) -> Database:
    """
    Get or create the singleton Database instance.

    Args:
        db_path: Legacy single-DB path (backward compatible)
        cache_path: Path to cache database
        trades_path: Path to trades database

    Returns:
        Database singleton instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(
            db_path=db_path,
            cache_path=cache_path,
            trades_path=trades_path,
        )
    return _db_instance


def reset_database() -> None:
    """Reset the singleton (for testing)."""
    global _db_instance
    if _db_instance is not None:
        _db_instance.close()
    _db_instance = None
