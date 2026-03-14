"""
Monthly-chunked parallel backtest runner.

Splits a date range into monthly chunks and processes them in parallel
using ThreadPoolExecutor. Each thread gets its own Database and AlpacaClient
to avoid shared state issues.

Usage:
    runner = MonthlyBacktestRunner(max_workers=2, verbose=False)
    master_csv = runner.run_all(start, end, output_dir="backtest_results")
"""

import csv
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from backtest import BacktestResult, BacktestRunner
from batch_backtest import (
    CSV_HEADERS,
    fetch_daily_bars_cached,
    find_big_movers,
    get_1min_bars_cached,
    run_batch_backtest,
    run_batch_backtest_parallel,
    utc_to_et_str,
    write_csv_report,
)
from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rich CSV columns — extends the base CSV with pattern, plan, and daily data
# ---------------------------------------------------------------------------

RICH_CSV_HEADERS = [
    # Base trade fields
    "symbol", "date", "entry_time_et", "entry_price", "stop_loss",
    "target", "shares", "exit_time_et", "exit_price", "exit_reason",
    "pnl", "pnl_pct",
    # Trade details
    "bars_held", "risk_per_share", "reward_per_share", "risk_reward_ratio",
    "total_risk",
    # Pattern metrics
    "pole_gain_pct", "retracement_pct", "pullback_candles",
    "avg_pole_volume", "avg_flag_volume", "pole_height",
    "flag_low", "flag_high",
    # Entry bar OHLCV — the actual 1-min bar where entry was triggered
    "entry_bar_open", "entry_bar_high", "entry_bar_low",
    "entry_bar_close", "entry_bar_volume",
    # Daily bar context
    "day_open", "day_high", "day_low", "day_close", "day_volume",
    "intraday_move_pct",
    # Volume analysis
    "avg_volume_daily", "relative_volume",
    # Universe metadata
    "sector", "float_shares",
    # Derived
    "entry_minutes_from_open", "patterns_detected_that_day",
]


@dataclass
class MonthResult:
    """Result from processing a single month."""

    month_label: str
    csv_path: str
    num_movers: int
    num_trades: int
    total_pnl: float
    elapsed_seconds: float
    results: List[BacktestResult]


def build_rich_row(
    trade,
    result: BacktestResult,
    daily_bars_dict: Dict,
    universe_dict: Dict,
) -> list:
    """
    Build a single rich CSV row from a trade and its context.

    Args:
        trade: SimulatedTrade instance
        result: BacktestResult containing the trade
        daily_bars_dict: Dict mapping symbol -> list of daily bar dicts
        universe_dict: Dict mapping symbol -> universe record dict

    Returns:
        List of values matching RICH_CSV_HEADERS column order
    """
    plan = trade.plan
    pattern = plan.pattern if plan else None
    symbol = trade.symbol

    # Daily bar context for this symbol/date
    day_bar = _find_daily_bar(symbol, result.trade_date, daily_bars_dict)
    day_open = day_bar.get('open', '') if day_bar else ''
    day_high = day_bar.get('high', '') if day_bar else ''
    day_low = day_bar.get('low', '') if day_bar else ''
    day_close = day_bar.get('close', '') if day_bar else ''
    day_volume = day_bar.get('volume', '') if day_bar else ''

    intraday_move_pct = ''
    if day_bar and day_bar.get('low', 0) > 0:
        intraday_move_pct = f"{((day_bar['high'] - day_bar['low']) / day_bar['low']) * 100:.2f}"

    # Universe metadata
    uni = universe_dict.get(symbol, {})
    sector = uni.get('sector', '')
    float_shares = uni.get('float_shares', '')
    avg_volume_daily = uni.get('avg_volume_daily', '')

    # Relative volume: day_volume / avg_volume_daily
    relative_volume = ''
    if isinstance(day_volume, (int, float)) and day_volume > 0 and isinstance(avg_volume_daily, (int, float)) and avg_volume_daily > 0:
        relative_volume = f"{day_volume / avg_volume_daily:.2f}"

    # Entry minutes from open (9:30 ET = 13:30 UTC)
    entry_minutes = ''
    if trade.entry_time:
        market_open_utc_minutes = 13 * 60 + 30
        entry_utc_minutes = trade.entry_time.hour * 60 + trade.entry_time.minute
        entry_minutes = entry_utc_minutes - market_open_utc_minutes
        if entry_minutes < 0:
            entry_minutes = ''

    return [
        trade.symbol,
        result.trade_date,
        utc_to_et_str(trade.entry_time),
        f"{trade.entry_price:.2f}",
        f"{trade.stop_loss:.2f}",
        f"{trade.take_profit:.2f}",
        trade.shares,
        utc_to_et_str(trade.exit_time),
        f"{trade.exit_price:.2f}" if trade.exit_price else "",
        trade.exit_reason or "",
        f"{trade.pnl:.2f}",
        f"{trade.pnl_pct:.2f}",
        # Trade details
        trade.bars_held,
        f"{plan.risk_per_share:.4f}" if plan else "",
        f"{plan.reward_per_share:.4f}" if plan else "",
        f"{plan.risk_reward_ratio:.2f}" if plan else "",
        f"{plan.total_risk:.2f}" if plan else "",
        # Pattern metrics
        f"{pattern.pole_gain_pct:.2f}" if pattern else "",
        f"{pattern.retracement_pct:.2f}" if pattern else "",
        pattern.pullback_candle_count if pattern else "",
        f"{pattern.avg_pole_volume:.0f}" if pattern else "",
        f"{pattern.avg_flag_volume:.0f}" if pattern else "",
        f"{pattern.pole_height:.4f}" if pattern else "",
        f"{pattern.flag_low:.2f}" if pattern else "",
        f"{pattern.flag_high:.2f}" if pattern else "",
        # Entry bar OHLCV
        f"{trade.entry_bar_open:.2f}" if trade.entry_bar_open is not None else "",
        f"{trade.entry_bar_high:.2f}" if trade.entry_bar_high is not None else "",
        f"{trade.entry_bar_low:.2f}" if trade.entry_bar_low is not None else "",
        f"{trade.entry_bar_close:.2f}" if trade.entry_bar_close is not None else "",
        trade.entry_bar_volume if trade.entry_bar_volume is not None else "",
        # Daily bar context
        f"{day_open:.2f}" if isinstance(day_open, (int, float)) else day_open,
        f"{day_high:.2f}" if isinstance(day_high, (int, float)) else day_high,
        f"{day_low:.2f}" if isinstance(day_low, (int, float)) else day_low,
        f"{day_close:.2f}" if isinstance(day_close, (int, float)) else day_close,
        day_volume,
        intraday_move_pct,
        # Volume analysis
        avg_volume_daily,
        relative_volume,
        # Universe metadata
        sector,
        float_shares,
        # Derived
        entry_minutes,
        result.patterns_detected,
    ]


def _find_daily_bar(
    symbol: str, trade_date_str: str, daily_bars_dict: Dict
) -> Optional[Dict]:
    """
    Find the daily bar for a symbol on a specific date.

    Args:
        symbol: Stock symbol
        trade_date_str: Date string (YYYY-MM-DD)
        daily_bars_dict: Dict mapping symbol -> list of daily bar dicts

    Returns:
        Daily bar dict or None
    """
    bars = daily_bars_dict.get(symbol, [])
    for bar in bars:
        bar_date = bar.get('date')
        if bar_date is None:
            continue
        # Handle both date objects and strings
        if isinstance(bar_date, date):
            bar_date_str = bar_date.isoformat()
        else:
            bar_date_str = str(bar_date)
        if bar_date_str == trade_date_str:
            return bar
    return None


def write_rich_csv_report(
    results: List[BacktestResult],
    output_path: str,
    daily_bars_dict: Dict,
    universe_dict: Dict,
) -> int:
    """
    Write rich CSV report with extended columns.

    Args:
        results: List of BacktestResult objects
        output_path: Path for the CSV file
        daily_bars_dict: Dict mapping symbol -> list of daily bar dicts
        universe_dict: Dict mapping symbol -> universe record dict

    Returns:
        Number of trade rows written
    """
    trade_count = 0
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(RICH_CSV_HEADERS)

        for result in results:
            for trade in result.trades_simulated:
                row = build_rich_row(trade, result, daily_bars_dict, universe_dict)
                writer.writerow(row)
                trade_count += 1

    logger.info(f"Rich CSV report written to {output_path} ({trade_count} trades)")
    return trade_count


# ---------------------------------------------------------------------------
# Monthly splitting and parallel execution
# ---------------------------------------------------------------------------


def split_into_months(start: date, end: date) -> List[Tuple[date, date]]:
    """
    Split a date range into monthly chunks.

    Each chunk is (month_start, month_end) where month_end is the last day
    of the month or the overall end date, whichever is earlier.

    Args:
        start: Start date (inclusive)
        end: End date (inclusive)

    Returns:
        List of (month_start, month_end) tuples
    """
    if start > end:
        return []

    chunks = []
    current = start

    while current <= end:
        # End of this month
        if current.month == 12:
            month_end = date(current.year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = date(current.year, current.month + 1, 1) - timedelta(days=1)

        # Cap at overall end
        chunk_end = min(month_end, end)
        chunks.append((current, chunk_end))

        # Move to first of next month
        current = chunk_end + timedelta(days=1)

    return chunks


class MonthlyBacktestRunner:
    """
    Runs backtests in monthly chunks with optional parallelism.

    Each thread creates its own Database and AlpacaClient connections
    to avoid shared-state issues. SQLite WAL mode supports concurrent reads
    with busy_timeout for write contention.
    """

    def __init__(self, max_workers: int = 2, scan_workers: int = 1, verbose: bool = False):
        """
        Initialize MonthlyBacktestRunner.

        Args:
            max_workers: Number of parallel month threads (default 2)
            scan_workers: Number of parallel processes for scanning movers
                within each month. >1 uses multiprocessing (best for cached
                re-runs where all data is in SQLite). Default 1 (sequential).
            verbose: Enable verbose/debug logging
        """
        self.max_workers = max_workers
        self.scan_workers = scan_workers
        self.verbose = verbose

    def run_month(
        self,
        month_start: date,
        month_end: date,
        output_dir: str,
        month_idx: int = 0,
        total_months: int = 0,
    ) -> MonthResult:
        """
        Run backtest for a single month chunk.

        Creates its own DB and API client connections for thread safety.

        Args:
            month_start: First day of the month chunk
            month_end: Last day of the month chunk
            output_dir: Directory to write monthly CSV
            month_idx: 1-based month index for progress reporting
            total_months: Total months for progress reporting

        Returns:
            MonthResult with stats and CSV path
        """
        t0 = time.time()
        month_label = f"{month_start.year}-{month_start.month:02d}"
        progress = f"[Month {month_idx}/{total_months}]" if total_months else ""

        logger.info(f"{progress} Processing {month_label} ({month_start} to {month_end})...")

        # Each thread gets its own connections
        load_dotenv()
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_API_SECRET")
        if not api_key or not api_secret:
            logger.error("Missing ALPACA_API_KEY or ALPACA_API_SECRET")
            return MonthResult(
                month_label=month_label, csv_path="", num_movers=0,
                num_trades=0, total_pnl=0.0, elapsed_seconds=0.0, results=[],
            )

        client = AlpacaClient(api_key=api_key, api_secret=api_secret)
        # Each thread gets its own Database connection — never use the singleton
        # get_database() here, as it shares a single connection across threads.
        db = Database()

        try:
            # Load universe
            universe = db.get_active_universe()
            symbols = [s['symbol'] for s in universe]
            universe_dict = {s['symbol']: s for s in universe}

            if not symbols:
                logger.warning(f"{progress} No active symbols in universe")
                return MonthResult(
                    month_label=month_label, csv_path="", num_movers=0,
                    num_trades=0, total_pnl=0.0,
                    elapsed_seconds=time.time() - t0, results=[],
                )

            # Fetch daily bars and find movers (with scanner filters from config.yaml)
            daily_bars = fetch_daily_bars_cached(
                symbols, month_start, month_end, client, db
            )

            from config import Config
            cfg = Config._load_yaml_only()
            scanner_cfg = cfg.get("scanner", {})
            movers = find_big_movers(
                daily_bars,
                universe_dict=universe_dict,
                price_min=float(scanner_cfg.get("price_min", 2.0)),
                price_max=float(scanner_cfg.get("price_max", 20.0)),
                float_max=int(scanner_cfg.get("float_max", 10_000_000)),
            )

            if not movers:
                logger.info(f"{progress} {month_label}: 0 movers, skipping")
                return MonthResult(
                    month_label=month_label, csv_path="", num_movers=0,
                    num_trades=0, total_pnl=0.0,
                    elapsed_seconds=time.time() - t0, results=[],
                )

            # Pre-fetch any uncached 1-min bars before parallel scan.
            # The parallel workers only read from cache (no API access),
            # so all bars must be cached first.
            if self.scan_workers > 1:
                uncached = 0
                for sym, d in movers:
                    cached = db.get_intraday_bars_cached(sym, d.isoformat())
                    if not cached:
                        uncached += 1
                        get_1min_bars_cached(sym, d, client, db)
                if uncached:
                    logger.info(f"{progress} Pre-fetched {uncached} uncached 1-min bar sets")

            # Load volume profiles for bucket rvol check
            volume_profiles = db.get_all_volume_profiles()
            logger.info(f"{progress} Volume profiles loaded for {len(volume_profiles)} symbols")

            # Run backtests — use multiprocessing if scan_workers > 1
            if self.scan_workers > 1:
                results = run_batch_backtest_parallel(
                    movers, db_path=str(db.db_path), max_workers=self.scan_workers
                )
            else:
                runner = BacktestRunner()  # uses from_config() for all settings
                results = run_batch_backtest(movers, client, runner, db=db,
                                            universe_dict=universe_dict,
                                            volume_profiles=volume_profiles)

            # Write rich CSV for this month
            csv_filename = f"backtest_{month_start.year}_{month_start.month:02d}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            write_rich_csv_report(results, csv_path, daily_bars, universe_dict)

            # Compute stats and print month summary
            all_trades = [t for r in results for t in r.trades_simulated]
            total_pnl = sum(t.pnl for t in all_trades)
            elapsed = time.time() - t0

            logger.info(
                f"{progress} {month_label}: {len(movers)} movers, "
                f"{len(all_trades)} trades, P&L ${total_pnl:+.2f} ({elapsed:.0f}s)"
            )

            self._print_month_summary(month_label, all_trades, len(movers), elapsed)

            return MonthResult(
                month_label=month_label,
                csv_path=csv_path,
                num_movers=len(movers),
                num_trades=len(all_trades),
                total_pnl=total_pnl,
                elapsed_seconds=elapsed,
                results=results,
            )

        finally:
            db.close()

    def aggregate_csvs(self, csv_paths: List[str], master_path: str) -> int:
        """
        Aggregate monthly CSV files into a single master CSV.

        Args:
            csv_paths: List of monthly CSV file paths
            master_path: Path for the master CSV file

        Returns:
            Total number of data rows in master CSV
        """
        valid_paths = [p for p in csv_paths if p and os.path.exists(p)]
        if not valid_paths:
            logger.warning("No valid CSV files to aggregate")
            return 0

        total_rows = 0
        with open(master_path, 'w', newline='', encoding='utf-8') as master_f:
            writer = csv.writer(master_f)
            header_written = False

            for csv_path in valid_paths:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader)

                    if not header_written:
                        writer.writerow(header)
                        header_written = True

                    for row in reader:
                        writer.writerow(row)
                        total_rows += 1

        logger.info(
            f"Master CSV aggregated: {total_rows} trades from "
            f"{len(valid_paths)} monthly files → {master_path}"
        )
        return total_rows

    def run_all(self, start: date, end: date, output_dir: str = "backtest_results") -> str:
        """
        Run backtests for the full date range, split into monthly chunks.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)
            output_dir: Directory for CSV output

        Returns:
            Path to master CSV file
        """
        os.makedirs(output_dir, exist_ok=True)
        months = split_into_months(start, end)
        total_months = len(months)

        logger.info(
            f"Monthly backtest: {start} to {end} "
            f"({total_months} month chunks, {self.max_workers} workers)"
        )
        logger.warning(
            "NOTE: Using today's universe for all months. "
            "Stocks that were delisted, IPO'd, or moved out of price range "
            "are not accounted for — results have survivorship bias."
        )

        month_results: List[MonthResult] = []
        csv_paths: List[str] = []

        if self.max_workers <= 1:
            # Sequential execution
            for idx, (m_start, m_end) in enumerate(months, 1):
                result = self.run_month(m_start, m_end, output_dir, idx, total_months)
                month_results.append(result)
                if result.csv_path:
                    csv_paths.append(result.csv_path)
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for idx, (m_start, m_end) in enumerate(months, 1):
                    future = executor.submit(
                        self.run_month, m_start, m_end, output_dir, idx, total_months
                    )
                    futures[future] = idx

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        month_results.append(result)
                        if result.csv_path:
                            csv_paths.append(result.csv_path)
                    except Exception as e:
                        idx = futures[future]
                        logger.error(f"Month {idx} failed: {e}")

        # Sort CSVs by filename (chronological order)
        csv_paths.sort()

        # Aggregate into master CSV
        start_str = f"{start.year}_{start.month:02d}"
        end_str = f"{end.year}_{end.month:02d}"
        master_filename = f"backtest_full_{start_str}_to_{end_str}.csv"
        master_path = os.path.join(output_dir, master_filename)

        total_rows = self.aggregate_csvs(csv_paths, master_path)

        # Print summary table
        self._print_summary(month_results, total_rows, master_path)

        return master_path

    @staticmethod
    def _print_month_summary(
        month_label: str, trades: list, num_movers: int, elapsed: float
    ) -> None:
        """Print a compact results summary after each month completes."""
        if not trades:
            print(f"\n  [{month_label}] {num_movers} movers | 0 trades | ({elapsed:.0f}s)")
            return

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in trades)
        win_rate = len(wins) / len(trades) * 100

        print(f"\n  [{month_label}] {num_movers} movers | {len(trades)} trades | "
              f"WR {win_rate:.0f}% | P&L ${total_pnl:+,.2f} | ({elapsed:.0f}s)")

        # Show individual trades
        for t in trades:
            pnl_marker = "W" if t.pnl > 0 else "L"
            exit_info = t.exit_reason or "?"
            entry_et = ""
            if t.entry_time:
                et = t.entry_time.replace(tzinfo=None) - timedelta(hours=4)
                entry_et = et.strftime("%H:%M")
            print(f"    {pnl_marker} {t.symbol:<6} ${t.pnl:+8.2f}  "
                  f"entry ${t.entry_price:.2f} @ {entry_et}  "
                  f"exit ${t.exit_price:.2f} ({exit_info})")

    def _print_summary(
        self,
        month_results: List[MonthResult],
        total_rows: int,
        master_path: str,
    ) -> None:
        """Print month-by-month summary table."""
        # Sort by month label
        month_results.sort(key=lambda r: r.month_label)

        total_movers = sum(r.num_movers for r in month_results)
        total_trades = sum(r.num_trades for r in month_results)
        total_pnl = sum(r.total_pnl for r in month_results)
        total_time = sum(r.elapsed_seconds for r in month_results)

        print("\n" + "=" * 80)
        print("  MONTHLY BACKTEST SUMMARY")
        print("=" * 80)
        print(f"  {'Month':<12} {'Movers':>8} {'Trades':>8} {'P&L':>12} {'Time':>8}")
        print("-" * 80)

        for r in month_results:
            pnl_str = f"${r.total_pnl:+,.2f}"
            time_str = f"{r.elapsed_seconds:.0f}s"
            print(f"  {r.month_label:<12} {r.num_movers:>8} {r.num_trades:>8} {pnl_str:>12} {time_str:>8}")

        print("-" * 80)
        pnl_str = f"${total_pnl:+,.2f}"
        time_str = f"{total_time:.0f}s"
        print(f"  {'TOTAL':<12} {total_movers:>8} {total_trades:>8} {pnl_str:>12} {time_str:>8}")
        print("=" * 80)
        print(f"  Master CSV: {master_path} ({total_rows} rows)")
        print("=" * 80 + "\n")
