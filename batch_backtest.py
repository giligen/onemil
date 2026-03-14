"""
Batch backtester for momentum day trading strategy.

Scans the universe for stocks with 10%+ intraday moves in a date range,
runs backtests on each qualifying (symbol, date) pair, and produces a
CSV report for TradingView validation.

Usage:
    python batch_backtest.py
    python batch_backtest.py --start 2026-03-01 --end 2026-03-13
    python batch_backtest.py --verbose
"""

import argparse
import csv
import logging
import os
import sys
from datetime import datetime, timezone, date, timedelta
from typing import List, Tuple, Dict, Optional

import pandas as pd
from dotenv import load_dotenv

from backtest import BacktestRunner, BacktestResult
from data_sources.alpaca_client import AlpacaClient, AlpacaAPIError
from persistence.database import Database, get_database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTRADAY_MOVE_THRESHOLD = 0.10  # 10% (high - low) / low
DEFAULT_START = "2026-03-01"
DEFAULT_END = "2026-03-13"
CSV_OUTPUT = "backtest_results_march_2026.csv"

CSV_HEADERS = [
    "symbol", "date", "entry_time_et", "entry_price", "stop_loss",
    "target", "shares", "exit_time_et", "exit_price", "exit_reason",
    "pnl", "pnl_pct",
]


# ---------------------------------------------------------------------------
# Step 1: Find 10%+ intraday movers
# ---------------------------------------------------------------------------


def find_big_movers(
    daily_bars: Dict[str, List[Dict]], threshold: float = INTRADAY_MOVE_THRESHOLD
) -> List[Tuple[str, date]]:
    """
    Filter daily bars for (symbol, date) pairs with intraday move >= threshold.

    Intraday move = (high - low) / low.

    Args:
        daily_bars: Dict mapping symbol -> list of daily bar dicts
        threshold: Minimum (high-low)/low ratio (default 0.10 = 10%)

    Returns:
        Sorted list of (symbol, date) tuples qualifying for backtest
    """
    movers = []
    for symbol, bars in daily_bars.items():
        for bar in bars:
            low = bar['low']
            high = bar['high']
            if low <= 0:
                continue
            move = (high - low) / low
            if move >= threshold:
                bar_date = bar['date'] if isinstance(bar['date'], date) else bar['date']
                movers.append((symbol, bar_date))
                logger.debug(
                    f"  {symbol} {bar_date}: move {move:.1%} "
                    f"(low=${low:.2f}, high=${high:.2f})"
                )
    movers.sort(key=lambda x: (x[1], x[0]))
    logger.info(f"Found {len(movers)} symbol/date pairs with {threshold:.0%}+ intraday move")
    return movers


# ---------------------------------------------------------------------------
# Step 1.5: Cached daily bars fetching
# ---------------------------------------------------------------------------


def fetch_daily_bars_cached(
    symbols: List[str],
    start_date: date,
    end_date: date,
    client: AlpacaClient,
    db: Database,
) -> Dict[str, List[Dict]]:
    """
    Fetch daily bars with DB caching — only hits API for uncached symbols.

    Args:
        symbols: List of stock symbols
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        client: AlpacaClient for API fetches
        db: Database for caching

    Returns:
        Dict mapping symbol -> list of daily bar dicts
    """
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    # Check which symbols already have cached data
    cached_symbols = db.get_cached_daily_bar_symbols(start_str, end_str)
    uncached = [s for s in symbols if s not in cached_symbols]

    logger.info(
        f"Daily bars: {len(cached_symbols)} cached, "
        f"{len(uncached)} need API fetch"
    )

    # Fetch uncached from API and store
    if uncached:
        logger.info(f"Fetching daily bars from API for {len(uncached)} symbols...")
        api_bars = client.get_daily_bars_range(uncached, start_date, end_date)

        # Flatten for DB storage
        flat_bars = []
        for symbol, bars in api_bars.items():
            for bar in bars:
                flat_bars.append({
                    'symbol': symbol,
                    'date': bar['date'].isoformat() if isinstance(bar['date'], date) else bar['date'],
                    'open': bar['open'],
                    'high': bar['high'],
                    'low': bar['low'],
                    'close': bar['close'],
                    'volume': bar['volume'],
                })
        db.save_daily_bars(flat_bars)
        logger.info(f"Cached {len(flat_bars)} daily bars to DB")

    # Return all from cache (now includes freshly fetched)
    all_bars = db.get_daily_bars_cached(symbols, start_str, end_str)
    logger.info(f"Total: {len(all_bars)} symbols with daily bar data")
    return all_bars


# ---------------------------------------------------------------------------
# Step 2: Backtest each qualifying day (with 1-min bar caching)
# ---------------------------------------------------------------------------


def get_1min_bars_cached(
    symbol: str,
    trade_date: date,
    client: AlpacaClient,
    db: Database,
) -> pd.DataFrame:
    """
    Get 1-min bars for a symbol/date, using DB cache first.

    Args:
        symbol: Stock symbol
        trade_date: Trade date
        client: AlpacaClient for API fetch if not cached
        db: Database for cache

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    date_str = trade_date.isoformat()

    # Check cache
    cached = db.get_intraday_bars_cached(symbol, date_str)
    if cached:
        logger.debug(f"Cache hit: {len(cached)} 1-min bars for {symbol} on {date_str}")
        return pd.DataFrame(cached)

    # Fetch from API
    dt = datetime(trade_date.year, trade_date.month, trade_date.day)
    market_open = dt.replace(hour=13, minute=30, second=0, tzinfo=timezone.utc)
    market_close = dt.replace(hour=20, minute=0, second=0, tzinfo=timezone.utc)

    bars = client.get_historical_1min_bars(symbol, market_open, market_close)

    # Cache the results
    if not bars.empty:
        bar_records = bars.to_dict('records')
        db.save_intraday_bars(symbol, date_str, bar_records)
        logger.debug(f"Cached {len(bar_records)} 1-min bars for {symbol} on {date_str}")

    return bars


def run_batch_backtest(
    movers: List[Tuple[str, date]],
    client: AlpacaClient,
    runner: BacktestRunner,
    db: Optional[Database] = None,
) -> List[BacktestResult]:
    """
    Run backtests on all qualifying (symbol, date) pairs.

    Fetches 1-min bars (cached) for each pair, runs the backtest, collects results.
    API errors are logged and skipped (never abort the batch).

    Args:
        movers: List of (symbol, date) pairs to backtest
        client: AlpacaClient for fetching historical bars
        runner: BacktestRunner instance
        db: Database for 1-min bar caching (optional, fetches without caching if None)

    Returns:
        List of BacktestResult objects (one per successful run)
    """
    results = []
    total = len(movers)

    for idx, (symbol, trade_date) in enumerate(movers, 1):
        date_str = trade_date.isoformat()

        try:
            if db:
                bars = get_1min_bars_cached(symbol, trade_date, client, db)
            else:
                dt = datetime(trade_date.year, trade_date.month, trade_date.day)
                market_open = dt.replace(hour=13, minute=30, second=0, tzinfo=timezone.utc)
                market_close = dt.replace(hour=20, minute=0, second=0, tzinfo=timezone.utc)
                bars = client.get_historical_1min_bars(symbol, market_open, market_close)

            if bars.empty:
                logger.warning(f"[{idx}/{total}] {symbol} {date_str} — no bars, skipping")
                continue

            result = runner.run(symbol, bars, date_str)
            results.append(result)

            # Verbose progress line
            n_patterns = result.patterns_detected
            n_trades = len(result.trades_simulated)
            pnl = result.summary_pnl
            logger.info(
                f"[{idx}/{total}] {symbol} {date_str} — "
                f"{n_patterns} patterns, {n_trades} trade(s), "
                f"P&L ${pnl:+.2f}"
            )

        except AlpacaAPIError as e:
            logger.error(f"[{idx}/{total}] {symbol} {date_str} — API error: {e}, skipping")
        except Exception as e:
            logger.error(f"[{idx}/{total}] {symbol} {date_str} — unexpected error: {e}, skipping")

    logger.info(f"Batch backtest complete: {len(results)}/{total} runs succeeded")
    return results


# ---------------------------------------------------------------------------
# Step 3: CSV report + console summary
# ---------------------------------------------------------------------------


def utc_to_et_str(ts: datetime) -> str:
    """Convert UTC datetime to ET string (UTC-4 for EDT March 2026)."""
    if ts is None:
        return ""
    et = ts.replace(tzinfo=None) - __import__('datetime').timedelta(hours=4)
    return et.strftime("%H:%M:%S")


def write_csv_report(results: List[BacktestResult], output_path: str) -> int:
    """
    Write trade-level CSV report from backtest results.

    One row per trade for TradingView validation.

    Args:
        results: List of BacktestResult objects
        output_path: Path for the CSV file

    Returns:
        Number of trade rows written
    """
    trade_count = 0
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)

        for result in results:
            for trade in result.trades_simulated:
                writer.writerow([
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
                ])
                trade_count += 1

    logger.info(f"CSV report written to {output_path} ({trade_count} trades)")
    return trade_count


def print_summary(
    total_universe: int,
    movers: List[Tuple[str, date]],
    results: List[BacktestResult],
) -> None:
    """
    Print batch backtest summary to console.

    Args:
        total_universe: Number of symbols in active universe
        movers: List of qualifying (symbol, date) pairs
        results: List of BacktestResult objects
    """
    all_trades = [t for r in results for t in r.trades_simulated]
    wins = [t for t in all_trades if t.pnl > 0]
    losses = [t for t in all_trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in all_trades)
    win_rate = len(wins) / len(all_trades) * 100 if all_trades else 0.0

    print("\n" + "=" * 70)
    print("  BATCH BACKTEST SUMMARY — March 2026")
    print("=" * 70)
    print(f"  Universe size:           {total_universe} symbols")
    print(f"  Symbol/date combos:      {total_universe * 9} (approx trading days)")
    print(f"  10%+ intraday movers:    {len(movers)}")
    print(f"  Backtests completed:     {len(results)}")
    print("-" * 70)
    print(f"  Total trades taken:      {len(all_trades)}")
    print(f"  Winning trades:          {len(wins)}")
    print(f"  Losing trades:           {len(losses)}")
    print(f"  Win rate:                {win_rate:.1f}%")
    print(f"  Total P&L:              ${total_pnl:+.2f}")
    if wins:
        print(f"  Avg win:                ${sum(t.pnl for t in wins) / len(wins):+.2f}")
    if losses:
        print(f"  Avg loss:               ${sum(t.pnl for t in losses) / len(losses):+.2f}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for batch backtesting."""
    parser = argparse.ArgumentParser(
        description="Batch backtest: scan universe for 10%+ movers and run strategy"
    )
    parser.add_argument(
        "--start", type=str, default=DEFAULT_START,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})"
    )
    parser.add_argument(
        "--end", type=str, default=DEFAULT_END,
        help=f"End date YYYY-MM-DD (default: {DEFAULT_END})"
    )
    parser.add_argument(
        "--output", type=str, default=CSV_OUTPUT,
        help=f"CSV output path (default: {CSV_OUTPUT})"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose/debug logging"
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load environment
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        logger.error("Missing ALPACA_API_KEY or ALPACA_API_SECRET in environment")
        sys.exit(1)

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    logger.info(f"Batch backtest: {start_date} to {end_date}")

    # Step 1: Load universe
    db = get_database()
    universe = db.get_active_universe()
    symbols = [s['symbol'] for s in universe]
    logger.info(f"Loaded {len(symbols)} symbols from active universe")

    if not symbols:
        logger.error("No active symbols in universe — run universe builder first")
        sys.exit(1)

    # Step 2: Fetch daily bars (cached) and find 10%+ movers
    client = AlpacaClient(api_key=api_key, api_secret=api_secret)
    logger.info("Fetching daily bars for date range (cache-first)...")
    daily_bars = fetch_daily_bars_cached(symbols, start_date, end_date, client, db)
    movers = find_big_movers(daily_bars)

    if not movers:
        logger.warning("No symbols with 10%+ intraday move found — nothing to backtest")
        print_summary(len(symbols), movers, [])
        return

    # Step 3: Run backtests (1-min bars also cached)
    runner = BacktestRunner()
    results = run_batch_backtest(movers, client, runner, db=db)

    # Step 4: Write CSV + print summary
    write_csv_report(results, args.output)
    print_summary(len(symbols), movers, results)


if __name__ == "__main__":
    main()
