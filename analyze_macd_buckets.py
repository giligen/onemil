"""
MACD Histogram Bucket Analysis.

For each trade in a backtest CSV, computes the warmed-up MACD histogram
at entry time and buckets trades by histogram value (as % of entry price).

Uses previous trading day's bars for MACD warm-up to avoid cold-start.

Usage:
    python3 analyze_macd_buckets.py backtest_15month_current.csv
"""

import argparse
import logging
import sys
from datetime import datetime, date, timedelta
from collections import defaultdict

import pandas as pd
import pytz

from persistence.database import Database
from data_sources.alpaca_client import AlpacaClient
from trading.indicators import macd_histogram

# Module-level API client for fetching uncached bars
_api_client = None

logger = logging.getLogger(__name__)
ET = pytz.timezone('US/Eastern')

# MACD bucket edges (as % of entry price)
BUCKET_EDGES = [-float('inf'), -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0, float('inf')]
BUCKET_LABELS = [
    "< -1%", "-1:-0.5%", "-0.5:-0.2%", "-0.2:-0.1%", "-0.1:0%",
    "0:0.1%", "0.1:0.2%", "0.2:0.5%", "0.5:1%", "> 1%",
]


def get_previous_trading_date(trade_date: date) -> date:
    """Get previous trading day (skip weekends)."""
    prev = trade_date - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return prev


def compute_macd_at_entry(
    symbol: str,
    trade_date: date,
    entry_time_et_str: str,
    db: Database,
) -> float | None:
    """
    Compute warmed-up MACD histogram at entry time.

    Args:
        symbol: Stock symbol
        trade_date: Trading date
        entry_time_et_str: Entry time in ET (HH:MM:SS)
        db: Database for cached bars

    Returns:
        MACD histogram value at entry bar, or None if insufficient data
    """
    date_str = trade_date.isoformat()

    # Load today's bars
    cached = db.get_intraday_bars_cached(symbol, date_str)
    if not cached:
        return None
    bars = pd.DataFrame(cached)
    if bars.empty:
        return None

    # Load previous day's bars for warm-up — fetch from API if not cached
    prev_date = get_previous_trading_date(trade_date)
    prev_cached = db.get_intraday_bars_cached(symbol, prev_date.isoformat())
    warmup_closes = pd.Series(dtype=float)
    if prev_cached:
        prev_bars = pd.DataFrame(prev_cached)
        if not prev_bars.empty:
            warmup_closes = prev_bars['close'].tail(60).reset_index(drop=True)
    elif _api_client is not None:
        # Fetch from API and cache for future runs
        try:
            from batch_backtest import _market_hours_utc
            mo, mc = _market_hours_utc(prev_date)
            prev_bars = _api_client.get_historical_1min_bars(symbol, mo, mc)
            if prev_bars is not None and not prev_bars.empty:
                db.save_intraday_bars(symbol, prev_date.isoformat(), prev_bars.to_dict('records'))
                warmup_closes = prev_bars['close'].tail(60).reset_index(drop=True)
        except Exception:
            pass

    # Find entry bar index by matching entry time
    entry_h, entry_m = int(entry_time_et_str.split(':')[0]), int(entry_time_et_str.split(':')[1])

    entry_bar_idx = None
    for i, row in bars.iterrows():
        ts = row['timestamp']
        if hasattr(ts, 'astimezone'):
            bar_et = ts.astimezone(ET)
        elif isinstance(ts, str):
            bar_et = pd.Timestamp(ts).tz_localize('UTC').tz_convert(ET)
        else:
            continue
        if bar_et.hour == entry_h and bar_et.minute == entry_m:
            entry_bar_idx = i
            break

    if entry_bar_idx is None:
        # Fallback: find closest bar
        for i, row in bars.iterrows():
            ts = row['timestamp']
            if hasattr(ts, 'astimezone'):
                bar_et = ts.astimezone(ET)
            elif isinstance(ts, str):
                bar_et = pd.Timestamp(ts).tz_localize('UTC').tz_convert(ET)
            else:
                continue
            bar_minutes = bar_et.hour * 60 + bar_et.minute
            entry_minutes = entry_h * 60 + entry_m
            if bar_minutes >= entry_minutes:
                entry_bar_idx = i
                break

    if entry_bar_idx is None:
        return None

    # Build closes up to entry bar, prepend warm-up
    today_closes = bars['close'].iloc[:entry_bar_idx + 1]
    if len(warmup_closes) > 0:
        all_closes = pd.concat([warmup_closes, today_closes], ignore_index=True)
    else:
        all_closes = today_closes.reset_index(drop=True)

    # Need at least slow + signal bars for meaningful MACD
    min_bars = 26 + 9  # 35
    if len(all_closes) < min_bars:
        return None

    hist = macd_histogram(all_closes)
    return float(hist.iloc[-1])


def bucket_index(macd_pct: float) -> int:
    """Find which bucket a MACD % value falls into."""
    for i in range(len(BUCKET_EDGES) - 1):
        if macd_pct < BUCKET_EDGES[i + 1]:
            return i
    return len(BUCKET_LABELS) - 1


def main():
    """Run MACD bucket analysis."""
    parser = argparse.ArgumentParser(description="MACD histogram bucket analysis")
    parser.add_argument("csv", help="Backtest CSV file path")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(message)s")

    # Suppress DB logging
    logging.getLogger('persistence.database').setLevel(logging.ERROR)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} trades from {args.csv}")

    # Initialize API client for fetching uncached prev-day bars
    global _api_client
    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if api_key and api_secret:
        _api_client = AlpacaClient(api_key, api_secret)
        print("API client ready — will fetch uncached prev-day bars")

    db = Database(db_path="data/onemil.db")

    # Compute MACD for each trade
    trades_with_macd = []
    no_macd = 0

    for idx, row in df.iterrows():
        symbol = row['symbol']
        trade_date = date.fromisoformat(row['date'])
        entry_time = row['entry_time_et']
        entry_price = row['entry_price']
        pnl = row['pnl']

        macd_val = compute_macd_at_entry(symbol, trade_date, entry_time, db)

        if macd_val is None:
            no_macd += 1
            continue

        macd_pct = (macd_val / entry_price) * 100  # as % of price
        trades_with_macd.append({
            'symbol': symbol,
            'date': row['date'],
            'entry_price': entry_price,
            'pnl': pnl,
            'macd_val': macd_val,
            'macd_pct': macd_pct,
            'win': pnl > 0,
        })

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} trades...")

    db.close()

    print(f"\nMACD computed for {len(trades_with_macd)} trades ({no_macd} skipped — insufficient data)")

    # Bucket analysis
    buckets = defaultdict(list)
    for t in trades_with_macd:
        bi = bucket_index(t['macd_pct'])
        buckets[bi].append(t)

    # Print results
    total_pnl = sum(t['pnl'] for t in trades_with_macd)
    total_trades = len(trades_with_macd)

    print(f"\n{'=' * 85}")
    print(f"  MACD Histogram Bucket Analysis ({total_trades} trades, warm-up enabled)")
    print(f"  Config: regime ON, trail 1R@2R, green=2, 15-month honest backtest")
    print(f"{'=' * 85}")
    print(f"{'Bucket':>14} {'N':>5} {'WR':>7} {'Avg PnL':>10} {'Total PnL':>12} {'Avg Win':>10} {'Avg Loss':>10}")
    print(f"{'-' * 85}")

    for i, label in enumerate(BUCKET_LABELS):
        trades = buckets.get(i, [])
        n = len(trades)
        if n == 0:
            print(f"{label:>14} {'0':>5} {'—':>7} {'—':>10} {'—':>12} {'—':>10} {'—':>10}")
            continue

        wins = [t for t in trades if t['win']]
        losses = [t for t in trades if not t['win']]
        wr = len(wins) / n * 100
        avg_pnl = sum(t['pnl'] for t in trades) / n
        total = sum(t['pnl'] for t in trades)
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0

        print(
            f"{label:>14} {n:>5} {wr:>6.1f}% "
            f"{'${:>+,.0f}'.format(avg_pnl):>10} "
            f"{'${:>+,.0f}'.format(total):>12} "
            f"{'${:>+,.0f}'.format(avg_win):>10} "
            f"{'${:>+,.0f}'.format(avg_loss):>10}"
        )

    print(f"{'-' * 85}")
    avg_total = total_pnl / total_trades if total_trades else 0
    total_wins = sum(1 for t in trades_with_macd if t['win'])
    wr_total = total_wins / total_trades * 100 if total_trades else 0
    avg_w = sum(t['pnl'] for t in trades_with_macd if t['win']) / total_wins if total_wins else 0
    avg_l = sum(t['pnl'] for t in trades_with_macd if not t['win']) / (total_trades - total_wins) if (total_trades - total_wins) else 0
    print(
        f"{'TOTAL':>14} {total_trades:>5} {wr_total:>6.1f}% "
        f"{'${:>+,.0f}'.format(avg_total):>10} "
        f"{'${:>+,.0f}'.format(total_pnl):>12} "
        f"{'${:>+,.0f}'.format(avg_w):>10} "
        f"{'${:>+,.0f}'.format(avg_l):>10}"
    )
    print(f"{'=' * 85}")

    # Recommendation section
    print(f"\n{'=' * 85}")
    print(f"  RECOMMENDATIONS")
    print(f"{'=' * 85}")

    for i, label in enumerate(BUCKET_LABELS):
        trades = buckets.get(i, [])
        n = len(trades)
        if n == 0:
            continue
        wr = sum(1 for t in trades if t['win']) / n * 100
        avg_pnl = sum(t['pnl'] for t in trades) / n
        total = sum(t['pnl'] for t in trades)

        if n < 10:
            action = "SKIP (too few trades to be statistically meaningful)"
        elif wr < 30 and avg_pnl < 0:
            action = "SKIP (low WR + negative avg PnL)"
        elif wr < 35 and avg_pnl < 50:
            action = "REDUCE RISK 0.5x (marginal edge)"
        elif wr >= 45 and avg_pnl > 100:
            action = "INCREASE RISK 1.5x (strong edge)"
        elif wr >= 42 and avg_pnl > 50:
            action = "INCREASE RISK 1.25x (good edge)"
        else:
            action = "NORMAL (1.0x risk)"

        print(f"  {label:>14}: {action}")
        print(f"                  N={n}, WR={wr:.1f}%, Avg=${avg_pnl:+,.0f}, Total=${total:+,.0f}")

    print(f"{'=' * 85}")


if __name__ == "__main__":
    main()
