"""
Slippage breakdown analysis for MACD wave trades.

Reads the trades DB and prints a per-trade and aggregate report on the
decomposition of execution slippage:

  bar close → loop processed    (polling interval + bar fetch RTT)
  loop     → quote fetched     (MACD compute + quote RTT)
  quote    → order submitted   (decision + submit RTT)
  submit   → fill              (Alpaca fill latency)

And the price-drift components:

  drift_bar_to_ask_bps  — (ask_at_quote - bar_close) / bar_close × 10000
  drift_ask_to_fill_bps — (fill - ask_at_quote) / ask_at_quote × 10000  (≤0 expected)
  drift_bar_to_fill_bps — (fill - bar_close) / bar_close × 10000  (TRUE slippage vs BT reference)

Important timing caveats:
  * `bar_close_to_loop_ms` — after T1.2, this is just the batch-fetch RTT for
    symbols covered by the batched call; symbols missed by the batch still
    include per-symbol REST RTT. Expect medians to shrink (N-1)*300ms after T1.2.
  * `submit_to_fill_ms` — with T3.1 (OrderStreamWatcher) we rely on a push
    callback; on a stream-buggy cycle we wait up to 5s for a push before
    falling back to REST, so occasional outliers of ~5-65s are NOT regressions,
    they're the fallback engaging. Median should be <2000ms in healthy state.

Usage:
  python analyze_slippage.py                        # local trades.db
  python analyze_slippage.py --db data/trades.db    # explicit path
  python analyze_slippage.py --since 2026-04-01     # filter by date
"""

import argparse
import sqlite3
from pathlib import Path
from statistics import mean, median

COLS = [
    'trade_date', 'symbol', 'entry_price', 'fill_price',
    'bar_close_price', 'entry_quote_bid', 'entry_quote_ask', 'entry_quote_spread',
    'bar_close_at', 'loop_processed_at', 'quote_fetched_at',
    'order_submitted_at', 'order_filled_at',
    'bar_close_to_loop_ms', 'loop_to_quote_ms', 'quote_to_submit_ms', 'submit_to_fill_ms',
    'drift_bar_to_ask_bps', 'drift_ask_to_fill_bps', 'drift_bar_to_fill_bps',
    'pnl', 'order_status',
]


def load_trades(db_path: Path, since: str | None):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    q = (
        f"SELECT {', '.join(COLS)} FROM trades "
        "WHERE strategy = 'macd_wave' AND fill_price IS NOT NULL"
    )
    params = []
    if since:
        q += " AND trade_date >= ?"
        params.append(since)
    q += " ORDER BY trade_date, id"
    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    conn.close()
    return rows


def _nums(rows, key):
    return [r[key] for r in rows if r.get(key) is not None]


def _summary(values, fmt):
    if not values:
        return "n/a"
    return (
        f"mean={fmt.format(mean(values))}  "
        f"median={fmt.format(median(values))}  "
        f"min={fmt.format(min(values))}  "
        f"max={fmt.format(max(values))}  "
        f"n={len(values)}"
    )


def print_per_trade(rows):
    print("\nPer-trade detail:")
    header = (
        f"{'date':10s} {'sym':6s} "
        f"{'bar_cls':>8s} {'ask':>7s} {'fill':>7s} "
        f"{'wait_ms':>7s} {'quot_ms':>7s} {'sub_ms':>7s} {'fill_ms':>7s} "
        f"{'bar→ask':>8s} {'ask→fill':>9s} {'bar→fill':>9s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        def f(v, fmt="{:>7.2f}"):
            return fmt.format(v) if v is not None else "   -   "
        print(
            f"{r['trade_date']:10s} {r['symbol']:6s} "
            f"{f(r['bar_close_price'],'{:>8.2f}')} "
            f"{f(r['entry_quote_ask'])} "
            f"{f(r['fill_price'])} "
            f"{f(r['bar_close_to_loop_ms'],'{:>7.0f}')} "
            f"{f(r['loop_to_quote_ms'],'{:>7.0f}')} "
            f"{f(r['quote_to_submit_ms'],'{:>7.0f}')} "
            f"{f(r['submit_to_fill_ms'],'{:>7.0f}')} "
            f"{f(r['drift_bar_to_ask_bps'],'{:>8.1f}')} "
            f"{f(r['drift_ask_to_fill_bps'],'{:>9.1f}')} "
            f"{f(r['drift_bar_to_fill_bps'],'{:>9.1f}')}"
        )


def print_summary(rows):
    print("\nAggregate latency (ms):")
    for key, label in [
        ('bar_close_to_loop_ms', 'bar close  → loop   '),
        ('loop_to_quote_ms',    'loop       → quote  '),
        ('quote_to_submit_ms',  'quote      → submit '),
        ('submit_to_fill_ms',   'submit     → fill   '),
    ]:
        print(f"  {label} {_summary(_nums(rows, key), '{:.0f}')}")

    print("\nDrift (basis points, 100 bps = 1%):")
    for key, label in [
        ('drift_bar_to_ask_bps',  'bar → ask  (wait+pipeline)'),
        ('drift_ask_to_fill_bps', 'ask → fill (execution)    '),
        ('drift_bar_to_fill_bps', 'bar → fill (TOTAL)        '),
    ]:
        print(f"  {label} {_summary(_nums(rows, key), '{:+6.1f}')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db', default='data/trades.db')
    p.add_argument('--since', default=None, help='ISO date, e.g. 2026-04-01')
    p.add_argument('--detail', action='store_true', help='per-trade table')
    args = p.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: {db_path} not found")
        return

    rows = load_trades(db_path, args.since)
    print(f"Loaded {len(rows)} MACD wave filled trades from {db_path}")
    if args.since:
        print(f"  (since {args.since})")

    if not rows:
        return

    instrumented = [r for r in rows if r.get('bar_close_price') is not None]
    print(f"  {len(instrumented)} instrumented (Migration 10 — post-deploy only)")
    if not instrumented:
        print("\nNo instrumented trades yet. Deploy the new engine, then re-run.")
        return

    if args.detail:
        print_per_trade(instrumented)
    print_summary(instrumented)


if __name__ == "__main__":
    main()
