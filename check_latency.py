#!/usr/bin/env python3
"""Daily Alpaca submit-latency tracker.

Prints the distribution of `quote_to_submit_ms` and `bar_close_to_loop_ms`
per strategy per day. Used to spot anomalies (e.g., the 2026-04-15 cloud
incident that pushed q2s from a typical 220-450ms baseline to 3.3s).

Usage:
    python3 check_latency.py                         # last 7 days
    python3 check_latency.py --days 30               # last 30 days
    python3 check_latency.py --since 2026-04-01      # absolute start

Reads `data/trades.db`. Read-only.
"""
from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
from datetime import date, timedelta
from typing import List, Optional

DB_PATH = 'data/trades.db'
WARN_THRESHOLD_MS = 1000  # mirrors OrderExecutor._SUBMIT_LATENCY_WARN_MS


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile (no interp) — robust for tiny samples."""
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(len(s) * pct / 100)) - 1))
    return s[k]


def _fmt(v: Optional[float], unit: str = 'ms') -> str:
    if v is None:
        return '   —  '
    return f'{int(round(v)):>5} {unit}'


def _print_table(rows: List[dict]) -> None:
    if not rows:
        print('  (no rows)')
        return
    header = (
        f'{"date":<12} {"strategy":<10} {"n":>3} '
        f'{"q2s_min":>9} {"q2s_p50":>9} {"q2s_p95":>9} {"q2s_max":>9} '
        f'{"slow":>5}  {"b2l_p50":>9} {"b2l_max":>9}'
    )
    print(header)
    print('-' * len(header))
    for r in rows:
        slow_ratio = (
            f'{r["slow_count"]}/{r["n"]}' if r['n'] else '0/0'
        )
        print(
            f'{r["date"]:<12} {r["strategy"]:<10} {r["n"]:>3} '
            f'{_fmt(r["q2s_min"])} {_fmt(r["q2s_p50"])} '
            f'{_fmt(r["q2s_p95"])} {_fmt(r["q2s_max"])} '
            f'{slow_ratio:>5}  '
            f'{_fmt(r["b2l_p50"])} {_fmt(r["b2l_max"])}'
        )


def query(since: str) -> List[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        '''
        SELECT trade_date, strategy,
               quote_to_submit_ms, bar_close_to_loop_ms
        FROM trades
        WHERE trade_date >= ?
          AND quote_to_submit_ms IS NOT NULL
        ORDER BY trade_date, strategy
        ''',
        (since,),
    )
    by_key: dict = {}
    for trade_date, strategy, q2s, b2l in cur.fetchall():
        key = (trade_date, strategy or 'unknown')
        rec = by_key.setdefault(key, {'q2s': [], 'b2l': []})
        rec['q2s'].append(q2s)
        if b2l is not None:
            rec['b2l'].append(b2l)
    conn.close()

    rows = []
    for (trade_date, strategy), rec in sorted(by_key.items()):
        q = rec['q2s']
        b = rec['b2l']
        rows.append({
            'date': trade_date,
            'strategy': strategy,
            'n': len(q),
            'q2s_min': min(q),
            'q2s_p50': statistics.median(q),
            'q2s_p95': _percentile(q, 95),
            'q2s_max': max(q),
            'slow_count': sum(1 for v in q if v > WARN_THRESHOLD_MS),
            'b2l_p50': statistics.median(b) if b else None,
            'b2l_max': max(b) if b else None,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', type=int, default=7,
                        help='Look back N days (default 7)')
    parser.add_argument('--since', type=str, default=None,
                        help='Absolute start date YYYY-MM-DD (overrides --days)')
    args = parser.parse_args()

    if args.since:
        since = args.since
    else:
        since = (date.today() - timedelta(days=args.days)).isoformat()

    print(f'Submit-latency report — since {since}')
    print(f'  Slow threshold: q2s > {WARN_THRESHOLD_MS}ms\n')

    rows = query(since)
    _print_table(rows)

    # Cross-day summary so the anomaly day jumps out.
    print('\nCross-day summary (per strategy):')
    by_strat: dict = {}
    for r in rows:
        by_strat.setdefault(r['strategy'], []).extend([
            r['q2s_p50'], r['q2s_p95'], r['q2s_max'],
        ])
    for strat in sorted(by_strat):
        max_seen = max(by_strat[strat])
        median_p50 = statistics.median(
            [r['q2s_p50'] for r in rows if r['strategy'] == strat]
        )
        n_slow_days = sum(
            1 for r in rows
            if r['strategy'] == strat and r['slow_count'] > 0
        )
        n_total_days = sum(1 for r in rows if r['strategy'] == strat)
        print(
            f'  {strat:<10} median p50={int(round(median_p50)):>5}ms  '
            f'max ever={int(round(max_seen)):>5}ms  '
            f'days with slow trades={n_slow_days}/{n_total_days}'
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
