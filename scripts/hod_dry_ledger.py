#!/usr/bin/env python3
"""HOD-break DRY-RUN ledger: print day-by-day and cumulative book since a start date.

Prints: date, trades, R, $, green/red, per-symbol list, then totals with worst/best day, mean R per trade.

Usage: python3 scripts/hod_dry_ledger.py [--start YYYY-MM-DD] [--json]
Default start: 2026-09-14. Each weekday from start to today shows one line with its DRY-RUN EXECUTABLE book.
"""
import ast
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

ET = ZoneInfo('US/Eastern')

DRY_RUN_RE = re.compile(
    r'DRY-RUN EXECUTABLE book.*?:\s*(\d+)\s+trades?,\s+([-+]?[\d.]+)R,\s+\$([-+]?[\d,]+)\s*\|\s*(\[.*\])'
)


def parse_dry_run_line(output: str):
    """Parse the '  DRY-RUN EXECUTABLE book' line; return (trades, R, USD, symbols_with_r, is_green) or None.

    Line format: "  DRY-RUN EXECUTABLE book (first 12/day, 4 concurrent, logged sizes): 7 trades, +9.6R,
    $+922 | [('CRWD', 1.1), ('FBL', 2.26), ...]"
    """
    for line in output.split('\n'):
        if 'DRY-RUN EXECUTABLE book' in line:
            m = DRY_RUN_RE.search(line)
            if m:
                trades = int(m.group(1))
                r = float(m.group(2))
                usd = float(m.group(3).replace(',', ''))
                symbols_str = m.group(4)
                try:
                    symbols = ast.literal_eval(symbols_str)  # [('FBL', -1.08), ...]
                except (ValueError, SyntaxError):
                    symbols = []
                is_green = r > 0
                return (trades, r, usd, symbols, is_green)
    if 'no signals today' in output:
        # Early-return path in hod_break_eod_check.py: zero signals, no DRY-RUN line at all.
        return (0, 0.0, 0.0, [], False)
    return None


def analyze_day(day_str: str):
    """Call hod_break_eod_check.py for a day and extract DRY-RUN EXECUTABLE metrics."""
    try:
        result = subprocess.run(
            [sys.executable, 'scripts/hod_break_eod_check.py', day_str],
            capture_output=True, text=True, timeout=150, cwd=ROOT
        )
        if result.returncode != 0:
            print(f"  [{day_str}] hod_break_eod_check.py exited {result.returncode}: "
                  f"{result.stderr.strip()[-300:]}", file=sys.stderr)
            return None
        parsed = parse_dry_run_line(result.stdout)
        if parsed is None:
            print(f"  [{day_str}] WARNING: no DRY-RUN EXECUTABLE line found in output "
                  f"(no trading data for this day?)", file=sys.stderr)
        return parsed
    except subprocess.TimeoutExpired:
        print(f"  [{day_str}] ERROR: hod_break_eod_check.py timed out after 150s", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [{day_str}] ERROR: {e}", file=sys.stderr)
        return None


def main():
    argv = list(sys.argv[1:])
    start_date_str = '2026-09-14'
    json_output = False

    i = 0
    while i < len(argv):
        if argv[i] == '--start' and i + 1 < len(argv):
            start_date_str = argv[i + 1]
            i += 2
        elif argv[i] == '--json':
            json_output = True
            i += 1
        else:
            i += 1

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    today = datetime.now(timezone.utc).astimezone(ET).date()

    ledger = []  # (date_str, trades, R, USD, symbols_with_r, status)
    skipped = []

    current_date = start_date
    while current_date <= today:
        if current_date.weekday() < 5:  # weekday only
            day_str = current_date.strftime('%Y-%m-%d')
            result = analyze_day(day_str)
            if result:
                trades, dr, dusd, symbols, is_green = result
                status = 'FLAT' if trades == 0 or dr == 0 else ('GREEN' if is_green else 'RED')
                ledger.append((day_str, trades, dr, dusd, symbols, status))
            else:
                skipped.append(day_str)
        current_date += timedelta(days=1)

    if json_output:
        print(json.dumps([{
            'date': d, 'trades': t, 'R': r, 'USD': u, 'status': st,
            'symbols': [{'symbol': s, 'R': sr} for s, sr in syms]
        } for d, t, r, u, syms, st in ledger], indent=2))
    else:
        # Print day by day
        print(f"{'Date':<12} {'Trades':>6} {'R':>7} {'$':>10} {'Status':<5} Symbols (R)")
        print('-' * 100)

        total_trades = 0; total_r = 0.0; total_usd = 0.0; green_days = 0; all_daily_rs = []
        worst_day = (None, float('inf')); best_day = (None, float('-inf'))

        for day_str, trades, dr, dusd, symbols, status in ledger:
            sym_str = ', '.join(f"{s} {sr:+.2f}" for s, sr in symbols) if symbols else '(none)'
            print(f"{day_str}  {trades:6d} {dr:+7.1f} {dusd:+10,.0f}  {status:<5}  {sym_str}")
            total_trades += trades
            total_r += dr
            total_usd += dusd
            all_daily_rs.append(dr)
            if status == 'GREEN':
                green_days += 1
            if dr < worst_day[1]:
                worst_day = (day_str, dr)
            if dr > best_day[1]:
                best_day = (day_str, dr)

        print('-' * 100)
        total_days = len(ledger)
        if skipped:
            print(f"(skipped {len(skipped)} day(s) with no parseable DRY-RUN line: {', '.join(skipped)})")
        if total_days == 0:
            print("No data.")
            return
        green_pct = f"{green_days}/{total_days}"
        mean_r_per_trade = total_r / total_trades if total_trades else 0.0
        se_r = (np.std(all_daily_rs, ddof=1) / np.sqrt(len(all_daily_rs))) if len(all_daily_rs) > 1 else 0.0

        print(f"TOTAL:       {total_trades:6d} {total_r:+7.1f} {total_usd:+10,.0f}  {green_pct:<5}")
        print(f"Worst day: {worst_day[0]} ({worst_day[1]:+.1f}R)")
        print(f"Best day:  {best_day[0]} ({best_day[1]:+.1f}R)")
        print(f"Mean R/trade: {mean_r_per_trade:+.2f} ± {se_r:.2f}")


if __name__ == '__main__':
    main()
