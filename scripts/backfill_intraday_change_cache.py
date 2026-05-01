#!/usr/bin/env python3
"""Backfill intraday_change_at_entry on the bull-flag cache CSV.

The column was added in commit 8e23215 (2026-04-17). Pre-existing cache
rows have it blank/NaN, which makes the BT's Stage-2 intraday-change
filter fail open (passthrough) for those rows. This script computes
the field in-place from the 1-min bars cache so the filter can run
honestly without a full cache rebuild.

Formula mirrors scanner.realtime_scanner._run_intraday_cycle:
  ic = max(gap_pct, range_pct) at the entry bar minute
where:
  gap_pct   = (close - prev_close) / prev_close * 100
  range_pct = (day_high - day_low) / day_low * 100
both computed from 9:30 ET to entry minute (inclusive).

Writes back to the same CSV (in place). Backup recommended:
  cp data/bull_flag_cache_e50_x30.csv data/bull_flag_cache_e50_x30.csv.bak
before running.
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
CACHE_CSV = ROOT / 'data' / 'bull_flag_cache_e50_x30.csv'
DB = ROOT / 'data' / 'cache.db'


def parse_entry_time(date_str: str, entry_time_et: str) -> pd.Timestamp:
    et = pd.Timestamp(f"{date_str} {entry_time_et}", tz='US/Eastern')
    return et.tz_convert('UTC')


def compute_ic(con, symbol: str, date_str: str,
               entry_ts_utc: pd.Timestamp,
               prev_close: float) -> float | None:
    df = pd.read_sql_query(
        "SELECT timestamp, high, low, close "
        "FROM intraday_bars_1min WHERE symbol=? AND bar_date=? "
        "ORDER BY timestamp",
        con, params=(symbol, date_str),
    )
    if df.empty or prev_close <= 0:
        return None
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    rth_start = pd.Timestamp(f"{date_str} 13:30", tz='UTC')
    pre_entry = df[(df['timestamp'] >= rth_start) &
                   (df['timestamp'] <= entry_ts_utc)]
    if pre_entry.empty:
        return None
    day_high = pre_entry['high'].max()
    day_low = pre_entry['low'].min()
    cur = pre_entry['close'].iloc[-1]
    gap_pct = (cur - prev_close) / prev_close * 100
    range_pct = ((day_high - day_low) / day_low * 100) if day_low > 0 else 0
    return max(gap_pct, range_pct)


def get_prev_close(con, symbol: str, date_str: str) -> float | None:
    row = con.execute(
        "SELECT close FROM daily_bars WHERE symbol=? AND bar_date<? "
        "ORDER BY bar_date DESC LIMIT 1",
        (symbol, date_str),
    ).fetchone()
    return float(row[0]) if row else None


def main():
    print(f"Reading cache: {CACHE_CSV}")
    rows = []
    with open(CACHE_CSV) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    print(f"  {len(rows)} rows; ic-column present: "
          f"{'intraday_change_at_entry' in fieldnames}")
    if 'intraday_change_at_entry' not in fieldnames:
        print("ERROR: column missing. Rebuild cache via --build-cache.")
        sys.exit(1)

    con = sqlite3.connect(str(DB))

    backfilled = 0
    skipped = 0
    already = 0
    for row in rows:
        existing = row.get('intraday_change_at_entry', '')
        if existing not in ('', None, 'None'):
            try:
                float(existing)
                already += 1
                continue
            except ValueError:
                pass

        symbol = row.get('symbol', '')
        date_str = row.get('date', '')
        entry_time_et = row.get('entry_time_et', '')
        if not symbol or not date_str or not entry_time_et:
            skipped += 1
            continue

        try:
            entry_ts = parse_entry_time(date_str, entry_time_et)
        except Exception:
            skipped += 1
            continue

        prev_close = get_prev_close(con, symbol, date_str)
        if prev_close is None:
            skipped += 1
            continue

        ic = compute_ic(con, symbol, date_str, entry_ts, prev_close)
        if ic is None:
            skipped += 1
            continue

        row['intraday_change_at_entry'] = f"{ic:.2f}"
        backfilled += 1

    print(f"  backfilled: {backfilled}, already-set: {already}, "
          f"skipped (no data): {skipped}")

    # Write back in place
    with open(CACHE_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  wrote {len(rows)} rows to {CACHE_CSV}")
    con.close()


if __name__ == '__main__':
    main()
