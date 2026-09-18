#!/usr/bin/env python3
"""Stage N1 step 1 — pull EQUS.MINI `tbbo` for every ORB candidate symbol-day.

Window: 09:29:30 -> 09:46:00 America/New_York per date (the 5-min opening range
plus the breakout window). One `timeseries.get_range` per DATE with all that
day's symbols. Resumable: a date whose parquet already exists is skipped.

Usage:
    python3 fetch_tbbo.py --cost-only     # metadata.get_cost for the whole plan
    python3 fetch_tbbo.py --budget 150    # fetch (stops if priced cost > budget)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')

load_dotenv('/home/ec2-user/onemil/.env')

import databento as db  # noqa: E402

D1 = 'research/fuckup_audit/D1_orb'
OUT = 'research/fuckup_audit/N_databento/N1/tbbo'
DATASET = 'EQUS.MINI'
SCHEMA = 'tbbo'
START_ET = '09:29:30'
END_ET = '09:46:00'


def candidate_plan() -> dict:
    """{date_str: [symbols]} from the D1 candidate dump (ticker 'NA' is real)."""
    d = pd.read_csv(f'{D1}/candidates_dump.csv', usecols=['symbol', 'date'],
                    keep_default_na=False, na_values=[''])
    d['date'] = pd.to_datetime(d['date']).dt.strftime('%Y-%m-%d')
    return {k: sorted(set(v)) for k, v in d.groupby('date')['symbol']}


def utc_window(date_str: str) -> tuple[str, str]:
    """ET window -> UTC ISO strings for that calendar date (DST-correct)."""
    s = pd.Timestamp(f'{date_str} {START_ET}', tz='America/New_York')
    e = pd.Timestamp(f'{date_str} {END_ET}', tz='America/New_York')
    return (s.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%S'),
            e.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%S'))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cost-only', action='store_true')
    ap.add_argument('--budget', type=float, default=150.0)
    ap.add_argument('--cost-sample', type=int, default=0,
                    help='price only the first N dates and extrapolate (cost calls are slow)')
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    plan = candidate_plan()
    dates = sorted(plan)
    print(f"plan: {len(dates)} dates, {sum(len(v) for v in plan.values())} symbol-days",
          flush=True)

    client = db.Historical(os.environ['DATABENTO_API_KEY'])

    # ---- price first ----
    price_dates = dates if not args.cost_sample else dates[:args.cost_sample]
    total = 0.0
    t0 = time.time()
    for i, dt in enumerate(price_dates, 1):
        s, e = utc_window(dt)
        total += float(client.metadata.get_cost(
            dataset=DATASET, schema=SCHEMA, symbols=plan[dt], start=s, end=e,
            stype_in='raw_symbol'))
        if i % 25 == 0 or i == len(price_dates):
            print(f"  priced {i}/{len(price_dates)} dates  running ${total:.2f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    est = total * (len(dates) / len(price_dates))
    print(f"PRICED: ${total:.2f} over {len(price_dates)} dates -> whole-plan estimate ${est:.2f}",
          flush=True)
    if args.cost_only:
        return 0
    if est > args.budget:
        print(f"STOP: priced cost ${est:.2f} exceeds budget ${args.budget:.2f}", flush=True)
        return 2

    # ---- fetch ----
    spent = 0.0
    done = 0
    t0 = time.time()
    for i, dt in enumerate(dates, 1):
        path = f'{OUT}/{dt}.parquet'
        if os.path.exists(path):
            done += 1
            continue
        s, e = utc_window(dt)
        for attempt in range(4):
            try:
                data = client.timeseries.get_range(
                    dataset=DATASET, schema=SCHEMA, symbols=plan[dt],
                    start=s, end=e, stype_in='raw_symbol')
                break
            except Exception as ex:                       # network / rate limit
                print(f"  WARN {dt} attempt {attempt + 1}: {type(ex).__name__}: {ex}",
                      flush=True)
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        df = data.to_df()
        if len(df):
            df = df.reset_index()
            keep = [c for c in ('ts_event', 'ts_recv', 'symbol', 'price', 'size',
                                'side', 'bid_px_00', 'ask_px_00', 'bid_sz_00',
                                'ask_sz_00', 'sequence') if c in df.columns]
            df = df[keep]
        df.to_parquet(path, index=False, compression='zstd')
        done += 1
        if done % 10 == 0 or i == len(dates):
            print(f"  fetched {done}/{len(dates)} dates  last={dt} rows={len(df)}  "
                  f"elapsed {time.time() - t0:.0f}s", flush=True)
    print(f"DONE: {done} dates on disk", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
