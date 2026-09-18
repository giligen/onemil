#!/usr/bin/env python3
"""Stage N2 step 1 — pull EQUS.SUMMARY ohlcv-1d ALL_SYMBOLS 2024-07-01 -> 2024-12-31.

The 252-day lookback K2 (52-week-high breakout) needs half a year of history BEFORE the existing
Databento daily panel starts (2025-01-02).  This writes a NEW parquet in the SAME column layout as
`data/research/databento/equs_daily_2025_2026.parquet` (bar_date, symbol, instrument_id, open,
high, low, close, volume).  The existing parquet is never opened for writing.

Usage:
    python3 fetch_daily_2024h2.py --cost-only
    python3 fetch_daily_2024h2.py --budget 5
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')
load_dotenv('/home/ec2-user/onemil/.env')

import databento as db  # noqa: E402

DATASET = 'EQUS.SUMMARY'
SCHEMA = 'ohlcv-1d'
START = '2024-07-01'
END = '2025-01-01'          # exclusive upper bound -> last bar 2024-12-31
OUT_DBN = 'data/research/databento/equs_summary_ohlcv1d_ALL_20240701_20241231.dbn.zst'
OUT_PARQUET = 'data/research/databento/equs_daily_2024H2.parquet'
REF_PARQUET = 'data/research/databento/equs_daily_2025_2026.parquet'


def log(m):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {m}', flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cost-only', action='store_true')
    ap.add_argument('--budget', type=float, default=5.0)
    args = ap.parse_args()

    client = db.Historical(os.environ['DATABENTO_API_KEY'])
    cost = float(client.metadata.get_cost(dataset=DATASET, schema=SCHEMA, symbols='ALL_SYMBOLS',
                                          stype_in='raw_symbol', start=START, end=END))
    log(f'PRICED {DATASET} {SCHEMA} ALL_SYMBOLS {START}..{END}: ${cost:.4f}')
    if args.cost_only:
        return 0
    if cost > args.budget:
        log(f'STOP: ${cost:.2f} exceeds budget ${args.budget:.2f}')
        return 2

    if os.path.exists(OUT_PARQUET):
        log(f'{OUT_PARQUET} already exists — nothing to do')
        return 0

    log('fetching…')
    data = client.timeseries.get_range(dataset=DATASET, schema=SCHEMA, symbols='ALL_SYMBOLS',
                                       stype_in='raw_symbol', start=START, end=END)
    if not os.path.exists(OUT_DBN):
        data.to_file(OUT_DBN)
        log(f'raw DBN -> {OUT_DBN} ({os.path.getsize(OUT_DBN) / 1e6:.1f} MB)')

    df = data.to_df(price_type='float')
    log(f'{len(df):,} records')
    df = df.reset_index()
    ts = pd.to_datetime(df['ts_event'], utc=True)
    out = pd.DataFrame({
        'bar_date': ts.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d'),
        'symbol': df['symbol'].astype(object),
        'instrument_id': df['instrument_id'].astype('uint32'),
        'open': df['open'].astype('float64'),
        'high': df['high'].astype('float64'),
        'low': df['low'].astype('float64'),
        'close': df['close'].astype('float64'),
        'volume': df['volume'].astype('uint64'),
    })
    out = out.sort_values(['bar_date', 'symbol'], kind='mergesort').reset_index(drop=True)
    out.to_parquet(OUT_PARQUET, index=False)
    log(f'wrote {OUT_PARQUET}: {len(out):,} rows  {out.bar_date.min()}..{out.bar_date.max()}  '
        f'{out.symbol.nunique():,} symbols  ({os.path.getsize(OUT_PARQUET) / 1e6:.1f} MB)')

    ref = pd.read_parquet(REF_PARQUET, columns=None, filters=[('bar_date', '==', '2025-01-02')])
    log('layout check vs reference: '
        f'cols equal={list(ref.columns) == list(out.columns)}  '
        f'dtypes equal={[str(x) for x in ref.dtypes] == [str(x) for x in out.dtypes]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
