#!/usr/bin/env python3
"""Multi-day DATA stage, step 2b — consolidate the fetch batches into year partitions.

`fetch_prices.py` writes one parquet per (adjustment, symbol batch) — the shape
that makes the pull resumable. The families want the panel by DATE, so this step
rewrites it as `prices_by_year/<adj>/year=YYYY.parquet` with `symbol` stored as a
dictionary (categorical) column.

Memory discipline: one YEAR of one adjustment is materialised at a time (≈ 3M rows
≈ 150 MB), assembled by reading each batch file with a pushdown filter on the date
range. The whole 30M-row frame is never held.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc

HERE = Path(__file__).resolve().parent
PRICES = HERE / 'prices'
OUT = HERE / 'prices_by_year'
YEARS = range(2016, 2027)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--adj', choices=['raw', 'all'], action='append',
                    help='restrict to one adjustment (repeatable); default both')
    args = ap.parse_args()
    for adj in (args.adj or ['raw', 'all']):
        src = PRICES / adj
        if not src.is_dir():
            print(f'skip {adj}: no batches', flush=True)
            continue
        dataset = ds.dataset(src, format='parquet')
        outdir = OUT / adj
        outdir.mkdir(parents=True, exist_ok=True)
        for year in YEARS:
            path = outdir / f'year={year}.parquet'
            lo, hi = f'{year}-01-01', f'{year}-12-31'
            expr = ((pc.field('date') >= pd.Timestamp(lo).date()) &
                    (pc.field('date') <= pd.Timestamp(hi).date()))
            tbl = dataset.to_table(filter=expr)
            if tbl.num_rows == 0:
                print(f'[{adj}] {year}: empty', flush=True)
                continue
            df = tbl.to_pandas()
            # a top-up pull re-fetches sessions already present: last write wins
            df = df.drop_duplicates(['symbol', 'date'], keep='last')
            df['symbol'] = df['symbol'].astype('category')
            df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
            df.to_parquet(path, index=False, compression='snappy')
            print(f'[{adj}] {year}: {len(df):,} rows, {df["symbol"].nunique():,} symbols '
                  f'-> {path.name}', flush=True)
            del df, tbl


if __name__ == '__main__':
    main()
