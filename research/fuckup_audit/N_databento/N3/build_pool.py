#!/usr/bin/env python3
"""Stage N3 step 2 — the paper's daily eligibility filters on the XNAS daily panel.

Zarattini-Barbon-Aziz (SSRN 4729284) §2.1, all known before the open:
  1. opening price > $5            -> here: prior close > $5 (pre-open causal); the 09:30 open is
                                      re-checked at 09:35 from the 1-min tape in rank_top20.py
  2. 14-day average daily volume >= 1,000,000 shares
  3. 14-day ATR > $0.50

APPROXIMATION: our volume is Nasdaq-venue volume (XNAS.ITCH), not consolidated tape volume.  The
ADV threshold is scaled by the measured venue share (calibrate_venue_share.py); `--adv-min` takes
the already-scaled share count.

Writes N3/pool.parquet (bar_date, symbol, prev_close, adv14, atr14) and N3/pool_counts.csv.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pqf

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')

N3 = 'research/fuckup_audit/N_databento/N3'
DAILY = f'{N3}/xnas_daily.parquet'
SYM_OK = re.compile(r'^[A-Z]{1,5}$')          # common stock / ETF symbology; no units/rights/warrants
TEST_TICKER = re.compile(r'^Z[A-Z]ZZT$')      # standing rule from the F6 reconciliation


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--adv-min', type=float, required=True,
                    help='minimum 14-day average XNAS-venue share volume')
    ap.add_argument('--start', default='2019-01-01')
    ap.add_argument('--end', default='2023-12-31')
    ap.add_argument('--out', default=f'{N3}/pool.parquet')
    args = ap.parse_args()

    # pass 1 (cheap): which symbols could ever pass?  keeps the heavy pass small.
    pf = pqf.ParquetFile(DAILY)
    keep = {}
    for batch in pf.iter_batches(batch_size=1_000_000, columns=['symbol', 'close', 'volume']):
        b = batch.to_pandas()
        b['symbol'] = b.symbol.astype(str)
        g = b.groupby('symbol', sort=False).agg(c=('close', 'max'), v=('volume', 'max'))
        for s, c, v in zip(g.index, g.c, g.v):
            p = keep.get(s)
            keep[s] = (max(p[0], c), max(p[1], v)) if p else (c, v)
        del b, g
    cand = {s for s, (c, v) in keep.items()
            if c > 5.0 and v >= args.adv_min and SYM_OK.match(s) and not TEST_TICKER.match(s)}
    print(f'pass 1: {len(keep):,} symbols -> {len(cand):,} candidates', flush=True)
    del keep

    rows = []
    for batch in pf.iter_batches(batch_size=1_000_000,
                                 columns=['bar_date', 'symbol', 'high', 'low', 'close', 'volume']):
        b = batch.to_pandas()
        b['symbol'] = b.symbol.astype(str)
        rows.append(b[b.symbol.isin(cand)])
        del b
    d = pd.concat(rows, ignore_index=True)
    del rows
    d['symbol'] = d.symbol.astype('category')
    for c in ('high', 'low', 'close'):
        d[c] = d[c].astype('float32')
    d['volume'] = d.volume.astype('float32')
    d = d.sort_values(['symbol', 'bar_date'], ignore_index=True)
    print(f'pass 2: {len(d):,} rows', flush=True)

    g = d.groupby('symbol', observed=True, sort=False)
    d['prev_close'] = g.close.shift(1)
    d['adv14'] = g.volume.shift(1).rolling(14, min_periods=14).mean().to_numpy()
    tr = np.maximum(d.high - d.low,
                    np.maximum((d.high - d.prev_close).abs(), (d.low - d.prev_close).abs()))
    d['atr14'] = tr.groupby(d.symbol, observed=True, sort=False).shift(1)\
                   .rolling(14, min_periods=14).mean().to_numpy()

    m = ((d.prev_close > 5.0) & (d.adv14 >= args.adv_min) & (d.atr14 > 0.50)
         & (d.bar_date >= args.start) & (d.bar_date <= args.end))
    pool = d.loc[m, ['bar_date', 'symbol', 'prev_close', 'adv14', 'atr14']].copy()
    pool['symbol'] = pool.symbol.astype(str)
    pool = pool.reset_index(drop=True)
    pool.to_parquet(args.out, index=False)
    cnt = pool.groupby('bar_date').size().rename('n')
    cnt.to_csv(f'{N3}/pool_counts.csv')
    print(f'pool: {len(pool):,} symbol-days, {pool.bar_date.nunique()} sessions, '
          f'{pool.symbol.nunique():,} symbols', flush=True)
    print(f'per-day pool size: median {cnt.median():.0f}, p10 {cnt.quantile(.1):.0f}, '
          f'p90 {cnt.quantile(.9):.0f}, max {cnt.max():.0f}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
