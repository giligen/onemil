#!/usr/bin/env python3
"""Build the compact matrix panel for the F2 / A1 families.

Reads the year-partitioned Alpaca SIP daily parquets (raw + adjusted) and writes ONE npz holding
dense (symbol x session) float32 matrices for the ~3.8K common stocks that have an Item-2.02 8-K:

    close_adj  -- split/dividend adjusted close (all features/returns)
    close_raw  -- unadjusted close (the >= $5 gate and share counts)
    dvol       -- dollar volume (vwap * volume), for ADV20 and the impact model

Plus the global session calendar (from SPY, which trades every session) and the symbol index.
Memory: 3 matrices of ~3.8K x ~2.7K float32 = ~125 MB; one price-year at a time on the read side.
"""
import sys, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

D = '/home/ec2-user/onemil/research/multiday/data'
OUT = f'{D}/panel_f2a1.npz'
YEARS = list(range(2016, 2027))


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def main():
    uni = pd.read_parquet(f'{D}/universe.parquet',
                          columns=['symbol', 'kind', 'easy_to_borrow', 'shortable', 'exchange'])
    ev = pd.read_parquet(f'{D}/earnings_events.parquet', columns=['symbol'])
    ev_syms = set(ev['symbol'].dropna().unique())
    common = set(uni.loc[uni['kind'] == 'common', 'symbol'])
    target = sorted((common & ev_syms) | {'SPY'})
    log(f'universe: {len(uni)} | common {len(common)} | event syms {len(ev_syms)} | target {len(target)}')

    tset = set(target)
    sidx = {s: i for i, s in enumerate(target)}

    # pass 1: session calendar from SPY
    dates = []
    for y in YEARS:
        t = pq.read_table(f'{D}/prices_by_year/all/year={y}.parquet', columns=['symbol', 'date'])
        df = t.to_pandas()
        dates.append(df.loc[df['symbol'] == 'SPY', 'date'].values)
        del t, df
    cal = np.unique(np.concatenate(dates))
    cal = np.sort(cal)
    didx = {d: i for i, d in enumerate(cal)}
    log(f'sessions: {len(cal)}  {cal[0]} -> {cal[-1]}')

    n_s, n_d = len(target), len(cal)
    close_adj = np.full((n_s, n_d), np.nan, dtype=np.float32)
    close_raw = np.full((n_s, n_d), np.nan, dtype=np.float32)
    dvol = np.full((n_s, n_d), np.nan, dtype=np.float32)

    for y in YEARS:
        for adj, mats in (('all', ('close_adj', 'dvol')), ('raw', ('close_raw',))):
            cols = ['symbol', 'date', 'close'] + (['volume', 'vwap'] if adj == 'all' else [])
            df = pq.read_table(f'{D}/prices_by_year/{adj}/year={y}.parquet', columns=cols).to_pandas()
            df['symbol'] = df['symbol'].astype(str)
            df = df[df['symbol'].isin(tset)]
            si = df['symbol'].map(sidx).to_numpy(dtype=np.int32)
            di = df['date'].map(didx)
            ok = di.notna().to_numpy()
            di = di.fillna(0).to_numpy(dtype=np.int32)
            si, di = si[ok], di[ok]
            c = df['close'].to_numpy(dtype=np.float32)[ok]
            if adj == 'all':
                close_adj[si, di] = c
                dv = (df['vwap'].to_numpy(dtype=np.float64) * df['volume'].to_numpy(dtype=np.float64))[ok]
                dvol[si, di] = dv.astype(np.float32)
            else:
                close_raw[si, di] = c
            del df, si, di, c
        log(f'  {y} loaded')

    np.savez_compressed(OUT, close_adj=close_adj, close_raw=close_raw, dvol=dvol,
                        symbols=np.array(target, dtype=object),
                        sessions=np.array([str(d) for d in cal], dtype=object))
    log(f'wrote {OUT}  coverage close_adj {np.isfinite(close_adj).mean():.3f}')


if __name__ == '__main__':
    sys.exit(main())
