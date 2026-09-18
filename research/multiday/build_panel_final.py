#!/usr/bin/env python3
"""Supplementary panel for the FINAL stage (F5, A2, A3, A4, F1, F6).

`panel_f3f4.npz` already carries close_adj / close_raw / adv_raw (the RAW-panel ADV fix) /
symbols / sessions / sic2 / etb on the common-stock universe + SPY.  This adds the two
matrices the last six families need and nothing else, on the SAME symbol x session index:

  * `open_adj`  -- the official opening auction print, split+dividend adjusted.  F6's
    overnight (close_{t-1} -> open_t) / intraday (open_t -> close_t) decomposition is the
    only thing that reads it; no family TRADES the open (Goyal-Jegadeesh-Wu: opening
    auctions are illiquid, AMENDMENT 2(a), inherited).

  * `splitcum`  -- the cumulative share-count multiplier, built from the EXACT daily
    corporate-action factor  f_t = (adj_t/adj_{t-1}) / (raw_t/raw_{t-1})  that
    REPORT_F4_F3.md S6 established as the ground truth (it is Alpaca's own factor, not a
    detector).  A split gives f ~ the split ratio; an ordinary dividend gives f ~ 1.000x.
    Anything with |log f| > 0.02 is treated as a share-count event; everything else is a
    dividend and leaves the count alone.  `shares(t) / splitcum(t)` is therefore a share
    count on ONE basis across time -- which is what A3's 12-month issuance ratio and A2's
    short-interest-over-shares denominator both need.  This is available here and was NOT
    available to F3's PIT re-run (which had only an unadjusted venue tape and had to
    *detect* splits at 38% precision); the difference is that we own both panels.

Writes `data/panel_final.npz`.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

D = '/home/ec2-user/onemil/research/multiday/data'
OUT = f'{D}/panel_final.npz'
YEARS = list(range(2016, 2027))
SPLIT_LOG_TOL = 0.02          # |log factor| above this is a share-count event, not a dividend


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def main():
    z = np.load(f'{D}/panel_f3f4.npz', allow_pickle=True)
    symbols = [str(s) for s in z['symbols']]
    sessions = [str(s) for s in z['sessions']]
    close_adj = z['close_adj']
    close_raw = z['close_raw']
    sidx = {s: i for i, s in enumerate(symbols)}
    didx = {d: i for i, d in enumerate(sessions)}
    n_s, n_d = close_adj.shape
    log(f'panel {n_s} x {n_d}')

    open_adj = np.full((n_s, n_d), np.nan, dtype=np.float32)
    tset = set(symbols)
    for y in YEARS:
        df = pq.read_table(f'{D}/prices_by_year/all/year={y}.parquet',
                           columns=['symbol', 'date', 'open']).to_pandas()
        df['symbol'] = df['symbol'].astype(str)
        df = df[df['symbol'].isin(tset)]
        si = df['symbol'].map(sidx).to_numpy(dtype=np.int32)
        di = df['date'].astype(str).map(didx)
        ok = di.notna().to_numpy()
        open_adj[si[ok], di.fillna(0).to_numpy(dtype=np.int32)[ok]] = \
            df['open'].to_numpy(dtype=np.float32)[ok]
        del df
        log(f'  {y} opens loaded')

    # ---- exact corporate-action factor -> cumulative share multiplier -----------
    with np.errstate(invalid='ignore', divide='ignore'):
        ra = close_adj[:, 1:] / close_adj[:, :-1]
        rr = close_raw[:, 1:] / close_raw[:, :-1]
        fac = ra / rr
    lf = np.log(np.where(np.isfinite(fac) & (fac > 0), fac, 1.0))
    ev = np.abs(lf) > SPLIT_LOG_TOL
    step = np.where(ev, lf, 0.0)
    splitcum = np.ones((n_s, n_d), dtype=np.float32)
    splitcum[:, 1:] = np.exp(np.cumsum(step, axis=1)).astype(np.float32)
    n_ev = int(ev.sum())
    log(f'share-count events detected: {n_ev:,} on {int((ev.any(axis=1)).sum()):,} symbols '
        f'({n_ev / max(np.isfinite(fac).sum(), 1):.6%} of finite factor cells)')
    # sanity: the biggest multipliers
    tot = splitcum[:, -1]
    ordr = np.argsort(-np.abs(np.log(np.where(tot > 0, tot, 1.0))))[:8]
    for i in ordr:
        log(f'    {symbols[i]:8s} cumulative share multiplier {tot[i]:.4g}')

    np.savez_compressed(OUT, open_adj=open_adj, splitcum=splitcum)
    log(f'wrote {OUT}')


if __name__ == '__main__':
    sys.exit(main())
