#!/usr/bin/env python3
"""frames15 — shared loaders, the field joins, and the cluster-robust two-sample t."""
import glob
import os
import sys

os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL', 'system')
os.environ.setdefault('MALLOC_ARENA_MAX', '2')

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
for p in ('hod_break', 'hod_frames3', 'hod_frames4', 'hod_frames5', 'hod_frames6'):
    sys.path.insert(0, f'{ROOT}/research/mature_method/{p}')
from common6 import (D6, S, S2, SPLITS, base_book, repro_line, attach_instrument,  # noqa: E402
                     load_breaks4, walk_from, mde)                                # noqa: E402,F401
from common4 import book_ranked, clustered_t, halves                              # noqa: E402,F401

D = f'{ROOT}/research/mature_method/frames15'
TEST_FROM = '2026-06-01'
RISK = 100.0
STOP_PCT = 0.02          # the declared standalone stop; R = 2 % of price


def split_of(day):
    return np.where(day < '2026-01-01', 'TRAIN', np.where(day < TEST_FROM, 'VAL', 'TEST'))


# --------------------------------------------------------------------- the daily field table
def daily_fields(cols=None, years=('2025', '2026')):
    """daily15 (the p_* AS-OF-PRIOR-CLOSE columns are computed at BUILD time, in daily.py)."""
    d = pd.concat([pd.read_parquet(f'{D}/daily15_{y}.parquet', columns=cols) for y in years],
                  ignore_index=True)
    d['symbol'] = d.symbol.astype(str)
    return d


def _stream_filter(path, cols, keyfn, want, batch=250_000):
    """Read a parquet in batches, keeping only rows whose key is in `want` (the 3 GB rail)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    parts = []
    for b in pf.iter_batches(batch_size=batch, columns=cols):
        t = b.to_pandas()
        del b
        t['symbol'] = t.symbol.astype(str)
        for c in t.columns:
            if c in ('day', 'date'):
                t[c] = t[c].astype(str)
        m = [k in want for k in keyfn(t)]
        if any(m):
            parts.append(t[m])
        del t
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def daily_for_keys(want, cols, years=('2025', '2026')):
    return pd.concat([_stream_filter(f'{D}/daily15_{y}.parquet', cols,
                                     lambda t: zip(t.symbol, t.date), want) for y in years],
                     ignore_index=True)


def hourly_for_keys(want, cols):
    return _stream_filter(f'{D}/hourly15.parquet', cols,
                          lambda t: zip(t.symbol, t.day, t.hour), want)


def daily_for_symbols(symbols, cols, years=('2025', '2026')):
    """daily15 rows for a SYMBOL SUBSET only — the 3 GB rail: never materialise the whole panel."""
    import pyarrow.parquet as pq
    syms = pa_list(symbols)
    parts = []
    for y in years:
        t = pq.read_table(f'{D}/daily15_{y}.parquet', columns=cols,
                          filters=[('symbol', 'in', syms)])
        parts.append(t.to_pandas())
        del t
    d = pd.concat(parts, ignore_index=True)
    d['symbol'] = d.symbol.astype(str)
    d['date'] = d.date.astype(str)
    return d


def pa_list(x):
    return list(dict.fromkeys(str(v) for v in x))


def hourly_fields():
    h = pd.read_parquet(f'{D}/hourly15.parquet')
    h['symbol'] = h.symbol.astype(str)
    return h


def last_closed_hour(m):
    """ET minute -> the last session hour that has CLOSED (H9 = 09:30-09:59). NaN before 10:00."""
    m = np.asarray(m, dtype=float)
    return np.where(m >= 600, np.floor(m / 60) - 1, np.nan)


# --------------------------------------------------------------------- inference
def clust_t2(x, keep, day):
    """Cluster-robust (by trading day) t of mean(x[keep]) - mean(x[~keep]). OLS on a constant+dummy."""
    x = np.asarray(x, float); k = np.asarray(keep, bool); day = np.asarray(day)
    ok = np.isfinite(x)
    x, k, day = x[ok], k[ok], day[ok]
    n1, n0 = k.sum(), (~k).sum()
    if n1 < 3 or n0 < 3:
        return np.nan, np.nan
    X = np.column_stack([np.ones(len(x)), k.astype(float)])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ x)
    e = x - X @ beta
    meat = np.zeros((2, 2))
    df = pd.DataFrame({'d': day, 'a': e, 'b': e * k.astype(float)})
    gsum = df.groupby('d')[['a', 'b']].sum().values
    for row in gsum:
        meat += np.outer(row, row)
    V = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(V[1, 1])
    return float(beta[1]), (float(beta[1] / se) if se > 0 else np.nan)


def clust_t1(x, day):
    """Cluster-robust t of the mean of x (clusters = trading days)."""
    x = np.asarray(x, float); day = np.asarray(day)
    ok = np.isfinite(x); x, day = x[ok], day[ok]
    if len(x) < 3:
        return np.nan, np.nan
    mu = x.mean()
    g = pd.Series(x - mu).groupby(day).sum().values
    se = np.sqrt((g ** 2).sum()) / len(x)
    return float(mu), (float(mu / se) if se > 0 else np.nan)


def mde_pct(x, day, power=0.8):
    """Two-sided 80 %-power MDE on the mean, with a day-cluster deflation."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    nd = pd.Series(day).nunique()
    per = len(x) / max(nd, 1)
    eff = len(x) / max(1.0, 1.0 + (per - 1) * 0.1)
    return 2.8 * x.std(ddof=1) / np.sqrt(eff)


# --------------------------------------------------------------------- week shape at $100 risk
def week_shape(df, split, rr_col='rr'):
    """Owner metrics on an arbitrary trade frame with columns day/split/<rr_col>."""
    d = df[df.split == split]
    wk = S.ALL_WEEKS[split]
    pnl = d[rr_col] * RISK
    w = pnl.groupby(d.day.str[:10].map(
        lambda x: str(pd.Period(x, freq='W-FRI')))).sum().reindex(wk).fillna(0.0)
    streak = mx = 0
    for v in (w < 0).values:
        streak = streak + 1 if v else 0
        mx = max(mx, streak)
    return dict(n=len(d), per_wk=len(d) / len(wk), green=float((w > 0).mean() * 100),
                total=float(w.sum()), wk_mean=float(w.mean()), worst=float(w.min()),
                redstreak=mx)


def null_green(df, split, rr_col='rr', draws=2000, seed=15):
    """Count-matched permutation null on green weeks (pick count per week held fixed)."""
    d = df[df.split == split]
    if len(d) < 5:
        return (np.nan, np.nan, np.nan)
    wk = S.ALL_WEEKS[split]
    w = d.day.str[:10].map(lambda x: str(pd.Period(x, freq='W-FRI')))
    cnt = w.value_counts().reindex(wk).fillna(0).astype(int).values
    pnl = (d[rr_col] * RISK).values.copy()
    rng = np.random.default_rng(seed)
    edges = np.cumsum(cnt)[:-1]
    out = np.empty(draws)
    for i in range(draws):
        p = rng.permutation(pnl)
        out[i] = np.mean([s.sum() > 0 for s in np.split(p, edges)]) * 100
    return float(np.percentile(out, 5)), float(np.percentile(out, 50)), float(np.percentile(out, 95))
