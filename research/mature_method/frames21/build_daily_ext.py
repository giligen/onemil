#!/usr/bin/env python3
"""frames21 — the DENSE daily panel for the 2024H2 extension window, built with the UNCHANGED logic
of frames15/daily.py (same G class, same field formulas), only the date range and output path
differ. Loads the FULL 2024 raw panel (research/multiday/data/prices_by_year, already on disk, zero
cost) so adv20/interest5/ret1/ret5 have a proper >=20-session trailing lookback at 2024-07-01 (the
shipped daily15_2024.parquet truncates its LOADED frame at 2024-11-01, which starves the first ~20
sessions of Nov of lookback -- documented in frames21/DATA.md, not fixed here, out of scope).

Output: frames21/daily21_2024h2.parquet, rows 2024-07-01..2024-12-31 only (the extension window).
Caveat: fwd1/fwd2/fwd5 (forward returns) near 2024-12-31 need 2025-01 data this script does not
load, so they under-cover at the trailing edge -- irrelevant to the mirror-short signal (armB_intra
never reads daily15 fields), kept only for parity with frames15's schema.
"""
import gc
import os
import sys

os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL', 'system')
os.environ.setdefault('MALLOC_ARENA_MAX', '2')

import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402
import pyarrow as pa                                           # noqa: E402
import pyarrow.parquet as pq                                   # noqa: E402

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
D = f'{ROOT}/research/mature_method/frames21'
PAN = f'{ROOT}/research/multiday/data/prices_by_year'
YEARS = (2024,)
OUT_FROM, OUT_TO = '2024-07-01', '2024-12-31'
F32 = np.float32


def load(kind, cols):
    parts = []
    for y in YEARS:
        t = pq.read_table(f'{PAN}/{kind}/year={y}.parquet', columns=cols)
        df = t.to_pandas(self_destruct=True, split_blocks=True)
        del t
        df['date'] = df.date.astype(str)
        for c in cols:
            if c not in ('symbol', 'date'):
                df[c] = df[c].astype(F32)
        parts.append(df)
        gc.collect()
    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    return df


class G:
    def __init__(self, gid):
        self.gid = gid
        n = len(gid)
        self.n = n
        newg = np.empty(n, dtype=bool)
        newg[0] = True
        newg[1:] = gid[1:] != gid[:-1]
        gstart_at = np.flatnonzero(newg)
        gi = np.cumsum(newg) - 1
        self.pos = np.arange(n) - gstart_at[gi]
        glen = np.diff(np.append(gstart_at, n))
        self.rem = glen[gi] - 1 - self.pos

    def shift(self, x, k):
        out = np.full(self.n, np.nan, dtype=np.float64)
        if k < self.n:
            out[k:] = x[:-k] if k else x
        out[self.pos < k] = np.nan
        return out

    def lead(self, x, k):
        out = np.full(self.n, np.nan, dtype=np.float64)
        if k < self.n:
            out[:-k] = x[k:]
        out[self.rem < k] = np.nan
        return out

    def prior_mean(self, x, w):
        P = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x, nan=0.0))])
        out = np.full(self.n, np.nan, dtype=np.float64)
        i = np.arange(self.n)
        ok = self.pos >= w
        out[ok] = (P[i[ok]] - P[i[ok] - w]) / w
        return out

    def trail_sum_incl(self, x, w):
        P = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x, nan=0.0))])
        out = np.full(self.n, np.nan, dtype=np.float64)
        i = np.arange(self.n)
        ok = self.pos >= (w - 1)
        out[ok] = P[i[ok] + 1] - P[i[ok] + 1 - w]
        return out


def main():
    print('[build_daily_ext] loading FULL 2024 raw panel ...', flush=True)
    d = load('raw', ['symbol', 'date', 'high', 'low', 'close', 'volume'])
    d = d.sort_values(['symbol', 'date'], kind='mergesort', ignore_index=True)
    print(f'  raw {len(d):,} rows / {d.symbol.nunique():,} symbols', flush=True)

    a = load('all', ['symbol', 'date', 'close']).rename(columns={'close': 'aclose'})
    d = d.merge(a, on=['symbol', 'date'], how='left')
    del a
    gc.collect()

    gid = pd.factorize(d.symbol, sort=False)[0]
    g = G(gid)
    vol = d.volume.values.astype(np.float64)
    cls = d.close.values.astype(np.float64)
    acl = d.aclose.values.astype(np.float64)
    hi_, lo_ = d.high.values.astype(np.float64), d.low.values.astype(np.float64)

    adv20 = g.prior_mean(vol, 20)
    with np.errstate(divide='ignore', invalid='ignore'):
        rvd = vol / np.where(adv20 > 0, adv20, np.nan)
    interest5 = g.trail_sum_incl((rvd >= 1.5).astype(np.float64), 5)
    interest5[np.isnan(rvd)] = np.nan

    ret1 = acl / g.shift(acl, 1) - 1.0
    ret5 = acl / g.shift(acl, 5) - 1.0
    rng = hi_ - lo_
    with np.errstate(divide='ignore', invalid='ignore'):
        clspos = np.where(rng > 0, (cls - lo_) / rng, np.nan)
    weak = (rvd >= 1.5) & (ret1 < 0) & (clspos <= 1 / 3)
    rret = cls / g.shift(cls, 1) - 1.0
    ca_bad = np.abs(rret - ret1) > 0.01

    out = {'symbol': d.symbol.values, 'date': d.date.values, 'close': d.close.values,
           'volume': d.volume.values}
    del d, hi_, lo_, rng, rret
    gc.collect()
    out['adv20'] = adv20.astype(F32)
    out['adv20d'] = (adv20 * cls).astype(F32)
    out['rvd'] = rvd.astype(F32)
    out['interest5'] = interest5.astype(F32)
    for j in range(1, 6):
        out[f'rvd_{j}'] = g.shift(rvd, j - 1).astype(F32)
    out['ret1'] = ret1.astype(F32)
    out['ret5'] = ret5.astype(F32)
    out['clspos'] = clspos.astype(F32)
    out['weak'] = weak
    out['ca_bad'] = ca_bad
    for h in (1, 2, 5):
        out[f'fwd{h}'] = (g.lead(acl, h) / acl - 1.0).astype(F32)
    out['p_interest5'] = g.shift(interest5, 1).astype(F32)
    out['p_rvd'] = g.shift(rvd, 1).astype(F32)
    out['p_ret1'] = g.shift(ret1, 1).astype(F32)
    out['p_ret5'] = g.shift(ret5, 1).astype(F32)
    out['p_clspos'] = g.shift(clspos, 1).astype(F32)
    out['p_close'] = g.shift(cls, 1).astype(F32)
    out['p_weak'] = g.shift(weak.astype(np.float64), 1) > 0.5
    del adv20, rvd, interest5, ret1, ret5, clspos, weak, ca_bad, acl, cls, vol, g
    gc.collect()

    dates = out['date']
    m = (dates >= OUT_FROM) & (dates <= OUT_TO)
    names, arrs = [], []
    for k, v in out.items():
        sl = v[m]
        arrs.append(pa.array(pd.Categorical(sl)) if k in ('symbol', 'date') else pa.array(sl))
        names.append(k)
    tbl = pa.Table.from_arrays(arrs, names=names)
    pq.write_table(tbl, f'{D}/daily21_2024h2.parquet')
    print(f'[build_daily_ext] wrote daily21_2024h2.parquet {tbl.num_rows:,} rows '
          f'({OUT_FROM}..{OUT_TO})', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
