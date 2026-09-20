#!/usr/bin/env python3
"""frames15 step 2 — the DENSE daily panel: session RV against the symbol's own trailing-20 mean.

Source: `research/multiday/data/prices_by_year/{raw,all}` (the point-in-time daily panel used by
frames13 F40 — 11,823 symbols, delisted included), 2024-11 -> 2026. Read-only.

Per (symbol, session t), everything strictly causal at t's CLOSE:
  adv20      mean daily volume over the 20 sessions STRICTLY BEFORE t
  rvd        volume(t) / adv20(t)                        -- t's own session RV
  interest5  # of the 5 sessions ENDING AT t with rvd >= 1.5       (0..5)
  rvd_1..5   rvd at t, t-1, ... t-4  (the "prior j sessions" ladder)
  ret1/ret5  adjusted 1- and 5-session returns to t
  clspos     (close - low) / (high - low) on t
  weak       rvd >= 1.5 and ret1 < 0 and clspos <= 1/3   -- V5, as of t's close
  p_*        the same fields AS OF THE PRIOR CLOSE (what a decision inside session t may use)
  fwd1/2/5   adjusted close(t+h)/close(t) - 1            -- the multi-day hold legs
  ca_bad     raw-vs-adjusted 1-session return disagree by > 1 pp (frames13 G-SCALE rail)

All group operations are numpy on the sorted arrays (contiguous groups) — pandas groupby transforms
do not fit inside the 3 GB `ulimit -v` rail. Writes `daily15_{year}.parquet`.
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
D = f'{ROOT}/research/mature_method/frames15'
PAN = f'{ROOT}/research/multiday/data/prices_by_year'
YEARS = (2024, 2025, 2026)
FROM = '2024-11-01'
F32 = np.float32


def load(kind, cols):
    parts = []
    for y in YEARS:
        t = pq.read_table(f'{PAN}/{kind}/year={y}.parquet', columns=cols)
        df = t.to_pandas(self_destruct=True, split_blocks=True)
        del t
        df['date'] = df.date.astype(str)
        if y == 2024:
            df = df[df.date >= FROM]
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
    """Contiguous-group array ops on a frame sorted by (symbol, date)."""

    def __init__(self, gid):
        self.gid = gid
        n = len(gid)
        self.n = n
        start = np.zeros(n, dtype=np.int64)
        newg = np.empty(n, dtype=bool)
        newg[0] = True
        newg[1:] = gid[1:] != gid[:-1]
        gstart_at = np.flatnonzero(newg)
        gi = np.cumsum(newg) - 1
        self.pos = np.arange(n) - gstart_at[gi]                # index within the group
        glen = np.diff(np.append(gstart_at, n))
        self.rem = glen[gi] - 1 - self.pos                     # rows remaining after this one
        del start

    def shift(self, x, k):
        """x at t-k inside the group, NaN where unavailable."""
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
        """mean of the w values STRICTLY BEFORE t."""
        P = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x, nan=0.0))])
        out = np.full(self.n, np.nan, dtype=np.float64)
        i = np.arange(self.n)
        ok = self.pos >= w
        out[ok] = (P[i[ok]] - P[i[ok] - w]) / w
        return out

    def trail_sum_incl(self, x, w):
        """sum of the w values ENDING AT t (inclusive)."""
        P = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x, nan=0.0))])
        out = np.full(self.n, np.nan, dtype=np.float64)
        i = np.arange(self.n)
        ok = self.pos >= (w - 1)
        out[ok] = P[i[ok] + 1] - P[i[ok] + 1 - w]
        return out


def main():
    print('[daily] loading raw panel ...', flush=True)
    d = load('raw', ['symbol', 'date', 'high', 'low', 'close', 'volume'])
    d = d.sort_values(['symbol', 'date'], kind='mergesort', ignore_index=True)
    print(f'  raw {len(d):,} rows / {d.symbol.nunique():,} symbols', flush=True)

    a = load('all', ['symbol', 'date', 'close']).rename(columns={'close': 'aclose'})
    d = d.merge(a, on=['symbol', 'date'], how='left')
    del a
    gc.collect()
    print('  adjusted close joined', flush=True)

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
    print(f'[daily] computed {len(dates):,} rows; writing per year ...', flush=True)
    bnd = {'2024': ('2024-01-01', '2025-01-01'), '2025': ('2025-01-01', '2026-01-01'),
           '2026': ('2026-01-01', '2027-01-01')}
    cov = {}
    for y in ('2024', '2025', '2026'):
        lo, hi2 = bnd[y]
        m = (dates >= lo) & (dates < hi2)
        if not m.any():
            continue
        names, arrs = [], []
        for k, v in out.items():                      # column by column: no pandas consolidation
            sl = v[m]
            arrs.append(pa.array(pd.Categorical(sl)) if k in ('symbol', 'date') else pa.array(sl))
            names.append(k)
            del sl
        tbl = pa.Table.from_arrays(arrs, names=names)
        del arrs
        gc.collect()
        pq.write_table(tbl, f'{D}/daily15_{y}.parquet')
        n = tbl.num_rows
        print(f'  wrote daily15_{y}.parquet {n:,} rows', flush=True)
        del tbl, m
        gc.collect()
    return 0


if __name__ == '__main__':
    sys.exit(main())
