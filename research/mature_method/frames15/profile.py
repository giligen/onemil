#!/usr/bin/env python3
"""frames15 step 3 — THE PER-STOCK HOURLY PROFILE and the V1/V2/V4 fields.

    share_h(s,t)    = mean over the symbol's prior AVAILABLE sessions (rolling 20, min 3, and the
                      most recent prior appearance within 60 sessions) of hour-h vol / that
                      session's RTH vol                      -- a SHAPE, from the symbol's own tape
    hourmean_h(s,t) = adv20(s,t) * share_h(s,t)               -- LEVEL from the DENSE daily panel
    hrv_h(s,t)      = hour-h volume on t / hourmean_h(s,t)    -- V1, bar-level, NOT cumulative
    hrv_raw_h       = hour-h volume / the symbol's own prior mean hour-h volume (the sensitivity)

Every denominator uses sessions STRICTLY BEFORE t; a same-day hour never enters its own denominator.
Writes `hourly15.parquet`.
"""
import glob
import os
import sys

os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL', 'system')
os.environ.setdefault('MALLOC_ARENA_MAX', '2')

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
D = f'{ROOT}/research/mature_method/frames15'
WIN, MINP, STALE = 20, 3, 60


def main():
    files = sorted(glob.glob(f'{D}/hourly_*.parquet'))
    h = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    h['symbol'] = h.symbol.astype(str)
    h['day'] = h.day.astype(str)
    print(f'[profile] {len(h):,} symbol-hours, {h.symbol.nunique():,} symbols, '
          f'{h.day.nunique()} sessions', flush=True)

    sess = sorted(h.day.unique())
    sidx = {d: i for i, d in enumerate(sess)}
    h['di'] = h.day.map(sidx).astype(np.int32)
    h['share'] = np.where(h.day_v > 0, h.v / h.day_v, np.nan)
    h['hour_ret'] = np.where(h.o > 0, h.c / h.o - 1.0, np.nan)

    h = h.sort_values(['symbol', 'hour', 'di'], kind='mergesort')
    g = h.groupby(['symbol', 'hour'], sort=False)
    h['pshare'] = g.share.transform(lambda s: s.shift(1).rolling(WIN, min_periods=MINP).mean())
    h['pvmean'] = g.v.transform(lambda s: s.shift(1).rolling(WIN, min_periods=MINP).mean())
    h['pdi'] = g.di.shift(1)
    fresh = (h.di - h.pdi) <= STALE
    h.loc[~fresh.fillna(False), ['pshare', 'pvmean']] = np.nan

    dd = pd.concat([pd.read_parquet(p, columns=['symbol', 'date', 'adv20', 'close', 'adv20d'])
                    for p in sorted(glob.glob(f'{D}/daily15_*.parquet'))], ignore_index=True)
    dd = dd.rename(columns={'date': 'day'})
    dd['symbol'] = dd.symbol.astype(str)
    h = h.merge(dd, on=['symbol', 'day'], how='left')

    h['hourmean'] = h.adv20 * h.pshare
    h['hrv'] = np.where(h.hourmean > 0, h.v / h.hourmean, np.nan)
    h['hrv_raw'] = np.where(h.pvmean > 0, h.v / h.pvmean, np.nan)

    # --- V2: sustained over the last N CLOSED hours, per symbol-day, hours are consecutive
    h = h.sort_values(['symbol', 'day', 'hour'], kind='mergesort')
    gd = h.groupby(['symbol', 'day'], sort=False)
    h['hour_prev'] = gd.hour.shift(1)
    h['hrv_p1'] = gd.hrv.shift(1)
    h['hour_prev2'] = gd.hour.shift(2)
    h['hrv_p2'] = gd.hrv.shift(2)
    c1 = (h.hour - h.hour_prev) == 1
    c2 = (h.hour - h.hour_prev2) == 2
    for k in (2.0, 3.0):
        tag = int(k)
        h[f'sus{tag}_2'] = c1 & (h.hrv >= k) & (h.hrv_p1 >= k)
        h[f'sus{tag}_3'] = c1 & c2 & (h.hrv >= k) & (h.hrv_p1 >= k) & (h.hrv_p2 >= k)

    keep = ['symbol', 'day', 'di', 'hour', 'v', 'day_v', 'share', 'pshare', 'hourmean', 'hrv',
            'hrv_raw', 'hour_ret', 'adv20', 'adv20d', 'close', 'sus2_2', 'sus2_3', 'sus3_2',
            'sus3_3']
    h[keep].to_parquet(f'{D}/hourly15.parquet', index=False)
    cov = h.hrv.notna().mean() * 100
    print(f'[profile] hourly15.parquet {len(h):,} rows; hrv coverage {cov:.1f}% '
          f'hrv_raw {h.hrv_raw.notna().mean()*100:.1f}%', flush=True)
    q = h.hrv.dropna()
    print(f'[profile] hrv quantiles p25 {q.quantile(.25):.2f} med {q.median():.2f} '
          f'p75 {q.quantile(.75):.2f} p90 {q.quantile(.90):.2f} p99 {q.quantile(.99):.2f}',
          flush=True)
    print('[profile] fire rates: hrv>=2 %.3f  hrv>=3 %.3f  sus2_2 %.4f  sus2_3 %.4f'
          % ((q >= 2).mean(), (q >= 3).mean(), h.sus2_2.mean(), h.sus2_3.mean()), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
