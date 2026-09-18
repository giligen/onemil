#!/usr/bin/env python3
"""Stage N1 step 3c — power of the test, and what each veto actually removed.

MDE = the smallest mean difference in R/pick the TRAIN split could have shown at
80% power, two-sided alpha 0.05: 2.80 * sd(R) / sqrt(n).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')
sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/N_databento/N1')
from trading.orb_csv import read_orb_csv  # noqa: E402
from cells import BOOKS, N1, FEATS, enrich, SPLITS  # noqa: E402

base = enrich(read_orb_csv(f'{BOOKS}/book_baseline.csv'))
rows = []
for sp, (lo, hi) in SPLITS.items():
    s = base[(base['date'] >= lo) & (base['date'] <= hi)]
    r = s['R'].dropna()
    n = len(r)
    sd = float(r.std(ddof=1))
    rows.append({'split': sp, 'picks': n, 'sd_R': sd,
                 'mde_R_per_pick': 2.80 * sd / np.sqrt(n) if n > 2 else np.nan,
                 'mde_$_book': (2.80 * sd / np.sqrt(n)) * n * float(
                     s['_sized_pnl'].sum() / s['R'].sum()) if n > 2 and s['R'].sum() else np.nan,
                 'picks_per_week': n / (len(pd.unique(s['week'])) or 1)})
mde = pd.DataFrame(rows)
print(mde.to_string(index=False, float_format=lambda x: f'{x:,.3f}'))
mde.to_csv(f'{N1}/mde.csv', index=False)

print("\n--- what each veto removed (baseline picks minus cell picks) ---")
out = []
for f, sign in FEATS:
    p = f'{BOOKS}/book_veto_{f}.csv'
    if not os.path.exists(p):
        continue
    c = enrich(read_orb_csv(p))
    key = set(zip(c['symbol'], c['date']))
    gone = base[~base.apply(lambda r: (r['symbol'], r['date']) in key, axis=1)]
    out.append({'veto': f, 'removed': len(gone),
                'removed_$': float(gone['_sized_pnl'].sum()),
                'removed_R_mean': float(gone['R'].mean()) if len(gone) else np.nan,
                'book_delta_$': float(c['_sized_pnl'].sum() - base['_sized_pnl'].sum())})
t = pd.DataFrame(out)
print(t.to_string(index=False, float_format=lambda x: f'{x:,.2f}'))
t.to_csv(f'{N1}/veto_removed.csv', index=False)

print("\n--- feature coverage on the 215 BASELINE picks ---")
ft = pd.read_parquet(f'{N1}/features.parquet')
b = base.drop(columns=[c for c in ('n_trades_range', 'n_trades_break60')
                       if c in base.columns]).merge(
    ft[['symbol', 'date', 'n_trades_range', 'n_trades_break60']],
    on=['symbol', 'date'], how='left')
cov = {'picks': len(b),
       'n_trades_range_lt20_pct': 100 * float((b['n_trades_range'] < 20).mean()),
       'ofi_range_nan_pct': 100 * float(b['ofi_range'].isna().mean()),
       'ofi_break60_nan_pct': 100 * float(b['ofi_break60'].isna().mean()),
       'spread_at_break_nan_pct': 100 * float(b['spread_at_break_bps'].isna().mean())}
print(cov)
pd.Series(cov).to_csv(f'{N1}/book_coverage.csv')
