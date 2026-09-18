#!/usr/bin/env python3
"""S1-PASSIVE step 5 — decomposition: how much of the loss is the PASSIVE FILL and how
much is the MEASURED COST? Plus the spread descriptives, the breakeven spread, the
median-spread sensitivity, and the monthly table of the best cell.

Descriptives only. No new cell, no new threshold — the 6 cells are frozen in PREREG.md.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
P = f'{ROOT}/research/fuckup_audit/O_halt/PASSIVE'
R_PCT = 2.0
BAND_COST_R = 0.40 / 2.0 * 0.5 * (0.25 + 0.875)   # O_halt's ~0.40% band spread, its weights


def split_of(d):
    return np.where(d <= '2025-12-31', 'TRAIN', np.where(d <= '2026-05-31', 'VAL', 'TEST'))


def ms(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return 'n<3'
    se = x.std(ddof=1) / np.sqrt(n)
    return f'{x.mean():+.3f} (t {x.mean()/se:5.2f}, n {n})'


def main():
    f = pd.read_parquet(f'{P}/scored_trades.parquet')
    f['split'] = split_of(f['day'].values)

    print('--- measured NBBO spread, % of price (the cost measurement) ---')
    f['entry_sp_pct'] = f['mean_spread'] / f['fill_px'] * 100
    f['cov_sp_pct'] = f['cov_spread'] / f['cov_h5'] * 100
    g = f[(f.b == 0.0) & (f.arm == 'touch')]
    print(g.groupby('split')[['entry_sp_pct', 'cov_sp_pct']].agg(['mean', 'median']).round(3).to_string())
    print('\ncost_curve.md band constant charged by O_halt: ~0.40% of price')
    print('cover-spread source:', f['cov_spread_src'].value_counts(normalize=True).round(3).to_dict())

    print('\n--- decomposition on the b=0 / touch cell (the best one) ---')
    for split in ('TRAIN', 'VAL', 'TEST'):
        s = g[g.split == split]
        gross_pass = (s.fill_px - s.cov_h5) / s.fill_px * 100 / R_PCT          # passive sell, next-open cover
        gross_pass_cl = (s.fill_px - s.close_h5) / s.fill_px * 100 / R_PCT     # passive sell, close cover
        gross_mkt_cl = (s.reopen - s.close_h5) / s.reopen * 100 / R_PCT        # O_halt: sell at reopen, close cover
        half_x = s.half_exit
        print(f'\n{split}  n {len(s)}')
        print(f'  A O_halt fill  , O_halt cover, band cost    : {ms(gross_mkt_cl - BAND_COST_R)}')
        print(f'  B O_halt fill  , O_halt cover, MEASURED cost: {ms(gross_mkt_cl - half_x)}')
        print(f'  C passive fill , O_halt cover, band cost    : {ms(gross_pass_cl - BAND_COST_R)}')
        print(f'  D passive fill , O_halt cover, MEASURED cost: {ms(gross_pass_cl - half_x)}')
        print(f'  E passive fill , next-open cov, MEASURED cost (THE CELL): {ms(gross_pass - half_x)}')
        print(f'    gross only: O_halt-fill {gross_mkt_cl.mean():+.3f}  passive {gross_pass_cl.mean():+.3f}'
              f'  passive+next-open {gross_pass.mean():+.3f}   cost {half_x.mean():.3f} R')

    print('\n--- breakeven spread (the cell gross vs what a half-spread costs) ---')
    for split in ('TRAIN', 'VAL', 'TEST'):
        s = g[g.split == split]
        gr = ((s.fill_px - s.cov_h5) / s.fill_px * 100 / R_PCT).mean()
        print(f'{split}: gross {gr:+.3f} R  -> breakeven cover spread '
              f'{gr * R_PCT * 2 / 1.0:.2f}% of price   (measured mean {s.cov_sp_pct.mean():.2f}%, '
              f'median {s.cov_sp_pct.median():.2f}%)')

    print('\n--- sensitivity: the SAME cell charged the MEDIAN cover spread instead of the mean ---')
    for split in ('TRAIN', 'VAL', 'TEST'):
        s = g[g.split == split]
        med_half = 0.5 * s.cov_sp_pct.median() / R_PCT
        gr = (s.fill_px - s.cov_h5) / s.fill_px * 100 / R_PCT
        print(f'{split}: {ms(gr - med_half)}   (median-spread half = {med_half:.3f} R)')

    print('\n--- b=0/touch monthly net R ---')
    g2 = g.copy()
    g2['month'] = g2['day'].str.slice(0, 7)
    m = g2.groupby('month')['net_R'].agg(['size', 'sum']).round(2)
    print(f'months {len(m)}, green {int((m["sum"] > 0).sum())}')
    print(m.T.to_string())

    print('\n--- borrow (TODAY flags) ---')
    print(g.groupby('split')['tradeable'].agg(['size', 'sum', 'mean']).round(4).to_string())

    print('\n--- capacity / economics on the b=0 touch cell ---')
    for split in ('TRAIN', 'VAL', 'TEST'):
        s = g[g.split == split]
        wk = pd.to_datetime(s['day']).dt.to_period('W').nunique()
        rwk = s['net_R'].sum() / wk
        print(f'{split}: {len(s)/wk:.1f} trades/wk, {rwk:+.2f} R/wk -> '
              f'${rwk*100*4.33:+,.0f}/mo at $100 risk, ${rwk*375*4.33:+,.0f}/mo at $375')


if __name__ == '__main__':
    main()
