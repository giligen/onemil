#!/usr/bin/env python3
"""A4 support — decompose the score4 -> corrected move into its three parts, on the TRAIN booked trades.

chain:  b  = score4                                (banded spread 1.90/1.20/0.80/0.60/0.50 %, entry 1.0*half, target 0)
        b_sp = b with the MEASURED spread table only (entry still 1.0*half, target still 0)
        cp  = b_sp with the entry charge cut to 0.25*half            (= contract c')
        c   = cp with the target exit charged 0.875*half             (= contract c)
Also the mean cost in R per contract, and the max TRAIN t across the 52 cells vs the noise benchmark.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, f'{ROOT}/research/fuckup_audit/A')
import acore
A = 'research/fuckup_audit/A'
c = acore.load()
rr = c.r_pct.clip(lower=0.05)
half_b = 0.5 * c.spread_pct / rr
half_c = 0.5 * c.spread_pct_c / rr
out = []
for key, dk in c.groupby('key'):
    for tag in ('hold', '2r'):
        x = dk[dk.split == 'TRAIN']
        if len(x) < 40: continue
        rb = x[f'why_{tag}'].map(acore.RATIO_B).fillna(0.875)
        rc = x[f'why_{tag}'].map(acore.RATIO_C).fillna(0.875)
        hb, hc = half_b.loc[x.index], half_c.loc[x.index]
        xx = x.copy()
        xx['b_sp'] = x[f'rr_{tag}'] - hc - hc * rb
        rows = [(r.day, int(r.entry_m), int(getattr(r, f'exit_m_{tag}')), r.symbol,
                 getattr(r, f'a_{tag}'), getattr(r, f'b_{tag}'), r.b_sp, getattr(r, f'cp_{tag}'), getattr(r, f'c_{tag}'))
                for r in xx.itertuples()]
        t = pd.DataFrame(acore.run_book(rows, 12, 4), columns=['day', 'em', 'xm', 'sym', 'a', 'b', 'b_sp', 'cp', 'c'])
        out.append(dict(key=key, exit=tag, n=len(t), gross=round(t.a.mean(), 3), s4=round(t.b.mean(), 3),
                        step1_spread_table=round(t.b_sp.mean() - t.b.mean(), 3),
                        step2_entry_025=round(t.cp.mean() - t.b_sp.mean(), 3),
                        step3_target_charge=round(t.c.mean() - t.cp.mean(), 3),
                        corrected=round(t.c.mean(), 3), total_delta=round(t.c.mean() - t.b.mean(), 3)))
D = pd.DataFrame(out).sort_values('corrected', ascending=False)
D.to_csv(f'{A}/a4_decomp.csv', index=False)
pd.set_option('display.width', 300)
T = pd.read_csv(f'{A}/a0_cells.csv', keep_default_na=False, na_values=[''])
tr = T[T.split == 'TRAIN']
L = ['# A4 support — where the corrected contract gets its money, TRAIN booked trades', '',
     D.to_string(index=False), '',
     'means over the 52 cells: ' + str(D[['step1_spread_table', 'step2_entry_025', 'step3_target_charge',
                                          'total_delta']].mean().round(3).to_dict()), '',
     f'max TRAIN t over the 52 cells under (c): {tr.corr_t.max():.2f}  (E[max] of 52 independent N(0,1) draws ~ 2.7)',
     f'median |t|: {tr.corr_t.abs().median():.2f} | cells with mean net R > 0: {(tr.corr_meanR > 0).sum()} of {len(tr)}', '']
open(f'{A}/a4_decomp.md', 'w').write('\n'.join(L))
print('\n'.join(L)); print('DONE', flush=True)
