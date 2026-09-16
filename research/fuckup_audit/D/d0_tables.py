#!/usr/bin/env python3
"""D0 step 7 — render every result table into research/fuckup_audit/D/results.md (the block that is
concatenated into REPORT.md after the pre-registration)."""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
D = 'research/fuckup_audit/D'
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 500); pd.set_option('display.max_columns', 60)
L = []
A = L.append

COLS = ['key', 'model', 'sel', 'n', 'tpw', 'meanR', 'se', 't', 'mde', 'WR', 'wkR', 'green', 'worst']
PER = {'trainpred': 'TRAINPRED (predicted TRAIN months 2025-10..12, 14 weeks)',
       'val': 'VAL (2026-01..05, 22 weeks)', 'test': 'TEST (2026-06..2026-09-11, 14 weeks) — READ ONCE'}

cells = {p: pd.read_csv(f'{D}/cells_{p}.csv', keep_default_na=False, na_values=['']) for p in PER}

A('## 1. The 18 declared cells + the two controls, real tape')
A('')
A('`meanR` = mean net R per booked trade under Stage A contract (c); `mde` = 2.8*SE, the smallest per-trade effect the')
A('cell could have seen at 80% power; `wkR` = mean weekly R at 4 slots; `green` = share of weeks positive;')
A('`worst` = worst week in R. RAND12 `se` is the sd across its 20 seeds, not a t-test SE.')
A('')
for p, lab in PER.items():
    c = cells[p]
    r = c[c.tape == 'real']
    A(f'### {lab}')
    A('')
    A('```')
    A(r[COLS].to_string(index=False))
    A('```')
    A('')

A('## 2. Selection minus the unselected book (FCFS), per period')
A('')
rows = []
for p in PER:
    c = cells[p]
    r = c[c.tape == 'real']
    f = r[r.model == 'FCFS'].set_index('key').meanR
    for _, x in r[~r.model.isin(['FCFS', 'RAND12'])].iterrows():
        rows.append(dict(period=p, key=x.key, model=x.model, sel=x.sel, meanR=x.meanR, t=x.t,
                         vs_fcfs=round(x.meanR - f[x.key], 4)))
S = pd.DataFrame(rows)
piv = S.pivot_table(index=['key', 'model', 'sel'], columns='period', values=['meanR', 't', 'vs_fcfs'])
piv.columns = [f'{a}_{b}' for a, b in piv.columns]
order = ['meanR_trainpred', 'meanR_val', 'meanR_test', 't_trainpred', 't_val', 't_test',
         'vs_fcfs_trainpred', 'vs_fcfs_val', 'vs_fcfs_test']
A('```')
A(piv[order].round(4).to_string())
A('```')
A('')
A('mean of (selected - FCFS) over the 18 cells: ' +
  ', '.join(f'{k} {v:+.4f}' for k, v in S.groupby('period').vs_fcfs.mean().round(4).items()))
A('')
A('share of the 18 cells with mean net R > 0: ' +
  ', '.join(f'{k} {v:.2f}' for k, v in S.groupby('period').apply(lambda g: (g.meanR > 0).mean(), include_groups=False).items()))
A('')

A('## 3. The Nagel reversed-tape gate and the shuffled-target null')
A('')
A('Same pipeline, sign of every target flipped (`rev`) / targets permuted within day and the selected rows scored on')
A('the TRUE outcome (`shuf`). A cell FAILS the gate if its reversed twin is profitable on VAL (mean > 0 AND t > 1).')
A('')
for p, lab in PER.items():
    c = cells[p]
    r = c[c.tape != 'real'][['key', 'model', 'sel', 'tape', 'n', 'meanR', 't', 'wkR', 'green']]
    A(f'### {lab}')
    A('```')
    A(r.to_string(index=False))
    A('```')
    A('')

A('## 4. Decile calibration of the real-tape prediction (whole candidate pool, not the book)')
A('')
for p in ('trainpred', 'val', 'test'):
    cal = pd.read_csv(f'{D}/calib_{p}.csv', keep_default_na=False, na_values=[''])
    t = cal.pivot_table(index=['key', 'model'], columns='decile', values='meanR')
    rho = cal.groupby(['key', 'model']).rho.first()
    A(f'### {PER[p]}')
    A('```')
    A(t.round(3).to_string())
    A('')
    A('Spearman rho(decile, realised mean net R):')
    A(rho.round(3).to_string())
    A('```')
    A('')

A('## 5. Tail dependence of the booked cells (real tape)')
A('')
A('`cut1` / `cut5` = mean net R with the top 1% / top 5% of the cell\'s booked trades removed; `cap3` = winners capped at +3R.')
A('')
for p, lab in PER.items():
    c = cells[p]
    r = c[(c.tape == 'real') & c.cut5_meanR.notna()]
    A(f'### {lab}')
    A('```')
    A(r[['key', 'model', 'sel', 'n', 'meanR', 'cut1_meanR', 'cut5_meanR', 'cap3_meanR', 'wkR', 'cut5_wkR', 'cap3_wkR']]
      .round(4).to_string(index=False))
    A('```')
    A('')

A('## 6. Permutation importance on VAL and its stability across the monthly refits')
A('')
I = pd.read_csv(f'{D}/importance.csv', keep_default_na=False, na_values=[''])
I['period'] = np.where(I.month < '2026-01', 'TRAINPRED', np.where(I.month < '2026-06', 'VAL', 'TEST'))
I['rank'] = I.groupby(['key', 'month']).imp.rank(ascending=False)
for key in I.key.unique():
    v = I[(I.key == key) & (I.period == 'VAL')].groupby('feat').imp.agg(['mean', 'std'])
    r = I[(I.key == key) & (I.period != 'TEST')].groupby('feat')['rank'].agg(['mean', 'std'])
    t = v.join(r, lsuffix='_imp', rsuffix='_rank').sort_values('mean_imp', ascending=False).head(12)
    A(f'### {key} — top 12 by mean VAL permutation importance (of 33 features)')
    A('```')
    A(t.round(4).to_string())
    A('```')
    A('')

A('## 7. The transparent baseline that was actually chosen')
A('')
B = pd.read_csv(f'{D}/baseline_univariate.csv', keep_default_na=False, na_values=[''])
A('```')
A(B[(B.tape == 'real') & (B.chosen)][['key', 'feat', 'rho', 'spread', 'n_dec']].to_string(index=False))
A('')
A('the same fit on the reversed tape (it must, and does, flip every direction):')
A(B[(B.tape == 'rev') & (B.chosen)][['key', 'feat', 'rho', 'spread']].to_string(index=False))
A('```')
A('')

A('## 8. Per-month net R of the five cells that cleared G2 on VAL')
A('')
bk = pd.concat([pd.read_csv(f'{D}/booked_{p}.csv', keep_default_na=False, na_values=[''])
                for p in ('trainpred', 'val', 'test')], ignore_index=True)
surv = [('F6 {}', 'reg', 'S2'), ('F6 {}', 'clf', 'S2'), ('F8 {"N": 15}', 'reg', 'S1'),
        ('F8 {"N": 15}', 'reg', 'S2'), ('F8 {"N": 15}', 'clf', 'S1')]
A('```')
for k, m, s in surv:
    x = bk[(bk.key == k) & (bk.model == m) & (bk.sel == s)]
    g = x.groupby('month').net.agg(n='size', sumR='sum', meanR='mean').round(3)
    A(f'{k}  {m}  {s}')
    A(g.to_string())
    A('')
A('```')
A('')

A('## 9. Search-adjusted permutation p on VAL')
A('')
ps = pd.read_csv(f'{D}/perm_val_summary.csv', keep_default_na=False, na_values=[''])
A('```')
A(ps.to_string(index=False))
A('```')
A('')
open(f'{D}/results.md', 'w').write('\n'.join(L))
print('wrote results.md', len(L), 'lines')
