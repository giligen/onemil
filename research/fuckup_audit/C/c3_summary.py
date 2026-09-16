#!/usr/bin/env python3
"""Stage C — derived tables for C/REPORT.md, built ONLY from the score5c result CSVs (no re-scoring, no new cell).
Usage: ulimit -v 1800000; nice -n 10 python3 research/fuckup_audit/C/c3_summary.py
"""
import os, glob
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
D = 'research/fuckup_audit/C'
RUNS = [('all-day  contract (c)      [companion]', f'{D}/score5_results.csv'),
        ('>=10:00  contract (c)      [PRIMARY]', f'{D}/score5_results.m600.csv'),
        ('>=10:00  queue-OK resting  [sens Q]', f'{D}/score5_results.queueok.m600.csv'),
        ('all-day  queue-OK resting  [sens Q]', f'{D}/score5_results.queueok.csv'),
        ('>=10:00  free target (c\')  [sens T0]', f'{D}/score5_results.freetgt.m600.csv'),
        ('>=10:00  legacy spread     [sens L]', f'{D}/score5_results.legacy.m600.csv'),
        ('all-day  legacy spread     [sens L]', f'{D}/score5_results.legacy.csv')]
G1_T, G2_T, G2_GREEN, MIN_TPW = 2.0, 1.0, 0.55, 5.0
L = ['# Stage C — derived tables (built from the score5c result CSVs only)', '']

rows = []
for name, p in RUNS:
    if not os.path.exists(p):
        rows.append(dict(run=name, note='MISSING')); continue
    T = pd.read_csv(p)
    g1 = T[(T.TRAIN_meanR > 0) & (T.TRAIN_t >= G1_T) & (T.TRAIN_tpw >= MIN_TPW)]
    bar = len(g1) // 10
    g2 = g1[(g1.VAL_meanR > 0) & (g1.VAL_t >= G2_T) & (g1.VAL_green >= G2_GREEN)
            & (g1.VAL_wkR >= bar * g1.VAL_wkSE.fillna(0))]
    rows.append(dict(run=name, cells_with_book=len(T), TRAIN_pos=int((T.TRAIN_meanR > 0).sum()),
                     TRAIN_gross_pos=int((T.TRAIN_gross > 0).sum()),
                     best_meanR=round(T.TRAIN_meanR.max(), 4), best_t=round(T.TRAIN_t.max(), 2),
                     G1=len(g1), G2=len(g2),
                     VAL_pos=int((T.VAL_meanR > 0).sum()),
                     both_pos=int(((T.TRAIN_meanR > 0) & (T.VAL_meanR > 0)).sum())))
L += ['## 1. Every run, the gate counts', pd.DataFrame(rows).to_string(index=False), '']

P = pd.read_csv(f'{D}/score5_results.m600.csv')
A = pd.read_csv(f'{D}/score5_results.csv')
P['fam'] = P.key.str.split(' ').str[0]

L += ['## 2. PRIMARY (>=10:00): closest miss per family-config — the best cell by TRAIN t among the cells with a '
      'positive TRAIN mean, else the best mean; `mde` = 2.8 x SE is the smallest per-trade effect that cell could '
      'have seen at 80% power', '']
best = []
for key, g in P.groupby('key'):
    gg = g[g.TRAIN_meanR > 0]
    r = (gg.sort_values('TRAIN_t', ascending=False) if len(gg) else g.sort_values('TRAIN_meanR', ascending=False)).iloc[0]
    best.append(r)
B = pd.DataFrame(best)[['key', 'fill', 'outcome', 'TRAIN_n', 'TRAIN_tpw', 'TRAIN_meanR', 'TRAIN_gross', 'TRAIN_se',
                        'TRAIN_mde', 'TRAIN_t', 'TRAIN_WR', 'TRAIN_green', 'VAL_meanR', 'VAL_gross', 'VAL_t',
                        'VAL_green', 'ex5', 'cap3']].sort_values('TRAIN_t', ascending=False)
L += [B.to_string(index=False), '']

L += ['## 3. The H9 families F11-F14 — their first honest numbers (PRIMARY window, every cell with a book)', '']
H9 = P[P.fam.isin(['F11', 'F12', 'F13', 'F14'])][
    ['key', 'fill', 'outcome', 'TRAIN_n', 'TRAIN_tpw', 'TRAIN_meanR', 'TRAIN_gross', 'TRAIN_se', 'TRAIN_mde',
     'TRAIN_t', 'TRAIN_WR', 'TRAIN_stopP', 'VAL_n', 'VAL_meanR', 'VAL_gross', 'VAL_t', 'ex5']]
L += [H9.sort_values('TRAIN_meanR', ascending=False).to_string(index=False), '']
declared = ['F11 {"N": 15, "base": "F8"}', 'F11 {"base": "F6"}', 'F12 {"N": 15, "base": "F8"}', 'F12 {"base": "F6"}',
            'F13 {"K": 5, "X": 0.04}', 'F14 {"N": 15}']
L += ['Declared H9 keys with NO scoreable book in the primary window: '
      + str([k for k in declared if k not in set(P.key)]), '']

L += ['## 4. Fill comparison at BOOK level (PRIMARY window): the same family x outcome, next-open vs resting', '']
m = P.merge(P, on=['key', 'outcome'], suffixes=('_n', '_r'))
m = m[(m.fill_n == 'next') & (m.fill_r == 'rest')]
L += [m[['key', 'outcome', 'TRAIN_n_n', 'TRAIN_meanR_n', 'TRAIN_gross_n', 'TRAIN_n_r', 'TRAIN_meanR_r',
         'TRAIN_gross_r', 'VAL_meanR_n', 'VAL_meanR_r']]
      .rename(columns={'TRAIN_n_n': 'n_next', 'TRAIN_n_r': 'n_rest'}).to_string(index=False), '',
      f'book-level TRAIN mean over the {len(m)} matched pairs: next {m.TRAIN_meanR_n.mean():.4f} '
      f'vs rest {m.TRAIN_meanR_r.mean():.4f}  |  VAL next {m.VAL_meanR_n.mean():.4f} vs rest {m.VAL_meanR_r.mean():.4f}',
      f'gross: next {m.TRAIN_gross_n.mean():.4f} vs rest {m.TRAIN_gross_r.mean():.4f}', '']

L += ['## 5. Outcome (stop x exit) comparison at BOOK level, PRIMARY window, averaged over the family-configs', '']
o = P.groupby(['fill', 'outcome']).agg(cells=('key', 'size'), TRAIN_meanR=('TRAIN_meanR', 'mean'),
                                       TRAIN_gross=('TRAIN_gross', 'mean'), TRAIN_stopP=('TRAIN_stopP', 'mean'),
                                       VAL_meanR=('VAL_meanR', 'mean'), TRAIN_tpw=('TRAIN_tpw', 'mean')).round(4)
L += [o.to_string(), '']

L += ['## 6. all-day vs >=10:00, the same 107 cells', '']
j = A.merge(P, on=['key', 'fill', 'outcome'], suffixes=('_all', '_600'))
L += [f'cells matched: {len(j)}',
      f'TRAIN mean net R: all-day {j.TRAIN_meanR_all.mean():.4f} -> >=10:00 {j.TRAIN_meanR_600.mean():.4f}',
      f'TRAIN mean GROSS: all-day {j.TRAIN_gross_all.mean():.4f} -> >=10:00 {j.TRAIN_gross_600.mean():.4f}',
      f'trades/week:      all-day {j.TRAIN_tpw_all.mean():.1f} -> >=10:00 {j.TRAIN_tpw_600.mean():.1f}',
      f'cells improved by the 10:00 cut: {int((j.TRAIN_meanR_600 > j.TRAIN_meanR_all).sum())} of {len(j)}', '']
open(f'{D}/c3_summary.md', 'w').write('\n'.join(L))
print('\n'.join(L[:6]))
print('wrote', f'{D}/c3_summary.md')
