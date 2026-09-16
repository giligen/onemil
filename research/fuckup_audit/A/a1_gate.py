#!/usr/bin/env python3
"""A1 — the PLAN §1 re-gate applied to the corrected contract (c). Secondary, reported not gated: contract (d).

G1 TRAIN: mean net R > 0 AND t >= 2.0 AND >= 5 trades/week.
G2 VAL  : mean net R > 0 AND t >= 1.0 AND >= 55% weeks green (bar raised by 1 SE of weekly R per 10 cells passing G1).
G3 TEST : read ONCE, only for G2 survivors.
Also: what the OLD gate (TRAIN >= +10R/week) would have said under (c), and the closest miss per family with its MDE.
"""
import os
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
A = 'research/fuckup_audit/A'
T = pd.read_csv(f'{A}/a0_cells.csv', keep_default_na=False, na_values=[''])
tr = T[T.split == 'TRAIN'].set_index(['key', 'exit'])
va = T[T.split == 'VAL'].set_index(['key', 'exit'])
L = ['# A1 — the re-gate under the corrected contract', '']
for lab, m, tcol in (('(c) corrected', 'corr', 'corr_t'), ('(d) corrected + live liquidity gate', 'gate', 'gate_t')):
    g1 = tr[(tr[f'{m}_meanR'] > 0) & (tr[tcol] >= 2.0) & (tr[f'{m}_tpw'] >= 5)]
    old = tr[(tr[f'{m}_wkR'] >= 10.0) & (tr[f'{m}_tpw'] >= 5)]
    pos = tr[tr[f'{m}_meanR'] > 0].sort_values(tcol, ascending=False)
    L += [f'## contract {lab}', '',
          f'G1 (TRAIN mean>0, t>=2, >=5 tpw): **{len(g1)} of {len(tr)} pass**',
          f'old score4 gate (TRAIN >= +10R/week): {len(old)} of {len(tr)} pass', '',
          f'TRAIN cells with a POSITIVE mean net R ({len(pos)} of {len(tr)}), ranked by t:', '',
          pos.reset_index()[['key', 'exit', f'{m}_n', f'{m}_tpw', f'{m}_meanR', f'{m}_se', tcol, f'{m}_WR',
                             f'{m}_wkR', f'{m}_green', f'{m}_worst']].to_string(index=False), '']
    if len(g1):
        L += ['G1 survivors on VAL:', '', va.loc[g1.index].reset_index().to_string(index=False), '']
    else:
        L += ['**Nothing clears G1, so G2 is not evaluated and TEST is not read for selection.**', '']
# closest miss per family, contract (c), with the minimum detectable effect
tr2 = tr.reset_index()
tr2['fam'] = tr2.key.str.split(' ').str[0]
best = tr2.sort_values('corr_t', ascending=False).groupby('fam').head(1)
best['MDE_R'] = (2.8 * best.corr_se).round(3)
L += ['## closest miss per family on TRAIN, contract (c), with its minimum detectable effect (MDE = 2.8 x SE)', '',
      best[['fam', 'key', 'exit', 'corr_n', 'corr_tpw', 'corr_meanR', 'corr_se', 'MDE_R', 'corr_t', 'corr_WR',
            'corr_wkR', 'corr_green']].to_string(index=False), '']
open(f'{A}/a1_gate.md', 'w').write('\n'.join(L))
print('\n'.join(L))
