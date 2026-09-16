#!/usr/bin/env python3
"""A2 — time-of-day bands (H5). Base families + the F5 reference row, contract (c), book re-run inside each window.

Windows: ALL (entry_m <= 841, the score4 population), >=10:00 (600), >=10:30 (630), and 09:30-10:00 only (< 600).
TRAIN and VAL only — TEST is not read in A2.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, f'{ROOT}/research/fuckup_audit/A')
import acore
A = 'research/fuckup_audit/A'
KEYS = ['F8 {"N": 5}', 'F8 {"N": 15}', 'F8 {"N": 30}', 'F6 {}', 'F1 {"P": 0.12}', 'F5 {"K": 5, "X": 0.04}']
WINDOWS = [('ALL', 0, 842), ('>=10:00', 600, 842), ('>=10:30', 630, 842), ('09:30-10:00', 0, 600)]
c = acore.load(keys=KEYS)
WK = acore.week_index(c)
out = []
for key, dk in c.groupby('key'):
    for tag in ('hold', '2r'):
        for wlab, lo, hi in WINDOWS:
            for sp in ('TRAIN', 'VAL'):
                x = dk[(dk.split == sp) & (dk.entry_m >= lo) & (dk.entry_m < hi)]
                t = acore.book_rows(x, tag)
                if t is None:
                    out.append(dict(key=key, exit=tag, window=wlab, split=sp, corr_n=len(x), note='under 40 candidates'))
                    continue
                row = dict(key=key, exit=tag, window=wlab, split=sp)
                for cc, lab in (('a', 'gross'), ('c', 'corr')):
                    for k, v in acore.stats(t, cc, WK[sp]).items():
                        row[f'{lab}_{k}'] = v
                mix = t.why.value_counts(normalize=True)
                for w in ('stop', 'target', 'eod'):
                    row[f'mix_{w}'] = round(float(mix.get(w, 0.0)), 3)
                g = acore.book_rows(x[x.sp_over_r_c <= acore.GATE_SP_OVER_R], tag)
                if g is not None:
                    for k, v in acore.stats(g, 'c', WK[sp]).items():
                        row[f'gate_{k}'] = v
                out.append(row)
T = pd.DataFrame(out)
T.to_csv(f'{A}/a2_cells.csv', index=False)
cols = ['key', 'exit', 'window', 'corr_n', 'corr_tpw', 'gross_meanR', 'corr_meanR', 'corr_se', 'corr_t', 'corr_WR',
        'corr_wkR', 'corr_green', 'corr_worst', 'gate_meanR', 'gate_tpw', 'mix_stop', 'mix_target', 'mix_eod']
pd.set_option('display.width', 400); pd.set_option('display.max_columns', 60)
L = ['# A2 — time bands, contract (c), book re-run inside each window', '',
     f'cells: {len(KEYS)} keys x 2 exits x {len(WINDOWS)} windows x 2 splits = {len(T)}', '']
for sp in ('TRAIN', 'VAL'):
    L += [f'## {sp}', '', T[T.split == sp].sort_values(['key', 'exit', 'window'])[cols].to_string(index=False), '']
open(f'{A}/a2_tables.md', 'w').write('\n'.join(L))
print('\n'.join(L)); print('DONE', flush=True)
