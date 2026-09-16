#!/usr/bin/env python3
"""D0 step 3 — parity anchor. The unselected book of this table must reproduce Stage A's A2 rows
(`>=10:00` window, contract (c), exit `hold`) cell for cell. This catches coding errors in the
extraction, the cost contract and the book wiring; it cannot catch specification errors.
"""
import os, sys
import pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/D')
import d0_core as K

c = K.load()
WK = K.week_index(c)
print('weeks', {s: len(v) for s, v in WK.items()}, flush=True)
ref = pd.read_csv(f'{ROOT}/research/fuckup_audit/A/a2_cells.csv', keep_default_na=False, na_values=[''])
ref = ref[(ref.window == '>=10:00') & (ref.exit == 'hold') & ref.key.isin(K.KEYS)]
rows = []
for key in K.KEYS:
    for sp in ('TRAIN', 'VAL'):
        t = K.book(c[(c.key == key) & (c.split == sp)])
        s = K.stats(t, WK[sp])
        r = ref[(ref.key == key) & (ref.split == sp)]
        rows.append(dict(key=key, split=sp, n=s['n'], meanR=s['meanR'], tpw=s['tpw'],
                         ref_n=int(r.corr_n.iloc[0]), ref_meanR=float(r.corr_meanR.iloc[0]),
                         ref_tpw=float(r.corr_tpw.iloc[0])))
P = pd.DataFrame(rows)
P['d_n'] = P.n - P.ref_n
P['d_meanR'] = (P.meanR - P.ref_meanR).round(5)
P.to_csv(f'{ROOT}/research/fuckup_audit/D/d0_parity.csv', index=False)
print(P.to_string(index=False), flush=True)
print('max |d_n|', P.d_n.abs().max(), 'max |d_meanR|', P.d_meanR.abs().max(), flush=True)
