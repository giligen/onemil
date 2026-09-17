#!/usr/bin/env python3
"""D1 step 3 — parity of the D1 target against Stage C's scorer.

The D1 feature table recomputes the contract-(c) net R from candidates4's raw columns. If that recomputation is
right, booking the SAME population with run_book(12,4) and NO selection must reproduce, cell for cell, Stage C's
`next` rows for `hold-to-close` (target p) and `2R stop-1%` (target s) — same n, same mean net R.

Compared: C/score5_results.csv (the all-day Stage C run) TRAIN_n / TRAIN_meanR / VAL_n / VAL_meanR.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/D1')
import d1_core as K

C = 'research/fuckup_audit/C/score5_results.csv'
OUTC = {'p': 'hold-to-close', 's': '2R stop-1%'}

c = K.load()
weeks = K.week_index(c)
sc = pd.read_csv(C, keep_default_na=False, na_values=[''])
rows = []
for tgt in K.TARGETS:
    for key in K.KEYS + [K.EXTRA_KEY]:
        x = c[c.key == key]
        for sp in ('TRAIN', 'VAL'):
            t = K.book(x[x.split == sp], tgt)
            if t is None:
                continue
            st = K.stats(t, weeks[sp])
            ref = sc[(sc.key == key) & (sc.fill == 'next') & (sc.outcome == OUTC[tgt])]
            r = dict(tgt=tgt, key=key, split=sp, d1_n=st['n'], d1_meanR=st['meanR'])
            if len(ref):
                r['c_n'] = int(ref[f'{sp}_n'].iloc[0])
                r['c_meanR'] = float(ref[f'{sp}_meanR'].iloc[0])
                r['dn'] = r['d1_n'] - r['c_n']
                r['dmeanR'] = round(r['d1_meanR'] - r['c_meanR'], 6)
            rows.append(r)
T = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(T.to_string(index=False), flush=True)
print('\nmax |dn| =', int(T.dn.abs().max()), ' max |d meanR| =', float(T.dmeanR.abs().max()), flush=True)
T.to_csv('research/fuckup_audit/D1/d1_parity.csv', index=False)
