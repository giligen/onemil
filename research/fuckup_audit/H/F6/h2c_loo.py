#!/usr/bin/env python3
"""Stage H / F6 — leave-one-out of the FROZEN stack (S4) on TRAIN, both universes.
(`h2_filters.md`'s leave-one-out table is for the S1 ordering that was current when that script ran;
this is the same computation for the stack that was actually frozen.)"""
import sys, time
import pandas as pd
sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/H/F6')
import h_core as C, h2_filters as F
from h3_val_test import FROZEN

by = {c['name']: c for c in F.CUTS}
L = ['# Stage H / F6 — leave-one-out of the frozen stack (TRAIN)', '',
     f'generated {time.strftime("%Y-%m-%d %H:%M:%S")}', '',
     'Frozen stack: `' + '` + `'.join(FROZEN) + '`.', '']
rows = []
for pop in ('Q', 'P'):
    d, wk = F.prep(pop)
    x = C.scoreable(d, 'hold', floor=True)
    L += [f'## universe {pop}', '', '| ' + ' | '.join(F.HDR) + ' |',
          '|' + '|'.join(['---'] * len(F.HDR)) + '|']
    full = pd.Series(True, index=x.index)
    for k in FROZEN:
        full &= by[k]['fn'](x)
    for tag, mm in [('FROZEN (all three)', full)] + [
            (f'minus {k}', None) for k in FROZEN]:
        if mm is None:
            k = tag.split(' ', 1)[1]
            mm = pd.Series(True, index=x.index)
            for j in FROZEN:
                if j != k:
                    mm &= by[j]['fn'](x)
        r = F.row(f'{tag}  ({mm.mean():.0%})', x[mm], wk, 'TRAIN')
        if r:
            L.append(F.fmt(r[0], F.KEYS))
            rows.append(dict(pop=pop, **r[0]))
    L.append('')
pd.DataFrame(rows).to_csv(f'{C.H}/h2c_loo.csv', index=False)
open(f'{C.H}/h2c_loo.md', 'w').write('\n'.join(L))
print('\n'.join(L))
