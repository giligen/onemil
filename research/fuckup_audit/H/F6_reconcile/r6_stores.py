#!/usr/bin/env python3
"""R6 - are the two bar stores the same tape? For every TEST F6 candidate key (A's or B's), read the RTH bars
from data/cache.db AND research/bf_zero/bars_sip.db and compare minute by minute."""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, f'{ROOT}/research/fuckup_audit/H/F6_reconcile')
from pipeline import Bars

OUT = 'research/fuckup_audit/H/F6_reconcile'
RD = lambda p, **k: pd.read_csv(p, keep_default_na=False, na_values=[''], dtype={'symbol': str, 'day': str}, **k)
log = lambda *a: (print(*a), sys.stdout.flush())

cj = RD(f'{OUT}/cand_join.csv'); cj = cj[cj.split == 'TEST']
bars = Bars()
rec = []
for i, r in enumerate(cj.itertuples(index=False)):
    if i % 400 == 0:
        log('  %d/%d' % (i, len(cj)))
    a = bars.raw(r.symbol, r.day, 'cache')
    b = bars.raw(r.symbol, r.day, 'sip')
    d = dict(day=r.day, symbol=r.symbol, in_A=r.in_A, in_B=r.in_B,
             n_cache=0 if a is None else len(a[0]), n_sip=0 if b is None else len(b[0]))
    if a is not None and b is not None:
        ma, mb = set(a[0].tolist()), set(b[0].tolist())
        both = sorted(ma & mb)
        d['minutes_only_cache'] = len(ma - mb); d['minutes_only_sip'] = len(mb - ma); d['minutes_both'] = len(both)
        ia = {m: k for k, m in enumerate(a[0])}; ib = {m: k for k, m in enumerate(b[0])}
        diffs = 0; maxrel = 0.0
        for m in both:
            for f in range(1, 5):                     # o,h,l,c
                va, vb = a[f][ia[m]], b[f][ib[m]]
                if abs(va - vb) > 1e-9:
                    diffs += 1
                    maxrel = max(maxrel, abs(va - vb) / max(abs(vb), 1e-9))
        d['ohlc_cells_diff'] = diffs; d['max_rel_diff'] = maxrel
        d['first_m_cache'] = int(a[0][0]); d['first_m_sip'] = int(b[0][0])
    rec.append(d)
f = pd.DataFrame(rec).fillna(-1)
f.to_csv(f'{OUT}/store_compare_test.csv', index=False)
both = f[(f.n_cache > 0) & (f.n_sip > 0)]
L = ['# R6 bar-store comparison, TEST F6 candidate keys', '',
     f'keys examined: {len(f)}',
     f'  in cache.db only: {int(((f.n_cache>0)&(f.n_sip==0)).sum())}',
     f'  in bars_sip.db only: {int(((f.n_cache==0)&(f.n_sip>0)).sum())}',
     f'  in both: {len(both)}',
     f'  in neither: {int(((f.n_cache==0)&(f.n_sip==0)).sum())}', '']
if len(both):
    L += [f'of the {len(both)} keys present in BOTH stores:',
          f'  identical minute sets: {int(((both.minutes_only_cache==0)&(both.minutes_only_sip==0)).sum())}',
          f'  identical OHLC on every shared minute: {int((both.ohlc_cells_diff==0).sum())}',
          f'  any OHLC cell differing: {int((both.ohlc_cells_diff>0).sum())}',
          f'  max relative OHLC difference seen: {both.max_rel_diff.max():.3g}',
          f'  mean extra minutes in cache / in sip: {both.minutes_only_cache.mean():.2f} / {both.minutes_only_sip.mean():.2f}', '']
open(f'{OUT}/r6_stores.md', 'w').write('\n'.join(L) + '\n')
log('\n'.join(L))
