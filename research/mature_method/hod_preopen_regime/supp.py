#!/usr/bin/env python3
"""hod_preopen_regime — POST-HOC supplementary block, explicitly NOT declared cells.

Everything here is counted in the multiplicity total and is reported as robustness around the one
survivor of §3 (A3 = SPY 09:30->09:35 up), never as a claim. TEST is not touched.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_preopen_regime')
import score as S, score2 as S2, score3 as S3                     # noqa: E402

D = f'{ROOT}/research/mature_method/hod_preopen_regime'
SPLITS = S.SPLITS
p = S2.load_pop(); S.build_impute(p)
pre = {nm: S2.sig_set(p, **kw) for nm, kw in S2.BASES.items() if nm in ('B0', 'B2', 'B3')}
per_min, per_day = S3.idx_tape()
dfx = pd.read_csv(f'{D}/day_fields.csv', dtype={'day': str}).set_index('day')
for nm in pre:
    x = pre[nm].merge(per_min, on=['day', 'break_m'], how='left'); x.index = pre[nm].index
    for c in dfx.columns:
        if c not in x.columns:
            x[c] = x.day.map(dfx[c])
    for c in per_day.columns:
        x[c] = x.day.map(per_day[c])
    pre[nm] = x
B0, B2, B3 = pre['B0'], pre['B2'], pre['B3']

print('== POST-HOC: variants around A3 (SPY 09:30->09:35 up) ==')
print(S3.HDR + '\n' + S3.SEP)
books = {}


def go(nm, b):
    books[nm] = b
    print(S3.line(nm, b), flush=True)


for bn, base in (('B0', B0), ('B2', B2), ('B3', B3)):
    m = (base.spy_r5_pct > 0).fillna(False)
    go(f'A3 x {bn}', S.apply_book(base[m], 12, 4))
m2 = (B2.spy_r5_pct > 0).fillna(False)
go('A3 x rv>=5 [B2]', S.apply_book(B2[m2 & (B2.rv_profile >= 5)], 12, 4))
go('A3 x qqq_r5>0 [B2]', S.apply_book(B2[m2 & (B2.qqq_r5_pct > 0).fillna(False)], 12, 4))
go('A3 x imb not sell [B2]', S.apply_book(B2[m2 & (B2.qqq_imb_side != 'A')], 12, 4))
go('A3 x entry_m>=600 [B2]', S.apply_book(B2[m2 & (B2.entry_m >= 600)], 12, 4))
go('A3 x entry_m<600 [B2]', S.apply_book(B2[m2 & (B2.entry_m < 600)], 12, 4))

print('\n== TAIL DEPENDENCE (net R: as-is -> ex top 1% -> ex top 5%) ==')
for nm in ('A3 x B2', 'A3 x B0'):
    b = books[nm]
    for sp in SPLITS:
        d = b[b.split == sp]
        a = d.net.mean()
        e1 = d.net[d.net <= d.net.quantile(0.99)].mean()
        e5 = d.net[d.net <= d.net.quantile(0.95)].mean()
        print(f'  {nm:10s} {sp:5s} n={len(d):5d}  {a:+.3f} -> {e1:+.3f} -> {e5:+.3f}')

print('\n== WEEK-BY-WEEK, A3 on B2 at $100 risk (the owner metric, dollars beside the ratio) ==')
b = books['A3 x B2']
for sp in SPLITS:
    d = b[b.split == sp]
    wk = S.ALL_WEEKS[sp]
    w = d.groupby('wk').pnl.sum().reindex(wk).fillna(0.0)
    nt = d.groupby('wk').size().reindex(wk).fillna(0).astype(int)
    print(f'-- {sp}: {int((w>0).sum())}/{len(wk)} green, total ${w.sum():+,.0f}, '
          f'worst ${w.min():+,.0f}, best ${w.max():+,.0f}, mean ${w.mean():+,.0f}/wk')
    print('   ' + ' '.join(f'{v:+.0f}({n})' for v, n in zip(w.values[-26:], nt.values[-26:])))

print('\n== MONTHLY, A3 on B2 ==')
mo = b.groupby(b.day.str[:7]).pnl.agg(['sum', 'size'])
print('  ' + ' '.join(f'{k}:{v:+.0f}({int(n)})' for k, (v, n) in mo.iterrows()))

print('\n== the separation of A3 by YEAR-HALF (does the sign hold in both halves of TRAIN?) ==')
for lab, lo, hi in (('H1-25', '2025-01-01', '2025-07-01'), ('H2-25', '2025-07-01', '2026-01-01'),
                    ('VAL', '2026-01-01', '2026-06-01')):
    d = B2[(B2.day >= lo) & (B2.day < hi)]
    k = d[(d.spy_r5_pct > 0).fillna(False)]; r = d[~(d.spy_r5_pct > 0).fillna(False)]
    dm = k.rr.mean() - r.rr.mean()
    vc = S3._cluster_var(k.net.values, k.day.values) + S3._cluster_var(r.net.values, r.day.values)
    print(f'  {lab}: gross sep {dm:+.3f} R  (n_kept {len(k)} on {k.day.nunique()} days, '
          f'n_rej {len(r)} on {r.day.nunique()} days, clustered t '
          f'{(k.net.mean()-r.net.mean())/np.sqrt(vc):+.2f})')

print('\n== NULL on the post-hoc survivors ==')
for nm in ('A3 x B2', 'A3 x B0', 'A3 x B3', 'A3 x rv>=5 [B2]'):
    for sp in SPLITS:
        o, mu, p5, p95 = S.null_band(books[nm], sp)
        out = 'ABOVE' if o > p95 else ('below' if o < p5 else 'inside')
        print(f'  {nm:18s} {sp:5s} obs {o:5.1f} vs null {mu:5.1f} [{p5:.1f}, {p95:.1f}] -> {out}')
