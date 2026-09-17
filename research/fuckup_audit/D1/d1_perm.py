#!/usr/bin/env python3
"""D1 step 6 — the search-adjusted permutation p on VAL over ALL 48 declared model cells.

Conditional permutation: the fitted models, the selections and the book are held FIXED; the realised net R is
permuted WITHIN each (target, family, day). The statistic is the maximum weekly R over the 48 cells; p = the share
of draws whose max >= the observed max. run_book is driven by entry/exit minutes, never by net, so the booked SET is
invariant under the permutation and can be frozen once.

Conditional on the fits, therefore an UNDERSTATED p (D0's caveat carries: the one full-pipeline shuffled draw
reached 65% of the observed statistic there).
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/D1')
import d1_core as K
import d1_eval as E

D1 = 'research/fuckup_audit/D1'
NDRAW = 500
P = pd.read_csv(f'{D1}/preds.csv', dtype=E.DT, keep_default_na=False, na_values=[''])
p = P[P.split == 'VAL'].reset_index(drop=True)
weeks = sorted(p.wk.unique())

cells = [(tgt, key, mod, s) for tgt in K.TARGETS for key in K.KEYS for mod, _ in E.MODELS for s in ('S1', 'S2')]
sel_idx = {}
for tgt, key, mod, s in cells:
    x = p[(p.tgt == tgt) & (p.key == key)]
    if not len(x):
        continue
    y = E.sel_s1(x, f'{mod}_real') if s == 'S1' else E.sel_s2(x, mod, 'real')
    t = E.book(y, 'real')
    if t is None:
        continue
    z = t[['day', 'symbol', 'wk']].copy(); z['tgt'] = tgt; z['key'] = key
    sel_idx[(tgt, key, mod, s)] = z

base = p.set_index(['tgt', 'key', 'day', 'symbol'])
obs = {}
for k, t in sel_idx.items():
    m = base.net.reindex(pd.MultiIndex.from_arrays([t.tgt, t.key, t.day, t.symbol])).values
    obs[k] = float(pd.Series(m).groupby(t.wk.values).sum().reindex(weeks).fillna(0).mean())
obs_max = max(obs.values())
print('cells with a book:', len(sel_idx), flush=True)
print('observed weekly R per cell:', {f'{a}|{b.split()[0]}{b.split()[-1]}|{c}|{d}': round(v, 2)
                                      for (a, b, c, d), v in obs.items()}, flush=True)
print('observed MAX weekly R:', round(obs_max, 3), flush=True)

rs = np.random.RandomState(20260916)
grp = p.groupby(['tgt', 'key', 'day']).indices
nulls = []
for d in range(NDRAW):
    sh = p.net.values.copy()
    for _g, ii in grp.items():
        sh[ii] = rs.permutation(sh[ii])
    s = pd.Series(sh, index=base.index)
    mx = -1e9
    for k, t in sel_idx.items():
        m = s.reindex(pd.MultiIndex.from_arrays([t.tgt, t.key, t.day, t.symbol])).values
        mx = max(mx, float(pd.Series(m).groupby(t.wk.values).sum().reindex(weeks).fillna(0).mean()))
    nulls.append(mx)
    if (d + 1) % 100 == 0:
        print(f'{d+1}/{NDRAW}', flush=True)
nulls = np.array(nulls)
pval = float((nulls >= obs_max).mean())
print(f'search-adjusted permutation p (max weekly R over {len(sel_idx)} cells) = {pval:.3f}', flush=True)
print('null max weekly R: mean %.2f sd %.2f p50 %.2f p95 %.2f max %.2f' %
      (nulls.mean(), nulls.std(ddof=1), np.percentile(nulls, 50), np.percentile(nulls, 95), nulls.max()), flush=True)
pd.DataFrame(dict(null_max_wkR=nulls)).to_csv(f'{D1}/perm_val.csv', index=False)
pd.DataFrame([dict(obs_max_wkR=round(obs_max, 3), p=pval, ndraw=NDRAW, ncells=len(sel_idx),
                   null_mean=round(float(nulls.mean()), 3),
                   null_p95=round(float(np.percentile(nulls, 95)), 3))]).to_csv(f'{D1}/perm_val_summary.csv', index=False)
print('DONE', flush=True)
