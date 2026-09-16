#!/usr/bin/env python3
"""D0 step 6 — the search-adjusted permutation p on VAL (PLAN §1: "the null taken as the MAX weekly
R over ALL cells of that stage").

Conditional permutation: the fitted models, the selections and the book are held FIXED; the realised
net R is permuted WITHIN each day among that family's candidates, 500 times. The statistic is the
maximum weekly R over all 18 declared cells; p = share of draws whose max >= the observed max.
This asks exactly "does the selection rule pick rows whose outcomes beat a random re-assignment of
the same day's outcomes?", and it charges the whole 18-cell search.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/D')
import d0_core as K
import d0_eval as E

D = 'research/fuckup_audit/D'
NDRAW = 500
P = pd.read_csv(f'{D}/preds.csv', dtype={'symbol': str, 'day': str, 'key': str, 'split': str,
                                         'month': str, 'wk': str, 'why_hold': str},
                keep_default_na=False, na_values=[''])
p = P[P.split == 'VAL'].reset_index(drop=True)
weeks = sorted(p.wk.unique())
cells = []
for key in K.KEYS:
    for mod, _ in E.MODELS:
        cells.append((key, mod, 'S1'))
        cells.append((key, mod, 'S2'))

# freeze the SELECTED ROW SETS once (they do not depend on the outcome permutation)
sel_idx = {}
for key, mod, s in cells:
    x = p[p.key == key]
    y = E.sel_s1(x, f'{mod}_real') if s == 'S1' else E.sel_s2(x, mod, 'real')
    t = K.book(y)
    sel_idx[(key, mod, s)] = None if t is None else y.set_index(['day', 'symbol']).index
    # book membership has to be recomputed per draw only if net changed the book — it does not:
    # run_book is driven by entry_m / exit_m, never by net. So the booked SET is fixed.
    if t is not None:
        z = t[['day', 'symbol', 'wk']].copy(); z['key'] = key
        sel_idx[(key, mod, s)] = z

obs = {}
for k, t in sel_idx.items():
    if t is None:
        continue
    m = p.set_index(['key', 'day', 'symbol']).net.reindex(
        pd.MultiIndex.from_arrays([t.key, t.day, t.symbol])).values
    obs[k] = float(pd.Series(m).groupby(t.wk.values).sum().reindex(weeks).fillna(0).mean())
obs_max = max(obs.values())
print('observed weekly R per cell:', {f'{a.split()[0]}{a.split()[-1]}|{b}|{c}': round(v, 2) for (a, b, c), v in obs.items()}, flush=True)
print('observed MAX weekly R over the 18 cells:', round(obs_max, 3), flush=True)

rs = np.random.RandomState(20260916)
base = p.set_index(['key', 'day', 'symbol'])
nulls = []
grp = p.groupby(['key', 'day']).indices   # permute WITHIN (family, day): each cell selects inside one family
for d in range(NDRAW):
    sh = p.net.values.copy()
    for _day, ii in grp.items():
        sh[ii] = rs.permutation(sh[ii])
    s = pd.Series(sh, index=base.index)
    mx = -1e9
    for k, t in sel_idx.items():
        if t is None:
            continue
        m = s.reindex(pd.MultiIndex.from_arrays([t.key, t.day, t.symbol])).values
        w = float(pd.Series(m).groupby(t.wk.values).sum().reindex(weeks).fillna(0).mean())
        mx = max(mx, w)
    nulls.append(mx)
    if (d + 1) % 100 == 0:
        print(f'{d+1}/{NDRAW}', flush=True)
nulls = np.array(nulls)
pval = float((nulls >= obs_max).mean())
print(f'search-adjusted permutation p (max weekly R over 18 cells) = {pval:.3f}', flush=True)
print('null max weekly R: mean %.2f sd %.2f p50 %.2f p95 %.2f max %.2f' %
      (nulls.mean(), nulls.std(ddof=1), np.percentile(nulls, 50), np.percentile(nulls, 95), nulls.max()), flush=True)
pd.DataFrame(dict(null_max_wkR=nulls)).to_csv(f'{D}/perm_val.csv', index=False)
pd.DataFrame([dict(obs_max_wkR=round(obs_max, 3), p=pval, ndraw=NDRAW,
                   null_mean=round(float(nulls.mean()), 3), null_p95=round(float(np.percentile(nulls, 95)), 3))]
             ).to_csv(f'{D}/perm_val_summary.csv', index=False)
print('DONE', flush=True)
