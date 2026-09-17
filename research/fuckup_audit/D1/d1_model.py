#!/usr/bin/env python3
"""D1 step 4 — the walk-forward models and the transparent baseline, on three tapes, for two targets.

D0's d0_model.py with ONE change of scope: the loop runs over both stop/exit targets ('p' = hold/touch stop,
's' = 2R/stop-1%), and the feature matrix is candidates4's. Hyper-parameters, the walk-forward schedule, the
baseline rule and the three tapes are byte-identical to D0.

Tapes
  real : the target is contract-(c) net R
  rev  : NAGEL REVERSED TAPE — net -> -net. Identical pipeline. A selected book that is profitable HERE is fitting
         the population's momentum, and the cell FAILS (PLAN §3 H4).
  shuf : targets permuted within each day (the null of the selection machinery).

Walk-forward: train on everything strictly before the predicted month; first predicted month 2025-10; monthly refit
through 2026-09. One fixed config, no tuning. The transparent baseline is fitted on months < 2025-10 and frozen.

Permutation importance is computed for the PRIMARY target on the REAL tape only (declared reduction: it is a
diagnostic, and 4 families x 12 months x 53 features x 3 repeats on three tapes and two targets is not affordable
on this node).

Writes D1/preds.csv (one row per candidate that received a prediction), D1/importance.csv,
D1/baseline_univariate.csv. It evaluates NOTHING — d1_eval.py does, so TEST stays unread until VAL is frozen.
"""
import os, sys, json, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/D1')
import d1_core as K

D1 = 'research/fuckup_audit/D1'
TAG = os.environ.get('D1_TAG', '')
DROPG = os.environ.get('D1_DROP', '')
PMNEWS = ('f_log_pm_dollar', 'f_pm_missing', 'f_pm_hi', 'f_news_pre', 'f_news_intraday', 'f_news_missing')
HP = dict(max_depth=4, max_iter=200, learning_rate=0.05, min_samples_leaf=200,
          l2_regularization=1.0, random_state=0)
FIRST_PRED = '2025-10'
RS = np.random.RandomState(12345)

c = K.load(keys=K.KEYS)
drop = PMNEWS if DROPG == 'pmnews' else tuple(x for x in DROPG.split(',') if x)
FEATS = K.feat_cols(c, drop=drop)
print('TAG', repr(TAG), '| dropped:', drop, flush=True)
print(f'{len(c)} rows | {len(FEATS)} features', flush=True)
json.dump({'tag': TAG, 'dropped': list(drop), 'features': FEATS, 'hp': HP, 'first_pred': FIRST_PRED,
           'news_coverage': float(c.f_news_pre.notna().mean()),
           'pm_coverage': float(1 - c.f_pm_missing.mean())}, open(f'{D1}/model_config{TAG}.json', 'w'), indent=1)

months = sorted(c.month.unique())
PRED_MONTHS = [m for m in months if m >= FIRST_PRED]
print('prediction months', PRED_MONTHS[0], '->', PRED_MONTHS[-1], len(PRED_MONTHS), flush=True)


def baseline_fit(x, y):
    """Decile monotonicity on the first training window only. Returns (chosen, edges, directions, table)."""
    rows, edges = [], {}
    for f in FEATS:
        v = x[f].values
        ok = ~np.isnan(v)
        if ok.sum() < 2000 or len(np.unique(v[ok])) < 10:
            rows.append(dict(feat=f, rho=np.nan, spread=np.nan, note='too few distinct/known'))
            continue
        q = np.unique(np.nanquantile(v, np.linspace(0, 1, 11)))
        if len(q) < 4:
            rows.append(dict(feat=f, rho=np.nan, spread=np.nan, note='degenerate deciles'))
            continue
        d = np.digitize(v, q[1:-1])
        mu = pd.Series(y[ok]).groupby(pd.Series(d[ok])).mean().reindex(range(len(q) - 1))
        rho = spearmanr(mu.index.values[mu.notna()], mu.values[mu.notna()]).correlation
        k = max(1, len(mu.dropna()) // 3)
        spread = float(mu.dropna().iloc[-k:].mean() - mu.dropna().iloc[:k].mean())
        rows.append(dict(feat=f, rho=round(float(rho), 3), spread=round(spread, 4), n_dec=int(len(q) - 1), note=''))
        edges[f] = q
    T = pd.DataFrame(rows)
    elig = T[(T.rho.abs() >= 0.6) & T.spread.notna()].copy()
    elig['abs_spread'] = elig.spread.abs()
    chosen = elig.sort_values('abs_spread', ascending=False).head(3)
    return list(chosen.feat), edges, {f: (1 if r > 0 else -1) for f, r in zip(chosen.feat, chosen.rho)}, T


def baseline_score(x, chosen, edges, direc):
    s = np.zeros(len(x))
    for f in chosen:
        q = edges[f]
        v = x[f].values
        d = np.digitize(v, q[1:-1]).astype(float)
        d[np.isnan(v)] = (len(q) - 2) / 2.0
        s += d if direc[f] > 0 else (len(q) - 2 - d)
    return s


out_rows, imp_rows, base_tables = [], [], []
t0 = time.time()
for tgt in K.TARGETS:
    for key in K.KEYS:
        dk = c[(c.key == key) & c[f'net_{tgt}'].notna()].reset_index(drop=True)
        if not len(dk):
            continue
        X = dk[FEATS]
        tr_mask = dk.month.values < FIRST_PRED
        net = dk[f'net_{tgt}'].values.astype(float)
        tapes = {'real': net.copy(), 'rev': -net}
        sh = net.copy()
        for _d, idx in dk.groupby('day').groups.items():
            ii = np.asarray(list(idx))
            sh[ii] = RS.permutation(sh[ii])
        tapes['shuf'] = sh
        res = dk[['day', 'symbol', 'key', 'sig_m', 'next_entry_m', 'wk', 'month', 'split']].copy()
        res['tgt'] = tgt
        res['net'] = net
        res['xm'] = dk[f'xm_{tgt}'].values
        res['why'] = dk[f'why_{tgt}'].values
        res['net_shuf'] = sh
        for tape, y in tapes.items():
            ybin = (y > 0).astype(int)
            ch, ed, di, BT = baseline_fit(X[tr_mask], y[tr_mask])
            BT['key'] = key; BT['tape'] = tape; BT['tgt'] = tgt
            BT['chosen'] = BT.feat.isin(ch)
            base_tables.append(BT)
            res[f'base_{tape}'] = baseline_score(X, ch, ed, di)
            res[f'base_{tape}_thr'] = np.median(res[f'base_{tape}'].values[tr_mask])
            pr = np.full(len(dk), np.nan); pc = np.full(len(dk), np.nan)
            for m in PRED_MONTHS:
                tr = dk.month.values < m
                te = dk.month.values == m
                if tr.sum() < 500 or te.sum() == 0:
                    continue
                g = HistGradientBoostingRegressor(**HP).fit(X[tr], y[tr])
                pr[te] = g.predict(X[te])
                h = HistGradientBoostingClassifier(**HP).fit(X[tr], ybin[tr])
                pc[te] = h.predict_proba(X[te])[:, 1]
                if tape == 'real' and tgt == 'p':
                    pi = permutation_importance(g, X[te], y[te], n_repeats=3, random_state=0, n_jobs=1)
                    for f, mu in zip(FEATS, pi.importances_mean):
                        imp_rows.append(dict(key=key, month=m, feat=f, imp=float(mu)))
            res[f'reg_{tape}'] = pr
            res[f'clf_{tape}'] = pc
            print(f'{tgt} {key} {tape} done | {(time.time()-t0)/60:.1f} min | baseline {ch} {di}', flush=True)
        out_rows.append(res)

P = pd.concat(out_rows, ignore_index=True)
P = P[P.reg_real.notna()].reset_index(drop=True)
P.to_csv(f'{D1}/preds{TAG}.csv', index=False)
pd.concat(base_tables, ignore_index=True).to_csv(f'{D1}/baseline_univariate{TAG}.csv', index=False)
pd.DataFrame(imp_rows).to_csv(f'{D1}/importance{TAG}.csv', index=False)
print('preds', P.shape, 'months', P.month.min(), P.month.max(), flush=True)
print(P.groupby(['tgt', 'key', 'split']).size().to_string(), flush=True)
print('DONE', flush=True)
