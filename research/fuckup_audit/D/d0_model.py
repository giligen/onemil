#!/usr/bin/env python3
"""D0 step 4 — the walk-forward models and the transparent baseline, on three tapes.

Tapes
  real : the target is Stage A contract (c) net R (hold exit)
  rev  : NAGEL REVERSED TAPE — net -> -net, win -> 1-win. The pipeline is identical. If the selected
         book is profitable HERE too, the model is fitting population momentum, and the cell fails.
  shuf : targets permuted WITHIN each day (the null of the selection machinery).

Walk-forward: train on everything strictly before the predicted month, first prediction month 2025-10
(training window 2025-01-02..2025-09-30), monthly refit through 2026-09. One fixed config, no tuning.
Transparent baseline: decile monotonicity on TRAIN only, frozen, never refit.

Writes research/fuckup_audit/D/preds.csv (one row per candidate that received a prediction) and
research/fuckup_audit/D/importance.csv. It evaluates NOTHING — d0_eval.py does that, so that TEST can
stay unread until the VAL table is frozen in writing.
"""
import os, sys, json, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/D')
import d0_core as K

D = 'research/fuckup_audit/D'
HP = dict(max_depth=4, max_iter=200, learning_rate=0.05, min_samples_leaf=200,
          l2_regularization=1.0, random_state=0)
FIRST_PRED = '2025-10'
NEWS_MIN_COV = 0.80
RS = np.random.RandomState(12345)

c = K.load()
news_cols = ['f_news_pre', 'f_news_intraday']
cov = float(c.f_news_pre.notna().mean()) if 'f_news_pre' in c.columns else 0.0
drop = () if cov >= NEWS_MIN_COV else tuple(x for x in news_cols if x in c.columns)
FEATS = K.feat_cols(c, drop=drop)
print(f'news coverage {cov:.3f} -> news features {"IN" if not drop else "OUT"} | {len(FEATS)} features', flush=True)
json.dump({'news_coverage': cov, 'news_in': not bool(drop), 'features': FEATS},
          open(f'{D}/model_config.json', 'w'), indent=1)

months = sorted(c.month.unique())
PRED_MONTHS = [m for m in months if m >= FIRST_PRED]
print('prediction months', PRED_MONTHS[0], '->', PRED_MONTHS[-1], len(PRED_MONTHS), flush=True)


def baseline_fit(x, y):
    """Decile monotonicity on TRAIN rows only. Returns (chosen features, edges, directions, rows table)."""
    rows = []
    edges = {}
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
        mu = pd.Series(y[ok]).groupby(pd.Series(d[ok])).mean()
        mu = mu.reindex(range(len(q) - 1))
        rho = spearmanr(mu.index.values[mu.notna()], mu.values[mu.notna()]).correlation
        k = max(1, len(mu.dropna()) // 3)
        spread = float(mu.dropna().iloc[-k:].mean() - mu.dropna().iloc[:k].mean())
        rows.append(dict(feat=f, rho=round(float(rho), 3), spread=round(spread, 4),
                         n_dec=int(len(q) - 1), note=''))
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
        d[np.isnan(v)] = (len(q) - 2) / 2.0                      # a NaN scores mid-rank
        s += d if direc[f] > 0 else (len(q) - 2 - d)
    return s


out_rows = []
imp_rows = []
base_tables = []
t0 = time.time()
for key in K.KEYS:
    dk = c[c.key == key].reset_index(drop=True)
    X = dk[FEATS]
    # the baseline is fitted on the FIRST training window only (2025-01-02..2025-09-30), frozen
    # forever, so that it is out-of-sample everywhere the walk-forward models are (amended before any
    # fit was run; see PREREG.md §0.3 note).
    tr_mask = dk.month.values < FIRST_PRED
    tapes = {}
    tapes['real'] = dk.net.values.copy()
    tapes['rev'] = -dk.net.values
    sh = dk.net.values.copy()
    for _, idx in dk.groupby('day').groups.items():
        ii = np.asarray(list(idx))
        sh[ii] = RS.permutation(sh[ii])
    tapes['shuf'] = sh
    res = dk[['day', 'symbol', 'key', 'sig_m', 'entry_m', 'exit_m_hold', 'why_hold', 'wk', 'month',
              'split', 'net', 'win']].copy()
    res['net_shuf'] = sh
    for tape, y in tapes.items():
        ybin = (y > 0).astype(int)
        ch, ed, di, BT = baseline_fit(X[tr_mask], y[tr_mask])
        BT['key'] = key; BT['tape'] = tape
        BT['chosen'] = BT.feat.isin(ch)
        base_tables.append(BT)
        res[f'base_{tape}'] = baseline_score(X, ch, ed, di)
        res[f'base_{tape}_thr'] = np.median(res[f'base_{tape}'].values[tr_mask])
        pr = np.full(len(dk), np.nan)
        pc = np.full(len(dk), np.nan)
        for m in PRED_MONTHS:
            tr = dk.month.values < m
            te = dk.month.values == m
            if tr.sum() < 500 or te.sum() == 0:
                continue
            g = HistGradientBoostingRegressor(**HP).fit(X[tr], y[tr])
            pr[te] = g.predict(X[te])
            h = HistGradientBoostingClassifier(**HP).fit(X[tr], ybin[tr])
            pc[te] = h.predict_proba(X[te])[:, 1]
            if tape == 'real':
                pi = permutation_importance(g, X[te], y[te], n_repeats=3, random_state=0, n_jobs=1)
                for f, mu in zip(FEATS, pi.importances_mean):
                    imp_rows.append(dict(key=key, month=m, feat=f, imp=float(mu)))
        res[f'reg_{tape}'] = pr
        res[f'clf_{tape}'] = pc
        print(f'{key} {tape} done | {(time.time()-t0)/60:.1f} min | baseline {ch} {di}', flush=True)
    out_rows.append(res)

P = pd.concat(out_rows, ignore_index=True)
P = P[P.reg_real.notna()].reset_index(drop=True)
P.to_csv(f'{D}/preds.csv', index=False)
pd.concat(base_tables, ignore_index=True).to_csv(f'{D}/baseline_univariate.csv', index=False)
pd.DataFrame(imp_rows).to_csv(f'{D}/importance.csv', index=False)
print('preds', P.shape, 'months', P.month.min(), P.month.max(), flush=True)
print(P.groupby(['key', 'split']).size().to_string(), flush=True)
print('DONE', flush=True)
