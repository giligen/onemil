#!/usr/bin/env python3
"""Meta-label study, step 2 — the PURGED, EMBARGOED WALK-FORWARD scorer.

No random K-fold, ever (PREREG).  For each test month M:

  train window = rows dated in [M_start - 182 days, E)   (26 weeks, the window
                 scripts/orb_weekly_refit.py already uses for the selection)
  embargo      = E is the 6th trading session before M_start, i.e. the 5
                 sessions immediately before the test month are dropped
  purge        = any training row whose OUTCOME window overlaps the test month
                 is dropped.  The ORB outcome window is 09:35 -> 15:45 of the
                 candidate's own session, so no row dated before M_start can
                 overlap; the purge is implemented and its count reported (it is
                 0 by construction, which is the honest statement, not a claim
                 that purging was unnecessary in general).
  minimum      = 50 distinct training sessions, else the month is left UNSCORED
                 and the pipeline falls back to the shipped ranking there.

Writes a sidecar CSV (symbol, date, <score col>) for study_orb_pipeline_static_lock.py.

usage: walkforward.py --model m1|m2|m3 --flags 0|1 --depth D --trees T
                      --arm meas_cost|asis --last-month YYYY-MM --out PATH
                      [--shuffle] [--drop-family NAME]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/meta_label')
from build_dataset import FAMILIES, FEATURES, VETO_FLAGS  # noqa: E402

M = f'{ROOT}/research/meta_label'
WINDOW_DAYS = 182          # 26 weeks
EMBARGO_SESSIONS = 5
MIN_TRAIN_SESSIONS = 50
FIRST_SCORED_MONTH = '2025-04'
LR = 0.05
MIN_CHILD = 20


def grades(r: np.ndarray) -> np.ndarray:
    """Relevance grades for the pairwise ranker (declared before any fit)."""
    g = np.zeros(len(r), dtype=int)
    g[r > 0] = 1
    g[r > 1] = 2
    g[r > 2] = 3
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=('m1', 'm2', 'm3'))
    ap.add_argument('--flags', type=int, default=0)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--trees', type=int, default=200)
    ap.add_argument('--arm', default='meas_cost')
    ap.add_argument('--last-month', default='2026-05')
    ap.add_argument('--out', required=True)
    ap.add_argument('--col', default='meta_score')
    ap.add_argument('--shuffle', action='store_true')
    ap.add_argument('--drop-family', default='')
    a = ap.parse_args()

    import xgboost as xgb

    d = pd.read_csv(f'{M}/meta_dataset.csv', keep_default_na=False, na_values=[''],
                    dtype={'symbol': str, 'date': str})
    d = d.sort_values(['date', 'symbol']).reset_index(drop=True)
    d['month'] = d.date.str[:7]

    cols = list(FEATURES) + (list(VETO_FLAGS) if a.flags else [])
    if a.drop_family:
        cols = [c for c in cols if c not in FAMILIES[a.drop_family]]
    X = d[cols].astype(float).to_numpy()
    R = d[f'R_{a.arm}'].astype(float).to_numpy()
    sessions = np.array(sorted(d.date.unique()))

    months = [m for m in sorted(d.month.unique())
              if FIRST_SCORED_MONTH <= m <= a.last_month]
    score = np.full(len(d), np.nan)
    n_purged_tot = 0
    rows = []

    for mi, mo in enumerate(months):
        m_start = f'{mo}-01'
        prior = sessions[sessions < m_start]
        if len(prior) <= EMBARGO_SESSIONS:
            continue
        emb_end = prior[-EMBARGO_SESSIONS]          # first embargoed session
        win_start = (pd.Timestamp(m_start) - pd.Timedelta(days=WINDOW_DAYS)
                     ).strftime('%Y-%m-%d')
        tr = (d.date >= win_start) & (d.date < emb_end)
        # purge: drop any training row whose outcome window overlaps the test
        # month (same-session outcome => vacuous here, counted for the record)
        m_end = (pd.Timestamp(m_start) + pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')
        purge = tr & (d.date >= m_start) & (d.date <= m_end)
        n_purged_tot += int(purge.sum())
        tr = tr & ~purge
        te = (d.month == mo).to_numpy()
        n_sess = d.loc[tr, 'date'].nunique()
        if n_sess < MIN_TRAIN_SESSIONS or te.sum() == 0:
            rows.append((mo, n_sess, int(tr.sum()), int(te.sum()), 'UNSCORED'))
            continue

        Xtr, Xte = X[tr.to_numpy()], X[te]
        Rtr = R[tr.to_numpy()].copy()
        if a.shuffle:
            rng = np.random.RandomState(10_000 + mi)
            rng.shuffle(Rtr)                      # label <-> feature link broken

        common = dict(max_depth=a.depth, n_estimators=a.trees, learning_rate=LR,
                      min_child_weight=MIN_CHILD, n_jobs=1, tree_method='hist',
                      random_state=42, verbosity=0)
        if a.model == 'm1':
            mdl = xgb.XGBClassifier(objective='binary:logistic',
                                    eval_metric='logloss', **common)
            mdl.fit(Xtr, (Rtr > 0).astype(int))
            s = mdl.predict_proba(Xte)[:, 1]
        elif a.model == 'm3':
            mdl = xgb.XGBRegressor(objective='reg:squarederror', **common)
            mdl.fit(Xtr, Rtr)
            s = mdl.predict(Xte)
        else:                                      # m2 — LambdaMART-style
            qid = pd.factorize(d.loc[tr, 'date'])[0]
            mdl = xgb.XGBRanker(objective='rank:pairwise', **common)
            mdl.fit(Xtr, grades(Rtr), qid=qid)
            s = mdl.predict(Xte)
        score[te] = s
        rows.append((mo, n_sess, int(tr.sum()), int(te.sum()), 'scored'))

    out = d[['symbol', 'date']].copy()
    out[a.col] = score
    out.to_csv(a.out, index=False)
    n_scored = int(np.isfinite(score).sum())
    print(f'{a.model} flags={a.flags} depth={a.depth} trees={a.trees} arm={a.arm}'
          f'{" SHUFFLED" if a.shuffle else ""}'
          f'{" -" + a.drop_family if a.drop_family else ""}: '
          f'{len(cols)} features, {len([r for r in rows if r[4] == "scored"])} '
          f'months scored, {n_scored}/{len(d)} rows scored, '
          f'{n_purged_tot} rows purged -> {a.out}')
    for r in rows:
        if r[4] == 'UNSCORED':
            print(f'    {r[0]}: UNSCORED ({r[1]} train sessions < {MIN_TRAIN_SESSIONS})')


if __name__ == '__main__':
    main()
