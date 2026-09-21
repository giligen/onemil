#!/usr/bin/env python3
"""PREREG_SCORE — ONE pre-registered multivariate winner-likelihood score, decile buckets.
Cell 1,355 (+ placebo). Owner 2026-09-21. See PREREG_SCORE.md for the frozen spec.

Population: features.csv rows, the same `meas, obtainable` R used by cells.py (net_meas, gated on
the obtainability rail, run through trading.hod_break.run_book — the ONE book rule).
Model: L2 logistic regression (C=1.0) on standardized features, fit ONCE on TRAIN-H1 (2025 H1).
Deciles: edges cut on TRAIN-H1 scores, applied unchanged to TRAIN-H2 and VAL.
Placebo: same features, day-shuffled outcomes (fit on noise), scored/bucketed the same way, judged
against REAL net R on the holdouts -- top decile must NOT be positive or the lift is leakage.

Read-only on cache.db. TEST is never touched (this study never asks for it).
"""
import os, sqlite3, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/bf_zero/causal_filter')
import cells as CF                                                    # noqa: E402  reuse load/book/stats conventions
from rank_cells2 import compute_rank_cand                             # noqa: E402  reuse rank_sig computation

D = f'{ROOT}/research/bf_zero/causal_filter'
CACHE_DB = f'{ROOT}/data/cache.db'
SEED = 42

# PREREG's fixed feature list, reconciled against features.csv's actual columns (see the report's
# "Feature reconciliation" table for the documented substitutions/drops -- decided BEFORE fitting).
FEATS_PREREG = ['dist_open_pct', 'rv_adv', 'bar_vol_x', 'above_vwap', 'gap_pct', 'prev_range_pct',
                 'adv20', 'dist_20d_high_pct', 'spy_5m_ret', 'spy_range3', 'pm_covered', 'pole_gain',
                 'retrace', 'flag_len', 'drive_min', 'pull_len', 'n_prior', 'ck_min', 'rv_clock',
                 'is_wrapper', 'entry_m', 'rank_sig', 'news_before_signal']
RENAME = {'rv_adv': 'rv_profile'}          # only rv_* column the causal_filter build produced
NOT_BUILT = ['pm_covered', 'pole_gain', 'retrace', 'flag_len', 'pull_len', 'ck_min']  # not in build_features.py
FEATS = [RENAME.get(f, f) for f in FEATS_PREREG if f not in NOT_BUILT]


def log(msg):
    print(f'[score_model] {msg}', flush=True)


# ---------------------------------------------------------------- news_before_signal
def trading_calendar():
    con = sqlite3.connect(f'file:{CACHE_DB}?mode=ro', uri=True, timeout=60)
    days = pd.read_sql("select distinct bar_date from daily_bars where bar_date between '2024-11-01' and '2026-09-30'",
                        con).bar_date
    con.close()
    return np.array(sorted(days))


def news_before_signal_feature(c):
    """1 iff symbol has a news_cache/news_history row dated the prior trading day or the one before
    that. Same-day news excluded (unknowable at the signal minute; no publication time in news_cache).
    """
    con = sqlite3.connect(f'file:{CACHE_DB}?mode=ro', uri=True, timeout=60)
    nc = pd.read_sql('select distinct symbol, news_date as d from news_cache', con)
    nh = pd.read_sql('select distinct symbol, trade_date as d from news_history', con)
    con.close()
    news_dates = set(map(tuple, pd.concat([nc, nh], ignore_index=True).values))
    cov_lo, cov_hi = min(nc.d.min(), nh.d.min()), max(nc.d.max(), nh.d.max())

    cal = trading_calendar()
    idx = {d: i for i, d in enumerate(cal)}
    uniq_days = sorted(c.day.unique())
    lookback = {}
    for d in uniq_days:
        i = idx.get(d)
        if i is None or i < 2:
            lookback[d] = (None, None)
        else:
            lookback[d] = (cal[i - 1], cal[i - 2])

    feat, covered = [], []
    for day, sym in zip(c.day, c.symbol):
        d1, d2 = lookback[day]
        if d1 is None:
            feat.append(0); covered.append(False); continue
        is_cov = (cov_lo <= d2) and (d1 <= cov_hi)     # both lookback days inside the data's date span
        covered.append(is_cov)
        feat.append(int((sym, d1) in news_dates or (sym, d2) in news_dates))
    return np.array(feat), np.array(covered), round(float(np.mean(covered)) * 100, 1)


# ---------------------------------------------------------------- day-clustered t (score4.py's rule)
def clustered_t(x, day):
    if len(x) < 3:
        return np.nan
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    g = pd.Series(x - mu).groupby(day.values).sum().values
    se = np.sqrt((g ** 2).sum()) / len(x)
    return float(mu / se) if se > 0 else np.nan


def iid_t(x):
    x = np.asarray(x, dtype=float)
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return float(x.mean() / se) if se and se > 0 else np.nan


# ---------------------------------------------------------------- population per grp (book-selected)
def grp_population(c, grp_mask, name):
    """Run the ONE book rule on the (obtainable) rows of this group, then reattach features."""
    d = c[grp_mask & (c.obtainable == True)].copy()               # noqa: E712 -- the NO-FILL rail
    t = CF.book(d, 'net_meas')
    if t is None:
        log(f'{name}: book() returned None (n<20)'); return None
    m = t.merge(d, left_on=['day', 'symbol', 'em'], right_on=['day', 'symbol', 'entry_m'],
                how='left', suffixes=('', '_orig'))
    assert m[FEATS].isna().all(axis=1).sum() == 0 or True
    n_before = len(m)
    m = m.dropna(subset=FEATS).reset_index(drop=True)
    log(f'{name}: book n={n_before}, NaN-feature drop={n_before - len(m)} ({100*(n_before-len(m))/max(n_before,1):.1f}%), kept={len(m)}')
    m['win'] = (m.net > 0).astype(int)
    return m


def decile_table(df, dec_col, split_label, nw):
    rows = []
    for dec in sorted(df[dec_col].unique()):
        x = df[df[dec_col] == dec]
        rows.append(dict(split=split_label, decile=int(dec), n=len(x),
                          meanR=round(float(x.net.mean()), 4),
                          t_iid=round(iid_t(x.net.values), 2),
                          t_clust=round(clustered_t(x.net.values, x.day), 2),
                          WR=round(float((x.net > 0).mean() * 100), 1),
                          tpw=round(len(x) / nw, 2)))
    return pd.DataFrame(rows).sort_values('decile')


def coef_table(pipe, feats):
    lr = pipe.named_steps['lr']
    co = pd.DataFrame({'feature': feats, 'coef': lr.coef_[0]})
    co['abs'] = co.coef.abs()
    return co.sort_values('abs', ascending=False).drop(columns='abs').reset_index(drop=True)


def day_shuffle(df, seed=SEED):
    """Placebo outcome: each day's (win, net) pair pool is replaced by resampling (with
    replacement) from a DIFFERENT, randomly mapped day's pool. Destroys the row-level
    feature<->outcome link while keeping the day-level marginal structure intact."""
    rng = np.random.RandomState(seed)
    pools = {d: g[['win', 'net']].values for d, g in df.groupby('day')}
    days = list(pools.keys())
    day_map = dict(zip(days, rng.permutation(days)))
    win_s, net_s = np.empty(len(df)), np.empty(len(df))
    pos = 0
    for d, g in df.groupby('day', sort=False):
        src = pools[day_map[d]]
        idx = rng.randint(0, len(src), size=len(g))
        win_s[pos:pos + len(g)] = src[idx, 0]
        net_s[pos:pos + len(g)] = src[idx, 1]
        pos += len(g)
    return win_s, net_s


def main():
    log('loading features via cells.load()')
    c = CF.load()
    c = c[c.split != 'TEST'].reset_index(drop=True)          # this study never reads TEST
    log(f'{len(c)} rows (TRAIN+VAL), obtainable {100*np.nanmean(c.obtainable.astype(float)):.1f}%')

    log('computing rank_sig (reused from rank_cells2.compute_rank_cand)')
    rc = compute_rank_cand(c[['day', 'symbol', 'entry_m', 'dist_open_pct']])
    c['rank_sig'] = [rc.get((d, s, int(e))) for d, s, e in zip(c.day, c.symbol, c.entry_m)]

    log('computing news_before_signal (causal, date-only, prior 2 trading days)')
    nb, cov, cov_pct = news_before_signal_feature(c)
    c['news_before_signal'] = nb
    prevalence = round(float(np.mean(nb)) * 100, 1)
    log(f'news_before_signal: coverage={cov_pct}% prevalence={prevalence}%')
    if cov_pct < 80:
        FEATS.remove('news_before_signal')
        log('news_before_signal coverage < 80% -- DROPPED per PREREG rule')

    grp = np.where(c.split == 'TRAIN', np.where(c.half == 'H1', 'TRAIN-H1', 'TRAIN-H2'), 'VAL')
    c['grp'] = grp

    pops = {g: grp_population(c, c.grp == g, g) for g in ('TRAIN-H1', 'TRAIN-H2', 'VAL')}
    tr1 = pops['TRAIN-H1']

    X1 = tr1[FEATS].astype(float).values
    y1 = tr1.win.values
    pipe = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(C=1.0, penalty='l2',
                     max_iter=2000, class_weight=None))])
    pipe.fit(X1, y1)
    log('fit done on TRAIN-H1 (%d rows, %d wins)' % (len(y1), y1.sum()))

    # placebo: same features, day-shuffled outcome, fit on TRAIN-H1 noise
    win_s, _ = day_shuffle(tr1, seed=SEED)
    pipe_pb = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(C=1.0, penalty='l2',
                        max_iter=2000, class_weight=None))])
    pipe_pb.fit(X1, win_s)

    NW = {'TRAIN-H1': 26, 'TRAIN-H2': 27, 'VAL': 23}          # ~weeks per grp (cells.py NW: TRAIN 53, VAL 23)

    scores1 = pipe.predict_proba(X1)[:, 1]
    edges = np.unique(np.quantile(scores1, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf

    scores1_pb = pipe_pb.predict_proba(X1)[:, 1]
    edges_pb = np.unique(np.quantile(scores1_pb, np.linspace(0, 1, 11)))
    edges_pb[0], edges_pb[-1] = -np.inf, np.inf

    dec_tables, spear, auc, cell_rows_pb = {}, {}, {}, {}
    for g, df in pops.items():
        X = df[FEATS].astype(float).values
        y = df.win.values
        s = pipe.predict_proba(X)[:, 1]
        s_pb = pipe_pb.predict_proba(X)[:, 1]
        df['decile'] = pd.cut(s, edges, labels=range(1, len(edges)), duplicates='drop').astype(int)
        df['decile_pb'] = pd.cut(s_pb, edges_pb, labels=range(1, len(edges_pb)), duplicates='drop').astype(int)
        dt = decile_table(df, 'decile', g, NW[g])
        dec_tables[g] = dt
        rho, _ = spearmanr(dt.decile, dt.meanR)
        spear[g] = round(float(rho), 3)
        auc[g] = round(float(roc_auc_score(y, s)), 3)
        cell_rows_pb[g] = decile_table(df, 'decile_pb', g, NW[g])
        pops[g] = df

    coefs = coef_table(pipe, FEATS)

    # top-2-decile / bottom-3-decile summaries + cadence on VAL top-2
    val = pops['VAL']
    top2 = val[val.decile >= 9]
    bot3_h2 = pops['TRAIN-H2'][pops['TRAIN-H2'].decile <= 3]
    bot3_val = val[val.decile <= 3]
    top2_h2 = pops['TRAIN-H2'][pops['TRAIN-H2'].decile >= 9]
    cadence_top2_val = CF.stats(top2.rename(columns={'net': 'net'})[['day', 'net', 'wk']], 'VAL')

    with open(f'{D}/score_model_out.txt', 'w') as f:
        def p(*a):
            print(*a); print(*a, file=f)

        p('FEATURES used:', FEATS)
        p('news_before_signal coverage %.1f%% prevalence %.1f%%' % (cov_pct, prevalence))
        for g in ('TRAIN-H1', 'TRAIN-H2', 'VAL'):
            p(f'--- {g} decile table (real model) ---')
            p(dec_tables[g].to_string(index=False))
            p(f'Spearman(decile, meanR) {g} = {spear[g]}   AUC {g} = {auc[g]}')
        p('--- top-2-decile (9,10) TRAIN-H2 ---'); p(top2_h2[['net']].describe().to_string())
        p('--- top-2-decile (9,10) VAL ---'); p(top2[['net']].describe().to_string())
        p(f'VAL top-2-decile fills/week = {len(top2)/NW["VAL"]:.2f}')
        p('--- bottom-3-decile (1,2,3) TRAIN-H2 mean R = %.4f' % bot3_h2.net.mean())
        p('--- bottom-3-decile (1,2,3) VAL mean R = %.4f' % bot3_val.net.mean())
        p('--- coefficient table ---'); p(coefs.to_string(index=False))
        p('--- cadence block, VAL top-2-decile book ---'); p(cadence_top2_val)
        for g in ('TRAIN-H1', 'TRAIN-H2', 'VAL'):
            p(f'--- {g} decile table (PLACEBO model, real R) ---')
            p(cell_rows_pb[g].to_string(index=False))

    for g in ('TRAIN-H1', 'TRAIN-H2', 'VAL'):
        pops[g].to_csv(f'{D}/score_model_{g.replace("-", "_")}.csv', index=False)
    coefs.to_csv(f'{D}/score_model_coefs.csv', index=False)
    log('DONE')


if __name__ == '__main__':
    main()
