#!/usr/bin/env python3
"""Cell 1,678: make the post-entry shape signal pay -- decomposition, the
remaining-R regression, the ORB transfer, the flat-price subset.

PREREG: research/hod_entry/PREREG_1678.md (FROZEN 2026-09-30 15:05 UTC).

Reuses (imported via importlib, module name starts with a digit):
  f1668.BarStore, .find_fill_index, .walk_k, .dR_cut, .load_population,
  .EOD_M, .ENTRY_BPS, .CUT_BPS, .BARS_DB
  f1670.path_features (family P: mtm_R, mfe_R, mae_R, min_since_new_high,
  bars_since_higher_low, level_retouched)
  f1676.g7_trailing_candle, .candle_shape, .mfe_ge_1R, .money_reads_at_k,
  iid_t, day_clustered_t, ex_top5_mean, mde, stats_for_subset
  f1677.x6_exit_dR, .stats_block, .decompose, .placebo_block, .first_qualify
  (the give-back decomposition / placebo rule / X6 comparator this PREREG's
  header names explicitly)
  gfix (1676_g6g7_fix.py) -- not imported directly; its close_R-drop
  convention is reproduced here (build_shape_at_k never writes a close_R
  column, so no drop step is needed).

Scope disclosures (stated up front, not hidden):
 - G7 shape features here are CLV/body/wick/range_atr/vol_ratio for
   w in {5,10,15} at each k -- NO close_R (that IS mtm, the money feature;
   dropping it is the PREREG's own instruction), no TA-lib flags, no
   climax/effort-residual (those needed the full non-overlapping-series
   machinery 1676 built at real cost on a shared 2-CPU node; bounding this
   is disclosed, matching 1676's own documented TA-lib scope bound).
 - Decision rule X(c)/X+(c) scans the FIXED checkpoint grid k in
   {15,30,45,60,90,120} (Part 1's own k-grid restricted to >=15), not
   literally every minute -- the same simplification 1677's TS variant
   made and disclosed ("checks only the fixed checkpoints ... not literally
   every minute").
 - ORB Part 3: analysis_results/orb_bplus_book.csv carries entry_price,
   pnl_pct and exit_reason but NOT entry_time/stop/exit_time (confirmed
   against research/exec_cost/compute.py's own caveat: "ORB honest book
   carries no entry_time/stop/shares"). This script reconstructs, per the
   documented RULE (research/orb_machine_rules.md L8): stop = range_low of
   the first 5 RTH minutes (09:30-09:34 ET); entry bar = first bar at/after
   09:35 ET whose high touches entry_price (+/- tolerance); exit_price =
   entry_price*(1+pnl_pct/100) (self-contained, no share count needed);
   exit bar = first bar after entry whose range contains exit_price,
   direction chosen by exit_reason (stop/lock-family -> low<=exit_price;
   tag-family -> high>=exit_price; eod-family -> the EOD bar). This is a
   RECONSTRUCTION, disclosed prominently in RESULT_1678.md, not the live
   engine's actual static-lock/touchgo state machine.

Usage:
    python3 1678_remaining_r.py --stage hod    # Part 1, 2, 4 (needs no ORB backfill)
    python3 1678_remaining_r.py --stage orb    # Part 3 (run after the backfill completes)
    python3 1678_remaining_r.py --stage report # assemble RESULT_1678.md from the CSVs on disk
"""
import argparse
import importlib.util
import logging
import math
import os
import sqlite3
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODELS_DIR = os.path.join(HERE, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def _load_module(name, fname, root=HERE):
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, fname))
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = old_argv
    return mod


f1668 = _load_module('f1668_1678', '1668_failure.py')
f1670 = _load_module('f1670_1678', '1670_timing_map.py')
f1676 = _load_module('f1676_1678', '1676_shapes.py')
f1677 = _load_module('f1677_1678', '1677_take_profit.py')

LOG_FILE = os.path.join(HERE, '1678_remaining_r.log')
DECOMP_CSV = os.path.join(HERE, '1678_decomp.csv')
READS_CSV = os.path.join(HERE, '1678_reads.csv')
ORB_PERFILL_CSV = os.path.join(HERE, '1678_orb_per_fill.csv')
PERFILLK_CSV = os.path.join(HERE, '1678_perfillk.csv')
FILLLEVEL_CSV = os.path.join(HERE, '1678_filllevel.csv')
RESULT_MD = os.path.join(HERE, 'RESULT_1678.md')
ORB_BOOK_CSV = os.path.join(ROOT, 'analysis_results', 'orb_bplus_book.csv')

K_GRID = [5, 10, 15, 30, 45, 60, 90, 120]
DECISION_KS = [k for k in K_GRID if k >= 15]
G7_W = [5, 10, 15]
SCORINGS = [('TRAIN-H2', 'VAL'), ('VAL', 'TRAIN-H2')]
C_GRID = [0.05, 0.10, 0.20]
PLACEBO_KS_PART1 = [15, 60]
SEED = 1678
N_PLACEBO_SEEDS = 10
N_BOOTSTRAP = 200
PASS_DR, PASS_T, PASS_VS_PLACEBO = 0.05, 2.5, 0.03
ORB_LIVE_RISK = 375.0

logger = logging.getLogger('1678')


def setup_logging():
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(sh)


# ---------------------------------------------------------------------------
# Stats kit -- reused verbatim from f1676 / f1677 (no reimplementation).
# ---------------------------------------------------------------------------
iid_t, day_clustered_t, ex_top5_mean, mde = f1676.iid_t, f1676.day_clustered_t, f1676.ex_top5_mean, f1676.mde
stats_block, decompose, placebo_block, first_qualify = f1677.stats_block, f1677.decompose, f1677.placebo_block, f1677.first_qualify


# ---------------------------------------------------------------------------
# Part 0: per-fill-per-k feature/label sweep (HOD population), extending
# 1676/1670's k-grids to {90,120} via the same reused primitives.
# ---------------------------------------------------------------------------

def build_shape_at_k(bars, i0, k, entry, R_unit, atr14, mean_vol):
    out = {}
    for w in G7_W:
        c = f1676.g7_trailing_candle(bars, i0, k, w)
        pfx = f'w{w}'
        if c is None:
            for feat in ('clv', 'body', 'uwick', 'lwick', 'range_atr', 'vol_ratio'):
                out[f'{pfx}_{feat}'] = np.nan
            continue
        sh = f1676.candle_shape(c['o'], c['h'], c['l'], c['c'])
        rng = c['h'] - c['l']
        out[f'{pfx}_clv'] = sh['clv']
        out[f'{pfx}_body'] = sh['body']
        out[f'{pfx}_uwick'] = sh['uwick']
        out[f'{pfx}_lwick'] = sh['lwick']
        out[f'{pfx}_range_atr'] = rng / atr14 if atr14 and atr14 > 0 else np.nan
        out[f'{pfx}_vol_ratio'] = c['v'] / mean_vol if mean_vol and mean_vol > 0 else np.nan
    return out


def shape_cols():
    return [f'w{w}_{feat}' for w in G7_W for feat in ('clv', 'body', 'uwick', 'lwick', 'range_atr', 'vol_ratio')]


def process_fill(r, store):
    sym, date = r.symbol, r.date
    bars = store.day_bars(sym, date)
    if bars is None or len(bars['o']) < 5:
        return None
    i0 = f1668.find_fill_index(bars, r.fill_min)
    if i0 is None or i0 + 1 >= len(bars['o']):
        return None
    entry, stop, target = r.entry_price, r.stop, r.target_price
    R_unit = entry - stop
    if not (R_unit > 0):
        return None
    level = r.level
    atr14 = r.atr14
    mean_vol = bars['v'][:i0 + 1].mean() if i0 >= 1 else np.nan
    x6dR = f1677.x6_exit_dR(bars, i0, entry, stop, target, R_unit, r.net_R)
    rows = []
    for k in K_GRID:
        w = f1668.walk_k(bars, i0, k, stop, target)
        if w is None:
            continue
        open_k = (w['preempt'] == '')
        pf = f1670.path_features(bars, i0, i0 + k, entry, R_unit, level)
        shp = build_shape_at_k(bars, i0, k, entry, R_unit, atr14, mean_vol)
        remaining_R = r.net_R - pf['mtm_R']
        succ15 = f1676.mfe_ge_1R(bars, i0, k, entry, R_unit, 15)
        dR_full = f1668.dR_cut(entry, stop, r.net_R, w['next_open'])
        cost_R = (f1668.CUT_BPS * w['next_open'] / R_unit) if w['next_open'] is not None else np.nan
        row = dict(fill_id=r.fill_id, date=date, symbol=sym, half=r.half, net_R=r.net_R, k=k,
                   open_k=open_k, mtm_R=pf['mtm_R'], mfe_R=pf['mfe_R'], mae_R=pf['mae_R'],
                   min_since_new_high=pf['min_since_new_high'], bars_since_higher_low=pf['bars_since_higher_low'],
                   level_retouched=pf['level_retouched'], remaining_R=remaining_R,
                   success_next15=succ15, dR_full=dR_full, cost_R=cost_R, x6_dR=x6dR)
        row.update(shp)
        rows.append(row)
    return rows


def sweep(pop, store, resume=False):
    done_ids = set()
    existing = None
    if resume and os.path.exists(PERFILLK_CSV):
        existing = pd.read_csv(PERFILLK_CSV, dtype={'fill_id': str})
        done_ids = set(existing['fill_id'].astype(str))
        logger.info('resume: %d fills already in %s', len(done_ids), PERFILLK_CSV)
    pop = pop.copy()
    pop['fill_id'] = pop['fill_id'].astype(str)
    todo = pop[~pop['fill_id'].isin(done_ids)]
    logger.info('sweeping %d fills (todo, of %d population)', len(todo), len(pop))
    all_rows = [] if existing is None else existing.to_dict('records')
    t0 = time.time()
    n_ok = n_bad = 0
    for i, r in enumerate(todo.itertuples()):
        try:
            rows = process_fill(r, store)
        except Exception as e:
            logger.warning('fill %s/%s raised %s: %s', r.symbol, r.date, type(e).__name__, e)
            rows = None
        if not rows:
            n_bad += 1
        else:
            all_rows.extend(rows)
            n_ok += 1
        if (i + 1) % 500 == 0 or (i + 1) == len(todo):
            elapsed = time.time() - t0
            logger.info('progress %d/%d fills (ok=%d bad=%d) %.1fs elapsed, %.2f fills/s',
                        i + 1, len(todo), n_ok, n_bad, elapsed, (i + 1) / max(elapsed, 1e-9))
            df_out = pd.DataFrame(all_rows)
            tmp = PERFILLK_CSV + '.tmp'
            df_out.to_csv(tmp, index=False)
            os.replace(tmp, PERFILLK_CSV)
    logger.info('sweep done: %d ok, %d bad, %d total (fill,k) rows written', n_ok, n_bad, len(all_rows))
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Feature sets (Part 1's a/b/c/d).
# ---------------------------------------------------------------------------

def featset_cols(which):
    path = ['mtm_R', 'mfe_R', 'mae_R', 'min_since_new_high', 'bars_since_higher_low', 'level_retouched']
    shape = shape_cols()
    return {'mtm': ['mtm_R'], 'path': path, 'shape': shape, 'path_shape': path + shape}[which]


def fit_scoring(df_k, feat_cols, label_col, task, seed=SEED, do_placebo=False):
    """One HGB fit per (train_half, test_half) direction. task in {'clf','reg'}.
    Returns list of 2 dicts (one per scoring direction) with the OOS metric,
    the OOS predictions (index-aligned) for downstream reuse, and (optionally)
    a within-day label-shuffle placebo metric."""
    feat_cols = [c for c in feat_cols if c in df_k.columns]
    sub = df_k[['fill_id', 'date', 'half', label_col] + feat_cols].dropna(subset=[label_col]).copy()
    for c in feat_cols:
        sub[c] = pd.to_numeric(sub[c], errors='coerce')
    halves = {'TRAIN-H2': sub[sub.half == 'TRAIN-H2'], 'VAL': sub[sub.half == 'VAL']}
    out = []
    for train_h, test_h in SCORINGS:
        tr, te = halves[train_h], halves[test_h]
        rec = dict(scoring=f'{train_h}->{test_h}', n_train=len(tr), n_test=len(te),
                   metric=np.nan, placebo_metric=np.nan, pred_index=None, pred=None)
        min_n = 30
        ok = len(tr) >= min_n and len(te) >= min_n
        if task == 'clf':
            ok = ok and tr[label_col].nunique() > 1 and te[label_col].nunique() > 1
        if not ok or not feat_cols:
            out.append(rec)
            continue
        try:
            Xtr, Xte = tr[feat_cols].values, te[feat_cols].values
            ytr, yte = tr[label_col].values, te[label_col].values
            if task == 'clf':
                model = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
                model.fit(Xtr, ytr.astype(int))
                p = model.predict_proba(Xte)[:, 1]
                rec['metric'] = roc_auc_score(yte, p)
                rec['pred'] = p
            else:
                model = HistGradientBoostingRegressor(max_iter=200, random_state=seed)
                model.fit(Xtr, ytr.astype(float))
                p = model.predict(Xte)
                rec['metric'] = r2_score(yte, p)
                rec['pred'] = p
            rec['pred_index'] = te['fill_id'].values
            rec['pred_dates'] = te['date'].values
            rec['model'] = model
            if do_placebo:
                rng = np.random.RandomState(seed)
                y_shuf = tr.assign(_y=ytr).groupby('date')['_y'].transform(lambda s: rng.permutation(s.values)).values
                if task == 'clf':
                    mp = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
                    mp.fit(Xtr, y_shuf.astype(int))
                    rec['placebo_metric'] = roc_auc_score(yte, mp.predict_proba(Xte)[:, 1])
                else:
                    mp = HistGradientBoostingRegressor(max_iter=200, random_state=seed)
                    mp.fit(Xtr, y_shuf.astype(float))
                    rec['placebo_metric'] = r2_score(yte, mp.predict(Xte))
        except Exception as e:
            logger.warning('fit_scoring failed %s/%s/%s->%s: %s', label_col, task, train_h, test_h, e)
        out.append(rec)
    return out


def bootstrap_increment_ci(pred_b, pred_d, y_true, task, n=N_BOOTSTRAP, seed=SEED):
    """200-draw bootstrap CI of metric(d)-metric(b) resampling the ALREADY
    scored rows (no refitting -- predictions are fixed, matching 1670/1676/
    1677's convention of bootstrapping the scored half, not the fit)."""
    rng = np.random.RandomState(seed)
    n_rows = len(y_true)
    if n_rows < 10:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan)
    deltas = []
    for _ in range(n):
        idx = rng.randint(0, n_rows, n_rows)
        yt = y_true[idx]
        try:
            if task == 'clf':
                if len(np.unique(yt)) < 2:
                    continue
                m_b = roc_auc_score(yt, pred_b[idx])
                m_d = roc_auc_score(yt, pred_d[idx])
            else:
                m_b = r2_score(yt, pred_b[idx])
                m_d = r2_score(yt, pred_d[idx])
            deltas.append(m_d - m_b)
        except Exception:
            continue
    if not deltas:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan)
    deltas = np.array(deltas)
    return dict(mean=float(deltas.mean()), lo=float(np.percentile(deltas, 2.5)), hi=float(np.percentile(deltas, 97.5)))


# ---------------------------------------------------------------------------
# Part 1
# ---------------------------------------------------------------------------

def run_part1(perfillk):
    rows = []
    pred_store = {}   # (featset, label, k, scoring) -> dict(pred_index, pred, dates, model)
    labels = [('success_next15', 'clf'), ('remaining_R', 'reg')]
    for k in K_GRID:
        df_k = perfillk[(perfillk.k == k) & (perfillk.open_k == True)]  # noqa: E712
        for label_col, task in labels:
            fits = {}
            for fs in ('mtm', 'path', 'shape', 'path_shape'):
                do_pb = (k in PLACEBO_KS_PART1 and fs == 'path_shape')
                res = fit_scoring(df_k, featset_cols(fs), label_col, task, do_placebo=do_pb)
                fits[fs] = res
                for r in res:
                    rows.append(dict(part='1', k=k, label=label_col, task=task, featset=fs,
                                      scoring=r['scoring'], n_train=r['n_train'], n_test=r['n_test'],
                                      metric=r['metric'], placebo_metric=r['placebo_metric']))
                    pred_store[(fs, label_col, k, r['scoring'])] = r
                    if fs in ('path', 'path_shape') and r.get('model') is not None:
                        tag = r['scoring'].replace('->', '_to_').replace('-', '')
                        joblib.dump(r['model'], os.path.join(MODELS_DIR, f'1678_{fs}_{label_col}_k{k}_{tag}.joblib'))
            # increment (d)-(b): path_shape - path, bootstrap CI on the scored half
            for r_b, r_d in zip(fits['path'], fits['path_shape']):
                if r_b['scoring'] != r_d['scoring'] or r_b['pred'] is None or r_d['pred'] is None:
                    continue
                # align on fill_id intersection (should be identical index already)
                common = np.intersect1d(r_b['pred_index'], r_d['pred_index'])
                ib = pd.Index(r_b['pred_index']).get_indexer(common)
                idd = pd.Index(r_d['pred_index']).get_indexer(common)
                y_true = df_k.set_index('fill_id').loc[common, label_col].values
                ci = bootstrap_increment_ci(np.asarray(r_b['pred'])[ib], np.asarray(r_d['pred'])[idd], y_true, task)
                rows.append(dict(part='1_increment', k=k, label=label_col, task=task, featset='path_shape-path',
                                  scoring=r_b['scoring'], n_train=np.nan, n_test=len(common),
                                  metric=ci['mean'], placebo_metric=np.nan, ci_lo=ci['lo'], ci_hi=ci['hi']))
        logger.info('Part1 done: k=%d', k)
    return pd.DataFrame(rows), pred_store


# ---------------------------------------------------------------------------
# Part 2: decision rule X(c)/X+ scanning DECISION_KS, first firing wins.
# Reuses f1677.first_qualify's scan pattern (TS variant) adapted to the
# remaining-R regressor's threshold rule instead of a classifier tau gate.
# ---------------------------------------------------------------------------

def build_pivots(perfillk, pred_store, scoring, featset, label='remaining_R'):
    """Wide (fill_id x k) pivots for one scoring direction: open, mtm,
    dR_full, cost, and the OOS predicted remaining_R from pred_store."""
    test_half = scoring.split('->')[1]
    base = perfillk[perfillk.half == test_half]
    piv = {}
    for field in ('open_k', 'mtm_R', 'dR_full', 'cost_R', 'x6_dR', 'net_R'):
        piv[field] = base.pivot_table(index='fill_id', columns='k', values=field, aggfunc='first')
    pred_wide = {}
    for k in K_GRID:
        r = pred_store.get((featset, label, k, scoring))
        if r is None or r['pred'] is None:
            continue
        pred_wide[k] = pd.Series(r['pred'], index=r['pred_index'])
    piv['pred'] = pd.DataFrame(pred_wide)
    dates_of = base.drop_duplicates('fill_id').set_index('fill_id')['date']
    return piv, dates_of.reindex(piv['open_k'].index)


def scan_rule(piv, dates_of, c, variant):
    checkpoints = [k for k in DECISION_KS if k in piv['pred'].columns]
    idx = piv['open_k'].index
    active = pd.Series(True, index=idx)
    fired = pd.Series(False, index=idx)
    realized = pd.Series(0.0, index=idx)
    ever_pool = pd.Series(False, index=idx)
    for kk in checkpoints:
        open_kk = piv['open_k'][kk].reindex(idx).fillna(False).astype(bool)
        mtm_kk = piv['mtm_R'][kk].reindex(idx)
        pred_kk = piv['pred'][kk].reindex(idx)
        gate = open_kk if variant == 'X' else (open_kk & (mtm_kk >= 0.5))
        pool_kk = active & gate & pred_kk.notna()
        ever_pool |= pool_kk
        cond = pool_kk & (pred_kk < -c)
        realized[cond] = piv['dR_full'][kk].reindex(idx)[cond]
        fired |= cond
        active &= ~cond
    return ever_pool, fired, realized, checkpoints


def run_part2(perfillk, pred_store):
    rows = []
    for featset in ('path', 'path_shape'):
        for scoring_dir in ('TRAIN-H2->VAL', 'VAL->TRAIN-H2'):
            piv, dates_of = build_pivots(perfillk, pred_store, scoring_dir, featset)
            if piv['pred'].empty:
                logger.warning('Part2: no predictions for featset=%s scoring=%s', featset, scoring_dir)
                continue
            for c in C_GRID:
                for variant in ('X', 'X+'):
                    ever_pool, fired, realized, checkpoints = scan_rule(piv, dates_of, c, variant)
                    n_pool, n_fired = int(ever_pool.sum()), int(fired.sum())
                    dR_fired = realized[fired]
                    dates_fired = dates_of.reindex(dR_fired.index)
                    st = stats_block(dR_fired.values, dates_fired.values)
                    # cost/decompose: approximated by the START checkpoint's per-R cost scale
                    # (reporting only), same disclosed approximation 1677's TS variant made.
                    dec = decompose(dR_fired, piv['cost_R'][checkpoints[0]].reindex(dR_fired.index).fillna(0))
                    m_thresh = -np.inf if variant == 'X' else 0.5
                    qual_mask, assigned_dR = first_qualify(
                        piv['open_k'][checkpoints], piv['mtm_R'][checkpoints], piv['dR_full'][checkpoints],
                        checkpoints, m_thresh)
                    plc = placebo_block(qual_mask, assigned_dR, dates_of, n_fired) if n_fired else dict(mean_dR=np.nan, day_t=np.nan, ex_top5=np.nan)
                    x6_mean = piv['x6_dR'][checkpoints[0]].reindex(dR_fired.index).mean() if n_fired else np.nan
                    rows.append(dict(part='2', featset=featset, scoring=scoring_dir, c=c, variant=variant,
                                      n_pool=n_pool, n_fired=n_fired,
                                      share_fired=(n_fired / n_pool if n_pool else np.nan), **st, **dec,
                                      placebo_mean_dR=plc['mean_dR'], placebo_day_t=plc['day_t'],
                                      placebo_ex_top5=plc['ex_top5'], x6_mean_dR=x6_mean))
            logger.info('Part2 done: featset=%s scoring=%s', featset, scoring_dir)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 4: the non-tautological subset.
# ---------------------------------------------------------------------------

def run_part4(perfillk, pred_store):
    rows = []
    for k in (30, 45, 60):
        df_k = perfillk[(perfillk.k == k) & (perfillk.open_k == True)].copy()  # noqa: E712
        df_k = df_k.set_index('fill_id')
        for scoring_dir in ('TRAIN-H2->VAL', 'VAL->TRAIN-H2'):
            r = pred_store.get(('path_shape', 'success_next15', k, scoring_dir))
            if r is None or r['pred'] is None:
                continue
            test_half = scoring_dir.split('->')[1]
            sub_half = df_k[df_k.half == test_half]
            p_succ = pd.Series(r['pred'], index=r['pred_index']).reindex(sub_half.index)
            subset_mask = (sub_half['mtm_R'] <= 0.3) & (p_succ >= 0.6)
            subset = sub_half[subset_mask.fillna(False)]
            n_subset = len(subset)
            if n_subset == 0:
                rows.append(dict(part='4', k=k, scoring=scoring_dir, n_subset=0))
                continue
            # 'add' dR via f1676.money_reads_at_k requires exit_price/exit_type/bars -- computed
            # in a light re-walk here (population row has exit_price/exit_type; bars refetched).
            add_dRs, remaining_Rs, dates = [], [], []
            store = f1668.BarStore(f1668.BARS_DB)
            pop_idx = POP_BY_FILL if 'POP_BY_FILL' in globals() else None
            for fid, row in subset.iterrows():
                prow = pop_idx.loc[fid] if pop_idx is not None and fid in pop_idx.index else None
                if prow is None:
                    continue
                bars = store.day_bars(row['symbol'], row['date'])
                if bars is None:
                    continue
                i0 = f1668.find_fill_index(bars, prow['fill_min'])
                if i0 is None:
                    continue
                entry, stop = prow['entry_price'], prow['stop']
                R_unit = entry - stop
                if not (R_unit > 0):
                    continue
                mr = f1676.money_reads_at_k(bars, i0, k, entry, stop, R_unit, prow['exit_price'], prow['exit_type'])
                add_dRs.append(mr['add'])
                remaining_Rs.append(row['remaining_R'])
                dates.append(row['date'])
            store.close()
            add_dRs = np.array(add_dRs, dtype=float)
            dates = np.array(dates)
            st_add = stats_block(add_dRs, dates)
            st_rem = dict(mean=np.nanmean(remaining_Rs) if remaining_Rs else np.nan,
                          day_t=day_clustered_t(pd.DataFrame({'date': dates, 'net_R': remaining_Rs}), 'date', 'net_R') if remaining_Rs else np.nan)
            rows.append(dict(part='4', k=k, scoring=scoring_dir, n_subset=n_subset,
                              add_mean_dR=st_add['mean_dR'], add_day_t=st_add['day_t'], add_ex_top5=st_add['ex_top5'],
                              add_mde=st_add['mde'], subset_remaining_R_mean=st_rem['mean'], subset_remaining_R_day_t=st_rem['day_t']))
        logger.info('Part4 done: k=%d', k)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 3: ORB transfer.
# ---------------------------------------------------------------------------

def orb_reconstruct_stop_entry(bars, entry_price, tol=0.02):
    """range_low/range_high from the first 5 RTH minutes (09:30-09:34 ET,
    minarr 570-574); entry bar = first bar at/after 09:35 ET whose high is
    within `tol` (fraction of price) of entry_price."""
    open_mask = (bars['minarr'] >= 570) & (bars['minarr'] <= 574)
    if open_mask.sum() == 0:
        return None
    range_low = bars['l'][open_mask].min()
    range_high = bars['h'][open_mask].max()
    post_idx = np.where(bars['minarr'] >= 575)[0]
    fill_i = None
    for j in post_idx:
        if bars['h'][j] >= entry_price * (1 - tol):
            fill_i = j
            break
    if fill_i is None:
        return None
    return dict(range_low=range_low, range_high=range_high, i0=fill_i)


def orb_reconstruct_exit(bars, i0, exit_price, exit_reason):
    n = len(bars['o'])
    up = exit_reason in ('tag_bb', 'tag_b1')
    for j in range(i0 + 1, n):
        if up and bars['h'][j] >= exit_price:
            return j
        if (not up) and bars['l'][j] <= exit_price:
            return j
        if bars['minarr'][j] >= f1668.EOD_M:
            return j
    return n - 1 if n > i0 + 1 else None


def run_part3(pred_store):
    setup_logging()
    logger.info('Part 3: ORB transfer starting')
    orb_csv = _load_module('orb_csv_1678', 'trading/orb_csv.py', root=ROOT)
    df = orb_csv.read_orb_csv(ORB_BOOK_CSV)
    ent = df[df['entered'] == 1].copy()
    ent['date'] = ent['date'].astype(str)
    logger.info('ORB book: %d entered fills 2025-01..2026-09', len(ent))

    store = f1668.BarStore(f1668.BARS_DB)
    rows = []
    n_no_bars = n_no_range = n_no_fill = n_bad_R = n_ok = 0
    for i, r in enumerate(ent.itertuples()):
        bars = store.day_bars(r.symbol, r.date)
        if bars is None or len(bars['o']) < 5:
            n_no_bars += 1
            continue
        rec = orb_reconstruct_stop_entry(bars, r.entry_price)
        if rec is None:
            n_no_range += 1
            continue
        i0 = rec['i0']
        entry, stop = r.entry_price, rec['range_low']
        R_unit = entry - stop
        if not (R_unit > 0):
            n_bad_R += 1
            continue
        exit_price = entry * (1 + r.pnl_pct / 100.0)
        exit_i = orb_reconstruct_exit(bars, i0, exit_price, r.exit_reason)
        if exit_i is None:
            n_no_fill += 1
            continue
        own_net_R = (exit_price - entry) / R_unit
        mfe_R_whole = (bars['h'][i0 + 1:exit_i + 1].max() - entry) / R_unit if exit_i > i0 else np.nan
        atr14 = np.nan  # not available for ORB; range_atr features degrade to NaN (disclosed)
        mean_vol = bars['v'][:i0 + 1].mean() if i0 >= 1 else np.nan
        for k in DECISION_KS:
            w = f1668.walk_k(bars, i0, k, stop, entry + 999 * R_unit)  # ORB has no fixed target; never hit
            if w is None:
                continue
            open_k = (i0 + k) < exit_i
            pf = f1670.path_features(bars, i0, i0 + k, entry, R_unit, np.nan)
            shp = build_shape_at_k(bars, i0, k, entry, R_unit, atr14, mean_vol)
            next_open = w['next_open']
            gross = ((next_open - entry) / R_unit) if next_open is not None else np.nan
            cost = (f1668.ENTRY_BPS * entry + f1668.CUT_BPS * next_open) / R_unit if next_open is not None else np.nan
            dR_cut_vs_own = (gross - cost - own_net_R) if next_open is not None else np.nan
            row = dict(fill_id=f'{r.symbol}_{r.date}', date=r.date, symbol=r.symbol, k=k,
                       open_k=open_k, mtm_R=pf['mtm_R'], mfe_R=pf['mfe_R'], mae_R=pf['mae_R'],
                       min_since_new_high=pf['min_since_new_high'], bars_since_higher_low=pf['bars_since_higher_low'],
                       level_retouched=pf['level_retouched'], own_net_R=own_net_R, dR_full=dR_cut_vs_own,
                       cost_R=(f1668.CUT_BPS * next_open / R_unit) if next_open is not None else np.nan)
            row.update(shp)
            rows.append(row)
        n_ok += 1
        if (i + 1) % 100 == 0:
            logger.info('ORB sweep %d/%d (ok=%d)', i + 1, len(ent), n_ok)
    store.close()
    orb_perfillk = pd.DataFrame(rows)
    orb_perfillk.to_csv(ORB_PERFILL_CSV, index=False)
    n_total = len(ent)
    logger.info('ORB coverage: %d/%d entered fills usable (%.1f%%) | no_bars=%d no_range/fill=%d bad_R=%d no_exit=%d',
                n_ok, n_total, 100.0 * n_ok / max(n_total, 1), n_no_bars, n_no_range, n_bad_R, n_no_fill)

    # give-back anatomy on the ORB book's OWN exit
    own = orb_perfillk.drop_duplicates('fill_id')[['fill_id', 'date', 'own_net_R']].copy()
    mfe_by_fill = orb_perfillk.groupby('fill_id')['mfe_R'].max()
    own = own.set_index('fill_id')
    own['mfe_R'] = mfe_by_fill
    ever_1r = own['mfe_R'] >= 1.0
    closed_le0 = own['own_net_R'] <= 0.0
    giveback_share = (ever_1r & closed_le0).mean()
    r_given_back = (own.loc[ever_1r, 'mfe_R'] - own.loc[ever_1r, 'own_net_R']).mean()

    # apply Part2 HOD-trained models UNCHANGED, both halves' models, ORB's own R units
    reads = []
    for featset in ('path', 'path_shape'):
        for scoring_dir in ('TRAIN-H2->VAL', 'VAL->TRAIN-H2'):
            model_by_k = {}
            for k in DECISION_KS:
                r_ = pred_store.get((featset, 'remaining_R', k, scoring_dir))
                if r_ is not None and r_.get('model') is not None:
                    model_by_k[k] = r_['model']
            if not model_by_k:
                logger.warning('Part3: no HOD models available for featset=%s scoring=%s (run --stage hod first)', featset, scoring_dir)
                continue
            piv = {}
            for field in ('open_k', 'mtm_R', 'dR_full', 'cost_R'):
                piv[field] = orb_perfillk.pivot_table(index='fill_id', columns='k', values=field, aggfunc='first')
            pred_wide = {}
            for k in DECISION_KS:
                if k not in model_by_k:
                    continue
                sub = orb_perfillk[orb_perfillk.k == k].set_index('fill_id')
                cols = [c for c in featset_cols(featset) if c in sub.columns]
                X = sub[cols].apply(pd.to_numeric, errors='coerce').values
                pred_wide[k] = pd.Series(model_by_k[k].predict(X), index=sub.index)
            piv['pred'] = pd.DataFrame(pred_wide)
            dates_of = orb_perfillk.drop_duplicates('fill_id').set_index('fill_id')['date']
            for c in C_GRID:
                for variant in ('X', 'X+'):
                    ever_pool, fired, realized, checkpoints = scan_rule(piv, dates_of, c, variant)
                    n_pool, n_fired = int(ever_pool.sum()), int(fired.sum())
                    dR_fired = realized[fired]
                    dates_fired = dates_of.reindex(dR_fired.index)
                    st = stats_block(dR_fired.values, dates_fired.values)
                    dec = decompose(dR_fired, piv['cost_R'][checkpoints[0]].reindex(dR_fired.index).fillna(0))
                    dollar_effect = st['mean_dR'] * ORB_LIVE_RISK if pd.notna(st['mean_dR']) else np.nan
                    reads.append(dict(part='3', featset=featset, scoring=scoring_dir, c=c, variant=variant,
                                      n_pool=n_pool, n_fired=n_fired, share_fired=(n_fired / n_pool if n_pool else np.nan),
                                      **st, **dec, dollar_effect_at_375=dollar_effect))
    reads_df = pd.DataFrame(reads)
    reads_df.to_csv(os.path.join(HERE, '1678_orb_reads.csv'), index=False)
    with open(os.path.join(HERE, '1678_orb_summary.txt'), 'w') as f:
        f.write(f'n_entered={n_total} n_usable={n_ok} coverage={100.0*n_ok/max(n_total,1):.1f}% '
                f'no_bars={n_no_bars} no_range_or_fill={n_no_range} bad_R={n_bad_R} no_exit={n_no_fill}\n')
        f.write(f'giveback: share_ever>=1R_closed<=0 = {giveback_share:.3f} (n={int(ever_1r.sum())}), '
                f'mean_R_given_back = {r_given_back:.3f}\n')
    logger.info('Part3 done: %d reads', len(reads_df))
    return orb_perfillk, reads_df, dict(n_total=n_total, n_ok=n_ok, giveback_share=giveback_share, r_given_back=r_given_back)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

POP_BY_FILL = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['hod', 'orb', 'report'], default='hod')
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    setup_logging()
    f1668.check_disk(5.0)
    global POP_BY_FILL

    if a.stage == 'hod':
        pop = f1668.load_population()
        pop['fill_id'] = pop['fill_id'].astype(str)
        POP_BY_FILL = pop.set_index('fill_id')
        store = f1668.BarStore(f1668.BARS_DB)
        perfillk = sweep(pop, store, resume=a.resume)
        store.close()
        perfillk.to_csv(PERFILLK_CSV, index=False)
        decomp_df, pred_store = run_part1(perfillk)
        decomp_df.to_csv(DECOMP_CSV, index=False)
        reads2 = run_part2(perfillk, pred_store)
        reads4 = run_part4(perfillk, pred_store)
        reads_all = pd.concat([reads2, reads4], ignore_index=True, sort=False)
        reads_all.to_csv(READS_CSV, index=False)
        logger.info('HOD stage complete: %d decomp rows, %d reads rows', len(decomp_df), len(reads_all))
    elif a.stage == 'orb':
        # rebuild pred_store by re-fitting is wasteful; instead reload models from disk.
        perfillk = pd.read_csv(PERFILLK_CSV, dtype={'fill_id': str})
        pred_store = {}
        for fs in ('path', 'path_shape'):
            for k in K_GRID:
                for scoring_dir in ('TRAIN-H2->VAL', 'VAL->TRAIN-H2'):
                    tag = scoring_dir.replace('->', '_to_').replace('-', '')
                    fp = os.path.join(MODELS_DIR, f'1678_{fs}_remaining_R_k{k}_{tag}.joblib')
                    if os.path.exists(fp):
                        pred_store[(fs, 'remaining_R', k, scoring_dir)] = dict(model=joblib.load(fp))
        run_part3(pred_store)
    else:
        logger.info('report stage: see write_result_md invocation in a separate step')
    logger.info('main() done for stage=%s', a.stage)


if __name__ == '__main__':
    main()
