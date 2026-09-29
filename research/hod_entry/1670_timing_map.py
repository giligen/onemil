#!/usr/bin/env python3
"""Cell 1,670: the feature-timing map -- where the information about a
HOD-break trade's fate lives, by minute (k) and by feature family.

PREREG: research/hod_entry/PREREG_1670.md (FROZEN 2026-09-29 19:58 UTC),
including amendment 1 (R4, the model-gated ADD) and amendment 2 (R5, the ADD
after +R with the locked stop).

Owner ask (2026-09-29): "maybe you also trained the model on the wrong
features, e.g. features during entry vs features 5 min in or 10 min in, or
maybe 60 min in; feature selection is an art."

Reuses (imported via importlib, not copied) from 1668_failure.py: BarStore,
find_fill_index, find_break_bar, spy_close_at_or_before, walk_k, dR_cut,
shape_features, talib_features, load_population, iid_t, day_clustered_t,
ex_top5_mean, mde, minute_of_day, et_offset_minutes, CDL_NAMES,
TALIB_AVAILABLE, EOD_M, LEVEL_TOL, ENTRY_BPS, CUT_BPS; and from
1669_fast_failure.py: load_population (adds F11-F15, minutes_since_open),
walk_to_exit, OPEN_M. New in this cell: bars through k=30/60, the SPY
channel, the open-at-k population/label logic generalised to any k, five
feature families (A/P/V/M/S) with an explicit formula per feature, the R4
ADD-when-likely-to-win rule and R5 the ADD-after-+R pyramid (with and without
a model gate).

Usage:
    python3 1670_timing_map.py [--resume] [--limit N]

Outputs (research/hod_entry/):
    1670_map.csv         -- R1: family x k x direction AUC (+ALL's placebo)
    1670_reads.csv       -- R2 (money/cut), R3 (importances), R4 (ADD), R5
                             (ADD-after-+R), one 'part' column distinguishes
    1670_per_fill_k.csv  -- fill_id x k long table: open flag, labels, mtm R,
                             P(stop) per family
    1670_wide_cache.csv  -- internal --resume checkpoint (one row per fill)
    1670_timing_map.log
    RESULT_1670.md
"""
import argparse
import importlib.util
import logging
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


f1668 = _load('f1668_1670', '1668_failure.py')
f1669 = _load('f1669_1670', '1669_fast_failure.py')

F1667_CSV = os.path.join(HERE, '1667_features.csv')
WIDE_CACHE = os.path.join(HERE, '1670_wide_cache.csv')
MAP_CSV = os.path.join(HERE, '1670_map.csv')
READS_CSV = os.path.join(HERE, '1670_reads.csv')
PERFILLK_CSV = os.path.join(HERE, '1670_per_fill_k.csv')
LOG_FILE = os.path.join(HERE, '1670_timing_map.log')
RESULT_MD = os.path.join(HERE, 'RESULT_1670.md')

HALVES = ['TRAIN-H2', 'VAL']
KS = [0, 1, 2, 5, 10, 30, 60]
TAUS = [0.5, 0.6, 0.7, 0.8]
FAMILIES = ['A', 'P', 'V', 'M', 'S', 'ALL']
R5_R = [1.0, 1.5]
R5_LOCKS = ['breakeven', 'plus0.5R']
R5_TARGETMULTS = [2, 3]
# Exit-side cost schedule (PREREG population section): entry 7bps (both base
# and any add unit), stop 6bps, target 0bps, EOD 11bps, early cut 6bps.
EXIT_COST_BPS = {'stop': 0.0006, 'target': 0.0, 'eod': 0.0011}
RNG_SEED = 1670
DIR_TO_SCORED_HALF = {'TRAIN->VAL': 'VAL', 'VAL->TRAIN-H2 (swap)': 'TRAIN-H2'}
DIR_FOR_HALF = {'VAL': 'TRAIN->VAL', 'TRAIN-H2': 'VAL->TRAIN-H2 (swap)'}

logger = logging.getLogger('1670')


def setup_logging():
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(levelname)s %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)
    for m in (f1668, f1669):
        m.logger.handlers = [fh, sh]
        m.logger.setLevel(logging.INFO)
        m.logger.propagate = False


# ---------------------------------------------------------------------------
# Population: f1669's join (1663 + fills_1658 + causal level + F11-F15) plus
# the extra 1667 columns family A and M need (gap%, 5d/20d return, ADV20,
# SPY 5-day return).
# ---------------------------------------------------------------------------

def load_population():
    pop = f1669.load_population()
    extra = pd.read_csv(F1667_CSV, dtype={'date': str, 'symbol': str},
                         usecols=['date', 'symbol', 'F1', 'F6', 'F7', 'F8', 'F10'])
    before = len(pop)
    pop = pop.merge(extra, on=['date', 'symbol'], how='left', validate='one_to_one')
    n_miss = pop[['F1', 'F6', 'F7', 'F8', 'F10']].isna().any(axis=1).sum()
    logger.info('1667 F1/F6/F7/F8/F10 join: %d/%d rows (%d with >=1 missing)', before, len(pop), n_miss)
    return pop


# ---------------------------------------------------------------------------
# Feature families -- each a pure function of bars[i0 .. i0+k] (inclusive),
# matching the PREREG's 'at k=0, P/V/S use the fill bar only' rule (the
# window collapses to the single fill bar when k=0).
# ---------------------------------------------------------------------------

def path_features(bars, i0, hi_i, entry, R, level):
    """Family P: mark-to-market R, MFE/MAE in R, minutes since the last new
    high, whether the level was re-touched, bars since the last higher low."""
    h, l, c, m = bars['h'][i0:hi_i + 1], bars['l'][i0:hi_i + 1], bars['c'][i0:hi_i + 1], bars['minarr'][i0:hi_i + 1]
    mtm_R = (c[-1] - entry) / R
    mfe_R = (h.max() - entry) / R
    mae_R = (entry - l.min()) / R
    run_max, last_nh = -np.inf, 0
    for idx in range(len(h)):
        if h[idx] > run_max:
            run_max, last_nh = h[idx], idx
    min_since_new_high = float(m[-1] - m[last_nh])
    last_hl = None
    for idx in range(1, len(l)):
        if l[idx] > l[idx - 1]:
            last_hl = idx
    bars_since_higher_low = float((len(l) - 1) - last_hl) if last_hl is not None else np.nan
    level_retouched = bool((l <= level + f1668.LEVEL_TOL).any()) if pd.notna(level) else np.nan
    return dict(mtm_R=mtm_R, mfe_R=mfe_R, mae_R=mae_R, min_since_new_high=min_since_new_high,
                bars_since_higher_low=bars_since_higher_low, level_retouched=level_retouched)


def vol_features(bars, i0, hi_i, k, break_bar_v, adv20):
    """Family V: cumulative volume since fill / break-bar volume, last-3-bar
    volume / mean bar volume, dollar volume since fill / ADV20. (Progress per
    unit volume is folded in by the caller, which already has mtm_R.)"""
    have_bb = pd.notna(break_bar_v) and break_bar_v > 0
    since_lo, since_hi = (i0, i0 + 1) if k == 0 else (i0 + 1, hi_i + 1)
    v_since, c_since = bars['v'][since_lo:since_hi], bars['c'][since_lo:since_hi]
    vol_since_fill = float(v_since.sum())
    vol_since_fill_ratio = (vol_since_fill / break_bar_v) if have_bb else np.nan
    win_v = bars['v'][i0:hi_i + 1]
    last3, mean_bar_v = win_v[-3:], win_v.mean()
    last3_vol_ratio = (last3.mean() / mean_bar_v) if mean_bar_v > 0 else np.nan
    dvol_since_fill = float((v_since * c_since).sum())
    dvol_since_fill_adv20 = (dvol_since_fill / adv20) if (pd.notna(adv20) and adv20 > 0) else np.nan
    return dict(vol_since_fill_ratio=vol_since_fill_ratio, last3_vol_ratio=last3_vol_ratio,
                dvol_since_fill_adv20=dvol_since_fill_adv20)


def market_features(spy_bars, spy_at_fill, spy_at_open, target_min):
    """Family M: SPY return fill->k, SPY return open->k (5-day SPY return is
    a per-fill constant, F10, added by the caller)."""
    spy_at_k = f1668.spy_close_at_or_before(spy_bars, target_min) if spy_bars is not None else None
    ret_fill = (spy_at_k / spy_at_fill - 1.0) if (spy_at_k is not None and spy_at_fill not in (None, 0)) else np.nan
    ret_open = (spy_at_k / spy_at_open - 1.0) if (spy_at_k is not None and spy_at_open not in (None, 0)) else np.nan
    return dict(spy_ret_fill=ret_fill, spy_ret_open=ret_open)


def fam_A_cols():
    return ['r_pct', 'atr14_pct', 'F1', 'F6', 'F7', 'F8', 'F11', 'F12', 'F14', 'F15', 'minutes_since_open']


def fam_P_cols(k):
    return [f'{c}_{k}' for c in ('mtm_R', 'mfe_R', 'mae_R', 'min_since_new_high',
                                  'bars_since_higher_low', 'level_retouched')]


def fam_V_cols(k):
    return [f'{c}_{k}' for c in ('vol_since_fill_ratio', 'last3_vol_ratio',
                                  'dvol_since_fill_adv20', 'progress_per_vol')]


def fam_M_cols(k):
    return [f'spy_ret_fill_{k}', f'spy_ret_open_{k}', 'spy_ret5']


def fam_S_cols(per_fill, k):
    prefixes = ('clv_last_', 'wick_last_', 'body_last_', 'clv_mean_', 'red_share_',
                'talib_nbull_', 'talib_nbear_', 'cdl_')
    suf = f'_{k}'
    return [c for c in per_fill.columns if c.endswith(suf) and c.startswith(prefixes)]


def fam_cols(per_fill, family, k):
    if family == 'A':
        cols = fam_A_cols()
    elif family == 'P':
        cols = fam_P_cols(k)
    elif family == 'V':
        cols = fam_V_cols(k)
    elif family == 'M':
        cols = fam_M_cols(k)
    elif family == 'S':
        cols = fam_S_cols(per_fill, k)
    elif family == 'ALL':
        seen, cols = set(), []
        for c in fam_A_cols() + fam_P_cols(k) + fam_V_cols(k) + fam_M_cols(k) + fam_S_cols(per_fill, k):
            if c not in seen:
                seen.add(c)
                cols.append(c)
    else:
        raise ValueError(family)
    return [c for c in cols if c in per_fill.columns]


# ---------------------------------------------------------------------------
# Per-fill sweep: builds the wide per-fill table (one row per fill, columns
# for every k and every r-trigger).
# ---------------------------------------------------------------------------

def compute_fill_row(r, bars, spy_bars):
    entry, stop, target, level = r['entry_price'], r['stop'], r['target_price'], r['level']
    R = entry - stop
    n = len(bars['o'])
    i0 = f1668.find_fill_index(bars, r['fill_min'])
    rec = dict(fill_id=r['fill_id'], date=r['date'], symbol=r['symbol'], split=r['half'],
               entry=entry, stop=stop, target_price=target, level=level,
               r_pct=r['r_pct'], atr14_pct=r['atr14_pct'],
               F1=r['F1'], F6=r['F6'], F7=r['F7'], F8=r['F8'],
               F11=r['F11'], F12=r['F12'], F14=r['F14'], F15=r['F15'],
               minutes_since_open=r['minutes_since_open'], spy_ret5=r['F10'],
               base_exit_type=r['exit_type'], base_net_R=r['net_R'], exit_price=r['exit_price'],
               i0_found=(i0 is not None))
    if i0 is None:
        return rec
    rec['i0'] = i0
    break_idx = f1668.find_break_bar(bars, r['fill_min'], level)
    break_bar_v = bars['v'][break_idx] if break_idx is not None else np.nan
    adv20 = r['F8']
    spy_at_fill = f1668.spy_close_at_or_before(spy_bars, bars['minarr'][i0])
    spy_at_open = f1668.spy_close_at_or_before(spy_bars, f1669.OPEN_M)

    exit_idx_orig, exit_kind_check = f1669.walk_to_exit(bars, i0, stop, target)
    rec['orig_exit_idx'] = exit_idx_orig
    if exit_idx_orig is not None and exit_kind_check != r['exit_type']:
        rec['exit_mismatch'] = True

    for k in KS:
        if k == 0:
            hi_i, open_k = i0, True
        else:
            w = f1668.walk_k(bars, i0, k, stop, target)
            open_k = (w is not None) and (w['preempt'] == '')
            hi_i = (i0 + k) if (w is not None) else None
        rec[f'open_{k}'] = open_k
        if not open_k or hi_i is None:
            continue
        next_open_k = bars['o'][hi_i + 1] if (hi_i + 1) < n else None
        rec[f'label_stop_{k}'] = bool(r['exit_type'] == 'stop')
        rec[f'label_target_{k}'] = bool(r['exit_type'] == 'target')
        rec[f'next_open_{k}'] = next_open_k
        rec[f'dR_cut_{k}'] = f1668.dR_cut(entry, stop, r['net_R'], next_open_k)

        pf = path_features(bars, i0, hi_i, entry, R, level)
        for key, v in pf.items():
            rec[f'{key}_{k}'] = v
        vf = vol_features(bars, i0, hi_i, k, break_bar_v, adv20)
        for key, v in vf.items():
            rec[f'{key}_{k}'] = v
        vr = vf['vol_since_fill_ratio']
        rec[f'progress_per_vol_{k}'] = (pf['mtm_R'] / vr) if (pd.notna(vr) and vr != 0) else np.nan

        mf = market_features(spy_bars, spy_at_fill, spy_at_open, bars['minarr'][hi_i])
        for key, v in mf.items():
            rec[f'{key}_{k}'] = v

        sf = f1668.shape_features(bars, i0, k)
        for key, v in sf.items():
            rec[f'{key}_{k}'] = v
        if f1668.TALIB_AVAILABLE:
            tf, _ = f1668.talib_features(bars, i0, k, want_fired_any=False)
            for key, v in tf.items():
                rec[f'{key}_{k}'] = v

        # R4 variant A' -- add's own stop tightened to level-$0.01; reported,
        # not preferred. Scan bars[add_bar .. orig_exit_idx] for an early
        # tight-stop touch; otherwise the add shares the base's own exit.
        if next_open_k is not None and pd.notna(level) and exit_idx_orig is not None:
            tight_stop = level - 0.01
            add_bar = hi_i + 1
            exit_idx3, exit_price3, exit_kind3 = add_bar, r['exit_price'], r['exit_type']
            for j in range(add_bar, exit_idx_orig + 1):
                if bars['l'][j] <= tight_stop:
                    exit_idx3, exit_price3, exit_kind3 = j, tight_stop, 'stop'
                    break
            gross = (exit_price3 - next_open_k) / R
            cost = (f1668.ENTRY_BPS * next_open_k + EXIT_COST_BPS[exit_kind3] * exit_price3) / R
            rec[f'addR_prime_{k}'] = gross - cost

    # R5 raw walk: first bar (any minute) reaching +r R, or the original
    # resolution, whichever comes first -- independent of the k grid.
    for r_mult in R5_R:
        target_r_price = entry + r_mult * R
        j_r, kind_r = None, None
        for j in range(i0 + 1, n):
            if bars['minarr'][j] >= f1668.EOD_M:
                kind_r, j_r = 'eod', j
                break
            if bars['l'][j] <= stop:
                kind_r, j_r = 'stop', j
                break
            if bars['h'][j] >= target:
                kind_r, j_r = 'target', j
                break
            if bars['c'][j] >= target_r_price:
                kind_r, j_r = 'r_trigger', j
                break
        tag = str(r_mult).replace('.', 'p')
        rec[f'r{tag}_kind'] = kind_r
        if kind_r == 'r_trigger':
            rec[f'r{tag}_trigger_idx'] = j_r
            rec[f'r{tag}_minutes_elapsed'] = float(bars['minarr'][j_r] - bars['minarr'][i0])
    return rec


def sweep(pop, store, resume=False, limit=None):
    if limit:
        pop = pop.iloc[:limit].copy()
    if resume and os.path.exists(WIDE_CACHE):
        cached = pd.read_csv(WIDE_CACHE, dtype={'date': str, 'symbol': str})
        if len(cached) == len(pop):
            logger.info('--resume: reusing complete wide cache (%d rows)', len(cached))
            return cached
        logger.info('--resume requested but cache incomplete (%d vs %d) -- recomputing', len(cached), len(pop))

    rows = []
    n_no_bars = n_no_fill_idx = 0
    t0 = time.time()
    pop = pop.reset_index(drop=True)
    for i, r in pop.iterrows():
        bars = store.day_bars(r['symbol'], r['date'])
        if bars is None:
            n_no_bars += 1
            rows.append(dict(fill_id=r['fill_id'], date=r['date'], symbol=r['symbol'], split=r['half'],
                              i0_found=False))
            continue
        spy_bars = store.spy_bars(r['date'])
        rec = compute_fill_row(r, bars, spy_bars)
        if not rec.get('i0_found', False):
            n_no_fill_idx += 1
        rows.append(rec)
        if (i + 1) % 500 == 0 or (i + 1) == len(pop):
            elapsed = time.time() - t0
            logger.info('sweep %d/%d fills (%.1fs elapsed, %d no-bars, %d no-fill-idx)',
                        i + 1, len(pop), elapsed, n_no_bars, n_no_fill_idx)
            out = pd.DataFrame(rows)
            tmp = WIDE_CACHE + '.tmp'
            out.to_csv(tmp, index=False)
            os.replace(tmp, WIDE_CACHE)
    logger.info('sweep done: %d fills, %d no-bars, %d no-fill-idx', len(pop), n_no_bars, n_no_fill_idx)
    return pd.DataFrame(rows)


def normalize_bool_cols(df):
    """After a --resume CSV round-trip, 'True'/'False'/NaN columns can come
    back as object dtype; coerce open_*/label_*/exit_mismatch columns back to
    proper booleans (NaN preserved)."""
    for c in df.columns:
        if c.startswith('open_') or c.startswith('label_') or c == 'exit_mismatch' or c == 'i0_found':
            df[c] = df[c].map({True: True, False: False, 'True': True, 'False': False,
                                np.True_: True, np.False_: False})
    return df


# ---------------------------------------------------------------------------
# Shared classifier map (R1 and R4a share this: only the label and family
# list differ)
# ---------------------------------------------------------------------------

def build_xy_k(per_fill, half, k, family, label_col):
    open_col = f'open_{k}'
    sub = per_fill[(per_fill['split'] == half) & (per_fill[open_col] == True) &  # noqa: E712
                   per_fill[label_col].notna()].copy()
    cols = fam_cols(per_fill, family, k)
    X = sub[cols].astype(float) if cols else pd.DataFrame(index=sub.index)
    y = sub[label_col].astype(int)
    return X, y, sub, cols


def run_classifier_map(per_fill, label_fn, families, want_placebo_importance):
    """label_fn(k) -> label column name. Returns (reads_df, pstop_oos
    dict[(family,k)->Series aligned to per_fill.index], models
    dict[(k,direction)]->(model,cols) for family=='ALL' only,
    importance_tables dict[k]->DataFrame)."""
    reads = []
    pstop_oos, all_models, imp_tables = {}, {}, {}
    rng = np.random.RandomState(RNG_SEED)
    for k in KS:
        for family in families:
            label_col = label_fn(k)
            Xtr, ytr, subtr, cols = build_xy_k(per_fill, 'TRAIN-H2', k, family, label_col)
            Xva, yva, subva, _ = build_xy_k(per_fill, 'VAL', k, family, label_col)
            logger.info('%s family=%s k=%d: TRAIN-H2 n=%d (%.2f%% pos), VAL n=%d (%.2f%% pos), %d feats',
                        label_col, family, k, len(ytr), 100 * ytr.mean() if len(ytr) else np.nan,
                        len(yva), 100 * yva.mean() if len(yva) else np.nan, len(cols))
            directions = {'TRAIN->VAL': (Xtr, ytr, subtr, Xva, yva, subva),
                          'VAL->TRAIN-H2 (swap)': (Xva, yva, subva, Xtr, ytr, subtr)}
            k_pstop = pd.Series(index=per_fill.index, dtype=float)
            for direction, (Xa, ya, suba, Xb, yb, subb) in directions.items():
                if ya.nunique() < 2 or len(Xb) == 0:
                    logger.warning('%s family=%s k=%d %s: skipped (single-class train or empty score half)',
                                    label_col, family, k, direction)
                    reads.append(dict(family=family, k=k, direction=direction, n_train=len(ya),
                                       n_score=len(yb), auc=np.nan, placebo_auc=np.nan))
                    continue
                model = HistGradientBoostingClassifier(max_iter=200, random_state=RNG_SEED)
                model.fit(Xa, ya)
                p = model.predict_proba(Xb)[:, 1]
                auc = roc_auc_score(yb, p) if yb.nunique() > 1 else np.nan
                k_pstop.loc[subb.index] = p
                placebo_auc = np.nan
                if family == 'ALL':
                    all_models[(k, direction)] = (model, cols)
                    if want_placebo_importance:
                        ya_shuf = pd.Series(ya.values, index=ya.index).groupby(
                            suba['date'].values).transform(lambda s: rng.permutation(s.values))
                        if ya_shuf.nunique() > 1:
                            model_pl = HistGradientBoostingClassifier(max_iter=200, random_state=RNG_SEED)
                            model_pl.fit(Xa, ya_shuf)
                            p_pl = model_pl.predict_proba(Xb)[:, 1]
                            placebo_auc = roc_auc_score(yb, p_pl) if yb.nunique() > 1 else np.nan
                        if direction == 'TRAIN->VAL':
                            logger.info('permutation importance %s family=ALL k=%d (n_repeats=5,n_jobs=1)',
                                        label_col, k)
                            pi = permutation_importance(model, Xb, yb, n_repeats=5,
                                                         random_state=RNG_SEED, n_jobs=1)
                            imp_tables[k] = pd.DataFrame({
                                'feature': cols, 'importance_mean': pi.importances_mean,
                                'importance_std': pi.importances_std}).sort_values(
                                'importance_mean', ascending=False)
                reads.append(dict(family=family, k=k, direction=direction, n_train=len(ya),
                                   n_score=len(yb), auc=auc, placebo_auc=placebo_auc))
            pstop_oos[(family, k)] = k_pstop
        logger.info('%s map k=%d done', label_fn(k), k)
    return pd.DataFrame(reads), pstop_oos, all_models, imp_tables


# ---------------------------------------------------------------------------
# R2 -- the money table: cut at fixed tau (+ 'remaining stop >= 0.5R' gate)
# ---------------------------------------------------------------------------

def decompose_cut(sdf, tau, variant):
    """Paired dR of the cut rule + the 1,669 saved/forgone/cost
    decomposition, with the identity mean(dR|fired) == saved_contrib -
    forgone_contrib - cost_mean verified numerically (residual reported)."""
    R_unit = sdf['entry'] - sdf['stop']
    cost = f1668.CUT_BPS * sdf['next_open'] / R_unit
    fire = sdf['p'] >= tau
    if variant == 'remstop0.5':
        fire = fire & ((sdf['mtm_R'] + 1.0) >= 0.5)
    fired, cost_f = sdf[fire], cost[fire]
    n_scored, n_fired = len(sdf), len(fired)
    out = dict(n_scored=n_scored, n_fired=n_fired, share_cut=(n_fired / n_scored if n_scored else np.nan))
    if n_fired == 0:
        return out
    dR = fired['dR']
    tp_mask, fp_mask = fired['y'] == 1, fired['y'] == 0
    saved_vals = dR[tp_mask] + cost_f[tp_mask]
    forgone_vals = -(dR[fp_mask] + cost_f[fp_mask])
    saved_mean = saved_vals.mean() if tp_mask.any() else np.nan
    forgone_mean = forgone_vals.mean() if fp_mask.any() else np.nan
    cost_mean = cost_f.mean()
    share_tp = float(tp_mask.mean())
    saved_contrib = share_tp * saved_mean if pd.notna(saved_mean) else 0.0
    forgone_contrib = (1 - share_tp) * forgone_mean if pd.notna(forgone_mean) else 0.0
    identity_rhs = saved_contrib - forgone_contrib - cost_mean
    if pd.notna(saved_mean) and pd.notna(forgone_mean):
        denom = saved_mean + forgone_mean + cost_mean
        breakeven_prec = ((forgone_mean + cost_mean) / denom) if denom else np.nan
    else:
        breakeven_prec = np.nan
    out.update(mean_dR=dR.mean(), iid_t=f1668.iid_t(dR), day_t=f1668.day_clustered_t(fired['date'], dR),
               ex_top5_dR=f1668.ex_top5_mean(dR), mde=f1668.mde(dR.std(ddof=1), n_fired),
               achieved_precision=share_tp, saved_mean_TP=saved_mean, forgone_mean_FP=forgone_mean,
               cost_mean=cost_mean, breakeven_precision=breakeven_prec,
               identity_residual=dR.mean() - identity_rhs)
    return out


def run_r2_money(per_fill, pstop_oos):
    reads = []
    for k in KS:
        p_all = pstop_oos[('ALL', k)]
        base = per_fill[per_fill[f'open_{k}'] == True].copy()  # noqa: E712
        base['p'] = p_all.reindex(base.index)
        base = base.dropna(subset=['p'])
        base['dR'] = base[f'dR_cut_{k}']
        base['mtm_R'] = base[f'mtm_R_{k}']
        base['y'] = base[f'label_stop_{k}'].astype(int)
        base['next_open'] = base[f'next_open_{k}']
        for direction, half in DIR_TO_SCORED_HALF.items():
            sdf = base[base['split'] == half][['date', 'entry', 'stop', 'next_open', 'p', 'y', 'dR', 'mtm_R']]
            for tau in TAUS:
                for variant in ('base', 'remstop0.5'):
                    res = decompose_cut(sdf, tau, variant)
                    res.update(part='R2', k=k, direction=direction, tau=tau, variant=variant)
                    reads.append(res)
    return pd.DataFrame(reads)


# ---------------------------------------------------------------------------
# R4 -- the ADD when P(target) >= tau
# ---------------------------------------------------------------------------

def run_r4_add(per_fill, ptarget_oos):
    reads = []
    R_unit = per_fill['entry'] - per_fill['stop']
    exit_price, exit_type = per_fill['exit_price'], per_fill['base_exit_type']
    cost_exit = exit_type.map(EXIT_COST_BPS)
    for k in KS:
        add_entry = per_fill[f'next_open_{k}']
        gross = (exit_price - add_entry) / R_unit
        cost = (f1668.ENTRY_BPS * add_entry + cost_exit * exit_price) / R_unit
        addR = {'A': gross - cost, "A'": per_fill.get(f'addR_prime_{k}', pd.Series(np.nan, index=per_fill.index))}
        p_t = ptarget_oos[('ALL', k)]
        base = per_fill[per_fill[f'open_{k}'] == True].copy()  # noqa: E712
        base['p'] = p_t.reindex(base.index)
        for variant in ('A', "A'"):
            base[f'addR_{variant}'] = addR[variant].reindex(base.index)
        base = base.dropna(subset=['p'])
        for direction, half in DIR_TO_SCORED_HALF.items():
            sub = base[base['split'] == half]
            for tau in TAUS:
                fire = sub['p'] >= tau
                for variant in ('A', "A'"):
                    addR_v = sub[f'addR_{variant}']
                    valid = fire & addR_v.notna()
                    dR = np.where(valid, addR_v, 0.0)
                    n = len(sub)
                    fired_addR = addR_v[valid]
                    worst_day = (pd.DataFrame({'date': sub['date'], 'dR': dR}).groupby('date')['dR']
                                 .mean().min()) if n else np.nan
                    reads.append(dict(part='R4', variant=variant, k=k, direction=direction, tau=tau, n=n,
                                       share_added=float(valid.mean()) if n else np.nan,
                                       mean_dR=dR.mean() if n else np.nan, iid_t=f1668.iid_t(dR),
                                       day_t=f1668.day_clustered_t(sub['date'], dR),
                                       ex_top5_dR=f1668.ex_top5_mean(dR),
                                       mde=f1668.mde(pd.Series(dR).std(ddof=1), n) if n else np.nan,
                                       add_own_R=fired_addR.mean() if len(fired_addR) else np.nan,
                                       worst_day_R=worst_day, dollar_exposure='2x base'))
    return pd.DataFrame(reads)


# ---------------------------------------------------------------------------
# R5 -- the ADD after +r R, with the whole position's stop locked
# ---------------------------------------------------------------------------

def walk_from(bars, start_idx, stop, target):
    """Bar-by-bar precedence walk from start_idx (inclusive) to EOD; stop
    before target within a bar (same convention as f1669.walk_to_exit, just
    starting mid-day instead of at i0+1)."""
    n = len(bars['o'])
    for j in range(start_idx, n):
        if bars['minarr'][j] >= f1668.EOD_M:
            return j, 'eod'
        if bars['l'][j] <= stop:
            return j, 'stop'
        if bars['h'][j] >= target:
            return j, 'target'
    return None, None


def r5_outcome(bars, row, r_mult, lock, targetmult):
    tag = str(r_mult).replace('.', 'p')
    if row.get(f'r{tag}_kind') != 'r_trigger':
        return 0.0, False, False
    j_r = int(row[f'r{tag}_trigger_idx'])
    entry, stop = row['entry'], row['stop']
    R = entry - stop
    n = len(bars['o'])
    if j_r + 1 >= n:
        return 0.0, False, False
    add_entry = bars['o'][j_r + 1]
    new_stop = entry if lock == 'breakeven' else entry + 0.5 * R
    new_target = entry + targetmult * R
    exit_idx2, exit_kind2 = walk_from(bars, j_r + 1, new_stop, new_target)
    if exit_idx2 is None:
        exit_idx2, exit_kind2 = n - 1, 'eod'
    exit_price2 = {'stop': new_stop, 'target': new_target, 'eod': bars['c'][exit_idx2]}[exit_kind2]
    base_R_new = (exit_price2 - entry) / R - (f1668.ENTRY_BPS * entry + EXIT_COST_BPS[exit_kind2] * exit_price2) / R
    add_R = (exit_price2 - add_entry) / R - (f1668.ENTRY_BPS * add_entry + EXIT_COST_BPS[exit_kind2] * exit_price2) / R
    dR = (base_R_new + add_R) - row['base_net_R']
    return dR, True, (exit_kind2 == 'stop')


def compute_model_gate(per_fill, r_mult, all_models_target):
    """P(T_k*)>=0.6 gate for fills that reached +r_mult R, k* = nearest k<=
    minutes-elapsed-at-trigger, model scored out of sample (batched by k*)."""
    tag = str(r_mult).replace('.', 'p')
    reached = per_fill[f'r{tag}_kind'] == 'r_trigger'
    gate = pd.Series(False, index=per_fill.index)
    # DIR_FOR_HALF maps half -> the direction whose model was TRAINED ON THE
    # OTHER HALF (out-of-sample for this half); see module constant.
    for half in HALVES:
        model_direction = DIR_FOR_HALF[half]
        sub = per_fill[reached & (per_fill['split'] == half)]
        if sub.empty:
            continue
        me = sub[f'r{tag}_minutes_elapsed']
        kstars = me.apply(lambda m: max([k for k in KS if k <= m], default=KS[0]))
        for kstar, idxs in sub.groupby(kstars).groups.items():
            key = (kstar, model_direction)
            if key not in all_models_target:
                continue
            model, cols = all_models_target[key]
            if not cols:
                continue
            X = per_fill.loc[idxs, cols].astype(float)
            ok = ~X.isna().any(axis=1)
            if ok.any():
                p = model.predict_proba(X[ok])[:, 1]
                gate.loc[X[ok].index] = (p >= 0.6)
    return gate


def run_r5(per_fill, store, all_models_target):
    bars_cache = {}

    def get_bars(symbol, date):
        key = (symbol, date)
        if key not in bars_cache:
            bars_cache[key] = store.day_bars(symbol, date)
        return bars_cache[key]

    combos = [(r, lock, tm) for r in R5_R for lock in R5_LOCKS for tm in R5_TARGETMULTS]
    outcome = {c: {} for c in combos}  # c -> {idx: (dR, reached, gave_back)}
    n_done = 0
    for idx, row in per_fill.iterrows():
        if not row.get('i0_found', False):
            continue
        bars = get_bars(row['symbol'], row['date'])
        if bars is None:
            continue
        for c in combos:
            outcome[c][idx] = r5_outcome(bars, row, *c)
        n_done += 1
        if n_done % 500 == 0:
            logger.info('R5 walk %d/%d fills', n_done, len(per_fill))
    logger.info('R5 walk done: %d fills', n_done)

    gates = {r_mult: compute_model_gate(per_fill, r_mult, all_models_target) for r_mult in R5_R}

    reads = []
    for r_mult, lock, tm in combos:
        d = pd.DataFrame.from_dict(outcome[(r_mult, lock, tm)], orient='index',
                                    columns=['dR', 'reached', 'gave_back'])
        for model_gate in (False, True):
            for direction, half in DIR_TO_SCORED_HALF.items():
                sub_idx = d.index[per_fill.loc[d.index, 'split'] == half]
                dR = d.loc[sub_idx, 'dR'].copy()
                reached = d.loc[sub_idx, 'reached']
                gave_back = d.loc[sub_idx, 'gave_back']
                if model_gate:
                    dR = dR.where(gates[r_mult].reindex(sub_idx, fill_value=False), 0.0)
                n = len(dR)
                fired_mask = (dR != 0.0) & reached if model_gate else reached
                reads.append(dict(
                    part='R5', r=r_mult, lock=lock, targetmult=tm, model_gate=model_gate,
                    direction=direction, n=n, share_reached=float(reached.mean()) if n else np.nan,
                    share_fired=float(fired_mask.mean()) if n else np.nan,
                    mean_dR=dR.mean() if n else np.nan, iid_t=f1668.iid_t(dR),
                    day_t=f1668.day_clustered_t(per_fill.loc[sub_idx, 'date'], dR),
                    ex_top5_dR=f1668.ex_top5_mean(dR), mde=f1668.mde(dR.std(ddof=1), n) if n else np.nan,
                    add_own_R=(d.loc[sub_idx][fired_mask]['dR'].mean() if fired_mask.any() else np.nan),
                    give_back_share=(gave_back[fired_mask].mean() if fired_mask.any() else np.nan),
                    worst_day_R=(pd.DataFrame({'date': per_fill.loc[sub_idx, 'date'], 'dR': dR})
                                 .groupby('date')['dR'].mean().min() if n else np.nan),
                    exposure='2x base'))
    return pd.DataFrame(reads)


# ---------------------------------------------------------------------------
# 1670_per_fill_k.csv -- long format
# ---------------------------------------------------------------------------

def build_per_fill_k(per_fill, pstop_oos):
    rows = []
    for k in KS:
        rows_k = pd.DataFrame({
            'fill_id': per_fill['fill_id'], 'k': k, 'open': per_fill[f'open_{k}'],
            'label_stop': per_fill.get(f'label_stop_{k}'), 'mtm_R': per_fill.get(f'mtm_R_{k}'),
        })
        for family in FAMILIES:
            rows_k[f'p_stop_{family}'] = pstop_oos[(family, k)].reindex(per_fill.index).values
        rows.append(rows_k)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# RESULT.md
# ---------------------------------------------------------------------------

def fmt(x, nd=3):
    return 'nan' if pd.isna(x) else f'{x:.{nd}f}'


def write_result_md(per_fill, map_df, tmap_df, reads_r2, reads_r4, reads_r5, imp_tables, n_pop, m_coverage):
    L = ['# RESULT 1,670 -- feature-timing map: where the information lives\n']
    L.append(f'PREREG: `research/hod_entry/PREREG_1670.md` (FROZEN, incl. amendments 1-2). '
             f'Population n={n_pop} (1663 primary book, r_pct>=1.5%, 5,506 rows). '
             f'SPY (family M) coverage by k: ' +
             ', '.join(f'k{k}={m_coverage.get(k, float("nan")):.0%}' for k in KS) + '.\n')

    L.append('## R1 -- AUC map (family x k), out-of-sample, label = stop-out after k\n')
    for direction in DIR_TO_SCORED_HALF:
        L.append(f'**{direction}**\n')
        piv = map_df[map_df['direction'] == direction].pivot(index='k', columns='family', values='auc')
        piv = piv.reindex(columns=FAMILIES)
        L.append('| k | ' + ' | '.join(FAMILIES) + ' | n_open |')
        L.append('|---|' + '---|' * len(FAMILIES) + '---|')
        n_by_k = map_df[(map_df['direction'] == direction) & (map_df['family'] == 'ALL')].set_index('k')['n_score']
        for k in KS:
            row = piv.loc[k] if k in piv.index else pd.Series({f: np.nan for f in FAMILIES})
            L.append(f'| {k} | ' + ' | '.join(fmt(row.get(f)) for f in FAMILIES) + f' | {int(n_by_k.get(k, 0))} |')
        L.append('')
    placebo = map_df[map_df['family'] == 'ALL'][['k', 'direction', 'placebo_auc']]
    L.append('Placebo AUC (family=ALL, within-day label shuffle): ' +
              '; '.join(f'k{r.k}/{r.direction}={fmt(r.placebo_auc)}' for r in placebo.itertuples()) + '\n')

    L.append('## R2 -- money table (family=ALL, cut at fixed tau)\n')
    L.append('Best (k, tau, variant) by mean paired dR per direction (full 112-row table: `1670_reads.csv`, part=R2):\n')
    L.append('| direction | k | tau | variant | n_fired | mean_dR | iid_t | day_t | ex_top5_dR | mde | '
             'achieved_prec | breakeven_prec |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for direction in DIR_TO_SCORED_HALF:
        sub = reads_r2[(reads_r2['direction'] == direction) & (reads_r2['n_fired'] > 0)]
        if sub.empty:
            continue
        best = sub.loc[sub['mean_dR'].idxmax()]
        L.append(f"| {direction} | {best.k} | {best.tau} | {best.variant} | {int(best.n_fired)} | "
                  f"{fmt(best.mean_dR)} | {fmt(best.iid_t)} | {fmt(best.day_t)} | {fmt(best.ex_top5_dR)} | "
                  f"{fmt(best.mde)} | {fmt(best.achieved_precision)} | {fmt(best.breakeven_precision)} |")
    L.append('')

    L.append('## R3 -- permutation importance, top 10, family=ALL, best-AUC k per scoring\n')
    if not map_df.empty:
        best_k_all = map_df[(map_df['family'] == 'ALL') & (map_df['direction'] == 'TRAIN->VAL')]
        best_k = int(best_k_all.loc[best_k_all['auc'].idxmax(), 'k']) if best_k_all['auc'].notna().any() else KS[0]
        L.append(f'Best-AUC k (TRAIN->VAL direction) = {best_k}\n')
        if best_k in imp_tables:
            top = imp_tables[best_k].head(10)
            L.append('| feature | importance_mean | importance_std |')
            L.append('|---|---|---|')
            for row in top.itertuples():
                L.append(f'| {row.feature} | {fmt(row.importance_mean, 4)} | {fmt(row.importance_std, 4)} |')
    L.append('')

    L.append('## R4 -- the ADD when P(target after k) >= tau\n')
    L.append('| variant | direction | k | tau | share_added | mean_dR | iid_t | day_t | ex_top5_dR | mde | add_own_R |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for variant in ('A', "A'"):
        for direction in DIR_TO_SCORED_HALF:
            sub = reads_r4[(reads_r4['variant'] == variant) & (reads_r4['direction'] == direction) & (reads_r4['n'] > 0)]
            if sub.empty:
                continue
            best = sub.loc[sub['mean_dR'].idxmax()]
            L.append(f"| {variant} | {direction} | {best.k} | {best.tau} | {fmt(best.share_added)} | "
                      f"{fmt(best.mean_dR)} | {fmt(best.iid_t)} | {fmt(best.day_t)} | {fmt(best.ex_top5_dR)} | "
                      f"{fmt(best.mde)} | {fmt(best.add_own_R)} |")
    L.append('')

    L.append('## R5 -- the ADD after +r R, whole-position stop locked (32 paired reads, part=R5 in `1670_reads.csv`)\n')
    L.append('| r | lock | target | model_gate | direction | share_reached | share_fired | mean_dR | iid_t | day_t | '
             'ex_top5_dR | mde | give_back | worst_day_R |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for row in reads_r5.itertuples():
        L.append(f"| {row.r} | {row.lock} | {row.targetmult}R | {row.model_gate} | {row.direction} | "
                  f"{fmt(row.share_reached)} | {fmt(row.share_fired)} | {fmt(row.mean_dR)} | {fmt(row.iid_t)} | "
                  f"{fmt(row.day_t)} | {fmt(row.ex_top5_dR)} | {fmt(row.mde)} | {fmt(row.give_back_share)} | "
                  f"{fmt(row.worst_day_R)} |")
    L.append('')

    L.append('## Verdicts vs the pass bar\n')
    L.append('Pass bar (PREREG): paired dR>=+0.05R, t>=2.5, ex-top-5%>0 on BOTH scorings, placebo AUC<=0.55.\n')

    def passes(sub_df, dr_col='mean_dR', t_col='iid_t', ex_col='ex_top5_dR'):
        by_dir = {}
        for direction in DIR_TO_SCORED_HALF:
            s = sub_df[sub_df['direction'] == direction]
            by_dir[direction] = s
        return by_dir

    r2_pass = reads_r2[(reads_r2['mean_dR'] >= 0.05) & (reads_r2['day_t'] >= 2.5) & (reads_r2['ex_top5_dR'] > 0)]
    r4_pass = reads_r4[(reads_r4['mean_dR'] >= 0.05) & (reads_r4['day_t'] >= 2.5) & (reads_r4['ex_top5_dR'] > 0)]
    r5_pass = reads_r5[(reads_r5['mean_dR'] >= 0.05) & (reads_r5['day_t'] >= 2.5) & (reads_r5['ex_top5_dR'] > 0)]
    L.append(f'R2 (cut) reads clearing dR/day-t/ex-top5% individually (both scorings NOT yet cross-checked '
              f'per-cell): {len(r2_pass)}/{len(reads_r2)}. R4 (ADD): {len(r4_pass)}/{len(reads_r4)}. '
              f'R5 (ADD-after-+R): {len(r5_pass)}/{len(reads_r5)}.\n')
    L.append('A pass on one scoring only is not a pass -- see `1670_reads.csv` for both-scoring cross-checks '
              'before any cell is proposed for paper.\n')

    L.append('## Adequacy\n')
    L.append('Not allowed items honored: tau/k not fit (grid only), no post-map feature/family changes, no '
              'exit-type feature in any family, pooled-only numbers avoided (both scorings shown throughout), '
              'training-half dR never reported as a result. R4 variant A\' and R5 model-gate reuse the ALL-family '
              'T_k model at k*=nearest k<=minutes-elapsed, scored out of sample, per the PREREG\'s explicit rule.\n')

    with open(RESULT_MD, 'w') as f:
        f.write('\n'.join(L))
    logger.info('wrote %s (%d lines)', RESULT_MD, len(L))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    setup_logging()
    f1668.check_disk(min_gb=5.0)
    logger.info('=== cell 1,670: feature-timing map ===')

    pop = load_population()
    if args.limit:
        pop = pop.iloc[:args.limit].copy()
        logger.info('--limit %d applied', args.limit)

    store = f1668.BarStore(f1668.BARS_DB)
    per_fill = sweep(pop, store, resume=args.resume, limit=args.limit)
    per_fill = normalize_bool_cols(per_fill)
    n_pop = len(per_fill)
    logger.info('wide per-fill table: %d rows, %d columns', per_fill.shape[0], per_fill.shape[1])

    m_coverage = {}
    for k in KS:
        openk = per_fill[per_fill[f'open_{k}'] == True]  # noqa: E712
        col = f'spy_ret_fill_{k}'
        cov = openk[col].notna().mean() if (col in openk.columns and len(openk)) else np.nan
        m_coverage[k] = cov
        if pd.notna(cov) and cov < 0.8:
            logger.warning('family M coverage at k=%d is %.1f%% (<80%% rail) -- VOID for that k', k, 100 * cov)

    logger.info('--- R1: AUC map (label=stop-out after k) ---')
    map_df, pstop_oos, all_models_stop, imp_tables = run_classifier_map(
        per_fill, lambda k: f'label_stop_{k}', FAMILIES, want_placebo_importance=True)
    map_df.to_csv(MAP_CSV, index=False)
    logger.info('wrote %s (%d rows)', MAP_CSV, len(map_df))

    logger.info('--- R4a: AUC map (label=target after k, family=ALL only) ---')
    tmap_df, ptarget_oos, all_models_target, _ = run_classifier_map(
        per_fill, lambda k: f'label_target_{k}', ['ALL'], want_placebo_importance=True)

    logger.info('--- R2: money table (cuts) ---')
    reads_r2 = run_r2_money(per_fill, pstop_oos)

    logger.info('--- R4b: the ADD when P(target) >= tau ---')
    reads_r4 = run_r4_add(per_fill, ptarget_oos)

    logger.info('--- R5: the ADD after +r R ---')
    reads_r5 = run_r5(per_fill, store, all_models_target)
    store.close()

    reads_r2['part'] = 'R2'
    reads_r4['part'] = 'R4'
    reads_r5['part'] = 'R5'
    tmap_df['part'] = 'R4a_auc'
    all_reads = pd.concat([reads_r2, tmap_df, reads_r4, reads_r5], ignore_index=True, sort=False)
    all_reads.to_csv(READS_CSV, index=False)
    logger.info('wrote %s (%d rows)', READS_CSV, len(all_reads))

    logger.info('--- building 1670_per_fill_k.csv ---')
    per_fill_k = build_per_fill_k(per_fill, pstop_oos)
    per_fill_k.to_csv(PERFILLK_CSV, index=False)
    logger.info('wrote %s (%d rows)', PERFILLK_CSV, len(per_fill_k))

    write_result_md(per_fill, map_df, tmap_df, reads_r2, reads_r4, reads_r5, imp_tables, n_pop, m_coverage)
    logger.info('=== done ===')


if __name__ == '__main__':
    main()
