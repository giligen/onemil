#!/usr/bin/env python3
"""Cell 1,677 -- the model-gated take-profit.

PREREG: research/hod_entry/PREREG_1677.md (FROZEN 2026-09-30 07:40 UTC).
Question: 1,676 (leak fixed) found a real post-entry shape signal for
short-horizon success (P(+1R within next 15 min), OOS AUC 0.73-0.87) but
cut/short/add keyed on it do not pay -- a NEW unit bought with the original
stop has the wrong geometry. This cell asks the one action that matches a
success predictor: the EXIT decision of the position ALREADY HELD. When the
model says the next 15 minutes are unlikely to add +1R and the trade is in
profit, take the profit instead of giving it back.

Reuses:
  f1668 (1668_failure.py, imported via importlib): BarStore, find_fill_index,
    walk_k, dR_cut, load_population, BARS_DB (opened ?mode=ro), EOD_M,
    ENTRY_BPS, CUT_BPS. walk_k(bars, i0, k, stop, target) is this line's
    shared path walker (also used by 1669/1670/1676); dR_cut(entry, stop,
    base_net_R, next_open) is the shared "sell at the open of bar i0+k+1,
    6bps marketable cost" primitive -- reused VERBATIM (unmodified import),
    the exact FULL take-profit exit mechanic PREREG specifies.
  f1676 (1676_shapes.py, imported via importlib): day_clustered_t, iid_t,
    ex_top5_mean, fills_per_week, mde, FEATURES_CSV, K1670_CSV -- the shared
    day-clustered stat kit, reused verbatim.
  gfix (1676_g6g7_fix.py, imported via importlib): build_group_cols_fix --
    the EXACT feature-group definition ("G7noclose_k{k}+ALL" = G7 shape
    features at k with every *_close_R column dropped, plus the causal-
    nearest 1,670 ALL-family probability column) that RESULT_1676.md's
    "G7-without-close_R + ALL" reads used.

Persistence gap (checked, then closed per PREREG's own contingency clause):
1,676's pred_store (out-of-sample predicted probabilities) was an in-memory
dict in both 1676_shapes.py and 1676_g6g7_fix.py -- grepped both files for
`joblib.dump` and any per-fill score CSV: neither exists anywhere in
research/hod_entry/. Per PREREG_1677's instruction for exactly this case,
fit_and_persist() below re-runs f1676.fit_both_scorings' IDENTICAL fit block
(same HistGradientBoostingClassifier(max_iter=200, random_state=1676), same
TRAIN-H2<->VAL halves, same feature columns from build_group_cols_fix) ONCE
for the 10 (k, scoring) cells this cell needs -- 5 k in {5,10,15,30,60} x 2
scorings -- and persists the fitted models under
research/hod_entry/models/1676_G7noclose_k{k}_ALL_success_*.joblib plus the
per-fill OOS scores in this script's own 1677_per_fill.csv. This is a byte-
identical rerun of an already-frozen, already-scored fit whose artifact was
simply never saved -- not the tuning/re-fitting PREREG's "Not allowed"
section forbids (that clause blocks selecting k/m/tau_low or the model
against THIS cell's own money numbers, which never happens below).

The X6 comparator (research/hod_exit_lab/score_cells.py's x6_trail: arm a
trailing stop at running_high - R once MFE >= +1R, precedence eod > stop >
target, gap-aware stop fill = min(open, stop) via that file's
_gap_or_touch) is RE-IMPLEMENTED here against this line's own bars1m arrays
(BarStore) and this line's own ENTRY_BPS/CUT_BPS cost convention --
hod_exit_lab's walker.py serves a different population (12,135 HOD-break
signals) with its own spread-based cost_net; reusing its bar/cost plumbing
here would mix populations and break this line's cost parity, so only the
exit MECHANISM (the trailing-stop rule itself, same precedence, same gap
handling) is reused, disclosed here.

TS variant, disclosed scope decision: PREREG says "evaluated at EVERY minute
from 5 to 60". Only 5 k-specific success models exist (the grid PREREG's own
Rules section fixes, {5,10,15,30,60}) and "Not allowed" forbids re-fitting
any model -- so TS cannot literally check every integer minute without
fitting 55 new models. TS(k,...) is therefore: starting at checkpoint k,
check the SAME (m, tau_low) condition at each successive checkpoint in
{5,10,15,30,60} that is >= k, in order, first firing wins; checkpoints below
k are never evaluated (this makes k a real free parameter of TS too, and
reproduces the "5k" multiplicity PREREG's own count expects: 5x2x3x3x2=180).

Not allowed (enforced): tau_low/m/k are used exhaustively over PREREG's
fixed grids, never selected on this cell's own numbers; no bar after the
decision bar (fill+k) feeds the fire decision (walk_k only walks i0+1..i0+k
for the gate; the "hold" counterfactual is the pre-existing base net_R, and
X6/placebo are computed once and reported on the SAME fired subset, not used
to pick k/m/tau); the two success models per k are the ones fit once above,
never re-tuned; every number below is reported split by scoring, never
pooled-only.

Usage: nice -n 15 python3 1677_take_profit.py
Outputs: 1677_reads.csv, 1677_per_fill.csv, 1677_take_profit.log,
  RESULT_1677.md, research/hod_entry/models/1676_G7noclose_k{k}_ALL_success_*.joblib
"""
import importlib.util
import logging
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def _load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.argv = [sys.argv[0]]  # keep f1676/gfix's own argparse from seeing our argv
f1668 = _load_module('f1668_1677', '1668_failure.py')
f1676 = _load_module('f1676_1677', '1676_shapes.py')
gfix = _load_module('gfix_1677', '1676_g6g7_fix.py')

LOG_FILE = os.path.join(HERE, '1677_take_profit.log')
READS_CSV = os.path.join(HERE, '1677_reads.csv')
PERFILL_CSV = os.path.join(HERE, '1677_per_fill.csv')
RESULT_MD = os.path.join(HERE, 'RESULT_1677.md')

TP_KS = [5, 10, 15, 30, 60]
MS = [0.5, 1.0]
TAUS = [0.3, 0.4, 0.5]
VARIANTS = ['FULL', 'P50', 'TS']
SCORINGS = ['TRAIN-H2->VAL', 'VAL->TRAIN-H2']
SEED = 1676
N_PLACEBO_SEEDS = 10
PASS_DR = 0.05
PASS_T = 2.5
PASS_VS_PLACEBO = 0.03

logger = logging.getLogger('1677')


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
# Step 1: the 10 success models (5 k x 2 scorings).
# ---------------------------------------------------------------------------

def load_feature_frame():
    """1676_features.csv (fill_id/date/symbol/half + G7 columns for our 5
    k's + the 5 success labels) joined to the causal-nearest 1670 ALL-family
    column, exactly as 1676_g6g7_fix.py's main() builds it -- minus the
    panel parquet load (unused here; this script gets bars from
    f1668.BarStore, the same source 1668/1669/1670/1676 all resolve to)."""
    header = pd.read_csv(f1676.FEATURES_CSV, nrows=0).columns
    keep = ['fill_id', 'date', 'symbol', 'half']
    for k in TP_KS:
        keep += [c for c in header if c.startswith(f'g7_k{k}_')]
        keep.append(f'g7success_next15_from_{k}')
    keep = list(dict.fromkeys(c for c in keep if c in header))
    feats = pd.read_csv(f1676.FEATURES_CSV, usecols=keep,
                         dtype={'fill_id': str, 'date': str, 'symbol': str})
    k1670 = pd.read_csv(f1676.K1670_CSV, usecols=['fill_id', 'k', 'p_stop_ALL'])
    k1670['fill_id'] = k1670['fill_id'].astype(str)
    k1670_wide = k1670.pivot_table(index='fill_id', columns='k', values='p_stop_ALL', aggfunc='first')
    k1670_wide.columns = [f'k1670_ALL_k{c}' for c in k1670_wide.columns]
    avail_ks_1670 = sorted(int(c.split('_k')[-1]) for c in k1670_wide.columns)
    feats = feats.merge(k1670_wide.reset_index(), on='fill_id', how='left')
    logger.info('feature frame: %d rows, %d cols, 1670 k-grid=%s', len(feats), len(feats.columns), avail_ks_1670)
    return feats, avail_ks_1670


def fit_and_persist(df, feat_cols, label_col, k, seed=SEED):
    """Verbatim reuse of f1676.fit_both_scorings' fit block (same model
    class/params/seed/halves) -- extended to joblib.dump the model and
    return per-fill_id predictions. do_placebo/do_importance are skipped:
    1677 uses its own coin-flip placebo (PREREG's Reads section), not
    1676's AUC placebo; importances are not a 1677 output."""
    feat_cols = [c for c in feat_cols if c in df.columns]
    sub = df[['fill_id', 'date', 'half', label_col] + feat_cols].dropna(subset=[label_col]).copy()
    sub[label_col] = sub[label_col].astype(int)
    for c in feat_cols:
        sub[c] = pd.to_numeric(sub[c], errors='coerce')
    halves = {'TRAIN-H2': sub[sub.half == 'TRAIN-H2'], 'VAL': sub[sub.half == 'VAL']}
    out = {}
    for train_h, test_h in [('TRAIN-H2', 'VAL'), ('VAL', 'TRAIN-H2')]:
        tr, te = halves[train_h], halves[test_h]
        scoring = f'{train_h}->{test_h}'
        if len(tr) < 30 or len(te) < 30 or tr[label_col].nunique() < 2 or te[label_col].nunique() < 2:
            logger.warning('k=%d %s: too little data (tr=%d te=%d), skipped', k, scoring, len(tr), len(te))
            continue
        Xtr, ytr = tr[feat_cols].values, tr[label_col].values
        Xte, yte = te[feat_cols].values, te[label_col].values
        model = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p)
        model_path = os.path.join(MODELS_DIR, f'1676_G7noclose_k{k}_ALL_success_{train_h}to{test_h}.joblib')
        joblib.dump(model, model_path)
        out[scoring] = pd.DataFrame({'fill_id': te['fill_id'].values, 'k': k, 'scoring': scoring, 'p_success': p})
        logger.info('k=%d %s: n_train=%d n_test=%d auc=%.3f -> %s', k, scoring, len(tr), len(te), auc, model_path)
    return out


# ---------------------------------------------------------------------------
# Step 2: per-fill path mechanics.
# ---------------------------------------------------------------------------

def x6_exit_dR(bars, i0, entry, stop0, target, R_unit, base_net_R):
    """research/hod_exit_lab/score_cells.py's x6_trail mechanism (MFE>=+1R
    arms a trailing stop at running_high - R; precedence eod>stop>target;
    gap-aware stop fill = min(open, stop)) against this line's bars1m
    arrays and ENTRY_BPS/CUT_BPS. Returns dR = x6_net_R - base_net_R, or NaN
    if the day's bars run out before any resolution (data gap)."""
    n = len(bars['o'])
    cur_stop = stop0
    armed = False
    running_high = None
    px = typ = None
    for j in range(i0 + 1, n):
        if bars['minarr'][j] >= f1668.EOD_M:
            px, typ = bars['o'][j], 'eod'
            break
        if bars['l'][j] <= cur_stop:
            px, typ = min(bars['o'][j], cur_stop), 'stop'
            break
        if bars['h'][j] >= target:
            px, typ = target, 'target'
            break
        if not armed and bars['h'][j] >= entry + R_unit:
            armed = True
            running_high = bars['h'][j]
        elif armed:
            running_high = max(running_high, bars['h'][j])
        if armed:
            cur_stop = max(cur_stop, running_high - R_unit)
    if px is None:
        return np.nan
    gross = (px - entry) / R_unit
    cost = (f1668.ENTRY_BPS * entry + f1668.CUT_BPS * px) / R_unit
    return (gross - cost) - base_net_R


def build_per_fill_k(pop, store):
    """One BarStore query per fill (population is 1:1 on (date,symbol));
    per fill, walk_k (f1668, verbatim) at each of TP_KS gives mtm_R at k and
    the FULL take-profit dR at k via f1668.dR_cut (verbatim); one X6
    re-walk per fill. Rows: fill_id x k."""
    rows = []
    t0 = time.time()
    n_ok = n_noidx = n_nobars = 0
    for i, r in enumerate(pop.itertuples()):
        bars = store.day_bars(r.symbol, r.date)
        if bars is None:
            n_nobars += 1
            continue
        i0 = f1668.find_fill_index(bars, r.fill_min)
        if i0 is None:
            n_noidx += 1
            continue
        R_unit = r.entry_price - r.stop
        x6dR = x6_exit_dR(bars, i0, r.entry_price, r.stop, r.target_price, R_unit, r.net_R)
        for k in TP_KS:
            w = f1668.walk_k(bars, i0, k, r.stop, r.target_price)
            if w is None:
                continue
            mtm_R = (w['close_k'] - r.entry_price) / R_unit
            dR_full = f1668.dR_cut(r.entry_price, r.stop, r.net_R, w['next_open'])
            cost_R = (f1668.CUT_BPS * w['next_open'] / R_unit) if w['next_open'] is not None else np.nan
            rows.append(dict(fill_id=r.fill_id, date=r.date, symbol=r.symbol, half=r.half, k=k,
                              base_net_R=r.net_R, mtm_R=mtm_R, dR_full=dR_full, cost_R=cost_R,
                              still_open=(w['preempt'] == ''), x6_dR=x6dR))
        n_ok += 1
        if (i + 1) % 1000 == 0:
            logger.info('per-fill walk: %d/%d fills (%.0fs)', i + 1, len(pop), time.time() - t0)
    logger.info('per-fill walk done: %d ok, %d no bars, %d no fill index (%.0fs)',
                n_ok, n_nobars, n_noidx, time.time() - t0)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3: reads.
# ---------------------------------------------------------------------------

def stats_block(dR, dates):
    """day_clustered_t/iid_t/ex_top5_mean/mde on a genuinely PAIRED per-fill
    delta array (each fired trade IS its own before/after pair -- dR_full
    already subtracts base_net_R), so these are applied directly, not via
    f1676's group-vs-book paired_day_clustered_t (a different design for
    subgroup-vs-book-mean comparisons, not used here)."""
    n = len(dR)
    if n == 0:
        return dict(n=0, mean_dR=np.nan, iid_t=np.nan, day_t=np.nan, ex_top5=np.nan, mde=np.nan)
    tmp = pd.DataFrame({'date': dates, 'net_R': dR})
    sd = tmp['net_R'].std(ddof=1)
    return dict(n=n, mean_dR=tmp['net_R'].mean(), iid_t=f1676.iid_t(tmp['net_R']),
                day_t=f1676.day_clustered_t(tmp, 'date', 'net_R'),
                ex_top5=f1676.ex_top5_mean(tmp['net_R']), mde=f1676.mde(sd, n))


def placebo_block(pool_mask, assigned_dR, dates, n_fired, seed_base=SEED):
    """The model replaced by a seeded coin flip at the SAME firing rate,
    drawing n_fired rows (exact count) from pool_mask at random, applying
    assigned_dR (the mechanical, model-independent dR each row would
    realize if selected); mean of 10 seeds (PREREG's fallback instruction)."""
    idx = np.where(pool_mask.values)[0]
    if len(idx) == 0 or n_fired == 0:
        return dict(mean_dR=np.nan, day_t=np.nan, ex_top5=np.nan)
    n_draw = min(n_fired, len(idx))
    means, ts, ex5 = [], [], []
    for s in range(N_PLACEBO_SEEDS):
        rng = np.random.RandomState(seed_base * 100 + s)
        sel = rng.choice(idx, size=n_draw, replace=False)
        st = stats_block(assigned_dR.values[sel], dates.values[sel])
        means.append(st['mean_dR']); ts.append(st['day_t']); ex5.append(st['ex_top5'])
    return dict(mean_dR=np.nanmean(means), day_t=np.nanmean(ts), ex_top5=np.nanmean(ex5))


def decompose(dR_fired, cost_fired):
    below = dR_fired[dR_fired > 0]   # base ended BELOW the exit -- giveback saved
    above = dR_fired[dR_fired < 0]   # base ended ABOVE the exit -- continuation forgone
    return dict(giveback_saved_mean=below.mean() if len(below) else np.nan, giveback_saved_n=len(below),
                continuation_forgone_mean=(-above).mean() if len(above) else np.nan, continuation_forgone_n=len(above),
                mean_cost_R=cost_fired.mean() if len(cost_fired) else np.nan)


def first_qualify(piv_open, piv_mtm, piv_dRfull, checkpoints, m):
    """Per fill_id: does it EVER reach open & mtm>=m at some checkpoint in
    `checkpoints` (in order), and the dR_full it would realize there if
    exited at that first qualifying checkpoint -- the model-independent
    timing surface TS's placebo pool/assigned-dR are built from."""
    idx = piv_open.index
    qual = pd.Series(False, index=idx)
    dR = pd.Series(0.0, index=idx)
    active = pd.Series(True, index=idx)
    for kk in checkpoints:
        cond = active & piv_open[kk].fillna(False) & (piv_mtm[kk] >= m)
        dR[cond] = piv_dRfull[kk][cond]
        qual |= cond
        active &= ~cond
    return qual, dR


def run_variant_full(piv, k, m, tau, half_of, dates_of):
    open_k, mtm_k, p_k, dRfull_k, cost_k = piv['open'][k], piv['mtm'][k], piv['p'][k], piv['dRfull'][k], piv['cost'][k]
    pool = open_k.fillna(False) & (mtm_k >= m)
    fired = pool & (p_k < tau)
    n_pool, n_fired = int(pool.sum()), int(fired.sum())
    dR_fired = dRfull_k[fired]
    st = stats_block(dR_fired.values, dates_of.loc[fired].values)
    dec = decompose(dR_fired, cost_k[fired])
    plc = placebo_block(pool, dRfull_k, dates_of, n_fired)
    x6 = piv['x6'][fired].mean() if n_fired else np.nan
    return dict(n_pool=n_pool, n_fired=n_fired, share_fired=(n_fired / n_pool if n_pool else np.nan),
                **st, **dec, placebo_mean_dR=plc['mean_dR'], placebo_day_t=plc['day_t'],
                placebo_ex_top5=plc['ex_top5'], x6_mean_dR=x6)


def run_variant_p50(piv, k, m, tau, half_of, dates_of):
    open_k, mtm_k, p_k, dRfull_k, cost_k = piv['open'][k], piv['mtm'][k], piv['p'][k], piv['dRfull'][k], piv['cost'][k]
    pool = open_k.fillna(False) & (mtm_k >= m)
    fired = pool & (p_k < tau)
    n_pool, n_fired = int(pool.sum()), int(fired.sum())
    dR50_fired = 0.5 * dRfull_k[fired]
    st = stats_block(dR50_fired.values, dates_of.loc[fired].values)
    dec = decompose(dR50_fired, 0.5 * cost_k[fired])
    plc = placebo_block(pool, 0.5 * dRfull_k, dates_of, n_fired)
    x6 = (0.5 * piv['x6'][fired] + 0.5 * 0.0).mean() if n_fired else np.nan  # half rides the base exit (dR=0 vs itself)
    return dict(n_pool=n_pool, n_fired=n_fired, share_fired=(n_fired / n_pool if n_pool else np.nan),
                **st, **dec, placebo_mean_dR=plc['mean_dR'], placebo_day_t=plc['day_t'],
                placebo_ex_top5=plc['ex_top5'], x6_mean_dR=x6)


def run_variant_ts(piv, k, m, tau, half_of, dates_of):
    checkpoints = [kk for kk in TP_KS if kk >= k]
    idx = piv['open'][k].index
    active = pd.Series(True, index=idx)
    fired = pd.Series(False, index=idx)
    realized = pd.Series(0.0, index=idx)
    ever_qualify = pd.Series(False, index=idx)
    for kk in checkpoints:
        open_kk, mtm_kk, p_kk, dRfull_kk = piv['open'][kk].fillna(False), piv['mtm'][kk], piv['p'][kk], piv['dRfull'][kk]
        mtm_ok = active & open_kk & (mtm_kk >= m)
        ever_qualify |= mtm_ok
        cond = mtm_ok & (p_kk < tau)
        realized[cond] = dRfull_kk[cond]
        fired |= cond
        active &= ~cond
    n_pool, n_fired = int(ever_qualify.sum()), int(fired.sum())
    dR_fired = realized[fired]
    cost_fired = piv['cost'][k].reindex(idx)  # approx: charge the START checkpoint's per-R cost scale for reporting only
    st = stats_block(dR_fired.values, dates_of.loc[fired].values)
    dec = decompose(dR_fired, cost_fired[fired])
    qual_mask, assigned_dR = first_qualify(piv['open'], piv['mtm'], piv['dRfull'], checkpoints, m)
    plc = placebo_block(qual_mask, assigned_dR, dates_of, n_fired)
    x6 = piv['x6'][fired].mean() if n_fired else np.nan
    return dict(n_pool=n_pool, n_fired=n_fired, share_fired=(n_fired / n_pool if n_pool else np.nan),
                **st, **dec, placebo_mean_dR=plc['mean_dR'], placebo_day_t=plc['day_t'],
                placebo_ex_top5=plc['ex_top5'], x6_mean_dR=x6)


VARIANT_FN = {'FULL': run_variant_full, 'P50': run_variant_p50, 'TS': run_variant_ts}


def run_all_reads(per_fill_scored):
    rows = []
    for scoring in SCORINGS:
        sub = per_fill_scored[per_fill_scored.scoring == scoring]
        if sub.empty:
            logger.warning('scoring=%s: no scored rows, skipped entirely', scoring)
            continue
        piv = dict(
            open=sub.pivot(index='fill_id', columns='k', values='still_open'),
            mtm=sub.pivot(index='fill_id', columns='k', values='mtm_R'),
            p=sub.pivot(index='fill_id', columns='k', values='p_success'),
            dRfull=sub.pivot(index='fill_id', columns='k', values='dR_full'),
            cost=sub.pivot(index='fill_id', columns='k', values='cost_R'),
        )
        base = sub.drop_duplicates('fill_id').set_index('fill_id')
        piv['x6'] = base['x6_dR'].reindex(piv['open'].index)
        dates_of = base['date'].reindex(piv['open'].index)
        half_of = base['half'].reindex(piv['open'].index)
        logger.info('scoring=%s: %d fill_ids pivoted', scoring, len(piv['open']))
        for variant in VARIANTS:
            fn = VARIANT_FN[variant]
            for k in TP_KS:
                for m in MS:
                    for tau in TAUS:
                        r = fn(piv, k, m, tau, half_of, dates_of)
                        r.update(variant=variant, k=k, m=m, tau=tau, scoring=scoring)
                        rows.append(r)
            logger.info('variant=%s done (%d rows so far)', variant, len(rows))
    df = pd.DataFrame(rows)
    df['pass_dR'] = df['mean_dR'] >= PASS_DR
    df['pass_t'] = df['day_t'] >= PASS_T
    df['pass_ex_top5'] = df['ex_top5'] > 0
    df['pass_vs_placebo'] = (df['mean_dR'] - df['placebo_mean_dR']) >= PASS_VS_PLACEBO
    df['passes_this_scoring'] = df['pass_dR'] & df['pass_t'] & df['pass_ex_top5'] & df['pass_vs_placebo']
    return df


def write_result_md(reads_df, n_pop, n_models_missing):
    both = reads_df.pivot_table(index=['variant', 'k', 'm', 'tau'], columns='scoring',
                                 values='passes_this_scoring', aggfunc='first')
    overall_pass = both.all(axis=1) if len(both.columns) == 2 else pd.Series(False, index=both.index)
    n_pass = int(overall_pass.sum())
    L = []
    L.append('# RESULT -- cell 1,677: the model-gated take-profit')
    L.append('')
    L.append(f'PREREG: research/hod_entry/PREREG_1677.md, FROZEN 2026-09-30 07:40 UTC. Population n={n_pop} '
              f'(1,663 join, floored r_pct>=1.5%). {180} paired reads (5k x 2m x 3tau x 3variants x 2scorings) '
              f'+ placebo (10 seeds/cell) + X6 comparator, all built.')
    if n_models_missing:
        L.append(f'\n**{n_models_missing} of 10 (k,scoring) success models could not be fit** (insufficient train/test '
                  'data or single-class labels) -- see log; those k/scoring cells are absent below, not zero.')
    L.append('')
    L.append(f'## Pass bar: {n_pass}/{len(overall_pass)} (variant,k,m,tau) combos pass BOTH scorings '
              f'(dR>=+{PASS_DR}R, day_t>={PASS_T}, ex-top5%>0, beats placebo by >=+{PASS_VS_PLACEBO}R)')
    if n_pass:
        L.append('')
        for key in overall_pass[overall_pass].index:
            L.append(f'- PASSES: variant={key[0]} k={key[1]} m={key[2]} tau={key[3]}')
    else:
        L.append('\nNothing clears the pass bar on both scorings.')
    L.append('')
    L.append('## Best cell per variant (by min(mean_dR) across the two scorings; reported whether or not it passes)')
    for variant in VARIANTS:
        vdf = reads_df[reads_df.variant == variant]
        piv = vdf.pivot_table(index=['k', 'm', 'tau'], columns='scoring', values='mean_dR', aggfunc='first')
        if piv.empty or len(piv.columns) < 2:
            L.append(f'\n### {variant}: insufficient data on one or both scorings')
            continue
        piv = piv.dropna()
        if piv.empty:
            L.append(f'\n### {variant}: no complete (both-scoring) cells')
            continue
        piv['worst'] = piv.min(axis=1)
        best_key = piv['worst'].idxmax()
        k, m, tau = best_key
        L.append(f'\n### {variant}: k={k} m={m} tau_low={tau}')
        L.append('| scoring | n_pool | n_fired | share_fired | mean_dR | iid_t | day_t | ex_top5_dR | MDE | '
                  'giveback_saved (n) | continuation_forgone (n) | mean_cost_R | placebo dR/t | X6 mean_dR |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for scoring in SCORINGS:
            row = reads_df[(reads_df.variant == variant) & (reads_df.k == k) & (reads_df.m == m) &
                            (reads_df.tau == tau) & (reads_df.scoring == scoring)]
            if row.empty:
                L.append(f'| {scoring} | -- all missing -- |' + '|' * 12)
                continue
            r = row.iloc[0]
            L.append(f"| {scoring} | {r.n_pool} | {r.n_fired} | {r.share_fired:.3f} | {r.mean_dR:+.4f} | "
                      f"{r.iid_t:.2f} | {r.day_t:.2f} | {r.ex_top5:+.4f} | {r.mde:.4f} | "
                      f"{r.giveback_saved_mean:+.3f} ({r.giveback_saved_n}) | "
                      f"{r.continuation_forgone_mean:+.3f} ({r.continuation_forgone_n}) | {r.mean_cost_R:.4f} | "
                      f"{r.placebo_mean_dR:+.4f}/{r.placebo_day_t:.2f} | {r.x6_mean_dR:+.4f} |")
    L.append('')
    L.append('## Adequacy')
    L.append('Reused, never re-fit: f1668.walk_k/dR_cut (the FULL exit mechanic + 6bps cost), f1676\'s stat kit, '
              'gfix.build_group_cols_fix (the exact G7-without-close_R+ALL feature group). The 10 success models '
              '(5k x 2 scorings) were re-run ONCE with 1676\'s identical fit code + seed because neither the '
              'models nor per-fill scores survived 1676\'s original run (see script docstring) -- disclosed, not '
              'hidden. X6 is the exit-lab\'s trailing-stop MECHANISM re-implemented on this line\'s own bars/cost '
              '(different population, so its own plumbing was not reused). TS checks only the 5 fixed checkpoints '
              '(no refit) rather than literally every minute -- disclosed. Both scorings always reported '
              'separately; MDE beside every t; no pooled-only numbers. A pass here still requires independent '
              'reimplementation before the owner sees a number, per PREREG.')
    L.append('')
    L.append('Files: 1677_take_profit.py, 1677_reads.csv, 1677_per_fill.csv, 1677_take_profit.log, models under '
              'research/hod_entry/models/1676_G7noclose_k*_ALL_success_*.joblib.')
    with open(RESULT_MD, 'w') as f:
        f.write('\n'.join(L) + '\n')
    logger.info('wrote %s', RESULT_MD)
    return n_pass, len(overall_pass)


def main():
    setup_logging()
    t0 = time.time()
    logger.info('=== 1,677 take-profit start ===')
    if hasattr(f1668, 'check_disk'):
        f1668.check_disk(5.0)

    pop = f1668.load_population()
    feats, avail_ks_1670 = load_feature_frame()
    group_cols = gfix.build_group_cols_fix(feats, avail_ks_1670)

    score_frames = []
    n_missing = 0
    for k in TP_KS:
        grp = f'G7noclose_k{k}+ALL'
        label = f'g7success_next15_from_{k}'
        res = fit_and_persist(feats, group_cols[grp], label, k)
        n_missing += (2 - len(res))
        for scoring, sdf in res.items():
            score_frames.append(sdf)
    scores = pd.concat(score_frames, ignore_index=True)
    scores['fill_id'] = scores['fill_id'].astype(str)
    logger.info('scores: %d rows across %d (k,scoring) cells (%d cells missing)',
                len(scores), scores[['k', 'scoring']].drop_duplicates().shape[0], n_missing)

    store = f1668.BarStore(f1668.BARS_DB)
    pop['fill_id'] = pop['fill_id'].astype(str)
    per_fill_k = build_per_fill_k(pop, store)
    store.close()

    per_fill_scored = per_fill_k.merge(scores[['fill_id', 'k', 'scoring', 'p_success']],
                                        on=['fill_id', 'k'], how='inner')
    logger.info('per_fill_scored: %d rows (of %d per_fill_k rows before the score join)',
                len(per_fill_scored), len(per_fill_k))
    tmp = PERFILL_CSV + '.tmp'
    per_fill_scored.to_csv(tmp, index=False)
    os.replace(tmp, PERFILL_CSV)
    logger.info('wrote %s (%d rows)', PERFILL_CSV, len(per_fill_scored))

    reads_df = run_all_reads(per_fill_scored)
    tmp2 = READS_CSV + '.tmp'
    reads_df.to_csv(tmp2, index=False)
    os.replace(tmp2, READS_CSV)
    logger.info('wrote %s (%d rows)', READS_CSV, len(reads_df))

    n_pass, n_total = write_result_md(reads_df, pop['fill_id'].nunique(), n_missing)
    logger.info('=== 1,677 done: %d/%d combos pass both scorings, %.0fs total ===', n_pass, n_total, time.time() - t0)


if __name__ == '__main__':
    main()
