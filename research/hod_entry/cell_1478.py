#!/usr/bin/env python3
"""Cell 1,478 -- research/hod_entry/PREREG_1478.md (FROZEN 2026-09-26) + its same-day amendment.

Supervised big-day predictor at the arm bar (owner: "How can you predict the lookahead ... be
creative"). Base book: the 9,911 cell-1,438 fills (correct levels), status == 'fill', TRAIN-H2 /
VAL only (TEST sealed, never read here). Labels are the ONLY look-ahead, used solely as training
targets -- every feature is causal at the close of arm bar j (features_1478_A/B/C.csv, built by
prior agents per the PREREG's Feature list, already reference bars_fills_1478.db -- the amendment's
single-source SIP store -- for every bar-derived quantity).

Pipeline:
  1. Labels. L1 = full-day (high-low)/low >= 10% on the Databento PIT daily bar (causal join by
     instrument_id, never the raw 'symbol' column which is null on some rows). L2 = L1 AND
     day close >= $10 AND prior SESSION's volume >= 1,000,000 (shift(1) within instrument_id,
     i.e. the trading-calendar-previous row in the PIT daily panel, not a calendar-day shift).
  2. Outcome. cell_1457_features.csv's net_R_corr_v2 standard (recomputed here via
     cell_1457.build_base_cost so the ORIGINAL per-row stop slip is recoverable), with the
     stop-limit exit slip on why in {stop, stop_bar} rows SUBSTITUTED by the PREREG amendment's
     expected-value slip: 0.88 * (2.9 / 3.2 bps, TRAIN-H2 / VAL, the filled-stop holdout means)
     + 0.12 * (94 / 76 bps, the 12% no-fill tail's measured mean), converted to R via
     exit_price * bps / 1e4 / R (the PREREG's own phrasing: "in R via exit price / R"). All other
     why (target, eod, eod_fallback) rows are UNCHANGED from net_R_corr_v2.
  3. Features = the union of numeric columns in features_1478_A/B/C.csv joined on
     (day, symbol, fill_min) -- an EXACT float merge (verified 9,911/9,911 on all three joins
     before this file was written) -- excluding id columns, the object-typed 'sic2' join key, and
     the three DECOY columns.
  4. STEP 1 decoy model: HistGradientBoostingClassifier on ONLY the decoy columns
     (store_served_1438, rth_bar_count_1438, tick_window_has_bar_j) -> VAL AUC per label; VOID
     (decoy_void=True) if AUC > 0.55, but every following step still runs and is still reported.
  5. STEP 2 real models (decoy columns excluded): HistGradientBoostingClassifier, seed 1478,
     5-fold CV INSIDE TRAIN-H2 over the frozen grid (max_depth {3,5} x learning_rate {0.03,0.1} x
     max_iter {200,600} x min_samples_leaf {50,200} = 16 combos), refit on all of TRAIN-H2, applied
     ONCE to VAL; and LogisticRegression (standardised, median-imputed), same features, no CV.
  6. Selection: probability >= the threshold that keeps TRAIN-H2's own top tercile (frozen on
     TRAIN-H2 predictions only, then applied unchanged to VAL).
  7. A label-shuffled placebo (seed 1478): TRAIN-H2 labels permuted, the SAME CV+refit pipeline,
     applied ONCE to the TRUE VAL labels -- tests for leakage/false signal.

Usage:
    nice -n 19 python3 research/hod_entry/cell_1478.py [--dry-run]

    --dry-run scores a fixed 400-row sample (seed 1478) instead of the full 9,911-row book, for
    fast pipeline verification; writes DRYRUN-suffixed outputs so they never collide with a full
    run.

Outputs: research/hod_entry/model_1478_predictions.csv (one row per fill: probabilities, kept
flags, both labels, outcome_R) and research/hod_entry/RESULT_1478.md.

Not allowed (PREREG): any feature using data after bar j; tuning outside the fixed grid; choosing
the threshold on VAL; reading TEST; dropping NaN rows selectively.
"""
import argparse
import itertools
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from research.hod_entry import cell_1445 as c1445          # noqa: E402 -- base book, cost, scoring
from research.hod_entry import cell_1457 as c1457          # noqa: E402 -- build_base_cost standard

SEED = 1478
GRID = dict(max_depth=[3, 5], learning_rate=[0.03, 0.1], max_iter=[200, 600],
            min_samples_leaf=[50, 200])
DECOY_COLS = ['store_served_1438', 'rth_bar_count_1438', 'tick_window_has_bar_j']
ID_COLS = ['day', 'symbol', 'fill_min', 'split']
# Defense in depth: feature_columns() is meant to run on features_1478_A/B/C ALONE (never on the
# label/outcome-enriched `merged` frame), but these names are dropped unconditionally too, so a
# future caller who passes the wrong frame cannot leak the label (e.g. full_day_range_pct IS the
# L1 threshold) into the feature matrix.
LABEL_OUTCOME_COLS = ['L1', 'L2', 'outcome_R', 'day_high', 'day_low', 'day_close',
                      'full_day_range_pct', 'prev_session_volume', 'fill', 'stop', 'level',
                      'exit_m', 'exit_price', 'R', 'raw_R', 'cost_R', 'net_R', 'old_slip_R',
                      'new_slip_R', 'net_R_costfix', 'net_R_corr', 'half_entry', 'spread_frac',
                      'nbbo_fallback', 'why', 'exit_half_src']
DECOY_VOID_AUC = 0.55
BASE_CACHEONLY_SHARE = 0.195          # PREREG's disclosed base rate
CACHEONLY_TOL = 0.05                  # +/- 5pp check on the kept set
STOP_WHY = {'stop', 'stop_bar'}
# Expected-value stop-limit slip, PREREG_1478 amendment: 0.88 * filled-stop bps + 0.12 * no-fill-
# tail bps, per holdout (TRAIN-H2 / VAL).
SLIP_STOP_BPS = {'TRAIN': 0.88 * 2.9 + 0.12 * 94.0, 'VAL': 0.88 * 3.2 + 0.12 * 76.0}
TOP_TERCILE_Q = 2.0 / 3.0
PASS_KEPT_MEAN = 0.15
PASS_T = 2.5
PASS_FILLS_WK = 3.0
PASS_AUC = 0.60
PLACEBO_AUC_MAX = 0.53
PLACEBO_MEAN_TOL = 0.05

FEATURES_A = os.path.join(HERE, 'features_1478_A.csv')
FEATURES_B = os.path.join(HERE, 'features_1478_B.csv')
FEATURES_C = os.path.join(HERE, 'features_1478_C.csv')
CAUSAL_CSV = os.path.join(HERE, 'causal_arming_causal.csv')


def log(msg):
    """Verbose progress line, flushed immediately (print() is buffered under nohup otherwise)."""
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ================================================================================================
# Step 1: labels (the ONLY look-ahead, used solely as training targets)
# ================================================================================================

def build_labels(fills):
    """Attach day_high/day_low/day_close (the signal day's OWN PIT daily bar -- the declared
    look-ahead, used only to build L1/L2) and prev_session_volume (shift(1) within instrument_id --
    the trading-calendar-previous PIT daily row, causal by construction since it excludes the
    signal day). Resolution is via instrument_id (c1445.resolve_instrument_ids), never the
    parquet's raw 'symbol' column, which is null on a minority of rows."""
    map_df = c1445.load_symbol_map()
    pairs = list(zip(fills.symbol, fills.day))
    instr_by_sd = c1445.resolve_instrument_ids(pairs, map_df)
    n_resolved = sum(1 for sd in pairs if sd in instr_by_sd)
    log(f'build_labels: instrument id resolved for {n_resolved}/{len(fills)} fills')

    iids = set(instr_by_sd.values())
    daily = pd.read_parquet(c1445.DAILY_PARQUET,
                             columns=['bar_date', 'symbol', 'instrument_id', 'high', 'low',
                                      'close', 'volume'])
    daily = daily[daily.instrument_id.isin(iids)].copy()
    daily['bar_date'] = pd.to_datetime(daily['bar_date'])
    daily = daily.sort_values(['instrument_id', 'bar_date']).reset_index(drop=True)
    daily['prev_volume'] = daily.groupby('instrument_id', sort=False)['volume'].shift(1)
    idx = daily.set_index(['instrument_id', 'bar_date'])

    cols = ['high', 'low', 'close', 'prev_volume']
    recs = []
    for r in fills.itertuples():
        iid = instr_by_sd.get((r.symbol, r.day))
        day_ts = pd.Timestamp(r.day)
        vals = {c: np.nan for c in cols}
        if iid is not None and (iid, day_ts) in idx.index:
            drow = idx.loc[(iid, day_ts)]
            if isinstance(drow, pd.DataFrame):
                drow = drow.iloc[0]
            vals = {c: drow[c] for c in cols}
        recs.append(vals)
    feat = pd.DataFrame(recs, index=fills.index)

    out = fills.copy()
    out['day_high'] = feat['high']
    out['day_low'] = feat['low']
    out['day_close'] = feat['close']
    out['prev_session_volume'] = feat['prev_volume']
    out['full_day_range_pct'] = (out['day_high'] - out['day_low']) / out['day_low'] * 100.0

    n_missing = int(out['full_day_range_pct'].isna().sum())
    log(f'build_labels: full_day_range_pct missing for {n_missing}/{len(out)} rows '
        f'({n_missing / len(out):.2%})')

    out['L1'] = np.where(out['full_day_range_pct'].isna(), np.nan,
                          (out['full_day_range_pct'] >= 10.0).astype(float))
    l1_is_one = out['L1'] == 1.0
    l2_incomplete = out['L1'].isna() | (l1_is_one & (out['day_close'].isna() |
                                                      out['prev_session_volume'].isna()))
    l2_true = l1_is_one & (out['day_close'] >= 10.0) & (out['prev_session_volume'] >= 1_000_000)
    out['L2'] = np.where(l2_incomplete, np.nan, l2_true.astype(float))

    for lbl in ('L1', 'L2'):
        for split in ('TRAIN', 'VAL'):
            sub = out.loc[out.split == split, lbl]
            log(f'  {lbl} {split}: base rate {sub.mean():.4f} (n={sub.notna().sum()}, '
                f'{sub.isna().sum()} NaN)')
    return out


# ================================================================================================
# Step 2: outcome R -- cell_1457's net_R_corr_v2 standard with the amendment's stop-limit slip
# substitution on why in {stop, stop_bar}
# ================================================================================================

def substitute_stop_slip(enriched):
    """Pure arithmetic: outcome_R = net_R_corr_v2 EXCEPT on why in {stop, stop_bar}, where the
    per-row slip cell_1457 already subtracted is replaced by the PREREG amendment's EXPECTED-VALUE
    slip -- 0.88 * filled-stop bps + 0.12 * no-fill-tail bps, per split (SLIP_STOP_BPS), converted
    to R via exit_price * bps / 1e4 / R (the PREREG's own phrasing: "in R via exit price / R").
    Requires columns: why, split, exit_price, R, net_R_costfix, net_R_corr. target/eod/
    eod_fallback rows pass through net_R_corr unchanged."""
    is_stop = enriched['why'].isin(STOP_WHY)
    bps = enriched['split'].map(SLIP_STOP_BPS).astype(float)
    new_slip_R = enriched['exit_price'] * bps / 1e4 / enriched['R']

    outcome = enriched['net_R_corr'].copy()
    outcome.loc[is_stop] = enriched.loc[is_stop, 'net_R_costfix'] - new_slip_R.loc[is_stop]
    return outcome, new_slip_R, is_stop


def build_outcome(base):
    """outcome_R = net_R_corr_v2 (cell_1457 standard: half-spread once via corrected_cost, measured
    per-row stop slip with EOD-specific fallback) with the amendment's expected-value stop-limit
    slip substituted on why in {stop, stop_bar} rows -- see substitute_stop_slip()."""
    enriched, eod_fb_bps = c1457.build_base_cost(base)
    old_slip_R = enriched['net_R_costfix'] - enriched['net_R_corr']
    outcome, new_slip_R, is_stop = substitute_stop_slip(enriched)

    enriched['old_slip_R'] = old_slip_R
    enriched['new_slip_R'] = new_slip_R
    enriched['outcome_R'] = outcome
    log(f'build_outcome: substituted expected slip on {int(is_stop.sum())} stop/stop_bar rows '
        f'(TRAIN-H2 {SLIP_STOP_BPS["TRAIN"]:.3f}bps, VAL {SLIP_STOP_BPS["VAL"]:.3f}bps); '
        f'eod fallback (unchanged, 1457 standard) = {eod_fb_bps:.2f}bps; '
        f'book mean net_R_corr_v2={enriched["net_R_corr"].mean():.4f} -> '
        f'outcome_R={enriched["outcome_R"].mean():.4f}')
    return enriched


# ================================================================================================
# Step 3: feature matrix -- merge A/B/C, drop id/decoy/object columns
# ================================================================================================

def load_and_merge_features():
    a = pd.read_csv(FEATURES_A)
    b = pd.read_csv(FEATURES_B)
    c = pd.read_csv(FEATURES_C)
    m = a.merge(b, on=['day', 'symbol', 'fill_min'], how='inner', suffixes=('', '_b'))
    assert len(m) == len(a), f'A x B merge dropped rows: {len(a)} -> {len(m)}'
    if 'split_b' in m.columns:
        assert (m['split'] == m['split_b']).all(), 'A/B split mismatch'
        m = m.drop(columns=['split_b'])
    m = m.merge(c, on=['day', 'symbol', 'fill_min'], how='inner', suffixes=('', '_c'))
    assert len(m) == len(a), f'x C merge dropped rows: {len(a)} -> {len(m)}'
    log(f'load_and_merge_features: {len(m)} rows, {len(m.columns)} columns after A+B+C merge')
    return m


def feature_columns(merged):
    """Every numeric column not in ID_COLS/DECOY_COLS/LABEL_OUTCOME_COLS/the object-typed 'sic2'
    join key. Intended input: features_1478_A/B/C merged ALONE (see load_and_merge_features) --
    LABEL_OUTCOME_COLS is a defense-in-depth blocklist, not the primary guarantee."""
    drop = set(ID_COLS) | set(DECOY_COLS) | set(LABEL_OUTCOME_COLS) | {'sic2'}
    obj_cols = [c for c in merged.columns if merged[c].dtype == object and c not in ID_COLS]
    if obj_cols:
        log(f'feature_columns: dropping non-numeric columns from the feature matrix: {obj_cols}')
    drop |= set(obj_cols)
    cols = [c for c in merged.columns if c not in drop]
    log(f'feature_columns: {len(cols)} features (decoy columns {DECOY_COLS} excluded)')
    return cols


# ================================================================================================
# Step 4: models
# ================================================================================================

def fit_cv_hgb(X, y, seed=SEED):
    """5-fold StratifiedKFold CV over the frozen grid, scored by ROC AUC; refit the best combo on
    all of X/y. Returns (fitted_model, best_params, mean_cv_auc, cv_table[list of dict])."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=int)
    cv_table = []
    best = None
    combos = list(itertools.product(GRID['max_depth'], GRID['learning_rate'], GRID['max_iter'],
                                     GRID['min_samples_leaf']))
    for max_depth, lr, max_iter, msl in combos:
        aucs = []
        for tr_idx, te_idx in skf.split(Xv, yv):
            mdl = HistGradientBoostingClassifier(max_depth=max_depth, learning_rate=lr,
                                                  max_iter=max_iter, min_samples_leaf=msl,
                                                  random_state=seed)
            mdl.fit(Xv[tr_idx], yv[tr_idx])
            p = mdl.predict_proba(Xv[te_idx])[:, 1]
            aucs.append(roc_auc_score(yv[te_idx], p))
        mean_auc = float(np.mean(aucs))
        params = dict(max_depth=max_depth, learning_rate=lr, max_iter=max_iter,
                      min_samples_leaf=msl)
        cv_table.append(dict(**params, cv_auc=mean_auc))
        if best is None or mean_auc > best[0]:
            best = (mean_auc, params)
    cv_auc, best_params = best
    final = HistGradientBoostingClassifier(random_state=seed, **best_params)
    final.fit(Xv, yv)
    log(f'fit_cv_hgb: best {best_params} cv_auc={cv_auc:.4f} (of {len(combos)} combos)')
    return final, best_params, cv_auc, cv_table


def fit_lr(X, y, seed=SEED):
    """Standardised, median-imputed LogisticRegression -- no CV/tuning (PREREG: "the simple
    model")."""
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    Xi = imputer.fit_transform(X.to_numpy(dtype=float))
    Xs = scaler.fit_transform(Xi)
    mdl = LogisticRegression(max_iter=2000, random_state=seed)
    mdl.fit(Xs, y.to_numpy(dtype=int))
    return mdl, imputer, scaler


def lr_predict_proba(mdl, imputer, scaler, X):
    Xi = imputer.transform(X.to_numpy(dtype=float))
    Xs = scaler.transform(Xi)
    return mdl.predict_proba(Xs)[:, 1]


def top_tercile_threshold(prob_train):
    """The selection rule (frozen): probability >= the threshold that keeps TRAIN-H2's own top
    tercile. A pure function of TRAIN predictions only -- VAL never enters this computation."""
    return float(np.quantile(np.asarray(prob_train, dtype=float), TOP_TERCILE_Q))


def shuffle_labels(y, seed=SEED):
    """Label-shuffled placebo: a random permutation of y's own values (same index, same class
    counts, no information from any feature)."""
    rng = np.random.RandomState(seed)
    yv = pd.Series(y).to_numpy()
    return pd.Series(rng.permutation(yv), index=pd.Series(y).index)


# ================================================================================================
# Step 5: scoring
# ================================================================================================

def score_holdout(df_split, prob, threshold, outcome_col, label_col, cache_col, weeks):
    """kept = prob >= threshold (frozen on TRAIN-H2). Returns the score dict for one holdout."""
    kept = prob >= threshold
    kept_df = df_split.loc[kept]
    dropped_df = df_split.loc[~kept]
    kept_R = kept_df[outcome_col]
    dropped_R = dropped_df[outcome_col]
    t = c1445.day_clustered_t(kept_R, kept_df['day']) if len(kept_R) else float('nan')
    ex5 = c1445.ex_top5_mean(kept_R) if len(kept_R) else float('nan')
    slot_in = kept_df.rename(columns={'fill_min': 'entry_m'})[['day', 'entry_m', 'exit_m']]
    fwk = c1445.fills_per_week(slot_in, weeks) if len(kept_R) else 0.0
    return dict(
        n_kept=int(kept.sum()), n_dropped=int((~kept).sum()),
        base_rate=float(df_split[label_col].mean()),
        kept_mean=float(kept_R.mean()) if len(kept_R) else float('nan'),
        dropped_mean=float(dropped_R.mean()) if len(dropped_R) else float('nan'),
        t_kept=float(t) if t is not None else float('nan'),
        ex_top5=float(ex5) if ex5 is not None else float('nan'),
        fills_wk=float(fwk),
        kept_bigday_rate=float(kept_df[label_col].mean()) if len(kept_df) else float('nan'),
        kept_cacheonly_share=float(kept_df[cache_col].mean()) if len(kept_df) else float('nan'),
    )


def decile_table(prob, outcome_R, label):
    """VAL decile table (report-only): decile 10 = highest predicted probability."""
    dec = pd.qcut(prob, 10, labels=False, duplicates='drop')
    rows = []
    for d in sorted(pd.unique(dec)):
        m = dec == d
        rows.append(dict(decile=int(d) + 1, n=int(m.sum()), mean_prob=float(np.mean(prob[m])),
                          mean_R=float(np.mean(outcome_R[m])), bigday_rate=float(np.mean(label[m]))))
    return rows


# ================================================================================================
# Step 6: per-label pipeline
# ================================================================================================

def run_label(label_col, merged, feature_cols, weeks):
    valid = merged[label_col].notna()
    n_dropped_label_nan = int((~valid).sum())
    work = merged.loc[valid].reset_index(drop=True)
    tr = work.loc[work.split == 'TRAIN'].reset_index(drop=True)
    va = work.loc[work.split == 'VAL'].reset_index(drop=True)
    log(f'run_label {label_col}: {n_dropped_label_nan} rows dropped for NaN label; '
        f'TRAIN-H2 n={len(tr)}, VAL n={len(va)}')

    ytr = tr[label_col].astype(int)
    yva = va[label_col].astype(int)
    Xtr, Xva = tr[feature_cols], va[feature_cols]
    Xtr_decoy, Xva_decoy = tr[DECOY_COLS], va[DECOY_COLS]

    # ---- STEP 1: decoy model ----
    decoy_mdl, decoy_params, decoy_cv_auc, _ = fit_cv_hgb(Xtr_decoy, ytr)
    decoy_val_auc = float(roc_auc_score(yva, decoy_mdl.predict_proba(Xva_decoy.to_numpy())[:, 1]))
    decoy_void = decoy_val_auc > DECOY_VOID_AUC
    log(f'{label_col} DECOY: cv_auc={decoy_cv_auc:.4f} val_auc={decoy_val_auc:.4f} '
        f'void={decoy_void}')

    # ---- STEP 2: real HGB ----
    hgb_mdl, hgb_params, hgb_cv_auc, hgb_cv_table = fit_cv_hgb(Xtr, ytr)
    hgb_prob_tr = hgb_mdl.predict_proba(Xtr.to_numpy())[:, 1]
    hgb_prob_va = hgb_mdl.predict_proba(Xva.to_numpy())[:, 1]
    hgb_val_auc = float(roc_auc_score(yva, hgb_prob_va))
    hgb_threshold = top_tercile_threshold(hgb_prob_tr)
    log(f'{label_col} HGB: cv_auc={hgb_cv_auc:.4f} val_auc={hgb_val_auc:.4f} '
        f'threshold(top-tercile of TRAIN-H2)={hgb_threshold:.4f}')

    # ---- LR ----
    lr_mdl, lr_imp, lr_sca = fit_lr(Xtr, ytr)
    lr_prob_tr = lr_predict_proba(lr_mdl, lr_imp, lr_sca, Xtr)
    lr_prob_va = lr_predict_proba(lr_mdl, lr_imp, lr_sca, Xva)
    lr_val_auc = float(roc_auc_score(yva, lr_prob_va))
    lr_threshold = top_tercile_threshold(lr_prob_tr)
    log(f'{label_col} LR: val_auc={lr_val_auc:.4f} threshold={lr_threshold:.4f}')

    # ---- placebo: TRAIN-H2 labels shuffled, same pipeline, applied ONCE to TRUE VAL labels ----
    ytr_shuf = shuffle_labels(ytr, seed=SEED)
    placebo_mdl, placebo_params, placebo_cv_auc, _ = fit_cv_hgb(Xtr, ytr_shuf)
    placebo_prob_tr = placebo_mdl.predict_proba(Xtr.to_numpy())[:, 1]
    placebo_prob_va = placebo_mdl.predict_proba(Xva.to_numpy())[:, 1]
    placebo_val_auc = float(roc_auc_score(yva, placebo_prob_va))
    placebo_threshold = top_tercile_threshold(placebo_prob_tr)
    placebo_kept_va = placebo_prob_va >= placebo_threshold
    whole_book_va_mean = float(va['outcome_R'].mean())
    placebo_kept_mean = float(va.loc[placebo_kept_va, 'outcome_R'].mean()) if placebo_kept_va.any() \
        else float('nan')
    placebo_ok = (placebo_val_auc <= PLACEBO_AUC_MAX and
                  abs(placebo_kept_mean - whole_book_va_mean) <= PLACEBO_MEAN_TOL)
    log(f'{label_col} PLACEBO: val_auc={placebo_val_auc:.4f} kept_mean={placebo_kept_mean:.4f} '
        f'whole_book_mean={whole_book_va_mean:.4f} ok={placebo_ok}')

    # ---- permutation importance (HGB, VAL) ----
    pi = permutation_importance(hgb_mdl, Xva.to_numpy(), yva.to_numpy(), n_repeats=10,
                                 random_state=SEED, scoring='roc_auc', n_jobs=1)
    order = np.argsort(pi.importances_mean)[::-1][:10]
    top_importances = [dict(feature=feature_cols[i], importance=float(pi.importances_mean[i]))
                        for i in order]

    # ---- scoring: kept/dropped per holdout, per model ----
    rows = []
    for model_name, prob_tr, prob_va, thr, val_auc in (
            ('HGB', hgb_prob_tr, hgb_prob_va, hgb_threshold, hgb_val_auc),
            ('LR', lr_prob_tr, lr_prob_va, lr_threshold, lr_val_auc)):
        sc_tr = score_holdout(tr, prob_tr, thr, 'outcome_R', label_col, 'store_served_1438', weeks['TRAIN'])
        sc_va = score_holdout(va, prob_va, thr, 'outcome_R', label_col, 'store_served_1438', weeks['VAL'])
        for holdout, sc, auc in (('TRAIN-H2', sc_tr, hgb_cv_auc if model_name == 'HGB' else np.nan),
                                  ('VAL', sc_va, val_auc)):
            official_pass = (model_name == 'HGB' and holdout == 'VAL' and
                              sc['kept_mean'] >= PASS_KEPT_MEAN and sc['t_kept'] >= PASS_T and
                              sc['ex_top5'] > 0 and sc['fills_wk'] >= PASS_FILLS_WK and
                              sc_tr['dropped_mean'] < sc_tr['kept_mean'] and
                              sc['dropped_mean'] < sc['kept_mean'] and
                              val_auc >= PASS_AUC and placebo_ok and not decoy_void)
            rows.append(dict(label=label_col, model=model_name, holdout=holdout, auc=float(auc),
                              passes_bar=bool(official_pass), threshold=thr, **sc))

    va_deciles = decile_table(hgb_prob_va, va['outcome_R'].to_numpy(), va[label_col].to_numpy())

    preds = pd.concat([tr.assign(hgb_prob=hgb_prob_tr, lr_prob=lr_prob_tr),
                        va.assign(hgb_prob=hgb_prob_va, lr_prob=lr_prob_va)], ignore_index=True)
    preds[f'hgb_kept_{label_col}'] = preds['hgb_prob'] >= hgb_threshold
    preds[f'lr_kept_{label_col}'] = preds['lr_prob'] >= lr_threshold
    preds = preds.rename(columns={'hgb_prob': f'hgb_prob_{label_col}',
                                   'lr_prob': f'lr_prob_{label_col}'})

    return dict(
        label=label_col, rows=rows, decoy_val_auc=decoy_val_auc, decoy_void=decoy_void,
        hgb_params=hgb_params, hgb_cv_auc=hgb_cv_auc, hgb_threshold=hgb_threshold,
        placebo_val_auc=placebo_val_auc, placebo_kept_mean=placebo_kept_mean,
        placebo_ok=placebo_ok, top_importances=top_importances, va_deciles=va_deciles,
        preds=preds[['day', 'symbol', 'fill_min', 'split', f'hgb_prob_{label_col}',
                      f'hgb_kept_{label_col}', f'lr_prob_{label_col}', f'lr_kept_{label_col}']],
    )


# ================================================================================================
# main
# ================================================================================================

def load_base_fills():
    causal = pd.read_csv(CAUSAL_CSV, low_memory=False)
    fills = causal[causal.status == 'fill'].reset_index(drop=True)
    keep_cols = ['day', 'symbol', 'split', 'fill_min', 'fill', 'stop', 'level', 'exit_m',
                 'exit_price', 'why', 'R', 'raw_R', 'cost_R', 'net_R', 'exit_half_src']
    return fills[keep_cols].copy()


def fmt_rows_md(rows):
    hdr = ('label', 'model', 'holdout', 'n_kept', 'n_dropped', 'base_rate', 'kept_mean',
           'dropped_mean', 't_kept', 'ex_top5', 'fills_wk', 'kept_bigday_rate',
           'kept_cacheonly_share', 'auc', 'passes_bar')
    lines = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    for r in rows:
        vals = []
        for k in hdr:
            v = r.get(k)
            if isinstance(v, float):
                vals.append(f'{v:.4f}')
            else:
                vals.append(str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    suffix = '_DRYRUN' if args.dry_run else ''

    t0 = time.time()
    log('cell_1478: loading base fills')
    base = load_base_fills()
    if args.dry_run:
        base = base.sample(n=min(400, len(base)), random_state=SEED).reset_index(drop=True)
        log(f'--dry-run: subsampled to {len(base)} rows')

    log('cell_1478: building labels (PIT daily bar)')
    base = build_labels(base)
    log('cell_1478: building outcome R (1457 standard + amendment slip substitution)')
    base = build_outcome(base)

    log('cell_1478: loading/merging features A+B+C')
    feats = load_and_merge_features()
    # fcols is computed from feats ALONE, before the label/outcome columns (L1, L2, outcome_R,
    # day_high/day_low/day_close, full_day_range_pct, prev_session_volume, fill/stop/exit_price/R/
    # net_R/...) are merged in -- otherwise the label itself (e.g. full_day_range_pct, which IS L1
    # thresholded) would leak into the feature matrix. feature_columns() never sees `base`.
    fcols = feature_columns(feats)
    merged = base.merge(feats, on=['day', 'symbol', 'fill_min'], how='inner', suffixes=('', '_f'))
    assert len(merged) == len(base), f'base x features merge dropped rows: {len(base)} -> {len(merged)}'
    if 'split_f' in merged.columns:
        assert (merged['split'] == merged['split_f']).all(), 'base/features split mismatch'
        merged = merged.drop(columns=['split_f'])
    assert not (set(fcols) & {'L1', 'L2', 'outcome_R', 'day_high', 'day_low', 'day_close',
                               'full_day_range_pct', 'prev_session_volume', 'fill', 'stop',
                               'level', 'exit_m', 'exit_price', 'R', 'raw_R', 'cost_R', 'net_R',
                               'old_slip_R', 'new_slip_R'}), 'label/outcome leaked into features'
    log(f'cell_1478: merged book = {len(merged)} rows')
    weeks = {s: c1445.weeks_spanned(merged.loc[merged.split == s, 'day']) for s in ('TRAIN', 'VAL')}
    log(f'weeks spanned: {weeks}')

    results = {}
    for label_col in ('L1', 'L2'):
        results[label_col] = run_label(label_col, merged, fcols, weeks)

    # ---- write predictions ----
    pred_out = merged[['day', 'symbol', 'fill_min', 'split', 'why', 'outcome_R', 'L1', 'L2',
                        'store_served_1438']].copy()
    for label_col in ('L1', 'L2'):
        p = results[label_col]['preds']
        pred_out = pred_out.merge(p, on=['day', 'symbol', 'fill_min', 'split'], how='left')
    pred_path = os.path.join(HERE, f'model_1478_predictions{suffix}.csv')
    pred_out.to_csv(pred_path, index=False)
    log(f'wrote {pred_path} ({len(pred_out)} rows)')

    # ---- RESULT.md ----
    all_rows = results['L1']['rows'] + results['L2']['rows']
    lines = ['# RESULT 1,478 -- supervised big-day predictor at the arm bar', '']
    lines.append('## Scoring (kept = prob >= TRAIN-H2 top-tercile threshold, frozen on TRAIN only)')
    lines.append('')
    lines.append(fmt_rows_md(all_rows))
    lines.append('')
    lines.append('## Decoy model (metadata-only: store_served_1438, rth_bar_count_1438, '
                  'tick_window_has_bar_j)')
    for lbl in ('L1', 'L2'):
        r = results[lbl]
        lines.append(f'* {lbl}: VAL AUC = {r["decoy_val_auc"]:.4f} '
                      f'(void if > {DECOY_VOID_AUC}) -> decoy_void = {r["decoy_void"]}')
    lines.append('')
    lines.append('## Placebo (label-shuffled TRAIN-H2, seed 1478, applied once to TRUE VAL labels)')
    for lbl in ('L1', 'L2'):
        r = results[lbl]
        lines.append(f'* {lbl}: VAL AUC = {r["placebo_val_auc"]:.4f} (pass <= {PLACEBO_AUC_MAX}); '
                      f'placebo kept mean = {r["placebo_kept_mean"]:.4f} vs whole-book VAL mean '
                      f'(diff tol {PLACEBO_MEAN_TOL}); placebo_ok = {r["placebo_ok"]}')
    lines.append('')
    lines.append('## Top-10 permutation importances (HGB, VAL, ROC AUC drop)')
    for lbl in ('L1', 'L2'):
        lines.append(f'### {lbl} (best params {results[lbl]["hgb_params"]}, '
                      f'CV AUC {results[lbl]["hgb_cv_auc"]:.4f})')
        for d in results[lbl]['top_importances']:
            lines.append(f'  - {d["feature"]}: {d["importance"]:.4f}')
    lines.append('')
    lines.append('## VAL decile table (HGB probability, report-only; decile 10 = highest prob)')
    for lbl in ('L1', 'L2'):
        lines.append(f'### {lbl}')
        lines.append('| decile | n | mean_prob | mean_R | bigday_rate |')
        lines.append('|---|---|---|---|---|')
        for d in results[lbl]['va_deciles']:
            lines.append(f'| {d["decile"]} | {d["n"]} | {d["mean_prob"]:.4f} | {d["mean_R"]:.4f} '
                          f'| {d["bigday_rate"]:.4f} |')
    lines.append('')
    lines.append(f'## Cache-only share check (base {BASE_CACHEONLY_SHARE:.3f}, tol '
                 f'+/-{CACHEONLY_TOL})')
    for r in all_rows:
        ok = abs(r['kept_cacheonly_share'] - BASE_CACHEONLY_SHARE) <= CACHEONLY_TOL \
            if not np.isnan(r['kept_cacheonly_share']) else False
        lines.append(f'* {r["label"]}/{r["model"]}/{r["holdout"]}: kept cache-only share = '
                      f'{r["kept_cacheonly_share"]:.4f} -> within tol = {ok}')
    lines.append('')
    lines.append(f'Elapsed: {time.time() - t0:.0f}s')
    result_path = os.path.join(HERE, f'RESULT_1478{suffix}.md')
    with open(result_path, 'w') as fh:
        fh.write('\n'.join(lines))
    log(f'wrote {result_path}')
    log('cell_1478: DONE')


if __name__ == '__main__':
    main()
