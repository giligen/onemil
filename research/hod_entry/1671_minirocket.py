#!/usr/bin/env python3
"""Cell 1,671: raw-sequence classifier (MiniRocket) vs the 1,670 feature GBM.

PREREG: research/hod_entry/PREREG_1671.md (FROZEN 2026-09-29 20:10 UTC).
Owner question: before any GPU time-series foundation-model fine-tune, does
a cheap raw-sequence baseline (MiniRocket -> RidgeClassifierCV) beat the
1,670 hand-crafted-feature GBM on the IDENTICAL labels and splits? If not,
no foundation model is started.

Reuses (importlib, NOT copied):
  1670_timing_map.py (as f1670): fam_cols/fam_A_cols/... (the ALL feature
    set per k), DIR_TO_SCORED_HALF (both scorings), decompose_cut (the R2
    money/cut + 1,669 saved/forgone/cost decomposition), EXIT_COST_BPS via
    its own f1668 handle. Its module-level import already loads 1668/1669
    (f1670.f1668, f1670.f1669) -- BarStore, find_fill_index convention,
    spy_close_at_or_before, iid_t, day_clustered_t, ex_top5_mean, mde,
    CUT_BPS, ENTRY_BPS, BARS_DB, minute_of_day/et_offset_minutes (DST-aware
    UTC->ET), check_disk.
  1670_wide_cache.csv: the exact 1670 population (1663 join, r_pct>=1.5%,
    halves=split) with entry/stop/level/i0 and, per k, open_k/label_stop_k/
    next_open_k/dR_cut_k/mtm_R_k and every ALL-family feature column --
    used AS-IS, not rebuilt.
  1670_map.csv: the frozen GBM-ALL AUC (+placebo) per k/direction -- the R1
    GBM comparator row, reused verbatim, GBM is NOT retrained here.
  1670_per_fill_k.csv: p_stop_ALL (GBM's OOS P(stop)) reused verbatim for
    the "GBM P" column of 1671_scores.csv.

New in this cell: the raw multivariate bar sequence per (fill, k) built
directly from bars_sip.db (not from any hand-crafted feature), MiniRocket
transform, Ridge classification, isotonic calibration for the fixed-tau
money cut, and the stacked model (1670 ALL features + MiniRocket's decision
score).

Sequence definition (PREREG "Inputs" section): bar array positions
i0-30 .. i0+k within that (symbol, day)'s bars (i0 = the fill bar, from
1670_wide_cache; SEQ_BEFORE=30 fixed by the PREREG, not tuned). When
i0 < 30 the missing early positions are LEFT-PADDED by repeating the first
REAL bar's channel values, flagged with a mask channel (1=padded, 0=real).
Channels (7): open/high/low/close in R units (x-entry)/(entry-stop); volume
divided by the day's MEAN BAR VOLUME SO FAR -- an expanding, causal mean
from the day's open (index 0) through that bar (distinct from the SPY
channel's fixed anchor "at the fill bar": volume's "so far" changes as the
window is scanned, SPY's does not); SPY close in bps relative to its value
at the fill bar (spy_close_at_or_before, same causal join 1668/1670 use);
mask (1=padded).

Model per (k, direction): MiniRocketMultivariate(num_kernels, random_state=
RNG_SEED, n_jobs=1) fit on the TRAIN half's sequences only, applied
(.transform) to the SCORE half -- never fit on scoring data. RidgeClassif-
ierCV (default alphas) on the transformed TRAIN half; AUC via
decision_function (threshold-free) on the SCORE half. Placebo: same
transformed features, TRAIN-half labels shuffled WITHIN DAY (matches
1670's placebo pattern), refit, AUC on the SCORE half's real labels. Stack:
HistGradientBoostingClassifier(max_iter=200) on [1670 ALL features +
MiniRocket's decision score as one extra column] (in-sample decision score
on the TRAIN half it was fit on, OOS decision score on the SCORE half),
same TRAIN/SCORE split. Isotonic calibration (sklearn.isotonic.
IsotonicRegression, fit on the TRAIN half's (decision_score, label) only)
maps MiniRocket's raw decision score to a probability for the fixed-tau R2
money cut (GBM's own R2 already exists in 1670_reads.csv and is not
recomputed here; R2 in this cell is MiniRocket's cut only, matching the
PREREG's stated multiplicity 5k x 4tau x 2scoring = 40, i.e. one model).

Usage:
    python3 1671_minirocket.py [--limit N] [--num-kernels N]

Outputs (research/hod_entry/):
    1671_reads.csv       -- part='R1' (model x k x direction: auc,
                             placebo_auc) and part='R2' (MiniRocket money
                             cut: k x tau x direction, decompose_cut fields)
    1671_scores.csv       -- fill_id x k: MiniRocket score (raw OOS
                              decision_function), GBM P (p_stop_ALL,
                              reused), stacked P (OOS predict_proba)
    1671_minirocket.log
    RESULT_1671.md
"""
import argparse
import importlib.util
import logging
import os
import shutil
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import roc_auc_score
from sktime.transformations.panel.rocket import MiniRocketMultivariate

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


f1670 = _load('f1670_1671', '1670_timing_map.py')
f1668 = f1670.f1668  # BarStore, spy_close_at_or_before, iid_t, day_clustered_t, ex_top5_mean, mde, CUT_BPS, BARS_DB

WIDE_CSV = os.path.join(HERE, '1670_wide_cache.csv')
MAP_CSV = os.path.join(HERE, '1670_map.csv')
PERFILLK_CSV = os.path.join(HERE, '1670_per_fill_k.csv')
READS_CSV = os.path.join(HERE, '1671_reads.csv')
SCORES_CSV = os.path.join(HERE, '1671_scores.csv')
LOG_FILE = os.path.join(HERE, '1671_minirocket.log')
RESULT_MD = os.path.join(HERE, 'RESULT_1671.md')

KS = [0, 1, 2, 5, 10]
TAUS = [0.5, 0.6, 0.7, 0.8]
SEQ_BEFORE = 30
NUM_KERNELS_DEFAULT = 10000
NUM_KERNELS_FALLBACK = 5000
SLOW_FIT_SECONDS = 90.0  # PREREG budget guard: if the single largest fit_transform
                          # exceeds this, fall back to 5,000 kernels and log it.
RNG_SEED = 1671
DIRECTIONS = list(f1670.DIR_TO_SCORED_HALF.items())  # [(direction, score_half), ...]

logger = logging.getLogger('1671')


def setup_logging():
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(levelname)s %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)


def check_disk(min_gb=5.0):
    free_gb = shutil.disk_usage('/').free / 1e9
    if free_gb < min_gb:
        logger.error('disk free %.1fGB < %.1fGB minimum -- aborting', free_gb, min_gb)
        sys.exit(1)
    logger.info('disk free: %.1fGB', free_gb)


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------

def build_sequence(bars, spy_bars, i0, k, entry, stop):
    """Multivariate sequence for one fill at horizon k. Array positions
    i0-SEQ_BEFORE .. i0+k within that (symbol,day)'s bars, left-padded at
    the day's open when i0 < SEQ_BEFORE. Returns ndarray (7, SEQ_BEFORE+k+1)
    or None if bar k does not exist yet (should not happen for open_k==True
    rows; logged if it does)."""
    n = len(bars['o'])
    hi = i0 + k
    if hi >= n:
        return None
    R = entry - stop
    if not np.isfinite(R) or R == 0:
        return None

    n_pad = max(0, SEQ_BEFORE - i0)
    lo_real = max(0, i0 - SEQ_BEFORE)
    real_idx = np.arange(lo_real, hi + 1)
    L = SEQ_BEFORE + k + 1
    if len(real_idx) + n_pad != L:
        return None

    o_r = (bars['o'][real_idx] - entry) / R
    h_r = (bars['h'][real_idx] - entry) / R
    l_r = (bars['l'][real_idx] - entry) / R
    c_r = (bars['c'][real_idx] - entry) / R

    # Volume / the day's mean bar volume SO FAR: an expanding, causal mean
    # from the day's open (index 0) through each bar -- no look-ahead.
    v_full = bars['v'][:hi + 1]
    cum_v = np.cumsum(v_full)
    counts = np.arange(1, hi + 2, dtype=float)
    running_mean_v = cum_v / counts
    vol_ratio_full = np.where(running_mean_v > 0, v_full / np.where(running_mean_v > 0, running_mean_v, 1.0), np.nan)
    vol_r = vol_ratio_full[real_idx]

    spy_at_fill = f1668.spy_close_at_or_before(spy_bars, bars['minarr'][i0]) if spy_bars is not None else None
    if spy_at_fill in (None, 0) or (isinstance(spy_at_fill, float) and pd.isna(spy_at_fill)):
        spy_bps_r = np.full(len(real_idx), np.nan)
    else:
        spy_vals = [f1668.spy_close_at_or_before(spy_bars, m) for m in bars['minarr'][real_idx]]
        spy_bps_r = np.array([((v / spy_at_fill - 1.0) * 10000.0) if v is not None else np.nan
                               for v in spy_vals])

    mask_real = np.zeros(len(real_idx))

    if n_pad > 0:
        def pad(arr):
            return np.concatenate([np.full(n_pad, arr[0]), arr])
        o_r, h_r, l_r, c_r = pad(o_r), pad(h_r), pad(l_r), pad(c_r)
        vol_r, spy_bps_r = pad(vol_r), pad(spy_bps_r)
        mask_full = np.concatenate([np.ones(n_pad), mask_real])
    else:
        mask_full = mask_real

    # Causal forward-fill of any residual SPY/volume NaN inside the real
    # span (thin-bar or SPY-gap minutes); if a whole sequence is NaN it is
    # dropped by the caller.
    vol_r = pd.Series(vol_r).ffill().bfill().values
    spy_bps_r = pd.Series(spy_bps_r).ffill().bfill().values

    return np.vstack([o_r, h_r, l_r, c_r, vol_r, spy_bps_r, mask_full]).astype(np.float64)


def build_all_sequences(wide, store, ks, limit=None):
    """One pass over `wide` (one row per fill): fetch bars once per fill,
    build the length-(SEQ_BEFORE+k+1) sequence for every k in `ks` where
    open_k is True. Returns (seqs: {k: {fill_id: ndarray}}, n_rows_no_bars,
    n_seq_failed: {k: count})."""
    seqs = {k: {} for k in ks}
    n_no_bars = 0
    n_failed = {k: 0 for k in ks}
    day_cache = {}
    rows = wide if limit is None else wide.iloc[:limit]
    t0 = time.time()
    for i, row in enumerate(rows.itertuples()):
        if (i + 1) % 1000 == 0:
            logger.info('sequence build: %d/%d fills (%.1fs elapsed)', i + 1, len(rows), time.time() - t0)
        key = (row.symbol, row.date)
        if key not in day_cache:
            day_cache[key] = store.day_bars(row.symbol, row.date)
        bars = day_cache[key]
        if bars is None or not bool(row.i0_found) or pd.isna(row.i0):
            n_no_bars += 1
            continue
        i0 = int(row.i0)
        spy_bars = store.spy_bars(row.date)
        entry, stop = float(row.entry), float(row.stop)
        for k in ks:
            if not bool(getattr(row, f'open_{k}')):
                continue
            seq = build_sequence(bars, spy_bars, i0, k, entry, stop)
            if seq is None or not np.isfinite(seq).all():
                n_failed[k] += 1
                continue
            seqs[k][row.fill_id] = seq
    logger.info('sequence build done: %d fills, %d with no bars/i0, per-k build failures=%s, %.1fs',
                len(rows), n_no_bars, n_failed, time.time() - t0)
    return seqs, n_no_bars, n_failed


# ---------------------------------------------------------------------------
# Per-k, per-direction model fits
# ---------------------------------------------------------------------------

def within_day_shuffle(y, dates, seed):
    rng = np.random.RandomState(seed)
    return pd.Series(y.values, index=y.index).groupby(dates.values).transform(
        lambda s: rng.permutation(s.values))


def fit_score_k(wide, seqs_k, k, num_kernels, rng_seed):
    """Both scorings for one k. Returns list of R1 read dicts, and dicts
    keyed by direction: mr_raw_oos (Series fill_id->raw decision score),
    mr_calibrated_oos (Series fill_id->isotonic-calibrated P), stack_p_oos
    (Series fill_id->stack predict_proba)."""
    label_col = f'label_stop_{k}'
    idx = wide.set_index('fill_id')
    reads = []
    mr_raw_by_dir, mr_cal_by_dir, stack_p_by_dir = {}, {}, {}

    for direction, score_half in DIRECTIONS:
        train_half = 'TRAIN-H2' if score_half == 'VAL' else 'VAL'
        fids_train = [fid for fid in idx.index[idx['split'] == train_half] if fid in seqs_k]
        fids_score = [fid for fid in idx.index[idx['split'] == score_half] if fid in seqs_k]
        ytr = idx.loc[fids_train, label_col].astype(int)
        ysc = idx.loc[fids_score, label_col].astype(int)
        logger.info('k=%d %s: MiniRocket train n=%d (%.2f%% pos) score n=%d (%.2f%% pos)',
                    k, direction, len(ytr), 100 * ytr.mean() if len(ytr) else np.nan,
                    len(ysc), 100 * ysc.mean() if len(ysc) else np.nan)
        if len(fids_train) == 0 or len(fids_score) == 0 or ytr.nunique() < 2:
            logger.warning('k=%d %s: skipped (empty half or single-class train)', k, direction)
            reads.append(dict(part='R1', model='MiniRocket', k=k, direction=direction,
                               n_train=len(fids_train), n_score=len(fids_score), auc=np.nan, placebo_auc=np.nan))
            reads.append(dict(part='R1', model='Stack', k=k, direction=direction,
                               n_train=len(fids_train), n_score=len(fids_score), auc=np.nan, placebo_auc=np.nan))
            continue

        Xtr = np.stack([seqs_k[f] for f in fids_train])
        Xsc = np.stack([seqs_k[f] for f in fids_score])

        t0 = time.time()
        mr = MiniRocketMultivariate(num_kernels=num_kernels, random_state=rng_seed, n_jobs=1)
        Xtr_t = mr.fit_transform(Xtr)
        fit_secs = time.time() - t0
        Xsc_t = mr.transform(Xsc)
        logger.info('k=%d %s: MiniRocket(kernels=%d) fit_transform %d insts %.1fs, transform %d insts',
                    k, direction, num_kernels, len(fids_train), fit_secs, len(fids_score))

        clf = RidgeClassifierCV()
        clf.fit(Xtr_t, ytr.values)
        dtr = clf.decision_function(Xtr_t)
        dsc = clf.decision_function(Xsc_t)
        auc = roc_auc_score(ysc.values, dsc) if ysc.nunique() > 1 else np.nan

        y_shuf = within_day_shuffle(ytr, idx.loc[fids_train, 'date'], seed=rng_seed + k)
        placebo_auc = np.nan
        if pd.Series(y_shuf).nunique() > 1 and ysc.nunique() > 1:
            clf_pl = RidgeClassifierCV()
            clf_pl.fit(Xtr_t, y_shuf)
            dsc_pl = clf_pl.decision_function(Xsc_t)
            placebo_auc = roc_auc_score(ysc.values, dsc_pl)
        reads.append(dict(part='R1', model='MiniRocket', k=k, direction=direction,
                           n_train=len(fids_train), n_score=len(fids_score), auc=auc, placebo_auc=placebo_auc))

        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(dtr, ytr.values)
        p_cal_sc = iso.predict(dsc)
        mr_raw_by_dir[direction] = pd.Series(dsc, index=fids_score)
        mr_cal_by_dir[direction] = pd.Series(p_cal_sc, index=fids_score)

        all_cols = f1670.fam_cols(wide, 'ALL', k)
        Xtr_all = wide.set_index('fill_id').loc[fids_train, all_cols].astype(float).copy()
        Xsc_all = wide.set_index('fill_id').loc[fids_score, all_cols].astype(float).copy()
        Xtr_all['minirocket_score'] = dtr
        Xsc_all['minirocket_score'] = dsc
        gbm = HistGradientBoostingClassifier(max_iter=200, random_state=rng_seed)
        gbm.fit(Xtr_all.values, ytr.values)
        p_stack_sc = gbm.predict_proba(Xsc_all.values)[:, 1]
        auc_stack = roc_auc_score(ysc.values, p_stack_sc) if ysc.nunique() > 1 else np.nan
        stack_p_by_dir[direction] = pd.Series(p_stack_sc, index=fids_score)
        reads.append(dict(part='R1', model='Stack', k=k, direction=direction,
                           n_train=len(fids_train), n_score=len(fids_score), auc=auc_stack, placebo_auc=np.nan))

    return reads, mr_raw_by_dir, mr_cal_by_dir, stack_p_by_dir


def run_r2_k(wide, k, mr_cal_by_dir):
    """MiniRocket's fixed-tau money cut, both scorings -- reuses f1670's
    decompose_cut (the 1,669 saved/forgone/cost decomposition) verbatim."""
    idx = wide.set_index('fill_id')
    reads = []
    for direction, score_half in DIRECTIONS:
        p_cal = mr_cal_by_dir.get(direction)
        if p_cal is None or len(p_cal) == 0:
            continue
        cols = ['date', 'entry', 'stop', f'next_open_{k}', f'label_stop_{k}', f'dR_cut_{k}', f'mtm_R_{k}']
        sub = idx.loc[p_cal.index, cols].copy()
        sub = sub.rename(columns={f'next_open_{k}': 'next_open', f'label_stop_{k}': 'y',
                                   f'dR_cut_{k}': 'dR', f'mtm_R_{k}': 'mtm_R'})
        sub['y'] = sub['y'].astype(int)
        sub['p'] = p_cal.values
        for tau in TAUS:
            res = f1670.decompose_cut(sub, tau, 'base')
            res.update(part='R2', model='MiniRocket', k=k, direction=direction, tau=tau)
            reads.append(res)
    return reads


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return 'nan'
    return f'{x:.{nd}f}'


def write_result_md(reads_df, gbm_map, n_pop, num_kernels, decision_rows):
    lines = []
    lines.append('# RESULT 1,671 -- raw-sequence (MiniRocket) vs the 1,670 feature GBM\n')
    lines.append(f'PREREG: PREREG_1671.md (FROZEN). Population n={n_pop} (1670\'s 1663 join, r_pct>=1.5%). '
                 f'MiniRocket num_kernels={num_kernels}.\n')

    lines.append('## R1 -- out-of-sample AUC per k and scoring\n')
    lines.append('| k | direction | GBM (1670) | GBM placebo | MiniRocket | MR placebo | Stack |')
    lines.append('|---|---|---|---|---|---|---|')
    r1 = reads_df[reads_df['part'] == 'R1']
    for k in KS:
        for direction, _ in DIRECTIONS:
            g = gbm_map[(gbm_map['k'] == k) & (gbm_map['direction'] == direction)]
            g_auc = g['auc'].iloc[0] if len(g) else np.nan
            g_pl = g['placebo_auc'].iloc[0] if len(g) else np.nan
            mr = r1[(r1['k'] == k) & (r1['direction'] == direction) & (r1['model'] == 'MiniRocket')]
            st = r1[(r1['k'] == k) & (r1['direction'] == direction) & (r1['model'] == 'Stack')]
            mr_auc = mr['auc'].iloc[0] if len(mr) else np.nan
            mr_pl = mr['placebo_auc'].iloc[0] if len(mr) else np.nan
            st_auc = st['auc'].iloc[0] if len(st) else np.nan
            lines.append(f'| {k} | {direction} | {fmt(g_auc)} | {fmt(g_pl)} | {fmt(mr_auc)} | '
                         f'{fmt(mr_pl)} | {fmt(st_auc)} |')

    lines.append('\n## R2 -- MiniRocket fixed-tau money cut (calibrated P, both scorings)\n')
    lines.append('| k | direction | tau | n_fired | share | mean dR | iid t | day t | ex-top5 dR | MDE | '
                 'achieved prec | breakeven prec |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    r2 = reads_df[reads_df['part'] == 'R2']
    for _, r in r2.iterrows():
        lines.append(f"| {int(r['k'])} | {r['direction']} | {r['tau']} | {int(r.get('n_fired', 0) or 0)} | "
                     f"{fmt(r.get('share_cut'))} | {fmt(r.get('mean_dR'))} | {fmt(r.get('iid_t'))} | "
                     f"{fmt(r.get('day_t'))} | {fmt(r.get('ex_top5_dR'))} | {fmt(r.get('mde'))} | "
                     f"{fmt(r.get('achieved_precision'))} | {fmt(r.get('breakeven_precision'))} |")

    lines.append('\n## Pass bar (PREREG)\n')
    lines.append('R2 ships to paper only if paired dR >= +0.05 R, t >= 2.5, ex-top-5% dR > 0 on BOTH scorings, '
                 'placebo <= 0.55 AUC. Foundation-model GO only if MiniRocket or Stack beats GBM by >= 0.03 AUC '
                 'on BOTH scorings at some k AND achieved precision at that k is within 0.10 of break-even.\n')
    for row in decision_rows:
        lines.append(f'- {row}')

    with open(RESULT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    logger.info('wrote %s (%d lines)', RESULT_MD, len(lines))


def compute_decision(reads_df, gbm_map):
    """PREREG foundation-model decision: does MiniRocket or Stack beat GBM
    by >=0.03 AUC on BOTH scorings at some k, AND is achieved precision at
    that k within 0.10 of break-even (checked against the R2 tau grid for
    that k, MiniRocket only -- Stack has no R2 cut in this cell)."""
    r1 = reads_df[reads_df['part'] == 'R1']
    r2 = reads_df[reads_df['part'] == 'R2']
    lines = []
    qualifying = []
    for model in ('MiniRocket', 'Stack'):
        for k in KS:
            gains = {}
            for direction, _ in DIRECTIONS:
                g = gbm_map[(gbm_map['k'] == k) & (gbm_map['direction'] == direction)]
                m = r1[(r1['k'] == k) & (r1['direction'] == direction) & (r1['model'] == model)]
                if len(g) == 0 or len(m) == 0 or pd.isna(g['auc'].iloc[0]) or pd.isna(m['auc'].iloc[0]):
                    continue
                gains[direction] = m['auc'].iloc[0] - g['auc'].iloc[0]
            if len(gains) == 2 and all(v >= 0.03 for v in gains.values()):
                qualifying.append((model, k, gains))
                lines.append(f'{model} k={k}: AUC gain {[fmt(v) for v in gains.values()]} (>=0.03 both scorings)')
    if not qualifying:
        lines.append('No (model, k) clears the >=0.03-AUC-on-both-scorings bar. '
                      'Foundation-model decision: NO.')
        return lines, False
    any_precision_ok = False
    for model, k, gains in qualifying:
        r2k = r2[r2['k'] == k]
        for _, r in r2k.iterrows():
            ap, bp = r.get('achieved_precision'), r.get('breakeven_precision')
            if pd.notna(ap) and pd.notna(bp) and abs(ap - bp) <= 0.10:
                any_precision_ok = True
                lines.append(f'{model} k={k} tau={r["tau"]} {r["direction"]}: achieved precision {fmt(ap)} '
                             f'vs break-even {fmt(bp)} (within 0.10) -- precision condition MET')
    decision = any_precision_ok
    lines.append(f'Foundation-model decision: {"YES" if decision else "NO"}')
    return lines, decision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='limit population rows (smoke test)')
    ap.add_argument('--num-kernels', type=int, default=None, help='override MiniRocket num_kernels')
    args = ap.parse_args()

    setup_logging()
    check_disk(min_gb=5.0)
    logger.info('=== cell 1,671: raw-sequence (MiniRocket) vs feature GBM ===')

    wide = pd.read_csv(WIDE_CSV, dtype={'date': str, 'symbol': str})
    wide = f1670.normalize_bool_cols(wide)
    n_pop = len(wide)
    logger.info('loaded %s: %d rows, split=%s', WIDE_CSV, n_pop, dict(wide['split'].value_counts()))
    for k in KS:
        oc = wide[f'open_{k}']
        logger.info('k=%d: open=%d/%d (%.1f%%)', k, oc.sum(), n_pop, 100 * oc.mean())

    store = f1668.BarStore(f1668.BARS_DB)

    num_kernels = args.num_kernels or NUM_KERNELS_DEFAULT
    if args.num_kernels is None:
        # Pre-flight timing probe on the largest k (most timepoints, most
        # instances) before committing the kernel count for the whole run.
        probe_k = max(KS)
        probe_seqs, _, _ = build_all_sequences(wide, store, [probe_k], limit=args.limit)
        fids = list(probe_seqs[probe_k].keys())
        train_fids = [f for f in fids if wide.set_index('fill_id').loc[f, 'split'] == 'TRAIN-H2']
        if len(train_fids) >= 20:
            Xprobe = np.stack([probe_seqs[probe_k][f] for f in train_fids])
            t0 = time.time()
            MiniRocketMultivariate(num_kernels=NUM_KERNELS_DEFAULT, random_state=RNG_SEED,
                                    n_jobs=1).fit_transform(Xprobe)
            probe_secs = time.time() - t0
            logger.info('kernel-count probe: k=%d n=%d kernels=%d took %.1fs',
                        probe_k, len(train_fids), NUM_KERNELS_DEFAULT, probe_secs)
            if probe_secs > SLOW_FIT_SECONDS:
                logger.warning('probe fit_transform %.1fs > %.0fs budget guard -- falling back to '
                                'num_kernels=%d for the whole run', probe_secs, SLOW_FIT_SECONDS,
                                NUM_KERNELS_FALLBACK)
                num_kernels = NUM_KERNELS_FALLBACK
        del probe_seqs

    seqs, n_no_bars, n_failed = build_all_sequences(wide, store, KS, limit=args.limit)
    store.close()

    gbm_map = pd.read_csv(MAP_CSV)
    gbm_map = gbm_map[(gbm_map['family'] == 'ALL') & (gbm_map['k'].isin(KS))]

    per_fill_k = pd.read_csv(PERFILLK_CSV)  # fill_id,k,open,label_stop,mtm_R,p_stop_* -- GBM P reused as-is

    all_reads = []
    scores_rows = []
    for k in KS:
        logger.info('--- k=%d: MiniRocket + Stack fit/score ---', k)
        reads_k, mr_raw, mr_cal, stack_p = fit_score_k(wide, seqs[k], k, num_kernels, RNG_SEED)
        all_reads.extend(reads_k)
        r2_k = run_r2_k(wide, k, mr_cal)
        all_reads.extend(r2_k)

        mr_raw_all = pd.concat(mr_raw.values()) if mr_raw else pd.Series(dtype=float)
        stack_p_all = pd.concat(stack_p.values()) if stack_p else pd.Series(dtype=float)
        gbm_p_k = per_fill_k[per_fill_k['k'] == k].set_index('fill_id')['p_stop_ALL']
        fids_k = sorted(set(mr_raw_all.index) | set(stack_p_all.index) | set(gbm_p_k.index))
        for fid in fids_k:
            scores_rows.append(dict(fill_id=fid, k=k,
                                     minirocket_score=mr_raw_all.get(fid, np.nan),
                                     gbm_p=gbm_p_k.get(fid, np.nan),
                                     stacked_p=stack_p_all.get(fid, np.nan)))

        pd.DataFrame(all_reads).to_csv(READS_CSV, index=False)  # atomic-ish incremental write
        pd.DataFrame(scores_rows).to_csv(SCORES_CSV, index=False)
        logger.info('k=%d done, incremental write: %s (%d rows), %s (%d rows)',
                    k, READS_CSV, len(all_reads), SCORES_CSV, len(scores_rows))

    reads_df = pd.DataFrame(all_reads)
    reads_df.to_csv(READS_CSV, index=False)
    pd.DataFrame(scores_rows).to_csv(SCORES_CSV, index=False)
    logger.info('final write: %s (%d rows), %s (%d rows)', READS_CSV, len(reads_df), SCORES_CSV, len(scores_rows))

    decision_lines, decision = compute_decision(reads_df, gbm_map)
    for line in decision_lines:
        logger.info(line)
    write_result_md(reads_df, gbm_map, n_pop, num_kernels, decision_lines)
    logger.info('=== done. foundation-model decision: %s ===', 'YES' if decision else 'NO')


if __name__ == '__main__':
    main()
