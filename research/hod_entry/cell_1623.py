#!/usr/bin/env python3
"""Cell 1,623 -- research/hod_entry/PREREG_1623.md (FROZEN 2026-09-28 17:15 UTC).

The day's own feedback (causal intraday gate). For each fill f on day d:
    F(f)     = mean outcome_R of day d's OTHER fills whose OWN exit minute is strictly
               before f's fill minute (resolved before the decision)
    n_res(f) = count of such resolved fills
Gate G+ (the candidate rule): trade f only if n_res >= 2 and F >= +0.5 R.
Gate G- (the mirror, report-only): trade f only if n_res >= 2 and F <= -0.5 R.
Remainder: n_res >= 2 and -0.5 < F < 0.5. Insufficient: n_res < 2 (F not tradeable either way).

Base population: the SAME 9,911 fills as cells 1,438 / 1,445 / 1,481 (status == 'fill',
VAL union TRAIN-H2 by half; TEST is sealed and never read by this script).

Sources and the exit-minute substitution (disclosed, not silent):
  * causal_arming_causal.csv -- day, symbol, split, half, fill_min, exit_m, status. This is
    "base fills" per the PREREG. Its own exit_m is COMPLETE (0/9,911 missing) and is the one
    used for F/n_res (see note below for why, not model_1478_L3_predictions.csv's rebuild).
  * model_1478_L3_predictions.csv -- outcome_R (cell 1,478's standard-cost net R). This is the
    ONLY outcome variable used (matched 1:1 on day+symbol+fill_min, 9,911/9,911, 0 unmatched).
    It is NOT the same number as causal_arming_causal.csv's own net_R (corr 0.996 but only
    0.14% exactly equal, mean |diff| 0.085 R -- a different cost-model variant; see cell
    1,445's docstring on the double-charged entry half-spread in the raw CSV's cost_R).
  * features_1478_A.csv -- store_served_1438 (the cache-only flag), matched 1:1 on the same
    key, 9,911/9,911. Cross-checked at runtime against model_1478_L3_predictions.csv's OWN
    store_served_1438 column: the two agree on all 9,911 rows (an independent-check freebie).

  Exit-minute source note: the PREREG (and this cell's task) point at
  model_1478_L3_predictions.csv for BOTH outcome_R and the exit minute. That file has no exit_m
  column at all (see the header check logged below). The two fallbacks offered are (1) cell
  1,481's base outcome builder and (2) rebuild_1481_fills.csv's exit_m column. Diagnostic run
  before writing this script: rebuild_1481_fills.csv's status == 'fill' subset covers only
  8,973/9,911 of the base rows (the other 938 are the INDEPENDENT rebuild's own
  bar_tick_disagree / no_retest population -- i.e. cell 1,481's second, from-prose
  reimplementation used for its own independent check, not guaranteed row-for-row identical to
  the production base book) and where both exist, the two exit_m values agree on only
  1,752-vs-8,973 = 80.5% of rows (small few-minute drifts, e.g. 723 vs 717). Using it would
  both drop 9.5% of the base book AND import a cross-file re-derivation gap. causal_arming_causal.csv's OWN exit_m -- fallback (1), the original base-rule builder's own column,
  paired with the SAME row's fill_min -- is complete and internally consistent, so it is the
  one used throughout. The runtime cross-reference numbers are logged and reported in
  RESULT_1623.md so this choice stays auditable.

Report per holdout (TRAIN-H2, VAL): n kept (G+), mean net R, day-clustered t, ex-top-5 %,
fills/week at the live 12/day-4-concurrent slotting, dropped mean (kept-vs-dropped delta),
the kept cache-only share, the G-/remainder/insufficient buckets beside, a 5-bin
autocorrelation table of outcome_R vs F, and the shuffle placebo: the SAME gate applied to the
OTHER holdout's days with a random day-to-day permutation (seed 1623) substituted as the
"day's own context" -- margin (this holdout's real kept mean minus the shuffled-other kept
mean) and t (day-clustered t of the shuffled-other kept sample's own mean, the same idiom
`day_clustered_t` uses everywhere else in this line: a t on ONE sample's mean, not a two-sample
test -- the PREREG names "margin and t" as the placebo's two outputs and every other "t" in
this programme attaches to a single kept-sample mean, so this cell follows that convention).

Usage:
    python3 research/hod_entry/cell_1623.py

Outputs: research/hod_entry/cell_1623_fills.csv (day, symbol, split, F, n_res, gate_plus,
gate_minus, outcome_R -- one row per base fill) and research/hod_entry/RESULT_1623.md (the
pass-bar table, the bucket/autocorrelation/placebo detail, and the verdict).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from research.hod_entry import cell_1445 as h1445   # noqa: E402  (day_clustered_t, ex_top5_mean, weeks_spanned, fills_per_week)

FILLS_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
PRED_CSV = os.path.join(HERE, 'model_1478_L3_predictions.csv')
FEAT_CSV = os.path.join(HERE, 'features_1478_A.csv')
REBUILD_CSV = os.path.join(HERE, 'rebuild_1481_fills.csv')
OUT_FILLS_CSV = os.path.join(HERE, 'cell_1623_fills.csv')
OUT_RESULT_MD = os.path.join(HERE, 'RESULT_1623.md')

GATE_PLUS_THRESH = 0.5
GATE_MINUS_THRESH = -0.5
MIN_N_RES = 2
SHUFFLE_SEED = 1623
CACHE_ONLY_BASELINE_PCT = 19.5
CACHE_ONLY_TOL_PP = 5.0

# frozen pass bar (PREREG_1623.md lines 36-39), scored on VAL, TRAIN-H2 as corroboration
BAR_KEPT_MEAN_VAL = 0.15
BAR_T_VAL = 2.5
BAR_FILLS_WK_VAL = 3.0
BAR_TRAIN_T = 1.0
BAR_PLACEBO_MARGIN = 0.10
BAR_PLACEBO_T = 2.0


def log(msg):
    """Verbose progress line, flushed immediately (print() is buffered otherwise)."""
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# --------------------------------------------------------------------------------------------
# Step 0: base book + merges (causality trace logged at every join)
# --------------------------------------------------------------------------------------------

def load_base_book():
    """The 9,911 fills of cell 1,438: status == 'fill', TRAIN-H2 + VAL (TEST excluded, never
    read). Adds a `holdout` column ('TRAIN-H2' / 'VAL'). Identical filter to cell_1445's
    load_base_book() -- same fixed population, never recomputed."""
    df = pd.read_csv(FILLS_CSV, low_memory=False)
    f = df[df.status == 'fill'].copy()
    keep = (f.split == 'VAL') | ((f.split == 'TRAIN') & (f.half == 'H2'))
    f = f[keep].copy()
    f['holdout'] = np.where(f.split == 'VAL', 'VAL', 'TRAIN-H2')
    n = len(f)
    if n != 9911:
        log(f'WARNING base book has {n} rows, PREREG expects 9,911 -- proceeding, but flag this')
    else:
        log(f'base book: {n} rows (matches the known 9,911-fill population)')
    return f[['day', 'symbol', 'split', 'half', 'holdout', 'fill_min', 'exit_m']].reset_index(drop=True)


def merge_outcome_and_flags(base):
    """Merge in outcome_R (model_1478_L3_predictions.csv) and store_served_1438
    (features_1478_A.csv), both on (day, symbol, fill_min). Any unmatched row is EXCLUDED and
    counted (never imputed) with a WARNING."""
    pred = pd.read_csv(PRED_CSV, low_memory=False)
    if 'exit_m' in pred.columns:
        log('NOTE model_1478_L3_predictions.csv DOES carry exit_m after all -- using it would '
            'change the exit-minute source; investigate before trusting this run')
    else:
        log('confirmed: model_1478_L3_predictions.csv has NO exit_m column '
            f'(columns: {list(pred.columns)}) -- exit_m stays sourced from causal_arming_causal.csv')

    df = base.merge(pred[['day', 'symbol', 'fill_min', 'split', 'outcome_R', 'store_served_1438']],
                     on=['day', 'symbol', 'fill_min'], how='left', suffixes=('', '_pred'))
    n_unmatched = df.outcome_R.isna().sum()
    if n_unmatched:
        log(f'WARNING {n_unmatched} base rows have no outcome_R match in model_1478_L3_predictions.csv -- EXCLUDED')
        df = df[df.outcome_R.notna()].copy()
    else:
        log(f'outcome_R matched for all {len(df)} base rows (0 unmatched)')

    split_mismatch = (df.split != df.split_pred).sum()
    if split_mismatch:
        log(f'WARNING {split_mismatch} rows have split != model_1478_L3_predictions.csv split -- EXCLUDED')
        df = df[df.split == df.split_pred].copy()
    else:
        log('split label agrees between causal_arming_causal.csv and model_1478_L3_predictions.csv for every row')
    df = df.drop(columns=['split_pred']).rename(columns={'store_served_1438': 'store_served_1438_pred'})

    feat = pd.read_csv(FEAT_CSV, low_memory=False)
    df = df.merge(feat[['day', 'symbol', 'fill_min', 'store_served_1438']],
                   on=['day', 'symbol', 'fill_min'], how='left')
    n_unmatched_feat = df.store_served_1438.isna().sum()
    if n_unmatched_feat:
        log(f'WARNING {n_unmatched_feat} base rows have no store_served_1438 match in features_1478_A.csv -- EXCLUDED')
        df = df[df.store_served_1438.notna()].copy()
    else:
        log(f'store_served_1438 (cache-only flag) matched for all {len(df)} base rows via features_1478_A.csv')

    agree = (df.store_served_1438 == df.store_served_1438_pred).mean()
    log(f'independent check: features_1478_A.csv store_served_1438 agrees with '
        f"model_1478_L3_predictions.csv's own store_served_1438 on {agree * 100:.2f}% of rows")
    return df.drop(columns=['store_served_1438_pred']).reset_index(drop=True)


def independent_check_exit_m(df):
    """Diagnostic only (does not alter df): how well would rebuild_1481_fills.csv's exit_m --
    fallback (2) -- have covered and agreed with the exit_m actually used. Logged and reported
    so the source choice in the module docstring is auditable, not asserted."""
    reb = pd.read_csv(REBUILD_CSV, low_memory=False)
    reb_f = reb[reb.status == 'fill'][['day', 'symbol', 'fill_min', 'exit_m']].rename(
        columns={'exit_m': 'exit_m_rebuild'})
    chk = df.merge(reb_f, on=['day', 'symbol', 'fill_min'], how='left')
    n_present = int(chk.exit_m_rebuild.notna().sum())
    both = chk.dropna(subset=['exit_m_rebuild'])
    match_pct = float((both.exit_m == both.exit_m_rebuild).mean() * 100) if len(both) else float('nan')
    log(f'independent-check cross-reference: rebuild_1481_fills.csv exit_m present for '
        f'{n_present}/{len(df)} base rows; exact match {match_pct:.1f}% where present '
        f'(the exit_m actually used is causal_arming_causal.csv\'s own column, unchanged by this check)')
    return dict(n_present=n_present, n_total=len(df), match_pct=match_pct)


def validate_invariants(df):
    """Runtime causality checks -- never assumed, always logged. Any violation is EXCLUDED and
    counted (never imputed)."""
    bad = df[df.exit_m < df.fill_min]
    if len(bad):
        log(f'WARNING {len(bad)} rows have exit_m < fill_min (violates exit-after-entry) -- EXCLUDED')
        df = df[df.exit_m >= df.fill_min].copy()
    else:
        log('OK: exit_m >= fill_min holds for every row (exit-after-entry causality invariant)')

    mix = df.groupby('day')['holdout'].nunique()
    n_mixed = int((mix > 1).sum())
    if n_mixed:
        log(f'WARNING {n_mixed} days have fills in BOTH holdouts (F is still computed per-day '
            'correctly across the whole base book; each row keeps its own holdout label for reporting)')
    else:
        log('OK: every day maps to exactly one holdout (TRAIN-H2 xor VAL) -- no day-cohort mixing')
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Step 1: the causal intraday feedback F(f), n_res(f)
# --------------------------------------------------------------------------------------------

def compute_feedback(df):
    """F(f), n_res(f) for every fill f: mean outcome_R / count of the SAME day's OTHER fills
    whose OWN exit_m is strictly before f's fill_min. Vectorized per day (each day is a small
    DxD comparison; 9,911 rows total). Grouped over the WHOLE base book (both holdouts) since a
    day never mixes holdouts (validated above) and F is inherently a within-day quantity."""
    n = len(df)
    F = np.full(n, np.nan)
    n_res = np.zeros(n, dtype=int)
    fm_all = df['fill_min'].to_numpy()
    em_all = df['exit_m'].to_numpy()
    oc_all = df['outcome_R'].to_numpy()
    n_days = 0
    for _, idx in df.groupby('day').indices.items():
        n_days += 1
        idx = np.asarray(idx)
        fm, em, oc = fm_all[idx], em_all[idx], oc_all[idx]
        resolved = em[None, :] < fm[:, None]          # [i, j]: j resolved strictly before i's fill
        np.fill_diagonal(resolved, False)              # defensive self-exclude (already implied: exit_m[i] >= fill_min[i])
        nr = resolved.sum(axis=1)
        with np.errstate(invalid='ignore'):
            fsum = resolved @ oc
            fmean = np.where(nr > 0, fsum / np.maximum(nr, 1), np.nan)
        F[idx] = fmean
        n_res[idx] = nr
    df = df.copy()
    df['F'] = F
    df['n_res'] = n_res
    log(f'F/n_res computed over {n_days} distinct days, {n} fills; '
        f'n_res>=1 (F defined): {(df.n_res >= 1).sum()}, n_res>=2 (gateable): {(df.n_res >= 2).sum()}')
    return df


def assign_gates(df):
    """G+ (candidate rule), G- (mirror, report-only), remainder (n_res>=2, |F|<0.5),
    insufficient (n_res<2). Mutually exclusive and exhaustive -- asserted, not assumed."""
    df = df.copy()
    df['gate_plus'] = ((df.n_res >= MIN_N_RES) & (df.F >= GATE_PLUS_THRESH)).astype(int)
    df['gate_minus'] = ((df.n_res >= MIN_N_RES) & (df.F <= GATE_MINUS_THRESH)).astype(int)
    df['remainder'] = ((df.n_res >= MIN_N_RES) & (df.F > GATE_MINUS_THRESH) & (df.F < GATE_PLUS_THRESH)).astype(int)
    df['insufficient'] = (df.n_res < MIN_N_RES).astype(int)
    total_flag = df.gate_plus + df.gate_minus + df.remainder + df.insufficient
    bad = (total_flag != 1).sum()
    if bad:
        log(f'ERROR {bad} rows do not fall into exactly one bucket -- gate logic bug, investigate before reading any number')
    else:
        log('OK: gate_plus/gate_minus/remainder/insufficient partition every row exactly once')
    log(f'bucket counts (both holdouts): G+={df.gate_plus.sum()} G-={df.gate_minus.sum()} '
        f'remainder={df.remainder.sum()} insufficient={df.insufficient.sum()}')
    return df


# --------------------------------------------------------------------------------------------
# Step 2: per-holdout statistics
# --------------------------------------------------------------------------------------------

def bucket_stats(subset):
    """n, mean, day-clustered t, ex-top-5 % for one bucket (any subset of rows)."""
    n = len(subset)
    if n == 0:
        return dict(n=0, mean=float('nan'), t=float('nan'), ex_top5=float('nan'))
    return dict(n=n, mean=float(subset.outcome_R.mean()),
                t=h1445.day_clustered_t(subset.outcome_R, subset.day),
                ex_top5=h1445.ex_top5_mean(subset.outcome_R))


def holdout_breakdown(df, holdout_name):
    """The full per-holdout row: G+ (kept) vs dropped (G- + remainder + insufficient), the
    four buckets individually, fills/wk at 12/4, and the kept cache-only share."""
    hdf = df[df.holdout == holdout_name]
    weeks = h1445.weeks_spanned(hdf.day)
    kept = hdf[hdf.gate_plus == 1]
    dropped = hdf[hdf.gate_plus == 0]

    kept_stats = bucket_stats(kept)
    dropped_stats = bucket_stats(dropped)
    fwk = h1445.fills_per_week(kept, weeks) if len(kept) else 0.0
    cache_share = float(kept.store_served_1438.mean() * 100) if len(kept) else float('nan')

    buckets = {
        'G+': kept_stats,
        'G-': bucket_stats(hdf[hdf.gate_minus == 1]),
        'remainder': bucket_stats(hdf[hdf.remainder == 1]),
        'insufficient (n_res<2)': bucket_stats(hdf[hdf.insufficient == 1]),
    }
    return dict(holdout=holdout_name, weeks=weeks, n_kept=kept_stats['n'], kept_mean=kept_stats['mean'],
                t_kept=kept_stats['t'], ex_top5=kept_stats['ex_top5'], fills_wk=fwk,
                n_dropped=dropped_stats['n'], dropped_mean=dropped_stats['mean'],
                delta_R=kept_stats['mean'] - dropped_stats['mean'] if len(kept) and len(dropped) else float('nan'),
                cache_only_share_pct=cache_share, buckets=buckets)


def autocorr_table(df, holdout_name, n_bins=5):
    """5-bin table of outcome_R vs F, over rows where F is defined (n_res >= 1), within one
    holdout. Quantile bins on F (duplicates dropped if F has ties collapsing a bin edge)."""
    hdf = df[(df.holdout == holdout_name) & df.F.notna()].copy()
    if len(hdf) < n_bins:
        log(f'WARNING {holdout_name}: only {len(hdf)} rows with F defined, cannot form {n_bins} bins')
        return []
    hdf['bin'] = pd.qcut(hdf.F, n_bins, labels=False, duplicates='drop')
    rows = []
    for b in sorted(hdf.bin.unique()):
        sub = hdf[hdf.bin == b]
        rows.append(dict(bin=int(b) + 1, n=len(sub), f_min=float(sub.F.min()), f_max=float(sub.F.max()),
                          f_mean=float(sub.F.mean()), outcome_mean=float(sub.outcome_R.mean()),
                          t=h1445.day_clustered_t(sub.outcome_R, sub.day)))
    return rows


def shuffle_placebo(df, holdout_name, seed=SHUFFLE_SEED):
    """The G+ gate applied to the OTHER holdout, with each fill's 'day context' substituted by
    a RANDOMLY PERMUTED day within that other holdout (seed-fixed, one draw). outcome_R stays
    the fill's own real number; only the source of F's resolved-context is shuffled -- this
    tests whether ANY day's early-trade pattern gates this fill's own later outcome as well as
    its OWN day's pattern does. margin = holdout_name's real kept mean minus this placebo's kept
    mean; t = day-clustered t of the placebo kept sample's own mean (see module docstring)."""
    other_name = 'VAL' if holdout_name == 'TRAIN-H2' else 'TRAIN-H2'
    other = df[df.holdout == other_name].reset_index(drop=True)
    blocks = {day: (sub.exit_m.to_numpy(), sub.outcome_R.to_numpy()) for day, sub in other.groupby('day')}
    days = np.array(sorted(blocks.keys()))
    rng = np.random.RandomState(seed)
    shuffled_days = rng.permutation(days)
    perm_map = dict(zip(days, shuffled_days))

    n = len(other)
    F_shuf = np.full(n, np.nan)
    n_res_shuf = np.zeros(n, dtype=int)
    fm = other.fill_min.to_numpy()
    day_arr = other.day.to_numpy()
    for i in range(n):
        em_b, oc_b = blocks[perm_map[day_arr[i]]]
        mask = em_b < fm[i]
        nr = int(mask.sum())
        n_res_shuf[i] = nr
        F_shuf[i] = oc_b[mask].mean() if nr > 0 else np.nan

    placebo_gate = (n_res_shuf >= MIN_N_RES) & (F_shuf >= GATE_PLUS_THRESH)
    placebo_kept = other[placebo_gate]
    placebo_stats = bucket_stats(placebo_kept)
    n_fixed_points = int((shuffled_days == days).sum())
    log(f'placebo for {holdout_name} (built on {other_name} shuffled, seed {seed}): '
        f'{len(days)} days permuted ({n_fixed_points} unshuffled by chance), '
        f'placebo kept n={placebo_stats["n"]}, mean={placebo_stats["mean"]}')
    return dict(other_holdout=other_name, seed=seed, n_days=len(days), n_fixed_points=n_fixed_points,
                placebo_n=placebo_stats['n'], placebo_mean=placebo_stats['mean'], placebo_t=placebo_stats['t'])


# --------------------------------------------------------------------------------------------
# Step 3: report
# --------------------------------------------------------------------------------------------

def write_fills_csv(df):
    out = df[['day', 'symbol', 'holdout', 'F', 'n_res', 'gate_plus', 'gate_minus', 'outcome_R']].copy()
    out = out.rename(columns={'holdout': 'split'})
    out.to_csv(OUT_FILLS_CSV, index=False)
    log(f'wrote {OUT_FILLS_CSV} ({len(out)} rows)')


def fmt(x, nd=4):
    return 'nan' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{nd}f}'


def write_result_md(rows, autocorr, placebos, exit_check, cache_agree, verdict, verdict_reasons):
    val = next(r for r in rows if r['holdout'] == 'VAL')
    th2 = next(r for r in rows if r['holdout'] == 'TRAIN-H2')
    pv = placebos['VAL']

    lines = []
    lines.append('# RESULT -- cell 1,623: the day\'s own feedback gate\n')
    lines.append('`PREREG_1623.md`. Base = the 9,911 fills of cell 1,438 (status==\'fill\', VAL '
                  'union TRAIN-H2, TEST sealed/never read). outcome_R = cell 1,478\'s '
                  'standard-cost net R (model_1478_L3_predictions.csv, matched 1:1, 0 unmatched). '
                  'F(f) = mean outcome_R of the day\'s OTHER fills whose OWN exit_m is strictly '
                  'before f\'s fill_min; n_res(f) = their count. G+: n_res>=2 and F>=+0.5 R '
                  '(the candidate rule, "kept" below); G-: n_res>=2 and F<=-0.5 R (report-only, '
                  'the mirror); remainder: n_res>=2 and -0.5<F<0.5; insufficient: n_res<2.\n')
    lines.append(f'**Exit-minute source**: model_1478_L3_predictions.csv has no exit_m column '
                 f'(confirmed at runtime). Used causal_arming_causal.csv\'s own exit_m (complete, '
                 f'0/9,911 missing, paired with the same row\'s fill_min). Cross-checked against '
                 f'fallback (2), rebuild_1481_fills.csv: present for {exit_check["n_present"]}/'
                 f'{exit_check["n_total"]} base rows, exact match {exit_check["match_pct"]:.1f}% '
                 f'where present (the ~20% drift and the missing 9.5% are cell 1,481\'s own '
                 f'independent-rebuild disagree/no_retest population, not used here). '
                 f'store_served_1438 (features_1478_A.csv) agrees with model_1478_L3_predictions.csv\'s '
                 f'own copy of the same flag on {cache_agree * 100:.2f}% of rows.\n')

    lines.append('## Main table -- G+ (kept) vs dropped, per holdout\n')
    lines.append('| holdout | n kept | mean net R | t | ex-top5 | fills/wk (12/4) | n dropped | '
                  'dropped mean | delta (kept-dropped) | kept cache-only % |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        lines.append(f"| {r['holdout']} | {r['n_kept']} | {fmt(r['kept_mean'])} | {fmt(r['t_kept'], 2)} | "
                      f"{fmt(r['ex_top5'])} | {fmt(r['fills_wk'], 2)} | {r['n_dropped']} | "
                      f"{fmt(r['dropped_mean'])} | {fmt(r['delta_R'])} | {fmt(r['cache_only_share_pct'], 1)} |")

    lines.append('\n## All four buckets, per holdout (G- and remainder/insufficient are report-only)\n')
    lines.append('| holdout | bucket | n | mean net R | t | ex-top5 |')
    lines.append('|---|---|---|---|---|---|')
    for r in rows:
        for bname, b in r['buckets'].items():
            lines.append(f"| {r['holdout']} | {bname} | {b['n']} | {fmt(b['mean'])} | {fmt(b['t'], 2)} | {fmt(b['ex_top5'])} |")

    lines.append('\n## 5-bin autocorrelation table: outcome_R of f vs F(f) (rows with F defined, n_res>=1)\n')
    lines.append('| holdout | bin | n | F range | mean F | mean outcome_R | t |')
    lines.append('|---|---|---|---|---|---|---|')
    for hname, tbl in autocorr.items():
        for row in tbl:
            lines.append(f"| {hname} | {row['bin']}/5 | {row['n']} | [{fmt(row['f_min'],2)}, {fmt(row['f_max'],2)}] | "
                          f"{fmt(row['f_mean'])} | {fmt(row['outcome_mean'])} | {fmt(row['t'], 2)} |")

    lines.append('\n## Shuffle placebo (seed 1623, single permutation draw per holdout)\n')
    lines.append('Gate applied to the OTHER holdout with each fill\'s day-context substituted by a randomly '
                  'permuted day (own outcome_R kept real). margin = this holdout\'s real G+ kept mean minus '
                  'the placebo\'s G+ kept mean; t = day-clustered t of the placebo kept sample\'s own mean.\n')
    lines.append('| holdout | built on (shuffled) | days permuted | unshuffled by chance | placebo n | placebo mean | margin | t |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for r in rows:
        p = placebos[r['holdout']]
        margin = r['kept_mean'] - p['placebo_mean'] if not np.isnan(p['placebo_mean']) else float('nan')
        lines.append(f"| {r['holdout']} | {p['other_holdout']} | {p['n_days']} | {p['n_fixed_points']} | "
                      f"{p['placebo_n']} | {fmt(p['placebo_mean'])} | {fmt(margin)} | {fmt(p['placebo_t'], 2)} |")

    lines.append('\n## Pass-bar checklist (frozen; PREREG_1623.md lines 36-39, scored on VAL)\n')
    for desc, ok in verdict_reasons:
        lines.append(f'- [{"x" if ok else " "}] {desc}')
    lines.append(f'\n**Verdict: {verdict}**\n')

    lines.append('## Caveats (read as an adversary)\n')
    lines.append('- The shuffle placebo is a SINGLE permutation draw (seed 1623, as the PREREG specifies one '
                  'seed, not a distribution) -- noisier than the 1,000-draw count-matched null used elsewhere '
                  'in this line (cell 1,445\'s `null_percentile_of`); a different draw could move the margin/t '
                  'meaningfully for small kept-n holdouts. Re-run with several seeds before treating a narrow '
                  'pass/fail as final.\n'
                  '- "Margin and t" for the placebo is read here as (this holdout\'s real kept mean minus the '
                  'placebo\'s own kept mean) and (day-clustered t of the placebo kept sample alone) -- matching '
                  'how every other t in this line attaches to one sample\'s mean -- rather than a two-sample '
                  'difference test; a stricter reader could ask for the latter too.\n'
                  '- The autocorrelation table and the G-/remainder/insufficient buckets are report-only, not '
                  'part of the frozen pass bar; do not read a bin\'s or bucket\'s sign as a verdict on its own.\n'
                  '- exit_m is causal_arming_causal.csv\'s own column, not independently re-walked in this '
                  'script; the cross-reference above measures agreement with cell 1,481\'s independent rebuild '
                  'but this script does not re-derive exit_m from bars_fills_1478.db itself.\n'
                  '- TEST is untouched by this script (VAL/TRAIN-H2 only), per the frozen spec.\n')
    with open(OUT_RESULT_MD, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    log(f'wrote {OUT_RESULT_MD}')


def evaluate_pass_bar(val, th2, placebo_val):
    margin = val['kept_mean'] - placebo_val['placebo_mean'] if not np.isnan(placebo_val['placebo_mean']) else float('nan')
    same_sign = (not np.isnan(val['kept_mean']) and not np.isnan(th2['kept_mean'])
                 and np.sign(val['kept_mean']) == np.sign(th2['kept_mean']) and val['kept_mean'] != 0)
    checks = [
        (f"VAL kept mean net R >= {BAR_KEPT_MEAN_VAL} (got {fmt(val['kept_mean'])})", val['kept_mean'] >= BAR_KEPT_MEAN_VAL if not np.isnan(val['kept_mean']) else False),
        (f"VAL day-clustered t >= {BAR_T_VAL} (got {fmt(val['t_kept'], 2)})", val['t_kept'] >= BAR_T_VAL if not np.isnan(val['t_kept']) else False),
        (f"VAL ex-top-5% > 0 (got {fmt(val['ex_top5'])})", val['ex_top5'] > 0 if not np.isnan(val['ex_top5']) else False),
        (f"VAL fills/wk at 12/4 >= {BAR_FILLS_WK_VAL} (got {fmt(val['fills_wk'], 2)})", val['fills_wk'] >= BAR_FILLS_WK_VAL),
        (f"dropped < kept on BOTH holdouts (VAL {fmt(val['dropped_mean'])}<{fmt(val['kept_mean'])}; "
         f"TRAIN-H2 {fmt(th2['dropped_mean'])}<{fmt(th2['kept_mean'])})",
         (val['dropped_mean'] < val['kept_mean']) and (th2['dropped_mean'] < th2['kept_mean'])
         if not any(np.isnan(x) for x in [val['dropped_mean'], val['kept_mean'], th2['dropped_mean'], th2['kept_mean']]) else False),
        (f"TRAIN-H2 same sign as VAL and t >= {BAR_TRAIN_T} (got mean {fmt(th2['kept_mean'])}, t {fmt(th2['t_kept'], 2)})",
         same_sign and (th2['t_kept'] >= BAR_TRAIN_T if not np.isnan(th2['t_kept']) else False)),
        (f"placebo margin >= {BAR_PLACEBO_MARGIN} and t >= {BAR_PLACEBO_T} (got margin {fmt(margin)}, t {fmt(placebo_val['placebo_t'], 2)})",
         (margin >= BAR_PLACEBO_MARGIN if not np.isnan(margin) else False)
         and (placebo_val['placebo_t'] >= BAR_PLACEBO_T if not np.isnan(placebo_val['placebo_t']) else False)),
        (f"kept cache-only share within {CACHE_ONLY_TOL_PP}pp of {CACHE_ONLY_BASELINE_PCT}% (got {fmt(val['cache_only_share_pct'], 1)}%)",
         abs(val['cache_only_share_pct'] - CACHE_ONLY_BASELINE_PCT) <= CACHE_ONLY_TOL_PP if not np.isnan(val['cache_only_share_pct']) else False),
    ]
    verdict = 'PASS' if all(ok for _, ok in checks) else 'FAIL'
    return verdict, checks


def main():
    log('=== cell 1,623: the day\'s own feedback gate ===')
    base = load_base_book()
    df = merge_outcome_and_flags(base)
    exit_check = independent_check_exit_m(df)
    # re-derive the store_served_1438 agreement number (already logged once inside
    # merge_outcome_and_flags) so write_result_md can cite it without a second return value
    pred = pd.read_csv(PRED_CSV, low_memory=False)[['day', 'symbol', 'fill_min', 'store_served_1438']]
    chk = df.merge(pred, on=['day', 'symbol', 'fill_min'], how='left', suffixes=('', '_pred2'))
    cache_agree = float((chk.store_served_1438 == chk.store_served_1438_pred2).mean())

    df = validate_invariants(df)
    df = compute_feedback(df)
    df = assign_gates(df)

    rows = [holdout_breakdown(df, h) for h in ('TRAIN-H2', 'VAL')]
    autocorr = {h: autocorr_table(df, h) for h in ('TRAIN-H2', 'VAL')}
    placebos = {h: shuffle_placebo(df, h) for h in ('TRAIN-H2', 'VAL')}

    val_row = next(r for r in rows if r['holdout'] == 'VAL')
    th2_row = next(r for r in rows if r['holdout'] == 'TRAIN-H2')
    verdict, checks = evaluate_pass_bar(val_row, th2_row, placebos['VAL'])
    log(f'VERDICT: {verdict}')
    for desc, ok in checks:
        log(f'  [{"PASS" if ok else "FAIL"}] {desc}')

    write_fills_csv(df)
    write_result_md(rows, autocorr, placebos, exit_check, cache_agree, verdict, checks)
    log('done')


if __name__ == '__main__':
    main()
