#!/usr/bin/env python3
"""Cell 1,478-L3 -- research/hod_entry/PREREG_1478.md Amendment 2 (2026-09-26 13:40 UTC).

L1 is PARTLY REALISED at arm bar j (range_to_j is a feature; a name 8% off its low needs 2% more)
and two of L2's three terms are themselves features -- both AUCs mostly measure the label's own
construction. Corrective label, same features/grid/seed/threshold/pass-bar as L1/L2: **L3 = the
break EXTENDS -- the day's high AFTER bar j >= level x 1.05** (the future part of the big day;
nothing at bar j realises it).

level and fill_min: causal_arming_causal.csv rows status=='fill' (cell_1478.load_base_fills).
Arm bar j: the last RTH bar in bars_fills_1478.db (the amendment's single-source SIP store) with
ET minute < fill_min, per (symbol, day). Day's high after j: max(h) over RTH bars with ET minute
> j's minute, same (symbol, day). L3 = 1 if that max >= level * 1.05, 0 otherwise; NaN if no bar
exists after arm bar j (or no bar before fill_min at all) -- reported exactly like L1/L2's NaN
rows, never dropped selectively.

This file builds ONLY the L3 label; every other step (outcome_R, feature merge, HGB/LR fit,
decoy model, placebo, scoring, deciles, permutation importances) is cell_1478.py's own
build_outcome / load_and_merge_features / feature_columns / run_label, called unmodified so L3
is fit exactly as L1/L2 were.

Usage:
    nice -n 19 python3 research/hod_entry/cell_1478_L3.py [--dry-run]

Outputs: model_1478_L3_predictions.csv, and a '## L3 (amendment 2)' section APPENDED to
RESULT_1478.md (L1/L2's sections are untouched).
"""
import argparse
import os
import sqlite3
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from research.hod_entry import cell_1445 as c1445          # noqa: E402 -- weeks_spanned
from research.hod_entry import cell_1478 as c1478          # noqa: E402 -- the L1/L2 pipeline

BARS_DB = os.path.join(HERE, 'bars_fills_1478.db')
L3_MULT = 1.05


def log(msg):
    """Verbose progress line, flushed immediately (print() is buffered under nohup otherwise)."""
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def load_bars():
    """RTH bars from bars_fills_1478.db (04:00-20:00 ET, the amendment's single-source SIP store);
    ET minute-of-day computed from the stored UTC timestamp (America/New_York, DST-correct)."""
    con = sqlite3.connect(BARS_DB)
    bars = pd.read_sql_query('select symbol, day, t, h from bars', con)
    con.close()
    ts = pd.to_datetime(bars['t'], utc=True).dt.tz_convert('America/New_York')
    bars['et_minute'] = (ts.dt.hour * 60 + ts.dt.minute).astype(float)
    return bars


def build_label_l3(fills, bars):
    """L3 = day's high AFTER arm bar j >= level * 1.05. Arm bar j = the last bar with
    et_minute < fill_min per (symbol, day); 'after j' = et_minute > j's et_minute, same
    (symbol, day). NaN when no bar precedes fill_min, or none follows the arm bar (never
    filled with a default -- these rows drop out of run_label's NaN-label filter exactly like
    L1/L2's incomplete rows)."""
    bars = bars.sort_values(['symbol', 'day', 'et_minute'])
    groups = {key: g[['et_minute', 'h']].to_numpy()
              for key, g in bars.groupby(['symbol', 'day'], sort=False)}
    n_no_group, n_no_arm, n_no_after = 0, 0, 0
    max_high_after = np.full(len(fills), np.nan)
    for i, r in enumerate(fills.itertuples()):
        arr = groups.get((r.symbol, r.day))
        if arr is None:
            n_no_group += 1
            continue
        minutes, highs = arr[:, 0], arr[:, 1]
        before = minutes < r.fill_min
        if not before.any():
            n_no_arm += 1
            continue
        j_minute = minutes[before].max()
        after = minutes > j_minute
        if not after.any():
            n_no_after += 1
            continue
        max_high_after[i] = highs[after].max()
    log(f'build_label_l3: no bar group for {n_no_group}, no bar before fill_min (no arm bar) for '
        f'{n_no_arm}, no bar after arm bar for {n_no_after} of {len(fills)} fills')

    out = fills.copy()
    out['max_high_after_j'] = max_high_after
    out['L3'] = np.where(np.isnan(max_high_after), np.nan,
                          (max_high_after >= out['level'] * L3_MULT).astype(float))
    for split in ('TRAIN', 'VAL'):
        sub = out.loc[out.split == split, 'L3']
        log(f'  L3 {split}: base rate {sub.mean():.4f} (n={sub.notna().sum()}, '
            f'{sub.isna().sum()} NaN)')
    return out


def fmt_l3_section(result, weeks):
    """'## L3 (amendment 2)' RESULT.md section: base rate, scoring table, decoy AUC, placebo,
    top-10 permutation importances, VAL decile table, cache-only share check -- same fields as
    L1/L2's sections in cell_1478.main(), for L3 alone."""
    lines = ['', '## L3 (amendment 2): the break EXTENDS (day high after bar j >= level x 1.05)',
             '', f'Weeks spanned: {weeks}', '']
    lines.append('### Scoring (kept = prob >= TRAIN-H2 top-tercile threshold, frozen on TRAIN only)')
    lines.append('')
    lines.append(c1478.fmt_rows_md(result['rows']))
    lines.append('')
    lines.append('### Decoy model (metadata-only: store_served_1438, rth_bar_count_1438, '
                  'tick_window_has_bar_j)')
    lines.append(f'* L3: VAL AUC = {result["decoy_val_auc"]:.4f} '
                 f'(void if > {c1478.DECOY_VOID_AUC}) -> decoy_void = {result["decoy_void"]}')
    lines.append('')
    lines.append('### Placebo (label-shuffled TRAIN-H2, seed 1478, applied once to TRUE VAL labels)')
    lines.append(f'* L3: VAL AUC = {result["placebo_val_auc"]:.4f} '
                 f'(pass <= {c1478.PLACEBO_AUC_MAX}); placebo kept mean = '
                 f'{result["placebo_kept_mean"]:.4f} vs whole-book VAL mean (diff tol '
                 f'{c1478.PLACEBO_MEAN_TOL}); placebo_ok = {result["placebo_ok"]}')
    lines.append('')
    lines.append(f'### Top-10 permutation importances (HGB, VAL, ROC AUC drop) -- best params '
                 f'{result["hgb_params"]}, CV AUC {result["hgb_cv_auc"]:.4f}')
    for d in result['top_importances']:
        lines.append(f'  - {d["feature"]}: {d["importance"]:.4f}')
    lines.append('')
    lines.append('### VAL decile table (HGB probability, report-only; decile 10 = highest prob)')
    lines.append('| decile | n | mean_prob | mean_R | bigday_rate |')
    lines.append('|---|---|---|---|---|')
    for d in result['va_deciles']:
        lines.append(f'| {d["decile"]} | {d["n"]} | {d["mean_prob"]:.4f} | {d["mean_R"]:.4f} '
                      f'| {d["bigday_rate"]:.4f} |')
    lines.append('')
    lines.append(f'### Cache-only share check (base {c1478.BASE_CACHEONLY_SHARE:.3f}, tol '
                 f'+/-{c1478.CACHEONLY_TOL})')
    for r in result['rows']:
        ok = abs(r['kept_cacheonly_share'] - c1478.BASE_CACHEONLY_SHARE) <= c1478.CACHEONLY_TOL \
            if not np.isnan(r['kept_cacheonly_share']) else False
        lines.append(f'* {r["label"]}/{r["model"]}/{r["holdout"]}: kept cache-only share = '
                      f'{r["kept_cacheonly_share"]:.4f} -> within tol = {ok}')
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    suffix = '_DRYRUN' if args.dry_run else ''

    t0 = time.time()
    log('cell_1478_L3: loading base fills')
    base = c1478.load_base_fills()
    if args.dry_run:
        base = base.sample(n=min(400, len(base)), random_state=c1478.SEED).reset_index(drop=True)
        log(f'--dry-run: subsampled to {len(base)} rows')

    log('cell_1478_L3: loading bars_fills_1478.db')
    bars = load_bars()
    log(f'  {len(bars)} RTH bars, '
        f'{bars[["symbol", "day"]].drop_duplicates().shape[0]} symbol-days')

    log('cell_1478_L3: building label L3 (break extends)')
    base = build_label_l3(base, bars)

    log('cell_1478_L3: building outcome R (1457 standard + amendment slip substitution, '
        'same as L1/L2)')
    base = c1478.build_outcome(base)

    log('cell_1478_L3: loading/merging features A+B+C (same as L1/L2, decoy columns excluded)')
    feats = c1478.load_and_merge_features()
    fcols = c1478.feature_columns(feats)
    merged = base.merge(feats, on=['day', 'symbol', 'fill_min'], how='inner', suffixes=('', '_f'))
    assert len(merged) == len(base), \
        f'base x features merge dropped rows: {len(base)} -> {len(merged)}'
    if 'split_f' in merged.columns:
        assert (merged['split'] == merged['split_f']).all(), 'base/features split mismatch'
        merged = merged.drop(columns=['split_f'])
    assert not (set(fcols) & {'L3', 'max_high_after_j'}), 'label leaked into features'
    log(f'cell_1478_L3: merged book = {len(merged)} rows')
    weeks = {s: c1445.weeks_spanned(merged.loc[merged.split == s, 'day']) for s in ('TRAIN', 'VAL')}
    log(f'weeks spanned: {weeks}')

    log('cell_1478_L3: running label L3 through the L1/L2 pipeline (run_label, unmodified)')
    result = c1478.run_label('L3', merged, fcols, weeks)

    pred_out = merged[['day', 'symbol', 'fill_min', 'split', 'why', 'outcome_R', 'L3',
                        'store_served_1438']].merge(
        result['preds'], on=['day', 'symbol', 'fill_min', 'split'], how='left')
    pred_path = os.path.join(HERE, f'model_1478_L3_predictions{suffix}.csv')
    pred_out.to_csv(pred_path, index=False)
    log(f'wrote {pred_path} ({len(pred_out)} rows)')

    section = fmt_l3_section(result, weeks)
    result_path = os.path.join(HERE, f'RESULT_1478{suffix}.md')
    with open(result_path, 'a') as fh:
        fh.write('\n'.join(section) + '\n')
    log(f'appended L3 section to {result_path}')
    log(f'cell_1478_L3: DONE in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
