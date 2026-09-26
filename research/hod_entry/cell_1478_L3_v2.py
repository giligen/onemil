#!/usr/bin/env python3
"""Cell 1,478-L3-v2 -- PREREG_1478.md Amendment 3 (2026-09-26 18:20 UTC) re-run.

Amendment 3 found that build_features_1478_A.bar_features_for_fill's original arm-bar rule
(`rth.m < fill_min` with a FRACTIONAL fill_min) selected bar j = the FILL bar itself in ~98% of
rows -- a look-ahead. build_features_1478_A.py now has an ARM_BAR_CLOSED switch (default False =
old behaviour, unchanged); this script builds features_1478_A_v2.csv with the corrected rule
(arm_bar_closed=True: bar j = last RTH bar fully closed before fill_min) via
`build_features_1478_A.py --arm-bar-closed --out-csv features_1478_A_v2.csv` (run separately --
NOT by this script, which only re-fits the L3 model on the result).

This script is cell_1478_L3.py verbatim EXCEPT it points cell_1478.FEATURES_A at
features_1478_A_v2.csv (monkeypatched at import time, restored after) instead of editing the
shared cell_1478.py or overwriting features_1478_A.csv / model_1478_L3_predictions.csv (both are
read by other cells and must not change). Same protocol as the original run (L1/L2's own
pipeline: run_label, unmodified) -- same features B+C, same grid, same seed 1478, same TRAIN-H2
top-tercile threshold, same shuffled-label placebo, same metadata decoy model.

Usage:
    python3 research/hod_entry/cell_1478_L3_v2.py [--dry-run]

Outputs: model_1478_L3_v2_predictions.csv, RESULT_1478_L3_v2.md (new file, NOT appended to
RESULT_1478.md).
"""
import argparse
import os
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

from research.hod_entry import cell_1445 as c1445          # noqa: E402
from research.hod_entry import cell_1478 as c1478          # noqa: E402
from research.hod_entry.cell_1478_L3 import load_bars, build_label_l3, fmt_l3_section  # noqa: E402

FEATURES_A_V2 = os.path.join(HERE, 'features_1478_A_v2.csv')
# Original (leaky) L3 run, RESULT_1478.md lines 130-160 (HGB/VAL row): AUC 0.7232, kept mean
# -0.0908, decoy VAL AUC 0.5795 (void=True, > 0.55), placebo VAL AUC 0.4827 (ok).
ORIG_AUC = 0.7232
ORIG_VAL_KEPT_MEAN = -0.0908
ORIG_DECOY_AUC = 0.5795
ORIG_PLACEBO_AUC = 0.4827


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    assert os.path.exists(FEATURES_A_V2), \
        f'{FEATURES_A_V2} missing -- run build_features_1478_A.py --arm-bar-closed --out-csv ' \
        f'features_1478_A_v2.csv first'

    t0 = time.time()
    log(f'cell_1478_L3_v2: pointing cell_1478.FEATURES_A at {FEATURES_A_V2} '
        f'(features_1478_A.csv and model_1478_L3_predictions.csv are untouched)')
    orig_features_a = c1478.FEATURES_A
    c1478.FEATURES_A = FEATURES_A_V2

    try:
        log('cell_1478_L3_v2: loading base fills')
        base = c1478.load_base_fills()
        if args.dry_run:
            base = base.sample(n=min(400, len(base)), random_state=c1478.SEED).reset_index(drop=True)
            log(f'--dry-run: subsampled to {len(base)} rows')

        log('cell_1478_L3_v2: loading bars_fills_1478.db')
        bars = load_bars()
        log(f'  {len(bars)} RTH bars, '
            f'{bars[["symbol", "day"]].drop_duplicates().shape[0]} symbol-days')

        log('cell_1478_L3_v2: building label L3 (break extends) -- unchanged, label logic is '
            'independent of the arm-bar amendment (uses fill_min directly, not features_1478_A)')
        base = build_label_l3(base, bars)

        log('cell_1478_L3_v2: building outcome R (unchanged)')
        base = c1478.build_outcome(base)

        log(f'cell_1478_L3_v2: loading/merging features A(v2)+B+C from {c1478.FEATURES_A}')
        feats = c1478.load_and_merge_features()
        fcols = c1478.feature_columns(feats)
        merged = base.merge(feats, on=['day', 'symbol', 'fill_min'], how='inner', suffixes=('', '_f'))
        assert len(merged) == len(base), \
            f'base x features merge dropped rows: {len(base)} -> {len(merged)}'
        if 'split_f' in merged.columns:
            assert (merged['split'] == merged['split_f']).all(), 'base/features split mismatch'
            merged = merged.drop(columns=['split_f'])
        assert not (set(fcols) & {'L3', 'max_high_after_j'}), 'label leaked into features'
        log(f'cell_1478_L3_v2: merged book = {len(merged)} rows')
        weeks = {s: c1445.weeks_spanned(merged.loc[merged.split == s, 'day']) for s in ('TRAIN', 'VAL')}
        log(f'weeks spanned: {weeks}')

        log('cell_1478_L3_v2: running label L3 through the L1/L2/L3 pipeline (run_label, unmodified)')
        result = c1478.run_label('L3', merged, fcols, weeks)

        pred_out = merged[['day', 'symbol', 'fill_min', 'split', 'why', 'outcome_R', 'L3',
                            'store_served_1438']].merge(
            result['preds'], on=['day', 'symbol', 'fill_min', 'split'], how='left')
        suffix = '_DRYRUN' if args.dry_run else ''
        pred_path = os.path.join(HERE, f'model_1478_L3_v2_predictions{suffix}.csv')
        pred_out.to_csv(pred_path, index=False)
        log(f'wrote {pred_path} ({len(pred_out)} rows)')

        section = fmt_l3_section(result, weeks)
        val_row_hgb = next(r for r in result['rows'] if r['model'] == 'HGB' and r['holdout'] == 'VAL')
        tr_row_hgb = next(r for r in result['rows'] if r['model'] == 'HGB' and r['holdout'] == 'TRAIN-H2')
        header = [
            '# Cell 1,478-L3-v2: re-run under Amendment 3 (arm bar = last bar CLOSED before fill_min)',
            '', f'Built from {FEATURES_A_V2} (build_features_1478_A.py --arm-bar-closed), '
                f'features B+C unchanged.', '',
            '## Side-by-side vs the original (leaky) L3 run (RESULT_1478.md, PREREG Amendment 2)',
            '', '| | original (leaky arm bar) | v2 (corrected arm bar) |',
            '|---|---|---|',
            f'| VAL AUC (HGB) | {ORIG_AUC:.4f} | {val_row_hgb["auc"]:.4f} |',
            f'| kept VAL mean net R | {ORIG_VAL_KEPT_MEAN:.4f} | {val_row_hgb["kept_mean"]:.4f} |',
            f'| kept VAL t (day-clustered) | (not reported in the original) | {val_row_hgb["t_kept"]:.4f} |',
            f'| kept TRAIN-H2 mean net R | 0.1446 (orig) | {tr_row_hgb["kept_mean"]:.4f} |',
            f'| kept TRAIN-H2 t | 2.0710 (orig) | {tr_row_hgb["t_kept"]:.4f} |',
            f'| TRAIN-H2 CV AUC | 0.7251 (orig) | {result["hgb_cv_auc"]:.4f} |',
            f'| decoy VAL AUC | {ORIG_DECOY_AUC:.4f} (void, >0.55) | {result["decoy_val_auc"]:.4f} '
            f'(void={result["decoy_void"]}) |',
            f'| placebo VAL AUC | {ORIG_PLACEBO_AUC:.4f} | {result["placebo_val_auc"]:.4f} |',
            '',
            f'**Verdict: the original AUC 0.72 does {"" if val_row_hgb["auc"] >= 0.60 - 1e-9 and val_row_hgb["auc"] > ORIG_AUC - 0.05 else "NOT "}'
            f'survive the arm-bar correction** '
            f'(v2 VAL AUC {val_row_hgb["auc"]:.4f} vs original {ORIG_AUC:.4f}; '
            f'v2 kept VAL mean {val_row_hgb["kept_mean"]:.4f} vs original {ORIG_VAL_KEPT_MEAN:.4f}).',
            '',
        ]
        result_path = os.path.join(HERE, f'RESULT_1478_L3_v2{suffix}.md')
        with open(result_path, 'w') as fh:
            fh.write('\n'.join(header) + '\n'.join(section) + '\n')
        log(f'wrote {result_path}')
        log(f'cell_1478_L3_v2: DONE in {time.time() - t0:.0f}s')
    finally:
        c1478.FEATURES_A = orig_features_a


if __name__ == '__main__':
    main()
