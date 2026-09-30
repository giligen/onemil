#!/usr/bin/env python3
"""Cell 1,676 -- post-entry fix: correct G6/G7 label routing.

Follow-up to RESULT_1676.md's self-caught leakage: run_R3 paired G6's
post-entry features (bars fill+1..fill+k) with the ARM/entry-relative labels
(fail_any/fail10/success15), which share the SAME bars as the label window --
near-tautological. This script re-scores G6 and G7 with labels that start
STRICTLY AFTER the feature window (fail_after_k / success_next15_from_k,
already computed in 1676_features.csv, unused by the original run_R3), and
adds a G7-without-close_R ablation (close_R = distance to the FIXED entry+1R
target dominated every G7 importance table 7-50x -- path, not shape).

No bars_sip re-sweep: every column used here already exists in
1676_features.csv. G6 stays on its originally-computed 1m/5m frames (10m/15m
would need a fresh sweep, out of this fix's budget -- disclosed, not hidden).

Reuses (imported) from 1676_shapes.py: fit_both_scorings, nearest_k,
stats_for_subset, paired_day_clustered_t, ex_top5_mean, fills_per_week,
G7_KS, G7_PLACEBO_KS, G7_IMPORTANCE_KS, TAUS_G7, load_all, FEATURES_CSV,
READS_CSV, RESULT_MD.

Usage: python3 1676_g6g7_fix.py
Outputs: appends R3_G6fix/R3_G7fix/R4_G7fix rows to 1676_reads.csv (old rows
kept, not deleted -- audit trail); replaces the G6/G7 sections of
RESULT_1676.md (self-correction note kept).
"""
import importlib.util
import logging
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('f1676', os.path.join(HERE, '1676_shapes.py'))
f1676 = importlib.util.module_from_spec(_spec)
import sys as _sys
_sys.argv = [_sys.argv[0]]  # prevent f1676's own argparse from seeing our argv
_spec.loader.exec_module(f1676)

logger = f1676.logger
G6_KS = f1676.G6_KS
G7_KS = f1676.G7_KS
G7_PLACEBO_KS = f1676.G7_PLACEBO_KS
G7_IMPORTANCE_KS = f1676.G7_IMPORTANCE_KS
TAUS_G7 = f1676.TAUS_G7
MONEY_KS = [1, 5, 15]


def build_group_cols_fix(df, avail_ks_1670):
    cols = df.columns
    g = {}
    for k in G6_KS:
        g[f'G6_k{k}'] = [c for c in cols if c.startswith(f'g6_k{k}_')]
        kk = f1676.nearest_k(k, avail_ks_1670)
        c1670 = f'k1670_ALL_k{kk}' if kk is not None and f'k1670_ALL_k{kk}' in cols else None
        g[f'G6_k{k}+ALL'] = g[f'G6_k{k}'] + ([c1670] if c1670 else [])
    for k in G7_KS:
        base = [c for c in cols if c.startswith(f'g7_k{k}_') and '_act_' not in c]
        noclose = [c for c in base if not c.endswith('_close_R')]
        kk = f1676.nearest_k(k, avail_ks_1670)
        c1670 = f'k1670_ALL_k{kk}' if kk is not None and f'k1670_ALL_k{kk}' in cols else None
        g[f'G7_k{k}'] = base
        g[f'G7_k{k}+ALL'] = base + ([c1670] if c1670 else [])
        g[f'G7noclose_k{k}'] = noclose
        g[f'G7noclose_k{k}+ALL'] = noclose + ([c1670] if c1670 else [])
    return g


def main():
    t0 = time.time()
    f1676.setup_logging()  # f1676's own logger has no handlers until this runs (main() never
                            # auto-executes on import, so setup_logging() was never called in
                            # this process) -- without it every logger.info() here is silently
                            # dropped and progress is invisible.
    logger.info('=== 1,676 G6/G7 post-entry fix start ===')
    f1676.f1668.check_disk(5.0)
    pop, k1670_wide, avail_ks_1670, panel_by_sym = f1676.load_all()
    feats = pd.read_csv(f1676.FEATURES_CSV, dtype={'fill_id': str, 'date': str, 'symbol': str})
    feats['fill_id'] = feats['fill_id'].astype(str)
    k1670_wide2 = k1670_wide.reset_index()
    k1670_wide2['fill_id'] = k1670_wide2['fill_id'].astype(str)
    feats = feats.merge(k1670_wide2, on='fill_id', how='left')
    logger.info('loaded %d fills, %d columns', len(feats), len(feats.columns))

    group_cols = build_group_cols_fix(feats, avail_ks_1670)
    rows, pred_store = [], {}

    for k in G6_KS:
        for grp in (f'G6_k{k}', f'G6_k{k}+ALL'):
            for label in (f'fail_after_{k}', f'success_next15_from_{k}'):
                res = f1676.fit_both_scorings(feats, group_cols[grp], label, do_importance=True, do_placebo=True)
                for r in res:
                    rows.append(dict(read='R3_G6fix', group=grp, label=label, k=k, n_feat=len(group_cols[grp]),
                                      **{kk: v for kk, v in r.items() if not kk.startswith('_')}))
                    pred_store[(grp, label, r['scoring'])] = (r['_pred_index'], r['_pred'])
        logger.info('G6fix done: k=%d', k)

    for k in G7_KS:
        for grp in (f'G7_k{k}', f'G7_k{k}+ALL', f'G7noclose_k{k}', f'G7noclose_k{k}+ALL'):
            for label in (f'g7fail_after_{k}', f'g7success_next15_from_{k}'):
                res = f1676.fit_both_scorings(feats, group_cols[grp], label,
                                              do_importance=(k in G7_IMPORTANCE_KS),
                                              do_placebo=(k in G7_PLACEBO_KS))
                for r in res:
                    rows.append(dict(read='R3_G7fix', group=grp, label=label, k=k, n_feat=len(group_cols[grp]),
                                      **{kk: v for kk, v in r.items() if not kk.startswith('_')}))
                    pred_store[(grp, label, r['scoring'])] = (r['_pred_index'], r['_pred'])
        if k in G7_PLACEBO_KS or k % 10 == 0:
            logger.info('G7fix done through k=%d (%.0fs elapsed)', k, time.time() - t0)

    r3fix_df = pd.DataFrame(rows)
    logger.info('R3 fix done: %d rows, %.0fs elapsed', len(r3fix_df), time.time() - t0)

    money_rows = []
    for k in MONEY_KS:
        for scoring in ['TRAIN-H2->VAL', 'VAL->TRAIN-H2']:
            test_half = scoring.split('->')[1]
            fail_key = (f'G7noclose_k{k}', f'g7fail_after_{k}', scoring)
            succ_key = (f'G7noclose_k{k}', f'g7success_next15_from_{k}', scoring)
            for action, pkey in [('cut', fail_key), ('short', fail_key), ('add', succ_key)]:
                if pkey not in pred_store or pred_store[pkey][0] is None:
                    continue
                idx, pred = pred_store[pkey]
                col = f'g7_k{k}_act_{action}'
                if col not in feats.columns:
                    continue
                sub = feats.loc[idx].copy()
                sub['_p'] = pred
                sub['_act'] = sub[col]
                elig = sub.dropna(subset=['_act'])
                if len(elig) < 10:
                    continue
                book_mean = elig['_act'].mean()
                for tau in TAUS_G7:
                    gated_mask = (elig['_p'] >= tau).values
                    gated = elig[gated_mask]
                    if gated_mask.sum() < 5:
                        continue
                    tmp = elig[['date']].copy()
                    tmp['net_R'] = elig['_act']
                    t = f1676.paired_day_clustered_t(tmp, gated_mask)
                    money_rows.append(dict(read='R4_G7fix', k=k, action=action, tau=tau, scoring=scoring, half=test_half,
                                            n=int(gated_mask.sum()), mean_action=gated['_act'].mean(), book_mean=book_mean,
                                            mean_dR=gated['_act'].mean() - book_mean, day_t=t,
                                            ex_top5=f1676.ex_top5_mean(gated['_act']), fpw=f1676.fills_per_week(gated)))
    r4fix_df = pd.DataFrame(money_rows)
    logger.info('R4 fix done: %d rows, %.0fs elapsed', len(r4fix_df), time.time() - t0)

    existing = pd.read_csv(f1676.READS_CSV)
    combined = pd.concat([existing, r3fix_df, r4fix_df], ignore_index=True, sort=False)
    tmp = f1676.READS_CSV + '.tmp'
    combined.to_csv(tmp, index=False)
    os.replace(tmp, f1676.READS_CSV)
    logger.info('wrote %s (%d total rows)', f1676.READS_CSV, len(combined))

    r3fix_df.to_csv(os.path.join(HERE, '1676_g6g7fix_reads.csv'), index=False)
    r4fix_df.to_csv(os.path.join(HERE, '1676_g6g7fix_money.csv'), index=False)
    logger.info('=== 1,676 G6/G7 post-entry fix done, %.0fs total ===', time.time() - t0)


if __name__ == '__main__':
    main()
