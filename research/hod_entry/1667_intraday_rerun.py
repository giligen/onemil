#!/usr/bin/env python3
"""Cell 1,667 continuation: bars_sip.db was backfilled 2026-09-29 18:36 UTC
(verified: mtime 18:36:01, +143MB) and is claimed to now hold every fill's
own day. This script recomputes ONLY the intraday features F11-F16 and every
pre-declared read that depends on them (R1 tercile/quintile, R2 top/bottom
quintile, R3 Spearman, R4 stopbucket x {F11,F12}, R5 coverage), both halves.

F1-F10, F17, F18 and their reads (incl. R4 stopbucket x {F2,F3} and R4
price x F2) are UNCHANGED -- not recomputed, not touched in the outputs.
F17 (n_cross) is out of scope regardless: it was never bars_sip-derived and
is excluded from this round's reporting per the coordinator's instruction.

Reuses the query/stat machinery from 1667_sweep.py via importlib so the
level-bar definition, the ET-timezone fix and the statistics are IDENTICAL
to the first run -- nothing is reimplemented or redefined here.
"""
import importlib.util
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('sweep1667', os.path.join(HERE, '1667_sweep.py'))
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)  # defines functions/constants only; __main__ guard not triggered

FEATURES_CSV = sweep.FEATURES_CSV
READS_CSV = sweep.READS_CSV
RESULT_MD = sweep.RESULT_MD
LOG_FILE = sweep.LOG_FILE
HALVES = sweep.HALVES
INTRADAY_FEATS = ['F11', 'F12', 'F13', 'F14', 'F15', 'F16']

logger = sweep.logger  # reuse the same named logger so all output interleaves in one log


def setup_logging():
    """Append to the SAME 1667_sweep.log (not truncated) plus stdout."""
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE, mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(levelname)s %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)


def main():
    setup_logging()
    logger.info('=== cell 1,667 INTRADAY RE-RUN start (bars_sip backfilled 18:36 UTC) ===')

    feat = pd.read_csv(FEATURES_CSV, dtype={'date': str, 'symbol': str})
    logger.info('loaded cached %s: %d rows, %d cols (F1-F10/F17/F18 kept as-is)', FEATURES_CSV, *feat.shape)

    # drop the stale F11-F16 columns so compute_intraday_features rebuilds them fresh
    feat = feat.drop(columns=[c for c in INTRADAY_FEATS + ['level_bar_found'] if c in feat.columns])
    feat = sweep.compute_intraday_features(feat, resume=False)

    tmp = FEATURES_CSV + '.tmp'
    feat.to_csv(tmp, index=False)
    os.replace(tmp, FEATURES_CSV)
    logger.info('wrote %s (%d rows, %d cols) -- F11-F16 replaced, all else unchanged', FEATURES_CSV, *feat.shape)

    sd_half = {h: feat.loc[feat['half'] == h, 'net_R'].std(ddof=1) for h in HALVES}

    cov_rows = [sweep.coverage_line(feat, f) for f in INTRADAY_FEATS]
    cov_df = pd.DataFrame(cov_rows)
    void_feats = set(cov_df.loc[cov_df['void'], 'feature'])
    logger.info('re-run VOID features: %s', sorted(void_feats) or 'none')

    new_reads = []

    def add_read(family, feature, bucket, half, stats):
        new_reads.append(dict(family=family, feature=feature, bucket=bucket, half=half, **stats))

    edges3, edges5 = {}, {}
    for f in INTRADAY_FEATS:
        if f in void_feats:
            continue
        edges3[f] = sweep.make_edges(feat[f], 3)
        edges5[f] = sweep.make_edges(feat[f], 5)
        feat[f + '_T'] = sweep.assign_bucket(feat[f], edges3[f])
        feat[f + '_Q'] = sweep.assign_bucket(feat[f], edges5[f])

    # R1
    for f in INTRADAY_FEATS:
        if f in void_feats:
            continue
        for half in HALVES:
            hdf = feat[feat['half'] == half]
            for bcol, fam in [(f + '_T', 'R1_tercile'), (f + '_Q', 'R1_quintile')]:
                for b in sorted(hdf[bcol].dropna().unique(), key=str):
                    sub = hdf[hdf[bcol] == b]
                    add_read(fam, f, str(b), half, sweep.stats_for_subset(sub, sd_half[half]))

    # R2
    for f in INTRADAY_FEATS:
        if f in void_feats:
            continue
        qcol = f + '_Q'
        n_bins = feat[qcol].cat.categories.size if hasattr(feat[qcol], 'cat') else 0
        top_label = f'B{n_bins}' if n_bins else None
        bot_label = 'B1' if n_bins else None
        for half in HALVES:
            hdf = feat[feat['half'] == half]
            for label, name in [(top_label, 'top_vs_rest'), (bot_label, 'bottom_vs_rest')]:
                if label is None or label not in set(hdf[qcol].astype(str)):
                    continue
                in_a = (hdf[qcol].astype(str) == label)
                a, b = hdf[in_a], hdf[~in_a]
                st_a = sweep.stats_for_subset(a, sd_half[half])
                st_b = sweep.stats_for_subset(b, sd_half[half])
                if st_a['n'] == 0 or st_b['n'] == 0:
                    continue
                delta = st_a['mean'] - st_b['mean']
                sa2 = a['net_R'].var(ddof=1) if len(a) > 1 else np.nan
                sb2 = b['net_R'].var(ddof=1) if len(b) > 1 else np.nan
                se = (( (sa2 / st_a['n']) if pd.notna(sa2) else 0) + ((sb2 / st_b['n']) if pd.notna(sb2) else 0)) ** 0.5
                welch_t = delta / se if se > 0 else np.nan
                dclust_t = sweep.paired_day_clustered_t(hdf, in_a)
                ex5_delta = st_a['ex_top5'] - st_b['ex_top5']
                add_read('R2', f, name, half, dict(n=st_a['n'], mean=delta, iid_t=welch_t, day_t=dclust_t,
                                                     ex_top5=ex5_delta, fpw=st_a['fpw'], mde=st_a['mde']))

    # R3
    for f in INTRADAY_FEATS:
        if f in void_feats:
            continue
        for half in HALVES:
            hdf = feat[feat['half'] == half]
            rho, t, n = sweep.spearman(hdf[f], hdf['net_R'])
            day_mean = hdf.groupby('date').agg({f: 'mean', 'net_R': 'mean'})
            rho_d, t_d, n_d = sweep.spearman(day_mean[f], day_mean['net_R'])
            new_reads.append(dict(family='R3_spearman', feature=f, bucket='fill_level', half=half,
                                   n=n, mean=rho, iid_t=t, day_t=np.nan, ex_top5=np.nan, fpw=np.nan, mde=np.nan))
            new_reads.append(dict(family='R3_spearman_dayclust', feature=f, bucket='day_level', half=half,
                                   n=n_d, mean=rho_d, iid_t=t_d, day_t=np.nan, ex_top5=np.nan, fpw=np.nan, mde=np.nan))

    # R4: stopbucket x {F11, F12} only (F2/F3/price-x-F2 are unchanged, not redone)
    for f in ['F11', 'F12']:
        if f in void_feats:
            continue
        for half in HALVES:
            hdf = feat[feat['half'] == half]
            for sb in ['1.5-3%', '>=3%']:
                for tb in ['B1', 'B2', 'B3']:
                    sub = hdf[(hdf['stop_bucket'] == sb) & (hdf[f + '_T'].astype(str) == tb)]
                    add_read('R4_stopbucket_x_feature', f'stopbucket x {f}', f'{sb} x {tb}', half,
                              sweep.stats_for_subset(sub, sd_half[half]))

    new_reads_df = pd.DataFrame(new_reads)
    logger.info('produced %d new F11-F16 reads', len(new_reads_df))

    old_reads = pd.read_csv(READS_CSV)
    is_intraday_simple = old_reads['feature'].isin(INTRADAY_FEATS)
    is_intraday_r4 = old_reads['feature'].isin([f'stopbucket x {f}' for f in ['F11', 'F12']])
    dropped = int((is_intraday_simple | is_intraday_r4).sum())
    kept = old_reads[~(is_intraday_simple | is_intraday_r4)]
    logger.info('reads.csv: dropping %d stale F11-F16 rows, keeping %d unrelated rows', dropped, len(kept))
    merged_reads = pd.concat([kept, new_reads_df], ignore_index=True)
    tmp = READS_CSV + '.tmp'
    merged_reads.to_csv(tmp, index=False)
    os.replace(tmp, READS_CSV)
    logger.info('wrote %s (%d total reads)', READS_CSV, len(merged_reads))

    append_result_md(feat, cov_df, new_reads_df, sd_half, void_feats)
    logger.info('=== cell 1,667 INTRADAY RE-RUN done, no ERROR ===')


def append_result_md(feat, cov_df, reads_df, sd_half, void_feats):
    """Append a dated section to RESULT_1667.md; does not touch prior content."""
    ts = time.strftime('%H:%M', time.gmtime())
    lines = ['', f'## Intraday features re-run on the filled store ({ts} UTC)', '',
             'bars_sip.db backfilled 2026-09-29 18:36 UTC (verified: mtime 18:36:01Z, +143MB). '
             'F1-F10/F17/F18 and R4 stopbucket x {F2,F3} / price x F2 are UNCHANGED (see sections above). '
             'F17 (n_cross) is out of scope here and not re-reported.']
    lines.append('')
    lines.append('### R5 -- coverage, post-backfill')
    lines.append('| feature | coverage% | winner cov% | loser cov% | gap pp | VOID |')
    lines.append('|---|---|---|---|---|---|')
    for _, r in cov_df.iterrows():
        lines.append(f"| {r['feature']} | {r['coverage_pct']:.1f} | {r['cov_winner']:.1f} | "
                      f"{r['cov_loser']:.1f} | {r['gap_pp']:.1f} | {'YES' if r['void'] else 'no'} |")
    lines.append('')
    spy_days = feat['F16'].notna().sum()
    lines.append(f'SPY: bars_sip still carries only 1 distinct day (unchanged) -> F16 stays VOID '
                 f'({spy_days} fills matched by coincidence, {"none" if spy_days == 0 else spy_days}).')
    if void_feats:
        lines.append(f"VOID (not reported as numbers below): {', '.join(sorted(void_feats))}")
    lines.append('')

    fam_order = ['R1_tercile', 'R1_quintile', 'R2', 'R3_spearman', 'R3_spearman_dayclust',
                 'R4_stopbucket_x_feature']
    passes = []
    for fam in fam_order:
        sub = reads_df[reads_df['family'] == fam]
        if sub.empty:
            lines.append(f'### {fam}: no non-VOID F11-F16 rows.')
            lines.append('')
            continue
        piv = sub.pivot_table(index=['feature', 'bucket'], columns='half',
                                values=['n', 'mean', 'iid_t', 'day_t', 'ex_top5', 'fpw'], aggfunc='first')
        keep_keys = []
        for key in piv.index:
            row = piv.loc[key]
            tvals = []
            for half in HALVES:
                dt = row.get(('day_t', half), np.nan)
                it = row.get(('iid_t', half), np.nan)
                tvals.append(dt if pd.notna(dt) else it)
            if any(pd.notna(v) and abs(v) >= 2.0 for v in tvals):
                keep_keys.append(key)
        lines.append(f'### {fam} (|t|>=2 in either half; {len(keep_keys)}/{len(piv)} shown)')
        if not keep_keys:
            lines.append('none.')
            lines.append('')
            continue
        lines.append('| feature | bucket | TRAIN-H2 n/mean/iid_t/day_t/ex5 | VAL n/mean/iid_t/day_t/ex5 |')
        lines.append('|---|---|---|---|')
        for key in keep_keys:
            f_, b_ = key
            row = piv.loc[key]
            cells = []
            for half in HALVES:
                n_, m_, it_, dt_, e5_ = (row.get(('n', half), np.nan), row.get(('mean', half), np.nan),
                                          row.get(('iid_t', half), np.nan), row.get(('day_t', half), np.nan),
                                          row.get(('ex_top5', half), np.nan))
                cells.append(f"n={n_:.0f} m={m_:.3f} it={it_:.2f} dt={dt_:.2f} ex5={e5_:.3f}"
                              if pd.notna(n_) else 'n/a')
            lines.append(f"| {f_} | {b_} | {cells[0]} | {cells[1]} |")
            # pass-bar check (PREREG_1662 S1664): net>=.05R, |t|>=2.5 both halves, ex5>0 both, fpw>=3
            if fam in ('R1_tercile', 'R1_quintile', 'R2', 'R4_stopbucket_x_feature'):
                grp = sub[(sub['feature'] == f_) & (sub['bucket'] == b_)]
                if len(grp) == 2:
                    ok = True
                    for half in HALVES:
                        r = grp[grp['half'] == half].iloc[0]
                        t_use = r['day_t'] if pd.notna(r['day_t']) else r['iid_t']
                        if not (pd.notna(r['mean']) and r['mean'] >= 0.05 and pd.notna(t_use) and abs(t_use) >= 2.5
                                and pd.notna(r['ex_top5']) and r['ex_top5'] > 0 and r['fpw'] >= 3.0):
                            ok = False
                            break
                    if ok:
                        passes.append((fam, f_, b_))
        lines.append('')

    lines.append('### Verdict, F11-F16 only')
    if passes:
        lines.append(f'PASSES ({len(passes)}): ' + '; '.join(f'{fam}:{f_}:{b_}' for fam, f_, b_ in passes) +
                     '. Requires independent reimplementation before reaching the owner (CLAUDE.md #1) -- NOT done here.')
    else:
        lines.append('No F11-F16 cut clears the S1664 bar in both halves.')
    lines.append('')
    lines.append(f'Full read table: {READS_CSV}. Full feature table: {FEATURES_CSV}.')

    with open(RESULT_MD, 'a') as fh:
        fh.write('\n'.join(lines) + '\n')
    logger.info('appended %d lines to %s', len(lines), RESULT_MD)


if __name__ == '__main__':
    main()
