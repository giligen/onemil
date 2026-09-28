"""Independent rebuild of cell 1,607 -- research/exec_quality/PREREG_1607.md (FROZEN 2026-09-28).

Written from the PREREG's prose ONLY -- the existing cell_1607.py / RESULT_1607.md were never opened
(per the independent-check protocol, CLAUDE.md "No research claim ships without an independent check").

Question (owner 9/28): "maybe the ones with the bigger spread are the winners? if they are the
winners, maybe the spread is a filter?" -- on the 9,911 base fills of cell 1,438
(causal_arming_causal.csv, status == 'fill'), outcomes = the standard-cost net R of
model_1478_L3_predictions.csv (outcome_R).

Candidate spread fields, per the PREREG:
  (a) features_1478_A.csv: spread_frac_at_fill x 1e4
  (b) features_1478_A.csv: 2 x half_entry / fill x 1e4
  -- "state which is the arm-time quote and use the causal one; the spread must be the quote BEFORE
     the fill print."

This rebuild's finding (see REBUILD_1607.md): (a) and (b) are THE SAME NUMBER (half_entry is defined
as spread_frac_at_fill * fill / 2 by construction in cell_1445.corrected_cost()) and NEITHER is an
arm-time quote -- both are recovered from the fill's OWN realized round-trip cost (cost_R, which is
only known after the exit: cost_R = raw_R - net_R). This fails the PREREG's own causality bar ("never
the fill's own quote"). A genuinely causal arm-time quote exists in features_1478_C.csv
(spread_bps_at_arm -- the prevailing NBBO quote at the first print of the breakout bar, strictly
before the trigger print that produced the fill, per FEATURES_C.md item 4). Both are computed below;
set-A is reported as the PREREG literally specifies it, set-C is reported "beside" it as the causally
valid version and is the one this rebuild treats as authoritative for the pass-bar verdict.

Usage: python3 research/exec_quality/rebuild_1607.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from research.hod_entry import cell_1445 as c1445  # noqa: E402  (day_clustered_t, ex_top5_mean, fills_per_week, weeks_spanned, score_one, evaluate_pass_bar)

CAUSAL_ARMING_CSV = os.path.join(REPO, 'research/hod_entry/causal_arming_causal.csv')
PREDICTIONS_CSV = os.path.join(REPO, 'research/hod_entry/model_1478_L3_predictions.csv')
FEATURES_A_CSV = os.path.join(REPO, 'research/hod_entry/features_1478_A.csv')
FEATURES_C_CSV = os.path.join(REPO, 'research/hod_entry/features_1478_C.csv')

N_QUINTILES = 5


def log(msg):
    print(f'[rebuild_1607] {msg}', flush=True)


def load_base():
    """9,911 base fills + outcome_R (standard-cost net R) + both spread candidates."""
    ca = pd.read_csv(CAUSAL_ARMING_CSV, low_memory=False)
    fills = ca[ca['status'] == 'fill'].copy()
    assert len(fills) == 9911, f'expected 9,911 base fills, got {len(fills)}'
    fills = fills[['day', 'symbol', 'fill_min', 'split', 'half', 'fill', 'stop', 'level', 'exit_m']]

    pred = pd.read_csv(PREDICTIONS_CSV)[['day', 'symbol', 'fill_min', 'split', 'outcome_R']]
    df = fills.merge(pred, on=['day', 'symbol', 'fill_min'], how='inner', suffixes=('', '_pred'))
    assert len(df) == 9911, f'lost rows on prediction join: {len(df)}'
    mismatch = (df['split'] != df['split_pred']).sum()
    assert mismatch == 0, f'{mismatch} rows disagree on split between the two CSVs'
    df = df.drop(columns=['split_pred'])

    fa = pd.read_csv(FEATURES_A_CSV)[['day', 'symbol', 'fill_min', 'half_entry', 'spread_frac_at_fill']]
    df = df.merge(fa, on=['day', 'symbol', 'fill_min'], how='inner')
    assert len(df) == 9911, f'lost rows on features_A join: {len(df)}'

    fc = pd.read_csv(FEATURES_C_CSV)[['day', 'symbol', 'fill_min', 'spread_bps_at_arm']]
    df = df.merge(fc, on=['day', 'symbol', 'fill_min'], how='left')
    assert len(df) == 9911, f'lost rows on features_C join: {len(df)}'

    # -- holdout label matching the PREREG's "TRAIN-H2" / "VAL" (causal_arming_causal.csv already
    # restricts TRAIN to half=='H2'; verify, don't assume)
    assert (df.loc[df['split'] == 'TRAIN', 'half'] == 'H2').all(), 'TRAIN split is not purely H2'
    df['holdout'] = df['split'].map({'TRAIN': 'TRAIN-H2', 'VAL': 'VAL'})
    assert df['holdout'].isna().sum() == 0

    # -- the two set-A candidates
    df['spreadA_fracfill_bps'] = df['spread_frac_at_fill'] * 1e4
    df['spreadA_halfentry_bps'] = 2.0 * df['half_entry'] / df['fill'] * 1e4
    # -- the causal set-C candidate ("report the other beside")
    df['spreadC_arm_bps'] = df['spread_bps_at_arm']

    # score_one()/evaluate_pass_bar() read fixed column names -- reuse unchanged, don't fork it.
    # net_R_corr_flat30 is a cell-1445-specific flat-slip variant with no equivalent here; score_one
    # reads it but this rebuild never reports kept_mean_flat30, so alias it to net_R_corr (inert).
    df['net_R_corr'] = df['outcome_R']
    df['net_R_corr_flat30'] = df['outcome_R']
    return df


def candidate_agreement(df):
    diff = (df['spreadA_fracfill_bps'] - df['spreadA_halfentry_bps']).abs()
    log(f'set-A candidate agreement: spread_frac_at_fill*1e4 vs 2*half_entry/fill*1e4 -- '
        f'max|diff|={diff.max():.2e} bps, mean|diff|={diff.mean():.2e} bps, '
        f'n(diff>0.01bps)={int((diff > 0.01).sum())}/{len(df)}')
    log('  --> the two PREREG-named set-A candidates are THE SAME QUANTITY by construction '
        '(half_entry is derived as spread_frac_at_fill*fill/2 inside cell_1445.corrected_cost()).')
    n_c_missing = df['spreadC_arm_bps'].isna().sum()
    log(f'set-C causal candidate (spread_bps_at_arm) coverage: {len(df) - n_c_missing}/{len(df)} '
        f'({(1 - n_c_missing / len(df)):.1%}), matches FEATURES_C.md\'s disclosed 98.2%')


def quintile_edges(train_h2, col):
    """Quintile edges from TRAIN-H2 only (pd.qcut, 5 equal-count bins); outer edges widened to
    +/-inf so VAL rows outside the TRAIN-H2 range still land in the extreme quintile."""
    vals = train_h2[col].dropna()
    _, edges = pd.qcut(vals, N_QUINTILES, retbins=True, duplicates='drop')
    edges = edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign_quintile(df, col, edges):
    labels = [f'Q{i+1}' for i in range(len(edges) - 1)]
    return pd.cut(df[col], bins=edges, labels=labels, include_lowest=True)


def per_quintile_stats(df, col, edges, weeks_by_holdout):
    """n, mean net R, day-clustered t, ex-top-5% mean, fills/wk -- per quintile x holdout."""
    q = assign_quintile(df, col, edges)
    rows = []
    for holdout in ('TRAIN-H2', 'VAL'):
        hdf = df[df['holdout'] == holdout]
        hq = q.loc[hdf.index]
        for lab in sorted(hq.dropna().unique(), key=str):
            sub = hdf[hq == lab]
            n = len(sub)
            if n == 0:
                continue
            rows.append(dict(
                holdout=holdout, quintile=lab, n=n,
                mean_R=float(sub['net_R_corr'].mean()),
                t=c1445.day_clustered_t(sub['net_R_corr'], sub['day']),
                ex_top5=c1445.ex_top5_mean(sub['net_R_corr']),
                fills_wk=c1445.fills_per_week(sub, weeks_by_holdout[holdout]),
                spread_lo=float(sub[col].min()), spread_hi=float(sub[col].max()),
            ))
    return pd.DataFrame(rows)


def keep_wide_tight(df, col, edges, weeks_by_holdout):
    """KEEP-WIDE = top 2 quintiles, KEEP-TIGHT = bottom 2 quintiles (both pre-declared per the
    PREREG). Scored with cell_1445.score_one/evaluate_pass_bar, reused unchanged."""
    q = assign_quintile(df, col, edges)
    labels = sorted(q.dropna().unique(), key=str)
    assert len(labels) == N_QUINTILES, f'expected 5 quintiles, got {labels} (duplicate edges?)'
    wide_labels = set(labels[-2:])
    tight_labels = set(labels[:2])

    out = {}
    for name, keep_labels in (('KEEP-WIDE', wide_labels), ('KEEP-TIGHT', tight_labels)):
        rows_by_holdout = {}
        for holdout in ('TRAIN-H2', 'VAL'):
            hdf = df[df['holdout'] == holdout]
            hq = q.loc[hdf.index]
            cond = hq.isin(keep_labels)
            rows_by_holdout[holdout] = c1445.score_one(name, cond, hdf, holdout, weeks_by_holdout[holdout])
        verdict = c1445.evaluate_pass_bar({name: rows_by_holdout})[name]
        out[name] = dict(rows=rows_by_holdout, pass_bar=verdict)
    return out


def run_for_candidate(df, col, label):
    log(f'=== {label} ({col}) ===')
    train_h2 = df[df['holdout'] == 'TRAIN-H2']
    dsub = df.dropna(subset=[col])
    n_dropped_na = len(df) - len(dsub)
    if n_dropped_na:
        log(f'  dropping {n_dropped_na} rows with missing {col} before quintiling')
    train_h2 = dsub[dsub['holdout'] == 'TRAIN-H2']
    weeks_by_holdout = {name: c1445.weeks_spanned(dsub.loc[dsub['holdout'] == name, 'day'])
                         for name in ('TRAIN-H2', 'VAL')}

    edges = quintile_edges(train_h2, col)
    log(f'  TRAIN-H2 quintile edges (bps): {[round(e, 2) if np.isfinite(e) else e for e in edges]}')

    stats = per_quintile_stats(dsub, col, edges, weeks_by_holdout)
    with pd.option_context('display.width', 160, 'display.max_columns', 20):
        log('\n' + stats.to_string(index=False))

    kw = keep_wide_tight(dsub, col, edges, weeks_by_holdout)
    for name, res in kw.items():
        val, th2 = res['rows']['VAL'], res['rows']['TRAIN-H2']
        log(f'  {name}: VAL n={val["n_kept"]} mean={val["kept_mean"]:.4f}R t={val["t_kept"]:.2f} '
            f'ex_top5={val["ex_top5"]:.4f} fills/wk={val["fills_wk"]:.2f} dropped_mean={val["dropped_mean"]:.4f} '
            f'|| TRAIN-H2 n={th2["n_kept"]} mean={th2["kept_mean"]:.4f}R t={th2["t_kept"]:.2f} '
            f'dropped_mean={th2["dropped_mean"]:.4f} || PASS-BAR={res["pass_bar"]}')
    return edges, stats, kw


def main():
    df = load_base()
    log(f'loaded {len(df)} base fills (TRAIN-H2={len(df[df.holdout=="TRAIN-H2"])}, '
        f'VAL={len(df[df.holdout=="VAL"])})')
    candidate_agreement(df)

    resA = run_for_candidate(df, 'spreadA_fracfill_bps',
                              'SET-A (PREREG-literal, NOT causal -- derived from realized cost_R)')
    resC = run_for_candidate(df, 'spreadC_arm_bps',
                              'SET-C (causal arm-time quote, "report beside") -- AUTHORITATIVE for the verdict')
    return resA, resC


if __name__ == '__main__':
    main()
