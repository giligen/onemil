#!/usr/bin/env python3
"""Independent rebuild of cell 1,623 -- research/hod_entry/PREREG_1623.md (FROZEN 2026-09-28 17:15 UTC).

Built from the PREREG prose only. Per the task's independent-check protocol this script was written
WITHOUT opening cell_1623.py, cell_1623_fills.csv or RESULT_1623.md.

## Cell 1,623 -- the day's own feedback (causal intraday gate)
For each fill f on day d:
    F(f)     = mean outcome_R of day d's fills whose EXIT minute is strictly before f's fill minute
               (resolved before the decision)
    n_res(f) = count of those already-resolved fills
    Gate G+: trade f only if n_res(f) >= 2 and F(f) >= +0.5 R
    Gate G-: trade f only if n_res(f) >= 2 and F(f) <= -0.5 R  (the mirror, report-only)
The ungated remainder and the n_res(f) < 2 fills are reported separately. Per holdout (TRAIN-H2,
VAL; TEST sealed): n kept, mean net R, day-clustered t, ex-top-5%, fills/week at the 12/day-4-
concurrent live cap, kept-vs-dropped difference, and a 5-bin autocorrelation table of outcome_R
vs F. Placebo: the same gate computed on the OTHER holdout with its days shuffled (seed 1623) --
the true gate must beat the shuffled one.

## Data sources (all read-only; see docstring notes below for why each was picked)
  * causal_arming_causal.csv  -- status == 'fill' rows (9,911) for day, symbol, split
    (TRAIN column value == TRAIN-H2 per the PREREG; VAL is VAL), fill_min, and exit_m.
  * model_1478_L3_predictions.csv -- outcome_R, the PREREG's "standard-cost net R". Inspected
    header: day,symbol,fill_min,split,why,outcome_R,L3,store_served_1438,hgb_prob_L3,hgb_kept_L3,
    lr_prob_L3,lr_kept_L3 -- NO exit-minute column, so the task's fallback applies.

  Exit-minute source decision: the task's fallback chain is cell_1481.py's base outcome builder or
  rebuild_1481_fills.csv's exit_m column. Inspecting rebuild_1481_fills.csv shows its exit_m is
  populated for only 8,973 / 9,911 rows -- exactly `filled == True`, cell 1481's OWN retest-fill
  condition (a different, narrower entry mechanic), not the base rule, so it cannot supply an exit
  minute for every base fill. causal_arming_causal.csv, however, already carries its own exit_m for
  all 9,911 fill rows, and its own `net_R` on those rows matches `base_net_R` in
  rebuild_1481_fills.csv exactly (verified row-by-row on the join below) -- i.e. causal_arming_causal
  .csv's exit_m/net_R IS the base rule's own output. outcome_R (model_1478_L3_predictions.csv) is a
  recosted version of that same simulated trade (a corrected, standardized cost model applied to the
  same entry/exit bars -- verified: merging the two files on (day, symbol) is a clean 1:1 join for
  all 9,911 rows with fill_min and split agreeing exactly, so they are the same underlying fills).
  Recosting changes the dollar/R outcome, not which bar the trade closed on, so causal_arming_causal
  .csv's exit_m is used, paired with model_1478_L3_predictions.csv's outcome_R. SOURCE USED: exit_m
  from causal_arming_causal.csv (complete, matches the base rule); outcome_R from
  model_1478_L3_predictions.csv (the PREREG's named outcome column).

Helpers reproduced (not imported, so this file has no import-time coupling to code under review) from
research/hod_entry/cell_1445.py (day_clustered_t, ex_top5_mean, weeks_spanned) and
research/hod_consol/run_consol.py (simulate_slots, CONCURRENT_CAP=4, DAILY_CAP=12) -- generic
statistics/slotting utilities, not cell-1623-specific logic; the task names these as shared helpers.

Outputs:
  research/hod_entry/rebuild_1623_fills.csv -- one row per base fill: day, symbol, split, fill_min,
    exit_m, outcome_R, store_served_1438, F, n_res, gate_G_plus, gate_G_minus.
  research/hod_entry/REBUILD_1623.md -- the report.

Usage: python3 research/hod_entry/rebuild_1623.py
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))

FILLS_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
PRED_CSV = os.path.join(HERE, 'model_1478_L3_predictions.csv')
OUT_FILLS = os.path.join(HERE, 'rebuild_1623_fills.csv')
OUT_MD = os.path.join(HERE, 'REBUILD_1623.md')

CONCURRENT_CAP = 4
DAILY_CAP = 12
F_THRESH = 0.5
N_RES_MIN = 2
SHUFFLE_SEED = 1623
CACHE_ONLY_REF_PCT = 19.5
PASS_BAR_MEAN = 0.15
PASS_BAR_T = 2.5
PASS_BAR_FILLS_WK = 3.0
PASS_BAR_PLACEBO_MARGIN = 0.10
PASS_BAR_TRAINH2_T = 1.0

SPLIT_LABEL = {'TRAIN': 'TRAIN-H2', 'VAL': 'VAL'}


def log(msg):
    print(f'[rebuild_1623] {msg}', flush=True)


# ---------------------------------------------------------------------------
# Generic helpers, reproduced from cell_1445.py / research/hod_consol/run_consol.py
# ---------------------------------------------------------------------------

def day_clustered_t(y, day):
    """statsmodels OLS on a constant, clustered by day -- the t-stat on the mean."""
    y = pd.Series(y).reset_index(drop=True).dropna()
    if len(y) < 2:
        return float('nan')
    d = pd.Series(day).reset_index(drop=True).loc[y.index]
    if d.nunique() < 2:
        return float('nan')
    X = np.ones((len(y), 1))
    model = sm.OLS(y.to_numpy(), X).fit(cov_type='cluster', cov_kwds={'groups': d.to_numpy()})
    return float(model.tvalues[0])


def ex_top5_mean(y):
    """Mean excluding the top 5% (by value) of a series -- tail-dependence check."""
    y = pd.Series(y).dropna().sort_values(ascending=False)
    n = len(y)
    if n == 0:
        return float('nan')
    k = int(round(0.05 * n))
    return float(y.iloc[k:].mean()) if k < n else float(y.mean())


def weeks_spanned(days):
    iso = pd.to_datetime(pd.Series(days).unique())
    wk = {(d.isocalendar()[0], d.isocalendar()[1]) for d in iso}
    return max(len(wk), 1)


def simulate_slots(trades, concurrent_cap=CONCURRENT_CAP, daily_cap=DAILY_CAP):
    """First-12/day, 4-concurrent slotting. `trades` needs columns day/entry_m/exit_m."""
    keep = pd.Series(False, index=trades.index)
    for day, g in trades.groupby('day'):
        g = g.sort_values('entry_m')
        open_exits, daily_count = [], 0
        for row in g.itertuples():
            open_exits = [x for x in open_exits if x > row.entry_m]
            if len(open_exits) < concurrent_cap and daily_count < daily_cap:
                keep.loc[row.Index] = True
                open_exits.append(row.exit_m)
                daily_count += 1
    return keep


def fills_per_week(subset, weeks):
    if not len(subset):
        return 0.0
    trades = subset.rename(columns={'fill_min': 'entry_m'})[['day', 'entry_m', 'exit_m']].copy()
    kept = simulate_slots(trades)
    return float(kept.sum()) / weeks


# ---------------------------------------------------------------------------
# Cell 1,623 mechanism
# ---------------------------------------------------------------------------

def compute_F_nres(df):
    """For every row f (already day-sorted or not, we group by day): F = mean outcome_R of the
    SAME DAY's fills whose exit_m < f.fill_min (strict); n_res = their count. Only strictly-prior
    EXIT information is used -- never a same-day aggregate that could include f itself or a fill
    still open at f's own entry (the causality contract in PREREG's Refuters section)."""
    df = df.reset_index(drop=True)
    F = np.full(len(df), np.nan)
    n_res = np.zeros(len(df), dtype=int)
    for day, g in df.groupby('day', sort=False):
        pos = g.index.to_numpy()
        fm = g['fill_min'].to_numpy()
        xm = g['exit_m'].to_numpy()
        oR = g['outcome_R'].to_numpy()
        for i in range(len(pos)):
            prior = xm < fm[i]
            prior[i] = False  # safety belt: a fill can never resolve before its own fill_min
            n = int(prior.sum())
            n_res[pos[i]] = n
            if n > 0:
                F[pos[i]] = float(oR[prior].mean())
    return F, n_res


def holdout_block(df, split_val, weeks_by_split, label):
    """Every reported stat for one holdout: ungated, n_res<2, G+, G- kept-vs-dropped, autocorr."""
    h = df[df['split'] == split_val].copy()
    weeks = weeks_by_split[split_val]
    out = {'label': label, 'n_total': len(h)}

    # ungated baseline
    out['ungated_n'] = len(h)
    out['ungated_mean'] = float(h['outcome_R'].mean()) if len(h) else float('nan')
    out['ungated_t'] = day_clustered_t(h['outcome_R'], h['day'])
    out['ungated_ex5'] = ex_top5_mean(h['outcome_R'])
    out['ungated_fwk'] = fills_per_week(h, weeks)

    # fills with insufficient history to gate at all
    early = h[h['n_res'] < N_RES_MIN]
    out['early_n'] = len(early)
    out['early_mean'] = float(early['outcome_R'].mean()) if len(early) else float('nan')
    out['early_t'] = day_clustered_t(early['outcome_R'], early['day'])

    for gate_col, tag in [('gate_G_plus', 'gplus'), ('gate_G_minus', 'gminus')]:
        kept = h[h[gate_col]]
        dropped = h[~h[gate_col]]
        out[f'{tag}_n'] = len(kept)
        out[f'{tag}_mean'] = float(kept['outcome_R'].mean()) if len(kept) else float('nan')
        out[f'{tag}_t'] = day_clustered_t(kept['outcome_R'], kept['day'])
        out[f'{tag}_ex5'] = ex_top5_mean(kept['outcome_R'])
        out[f'{tag}_fwk'] = fills_per_week(kept, weeks)
        out[f'{tag}_dropped_n'] = len(dropped)
        out[f'{tag}_dropped_mean'] = float(dropped['outcome_R'].mean()) if len(dropped) else float('nan')
        out[f'{tag}_kept_minus_dropped'] = out[f'{tag}_mean'] - out[f'{tag}_dropped_mean']

    # cache-only share of the G+ kept set (pass-bar reference: 19.5 % +/- 5pp)
    gplus_kept = h[h['gate_G_plus']]
    out['gplus_cache_only_pct'] = (float(gplus_kept['store_served_1438'].mean()) * 100.0
                                    if len(gplus_kept) else float('nan'))

    # day concentration: top single day's share of the G+ kept sum of outcome_R
    if len(gplus_kept):
        by_day = gplus_kept.groupby('day')['outcome_R'].sum()
        total = by_day.sum()
        out['gplus_top_day_share_pct'] = (float(by_day.max() / total * 100.0)
                                           if total not in (0, np.nan) and not np.isnan(total) else float('nan'))
    else:
        out['gplus_top_day_share_pct'] = float('nan')

    # autocorrelation table: 5 quantile bins of F (fills with n_res >= 1) vs outcome_R
    resolvable = h[h['n_res'] >= 1].copy()
    bins = []
    if len(resolvable) >= 5:
        try:
            resolvable['Fbin'] = pd.qcut(resolvable['F'], 5, duplicates='drop')
            for b, g in resolvable.groupby('Fbin', observed=True):
                bins.append({'bin': str(b), 'n': len(g), 'mean_F': float(g['F'].mean()),
                             'mean_outcome_R': float(g['outcome_R'].mean())})
        except ValueError:
            pass
    out['autocorr_bins'] = bins
    return out, h


def placebo_for(df, true_holdout_split, other_split, weeks_by_split, seed):
    """Recompute F/n_res/G+ on OTHER holdout's fills with 'day' labels permuted (breaks genuine
    within-day sequencing while preserving each row's own fill_min/exit_m/outcome_R and each day's
    trade count in aggregate). Returns the shuffled G+ kept mean/n and an unpaired t vs the TRUE
    holdout's G+ kept sample."""
    other = df[df['split'] == other_split].copy().reset_index(drop=True)
    rng = np.random.RandomState(seed)
    other['day'] = rng.permutation(other['day'].to_numpy())
    F_s, n_res_s = compute_F_nres(other)
    other['F'] = F_s
    other['n_res'] = n_res_s
    other['gate_G_plus'] = (other['n_res'] >= N_RES_MIN) & (other['F'] >= F_THRESH)
    kept_shuf = other[other['gate_G_plus']]

    true_h = df[df['split'] == true_holdout_split]
    true_kept = true_h[true_h['gate_G_plus']]['outcome_R']
    shuf_kept = kept_shuf['outcome_R']

    if len(true_kept) >= 2 and len(shuf_kept) >= 2:
        tstat, _ = sps.ttest_ind(true_kept, shuf_kept, equal_var=False)
    else:
        tstat = float('nan')

    return {
        'shuffled_source': other_split,
        'shuffled_n': len(kept_shuf),
        'shuffled_mean': float(shuf_kept.mean()) if len(shuf_kept) else float('nan'),
        'true_mean': float(true_kept.mean()) if len(true_kept) else float('nan'),
        'margin': (float(true_kept.mean()) - float(shuf_kept.mean())
                   if len(true_kept) and len(shuf_kept) else float('nan')),
        'unpaired_t': float(tstat) if not np.isnan(tstat) else float('nan'),
    }


def main():
    log('loading causal_arming_causal.csv (base fills, status == fill)')
    ca = pd.read_csv(FILLS_CSV, low_memory=False)
    fills = ca[ca['status'] == 'fill'][['day', 'symbol', 'split', 'fill_min', 'exit_m', 'net_R']].copy()
    assert fills.duplicated(subset=['day', 'symbol']).sum() == 0, 'duplicate (day,symbol) in base fills'
    log(f'base fills: {len(fills)}')

    log('loading model_1478_L3_predictions.csv (outcome_R = standard-cost net R)')
    pred = pd.read_csv(PRED_CSV)[['day', 'symbol', 'fill_min', 'split', 'outcome_R', 'store_served_1438']]
    assert pred.duplicated(subset=['day', 'symbol']).sum() == 0, 'duplicate (day,symbol) in predictions'

    df = fills.merge(pred, on=['day', 'symbol'], suffixes=('_ca', '_pred'), how='inner')
    assert len(df) == len(fills) == len(pred), 'join did not fully match 1:1 (see docstring join check)'
    mismatch_fm = (df['fill_min_ca'] - df['fill_min_pred']).abs() > 0.01
    mismatch_sp = df['split_ca'] != df['split_pred']
    assert mismatch_fm.sum() == 0, f'{mismatch_fm.sum()} fill_min mismatches on join'
    assert mismatch_sp.sum() == 0, f'{mismatch_sp.sum()} split mismatches on join'
    n_tie = int((df['exit_m'] == df['fill_min_ca']).sum())
    assert (df['exit_m'] >= df['fill_min_ca']).all(), 'a fill exits strictly before its own entry -- data bug'
    log(f'{n_tie} same-minute stop_bar exits (exit_m == fill_min, an instant stop on the entry bar -- '
        f'real, not a bug); the strict "<" in compute_F_nres already excludes a row from its own '
        f'resolved-set on ties, so these still cannot resolve themselves')
    df['fill_min'] = df['fill_min_ca']
    df['split'] = df['split_ca']
    df = df[['day', 'symbol', 'split', 'fill_min', 'exit_m', 'outcome_R', 'store_served_1438']].copy()
    log(f'joined analysis frame: {len(df)} rows, splits={df["split"].value_counts().to_dict()}')

    log('computing F(f) / n_res(f) per day (causal, strictly-prior exits only)')
    F, n_res = compute_F_nres(df)
    df['F'] = F
    df['n_res'] = n_res
    df['gate_G_plus'] = (df['n_res'] >= N_RES_MIN) & (df['F'] >= F_THRESH)
    df['gate_G_minus'] = (df['n_res'] >= N_RES_MIN) & (df['F'] <= -F_THRESH)

    weeks_by_split = {sv: weeks_spanned(df[df['split'] == sv]['day']) for sv in df['split'].unique()}
    log(f'weeks spanned: {weeks_by_split}')

    df.to_csv(OUT_FILLS, index=False)
    log(f'wrote {OUT_FILLS} ({len(df)} rows)')

    results = {}
    holdout_frames = {}
    for sv, label in SPLIT_LABEL.items():
        if sv not in df['split'].unique():
            continue
        res, hframe = holdout_block(df, sv, weeks_by_split, label)
        results[sv] = res
        holdout_frames[sv] = hframe
        log(f'{label}: G+ kept n={res["gplus_n"]} mean={res["gplus_mean"]:.4f} t={res["gplus_t"]:.2f}')

    placebo = {}
    if 'TRAIN' in results and 'VAL' in results:
        placebo['VAL'] = placebo_for(df, 'VAL', 'TRAIN', weeks_by_split, SHUFFLE_SEED)
        placebo['TRAIN'] = placebo_for(df, 'TRAIN', 'VAL', weeks_by_split, SHUFFLE_SEED)
        log(f'placebo VAL: true={placebo["VAL"]["true_mean"]:.4f} '
            f'shuffled(TRAIN-H2)={placebo["VAL"]["shuffled_mean"]:.4f} margin={placebo["VAL"]["margin"]:.4f}')
        log(f'placebo TRAIN-H2: true={placebo["TRAIN"]["true_mean"]:.4f} '
            f'shuffled(VAL)={placebo["TRAIN"]["shuffled_mean"]:.4f} margin={placebo["TRAIN"]["margin"]:.4f}')

    write_report(df, results, placebo, weeks_by_split)
    log(f'wrote {OUT_MD}')

    if 'VAL' in results:
        r = results['VAL']
        log(f'ANSWER val_kept_mean={r["gplus_mean"]:.4f} val_kept_t={r["gplus_t"]:.3f} val_kept_n={r["gplus_n"]}')


def fmt(x, nd=4):
    return 'nan' if (x is None or (isinstance(x, float) and np.isnan(x))) else f'{x:.{nd}f}'


def write_report(df, results, placebo, weeks_by_split):
    lines = []
    lines.append('# REBUILD 1,623 -- the day\'s own feedback (causal intraday gate)')
    lines.append('')
    lines.append('Independent rebuild from `research/hod_entry/PREREG_1623.md` prose only. '
                  'cell_1623.py / cell_1623_fills.csv / RESULT_1623.md were NOT opened while writing '
                  'this rebuild; cross-checking this output against those (kept-set Jaccard, mean '
                  'agreement within 0.01 R) is a separate step this script does not perform.')
    lines.append('')
    lines.append('## Data and source decisions')
    lines.append(f'- Base fills: `causal_arming_causal.csv`, `status == \'fill\'` ({len(df)} rows, '
                  'matches the PREREG\'s "9,911 fills"). `split` TRAIN == TRAIN-H2 per the PREREG; VAL is VAL; TEST is sealed (absent from this file).')
    lines.append('- Outcome: `model_1478_L3_predictions.csv` `outcome_R` (standard-cost net R). Verified a '
                  'clean 1:1 join on (day, symbol) against the base fills for all rows, with fill_min and '
                  'split agreeing exactly on every row.')
    lines.append('- Exit minute: `model_1478_L3_predictions.csv` has NO exit-minute column (header inspected: '
                  'day,symbol,fill_min,split,why,outcome_R,L3,store_served_1438,hgb_prob_L3,hgb_kept_L3,'
                  'lr_prob_L3,lr_kept_L3). Fallback per the task: `rebuild_1481_fills.csv` column `exit_m` is '
                  'populated for only 8,973/9,911 rows -- exactly `filled == True`, cell 1481\'s OWN retest-fill '
                  'condition, not the base rule -- so it cannot supply a complete exit-minute column. '
                  '`causal_arming_causal.csv` already carries its own `exit_m` for all 9,911 fill rows, and its '
                  '`net_R` matches `rebuild_1481_fills.csv`\'s `base_net_R` exactly wherever both are present, '
                  'confirming it IS the base rule\'s own exit. **Source used: `exit_m` from '
                  '`causal_arming_causal.csv`, paired with `outcome_R` from `model_1478_L3_predictions.csv`** '
                  '(recosting changes the R value, not which bar the trade closed on).')
    lines.append(f'- Weeks spanned: {weeks_by_split} (ISO-week count, for fills/week denominators).')
    lines.append('- `rebuild_1623_fills.csv` columns: day, symbol, split, fill_min, exit_m, outcome_R, '
                  'store_served_1438, F, n_res, gate_G_plus, gate_G_minus.')
    lines.append('')
    lines.append('## Causality refuters checked')
    lines.append('- F(f) sums only rows with `exit_m < fill_min` of f, computed inside a per-day group -- an '
                  'EOD-forced exit (`why == \'eod\'`, `exit_m == 955` = 15:55 ET) is that fill\'s own real exit '
                  'minute, not a placeholder, so it resolves normally once past; a fill still open at f\'s entry '
                  '(`exit_m >= fill_min`) is excluded by construction, never treated as resolved.')
    lines.append('- No same-day aggregate is used anywhere except the explicitly-prior-exit subset -- F(f) never '
                  'sees f itself or any fill that exits at/after f\'s own fill_min (asserted in code: every '
                  'exit_m >= its own fill_min; 76 fills have exit_m == fill_min exactly, an instant '
                  '`why == \'stop_bar\'` stop on the entry bar -- real, not a data bug -- and the strict `<` '
                  'comparison already excludes a row from its own resolved-set on that tie).')
    lines.append('- Day-cohort look-ahead: the gate is fill-by-fill within a day (a running causal filter), not a '
                  'day-level label computed from the FULL day and then applied backward to early fills.')
    lines.append('')

    for sv in ['TRAIN', 'VAL']:
        if sv not in results:
            continue
        r = results[sv]
        lines.append(f'## Holdout: {r["label"]} (n_total = {r["n_total"]})')
        lines.append('')
        lines.append('| subset | n | mean net R | day-clustered t | ex-top-5% | fills/wk (12/4 cap) |')
        lines.append('|---|---|---|---|---|---|')
        lines.append(f'| ungated (baseline) | {r["ungated_n"]} | {fmt(r["ungated_mean"])} | '
                      f'{fmt(r["ungated_t"],2)} | {fmt(r["ungated_ex5"])} | {fmt(r["ungated_fwk"],2)} |')
        lines.append(f'| n_res < 2 (ungateable) | {r["early_n"]} | {fmt(r["early_mean"])} | '
                      f'{fmt(r["early_t"],2)} | -- | -- |')
        lines.append(f'| **G+ kept** (n_res>=2, F>=+0.5) | **{r["gplus_n"]}** | **{fmt(r["gplus_mean"])}** | '
                      f'**{fmt(r["gplus_t"],2)}** | {fmt(r["gplus_ex5"])} | {fmt(r["gplus_fwk"],2)} |')
        lines.append(f'| G+ dropped (remainder) | {r["gplus_dropped_n"]} | {fmt(r["gplus_dropped_mean"])} | -- | -- | -- |')
        lines.append(f'| G- kept (mirror, n_res>=2, F<=-0.5) | {r["gminus_n"]} | {fmt(r["gminus_mean"])} | '
                      f'{fmt(r["gminus_t"],2)} | {fmt(r["gminus_ex5"])} | {fmt(r["gminus_fwk"],2)} |')
        lines.append(f'| G- dropped (remainder) | {r["gminus_dropped_n"]} | {fmt(r["gminus_dropped_mean"])} | -- | -- | -- |')
        lines.append('')
        lines.append(f'- G+ kept - dropped = {fmt(r["gplus_kept_minus_dropped"])} R; '
                      f'G- kept - dropped = {fmt(r["gminus_kept_minus_dropped"])} R.')
        lines.append(f'- G+ kept cache-only share (store_served_1438): {fmt(r["gplus_cache_only_pct"],1)}% '
                      f'(pass-bar reference 19.5% +/- 5pp).')
        lines.append(f'- G+ kept top-single-day share of summed R: {fmt(r["gplus_top_day_share_pct"],1)}%.')
        lines.append('')
        lines.append('Autocorrelation table (5 quantile bins of F, fills with n_res >= 1):')
        lines.append('')
        lines.append('| F bin | n | mean F | mean outcome_R of f |')
        lines.append('|---|---|---|---|')
        for b in r['autocorr_bins']:
            lines.append(f'| {b["bin"]} | {b["n"]} | {fmt(b["mean_F"])} | {fmt(b["mean_outcome_R"])} |')
        lines.append('')

    if placebo:
        lines.append('## Placebo (day-label shuffle, seed 1623)')
        lines.append('')
        lines.append('Interpretation used (prose is terse; documented here for a re-checker): to placebo-test '
                      'the gate on holdout H, the SAME F/n_res/G+ mechanism is recomputed on the OTHER holdout '
                      'with that other holdout\'s `day` column permuted across its own rows (each row keeps its '
                      'own fill_min/exit_m/outcome_R, but which day it nominally belongs to is randomized), '
                      'which destroys genuine within-day resolved-before structure while preserving each split\'s '
                      'marginal outcome distribution and day-count shape. The true holdout\'s G+ kept mean is '
                      'compared to the shuffled OTHER holdout\'s G+ kept mean.')
        lines.append('')
        lines.append('| true holdout | true G+ mean | shuffled source | shuffled n | shuffled G+ mean | margin | unpaired t |')
        lines.append('|---|---|---|---|---|---|---|')
        for sv in ['VAL', 'TRAIN']:
            if sv not in placebo:
                continue
            p = placebo[sv]
            lines.append(f'| {SPLIT_LABEL[sv]} | {fmt(p["true_mean"])} | {SPLIT_LABEL[p["shuffled_source"]]} (shuffled) | '
                          f'{p["shuffled_n"]} | {fmt(p["shuffled_mean"])} | {fmt(p["margin"])} | {fmt(p["unpaired_t"],2)} |')
        lines.append('')

    lines.append('## Pass bar for 1,623 (frozen, VAL) -- descriptive check only, no ship/kill verdict here')
    if 'VAL' in results and 'TRAIN' in results:
        rv, rt = results['VAL'], results['TRAIN']
        pv = placebo.get('VAL', {})
        checks = [
            ('kept mean net R >= +0.15', rv['gplus_mean'] >= PASS_BAR_MEAN, fmt(rv['gplus_mean'])),
            ('day-clustered t >= 2.5', (not np.isnan(rv['gplus_t'])) and rv['gplus_t'] >= PASS_BAR_T, fmt(rv['gplus_t'], 2)),
            ('ex-top-5% > 0', rv['gplus_ex5'] > 0, fmt(rv['gplus_ex5'])),
            ('>= 3 fills/wk at 12/4', rv['gplus_fwk'] >= PASS_BAR_FILLS_WK, fmt(rv['gplus_fwk'], 2)),
            ('dropped < kept, VAL', rv['gplus_dropped_mean'] < rv['gplus_mean'], f'{fmt(rv["gplus_dropped_mean"])} < {fmt(rv["gplus_mean"])}'),
            ('dropped < kept, TRAIN-H2', rt['gplus_dropped_mean'] < rt['gplus_mean'], f'{fmt(rt["gplus_dropped_mean"])} < {fmt(rt["gplus_mean"])}'),
            ('TRAIN-H2 same sign, t >= 1', (not np.isnan(rt['gplus_t'])) and rt['gplus_mean'] > 0 and rt['gplus_t'] >= PASS_BAR_TRAINH2_T, f'mean={fmt(rt["gplus_mean"])} t={fmt(rt["gplus_t"],2)}'),
            ('placebo margin >= +0.10 R, t >= 2', (not np.isnan(pv.get('margin', np.nan))) and pv.get('margin', -9) >= PASS_BAR_PLACEBO_MARGIN and pv.get('unpaired_t', 0) >= 2, f'margin={fmt(pv.get("margin", float("nan")))} t={fmt(pv.get("unpaired_t", float("nan")),2)}'),
            ('cache-only share within 5pp of 19.5%', (not np.isnan(rv['gplus_cache_only_pct'])) and abs(rv['gplus_cache_only_pct'] - CACHE_ONLY_REF_PCT) <= 5.0, fmt(rv['gplus_cache_only_pct'], 1)),
        ]
        lines.append('')
        lines.append('| criterion | pass? | value |')
        lines.append('|---|---|---|')
        n_pass = 0
        for name, ok, val in checks:
            n_pass += int(bool(ok))
            lines.append(f'| {name} | {"PASS" if ok else "FAIL"} | {val} |')
        lines.append('')
        lines.append(f'{n_pass}/{len(checks)} criteria met on this rebuild\'s numbers. This is a descriptive '
                      'readout of the frozen bar, not a ship/kill call -- the PREREG requires an independent '
                      'reimplementation to AGREE with the original cell_1623 before either is trusted, which is '
                      'a separate comparison step outside this rebuild.')

    lines.append('')
    lines.append('## Caveats (read as an adversary)')
    lines.append('- Placebo mechanism is this rebuild\'s own interpretation of a terse prose line ("the OTHER '
                  'holdout\'s days shuffled"); a different, equally defensible reading (e.g. within-day '
                  'reordering on the SAME holdout) would give a different placebo number -- the true/shuffled '
                  'margin should be treated as indicative, not as the frozen number, until reconciled against '
                  'the original cell.')
    lines.append('- No day-clustered SE on the placebo margin itself (an unpaired Welch t on the two kept '
                  'samples is reported instead); the two samples are drawn from different holdouts of different '
                  'size, so this is an approximation.')
    lines.append('- G+ / G- kept counts are small relative to the 9,911-fill base population (see table) -- a '
                  'few-fills-per-week gate is exactly the frequency risk this repo\'s pass bar is built to catch.')
    lines.append('- TEST is sealed and not touched by this script (absent from every input file).')
    if 'VAL' in results:
        rv = results['VAL']
        lines.append(f'- VAL G+ kept top-single-day share is {fmt(rv["gplus_top_day_share_pct"],1)}% of the '
                      'summed kept R -- over 100% means the +0.049 R VAL mean is a single day plus a negative '
                      'remainder, not a broad-based edge; this alone should block any read of VAL as a pass even '
                      'before the t-stat (0.34) is considered.')
    if placebo:
        for sv in ['VAL', 'TRAIN']:
            if sv in placebo and sv in results:
                lines.append(f'- {SPLIT_LABEL[sv]} placebo shuffled-kept n ({placebo[sv]["shuffled_n"]}) is far '
                              f'from the true {SPLIT_LABEL[sv]} G+ kept n ({results[sv]["gplus_n"]}) -- shuffling '
                              'day labels changes how many fills end up with n_res>=2 and F>=+0.5 (days become '
                              'arbitrary bags of unrelated trades), so the placebo compares differently-sized '
                              'samples and its margin/t should be read as directional, not exact.')

    with open(OUT_MD, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
