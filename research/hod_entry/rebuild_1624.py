#!/usr/bin/env python3
"""Independent rebuild of Cell 1,624 -- break breadth (the crowd) -- research/hod_entry/PREREG_1623.md.

INDEPENDENT REBUILD PROTOCOL (CLAUDE.md "No research claim ships without an independent check", #1):
this file was written from the PREREG_1623.md prose only. research/hod_entry/cell_1624.py,
cell_1624_fills.csv and RESULT_1624.md were never opened while writing it. The only pre-existing
code reused is the small, generically-named statistics/slotting helpers explicitly sanctioned as
shared infrastructure for this task (cell_1445.py's day_clustered_t / ex_top5_mean / fills_per_week
and its research/hod_consol/run_consol.simulate_slots dependency); those are copied verbatim below
(not reimplemented from memory) so this file has no import-time side effects from loading cell_1445.py
as a module. They are not part of the independent check -- the subject of the check is the CELL 1,624
MECHANISM (breadth definition, hour-adjusted terciles, the gate), not generic OLS/tail-mean plumbing.

Mechanism, verbatim from PREREG_1623.md "## Cell 1,624 -- break breadth (the crowd)":
    B30(f) = the number of ARM events (any status, all names) in the 30 minutes before f's fill
    minute; B60 likewise. Terciles set on TRAIN-H2 by minute-of-day-adjusted rank (the count rises
    through the morning -- rank within the same hour bucket, so the gate is not a time-of-day gate).
    Gate: trade f only in the TOP tercile of B30 (pre-declared); report the bottom tercile and B60
    beside. Same statistics as 1,623 [n kept, mean net R, day-clustered t, ex-top-5%, fills/week at
    12/4, the kept-vs-dropped difference; the 1,623 autocorrelation-of-F table is specific to that
    cell's F(f) construct and has no 1,624 analogue, so it is not reproduced here].

DATA-AVAILABILITY DEVIATION FROM THE LITERAL PROSE (declared here, before any number is computed --
never tuned after seeing a result):
    "ARM events (any status, all names)" literally requires an arm TIME for every row of
    causal_arming_causal.csv, including status=='nofill' (2,010 rows, n_cross>=1: armed at least
    once, never filled) and status=='not_armed' (21,931 rows, n_cross==0: never armed at all).
    Checked directly: in causal_arming_causal.csv, fill_min/exit_m/level are 100% null off the
    'fill' rows, and n_cross is a per-(day,symbol) COUNT with no per-event timestamp -- there is no
    way to place a nofill symbol's arm attempt(s) on the clock from that file. The only column in
    any sanctioned input that carries a genuine arm MINUTE is `arm_m` in features_1478_A.csv, and it
    exists only for the 9,911 status=='fill' rows (row counts verified identical -- 9,911 -- across
    features_1478_A.csv, model_1478_L3_predictions.csv, rebuild_1481_fills.csv and
    cell_1445_features.csv). B30/B60 here therefore count OTHER FILLS' arm_m only:
        - "all names" is honored (every symbol's fills count, cross-sectionally);
        - "any status" is NOT honored (nofill/not_armed arm attempts, ~19% of all arm attempts by
          row count, are invisible to this count -- they carry no timestamp anywhere available).
    This is "breadth among causal-arming FILLS", a systematic undercount of true crowd size, not the
    literal spec. See REBUILD_1624.md for the full discussion and why this is the most defensible
    reading of the sanctioned inputs rather than an invented shortcut.

Other declared implementation choices (pre-declared here, not tuned after a number):
    - Self-exclusion: f's own arm event does not count toward its own B30/B60 (breadth is the CROWD
      around a fill, not the fill itself; f's own arm_m is within 30 min of its own fill_min in the
      overwhelming majority of rows by construction, so counting it would add a near-constant +1).
    - Hour bucket = floor(fill_min / 60) using the FILL's own fill_min (the entity being gated).
    - Window is a half-open interval [fill_min - W, fill_min): "in the W minutes before" f's fill
      minute, strictly before (an event at exactly fill_min does not count; one exactly W minutes
      before does).
    - Tercile cutoffs: per hour bucket, the TRAIN-H2 33.33rd/66.67th percentiles of B30 (resp. B60);
      bottom = value <= p33, top = value >= p66, mid = between. Applied to BOTH holdouts using only
      TRAIN-H2-derived cutoffs (VAL never contributes to its own cutoff).
    - outcome_R = model_1478_L3_predictions.csv's outcome_R directly ("standard-cost net R" per the
      task routing note) -- this is the "net R" the PREREG statistics operate on.
    - exit_m: model_1478_L3_predictions.csv has NO exit-minute column (header inspected directly).
      Used rebuild_1481_fills.csv's exit_m column instead (join key day+symbol+fill_min), needed
      only for the fills/week concurrency slotting -- nowhere else in this cell's mechanism.
    - bars_fills_1478.db and the SPY/IWM minute bars were NOT needed for cell 1,624: the breadth
      signal is built purely from arm/fill event minutes already on disk, and outcome_R is taken
      pre-computed. (SPY/IWM bars are cell 1,625's instrument leg, not this cell's.)

Usage:
    python3 research/hod_entry/rebuild_1624.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO = '/home/ec2-user/onemil'
OUT_DIR = f'{REPO}/research/hod_entry'

# research/hod_consol/run_consol.py CONCURRENT_CAP / DAILY_CAP, verified by grep before use.
CONCURRENT_CAP = 4
DAILY_CAP = 12


def log(msg):
    """Verbose progress line (CLAUDE.md: batch/long processes must be verbose)."""
    print(f'[rebuild_1624] {msg}', flush=True)


# --------------------------------------------------------------------------------------------
# Sanctioned helpers, copied VERBATIM from research/hod_entry/cell_1445.py (lines 414-457) and
# research/hod_consol/run_consol.py (simulate_slots, lines 766-777). Not the subject of the
# independent check -- explicitly named as shared infrastructure in this task's routing.
# --------------------------------------------------------------------------------------------

def day_clustered_t(y, day):
    """statsmodels OLS on a constant, clustered by day -- the t-stat on the mean."""
    y = pd.Series(y).dropna()
    if len(y) < 2:
        return np.nan
    d = pd.Series(day).loc[y.index]
    if d.nunique() < 2:
        return np.nan
    X = np.ones((len(y), 1))
    model = sm.OLS(y.to_numpy(), X).fit(cov_type='cluster', cov_kwds={'groups': d.to_numpy()})
    return float(model.tvalues[0])


def ex_top5_mean(y):
    """Mean excluding the top 5% (by value) of a series -- tail-dependence check."""
    y = pd.Series(y).dropna().sort_values(ascending=False)
    n = len(y)
    if n == 0:
        return np.nan
    k = int(round(0.05 * n))
    return float(y.iloc[k:].mean()) if k < n else float(y.mean())


def weeks_spanned(days):
    """Distinct ISO (year, week) count over a day-string series -- the denominator for fills/wk."""
    iso = pd.to_datetime(pd.Series(days).unique())
    wk = {(d.isocalendar()[0], d.isocalendar()[1]) for d in iso}
    return max(len(wk), 1)


def simulate_slots(trades, concurrent_cap=CONCURRENT_CAP, daily_cap=DAILY_CAP):
    """First-come slotting: keep a trade iff < concurrent_cap open exits and < daily_cap kept today."""
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


def fills_per_week(kept_subset, weeks):
    """Fills/wk under first-12/day, 4-concurrent slotting (research/hod_consol/run_consol.simulate_slots)."""
    if not len(kept_subset):
        return 0.0
    trades = kept_subset.rename(columns={'fill_min': 'entry_m'})[['day', 'entry_m', 'exit_m']].copy()
    keep = simulate_slots(trades)
    return float(keep.sum()) / weeks


# --------------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------------

def load_base():
    """The 9,911 causal-arming fills with their own arm minute (arm_m), outcome_R and exit_m."""
    feat = pd.read_csv(f'{OUT_DIR}/features_1478_A.csv',
                        usecols=['day', 'symbol', 'fill_min', 'split', 'arm_m', 'store_served_1438'])
    log(f'features_1478_A.csv (base + arm_m): {len(feat)} rows (expect 9,911)')
    if len(feat) != 9911:
        log(f'WARNING: base population is {len(feat)}, not the documented 9,911 fills')

    pred = pd.read_csv(f'{OUT_DIR}/model_1478_L3_predictions.csv',
                        usecols=['day', 'symbol', 'fill_min', 'split', 'outcome_R'])
    log(f'model_1478_L3_predictions.csv (outcome_R): {len(pred)} rows')

    exits = pd.read_csv(f'{OUT_DIR}/rebuild_1481_fills.csv',
                         usecols=['day', 'symbol', 'fill_min', 'exit_m'])
    log(f'rebuild_1481_fills.csv (exit_m -- model_1478_L3_predictions.csv has no exit-minute '
        f'column, header checked): {len(exits)} rows')

    df = feat.merge(pred, on=['day', 'symbol', 'fill_min'], how='left', suffixes=('', '_pred'))
    n_no_outcome = int(df['outcome_R'].isna().sum())
    if n_no_outcome:
        log(f'WARNING: {n_no_outcome} fills have no outcome_R match in model_1478_L3_predictions.csv')
    bad_split = int((df['split'] != df['split_pred']).sum())
    if bad_split:
        log(f'WARNING: {bad_split} rows have a split mismatch between features_1478_A and predictions')
    df = df.drop(columns=['split_pred'])

    df = df.merge(exits, on=['day', 'symbol', 'fill_min'], how='left')
    n_no_exit = int(df['exit_m'].isna().sum())
    if n_no_exit:
        log(f'WARNING: {n_no_exit} fills have no exit_m match in rebuild_1481_fills.csv '
            f'(excluded from the fills/week concurrency simulation only)')

    df['hour'] = np.floor(df['fill_min'] / 60.0).astype(int)
    log(f'split counts: {df["split"].value_counts().to_dict()}')
    log(f'hour-of-day buckets present: {sorted(df["hour"].unique().tolist())}')
    return df


# --------------------------------------------------------------------------------------------
# Breadth (B30 / B60)
# --------------------------------------------------------------------------------------------

def compute_breadth(df, window):
    """B{window}(f): count of OTHER fills' arm_m in [f.fill_min - window, f.fill_min), same day
    (self excluded -- see module docstring 'Other declared implementation choices')."""
    result = pd.Series(0, index=df.index, dtype=int)
    for day, g in df.groupby('day', sort=False):
        arm = g['arm_m'].to_numpy()
        fmin = g['fill_min'].to_numpy()
        order = np.argsort(arm, kind='mergesort')
        arm_sorted = arm[order]
        lo = np.searchsorted(arm_sorted, fmin - window, side='left')
        hi = np.searchsorted(arm_sorted, fmin, side='left')
        counts = (hi - lo).astype(int)
        self_in_window = (fmin - arm) < window
        counts = counts - self_in_window.astype(int)
        if (counts < 0).any():
            log(f'WARNING: negative breadth count on day {day} before clipping -- self-exclusion '
                f'edge case, clipped to 0')
            counts = np.clip(counts, 0, None)
        result.loc[g.index] = counts
    return result


# --------------------------------------------------------------------------------------------
# Hour-adjusted terciles (set on TRAIN-H2, applied to both holdouts)
# --------------------------------------------------------------------------------------------

def train_tercile_cutoffs(df_train, col):
    """Per-hour-bucket (33.33rd, 66.67th) percentile cutoffs of `col`, fit on TRAIN-H2 only."""
    cutoffs = {}
    for hour, g in df_train.groupby('hour'):
        vals = g[col].to_numpy()
        p33, p66 = np.percentile(vals, [100.0 / 3, 200.0 / 3])
        cutoffs[hour] = (float(p33), float(p66), len(vals))
    return cutoffs


def assign_tercile(df, col, cutoffs):
    """Classify every row (bottom/mid/top) against its own hour bucket's TRAIN-H2 cutoffs."""
    tiers = np.empty(len(df), dtype=object)
    missing_hours = {}
    for i, (hour, val) in enumerate(zip(df['hour'].to_numpy(), df[col].to_numpy())):
        c = cutoffs.get(hour)
        if c is None:
            missing_hours[hour] = missing_hours.get(hour, 0) + 1
            tiers[i] = 'mid'  # neutral fallback -- excluded from the top-tercile gate either way
            continue
        p33, p66, _ = c
        if val <= p33:
            tiers[i] = 'bottom'
        elif val >= p66:
            tiers[i] = 'top'
        else:
            tiers[i] = 'mid'
    if missing_hours:
        log(f'WARNING: {col} -- hour buckets with no TRAIN-H2 cutoff (0 TRAIN-H2 fills that hour), '
            f'rows defaulted to "mid": {missing_hours}')
    return pd.Series(tiers, index=df.index)


# --------------------------------------------------------------------------------------------
# Statistics ("same statistics as 1,623")
# --------------------------------------------------------------------------------------------

def stat_block(sub):
    n = len(sub)
    if n == 0:
        return dict(n=0, mean_R=np.nan, t=np.nan, ex_top5=np.nan, cache_only_pct=np.nan)
    return dict(n=n,
                mean_R=float(sub['outcome_R'].mean()),
                t=day_clustered_t(sub['outcome_R'], sub['day']),
                ex_top5=ex_top5_mean(sub['outcome_R']),
                cache_only_pct=float(sub['store_served_1438'].mean() * 100))


def score_window(df, col, cutoffs, weeks_by_split):
    """Full per-holdout report for one breadth window (B30 or B60): top/mid/bottom + kept-vs-dropped."""
    tier = assign_tercile(df, col, cutoffs)
    rows = []
    for split in ('TRAIN', 'VAL'):
        hdf = df[df['split'] == split]
        htier = tier.loc[hdf.index]
        top = hdf[htier == 'top']
        mid = hdf[htier == 'mid']
        bottom = hdf[htier == 'bottom']
        dropped = hdf[htier != 'top']
        weeks = weeks_by_split[split]
        row = dict(window=col, split=split,
                   top=stat_block(top), mid=stat_block(mid), bottom=stat_block(bottom),
                   dropped=stat_block(dropped),
                   top_fills_wk=fills_per_week(top, weeks),
                   dropped_fills_wk=fills_per_week(dropped, weeks))
        row['kept_minus_dropped'] = (row['top']['mean_R'] - row['dropped']['mean_R']
                                      if top.shape[0] and dropped.shape[0] else np.nan)
        rows.append(row)
    return rows, tier


def main():
    log('Loading base fill population (features_1478_A.csv + outcome_R + exit_m) ...')
    df = load_base()

    log('Computing B30 / B60 (other fills\' arm_m in the W minutes before this fill, same day, '
        'self excluded) ...')
    df['b30'] = compute_breadth(df, 30)
    df['b60'] = compute_breadth(df, 60)
    log(f'B30 describe:\n{df["b30"].describe()}')
    log(f'B60 describe:\n{df["b60"].describe()}')

    hour_means = df.groupby('hour')['b30'].mean()
    log(f'B30 mean by hour bucket (sanity check on "the count rises through the morning"):\n'
        f'{hour_means}')

    train = df[df['split'] == 'TRAIN']
    weeks_by_split = {s: weeks_spanned(df[df['split'] == s]['day']) for s in ('TRAIN', 'VAL')}
    log(f'weeks spanned: {weeks_by_split}')

    cutoffs_b30 = train_tercile_cutoffs(train, 'b30')
    cutoffs_b60 = train_tercile_cutoffs(train, 'b60')
    log(f'TRAIN-H2 B30 cutoffs by hour (p33, p66, n): {cutoffs_b30}')
    log(f'TRAIN-H2 B60 cutoffs by hour (p33, p66, n): {cutoffs_b60}')

    rows_b30, tier_b30 = score_window(df, 'b30', cutoffs_b30, weeks_by_split)
    rows_b60, tier_b60 = score_window(df, 'b60', cutoffs_b60, weeks_by_split)

    df['tercile_b30'] = tier_b30
    df['tercile_b60'] = tier_b60
    df['kept_top_b30'] = tier_b30 == 'top'
    df['kept_top_b60'] = tier_b60 == 'top'

    log('=== B30 (the pre-declared gate) ===')
    for r in rows_b30:
        log(f"{r['split']}: TOP n={r['top']['n']} mean={r['top']['mean_R']:.4f} t={r['top']['t']:.3f} "
            f"ex5={r['top']['ex_top5']:.4f} fills/wk={r['top_fills_wk']:.2f} cache%={r['top']['cache_only_pct']:.1f} "
            f"| BOTTOM n={r['bottom']['n']} mean={r['bottom']['mean_R']:.4f} t={r['bottom']['t']:.3f} "
            f"| DROPPED n={r['dropped']['n']} mean={r['dropped']['mean_R']:.4f} "
            f"| kept-dropped={r['kept_minus_dropped']:.4f}")

    log('=== B60 (beside) ===')
    for r in rows_b60:
        log(f"{r['split']}: TOP n={r['top']['n']} mean={r['top']['mean_R']:.4f} t={r['top']['t']:.3f} "
            f"ex5={r['top']['ex_top5']:.4f} fills/wk={r['top_fills_wk']:.2f}")

    out_cols = ['day', 'symbol', 'split', 'fill_min', 'arm_m', 'exit_m', 'hour', 'b30', 'b60',
                'tercile_b30', 'tercile_b60', 'kept_top_b30', 'kept_top_b60', 'outcome_R',
                'store_served_1438']
    fills_path = f'{OUT_DIR}/rebuild_1624_fills.csv'
    df[out_cols].to_csv(fills_path, index=False)
    log(f'Wrote {fills_path} ({len(df)} rows)')

    val_top_b30 = [r for r in rows_b30 if r['split'] == 'VAL'][0]['top']
    log(f"HEADLINE -- top-tercile (B30) kept VAL: mean_R={val_top_b30['mean_R']:.4f} "
        f"t={val_top_b30['t']:.3f} n={val_top_b30['n']}")

    return rows_b30, rows_b60, hour_means, weeks_by_split, cutoffs_b30, cutoffs_b60


if __name__ == '__main__':
    main()
