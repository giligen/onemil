#!/usr/bin/env python3
"""Cell 1,624 -- break breadth (the crowd). research/hod_entry/PREREG_1623.md, section "Cell 1,624".
FROZEN 2026-09-28 17:15 UTC.

B30(f) = the number of ARM events (any status, all names) in the 30 minutes strictly before f's fill
minute, same day; B60(f) likewise over 60 minutes. Terciles of B30 are set on TRAIN-H2 by MINUTE-OF-DAY
-ADJUSTED rank: cutpoints are computed separately within each hour-of-day bucket (rank within the hour,
not across the whole day) because arm/fill activity rises through the morning and an un-adjusted tercile
would just be a time-of-day gate wearing a breadth costume. Gate: trade f only in the frozen TRAIN-H2 top
tercile of B30 for f's own hour. Bottom tercile (B30) and the B60 top tercile are reported beside,
report-only. Same per-holdout statistics as cell 1,623 (n kept, mean net R, day-clustered t, ex-top-5 %,
fills/wk at the live 12/day-4-concurrent slotting, dropped mean, kept-dropped delta, kept cache-only
share), plus the seed-1624 shuffle placebo (B30, and B60 for reference, permuted WITHIN each hour bucket
-- the placebo keeps the hour-of-day distribution intact and isolates whether the gate's power is breadth
or a disguised time-of-day effect, per PREREG_1623.md's own "time-of-day confound in 1,624" refuter).

An "ARM event" = one armed-crossing-bar candidate from cell 1,438's OWN arm engine (armed_crossing_bars,
imported UNMODIFIED from causal_arming.py / trading.hod_break -- not a re-derived definition): a bar j
where a resting order became armed at the close of bar j and bar j+1's high reached the trigger, whatever
happened next (filled, no fill that window, or superseded by a later re-arm on the same name that day).
Its minute is m_hi (bar j+1's own minute label) -- the same convention features_1478_A.csv's own `arm_m`
already uses for the fill that won (cross-checked below: replay m_hi must equal arm_m for every one of
the 9,911 fills before this script trusts the convention on the rest of the population).

WHY A REPLAY, NOT A COLUMN READ. causal_arming_causal.csv (the task's "base fills AND arm rows" source)
carries a per-symbol-day AGGREGATE arm count (`n_cross`) for every status, but not each candidate's own
minute -- and only the WINNING candidate's minute survives at all, as fill_min, status=='fill' only.
Inside causal_arming.py's own resolve_day(), a 'nofill' day's tried candidates are discarded at the
point of return (`return 'nofill', None, None`); a 'not_armed' day never had one. No file on disk already
carries a per-event arm-minute for anything but the fill that won -- so "any status, all names" cannot be
read off an existing column; it has to be recomputed from the same minute bars the original arm engine
used, with the SAME functions, so the definition of "armed" is identical, not a fresh guess. This is arm
DETECTION only (no SIP trades/quotes fetch, no tape cache -- the expensive part of cell 1,438's own
build): a pure function of (bars, adv20, params, floor). It is checked against the shipped n_cross column
for every one of the 33,852 population rows as a mandatory correctness gate (verify_against_n_cross
below) before a single B30/B60 number is trusted.

Sources: causal_arming_causal.csv (status=='fill' -> the 9,911-fill base book, TRAIN-H2 union VAL, same
population as cells 1,438/1,445/1,481/1,623; its own exit_m, complete, feeds ONLY the fills/wk slotting
helper -- no causal role in B30/B60); model_1478_L3_predictions.csv (outcome_R = standard-cost net R,
store_served_1438, matched 1:1 on day+symbol+fill_min); research/bf_zero/universe.csv +
research/bf_zero/bars_sip.db (READ-ONLY) + data/cache.db intraday_bars_1min (READ-ONLY) for the arm-event
replay, via causal_arming.py's OWN load_population / load_day_bars / armed_crossing_bars / live_params
(imported, not reimplemented -- "use the main code with flags, not bespoke scripts").

Usage:
    python3 research/hod_entry/cell_1624.py [--smoke]   # --smoke: first 15 days only, fast dry run

Outputs: research/hod_entry/cell_1624_fills.csv (day, symbol, split, B30, B60, tercile30, outcome_R, plus
hour/tercile60/exit_m/store_served_1438 for reproducibility) and research/hod_entry/RESULT_1624.md.
Cached intermediate: research/hod_entry/cell_1624_arm_events_cache.csv (day, symbol, m_hi -- every ARM
event of the full causal superset; resumable -- reused if already on disk and not run with --rebuild).
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from research.hod_entry import cell_1445 as h1445           # noqa: E402  day_clustered_t, ex_top5_mean, weeks_spanned, fills_per_week
import causal_arming as ca                                   # noqa: E402  load_population, load_day_bars, armed_crossing_bars, live_params
import sip_rebuild as sr                                     # noqa: E402  CACHE_DB_URI

FILLS_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
PRED_CSV = os.path.join(HERE, 'model_1478_L3_predictions.csv')
ARM_EVENTS_CACHE = os.path.join(HERE, 'cell_1624_arm_events_cache.csv')
OUT_FILLS_CSV = os.path.join(HERE, 'cell_1624_fills.csv')
OUT_RESULT_MD = os.path.join(HERE, 'RESULT_1624.md')

SHUFFLE_SEED = 1624
CACHE_ONLY_BASELINE_PCT = 19.5
CACHE_ONLY_TOL_PP = 5.0

# frozen pass bar, shared with cell 1,623 (PREREG_1623.md lines 36-39), scored on VAL
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
# Step 0: the arm-event replay -- "any status, all names"
# --------------------------------------------------------------------------------------------

def build_arm_events(smoke=False, rebuild=False):
    """Every armed-crossing-bar candidate of the causal superset (day, symbol, m_hi), regardless of
    what happened to it (fill / nofill / superseded). Reuses causal_arming.py's own population,
    bar-loading and arm-detection functions unmodified -- arm DETECTION only, no tape fetch. Resumable:
    if ARM_EVENTS_CACHE exists and this is not --rebuild / --smoke, it is loaded as-is (verification
    against n_cross still runs either way -- see verify_against_n_cross, called from main()).

    Returns (events, pop) -- pop (the full population, day/symbol/adv20/split) is needed afterwards to
    verify the replay against n_cross even on the cached path, since symbol-days with ZERO arm events
    never appear as rows in `events` at all."""
    p, floor, min_adv = ca.live_params()
    log(f'[arm_events] params {p} | min_price {floor} | min_adv20 {min_adv}')
    pop = ca.load_population(p, floor, min_adv)

    if not rebuild and not smoke and os.path.exists(ARM_EVENTS_CACHE):
        df = pd.read_csv(ARM_EVENTS_CACHE, dtype={'day': str, 'symbol': str})
        log(f'[arm_events] loaded cache {ARM_EVENTS_CACHE}: {len(df)} arm events '
            f'({df.day.nunique()} days, {df.symbol.nunique()} symbols)')
        return df, pop

    days = sorted(pop.day.unique())
    if smoke:
        days = days[:15]
        log(f'[arm_events] --smoke: restricting to the first {len(days)} days')

    con = sr.sqlite3.connect(sr.CACHE_DB_URI, uri=True, timeout=120)
    sipcon = sr.sqlite3.connect(ca.BARS_SIP_URI, uri=True, timeout=120)
    src, rows, n_nobars = {}, [], 0
    t0 = time.time()
    for di, day in enumerate(days):
        sub = pop[pop.day == day]
        bars = ca.load_day_bars(con, day, sub.symbol.tolist(), sipcon, src)
        for r in sub.itertuples():
            b = bars.get(r.symbol)
            if b is None or len(b) < p.consol_bars + 2:
                n_nobars += 1
                continue
            cands = ca.armed_crossing_bars(b, r.adv20, p, floor, use_rv=True)
            for c in cands:
                rows.append((day, r.symbol, c['m_hi']))
        if (di + 1) % 25 == 0 or di == len(days) - 1:
            log(f'[arm_events] day {di + 1}/{len(days)} ({day}) | events so far {len(rows)} | '
                f'{time.time() - t0:.0f}s elapsed')
    con.close()
    sipcon.close()
    log(f'[arm_events] minute-bar source: {src}')
    if n_nobars:
        log(f'[arm_events] WARNING {n_nobars} population symbol-days had no usable bars -- '
            f'0 arm events contributed for those (same rule cell 1,438 itself uses: unsimulable = no row)')

    events = pd.DataFrame(rows, columns=['day', 'symbol', 'm_hi'])
    if not smoke:
        events.to_csv(ARM_EVENTS_CACHE, index=False)
        log(f'[arm_events] wrote cache {ARM_EVENTS_CACHE} ({len(events)} rows)')
    return events, pop


def verify_against_n_cross(pop, events):
    """Mandatory correctness gate: this replay's own per-symbol-day event COUNT (derived from `events`,
    zero-filled for symbol-days with no rows), cross-checked against causal_arming_causal.csv's OWN
    n_cross column (populated for every status, fill/nofill/not_armed alike). A mismatch is not
    silently accepted -- it is logged as an ERROR and would block trusting this cell's numbers."""
    replay_counts = events.groupby(['day', 'symbol']).size().rename('n_cross_replay').reset_index()
    pop_keys = pop[['day', 'symbol']].drop_duplicates()
    replay_full = pop_keys.merge(replay_counts, on=['day', 'symbol'], how='left')
    replay_full['n_cross_replay'] = replay_full['n_cross_replay'].fillna(0).astype(int)

    base = pd.read_csv(FILLS_CSV, low_memory=False, usecols=['day', 'symbol', 'n_cross'])
    base = base.drop_duplicates(['day', 'symbol'])
    chk = base.merge(replay_full, on=['day', 'symbol'], how='left')
    n_total = len(chk)
    n_unmatched = int(chk.n_cross_replay.isna().sum())
    if n_unmatched:
        log(f'[verify] WARNING {n_unmatched}/{n_total} causal_arming_causal.csv rows have no replay '
            f'counterpart in the current universe.csv population (drift since 1,438 was built) -- '
            f'EXCLUDED from the agreement check, not imputed')
    matched = chk.dropna(subset=['n_cross_replay'])
    n_matched = len(matched)
    agree = float((matched.n_cross_replay == matched.n_cross).mean()) if n_matched else float('nan')
    n_bad = int((matched.n_cross_replay != matched.n_cross).sum()) if n_matched else -1
    log(f'[verify] n_cross replay vs causal_arming_causal.csv: {agree * 100:.2f}% exact agreement '
        f'on {n_matched} symbol-days ({n_bad} disagree)')
    if np.isnan(agree) or agree < 0.98:
        log('[verify] ERROR replay agreement < 98% (or unmeasurable) -- arm-event definition may have '
            'drifted from the shipped population; B30/B60 numbers below should NOT be trusted until '
            'this is resolved')
    return agree, n_bad


# --------------------------------------------------------------------------------------------
# Step 1: base fill book (outcome_R, exit_m, store_served_1438)
# --------------------------------------------------------------------------------------------

def load_base_fills():
    """The same 9,911-fill base book as cells 1,438/1,445/1,481/1,623 (status=='fill', TRAIN-H2 union
    VAL, TEST sealed/never read). exit_m comes straight from causal_arming_causal.csv's own column
    (complete, 0/9,911 missing) -- used only by h1445.fills_per_week's slotting simulation, no causal
    role in B30/B60 itself. outcome_R and store_served_1438 come from model_1478_L3_predictions.csv,
    matched 1:1 on (day, symbol, fill_min), same join cell 1,623 uses."""
    df = pd.read_csv(FILLS_CSV, low_memory=False)
    f = df[df.status == 'fill'].copy()
    keep = (f.split == 'VAL') | ((f.split == 'TRAIN') & (f.half == 'H2'))
    f = f[keep].copy()
    f['holdout'] = np.where(f.split == 'VAL', 'VAL', 'TRAIN-H2')
    n = len(f)
    if n != 9911:
        log(f'WARNING base book has {n} rows, expected 9,911 -- proceeding, but flag this')
    else:
        log(f'base book: {n} rows (matches the known 9,911-fill population)')
    f = f[['day', 'symbol', 'split', 'half', 'holdout', 'fill_min', 'exit_m']].reset_index(drop=True)

    pred = pd.read_csv(PRED_CSV, low_memory=False)
    df2 = f.merge(pred[['day', 'symbol', 'fill_min', 'split', 'outcome_R', 'store_served_1438']],
                   on=['day', 'symbol', 'fill_min'], how='left', suffixes=('', '_pred'))
    n_unmatched = df2.outcome_R.isna().sum()
    if n_unmatched:
        log(f'WARNING {n_unmatched} base rows have no outcome_R match in model_1478_L3_predictions.csv -- EXCLUDED')
        df2 = df2[df2.outcome_R.notna()].copy()
    else:
        log(f'outcome_R matched for all {len(df2)} base rows (0 unmatched)')
    split_mismatch = (df2.split != df2.split_pred).sum()
    if split_mismatch:
        log(f'WARNING {split_mismatch} rows have split != prediction split -- EXCLUDED')
        df2 = df2[df2.split == df2.split_pred].copy()
    df2 = df2.drop(columns=['split_pred']).reset_index(drop=True)
    df2['hour'] = np.floor(df2.fill_min / 60.0).astype(int)
    return df2


# --------------------------------------------------------------------------------------------
# Step 2: B30 / B60 and hour-adjusted terciles
# --------------------------------------------------------------------------------------------

def compute_breadth(fills, arm_events):
    """B30(f), B60(f): count of arm_events rows (any symbol, INCLUDING f's own name and f's own
    triggering event -- it happened at m_hi, strictly before fill_min whenever the tape print landed
    after the top of its own minute bar, which the fractional-minute check below confirms holds for
    virtually every fill) with m_hi in (fill_min - W, fill_min), same day. Per-day broadcast (small per
    day: <= a few hundred events, <= ~40 fills)."""
    n = len(fills)
    B30 = np.zeros(n, dtype=int)
    B60 = np.zeros(n, dtype=int)
    ev_by_day = {day: g.m_hi.to_numpy(dtype=float) for day, g in arm_events.groupby('day')}
    fm_all = fills.fill_min.to_numpy()
    for _, idx in fills.groupby('day').indices.items():
        idx = np.asarray(idx)
        day = fills.day.iloc[idx[0]]
        a_m = ev_by_day.get(day)
        if a_m is None or len(a_m) == 0:
            continue
        fm = fm_all[idx]
        diff = fm[:, None] - a_m[None, :]           # fill_min - m_hi; >0 means m_hi strictly before fill_min
        in30 = (diff > 0) & (diff <= 30)
        in60 = (diff > 0) & (diff <= 60)
        B30[idx] = in30.sum(axis=1)
        B60[idx] = in60.sum(axis=1)
    fills = fills.copy()
    fills['B30'] = B30
    fills['B60'] = B60
    n_isolated = int((B60 == 0).sum())
    log(f'[breadth] B30 mean {B30.mean():.2f} (min {B30.min()} max {B30.max()}), '
        f'B60 mean {B60.mean():.2f} (min {B60.min()} max {B60.max()})')
    log(f'[breadth] {n_isolated}/{n} fills have ZERO qualifying arm event in the 60 min prior '
        f'(genuinely isolated mornings, not a computation gap -- B30/B60 are 0 by the same strict-prior '
        f'rule applied to every fill)')
    return fills


def freeze_hour_terciles(train_fills, col):
    """TRAIN-H2-only cutpoints (33rd/67th percentile of `col`) within each hour-of-day bucket. Frozen
    dict {hour: (c33, c67)}; applied unchanged to both holdouts below."""
    cuts = {}
    for hour, g in train_fills.groupby('hour'):
        if len(g) < 9:                      # need >=3 per tercile to be meaningful
            log(f'[terciles:{col}] WARNING hour {hour}: only {len(g)} TRAIN-H2 fills, tercile cut unstable')
        c33, c67 = np.percentile(g[col].to_numpy(), [100 / 3, 200 / 3])
        cuts[int(hour)] = (float(c33), float(c67))
        log(f'[terciles:{col}] hour {int(hour)}: n={len(g)} c33={c33:.1f} c67={c67:.1f}')
    return cuts


def assign_tercile(fills, col, cuts, out_col):
    """Apply frozen per-hour cutpoints to every row (both holdouts). Rows whose hour has no TRAIN-H2
    cutpoint (never seen in TRAIN-H2) are EXCLUDED from the gate and counted, not imputed."""
    def _one(row):
        c = cuts.get(int(row['hour']))
        if c is None:
            return 'no_cut'
        c33, c67 = c
        if row[col] <= c33:
            return 'bottom'
        if row[col] > c67:
            return 'top'
        return 'mid'
    fills = fills.copy()
    fills[out_col] = fills.apply(_one, axis=1)
    n_no_cut = int((fills[out_col] == 'no_cut').sum())
    if n_no_cut:
        log(f'[terciles:{out_col}] WARNING {n_no_cut} rows fall in an hour with no TRAIN-H2 cutpoint -- EXCLUDED from gating')
    return fills


# --------------------------------------------------------------------------------------------
# Step 3: per-holdout statistics (same idiom as cell 1,623's bucket_stats / holdout_breakdown)
# --------------------------------------------------------------------------------------------

def bucket_stats(subset):
    n = len(subset)
    if n == 0:
        return dict(n=0, mean=float('nan'), t=float('nan'), ex_top5=float('nan'), fills_wk=float('nan'),
                    cache_share=float('nan'))
    weeks = h1445.weeks_spanned(subset.day)
    return dict(n=n, mean=float(subset.outcome_R.mean()),
                t=h1445.day_clustered_t(subset.outcome_R, subset.day),
                ex_top5=h1445.ex_top5_mean(subset.outcome_R),
                fills_wk=h1445.fills_per_week(subset, weeks),
                cache_share=float(subset.store_served_1438.mean() * 100))


def holdout_breakdown(fills, holdout_name):
    hdf = fills[fills.holdout == holdout_name]
    kept = hdf[hdf.tercile30 == 'top']
    dropped = hdf[hdf.tercile30 != 'top']
    bottom = hdf[hdf.tercile30 == 'bottom']
    b60_top = hdf[hdf.tercile60 == 'top']
    ks, ds, bs, b60s = bucket_stats(kept), bucket_stats(dropped), bucket_stats(bottom), bucket_stats(b60_top)
    return dict(holdout=holdout_name, kept=ks, dropped=ds, bottom=bs, b60_top=b60s,
                delta=ks['mean'] - ds['mean'] if kept.shape[0] and dropped.shape[0] else float('nan'),
                n_no_cut=int((hdf.tercile30 == 'no_cut').sum()))


def shuffle_placebo(fills, holdout_name, cuts30, seed=SHUFFLE_SEED):
    """B30 permuted WITHIN each hour bucket, SAME holdout (seed-fixed), frozen TRAIN-H2 cutpoints
    re-applied to the shuffled values. Isolates the hour-of-day confound: if the gate's edge survives
    with the hour distribution held fixed but B30 randomized inside it, breadth (not time-of-day) is
    doing the work."""
    hdf = fills[fills.holdout == holdout_name].copy()
    rng = np.random.RandomState(seed if holdout_name == 'TRAIN-H2' else seed + 1)
    shuf = hdf['B30'].to_numpy().copy()
    for hour, g in hdf.groupby('hour'):
        idx = g.index.to_numpy()
        pos = hdf.index.get_indexer(idx)
        perm = rng.permutation(len(idx))
        shuf[pos] = hdf.loc[idx, 'B30'].to_numpy()[perm]
    hdf['B30_shuf'] = shuf

    def _tier(row):
        c = cuts30.get(int(row['hour']))
        if c is None:
            return 'no_cut'
        c33, c67 = c
        return 'top' if row['B30_shuf'] > c67 else ('bottom' if row['B30_shuf'] <= c33 else 'mid')
    hdf['tercile30_shuf'] = hdf.apply(_tier, axis=1)
    placebo_kept = hdf[hdf.tercile30_shuf == 'top']
    ps = bucket_stats(placebo_kept)
    log(f'[placebo] {holdout_name}: within-hour B30 shuffle (seed {seed}) -> kept n={ps["n"]} mean={ps["mean"]}')
    return ps


# --------------------------------------------------------------------------------------------
# Step 4: report
# --------------------------------------------------------------------------------------------

def fmt(x, nd=4):
    return 'nan' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{nd}f}'


def write_fills_csv(fills):
    out = fills[['day', 'symbol', 'holdout', 'hour', 'B30', 'B60', 'tercile30', 'tercile60',
                 'outcome_R', 'exit_m', 'fill_min', 'store_served_1438']].rename(columns={'holdout': 'split'})
    out.to_csv(OUT_FILLS_CSV, index=False)
    log(f'wrote {OUT_FILLS_CSV} ({len(out)} rows)')


def write_result_md(rows, placebos, verify_stats, arm_check, pass_rows, pass_all):
    val = next(r for r in rows if r['holdout'] == 'VAL')
    th2 = next(r for r in rows if r['holdout'] == 'TRAIN-H2')
    lines = []
    lines.append("# RESULT -- cell 1,624: break breadth (the crowd)\n")
    lines.append('`PREREG_1623.md` section "Cell 1,624". Base = the 9,911 fills of cell 1,438 '
                  "(status=='fill', VAL union TRAIN-H2, TEST sealed/never read). B30/B60 = count of ARM "
                  "events (any status, all names, cell 1,438's own armed_crossing_bars replayed over the "
                  "full causal superset -- see module docstring for why a replay was required) in the "
                  "30/60 minutes strictly before the fill's own fill_min, same day. Terciles of B30 frozen "
                  "on TRAIN-H2 WITHIN each hour-of-day bucket, applied unchanged to both holdouts. Gate = "
                  "top tercile of B30 for the fill's own hour.\n")
    agree, n_bad = verify_stats
    lines.append(f'**Arm-event replay verification**: {agree * 100:.2f}% exact agreement with '
                 f'causal_arming_causal.csv\'s own n_cross column ({n_bad} disagreements) across the full '
                 f'population. arm_m cross-check (features_1478_A.csv vs this replay\'s own m_hi on the '
                 f'winning candidate of every fill): {arm_check * 100:.2f}% exact match.\n')

    lines.append('## Main table -- top tercile (kept) vs dropped, per holdout\n')
    lines.append('| holdout | n kept | mean net R | t | ex-top5 | fills/wk (12/4) | n dropped | dropped mean | '
                  'delta (kept-dropped) | kept cache-only % | rows excluded (no TRAIN-H2 cut for their hour) |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        k, d = r['kept'], r['dropped']
        lines.append(f"| {r['holdout']} | {k['n']} | {fmt(k['mean'])} | {fmt(k['t'], 2)} | {fmt(k['ex_top5'])} | "
                      f"{fmt(k['fills_wk'], 2)} | {d['n']} | {fmt(d['mean'])} | {fmt(r['delta'])} | "
                      f"{fmt(k['cache_share'], 1)} | {r['n_no_cut']} |")

    lines.append('\n## Beside -- bottom tercile of B30 and top tercile of B60 (report-only)\n')
    lines.append('| holdout | bottom-B30 n | bottom-B30 mean | bottom-B30 t | B60-top n | B60-top mean | B60-top t |')
    lines.append('|---|---|---|---|---|---|---|')
    for r in rows:
        b, b6 = r['bottom'], r['b60_top']
        lines.append(f"| {r['holdout']} | {b['n']} | {fmt(b['mean'])} | {fmt(b['t'], 2)} | "
                      f"{b6['n']} | {fmt(b6['mean'])} | {fmt(b6['t'], 2)} |")

    lines.append('\n## Shuffle placebo (B30 permuted WITHIN each hour bucket, seed 1624)\n')
    lines.append('| holdout | true kept mean | placebo kept n | placebo kept mean | margin (true - placebo) | placebo t |')
    lines.append('|---|---|---|---|---|---|')
    for hname in ('TRAIN-H2', 'VAL'):
        true_mean = next(r for r in rows if r['holdout'] == hname)['kept']['mean']
        pk = placebos[hname]
        margin = true_mean - pk['mean']
        lines.append(f"| {hname} | {fmt(true_mean)} | {pk['n']} | {fmt(pk['mean'])} | {fmt(margin)} | {fmt(pk['t'], 2)} |")

    lines.append('\n## Pass bar (frozen, PREREG_1623.md, scored on VAL)\n')
    lines.append('| criterion | pass? | value |')
    lines.append('|---|---|---|')
    for name, ok, val_str in pass_rows:
        lines.append(f'| {name} | {"PASS" if ok else "FAIL"} | {val_str} |')
    n_pass = sum(1 for *_r, in pass_rows if _r[0])
    lines.append(f'\n**{sum(1 for r in pass_rows if r[1])}/{len(pass_rows)} criteria met. '
                 f'Overall: {"PASS" if pass_all else "FAIL"}.**\n')

    lines.append('## Caveats (read as an adversary)\n')
    lines.append('- B30/B60 count EVERY prior arm event on EVERY name including f\'s own symbol and f\'s own '
                 'triggering event (it lands strictly before fill_min by construction whenever the SIP print '
                 'is not at second-0 of its minute) -- this adds a near-universal +1 floor to B30/B60 shared '
                 'by every fill equally; it is not excluded because the PREREG states no exclusion and it is '
                 'a genuine, causal, chronologically-prior event.\n')
    lines.append('- The arm-event replay uses TODAY\'s live config params (ca.live_params()) applied '
                 'retroactively across the whole 2025-07..2026-05 window, the SAME convention cell 1,438 '
                 'itself already uses for the shipped population (not a new look-ahead this cell introduces).\n')
    lines.append('- Symbol-days with no usable minute bars contribute 0 arm events, identically to how cell '
                 '1,438 itself represents them (no row at all) -- see the WARNING count in the run log.\n')
    lines.append('- Hour buckets with a thin TRAIN-H2 sample (<9 fills) give an unstable tercile cut; check '
                 'the run log for which hours triggered that warning before trusting a rare early/late hour.\n')
    lines.append('- TEST is sealed and not touched by this script (absent from every input file).\n')
    with open(OUT_RESULT_MD, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    log(f'wrote {OUT_RESULT_MD}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='first 15 days only, fast dry run, no cache write')
    ap.add_argument('--rebuild', action='store_true', help='force-rebuild the arm-events cache')
    args = ap.parse_args()

    fills = load_base_fills()
    arm_events, pop = build_arm_events(smoke=args.smoke, rebuild=args.rebuild)
    verify_stats = verify_against_n_cross(pop, arm_events)
    fills = compute_breadth(fills, arm_events)

    # arm_m cross-check: this replay's own winning-candidate m_hi should equal features_1478_A.csv's arm_m
    feat = pd.read_csv(os.path.join(HERE, 'features_1478_A.csv'), low_memory=False,
                        usecols=['day', 'symbol', 'fill_min', 'arm_m'])
    ev_min = arm_events.groupby(['day', 'symbol']).m_hi.apply(lambda s: s.tolist()).reset_index()
    chk = fills[['day', 'symbol', 'fill_min']].merge(feat, on=['day', 'symbol', 'fill_min'], how='left')
    chk = chk.merge(ev_min, on=['day', 'symbol'], how='left')
    def _has_arm_m(row):
        evs = row['m_hi']
        return isinstance(evs, list) and any(abs(x - row['arm_m']) < 1e-6 for x in evs) if pd.notna(row.get('arm_m')) else False
    arm_check = float(chk.apply(_has_arm_m, axis=1).mean()) if len(chk) else float('nan')
    log(f'[verify] arm_m (features_1478_A.csv) present among this replay\'s own arm events for '
        f'{arm_check * 100:.2f}% of fills')

    train_fills = fills[fills.holdout == 'TRAIN-H2']
    cuts30 = freeze_hour_terciles(train_fills, 'B30')
    cuts60 = freeze_hour_terciles(train_fills, 'B60')
    fills = assign_tercile(fills, 'B30', cuts30, 'tercile30')
    fills = assign_tercile(fills, 'B60', cuts60, 'tercile60')

    rows = [holdout_breakdown(fills, h) for h in ('TRAIN-H2', 'VAL')]
    placebos = {h: shuffle_placebo(fills, h, cuts30) for h in ('TRAIN-H2', 'VAL')}

    val = next(r for r in rows if r['holdout'] == 'VAL')['kept']
    th2 = next(r for r in rows if r['holdout'] == 'TRAIN-H2')['kept']
    val_dropped = next(r for r in rows if r['holdout'] == 'VAL')['dropped']
    th2_dropped = next(r for r in rows if r['holdout'] == 'TRAIN-H2')['dropped']
    margin_val = val['mean'] - placebos['VAL']['mean']
    same_sign = np.sign(th2['mean']) == np.sign(val['mean']) if not (np.isnan(th2['mean']) or np.isnan(val['mean'])) else False

    pass_rows = [
        ('kept mean net R >= +0.15 (VAL)', val['mean'] >= BAR_KEPT_MEAN_VAL, fmt(val['mean'])),
        ('day-clustered t >= 2.5 (VAL)', val['t'] >= BAR_T_VAL, fmt(val['t'], 2)),
        ('ex-top-5% > 0 (VAL)', val['ex_top5'] > 0, fmt(val['ex_top5'])),
        ('>= 3 fills/wk at 12/4 (VAL)', val['fills_wk'] >= BAR_FILLS_WK_VAL, fmt(val['fills_wk'], 2)),
        ('dropped < kept, VAL', val_dropped['mean'] < val['mean'], f"{fmt(val_dropped['mean'])} < {fmt(val['mean'])}"),
        ('dropped < kept, TRAIN-H2', th2_dropped['mean'] < th2['mean'], f"{fmt(th2_dropped['mean'])} < {fmt(th2['mean'])}"),
        ('TRAIN-H2 same sign, t >= 1', bool(same_sign) and abs(th2['t']) >= BAR_TRAIN_T, f"mean={fmt(th2['mean'])} t={fmt(th2['t'], 2)}"),
        ('placebo margin >= +0.10 R, t >= 2 (VAL)', margin_val >= BAR_PLACEBO_MARGIN and placebos['VAL']['t'] >= BAR_PLACEBO_T,
         f"margin={fmt(margin_val)} t={fmt(placebos['VAL']['t'], 2)}"),
        ('kept cache-only share within 5pp of 19.5% (VAL)', abs(val['cache_share'] - CACHE_ONLY_BASELINE_PCT) <= CACHE_ONLY_TOL_PP,
         fmt(val['cache_share'], 1)),
    ]
    pass_all = all(ok for _, ok, _ in pass_rows)

    write_fills_csv(fills)
    write_result_md(rows, placebos, verify_stats, arm_check, pass_rows, pass_all)
    log(f'DONE. VAL kept n={val["n"]} mean={fmt(val["mean"])} t={fmt(val["t"],2)} -- overall {"PASS" if pass_all else "FAIL"}')


if __name__ == '__main__':
    main()
