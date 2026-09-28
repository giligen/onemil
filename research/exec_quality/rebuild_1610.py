#!/usr/bin/env python3
"""Independent rebuild of cells 1,610-1,616 (research/exec_quality/PREREG_1610.md, FROZEN
2026-09-28) from the PROSE SPEC ONLY. This module was written WITHOUT opening cell_1610.py,
cell_1610_*.csv or RESULT_1610.md (forbidden by the delegating task) -- every mechanism below is
derived from PREREG_1610.md plus the explicitly-cited helper modules (cell_1478.py for
SLIP_STOP_BPS, cell_1445.py for day_clustered_t/ex_top5_mean/weeks_spanned, sip_rebuild.py for
walk_path's EOD-first / stop-first / gap-through-at-the-open semantics).

Part A -- the drift map (report-only): for every fill and for each TRAIN-H2-edged spread quintile,
the empirical first-passage P(+k before -m) for the 6x6=36 (k,m) % grid, beside the driftless
value, a Wilson 95% CI, mean signed excursion at 5/15/30/60/120 min, and realised entry/exit cost.

Part B -- six barrier cells (1,610-1,615) plus the report-only quintile-optimal cell (1,616),
walked on minute bars with sip_rebuild.walk_path's exact rule ordering (EOD-bar check first, then
stop with gap-through-at-the-open, then target), paired against outcome_R (model_1478_L3_predictions.csv).

Outputs: rebuild_1610_driftmap.csv (Part A grid), rebuild_1610_fills.csv (per-fill Part B detail),
REBUILD_1610.md (both parts, pass-bar check, and every documented assumption).

Usage: python3 research/exec_quality/rebuild_1610.py
"""
import math
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = '/home/ec2-user/onemil/research/exec_quality'
HOD = '/home/ec2-user/onemil/research/hod_entry'
sys.path.insert(0, HOD)
import cell_1445 as c1445  # day_clustered_t, ex_top5_mean, weeks_spanned -- sanctioned helpers

ET = ZoneInfo('America/New_York')

# from sip_rebuild.py: OPEN_M, EOD_M = 570, 955 (RTH open / the 15:55 ET cutoff bar)
OPEN_M = 570
EOD_M = 955
RTH_HI = 960  # exclusive upper bound used only to scope the bar store to RTH

# from cell_1478.py: SLIP_STOP_BPS = {'TRAIN': 0.88*2.9 + 0.12*94.0, 'VAL': 0.88*3.2 + 0.12*76.0}
SLIP_STOP_BPS = {'TRAIN': 0.88 * 2.9 + 0.12 * 94.0, 'VAL': 0.88 * 3.2 + 0.12 * 76.0}
# PREREG_1610.md Inputs/Data: "EOD at the bid {TRAIN: 11.5, VAL: 9.7} bps"
EOD_BID_BPS = {'TRAIN': 11.5, 'VAL': 9.7}

K_M_GRID = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]  # % of price -- 6x6 = 36 pairs (Part A)
HORIZONS_MIN = [5, 15, 30, 60, 120]
QUINTILES = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ================================================================================================
# Data loading
# ================================================================================================

def load_fills():
    """The 9,911 base fills (causal_arming_causal.csv status=='fill') joined to half_entry
    (features_1478_A), spread_bps_at_arm (features_1478_C) and outcome_R (model_1478_L3_predictions),
    all keyed on (day, symbol) -- verified 1:1 and fill_min-consistent across every source file
    before this script was written (one arm/fill per symbol per day in this population)."""
    base = pd.read_csv(f'{HOD}/causal_arming_causal.csv', low_memory=False)
    fills = base[base['status'] == 'fill'].copy().reset_index(drop=True)
    assert len(fills) == 9911, f'expected 9,911 fills, got {len(fills)}'
    # PREREG: "split (TRAIN = TRAIN-H2, VAL)" -- verify the file's own half column agrees.
    assert (fills.loc[fills.split == 'TRAIN', 'half'] == 'H2').all()
    assert fills.loc[fills.split == 'VAL', 'half'].isna().all()

    feat_a = pd.read_csv(f'{HOD}/features_1478_A.csv')[['day', 'symbol', 'half_entry']]
    feat_c = pd.read_csv(f'{HOD}/features_1478_C.csv')[['day', 'symbol', 'spread_bps_at_arm']]
    pred = pd.read_csv(f'{HOD}/model_1478_L3_predictions.csv')[['day', 'symbol', 'outcome_R']]

    n0 = len(fills)
    fills = fills.merge(feat_a, on=['day', 'symbol'], how='left')
    fills = fills.merge(feat_c, on=['day', 'symbol'], how='left')
    fills = fills.merge(pred, on=['day', 'symbol'], how='left')
    assert len(fills) == n0, 'join fanned out -- (day,symbol) not unique'
    for col in ('half_entry', 'spread_bps_at_arm', 'outcome_R'):
        n_missing = fills[col].isna().sum()
        if n_missing:
            log(f'WARNING: {n_missing} fills missing {col} after join')

    fills['R_base'] = fills['fill'] - fills['stop']
    assert (fills['R_base'] > 0).all(), 'non-positive base R'
    fills['R_base_pct'] = fills['R_base'] / fills['fill'] * 100.0
    fills['fill_bar_m'] = np.floor(fills['fill_min']).astype(int)
    # cross-check: the source file's own R column should equal fill-stop (independent agreement)
    max_diff = (fills['R_base'] - fills['R']).abs().max()
    log(f'load_fills: {len(fills)} fills (TRAIN-H2={sum(fills.split=="TRAIN")}, '
        f'VAL={sum(fills.split=="VAL")}); R_base vs source R max|diff|={max_diff:.2e} '
        f'(cross-check, should be ~0)')

    fills['c_in'] = fills['half_entry'] + (fills['fill'] - fills['level'])
    bps = fills['split'].map(SLIP_STOP_BPS).astype(float)
    fills['c_out'] = fills['stop'] * bps / 1e4  # "the stop-limit standard in price", per fill
    log(f'  c_in mean={fills.c_in.mean():.4f} $/sh, c_out mean={fills.c_out.mean():.4f} $/sh')
    return fills


def assign_quintiles(fills):
    """Spread quintile edges fit on TRAIN-H2's causal arm-time spread ONLY, then applied to every
    fill in both holdouts (never re-fit on VAL) -- PREREG Part A and the 1,616 refuter list both
    require this."""
    train_spread = fills.loc[fills.split == 'TRAIN', 'spread_bps_at_arm']
    edges = train_spread.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
    bins = [-np.inf] + list(edges) + [np.inf]
    fills = fills.copy()
    fills['quintile'] = pd.cut(fills['spread_bps_at_arm'], bins=bins, labels=QUINTILES,
                                include_lowest=True)
    log(f'assign_quintiles: TRAIN-H2 edges (bps) = {[round(e, 2) for e in edges]}')
    log('  counts:\n' + fills.groupby(['split', 'quintile'], observed=True).size().unstack().to_string())
    return fills, edges


def et_minute_of_day(ts_series):
    """Vectorized ISO8601 UTC -> ET fractional minute-of-day (DST-aware via zoneinfo)."""
    dt_utc = pd.to_datetime(ts_series, utc=True)
    dt_et = dt_utc.dt.tz_convert(ET)
    return dt_et.dt.hour * 60 + dt_et.dt.minute + dt_et.dt.second / 60.0


def load_bars_store(need_pairs):
    """Load bars_fills_1478.db bars(symbol,day,t,o,h,l,c,v) for exactly the (symbol,day) pairs in
    `need_pairs`, convert t to ET minute-of-day, restrict to RTH [570,960), and group into a dict
    {(symbol,day): (m,o,h,l,c) numpy arrays sorted by m} for O(1) per-fill lookup."""
    import sqlite3
    con = sqlite3.connect(f'{HOD}/bars_fills_1478.db')
    t0 = time.time()
    df = pd.read_sql('SELECT symbol, day, t, o, h, l, c FROM bars', con)
    con.close()
    log(f'load_bars_store: {len(df)} raw bar rows loaded in {time.time()-t0:.1f}s')
    df['m'] = et_minute_of_day(df['t'])
    df = df[(df['m'] >= OPEN_M) & (df['m'] < RTH_HI)]
    df = df.sort_values(['symbol', 'day', 'm'])
    log(f'  {len(df)} rows after RTH filter [{OPEN_M},{RTH_HI})')

    store = {}
    n_missing = 0
    for (sym, day), g in df.groupby(['symbol', 'day'], sort=False):
        store[(sym, day)] = (g['m'].to_numpy(), g['o'].to_numpy(), g['h'].to_numpy(),
                              g['l'].to_numpy(), g['c'].to_numpy())
    for sym, day in need_pairs:
        if (sym, day) not in store:
            n_missing += 1
    log(f'load_bars_store: {len(store)} (symbol,day) groups; {n_missing}/{len(need_pairs)} '
        f'needed fills have NO RTH bars at all')
    return store


# ================================================================================================
# Walk primitives -- sip_rebuild.walk_path semantics (EOD-bar check first, then stop with
# gap-through-at-the-open, then target; "stop first on a bar touching both")
# ================================================================================================

def _first_true(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else -1


def walk_single(path, stop_px, target_px):
    """One stop/target pair on one fill's post-fill bar path (path = (m,o,h,l,c) numpy arrays,
    already sliced to m >= the fill bar). Returns (exit_m, exit_price, why) with why in
    {'stop','target','eod','eod_fallback'}. Mirrors sip_rebuild.walk_path exactly: a bar with
    m >= EOD_M exits at ITS OPEN before its own low/high are checked against stop/target; a bar
    touching both stop and target exits at the stop (gap-through: exit at the open if the bar
    opened through the stop, else at the stop price); a target exit is always at the exact target
    price (no gap adjustment -- it is a resting limit, not a stop)."""
    m, o, h, l, c = path
    eod_i = _first_true(m >= EOD_M)
    n_pre = eod_i if eod_i >= 0 else len(m)
    stop_i = _first_true(l[:n_pre] <= stop_px)
    targ_i = _first_true(h[:n_pre] >= target_px)
    if stop_i >= 0 and (targ_i < 0 or stop_i <= targ_i):
        px = o[stop_i] if o[stop_i] <= stop_px else stop_px
        return int(m[stop_i]), float(px), 'stop'
    if targ_i >= 0:
        return int(m[targ_i]), float(target_px), 'target'
    if eod_i >= 0:
        return int(m[eod_i]), float(o[eod_i]), 'eod'
    return int(m[-1]), float(c[-1]), 'eod_fallback'  # data ended before 15:55 (logged separately)


def first_passage(path, fill_px, k_pct, m_pct):
    """Part A: does price move +k% before -m% (from the fill), both measured off `fill_px`. Same
    EOD-first / down-first-on-a-dual-touch / gap-through convention as walk_single, with 'down'
    playing the stop role and 'up' the target role. Returns (exit_m, exit_price, outcome) with
    outcome in {'down','up','neither','neither_fallback'}."""
    m, o, h, l, c = path
    up_px = fill_px * (1 + k_pct / 100.0)
    down_px = fill_px * (1 - m_pct / 100.0)
    eod_i = _first_true(m >= EOD_M)
    n_pre = eod_i if eod_i >= 0 else len(m)
    down_i = _first_true(l[:n_pre] <= down_px)
    up_i = _first_true(h[:n_pre] >= up_px)
    if down_i >= 0 and (up_i < 0 or down_i <= up_i):
        px = o[down_i] if o[down_i] <= down_px else down_px
        return int(m[down_i]), float(px), 'down'
    if up_i >= 0:
        return int(m[up_i]), float(up_px), 'up'
    if eod_i >= 0:
        return int(m[eod_i]), float(o[eod_i]), 'neither'
    return int(m[-1]), float(c[-1]), 'neither_fallback'


def net_result(exit_price, why, fill_px, c_in, c_out, split):
    """Cost model shared by every barrier cell: entry always pays c_in; a stop-type exit pays the
    stop-limit-standard c_out; an EOD-type exit pays the EOD-at-the-bid bps of the exit price; a
    target exit (a resting limit fill) pays nothing further. No cost primitive for a "target" exit
    is given in the PREREG's Inputs, consistent with walk_path's own target leg returning the exact
    limit price with no adjustment."""
    if why == 'stop':
        exit_cost = c_out
    elif why in ('eod', 'eod_fallback'):
        exit_cost = exit_price * EOD_BID_BPS[split] / 1e4
    else:  # target
        exit_cost = 0.0
    return (exit_price - fill_px) - c_in - exit_cost


# ================================================================================================
# Part A -- the drift map
# ================================================================================================

def build_driftmap(fills, store):
    """For every fill: walk all 36 (k,m) pairs once (shared bar scan), record outcome/exit, and
    the signed excursion at each horizon. Returns (grid_rows_df for cell 1,616 reuse, driftmap_agg_df)."""
    n = len(fills)
    n_pairs = len(K_M_GRID) ** 2
    outcome_grid = np.empty((n, n_pairs), dtype=object)
    exitpx_grid = np.full((n, n_pairs), np.nan)
    exc = {h: np.full(n, np.nan) for h in HORIZONS_MIN}
    entry_gap_bps = np.full(n, np.nan)
    n_no_bars = 0

    for i, row in enumerate(fills.itertuples()):
        if i % 2000 == 0:
            log(f'  build_driftmap: fill {i}/{n}')
        path = store.get((row.symbol, row.day))
        entry_gap_bps[i] = (row.fill - row.level) / row.fill * 1e4
        if path is None or len(path[0]) == 0:
            n_no_bars += 1
            continue
        m_arr = path[0]
        sl = m_arr >= row.fill_bar_m
        p = tuple(a[sl] for a in path)
        if len(p[0]) == 0:
            n_no_bars += 1
            continue
        for j, k_pct in enumerate(K_M_GRID):
            for kk, m_pct in enumerate(K_M_GRID):
                pair_idx = j * len(K_M_GRID) + kk
                _, px, outcome = first_passage(p, row.fill, k_pct, m_pct)
                outcome_grid[i, pair_idx] = outcome
                exitpx_grid[i, pair_idx] = px
        for hz in HORIZONS_MIN:
            target_m = min(row.fill_bar_m + hz, EOD_M)
            sel = p[0] <= target_m
            if sel.any():
                last_c = p[4][sel][-1]
                exc[hz][i] = (last_c / row.fill - 1) * 1e4

    log(f'build_driftmap: {n_no_bars}/{n} fills with no usable RTH bar path')
    fills = fills.copy()
    fills['entry_gap_bps'] = entry_gap_bps
    for hz in HORIZONS_MIN:
        fills[f'exc_{hz}m_bps'] = exc[hz]

    # ---- aggregate the 36-pair grid per (holdout, quintile-or-ALL) ----
    agg_rows = []
    pairs = [(k, m) for k in K_M_GRID for m in K_M_GRID]
    for split in ('TRAIN', 'VAL'):
        split_mask = (fills['split'] == split).to_numpy()
        buckets = [('ALL', split_mask)]
        for q in QUINTILES:
            buckets.append((q, split_mask & (fills['quintile'] == q).to_numpy()))
        for bucket_name, mask in buckets:
            idx = np.flatnonzero(mask)
            n_tot = len(idx)
            for pi, (k_pct, m_pct) in enumerate(pairs):
                outs = outcome_grid[idx, pi]
                n_up = int(np.sum(outs == 'up'))
                n_down = int(np.sum(outs == 'down'))
                n_neither = n_tot - n_up - n_down
                n_resolved = n_up + n_down
                p_uncond = n_up / n_tot if n_tot else np.nan
                p_cond = n_up / n_resolved if n_resolved else np.nan
                driftless = m_pct / (k_pct + m_pct)  # see REBUILD_1610.md Note 1 (derivation)
                lo, hi = wilson_ci(n_up, n_resolved)
                agg_rows.append(dict(holdout=split, quintile=bucket_name, k_pct=k_pct, m_pct=m_pct,
                                      n_total=n_tot, n_up_first=n_up, n_down_first=n_down,
                                      n_neither=n_neither, p_up_unconditional=p_uncond,
                                      p_up_conditional=p_cond, driftless_p=driftless,
                                      wilson_lo=lo, wilson_hi=hi,
                                      resolves_driftless=(not np.isnan(p_cond)) and (lo <= driftless <= hi)))
    driftmap = pd.DataFrame(agg_rows)
    return fills, outcome_grid, exitpx_grid, pairs, driftmap


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z ** 2 / n
    center = phat + z ** 2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return ((center - margin) / denom, (center + margin) / denom)


# ================================================================================================
# Part B -- the six barrier cells + the report-only quintile-optimal cell
# ================================================================================================

def barrier_levels(fills):
    """fill/stop/target price for cells 1,610-1,615, all keyed off the BASE R = fill - stop."""
    f, R = fills['fill'], fills['R_base']
    lv = {}
    lv[1610] = (f - 0.75 * R, f + 1.5 * R)
    lv[1611] = (f - 0.75 * R, f + 2.0 * R)
    lv[1612] = (f - R, f + 2.0 * R + fills['c_in'] + fills['c_out'])
    lv[1613] = (f - R - fills['c_out'], f + 2.0 * R)
    lv[1614] = (f - R - fills['c_out'], f + 2.0 * R + fills['c_in'] + fills['c_out'])
    R_bps = R / f * 1e4
    s = (fills['spread_bps_at_arm'] / R_bps).clip(upper=1.0)
    lv[1615] = (f - R * (1 + s), f + 2.0 * R * (1 + s))
    return lv


def run_barrier_cells(fills, store):
    """Walk cells 1,610-1,615 on the actual bar path for every fill."""
    lv = barrier_levels(fills)
    n = len(fills)
    out = {cid: {'stop': np.full(n, np.nan), 'target': np.full(n, np.nan),
                 'exit_m': np.full(n, np.nan), 'exit_price': np.full(n, np.nan),
                 'why': np.empty(n, dtype=object)}
           for cid in lv}
    for i, row in enumerate(fills.itertuples()):
        if i % 2000 == 0:
            log(f'  run_barrier_cells: fill {i}/{n}')
        path = store.get((row.symbol, row.day))
        if path is None or len(path[0]) == 0:
            for cid in lv:
                out[cid]['why'][i] = 'no_bars'
            continue
        sl = path[0] >= row.fill_bar_m
        p = tuple(a[sl] for a in path)
        if len(p[0]) == 0:
            for cid in lv:
                out[cid]['why'][i] = 'no_bars'
            continue
        for cid, (stop_s, target_s) in lv.items():
            stop_px, target_px = stop_s.iloc[i], target_s.iloc[i]
            exit_m, exit_price, why = walk_single(p, stop_px, target_px)
            out[cid]['stop'][i] = stop_px
            out[cid]['target'][i] = target_px
            out[cid]['exit_m'][i] = exit_m
            out[cid]['exit_price'][i] = exit_price
            out[cid]['why'][i] = why
    return out


def add_cell_columns(fills, cid, stop, target, exit_m, exit_price, why):
    """net R in BASE-R units, % of price, and paired delta vs outcome_R, for one cell."""
    f = fills.copy()
    net_price = np.array([net_result(exit_price[i], why[i], f['fill'].iloc[i], f['c_in'].iloc[i],
                                      f['c_out'].iloc[i], f['split'].iloc[i])
                           if why[i] != 'no_bars' else np.nan for i in range(len(f))])
    f[f'c{cid}_stop'] = stop
    f[f'c{cid}_target'] = target
    f[f'c{cid}_exit_m'] = exit_m
    f[f'c{cid}_exit_price'] = exit_price
    f[f'c{cid}_why'] = why
    f[f'c{cid}_net_price'] = net_price
    f[f'c{cid}_net_R_base'] = net_price / f['R_base']
    f[f'c{cid}_net_pct'] = net_price / f['fill'] * 100.0
    f[f'c{cid}_paired_delta_R'] = f[f'c{cid}_net_R_base'] - f['outcome_R']
    f[f'c{cid}_paired_delta_pct'] = f[f'c{cid}_paired_delta_R'] * f['R_base_pct']
    return f


def run_cell_1616(fills, outcome_grid, exitpx_grid, pairs):
    """Report-only: per TRAIN-H2 spread quintile, pick the (k,m) grid pair with the highest
    TRAIN-H2 mean net %-of-price (using the SAME cost model as every other cell); apply that fixed
    pair to VAL fills in the same quintile. TRAIN-H2's own number for the selected pair is reported
    too, clearly marked in-sample (it is not evaluated out of sample by construction)."""
    n = len(fills)
    net_pct_grid = np.full((n, len(pairs)), np.nan)
    for i, row in enumerate(fills.itertuples()):
        for pi, (k_pct, m_pct) in enumerate(pairs):
            outcome = outcome_grid[i, pi]
            if outcome is None:
                continue
            px = exitpx_grid[i, pi]
            why = {'down': 'stop', 'up': 'target'}.get(outcome, 'eod')
            net_price = net_result(px, why, row.fill, row.c_in, row.c_out, row.split)
            net_pct_grid[i, pi] = net_price / row.fill * 100.0

    selected = {}
    train_mask = (fills['split'] == 'TRAIN').to_numpy()
    for q in QUINTILES:
        qmask = train_mask & (fills['quintile'] == q).to_numpy()
        means = np.nanmean(net_pct_grid[qmask, :], axis=0) if qmask.any() else np.full(len(pairs), np.nan)
        best_pi = int(np.nanargmax(means)) if not np.all(np.isnan(means)) else None
        selected[q] = (pairs[best_pi], means[best_pi]) if best_pi is not None else (None, np.nan)
        log(f'  1616 TRAIN-H2-selected pair for {q}: k,m={selected[q][0]} '
            f'TRAIN-H2 mean net%={selected[q][1]:.4f}')

    stop = np.full(n, np.nan); target = np.full(n, np.nan)
    exit_m = np.full(n, np.nan); exit_price = np.full(n, np.nan); why = np.empty(n, dtype=object)
    k_sel = np.full(n, np.nan); m_sel = np.full(n, np.nan)
    for i, row in enumerate(fills.itertuples()):
        q = row.quintile
        pair, _ = selected.get(q, (None, np.nan))
        if pair is None:
            why[i] = 'no_selection'
            continue
        k_pct, m_pct = pair
        k_sel[i] = k_pct; m_sel[i] = m_pct
        pi = pairs.index(pair)
        outcome = outcome_grid[i, pi]
        px = exitpx_grid[i, pi]
        stop[i] = row.fill * (1 - m_pct / 100.0)
        target[i] = row.fill * (1 + k_pct / 100.0)
        exit_price[i] = px
        exit_m[i] = np.nan
        why[i] = {'down': 'stop', 'up': 'target', 'neither': 'eod', 'neither_fallback': 'eod_fallback'}.get(outcome, 'no_bars')

    f = add_cell_columns(fills, 1616, stop, target, exit_m, exit_price, why)
    f['c1616_k_selected'] = k_sel
    f['c1616_m_selected'] = m_sel
    f['c1616_selected_on'] = 'TRAIN-H2_per_quintile'
    return f, selected


# ================================================================================================
# Stats for REBUILD_1610.md
# ================================================================================================

def cell_stats(f, cid, holdout, bucket_name, mask):
    sub = f.loc[mask]
    n = len(sub)
    if n == 0:
        return None
    net_pct = sub[f'c{cid}_net_pct']
    net_R = sub[f'c{cid}_net_R_base']
    delta_pct = sub[f'c{cid}_paired_delta_pct']
    delta_R = sub[f'c{cid}_paired_delta_R']
    t = c1445.day_clustered_t(net_pct, sub['day'])
    t_delta = c1445.day_clustered_t(delta_pct, sub['day'])
    ex5_pct = c1445.ex_top5_mean(net_pct)
    ex5_delta = c1445.ex_top5_mean(delta_pct)
    weeks = c1445.weeks_spanned(sub['day'])
    why_mix = sub[f'c{cid}_why'].value_counts(normalize=True).to_dict()
    return dict(cell=cid, holdout=holdout, bucket=bucket_name, n=n,
                mean_net_pct=net_pct.mean(), mean_net_R=net_R.mean(), day_t=t,
                ex_top5_net_pct=ex5_pct, mean_delta_pct=delta_pct.mean(),
                mean_delta_R=delta_R.mean(), day_t_delta=t_delta, ex_top5_delta_pct=ex5_delta,
                median_R_pct=sub['R_base_pct'].median(), fills_per_week=n / weeks, weeks=weeks,
                why_mix=why_mix)


def build_partB_summary(f):
    cells = [1610, 1611, 1612, 1613, 1614, 1615, 1616]
    rows = []
    for cid in cells:
        for holdout in ('TRAIN', 'VAL'):
            hmask = (f['split'] == holdout).to_numpy()
            buckets = [('ALL', hmask)]
            for q in QUINTILES:
                buckets.append((q, hmask & (f['quintile'] == q).to_numpy()))
            for bname, mask in buckets:
                r = cell_stats(f, cid, holdout, bname, mask)
                if r:
                    rows.append(r)
    return pd.DataFrame(rows)


# ================================================================================================
# Main
# ================================================================================================

def main():
    t0 = time.time()
    fills = load_fills()
    fills, edges = assign_quintiles(fills)
    need_pairs = list(zip(fills['symbol'], fills['day']))
    store = load_bars_store(need_pairs)

    log('=== Part A: drift map ===')
    fills, outcome_grid, exitpx_grid, pairs, driftmap = build_driftmap(fills, store)
    driftmap_path = f'{HERE}/rebuild_1610_driftmap.csv'
    driftmap.to_csv(driftmap_path, index=False)
    log(f'wrote {driftmap_path} ({len(driftmap)} rows)')

    log('=== Part B: barrier cells 1,610-1,615 ===')
    cell_out = run_barrier_cells(fills, store)
    f = fills
    for cid in (1610, 1611, 1612, 1613, 1614, 1615):
        o = cell_out[cid]
        f = add_cell_columns(f, cid, o['stop'], o['target'], o['exit_m'], o['exit_price'], o['why'])

    log('=== Part B: cell 1,616 (report-only, quintile-optimal) ===')
    f, selected_1616 = run_cell_1616(f, outcome_grid, exitpx_grid, pairs)

    fills_path = f'{HERE}/rebuild_1610_fills.csv'
    f.to_csv(fills_path, index=False)
    log(f'wrote {fills_path} ({len(f)} rows, {len(f.columns)} cols)')

    summary = build_partB_summary(f)
    summary_path = f'{HERE}/rebuild_1610_summary.csv'
    summary.to_csv(summary_path, index=False)
    log(f'wrote {summary_path} ({len(summary)} rows) [supporting detail for REBUILD_1610.md]')

    log(f'DONE in {time.time()-t0:.1f}s')
    return fills, driftmap, f, summary, edges, selected_1616


if __name__ == '__main__':
    main()
