#!/usr/bin/env python3
"""HOD-break Exit Lab — score_cells.py (PREREG.md, cells 1,359-1,379).

Scores all 21 cells against B0 (the live rule), PAIRED per trade where the cell changes the exit
(X1-X12), or as a kept/dropped cohort split of B0's own trades where the cell changes the
population (D1-D5, W1, W2). Reads ONLY the cached artifacts walker.py already wrote
(paths.parquet, signals.parquet, b0_trades.csv) plus small reference files (features.csv for
n_prior/entry_m join keys, SPY_1min.parquet for D1, hmm_labels.csv for D4). NO DB ACCESS — per
PREREG.md "Method", the DB is touched exactly once (by walker.py) and never again here.

Explicit mechanism notes (documented here because PREREG.md states the RULE, not the fill code):
  - Look-ahead rule (PREREG.md line 50-51), applied identically to every path-walking cell: a level
    computed from bar t's high/low can only ACT from bar t+1 (never the same bar); within a bar,
    stop is checked before target (conservative, matches b0_fill); a lock/trail stop gapped through
    fills at that bar's open, not the stale stop price.
  - X8 (partial) cost: PREREG does not specify partial-fill cost mechanics. This script charges the
    SAME total cost formula as every other cell (2 x half-spread + 2bp both legs) using the
    SIZE-WEIGHTED BLENDED exit price (0.5 x partial-fill price + 0.5 x runner exit price) as the
    stand-in "exit_px" for the slippage term - i.e. the entry-side spread is paid once (correct,
    entered as one order) and the exit-side spread/slippage is approximated as a single blended
    round trip rather than two separately-sized legs. Flagged in CELLS.md; does not change the
    ranking of X8 against the pass bar materially given typical spread sizes (~0.1-0.4R).
  - X11/X12 (stop distance change): target stays at the ORIGINAL dollar level (entry + 2*R_base);
    only the stop moves. R_own = entry - stop_new. own-R metrics divide by R_own; baseline-R metrics
    divide by R_base (= B0's R for that trade). net_R_base = net_R_own * (R_own / R_base) - both are
    reported per PREREG pass-bar rule 6.
  - D2 "breadth" is read as MARKET-WIDE signal count that day before this signal's entry_m (not
    same-symbol n_prior, which FACTS.md documents as a different, per-symbol field already used by
    D5). Threshold = the TRAIN-H1 median of that count, computed once and reused for TRAIN/VAL.
  - W1 week gate and W2 daily kill are both defined on B0's OWN net_R (the exit never changes for
    D/W cells): W1 keeps trades in week t only if week t-1's SUMMED B0 net_R > 0 (weeks ordered
    chronologically by their start date, across the full TRAIN+VAL timeline so the VAL split's first
    weeks can see TRAIN's last week). W2 kills NEW ENTRIES for the rest of a calendar day once the
    SUM of already-CLOSED (exit_m <= this entry_m) B0 net_R that day drops to <= -3; the kill is
    causal (uses only trades that have already exited) and, once tripped, persists for the rest of
    that day (standard kill-switch semantics), even if a later closed trade would nudge the running
    sum back up.
  - D/W cells are "cohort" cells: the exit is unchanged, so "dR" is defined per PREREG's own framing
    ("kept cohort" vs "dropped cohort") as KEPT_i = b0_net_R_i - mean(ALL B0 net_R in that split);
    i.e. each kept trade's pull above/below the full B0 population mean for that split. The
    day-clustered t-stat and the +0.10 R bar are computed on that quantity. This is the natural
    generalization of "paired ΔR" to a pure selection filter (a kept trade's R doesn't change; what
    changes is which trades you're averaging).

Usage:
  python3 score_cells.py --smoke   # runs on *_smoke.{parquet,csv}, writes to smoke_out/ (no clobber)
  python3 score_cells.py           # full run on paths.parquet / signals.parquet / b0_trades.csv
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
LAB = f'{ROOT}/research/hod_exit_lab'
CF_DIR = f'{ROOT}/research/bf_zero/causal_filter'
FEATURES_CSV = f'{CF_DIR}/features.csv'
SPY_PARQUET = f'{ROOT}/research/index_orb/cache/SPY_1min.parquet'
HMM_CSV = f'{ROOT}/research/regime/hmm_labels.csv'
CADENCE = f'{ROOT}/scripts/cadence_bar.py'

EOD_M = 955
TARGET_R_MULT = 2.0
SLIP_BP = 0.0002
NOON_M = 720          # 12:00 ET
TWO_PM_M = 840         # 14:00 ET

X_CELLS = ['X1', 'X2', 'X3', 'X4', 'X5', 'X5b', 'X6', 'X7', 'X7b', 'X8', 'X9', 'X10', 'X11', 'X12']
DW_CELLS = ['D1', 'D2', 'D3', 'D4', 'D5', 'W1', 'W2']


def log(msg):
    print(f'[score_cells] {msg}', flush=True)


# --------------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------------

def load_data(smoke):
    sfx = '_smoke' if smoke else ''
    paths = pd.read_parquet(f'{LAB}/paths{sfx}.parquet')
    sig = pd.read_parquet(f'{LAB}/signals{sfx}.parquet')
    b0 = pd.read_csv(f'{LAB}/b0_trades{sfx}.csv', dtype={'day': str, 'symbol': str})
    b0 = b0[b0.split.isin(['TRAIN', 'VAL'])].copy()          # TEST stays sealed
    b0 = b0[b0.net_R.notna()].copy()                          # NBBO coverage gaps drop cleanly
    sig = sig.merge(b0[['day', 'symbol', 'entry_m']].drop_duplicates(), on=['day', 'symbol', 'entry_m'])
    log(f'b0 trades: {len(b0)} (TRAIN={ (b0.split=="TRAIN").sum() }, VAL={ (b0.split=="VAL").sum() })')

    f = pd.read_csv(FEATURES_CSV, dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
    feat = f[['day', 'symbol', 'entry_m', 'n_prior']].drop_duplicates(['day', 'symbol', 'entry_m'])

    spy_open930, spy_at_m = load_spy()
    hmm = load_hmm()

    idx = paths.sort_values(['day', 'symbol', 'm']).set_index(['day', 'symbol']).sort_index()
    return b0, sig, feat, idx, spy_open930, spy_at_m, hmm


def load_spy():
    s = pd.read_parquet(SPY_PARQUET)
    ts = s.timestamp.dt.tz_convert('America/New_York')
    day = ts.dt.strftime('%Y-%m-%d')
    hm = ts.dt.strftime('%H:%M')
    key = day + ' ' + hm
    at_m = dict(zip(key, s['close'].values))
    open930 = dict(zip(day[hm == '09:30'], s.loc[hm == '09:30', 'open'].values))
    log(f'spy: {len(open930)} daily 09:30 opens, {len(at_m)} minute closes')
    return open930, at_m


def load_hmm():
    h = pd.read_csv(HMM_CSV, dtype={'bar_date': str})
    return dict(zip(h.bar_date, h.hmm_state))


# --------------------------------------------------------------------------------------------
# Shared fill mechanics
# --------------------------------------------------------------------------------------------

def _gap_or_touch(o, l, level):
    """Stop-side fill price: gap-through fills at the open, else the stop price itself."""
    return o if o <= level else level


def cost_net(entry, exit_px, R, spread_mean):
    """Same formula as walker.b0_fill's cost: half-spread both legs + 2bp per side, in units of R."""
    if pd.isna(spread_mean):
        return np.nan, np.nan
    half_R = (0.5 * spread_mean) / R
    slip_R = SLIP_BP * (entry + exit_px) / R
    cost_R = 2 * half_R + slip_R
    raw_rr = (exit_px - entry) / R
    return cost_R, raw_rr - cost_R


def b0_style_fill(entry, stop, target, bars):
    """Identical priority to walker.b0_fill: eod first (>=EOD_M), then stop (incl. both-touch =
    stop, conservative), then target. `bars` = itertuples of the path AFTER entry_m, m<=EOD_M."""
    for row in bars:
        if row.m >= EOD_M:
            return int(row.m), float(row.o), 'eod'
        if row.l <= stop:
            return int(row.m), float(_gap_or_touch(row.o, row.l, stop)), 'stop'
        if row.h >= target:
            return int(row.m), float(target), 'target'
    return None  # should not happen; EOD_M row is always in range for a valid path


# --------------------------------------------------------------------------------------------
# X-cell exit variants — each returns (exit_m, exit_price, why)
# --------------------------------------------------------------------------------------------

def x1_no_target(entry, stop, R, bars):
    for row in bars:
        if row.m >= EOD_M:
            return int(row.m), float(row.o), 'eod'
        if row.l <= stop:
            return int(row.m), float(_gap_or_touch(row.o, row.l, stop)), 'stop'
    return None


def x_target_mult(entry, stop, R, bars, mult):
    return b0_style_fill(entry, stop, entry + mult * R, bars)


def x5_lock(entry, stop0, target, R, bars, trigger_mult, lock_mult):
    """X5 (trigger=1, lock=0 i.e. breakeven) and X5b (trigger=1.5, lock=0.5). Lock arms and ratchets
    from bar t's high, takes effect starting bar t+1 (checked BEFORE this bar's own update)."""
    cur_stop = stop0
    for row in bars:
        if row.m >= EOD_M:
            return int(row.m), float(row.o), 'eod'
        if row.l <= cur_stop:
            return int(row.m), float(_gap_or_touch(row.o, row.l, cur_stop)), 'stop'
        if row.h >= target:
            return int(row.m), float(target), 'target'
        if row.h >= entry + trigger_mult * R:
            cur_stop = max(cur_stop, entry + lock_mult * R)
    return None


def x6_trail(entry, stop0, target, R, bars):
    cur_stop = stop0
    armed = False
    running_high = None
    for row in bars:
        if row.m >= EOD_M:
            return int(row.m), float(row.o), 'eod'
        if row.l <= cur_stop:
            return int(row.m), float(_gap_or_touch(row.o, row.l, cur_stop)), 'stop'
        if row.h >= target:
            return int(row.m), float(target), 'target'
        if not armed and row.h >= entry + 1.0 * R:
            armed = True
            running_high = row.h
        elif armed:
            running_high = max(running_high, row.h)
        if armed:
            cur_stop = max(cur_stop, running_high - R)
    return None


def x_time_stop(entry, stop0, target, R, bars, entry_m, delay_min, ref_price):
    cur_stop = stop0
    trig_m = None
    for row in bars:
        if row.m >= EOD_M:
            return int(row.m), float(row.o), 'eod'
        if row.l <= cur_stop:
            return int(row.m), float(_gap_or_touch(row.o, row.l, cur_stop)), 'stop'
        if row.h >= target:
            return int(row.m), float(target), 'target'
        if trig_m is not None and row.m > trig_m:
            return int(row.m), float(row.o), 'time_stop'
        if trig_m is None and (row.m - entry_m) >= delay_min and row.c < ref_price:
            trig_m = row.m
    return None


def x8_partial(entry, stop0, R, bars):
    """50% at +1R, runner locks to breakeven (X5 mechanics) with NO target, 15:55 close. Returns
    (exit_m, blended_exit_price, why, half1_price_or_None) — half1_price is None if the whole
    position stopped out before the +1R touch (single-leg trade)."""
    cur_stop = stop0
    half1_filled = False
    half1_price = None
    for row in bars:
        if row.m >= EOD_M:
            if half1_filled:
                blend = 0.5 * half1_price + 0.5 * row.o
                return int(row.m), float(blend), 'eod_partial', half1_price
            return int(row.m), float(row.o), 'eod', None
        if row.l <= cur_stop:
            px = _gap_or_touch(row.o, row.l, cur_stop)
            if half1_filled:
                blend = 0.5 * half1_price + 0.5 * px
                return int(row.m), float(blend), 'stop_runner', half1_price
            return int(row.m), float(px), 'stop', None
        if not half1_filled and row.h >= entry + 1.0 * R:
            half1_filled = True
            half1_price = entry + 1.0 * R
            cur_stop = max(cur_stop, entry)  # lock arms, effective next bar
    return None


def x9_vwap(entry, stop0, target, R, bars):
    trig_m = None
    for row in bars:
        if row.m >= EOD_M:
            return int(row.m), float(row.o), 'eod'
        if row.l <= stop0:
            return int(row.m), float(_gap_or_touch(row.o, row.l, stop0)), 'stop'
        if row.h >= target:
            return int(row.m), float(target), 'target'
        if trig_m is not None and row.m > trig_m:
            return int(row.m), float(row.o), 'vwap_exit'
        if trig_m is None and row.c < row.vwap:
            trig_m = row.m
    return None


def x10_clock(entry, stop0, target, R, bars):
    for row in bars:
        if row.m >= NOON_M:
            return int(row.m), float(row.o), 'clock_1200'
        if row.m >= EOD_M:
            return int(row.m), float(row.o), 'eod'
        if row.l <= stop0:
            return int(row.m), float(_gap_or_touch(row.o, row.l, stop0)), 'stop'
        if row.h >= target:
            return int(row.m), float(target), 'target'
    return None


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------

def day_clustered_t(dr, day):
    """Day-clustered: per-day mean of dR, then t = mean(day means) / (std(day means)/sqrt(n_days))."""
    dm = pd.Series(dr.values, index=day.values).groupby(level=0).mean()
    n = len(dm)
    if n < 2 or dm.std(ddof=1) == 0 or pd.isna(dm.std(ddof=1)):
        return np.nan, n
    se = dm.std(ddof=1) / np.sqrt(n)
    return float(dm.mean() / se), n


def weekly_mdd(net_R, wk):
    """Weekly cumulative-R max drawdown (R units, reported as a positive magnitude)."""
    w = pd.Series(net_R.values, index=wk.values).groupby(level=0).sum().sort_index()
    cum = w.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    return float(-dd.min()) if len(dd) else 0.0


def ex_top5(net_R):
    if len(net_R) < 20:
        return float(net_R.mean())
    thr = net_R.quantile(0.95)
    return float(net_R[net_R < thr].mean())


def score_exit_cell(cell_id, name, work, note=''):
    """work: DataFrame with day, symbol, entry_m, split, half, wk, net_R (variant), b0_net_R."""
    out = dict(id=cell_id, name=name, kind='exit', note=note)
    w = work.dropna(subset=['net_R', 'b0_net_R']).copy()
    w['dR'] = w.net_R - w.b0_net_R
    tr = w[w.split == 'TRAIN']
    va = w[w.split == 'VAL']
    h1 = tr[tr.half == 'H1']
    h2 = tr[tr.half == 'H2']
    out['train_n'], out['val_n'] = len(tr), len(va)
    out['train_dR'] = float(tr.dR.mean()) if len(tr) else np.nan
    out['val_dR'] = float(va.dR.mean()) if len(va) else np.nan
    out['h1_dR'] = float(h1.dR.mean()) if len(h1) else np.nan
    out['h2_dR'] = float(h2.dR.mean()) if len(h2) else np.nan
    t, ndays = day_clustered_t(va.dR, va.day) if len(va) else (np.nan, 0)
    out['val_t_dR'], out['val_t_ndays'] = t, ndays
    out['dropped_R_train'] = out['train_dR']
    out['dropped_R_val'] = out['val_dR']

    et5_var_val = ex_top5(va.net_R) if len(va) else np.nan
    et5_b0_val = ex_top5(va.b0_net_R) if len(va) else np.nan
    et5_var_tr = ex_top5(tr.net_R) if len(tr) else np.nan
    et5_b0_tr = ex_top5(tr.b0_net_R) if len(tr) else np.nan
    out['extop5_variant_val'], out['extop5_b0_val'] = et5_var_val, et5_b0_val
    out['extop5_variant_train'], out['extop5_b0_train'] = et5_var_tr, et5_b0_tr
    extop5_ok = (et5_var_val >= et5_b0_val) and (et5_var_tr >= et5_b0_tr) if len(va) and len(tr) else False
    out['extop5_ok'] = bool(extop5_ok)

    mdd_var_tr = weekly_mdd(tr.net_R, tr.wk)
    mdd_b0_tr = weekly_mdd(tr.b0_net_R, tr.wk)
    mdd_var_va = weekly_mdd(va.net_R, va.wk)
    mdd_b0_va = weekly_mdd(va.b0_net_R, va.wk)
    out['mdd_variant_train'], out['mdd_b0_train'] = mdd_var_tr, mdd_b0_tr
    out['mdd_variant_val'], out['mdd_b0_val'] = mdd_var_va, mdd_b0_va
    mdd_ok = (mdd_var_tr <= 1.25 * mdd_b0_tr if mdd_b0_tr > 0 else mdd_var_tr <= 0.01) and \
             (mdd_var_va <= 1.25 * mdd_b0_va if mdd_b0_va > 0 else mdd_var_va <= 0.01)
    out['mdd_ok'] = bool(mdd_ok)

    rule1 = (out['train_dR'] >= 0.10) and (out['val_dR'] >= 0.10) and (not pd.isna(t)) and (t >= 2.0)
    rule2 = (out['h1_dR'] > 0) and (out['h2_dR'] > 0)
    rule3 = out['extop5_ok']
    rule4 = out['mdd_ok']
    out['pass'] = bool(rule1 and rule2 and rule3 and rule4)
    out['rule1_paired'], out['rule2_halves'] = bool(rule1), bool(rule2)
    out['_trades'] = w
    return out


def score_cohort_cell(cell_id, name, b0, keep_mask, note=''):
    out = dict(id=cell_id, name=name, kind='cohort', note=note)
    b0 = b0.copy()
    b0['keep'] = keep_mask.values if hasattr(keep_mask, 'values') else keep_mask
    for split in ('TRAIN', 'VAL'):
        pop = b0[b0.split == split]
        base_mean = pop.net_R.mean() if len(pop) else np.nan
        kept = pop[pop.keep]
        dropped = pop[~pop.keep]
        dR = kept.net_R - base_mean
        if split == 'TRAIN':
            h1 = kept[kept.half == 'H1']
            h2 = kept[kept.half == 'H2']
            out['h1_dR'] = float((h1.net_R - pop[pop.half == 'H1'].net_R.mean()).mean()) if len(h1) else np.nan
            out['h2_dR'] = float((h2.net_R - pop[pop.half == 'H2'].net_R.mean()).mean()) if len(h2) else np.nan
            out['train_dR'] = float(dR.mean()) if len(kept) else np.nan
            out['train_n'] = len(kept)
            out['dropped_R_train'] = float(dropped.net_R.mean()) if len(dropped) else np.nan
        else:
            out['val_dR'] = float(dR.mean()) if len(kept) else np.nan
            out['val_n'] = len(kept)
            out['dropped_R_val'] = float(dropped.net_R.mean()) if len(dropped) else np.nan
            t, ndays = day_clustered_t(dR, kept.day) if len(kept) else (np.nan, 0)
            out['val_t_dR'], out['val_t_ndays'] = t, ndays
            nwk = kept.wk.nunique()
            out['val_fills_wk'] = float(len(kept) / nwk) if nwk else 0.0
            et5_k = ex_top5(kept.net_R) if len(kept) else np.nan
            et5_b0 = ex_top5(pop.net_R) if len(pop) else np.nan
            out['extop5_variant_val'], out['extop5_b0_val'] = et5_k, et5_b0
    tr_pop = b0[b0.split == 'TRAIN']
    kept_tr = tr_pop[tr_pop.keep]
    et5_k_tr = ex_top5(kept_tr.net_R) if len(kept_tr) else np.nan
    et5_b0_tr = ex_top5(tr_pop.net_R) if len(tr_pop) else np.nan
    out['extop5_variant_train'], out['extop5_b0_train'] = et5_k_tr, et5_b0_tr
    extop5_ok = (out.get('extop5_variant_val', np.nan) >= out.get('extop5_b0_val', np.nan)) and \
                (et5_k_tr >= et5_b0_tr)
    out['extop5_ok'] = bool(extop5_ok) if not pd.isna(extop5_ok) else False

    mdd_k_tr = weekly_mdd(kept_tr.net_R, kept_tr.wk)
    mdd_b0_tr = weekly_mdd(tr_pop.net_R, tr_pop.wk)
    va_pop = b0[b0.split == 'VAL']
    kept_va = va_pop[va_pop.keep]
    mdd_k_va = weekly_mdd(kept_va.net_R, kept_va.wk)
    mdd_b0_va = weekly_mdd(va_pop.net_R, va_pop.wk)
    out['mdd_variant_train'], out['mdd_b0_train'] = mdd_k_tr, mdd_b0_tr
    out['mdd_variant_val'], out['mdd_b0_val'] = mdd_k_va, mdd_b0_va
    mdd_ok = (mdd_k_tr <= 1.25 * mdd_b0_tr if mdd_b0_tr > 0 else mdd_k_tr <= 0.01) and \
             (mdd_k_va <= 1.25 * mdd_b0_va if mdd_b0_va > 0 else mdd_k_va <= 0.01)
    out['mdd_ok'] = bool(mdd_ok)

    t = out.get('val_t_dR', np.nan)
    rule1 = (out.get('train_dR', np.nan) >= 0.10) and (out.get('val_dR', np.nan) >= 0.10) and \
            (not pd.isna(t)) and (t >= 2.0)
    rule2 = (out.get('h1_dR', np.nan) > 0) and (out.get('h2_dR', np.nan) > 0)
    rule3 = out['extop5_ok']
    rule4 = out['mdd_ok']
    rule5 = (out.get('val_fills_wk', 0) >= 3.0) and \
            (out.get('dropped_R_train', np.inf) < 0) and (out.get('dropped_R_val', np.inf) < 0)
    out['pass'] = bool(rule1 and rule2 and rule3 and rule4 and rule5)
    out['rule5_cohort'] = bool(rule5)
    out['_kept'] = b0[b0.keep]
    out['_dropped'] = b0[~b0.keep]
    return out


# --------------------------------------------------------------------------------------------
# Driver: build per-signal variant frame for an X-cell
# --------------------------------------------------------------------------------------------

def run_exit_variant(b0, sig, idx, fill_fn, needs_R_own=False, R_own_mult=None):
    """fill_fn(entry, stop, target, R, bars) -> (exit_m, exit_px, why) [+ half1_price for X8, handled
    separately]. Iterates b0's trades, re-walks the cached path with the variant's fill rule, and
    returns a frame with day, symbol, entry_m, split, half, wk, net_R (variant), b0_net_R."""
    sm = sig.set_index(['day', 'symbol', 'entry_m']).spread_mean
    rows = []
    for r in b0.itertuples():
        key2 = (r.day, r.symbol)
        if key2 not in idx.index:
            continue
        g = idx.loc[[key2]]
        after = g[(g.m > r.entry_m) & (g.m <= EOD_M)]
        if after.empty:
            continue
        entry, stop0, R = r.entry, r.stop, r.R
        target0 = entry + TARGET_R_MULT * R
        if needs_R_own:
            stop_new = entry - R_own_mult * R
            res = b0_style_fill(entry, stop_new, target0, after.itertuples())
            if res is None:
                continue
            exit_m, exit_px, why = res
            R_own = entry - stop_new
            try:
                spread_mean = sm.loc[(r.day, r.symbol, r.entry_m)]
            except KeyError:
                spread_mean = np.nan
            cost_own, net_own = cost_net(entry, exit_px, R_own, spread_mean)
            net_base = net_own * (R_own / R) if not pd.isna(net_own) else np.nan
            rows.append(dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, exit_m=exit_m,
                              exit_price=exit_px, why=why, split=r.split, half=r.half, wk=r.wk,
                              net_R=net_base, net_R_own=net_own, cost_R_own=cost_own, R_own=R_own,
                              b0_net_R=r.net_R))
            continue
        res = fill_fn(entry, stop0, target0, R, after.itertuples())
        if res is None:
            continue
        exit_m, exit_px, why = res[:3]
        try:
            spread_mean = sm.loc[(r.day, r.symbol, r.entry_m)]
        except KeyError:
            spread_mean = np.nan
        cost_R, net_R = cost_net(entry, exit_px, R, spread_mean)
        row = dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, exit_m=exit_m, exit_price=exit_px,
                   why=why, split=r.split, half=r.half, wk=r.wk, net_R=net_R, cost_R=cost_R,
                   b0_net_R=r.net_R)
        if len(res) == 4:  # X8 partial: half1_price marker
            row['half1_price'] = res[3]
        rows.append(row)
    return pd.DataFrame(rows)


def run_x8(b0, sig, idx):
    sm = sig.set_index(['day', 'symbol', 'entry_m']).spread_mean
    rows = []
    for r in b0.itertuples():
        key2 = (r.day, r.symbol)
        if key2 not in idx.index:
            continue
        g = idx.loc[[key2]]
        after = g[(g.m > r.entry_m) & (g.m <= EOD_M)]
        if after.empty:
            continue
        res = x8_partial(r.entry, r.stop, r.R, after.itertuples())
        if res is None:
            continue
        exit_m, exit_px, why, half1 = res
        try:
            spread_mean = sm.loc[(r.day, r.symbol, r.entry_m)]
        except KeyError:
            spread_mean = np.nan
        cost_R, net_R = cost_net(r.entry, exit_px, r.R, spread_mean)
        rows.append(dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, exit_m=exit_m,
                          exit_price=exit_px, why=why, split=r.split, half=r.half, wk=r.wk,
                          net_R=net_R, cost_R=cost_R, b0_net_R=r.net_R, half1_price=half1))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# D/W cohort masks
# --------------------------------------------------------------------------------------------

def mask_d1(b0, idx, spy_open930, spy_at_m):
    keep = []
    for r in b0.itertuples():
        o930 = spy_open930.get(r.day, np.nan)
        key2 = (r.day, r.symbol)
        at_entry = np.nan
        if key2 in idx.index:
            g = idx.loc[[key2]]
            row = g[g.m == r.entry_m]
            if len(row):
                hh = row.iloc[0].hhmm
                at_entry = spy_at_m.get(f'{r.day} {hh}', np.nan)
        keep.append(bool(at_entry > o930) if pd.notna(at_entry) and pd.notna(o930) else False)
    return pd.Series(keep, index=b0.index)


def mask_d2(b0):
    cnt = {}
    order = b0.sort_values('entry_m')
    per_day = {}
    for r in order.itertuples():
        per_day.setdefault(r.day, []).append(r.entry_m)
    breadth = {}
    for r in b0.itertuples():
        n_before = sum(1 for m in per_day[r.day] if m < r.entry_m)
        breadth[(r.day, r.symbol, r.entry_m, r.Index)] = n_before
    b0 = b0.copy()
    b0['_breadth'] = [breadth[(r.day, r.symbol, r.entry_m, r.Index)] for r in b0.itertuples()]
    h1_median = b0.loc[b0.half == 'H1', '_breadth'].median()
    return b0['_breadth'] <= h1_median, h1_median


def mask_d3(b0):
    return ~((b0.entry_m >= NOON_M) & (b0.entry_m < TWO_PM_M))


def mask_d4(b0, hmm):
    return b0.day.map(lambda d: hmm.get(d, -1) == 0)


def mask_d5(b0, feat):
    m = b0.merge(feat, on=['day', 'symbol', 'entry_m'], how='left')
    return (m.n_prior == 0).values


def mask_w1(b0):
    wk_sum = b0.groupby('wk').net_R.sum()
    wk_order = sorted(wk_sum.index, key=lambda w: w.split('/')[0])
    prior_ok = {}
    for i, w in enumerate(wk_order):
        prior_ok[w] = (i > 0) and (wk_sum[wk_order[i - 1]] > 0)
    return b0.wk.map(prior_ok).fillna(False)


def mask_w2(b0):
    keep = pd.Series(True, index=b0.index)
    for day, g in b0.groupby('day'):
        g = g.sort_values('entry_m')
        killed = False
        for idx_row in g.itertuples():
            if killed:
                keep.loc[idx_row.Index] = False
                continue
            closed_sum = g[g.exit_m <= idx_row.entry_m].net_R.sum()
            if closed_sum <= -3.0:
                killed = True
                keep.loc[idx_row.Index] = False
    return keep


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    smoke = args.smoke
    outdir = LAB if not smoke else f'{LAB}/smoke_out'
    os.makedirs(f'{outdir}/trades', exist_ok=True)

    b0, sig, feat, idx, spy_open930, spy_at_m, hmm = load_data(smoke)

    results = []

    log('scoring X-cells (path re-walk)...')
    specs = [
        ('X1', 'no target: stop or 15:55 only', lambda e, s, t, R, bars: x1_no_target(e, s, R, bars)),
        ('X2', 'target +1R', lambda e, s, t, R, bars: x_target_mult(e, s, R, bars, 1.0)),
        ('X3', 'target +3R', lambda e, s, t, R, bars: x_target_mult(e, s, R, bars, 3.0)),
        ('X4', 'target +5R', lambda e, s, t, R, bars: x_target_mult(e, s, R, bars, 5.0)),
        ('X5', 'breakeven lock @ +1R', lambda e, s, t, R, bars: x5_lock(e, s, t, R, bars, 1.0, 0.0)),
        ('X5b', 'ORB-style lock: +1.5R trig -> +0.5R', lambda e, s, t, R, bars: x5_lock(e, s, t, R, bars, 1.5, 0.5)),
        ('X6', 'trailing stop, MFE-1R once armed @1R', lambda e, s, t, R, bars: x6_trail(e, s, t, R, bars)),
        ('X7', 'time stop 60min, exit if close<entry', lambda e, s, t, R, bars: None),  # handled below (needs entry_m)
        ('X7b', 'time stop 30min, exit if close<entry+0.25R', lambda e, s, t, R, bars: None),
        ('X9', 'VWAP exit: close<VWAP -> next open', lambda e, s, t, R, bars: x9_vwap(e, s, t, R, bars)),
        ('X10', 'clock exit @ 12:00 open', lambda e, s, t, R, bars: x10_clock(e, s, t, R, bars)),
    ]
    x_frames = {}
    for cid, desc, fn in specs:
        if cid in ('X7', 'X7b'):
            continue
        wf = run_exit_variant(b0, sig, idx, fn)
        x_frames[cid] = (desc, wf)
        log(f'  {cid}: {len(wf)} trades filled')

    # X7 / X7b need entry_m inside the fill fn -> custom loop
    for cid, desc, delay, ref_mode in (('X7', 'time stop 60min, exit if close<entry', 60, 'entry'),
                                        ('X7b', 'time stop 30min, exit if close<entry+0.25R', 30, 'entry+0.25R')):
        sm = sig.set_index(['day', 'symbol', 'entry_m']).spread_mean
        rows = []
        for r in b0.itertuples():
            key2 = (r.day, r.symbol)
            if key2 not in idx.index:
                continue
            g = idx.loc[[key2]]
            after = g[(g.m > r.entry_m) & (g.m <= EOD_M)]
            if after.empty:
                continue
            target0 = r.entry + TARGET_R_MULT * r.R
            ref_price = r.entry if ref_mode == 'entry' else r.entry + 0.25 * r.R
            res = x_time_stop(r.entry, r.stop, target0, r.R, after.itertuples(), r.entry_m, delay, ref_price)
            if res is None:
                continue
            exit_m, exit_px, why = res
            try:
                spread_mean = sm.loc[(r.day, r.symbol, r.entry_m)]
            except KeyError:
                spread_mean = np.nan
            cost_R, net_R = cost_net(r.entry, exit_px, r.R, spread_mean)
            rows.append(dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, exit_m=exit_m,
                              exit_price=exit_px, why=why, split=r.split, half=r.half, wk=r.wk,
                              net_R=net_R, cost_R=cost_R, b0_net_R=r.net_R))
        x_frames[cid] = (desc, pd.DataFrame(rows))
        log(f'  {cid}: {len(rows)} trades filled')

    log('  X8: partial exit')
    x_frames['X8'] = ('50% @+1R, runner=breakeven-lock no target', run_x8(b0, sig, idx))

    log('  X11/X12: stop distance change')
    x11 = run_exit_variant(b0, sig, idx, None, needs_R_own=True, R_own_mult=1.5)
    x12 = run_exit_variant(b0, sig, idx, None, needs_R_own=True, R_own_mult=0.75)
    x_frames['X11'] = ('stop widened 1.5x', x11)
    x_frames['X12'] = ('stop tightened 0.75x', x12)

    for cid in X_CELLS:
        desc, wf = x_frames[cid]
        if wf.empty:
            results.append(dict(id=cid, name=desc, kind='exit', **{'pass': False}, note='EMPTY (no fills)'))
            continue
        r = score_exit_cell(cid, desc, wf)
        if cid in ('X11', 'X12'):
            r['note'] = (r.get('note', '') + ' own-R reported in trades CSV (net_R_own, cost_R_own, R_own); '
                         'pass-bar metrics above are in BASELINE-R per PREREG rule 6.').strip()
        results.append(r)
        wf.assign(date=wf.day, pnl_R=wf.net_R).to_csv(f'{outdir}/trades/{cid}.csv', index=False)

    log('scoring D/W cells (cohort filters on B0)...')
    d1 = mask_d1(b0, idx, spy_open930, spy_at_m)
    r = score_cohort_cell('D1', 'SPY above 09:30 open at signal minute', b0, d1)
    results.append(r)
    r['_kept'].assign(date=lambda x: x.day, pnl_R=lambda x: x.net_R).to_csv(f'{outdir}/trades/D1.csv', index=False)

    d2, thr2 = mask_d2(b0)
    r = score_cohort_cell('D2', f'breadth <= TRAIN-H1 median ({thr2:.0f} prior mkt-wide signals)', b0, d2)
    results.append(r)
    r['_kept'].assign(date=lambda x: x.day, pnl_R=lambda x: x.net_R).to_csv(f'{outdir}/trades/D2.csv', index=False)

    d3 = mask_d3(b0)
    r = score_cohort_cell('D3', 'exclude entries 12:00-14:00 ET', b0, d3)
    results.append(r)
    r['_kept'].assign(date=lambda x: x.day, pnl_R=lambda x: x.net_R).to_csv(f'{outdir}/trades/D3.csv', index=False)

    d4 = mask_d4(b0, hmm)
    r = score_cohort_cell('D4', 'HMM calm state (hmm_state==0) on the day', b0, d4)
    results.append(r)
    r['_kept'].assign(date=lambda x: x.day, pnl_R=lambda x: x.net_R).to_csv(f'{outdir}/trades/D4.csv', index=False)

    d5 = mask_d5(b0, feat)
    r = score_cohort_cell('D5', 'first signal of day per symbol only (n_prior==0)', b0, d5)
    results.append(r)
    r['_kept'].assign(date=lambda x: x.day, pnl_R=lambda x: x.net_R).to_csv(f'{outdir}/trades/D5.csv', index=False)

    w1 = mask_w1(b0)
    r = score_cohort_cell('W1', 'week gate: trade wk t only if wk t-1 B0 net R > 0', b0, w1)
    results.append(r)
    r['_kept'].assign(date=lambda x: x.day, pnl_R=lambda x: x.net_R).to_csv(f'{outdir}/trades/W1.csv', index=False)

    w2 = mask_w2(b0)
    r = score_cohort_cell('W2', 'daily kill: no new entries after day realized R <= -3', b0, w2)
    results.append(r)
    r['_kept'].assign(date=lambda x: x.day, pnl_R=lambda x: x.net_R).to_csv(f'{outdir}/trades/W2.csv', index=False)

    # B0 itself, for cadence_bar reference
    b0.assign(date=b0.day, pnl_R=b0.net_R).to_csv(f'{outdir}/trades/B0.csv', index=False)

    # ---------------- cadence bar ----------------
    log('running cadence_bar.py on B0 and every passing cell (VAL split)...')
    cadence = {}
    to_run = ['B0'] + [r['id'] for r in results if r.get('pass')]
    for cid in to_run:
        csvp = f'{outdir}/trades/{cid}.csv'
        if not os.path.exists(csvp):
            continue
        try:
            p = subprocess.run(['python3', CADENCE, '--trades', csvp, '--split', 'VAL'],
                                cwd=ROOT, capture_output=True, text=True, timeout=120)
            cadence[cid] = dict(returncode=p.returncode, stdout=p.stdout[-4000:], stderr=p.stderr[-2000:])
            log(f'  cadence[{cid}]: rc={p.returncode}')
        except Exception as e:
            cadence[cid] = dict(error=str(e))
            log(f'  cadence[{cid}]: ERROR {e}')

    # ---------------- report-only tables ----------------
    log('building report-only tables (day-of-week, time-of-day, entry-minute deciles)...')
    b0r = b0.copy()
    b0r['dow'] = pd.to_datetime(b0r.day).dt.day_name()
    dow_tab = b0r.groupby('dow').net_R.agg(['mean', 'count']).reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    HB_EDGES = [569, 585, 600, 660, 780, 960]
    HB_LAB = ['09:30-09:45', '09:45-10:00', '10:00-11:00', '11:00-13:00', '13:00+']
    b0r['tod'] = pd.cut(b0r.entry_m, HB_EDGES, labels=HB_LAB)
    tod_tab = b0r.groupby('tod', observed=True).net_R.agg(['mean', 'count'])
    b0r['em_decile'] = pd.qcut(b0r.entry_m, 10, duplicates='drop')
    dec_tab = b0r.groupby('em_decile', observed=True).net_R.agg(['mean', 'count'])

    # ---------------- write cells.json ----------------
    clean = []
    for r in results:
        rr = {k: v for k, v in r.items() if not k.startswith('_')}
        for k, v in list(rr.items()):
            if isinstance(v, (np.floating, np.integer)):
                rr[k] = float(v) if not pd.isna(v) else None
            elif isinstance(v, bool):
                rr[k] = bool(v)
            elif isinstance(v, float) and pd.isna(v):
                rr[k] = None
        clean.append(rr)
    with open(f'{outdir}/cells.json', 'w') as fh:
        json.dump(dict(cells=clean, cadence=cadence), fh, indent=2, default=str)
    log(f'wrote {outdir}/cells.json')

    # ---------------- write CELLS.md ----------------
    write_md(outdir, results, cadence, dow_tab, tod_tab, dec_tab, b0)
    log(f'wrote {outdir}/CELLS.md')
    log('DONE')


def write_md(outdir, results, cadence, dow_tab, tod_tab, dec_tab, b0):
    lines = ['# CELLS — HOD Exit Lab pass 1 (X1-X12, D1-D5, W1-W2)', '']
    lines.append(f'B0 population: {len(b0)} trades (TRAIN={ (b0.split=="TRAIN").sum() }, '
                 f'VAL={ (b0.split=="VAL").sum() }). B0 TRAIN mean net R = '
                 f'{b0[b0.split=="TRAIN"].net_R.mean():.4f}, VAL mean net R = '
                 f'{b0[b0.split=="VAL"].net_R.mean():.4f}.')
    lines.append('')
    lines.append('## Exit cells (X1-X12), paired ΔR vs B0')
    lines.append('| cell | rule | train ΔR | val ΔR | val t | H1 ΔR | H2 ΔR | ex-top5 ok | MDD ok | PASS |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for r in results:
        if r['kind'] != 'exit':
            continue
        lines.append(f"| {r['id']} | {r['name']} | {fmt(r.get('train_dR'))} | {fmt(r.get('val_dR'))} | "
                     f"{fmt(r.get('val_t_dR'))} | {fmt(r.get('h1_dR'))} | {fmt(r.get('h2_dR'))} | "
                     f"{r.get('extop5_ok')} | {r.get('mdd_ok')} | {'**PASS**' if r.get('pass') else 'fail'} |")
    lines.append('')
    lines.append('## Cohort cells (D1-D5, W1-W2), kept cohort vs full-B0-mean ΔR')
    lines.append('| cell | rule | train ΔR | val ΔR | val t | H1 ΔR | H2 ΔR | val fills/wk | dropped R (tr/val) | ex-top5 ok | MDD ok | PASS |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in results:
        if r['kind'] != 'cohort':
            continue
        lines.append(f"| {r['id']} | {r['name']} | {fmt(r.get('train_dR'))} | {fmt(r.get('val_dR'))} | "
                     f"{fmt(r.get('val_t_dR'))} | {fmt(r.get('h1_dR'))} | {fmt(r.get('h2_dR'))} | "
                     f"{fmt(r.get('val_fills_wk'))} | {fmt(r.get('dropped_R_train'))}/{fmt(r.get('dropped_R_val'))} | "
                     f"{r.get('extop5_ok')} | {r.get('mdd_ok')} | {'**PASS**' if r.get('pass') else 'fail'} |")
    lines.append('')
    lines.append('## Cadence bar (scripts/cadence_bar.py --split VAL), B0 + passing cells')
    for cid, c in cadence.items():
        lines.append(f'### {cid}')
        lines.append('```')
        lines.append((c.get('stdout') or c.get('error', ''))[-3000:])
        lines.append('```')
    lines.append('')
    lines.append('## Report-only: day-of-week (B0, all splits kept)')
    lines.append('| day | mean net R | n |')
    lines.append('|---|---|---|')
    for d, row in dow_tab.iterrows():
        lines.append(f"| {d} | {fmt(row['mean'])} | {int(row['count']) if pd.notna(row['count']) else 0} |")
    lines.append('')
    lines.append('## Report-only: time-of-day buckets')
    lines.append('| bucket | mean net R | n |')
    lines.append('|---|---|---|')
    for d, row in tod_tab.iterrows():
        lines.append(f"| {d} | {fmt(row['mean'])} | {int(row['count'])} |")
    lines.append('')
    lines.append('## Report-only: entry-minute deciles')
    lines.append('| decile (entry_m range) | mean net R | n |')
    lines.append('|---|---|---|')
    for d, row in dec_tab.iterrows():
        lines.append(f"| {d} | {fmt(row['mean'])} | {int(row['count'])} |")
    lines.append('')
    with open(f'{outdir}/CELLS.md', 'w') as fh:
        fh.write('\n'.join(lines))


def fmt(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 'NA'
    return f'{v:.3f}' if isinstance(v, float) else str(v)


if __name__ == '__main__':
    main()
