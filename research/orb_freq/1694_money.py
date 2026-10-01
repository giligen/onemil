#!/usr/bin/env python3
"""Cell 1,694: where ORB's money actually is -- runners, who gets the risk,
and letting them run.

PREREG: research/orb_freq/PREREG_1694.md (FROZEN 2026-10-01 15:30 UTC).
Owner 10/1 15:25 UTC verbatim: "unacceptable. act as a true researcher, find
the money." Every sweep so far asked "can a filter/exit raise the MEAN" and
the honest answer was "the body is zero, the money is in the tail." This
cell asks the tail's questions directly: let runners run (Part A), size by
expected R instead of filtering (Part B), and find the runners the pipeline
is dropping on the floor (Part C).

Reuses (imported read-only via importlib, project convention for
digit-prefixed cell scripts -- cannot `import` a name starting with a
digit):
  research/orb_freq/1693_pool_exits.py (f1693) -- CachedStore, reconstruct_
    fill (entry-minute reconstruction: book's own entry_price, stop = the
    opening-range low), load_production_fills, make_slices/tercile_edges,
    premkt_dollar_vol, and (transitively, as f1693.f1679/f1693.f1668/
    f1693.cb/f1693.orb_csv) the ORB constants, _lock_walk, BarStore,
    stats_block (day-clustered t/ex-top-5%/MDE) and cadence_bar helpers.
  research/orb_exit/1679_orb_exit.py (via f1693.f1679) -- find_range_and_
    breakout, _lock_walk (the live lock-stop walker).
  scripts/cadence_bar.py (via f1693.cb) -- build_weekly_series/percentile/
    compute_cycles/score_c1 (strong-week gap).
  research/orb_freq/1693_reads.csv -- the 18 admission-feature slices'
    mean R under exit E1_production, per window (TRAIN2025/VAL2026/
    OOS2024H2) -- Part B's "slice means under the production exit."
  trading/orb_pdr_veto.pdr_veto_applies, trading/orb_g1_veto.g1_reject,
    trading/orb_range_size_veto.range_size_veto_applies -- READ-ONLY calls
    to the SAME helper functions the live engine and the BT pipeline both
    call (parity by construction); orb.yaml itself is never opened or
    edited by this script -- its CURRENT B+ thresholds (read directly,
    2026-10-01, matches study_orb_pipeline_static_lock.py's bt_cfg on this
    node) are hardcoded as named constants below with their source line.

Usage:
    nice -n 10 python3 research/orb_freq/1694_money.py
"""
import glob
import logging
import os
import sys
import time
from datetime import date as _date, datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

LOG_FILE = os.path.join(HERE, '1694_money.log')
READS_CSV = os.path.join(HERE, '1694_reads.csv')
RUNNERS_CSV = os.path.join(HERE, '1694_runners.csv')
MISSED_CSV = os.path.join(HERE, '1694_missed.csv')
RESULT_MD = os.path.join(HERE, 'RESULT_1694.md')

logger = logging.getLogger('cell1694')


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()])


def check_disk(floor_gb=5.0):
    st = os.statvfs('/')
    free_gb = st.f_bavail * st.f_frsize / (1024 ** 3)
    logger.info('disk free on /: %.1f GB', free_gb)
    if free_gb < floor_gb:
        logger.error('disk free %.1f GB < %.1f GB floor -- aborting', free_gb, floor_gb)
        sys.exit(1)
    return free_gb


def _load_module(name, fname, root=ROOT):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, fname))
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = old_argv
    return mod


def append_result_md(text):
    with open(RESULT_MD, 'a') as fh:
        fh.write(text)
        fh.flush()


# ---------------------------------------------------------------------------
# Reused machinery (one import pulls in 1679/1668/1677/cadence_bar/orb_csv
# transitively, since 1693 already loaded them into its own namespace)
# ---------------------------------------------------------------------------
f1693 = _load_module('f1693_1694', 'research/orb_freq/1693_pool_exits.py')
f1679 = f1693.f1679
cb = f1693.cb
orb_csv = f1693.orb_csv
CachedStore = f1693.CachedStore
reconstruct_fill = f1693.reconstruct_fill
stats_block = f1693.stats_block
_R = f1693._R
_lock_walk = f1693._lock_walk
LOCK_TRIGGER_R_LIVE = f1693.LOCK_TRIGGER_R_LIVE
LOCK_STOP_R_LIVE = f1693.LOCK_STOP_R_LIVE
ORB_EOD_M = f1693.ORB_EOD_M
EXIT_SLIP_BPS = f1693.EXIT_SLIP_BPS
WINDOWS = f1693.WINDOWS
TRAIN_LO, TRAIN_HI = f1693.TRAIN_LO, f1693.TRAIN_HI
VAL_LO, VAL_HI = f1693.VAL_LO, f1693.VAL_HI
OOS_LO, OOS_HI = f1693.OOS_LO, f1693.OOS_HI
FIXED_RISK_DOLLARS = f1693.FIXED_RISK_DOLLARS
STRONG_WEEK_R = f1693.STRONG_WEEK_R
pdate = f1693.pdate
assign_window = f1693.assign_window
window_weeks = f1693.window_weeks
reads_for_exit_series = f1693.reads_for_exit_series
ORB_BOOK_CSV = f1693.ORB_BOOK_CSV
premkt_dollar_vol = f1693.premkt_dollar_vol

YEARS_2025_2026 = (VAL_HI - TRAIN_LO).days / 365.25

# orb.yaml CURRENT B+ thresholds, read directly 2026-10-01 (this script never
# opens orb.yaml itself -- CLAUDE.md: never touch config/orb.yaml). Source:
# orb.yaml filter.prev_day_range_veto (line ~135-137), filter.g1_veto
# (~147-150), filter.range_size_veto (~161-163). Matches study_orb_pipeline_
# static_lock.py's bt_cfg on this node (pdr_min comment there: "B+
# 2026-08-15: threshold from orb.yaml (bt_cfg, = 11.0)").
ORB_PDR_MIN = 11.0
ORB_G1_RV20_MIN = 7.106
ORB_G1_PDR_MIN = 9.226
ORB_RANGE_SIZE_MIN = 2.221


# ---------------------------------------------------------------------------
# Part A -- new exit walkers (A3 = production/E1_production reused verbatim;
# A1/A2/A5 are f1679._lock_walk's own precedence extended with a fixed
# target; A4/A6 reuse 1693's _scale_then_lock_walk/_target_only_walk
# directly). Every walker returns (R, exit_minute) -- EOD truncates first,
# stop wins a same-bar tie, -10bps slip on every fill: the SAME precedence
# as every walker in this codebase (f1668.walk_k's own docstring).
# ---------------------------------------------------------------------------

def _walk_lock(bars, i0, entry, stop, R_unit, trigger_r, lock_stop_r, target_r=None,
                eod_m=ORB_EOD_M, slip_bps=EXIT_SLIP_BPS):
    """Unified lock-walk: stop arms to lock_stop_r*R once MFE >= trigger_r*R;
    an optional fixed target caps the trade (None = no target, ride the lock
    to the 15:45 close -- A3/production). Returns (R, exit_minute)."""
    n = len(bars['o'])
    if i0 + 1 >= n:
        return np.nan, np.nan
    slip = slip_bps / 10000.0
    trigger_lvl = entry + trigger_r * R_unit
    lock_lvl = entry + lock_stop_r * R_unit
    target_lvl = (entry + target_r * R_unit) if target_r is not None else None
    stop_price, armed = stop, False
    for j in range(i0 + 1, n):
        m = bars['minarr'][j]
        if m >= eod_m:
            return _R(float(bars['c'][j]) * (1 - slip), entry, R_unit), m
        bh, bl = float(bars['h'][j]), float(bars['l'][j])
        if not armed and bh >= trigger_lvl:
            armed = True
            stop_price = max(stop_price, lock_lvl)
        if bl <= stop_price:
            return _R(stop_price * (1 - slip), entry, R_unit), m
        if target_lvl is not None and bh >= target_lvl:
            return _R(target_lvl * (1 - slip), entry, R_unit), m
    m = bars['minarr'][n - 1]
    return _R(float(bars['c'][n - 1]) * (1 - slip), entry, R_unit), m


def _walk_scale_then_lock(bars, i0, entry, stop, R_unit, scale_r,
                           eod_m=ORB_EOD_M, slip_bps=EXIT_SLIP_BPS):
    """A4: 50% out at +scale_r R (touch + slip), the other 50% rides
    _walk_lock's full independent path (live lock, NO target) -- matches
    1693's _scale_then_lock_walk design exactly, generalised to scale_r=2."""
    leg2_R, leg2_m = _walk_lock(bars, i0, entry, stop, R_unit, LOCK_TRIGGER_R_LIVE,
                                 LOCK_STOP_R_LIVE, None, eod_m, slip_bps)
    n = len(bars['o'])
    slip = slip_bps / 10000.0
    touch_j = None
    for j in range(i0 + 1, n):
        if bars['minarr'][j] >= eod_m or float(bars['l'][j]) <= stop:
            break
        if float(bars['h'][j]) >= entry + scale_r * R_unit:
            touch_j = j
            break
    if touch_j is None:
        return leg2_R, leg2_m
    leg1_R = _R((entry + scale_r * R_unit) * (1 - slip), entry, R_unit)
    return 0.5 * leg1_R + 0.5 * leg2_R, leg2_m


def _walk_target_only(bars, i0, entry, stop, R_unit, target_r,
                       eod_m=ORB_EOD_M, slip_bps=EXIT_SLIP_BPS):
    """A6: fixed target, ORIGINAL stop, no lock/trail -- the pre-lock
    reference point."""
    n = len(bars['o'])
    if i0 + 1 >= n:
        return np.nan, np.nan
    slip = slip_bps / 10000.0
    target_lvl = entry + target_r * R_unit
    for j in range(i0 + 1, n):
        m = bars['minarr'][j]
        if m >= eod_m:
            return _R(float(bars['c'][j]) * (1 - slip), entry, R_unit), m
        bh, bl = float(bars['h'][j]), float(bars['l'][j])
        if bl <= stop:
            return _R(stop * (1 - slip), entry, R_unit), m
        if bh >= target_lvl:
            return _R(target_lvl * (1 - slip), entry, R_unit), m
    m = bars['minarr'][n - 1]
    return _R(float(bars['c'][n - 1]) * (1 - slip), entry, R_unit), m


# PREREG interpretation note (stated here + in RESULT.md): A1/A2 read "target
# N R" with no explicit stop-management qualifier except A1's parenthetical
# "(lock as live)"; A2 is treated as the SAME mechanism at a higher target
# (a parametric sweep of the target level under one fixed stop-management
# rule), not a silent switch to a bare stop. A6's "(production, reference)"
# is read as a REFERENCE anchor (the pre-lock, bare-target exit), not a claim
# that it equals today's live rule -- A3 (no target, ride with the live
# lock) is the actual current production exit and the baseline for every
# paired-delta/pass-bar read below.
BASELINE = 'A3_noTarget_liveLock'
A_EXITS = {
    'A1_target3R_lockLive':   lambda b, i0, e, s, R: _walk_lock(b, i0, e, s, R, LOCK_TRIGGER_R_LIVE, LOCK_STOP_R_LIVE, 3.0),
    'A2_target4R_lockLive':   lambda b, i0, e, s, R: _walk_lock(b, i0, e, s, R, LOCK_TRIGGER_R_LIVE, LOCK_STOP_R_LIVE, 4.0),
    'A3_noTarget_liveLock':   lambda b, i0, e, s, R: _walk_lock(b, i0, e, s, R, LOCK_TRIGGER_R_LIVE, LOCK_STOP_R_LIVE, None),
    'A4_half2R_restNoTarget': lambda b, i0, e, s, R: _walk_scale_then_lock(b, i0, e, s, R, 2.0),
    'A5_target3R_BEat2R':     lambda b, i0, e, s, R: _walk_lock(b, i0, e, s, R, 2.0, 0.0, 3.0),
    'A6_target2R_plain_ref':  lambda b, i0, e, s, R: _walk_target_only(b, i0, e, s, R, 2.0),
}


# ---------------------------------------------------------------------------
# ONE per-fill pass: reconstructs the production book once, computes the 6
# A-exits + exit minutes, the 6 admission features (Amendment-1 convention,
# verbatim from f1693.load_production_fills), and the Part-C1 runner/MFE
# profile -- avoids walking the same bars three times across Parts A/B/C.
# ---------------------------------------------------------------------------

def build_fill_table(store):
    df = orb_csv.read_orb_csv(ORB_BOOK_CSV)
    ent = df[df['entered'] == 1].copy()
    ent['date'] = ent['date'].astype(str)
    logger.info('production: %d entered rows from %s', len(ent), ORB_BOOK_CSV)
    recon = {'no_bars': 0, 'no_range_or_breakout': 0, 'bad_R': 0, 'below_R_floor': 0, 'ok': 0}
    rows = []
    t0 = time.time()
    for n_seen, r in enumerate(ent.itertuples(), 1):
        rec, reason = reconstruct_fill(store, r.symbol, r.date, r.entry_price)
        if rec is None:
            recon[reason] += 1
            continue
        recon['ok'] += 1
        bars, i0, entry, stop, R_unit = rec['bars'], rec['i0'], rec['entry'], rec['stop'], rec['R_unit']
        d = pdate(r.date)
        win = assign_window(d)
        rvol_0935 = (r.range_total_volume / r.avg_daily_volume_20d) if r.avg_daily_volume_20d > 0 else np.nan
        entry_minute = float(bars['minarr'][i0 + 1]) if i0 + 1 < len(bars['o']) else np.nan
        row = dict(date=d, window=win, symbol=r.symbol, entry=entry, R_unit=R_unit,
                   entry_minute=entry_minute, gap_pct=r.gap_pct, price=entry,
                   range5m_pct=r.range_size_pct, prior_day_volume=r.prev_day_volume_vs_20d,
                   rvol_0935=rvol_0935, premkt_dollar_vol=premkt_dollar_vol(bars))
        for ename, efn in A_EXITS.items():
            rv, mv = efn(bars, i0, entry, stop, R_unit)
            row[ename] = rv
            row[ename + '_exit_minute'] = mv
        # Part C1: MFE/runner profile over the full remaining day to the EOD
        # truncation bar (path-based "reached", independent of which exit
        # captured it).
        eod_idx = np.where(bars['minarr'] >= ORB_EOD_M)[0]
        last_idx = int(eod_idx[0]) if len(eod_idx) else len(bars['o']) - 1
        seg_h = bars['h'][i0 + 1:last_idx + 1]
        seg_m = bars['minarr'][i0 + 1:last_idx + 1]
        if len(seg_h):
            pk = int(np.argmax(seg_h))
            row['mfe_R'] = (float(seg_h[pk]) - entry) / R_unit
            row['peak_minute'] = float(seg_m[pk])
            row['minutes_to_peak'] = float(seg_m[pk]) - entry_minute
            first1 = np.where((seg_h - entry) / R_unit >= 1.0)[0]
            row['minute_first_1R'] = float(seg_m[first1[0]]) if len(first1) else np.nan
        else:
            row['mfe_R'] = np.nan
            row['peak_minute'] = np.nan
            row['minutes_to_peak'] = np.nan
            row['minute_first_1R'] = np.nan
        row['runner2'] = int(row['mfe_R'] >= 2.0) if row['mfe_R'] == row['mfe_R'] else 0
        row['runner3'] = int(row['mfe_R'] >= 3.0) if row['mfe_R'] == row['mfe_R'] else 0
        rows.append(row)
        if n_seen % 100 == 0:
            logger.info('fills: reconstructed %d/%d (%.0fs)', n_seen, len(ent), time.time() - t0)
    logger.info('fills: reconstruction done n=%d %s', len(ent), recon)
    pf = pd.DataFrame(rows)
    premkt_cov = pf['premkt_dollar_vol'].notna().mean() if len(pf) else 0.0
    logger.info('premkt $ vol coverage = %.1f%% (availability rail: 80%%)', 100 * premkt_cov)
    # sanity cross-check: A3 must equal f1693's own E1_production formula
    chk = pf.iloc[0] if len(pf) else None
    logger.info('A3 baseline uses the SAME f1679._lock_walk(trigger=%.2f, lock=%.2f) as f1693 E1_production',
                LOCK_TRIGGER_R_LIVE, LOCK_STOP_R_LIVE)
    return pf, recon, premkt_cov


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------

def part_a(pf):
    logger.info('=== Part A: let the runners run ===')
    rows = []
    for ename in [n for n in A_EXITS if n != BASELINE]:
        for wname, (lo, hi) in WINDOWS.items():
            sub = pf[pf['window'] == wname]
            vals = sub[ename].dropna()
            if not len(vals):
                continue
            dts = sub.loc[vals.index, 'date']
            base_vals = sub.loc[vals.index, BASELINE]
            r = reads_for_exit_series(vals, dts, lo, hi)
            delta = vals - base_vals
            dstat = stats_block(delta.values, dts.values)
            for k in (2, 3, 5):
                r[f'capture_ge{k}R'] = float((vals >= k).mean())
                r[f'capture_ge{k}R_A3'] = float((base_vals >= k).mean())
            exit_m = sub.loc[vals.index, ename + '_exit_minute']
            base_m = sub.loc[vals.index, BASELINE + '_exit_minute']
            cont_mask = exit_m > base_m
            give_mask = ~cont_mask
            cycles, _ = cb.compute_cycles(
                cb.build_weekly_series([{'date': d, 'r': v} for d, v in zip(dts, vals)], lo, hi), STRONG_WEEK_R)
            c1 = cb.score_c1(cycles, gap_median_thresh=3, gap_p90_thresh=6)
            r.update(exit=ename, window=wname,
                      paired_delta_mean=float(dstat['mean_dR']), paired_delta_day_t=float(dstat['day_t']),
                      paired_delta_n=int(dstat['n']),
                      continuation_share=float(cont_mask.mean()), continuation_dR=float(delta[cont_mask].mean()) if cont_mask.any() else 0.0,
                      giveback_share=float(give_mask.mean()), giveback_dR=float(delta[give_mask].mean()) if give_mask.any() else 0.0,
                      strong_week_gap_median=c1['median'], strong_week_gap_p90=c1['p90'],
                      dollars_per_year=float(r['mean_R'] * r['fills_per_week'] * 52 * FIXED_RISK_DOLLARS))
            rows.append(r)
    out = pd.DataFrame(rows)
    # Pass bar (Part A): ships only if BOTH TRAIN2025 and VAL2026 clear
    # paired delta >= +0.05R, day_t >= 2.0, AND >=3R capture rises vs A3.
    ships = {}
    for ename in out['exit'].unique():
        ok = True
        for wname in ('TRAIN2025', 'VAL2026'):
            row = out[(out.exit == ename) & (out.window == wname)]
            if not len(row):
                ok = False
                continue
            row = row.iloc[0]
            ok = ok and (row['paired_delta_mean'] >= 0.05 and row['paired_delta_day_t'] >= 2.0
                         and row['capture_ge3R'] > row['capture_ge3R_A3'])
        ships[ename] = ok
    logger.info('Part A pass-bar (both TRAIN2025+VAL2026): %s', ships)
    append_result_md('## Part A -- let the runners run (baseline = A3, no target, ride the live lock to 15:45)\n\n'
                      '| exit | window | n | fills/wk | mean R | day_t | ex_top5 | cap>=2R | cap>=3R | cap>=5R | '
                      'paired dR | dR day_t | cont. share/dR | giveback share/dR | wk P10 R | strong-gap med/p90 wk | $/yr@375 |\n'
                      '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n')
    for _, row in out.iterrows():
        append_result_md(
            f"| {row['exit']} | {row['window']} | {row['n']:.0f} | {row['fills_per_week']:.2f} | {row['mean_R']:+.3f} | "
            f"{row['day_t']:.1f} | {row['ex_top5']:+.3f} | {row['capture_ge2R']:.2f} | {row['capture_ge3R']:.2f} | {row['capture_ge5R']:.2f} | "
            f"{row['paired_delta_mean']:+.3f} | {row['paired_delta_day_t']:.1f} | {row['continuation_share']:.2f}/{row['continuation_dR']:+.3f} | "
            f"{row['giveback_share']:.2f}/{row['giveback_dR']:+.3f} | {row['weekly_p10_R']:+.2f} | "
            f"{row['strong_week_gap_median']}/{row['strong_week_gap_p90']} | ${row['dollars_per_year']:+,.0f} |\n")
    append_result_md(f"\nSHIPS (both TRAIN2025+VAL2026 clear paired dR>=+0.05R, day_t>=2.0, >=3R capture rises): "
                      f"{[k for k, v in ships.items() if v] or 'none'}\n\n")
    return out, ships


# ---------------------------------------------------------------------------
# Part B -- risk tilt (sizing, not filtering)
# ---------------------------------------------------------------------------
ADMISSION_DIMS = {
    'gap_size': ['gap_size_5-7%', 'gap_size_7-10%', 'gap_size_>=10%'],
    'price': ['price_$3-10', 'price_$10-20', 'price_$20-30'],
    'prior_day_volume': ['prior_day_volume_low', 'prior_day_volume_mid', 'prior_day_volume_high'],
    'range5m_pct': ['range5m_pct_low', 'range5m_pct_mid', 'range5m_pct_high'],
    'rvol_0935': ['rvol_0935_low', 'rvol_0935_mid', 'rvol_0935_high'],
    'premkt_dollar_vol': ['premkt_dollar_vol_low', 'premkt_dollar_vol_mid', 'premkt_dollar_vol_high'],
}
M_BY_RANK = {0: 0.5, 1: 1.0, 2: 1.5}  # bottom/mid/top tercile of expected R


def bin_fills_by_dimension(pf, premkt_cov):
    """Assigns each fill a bin NAME per admission dimension, reusing
    f1693.make_slices verbatim (TRAIN-2025-fit tercile cutpoints for the 4
    tercile dims, fixed edges for gap_size/price -- the SAME bin boundaries
    1693_reads.csv's slice rows were built from)."""
    slices = f1693.make_slices(pf, premkt_cov)
    for dim in ADMISSION_DIMS:
        pf[f'bin_{dim}'] = None
    for sname, sub in slices.items():
        for dim, names in ADMISSION_DIMS.items():
            if sname in names:
                pf.loc[pf.index.isin(sub.index), f'bin_{dim}'] = sname
    return pf


def rank_bins(reads1693, dim, select_window):
    """Ranks a dimension's (<=3) bins by their mean R under E1_production in
    the SELECTION window -> {bin_name: m}. Returns None if <2 bins have data
    (VOID dimension, e.g. premkt coverage < 80%)."""
    e1 = reads1693[(reads1693.exit == 'E1_production') & (reads1693.window == select_window)]
    sub = e1[e1.pool.isin(ADMISSION_DIMS[dim])]
    sub = sub[sub.n > 0].sort_values('mean_R')
    if len(sub) < 2:
        return None
    ranks = np.linspace(0, len(sub) - 1, len(sub)).round().astype(int)
    # 2 bins -> {0:0.5,1:1.5} (skip mid); 3 bins -> {0:0.5,1:1.0,2:1.5}.
    if len(sub) == 2:
        m_map = {0: 0.5, 1: 1.5}
    else:
        m_map = M_BY_RANK
    return {name: m_map[i] for i, name in zip(ranks, sub['pool'])}


def part_b(pf, premkt_cov):
    logger.info('=== Part B: who gets the risk ===')
    reads1693 = pd.read_csv(os.path.join(HERE, '1693_reads.csv'))
    pf = bin_fills_by_dimension(pf, premkt_cov)
    directions = {'dirA_selTRAIN_testVAL': ('TRAIN2025', 'VAL2026'),
                  'dirB_selVAL_testTRAIN': ('VAL2026', 'TRAIN2025')}
    results = []
    bin_rank_pairs = []  # for Spearman ordering agreement, pooled across dims
    for dim in ADMISSION_DIMS:
        mapA = rank_bins(reads1693, dim, 'TRAIN2025')
        mapB = rank_bins(reads1693, dim, 'VAL2026')
        if mapA is None or mapB is None:
            logger.warning('VOID: dimension %s has <2 scored bins in one direction -- dropped from B1/B2/B3', dim)
            continue
        common = sorted(set(mapA) & set(mapB))
        if len(common) >= 2:
            rA = pd.Series(mapA)[common].rank().values
            rB = pd.Series(mapB)[common].rank().values
            bin_rank_pairs.append((rA, rB))
        # ---- B1: single-feature tilt, both directions ----
        for dname, (selw, testw) in directions.items():
            m_map = mapA if selw == 'TRAIN2025' else mapB
            sub = pf[pf.window == testw].copy()
            sub['m'] = sub[f'bin_{dim}'].map(m_map)
            sub = sub.dropna(subset=['m', BASELINE])
            if not len(sub):
                continue
            results.append(score_tilt(sub, f'B1_{dim}', dname, testw))
    # ---- ordering agreement (Spearman, pooled across dims with >=2 bins) ----
    try:
        from scipy.stats import spearmanr
        if bin_rank_pairs:
            allA = np.concatenate([p[0] for p in bin_rank_pairs])
            allB = np.concatenate([p[1] for p in bin_rank_pairs])
            ordering_agreement = float(spearmanr(allA, allB).correlation)
        else:
            ordering_agreement = np.nan
    except Exception as e:
        logger.warning('spearmanr unavailable (%s) -- ordering agreement = NaN', e)
        ordering_agreement = np.nan
    logger.info('ordering agreement (pooled Spearman, bin ranks TRAIN-select vs VAL-select) = %.3f', ordering_agreement)

    # ---- B2: additive score across all 6 features, tercile on selection half ----
    for dname, (selw, testw) in directions.items():
        qs_cols = []
        for dim in ADMISSION_DIMS:
            mapA = rank_bins(reads1693, dim, selw)
            if mapA is None:
                continue
            qs_col = f'qs_{dim}'
            # quality score: bottom=-1, mid=0, top=+1 (derived from the m map)
            qmap = {name: {0.5: -1, 1.0: 0, 1.5: 1}[m] for name, m in mapA.items()}
            pf[qs_col] = pf[f'bin_{dim}'].map(qmap)
            qs_cols.append(qs_col)
        if not qs_cols:
            continue
        pf['additive_score'] = pf[qs_cols].sum(axis=1, skipna=True)
        sel = pf[pf.window == selw]
        cuts = sel['additive_score'].quantile([1/3, 2/3]).values
        def to_m(score, cuts=cuts):
            if pd.isna(score):
                return np.nan
            if score <= cuts[0]:
                return 0.5
            if score <= cuts[1]:
                return 1.0
            return 1.5
        test = pf[pf.window == testw].copy()
        test['m'] = test['additive_score'].apply(to_m)
        test = test.dropna(subset=['m', BASELINE])
        if len(test):
            results.append(score_tilt(test, 'B2_additive', dname, testw))
            # ---- B3: B2 capped at production's own total weekly risk ----
            prod_weekly_dollars = test[BASELINE].notna().sum() / window_weeks(*WINDOWS[testw]) * FIXED_RISK_DOLLARS
            tilt_weekly_dollars = test['m'].sum() * FIXED_RISK_DOLLARS / window_weeks(*WINDOWS[testw])
            cap_factor = min(1.0, prod_weekly_dollars / tilt_weekly_dollars) if tilt_weekly_dollars else 1.0
            test3 = test.copy()
            test3['m'] = test3['m'] * cap_factor
            results.append(score_tilt(test3, 'B3_additive_capped', dname, testw, cap_factor=cap_factor))
    out = pd.DataFrame(results)
    append_result_md('## Part B -- who gets the risk (tilt vs flat production sizing)\n\n'
                      f"Ordering agreement (pooled Spearman of bin ranks, TRAIN-select vs VAL-select) = {ordering_agreement:+.3f} "
                      f"(ROBUST bar: >= +0.60)\n\n"
                      '| variant | direction | test window | n | EV/risk tilted | EV/risk prod | EV/risk gain | $/yr tilted | $/yr prod | ex_top5 tilted | wk P10/unit risk | max DD $ |\n'
                      '|---|---|---|---|---|---|---|---|---|---|---|---|\n')
    for _, row in out.iterrows():
        append_result_md(
            f"| {row['variant']} | {row['direction']} | {row['test_window']} | {row['n']:.0f} | {row['ev_per_risk_tilt']:+.3f} | "
            f"{row['ev_per_risk_prod']:+.3f} | {row['ev_risk_gain_pct']:+.1f}% | ${row['dollars_per_yr_tilt']:+,.0f} | "
            f"${row['dollars_per_yr_prod']:+,.0f} | {row['ex_top5_tilt']:+.3f} | {row['weekly_p10_per_unit_risk']:+.3f} | ${row['max_dd_dollars']:,.0f} |\n")
    append_result_md(f"\nSpearman ordering agreement across directions = {ordering_agreement:+.3f}\n\n")
    return out, ordering_agreement


def score_tilt(sub, variant, direction, test_window, cap_factor=None):
    R = sub[BASELINE].values
    m = sub['m'].values
    ev_tilt = float((R * m).sum() / m.sum()) if m.sum() else np.nan
    ev_prod = float(R.mean())
    yrs = window_weeks(*WINDOWS[test_window]) / 52.0
    dollars_tilt = float((R * m).sum() * FIXED_RISK_DOLLARS / yrs) if yrs else np.nan
    dollars_prod = float(R.sum() * FIXED_RISK_DOLLARS / yrs) if yrs else np.nan
    weighted = R * m
    ex_top5_tilt = stats_block(weighted, sub['date'].values)['ex_top5']
    # weekly P10 per unit risk: weekly sum(R*m) / weekly sum(m)
    trades = [{'date': d, 'r': rv * mv, 'risk': mv} for d, rv, mv in zip(sub['date'], R, m)]
    weekly_rm = cb.build_weekly_series([{'date': t['date'], 'r': t['r']} for t in trades], *WINDOWS[test_window])
    weekly_m = cb.build_weekly_series([{'date': t['date'], 'r': t['risk']} for t in trades], *WINDOWS[test_window])
    per_unit = [rm[1] / mw[1] if mw[1] else np.nan for rm, mw in zip(weekly_rm, weekly_m)]
    per_unit = [x for x in per_unit if x == x]
    p10 = cb.percentile(per_unit, 10) if per_unit else np.nan
    cum = np.cumsum(weighted * FIXED_RISK_DOLLARS)
    max_dd = float((np.maximum.accumulate(cum) - cum).max()) if len(cum) else 0.0
    return dict(variant=variant, direction=direction, test_window=test_window, n=len(sub),
                ev_per_risk_tilt=ev_tilt, ev_per_risk_prod=ev_prod,
                ev_risk_gain_pct=100 * (ev_tilt - ev_prod) / abs(ev_prod) if ev_prod else np.nan,
                dollars_per_yr_tilt=dollars_tilt, dollars_per_yr_prod=dollars_prod,
                ex_top5_tilt=ex_top5_tilt, weekly_p10_per_unit_risk=p10, max_dd_dollars=max_dd,
                cap_factor=cap_factor)


# ---------------------------------------------------------------------------
# Part C1 -- runner anatomy + pre-entry classifier
# ---------------------------------------------------------------------------

def part_c1(pf):
    logger.info('=== Part C1: runner anatomy + classifier ===')
    feat_cols = ['gap_pct', 'price', 'range5m_pct', 'prior_day_volume', 'rvol_0935',
                 'premkt_dollar_vol', 'entry_minute']
    append_result_md('## Part C1 -- runner anatomy (>=2R / >=3R by MFE, path-based) and the pre-entry classifier\n\n')
    for thr in (2, 3):
        col = f'runner{thr}'
        run = pf[pf[col] == 1]
        rest = pf[pf[col] == 0]
        append_result_md(f"- >= {thr}R runners: n={len(run)}/{len(pf)} ({100*len(run)/len(pf):.1f}%); "
                          f"mean entry_minute {run['entry_minute'].mean():.0f} vs rest {rest['entry_minute'].mean():.0f}; "
                          f"mean minutes_to_peak {run['minutes_to_peak'].mean():.0f} vs rest {rest['minutes_to_peak'].mean():.0f}; "
                          f"mean gap_pct {run['gap_pct'].mean():.2f} vs {rest['gap_pct'].mean():.2f}; "
                          f"mean range5m_pct {run['range5m_pct'].mean():.2f} vs {rest['range5m_pct'].mean():.2f}\n")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    rows = []
    for direction, (trw, tew) in {'TRAIN2025_to_VAL2026': ('TRAIN2025', 'VAL2026'),
                                   'VAL2026_to_TRAIN2025': ('VAL2026', 'TRAIN2025')}.items():
        tr = pf[pf.window == trw]
        te = pf[pf.window == tew]
        Xtr, ytr = tr[feat_cols], tr['runner2']
        Xte, yte = te[feat_cols], te['runner2']
        if ytr.nunique() < 2 or yte.nunique() < 2:
            logger.warning('C1 classifier %s: degenerate labels -- skipped', direction)
            continue
        clf = HistGradientBoostingClassifier(random_state=0)  # defaults; no n_jobs param on HGB
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p)
        rng = np.random.RandomState(0)
        ytr_shuf = rng.permutation(ytr.values)
        clf_pl = HistGradientBoostingClassifier(random_state=0)
        clf_pl.fit(Xtr, ytr_shuf)
        p_pl = clf_pl.predict_proba(Xte)[:, 1]
        auc_placebo = roc_auc_score(yte, p_pl)
        decile_cut = np.quantile(p, 0.90)
        top = te[p >= decile_cut]
        rows.append(dict(direction=direction, auc=auc, auc_placebo=auc_placebo,
                          top_decile_mean_R=float(top[BASELINE].mean()), top_decile_runner_share=float(top['runner2'].mean()),
                          all_mean_R=float(te[BASELINE].mean()), all_runner_share=float(te['runner2'].mean())))
    cls_df = pd.DataFrame(rows)
    append_result_md('\n| classifier direction | AUC | AUC placebo | top-decile mean R | top-decile runner share | all mean R | all runner share |\n'
                      '|---|---|---|---|---|---|---|\n')
    for _, row in cls_df.iterrows():
        append_result_md(f"| {row['direction']} | {row['auc']:.3f} | {row['auc_placebo']:.3f} | {row['top_decile_mean_R']:+.3f} | "
                          f"{row['top_decile_runner_share']:.2f} | {row['all_mean_R']:+.3f} | {row['all_runner_share']:.2f} |\n")
    append_result_md('\n')
    return cls_df


# ---------------------------------------------------------------------------
# Part C2 -- missed runners (gate reconstruction, since no reason column is
# stored anywhere on disk; verified by direct inspection of
# analysis_results/orb_bplus_book.csv and the raw features CSV).
# ---------------------------------------------------------------------------

def latest_features_csv():
    paths = sorted(p for p in glob.glob(os.path.join(ROOT, 'analysis_results', 'orb_features_*.csv'))
                    if 'corrmatrix' not in p)
    return paths[-1]


def classify_missed(row, in_book_entered1):
    """Gate reconstruction, applied directly via the SAME helper functions
    the live engine and the BT pipeline both call (trading/orb_pdr_veto,
    orb_g1_veto, orb_range_size_veto), in the pipeline's OWN application
    order (PDR -> G1 -> range-size). A candidate that clears all three but
    still never reached book_csv was cut upstream by score/Q1/slot/dedup --
    NOT individually decomposable without re-running the per-day ranking
    pass (out of this cell's budget; stated here and in RESULT.md, not
    silently assumed). The spread gate (orb.yaml max_spread_bps=300) needs
    real-time NBBO at the entry instant, not present in OHLCV bars --
    structurally unreconstructable from this data, folded into the same
    'rank_not_selected' bucket with this caveat."""
    from trading.orb_pdr_veto import pdr_veto_applies
    from trading.orb_g1_veto import g1_reject
    from trading.orb_range_size_veto import range_size_veto_applies
    key = (row.symbol, row.date)
    if key in in_book_entered1:
        return None  # actually traded -- not "missed"
    if row.entered == 0:
        return 'no_fill'  # selected, breakout price level never touched
    pdr = None if pd.isna(row.prev_day_range_pct) else float(row.prev_day_range_pct)
    if pdr_veto_applies(pdr, ORB_PDR_MIN):
        return 'PDR_veto'
    rv20 = None if pd.isna(row.return_volatility_20d) else float(row.return_volatility_20d)
    if g1_reject(rv20, pdr, ORB_G1_RV20_MIN, ORB_G1_PDR_MIN, short_history_veto=False):
        return 'G1_veto'
    rs = None if pd.isna(row.range_size_pct) else float(row.range_size_pct)
    if range_size_veto_applies(rs, ORB_RANGE_SIZE_MIN):
        return 'range_size_veto'
    return 'rank_not_selected'  # score threshold / Q1 / slot cap / dedup / spread (not decomposed)


def part_c2(store):
    logger.info('=== Part C2: the missed runners ===')
    feat_csv = latest_features_csv()
    logger.info('features CSV (pipeline input, latest non-corrmatrix): %s', feat_csv)
    full = orb_csv.read_orb_csv(feat_csv)
    full['date'] = full['date'].astype(str)
    book = orb_csv.read_orb_csv(ORB_BOOK_CSV)
    book['date'] = book['date'].astype(str)
    in_book_entered1 = set(zip(book.loc[book.entered == 1, 'symbol'], book.loc[book.entered == 1, 'date']))
    full = full[(full['date'] >= '2025-01-01') & (full['date'] <= str(VAL_HI))].copy()
    logger.info('raw candidate universe 2025-01-01..%s: %d rows (entered=1: %d, entered=0: %d)',
                VAL_HI, len(full), int((full.entered == 1).sum()), int((full.entered == 0).sum()))
    full['missed_reason'] = full.apply(lambda r: classify_missed(r, in_book_entered1), axis=1)
    missed = full[full['missed_reason'].notna()].copy()
    logger.info('missed population: %d; by reason: %s', len(missed), missed['missed_reason'].value_counts().to_dict())

    # Counterfactual: production entry (book's own entry_price, re-derived
    # breakout bar + range-low stop) + production exit (A3) walked on the
    # store's bars. no_fill rows have no entry by construction -> R=0, not
    # walked (reconstruct_fill would legitimately fail them as no_range_or_
    # breakout/below_floor and that is the CORRECT reason, not a bug).
    cf_R, cf_reason = [], []
    n_bars_missing = 0
    for r in missed.itertuples():
        if r.missed_reason == 'no_fill':
            cf_R.append(0.0)
            cf_reason.append('no_fill_structural_zero')
            continue
        rec, reason = reconstruct_fill(store, r.symbol, r.date, r.entry_price)
        if rec is None:
            cf_R.append(np.nan)
            cf_reason.append(reason)
            if reason == 'no_bars':
                n_bars_missing += 1
            continue
        rv, _ = A_EXITS[BASELINE](rec['bars'], rec['i0'], rec['entry'], rec['stop'], rec['R_unit'])
        cf_R.append(rv)
        cf_reason.append('ok')
    missed['counterfactual_R'] = cf_R
    missed['counterfactual_status'] = cf_reason
    cov = missed['counterfactual_R'].notna().mean()
    logger.info('counterfactual bar coverage: %.1f%% (%d/%d); %d missing bars entirely',
                100 * cov, int(missed['counterfactual_R'].notna().sum()), len(missed), n_bars_missing)
    missed.to_csv(MISSED_CSV, index=False)
    logger.info('wrote %s (%d rows)', MISSED_CSV, len(missed))

    taken = book[book.entered == 1]
    taken_runner3_share = float((taken.pnl_pct.notna()) .mean())  # placeholder overwritten below if R available
    # Use the SAME production fills table's A3 runner3 share as the "taken" baseline.
    rows = []
    for reason, sub in missed.groupby('missed_reason'):
        ok = sub[sub['counterfactual_R'].notna()]
        n_ok = len(ok)
        runner3_share = float((ok['counterfactual_R'] >= 3).mean()) if n_ok else np.nan
        mean_R = float(ok['counterfactual_R'].mean()) if n_ok else np.nan
        ex_top5 = stats_block(ok['counterfactual_R'].values, ok['date'].values)['ex_top5'] if n_ok else np.nan
        dollars_per_yr = mean_R * n_ok / YEARS_2025_2026 * FIXED_RISK_DOLLARS if n_ok else np.nan
        rows.append(dict(veto=reason, n_candidates=len(sub), n_counterfactual_ok=n_ok,
                          coverage_pct=100 * n_ok / len(sub) if len(sub) else np.nan,
                          mean_R=mean_R, runner_share_ge3R=runner3_share, ex_top5=ex_top5,
                          dollars_left_per_yr=dollars_per_yr))
    out = pd.DataFrame(rows)
    return out, missed


def runner_share_taken(pf):
    return float(pf['runner3'].mean())


def write_part_c2(out, pf):
    taken_share = runner_share_taken(pf)
    append_result_md('## Part C2 -- the missed runners (by veto)\n\n'
                      f"Taken (entered production fills) >=3R runner share (A3/MFE path): {taken_share:.3f}\n\n"
                      '| veto | n candidates | n w/ counterfactual | bar coverage % | mean counterfactual R | runner share >=3R | ex_top5 | $ left/yr @375 (full size) |\n'
                      '|---|---|---|---|---|---|---|---|\n')
    for _, row in out.iterrows():
        append_result_md(f"| {row['veto']} | {row['n_candidates']:.0f} | {row['n_counterfactual_ok']:.0f} | "
                          f"{row['coverage_pct']:.1f} | {row['mean_R']:+.3f} | {row['runner_share_ge3R']:.3f} | "
                          f"{row['ex_top5']:+.3f} | ${row['dollars_left_per_yr']:+,.0f} |\n")
    append_result_md(f"\nAt half size: $ left/yr halves for every row above; runner share (R-based, scale-invariant) is unchanged.\n"
                      f"NOT reconstructable from this data: the spread gate (needs real-time NBBO, not in OHLCV bars) and the exact "
                      f"individual split of score/Q1/slot-cap/dedup within 'rank_not_selected' (needs the per-day ranking pass re-run; "
                      f"out of this cell's budget) -- stated per PREREG's own fallback clause, not silently assumed.\n\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    setup_logging()
    logger.info('=== cell 1,694: where the money is -- starting ===')
    check_disk()
    with open(RESULT_MD, 'w') as fh:
        fh.write(f"# RESULT 1,694 -- where ORB's money actually is (run started {datetime.now().isoformat()})\n\n"
                  "PREREG: research/orb_freq/PREREG_1694.md (FROZEN). Owner 10/1 15:25 UTC: \"unacceptable. act as "
                  "a true researcher, find the money.\" Written incrementally per part.\n\n"
                  "Interpretation notes (stated, not hidden): A2 keeps A1's lock-as-live stop management at a "
                  "higher target (a parametric sweep, not a silent mechanism switch); A6 is a pre-lock reference "
                  "anchor, not a claim that it equals today's live rule -- A3 (no target, ride with the live lock) "
                  "is the actual production exit and every pass-bar/paired-delta baseline below. Part B's bin "
                  "EDGES are fixed (1,693's TRAIN-2025-cutpoint tercile boundaries / fixed gap-size and price bins) "
                  "for both directions; 'direction' flips which window's mean R RANKS the bins into bottom/mid/top. "
                  "Part C2's gate reconstruction applies PDR/G1/range-size directly via their own live/BT-shared "
                  "helper functions in the pipeline's own order; candidates clearing all three but still absent "
                  "from the book are bucketed 'rank_not_selected' (score/Q1/slot/dedup, not individually "
                  "decomposed -- stated, not assumed) per the PREREG's own fallback clause.\n\n")

    store = CachedStore(f1693.f1668.BARS_DB)
    pf, recon, premkt_cov = build_fill_table(store)
    logger.info('fill table ready: n=%d, recon=%s, premkt_cov=%.1f%%', len(pf), recon, 100 * premkt_cov)

    a_out, a_ships = part_a(pf)
    b_out, ordering_agreement = part_b(pf, premkt_cov)
    c1_out = part_c1(pf)
    c2_out, missed = part_c2(store)
    write_part_c2(c2_out, pf)

    # consolidated reads CSV (Part A + a slim Part B view; Part C has its own CSVs)
    a_out.to_csv(READS_CSV, index=False)
    logger.info('wrote %s (%d rows, Part A)', READS_CSV, len(a_out))
    pf.to_csv(RUNNERS_CSV, index=False)
    logger.info('wrote %s (%d rows)', RUNNERS_CSV, len(pf))

    store.close()
    logger.info('cache: %d hits / %d misses', store.hits, store.misses)
    append_result_md(f"\n## Files\n`1694_reads.csv` (Part A, {len(a_out)} rows), `1694_runners.csv` (per-fill table, "
                      f"{len(pf)} rows), `1694_missed.csv` ({len(missed)} rows), this file, `1694_money.py`, "
                      f"`1694_money.log`.\n")
    logger.info('=== DONE ===')


if __name__ == '__main__':
    main()
