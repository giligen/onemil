#!/usr/bin/env python3
"""Cell 1,699: the SECOND layer batch on the ORB stack (base -> RVOL tilt ->
add at +1R), each layer read ON TOP of the stack that precedes it.

PREREG: research/orb_freq/PREREG_1699.md (FROZEN 2026-10-01 18:30 UTC).
Owner 10/1: "think how hard I had to push... push harder." Same book, method,
directions and statistics as PREREG_1698.

Layers: L9 add-level sweep (+0.5/+1.0/+1.5R, one add; also 2 units at +1R),
L10 lock re-read on the COMBINED 2-unit position, L11 P1 pool (idea1) joined
as a frequency layer with its regime-specific half-out at +1R, L12 gapper-
count tilt, L13 adds on base and P1 at the chosen level, L14 compounding
equity curve with the above-water rule.

Reuses (imported read-only via importlib, unchanged -- project convention):
  research/orb_freq/1693_pool_exits.py (f1693) -- CachedStore, reconstruct_fill
    (f1679's own breakout/stop rule), _scale_then_lock_walk (the regime-
    specific half-out), _R, assign_window, WINDOWS, window_weeks,
    reads_for_exit_series, FIXED_RISK_DOLLARS, f1679/f1668/cb/orb_csv.
  1687_cells.py's idea43_walk is NOT imported (its module-level
    logging.basicConfig(mode='w') would truncate 1687_cells.log, another
    cell's file) -- its formula (total_R = 2*r1 - add_r when the add
    triggers before a stop-out) is reproduced verbatim below, generalized
    to add_r in {0.5,1.0,1.5} and n_add in {1,2} (add_combined_R).
  research/orb_freq/1694_runners.csv -- bin_rvol_0935 (the frozen, parity-
    checked L1 tilt label, PARITY_1697_tilt.md: 462/462 production fills
    100%) and A3_noTarget_liveLock, used ONLY as an independent-reimplementation
    cross-check against this script's own freshly-walked r1 (never as the
    source of r1 itself -- CLAUDE.md independent-check rule).
  research/orb_freq/1684_pool_books.csv -- P1 (pool=='idea1' & window==
    'in_regime'), entry_price only; stop is RE-reconstructed on bars_sip.db
    (f1693.reconstruct_fill), matching the rest of the stack's bar source --
    RESULT_1690.md found a 0.93R mean mismatch using cache.db bars for this
    same pool, so cache.db is deliberately NOT used here.
  analysis_results/orb_bplus_book.csv (f1679.ORB_BOOK_CSV) -- entered-inclusive
    (CLAUDE_HISTORY.md "Entered-inclusive book"): ALL rows (entered 0 or 1)
    give the day's candidate count for L12, no separate "nightly" file exists.

Read-only on data/cache.db (unused directly) and the bar store (bars_sip.db
via f1668.BarStore, opened ?mode=ro with a 30s backoff on lock -- f1668's own
convention, reused unchanged, never re-implemented here). Never touches
config/orb.yaml/trading/*.py, no git commit, no money spent, one process,
nice -n 10.

Run: nice -n 10 python3 research/orb_freq/1699_layers2.py
"""
import os
import sys
import time
import json
import logging
import importlib.util
from datetime import date as _date, datetime

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, 'research/orb_freq')
LOG_PATH = os.path.join(OUT_DIR, '1699_layers2.log')
READS_CSV = os.path.join(OUT_DIR, '1699_reads.csv')
WEEKLY_CSV = os.path.join(OUT_DIR, '1699_weekly_q3.csv')
EQUITY_CSV = os.path.join(OUT_DIR, '1699_equity.csv')
SUMMARY_JSON = os.path.join(OUT_DIR, '1699_summary.json')
RESULT_MD = os.path.join(OUT_DIR, 'RESULT_1699.md')
POOL_BOOK = os.path.join(OUT_DIR, '1684_pool_books.csv')
RUNNERS_CSV = os.path.join(OUT_DIR, '1694_runners.csv')

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(LOG_PATH, mode='w'), logging.StreamHandler()])
logger = logging.getLogger('cell1699')


def check_disk(floor_gb=5.0):
    st = os.statvfs(ROOT)
    free_gb = st.f_bavail * st.f_frsize / 1e9
    logger.info('disk free: %.1f GB (floor %.1f GB)', free_gb, floor_gb)
    if free_gb < floor_gb:
        raise RuntimeError(f'disk free {free_gb:.1f}GB below floor {floor_gb}GB -- refusing to run')


def _load_module(name, relpath):
    path = os.path.join(ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old_argv
    return mod


# ---------------------------------------------------------------------------
# Generalized add/lock walkers (idea43_walk's formula reproduced + extended)
# ---------------------------------------------------------------------------

def scan_add_trigger(bars, i0, stop, add_lvl, eod_m):
    """Did price touch add_lvl (this bar's high) before a stop-out or EOD?
    Stop/EOD checked first each bar -- same precedence as idea43_walk."""
    n = len(bars['o'])
    for j in range(i0 + 1, n):
        if bars['minarr'][j] >= eod_m:
            return False
        if float(bars['l'][j]) <= stop:
            return False
        if float(bars['h'][j]) >= add_lvl:
            return True
    return False


def add_combined_R(r1, add_r, n_add, triggered):
    """idea43_walk's formula (total_R = 2*r1 - add_r when triggered, n_add=1)
    generalized to n_add extra units, each entering at the SAME add_lvl with
    the SAME shared stop: each extra unit contributes (r1 - add_r) in
    original-R units (sum, not averaged -- $ = R_sum * $375, same convention
    as WEEKLY_Q3_base_tilt_add.md's $Add column)."""
    if not triggered:
        return r1
    return r1 + n_add * (r1 - add_r)


def two_unit_lock_walk(bars, i0, entry, stop, R_unit, add_r, lock_arm_r, lock_stop_r, eod_m, slip_bps):
    """L10: re-read the lock for the COMBINED (1 base + 1 add-at-add_r) unit
    position. Price-to-combinedR map: pre-add R(px)=(px-entry)/R_unit; post-
    add (add_lvl touched) R(px)=2*(px-entry)/R_unit - add_r (continuous at
    px=add_lvl, verified: both give add_r there). Same same-bar precedence as
    f1679._lock_walk (update add/arm state off this bar's high, THEN check
    this bar's low against the just-updated stop)."""
    n = len(bars['o'])
    if i0 + 1 >= n:
        return np.nan
    slip = slip_bps / 10000.0
    add_lvl = entry + add_r * R_unit
    added = False
    armed = False
    stop_price = stop

    def combined_R(px):
        if not added:
            return (px - entry) / R_unit
        return 2.0 * (px - entry) / R_unit - add_r

    for j in range(i0 + 1, n):
        if bars['minarr'][j] >= eod_m:
            return combined_R(float(bars['c'][j]) * (1 - slip))
        bh, bl = float(bars['h'][j]), float(bars['l'][j])
        if not added and bh >= add_lvl:
            added = True
        if added:
            arm_price = entry + (lock_arm_r + add_r) * R_unit / 2.0
            new_stop_price = entry + (lock_stop_r + add_r) * R_unit / 2.0
        else:
            arm_price = entry + lock_arm_r * R_unit
            new_stop_price = entry + lock_stop_r * R_unit
        if not armed and bh >= arm_price:
            armed = True
            stop_price = max(stop_price, new_stop_price)
        if bl <= stop_price:
            return combined_R(stop_price * (1 - slip))
    return combined_R(float(bars['c'][n - 1]) * (1 - slip))


# ---------------------------------------------------------------------------
# Stats helpers (own/paired reads, inlined from 1687_cells.py's pattern --
# NOT imported, to avoid its module-level logging.basicConfig(mode='w')
# truncating 1687_cells.log, another already-completed cell's file)
# ---------------------------------------------------------------------------

def own_reads(vals, dates, lo, hi, f1693, label=''):
    r = f1693.reads_for_exit_series(vals, dates, lo, hi)
    weeks = f1693.window_weeks(lo, hi)
    r['dollars_per_yr_375'] = (r['mean_R'] * r['fills_per_week'] * 52.0 * f1693.FIXED_RISK_DOLLARS) if weeks else np.nan
    r['label'] = label
    return r


def paired_reads(df, value_col, base_col, f1693, label=''):
    out = {}
    for wname, (lo, hi) in f1693.WINDOWS.items():
        sub = df[df['window'] == wname]
        mask = sub[value_col].notna() & sub[base_col].notna()
        delta = (sub[value_col] - sub[base_col])[mask]
        dates = sub.loc[mask, 'date']
        r = f1693.reads_for_exit_series(delta, dates, lo, hi)
        weeks = f1693.window_weeks(lo, hi)
        r['dollars_per_yr_375'] = (r['mean_R'] * r['fills_per_week'] * 52.0 * f1693.FIXED_RISK_DOLLARS) if weeks else np.nan
        r['window'] = wname
        r['label'] = label
        r['n_total_window'] = len(sub)
        out[wname] = r
        logger.info('%-28s %-10s n=%4d meanDR=%+.4f day_t=%+.2f ex_top5=%+.4f $/yr@375=%+8.0f',
                    label, wname, r['n'], r['mean_R'], r['day_t'], r['ex_top5'], r['dollars_per_yr_375'])
    return out


def layer_joins(reads_both, thresh=0.03, ex5_floor=-0.02):
    """Pass rule (PREREG_1699): paired dR>=+0.03R both directions, same
    sign, ex-top-5% dR>=-0.02. TRAIN2025/VAL2026 are the two directions;
    OOS2024H2 has zero production fills (confirmed in 1687_cells.py's own
    caveat) so is reported, never gates."""
    tr, va = reads_both.get('TRAIN2025'), reads_both.get('VAL2026')
    if tr is None or va is None or tr['n'] == 0 or va['n'] == 0:
        return False, 'insufficient fills in TRAIN2025/VAL2026'
    ok = (tr['mean_R'] >= thresh and va['mean_R'] >= thresh and
          np.sign(tr['mean_R']) == np.sign(va['mean_R']) and
          tr['ex_top5'] >= ex5_floor and va['ex_top5'] >= ex5_floor)
    return bool(ok), f"TRAIN dR={tr['mean_R']:+.4f} VAL dR={va['mean_R']:+.4f} ex5 {tr['ex_top5']:+.3f}/{va['ex_top5']:+.3f}"


# ---------------------------------------------------------------------------
# Base book: production fills, r1 + add sweep + L10's 8 lock combos
# ---------------------------------------------------------------------------

LOCK_ARM_GRID = [1.5, 1.75, 2.0, 2.5]
LOCK_STOP_GRID = [0.5, 1.0]
ADD_R_GRID = [0.5, 1.0, 1.5]


def build_base_book(f1693, f1679, store):
    raw = f1693.orb_csv.read_orb_csv(f1679.ORB_BOOK_CSV)
    ent = raw[raw['entered'] == 1].copy()
    ent['date'] = ent['date'].astype(str)
    logger.info('base book: %d entered rows from %s (of %d total rows, entered-inclusive)',
                len(ent), f1679.ORB_BOOK_CSV, len(raw))
    rows = []
    recon = {'no_bars': 0, 'no_range_or_breakout': 0, 'bad_R': 0, 'below_R_floor': 0, 'ok': 0}
    t0 = time.time()
    for n_seen, r in enumerate(ent.itertuples(), 1):
        rec, reason = f1693.reconstruct_fill(store, r.symbol, r.date, r.entry_price)
        if rec is None:
            recon[reason] += 1
            continue
        recon['ok'] += 1
        bars, i0, entry, stop, R_unit = rec['bars'], rec['i0'], rec['entry'], rec['stop'], rec['R_unit']
        eod_m, slip_bps = f1679.ORB_EOD_M, f1679.EXIT_SLIP_BPS
        px1 = f1679._lock_walk(bars, i0, entry, stop, R_unit, f1679.LOCK_TRIGGER_R_LIVE,
                                f1679.LOCK_STOP_R_LIVE, eod_m, slip_bps)
        r1 = f1693._R(px1, entry, R_unit)
        row = {'date': pd.Timestamp(r.date), 'symbol': r.symbol, 'window': f1693.assign_window(f1693.pdate(r.date)),
               'r1': r1}
        for add_r in ADD_R_GRID:
            trig = scan_add_trigger(bars, i0, stop, entry + add_r * R_unit, eod_m)
            row[f'trig_{add_r}'] = trig
            row[f'add1_{add_r}'] = add_combined_R(r1, add_r, 1, trig)
        row['add2_1.0'] = add_combined_R(r1, 1.0, 2, row['trig_1.0'])
        for arm in LOCK_ARM_GRID:
            for lstop in LOCK_STOP_GRID:
                row[f'lock_{arm}_{lstop}'] = two_unit_lock_walk(
                    bars, i0, entry, stop, R_unit, 1.0, arm, lstop, eod_m, slip_bps)
        rows.append(row)
        if n_seen % 150 == 0:
            logger.info('base book: walked %d/%d (%.0fs)', n_seen, len(ent), time.time() - t0)
    logger.info('base book: reconstruction done n=%d %s (%.0fs)', len(ent), recon, time.time() - t0)
    return pd.DataFrame(rows), recon


def attach_tilt(df):
    """Merge the frozen L1 RVOL tilt label from 1694_runners.csv (parity-
    checked elsewhere, PARITY_1697_tilt.md) and cross-check this script's
    freshly-walked r1 against that file's A3_noTarget_liveLock (independent-
    reimplementation check, CLAUDE.md rule 1)."""
    run = pd.read_csv(RUNNERS_CSV)
    run['date'] = pd.to_datetime(run['date'])
    run = run[['date', 'symbol', 'bin_rvol_0935', 'A3_noTarget_liveLock']]
    m = df.merge(run, on=['date', 'symbol'], how='left')
    both = m['A3_noTarget_liveLock'].notna() & m['r1'].notna()
    if both.sum() > 0:
        diff = (m.loc[both, 'r1'] - m.loc[both, 'A3_noTarget_liveLock']).abs()
        logger.info('independent-reimplementation check vs 1694_runners.csv A3_noTarget_liveLock: '
                    'n_matched=%d/%d, mean|diff|=%.5f, max|diff|=%.5f',
                    both.sum(), len(m), diff.mean(), diff.max())
        if diff.mean() > 0.01:
            logger.warning('r1 mean|diff| %.4f vs 1694_runners.csv exceeds 0.01R -- '
                            'investigate before trusting downstream numbers', diff.mean())
    else:
        logger.warning('independent-reimplementation check: 0 rows matched 1694_runners.csv on (date,symbol)')
    mult_map = {}
    for v in m['bin_rvol_0935'].dropna().unique():
        lv = str(v).lower()
        mult_map[v] = 1.5 if 'low' in lv else (0.5 if 'high' in lv else 1.0)
    m['tilt_mult'] = m['bin_rvol_0935'].map(mult_map).fillna(1.0)
    n_neutral = (m['bin_rvol_0935'].isna()).sum()
    if n_neutral:
        logger.warning('%d/%d fills have no bin_rvol_0935 (unmerged) -- neutral 1.0x tilt applied, not imputed',
                        n_neutral, len(m))
    return m


# ---------------------------------------------------------------------------
# L12: gapper-count tilt
# ---------------------------------------------------------------------------

def day_candidate_counts(f1679, orb_csv):
    raw = orb_csv.read_orb_csv(f1679.ORB_BOOK_CSV)
    raw['date'] = pd.to_datetime(raw['date'].astype(str))
    cnt = raw.groupby('date').size().rename('day_count').reset_index()
    logger.info('L12: day candidate counts (entered-inclusive) over %d trading days, mean %.1f, median %.1f',
                len(cnt), cnt['day_count'].mean(), cnt['day_count'].median())
    return cnt


def tercile_fit(train_vals):
    train_vals = train_vals.dropna()
    if len(train_vals) < 10:
        return None
    return float(train_vals.quantile(1 / 3)), float(train_vals.quantile(2 / 3))


def tercile_label(x, edges):
    if edges is None or pd.isna(x):
        return np.nan
    e1, e2 = edges
    if x < e1:
        return 'low'
    if x < e2:
        return 'mid'
    return 'high'


def l12_tilt_direction(df, fit_window, test_window, r_col):
    """Fit tercile edges + ordering (by cell mean of r_col) on fit_window,
    apply unchanged to test_window. Returns (mult_series_full, ordering_dict,
    ev_gain_on_test) -- mult applied to ALL rows using fit_window's edges."""
    fit_lo, fit_hi = fit_window
    fit_mask = (df['date'] >= pd.Timestamp(fit_lo)) & (df['date'] <= pd.Timestamp(fit_hi))
    edges = tercile_fit(df.loc[fit_mask, 'day_count'])
    if edges is None:
        return None, None, None
    labels = df['day_count'].apply(lambda x: tercile_label(x, edges))
    cell_mean = df.loc[fit_mask].groupby(labels[fit_mask])[r_col].mean()
    order = cell_mean.sort_values()
    rank_mult = {}
    tiers = ['low', 'mid', 'high'] if len(order) < 3 else list(order.index)
    mult_by_label = {tiers[0]: 0.5, tiers[-1]: 1.5}
    for lab in ('low', 'mid', 'high'):
        if lab not in mult_by_label:
            mult_by_label[lab] = 1.0
    mult = labels.map(mult_by_label).fillna(1.0)
    test_lo, test_hi = test_window
    test_mask = (df['date'] >= pd.Timestamp(test_lo)) & (df['date'] <= pd.Timestamp(test_hi))
    base_ev = df.loc[test_mask, r_col].mean()
    tilted_ev = (df.loc[test_mask, r_col] * mult[test_mask]).mean()
    base_risk = 1.0
    tilted_risk = mult[test_mask].mean()
    ev_gain = (tilted_ev / tilted_risk) / (base_ev / base_risk) - 1.0 if base_ev != 0 and tilted_risk != 0 else np.nan
    return mult, mult_by_label, ev_gain


# ---------------------------------------------------------------------------
# P1 pool (L11/L13)
# ---------------------------------------------------------------------------

def build_p1_book(f1693, f1679, store):
    d = pd.read_csv(POOL_BOOK, keep_default_na=False, na_values=[''])
    d = d[(d['pool'] == 'idea1') & (d['window'] == 'in_regime')].copy()
    d['date'] = pd.to_datetime(d['date'])
    logger.info('P1 pool (idea1, in_regime): %d rows from %s', len(d), POOL_BOOK)
    rows = []
    recon = {'no_bars': 0, 'no_range_or_breakout': 0, 'bad_R': 0, 'below_R_floor': 0, 'ok': 0}
    for r in d.itertuples():
        date_str = r.date.strftime('%Y-%m-%d')
        rec, reason = f1693.reconstruct_fill(store, r.symbol, date_str, r.entry_price)
        if rec is None:
            recon[reason] += 1
            continue
        recon['ok'] += 1
        bars, i0, entry, stop, R_unit = rec['bars'], rec['i0'], rec['entry'], rec['stop'], rec['R_unit']
        eod_m, slip_bps = f1679.ORB_EOD_M, f1679.EXIT_SLIP_BPS
        px1 = f1679._lock_walk(bars, i0, entry, stop, R_unit, f1679.LOCK_TRIGGER_R_LIVE,
                                f1679.LOCK_STOP_R_LIVE, eod_m, slip_bps)
        r1 = f1693._R(px1, entry, R_unit)
        halfout = f1693._scale_then_lock_walk(bars, i0, entry, stop, R_unit, 1.0, eod_m, slip_bps)
        trig = scan_add_trigger(bars, i0, stop, entry + 1.0 * R_unit, eod_m)
        rows.append({'date': r.date, 'symbol': r.symbol, 'window': f1693.assign_window(r.date.date()),
                     'r1': r1, 'halfout_1R': halfout, 'add_1.0': add_combined_R(r1, 1.0, 1, trig)})
    n_total = len(d)
    n_ok = len(rows)
    logger.info('P1 bar-walk coverage (bars_sip.db): %d/%d (%.1f%%) %s', n_ok, n_total, 100 * n_ok / max(n_total, 1), recon)
    if n_ok / max(n_total, 1) < 0.80:
        logger.warning('P1 coverage %.1f%% below the 80%% availability rail -- reported on the reduced population, not hidden',
                        100 * n_ok / max(n_total, 1))
    return pd.DataFrame(rows), recon, n_ok / max(n_total, 1)


# ---------------------------------------------------------------------------
# L14: compounding equity curve, above-water rule
# ---------------------------------------------------------------------------

def week_monday(d):
    d = pd.Timestamp(d)
    return (d - pd.Timedelta(days=d.weekday())).normalize()


def equity_curve(df, r_col, start_equity=65000.0, risk_frac=0.005):
    """Risk per fill = risk_frac * high_water_equity, frozen for the ISO
    week, updated AFTER each week's fills settle. Above-water rule: the
    sizing basis (high_water_equity) only ever increases when the actual
    equity is at/above it -- a losing week never raises next week's risk$,
    matching the ORB ramp's own above-water rule (project convention,
    docs/bf_p1_ramp.md / project_orb_ramp_above_water_rule)."""
    d = df.dropna(subset=[r_col]).sort_values('date').copy()
    d['wk'] = d['date'].apply(week_monday)
    weekly = d.groupby('wk')[r_col].sum().sort_index()
    equity = start_equity
    high_water = start_equity
    out = []
    for wk, r_sum in weekly.items():
        risk_dollars = risk_frac * high_water
        pnl = r_sum * risk_dollars
        equity += pnl
        if equity > high_water:
            high_water = equity
        out.append({'week': wk, 'risk_dollars_used': risk_dollars, 'r_sum': r_sum,
                    'pnl': pnl, 'equity': equity, 'high_water': high_water})
    curve = pd.DataFrame(out)
    if len(curve) == 0:
        return curve, start_equity, 0.0
    running_max = curve['equity'].cummax()
    dd = (running_max - curve['equity'])
    max_dd = float(dd.max())
    return curve, float(curve['equity'].iloc[-1]), max_dd


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    check_disk()
    logger.info('=== cell 1,699: second layer batch on the ORB stack -- starting ===')

    f1693 = _load_module('f1693_1699', 'research/orb_freq/1693_pool_exits.py')
    f1679 = f1693.f1679
    orb_csv = f1693.orb_csv
    logger.info('loaded 1693_pool_exits.py (f1693) and its own f1679/f1668/f1677/cb/orb_csv')

    store = f1693.CachedStore(f1693.f1668.BARS_DB)

    base, base_recon = build_base_book(f1693, f1679, store)
    base = attach_tilt(base)
    base['R_tilt'] = base['r1'] * base['tilt_mult']

    summary = {'base_recon': base_recon, 'n_base': len(base)}
    all_reads = {}

    # ---- L9: add-level sweep (read on top of base+tilt) ----
    logger.info('--- L9: add-level sweep (on top of base+tilt) ---')
    l9_candidates = {}
    for add_r in ADD_R_GRID:
        base[f'R_L9_{add_r}'] = base[f'add1_{add_r}'] * base['tilt_mult']
        reads = paired_reads(base, f'R_L9_{add_r}', 'R_tilt', f1693, label=f'L9_add_{add_r}R')
        all_reads[f'L9_add_{add_r}R'] = reads
        joins, note = layer_joins(reads)
        l9_candidates[f'add_{add_r}'] = (joins, reads, note)
    base['R_L9_2u'] = base['add2_1.0'] * base['tilt_mult']
    reads_2u = paired_reads(base, 'R_L9_2u', 'R_tilt', f1693, label='L9_add_2units_1.0R')
    all_reads['L9_add_2units_1.0R'] = reads_2u
    joins_2u, note_2u = layer_joins(reads_2u)
    l9_candidates['2units_1.0'] = (joins_2u, reads_2u, note_2u)

    passing = {k: v for k, v in l9_candidates.items() if v[0]}
    if passing:
        l9_winner = max(passing, key=lambda k: (passing[k][1]['TRAIN2025']['mean_R'] + passing[k][1]['VAL2026']['mean_R']))
    else:
        l9_winner = 'add_1.0'  # reference: the live stack's existing +1R single add
    logger.info('L9 winner: %s (joins=%s)', l9_winner, l9_winner in passing)
    if l9_winner == '2units_1.0':
        add_r_star, n_add_star = 1.0, 2
        base['R_L9star'] = base['R_L9_2u']
    else:
        add_r_star = float(l9_winner.split('_')[1])
        n_add_star = 1
        base['R_L9star'] = base[f'R_L9_{add_r_star}']
    summary['L9_winner'] = l9_winner
    summary['L9_add_r_star'] = add_r_star
    summary['L9_n_add_star'] = n_add_star

    # ---- L10: lock re-read on the combined position (at L9's chosen add_r) ----
    logger.info('--- L10: lock re-read on the combined 2-unit position (add_r=%.1f) ---', add_r_star)
    if add_r_star != 1.0:
        logger.warning('L10 grid was walked at add_r=1.0 only (budget); L9 winner add_r=%.1f -- '
                        'L10 compared against the add_r=1.0 reference, flagged not re-walked', add_r_star)
    l10_candidates = {}
    for arm in LOCK_ARM_GRID:
        for lstop in LOCK_STOP_GRID:
            col = f'R_L10_{arm}_{lstop}'
            base[col] = base[f'lock_{arm}_{lstop}'] * base['tilt_mult']
            reads = paired_reads(base, col, 'R_L9star', f1693, label=f'L10_arm{arm}_stop{lstop}')
            all_reads[f'L10_arm{arm}_stop{lstop}'] = reads
            joins, note = layer_joins(reads)
            l10_candidates[(arm, lstop)] = (joins, reads, note)
    passing10 = {k: v for k, v in l10_candidates.items() if v[0]}
    if passing10:
        l10_winner = max(passing10, key=lambda k: (passing10[k][1]['TRAIN2025']['mean_R'] + passing10[k][1]['VAL2026']['mean_R']))
        base['R_L10star'] = base[f'R_L10_{l10_winner[0]}_{l10_winner[1]}']
        logger.info('L10 winner: arm=%s stop=%s (joins)', *l10_winner)
    else:
        l10_winner = None
        base['R_L10star'] = base['R_L9star']
        logger.info('L10: no combo passed the bar -- stack keeps the live 1.75/0.5 reference (per-unit)')
    summary['L10_winner'] = str(l10_winner)

    # ---- L12: gapper-count tilt (fit one window, test the other, both ways) ----
    logger.info('--- L12: gapper-count tilt ---')
    cnt = day_candidate_counts(f1679, orb_csv)
    base = base.merge(cnt, on='date', how='left')
    n_missing_cnt = base['day_count'].isna().sum()
    if n_missing_cnt:
        logger.warning('%d/%d base fills have no matching day_count row -- excluded from L12 tilt (neutral 1.0x)',
                        n_missing_cnt, len(base))
    TRAIN_W, VAL_W = f1693.WINDOWS['TRAIN2025'], f1693.WINDOWS['VAL2026']
    mult_A, map_A, ev_A = l12_tilt_direction(base, TRAIN_W, VAL_W, 'R_L10star')
    mult_B, map_B, ev_B = l12_tilt_direction(base, VAL_W, TRAIN_W, 'R_L10star')
    logger.info('L12 direction A (fit TRAIN, test VAL): map=%s EV/risk gain on VAL=%+.1f%%', map_A, 100 * (ev_A or 0))
    logger.info('L12 direction B (fit VAL, test TRAIN): map=%s EV/risk gain on TRAIN=%+.1f%%', map_B, 100 * (ev_B or 0))
    order_agrees = (map_A is not None and map_B is not None and
                    {k: v for k, v in map_A.items()} == {k: v for k, v in map_B.items()})
    l12_joins = bool(order_agrees and ev_A is not None and ev_B is not None and ev_A >= 0.10 and ev_B >= 0.10)
    logger.info('L12 joins=%s (ordering agrees=%s, EV/risk gain A=%.1f%% B=%.1f%%)',
                l12_joins, order_agrees, 100 * (ev_A or 0), 100 * (ev_B or 0))
    summary['L12_joins'] = l12_joins
    summary['L12_ev_gain_A'] = ev_A
    summary['L12_ev_gain_B'] = ev_B
    if l12_joins:
        base['R_final'] = base['R_L10star'] * mult_A
    else:
        base['R_final'] = base['R_L10star']

    # ---- L11: P1 pool joined as a frequency layer ----
    logger.info('--- L11: P1 pool (idea1) as a frequency layer ---')
    p1, p1_recon, p1_cov = build_p1_book(f1693, f1679, store)
    summary['P1_recon'] = p1_recon
    summary['P1_coverage'] = p1_cov
    p1_r1_reads = {w: own_reads(p1.loc[p1['window'] == w, 'r1'], p1.loc[p1['window'] == w, 'date'], lo, hi, f1693, f'P1_r1_{w}')
                   for w, (lo, hi) in f1693.WINDOWS.items()}
    p1_halfout_reads = {w: own_reads(p1.loc[p1['window'] == w, 'halfout_1R'], p1.loc[p1['window'] == w, 'date'], lo, hi, f1693, f'P1_halfout_{w}')
                        for w, (lo, hi) in f1693.WINDOWS.items()}
    p1_pair = paired_reads(p1, 'halfout_1R', 'r1', f1693, label='L11_P1_halfout_vs_plain')
    all_reads['L11_P1_halfout_vs_plain'] = p1_pair
    l11_joins, l11_note = layer_joins(p1_pair)
    logger.info('L11 P1 half-out vs plain joins=%s (%s)', l11_joins, l11_note)
    summary['L11_joins'] = l11_joins
    logger.warning('L11 tilt-transfer ("does the tilt transfer to P1?"): NOT COMPUTED -- no rvol_0935 '
                   'feature exists for the P1 population (1684_pool_books.csv has no volume columns; '
                   'building one needs a new 20d-avg-volume + 09:35 cumulative-volume fetch, out of this '
                   'cell''s budget and no owner GO for a new pull). Reported as a gap, not skipped silently.')
    p1_exit_col = 'halfout_1R' if l11_joins else 'r1'
    union_dollars_by_wk = None  # built below alongside the Q3 table

    # ---- L13: adds on base (already folded via L9) and on P1 ----
    logger.info('--- L13: add at the L9-chosen level, on P1 ---')
    if add_r_star != 1.0:
        logger.warning('L13 P1 add was walked at add_r=1.0 only (same budget constraint as L10); '
                        'L9 winner add_r=%.1f -- P1+add reported at the 1.0R reference level', add_r_star)
    p1_add_pair = paired_reads(p1, 'add_1.0', 'r1', f1693, label='L13_P1_add_vs_plain')
    all_reads['L13_P1_add_vs_plain'] = p1_add_pair
    l13_joins, l13_note = layer_joins(p1_add_pair)
    logger.info('L13 P1+add vs P1 plain joins=%s (%s); base-side add is L9 (already in the stack)', l13_joins, l13_note)
    summary['L13_joins'] = l13_joins

    # ---- Weekly Q3 2026 table: base vs final stack vs +P1 union ----
    logger.info('--- Q3 2026 weekly table ---')
    q3_lo, q3_hi = pd.Timestamp('2026-06-29'), pd.Timestamp('2026-09-27')
    b = base[(base['date'] >= q3_lo) & (base['date'] <= q3_hi)].copy()
    b['wk'] = b['date'].apply(week_monday)
    p1q = p1[(p1['date'] >= q3_lo) & (p1['date'] <= q3_hi)].copy()
    p1q['wk'] = p1q['date'].apply(week_monday)
    p1q['dollar'] = p1q[p1_exit_col] * f1693.FIXED_RISK_DOLLARS
    weeks = pd.date_range(q3_lo, q3_hi, freq='W-MON')
    wk_rows = []
    cum_base, cum_final, cum_union = 0.0, 0.0, 0.0
    for i, wk in enumerate(weeks, 1):
        sub = b[b['wk'] == wk]
        subp1 = p1q[p1q['wk'] == wk]
        n_fills = len(sub) + len(subp1)
        d_base = float(sub['r1'].sum() * f1693.FIXED_RISK_DOLLARS)
        d_final = float(sub['R_final'].sum() * f1693.FIXED_RISK_DOLLARS)
        d_union = d_final + float(subp1['dollar'].sum())
        cum_base += d_base
        cum_final += d_final
        cum_union += d_union
        wk_rows.append({'wk_num': 26 + i, 'monday': wk.strftime('%m-%d'), 'fills_base': len(sub),
                        'fills_p1': len(subp1), 'dollar_base': round(d_base), 'dollar_final_1699': round(d_final),
                        'dollar_union_with_P1': round(d_union), 'green_base': None if n_fills == 0 else d_base > 0,
                        'green_final': None if n_fills == 0 else d_final > 0,
                        'green_union': None if n_fills == 0 else d_union > 0})
    weekly_df = pd.DataFrame(wk_rows)
    weekly_df.to_csv(WEEKLY_CSV, index=False)

    def dd_from_zero(series):
        cum = series.cumsum()
        return float((cum.cummax() - cum).max())

    q3_summary = {
        'base': {'total': float(weekly_df['dollar_base'].sum()), 'green': int((weekly_df['green_base'] == True).sum()),
                 'worst': float(weekly_df['dollar_base'].min()), 'dd': dd_from_zero(weekly_df['dollar_base'])},
        'final_1699': {'total': float(weekly_df['dollar_final_1699'].sum()), 'green': int((weekly_df['green_final'] == True).sum()),
                       'worst': float(weekly_df['dollar_final_1699'].min()), 'dd': dd_from_zero(weekly_df['dollar_final_1699'])},
        'union_with_P1': {'total': float(weekly_df['dollar_union_with_P1'].sum()), 'green': int((weekly_df['green_union'] == True).sum()),
                          'worst': float(weekly_df['dollar_union_with_P1'].min()), 'dd': dd_from_zero(weekly_df['dollar_union_with_P1'])},
    }
    summary['q3_weekly'] = q3_summary
    logger.info('Q3 totals: base=$%.0f final1699=$%.0f union+P1=$%.0f', q3_summary['base']['total'],
                q3_summary['final_1699']['total'], q3_summary['union_with_P1']['total'])

    # ---- L14: compounding equity curve ----
    logger.info('--- L14: compounding equity curve (0.5%% of equity, above-water, from $65,000) ---')
    curve, end_equity, max_dd = equity_curve(base, 'R_final', start_equity=65000.0, risk_frac=0.005)
    curve.to_csv(EQUITY_CSV, index=False)
    q3_curve = curve[(curve['week'] >= q3_lo) & (curve['week'] <= q3_hi)]
    q3_dd = float((q3_curve['equity'].cummax() - q3_curve['equity']).max()) if len(q3_curve) else 0.0
    q3_end = float(q3_curve['equity'].iloc[-1]) if len(q3_curve) else float('nan')
    q3_pnl = float(q3_curve['pnl'].sum()) if len(q3_curve) else 0.0
    logger.info('L14 full 2025-01..latest: end equity=$%.0f max DD=$%.0f', end_equity, max_dd)
    logger.info('L14 Q3 2026 only: pnl=$%.0f end equity=$%.0f within-Q3 DD=$%.0f', q3_pnl, q3_end, q3_dd)
    summary['L14'] = {'end_equity': end_equity, 'max_dd': max_dd, 'q3_pnl': q3_pnl, 'q3_end_equity': q3_end, 'q3_dd': q3_dd,
                       'n_weeks': int(len(curve))}

    # ---- write reads CSV ----
    flat = []
    for layer_label, wdict in all_reads.items():
        for wname, r in wdict.items():
            flat.append({'layer': layer_label, 'window': wname, **{k: v for k, v in r.items() if k not in ('label',)}})
    pd.DataFrame(flat).to_csv(READS_CSV, index=False)

    with open(SUMMARY_JSON, 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info('=== cell 1,699 done in %.0fs ===', time.time() - t_start)
    store.close()


if __name__ == '__main__':
    main()
