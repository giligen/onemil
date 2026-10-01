#!/usr/bin/env python3
"""Cell 1,698: six further layers on the ORB production stack (base + RVOL
tilt L1 + add-at-+1R L2), each read ON TOP of that FROZEN stack, in both
selection directions, with the Q3 2026 week-by-week $ table of the final
stack beside base.

PREREG: research/orb_freq/PREREG_1698.md (FROZEN 2026-10-01 18:10 UTC).
Owner's verbatim ask: "Love the new P&L. More. Layers?"

Reuses (imported read-only via importlib -- digit-prefixed filenames can't
be `import`ed, project convention -- nothing below is modified):
  research/orb_exit/1679_orb_exit.py (f1679) -- _lock_walk (the live-lock
    walker), ORB_EOD_M, EXIT_SLIP_BPS, LOCK_TRIGGER_R_LIVE, LOCK_STOP_R_LIVE.
  research/orb_freq/1687_cells.py (f1687) -- idea43_walk, called VERBATIM
    for the L2 add. L3/L4/L8 below are NEW functions that extend its same
    mechanism (stop-wins-a-tie, -slip-on-every-touch) without touching the
    file, per the PREREG's explicit instruction.
  research/hod_entry/1668_failure.py (f1668) -- BarStore (bars_sip.db,
    read-only) and spy_close_at_or_before.
  research/hod_entry/1677_take_profit.py (f1677) -- stats_block (n,
    mean_dR, iid_t, day_t, ex_top5, mde).
  scripts/cadence_bar.py (cb) -- build_weekly_series, percentile,
    max_drawdown_and_underwater.
  research/orb_freq/1694_runners.csv -- the per-fill base R
    (A3_noTarget_liveLock), R_unit, entry, entry_minute, date, symbol,
    window, and the two FROZEN bin labels (bin_rvol_0935, bin_price) for
    2025-26 (478 fills; PARITY_1697_tilt.md: 100% agreement with the
    live engine's tercile).

OOS2024H2 is reported UNTESTED (n=0): analysis_results/orb_bplus_book.csv
(the source of 1694_runners.csv) only spans 2025-01..2026-09, same caveat
as 1687_cells.passes_1679_bar.

Does NOT touch config, orb.yaml, the live service, trading/*.py, or
cache.db. bars_sip.db is opened read-only (?mode=ro) with a 30s backoff on
"database is locked" -- another cell is appending to it right now; never
killed, never written.

Usage:
    nice -n 10 python3 research/orb_freq/1698_layers.py
"""
import importlib.util
import logging
import os
import shutil
import sys
import time
from datetime import date as _date, datetime, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

LOG_FILE = os.path.join(HERE, '1698_layers.log')
READS_CSV = os.path.join(HERE, '1698_reads.csv')
WEEKLY_CSV = os.path.join(HERE, '1698_weekly_q3.csv')
RESULT_MD = os.path.join(HERE, 'RESULT_1698.md')
RUNNERS_CSV = os.path.join(HERE, '1694_runners.csv')
BARS_DB = os.path.join(ROOT, 'research', 'bf_zero', 'bars_sip.db')

FIXED_RISK_DOLLARS = 375.0
PASS_DR = 0.03
PASS_EXTOP5 = -0.02
PASS_SPEARMAN = 0.6
PASS_EV_GAIN = 0.10

logger = logging.getLogger('cell1698')


def setup_logging():
    """Verbose progress to both the log file and stdout (CLAUDE.md: batch
    processors are verbose; find the root cause in the logging)."""
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE, mode='w')
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)


def check_disk(floor_gb=5.0):
    """Research job hygiene: df >= 5GB before any run (feedback_research_job_hygiene)."""
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    if free_gb < floor_gb:
        logger.error('disk free %.1fGB < floor %.1fGB -- aborting', free_gb, floor_gb)
        sys.exit(1)
    logger.info('disk free %.1fGB OK', free_gb)


def _load_module(name, relpath):
    """Project convention for digit-prefixed filenames: load by path, with
    sys.argv neutralized during exec so a module-level argparse (none
    expected here) can't choke on OUR argv."""
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
# Reused modules (module-level code in each is defs/constants only --
# verified before this cell ran; none execute I/O at import time)
# ---------------------------------------------------------------------------
f1679 = _load_module('f1679_1698', 'research/orb_exit/1679_orb_exit.py')
f1687 = _load_module('f1687_1698', 'research/orb_freq/1687_cells.py')
f1668 = _load_module('f1668_1698', 'research/hod_entry/1668_failure.py')
f1677 = _load_module('f1677_1698', 'research/hod_entry/1677_take_profit.py')
cb = _load_module('cb_1698', 'scripts/cadence_bar.py')

_lock_walk = f1679._lock_walk
ORB_EOD_M = f1679.ORB_EOD_M
EXIT_SLIP_BPS = f1679.EXIT_SLIP_BPS
LOCK_TRIGGER_R_LIVE = f1679.LOCK_TRIGGER_R_LIVE
LOCK_STOP_R_LIVE = f1679.LOCK_STOP_R_LIVE
SLIP = EXIT_SLIP_BPS / 10000.0
stats_block = f1677.stats_block
spy_close_at_or_before = f1668.spy_close_at_or_before
idea43_walk = f1687.idea43_walk


class _ExitShim:
    """Minimal namespace satisfying f1687.idea43_walk's `f1693` parameter
    contract (ORB_EOD_M, EXIT_SLIP_BPS, _lock_walk, LOCK_TRIGGER_R_LIVE,
    LOCK_STOP_R_LIVE, _R) WITHOUT loading all of 1693_pool_exits.py -- f1679
    has everything idea43_walk needs except `_R`, a one-line helper
    (1693_pool_exits._R, reproduced verbatim: (px-entry)/R_unit)."""
    ORB_EOD_M = f1679.ORB_EOD_M
    EXIT_SLIP_BPS = f1679.EXIT_SLIP_BPS
    LOCK_TRIGGER_R_LIVE = f1679.LOCK_TRIGGER_R_LIVE
    LOCK_STOP_R_LIVE = f1679.LOCK_STOP_R_LIVE
    _lock_walk = staticmethod(f1679._lock_walk)

    @staticmethod
    def _R(px, entry, R_unit):
        return (px - entry) / R_unit if px is not None else np.nan


EXIT_SHIM = _ExitShim()

# ---------------------------------------------------------------------------
# Windows (fixed, never inferred from data -- CLAUDE.md statistics rule;
# identical bounds to 1693_pool_exits.WINDOWS)
# ---------------------------------------------------------------------------
TRAIN_LO, TRAIN_HI = _date(2025, 1, 1), _date(2025, 12, 31)
VAL_LO, VAL_HI = _date(2026, 1, 1), _date(2026, 9, 18)
WINDOWS = {'TRAIN2025': (TRAIN_LO, TRAIN_HI), 'VAL2026': (VAL_LO, VAL_HI)}
Q3_LO, Q3_HI = _date(2026, 7, 1), _date(2026, 9, 30)


def pdate(s):
    return datetime.strptime(str(s)[:10], '%Y-%m-%d').date()


def window_weeks(lo, hi):
    return (hi - lo).days / 7.0


# ---------------------------------------------------------------------------
# bars_sip.db: read-only, 30s backoff on "database is locked" (another cell
# is appending right now -- never kill it, never overwrite)
# ---------------------------------------------------------------------------

def open_store_readonly(path, max_wait_s=900, backoff_s=30):
    waited = 0.0
    while True:
        try:
            store = f1668.BarStore(path)
            store.con.execute('SELECT 1').fetchone()
            return store
        except Exception as e:
            if 'locked' not in str(e).lower() or waited >= max_wait_s:
                logger.error('bars_sip.db open failed (waited %.0fs): %s', waited, e)
                raise
            logger.warning('bars_sip.db locked on open, backing off %ds (waited %.0fs): %s',
                            backoff_s, waited, e)
            time.sleep(backoff_s)
            waited += backoff_s


class CachedStore:
    """Process-wide (symbol, date) cache -- L7's SPY lookups and the ~460
    fills share many dates, so repeat lookups are a dict hit, not a query."""

    def __init__(self, path):
        self._store = open_store_readonly(path)
        self._cache = {}

    def day_bars(self, symbol, date_str):
        key = (symbol, str(date_str))
        if key not in self._cache:
            waited = 0.0
            while True:
                try:
                    self._cache[key] = self._store.day_bars(symbol, str(date_str))
                    break
                except Exception as e:
                    if 'locked' not in str(e).lower() or waited >= 900:
                        logger.error('day_bars(%s,%s) failed (waited %.0fs): %s', symbol, date_str, waited, e)
                        raise
                    logger.warning('bars_sip.db locked on day_bars(%s,%s), backing off 30s', symbol, date_str)
                    time.sleep(30)
                    waited += 30
        return self._cache[key]

    def close(self):
        self._store.close()


# ---------------------------------------------------------------------------
# Bar-walk primitives (new; same stop-wins-a-tie / -slip convention as every
# walker in this codebase, e.g. f1687.idea43_walk's own inline loop)
# ---------------------------------------------------------------------------

def touch_before_stop(bars, i0, stop, level, eod_m):
    """First bar index j>i0 whose HIGH >= level, provided no bar from i0+1..j
    had LOW <= stop first (stop wins a same-bar tie). None if stopped out,
    EOD reached, or never touched."""
    n = len(bars['o'])
    for j in range(i0 + 1, n):
        if bars['minarr'][j] >= eod_m:
            return None
        if float(bars['l'][j]) <= stop:
            return None
        if float(bars['h'][j]) >= level:
            return j
    return None


def forward_stop_only(bars, j0, stop_level, eod_m):
    """First bar index j>j0 whose LOW <= stop_level -- a separate, tighter
    protective stop tracked for ONE unit only (the group's original stop is
    not re-checked here). None if never touched before EOD."""
    n = len(bars['o'])
    for j in range(j0 + 1, n):
        if bars['minarr'][j] >= eod_m:
            return None
        if float(bars['l'][j]) <= stop_level:
            return j
    return None


def compute_new_legs(bars, i0, entry, stop, R_unit, r1):
    """L3a/L3b/L4b/L4c/L8's new leg values, each a (base_leg, flat_extra)
    pair in ORIGINAL-R units: total_$ equivalent = tilt_mult*base_leg +
    flat_extra (mirrors the frozen stack's own tilt_mult*r1 + add_incr).
    added/add1_j/add2_j are returned too (used by the caller's idea43
    cross-check and by L5's structural no-op test)."""
    lvl1, lvl2 = entry + 1.0 * R_unit, entry + 2.0 * R_unit
    add1_j = touch_before_stop(bars, i0, stop, lvl1, ORB_EOD_M)
    added = add1_j is not None
    add2_j = touch_before_stop(bars, i0, stop, lvl2, ORB_EOD_M) if added else None

    out = {}

    # L3a: unit3 added at +2R, ORIGINAL stop for all three units
    out['L3a_base'] = r1
    out['L3a_extra'] = (r1 - 2.0) if add2_j is not None else 0.0

    # L3b: unit3's stop = breakeven of unit2 (lvl1) instead of the original
    out['L3b_base'] = r1
    if add2_j is None:
        out['L3b_extra'] = 0.0
    else:
        j3 = forward_stop_only(bars, add2_j, lvl1, ORB_EOD_M)
        out['L3b_extra'] = ((lvl1 * (1 - SLIP) - lvl2) / R_unit) if j3 is not None else (r1 - 2.0)

    # L4b: unit2's stop = breakeven of the add (its own entry, lvl1)
    out['L4b_base'] = r1
    if not added:
        out['L4b_extra'] = 0.0
    else:
        j2 = forward_stop_only(bars, add1_j, lvl1, ORB_EOD_M)
        out['L4b_extra'] = ((lvl1 * (1 - SLIP) - lvl1) / R_unit) if j2 is not None else (r1 - 1.0)

    # L4c: unit2's stop = the live lock, walked as its own independent position
    out['L4c_base'] = r1
    if not added:
        out['L4c_extra'] = 0.0
    else:
        add_R_unit = lvl1 - stop
        exit_px2 = _lock_walk(bars, add1_j, lvl1, stop, add_R_unit,
                               LOCK_TRIGGER_R_LIVE, LOCK_STOP_R_LIVE, ORB_EOD_M, EXIT_SLIP_BPS)
        out['L4c_extra'] = ((exit_px2 - lvl1) / R_unit) if exit_px2 is not None else (r1 - 1.0)

    # L8: base 50% out at +2R ONCE THE ADD IS ON; rest + add ride the live lock unchanged
    if added and add2_j is not None:
        partial_R = (lvl2 * (1 - SLIP) - entry) / R_unit
        out['L8_base'] = 0.5 * partial_R + 0.5 * r1
    else:
        out['L8_base'] = r1

    return out, added, add1_j, add2_j


def spy_return_0935(store, d):
    """SPY's return from the PRIOR trading day's close to 09:35 ET on date
    d (minute-of-day 575, f1668's minute_of_day convention: 570=09:30).
    Walks back up to 5 calendar days (skipping weekends) for the prior
    close. NaN (logged) if either side is unavailable -- counted toward
    availability, never silently zeroed."""
    today = store.day_bars('SPY', str(d))
    if today is None or len(today['o']) == 0:
        return np.nan
    px_0935 = spy_close_at_or_before(today, 575.0)
    prior_close = None
    for back in range(1, 6):
        pdd = d - timedelta(days=back)
        if pdd.weekday() >= 5:
            continue
        pb = store.day_bars('SPY', str(pdd))
        if pb is not None and len(pb['o']) > 0:
            prior_close = spy_close_at_or_before(pb, 1e9)
            break
    if px_0935 is None or prior_close is None or not prior_close:
        return np.nan
    return (px_0935 - prior_close) / prior_close


# ---------------------------------------------------------------------------
# Stats glue (reads_for_exit_series' own logic, reproduced here so this cell
# does not need to load 1693_pool_exits.py -- same stats_block + cadence_bar
# primitives, generic, not a "frozen mechanism")
# ---------------------------------------------------------------------------

def window_read(vals, dates, lo, hi):
    vals = np.asarray(vals, dtype=float)
    dates = np.asarray(dates)
    st = stats_block(vals, dates)
    wk = window_weeks(lo, hi)
    if len(vals):
        weekly = cb.build_weekly_series([{'date': d, 'r': v} for d, v in zip(dates, vals)], lo, hi)
        wr = [w[1] for w in weekly]
        p10 = cb.percentile(wr, 10)
    else:
        p10 = np.nan
    return dict(n=st['n'], mean_dR=st['mean_dR'], iid_t=st['iid_t'], day_t=st['day_t'],
                ex_top5=st['ex_top5'], mde=st['mde'], weekly_p10_dR=p10,
                fills_per_week=(st['n'] / wk if wk else np.nan))


def dollar_curve_stats(dates, dollars, lo, hi):
    weekly = cb.build_weekly_series([{'date': d, 'r': v} for d, v in zip(dates, dollars)], lo, hi)
    wr = [w[1] for w in weekly]
    if not wr:
        return np.nan, np.nan
    worst = min(wr)
    mdd, _ = cb.max_drawdown_and_underwater(wr)
    return worst, mdd


def spearman(a, b):
    """Manual Spearman rank correlation (no scipy dependency for a 9- or
    3-point robustness check)."""
    a, b = pd.Series(a), pd.Series(b)
    if len(a) < 2:
        return np.nan
    ra, rb = a.rank(), b.rank()
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def fit_tilt(fit_df, key_col_fn):
    """Cell means of r1 keyed by key_col_fn(row) on fit_df, ranked into
    terciles -> multiplier 0.5/1.0/1.5 (L6/L7's shared rule). Returns
    (mult_map, cellmeans Series keyed the same way)."""
    keys = fit_df.apply(key_col_fn, axis=1)
    tmp = pd.DataFrame({'_key': keys, 'r1': fit_df['r1'].values})
    cellmeans = tmp.dropna(subset=['_key']).groupby('_key')['r1'].mean()
    n = len(cellmeans)
    if n == 0:
        return {}, cellmeans
    ranks = cellmeans.rank(method='first')
    mult_map = {k: {1: 0.5, 2: 1.0, 3: 1.5}[int(np.ceil(r / n * 3))] for k, r in ranks.items()}
    return mult_map, cellmeans


# ---------------------------------------------------------------------------
# Per-fill reconstruction
# ---------------------------------------------------------------------------

def reconstruct_all(pf, store):
    """For every row in pf (1694_runners.csv, already filtered to rows with
    a usable base_R/R_unit/entry_minute): load bars_sip.db bars, locate the
    entry bar i0 by entry_minute, recompute idea43_R VERBATIM via
    f1687.idea43_walk, and the new L3/L4/L8 legs. Rows whose bars/i0 can't
    be reconstructed are DROPPED and counted (logged WARNING)."""
    records = []
    fail = {'no_bars': 0, 'no_i0_match': 0}
    mismatch = 0
    t0 = time.time()
    for n_seen, row in enumerate(pf.itertuples(), 1):
        bars = store.day_bars(row.symbol, row.date)
        if bars is None or len(bars['o']) < 2:
            fail['no_bars'] += 1
            continue
        idx = np.where(bars['minarr'] == float(row.entry_minute))[0]
        if len(idx) == 0:
            fail['no_i0_match'] += 1
            continue
        i0 = int(idx[0])
        entry, R_unit, r1 = float(row.entry), float(row.R_unit), float(row.A3_noTarget_liveLock)
        stop = entry - R_unit

        idea43_R = idea43_walk(bars, i0, entry, stop, R_unit, EXIT_SHIM)  # verbatim, unchanged
        legs, added, add1_j, add2_j = compute_new_legs(bars, i0, entry, stop, R_unit, r1)
        add_incr = idea43_R - r1
        expected = (r1 - 1.0) if added else 0.0
        if abs(add_incr - expected) > 0.05:
            mismatch += 1
            logger.warning('idea43/touch mismatch sym=%s date=%s added=%s add_incr=%.3f expected=%.3f',
                            row.symbol, row.date, added, add_incr, expected)

        rec = dict(date=row.date, pdate=pdate(row.date), symbol=row.symbol, window=row.window,
                   entry=entry, R_unit=R_unit, entry_minute=row.entry_minute, r1=r1,
                   tilt_mult=row.tilt_mult, bin_rvol_0935=row.bin_rvol_0935, bin_price=row.bin_price,
                   idea43_R=idea43_R, add_incr=add_incr, added=added)
        for k, v in legs.items():
            rec[k] = v
        records.append(rec)
        if n_seen % 100 == 0:
            logger.info('reconstructed %d/%d (%.0fs)', n_seen, len(pf), time.time() - t0)
    logger.info('reconstruction done: %d ok, failures=%s, idea43/touch mismatches=%d', len(records), fail, mismatch)
    return pd.DataFrame(records), fail, mismatch


def attach_spy(df, store):
    """SPY return (prior close -> 09:35) for every unique date in df, cached
    once per date (not per fill)."""
    uniq = sorted(df['pdate'].unique())
    spy_map = {}
    for i, d in enumerate(uniq, 1):
        spy_map[d] = spy_return_0935(store, d)
        if i % 50 == 0:
            logger.info('SPY return: %d/%d dates', i, len(uniq))
    cov = 1.0 - sum(pd.isna(v) for v in spy_map.values()) / max(len(spy_map), 1)
    logger.info('SPY return coverage: %.1f%% of %d dates', 100 * cov, len(uniq))
    df['spy_ret_0935'] = df['pdate'].map(spy_map)
    return df


# ---------------------------------------------------------------------------
# L5: day-risk budget
# ---------------------------------------------------------------------------

def apply_day_risk_budget(df, B):
    """Cap the day's total open BASE risk at B*$375 (tilt_mult*$375 per
    fill); fills beyond the cap are sized to the remaining budget, then
    (once the budget is 0) skipped. Processed in entry-minute order within
    each day. Returns a scale in [0,1] per row, aligned to df's index."""
    d2 = df.sort_values(['pdate', 'entry_minute'])
    scale = pd.Series(1.0, index=d2.index)
    running = {}
    budget = B * FIXED_RISK_DOLLARS
    for idx, row in d2.iterrows():
        used = running.get(row['pdate'], 0.0)
        risk_size = row['tilt_mult'] * FIXED_RISK_DOLLARS
        avail = budget - used
        if avail <= 0:
            s = 0.0
        elif risk_size <= avail:
            s = 1.0
        else:
            s = avail / risk_size
        scale.loc[idx] = s
        running[row['pdate']] = used + s * risk_size
    return scale.reindex(df.index)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_read_row(layer, direction, d_R, dates, lo, hi, layer_dollars, base_dollars, risk_multiple=np.nan, notes=''):
    r = window_read(d_R, dates, lo, hi)
    worst_layer, mdd_layer = dollar_curve_stats(dates, layer_dollars, lo, hi)
    worst_base, mdd_base = dollar_curve_stats(dates, base_dollars, lo, hi)
    passed = bool(r['n'] > 0 and r['mean_dR'] >= PASS_DR and r['ex_top5'] >= PASS_EXTOP5)
    logger.info('%s %s: n=%d mean_dR=%+.4f day_t=%.2f ex_top5=%+.4f worst_wk=$%.0f(base $%.0f) pass=%s',
                layer, direction, r['n'], r['mean_dR'], r['day_t'], r['ex_top5'], worst_layer, worst_base, passed)
    return dict(layer=layer, direction=direction, n=r['n'], mean_dR=r['mean_dR'], iid_t=r['iid_t'],
                day_t=r['day_t'], ex_top5_dR=r['ex_top5'], mde=r['mde'], weekly_p10_dR=r['weekly_p10_dR'],
                worst_week_dollars=worst_layer, max_dd_dollars=mdd_layer, base_worst_week_dollars=worst_base,
                base_max_dd_dollars=mdd_base, risk_multiple=risk_multiple, pass_dR_bar=passed, notes=notes)


def main():
    setup_logging()
    check_disk()
    logger.info('cell 1698: six layers on the ORB stack, both directions, Q3 2026 weekly table')

    pf = pd.read_csv(RUNNERS_CSV)
    pf['date'] = pf['date'].astype(str)
    n0 = len(pf)
    pf = pf[pf['A3_noTarget_liveLock'].notna() & pf['R_unit'].notna() & pf['entry_minute'].notna()].copy()
    tilt_map = {'rvol_0935_low': 1.5, 'rvol_0935_mid': 1.0, 'rvol_0935_high': 0.5}
    pf['tilt_mult'] = pf['bin_rvol_0935'].map(tilt_map).fillna(1.0)
    logger.info('1694_runners.csv: %d rows, %d usable (base_R/R_unit/entry_minute present)', n0, len(pf))

    store = CachedStore(BARS_DB)
    df, fail, mismatch = reconstruct_all(pf, store)
    df = attach_spy(df, store)
    store.close()

    # NOTE (empirically confirmed against WEEKLY_Q3_base_tilt_add.md's published
    # Q3 totals: tilt_mult*idea43_R reproduces $5,142 for 92 fills vs the
    # published $5,021/91 fills -- within the 1-fill population diff; the
    # OTHER split, tilt_mult*r1+add_incr, undershoots at $3,263): the RVOL
    # tilt is a SHARE-COUNT decision made at entry, so it scales the WHOLE
    # blended position (base unit AND any add unit bought at the same
    # tilted share size), not the base leg alone.
    df['stack_R'] = df['tilt_mult'] * df['idea43_R']
    df['stack_dollars'] = df['stack_R'] * FIXED_RISK_DOLLARS
    # L3a/L3b ADD a 3rd unit on top of the (base+add1) idea43_R blend -- tilt
    # the whole thing, same convention as the stack.
    df['L3a_add2R_origStop_R'] = df['tilt_mult'] * (df['idea43_R'] + df['L3a_extra'])
    df['L3b_add2R_add1BE_stop_R'] = df['tilt_mult'] * (df['idea43_R'] + df['L3b_extra'])
    # L4b/L4c REPLACE the add leg entirely -- tilt (base r1 + the new add leg).
    df['L4b_addStop_BE_R'] = df['tilt_mult'] * (df['r1'] + df['L4b_extra'])
    df['L4c_addStop_liveLock_R'] = df['tilt_mult'] * (df['r1'] + df['L4c_extra'])
    # L8 modifies the BASE leg only; the add leg (add_incr) rides unchanged.
    df['L8_basePartial2R_R'] = df['tilt_mult'] * (df['L8_base'] + df['add_incr'])

    reads = []
    simple_layers = ['L3a_add2R_origStop', 'L3b_add2R_add1BE_stop', 'L4b_addStop_BE',
                      'L4c_addStop_liveLock', 'L8_basePartial2R']
    for lyr in simple_layers:
        for wname, (lo, hi) in WINDOWS.items():
            sub = df[df['window'] == wname]
            d_R = sub[lyr + '_R'] - sub['stack_R']
            reads.append(build_read_row(lyr, wname, d_R, sub['pdate'], lo, hi,
                                         sub[lyr + '_R'] * FIXED_RISK_DOLLARS, sub['stack_dollars']))

    # ---- L5: day-risk budget, B in {2,3,4} ----
    for B in (2, 3, 4):
        scale = apply_day_risk_budget(df, B)
        l5_R = scale * df['stack_R']
        for wname, (lo, hi) in WINDOWS.items():
            mask = df['window'] == wname
            sub_dates = df.loc[mask, 'pdate']
            d_R = l5_R[mask] - df.loc[mask, 'stack_R']
            reads.append(build_read_row(f'L5_budget{B}x', wname, d_R, sub_dates, lo, hi,
                                         l5_R[mask] * FIXED_RISK_DOLLARS, df.loc[mask, 'stack_dollars'],
                                         risk_multiple=scale[mask].mean(),
                                         notes='drawdown layer, not an EV layer (owner decides)'))

    # ---- L6: 2D tilt (RVOL tercile x price band), L7: SPY-context tilt ----
    def key_2d(row):
        a, b = row['bin_rvol_0935'], row['bin_price']
        return (a, b) if pd.notna(a) and pd.notna(b) else None

    l6_ev_gain = {}
    for direction_name, (fit_w, test_w) in [('VAL2026', ('TRAIN2025', 'VAL2026')), ('TRAIN2025', ('VAL2026', 'TRAIN2025'))]:
        fit_df, test_df = df[df['window'] == fit_w], df[df['window'] == test_w].copy()
        mult_map, cellmeans = fit_tilt(fit_df, key_2d)
        keys_test = test_df.apply(key_2d, axis=1)
        new_mult = keys_test.map(mult_map)
        new_mult = new_mult.fillna(test_df['tilt_mult'])
        test_df['L6_R'] = new_mult * test_df['idea43_R']
        d_R = test_df['L6_R'] - test_df['stack_R']
        ev_old = test_df['stack_R'].mean() / test_df['tilt_mult'].mean()
        ev_new = test_df['L6_R'].mean() / new_mult.mean()
        l6_ev_gain[test_w] = (ev_new - ev_old) / abs(ev_old) if ev_old else np.nan
        lo, hi = WINDOWS[test_w]
        reads.append(build_read_row('L6_tilt2D', test_w, d_R, test_df['pdate'], lo, hi,
                                     test_df['L6_R'] * FIXED_RISK_DOLLARS, test_df['stack_dollars'],
                                     risk_multiple=new_mult.mean(),
                                     notes=f'edges fit on {fit_w}, {len(cellmeans)}/9 cells, EV/risk gain {l6_ev_gain[test_w]:+.1%}'))
    # L6 robustness: cell-mean rank agreement between the two halves' OWN fits
    mult_train, cm_train = fit_tilt(df[df['window'] == 'TRAIN2025'], key_2d)
    mult_val, cm_val = fit_tilt(df[df['window'] == 'VAL2026'], key_2d)
    common_keys = sorted(set(cm_train.index) & set(cm_val.index))
    l6_spearman = spearman([cm_train[k] for k in common_keys], [cm_val[k] for k in common_keys]) if common_keys else np.nan
    logger.info('L6 2D-tilt robustness: %d/9 common cells, spearman=%.2f (pass>=%.1f)',
                len(common_keys), l6_spearman if not np.isnan(l6_spearman) else -9, PASS_SPEARMAN)

    def key_spy(edges):
        e1, e2 = edges
        def f(row):
            x = row['spy_ret_0935']
            if pd.isna(x):
                return None
            return 'low' if x <= e1 else ('mid' if x <= e2 else 'high')
        return f

    spy_edges = {}
    for wname in WINDOWS:
        s = df.loc[df['window'] == wname, 'spy_ret_0935'].dropna()
        spy_edges[wname] = (float(s.quantile(1 / 3)), float(s.quantile(2 / 3))) if len(s) >= 10 else (np.nan, np.nan)

    l7_ev_gain = {}
    for fit_w, test_w in [('TRAIN2025', 'VAL2026'), ('VAL2026', 'TRAIN2025')]:
        kf = key_spy(spy_edges[fit_w])
        fit_df, test_df = df[df['window'] == fit_w], df[df['window'] == test_w].copy()
        mult_map, cellmeans = fit_tilt(fit_df, kf)
        keys_test = test_df.apply(kf, axis=1)
        new_mult = keys_test.map(mult_map).fillna(test_df['tilt_mult'])
        test_df['L7_R'] = new_mult * test_df['idea43_R']
        d_R = test_df['L7_R'] - test_df['stack_R']
        ev_old = test_df['stack_R'].mean() / test_df['tilt_mult'].mean()
        ev_new = test_df['L7_R'].mean() / new_mult.mean()
        l7_ev_gain[test_w] = (ev_new - ev_old) / abs(ev_old) if ev_old else np.nan
        lo, hi = WINDOWS[test_w]
        reads.append(build_read_row('L7_tiltSPY', test_w, d_R, test_df['pdate'], lo, hi,
                                     test_df['L7_R'] * FIXED_RISK_DOLLARS, test_df['stack_dollars'],
                                     risk_multiple=new_mult.mean(),
                                     notes=f'edges fit on {fit_w} ({len(cellmeans)}/3 cells), EV/risk gain {l7_ev_gain[test_w]:+.1%}'))
    mult_train_spy, cm_train_spy = fit_tilt(df[df['window'] == 'TRAIN2025'], key_spy(spy_edges['TRAIN2025']))
    mult_val_spy, cm_val_spy = fit_tilt(df[df['window'] == 'VAL2026'], key_spy(spy_edges['VAL2026']))
    common_spy = sorted(set(cm_train_spy.index) & set(cm_val_spy.index))
    l7_spearman = spearman([cm_train_spy[k] for k in common_spy], [cm_val_spy[k] for k in common_spy]) if common_spy else np.nan
    logger.info('L7 SPY-tilt robustness: %d/3 common cells, spearman=%.2f (pass>=%.1f)',
                len(common_spy), l7_spearman if not np.isnan(l7_spearman) else -9, PASS_SPEARMAN)

    # ---- pass/fail per layer (both directions same-signed >= bar; tilts need spearman+EV gain too) ----
    reads_df = pd.DataFrame(reads)
    passing = []
    for lyr in simple_layers + ['L6_tilt2D', 'L7_tiltSPY']:
        rows = reads_df[reads_df['layer'] == lyr]
        ok = len(rows) == 2 and bool((rows['mean_dR'] >= PASS_DR).all() and (rows['ex_top5_dR'] >= PASS_EXTOP5).all())
        if lyr == 'L6_tilt2D':
            ok = (ok and (not np.isnan(l6_spearman)) and l6_spearman >= PASS_SPEARMAN
                  and all(v >= PASS_EV_GAIN for v in l6_ev_gain.values()))
        if lyr == 'L7_tiltSPY':
            ok = (ok and (not np.isnan(l7_spearman)) and l7_spearman >= PASS_SPEARMAN
                  and all(v >= PASS_EV_GAIN for v in l7_ev_gain.values()))
        if ok:
            passing.append(lyr)
    logger.info('layers passing the stack-join bar: %s', passing or 'NONE')

    # drawdown-layer check for L5 (EV cost <=0.02R, worst-week cut >=25%)
    dd_candidates = []
    for B in (2, 3, 4):
        rows = reads_df[reads_df['layer'] == f'L5_budget{B}x']
        ev_cost = -rows['mean_dR'].mean()
        cut = 1 - (rows['worst_week_dollars'].abs().mean() / max(rows['base_worst_week_dollars'].abs().mean(), 1e-9))
        if ev_cost <= 0.02 and cut >= 0.25:
            dd_candidates.append((B, ev_cost, cut))
    logger.info('L5 drawdown candidates (EV cost<=0.02R, worst-wk cut>=25%%): %s', dd_candidates)

    # ---- final stack: baseline + additive deltas of passing non-tilt layers; at most one of L6/L7 ----
    final_R = df['stack_R'].copy()
    final_dollars_col = df['stack_dollars'].copy()
    stack_desc = ['base+L1 tilt+L2 add (frozen)']
    for lyr in [l for l in passing if l in simple_layers]:
        final_R = final_R + (df[lyr + '_R'] - df['stack_R'])
        stack_desc.append(lyr)
    tilt_candidates = [l for l in passing if l in ('L6_tilt2D', 'L7_tiltSPY')]
    if tilt_candidates:
        best = max(tilt_candidates, key=lambda l: reads_df[reads_df['layer'] == l]['mean_dR'].mean())
        logger.info('tilt replacement in final stack: %s', best)
        stack_desc.append(best + ' (replaces L1 tilt)')
    final_dollars = final_R * FIXED_RISK_DOLLARS

    # ---- Q3 2026 weekly table: base vs final stack ----
    q3 = df[(df['pdate'] >= Q3_LO) & (df['pdate'] <= Q3_HI)].copy()
    q3['final_dollars'] = final_dollars.loc[q3.index]
    weekly_base = dict(cb.build_weekly_series([{'date': d, 'r': v} for d, v in zip(q3['pdate'], q3['stack_dollars'])], Q3_LO, Q3_HI))
    weekly_final = dict(cb.build_weekly_series([{'date': d, 'r': v} for d, v in zip(q3['pdate'], q3['final_dollars'])], Q3_LO, Q3_HI))
    fills_by_week = {}
    for d in q3['pdate']:
        wk = cb.week_monday(d)
        fills_by_week[wk] = fills_by_week.get(wk, 0) + 1
    weeks = sorted(weekly_base.keys())
    weekly_rows = []
    for i, wk in enumerate(weeks, 1):
        weekly_rows.append(dict(week_monday=str(wk), fills=fills_by_week.get(wk, 0),
                                 base_dollars=weekly_base[wk], final_dollars=weekly_final[wk],
                                 green_base='Y' if weekly_base[wk] > 0 else ('-' if fills_by_week.get(wk, 0) == 0 else 'N'),
                                 green_final='Y' if weekly_final[wk] > 0 else ('-' if fills_by_week.get(wk, 0) == 0 else 'N')))
    weekly_df = pd.DataFrame(weekly_rows)
    weekly_df.to_csv(WEEKLY_CSV, index=False)

    base_total = q3['stack_dollars'].sum()
    final_total = q3['final_dollars'].sum()
    base_green = sum(1 for w in weekly_rows if w['green_base'] == 'Y')
    final_green = sum(1 for w in weekly_rows if w['green_final'] == 'Y')
    base_worst, base_mdd = dollar_curve_stats(q3['pdate'], q3['stack_dollars'], Q3_LO, Q3_HI)
    final_worst, final_mdd = dollar_curve_stats(q3['pdate'], q3['final_dollars'], Q3_LO, Q3_HI)
    logger.info('Q3 2026: base total=$%.0f final total=$%.0f (n=%d fills, %d weeks)', base_total, final_total, len(q3), len(weeks))

    reads_df.to_csv(READS_CSV, index=False)

    # ---- RESULT.md ----
    lines = []
    lines.append('# RESULT 1,698: six further layers on the ORB stack (base + L1 tilt + L2 add)\n')
    lines.append(f'Reconstruction: {len(df)} usable fills of {len(pf)} in 1694_runners.csv '
                 f'(failures={fail}, idea43 cross-check mismatches={mismatch}). OOS2024H2 UNTESTED (n=0, not in this book).\n')
    lines.append('## Layer table (paired dR per fill vs the stack below it; both directions)\n')
    lines.append('| Layer | Dir | n | mean dR | day_t | ex_top5 dR | worst wk $ (base $) | risk x | pass |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for _, r in reads_df.iterrows():
        risk_str = 'n/a' if pd.isna(r['risk_multiple']) else f"{r['risk_multiple']:.2f}x"
        lines.append(f"| {r['layer']} | {r['direction']} | {r['n']} | {r['mean_dR']:+.3f} | {r['day_t']:.2f} | "
                     f"{r['ex_top5_dR']:+.3f} | ${r['worst_week_dollars']:,.0f} (${r['base_worst_week_dollars']:,.0f}) | "
                     f"{risk_str} | {'Y' if r['pass_dR_bar'] else 'n'} |")
    lines.append('')
    lines.append(f'L6 2D-tilt rank robustness (9 cells, TRAIN vs VAL fit): spearman={l6_spearman:+.2f} (bar >= {PASS_SPEARMAN})')
    lines.append(f'L7 SPY-tilt rank robustness (3 cells, TRAIN vs VAL fit): spearman={l7_spearman:+.2f} (bar >= {PASS_SPEARMAN})')
    lines.append(f'L5 drawdown candidates (worst-wk cut>=25% at EV cost<=0.02R): {dd_candidates or "none"}')
    lines.append('')
    lines.append(f'**Layers joining the final stack:** {", ".join(stack_desc)}')
    lines.append('')
    lines.append('## Q3 2026 weekly $: BASE (stack: base+L1 tilt+L2 add) vs FINAL STACK')
    lines.append('| Wk Mon | Fills | $Base | $Final |')
    lines.append('|---|---|---|---|')
    for w in weekly_rows:
        lines.append(f"| {w['week_monday']} | {w['fills']} | ${w['base_dollars']:,.0f} | ${w['final_dollars']:,.0f} |")
    lines.append('')
    lines.append(f"**Q3 totals**: Base ${base_total:,.0f} ({base_green}/{len(weeks)} green, worst wk ${base_worst:,.0f}, "
                 f"max DD ${base_mdd:,.0f}) vs Final ${final_total:,.0f} ({final_green}/{len(weeks)} green, "
                 f"worst wk ${final_worst:,.0f}, max DD ${final_mdd:,.0f}).")
    lines.append('')
    lines.append('**Caveats**: single quarter for the weekly table (n not an OOS claim); L6/L7 multipliers are '
                 'selection-half cell means on <=9 cells -- thin per-cell n; when >1 layer passes the final stack '
                 'sums each layer\'s OWN delta additively (not a jointly re-optimized combination); L5 is reported '
                 'as a drawdown layer, never counted toward the EV pass bar; tilt and add are layered on the SAME '
                 'fills throughout (lever-isolation, not independently-selected edges).')
    with open(RESULT_MD, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    logger.info('wrote %s (%d lines)', RESULT_MD, len(lines))
    logger.info('DONE')


if __name__ == '__main__':
    main()
