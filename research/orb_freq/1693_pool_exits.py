#!/usr/bin/env python3
"""Cell 1,693: every ORB sub-pool x its OWN exit, both selection directions,
plus (Amendment 1) production split by one admission number at a time.

PREREG: research/orb_freq/PREREG_1693.md (FROZEN 2026-10-01 13:20 UTC,
Amendment 1 13:35 UTC). Owner's verbatim ask: "we need increased frequency
on additional sub-pools, each with its criteria/filter and its own exit
strategy."

Reuses (imported read-only via importlib, unchanged -- module names start
with a digit so cannot be `import`ed normally, project convention):
  research/orb_exit/1679_orb_exit.py (f1679) -- find_range_and_breakout
    (entry-minute reconstruction: book's own entry_price, stop = the
    opening-range low), _lock_walk (the live lock-stop walker, reused
    verbatim for E1/E7/E8 and as the "rest" leg of E5/E6), and the ORB
    constants (ORB_EOD_M=945 i.e. 15:45 ET, EXIT_SLIP_BPS=10bps,
    R_FLOOR_PCT=0.5%, LOCK_TRIGGER_R_LIVE=1.75, LOCK_STOP_R_LIVE=0.5 --
    all orb.yaml values, read-only, never written here).
  research/hod_entry/1668_failure.py (f1668) -- BarStore (bars_sip.db,
    read-only) and BARS_DB.
  research/hod_entry/1677_take_profit.py (f1677) -- stats_block (n,
    mean_dR, iid_t, day_t, ex_top5, mde -- the project's canonical
    day-clustered-t/iid-t/ex-top-5%/MDE primitive, itself built on
    1676_shapes.py; NOT reimplemented here).
  scripts/cadence_bar.py (cb) -- build_weekly_series/week_monday/percentile
    (weekly bucketing, zero-fill weeks), compute_cycles/score_c1 (strong-week
    gap), score_c3 (weekly P10/min/max-drawdown/underwater), score_c4
    (count-matched sign-shuffle green-week null).
  trading/orb_csv.py (read-only parse utility; "every ORB CSV goes through
    orb_csv.read_orb_csv" -- CLAUDE.md -- ticker literal "NA" is not a NaN).

Two seeds from the original PREREG (2-3% gap band x F1/F3/F4/F5/F6, and F2
premarket $ volume for bands B/C) are NOT built: both require NEW minute bars
via the bars_sip.db appender for a candidate universe that does not exist in
any CSV on disk (cells 1684/1685/1689a each needed 5-6 separate pipeline
files to build their own universe) -- infeasible inside this cell's call
budget, and no owner GO for a new data pull (project rule: "no new paid data
pull without his GO, evidence first"). Said explicitly in RESULT_1693.md;
everything else runs. Amendment 1's production-slice "pre-market dollar
volume" dimension is DIFFERENT and IS computed: it reuses bars already
queried for entry reconstruction (no new fetch), gated on measured coverage.

Usage:
    nice -n 10 python3 research/orb_freq/1693_pool_exits.py
"""
import importlib.util
import logging
import os
import sys
import time
from datetime import date as _date, datetime, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

LOG_FILE = os.path.join(HERE, '1693_pool_exits.log')
READS_CSV = os.path.join(HERE, '1693_reads.csv')
UNION_CSV = os.path.join(HERE, '1693_union.csv')
RESULT_MD = os.path.join(HERE, 'RESULT_1693.md')

logger = logging.getLogger('cell1693')


def setup_logging():
    """Verbose progress to both the log file and stdout (CLAUDE.md: batch
    processors are verbose; find the root cause in the logging)."""
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()])


def check_disk(floor_gb=5.0):
    """Hard disk-space floor, matching 1679's own guard. No seed build or
    append happens before this -- we build none in this cell, but the same
    discipline applies to the (smaller) outputs this cell does write."""
    st = os.statvfs('/')
    free_gb = st.f_bavail * st.f_frsize / (1024 ** 3)
    logger.info('disk free on /: %.1f GB', free_gb)
    if free_gb < floor_gb:
        logger.error('disk free %.1f GB < %.1f GB floor -- aborting', free_gb, floor_gb)
        sys.exit(1)
    return free_gb


def _load_module(name, fname, root=ROOT):
    """Load a digit-prefixed cell script as a module, verbatim pattern from
    research/orb_exit/1679_orb_exit.py._load_module."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, fname))
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = old_argv
    return mod


# ---------------------------------------------------------------------------
# Windows (fixed, never inferred from data -- CLAUDE.md statistics rule)
# ---------------------------------------------------------------------------
TRAIN_LO, TRAIN_HI = _date(2025, 1, 1), _date(2025, 12, 31)
VAL_LO, VAL_HI = _date(2026, 1, 1), _date(2026, 9, 18)
OOS_LO, OOS_HI = _date(2024, 7, 1), _date(2024, 12, 31)
WINDOWS = {'TRAIN2025': (TRAIN_LO, TRAIN_HI), 'VAL2026': (VAL_LO, VAL_HI), 'OOS2024H2': (OOS_LO, OOS_HI)}
SELECT_MEAN_R, SELECT_T = 0.05, 1.5
FIXED_RISK_DOLLARS = 375.0   # project-wide ORB per-trade risk convention (L2_RISK / ORB_LIVE_RISK)
STRONG_WEEK_R = 5.0          # docs/cadence_bar.md


def pdate(s):
    return datetime.strptime(str(s)[:10], '%Y-%m-%d').date()


def assign_window(d):
    if TRAIN_LO <= d <= TRAIN_HI:
        return 'TRAIN2025'
    if VAL_LO <= d <= VAL_HI:
        return 'VAL2026'
    if OOS_LO <= d <= OOS_HI:
        return 'OOS2024H2'
    return None


def window_weeks(lo, hi):
    return (hi - lo).days / 7.0


# ---------------------------------------------------------------------------
# Reused modules
# ---------------------------------------------------------------------------
f1679 = _load_module('f1679_1693', 'research/orb_exit/1679_orb_exit.py')
f1668 = _load_module('f1668_1693', 'research/hod_entry/1668_failure.py')
f1677 = _load_module('f1677_1693', 'research/hod_entry/1677_take_profit.py')
cb = _load_module('cadence_bar_1693', 'scripts/cadence_bar.py')
orb_csv = _load_module('orb_csv_1693', 'trading/orb_csv.py')

find_range_and_breakout = f1679.find_range_and_breakout
_lock_walk = f1679._lock_walk
ORB_EOD_M = f1679.ORB_EOD_M
EXIT_SLIP_BPS = f1679.EXIT_SLIP_BPS
R_FLOOR_PCT = f1679.R_FLOOR_PCT
LOCK_TRIGGER_R_LIVE = f1679.LOCK_TRIGGER_R_LIVE
LOCK_STOP_R_LIVE = f1679.LOCK_STOP_R_LIVE
stats_block = f1677.stats_block
ORB_BOOK_CSV = f1679.ORB_BOOK_CSV


class CachedStore:
    """f1668.BarStore wrapped with a process-wide (symbol, date) cache: the
    18 general pools and 18 production slices share many of the same
    underlying (symbol, date) fills, so this turns N pool memberships of the
    same fill into one SQLite query."""

    def __init__(self, path):
        self._store = f1668.BarStore(path)
        self._cache = {}
        self.hits = 0
        self.misses = 0

    def day_bars(self, symbol, day):
        key = (symbol, str(day))
        if key not in self._cache:
            self._cache[key] = self._store.day_bars(symbol, str(day))
            self.misses += 1
        else:
            self.hits += 1
        return self._cache[key]

    def close(self):
        self._store.close()


# ---------------------------------------------------------------------------
# Entry/stop reconstruction (f1679's own rule, reused verbatim)
# ---------------------------------------------------------------------------

def reconstruct_fill(store, symbol, date_str, entry_price):
    """Entry MINUTE = the breakout bar (first bar in [09:35,10:35) ET whose
    high > the opening-range high); entry price = the book's own value (not
    re-derived); stop = the opening-range low -- f1679's rule, byte-identical
    call. Returns (dict-or-None, reason-or-None)."""
    bars = store.day_bars(symbol, date_str)
    if bars is None or len(bars['o']) < 5:
        return None, 'no_bars'
    rec = find_range_and_breakout(bars)
    if rec is None:
        return None, 'no_range_or_breakout'
    i0 = rec['i0']
    entry = float(entry_price)
    stop = rec['range_low']
    R_unit = entry - stop
    if not (R_unit > 0):
        return None, 'bad_R'
    if R_unit / entry < R_FLOOR_PCT:
        return None, 'below_R_floor'
    return dict(bars=bars, i0=i0, entry=entry, stop=stop, R_unit=R_unit), None


# ---------------------------------------------------------------------------
# Exit menu E1-E12. E1/E7/E8 call f1679._lock_walk verbatim (returns a PRICE,
# wrapped to R by _R below). E5/E6's "rest" leg is f1679._lock_walk too,
# walked over the FULL remaining path independently of the scale touch --
# the exact design of 1679's own scale50_1R_plus_live. E2-E4/E9-E12 are new
# (not in 1679's menu) and follow the SAME stop-priority-on-a-tie and -10bps
# slippage convention as every walker in this codebase (f1668.walk_k's own
# docstring: "a stop or target touch takes precedence ... stop wins a tie").
# ---------------------------------------------------------------------------

def _R(px, entry, R_unit):
    return (px - entry) / R_unit if px is not None else np.nan


def _target_only_walk(bars, i0, entry, stop, R_unit, target_r, eod_m=ORB_EOD_M, slip_bps=EXIT_SLIP_BPS):
    """E2-E4: fixed target, original stop, no trailing. Stop wins a same-bar tie."""
    n = len(bars['o'])
    if i0 + 1 >= n:
        return np.nan
    slip = slip_bps / 10000.0
    target_lvl = entry + target_r * R_unit
    for j in range(i0 + 1, n):
        if bars['minarr'][j] >= eod_m:
            return _R(float(bars['c'][j]) * (1 - slip), entry, R_unit)
        bh, bl = float(bars['h'][j]), float(bars['l'][j])
        if bl <= stop:
            return _R(stop * (1 - slip), entry, R_unit)
        if bh >= target_lvl:
            return _R(target_lvl * (1 - slip), entry, R_unit)
    return _R(float(bars['c'][n - 1]) * (1 - slip), entry, R_unit)


def _scale_then_lock_walk(bars, i0, entry, stop, R_unit, scale_r, eod_m=ORB_EOD_M, slip_bps=EXIT_SLIP_BPS):
    """E5/E6: 50% out at +scale_r R (touch, with slip), the other 50% rides
    f1679._lock_walk's FULL independent path (own share lot -- matches
    1679's scale50_1R_plus_live exactly, including: if the scale level is
    never touched before being stopped out, result == the rest leg alone)."""
    n = len(bars['o'])
    if i0 + 1 >= n:
        return np.nan
    slip = slip_bps / 10000.0
    px_e = _lock_walk(bars, i0, entry, stop, R_unit, LOCK_TRIGGER_R_LIVE, LOCK_STOP_R_LIVE, eod_m, slip_bps)
    if px_e is None:
        return np.nan
    leg2_R = (px_e - entry) / R_unit
    touch_j = None
    for j in range(i0 + 1, n):
        if bars['minarr'][j] >= eod_m:
            break
        if float(bars['l'][j]) <= stop:
            break
        if float(bars['h'][j]) >= entry + scale_r * R_unit:
            touch_j = j
            break
    if touch_j is None:
        return leg2_R
    leg1_px = (entry + scale_r * R_unit) * (1 - slip)
    leg1_R = (leg1_px - entry) / R_unit
    return 0.5 * leg1_R + 0.5 * leg2_R


def _trail_gated_walk(bars, i0, entry, stop, R_unit, gate_r, trail_r, eod_m=ORB_EOD_M, slip_bps=EXIT_SLIP_BPS):
    """E9: stop stays at the original until MFE first reaches +gate_r R; from
    then on it trails at running-high - trail_r*R (never lowered)."""
    n = len(bars['o'])
    if i0 + 1 >= n:
        return np.nan
    slip = slip_bps / 10000.0
    run_high, stop_price, gated = entry, stop, False
    for j in range(i0 + 1, n):
        if bars['minarr'][j] >= eod_m:
            return _R(float(bars['c'][j]) * (1 - slip), entry, R_unit)
        bh, bl = float(bars['h'][j]), float(bars['l'][j])
        run_high = max(run_high, bh)
        if run_high >= entry + gate_r * R_unit:
            gated = True
        if gated:
            stop_price = max(stop_price, run_high - trail_r * R_unit)
        if bl <= stop_price:
            return _R(stop_price * (1 - slip), entry, R_unit)
    return _R(float(bars['c'][n - 1]) * (1 - slip), entry, R_unit)


def _trail_10m_low_walk(bars, i0, entry, stop, R_unit, gate_r=1.0, window=10, eod_m=ORB_EOD_M, slip_bps=EXIT_SLIP_BPS):
    """E10: once MFE >= +gate_r R, the stop ratchets up to the rolling
    `window`-minute low (inclusive, ending at the current bar), never lowered."""
    n = len(bars['o'])
    if i0 + 1 >= n:
        return np.nan
    slip = slip_bps / 10000.0
    run_high, stop_price, gated = entry, stop, False
    for j in range(i0 + 1, n):
        if bars['minarr'][j] >= eod_m:
            return _R(float(bars['c'][j]) * (1 - slip), entry, R_unit)
        bh, bl = float(bars['h'][j]), float(bars['l'][j])
        run_high = max(run_high, bh)
        if run_high >= entry + gate_r * R_unit:
            gated = True
        if gated:
            lo_start = max(i0 + 1, j - window + 1)
            lo10 = float(bars['l'][lo_start:j + 1].min())
            stop_price = max(stop_price, lo10)
        if bl <= stop_price:
            return _R(stop_price * (1 - slip), entry, R_unit)
    return _R(float(bars['c'][n - 1]) * (1 - slip), entry, R_unit)


def _time_stop_walk(bars, i0, entry, stop, R_unit, k=60, mtm_thresh=0.5, eod_m=ORB_EOD_M, slip_bps=EXIT_SLIP_BPS):
    """E11: stop-only (no lock) until bar i0+k; at that bar, if mark-to-market
    (this bar's OPEN, the causal/executable price) is < mtm_thresh R, exit
    there; else keep riding stop-only to EOD."""
    n = len(bars['o'])
    if i0 + 1 >= n:
        return np.nan
    slip = slip_bps / 10000.0
    for j in range(i0 + 1, n):
        if bars['minarr'][j] >= eod_m:
            return _R(float(bars['c'][j]) * (1 - slip), entry, R_unit)
        if float(bars['l'][j]) <= stop:
            return _R(stop * (1 - slip), entry, R_unit)
        if j >= i0 + k:
            mtm_R = (float(bars['o'][j]) - entry) / R_unit
            if mtm_R < mtm_thresh:
                return _R(float(bars['o'][j]) * (1 - slip), entry, R_unit)
    return _R(float(bars['c'][n - 1]) * (1 - slip), entry, R_unit)


def _power_hour_walk(bars, i0, entry, stop, R_unit, trigger_r=LOCK_TRIGGER_R_LIVE,
                      lock_stop_r=LOCK_STOP_R_LIVE, power_m=900.0, eod_m=ORB_EOD_M, slip_bps=EXIT_SLIP_BPS):
    """E12: E1's own lock mechanics, EXCEPT from 15:00 ET (power_m=900) the
    stop-touch check is suspended on any bar where the close is already
    >= +1R (so a profitable power-hour move isn't stopped out early); the
    position then rides to the 15:45 close."""
    n = len(bars['o'])
    if i0 + 1 >= n:
        return np.nan
    slip = slip_bps / 10000.0
    trigger_lvl, lock_stop = entry + trigger_r * R_unit, entry + lock_stop_r * R_unit
    stop_price, armed = stop, False
    for j in range(i0 + 1, n):
        m = bars['minarr'][j]
        if m >= eod_m:
            return _R(float(bars['c'][j]) * (1 - slip), entry, R_unit)
        bh, bl, bc = float(bars['h'][j]), float(bars['l'][j]), float(bars['c'][j])
        if not armed and bh >= trigger_lvl:
            armed = True
            stop_price = max(stop_price, lock_stop)
        mtm_R = (bc - entry) / R_unit
        if m >= power_m and mtm_R >= 1.0:
            continue
        if bl <= stop_price:
            return _R(stop_price * (1 - slip), entry, R_unit)
    return _R(float(bars['c'][n - 1]) * (1 - slip), entry, R_unit)


EXITS = {
    'E1_production': lambda b, i0, e, s, R: _R(_lock_walk(b, i0, e, s, R, LOCK_TRIGGER_R_LIVE, LOCK_STOP_R_LIVE), e, R),
    'E2_target1R': lambda b, i0, e, s, R: _target_only_walk(b, i0, e, s, R, 1.0),
    'E3_target1_5R': lambda b, i0, e, s, R: _target_only_walk(b, i0, e, s, R, 1.5),
    'E4_target3R': lambda b, i0, e, s, R: _target_only_walk(b, i0, e, s, R, 3.0),
    'E5_scale50_1R': lambda b, i0, e, s, R: _scale_then_lock_walk(b, i0, e, s, R, 1.0),
    'E6_scale50_1_5R': lambda b, i0, e, s, R: _scale_then_lock_walk(b, i0, e, s, R, 1.5),
    'E7_BE_lock_1R': lambda b, i0, e, s, R: _R(_lock_walk(b, i0, e, s, R, 1.0, 0.0), e, R),
    'E8_lock0_5R_at1R': lambda b, i0, e, s, R: _R(_lock_walk(b, i0, e, s, R, 1.0, 0.5), e, R),
    'E9_trail_MFE1R': lambda b, i0, e, s, R: _trail_gated_walk(b, i0, e, s, R, 1.0, 1.0),
    'E10_trail10mLow': lambda b, i0, e, s, R: _trail_10m_low_walk(b, i0, e, s, R),
    'E11_timestop60m': lambda b, i0, e, s, R: _time_stop_walk(b, i0, e, s, R),
    'E12_powerHour': lambda b, i0, e, s, R: _power_hour_walk(b, i0, e, s, R),
}
EXIT_NAMES = list(EXITS.keys())


# ---------------------------------------------------------------------------
# Per-pool scoring + both-direction classification
# ---------------------------------------------------------------------------

def build_per_fill_table(rows, store, label):
    """rows: iterable of (date_str, symbol, entry_price). Reconstructs each
    fill once and computes all 12 exits. Returns (DataFrame, recon_counter)."""
    recon = {'no_bars': 0, 'no_range_or_breakout': 0, 'bad_R': 0, 'below_R_floor': 0, 'ok': 0}
    out = []
    t0 = time.time()
    for n_seen, (date_str, symbol, entry_price) in enumerate(rows, 1):
        rec, reason = reconstruct_fill(store, symbol, date_str, entry_price)
        if rec is None:
            recon[reason] += 1
            continue
        recon['ok'] += 1
        d = pdate(date_str)
        win = assign_window(d)
        row = {'date': d, 'window': win, 'symbol': symbol, 'entry': rec['entry'], 'R_unit': rec['R_unit']}
        for ename, efn in EXITS.items():
            row[ename] = efn(rec['bars'], rec['i0'], rec['entry'], rec['stop'], rec['R_unit'])
        out.append(row)
        if n_seen % 1000 == 0:
            logger.info('%s: reconstructed %d/%d fills (%.0fs)', label, n_seen, len(rows) if hasattr(rows, '__len__') else n_seen, time.time() - t0)
    logger.info('%s: reconstruction done n=%d %s', label, n_seen if rows else 0, recon)
    return pd.DataFrame(out), recon


def reads_for_exit_series(vals, dates, lo, hi):
    """stats_block + weekly P10/worst-week (cb's own zero-filled weekly
    bucketing) for one (pool, exit, window) cell."""
    st = stats_block(vals.values if hasattr(vals, 'values') else np.asarray(vals),
                      dates.values if hasattr(dates, 'values') else np.asarray(dates))
    wk = window_weeks(lo, hi)
    if len(vals):
        trades = [{'date': d, 'r': v} for d, v in zip(dates, vals)]
        weekly = cb.build_weekly_series(trades, lo, hi)
        wr = [w[1] for w in weekly]
        p10 = cb.percentile(wr, 10)
        worst = min(wr)
    else:
        p10, worst = np.nan, np.nan
    return dict(n=st['n'], fills_per_week=(st['n'] / wk if wk else np.nan), mean_R=st['mean_dR'],
                iid_t=st['iid_t'], day_t=st['day_t'], ex_top5=st['ex_top5'], mde=st['mde'],
                weekly_p10_R=p10, worst_week_R=worst)


def score_table(pf, pool_name):
    """Builds the 12-exit x 3-window reads table for one pool's per-fill table."""
    rows = []
    for ename in EXIT_NAMES:
        for wname, (lo, hi) in WINDOWS.items():
            sub = pf[pf['window'] == wname] if len(pf) else pf
            vals = sub[ename].dropna() if len(sub) else pd.Series([], dtype=float)
            dts = sub.loc[vals.index, 'date'] if len(vals) else pd.Series([], dtype=object)
            r = reads_for_exit_series(vals, dts, lo, hi)
            r.update(pool=pool_name, exit=ename, window=wname)
            rows.append(r)
    return pd.DataFrame(rows)


def classify_pool(reads_df, pool_name):
    """Direction A: select on TRAIN2025 (mean_R>=0.05, day_t>=1.5), confirm on
    VAL2026 (mean_R>0, ex_top5>0, >=1 fill/wk). Direction B: mirror, 2026->2025.
    robust = both directions confirm; regime_specific = exactly one; else fails."""
    out = []
    for ename in EXIT_NAMES:
        tr = reads_df[(reads_df.exit == ename) & (reads_df.window == 'TRAIN2025')].iloc[0]
        va = reads_df[(reads_df.exit == ename) & (reads_df.window == 'VAL2026')].iloc[0]
        oos = reads_df[(reads_df.exit == ename) & (reads_df.window == 'OOS2024H2')].iloc[0]
        selA = bool(tr.n > 0 and tr.mean_R >= SELECT_MEAN_R and tr.day_t >= SELECT_T)
        confA = bool(selA and va.n > 0 and va.mean_R > 0 and va.ex_top5 > 0 and va.fills_per_week >= 1)
        selB = bool(va.n > 0 and va.mean_R >= SELECT_MEAN_R and va.day_t >= SELECT_T)
        confB = bool(selB and tr.n > 0 and tr.mean_R > 0 and tr.ex_top5 > 0 and tr.fills_per_week >= 1)
        cls = 'robust' if (confA and confB) else ('regime_specific' if (confA or confB) else 'fails')
        out.append(dict(pool=pool_name, exit=ename, selA=selA, confA=confA, selB=selB, confB=confB,
                         classification=cls, train_mean_R=tr.mean_R, train_t=tr.day_t, train_n=tr.n,
                         val_mean_R=va.mean_R, val_t=va.day_t, val_n=va.n,
                         oos_mean_R=oos.mean_R, oos_n=oos.n))
    return pd.DataFrame(out)


def pool_summary_line(cls_df, pool_name):
    """One RESULT.md line per pool: best exit per direction + overall class."""
    best_cls = 'fails'
    for want in ('robust', 'regime_specific'):
        if (cls_df.classification == want).any():
            best_cls = want
            break
    bestA = cls_df[cls_df.confA].sort_values('val_mean_R', ascending=False)
    bestB = cls_df[cls_df.confB].sort_values('train_mean_R', ascending=False)
    aline = (f"{bestA.iloc[0]['exit']} (TRAIN {bestA.iloc[0]['train_mean_R']:+.3f}R t{bestA.iloc[0]['train_t']:.1f} -> "
             f"VAL {bestA.iloc[0]['val_mean_R']:+.3f}R n{int(bestA.iloc[0]['val_n'])})") if len(bestA) else 'none select'
    bline = (f"{bestB.iloc[0]['exit']} (VAL {bestB.iloc[0]['val_mean_R']:+.3f}R t{bestB.iloc[0]['val_t']:.1f} -> "
             f"TRAIN {bestB.iloc[0]['train_mean_R']:+.3f}R n{int(bestB.iloc[0]['train_n'])})") if len(bestB) else 'none select'
    oos_mean = cls_df['oos_mean_R'].mean()
    return f"- **{pool_name}** [{best_cls}] dirA: {aline} | dirB: {bline} | mean OOS-2024H2 R across exits: {oos_mean:+.3f}"


def append_result_md(text):
    with open(RESULT_MD, 'a') as fh:
        fh.write(text)
        fh.flush()


# ---------------------------------------------------------------------------
# Production admission-feature slices (Amendment 1)
# ---------------------------------------------------------------------------

def premkt_dollar_vol(bars):
    """Sum(close * volume) over bars before 09:30 ET (minarr < 570). NaN if
    this (symbol, date) has no bars before the open -- no premarket coverage,
    counted toward the dimension's availability rail, not silently zeroed."""
    mask = bars['minarr'] < 570.0
    if not mask.any():
        return np.nan
    return float((bars['c'][mask] * bars['v'][mask]).sum())


def load_production_fills(store):
    """analysis_results/orb_bplus_book.csv, entered==1, via trading/orb_csv
    (read-only; CLAUDE.md: every ORB CSV goes through read_orb_csv). Builds
    the 12-exit table ONCE plus the 6 admission features per fill (so the 18
    slices below cost zero extra bar lookups or exit walks)."""
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
        d = pdate(r.date)
        win = assign_window(d)
        rvol_0935 = (r.range_total_volume / r.avg_daily_volume_20d) if r.avg_daily_volume_20d > 0 else np.nan
        row = {'date': d, 'window': win, 'symbol': r.symbol, 'entry': rec['entry'], 'R_unit': rec['R_unit'],
               'gap_pct': r.gap_pct, 'price': rec['entry'], 'range5m_pct': r.range_size_pct,
               'prior_day_volume': r.prev_day_volume_vs_20d, 'rvol_0935': rvol_0935,
               'premkt_dollar_vol': premkt_dollar_vol(rec['bars'])}
        for ename, efn in EXITS.items():
            row[ename] = efn(rec['bars'], rec['i0'], rec['entry'], rec['stop'], rec['R_unit'])
        rows.append(row)
        if n_seen % 200 == 0:
            logger.info('production: reconstructed %d/%d (%.0fs)', n_seen, len(ent), time.time() - t0)
    logger.info('production: reconstruction done n=%d %s', len(ent), recon)
    pf = pd.DataFrame(rows)
    premkt_cov = pf['premkt_dollar_vol'].notna().mean() if len(pf) else 0.0
    logger.info('production: pre-market $ volume coverage = %.1f%% (availability rail: 80%%)', 100 * premkt_cov)
    return pf, recon, premkt_cov


DIMENSIONS_FIXED = {
    'gap_size': ('gap_pct', [(5.0, 7.0), (7.0, 10.0), (10.0, float('inf'))], ['5-7%', '7-10%', '>=10%']),
    'price': ('price', [(3.0, 10.0), (10.0, 20.0), (20.0, 30.0)], ['$3-10', '$10-20', '$20-30']),
}
DIMENSIONS_TERCILE = ['prior_day_volume', 'range5m_pct', 'rvol_0935', 'premkt_dollar_vol']


def tercile_edges(pf_train, col):
    s = pf_train[col].dropna()
    if len(s) < 10:
        return None
    return float(s.quantile(1 / 3)), float(s.quantile(2 / 3))


def make_slices(pf, premkt_cov):
    """18 slices: 2 fixed-bin dimensions x 3 bins + 4 tercile dimensions
    (TRAIN-2025-only cutpoints, applied unchanged to VAL/OOS) x 3 bins.
    pre-market $ volume is VOID (dropped) if measured coverage < 80%."""
    slices = {}
    train = pf[pf['window'] == 'TRAIN2025']
    for dim, (col, bins, labels) in DIMENSIONS_FIXED.items():
        for (lo, hi), lab in zip(bins, labels):
            mask = (pf[col] >= lo) & (pf[col] < hi)
            slices[f'{dim}_{lab}'] = pf[mask]
    for col in DIMENSIONS_TERCILE:
        if col == 'premkt_dollar_vol' and premkt_cov < 0.80:
            logger.warning('VOID: premkt_dollar_vol coverage %.1f%% < 80%% rail -- dimension dropped', 100 * premkt_cov)
            continue
        edges = tercile_edges(train, col)
        if edges is None:
            logger.warning('VOID: %s has <10 non-null TRAIN2025 rows -- dimension dropped', col)
            continue
        e1, e2 = edges
        bins = [(-float('inf'), e1), (e1, e2), (e2, float('inf'))]
        labels = ['low', 'mid', 'high']
        for (lo, hi), lab in zip(bins, labels):
            mask = (pf[col] >= lo) & (pf[col] < hi)
            slices[f'{col}_{lab}'] = pf[mask]
    return slices


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    setup_logging()
    logger.info('=== cell 1,693: ORB sub-pool x exit, both directions -- starting ===')
    check_disk()

    with open(RESULT_MD, 'w') as fh:
        fh.write(f"# RESULT 1,693 -- every ORB sub-pool x its own exit (run started {datetime.now().isoformat()})\n\n"
                 "PREREG: research/orb_freq/PREREG_1693.md (FROZEN, Amendment 1). Written incrementally.\n\n"
                 "## Seeds NOT built (declared per PREREG's own escape clause)\n"
                 "Both original missing seeds (2-3% gap band x F1/F3/F4/F5/F6; F2 premarket $ volume for bands B/C) "
                 "require NEW minute bars via the bars_sip.db appender for a candidate universe that exists in no "
                 "CSV on disk -- cells 1684/1685/1689a each needed 5-6 pipeline files to build their own universe "
                 "from scratch. Infeasible inside this cell's call budget alongside the mandatory 12-exit grid, and "
                 "there is no owner GO for a new data pull. Everything else below runs on existing data.\n\n")

    store = CachedStore(f1668.BARS_DB)
    all_reads, all_cls = [], []

    # --- validation: does the reconstructed opening range match the book's OWN
    # range_size_pct feature (computed by the original backtest, independent of
    # this script)? This is the meaningful check -- NOT a comparison of E1's R
    # to 1684_pool_books.csv's own 'R' column: that column turned out (checked
    # directly: _sized_pnl/R == $375.00 exactly and the implied R_unit/entry ==
    # 11.25% of price on EVERY row, std 1e-13) to be a FIXED-11.25%-of-price
    # notional risk unit 1684 used for cross-pool comparability across
    # idea1/idea2/.../prod, not the true opening-range stop -- comparing E1's
    # range-low-based R to that column compares two different R denominators
    # by construction and was dropped as the wrong test.
    logger.info('--- validation: reconstructed opening-range % vs orb_bplus_book.csv range_size_pct (ground truth) ---')
    bplus_val = orb_csv.read_orb_csv(ORB_BOOK_CSV)
    bplus_val = bplus_val[bplus_val['entered'] == 1].copy()
    bplus_val['date'] = bplus_val['date'].astype(str)
    my_pct, book_pct, n_fail = [], [], 0
    for r in bplus_val.itertuples():
        rec, reason = reconstruct_fill(store, r.symbol, r.date, r.entry_price)
        if rec is None:
            n_fail += 1
            continue
        my_pct.append(rec['R_unit'] / rec['entry'] * 100.0)
        book_pct.append(r.range_size_pct)
    my_pct, book_pct = np.array(my_pct), np.array(book_pct)
    corr = np.corrcoef(my_pct, book_pct)[0, 1] if len(my_pct) > 1 else np.nan
    val_line = (f"n={len(my_pct)}/{len(bplus_val)} reconstructed (coverage {100*len(my_pct)/len(bplus_val):.1f}%, "
                f"{n_fail} dropped), corr(my opening-range %, book's range_size_pct)={corr:.4f}, "
                f"mean diff={np.mean(my_pct - book_pct):+.3f}pp (my range measured to ENTRY, book's to range_high -- "
                f"a small, structural, expected offset, not noise)")
    logger.info('validation: %s', val_line)
    append_result_md(f"## Validation (reconstructed opening-range % vs the book's OWN range_size_pct ground truth)\n{val_line}\n\n")

    # --- 18 general pools (1684/1685/1689a) ---
    append_result_md("## General pools (18) -- best exit per direction, classification\n")
    books = {
        '1684': ('1684_pool_books.csv', ['idea1', 'idea2', 'idea10', 'idea11']),
        '1685': ('1685_pool_books.csv', ['AF6', 'BF1', 'BF3', 'BF4', 'BF5', 'BF6', 'CF1', 'CF3', 'CF4', 'CF5', 'CF6']),
        '1689a': ('1689a_pool_books.csv', ['19PRIME', '19', '20']),
    }
    general_pf = {}
    for src, (fname, pools) in books.items():
        dfb = pd.read_csv(os.path.join(HERE, fname))
        for pool in pools:
            sub = dfb[dfb['pool'] == pool]
            rows = list(zip(sub['date'], sub['symbol'], sub['entry_price']))
            pf, recon = build_per_fill_table(rows, store, f'{src}/{pool}')
            general_pf[pool] = pf
            rt = score_table(pf, pool)
            ct = classify_pool(rt, pool)
            all_reads.append(rt)
            all_cls.append(ct)
            append_result_md(pool_summary_line(ct, pool) + f" (n_fills reconstructed={recon['ok']}, dropped={sum(v for k, v in recon.items() if k != 'ok')})\n")
            logger.info('%s done', pool)
    append_result_md("\n")

    # --- production + 18 admission-feature slices (Amendment 1) ---
    logger.info('--- production book + admission-feature slices ---')
    pf_prod, prod_recon, premkt_cov = load_production_fills(store)
    slices = make_slices(pf_prod, premkt_cov)
    append_result_md(f"## Production ({prod_recon['ok']} fills reconstructed) -- admission-feature slices "
                      f"({len(slices)} of 18 computed; premkt $ vol coverage={100*premkt_cov:.1f}%)\n")
    robust_slices = {}
    for slice_name, sub in slices.items():
        rt = score_table(sub, slice_name)
        ct = classify_pool(rt, slice_name)
        all_reads.append(rt)
        all_cls.append(ct)
        is_robust_both = bool((ct.classification == 'robust').any())
        if is_robust_both:
            best = ct[ct.classification == 'robust'].sort_values('val_mean_R', ascending=False).iloc[0]
            robust_slices[slice_name] = best['exit']
        append_result_md(pool_summary_line(ct, slice_name) + f" (n={len(sub)})\n")
    append_result_md(f"\nSlices promoted (ROBUST both directions -> replaces production's exit for that slice only): "
                      f"{robust_slices if robust_slices else 'none'}\n\n")

    # --- assemble reads/union CSVs ---
    reads_df = pd.concat(all_reads, ignore_index=True)
    cls_df = pd.concat(all_cls, ignore_index=True)
    reads_df.to_csv(READS_CSV, index=False)
    logger.info('wrote %s (%d rows)', READS_CSV, len(reads_df))

    robust_pairs = cls_df[cls_df.classification == 'robust']
    regime_pairs = cls_df[cls_df.classification == 'regime_specific']
    append_result_md("## Robust pairs (both directions confirm) -- general pools\n")
    for r in robust_pairs.itertuples():
        append_result_md(f"- {r.pool} x {r.exit}: TRAIN {r.train_mean_R:+.3f}R (t{r.train_t:.1f}, n{r.train_n}) / "
                          f"VAL {r.val_mean_R:+.3f}R (t{r.val_t:.1f}, n{r.val_n}) / OOS2024H2 {r.oos_mean_R:+.3f}R (n{r.oos_n})\n")
    if not len(robust_pairs):
        append_result_md("- none\n")
    append_result_md("\n## Regime-specific pairs (one direction only; reported, not shipped) -- general pools\n")
    for r in regime_pairs.itertuples():
        append_result_md(f"- {r.pool} x {r.exit}: TRAIN {r.train_mean_R:+.3f}R (t{r.train_t:.1f}, n{r.train_n}) / "
                          f"VAL {r.val_mean_R:+.3f}R (t{r.val_t:.1f}, n{r.val_n}) / OOS2024H2 {r.oos_mean_R:+.3f}R (n{r.oos_n})\n")
    if not len(regime_pairs):
        append_result_md("- none\n")

    # --- build the union: production (slice-aware exits) + robust/regime general pairs ---
    logger.info('--- union construction ---')
    prod_union = pf_prod[pf_prod['window'].notna()].copy()
    prod_union['union_R'] = prod_union['E1_production']
    for slice_name, winning_exit in robust_slices.items():
        sub = slices[slice_name]
        prod_union.loc[prod_union.index.isin(sub.index), 'union_R'] = sub.loc[sub.index.isin(prod_union.index), winning_exit]
    prod_union['src'] = 'production'

    def pairs_to_fills(pairs_df, pf_map):
        out = []
        for r in pairs_df.itertuples():
            pf = pf_map.get(r.pool)
            if pf is None or not len(pf):
                continue
            sub = pf[pf['window'].notna()].copy()
            sub['union_R'] = sub[r.exit]
            sub['src'] = f'{r.pool}/{r.exit}'
            out.append(sub[['date', 'window', 'symbol', 'union_R', 'src']])
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=['date', 'window', 'symbol', 'union_R', 'src'])

    robust_fills = pairs_to_fills(robust_pairs, general_pf)
    regime_fills = pairs_to_fills(regime_pairs, general_pf)
    base_cols = ['date', 'window', 'symbol', 'union_R', 'src']

    def dedup_union(frames):
        u = pd.concat([f[base_cols] for f in frames], ignore_index=True)
        before = len(u)
        u = u.drop_duplicates(subset=['date', 'symbol'], keep='first')
        logger.info('union dedup: %d -> %d rows (%d overlap collapsed)', before, len(u), before - len(u))
        return u.dropna(subset=['union_R'])

    union_robust = dedup_union([prod_union, robust_fills])
    union_all = dedup_union([prod_union, robust_fills, regime_fills])
    prod_alone = prod_union[base_cols].dropna(subset=['union_R'])

    def union_metrics(u, lo, hi, label):
        sub = u[(u['date'] >= lo) & (u['date'] <= hi)]
        vals, dts = sub['union_R'], sub['date']
        base = reads_for_exit_series(vals, dts, lo, hi)
        trades = [{'date': d, 'r': v} for d, v in zip(dts, vals)]
        weekly = cb.build_weekly_series(trades, lo, hi)
        weekly_r = [w[1] for w in weekly]
        cycles, strong_idx = cb.compute_cycles(weekly, STRONG_WEEK_R)
        c1 = cb.score_c1(cycles, gap_median_thresh=3, gap_p90_thresh=6)
        c3 = cb.score_c3(weekly, p10_thresh=-2, min_thresh=-999, mdd_thresh=999, underwater_thresh=999)
        c4 = cb.score_c4(weekly, trades, green_thresh=0.55, green_margin=0.0, n_null=500)
        fpw = base['fills_per_week']
        by_day = sub.groupby('date')['union_R'].sum()
        worst_day = (by_day.idxmin(), float(by_day.min()) * FIXED_RISK_DOLLARS) if len(by_day) else (None, np.nan)
        logger.info('%s [%s..%s]: %s', label, lo, hi, base)
        return dict(label=label, lo=lo, hi=hi, **base,
                    weekly_p10_R_per_fill=(base['weekly_p10_R'] / fpw if fpw else np.nan),
                    weekly_p10_dollars_fixed375=base['weekly_p10_R'] * FIXED_RISK_DOLLARS,
                    weekly_worst_dollars_fixed375=base['worst_week_R'] * FIXED_RISK_DOLLARS,
                    max_drawdown_R=c3['mdd'], longest_underwater_weeks=c3['underwater'],
                    strong_week_gap_median=c1['median'], strong_week_gap_p90=c1['p90'],
                    green_share=c4['green'], green_null_mean=c4['null'],
                    shared_worst_day=worst_day[0], shared_worst_day_dollars=worst_day[1])

    union_rows = []
    for label, u in (('production_alone', prod_alone), ('union_robust', union_robust), ('union_robust+regime', union_all)):
        for wname, (lo, hi) in WINDOWS.items():
            union_rows.append(union_metrics(u, lo, hi, f'{label}/{wname}'))
    # also the full 2025-2026 VAL span combined (TRAIN+VAL), the headline comparison
    for label, u in (('production_alone', prod_alone), ('union_robust', union_robust), ('union_robust+regime', union_all)):
        union_rows.append(union_metrics(u, TRAIN_LO, VAL_HI, f'{label}/FULL_2025_2026'))
    union_df = pd.DataFrame(union_rows)
    union_df.to_csv(UNION_CSV, index=False)
    logger.info('wrote %s (%d rows)', UNION_CSV, len(union_df))

    # Fixed-weekly-risk-budget comparison (Amendment 1): hold total weekly $
    # risk constant at production-alone's own level; recompute the union's
    # per-fill $ risk so its weekly total matches, then re-scale weekly R-P10.
    full_prod = union_df[union_df.label == 'production_alone/FULL_2025_2026'].iloc[0]
    full_union = union_df[union_df.label == 'union_robust/FULL_2025_2026'].iloc[0]
    W = full_prod['fills_per_week'] * FIXED_RISK_DOLLARS
    per_fill_risk_union = W / full_union['fills_per_week'] if full_union['fills_per_week'] else np.nan
    weekly_dollar_p10_fixedW = full_union['weekly_p10_R'] * per_fill_risk_union

    append_result_md(
        "## Union (2025-01-01..2026-09-18, production + robust pairs; fixed $375/fill unless noted)\n"
        f"- production ALONE: n={full_prod['n']:.0f}, {full_prod['fills_per_week']:.2f} fills/wk, "
        f"mean {full_prod['mean_R']:+.3f} R/fill, weekly P10 {full_prod['weekly_p10_R']:+.2f} R "
        f"(${full_prod['weekly_p10_dollars_fixed375']:+.0f}), worst week ${full_prod['weekly_worst_dollars_fixed375']:+.0f}, "
        f"max drawdown {full_prod['max_drawdown_R']} R, strong-week gap median/p90={full_prod['strong_week_gap_median']}/{full_prod['strong_week_gap_p90']} wk, "
        f"green {full_prod['green_share']} vs null {full_prod['green_null_mean']}, worst day {full_prod['shared_worst_day']} (${full_prod['shared_worst_day_dollars']:+.0f})\n"
        f"- UNION (production+robust): n={full_union['n']:.0f}, {full_union['fills_per_week']:.2f} fills/wk, "
        f"mean {full_union['mean_R']:+.3f} R/fill, weekly P10 {full_union['weekly_p10_R']:+.2f} R "
        f"(${full_union['weekly_p10_dollars_fixed375']:+.0f} at $375/fill flat), worst week ${full_union['weekly_worst_dollars_fixed375']:+.0f}, "
        f"max drawdown {full_union['max_drawdown_R']} R, strong-week gap median/p90={full_union['strong_week_gap_median']}/{full_union['strong_week_gap_p90']} wk, "
        f"green {full_union['green_share']} vs null {full_union['green_null_mean']}, worst day {full_union['shared_worst_day']} (${full_union['shared_worst_day_dollars']:+.0f})\n"
        f"- weekly P10 PER FILL: production {full_prod['weekly_p10_R_per_fill']:+.3f} R/fill-week vs union {full_union['weekly_p10_R_per_fill']:+.3f} R/fill-week\n"
        f"- at a FIXED total weekly risk budget W=${W:.0f} (= production's own weekly risk): union's per-fill risk "
        f"recalibrates to ${per_fill_risk_union:.0f}/fill, weekly $ P10 = ${weekly_dollar_p10_fixedW:+.0f} vs production's "
        f"${full_prod['weekly_p10_dollars_fixed375']:+.0f}\n"
        f"- see {os.path.basename(UNION_CSV)} for the robust+regime variant and the per-window (TRAIN/VAL/OOS2024H2) breakdown\n")

    store.close()
    logger.info('cache: %d hits / %d misses', store.hits, store.misses)
    logger.info('=== DONE ===')


if __name__ == '__main__':
    main()
