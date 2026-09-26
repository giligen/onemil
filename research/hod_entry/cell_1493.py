"""
cell_1493.py -- PREREG cells 1,493-1,547 builder: the RETEST BOUNCE, the whole exit surface
(6 stops x 9 exits + the mirror M of the 1,480 short = 55 cells) evaluated on every 1,481 retest
fill, on both holdouts, with the L3 stratum and the 1,482 replication as report-only rows.

Spec: research/hod_entry/PREREG_1493.md (read that first -- this docstring is not a substitute).

Population: research/hod_entry/rebuild_1481_fills.csv, status == 'fill' (8,973 rows). Each row is
one retest fill: entry = level - $0.01, filled at the first tape print strictly below the level
within 15 RTH minutes of the base-fill bar (retest_ts = the fill print's ns timestamp, retest_minute
= its ET minute-of-day, floor'd -- confirmed against both the tape pickles and bars_fills_1478.db:
the bars' ISO-UTC `t` converted to America/New_York gives the same integer minute index used to key
the pickle filenames and used as `retest_minute`/`fill_min`).

Walk semantics (from the PREREG, reusing sip_rebuild.walk_path's bar-priority convention):
  1. Tape phase (the retest minute itself): every trade print with ts > retest_ts, in time order.
     price <= stop -> stopped (fill AT the stop unless the print itself trades through it, in which
     case the worse print price is used -- the tape analogue of walk_path's gap-through-at-the-open
     rule); price > target -> target (fills at the target, a resting limit). Whichever is earlier.
  2. Bar phase (from bars_fills_1478.db, minute m_r+1 onward, sorted): for each bar, in order --
     m >= EOD_M (955, 15:55 ET) -> exit at that bar's OPEN, why='eod'; else if a time cap is set
     (T30/T60) and m >= cap -> exit at that bar's OPEN, why='time' (a scheduled flatten, checked
     before this bar's own stop/target); else low <= stop -> stop (open if the open itself gapped
     through, else the stop price); else high > target -> target (fills at target). If the bars run
     out before 955 (a data gap), exit at the last bar's close, why='eod_fallback', WARNING logged.

Costs (programme constants, entry and target legs are zero -- both are passive/limit fills):
  stop  -- SLIP_STOP_BPS (research/hod_entry/cell_1478.py): expected-value bps folding the 12%
           no-fill tail, {TRAIN: 0.88*2.9+0.12*94.0, VAL: 0.88*3.2+0.12*76.0} bps.
  eod/time -- EOD_BID_BPS (RESULT_1443.md EOD means at the bid), {TRAIN: 11.5, VAL: 9.7} bps.
  target -- 0. entry -- 0 (passive, no spread charged, per the PREREG).
Units: net_pct = raw_pct - cost_pct, in PERCENTAGE POINTS of the entry price (raw_pct = (exit-entry)
/entry*100). net_R = net_pct/100 * entry / (entry - stop_price) -- R := entry - stop_price in $.

Reused verbatim: research/hod_entry/cell_1445.py day_clustered_t (OLS on a constant, day-clustered
cov) and ex_top5_mean (drop the top 5% by value); the EOD_M=955 constant and the gap-through /
target-needs-high>target bar convention from research/hod_entry/sip_rebuild.py walk_path.

Every fallback (missing tape pickle, missing bars, a bar gap before 955, an unjoinable L3/feature
row) is WARNING-logged and the affected fill is EXCLUDED from that computation and COUNTED, never
imputed, per the project's fallback-must-log rule.
"""
import argparse
import datetime as dt
import logging
import os
import pickle
import sqlite3
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cell_1445 import day_clustered_t, ex_top5_mean  # noqa: E402  (reused verbatim)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                     stream=sys.stdout)
log = logging.getLogger('cell_1493')

# ------------------------------------------------------------------------------------------------ constants
FILLS_CSV = os.path.join(HERE, 'rebuild_1481_fills.csv')
REPLICATION_CSV = os.path.join(HERE, 'cell_1482_fills.csv')
BARS_DB = os.path.join(HERE, 'bars_fills_1478.db')
TAPE_DIR_PRIMARY = os.path.join(HERE, 'sip_cache_1481')
TAPE_DIR_FALLBACK = os.path.join(HERE, 'sip_cache_1480')
CAUSAL_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
L3_CSV = os.path.join(HERE, 'model_1478_L3_predictions.csv')
FEAT_CSV = os.path.join(HERE, 'features_1478_A.csv')

EOD_M = 955          # 15:55 ET minutes-since-midnight (sip_rebuild.py OPEN_M, EOD_M = 570, 955)
RTH_OPEN_M = 570     # 09:30 ET
PLACEBO_LO, PLACEBO_HI = 585, 900   # 09:45 - 15:00 ET

SLIP_STOP_BPS = {'TRAIN': 0.88 * 2.9 + 0.12 * 94.0, 'VAL': 0.88 * 3.2 + 0.12 * 76.0}  # cell_1478.py
EOD_BID_BPS = {'TRAIN': 11.5, 'VAL': 9.7}                                             # RESULT_1443.md
L3_PROB_CUT = 0.3070
WINNER_CAP_PCT = 3.0
NULL_DRAWS = 1000
SEED = 1493

STOP_PCTS = {'0.5%': 0.005, '1.0%': 0.010, '1.5%': 0.015, '2.0%': 0.020, '3.0%': 0.030}
STOP_ORDER = ['0.5%', '1.0%', '1.5%', '2.0%', '3.0%', 'CL']   # tie-break: later = "wider"
TARGET_PCTS = {'tgt0.5': 0.005, 'tgt0.75': 0.0075, 'tgt1.0': 0.010, 'tgt1.5': 0.015,
               'tgt2.0': 0.020, 'tgt3.0': 0.030}
EXIT_ORDER = ['tgt0.5', 'tgt0.75', 'tgt1.0', 'tgt1.5', 'tgt2.0', 'tgt3.0', 'NONE', 'T30', 'T60']
TIME_CAPS = {'T30': 30, 'T60': 60}
MIRROR = 'M'
ALL_STOPS = STOP_ORDER
ALL_EXITS = EXIT_ORDER + [MIRROR]  # M carries its own stop too -- handled outside the 6x9 product


def cell_grid():
    """The 54 (stop, exit) pairs plus the mirror M as a 55th pseudo-pair ('CL_or_own', 'M')."""
    cells = [(s, e) for s in STOP_ORDER for e in EXIT_ORDER]
    cells.append((MIRROR, MIRROR))
    return cells


# ------------------------------------------------------------------------------------------------ loaders

def load_fills():
    """The 1,481 rebuild's retest fills, status == 'fill' only (8,973 of 9,911 rows)."""
    df = pd.read_csv(FILLS_CSV, dtype={'day': str, 'symbol': str})
    n_all = len(df)
    df = df[df['status'] == 'fill'].reset_index(drop=True).copy()
    log.info(f'[load_fills] {len(df)}/{n_all} rows are status==fill')
    df['retest_minute'] = df['retest_minute'].astype(int)
    return df


def load_bars():
    """All bars, with an et_min column (America/New_York hour*60+minute of `t`), grouped by
    (symbol, day) into sorted numpy arrays for fast per-fill slicing. Verified 2026-09-26 against
    sip_cache_1481/AAOI_2025-07-02_728.pkl: the tape's first print at ET minute 728 matches
    bars.t == '2025-07-02T16:08:00+00:00' -> tz_convert('America/New_York') -> 12:08 -> 728."""
    t0 = time.time()
    con = sqlite3.connect(BARS_DB)
    df = pd.read_sql_query('SELECT symbol, day, t, o, h, l, c FROM bars', con)
    con.close()
    ts = pd.to_datetime(df['t'], utc=True).dt.tz_convert('America/New_York')
    df['m'] = ts.dt.hour * 60 + ts.dt.minute
    df = df.sort_values(['symbol', 'day', 'm'])
    grouped = {}
    for (sym, day), g in df.groupby(['symbol', 'day'], sort=False):
        grouped[(sym, day)] = g[['m', 'o', 'h', 'l', 'c']].to_numpy(dtype=float)
    log.info(f'[load_bars] {len(df)} bar rows, {len(grouped)} (symbol,day) groups, {time.time()-t0:.1f}s')
    return grouped


def load_tape(symbol, day, m_r):
    """(trades, quotes) for the retest minute, primary dir then fallback. None + WARNING if absent."""
    fname = f'{symbol}_{day}_{int(m_r)}.pkl'
    for d in (TAPE_DIR_PRIMARY, TAPE_DIR_FALLBACK):
        p = os.path.join(d, fname)
        if os.path.exists(p):
            with open(p, 'rb') as f:
                return pickle.load(f)
    log.warning(f'[load_tape] MISSING tape for {fname} in both sip_cache_1481 and sip_cache_1480 '
                f'-- fill excluded from every cell')
    return None


# ------------------------------------------------------------------------------------------------ per-cell walk

def stop_target_for_cell(row, stop_key, exit_key):
    """(stop_price, target_price_or_None, time_cap_minute_or_None) for one (fill row, cell).
    Returns (None, None, None) if the cell is ill-defined for this fill (R <= 0), WARNING logged."""
    entry = row['entry']
    if stop_key == MIRROR:
        stop = min(row['dip_low'] - 0.01, entry * 0.99)
        target = entry * 1.02
        return _guard(entry, stop, target, row)
    stop = row['stop'] if stop_key == 'CL' else entry * (1 - STOP_PCTS[stop_key])
    if exit_key == 'NONE':
        target, cap = None, None
    elif exit_key in TIME_CAPS:
        target, cap = None, TIME_CAPS[exit_key]
    else:
        target, cap = entry * (1 + TARGET_PCTS[exit_key]), None
    s, t_, _ = _guard(entry, stop, target, row)
    return s, t_, cap


def _guard(entry, stop, target, row):
    if not (stop < entry) or (entry - stop) <= 1e-6:
        log.warning(f"[grid] {row['day']} {row['symbol']}: stop {stop:.4f} not below entry "
                    f"{entry:.4f} -- fill excluded from this cell")
        return None, None, None
    return stop, target, None


def prep_tape_prices(tape, retest_ts):
    """Sorted trade prices strictly after the fill print, as a plain numpy array (once per fill,
    reused across all 55 cells -- the tape does not depend on the cell)."""
    if tape is None:
        return None
    trades, _quotes = tape
    after = trades.loc[trades['ts'] > retest_ts].sort_values('ts')
    return after['price'].to_numpy(dtype=float)


def prep_path(bars_group, m_r):
    """Bars from minute m_r+1 onward, already sorted (once per fill, reused across all 55 cells)."""
    if bars_group is None:
        return None
    path = bars_group[bars_group[:, 0] >= m_r + 1]
    return path if len(path) else None


def walk_one(row, stop, target, time_cap, path, prices_after):
    """Vectorized tape-then-bar walk (numpy, no per-row Python loop). Tape phase: first print
    <= stop -> stopped (worse of print/stop, the tape analogue of gap-through); first print >
    target -> target; whichever index is earlier. Bar phase: per-bar priority eod > time-cap >
    stop > target, first bar where any fires. Returns dict(exit_m, exit_price, why) or None
    (+ WARNING) if nothing resolves."""
    if prices_after is not None and len(prices_after):
        stop_hit = prices_after <= stop
        i_stop = int(np.argmax(stop_hit)) if stop_hit.any() else None
        i_tgt = None
        if target is not None:
            tgt_hit = prices_after > target
            i_tgt = int(np.argmax(tgt_hit)) if tgt_hit.any() else None
        if i_stop is not None and (i_tgt is None or i_stop <= i_tgt):
            m_r = row['retest_minute']
            return dict(exit_m=m_r, exit_price=float(min(prices_after[i_stop], stop)), why='stop')
        if i_tgt is not None:
            return dict(exit_m=row['retest_minute'], exit_price=float(target), why='target')
    if path is None:
        log.warning(f"[walk] {row['day']} {row['symbol']}: no bars after the retest minute, tape "
                    f"did not resolve -- fill excluded")
        return None
    m, o, h, l, c = path[:, 0], path[:, 1], path[:, 2], path[:, 3], path[:, 4]
    eod_mask = m >= EOD_M
    cap_mask = (m >= (row['retest_minute'] + time_cap)) if time_cap is not None else np.zeros(len(m), bool)
    stop_mask = l <= stop
    tgt_mask = (h > target) if target is not None else np.zeros(len(m), bool)
    any_mask = eod_mask | cap_mask | stop_mask | tgt_mask
    if not any_mask.any():
        log.warning(f"[walk] {row['day']} {row['symbol']}: path ended before 15:55 (last m="
                    f"{int(m[-1])}) -- exit at its close")
        return dict(exit_m=int(m[-1]), exit_price=float(c[-1]), why='eod_fallback')
    idx = int(np.argmax(any_mask))
    if eod_mask[idx]:
        return dict(exit_m=int(m[idx]), exit_price=float(o[idx]), why='eod')
    if cap_mask[idx]:
        return dict(exit_m=int(m[idx]), exit_price=float(o[idx]), why='time')
    if stop_mask[idx]:
        px = o[idx] if o[idx] <= stop else stop
        return dict(exit_m=int(m[idx]), exit_price=float(px), why='stop')
    return dict(exit_m=int(m[idx]), exit_price=float(target), why='target')


def cost_and_pct(entry, stop, exit_price, why, split):
    """(raw_pct, cost_pct, net_pct, R_dollar, net_R). Percentage POINTS of entry price."""
    raw_pct = (exit_price - entry) / entry * 100.0
    if why == 'target':
        cost_pct = 0.0
    elif why in ('stop',):
        cost_pct = SLIP_STOP_BPS[split] / 100.0
    elif why in ('eod', 'time', 'eod_fallback'):
        cost_pct = EOD_BID_BPS[split] / 100.0
    else:
        cost_pct = 0.0
    net_pct = raw_pct - cost_pct
    R_dollar = entry - stop
    net_R = (net_pct / 100.0 * entry) / R_dollar if R_dollar > 1e-9 else np.nan
    return raw_pct, cost_pct, net_pct, R_dollar, net_R


# ------------------------------------------------------------------------------------------------ full grid run

def run_grid(fills, bars, tape_cache_limit=None, verbose_every=1000):
    """Evaluate all 55 cells on every fill. Returns a long DataFrame:
    day, symbol, split, wk, cell (stop|exit, 'M' for the mirror), exit_m, exit_px, why, net_pct, net_R."""
    grid = cell_grid()
    rows = []
    n = len(fills)
    t0 = time.time()
    for i, row in enumerate(fills.itertuples(index=False), start=1):
        r = row._asdict()
        tape = load_tape(r['symbol'], r['day'], r['retest_minute'])
        bars_group = bars.get((r['symbol'], r['day']))
        if bars_group is None:
            log.warning(f"[run_grid] {r['day']} {r['symbol']}: no bars group at all for this "
                        f"symbol-day -- every cell for this fill excluded")
        prices_after = prep_tape_prices(tape, r['retest_ts'])
        path = prep_path(bars_group, r['retest_minute'])
        for stop_key, exit_key in grid:
            cell = 'M' if stop_key == MIRROR else f'{stop_key}|{exit_key}'
            stop, target, cap = stop_target_for_cell(r, stop_key, exit_key)
            if stop is None:
                continue
            res = walk_one(r, stop, target, cap, path, prices_after)
            if res is None:
                continue
            raw_pct, cost_pct, net_pct, R_d, net_R = cost_and_pct(
                r['entry'], stop, res['exit_price'], res['why'], r['split'])
            rows.append(dict(day=r['day'], symbol=r['symbol'], split=r['split'], wk=r['wk'],
                              cell=cell, fill_min=r['fill_min'], retest_minute=r['retest_minute'],
                              exit_m=res['exit_m'], exit_px=res['exit_price'], why=res['why'],
                              raw_pct=raw_pct, cost_pct=cost_pct, net_pct=net_pct, R=R_d, net_R=net_R))
        if i % verbose_every == 0 or i == n:
            log.info(f'[run_grid] {i}/{n} fills walked ({time.time()-t0:.1f}s, '
                      f'{len(rows)} cell-rows so far)')
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ base-fill null population

def load_base_fill_null_population():
    """causal_arming_causal.csv rows with status=='fill' (9,911), joined to model_1478's outcome_R
    by (day,symbol), converted to % of price = outcome_R * (fill - stop) / fill (the PREREG's
    'R as % of price' factor). Rows that fail to join are WARNING-logged and dropped."""
    caus = pd.read_csv(CAUSAL_CSV, dtype={'day': str, 'symbol': str}, low_memory=False)
    caus = caus[caus['status'] == 'fill'].copy()
    l3 = pd.read_csv(L3_CSV, dtype={'day': str, 'symbol': str})
    m = caus.merge(l3[['day', 'symbol', 'outcome_R', 'hgb_prob_L3']], on=['day', 'symbol'], how='left')
    missing = m['outcome_R'].isna().sum()
    if missing:
        log.warning(f'[null_pop] {missing}/{len(m)} base fills had no L3 outcome_R join -- dropped')
    m = m.dropna(subset=['outcome_R'])
    r_pct = (m['fill'] - m['stop']) / m['fill']
    m['base_pct'] = m['outcome_R'] * r_pct * 100.0
    return m[['day', 'symbol', 'base_pct']]


def load_cache_only_flags():
    """features_1478_A.csv store_served_1438, joined by (day,symbol) -- the cache-only share."""
    feat = pd.read_csv(FEAT_CSV, usecols=['day', 'symbol', 'store_served_1438'],
                        dtype={'day': str, 'symbol': str})
    return feat.drop_duplicates(['day', 'symbol'])


# ------------------------------------------------------------------------------------------------ scoring

def fills_per_week(n, weeks):
    return n / weeks if weeks else 0.0


def score_cell(sub, weeks_by_split, split):
    """One surface row for one (cell, holdout, stratum) subset of the long df."""
    s = sub[sub['split'] == split]
    n = len(s)
    if n == 0:
        return None
    weeks = weeks_by_split.get(split, 1)
    t = day_clustered_t(s['net_pct'], s['day'])
    return dict(n=n, mean_pct=float(s['net_pct'].mean()), t=t,
                ex_top5_pct=ex_top5_mean(s['net_pct']), mean_R=float(s['net_R'].mean(skipna=True)),
                fills_wk=fills_per_week(n, weeks),
                p_target=float((s['why'] == 'target').mean()),
                p_stop=float((s['why'] == 'stop').mean()),
                p_time=float(s['why'].isin(['eod', 'time', 'eod_fallback']).mean()),
                med_hold_min=float((s['exit_m'] - s['fill_min']).median()))


def weeks_span(days):
    iso = pd.to_datetime(pd.Series(days).unique())
    wk = {(d.isocalendar()[0], d.isocalendar()[1]) for d in iso}
    return max(len(wk), 1)


def build_surface(long_df, fills):
    """cell_1493_surface.csv rows: one per cell x holdout (TRAIN/VAL) x stratum (all/L3)."""
    weeks_by_split = {sp: weeks_span(fills.loc[fills['split'] == sp, 'day']) for sp in ('TRAIN', 'VAL')}
    l3 = pd.read_csv(L3_CSV, dtype={'day': str, 'symbol': str})[['day', 'symbol', 'hgb_prob_L3']]
    long_l3 = long_df.merge(l3, on=['day', 'symbol'], how='left')
    l3_pop = long_l3[long_l3['hgb_prob_L3'] >= L3_PROB_CUT]
    out = []
    for cell, sub in long_df.groupby('cell'):
        stop_key = 'M' if cell == 'M' else cell.split('|')[0]
        exit_key = 'M' if cell == 'M' else cell.split('|')[1]
        for split in ('TRAIN', 'VAL'):
            row = score_cell(sub, weeks_by_split, split)
            if row:
                out.append(dict(cell=cell, stop=stop_key, exit=exit_key, holdout=split,
                                 stratum='all', **row))
            l3sub = l3_pop[l3_pop['cell'] == cell]
            row_l3 = score_cell(l3sub, weeks_by_split, split)
            if row_l3:
                out.append(dict(cell=cell, stop=stop_key, exit=exit_key, holdout=split,
                                 stratum='L3', **row_l3))
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ selection

def select_cell(surface):
    """TRAIN-H2 (here: TRAIN, the rebuild's split already = TRAIN-H2 per the PREREG) selection:
    stop in {1.0%,1.5%,2.0%,3.0%,CL} (0.5% excluded, report-only, fails the R-vs-spread rail),
    >=3 fills/week, day-clustered t>=2, best mean net %, ties -> the wider stop (STOP_ORDER)."""
    tr = surface[(surface['holdout'] == 'TRAIN') & (surface['stratum'] == 'all') &
                 (surface['stop'] != '0.5%') & (surface['stop'] != 'M') & (surface['exit'] != 'M')]
    elig = tr[(tr['fills_wk'] >= 3) & (tr['t'] >= 2)]
    if not len(elig):
        log.warning('[select] NO cell clears stop>=1.0%, >=3 fills/wk, t>=2 on TRAIN-H2')
        return None
    elig = elig.copy()
    elig['stop_rank'] = elig['stop'].map(lambda s: STOP_ORDER.index(s))
    elig = elig.sort_values(['mean_pct', 'stop_rank'], ascending=[False, False])
    return elig.iloc[0]['cell']


# ------------------------------------------------------------------------------------------------ placebo

def placebo_minute(day, symbol, fill_min, seed=SEED):
    """One random RTH minute in [09:45,15:00], never inside the fill's own 15-min retest window
    [floor(fill_min), floor(fill_min)+15). Deterministic per (day,symbol) via a seeded RNG."""
    h = abs(hash((seed, day, symbol))) % (2**32)
    rng = np.random.RandomState(h)
    lo, hi = int(fill_min), int(fill_min) + 15
    for _ in range(50):
        m = rng.randint(PLACEBO_LO, PLACEBO_HI + 1)
        if not (lo <= m < hi):
            return m
    log.warning(f'[placebo] {day} {symbol}: could not avoid the retest window in 50 draws -- using {m} anyway')
    return m


def run_placebo(fills, bars, cells_to_test):
    """Same exits (same $ stop/target levels as the real fill), entered at a random RTH minute's
    bar OPEN (no tape available off-cycle, so entry uses the bar-walk convention only -- passive,
    no spread, per the PREREG). Returns a long df like run_grid's, cell x day x symbol x split."""
    rows = []
    n = len(fills)
    t0 = time.time()
    for i, row in enumerate(fills.itertuples(index=False), start=1):
        r = row._asdict()
        bars_group = bars.get((r['symbol'], r['day']))
        if bars_group is None:
            continue
        pm = placebo_minute(r['day'], r['symbol'], r['fill_min'])
        at_open = bars_group[bars_group[:, 0] == pm]
        if not len(at_open):
            log.warning(f"[placebo] {r['day']} {r['symbol']}: no bar at placebo minute {pm} -- excluded")
            continue
        p_entry = float(at_open[0, 1])
        for stop_key, exit_key in cells_to_test:
            cell = 'M' if stop_key == MIRROR else f'{stop_key}|{exit_key}'
            stop, target, cap = stop_target_for_cell(r, stop_key, exit_key)
            if stop is None:
                continue
            # re-anchor the SAME dollar stop/target levels to the placebo entry price's own frame
            # only for pct-based stops/targets (recomputed off p_entry); CL/M keep their absolute
            # $ levels (they are tied to the setup's own geometry, not the entry price).
            if stop_key not in ('CL', MIRROR):
                stop = p_entry * (1 - STOP_PCTS[stop_key])
            if exit_key not in ('NONE', 'M') and exit_key not in TIME_CAPS:
                target = p_entry * (1 + TARGET_PCTS[exit_key])
            path = bars_group[bars_group[:, 0] >= pm + 1]
            if not len(path):
                continue
            cap_m = (pm + cap) if cap is not None else None
            res = None
            for m, o, h, l, c in path:
                if m >= EOD_M:
                    res = dict(exit_m=int(m), exit_price=float(o), why='eod'); break
                if cap_m is not None and m >= cap_m:
                    res = dict(exit_m=int(m), exit_price=float(o), why='time'); break
                if l <= stop:
                    px = o if o <= stop else stop
                    res = dict(exit_m=int(m), exit_price=float(px), why='stop'); break
                if target is not None and h > target:
                    res = dict(exit_m=int(m), exit_price=float(target), why='target'); break
            if res is None:
                last = path[-1]
                res = dict(exit_m=int(last[0]), exit_price=float(last[4]), why='eod_fallback')
            raw_pct, cost_pct, net_pct, R_d, net_R = cost_and_pct(
                p_entry, stop, res['exit_price'], res['why'], r['split'])
            rows.append(dict(day=r['day'], symbol=r['symbol'], split=r['split'], cell=cell,
                              net_pct=net_pct, why=res['why']))
        if i % 1000 == 0 or i == n:
            log.info(f'[placebo] {i}/{n} fills ({time.time()-t0:.1f}s)')
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ count-matched null

def count_matched_null(selected_cell_df, base_null_pop, n_draws=NULL_DRAWS, seed=SEED):
    """1,000 draws (seed 1493): for each day in the selected cell's VAL fills, draw (with
    replacement) that day's fill-count worth of base-fill % outcomes from the SAME day in
    base_null_pop (falling back to the whole population with a WARNING if that day has none in
    the null population); the draw's mean vs the actual VAL mean gives the percentile."""
    rng = np.random.RandomState(seed)
    by_day = {d: g['base_pct'].to_numpy() for d, g in base_null_pop.groupby('day')}
    all_vals = base_null_pop['base_pct'].to_numpy()
    counts = selected_cell_df.groupby('day').size()
    fallback_days = 0
    null_means = np.empty(n_draws)
    for k in range(n_draws):
        draw = []
        for day, cnt in counts.items():
            pool = by_day.get(day)
            if pool is None or not len(pool):
                pool = all_vals
                if k == 0:
                    fallback_days += 1
            draw.append(rng.choice(pool, size=int(cnt), replace=True))
        null_means[k] = np.concatenate(draw).mean()
    if fallback_days:
        log.warning(f'[null] {fallback_days} day(s) in the selected cell had no same-day base-fill '
                    f'population -- fell back to the pooled null distribution for those draws')
    actual = selected_cell_df['net_pct'].mean()
    pctile = float((null_means < actual).mean() * 100.0)
    return pctile, actual, null_means


# ------------------------------------------------------------------------------------------------ neighbour stability

def neighbour_cells(selected_cell):
    """Adjacent-stop and adjacent-exit cells of the selected (stop|exit) cell, per the ordered
    STOP_ORDER / EXIT_ORDER lists (a disclosed adjacency convention, not part of the PREREG's own
    text). Returns a list of cell strings."""
    stop_key, exit_key = selected_cell.split('|')
    out = []
    si = STOP_ORDER.index(stop_key)
    for adj in (si - 1, si + 1):
        if 0 <= adj < len(STOP_ORDER) - 1:  # exclude 'CL' pseudo-neighbour-by-index unless adjacent
            out.append(f'{STOP_ORDER[adj]}|{exit_key}')
    if si == len(STOP_ORDER) - 2:  # neighbour of the last pct stop is CL too
        out.append(f'CL|{exit_key}')
    ei = EXIT_ORDER.index(exit_key)
    for adj in (ei - 1, ei + 1):
        if 0 <= adj < len(EXIT_ORDER):
            out.append(f'{stop_key}|{EXIT_ORDER[adj]}')
    return sorted(set(out) - {selected_cell})


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', type=int, default=0, help='run on the first N fills only')
    ap.add_argument('--out-dir', default=HERE)
    args = ap.parse_args()

    fills = load_fills()
    if args.smoke:
        fills = fills.iloc[:args.smoke].copy()
        log.info(f'[main] SMOKE run on {len(fills)} fills')
    bars = load_bars()

    long_df = run_grid(fills, bars)
    fills_csv = os.path.join(args.out_dir, 'cell_1493_fills.csv' if not args.smoke else
                              'cell_1493_fills_SMOKE.csv')
    long_df.to_csv(fills_csv, index=False)
    log.info(f'[main] wrote {fills_csv} ({len(long_df)} rows)')

    surface = build_surface(long_df, fills)
    selected = select_cell(surface)
    surface['selected'] = surface['cell'] == selected
    log.info(f'[main] SELECTED cell (TRAIN-H2): {selected}')
    if selected is None:
        # No cell clears the TRAIN-H2 gate (stop>=1.0%, >=3 fills/wk, t>=2): report a flat surface.
        # The placebo/null machinery still needs ONE cell to illustrate on -- use the best-by-mean
        # TRAIN cell among the eligible stop set as a REPORT-ONLY stand-in, never as a pass.
        tr_elig = surface[(surface['holdout'] == 'TRAIN') & (surface['stratum'] == 'all') &
                           (surface['stop'] != '0.5%') & (surface['stop'] != 'M')]
        best_for_illustration = tr_elig.sort_values('mean_pct', ascending=False).iloc[0]['cell']
        log.warning(f'[main] no cell passes selection; illustrating the null/placebo machinery on '
                    f'{best_for_illustration} (best TRAIN mean, NOT a passing cell) for the record')
    else:
        best_for_illustration = selected

    base_null_pop = load_base_fill_null_population()
    val_sel = long_df[(long_df['cell'] == best_for_illustration) & (long_df['split'] == 'VAL')]
    if len(val_sel):
        pctile, actual, _ = count_matched_null(val_sel, base_null_pop)
        surface.loc[surface['cell'] == best_for_illustration, 'null_pctile'] = pctile
        log.info(f'[main] count-matched null percentile for {best_for_illustration} on VAL: '
                 f'{pctile:.1f} (actual mean {actual:.4f}%)')

    none_cells = [(s, 'NONE') for s in STOP_ORDER]
    to_placebo = list(dict.fromkeys(none_cells + [tuple(best_for_illustration.split('|'))
                                                    if best_for_illustration != 'M' else (MIRROR, MIRROR)]))
    placebo_df = run_placebo(fills, bars, to_placebo)
    placebo_csv = os.path.join(args.out_dir, 'cell_1493_placebo.csv' if not args.smoke else
                                'cell_1493_placebo_SMOKE.csv')
    placebo_df.to_csv(placebo_csv, index=False)
    for cell, g in placebo_df.groupby('cell'):
        gv = g[g['split'] == 'VAL']
        if len(gv):
            surface.loc[(surface['cell'] == cell) & (surface['holdout'] == 'VAL') &
                        (surface['stratum'] == 'all'), 'placebo_pct'] = gv['net_pct'].mean()

    surface_csv = os.path.join(args.out_dir, 'cell_1493_surface.csv' if not args.smoke else
                                'cell_1493_surface_SMOKE.csv')
    surface.to_csv(surface_csv, index=False)
    log.info(f'[main] wrote {surface_csv} ({len(surface)} rows)')
    log.info('[main] DONE')


if __name__ == '__main__':
    main()
