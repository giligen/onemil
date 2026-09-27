#!/usr/bin/env python3
"""Cells 1,562-1,563 -- ORB retest bid at range_high minus a tick (PREREG_1562, incl. Amendment 1).

Question (owner 2026-09-27): does the HOD-break retest learning (a resting bid at level - $0.01
recovers most of the break's immediacy cost) generalise to ORB, the one book with positive gross?

Population (Amendment 1): the 410 cell-1,426 zero-latency signals that found an XNAS trigger print
inside [09:35, 09:40) ET -- 301 `status=='filled'` (the chase rule entered) + 109
`status=='skipped_guard'` (the chase guard declined; a resting bid could still take these).
Source: research/orb_latency_bt/results.csv (delay_s==0) joined to population.csv on (symbol, date).
228 signals (207 unfilled_no_tstar + 21 missing_tick_data) are excluded and counted, per Amendment 1.

Splits (Amendment 1, by entry date): TRAIN = 2023-01-12..2025-06-30, VAL = 2025-07-01..2026-09-23.

DATA-AVAILABILITY LIMITATION (disclosed, not silent -- read this before trusting any TRAIN number):
  * The tape this task was pointed at (research/hod_ofi/raw/*.parquet) is the HOD-OFI study's own
    tape -- different symbols, 2025-01+ only. It does NOT cover this population (2023-01..2026-09,
    symbols VTAK/BBAI/GNS/...). The correct tape for this population is what replay.py itself reads:
    research/orb_latency_bt/raw/{date}__{symbol}.parquet (617 files, verified to span the full
    population's date range). That is the tape actually used here.
  * That tape's fetch window is 09:34:50-09:40:00 ET only (replay.py:83-85 WIN_START_S/WIN_END_S,
    window_utc() docstring) -- it was built to find the cell-1,426 trigger print, not to support a
    15/30-minute retest search. Beyond 09:40 ET the retest search falls back to 1-minute bars.
  * `data/cache.db` (production, read-only) `intraday_bars_1min` covers 2025-01-02..2026-09-25 only
    (verified below by MIN/MAX query) -- there is no minute-bar source in this repo for 2023-2024
    (research/orb_2023/bars.db and research/orb_2024's bars.db are referenced by
    study_orb_pipeline_static_lock.py but do not exist on disk; gitignored/deleted, and this task's
    step budget does not include re-fetching them).
  * Consequence: for signals dated < 2025-01-02 (most of nominal TRAIN, 2023-01-12..2024-12-31), the
    retest-fill search is truncated at the tape's 09:40:00 ET cutoff (WARNING logged, counted) and the
    post-fill exit walk cannot be done at all (no bars) -- these signals are excluded from the SCORED
    book and reported only as tape-window diagnostics. The SCORED TRAIN book is therefore actually
    2025-01-02..2025-06-30, not the full nominal TRAIN window. This materially weakens TRAIN power and
    is exactly the kind of caveat this project requires reading before trusting a headline number.

Rule (PREREG, Amendment 1):
  * Trigger print: the results.csv `t_star` (seconds since 09:35:00 ET) at delay_s==0 for this signal
    -- this IS the frozen "first XNAS trade print >= trigger inside [09:35,09:40)" (replay.py:299-307,
    find_t_star); not re-derived here (PREREG "Not allowed": moving the windows after seeing a number
    -- t_star is an input, not a result, of this cell).
  * Cell 1,562: resting buy limit at level - $0.01 for 15 minutes after the trigger print.
  * Cell 1,563: resting buy limit at level * (1 - 0.002) for 30 minutes after the trigger print.
  * Fill: the first print STRICTLY BELOW the limit (report-only: at-or-below share also tracked).
  * Stop = the live ORB stop = range_low (study_orb_features.py:262-263: range_high/range_low = the
    5-min opening range's max high / min low). range_low is reconstructed from population.csv's
    `range_size_pct` as range_low = trigger * (1 - range_size_pct/100) (study_orb_features.py naming
    convention -- range_size_pct as a fraction of range_high; NOT independently re-derived from the
    sizing code in this task's budget, disclosed as an assumption, see RESULT_1562.md caveat 2).
  * R' = retest_fill_price - stop (PREREG's literal definition).
  * Target/time exit = the live ORB "static lock" rule recomputed from R', with LOCK_TRIGGER_R and
    LOCK_STOP_R read off R' instead of the original book's range_size:
      arm level   = fill + 1.75 * R'   (study_orb_pipeline_static_lock.py:143-144,344 LOCK_TRIGGER_R)
      lock stop   = fill + 0.50 * R'   (study_orb_pipeline_static_lock.py:143,145,345 LOCK_STOP_R)
      hard stop   = stop (range_low)   (study_orb_pipeline_static_lock.py:346 stop_price=range_low)
      ratchet     = once armed, stop = max(hard_stop, lock_stop) (:378-382 loop)
      time exit   = 15:55 ET (PREREG text -- explicitly overrides the BT walker's 15:45 force-close,
                    study_orb_pipeline_static_lock.py:304,326-332,338-343 FORCE_CLOSE_ET)
    NOT reproduced: the Rule-M/Rule-D "touchgo" early exits and the SZ1 ATR stop floor
    (study_orb_pipeline_static_lock.py:313-389 docstring items 1-2, :392-524) -- PREREG only asks for
    "the stop, the target as a function of R, partials if any, the time exit", and the touchgo rules
    are a same-bar/next-bar refinement on top of the core lock structure, not part of "the rule" as
    named; disclosed limitation, RESULT_1562.md caveat 3. No partials in the live rule (it is a single
    ratcheted stop, not a scale-out).
  * Costs: entry (passive limit) = 0bps. Target/lock-arm alone is not an exit; every exit is either a
    stop/lock fill (cost = SLIP_STOP_BPS[split], research/hod_entry/cell_1478.py:99, blended
    0.88*filled + 0.12*no-fill-tail bps) or an EOD fill at 15:55 (cost = EOD_BPS[split], PREREG
    "11.5/9.7 bps" TRAIN/VAL).
  * Base leg = the zero-latency replay fill (results.csv, pnl_replay -- already the BT's total SIZED
    dollar P&L per trade, replay.py "P&L: BT _sized_pnl + shares*(entry_price-fill_price)"; verified
    against REPORT.md's own total_usd/mean_R=P&L/375 arithmetic). base_R = pnl_replay / 375
    (replay.py:82 R_DENOM, the SAME convention REPORT.md already publishes for this population) --
    NOT re-costed to the SLIP_STOP_BPS standard as the PREREG's prose asks (disclosed limitation,
    RESULT_1562.md caveat 1: recovering the raw pre-slip per-share exit price to re-apply reason-
    specific costs needs the position-sizing formula in trading/orb_planner.py, which population.csv's
    own `pnl` column does not resolve on its own -- entry_price + pnl gives IMPOSSIBLE negative prices
    for exit_reason=='stop' rows, e.g. VTAK entry $3.70 pnl -467.95 -> "exit" -$464 -- meaning
    population.csv's per-share `pnl`/`shares` are in an internal sizing unit, not literal dollars-per-
    share; tracing that formula was out of this task's step budget).
  * Comparability of base_R and retest_R: retest_R (per signal) = (exit_price - fill_price_retest) /
    R'. Under fixed-fractional position sizing to a constant dollar risk (R_DENOM=375, the ORB
    program's own convention throughout replay.py/REPORT.md), a trade risking exactly $375 against a
    per-share stop distance D returns dollar P&L = 375 * (pnl_per_share / D) -- i.e. the R-multiple
    IS the price-distance ratio, independent of $ notional. retest_R as defined above is therefore
    directly comparable to base_R (pnl_replay/375) AS LONG AS the base leg's original sizing also
    targeted a fixed dollar risk off ITS OWN entry-to-stop distance (standard practice for this
    system, e.g. CLAUDE.md's "$150-375 risk per trade" language for other books) -- this equivalence
    is an assumption, not verified against orb_planner.py's sizing code, RESULT_1562.md caveat 1.

Verbose progress and a WARNING on every fallback/exclusion path, per project convention. Read-only on
data/cache.db (sqlite URI ?mode=ro); never writes cache.db, bars_sip.db, config.yaml or trading/.
"""
from __future__ import annotations

import glob
import os
import sqlite3
import sys
import warnings
from datetime import time as dtime

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, f'{ROOT}/research/hod_entry')
from cell_1445 import day_clustered_t, ex_top5_mean, winner_capped_mean, weeks_spanned  # noqa: E402

POP_CSV = f'{ROOT}/research/orb_latency_bt/population.csv'
RES_CSV = f'{ROOT}/research/orb_latency_bt/results.csv'
RAW_DIR = f'{ROOT}/research/orb_latency_bt/raw'
CACHE_DB = f'{ROOT}/data/cache.db'
OUT_DIR = f'{ROOT}/research/orb_retest'
ET = 'America/New_York'

# --- cost constants (PREREG_1562, research/hod_entry/cell_1478.py:99) ------------------------------
SLIP_STOP_BPS = {'TRAIN': 0.88 * 2.9 + 0.12 * 94.0, 'VAL': 0.88 * 3.2 + 0.12 * 76.0}
EOD_BPS = {'TRAIN': 11.5, 'VAL': 9.7}

# --- live ORB static-lock exit constants (study_orb_pipeline_static_lock.py:143-145) ---------------
LOCK_TRIGGER_R = 1.75
LOCK_STOP_R = 0.5

R_DENOM = 375.0            # replay.py:82 -- the ORB program's own R normalisation
TRAIN_VAL_CUTOFF = '2025-07-01'
CACHE_DB_START = '2025-01-02'   # verified MIN(bar_date) below
TIME_EXIT_ET = '15:55:00'

CELLS = {
    1562: {'window_min': 15, 'limit_fn': lambda level: round(level - 0.01, 4)},
    1563: {'window_min': 30, 'limit_fn': lambda level: round(level * (1 - 0.002), 4)},
}


def log(msg: str) -> None:
    print(f'[cell_1562] {msg}', flush=True)


def warn(msg: str) -> None:
    print(f'[cell_1562] WARNING: {msg}', flush=True)


def split_of(date_str: str) -> str:
    return 'TRAIN' if date_str < TRAIN_VAL_CUTOFF else 'VAL'


def et_ts(date_str: str, hhmmss: str) -> pd.Timestamp:
    return pd.Timestamp(f'{date_str} {hhmmss}', tz=ET).tz_convert('UTC')


# --------------------------------------------------------------------------------------------------
# Population
# --------------------------------------------------------------------------------------------------
def load_signals() -> pd.DataFrame:
    """The 410-signal population: results.csv delay_s==0, status in (filled, skipped_guard), joined
    to population.csv for range/entry/pnl/exit_reason fields. Asserts the frozen count (Amendment 1)."""
    pop = pd.read_csv(POP_CSV)
    res = pd.read_csv(RES_CSV)
    res0 = res[res.delay_s == 0].copy()
    log(f'results.csv delay_s==0: {len(res0)} rows, status counts: '
        f'{res0.status.value_counts().to_dict()}')
    sig = res0[res0.status.isin(['filled', 'skipped_guard'])].copy()
    excluded = res0[~res0.status.isin(['filled', 'skipped_guard'])]
    log(f'excluded (no trigger print in window): {len(excluded)} '
        f'({excluded.status.value_counts().to_dict()})')
    merged = sig.merge(pop, on=['symbol', 'date'], how='left', suffixes=('', '_pop'))
    n_missing_pop = merged['entry_price'].isna().sum()
    if n_missing_pop:
        warn(f'{n_missing_pop} signals had no population.csv match -- dropped')
        merged = merged.dropna(subset=['entry_price'])
    assert len(merged) == 410, f'expected 410 signals (Amendment 1), got {len(merged)}'
    merged['split'] = merged['date'].map(split_of)
    merged['range_low'] = merged['trigger'] * (1 - merged['range_size_pct'] / 100.0)
    merged['r_base'] = merged['trigger'] - merged['range_low']
    log(f'loaded {len(merged)} signals: {(merged.status == "filled").sum()} filled '
        f'(base leg = a real chase entry), {(merged.status == "skipped_guard").sum()} skipped_guard '
        f'(base leg = a zero trade)')
    return merged


# --------------------------------------------------------------------------------------------------
# Tape + bars
# --------------------------------------------------------------------------------------------------
_bars_cache: dict = {}


def load_tape_trades(symbol: str, date: str) -> pd.DataFrame | None:
    path = f'{RAW_DIR}/{date}__{symbol}.parquet'
    if not os.path.exists(path):
        warn(f'{date} {symbol}: no tape file at {path}')
        return None
    df = pd.read_parquet(path)
    trades = df[df['schema'] == 'trades'].sort_values('ts_event').reset_index(drop=True)
    if trades.empty:
        warn(f'{date} {symbol}: tape file has no trades-schema rows')
        return None
    return trades


def load_bars(symbol: str, date: str) -> pd.DataFrame | None:
    """1-min bars from data/cache.db (READ-ONLY), symbol/date. None if date < CACHE_DB_START (no
    minute-bar source exists in this repo for that date) or the symbol-day is missing."""
    key = (symbol, date)
    if key in _bars_cache:
        return _bars_cache[key]
    if date < CACHE_DB_START:
        warn(f'{date} {symbol}: no minute-bar source for dates before {CACHE_DB_START} '
             '(cache.db coverage start; research/orb_2023|2024 bars.db not present on disk)')
        _bars_cache[key] = None
        return None
    con = sqlite3.connect(f'file:{CACHE_DB}?mode=ro', uri=True, timeout=10)
    try:
        df = pd.read_sql_query(
            'SELECT timestamp, open, high, low, close FROM intraday_bars_1min '
            'WHERE symbol=? AND bar_date=? ORDER BY timestamp',
            con, params=(symbol, date))
    finally:
        con.close()
    if df.empty:
        warn(f'{date} {symbol}: cache.db has no bars for this symbol-day')
        _bars_cache[key] = None
        return None
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    _bars_cache[key] = df
    return df


def find_retest(symbol: str, date: str, trigger_ts: pd.Timestamp, window_end: pd.Timestamp,
                 limit_price: float) -> dict:
    """Search tape (trades schema, strictly after the trigger print) then 1-min bars (bar low vs
    limit, fill assumed at the limit price -- the obtainable, not-better-than-obtainable price for a
    resting order) for the first price at-or-below `limit_price`, tracking the strict fill separately
    (PREREG: report-only at-or-below share) and the running minimum price seen (dip/withdrawal stats).
    """
    dip_min = None
    dip_ts = None
    fill = False
    fill_price = None
    fill_ts = None
    source = None
    at_or_below_ts = None   # first at-or-below print, even if strict fill never happens
    tape_end = trigger_ts
    trades = load_tape_trades(symbol, date)
    if trades is not None and len(trades):
        seg = trades[(trades['ts_event'] > trigger_ts) & (trades['ts_event'] < window_end)]
        if len(seg):
            tape_end = seg['ts_event'].max()
            mrow = seg.loc[seg['price'].idxmin()]
            dip_min, dip_ts = float(mrow['price']), mrow['ts_event']
            below = seg[seg['price'] < limit_price]
            if len(below):
                r0 = below.iloc[0]
                fill, fill_price, fill_ts, source = True, float(r0['price']), r0['ts_event'], 'tape'
            at_below = seg[seg['price'] <= limit_price]
            if len(at_below):
                at_or_below_ts = at_below.iloc[0]['ts_event']
    data_gap = False
    if not fill and tape_end < window_end:
        bars = load_bars(symbol, date)
        if bars is None:
            data_gap = True
            warn(f'{date} {symbol}: retest search truncated at tape end {tape_end} '
                 f'(window needs {window_end}) -- no bars to complete the search')
        else:
            seg = bars[(bars['timestamp'] > tape_end) & (bars['timestamp'] < window_end)]
            for _, b in seg.iterrows():
                if dip_min is None or b['low'] < dip_min:
                    dip_min, dip_ts = float(b['low']), b['timestamp']
                if b['low'] <= limit_price and at_or_below_ts is None:
                    at_or_below_ts = b['timestamp']
                if b['low'] < limit_price:
                    fill, fill_price, fill_ts, source = True, float(limit_price), b['timestamp'], 'bar'
                    break
    return dict(fill=fill, fill_price=fill_price, fill_ts=fill_ts, source=source,
                dip_min=dip_min, dip_ts=dip_ts, at_or_below_ts=at_or_below_ts,
                data_gap=data_gap, tape_end=tape_end)


def exit_walk(symbol: str, date: str, fill_ts: pd.Timestamp, fill_price: float, stop: float,
              split: str) -> dict | None:
    """Static-lock exit from R' = fill_price - stop (study_orb_pipeline_static_lock.py:143-146,
    344-346, 378-382), time exit 15:55 ET (PREREG override of the BT's 15:45). Returns None if no
    bars are available (date < CACHE_DB_START) -- the caller counts this as excluded."""
    r_prime = fill_price - stop
    if r_prime <= 0:
        warn(f'{date} {symbol}: fill {fill_price} <= stop {stop} (R\' non-positive) -- '
             'degenerate, excluded from the exit walk')
        return None
    bars = load_bars(symbol, date)
    if bars is None:
        return None
    trig_lvl = fill_price + LOCK_TRIGGER_R * r_prime
    lock_lvl = fill_price + LOCK_STOP_R * r_prime
    time_exit_ts = et_ts(date, TIME_EXIT_ET)
    seg = bars[(bars['timestamp'] >= fill_ts.floor('min')) &
               (bars['timestamp'] <= time_exit_ts)].reset_index(drop=True)
    if seg.empty:
        warn(f'{date} {symbol}: no bars between fill and 15:55 ET -- excluded')
        return None
    armed = False
    cur_stop = stop
    for _, b in seg.iterrows():
        if b['timestamp'] < fill_ts:
            continue
        if not armed and b['high'] >= trig_lvl:
            armed = True
            cur_stop = max(cur_stop, lock_lvl)
        if b['low'] <= cur_stop:
            bps = SLIP_STOP_BPS[split]
            return dict(exit_price=cur_stop * (1 - bps / 10000.0),
                        reason='lock' if armed else 'stop', exit_ts=b['timestamp'], r_prime=r_prime)
    last = seg.iloc[-1]
    bps = EOD_BPS[split]
    return dict(exit_price=last['close'] * (1 - bps / 10000.0), reason='eod',
                exit_ts=last['timestamp'], r_prime=r_prime)


# --------------------------------------------------------------------------------------------------
# Per-signal, per-cell row builder
# --------------------------------------------------------------------------------------------------
def score_signal(row: pd.Series, cell_id: int) -> dict:
    cfg = CELLS[cell_id]
    split = row['split']
    trigger_ts = et_ts(row['date'], '09:35:00') + pd.Timedelta(seconds=float(row['t_star']))
    window_end = trigger_ts + pd.Timedelta(minutes=cfg['window_min'])
    level = row['trigger']
    limit_price = cfg['limit_fn'](level)
    r = find_retest(row['symbol'], row['date'], trigger_ts, window_end, limit_price)

    base_r = (row['pnl_replay'] / R_DENOM) if row['status'] == 'filled' else 0.0

    out = dict(cell=cell_id, symbol=row['symbol'], date=row['date'], split=split,
               status_base=row['status'], trigger=level, limit=limit_price,
               trigger_ts=trigger_ts, window_end=window_end,
               range_low=row['range_low'], r_base_signal=row['r_base'],
               dip_min=r['dip_min'], dip_ts=r['dip_ts'], data_gap=r['data_gap'],
               retest_fill=r['fill'], fill_price=r['fill_price'], fill_ts=r['fill_ts'],
               fill_source=r['source'], at_or_below=r['at_or_below_ts'] is not None,
               base_r=base_r, retest_r=np.nan, paired_dr=np.nan, exit_reason=None,
               minutes_to_retest=np.nan, dip_bps=np.nan, withdrawn_15min=np.nan,
               r_prime=np.nan, r_prime_pct_price=np.nan, excluded=False, exclude_reason=None)

    if r['dip_min'] is not None:
        out['dip_bps'] = (level - r['dip_min']) / level * 10000.0
        out['withdrawn_15min'] = bool(r['dip_ts'] <= trigger_ts + pd.Timedelta(minutes=15))

    if r['data_gap'] and not r['fill']:
        out['excluded'] = True
        out['exclude_reason'] = 'no_data_beyond_tape'
        return out

    if not r['fill']:
        out['retest_r'] = 0.0
        out['paired_dr'] = 0.0 - base_r
        out['minutes_to_retest'] = np.nan
        return out

    out['minutes_to_retest'] = (r['fill_ts'] - trigger_ts).total_seconds() / 60.0
    ew = exit_walk(row['symbol'], row['date'], r['fill_ts'], r['fill_price'], row['range_low'], split)
    if ew is None:
        out['excluded'] = True
        out['exclude_reason'] = 'no_bars_for_exit_walk'
        return out

    out['exit_reason'] = ew['reason']
    out['r_prime'] = ew['r_prime']
    out['r_prime_pct_price'] = ew['r_prime'] / r['fill_price'] * 100.0
    pnl_per_share = ew['exit_price'] - r['fill_price']
    out['retest_r'] = pnl_per_share / ew['r_prime']
    out['paired_dr'] = out['retest_r'] - base_r
    return out


def build_fills_csv(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(signals)
    for cell_id in CELLS:
        log(f'--- scoring cell {cell_id} ({CELLS[cell_id]["window_min"]}-min window) ---')
        for i, (_, row) in enumerate(signals.iterrows()):
            if i and i % 100 == 0:
                log(f'  cell {cell_id}: {i}/{n} signals scored')
            rows.append(score_signal(row, cell_id))
    df = pd.DataFrame(rows)
    log(f'scored {len(df)} signal-cell rows total '
        f'(excluded: {df.excluded.sum()} of {len(df)})')
    return df


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run'])
    args = ap.parse_args()
    con = sqlite3.connect(f'file:{CACHE_DB}?mode=ro', uri=True, timeout=10)
    lo, hi = con.execute('SELECT MIN(bar_date), MAX(bar_date) FROM intraday_bars_1min').fetchone()
    con.close()
    log(f'cache.db intraday_bars_1min coverage: {lo} .. {hi}')
    signals = load_signals()
    fills = build_fills_csv(signals)
    fills.to_csv(f'{OUT_DIR}/cell_1562_fills.csv', index=False)
    log(f'wrote {OUT_DIR}/cell_1562_fills.csv ({len(fills)} rows)')
