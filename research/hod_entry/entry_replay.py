#!/usr/bin/env python3
"""HOD-break ENTRY replay — PREREG.md cells 1,423-1,425 (research/hod_entry/PREREG.md, FROZEN
2026-09-25). Replays every HOD-break B0 signal (research/hod_exit_lab/b0_trades.csv, TEST split
dropped on read and never touched) with a resting buy-stop-limit entry at the level instead of
B0's next-bar-open entry, using XNAS tick data (research/hod_ofi/raw/*.parquet) for the fill.

Three entries, all sharing B0's stop, B0's 2R target recomputed from the new R, B0's 15:55 exit
and B0's path-walk rules (research/hod_exit_lab/walker.py:b0_fill, reproduced here verbatim):
  E1 — resting stop-limit at the level. Trigger = level + 0.01 (tick). Limit = level x 1.0015
       (15 bps chase cap). Fill instant = first XNAS print >= trigger inside the break bar; fill
       price = the prevailing XNAS ask at that instant (mbp-1, strictly prior record) if
       ask <= limit, else NO FILL. A print <= stop after the fill, still inside the break bar,
       stops the trade at the stop price (conservative, checked off the tape). From the next bar
       on (paths.parquet, which already starts at entry_m — the bar right after the break bar —
       so no `m > entry_m` re-filter is needed/possible), the B0 walker (b0_fill) decides the exit.
  E2 — idealised level fill: same fill instant as E1, fill price = trigger exactly (no ask, no
       cap). Report-only upper bound.
  E3 — B0's own entry/exit (next-bar open, from b0_trades.csv, unchanged), re-costed with the same
       measured half-spread convention as E1/E2's entry leg (the pairing control).

Level: the HOD level a signal broke = the running max of CLOSED 1-min highs strictly before the
break bar (`trading/hod_break.py:detect` — level = hod[i-1] where hod = np.maximum.accumulate(h)).
Reproduced here from `data/cache.db`'s `intraday_bars_1min` table (regular hours 09:30-15:59 ET,
deduped by minute) per PREREG's explicit data pointer — NOT taken from the `level` column already
present in `research/bf_zero/causal_filter/features.csv` (spot-checked 8 signals against
cache.db: 4/4 with data matched exactly, 1 mismatched (bars_sip.db vs cache.db source drift), 3 had
no cache.db bar for the break-bar minute — this is why we recompute from cache.db rather than trust
the stored column, and why a missing/mismatched break-bar bar makes a signal VOID for this study,
counted in the availability rail).

Cost (PREREG's cost sentence): entry leg = half the quoted XNAS spread at the fill instant (from
mbp-1, no added slippage) / R. Exit leg = B0's OWN exit-leg cost rule, unchanged: half of the day's
measured NBBO spread_mean (`research/bf_zero/causal_filter/nbbo.csv`) + 2 bps of the exit price,
both / R (SLIP_BP = 0.0002, from walker.py). cost_R = entry_leg + exit_leg; net_R = raw_rr - cost_R.

Data sources (read-only, never modified):
  * research/hod_exit_lab/b0_trades.csv   — the book (TEST dropped on read)
  * research/hod_exit_lab/paths.parquet   — per-minute OHLC after entry_m, for the post-break walk
  * research/bf_zero/causal_filter/features.csv — level (cross-check only) and entry_m keys
  * research/bf_zero/causal_filter/nbbo.csv     — spread_mean per (day,symbol)
  * research/hod_ofi/raw/*.parquet        — XNAS trades + mbp-1 quotes, kind='signal' rows only
  * data/cache.db (read-only URI, short-lived connection, one dedicated pass) — 1-min bars for the
    level recompute. Opened once, closed immediately after that pass — never held open elsewhere.

Owner note (2026-09-25): run at NORMAL priority — no nice/ionice (overrides the general market-hours
heavy-step rule for this task only).

Usage:
    python3 entry_replay.py run --out research/hod_entry/replay_signals.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

B0_CSV = f'{ROOT}/research/hod_exit_lab/b0_trades.csv'
PATHS_PARQUET = f'{ROOT}/research/hod_exit_lab/paths.parquet'
FEATURES_CSV = f'{ROOT}/research/bf_zero/causal_filter/features.csv'
NBBO_CSV = f'{ROOT}/research/bf_zero/causal_filter/nbbo.csv'
RAW_DIR = f'{ROOT}/research/hod_ofi/raw'
CACHE_DB = f'{ROOT}/data/cache.db'
ET = 'America/New_York'

OPEN_M = 570            # 09:30 ET
CLOSE_M = 959           # 15:59 ET
EOD_M = 955             # 15:55 ET force-flat minute (matches walker.py)
TARGET_R = 2.0
SLIP_BP = 0.0002        # 2 bp per side, walker.py:54
TICK = 0.01
E1_TRIGGER_OFFSET = 0.01
E1_LIMIT_BPS = 0.0015   # 15 bps chase cap (frozen)
LENS_LIMIT_BPS = (0.0005, 0.0015, 0.0030)   # lens (ii): 5 / 15 / 30 bps, report-only


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- level (cache.db)
def compute_level(bars: pd.DataFrame, break_m: int) -> float | None:
    """Level = running max of CLOSED 1-min highs strictly before the break bar, i.e.
    max(h for m < break_m). `bars` must have columns m (int minute-of-day) and h (float high),
    one row per minute, already restricted to regular hours and deduped by minute.
    Returns None if there is no bar before break_m (can't compute a level) — the caller marks the
    signal VOID. Mirrors trading/hod_break.py:detect (level = hod[i-1], hod = cummax(h))."""
    prior = bars[bars.m < break_m]
    if prior.empty:
        return None
    return float(prior.h.max())


def fetch_day_bars(con: sqlite3.Connection, symbol: str, day: str) -> pd.DataFrame | None:
    """One indexed query per (symbol, day) against cache.db's intraday_bars_1min, restricted to
    regular hours and deduped by minute (same convention as hod_exit_lab/walker.py:fetch_day_bars,
    reproduced here against cache.db instead of bars_sip.db per PREREG's data pointer)."""
    cur = con.execute(
        "SELECT timestamp, high FROM intraday_bars_1min WHERE symbol=? AND bar_date=? ORDER BY timestamp",
        (symbol, day))
    rows = cur.fetchall()
    if not rows:
        return None
    d = pd.DataFrame(rows, columns=['t', 'h'])
    ts = pd.to_datetime(d.t, utc=True).dt.tz_convert(ET)
    d['m'] = ts.dt.hour * 60 + ts.dt.minute
    d = d[(d.m >= OPEN_M) & (d.m <= CLOSE_M)].sort_values('m').drop_duplicates('m').reset_index(drop=True)
    return d if len(d) else None


def load_levels(pairs: list[tuple[str, str, int]]) -> dict:
    """One short-lived read-only connection to cache.db for the WHOLE level-recompute pass (closed
    immediately after). `pairs` = [(day, symbol, entry_m), ...], entry_m = the B0 entry bar's
    minute (break bar = entry_m - 1). Returns {(day,symbol,entry_m): level_or_None}."""
    con = sqlite3.connect(f'file:{CACHE_DB}?mode=ro', uri=True)
    out = {}
    bars_cache = {}
    n = len(pairs)
    for i, (day, symbol, entry_m) in enumerate(pairs):
        key = (day, symbol)
        if key not in bars_cache:
            bars_cache[key] = fetch_day_bars(con, symbol, day)
        bars = bars_cache[key]
        out[(day, symbol, entry_m)] = compute_level(bars, entry_m - 1) if bars is not None else None
        if (i + 1) % 1000 == 0 or i + 1 == n:
            log(f'[level] {i + 1}/{n} pairs, {sum(v is None for v in out.values())} None so far')
    con.close()
    return out


# ---------------------------------------------------------------- ticks (hod_ofi raw)
def to_et_sec(ts: pd.Series) -> pd.Series:
    """ET seconds-since-midnight, fractional. Identical to research/hod_ofi/pipeline.py:to_et_sec."""
    t = pd.to_datetime(ts, utc=True).dt.tz_convert(ET)
    return t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second + t.dt.microsecond / 1e6


def load_day_ticks(day: str) -> pd.DataFrame | None:
    """Concat every raw/{day}__*.parquet chunk for this day (research/hod_ofi/pipeline.py's own
    per-day glob pattern), kind=='signal' rows only (placebo rows dropped — not used here), with
    'sec' (ET seconds) computed once for the whole day."""
    paths = sorted(glob.glob(f'{RAW_DIR}/{day}__*.parquet'))
    if not paths:
        return None
    raw = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    raw = raw[raw.kind == 'signal'].copy()
    if raw.empty:
        return None
    raw['sec'] = to_et_sec(raw['ts_event'])
    return raw


def prevailing_quote(mbp: pd.DataFrame, at_sec: float) -> tuple | None:
    """The prevailing mbp-1 quote at `at_sec`: the STRICTLY PRIOR record (sec < at_sec, last one)
    — never a record at or after the instant being priced, so a fill can never see its own quote
    update. Returns (bid, ask) or None if no prior record / not two-sided (bid>0, ask>0, ask>=bid)."""
    prior = mbp[mbp.sec < at_sec]
    if prior.empty:
        return None
    row = prior.loc[prior.sec.idxmax()]
    bid, ask = float(row.bid_px_00), float(row.ask_px_00)
    if not (bid > 0 and ask > 0 and ask >= bid):
        return None
    return bid, ask


def first_trigger_print(trades: pd.DataFrame, lo_sec: float, hi_sec: float, trigger: float):
    """First XNAS trade print with price >= trigger inside [lo_sec, hi_sec) (the break bar).
    Returns (sec, price) of that print, or None if the trigger is never crossed on XNAS."""
    win = trades[(trades.sec >= lo_sec) & (trades.sec < hi_sec) & (trades.price >= trigger)]
    if win.empty:
        return None
    row = win.loc[win.sec.idxmin()]
    return float(row.sec), float(row.price)


def intrabar_stop(trades: pd.DataFrame, after_sec: float, hi_sec: float, stop: float):
    """First print <= stop strictly after `after_sec` (the fill) and before `hi_sec` (the break
    bar's close) — PREREG's conservative same-bar stop-out, checked off the tape, not the bar low."""
    win = trades[(trades.sec > after_sec) & (trades.sec < hi_sec) & (trades.price <= stop)]
    if win.empty:
        return None
    row = win.loc[win.sec.idxmin()]
    return float(row.sec)


# ---------------------------------------------------------------- path walk (B0's rule, verbatim)
def b0_fill(entry: float, stop: float, target: float, path_after: pd.DataFrame):
    """Exact reproduction of research/hod_exit_lab/walker.py:b0_fill. `path_after` = paths.parquet
    rows for this (day,symbol), m ascending, already the bars from the NEXT bar on (paths.parquet
    starts at the original B0 entry bar, entry_m, which for every E-variant here IS 'the next bar'
    after the break bar — entry_m-1 — so no extra m>entry_m filter is applied or needed).
    Returns (exit_m, exit_price, why) with why in {'eod','stop','target'}."""
    for row in path_after.itertuples():
        if row.m >= EOD_M:
            return int(row.m), float(row.o), 'eod'
        if row.l <= stop:
            px = row.o if row.o <= stop else stop
            return int(row.m), float(px), 'stop'
        if row.h >= target:
            return int(row.m), float(target), 'target'
    last = path_after.iloc[-1]
    return int(last.m), float(last.c), 'eod'


# ---------------------------------------------------------------- cost
def entry_leg_cost_tick(tick_spread: float, R: float) -> float:
    """PREREG cost sentence, entry leg for E1/E2/E3: half the quoted XNAS spread at the fill
    instant, / R. No added slippage (unlike B0's own entry leg, which used SLIP_BP too)."""
    return 0.5 * tick_spread / R


def exit_leg_cost(spread_mean: float, R: float, exit_price: float) -> float:
    """B0's exit-leg cost rule, UNCHANGED (PREREG: 'B0's exit cost rule unchanged'): half of the
    day's measured NBBO spread_mean + SLIP_BP (2 bp) of the exit price, both / R. Identical in form
    to walker.py:simulate_b0's per-leg contribution (cost_R = 2*half_R + slip_R was BOTH legs
    combined with the SAME spread_mean; this is just the exit half of that same formula)."""
    return 0.5 * spread_mean / R + SLIP_BP * exit_price / R


# ---------------------------------------------------------------- per-signal replay
def simulate_signal(row, level: float | None, day_ticks: pd.DataFrame | None,
                     paths_grp: pd.DataFrame | None, spread_mean: float,
                     limit_bps: float = E1_LIMIT_BPS) -> dict:
    """Replay ONE signal for E1 (capped stop-limit), E2 (idealised) and E3 (B0 re-costed).
    `row` = a record from the merged b0_trades frame (entry,stop,R,exit_price,raw_rr,entry_m,day,
    symbol). Returns a flat dict with usability flags, fill outcomes, and E1/E2/E3 net_R (NaN
    where not applicable). `limit_bps` lets lens (ii) rerun E1 at 5/15/30 bps without duplicating
    the whole function."""
    out = dict(day=row.day, symbol=row.symbol, entry_m=row.entry_m, split=row.split, half=row.half,
               level=level, chase_R=np.nan,
               usable=False, void_reason=None, trigger_lag_s=np.nan,
               e1_fill=False, e1_reason=None, e1_net_R=np.nan, e1_why=None, e1_exit_m=np.nan,
               e2_net_R=np.nan, e2_why=None, e2_exit_m=np.nan,
               e3_net_R=np.nan, e3_why=None,
               e1_ask1tick_net_R=np.nan)

    if level is None or level <= 0:
        out['void_reason'] = 'no_level'
        return out
    R_b0 = float(row.R)
    out['chase_R'] = (float(row.entry) - level) / R_b0 if R_b0 else np.nan

    S = row.entry_m * 60
    if day_ticks is None:
        out['void_reason'] = 'no_raw_ticks'
        return out
    sub = day_ticks[(day_ticks.symbol_win == row.symbol) & (day_ticks.entry_m == row.entry_m)]
    trades = sub[sub.schema == 'trades']
    mbp = sub[sub.schema == 'mbp-1']
    trades_break = trades[(trades.sec >= S - 60) & (trades.sec < S)]
    if trades_break.empty:
        out['void_reason'] = 'no_trade_in_break_bar'
        return out

    trigger = level + E1_TRIGGER_OFFSET
    cross = first_trigger_print(trades, S - 60, S, trigger)
    if cross is None:
        out['void_reason'] = 'no_trigger_cross_xnas'
        return out
    fill_sec, _print_px = cross
    out['trigger_lag_s'] = fill_sec - (S - 60)
    q = prevailing_quote(mbp, fill_sec)
    if q is None:
        out['void_reason'] = 'no_quote_at_fill'
        return out
    bid, ask = q
    tick_spread = ask - bid
    out['usable'] = True
    no_path = paths_grp is None or paths_grp.empty
    # usable (availability rail counts it) even with no post-break path: E1/E2 need the path walk
    # and are skipped below, but E3 re-costs B0's OWN already-published exit and needs no path.

    def run_variant(entry_px: float, ask_used: float, limit_bp: float):
        limit = level * (1.0 + limit_bp)
        if ask_used > limit:
            return None  # NO FILL
        stop_sec = intrabar_stop(trades, fill_sec, S, row.stop)
        R_new = entry_px - row.stop
        if R_new <= 0:
            return None
        if stop_sec is not None:
            exit_m, exit_px, why = row.entry_m - 1, float(row.stop), 'stop_intrabar'
        else:
            target_new = entry_px + TARGET_R * R_new
            exit_m, exit_px, why = b0_fill(entry_px, row.stop, target_new, paths_grp)
        raw_rr = (exit_px - entry_px) / R_new
        cost_R = entry_leg_cost_tick(tick_spread, R_new) + exit_leg_cost(spread_mean, R_new, exit_px)
        return dict(net_R=raw_rr - cost_R, why=why, exit_m=exit_m, R=R_new, exit_px=exit_px)

    # E1 (capped) and E2 (idealised) both need the post-break path walk
    if not no_path:
        r1 = run_variant(ask, ask, limit_bps)
        if r1 is None:
            out['e1_reason'] = 'ask_above_limit'
        else:
            out['e1_fill'] = True
            out['e1_net_R'] = r1['net_R']; out['e1_why'] = r1['why']; out['e1_exit_m'] = r1['exit_m']
            # lens (iii): ask + 1 tick slippage sensitivity
            r1s = run_variant(ask + TICK, ask, limit_bps)
            if r1s is not None:
                out['e1_ask1tick_net_R'] = r1s['net_R']

        # E2 (idealised: fill at trigger exactly, no ask/cap gate -> pass +inf as the limit budget)
        r2 = run_variant(trigger, level, np.inf)
        if r2 is not None:
            out['e2_net_R'] = r2['net_R']; out['e2_why'] = r2['why']; out['e2_exit_m'] = r2['exit_m']
    else:
        out['void_reason'] = out['void_reason'] or 'no_path_after_entry'

    # E3 (B0 re-costed at the SAME entry/exit, tick-measured entry-leg cost at S)
    q3 = prevailing_quote(mbp, S)
    if q3 is not None:
        tick_spread_3 = q3[1] - q3[0]
        cost_R3 = entry_leg_cost_tick(tick_spread_3, R_b0) + exit_leg_cost(spread_mean, R_b0, float(row.exit_price))
        out['e3_net_R'] = float(row.raw_rr) - cost_R3
        out['e3_why'] = row.why

    return out


# ---------------------------------------------------------------- driver
def build_population() -> pd.DataFrame:
    """b0_trades.csv, TEST dropped ON READ (never touched again); restricted to TRAIN-H2 and VAL
    (the two splits this study scores; TRAIN-H1 is not part of the PREREG's tables)."""
    b0 = pd.read_csv(B0_CSV, dtype={'symbol': str, 'day': str})
    b0 = b0[b0.split != 'TEST'].copy()
    work = b0[(b0.split == 'VAL') | ((b0.split == 'TRAIN') & (b0.half == 'H2'))].reset_index(drop=True)
    log(f'[pop] b0_trades TEST dropped on read; TRAIN-H2+VAL working set = {len(work)} of '
        f'{len(b0)} non-TEST signals')
    nbbo = pd.read_csv(NBBO_CSV, dtype={'symbol': str, 'day': str})
    nbbo = nbbo.drop_duplicates(['day', 'symbol'])[['day', 'symbol', 'spread_mean']]
    work = work.merge(nbbo, on=['day', 'symbol'], how='left')
    log(f'[pop] NBBO spread_mean coverage = {work.spread_mean.notna().mean()*100:.1f}%')
    return work


def run(out_csv: str):
    work = build_population()
    pairs = list(work[['day', 'symbol', 'entry_m']].itertuples(index=False, name=None))
    levels = load_levels(pairs)

    paths = pd.read_parquet(PATHS_PARQUET)
    paths_idx = {k: g.sort_values('m') for k, g in paths.groupby(['day', 'symbol'])}
    log(f'[paths] {len(paths_idx)} (day,symbol) groups loaded')

    results = []
    n_days = work.day.nunique()
    for di, (day, grp) in enumerate(work.groupby('day')):
        day_ticks = load_day_ticks(day)
        for row in grp.itertuples():
            key = (row.day, row.symbol)
            paths_grp = paths_idx.get(key)
            level = levels.get((row.day, row.symbol, row.entry_m))
            sm = row.spread_mean if pd.notna(row.spread_mean) else np.nan
            res = simulate_signal(row, level, day_ticks, paths_grp, sm)
            results.append(res)
        if (di + 1) % 25 == 0 or di + 1 == n_days:
            log(f'[replay] {di + 1}/{n_days} days, {len(results)} signals so far')

    out = pd.DataFrame(results)
    out.to_csv(out_csv, index=False)
    log(f'[replay] DONE -> {out_csv} ({len(out)} rows, usable={out.usable.sum()}, '
        f'e1_fill={out.e1_fill.sum()})')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')
    r = sub.add_parser('run')
    r.add_argument('--out', default=f'{ROOT}/research/hod_entry/replay_signals.csv')
    args = ap.parse_args()
    if args.cmd == 'run':
        run(args.out)
    else:
        ap.print_help()
