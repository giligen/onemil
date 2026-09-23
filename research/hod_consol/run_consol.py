#!/usr/bin/env python3
"""Base-under-the-high signal builder + scorer -- research/hod_consol/PREREG.md, cells 1,400-1,402.

Reuses machinery from research/hod_pmh_causal/build_pmh_causal.py (load_candidates,
fetch_day_bars_dual over bars_sip.db + bars_rth.db, wait_for_db_window, split_of, bar_density,
conservative_fill, minute_since_entry_table, orb_overlap, the parquet writers) and from
research/hod_pmh/score_pmh.py's scoring pattern (day-clustered t, ex-top-5%, first-12/day
4-concurrent slot rule, cadence_bar.py call, D1/D3 placebos), importing fill/cost/stat primitives
directly from research/hod_exit_lab/score_cells.py.

ONLY the signal changes from PMH-causal (PREREG.md, replaces the pre-market-high break entirely --
no pre-market data is used at all):
  FIRST 1-min bar t, 10:00 <= t <= 14:00 ET, per symbol-day, such that:
    H_t = running high of the day through bar t; tau = latest bar with high == H_t (a re-touch
          resets tau); age = t - tau >= 20 (clock minutes, "no new high for 20 min")
    B_t = min low of the last 20 bars BY POSITION (bars t-19..t inclusive, a bar-count window)
    depth = (H_t - B_t) / H_t <= 6%
    press: close_t >= B_t + 0.67*(H_t - B_t)                      ("top third of the base")
    in-play: H_t >= 1.01 * the 09:30 bar open
    cost gate: close_t - B_t >= 1.5% of close_t
  Entry = open of bar t+1. Stop = B_t. R = entry - B_t.

Resolved ambiguities (see the final report for the short version):
  * "t-tau>=20" (age) uses clock-minute difference; "bars t-19..t" (base) uses a positional
    (bar-count) window -- the literal reading of each clause taken separately.
  * "if the entry opens at or below B_t the trade is a stop at the open": MAIN-SESSION CORRECTION
    2026-09-23 — the implementer dropped these (survivorship: it removes guaranteed losers). They are
    now KEPT as trades (column sao=1): R unit = the decision-time R_est = close_t - B_t (>= the 1.5 %
    floor by construction), exit = entry at the entry bar's open (gross 0), net = -cost in R_est units.
    They are excluded only from the descriptive drift and DECOMP exhibits (whose path walks assume a
    stop below the entry) and their count is printed there.
  * D1/D3 placebos need bars BEFORE each signal's own entry_m (a random placebo minute can fall
    before the real entry, and D1/D3 both use a "prior 20 bars" stop) -- paths.parquet therefore
    caches the FULL RTH day (09:30-15:59) for every signal's symbol-day, not just entry_m->EOD as
    build_pmh_causal.py did; this is a necessary, not a size, change to the reused machinery.
  * D3 — MAIN-SESSION CORRECTION 2026-09-23: the PREREG pool is "a random OTHER universe symbol-day
    of the same date". The pass bar now draws from a seeded uniform sample of D3_POOL_K universe
    names per non-TEST date (pm_candidates, ANY signal status), whose full-day bars are cached during
    the one pass (pool_paths.parquet). The implementer's signal-producing-pool version is kept as a
    reported diagnostic only (d3_signalpool_val).

Usage:
  python3 run_consol.py --smoke 30
  python3 run_consol.py
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
LAB = f'{ROOT}/research/hod_exit_lab'
PMH_CAUSAL = f'{ROOT}/research/hod_pmh_causal'
HERE = f'{ROOT}/research/hod_consol'
BARS_SIP_DB = f'{ROOT}/research/bf_zero/bars_sip.db'
BARS_RTH_DB = f'{PMH_CAUSAL}/bars_rth.db'
CANDIDATES_CSV = f'{PMH_CAUSAL}/pm_candidates.csv'
ORB_RUNB = f'{ROOT}/research/orb_seed_wide/out/runB_true.csv'
ORB_RUNCOMB = f'{ROOT}/research/orb_seed_wide/out/runCOMB_true.csv'
CADENCE = f'{ROOT}/scripts/cadence_bar.py'
TRADES_DIR = f'{HERE}/trades'

sys.path.insert(0, LAB)
import walker as W  # noqa: E402 -- OPEN_M/CLOSE_M/EOD_M, build-stage helper pattern
from score_cells import cost_net, b0_style_fill, x1_no_target, day_clustered_t, ex_top5, \
    EOD_M  # noqa: E402 -- score-stage fill/cost/stat primitives (score-stage EOD_M == W.EOD_M == 955)

# ---- signal constants (PREREG.md, frozen -- no threshold changed) ----
SCAN_START_M, SCAN_END_M = 600, 840       # 10:00-14:00 ET inclusive
BASE_WINDOW = 20                          # bars t-19..t (base), and the placebos' "prior 20 bars"
AGE_MIN_MINUTES = 20                      # t - tau >= 20 (clock minutes)
DEPTH_MAX = 0.06
PRESS_FRAC = 0.67                         # "top third of the base" -- literal PREREG value
INPLAY_MULT = 1.01
COST_GATE_PCT = 0.015
DENSITY_MIN = 0.80                        # bar-density-in-hold rail
PROXY_HALF_SPREAD_PCT = 0.0015            # 15bp half-spread, BOTH legs (PREREG cost, + 2bp/side via cost_net's SLIP_BP)
BLOCK_START, BLOCK_END = 1325, 2005       # UTC HHMM heavy-DB blackout (market-hours rule)

TRAIN_HALF_SPLIT = '2025-07-02'           # calendar midpoint of TRAIN, PMH-population convention reused
CONCURRENT_CAP = 4
DAILY_CAP = 12
N_DRAWS = 8
SEED = 1400
PASS_BAR_VAL_T = 2.0
PASS_BAR_R = 0.10
PLACEBO_BEAT_R = 0.10
PLACEBO_MAX_REDRAWS = 5
D3_POOL_K = 16                            # universe names sampled per date for the PREREG D3 pool

MARKS = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]


def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)


def wait_for_db_window(poll_s=60):
    while True:
        now = int(datetime.now(timezone.utc).strftime('%H%M'))
        if not (BLOCK_START <= now < BLOCK_END):
            log(f'DB window clear (UTC {now:04d}) -- proceeding')
            return
        log(f'UTC {now:04d} inside blackout [{BLOCK_START},{BLOCK_END}) -- sleeping {poll_s}s')
        time.sleep(poll_s)


def load_candidates():
    rows = []
    with open(CANDIDATES_CSV) as fh:
        for r in csv.DictReader(fh):
            rows.append((r['symbol'], r['bar_date']))
    log(f'[candidates] {len(rows)} (symbol,day) pairs loaded from pm_candidates.csv (causal universe)')
    return rows


def fetch_day_bars_dual(sip_con, rth_con, symbol, day):
    """Identical to build_pmh_causal.fetch_day_bars_dual: bars_sip.db first, bars_rth.db fallback,
    same UTC->ET parse + [09:30,15:59] restriction."""
    cur = sip_con.execute('SELECT t,o,h,l,c,v FROM bars WHERE symbol=? AND day=? ORDER BY t', (symbol, day))
    rows = cur.fetchall()
    src = 'sip'
    if not rows and rth_con is not None:
        cur = rth_con.execute('SELECT t,o,h,l,c,v FROM bars WHERE symbol=? AND day=? ORDER BY t', (symbol, day))
        rows = cur.fetchall()
        src = 'rth'
    if not rows:
        return None, None
    d = pd.DataFrame(rows, columns=['t', 'o', 'h', 'l', 'c', 'v'])
    ts = pd.to_datetime(d.t, utc=True).dt.tz_convert('America/New_York')
    d['m'] = ts.dt.hour * 60 + ts.dt.minute
    d = d[(d.m >= W.OPEN_M) & (d.m <= W.CLOSE_M)].sort_values('m').drop_duplicates('m').reset_index(drop=True)
    return (d if len(d) else None), src


def split_of(day):
    if day < '2026-01-01':
        return 'TRAIN'
    if day < '2026-06-01':
        return 'VAL'
    return 'TEST'


def week_monday(d):
    dt = datetime.strptime(d, '%Y-%m-%d').date()
    return (dt - timedelta(days=dt.weekday())).isoformat()


def half_of(day, split):
    if split != 'TRAIN':
        return None
    return 'H1' if day < TRAIN_HALF_SPLIT else 'H2'


# ------------------------------------------------------------------------------------------------
# Signal (PREREG.md -- the only piece replaced from build_pmh_causal.py)
# ------------------------------------------------------------------------------------------------

def find_signal(d):
    """FIRST 1-min bar t, 10:00<=t<=14:00 ET, satisfying age/depth/press/inplay/cost. Returns
    (dict, 'ok') for the qualifying bar or (None, reason)."""
    open_row = d[d.m == W.OPEN_M]
    if open_row.empty:
        return None, 'no_930_bar'
    open_930 = float(open_row.iloc[0].o)
    h = d.h.values.astype(float)
    l = d.l.values.astype(float)
    c = d.c.values.astype(float)
    m = d.m.values.astype(int)
    n = len(d)
    if n < BASE_WINDOW:
        return None, 'lt_20_bars'
    H = np.maximum.accumulate(h)
    touch = (h == H)
    positions = np.arange(n)
    touch_pos = np.where(touch, positions, -1)
    tau_pos = np.maximum.accumulate(touch_pos)
    tau_m = m[tau_pos]
    age = m - tau_m
    B = pd.Series(l).rolling(BASE_WINDOW, min_periods=BASE_WINDOW).min().values
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = (H - B) / H
    press = c >= (B + PRESS_FRAC * (H - B))
    inplay = H >= INPLAY_MULT * open_930
    costgate = (c - B) >= COST_GATE_PCT * c
    scan = (m >= SCAN_START_M) & (m <= SCAN_END_M)
    ok = scan & (age >= AGE_MIN_MINUTES) & ~np.isnan(B) & (depth <= DEPTH_MAX) & press & inplay & costgate
    idxs = np.where(ok)[0]
    if len(idxs) == 0:
        return None, 'no_qualifying_bar'
    i = int(idxs[0])
    return dict(pos=i, m=int(m[i]), H_t=float(H[i]), B_t=float(B[i]), age=int(age[i]),
                depth=float(depth[i]), close=float(c[i])), 'ok'


def build_signal(d, sigbar, symbol, day):
    """Entry = open of bar t+1, stop = B_t, R = entry - B_t.

    Stop-at-open (entry opens at/below B_t) is KEPT per the PREREG ("the trade is a stop at the
    open"): sao=1, R unit = decision-time R_est = close_t - B_t; walk() books exit = entry and charges
    the round-trip cost (see module docstring correction)."""
    pos = sigbar['pos']
    if pos + 1 >= len(d):
        return None, 'no_next_bar'
    entry_row = d.iloc[pos + 1]
    entry_m, entry = int(entry_row.m), float(entry_row.o)
    B_t = sigbar['B_t']
    R = entry - B_t
    sao = 0
    if R <= 0:
        sao = 1
        R = sigbar['close'] - B_t
    return dict(day=day, symbol=symbol, signal_m=sigbar['m'], H_t=sigbar['H_t'], B_t=B_t,
                age=sigbar['age'], depth=sigbar['depth'], entry_m=entry_m, entry=entry, stop=B_t,
                R=R, r_pct=100.0 * R / entry, price=entry, target=entry + 2.0 * R, sao=sao), None


# ------------------------------------------------------------------------------------------------
# Build-stage machinery (reused near-verbatim from build_pmh_causal.py)
# ------------------------------------------------------------------------------------------------

def bar_density(d, entry_m):
    expected = W.EOD_M - entry_m
    if expected <= 0:
        return 1.0, 0, 0
    have = int(((d.m > entry_m) & (d.m <= W.EOD_M)).sum())
    return have / expected, have, expected


def conservative_fill(entry, stop, target, path_after, entry_m):
    """Same physics as walker.b0_fill but walks the FULL EXPECTED MINUTE GRID; a minute with no bar
    row is treated as a stop touch (conservative)."""
    have = {int(r.m): r for r in path_after.itertuples()}
    for m in range(entry_m + 1, W.EOD_M + 1):
        if m >= W.EOD_M:
            row = have.get(m)
            px = float(row.o) if row is not None else float(list(have.values())[-1].c) if have else entry
            return m, px, 'eod'
        row = have.get(m)
        if row is None:
            return m, float(stop), 'stop_missing_minute'
        if row.l <= stop:
            px = row.o if row.o <= stop else stop
            return int(row.m), float(px), 'stop'
        if row.h >= target:
            return int(row.m), float(target), 'target'
    return W.EOD_M, entry, 'eod'


def d3_pool_sample(cands):
    """Seeded uniform sample of D3_POOL_K universe symbol-days per non-TEST date (any signal status)."""
    rng = random.Random(SEED + 999)
    by_day = {}
    for s, dd in cands:
        if split_of(dd) != 'TEST':
            by_day.setdefault(dd, set()).add(s)
    pool = set()
    for dd in sorted(by_day):
        syms = sorted(by_day[dd])
        for s in (rng.sample(syms, D3_POOL_K) if len(syms) > D3_POOL_K else syms):
            pool.add((s, dd))
    log(f'[d3-pool] {len(pool)} universe symbol-days sampled over {len(by_day)} dates (K={D3_POOL_K})')
    return pool


def one_pass(cands, limit=None, offset=0):
    if limit:
        cands = cands[offset:offset + limit]
    n = len(cands)
    sip_con = sqlite3.connect(BARS_SIP_DB, timeout=30)
    rth_con = sqlite3.connect(BARS_RTH_DB, timeout=30) if os.path.exists(BARS_RTH_DB) else None
    d3_pool = d3_pool_sample(cands)
    pool_frames = []
    sig_rows, path_frames = [], []
    n_no_bars = n_no_signal = n_no_next_bar = n_stop_at_open = n_with_bars = 0
    src_counts = {'sip': 0, 'rth': 0}
    t0 = time.time()
    for i, (symbol, day) in enumerate(cands, 1):
        d, src = fetch_day_bars_dual(sip_con, rth_con, symbol, day)
        if d is None or len(d) < 5:
            n_no_bars += 1
            continue
        n_with_bars += 1
        src_counts[src] += 1
        if (symbol, day) in d3_pool:
            q = d[['m', 'o', 'h', 'l', 'c']].copy()
            q['day'] = day
            q['symbol'] = symbol
            pool_frames.append(q[['day', 'symbol', 'm', 'o', 'h', 'l', 'c']])
        sigbar, reason = find_signal(d)
        if sigbar is None:
            n_no_signal += 1
            continue
        sig, reason2 = build_signal(d, sigbar, symbol, day)
        if sig is None:
            n_no_next_bar += 1
            continue
        if sig['sao']:
            n_stop_at_open += 1
            log(f'[WARNING] {symbol} {day}: entry opened at/below B_t (stop-at-open) -- kept as a '
                f'trade, booked at -cost in R_est units')
        dens, have, expected = bar_density(d, sig['entry_m'])
        sig['bar_density'] = dens
        sig['bars_have'] = have
        sig['bars_expected'] = expected
        sig['bar_source'] = src
        sig_rows.append(sig)
        p = d.copy()   # FULL day cached (D1/D3 placebos need bars before entry_m too)
        p['day'] = day
        p['symbol'] = symbol
        path_frames.append(p[['day', 'symbol', 'm', 'o', 'h', 'l', 'c']])
        if i % 1000 == 0 or i == n:
            log(f'[pass] {i}/{n} | {time.time()-t0:.0f}s | signals={len(sig_rows)} '
                f'no_bars={n_no_bars} no_signal={n_no_signal} stop_at_open={n_stop_at_open} '
                f'no_next_bar={n_no_next_bar} src={src_counts}')
    sip_con.close()
    if rth_con is not None:
        rth_con.close()
    log(f'[pass] DONE {n} candidates in {time.time()-t0:.0f}s -> {len(sig_rows)} signals, '
        f'bar_source={src_counts}, with_bars={n_with_bars}/{n} '
        f'({(n_with_bars/n*100 if n else 0):.1f}%)')
    return sig_rows, path_frames, pool_frames, dict(n_candidates=n, n_with_bars=n_with_bars,
                                                     n_no_bars=n_no_bars, n_no_signal=n_no_signal,
                                                     n_stop_at_open=n_stop_at_open,
                                                     n_no_next_bar=n_no_next_bar)


def write_signals(sig_rows):
    sig = pd.DataFrame(sig_rows)
    if sig.empty:
        log('[signals] WARNING: zero signals found')
        sig.to_parquet(f'{HERE}/signals.parquet', index=False)
        return sig
    sig['split'] = sig.day.map(split_of)
    sig['wk'] = sig.day.map(week_monday)
    sig['half'] = [half_of(dd, s) for dd, s in zip(sig.day, sig.split)]
    sig['half_spread_proxy'] = sig.entry * PROXY_HALF_SPREAD_PCT
    log(f'[signals] {len(sig)} signals | splits={sig.split.value_counts().to_dict()} | '
        f'cost = PROXY 15bp half-spread both legs + 2bp/side slip (no measured NBBO, per PREREG)')
    sig.to_parquet(f'{HERE}/signals.parquet', index=False)
    log(f'[write] signals.parquet ({len(sig)} rows)')
    return sig


def write_paths(path_frames):
    paths = pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame(
        columns=['day', 'symbol', 'm', 'o', 'h', 'l', 'c'])
    paths.to_parquet(f'{HERE}/paths.parquet', index=False)
    log(f'[write] paths.parquet ({len(paths)} rows, {len(path_frames)} groups, FULL RTH day cached)')
    return paths


def simulate_arms(sig, paths):
    """PRIMARY arm: b0_style_fill (C1 target+2R) on bar_density>=DENSITY_MIN signals only, cost via
    score_cells.cost_net (proxy spread + 2bp/side slip). CONSERVATIVE arm: every signal, missing
    minute = stop touch, gross only (descriptive)."""
    idx = paths.sort_values(['day', 'symbol', 'm']).set_index(['day', 'symbol']).sort_index()
    prim_rows, cons_rows = [], []
    n_no_path = 0
    for r in sig.itertuples():
        key = (r.day, r.symbol)
        if key not in idx.index:
            n_no_path += 1
            continue
        g = idx.loc[[key]]
        after = g[g.m > r.entry_m]
        if after.empty:
            n_no_path += 1
            continue
        cexit_m, cexit_px, cwhy = conservative_fill(r.entry, r.stop, r.target, after, r.entry_m)
        craw = (cexit_px - r.entry) / r.R
        cons_rows.append(dict(day=r.day, symbol=r.symbol, raw_rr=craw, why=cwhy,
                               bar_density=r.bar_density, split=r.split))
        if r.bar_density >= DENSITY_MIN:
            exit_m, exit_px, why = b0_style_fill(r.entry, r.stop, r.target, after.itertuples())
            spread_mean = 2 * PROXY_HALF_SPREAD_PCT * r.entry
            cost_R, net_rr = cost_net(r.entry, exit_px, r.R, spread_mean)
            raw_rr = (exit_px - r.entry) / r.R
            prim_rows.append(dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, exit_m=exit_m,
                                   raw_rr=raw_rr, cost_R=cost_R, net_rr=net_rr, why=why,
                                   split=r.split, bar_density=r.bar_density))
    prim = pd.DataFrame(prim_rows)
    cons = pd.DataFrame(cons_rows)
    log(f'[b0] PRIMARY arm (density>={DENSITY_MIN}, C1 exit): {len(prim)}/{len(sig)} simulated '
        f'({n_no_path} had no cached path)')
    log(f'[b0] CONSERVATIVE arm (all signals, missing-minute=stop, C1 exit): {len(cons)}/{len(sig)} simulated')
    return prim, cons


def minute_since_entry_table(sig, paths):
    """Mean/median UNMANAGED R (close vs entry, no stop/target/EOD applied) at fixed minute marks
    since entry, TRAIN+VAL only."""
    idx = paths.sort_values(['day', 'symbol', 'm']).set_index(['day', 'symbol']).sort_index()
    s = sig[sig.split.isin(['TRAIN', 'VAL'])]
    rows = {mk: [] for mk in MARKS}
    for r in s.itertuples():
        key = (r.day, r.symbol)
        if key not in idx.index:
            continue
        g = idx.loc[[key]]
        for mk in MARKS:
            target_m = r.entry_m + mk
            w = g[g.m <= target_m]
            if w.empty:
                continue
            last = w.iloc[-1]
            if last.m < r.entry_m:
                continue
            rows[mk].append((last.c - r.entry) / r.R)
    out = []
    for mk in MARKS:
        v = np.array(rows[mk])
        if len(v) == 0:
            continue
        out.append(dict(minute=mk, n=len(v), mean_R=float(v.mean()), median_R=float(np.median(v)),
                         p10_R=float(np.percentile(v, 10)), p90_R=float(np.percentile(v, 90))))
    return pd.DataFrame(out)


def orb_overlap(sig):
    """Share of signals (TRAIN+VAL) that are also picks in runB_true.csv or runCOMB_true.csv on the
    same (day,symbol)."""
    frames = []
    for p in (ORB_RUNB, ORB_RUNCOMB):
        if os.path.exists(p):
            dd = pd.read_csv(p, usecols=['symbol', 'date', 'entered'],
                              dtype={'symbol': str, 'date': str})
            frames.append(dd)
    if not frames:
        log('[overlap] WARNING: neither runB_true.csv nor runCOMB_true.csv found -- overlap = n/a')
        return 0, 0, 0.0, 0.0
    orb = pd.concat(frames, ignore_index=True).drop_duplicates(['symbol', 'date'])
    orb_pairs = set(map(tuple, orb[['date', 'symbol']].values))
    orb_entered_pairs = set(map(tuple, orb[orb.entered == 1][['date', 'symbol']].values))
    s = sig[sig.split.isin(['TRAIN', 'VAL'])]
    n = len(s)
    if n == 0:
        return 0, 0, 0.0, 0.0
    matched_any = sum(1 for r in s.itertuples() if (r.day, r.symbol) in orb_pairs)
    matched_entered = sum(1 for r in s.itertuples() if (r.day, r.symbol) in orb_entered_pairs)
    share_any = matched_any / n
    share_entered = matched_entered / n
    log(f'[overlap] ORB pool={len(orb_pairs)} pairs. {matched_any}/{n} signals (TRAIN+VAL) also '
        f'ORB picks (any) = {share_any*100:.1f}%; entered-only = {share_entered*100:.1f}%')
    return matched_any, n, share_any, share_entered


def write_drift(avail, prim, cons, mset_table, overlap, bar_source_counts, stats):
    matched_any, n_overlap, share_any, share_entered = overlap
    total_sig = len(cons)
    n_dropped_density = total_sig - len(prim)
    dropped_share = n_dropped_density / total_sig if total_sig else 0.0
    winners = cons[cons.raw_rr > 0]
    losers = cons[cons.raw_rr <= 0]
    w_drop = (winners.bar_density < DENSITY_MIN).mean() if len(winners) else float('nan')
    l_drop = (losers.bar_density < DENSITY_MIN).mean() if len(losers) else float('nan')
    gap_pp = abs(w_drop - l_drop) * 100 if pd.notna(w_drop) and pd.notna(l_drop) else float('nan')

    lines = ['# DRIFT.md -- base-under-the-high descriptive exhibit', '',
             'PREREG.md "Exhibits first" #1, format of research/hod_exit_lab/DRIFT.md. Universe = '
             'research/hod_pmh_causal/pm_candidates.csv (23,767 ORB wide-seed causal symbol-days). '
             'No pre-market data used (dropped per PREREG). TRAIN/VAL only below; TEST is sealed.',
             '',
             '## Availability rail (all candidates, RTH 1-min bars)', '',
             f"- Candidates with usable RTH 1-min bars: {avail['n_with_bars']}/{avail['n_candidates']} "
             f"({avail['share']*100:.1f}%) -- "
             f"{'PASS (>=80% rail)' if avail['share'] >= 0.80 else '**VOID -- below the 80% availability rail**'}",
             f"- No usable bars: {stats['n_no_bars']} | no qualifying signal bar: {stats['n_no_signal']} | "
             f"stop-at-open (kept in the cells at -cost, excluded from this exhibit): {stats['n_stop_at_open']} | no bar after signal: {stats['n_no_next_bar']}",
             '',
             '## Bar-density-in-hold rail (signals only)', '',
             f"- Total signals (signal bar found, entry/stop obtainable): {total_sig}",
             f"- Dropped from PRIMARY arm (bar_density < {DENSITY_MIN*100:.0f}%): {n_dropped_density} "
             f"({dropped_share*100:.1f}% of signals)",
             f"- Bar source: {bar_source_counts}",
             f"- Winner/loser missingness gap (conservative-arm raw_rr>0 vs <=0, share with "
             f"bar_density<{DENSITY_MIN*100:.0f}%): winners={w_drop*100:.1f}%, losers={l_drop*100:.1f}%, "
             f"gap={gap_pp:.1f}pp {'(<=5pp OK)' if pd.notna(gap_pp) and gap_pp <= 5 else '(>5pp -- flag)' if pd.notna(gap_pp) else '(n/a)'}",
             '',
             '## Cost', '',
             '- No nbbo.csv coverage for these minutes (PREREG): proxy 15bp half-spread on BOTH legs '
             '+ 2bp/side slip, for 100% of signals. Gross (raw_rr) vs net (proxy-cost) both reported '
             'below (C1 exit, used descriptively for this exhibit).', '',
             '## PRIMARY arm by split (gross vs net, proxy cost, C1 exit)', '',
             '| split | n | mean gross R | median gross R | mean net R | median net R |',
             '|---|---|---|---|---|---|']
    for sp, g in prim.groupby('split'):
        lines.append(f"| {sp} | {len(g)} | {g.raw_rr.mean():.3f} | {g.raw_rr.median():.3f} | "
                      f"{g.net_rr.mean():.3f} | {g.net_rr.median():.3f} |")
    lines += ['', '## Minute-since-entry table (unmanaged path, TRAIN+VAL, close-vs-entry in R, '
              'no stop/target/EOD applied)', '',
              '| minute | n | mean R | median R | p10 R | p90 R |', '|---|---|---|---|---|---|']
    for r in mset_table.itertuples():
        lines.append(f"| {r.minute} | {r.n} | {r.mean_R:.3f} | {r.median_R:.3f} | {r.p10_R:.3f} | {r.p90_R:.3f} |")
    unmanaged_2h = mset_table[mset_table.minute == 120]
    if len(unmanaged_2h):
        m2 = unmanaged_2h.iloc[0]
        verdict = ('as informationless as HOD-break -- reported as such' if m2.mean_R <= 0.15
                   else 'shows drift above the +0.15R HOD-break threshold')
        lines += ['', f"Unmanaged mean R at 2h (minute=120) = {m2.mean_R:.3f}R (n={int(m2.n)}) -- {verdict}."]
    overlap_verdict = ('>50% -- population is ORB by another entry; compare to the ORB book, not HOD.'
                        if share_any > 0.50 else '<=50% -- a distinct population from the ORB production book.')
    lines += ['', '## ORB-overlap (share of signals that are also ORB picks, same day+symbol)', '',
              f"Any-row match: **{share_any*100:.1f}%** ({matched_any}/{n_overlap}). "
              f"entered==1-only match: {share_entered*100:.1f}%. {overlap_verdict}", '']
    with open(f'{HERE}/DRIFT.md', 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    log(f'[drift] DRIFT.md written ({len(lines)} lines)')


# ------------------------------------------------------------------------------------------------
# DECOMP.md -- the owner's question (PREREG.md "Exhibits first" #2) -- NEW
# ------------------------------------------------------------------------------------------------

def compute_decomp(sig, idx):
    """Per base entry: does the stock later close above H_t (outcome label, descriptive only)?
    mean C1 net R broken out by break/never-break/all (unconditional -- prices in failed bases);
    for the breaking subset, base-entry C1 R vs a same-name-day HOD-break-equivalent entry (next
    open after the first close>H_t, same stop B_t, same C1 exit)."""
    rows = []
    for r in sig.itertuples():
        key = (r.day, r.symbol)
        if key not in idx.index:
            continue
        g = idx.loc[[key]]
        after_entry = g[(g.m > r.entry_m) & (g.m <= EOD_M)]
        if after_entry.empty:
            continue
        base_res = fill_c1(r.entry, r.stop, r.R, after_entry.itertuples())
        if base_res is None:
            continue
        _, base_exit_px, _ = base_res
        spread_mean = 2 * PROXY_HALF_SPREAD_PCT * r.entry
        _, base_net_R = cost_net(r.entry, base_exit_px, r.R, spread_mean)

        brk = after_entry[after_entry.c > r.H_t]
        breaks = len(brk) > 0
        hod_net_R = float('nan')
        if breaks:
            brk_m = int(brk.iloc[0].m)
            nxt = g[g.m > brk_m]
            if not nxt.empty:
                hod_entry_row = nxt.iloc[0]
                hod_entry_m, hod_entry = int(hod_entry_row.m), float(hod_entry_row.o)
                hod_R = hod_entry - r.B_t
                if hod_R > 0:
                    hod_after = g[(g.m > hod_entry_m) & (g.m <= EOD_M)]
                    if not hod_after.empty:
                        hod_res = fill_c1(hod_entry, r.B_t, hod_R, hod_after.itertuples())
                        if hod_res is not None:
                            _, hod_exit_px, _ = hod_res
                            hod_spread = 2 * PROXY_HALF_SPREAD_PCT * hod_entry
                            _, hod_net_R = cost_net(hod_entry, hod_exit_px, hod_R, hod_spread)
        rows.append(dict(day=r.day, symbol=r.symbol, split=r.split, breaks=breaks,
                          base_net_R=base_net_R, hod_net_R=hod_net_R))
    return pd.DataFrame(rows)


def write_decomp(dec):
    lines = ["# DECOMP.md -- the owner's question: \"entering an R earlier\"", '',
             'PREREG.md "Exhibits first" #2. Base entries (the signal shared by cells 1,400-1,402) '
             'vs a same-name-day counterfactual HOD-break entry (next open after the first '
             'close>H_t following the base entry, same stop B_t, same C1 target+2R exit). '
             'TRAIN/VAL only, TEST sealed. Both legs use the PROXY cost (no measured NBBO for '
             'these minutes).', '']
    for sp in ('TRAIN', 'VAL'):
        d = dec[dec.split == sp]
        if d.empty:
            lines += [f'## {sp}', '', '(no signals)', '']
            continue
        n = len(d)
        share_break = float(d.breaks.mean())
        mean_break = d[d.breaks].base_net_R.mean() if d.breaks.any() else float('nan')
        mean_nobreak = d[~d.breaks].base_net_R.mean() if (~d.breaks).any() else float('nan')
        mean_all = d.base_net_R.mean()
        dbrk = d[d.breaks & d.hod_net_R.notna()]
        paired_base = dbrk.base_net_R.mean() if len(dbrk) else float('nan')
        paired_hod = dbrk.hod_net_R.mean() if len(dbrk) else float('nan')
        paired_delta = (dbrk.base_net_R - dbrk.hod_net_R).mean() if len(dbrk) else float('nan')
        lines += [f'## {sp} (n={n})', '',
                  f'- Share of base entries whose stock later closes above H_t (breaks): '
                  f'{share_break*100:.1f}%',
                  f'- Mean C1 net R, base entries that break: {mean_break:.3f} (n={int(d.breaks.sum())})',
                  f'- Mean C1 net R, base entries that never break: {mean_nobreak:.3f} '
                  f'(n={int((~d.breaks).sum())})',
                  f'- Mean C1 net R, ALL base entries (unconditional -- prices in the failed bases): '
                  f'{mean_all:.3f} (n={n})',
                  f'- Breaking subset ONLY, paired same-name-day comparison (n={len(dbrk)}): '
                  f'base-entry C1 R = {paired_base:.3f} vs HOD-break-equivalent-entry C1 R = '
                  f'{paired_hod:.3f} -- "entering an R earlier" delta = {paired_delta:.3f} R',
                  '']
    with open(f'{HERE}/DECOMP.md', 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    log('wrote DECOMP.md')


# ------------------------------------------------------------------------------------------------
# Cell exit rules -- each fill_fn(entry, stop, R, bars) -> (exit_m, exit_price, why) or None
# ------------------------------------------------------------------------------------------------

def fill_c1(entry, stop, R, bars):
    return b0_style_fill(entry, stop, entry + 2.0 * R, bars)


def fill_c2(entry, stop, R, bars):
    return x1_no_target(entry, stop, R, bars)


def fill_c3(entry, stop, R, bars):
    """Breakeven lock: high >= entry+1R -> stop=entry from the NEXT bar, no target, flat 15:55."""
    cur_stop = stop
    for row in bars:
        if row.m >= EOD_M:
            return int(row.m), float(row.o), 'eod'
        if row.l <= cur_stop:
            px = row.o if row.o <= cur_stop else cur_stop
            return int(row.m), float(px), 'stop'
        if row.h >= entry + 1.0 * R:
            cur_stop = max(cur_stop, entry)
    return None


CELL_FILL = {'1400': fill_c1, '1401': fill_c2, '1402': fill_c3}
CELL_NAME = {'1400': 'C1 target +2R, stop, 15:55', '1401': 'C2 no target, stop, 15:55',
             '1402': 'C3 breakeven lock at +1R, no target, 15:55'}


def walk(sig_frame, idx, fill_fn):
    rows = []
    for r in sig_frame.itertuples():
        key = (r.day, r.symbol)
        if getattr(r, 'sao', 0):
            spread_mean = 2 * PROXY_HALF_SPREAD_PCT * r.entry
            cost_R, net_R = cost_net(r.entry, r.entry, r.R, spread_mean)
            rows.append(dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, signal_m=r.signal_m,
                             exit_m=r.entry_m, exit_price=r.entry, why='stop_at_open', split=r.split,
                             half=r.half, wk=r.wk, entry=r.entry, stop=r.stop, R=r.R, price=r.price,
                             net_R=net_R, cost_R=cost_R))
            continue
        if key not in idx.index:
            continue
        g = idx.loc[[key]]
        bars = g[(g.m > r.entry_m) & (g.m <= EOD_M)]
        if bars.empty:
            continue
        res = fill_fn(r.entry, r.stop, r.R, bars.itertuples())
        if res is None:
            continue
        exit_m, exit_px, why = res
        spread_mean = 2 * PROXY_HALF_SPREAD_PCT * r.entry
        cost_R, net_R = cost_net(r.entry, exit_px, r.R, spread_mean)
        rows.append(dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, signal_m=r.signal_m,
                          exit_m=exit_m, exit_price=exit_px, why=why, split=r.split, half=r.half,
                          wk=r.wk, entry=r.entry, stop=r.stop, R=r.R, price=r.price,
                          net_R=net_R, cost_R=cost_R))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
# D1 / D3 placebos (PREREG.md "Placebos" -- always C1 exit, regardless of the cell under test)
# ------------------------------------------------------------------------------------------------

def _floor_ok(entry_r, stop_r):
    Rr = entry_r - stop_r
    return Rr > 0 and (entry_r - stop_r) >= COST_GATE_PCT * entry_r, Rr


def d1_placebo(work, idx, rng):
    """Same name-day, seeded random minute in 10:00-14:00, stop = min low of the PRIOR 20 bars
    (excludes the draw bar), same 1.5% floor (<=5 redraws per draw else dropped), C1 exit."""
    vals = []
    for r in work.itertuples():
        key = (r.day, r.symbol)
        if key not in idx.index:
            continue
        g = idx.loc[[key]]
        draws = []
        for _ in range(N_DRAWS):
            for _try in range(PLACEBO_MAX_REDRAWS):
                m_r = rng.randint(SCAN_START_M, SCAN_END_M)
                prior = g[(g.m >= m_r - BASE_WINDOW) & (g.m < m_r)]
                entry_bar = g[g.m == m_r]
                if len(prior) < BASE_WINDOW or entry_bar.empty:
                    continue
                entry_r = float(entry_bar.iloc[0].o)
                stop_r = float(prior.l.min())
                floor_ok, Rr = _floor_ok(entry_r, stop_r)
                if not floor_ok:
                    continue
                after = g[g.m > m_r]
                if after.empty:
                    continue
                res = fill_c1(entry_r, stop_r, Rr, after.itertuples())
                if res is None:
                    continue
                _, exit_px, _ = res
                spread_mean = 2 * PROXY_HALF_SPREAD_PCT * entry_r
                _, net_R = cost_net(entry_r, exit_px, Rr, spread_mean)
                if not pd.isna(net_R):
                    draws.append(net_R)
                break
        if draws:
            vals.append(float(np.mean(draws)))
    return vals


def d3_placebo(work, by_date, idx, rng):
    """Same date, the real signal's own clock minute (entry_m), a randomly drawn OTHER symbol from
    by_date[day] (the caller picks the pool: the PREREG universe sample for the pass bar, or the
    signal-producing symbols as a diagnostic); stop = that symbol's prior-20-bar low before entry_m;
    same 1.5% floor (<=5 redraws of the OTHER SYMBOL, else dropped), C1 exit."""
    vals, covered, total = [], 0, 0
    for r in work.itertuples():
        total += 1
        others = [s for s in by_date.get(r.day, []) if s != r.symbol]
        if not others:
            continue
        draws = []
        for _ in range(N_DRAWS):
            for _try in range(PLACEBO_MAX_REDRAWS):
                osym = others[rng.randrange(len(others))]
                okey = (r.day, osym)
                if okey not in idx.index:
                    continue
                g = idx.loc[[okey]]
                prior = g[(g.m >= r.entry_m - BASE_WINDOW) & (g.m < r.entry_m)]
                entry_bar = g[g.m == r.entry_m]
                if len(prior) < BASE_WINDOW or entry_bar.empty:
                    continue
                entry_r = float(entry_bar.iloc[0].o)
                stop_r = float(prior.l.min())
                floor_ok, Rr = _floor_ok(entry_r, stop_r)
                if not floor_ok:
                    continue
                after = g[g.m > r.entry_m]
                if after.empty:
                    continue
                res = fill_c1(entry_r, stop_r, Rr, after.itertuples())
                if res is None:
                    continue
                _, exit_px, _ = res
                spread_mean = 2 * PROXY_HALF_SPREAD_PCT * entry_r
                _, net_R = cost_net(entry_r, exit_px, Rr, spread_mean)
                if not pd.isna(net_R):
                    draws.append(net_R)
                break
        if draws:
            vals.append(float(np.mean(draws)))
            covered += 1
    return vals, covered, total


# ------------------------------------------------------------------------------------------------
# Slot simulator (first-12/day, 4-concurrent) -- research/hod_exit_lab/score_pass2.py::simulate_slots
# ------------------------------------------------------------------------------------------------

def simulate_slots(trades, concurrent_cap=CONCURRENT_CAP, daily_cap=DAILY_CAP):
    keep = pd.Series(False, index=trades.index)
    for day, g in trades.groupby('day'):
        g = g.sort_values('entry_m')
        open_exits, daily_count = [], 0
        for row in g.itertuples():
            open_exits = [x for x in open_exits if x > row.entry_m]
            if len(open_exits) < concurrent_cap and daily_count < daily_cap:
                keep.loc[row.Index] = True
                open_exits.append(row.exit_m)
                daily_count += 1
    return keep


def score_cell(cell_id, work, d1_val_mean, d3_val_mean, slot_fills_wk):
    out = dict(id=cell_id, name=CELL_NAME[cell_id])
    tr = work[work.split == 'TRAIN']
    va = work[work.split == 'VAL']
    h1 = tr[tr.half == 'H1']
    h2 = tr[tr.half == 'H2']
    out['train_n'], out['val_n'] = int(len(tr)), int(len(va))
    out['train_R'] = float(tr.net_R.mean()) if len(tr) else float('nan')
    out['val_R'] = float(va.net_R.mean()) if len(va) else float('nan')
    out['h1_R'] = float(h1.net_R.mean()) if len(h1) else float('nan')
    out['h2_R'] = float(h2.net_R.mean()) if len(h2) else float('nan')
    t, ndays = day_clustered_t(va.net_R, va.day) if len(va) else (float('nan'), 0)
    out['val_t'], out['val_t_ndays'] = t, ndays
    out['extop5_train'] = ex_top5(tr.net_R) if len(tr) else float('nan')
    out['extop5_val'] = ex_top5(va.net_R) if len(va) else float('nan')
    out['val_fills_wk'] = slot_fills_wk
    out['d1_val'] = d1_val_mean
    out['d3_val'] = d3_val_mean

    rule_R = (out['train_R'] >= PASS_BAR_R) and (out['val_R'] >= PASS_BAR_R)
    rule_t = (not pd.isna(t)) and (t >= PASS_BAR_VAL_T)
    rule_halves = (out['h1_R'] > 0) and (out['h2_R'] > 0)
    rule_et5 = (out['extop5_train'] > 0) and (out['extop5_val'] > 0)
    rule_fills = out['val_fills_wk'] >= 3.0
    rule_d1 = (not pd.isna(d1_val_mean)) and (out['val_R'] - d1_val_mean >= PLACEBO_BEAT_R)
    rule_d3 = (not pd.isna(d3_val_mean)) and (out['val_R'] - d3_val_mean >= PLACEBO_BEAT_R)
    out['pass'] = bool(rule_R and rule_t and rule_halves and rule_et5 and rule_fills
                        and rule_d1 and rule_d3)
    out['_rules'] = dict(rule_R=bool(rule_R), rule_t=bool(rule_t), rule_halves=bool(rule_halves),
                          rule_et5=bool(rule_et5), rule_fills=bool(rule_fills),
                          rule_d1=bool(rule_d1), rule_d3=bool(rule_d3))
    return out


def run_cadence(cell_id, work):
    va = work[work.split == 'VAL'].copy()
    if va.empty:
        return dict(error='no VAL trades')
    csvp = f'{TRADES_DIR}/{cell_id}.csv'
    va.assign(date=va.day, pnl_R=va.net_R).to_csv(csvp, index=False)
    try:
        p = subprocess.run(['python3', CADENCE, '--trades', csvp, '--split', 'VAL'],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
        return dict(returncode=p.returncode, stdout=p.stdout[-4000:], stderr=p.stderr[-1500:])
    except Exception as e:
        return dict(error=str(e))


def fmt(v, nd=3):
    if v is None:
        return 'n/a'
    try:
        if pd.isna(v):
            return 'n/a'
    except TypeError:
        pass
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f'{v:.{nd}f}'


def write_cells(results):
    clean = []
    for r in results:
        c = {k: v for k, v in r.items() if not k.startswith('_')}
        c['cadence_stdout'] = c.get('cadence', {}).get('stdout', '')
        clean.append(c)
    with open(f'{HERE}/cells.json', 'w') as fh:
        json.dump(dict(cells=clean), fh, indent=2, default=str)
    log('wrote cells.json')

    lines = ['# CELLS.md -- base-under-the-high (PREREG.md, cells 1,400-1,402)', '',
             'Population: research/hod_pmh_causal/pm_candidates.csv (23,767 ORB wide-seed causal '
             'symbol-days, gap>=3%/open $3-50/prior-vol>=500K, all knowable at 09:30:00 ET). '
             'Signal: FIRST 1-min bar 10:00-14:00 ET pressing a >=20-min-old high with a base '
             '<=6% deep, cost-gated >=1.5%. TRAIN=2025 (halves split 2025-07-02), VAL=2026-01..05, '
             'TEST sealed. Cost = PROXY 15bp half-spread both legs + 2bp/side slip (no measured '
             'NBBO for these minutes, per PREREG) -- any pass is re-scored on measured Alpaca NBBO '
             'before being reported as a pass.', '',
             '## Placebos', '',
             'D1: same name-day, seeded random minute in 10:00-14:00, stop = min low of the PRIOR '
             '20 bars (excludes the draw bar itself), same 1.5% floor, <=5 redraws per draw else '
             'that draw is dropped, C1 exit. D3: a random OTHER signal-producing symbol on the SAME '
             'DATE (pool = other signal symbol-days that date -- paths are only cached where a '
             'signal fired, not the full raw candidate universe), entered at the real signal\'s own '
             'clock minute, its own prior-20-bar-low stop and floor, C1 exit. Both placebos ALWAYS '
             'use the C1 (target +2R) exit rule regardless of which cell is being tested, per '
             'PREREG. N_DRAWS=8 seeded draws per VAL trade.', '',
             '## Cells', '',
             '| cell | name | train_n | val_n | train_R | val_R | val_t | h1_R | h2_R | '
             'extop5_tr | extop5_val | fills/wk(VAL,4c/12d) | D1(VAL) | D3(VAL) | pass |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        lines.append('| ' + ' | '.join([
            r['id'], r['name'], fmt(r['train_n'], 0), fmt(r['val_n'], 0), fmt(r['train_R']),
            fmt(r['val_R']), fmt(r['val_t']), fmt(r['h1_R']), fmt(r['h2_R']),
            fmt(r['extop5_train']), fmt(r['extop5_val']), fmt(r['val_fills_wk'], 2),
            fmt(r['d1_val']), fmt(r['d3_val']), str(r['pass'])]) + ' |')
    lines += ['', '## Pass-bar rule detail per cell', '']
    for r in results:
        lines.append(f"- **{r['id']}** ({r['name']}): {r['_rules']} | D1 coverage "
                      f"{fmt(r['d1_coverage']*100 if not pd.isna(r['d1_coverage']) else float('nan'),1)}% "
                      f"| D3 (universe pool, pass bar) coverage {fmt(r['d3_coverage']*100 if not pd.isna(r['d3_coverage']) else float('nan'),1)}% "
                      f"| D3 signal-pool diagnostic {fmt(r.get('d3_signalpool_val', float('nan')))} "
                      f"| stop-at-open VAL trades {r.get('n_stop_at_open_val', 0)}")
    lines += ['', '## Cadence bar (scripts/cadence_bar.py --split VAL), per cell', '']
    for r in results:
        cad = r.get('cadence', {})
        lines.append(f"### {r['id']}")
        lines.append('```')
        lines.append(cad.get('stdout', cad.get('error', 'n/a')).strip() or '(no stdout)')
        lines.append('```')
    with open(f'{HERE}/CELLS.md', 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    log('wrote CELLS.md')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--smoke-offset', type=int, default=0,
                    help='with --smoke: start at this candidate index (to smoke a VAL slice)')
    args = ap.parse_args()
    os.makedirs(TRADES_DIR, exist_ok=True)

    cands = load_candidates()
    if not args.smoke:
        wait_for_db_window()

    sig_rows, path_frames, pool_frames, stats = one_pass(cands, limit=args.smoke or None,
                                                          offset=args.smoke_offset)
    avail = dict(stats)
    avail['share'] = stats['n_with_bars'] / stats['n_candidates'] if stats['n_candidates'] else 0.0
    log(f"[availability] {avail['n_with_bars']}/{avail['n_candidates']} candidates had usable RTH "
        f"1-min bars ({avail['share']*100:.1f}%) -- "
        f"{'PASS' if avail['share'] >= 0.80 else 'VOID -- below 80% rail'}")

    sig = write_signals(sig_rows)
    paths = write_paths(path_frames)

    if sig.empty or paths.empty:
        log('[ALL DONE] zero signals/paths -- nothing further to score')
        print('ALL DONE')
        return

    bar_source_counts = sig.bar_source.value_counts().to_dict()
    sig_walkable = sig[sig.sao == 0]          # stop-at-open rows are scored in the cells, not walked here
    prim, cons = simulate_arms(sig_walkable, paths)
    mset = minute_since_entry_table(sig_walkable, paths)
    overlap = orb_overlap(sig) if not args.smoke else (0, 0, 0.0, 0.0)
    write_drift(avail, prim, cons, mset, overlap, bar_source_counts, stats)

    # ---------------- scoring stage ----------------
    scored = sig[sig.split.isin(['TRAIN', 'VAL'])].copy()
    idx = paths.sort_values(['day', 'symbol', 'm']).set_index(['day', 'symbol']).sort_index()
    log(f'[score] scoring population TRAIN={(scored.split=="TRAIN").sum()} '
        f'VAL={(scored.split=="VAL").sum()} (TEST sealed)')

    dec = compute_decomp(scored[scored.sao == 0], idx)
    write_decomp(dec)
    n_sao = scored.groupby('split').sao.sum().to_dict()
    with open(f'{HERE}/DECOMP.md', 'a') as fh:
        fh.write(f'\nStop-at-open base entries (entry bar opened at/below the base low) are excluded from '
                 f'this exhibit and scored in the cells at -cost: {n_sao}.\n')

    pool = (pd.concat(pool_frames, ignore_index=True) if pool_frames
            else pd.DataFrame(columns=['day', 'symbol', 'm', 'o', 'h', 'l', 'c']))
    if len(pool):
        pool.to_parquet(f'{HERE}/pool_paths.parquet', index=False)
    pool_idx = pool.sort_values(['day', 'symbol', 'm']).set_index(['day', 'symbol']).sort_index()
    pool_by_date = (pool.groupby('day').symbol.apply(lambda s: sorted(set(s))).to_dict()
                    if len(pool) else {})
    sigpool_by_date = scored.groupby('day').symbol.apply(lambda s: sorted(set(s))).to_dict()
    log(f'[d3-pool] cached {pool_idx.index.nunique() if len(pool) else 0} universe symbol-days for D3')

    c1_work = walk(scored, idx, fill_c1)
    c2_work = walk(scored, idx, fill_c2)
    c3_work = walk(scored, idx, fill_c3)
    log(f'C1 n={len(c1_work)} C2 n={len(c2_work)} C3 n={len(c3_work)}')
    works = {'1400': c1_work, '1401': c2_work, '1402': c3_work}
    cell_order = {'1400': 1, '1401': 2, '1402': 3}

    results = []
    for cid, work in works.items():
        va = work[work.split == 'VAL']
        rng1 = random.Random(SEED + cell_order[cid])
        rng3 = random.Random(SEED + 500 + cell_order[cid])
        log(f'{cid}: D1 placebo ({len(va)} VAL trades x {N_DRAWS} draws)...')
        d1v = d1_placebo(va, idx, rng1)
        log(f'{cid}: D3 placebo ({len(va)} VAL trades x up to {N_DRAWS} draws)...')
        d3v, cov, tot = d3_placebo(va, pool_by_date, pool_idx, rng3)
        d3sv, _, _ = d3_placebo(va, sigpool_by_date, idx, random.Random(SEED + 700 + cell_order[cid]))

        d1_mean = float(np.mean(d1v)) if d1v else float('nan')
        d3_mean = float(np.mean(d3v)) if d3v else float('nan')
        d3s_mean = float(np.mean(d3sv)) if d3sv else float('nan')

        slot_keep = simulate_slots(va) if len(va) else pd.Series(dtype=bool)
        nwk = va.loc[slot_keep.index[slot_keep]].wk.nunique() if len(va) else 0
        fills_wk = float(slot_keep.sum() / nwk) if nwk else 0.0

        r = score_cell(cid, work, d1_mean, d3_mean, fills_wk)
        r['d1_coverage'] = len(d1v) / len(va) if len(va) else 0.0
        r['d3_coverage'] = cov / tot if tot else 0.0
        r['d3_signalpool_val'] = d3s_mean
        r['n_stop_at_open_val'] = int((va.why == 'stop_at_open').sum()) if len(va) else 0
        r['cadence'] = run_cadence(cid, work)
        results.append(r)
        work.assign(date=work.day, pnl_R=work.net_R).to_csv(f'{TRADES_DIR}/{cid}_all.csv', index=False)
        log(f'{cid}: train_R={fmt(r["train_R"])} val_R={fmt(r["val_R"])} val_t={fmt(r["val_t"])} '
            f'd1={fmt(d1_mean)} d3={fmt(d3_mean)} fills/wk={fmt(fills_wk,2)} pass={r["pass"]}')

    write_cells(results)
    log('ALL DONE')
    print('ALL DONE')


if __name__ == '__main__':
    main()
