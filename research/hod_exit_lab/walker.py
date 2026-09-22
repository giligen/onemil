#!/usr/bin/env python3
"""HOD-break Exit Lab — walker.py (PREREG.md, cells 1,359-1,379).

ONE pass over bars_sip.db (query by symbol+day, never a table scan) that walks every HOD-break
signal in features.csv from its entry minute to the 15:55 bar and caches the per-minute path. Every
later exit-cell study reads the cached parquet files this script writes; the DB is never touched
again.

Pipeline (see PREREG.md "Method"):
  (a) load_signals()        -> the population (features.csv + nbbo.csv measured spread)
      one_db_pass()          -> paths.parquet (per day,symbol,minute) + signals.parquet
  (b) simulate_b0()          -> b0_trades.csv (B0 fill physics + PREREG cost, literal prose)
  (c) reproduction_gate()    -> compares B0 TRAIN mean R to the causal-filter study's baseline
  (d) drift_profile()        -> DRIFT.md (excursion/MFE/MAE/give-back exhibit; no cell, descriptive)

Two verified deviations from FACTS.md, discovered empirically and handled explicitly (not silently
patched over) - see the printed [schema-check] block at start of run:
  1. bars_sip.db's `t` column is an ISO-8601 UTC timestamp, NOT a bare Eastern "HH:MM" string, and
     the table is NOT pre-trimmed to regular hours (rows exist past 21:00 UTC / 16:00 ET). We parse
     as UTC and convert to America/New_York ourselves, then filter to [09:30,15:59] ET.
  2. features.csv currently holds 15,656 rows (TRAIN 7390 + VAL 4745 + TEST 3521), not the 12,135
     PREREG.md cites — 12,135 = TRAIN+VAL only; PREREG's population count excludes the sealed TEST
     rows, which this script also never scores (TEST is loaded/cached for future use, never in the
     reproduction gate or DRIFT.md).

Usage:
  python3 walker.py            # full run: waits out the DB blackout window, then (a)(b)(c)(d)
  python3 walker.py --smoke N  # first N signals only, no blackout wait, no file writes to the
                                # real paths.parquet/signals.parquet (writes to *_smoke.parquet) -
                                # a fast correctness check before committing to the full pass.
"""
import argparse
import os
import sys
import sqlite3
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
LAB = f'{ROOT}/research/hod_exit_lab'
CF_DIR = f'{ROOT}/research/bf_zero/causal_filter'
FEATURES_CSV = f'{CF_DIR}/features.csv'
NBBO_CSV = f'{CF_DIR}/nbbo.csv'
BARS_DB = f'{ROOT}/research/bf_zero/bars_sip.db'
SPY_PARQUET = f'{ROOT}/research/index_orb/cache/SPY_1min.parquet'

OPEN_M = 570          # 09:30 ET, minutes since midnight
EOD_M = 955            # 15:55 ET force-flat minute (matches build_candidates.py / hod_break.py)
CLOSE_M = 959           # 15:59 ET, last regular-hours bar
TARGET_R = 2.0          # B0 target = entry + 2R (PREREG.md line 13-14)
SLIP_BP = 0.0002        # 2 bp per side (PREREG.md cost sentence)
BLOCK_START, BLOCK_END = 1325, 2005   # UTC HHMM heavy-DB blackout (live engine hours)
HB_EDGES = [569, 585, 600, 660, 780, 960]
HB_LAB = ['09:30-09:45', '09:45-10:00', '10:00-11:00', '11:00-13:00', '13:00+']


def log(msg):
    """Verbose progress line, flushed immediately (nohup-safe)."""
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)


def wait_for_db_window(poll_s=60):
    """Block while UTC clock is inside [BLOCK_START, BLOCK_END) — the live-engine hours PREREG.md
    forbids heavy DB work during. No-ops immediately if already outside the window."""
    while True:
        now = int(datetime.now(timezone.utc).strftime('%H%M'))
        if not (BLOCK_START <= now < BLOCK_END):
            log(f'DB window clear (UTC {now:04d}) — proceeding')
            return
        log(f'UTC {now:04d} inside blackout [{BLOCK_START},{BLOCK_END}) — sleeping {poll_s}s')
        time.sleep(poll_s)


def schema_check():
    """Print the two documented FACTS.md deviations with live evidence, once, at start of run."""
    con = sqlite3.connect(BARS_DB)
    row = con.execute("SELECT t FROM bars WHERE symbol='BG' AND day='2025-01-10' ORDER BY t LIMIT 1").fetchone()
    last = con.execute("SELECT t FROM bars WHERE symbol='BG' AND day='2025-01-10' ORDER BY t DESC LIMIT 1").fetchone()
    con.close()
    log(f'[schema-check] bars.t sample: first={row[0]!r} last={last[0]!r} '
        f'(ISO-8601 UTC, extended hours present — FACTS.md documents bare HH:MM/regular-hours-only; '
        f'we parse+filter ourselves)')


def load_signals():
    """Part (a), population. Reads features.csv, merges nbbo.csv's per-(day,symbol) measured NBBO
    spread (drop_duplicates(['day','symbol']), keep='first' — mirrors cells.py's load() exactly),
    and derives R (dollar risk = entry-stop), half_spread (0.5 x full spread, $) and target
    (entry + 2R). Returns the full signals frame (TRAIN+VAL+TEST; TEST is carried but never scored
    downstream)."""
    f = pd.read_csv(FEATURES_CSV, dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
    n = pd.read_csv(NBBO_CSV, dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
    n = n.drop_duplicates(['day', 'symbol'])[['day', 'symbol', 'spread_mean', 'ask_dec', 'bid_dec']]
    f = f.merge(n, on=['day', 'symbol'], how='left')
    f['R'] = f.entry - f.stop
    f['half_spread'] = 0.5 * f.spread_mean
    f['target'] = f.entry + TARGET_R * f.R
    sig = f[['day', 'symbol', 'entry_m', 'entry', 'stop', 'target', 'R', 'r_pct', 'price',
             'half_spread', 'spread_mean', 'split', 'half', 'wk']].copy()
    ng = sig[['day', 'symbol']].drop_duplicates().shape[0]
    log(f'[signals] {len(sig)} signals loaded, {ng} unique (day,symbol) groups, '
        f'splits={sig.split.value_counts().to_dict()}, '
        f'NBBO coverage={sig.spread_mean.notna().mean()*100:.1f}%')
    return sig


def load_spy():
    """Load SPY 1-min bars once (24MB); index by 'YYYY-MM-DD HH:MM' (Eastern) -> close. Coverage
    ends 2026-05-29 — TEST-split days (Jun 2026+) come back NaN, which is expected and harmless
    (TEST is never scored)."""
    s = pd.read_parquet(SPY_PARQUET)
    ts = s.timestamp.dt.tz_convert('America/New_York')
    key = ts.dt.strftime('%Y-%m-%d %H:%M')
    d = dict(zip(key, s['close'].values))
    log(f'[spy] {len(d)} minute bars loaded, {ts.min()} .. {ts.max()}')
    return d


def fetch_day_bars(con, symbol, day):
    """ONE SQL query for this (symbol, day) — indexed, never a scan. `t` is ISO-8601 UTC; convert
    to America/New_York, derive minute-of-day `m`, and restrict to regular hours [09:30,15:59] ET
    (the DB itself is NOT pre-trimmed — see schema_check)."""
    cur = con.execute('SELECT t,o,h,l,c,v FROM bars WHERE symbol=? AND day=? ORDER BY t', (symbol, day))
    rows = cur.fetchall()
    if not rows:
        return None
    d = pd.DataFrame(rows, columns=['t', 'o', 'h', 'l', 'c', 'v'])
    ts = pd.to_datetime(d.t, utc=True).dt.tz_convert('America/New_York')
    d['m'] = ts.dt.hour * 60 + ts.dt.minute
    d['hhmm'] = ts.dt.strftime('%H:%M')
    d = d[(d.m >= OPEN_M) & (d.m <= CLOSE_M)].sort_values('m').drop_duplicates('m').reset_index(drop=True)
    return d if len(d) else None


def session_vwap(d):
    """Cumulative session VWAP since 09:30: typical price (H+L+C)/3 weighted by bar volume."""
    tp = (d.h + d.l + d.c) / 3.0
    cv = d.v.cumsum()
    cpv = (tp * d.v).cumsum()
    return np.where(cv > 0, cpv / cv, d.c)


def one_db_pass(sig, spy, limit=None):
    """Part (a), the ONE DB pass. One query per (day,symbol) group. `walk_start_m` for a group is
    the EARLIEST entry_m among signals sharing that (day,symbol) — empirically (2026-09-22
    features.csv) every (day,symbol) pair is unique, so this reduces to walk_start_m == entry_m,
    but the code handles the general re-break case correctly (minutes_since_entry anchored to the
    group's first signal; a later signal's own minutes-since-ITS-entry is `m - that signal's
    entry_m`, trivial arithmetic downstream). Returns paths (DataFrame)."""
    groups = list(sig.groupby(['day', 'symbol'], sort=False))
    if limit:
        groups = groups[:limit]
    ng = len(groups)
    con = sqlite3.connect(BARS_DB)
    t0 = time.time()
    frames, missing = [], []
    for i, ((day, symbol), g) in enumerate(groups, 1):
        d = fetch_day_bars(con, symbol, day)
        if d is None or len(d) < 5:
            missing.append((day, symbol))
            continue
        d['vwap'] = session_vwap(d)
        walk_start_m = int(g.entry_m.min())
        w = d[(d.m >= walk_start_m) & (d.m <= EOD_M)].copy()
        if w.empty:
            missing.append((day, symbol))
            continue
        w['day'] = day
        w['symbol'] = symbol
        w['minutes_since_entry'] = w.m - walk_start_m
        w['spy_close'] = [spy.get(f'{day} {hh}', np.nan) for hh in w.hhmm]
        frames.append(w[['day', 'symbol', 'm', 'hhmm', 'o', 'h', 'l', 'c', 'vwap', 'spy_close',
                          'minutes_since_entry']])
        if i % 1000 == 0 or i == ng:
            npaths = sum(len(x) for x in frames)
            log(f'[db-pass] {i}/{ng} groups | {time.time()-t0:.0f}s elapsed | '
                f'{npaths} path rows so far | {len(missing)} missing')
    con.close()
    paths = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=['day', 'symbol', 'm', 'hhmm', 'o', 'h', 'l', 'c', 'vwap', 'spy_close', 'minutes_since_entry'])
    log(f'[db-pass] DONE {ng} groups in {time.time()-t0:.0f}s, {len(paths)} path rows, '
        f'{len(missing)} groups missing/short bar coverage')
    if missing:
        log(f'[db-pass] WARNING: {len(missing)} (day,symbol) groups had no/short bar coverage — '
            f'dropped from paths.parquet. First 5: {missing[:5]}')
    return paths


def b0_fill(entry, stop, target, path_after):
    """One signal's B0 exit, exactly per PREREG.md's fill-physics sentence (line 14-16), walking
    bars strictly AFTER the entry bar:
      - a bar whose OPEN already gapped through the stop fills at that open (gap-through);
      - otherwise a bar whose low <= stop fills at the stop price;
      - a bar that touches BOTH stop and target (low<=stop AND high>=target) counts as a stop
        (checked first below — conservative, per PREREG);
      - a bar whose high >= target (no overshoot buffer — literal PREREG text, NOT
        build_candidates.py's 1.002x buffer) fills at the target;
      - reaching the 15:55 (EOD_M) bar with neither triggered exits at THAT bar's open.
    `path_after` must already be filtered to m > entry_m, m <= EOD_M, sorted by m.
    Returns (exit_m, exit_price, why)."""
    for row in path_after.itertuples():
        if row.m >= EOD_M:
            return int(row.m), float(row.o), 'eod'
        if row.l <= stop:
            px = row.o if row.o <= stop else stop
            return int(row.m), float(px), 'stop'
        if row.h >= target:
            return int(row.m), float(target), 'target'
    last = path_after.iloc[-1]                                    # defensive fallback, should not
    return int(last.m), float(last.c), 'eod'                      # trigger: EOD_M row is in range


def simulate_b0(sig, paths):
    """Part (b): B0 fill physics (b0_fill) + PREREG cost — literally 'half-spread both legs + 2bp
    per side' — applied to the cached paths. cost_R = 2 x (0.5 x spread_mean)/R  [half-spread on
    the entry leg AND the exit leg]  +  2bp x (entry+exit_price)/R  [2bp slippage per side]. This
    is intentionally NOT cells.py's why-weighted ratio cost (0.875/0.412/0.0) — that is a
    different, portfolio-calibrated cost model used only as the reproduction-gate REFERENCE, never
    as B0's own cost here."""
    idx = paths.sort_values(['day', 'symbol', 'm']).set_index(['day', 'symbol']).sort_index()
    rows, n_no_path = [], 0
    for r in sig.itertuples():
        key = (r.day, r.symbol)
        if key not in idx.index:
            n_no_path += 1
            continue
        g = idx.loc[[key]]                                        # always a DataFrame (list-key)
        after = g[g.m > r.entry_m]
        if after.empty:
            n_no_path += 1
            continue
        exit_m, exit_px, why = b0_fill(r.entry, r.stop, r.target, after)
        raw_rr = (exit_px - r.entry) / r.R
        if pd.notna(r.spread_mean):
            half_R = (0.5 * r.spread_mean) / r.R
            slip_R = SLIP_BP * (r.entry + exit_px) / r.R
            cost_R = 2 * half_R + slip_R
            net_R = raw_rr - cost_R
        else:
            cost_R = net_R = np.nan
        rows.append(dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, exit_m=exit_m,
                          entry=r.entry, stop=r.stop, target=r.target, exit_price=exit_px, why=why,
                          R=r.R, raw_rr=raw_rr, cost_R=cost_R, net_R=net_R,
                          split=r.split, half=r.half, wk=r.wk))
    if n_no_path:
        log(f'[b0] WARNING: {n_no_path} signals had no cached path rows after entry_m — dropped')
    t = pd.DataFrame(rows)
    log(f'[b0] {len(t)} trades simulated | TRAIN n={(t.split=="TRAIN").sum()} '
        f'VAL n={(t.split=="VAL").sum()} TEST n={(t.split=="TEST").sum()} (TEST cached, never scored)')
    return t


def reproduction_gate(b0):
    """Part (c). 'The causal-filter study's baseline' has two readings; both are printed, the GATE
    is against (A):
      A) cells.py helpers, RAW population: features.csv's own rr/why -> cells.py's net_meas
         formula, obtainable==True, NO run_book slot/concurrency restriction. This matches
         PREREG.md's own B0 population definition (all signals, no book).
      B) cells.csv's PUBLISHED 'BASELINE' row, meas arm (cited only): that number additionally
         passes every kept row through trading.hod_break.run_book(rows,12,4) — a PORTFOLIO-level
         daily-cap/concurrency rule, not part of B0's per-trade exit definition — so it is expected
         to differ and is not the reproduction target.
    If |deltaA| > 0.01R, decomposes it into a fill-convention component (my raw_rr vs features.csv's
    rr, same rows) and a cost-convention component (my flat cost vs cells.py's why-weighted ratio
    cost), since features.csv's rr/why were built by build_candidates.py's exits(), which differs
    from PREREG's literal fill rule in two ways: target trigger high>=target*1.002 (not plain
    high>=target) and a 0.1% stop-fill slip (PREREG's B0 has neither — its cost is spread+2bp only)."""
    sys.path.insert(0, CF_DIR)
    import cells as C
    c = C.load()
    ref_pop = c[(c.split == 'TRAIN') & (c.obtainable == True)]     # noqa: E712 — the NO-FILL rail
    refA = float(ref_pop.net_meas.mean())
    b0tr = b0[(b0.split == 'TRAIN') & b0.net_R.notna()]
    mine = float(b0tr.net_R.mean())
    deltaA = mine - refA
    log(f'[gate] MY B0 TRAIN mean net R  = {mine:.4f}  (n={len(b0tr)})')
    log(f'[gate] Reference A (cells.py helpers, raw population, meas cost, NO run_book) '
        f'TRAIN mean net R = {refA:.4f}  (n={len(ref_pop)})')
    log(f'[gate] delta A (THE GATE) = {deltaA:+.4f} R')
    refB = deltaB = None
    cellscsv = f'{CF_DIR}/cells.csv'
    if os.path.exists(cellscsv):
        cc = pd.read_csv(cellscsv)
        row = cc[cc.cell.str.contains('BASELINE', na=False) & (cc.arm == 'meas')]
        if len(row):
            refB = float(row.iloc[0].TRAIN_meanR)
            deltaB = mine - refB
            log(f'[gate] Reference B (published cells.csv BASELINE/meas, INCLUDES run_book 12/4 '
                f'slot filter — cited only, NOT the gate) TRAIN meanR = {refB:.4f} '
                f'(n={int(row.iloc[0].TRAIN_n)})')
            log(f'[gate] delta B = {deltaB:+.4f} R')
    investigation = None
    if abs(deltaA) > 0.01:
        log('[gate] |deltaA| > 0.01 R — decomposing into fill-convention vs cost-convention parts')
        m = b0tr.merge(c[c.split == 'TRAIN'][['day', 'symbol', 'rr', 'net_meas', 'why']],
                        on=['day', 'symbol'], how='inner', suffixes=('', '_study'))
        fill_delta = float((m.raw_rr - m.rr).mean())
        study_cost = m.rr - m.net_meas
        cost_delta = float((m.cost_R - study_cost).mean())
        agree_why = float((m.why == m.why_study).mean() * 100)
        log(f'[gate] on {len(m)} matched (day,symbol) rows: '
            f'mean(my_raw_rr - study_rr) = {fill_delta:+.4f} R (fill-convention gap: PREREG '
            f'high>=target vs build_candidates high>=target*1.002 + 0.1% stop slip); '
            f'mean(my_cost_R - study_cost_R) = {cost_delta:+.4f} R (cost-convention gap: flat '
            f'half-spread+2bp both legs vs cells.py why-weighted ratio 0.875/0.412/0.0); '
            f'exit-reason agreement = {agree_why:.1f}%')
        investigation = dict(fill_delta=fill_delta, cost_delta=cost_delta, agree_why_pct=agree_why,
                              n_matched=len(m))
    return dict(mine=mine, refA=refA, deltaA=deltaA, refB=refB, deltaB=deltaB,
                investigation=investigation)


def _excursion(sig, paths, b0):
    """Per-signal MFE/MAE/minutes-to-MFE/give-back, from entry (k=0) to min(entry_m+390, 955)."""
    idx = paths.sort_values(['day', 'symbol', 'm']).set_index(['day', 'symbol']).sort_index()
    b0i = b0.set_index(['day', 'symbol'])
    out = []
    for r in sig.itertuples():
        key = (r.day, r.symbol)
        if key not in idx.index or key not in b0i.index:
            continue
        g = idx.loc[[key]]
        w = g[(g.m >= r.entry_m) & (g.m <= min(r.entry_m + 390, EOD_M))]
        if w.empty:
            continue
        rr_series = (w.c.values - r.entry) / r.R
        k = (w.m.values - r.entry_m)
        mfe_i = int(np.argmax(rr_series))
        mfe, mae = float(rr_series[mfe_i]), float(rr_series.min())
        end_r = float(b0i.loc[key, 'raw_rr']) if isinstance(b0i.loc[key], pd.Series) else float(b0i.loc[key].iloc[0].raw_rr)
        gave_back = (mfe >= 1.0) and (end_r <= 0.0)
        hb = pd.cut([r.entry_m], HB_EDGES, labels=HB_LAB)[0]
        out.append(dict(day=r.day, symbol=r.symbol, split=r.split, hb=hb, mfe=mfe, mae=mae,
                         minutes_to_mfe=int(k[mfe_i]), end_r=end_r, gave_back=gave_back,
                         r_given_back=(mfe - end_r) if gave_back else np.nan))
    return pd.DataFrame(out)


def drift_profile(sig, paths, b0):
    """Part (d): the PREREG descriptive exhibit (no cell — informs pass 2 only). TRAIN/VAL only
    (TEST is sealed, never reported). Writes DRIFT.md."""
    ex = _excursion(sig[sig.split.isin(['TRAIN', 'VAL'])], paths, b0)
    log(f'[drift] excursion table built on {len(ex)} signals')
    lines = ['# DRIFT.md — HOD-break Exit Lab descriptive exhibit (no cell; PREREG.md "Descriptive exhibit")',
              '',
              f'Generated by walker.py. TRAIN/VAL only ({len(ex)} signals with cached path + B0 '
              f'outcome); TEST is sealed and excluded from every table below.',
              '',
              '## By split',
              '',
              '| split | n | mean MFE (R) | median MFE (R) | mean MAE (R) | median MAE (R) | '
              'median min-to-MFE | give-back share | mean R given back |',
              '|---|---|---|---|---|---|---|---|---|']
    for sp, g in ex.groupby('split'):
        gb = g[g.gave_back]
        lines.append(f"| {sp} | {len(g)} | {g.mfe.mean():.3f} | {g.mfe.median():.3f} | "
                      f"{g.mae.mean():.3f} | {g.mae.median():.3f} | {g.minutes_to_mfe.median():.0f} | "
                      f"{len(gb)/len(g)*100:.1f}% ({len(gb)}/{len(g)}) | "
                      f"{gb.r_given_back.mean():.3f} |")
    lines += ['', '## By time-of-day bucket of entry (TRAIN+VAL combined)', '',
               '| bucket | n | mean MFE (R) | median MFE (R) | mean MAE (R) | median min-to-MFE | '
               'give-back share | mean R given back |', '|---|---|---|---|---|---|---|---|']
    for hb in HB_LAB:
        g = ex[ex.hb == hb]
        if g.empty:
            continue
        gb = g[g.gave_back]
        lines.append(f"| {hb} | {len(g)} | {g.mfe.mean():.3f} | {g.mfe.median():.3f} | "
                      f"{g.mae.mean():.3f} | {g.minutes_to_mfe.median():.0f} | "
                      f"{len(gb)/len(g)*100:.1f}% ({len(gb)}/{len(g)}) | "
                      f"{gb.r_given_back.mean():.3f} |")
    lines += ['', '## By time-of-day bucket x split', '',
               '| bucket | split | n | mean MFE (R) | mean MAE (R) | give-back share | '
               'mean R given back |', '|---|---|---|---|---|---|---|']
    for (hb, sp), g in ex.groupby(['hb', 'split'], observed=True):
        if g.empty:
            continue
        gb = g[g.gave_back]
        lines.append(f"| {hb} | {sp} | {len(g)} | {g.mfe.mean():.3f} | {g.mae.mean():.3f} | "
                      f"{len(gb)/len(g)*100:.1f}% | "
                      f"{(gb.r_given_back.mean() if len(gb) else float('nan')):.3f} |")
    lines += ['', '## Excursion by minute-since-entry (mean/median R, TRAIN+VAL, 15-min marks)', '',
               '| minute | n with data | mean R | median R |', '|---|---|---|---|']
    idx = paths.sort_values(['day', 'symbol', 'm']).set_index(['day', 'symbol']).sort_index()
    sub = sig[sig.split.isin(['TRAIN', 'VAL'])]
    for k in range(0, 391, 15):
        vals = []
        for r in sub.itertuples():
            key = (r.day, r.symbol)
            if key not in idx.index:
                continue
            g = idx.loc[[key]]
            row = g[g.m == r.entry_m + k]
            if len(row):
                vals.append((float(row.c.iloc[0]) - r.entry) / r.R)
        if vals:
            v = np.array(vals)
            lines.append(f"| {k} | {len(v)} | {v.mean():.3f} | {np.median(v):.3f} |")
    with open(f'{LAB}/DRIFT.md', 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    log(f'[drift] DRIFT.md written ({len(lines)} lines)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=int, default=0, help='run only the first N (day,symbol) groups, skip the blackout wait, write *_smoke files')
    args = ap.parse_args()

    schema_check()
    sig = load_signals()
    spy = load_spy()

    if args.smoke:
        log(f'[smoke] SMOKE TEST — first {args.smoke} groups only, no blackout wait')
        paths = one_db_pass(sig, spy, limit=args.smoke)
        paths.to_parquet(f'{LAB}/paths_smoke.parquet', index=False)
        keys = set(zip(paths.day, paths.symbol))
        sig_s = sig[sig.apply(lambda r: (r.day, r.symbol) in keys, axis=1)]
        sig_s.to_parquet(f'{LAB}/signals_smoke.parquet', index=False)
        b0 = simulate_b0(sig_s, paths)
        b0.to_csv(f'{LAB}/b0_trades_smoke.csv', index=False)
        log('[smoke] DONE — inspect *_smoke files, then run without --smoke for the full pass')
        return

    wait_for_db_window()
    paths = one_db_pass(sig, spy)
    paths.to_parquet(f'{LAB}/paths.parquet', index=False)
    sig.to_parquet(f'{LAB}/signals.parquet', index=False)
    log(f'[write] paths.parquet ({len(paths)} rows) + signals.parquet ({len(sig)} rows) written')

    b0 = simulate_b0(sig, paths)
    b0.to_csv(f'{LAB}/b0_trades.csv', index=False)
    log(f'[write] b0_trades.csv ({len(b0)} rows) written')

    gate = reproduction_gate(b0)
    log(f'[gate] SUMMARY mine={gate["mine"]:.4f} refA={gate["refA"]:.4f} deltaA={gate["deltaA"]:+.4f} '
        f'refB={gate["refB"]}')

    drift_profile(sig, paths, b0)
    log('ALL DONE — paths.parquet, signals.parquet, b0_trades.csv, DRIFT.md all written')


if __name__ == '__main__':
    main()
