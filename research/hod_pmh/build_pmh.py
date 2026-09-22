#!/usr/bin/env python3
"""PMH (pre-market-high break) signal builder — research/hod_exit_lab/PREREG_PMH.md, cells 1,389-1,392.

ONE pass over bars_sip.db per (symbol,day): the same query is used both to search for the break
(09:31-11:30 ET) and, if a signal fires, to slice the entry->15:55 path — no second DB pass.

Deliverables (this directory):
  1. PMH level + pm_volume per symbol-day from data/cache.db::intraday_bars_1min (04:00-09:29 ET);
     availability rail (>=20 premarket bars with prints); availability share printed + in DRIFT.md.
  2. signals.parquet — day,symbol,entry_m,entry,stop,R,r_pct,half_spread,split (+ price,target,why
     helper cols). Break = first RTH 1-min CLOSE > PMH, 09:31-11:30 ET (09:30 excluded). Entry = next
     bar's OPEN. Stop = low of the trailing 15 min before the break bar, floored at 1.5% of price
     (interpretation: stop = min(trailing_low, entry*(1-0.015)) — the floor sets a MINIMUM risk
     distance, it never tightens a stop that is already >= 1.5% away).
  3. paths.parquet — per-minute path entry_m -> 955 (15:55 ET), cached in the same DB pass as (2).
  4. DRIFT.md — same tables as research/hod_exit_lab/DRIFT.md (walker.py's drift_profile), for this
     population, plus the OVERLAP line vs the HOD-break population (features.csv).

Bar convention (verified empirically, matches walker.py's documented deviation from FACTS.md):
bars_sip.db's `t` is ISO-8601 UTC and the table is NOT pre-trimmed to regular hours; we reuse
walker.fetch_day_bars() (imported, not re-implemented) which parses UTC -> America/New_York and
restricts to [09:30,15:59] ET. cache.db's intraday_bars_1min `timestamp` is likewise ISO-8601 UTC
(verified: AAOI 2026-01-15 spans 2026-01-15T09:00:00+00:00 .. T21:00:00+00:00 = 04:00-16:00 ET).

Fill physics for the DRIFT exhibit's B0 reference (target/stop/EOD walk that produces raw_rr, needed
only for the excursion table's give-back logic, exactly as walker.py's own DRIFT.md uses it) reuses
walker.b0_fill() verbatim with TARGET_R=2.0 (walker.TARGET_R) — no new fill logic is invented here.

Usage:
  python3 build_pmh.py --smoke 50   # first 50 (symbol,day) candidates only, fast correctness check
  python3 build_pmh.py              # full run (waits out the 13:25-20:05 UTC DB blackout window)
"""
import argparse
import csv
import os
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
LAB = f'{ROOT}/research/hod_exit_lab'
PMH = f'{ROOT}/research/hod_pmh'
CF_DIR = f'{ROOT}/research/bf_zero/causal_filter'
CACHE_DB = f'{ROOT}/data/cache.db'
CANDIDATES_CSV = f'{LAB}/pm_candidates.csv'
NBBO_CSV = f'{CF_DIR}/nbbo.csv'
FEATURES_CSV = f'{CF_DIR}/features.csv'

sys.path.insert(0, LAB)
import walker as W  # noqa: E402 — reuse fetch_day_bars, b0_fill, constants (see module docstring)

BREAK_START_M, BREAK_END_M = 571, 690     # 09:31-11:30 ET inclusive (09:30=570 excluded)
PM_START_M, PM_END_M = 240, 569           # 04:00-09:29 ET, minutes since midnight
PM_MIN_BARS = 20                          # availability rail
PM_MIN_VOL = 50_000                       # level-validity floor (PREREG)
STOP_FLOOR_PCT = 0.015                    # 1.5% of price, MINIMUM risk distance
TRAIL_MIN = 15                            # trailing window for the stop, minutes


def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)


def wait_for_backfill():
    """Block until pm_backfill.log contains a DONE line (the premarket backfill this study depends
    on). Polls every 30s; never assumes completion from elapsed time alone."""
    p = f'{LAB}/pm_backfill.log'
    while True:
        if os.path.exists(p):
            with open(p) as fh:
                if 'DONE' in fh.read():
                    log('[wait] pm_backfill.log contains DONE — proceeding')
                    return
        log('[wait] pm_backfill.log not DONE yet — sleeping 30s')
        time.sleep(30)


def load_candidates():
    """pm_candidates.csv: symbol,bar_date -> the 15,656 HOD-universe symbol-days (PREREG population
    for the pre-market level)."""
    rows = []
    with open(CANDIDATES_CSV) as fh:
        for r in csv.DictReader(fh):
            rows.append((r['symbol'], r['bar_date']))
    log(f'[candidates] {len(rows)} (symbol,day) pairs loaded from pm_candidates.csv')
    return rows


def pm_level(cache_con, symbol, day):
    """PMH level + pm_volume + bar count for one symbol-day from cache.db's intraday_bars_1min,
    window 04:00-09:29 ET (PM_START_M..PM_END_M). Returns dict or None if the table has zero rows
    for this (symbol,day) at all (no cache entry — distinct from '0 prints in window', which is a
    real availability-rail failure and IS returned with n_bars=0)."""
    cur = cache_con.execute(
        'SELECT timestamp,high,volume FROM intraday_bars_1min WHERE symbol=? AND bar_date=?',
        (symbol, day))
    rows = cur.fetchall()
    if not rows:
        return None
    d = pd.DataFrame(rows, columns=['t', 'h', 'v'])
    ts = pd.to_datetime(d.t, utc=True).dt.tz_convert('America/New_York')
    d['m'] = ts.dt.hour * 60 + ts.dt.minute
    w = d[(d.m >= PM_START_M) & (d.m <= PM_END_M)]
    n_bars = len(w)
    pm_vol = float(w.v.sum()) if n_bars else 0.0
    pmh = float(w.h.max()) if n_bars else np.nan
    return dict(symbol=symbol, day=day, n_pm_bars=n_bars, pm_volume=pm_vol, pmh=pmh,
                available=n_bars >= PM_MIN_BARS, has_level=pm_vol >= PM_MIN_VOL)


def find_signal(d, pmh):
    """Search d (walker.fetch_day_bars output, m in [570,959]) for the first bar in
    [BREAK_START_M,BREAK_END_M] whose CLOSE > pmh. Returns the break row (pd.Series) or None."""
    w = d[(d.m >= BREAK_START_M) & (d.m <= BREAK_END_M) & (d.c > pmh)]
    return w.iloc[0] if len(w) else None


def build_signal(d, brk, symbol, day):
    """Given the break bar, build one signal row per PREREG: entry = next bar's open; stop = trailing
    15-min low floored at 1.5% of price (min(trailing_low, entry*(1-0.015)) — see module docstring).
    Returns dict or None (no next bar / non-positive R -> dropped, caller counts it)."""
    nxt = d[d.m > brk.m]
    if nxt.empty:
        return None
    entry_m, entry = int(nxt.iloc[0].m), float(nxt.iloc[0].o)
    trail = d[(d.m >= brk.m - TRAIL_MIN) & (d.m < brk.m)]
    trailing_low = float(trail.l.min()) if len(trail) else float(brk.l)
    floor_stop = entry * (1 - STOP_FLOOR_PCT)
    stop = min(trailing_low, floor_stop)
    R = entry - stop
    if R <= 0:
        return None
    return dict(day=day, symbol=symbol, break_m=int(brk.m), entry_m=entry_m, entry=entry,
                stop=stop, R=R, r_pct=100.0 * R / entry, price=entry,
                target=entry + W.TARGET_R * R)


def load_nbbo():
    n = pd.read_csv(NBBO_CSV, dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
    n = n.drop_duplicates(['day', 'symbol'])[['day', 'symbol', 'spread_mean']]
    log(f'[nbbo] {len(n)} unique (day,symbol) rows loaded from nbbo.csv')
    return n


def split_of(day):
    if day < '2026-01-01':
        return 'TRAIN'
    if day < '2026-06-01':
        return 'VAL'
    return 'TEST'


def one_pass(cands, limit=None):
    """The ONE pass: per (symbol,day), (a) compute the PM level from cache.db, (b) if available+
    has_level, query bars_sip.db ONCE (walker.fetch_day_bars) to search for the break AND, on a hit,
    slice the same frame for the path -> paths list. Returns (level_rows, sig_rows, path_frames)."""
    if limit:
        cands = cands[:limit]
    n = len(cands)
    cache_con = sqlite3.connect(CACHE_DB)
    bars_con = sqlite3.connect(W.BARS_DB)
    level_rows, sig_rows, path_frames = [], [], []
    n_no_level, n_no_bars_sip, n_no_break, n_dropped_fill = 0, 0, 0, 0
    t0 = time.time()
    for i, (symbol, day) in enumerate(cands, 1):
        lv = pm_level(cache_con, symbol, day)
        if lv is None:
            n_no_level += 1
        else:
            level_rows.append(lv)
        if lv is None or not (lv['available'] and lv['has_level']):
            continue
        d = W.fetch_day_bars(bars_con, symbol, day)
        if d is None or len(d) < 5:
            n_no_bars_sip += 1
            continue
        brk = find_signal(d, lv['pmh'])
        if brk is None:
            n_no_break += 1
            continue
        sig = build_signal(d, brk, symbol, day)
        if sig is None:
            n_dropped_fill += 1
            continue
        sig_rows.append(sig)
        p = d[(d.m >= sig['entry_m']) & (d.m <= W.EOD_M)].copy()
        p['day'] = day
        p['symbol'] = symbol
        path_frames.append(p[['day', 'symbol', 'm', 'o', 'h', 'l', 'c']])
        if i % 1000 == 0 or i == n:
            log(f'[pass] {i}/{n} | {time.time()-t0:.0f}s | levels={len(level_rows)} '
                f'signals={len(sig_rows)} no_level_row={n_no_level} no_bars_sip={n_no_bars_sip} '
                f'no_break={n_no_break} dropped_fill={n_dropped_fill}')
    cache_con.close()
    bars_con.close()
    log(f'[pass] DONE {n} candidates in {time.time()-t0:.0f}s -> {len(sig_rows)} signals, '
        f'{len(path_frames)} path groups')
    return level_rows, sig_rows, path_frames


def report_availability(level_rows, n_candidates):
    lv = pd.DataFrame(level_rows)
    n_with_row = len(lv)
    avail_share = lv.available.mean() if n_with_row else 0.0
    level_share = lv.has_level.mean() if n_with_row else 0.0
    both_share = (lv.available & lv.has_level).mean() if n_with_row else 0.0
    log(f'[availability] {n_with_row}/{n_candidates} candidates had >=1 cache.db row '
        f'({n_with_row/n_candidates*100:.1f}%)')
    log(f'[availability] >=20 PM bars with prints: {avail_share*100:.1f}% of those '
        f'({(lv.available.sum() if n_with_row else 0)}/{n_with_row})')
    log(f'[availability] PM volume >= 50,000 sh (level valid): {level_share*100:.1f}%')
    log(f'[availability] BOTH (usable for signal search): {both_share*100:.1f}% '
        f'({(lv.available & lv.has_level).sum() if n_with_row else 0}/{n_candidates})')
    return dict(n_candidates=n_candidates, n_with_row=n_with_row,
                avail_share=float(avail_share), level_share=float(level_share),
                both_share=float(both_share))


def write_signals(sig_rows, nbbo):
    sig = pd.DataFrame(sig_rows)
    if sig.empty:
        log('[signals] WARNING: zero signals found — writing empty signals.parquet')
        sig = pd.DataFrame(columns=['day', 'symbol', 'break_m', 'entry_m', 'entry', 'stop', 'R',
                                     'r_pct', 'price', 'target', 'half_spread', 'spread_mean', 'split'])
        sig.to_parquet(f'{PMH}/signals.parquet', index=False)
        return sig
    sig = sig.merge(nbbo, on=['day', 'symbol'], how='left')
    sig['half_spread'] = 0.5 * sig.spread_mean
    sig['split'] = sig.day.map(split_of)
    cov = sig.spread_mean.notna().mean()
    log(f'[signals] {len(sig)} signals | splits={sig.split.value_counts().to_dict()} | '
        f'NBBO coverage (half-spread @ HOD signal minute, same name-day) = {cov*100:.1f}%')
    sig.to_parquet(f'{PMH}/signals.parquet', index=False)
    log(f'[write] signals.parquet ({len(sig)} rows)')
    return sig


def write_paths(path_frames):
    paths = pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame(
        columns=['day', 'symbol', 'm', 'o', 'h', 'l', 'c'])
    paths.to_parquet(f'{PMH}/paths.parquet', index=False)
    log(f'[write] paths.parquet ({len(paths)} rows, {len(path_frames)} groups)')
    return paths


def simulate_b0(sig, paths):
    """B0 walk (target=entry+2R / stop / 15:55) via walker.b0_fill, reused verbatim, to get raw_rr
    per signal for the DRIFT excursion table's give-back logic (no cost — DRIFT is descriptive)."""
    idx = paths.sort_values(['day', 'symbol', 'm']).set_index(['day', 'symbol']).sort_index()
    rows = []
    for r in sig.itertuples():
        key = (r.day, r.symbol)
        if key not in idx.index:
            continue
        g = idx.loc[[key]]
        after = g[g.m > r.entry_m]
        if after.empty:
            continue
        after = after.rename(columns={})  # already o,h,l,c,m — matches b0_fill's expected columns
        exit_m, exit_px, why = W.b0_fill(r.entry, r.stop, r.target, after)
        raw_rr = (exit_px - r.entry) / r.R
        rows.append(dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, R=r.R, raw_rr=raw_rr,
                          why=why, split=r.split))
    b0 = pd.DataFrame(rows)
    log(f'[b0] {len(b0)} B0 outcomes simulated for the DRIFT reference (target/stop/EOD, no cost)')
    return b0


def excursion_table(sig, paths, b0):
    """Mirrors walker._excursion exactly: per-signal MFE/MAE/minutes-to-MFE/give-back from entry to
    min(entry_m+390, EOD_M)."""
    idx = paths.sort_values(['day', 'symbol', 'm']).set_index(['day', 'symbol']).sort_index()
    b0i = b0.set_index(['day', 'symbol'])
    out = []
    for r in sig.itertuples():
        key = (r.day, r.symbol)
        if key not in idx.index or key not in b0i.index:
            continue
        g = idx.loc[[key]]
        w = g[(g.m >= r.entry_m) & (g.m <= min(r.entry_m + 390, W.EOD_M))]
        if w.empty:
            continue
        rr_series = (w.c.values - r.entry) / r.R
        k = (w.m.values - r.entry_m)
        mfe_i = int(np.argmax(rr_series))
        mfe, mae = float(rr_series[mfe_i]), float(rr_series.min())
        b0row = b0i.loc[key]
        end_r = float(b0row['raw_rr']) if isinstance(b0row, pd.Series) else float(b0row.iloc[0].raw_rr)
        gave_back = (mfe >= 1.0) and (end_r <= 0.0)
        hb = pd.cut([r.entry_m], W.HB_EDGES, labels=W.HB_LAB)[0]
        out.append(dict(day=r.day, symbol=r.symbol, split=r.split, hb=hb, mfe=mfe, mae=mae,
                         minutes_to_mfe=int(k[mfe_i]), end_r=end_r, gave_back=gave_back,
                         r_given_back=(mfe - end_r) if gave_back else np.nan))
    return pd.DataFrame(out)


def overlap_with_hod():
    """Share of PMH breaks (TRAIN+VAL) that are also HOD-break signals: same (day,symbol),
    features.csv entry_m within 5 minutes of the PMH signal's entry_m."""
    f = pd.read_csv(FEATURES_CSV, dtype={'symbol': str, 'day': str},
                     usecols=['day', 'symbol', 'entry_m'], keep_default_na=False, na_values=[''])
    fg = f.groupby(['day', 'symbol']).entry_m.apply(list).to_dict()
    sig = pd.read_parquet(f'{PMH}/signals.parquet')
    sig = sig[sig.split.isin(['TRAIN', 'VAL'])]
    n = len(sig)
    if n == 0:
        return 0, 0, 0.0
    matched = 0
    for r in sig.itertuples():
        ems = fg.get((r.day, r.symbol))
        if ems and any(abs(e - r.entry_m) <= 5 for e in ems):
            matched += 1
    share = matched / n
    log(f'[overlap] {matched}/{n} PMH breaks (TRAIN+VAL) also fire as HOD-break signals '
        f'(same day+symbol, entry_m within 5 min) = {share*100:.1f}%')
    return matched, n, share


def write_drift(ex, avail, overlap):
    matched, n_overlap, share = overlap
    lines = ['# DRIFT.md — PMH-break descriptive exhibit (PREREG_PMH.md "First deliverable, before '
             'any cell is scored")',
             '',
             f'Generated by build_pmh.py. TRAIN/VAL only ({len(ex)} signals with cached path + B0 '
             f'outcome); TEST is sealed and excluded from every table below.',
             '',
             '## Availability (pm_candidates.csv, all 15,656 HOD-universe symbol-days)',
             '',
             f"- Candidates with >=1 cache.db premarket row: {avail['n_with_row']}/{avail['n_candidates']}",
             f"- Availability rail (>=20 PM bars with prints): {avail['avail_share']*100:.1f}% of those",
             f"- PM volume >= 50,000 sh (level valid): {avail['level_share']*100:.1f}%",
             f"- Both (usable for signal search): {avail['both_share']*100:.1f}% of all candidates",
             '',
             '## By split', '',
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
    for hb in W.HB_LAB:
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
    lines += ['', '## OVERLAP with the HOD-break population (features.csv)', '',
              f"Share of PMH breaks (TRAIN+VAL) that are also HOD-break signals (same day+symbol, "
              f"entry_m within 5 minutes): **{share*100:.1f}%** ({matched}/{n_overlap}). "
              f"{'>80% — same population renamed; pass closed at this exhibit per PREREG.' if share > 0.80 else '<=80% — a distinct population per PREREG.'}",
              '']
    with open(f'{PMH}/DRIFT.md', 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    log(f'[drift] DRIFT.md written ({len(lines)} lines)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=int, default=0)
    args = ap.parse_args()

    cands = load_candidates()
    nbbo = load_nbbo()

    if not args.smoke:
        wait_for_backfill()
        W.wait_for_db_window()

    level_rows, sig_rows, path_frames = one_pass(cands, limit=args.smoke or None)
    avail = report_availability(level_rows, len(cands) if not args.smoke else min(args.smoke, len(cands)))

    sig = write_signals(sig_rows, nbbo)
    paths = write_paths(path_frames)

    if sig.empty or paths.empty:
        log('[drift] SKIPPED — zero signals/paths (smoke sample too small, or genuinely zero breaks)')
        return

    b0 = simulate_b0(sig, paths)
    ex = excursion_table(sig[sig.split.isin(['TRAIN', 'VAL'])], paths, b0)
    log(f'[drift] excursion table built on {len(ex)} signals')
    overlap = overlap_with_hod() if not args.smoke else (0, 0, 0.0)
    write_drift(ex, avail, overlap)
    log('ALL DONE — signals.parquet, paths.parquet, DRIFT.md written')


if __name__ == '__main__':
    main()
