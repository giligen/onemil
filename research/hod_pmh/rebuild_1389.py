#!/usr/bin/env python3
"""Independent rebuild of PMH cell 1,389 (P1: target +2R, stop, flat 15:55).

Written from the prose spec in research/hod_exit_lab/PREREG_PMH.md and the data
conventions in research/hod_exit_lab/FACTS.md ONLY (with the correction that the
bars_sip.db `t` column and the cache.db `timestamp` column are both full ISO-8601
UTC timestamps, not trimmed HH:MM Eastern strings).

Spec as implemented:
  * Universe      : research/hod_exit_lab/pm_candidates.csv (HOD-break symbol-days).
  * PMH level     : max(high) of pre-market 1-min bars 04:00-09:29 ET from
                    data/cache.db::intraday_bars_1min.
  * Availability  : >= 20 pre-market bars with prints, and pre-market volume >= 50,000.
  * Break         : first regular-session bar 09:31-11:30 ET (09:30 excluded) whose
                    CLOSE > PMH, using research/bf_zero/bars_sip.db::bars.
  * Entry         : OPEN of the next 1-minute bar.
  * Stop          : min low of the regular-session bars in [break-15, break-1] ET,
                    floored so that risk >= 1.5% of the entry price.
  * Exit (P1)     : +2R target, stop, else flat at the 15:55 OPEN.  Stop is checked
                    before target inside the same bar (conservative).
  * One trade per symbol-day.
  * Cost          : measured NBBO spread at the HOD signal minute of the same
                    name-day (research/bf_zero/causal_filter/nbbo.csv::spread_mean),
                    charged per FACTS/cells.py:
                        half  = 0.5 * sp_pct / r_pct
                        net   = rr - half - half * RATIO[why]
                    RATIO = {'stop': 0.875, 'eod': 0.412, 'target': 0.0}
                    A second arm `net_sym` charges a symmetric half-spread on both
                    sides (ratio == 1.0) as a sensitivity line.
  * Splits        : TRAIN day < 2026-01-01, VAL 2026-01-01..2026-05-31, TEST after
                    (TEST is scored but never reported).
"""

from __future__ import annotations

import csv
import os
import sqlite3
import sys
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

ROOT = '/home/ec2-user/onemil'
CACHE_DB = f'{ROOT}/data/cache.db'
SIP_DB = f'{ROOT}/research/bf_zero/bars_sip.db'
CANDIDATES = f'{ROOT}/research/hod_exit_lab/pm_candidates.csv'
NBBO = f'{ROOT}/research/bf_zero/causal_filter/nbbo.csv'
OUT = f'{ROOT}/research/hod_pmh/rebuild_1389_trades.csv'

ET = ZoneInfo('America/New_York')

PM_START, PM_END = 4 * 60, 9 * 60 + 29          # 04:00 .. 09:29 ET inclusive
BREAK_FIRST, BREAK_LAST = 9 * 60 + 31, 11 * 60 + 30   # 09:31 .. 11:30 ET inclusive
SESSION_OPEN = 9 * 60 + 30
FLAT_M = 15 * 60 + 55                            # 15:55 ET
MIN_PM_BARS = 20
MIN_PM_VOL = 50_000
STOP_FLOOR_PCT = 0.015
TARGET_R = 2.0
RATIO = {'stop': 0.875, 'eod': 0.412, 'target': 0.0}


def et_offset_minutes(day: str) -> int:
    """Return the UTC->ET offset in minutes (negative) for a trading day."""
    y, m, d = (int(x) for x in day.split('-'))
    # 12:00 UTC on that date is always inside the same ET calendar day.
    dt = datetime(y, m, d, 12, 0, tzinfo=timezone.utc).astimezone(ET)
    return int(dt.utcoffset().total_seconds() // 60)


def et_minute(ts: str, off: int) -> int:
    """Minutes since ET midnight for an ISO-8601 UTC timestamp string."""
    hh = int(ts[11:13]); mm = int(ts[14:16])
    return (hh * 60 + mm + off) % 1440


def split_of(day: str) -> str:
    if day < '2026-01-01':
        return 'TRAIN'
    if day < '2026-06-01':
        return 'VAL'
    return 'TEST'


def half_of(day: str) -> str:
    if day < '2025-07-01':
        return 'H1'
    if day < '2026-01-01':
        return 'H2'
    return ''


def load_nbbo() -> dict:
    """(day, symbol) -> mean NBBO spread in dollars at the HOD signal minute."""
    out = {}
    with open(NBBO, newline='') as fh:
        for row in csv.DictReader(fh):
            sp = row.get('spread_mean') or ''
            if not sp:
                continue
            try:
                out[(row['day'], row['symbol'])] = float(sp)
            except ValueError:
                continue
    return out


def load_candidates() -> list:
    with open(CANDIDATES, newline='') as fh:
        return [(r['symbol'], r['bar_date']) for r in csv.DictReader(fh)]


def main() -> None:
    cands = load_candidates()
    nbbo = load_nbbo()
    print(f'candidates={len(cands)} nbbo_rows={len(nbbo)}', flush=True)

    cc = sqlite3.connect(f'file:{CACHE_DB}?mode=ro', uri=True)
    sc = sqlite3.connect(f'file:{SIP_DB}?mode=ro', uri=True)
    cc.execute('PRAGMA query_only=1')
    sc.execute('PRAGMA query_only=1')
    cur_c, cur_s = cc.cursor(), sc.cursor()

    off_cache: dict = {}
    trades = []
    n_no_pm = n_thin_pm = n_low_vol = n_no_rth = n_no_break = n_no_entry = n_bad_stop = 0

    for i, (sym, day) in enumerate(cands):
        if i % 1000 == 0:
            print(f'  ..{i}/{len(cands)} trades={len(trades)}', flush=True)
        off = off_cache.get(day)
        if off is None:
            off = off_cache[day] = et_offset_minutes(day)

        # ---- pre-market level -------------------------------------------------
        cur_c.execute(
            'SELECT timestamp, high, volume FROM intraday_bars_1min '
            'WHERE symbol=? AND bar_date=?', (sym, day))
        pm_hi, pm_n, pm_v = None, 0, 0.0
        for ts, h, v in cur_c.fetchall():
            m = et_minute(ts, off)
            if PM_START <= m <= PM_END:
                pm_n += 1
                pm_v += float(v or 0.0)
                if pm_hi is None or h > pm_hi:
                    pm_hi = h
        if pm_hi is None:
            n_no_pm += 1
            continue
        if pm_n < MIN_PM_BARS:
            n_thin_pm += 1
            continue
        if pm_v < MIN_PM_VOL:
            n_low_vol += 1
            continue

        # ---- regular-session path --------------------------------------------
        cur_s.execute(
            'SELECT t, o, h, l, c FROM bars WHERE symbol=? AND day=?', (sym, day))
        bars = {}
        for t, o, h, l, c in cur_s.fetchall():
            m = et_minute(t, off)
            if SESSION_OPEN <= m <= 16 * 60:
                bars[m] = (o, h, l, c)
        if not bars:
            n_no_rth += 1
            continue

        mins = sorted(bars)
        brk = None
        for m in mins:
            if m < BREAK_FIRST:
                continue
            if m > BREAK_LAST:
                break
            if bars[m][3] > pm_hi:
                brk = m
                break
        if brk is None:
            n_no_break += 1
            continue

        nxt = next((m for m in mins if m > brk), None)
        if nxt is None or nxt > FLAT_M:
            n_no_entry += 1
            continue
        entry = bars[nxt][0]

        lows = [bars[m][2] for m in mins if brk - 15 <= m <= brk - 1]
        floor_stop = entry * (1.0 - STOP_FLOOR_PCT)
        stop = min(lows) if lows else floor_stop
        stop = min(stop, floor_stop)
        R = entry - stop
        if R <= 0:
            n_bad_stop += 1
            continue
        target = entry + TARGET_R * R

        why, exit_px, exit_m = None, None, None
        for m in mins:
            if m < nxt or m >= FLAT_M:
                continue
            o, h, l, c = bars[m]
            if l <= stop:
                why, exit_px, exit_m = 'stop', stop, m
                break
            if h >= target:
                why, exit_px, exit_m = 'target', target, m
                break
        if why is None:
            flat = next((m for m in mins if m >= FLAT_M), None)
            if flat is not None:
                why, exit_px, exit_m = 'eod', bars[flat][0], flat
            else:
                last = mins[-1]
                why, exit_px, exit_m = 'eod', bars[last][3], last

        rr = (exit_px - entry) / R
        r_pct = R / entry * 100.0
        sp = nbbo.get((day, sym))
        if sp is None:
            cost = cost_sym = ''
            net = net_sym = ''
        else:
            sp_pct = sp / entry * 100.0
            half = 0.5 * sp_pct / max(r_pct, 0.05)
            cost = half + half * RATIO[why]
            cost_sym = 2.0 * half
            net = rr - cost
            net_sym = rr - cost_sym

        trades.append(dict(
            day=day, symbol=sym, break_m=brk, entry_m=nxt, exit_m=exit_m,
            pmh=round(pm_hi, 4), pm_bars=pm_n, pm_vol=int(pm_v),
            entry=round(entry, 4), stop=round(stop, 4), R=round(R, 6),
            r_pct=round(r_pct, 4), target=round(target, 4),
            exit_price=round(exit_px, 4), why=why, rr=round(rr, 6),
            spread_mean=sp if sp is not None else '',
            cost_R=round(cost, 6) if cost != '' else '',
            net_R=round(net, 6) if net != '' else '',
            net_sym_R=round(net_sym, 6) if net_sym != '' else '',
            split=split_of(day), half=half_of(day)))

    cc.close(); sc.close()

    cols = list(trades[0].keys()) if trades else []
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(trades)

    print('--- rejections ---', flush=True)
    print(f'no_pm_bars={n_no_pm} thin_pm(<20)={n_thin_pm} pm_vol<50k={n_low_vol} '
          f'no_rth={n_no_rth} no_break={n_no_break} no_entry_bar={n_no_entry} '
          f'bad_stop={n_bad_stop}', flush=True)
    print(f'availability: {len(cands) - n_no_pm - n_thin_pm} / {len(cands)} '
          f'symbol-days have >= {MIN_PM_BARS} pre-market bars', flush=True)

    for sp_name in ('TRAIN', 'VAL', 'TEST'):
        sub = [t for t in trades if t['split'] == sp_name]
        if not sub:
            print(f'{sp_name}: 0 trades', flush=True)
            continue
        g = sum(t['rr'] for t in sub) / len(sub)
        nn = [t['net_R'] for t in sub if t['net_R'] != '']
        ns = [t['net_sym_R'] for t in sub if t['net_sym_R'] != '']
        print(f'{sp_name}: n={len(sub)} gross_mean_R={g:.4f} '
              f'net_mean_R={(sum(nn)/len(nn) if nn else float("nan")):.4f} '
              f'(n_cost={len(nn)}) net_sym_mean_R='
              f'{(sum(ns)/len(ns) if ns else float("nan")):.4f}', flush=True)
    print(f'wrote {OUT} ({len(trades)} trades)', flush=True)


if __name__ == '__main__':
    main()
