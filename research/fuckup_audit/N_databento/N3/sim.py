#!/usr/bin/env python3
"""Stage N3 step 3 — the published stocks-in-play ORB through OUR simulator and OUR cost contract.

The rule (Zarattini-Barbon-Aziz 2024, SSRN 4729284, §2.1/§4), unchanged:
  direction  = colour of the 09:30-09:35 candle (up -> long, down -> short, doji -> no order)
  entry      = stop order at the 5-min high (long) / low (short), placed 09:35, live all session
  risk unit  = R = 0.10 x ATR14  (their whole stop)
  exits      = the stop, or the 15:59 close.  No target.
  sizing     = equal dollar risk per trade -> P&L is reported in R.

Two fill conventions:
  paper  : the order fills AT the level (their assumption); if the bar gapped through it, at that
           bar's open -- a price the market actually offered (PLAN 1b obtainability).
  ours   : the level is a SIGNAL; the fill is the NEXT bar's open (the engine's reaction fill).

Five cost arms (PLAN 3 / Stage A0 contract):
  zero            no costs
  paper_comm      $0.0035/share/leg, their model
  ours_band       spread from lit_review_2026/cost_curve.csv (price band x hour band) of OUR
                  2025-26 mover population; entry 0.25 x half (next-open) or 1.00 x half (resting
                  fill executed on arrival); stop 0.875 x half; eod 0.412 x half
  ours_10bps      the same coefficients on a flat 10 bps quoted spread (a liquid-name cost)
  ours_40bps      the same coefficients on 40 bps (our own measured anchor on $5+ movers)

Writes N3/trades.csv, N3/tables.md.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')

N3 = 'research/fuckup_audit/N_databento/N3'
DB = f'{N3}/tape.db'
TOP20 = f'{N3}/top20.csv'
COST_CURVE = 'research/lit_review_2026/cost_curve.csv'
COMMISSION_PER_SHARE = 0.0035
PBANDS = [(5, 10, '$5-10'), (10, 20, '$10-20'), (20, 50, '$20-50'),
          (50, 200, '$50-200'), (200, 1e9, '$200+')]
HBANDS = [(575, '09:30-09:35'), (600, '09:35-10:00'), (660, '10:00-11:00'),
          (780, '11:00-13:00'), (10 ** 9, '13:00+')]


def price_band(p):
    for lo, hi, lab in PBANDS:
        if lo <= p < hi:
            return lab
    return '$5-10'


def hour_band(m):
    for hi, lab in HBANDS:
        if m < hi:
            return lab
    return '13:00+'


def load_spread_table():
    """median quoted spread as a FRACTION of price, by (price band, hour band)."""
    c = pd.read_csv(COST_CURVE, keep_default_na=False, na_values=[''])
    c = c[c.spread.notna() & (c.price > 0)]
    c['frac'] = c.spread.astype(float) / c.price.astype(float)
    t = c.groupby(['pb', 'hb']).frac.median().to_dict()
    overall = float(c.frac.median())
    return t, overall


def simulate_day(bars, picks, out, atr_scale=1.0):
    """bars: DataFrame for one session (symbol, m, o, h, l, c, v).  picks: that day's top-20."""
    by_sym = {s: g.sort_values('m') for s, g in bars.groupby('symbol', sort=False)}
    for p in picks.itertuples():
        row = dict(day=p.day, symbol=p.symbol, rv=float(p.rv), side=int(p.side),
                   atr14=float(p.atr14), o5=float(p.o5), h5=float(p.h5), l5=float(p.l5),
                   c5=float(p.c5))
        if p.side == 0 or not np.isfinite(p.atr14) or p.atr14 <= 0:
            out.append({**row, 'status': 'no_order'})
            continue
        g = by_sym.get(p.symbol)
        if g is None or len(g) < 2:
            out.append({**row, 'status': 'no_tape'})
            continue
        g = g[g.m >= 575]
        if len(g) < 2:
            out.append({**row, 'status': 'no_tape'})
            continue
        side = int(p.side)
        level = float(p.h5) if side > 0 else float(p.l5)
        R = 0.10 * float(p.atr14) * atr_scale
        m = g.m.to_numpy()
        o, h, l, c = (g.o.to_numpy(), g.h.to_numpy(), g.l.to_numpy(), g.c.to_numpy())
        trig = np.flatnonzero(h >= level) if side > 0 else np.flatnonzero(l <= level)
        if not len(trig):
            out.append({**row, 'status': 'no_trigger'})
            continue
        i = int(trig[0])
        # paper fill: at the level, or at the open when the bar gapped through it
        e_paper = max(level, o[i]) if side > 0 else min(level, o[i])
        # our fill: the level is the signal, the fill is the next bar's open
        e_ours = float(o[i + 1]) if i + 1 < len(m) else np.nan
        for tag, entry, first in (('paper', float(e_paper), i), ('ours', e_ours, i + 1)):
            key = f'{tag}_'
            if not np.isfinite(entry) or first >= len(m):
                row[key + 'status'] = 'no_fill'
                continue
            stop_px = entry - side * R
            why, exit_px, exit_m = 'eod', float(c[-1]), int(m[-1])
            for j in range(first, len(m)):
                hit = (l[j] <= stop_px) if side > 0 else (h[j] >= stop_px)
                if hit:
                    # gap through the stop -> the open, never better than the stop level
                    exit_px = min(stop_px, o[j]) if side > 0 else max(stop_px, o[j])
                    why, exit_m = 'stop', int(m[j])
                    break
            row[key + 'status'] = 'filled'
            row[key + 'entry'] = entry
            row[key + 'entry_m'] = int(m[first])
            row[key + 'exit'] = float(exit_px)
            row[key + 'exit_m'] = exit_m
            row[key + 'why'] = why
            row[key + 'rr'] = side * (exit_px - entry) / R
            row[key + 'r_dollars'] = R
        row['status'] = 'ok'
        out.append(row)


def add_costs(t, spread_tbl, overall):
    for tag, entry_coef in (('paper', 1.00), ('ours', 0.25)):
        e = t[f'{tag}_entry']
        R = t[f'{tag}_r_dollars']
        why = t[f'{tag}_why']
        exit_coef = np.where(why == 'stop', 0.875, 0.412)
        pb = e.map(lambda x: price_band(x) if np.isfinite(x) else '$5-10')
        hb = t[f'{tag}_entry_m'].map(lambda x: hour_band(x) if np.isfinite(x) else '09:35-10:00')
        frac = np.array([spread_tbl.get((a, b), overall) for a, b in zip(pb, hb)])
        half_band = 0.5 * frac * e / R
        t[f'{tag}_half_band'] = half_band
        t[f'{tag}_net_band'] = t[f'{tag}_rr'] - (entry_coef + exit_coef) * half_band
        for bps in (10, 40):
            half = 0.5 * (bps / 10000.0) * e / R
            t[f'{tag}_net_{bps}bps'] = t[f'{tag}_rr'] - (entry_coef + exit_coef) * half
        t[f'{tag}_net_comm'] = t[f'{tag}_rr'] - 2 * COMMISSION_PER_SHARE / R
    return t


def year_table(t, col, label):
    t = t[t[col].notna()].copy()
    t['year'] = t.day.str[:4]
    rows = []
    for y, g in t.groupby('year'):
        rows.append(dict(year=y, trades=len(g), r_per_trade=g[col].mean(),
                         wr=float((g[col] > 0).mean() * 100), total_r=g[col].sum(),
                         t_stat=g[col].mean() / (g[col].std(ddof=1) / np.sqrt(len(g)))
                         if len(g) > 1 and g[col].std(ddof=1) > 0 else np.nan))
    a = t[col]
    rows.append(dict(year='ALL', trades=len(t), r_per_trade=a.mean(),
                     wr=float((a > 0).mean() * 100), total_r=a.sum(),
                     t_stat=a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))))
    d = pd.DataFrame(rows)
    d.insert(0, 'arm', label)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--atr-scale', type=float, default=1.0,
                    help='multiply the daily-file ATR14 by this (RTH-vs-feed-day range correction)')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    suffix = f'_{args.tag}' if args.tag else ''
    picks = pd.read_csv(TOP20, dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    prev = dict(zip(zip(picks.day, picks.symbol), picks.prev_close))
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    out = []
    days = sorted(picks.day.unique())
    for k, day in enumerate(days):
        bars = pd.read_sql('select symbol, m, o, h, l, c, v from bars where day=?', con,
                           params=(day,))
        simulate_day(bars, picks[picks.day == day], out, args.atr_scale)
        if k % 200 == 0:
            print(f'  [{k}/{len(days)}] {day}', flush=True)
    con.close()
    t = pd.DataFrame(out)
    t['prev_close'] = [prev.get((d, s2)) for d, s2 in zip(t.day, t.symbol)]
    spread_tbl, overall = load_spread_table()
    t = add_costs(t, spread_tbl, overall)
    t.to_csv(f'{N3}/trades{suffix}.csv', index=False)

    n_pick = len(t)
    print(f'picks {n_pick:,} | no_order {int((t.status == "no_order").sum()):,} | '
          f'no_tape {int((t.status == "no_tape").sum()):,} | '
          f'no_trigger {int((t.status == "no_trigger").sum()):,} | '
          f'paper fills {int((t.paper_status == "filled").sum()):,} | '
          f'our fills {int((t.ours_status == "filled").sum()):,}', flush=True)

    arms = [('paper_net_comm', 'A  paper fill + paper commission'),
            ('paper_rr', 'A0 paper fill, zero cost'),
            ('ours_rr', 'B0 our fill (next-bar open), zero cost'),
            ('ours_net_10bps', 'B1 our fill + 10 bps spread contract'),
            ('ours_net_40bps', 'B2 our fill + 40 bps spread contract'),
            ('ours_net_band', 'B3 our fill + OUR banded cost contract'),
            ('paper_net_band', 'A3 paper fill + OUR banded cost contract')]
    tabs = pd.concat([year_table(t, c, lab) for c, lab in arms], ignore_index=True)
    tabs.to_csv(f'{N3}/year_tables{suffix}.csv', index=False)

    lines = [f'# N3 — stocks-in-play ORB, 2019-2023, XNAS.ITCH tape, our simulator '
             f'(atr_scale={args.atr_scale})', '']
    for c, lab in arms:
        g = tabs[tabs.arm == lab]
        lines += [f'## {lab}', '',
                  '| year | trades | R/trade | WR % | total R | t |', '|---|---|---|---|---|---|']
        for r in g.itertuples():
            lines.append(f'| {r.year} | {r.trades:,} | {r.r_per_trade:+.3f} | {r.wr:.1f} | '
                         f'{r.total_r:+.0f} | {r.t_stat:+.1f} |')
        lines.append('')

    # mechanism: how often does the paper fill die inside its own trigger bar?
    fa = t[t.paper_status == 'filled']
    same_bar = float((fa.paper_exit_m == fa.paper_entry_m).mean() * 100)
    fb = t[t.ours_status == 'filled']
    same_bar_b = float((fb.ours_exit_m == fb.ours_entry_m).mean() * 100)
    r_pct = (0.10 * t.atr14 / t.o5 * 100).median()
    lines += ['## mechanism', '',
              f'- median R = 0.10 x ATR14 = **{r_pct:.3f}% of the 09:30 price** '
              f'({r_pct * 100:.0f} bps) — one 40 bps round trip is '
              f'{0.40 / r_pct:.2f} R of cost.',
              f'- paper fill stopped out inside its own trigger bar: **{same_bar:.1f}%** of fills; '
              f'our next-bar-open fill: {same_bar_b:.1f}%.',
              f'- stop rate: paper {float((fa.paper_why == "stop").mean() * 100):.1f}%, '
              f'ours {float((fb.ours_why == "stop").mean() * 100):.1f}%.', '']

    # RV buckets, the paper's Figure 4 comparison (their base filters, net of commission:
    # RV < 1 -> -0.02R, RV > 1 -> +0.08R, RV > 30x -> +0.38R)
    bucket_arms = [('paper_net_comm', 'A  paper fill + commission'),
                   ('ours_rr', 'B0 our fill, zero cost'),
                   ('ours_net_10bps', 'B1 our fill + 10 bps')]
    lines += ['## R/trade by relative volume', '',
              '| RV bucket | trades | ' + ' | '.join(l for _, l in bucket_arms) + ' |',
              '|---|---|' + '---|' * len(bucket_arms)]
    f = t[t.paper_status == 'filled'].copy()
    f['bucket'] = pd.cut(f.rv, [1, 2, 3, 5, 10, 30, 1e9],
                         labels=['1-2x', '2-3x', '3-5x', '5-10x', '10-30x', '>30x'])
    for b, g in f.groupby('bucket', observed=True):
        cells = ' | '.join(f'{g[c].mean():+.3f}' for c, _ in bucket_arms)
        lines.append(f'| {b} | {len(g):,} | {cells} |')
    lines.append('')

    # daily portfolio hit ratio (the paper reports 48.4% of days positive)
    lines += ['## daily portfolio hit ratio (equal risk per trade)', '',
              '| arm | sessions | % days total R > 0 |', '|---|---|---|']
    for c, lab in arms:
        d = t[t[c].notna()].groupby('day')[c].sum()
        lines.append(f'| {lab} | {len(d):,} | {float((d > 0).mean() * 100):.1f} |')
    lines.append('')

    # tail dependence (PLAN 1, item 5)
    lines += ['## tail dependence', '',
              '| variant | ' + ' | '.join(l for _, l in bucket_arms) + ' |',
              '|---|' + '---|' * len(bucket_arms)]
    for lab in ('all', 'ex top 1%', 'ex top 5%', 'winners capped at +5R', 'winners capped at +10R'):
        cells = []
        for c, _ in bucket_arms:
            a = t[c].dropna()
            v = {'all': a, 'ex top 1%': a[a < a.quantile(.99)], 'ex top 5%': a[a < a.quantile(.95)],
                 'winners capped at +5R': a.clip(upper=5),
                 'winners capped at +10R': a.clip(upper=10)}[lab]
            cells.append(f'{v.mean():+.3f}')
        lines.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
    lines.append('')

    # long vs short
    lines += ['## side split', '',
              '| side | trades | ' + ' | '.join(l for _, l in bucket_arms) + ' |',
              '|---|---|' + '---|' * len(bucket_arms)]
    for sd, lab in ((1, 'long'), (-1, 'short')):
        g = t[(t.side == sd)]
        cells = ' | '.join(f'{g[c].mean():+.3f}' for c, _ in bucket_arms)
        lines.append(f'| {lab} | {int(g.paper_status.eq("filled").sum()):,} | {cells} |')
    lines.append('')

    # price-scale guard: split-like jumps between the daily file and the 09:30 tape
    jump = (t.o5 / t.prev_close - 1).abs() > 0.5 if 'prev_close' in t else None
    lines += ['## price-scale guard', '']
    lines.append(f'- picks whose 09:30 open differs from the daily-file prior close by more than '
                 f'50% (split-like): **{int(jump.sum()) if jump is not None else 0}** of '
                 f'{len(t):,}.')
    if jump is not None and jump.sum():
        for c, lab in bucket_arms:
            a, b = t.loc[~jump, c].mean(), t.loc[jump, c].mean()
            lines.append(f'  - {lab}: ex-jump {a:+.3f} vs jump rows {b:+.3f}')
    lines.append('')

    open(f'{N3}/tables{suffix}.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
