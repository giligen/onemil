#!/usr/bin/env python3
"""F43 — THE GEOMETRY FLOOR MAP.  Score the 12 declared cells.

  python3 f43.py join     # X6 (hold to next open) and X7 (MOC) — the daily-panel legs + the
                          #   price-scale check, written to x67.csv
  python3 f43.py score    # the 12 cells + the declared diagnostics

The cost charged per geometry is F45's MEASURED minute-of-day table, never the imputation
(PREREG §2.3).  Stores READ-ONLY.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

D6 = f'{ROOT}/research/mature_method/hod_frames6'
D14 = f'{ROOT}/research/mature_method/frames14'
STOPS = (2, 3, 4)
GEOS = ('x1', 'x2', 'x3', 'x4', 'x5')
TEST_FROM = '2026-06-01'
PB_EDGES = [0, 5, 10, 20, 30, 50, 100, 1e9]
PB_LAB = ['<$5', '$5-10', '$10-20', '$20-30', '$30-50', '$50-100', '$100+']
# a target fill is a RESTING limit -> free; a stop / flat / eod fill is marketable; an auction
# (MOC, and the opening cross) is a single-price cross with no quoted spread.
FREE_WHY = {'target'}


def split_of(day):
    return 'VAL' if day >= '2026-01-01' else 'TRAIN'


def half_of(day):
    return 'VAL' if day >= '2026-01-01' else ('H1-25' if day < '2025-07-01' else 'H2-25')


def clustered_t(v, days):
    """Day-clustered t on the mean of v."""
    v = np.asarray(v, float)
    d = pd.Series(np.asarray(days))
    ok = np.isfinite(v)
    v, d = v[ok], d[ok].values
    if len(v) < 5:
        return np.nan
    g = pd.DataFrame(dict(v=v, d=d)).groupby('d').v
    s, n = g.sum().values, g.size().values
    mu = v.mean()
    num = (s - n * mu)
    var = (num ** 2).sum()
    se = np.sqrt(var) / len(v)
    return mu / se if se > 0 else np.nan


# ------------------------------------------------------------------ the MEASURED cost surface
class Cost:
    """`sp(price, minute)` = the measured FULL NBBO spread as % of price, from F45's table.

    Median per (price band x declared clock), linearly interpolated in minute between clocks and
    clamped outside them.  A band with no measurement at a clock falls back to the all-band median
    at that clock (loudly, once).
    """

    def __init__(self):
        t = pd.read_csv(f'{D14}/f45_minutes.csv', dtype={'day': str, 'symbol': str})
        t = t[t.sp_mean.notna() & (t.n_q > 0) & (t.price > 0)].copy()
        t['sp_pct'] = t.sp_mean / t.price * 100
        self.clocks = np.array(sorted(t.clock_m.unique()), float)
        self.glob = t.groupby('clock_m').sp_pct.median().reindex(self.clocks).values
        self.by = {}
        miss = []
        for pb in PB_LAB:
            z = t[t.pb == pb]
            if not len(z):
                miss.append(pb)
                continue
            v = z.groupby('clock_m').sp_pct.median().reindex(self.clocks).values
            bad = ~np.isfinite(v)
            if bad.any():
                v = np.where(bad, self.glob, v)
            self.by[pb] = v
        if miss:
            print(f'  WARNING Cost: no measurement for price bands {miss}; they use the all-band '
                  f'median at every clock', flush=True)
        print(f'  cost surface: {len(self.clocks)} clocks x {len(self.by)} price bands, '
              f'all-band median {np.nanmedian(self.glob):.3f} % of price', flush=True)

    def sp(self, price, minute):
        price = np.asarray(price, float); minute = np.asarray(minute, float)
        pbi = np.searchsorted(np.array(PB_EDGES[1:-1], float), price, 'right')
        out = np.empty(len(price))
        for i, pb in enumerate(PB_LAB):
            m = pbi == i
            if not m.any():
                continue
            v = self.by.get(pb, self.glob)
            out[m] = np.interp(minute[m], self.clocks, v)
        return out


# ------------------------------------------------------------------ X6 / X7 — the daily legs
def join67():
    """The MOC close of day t and the next session's open, from `daily_bars` — the SAME source as
    the intraday bars, so no cross-vendor price scale is involved.  The scale is checked anyway."""
    w = pd.read_csv(f'{D14}/w43.csv', dtype={'day': str, 'ctrl': str},
                    usecols=['day', 'ctrl', 'ctrl_entry_m', 'entry'])
    syms = sorted(w.ctrl.unique())
    d0, d1 = w.day.min(), w.day.max()
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=180)
    out = []
    CH = 400
    for i in range(0, len(syms), CH):
        ch = syms[i:i + CH]
        q = ('select symbol, date, open, close from daily_bars where date >= ? and date <= ? '
             f'and symbol in ({",".join("?" * len(ch))})')
        out.append(pd.read_sql(q, con, params=[d0, (pd.Timestamp(d1) + pd.Timedelta(days=10)
                                                    ).strftime('%Y-%m-%d')] + ch))
    con.close()
    db = pd.concat(out, ignore_index=True)
    db['date'] = db.date.astype(str).str.slice(0, 10)
    db = db.sort_values(['symbol', 'date']).drop_duplicates(['symbol', 'date'])
    db['next_open'] = db.groupby('symbol').open.shift(-1)
    db['next_date'] = db.groupby('symbol').date.shift(-1)
    j = w.merge(db[['symbol', 'date', 'close', 'next_open', 'next_date']],
                left_on=['ctrl', 'day'], right_on=['symbol', 'date'], how='left')
    cov_c = j.close.notna().mean(); cov_n = j.next_open.notna().mean()
    print(f'  X7 close join {cov_c:.1%} | X6 next-open join {cov_n:.1%} of {len(j):,} keys',
          flush=True)
    # price-scale check: the daily close must live inside the day's own intraday price scale.
    r = (j.close / j.entry).dropna()
    bad = float(((r < 0.5) | (r > 2.0)).mean())
    print(f'  price-scale check (daily close / intraday entry price): median {r.median():.4f}, '
          f'{bad:.2%} outside [0.5, 2.0] -> DROPPED', flush=True)
    j.loc[(j.close / j.entry < 0.5) | (j.close / j.entry > 2.0), ['close', 'next_open']] = np.nan
    j['x6_pct'] = (j.next_open / j.entry - 1.0) * 100
    j['x7_pct'] = (j.close / j.entry - 1.0) * 100
    j[['day', 'ctrl', 'ctrl_entry_m', 'entry', 'close', 'next_open', 'next_date',
       'x6_pct', 'x7_pct']].to_csv(f'{D14}/x67.csv', index=False)
    print(f'  x67.csv written ({len(j):,} rows)', flush=True)
    return 0


# ------------------------------------------------------------------ scoring
def load(cost):
    w = pd.read_csv(f'{D14}/w43.csv', dtype={'day': str, 'ctrl': str})
    w = w[w.day < TEST_FROM]
    x = pd.read_csv(f'{D14}/x67.csv', dtype={'day': str, 'ctrl': str},
                    usecols=['day', 'ctrl', 'ctrl_entry_m', 'x6_pct', 'x7_pct'])
    w = w.merge(x, on=['day', 'ctrl', 'ctrl_entry_m'], how='left')
    w['half'] = [half_of(d) for d in w.day]
    w['split'] = [split_of(d) for d in w.day]
    w['entry_cost'] = 0.5 * cost.sp(w.entry.values, w.ctrl_entry_m.values)
    return w


def geo_frame(w, cost, geo, s):
    """One geometry x one stop width -> a frame with gross %, cost %, net %, in % of ENTRY PRICE."""
    if geo in ('x6', 'x7'):
        g = w[['day', 'ctrl', 'entry', 'half', 'split', 'entry_cost', 'ctrl_entry_m']].copy()
        g['gross'] = w[f'{geo}_pct']
        if geo == 'x7':
            g['exit_cost'] = 0.0                       # the closing cross: no quoted spread
        else:
            # the overnight exit is marketable in the minute AFTER the cross (F40's standing rule)
            g['exit_cost'] = 0.5 * cost.sp(w.entry.values, np.full(len(w), 571.0))
        g['why'] = geo
    else:
        rr = w[f'{geo}_rr_{s}']
        why = w[f'{geo}_why_{s}'].astype(str)
        xm = w[f'{geo}_xm_{s}']
        g = w[['day', 'ctrl', 'entry', 'half', 'split', 'entry_cost', 'ctrl_entry_m']].copy()
        g['gross'] = rr * s                            # rr x (R/E) x 100 = % of entry price
        g['why'] = why
        free = why.str.replace('pp+', '', regex=False).isin(FREE_WHY)
        g['exit_cost'] = np.where(free, 0.0, 0.5 * cost.sp(w.entry.values, xm.values))
    g['net'] = g.gross - g.entry_cost - g.exit_cost
    g['rr_net'] = g.net / s                            # net in R, for the cell's own stop width
    return g.dropna(subset=['net'])


def ex_top5(g, col='net'):
    """Mean of col with the top 5 % removed INSIDE each split (F35's cap rule)."""
    out = {}
    for sp in ('H1-25', 'H2-25', 'VAL'):
        z = g[g.half == sp]
        if len(z) < 40:
            out[sp] = np.nan
            continue
        cut = z[col].quantile(0.95)
        out[sp] = float(z[z[col] <= cut][col].mean())
    return out


UNCAPPED = {'x5', 'x6', 'x7'}
NAME = {'x1': '+2R bracket, flat 15:55', 'x2': 'ORB static lock, flat 15:45',
        'x3': 'BF R-trail, flat 15:45', 'x4': 'BF R-trail + 50%@2R partial, flat 15:45',
        'x5': 'bare stop, flat 15:55', 'x6': 'hold to NEXT OPEN', 'x7': 'MOC (official close)'}
CELLS = [(g, 2) for g in ('x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7')] + \
        [(g, 3) for g in ('x2', 'x4', 'x5', 'x6', 'x7')]


def score():
    cost = Cost()
    w = load(cost)
    print(f'\n  population {len(w):,} detector-free control keys over {w.day.nunique()} sessions '
          f'(TRAIN {int((w.split=="TRAIN").sum()):,} / VAL {int((w.split=="VAL").sum()):,})',
          flush=True)
    # ---- G-X1: the reproduction gate against F34
    g1 = geo_frame(w, cost, 'x1', 2)
    print(f'\nGATE G-X1: F34 read the +2R bracket at s=2 % as GROSS -0.136 % TRAIN (net -0.389 % '
          f'under the IMPUTED cost; this pass charges the MEASURED one, so only GROSS is a gate).',
          flush=True)
    gt = float(g1[g1.split == 'TRAIN'].gross.mean())
    for sp in ('TRAIN', 'VAL'):
        z = g1[g1.split == sp]
        print(f'   frames14 gross {z.gross.mean():+.4f} % | measured cost '
              f'{(z.entry_cost + z.exit_cost).mean():.4f} % | net {z.net.mean():+.4f} % '
              f'({sp}, n={len(z):,})', flush=True)
    assert abs(gt - (-0.136)) < 0.01, f'X1 does not reproduce F34 gross (-0.136 vs {gt:.4f})'
    print('   G-X1 PASSES', flush=True)
    rows = []
    print(f'\n### THE 12 SCORED CELLS — unconditional, no admission rule, % of ENTRY PRICE, '
          f'cost MEASURED (F45)\n', flush=True)
    print(f'{"cell":>5} {"geometry":<42} {"s%":>3} {"n":>8} {"gross":>8} {"cost":>7} {"net":>8} '
          f'{"t":>7} {"H1-25":>8} {"H2-25":>8} {"VAL":>8} {"net R":>8} {"ex5 ok":>7}', flush=True)
    for geo, s in CELLS:
        g = geo_frame(w, cost, geo, s)
        if not len(g):
            continue
        t = clustered_t(g.net.values, g.day.values)
        hv = {h: float(g[g.half == h].net.mean()) for h in ('H1-25', 'H2-25', 'VAL')}
        allpos = all(v > 0 for v in hv.values() if v == v)
        e5 = ex_top5(g) if geo in UNCAPPED else None
        e5ok = (all(v > 0 for v in e5.values() if v == v) if e5 else None)
        rows.append(dict(cell=f'{geo.upper()}-{s}', geo=geo, s=s, n=len(g),
                         gross=g.gross.mean(), cost=(g.entry_cost + g.exit_cost).mean(),
                         net=g.net.mean(), t=t, **{f'net_{k}': v for k, v in hv.items()},
                         net_R=g.rr_net.mean(), all_halves_pos=allpos,
                         **({f'ex5_{k}': v for k, v in e5.items()} if e5 else {}),
                         ex5_ok=e5ok,
                         passes=bool(allpos and (e5ok if e5 is not None else False))))
        e5s = 'n/a' if e5 is None else ('YES' if e5ok else 'no')
        print(f'{geo.upper()+"-"+str(s):>5} {NAME[geo]:<42} {s:3d} {len(g):8,d} '
              f'{g.gross.mean():+8.3f} {(g.entry_cost+g.exit_cost).mean():7.3f} '
              f'{g.net.mean():+8.3f} {t:+7.1f} {hv["H1-25"]:+8.3f} {hv["H2-25"]:+8.3f} '
              f'{hv["VAL"]:+8.3f} {g.rr_net.mean():+8.3f} {e5s:>7}', flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(f'{D14}/cells43.csv', index=False)
    npass = int(R.passes.sum())
    print(f'\n  cells clearing the pre-committed bar (positive on BOTH TRAIN halves AND VAL, and '
          f'ex-top-5 % positive where the geometry is uncapped): {npass} of {len(R)}', flush=True)
    # ---- the declared diagnostics
    print('\n  DIAGNOSTIC — by entry-hour band (net % of price, s = 2 %):', flush=True)
    bands = [('09:37-10:30', 577, 630), ('10:30-11:30', 630, 690), ('11:30-13:00', 690, 780),
             ('13:00-14:01', 780, 842)]
    print(f'{"geo":>4} ' + ' '.join(f'{b[0]:>12}' for b in bands), flush=True)
    for geo in ('x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'):
        gm = geo_frame(w, cost, geo, 2)
        cells = []
        for _, lo, hi in bands:
            z = gm[(gm.ctrl_entry_m >= lo) & (gm.ctrl_entry_m < hi)]
            cells.append(f'{z.net.mean():+12.3f}' if len(z) else f'{"-":>12}')
        print(f'{geo.upper():>4} ' + ' '.join(cells), flush=True)
    print('\n  DIAGNOSTIC — the auction-free variant of X6 (exit IN the opening cross, ratio 0):',
          flush=True)
    for s in (2,):
        g = geo_frame(w, cost, 'x6', s).copy()
        g2 = g.copy(); g2['net'] = g2.gross - g2.entry_cost
        hv = {h: float(g2[g2.half == h].net.mean()) for h in ('H1-25', 'H2-25', 'VAL')}
        e5 = ex_top5(g2)
        print(f'    net {g2.net.mean():+.3f} %  H1 {hv["H1-25"]:+.3f}  H2 {hv["H2-25"]:+.3f}  '
              f'VAL {hv["VAL"]:+.3f}  | ex-top-5 % H1 {e5["H1-25"]:+.3f} H2 {e5["H2-25"]:+.3f} '
              f'VAL {e5["VAL"]:+.3f}', flush=True)
    return 0


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else 'score'
    sys.exit(join67() if a == 'join' else score())
