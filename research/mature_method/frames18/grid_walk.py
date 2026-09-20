#!/usr/bin/env python3
"""frames18 F55 — the passive mirror short over a (offset k, window w) grid, with the Reg SHO 201
rail.  PREREG frames18/PREREG.md (frozen before this ran).

Population, entry mechanic and exit are frames17 F52's, unchanged.  What is new:
  * k in {0.4,0.6,0.8,1.0}% x w in {3,5,10} min = 12 cells.
  * the SSR rail: session running-min low <= prev_close*0.90 makes the fill bar SSR-active; such a
    fill is valid only if the limit is strictly above the NBB at that minute (NBB measured from SIP
    quotes; unavailable -> fill VOID).

The fill bar depends only on k (first bar in e..e+9 whose high >= limit); w only decides whether
that touch is inside the order's life.  So the touch offset is computed once per (row,k) and the
three windows are read off it.
"""
import glob
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames16')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_frames5')
from common5 import attach_instrument                                  # noqa: E402
from short_walk import swalk                                           # noqa: E402

D = f'{ROOT}/research/mature_method/frames18'
D16 = f'{ROOT}/research/mature_method/frames16'
SIP = f'{ROOT}/research/bf_zero/bars_sip.db'
CACHE = f'{ROOT}/data/cache.db'
BORROW = f'{ROOT}/research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv'
TEST_FROM = '2026-06-01'
KS = (0.004, 0.006, 0.008, 0.010)
WS = (3, 5, 10)
WMAX = max(WS)


def population():
    fs = sorted(glob.glob(f'{D16}/sw_*.csv'))
    d = pd.concat([pd.read_csv(f, dtype={'symbol': str, 'day': str},
                               keep_default_na=False, na_values=['']) for f in fs],
                  ignore_index=True)
    d = d[(d.day < TEST_FROM) & d.gate5.astype(bool) & d.f_mir2.astype(bool)
          & (d.price >= 5)].copy()
    d = attach_instrument(d)
    d = d[d.asset_class != 'wrapper']
    b = pd.read_csv(BORROW, dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    b['etb'] = b.shortable.astype(str).str.lower().eq('true') & \
        b.easy_to_borrow.astype(str).str.lower().eq('true')
    m = dict(zip(b.symbol, b.etb))
    d['etb'] = d.symbol.map(m).fillna(False).astype(bool)
    d['split'] = np.where(d.day < '2026-01-01', 'TRAIN', 'VAL')
    print(f'[pop] {len(d):,} MIR2 gate5 price>=$5 ex-wrapper rows, ETB '
          f'{float(d.etb.mean())*100:.1f}%', flush=True)
    return d[d.etb].copy()


def prev_closes(pairs):
    """{(day,symbol): prev trading day's close} from cache.db::daily_bars (Alpaca, same vendor as
    the 1-min tape).  Missing -> absent from the dict -> SSR undetermined."""
    syms = sorted({s for _, s in pairs})
    con = sqlite3.connect(f'file:{CACHE}?mode=ro', uri=True, timeout=300)
    out = {}
    CH = 800
    frames = []
    for i in range(0, len(syms), CH):
        ch = syms[i:i + CH]
        qs = ','.join('?' * len(ch))
        frames.append(pd.read_sql(
            f"select symbol,bar_date,close from daily_bars where symbol in ({qs}) "
            f"and bar_date >= '2024-12-01' and bar_date < '{TEST_FROM}'", con, params=ch))
    con.close()
    db = pd.concat(frames, ignore_index=True)
    db['bar_date'] = db.bar_date.astype(str).str.slice(0, 10)
    db = db.drop_duplicates(['symbol', 'bar_date']).sort_values(['symbol', 'bar_date'],
                                                                kind='mergesort')
    db['prev'] = db.groupby('symbol')['close'].shift(1)
    db = db.dropna(subset=['prev'])
    out = {(d_, s): float(p) for s, d_, p in zip(db.symbol, db.bar_date, db.prev)}
    print(f'[prevclose] {len(out):,} (day,symbol) prev closes loaded', flush=True)
    return out


def run():
    d = population()
    pc = prev_closes(list(zip(d.day, d.symbol)))
    con = sqlite3.connect(f'file:{SIP}?mode=ro', uri=True, timeout=300)
    rows = []
    ndays = d.day.nunique()
    for i, (day, g) in enumerate(d.groupby('day', sort=True)):
        syms = sorted(g.symbol.unique())
        qs = ','.join('?' * len(syms))
        b = pd.read_sql(f'select symbol,t,o,h,l,c,v from bars where day=? and symbol in ({qs})',
                        con, params=[day] + syms)
        if b.empty:
            continue
        off = 4 if ('2025-03-09' <= day < '2025-11-02') or ('2026-03-08' <= day < '2026-11-01') \
            else 5
        b['m'] = ((b.t.str.slice(11, 13).astype(int) - off) * 60
                  + b.t.str.slice(14, 16).astype(int))
        b = b[(b.m >= 570) & (b.m < 960)].sort_values(['symbol', 'm'], kind='mergesort')
        byx = {s: x for s, x in b.groupby('symbol', sort=False)}
        for r in g.itertuples():
            x = byx.get(r.symbol)
            if x is None:
                continue
            mv = x.m.values
            o, hi, lo, c = (x[kk].values.astype(float) for kk in ('o', 'h', 'l', 'c'))
            idx = np.flatnonzero(mv == int(r.entry_m))
            if not len(idx):
                continue
            e = int(idx[0])
            # --- SSR state: running min low <= prev_close*0.90, persists for the rest of the day
            p = pc.get((day, r.symbol))
            runmin = np.minimum.accumulate(lo)
            for k in KS:
                limit = r.ref * (1.0 + k)
                fj = None
                for off2 in range(WMAX):
                    j = e + off2
                    if j >= len(mv):
                        break
                    if hi[j] >= limit:
                        fj = j
                        break
                base = dict(day=day, symbol=r.symbol, hour=int(r.hour), k=k, ref=r.ref,
                            split=r.split, rr_reacting=r.rr_a_bare,
                            react_entry_m=int(r.entry_m), react_entry=float(r.entry),
                            react_exit_m=int(r.exitm_a_bare), react_rpct=float(r.rpct_a))
                if fj is None:
                    base.update(touch_off=-1, fill_m=-1, entry=np.nan, rpct=np.nan,
                                rr=np.nan, why='unfilled', exit_m=-1,
                                ssr_active=False, ssr_undet=(p is None), limit=limit)
                else:
                    entry = limit
                    stop = entry * 1.02
                    rr, why, xm = swalk(o, hi, lo, c, mv, fj, entry, stop, 'bare')
                    ssr = (p is not None) and bool(runmin[fj] <= p * 0.90)
                    base.update(touch_off=fj - e, fill_m=int(mv[fj]), entry=entry,
                                rpct=(stop - entry) / entry, rr=rr, why=why, exit_m=int(xm),
                                ssr_active=ssr, ssr_undet=(p is None), limit=limit)
                rows.append(base)
        if i % 60 == 0:
            print(f'  {i+1}/{ndays} days -> {len(rows):,} rows', flush=True)
    con.close()
    out = pd.DataFrame(rows)
    out.to_csv(f'{D}/grid18.csv', index=False)
    tot = len(out)
    print(f'[walk] {tot:,} (signal x k) rows -> grid18.csv', flush=True)
    for w in WS:
        fr = ((out.touch_off >= 0) & (out.touch_off < w)).mean()
        print(f'  w={w:2d}min raw fill rate {fr*100:.1f}%', flush=True)
    ssr = out[(out.touch_off >= 0) & out.ssr_active]
    print(f'[ssr] fills with SSR active (any w): {len(ssr):,}; undetermined prev-close rows '
          f'{int(out.ssr_undet.sum()):,}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(run())
