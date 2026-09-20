#!/usr/bin/env python3
"""frames21 step 4 — frames15/armB_intra.py's mirror-short signal construction, UNCHANGED logic,
run on the 2024H2 extension store (hourly_ext15.parquet + frames21/raw/bars_sip_ext.db) instead of
frames15's own hourly15.parquet + research/bf_zero/bars_sip.db. DATA ONLY: emits every signal-hour
row with gate5 + f_mir + the three walked exits (bare/tgt/lock) exactly as armB_intra.py does -- no
scoring, no cell tables, no t-stats. Writes frames21/signals_ext.csv (one row per signal-hour that
found a valid next-bar entry, all months of the extension pooled -- the whole window fits in memory
so no per-month checkpoint file is needed here).
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
D = f'{ROOT}/research/mature_method/frames21'
SIP = f'{D}/raw/bars_sip_ext.db'
OPEN_M, EOD_M = 570, 955
CAP = 0.006
STOP = 0.02
SLIP = 0.001
LOCK_ARM, LOCK_STOP = 1.75, 0.5


def signals_by_day():
    h = pd.read_parquet(f'{D}/hourly_ext15.parquet',
                        columns=['symbol', 'day', 'hour', 'hrv', 'hour_ret', 'sus2_3'])
    h['symbol'] = h.symbol.astype(str)
    h['day'] = h.day.astype(str)
    m = ((h.hrv >= 3.0) | h.sus2_3.fillna(False).astype(bool))
    h = h[m & h.hour.between(9, 14)]
    h['f_hrv3'] = (h.hrv >= 3.0)
    h['f_sus'] = h.sus2_3.fillna(False).astype(bool)
    h['f_abs'] = (h.hrv >= 3.0) & (h.hour_ret.abs() <= 0.01)
    h['f_mir'] = (h.hrv >= 3.0) & (h.hour_ret.abs() > 0.02)
    print(f'[signals_ext] {len(h):,} signal hours over {h.day.nunique()} sessions '
          f'(hrv>=3 {int(h.f_hrv3.sum()):,}, sus(2,3) {int(h.f_sus.sum()):,}, '
          f'abs {int(h.f_abs.sum()):,}, mirror {int(h.f_mir.sum()):,})', flush=True)
    return h


def walk(o, hi, lo, c, m, e, entry, stop, mode):
    n = len(o)
    R = entry - stop
    tgt = entry + 2.0 * R
    arm = entry + LOCK_ARM * R
    lock = entry + LOCK_STOP * R
    cur_stop = stop
    armed = False
    for k in range(e + 1, n):
        if m[k] >= EOD_M:
            return (o[k] - entry) / R, 'eod', m[k]
        if mode == 'lock' and armed is False and hi[k] >= arm:
            armed = True
            cur_stop = lock
        if lo[k] <= cur_stop:
            px = min(cur_stop, o[k]) * (1.0 - SLIP)
            return (px - entry) / R, ('lock_stop' if armed else 'stop'), m[k]
        if mode == 'tgt' and c[k] >= tgt:
            return 2.0, 'target', m[k]
        if mode == 'lock' and armed is False and hi[k] >= arm:
            armed = True
            cur_stop = lock
    return (c[-1] - entry) / R, 'eod', m[-1]


def run():
    sig = signals_by_day()
    months = sorted(sig.day.str[:7].unique())
    con = sqlite3.connect(f'file:{SIP}?mode=ro', uri=True, timeout=120)
    all_rows = []
    for mo in months:
        rows = []
        for day, g in sig[sig.day.str[:7] == mo].groupby('day', sort=True):
            syms = sorted(g.symbol.unique())
            qs = ','.join('?' * len(syms))
            b = pd.read_sql(f'select symbol,t,o,h,l,c,v from bars where day=? and symbol in ({qs})',
                            con, params=[day] + syms)
            if b.empty:
                continue
            off = 4 if ('2025-03-09' <= day < '2025-11-02') or ('2026-03-08' <= day < '2026-11-01') \
                else 5
            hh = b.t.str.slice(11, 13).astype(int)
            mm = b.t.str.slice(14, 16).astype(int)
            b['m'] = (hh - off) * 60 + mm
            b = b[(b.m >= OPEN_M) & (b.m < 960)].sort_values(['symbol', 'm'], kind='mergesort')
            byx = {s: x for s, x in b.groupby('symbol', sort=False)}
            for r in g.itertuples():
                x = byx.get(r.symbol)
                if x is None or len(x) < 20:
                    continue
                mv = x.m.values
                o, hi, lo, c = (x[k].values.astype(float) for k in ('o', 'h', 'l', 'c'))
                cut = (r.hour + 1) * 60
                pre = mv < cut
                if not pre.any():
                    continue
                sess_open = o[0]
                gate5 = bool(hi[pre].max() >= sess_open * 1.05)
                nxt = np.flatnonzero(mv >= cut)
                if not len(nxt):
                    continue
                e = int(nxt[0])
                if mv[e] > cut + 5:
                    continue
                ref = float(c[pre][-1])
                entry = float(o[e])
                if entry > ref * (1.0 + CAP) or entry <= 0:
                    continue
                stop = entry * (1.0 - STOP)
                rec = dict(day=day, symbol=r.symbol, hour=int(r.hour), entry_m=int(mv[e]),
                           entry=entry, gate5=gate5, f_hrv3=bool(r.f_hrv3), f_sus=bool(r.f_sus),
                           f_abs=bool(r.f_abs), f_mir=bool(r.f_mir), price=entry)
                for mode, tag in (('bare', 'bare'), ('tgt', 'tgt'), ('lock', 'lock')):
                    rr, why, xm = walk(o, hi, lo, c, mv, e, entry, stop, mode)
                    rec[f'rr_{tag}'] = rr
                    rec[f'why_{tag}'] = why
                    rec[f'exitm_{tag}'] = xm
                rows.append(rec)
        print(f'  {mo}: {len(rows):,} trades', flush=True)
        all_rows.extend(rows)
    con.close()
    out = pd.DataFrame(all_rows)
    out.to_csv(f'{D}/signals_ext.csv', index=False)
    print(f'[signals_ext] wrote signals_ext.csv {len(out):,} rows', flush=True)


if __name__ == '__main__':
    run()
