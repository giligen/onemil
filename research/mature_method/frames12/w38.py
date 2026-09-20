#!/usr/bin/env python3
"""F38 stage 1 — the LAST-HOUR bar walk: the floor FIRST, then the detector, then the placebo.

Row kinds (long format, one CSV):
  `F`   the detector-free FLOOR — an eligible PIT-panel non-signal name at a random minute in
        14:00-15:30, stop at a fixed 2 % and 3 % of the entry price, exits {+2R bracket, MOC}
  `S`   the last-hour HOD break (`sig38.csv`), exits {+2R, bare, lock, MOC, next open}
  `O`   the ORB-style object — the break of the 14:00-14:30 range high in 14:30-15:30, stop at the
        range low, exits {+2R, MOC}
  `CB`  matched non-signal name at the SIGNAL'S minute, the signal's own R geometry
  `CA`  the SAME name-day, a minute strictly AFTER the signal, the signal's own R geometry

Resumable per session.  Every store READ-ONLY.  TEST never loaded.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c12 import (D12, CAP, bars_arrays, load_bars, load_panel, price_exit)   # noqa: E402

OUT = f'{D12}/w38.csv'
ST = f'{D12}/w38_state.json'
WIN = (840, 931)
ORB_LO, ORB_HI = 840, 870        # the 14:00-14:30 range window (minutes 840..869)
KINDS = ('X2R', 'XBR', 'XLK', 'XMO', 'XNO')
WALK = {'X2R': 'bracket', 'XBR': 'bare', 'XLK': 'lock', 'XMO': 'moc', 'XNO': 'next'}
HDR = (['day', 'kind', 'symbol', 'key_sym', 'key_m', 'entry_m', 'level', 'next_open', 'stop',
        'r_pct', 'sw']
       + [f'{k}_{f}' for k in KINDS for f in ('rr', 'why', 'em')])


def row(day, kind, sym, key_sym, key_m, entry_m, level, nxt, stop, sw, arr, e, close_px,
        next_open_px, which):
    o, h, l, c, m = arr
    r_pct = (nxt - stop) / nxt * 100.0 if nxt > 0 else np.nan
    out = [day, kind, sym, key_sym, key_m, int(entry_m), level, nxt, stop, r_pct, sw]
    for k in KINDS:
        if k not in which:
            out += [np.nan, '', -1]
            continue
        em, rr, why = price_exit(WALK[k], o, h, l, c, m, e, stop, close_px=close_px,
                                 next_open_px=next_open_px)
        out += [rr, why, em]
    return out


def main() -> int:
    sig = pd.read_csv(f'{D12}/sig38.csv', dtype={'day': str, 'symbol': str})
    ctl = pd.read_csv(f'{D12}/ctrl38.csv', dtype={'day': str, 'symbol': str, 'ctrl': str})
    flr = pd.read_csv(f'{D12}/floor38.csv', dtype={'day': str, 'symbol': str})
    u = load_panel()[['day', 'symbol', 'close', 'next_open_d']]
    DC = {(d, s): (c, n) for d, s, c, n in u.itertuples(index=False)}
    pop = pd.read_csv(f'{D12}/orbpop38.csv', dtype={'day': str, 'symbol': str})
    days = sorted(set(sig.day) | set(flr.day))
    done = set(json.load(open(ST))['days']) if os.path.exists(ST) else set()
    if not os.path.exists(OUT):
        with open(OUT, 'w') as f:
            f.write(','.join(HDR) + '\n')
    todo = [d for d in days if d not in done]
    print(f'  {len(todo)} sessions to walk ({len(done)} done)', flush=True)
    SG = {d: g for d, g in sig.groupby('day')}
    CT = {d: g for d, g in ctl.groupby('day')}
    FL = {d: g for d, g in flr.groupby('day')}
    OP = {d: g for d, g in pop.groupby('day')}
    nrow = 0
    for i, day in enumerate(todo):
        need = set()
        for T in (SG, FL, OP):
            g = T.get(day)
            if g is not None:
                need |= set(g.symbol.astype(str))
        g = CT.get(day)
        if g is not None:
            need |= set(g.ctrl.astype(str)) | set(g.symbol.astype(str))
        bars = load_bars(day, sorted(need))
        A = {}
        for s_, b_ in bars.items():
            x = bars_arrays(b_)
            if x is not None:
                A[s_] = (x[0], x[1], x[2], x[3], x[5])       # o,h,l,c,m
        rows = []

        def idx_of(sym, minute):
            a = A.get(sym)
            if a is None:
                return None, None
            j = np.flatnonzero(a[4] == int(minute))
            return (a, int(j[0])) if len(j) else (a, None)

        # ---- the FLOOR, read first ------------------------------------------------------
        g = FL.get(day)
        if g is not None:
            for r in g.itertuples():
                a, e = idx_of(r.symbol, r.ctrl_m)
                if a is None or e is None or e + 1 >= len(a[0]):
                    continue
                E = float(a[0][e])
                if not (E > 0):
                    continue
                cp, no = DC.get((day, r.symbol), (np.nan, np.nan))
                for sw in (0.02, 0.03):
                    rows.append(row(day, 'F', r.symbol, r.symbol, int(r.ctrl_m), int(r.ctrl_m),
                                    np.nan, E, E * (1 - sw), sw, a, e, cp, no,
                                    ('X2R', 'XMO')))
        # ---- the last-hour DETECTOR -----------------------------------------------------
        g = SG.get(day)
        if g is not None:
            for r in g.itertuples():
                a, e = idx_of(r.symbol, r.entry_m)
                if a is None or e is None:
                    continue
                cp, no = DC.get((day, r.symbol), (np.nan, np.nan))
                rows.append(row(day, 'S', r.symbol, r.symbol, int(r.entry_m), int(r.entry_m),
                                float(r.level), float(r.next_open), float(r.stop), np.nan,
                                a, e, cp, no, KINDS))
        # ---- the ORB-style 14:00-14:30 range break --------------------------------------
        g = OP.get(day)
        if g is not None:
            for r in g.itertuples():
                a = A.get(r.symbol)
                if a is None:
                    continue
                o, h, l, c, m = a
                w = (m >= ORB_LO) & (m < ORB_HI)
                if w.sum() < 20:
                    continue
                rh, rl = float(h[w].max()), float(l[w].min())
                if not (rh > rl > 0):
                    continue
                post = np.flatnonzero((m >= ORB_HI) & (m <= 930) & (h > rh))
                if not len(post):
                    continue
                e0 = int(post[0])
                if e0 + 1 >= len(o) or int(m[e0 + 1]) > 931:
                    continue
                nxt = float(o[e0 + 1])
                if nxt > rh * (1 + CAP) or nxt <= rl:
                    continue
                cp, no = DC.get((day, r.symbol), (np.nan, np.nan))
                rows.append(row(day, 'O', r.symbol, r.symbol, int(m[e0]), int(m[e0 + 1]), rh,
                                nxt, rl, np.nan, a, e0 + 1, cp, no, ('X2R', 'XMO')))
        # ---- the two placebo arms -------------------------------------------------------
        g = CT.get(day)
        if g is not None:
            for r in g.itertuples():
                a, e = idx_of(r.ctrl, r.ctrl_m)
                if a is None or e is None or e + 1 >= len(a[0]):
                    continue
                E = float(a[0][e])
                if not (E > 0) or not (r.r_pct == r.r_pct):
                    continue
                cp, no = DC.get((day, r.ctrl), (np.nan, np.nan))
                rows.append(row(day, r.arm, r.ctrl, r.symbol, int(r.entry_m), int(r.ctrl_m),
                                np.nan, E, E * (1 - float(r.r_pct) / 100.0), np.nan,
                                a, e, cp, no, ('X2R', 'XBR', 'XMO')))
        if rows:
            pd.DataFrame(rows, columns=HDR).to_csv(OUT, mode='a', header=False, index=False)
            nrow += len(rows)
        done.add(day)
        json.dump({'days': sorted(done)}, open(ST, 'w'))
        if i % 25 == 0 or i == len(todo) - 1:
            print(f'  {i+1}/{len(todo)} {day} rows {nrow:,}', flush=True)
    print(f'  DONE — {nrow:,} rows -> w38.csv', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
