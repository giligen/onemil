#!/usr/bin/env python3
"""frames9 / F28 stage 1 — THE POND WALK.

Prices, under the shipped +2 R cap (X0) and the bare stop (G3), three objects per pond:

  sig   HOD's admitted signals (B2 cascade, price floor moved to $5 so every rung is a subset)
        that fall inside the pond
  b     the 12 nearest non-signal names OF THE POND at the SIGNAL's own minute
  u     12 random names OF THE POND at the SIGNAL's own minute — the pond's universe bound

Controls carry the signal's own `r_pct` applied to the control bar's open, so the only thing that
moves between HOD's own pond and these two is the NAME POPULATION: the clock, the stop width and
the bracket are all HOD's.

Resumable per session.  Read-only on every store.  TEST is never touched (`FREEZE.md`).

  python3 w9.py build   # pools + the reproduction gates
  python3 w9.py walk    # the bar pass (resumable, `w9_state.json`)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c9                                                              # noqa: E402
import g8                                                              # noqa: E402
from pass2 import load_bars                                            # noqa: E402

D9 = c9.D9
ST = f'{D9}/w9_state.json'
OUT = {'sig': f'{D9}/w9_sig.csv', 'b': f'{D9}/w9_b.csv', 'u': f'{D9}/w9_u.csv'}
HDR = {
    'sig': ['day', 'symbol', 'entry_m'] + [f'{x}_{g}' for g in c9.GE for x in ('rr', 'why', 'xm')],
    'b': ['day', 'symbol', 'entry_m', 'pond', 'ctrl']
         + [f'{x}_{g}' for g in c9.GE for x in ('rr', 'why')],
    'u': ['day', 'symbol', 'entry_m', 'pond', 'ctrl']
         + [f'{x}_{g}' for g in c9.GE for x in ('rr', 'why')],
}


def arrays(gg, m_lo=570, m_hi=960):
    r = gg[(gg.m >= m_lo) & (gg.m < m_hi)]
    if len(r) < 10:
        return None
    o, h, l, c, v = (r[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v'))
    return o, h, l, c, v, r.m.values.astype(int)


def build():
    print('== frames9 / F28 stage 1 — reproduction gates ==', flush=True)
    c9.gates()
    print('== the ponds ==', flush=True)
    P = c9.ponds()
    print('== the signal set (rung $5 — the superset of every rung) ==', flush=True)
    s = c9.signals(5.0)
    s.to_csv(f'{D9}/sig9.csv', index=False)
    for pond in c9.PONDS:
        B, U, miss = c9.pools(s, P[pond], pond)
        B['pond'] = pond
        U['pond'] = pond
        B.to_csv(f'{D9}/pool_b_{pond}.csv', index=False)
        U.to_csv(f'{D9}/pool_u_{pond}.csv', index=False)
        print(f'  {pond}: arm b {len(B)} rows / arm u {len(U)} rows for '
              f'{int(s[f"in_{pond}"].sum()) - miss} signals (unmatchable {miss}) | '
              f'median |dlog| {B.dist.median():.3f}', flush=True)


def walk():
    s = pd.read_csv(f'{D9}/sig9.csv', dtype={'day': str, 'symbol': str},
                    keep_default_na=False, na_values=[''])
    s = s[s.in_UNION]
    PB = pd.concat([pd.read_csv(f'{D9}/pool_b_{p}.csv', dtype={'day': str, 'symbol': str,
                                                               'ctrl': str, 'pond': str})
                    for p in c9.PONDS], ignore_index=True)
    PU = pd.concat([pd.read_csv(f'{D9}/pool_u_{p}.csv', dtype={'day': str, 'symbol': str,
                                                               'ctrl': str, 'pond': str})
                    for p in c9.PONDS], ignore_index=True)
    print(f'signals {len(s)}  arm b {len(PB)}  arm u {len(PU)}', flush=True)
    RP = {(r.day, r.symbol, int(r.entry_m)): float(r.r_pct) for r in s.itertuples()}
    SD = {d: g for d, g in s.groupby('day')}
    BD = {d: g for d, g in PB.groupby('day')}
    UD = {d: g for d, g in PU.groupby('day')}
    done = set(json.load(open(ST))['done']) if os.path.exists(ST) else set()
    days = [d for d in sorted(s.day.unique()) if d not in done]
    print(f'{len(days)} sessions to walk ({len(done)} done)', flush=True)

    for nd, day in enumerate(days):
        ss = SD[day]
        gb, gu = BD.get(day), UD.get(day)
        syms = set(ss.symbol)
        for g in (gb, gu):
            if g is not None:
                syms |= set(g.ctrl.astype(str))
        bars = load_bars(day, sorted(syms))
        arr = {}
        for k, gg in bars.items():
            a = arrays(gg)
            if a is not None:
                arr[k] = a
        buf = {k: [] for k in OUT}

        for r in ss.itertuples():
            A = arr.get(r.symbol)
            if A is None:
                continue
            o, h, l, c, v, m = A
            k = np.where(m == int(r.entry_m))[0]
            if not len(k):
                continue
            R = g8.geoms(o, h, l, c, m, int(k[0]), float(r.stop), want=c9.GE)
            if R is None:
                continue
            row = [day, r.symbol, int(r.entry_m)]
            for g in c9.GE:
                row += [R[g][0], R[g][1], R[g][2]]
            buf['sig'].append(row)

        for tag, pool in (('b', gb), ('u', gu)):
            if pool is None:
                continue
            for r in pool.itertuples():
                A = arr.get(str(r.ctrl))
                rp = RP.get((day, r.symbol, int(r.entry_m)))
                if A is None or rp is None:
                    continue
                o, h, l, c, v, m = A
                k = np.where(m == int(r.entry_m))[0]
                if not len(k):
                    continue
                e = int(k[0])
                R = g8.geoms(o, h, l, c, m, e, float(o[e]) * (1.0 - rp / 100.0), want=c9.GE)
                if R is None:
                    continue
                row = [day, r.symbol, int(r.entry_m), str(r.pond), str(r.ctrl)]
                for g in c9.GE:
                    row += [R[g][0], R[g][1]]
                buf[tag].append(row)

        for k, rows in buf.items():
            if rows:
                pd.DataFrame(rows, columns=HDR[k]).to_csv(
                    OUT[k], mode='a', header=not os.path.exists(OUT[k]), index=False)
        done.add(day)
        json.dump({'done': sorted(done)}, open(ST, 'w'))
        if nd % 20 == 0 or nd == len(days) - 1:
            print(f'  [{nd + 1}/{len(days)}] {day} syms {len(arr)} sig {len(buf["sig"])} '
                  f'b {len(buf["b"])} u {len(buf["u"])}', flush=True)
    print('W9 WALK DONE', flush=True)


if __name__ == '__main__':
    {'build': build, 'walk': walk}[sys.argv[1]]()
