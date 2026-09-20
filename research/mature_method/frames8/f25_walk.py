#!/usr/bin/env python3
"""frames8 / F25 stage 1 — THE TRANSPLANT WALK.

Re-prices, under every declared geometry, three populations that already exist:

  sig   the 7,027 admitted HOD signals (B2's 2,328 booked trades are a subset; the full set is
        needed for the RB re-booking cell, where a geometry's own exit_m changes slot occupancy)
  b     pass 6's arm b — the matched non-signal name at the signal's OWN minute   (51,051 keys)
  d     pass 6's arm d — the matched non-signal name at random eligible minutes   (288,174 keys)
  a     arm a' — the same name-day at a random LATER minute, 10 per booked trade  (causal)

Controls carry the booked trade's own `r_pct` applied to the control bar's open, exactly as
`hod_frames6/walk20.py` built them, so the GEOMETRY is the only thing that moves.

Resumable per session.  Read-only on every store.  TEST is never touched.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames8')
sys.path.insert(0, f'{ROOT}/research/mature_method/frames7')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_frames6')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import g8                                                              # noqa: E402
import c7                                                              # noqa: E402
from pass2 import load_bars                                            # noqa: E402

D8 = f'{ROOT}/research/mature_method/frames8'
D6 = f'{ROOT}/research/mature_method/hod_frames6'
SEED = 20260920
N_A = 10
G = g8.GEOMS
SUBS = ('G1a', 'G1b', 'G2v', 'G2pl')
ST = f'{D8}/walk25_state.json'
OUT = {k: f'{D8}/w_{k}.csv' for k in ('sig', 'b', 'd', 'a')}
HDR = {
    'sig': ['day', 'symbol', 'entry_m'] + [f'{x}_{g}' for g in G for x in ('rr', 'why', 'xm')]
           + [f'{x}_{s}' for s in SUBS for x in ('rr', 'why', 'xm')],
    'b': ['day', 'symbol', 'entry_m', 'ctrl'] + [f'{x}_{g}' for g in G for x in ('rr', 'why')],
    'd': ['day', 'symbol', 'entry_m', 'ctrl', 'ctrl_entry_m']
         + [f'{x}_{g}' for g in G for x in ('rr', 'why')],
    'a': ['day', 'symbol', 'entry_m', 'ctrl_entry_m']
         + [f'{x}_{g}' for g in G for x in ('rr', 'why')],
}


def load_sig():
    """The admitted signal set + its ATR14 (for the G1a sub-arm), ONE definition."""
    from common6 import base_book
    b, s = base_book(verbose=False)
    s = s[['day', 'symbol', 'entry_m', 'stop', 'r_pct', 'level', 'rr', 'next_open',
           'sp_pct', 'imputed', 'split', 'wk', 'exit_m']].copy()
    at = pd.read_csv(f'{ROOT}/research/mature_method/hod_filter_stack/sig2.csv',
                     usecols=['day', 'symbol', 'entry_m', 'atr14_pct'],
                     dtype={'day': str, 'symbol': str}, keep_default_na=False, na_values=[''])
    at = at.drop_duplicates(['day', 'symbol', 'entry_m'])
    s = s.merge(at, on=['day', 'symbol', 'entry_m'], how='left')
    s['booked'] = pd.Series(list(zip(s.day, s.symbol, s.entry_m))).isin(
        set(zip(b.day, b.symbol, b.entry_m))).values
    return s, b


def a_keys(bk):
    """10 CAUSAL later minutes per booked trade, drawn once with a fixed seed."""
    p = f'{D8}/a_keys.csv'
    if os.path.exists(p):
        return pd.read_csv(p, dtype={'day': str, 'symbol': str})
    pa = pd.read_csv(f'{D6}/pa6.csv', dtype={'day': str, 'symbol': str},
                     usecols=['day', 'symbol', 'entry_m', 'ctrl_entry_m'],
                     keep_default_na=False, na_values=[''])
    pa = pa[pa.ctrl_entry_m > pa.entry_m]
    rng = np.random.default_rng(SEED)
    out = []
    for k, g in pa.groupby(['day', 'symbol', 'entry_m'], sort=True):
        v = g.ctrl_entry_m.values
        pick = v if len(v) <= N_A else rng.choice(v, size=N_A, replace=False)
        for x in pick:
            out.append((k[0], k[1], int(k[2]), int(x)))
    d = pd.DataFrame(out, columns=['day', 'symbol', 'entry_m', 'ctrl_entry_m'])
    d.to_csv(p, index=False)
    return d


def main():
    sig, bk = load_sig()
    print(f'signals {len(sig)}  booked {len(bk)}', flush=True)
    pb = pd.read_csv(f'{D6}/pb6.csv', dtype={'day': str, 'symbol': str, 'ctrl': str},
                     usecols=['day', 'symbol', 'entry_m', 'ctrl'],
                     keep_default_na=False, na_values=[''])
    pd_ = pd.read_csv(f'{D6}/pd6.csv', dtype={'day': str, 'symbol': str, 'ctrl': str},
                      usecols=['day', 'symbol', 'entry_m', 'ctrl', 'ctrl_entry_m'],
                      keep_default_na=False, na_values=[''])
    pa = a_keys(bk)
    print(f'arm b {len(pb)}  arm d {len(pd_)}  arm a\' {len(pa)}', flush=True)
    RP = {(r.day, r.symbol, int(r.entry_m)): float(r.r_pct) for r in sig.itertuples()}
    SB = {d: g for d, g in sig.groupby('day')}
    PB = {d: g for d, g in pb.groupby('day')}
    PD = {d: g for d, g in pd_.groupby('day')}
    PA = {d: g for d, g in pa.groupby('day')}
    done = set(json.load(open(ST))['done']) if os.path.exists(ST) else set()
    days = [d for d in sorted(sig.day.unique()) if d not in done]
    print(f'{len(days)} sessions to walk ({len(done)} done)', flush=True)

    for nd, day in enumerate(days):
        ss = SB.get(day, sig.iloc[:0])
        gb = PB.get(day)
        gd = PD.get(day)
        ga = PA.get(day)
        syms = set(ss.symbol)
        for g in (gb, gd):
            if g is not None:
                syms |= set(g.ctrl.astype(str))
        bars = load_bars(day, sorted(syms))
        arr = {}
        for s_, gg in bars.items():
            a = c7.arrays(gg)
            if a is not None:
                arr[s_] = a
        buf = {k: [] for k in OUT}

        for r in ss.itertuples():
            A = arr.get(r.symbol)
            if A is None:
                continue
            o, h, l, c, v, m = A
            k = np.where(m == int(r.entry_m))[0]
            if not len(k):
                continue
            e = int(k[0])
            R = g8.geoms(o, h, l, c, m, e, float(r.stop))
            if R is None:
                continue
            atr = (float(r.atr14_pct) / 100.0 * float(o[e])
                   if r.atr14_pct == r.atr14_pct else None)
            S = g8.sub_arms(o, h, l, c, v, m, e, float(r.stop),
                            level=float(r.level), atr14=atr)
            row = [day, r.symbol, int(r.entry_m)]
            for g in G:
                row += [R[g][0], R[g][1], R[g][2]]
            for s2 in SUBS:
                t = S.get(s2)
                row += [np.nan, '', -1] if t is None else [t[0], t[1], t[2]]
            buf['sig'].append(row)

        if ga is not None:
            for r in ga.itertuples():
                A = arr.get(r.symbol)
                rp = RP.get((day, r.symbol, int(r.entry_m)))
                if A is None or rp is None:
                    continue
                o, h, l, c, v, m = A
                k = np.where(m == int(r.ctrl_entry_m))[0]
                if not len(k):
                    continue
                e = int(k[0])
                R = g8.geoms(o, h, l, c, m, e, float(o[e]) * (1.0 - rp / 100.0))
                if R is None:
                    continue
                row = [day, r.symbol, int(r.entry_m), int(r.ctrl_entry_m)]
                for g in G:
                    row += [R[g][0], R[g][1]]
                buf['a'].append(row)

        if gb is not None:
            for r in gb.itertuples():
                A = arr.get(str(r.ctrl))
                rp = RP.get((day, r.symbol, int(r.entry_m)))
                if A is None or rp is None:
                    continue
                o, h, l, c, v, m = A
                k = np.where(m == int(r.entry_m))[0]
                if not len(k):
                    continue
                e = int(k[0])
                R = g8.geoms(o, h, l, c, m, e, float(o[e]) * (1.0 - rp / 100.0))
                if R is None:
                    continue
                row = [day, r.symbol, int(r.entry_m), str(r.ctrl)]
                for g in G:
                    row += [R[g][0], R[g][1]]
                buf['b'].append(row)

        if gd is not None:
            for r in gd.itertuples():
                A = arr.get(str(r.ctrl))
                rp = RP.get((day, r.symbol, int(r.entry_m)))
                if A is None or rp is None:
                    continue
                o, h, l, c, v, m = A
                k = np.where(m == int(r.ctrl_entry_m))[0]
                if not len(k):
                    continue
                e = int(k[0])
                R = g8.geoms(o, h, l, c, m, e, float(o[e]) * (1.0 - rp / 100.0))
                if R is None:
                    continue
                row = [day, r.symbol, int(r.entry_m), str(r.ctrl), int(r.ctrl_entry_m)]
                for g in G:
                    row += [R[g][0], R[g][1]]
                buf['d'].append(row)

        for k, rows in buf.items():
            if rows:
                pd.DataFrame(rows, columns=HDR[k]).to_csv(
                    OUT[k], mode='a', header=not os.path.exists(OUT[k]), index=False)
        done.add(day)
        json.dump({'done': sorted(done)}, open(ST, 'w'))
        if nd % 10 == 0 or nd == len(days) - 1:
            print(f'  [{nd + 1}/{len(days)}] {day} syms {len(arr)} '
                  f'sig {len(buf["sig"])} a {len(buf["a"])} b {len(buf["b"])} d {len(buf["d"])}',
                  flush=True)
    print('WALK25 DONE', flush=True)


if __name__ == '__main__':
    main()
