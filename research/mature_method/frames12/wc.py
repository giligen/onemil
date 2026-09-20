#!/usr/bin/env python3
"""A generic PLACEBO control walker — used by F39 (and re-usable by any later frame).

Input: a control-key CSV with columns `day, arm, ctrl, ctrl_m, symbol, entry_m, r_pct` — the arm is
`CB` (a matched non-signal name at the SIGNAL'S own minute) or `CA` (the SAME name-day at a minute
strictly AFTER the signal, the pass-6 causality rule).  Every control is priced on the SIGNAL'S own
R geometry: stop at `ctrl_open x (1 - r_pct/100)`.

Output: the same keys plus `rr/why/em` for the +2 R bracket, the bare stop, the static lock, the
MOC and the next-open exits.  Resumable per session; every store READ-ONLY.

  python3 wc.py KEYS.csv OUT.csv STATE.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c12 import D12, bars_arrays, load_bars, load_panel, price_exit    # noqa: E402

KINDS = ('X2R', 'XBR', 'XLK', 'XMO', 'XNO')
WALK = {'X2R': 'bracket', 'XBR': 'bare', 'XLK': 'lock', 'XMO': 'moc', 'XNO': 'next'}
HDR = (['day', 'arm', 'ctrl', 'ctrl_m', 'symbol', 'entry_m', 'r_pct', 'next_open', 'stop']
       + [f'{k}_{f}' for k in KINDS for f in ('rr', 'why', 'em')])


def main(keys, out, st) -> int:
    c = pd.read_csv(keys, dtype={'day': str, 'symbol': str, 'ctrl': str})
    u = load_panel()[['day', 'symbol', 'close', 'next_open_d']]
    DC = {(d, s): (cl, n) for d, s, cl, n in u.itertuples(index=False)}
    done = set(json.load(open(st))['days']) if os.path.exists(st) else set()
    if not os.path.exists(out):
        with open(out, 'w') as f:
            f.write(','.join(HDR) + '\n')
    days = [d for d in sorted(c.day.unique()) if d not in done]
    print(f'  {len(days)} sessions to walk ({len(done)} done)', flush=True)
    G = {d: g for d, g in c.groupby('day')}
    nrow = 0
    for i, day in enumerate(days):
        g = G[day]
        bars = load_bars(day, sorted(g.ctrl.astype(str).unique()))
        A = {}
        for s_, b_ in bars.items():
            x = bars_arrays(b_)
            if x is not None:
                A[s_] = (x[0], x[1], x[2], x[3], x[5])
        rows = []
        for r in g.itertuples():
            a = A.get(r.ctrl)
            if a is None:
                continue
            o, h, l, cl_, m = a
            j = np.flatnonzero(m == int(r.ctrl_m))
            if not len(j) or int(j[0]) + 1 >= len(o):
                continue
            e = int(j[0]); E = float(o[e])
            if not (E > 0) or not (r.r_pct == r.r_pct):
                continue
            stop = E * (1.0 - float(r.r_pct) / 100.0)
            cp, no = DC.get((day, r.ctrl), (np.nan, np.nan))
            row = [day, r.arm, r.ctrl, int(r.ctrl_m), r.symbol, int(r.entry_m), float(r.r_pct),
                   E, stop]
            for k in KINDS:
                em, rr, why = price_exit(WALK[k], o, h, l, cl_, m, e, stop, close_px=cp,
                                         next_open_px=no)
                row += [rr, why, em]
            rows.append(row)
        if rows:
            pd.DataFrame(rows, columns=HDR).to_csv(out, mode='a', header=False, index=False)
            nrow += len(rows)
        done.add(day)
        json.dump({'days': sorted(done)}, open(st, 'w'))
        if i % 25 == 0 or i == len(days) - 1:
            print(f'  {i+1}/{len(days)} {day} rows {nrow:,}', flush=True)
    print(f'  DONE — {nrow:,} rows -> {out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
