#!/usr/bin/env python3
"""F34 stage 1 — THE FLOOR WALK.

Prices the UNCONDITIONAL long bracket on pass 6's 288,174 detector-free arm-d controls with a
**FIXED stop at s % of the entry price**, s in {2, 3, 4}, target +2R = +2s %, flat at 15:55.

This is NOT F22/F25's object: those walked the control at the BOOKED trade's own `r_pct`, so the
stop width carried the signal's information. Here the stop is a pure function of price, so every
cell is a property of the UNIVERSE and the clock alone.

Exit convention is `hod_frames6.common6.walk_from` verbatim, and the reproduction gate asserts that:
handed the booked stop, this walker must reproduce `book6.rr`.

Resumable per session. Every store READ-ONLY. TEST never loaded (day < 2026-06-01).

  python3 w34.py            # appends to w34.csv, state in w34_state.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
for _p in ('frames11', 'hod_frames6', 'hod_frames5', 'hod_frames4', 'hod_frames3', 'hod_frames2'):
    sys.path.insert(0, f'{ROOT}/research/mature_method/{_p}')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')

from common6 import walk_from, bars_arrays, EOD_M, SLIP           # noqa: E402
from pass2 import load_bars                                       # noqa: E402

D6 = f'{ROOT}/research/mature_method/hod_frames6'
D11 = f'{ROOT}/research/mature_method/frames11'
OUT = f'{D11}/w34.csv'
ST = f'{D11}/w34_state.json'
TEST_FROM = '2026-06-01'
STOPS = (0.02, 0.03, 0.04)
HDR = ['day', 'ctrl', 'ctrl_entry_m', 'entry'] + [f'{k}_{int(s*100)}' for s in STOPS
                                                  for k in ('rr', 'why')]


def repro_gate():
    """The walker must reproduce book6.rr when handed the booked stop (pass 6 / pass 8's gate)."""
    bk = pd.read_csv(f'{D6}/book6.csv', dtype={'day': str, 'symbol': str})
    bk = bk[bk.split.isin(('TRAIN', 'VAL'))]
    g = bk.sample(120, random_state=11)
    worst = 0.0
    for day, gg in g.groupby('day'):
        bars = load_bars(day, sorted(gg.symbol.unique()))
        for r in gg.itertuples():
            a = bars.get(r.symbol)
            if a is None:
                continue
            arr = bars_arrays(a)
            if arr is None:
                continue
            o, h, l, c, v, m = arr
            idx = np.flatnonzero(m == int(r.entry_m))
            if not len(idx):
                continue
            _, _, _, rr = walk_from(o, h, l, c, m, int(idx[0]), float(r.stop))
            if rr == rr:
                worst = max(worst, abs(rr - float(r.rr)))
    print(f'  REPRO walk_from vs book6.rr on 120 booked trades: max |diff| = {worst:.2e}',
          flush=True)
    assert worst < 1e-9, f'walker does not reproduce book6.rr (max diff {worst})'


def walk_fixed(o, h, l, c, m, e, s):
    """The bracket with a stop `s` BELOW the entry as a fraction of price. Same convention."""
    E = float(o[e])
    if not (E > 0):
        return np.nan, ''
    _, _, why, rr = walk_from(o, h, l, c, m, e, E * (1.0 - s))
    return rr, why


def main() -> int:
    repro_gate()
    d = pd.read_csv(f'{D6}/pd6.csv', dtype={'day': str, 'symbol': str, 'ctrl': str})
    d = d[d.day < TEST_FROM]
    print(f'  arm-d controls: {len(d):,} over {d.day.nunique()} sessions (TEST cut off)', flush=True)
    done = set()
    if os.path.exists(ST):
        done = set(json.load(open(ST))['days'])
    if not os.path.exists(OUT):
        with open(OUT, 'w') as f:
            f.write(','.join(HDR) + '\n')
    days = sorted(set(d.day) - done)
    print(f'  {len(days)} sessions to walk ({len(done)} already done)', flush=True)
    nrow = 0
    for i, day in enumerate(days):
        gg = d[d.day == day]
        syms = sorted(gg.ctrl.unique())
        bars = load_bars(day, syms)
        arrs = {}
        for s_, a in bars.items():
            x = bars_arrays(a)
            if x is not None:
                arrs[s_] = x
        rows = []
        for r in gg.itertuples():
            a = arrs.get(r.ctrl)
            if a is None:
                continue
            o, h, l, c, v, m = a
            idx = np.flatnonzero(m == int(r.ctrl_entry_m))
            if not len(idx):
                continue
            e = int(idx[0])
            if e + 1 >= len(o) or int(m[e]) >= EOD_M:
                continue
            row = [day, r.ctrl, int(r.ctrl_entry_m), float(o[e])]
            for s in STOPS:
                rr, why = walk_fixed(o, h, l, c, m, e, s)
                row += [rr, why]
            rows.append(row)
        if rows:
            pd.DataFrame(rows, columns=HDR).to_csv(OUT, mode='a', header=False, index=False)
            nrow += len(rows)
        done.add(day)
        json.dump({'days': sorted(done)}, open(ST, 'w'))
        if i % 20 == 0 or i == len(days) - 1:
            print(f'  {i+1}/{len(days)} {day} rows so far {nrow:,}', flush=True)
    print(f'  DONE — {nrow:,} rows written to w34.csv', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
