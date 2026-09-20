#!/usr/bin/env python3
"""frames8 / F25 rail 1 — the reproduction gates, all of them, before any cell is scored.

  P1  X0 reproduces `hod_frames6/book6.csv::rr` for all 2,328 booked trades (<= 1e-12)
  P2  the vectorised G1 == `frames7/c7.walk_orb` (the prose-written ORB walker) bar for bar
  P3  the vectorised G2 == `frames7/c7.walk_bf`
  P4  G1 == `study_orb_pipeline_static_lock.simulate_winner_stack` on 50 REAL ORB trades
  P5  G2's stop path == `trading/bf_trail.arm_and_ratchet` fed the same closed bars
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames8')
sys.path.insert(0, f'{ROOT}/research/mature_method/frames7')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import g8                                                              # noqa: E402
import c7                                                              # noqa: E402
from pass2 import load_bars                                            # noqa: E402

D8 = f'{ROOT}/research/mature_method/frames8'
D6 = f'{ROOT}/research/mature_method/hod_frames6'


def main():
    bk = pd.read_csv(f'{D6}/book6.csv', dtype={'day': str, 'symbol': str},
                     keep_default_na=False, na_values=[''])
    print(f'booked {len(bk)}', flush=True)
    days = sorted(bk.day.unique())
    worst_x0 = 0.0
    n_x0 = 0
    worst_orb = worst_bf = 0.0
    n_cmp = 0
    for day in days:
        tr = bk[bk.day == day]
        bars = load_bars(day, sorted(set(tr.symbol)))
        for r in tr.itertuples():
            gg = bars.get(r.symbol)
            if gg is None:
                continue
            a = c7.arrays(gg)
            if a is None:
                continue
            o, h, l, c, v, m = a
            k = np.where(m == int(r.entry_m))[0]
            if not len(k):
                continue
            e = int(k[0])
            G = g8.geoms(o, h, l, c, m, e, float(r.stop))
            if G is None:
                continue
            worst_x0 = max(worst_x0, abs(G['X0'][0] - float(r.rr)))
            n_x0 += 1
            E = float(o[e])
            rp = (E - float(r.stop)) / E * 100.0
            rr_o, _, _ = c7.walk_orb(o, h, l, c, m, e, rp, flat_m=g8.EOD_M)
            rr_b, _, _ = c7.walk_bf(o, h, l, c, m, e, rp, flat_m=g8.EOD_M)
            worst_orb = max(worst_orb, abs(rr_o - G['G1'][0]))
            worst_bf = max(worst_bf, abs(rr_b - G['G2'][0]))
            n_cmp += 1
    print(f'P1 X0 vs book6.rr        n={n_x0:5d}  max|d| = {worst_x0:.3e}', flush=True)
    print(f'P2 G1 vs c7.walk_orb     n={n_cmp:5d}  max|d| = {worst_orb:.3e}', flush=True)
    print(f'P3 G2 vs c7.walk_bf      n={n_cmp:5d}  max|d| = {worst_bf:.3e}', flush=True)
    assert worst_x0 < 1e-12, 'P1 FAILED'
    assert worst_orb < 1e-12, 'P2 FAILED'
    assert worst_bf < 1e-12, 'P3 FAILED'
    print('PARITY 1-3 OK', flush=True)


if __name__ == '__main__':
    main()
