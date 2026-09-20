#!/usr/bin/env python3
"""frames15 stage 1 — the B2 reproduction gate, then the 7,027 pre-book signals to CSV.

Split out of armA so the (memory-heavy) B2 machinery and the (memory-heavy) daily panel are never
resident in the same process — the 3 GB `ulimit -v` rail.
"""
import sys

import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/frames15')
from common15 import D, S, SPLITS, base_book, load_breaks4   # noqa: E402

REF = {'TRAIN': (1622, -0.039, -17346.0), 'VAL': (706, 0.083, 893.0)}


def main():
    br = load_breaks4(verbose=False)
    b0, s = base_book(br, verbose=False)
    print('== reproduction gate ==', flush=True)
    for sp in SPLITS:
        w = S.week_stats(b0, sp)
        n, g, t = REF[sp]
        assert w['n'] == n and abs(w['total'] - t) < 1.0 and abs(w['gross'] - g) < 5e-4, \
            f'REPRO FAIL {sp}: {w["n"]}/{w["gross"]}/{w["total"]}'
        print(f'  B2 {sp:5s} n={w["n"]} gross={w["gross"]:+.3f} net={w["net"]:+.3f} '
              f'green={w["green"]:.1f}% $={w["total"]:+,.0f}  MATCH', flush=True)
    key = ['day', 'symbol', 'entry_m']
    bk = b0[key].assign(booked=1)
    s = s.merge(bk, on=key, how='left')
    s['booked'] = s.booked.fillna(0).astype(int)
    s.to_csv(f'{D}/sig15.csv', index=False)
    print(f'  sig15.csv {len(s)} signals, {int(s.booked.sum())} booked (expect 2328)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
