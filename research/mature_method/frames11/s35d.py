#!/usr/bin/env python3
"""F35 SUPPLEMENT stage 2 — the one earned admission cell: low `rv_profile` on the UNCAPPED book.

`rv_profile` is the ONLY field of eleven that separates the uncapped (G3) book's real tail on BOTH
the net and the gross label, on TRAIN and on VAL. Per the PREREG amendment the admission cell is
the parameter-free MEDIAN cut on the TRAIN book, keeping the tail's side (LOW rv), re-booked from
the full admitted signal set with `run_book(12, 4)` under the geometry's own exit minutes.

Two further declared reads, both counted: the same cut at the TERCILE (a robustness read of the cut
itself, NOT a search — one alternative, stated), and the cell's dollar path week by week.

  python3 s35d.py           # writes admit35c.csv
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
for _p in ('frames11', 'frames10', 'frames9', 'frames8', 'frames7', 'hod_frames6',
           'hod_frames5', 'hod_frames4', 'hod_frames3', 'hod_frames2'):
    sys.path.insert(0, f'{ROOT}/research/mature_method/{_p}')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')

import s35                                                        # noqa: E402
import s35b                                                       # noqa: E402
from common6 import base_book, mde                                # noqa: E402
from common4 import book_ranked                                   # noqa: E402
from common3 import clustered_t                                   # noqa: E402

D8 = f'{ROOT}/research/mature_method/frames8'
D11 = f'{ROOT}/research/mature_method/frames11'
SPLITS = ('TRAIN', 'VAL')
RISK = 100.0


def g3_signals(sig):
    w = pd.read_csv(f'{D8}/w_sig.csv', dtype={'day': str, 'symbol': str},
                    usecols=['day', 'symbol', 'entry_m', 'rr_G3', 'why_G3', 'xm_G3'])
    ck = sig[['day', 'symbol', 'entry_m']].copy()
    ck['cost'] = sig.rr - sig.net
    s = sig.merge(w, on=['day', 'symbol', 'entry_m'], how='inner').merge(
        ck, on=['day', 'symbol', 'entry_m'], how='left')
    s['rr'] = s.rr_G3
    s['why'] = s.why_G3
    s['exit_m'] = s.xm_G3
    s = s[s.rr.notna()].copy()
    s['net'] = s.rr - s.cost
    s['pnl'] = s.net * RISK
    return s


def score(bk, tag, rng, out):
    for sp in SPLITS:
        r = s35b.week_row(bk, sp, tag)
        r['null_green_p95'] = s35b.green_null(bk, sp, rng)
        r['ex_top5'] = s35b.ex_top5(bk, sp)
        d = bk[bk.split == sp]
        r['mde'] = mde(d) if len(d) > 5 else np.nan
        out.append(r)
        print(f'  {tag:34s} {sp:5s} n={r["n"]:5d} /wk={r["per_wk"]:5.1f} gross={r["gross"]:+.3f} '
              f'net={r["net"]:+.3f} green={r["green"]:5.1f} (null p95 {r["null_green_p95"]:5.1f}) '
              f'${r["total"]:+,.0f} t={r["t"]:+.2f} ex5={r["ex_top5"]:+.3f} MDE={r["mde"]:.3f}',
              flush=True)
    for lab, q in (('H1', bk[(bk.split == 'TRAIN') & (bk.day < '2025-07-01')]),
                   ('H2', bk[(bk.split == 'TRAIN') & (bk.day >= '2025-07-01')])):
        print(f'    {lab}: n={len(q):4d} net={q.net.mean():+.3f} ${q.pnl.sum():+,.0f}', flush=True)


def main() -> int:
    rng = np.random.default_rng(20260924)
    print('F35 SUPPLEMENT stage 2 — the earned admission cell (low rv_profile, UNCAPPED G3)',
          flush=True)
    b0, sig = base_book(verbose=False)
    s35.repro_gate(b0[b0.split.isin(SPLITS)])
    s = g3_signals(sig)
    out = []

    base = book_ranked(s, 12, 4)
    base = base[base.split.isin(SPLITS)]
    score(base, 'G3 baseline (no admission rule)', rng, out)

    tr = base[base.split == 'TRAIN']
    for tag, cut in (('A3 rv <= TRAIN median', float(tr.rv_profile.median())),
                     ('A4 rv <= TRAIN tercile', float(tr.rv_profile.quantile(1 / 3)))):
        keep = s[s.rv_profile <= cut]
        bk = book_ranked(keep, 12, 4)
        bk = bk[bk.split.isin(SPLITS)]
        score(bk, f'{tag} ({cut:.2f})', rng, out)
        # the weekly dollar path for the last two quarters of each split (RUNBOOK step 8)
        for sp in SPLITS:
            d = bk[bk.split == sp]
            wk = d.groupby('wk').pnl.sum().sort_index()
            print(f'    {sp} last 8 weeks $: '
                  f'{[round(float(x)) for x in wk.tail(8).values]}', flush=True)

    pd.DataFrame(out).to_csv(f'{D11}/admit35c.csv', index=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
