#!/usr/bin/env python3
"""green_weeks — the two diagnostics the cell tables cannot show.

1. ORB flat-week decomposition: a flat week is either a week the book made no
   PICK at all, or a week whose only picks were modelled non-fills (booking $0).
   Exit design cannot touch either — this is why ORB's flat share is identical
   in all 17 cells.
2. The thesis test in one number: win rate vs green-week share across cells.
   If the thesis held, the cells with the highest WR would have the most green
   weeks.  Reported as a rank correlation per book per split.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, f'{ROOT}/research/green_weeks')
import weekshape as W                                       # noqa: E402

D = 'research/green_weeks'


def orb_flat_decomposition():
    d = pd.read_csv(f'{D}/orb_book_X0_n8.csv', keep_default_na=False,
                    na_values=[''], dtype={'symbol': str})
    d['day'] = d['date'].astype(str).str[:10]
    d['wk'] = pd.to_datetime(d['day']).dt.to_period('W-FRI')
    print("\n## ORB flat weeks — what they actually are (shipped exit, N=8)")
    for split, (a, b) in (('TRAIN', ('2025-01-01', '2025-12-31')),
                          ('VAL', ('2026-01-01', '2026-05-31'))):
        weeks = W.week_index(a, b)
        s = d[(d.day >= a) & (d.day <= b)]
        pnl = s.groupby('wk')['_sized_pnl'].sum().reindex(weeks, fill_value=0.0)
        fills = s[s.entered == 1].groupby('wk').size().reindex(weeks, fill_value=0)
        picks = s.groupby('wk').size().reindex(weeks, fill_value=0)
        flat = pnl == 0
        print(f"  {split}: {len(weeks)} weeks · flat {int(flat.sum())} "
              f"({100 * flat.mean():.1f}%) = "
              f"{int((flat & (picks == 0)).sum())} weeks with NO pick + "
              f"{int((flat & (picks > 0) & (fills == 0)).sum())} weeks whose only "
              f"picks were modelled non-fills + "
              f"{int((flat & (fills > 0)).sum())} weeks that filled and netted "
              f"exactly zero")


def thesis_rank_test():
    c = pd.read_csv(f'{D}/cells.csv')
    print("\n## The thesis, as one number: does a higher win rate buy green weeks?")
    print("   Spearman rank correlation across cells, WR vs green-week share.")
    print("   Thesis predicts POSITIVE and large.  (n = cells in the book.)")
    for book in ('ORB', 'HOD', 'BF'):
        for split in ('TRAIN', 'VAL'):
            s = c[(c.book == book) & (c.split == split)]
            s = s[np.isfinite(s.wr) & np.isfinite(s.green_pct)]
            if len(s) < 4:
                continue
            rho = s['wr'].rank().corr(s['green_pct'].rank())
            rho_p = s['pnl'].rank().corr(s['green_pct'].rank())
            print(f"   {book:4s} {split:5s} n={len(s):2d}  "
                  f"rho(WR, green%) = {rho:+.2f}   "
                  f"rho(total P&L, green%) = {rho_p:+.2f}")


if __name__ == '__main__':
    orb_flat_decomposition()
    thesis_rank_test()
