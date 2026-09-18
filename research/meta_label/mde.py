#!/usr/bin/env python3
"""Minimum detectable effect of this study's book, per PLAN S1's phrasing rule.

Two forms, both at 80% power / two-sided 5% (z = 1.96 + 0.84 = 2.80):
  per pick  — the smallest mean R/pick difference the split's own pick count and
              R dispersion can resolve;
  per day   — the smallest mean daily-P&L difference the paired (same-day)
              comparison against M0 can resolve, using M0's own daily dispersion
              as the scale (the paired test is the one the cells are judged on).
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, f'{ROOT}/research/meta_label')
from analyze import load_book  # noqa: E402

Z = 2.802


def main() -> None:
    for arm in ('meas_cost', 'asis'):
        b = load_book(f'{ROOT}/research/meta_label/books/book_M0_{arm}.csv')
        print(f'--- M0 / {arm} ---')
        # TEST is SEALED (PREREG): not printed, not compared, not read.
        for sp in ('PRE', 'TRAIN', 'VAL'):
            g = b[b.split == sp]
            if not len(g):
                continue
            r = g.R.to_numpy(float)
            daily = g.groupby('date')._sized_pnl.sum()
            print(f'  {sp:5s} picks={len(g):4d} R sd={r.std(ddof=1):.3f} '
                  f'MDE/pick={Z * r.std(ddof=1) / np.sqrt(len(r)):.3f} R  | '
                  f'days={len(daily):3d} daily sd=${daily.std(ddof=1):,.0f} '
                  f'MDE/day=${Z * daily.std(ddof=1) / np.sqrt(len(daily)):,.0f} '
                  f'(= ${Z * daily.std(ddof=1) / np.sqrt(len(daily)) * len(daily):,.0f} '
                  f'over the split)')


if __name__ == '__main__':
    main()
