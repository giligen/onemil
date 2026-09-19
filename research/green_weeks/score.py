#!/usr/bin/env python3
"""green_weeks — the scorer.  Ranks every cell on WEEK SHAPE (PREREG §2).

TEST is SEALED (FREEZE.md): nothing about it is computed or printed without
`--reveal-test BOOK:CELL` for each cell being opened.

Usage:
  python3 research/green_weeks/score.py
  python3 research/green_weeks/score.py --reveal-test ORB:X0 ORB:E2b
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
M = 'research/fuckup_audit/M'

TEST_END = {'ORB': '2026-09-16', 'HOD': '2026-09-11', 'BF': '2026-08-31'}

ORB_CELLS = ['X0', 'E1a', 'E1b', 'E1c', 'E1d', 'E2a', 'E2b', 'E2c', 'E3',
             'E4a', 'E4b', 'E5']
ORB_M = ['X1', 'X2', 'X3', 'X4', 'X5']          # re-ranked, not re-run
HOD_CELLS = ['E0', 'E1a', 'E1b', 'E1c', 'E1d', 'E2a', 'E2b', 'E2c', 'E3',
             'E4a', 'E4b', 'E5']
BF_CELLS = ['E0_CACHE', 'E0', 'E1a', 'E1b', 'E1c', 'E1d', 'E2a', 'E2b',
            'E2c', 'E5']

LABEL = {
    'X0': 'E0 shipped', 'E0': 'E0 shipped', 'E0_CACHE': "E0 regen-7 own exits",
    'E1a': 'E1 target +0.50R (whole)', 'E1b': 'E1 target +0.75R (whole)',
    'E1c': 'E1 target +1.00R (whole)', 'E1d': 'E1 target +1.50R (whole)',
    'E2a': 'E2 50% @ +0.50R + BE', 'E2b': 'E2 50% @ +1.00R + BE',
    'E2c': 'E2 50% @ +1.50R + BE', 'E3': 'E3 50% @ +1R + 0.5R trail',
    'E4a': 'E4 time box (short)', 'E4b': 'E4 time box (long)',
    'E5': 'E5 breakeven-only @ +0.75R',
    'X1': 'M-X1 10-min TS < +0.25R', 'X2': 'M-X2 5-min TS < 0R',
    'X3': 'M-X3 no lock', 'X4': 'M-X4 no lock + 10-min TS',
    'X5': 'M-X5 no lock + TS + BE@1R',
}

BASE = {'ORB': 'X0', 'HOD': 'E0', 'BF': 'E0'}


def load(book, cell):
    if book == 'ORB':
        p = f'{M}/book_{cell}_n8.csv' if cell in ORB_M else f'{D}/orb_book_{cell}_n8.csv'
        if not os.path.exists(p):
            return None, None
        d = pd.read_csv(p, keep_default_na=False, na_values=[''],
                        dtype={'symbol': str})
        d['day'] = d['date'].astype(str).str[:10]
        return d, '_sized_pnl'
    if book == 'HOD':
        p = f'{D}/hod_book_{cell}.csv'
        if not os.path.exists(p):
            return None, None
        return pd.read_csv(p, keep_default_na=False, dtype={'symbol': str}), 'rr'
    p = f'{D}/bf_runs/{cell}.csv'
    if not os.path.exists(p):
        return None, None
    d = pd.read_csv(p, keep_default_na=False, na_values=[''],
                    dtype={'symbol': str})
    d['day'] = d['date'].astype(str).str[:10]
    d['pnl'] = pd.to_numeric(d['pnl'], errors='coerce').fillna(0.0)
    return d, 'pnl'


HDR = (f"{'cell':9s} {'green%':>7s} {'flat%':>6s} {'red%':>6s} {'redstk':>6s} "
       f"{'worstwk':>9s} {'mo_grn%':>7s} {'mdd':>9s} {'tr/wk':>6s} "
       f"{'WR%':>5s} {'total':>10s} {'top5%':>6s} {'disc':>5s}")


def fmt(r, disc):
    return (f"{r['cell']:9s} {r['green_pct']:7.1f} {r['flat_pct']:6.1f} "
            f"{r['red_pct']:6.1f} {int(r['red_streak']):6d} "
            f"{r['worst_wk']:9.1f} {r['mo_green_pct']:7.1f} {r['mdd']:9.1f} "
            f"{r['tr_per_wk']:6.2f} {r['wr']:5.1f} {r['pnl']:10.1f} "
            f"{100 * r['top5_share']:6.1f} {disc:5d}")


def main():
    argv = sys.argv[1:]
    reveal = set()
    if '--reveal-test' in argv:
        i = argv.index('--reveal-test')
        reveal = set(argv[i + 1:])

    rows, wk = [], {}
    books = {'ORB': ORB_CELLS + ORB_M, 'HOD': HOD_CELLS, 'BF': BF_CELLS}
    for book, cells in books.items():
        for cell in cells:
            d, val = load(book, cell)
            if d is None:
                print(f"(missing: {book}:{cell})")
                continue
            r = W.score_all(d, TEST_END[book], value=val,
                            reveal_test=f'{book}:{cell}' in reveal)
            for split, m in r.items():
                wk[(book, cell, split)] = m.pop('_wk')
                rows.append({'book': book, 'cell': cell,
                             'label': LABEL.get(cell, cell), **m})
    df = pd.DataFrame(rows)
    df.to_csv(f'{D}/cells.csv', index=False)

    for book in books:
        b = df[df.book == book]
        if not len(b):
            continue
        base = BASE[book]
        unit = 'R' if book == 'HOD' else '$'
        for split in ('TRAIN', 'VAL', 'TEST'):
            sub = b[b.split == split]
            if not len(sub):
                continue
            sub = sub.sort_values('green_pct', ascending=False)
            print(f"\n### {book} · {split} · {int(sub.n_weeks.iloc[0])} weeks · "
                  f"totals in {unit} — RANKED ON GREEN WEEKS")
            print(HDR)
            for _, r in sub.iterrows():
                k = (book, base, split)
                dsc = (W.discordant(wk[k], wk[(book, r.cell, split)])[0]
                       if k in wk else -1)
                print(fmt(r, dsc))
        # the pre-committed ranking: min(green% TRAIN, green% VAL)
        piv = b[b.split.isin(('TRAIN', 'VAL'))].pivot(
            index='cell', columns='split', values='green_pct')
        if {'TRAIN', 'VAL'} <= set(piv.columns):
            piv['min_green'] = piv[['TRAIN', 'VAL']].min(axis=1)
            print(f"\n{book} — PREREG §2 ranking = min(green% TRAIN, green% VAL):")
            for cell, r in piv.sort_values('min_green', ascending=False).iterrows():
                mark = ' <= shipped' if cell == base else ''
                print(f"   {cell:9s} TRAIN {r['TRAIN']:5.1f}  VAL {r['VAL']:5.1f}"
                      f"   min {r['min_green']:5.1f}{mark}")
    print(f"\nwrote {D}/cells.csv")
    print("disc = weeks green in exactly one of {cell, shipped} (PREREG §7: "
          "< 8 on TRAIN / < 5 on VAL = NOT RESOLVED)")


if __name__ == '__main__':
    main()
