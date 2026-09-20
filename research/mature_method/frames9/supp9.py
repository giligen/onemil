#!/usr/bin/env python3
"""frames9 — the two SUPPLEMENTARY diagnostics (declared as diagnostics, counted, not cells).

A  the ABSOLUTE book behind F29's largest split.  A margin is a difference; the owner is paid an
   absolute.  For every level of the split that owns the margin, print the signal's own gross, the
   matched control's gross, the pond the level fishes in, and the BOOK (`run_book(12,4)`, $100 risk)
   with green weeks against a count-matched null.  This is a POST-HOC cut of 28 levels and is
   labelled as such — it is a hypothesis for the next pass's PREREG, not a result.

B  WHY pass 7's ponds read +0.070 / +0.053 and the same ponds read negative at HOD's clock.  The
   pond bound is a RATIO: (what the name does between entry and the exit) / R.  Bucket the pond
   bound by the stop width R and by the clock, on the arm-u population this pass walked.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c9                                                              # noqa: E402
import g8                                                              # noqa: E402
from common4 import book_ranked, clustered_t                           # noqa: E402
from common5 import S                                                  # noqa: E402
from s9 import net_of, report, mkbook                                  # noqa: E402
from f29 import margin_frame                                           # noqa: E402

D8, D9 = c9.D8, c9.D9
KEYS = ['day', 'symbol', 'entry_m']


def supp_a():
    print('\n================ SUPP A — the ABSOLUTE book behind F29 (POST-HOC, 28 levels) '
          '================', flush=True)
    d = margin_frame()
    W = pd.read_csv(f'{D8}/w_sig.csv', dtype={'day': str, 'symbol': str},
                    keep_default_na=False, na_values=[''])
    d = d.drop(columns=[c for c in ('rr_G3', 'rr_X0') if c in d.columns]).merge(
        W, on=KEYS, how='inner')
    rows = []
    for lv, g in d.groupby(d.cls.fillna('unknown')):
        if len(g) < 30:
            continue
        for geom in ('G3', 'X0'):
            for sp in c9.SPLITS:
                x = g[g.split == sp]
                print(f'  {lv:8s} {geom} {sp:5s} n={len(x):4d}  signal gross={x[f"rr_{geom}"].mean():+.4f}'
                      f'  matched control={x[f"cb_{geom}"].mean():+.4f}'
                      f'  margin={x[f"m_{geom}"].mean():+.4f}', flush=True)
            b = mkbook(g, geom)
            bk = book_ranked(b, 12, 4)
            bk['pnl'] = bk.net * 100.0
            report(f'{lv}/{geom}', bk, rows, note='POST-HOC F29 cut')
    pd.DataFrame(rows).to_csv(f'{D9}/supp_a_books.csv', index=False)


def supp_b():
    print('\n================ SUPP B — the pond bound is a RATIO: bucket it by stop width and '
          'clock ================', flush=True)
    s = pd.read_csv(f'{D9}/sig9.csv', dtype={'day': str, 'symbol': str},
                    keep_default_na=False, na_values=[''],
                    usecols=KEYS + ['r_pct', 'split'])
    U = pd.read_csv(f'{D9}/w9_u.csv', dtype={'day': str, 'symbol': str, 'ctrl': str,
                                             'pond': str}, keep_default_na=False, na_values=[''])
    U = U.merge(s, on=KEYS, how='inner')
    U['half'] = np.where(U.day < '2025-07-01', 'H1', 'H2')
    print(f'  arm u rows {len(U)} | r_pct median {U.r_pct.median():.2f} % '
          f'| entry minute median {U.entry_m.median():.0f}', flush=True)
    for name, col, edges, labs in (
            ('stop width r_pct', 'r_pct', [0, 1.5, 3.0, 6.0, 1e9],
             ['<1.5%', '1.5-3%', '3-6%', '>=6%']),
            ('clock', 'entry_m', [577, 630, 690, 780, 842],
             ['09:37-10:30', '10:30-11:30', '11:30-13:00', '13:00-14:01'])):
        U['_b'] = pd.cut(U[col], edges, labels=labs, right=False)
        print(f'\n  -- pond bound by {name} (arm u, the pond\'s random name at the HOD clock) --')
        print('  | bucket | n | bound G3 | H1 | H2 | VAL | bound X0 | eod % (G3) | stop % (G3) |')
        print('  |---|---|---|---|---|---|---|---|---|')
        for lv, g in U.groupby('_b', observed=True):
            if not len(g):
                continue
            w = g.why_G3.astype(str)
            print(f'  | {lv} | {len(g)} | {g.rr_G3.mean():+.4f} | '
                  f'{g[g.half=="H1"].rr_G3.mean():+.4f} | '
                  f'{g[(g.half=="H2")&(g.split=="TRAIN")].rr_G3.mean():+.4f} | '
                  f'{g[g.split=="VAL"].rr_G3.mean():+.4f} | {g.rr_X0.mean():+.4f} | '
                  f'{100*(w=="eod").mean():.0f} % | {100*(w=="stop").mean():.0f} % |')
    # the price move behind the ratio, so the reading is not an artefact of the R unit
    U['move_pct'] = U.rr_G3 * U.r_pct
    print(f'\n  the same arm u expressed in PRICE: mean move entry->exit = '
          f'{U.move_pct.mean():+.3f} % (median stop {U.r_pct.median():.2f} %), '
          f'TRAIN {U[U.split=="TRAIN"].move_pct.mean():+.3f} % / '
          f'VAL {U[U.split=="VAL"].move_pct.mean():+.3f} %', flush=True)


if __name__ == '__main__':
    supp_a()
    supp_b()
