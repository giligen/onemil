#!/usr/bin/env python3
"""F37 stage 2 — SCORE the ten declared cells: the 5-minute and 15-minute HOD break.

  python3 s37.py > s37.log
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c12 import (D12, S, SPLITS, Sheet12, attach_cost12, book_ranked, build_cost_model,  # noqa: E402
                 cascade, clustered_t, repro_gate)
from common6 import base_book                                            # noqa: E402

CELLS = [('M1 5m-k3-X2R', 5, 3, 'X2R', True), ('M2 5m-k5-X2R', 5, 5, 'X2R', True),
         ('M3 15m-k3-X2R', 15, 3, 'X2R', True), ('M4 15m-k5-X2R', 15, 5, 'X2R', True),
         ('M5 5m-k5-XBR', 5, 5, 'XBR', False), ('M6 15m-k3-XBR', 15, 3, 'XBR', False),
         ('M7 5m-k5-XLK', 5, 5, 'XLK', False), ('M8 15m-k3-XLK', 15, 3, 'XLK', False),
         ('M9 5m-k3-XBR', 5, 3, 'XBR', False)]
U34 = {'TRAIN': -0.389, 'VAL': -0.360}


def prep(d, tag):
    x = d.copy()
    x['rr'] = x[f'{tag}_rr']; x['why'] = x[f'{tag}_why']; x['exit_m'] = x[f'{tag}_em']
    x = x[x.rr.notna() & (x.why != '')]
    return attach_cost12(x)


def main() -> int:
    build_cost_model()
    w = pd.read_csv(f'{D12}/w37.csv', dtype={'day': str, 'symbol': str, 'kind': str,
                                             'key_sym': str}, keep_default_na=False,
                    na_values=[''])
    w['split'] = S.split_of(w.day.values)
    w = w[w.split.isin(SPLITS)]
    w['wk'] = pd.to_datetime(w.day).dt.to_period('W-FRI').astype(str)
    sig = w[w.kind == 'S']
    print('\n== F37 — signal FREQUENCY per bar size, checked BEFORE any P&L (PREREG §4.3) ==')
    print('| bar | k | raw signals TRAIN | /wk | VAL | /wk |')
    print('|---|---|---|---|---|---|')
    for size in (5, 15):
        for k in (3, 5):
            z = sig[(sig.bar == size) & (sig.k == k)]
            print(f'| {size}m | {k} | {int((z.split=="TRAIN").sum())} | '
                  f'{int((z.split=="TRAIN").sum())/S.NW["TRAIN"]:.1f} | '
                  f'{int((z.split=="VAL").sum())} | '
                  f'{int((z.split=="VAL").sum())/S.NW["VAL"]:.1f} |', flush=True)

    sh = Sheet12()
    books = {}
    print('\n== B0 — the 1-minute baseline (B2), re-printed for the comparison ==\n', flush=True)
    b0, _ = base_book(verbose=False)
    b0 = b0.copy()
    b0['cost_R'] = b0.rr - b0.net
    b0['gross_pct'] = b0.rr * b0.r_pct
    b0['net_pct'] = b0.net * b0.r_pct
    b0['cost_pct'] = b0.cost_R * b0.r_pct
    sh.show('B0 1m-k5-X2R (B2 reference)', b0, capped=True)

    print('\n== F37 M1-M9 — the coarse-bar cells ==\n', flush=True)
    for nm, size, k, tag, capped in CELLS:
        x = prep(sig[(sig.bar == size) & (sig.k == k)], tag)
        x = cascade(x)
        if not len(x):
            print(f'| {nm:<35s} | NO SIGNALS after the cascade', flush=True)
            continue
        x = x.sort_values(['day', 'entry_m', 'symbol'], kind='mergesort').reset_index(drop=True)
        b = book_ranked(x, 12, 4)
        books[nm] = b
        sh.show(nm, b, capped=capped)
    sh.dump(f'{D12}/cells37.csv')
    sh.nulls(f'{D12}/nulls37.csv')

    print('\n== the four-row PLACEBO DECOMPOSITION (net R | net % of entry price) ==')
    print('| cell | split | universe bound % | matched non-signal | same name-day later min | '
          'THE SIGNAL | n |')
    print('|---|---|---|---|---|---|---|')
    for nm, size, k, tag, capped in CELLS:
        b = books.get(nm)
        if b is None or not len(b):
            continue
        for sp in SPLITS:
            d = b[b.split == sp]
            if not len(d):
                continue
            keys = set(zip(d.day, d.symbol, d.entry_m))
            out = []
            for arm in ('CB', 'CA'):
                a = w[(w.kind == arm) & (w.split == sp)]
                a = a[a.set_index(['day', 'key_sym', 'key_m']).index.isin(keys)]
                a = prep(a, tag if tag != 'XLK' else 'X2R')
                out.append((a.net.mean() if len(a) else np.nan,
                            a.net_pct.mean() if len(a) else np.nan, len(a)))
            print(f'| {nm} | {sp} | {U34[sp]:+.3f} | {out[0][0]:+.3f} / {out[0][1]:+.3f} '
                  f'(n {out[0][2]}) | {out[1][0]:+.3f} / {out[1][1]:+.3f} (n {out[1][2]}) | '
                  f'{d.net.mean():+.3f} / {d.net_pct.mean():+.3f} | {len(d)} |', flush=True)

    print('\n== exit mix and cost, per cell (the STOP SHARE is the prediction PREREG §4.3 made) ==')
    print('| cell | split | exit mix | cost R | cost % | r % | imputed % |')
    print('|---|---|---|---|---|---|---|')
    b0c = b0.copy()
    for nm, b in [('B0 1m-k5-X2R (B2 reference)', b0c)] + [(n, books[n]) for n in books]:
        for sp in SPLITS:
            d = b[b.split == sp]
            if not len(d):
                continue
            mix = ' '.join(f'{kk}:{v/len(d):.2f}' for kk, v in d.why.value_counts().items())
            print(f'| {nm} | {sp} | {mix} | {d.cost_R.mean():+.3f} | {d.cost_pct.mean():+.3f} | '
                  f'{d.r_pct.mean():.2f} | {d.imputed.mean()*100:.0f} |', flush=True)
    return 0


if __name__ == '__main__':
    repro_gate()
    sys.exit(main())
