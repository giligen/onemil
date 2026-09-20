#!/usr/bin/env python3
"""F38 stage 2 — SCORE the ten declared cells: THE FLOOR FIRST, then the detector, then the placebo.

  python3 s38.py > s38.log
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c12 import (D12, S, SPLITS, Sheet12, attach_cost12, book_ranked, build_cost_model,  # noqa: E402
                 cascade, clustered_t, repro_gate)

F34_1300 = {'TRAIN': -0.402, 'VAL': -0.255}      # F34's 13:00-14:01 cell at a 2 % stop, % of price
BOOK_CELLS = [('L1 lasthour +2R', 'S', 'X2R', True), ('L2 lasthour bare', 'S', 'XBR', False),
              ('L3 lasthour nextopen', 'S', 'XNO', False), ('L4 lasthour MOC', 'S', 'XMO', False),
              ('L5 lasthour lock', 'S', 'XLK', False),
              ('O1 orb1400 +2R', 'O', 'X2R', True), ('O2 orb1400 MOC', 'O', 'XMO', False)]


def prep(d, tag):
    x = d.copy()
    x['rr'] = x[f'{tag}_rr']; x['why'] = x[f'{tag}_why']; x['exit_m'] = x[f'{tag}_em']
    x = x[x.rr.notna() & (x.why != '')]
    return attach_cost12(x)


def main() -> int:
    build_cost_model()
    w = pd.read_csv(f'{D12}/w38.csv', dtype={'day': str, 'symbol': str, 'kind': str,
                                             'key_sym': str}, keep_default_na=False,
                    na_values=[''])
    w['split'] = S.split_of(w.day.values)
    w = w[w.split.isin(SPLITS)]
    w['wk'] = pd.to_datetime(w.day).dt.to_period('W-FRI').astype(str)

    # -------------------------------------------------------------- THE FLOOR, read first
    print('\n== F38 U1-U3 — THE FLOOR IN 14:00-15:30, walked BEFORE any detector ==')
    print('| cell | split | n | gross % | cost % | net % | t | tc | H1 % | H2 % | MDE % |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    floor_rows = []
    for nm, sw, tag in (('U1 floor 2% bracket', 0.02, 'X2R'), ('U2 floor 3% bracket', 0.03, 'X2R'),
                        ('U3 floor 2% MOC', 0.02, 'XMO')):
        f = prep(w[(w.kind == 'F') & (w.sw == sw)], tag)
        for sp in SPLITS:
            d = f[f.split == sp]
            if len(d) < 5:
                continue
            tr = f[f.split == 'TRAIN']
            h1 = tr[tr.day < '2025-07-01'].net_pct.mean()
            h2 = tr[tr.day >= '2025-07-01'].net_pct.mean()
            t = d.net_pct.mean() / (d.net_pct.std(ddof=1) / np.sqrt(len(d)))
            mde = 2.80 * d.net_pct.std(ddof=1) / np.sqrt(len(d))
            print(f'| {nm} | {sp} | {len(d):,} | {d.gross_pct.mean():+.3f} | '
                  f'{d.cost_pct.mean():+.3f} | {d.net_pct.mean():+.3f} | {t:+.2f} | '
                  f'{clustered_t(d, "net_pct"):+.2f} | {h1:+.3f} | {h2:+.3f} | {mde:.3f} |',
                  flush=True)
            floor_rows.append(dict(cell=nm, split=sp, n=len(d), gross_pct=d.gross_pct.mean(),
                                   cost_pct=d.cost_pct.mean(), net_pct=d.net_pct.mean(),
                                   t=t, tc=clustered_t(d, 'net_pct'), h1=h1, h2=h2, mde=mde))
        print(f'    exit mix: ' + ' '.join(f'{k}:{v/len(f):.2f}'
                                           for k, v in f.why.value_counts().items()), flush=True)
    pd.DataFrame(floor_rows).to_csv(f'{D12}/floor_cells38.csv', index=False)
    print(f"\n  F34's 13:00-14:01 cell at a 2 % stop, for the gradient: "
          f"TRAIN {F34_1300['TRAIN']:+.3f} % / VAL {F34_1300['VAL']:+.3f} %", flush=True)

    # -------------------------------------------------------------- the detector cells
    sh = Sheet12()
    books = {}
    print('\n== F38 L1-L5, O1-O2 — the last-hour detector ==\n', flush=True)
    for nm, kind, tag, capped in BOOK_CELLS:
        x = prep(w[w.kind == kind], tag)
        x = cascade(x) if kind == 'O' else x        # the S rows already passed the cascade in p38
        if not len(x):
            print(f'| {nm:<35s} | NO SIGNALS', flush=True)
            continue
        x = x.sort_values(['day', 'entry_m', 'symbol'], kind='mergesort').reset_index(drop=True)
        b = book_ranked(x, 12, 4)
        books[nm] = b
        sh.show(nm, b, capped=capped)
    sh.dump(f'{D12}/cells38.csv')
    sh.nulls(f'{D12}/nulls38.csv')

    # -------------------------------------------------------------- the placebo decomposition
    print('\n== the four-row PLACEBO DECOMPOSITION (net R | net % of entry price) ==')
    print('| cell | split | floor 14:00-15:30 % | matched non-signal | same name-day later min | '
          'THE SIGNAL | n |')
    print('|---|---|---|---|---|---|---|')
    fl = {(r['cell'], r['split']): r['net_pct'] for r in floor_rows}
    for nm, kind, tag, capped in BOOK_CELLS:
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
                a = prep(a, tag if tag in ('X2R', 'XBR', 'XMO') else 'X2R')
                out.append((a.net.mean() if len(a) else np.nan,
                            a.net_pct.mean() if len(a) else np.nan, len(a)))
            base = fl.get(('U3 floor 2% MOC' if tag == 'XMO' else 'U1 floor 2% bracket', sp),
                          np.nan)
            print(f'| {nm} | {sp} | {base:+.3f} | {out[0][0]:+.3f} / {out[0][1]:+.3f} '
                  f'(n {out[0][2]}) | {out[1][0]:+.3f} / {out[1][1]:+.3f} (n {out[1][2]}) | '
                  f'{d.net.mean():+.3f} / {d.net_pct.mean():+.3f} | {len(d)} |', flush=True)

    print('\n== exit mix and cost, per book cell ==')
    print('| cell | split | exit mix | cost R | cost % | r % | imputed % |')
    print('|---|---|---|---|---|---|---|')
    for nm, kind, tag, capped in BOOK_CELLS:
        b = books.get(nm)
        if b is None or not len(b):
            continue
        for sp in SPLITS:
            d = b[b.split == sp]
            if not len(d):
                continue
            mix = ' '.join(f'{k}:{v/len(d):.2f}' for k, v in d.why.value_counts().items())
            print(f'| {nm} | {sp} | {mix} | {d.cost_R.mean():+.3f} | {d.cost_pct.mean():+.3f} | '
                  f'{d.r_pct.mean():.2f} | {d.imputed.mean()*100:.0f} |', flush=True)
    return 0


if __name__ == '__main__':
    repro_gate()
    sys.exit(main())
