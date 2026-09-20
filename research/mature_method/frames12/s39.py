#!/usr/bin/env python3
"""F39 stage 3 — SCORE the eleven declared cells, with the four-row placebo decomposition.

  python3 s39.py > s39.log
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c12 import (D12, S, SPLITS, RISK, Sheet12, attach_cost12, book_ranked,   # noqa: E402
                 build_cost_model, clustered_t, repro_gate)

CELLS = [('1  H5-S20-X2R', 'H5', 'S20', 'X2R'), ('2  H5-S20-XLK', 'H5', 'S20', 'XLK'),
         ('3  H5-S20-XBR', 'H5', 'S20', 'XBR'), ('4  H5-S20-XNO', 'H5', 'S20', 'XNO'),
         ('5  H20-S20-X2R', 'H20', 'S20', 'X2R'), ('6  H20-S20-XLK', 'H20', 'S20', 'XLK'),
         ('7  H20-S20-XBR', 'H20', 'S20', 'XBR'), ('8  H20-S20-XNO', 'H20', 'S20', 'XNO'),
         ('9  H252-S20-X2R', 'H252', 'S20', 'X2R'), ('10 H5-SPD-X2R', 'H5', 'SPD', 'X2R'),
         ('11 H20-SPD-X2R', 'H20', 'SPD', 'X2R')]
U34 = {'TRAIN': -0.389, 'VAL': -0.360}          # F34's unconditional 2 %-stop floor, % of price


def load_controls():
    c = pd.read_csv(f'{D12}/w39c.csv', dtype={'day': str, 'symbol': str, 'ctrl': str,
                                              'arm': str}, keep_default_na=False, na_values=[''])
    c['split'] = S.split_of(c.day.values)
    return c[c.split.isin(SPLITS)]


def ctrl_stats(c, tag, keys, split):
    """Mean net R and net % of price for one control arm on the booked keys of one split."""
    x = c[c.set_index(['day', 'symbol', 'entry_m']).index.isin(keys)]
    x = x[x.split == split].copy()
    x['rr'] = x[f'{tag}_rr']; x['why'] = x[f'{tag}_why']
    x = x[x.rr.notna()]
    if not len(x):
        return np.nan, np.nan, 0
    x = x.rename(columns={'ctrl': '_c', 'symbol': '_s'})
    x['symbol'] = x._c; x['entry_m'] = x.ctrl_m
    x = attach_cost12(x)
    return float(x.net.mean()), float(x.net_pct.mean()), len(x)


def main() -> int:
    build_cost_model()
    s = pd.read_csv(f'{D12}/sig39.csv', dtype={'day': str, 'symbol': str, 'lvl': str,
                                               'why': str}, keep_default_na=False, na_values=[''])
    c = load_controls()
    sh = Sheet12()
    books = {}
    print('\n== F39 — THE MULTI-DAY HIGH: the eleven declared cells ==\n', flush=True)
    for name, lv, st, ex in CELLS:
        x = s[(s.lvl == lv) & (s.stop_tag == st) & (s.exit_tag == ex)].copy()
        if not len(x):
            print(f'| {name:<35s} | NO SIGNALS', flush=True)
            continue
        x = x.sort_values(['day', 'entry_m', 'symbol'], kind='mergesort').reset_index(drop=True)
        b = book_ranked(x, 12, 4)
        books[name] = b
        sh.show(name, b, capped=(ex == 'X2R'))
    sh.dump(f'{D12}/cells39.csv')
    sh.nulls(f'{D12}/nulls39.csv')

    print('\n== the four-row PLACEBO DECOMPOSITION (net R | net % of entry price) ==')
    print('| cell | split | universe bound % | matched non-signal | same name-day later min | '
          'THE SIGNAL | n sig |')
    print('|---|---|---|---|---|---|---|')
    for name, lv, st, ex in CELLS:
        b = books.get(name)
        if b is None or not len(b):
            continue
        for sp in SPLITS:
            d = b[b.split == sp]
            if not len(d):
                continue
            keys = set(zip(d.day, d.symbol, d.entry_m))
            cb = ctrl_stats(c[c.arm == 'CB'], ex, keys, sp)
            ca = ctrl_stats(c[c.arm == 'CA'], ex, keys, sp)
            print(f'| {name} | {sp} | {U34[sp]:+.3f} | {cb[0]:+.3f} / {cb[1]:+.3f} (n {cb[2]}) | '
                  f'{ca[0]:+.3f} / {ca[1]:+.3f} (n {ca[2]}) | '
                  f'{d.net.mean():+.3f} / {d.net_pct.mean():+.3f} | {len(d)} |', flush=True)

    print('\n== exit mix and cost, per cell ==')
    print('| cell | split | exit mix | mean cost R | mean cost % | mean r % | imputed % |')
    print('|---|---|---|---|---|---|---|')
    for name, lv, st, ex in CELLS:
        b = books.get(name)
        if b is None or not len(b):
            continue
        for sp in SPLITS:
            d = b[b.split == sp]
            if not len(d):
                continue
            mix = ' '.join(f'{k}:{v/len(d):.2f}' for k, v in d.why.value_counts().items())
            print(f'| {name} | {sp} | {mix} | {d.cost_R.mean():+.3f} | {d.cost_pct.mean():+.3f} | '
                  f'{d.r_pct.mean():.2f} | {d.imputed.mean()*100:.0f} |', flush=True)

    print('\n== overnight exposure (the XNO cells) ==')
    for name, lv, st, ex in CELLS:
        if ex != 'XNO':
            continue
        b = books.get(name)
        if b is None or not len(b):
            continue
        held = b[b.why == 'nextopen']
        tail = held[held.day >= '2026-05-28']
        print(f'  {name}: {len(held)} of {len(b)} trades carried overnight '
              f'({len(held)/len(b):.0%}); of those {len(tail)} sit on the last two VAL sessions '
              f'(the FREEZE.md exception)', flush=True)
        if len(held):
            print(f'    overnight leg rr: mean {held.rr.mean():+.3f} '
                  f'min {held.rr.min():+.3f} max {held.rr.max():+.3f}', flush=True)
    return 0


if __name__ == '__main__':
    repro_gate()
    sys.exit(main())
