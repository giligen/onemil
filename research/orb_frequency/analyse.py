#!/usr/bin/env python3
"""Score every declared cell on the owner's metric and apply PREREG §7.

TRAIN + VAL only.  TEST stays sealed (FREEZE.md) until the recommendation is
committed; `--reveal-test` then scores exactly two cells.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

D = '/home/ec2-user/onemil/research/orb_frequency'
sys.path.insert(0, D)
from score import score_book, load_book, week_frame, split_of  # noqa: E402

LADDER_LABEL = {
    'L_base': 'shipped B+ (all gates on)',
    'L_q1off': 'Q1 filter OFF',
    'L_pdr8': 'PDR veto 11.0 -> 8.0',
    'L_pdr6': 'PDR veto 11.0 -> 6.0',
    'L_pdroff': 'PDR veto OFF',
    'L_g1off': 'G1 fingerprint OFF',
    'L_rs15': 'range-size veto 2.221 -> 1.5',
    'L_rs10': 'range-size veto 2.221 -> 1.0',
    'L_rsoff': 'range-size veto OFF',
    'L_catoff': 'catalyst veto OFF',
    'L_thr05': 'composite threshold -> -0.5',
    'L_thrall': 'composite threshold OFF (all candidates rank)',
    'F0': 'F0 = shipped B+',
    'F1': 'F1 = F0 - range-size',
    'F2': 'F2 = F0 - range-size, PDR 8.0',
    'F3': 'F3 = F0 - range-size - G1',
    'F4': 'F4 = F0 - range-size - G1 - PDR',
    'F5': 'F5 = F4 - catalyst',
    'F6': 'F6 = CEILING (every gate off)',
}


def collect(dump: str = 'meas') -> pd.DataFrame:
    rows = []
    for tag, label in LADDER_LABEL.items():
        p = f'{D}/book_{tag}_{dump}.csv'
        if not os.path.exists(p):
            continue
        r = score_book(p, tag)
        r['label'] = label
        r['dump'] = dump
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def wide(g: pd.DataFrame, cols) -> pd.DataFrame:
    tr = g[g.split == 'TRAIN'].set_index('cell')
    va = g[g.split == 'VAL'].set_index('cell')
    out = pd.DataFrame({'label': tr['label']})
    for c in cols:
        out[f'TR_{c}'] = tr[c]
        out[f'VA_{c}'] = va[c]
    return out


def survives(tr: pd.Series, va: pd.Series, b_tr: pd.Series, b_va: pd.Series) -> tuple:
    """PREREG §7, verbatim.  Returns (bool, list-of-failed-rules)."""
    fail = []
    if not (tr.green_wk_pct >= b_tr.green_wk_pct and va.green_wk_pct >= b_va.green_wk_pct
            and (tr.green_wk_pct > b_tr.green_wk_pct or va.green_wk_pct > b_va.green_wk_pct)):
        fail.append('1 green-week')
    if not (tr.flat_wk_pct < b_tr.flat_wk_pct and va.flat_wk_pct < b_va.flat_wk_pct):
        fail.append('2 flat-week')
    if not (tr.red_streak <= b_tr.red_streak + 1 and va.red_streak <= b_va.red_streak + 1):
        fail.append('3 red-streak')
    if not (tr.pnl >= 0 and va.pnl >= 0):
        fail.append('4 P&L floor')
    if not (tr.worst_wk >= 1.5 * b_tr.worst_wk and va.worst_wk >= 1.5 * b_va.worst_wk):
        fail.append('5 worst-week')
    return (not fail), fail


def mcnemar_mde(n_weeks: int, p: float = 0.35, disc_frac: float = 0.25) -> float:
    """Rough paired MDE80 in pp: the books nest, so only discordant weeks count."""
    nd = max(n_weeks * disc_frac, 1)
    return 100.0 * 2.80 * np.sqrt(0.25 / nd) * 2


def main():
    for dump in ('meas', 'asis'):
        g = collect(dump)
        g.to_csv(f'{D}/grid_{dump}.csv', index=False)
    g = collect('meas')
    cols = ['picks_per_wk', 'green_wk_pct', 'flat_wk_pct', 'red_wk_pct',
            'red_streak', 'worst_wk', 'green_mo_pct', 'mdd', 'pnl', 'R_pick',
            'R_pick_ex5', 't']
    w = wide(g, cols)
    pd.set_option('display.width', 300)
    print('=== ALL DECLARED CELLS, measured fill model, N=8 '
          '(TR = TRAIN 2025, VA = VAL 2026-01..05) ===')
    print(w[['label', 'TR_picks_per_wk', 'VA_picks_per_wk', 'TR_green_wk_pct',
             'VA_green_wk_pct', 'TR_flat_wk_pct', 'VA_flat_wk_pct',
             'TR_red_streak', 'VA_red_streak', 'TR_pnl', 'VA_pnl',
             'TR_R_pick', 'VA_R_pick']].round(2).to_string())

    tr = g[g.split == 'TRAIN'].set_index('cell')
    va = g[g.split == 'VAL'].set_index('cell')
    b_tr, b_va = tr.loc['L_base'], va.loc['L_base']
    res = []
    for cell in tr.index:
        if cell in ('L_base', 'F0'):
            continue
        ok, fail = survives(tr.loc[cell], va.loc[cell], b_tr, b_va)
        res.append({'cell': cell, 'label': LADDER_LABEL[cell], 'survives': ok,
                    'failed': ','.join(fail),
                    'pooled_green': (tr.loc[cell].green_wk_pct * tr.loc[cell].weeks
                                     + va.loc[cell].green_wk_pct * va.loc[cell].weeks)
                    / (tr.loc[cell].weeks + va.loc[cell].weeks)})
    r = pd.DataFrame(res).sort_values('pooled_green', ascending=False)
    r.to_csv(f'{D}/survival.csv', index=False)
    print('\n=== PREREG §7 survival rule ===')
    print(r.round(2).to_string(index=False))
    pooled_base = (b_tr.green_wk_pct * b_tr.weeks + b_va.green_wk_pct * b_va.weeks) \
        / (b_tr.weeks + b_va.weeks)
    print(f'\nshipped B+ pooled green-week % = {pooled_base:.1f}  '
          f'(TRAIN {b_tr.green_wk_pct:.1f} / VAL {b_va.green_wk_pct:.1f}; '
          f'flat TRAIN {b_tr.flat_wk_pct:.1f} / VAL {b_va.flat_wk_pct:.1f})')
    print(f'MDE80 green-week share, unpaired: TRAIN +-{b_tr.mde80_green_pp:.1f}pp '
          f'({int(b_tr.weeks)} wk), VAL +-{b_va.mde80_green_pp:.1f}pp ({int(b_va.weeks)} wk); '
          f'paired/McNemar approx TRAIN +-{mcnemar_mde(b_tr.weeks):.1f}pp, '
          f'VAL +-{mcnemar_mde(b_va.weeks):.1f}pp')


if __name__ == '__main__':
    main()
