#!/usr/bin/env python3
"""Score the 10 declared cells on the owner's metric + PREREG §4's rule.

PRIMARY = % GREEN WEEKS over every market week (no-pick weeks flat, in the
denominator).  Every cell is scored against its OWN count-matched permutation
null (PREREG §4 clause N).  TEST stays sealed until the recommendation is
committed.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
D = f'{ROOT}/research/orb_gates2'
sys.path.insert(0, f'{ROOT}/research/orb_frequency')
from score import (COLS, calendar, load_book, score_book,  # noqa: E402
                   split_of, week_frame)

LABEL = {
    'G0': 'G0 shipped B+ (baseline)',
    'G1': 'G1 Q1 filter OFF',
    'G2': 'G2 range-size veto OFF',
    'G3': 'G3 catalyst veto OFF',
    'G4': 'G4 Q1 + range-size OFF',
    'G5': 'G5 G4 + catalyst OFF',
    'G6': 'G6 catalyst PARTIAL (min_cohort 2->1)',
    'G7': 'G7 PDR veto OFF',
    'G8': 'G8 G4 + catalyst PARTIAL',
    'G9': 'G9 G4 + PDR OFF (defect cleanup)',
}
CELLS = list(LABEL)
NDRAW = 2000
SEED = 20260919


def perm_null(book: pd.DataFrame, split: str, ndraw: int = NDRAW) -> tuple:
    """Count-matched null: shuffle the cell's own pick P&L across its own picks,
    each week's pick COUNT held fixed.  Returns (mean, p5, p95) green-week %."""
    w = week_frame(book, split)
    nw = len(w)
    b = book[book['date'].map(split_of) == split]
    if not nw or not len(b):
        return (0.0, 0.0, 0.0)
    counts = w['n'].values
    vals = b['_sized_pnl'].values.astype(float)
    rng = np.random.default_rng(SEED)
    idx = np.repeat(np.arange(nw), counts)          # pick -> week slot
    out = np.empty(ndraw)
    for i in range(ndraw):
        v = rng.permutation(vals)
        s = np.bincount(idx, weights=v, minlength=nw)
        out[i] = 100.0 * float((s > 0).sum()) / nw
    return (float(out.mean()), float(np.percentile(out, 5)),
            float(np.percentile(out, 95)))


def collect(dump: str, splits=('TRAIN', 'VAL'), reveal=False) -> pd.DataFrame:
    rows = []
    for tag in CELLS:
        p = f'{D}/book_{tag}_{dump}.csv'
        if not os.path.exists(p):
            continue
        r = score_book(p, tag, splits, reveal)
        b = load_book(p)
        nulls = {sp: perm_null(b, sp) for sp in splits}
        r['null_mean'] = r['split'].map(lambda s: nulls[s][0])
        r['null_p5'] = r['split'].map(lambda s: nulls[s][1])
        r['null_p95'] = r['split'].map(lambda s: nulls[s][2])
        r['label'] = LABEL[tag]
        r['dump'] = dump
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def rule(tr, va, b_tr, b_va) -> tuple:
    """PREREG §4 clauses 1-5, verbatim.  Returns (ok, failed, flat15_ok)."""
    fail = []
    if not (tr.green_wk_pct >= b_tr.green_wk_pct and va.green_wk_pct >= b_va.green_wk_pct
            and (tr.green_wk_pct > b_tr.green_wk_pct or va.green_wk_pct > b_va.green_wk_pct)):
        fail.append('1 green')
    if not (tr.flat_wk_pct < b_tr.flat_wk_pct and va.flat_wk_pct < b_va.flat_wk_pct):
        fail.append('2 flat')
    if not (tr.red_streak <= b_tr.red_streak + 2 and va.red_streak <= b_va.red_streak + 2):
        fail.append('3 streak')
    if not (tr.pnl >= 0 and va.pnl >= 0):
        fail.append('4 pnl')
    k_tr = np.sqrt(tr.picks_per_wk / b_tr.picks_per_wk) if b_tr.picks_per_wk else 1.0
    k_va = np.sqrt(va.picks_per_wk / b_va.picks_per_wk) if b_va.picks_per_wk else 1.0
    if not (tr.worst_wk >= k_tr * 1.5 * b_tr.worst_wk
            and va.worst_wk >= k_va * 1.5 * b_va.worst_wk):
        fail.append('5 worst-wk(scaled)')
    flat15 = (tr.worst_wk >= 1.5 * b_tr.worst_wk and va.worst_wk >= 1.5 * b_va.worst_wk)
    return (not fail), fail, bool(flat15)


def null_clause(tr, va) -> bool:
    """PREREG §4 clause N."""
    beats95 = (tr.green_wk_pct > tr.null_p95) or (va.green_wk_pct > va.null_p95)
    ge_mean = (tr.green_wk_pct >= tr.null_mean) and (va.green_wk_pct >= va.null_mean)
    return bool(beats95 and ge_mean)


def main():
    pd.set_option('display.width', 400)
    for dump in ('meas', 'asis'):
        g = collect(dump)
        g.to_csv(f'{D}/grid_{dump}.csv', index=False)
    g = collect('meas')
    tr = g[g.split == 'TRAIN'].set_index('cell')
    va = g[g.split == 'VAL'].set_index('cell')
    b_tr, b_va = tr.loc['G0'], va.loc['G0']

    res = []
    for c in tr.index:
        ok, fail, flat15 = rule(tr.loc[c], va.loc[c], b_tr, b_va)
        nul = null_clause(tr.loc[c], va.loc[c])
        pooled = (tr.loc[c].green_wk_pct * tr.loc[c].weeks
                  + va.loc[c].green_wk_pct * va.loc[c].weeks) \
            / (tr.loc[c].weeks + va.loc[c].weeks)
        res.append({
            'cell': c, 'label': LABEL[c], 'pooled_green': pooled,
            'TR_green': tr.loc[c].green_wk_pct, 'VA_green': va.loc[c].green_wk_pct,
            'TR_null_p95': tr.loc[c].null_p95, 'VA_null_p95': va.loc[c].null_p95,
            'TR_null_mean': tr.loc[c].null_mean, 'VA_null_mean': va.loc[c].null_mean,
            'TR_flat': tr.loc[c].flat_wk_pct, 'VA_flat': va.loc[c].flat_wk_pct,
            'TR_pk': tr.loc[c].picks_per_wk, 'VA_pk': va.loc[c].picks_per_wk,
            'TR_streak': tr.loc[c].red_streak, 'VA_streak': va.loc[c].red_streak,
            'TR_worst': tr.loc[c].worst_wk, 'VA_worst': va.loc[c].worst_wk,
            'TR_mdd': tr.loc[c].mdd, 'VA_mdd': va.loc[c].mdd,
            'TR_pnl': tr.loc[c].pnl, 'VA_pnl': va.loc[c].pnl,
            'TR_Rpk': tr.loc[c].R_pick, 'VA_Rpk': va.loc[c].R_pick,
            'TR_Rex5': tr.loc[c].R_pick_ex5, 'VA_Rex5': va.loc[c].R_pick_ex5,
            'TR_t': tr.loc[c].t, 'VA_t': va.loc[c].t,
            'top5_TR': tr.loc[c].top5_share, 'top5_VA': va.loc[c].top5_share,
            'explore_bar': ok if c != 'G0' else None,
            'failed': ','.join(fail), 'flat1.5x_rail': flat15,
            'null_clause': nul,
            'claim_bar': bool(tr.loc[c].t >= 2.0
                              and np.sign(va.loc[c].R_pick) == np.sign(tr.loc[c].R_pick)),
        })
    r = pd.DataFrame(res).sort_values('pooled_green', ascending=False)
    r.to_csv(f'{D}/cells.csv', index=False)
    print('=== CELLS ranked on pooled green-week %, measured fill model, N=8 ===')
    print(r[['cell', 'TR_pk', 'VA_pk', 'TR_green', 'VA_green', 'TR_null_p95',
             'VA_null_p95', 'TR_flat', 'VA_flat', 'TR_streak', 'VA_streak',
             'TR_worst', 'VA_worst', 'TR_pnl', 'VA_pnl', 'explore_bar',
             'null_clause', 'claim_bar', 'failed']].round(1).to_string(index=False))
    print(f'\nMDE80 green-week (unpaired): TRAIN +-{b_tr.mde80_green_pp:.1f}pp '
          f'({int(b_tr.weeks)} wk), VAL +-{b_va.mde80_green_pp:.1f}pp ({int(b_va.weeks)} wk)')
    ga = collect('asis')
    ga.to_csv(f'{D}/grid_asis.csv', index=False)
    at = ga[ga.split == 'TRAIN'].set_index('cell')
    av = ga[ga.split == 'VAL'].set_index('cell')
    print('\n=== as-is bracket (secondary fill model) ===')
    b = pd.DataFrame({'TR_green': at.green_wk_pct, 'VA_green': av.green_wk_pct,
                      'TR_flat': at.flat_wk_pct, 'VA_flat': av.flat_wk_pct,
                      'TR_pnl': at.pnl, 'VA_pnl': av.pnl,
                      'TR_null_p95': at.null_p95, 'VA_null_p95': av.null_p95})
    print(b.round(1).to_string())


if __name__ == '__main__':
    main()
