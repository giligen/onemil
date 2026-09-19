#!/usr/bin/env python3
"""PREREG §5 mandatory reporting: the null table, both rails, the availability
audit on every gating field, the three dead-knob confirmations and the tails.

No pipeline run: every number here is read off the declared cells' books, the
features CSV and stage 1's already-committed artifacts.
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
D = f'{ROOT}/research/orb_gates2'
F1 = f'{ROOT}/research/orb_frequency'
sys.path.insert(0, ROOT)
sys.path.insert(0, F1)
from trading.orb_csv import read_orb_csv       # noqa: E402
from score import load_book, split_of          # noqa: E402

FEATURES = f'{ROOT}/analysis_results/orb_features_20260916_2053.csv'
SPLITS = ('TRAIN', 'VAL')

# composite inputs, from orb.yaml's z-params (the 7-feature composite)
COMPOSITE_FIELDS = ['range_size_pct', 'range_total_volume', 'range_close_position',
                    'range_return_pct', 'gap_pct', 'prev_day_range_pct',
                    'avg_daily_volume_20d']
GATING_FIELDS = sorted(set(COMPOSITE_FIELDS + ['prev_day_range_pct',
                                               'return_volatility_20d',
                                               'range_size_pct']))


def availability():
    d = read_orb_csv(FEATURES)
    d['date'] = pd.to_datetime(d['date'])
    d['split'] = d['date'].map(split_of)
    rows = []
    for f in GATING_FIELDS:
        for sp in SPLITS + ('TEST',):
            s = d[d.split == sp][f]
            miss = float(pd.to_numeric(s, errors='coerce').isna().mean()) * 100
            rows.append({'field': f, 'split': sp, 'n': len(s), 'missing_pct': miss})
    # the news pair (tri-state) and the anchor
    news = set()
    for p in sorted(glob.glob(f'{ROOT}/data/research/orb_news_catalyst_*.csv')):
        n = read_orb_csv(p)
        news |= set(zip(n['symbol'], n['day']))
    key = list(zip(d['symbol'], d['date'].dt.strftime('%Y-%m-%d')))
    d['_news_known'] = [k in news for k in key]
    sys.path.insert(0, f'{D}')
    from partial_catalyst import _anchors
    a = _anchors()
    d['_anchor_known'] = d['symbol'].map(lambda s: a.get(s) is not None)
    for sp in SPLITS + ('TEST',):
        s = d[d.split == sp]
        rows.append({'field': 'news pair (tri-state)', 'split': sp, 'n': len(s),
                     'missing_pct': 100.0 * float((~s['_news_known']).mean())})
        rows.append({'field': 'underlying anchor', 'split': sp, 'n': len(s),
                     'missing_pct': 100.0 * float((~s['_anchor_known']).mean())})
    r = pd.DataFrame(rows)
    r.to_csv(f'{D}/availability.csv', index=False)
    print('=== AVAILABILITY AUDIT (candidate population, % missing) ===')
    print(r.pivot(index='field', columns='split', values='missing_pct')
          .round(2).to_string())
    return r


def dead_knobs():
    print('\n=== DEAD KNOBS ===')
    # (1) composite threshold: stage 1 ran it; verify its books are identical
    for a, b, lab in (('L_base', 'L_thr05', 'threshold -> -0.5'),
                      ('L_base', 'L_thrall', 'threshold OFF'),
                      ('L_pdroff', 'L_pdr8', 'PDR 11 -> 8'),
                      ('L_pdroff', 'L_pdr6', 'PDR 11 -> 6'),
                      ('L_rsoff', 'L_rs15', 'range-size 2.221 -> 1.5'),
                      ('L_rsoff', 'L_rs10', 'range-size 2.221 -> 1.0')):
        pa, pb = f'{F1}/book_{a}_meas.csv', f'{F1}/book_{b}_meas.csv'
        x, y = read_orb_csv(pa), read_orb_csv(pb)
        print(f'  stage-1 {lab:28s} {a} vs {b}: equals={x.equals(y)} '
              f'({len(x)} vs {len(y)} picks)')
    # (2) PDR vs G1 redundancy, measured directly on this stage's books
    g0 = load_book(f'{D}/book_G0_meas.csv')
    g7 = load_book(f'{D}/book_G7_meas.csv')
    print(f'  this stage: G0 {len(g0)} picks ${g0["_sized_pnl"].sum():,.0f} vs '
          f'G7 (PDR off) {len(g7)} picks ${g7["_sized_pnl"].sum():,.0f} '
          f'-> PDR@11.0 is worth {len(g7)-len(g0)} picks over 90 weeks')
    d = read_orb_csv(FEATURES)
    pdr = pd.to_numeric(d['prev_day_range_pct'], errors='coerce')
    print(f'  candidates with 9.226 <= pdr <= 11.0 (the only band PDR@11 can bind '
          f'that G1 would not): {int(((pdr > 9.226) & (pdr <= 11.0)).sum())} of '
          f'{len(d)} ({100*float(((pdr > 9.226) & (pdr <= 11.0)).mean()):.1f}%)')
    # (3) touchgo Rule D
    for tag in ('G0', 'G3', 'G5'):
        b = load_book(f'{D}/book_{tag}_meas.csv')
        f = b[b['entered'] == 1]
        print(f'  touchgo Rule D in {tag}: {int((f["exit_reason"] == "tag_b1").sum())}'
              f' of {len(f)} fills exit tag_b1;  Rule M: '
              f'{int((f["exit_reason"] == "tag_bb").sum())}')


def tails_and_nulls():
    g = pd.read_csv(f'{D}/grid_meas.csv')
    cols = ['cell', 'split', 'picks_per_wk', 'green_wk_pct', 'null_mean',
            'null_p5', 'null_p95', 'flat_wk_pct', 'red_streak', 'worst_wk',
            'green_mo_pct', 'mdd', 'pnl', 'R_pick', 'R_pick_ex1', 'R_pick_ex5',
            't', 'top1_share', 'top5_share', 'top10_share']
    out = g[cols].copy()
    out['vs_null'] = out['green_wk_pct'] - out['null_mean']
    out.to_csv(f'{D}/nulls_and_tails.csv', index=False)
    print('\n=== OBSERVED GREEN% vs ITS OWN COUNT-MATCHED NULL ===')
    print(out[['cell', 'split', 'picks_per_wk', 'green_wk_pct', 'null_mean',
               'null_p5', 'null_p95', 'vs_null']].round(1).to_string(index=False))
    print('\n=== TAILS (diagnostic only) ===')
    print(out[['cell', 'split', 'R_pick', 'R_pick_ex1', 'R_pick_ex5', 't',
               'top1_share', 'top5_share', 'top10_share']].round(3).to_string(index=False))


def rails():
    g = pd.read_csv(f'{D}/grid_meas.csv')
    tr = g[g.split == 'TRAIN'].set_index('cell')
    va = g[g.split == 'VAL'].set_index('cell')
    b_tr, b_va = tr.loc['G0'], va.loc['G0']
    rows = []
    for c in tr.index:
        k_tr = np.sqrt(tr.loc[c].picks_per_wk / b_tr.picks_per_wk)
        k_va = np.sqrt(va.loc[c].picks_per_wk / b_va.picks_per_wk)
        rows.append({
            'cell': c,
            'TR_worst': tr.loc[c].worst_wk,
            'TR_rail_scaled': k_tr * 1.5 * b_tr.worst_wk,
            'TR_rail_flat15': 1.5 * b_tr.worst_wk,
            'VA_worst': va.loc[c].worst_wk,
            'VA_rail_scaled': k_va * 1.5 * b_va.worst_wk,
            'VA_rail_flat15': 1.5 * b_va.worst_wk,
            'pass_scaled': bool(tr.loc[c].worst_wk >= k_tr * 1.5 * b_tr.worst_wk
                                and va.loc[c].worst_wk >= k_va * 1.5 * b_va.worst_wk),
            'pass_flat15': bool(tr.loc[c].worst_wk >= 1.5 * b_tr.worst_wk
                                and va.loc[c].worst_wk >= 1.5 * b_va.worst_wk),
            'TR_mdd': tr.loc[c].mdd, 'VA_mdd': va.loc[c].mdd,
            'TR_green_mo': tr.loc[c].green_mo_pct, 'VA_green_mo': va.loc[c].green_mo_pct,
        })
    r = pd.DataFrame(rows)
    r.to_csv(f'{D}/rails.csv', index=False)
    print('\n=== BOTH WORST-WEEK RAILS, MDD, GREEN MONTHS ===')
    print(r.round(0).to_string(index=False))


if __name__ == '__main__':
    pd.set_option('display.width', 400)
    availability()
    dead_knobs()
    tails_and_nulls()
    rails()
