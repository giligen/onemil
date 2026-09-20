#!/usr/bin/env python3
"""F39 stage 2 — the cascade, the price-scale check, and the control keys.  No bar is read here.

Reads `w39.csv` (the multi-day-high walk), applies the shipped pre-book cascade unchanged, runs the
mandatory daily-vs-intraday price-scale check (CLAUDE.md rail 3 / PREREG §1.4), and writes
`sig39.csv` (the cascade-passing signals, all five (stop, exit) combinations on one row) and
`ctrl39.csv` (the two placebo arms' keys).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c12 import (D12, S, SPLITS, attach_cost12, build_cost_model, cascade,   # noqa: E402
                 FIRST_ENTRY_M, LAST_ENTRY_M, load_panel)

COMBOS = (('S20', 'X2R'), ('S20', 'XBR'), ('S20', 'XLK'), ('S20', 'XNO'), ('SPD', 'X2R'))
NPOOL, NLATER, SEED = 10, 10, 39


def main() -> int:
    build_cost_model()
    d = pd.read_csv(f'{D12}/w39.csv', dtype={'day': str, 'symbol': str, 'lvl': str},
                    keep_default_na=False, na_values=[''])
    n0 = len(d)
    d['split'] = S.split_of(d.day.values)
    d = d[d.split.isin(SPLITS)]
    d['wk'] = pd.to_datetime(d.day).dt.to_period('W-FRI').astype(str)

    # ---- the PRICE-SCALE CHECK, before anything is scored ---------------------------------
    sc = d.daily_close / d.last_close
    bad = (~sc.between(0.99, 1.01)) | sc.isna()
    print(f'  price-scale check (daily close vs intraday last close): '
          f'{int(bad.sum())} of {len(d)} rows = {bad.mean():.2%} DROPPED', flush=True)
    d = d[~bad]

    # ---- the cascade, applied to the PRIMARY (S20, X2R) key so membership is ONE set -------
    d = d[d.fill_capped == 1]
    d = d[(d.rv_profile >= 1.0) & (d.entry_m >= FIRST_ENTRY_M) & (d.entry_m <= LAST_ENTRY_M)]
    rows = []
    for stop_tag, ex in COMBOS:
        x = d.copy()
        x['stop'] = x[f'stop_{stop_tag}']
        x['rr'] = x[f'{stop_tag}_{ex}_rr']
        x['why'] = x[f'{stop_tag}_{ex}_why']
        x['exit_m'] = x[f'{stop_tag}_{ex}_exit_m']
        x['r_pct'] = (x.next_open - x.stop) / x.next_open * 100.0
        x = x[x.rr.notna() & x.stop.notna() & (x.stop < x.next_open)]
        x = attach_cost12(x)
        s = cascade(x)
        s['stop_tag'] = stop_tag; s['exit_tag'] = ex
        rows.append(s)
    s = pd.concat(rows, ignore_index=True)
    keep = ['day', 'symbol', 'split', 'wk', 'lvl', 'stop_tag', 'exit_tag', 'entry_m', 'break_m',
            'level', 'next_open', 'stop', 'r_pct', 'rr', 'why', 'exit_m', 'net', 'netb', 'cost_R',
            'gross_pct', 'net_pct', 'cost_pct', 'sp_pct', 'imputed', 'adv20']
    s[keep].to_csv(f'{D12}/sig39.csv', index=False)
    print(f'  w39 rows {n0:,} -> cascade-passing signal-cells {len(s):,}', flush=True)
    for lv in ('H5', 'H20', 'H252'):
        z = s[(s.lvl == lv) & (s.stop_tag == 'S20') & (s.exit_tag == 'X2R')]
        print(f'    {lv:5s} pre-book signals {len(z):5d}  '
              + '  '.join(f'{sp} {int((z.split==sp).sum())}' for sp in SPLITS), flush=True)

    # ---- the control keys (one set, shared by every cell) ---------------------------------
    base = s[(s.stop_tag == 'S20') & (s.exit_tag == 'X2R')].drop_duplicates(['day', 'symbol',
                                                                             'entry_m'])
    u = load_panel()
    u = u[(u.prev_close >= 17.0) & (u.adv20 >= 100000)]
    sig_keys = set(zip(s.day, s.symbol))
    u = u[~pd.Series(list(zip(u.day, u.symbol)), index=u.index).isin(sig_keys)]
    u = u.assign(lp=np.log(u.prev_close), la=np.log(u.adv20))
    UD = {dd: g for dd, g in u[u.day.isin(set(base.day))].groupby('day')}
    rng = np.random.default_rng(SEED)
    out = []
    for r in base.itertuples():
        g = UD.get(r.day)
        if g is not None and len(g) and r.adv20 == r.adv20 and r.adv20 > 0:
            dist = (np.abs(g.lp.values - np.log(float(r.next_open))) +
                    np.abs(g.la.values - np.log(float(r.adv20))))
            for z in np.argsort(dist, kind='mergesort')[:NPOOL]:
                out.append((r.day, 'CB', g.symbol.values[z], int(r.entry_m), r.symbol,
                            int(r.entry_m), float(r.r_pct)))
        lo = int(r.entry_m) + 1
        if lo <= LAST_ENTRY_M:
            cand = np.arange(lo, LAST_ENTRY_M + 1)
            pick = cand if len(cand) <= NLATER else rng.choice(cand, NLATER, replace=False)
            for mm in sorted(int(z) for z in pick):
                out.append((r.day, 'CA', r.symbol, mm, r.symbol, int(r.entry_m), float(r.r_pct)))
    c = pd.DataFrame(out, columns=['day', 'arm', 'ctrl', 'ctrl_m', 'symbol', 'entry_m', 'r_pct'])
    c.to_csv(f'{D12}/ctrl39.csv', index=False)
    print(f'  ctrl39.csv {len(c):,} rows (CB {int((c.arm=="CB").sum()):,}, '
          f'CA {int((c.arm=="CA").sum()):,}) over {c.day.nunique()} sessions', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
