#!/usr/bin/env python3
"""F38 stage 0 — the LAST-HOUR signal list and the control keys.  No bar is read here.

Writes `sig38.csv` (the last-hour HOD-break signals: the FIRST break per symbol-day with an entry
minute in [840, 931], the shipped cascade applied unchanged), `ctrl38.csv` (10 matched non-signal
names per signal at the SIGNAL'S OWN minute, matched on prior close, ADV20 and wrapper-vs-common)
and `floor38.csv` (the detector-free floor keys: every eligible PIT-panel non-signal name of the
session at 2 random minutes in [840, 931]).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c12 import (D12, ROOT, S, SPLITS, attach_cost12, build_cost_model, cascade,   # noqa: E402
                 load_breaks4, load_panel, repro_gate)

WIN = (840, 931)                # 14:00 .. 15:31 entry minutes
NPOOL = 10
NFLOOR_MIN = 2                  # random minutes per non-signal name per session
SEED = 12


def main() -> int:
    build_cost_model()
    br = load_breaks4(verbose=True)
    br = br[br.split.isin(SPLITS)]
    # the shipped detector, unchanged; only the CLOCK moves
    d = br[br.stop_n.notna() & (br.dist_open_pct >= 5.0) &
           (br.rv_profile >= 1.0) & (br.rv_profile < 5.0) &
           (br.entry_m >= WIN[0]) & (br.entry_m <= WIN[1])]
    d = d.drop_duplicates(['day', 'symbol'], keep='first')      # the FIRST break in the window
    d = d[(d.fill_capped == 1) & d.r_pct_n.notna()]
    x = d.rename(columns={'r_pct_n': 'r_pct', 'rr_n': 'rr', 'why_n': 'why',
                          'exit_m_n': 'exit_m', 'stop_n': 'stop'}).copy()
    x = x.drop(columns=[c for c in ('spread_mean', 'ask_dec', 'bid_dec') if c in x.columns])
    x = attach_cost12(x)
    s = cascade(x)
    print(f'  F38 last-hour signals: {len(d)} raw -> {len(s)} after the shipped cascade '
          f'({s.day.nunique()} sessions)', flush=True)
    for sp in SPLITS:
        print(f'    {sp}: {int((s.split == sp).sum())}', flush=True)
    keep = ['day', 'symbol', 'split', 'wk', 'entry_m', 'break_m', 'level', 'next_open', 'stop',
            'r_pct', 'rr', 'why', 'exit_m', 'sp_pct', 'imputed', 'adv20']
    s[keep].to_csv(f'{D12}/sig38.csv', index=False)

    # ------------------------------------------------------------------ the control pools
    u = load_panel()
    u = u[(u.prev_close >= 17.0) & (u.adv20 >= 100000)]
    sig = pd.read_csv(f'{ROOT}/research/mature_method/hod_frames2/breaks2.csv',
                      usecols=['day', 'symbol'], dtype=str).drop_duplicates()
    sset = set(zip(sig.day, sig.symbol))
    u = u[~pd.Series(list(zip(u.day, u.symbol)), index=u.index).isin(sset)]
    u = u.assign(lp=np.log(u.prev_close), la=np.log(u.adv20))
    days = set(s.day.unique())
    alldays = set(br.day.unique())      # the FLOOR and the ORB object run on EVERY session
    rng = np.random.default_rng(SEED)

    # arm b — matched non-signal name, the SIGNAL'S own minute, the signal's own R geometry
    UD = {d_: g for d_, g in u[u.day.isin(days)].groupby('day')}
    rows = []
    for r in s.itertuples():
        g = UD.get(r.day)
        if g is None or not (r.adv20 == r.adv20 and r.adv20 > 0):
            continue
        pc = float(r.next_open)                       # the signal's own price scale
        dist = np.abs(g.lp.values - np.log(pc)) + np.abs(g.la.values - np.log(float(r.adv20)))
        idx = np.argsort(dist, kind='mergesort')[:NPOOL]
        for k in idx:
            rows.append((r.day, r.symbol, int(r.entry_m), float(r.r_pct),
                         g.symbol.values[k], int(r.entry_m), 'CB'))
    # arm a — the SAME name-day, a minute strictly AFTER the signal (pass-6 causality rule)
    for r in s.itertuples():
        lo = int(r.entry_m) + 1
        if lo > WIN[1]:
            continue
        cand = np.arange(lo, WIN[1] + 1)
        pick = cand if len(cand) <= NPOOL else rng.choice(cand, NPOOL, replace=False)
        for mm in sorted(int(v) for v in pick):
            rows.append((r.day, r.symbol, int(r.entry_m), float(r.r_pct), r.symbol, mm, 'CA'))
    c = pd.DataFrame(rows, columns=['day', 'symbol', 'entry_m', 'r_pct', 'ctrl', 'ctrl_m', 'arm'])
    c.to_csv(f'{D12}/ctrl38.csv', index=False)
    print(f'  ctrl38.csv {len(c)} rows (CB {int((c.arm=="CB").sum())}, '
          f'CA {int((c.arm=="CA").sum())})', flush=True)

    # the FLOOR — detector-free, every eligible non-signal name, 2 random minutes in the window
    f = u[u.day.isin(alldays)][['day', 'symbol']].copy()
    f = pd.concat([f.assign(ctrl_m=rng.integers(WIN[0], WIN[1] + 1, len(f)))
                   for _ in range(NFLOOR_MIN)], ignore_index=True)
    f = f.drop_duplicates(['day', 'symbol', 'ctrl_m'])
    f.to_csv(f'{D12}/floor38.csv', index=False)
    print(f'  floor38.csv {len(f)} rows over {f.day.nunique()} sessions '
          f'({len(f)/max(f.day.nunique(),1):.0f}/session)', flush=True)

    # the ORB-style object's population = the B2 population's symbol-days (the same pond, a
    # different detector), restricted to the sessions the last-hour book trades.
    p = br[['day', 'symbol']].drop_duplicates()
    p = p[p.day.isin(alldays)]
    p.to_csv(f'{D12}/orbpop38.csv', index=False)
    print(f'  orbpop38.csv {len(p)} symbol-days over {p.day.nunique()} sessions', flush=True)
    return 0


if __name__ == '__main__':
    repro_gate()
    sys.exit(main())
