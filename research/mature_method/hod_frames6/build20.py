#!/usr/bin/env python3
"""hod_frames6 / F20 stage 1 — the reproduction gate, the booked book, and the match pools.

Writes `book6.csv` (the 2,328 booked B2 trades with everything the placebo needs) and `pool6.csv`
(for every booked trade, the 25 nearest NON-SIGNAL symbols of the same session, matched on prior
close, ADV20 and wrapper-vs-common).  No bar is read here; this is the join stage.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import (D6, ROOT, S, SPLITS, base_book, repro_line, attach_instrument,   # noqa: E402
                     load_breaks4)

NPOOL = 25


def main():
    print('== F20 stage 1 — reproduction gate ==', flush=True)
    br = load_breaks4(verbose=True)
    b, s = base_book(br, verbose=False)
    repro_line(b, 'B2 ')
    ref = {'TRAIN': (1622, -0.039, -17346.0), 'VAL': (706, 0.083, 893.0)}
    for sp in SPLITS:
        w = S.week_stats(b, sp); n, g, t = ref[sp]
        assert w['n'] == n, f'R1/R2 FAIL {sp}: n {w["n"]} != {n}'
        assert abs(w['total'] - t) < 1.0, f'R1/R2 FAIL {sp}: total {w["total"]} != {t}'
        assert abs(w['gross'] - g) < 5e-4, f'R1/R2 FAIL {sp}: gross {w["gross"]} != {g}'
    print('  R1/R2 reproduction gate: MATCH (n, gross, $ all exact)', flush=True)

    b = attach_instrument(b)
    keep = ['day', 'symbol', 'split', 'wk', 'entry_m', 'break_m', 'exit_m', 'level', 'price',
            'next_open', 'stop', 'r_pct', 'rr', 'net', 'pnl', 'why', 'sp_pct', 'imputed',
            'asset_class', 'anchor', 'spy_r5_pct', 'dollar_frac', 'adv20']
    bk = b[keep].copy().sort_values(['day', 'entry_m', 'symbol'], kind='mergesort')
    bk.to_csv(f'{D6}/book6.csv', index=False)
    print(f'  book6.csv {len(bk)} booked trades over {bk.day.nunique()} sessions', flush=True)

    # ---------------------------------------------------------------- the non-signal match pool
    u = pd.read_csv(f'{ROOT}/research/bf_zero/universe.csv',
                    usecols=['symbol', 'bar_date', 'close', 'adv20'],
                    dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    u = u.rename(columns={'bar_date': 'day'})
    u['close'] = pd.to_numeric(u.close, errors='coerce')
    u['adv20'] = pd.to_numeric(u.adv20, errors='coerce')
    u = u.sort_values(['symbol', 'day'], kind='mergesort')
    u['prev_close'] = u.groupby('symbol', sort=False).close.shift(1)
    days = set(bk.day.unique())
    u = u[u.day.isin(days) & u.prev_close.notna() & (u.prev_close > 0) & (u.adv20 > 0)]
    sig = pd.read_csv(f'{ROOT}/research/mature_method/hod_frames2/breaks2.csv',
                      usecols=['day', 'symbol'], dtype=str).drop_duplicates()
    sigset = set(zip(sig.day, sig.symbol))
    u = u[~pd.Series(list(zip(u.day, u.symbol)), index=u.index).isin(sigset)]
    u = attach_instrument(u.assign(symbol=u.symbol.astype(str)))
    u['wrap'] = u.asset_class.astype(str)      # 'stock' | 'wrapper' | 'unknown' — matched exactly
    u['lp'] = np.log(u.prev_close); u['la'] = np.log(u.adv20)
    print(f'  non-signal candidate symbol-days on booked sessions: {len(u)} '
          f'({u.day.nunique()} sessions, median {u.groupby("day").size().median():.0f}/day)',
          flush=True)

    # booked trades need their OWN prev_close / adv20 on the same definition
    ub = pd.read_csv(f'{ROOT}/research/bf_zero/universe.csv',
                     usecols=['symbol', 'bar_date', 'close'], dtype={'symbol': str},
                     keep_default_na=False, na_values=[''])
    ub = ub.rename(columns={'bar_date': 'day'})
    ub['close'] = pd.to_numeric(ub.close, errors='coerce')
    ub = ub.sort_values(['symbol', 'day'], kind='mergesort')
    ub['prev_close'] = ub.groupby('symbol', sort=False).close.shift(1)
    bk = bk.merge(ub[['day', 'symbol', 'prev_close']], on=['day', 'symbol'], how='left')
    bk['wrap'] = bk.asset_class.astype(str)

    UD = {d: g for d, g in u.groupby('day')}
    rows = []
    miss = 0
    for r in bk.itertuples():
        g = UD.get(r.day)
        if g is None or not (r.prev_close == r.prev_close and r.prev_close > 0) or \
           not (r.adv20 == r.adv20 and r.adv20 > 0):
            miss += 1; continue
        gg = g[g.wrap == r.wrap]
        if len(gg) < NPOOL:
            gg = g
        d = (np.abs(gg.lp.values - np.log(r.prev_close)) +
             np.abs(gg.la.values - np.log(r.adv20)))
        idx = np.argsort(d, kind='mergesort')[:NPOOL]
        for rank, k in enumerate(idx):
            rows.append((r.day, r.symbol, int(r.entry_m), gg.symbol.values[k], rank,
                         float(d[k])))
    p = pd.DataFrame(rows, columns=['day', 'symbol', 'entry_m', 'ctrl', 'rank', 'dist'])
    p.to_csv(f'{D6}/pool6.csv', index=False)
    print(f'  pool6.csv {len(p)} rows | booked trades with a pool '
          f'{p.drop_duplicates(["day","symbol","entry_m"]).shape[0]} / {len(bk)} '
          f'(unmatchable {miss}) | median |Δlog| {p.dist.median():.3f}', flush=True)
    print(f'  distinct control symbols to load: '
          f'{p.drop_duplicates(["day","ctrl"]).shape[0]} symbol-days over {p.day.nunique()} days',
          flush=True)


if __name__ == '__main__':
    main()
