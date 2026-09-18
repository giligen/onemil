#!/usr/bin/env python3
"""Stage N3 step 3a — the paper's stocks-in-play selection, causal at 09:35 ET.

RelativeVolume(t,j) = ORVolume(t,j) / mean_{i=1..14} ORVolume(t-i,j), ORVolume = 09:30-09:35 volume.
Keep RV >= 1.0, rank DESC, take the top 20 of the eligible pool.  Direction is the colour of the
first 5-min candle (up -> long, down -> short, doji -> no order but the slot is still spent).

Every input is known at 09:35: the daily filters are T-1, the window volume and the candle are the
first five minutes of the session itself.

Writes N3/top20.csv and N3/rank_sensitivity.md.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')

N3 = 'research/fuckup_audit/N_databento/N3'
DB = f'{N3}/tape.db'
POOL = f'{N3}/pool.parquet'
START, END = '2019-01-01', '2023-12-31'


def load_window() -> pd.DataFrame:
    """RV per symbol-day: ORVolume / the mean of the previous 14 same-window volumes."""
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    w = pd.read_sql('select day, symbol, o5, h5, l5, c5, v5, nbars from window', con)
    con.close()
    w['symbol'] = w.symbol.astype('category')
    for c in ('o5', 'h5', 'l5', 'c5', 'v5'):
        w[c] = w[c].astype('float32')
    w['nbars'] = w.nbars.astype('int8')
    w = w.sort_values(['symbol', 'day'], ignore_index=True)
    g = w.groupby('symbol', observed=True, sort=False)
    w['base'] = g.v5.shift(1).rolling(14, min_periods=14).mean().to_numpy(dtype='float32')
    w['rv'] = (w.v5 / w.base).astype('float32')
    # only rows that can be picked survive -- the paper needs RV >= 1 and a 09:30 open over $5
    w = w[(w.rv >= 1.0) & w.rv.notna() & (w.o5 > 5.0) & (w.nbars >= 1)
          & (w.day >= START) & (w.day <= END)].copy()
    w['symbol'] = w.symbol.astype(str)
    return w.reset_index(drop=True)


def build(w: pd.DataFrame, adv_min: float, top_n: int = 20) -> pd.DataFrame:
    pool = pd.read_parquet(POOL, columns=['bar_date', 'symbol', 'prev_close', 'adv14', 'atr14'])
    pool['symbol'] = pool.symbol.astype(str)
    pool = pool[pool.adv14 >= adv_min]
    j = w.merge(pool, left_on=['day', 'symbol'], right_on=['bar_date', 'symbol'], how='inner')
    j['rank'] = j.groupby('day').rv.rank(ascending=False, method='first')
    top = j[j['rank'] <= top_n].copy()
    top['side'] = np.where(top.c5 > top.o5, 1, np.where(top.c5 < top.o5, -1, 0))
    return top[['day', 'symbol', 'rv', 'v5', 'base', 'o5', 'h5', 'l5', 'c5', 'side',
                'prev_close', 'adv14', 'atr14', 'rank']].sort_values(['day', 'rank'])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--adv-min', type=float, default=118424.0)
    args = ap.parse_args()

    w = load_window()
    print(f'window rows eligible: {len(w):,}', flush=True)
    top = build(w, args.adv_min)
    top.to_csv(f'{N3}/top20.csv', index=False)
    print(f'top20: {len(top):,} picks, {top.day.nunique()} sessions, '
          f'{top.symbol.nunique():,} symbols', flush=True)
    print(f'side: long {int((top.side == 1).sum()):,} short {int((top.side == -1).sum()):,} '
          f'doji {int((top.side == 0).sum()):,}', flush=True)
    print(f'rv: median {top.rv.median():.2f}, p90 {top.rv.quantile(.9):.1f}, '
          f'max {top.rv.max():.0f}', flush=True)

    lines = ['# Top-20 selection sensitivity to the ADV translation', '',
             'The paper filters on 1M CONSOLIDATED shares; we hold Nasdaq-venue volume only, so the',
             'threshold is scaled by the measured venue share.  Overlap = share of the primary',
             'top-20 picks that also appear in the alternative threshold\'s top-20 for the same day.',
             '', '| ADV_min (XNAS shares) | implied consolidated | picks | overlap with primary |',
             '|---|---|---|---|']
    base_keys = set(zip(top.day, top.symbol))
    for adv, impl in [(118424, '1.0M @ share .118'), (200000, '1.7M @ share .118 / 1.0M @ .20'),
                      (300000, '2.5M @ share .118 / 1.0M @ .30')]:
        t = build(w, float(adv))
        keys = set(zip(t.day, t.symbol))
        ov = len(base_keys & keys) / max(len(base_keys), 1) * 100
        lines.append(f'| {adv:,} | {impl} | {len(t):,} | {ov:.1f}% |')
    open(f'{N3}/rank_sensitivity.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines[5:]), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
