#!/usr/bin/env python3
"""Stage P — cross-check: Databento EQUS.MINI venue BBO vs Alpaca SIP NBBO.

The entry-minute `tbbo` is already on disk (Stage N1, 09:29:30-09:46 ET).  Its
`bid_px_00/ask_px_00` is ONE publisher's book, not the consolidated NBBO, so it
cannot be the measurement.  This quantifies by how much, on the overlap, and is
the reason Stage P bought no new Databento data.

Usage: python3 crosscheck_dbn.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

P = f'{ROOT}/research/fuckup_audit/P_cost'
TBBO = f'{ROOT}/research/fuckup_audit/N_databento/N1/tbbo'


def main() -> int:
    sp = pd.read_parquet(f'{P}/spreads.parquet',
                         columns=['symbol', 'date', 'entry_price', 'range_high',
                                  'entry_fill_ts', 'entry_spread', 'entry_bps'])
    sp = sp[sp.entry_spread.notna()].copy()
    sp['entry_fill_ts'] = pd.to_datetime(sp.entry_fill_ts, utc=True)
    rows = []
    for day, g in sp.groupby('date'):
        f = f'{TBBO}/{day}.parquet'
        if not os.path.exists(f):
            continue
        t = pd.read_parquet(f, columns=['ts_event', 'symbol', 'price',
                                        'bid_px_00', 'ask_px_00'])
        t = t[t.symbol.isin(set(g.symbol))]
        if not len(t):
            continue
        for r in g.itertuples():
            s = t[(t.symbol == r.symbol) & (t.price > r.range_high)]
            if not len(s):
                continue
            s = s.sort_values('ts_event').iloc[0]
            b, a = float(s.bid_px_00), float(s.ask_px_00)
            if not (b > 0 and a >= b):
                continue
            rows.append(dict(symbol=r.symbol, date=day, price=r.entry_price,
                             nbbo_bps=r.entry_bps,
                             venue_bps=(a - b) / r.entry_price * 1e4))
    d = pd.DataFrame(rows)
    d.to_csv(f'{P}/crosscheck_dbn.csv', index=False)
    if len(d):
        d['ratio'] = d.venue_bps / d.nbbo_bps.replace(0, np.nan)
        print(f'overlap n={len(d)}  median NBBO {d.nbbo_bps.median():.1f} bps  '
              f'median venue-BBO {d.venue_bps.median():.1f} bps  '
              f'median ratio {d.ratio.median():.2f}x  '
              f'p90 ratio {d.ratio.quantile(0.9):.2f}x', flush=True)
    else:
        print('no overlap rows', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
