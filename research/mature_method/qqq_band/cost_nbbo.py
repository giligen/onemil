#!/usr/bin/env python3
"""Measured cost for the QQQ noise-band sleeve — Alpaca SIP NBBO at the book's own leg instants.

For each sampled leg the modelled fill is the OPEN of bar k (the live convention: decide on the
close of bar k-1, market order, fill at the next bar's open).  We fetch the first NBBO quote at or
after that instant and charge what a marketable order would ACTUALLY have paid relative to the
modelled fill:  buy -> (ask - px)/px ;  sell -> (px - bid)/px , in bp.

TEST (2026) is sealed: no leg from 2026 is sampled.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest

HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
N_PER_SPLIT = int(sys.argv[1]) if len(sys.argv) > 1 else 250

load_dotenv()
cli = StockHistoricalDataClient(os.getenv('ALPACA_API_KEY'),
                                os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_API_SECRET'))

legs = pd.read_csv(HERE + 'legs.csv')
legs['date'] = pd.to_datetime(legs['date'])
legs['ts'] = (legs['date'] + pd.Timedelta(minutes=570) + pd.to_timedelta(legs['k'], 'm'))
legs['ts'] = legs['ts'].dt.tz_localize('America/New_York', nonexistent='shift_forward',
                                       ambiguous=True).dt.tz_convert('UTC')
rng = np.random.default_rng(20260919)
samp = []
for name, lo, hi in (('TRAIN', '2016-01-01', '2023-12-31'), ('VAL', '2024-01-01', '2025-12-31')):
    s = legs[(legs['date'] >= lo) & (legs['date'] <= hi)]
    take = s.iloc[rng.choice(len(s), size=min(N_PER_SPLIT, len(s)), replace=False)].copy()
    take['split'] = name
    samp.append(take)
S = pd.concat(samp, ignore_index=True)
print(f'sampling {len(S)} legs of {len(legs)}', flush=True)

rows = []
for i, r in S.iterrows():
    t0 = r['ts']
    try:
        q = cli.get_stock_quotes(StockQuotesRequest(symbol_or_symbols='QQQ', start=t0,
                                                    end=t0 + pd.Timedelta(seconds=2),
                                                    limit=1, feed='sip'))
        d = q.data.get('QQQ', [])
    except Exception as e:                                        # noqa: BLE001
        print(f'  WARNING quote fetch failed {t0}: {e}', flush=True)
        d = []
    if not d:
        rows.append(dict(split=r['split'], date=str(r['date'])[:10], k=r['k'], side=r['side'],
                         px=r['px'], bid=np.nan, ask=np.nan, half_bp=np.nan, cost_bp=np.nan))
        continue
    b, a = float(d[0].bid_price), float(d[0].ask_price)
    mid = 0.5 * (a + b)
    px = float(r['px'])
    half = (a - b) / 2.0 / mid * 1e4 if mid > 0 else np.nan
    cost = ((a - px) if r['side'] > 0 else (px - b)) / px * 1e4
    rows.append(dict(split=r['split'], date=str(r['date'])[:10], k=r['k'], side=r['side'],
                     px=px, bid=b, ask=a, half_bp=half, cost_bp=cost))
    if len(rows) % 50 == 0:
        print(f'  {len(rows)}/{len(S)}', flush=True)
    time.sleep(0.05)

Q = pd.DataFrame(rows)
Q.to_csv(HERE + 'cost_nbbo.csv', index=False)
ok = Q[Q['cost_bp'].notna()]
print(f'\ncoverage {len(ok)}/{len(Q)}')
for sp in ('TRAIN', 'VAL'):
    o = ok[ok['split'] == sp]
    if len(o) == 0:
        continue
    print(f'{sp}: n={len(o)}  half-spread mean {o["half_bp"].mean():.3f} bp median '
          f'{o["half_bp"].median():.3f}  |  cost vs modelled fill mean {o["cost_bp"].mean():+.3f} bp '
          f'median {o["cost_bp"].median():+.3f} p90 {o["cost_bp"].quantile(0.9):+.3f}')
print(f'ALL: half-spread mean {ok["half_bp"].mean():.3f} bp | cost mean {ok["cost_bp"].mean():+.3f} bp')
