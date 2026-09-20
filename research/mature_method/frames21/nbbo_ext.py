#!/usr/bin/env python3
"""frames21 step 5 — per-trade MEASURED NBBO (Alpaca SIP consolidated quotes) for the entry and exit
legs of the 2024H2 extension mirror-short signals, same convention as frames16/nbbo.py (mean/median
ask-bid over each one-minute window). Population: gate5 (causal membership) & f_mir (the mirror
flag) & price >= $5 & not a leveraged wrapper (attach_instrument) -- same restriction nbbo.py used.
Legs: entry_m, exitm_bare, exitm_tgt, exitm_lock (frames21's own 3 declared exits, armB_intra.py).
Not Databento -- not priced against the $80 cap. Resumable, appends to frames21/nbbo_ext.csv.
"""
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames15')
from common15 import attach_instrument                                # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient          # noqa: E402
from alpaca.data.requests import StockQuotesRequest                   # noqa: E402
from alpaca.data.enums import DataFeed                                # noqa: E402
from config import Config                                             # noqa: E402

D = f'{ROOT}/research/mature_method/frames21'
OUT = f'{D}/nbbo_ext.csv'
ET = ZoneInfo('America/New_York')
LEGS = ['entry_m', 'exitm_bare', 'exitm_tgt', 'exitm_lock']


def population():
    d = pd.read_csv(f'{D}/signals_ext.csv', dtype={'symbol': str, 'day': str},
                    keep_default_na=False, na_values=[''])
    d = d[d.gate5.astype(bool) & d.f_mir.astype(bool) & (d.price >= 5)].copy()
    d = attach_instrument(d)
    d = d[d.asset_class != 'wrapper']
    return d


def main():
    d = population()
    keys = set()
    for c in LEGS:
        keys |= set(zip(d.day, d.symbol, d[c].astype(int)))
    print(f'[nbbo_ext] {len(d):,} trades -> {len(keys):,} distinct (day, symbol, minute) legs',
          flush=True)
    done = set()
    if os.path.exists(OUT):
        d0 = pd.read_csv(OUT, dtype={'day': str, 'symbol': str})
        done = set(zip(d0.day, d0.symbol, d0.m.astype(int)))
    todo = sorted(keys - done)
    print(f'[nbbo_ext] already measured {len(done):,}; todo {len(todo):,}', flush=True)
    cl = StockHistoricalDataClient(Config().alpaca_api_key, Config().alpaca_api_secret)
    rows = []
    for i, (day, sym, m) in enumerate(todo):
        t0 = datetime.strptime(day, '%Y-%m-%d').replace(hour=m // 60, minute=m % 60, tzinfo=ET)
        try:
            q = cl.get_stock_quotes(StockQuotesRequest(
                symbol_or_symbols=sym, start=t0, end=t0 + timedelta(minutes=1),
                feed=DataFeed.SIP, limit=6000)).data.get(sym, [])
            sp = np.array([float(x.ask_price) - float(x.bid_price) for x in q
                           if x.ask_price and x.bid_price and x.ask_price > x.bid_price])
            mid = np.array([(float(x.ask_price) + float(x.bid_price)) / 2.0 for x in q
                            if x.ask_price and x.bid_price and x.ask_price > x.bid_price])
            rows.append(dict(day=day, symbol=sym, m=m,
                             sp_mean=float(sp.mean()) if len(sp) else np.nan,
                             sp_med=float(np.median(sp)) if len(sp) else np.nan,
                             mid_med=float(np.median(mid)) if len(mid) else np.nan,
                             n_q=int(len(sp)), err=''))
        except Exception as e:
            rows.append(dict(day=day, symbol=sym, m=m, sp_mean=np.nan, sp_med=np.nan,
                             mid_med=np.nan, n_q=0, err=str(e)[:60]))
        if len(rows) >= 200:
            pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
            rows = []
        if i % 500 == 0:
            print(f'  {i+1}/{len(todo)}', flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    if os.path.exists(OUT):
        q = pd.read_csv(OUT, dtype={'day': str, 'symbol': str})
        print(f'[nbbo_ext] DONE {len(q):,} legs - with quotes {int((q.n_q > 0).sum()):,} '
              f'({(q.n_q > 0).mean()*100:.1f} %)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
