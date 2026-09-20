#!/usr/bin/env python3
"""ORB short mirror — measured NBBO per (day, symbol, minute). PREREG §3.

Alpaca SIP consolidated quotes, mean/median (ask-bid) over the one-minute window
(the frames16 nbbo.py convention). Resumable.  Usage: nbbo.py <legs.csv>
legs.csv columns: day,symbol,m
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
from alpaca.data.historical import StockHistoricalDataClient   # noqa: E402
from alpaca.data.requests import StockQuotesRequest            # noqa: E402
from alpaca.data.enums import DataFeed                         # noqa: E402
from config import Config                                      # noqa: E402

D = f'{ROOT}/research/orb_short'
OUT = f'{D}/nbbo_short.csv'
ET = ZoneInfo('America/New_York')


def main():
    legs = pd.read_csv(sys.argv[1], dtype={'day': str, 'symbol': str},
                       keep_default_na=False, na_values=[''])
    keys = sorted(set(zip(legs.day, legs.symbol, legs.m.astype(int))))
    done = set()
    if os.path.exists(OUT):
        d0 = pd.read_csv(OUT, dtype={'day': str, 'symbol': str},
                         keep_default_na=False, na_values=[''])
        done = set(zip(d0.day, d0.symbol, d0.m.astype(int)))
    todo = [k for k in keys if k not in done]
    print(f'[nbbo] {len(keys):,} legs, {len(done):,} done, {len(todo):,} todo', flush=True)
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
        if i % 250 == 0:
            print(f'  {i+1}/{len(todo)}', flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    q = pd.read_csv(OUT, dtype={'day': str, 'symbol': str})
    print(f'[nbbo] DONE {len(q):,} legs · with quotes {int((q.n_q > 0).sum()):,}', flush=True)


if __name__ == '__main__':
    main()
