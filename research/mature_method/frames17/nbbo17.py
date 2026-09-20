#!/usr/bin/env python3
"""frames17 F52 — measured NBBO for the exit legs `nbbo16.csv` does not already cover.

RUNBOOK step 3 ("measured cost, never the band"), same method as `frames16/nbbo.py`: Alpaca SIP
consolidated quotes, mean/median (ask - bid) over each one-minute window. Only EXIT legs are needed
(entry is charged zero by PREREG §4); legs already measured for frames16's MIR2 population are
reused from `frames16/nbbo16.csv` by `score17.py` directly — this script fills the gap only.
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
from alpaca.data.historical import StockHistoricalDataClient          # noqa: E402
from alpaca.data.requests import StockQuotesRequest                   # noqa: E402
from alpaca.data.enums import DataFeed                                # noqa: E402
from config import Config                                             # noqa: E402

D = f'{ROOT}/research/mature_method/frames17'
D16 = f'{ROOT}/research/mature_method/frames16'
OUT = f'{D}/nbbo17.csv'
ET = ZoneInfo('America/New_York')


def needed_legs():
    d = pd.read_csv(f'{D}/passive17.csv', dtype={'day': str, 'symbol': str})
    f = d[d.filled].copy()
    f['exit_m'] = f.exit_m.astype(int)
    legs = f[['day', 'symbol', 'exit_m']].drop_duplicates()
    q16 = pd.read_csv(f'{D16}/nbbo16.csv', dtype={'day': str, 'symbol': str})
    q16 = q16[q16.n_q > 0].drop_duplicates(['day', 'symbol', 'm'])
    have16 = set(zip(q16.day, q16.symbol, q16.m.astype(int)))
    todo = [tuple(r) for r in legs.itertuples(index=False) if tuple(r) not in have16]
    return sorted(set(todo))


def main():
    todo_all = needed_legs()
    print(f'[nbbo17] {len(todo_all):,} exit legs not already in nbbo16.csv', flush=True)
    done = set()
    if os.path.exists(OUT):
        d0 = pd.read_csv(OUT, dtype={'day': str, 'symbol': str})
        done = set(zip(d0.day, d0.symbol, d0.m.astype(int)))
    todo = [t for t in todo_all if t not in done]
    print(f'[nbbo17] already measured {len(done):,}; todo {len(todo):,}', flush=True)
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
        if i % 200 == 0:
            print(f'  {i+1}/{len(todo)}', flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    if os.path.exists(OUT):
        q = pd.read_csv(OUT, dtype={'day': str, 'symbol': str})
        print(f'[nbbo17] DONE {len(q):,} legs · with quotes {int((q.n_q > 0).sum()):,} '
              f'({(q.n_q > 0).mean()*100:.1f} %)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
