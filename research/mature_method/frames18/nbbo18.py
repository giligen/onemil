#!/usr/bin/env python3
"""frames18 F55 — measure the NBBO legs neither nbbo16.csv nor nbbo17.csv covers.

Two kinds of leg:
  * EXIT legs of the new grid cells (charged 0.5 * spread / rpct, in R).
  * FILL-MINUTE legs of SSR-active fills (the Reg SHO 201 rail needs the NBB there).

Same Alpaca SIP consolidated-quote method as frames16/nbbo.py.  `bid_med` / `bid_max` are recorded
as DIAGNOSTICS only — the pre-committed NBB is `mid_med - sp_med/2` (PREREG §3a).
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

D = f'{ROOT}/research/mature_method/frames18'
D16 = f'{ROOT}/research/mature_method/frames16'
D17 = f'{ROOT}/research/mature_method/frames17'
OUT = f'{D}/nbbo18.csv'
ET = ZoneInfo('America/New_York')


def needed():
    d = pd.read_csv(f'{D}/grid18.csv', dtype={'day': str, 'symbol': str})
    f = d[d.touch_off >= 0]
    legs = set(zip(f.day, f.symbol, f.exit_m.astype(int)))
    s = f[f.ssr_active]
    legs |= set(zip(s.day, s.symbol, s.fill_m.astype(int)))
    have = set()
    for p in (f'{D16}/nbbo16.csv', f'{D17}/nbbo17.csv'):
        q = pd.read_csv(p, dtype={'day': str, 'symbol': str})
        q = q[q.n_q > 0]
        have |= set(zip(q.day, q.symbol, q.m.astype(int)))
    return sorted(legs - have)


def main():
    todo_all = needed()
    done = set()
    if os.path.exists(OUT):
        d0 = pd.read_csv(OUT, dtype={'day': str, 'symbol': str})
        done = set(zip(d0.day, d0.symbol, d0.m.astype(int)))
    todo = [t for t in todo_all if t not in done]
    print(f'[nbbo18] {len(todo_all):,} uncovered legs; already measured {len(done):,}; '
          f'todo {len(todo):,}', flush=True)
    cl = StockHistoricalDataClient(Config().alpaca_api_key, Config().alpaca_api_secret)
    rows = []
    for i, (day, sym, m) in enumerate(todo):
        t0 = datetime.strptime(day, '%Y-%m-%d').replace(hour=m // 60, minute=m % 60, tzinfo=ET)
        try:
            q = cl.get_stock_quotes(StockQuotesRequest(
                symbol_or_symbols=sym, start=t0, end=t0 + timedelta(minutes=1),
                feed=DataFeed.SIP, limit=6000)).data.get(sym, [])
            ok = [x for x in q if x.ask_price and x.bid_price and x.ask_price > x.bid_price]
            sp = np.array([float(x.ask_price) - float(x.bid_price) for x in ok])
            mid = np.array([(float(x.ask_price) + float(x.bid_price)) / 2.0 for x in ok])
            bid = np.array([float(x.bid_price) for x in ok])
            rows.append(dict(day=day, symbol=sym, m=m,
                             sp_mean=float(sp.mean()) if len(sp) else np.nan,
                             sp_med=float(np.median(sp)) if len(sp) else np.nan,
                             mid_med=float(np.median(mid)) if len(mid) else np.nan,
                             bid_med=float(np.median(bid)) if len(bid) else np.nan,
                             bid_max=float(bid.max()) if len(bid) else np.nan,
                             n_q=int(len(sp)), err=''))
        except Exception as e:
            rows.append(dict(day=day, symbol=sym, m=m, sp_mean=np.nan, sp_med=np.nan,
                             mid_med=np.nan, bid_med=np.nan, bid_max=np.nan, n_q=0,
                             err=str(e)[:60]))
        if len(rows) >= 150:
            pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
            rows = []
        if i % 100 == 0:
            print(f'  {i+1}/{len(todo)}', flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    q = pd.read_csv(OUT, dtype={'day': str, 'symbol': str})
    print(f'[nbbo18] DONE {len(q):,} legs · with quotes {int((q.n_q > 0).sum()):,} '
          f'({(q.n_q > 0).mean()*100:.1f}%)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
