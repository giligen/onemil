#!/usr/bin/env python3
"""hod_frames5 / F18 — the DEDICATED per-trade NBBO fetch (PREREG §4.3).

Same source, same convention, same two causal instants as
`research/bf_zero/causal_filter/fetch_nbbo.py` (Alpaca **SIP** consolidated quotes):
  * the SIGNAL minute (`entry_m - 1`) -> `spread_mean` (MEAN of ask-bid over the minute), `n_sig`
  * the DECISION instant (the open of `entry_m`) -> `ask_dec` / `bid_dec`, the LAST quote at or
    before it, for the capped-limit obtainability test.

Input: `nbbo5_todo.csv` (written by build_todo5.py).  Resumable; appends to
`nbbo5.csv`.  Read-only on the repo; writes only inside this directory.
"""
import os, sys, time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
from config import Config                                     # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockQuotesRequest           # noqa: E402
from alpaca.data.enums import DataFeed                        # noqa: E402

D = f'{ROOT}/research/mature_method/hod_frames5'
OUT = f'{D}/nbbo5.csv'
ET = ZoneInfo('America/New_York')
FIELDS = ['day', 'symbol', 'entry_m', 'n_sig', 'spread_mean', 'spread_med',
          'ask_dec', 'bid_dec', 'n_dec', 'err']
LIMIT = int(os.environ.get('MM_FETCH_LIMIT', '0'))

c = pd.read_csv(f'{D}/nbbo5_todo.csv', dtype={'symbol': str, 'day': str},
                keep_default_na=False, na_values=[''])
done = set()
if os.path.exists(OUT):
    d0 = pd.read_csv(OUT, dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
    done = set(zip(d0.day, d0.symbol, d0.entry_m))
todo = [r for r in c.itertuples() if (r.day, r.symbol, int(r.entry_m)) not in done]
if LIMIT:
    todo = todo[:LIMIT]
print(f'rows {len(c)} | already fetched {len(done)} | todo {len(todo)}', flush=True)

cfg = Config()
client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
rows, t0 = [], time.time()
for i, r in enumerate(todo):
    sig_m = int(r.entry_m) - 1
    start = datetime.strptime(r.day, '%Y-%m-%d').replace(tzinfo=ET) + timedelta(minutes=sig_m)
    rec = dict(day=r.day, symbol=r.symbol, entry_m=int(r.entry_m), n_sig=0, spread_mean=np.nan,
               spread_med=np.nan, ask_dec=np.nan, bid_dec=np.nan, n_dec=0, err='')
    try:
        q = client.get_stock_quotes(StockQuotesRequest(
            symbol_or_symbols=r.symbol, start=start, end=start + timedelta(minutes=1),
            feed=DataFeed.SIP, limit=6000)).data.get(r.symbol, [])
        sp = np.array([float(x.ask_price) - float(x.bid_price) for x in q
                       if x.ask_price and x.bid_price and x.ask_price > x.bid_price])
        if len(sp):
            rec['spread_mean'] = float(sp.mean()); rec['spread_med'] = float(np.median(sp))
            rec['n_sig'] = int(len(sp))
        last = [x for x in q if x.ask_price and x.bid_price]
        if not last:
            q2 = client.get_stock_quotes(StockQuotesRequest(
                symbol_or_symbols=r.symbol, start=start - timedelta(minutes=2), end=start,
                feed=DataFeed.SIP, limit=6000)).data.get(r.symbol, [])
            last = [x for x in q2 if x.ask_price and x.bid_price]
        if last:
            rec['ask_dec'] = float(last[-1].ask_price)
            rec['bid_dec'] = float(last[-1].bid_price)
            rec['n_dec'] = len(last)
    except Exception as e:                                          # loud, never silent
        rec['err'] = type(e).__name__
        print(f'  WARN {r.day} {r.symbol}: {type(e).__name__} {e}', flush=True)
    rows.append(rec)
    if len(rows) >= 100 or i == len(todo) - 1:
        pd.DataFrame(rows)[FIELDS].to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
        rows = []
        el = time.time() - t0
        print(f'{i + 1}/{len(todo)} {el / 60:.1f} min ({(i + 1) / max(el, 1) * 60:.0f}/min)',
              flush=True)
print('DONE', flush=True)
