#!/usr/bin/env python3
"""F40 — the MEASURED spread at the two instants leg (b) trades, and at the opening cross.

The frame forbids assuming the open is free ("opening auctions are illiquid", JFQA 2026).  So:
for a stratified random sample of eligible name-nights we fetch Alpaca **SIP** quotes for

  * 15:55-15:56 ET of day t   — where leg (b) BUYS marketable
  * 09:30-09:31 ET of day t+1 — the first RTH minute, the opening cross's own liquidity
  * 09:31-09:32 ET of day t+1 — where leg (b) SELLS marketable

and record the MEAN ask-bid over each minute (the Stage-P convention), as a % of the day-t close.
`rt_pct` = the round-trip quoted cost leg (b) pays = half-spread at each end.

Resumable: appends to `openspread.csv`.  Read-only on every store.
"""
import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from config import Config                                     # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient   # noqa: E402
from alpaca.data.requests import StockQuotesRequest            # noqa: E402
from alpaca.data.enums import DataFeed                         # noqa: E402

sys.path.insert(0, f'{ROOT}/research/mature_method/frames13')
from f40 import load, sub, PB_LAB, AB_LAB, sym_of               # noqa: E402

D13 = f'{ROOT}/research/mature_method/frames13'
OUT = f'{D13}/openspread.csv'
ET = ZoneInfo('America/New_York')
SEED = 20260920
PER_CELL = 25          # name-nights per (price band x ADV band) cell, 2025+2026 only
FIELDS = ['day', 'next_day', 'symbol', 'close', 'pb', 'ab', 'sp_1555', 'sp_0930', 'sp_0931',
          'rt_pct', 'open_cross_pct', 'err']


def minute_spread(client, sym, day, hh, mm):
    """MEAN ask-bid over one minute, SIP.  Returns (mean_spread, n_quotes)."""
    t0 = datetime.strptime(day, '%Y-%m-%d').replace(hour=hh, minute=mm, tzinfo=ET)
    q = client.get_stock_quotes(StockQuotesRequest(
        symbol_or_symbols=sym, start=t0, end=t0 + timedelta(minutes=1),
        feed=DataFeed.SIP, limit=6000)).data.get(sym, [])
    sp = np.array([float(x.ask_price) - float(x.bid_price) for x in q
                   if x.ask_price and x.bid_price and x.ask_price > x.bid_price])
    return (float(sp.mean()), len(sp)) if len(sp) else (np.nan, 0)


def main():
    X = load()
    X = sub(X, (X['era_i'] > 0) & (X['pb_i'] >= 0) & (X['ab_i'] >= 0))
    rng = np.random.default_rng(SEED)
    idx = []
    for pi in range(len(PB_LAB)):
        for ai in range(len(AB_LAB)):
            w = np.flatnonzero((X['pb_i'] == pi) & (X['ab_i'] == ai))
            if not len(w):
                continue
            idx.append(rng.choice(w, size=min(PER_CELL, len(w)), replace=False))
    idx = np.concatenate(idx)
    P = sub(X, idx)
    del X
    dt = P['d'].astype('datetime64[D]')
    p = pd.DataFrame(dict(
        date=pd.to_datetime(dt).strftime('%Y-%m-%d'),
        symbol=[sym_of(v) for v in P['sid']],
        close=P['close'],
        next_day=(pd.to_datetime(dt) + pd.to_timedelta(P['nights'], unit='D')).strftime('%Y-%m-%d'),
        pb=[PB_LAB[i] for i in P['pb_i']],
        ab=[AB_LAB[i] for i in P['ab_i']]))

    done = set()
    if os.path.exists(OUT):
        d0 = pd.read_csv(OUT, dtype={'day': str, 'symbol': str})
        done = set(zip(d0.day, d0.symbol))
    todo = [r for r in p.itertuples() if (r.date, r.symbol) not in done]
    print(f'sample {len(p)} name-nights | already done {len(done)} | todo {len(todo)}', flush=True)

    cfg = Config()
    cl = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    rows = []
    t0 = time.time()
    for i, r in enumerate(todo):
        rec = dict(day=r.date, next_day=r.next_day, symbol=r.symbol, close=float(r.close),
                   pb=str(r.pb), ab=str(r.ab), sp_1555=np.nan, sp_0930=np.nan, sp_0931=np.nan,
                   rt_pct=np.nan, open_cross_pct=np.nan, err='')
        try:
            rec['sp_1555'], _ = minute_spread(cl, r.symbol, r.date, 15, 55)
            rec['sp_0930'], _ = minute_spread(cl, r.symbol, r.next_day, 9, 30)
            rec['sp_0931'], _ = minute_spread(cl, r.symbol, r.next_day, 9, 31)
            px = float(r.close)
            rec['rt_pct'] = (0.5 * rec['sp_1555'] + 0.5 * rec['sp_0931']) / px * 100.0
            rec['open_cross_pct'] = rec['sp_0930'] / px * 100.0
        except Exception as e:
            rec['err'] = str(e)[:120]
        rows.append(rec)
        if len(rows) >= 20 or i == len(todo) - 1:
            pd.DataFrame(rows, columns=FIELDS).to_csv(
                OUT, mode='a', header=not os.path.exists(OUT), index=False)
            rows = []
            print(f'  {i + 1}/{len(todo)}  {time.time() - t0:.0f}s', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
