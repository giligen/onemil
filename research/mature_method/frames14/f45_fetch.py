#!/usr/bin/env python3
"""F45 stage 1 — THE MINUTE-OF-DAY NBBO TABLE on the universe the imputation table was fit on.

`hod_break/score.py::build_impute` fits IMPUTE[(price band, hour band)] on
`research/bf_zero/causal_filter/nbbo.csv`, whose minutes run 577 (09:37) -> 841 (14:01). The table
nevertheless prices 09:30-09:37 (the `0930-0945` cell) and everything out to 16:00 (the `1300+`
cell). This script MEASURES the 14 clocks declared in PREREG §1.2 on the SAME population, so any gap
is a clock effect and not a population effect.

Alpaca SIP consolidated quotes, mean ask-bid over each one-minute window (the Stage-P/frames13
convention). Resumable: appends to f45_minutes.csv, one row per (day, symbol, clock).

  python3 f45_fetch.py
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
from config import Config                                      # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient    # noqa: E402
from alpaca.data.requests import StockQuotesRequest             # noqa: E402
from alpaca.data.enums import DataFeed                          # noqa: E402

D14 = f'{ROOT}/research/mature_method/frames14'
OUT = f'{D14}/f45_minutes.csv'
NBBO = f'{ROOT}/research/bf_zero/causal_filter/nbbo.csv'
ET = ZoneInfo('America/New_York')
SEED = 20260920
PER_BAND = 50                                   # symbol-days per price band

PB_EDGES = [0, 5, 10, 20, 30, 50, 100, 1e9]     # score.py's own bands
PB_LAB = ['<$5', '$5-10', '$10-20', '$20-30', '$30-50', '$50-100', '$100+']

# the 14 declared clocks, as minutes-from-midnight ET
CLOCKS = [(9, 30), (9, 31), (9, 35), (9, 37), (9, 40), (9, 45), (10, 0), (11, 0),
          (12, 0), (13, 0), (14, 0), (15, 0), (15, 45), (15, 55)]
FIELDS = ['day', 'symbol', 'pb', 'price', 'clock_m', 'sp_mean', 'sp_med', 'n_q', 'err']


def minute_spread(client, sym, day, hh, mm):
    """(mean ask-bid, median ask-bid, n quotes) over one minute, SIP."""
    t0 = datetime.strptime(day, '%Y-%m-%d').replace(hour=hh, minute=mm, tzinfo=ET)
    q = client.get_stock_quotes(StockQuotesRequest(
        symbol_or_symbols=sym, start=t0, end=t0 + timedelta(minutes=1),
        feed=DataFeed.SIP, limit=6000)).data.get(sym, [])
    sp = np.array([float(x.ask_price) - float(x.bid_price) for x in q
                   if x.ask_price and x.bid_price and x.ask_price > x.bid_price])
    if not len(sp):
        return np.nan, np.nan, 0
    return float(sp.mean()), float(np.median(sp)), int(len(sp))


def sample():
    """PER_BAND symbol-days per price band, drawn from the imputation table's OWN population."""
    d = pd.read_csv(NBBO, dtype={'day': str, 'symbol': str})
    d = d[d.spread_mean.notna() & (d.n_sig > 0)].copy()
    # the price the table keys on is the signal's `next_open`; nbbo.csv carries ask_dec/bid_dec, so
    # the mid at the signal minute is the available proxy and is used ONLY to pick the band.
    d['price'] = (d.ask_dec + d.bid_dec) / 2.0
    d = d[d.price > 0]
    d['pb'] = pd.cut(d.price, PB_EDGES, labels=PB_LAB)
    d = d.drop_duplicates(['day', 'symbol'])
    rng = np.random.default_rng(SEED)
    out = []
    for lab in PB_LAB:
        w = d[d.pb == lab]
        if not len(w):
            continue
        take = rng.choice(len(w), size=min(PER_BAND, len(w)), replace=False)
        out.append(w.iloc[take])
    s = pd.concat(out).reset_index(drop=True)
    print(f'sample {len(s)} symbol-days over {s.day.nunique()} sessions; per band:\n'
          f'{s.pb.value_counts().reindex(PB_LAB).to_string()}', flush=True)
    return s


def main():
    s = sample()
    done = set()
    if os.path.exists(OUT):
        d0 = pd.read_csv(OUT, dtype={'day': str, 'symbol': str})
        done = set(zip(d0.day, d0.symbol, d0.clock_m))
    todo = [(r, hh, mm) for r in s.itertuples() for hh, mm in CLOCKS
            if (r.day, r.symbol, hh * 60 + mm) not in done]
    print(f'{len(todo)} (symbol-day, clock) pulls to do ({len(done)} already on disk)', flush=True)
    cfg = Config()
    cl = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    rows, t0 = [], time.time()
    for i, (r, hh, mm) in enumerate(todo):
        rec = dict(day=r.day, symbol=r.symbol, pb=str(r.pb), price=float(r.price),
                   clock_m=hh * 60 + mm, sp_mean=np.nan, sp_med=np.nan, n_q=0, err='')
        try:
            rec['sp_mean'], rec['sp_med'], rec['n_q'] = minute_spread(cl, r.symbol, r.day, hh, mm)
        except Exception as e:                                  # loud, never silent
            rec['err'] = str(e)[:110]
        rows.append(rec)
        if len(rows) >= 200 or i == len(todo) - 1:
            pd.DataFrame(rows, columns=FIELDS).to_csv(
                OUT, mode='a', header=not os.path.exists(OUT), index=False)
            rows = []
            print(f'  {i + 1}/{len(todo)}  {time.time() - t0:.0f}s', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
