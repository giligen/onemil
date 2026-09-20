#!/usr/bin/env python3
"""F45 cells E3/E4 — the SIP NBBO at BULL FLAG's OWN detection and exit minutes.

BF's shipped Stage-2 charges a flat 50 bps entry slip (`backtest.py:2622`,
`trading.entry_slippage_pct = 0.005`), spread-blind, and NOTHING on the exit leg. This script
measures what the market actually quoted:

  * the ENTRY minute of all 896 regen-7 raw detections (`data/bull_flag_cache_causal_full_20260905.csv`)
  * the EXIT minute of the 56 P1 trades (`research/bf_frequency/runs/P1.csv`)

Mean ask-bid over the minute, Alpaca SIP, the Stage-P convention. Resumable; appends to f45_bf.csv.

  python3 f45_bf.py
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
OUT = f'{D14}/f45_bf.csv'
RAW = f'{ROOT}/data/bull_flag_cache_causal_full_20260905.csv'
P1 = f'{ROOT}/research/bf_frequency/runs/P1.csv'
ET = ZoneInfo('America/New_York')
FIELDS = ['pop', 'leg', 'day', 'symbol', 'clock_m', 'px', 'sp_mean', 'sp_med', 'n_q', 'err']


def minute_spread(client, sym, day, hh, mm):
    t0 = datetime.strptime(day, '%Y-%m-%d').replace(hour=hh, minute=mm, tzinfo=ET)
    q = client.get_stock_quotes(StockQuotesRequest(
        symbol_or_symbols=sym, start=t0, end=t0 + timedelta(minutes=1),
        feed=DataFeed.SIP, limit=6000)).data.get(sym, [])
    sp = np.array([float(x.ask_price) - float(x.bid_price) for x in q
                   if x.ask_price and x.bid_price and x.ask_price > x.bid_price])
    if not len(sp):
        return np.nan, np.nan, 0
    return float(sp.mean()), float(np.median(sp)), int(len(sp))


def _m(t):
    hh, mm = str(t).split(':')[:2]
    return int(hh) * 60 + int(mm)


def jobs():
    raw = pd.read_csv(RAW, dtype={'symbol': str, 'date': str})
    p1 = pd.read_csv(P1, dtype={'symbol': str, 'date': str})
    j = []
    for r in raw.itertuples():
        j.append(('raw', 'entry', r.date, r.symbol, _m(r.entry_time_et), float(r.entry_price)))
    for r in p1.itertuples():
        j.append(('p1', 'entry', r.date, r.symbol, _m(r.entry_time_et), float(r.entry_price)))
        j.append(('p1', 'exit', r.date, r.symbol, _m(r.exit_time_et), float(r.exit_price)))
    print(f'raw detections {len(raw)} | P1 trades {len(p1)} | pulls {len(j)}', flush=True)
    return j


def main():
    j = jobs()
    done = set()
    if os.path.exists(OUT):
        d0 = pd.read_csv(OUT, dtype={'day': str, 'symbol': str})
        done = set(zip(d0['pop'], d0.leg, d0.day, d0.symbol, d0.clock_m))
    todo = [t for t in j if (t[0], t[1], t[2], t[3], t[4]) not in done]
    print(f'{len(todo)} pulls to do ({len(done)} on disk)', flush=True)
    cfg = Config()
    cl = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    rows, t0 = [], time.time()
    for i, (pop, leg, day, sym, cm, px) in enumerate(todo):
        rec = dict(pop=pop, leg=leg, day=day, symbol=sym, clock_m=cm, px=px,
                   sp_mean=np.nan, sp_med=np.nan, n_q=0, err='')
        try:
            rec['sp_mean'], rec['sp_med'], rec['n_q'] = minute_spread(
                cl, sym, day, cm // 60, cm % 60)
        except Exception as e:
            rec['err'] = str(e)[:110]
        rows.append(rec)
        if len(rows) >= 100 or i == len(todo) - 1:
            pd.DataFrame(rows, columns=FIELDS).to_csv(
                OUT, mode='a', header=not os.path.exists(OUT), index=False)
            rows = []
            print(f'  {i + 1}/{len(todo)}  {time.time() - t0:.0f}s', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
