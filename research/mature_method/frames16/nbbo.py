#!/usr/bin/env python3
"""frames16 ARM 3 — per-trade MEASURED NBBO for EVERY leg of the mirror-short cells.

RUNBOOK step 3: "Measured cost, never the band. Per-trade NBBO at the decision minute." The
350-trade sample in `cost_check.py` found the F45 minute-of-day median is **1.80x too NARROW in the
mean** on this population, which is enough to erase the cell, so the whole population is measured
rather than imputed.

Alpaca SIP consolidated quotes, mean and median (ask - bid) over each one-minute window (the
Stage-P / frames13 / F45 convention). Resumable; one row per distinct (day, symbol, minute).
"""
import glob
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

D = f'{ROOT}/research/mature_method/frames16'
OUT = f'{D}/nbbo16.csv'
ET = ZoneInfo('America/New_York')
LEGS = ['entry_m', 'exitm_a_bare', 'exitm_a_tgt']


def population():
    fs = sorted(glob.glob(f'{D}/sw_*.csv'))
    d = pd.concat([pd.read_csv(f, dtype={'symbol': str, 'day': str},
                               keep_default_na=False, na_values=['']) for f in fs],
                  ignore_index=True)
    d = d[(d.day < '2026-06-01') & d.gate5.astype(bool) & (d.price >= 5)].copy()
    d = attach_instrument(d)
    # Only the MIR2 cells (S1-S4) are measured. Both placebos are already net-NEGATIVE at the
    # IMPUTED (too-narrow) spread, and a wider measured spread can only make them worse, so the
    # signal-vs-placebo ordering cannot be reversed by measuring them; S5/S6 (the hour-high stop)
    # are the weakest cells at the imputed cost and are reported at the imputed cost, labelled.
    d = d[(d.asset_class != 'wrapper') & d.f_mir2.astype(bool)]
    return d


def main():
    d = population()
    keys = set()
    for c in LEGS:
        keys |= set(zip(d.day, d.symbol, d[c].astype(int)))
    print(f'[nbbo] {len(d):,} trades -> {len(keys):,} distinct (day, symbol, minute) legs',
          flush=True)
    done = set()
    if os.path.exists(OUT):
        d0 = pd.read_csv(OUT, dtype={'day': str, 'symbol': str})
        done = set(zip(d0.day, d0.symbol, d0.m.astype(int)))
    todo = sorted(keys - done)
    print(f'[nbbo] already measured {len(done):,}; todo {len(todo):,}', flush=True)
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
    q = pd.read_csv(OUT, dtype={'day': str, 'symbol': str})
    print(f'[nbbo] DONE {len(q):,} legs · with quotes {int((q.n_q > 0).sum()):,} '
          f'({(q.n_q > 0).mean()*100:.1f} %)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
