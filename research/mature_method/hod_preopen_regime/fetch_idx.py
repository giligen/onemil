#!/usr/bin/env python3
"""hod_preopen_regime — data acquisition A: SPY and QQQ 1-minute bars WITH extended hours.

Source: Alpaca SIP consolidated 1-min bars (free on the account subscription). Extended hours are
included by the v2 bars endpoint, so the 04:00-09:29 ET premarket prints and the exact 09:30 RTH
opening bar are both present. We keep the raw UTC timestamp of every bar so that the availability
assertion in REPORT.md is made on the RAW timestamp, not on a derived minute index.

Output: idx_1min.csv  (symbol, day, ts_utc, m_et, o, h, l, c, v)
  m_et = minutes from ET midnight (570 = 09:30, 575 = 09:35) -- derived, but ts_utc is kept beside it.

Resumable by month. Read-only w.r.t. every project DB; writes only inside this directory.
"""
import os, sys, time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
from config import Config                                      # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient   # noqa: E402
from alpaca.data.requests import StockBarsRequest              # noqa: E402
from alpaca.data.timeframe import TimeFrame                    # noqa: E402
from alpaca.data.enums import DataFeed, Adjustment             # noqa: E402

D = f'{ROOT}/research/mature_method/hod_preopen_regime'
OUT = f'{D}/idx_1min.csv'
ET = ZoneInfo('America/New_York')
SYMS = ['SPY', 'QQQ']
START, END = '2024-12-30', '2026-09-12'          # T-1 needed for the 2025-01-02 gap

done_months = set()
if os.path.exists(OUT):
    d0 = pd.read_csv(OUT, dtype={'symbol': str, 'day': str}, usecols=['day'])
    done_months = set(d0.day.str[:7])

months = [str(p) for p in pd.period_range(START, END, freq='M')]
todo = [m for m in months if m not in done_months]
print(f'months {len(months)} | already fetched {len(done_months)} | todo {len(todo)}', flush=True)

cfg = Config()
cl = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
hdr = not os.path.exists(OUT)
t0 = time.time()
for i, mo in enumerate(todo):
    p = pd.Period(mo, freq='M')
    s = datetime.combine(p.start_time.date(), datetime.min.time()).replace(tzinfo=ET)
    e = datetime.combine(p.end_time.date(), datetime.min.time()).replace(tzinfo=ET) + timedelta(days=1)
    bars = cl.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=SYMS, timeframe=TimeFrame.Minute, start=s, end=e,
        feed=DataFeed.SIP, adjustment=Adjustment.RAW)).data
    rows = []
    for sym, bl in bars.items():
        for b in bl:
            ts = b.timestamp                       # tz-aware UTC
            l = ts.astimezone(ET)
            rows.append(dict(symbol=sym, day=l.strftime('%Y-%m-%d'), ts_utc=ts.isoformat(),
                             m_et=l.hour * 60 + l.minute, o=b.open, h=b.high, l=b.low,
                             c=b.close, v=b.volume))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, mode='a', header=hdr, index=False)
    hdr = False
    print(f'  [{i+1}/{len(todo)}] {mo}: {len(df)} bars  ({time.time()-t0:.0f}s)', flush=True)
print('DONE', flush=True)
