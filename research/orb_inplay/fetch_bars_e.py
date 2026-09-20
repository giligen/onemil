#!/usr/bin/env python3
"""orb_inplay Cell E — 09:35-15:55 ET 1-min bars for the picks_e.parquet symbols.

Mirrors fetch_bars.py exactly but reads picks_e.parquet and writes to a SEPARATE
file daybars_e.parquet (never touches the base book's daybars.parquet).
"""
import os, sys, time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment
from config import Config
ET = ZoneInfo('America/New_York'); D = f'{ROOT}/research/orb_inplay'; OUT = f'{D}/daybars_e.parquet'
pk = pd.read_parquet(f'{D}/picks_e.parquet')
pk = pk[pk.bar_date >= '2025-01-01']
days = sorted(pk.bar_date.unique())
done = set()
parts = []
if os.path.exists(OUT):
    p0 = pd.read_parquet(OUT); parts.append(p0); done = set(p0.bar_date.unique())
cl = StockHistoricalDataClient(Config().alpaca_api_key, Config().alpaca_api_secret)
t0 = time.time()
for k, day in enumerate(days):
    if day in done: continue
    syms = sorted(pk[pk.bar_date == day].symbol.unique())
    s = datetime.strptime(day, '%Y-%m-%d').replace(hour=9, minute=35, tzinfo=ET)
    for att in range(3):
        try:
            df = cl.get_stock_bars(StockBarsRequest(symbol_or_symbols=syms,
                timeframe=TimeFrame.Minute, start=s, end=s + timedelta(hours=6, minutes=21),
                feed=DataFeed.SIP, adjustment=Adjustment.RAW, limit=40000)).df
            if len(df):
                df = df.reset_index()
                df['ts'] = pd.to_datetime(df.timestamp, utc=True).dt.tz_convert(ET)
                df['m'] = df.ts.dt.hour * 60 + df.ts.dt.minute
                df = df[(df.m >= 575) & (df.m <= 955)]
                df['bar_date'] = day
                parts.append(df[['bar_date', 'symbol', 'm', 'open', 'high', 'low', 'close', 'volume']])
            break
        except Exception as e:
            if att == 2: print(f'  ERR {day}: {str(e)[:80]}', flush=True)
            time.sleep(2)
    if k % 25 == 0:
        pd.concat(parts, ignore_index=True).to_parquet(OUT, index=False)
        print(f'  {k+1}/{len(days)} {day} {time.time()-t0:.0f}s', flush=True)
pd.concat(parts, ignore_index=True).to_parquet(OUT, index=False)
print(f'[bars_e] DONE {time.time()-t0:.0f}s', flush=True)
