#!/usr/bin/env python3
"""hod_preopen_regime — data acquisition B: the Nasdaq opening-auction imbalance (NOII) for QQQ.

Dataset XNAS.ITCH, schema `imbalance`. QQQ is NASDAQ-listed so its opening cross is on this feed.
SPY is NYSE-Arca listed and is NOT disseminated on XNAS.ITCH -- stated in REPORT.md, not imputed.

Priced with metadata.get_cost BEFORE the pull; the pull aborts if the quote exceeds CAP_USD.
The API key is read from .env via load_dotenv and is never printed.

Output: qqq_imbalance.csv -- every raw imbalance message with its RAW ts_event, so the
availability assertion ("strictly before 09:30:00 ET") is made on the raw timestamp.
"""
import os, sys
from zoneinfo import ZoneInfo

import pandas as pd
import databento as db
from dotenv import load_dotenv

ROOT = '/home/ec2-user/onemil'
D = f'{ROOT}/research/mature_method/hod_preopen_regime'
OUT = f'{D}/qqq_imbalance.csv'
CAP_USD = 40.0
START, END = '2025-01-02', '2026-09-12'
ET = ZoneInfo('America/New_York')

load_dotenv(f'{ROOT}/.env')
cl = db.Historical(os.getenv('DATABENTO_API_KEY'))

kw = dict(dataset='XNAS.ITCH', schema='imbalance', start=START, end=END)
cost_qqq = cl.metadata.get_cost(symbols=['QQQ'], **kw)
cost_all = cl.metadata.get_cost(symbols='ALL_SYMBOLS', **kw)
print(f'PRICE  QQQ-only        ${cost_qqq:.4f}')
print(f'PRICE  ALL_SYMBOLS     ${cost_all:.2f}   (the breadth version)')
print(f'CAP    ${CAP_USD:.2f}')
if cost_qqq > CAP_USD:
    print('QQQ pull ALSO over the cap -- NOT pulled.'); sys.exit(1)
if cost_all > CAP_USD:
    print('breadth version OVER the cap -- NOT pulled; QQQ-only proceeds.')

if os.path.exists(OUT):
    print(f'{OUT} exists -- nothing pulled.'); sys.exit(0)

d = cl.timeseries.get_range(symbols=['QQQ'], **kw)
df = d.to_df()
df = df.reset_index()
print(f'pulled {len(df)} imbalance messages; columns {list(df.columns)}')
ts = pd.to_datetime(df['ts_event'] if 'ts_event' in df.columns else df['ts_recv'], utc=True)
df['ts_event_utc'] = ts.astype(str)
loc = ts.dt.tz_convert(ET)
df['day'] = loc.dt.strftime('%Y-%m-%d')
df['m_et'] = loc.dt.hour * 60 + loc.dt.minute
df['sec_et'] = loc.dt.second
df.to_csv(OUT, index=False)
print(f'wrote {OUT}: {len(df)} rows, {df.day.nunique()} sessions')
