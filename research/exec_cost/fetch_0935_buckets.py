#!/usr/bin/env python3
"""Diagnostic (owner add-on, not a pre-registered cell): NBBO half-spread by 10-second
bucket across 09:34:00-09:36:00 ET, pooled over the same 122 TRAIN+VAL ORB fills used for
cells 1,289/1,290. Answers: is the 09:35:00-09:35:10 bucket (where every ORB algo crosses)
worse than the buckets around it, and by how much in bps.
"""
from __future__ import annotations
import os, sys, time
from datetime import timedelta, time as dtime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
load_dotenv(f'{ROOT}/.env')

from config import Config
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest
from alpaca.data.enums import DataFeed

ET = ZoneInfo('America/New_York')
OUT_DIR = f'{ROOT}/research/exec_cost'
BUCKET_OUT = f'{OUT_DIR}/bucket_rows.csv'
QLIMIT = 4000

cfg = Config()
client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)


def load_pop():
    book = pd.read_csv(f'{ROOT}/analysis_results/orb_bplus_book.csv',
                        usecols=['symbol', 'date', 'entry_price', 'entered'])
    book = book[book.entered == 1].copy()
    book['date'] = pd.to_datetime(book.date)
    sp = pd.read_parquet(f'{ROOT}/research/fuckup_audit/P_cost/spreads.parquet',
                          columns=['symbol', 'date', 'entry_fill_ts', 'cov_entry',
                                   'entry_ask', 'entry_bid'])
    sp['date'] = pd.to_datetime(sp.date)
    m = book.merge(sp, on=['symbol', 'date'])
    m = m[m.cov_entry].copy()
    bins = [pd.Timestamp('2000-01-01'), pd.Timestamp('2025-12-31'),
            pd.Timestamp('2026-05-31'), pd.Timestamp('2099-01-01')]
    m['split'] = pd.cut(m.date, bins=bins, labels=['TRAIN', 'VAL', 'TEST'])
    m = m[m.split.isin(['TRAIN', 'VAL'])].copy()
    m['entry_fill_ts'] = pd.to_datetime(m.entry_fill_ts, utc=True)
    return m.reset_index(drop=True)


def bucket_of(ts_et):
    """10s bucket index 0..11 covering 09:34:00 (0) .. 09:35:50 (11); None if outside."""
    secs = (ts_et.hour * 3600 + ts_et.minute * 60 + ts_et.second) - (9 * 3600 + 34 * 60)
    if secs < 0 or secs >= 120:
        return None
    return secs // 10


def main():
    m = load_pop()
    print(f'population: {len(m)} rows', flush=True)
    rows = []
    if os.path.exists(BUCKET_OUT):
        prev = pd.read_csv(BUCKET_OUT)
        done = set(zip(prev.symbol, prev.date.astype(str)))
        rows = prev.to_dict('records')
        print(f'resuming, {len(done)} done', flush=True)
    else:
        done = set()

    for i, row in m.iterrows():
        key = (row.symbol, str(row.date.date()))
        if key in done:
            continue
        day_et = row.entry_fill_ts.tz_convert(ET).date()
        t0 = pd.Timestamp.combine(day_et, dtime(9, 34, 0)).tz_localize(ET).tz_convert('UTC')
        t1 = pd.Timestamp.combine(day_et, dtime(9, 36, 0)).tz_localize(ET).tz_convert('UTC')
        try:
            qt = client.get_stock_quotes(StockQuotesRequest(
                symbol_or_symbols=row.symbol, start=t0, end=t1, feed=DataFeed.SIP,
                limit=QLIMIT))
            qdf = qt.df.reset_index() if len(qt.data.get(row.symbol, [])) else pd.DataFrame()
            err = None
        except Exception as e:
            qdf = pd.DataFrame()
            err = str(e)
        if len(qdf):
            qdf['ts_et'] = pd.to_datetime(qdf['timestamp'], utc=True).dt.tz_convert(ET)
            qdf['bucket'] = qdf.ts_et.apply(bucket_of)
            qdf = qdf.dropna(subset=['bucket'])
            qdf['mid'] = (qdf.bid_price + qdf.ask_price) / 2.0
            qdf = qdf[qdf.mid > 0]
            qdf['half_spread_bps'] = (qdf.ask_price - qdf.bid_price) / 2.0 / qdf.mid * 10000.0
            for b, g in qdf.groupby('bucket'):
                rows.append(dict(symbol=row.symbol, date=str(row.date.date()),
                                  split=row.split, bucket=int(b), n=len(g),
                                  median_hs_bps=g.half_spread_bps.median(),
                                  p75_hs_bps=g.half_spread_bps.quantile(0.75)))
        if len(rows) % 200 == 0:
            pd.DataFrame(rows).to_csv(BUCKET_OUT, index=False)
            print(f'  ... {i+1}/{len(m)}', flush=True)
        time.sleep(0.03)

    pd.DataFrame(rows).to_csv(BUCKET_OUT, index=False)
    print('done', len(rows), flush=True)


if __name__ == '__main__':
    main()
