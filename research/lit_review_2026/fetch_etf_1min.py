#!/usr/bin/env python3
"""Literature-review test data: 1-minute SIP bars 2016-01-04 → today for the index/leveraged ETFs the intraday-momentum
and ORB papers are written on, from Alpaca REST (free), into research/lit_review_2026/etf_1min.db (bars(symbol, t, o, h,
l, c, v, n, vw), PRIMARY KEY (symbol, t)). Resumable by (symbol, month). One symbol-month per request (~8K bars)."""
import os, sqlite3, sys, time
from datetime import datetime, timezone, timedelta
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from config import Config
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment
SYMS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['SPY', 'QQQ', 'TQQQ', 'SQQQ', 'IWM', 'DIA', 'SOXL', 'UVXY']
START = datetime(2016, 1, 1, tzinfo=timezone.utc); END = datetime.now(timezone.utc)
cfg = Config(); client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
con = sqlite3.connect('research/lit_review_2026/etf_1min.db', timeout=60)
con.execute("create table if not exists bars (symbol text, t text, o real, h real, l real, c real, v real, n integer, vw real, primary key (symbol, t))")
con.execute("create table if not exists done (symbol text, month text, n integer, primary key (symbol, month))"); con.commit()
done = {(s, m) for s, m in con.execute("select symbol, month from done")}
months = []; d = START
while d < END:
    nxt = (d.replace(day=1) + timedelta(days=32)).replace(day=1); months.append((d, min(nxt, END))); d = nxt
t0 = time.time(); total = 0
for sym in SYMS:
    for a, b in months:
        key = (sym, a.strftime('%Y-%m'))
        if key in done: continue
        for attempt in range(4):
            try:
                r = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame(1, TimeFrameUnit.Minute), start=a, end=b, feed=DataFeed.SIP, adjustment=Adjustment.RAW))
                data = r.data.get(sym, []) if hasattr(r, 'data') else r.get(sym, [])
                rows = [(sym, x.timestamp.astimezone(timezone.utc).isoformat(), float(x.open), float(x.high), float(x.low), float(x.close), float(x.volume), int(x.trade_count or 0), float(x.vwap or 0)) for x in data]
                con.executemany("insert or replace into bars values (?,?,?,?,?,?,?,?,?)", rows)
                con.execute("insert or replace into done values (?,?,?)", (sym, key[1], len(rows))); con.commit(); total += len(rows)
                print(f'{sym} {key[1]} {len(rows)} bars | total {total:,} | {(time.time() - t0) / 60:.1f} min', flush=True); break
            except Exception as e:
                print(f'{sym} {key[1]} attempt {attempt + 1} failed: {str(e)[:120]}', flush=True); time.sleep(5 * (attempt + 1))
print('DONE', flush=True)
