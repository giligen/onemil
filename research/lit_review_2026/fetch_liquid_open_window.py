#!/usr/bin/env python3
"""H-B1 data: Zarattini-Barbon-Aziz 'stocks in play' replication out of sample (2025-01 → 2026-09).
Universe per day (point-in-time daily panel): prior close > $5, 14-day ADV >= 1M shares, ATR14 > $0.50.
Step 1: for every liquid name and day fetch the 09:30-09:34 1-min bars (5 bars) from Alpaca SIP → open-window volume → the
relative volume RV = today's window volume / mean of the previous 14 days' window volume (causal at 09:35).
Step 2: the top-20 RV names per day with RV >= 1 → fetch the full-day 1-min bars (04:00-20:00) into liquid_days.db.
Stores: research/lit_review_2026/liquid_open.db (window(symbol, day, v5, h5, l5, o5, c5), days(symbol, day, ...)), resumable."""
import os, sys, time, sqlite3
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from config import Config
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment
ET = ZoneInfo('America/New_York'); STEP = sys.argv[1] if len(sys.argv) > 1 else 'window'
cfg = Config(); client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
con = sqlite3.connect('research/lit_review_2026/liquid_open.db', timeout=60)
con.execute("create table if not exists window (symbol text, day text, o5 real, h5 real, l5 real, c5 real, v5 real, primary key (symbol, day))")
con.execute("create table if not exists window_done (day text primary key, n integer)")
con.execute("create table if not exists days (symbol text, day text, t text, o real, h real, l real, c real, v real, primary key (symbol, day, t))")
con.execute("create table if not exists days_done (symbol text, day text, n integer, primary key (symbol, day))"); con.commit()
d = pd.read_parquet('research/lit_review_2026/daily_panel.parquet', columns=['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume', 'prev_close'])
d = d[~d.symbol.astype(str).str.match(r'^Z[VWX]ZZ|^ZZ')].sort_values(['symbol', 'bar_date'])
g = d.groupby('symbol', observed=True)
d['adv14'] = g.volume.transform(lambda s: s.shift(1).rolling(14, min_periods=14).mean())
tr = np.maximum(d.high - d.low, np.maximum((d.high - d.prev_close).abs(), (d.low - d.prev_close).abs()))
d['atr14'] = tr.groupby(d.symbol, observed=True).transform(lambda s: s.shift(1).rolling(14, min_periods=14).mean())
liq = d[(d.prev_close > 5) & (d.adv14 >= 1e6) & (d.atr14 > 0.5)][['symbol', 'bar_date']]
liq['symbol'] = liq.symbol.astype(str); liq = liq[liq.symbol.str.match(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$')]   # Alpaca symbology; one bad symbol fails a whole 200-name request
print('liquid symbol-days', len(liq), 'days', liq.bar_date.nunique(), flush=True)

def fetch(symbols, start, end):
    out = {}
    for i in range(0, len(symbols), 200):
        chunk = symbols[i:i + 200]
        for attempt in range(4):
            try:
                r = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame(1, TimeFrameUnit.Minute), start=start, end=end, feed=DataFeed.SIP, adjustment=Adjustment.RAW))
                data = r.data if hasattr(r, 'data') else r
                for s in chunk: out[s] = data.get(s, [])
                break
            except Exception as e:
                print(f'  fetch failed ({str(e)[:100]}) attempt {attempt + 1}', flush=True); time.sleep(5 * (attempt + 1))
    return out

if STEP == 'window':
    done = {r[0] for r in con.execute("select day from window_done")}; t0 = time.time()
    for k, (day, gd) in enumerate(liq.groupby('bar_date')):
        if day in done: continue
        syms = sorted(gd.symbol); d0 = datetime.strptime(day, '%Y-%m-%d').replace(tzinfo=ET)
        bars = fetch(syms, d0.replace(hour=9, minute=30).astimezone(timezone.utc), d0.replace(hour=9, minute=35).astimezone(timezone.utc))
        rows = []
        for s, bs in bars.items():
            bs = [b for b in bs if b.timestamp.astimezone(ET).hour * 60 + b.timestamp.astimezone(ET).minute < 575]
            if not bs: continue
            rows.append((s, day, float(bs[0].open), max(float(b.high) for b in bs), min(float(b.low) for b in bs), float(bs[-1].close), float(sum(b.volume for b in bs))))
        con.executemany("insert or replace into window values (?,?,?,?,?,?,?)", rows); con.execute("insert or replace into window_done values (?,?)", (day, len(rows))); con.commit()
        if k % 10 == 0: print(f'{k + 1}/{liq.bar_date.nunique()} {day} liquid {len(syms)} windows {len(rows)} | {(time.time() - t0) / 60:.1f} min', flush=True)
    print('WINDOW DONE', flush=True)
else:
    w = pd.read_sql("select symbol, day, v5 from window", con).sort_values(['symbol', 'day'])
    w['base'] = w.groupby('symbol').v5.transform(lambda s: s.shift(1).rolling(14, min_periods=14).mean()); w['rv'] = w.v5 / w.base
    top = w[w.rv >= 1].sort_values(['day', 'rv'], ascending=[True, False]).groupby('day').head(20)
    top.to_csv('research/lit_review_2026/liquid_top20.csv', index=False); print('top-20 symbol-days', len(top), flush=True)
    done = {(a, b) for a, b in con.execute("select symbol, day from days_done")}; t0 = time.time()
    for k, (day, gd) in enumerate(top.groupby('day')):
        syms = [s for s in gd.symbol if (s, day) not in done]
        if not syms: continue
        d0 = datetime.strptime(day, '%Y-%m-%d').replace(tzinfo=ET)
        bars = fetch(syms, d0.replace(hour=4, minute=0).astimezone(timezone.utc), d0.replace(hour=20, minute=0).astimezone(timezone.utc))
        rows = [(s, day, b.timestamp.astimezone(timezone.utc).isoformat(), float(b.open), float(b.high), float(b.low), float(b.close), float(b.volume)) for s, bs in bars.items() for b in bs]
        con.executemany("insert or replace into days values (?,?,?,?,?,?,?,?)", rows); con.executemany("insert or replace into days_done values (?,?,?)", [(s, day, len(bars.get(s, []))) for s in syms]); con.commit()
        if k % 20 == 0: print(f'{k + 1}/{top.day.nunique()} {day} {len(syms)} names {len(rows)} bars | {(time.time() - t0) / 60:.1f} min', flush=True)
    print('DAYS DONE', flush=True)
