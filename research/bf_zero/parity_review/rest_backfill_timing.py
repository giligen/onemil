#!/usr/bin/env python3
"""(1) Is the JUST-CLOSED minute bar already in Alpaca REST at +1..+3 s after the minute mark (the engine's backfill can
run at any moment; the WS bar for that minute arrives at +0.1..0.3 s)? If REST lags, a backfill right after the mark
returns the day minus its last bar — harmless only because the WS bar was merged first. (2) Wall time of the engine's own
`get_1min_bars_multi` for a 200-symbol chunk from 09:30 to now (the `_backfill` chunk) vs DEFAULT_API_TIMEOUT=90 s. Read-only."""
import os, sys, time
from datetime import datetime, timezone, timedelta
ROOT = '/home/ec2-user/onemil'; sys.path.insert(0, ROOT); os.chdir(ROOT)
from config import Config
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from data_sources.alpaca_client import AlpacaClient
cfg = Config(); hist = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
syms = ['SPY', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'AAOI', 'ACHR', 'ABCL']
for delay in (1.0, 3.0):
    while datetime.now(timezone.utc).second > 50: time.sleep(0.2)
    while datetime.now(timezone.utc).second != 0: time.sleep(0.05)
    time.sleep(delay)
    now = datetime.now(timezone.utc); closed = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
    data = hist.get_stock_bars(StockBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame(1, TimeFrameUnit.Minute), start=now - timedelta(minutes=3), feed=DataFeed.SIP)).data
    have = {s: (data.get(s, [])[-1].timestamp if data.get(s) else None) for s in syms}
    print(f'REST at +{(now - closed).total_seconds() - 60:.1f}s after the {closed:%H:%M}Z bar closed: last ts per symbol = ' +
          ', '.join(f"{s}:{'PRESENT' if t == closed else ('older ' + t.strftime('%H:%M') if t else 'none')}" for s, t in have.items()), flush=True)
uni = [l.strip() for l in open(f'{ROOT}/logs/hod_stream_universe_2026-09-15.txt') if l.strip()][:200]
ac = AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)
et_min = datetime.now(timezone.utc).astimezone(__import__('zoneinfo').ZoneInfo('America/New_York'))
lookback = max(30, et_min.hour * 60 + et_min.minute - 570 + 5)
t0 = time.time(); got = ac.get_1min_bars_multi(uni, lookback_minutes=lookback); dt = time.time() - t0
n = sum(len(v) for v in got.values()); empty = [s for s in uni if s not in got or not len(got[s])]
firsts = {}
for s, df in got.items():
    if len(df):
        f = df['timestamp'].iloc[0].tz_convert('America/New_York') if hasattr(df['timestamp'].iloc[0], 'tz_convert') else df['timestamp'].iloc[0]
        firsts[f.strftime('%H:%M')] = firsts.get(f.strftime('%H:%M'), 0) + 1
print(f'engine backfill chunk: 200 symbols, lookback {lookback} min -> {n} bars in {dt:.1f}s (timeout {ac._api_timeout}s); empty {len(empty)} {empty[:10]}; first-bar minute distribution {dict(sorted(firsts.items())[:6])}', flush=True)
