#!/usr/bin/env python3
"""Does Alpaca REST (StockBarsRequest 1-Min SIP, no `end`) return the CURRENT in-progress minute? The engine's
`_backfill` -> `get_1min_bars_multi` sets no `end`; a partial bar stored via set_bar() is then scanned by detect() and
`next_idx` advances past it, so the completed bar (WS, seconds later) is never re-scanned. Read-only probe."""
import os, sys, time
from datetime import datetime, timezone, timedelta
ROOT = '/home/ec2-user/onemil'; sys.path.insert(0, ROOT); os.chdir(ROOT)
from config import Config
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
cfg = Config(); hist = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
syms = ['SPY', 'AAPL', 'TSLA', 'NVDA', 'AMD']
# wait until ~20-35 s into a minute so the current minute is clearly in progress
while not (20 <= datetime.now(timezone.utc).second <= 35):
    time.sleep(1)
now = datetime.now(timezone.utc); cur_min = now.replace(second=0, microsecond=0)
req = StockBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame(1, TimeFrameUnit.Minute), start=now - timedelta(minutes=4), feed=DataFeed.SIP)
data = hist.get_stock_bars(req).data
print(f'now {now:%H:%M:%S}Z current minute {cur_min:%H:%M}Z')
for s in syms:
    lst = data.get(s, [])
    last = lst[-1] if lst else None
    print(f"{s}: {len(lst)} bars, last ts {last.timestamp if last else None} -> {'IN-PROGRESS MINUTE RETURNED' if last and last.timestamp == cur_min else 'only completed minutes'}"
          f"{'' if not last else f' (v={last.volume:.0f}, n={last.trade_count})'}")
# also: the engine's own accessor
from data_sources.alpaca_client import AlpacaClient
ac = AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)
now2 = datetime.now(timezone.utc); cur2 = now2.replace(second=0, microsecond=0)
got = ac.get_1min_bars_multi(syms, lookback_minutes=5)
for s in syms:
    df = got.get(s)
    if df is None or not len(df): print(f'{s}: engine accessor returned nothing'); continue
    ts = df['timestamp'].iloc[-1]
    print(f"{s}: engine accessor last ts {ts} (now {now2:%H:%M:%S}Z) -> {'IN-PROGRESS' if ts == cur2 else 'completed only'} first ts {df['timestamp'].iloc[0]} n={len(df)}")
