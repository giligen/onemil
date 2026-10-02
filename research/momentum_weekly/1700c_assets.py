#!/usr/bin/env python3
"""Cell 1,700c step 1 -- Alpaca us_equity asset roster (active + inactive) + delisted-bars probe.

PREREG: research/momentum_weekly/PREREG_1700c.md (FROZEN). Writes 1700c_assets.csv (symbol, name, status,
exchange, tradable) for every us_equity asset Alpaca's assets endpoint returns with status active OR
inactive. Also tests whether Alpaca's historical-bars endpoint serves daily bars for 3 known delisted
tickers (SIVB, TWTR, FRC) over the window each was actually listed, and logs the answer plainly -- this
decides whether the inactive-asset universe in step 2 is worth fetching at all.

Read-only: GetAssetsRequest (trading API) and get_stock_bars (data API). Does not touch orders, positions,
cache.db, or any file outside research/momentum_weekly/.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research' / 'momentum_weekly'
OUT.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / '.env')

logging.basicConfig(
    filename=str(OUT / '1700c_assets.log'), filemode='w', level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('m1700c_assets')
log.addHandler(logging.StreamHandler(sys.stdout))

sys.path.insert(0, str(ROOT))
from alpaca.trading.requests import GetAssetsRequest  # noqa: E402
from alpaca.trading.enums import AssetClass, AssetStatus  # noqa: E402
from alpaca.data.requests import StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame  # noqa: E402
from data_sources.alpaca_client import AlpacaClient, AlpacaAPIError  # noqa: E402

t0 = time.time()
api_key = os.getenv('ALPACA_API_KEY')
api_secret = os.getenv('ALPACA_API_SECRET')
if not api_key or not api_secret:
    log.error('ALPACA_API_KEY / ALPACA_API_SECRET missing from .env -- cannot proceed, aborting per '
              'no-mock-fallback policy')
    raise SystemExit(1)

paper = os.getenv('ALPACA_PAPER', 'true').strip().lower() == 'true'
client = AlpacaClient(api_key, api_secret, paper=paper)
log.info('AlpacaClient initialized (paper=%s, matches .env ALPACA_PAPER -- data+trading READ-ONLY use '
         'in this script: GetAssetsRequest and get_stock_bars only, no orders/positions touched)', paper)

# ============================================================== asset roster ==
rows = []
for status, label in [(AssetStatus.ACTIVE, 'active'), (AssetStatus.INACTIVE, 'inactive')]:
    req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=status)
    try:
        assets = client._call_with_timeout(
            lambda r=req: client.trading_client.get_all_assets(r), f'get_all_assets({label})',
            timeout=90, timeout_retries=2, rate_limit_retries=5)
    except AlpacaAPIError as e:
        log.error('get_all_assets(%s) failed: %s -- this status is INCOMPLETE in the output, not silently '
                   'dropped: see completeness note', label, e)
        assets = []
    for a in assets:
        rows.append(dict(symbol=a.symbol, name=a.name or '', status=label,
                          exchange=str(a.exchange) if a.exchange else '', tradable=bool(a.tradable)))
    log.info('%.0fs fetched %d %s us_equity assets', time.time() - t0, len(assets), label)

assets_df = pd.DataFrame(rows)
assets_df.to_csv(OUT / '1700c_assets.csv', index=False)
log.info('%.0fs wrote 1700c_assets.csv: %d rows (%d active, %d inactive, %d distinct symbols, %d tradable)',
          time.time() - t0, len(assets_df), (assets_df.status == 'active').sum(),
          (assets_df.status == 'inactive').sum(), assets_df.symbol.nunique(), int(assets_df.tradable.sum()))

# ===================================================== delisted-bars probe ==
# Each ticker's probe window is the period it was actually listed and trading, chosen from public record,
# NOT the full requested panel window (2015-07..2026-09) -- asking Alpaca for bars after a delisting date
# would trivially return nothing and tell us nothing about historical coverage.
PROBES = [
    ('SIVB', '2022-01-03', '2023-03-10'),  # Silicon Valley Bank, FDIC receivership 2023-03-10
    ('TWTR', '2022-01-03', '2022-10-27'),  # Twitter Inc, went private 2022-10-28 (Musk deal close)
    ('FRC',  '2022-01-03', '2023-05-01'),  # First Republic Bank, FDIC seizure 2023-05-01
]
probe_rows = []
for sym, start, end in PROBES:
    try:
        req = StockBarsRequest(symbol_or_symbols=[sym], timeframe=TimeFrame.Day,
                                start=pd.Timestamp(start), end=pd.Timestamp(end), adjustment='all')
        barset = client._call_with_timeout(
            lambda r=req: client.data_client.get_stock_bars(r), f'probe_bars({sym})',
            timeout=60, timeout_retries=2, rate_limit_retries=5)
        bars = barset.data.get(sym, []) if hasattr(barset, 'data') else []
        n = len(bars)
        first = bars[0].timestamp.date() if n else None
        last = bars[-1].timestamp.date() if n else None
        log.info('PROBE %s %s..%s: %d daily bars returned (first=%s last=%s)', sym, start, end, n, first, last)
        probe_rows.append(dict(symbol=sym, window_start=start, window_end=end, n_bars=n,
                                first_bar=first, last_bar=last, served=n > 0))
    except Exception as e:
        log.error('PROBE %s failed: %s -- treated as NOT served (served=False), not silently skipped', sym, e)
        probe_rows.append(dict(symbol=sym, window_start=start, window_end=end, n_bars=0,
                                first_bar=None, last_bar=None, served=False))

probe_df = pd.DataFrame(probe_rows)
probe_df.to_csv(OUT / '1700c_delisted_probe.csv', index=False)
all_served = bool(probe_df['served'].all())
log.info('%.0fs DELISTED-BARS ANSWER: Alpaca %s serve daily bars for delisted tickers past their listed '
          'life -- %d/%d probed tickers returned bars (%s)', time.time() - t0,
          'DOES' if all_served else ('PARTIALLY' if probe_df['served'].any() else 'DOES NOT'),
          int(probe_df['served'].sum()), len(probe_df), probe_df[['symbol', 'served', 'n_bars']].to_dict('records'))
log.info('%.0fs DONE step 1', time.time() - t0)
