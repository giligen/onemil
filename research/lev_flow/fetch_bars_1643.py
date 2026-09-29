#!/usr/bin/env python3
"""
Fetch Alpaca minute and daily bars for cells 1,643-1,645 (leveraged-ETF rebalancing flow into the close).

Per research/lev_flow/PREREG_1643.md (frozen): minute bars (adjustment='all', SIP feed) for
2016-01-04 -> 2026-09-04 for the underlying ETFs SMH, QQQ, IWM, XLF, XLE, GDX, XBI, TLT, SPY, plus
Alpaca daily bars (adjustment='all') to get the official close and next official open used to build
r_t and the MOC exit.

Storage: parquet per symbol under research/lev_flow/data/, never sqlite/cache.db.
  - Minute bars are staged ONE symbol-month per file under data/minute_stage/<SYM>/<YYYY-MM>.parquet,
    written atomically (tmp name then os.rename) so a killed run resumes without re-fetching completed
    months (a month file existing = done). A final combine pass concatenates each symbol's staged
    months into data/minute/<SYM>.parquet, also atomic.
  - Daily bars are small (~2,700 rows/symbol for 11 years) and fetched in one shot per symbol into
    data/daily/<SYM>.parquet; padded a few weeks either side of the minute range so the first event's
    prior close and the last event's next open are both resolvable.

Usage: python3 research/lev_flow/fetch_bars_1643.py [minute|daily|all]
"""
import glob
import os
import sys
import time
from datetime import datetime, timedelta, timezone

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config import Config

SYMBOLS = ['SMH', 'QQQ', 'IWM', 'XLF', 'XLE', 'GDX', 'XBI', 'TLT', 'SPY']

MIN_START = datetime(2016, 1, 4, tzinfo=timezone.utc)
MIN_END = datetime(2026, 9, 5, tzinfo=timezone.utc)  # exclusive end -> covers through 2026-09-04

# Daily bars padded: need the close BEFORE 2016-01-04 (prior close of the first possible event) and
# the next open AFTER 2026-09-04 (next-open reversal read, cell 1,644, for the last few events).
DAILY_START = datetime(2015, 12, 1, tzinfo=timezone.utc)
DAILY_END = datetime(2026, 9, 12, tzinfo=timezone.utc)

DATA_DIR = os.path.join(ROOT, 'research/lev_flow/data')
MIN_STAGE_DIR = os.path.join(DATA_DIR, 'minute_stage')
MIN_FINAL_DIR = os.path.join(DATA_DIR, 'minute')
DAILY_FINAL_DIR = os.path.join(DATA_DIR, 'daily')

MAX_ATTEMPTS = 5


def _atomic_write_parquet(df: pd.DataFrame, final_path: str) -> None:
    """Write df to final_path via a temp file + os.rename so a kill mid-write never corrupts a cache."""
    tmp_path = final_path + '.tmp'
    df.to_parquet(tmp_path, index=False)
    os.rename(tmp_path, final_path)


def month_ranges(start: datetime, end: datetime):
    """Yield (month_start, month_end) UTC datetime pairs covering [start, end), one per calendar month."""
    out = []
    d = start
    while d < end:
        nxt = (d.replace(day=1) + timedelta(days=32)).replace(day=1)
        out.append((d, min(nxt, end)))
        d = nxt
    return out


def bars_to_df(bar_list, symbol: str) -> pd.DataFrame:
    """Convert alpaca-py Bar objects to a flat DataFrame (symbol, t, o, h, l, c, v, n, vw)."""
    rows = [{
        'symbol': symbol,
        't': x.timestamp.astimezone(timezone.utc),
        'o': float(x.open), 'h': float(x.high), 'l': float(x.low), 'c': float(x.close),
        'v': float(x.volume), 'n': int(x.trade_count or 0), 'vw': float(x.vwap or 0),
    } for x in bar_list]
    return pd.DataFrame(rows)


def fetch_minute(client: StockHistoricalDataClient) -> None:
    """Fetch 1-min adjustment='all' SIP bars for every symbol, staged monthly, resumable."""
    os.makedirs(MIN_STAGE_DIR, exist_ok=True)
    os.makedirs(MIN_FINAL_DIR, exist_ok=True)
    months = month_ranges(MIN_START, MIN_END)
    t0 = time.time()
    total = 0
    for sym in SYMBOLS:
        sym_dir = os.path.join(MIN_STAGE_DIR, sym)
        os.makedirs(sym_dir, exist_ok=True)
        for a, b in months:
            key = a.strftime('%Y-%m')
            final_month_path = os.path.join(sym_dir, f'{key}.parquet')
            if os.path.exists(final_month_path):
                continue
            for attempt in range(MAX_ATTEMPTS):
                try:
                    req = StockBarsRequest(
                        symbol_or_symbols=sym, timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                        start=a, end=b, feed=DataFeed.SIP, adjustment=Adjustment.ALL,
                    )
                    resp = client.get_stock_bars(req)
                    data = resp.data.get(sym, []) if hasattr(resp, 'data') else resp.get(sym, [])
                    df = bars_to_df(data, sym)
                    _atomic_write_parquet(df, final_month_path)
                    total += len(df)
                    print(f'[minute] {sym} {key}: {len(df)} bars | total {total:,} | '
                          f'{(time.time() - t0) / 60:.1f} min', flush=True)
                    break
                except Exception as e:
                    print(f'[minute] {sym} {key} attempt {attempt + 1} failed: {str(e)[:160]}', flush=True)
                    time.sleep(5 * (attempt + 1))
            else:
                print(f'[minute] ERROR: {sym} {key} FAILED after {MAX_ATTEMPTS} attempts, '
                      f'skipping (resumable: rerun to retry)', flush=True)

    # Combine each symbol's staged months into one final parquet.
    for sym in SYMBOLS:
        sym_dir = os.path.join(MIN_STAGE_DIR, sym)
        month_files = sorted(glob.glob(os.path.join(sym_dir, '*.parquet')))
        if not month_files:
            print(f'[minute] WARNING: no staged months for {sym}, cannot build final parquet', flush=True)
            continue
        combined = pd.concat([pd.read_parquet(f) for f in month_files], ignore_index=True)
        combined = combined.drop_duplicates(subset=['t']).sort_values('t').reset_index(drop=True)
        final_path = os.path.join(MIN_FINAL_DIR, f'{sym}.parquet')
        _atomic_write_parquet(combined, final_path)
        print(f'[minute] COMBINED {sym}: {len(combined):,} bars -> {final_path}', flush=True)


def fetch_daily(client: StockHistoricalDataClient) -> None:
    """Fetch daily adjustment='all' SIP bars for every symbol (official close / next open source)."""
    os.makedirs(DAILY_FINAL_DIR, exist_ok=True)
    t0 = time.time()
    for sym in SYMBOLS:
        final_path = os.path.join(DAILY_FINAL_DIR, f'{sym}.parquet')
        if os.path.exists(final_path):
            print(f'[daily] {sym} already fetched, skipping', flush=True)
            continue
        for attempt in range(MAX_ATTEMPTS):
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=sym, timeframe=TimeFrame(1, TimeFrameUnit.Day),
                    start=DAILY_START, end=DAILY_END, feed=DataFeed.SIP, adjustment=Adjustment.ALL,
                )
                resp = client.get_stock_bars(req)
                data = resp.data.get(sym, []) if hasattr(resp, 'data') else resp.get(sym, [])
                df = bars_to_df(data, sym)
                _atomic_write_parquet(df, final_path)
                print(f'[daily] {sym}: {len(df)} sessions | {(time.time() - t0) / 60:.1f} min', flush=True)
                break
            except Exception as e:
                print(f'[daily] {sym} attempt {attempt + 1} failed: {str(e)[:160]}', flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            print(f'[daily] ERROR: {sym} FAILED after {MAX_ATTEMPTS} attempts', flush=True)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    cfg = Config()
    if not cfg.alpaca_api_key or not cfg.alpaca_api_secret:
        print('ERROR: ALPACA_API_KEY / ALPACA_API_SECRET missing from environment -- cannot fetch', flush=True)
        sys.exit(1)
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    print(f'Fetching mode={mode} for {len(SYMBOLS)} symbols: {SYMBOLS}', flush=True)
    if mode in ('daily', 'all'):
        fetch_daily(client)
    if mode in ('minute', 'all'):
        fetch_minute(client)
    print('FETCH DONE', flush=True)
