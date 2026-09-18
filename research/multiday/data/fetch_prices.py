#!/usr/bin/env python3
"""Multi-day DATA stage, step 2 — Alpaca SIP daily bars, raw AND split/dividend adjusted.

2016-01-04 → 2026-09-18 for the whole `universe.parquet`, fetched TWICE
(`adjustment='raw'` and `adjustment='all'`) because the program needs features on
adjusted prices and share counts / price gates on raw prices (PLAN, Data §1).

Resumable: one parquet per (adjustment, symbol batch) under `prices/<adj>/`; an
existing file is never re-fetched. Memory is bounded by the batch (≤ 200 symbols
× ~2,700 sessions ≈ 540K rows ≈ 40 MB).

    python3 fetch_prices.py                 # both adjustments, all batches
    python3 fetch_prices.py --adj raw       # one adjustment
    python3 fetch_prices.py --batch-size 200

Consolidation into year-partitioned parquet is `partition_prices.py`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO = Path('/home/ec2-user/onemil')
sys.path.insert(0, str(REPO))
load_dotenv(REPO / '.env')

from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame  # noqa: E402
from alpaca.data.enums import DataFeed, Adjustment  # noqa: E402

HERE = Path(__file__).resolve().parent
UNIVERSE = HERE / 'universe.parquet'
PRICES = HERE / 'prices'

START = datetime(2016, 1, 4, tzinfo=timezone.utc)
END = datetime(2026, 9, 19, tzinfo=timezone.utc)   # exclusive-ish; API is inclusive of the day
ADJUSTMENTS = {'raw': Adjustment.RAW, 'all': Adjustment.ALL}
MAX_RETRIES = 6


def fetch_batch(client: StockHistoricalDataClient, symbols: list[str],
                adjustment: Adjustment, start: datetime | None = None) -> pd.DataFrame:
    """One multi-symbol daily-bar pull with 429/transient backoff. Returns a tidy frame."""
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start or START,
        end=END,
        feed=DataFeed.SIP,
        adjustment=adjustment,
    )
    backoff = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            bars = client.get_stock_bars(req)
            break
        except Exception as exc:  # noqa: BLE001 — every failure is logged and retried
            msg = str(exc)
            transient = ('429' in msg or 'rate limit' in msg.lower()
                         or '500' in msg or '502' in msg or '503' in msg
                         or 'timeout' in msg.lower() or 'connection' in msg.lower())
            if attempt == MAX_RETRIES - 1 or not transient:
                raise
            print(f'    retry {attempt + 1}/{MAX_RETRIES} after {backoff:.0f}s: {msg[:120]}',
                  flush=True)
            time.sleep(backoff)
            backoff *= 2
    else:  # pragma: no cover - the loop always breaks or raises
        raise RuntimeError('unreachable')

    df = bars.df
    if df is None or df.empty:
        return pd.DataFrame(columns=['symbol', 'date', 'open', 'high', 'low', 'close',
                                     'volume', 'trade_count', 'vwap'])
    df = df.reset_index()
    df['date'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(
        'America/New_York').dt.date
    out = pd.DataFrame({
        'symbol': df['symbol'].astype('string'),
        'date': df['date'],
        'open': df['open'].astype('float32'),
        'high': df['high'].astype('float32'),
        'low': df['low'].astype('float32'),
        'close': df['close'].astype('float32'),
        'volume': df['volume'].astype('float64'),
        'trade_count': df.get('trade_count', pd.Series(0, index=df.index)).astype('float64'),
        'vwap': df.get('vwap', pd.Series(float('nan'), index=df.index)).astype('float32'),
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--adj', choices=['raw', 'all'], action='append',
                    help='restrict to one adjustment (repeatable); default both')
    ap.add_argument('--batch-size', type=int, default=200)
    ap.add_argument('--topup-from', default=None,
                    help='re-fetch only sessions >= this date into topup_batch_*.parquet '
                         '(the two adjustments are pulled minutes apart, so the last '
                         'session of a live trading day can land in one and not the other)')
    args = ap.parse_args()

    key, secret = os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_API_SECRET')
    if not key or not secret:
        raise SystemExit('ALPACA_API_KEY/SECRET missing from .env')
    client = StockHistoricalDataClient(key, secret)

    symbols = sorted(pd.read_parquet(UNIVERSE)['symbol'].tolist())
    batches = [symbols[i:i + args.batch_size] for i in range(0, len(symbols), args.batch_size)]
    adjs = args.adj or ['raw', 'all']
    print(f'{len(symbols)} symbols in {len(batches)} batches × {adjs}', flush=True)

    t_start = time.time()
    for adj in adjs:
        outdir = PRICES / adj
        outdir.mkdir(parents=True, exist_ok=True)
        done_rows = 0
        prefix = 'topup_batch' if args.topup_from else 'batch'
        start = (datetime.fromisoformat(args.topup_from).replace(tzinfo=timezone.utc)
                 if args.topup_from else None)
        for i, batch in enumerate(batches):
            path = outdir / f'{prefix}_{i:03d}.parquet'
            if path.exists():
                continue
            t0 = time.time()
            df = fetch_batch(client, batch, ADJUSTMENTS[adj], start=start)
            df.to_parquet(path, index=False, compression='snappy')
            done_rows += len(df)
            print(f'[{adj}] batch {i + 1}/{len(batches)} '
                  f'({batch[0]}..{batch[-1]}) rows={len(df):,} '
                  f'{time.time() - t0:.0f}s elapsed={time.time() - t_start:.0f}s', flush=True)
        print(f'[{adj}] done, {done_rows:,} new rows', flush=True)
    print(f'TOTAL wall {time.time() - t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
