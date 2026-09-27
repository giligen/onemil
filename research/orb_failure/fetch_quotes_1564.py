#!/usr/bin/env python3
"""research/orb_failure/PREREG_1564.md -- resumable Alpaca NBBO fetch at 10:30:00 ET, per
candidate symbol-day, for cell_1564.py's entry half-spread. Never touches data/cache.db, config.
yaml, orb.yaml or any live config; writes only research/orb_failure/quotes_1564/<day>.json.

One request per (day, chunk of <=10 symbols) over a narrow [10:29:50, 10:30:10) ET window (SIP
feed); takes the quote timestamped closest to (and at-or-before) 10:30:00, else the closest after
gap-filling. Resumable: a day whose JSON file already exists is skipped -- delete the file to
re-fetch it.

Usage:
    python3 research/orb_failure/fetch_quotes_1564.py [--limit-days N]
"""
import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from trading.orb_csv import read_orb_csv  # noqa: E402

CANDIDATES_CSV = os.path.join(REPO, 'analysis_results/orb_features_20260925_2054.csv')
QUOTE_CACHE_DIR = os.path.join(HERE, 'quotes_1564')
ET = ZoneInfo('America/New_York')
BATCH_SYMBOLS = 10       # alpaca-py multi-symbol pagination limit, see fetch_bars_1478.py
PAUSE_S = 0.2
LOST_PCT_WARN = 2.0

log = logging.getLogger('fetch_quotes_1564')


def et_window_utc(day):
    d = dt.date.fromisoformat(day)
    start_et = dt.datetime(d.year, d.month, d.day, 10, 29, 50, tzinfo=ET)
    end_et = dt.datetime(d.year, d.month, d.day, 10, 30, 10, tzinfo=ET)
    return start_et.astimezone(dt.timezone.utc), end_et.astimezone(dt.timezone.utc)


def load_pairs(limit_days):
    df = read_orb_csv(CANDIDATES_CSV)
    df = df.drop_duplicates(subset=['symbol', 'date'])
    by_day = {}
    for r in df.itertuples():
        by_day.setdefault(str(r.date), []).append(r.symbol)
    days = sorted(by_day)
    if limit_days:
        days = days[:limit_days]
    return {d: sorted(set(by_day[d])) for d in days}


def fetch_day(client, symbols, day):
    from alpaca.data.requests import StockQuotesRequest
    from alpaca.data.enums import DataFeed
    start, end = et_window_utc(day)
    target_et = dt.datetime.combine(dt.date.fromisoformat(day), dt.time(10, 30, 0), tzinfo=ET)
    out = {}
    for i in range(0, len(symbols), BATCH_SYMBOLS):
        chunk = symbols[i:i + BATCH_SYMBOLS]
        req = StockQuotesRequest(symbol_or_symbols=chunk, start=start, end=end,
                                  feed=DataFeed.SIP, limit=10000)
        try:
            qs = client.get_stock_quotes(req)
        except Exception as e:
            log.error('quote fetch FAILED day=%s chunk=%s: %s', day, chunk[:3], e)
            time.sleep(1.0)
            continue
        data = getattr(qs, 'data', {}) or {}
        for sym, quote_list in data.items():
            best = None
            best_dt = None
            for q in quote_list:
                ts = q.timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=dt.timezone.utc)
                if ts <= target_et.astimezone(dt.timezone.utc):
                    if best_dt is None or ts > best_dt:
                        best, best_dt = q, ts
            if best is None and quote_list:
                best = quote_list[0]
            if best is not None and best.bid_price and best.ask_price:
                out[sym] = {'bid': float(best.bid_price), 'ask': float(best.ask_price)}
        time.sleep(PAUSE_S)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--limit-days', type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    os.makedirs(QUOTE_CACHE_DIR, exist_ok=True)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, '.env'))
    from config import Config
    from alpaca.data.historical import StockHistoricalDataClient
    cfg = Config()
    if not cfg.alpaca_api_key or not cfg.alpaca_api_secret:
        log.error('missing Alpaca API credentials (ALPACA_API_KEY/ALPACA_API_SECRET) - cannot fetch')
        return 1
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)

    by_day = load_pairs(a.limit_days)
    total_requested = sum(len(v) for v in by_day.values())
    n_lost = 0
    n_done_days = 0
    t0 = time.time()
    for day, symbols in by_day.items():
        out_path = os.path.join(QUOTE_CACHE_DIR, f'{day}.json')
        if os.path.exists(out_path):
            continue
        got = fetch_day(client, symbols, day)
        n_lost += len(symbols) - len(got)
        with open(out_path, 'w') as f:
            json.dump(got, f)
        n_done_days += 1
        if n_done_days % 25 == 0:
            log.info('progress: %d/%d days, %.0fs elapsed', n_done_days, len(by_day),
                      time.time() - t0)
    lost_pct = 100.0 * n_lost / max(total_requested, 1)
    level = log.error if lost_pct > LOST_PCT_WARN else log.info
    level('LOST %d/%d symbol-days (%.2f%%) with no quote in window', n_lost, total_requested,
          lost_pct)
    return 0


if __name__ == '__main__':
    sys.exit(main())
