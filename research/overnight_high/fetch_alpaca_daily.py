#!/usr/bin/env python3
"""PREREG_1550 step 1 — Alpaca daily bars for the EXTENSION sample (2019-01-02..2024-06-28).

Universe = every symbol in `data/research/databento/alpaca_assets_all_20260905.csv` UNION every
symbol ever carrying a definition record in the point-in-time listing feed (`research/scripts/
pit_listings.py`, `data/research/databento/pit_definition`, coverage 2024-07..2026-09) — this
recovers names still listed today AND names that existed/traded in the PIT window even if since
delisted, per PREREG_1550's disclosed survivorship caveat (delistings BEFORE 2024-07 are still
missing from this union; that is on record, not hidden).

Batches of <=10 symbols per request (alpaca-py silently drops symbols past ~10k bars/page on
finer bars; kept at 10 here for daily bars too, per the task spec, and because a batch-level drop
is exactly the failure this script's LOST accounting is built to catch). For every symbol missing
from a batch response, retries the symbol ALONE: if the solo retry also returns zero bars, the
symbol has no history in the window (expected, not counted as LOST); if the solo retry recovers
bars the batch had silently dropped, the recovered bars are kept and a WARNING is logged (batch
integrity bug, not a data gap). A symbol that raises persistently on the solo retry (transient
API/network failure, not "no data") is the only thing counted as LOST.

Resumable: symbols already recorded (DONE_LOG) are skipped on re-run. Output shards live under
`_fetch_state/shards/<batch_idx>.parquet`; the final concat step (re-run any time, cheap) writes
`research/overnight_high/alpaca_daily_2019_2024H1.parquet`.

Usage:
  python3 research/overnight_high/fetch_alpaca_daily.py            # run to completion
  python3 research/overnight_high/fetch_alpaca_daily.py --limit-batches 5   # smoke test
  python3 research/overnight_high/fetch_alpaca_daily.py --concat-only       # just rebuild the parquet from shards
"""
import argparse
import csv
import datetime as dt
import logging
import os
import re
import sys
import time

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

STATE_DIR = os.path.join(HERE, '_fetch_state')
SHARD_DIR = os.path.join(STATE_DIR, 'shards')
DONE_LOG = os.path.join(STATE_DIR, 'done_symbols.csv')
LOST_LOG = os.path.join(STATE_DIR, 'lost_symbols.csv')
UNIVERSE_CSV = os.path.join(HERE, 'symbol_universe.csv')
OUT_PARQUET = os.path.join(HERE, 'alpaca_daily_2019_2024H1.parquet')

ALPACA_ASSETS_CSV = os.path.join(ROOT, 'data/research/databento/alpaca_assets_all_20260905.csv')

START_DATE = '2019-01-02'
END_DATE = '2024-06-28'
BATCH_SYMBOLS = 10
PAUSE_S = 0.15
SOLO_RETRIES = 3
LOST_PCT_ERROR = 2.0

TEST_TICKER_RE = re.compile(r'^Z[A-Z]ZZT|^ZZ')

log = logging.getLogger('fetch_alpaca_daily')


def build_universe():
    """Union of the current Alpaca asset list and every symbol ever in the PIT listing feed."""
    from research.scripts.pit_listings import PitListings
    assets = pd.read_csv(ALPACA_ASSETS_CSV)
    a_syms = set(assets['symbol'].dropna().astype(str))
    pit = PitListings()
    pit_syms = set()
    for month in pit._months:
        pit_syms |= pit.listed_symbols(month + '01')
    # Alpaca's own asset list carries ~400 CUSIP-like placeholders for corporate-action leftovers
    # (CVR/ESC/CNT suffixes, e.g. "013CVR022", "97ESC024") -- 100% status=inactive, never a real
    # trading symbol Alpaca's data API recognizes. A real US equity ticker in this dataset never
    # contains a digit; filter these out here rather than discover it one "invalid symbol" API
    # error at a time (that IS a fallback path, logged, not a silent drop).
    junk = sorted(s for s in (a_syms | pit_syms) if s and re.search(r'[0-9]', s))
    if junk:
        log.warning('excluding %d CUSIP/CVR-like non-ticker symbols with digits (e.g. %s)', len(junk), junk[:5])
    union = sorted(s for s in (a_syms | pit_syms) if s and not TEST_TICKER_RE.match(s) and not re.search(r'[0-9]', s))
    rows = [{'symbol': s, 'in_alpaca_assets': s in a_syms, 'in_pit_feed': s in pit_syms} for s in union]
    pd.DataFrame(rows).to_csv(UNIVERSE_CSV, index=False)
    log.info('universe built: %d symbols (alpaca_assets %d, pit_feed %d, excluded_junk %d, union %d)',
              len(union), len(a_syms), len(pit_syms), len(junk), len(union))
    return union


def load_done():
    if not os.path.exists(DONE_LOG):
        return set()
    return set(pd.read_csv(DONE_LOG)['symbol'].astype(str))


def append_csv(path, rows, header):
    exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def fetch_batch(client, symbols, req_cls, tf, feed_cls):
    """One get_stock_bars call for <=10 symbols. Returns {symbol: DataFrame-able rows}."""
    req = req_cls(symbol_or_symbols=symbols, timeframe=tf, start=START_DATE, end=END_DATE,
                  feed=feed_cls.SIP, adjustment='raw', limit=10000)
    bars = client.get_stock_bars(req)
    return getattr(bars, 'data', {}) or {}


def bars_to_rows(symbol, bar_list):
    out = []
    for b in bar_list:
        ts = b.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        out.append({
            'symbol': symbol, 'bar_date': ts.astimezone(dt.timezone.utc).date().isoformat(),
            'open': float(b.open), 'high': float(b.high), 'low': float(b.low),
            'close': float(b.close), 'volume': float(b.volume),
        })
    return out


def concat_shards():
    shard_files = sorted(f for f in os.listdir(SHARD_DIR) if f.endswith('.parquet')) if os.path.isdir(SHARD_DIR) else []
    if not shard_files:
        log.error('no shards to concat under %s', SHARD_DIR)
        return 0
    frames = [pd.read_parquet(os.path.join(SHARD_DIR, f)) for f in shard_files]
    d = pd.concat(frames, ignore_index=True)
    d = d.drop_duplicates(subset=['symbol', 'bar_date']).sort_values(['symbol', 'bar_date'])
    d.to_parquet(OUT_PARQUET, index=False)
    log.info('wrote %s: %d rows, %d symbols, %s..%s', OUT_PARQUET, len(d), d.symbol.nunique(),
              d.bar_date.min(), d.bar_date.max())
    return len(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--limit-batches', type=int, default=0)
    ap.add_argument('--concat-only', action='store_true')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    os.makedirs(SHARD_DIR, exist_ok=True)

    if a.concat_only:
        concat_shards()
        return 0

    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, '.env'))
    from config import Config
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    cfg = Config()
    if not cfg.alpaca_api_key or not cfg.alpaca_api_secret:
        log.error('missing Alpaca API credentials (ALPACA_API_KEY/ALPACA_API_SECRET) - cannot fetch')
        return 1
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)

    universe = build_universe() if not os.path.exists(UNIVERSE_CSV) else \
        sorted(pd.read_csv(UNIVERSE_CSV)['symbol'].astype(str))
    done = load_done()
    todo = [s for s in universe if s not in done]
    log.info('universe %d, already done %d, todo %d', len(universe), len(done), len(todo))

    n_lost = 0
    n_no_data = 0
    n_ok = 0
    batch_idx = len([f for f in os.listdir(SHARD_DIR) if f.endswith('.parquet')])
    for i in range(0, len(todo), BATCH_SYMBOLS):
        if a.limit_batches and (i // BATCH_SYMBOLS) >= a.limit_batches:
            break
        chunk = todo[i:i + BATCH_SYMBOLS]
        try:
            data = fetch_batch(client, chunk, StockBarsRequest, TimeFrame.Day, DataFeed)
        except Exception as e:
            log.error('batch FAILED symbols=%s: %s -- will retry each solo', chunk, e)
            data = {}
        rows = []
        done_rows = []
        missing = [s for s in chunk if s not in data or not data.get(s)]
        for s in list(data.keys()):
            rows.extend(bars_to_rows(s, data[s]))
        for s in missing:
            recovered = False
            invalid = False
            for attempt in range(SOLO_RETRIES):
                try:
                    solo = fetch_batch(client, [s], StockBarsRequest, TimeFrame.Day, DataFeed)
                    bl = solo.get(s, [])
                    if bl:
                        log.warning('BATCH DROP recovered solo: %s (%d bars) -- batch silently dropped it', s, len(bl))
                        rows.extend(bars_to_rows(s, bl))
                        recovered = True
                    break
                except Exception as e:
                    if 'invalid symbol' in str(e).lower():
                        log.warning('%s: not a valid Alpaca data symbol (%s) -- excluded, not LOST', s, e)
                        invalid = True
                        break
                    log.warning('solo retry %d/%d failed for %s: %s', attempt + 1, SOLO_RETRIES, s, e)
                    time.sleep(0.5)
            else:
                log.error('LOST %s: solo retry exhausted, no successful response', s)
                n_lost += 1
                append_csv(LOST_LOG, [{'symbol': s, 'reason': 'solo_retry_exhausted'}], ['symbol', 'reason'])
                continue
            if invalid:
                append_csv(LOST_LOG, [{'symbol': s, 'reason': 'invalid_symbol_excluded'}], ['symbol', 'reason'])
            elif not recovered:
                n_no_data += 1
        if rows:
            shard = pd.DataFrame(rows)
            shard.to_parquet(os.path.join(SHARD_DIR, f'{batch_idx:06d}.parquet'), index=False)
            batch_idx += 1
        for s in chunk:
            done_rows.append({'symbol': s, 'n_bars': sum(1 for r in rows if r['symbol'] == s)})
        append_csv(DONE_LOG, done_rows, ['symbol', 'n_bars'])
        n_ok += len(chunk)
        if (i // BATCH_SYMBOLS) % 50 == 0:
            log.info('progress %d/%d symbols, lost=%d no_data=%d', i + len(chunk), len(todo), n_lost, n_no_data)
        time.sleep(PAUSE_S)

    total_seen = n_lost + n_no_data + sum(1 for _ in open(DONE_LOG)) if os.path.exists(DONE_LOG) else n_lost + n_no_data
    denom = max(1, n_lost + n_no_data)  # conservative: symbols with any signal of being real vs lost
    lost_pct = 100.0 * n_lost / denom
    log.info('run complete: lost=%d no_data=%d lost_pct(of lost+no_data)=%.2f%%', n_lost, n_no_data, lost_pct)
    if lost_pct > LOST_PCT_ERROR:
        log.error('COMPLETENESS GATE FAILED: lost_pct %.2f%% > %.1f%% -- fix before trusting the panel', lost_pct, LOST_PCT_ERROR)
        concat_shards()
        return 1
    concat_shards()
    return 0


if __name__ == '__main__':
    sys.exit(main())
