#!/usr/bin/env python3
"""
Fetch historical news for all backtest trades and cache in DB.

One-time batch job. Fetches from Alpaca News API for each (symbol, date)
pair in the cache CSV. Stores in news_history table for BT to read.

Usage:
    python3 fetch_news_history.py                    # all trades in cache
    python3 fetch_news_history.py --start 2025-01-01 # from a specific date
"""
import argparse
import csv
import re
import sqlite3
import time
import requests
import os
import logging
from datetime import datetime, timedelta
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

REAL_CATALYSTS = {
    'FDA_CLINICAL', 'EARNINGS', 'CONTRACT', 'MA', 'ANALYST',
    'PRODUCT', 'MGMT', 'SEC_FILING', 'CRYPTO',
}

def classify_headline(h: str) -> str:
    """Classify a news headline into category."""
    h = h.lower()
    if re.search(r'fda|phase [123]|clinical|trial|drug|therapy|approv|orphan|ind |nda|biologics', h): return 'FDA_CLINICAL'
    if re.search(r'earn|revenue|quarter|q[1-4]|eps|guidance|beat|miss|fiscal', h): return 'EARNINGS'
    if re.search(r'contract|deal|agreement|partner|collaborat|licens|amend|award', h): return 'CONTRACT_DEAL'
    if re.search(r'acqui|merge|buyout|takeover', h): return 'MA'
    if re.search(r'analyst|upgrade|downgrade|price target|initiat.*coverage', h): return 'ANALYST'
    if re.search(r'launch|new product|expansion|patent|initiative', h): return 'PRODUCT'
    if re.search(r'insider|ceo|cfo|director|appoint|resign|hire', h): return 'MGMT'
    if re.search(r'offering|ipo|shelf|registration|prospectus', h): return 'SEC_FILING'
    if re.search(r'why is|why are|stocks? moving|here are \d+|top \d+ stocks', h): return 'GARBAGE'
    if re.search(r'bitcoin|crypto|blockchain|mining', h): return 'CRYPTO'
    return 'OTHER'


def fetch_and_store(symbol: str, trade_date: str, entry_time_et: str,
                    headers: dict, conn: sqlite3.Connection) -> dict:
    """Fetch news for a symbol/date and store in DB. Returns best category."""
    # Check if already cached
    existing = conn.execute(
        "SELECT category, is_catalyst FROM news_history WHERE symbol=? AND trade_date=? LIMIT 1",
        (symbol, trade_date)
    ).fetchone()
    if existing:
        return {'cat': existing[0], 'cached': True}

    # Build time window: prev day 4PM ET (21:00 UTC) to entry time
    entry_h, entry_m = int(entry_time_et.split(':')[0]), int(entry_time_et.split(':')[1])
    utc_h = entry_h + 5  # EST to UTC
    entry_utc = f"{trade_date}T{utc_h:02d}:{entry_m:02d}:00Z"
    prev_date = (datetime.strptime(trade_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
    start = f"{prev_date}T21:00:00Z"

    try:
        url = (f"https://data.alpaca.markets/v1beta1/news?"
               f"symbols={symbol}&start={start}&end={entry_utc}&limit=10&sort=desc")
        resp = requests.get(url, headers=headers, timeout=10)
        articles = resp.json().get('news', [])
    except Exception as e:
        logger.warning(f"{symbol} {trade_date}: fetch failed: {e}")
        return {'cat': 'ERROR', 'cached': False}

    if not articles:
        # Store a NO_NEWS marker
        conn.execute(
            "INSERT OR IGNORE INTO news_history (symbol, trade_date, headline, category, is_catalyst) "
            "VALUES (?, ?, '', 'NO_NEWS', 0)",
            (symbol, trade_date)
        )
        conn.commit()
        return {'cat': 'NO_NEWS', 'cached': False}

    # Store all articles
    best_cat = 'NO_NEWS'
    for a in articles:
        headline = (a.get('headline') or '')[:500]
        article_time = (a.get('created_at') or '')[:30]
        source = (a.get('source') or '')[:50]
        cat = classify_headline(headline)
        is_catalyst = 1 if cat in REAL_CATALYSTS else 0

        conn.execute(
            "INSERT OR IGNORE INTO news_history "
            "(symbol, trade_date, article_time, headline, source, category, is_catalyst) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (symbol, trade_date, article_time, headline, source, cat, is_catalyst)
        )

        if cat in REAL_CATALYSTS and best_cat not in REAL_CATALYSTS:
            best_cat = cat
        elif best_cat == 'NO_NEWS':
            best_cat = cat

    conn.commit()
    return {'cat': best_cat, 'cached': False}


def main():
    parser = argparse.ArgumentParser(description="Fetch historical news for BT trades")
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--end", type=str, default="2026-12-31")
    parser.add_argument("--cache-file", type=str, default="data/bull_flag_cache_e50_x30.csv")
    args = parser.parse_args()

    cfg = Config()
    headers = {
        'APCA-API-KEY-ID': os.environ.get('APCA_API_KEY_ID', cfg.alpaca_api_key),
        'APCA-API-SECRET-KEY': os.environ.get('APCA_API_SECRET_KEY', cfg.alpaca_api_secret),
    }

    # Load unique (symbol, date, entry_time) from cache
    pairs = {}
    with open(args.cache_file) as f:
        for row in csv.DictReader(f):
            if args.start <= row['date'] <= args.end:
                key = (row['symbol'], row['date'])
                if key not in pairs:
                    pairs[key] = row['entry_time_et']

    logger.info(f"Unique (symbol, date) pairs to fetch: {len(pairs)}")

    conn = sqlite3.connect('data/cache.db')

    # Check existing
    existing = conn.execute("SELECT COUNT(DISTINCT symbol || trade_date) FROM news_history").fetchone()[0]
    logger.info(f"Already cached: {existing}")

    fetched = 0
    cached = 0
    for (sym, date), entry_time in sorted(pairs.items()):
        result = fetch_and_store(sym, date, entry_time, headers, conn)
        if result.get('cached'):
            cached += 1
        else:
            fetched += 1
            if fetched % 20 == 0:
                logger.info(f"  {fetched}/{len(pairs)}: {sym} {date} → {result['cat']}")
            time.sleep(0.3)

    conn.close()
    logger.info(f"Done: {fetched} fetched, {cached} cached, {len(pairs)} total")


if __name__ == "__main__":
    main()
