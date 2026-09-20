#!/usr/bin/env python3
"""Pass 2 step 2: pull 14:55-16:00 ET 1-min bars for need_pull.csv rows from
Alpaca (SIP), batched per day, into research/lev_rebalance/bars_1500_1600.db
(OUR OWN sqlite — never cache.db). Resumable via a `done` table.
Also backfills missing wrapper daily bars (for the flow proxy) into the
same db's `wrapper_daily` table.
"""
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from data_sources.alpaca_client import AlpacaClient  # noqa: E402

OUT = os.path.join(ROOT, 'research', 'lev_rebalance')
BARS_DB = os.path.join(OUT, 'bars_1500_1600.db')
ET = ZoneInfo('America/New_York')
UTC = ZoneInfo('UTC')
BATCH = 150
RETRIES = 3
RETRY_SLEEP_S = 4.0


def log(*a):
    print(*a, flush=True)


def et_window_utc(d, h0=14, m0=55, h1=16, m1=0):
    lo = datetime(d.year, d.month, d.day, h0, m0, tzinfo=ET).astimezone(UTC)
    hi = datetime(d.year, d.month, d.day, h1, m1, tzinfo=ET).astimezone(UTC)
    return lo, hi


def init_db(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS bars_1500_1600 (
        symbol TEXT, bar_date TEXT, timestamp TEXT, open REAL, high REAL,
        low REAL, close REAL, volume INTEGER,
        PRIMARY KEY (symbol, bar_date, timestamp))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS pull_done (
        symbol TEXT, bar_date TEXT, n_bars INTEGER, pulled_at TEXT,
        PRIMARY KEY (symbol, bar_date))""")
    conn.commit()


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main():
    need_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, 'need_pull.csv')
    log(f'need_pull source: {need_path}')
    need = pd.read_csv(need_path)
    need['bar_date'] = pd.to_datetime(need['bar_date']).dt.date
    by_day = defaultdict(list)
    for _, r in need.iterrows():
        by_day[r['bar_date']].append(r['symbol'])

    conn = sqlite3.connect(BARS_DB)
    init_db(conn)
    done = set(conn.execute("SELECT symbol, bar_date FROM pull_done").fetchall())

    api_key = os.environ['ALPACA_API_KEY']
    api_secret = os.environ['ALPACA_API_SECRET']
    alpaca = AlpacaClient(api_key, api_secret, paper=True)

    days = sorted(by_day.keys())
    log(f'{len(days)} days to process, {len(need)} total (symbol,day) pairs, batch={BATCH}')

    stats = dict(fetched_pairs=0, empty_pairs=0, failed_pairs=0, skipped_already_done=0)
    t0 = time.time()
    for di, d in enumerate(days):
        syms = sorted(set(by_day[d]))
        todo = [s for s in syms if (s, str(d)) not in done]
        stats['skipped_already_done'] += len(syms) - len(todo)
        if not todo:
            continue
        lo, hi = et_window_utc(d)
        for chunk in batches(todo, BATCH):
            bars_map = None
            for attempt in range(RETRIES):
                try:
                    bars_map = alpaca.get_1min_bars_range_multi(chunk, lo, hi)
                    break
                except Exception as e:
                    log(f'{d} chunk({len(chunk)}) attempt {attempt+1} failed: {e}')
                    time.sleep(RETRY_SLEEP_S)
            if bars_map is None:
                stats['failed_pairs'] += len(chunk)
                continue
            for sym in chunk:
                df = bars_map.get(sym)
                n = 0 if df is None else len(df)
                if n:
                    rows = []
                    for _, b in df.iterrows():
                        ts = pd.Timestamp(b['timestamp'])
                        if ts.tzinfo is None:
                            ts = ts.tz_localize('UTC')
                        rows.append((sym, str(d), ts.isoformat(), float(b['open']), float(b['high']),
                                     float(b['low']), float(b['close']), int(b['volume'])))
                    conn.executemany(
                        "INSERT OR REPLACE INTO bars_1500_1600 VALUES (?,?,?,?,?,?,?,?)", rows)
                    stats['fetched_pairs'] += 1
                else:
                    stats['empty_pairs'] += 1
                conn.execute(
                    "INSERT OR REPLACE INTO pull_done VALUES (?,?,?,?)",
                    (sym, str(d), n, datetime.now(UTC).isoformat()))
            conn.commit()
        if di % 20 == 0 or di == len(days) - 1:
            elapsed = time.time() - t0
            log(f'[{di+1}/{len(days)}] day={d} done — elapsed={elapsed:.0f}s stats={stats}')

    log('FINAL', json.dumps(stats))
    stats_name = 'pull_stats_' + os.path.basename(need_path).replace('.csv', '') + '.json'
    with open(os.path.join(OUT, stats_name), 'w') as f:
        json.dump(stats, f, indent=2)
    conn.close()


if __name__ == '__main__':
    main()
