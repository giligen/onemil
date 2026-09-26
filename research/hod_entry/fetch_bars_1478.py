"""PREREG_1478 amendment item 1 — single-source fresh SIP bar store.

Builds research/hod_entry/bars_fills_1478.db: ONE fresh Alpaca SIP 1-minute bar store, fetched
identically for every distinct (symbol, day) of the base book (cell 1,438 fills,
`causal_arming_causal.csv` rows status=='fill'), so no bar-derived feature can leak which of the
two legacy stores (cache.db / bars_sip.db) happened to hold more bars for a given symbol-day —
store identity IS the look-ahead cohort named by the amendment.

Span: 04:00-20:00 ET (pre-market through after-hours), built from America/New_York wall-clock
minutes so DST is handled exactly (no fixed UTC-offset padding).

Resumable per day: a day already recorded in fetch_log for ALL of its symbols is skipped on
re-run. Never touches data/cache.db, research/bf_zero/bars_sip.db, or any other existing store.

Usage:
  python3 research/hod_entry/fetch_bars_1478.py                 # run to completion
  python3 research/hod_entry/fetch_bars_1478.py --limit-days N  # smoke test
"""
import argparse
import csv
import datetime as dt
import logging
import os
import sqlite3
import sys
import time
from zoneinfo import ZoneInfo

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

BOOK_CSV = os.path.join(ROOT, 'research/hod_entry/causal_arming_causal.csv')
DB_PATH = os.path.join(HERE, 'bars_fills_1478.db')

ET = ZoneInfo('America/New_York')
SPAN_START_H, SPAN_END_H = 4, 20   # 04:00-20:00 ET
BATCH_SYMBOLS = 10  # alpaca-py's multi-symbol BarSet pagination silently drops symbols past
                    # ~10,000 total bars/page (measured: 47-symbol/16h requests dropped 19/47);
                    # 10 symbols * <=960 bars/day stays under one page every time
PAGE_LIMIT = 10000
PAUSE_S = 0.2
MIN_RTH_BARS = 100                 # completeness gate: < 100 RTH (09:30-16:00 ET) bars => LOST
LOST_PCT_ERROR = 2.0               # ERROR + stop if LOST > this % of requested

log = logging.getLogger('fetch_bars_1478')


def et_window_utc(day: str):
    """(start_utc, end_utc) datetimes for 04:00-20:00 ET on `day`, DST-correct."""
    d = dt.date.fromisoformat(day)
    start_et = dt.datetime(d.year, d.month, d.day, SPAN_START_H, 0, tzinfo=ET)
    end_et = dt.datetime(d.year, d.month, d.day, SPAN_END_H, 0, tzinfo=ET)
    return start_et.astimezone(dt.timezone.utc), end_et.astimezone(dt.timezone.utc)


def load_symbol_days():
    """Distinct (day, symbol) pairs from the base book's status=='fill' rows."""
    pairs = set()
    with open(BOOK_CSV, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('status') == 'fill':
                pairs.add((row['day'], row['symbol']))
    return pairs


def init_db(con):
    con.execute('''CREATE TABLE IF NOT EXISTS bars (
        symbol TEXT, day TEXT, t TEXT, o REAL, h REAL, l REAL, c REAL, v REAL,
        PRIMARY KEY(symbol, day, t))''')
    con.execute('''CREATE TABLE IF NOT EXISTS fetch_log (
        symbol TEXT, day TEXT, n_bars INTEGER, fetched_at TEXT,
        PRIMARY KEY(symbol, day))''')
    con.commit()


def done_pairs(con):
    return set(map(tuple, con.execute('SELECT symbol, day FROM fetch_log').fetchall()))


def rth_bar_count(rows):
    """Count of RTH (09:30-16:00 ET) bars among fetched rows (t is UTC ISO)."""
    n = 0
    for t in rows:
        ts = dt.datetime.fromisoformat(t).astimezone(ET)
        m = ts.hour * 60 + ts.minute
        if 570 <= m < 960:
            n += 1
    return n


def fetch_day(client, symbols, day):
    """Fetch one day's bars for up to BATCH_SYMBOLS symbols; returns {symbol: [(t,o,h,l,c,v),...]}."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    start, end = et_window_utc(day)
    out = {s: [] for s in symbols}
    for i in range(0, len(symbols), BATCH_SYMBOLS):
        chunk = symbols[i:i + BATCH_SYMBOLS]
        req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Minute,
                                start=start, end=end, feed=DataFeed.SIP, limit=PAGE_LIMIT)
        try:
            bars = client.get_stock_bars(req)
        except Exception as e:
            log.error('fetch FAILED day=%s chunk=%s: %s', day, chunk[:3], e)
            time.sleep(1.0)
            continue
        data = getattr(bars, 'data', {}) or {}
        for sym, bar_list in data.items():
            for b in bar_list:
                ts = b.timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=dt.timezone.utc)
                out.setdefault(sym, []).append((
                    ts.astimezone(dt.timezone.utc).isoformat(),
                    float(b.open), float(b.high), float(b.low), float(b.close), float(b.volume)))
        time.sleep(PAUSE_S)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--limit-days', type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, '.env'))
    from config import Config
    from alpaca.data.historical import StockHistoricalDataClient
    cfg = Config()
    if not cfg.alpaca_api_key or not cfg.alpaca_api_secret:
        log.error('missing Alpaca API credentials (ALPACA_API_KEY/ALPACA_API_SECRET) - cannot fetch')
        return 1
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)

    pairs = load_symbol_days()
    by_day = {}
    for day, sym in pairs:
        by_day.setdefault(day, []).append(sym)
    days = sorted(by_day)
    if a.limit_days:
        days = days[:a.limit_days]
    log.info('base book: %d distinct symbol-days across %d distinct days', len(pairs), len(sorted(by_day)))

    con = sqlite3.connect(DB_PATH)
    init_db(con)
    already = done_pairs(con)
    log.info('resume: %d symbol-days already fetched', len(already))

    n_requested = sum(len(v) for v in by_day.values()) if not a.limit_days else sum(len(by_day[d]) for d in days)
    n_received = 0
    n_lost = 0
    rth_counts = []
    lost_examples = []

    for di, day in enumerate(days, 1):
        syms = sorted(set(by_day[day]))
        todo = [s for s in syms if (s, day) not in already]
        if not todo:
            log.info('[%d/%d] day=%s: all %d symbols already fetched, skip', di, len(days), day, len(syms))
            for s in syms:
                cnt = con.execute('SELECT n_bars FROM fetch_log WHERE symbol=? AND day=?', (s, day)).fetchone()
                if cnt:
                    n_received += 1
                    rth_counts.append(cnt[0])
                    if cnt[0] < MIN_RTH_BARS:
                        n_lost += 1
                        if len(lost_examples) < 20:
                            lost_examples.append((day, s, cnt[0]))
            continue
        log.info('[%d/%d] day=%s: fetching %d symbols (%d already done)', di, len(days), day, len(todo), len(syms) - len(todo))
        data = fetch_day(client, todo, day)
        now = dt.datetime.utcnow().isoformat()
        for s in todo:
            rows = data.get(s, [])
            n_rth = rth_bar_count([r[0] for r in rows])
            if rows:
                con.executemany(
                    'INSERT OR IGNORE INTO bars (symbol, day, t, o, h, l, c, v) VALUES (?,?,?,?,?,?,?,?)',
                    [(s, day, t, o, h, l, c, v) for (t, o, h, l, c, v) in rows])
            con.execute(
                'INSERT OR REPLACE INTO fetch_log (symbol, day, n_bars, fetched_at) VALUES (?,?,?,?)',
                (s, day, n_rth, now))
            n_received += 1
            rth_counts.append(n_rth)
            if n_rth < MIN_RTH_BARS:
                n_lost += 1
                if len(lost_examples) < 20:
                    lost_examples.append((day, s, n_rth))
        con.commit()
        pct = 100.0 * n_lost / max(n_received, 1)
        log.info('  running: received=%d lost=%d (%.2f%%) median_rth=%.0f',
                  n_received, n_lost, pct,
                  sorted(rth_counts)[len(rth_counts) // 2] if rth_counts else float('nan'))
        if pct > LOST_PCT_ERROR:
            log.error('LOST %.2f%% > %.1f%% threshold at day %s (%d/%d received) - STOPPING per completeness gate',
                       pct, LOST_PCT_ERROR, day, n_received, n_requested)
            break

    rth_counts_sorted = sorted(rth_counts)
    median_rth = rth_counts_sorted[len(rth_counts_sorted) // 2] if rth_counts_sorted else float('nan')
    pct_final = 100.0 * n_lost / max(n_received, 1)
    log.info('DONE: requested=%d received=%d lost=%d (%.2f%% of received) median_rth_bars=%.0f',
              n_requested, n_received, n_lost, pct_final, median_rth)
    if lost_examples:
        log.warning('LOST examples (day, symbol, n_rth_bars): %s', lost_examples)
    con.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
