"""Cell 1,625 data prep -- SPY/IWM minute bars for the index-as-instrument frame.

PREREG_1623.md line 10-11: "SPY / IWM minute bars from data/cache.db (READ-ONLY) or Alpaca
if absent (state the source)." data/cache.db's intraday_bars_1min was VERIFIED (not assumed)
to cover SPY 2025-01-02..2026-03-20 in full, but IWM only 2025-04-07..2025-04-09 (1,094 rows --
essentially unpopulated). So SPY is cache-sourced except 2026-03-21..2026-05-29 (the range the
task named); IWM is Alpaca-sourced for nearly the full 230-day range. This is a genuine data-
availability finding, not a choice -- logged as a WARNING below and in RESULT_1625.md.

Resumable: re-running skips (symbol, day) pairs already present in index_bars_1625.parquet.
data/cache.db is opened READ-ONLY (sqlite URI mode=ro); nothing under data/ is written.
"""
import json
import logging
import os
import sqlite3
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CACHE_DB = os.path.join(ROOT, 'data', 'cache.db')
CAUSAL_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
OUT_PARQUET = os.path.join(HERE, 'index_bars_1625.parquet')
SYMBOLS = ['SPY', 'IWM']

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('fetch_index_bars_1625')


def required_days():
    """The 230 distinct days of the causal_arming_causal.csv base book (TRAIN-H2 + VAL)."""
    df = pd.read_csv(CAUSAL_CSV, usecols=['day'], low_memory=False)
    return sorted(df['day'].unique().tolist())


def from_cache(symbol, days):
    """Read symbol's minute bars for `days` from data/cache.db, READ-ONLY. Returns a DataFrame
    with columns symbol, day, t (UTC ISO8601), o, h, l, c, v, source='cache'."""
    uri = f'file:{CACHE_DB}?mode=ro'
    con = sqlite3.connect(uri, uri=True)
    try:
        placeholders = ','.join('?' * len(days))
        q = (f"SELECT symbol, bar_date, timestamp, open, high, low, close, volume "
             f"FROM intraday_bars_1min WHERE symbol = ? AND bar_date IN ({placeholders})")
        rows = pd.read_sql_query(q, con, params=[symbol] + days)
    finally:
        con.close()
    if rows.empty:
        return rows.assign(source='cache')
    rows = rows.rename(columns={'bar_date': 'day', 'timestamp': 't', 'open': 'o', 'high': 'h',
                                 'low': 'l', 'close': 'c', 'volume': 'v'})
    rows['day'] = rows['day'].astype(str).str.slice(0, 10)
    rows['t'] = pd.to_datetime(rows['t'], utc=True).dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
    rows['source'] = 'cache'
    return rows[['symbol', 'day', 't', 'o', 'h', 'l', 'c', 'v', 'source']]


def from_alpaca(client, symbol, days):
    """Pull symbol's minute bars for `days` from Alpaca SIP, one range request with internal
    pagination, then keep only rows on the requested days. Returns the same schema as from_cache
    with source='alpaca'."""
    import datetime as dt
    from zoneinfo import ZoneInfo
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    ET = ZoneInfo('America/New_York')
    d0 = dt.datetime.strptime(min(days), '%Y-%m-%d').replace(tzinfo=ET)
    d1 = dt.datetime.strptime(max(days), '%Y-%m-%d').replace(tzinfo=ET) + dt.timedelta(days=1)
    # No `limit` cap: alpaca-py's limit is a TOTAL-bars cap, not a page size, and 10,000 silently
    # truncated a multi-month range (found via the completeness gate below) -- omit it so the SDK
    # pages through next_page_token until the full range is returned.
    req = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=TimeFrame.Minute,
                            start=d0.astimezone(dt.timezone.utc), end=d1.astimezone(dt.timezone.utc),
                            feed=DataFeed.SIP)
    try:
        bars = client.get_stock_bars(req)
    except Exception as e:
        log.error('Alpaca fetch FAILED symbol=%s range=%s..%s: %s', symbol, days[0], days[-1], e)
        return pd.DataFrame(columns=['symbol', 'day', 't', 'o', 'h', 'l', 'c', 'v', 'source'])
    data = getattr(bars, 'data', {}) or {}
    recs = []
    dayset = set(days)
    for sym, bar_list in data.items():
        for b in bar_list:
            ts = b.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=dt.timezone.utc)
            ts_utc = ts.astimezone(dt.timezone.utc)
            day_et = ts_utc.astimezone(ET).strftime('%Y-%m-%d')
            if day_et not in dayset:
                continue
            recs.append((sym, day_et, ts_utc.isoformat(), float(b.open), float(b.high),
                         float(b.low), float(b.close), float(b.volume)))
    out = pd.DataFrame(recs, columns=['symbol', 'day', 't', 'o', 'h', 'l', 'c', 'v'])
    out['source'] = 'alpaca'
    return out


def main():
    days = required_days()
    log.info('required days: %d (%s..%s)', len(days), days[0], days[-1])

    existing = pd.DataFrame(columns=['symbol', 'day', 't', 'o', 'h', 'l', 'c', 'v', 'source'])
    if os.path.exists(OUT_PARQUET):
        existing = pd.read_parquet(OUT_PARQUET)
        log.info('resume: %d rows already in %s', len(existing), OUT_PARQUET)

    client = None
    frames = [existing] if len(existing) else []
    for symbol in SYMBOLS:
        have_days = set(existing[existing.symbol == symbol]['day'].unique()) if len(existing) else set()
        need = [d for d in days if d not in have_days]
        if not need:
            log.info('%s: all %d days already present (resumed) -- skipping', symbol, len(days))
            continue

        cache_rows = from_cache(symbol, need)
        cache_days = set(cache_rows['day'].unique()) if len(cache_rows) else set()
        log.info('%s: %d/%d needed days found in data/cache.db (READ-ONLY)', symbol, len(cache_days), len(need))
        frames.append(cache_rows)

        missing = [d for d in need if d not in cache_days]
        if missing:
            log.warning('%s: %d days absent from data/cache.db (%s..%s) -- pulling from Alpaca SIP',
                        symbol, len(missing), missing[0], missing[-1])
            if client is None:
                if ROOT not in sys.path:
                    sys.path.insert(0, ROOT)
                from dotenv import load_dotenv
                load_dotenv(os.path.join(ROOT, '.env'))
                from config import Config
                from alpaca.data.historical import StockHistoricalDataClient
                cfg = Config()
                if not cfg.alpaca_api_key or not cfg.alpaca_api_secret:
                    log.error('missing Alpaca API credentials (ALPACA_API_KEY/ALPACA_API_SECRET) -- cannot fetch')
                    sys.exit(1)
                client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
            alp_rows = from_alpaca(client, symbol, missing)
            got_days = set(alp_rows['day'].unique()) if len(alp_rows) else set()
            still_missing = sorted(set(missing) - got_days)
            if still_missing:
                log.warning('%s: %d days STILL missing after Alpaca pull (holidays/no-data?): %s',
                            symbol, len(still_missing), still_missing[:10])
            log.info('%s: Alpaca returned %d rows across %d/%d requested-missing days',
                     symbol, len(alp_rows), len(got_days), len(missing))
            frames.append(alp_rows)
            time.sleep(0.3)

    combined = pd.concat(frames, ignore_index=True) if frames else existing
    combined = combined.drop_duplicates(subset=['symbol', 't']).sort_values(['symbol', 't']).reset_index(drop=True)
    combined.to_parquet(OUT_PARQUET, index=False)
    log.info('wrote %s: %d rows', OUT_PARQUET, len(combined))

    for symbol in SYMBOLS:
        sub = combined[combined.symbol == symbol]
        cov_days = set(sub['day'].unique())
        lost = sorted(set(days) - cov_days)
        by_src = sub.groupby('source').size().to_dict() if len(sub) else {}
        log.info('%s FINAL: %d/%d days covered, by-source rows=%s, LOST days=%d %s',
                 symbol, len(cov_days), len(days), by_src, len(lost), lost[:10])
        if lost:
            log.error('%s: %d required days have ZERO bars -- these days will be excluded from '
                      'cell 1625 (exclusions counted, never imputed): %s', symbol, len(lost), lost)


if __name__ == '__main__':
    main()
