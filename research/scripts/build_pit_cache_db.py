"""Build the point-in-time SIDE cache (2026-09-05).

A SQLite file with the production cache schema (daily_bars +
intraday_bars_1min, persistence/database.py) holding ONLY the candidate
symbol-days the cache-based universes never saw (delisted / renamed /
filtered names), so the unchanged ORB feature study and the unchanged BF
Stage-1 can be pointed at it (ORB_CACHE_DB / a --config with
database.cache_path) and their output unioned with the production books.
The production cache.db is opened read-only and never written.

Sources:
  daily_bars        <- Databento EQUS.SUMMARY parquet (all dates for the
                       missing symbols, so LAG/prev-day/20d features work)
                       + SPY daily rows copied from cache.db (regime/context)
  intraday_bars_1min<- Databento EQUS.MINI side DB (the missing symbol-days,
                       RTH) + SPY 1-min rows copied from cache.db

Usage: python3 research/scripts/build_pit_cache_db.py [OUT_DB]
"""
import os
import sqlite3
import sys

import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
ROOT = '/home/ec2-user/onemil'
PARQUET = f'{ROOT}/data/research/databento/equs_daily_2025_2026.parquet'
SIDE_1MIN = f'{ROOT}/data/research/databento/pit_bars_1min.db'
CLASSIFIED = f'{ROOT}/data/research/databento/pit_missing_classified.csv'
OUT = sys.argv[1] if len(sys.argv) > 1 else f'{ROOT}/data/research/databento/pit_cache.db'

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_bars (
    symbol VARCHAR(10) NOT NULL, bar_date DATE NOT NULL, open REAL NOT NULL,
    high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    volume INTEGER NOT NULL, fetched_at TIMESTAMP NOT NULL,
    PRIMARY KEY (symbol, bar_date));
CREATE INDEX IF NOT EXISTS idx_daily_bars_date ON daily_bars(bar_date);
CREATE TABLE IF NOT EXISTS daily_bars_provisional (
    symbol VARCHAR(10) NOT NULL, bar_date DATE NOT NULL, open REAL NOT NULL,
    high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    volume INTEGER NOT NULL, fetched_at TIMESTAMP NOT NULL,
    PRIMARY KEY (symbol, bar_date));
CREATE TABLE IF NOT EXISTS intraday_bars_1min (
    symbol VARCHAR(10) NOT NULL, bar_date DATE NOT NULL, timestamp TIMESTAMP NOT NULL,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    volume INTEGER NOT NULL, PRIMARY KEY (symbol, timestamp));
CREATE INDEX IF NOT EXISTS idx_intraday_bars_symbol_date ON intraday_bars_1min(symbol, bar_date);
"""


def main() -> None:
    if os.path.exists(OUT):
        os.remove(OUT)   # side artifact, rebuilt from sources every time
    out = sqlite3.connect(OUT)
    out.executescript(SCHEMA)
    cls = pd.read_csv(CLASSIFIED)
    cls = cls[cls['book'].isin(['orb', 'bf'])]
    symbols = sorted(set(cls['symbol']) - {'ZVZZT'})
    print(f"missing symbols (orb ∪ bf): {len(symbols):,}")

    daily = pd.read_parquet(PARQUET)
    daily = daily[daily['symbol'].isin(symbols) & (daily['volume'] > 0)]
    daily = daily.drop_duplicates(['symbol', 'bar_date'])
    daily['fetched_at'] = '2026-09-05T00:00:00+00:00'
    daily[['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume', 'fetched_at']] \
        .to_sql('daily_bars', out, if_exists='append', index=False)
    print(f"daily_bars from Databento: {len(daily):,} rows")
    del daily

    prod = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
    spy_d = pd.read_sql("SELECT symbol, bar_date, open, high, low, close, volume, fetched_at "
                        "FROM daily_bars WHERE symbol='SPY'", prod)
    spy_d.to_sql('daily_bars', out, if_exists='append', index=False)
    spy_i = pd.read_sql("SELECT symbol, bar_date, timestamp, open, high, low, close, volume "
                        "FROM intraday_bars_1min WHERE symbol='SPY'", prod)
    spy_i.to_sql('intraday_bars_1min', out, if_exists='append', index=False)
    prod.close()
    print(f"SPY copied: {len(spy_d)} daily, {len(spy_i):,} 1-min rows")

    side = sqlite3.connect(f'file:{SIDE_1MIN}?mode=ro', uri=True, timeout=120)
    n = 0
    for chunk in pd.read_sql("SELECT symbol, day AS bar_date, t AS timestamp, o AS open, h AS high, "
                             "l AS low, c AS close, v AS volume FROM bars", side, chunksize=200_000):
        chunk = chunk[chunk['symbol'].isin(symbols)]
        chunk['volume'] = chunk['volume'].round().astype('int64')
        chunk.to_sql('intraday_bars_1min', out, if_exists='append', index=False)
        n += len(chunk)
    side.close()
    print(f"intraday_bars_1min from Databento: {n:,} rows")
    out.commit()
    q = out.execute("SELECT COUNT(DISTINCT symbol||bar_date) FROM intraday_bars_1min WHERE symbol<>'SPY'").fetchone()[0]
    print(f"symbol-days with 1-min bars: {q:,} -> {OUT}")
    out.close()


if __name__ == '__main__':
    main()
