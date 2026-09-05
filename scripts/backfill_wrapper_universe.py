"""Backfill leveraged wrappers into the production cache (2026-09-05, owner GO).

The universe-builder Step 9 fix adds NEW wrappers to `daily_bars` from the
next nightly build on; this one-off fills the history the bull-flag filter
withheld since 2026-04-04 so the ORB/ignition backtests see the same universe
live now sees:
  1. daily bars (Alpaca, same write path as the nightly job: Database.save_daily_bars)
     for every active, tradable wrapper Alpaca lists that daily_bars lacks;
  2. 1-min RTH bars (Database.save_intraday_bars) for the ORB candidate
     symbol-days of those wrappers (the point-in-time diff keys), so the
     feature study can simulate them.
Symbols are classified with the PRODUCTION rule
(`AlpacaClient._is_common_stock(..., exclude_leveraged=False)` True AND the
default False) — warrants/units/preferred/rights stay out.

Usage: python3 scripts/backfill_wrapper_universe.py [--dry-run]
"""
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, '/home/ec2-user/onemil')
sys.stdout.reconfigure(line_buffering=True)
load_dotenv('/home/ec2-user/onemil/.env')
from data_sources.alpaca_client import AlpacaClient  # noqa: E402
from persistence.database import get_database  # noqa: E402
from trading.orb_csv import read_orb_csv  # noqa: E402

ROOT = '/home/ec2-user/onemil'
CACHE = f'{ROOT}/data/cache.db'
CLASSIFIED = f'{ROOT}/data/research/databento/pit_missing_classified.csv'
ASSETS = f'{ROOT}/data/research/databento/alpaca_assets_all_20260905.csv'
START = date(2025, 1, 1)
CHUNK = 100
ET = 'America/New_York'


def wrapper_symbols() -> list:
    """Active tradable symbols the production rule keeps ONLY with exclude_leveraged=False."""
    a = read_orb_csv(ASSETS, keep_default_na=False)
    a = a[(a['status'] == 'active') & (a['tradable'].astype(str) == 'True')]
    out = [s for s, n in zip(a['symbol'], a['name'])
           if AlpacaClient._is_common_stock(s, n, exclude_leveraged=False)
           and not AlpacaClient._is_common_stock(s, n)]
    return sorted(set(out))


def main() -> None:
    dry = '--dry-run' in sys.argv
    client = AlpacaClient(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_API_SECRET'], paper=False)
    db = get_database(db_path=f'{ROOT}/data/onemil.db', cache_path=CACHE, trades_path=f'{ROOT}/data/trades.db')
    wr = wrapper_symbols()
    con = sqlite3.connect(f'file:{CACHE}?mode=ro', uri=True, timeout=60)
    have = set(r[0] for r in con.execute("SELECT DISTINCT symbol FROM daily_bars"))
    con.close()
    missing = [s for s in wr if s not in have]
    print(f"wrappers per production rule: {len(wr)} | already in daily_bars: {len(wr) - len(missing)} | to backfill: {len(missing)}")
    if dry:
        print(missing[:50], '...')
        return
    today = date.today()
    n_rows = 0
    for i in range(0, len(missing), CHUNK):
        batch = missing[i:i + CHUNK]
        bars = client.get_daily_bars_range(batch, START, today)
        flat = [{'symbol': s, 'date': b['date'].isoformat() if isinstance(b['date'], date) else b['date'],
                 'open': b['open'], 'high': b['high'], 'low': b['low'], 'close': b['close'], 'volume': b['volume']}
                for s, bl in bars.items() for b in bl]
        db.save_daily_bars(flat)
        n_rows += len(flat)
        print(f"  daily bars chunk {i // CHUNK + 1}/{(len(missing) + CHUNK - 1) // CHUNK}: {len(bars)} symbols, +{len(flat):,} rows")
    print(f"daily_bars backfilled: {n_rows:,} rows for {len(missing)} wrappers")

    # 1-min RTH bars for the ORB candidate symbol-days of ALL rule-wrappers
    # (idempotent: only symbol-days with no cached 1-min bars are fetched)
    cls = read_orb_csv(CLASSIFIED)
    keys = cls[(cls['book'] == 'orb') & (cls['symbol'].isin(wr))][['symbol', 'bar_date']].drop_duplicates()
    con = sqlite3.connect(f'file:{CACHE}?mode=ro', uri=True, timeout=60)
    cached = set(con.execute("SELECT DISTINCT symbol || '|' || bar_date FROM intraday_bars_1min "
                             "WHERE bar_date >= '2025-01-01'").fetchall())
    con.close()
    cached = {c[0] for c in cached}
    keys = keys[[f"{s}|{d}" not in cached for s, d in zip(keys['symbol'], keys['bar_date'])]]
    print(f"ORB candidate symbol-days needing 1-min bars: {len(keys)}")
    n_bars = 0
    for day, g in keys.groupby('bar_date'):
        syms = sorted(g['symbol'])
        st = pd.Timestamp(day).tz_localize(ET) + pd.Timedelta(hours=9, minutes=30)
        en = pd.Timestamp(day).tz_localize(ET) + pd.Timedelta(hours=16)
        try:
            by_sym = client.get_1min_bars_range_multi(syms, st.tz_convert('UTC').to_pydatetime(),
                                                      en.tz_convert('UTC').to_pydatetime())
        except Exception as e:
            print(f"  WARN {day}: 1-min fetch failed ({e}) — skipped {len(syms)} symbols")
            continue
        for s, df in (by_sym or {}).items():
            if df is None or len(df) == 0:
                continue
            db.save_intraday_bars(s, day, df.to_dict('records'))
            n_bars += len(df)
    print(f"intraday_bars_1min backfilled: {n_bars:,} bars")


if __name__ == '__main__':
    main()
