"""Ignition capcheck — point-in-time top-up (2026-09-05).

Fetches 1-min RTH bars from Databento EQUS.MINI (ohlcv-1m) for the
candidate symbol-days the cache-based universe never saw
(`universe_pit_missing.csv` from build_universe_pit.py — delisted /
renamed / not-yet-listed-at-seed-time names Alpaca may not serve) and
INSERT-OR-IGNOREs them into `topup.db::bars` — the same table capsim.py
already reads for uncovered keys. Per-day requests (symbols resolve
point-in-time per date, so delisted raw symbols work); resumable via a
state file; never touches cache.db. Cost probe 2026-09-05: ~$0.00017 per
symbol-day (~$0.45 for the whole set).

Usage: python3 fetch_missing_databento.py [END_DATE] [MISSING_CSV] [DB_PATH] [STATE_JSON]
  defaults: 2026-08-14, universe_pit_missing.csv, topup.db, fetch_databento_state.json
  (any CSV with symbol,bar_date columns works — ORB/BF point-in-time top-ups
  go to a side DB, never cache.db).
"""
import json
import os
import sqlite3
import sys
import time

import databento as db
import pandas as pd
from dotenv import load_dotenv

sys.stdout.reconfigure(line_buffering=True)
load_dotenv('/home/ec2-user/onemil/.env')
D = '/home/ec2-user/onemil/research/ignition_capcheck'
END = sys.argv[1] if len(sys.argv) > 1 else '2026-08-14'
MISSING = sys.argv[2] if len(sys.argv) > 2 else f'{D}/universe_pit_missing.csv'
TOPUP = sys.argv[3] if len(sys.argv) > 3 else f'{D}/topup.db'
STATE = sys.argv[4] if len(sys.argv) > 4 else f'{D}/fetch_databento_state.json'
TEST_TICKERS = {'ZVZZT'}


def rth_bars_for_day(client, day: str, symbols: list) -> pd.DataFrame:
    """One EQUS.MINI ohlcv-1m request for `symbols` on `day`, RTH only."""
    nxt = str((pd.Timestamp(day) + pd.Timedelta(days=1)).date())
    st = client.timeseries.get_range(dataset='EQUS.MINI', schema='ohlcv-1m',
                                     symbols=symbols, stype_in='raw_symbol',
                                     start=day, end=nxt)
    df = st.to_df().reset_index()
    if df.empty:
        return df
    ts = pd.to_datetime(df['ts_event'], utc=True)
    et = ts.dt.tz_convert('America/New_York')
    m = et.dt.hour * 60 + et.dt.minute
    df = df[(m >= 570) & (m < 960)].copy()
    df['t'] = ts[df.index].dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
    df['day'] = day
    return df[['symbol', 'day', 't', 'open', 'high', 'low', 'close', 'volume']]


def main() -> None:
    miss = pd.read_csv(MISSING, dtype={'symbol': str}, keep_default_na=False)
    miss = miss[(miss['bar_date'] <= END) & ~miss['symbol'].isin(TEST_TICKERS)]
    by_day = miss.groupby('bar_date')['symbol'].apply(list).to_dict()
    done = set(json.load(open(STATE))['done_days']) if os.path.exists(STATE) else set()
    days = [d for d in sorted(by_day) if d not in done]
    print(f"missing keys: {len(miss):,} across {len(by_day)} days; "
          f"{len(done)} done, {len(days)} to go")
    client = db.Historical(os.environ['DATABENTO_API_KEY'])
    conn = sqlite3.connect(TOPUP, timeout=120)
    conn.execute("CREATE TABLE IF NOT EXISTS bars (symbol TEXT, day TEXT, t TEXT, o REAL, h REAL, "
                 "l REAL, c REAL, v REAL, PRIMARY KEY (symbol, day, t))")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bars_day ON bars(day)")
    total = 0
    for i, day in enumerate(days):
        syms = sorted(set(by_day[day]))
        df = None
        for att in range(4):
            try:
                df = rth_bars_for_day(client, day, syms)
                break
            except Exception as e:  # network / 429 / symbology — retry then report
                print(f"  WARN {day} attempt {att}: {str(e)[:200]}")
                time.sleep(3 + 5 * att)
        if df is None:
            print(f"FAIL {day}: giving up ({len(syms)} symbols)")
            continue
        got = df['symbol'].nunique() if not df.empty else 0
        if not df.empty:
            conn.executemany(
                "INSERT OR IGNORE INTO bars(symbol, day, t, o, h, l, c, v) VALUES (?,?,?,?,?,?,?,?)",
                df.itertuples(index=False, name=None))
            conn.commit()
        total += len(df)
        done.add(day)
        json.dump({'done_days': sorted(done)}, open(STATE, 'w'))
        print(f"  {i + 1}/{len(days)} {day}: {got}/{len(syms)} symbols, +{len(df):,} bars (cum {total:,})")
    conn.close()
    print("FETCH DONE")


if __name__ == '__main__':
    main()
