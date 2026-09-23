"""Fetch Alpaca SIP 1-minute bars (04:00-20:00 ET window, padded for EST/EDT) for the 2024H2 candidates.

Writes ONLY research/day_breadth/y2024/bars.db, table bars(symbol, day, t, o, h, l, c, v) — the schema of
bars_sip.db / bars_rth.db, t = ISO UTC — so run_consol.fetch_day_bars_dual reads it unchanged. Batched by day,
100 symbols per request, 3 retries, resumable via fetch_state.json (a day is marked done only when every batch
of it succeeded). Market-hours rule: refuses to start between 13:25 and 20:05 UTC.

Usage: python3 research/day_breadth/y2024/fetch.py
"""
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DB = HERE / 'bars.db'
STATE = HERE / 'fetch_state.json'
BATCH = 100


def main():
    now = int(datetime.now(timezone.utc).strftime('%H%M'))
    if 1325 <= now < 2005:
        sys.exit(f'refusing to fetch inside the market-hours blackout (UTC {now:04d})')
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    from config import Config
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    cfg = Config()
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    cands = pd.read_csv(HERE / 'candidates.csv', keep_default_na=False, na_values=[''])  # ticker NA is real
    by_day = cands.groupby('bar_date').symbol.apply(lambda s: sorted(set(s))).to_dict()
    done = set(json.loads(STATE.read_text())['days']) if STATE.exists() else set()
    con = sqlite3.connect(DB)
    con.execute('CREATE TABLE IF NOT EXISTS bars (symbol TEXT, day TEXT, t TEXT, o REAL, h REAL, l REAL, c REAL, v REAL)')
    con.execute('CREATE INDEX IF NOT EXISTS idx_bars_sym_day ON bars(symbol, day)')
    total, t0 = 0, time.time()
    days = [d for d in sorted(by_day) if d not in done]
    print(f'{len(days)} days to fetch ({len(done)} done), {sum(len(by_day[d]) for d in days)} symbol-days', flush=True)
    for i, day in enumerate(days, 1):
        start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc) + timedelta(hours=8)
        end = start + timedelta(hours=16)
        syms, rows, ok = by_day[day], 0, True
        for j in range(0, len(syms), BATCH):
            chunk = syms[j:j + BATCH]
            for attempt in range(3):
                try:
                    df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Minute,
                                                                start=start, end=end, feed='sip')).df
                    if df is not None and len(df):
                        df = df.reset_index()
                        pd.DataFrame(dict(symbol=df.symbol, day=day,
                                          t=pd.to_datetime(df.timestamp, utc=True).dt.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
                                          o=df.open, h=df.high, l=df.low, c=df.close, v=df.volume)
                                     ).to_sql('bars', con, if_exists='append', index=False)
                        rows += len(df)
                    break
                except Exception as e:  # noqa: BLE001
                    print(f'[WARNING] {day} batch {j // BATCH} attempt {attempt + 1}: {e}', flush=True)
                    bad = str(e).split('invalid symbol: ')[-1].split('"')[0] if 'invalid symbol' in str(e) else None
                    if bad and bad in chunk:
                        chunk = [s for s in chunk if s != bad]
                        print(f'[WARNING] {day}: dropped invalid symbol {bad} from the batch (no Alpaca data)', flush=True)
                        if not chunk:
                            break
                        continue
                    time.sleep(2 * (attempt + 1))
            else:
                print(f'[ERROR] {day} batch {j // BATCH} skipped after 3 attempts; day not marked done', flush=True)
                ok = False
        if ok:
            con.commit()
            done.add(day)
            STATE.write_text(json.dumps({'days': sorted(done)}))
        total += rows
        print(f'[{i}/{len(days)}] {day}: {len(syms)} symbols, +{rows} bars ({time.time() - t0:.0f}s)', flush=True)
    con.close()
    print(f'DONE {total} bars', flush=True)


if __name__ == '__main__':
    main()
