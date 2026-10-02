#!/usr/bin/env python3
"""Cell 1701 -- Databento 1-min OHLCV bars for earnings-session symbol-days.

Fetches research/orb_earn/1701_fetch_list.csv (19,199 symbol,day rows,
2024-07-01..2026-09-04) from Databento EQUS.MINI ohlcv-1m (consolidated
US equities, point-in-time raw_symbol resolution -- covers delisted
names Alpaca may not serve), one get_range per day with that day's
symbols (~540 requests). Writes one parquet shard per day to
bars_db_1701/ (atomic tmp+rename, resumable -- skips days whose shard
already exists), merges all shards into bars_db_1701.parquet, and logs
a LOST list for symbol-days that returned zero bars.

HARD MONEY RULE: cost is priced via metadata.get_cost for the EXACT
request set before any paid call; the pull aborts WITHOUT spending if
the summed estimate exceeds CAP_USD. metadata.get_cost itself is a free
pricing call. Never touches cache.db or bars_sip.db.

Usage: python3 1701_databento_pull.py [--cost-only]
"""
import glob
import os
import sys
import time

import pandas as pd
import databento as db
from dotenv import load_dotenv

sys.stdout.reconfigure(line_buffering=True)

ROOT = '/home/ec2-user/onemil'
D = f'{ROOT}/research/orb_earn'
FETCH_LIST = f'{D}/1701_fetch_list.csv'
SHARD_DIR = f'{D}/bars_db_1701'
MERGED = f'{D}/bars_db_1701.parquet'
LOG = f'{D}/1701_databento.log'
COST_FILE = f'{D}/1701_databento_cost.txt'
LOST_FILE = f'{D}/1701_databento_lost.csv'
CAP_USD = 10.00  # owner-approved hard cap for this pull -- see PREREG_1701.md
DATASET = 'EQUS.MINI'
SCHEMA = 'ohlcv-1m'
RTH_START_MIN = 9 * 60 + 30   # 09:30 ET
RTH_END_MIN = 16 * 60         # 16:00 ET
MIN_BARS_FOR_COVERAGE = 300


def log(msg: str) -> None:
    """Timestamped line to stdout AND the log file (append)."""
    line = f"{pd.Timestamp.now(tz='UTC').isoformat()} {msg}"
    print(line)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def next_day(day: str) -> str:
    """Calendar day after `day` (YYYY-MM-DD) as a string."""
    return str((pd.Timestamp(day) + pd.Timedelta(days=1)).date())


def get_cost_with_retry(client, day: str, symbols: list, retries: int = 4) -> float:
    """metadata.get_cost for one day's symbol set, with retry. Free call."""
    nxt = next_day(day)
    for att in range(retries):
        try:
            return float(client.metadata.get_cost(
                dataset=DATASET, schema=SCHEMA, symbols=symbols,
                stype_in='raw_symbol', start=day, end=nxt))
        except Exception as e:
            log(f"  WARN get_cost {day} attempt {att}: {str(e)[:200]}")
            time.sleep(2 + 3 * att)
    raise RuntimeError(f"get_cost permanently failed for {day}")


def fetch_day(client, day: str, symbols: list, retries: int = 4) -> pd.DataFrame:
    """One EQUS.MINI ohlcv-1m get_range for `symbols` on `day`. Paid call."""
    nxt = next_day(day)
    for att in range(retries):
        try:
            st = client.timeseries.get_range(
                dataset=DATASET, schema=SCHEMA, symbols=symbols,
                stype_in='raw_symbol', start=day, end=nxt)
            return st.to_df().reset_index()
        except Exception as e:
            log(f"  WARN fetch {day} attempt {att}: {str(e)[:200]}")
            time.sleep(3 + 5 * att)
    raise RuntimeError(f"fetch failed for {day} after {retries} attempts")


def main() -> None:
    cost_only = '--cost-only' in sys.argv
    os.makedirs(SHARD_DIR, exist_ok=True)
    load_dotenv(f'{ROOT}/.env')
    key = os.environ.get('DATABENTO_API_KEY')
    if not key:
        log("ERROR: DATABENTO_API_KEY missing from environment -- abort")
        sys.exit(1)

    fl = pd.read_csv(FETCH_LIST, dtype={'symbol': str, 'day': str})
    by_day = fl.groupby('day')['symbol'].apply(lambda s: sorted(set(s))).to_dict()
    days = sorted(by_day)
    log(f"loaded {len(fl):,} symbol-days across {len(days)} days from {FETCH_LIST}")

    client = db.Historical(key)

    # ---- Pre-flight pricing: exact request set, one get_cost per day (free) ----
    log(f"pricing {len(days)} get_range requests dataset={DATASET} schema={SCHEMA} ...")
    total_cost = 0.0
    for idx, day in enumerate(days):
        total_cost += get_cost_with_retry(client, day, by_day[day])
        if (idx + 1) % 50 == 0 or idx == len(days) - 1:
            log(f"  pricing [{idx + 1}/{len(days)}] running_total=${total_cost:.4f}")

    cost_line = (f"COST_EST=${total_cost:.4f} dataset={DATASET} schema={SCHEMA} "
                 f"requests={len(days)} symbol_days={len(fl):,} cap=${CAP_USD:.2f}")
    log(cost_line)
    with open(COST_FILE, 'w') as f:
        f.write(cost_line + '\n')

    if total_cost > CAP_USD:
        log(f"ABORT: estimated cost ${total_cost:.4f} exceeds cap ${CAP_USD:.2f} -- "
            f"NO paid get_range request made")
        sys.exit(2)
    if cost_only:
        log("cost-only run finished (no data pulled)")
        return

    # ---- Pull, resumable per-day shard, atomic write ----
    lost_rows = []
    done_days = {os.path.basename(p)[:-len('.parquet')]
                 for p in glob.glob(f'{SHARD_DIR}/*.parquet')}
    log(f"{len(done_days)} day-shards already present -- resuming, "
        f"{len(days) - len(done_days)} to go")

    for i, day in enumerate(days):
        shard_path = f'{SHARD_DIR}/{day}.parquet'
        if day in done_days:
            continue
        syms = by_day[day]
        df = fetch_day(client, day, syms)
        if df.empty:
            for s in syms:
                lost_rows.append({'symbol': s, 'day': day, 'reason': 'no_bars_returned'})
            log(f"  [{i + 1}/{len(days)}] {day}: 0 bars for {len(syms)} symbols -- all LOST")
            continue
        ts = pd.to_datetime(df['ts_event'], utc=True)
        out = pd.DataFrame({
            'symbol': df['symbol'].astype(str),
            'day': day,
            'ts_utc': ts,
            'open': df['open'], 'high': df['high'], 'low': df['low'],
            'close': df['close'], 'volume': df['volume'],
            'source': f'databento:{DATASET}',
        })
        got_syms = set(out['symbol'].unique())
        for s in syms:
            if s not in got_syms:
                lost_rows.append({'symbol': s, 'day': day, 'reason': 'symbol_absent_from_response'})
        tmp_path = shard_path + '.tmp'
        out.to_parquet(tmp_path, index=False)
        os.rename(tmp_path, shard_path)
        if (i + 1) % 20 == 0 or i == len(days) - 1:
            log(f"  [{i + 1}/{len(days)}] {day}: {len(out):,} bars, "
                f"{len(got_syms)}/{len(syms)} symbols -- shard written")

    if lost_rows:
        lost_df = pd.DataFrame(lost_rows)
        if os.path.exists(LOST_FILE):
            prev = pd.read_csv(LOST_FILE)
            lost_df = pd.concat([prev, lost_df]).drop_duplicates(['symbol', 'day'])
        lost_df.to_csv(LOST_FILE, index=False)
    log(f"fetch pass done: {len(lost_rows)} new LOST rows this pass")

    # ---- Merge all shards into the single parquet ----
    shard_files = sorted(glob.glob(f'{SHARD_DIR}/*.parquet'))
    log(f"merging {len(shard_files)} day-shards into {MERGED} ...")
    frames = [pd.read_parquet(p) for p in shard_files]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=['symbol', 'day', 'ts_utc', 'open', 'high', 'low', 'close', 'volume', 'source'])
    tmp_merged = MERGED + '.tmp'
    merged.to_parquet(tmp_merged, index=False)
    os.rename(tmp_merged, MERGED)

    # ---- Coverage: regular session (09:30-16:00 ET) bars >= 300 per symbol-day ----
    fl_keys = set(zip(fl['symbol'], fl['day']))
    covered = 0
    if not merged.empty:
        et = pd.to_datetime(merged['ts_utc'], utc=True).dt.tz_convert('America/New_York')
        minute = et.dt.hour * 60 + et.dt.minute
        reg = merged[(minute >= RTH_START_MIN) & (minute < RTH_END_MIN)]
        counts = reg.groupby(['symbol', 'day']).size()
        covered = sum(1 for k in fl_keys if counts.get(k, 0) >= MIN_BARS_FOR_COVERAGE)
    pct = 100.0 * covered / len(fl_keys) if fl_keys else 0.0
    log(f"COVERAGE: {covered}/{len(fl_keys)} symbol-days with >= {MIN_BARS_FOR_COVERAGE} "
        f"regular-session bars ({pct:.1f}%) COST_EST=${total_cost:.4f}")


if __name__ == '__main__':
    main()
