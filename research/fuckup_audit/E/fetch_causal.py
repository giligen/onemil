#!/usr/bin/env python3
"""Stage E Part 2 — fetch the full-session 1-min SIP tape for the CAUSAL universe.

KEYS: research/fuckup_audit/E/fetch_keys.csv = (U1 u U2) minus what research/bf_zero/bars_sip.db
already holds. Part 1 measured 126,499 such keys; PLAN's Stage-E instruction caps the fetch at
120,000, "if larger, take U1 u U2 in full only for 2025-07 onward and report the truncation", so
fetch_keys.csv is already the capped set (bar_date >= 2025-07-01, 95,177 keys). The 31,322 keys of
2025-01..06 are the OVERFLOW: they are written to E/fetch_keys_overflow.csv by this script's
`--overflow` pass and are NOT fetched by default. Nothing is silently dropped either way.

WINDOW: 04:00:00 - 15:59:59 ET (premarket for pm_dollar_vol + the whole RTH session; the study is
flat at 15:55). DST-exact per day via zoneinfo, never a fixed offset.

STORE: research/fuckup_audit/E/bars_causal/day=YYYY-MM-DD/bars.parquet, pyarrow + zstd.
  symbol  string (dictionary-encoded on write)
  t       int16  ET MINUTE OF DAY (570 = 09:30, 955 = 15:55) — the convention of PLAN.md §1 and
                 of every candidate table in this tree. NOT bars_sip.db's ISO-UTC `t`; the loader
                 in E/build_candidates_causal.py converts bars_sip.db's ISO timestamps to the
                 same int16 so the two stores are one tape.
  o,h,l,c float32
  v       float32
Target <= 3 KB per symbol-day, hard cap 6 GB (the script refuses to continue past the cap).

INDEX: research/fuckup_audit/E/bars_causal_index.csv, one row per KEY (symbol, day, n_bars, src).
  src='alpaca'  >= 1 bar returned
  src='none'    Alpaca served nothing for this symbol-day — a SURVIVORSHIP RESIDUAL, recorded,
                never silently dropped (PLAN.md Stage-E stop rule: report the residual).
  src='badsym'  Alpaca rejects the ticker outright (or it fails the ticker regex) — the other
                half of the survivorship residual: a name the broker no longer knows.
  src='error'   the fetch failed after the retry ladder; retried on the next resume.

RESUMABLE: E/fetch_state.json records finished days; a finished day's parquet + index rows are
the real state. Re-running picks up where it stopped.

MARKET DATA ONLY — no trading call is ever made. Writes ONLY under research/fuckup_audit/E/.

RUN (detached, PLAN.md §1):
  setsid nohup bash -c "cd /home/ec2-user/onemil; ulimit -v 1300000; nice -n 10 \
    python3 research/fuckup_audit/E/fetch_causal.py > research/fuckup_audit/E/fetch.log 2>&1; \
    echo EXIT=\\$? >> research/fuckup_audit/E/fetch.log" >/dev/null 2>&1 </dev/null &
"""
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL', 'system')

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import logging
logging.basicConfig(level=logging.ERROR)

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ET = ZoneInfo('America/New_York')
E = f'{ROOT}/research/fuckup_audit/E'
KEYS = f'{E}/fetch_keys.csv'
OVERFLOW_KEYS = f'{E}/fetch_keys_overflow.csv'
STORE = f'{E}/bars_causal'
INDEX = f'{E}/bars_causal_index.csv'
STATE = f'{E}/fetch_state.json'

FETCH_BATCH = 100            # symbols per Alpaca request (the SDK paginates inside one call)
MIN_REQ_INTERVAL = 1.0       # seconds between batches — one batch can cost several pages
DISK_CAP_BYTES = 6 * 1024 ** 3
SYM_OK = re.compile(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$')

SCHEMA = pa.schema([('symbol', pa.string()), ('t', pa.int16()), ('o', pa.float32()),
                    ('h', pa.float32()), ('l', pa.float32()), ('c', pa.float32()),
                    ('v', pa.float32())])

_last_req = [0.0]


def log(m):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {m}', flush=True)


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {'days_done': [], 'bytes': 0}


def save_state(st):
    tmp = STATE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(st, f)
    os.replace(tmp, STATE)


def session_window(day):
    """(start_utc, end_utc) for 04:00:00 and 15:59:59 ET on `day`, and the 00:00 ET anchor."""
    d = datetime.strptime(day, '%Y-%m-%d')
    s = datetime(d.year, d.month, d.day, 4, 0, tzinfo=ET)
    e = datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET) - timedelta(seconds=1)
    return s.astimezone(timezone.utc), e.astimezone(timezone.utc)


def throttle():
    dt = time.time() - _last_req[0]
    if dt < MIN_REQ_INTERVAL:
        time.sleep(MIN_REQ_INTERVAL - dt)
    _last_req[0] = time.time()


def alpaca_client():
    from config import Config
    from data_sources.alpaca_client import AlpacaClient
    cfg = Config()
    return AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)


def fetch_day(client, day, symbols):
    """{symbol: [(minute_of_day_ET, o, h, l, c, v)]}. Raises only if a whole batch is unrecoverable."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed, Adjustment
    start, end = session_window(day)
    out = {}
    hard_fail = []
    bad_sym = [s for s in symbols if not SYM_OK.match(s)]
    for i in range(0, len(symbols), FETCH_BATCH):
        chunk = [s for s in symbols[i:i + FETCH_BATCH] if SYM_OK.match(s)]
        raw = {}
        ok = False
        for k, wait in enumerate((0, 5, 20, 60, 120)):
            if wait:
                time.sleep(wait)
            if not chunk:
                ok = True
                break
            throttle()
            req = StockBarsRequest(symbol_or_symbols=chunk,
                                   timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                                   start=start, end=end,
                                   feed=DataFeed.SIP, adjustment=Adjustment.RAW)
            try:
                raw = client._to_dict(client.data_client.get_stock_bars(req))
                ok = True
                break
            except Exception as e:
                msg = str(e)
                m = re.search(r'invalid symbol: ([A-Z0-9.\-]+)', msg)
                if m and m.group(1) in chunk:        # one bad symbol 400s the whole batch
                    log(f'  {day}: Alpaca rejects {m.group(1)} — dropped')
                    bad_sym.append(m.group(1))
                    chunk = [s for s in chunk if s != m.group(1)]
                    continue
                if '429' in msg or 'rate limit' in msg.lower():
                    log(f'  {day} batch {i // FETCH_BATCH}: 429 — backing off 30s')
                    time.sleep(30)
                    continue
                log(f'  {day} batch {i // FETCH_BATCH} try {k + 1}: {msg[:180]}')
        if not ok:
            hard_fail.extend(chunk)
            continue
        for s in chunk:
            rows = []
            for b in raw.get(s) or []:
                t = b.timestamp.astimezone(ET)
                m = t.hour * 60 + t.minute
                if m < 240 or m > 959:               # belt and braces on 04:00-15:59 ET
                    continue
                rows.append((m, float(b.open), float(b.high), float(b.low),
                             float(b.close), float(b.volume)))
            rows.sort()
            out[s] = rows
    return out, hard_fail, bad_sym


def write_day(day, got, keys, bad=()):
    """One parquet file per day. Returns (n_rows, bytes, index_rows)."""
    syms, ts, o, h, l, c, v = [], [], [], [], [], [], []
    idx = []
    for s in keys:
        rows = got.get(s)
        if rows is None:
            idx.append((s, day, 0, 'badsym' if s in bad else 'error'))
            continue
        if not rows:
            idx.append((s, day, 0, 'none'))
            continue
        idx.append((s, day, len(rows), 'alpaca'))
        for r in rows:
            syms.append(s)
            ts.append(r[0])
            o.append(r[1]); h.append(r[2]); l.append(r[3]); c.append(r[4]); v.append(r[5])
    d = f'{STORE}/day={day}'
    os.makedirs(d, exist_ok=True)
    path = f'{d}/bars.parquet'
    tbl = pa.Table.from_arrays(
        [pa.array(syms, type=pa.string()), pa.array(np.asarray(ts, dtype='int16')),
         pa.array(np.asarray(o, dtype='float32')), pa.array(np.asarray(h, dtype='float32')),
         pa.array(np.asarray(l, dtype='float32')), pa.array(np.asarray(c, dtype='float32')),
         pa.array(np.asarray(v, dtype='float32'))], schema=SCHEMA)
    pq.write_table(tbl, path, compression='zstd', compression_level=5,
                   use_dictionary=['symbol'], version='2.6')
    return len(syms), os.path.getsize(path), idx


def load_keys(path):
    by_day = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            by_day.setdefault(row['bar_date'], set()).add(row['symbol'])
    return {d: sorted(v) for d, v in by_day.items()}


def main(keys_path=KEYS):
    t0 = time.time()
    by_day = load_keys(keys_path)
    total_keys = sum(len(v) for v in by_day.values())
    days = sorted(by_day)
    log(f'keys {total_keys:,} over {len(days)} days ({days[0]}..{days[-1]}) from {keys_path}')

    st = load_state()
    done = set(st['days_done'])
    os.makedirs(STORE, exist_ok=True)
    new_index = not os.path.exists(INDEX)
    ix = open(INDEX, 'a', newline='')
    iw = csv.writer(ix)
    if new_index:
        iw.writerow(['symbol', 'day', 'n_bars', 'src'])

    client = alpaca_client()
    seen = n_alpaca = n_none = n_err = n_bad = 0
    n_rows = 0
    nbytes = int(st.get('bytes', 0))
    last_log = 0
    eta_printed = False
    for i, day in enumerate(days):
        syms = by_day[day]
        seen += len(syms)
        if day in done:
            continue
        got, hard, bad = fetch_day(client, day, syms)
        for s in hard:
            got.pop(s, None)
        rows, size, idx = write_day(day, got, syms, set(bad))
        for r in idx:
            iw.writerow(r)
            if r[3] == 'alpaca':
                n_alpaca += 1
            elif r[3] == 'none':
                n_none += 1
            elif r[3] == 'badsym':
                n_bad += 1
            else:
                n_err += 1
        ix.flush()
        n_rows += rows
        nbytes += size
        st['days_done'].append(day)
        st['bytes'] = nbytes
        if i % 5 == 0:
            save_state(st)
        if nbytes > DISK_CAP_BYTES:
            log(f'STOP: store is {nbytes / 1024 ** 3:.2f} GB, past the 6 GB cap')
            break
        if not eta_printed and seen >= 500:
            el = time.time() - t0
            log(f'ETA from the first {seen} keys: {(total_keys - seen) / (seen / el) / 60:.0f} min '
                f'total; {nbytes / max(seen, 1) / 1024:.2f} KB per symbol-day so far')
            eta_printed = True
        if seen - last_log >= 1000 or i == len(days) - 1:
            last_log = seen
            el = time.time() - t0
            rate = seen / el if el else 0
            log(f'day {i + 1}/{len(days)} {day} | keys {seen:,}/{total_keys:,} '
                f'| alpaca {n_alpaca:,} none {n_none:,} bad {n_bad:,} err {n_err:,} '
                f'| {nbytes / 1024 ** 2:.0f} MB, {nbytes / max(seen, 1) / 1024:.2f} KB/key '
                f'| {el / 60:.1f} min, ETA {(total_keys - seen) / rate / 60 if rate else 0:.0f} min')
    save_state(st)
    ix.close()
    log(f'DONE keys {seen:,} | alpaca {n_alpaca:,} none {n_none:,} bad {n_bad:,} err {n_err:,} '
        f'| rows {n_rows:,} | {nbytes / 1024 ** 2:.0f} MB '
        f'| {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main(OVERFLOW_KEYS if '--overflow' in sys.argv else KEYS)
