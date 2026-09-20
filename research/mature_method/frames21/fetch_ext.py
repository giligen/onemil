#!/usr/bin/env python3
"""frames21 — fetch 1-min SIP bars (Alpaca REST, feed SIP, raw) for the 2024H2 extension-window
candidate universe (research/mature_method/frames21/raw/candidates_ext.csv: day high >= open x 1.05,
close >= $1, adv20 >= 100K on research/multiday/data/prices_by_year -- the SAME rule bf_zero's
universe.csv / bars_sip.db causal superset used, per build_candidates.py's own docstring).

Pattern copied from research/bf_zero/refetch_thin_tape.py::fetch_day/do_fetch (unchanged fetch
logic: 04:00-20:00 ET window, feed=SIP, adjustment=RAW, 200-symbol batches, retry w/ backoff,
drop-one-bad-symbol-and-retry). Writes research/mature_method/frames21/raw/bars_sip_ext.db with the
SAME schema as bars_sip.db: bars(symbol, day, t, o, h, l, c, v), fetch_log(symbol, day, src, n_bars,
fetched_at). Alpaca pulls are NOT priced against the Databento $80 cap (they are not Databento).
Resumable via fetch_log.
"""
import os, re, sqlite3, sys, time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import pandas as pd  # noqa: E402

ET = ZoneInfo('America/New_York')
D = f'{ROOT}/research/mature_method/frames21'
STORE = f'{D}/raw/bars_sip_ext.db'
CAND = f'{D}/raw/candidates_ext.csv'
BATCH = 200
SYM_OK = re.compile(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$')


def log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def open_store():
    con = sqlite3.connect(STORE, timeout=120)
    con.execute("create table if not exists bars (symbol TEXT, day TEXT, t TEXT, o REAL, h REAL, l REAL, c REAL, v REAL, PRIMARY KEY (symbol, day, t))")
    con.execute("create index if not exists idx_bars_day on bars(day)")
    con.execute("create table if not exists fetch_log (symbol TEXT, day TEXT, src TEXT, n_bars INTEGER, fetched_at TEXT, PRIMARY KEY (symbol, day))")
    con.commit()
    return con


def alpaca_client():
    from config import Config
    from data_sources.alpaca_client import AlpacaClient
    cfg = Config()
    return AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)


def fetch_day(client, day, symbols):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed, Adjustment
    d = datetime.strptime(day, '%Y-%m-%d')
    start = datetime(d.year, d.month, d.day, 4, 0, tzinfo=ET).astimezone(timezone.utc)
    end = datetime(d.year, d.month, d.day, 20, 0, tzinfo=ET).astimezone(timezone.utc)
    out = {}
    for i in range(0, len(symbols), BATCH):
        chunk = symbols[i:i + BATCH]
        chunk = [s for s in chunk if SYM_OK.match(s)]
        raw = {}
        for k, wait in enumerate((0, 5, 20, 60)):
            if wait:
                time.sleep(wait)
            if not chunk:
                break
            req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                                    start=start, end=end, feed=DataFeed.SIP, adjustment=Adjustment.RAW)
            try:
                raw = client._to_dict(client.data_client.get_stock_bars(req))
                break
            except Exception as e:
                m = re.search(r'invalid symbol: ([A-Z0-9.\-]+)', str(e))
                if m and m.group(1) in chunk:
                    log(f'  {day}: Alpaca rejects {m.group(1)} - dropped from the batch')
                    chunk = [s for s in chunk if s != m.group(1)]
                    continue
                log(f'  {day} chunk {i // BATCH} try {k + 1} failed: {e}')
                if k == 3:
                    raise
        for s in chunk:
            out[s] = [(b.timestamp.astimezone(timezone.utc).isoformat(), float(b.open), float(b.high),
                       float(b.low), float(b.close), float(b.volume)) for b in (raw.get(s) or [])]
    return out


def main():
    K = pd.read_csv(CAND, dtype={'symbol': str}, keep_default_na=False)
    con = open_store()
    done = set(con.execute("select symbol, day from fetch_log").fetchall())
    todo = K[[(s, d) not in done for s, d in zip(K.symbol, K.date)]]
    ndays = todo.date.nunique()
    log(f'candidates {len(K)} | done {len(done)} | todo {len(todo)} over {ndays} days')
    client = alpaca_client()
    t0 = time.time()
    n_rows = n_keys = n_empty = 0
    for i, (day, g) in enumerate(todo.groupby('date')):
        syms = sorted(g.symbol.unique())
        bars = fetch_day(client, day, syms)
        rows = [(s, day, *b) for s in syms for b in bars.get(s, [])]
        now = datetime.now(timezone.utc).isoformat()
        con.executemany("insert or replace into bars values (?,?,?,?,?,?,?,?)", rows)
        con.executemany("insert or replace into fetch_log values (?,?,?,?,?)",
                         [(s, day, 'alpaca_sip', len(bars.get(s, [])), now) for s in syms])
        con.commit()
        n_rows += len(rows); n_keys += len(syms); n_empty += sum(1 for s in syms if not bars.get(s))
        if i % 5 == 0 or i == ndays - 1:
            el = time.time() - t0
            log(f'{i + 1}/{ndays} {day} syms {len(syms)} rows +{len(rows)} | keys {n_keys} empty {n_empty} '
                f'rows {n_rows:,} | {el / 60:.1f} min')
    log(f'FETCH DONE keys {n_keys} empty {n_empty} rows {n_rows:,}')


if __name__ == '__main__':
    sys.exit(main())
