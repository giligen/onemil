#!/usr/bin/env python3
"""bf_zero — replace the thin-tape 1-min bars with the consolidated SIP tape (parity review, 2026-09-15).

Finding: the study's side stores research/bf_zero/bars.db and data/research/databento/pit_bars_1min.db were fetched from
Databento EQUS.MINI (one publisher, 1-10% of SIP volume; highs/opens differ) — the live engine streams the SIP. Every
causal-superset symbol-day (day high >= open x 1.05, all prices) that build_candidates.load_bars served from a thin
store is re-fetched from Alpaca REST (1-min, feed SIP, adjustment raw, 04:00-20:00 ET) into a NEW store
research/bf_zero/bars_sip.db (same schema as bars.db: bars(symbol, day, t, o, h, l, c, v) + idx_bars_day, plus
fetch_log(symbol, day, src, n_bars, fetched_at)). The old stores are never written. The loader reads the new store when
BFZ_SIP_STORE=research/bf_zero/bars_sip.db is set (build_candidates.py).

Modes (one at a time):
  --fetch [--src topup.db,pit_bars_1min.db,bars.db,none]   (topup.db FAILED provenance: mixed tape, 65% exact) [--days-from D] [--days-to D]   resumable; fetch_log rows are skipped
  --copy-topup            copy the superset keys served by topup.db (verified SIP) from topup.db into the new store
  --verify N              N random fetched keys: load_bars(BFZ_SIP_STORE) cum-volume@10:30 vs a fresh REST call, <= 2%
  --cost DATASET          quote (never buy) the Databento cost of the keys Alpaca did not serve (n_bars=0), ohlcv-1m
Keys: research/bf_zero/parity_review/superset_provenance.csv (symbol, bar_date, open, src) from superset_provenance.py.
"""
import argparse, os, re, sqlite3, sys, time, random
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, f'{ROOT}/research/bf_zero')
import logging; logging.basicConfig(level=logging.ERROR)
import numpy as np, pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
ET = ZoneInfo('America/New_York')
KEYS = f'{ROOT}/research/bf_zero/parity_review/superset_provenance.csv'
STORE = f'{ROOT}/research/bf_zero/bars_sip.db'
TOPUP = f'{ROOT}/research/ignition_capcheck/topup.db'
BATCH = 200
SYM_OK = re.compile(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$')   # Alpaca symbology; anything else is logged as unserved


def log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def open_store():
    con = sqlite3.connect(STORE, timeout=120)
    con.execute("create table if not exists bars (symbol TEXT, day TEXT, t TEXT, o REAL, h REAL, l REAL, c REAL, v REAL, PRIMARY KEY (symbol, day, t))")
    con.execute("create index if not exists idx_bars_day on bars(day)")
    con.execute("create table if not exists fetch_log (symbol TEXT, day TEXT, src TEXT, n_bars INTEGER, fetched_at TEXT, PRIMARY KEY (symbol, day))")
    con.commit(); return con


def alpaca_client():
    from config import Config
    from data_sources.alpaca_client import AlpacaClient
    cfg = Config(); return AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)


def fetch_day(client, day, symbols):
    """{symbol: [(t, o, h, l, c, v), ...]} for one day, 04:00-20:00 ET, SIP, raw. Retries with backoff."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed, Adjustment
    d = datetime.strptime(day, '%Y-%m-%d')
    start = datetime(d.year, d.month, d.day, 4, 0, tzinfo=ET).astimezone(timezone.utc); end = datetime(d.year, d.month, d.day, 20, 0, tzinfo=ET).astimezone(timezone.utc)
    out = {}
    for i in range(0, len(symbols), BATCH):
        chunk = symbols[i:i + BATCH]
        chunk = [s for s in chunk if SYM_OK.match(s)]           # Databento symbology Alpaca rejects (e.g. 'CODI-A') -> unserved
        raw = {}
        for k, wait in enumerate((0, 5, 20, 60)):
            if wait: time.sleep(wait)
            if not chunk: break
            req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame(1, TimeFrameUnit.Minute), start=start, end=end, feed=DataFeed.SIP, adjustment=Adjustment.RAW)
            try:
                raw = client._to_dict(client.data_client.get_stock_bars(req)); break
            except Exception as e:
                m = re.search(r'invalid symbol: ([A-Z0-9.\-]+)', str(e))
                if m and m.group(1) in chunk:                        # one bad symbol 400s the whole batch: drop it, retry now
                    log(f'  {day}: Alpaca rejects {m.group(1)} — dropped from the batch'); chunk = [s for s in chunk if s != m.group(1)]; continue
                log(f'  {day} chunk {i // BATCH} try {k + 1} failed: {e}')
                if k == 3: raise
        for s in chunk:
            out[s] = [(b.timestamp.astimezone(timezone.utc).isoformat(), float(b.open), float(b.high), float(b.low), float(b.close), float(b.volume)) for b in (raw.get(s) or [])]
    return out


def do_fetch(srcs, day_from, day_to, keys_path=None, min_open=None):
    K = pd.read_csv(keys_path or KEYS, dtype={'symbol': str}, keep_default_na=False)
    K = K[K.src.isin(srcs)]
    if min_open is not None: K = K[pd.to_numeric(K.open, errors='coerce') >= min_open]
    if day_from: K = K[K.bar_date >= day_from]
    if day_to: K = K[K.bar_date <= day_to]
    con = open_store(); done = set(con.execute("select symbol, day from fetch_log").fetchall())
    todo = K[[(s, d) not in done for s, d in zip(K.symbol, K.bar_date)]]
    log(f'fetch: {len(K)} keys in {sorted(srcs)} | done {len(done)} | todo {len(todo)} over {todo.bar_date.nunique()} days')
    client = alpaca_client(); t0 = time.time(); n_rows = n_keys = n_empty = 0
    for i, (day, g) in enumerate(todo.groupby('bar_date')):
        syms = sorted(g.symbol); srcmap = dict(zip(g.symbol, g.src))
        bars = fetch_day(client, day, syms)
        rows = [(s, day, *b) for s in syms for b in bars.get(s, [])]
        now = datetime.now(timezone.utc).isoformat()
        con.executemany("insert or replace into bars values (?,?,?,?,?,?,?,?)", rows)
        con.executemany("insert or replace into fetch_log values (?,?,?,?,?)", [(s, day, srcmap[s], len(bars.get(s, [])), now) for s in syms])
        con.commit()
        n_rows += len(rows); n_keys += len(syms); n_empty += sum(1 for s in syms if not bars.get(s))
        if i % 10 == 0 or i == todo.bar_date.nunique() - 1:
            el = time.time() - t0
            log(f'{i + 1}/{todo.bar_date.nunique()} {day} syms {len(syms)} rows +{len(rows)} | keys {n_keys} empty {n_empty} rows {n_rows:,} | {el / 60:.1f} min')
    log(f'FETCH DONE keys {n_keys} empty {n_empty} rows {n_rows:,}')


def do_copy_topup():
    K = pd.read_csv(KEYS, dtype={'symbol': str}, keep_default_na=False); K = K[K.src == 'topup.db']
    con = open_store(); src = sqlite3.connect(f'file:{TOPUP}?mode=ro', uri=True); done = set(con.execute("select symbol, day from fetch_log").fetchall()); n = 0
    for day, g in K.groupby('bar_date'):
        syms = [s for s in g.symbol if (s, day) not in done]
        if not syms: continue
        rows = src.execute(f"select symbol, day, t, o, h, l, c, v from bars where day=? and symbol in ({','.join('?' * len(syms))})", [day] + syms).fetchall()
        cnt = pd.Series([r[0] for r in rows]).value_counts().to_dict() if rows else {}
        con.executemany("insert or replace into bars values (?,?,?,?,?,?,?,?)", rows)
        con.executemany("insert or replace into fetch_log values (?,?,?,?,?)", [(s, day, 'topup.db(copy)', int(cnt.get(s, 0)), datetime.now(timezone.utc).isoformat()) for s in syms])
        con.commit(); n += len(syms)
    log(f'copied {n} topup.db keys into the SIP store')


def do_verify(n):
    os.environ['BFZ_SIP_STORE'] = STORE
    import build_candidates as B
    con = sqlite3.connect(f'file:{STORE}?mode=ro', uri=True)
    keys = con.execute("select symbol, day from fetch_log where n_bars > 0 and src != 'topup.db(copy)'").fetchall(); random.seed(3); random.shuffle(keys)
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed, Adjustment
    client = alpaca_client(); bad = 0
    for sym, day in keys[:n]:
        gg = B.load_bars(day, [sym]).get(sym)
        sv = float(gg.v[(gg.m >= 570) & (gg.m <= 630)].sum()) if gg is not None else float('nan')
        d = datetime.strptime(day, '%Y-%m-%d')
        req = StockBarsRequest(symbol_or_symbols=[sym], timeframe=TimeFrame(1, TimeFrameUnit.Minute), feed=DataFeed.SIP, adjustment=Adjustment.RAW,
                               start=datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET).astimezone(timezone.utc), end=datetime(d.year, d.month, d.day, 10, 30, 59, tzinfo=ET).astimezone(timezone.utc))
        av = float(sum(b.volume for b in (client._to_dict(client.data_client.get_stock_bars(req)).get(sym) or [])))
        r = sv / av if av else float('nan'); ok = abs(r - 1) <= 0.02; bad += (not ok)
        print(f'{sym:6s} {day} load_bars cumV@10:30 {sv:12.0f} REST {av:12.0f} ratio {r:.4f} {"ok" if ok else "MISMATCH"}', flush=True)
    print(f'VERIFY {n - bad}/{n} within 2%', flush=True); return bad == 0


def do_cost(dataset):
    import databento as db
    from config import Config; Config()                       # loads .env (DATABENTO_API_KEY lives there, not in the shell)
    con = sqlite3.connect(f'file:{STORE}?mode=ro', uri=True)
    rem = pd.DataFrame(con.execute("select symbol, day from fetch_log where n_bars = 0").fetchall(), columns=['symbol', 'day'])
    log(f'remainder (Alpaca served no bars): {len(rem)} keys, {rem.symbol.nunique()} symbols, {rem.day.nunique()} days')
    if not len(rem): return
    c = db.Historical(os.environ['DATABENTO_API_KEY']); total = 0.0; n = 0
    for day, g in rem.groupby('day'):
        d = datetime.strptime(day, '%Y-%m-%d').date()
        try:
            cost = c.metadata.get_cost(dataset=dataset, symbols=sorted(g.symbol), schema='ohlcv-1m', start=str(d), end=str(d + pd.Timedelta(days=1).to_pytimedelta()))
        except Exception as e:
            log(f'  {day} get_cost failed: {e}'); continue
        total += float(cost); n += len(g)
    log(f'COST QUOTE {dataset} ohlcv-1m for {n} symbol-days: ${total:.2f} (nothing purchased)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch', action='store_true'); ap.add_argument('--src', default='topup.db,pit_bars_1min.db,bars.db,none')
    ap.add_argument('--days-from'); ap.add_argument('--days-to')
    ap.add_argument('--keys', help='alternative key CSV (symbol, bar_date, open, src)'); ap.add_argument('--min-open', type=float)
    ap.add_argument('--copy-topup', action='store_true'); ap.add_argument('--verify', type=int); ap.add_argument('--cost')
    a = ap.parse_args()
    if a.fetch: do_fetch(set(a.src.split(',')), a.days_from, a.days_to, a.keys, a.min_open)
    elif a.copy_topup: do_copy_topup()
    elif a.verify: sys.exit(0 if do_verify(a.verify) else 1)
    elif a.cost: do_cost(a.cost)
    else: ap.print_help()
