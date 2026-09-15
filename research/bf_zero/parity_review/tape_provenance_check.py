#!/usr/bin/env python3
"""Tape provenance assertion (parity review, 2026-09-15): every 1-min bar store the bf_zero loader reads must be the
consolidated SIP tape. For each store, 50 random symbol-days it holds (overlapping Alpaca's coverage) are compared bar by
bar with Alpaca REST SIP (RTH 09:30-15:59): a bar is EXACT when the minute exists on both sides and o/h/l/c match to 1c
and volume to 0.5%. PASS = >= 99% exact bars over the sample. Read-only. Usage: tape_provenance_check.py [store ...]
Stores: cache.db (intraday_bars_1min), research/bf_zero/bars_sip.db, research/ignition_capcheck/topup.db,
data/research/databento/pit_bars_1min.db, research/bf_zero/bars.db. Exit 1 if any checked store fails."""
import os, sys, sqlite3, random
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
import logging; logging.basicConfig(level=logging.ERROR)
import pandas as pd, numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from config import Config
from data_sources.alpaca_client import AlpacaClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment
ET = ZoneInfo('America/New_York'); N = int(os.environ.get('TPC_N', '50')); SEED = int(os.environ.get('TPC_SEED', '11'))
STORES = {'cache.db': ('data/cache.db', 'intraday_bars_1min'), 'bars_sip.db': ('research/bf_zero/bars_sip.db', 'bars'),
          'topup.db': ('research/ignition_capcheck/topup.db', 'bars'), 'pit_bars_1min.db': ('data/research/databento/pit_bars_1min.db', 'bars'),
          'bars.db': ('research/bf_zero/bars.db', 'bars')}
cfg = Config(); client = AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)
DAYS = [d.strftime('%Y-%m-%d') for d in pd.bdate_range('2025-01-02', '2026-09-11')]


def store_bars(con, table, sym, day):
    if table == 'intraday_bars_1min':
        g = pd.read_sql("select timestamp as t, open as o, high as h, low as l, close as c, volume as v from intraday_bars_1min where symbol=? and bar_date=?", con, params=[sym, day])
    else:
        g = pd.read_sql("select t, o, h, l, c, v from bars where symbol=? and day=?", con, params=[sym, day])
    if not len(g): return g
    ts = pd.to_datetime(g.t, utc=True).dt.tz_convert('America/New_York'); g['m'] = (ts.dt.hour * 60 + ts.dt.minute).values
    return g[(g.m >= 570) & (g.m < 960)].drop_duplicates('m').set_index('m').sort_index()


def alpaca_bars(sym, day):
    d = datetime.strptime(day, '%Y-%m-%d')
    req = StockBarsRequest(symbol_or_symbols=[sym], timeframe=TimeFrame(1, TimeFrameUnit.Minute), feed=DataFeed.SIP, adjustment=Adjustment.RAW,
                           start=datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET).astimezone(timezone.utc), end=datetime(d.year, d.month, d.day, 15, 59, 59, tzinfo=ET).astimezone(timezone.utc))
    try:
        raw = client._to_dict(client.data_client.get_stock_bars(req)).get(sym) or []
    except Exception as e:                       # e.g. Databento symbology Alpaca rejects ('CODI-A'): not an overlapping key
        print(f'  {sym} {day}: Alpaca rejected ({str(e)[:60]}) — skipped', flush=True); raw = []
    rows = [dict(m=b.timestamp.astimezone(ET).hour * 60 + b.timestamp.astimezone(ET).minute, o=float(b.open), h=float(b.high), l=float(b.low), c=float(b.close), v=float(b.volume)) for b in raw]
    g = pd.DataFrame(rows)
    return g[(g.m >= 570) & (g.m < 960)].drop_duplicates('m').set_index('m').sort_index() if len(g) else g


def sample_keys(con, table, n):
    """n random (symbol, day) keys the store holds, drawn day-first (indexed) to avoid a full scan."""
    random.seed(SEED); keys = []; tries = 0
    if table == 'intraday_bars_1min':
        # cache.db is indexed on (symbol, bar_date) only: sample symbol-first from the study universe, then a held day
        usyms = pd.read_csv(f'{ROOT}/research/bf_zero/universe.csv', usecols=['symbol'], dtype=str, keep_default_na=False).symbol.unique().tolist()
        while len(keys) < n and tries < 400:
            tries += 1; sym = random.choice(usyms)
            days = [r[0] for r in con.execute("select distinct bar_date from intraday_bars_1min where symbol=? and bar_date>='2025-01-02'", (sym,)).fetchall()]
            if days: keys.append((sym, random.choice(days)))
        return keys
    while len(keys) < n and tries < 400:
        tries += 1; day = random.choice(DAYS)
        syms = [r[0] for r in con.execute("select distinct symbol from bars where day=?", (day,)).fetchall()]
        if syms: keys.append((random.choice(syms), day))
    return keys


def check(name):
    path, table = STORES[name]
    if not os.path.exists(path): print(f'{name}: MISSING ({path})'); return None
    con = sqlite3.connect(f'file:{ROOT}/{path}?mode=ro', uri=True, timeout=60)
    keys = sample_keys(con, table, N); exact = total = 0; keys_used = 0; worst = []
    for sym, day in keys:
        s = store_bars(con, table, sym, day); a = alpaca_bars(sym, day)
        if not len(a): continue                    # Alpaca does not serve it (delisted) — not an overlapping key
        keys_used += 1
        j = s.join(a, how='outer', lsuffix='_s', rsuffix='_a')
        ok = ((j.o_s - j.o_a).abs() <= 0.011) & ((j.h_s - j.h_a).abs() <= 0.011) & ((j.l_s - j.l_a).abs() <= 0.011) & ((j.c_s - j.c_a).abs() <= 0.011) & ((j.v_s / j.v_a.replace(0, np.nan) - 1).abs() <= 0.005)
        ok = ok.fillna(False); exact += int(ok.sum()); total += len(j)
        vr = float(s.v.sum() / a.v.sum()) if a.v.sum() else float('nan'); worst.append((round(vr, 3), sym, day))
    pct = exact / total * 100 if total else float('nan'); worst.sort()
    print(f"{name:18s} keys {keys_used}/{len(keys)} served by Alpaca | exact bars {exact}/{total} = {pct:.1f}% | vol-ratio median {np.nanmedian([w[0] for w in worst]) if worst else float('nan'):.3f} | worst {worst[:3]} | {'PASS' if pct >= 99 else 'FAIL'}", flush=True)
    return pct >= 99


names = sys.argv[1:] or list(STORES)
res = {n: check(n) for n in names}
sys.exit(0 if all(v for v in res.values() if v is not None) else 1)
