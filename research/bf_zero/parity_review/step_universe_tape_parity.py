"""Bar-tape parity: the study's 1-min bars (Databento bars.db for days cache.db lacked; cache.db = Alpaca for the rest)
vs Alpaca SIP historical 1-min bars (what live streams/backfills). Compares o[0], first-bar minute, HOD by 10:30,
cum volume by 10:30, bar count. Read-only REST. argv: SYMBOL:DAY ..."""
import os, sys, sqlite3
os.chdir('/home/ec2-user/onemil'); sys.path.insert(0, '/home/ec2-user/onemil')
import logging; logging.basicConfig(level=logging.ERROR)
import pandas as pd, numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from config import Config
from data_sources.alpaca_client import AlpacaClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
ET = ZoneInfo('America/New_York'); ROOT = '/home/ec2-user/onemil'
cfg = Config(); client = AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)
pairs = [a.split(':') for a in sys.argv[1:]]
src = {'bars.db': sqlite3.connect(f'file:{ROOT}/research/bf_zero/bars.db?mode=ro', uri=True), 'cache.db': sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True)}


def study_bars(sym, day, db=None):
    if db:
        con = sqlite3.connect(f'file:{ROOT}/{db}?mode=ro', uri=True)
        g = pd.read_sql("select symbol, t, o, h, l, c, v from bars where day=? and symbol=?", con, params=[day, sym]); where = db.split('/')[-1]
    else:
        g = pd.read_sql("select symbol, timestamp as t, open as o, high as h, low as l, close as c, volume as v from intraday_bars_1min where bar_date=? and symbol=?", src['cache.db'], params=[day, sym])
        where = 'cache.db'
    if not len(g) and not db:
        g = pd.read_sql("select symbol, t, o, h, l, c, v from bars where day=? and symbol=?", src['bars.db'], params=[day, sym]); where = 'bars.db'
    if not len(g): return None, where
    ts = pd.to_datetime(g.t, utc=True).dt.tz_convert('America/New_York')
    g = g.assign(m=(ts.dt.hour * 60 + ts.dt.minute).values).sort_values('m').drop_duplicates('m')
    return g[(g.m >= 570) & (g.m < 960)].reset_index(drop=True), where


def alpaca_bars(sym, day):
    d = datetime.strptime(day, '%Y-%m-%d')
    start = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET).astimezone(timezone.utc); end = datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET).astimezone(timezone.utc)
    req = StockBarsRequest(symbol_or_symbols=[sym], timeframe=TimeFrame(1, TimeFrameUnit.Minute), start=start, end=end, feed=DataFeed.SIP)
    raw = client._to_dict(client.data_client.get_stock_bars(req)).get(sym) or []
    rows = [dict(m=b.timestamp.astimezone(ET).hour * 60 + b.timestamp.astimezone(ET).minute, o=float(b.open), h=float(b.high), l=float(b.low), c=float(b.close), v=float(b.volume)) for b in raw]
    g = pd.DataFrame(rows)
    return g[(g.m >= 570) & (g.m < 960)].sort_values('m').drop_duplicates('m').reset_index(drop=True) if len(g) else g


def stats(g):
    if g is None or not len(g): return None
    k = g[g.m <= 630]
    return dict(first_m=int(g.m[0]), o0=float(g.o[0]), hod1030=float(k.h.max()), cumv1030=float(k.v.sum()), n=len(g), n1030=len(k))


print(f"{'sym':6s} {'day':10s} {'src':8s} | {'first_m':>7s} {'o0 study/alp':>16s} {'HOD@10:30 study/alp':>22s} {'cumV@10:30 study/alp':>24s} {'n bars':>12s} flags")
for pr in pairs:
    sym, day = pr[0], pr[1]; db = pr[2] if len(pr) > 2 else None
    g, where = study_bars(sym, day, db); a = alpaca_bars(sym, day); s1 = stats(g); s2 = stats(a)
    if s1 is None or s2 is None:
        print(f"{sym:6s} {day} {where:8s} | study {s1} alpaca {s2}"); continue
    fl = []
    if s1['first_m'] != s2['first_m']: fl.append('FIRST_MIN')
    if abs(s1['o0'] - s2['o0']) > 0.011: fl.append('O0')
    if abs(s1['hod1030'] - s2['hod1030']) > 0.011: fl.append('HOD')
    if s2['cumv1030'] and abs(s1['cumv1030'] / s2['cumv1030'] - 1) > 0.02: fl.append(f"CUMV{s1['cumv1030']/s2['cumv1030']:.3f}")
    if s1['n'] != s2['n']: fl.append(f"NBARS{s1['n']}/{s2['n']}")
    print(f"{sym:6s} {day} {where:8s} | {s1['first_m']:3d}/{s2['first_m']:3d} {s1['o0']:8.2f}/{s2['o0']:<7.2f} {s1['hod1030']:10.2f}/{s2['hod1030']:<10.2f} {s1['cumv1030']:11.0f}/{s2['cumv1030']:<11.0f} {s1['n']:5d}/{s2['n']:<5d} {' '.join(fl)}")
