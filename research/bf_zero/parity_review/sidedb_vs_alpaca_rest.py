#!/usr/bin/env python3
"""Per-trade parity on the spec's OWN book: for spec-book trades whose bars came from the Databento side DBs
(topup.db / pit_bars_1min.db, both EQUS.MINI ohlcv-1m per their fetchers), fetch Alpaca SIP 1-min bars (REST, the
tape live sees) for the same symbol-day and (a) diff the tapes (bars, exact matches, volume ratio, HOD), (b) re-run
`trading.hod_break.simulate` on BOTH tapes with the book's adv20 — same trade? Also a few bars.db symbol-days (the
study's own fetch, which served ZERO book trades) and, as a control, cache.db-sourced book trades vs REST (should be
identical: same source). Plus the ADV-scale check (EQUS.SUMMARY daily vs Alpaca daily_bars, pyarrow-filtered).
Read-only; no Databento spend."""
import os, sys, sqlite3, random
import numpy as np, pandas as pd, pyarrow.parquet as pq
ROOT = '/home/ec2-user/onemil'; sys.path.insert(0, ROOT); os.chdir(ROOT)
OUT = f'{ROOT}/research/bf_zero/parity_review'
from config import Config
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from trading.hod_break import HodBreakParams, simulate
import pytz
ET = pytz.timezone('America/New_York'); P = HodBreakParams()
cfg = Config(); hist = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
random.seed(7)
S = pd.read_csv(f'{OUT}/spec_book_sources.csv')
side = {'topup.db': f'{ROOT}/research/ignition_capcheck/topup.db', 'pit_bars_1min.db': f'{ROOT}/data/research/databento/pit_bars_1min.db', 'bars.db': f'{ROOT}/research/bf_zero/bars.db'}
cons = {k: sqlite3.connect(f'file:{p}?mode=ro', uri=True, timeout=10) for k, p in side.items()}
cache = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=5)

def to_m(ts):
    t = pd.to_datetime(ts, utc=True).dt.tz_convert('America/New_York'); return (t.dt.hour * 60 + t.dt.minute).values

def rth(df):
    df = df.assign(m=to_m(df.t)).sort_values('m').drop_duplicates('m'); return df[(df.m >= 570) & (df.m < 960)].reset_index(drop=True)

def side_bars(name, sym, day):
    return rth(pd.read_sql("select t, o, h, l, c, v from bars where symbol=? and day=?", cons[name], params=(sym, day)))

def cache_bars(sym, day):
    return rth(pd.read_sql("select timestamp as t, open as o, high as h, low as l, close as c, volume as v from intraday_bars_1min where symbol=? and bar_date=?", cache, params=(sym, day)))

def alpaca_bars(sym, day):
    d = pd.Timestamp(day).to_pydatetime()
    o = ET.localize(d.replace(hour=9, minute=30)); c = ET.localize(d.replace(hour=15, minute=59))
    data = hist.get_stock_bars(StockBarsRequest(symbol_or_symbols=[sym], timeframe=TimeFrame(1, TimeFrameUnit.Minute), start=o, end=c, feed=DataFeed.SIP)).data
    lst = data.get(sym, [])
    if not lst: return pd.DataFrame(columns=['t', 'o', 'h', 'l', 'c', 'v', 'm'])
    return rth(pd.DataFrame([dict(t=b.timestamp, o=float(b.open), h=float(b.high), l=float(b.low), c=float(b.close), v=float(b.volume)) for b in lst]))

def sim(df, adv20):
    if len(df) < 10: return None
    o, h, l, c, v = (df[k].values.astype(float) for k in 'ohlcv'); m = df.m.values.astype(int)
    return simulate(o, h, l, c, v, m, adv20, P)

def compare(tag, sym, day, A, B, adv20):
    ka = dict(zip(A.m, zip(A.o, A.h, A.l, A.c, A.v))); kb = dict(zip(B.m, zip(B.o, B.h, B.l, B.c, B.v)))
    both = sorted(set(ka) & set(kb)); exact = sum(ka[m] == kb[m] for m in both)
    va = A.v.sum(); vb = B.v.sum()
    ta, tb = sim(A, adv20), sim(B, adv20)
    fa = f'{"none" if ta is None else f"entry_m {int(A.m[ta.entry_idx])} rr {ta.rr:+.2f} {ta.reason}"}'
    fb = f'{"none" if tb is None else f"entry_m {int(B.m[tb.entry_idx])} rr {tb.rr:+.2f} {tb.reason}"}'
    same = (ta is None and tb is None) or (ta is not None and tb is not None and A.m[ta.entry_idx] == B.m[tb.entry_idx] and abs(ta.rr - tb.rr) < 0.05)
    return dict(tag=tag, symbol=sym, day=day, n_side=len(A), n_alpaca=len(B), exact=exact, vol_ratio=round(va / vb, 3) if vb else None,
                hod_side=round(A.h.max(), 4) if len(A) else None, hod_alpaca=round(B.h.max(), 4) if len(B) else None,
                first_side=int(A.m[0]) if len(A) else None, first_alpaca=int(B.m[0]) if len(B) else None, sim_side=fa, sim_alpaca=fb, same_trade=same)

rows = []
for name in ('topup.db', 'pit_bars_1min.db'):
    sub = S[S.source.str.startswith(name)]
    pick = sub[sub.split == 'TEST'].sample(min(10, (sub.split == 'TEST').sum()), random_state=1) if (sub.split == 'TEST').any() else sub.sample(min(6, len(sub)), random_state=1)
    for r in pick.itertuples():
        rows.append(compare(name, r.symbol, r.day, side_bars(name, r.symbol, r.day), alpaca_bars(r.symbol, r.day), float(r.adv20)))
        print(rows[-1], flush=True)
# bars.db: random symbol-days from a mid-2026 day (served no book trades)
bd = pd.read_sql("select distinct symbol from bars where day='2026-06-15'", cons['bars.db']).symbol.tolist(); random.shuffle(bd)
for sym in bd[:6]:
    A = side_bars('bars.db', sym, '2026-06-15'); B = alpaca_bars(sym, '2026-06-15')
    rows.append(compare('bars.db', sym, '2026-06-15', A, B, float('nan'))); print(rows[-1], flush=True)
# control: cache.db-sourced TEST trades vs REST (same source -> should be identical)
for r in S[(S.source.str.startswith('cache')) & (S.split == 'TEST')].sample(6, random_state=2).itertuples():
    rows.append(compare('cache.db', r.symbol, r.day, cache_bars(r.symbol, r.day), alpaca_bars(r.symbol, r.day), float(r.adv20))); print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/sidedb_vs_alpaca_rest.csv', index=False)
print('\n=== summary by source (side-DB tape vs Alpaca SIP REST for the same symbol-day)')
print(R.groupby('tag').agg(n=('symbol', 'size'), bars_side=('n_side', 'mean'), bars_alpaca=('n_alpaca', 'mean'), exact_bars=('exact', 'mean'), vol_ratio_med=('vol_ratio', 'median'),
                           hod_equal=('hod_side', lambda x: int((x.values == R.loc[x.index, 'hod_alpaca'].values).sum())), same_trade=('same_trade', 'sum')).round(3).to_string())
# ADV scale: EQUS.SUMMARY daily (spec ADV20 source) vs Alpaca daily_bars, 3 symbols, 2026-09-11
syms = ['A', 'AAPL', 'AAOI']
tbl = pq.read_table(f'{ROOT}/data/research/databento/equs_daily_2025_2026.parquet', columns=['symbol', 'bar_date', 'volume'], filters=[('symbol', 'in', syms)]).to_pandas()
tbl['bar_date'] = tbl.bar_date.astype(str).str[:10]; summ = tbl[tbl.bar_date == '2026-09-11'].set_index('symbol').volume
alp = dict(cache.execute("select symbol, volume from daily_bars where bar_date='2026-09-11' and symbol in ('A','AAPL','AAOI')").fetchall())
print('\n2026-09-11 daily volume: EQUS.SUMMARY (spec ADV20 source) | Alpaca daily_bars (live ADV20 source)')
for s in syms: print(f'  {s}: {int(summ.get(s, 0)):>12,} | {int(alp.get(s, 0)):>12,}')
