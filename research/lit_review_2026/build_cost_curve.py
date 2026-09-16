#!/usr/bin/env python3
"""STAGE 2 — the cost curve: what edge must a strategy have, per universe segment, to be worth trading?

The week's lesson is arithmetic, not statistics. A $5-20 name in the first five minutes has a spread near 1.5% of price
while a typical intraday setup risks 2-4% of price, so a round trip costs 40-75% of one R against edges of about 0.1R.
This measures that ratio properly so every future search can start with a feasibility check instead of ending with one.

Sampling: signals from the honest candidate pool, stratified by price band x entry hour x 20-day dollar volume band, up to
N per cell. For each, Alpaca SIP NBBO in the signal minute: median spread, spread as % of price, and as a fraction of the
signal's own R. Output cost_curve.md + cost_curve.csv: per segment the median spread in bps of price, the median
spread/R, and the MINIMUM GROSS EDGE IN R a strategy needs there to clear a round trip.
Usage: python3 build_cost_curve.py [N_PER_CELL=40]"""
import os, sys, time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from config import Config
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest
from alpaca.data.enums import DataFeed
ET = ZoneInfo('America/New_York'); D = 'research/lit_review_2026'
N_CELL = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = f'{D}/cost_curve.csv'
src = 'research/bf_zero2/candidates3.csv' if os.path.exists('research/bf_zero2/candidates3.csv') else 'research/bf_zero2/candidates_full.csv'
c = pd.read_csv(src, usecols=lambda k: k in {'day', 'symbol', 'entry_m', 'price', 'r_pct', 'adv20'},
                dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''], low_memory=True)
c = c[(c.price >= 5) & c.r_pct.notna() & (c.r_pct >= 0.5) & c.adv20.notna()].copy()
c['pb'] = pd.cut(c.price, [5, 10, 20, 50, 200, 1e9], labels=['$5-10', '$10-20', '$20-50', '$50-200', '$200+'])
c['hb'] = pd.cut(c.entry_m, [569, 575, 600, 660, 780, 960], labels=['09:30-09:35', '09:35-10:00', '10:00-11:00', '11:00-13:00', '13:00+'])
c['lb'] = pd.cut(c.adv20 * c.price, [0, 5e6, 5e7, 1e13], labels=['<$5M/d', '$5-50M/d', '>$50M/d'])
samp = c.groupby(['pb', 'hb', 'lb'], observed=True, group_keys=False).apply(lambda g: g.sample(min(len(g), N_CELL), random_state=11))
print(f'source {src} | pool {len(c):,} | sampling {len(samp)} across {samp.groupby(["pb","hb","lb"], observed=True).ngroups} cells', flush=True)
done = set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT, dtype={'symbol': str, 'day': str}); done = set(zip(prev.day, prev.symbol, prev.entry_m))
cfg = Config(); cl = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
rows = []; t0 = time.time(); n = 0
for r in samp.itertuples():
    if (r.day, r.symbol, r.entry_m) in done: continue
    n += 1
    start = datetime.strptime(r.day, '%Y-%m-%d').replace(tzinfo=ET) + timedelta(minutes=int(r.entry_m))
    try:
        q = cl.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=r.symbol, start=start.astimezone(timezone.utc),
                                                   end=(start + timedelta(minutes=1)).astimezone(timezone.utc), feed=DataFeed.SIP, limit=600))
        qs = q.data.get(r.symbol, []) if hasattr(q, 'data') else q.get(r.symbol, [])
        sp = [(float(x.ask_price) - float(x.bid_price)) for x in qs if float(x.ask_price) > 0 and float(x.bid_price) > 0 and float(x.ask_price) >= float(x.bid_price)]
        rows.append(dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, price=r.price, r_pct=r.r_pct, pb=str(r.pb), hb=str(r.hb), lb=str(r.lb),
                         n_q=len(sp), spread=float(np.median(sp)) if sp else np.nan))
    except Exception as e:
        rows.append(dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, price=r.price, r_pct=r.r_pct, pb=str(r.pb), hb=str(r.hb), lb=str(r.lb), n_q=-1, spread=np.nan))
        if '429' in str(e) or 'too many' in str(e).lower(): time.sleep(15)
    time.sleep(0.35)
    if len(rows) >= 150:
        pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False); rows = []
        print(f'{n}/{len(samp)} | {(time.time()-t0)/60:.1f} min', flush=True)
if rows: pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
d = pd.read_csv(OUT, dtype={'symbol': str, 'day': str}); d = d[d.n_q > 0].copy()
d['sp_bps'] = d.spread / d.price * 1e4; d['sp_over_r'] = d.spread / (d.price * d.r_pct / 100)
g = d.groupby(['pb', 'hb'], observed=True).agg(n=('sp_bps', 'size'), spread_bps=('sp_bps', 'median'), sp_over_R=('sp_over_r', 'median'), r_pct=('r_pct', 'median')).round(3)
g['min_edge_R_roundtrip'] = (g.sp_over_R).round(3)
L = ['# Cost curve — the minimum gross edge a strategy needs, by segment', '',
     'spread = median NBBO spread in the signal minute. sp_over_R = one full spread as a fraction of the trade\'s own R.',
     'min_edge_R_roundtrip = the gross R per trade a strategy must earn in that segment just to break even on a round trip',
     '(half a spread in, half out). Compare it with the honest edges we have measured: about 0.1R at best.', '',
     g.to_string(), '', '## by liquidity', d.groupby(['lb', 'hb'], observed=True).agg(n=('sp_bps', 'size'), spread_bps=('sp_bps', 'median'), sp_over_R=('sp_over_r', 'median')).round(3).to_string()]
open(f'{D}/cost_curve.md', 'w').write('\n'.join(L)); print('\n'.join(L)); print('DONE', flush=True)
