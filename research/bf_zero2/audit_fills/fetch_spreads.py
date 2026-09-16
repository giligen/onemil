#!/usr/bin/env python3
"""Audit item 8 — measure the REAL spread at the book's own fill minute.

The existing spread study (research/bf_zero/spread_study_clean.csv) samples HOD-break signals, whose
earliest entry is 9:36; 81% of this book's trades fill at 9:31-9:32, where spreads are widest. So fetch
the SIP NBBO for a stratified sample of the book's own (day, symbol, entry minute) and compare with the
study's cost assumptions: a 30 bps entry slip through the level and a 20 bps half-spread on the exit.

Read-only Alpaca historical quotes; resumable; → audit_fills/book_spreads.csv
"""
import os, sys, time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from config import Config
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest
from alpaca.data.enums import DataFeed

A = 'research/bf_zero2/audit_fills'; ET = ZoneInfo('America/New_York')
OUT = f'{A}/book_spreads.csv'
N_PER_SPLIT = int(sys.argv[1]) if len(sys.argv) > 1 else 300

b = pd.read_csv('research/bf_zero2/f6_2r_book.csv', dtype={'day': str, 'symbol': str}, keep_default_na=False, na_values=[''])
b = b[b.symbol.notna() & (b.symbol != '')]
s = b.groupby('split', group_keys=False).apply(lambda g: g.sample(min(len(g), N_PER_SPLIT), random_state=11)).reset_index(drop=True)
done = pd.read_csv(OUT, dtype={'day': str, 'symbol': str}) if os.path.exists(OUT) else pd.DataFrame(columns=['day', 'symbol'])
dk = set(zip(done.day, done.symbol))
print(f'book {len(b)} | sample {len(s)} | already done {len(dk)}', flush=True)
cfg = Config(); cli = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
rows = []; t0 = time.time(); ok = fail = 0
for i, r in enumerate(s.itertuples()):
    if (r.day, r.symbol) in dk: continue
    start = datetime.strptime(r.day, '%Y-%m-%d').replace(tzinfo=ET) + timedelta(minutes=int(r.em))
    try:
        q = cli.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=r.symbol, start=start.astimezone(timezone.utc),
                                                    end=(start + timedelta(minutes=1)).astimezone(timezone.utc),
                                                    feed=DataFeed.SIP, limit=3000))
        qs = q.data.get(r.symbol, []) if hasattr(q, 'data') else q.get(r.symbol, [])
        sp = [float(x.ask_price) - float(x.bid_price) for x in qs if float(x.ask_price) > 0 and float(x.bid_price) > 0 and float(x.ask_price) >= float(x.bid_price)]
        mid = [(float(x.ask_price) + float(x.bid_price)) / 2 for x in qs if float(x.ask_price) > 0 and float(x.bid_price) > 0 and float(x.ask_price) >= float(x.bid_price)]
        if not sp:
            rows.append(dict(day=r.day, symbol=r.symbol, em=int(r.em), split=r.split, price=r.price, r_pct=r.r_pct, n_quotes=0)); fail += 1
        else:
            rows.append(dict(day=r.day, symbol=r.symbol, em=int(r.em), split=r.split, price=r.price, r_pct=r.r_pct,
                             n_quotes=len(sp), spread_med=float(np.median(sp)), spread_mean=float(np.mean(sp)),
                             spread_p90=float(np.percentile(sp, 90)), mid_med=float(np.median(mid)),
                             ask_max=max(float(x.ask_price) for x in qs if float(x.ask_price) > 0))); ok += 1
    except Exception as e:
        rows.append(dict(day=r.day, symbol=r.symbol, em=int(r.em), split=r.split, price=r.price, r_pct=r.r_pct, n_quotes=-1, err=str(e)[:60])); fail += 1
        if '429' in str(e) or 'too many' in str(e).lower(): time.sleep(20)
    time.sleep(0.25)
    if len(rows) >= 50:
        pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False); rows = []
        print(f'{i + 1}/{len(s)} ok {ok} fail {fail} | {(time.time() - t0) / 60:.1f} min', flush=True)
if rows: pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
print(f'FETCHED ok {ok} fail {fail}', flush=True)
d = pd.read_csv(OUT); d = d[d.n_quotes > 0]
d['sp_pct'] = d.spread_med / d.price * 100; d['sp_mean_pct'] = d.spread_mean / d.price * 100
print(d.groupby(pd.cut(d.price, [5, 10, 20, 50, 1e9]), observed=True).agg(n=('sp_pct', 'size'), med=('sp_pct', 'median'), mean_of_med=('sp_pct', 'mean'), mean_of_mean=('sp_mean_pct', 'mean')).round(3).to_string(), flush=True)
print('\nby entry minute:', flush=True)
print(d.groupby(pd.cut(d.em, [569, 572, 575, 585, 841]), observed=True).agg(n=('sp_pct', 'size'), med=('sp_pct', 'median'), mean=('sp_pct', 'mean')).round(3).to_string(), flush=True)
print('DONE', flush=True)
