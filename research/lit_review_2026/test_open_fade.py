#!/usr/bin/env python3
"""RUNBOOK row 10 — M18 (B-H-B10, C-H-C3): do prior-day ATTENTION names fade at the next open?
Selection is causal: on day t-1's close rank every $5+ name with 20-day dollar volume >= $2M by abs(close-to-close return)
x (volume / ADV20); take the top 20. On day t measure, from 1-minute SIP bars, the return from the 09:30 open to 09:35,
10:00, 10:30, 12:00 and the close; and the same for a matched control (ranks 100-120 of the same day, i.e. ordinary names).
Bars for day t are fetched for exactly those names (they are NOT in the >=5%-range store unless day t itself was wild -
using that store would re-introduce the look-ahead this program is about). Store: research/lit_review_2026/attention.db.
Usage: python3 test_open_fade.py fetch | score"""
import os, sys, time, sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
ET = ZoneInfo('America/New_York'); STEP = sys.argv[1] if len(sys.argv) > 1 else 'fetch'
DB = 'research/lit_review_2026/attention.db'
d = pd.read_parquet('research/lit_review_2026/daily_panel.parquet', columns=['symbol', 'bar_date', 'close', 'dvol20', 'ret_cc', 'vol_ratio'])
import re
bad = [c for c in d.symbol.cat.categories if re.match(r'^Z[VWX]ZZ|^ZZ', str(c))] if hasattr(d.symbol, 'cat') else []
if bad: d = d[~d.symbol.isin(bad)]
d = d[(d.close >= 5) & (d.dvol20 >= 2e6) & d.ret_cc.notna() & d.vol_ratio.notna() & (d.ret_cc.abs() <= 0.5)].copy()
d['symbol'] = d.symbol.astype(str); d = d[d.symbol.str.match(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$')]
d['attn'] = d.ret_cc.abs() * d.vol_ratio
d = d.sort_values(['bar_date', 'attn'], ascending=[True, False])
d['rank'] = d.groupby('bar_date').attn.rank(method='first', ascending=False)
days = sorted(d.bar_date.unique())
nxt = {a: b for a, b in zip(days[:-1], days[1:])}                       # day t-1 -> day t
sel = d[(d['rank'] <= 20) | ((d['rank'] > 100) & (d['rank'] <= 120))].copy()
sel['group'] = np.where(sel['rank'] <= 20, 'attention', 'control'); sel['trade_day'] = sel.bar_date.map(nxt)
sel = sel[sel.trade_day.notna()][['symbol', 'bar_date', 'trade_day', 'group', 'attn', 'ret_cc', 'dvol20']]
con = sqlite3.connect(DB, timeout=60)
con.execute("create table if not exists bars (symbol text, day text, m integer, o real, h real, l real, c real, v real, primary key (symbol, day, m))")
con.execute("create table if not exists done (day text primary key, n integer)"); con.commit()
if STEP == 'fetch':
    from config import Config
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed, Adjustment
    cfg = Config(); client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    have = {r[0] for r in con.execute("select day from done")}; t0 = time.time(); tds = sorted(sel.trade_day.unique())
    print(f'trade days {len(tds)} | symbol-days {len(sel)}', flush=True)
    for k, day in enumerate(tds):
        if day in have: continue
        syms = sorted(sel[sel.trade_day == day].symbol.unique()); d0 = datetime.strptime(day, '%Y-%m-%d').replace(tzinfo=ET)
        rows = []
        for i in range(0, len(syms), 200):
            chunk = syms[i:i + 200]
            for attempt in range(3):
                try:
                    r = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                                                               start=d0.replace(hour=9, minute=30).astimezone(timezone.utc),
                                                               end=d0.replace(hour=16, minute=0).astimezone(timezone.utc), feed=DataFeed.SIP, adjustment=Adjustment.RAW))
                    data = r.data if hasattr(r, 'data') else r
                    for s in chunk:
                        for b in data.get(s, []):
                            e = b.timestamp.astimezone(ET); rows.append((s, day, e.hour * 60 + e.minute, float(b.open), float(b.high), float(b.low), float(b.close), float(b.volume)))
                    break
                except Exception as ex:
                    print(f'  {day} chunk {i} attempt {attempt+1}: {str(ex)[:90]}', flush=True); time.sleep(4 * (attempt + 1))
        con.executemany("insert or replace into bars values (?,?,?,?,?,?,?,?)", rows); con.execute("insert or replace into done values (?,?)", (day, len(rows))); con.commit()
        if k % 20 == 0: print(f'{k+1}/{len(tds)} {day} {len(syms)} names {len(rows)} bars | {(time.time()-t0)/60:.1f} min', flush=True)
    print('FETCH DONE', flush=True)
else:
    out = []
    for day, g in sel.groupby('trade_day'):
        b = pd.read_sql("select symbol, m, o, c from bars where day=?", con, params=(day,))
        if not len(b): continue
        for r in g.itertuples():
            x = b[(b.symbol == r.symbol) & (b.m >= 570) & (b.m < 960)].sort_values('m')
            if len(x) < 60: continue
            o = float(x.o.iloc[0])
            if o <= 0: continue
            def px(mm):
                y = x[x.m <= mm]
                return float(y.c.iloc[-1]) if len(y) else np.nan
            out.append(dict(day=day, symbol=r.symbol, group=r.group, dvol20=r.dvol20, prev_ret=r.ret_cc,
                            r0935=px(574)/o-1, r1000=px(599)/o-1, r1030=px(629)/o-1, r1200=px(719)/o-1, rclose=float(x.c.iloc[-1])/o-1))
    T = pd.DataFrame(out); T = T[T[['r0935','r1000','r1030','r1200','rclose']].abs().max(axis=1) <= 0.5]
    T['split'] = np.where(T.day < '2026-01-01', 'TRAIN', np.where(T.day < '2026-06-01', 'VAL', 'TEST'))
    T['dvb'] = pd.cut(T.dvol20, [2e6, 1e7, 5e7, 1e12], labels=['$2-10M', '$10-50M', '>$50M'])
    T.to_csv('research/lit_review_2026/open_fade_rows.csv', index=False)
    L = [f'# RUNBOOK row 10 — M18 open fade on prior-day attention names | rows {len(T)}', '',
         'mean return in bps from the 09:30 open to each clock time; "attention" = prior-day top-20 by abs(return) x volume ratio, "control" = ranks 100-120 the same day', '']
    for sp in ('TRAIN', 'VAL', 'TEST'):
        t = T[T.split == sp].groupby(['group', 'dvb'], observed=True)[['r0935','r1000','r1030','r1200','rclose']].agg(lambda s: round(s.mean()*1e4))
        n = T[T.split == sp].groupby(['group', 'dvb'], observed=True).size().rename('n')
        L += [f'## {sp}', pd.concat([n, t], axis=1).to_string(), '']
    for sp in ('TRAIN', 'VAL', 'TEST'):
        a = T[(T.split == sp) & (T.group == 'attention')]; c = T[(T.split == sp) & (T.group == 'control')]
        if len(a) > 30 and len(c) > 30:
            diff = a.r1030.mean() - c.r1030.mean(); se = np.sqrt(a.r1030.var()/len(a) + c.r1030.var()/len(c))
            L.append(f'{sp}: attention minus control, open→10:30 = {diff*1e4:+.1f} bps (t {diff/se:+.2f}); attention alone {a.r1030.mean()*1e4:+.1f} bps, n {len(a)}')
    open('research/lit_review_2026/open_fade.md', 'w').write('\n'.join(L)); print('\n'.join(L)); print('DONE', flush=True)
