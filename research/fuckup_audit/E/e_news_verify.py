#!/usr/bin/env python3
"""Verification of E/news_presence_e.csv: coverage vs the E key set, fetch failures,
per-split news shares, and a 10-key independent re-call of the Alpaca news API.

Prints a markdown-ish block that E/news_summary.md is written from. Read-only except stdout.
"""
import json, os, random, sys, time
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
load_dotenv(f'{ROOT}/.env')
E = 'research/fuckup_audit/E'; D = 'research/fuckup_audit/D'
RD = dict(keep_default_na=False, na_values=[''])
URL = 'https://data.alpaca.markets/v1beta1/news'
H = {'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'], 'APCA-API-SECRET-KEY': os.environ['ALPACA_API_SECRET']}

keys = pd.concat([
    pd.read_csv(f'{E}/{f}', usecols=['bar_date', 'symbol', 'split'],
                dtype={'bar_date': str, 'symbol': str, 'split': str}, **RD)
    .rename(columns={'bar_date': 'day'}) for f in ('u1_keys.csv', 'u2_keys.csv')
], ignore_index=True).drop_duplicates(['day', 'symbol'])
print(f'E keys (U1uU2)            : {len(keys):,}')

e = pd.read_csv(f'{E}/news_presence_e.csv', dtype={'day': str, 'symbol': str}, **RD)
d = pd.read_csv(f'{D}/news_presence.csv', dtype={'day': str, 'symbol': str}, **RD)
print(f'E/news_presence_e.csv rows: {len(e):,}  (uniq {len(e.drop_duplicates(["day","symbol"])):,})')
print(f'D/news_presence.csv rows  : {len(d):,}  (uniq {len(d.drop_duplicates(["day","symbol"])):,})')
print(f'columns identical         : {list(e.columns) == list(d.columns)}  {list(e.columns)}')
print(f'fetch failures E          : {int((e.fetch_ok == 0).sum()):,} rows')
print(f'fetch failures D          : {int((d.fetch_ok == 0).sum()):,} rows')

both = pd.concat([e, d], ignore_index=True).drop_duplicates(['day', 'symbol'])
m = keys.merge(both, on=['day', 'symbol'], how='left', indicator=True)
miss = m[m._merge == 'left_only']
print(f'E keys covered by E u D   : {len(keys) - len(miss):,} / {len(keys):,} = '
      f'{(1 - len(miss) / len(keys)) * 100:.3f}%   missing {len(miss):,}')
if len(miss):
    print(miss[['day', 'symbol']].head(20).to_string(index=False))

m2 = m[m._merge == 'both'].copy()
m2['pre'] = m2.n_prev15_to_0930 > 0
m2['intr'] = m2.n_0930_to_1401 > 0
print('\nper split (all E keys, both files):')
print('| split | keys | pre-09:30 news | share | intraday news | share |')
print('|---|---:|---:|---:|---:|---:|')
for s in ['TRAIN', 'VAL', 'TEST']:
    g = m2[m2.split == s]
    if not len(g):
        continue
    print(f'| {s} | {len(g):,} | {int(g.pre.sum()):,} | {g.pre.mean()*100:.1f}% | '
          f'{int(g.intr.sum()):,} | {g.intr.mean()*100:.1f}% |')
print(f'| ALL | {len(m2):,} | {int(m2.pre.sum()):,} | {m2.pre.mean()*100:.1f}% | '
      f'{int(m2.intr.sum()):,} | {m2.intr.mean()*100:.1f}% |')

# ---- independent spot check: 10 random E-fetched keys, re-called one symbol at a time ----
print('\nspot check (10 random E rows, single-symbol API re-call):')
random.seed(20260916)
sample = e.sample(10, random_state=20260916)
bad = 0
print('| day | symbol | stored pre / intraday | recall pre / intraday | match |')
print('|---|---|---|---|---|')
for _, r in sample.iterrows():
    d0 = pd.Timestamp(r.day)
    st = ((d0 - pd.Timedelta(days=1)).tz_localize('America/New_York') + pd.Timedelta(hours=15)).tz_convert('UTC')
    en = (d0.tz_localize('America/New_York') + pd.Timedelta(hours=14, minutes=1)).tz_convert('UTC')
    op = d0.tz_localize('America/New_York') + pd.Timedelta(hours=9, minutes=30)
    ts, token = [], None
    for _p in range(25):
        p = {'symbols': r.symbol, 'start': st.isoformat(), 'end': en.isoformat(), 'limit': 50, 'sort': 'desc'}
        if token:
            p['page_token'] = token
        resp = requests.get(URL, params=p, headers=H, timeout=(5, 10))
        resp.raise_for_status()
        j = resp.json()
        for a in j.get('news', []):
            if r.symbol in (a.get('symbols') or []):
                ts.append(pd.Timestamp(a['created_at']))
        token = j.get('next_page_token')
        time.sleep(0.35)
        if not token:
            break
    pre = sum(1 for x in ts if x < op)
    intr = sum(1 for x in ts if x >= op)
    okm = (pre == int(r.n_prev15_to_0930)) and (intr == int(r.n_0930_to_1401))
    bad += (not okm)
    print(f'| {r.day} | {r.symbol} | {r.n_prev15_to_0930} / {r.n_0930_to_1401} | {pre} / {intr} | '
          f'{"OK" if okm else "MISMATCH"} |')
print(f'spot-check mismatches: {bad}/10')
