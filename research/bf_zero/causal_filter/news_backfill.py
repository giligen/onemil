#!/usr/bin/env python3
"""CAUSAL_FILTER — premarket news backfill for the HOD population.

The shipped nightly (`data/research/orb_news_catalyst_nightly.csv`) covers 1.8% of this population
(it is the ORB candidate set), so the PREREG's allowed sampled backfill is run for the whole
population instead — the vendor call is batched per trading day, so full coverage costs the same
order as a sample. Free API, one request per page, gentle pacing.

Window: prev calendar day 15:00 ET -> trade day 09:30 ET (strictly before the 09:30 open, so the
flag is causal for every signal minute in this book). Same definition as the shipped ORB gate
except the end is 09:30 not 09:35 — our signals start at the open, not at 09:35.

Output: causal_filter/news.csv (symbol, day, n_articles, earliest, latest), resumable per day.
"""
import os, sys, time
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
load_dotenv(f'{ROOT}/.env')
D = f'{ROOT}/research/bf_zero/causal_filter'
OUT = f'{D}/news.csv'
URL = 'https://data.alpaca.markets/v1beta1/news'
H = {'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'],
     'APCA-API-SECRET-KEY': os.environ['ALPACA_API_SECRET']}

RD = dict(dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
pop = pd.read_csv(f'{D}/population.csv', **RD)[['day', 'symbol']].drop_duplicates()
done = set()
if os.path.exists(OUT):
    done = set(pd.read_csv(OUT, **RD).day.unique())
days = [d for d in sorted(pop.day.unique()) if d not in done]
print(f'population {len(pop)} rows | days {pop.day.nunique()} | todo {len(days)}', flush=True)

for i, day in enumerate(days):
    syms = sorted(pop[pop.day == day].symbol.unique())
    d = pd.Timestamp(day)
    st = ((d - pd.Timedelta(days=1)).tz_localize('America/New_York') + pd.Timedelta(hours=15)).tz_convert('UTC').isoformat()
    en = (d.tz_localize('America/New_York') + pd.Timedelta(hours=9, minutes=30)).tz_convert('UTC').isoformat()
    arts, token, failed = [], None, False
    for _page in range(8):
        p = {'symbols': ','.join(syms), 'start': st, 'end': en, 'limit': 50, 'sort': 'desc'}
        if token:
            p['page_token'] = token
        r = None
        for att in (1, 2, 3):
            try:
                r = requests.get(URL, params=p, headers=H, timeout=(5, 30))
                r.raise_for_status()
                break
            except Exception as e:
                r = None
                if att == 3:
                    print(f'  FAIL {day}: {e}', flush=True)
                else:
                    time.sleep(3)
        if r is None:
            failed = True
            break
        j = r.json()
        arts += j.get('news', [])
        token = j.get('next_page_token')
        if not token:
            break
        time.sleep(0.15)
    if failed:                                              # never write a silent zero for a failed day
        continue
    per = {s: [] for s in syms}
    for a in arts:
        for s in a.get('symbols', []):
            if s in per:
                per[s].append(a)
    rows = [dict(symbol=s, day=day, n_articles=len(per[s]),
                 earliest=min((a['created_at'] for a in per[s]), default=''),
                 latest=max((a['created_at'] for a in per[s]), default='')) for s in syms]
    pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    if (i + 1) % 25 == 0:
        print(f'  {i + 1}/{len(days)} days', flush=True)
    time.sleep(0.2)
print('DONE', flush=True)
