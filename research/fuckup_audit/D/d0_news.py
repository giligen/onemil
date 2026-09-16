#!/usr/bin/env python3
"""D0 side job — causal news presence for every (day, symbol) of the D0 candidate table.

Window, per PLAN H4 and research/scripts/orb_news_backfill.py:
  prev CALENDAR day 15:00 ET  ->  trade day 14:01 ET (the population's last entry minute).
Per (day, symbol) we store enough to make the feature CAUSAL PER ROW:
  n_prev15_to_0930   articles in [prev 15:00 ET, 09:30 ET)        -> the ORB "premarket news" measurement
  last_prev_ts       latest created_at in that window (ISO, UTC)
  n_0930_to_1401     articles in [09:30 ET, 14:01 ET)
  mins_0930_to_1401  ET minute-of-day of each of those articles, space separated
A row with signal minute `sig_m` then has `n_articles_0930_to_sig = #{mins < sig_m}` and
`last_article_ts` = the latest article strictly before its own signal minute. Nothing after the
signal minute is ever used.

One API call per (day, symbol-chunk of 50), paginated (limit 50, <= 25 pages), 8 s timeout,
~0.35 s between calls (well under the 200 req/min data-API limit). Resumable: state json of done
days; the CSV is appended per day. I/O bound - run with ulimit -v 600000.
Output: research/fuckup_audit/D/news_presence.csv
"""
import json, os, sys, time
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
load_dotenv(f'{ROOT}/.env')
D = 'research/fuckup_audit/D'
TABLE = f'{D}/table.csv'
OUT = f'{D}/news_presence.csv'
STATE = f'{D}/news_state.json'
URL = 'https://data.alpaca.markets/v1beta1/news'
H = {'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'], 'APCA-API-SECRET-KEY': os.environ['ALPACA_API_SECRET']}
CHUNK = 50
MAX_PAGES = 25
SLEEP = 0.35
TIMEOUT = (5, 8)

t = pd.read_csv(TABLE, usecols=['day', 'symbol'], dtype={'day': str, 'symbol': str},
                keep_default_na=False, na_values=[''])
t = t.drop_duplicates()
by_day = {d: sorted(g.symbol.unique()) for d, g in t.groupby('day')}
days = sorted(by_day)
state = json.load(open(STATE)) if os.path.exists(STATE) else {'done': []}
done = set(state['done'])
todo = [d for d in days if d not in done]
print(f'{len(t):,} symbol-days | {len(days)} days | todo {len(todo)}', flush=True)

t0 = time.time()
n_keys = 0
for di, day in enumerate(todo):
    syms = by_day[day]
    d0 = pd.Timestamp(day)
    st = ((d0 - pd.Timedelta(days=1)).tz_localize('America/New_York') + pd.Timedelta(hours=15)).tz_convert('UTC')
    en = (d0.tz_localize('America/New_York') + pd.Timedelta(hours=14, minutes=1)).tz_convert('UTC')
    open_et = d0.tz_localize('America/New_York') + pd.Timedelta(hours=9, minutes=30)
    per = {s: [] for s in syms}
    ok = True
    for ci in range(0, len(syms), CHUNK):
        chunk = syms[ci:ci + CHUNK]
        token = None
        for _page in range(MAX_PAGES):
            p = {'symbols': ','.join(chunk), 'start': st.isoformat(), 'end': en.isoformat(),
                 'limit': 50, 'sort': 'desc'}
            if token:
                p['page_token'] = token
            r = None
            for att in (1, 2, 3):
                try:
                    r = requests.get(URL, params=p, headers=H, timeout=TIMEOUT)
                    if r.status_code == 429:
                        time.sleep(3.0); r = None; continue
                    r.raise_for_status()
                    break
                except Exception as e:
                    r = None
                    if att == 3:
                        print(f'FAIL {day} chunk{ci}: {e}', flush=True)
                        ok = False
                    else:
                        time.sleep(2.0)
            if r is None:
                break
            j = r.json()
            for a in j.get('news', []):
                ts = pd.Timestamp(a['created_at'])
                for sym in (a.get('symbols') or []):
                    if sym in per:
                        per[sym].append(ts)
            token = j.get('next_page_token')
            time.sleep(SLEEP)
            if not token:
                break
    rows = []
    for s in syms:
        ts = sorted(per[s])
        pre = [x for x in ts if x < open_et]
        during = [x for x in ts if x >= open_et]
        mins = []
        for x in during:
            e = x.tz_convert('America/New_York')
            mins.append(e.hour * 60 + e.minute)
        rows.append({'day': day, 'symbol': s, 'n_prev15_to_0930': len(pre),
                     'last_prev_ts': pre[-1].isoformat() if pre else '',
                     'n_0930_to_1401': len(during),
                     'mins_0930_to_1401': ' '.join(str(m) for m in sorted(mins)),
                     'fetch_ok': int(ok)})
    pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    n_keys += len(rows)
    state['done'].append(day)
    json.dump(state, open(STATE, 'w'))
    if n_keys // 500 != (n_keys - len(rows)) // 500 or di == len(todo) - 1:
        el = (time.time() - t0) / 60
        print(f'{di+1}/{len(todo)} days | {n_keys:,} keys | {el:.1f} min | {el/max(di+1,1)*len(todo):.0f} min eta',
              flush=True)
print('DONE', flush=True)
o = pd.read_csv(OUT, dtype={'day': str, 'symbol': str}, keep_default_na=False, na_values=[''])
print('rows', len(o), 'has_pre_news', round((o.n_prev15_to_0930 > 0).mean() * 100, 1), '%',
      'fetch_ok', round(o.fetch_ok.mean() * 100, 1), '%', flush=True)
