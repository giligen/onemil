#!/usr/bin/env python3
"""Stage E side job — causal news presence for every (day, symbol) of the E causal universe.

Thin wrapper around research/fuckup_audit/D/d0_news.py (same window, same columns, same
throttle), re-pointed at the Stage-E key set:

    KEYS = (E/u1_keys.csv  u  E/u2_keys.csv)  minus  the keys already in D/news_presence.csv

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
days; the CSV is appended per day. I/O bound - run with ulimit -v 1000000.

ONE deliberate deviation from D: if a chunk's window is deeper than MAX_PAGES x 50 articles the
page walk would silently drop the OLDEST articles (sort=desc), i.e. exactly the premarket ones.
Here a truncated chunk is split in half and re-fetched until it fits; every split is logged
(`TRUNC ...`) so the rate can be compared against D's file, whose counts would be understated
wherever this fires.

Output: research/fuckup_audit/E/news_presence_e.csv   (same columns as D/news_presence.csv)
State:  research/fuckup_audit/E/news_state.json
"""
import json, os, sys, time
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
load_dotenv(f'{ROOT}/.env')
E = 'research/fuckup_audit/E'
D = 'research/fuckup_audit/D'
U1 = f'{E}/u1_keys.csv'
U2 = f'{E}/u2_keys.csv'
DONE_KEYS = f'{D}/news_presence.csv'
OUT = f'{E}/news_presence_e.csv'
STATE = f'{E}/news_state.json'
URL = 'https://data.alpaca.markets/v1beta1/news'
H = {'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'], 'APCA-API-SECRET-KEY': os.environ['ALPACA_API_SECRET']}
CHUNK = 50
MAX_PAGES = 25
SLEEP = 0.35
TIMEOUT = (5, 8)

RD = dict(keep_default_na=False, na_values=[''])


def load_keys():
    """(day, symbol) of U1 u U2 minus the keys D already fetched."""
    parts = []
    for p in (U1, U2):
        parts.append(pd.read_csv(p, usecols=['bar_date', 'symbol'],
                                 dtype={'bar_date': str, 'symbol': str}, **RD)
                     .rename(columns={'bar_date': 'day'}))
    e = pd.concat(parts, ignore_index=True).drop_duplicates()
    d = pd.read_csv(DONE_KEYS, usecols=['day', 'symbol'],
                    dtype={'day': str, 'symbol': str}, **RD).drop_duplicates()
    d['_have'] = 1
    m = e.merge(d, on=['day', 'symbol'], how='left')
    todo = m[m._have.isna()][['day', 'symbol']]
    print(f'U1uU2 {len(e):,} keys | D already has {len(e) - len(todo):,} | todo {len(todo):,}', flush=True)
    return todo


def fetch_window(symbols, st, en, depth=0):
    """-> (per-symbol list of timestamps, ok, n_requests). Splits on page exhaustion."""
    per = {s: [] for s in symbols}
    ok = True
    nreq = 0
    token = None
    truncated = False
    for _page in range(MAX_PAGES):
        p = {'symbols': ','.join(symbols), 'start': st.isoformat(), 'end': en.isoformat(),
             'limit': 50, 'sort': 'desc'}
        if token:
            p['page_token'] = token
        r = None
        for att in (1, 2, 3):
            try:
                nreq += 1
                r = requests.get(URL, params=p, headers=H, timeout=TIMEOUT)
                if r.status_code == 429:
                    time.sleep(3.0); r = None; continue
                r.raise_for_status()
                break
            except Exception as ex:
                r = None
                if att == 3:
                    print(f'FAIL chunk[{symbols[0]}..{symbols[-1]}] {st.date()}: {ex}', flush=True)
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
    else:
        truncated = bool(token)
    if truncated and token and len(symbols) > 1:
        # page walk hit the cap -> the oldest (premarket) articles would be lost. Split and redo.
        h = len(symbols) // 2
        print(f'TRUNC {st.date()} n={len(symbols)} depth={depth} -> split', flush=True)
        a_per, a_ok, a_n = fetch_window(symbols[:h], st, en, depth + 1)
        b_per, b_ok, b_n = fetch_window(symbols[h:], st, en, depth + 1)
        a_per.update(b_per)
        return a_per, (a_ok and b_ok), nreq + a_n + b_n
    if truncated and token:
        print(f'TRUNC-UNSPLITTABLE {st.date()} {symbols[0]}', flush=True)
        ok = False
    return per, ok, nreq


def main():
    todo_keys = load_keys()
    by_day = {d: sorted(g.symbol.unique()) for d, g in todo_keys.groupby('day')}
    days = sorted(by_day)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {'done': []}
    done = set(state['done'])
    todo = [d for d in days if d not in done]
    total_keys = sum(len(by_day[d]) for d in todo)
    print(f'{len(days)} days | todo {len(todo)} days / {total_keys:,} keys', flush=True)

    t0 = time.time()
    n_keys = 0
    n_req = 0
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
            cper, cok, cn = fetch_window(chunk, st, en)
            n_req += cn
            ok = ok and cok
            for s, v in cper.items():
                per[s].extend(v)
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
            eta = el / max(n_keys, 1) * (total_keys - n_keys)
            print(f'{di+1}/{len(todo)} days | {n_keys:,}/{total_keys:,} keys | {n_req:,} req | '
                  f'{el:.1f} min | ETA {eta:.0f} min', flush=True)
    print('DONE', flush=True)
    o = pd.read_csv(OUT, dtype={'day': str, 'symbol': str}, **RD)
    print('rows', len(o), 'uniq', len(o.drop_duplicates(['day', 'symbol'])),
          'has_pre_news', round((o.n_prev15_to_0930 > 0).mean() * 100, 1), '%',
          'fetch_ok', round(o.fetch_ok.mean() * 100, 1), '%', flush=True)


if __name__ == '__main__':
    main()
