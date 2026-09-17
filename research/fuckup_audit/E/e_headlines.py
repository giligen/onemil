#!/usr/bin/env python3
"""Stage E — are the `news_only` articles recaps/lists or company events?

The ORB rule book says recap-only articles performed EQUAL to real catalysts for LONGS
(research/orb_machine_rules.md).  This re-checks that on the one family whose `news_only` effect
passed the pre-registered sign-agreement rule (F8 N=5): 30 random TRAIN trades of the bucket,
their premarket headlines pulled from the same Alpaca endpoint `D/d0_news.py` used, classified by
a keyword rule stated here in advance.

Writes E/score_e_headlines.md and E/score_e_headlines.csv.  Read-only elsewhere.
"""
import os, sys, time, json
import pandas as pd, requests
from dotenv import load_dotenv

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
load_dotenv(f'{ROOT}/.env')
E = f'{ROOT}/research/fuckup_audit/E'
RD = dict(keep_default_na=False, na_values=[''])
URL = 'https://data.alpaca.markets/v1beta1/news'
H = {'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'], 'APCA-API-SECRET-KEY': os.environ['ALPACA_API_SECRET']}

# the classifier, fixed BEFORE the pull
RECAP = ['moving', 'movers', 'gainers', 'losers', 'premarket', 'pre-market', 'market update',
         'stocks to watch', 'watchlist', 'why is', 'shares are trading', 'trading higher',
         'trading lower', 'biggest', 'top stocks', 'here is how', 'what you need to know',
         'wrap', 'roundup', 'session', 'benzinga', 'jim cramer', 'mid-day', 'midday']
EVENT = ['earnings', 'results', 'fda', 'approval', 'phase', 'trial', 'merger', 'acquisition',
         'acquire', 'offering', 'contract', 'award', 'guidance', 'upgrade', 'downgrade',
         'initiat', 'partnership', 'agreement', 'patent', 'dividend', 'split', 'ceo', 'appoint',
         'files', 'sec ', 'lawsuit', 'settlement', 'revenue', 'launch', 'order', 'acquisition']


def classify(h):
    lo = h.lower()
    r = any(k in lo for k in RECAP)
    e = any(k in lo for k in EVENT)
    if r and not e:
        return 'recap'
    if e and not r:
        return 'event'
    if e and r:
        return 'both'
    return 'other'


def main():
    samp = pd.read_csv(f'{E}/score_e_headline_sample.csv', dtype={'day': str, 'symbol': str}, **RD)
    rows = []
    for i, r in samp.iterrows():
        d0 = pd.Timestamp(r.day)
        st = ((d0 - pd.Timedelta(days=1)).tz_localize('America/New_York') + pd.Timedelta(hours=15)).tz_convert('UTC')
        en = (d0.tz_localize('America/New_York') + pd.Timedelta(hours=9, minutes=30)).tz_convert('UTC')
        p = {'symbols': r.symbol, 'start': st.isoformat(), 'end': en.isoformat(), 'limit': 50, 'sort': 'desc'}
        try:
            resp = requests.get(URL, params=p, headers=H, timeout=(5, 10))
            resp.raise_for_status()
            arts = resp.json().get('news', [])
        except Exception as e:
            print(f'FAIL {r.symbol} {r.day}: {e}', flush=True)
            continue
        for a in arts:
            rows.append(dict(day=r.day, symbol=r.symbol, net_hold=round(float(r.net_hold), 4),
                             ts=a.get('created_at', ''), source=a.get('source', ''),
                             headline=a.get('headline', ''), cls=classify(a.get('headline', ''))))
        time.sleep(0.35)
    T = pd.DataFrame(rows)
    T.to_csv(f'{E}/score_e_headlines.csv', index=False)
    pd.set_option('display.width', 250, 'display.max_colwidth', 110)
    L = ['# Stage E — the `news_only` headlines (F8 N=5, 30 random TRAIN trades of the bucket)', '',
         f'generated {time.strftime("%Y-%m-%d %H:%M:%S")} | window prev-day 15:00 ET -> 09:30 ET, '
         'the same endpoint and window `D/d0_news.py` used | classifier fixed before the pull', '',
         f'symbol-days sampled: {samp.shape[0]}, with at least one premarket article: '
         f'{T.groupby(["day","symbol"]).ngroups if len(T) else 0}, articles: {len(T)}', '',
         '## class mix (per article)', '', (T.cls.value_counts().to_string() if len(T) else '(none)'), '',
         '## per symbol-day: the dominant class and that trade\'s net R', '']
    if len(T):
        g = T.groupby(['day', 'symbol']).agg(n=('headline', 'size'),
                                             cls=('cls', lambda s: s.value_counts().index[0]),
                                             net=('net_hold', 'first')).reset_index()
        L += [g.to_string(index=False), '',
              '## mean net R of the sampled trades by dominant class', '',
              g.groupby('cls').net.agg(['size', 'mean']).round(4).to_string(), '',
              '## every headline', '', T[['day', 'symbol', 'ts', 'cls', 'headline']].to_string(index=False), '']
    open(f'{E}/score_e_headlines.md', 'w').write('\n'.join(L))
    print('\n'.join(L[:40]))


if __name__ == '__main__':
    main()
