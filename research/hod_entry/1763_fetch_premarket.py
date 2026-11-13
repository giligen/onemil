"""Cell 1,763 B1 data - pre-market (04:00-09:29 ET) SIP minute bars for the PIT gapper symbol-days (TRAIN/VAL only).

PREREG: research/hod_entry/PREREG_1763.md.  Source: Alpaca /v2/stocks/bars, timeframe=1Min, feed=sip,
adjustment=raw, <= 50 symbols per request, paginated, <= 4 requests/s.  One checkpoint per day:
research/hod_entry/1762_cache/premarket/<day>.parquet (atomic write; existing days are skipped -> resumable).
An empty day file (no bars returned) is still written so the coverage count is explicit.  TEST days never fetched.
Keys: ALPACA_HOD_API_KEY/SECRET, else ALPACA_API_KEY/SECRET (WARNING logged on fallback); never printed.

Usage: cd /home/ec2-user/onemil && ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/hod_entry/1763_fetch_premarket.py
"""
import os
import sys
import time
import requests
import pandas as pd
from dotenv import load_dotenv

HE = '/home/ec2-user/onemil/research/hod_entry'
OUT = f'{HE}/1762_cache/premarket'
URL = 'https://data.alpaca.markets/v2/stocks/bars'
COLS = ['symbol', 't', 'o', 'h', 'l', 'c', 'v', 'n', 'vw']
MIN_GAP_S = 0.25
BUDGET_S = 540


def log(m):
    """Timestamped, flushed progress line."""
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def headers():
    """Alpaca auth headers from .env (HOD keys first, generic keys as a logged fallback)."""
    load_dotenv('/home/ec2-user/onemil/.env')
    k, s = os.getenv('ALPACA_HOD_API_KEY'), os.getenv('ALPACA_HOD_SECRET_KEY') or os.getenv('ALPACA_HOD_API_SECRET')
    if not (k and s):
        log('WARNING: ALPACA_HOD keys missing - falling back to ALPACA_API_KEY/SECRET')
        k = os.getenv('ALPACA_API_KEY')
        s = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_API_SECRET')
    if not (k and s):
        log('ERROR: no Alpaca keys in .env'); sys.exit(2)
    return {'APCA-API-KEY-ID': k, 'APCA-API-SECRET-KEY': s}


_last = [0.0]


def get(h, params):
    """One rate-limited GET with retry on 429/5xx; returns the JSON body."""
    for attempt in range(6):
        wait = MIN_GAP_S - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        r = requests.get(URL, headers=h, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        log(f'WARNING: HTTP {r.status_code} ({r.text[:120]}) - retry {attempt + 1}')
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f'Alpaca request failed after retries: {params.get("start")}')


def fetch_day(h, day, syms):
    """All pre-market bars 04:00-09:29 ET of one day for the given symbols (chunks of 50, paginated)."""
    start = pd.Timestamp(f'{day} 04:00', tz='America/New_York').tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    end = pd.Timestamp(f'{day} 09:29:59', tz='America/New_York').tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    rows, nreq = [], 0
    for k in range(0, len(syms), 50):
        p = {'symbols': ','.join(syms[k:k + 50]), 'timeframe': '1Min', 'start': start, 'end': end, 'feed': 'sip',
             'adjustment': 'raw', 'limit': 10000}
        while True:
            js = get(h, p); nreq += 1
            for s, bl in (js.get('bars') or {}).items():
                rows += [(s, b['t'], b['o'], b['h'], b['l'], b['c'], b['v'], b.get('n'), b.get('vw')) for b in bl]
            tok = js.get('next_page_token')
            if not tok:
                break
            p['page_token'] = tok
    return pd.DataFrame(rows, columns=COLS), nreq


def main():
    """Fetch every TRAIN/VAL day not yet checkpointed; stop cleanly at the time budget (rerun resumes)."""
    os.makedirs(OUT, exist_ok=True)
    pop = pd.read_csv(f'{HE}/cell_1428_population.csv', usecols=['day', 'symbol', 'split'],
                      dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    pop = pop[pop.split.isin(['TRAIN', 'VAL'])]
    days = sorted(pop.day.unique())
    h = headers()
    t0, done = time.time(), 0
    for d in days:
        fn = f'{OUT}/{d}.parquet'
        if os.path.exists(fn):
            done += 1; continue
        syms = sorted(set(pop.symbol[pop.day == d]))
        df, nreq = fetch_day(h, d, syms)
        df.to_parquet(fn + '.tmp', index=False)
        os.replace(fn + '.tmp', fn)
        done += 1
        log(f'{d}: {df.symbol.nunique()}/{len(syms)} symbols with pre-market bars, {len(df):,} bars, {nreq} req; '
            f'{done}/{len(days)} days, {time.time() - t0:.0f}s')
        if time.time() - t0 > BUDGET_S:
            log('time budget reached - rerun to resume'); return
    log(f'ALL DONE: {done}/{len(days)} days checkpointed')


if __name__ == '__main__':
    main()
