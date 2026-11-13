"""Cell 1,769 fetch (PREREG_1769): build the 1,754 gap>=10 % event table (TRAIN 2016-20 + VAL 2021-23, TEST sealed by
1754's SEAL) with RAW open >= $5, then fetch Alpaca SIP raw 1-min bars 09:30-10:31 ET per event day, checkpointed per day.
Usage: cd /home/ec2-user/onemil && ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/signal_search/1769_fetch.py   (resumes)
"""
import os, sys, time, importlib.util
from zoneinfo import ZoneInfo
from datetime import datetime
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '1769_cache')
EVENTS = os.path.join(HERE, '1769_events.parquet')
URL = 'https://data.alpaca.markets/v2/stocks/bars'
NY = ZoneInfo('America/New_York')
MIN_RAW_OPEN, GAP_MIN, BUDGET_S, MIN_DT = 5.0, 0.10, 540, 0.21


def log(m):
    print(m, flush=True)


def load_1754():
    """Import 1754_score.py (module name starts with a digit)."""
    sp = importlib.util.spec_from_file_location('s1754', os.path.join(HERE, '1754_score.py'))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def build_events():
    """1754 events (gap>=10 %, attention, rank<=3000, scale-checked, TEST sealed) + RAW open_t >= $5 flag; write parquet."""
    S = load_1754()
    dates, syms, M = S.load_matrices()
    E, days = S.build_events(dates, syms, M)
    di, si = pd.Index(dates).get_indexer(E.date), pd.Index(syms).get_indexer(E.symbol)
    E['open_raw'] = M['Oraw'][di, si]; E['split'] = np.where(E.date < '2021-01-01', 'TRAIN', 'VAL')
    del M
    assert E.date.max() < S.SEAL, 'TEST leaked'
    E = E[E.gap >= GAP_MIN].copy()
    E['raw_ok'] = E.open_raw >= MIN_RAW_OPEN
    E.to_parquet(EVENTS)
    d = days[['date']].copy(); d['year'] = d.date.str[:4]; d.to_parquet(os.path.join(HERE, '1769_days.parquet'))
    log('events gap>=10 % per year (all / raw open>=5): ' + str(E.groupby('year').agg(n=('gap', 'size'), n5=('raw_ok', 'sum')).to_dict('index')))
    return E


def keys():
    """Alpaca headers from .env (never printed)."""
    load_dotenv('/home/ec2-user/onemil/.env')
    k = os.getenv('ALPACA_HOD_API_KEY') or os.getenv('ALPACA_API_KEY'); s = os.getenv('ALPACA_HOD_API_SECRET') or os.getenv('ALPACA_API_SECRET')
    if not k or not s:
        raise SystemExit('ERROR: Alpaca keys missing in .env')
    return {'APCA-API-KEY-ID': k, 'APCA-API-SECRET-KEY': s}


def rfc(day, hh, mm):
    """RFC-3339 timestamp for day hh:mm America/New_York with the correct DST offset."""
    return datetime.fromisoformat(f'{day}T{hh:02d}:{mm:02d}:00').replace(tzinfo=NY).isoformat()


def get(sess, params, last):
    """One GET with <=5 req/s pacing and retry on 429/5xx; returns json or raises."""
    for attempt in range(6):
        w = MIN_DT - (time.time() - last[0])
        if w > 0:
            time.sleep(w)
        last[0] = time.time()
        r = sess.get(URL, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        log(f'WARNING: HTTP {r.status_code} attempt {attempt}: {r.text[:200]}')
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 + 3 * attempt); continue
        raise RuntimeError(f'HTTP {r.status_code}: {r.text[:200]}')
    raise RuntimeError('retries exhausted')


def fetch_day(sess, day, symbols, last):
    """All 09:30-10:31 ET raw SIP minute bars for `symbols` on `day` -> DataFrame (symbol,t,o,h,l,c,v)."""
    rows = []
    for i in range(0, len(symbols), 50):
        p = dict(symbols=','.join(symbols[i:i + 50]), timeframe='1Min', start=rfc(day, 9, 30), end=rfc(day, 10, 31),
                 feed='sip', adjustment='raw', limit=10000)
        while True:
            j = get(sess, p, last)
            for s, bars in (j.get('bars') or {}).items():
                rows += [(s, b['t'], b['o'], b['h'], b['l'], b['c'], b['v']) for b in bars]
            if not j.get('next_page_token'):
                break
            p['page_token'] = j['next_page_token']
    return pd.DataFrame(rows, columns=['symbol', 't', 'o', 'h', 'l', 'c', 'v'])


def main():
    """Build events if absent, then fetch every missing event day until done or the time budget is spent."""
    t0 = time.time()
    E = pd.read_parquet(EVENTS) if os.path.exists(EVENTS) else build_events()
    P = E[E.raw_ok]
    days = sorted(P.date.unique())
    todo = [d for d in days if not os.path.exists(os.path.join(CACHE, f'{d}.parquet'))]
    log(f'event-days {len(days)}, events {len(P)}, to fetch {len(todo)}')
    sess = requests.Session(); sess.headers.update(keys()); last = [0.0]
    for k, d in enumerate(todo):
        if time.time() - t0 > BUDGET_S:
            log(f'budget spent at day {k}/{len(todo)}; rerun to resume'); return
        syms = sorted(P[P.date == d].symbol.unique())
        df = fetch_day(sess, d, syms, last)
        tmp = os.path.join(CACHE, f'{d}.parquet.tmp'); df.to_parquet(tmp); os.replace(tmp, os.path.join(CACHE, f'{d}.parquet'))
        if k % 50 == 0:
            log(f'{d}: {len(syms)} syms -> {df.symbol.nunique()} with bars, {len(df)} rows ({k + 1}/{len(todo)})')
    log('fetch complete')


if __name__ == '__main__':
    main()
