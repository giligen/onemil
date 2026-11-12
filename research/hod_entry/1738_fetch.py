"""PREREG 1,738 Stage 0 fetch: contract selection + 1-min option trade bars for a signal sample, cached per request.

READ-ONLY endpoints only (contracts listing, historical option bars). Cache: research/hod_entry/1738_cache/*.json.
Usage: nice -n 19 python3 1738_fetch.py [--n-per-split 200]
"""
import argparse, hashlib, json, os, sys, time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd, requests
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '1738_cache')
LAB = os.path.join(HERE, '..', 'hod_exit_lab')
CONTRACTS_URL = 'https://paper-api.alpaca.markets/v2/options/contracts'
BARS_URL = 'https://data.alpaca.markets/v1beta1/options/bars'
MIN_INTERVAL = 0.33  # <= ~180 requests/min
ET = ZoneInfo('America/New_York')
_last = [0.0]
STATS = {'net': 0, 'cached': 0, 'r429': 0}


def headers():
    """Auth headers for the HOD key pair (never printed)."""
    load_dotenv('/home/ec2-user/onemil/.env')
    return {'APCA-API-KEY-ID': os.environ['ALPACA_HOD_API_KEY'], 'APCA-API-SECRET-KEY': os.environ['ALPACA_HOD_API_SECRET']}


def cached_get(url, params):
    """GET with on-disk JSON cache, rate limit and 429/5xx back-off. Failures raise (no silent fallback)."""
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    path = os.path.join(CACHE, key + '.json')
    if os.path.exists(path):
        STATS['cached'] += 1
        return json.load(open(path))['resp']
    delay = 2.0
    for attempt in range(8):
        wait = MIN_INTERVAL - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        STATS['net'] += 1
        r = requests.get(url, headers=headers(), params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            STATS['r429'] += 1
            print(f'WARNING HTTP {r.status_code} on {url} {params}; backoff {delay}s', flush=True)
            time.sleep(delay); delay = min(delay * 2, 60); continue
        if r.status_code == 422 and 'invalid underlying symbols' in r.text:
            print(f'WARNING no option chain for {params.get("underlying_symbols")} (422 invalid underlying): counted as has_contract=False', flush=True)
            resp = {'option_contracts': [], 'next_page_token': None, 'invalid_underlying': r.text[:200]}
            json.dump({'url': url, 'params': params, 'resp': resp}, open(path, 'w'))
            return resp
        if r.status_code != 200:
            raise RuntimeError(f'HTTP {r.status_code} {url} {params} {r.text[:200]}')
        resp = r.json()
        tmp = path + '.tmp'
        json.dump({'url': url, 'params': params, 'resp': resp}, open(tmp, 'w'))
        os.replace(tmp, path)
        return resp
    raise RuntimeError(f'gave up after retries: {url} {params}')


def paged(url, params, field):
    """Collect all pages of a list/dict field."""
    out, tok = None, None
    while True:
        p = dict(params); 
        if tok: p['page_token'] = tok
        j = cached_get(url, p)
        part = j.get(field)
        if isinstance(part, dict):
            out = out or {}
            for k, v in part.items(): out.setdefault(k, []).extend(v)
        else:
            out = (out or []) + (part or [])
        tok = j.get('next_page_token')
        if not tok: return out


def pick_contract(sym, day, entry):
    """Nearest expiry with DTE 2..14 having a call strike >= entry; lowest such strike. Returns contract dict or None."""
    d = date.fromisoformat(day)
    cands = []
    for status in ('inactive', 'active'):
        if status == 'active' and d + timedelta(days=14) < date.today() - timedelta(days=1):
            continue
        cands += paged(CONTRACTS_URL, dict(underlying_symbols=sym, expiration_date_gte=(d + timedelta(days=2)).isoformat(),
                       expiration_date_lte=(d + timedelta(days=14)).isoformat(), type='call', status=status,
                       strike_price_gte=repr(float(entry)), limit=10000), 'option_contracts')
    cands = [c for c in cands if c.get('underlying_symbol') == sym and c.get('root_symbol') == sym and float(c['strike_price']) >= entry]
    if not cands: return None
    return min(cands, key=lambda c: (c['expiration_date'], float(c['strike_price'])))


def utc_iso(day, minute_et):
    """ET minute-of-day on `day` -> RFC3339 UTC."""
    dt = datetime.fromisoformat(day).replace(tzinfo=ET) + timedelta(minutes=int(minute_et))
    return dt.astimezone(ZoneInfo('UTC')).strftime('%Y-%m-%dT%H:%M:%SZ')


def fetch_bars(contract, day, entry_m, exit_m):
    """1-min trade bars for `contract` covering entry minute .. exit+15 min."""
    return paged(BARS_URL, dict(symbols=contract, timeframe='1Min', start=utc_iso(day, entry_m), end=utc_iso(day, exit_m + 16),
                                limit=10000), 'bars').get(contract, [])


def sample(n_per_split):
    """Random Stage-0 sample (seed 1738) joined to B0 exit minute and net_R."""
    s = pd.read_parquet(os.path.join(LAB, 'signals.parquet'))
    b = pd.read_csv(os.path.join(LAB, 'b0_trades.csv'))[['day', 'symbol', 'entry_m', 'exit_m', 'exit_price', 'why', 'net_R']]
    m = s.merge(b, on=['day', 'symbol', 'entry_m'], how='inner', validate='1:1')
    assert len(m) == len(s), (len(m), len(s))
    parts = [m[m.split == sp].sample(n_per_split, random_state=1738) for sp in ('TRAIN', 'VAL')]
    return pd.concat(parts).reset_index(drop=True)


def run(df):
    """Fetch contract + bars for each row; return rows with contract symbol and bar list."""
    rows = []
    for i, r in df.iterrows():
        c = pick_contract(r.symbol, r.day, r.entry)
        bars = fetch_bars(c['symbol'], r.day, r.entry_m, r.exit_m) if c else []
        rows.append(dict(contract=c['symbol'] if c else None, n_bars=len(bars)))
        if i % 20 == 0:
            print(f'{i}/{len(df)} {r.day} {r.symbol} -> {rows[-1]} net={STATS["net"]} cached={STATS["cached"]} r429={STATS["r429"]}', flush=True)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--n-per-split', type=int, default=200)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    df = sample(a.n_per_split)
    print('sample', len(df), df.split.value_counts().to_dict(), flush=True)
    run(df)
    print('DONE', STATS, flush=True)
