"""Fetch FILL activities, orders and 1D portfolio history for the HOD and ORB paper accounts (read-only GETs).

Writes raw JSON under research/book_review_20261111/raw/. Never prints keys."""
import json, os, sys, time, urllib.request, urllib.parse
from pathlib import Path

ROOT = Path('/home/ec2-user/onemil'); OUT = ROOT / 'research/book_review_20261111/raw'
BASE = 'https://paper-api.alpaca.markets'


def load_env():
    """Parse .env into a dict without exporting or printing values."""
    env = {}
    for line in (ROOT / '.env').read_text().splitlines():
        if '=' in line and not line.strip().startswith('#'):
            k, v = line.split('=', 1); env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def get(path, key, sec, params=None):
    """GET an Alpaca endpoint and return parsed JSON."""
    url = BASE + path + ('?' + urllib.parse.urlencode(params) if params else '')
    req = urllib.request.Request(url, headers={'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': sec})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def activities(key, sec, after):
    """All FILL activities after `after`, paginated by page_token."""
    out, tok = [], None
    while True:
        p = {'activity_types': 'FILL', 'after': after, 'direction': 'asc', 'page_size': 100}
        if tok: p['page_token'] = tok
        page = get('/v2/account/activities', key, sec, p)
        if not page: break
        out += page; tok = page[-1]['id']
        print(f'  activities page {len(page)} total {len(out)}', flush=True)
        if len(page) < 100: break
    return out


def orders(key, sec, after):
    """All orders (any status) after `after`, paginated by submitted time asc."""
    out, a = [], after
    while True:
        page = get('/v2/orders', key, sec, {'status': 'all', 'after': a, 'direction': 'asc', 'limit': 500, 'nested': 'true'})
        if not page: break
        out += page; a = page[-1]['submitted_at'] or page[-1]['created_at']
        print(f'  orders page {len(page)} total {len(out)}', flush=True)
        if len(page) < 500: break
    seen = {}
    for o in out: seen[o['id']] = o
    return list(seen.values())


def main():
    """Fetch both books' account data."""
    env = load_env()
    for book in ('HOD', 'ORB'):
        key, sec = env.get(f'ALPACA_{book}_API_KEY'), env.get(f'ALPACA_{book}_SECRET_KEY') or env.get(f'ALPACA_{book}_API_SECRET') or env.get(f'ALPACA_{book}_SECRET')
        if not key or not sec:
            print(f'ERROR: missing {book} key/secret in .env', flush=True); sys.exit(1)
        print(f'{book}: fetching', flush=True)
        acts = activities(key, sec, '2026-09-29T00:00:00Z')
        ords = orders(key, sec, '2026-09-29T00:00:00Z')
        ph = get('/v2/account/portfolio/history', key, sec, {'timeframe': '1D', 'date_start': '2026-09-25', 'date_end': '2026-11-10'})
        acct = get('/v2/account', key, sec)
        acct = {k: acct.get(k) for k in ('equity', 'last_equity', 'cash', 'long_market_value', 'short_market_value')}
        pos = get('/v2/positions', key, sec)
        json.dump({'activities': acts, 'orders': ords, 'history': ph, 'account': acct, 'positions': pos},
                  open(OUT / f'{book.lower()}_account.json', 'w'))
        print(f'{book}: {len(acts)} fills, {len(ords)} orders, {len(pos)} positions, history pts {len(ph.get("equity") or [])}', flush=True)


if __name__ == '__main__':
    main()
