"""Cell 1,747 fetch: Alpaca cash-dividend corporate actions (2016->2026) + split-adjusted daily bars for payers.
Stages: events | bars.  Resumable, atomic.  Reuses helpers of 1746_fetch.py.  Keys never printed."""
import importlib.util, os, sys, time
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
spec = importlib.util.spec_from_file_location('f46', os.path.join(HERE, '1746_fetch.py'))
f46 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f46)
CACHE = os.path.join(HERE, '1747_cache'); os.makedirs(CACHE, exist_ok=True)
CA_URL = 'https://data.alpaca.markets/v1/corporate-actions'
log = f46.log


def ca_page(params):
    """One corporate-actions GET with back-off."""
    delay = 2.0
    for _ in range(8):
        time.sleep(0.4)
        r = requests.get(CA_URL, headers=f46.headers(), params=params, timeout=90)
        if r.status_code == 429 or r.status_code >= 500:
            log(f'WARNING CA HTTP {r.status_code}; backoff {delay}s'); time.sleep(delay); delay = min(delay * 2, 60); continue
        if r.status_code != 200:
            raise RuntimeError(f'HTTP {r.status_code} {r.text[:200]}')
        return r.json()
    raise RuntimeError('CA gave up')


def stage_events():
    """Monthly windows (filter is on process_date) 2015-10 -> 2026-11, paged; dedup on id."""
    rows = []
    for m in pd.period_range('2015-10', '2026-11', freq='M'):
        s, e = m.start_time.strftime('%Y-%m-%d'), m.end_time.strftime('%Y-%m-%d')
        tok, n = None, 0
        while True:
            p = {'types': 'cash_dividend', 'start': s, 'end': e, 'limit': 1000}
            if tok:
                p['page_token'] = tok
            j = ca_page(p)
            cd = (j.get('corporate_actions') or {}).get('cash_dividends', [])
            rows += cd; n += len(cd)
            tok = j.get('next_page_token')
            if not tok:
                break
        log(f'events {m}: {n}')
    d = pd.DataFrame(rows).drop_duplicates('id')
    d.to_parquet(os.path.join(HERE, '1747_events_raw.parquet'))
    log(f'raw events {len(d)}; ex_date {d.ex_date.min()} .. {d.ex_date.max()}; columns {list(d.columns)}')


def stage_bars(adj='split'):
    """Split-adjusted (NOT dividend-adjusted) daily bars for eligible payers, batched, resumable."""
    ev = pd.read_parquet(os.path.join(HERE, '1747_events_raw.parquet'))
    cands = f46.candidate_names()
    syms = sorted(set(ev.symbol) & set(cands))
    log(f'payers eligible by name: {len(syms)}')
    bdir = os.path.join(CACHE, 'bars_parts'); os.makedirs(bdir, exist_ok=True)
    B = 100
    for i in range(0, len(syms), B):
        part = os.path.join(bdir, f'{adj}_{i // B:03d}.parquet')
        if os.path.exists(part):
            continue
        df = f46.get_bars(syms[i:i + B], adj, f46.BARS_START, f46.BARS_END)
        df.to_parquet(part + '.tmp'); os.replace(part + '.tmp', part)
        log(f'bars batch {i // B + 1}/{(len(syms) + B - 1) // B} rows {len(df)} got {df.symbol.nunique()}/{len(syms[i:i + B])}')
    d = pd.concat([pd.read_parquet(os.path.join(bdir, f)) for f in sorted(os.listdir(bdir)) if f.startswith(adj + '_') and f.endswith('.parquet')])
    d.to_parquet(os.path.join(CACHE, f'bars_{adj}.parquet'))
    log(f'bars {adj}: {len(d)} rows, {d.symbol.nunique()} symbols of {len(syms)} (LOST {len(syms) - d.symbol.nunique()})')


if __name__ == '__main__':
    for st in sys.argv[1:]:
        {'events': stage_events, 'bars': stage_bars, 'raw': lambda: stage_bars('raw')}[st]()
