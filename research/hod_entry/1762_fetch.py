"""Cell 1,762 fetch - minute bars (DB where present, else Alpaca SIP raw) and daily bars for the honest population.

PREREG: research/hod_entry/PREREG_1762.md. Population = cell_1428_population.csv (PIT gapper symbol-days, TRAIN+VAL;
TEST days are never fetched). Steps (all resumable):
  1. presence: per day, which population symbols already have >= 30 RTH minute bars (first bar <= 09:35) in
     bars_sip.db (read-only) -> 1762_cache/presence.parquet.  SPY is checked too (feature spy_ret).
  2. minute fetch: the rest (+ SPY when the DB lacks it) from Alpaca, <= 50 symbols/request, paged,
     <= 5 requests/s, one checkpoint file per day: 1762_cache/bars/<day>.parquet.
  3. daily fetch: symbols whose daily features are not fully covered by the 1746/1748 caches get raw + adjusted
     daily history (2025-03-01..2026-06-01) -> 1762_cache/daily.parquet.
Window offsets use the real ET offset of each day (DST aware).  Keys come from .env and are never printed.

Usage: ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/hod_entry/1762_fetch.py
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'ARROW_IO_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, time, sqlite3, resource, json
from zoneinfo import ZoneInfo
from datetime import datetime
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import requests
from dotenv import load_dotenv

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from research.hod_exit_lab import walker  # noqa: E402

HE = f'{ROOT}/research/hod_entry'
CACHE = f'{HE}/1762_cache'
BARS = f'{CACHE}/bars'
LOG = f'{HE}/1762_fetch.log'
DB = f'file:{ROOT}/research/bf_zero/bars_sip.db?mode=ro'
SS = f'{ROOT}/research/signal_search'
URL = 'https://data.alpaca.markets/v2/stocks/bars'
ET = ZoneInfo('America/New_York')
BATCH, GAP_S = 50, 0.22
MIN_BARS = 30


def log(msg):
    """Print and append a timestamped line with peak RSS (MB) to 1762_fetch.log."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    line = f'[{time.strftime("%H:%M:%S")}] rss={rss:.0f}MB {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def headers():
    """Alpaca auth headers from .env (HOD keys first, then the main account); never logged."""
    load_dotenv(f'{ROOT}/.env')
    k = os.getenv('ALPACA_HOD_API_KEY') or os.getenv('ALPACA_API_KEY')
    s = os.getenv('ALPACA_HOD_API_SECRET') or os.getenv('ALPACA_API_SECRET')
    if not k or not s:
        raise SystemExit('ERROR: no Alpaca data keys in .env')
    return {'APCA-API-KEY-ID': k, 'APCA-API-SECRET-KEY': s}


def get(hdr, params):
    """One rate-limited GET with retry/backoff on 429 / 5xx; errors are logged and re-raised at the end."""
    for attempt in range(6):
        time.sleep(GAP_S)
        r = requests.get(URL, headers=hdr, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        log(f'WARNING http {r.status_code} attempt {attempt}: {r.text[:120]}')
        time.sleep(2 ** attempt)
    raise RuntimeError(f'Alpaca request failed: {params.get("symbols", "")[:40]}')


def paged(hdr, params):
    """All pages of a bars request -> list of (symbol, bar dict)."""
    out, tok = [], None
    while True:
        p = dict(params, **({'page_token': tok} if tok else {}))
        j = get(hdr, p)
        for sym, bars in (j.get('bars') or {}).items():
            out += [(sym, b) for b in bars]
        tok = j.get('next_page_token')
        if not tok:
            return out


def to_minutes(t):
    """ISO-8601 UTC strings -> ET minute of day."""
    ts = pd.to_datetime(t, utc=True).dt.tz_convert('America/New_York')
    return (ts.dt.hour * 60 + ts.dt.minute).to_numpy()


def presence(pop):
    """Per (day, symbol incl. SPY): RTH bar count and first-bar minute in bars_sip.db -> presence.parquet."""
    path = f'{CACHE}/presence.parquet'
    if os.path.exists(path):
        return pd.read_parquet(path)
    walker.wait_for_db_window()
    con = sqlite3.connect(DB, uri=True)
    rows = []
    for k, (d, g) in enumerate(pop.groupby('day')):
        syms = list(g.symbol) + ['SPY']
        q = f"select symbol,t,o,h,l,c,v from bars where day=? and symbol in ({','.join('?' * len(syms))})"
        b = pd.read_sql_query(q, con, params=[d] + syms)
        if len(b):
            b['m'] = to_minutes(b.t)
            b = b[(b.m >= 570) & (b.m < 960)]
            s = b.groupby('symbol').m.agg(n_rth='size', first_m='min')
        else:
            s = pd.DataFrame(columns=['n_rth', 'first_m'])
        for sym in syms:
            n, f0 = (int(s.n_rth[sym]), int(s.first_m[sym])) if sym in s.index else (0, -1)
            rows.append((d, sym, n, f0))
        if k % 40 == 0:
            log(f'presence {k + 1} days')
    con.close()
    pr = pd.DataFrame(rows, columns=['day', 'symbol', 'n_rth', 'first_m'])
    pr['in_db'] = (pr.n_rth >= MIN_BARS) & (pr.first_m <= 575)
    pr.to_parquet(path, index=False)
    return pr


def fetch_minutes(pop, pr, hdr):
    """Fetch every population symbol-day (and SPY) missing from the DB; one checkpoint parquet per day."""
    os.makedirs(BARS, exist_ok=True)
    have = set(zip(pr.day[pr.in_db], pr.symbol[pr.in_db]))
    days = sorted(pop.day.unique())
    t0 = time.time()
    for k, d in enumerate(days):
        path = f'{BARS}/{d}.parquet'
        if os.path.exists(path):
            continue
        need = [s for s in pop.symbol[pop.day == d] if (d, s) not in have]
        if ('%s' % d, 'SPY') not in have:
            need.append('SPY')
        dt = datetime.fromisoformat(d)
        off = datetime(dt.year, dt.month, dt.day, 9, 30, tzinfo=ET).strftime('%z')
        off = off[:3] + ':' + off[3:]
        rows = []
        for i in range(0, len(need), BATCH):
            prm = dict(symbols=','.join(need[i:i + BATCH]), timeframe='1Min', start=f'{d}T09:30:00{off}',
                       end=f'{d}T16:00:00{off}', feed='sip', adjustment='raw', limit=10000)
            rows += [dict(symbol=s, t=b['t'], o=b['o'], h=b['h'], l=b['l'], c=b['c'], v=b['v'])
                     for s, b in paged(hdr, prm)]
        df = pd.DataFrame(rows, columns=['symbol', 't', 'o', 'h', 'l', 'c', 'v'])
        df.to_parquet(path + '.tmp', index=False)
        os.replace(path + '.tmp', path)
        log(f'minutes {d}: asked {len(need)} symbols, got {df.symbol.nunique()} / {len(df):,} bars '
            f'({k + 1}/{len(days)}, {time.time() - t0:.0f}s)')
        if time.time() - t0 > 540:
            log('time budget reached - rerun to resume')
            return False
    return True


def daily_gaps(pop):
    """Symbols whose (symbol, day) daily features are not covered by the 1746/1748 caches (>= 50 prior rows in
    100 calendar days and the day's own row)."""
    flt = ds.field('symbol').isin(sorted(pop.symbol.unique())) & (ds.field('date') >= '2025-03-01')
    fr = []
    for c in ('1748_cache', '1746_cache'):
        fr.append(ds.dataset(f'{SS}/{c}/bars_raw.parquet').to_table(columns=['symbol', 'date'], filter=flt,
                                                                    use_threads=False).to_pandas())
    d = pd.concat(fr).drop_duplicates()
    dates = {s: pd.to_datetime(g.date).sort_values().to_numpy() for s, g in d.groupby('symbol')}
    bad = set()
    for r in pop.itertuples(index=False):
        a = dates.get(r.symbol)
        t = np.datetime64(r.day)
        if a is None:
            bad.add(r.symbol); continue
        n_prior = int(((a < t) & (a >= t - np.timedelta64(100, 'D'))).sum())
        if n_prior < 50 or not (a == t).any():
            bad.add(r.symbol)
    return sorted(bad)


def fetch_daily(syms, hdr):
    """Raw + split/dividend-adjusted daily bars for the uncovered symbols -> daily.parquet (checkpointed per batch)."""
    path = f'{CACHE}/daily.parquet'
    if os.path.exists(path):
        log('daily.parquet exists - skipped')
        return
    parts = []
    for i in range(0, len(syms), BATCH):
        sl = ','.join(syms[i:i + BATCH])
        got = {}
        for adj in ('raw', 'all'):
            prm = dict(symbols=sl, timeframe='1Day', start='2025-03-01T00:00:00Z', end='2026-06-02T00:00:00Z',
                       feed='sip', adjustment=adj, limit=10000)
            got[adj] = pd.DataFrame([dict(symbol=s, date=b['t'][:10], h=b['h'], l=b['l'], c=b['c'], v=b['v'])
                                     for s, b in paged(hdr, prm)])
        raw = got['raw']
        if len(raw):
            adjc = got['all'][['symbol', 'date', 'c']].rename(columns={'c': 'cadj'}) if len(got['all']) else None
            raw = raw.merge(adjc, on=['symbol', 'date'], how='left') if adjc is not None else raw.assign(cadj=np.nan)
            parts.append(raw)
        log(f'daily batch {i // BATCH + 1}/{(len(syms) + BATCH - 1) // BATCH}')
    pd.concat(parts).to_parquet(path + '.tmp', index=False)
    os.replace(path + '.tmp', path)


def main():
    """Presence -> minute fetch (resumable) -> daily fetch; coverage summary to coverage_fetch.json."""
    os.makedirs(BARS, exist_ok=True)
    pop = pd.read_csv(f'{HE}/cell_1428_population.csv', usecols=['day', 'symbol', 'split'],
                      dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    pop = pop[pop.split.isin(['TRAIN', 'VAL'])]
    log(f'population {len(pop):,} symbol-days, {pop.day.nunique()} days, splits {pop.split.value_counts().to_dict()}')
    hdr = headers()
    pr = presence(pop)
    pp = pr[pr.symbol != 'SPY']
    log(f'in bars_sip.db: {int(pp.in_db.sum()):,}/{len(pp):,} = {pp.in_db.mean():.1%}')
    if not fetch_minutes(pop, pr, hdr):
        return
    gaps = daily_gaps(pop)
    log(f'daily: {len(gaps)} of {pop.symbol.nunique()} symbols need a fetch')
    fetch_daily(gaps, hdr)
    log('fetch done')


if __name__ == '__main__':
    main()
