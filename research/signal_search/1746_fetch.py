"""PREREG 1,746 fetch: FINRA Reg SHO daily short-volume files (2016-01-04 .. 2026-11-11) + Alpaca daily bars.

Stages (each resumable, atomic writes, one writer):
  days   : SPY 1Day bars -> trading-day list (1746_cache/trading_days.csv)
  finra  : CNMSshvolYYYYMMDD.txt per trading day -> 1746_cache/finra/YYYYMMDD.txt (404 -> .missing marker).
           Coverage = files / trading days; < 95 % -> VOID (logged ERROR, exit code 2 for the later stages).
  svr    : parse every file, keep eligible common-stock candidates (assets csv + is_bf_eligible + fund-name screen,
           median FINRA TotalVolume >= 100k) -> 1746_svr.parquet
  bars   : Alpaca 1Day bars for the candidate symbols, adjustment=raw (dollar volume, price filter) and
           adjustment=all (returns), 2015-12-01 .. 2026-11-12 -> 1746_cache/bars_{raw,all}.parquet
LOST counts are logged per stage.  Keys come from .env via load_dotenv and are never printed.
Usage: nice -n 19 python3 1746_fetch.py [days|finra|svr|bars|all]
"""
import csv, os, re, shutil, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
CACHE = os.path.join(HERE, '1746_cache')
FINRA_DIR = os.path.join(CACHE, 'finra')
BARS_URL = 'https://data.alpaca.markets/v2/stocks/bars'
FINRA_URL = 'https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt'
START, END = '2016-01-04', '2026-11-11'
BARS_START, BARS_END = '2015-12-01', '2026-11-12'
MIN_COVERAGE = 0.95
MIN_FREE_GB = 5.0
MIN_INTERVAL = 0.34           # <= 176 Alpaca requests / minute
MIN_MEDIAN_TOTVOL = 100_000
TEST_TICKER = re.compile(r'^Z[A-Z]ZZT$')
FUND_NAME = re.compile(r'\bETF\b|\bETN\b|\bETFS\b|EXCHANGE[- ]TRADED|\bFUND\b|ISHARES|SPDR|VANGUARD|INVESCO|PROSHARES|'
                       r'GLOBAL X|VANECK|ROUNDHILL|KRANESHARES|SCHWAB U\.?S|ARK |GOLD TRUST|SILVER TRUST|BITCOIN TRUST|'
                       r'ETHEREUM TRUST|PHYSICAL|MINISHARES|GOLD SHARES|QQQ TRUST|SERIES 1\b|INDEX|SELECT SECTOR|'
                       r'GRAYSCALE|WISDOMTREE|DIREXION|T-BILL|TREASURY|PORTFOLIO', re.I)
_last = [0.0]
_lock = threading.Lock()


def log(msg):
    """Flushed progress line."""
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def atomic_write(path, data, mode='wb'):
    """Write via tmp + rename."""
    tmp = path + '.tmp'
    with open(tmp, mode) as fh:
        fh.write(data)
    os.replace(tmp, path)


def headers():
    """Alpaca auth headers (never printed)."""
    load_dotenv(os.path.join(ROOT, '.env'))
    return {'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'], 'APCA-API-SECRET-KEY': os.environ['ALPACA_API_SECRET']}


def alpaca_get(params):
    """One Alpaca GET with global token spacing and 429/5xx back-off; other failures raise."""
    delay = 2.0
    for _ in range(8):
        with _lock:
            wait = MIN_INTERVAL - (time.time() - _last[0])
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        r = requests.get(BARS_URL, headers=headers(), params=params, timeout=90)
        if r.status_code == 429 or r.status_code >= 500:
            log(f'WARNING Alpaca HTTP {r.status_code}; backoff {delay}s')
            time.sleep(delay); delay = min(delay * 2, 60); continue
        if r.status_code != 200:
            raise RuntimeError(f'HTTP {r.status_code} {r.text[:200]}')
        return r.json()
    raise RuntimeError('Alpaca gave up after retries')


def get_bars(symbols, adjustment, start, end):
    """All 1Day bars for `symbols` (paged); returns DataFrame symbol,date,o,h,l,c,v."""
    rows, token = [], None
    while True:
        p = {'symbols': ','.join(symbols), 'timeframe': '1Day', 'start': start, 'end': end, 'adjustment': adjustment,
             'feed': 'sip', 'limit': 10000}
        if token:
            p['page_token'] = token
        j = alpaca_get(p)
        for s, bars in (j.get('bars') or {}).items():
            for b in bars:
                rows.append((s, b['t'][:10], b['o'], b['h'], b['l'], b['c'], b['v']))
        token = j.get('next_page_token')
        if not token:
            break
    return pd.DataFrame(rows, columns=['symbol', 'date', 'o', 'h', 'l', 'c', 'v'])


def stage_days():
    """SPY trading days START..END."""
    path = os.path.join(CACHE, 'trading_days.csv')
    df = get_bars(['SPY'], 'raw', START, '2026-11-13')
    df = df[(df.date >= START) & (df.date <= END)]
    atomic_write(path, df[['date']].to_csv(index=False), 'w')
    log(f'days: {len(df)} SPY trading days {df.date.min()}..{df.date.max()}')


def fetch_finra_one(day):
    """Fetch one FINRA file; returns 'ok' | 'missing' | 'error'."""
    d = day.replace('-', '')
    out, miss = os.path.join(FINRA_DIR, d + '.txt'), os.path.join(FINRA_DIR, d + '.missing')
    if os.path.exists(out):
        return 'ok'
    if os.path.exists(miss):
        return 'missing'
    for k in range(5):
        try:
            r = requests.get(FINRA_URL.format(d=d), timeout=60)
        except requests.RequestException as e:
            log(f'WARNING finra {d} {type(e).__name__}; retry {k}'); time.sleep(2 * (k + 1)); continue
        if r.status_code == 200 and r.content.startswith(b'Date|Symbol'):
            atomic_write(out, r.content); return 'ok'
        if r.status_code in (403, 404):
            atomic_write(miss, b''); return 'missing'
        log(f'WARNING finra {d} HTTP {r.status_code}; retry {k}'); time.sleep(2 * (k + 1))
    log(f'ERROR finra {d} failed after retries')
    return 'error'


def stage_finra():
    """Download every trading day's file; coverage gate."""
    days = pd.read_csv(os.path.join(CACHE, 'trading_days.csv')).date.tolist()
    res = {}
    with ThreadPoolExecutor(4) as ex:
        for i, (d, st) in enumerate(zip(days, ex.map(fetch_finra_one, days))):
            res[d] = st
            if i % 250 == 0:
                log(f'finra {i}/{len(days)} {d} {st}')
    ok = sum(v == 'ok' for v in res.values())
    lost = [d for d, v in res.items() if v != 'ok']
    cov = ok / len(days)
    log(f'finra: {ok}/{len(days)} trading days have a file, coverage {cov:.4f}, LOST {len(lost)}')
    pd.Series(lost, dtype=str).to_csv(os.path.join(CACHE, 'finra_lost_days.csv'), index=False)
    if cov < MIN_COVERAGE:
        log('ERROR coverage < 95 % -> VOID'); sys.exit(2)


def candidate_names():
    """symbol -> name for eligible common stocks (assets csv + live BF name rule + fund screen)."""
    from trading.bf_universe_filter import is_bf_eligible
    names = {}
    with open(os.path.join(ROOT, 'data/research/databento/alpaca_assets_all_20260905.csv'), newline='') as fh:
        for row in csv.DictReader(fh):
            if row['common'] == 'True' and row['symbol'] not in names:
                names[row['symbol']] = row['name']
    ok = {s: n for s, n in names.items() if re.match(r'^[A-Z]{1,5}$', s) and not TEST_TICKER.match(s)
          and is_bf_eligible(s, names) and not FUND_NAME.search(n)}
    log(f'candidates by name: {len(ok)} of {len(names)} common')
    return ok


def stage_svr():
    """Parse all FINRA files -> 1746_svr.parquet for candidate symbols."""
    cands = candidate_names()
    frames = []
    for fn in sorted(os.listdir(FINRA_DIR)):
        if not fn.endswith('.txt'):
            continue
        df = pd.read_csv(os.path.join(FINRA_DIR, fn), sep='|', dtype={'Symbol': str}, on_bad_lines='skip')
        df = df[df.Symbol.isin(cands)]
        df = df[['Date', 'Symbol', 'ShortVolume', 'ShortExemptVolume', 'TotalVolume']].copy()
        df['Date'] = pd.to_datetime(df.Date.astype(str), format='%Y%m%d').dt.strftime('%Y-%m-%d')
        frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    d.columns = ['date', 'symbol', 'short_vol', 'exempt_vol', 'total_vol']
    med = d.groupby('symbol').total_vol.median()
    keep = med[med >= MIN_MEDIAN_TOTVOL].index
    d = d[d.symbol.isin(keep)].drop_duplicates(['date', 'symbol'])
    log(f'svr: {len(d)} rows, {d.symbol.nunique()} symbols after median TotalVolume >= {MIN_MEDIAN_TOTVOL}')
    d.to_parquet(os.path.join(HERE, '1746_svr.parquet.tmp'))
    os.replace(os.path.join(HERE, '1746_svr.parquet.tmp'), os.path.join(HERE, '1746_svr.parquet'))


def stage_bars():
    """Daily bars (raw + all-adjusted) for the SVR symbols, batched, resumable per batch."""
    syms = sorted(pd.read_parquet(os.path.join(HERE, '1746_svr.parquet'), columns=['symbol']).symbol.unique())
    bdir = os.path.join(CACHE, 'bars_parts'); os.makedirs(bdir, exist_ok=True)
    B = 100
    batches = [syms[i:i + B] for i in range(0, len(syms), B)]
    for adj in ('raw', 'all'):
        for i, b in enumerate(batches):
            part = os.path.join(bdir, f'{adj}_{i:03d}.parquet')
            if os.path.exists(part):
                continue
            if shutil.disk_usage(HERE).free / 1e9 < MIN_FREE_GB:
                log('ERROR disk < 5 GB'); sys.exit(3)
            df = get_bars(b, adj, BARS_START, BARS_END)
            df.to_parquet(part + '.tmp'); os.replace(part + '.tmp', part)
            log(f'bars {adj} batch {i + 1}/{len(batches)} rows {len(df)} symbols_got {df.symbol.nunique()}/{len(b)}')
    for adj in ('raw', 'all'):
        parts = [pd.read_parquet(os.path.join(bdir, f)) for f in sorted(os.listdir(bdir)) if f.startswith(adj + '_')
                 and f.endswith('.parquet')]
        d = pd.concat(parts, ignore_index=True)
        d.to_parquet(os.path.join(CACHE, f'bars_{adj}.parquet.tmp')); os.replace(
            os.path.join(CACHE, f'bars_{adj}.parquet.tmp'), os.path.join(CACHE, f'bars_{adj}.parquet'))
        log(f'bars {adj}: {len(d)} rows, {d.symbol.nunique()} symbols of {len(syms)} (LOST {len(syms) - d.symbol.nunique()})')


FAC_PREFIXES = ('FNSQ', 'FNYX', 'FORF', 'FNQC', 'FNRA')
CNMS_START = '2018-08-01'
VALIDATE_DAYS = ('2018-08-01', '2018-08-15', '2018-09-12', '2018-12-12', '2019-03-13')


def fetch_fac_one(args):
    """Fetch one facility file (FNSQ/FNYX/FORF/FNQC/FNRA) for a day into 1746_cache/fac; True if non-empty file."""
    pref, d = args
    out = os.path.join(CACHE, 'fac', f'{pref}{d}.txt')
    if os.path.exists(out):
        return os.path.getsize(out) > 200
    for k in range(5):
        try:
            r = requests.get(f'https://cdn.finra.org/equity/regsho/daily/{pref}shvol{d}.txt', timeout=60)
        except requests.RequestException as e:
            log(f'WARNING fac {pref}{d} {type(e).__name__}'); time.sleep(2 * (k + 1)); continue
        if r.status_code == 200 and r.content.startswith(b'Date|Symbol'):
            atomic_write(out, r.content); return len(r.content) > 200
        if r.status_code in (403, 404):
            atomic_write(out, b''); return False
        time.sleep(2 * (k + 1))
    log(f'ERROR fac {pref}{d} failed'); return False


def merge_fac(d):
    """Sum the facility files of day d (YYYYMMDD) per symbol -> DataFrame Date,Symbol,ShortVolume,ShortExemptVolume,TotalVolume."""
    fr = []
    for p in FAC_PREFIXES:
        f = os.path.join(CACHE, 'fac', f'{p}{d}.txt')
        if os.path.exists(f) and os.path.getsize(f) > 200:
            fr.append(pd.read_csv(f, sep='|', dtype={'Symbol': str}, on_bad_lines='skip'))
    if not fr:
        return None
    x = pd.concat(fr)
    return x.groupby(['Date', 'Symbol'], as_index=False)[['ShortVolume', 'ShortExemptVolume', 'TotalVolume']].sum()


def stage_fac():
    """AMENDMENT 1: pre-2018-08 days have no CNMS file; consolidate by summing the per-facility files.  Validate the sum
    against the CNMS file on days where both exist; write consolidated files for pre-CNMS days into finra/."""
    os.makedirs(os.path.join(CACHE, 'fac'), exist_ok=True)
    days = pd.read_csv(os.path.join(CACHE, 'trading_days.csv')).date.tolist()
    pre = [d.replace('-', '') for d in days if d < CNMS_START]
    val = [d.replace('-', '') for d in VALIDATE_DAYS]
    jobs = [(p, d) for d in pre + val for p in FAC_PREFIXES]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(fetch_fac_one, jobs))
    # validation on overlap days
    for d in val:
        m = merge_fac(d); c = pd.read_csv(os.path.join(FINRA_DIR, d + '.txt'), sep='|', dtype={'Symbol': str})
        j = m.merge(c, on='Symbol', suffixes=('_f', '_c'))
        eq = ((j.ShortVolume_f == j.ShortVolume_c) & (j.TotalVolume_f == j.TotalVolume_c)).mean()
        log(f'VALIDATE {d}: facility-sum rows {len(m)} cnms rows {len(c)} matched {len(j)} exact-equal share {eq:.4f}')
    n = 0
    for d in pre:
        out = os.path.join(FINRA_DIR, d + '.txt')
        m = merge_fac(d)
        if m is None:
            continue
        atomic_write(out, m.to_csv(sep='|', index=False), 'w'); n += 1
    log(f'fac: wrote {n} consolidated pre-CNMS day files of {len(pre)}')


if __name__ == '__main__':
    os.makedirs(FINRA_DIR, exist_ok=True)
    if shutil.disk_usage(HERE).free / 1e9 < MIN_FREE_GB:
        log('ERROR disk < 5 GB'); sys.exit(3)
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for name, fn in (('days', stage_days), ('finra', stage_finra), ('fac', stage_fac), ('svr', stage_svr), ('bars', stage_bars)):
        if which in (name, 'all'):
            log(f'== stage {name}')
            fn()
    log('done')
