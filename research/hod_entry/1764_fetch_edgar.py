"""Cell 1,764 EDGAR fetch: ticker->CIK map (company_tickers.json, once) and one submissions JSON per CIK.

PREREG: research/hod_entry/PREREG_1764.md.  Network ONLY to sec.gov / data.sec.gov, <= 8 req/s, descriptive
User-Agent (contact e-mail from .env ALERT_EMAIL / OWNER_EMAIL if present, else admin@onemil.local).
Checkpoints: 1764_cache/company_tickers.json, 1764_cache/submissions/CIK##########.json (+ _pN.json older pages
named in filings.files whose filingTo >= 2024-06-01); atomic writes; resumes by skipping existing files.
Usage: cd /home/ec2-user/onemil && ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/hod_entry/1764_fetch_edgar.py
"""
import os, json, time, sys, re, resource
import urllib.request, urllib.error
import pandas as pd

ROOT = '/home/ec2-user/onemil'
HE = f'{ROOT}/research/hod_entry'
CACHE = f'{HE}/1764_cache'
SUB = f'{CACHE}/submissions'
MIN_RATE_GAP = 1 / 8.0
OLDEST_NEEDED = '2024-06-01'     # earliest t - 365 d is 2024-07-01; keep a margin
_last = [0.0]


def log(m):
    """Timestamped progress line with peak RSS."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f'[{time.strftime("%H:%M:%S")}] {m} (peak RSS {rss:.0f} MB)', flush=True)


def contact():
    """Contact e-mail from .env (ALERT_EMAIL / OWNER_EMAIL) else the placeholder; logs which one was used."""
    try:
        for line in open(f'{ROOT}/.env'):
            k, _, v = line.strip().partition('=')
            if k in ('ALERT_EMAIL', 'OWNER_EMAIL') and v.strip().strip('"\''):
                return v.strip().strip('"\'')
    except OSError as e:
        log(f'WARNING: .env unreadable ({e}); using placeholder contact')
    log('WARNING: no ALERT_EMAIL/OWNER_EMAIL in .env; using admin@onemil.local')
    return 'admin@onemil.local'


UA = f'onemil research {contact()}'


def get(url, tries=4):
    """Rate-limited GET (<= 8/s) with backoff on 429/5xx; returns bytes or None (logged WARNING) on 404."""
    for k in range(tries):
        wait = MIN_RATE_GAP - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept-Encoding': 'identity'})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                log(f'WARNING: 404 {url}')
                return None
            log(f'WARNING: HTTP {e.code} {url} try {k + 1}')
            time.sleep(2 * (k + 1))
        except Exception as e:
            log(f'WARNING: {type(e).__name__} {url} try {k + 1}')
            time.sleep(2 * (k + 1))
    log(f'ERROR: giving up on {url}')
    return None


def save(path, data):
    """Atomic write of bytes."""
    with open(path + '.tmp', 'wb') as f:
        f.write(data)
    os.replace(path + '.tmp', path)


def norm(s):
    """SEC ticker spelling: upper-case, '.' -> '-'."""
    return str(s).upper().replace('.', '-')


def ticker_map():
    """Fetch company_tickers.json once (cached) and return {ticker: cik}."""
    p = f'{CACHE}/company_tickers.json'
    if not os.path.exists(p):
        d = get('https://www.sec.gov/files/company_tickers.json')
        if d is None:
            sys.exit('ERROR: company_tickers.json unavailable')
        save(p, d)
    j = json.load(open(p))
    return {norm(v['ticker']): int(v['cik_str']) for v in j.values()}


def main():
    pop = pd.read_csv(f'{HE}/cell_1428_population.csv', dtype={'symbol': str}, keep_default_na=False)
    tm = ticker_map()
    syms = sorted(pop.symbol.unique())
    cik = {s: tm.get(norm(s)) for s in syms}
    ciks = sorted({c for c in cik.values() if c})
    log(f'{len(syms)} symbols, mapped {sum(1 for c in cik.values() if c)}, {len(ciks)} CIKs')
    pd.DataFrame({'symbol': syms, 'cik': [cik[s] for s in syms]}).to_csv(f'{CACHE}/symbol_cik.csv', index=False)
    t0, nf, npg = time.time(), 0, 0
    for k, c in enumerate(ciks):
        p = f'{SUB}/CIK{c:010d}.json'
        if os.path.exists(p):
            d = json.load(open(p))
        else:
            raw = get(f'https://data.sec.gov/submissions/CIK{c:010d}.json')
            if raw is None:
                continue
            save(p, raw); d = json.loads(raw); nf += 1
        for pg in d.get('filings', {}).get('files', []):
            if pg.get('filingTo', '9999') < OLDEST_NEEDED:
                continue
            pp = f'{SUB}/CIK{c:010d}_{pg["name"]}'
            if not os.path.exists(pp):
                raw = get(f'https://data.sec.gov/submissions/{pg["name"]}')
                if raw is not None:
                    save(pp, raw); npg += 1
        if (k + 1) % 100 == 0:
            log(f'{k + 1}/{len(ciks)} CIKs, {nf} new fetched, {npg} extra pages, {time.time() - t0:.0f}s')
        if time.time() - t0 > 560:
            log(f'time budget reached at {k + 1}/{len(ciks)}; rerun to resume')
            return
    log(f'DONE {len(ciks)} CIKs; new {nf}, extra pages {npg}')


if __name__ == '__main__':
    main()
