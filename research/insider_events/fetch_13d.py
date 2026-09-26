#!/usr/bin/env python3
"""BUILDER B - cell I4 (initial Schedule 13D filings), data fetch.

Pulls every SC 13D / SC 13D/A hit from the EDGAR full-text search API
(efts.sec.gov/LATEST/search-index) for 2016-01-01..2026-09-26, keeps only the
exact form 'SC 13D' (drops 'SC 13D/A' amendments), maps each filing's subject
company to a panel symbol via its display ticker or, failing that, its CIK
against research/multiday/data/universe.parquet, and writes
research/insider_events/data/sc13d.parquet.

Resumable: each (start,end) date chunk is fetched to its own JSONL file under
data/raw_chunks_13d/; a chunk already on disk is never re-fetched. Rate limit
1 request/second (SEC ceiling is 10/s; PREREG asks for 1/s). User-Agent
'onemil research giligen@gmail.com' per PREREG.

Usage:
    python3 fetch_13d.py            # fetch (resumable) + build parquet
    python3 fetch_13d.py --build-only   # skip fetch, just rebuild parquet from existing chunks
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
CHUNK_DIR = DATA / 'raw_chunks_13d'
OUT_PARQUET = DATA / 'sc13d.parquet'
UNIVERSE_PARQUET = HERE.parent / 'multiday' / 'data' / 'universe.parquet'

USER_AGENT = 'onemil research giligen@gmail.com'
BASE_URL = 'https://efts.sec.gov/LATEST/search-index'
MIN_INTERVAL = 1.0          # PREREG: <= 1 request/second
PAGE_SIZE = 100
MAX_FROM = 9900             # efts caps from+size at 10,000; split the range before that
MAX_RETRIES = 6

START_DATE = date(2016, 1, 1)
END_DATE = date(2026, 9, 26)

_session = requests.Session()
_session.headers.update({'User-Agent': USER_AGENT, 'Accept': 'application/json'})
_last_call = [0.0]


def log(m: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def _get(params: dict) -> dict | None:
    """Rate-limited GET with retry/backoff. Every fallback path logs why."""
    backoff = 2.0
    for attempt in range(MAX_RETRIES):
        wait = MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            r = _session.get(BASE_URL, params=params, timeout=30)
        except Exception as exc:  # noqa: BLE001
            log(f'WARNING transport error {exc} on {params} (attempt {attempt + 1}) - retrying')
            time.sleep(backoff)
            backoff *= 2
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except json.JSONDecodeError as exc:
                log(f'ERROR bad JSON on {params}: {exc}')
                return None
        if r.status_code in (403, 429, 500, 502, 503):
            log(f'WARNING EDGAR {r.status_code} on {params} - sleeping {backoff:.0f}s '
                f'(attempt {attempt + 1}/{MAX_RETRIES})')
            time.sleep(backoff)
            backoff *= 2
            continue
        log(f'ERROR unexpected {r.status_code} on {params}: {r.text[:300]}')
        return None
    log(f'ERROR GAVE UP on {params} after {MAX_RETRIES} attempts')
    return None


def search_forms13d(startdt: date, enddt: date, frm: int = 0, empty_query_ok: bool = True) -> dict | None:
    """One page of the full-text search API for SC 13D (root form; includes 13D/A -
    filtered post-hoc). Empty q='*' returned 0 hits when tested live (2026-09-26), so the
    documented fallback q='13D' is used unconditionally after the first probe fails."""
    params = {'forms': 'SC 13D', 'dateRange': 'custom',
              'startdt': startdt.isoformat(), 'enddt': enddt.isoformat(),
              'from': frm}
    params['q'] = '"*"' if empty_query_ok else '"13D"'
    return _get(params)


def fetch_range(startdt: date, enddt: date, q_works: list) -> None:
    """Fetch one date range to CHUNK_DIR/<start>_<end>.jsonl, splitting the range in half
    if the API reports more hits than from+size can page through (efts 10k cap)."""
    key = f'{startdt.isoformat()}_{enddt.isoformat()}'
    outp = CHUNK_DIR / f'{key}.jsonl'
    if outp.exists():
        log(f'skip {key} (already on disk)')
        return

    # probe once per run whether the empty query works; PREREG: try q=%22*%22 first,
    # fall back to q=%2213D%22 and say so.
    if not q_works:
        probe = search_forms13d(startdt, enddt, 0, empty_query_ok=True)
        empty_ok = bool(probe and probe.get('hits', {}).get('total', {}).get('value', 0) > 0)
        if not empty_ok:
            log('WARNING empty query q="*" returned 0 hits on the probe range - '
                'falling back to q="13D" for every request (PREREG documented fallback)')
        q_works.append(empty_ok)
    empty_ok = q_works[0]

    first = search_forms13d(startdt, enddt, 0, empty_query_ok=empty_ok)
    if first is None:
        log(f'ERROR could not fetch {key} at all - leaving unwritten for a later retry')
        return
    total = first['hits']['total']['value']
    if total == 0:
        outp.write_text('')
        log(f'{key}: 0 hits')
        return
    if total > MAX_FROM:
        mid = startdt + (enddt - startdt) // 2
        if mid <= startdt:
            log(f'ERROR range {key} cannot be split further ({total} hits in a single day) - '
                f'writing what the first page gives and moving on')
        else:
            log(f'{key}: {total} hits > cap {MAX_FROM} - splitting at {mid}')
            fetch_range(startdt, mid, q_works)
            fetch_range(mid + timedelta(days=1), enddt, q_works)
            return

    hits = list(first['hits']['hits'])
    frm = PAGE_SIZE
    while frm < total and frm <= MAX_FROM:
        page = search_forms13d(startdt, enddt, frm, empty_query_ok=empty_ok)
        if page is None:
            log(f'ERROR page from={frm} of {key} failed after retries - chunk left incomplete, '
                f'NOT writing (will retry whole chunk next run)')
            return
        page_hits = page['hits']['hits']
        if not page_hits:
            break
        hits.extend(page_hits)
        frm += PAGE_SIZE
    with outp.open('w') as f:
        for h in hits:
            f.write(json.dumps(h['_source']) + '\n')
    log(f'{key}: wrote {len(hits)} hits (reported total {total})')


def month_ranges(start: date, end: date):
    cur = start
    while cur <= end:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        chunk_end = min(nxt - timedelta(days=1), end)
        yield cur, chunk_end
        cur = nxt


def do_fetch() -> None:
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    ranges = list(month_ranges(START_DATE, END_DATE))
    log(f'{len(ranges)} monthly chunks, {START_DATE} .. {END_DATE}')
    q_works: list = []
    for i, (a, b) in enumerate(ranges):
        fetch_range(a, b, q_works)
        if (i + 1) % 12 == 0:
            log(f'progress: {i + 1}/{len(ranges)} chunks')


TICKER_RE = re.compile(r'\(([A-Z][A-Z0-9.\-]{0,5})\)\s*\(CIK')
CIK_RE = re.compile(r'CIK\s*0*([0-9]+)')


def parse_hit(src: dict) -> dict | None:
    form = src.get('form')
    if form != 'SC 13D':      # exclude SC 13D/A amendments (PREREG signal definition)
        return None
    ciks = src.get('ciks') or []
    names = src.get('display_names') or []
    if not ciks or not names:
        return None
    subject_cik = int(ciks[0])
    subject_name = names[0]
    m = TICKER_RE.search(subject_name)
    ticker = m.group(1) if m else None
    return dict(accession=src.get('adsh'), filed_date=src.get('file_date'),
                acceptance_datetime=None,   # not present on this API - see RESULT caveats
                subject_cik=subject_cik, subject_name=subject_name, ticker_raw=ticker)


def build_parquet() -> None:
    files = sorted(CHUNK_DIR.glob('*.jsonl'))
    log(f'reading {len(files)} chunk files')
    rows = []
    for fp in files:
        if fp.stat().st_size == 0:
            continue
        with fp.open() as f:
            for line in f:
                src = json.loads(line)
                r = parse_hit(src)
                if r is not None:
                    rows.append(r)
    df = pd.DataFrame(rows)
    n_raw = len(df)
    df = df.drop_duplicates(subset=['accession']).reset_index(drop=True)
    log(f'initial SC 13D rows: {n_raw} raw, {len(df)} after accession de-dup')

    uni = pd.read_parquet(UNIVERSE_PARQUET, columns=['symbol', 'cik'])
    uni = uni.dropna(subset=['cik']).copy()
    uni['cik'] = uni['cik'].astype('int64')
    tick2sym = {s.upper(): s for s in uni['symbol']}
    cik2sym = uni.drop_duplicates('cik').set_index('cik')['symbol'].to_dict()

    def map_row(r):
        t = r['ticker_raw']
        if t and t.upper() in tick2sym:
            return pd.Series([tick2sym[t.upper()], 'ticker'])
        s = cik2sym.get(int(r['subject_cik']))
        if s is not None:
            return pd.Series([s, 'cik'])
        return pd.Series([None, 'unmapped'])

    df[['symbol', 'map_method']] = df.apply(map_row, axis=1)
    df['filed_date'] = pd.to_datetime(df['filed_date'])
    n_unmapped = (df['map_method'] == 'unmapped').sum()
    log(f'mapped {len(df) - n_unmapped}/{len(df)} to a panel symbol '
        f'({100 * n_unmapped / max(len(df), 1):.1f}% unmapped)')
    df.to_parquet(OUT_PARQUET, index=False)
    log(f'wrote {OUT_PARQUET} ({len(df)} rows)')


if __name__ == '__main__':
    if '--build-only' not in sys.argv:
        do_fetch()
    build_parquet()
