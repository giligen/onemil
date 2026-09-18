#!/usr/bin/env python3
"""Shared SEC EDGAR plumbing for the multi-day DATA stage.

EDGAR is free but has two hard rules: a descriptive User-Agent (else 403) and
≤ 10 requests/second (else 403/429 and eventually an IP block). Both are enforced
here, in one place, so no caller can forget them.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

USER_AGENT = 'onemil research giligen@gmail.com'
MIN_INTERVAL = 0.12          # ≈ 8.3 req/s, inside the 10/s ceiling
MAX_RETRIES = 5

_session = requests.Session()
_session.headers.update({'User-Agent': USER_AGENT,
                         'Accept-Encoding': 'gzip, deflate',
                         'Host': 'data.sec.gov'})
_last_call = [0.0]


def get_json(url: str, allow_404: bool = False):
    """Rate-limited GET returning parsed JSON, or None on an allowed 404.

    Sleeps on 403/429 (EDGAR's throttle answer) with exponential backoff and
    logs every retry — a silent throttle would look like missing data.
    """
    headers = {'Host': 'data.sec.gov' if 'data.sec.gov' in url else 'www.sec.gov'}
    backoff = 2.0
    for attempt in range(MAX_RETRIES):
        wait = MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            r = _session.get(url, headers=headers, timeout=30)
        except Exception as exc:  # noqa: BLE001
            print(f'  EDGAR transport error {exc} on {url} (attempt {attempt + 1})', flush=True)
            time.sleep(backoff)
            backoff *= 2
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except json.JSONDecodeError as exc:
                print(f'  EDGAR bad JSON on {url}: {exc}', flush=True)
                return None
        if r.status_code == 404:
            if allow_404:
                return None
            print(f'  EDGAR 404 {url}', flush=True)
            return None
        if r.status_code in (403, 429, 500, 502, 503):
            print(f'  EDGAR {r.status_code} on {url} — sleeping {backoff:.0f}s '
                  f'(attempt {attempt + 1}/{MAX_RETRIES})', flush=True)
            time.sleep(backoff)
            backoff *= 2
            continue
        print(f'  EDGAR unexpected {r.status_code} on {url}', flush=True)
        return None
    print(f'  EDGAR GAVE UP on {url} after {MAX_RETRIES} attempts', flush=True)
    return None


def company_tickers() -> dict[str, int]:
    """{normalized ticker -> CIK} from https://www.sec.gov/files/company_tickers.json."""
    data = get_json('https://www.sec.gov/files/company_tickers.json')
    if not data:
        raise RuntimeError('company_tickers.json unavailable — cannot map tickers to CIK')
    out = {}
    for rec in data.values():
        t = str(rec['ticker']).upper().replace('-', '.').replace(' ', '.')
        out[t] = int(rec['cik_str'])
    return out


def submissions(cik: int) -> dict | None:
    """The main submissions record for a CIK (recent filings + pointers to older pages)."""
    return get_json(f'https://data.sec.gov/submissions/CIK{cik:010d}.json', allow_404=True)


def submissions_page(name: str) -> dict | None:
    """An older submissions page referenced by `filings.files[*].name`."""
    return get_json(f'https://data.sec.gov/submissions/{name}', allow_404=True)


def company_concept(cik: int, tag: str, taxonomy: str = 'us-gaap') -> dict | None:
    """XBRL companyconcept facts for one tag, e.g. EarningsPerShareDiluted."""
    return get_json(
        f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/{taxonomy}/{tag}.json',
        allow_404=True)


def shard_paths(outdir: Path, prefix: str, index: int) -> Path:
    """Checkpoint file for one shard of CIKs."""
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / f'{prefix}_{index:03d}.parquet'
