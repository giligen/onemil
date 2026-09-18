#!/usr/bin/env python3
"""Data pulls for the FINAL multi-day stage (F5, A2, A3, A4, F1, F6).

Three sources, each resumable, each written to its own parquet:

  1. FINRA consolidated equity short interest (A2)  -> short_interest.parquet
     Free `api.finra.org` dataset `otcMarket/consolidatedShortInterest`.  Keyed on
     `settlementDate`; the file's DISSEMINATION date is derived in the family stage
     (settlement + 8 business days, FINRA Rule 4560 practice) because the dataset does
     not carry it.  **History starts 2018-02-15** (2018-01-15 and everything before it
     return 0 rows) -- A2's TRAIN split is therefore 2018-02 -> 2021-12, not 2016-01.

  2. Alpaca cash dividends (A4)                     -> dividends.parquet
     `/v1/corporate-actions?types=cash_dividend`.  Carries `ex_date`, `record_date`,
     `payable_date`, `process_date`, `rate`, `special`.  `record_date` is only populated
     from ~2019-07 onward; the two-regime ex-date reconciliation therefore runs on the
     subset that has one.  Pulled through 2024-12-31 so the 2024-05-28 T+1 boundary can
     be reconciled ON DATES ONLY -- no TEST-window price is touched.

  3. EDGAR `dei:EntityCommonStockSharesOutstanding` (A3, and A2's SI ratio denominator)
                                                     -> shares_facts.parquet
     Cover-page share count with its `filed` date = point in time.  Falls back to
     `us-gaap:CommonStockSharesOutstanding` when the dei concept is absent.

Run: `nice -n 10 python3 fetch_final.py [finra|divs|shares|all]`
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edgar_common import get_json  # noqa: E402

D = Path('/home/ec2-user/onemil/research/multiday/data')
ENV = Path('/home/ec2-user/onemil/.env')

FINRA_URL = 'https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest'
FINRA_START = '2018-01-01'
FINRA_END = '2023-12-31'          # VAL end; TEST stays sealed
DIV_START = '2016-01-01'
DIV_END = '2024-12-31'            # dates only past 2023-12-31, for the T+1 boundary check


def log(m: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def alpaca_headers() -> dict:
    env = {}
    for line in ENV.read_text().splitlines():
        if '=' in line and not line.strip().startswith('#'):
            k, v = line.split('=', 1)
            env[k.strip()] = v.strip()
    return {'APCA-API-KEY-ID': env['ALPACA_API_KEY'],
            'APCA-API-SECRET-KEY': env['ALPACA_API_SECRET']}


# ------------------------------------------------------------------ 1. FINRA


def fetch_finra() -> None:
    """Semi-monthly consolidated short interest, one month-range query at a time."""
    out = D / 'short_interest.parquet'
    if out.exists():
        log(f'{out.name} exists -- skipping (delete to refetch)')
        return
    keep = ['symbolCode', 'settlementDate', 'currentShortPositionQuantity',
            'previousShortPositionQuantity', 'averageDailyVolumeQuantity',
            'daysToCoverQuantity', 'marketClassCode', 'stockSplitFlag', 'revisionFlag']
    months = pd.date_range(FINRA_START, FINRA_END, freq='MS')
    frames = []
    sess = requests.Session()
    for ms in months:
        me = (ms + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
        offset, got = 0, 0
        while True:
            body = {'limit': 5000, 'offset': offset,
                    'dateRangeFilters': [{'fieldName': 'settlementDate',
                                          'startDate': ms.strftime('%Y-%m-%d'),
                                          'endDate': me}]}
            for attempt in range(5):
                try:
                    r = sess.post(FINRA_URL, json=body,
                                  headers={'Accept': 'application/json'}, timeout=120)
                    break
                except Exception as exc:  # noqa: BLE001
                    log(f'  FINRA transport error {exc} (attempt {attempt + 1})')
                    time.sleep(3 * (attempt + 1))
            else:
                raise RuntimeError('FINRA unreachable')
            if r.status_code != 200:
                raise RuntimeError(f'FINRA {r.status_code} {r.text[:200]}')
            rows = r.json()
            if not rows:
                break
            frames.append(pd.DataFrame(rows)[keep])
            got += len(rows)
            offset += len(rows)
            if len(rows) < 5000:
                break
        log(f'  FINRA {ms:%Y-%m}: {got:,} rows')
    df = pd.concat(frames, ignore_index=True)
    df['settlementDate'] = pd.to_datetime(df['settlementDate'])
    for c in ('currentShortPositionQuantity', 'previousShortPositionQuantity',
              'averageDailyVolumeQuantity', 'daysToCoverQuantity'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.to_parquet(out, index=False)
    log(f'wrote {out} {len(df):,} rows | {df["settlementDate"].nunique()} settlement dates '
        f'| {df["symbolCode"].nunique():,} symbols')


# ------------------------------------------------------------------ 2. dividends


def fetch_dividends() -> None:
    out = D / 'dividends.parquet'
    if out.exists():
        log(f'{out.name} exists -- skipping (delete to refetch)')
        return
    h = alpaca_headers()
    sess = requests.Session()
    rows, token, page = [], None, 0
    # the endpoint caps a query window, so walk it a quarter at a time
    for qs in pd.date_range(DIV_START, DIV_END, freq='QS'):
        qe = min(pd.Timestamp(DIV_END), qs + pd.offsets.QuarterEnd(0))
        token = None
        while True:
            p = {'types': 'cash_dividend', 'start': qs.strftime('%Y-%m-%d'),
                 'end': qe.strftime('%Y-%m-%d'), 'limit': 1000}
            if token:
                p['page_token'] = token
            for attempt in range(5):
                try:
                    r = sess.get('https://data.alpaca.markets/v1/corporate-actions',
                                 headers=h, params=p, timeout=60)
                    break
                except Exception as exc:  # noqa: BLE001
                    log(f'  Alpaca transport error {exc} (attempt {attempt + 1})')
                    time.sleep(3 * (attempt + 1))
            else:
                raise RuntimeError('Alpaca unreachable')
            if r.status_code != 200:
                raise RuntimeError(f'Alpaca CA {r.status_code} {r.text[:200]}')
            j = r.json()
            batch = j.get('corporate_actions', {}).get('cash_dividends', []) or []
            rows.extend(batch)
            page += 1
            token = j.get('next_page_token')
            if not token:
                break
        log(f'  divs {qs:%Y-Q%q}: cumulative {len(rows):,} rows, {page} pages')
    df = pd.DataFrame(rows)
    for c in ('ex_date', 'record_date', 'payable_date', 'process_date'):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    df = df.drop_duplicates(subset=['symbol', 'ex_date', 'rate', 'record_date'], keep='first')
    df.to_parquet(out, index=False)
    log(f'wrote {out} {len(df):,} rows | {df["symbol"].nunique():,} symbols | '
        f'record_date present {df["record_date"].notna().mean():.1%}')


# ------------------------------------------------------------------ 3. shares outstanding


def fetch_shares() -> None:
    out = D / 'shares_facts.parquet'
    part = D / 'shares_parts'
    part.mkdir(exist_ok=True)
    if out.exists():
        log(f'{out.name} exists -- skipping (delete to refetch)')
        return
    uni = pd.read_parquet(D / 'universe.parquet', columns=['symbol', 'kind', 'cik'])
    ciks = sorted({int(c) for c in uni.loc[uni['kind'] == 'common', 'cik'].dropna()})
    log(f'shares: {len(ciks):,} CIKs (common stocks with a CIK)')
    buf, done = [], 0
    for i, cik in enumerate(ciks):
        chunk = i // 500
        f = part / f'shares_{chunk:03d}.parquet'
        if f.exists() and (i % 500) == 0:
            log(f'  chunk {chunk:03d} exists -- skipping 500 CIKs')
        if f.exists():
            continue
        j = get_json(f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}'
                     f'/dei/EntityCommonStockSharesOutstanding.json', allow_404=True)
        tag = 'dei:EntityCommonStockSharesOutstanding'
        if not j or not j.get('units'):
            j = get_json(f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}'
                         f'/us-gaap/CommonStockSharesOutstanding.json', allow_404=True)
            tag = 'us-gaap:CommonStockSharesOutstanding'
        if j and j.get('units'):
            for unit, arr in j['units'].items():
                if unit != 'shares':
                    continue
                for a in arr:
                    buf.append((cik, tag, a.get('end'), a.get('val'), a.get('filed'),
                                a.get('form'), a.get('accn')))
        done += 1
        if (i + 1) % 500 == 0 or i == len(ciks) - 1:
            chunk_done = i // 500
            fd = part / f'shares_{chunk_done:03d}.parquet'
            pd.DataFrame(buf, columns=['cik', 'tag', 'end', 'val', 'filed', 'form', 'accn']) \
                .to_parquet(fd, index=False)
            log(f'  chunk {chunk_done:03d} written ({len(buf):,} facts, {i + 1:,}/{len(ciks):,} CIKs)')
            buf = []
    frames = [pd.read_parquet(f) for f in sorted(part.glob('shares_*.parquet'))]
    df = pd.concat(frames, ignore_index=True)
    df['end'] = pd.to_datetime(df['end'], errors='coerce')
    df['filed'] = pd.to_datetime(df['filed'], errors='coerce')
    df = df.dropna(subset=['end', 'filed', 'val'])
    df = df.sort_values(['cik', 'end', 'filed']).drop_duplicates(['cik', 'end', 'filed'], keep='last')
    df.to_parquet(out, index=False)
    log(f'wrote {out} {len(df):,} facts | {df["cik"].nunique():,} CIKs')


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('finra', 'all'):
        fetch_finra()
    if what in ('divs', 'all'):
        fetch_dividends()
    if what in ('shares', 'all'):
        fetch_shares()
    return 0


if __name__ == '__main__':
    sys.exit(main())
