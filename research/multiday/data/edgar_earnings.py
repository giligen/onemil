#!/usr/bin/env python3
"""Multi-day DATA stage, steps 4 + 6 — the point-in-time earnings calendar and SIC.

Earnings release = an 8-K carrying Item **2.02** ("Results of Operations and
Financial Condition"), timestamped by EDGAR's `acceptanceDateTime` — the instant
the filing became public, which is the only point-in-time stamp in the record
(`filingDate` is a date and can precede or follow the tape session).

Event session (the decision session, F1/F2 anchor):
    the FIRST trading session whose closing auction is strictly AFTER the
    acceptance time. An 8-K accepted 07:00 ET trades that same session; one
    accepted 17:30 ET trades the next session. Early closes use the exchange
    calendar's real close (13:00 ET), not a hardcoded 16:00.

Also harvests, from the same submissions record, the 2-digit SIC used by the
industry-adjusted reversal family (F4) and writes it back into `universe.parquet`.

Resumable: one parquet per shard of CIKs under `edgar/`. Delete a shard to redo it.

    python3 edgar_earnings.py            # fetch shards then finalize
    python3 edgar_earnings.py --finalize # only rebuild the outputs from shards
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path('/home/ec2-user/onemil')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from edgar_common import (company_tickers, submissions, submissions_page,  # noqa: E402
                          shard_paths)

HERE = Path(__file__).resolve().parent
UNIVERSE = HERE / 'universe.parquet'
SHARDS = HERE / 'edgar'
EVENTS_OUT = HERE / 'earnings_events.parquet'
CIKMAP_OUT = HERE / 'cik_map.parquet'
SHARD_SIZE = 200
FIRST_YEAR = 2015          # keep 2015 filings so 2016-01 events have a lookback


def _filing_rows(block: dict, cik: int) -> list[dict]:
    """Extract Item-2.02 8-Ks from one submissions block (recent or an older page)."""
    if not block or 'accessionNumber' not in block:
        return []
    n = len(block['accessionNumber'])
    out = []
    for i in range(n):
        form = block['form'][i]
        if form not in ('8-K', '8-K/A'):
            continue
        items = block.get('items', [''] * n)[i] or ''
        if '2.02' not in items:
            continue
        fdate = block['filingDate'][i]
        if fdate and int(fdate[:4]) < FIRST_YEAR:
            continue
        out.append({
            'cik': cik,
            'form': form,
            'accession': block['accessionNumber'][i],
            'filing_date': fdate,
            'acceptance_utc': block.get('acceptanceDateTime', [None] * n)[i],
            'items': items,
        })
    return out


def fetch_shards(ciks: list[int]) -> None:
    """Pull submissions for every CIK, checkpointing one parquet per shard."""
    shards = [ciks[i:i + SHARD_SIZE] for i in range(0, len(ciks), SHARD_SIZE)]
    for si, shard in enumerate(shards):
        ev_path = shard_paths(SHARDS, 'events', si)
        meta_path = shard_paths(SHARDS, 'meta', si)
        if ev_path.exists() and meta_path.exists():
            continue
        rows, meta = [], []
        for cik in shard:
            rec = submissions(cik)
            if rec is None:
                meta.append({'cik': cik, 'sic': '', 'sic_desc': '', 'name': '',
                             'n_events': 0, 'ok': False})
                continue
            filings = rec.get('filings', {})
            got = _filing_rows(filings.get('recent', {}), cik)
            for page in filings.get('files', []):
                to_date = page.get('filingTo', '')
                if to_date and int(to_date[:4]) < FIRST_YEAR:
                    continue
                got += _filing_rows(submissions_page(page['name']), cik)
            rows += got
            meta.append({'cik': cik, 'sic': str(rec.get('sic', '') or ''),
                         'sic_desc': str(rec.get('sicDescription', '') or ''),
                         'name': str(rec.get('name', '') or ''),
                         'n_events': len(got), 'ok': True})
        cols = ['cik', 'form', 'accession', 'filing_date', 'acceptance_utc', 'items']
        pd.DataFrame(rows, columns=cols).to_parquet(ev_path, index=False)
        pd.DataFrame(meta).to_parquet(meta_path, index=False)
        print(f'shard {si + 1}/{len(shards)}: {len(rows)} item-2.02 8-Ks '
              f'from {len(shard)} CIKs', flush=True)


def session_calendar() -> pd.DataFrame:
    """NYSE sessions 2015-2026 with their real close times (early closes included)."""
    import os
    from dotenv import load_dotenv
    load_dotenv(REPO / '.env')
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetCalendarRequest
    from datetime import date

    client = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_API_SECRET'),
                           paper=False)
    cal = client.get_calendar(GetCalendarRequest(start=date(2015, 1, 1), end=date(2026, 12, 31)))
    # alpaca-py returns `close` as a datetime.time on some versions and a full
    # datetime on others — normalise to a time-of-day string either way.
    rows = []
    for c in cal:
        close = c.close
        tod = close.time() if hasattr(close, 'time') and not isinstance(close, str) else close
        rows.append({'session': pd.Timestamp(c.date), 'close': str(tod)[:8]})
    df = pd.DataFrame(rows)
    df['close_et'] = pd.to_datetime(
        df['session'].dt.strftime('%Y-%m-%d') + ' ' + df['close']
    ).dt.tz_localize('America/New_York')
    return df.sort_values('session').reset_index(drop=True)


def assign_sessions(events: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """Attach the first session whose close is strictly after the acceptance time."""
    acc = pd.to_datetime(events['acceptance_utc'], errors='coerce', utc=True)
    events['acceptance_utc'] = acc
    events['acceptance_et'] = acc.dt.tz_convert('America/New_York')
    closes = cal['close_et'].values
    idx = pd.Series(closes).searchsorted(acc.values, side='right')
    ok = idx < len(cal)
    events['event_session'] = pd.NaT
    events.loc[ok, 'event_session'] = cal['session'].values[idx[ok]]
    # time-of-day bucket relative to the regular session
    et = events['acceptance_et']
    mins = et.dt.hour * 60 + et.dt.minute
    events['acceptance_bucket'] = pd.cut(
        mins, bins=[-1, 569, 959, 1440],
        labels=['pre_open', 'intraday', 'post_close']).astype('object')
    events.loc[et.isna(), 'acceptance_bucket'] = None
    return events


def finalize(cik_map: pd.DataFrame) -> None:
    """Combine shards → earnings_events.parquet (+ SIC into universe.parquet)."""
    ev = pd.concat([pd.read_parquet(p) for p in sorted(SHARDS.glob('events_*.parquet'))],
                   ignore_index=True)
    meta = pd.concat([pd.read_parquet(p) for p in sorted(SHARDS.glob('meta_*.parquet'))],
                     ignore_index=True)
    print(f'raw item-2.02 8-Ks: {len(ev)} from {ev["cik"].nunique()} CIKs', flush=True)

    ev = ev.drop_duplicates(['cik', 'accession'])
    ev = assign_sessions(ev, session_calendar())
    ev = ev.merge(cik_map[['cik', 'symbol']], on='cik', how='inner')
    ev = ev[ev['event_session'].notna()]
    ev = ev[ev['event_session'] >= pd.Timestamp('2016-01-04')]
    ev = ev.sort_values(['symbol', 'acceptance_utc']).reset_index(drop=True)
    ev.to_parquet(EVENTS_OUT, index=False)
    print(f'{len(ev)} events on {ev["symbol"].nunique()} symbols -> {EVENTS_OUT}', flush=True)
    print(ev.groupby(ev['event_session'].dt.year).size().to_string(), flush=True)
    print(ev['acceptance_bucket'].value_counts(normalize=True).round(4).to_string(), flush=True)

    uni = pd.read_parquet(UNIVERSE)
    meta = meta[meta['ok']].drop_duplicates('cik')
    sic = cik_map.merge(meta[['cik', 'sic', 'sic_desc', 'name']], on='cik', how='left')
    sic['sic2'] = sic['sic'].str[:2].where(sic['sic'].str.len() >= 3, sic['sic'].str.zfill(4).str[:2])
    sic.loc[sic['sic'].fillna('') == '', 'sic2'] = None
    uni = uni.drop(columns=[c for c in ('cik', 'sic', 'sic2', 'sic_desc') if c in uni.columns])
    uni = uni.merge(sic[['symbol', 'cik', 'sic', 'sic2', 'sic_desc']], on='symbol', how='left')
    uni.to_parquet(UNIVERSE, index=False)
    print(f'universe.parquet: SIC on {uni["sic2"].notna().sum()}/{len(uni)} symbols '
          f'({100 * uni["sic2"].notna().mean():.1f}%)', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--finalize', action='store_true')
    args = ap.parse_args()

    uni = pd.read_parquet(UNIVERSE)
    tick2cik = company_tickers()
    rows = [{'symbol': s, 'cik': tick2cik[s]} for s in uni['symbol'] if s in tick2cik]
    cik_map = pd.DataFrame(rows)
    cik_map.to_parquet(CIKMAP_OUT, index=False)
    print(f'{len(cik_map)}/{len(uni)} universe symbols map to a CIK '
          f'({cik_map["cik"].nunique()} distinct CIKs)', flush=True)

    if not args.finalize:
        fetch_shards(sorted(cik_map['cik'].unique().tolist()))
    finalize(cik_map)


if __name__ == '__main__':
    main()
