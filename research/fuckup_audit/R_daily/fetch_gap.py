#!/usr/bin/env python3
"""Stage R_daily step 1 — the ONE missing slice of daily history: XNAS.ITCH ohlcv-1d 2024-01-01..2024-07-01.

On disk already (do NOT re-buy):
  * `N_databento/N3/xnas_daily.parquet`  — XNAS.ITCH 2018-05-01..2023-12-29 (12.28M rows, 18,575 symbols)
  * `data/research/databento/equs_daily_2024H2.parquet` + `equs_daily_2025_2026.parquet` — EQUS.SUMMARY 2024-07..now
  * `N_databento/N3/raw_daily/xnas_ohlcv1d_cal202409.parquet` — the ITCH calibration month (the SEAM overlap)

Convention (the N3 bug and its fix, `N_databento/N3/fix_symbol_map.py`): XNAS.ITCH ohlcv-1d stamps
`ts_event` at 00:00 UTC of the SESSION date, so the session date is the **UTC** date of ts_event
(an ET conversion shifts every bar one day earlier).  Instrument ids are re-issued EVERY DAY on
ITCH, so the id -> symbol join is an as-of join on the symbology interval, per day.

Usage:  python3 fetch_gap.py --cost-only | python3 fetch_gap.py --budget 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')
load_dotenv('/home/ec2-user/onemil/.env')

import databento as db  # noqa: E402

DATASET = 'XNAS.ITCH'
SCHEMA = 'ohlcv-1d'
R = 'research/fuckup_audit/R_daily'
RAW = f'{R}/raw'
START, END = '2024-01-01', '2024-07-01'
OUT = f'{R}/xnas_daily_2024H1.parquet'


def log(m):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {m}', flush=True)


def month_windows(s, e):
    out, cur, end = [], pd.Timestamp(s), pd.Timestamp(e)
    while cur < end:
        nxt = min(cur + pd.offsets.MonthBegin(1), end)
        out.append((cur.strftime('%Y-%m-%d'), nxt.strftime('%Y-%m-%d')))
        cur = nxt
    return out


def symbology_month(client, m0, m1):
    """Monthly instrument_id <-> raw_symbol intervals (FREE endpoint), cached as parquet."""
    pq = f'{RAW}/sym_{m0}.parquet'
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    for attempt in range(6):
        try:
            r = client.symbology.resolve(dataset=DATASET, symbols='ALL_SYMBOLS',
                                         stype_in='raw_symbol', stype_out='instrument_id',
                                         start_date=m0, end_date=m1)
            break
        except Exception as exc:                      # 504s are common on wide windows
            log(f'  symbology {m0} attempt {attempt + 1}: {str(exc)[:70]}')
            time.sleep(15 * (attempt + 1))
    else:
        raise RuntimeError(f'symbology {m0} unavailable')
    rows = [(int(iv['s']), int(iv['d0'].replace('-', '')), int(iv['d1'].replace('-', '')), sym)
            for sym, ivs in r['result'].items() for iv in ivs]
    m = pd.DataFrame(rows, columns=['instrument_id', 'd0i', 'd1i', 'symbol'])
    m['instrument_id'] = m.instrument_id.astype('uint32')
    m.to_parquet(pq, index=False)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cost-only', action='store_true')
    ap.add_argument('--budget', type=float, default=10.0)
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)

    client = db.Historical(os.environ['DATABENTO_API_KEY'])
    cost = float(client.metadata.get_cost(dataset=DATASET, schema=SCHEMA, symbols='ALL_SYMBOLS',
                                          stype_in='raw_symbol', start=START, end=END))
    log(f'PRICED {START}..{END}: ${cost:.4f} (budget ${args.budget:.2f})')
    if args.cost_only:
        return 0
    if cost > args.budget:
        log(f'STOP: ${cost:.4f} exceeds budget ${args.budget:.2f}')
        return 2
    if os.path.exists(OUT):
        log(f'{OUT} exists — nothing to do')
        return 0

    dbn = f'{RAW}/xnas_ohlcv1d_2024H1.dbn.zst'
    if not os.path.exists(dbn):
        log('fetch')
        data = client.timeseries.get_range(dataset=DATASET, schema=SCHEMA, symbols='ALL_SYMBOLS',
                                           stype_in='raw_symbol', start=START, end=END)
        data.to_file(dbn)
        log(f'  -> {dbn} ({os.path.getsize(dbn) / 1e6:.1f} MB)')

    maps = [symbology_month(client, m0, m1) for m0, m1 in month_windows(START, END)]
    m = pd.concat(maps, ignore_index=True).drop_duplicates(['instrument_id', 'd0i'])
    m['d0i'] = m.d0i.astype('int64')
    m['d1i'] = m.d1i.astype('int64')
    m = m.sort_values(['d0i', 'instrument_id']).reset_index(drop=True)
    del maps
    log(f'symbology {len(m):,} intervals, {m.symbol.nunique():,} symbols')

    store = db.DBNStore.from_file(dbn)
    df = store.to_df(price_type='float').reset_index()
    ts = pd.to_datetime(df['ts_event'], utc=True)
    d = pd.DataFrame({'bar_date': ts.dt.strftime('%Y-%m-%d'),          # UTC date == session date
                      'instrument_id': df['instrument_id'].to_numpy('uint32'),
                      'open': df['open'].to_numpy('float64'),
                      'high': df['high'].to_numpy('float64'),
                      'low': df['low'].to_numpy('float64'),
                      'close': df['close'].to_numpy('float64'),
                      'volume': df['volume'].to_numpy('float64')})
    del df, store
    d['di'] = d.bar_date.str.replace('-', '', regex=False).astype('int64')
    d = d.sort_values(['di', 'instrument_id']).reset_index(drop=True)
    d = pd.merge_asof(d, m, left_on='di', right_on='d0i', by='instrument_id', direction='backward')
    n0 = len(d)
    d = d[d.di < d.d1i].drop(columns=['di', 'd0i', 'd1i']).dropna(subset=['symbol'])
    d = d[['bar_date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
    d = d.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
    d.to_parquet(OUT, index=False)
    log(f'{OUT}: {len(d):,}/{n0:,} rows kept, {d.symbol.nunique():,} symbols, '
        f'{d.bar_date.nunique()} sessions ({d.bar_date.min()}..{d.bar_date.max()})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
