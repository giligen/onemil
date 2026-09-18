#!/usr/bin/env python3
"""Stage N3 step 1 — XNAS.ITCH ohlcv-1d ALL_SYMBOLS 2018-05-01 -> 2024-01-01 -> N3/xnas_daily.parquet.

Pulled in yearly chunks (resumable, memory-bounded).  ALL_SYMBOLS DBN carries no symbol mappings,
so instrument_id -> raw_symbol is resolved per chunk with the free symbology endpoint.

XNAS.ITCH is the Nasdaq TotalView-ITCH feed: Nasdaq-exchange executions only.  Volumes are a
venue subset of consolidated tape volume (documented as an approximation in REPORT.md).

Usage:  python3 fetch_daily.py --cost-only | python3 fetch_daily.py --budget 30
"""
from __future__ import annotations

import argparse
import json
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
N3 = 'research/fuckup_audit/N_databento/N3'
RAW = f'{N3}/raw_daily'
OUT = f'{N3}/xnas_daily.parquet'
CHUNKS = [('2018-05-01', '2019-01-01'), ('2019-01-01', '2020-01-01'), ('2020-01-01', '2021-01-01'),
          ('2021-01-01', '2022-01-01'), ('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01')]


def month_windows(s, e):
    """[s, e) split into calendar-month [start, end) pairs (ISO date strings)."""
    out, cur = [], pd.Timestamp(s)
    end = pd.Timestamp(e)
    while cur < end:
        nxt = min((cur + pd.offsets.MonthBegin(1)), end)
        out.append((cur.strftime('%Y-%m-%d'), nxt.strftime('%Y-%m-%d')))
        cur = nxt
    return out


def log(m):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {m}', flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cost-only', action='store_true')
    ap.add_argument('--budget', type=float, default=30.0)
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)

    client = db.Historical(os.environ['DATABENTO_API_KEY'])
    total = 0.0
    for s, e in CHUNKS:
        c = float(client.metadata.get_cost(dataset=DATASET, schema=SCHEMA, symbols='ALL_SYMBOLS',
                                           stype_in='raw_symbol', start=s, end=e))
        log(f'PRICED {s}..{e}: ${c:.4f}')
        total += c
    log(f'PRICED TOTAL: ${total:.4f} (budget ${args.budget:.2f})')
    if args.cost_only:
        return 0
    if total > args.budget:
        log(f'STOP: ${total:.2f} exceeds budget')
        return 2

    if os.path.exists(OUT):
        log(f'{OUT} exists — nothing to do')
        return 0

    parts = []
    for s, e in CHUNKS:
        tag = s[:4] if s != '2018-05-01' else '2018H2'
        dbn = f'{RAW}/xnas_ohlcv1d_{tag}.dbn.zst'
        pq = f'{RAW}/xnas_ohlcv1d_{tag}.parquet'
        if not os.path.exists(pq):
            if not os.path.exists(dbn):
                log(f'fetch {s}..{e}')
                data = client.timeseries.get_range(dataset=DATASET, schema=SCHEMA, symbols='ALL_SYMBOLS',
                                                   stype_in='raw_symbol', start=s, end=e)
                data.to_file(dbn)
                log(f'  -> {dbn} ({os.path.getsize(dbn) / 1e6:.1f} MB)')
            rows = []
            for m0, m1 in month_windows(s, e):
                symjson = f'{RAW}/symbology_{tag}_{m0}.json'
                if not os.path.exists(symjson):
                    for attempt in range(5):
                        try:
                            log(f'resolve symbology {m0}..{m1}')
                            r = client.symbology.resolve(
                                dataset=DATASET, symbols='ALL_SYMBOLS', stype_in='raw_symbol',
                                stype_out='instrument_id', start_date=m0, end_date=m1)
                            json.dump(r, open(symjson, 'w'))
                            break
                        except Exception as exc:  # 504s are common on wide windows
                            log(f'  symbology {m0} attempt {attempt + 1} failed: {str(exc)[:80]}')
                            time.sleep(15 * (attempt + 1))
                    else:
                        log(f'FATAL: symbology {m0}..{m1} unavailable')
                        return 3
                j = json.load(open(symjson))['result']
                for sym, ivs in j.items():
                    for iv in ivs:
                        rows.append((int(iv['s']), int(iv['d0'].replace('-', '')),
                                     int(iv['d1'].replace('-', '')), sym))
                del j
            m = pd.DataFrame(rows, columns=['instrument_id', 'd0i', 'd1i', 'symbol'])
            m = m.drop_duplicates(['instrument_id', 'd0i'])
            del rows
            m['instrument_id'] = m.instrument_id.astype('uint32')
            m['d0i'] = m.d0i.astype('int64'); m['d1i'] = m.d1i.astype('int64')
            m = m.sort_values(['d0i', 'instrument_id']).reset_index(drop=True)
            log(f'  symbology {len(m):,} intervals, {m.symbol.nunique():,} symbols')

            store = db.DBNStore.from_file(dbn)
            df = store.to_df(price_type='float').reset_index()
            ts = pd.to_datetime(df['ts_event'], utc=True)
            d = pd.DataFrame({'bar_date': ts.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d'),
                              'instrument_id': df['instrument_id'].to_numpy('uint32'),
                              'open': df['open'].to_numpy('float64'),
                              'high': df['high'].to_numpy('float64'),
                              'low': df['low'].to_numpy('float64'),
                              'close': df['close'].to_numpy('float64'),
                              'volume': df['volume'].to_numpy('float64')})
            del df, store
            # instrument ids are re-issued daily on ITCH -> asof-join on the interval start,
            # then drop rows past the interval end.  A plain merge explodes (146M rows).
            d['di'] = d.bar_date.str.replace('-', '', regex=False).astype('int64')
            d = d.sort_values(['di', 'instrument_id']).reset_index(drop=True)
            d = pd.merge_asof(d, m, left_on='di', right_on='d0i', by='instrument_id',
                              direction='backward')
            d = d[d.di < d.d1i]
            d = d.drop(columns=['di', 'd0i', 'd1i']).dropna(subset=['symbol'])
            d.to_parquet(pq, index=False)
            log(f'  -> {pq}: {len(d):,} rows, {d.symbol.nunique():,} symbols, '
                f'{d.bar_date.nunique()} dates')
            del d
        parts.append(pq)

    log('concat')
    all_df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    all_df = all_df.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
    all_df.to_parquet(OUT, index=False)
    log(f'{OUT}: {len(all_df):,} rows, {all_df.symbol.nunique():,} symbols, '
        f'{all_df.bar_date.min()}..{all_df.bar_date.max()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
