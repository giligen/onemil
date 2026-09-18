#!/usr/bin/env python3
"""Stage N3 step 1c — repair the instrument_id -> symbol join in the daily parquets.

BUG (caught by the price-scale sanity check): XNAS.ITCH ohlcv-1d carries ts_event at 00:00 UTC of
the SESSION date, so converting it to America/New_York shifted every bar one calendar day earlier
(the parquet had Sundays and no Fridays).  ITCH re-assigns instrument_ids alphabetically EVERY DAY,
so joining with a one-day-old map silently renamed every symbol late in the alphabet by a few
positions: NVDA 2023-06-15 came out at $11.42 and SPY at $0.87 while AAPL (id 32) was correct.

Fix: session date = the UTC date of ts_event = stored bar_date + 1 day; re-resolve the (free)
symbology monthly and redo the as-of join.  Re-pull of the paid data is not needed — the parquets
keep instrument_id.
"""
from __future__ import annotations

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

N3 = 'research/fuckup_audit/N_databento/N3'
RAW = f'{N3}/raw_daily'
SYMDIR = f'{RAW}/sym'
CHUNKS = [('2018H2', '2018-05-01', '2019-01-01'), ('2019', '2019-01-01', '2020-01-01'),
          ('2020', '2020-01-01', '2021-01-01'), ('2021', '2021-01-01', '2022-01-01'),
          ('2022', '2022-01-01', '2023-01-01'), ('2023', '2023-01-01', '2024-01-01')]


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
    """Monthly symbology as a compact parquet (instrument_id, d0i, d1i, symbol). Free endpoint."""
    pq = f'{SYMDIR}/sym_{m0}.parquet'
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    for attempt in range(6):
        try:
            r = client.symbology.resolve(dataset='XNAS.ITCH', symbols='ALL_SYMBOLS',
                                         stype_in='raw_symbol', stype_out='instrument_id',
                                         start_date=m0, end_date=m1)
            break
        except Exception as exc:
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
    os.makedirs(SYMDIR, exist_ok=True)
    client = db.Historical(os.environ['DATABENTO_API_KEY'])
    for tag, s, e in CHUNKS:
        src = f'{RAW}/xnas_ohlcv1d_{tag}.parquet'
        dst = f'{RAW}/fixed_{tag}.parquet'
        if os.path.exists(dst):
            log(f'{dst} exists')
            continue
        maps = [symbology_month(client, m0, m1) for m0, m1 in month_windows(s, e)]
        m = pd.concat(maps, ignore_index=True).drop_duplicates(['instrument_id', 'd0i'])
        m['d0i'] = m.d0i.astype('int64')
        m['d1i'] = m.d1i.astype('int64')
        m = m.sort_values(['d0i', 'instrument_id']).reset_index(drop=True)
        del maps

        d = pd.read_parquet(src, columns=['bar_date', 'instrument_id', 'open', 'high', 'low',
                                          'close', 'volume'])
        # the stored bar_date is the ET date of a 00:00 UTC ts_event -> session date is +1 day
        d['bar_date'] = (pd.to_datetime(d.bar_date) + pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')
        d['di'] = d.bar_date.str.replace('-', '', regex=False).astype('int64')
        d = d.sort_values(['di', 'instrument_id']).reset_index(drop=True)
        d = pd.merge_asof(d, m, left_on='di', right_on='d0i', by='instrument_id', direction='backward')
        n0 = len(d)
        d = d[d.di < d.d1i].drop(columns=['di', 'd0i', 'd1i']).dropna(subset=['symbol'])
        d.to_parquet(dst, index=False)
        log(f'{dst}: {len(d):,}/{n0:,} rows kept, {d.symbol.nunique():,} symbols, '
            f'{d.bar_date.nunique()} sessions ({d.bar_date.min()}..{d.bar_date.max()})')
        del d, m
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
