#!/usr/bin/env python3
"""frames21 step 2 — the PER-STOCK HOURLY VOLUME TABLE for the 2024H2 extension, same logic as
frames15/hourly.py (one_day: session-hour RTH volume, first open, last close, day volume),
unchanged, only the source store (frames21/raw/bars_sip_ext.db) and output path differ.
Writes one parquet per month: frames21/hourly_ext_YYYY-MM.parquet.
"""
import os
import sqlite3
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
D = f'{ROOT}/research/mature_method/frames21'
SIP = f'{D}/raw/bars_sip_ext.db'
ET = ZoneInfo('America/New_York')


def et_offset_hours(day: str) -> int:
    t = datetime.fromisoformat(f'{day}T17:00:00+00:00').astimezone(ET)
    return -int(t.utcoffset().total_seconds() // 3600)


def one_day(con, day: str) -> pd.DataFrame:
    df = pd.read_sql('select symbol,t,o,c,v from bars where day=?', con, params=(day,))
    if df.empty:
        return df
    off = et_offset_hours(day)
    hh = df.t.str.slice(11, 13).astype(np.int16)
    mm = df.t.str.slice(14, 16).astype(np.int16)
    m_et = (hh - off) * 60 + mm
    df = df.assign(m=m_et.values)
    df = df[(df.m >= 570) & (df.m < 960)]
    if df.empty:
        return df
    df = df.sort_values(['symbol', 'm'], kind='mergesort')
    df['hour'] = np.where(df.m < 600, 9, (df.m // 60).astype(int))
    g = df.groupby(['symbol', 'hour'], sort=False)
    out = g.agg(v=('v', 'sum'), o=('o', 'first'), c=('c', 'last'), nb=('v', 'size')).reset_index()
    dayv = df.groupby('symbol', sort=False).v.sum().rename('day_v')
    out = out.merge(dayv, on='symbol', how='left')
    out['day'] = day
    return out


def main():
    con = sqlite3.connect(f'file:{SIP}?mode=ro', uri=True, timeout=120)
    days = [r[0] for r in con.execute('select distinct day from fetch_log order by day')]
    months = sorted({d[:7] for d in days})
    print(f'[hourly_ext] {len(days)} sessions, {len(months)} months', flush=True)
    for mo in months:
        out_path = f'{D}/hourly_ext_{mo}.parquet'
        if os.path.exists(out_path):
            continue
        t0 = datetime.now()
        parts = []
        for d in [x for x in days if x[:7] == mo]:
            r = one_day(con, d)
            if len(r):
                parts.append(r)
        if parts:
            df = pd.concat(parts, ignore_index=True)
            df['symbol'] = df.symbol.astype(str)
            df.to_parquet(out_path, index=False)
            print(f'  {mo}: {len(df):>8,} sym-hours  {df.symbol.nunique():>5} symbols  '
                  f'{(datetime.now() - t0).total_seconds():5.1f}s', flush=True)
    con.close()
    print('[hourly_ext] done', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
