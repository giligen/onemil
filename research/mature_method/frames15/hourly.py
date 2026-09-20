#!/usr/bin/env python3
"""frames15 step 1 — the PER-STOCK HOURLY VOLUME TABLE (the heavy step, checkpointed per month).

Reads `research/bf_zero/bars_sip.db` READ-ONLY, one session at a time, and writes one parquet per
month with, for every (symbol, session, ET session hour): the hour's RTH volume, its first open and
last close (for the hour's own return), and the session's whole RTH volume.

Session hours, ET: H9 = 09:30-09:59, H10..H15 = each clock hour to 15:59.
Nothing else is written. Resumable: `hourly_state.json` holds the finished months.
"""
import json
import os
import sqlite3
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
D = f'{ROOT}/research/mature_method/frames15'
SIP = f'{ROOT}/research/bf_zero/bars_sip.db'
ET = ZoneInfo('America/New_York')
STATE = f'{D}/hourly_state.json'


def et_offset_hours(day: str) -> int:
    """UTC-to-ET offset in whole hours for that session (constant inside an RTH session)."""
    t = datetime.fromisoformat(f'{day}T17:00:00+00:00').astimezone(ET)
    return -int(t.utcoffset().total_seconds() // 3600)


def one_day(con, day: str) -> pd.DataFrame:
    df = pd.read_sql('select symbol,t,o,c,v from bars where day=?', con, params=(day,))
    if df.empty:
        return df
    off = et_offset_hours(day)
    # 't' is ISO UTC: '2025-07-01T13:34:00+00:00'
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
    state = json.load(open(STATE)) if os.path.exists(STATE) else {'done': []}
    print(f'[hourly] {len(days)} sessions, {len(months)} months, {len(state["done"])} already done',
          flush=True)
    for mo in months:
        if mo in state['done']:
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
            df.to_parquet(f'{D}/hourly_{mo}.parquet', index=False)
            print(f'  {mo}: {len(df):>8,} sym-hours  {df.symbol.nunique():>5} symbols  '
                  f'{(datetime.now() - t0).total_seconds():5.1f}s', flush=True)
        state['done'].append(mo)
        json.dump(state, open(STATE, 'w'))
    con.close()
    print('[hourly] done', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
