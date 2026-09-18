#!/usr/bin/env python3
"""Stage N3 step 2c/3b — the two XNAS.ITCH ohlcv-1m pulls, resumable, keyed by session date.

  window : ALL_SYMBOLS 09:30-09:35 ET, one request per session, only pool symbols stored.
           -> tape.db::window (the paper's ORVolume and the first 5-min candle)
  tape   : the 20 selected names 09:30-16:00 ET, one request per session.
           -> tape.db::bars (the simulation tape)

ET->UTC is computed per session with zoneinfo (EST/EDT differ by an hour — a fixed 13:30 UTC would
silently read 08:30 or 10:30 ET for half the year).

Usage: fetch_1m.py window --budget 12 | fetch_1m.py tape --budget 8
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')
load_dotenv('/home/ec2-user/onemil/.env')

import databento as db  # noqa: E402

ET = ZoneInfo('America/New_York')
N3 = 'research/fuckup_audit/N_databento/N3'
DB = f'{N3}/tape.db'
DAILY = f'{N3}/xnas_daily.parquet'
POOL = f'{N3}/pool.parquet'
TOP20 = f'{N3}/top20.csv'
WINDOW_START = '2018-11-15'          # 14 sessions of RV history before 2019-01-02
WINDOW_END = '2023-12-31'


def log(m):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {m}', flush=True)


def utc_window(day: str, h0: int, m0: int, h1: int, m1: int):
    d = datetime.strptime(day, '%Y-%m-%d')
    a = d.replace(hour=h0, minute=m0, tzinfo=ET).astimezone(timezone.utc)
    b = d.replace(hour=h1, minute=m1, tzinfo=ET).astimezone(timezone.utc)
    return a.strftime('%Y-%m-%dT%H:%M'), b.strftime('%Y-%m-%dT%H:%M')


def connect():
    con = sqlite3.connect(DB, timeout=120)
    con.execute('create table if not exists window (day text, symbol text, o5 real, h5 real, '
                'l5 real, c5 real, v5 real, nbars integer, primary key (day, symbol))')
    con.execute('create table if not exists window_done (day text primary key, n integer)')
    con.execute('create table if not exists bars (day text, symbol text, m integer, o real, '
                'h real, l real, c real, v real, primary key (day, symbol, m))')
    con.execute('create table if not exists bars_done (day text primary key, n integer)')
    con.commit()
    return con


def get_cost(client, symbols, stype, start, end, retries=6):
    for attempt in range(retries):
        try:
            return float(client.metadata.get_cost(dataset='XNAS.ITCH', schema='ohlcv-1m',
                                                  symbols=symbols, stype_in=stype,
                                                  start=start, end=end))
        except Exception as exc:
            log(f'  get_cost attempt {attempt + 1} failed: {str(exc)[:80]}')
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f'get_cost failed {start}..{end}')


def get_range(client, symbols, stype, start, end, retries=5):
    for attempt in range(retries):
        try:
            data = client.timeseries.get_range(dataset='XNAS.ITCH', schema='ohlcv-1m',
                                               symbols=symbols, stype_in=stype, start=start, end=end)
            return data.to_df(price_type='float')
        except Exception as exc:
            log(f'  get_range attempt {attempt + 1} failed: {str(exc)[:90]}')
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f'get_range failed {start}..{end}')


def sessions(lo, hi):
    d = pd.read_parquet(DAILY, columns=['bar_date'])
    s = pd.Index(sorted(d.bar_date.astype(str).unique()))
    return [x for x in s if lo <= x <= hi]


def to_et_minute(df):
    ts = pd.to_datetime(df['ts_event'], utc=True).dt.tz_convert(ET)
    return (ts.dt.hour * 60 + ts.dt.minute).to_numpy()


class SymMap:
    """instrument_id -> raw_symbol for one session.  ALL_SYMBOLS responses carry no symbol
    mappings and ITCH re-assigns ids alphabetically every day, so the map is per date."""

    SYMDIR = f'{N3}/raw_daily/sym'

    def __init__(self):
        self._month = None
        self._m = None
        self._day = None
        self._d = None

    def _load_month(self, day):
        key = day[:7]
        if self._month == key:
            return
        f = f'{self.SYMDIR}/sym_{key}-01.parquet'
        if not os.path.exists(f):
            raise FileNotFoundError(f'{f} — run fix_symbol_map.py first')
        m = pd.read_parquet(f)
        m['d0i'] = m.d0i.astype('int64')
        m['d1i'] = m.d1i.astype('int64')
        self._m, self._month, self._day = m, key, None

    def for_day(self, day):
        if self._day == day:
            return self._d
        self._load_month(day)
        di = int(day.replace('-', ''))
        s = self._m[(self._m.d0i <= di) & (di < self._m.d1i)]
        self._d = dict(zip(s.instrument_id.to_numpy('int64'), s.symbol.astype(str)))
        self._day = day
        return self._d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['window', 'tape'])
    ap.add_argument('--budget', type=float, required=True)
    args = ap.parse_args()

    client = db.Historical(os.environ['DATABENTO_API_KEY'])
    con = connect()
    smap = SymMap()

    if args.step == 'window':
        pool_syms = set(pd.read_parquet(POOL, columns=['symbol']).symbol.astype(str).unique())
        log(f'pool symbol set: {len(pool_syms):,}')
        days = sessions(WINDOW_START, WINDOW_END)
        done = {r[0] for r in con.execute('select day from window_done')}
        todo = [d for d in days if d not in done]
        log(f'window: {len(days)} sessions, {len(todo)} to fetch')
        spent = 0.0
        for k, day in enumerate(todo):
            s, e = utc_window(day, 9, 30, 9, 35)
            cost = get_cost(client, 'ALL_SYMBOLS', 'raw_symbol', s, e)
            if spent + cost > args.budget:
                log(f'STOP: budget ${args.budget} reached after ${spent:.2f}')
                return 2
            df = get_range(client, 'ALL_SYMBOLS', 'raw_symbol', s, e)
            spent += cost
            if len(df):
                df = df.reset_index()
                df['symbol'] = df['instrument_id'].astype('int64').map(smap.for_day(day))
                df = df[df.symbol.isin(pool_syms)]
            if len(df):
                df['m'] = to_et_minute(df)
                df = df[(df.m >= 570) & (df.m < 575)].sort_values(['symbol', 'm'])
                g = df.groupby('symbol', sort=False)
                agg = pd.DataFrame({'o5': g.open.first(), 'h5': g.high.max(), 'l5': g.low.min(),
                                    'c5': g.close.last(), 'v5': g.volume.sum(),
                                    'nbars': g.size()}).reset_index()
                rows = [(day, r.symbol, float(r.o5), float(r.h5), float(r.l5), float(r.c5),
                         float(r.v5), int(r.nbars)) for r in agg.itertuples()]
            else:
                rows = []
            con.executemany('insert or replace into window values (?,?,?,?,?,?,?,?)', rows)
            con.execute('insert or replace into window_done values (?,?)', (day, len(rows)))
            con.commit()
            if k % 50 == 0 or k == len(todo) - 1:
                log(f'  [{k + 1}/{len(todo)}] {day}: {len(rows)} symbols, spent ${spent:.3f}')
        log(f'window done, spent ${spent:.3f}')
        return 0

    # tape
    top = pd.read_csv(TOP20, dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    done = {r[0] for r in con.execute('select day from bars_done')}
    days = sorted(top.day.unique())
    todo = [d for d in days if d not in done]
    log(f'tape: {len(days)} sessions, {len(todo)} to fetch')
    spent = 0.0
    for k, day in enumerate(todo):
        syms = sorted(top.loc[top.day == day, 'symbol'].unique())
        s, e = utc_window(day, 9, 30, 16, 0)
        cost = get_cost(client, syms, 'raw_symbol', s, e)
        if spent + cost > args.budget:
            log(f'STOP: budget ${args.budget} reached after ${spent:.2f}')
            return 2
        df = get_range(client, syms, 'raw_symbol', s, e)
        spent += cost
        rows = []
        if len(df):
            df = df.reset_index()
            df['symbol'] = df['instrument_id'].astype('int64').map(smap.for_day(day))
            df = df.dropna(subset=['symbol'])
            df['m'] = to_et_minute(df)
            df = df[(df.m >= 570) & (df.m < 960)]
            rows = [(day, r.symbol, int(r.m), float(r.open), float(r.high), float(r.low),
                     float(r.close), float(r.volume)) for r in df.itertuples()]
        con.executemany('insert or replace into bars values (?,?,?,?,?,?,?,?)', rows)
        con.execute('insert or replace into bars_done values (?,?)', (day, len(rows)))
        con.commit()
        if k % 50 == 0 or k == len(todo) - 1:
            log(f'  [{k + 1}/{len(todo)}] {day}: {len(syms)} syms / {len(rows)} bars, '
                f'spent ${spent:.3f}')
    log(f'tape done, spent ${spent:.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
