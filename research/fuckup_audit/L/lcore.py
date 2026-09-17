#!/usr/bin/env python3
"""Stage L core — splits, the book, statistics, and the bar loader.

Every number in Stage L goes through this file.  The book rule is the repo's own
`trading.hod_break.run_book` (12 a day, 4 concurrent, first-come by entry minute, ties by symbol,
causal slot freeing) for B1-B4 and B6; B5 uses the ORB pipeline's own selection and never touches
`run_book`.  Costs are NOT recomputed here — each book arrives with its own `net`.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
from trading.hod_break import run_book                                        # noqa: E402

FA = f'{ROOT}/research/fuckup_audit'
L = f'{FA}/L'
BARS_SIP = f'{ROOT}/research/bf_zero/bars_sip.db'
CACHE_DB = f'{ROOT}/data/cache.db'
ETF_RET = f'{FA}/H/F14_F8_F11/etf_minute_ret.csv'
UNIVERSE = f'{ROOT}/research/bf_zero/universe.csv'

TEST_TICKER = r'^Z[A-Z]ZZT$'
SPLITS = (('TRAIN', '2025-01-01', '2025-12-31'),
          ('VAL', '2026-01-01', '2026-05-31'),
          ('TEST', '2026-06-01', '2099-12-31'))
BOOKS = ('B1', 'B2', 'B3', 'B4', 'B5', 'B6')
RISK_USD = 300.0            # B1-B4, B6 capacity (H/F6_sizing, I/REPORT)


def log(*a):
    print(*a, flush=True)


def split_of(day):
    """TRAIN / VAL / TEST for a YYYY-MM-DD string column."""
    d = pd.Series(day).astype(str)
    return np.where(d < '2026-01-01', 'TRAIN', np.where(d < '2026-06-01', 'VAL', 'TEST'))


def add_split_cols(d):
    d = d.copy()
    d['split'] = split_of(d.day)
    d['mon'] = d.day.astype(str).str[:7]
    d['wk'] = pd.to_datetime(d.day).dt.to_period('W-FRI').astype(str)
    return d


# ----------------------------------------------------------------------------- the book

def book(d, entry_col='entry_m', exit_col='exit_m'):
    """run_book(12, 4) over a population frame; returns the taken rows (same columns, original index)."""
    y = d[d[entry_col].notna() & d[exit_col].notna() & d['net'].notna()]
    if not len(y):
        return y
    rows = [(r[0], int(r[1]), int(r[2]), str(r[3]), i)
            for i, r in zip(y.index, y[['day', entry_col, exit_col, 'symbol']].itertuples(index=False))]
    taken = run_book(rows, 12, 4)
    out = d.loc[[r[4] for r in taken]].copy()
    # booked ordinal within the day, in the order the book took them
    out['ord'] = out.groupby('day').cumcount() + 1
    return out


def weeks_of(base, split):
    """The week index a split is scored on — fixed to the BASE book so filtered books cannot gain
    by deleting whole weeks from the denominator."""
    return sorted(base.loc[base.split == split, 'wk'].unique())


def stats(t, weeks, months):
    """Gate statistics of a booked frame over fixed week/month denominators."""
    if t is None or len(t) == 0:
        return dict(n=0, meanR=np.nan, t=np.nan, tpw=0.0, wkR=0.0, green=np.nan, worst=np.nan,
                    ex5=np.nan, cap3=np.nan, se=np.nan, mde=np.nan, WR=np.nan, totR=0.0, permo=0.0)
    v = np.sort(t.net.values.astype(float))
    n = len(v)
    sd = t.net.std(ddof=1) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 and sd == sd else np.nan
    w = t.groupby('wk').net.sum().reindex(weeks).fillna(0.0) if weeks else t.groupby('wk').net.sum()
    return dict(n=n, meanR=float(t.net.mean()),
                t=float(t.net.mean() / se) if se and se == se and se > 0 else np.nan,
                se=float(se) if se == se else np.nan,
                mde=float(2.8 * se) if se == se else np.nan,
                tpw=n / max(len(weeks), 1) if weeks else np.nan,
                WR=float((t.net > 0).mean() * 100),
                wkR=float(w.mean()), green=float((w > 0).mean()), worst=float(w.min()),
                ex5=float(v[:max(n - max(int(n * 0.05), 1), 1)].mean()),
                cap3=float(np.minimum(v, 3.0).mean()),
                totR=float(t.net.sum()),
                permo=float(t.net.sum() / max(months, 1) * RISK_USD))


def months_in(split):
    lo, hi = {'TRAIN': ('2025-01', '2025-12'), 'VAL': ('2026-01', '2026-05'),
              'TEST': ('2026-06', '2026-09')}[split]
    return (int(hi[:4]) - int(lo[:4])) * 12 + int(hi[5:]) - int(lo[5:]) + 1


# ----------------------------------------------------------------------------- bars

class Bars:
    """(symbol, day) -> DataFrame[m, o, h, l, c, v] over 09:30-15:59 ET.

    Source precedence declared in the Stage-L brief: `bars_sip.db` first, `data/cache.db` fallback.
    Both opened read-only.  A tiny LRU keeps the last key only (the walks are sorted by key)."""

    def __init__(self):
        self.sip = sqlite3.connect(f'file:{BARS_SIP}?mode=ro', uri=True)
        self.cache = sqlite3.connect(f'file:{CACHE_DB}?mode=ro', uri=True)
        self.n_sip = self.n_cache = self.n_none = 0

    def get(self, sym, day):
        r = self.sip.execute('select t, o, h, l, c, v from bars where symbol=? and day=? order by t',
                             (sym, day)).fetchall()
        src = 'sip'
        if not r:
            r = self.cache.execute('select timestamp, open, high, low, close, volume from '
                                   'intraday_bars_1min where symbol=? and bar_date=? order by timestamp',
                                   (sym, day)).fetchall()
            src = 'cache'
        if not r:
            self.n_none += 1
            return None
        if src == 'sip':
            self.n_sip += 1
        else:
            self.n_cache += 1
        b = pd.DataFrame(r, columns=['t', 'o', 'h', 'l', 'c', 'v'])
        ts = pd.to_datetime(b.t, utc=True, format='mixed').dt.tz_convert('America/New_York')
        b['m'] = ts.dt.hour * 60 + ts.dt.minute
        b = b[(b.m >= 570) & (b.m < 960)]
        return b.drop(columns=['t']).set_index('m')

    def close(self):
        self.sip.close()
        self.cache.close()


def spy_minute_ret():
    """(day, minute) -> SPY return since that day's 09:30 open, in %. Cached table built in Stage H."""
    b = pd.read_csv(ETF_RET, keep_default_na=False, na_values=[''], dtype={'day': str, 'symbol': str})
    s = b[b.symbol == 'SPY'][['day', 'm', 'ret']]
    return {(d, int(m)): float(r) for d, m, r in s.itertuples(index=False)}


def prev_day_range_map():
    """(symbol, day) -> previous trading day's range as % of its close, from the point-in-time universe."""
    u = pd.read_csv(UNIVERSE, keep_default_na=False, na_values=[''],
                    usecols=['symbol', 'bar_date', 'high', 'low', 'close'], dtype={'symbol': str, 'bar_date': str})
    u = u.sort_values(['symbol', 'bar_date'])
    g = u.groupby('symbol')
    pr = (g.high.shift(1) - g.low.shift(1)) / g.close.shift(1) * 100.0
    ok = pr.notna()
    return {(s, d): float(v) for s, d, v in zip(u.symbol[ok], u.bar_date[ok], pr[ok])}
