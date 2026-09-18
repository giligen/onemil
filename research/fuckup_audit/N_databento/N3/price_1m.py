#!/usr/bin/env python3
"""Stage N3 step 2b pricing — the two 1-min pulls.

(a) RV window: ohlcv-1m 09:30-09:35 ET for the eligible pool (~1.5K names), one request per session.
(a2) same window but ALL_SYMBOLS (no pool approximation) — priced for comparison.
(b) simulation tape: ohlcv-1m 09:30-16:00 ET for the 20 selected names, one request per session.

Extrapolated over the 1,258 sessions of 2019-01-01..2023-12-31.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')
load_dotenv('/home/ec2-user/onemil/.env')

import databento as db  # noqa: E402

POOL = sys.argv[1] if len(sys.argv) > 1 else 'research/fuckup_audit/N_databento/N3/pool.parquet'
SESSIONS = 1258
SAMPLE_DAYS = ['2019-03-06', '2021-06-09', '2023-09-13']


def log(m):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {m}', flush=True)


def main() -> int:
    c = db.Historical(os.environ['DATABENTO_API_KEY'])
    pool = pd.read_parquet(POOL, columns=['bar_date', 'symbol'])
    pool['symbol'] = pool.symbol.astype(str)
    for label, get_syms, t0, t1 in [
            ('(a)  pool 09:30-09:35', lambda d: sorted(pool[pool.bar_date == d].symbol), 'T13:30', 'T13:35'),
            ('(a2) ALL_SYMBOLS 09:30-09:35', lambda d: 'ALL_SYMBOLS', 'T13:30', 'T13:35'),
            ('(b)  20 names 09:30-16:00', lambda d: sorted(pool[pool.bar_date == d].symbol)[:20],
             'T13:30', 'T21:00')]:
        tot = 0.0
        for d in SAMPLE_DAYS:
            syms = get_syms(d)
            n = len(syms) if isinstance(syms, list) else 0
            cost = float(c.metadata.get_cost(dataset='XNAS.ITCH', schema='ohlcv-1m', symbols=syms,
                                             stype_in='raw_symbol', start=f'{d}{t0}', end=f'{d}{t1}'))
            log(f'  {label} {d} (n={n}): ${cost:.5f}')
            tot += cost
        avg = tot / len(SAMPLE_DAYS)
        log(f'{label}: avg ${avg:.5f}/session -> ${avg * SESSIONS:.2f} over {SESSIONS} sessions')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
