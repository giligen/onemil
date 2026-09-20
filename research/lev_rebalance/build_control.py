#!/usr/bin/env python3
"""Pass 2 step 2b: control pool candidate underlying-days (movers WITHOUT
any wrapper), sized up from pass 1's 60 symbols since the pull is cheap.
DRY — writes control_candidate_days.csv + control_need_pull.csv; no Alpaca
calls here (pull_bars.py is run separately against control_need_pull.csv).
"""
import csv
import json
import os
import random
import sqlite3
import sys
from datetime import date

import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from research.scripts.pit_listings import is_test_ticker  # noqa: E402

OUT = os.path.join(ROOT, 'research', 'lev_rebalance')
DB = f"file:{ROOT}/data/cache.db?mode=ro"
CLASS_MAP_PATH = os.path.join(ROOT, 'data', 'research', 'orb_asset_class_map_20260711.csv')

RUN_LO, RUN_HI = date(2025, 1, 1), date(2026, 5, 31)
MIN_PRICE = 5.0
SIG_THRESH = 0.05
N_CONTROL_SAMPLE = 400
CONTROL_SEED = 1291

conn = sqlite3.connect(DB, uri=True, timeout=30)


def log(*a):
    print(*a, flush=True)


def main():
    m = json.load(open(os.path.join(OUT, 'universe_map.json')))
    wrappers = {w for ws in m.values() for w in ws}
    underlyings = set(m.keys())
    exclude = wrappers | underlyings

    stocks = []
    with open(CLASS_MAP_PATH, newline='') as fh:
        for row in csv.DictReader(fh):
            s = row['symbol']
            if row['asset_class'] == 'stock' and s not in exclude and not is_test_ticker(s) \
               and s.isalpha() and len(s) <= 5:
                stocks.append(s)
    rng = random.Random(CONTROL_SEED)
    control_syms = sorted(rng.sample(stocks, min(N_CONTROL_SAMPLE, len(stocks))))
    log(f'control pool: {len(control_syms)} symbols (seed={CONTROL_SEED}) of {len(stocks)} eligible stock rows')

    q = ("SELECT symbol, bar_date, open, high, low, close, volume FROM daily_bars "
         "WHERE symbol IN ({}) AND bar_date >= ? AND bar_date <= ? ORDER BY symbol, bar_date")
    CH = 400
    parts = []
    for i in range(0, len(control_syms), CH):
        chunk = control_syms[i:i + CH]
        qq = q.format(','.join('?' * len(chunk)))
        df = pd.read_sql_query(qq, conn, params=[*chunk, str(date(2024, 12, 1)), str(RUN_HI)])
        parts.append(df)
    daily = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    daily['bar_date'] = pd.to_datetime(daily['bar_date']).dt.date
    daily = daily.sort_values(['symbol', 'bar_date'])
    daily['prior_close'] = daily.groupby('symbol')['close'].shift(1)
    log(f'control daily_bars rows: {len(daily)}, symbols with rows: {daily["symbol"].nunique()}')

    d = daily[(daily['bar_date'] >= RUN_LO) & (daily['bar_date'] <= RUN_HI)].copy()
    d = d[d['prior_close'].notna() & (d['prior_close'] >= MIN_PRICE)]
    hi_move = (d['high'] / d['prior_close'] - 1.0).abs()
    lo_move = (d['low'] / d['prior_close'] - 1.0).abs()
    d['pull_qualifies'] = (hi_move >= SIG_THRESH) | (lo_move >= SIG_THRESH)
    cand = d[d['pull_qualifies']].copy()
    log(f'control candidate underlying-days: {len(cand)}; distinct symbols: {cand["symbol"].nunique()}')

    cand[['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume', 'prior_close']].to_csv(
        os.path.join(OUT, 'control_candidate_days.csv'), index=False)

    syms = sorted(cand['symbol'].unique().tolist())
    covered = set()
    for i in range(0, len(syms), 400):
        chunk = syms[i:i + 400]
        qq = ("SELECT DISTINCT symbol, bar_date FROM intraday_bars_1min WHERE symbol IN ({}) "
              "AND bar_date >= ? AND bar_date <= ?").format(','.join('?' * len(chunk)))
        cur = conn.execute(qq, [*chunk, str(RUN_LO), str(RUN_HI)])
        for sym, bd in cur.fetchall():
            covered.add((sym, bd))
    cand['already_in_cache'] = cand.apply(lambda r: (r['symbol'], str(r['bar_date'])) in covered, axis=1)
    n_cov = int(cand['already_in_cache'].sum())
    log(f'control days already in cache.db: {n_cov} / {len(cand)} ({n_cov/max(len(cand),1):.1%})')

    need = cand[~cand['already_in_cache']]
    need[['symbol', 'bar_date']].to_csv(os.path.join(OUT, 'control_need_pull.csv'), index=False)
    log(f'control days needing Alpaca pull: {len(need)}')


if __name__ == '__main__':
    main()
