#!/usr/bin/env python3
"""Pass 2 step 1: build the full underlying universe + candidate underlying-days.
DRY — reads cache.db read-only, writes only into research/lev_rebalance/.
No Alpaca calls here.
"""
import csv
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date

import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from trading.orb_asset_class import underlying_anchor, load_class_map, WRAPPER  # noqa: E402
from research.scripts.pit_listings import is_test_ticker  # noqa: E402

OUT = os.path.join(ROOT, 'research', 'lev_rebalance')
DB = f"file:{ROOT}/data/cache.db?mode=ro"
CLASS_MAP_PATH = os.path.join(ROOT, 'data', 'research', 'orb_asset_class_map_20260711.csv')

TRAIN_LO, TRAIN_HI = date(2025, 1, 1), date(2025, 12, 31)
VAL_LO, VAL_HI = date(2026, 1, 1), date(2026, 5, 31)
RUN_LO, RUN_HI = TRAIN_LO, VAL_HI
MIN_PRICE = 5.0
SIG_THRESH = 0.05

conn = sqlite3.connect(DB, uri=True, timeout=30)


def log(*a):
    print(*a, flush=True)


def main():
    log('=== build full wrapper->underlying map ===')
    class_map = load_class_map(CLASS_MAP_PATH)
    rows = []
    with open(CLASS_MAP_PATH, newline='') as fh:
        for row in csv.DictReader(fh):
            if row['asset_class'] == WRAPPER:
                rows.append(row)
    log(f'wrapper rows in class map: {len(rows)}')

    underlying_to_wrappers = defaultdict(set)
    unresolved = 0
    for row in rows:
        sym = row['symbol']
        name = row['name']
        if is_test_ticker(sym):
            continue
        anchor = underlying_anchor(sym, name, class_map)
        if anchor is None or anchor == sym:
            unresolved += 1
            continue
        underlying_to_wrappers[anchor].add(sym)
    log(f'resolved underlyings: {len(underlying_to_wrappers)}; unresolved wrapper rows: {unresolved}')

    with open(os.path.join(OUT, 'universe_map.json'), 'w') as f:
        json.dump({k: sorted(v) for k, v in underlying_to_wrappers.items()}, f, indent=2)

    underlyings = sorted(underlying_to_wrappers.keys())
    underlyings = [u for u in underlyings if not is_test_ticker(u)]
    log(f'underlyings (pre daily_bars check): {len(underlyings)}')

    # Which underlyings actually have daily_bars rows in range, price>=5 at some point
    q = ("SELECT symbol, bar_date, open, high, low, close, volume FROM daily_bars "
         "WHERE symbol IN ({}) AND bar_date >= ? AND bar_date <= ? ORDER BY symbol, bar_date")
    CH = 400
    all_daily = []
    for i in range(0, len(underlyings), CH):
        chunk = underlyings[i:i + CH]
        qq = q.format(','.join('?' * len(chunk)))
        df = pd.read_sql_query(qq, conn, params=[*chunk, str(date(2024, 12, 1)), str(RUN_HI)])
        all_daily.append(df)
    daily = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    daily['bar_date'] = pd.to_datetime(daily['bar_date']).dt.date
    daily = daily.sort_values(['symbol', 'bar_date'])
    daily['prior_close'] = daily.groupby('symbol')['close'].shift(1)
    log(f'daily_bars rows pulled for underlyings: {len(daily)}; distinct symbols with rows: {daily["symbol"].nunique()}')

    # candidate underlying-days: pull filter = |high/prior_close-1|>=5% OR |low/prior_close-1|>=5%, price>=5, in TRAIN..VAL
    d = daily[(daily['bar_date'] >= RUN_LO) & (daily['bar_date'] <= RUN_HI)].copy()
    d = d[d['prior_close'].notna() & (d['prior_close'] >= MIN_PRICE)]
    hi_move = (d['high'] / d['prior_close'] - 1.0).abs()
    lo_move = (d['low'] / d['prior_close'] - 1.0).abs()
    d['pull_qualifies'] = (hi_move >= SIG_THRESH) | (lo_move >= SIG_THRESH)
    cand = d[d['pull_qualifies']].copy()
    log(f'candidate underlying-days (pull filter, superset of signal): {len(cand)}')
    log(f'distinct underlyings with >=1 candidate day: {cand["symbol"].nunique()}')
    log(f'distinct candidate days (calendar): {cand["bar_date"].nunique()}')

    cand[['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume', 'prior_close']].to_csv(
        os.path.join(OUT, 'candidate_underlying_days.csv'), index=False)

    # Already covered by cache.db intraday_bars_1min? Check per (symbol,date) whether
    # bars exist spanning 14:59-15:01 ET (cheap existence check via bar_date + symbol only,
    # exact time-window check done in the fetch step).
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
    log(f'candidate days already present (symbol,date) in cache.db intraday_bars_1min: {n_cov} / {len(cand)} ({n_cov/max(len(cand),1):.1%})')

    need = cand[~cand['already_in_cache']]
    log(f'candidate underlying-days needing an Alpaca pull: {len(need)}')
    need[['symbol', 'bar_date']].to_csv(os.path.join(OUT, 'need_pull.csv'), index=False)

    summary = dict(
        n_wrapper_rows=len(rows), n_underlyings_resolved=len(underlying_to_wrappers),
        n_candidate_underlying_days=int(len(cand)),
        n_distinct_underlyings_with_candidates=int(cand['symbol'].nunique()),
        n_distinct_candidate_calendar_days=int(cand['bar_date'].nunique()),
        n_already_cached=n_cov, n_need_pull=int(len(need)),
    )
    with open(os.path.join(OUT, 'universe_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log('SUMMARY', json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
