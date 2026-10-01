"""Cell 1,684 — ORB frequency add-on pools: ideas 1, 2, 10, 11 (PREREG_1684.md, FROZEN).

Four admission-widening pools, each a mechanism + explicit code, evaluated through the
PRODUCTION pipeline (study_orb_pipeline_static_lock.py) at the LIVE config (catalyst veto OFF,
8 slots, spread gate, Q1 skip, 15:45 close), per-pool selection chain as cell 1,328 (own ranking,
own slot cap, excluded from the day's production picks).

Pools
  idea1  gap_pct in [3,5)% at open (reuses the existing wide-seed >=3% build; <3% extension is
         OUT OF SCOPE for this budget, noted in RESULT_1684.md) AND (range_high-prev_close)/
         prev_close*100 >= 5.0 by 09:35 -- "the gap gate misses a move that runs after the open".
  idea2  (open/prev_day_high - 1)*100 >= 3.0  (gap measured vs YESTERDAY's HIGH, not close) --
         always a subset of gap_vs_close>=3% (prev_high>=prev_close algebraically), so it is built
         from the SAME existing wide (>=3%) CSVs, no new minute bars needed.
  idea10 gap_pct <= -5.0% at open (gap-DOWN reversal; same long-breakout mechanic, break of the
         5-min range HIGH = short covering) -- population NOT in any existing gap-up build; fresh.
  idea11 day-2 continuation: YESTERDAY's gap_pct >= 10.0% for this symbol, TODAY any gap -- fresh.
All pools: price (today's open) in [3,30], prior-day volume >= 500,000 (production's own bands),
excluding rows already admissible to production that day (gap_pct >= 5.0, price in [3,30]) so a
pool only ever ADDS frequency; raw overlap is measured and reported before the exclusion.

Data: data/cache.db::daily_bars (read-only) for admission on both windows (it covers 2024-06-03..
2026-09-30, i.e. all of both target windows); minute bars for idea10/idea11 from
research/bf_zero/bars_sip.db (shared store, read + APPEND ONLY via the designated backfill
wrapper, research/bf_zero/backfill_bars_sip.py, called through a scratchpad wrapper that points
its FEATURES file at this cell's candidates instead of HOD's). idea1/idea2 reuse the already-built
wide-seed feature CSVs (gap>=3%, $3-50): research/orb_seed_wide/out/orb_features_20260920_2142.csv
(2025-01..2026-05) + .../orb_features_20260921_1842.csv (2026-06..09) -- no new minute-bar work.

Windows: IN-REGIME 2025-01-01..2026-09-18 (bounded by the latest already-built quarter data),
halves by calendar year (2025 / 2026). OUT-OF-REGIME requested as 2023-01..2024-12; built here only
for 2024-07-01..2024-12-31 (2024H2) because research/bf_zero/bars_sip.db starts 2024-07-01 and the
2023-01..2024-06 minute-bar store that research/orb_2023 used (bars.db) no longer exists on disk --
rebuilding it is outside this cell's 80-tool-call / one-process budget. The existing PRODUCTION
reference for 2023-01..2024-06 (research/orb_2023/book_1418_liveexit.csv) is reported for context
ONLY; no new pool is built for that half. This gap is stated plainly in RESULT_1684.md, not hidden.

Usage: python3 research/orb_freq/1684_pools.py --stage {candidates,backfill,features,pipeline,score,union,all}
"""
import argparse
import logging
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research/orb_freq'
SCRATCH = Path(os.environ.get('CLAUDE_SCRATCH', '/tmp')) / 'orb1684'
SCRATCH.mkdir(parents=True, exist_ok=True)
CACHE_DB = ROOT / 'data/cache.db'
BARS_SIP = ROOT / 'research/bf_zero/bars_sip.db'
WIDE_CSVS = [ROOT / 'research/orb_seed_wide/out/orb_features_20260920_2142.csv',
             ROOT / 'research/orb_seed_wide/out/orb_features_20260921_1842.csv']
R_USD = 375.0
PRICE_MIN, PRICE_MAX = 3.0, 30.0
PREV_VOL_MIN = 500_000
PROD_GAP_MIN = 5.0

IN_REGIME = ('2025-01-01', '2026-09-18')
OUT_REGIME = ('2024-07-01', '2024-12-31')  # 2024H2 only -- see module docstring

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('1684')


def _ro(path):
    return sqlite3.connect(f'file:{path}?mode=ro', uri=True)


def load_daily(symbols=None, start='2024-06-01', end='2026-09-30'):
    """daily_bars rows [start,end] (optionally restricted to `symbols`), with prev_close/prev_high/
    prev_volume (both shifted within symbol) and gap_pct computed. Read-only on cache.db."""
    con = _ro(CACHE_DB)
    if symbols:
        ph = ','.join('?' * len(symbols))
        q = f"SELECT symbol, bar_date, open, high, low, close, volume FROM daily_bars WHERE bar_date BETWEEN ? AND ? AND symbol IN ({ph})"
        params = [start, end] + list(symbols)
    else:
        q = "SELECT symbol, bar_date, open, high, low, close, volume FROM daily_bars WHERE bar_date BETWEEN ? AND ?"
        params = [start, end]
    df = pd.read_sql_query(q, con, params=params)
    con.close()
    df = df.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
    g = df.groupby('symbol')
    df['prev_close'] = g['close'].shift(1)
    df['prev_high'] = g['high'].shift(1)
    df['prev_volume'] = g['volume'].shift(1)
    df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close'] * 100
    df['gap_vs_high_pct'] = (df['open'] - df['prev_high']) / df['prev_high'] * 100
    return df


def band_ok(df):
    return (df['open'] >= PRICE_MIN) & (df['open'] <= PRICE_MAX) & (df['prev_volume'] >= PREV_VOL_MIN)


def is_production(df):
    return band_ok(df) & (df['gap_pct'] >= PROD_GAP_MIN)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all')
    args = ap.parse_args()
    log.info('stage=%s (library module; see other 1684_* scripts for each stage)', args.stage)
