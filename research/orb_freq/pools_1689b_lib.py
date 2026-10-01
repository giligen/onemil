"""Cell 1,689b shared library -- ORB frequency sub-pools 21-30 (PREREG_1684.md amendment 2,
owner 10/1 ~10:00 UTC "get me more sub-pools"). Reuses the 1,684/1,685/1,689a harness verbatim
where possible (pools_1689a_lib's daily-panel loaders, 1684_score.py's stats/union/cadence).

Two already-on-disk seeds, per the owner's order-of-work:
  * "2-5% gapper seed" = WIDE_CSVS (in-regime, gap>=3% $3-50, research/orb_seed_wide/out/) UNION
    OUT_REGIME_WIDE_CSV (2024H2 equivalent, built for cell 1,684's idea1_pre|idea2|idea10|idea11).
    Empirically (checked before writing this file): gap in [2,3) is a <150-row sliver in BOTH
    windows (101/22,606 in-regime, 88/5,539 out-regime) -- the same artifact 1685_subpools.py found
    and scoped band A (2-3%) out of every F2/pre-market pool for. This cell uses gap>=3% for pools
    24/25/26/30 and states the 2-3% sub-band is not covered by any on-disk minute-bar build (same
    precedent, not hidden).
  * "the >=5% 100K-500K slice" = cell 1,689a's own 19' population (subpools_1689a/19PRIME_*).
Pools 21/27/28 need a FRESH candidate list + backfill + feature build (price bands outside both
seeds' $1-50 coverage, or a volume floor the seeds don't carry); see 1689b_pools.py --stage build.

WIDE feature CSVs carry ONLY derived %s (range_size_pct, gap_pct, prev_day_close_position, ...),
NOT the raw day's open/price or prior-day OHLC -- confirmed by reading study_orb_features.py's
extract_features() (range_high/range_low/open_p/close_p are LOCAL variables, never written to the
row) and by the CSV header itself. Every admission check in this cell either (a) joins back to the
daily-bar panel for open/prev_volume/prev2_close/prev_high/prev_low, or (b) queries bars_sip.db
directly for the same 09:30-09:35 range bars and 04:00-09:30 pre-market bars production uses (same
temp-table join trick as 1689a's _premarket_dollar_volume, same RANGE_MINUTES=5 window as
study_orb_features.extract_features), never a guess at an existing column's semantics.
"""
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research/orb_freq'
POOLDIR = OUT / 'subpools_1689b'
POOLDIR.mkdir(exist_ok=True)
sys.path.insert(0, str(OUT))

CACHE_DB = ROOT / 'data/cache.db'
BARS_SIP = ROOT / 'research/bf_zero/bars_sip.db'
EQUS_2024H2 = ROOT / 'data/research/databento/equs_daily_2024H2.parquet'
EQUS_2025_2026 = ROOT / 'data/research/databento/equs_daily_2025_2026.parquet'
WIDE_CSVS = [ROOT / 'research/orb_seed_wide/out/orb_features_20260920_2142.csv',
             ROOT / 'research/orb_seed_wide/out/orb_features_20260921_1842.csv']
OUT_REGIME_WIDE_CSV = OUT / 'out_out_regime/orb_features_20261001_0757.csv'
DAILY_SRC_IN = OUT / 'daily_source_in_regime.parquet'     # already built by 1684_features.py
DAILY_SRC_OUT = OUT / 'daily_source_out_regime.parquet'   # already built by 1684_features.py

R_USD = 375.0
IN_REGIME = ('2025-01-01', '2026-09-26')
OUT_REGIME = ('2024-07-01', '2024-12-31')   # 2024H2 only -- bars_sip.db starts 2024-07-01

# A plain logging.basicConfig(force=True) is NOT import-order-safe here: pools_1689a_lib.py (which
# load_panel_with_lags() imports lazily) does its OWN force=True basicConfig() call, which silently
# RETARGETS every logger using the root's handlers -- including this one -- to 1689a_slice.log the
# first time that module loads (verified directly: this cell's own INFO lines appeared in
# 1689a_slice.log, not 1689b_pools.log, on the first run). Attaching handlers directly to OUR named
# logger and disabling propagation makes it immune to any later basicConfig() elsewhere.
log = logging.getLogger('1689b')
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    _fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    for _h in (logging.StreamHandler(), logging.FileHandler(OUT / '1689b_pools.log', mode='a')):
        _h.setFormatter(_fmt)
        log.addHandler(_h)


def _ro(p):
    return sqlite3.connect(f'file:{p}?mode=ro', uri=True)


def _clean_symbol(s):
    return s.astype(str).str.replace(r'\+$', '.WS', regex=True)


def load_panel_with_lags(window):
    """window in {'in_regime','out_regime'}. Reuses pools_1689a_lib's own cache.db UNION databento
    loaders verbatim (source-of-record conventions unchanged), adds prev_high/prev_low/prev2_close
    so pools can read YESTERDAY's own return/range position (pool 23), not just prev_close/volume."""
    from pools_1689a_lib import load_in_regime_panel, load_out_regime_panel
    panel, xcheck = (load_in_regime_panel() if window == 'in_regime' else load_out_regime_panel())
    panel = panel.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
    g = panel.groupby('symbol')
    panel['prev_high'] = g['high'].shift(1)
    panel['prev_low'] = g['low'].shift(1)
    panel['prev2_close'] = g['close'].shift(2)
    panel['yday_return_pct'] = (panel['prev_close'] - panel['prev2_close']) / panel['prev2_close'] * 100
    prev_range = panel['prev_high'] - panel['prev_low']
    panel['yday_close_position'] = np.where(prev_range > 0,
                                             (panel['prev_close'] - panel['prev_low']) / prev_range, 0.5)
    return panel, xcheck


def load_wide_seed(window):
    """The already-feature-built '2-5% gapper seed' (empirically gap>=3%, see module docstring).
    Returns (symbol,date,...) feature rows, NOT yet joined to price/volume."""
    if window == 'in_regime':
        frames = [pd.read_csv(p, keep_default_na=False, na_values=['']) for p in WIDE_CSVS]
        w = pd.concat(frames, ignore_index=True).drop_duplicates(['symbol', 'date'])
    else:
        w = pd.read_csv(OUT_REGIME_WIDE_CSV, keep_default_na=False, na_values=[''])
        w = w.drop_duplicates(['symbol', 'date'])
    w['symbol'] = _clean_symbol(w['symbol'])
    return w


def join_price_fields(feat, panel):
    """Join a feature-row frame (symbol,date) to the lagged daily panel for open/prev_volume/
    prev_high/prev_low/prev2_close/yday_return_pct/yday_close_position. Inner-ish via left join,
    dropna on open/prev_volume after (rows with no daily-panel match are NOT silently kept)."""
    d = panel[['symbol', 'bar_date', 'open', 'prev_close', 'prev_volume', 'prev_high', 'prev_low',
               'prev2_close', 'yday_return_pct', 'yday_close_position']].rename(columns={'bar_date': 'date'})
    before = len(feat)
    m = feat.merge(d, on=['symbol', 'date'], how='left')
    n_nomatch = m['open'].isna().sum()
    if n_nomatch:
        log.warning('join_price_fields: %d/%d feature rows had NO daily-panel match (symbol/date) '
                    '-- dropped, not silently kept', n_nomatch, before)
    return m.dropna(subset=['open', 'prev_volume'])


def bar_aggregates(pairs, window):
    """Direct minute-bar query (temp-table join on the store's own (symbol,day) prefix) for EVERY
    (symbol,day) pair's 04:00-09:35 ET bars. Dispatches to the store that ACTUALLY holds these
    candidates' bars, same rule 1685_subpools.py's own _pipeline_env_for() documented and this cell
    verified empirically before writing this function (bars_sip.db coverage of WIDE in-regime
    gap[3,5) candidates was only 20.8%; cache.db's intraday_bars_1min -- production's own
    historical-scan store, PRIMARY KEY (symbol,timestamp), indexed on (symbol,bar_date) -- is the
    live scanner's record for in-regime WIDE-seed candidates; out-regime always used bars_sip.db
    (cell 1,684's own fresh out-regime build), confirmed 100% bar coverage there):
      window == 'in_regime'  -> data/cache.db :: intraday_bars_1min (read-only)
      window == 'out_regime' -> research/bf_zero/bars_sip.db :: bars (read-only)
    Returns one row per pair: premarket_usd/premarket_volume/premarket_bars/premarket_high
    (04:00-09:30), and range_high/range_low/range_bars/first_bar_low (09:30-09:35, the SAME
    RANGE_MINUTES=5 window study_orb_features.extract_features() uses) -- never a guess, always the
    actual bars production would see. Coverage (pairs with >=1 bar at all) is the caller's
    responsibility to report."""
    empty = pd.DataFrame(columns=['symbol', 'day', 'premarket_usd', 'premarket_volume',
                                   'premarket_bars', 'premarket_high', 'range_high', 'range_low',
                                   'range_bars', 'first_bar_low'])
    if not pairs:
        return empty
    if window == 'in_regime':
        # intraday_bars_1min is indexed (symbol,bar_date) but EXPLAIN QUERY PLAN showed a
        # temp-table JOIN (or a multi-row VALUES IN) forces a full table SCAN instead of the index
        # (verified directly before writing this -- a single symbol='X' AND bar_date='Y' predicate
        # uses "SEARCH ... USING INDEX", the join/VALUES form uses "SCAN intraday_bars_1min"). Per-
        # symbol "bar_date IN (...)" DOES use the index (checked the same way), so group by symbol.
        con = _ro(CACHE_DB)
        by_sym = {}
        for s, d in pairs:
            by_sym.setdefault(s, []).append(d)
        cur = con.cursor()
        rows = []
        for s, dates in by_sym.items():
            ph = ','.join('?' * len(dates))
            cur.execute(f"SELECT symbol, bar_date AS day, timestamp AS t, open AS o, high AS h, "
                        f"low AS l, close AS c, volume AS v FROM intraday_bars_1min "
                        f"WHERE symbol = ? AND bar_date IN ({ph})", [s] + dates)
            rows.extend(cur.fetchall())
        con.close()
        df = pd.DataFrame(rows, columns=['symbol', 'day', 't', 'o', 'h', 'l', 'c', 'v'])
    else:
        con = _ro(BARS_SIP)
        sel = ("SELECT b.symbol, b.day, b.t, b.o, b.h, b.l, b.c, b.v FROM bars b "
               "JOIN want w ON b.symbol = w.symbol AND b.day = w.day")
        cur = con.cursor()
        cur.execute('DROP TABLE IF EXISTS temp.want')
        cur.execute('CREATE TEMP TABLE want (symbol TEXT, day TEXT)')
        cur.executemany('INSERT INTO want VALUES (?, ?)', pairs)
        cur.execute('CREATE INDEX temp.idx_want ON want(symbol, day)')
        cur.execute(sel)
        df = pd.DataFrame(cur.fetchall(), columns=['symbol', 'day', 't', 'o', 'h', 'l', 'c', 'v'])
        cur.execute('DROP TABLE want')
        con.close()
    if df.empty:
        return empty
    ts = pd.to_datetime(df['t'], utc=True).dt.tz_convert('America/New_York')
    mins = ts.dt.hour * 60 + ts.dt.minute
    df['mins'] = mins
    premkt = df[(mins >= 4 * 60) & (mins < 9 * 60 + 30)].copy()
    typ = (premkt['h'] + premkt['l'] + premkt['c']) / 3.0
    premkt['usd'] = typ * premkt['v']
    pg = premkt.groupby(['symbol', 'day']).agg(premarket_usd=('usd', 'sum'), premarket_volume=('v', 'sum'),
                                                premarket_bars=('usd', 'size'), premarket_high=('h', 'max'))
    rng = df[(mins >= 9 * 60 + 30) & (mins < 9 * 60 + 35)].copy().sort_values(['symbol', 'day', 't'])
    rg = rng.groupby(['symbol', 'day']).agg(range_high=('h', 'max'), range_low=('l', 'min'),
                                             range_bars=('h', 'size'))
    first_bar = rng.groupby(['symbol', 'day']).first()[['l']].rename(columns={'l': 'first_bar_low'})
    out = pg.join(rg, how='outer').join(first_bar, how='outer').reset_index()
    return out
