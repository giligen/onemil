"""Cell 1,689a shared library -- recovery sub-pools 19'/19/20 (PREREG_1684.md amendment 1b).

Population: gap>=5% (yesterday's close -> the official 09:30 open from the minute bars; daily-bar
`open` is only the FETCH-LIST screen, buffered to >=4.0%, and an explicit flagged fallback when a
minute bar fetch fails outright -- see research/orb_freq/1689a_slice.py's `stage_candidates`),
price (today's daily open) $3-30, prior-day volume [100K,500K) -- the names production's absolute
500K floor drops. RESULT_1685.md confirmed empirically that NO existing feature build (wide-seed,
idea1/2/10/11) covers this slice in EITHER window (0/22,606 wide-seed rows have prev_volume<500K) --
built fresh here from data/cache.db + databento EQUS.SUMMARY (delisted included).

Constants and the two daily-bar panel loaders are shared by 1689a_slice.py (candidates/poolsplit/
pipeline/score/tercile stages) and 1689a_features.py (the study_orb_features.py loader-seam script),
same discipline as cell 1,684/1,685's pools_1684_lib.py.
"""
import logging
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research/orb_freq'
POOLDIR = OUT / 'subpools_1689a'
POOLDIR.mkdir(exist_ok=True)
CACHE_DB = ROOT / 'data/cache.db'
BARS_SIP = ROOT / 'research/bf_zero/bars_sip.db'
EQUS_2024H2 = ROOT / 'data/research/databento/equs_daily_2024H2.parquet'
EQUS_2025_2026 = ROOT / 'data/research/databento/equs_daily_2025_2026.parquet'

PRICE_MIN, PRICE_MAX = 3.0, 30.0
REC_VOL_LO, REC_VOL_HI = 100_000, 500_000
PROD_GAP_MIN = 5.0
FETCH_GAP_MIN = 4.0                 # buffered daily-open screen for the FETCH list only
F1_FROZEN_PROFILE = 0.049862        # 1685_subpools.log TRAIN(2025) median -- REUSED, never refit
F1_MULT = 3.0
F2_PREMARKET_USD = 5_000_000.0
R_USD = 375.0

IN_REGIME = ('2025-01-01', '2026-09-26')
OUT_REGIME = ('2024-07-01', '2024-12-31')   # 2024H2 only -- the 2023-01..2024-06 minute store is gone

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                     handlers=[logging.StreamHandler(), logging.FileHandler(OUT / '1689a_slice.log', mode='a')],
                     force=True)
log = logging.getLogger('1689a')


def _ro(p):
    return sqlite3.connect(f'file:{p}?mode=ro', uri=True)


def _clean_symbol(s):
    return s.astype(str).str.replace(r'\+$', '.WS', regex=True)


def _add_derived(df):
    df = df.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
    g = df.groupby('symbol')
    df['prev_close'] = g['close'].shift(1)
    df['prev_volume'] = g['volume'].shift(1)
    df['gap_pct_daily'] = (df['open'] - df['prev_close']) / df['prev_close'] * 100
    return df


def load_in_regime_panel(symbols=None):
    """cache.db (source of record on overlap) UNION databento equs_daily_2025_2026 (delisted
    cross-check; covers 2025-01-02..2026-09-04 only, ~3wk short of this cell's 09-26 admission end
    -- stated, not hidden), 2024-09-01.. for ADV20/prev_close/prev_volume lookback."""
    con = _ro(CACHE_DB)
    if symbols:
        ph = ','.join('?' * len(symbols))
        cache = pd.read_sql_query(
            f"SELECT symbol,bar_date,open,high,low,close,volume FROM daily_bars "
            f"WHERE bar_date>='2024-09-01' AND symbol IN ({ph})", con, params=list(symbols))
    else:
        cache = pd.read_sql_query(
            "SELECT symbol,bar_date,open,high,low,close,volume FROM daily_bars WHERE bar_date>='2024-09-01'", con)
    con.close()
    for c in ('open', 'high', 'low', 'close'):
        cache[c] = cache[c].astype('float64')
    cache['volume'] = cache['volume'].astype('int64')
    equs = pd.read_parquet(EQUS_2025_2026, columns=['bar_date', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
    equs['symbol'] = _clean_symbol(equs['symbol'])
    if symbols:
        equs = equs[equs['symbol'].isin(symbols)]
    both = pd.concat([cache, equs], ignore_index=True)
    before = len(both)
    both = both.drop_duplicates(['symbol', 'bar_date'], keep='first')
    cache_syms = set(cache['symbol'].unique())
    equs_only = set(equs['symbol'].unique()) - cache_syms
    log.info('in_regime panel: cache.db=%d rows databento=%d rows union=%d (dedup dropped %d); '
             '%d symbols ONLY in databento (delisted cross-check)', len(cache), len(equs), len(both),
             before - len(both), len(equs_only))
    both['bar_date'] = pd.to_datetime(both['bar_date']).dt.strftime('%Y-%m-%d')
    return _add_derived(both), equs_only


def load_out_regime_panel(symbols=None):
    """databento equs_daily_2024H2 (source of record, cell 1,684/1,685's own convention) UNION
    cache.db 2024-06-03..2024-12-31 (20d-lookback stub before 07-01 + cross-check overlap)."""
    equs = pd.read_parquet(EQUS_2024H2, columns=['bar_date', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
    equs['symbol'] = _clean_symbol(equs['symbol'])
    if symbols:
        equs = equs[equs['symbol'].isin(symbols)]
    con = _ro(CACHE_DB)
    if symbols:
        ph = ','.join('?' * len(symbols))
        cache = pd.read_sql_query(
            f"SELECT symbol,bar_date,open,high,low,close,volume FROM daily_bars WHERE bar_date>='2024-06-03' "
            f"AND bar_date<='2024-12-31' AND symbol IN ({ph})", con, params=list(symbols))
    else:
        cache = pd.read_sql_query(
            "SELECT symbol,bar_date,open,high,low,close,volume FROM daily_bars "
            "WHERE bar_date>='2024-06-03' AND bar_date<='2024-12-31'", con)
    con.close()
    for c in ('open', 'high', 'low', 'close'):
        cache[c] = cache[c].astype('float64')
    cache['volume'] = cache['volume'].astype('int64')
    both = pd.concat([equs, cache], ignore_index=True)
    before = len(both)
    both = both.drop_duplicates(['symbol', 'bar_date'], keep='first')
    equs_syms = set(equs['symbol'].unique())
    cache_only = set(cache['symbol'].unique()) - equs_syms
    log.info('out_regime panel: databento=%d rows cache.db=%d rows union=%d (dedup dropped %d); '
             '%d symbols ONLY in cache.db (reverse cross-check)', len(equs), len(cache), len(both),
             before - len(both), len(cache_only))
    both['bar_date'] = pd.to_datetime(both['bar_date']).dt.strftime('%Y-%m-%d')
    return _add_derived(both), cache_only
