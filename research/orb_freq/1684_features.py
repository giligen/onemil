"""Cell 1,684 feature build: patches study_orb_features' 4 loader seams (the
research/orb_2023/build_features_2023.py pattern) so its production feature/trade-simulation code
runs unchanged on THIS cell's candidate (date,symbol) pairs instead of the production universe.

--window in_regime : candidates = idea10 | idea11 rows of 1684_reads.csv (window==in_regime).
                      daily bars: data/cache.db (covers 2024-06-03..2026-09-30). minute bars + SPY:
                      research/bf_zero/bars_sip.db.
--window out_regime: candidates = idea1_pre | idea2 | idea10 | idea11 rows (window==out_regime).
                      daily bars: data/research/databento/equs_daily_2024H2.parquet UNION
                      cache.db rows 2024-06-03..2024-06-30 (20d-lookback stub, same construction as
                      research/orb_2024/SPEC.md seam 2). minute bars + SPY: bars_sip.db (it starts
                      2024-07-01, covering all of 2024H2).
Output: research/orb_freq/out_<window>/orb_features_*.csv (never touches a production out dir).
"""
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research/orb_freq'
WINDOW = os.environ['ORB1684_WINDOW']  # 'in_regime' | 'out_regime'
assert WINDOW in ('in_regime', 'out_regime')

os.environ.setdefault('ORB_FEATURES_OUT_DIR', str(OUT / f'out_{WINDOW}'))
os.makedirs(os.environ['ORB_FEATURES_OUT_DIR'], exist_ok=True)
os.environ.setdefault('ORB_CATALYST_VETO', '0')

sys.path.insert(0, str(ROOT))
import study_orb_features as feats  # noqa: E402

CACHE_DB = ROOT / 'data/cache.db'
BARS_SIP = ROOT / 'research/bf_zero/bars_sip.db'
EQUS_2024H2 = ROOT / 'data/research/databento/equs_daily_2024H2.parquet'
READS = OUT / '1684_reads.csv'

# --- seam 1: universe --------------------------------------------------------------------------
reads = pd.read_csv(READS, keep_default_na=False, na_values=[''])
reads = reads[reads['window'] == WINDOW]
if WINDOW == 'in_regime':
    mask = reads['idea10'].astype(bool) | reads['idea11'].astype(bool)
else:
    mask = (reads['idea1_pre'].astype(bool) | reads['idea2'].astype(bool)
            | reads['idea10'].astype(bool) | reads['idea11'].astype(bool))
cand = reads.loc[mask, ['bar_date', 'symbol']].drop_duplicates()
_UNIVERSE: Dict[str, List[str]] = {}
for d, s in cand.itertuples(index=False):
    _UNIVERSE.setdefault(str(d), []).append(str(s))
_SYMS = sorted({s for v in _UNIVERSE.values() for s in v} | {'SPY'})
print(f"[1684 features {WINDOW}] universe: {len(cand):,} (symbol,date) pairs, "
      f"{len(_SYMS):,} symbols, {len(_UNIVERSE)} days", flush=True)


def _load_broad_universe(**kwargs):
    return _UNIVERSE


feats.load_broad_universe = _load_broad_universe

# --- seam 2: daily bars frame -------------------------------------------------------------------
if WINDOW == 'in_regime':
    def _load_daily_bars_frame(db_path=None) -> pd.DataFrame:
        con = sqlite3.connect(f'file:{CACHE_DB}?mode=ro', uri=True)
        ph = ','.join('?' * len(_SYMS))
        df = pd.read_sql_query(
            f"SELECT symbol, bar_date, open, high, low, close, volume FROM daily_bars "
            f"WHERE symbol IN ({ph}) AND bar_date >= '2024-09-01'", con, params=_SYMS)
        con.close()
        for c in ('open', 'high', 'low', 'close'):
            df[c] = df[c].astype('float64')
        df['volume'] = df['volume'].astype('int64')
        df['bar_date'] = pd.to_datetime(df['bar_date'])
        print(f"  [1684] daily_bars (cache.db): {len(df):,} rows, {df.symbol.nunique()} symbols", flush=True)
        df = df.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
        dump = df.copy()
        dump['bar_date'] = dump['bar_date'].dt.strftime('%Y-%m-%d')
        dump.to_parquet(OUT / f'daily_source_{WINDOW}.parquet', index=False)
        return df
else:
    def _load_daily_bars_frame(db_path=None) -> pd.DataFrame:
        equs = pd.read_parquet(EQUS_2024H2, columns=['bar_date', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
        equs['symbol'] = equs['symbol'].str.replace(r'\+$', '.WS', regex=True)
        equs = equs[equs['symbol'].isin(_SYMS)]
        con = sqlite3.connect(f'file:{CACHE_DB}?mode=ro', uri=True)
        ph = ','.join('?' * len(_SYMS))
        stub = pd.read_sql_query(
            f"SELECT symbol, bar_date, open, high, low, close, volume FROM daily_bars "
            f"WHERE symbol IN ({ph}) AND bar_date >= '2024-06-03' AND bar_date < '2024-07-01'",
            con, params=_SYMS)
        con.close()
        df = pd.concat([stub, equs], ignore_index=True).drop_duplicates(['symbol', 'bar_date'])
        for c in ('open', 'high', 'low', 'close'):
            df[c] = df[c].astype('float64')
        df['volume'] = df['volume'].astype('int64')
        df['bar_date'] = pd.to_datetime(df['bar_date'])
        print(f"  [1684] daily_bars (EQUS 2024H2 + cache.db stub): {len(df):,} rows, "
              f"{df.symbol.nunique()} symbols", flush=True)
        df = df.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
        dump = df.copy()
        dump['bar_date'] = dump['bar_date'].dt.strftime('%Y-%m-%d')
        dump.to_parquet(OUT / f'daily_source_{WINDOW}.parquet', index=False)
        return df

feats.load_daily_bars_frame = _load_daily_bars_frame

# --- seam 3: SPY intraday ------------------------------------------------------------------------
_BARS_SIP_CON = sqlite3.connect(f'file:{BARS_SIP}?mode=ro', uri=True)


def _load_spy_intraday(db_path=None) -> pd.DataFrame:
    b = pd.read_sql_query("SELECT t, o, h, l, c, v FROM bars WHERE symbol='SPY' ORDER BY t", _BARS_SIP_CON)
    if b.empty:
        raise SystemExit("FATAL: no SPY rows in bars_sip.db -- cannot build SPY 5-min features (ORB_1684)")
    out = pd.DataFrame({
        'timestamp': pd.to_datetime(b['t'], utc=True),
        'open': b['o'], 'high': b['h'], 'low': b['l'], 'close': b['c'], 'volume': b['v']})
    print(f"  [1684] SPY intraday from bars_sip.db: {len(out):,} bars "
          f"{out.timestamp.min()}..{out.timestamp.max()}", flush=True)
    return out.sort_values('timestamp').reset_index(drop=True)


feats.load_spy_intraday = _load_spy_intraday

# --- seam 4: per-symbol-day intraday bulk --------------------------------------------------------
def _get_intraday_bars_bulk(self, symbol_dates: list) -> Dict[tuple, List[Dict]]:
    """Temp-table JOIN on (symbol,day) -- the bars table's own primary-key prefix -- instead of a
    day-range scan+sort (which tried to materialize/sort most of a 22GB table and hit 'disk full'
    on the first attempt for the in-regime window's 16K pairs spanning 2025-01..2026-09)."""
    if not symbol_dates:
        return {}
    requested = set(symbol_dates)
    cur = _BARS_SIP_CON.cursor()
    cur.execute("DROP TABLE IF EXISTS temp.want")
    cur.execute("CREATE TEMP TABLE want (symbol TEXT, day TEXT)")
    cur.executemany("INSERT INTO want VALUES (?, ?)", [(s, str(d)) for s, d in requested])
    cur.execute("CREATE INDEX temp.idx_want ON want(symbol, day)")
    cur.execute(
        "SELECT b.symbol, b.day, b.t, b.o, b.h, b.l, b.c, b.v FROM bars b "
        "JOIN want w ON b.symbol = w.symbol AND b.day = w.day ORDER BY b.symbol, b.day, b.t")
    result: Dict[tuple, List[Dict]] = {}
    for symbol, day, t, o, h, l, c, v in cur:
        key = (symbol, str(day))
        result.setdefault(key, []).append(
            {'timestamp': t, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})
    cur.execute("DROP TABLE want")
    missing = requested - set(result)
    if missing:
        print(f"  [1684] WARNING: {len(missing)}/{len(requested)} symbol-days have NO bars in "
              f"bars_sip.db (treated by study_orb_features as insufficient data -> no trade row)",
              flush=True)
    return result


feats.Database.get_intraday_bars_bulk = _get_intraday_bars_bulk

if __name__ == '__main__':
    start = min(_UNIVERSE) if _UNIVERSE else '2025-01-01'
    sys.argv = ['study_orb_features.py', '--force-full-regen', '--start-date', start]
    feats.main()
    print(f"[1684 features {WINDOW}] DONE -> {os.environ['ORB_FEATURES_OUT_DIR']}", flush=True)
