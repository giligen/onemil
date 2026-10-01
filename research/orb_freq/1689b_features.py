"""Cell 1,689b feature build for the BUILD pools (21 large-cap gap, 27 recovery price-floor,
28 recovery price-cap) -- patches study_orb_features' 4 loader seams exactly as 1684_features.py
does (the research/orb_2023/build_features_2023.py pattern), so production feature/trade-sim code
runs unchanged on THIS cell's candidates. The cheap pools (23-26,30) need NO feature build -- they
reuse the already-built WIDE seed, see pools_1689b_lib.py / 1689b_pools.py --stage prep.

--window in_regime|out_regime : candidates = any idea21/idea27/idea28 flag of
    subpools_1689b/build_candidates_<window>.csv. Daily bars: research/orb_freq/
    daily_source_1689b_<window>.parquet (already dumped by --stage prep, the FULL market panel --
    no re-query of cache.db here). Minute bars + SPY: research/bf_zero/bars_sip.db (appended by
    --stage backfill before this runs).
Output: research/orb_freq/out_1689b_<window>/orb_features_*.csv
"""
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research/orb_freq'
WINDOW = os.environ['ORB1689B_WINDOW']
assert WINDOW in ('in_regime', 'out_regime')

os.environ.setdefault('ORB_FEATURES_OUT_DIR', str(OUT / f'out_1689b_{WINDOW}'))
os.makedirs(os.environ['ORB_FEATURES_OUT_DIR'], exist_ok=True)
os.environ.setdefault('ORB_CATALYST_VETO', '0')

sys.path.insert(0, str(ROOT))
import study_orb_features as feats  # noqa: E402

BARS_SIP = ROOT / 'research/bf_zero/bars_sip.db'
DAILY_SRC = OUT / f'daily_source_1689b_{WINDOW}.parquet'
CANDS = OUT / f'subpools_1689b/build_candidates_{WINDOW}.csv'

# --- seam 1: universe ----------------------------------------------------------------------------
cand = pd.read_csv(CANDS, keep_default_na=False, na_values=[''])
mask = cand['idea21'].astype(bool) | cand['idea27'].astype(bool) | cand['idea28'].astype(bool)
cand = cand.loc[mask, ['date', 'symbol']].drop_duplicates()
_UNIVERSE: Dict[str, List[str]] = {}
for d, s in cand.itertuples(index=False):
    _UNIVERSE.setdefault(str(d), []).append(str(s))
_SYMS = sorted({s for v in _UNIVERSE.values() for s in v} | {'SPY'})
print(f"[1689b features {WINDOW}] universe: {len(cand):,} (symbol,date) pairs, "
      f"{len(_SYMS):,} symbols, {len(_UNIVERSE)} days", flush=True)


def _load_broad_universe(**kwargs):
    return _UNIVERSE


feats.load_broad_universe = _load_broad_universe


# --- seam 2: daily bars frame (read the already-dumped full-market parquet, no new DB query) -----
def _load_daily_bars_frame(db_path=None) -> pd.DataFrame:
    df = pd.read_parquet(DAILY_SRC)
    df['bar_date'] = pd.to_datetime(df['bar_date'])
    print(f"  [1689b] daily_bars (daily_source_1689b_{WINDOW}.parquet): {len(df):,} rows, "
          f"{df.symbol.nunique()} symbols", flush=True)
    return df.sort_values(['symbol', 'bar_date']).reset_index(drop=True)


feats.load_daily_bars_frame = _load_daily_bars_frame

# --- seam 3: SPY intraday -------------------------------------------------------------------------
_BARS_SIP_CON = sqlite3.connect(f'file:{BARS_SIP}?mode=ro', uri=True)


def _load_spy_intraday(db_path=None) -> pd.DataFrame:
    b = pd.read_sql_query("SELECT t, o, h, l, c, v FROM bars WHERE symbol='SPY' ORDER BY t", _BARS_SIP_CON)
    if b.empty:
        raise SystemExit("FATAL: no SPY rows in bars_sip.db -- cannot build SPY 5-min features (ORB_1689b)")
    out = pd.DataFrame({
        'timestamp': pd.to_datetime(b['t'], utc=True),
        'open': b['o'], 'high': b['h'], 'low': b['l'], 'close': b['c'], 'volume': b['v']})
    print(f"  [1689b] SPY intraday from bars_sip.db: {len(out):,} bars "
          f"{out.timestamp.min()}..{out.timestamp.max()}", flush=True)
    return out.sort_values('timestamp').reset_index(drop=True)


feats.load_spy_intraday = _load_spy_intraday


# --- seam 4: per-symbol-day intraday bulk ---------------------------------------------------------
def _get_intraday_bars_bulk(self, symbol_dates: list) -> Dict[tuple, List[Dict]]:
    """Temp-table JOIN on (symbol,day) -- same trick as 1684_features.py seam 4, never a day-range
    scan of the 133M-row bars table."""
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
        print(f"  [1689b] WARNING: {len(missing)}/{len(requested)} symbol-days have NO bars in "
              f"bars_sip.db (treated by study_orb_features as insufficient data -> no trade row)",
              flush=True)
    return result


feats.Database.get_intraday_bars_bulk = _get_intraday_bars_bulk

if __name__ == '__main__':
    start = min(_UNIVERSE) if _UNIVERSE else '2025-01-01'
    sys.argv = ['study_orb_features.py', '--force-full-regen', '--start-date', start]
    feats.main()
    print(f"[1689b features {WINDOW}] DONE -> {os.environ['ORB_FEATURES_OUT_DIR']}", flush=True)
