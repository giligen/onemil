"""Cell 1,689a feature build: patches study_orb_features' 4 loader seams (same pattern as
research/orb_freq/1684_features.py) so its production feature/trade-simulation code runs unchanged
on the recovery-slice candidates (gap>=5%, $3-30, prior-day volume 100K-500K) instead of the
production universe. This is where "the official 09:30 open from the minute bars" gets computed --
study_orb_features.py's own extract_features() builds gap_pct from the first minute bar of the 5-min
range, not the daily-bar open; a candidate whose minute bars never got fetched (or fetch failed)
simply produces NO row here (insufficient data), which is this cell's minute-bar coverage number.

--window in_regime : candidates = 1689a_candidates.csv rows (window==in_regime).
                      daily bars: pools_1689a_lib.load_in_regime_panel (cache.db + databento
                      EQUS.SUMMARY 2025-2026 cross-check). minute bars + SPY: bars_sip.db.
--window out_regime: candidates = 1689a_candidates.csv rows (window==out_regime).
                      daily bars: pools_1689a_lib.load_out_regime_panel (databento EQUS.SUMMARY
                      2024H2 + cache.db cross-check/lookback stub). minute bars + SPY: bars_sip.db.
Output: research/orb_freq/out_1689a_<window>/orb_features_*.csv (own dir, never touches 1684's).
"""
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research/orb_freq'
WINDOW = os.environ['ORB1689A_WINDOW']  # 'in_regime' | 'out_regime'
assert WINDOW in ('in_regime', 'out_regime')

os.environ.setdefault('ORB_FEATURES_OUT_DIR', str(OUT / f'out_1689a_{WINDOW}'))
os.makedirs(os.environ['ORB_FEATURES_OUT_DIR'], exist_ok=True)
os.environ.setdefault('ORB_CATALYST_VETO', '0')

sys.path.insert(0, str(OUT))
sys.path.insert(0, str(ROOT))
from pools_1689a_lib import load_in_regime_panel, load_out_regime_panel  # noqa: E402
import study_orb_features as feats  # noqa: E402

BARS_SIP = ROOT / 'research/bf_zero/bars_sip.db'
CANDIDATES = OUT / '1689a_candidates.csv'

# --- seam 1: universe --------------------------------------------------------------------------
cand = pd.read_csv(CANDIDATES, keep_default_na=False, na_values=[''])
cand = cand[cand['window'] == WINDOW]
_UNIVERSE: Dict[str, List[str]] = {}
for d, sy in cand[['day', 'symbol']].drop_duplicates().itertuples(index=False):
    _UNIVERSE.setdefault(str(d), []).append(str(sy))
_SYMS = sorted({sy for v in _UNIVERSE.values() for sy in v} | {'SPY'})
print(f"[1689a features {WINDOW}] universe: {len(cand):,} (symbol,date) pairs, "
      f"{len(_SYMS):,} symbols, {len(_UNIVERSE)} days", flush=True)


def _load_broad_universe(**kwargs):
    return _UNIVERSE


feats.load_broad_universe = _load_broad_universe

# --- seam 2: daily bars frame (cache.db + databento EQUS cross-check, restricted to _SYMS) ------
_loader = load_in_regime_panel if WINDOW == 'in_regime' else load_out_regime_panel


def _load_daily_bars_frame(db_path=None) -> pd.DataFrame:
    df, _xcheck = _loader(symbols=_SYMS)
    df = df[['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume']].copy()
    df['bar_date'] = pd.to_datetime(df['bar_date'])
    print(f"  [1689a] daily_bars ({WINDOW}, cache.db+databento): {len(df):,} rows, "
          f"{df.symbol.nunique()} symbols", flush=True)
    df = df.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
    dump = df.copy()
    dump['bar_date'] = dump['bar_date'].dt.strftime('%Y-%m-%d')
    dump.to_parquet(OUT / f'daily_source_1689a_{WINDOW}.parquet', index=False)
    return df


feats.load_daily_bars_frame = _load_daily_bars_frame

# --- seam 3: SPY intraday ------------------------------------------------------------------------
_BARS_SIP_CON = sqlite3.connect(f'file:{BARS_SIP}?mode=ro', uri=True)


def _load_spy_intraday(db_path=None) -> pd.DataFrame:
    b = pd.read_sql_query("SELECT t, o, h, l, c, v FROM bars WHERE symbol='SPY' ORDER BY t", _BARS_SIP_CON)
    if b.empty:
        raise SystemExit("FATAL: no SPY rows in bars_sip.db -- cannot build SPY 5-min features (ORB_1689A)")
    out = pd.DataFrame({
        'timestamp': pd.to_datetime(b['t'], utc=True),
        'open': b['o'], 'high': b['h'], 'low': b['l'], 'close': b['c'], 'volume': b['v']})
    print(f"  [1689a] SPY intraday from bars_sip.db: {len(out):,} bars "
          f"{out.timestamp.min()}..{out.timestamp.max()}", flush=True)
    return out.sort_values('timestamp').reset_index(drop=True)


feats.load_spy_intraday = _load_spy_intraday

# --- seam 4: per-symbol-day intraday bulk --------------------------------------------------------
def _get_intraday_bars_bulk(self, symbol_dates: list) -> Dict[tuple, List[Dict]]:
    """Temp-table JOIN on (symbol,day) -- same trick as 1684_features.py seam 4 -- never a day-range
    scan+sort of the 133M-row bars table."""
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
        print(f"  [1689a] WARNING: {len(missing)}/{len(requested)} symbol-days have NO bars in "
              f"bars_sip.db (treated by study_orb_features as insufficient data -> no trade row; "
              f"this is the minute-bar coverage gap, reported in RESULT_1689a.md)", flush=True)
    return result


feats.Database.get_intraday_bars_bulk = _get_intraday_bars_bulk

if __name__ == '__main__':
    start = min(_UNIVERSE) if _UNIVERSE else '2025-01-01'
    sys.argv = ['study_orb_features.py', '--force-full-regen', '--start-date', start]
    feats.main()
    print(f"[1689a features {WINDOW}] DONE -> {os.environ['ORB_FEATURES_OUT_DIR']}", flush=True)
