"""Build an ORB features CSV on a WIDER seed (gap>=3%, open $3-50) without touching production.

Research only (2026-09-20). Patches study_orb_broad's constants BEFORE study_orb_features
imports load_broad_universe, and CHUNKS the seed by month (the builder's bulk bar load
of the whole universe blew a 3.5 GB cap). Chunk 1 runs --force-full-regen; later chunks
run the builder's own incremental mode, which resumes from the side CSV's last date.
Output: ORB_FEATURES_OUT_DIR (side dir) — production CSVs are never written.
"""
import os
import sys
from datetime import date

os.environ.setdefault('ORB_FEATURES_OUT_DIR', 'research/orb_seed_wide/out')
os.makedirs(os.environ['ORB_FEATURES_OUT_DIR'], exist_ok=True)

import study_orb_broad as broad  # noqa: E402

GAP = float(os.environ.get('WIDE_MIN_GAP_PCT', '3.0'))
PMAX = float(os.environ.get('WIDE_MAX_OPEN_PRICE', '50.0'))
broad.MIN_GAP_PCT = GAP
broad.MAX_OPEN_PRICE = PMAX

import study_orb_features as feats  # noqa: E402

_orig_load = feats.load_broad_universe
_CHUNK_END = [None]


def _chunked_load(*a, **k):
    """The unchanged loader, then keep only days <= the current chunk end."""
    uni = _FULL_UNIVERSE
    end = _CHUNK_END[0]
    return {d: s for d, s in uni.items() if end is None or d <= end}


feats.load_broad_universe = _chunked_load


# --- daily-bars loader: universe symbols only (the all-symbols query blew the memory cap) ---
import sqlite3  # noqa: E402
import pandas as pd  # noqa: E402

import pickle, resource  # noqa: E402
_CACHE = 'research/orb_seed_wide/cache'
os.makedirs(_CACHE, exist_ok=True)
_UNI_PKL = f'{_CACHE}/universe_gap{GAP}_p{PMAX}.pkl'
if os.path.exists(_UNI_PKL):
    _FULL_UNIVERSE = pickle.load(open(_UNI_PKL, 'rb'))
    print(f"[wide seed] universe from cache: {sum(len(v) for v in _FULL_UNIVERSE.values()):,} pairs", flush=True)
else:
    _FULL_UNIVERSE = _orig_load()
    pickle.dump(_FULL_UNIVERSE, open(_UNI_PKL, 'wb'))
    print(f"[wide seed] universe queried and cached: {sum(len(v) for v in _FULL_UNIVERSE.values()):,} pairs", flush=True)


def _rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
_SYMS = sorted({sym for syms in _FULL_UNIVERSE.values() for sym in syms} | {'SPY'})
_DAILY_FLOOR = '2024-09-01'   # >= 60 sessions of history before the first built day


def _daily_for_universe(db_path: str = feats.CACHE_DB) -> pd.DataFrame:
    """Same columns as the original loader: temp-table join on the (symbol, bar_date) index, one pass."""
    pq = f'{_CACHE}/daily_since_{_DAILY_FLOOR}.parquet'
    if os.path.exists(pq):
        df = pd.read_parquet(pq)
        df['bar_date'] = pd.to_datetime(df['bar_date'])
        print(f"[wide seed] daily frame from cache: {len(df):,} rows  (maxrss {_rss()} MB)", flush=True)
        return df.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TEMP TABLE syms(symbol TEXT PRIMARY KEY)")
    conn.executemany("INSERT INTO syms VALUES (?)", [(x,) for x in _SYMS])
    df = pd.read_sql_query(
        "SELECT d.symbol, d.bar_date, d.open, d.high, d.low, d.close, d.volume "
        "FROM syms s JOIN daily_bars d ON d.symbol = s.symbol WHERE d.bar_date >= ?",
        conn, params=(_DAILY_FLOOR,))
    conn.close()
    df.to_parquet(pq, index=False)
    df['bar_date'] = pd.to_datetime(df['bar_date'])
    print(f"[wide seed] daily frame built+cached: {len(df):,} rows for {len(_SYMS):,} symbols  (maxrss {_rss()} MB)",
          flush=True)
    return df.sort_values(['symbol', 'bar_date']).reset_index(drop=True)


feats.load_daily_bars_frame = _daily_for_universe

# Month-end chunk boundaries, Jan-2025 .. May-2026 (TEST >= 2026-06 stays unbuilt).
# Override (owner 2026-09-21, TEST unsealed for the union rung): WIDE_MONTHS="2026-06:2026-09"
# builds those month chunks INCREMENTALLY on top of the existing features CSV (no full regen).
_months = [(2025, m) for m in range(1, 13)] + [(2026, m) for m in range(1, 6)]
_incremental_only = False
if os.environ.get('WIDE_MONTHS'):
    a, b = os.environ['WIDE_MONTHS'].split(':')
    ya, ma = map(int, a.split('-')); yb, mb = map(int, b.split('-'))
    _months = [(y, m) for y in range(ya, yb + 1) for m in range(1, 13)
               if (y, m) >= (ya, ma) and (y, m) <= (yb, mb)]
    _incremental_only = True
ends = []
for y, m in _months:
    nxt = date(y + (m == 12), (m % 12) + 1, 1)
    ends.append((nxt.toordinal() - 1))
ends = [date.fromordinal(o).isoformat() for o in ends]

print(f"[wide seed] gap>={GAP}%  open ${broad.MIN_OPEN_PRICE}-${PMAX}  "
      f"prev_vol>={broad.MIN_PREV_DAY_VOL:,}  out={os.environ['ORB_FEATURES_OUT_DIR']}  "
      f"chunks={len(ends)}", flush=True)

for i, end in enumerate(ends):
    _CHUNK_END[0] = end
    if i == 0 and not _incremental_only:
        sys.argv = ['study_orb_features.py', '--force-full-regen', '--start-date', '2025-01-01']
    else:
        sys.argv = ['study_orb_features.py']
    print(f"\n[wide seed] ===== chunk {i + 1}/{len(ends)} through {end} =====", flush=True)
    feats.main()
    print(f"[wide seed] chunk {i + 1} done  (maxrss {_rss()} MB)", flush=True)
print("[wide seed] ALL CHUNKS DONE", flush=True)
