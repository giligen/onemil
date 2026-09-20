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
    uni = _orig_load(*a, **k)
    end = _CHUNK_END[0]
    return {d: s for d, s in uni.items() if end is None or d <= end}


feats.load_broad_universe = _chunked_load


# --- daily-bars loader: universe symbols only (the all-symbols query blew the memory cap) ---
import sqlite3  # noqa: E402
import pandas as pd  # noqa: E402

_FULL_UNIVERSE = _orig_load()
_SYMS = sorted({sym for syms in _FULL_UNIVERSE.values() for sym in syms} | {'SPY'})
_DAILY_FLOOR = '2024-09-01'   # >= 60 sessions of history before the first built day


def _daily_for_universe(db_path: str = feats.CACHE_DB) -> pd.DataFrame:
    """Same columns as the original loader: ONE pass on the bar_date index, symbol filter in memory."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT symbol, bar_date, open, high, low, close, volume FROM daily_bars "
        "WHERE bar_date >= ?", conn, params=(_DAILY_FLOOR,))
    conn.close()
    n_all = len(df)
    df = df[df['symbol'].isin(set(_SYMS))].reset_index(drop=True)
    print(f"[wide seed] daily frame: {len(df):,} rows kept of {n_all:,} since {_DAILY_FLOOR} "
          f"for {len(_SYMS):,} symbols", flush=True)
    return df


feats.load_daily_bars_frame = _daily_for_universe

# Month-end chunk boundaries, Jan-2025 .. May-2026 (TEST >= 2026-06 stays unbuilt).
ends = []
for y, m in [(2025, m) for m in range(1, 13)] + [(2026, m) for m in range(1, 6)]:
    nxt = date(y + (m == 12), (m % 12) + 1, 1)
    ends.append((nxt.toordinal() - 1))
ends = [date.fromordinal(o).isoformat() for o in ends]

print(f"[wide seed] gap>={GAP}%  open ${broad.MIN_OPEN_PRICE}-${PMAX}  "
      f"prev_vol>={broad.MIN_PREV_DAY_VOL:,}  out={os.environ['ORB_FEATURES_OUT_DIR']}  "
      f"chunks={len(ends)}", flush=True)

for i, end in enumerate(ends):
    _CHUNK_END[0] = end
    if i == 0:
        sys.argv = ['study_orb_features.py', '--force-full-regen', '--start-date', '2025-01-01']
    else:
        sys.argv = ['study_orb_features.py']
    print(f"\n[wide seed] ===== chunk {i + 1}/{len(ends)} through {end} =====", flush=True)
    feats.main()
print("[wide seed] ALL CHUNKS DONE", flush=True)
