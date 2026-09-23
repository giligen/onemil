"""Build ORB features for the 2024H2 holdout (PREREG.md, cells 1,415-1,416), survivorship-free EQUS universe.
Research only (2026-09-23). Patches FOUR loader seams in study_orb_features before calling its main() —
see research/orb_2024/SPEC.md. Never writes data/cache.db, orb.yaml, or trading/. Output: ORB_FEATURES_OUT_DIR.

Usage: Y24_CELL=1415 python3 research/orb_2024/build_features_2024.py
       Y24_CELL=1416 python3 research/orb_2024/build_features_2024.py
"""
import os
import sqlite3
from pathlib import Path
from typing import Dict, List
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
Y24 = ROOT / 'research/day_breadth/y2024'
CELL = os.environ.get('Y24_CELL', '1415')

# cell_id: (gap_min_pct, gap_max_pct_or_None, price_min, price_max, out_dir)
CELLS = {
    '1415': (5.0, None, 3.0, 30.0, str(ROOT / 'research/orb_2024/out_1415')),
    '1416': (3.0, 5.0, 30.0, 50.0, str(ROOT / 'research/orb_2024/out_1416')),
}
GAP_MIN, GAP_MAX, PRICE_MIN, PRICE_MAX, OUT_DIR = CELLS[CELL]

os.environ.setdefault('ORB_FEATURES_OUT_DIR', OUT_DIR)
os.makedirs(os.environ['ORB_FEATURES_OUT_DIR'], exist_ok=True)
os.environ.setdefault('ORB_CATALYST_VETO', '0')

import study_orb_features as feats  # noqa: E402

ET = ZoneInfo('America/New_York')
FLOOR_935 = 15_000


def _map_sym(s: str) -> str:
    return s[:-1] + '.WS' if s.endswith('+') else s


def _band_universe() -> pd.DataFrame:
    """candidates.csv (symbol,bar_date) inner-joined to this cell's gap/price band, gap%/open recomputed
    from the SAME EQUS 2024H2 parquet candidates.csv was built from (research/day_breadth/y2024/universe.py)."""
    cands = pd.read_csv(Y24 / 'candidates.csv', dtype={'bar_date': str})
    daily = pd.read_parquet(ROOT / 'data/research/databento/equs_daily_2024H2.parquet',
                             columns=['bar_date', 'symbol', 'open', 'close'])
    daily['bar_date'] = daily['bar_date'].astype(str)
    daily = daily.sort_values(['symbol', 'bar_date'])
    daily['prev_close'] = daily.groupby('symbol')['close'].shift(1)
    daily['gap_pct'] = (daily['open'] - daily['prev_close']) / daily['prev_close'] * 100
    daily['symbol'] = daily['symbol'].map(_map_sym)
    merged = cands.merge(daily[['symbol', 'bar_date', 'open', 'gap_pct']], on=['symbol', 'bar_date'], how='inner')
    band = (merged['gap_pct'] >= GAP_MIN) & (merged['open'] >= PRICE_MIN) & (merged['open'] <= PRICE_MAX)
    if GAP_MAX is not None:
        band &= merged['gap_pct'] < GAP_MAX
    return merged.loc[band, ['symbol', 'bar_date']].reset_index(drop=True)


def _rth_935_floor(picked: pd.DataFrame) -> pd.DataFrame:
    """Keep pairs whose 09:30-09:35 ET RTH volume >= 15,000, from y2024/bars.db (t is ISO-UTC).
    DST-aware via two UTC time-of-day windows: EDT thru 2024-11-03 (Sun, DST end), EST from 2024-11-04."""
    con = sqlite3.connect(f'file:{Y24 / "bars.db"}?mode=ro', uri=True)
    frames = []
    for lo, hi, t0, t1 in (('2024-07-02', '2024-11-03', '13:30', '13:35'),
                            ('2024-11-04', '2024-12-31', '14:30', '14:35')):
        q = ("SELECT day, symbol, SUM(v) AS vol5 FROM bars "
             "WHERE day BETWEEN ? AND ? AND substr(t,12,5) >= ? AND substr(t,12,5) < ? "
             "GROUP BY day, symbol")
        frames.append(pd.read_sql_query(q, con, params=(lo, hi, t0, t1)))
    con.close()
    vol = pd.concat(frames, ignore_index=True).rename(columns={'day': 'bar_date'})
    vol['bar_date'] = vol['bar_date'].astype(str)
    m = picked.merge(vol, on=['symbol', 'bar_date'], how='left')
    m['vol5'] = m['vol5'].fillna(0.0)
    return m.loc[m['vol5'] >= FLOOR_935, ['symbol', 'bar_date']].reset_index(drop=True)


def _load_broad_universe(**kwargs) -> Dict[str, List[str]]:
    band = _band_universe()
    kept = _rth_935_floor(band)
    uni: Dict[str, List[str]] = {}
    for s, d in zip(kept['symbol'], kept['bar_date']):
        uni.setdefault(d, []).append(s)
    n_cands = len(pd.read_csv(Y24 / 'candidates.csv'))
    print(f"[y2024 {CELL}] candidates.csv={n_cands:,} band={len(band):,} 9:35-floor={len(kept):,} "
          f"pairs over {len(uni)} days (gap>={GAP_MIN}{'<' + str(GAP_MAX) if GAP_MAX else ''} "
          f"price {PRICE_MIN}-{PRICE_MAX})", flush=True)
    return uni


feats.load_broad_universe = _load_broad_universe


def _load_daily_bars_frame(db_path=None) -> pd.DataFrame:
    """EQUS 2024H2 daily (symbols mapped) UNION cache.db daily_bars 2024-06-03..2024-06-30 (early-July
    lookback; ONLY SPY has rows there per a direct check — non-SPY candidates entering before ~2024-07-30
    get a thin prior_df, degraded-not-broken 20d features per extract_features' own <5-day guard)."""
    daily = pd.read_parquet(ROOT / 'data/research/databento/equs_daily_2024H2.parquet',
                             columns=['bar_date', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
    daily['bar_date'] = daily['bar_date'].astype(str)
    daily['symbol'] = daily['symbol'].map(_map_sym)
    conn = sqlite3.connect(f'file:{ROOT / "data/cache.db"}?mode=ro', uri=True)
    lookback = pd.read_sql_query(
        "SELECT symbol, bar_date, open, high, low, close, volume FROM daily_bars "
        "WHERE bar_date BETWEEN '2024-06-03' AND '2024-06-30'", conn)
    conn.close()
    lookback['bar_date'] = lookback['bar_date'].astype(str)
    for f in (daily, lookback):
        for c in ('open', 'high', 'low', 'close'):
            f[c] = f[c].astype('float64')
        f['volume'] = f['volume'].astype('int64')
    out = pd.concat([daily, lookback], ignore_index=True)
    out['bar_date'] = pd.to_datetime(out['bar_date'])
    return out.sort_values(['symbol', 'bar_date']).reset_index(drop=True)


feats.load_daily_bars_frame = _load_daily_bars_frame


def _load_spy_intraday(db_path=None) -> pd.DataFrame:
    """SPY 1-min bars 2024-06..2024-12 from research/index_orb/cache/SPY_1min.parquet, converted from its
    tz-aware America/New_York 'timestamp' to the original loader's tz-aware UTC convention (instant-preserving,
    so the 09:30 ET bar lands on the same instant regardless of the EDT/EST offset that day)."""
    df = pd.read_parquet(ROOT / 'research/index_orb/cache/SPY_1min.parquet')
    ts = pd.to_datetime(df['timestamp'])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(ET)
    df = df.assign(timestamp=ts.dt.tz_convert('UTC'))
    df = df.loc[(df['timestamp'] >= pd.Timestamp('2024-06-01', tz='UTC')) &
                (df['timestamp'] < pd.Timestamp('2025-01-01', tz='UTC'))]
    keep = [c for c in ('timestamp', 'open', 'high', 'low', 'close', 'volume') if c in df.columns]
    return df[keep].sort_values('timestamp').reset_index(drop=True)


feats.load_spy_intraday = _load_spy_intraday

_Y24BARS = sqlite3.connect(f'file:{Y24 / "bars.db"}?mode=ro', uri=True)


def _get_intraday_bars_bulk(self, symbol_dates: list) -> Dict[tuple, List[Dict]]:
    """Same return structure as persistence.database.Database.get_intraday_bars_bulk, served from
    y2024/bars.db (table bars(symbol,day,t ISO-UTC,o,h,l,c,v)) instead of data/cache.db."""
    if not symbol_dates:
        return {}
    requested = set(symbol_dates)
    dates = [d for _, d in symbol_dates]
    cur = _Y24BARS.execute(
        "SELECT symbol, day, t, o, h, l, c, v FROM bars WHERE day >= ? AND day <= ? ORDER BY symbol, day, t",
        (min(dates), max(dates)))
    result: Dict[tuple, List[Dict]] = {}
    for symbol, day, t, o, h, l, c, v in cur:
        key = (symbol, str(day))
        if key not in requested:
            continue
        result.setdefault(key, []).append(
            {'timestamp': t, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})
    return result


feats.Database.get_intraday_bars_bulk = _get_intraday_bars_bulk


if __name__ == '__main__':
    import sys
    sys.argv = ['study_orb_features.py', '--force-full-regen', '--start-date', '2024-07-02']
    feats.main()
    print(f"[y2024 {CELL}] DONE -> {os.environ['ORB_FEATURES_OUT_DIR']}", flush=True)
