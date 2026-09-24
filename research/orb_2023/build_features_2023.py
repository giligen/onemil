"""Build ORB features for 2023-01..2024-06 (research/orb_2023/PREREG.md, cells 1,418-1,419), point-in-time universe.
Adapted from research/orb_2024/build_features_2024.py (same four seams).
Research only (2026-09-23). Patches FOUR loader seams in study_orb_features before calling its main() —
see research/orb_2024/SPEC.md. Never writes data/cache.db, orb.yaml, or trading/. Output: ORB_FEATURES_OUT_DIR.

Usage: Y23_CELL=1418 python3 research/orb_2023/build_features_2023.py
       Y23_CELL=1419 python3 research/orb_2023/build_features_2023.py
"""
import os
import sqlite3
from pathlib import Path
from typing import Dict, List
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
Y24 = ROOT / 'research/orb_2023'   # kept the name Y24 for a minimal diff; it points at the 2023 data
CELL = os.environ.get('Y23_CELL', '1418')

# cell_id: (gap_min_pct, gap_max_pct_or_None, price_min, price_max, out_dir)
CELLS = {
    '1418': (5.0, None, 3.0, 30.0, str(ROOT / 'research/orb_2023/out_1418')),
    '1419': (3.0, 5.0, 30.0, 50.0, str(ROOT / 'research/orb_2023/out_1419')),
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
    daily = pd.read_parquet(Y24 / 'daily_alpaca.parquet', columns=['bar_date', 'symbol', 'open', 'close'])
    daily['bar_date'] = daily['bar_date'].astype(str)
    daily = daily.sort_values(['symbol', 'bar_date'])
    daily['prev_close'] = daily.groupby('symbol')['close'].shift(1)
    daily['gap_pct'] = (daily['open'] - daily['prev_close']) / daily['prev_close'] * 100
    merged = cands.merge(daily[['symbol', 'bar_date', 'open', 'gap_pct']], on=['symbol', 'bar_date'], how='inner')
    band = (merged['gap_pct'] >= GAP_MIN) & (merged['open'] >= PRICE_MIN) & (merged['open'] <= PRICE_MAX)
    if GAP_MAX is not None:
        band &= merged['gap_pct'] < GAP_MAX
    return merged.loc[band, ['symbol', 'bar_date']].reset_index(drop=True)


def _rth_935_floor(picked: pd.DataFrame) -> pd.DataFrame:
    """Keep pairs whose 09:30-09:35 ET RTH volume >= 15,000, from bars.db (t ISO-UTC). DST-robust: pull UTC hours
    13-14 (09:30 ET is 13:30 UTC in EDT, 14:30 UTC in EST), convert every bar to ET, keep 09:30 <= ET < 09:35."""
    con = sqlite3.connect(f'file:{Y24 / "bars.db"}?mode=ro', uri=True)
    q = ("SELECT day, symbol, t, v FROM bars WHERE substr(t,12,2) IN ('13','14') "
         "AND substr(t,15,2) >= '30' AND substr(t,15,2) < '35'")
    b = pd.read_sql_query(q, con)
    con.close()
    et = pd.to_datetime(b['t'], utc=True).dt.tz_convert(ET)
    b = b[(et.dt.hour == 9) & (et.dt.minute >= 30) & (et.dt.minute < 35)]
    vol = b.groupby(['day', 'symbol'], as_index=False)['v'].sum().rename(columns={'day': 'bar_date', 'v': 'vol5'})
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
    print(f"[y2023 {CELL}] candidates.csv={n_cands:,} band={len(band):,} 9:35-floor={len(kept):,} "
          f"pairs over {len(uni)} days (gap>={GAP_MIN}{'<' + str(GAP_MAX) if GAP_MAX else ''} "
          f"price {PRICE_MIN}-{PRICE_MAX})", flush=True)
    return uni


feats.load_broad_universe = _load_broad_universe


def _load_daily_bars_frame(db_path=None) -> pd.DataFrame:
    """Consolidated Alpaca SIP daily bars 2022-11-15..2024-06-30 for every point-in-time ticker (incl. SPY): the
    file carries its own 20-day lead, so no cache.db lookback is needed."""
    out = pd.read_parquet(Y24 / 'daily_alpaca.parquet',
                          columns=['bar_date', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
    for c in ('open', 'high', 'low', 'close'):
        out[c] = out[c].astype('float64')
    out['volume'] = out['volume'].astype('int64')
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
    df = df.loc[(df['timestamp'] >= pd.Timestamp('2022-11-15', tz='UTC')) &
                (df['timestamp'] < pd.Timestamp('2024-07-01', tz='UTC'))]
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
    sys.argv = ['study_orb_features.py', '--force-full-regen', '--start-date', '2023-01-03']
    feats.main()
    print(f"[y2023 {CELL}] DONE -> {os.environ['ORB_FEATURES_OUT_DIR']}", flush=True)
