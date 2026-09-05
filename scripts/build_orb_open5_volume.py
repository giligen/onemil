"""ORB signal study C1 — opening 5-minute volume history (2026-09-05).

`rvol_open5` (Zarattini et al. "stocks in play"): volume of 9:30–9:35 ET on
the candidate day divided by the mean of the same 5-minute window over the
prior 14 trading days. This builds the per-(symbol, day) 9:30–9:34 volume for
every ORB candidate symbol-day AND its 14 prior trading days into a SIDE
database (never cache.db), cache.db bars first (grouped query), Alpaca 1-min
for the rest (5-bar windows, batched per day), resumable; then writes the
sidecar CSV the pipeline joins on (symbol, date).

Usage: python3 scripts/build_orb_open5_volume.py [FEATURES_CSV] [OUT_SIDECAR]
"""
import os
import sqlite3
import sys
from datetime import date, datetime

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, '/home/ec2-user/onemil')
sys.stdout.reconfigure(line_buffering=True)
load_dotenv('/home/ec2-user/onemil/.env')
from data_sources.alpaca_client import AlpacaClient  # noqa: E402
from study_orb_broad import rth_volume_by_935_for_pairs  # noqa: E402
from trading.orb_csv import read_orb_csv  # noqa: E402

ROOT = '/home/ec2-user/onemil'
SIDE_DB = f'{ROOT}/data/research/orb_open5_volume.db'
CLASSIFIED = f'{ROOT}/data/research/databento/pit_missing_classified.csv'
PRIOR_DAYS = 14
MIN_PRIOR = 10
ET = 'America/New_York'
CHUNK = 200


def latest_features() -> str:
    import glob
    return sorted(p for p in glob.glob(f'{ROOT}/analysis_results/orb_features_*.csv') if 'corrmatrix' not in p)[-1]


def candidate_keys(features_csv: str) -> pd.DataFrame:
    """ORB candidate (symbol, date) pairs: the features CSV ∪ the wrapper keys
    the 9/5 backfill added (they enter the next regen)."""
    f = read_orb_csv(features_csv, usecols=['symbol', 'date'])
    f['date'] = pd.to_datetime(f['date']).dt.strftime('%Y-%m-%d')
    cls = read_orb_csv(CLASSIFIED)
    w = cls[(cls['book'] == 'orb') & (cls['cls'] == 'active_filtered_by_is_common_stock')][['symbol', 'bar_date']]
    w = w.rename(columns={'bar_date': 'date'})
    return pd.concat([f, w], ignore_index=True).drop_duplicates()


def trading_calendar(client: AlpacaClient, start: date, end: date) -> list:
    bars = client.get_daily_bars_range(['SPY'], start, end).get('SPY', [])
    return sorted({(b['date'].isoformat() if isinstance(b['date'], date) else str(b['date'])[:10]) for b in bars})


def main() -> None:
    features_csv = sys.argv[1] if len(sys.argv) > 1 else latest_features()
    out_csv = sys.argv[2] if len(sys.argv) > 2 else f'{ROOT}/research/orb_signal_study/sidecar_rvol.csv'
    client = AlpacaClient(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_API_SECRET'], paper=False)
    keys = candidate_keys(features_csv)
    print(f"candidate keys: {len(keys):,} ({keys['symbol'].nunique():,} symbols) from {features_csv}")
    cal = trading_calendar(client, date(2024, 11, 1), date.today())
    idx = {d: i for i, d in enumerate(cal)}
    needed = set()
    for s, d in zip(keys['symbol'], keys['date']):
        i = idx.get(d)
        if i is None:
            continue
        for j in range(max(0, i - PRIOR_DAYS), i + 1):
            needed.add((s, cal[j]))
    print(f"symbol-days needed (keys + {PRIOR_DAYS} prior): {len(needed):,}")

    side = sqlite3.connect(SIDE_DB, timeout=120)
    side.execute("CREATE TABLE IF NOT EXISTS open5 (symbol TEXT, day TEXT, vol REAL, source TEXT, PRIMARY KEY (symbol, day))")
    have = {(r[0], r[1]) for r in side.execute("SELECT symbol, day FROM open5")}
    todo = sorted(needed - have)
    print(f"already in side DB: {len(have):,} | to resolve: {len(todo):,}")

    # 1) cache.db first (one grouped query per 20K pairs)
    cache = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
    got_cache = 0
    for i in range(0, len(todo), 20000):
        part = todo[i:i + 20000]
        vols = rth_volume_by_935_for_pairs(cache, part)
        rows = [(s, d, float(v), 'cache') for (s, d), v in vols.items() if v and v > 0]
        side.executemany("INSERT OR IGNORE INTO open5 VALUES (?,?,?,?)", rows)
        side.commit()
        got_cache += len(rows)
        print(f"  cache pass {i // 20000 + 1}: {len(rows):,} resolved (cum {got_cache:,})")
    cache.close()
    have = {(r[0], r[1]) for r in side.execute("SELECT symbol, day FROM open5")}
    todo = sorted(needed - have)
    print(f"after cache: {len(todo):,} left for Alpaca")

    # 2) Alpaca 9:30–9:35 windows, batched per day
    by_day = {}
    for s, d in todo:
        by_day.setdefault(d, []).append(s)
    n_api = 0
    for k, (d, syms) in enumerate(sorted(by_day.items())):
        st = pd.Timestamp(d).tz_localize(ET) + pd.Timedelta(hours=9, minutes=30)
        en = pd.Timestamp(d).tz_localize(ET) + pd.Timedelta(hours=9, minutes=35)
        rows = []
        for i in range(0, len(syms), CHUNK):
            chunk = syms[i:i + CHUNK]
            try:
                res = client.get_1min_bars_range_multi(chunk, st.tz_convert('UTC').to_pydatetime(), en.tz_convert('UTC').to_pydatetime())
            except Exception as e:
                print(f"  WARN {d} chunk {i}: {str(e)[:120]}")
                continue
            for s in chunk:
                df = (res or {}).get(s)
                v = float(df['volume'].sum()) if df is not None and len(df) else None
                rows.append((s, d, v, 'alpaca' if v is not None else 'none'))
        side.executemany("INSERT OR IGNORE INTO open5 VALUES (?,?,?,?)", rows)
        side.commit()
        n_api += len(rows)
        if (k + 1) % 25 == 0 or k == len(by_day) - 1:
            print(f"  alpaca {k + 1}/{len(by_day)} days ({d}), rows {n_api:,}")

    # 3) sidecar: rvol_open5 per candidate key
    v = pd.read_sql("SELECT symbol, day, vol FROM open5", side)
    side.close()
    vmap = {(s, d): x for s, d, x in zip(v['symbol'], v['day'], v['vol'])}
    out = []
    for s, d in zip(keys['symbol'], keys['date']):
        i = idx.get(d)
        cur = vmap.get((s, d))
        prior = [vmap.get((s, cal[j])) for j in range(max(0, (i or 0) - PRIOR_DAYS), i or 0)] if i is not None else []
        prior = [p for p in prior if p is not None and p > 0]
        rv = (cur / (sum(prior) / len(prior))) if (cur and len(prior) >= MIN_PRIOR) else float('nan')
        out.append({'symbol': s, 'date': d, 'vol_open5': cur, 'n_prior': len(prior), 'rvol_open5': rv})
    o = pd.DataFrame(out)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    o.to_csv(out_csv, index=False)
    print(f"sidecar -> {out_csv}: {len(o):,} keys | rvol available {int(o['rvol_open5'].notna().sum()):,} "
          f"({o['rvol_open5'].notna().mean() * 100:.1f}%) | median rvol {o['rvol_open5'].median():.2f} | "
          f"NaN reasons: no cur {int(o['vol_open5'].isna().sum())}, prior<{MIN_PRIOR} {int((o['n_prior'] < MIN_PRIOR).sum())}")
    print("BUILD DONE")


if __name__ == '__main__':
    main()
