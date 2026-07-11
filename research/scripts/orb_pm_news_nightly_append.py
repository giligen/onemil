"""Nightly PM$ + news append for BT ground-truth parity (2026-07-10).

The BT pipeline reads premarket dollar-volume and news flags from
data/research CSVs (globbed). Without nightly appends those CSVs freeze at
their backfill date and every LATER day fail-opens to mult 1.0 in BT while
live boosts — silently skewing the nightly BT-vs-live promotion numbers on
exactly the trades the sizing edge lives in.

Called by orb_backtest.py after feature regen (fail-soft there). Appends
only MISSING (symbol, day) pairs from the latest features CSV to:
  data/research/orb_premarket_dollar_vol_nightly.csv
  data/research/orb_news_catalyst_nightly.csv
(the pipeline's globs pick these up automatically).

Usage: python3 research/scripts/orb_pm_news_nightly_append.py [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / '.env')

from trading.orb_pm_mult import compute_pm_dollar_vol  # noqa: E402

PM_GLOB = 'data/research/orb_premarket_dollar_vol_*.csv'
NEWS_GLOB = 'data/research/orb_news_catalyst_*.csv'
PM_OUT = ROOT / 'data/research/orb_premarket_dollar_vol_nightly.csv'
NEWS_OUT = ROOT / 'data/research/orb_news_catalyst_nightly.csv'


def _latest_features_csv() -> str:
    paths = [p for p in sorted(glob.glob('analysis_results/orb_features_*.csv'))
             if 'corrmatrix' not in p]
    if not paths:
        raise SystemExit('no analysis_results/orb_features_*.csv found')
    return paths[-1]


def _covered(pattern: str) -> set:
    pairs = set()
    for p in glob.glob(pattern):
        try:
            df = pd.read_csv(p, usecols=['symbol', 'day'])
            pairs |= set(zip(df['symbol'], df['day']))
        except Exception as e:
            print(f"WARNING: unreadable {p}: {e}")
    return pairs


def _news_window_utc(day: str):
    d = pd.Timestamp(day)
    st = ((d - pd.Timedelta(days=1)).tz_localize('America/New_York')
          + pd.Timedelta(hours=15)).tz_convert('UTC')
    en = (d.tz_localize('America/New_York')
          + pd.Timedelta(hours=9, minutes=35)).tz_convert('UTC')
    return st, en


def fetch_news(pairs_by_day: dict) -> list:
    """One paginated news call per day for all its symbols (backfill parity:
    research/scripts/orb_news_backfill.py)."""
    import requests
    k = os.environ['ALPACA_API_KEY']
    s = os.environ['ALPACA_API_SECRET']
    hdr = {'APCA-API-KEY-ID': k, 'APCA-API-SECRET-KEY': s}
    rows = []
    for day, syms in sorted(pairs_by_day.items()):
        st, en = _news_window_utc(day)
        arts, token = [], None
        for _ in range(6):
            p = {'symbols': ','.join(sorted(syms)), 'start': st.isoformat(),
                 'end': en.isoformat(), 'limit': 50, 'sort': 'desc'}
            if token:
                p['page_token'] = token
            r = requests.get('https://data.alpaca.markets/v1beta1/news',
                             params=p, headers=hdr, timeout=(5, 30))
            r.raise_for_status()
            j = r.json()
            arts += j.get('news', [])
            token = j.get('next_page_token')
            if not token:
                break
        per = {sym: [] for sym in syms}
        for a in arts:
            for sym in a.get('symbols', []):
                if sym in per:
                    per[sym].append(a)
        for sym in sorted(syms):
            aa = per[sym]
            rows.append({
                'symbol': sym, 'day': day, 'n_articles': len(aa),
                'earliest': min((a['created_at'] for a in aa), default=''),
                'latest': max((a['created_at'] for a in aa), default=''),
                'headlines': ' || '.join((a['headline'] or '')[:110]
                                         for a in aa[:4])})
    return rows


def fetch_pm(pairs_by_day: dict) -> list:
    """Premarket dollar volume via historical 1-min bars, one batched call
    per day (4:00-9:29 ET), summed by the SHARED helper."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    client = StockHistoricalDataClient(os.environ['ALPACA_API_KEY'],
                                       os.environ['ALPACA_API_SECRET'])
    rows = []
    for day, syms in sorted(pairs_by_day.items()):
        d = pd.Timestamp(day).tz_localize('America/New_York')
        req = StockBarsRequest(
            symbol_or_symbols=sorted(syms),
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=(d + pd.Timedelta(hours=4)).tz_convert('UTC'),
            end=(d + pd.Timedelta(hours=9, minutes=30)).tz_convert('UTC'),
            feed=DataFeed.SIP)
        bars = client.get_stock_bars(req)
        data = bars.data if hasattr(bars, 'data') else {}
        for sym in sorted(syms):
            blist = data.get(sym, [])
            if blist:
                df = pd.DataFrame([{
                    'timestamp': b.timestamp, 'close': float(b.close),
                    'vwap': float(b.vwap) if b.vwap is not None else None,
                    'volume': int(b.volume)} for b in blist])
                dv = compute_pm_dollar_vol(df)
            else:
                dv = None
            rows.append({'symbol': sym, 'day': day, 'pm_dollar_vol': dv})
    return rows


def _append(out_path: Path, rows: list, dry: bool) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if dry:
        print(f"DRY RUN: would append {len(df)} rows to {out_path.name}")
        return
    header = not out_path.exists()
    df.to_csv(out_path, mode='a', header=header, index=False)
    print(f"appended {len(df)} rows to {out_path.name}")


def refresh_class_map(symbols, dry: bool) -> None:
    """Append newly-listed symbols to the asset-class map so BT classifies
    them like live does (live has an API fallback; BT is map-only — without
    this, a newsy new listing boosts live but not in BT ground truth)."""
    import csv as _csv
    from trading.orb_asset_class import DEFAULT_CLASS_MAP, classify_asset
    import requests
    known = set()
    try:
        with open(DEFAULT_CLASS_MAP, newline='') as fh:
            known = {r['symbol'] for r in _csv.DictReader(fh)}
    except Exception as e:
        print(f"WARNING: class map unreadable ({e}) — skipping refresh")
        return
    missing = sorted(set(symbols) - known)
    if not missing:
        print(f"class map: coverage complete ({len(known)} symbols)")
        return
    hdr = {'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'],
           'APCA-API-SECRET-KEY': os.environ['ALPACA_API_SECRET']}
    rows = []
    for sym in missing:
        try:
            r = requests.get(f'https://api.alpaca.markets/v2/assets/{sym}',
                             headers=hdr, timeout=(5, 15))
            name = r.json().get('name', '') if r.ok else ''
        except Exception:
            name = ''
        rows.append([sym, classify_asset(sym, name), (name or '')[:90]])
    if dry:
        print(f"DRY RUN: would append {len(rows)} class-map rows: "
              f"{[r[:2] for r in rows[:10]]}")
        return
    with open(DEFAULT_CLASS_MAP, 'a', newline='') as fh:
        _csv.writer(fh).writerows(rows)
    print(f"class map: appended {len(rows)} new symbols "
          f"({sum(1 for r in rows if r[1] == 'unknown')} unknown)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    feats = pd.read_csv(_latest_features_csv(), usecols=['symbol', 'date'])
    feats['day'] = feats['date'].astype(str).str[:10]
    all_pairs = set(zip(feats['symbol'], feats['day']))
    refresh_class_map({s for s, _ in all_pairs}, args.dry_run)

    for label, pattern, out, fetch in (
            ('PM$', PM_GLOB, PM_OUT, fetch_pm),
            ('news', NEWS_GLOB, NEWS_OUT, fetch_news)):
        missing = all_pairs - _covered(pattern)
        if not missing:
            print(f"{label}: coverage complete ({len(all_pairs)} pairs)")
            continue
        by_day: dict = {}
        for sym, day in missing:
            by_day.setdefault(day, set()).add(sym)
        print(f"{label}: {len(missing)} missing pairs over "
              f"{len(by_day)} days — fetching")
        _append(out, fetch(by_day), args.dry_run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
