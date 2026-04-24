#!/usr/bin/env python3
"""Head-to-head audit: BT extract_features vs Live _compute_features.

For a target (symbol, date), run both paths against the SAME source of cached
bars and print per-feature diffs + composite + quintile.

If they produce the same numbers on identical input → live & BT agree in
principle; any live-vs-BT divergence in production came from data-source
drift (e.g. WS bars missing, stale 20d cache), not algorithmic skew.
If they differ → we have a real code bug.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from study_orb_features import extract_features
from trading.orb_filter import FeatureParam, load_feature_params, composite_score, assign_quintile


CACHE_DB = 'data/cache.db'


def load_sym_intraday(sym: str, day: str, db_path: str = CACHE_DB) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, volume FROM intraday_bars_1min "
        "WHERE symbol=? AND bar_date=? ORDER BY timestamp",
        conn, params=(sym, day),
    )
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df.reset_index(drop=True)


def live_compute(bars_df: pd.DataFrame, sym: str, day: str,
                 daily_by_sym: Dict[str, pd.DataFrame]) -> Optional[Dict[str, float]]:
    """Replicate live path: _ingest_bars + _get_feature_context + _compute_features
    from the same cache.db daily+intraday data BT uses, so the only thing we're
    comparing is the computation logic, not the data source."""
    from study_orb import _session_open_timestamp
    from datetime import timedelta as td

    if bars_df.empty:
        return None
    open_ts = _session_open_timestamp(bars_df)
    if open_ts is None:
        return None
    range_end = open_ts + td(minutes=5)
    mask = (bars_df['timestamp'] >= open_ts) & (bars_df['timestamp'] < range_end)
    rb = bars_df.loc[mask].reset_index(drop=True)
    if len(rb) < 5:
        return None

    # --- _ingest_bars ---
    rh = float(rb['high'].max()); rl = float(rb['low'].min())
    range_volume = int(rb['volume'].sum())
    # live uses mean((h-l)/c) * 100 WITHOUT NaN guard
    avg_bar_range_pct = float(((rb['high'] - rb['low']) / rb['close']).mean() * 100.0)
    range_close = float(rb['close'].iloc[-1])
    range_open = float(rb['open'].iloc[0])

    # --- _get_feature_context ---
    sym_daily = daily_by_sym.get(sym)
    if sym_daily is None or sym_daily.empty:
        return None
    dtx = pd.Timestamp(day)
    prev_mask = sym_daily['bar_date'] < dtx
    if not prev_mask.any():
        return None
    # Live pulls ~25 bars via get_daily_bars_cached, so take all available prior.
    # Then window = bars_list[-20:]. No 5-day floor.
    prior_df = sym_daily.loc[prev_mask]
    prev_bar = prior_df.iloc[-1]
    pc = float(prev_bar['close']); ph = float(prev_bar['high']); pl = float(prev_bar['low'])
    window = prior_df.tail(20)
    high_20d = float(window['high'].max()) if len(window) else 0.0

    # --- _compute_features ---
    ref_open = range_open if range_open > 0 else rh
    feats: Dict[str, float] = {
        'range_size_pct': (rh - rl) / ref_open * 100.0,
        'range_total_volume': float(range_volume),
        'range_avg_bar_range_pct': avg_bar_range_pct,
        'range_close_position': (
            (range_close - rl) / (rh - rl) if rh > rl else 0.5
        ),
    }
    if pc > 0:
        feats['gap_pct'] = (ref_open - pc) / pc * 100.0
    if ph > pl > 0:
        feats['prev_day_close_position'] = (pc - pl) / (ph - pl)
    if high_20d > 0:
        feats['price_vs_20d_high_pct'] = (ref_open - high_20d) / high_20d * 100.0
    return feats


def bt_compute(bars_df: pd.DataFrame, sym: str, day: str,
               daily_by_sym: Dict[str, pd.DataFrame],
               spy_intraday: pd.DataFrame,
               spy_daily: pd.DataFrame) -> Optional[Dict[str, float]]:
    return extract_features(bars_df, sym, day, daily_by_sym, spy_intraday, spy_daily)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', required=True)
    p.add_argument('--date', required=True, help='YYYY-MM-DD')
    p.add_argument('--orb-yaml', default='orb.yaml')
    args = p.parse_args()

    print(f"Loading cache.db (scoped to {args.symbol} + SPY)...")
    conn = sqlite3.connect(CACHE_DB)
    daily_df = pd.read_sql_query(
        "SELECT symbol, bar_date, open, high, low, close, volume FROM daily_bars "
        "WHERE symbol IN (?, 'SPY') ORDER BY symbol, bar_date",
        conn, params=(args.symbol,))
    daily_df['bar_date'] = pd.to_datetime(daily_df['bar_date'])
    daily_by_sym: Dict[str, pd.DataFrame] = {
        s: g.reset_index(drop=True) for s, g in daily_df.groupby('symbol')
    }
    # SPY intraday only for the target date (BT only needs 9:30-9:34)
    spy_intraday = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, volume FROM intraday_bars_1min "
        "WHERE symbol='SPY' AND bar_date=? ORDER BY timestamp",
        conn, params=(args.date,))
    spy_intraday['timestamp'] = pd.to_datetime(spy_intraday['timestamp'], utc=True)
    conn.close()
    spy_daily = daily_by_sym.get('SPY', pd.DataFrame())

    bars_df = load_sym_intraday(args.symbol, args.date)
    print(f"{args.symbol} {args.date}: {len(bars_df)} 1-min bars in cache")

    with open(args.orb_yaml) as f:
        cfg = yaml.safe_load(f)
    z_params = load_feature_params(cfg['filter'])
    cutoffs = cfg['quintile_cutoffs']

    print("\n=== BT extract_features ===")
    bt_feats = bt_compute(bars_df, args.symbol, args.date,
                          daily_by_sym, spy_intraday, spy_daily)
    print("\n=== Live _compute_features (same data source) ===")
    live_feats = live_compute(bars_df, args.symbol, args.date, daily_by_sym)

    keys_used_by_filter = list(z_params.keys())
    print(f"\nFeatures in z-score filter: {keys_used_by_filter}\n")

    if bt_feats is None or live_feats is None:
        print(f"BT feats is None: {bt_feats is None}, Live feats is None: {live_feats is None}")
        return

    # Per-feature diff
    header = f"{'feature':30s} {'BT':>14s} {'Live':>14s} {'diff':>14s}"
    print(header)
    print('-' * len(header))
    for k in keys_used_by_filter:
        bt_v = bt_feats.get(k, float('nan'))
        lv_v = live_feats.get(k, float('nan'))
        diff = (lv_v - bt_v) if (bt_v is not None and lv_v is not None) else float('nan')
        print(f"{k:30s} {bt_v:>14.6f} {lv_v:>14.6f} {diff:>14.6f}")

    # Composite
    bt_comp = composite_score(bt_feats, z_params)
    live_comp = composite_score(live_feats, z_params)
    bt_q = assign_quintile(bt_comp, cutoffs) if bt_comp is not None else 'N/A'
    live_q = assign_quintile(live_comp, cutoffs) if live_comp is not None else 'N/A'
    print()
    print(f"BT   composite = {bt_comp:+.6f} → {bt_q}")
    print(f"Live composite = {live_comp:+.6f} → {live_q}")


if __name__ == '__main__':
    main()
