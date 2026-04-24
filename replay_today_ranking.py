#!/usr/bin/env python3
"""Replicate live's check_entries ranking for 2026-04-22, using cache.db
1-min bars and cache.db daily bars, to see what the ranking+dedup would
actually produce."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from study_orb import _session_open_timestamp
from trading.orb_filter import load_feature_params, composite_score, assign_quintile
from trading.orb_correlation import dedup_candidates


CACHE_DB = 'data/cache.db'


def load_intraday_for_day(day: str, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Load ONLY the given day's bars for the given symbols. Uses bar_date
    column which is indexed, and filters by symbol — so we never scan the
    whole 39M-row intraday table."""
    conn = sqlite3.connect(CACHE_DB)
    placeholders = ','.join('?' * len(symbols))
    df = pd.read_sql_query(
        f"SELECT symbol, timestamp, open, high, low, close, volume "
        f"FROM intraday_bars_1min "
        f"WHERE bar_date = ? AND symbol IN ({placeholders}) "
        f"ORDER BY symbol, timestamp",
        conn, params=[day] + symbols,
    )
    conn.close()
    if df.empty:
        return {}
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return {s: g.reset_index(drop=True) for s, g in df.groupby('symbol')}


def load_daily_for_symbols(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Load daily bars ONLY for specified symbols."""
    conn = sqlite3.connect(CACHE_DB)
    placeholders = ','.join('?' * len(symbols))
    df = pd.read_sql_query(
        f"SELECT symbol, bar_date, open, high, low, close, volume "
        f"FROM daily_bars WHERE symbol IN ({placeholders}) "
        f"ORDER BY symbol, bar_date",
        conn, params=symbols,
    )
    conn.close()
    if df.empty:
        return {}
    df['bar_date'] = pd.to_datetime(df['bar_date'])
    return {s: g.reset_index(drop=True) for s, g in df.groupby('symbol')}


def compute_range_and_features(bars_df: pd.DataFrame, sym: str, day: str,
                                daily_by_sym: Dict[str, pd.DataFrame]
                                ) -> Optional[Tuple[Dict[str, float], float, float]]:
    """Returns (features, range_high, range_low) mirroring live's path."""
    if bars_df.empty:
        return None
    open_ts = _session_open_timestamp(bars_df)
    if open_ts is None:
        return None
    from datetime import timedelta as td
    range_end = open_ts + td(minutes=5)
    mask = (bars_df['timestamp'] >= open_ts) & (bars_df['timestamp'] < range_end)
    rb = bars_df.loc[mask].reset_index(drop=True)
    if len(rb) < 5:
        return None

    rh = float(rb['high'].max()); rl = float(rb['low'].min())
    if rh <= 0 or rl <= 0 or rh <= rl:
        return None
    avg_bar_range_pct = float(((rb['high'] - rb['low']) / rb['close']).mean() * 100.0)
    range_close = float(rb['close'].iloc[-1])
    range_open = float(rb['open'].iloc[0])
    range_volume = float(rb['volume'].sum())

    sym_daily = daily_by_sym.get(sym)
    if sym_daily is None or sym_daily.empty:
        return None
    dtx = pd.Timestamp(day)
    prev_mask = sym_daily['bar_date'] < dtx
    if not prev_mask.any():
        return None
    prior_df = sym_daily.loc[prev_mask]
    prev_bar = prior_df.iloc[-1]
    pc = float(prev_bar['close']); ph = float(prev_bar['high']); pl = float(prev_bar['low'])
    window = prior_df.tail(20)
    high_20d = float(window['high'].max()) if len(window) else 0.0

    ref_open = range_open if range_open > 0 else rh
    feats: Dict[str, float] = {
        'range_size_pct': (rh - rl) / ref_open * 100.0,
        'range_total_volume': range_volume,
        'range_avg_bar_range_pct': avg_bar_range_pct,
        'range_close_position': (range_close - rl) / (rh - rl) if rh > rl else 0.5,
    }
    if pc > 0:
        feats['gap_pct'] = (ref_open - pc) / pc * 100.0
    if ph > pl > 0:
        feats['prev_day_close_position'] = (pc - pl) / (ph - pl)
    if high_20d > 0:
        feats['price_vs_20d_high_pct'] = (ref_open - high_20d) / high_20d * 100.0
    return feats, rh, rl


def main():
    day = '2026-04-22'
    with open('orb.yaml') as f:
        cfg = yaml.safe_load(f)
    z_params = load_feature_params(cfg['filter'])
    cutoffs = cfg['quintile_cutoffs']
    ranking_order = cfg['ranking']['order']
    max_concurrent = cfg['sizing']['max_concurrent']
    filter_threshold = cfg['filter']['threshold']

    # Live candidates (from logs) — the 42 symbols with range_complete today
    live_universe = [
        'LUNL', 'APLX', 'UNHG', 'RDWU', 'ALMU', 'BTM', 'WSHP',
        'CCUP', 'OKLL', 'CRMX', 'RDTL', 'RGTX', 'ADBG', 'AVXX', 'XNDU',
        'MRAL', 'BKKT', 'HOOG', 'WOLF', 'XXRP', 'ONDL', 'BITU', 'ONDG',
        'QBTX', 'FLWS', 'SMU', 'DFDV', 'MSTW', 'KYTX', 'ETHT', 'NBIG',
        'VNCE',
    ]
    # There are 42 total — we have 32 above from the log excerpt. Pick best effort.

    print(f"Loading cache.db (scoped to {len(live_universe)} symbols)...")
    daily_by_sym = load_daily_for_symbols(live_universe)
    intraday_by_sym = load_intraday_for_day(day, live_universe)
    print(f"  daily: {len(daily_by_sym)} syms, intraday: {len(intraday_by_sym)} syms")

    scored: List[Tuple[str, str, float]] = []  # (symbol, quintile, composite)
    skipped: List[Tuple[str, str]] = []         # (symbol, reason)

    for sym in live_universe:
        bars = intraday_by_sym.get(sym, pd.DataFrame())
        if bars.empty:
            skipped.append((sym, 'no_bars_in_cache'))
            continue
        result = compute_range_and_features(bars, sym, day, daily_by_sym)
        if result is None:
            skipped.append((sym, 'incomplete_range_or_no_daily'))
            continue
        feats, rh, rl = result
        comp = composite_score(feats, z_params)
        if comp is None:
            missing = [k for k in z_params if k not in feats]
            skipped.append((sym, f"missing_feats={missing}"))
            continue
        if comp < filter_threshold:
            skipped.append((sym, f"below_threshold_{comp:.3f}"))
            continue
        q = assign_quintile(comp, cutoffs)
        scored.append((sym, q, comp))

    # Rank + dedup (same code as live)
    q_rank = {q: i for i, q in enumerate(ranking_order)}
    scored.sort(key=lambda t: (q_rank.get(t[1], 99), -t[2]))
    ranked_syms = [t[0] for t in scored]
    top = dedup_candidates(ranked_syms, max_keep=max_concurrent,
                           by_family=True, by_super_group=True)

    print(f"\n=== Ranked & filtered candidates for {day} ===")
    print(f"{'rank':>4s} {'sym':8s} {'Q':3s} {'composite':>10s}  kept?")
    print('-' * 50)
    kept_set = set(top)
    for i, (sym, q, c) in enumerate(scored, 1):
        mark = '✓ KEEP' if sym in kept_set else '  drop'
        print(f"{i:>4d} {sym:8s} {q:3s} {c:>10.4f}  {mark}")

    print(f"\nFinal picks: {top}")
    print(f"Live actual: ['CRMX', 'BKKT', 'RGTX', 'BITU']")

    print(f"\n=== Skipped ({len(skipped)}) ===")
    for sym, reason in skipped[:20]:
        print(f"  {sym:8s}  {reason}")


if __name__ == '__main__':
    main()
