#!/usr/bin/env python3
"""Walk-forward: should MACD wave skip entries on thin-SPY days?

Bull flag already has an analogous gate (Layer 4 post-fill exit uses SPY 3d
range < 0.8% + weak breakout volume). MACD wave has NONE. 2026-04-16
showed the failure mode: SPY vol_ratio 0.66 (thin), 5 entries in 4 minutes,
all losers (-$2,390).

Methodology (post-hoc filtering of the existing filtered result set):
  - Load macd_wave_results.csv (630 trades Jan'25-Apr'26, already passed
    current filters).
  - For each trade_date compute SPY 3-day range % and SPY daily volume
    ratio (day vol / trailing-20d avg, computed using ONLY bars < trade_date
    → no look-ahead).
  - For each threshold variant, filter out "skip" trades and recompute P&L.
  - Evaluate across 3 chronological train/test splits — require a robust
    winner (mean test Δ > 0 AND min test Δ > 0).

Limitation: post-hoc filtering doesn't re-simulate max_concurrent=5
reshuffling when we skip a trade. Signal direction is still valid.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, List, Tuple

import pandas as pd


DB_PATH = 'data/cache.db'
RESULTS_PATH = 'macd_wave_results.csv'

SPLITS = [
    ('A: H1\'25 → H2\'25-Apr\'26',  '2025-01-01', '2025-06-30', '2025-07-01', '2026-04-15'),
    ('B: Y2025 → Q1+Apr\'26',       '2025-01-01', '2025-12-31', '2026-01-01', '2026-04-15'),
    ('C: Jan-Sep\'25 → Oct\'25-Apr\'26','2025-01-01', '2025-09-30', '2025-10-01', '2026-04-15'),
]


def load_spy_regime() -> pd.DataFrame:
    """Return per-date SPY 3d_range_pct and vol_ratio (no look-ahead)."""
    conn = sqlite3.connect(DB_PATH)
    spy = pd.read_sql_query(
        "SELECT bar_date, open, high, low, close, volume "
        "FROM daily_bars WHERE symbol='SPY' ORDER BY bar_date", conn,
    )
    conn.close()
    spy['bar_date'] = pd.to_datetime(spy['bar_date'])
    spy['range_pct'] = (spy['high'] - spy['low']) / spy['low'] * 100
    # 3-day rolling — strictly prior 3 sessions (exclude current bar to
    # avoid using today's range for a filter applied today).
    spy['range_3d_prior'] = spy['range_pct'].rolling(window=3, min_periods=3).mean().shift(1)
    spy['vol_avg_20d_prior'] = spy['volume'].rolling(window=20, min_periods=5).mean().shift(1)
    spy['vol_ratio'] = spy['volume'] / spy['vol_avg_20d_prior']
    return spy[['bar_date', 'range_3d_prior', 'vol_ratio']].rename(
        columns={'bar_date': 'date'}
    )


def run_variant(trades: pd.DataFrame, spy: pd.DataFrame,
                min_3d: float = None, min_vol_ratio: float = None,
                name: str = 'V0') -> pd.DataFrame:
    """Return trades that PASS the regime filter."""
    merged = trades.merge(spy, on='date', how='left')
    mask = pd.Series(True, index=merged.index)
    if min_3d is not None:
        mask &= merged['range_3d_prior'] >= min_3d
    if min_vol_ratio is not None:
        mask &= merged['vol_ratio'] >= min_vol_ratio
    kept = merged[mask].copy()
    kept.attrs['name'] = name
    return kept


def stats(trades: pd.DataFrame) -> Dict:
    if len(trades) == 0:
        return {'n': 0, 'wr': 0, 'pnl': 0}
    return {
        'n': len(trades),
        'wr': (trades['pnl_dollar'] > 0).mean() * 100,
        'pnl': trades['pnl_dollar'].sum(),
    }


def filter_split(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df['date'] >= start) & (df['date'] <= end)]


def main() -> None:
    spy = load_spy_regime()
    trades = pd.read_csv(RESULTS_PATH)
    trades['date'] = pd.to_datetime(trades['date'])
    trades = trades[(trades['date'] >= '2025-01-01') & (trades['date'] <= '2026-04-15')].copy()

    print(f"Loaded {len(trades)} trades; SPY dates: {len(spy)}")
    print(f"SPY 3d_range_prior: {spy['range_3d_prior'].describe()[['min','25%','50%','75%','max']].to_dict()}")
    print(f"SPY vol_ratio:      {spy['vol_ratio'].describe()[['min','25%','50%','75%','max']].to_dict()}")
    print()

    # Baseline (no filter)
    v0 = run_variant(trades, spy, name='V0_baseline')

    # Variants
    variants = [
        ('V1a: 3d >= 0.6',        dict(min_3d=0.6)),
        ('V1b: 3d >= 0.8',        dict(min_3d=0.8)),
        ('V1c: 3d >= 1.0',        dict(min_3d=1.0)),
        ('V1d: 3d >= 1.2',        dict(min_3d=1.2)),
        ('V2a: vol_ratio >= 0.6', dict(min_vol_ratio=0.6)),
        ('V2b: vol_ratio >= 0.7', dict(min_vol_ratio=0.7)),
        ('V2c: vol_ratio >= 0.8', dict(min_vol_ratio=0.8)),
        ('V3a: 3d>=0.8 AND vr>=0.7', dict(min_3d=0.8, min_vol_ratio=0.7)),
        ('V3b: 3d>=1.0 AND vr>=0.7', dict(min_3d=1.0, min_vol_ratio=0.7)),
    ]

    print('=' * 100)
    print('BASELINE (V0) + VARIANT P&L PER SPLIT (train + test), plus Δ vs V0 on test')
    print('=' * 100)
    header = f"{'Variant':<30} " + " ".join(
        f"{n.split(':')[0]+'_test_Δ':>16}" for n, _ in [('A', None), ('B', None), ('C', None)]
    ) + f" {'Mean Δ':>13} {'Min Δ':>13}  Verdict"
    # Per-split summary
    print()
    print(f"{'Split':<32} {'V0 test P&L':>16}")
    print('-' * 50)
    v0_tests = {}
    for split_name, train_s, train_e, test_s, test_e in SPLITS:
        v0_t = stats(filter_split(v0, test_s, test_e))
        v0_tests[split_name] = v0_t['pnl']
        print(f"{split_name:<32} ${v0_t['pnl']:>+14,.0f} ({v0_t['n']}t, {v0_t['wr']:.0f}% WR)")

    print()
    print(f"{'Variant':<32} {'A Δ':>12} {'B Δ':>12} {'C Δ':>12} {'Mean':>12} {'Min':>12}  Verdict")
    print('-' * 102)
    for vname, kwargs in variants:
        vtrades = run_variant(trades, spy, name=vname, **kwargs)
        deltas = []
        for split_name, _, _, test_s, test_e in SPLITS:
            v_t = stats(filter_split(vtrades, test_s, test_e))
            delta = v_t['pnl'] - v0_tests[split_name]
            deltas.append(delta)
        mean_d = sum(deltas) / len(deltas)
        min_d = min(deltas)
        verdict = '✓ ROBUST' if min_d > 0 else ('⚠ mixed' if mean_d > 0 else '✗ losing')
        cells = " ".join(f"${d:>+10,.0f}" for d in deltas)
        print(f"{vname:<32} {cells} ${mean_d:>+10,.0f} ${min_d:>+10,.0f}  {verdict}")

    # Per-variant trade count drop (how aggressive is the filter?)
    print()
    print('=' * 80)
    print('FILTER AGGRESSIVENESS (kept / baseline) and OVERALL 15.5mo P&L')
    print('=' * 80)
    print(f"{'Variant':<32} {'kept/total':<14} {'drop':>6} {'P&L 15.5mo':>14} {'ΔvsV0':>13}")
    print('-' * 80)
    v0_all = stats(v0)
    ratio0 = f"{v0_all['n']}/{v0_all['n']}"
    print(f"{'V0_baseline':<32} {ratio0:<14} {'0%':>6} ${v0_all['pnl']:>+12,.0f} {'—':>13}")
    for vname, kwargs in variants:
        vtrades = run_variant(trades, spy, name=vname, **kwargs)
        s = stats(vtrades)
        dropped = v0_all['n'] - s['n']
        drop_pct = dropped / v0_all['n'] * 100
        delta = s['pnl'] - v0_all['pnl']
        ratio_str = f"{s['n']}/{v0_all['n']}"
        drop_str = f"{drop_pct:.0f}%"
        print(f"{vname:<32} {ratio_str:<14} {drop_str:>6} ${s['pnl']:>+12,.0f} ${delta:>+11,.0f}")

    # Bucket analysis: trades on "thin" days vs "normal" days — is there an EV gradient?
    print()
    print('=' * 80)
    print('EV GRADIENT: trade buckets by SPY 3d_range_prior')
    print('=' * 80)
    merged = trades.merge(spy, on='date', how='left')
    bins = [-0.01, 0.6, 0.8, 1.0, 1.2, 1.5, 10]
    labels = ['<0.6', '0.6-0.8', '0.8-1.0', '1.0-1.2', '1.2-1.5', '>=1.5']
    merged['bkt'] = pd.cut(merged['range_3d_prior'], bins=bins, labels=labels)
    g = merged.groupby('bkt', observed=True).agg(
        n=('pnl_dollar', 'count'),
        wins=('pnl_dollar', lambda s: (s > 0).sum()),
        pnl=('pnl_dollar', 'sum'),
        avg=('pnl_dollar', 'mean'),
    ).reset_index()
    g['wr'] = (g['wins'] / g['n'] * 100).round(0)
    print(f"{'3d_range':<12} {'n':>4} {'WR':>5} {'Total P&L':>13} {'Avg/trade':>13}")
    print('-' * 50)
    for _, r in g.iterrows():
        print(f"{r['bkt']:<12} {r['n']:>4} {r['wr']:>4.0f}% ${r['pnl']:>+11,.0f} ${r['avg']:>+11,.0f}")

    print()
    print('EV GRADIENT: trade buckets by SPY vol_ratio')
    vbins = [0, 0.5, 0.7, 0.9, 1.1, 1.5, 100]
    vlabels = ['<0.5', '0.5-0.7', '0.7-0.9', '0.9-1.1', '1.1-1.5', '>=1.5']
    merged['vbkt'] = pd.cut(merged['vol_ratio'], bins=vbins, labels=vlabels)
    gv = merged.groupby('vbkt', observed=True).agg(
        n=('pnl_dollar', 'count'),
        wins=('pnl_dollar', lambda s: (s > 0).sum()),
        pnl=('pnl_dollar', 'sum'),
        avg=('pnl_dollar', 'mean'),
    ).reset_index()
    gv['wr'] = (gv['wins'] / gv['n'] * 100).round(0)
    print(f"{'vol_ratio':<12} {'n':>4} {'WR':>5} {'Total P&L':>13} {'Avg/trade':>13}")
    print('-' * 50)
    for _, r in gv.iterrows():
        print(f"{r['vbkt']:<12} {r['n']:>4} {r['wr']:>4.0f}% ${r['pnl']:>+11,.0f} ${r['avg']:>+11,.0f}")


if __name__ == '__main__':
    main()
