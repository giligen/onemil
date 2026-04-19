"""Analyze ORB performance by entry_price bucket.

Questions:
  1. Does raw edge vary by price? (Win rate, avg pnl_pct)
  2. Does filter + cap5 + adaptive pipeline improve when restricted to >$10 or >$15?
  3. Where's the sweet spot?
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from study_orb import SPLITS
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, ADAPTIVE_MULT_MIN, ADAPTIVE_MULT_MAX,
    fit_quintile_cutoffs, assign_quintile,
)

ACCOUNT = 100_000
MAX_CONCURRENT = 3           # best config from prior study
RISK_PER_TRADE = 2000        # best config from prior study
OLD_POSITION_USD = 50_000.0
MIN_STOP_PCT = 1.0
Q_PREF_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}


def apply_risk_parity(df, risk, cap):
    df = df.copy()
    stop_pct = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    uncap = risk / (stop_pct / 100.0)
    df['_rp_position'] = uncap.clip(upper=cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POSITION_USD
    return df


def fit_adaptive(train_kept, pnl_col='_rp_pnl'):
    avg = float(train_kept[pnl_col].mean()) if len(train_kept) else 1.0
    mults = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        sub = train_kept[train_kept['_quintile'] == q]
        if len(sub) == 0 or avg <= 0:
            mults[q] = 1.0
            continue
        mults[q] = max(ADAPTIVE_MULT_MIN, min(ADAPTIVE_MULT_MAX,
                                               float(sub[pnl_col].mean()) / avg))
    return mults


def select_top_k(dg, k):
    d = dg.copy()
    d['_q_rank'] = d['_quintile'].map(Q_PREF_ORDER)
    return d.sort_values(['_q_rank', '_composite'],
                          ascending=[True, False]).head(k)


def compute_dd(daily):
    if len(daily) == 0:
        return 0.0
    d = daily.sort_values('date').reset_index(drop=True)
    d['cum'] = d['daily_pnl'].cumsum()
    peak = -np.inf
    dd = 0.0
    for c in d['cum']:
        peak = max(peak, c)
        dd = min(dd, c - peak)
    return dd


def run_pipeline_filtered(df, price_floor):
    """Full cap3 + Q4-pref + adaptive pipeline, restricted to entry_price >= price_floor."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['entry_price'] >= price_floor]
    if len(df) < 100:
        return None

    per_pos_cap = ACCOUNT / MAX_CONCURRENT
    df = apply_risk_parity(df, RISK_PER_TRADE, per_pos_cap)

    per_split = []
    for _, tr_s, tr_e, te_s, te_e in SPLITS:
        train = df[(df['date'] >= tr_s) & (df['date'] <= tr_e)]
        test = df[(df['date'] >= te_s) & (df['date'] <= te_e)]
        if len(train) < 50 or len(test) < 50:
            continue
        params = fit_z_params(train, FILTER_FEATURES)
        df2 = df.copy()
        df2['_composite'] = composite_score(df2, params)
        train = df2[(df2['date'] >= tr_s) & (df2['date'] <= tr_e)]
        test = df2[(df2['date'] >= te_s) & (df2['date'] <= te_e)]
        test_kept = test[test['_composite'] >= FILTER_THRESHOLD].copy()
        train_kept = train[train['_composite'] >= FILTER_THRESHOLD].copy()
        if len(train_kept) < 10 or len(test_kept) < 10:
            continue
        cutoffs = fit_quintile_cutoffs(train_kept['_composite'])
        train_kept['_quintile'] = assign_quintile(train_kept['_composite'], cutoffs)
        test_kept['_quintile'] = assign_quintile(test_kept['_composite'], cutoffs)
        mults = fit_adaptive(train_kept)
        sel = pd.concat([select_top_k(dg, MAX_CONCURRENT)
                         for _, dg in test_kept.groupby('date')])
        sel['_sized_pnl'] = sel.apply(
            lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)
        daily = sel.groupby('date').agg(daily_pnl=('_sized_pnl', 'sum'),
                                         n=('_rp_pnl', 'count')).reset_index()
        per_split.append({
            'pnl': float(daily['daily_pnl'].sum()),
            'dd': compute_dd(daily),
            'worst_day': float(daily['daily_pnl'].min()) if len(daily) else 0,
            'n_trades': len(sel),
            'avg_trade_pnl': float(sel['_sized_pnl'].mean()) if len(sel) else 0,
            'wr': float((sel['_sized_pnl'] > 0).mean() * 100) if len(sel) else 0,
        })
    if not per_split:
        return None
    pnls = [s['pnl'] for s in per_split]
    return {
        'sum_pnl': sum(pnls),
        'min_split': min(pnls),
        'worst_dd': min(s['dd'] for s in per_split),
        'worst_day': min(s['worst_day'] for s in per_split),
        'total_trades': sum(s['n_trades'] for s in per_split),
        'avg_trade_pnl': float(np.mean([s['avg_trade_pnl'] for s in per_split])),
        'wr': float(np.mean([s['wr'] for s in per_split])),
        'per_split_pnl': pnls,
    }


def main():
    import glob
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'pnl_pct', 'date', 'range_size_pct',
                                     'entry_price'])
    print(f"Loaded {len(df):,} trades\n")

    # --- Step 1: RAW per-trade stats by price bucket ---
    print(f"{'='*95}")
    print(f"RAW PER-TRADE STATS BY PRICE BUCKET (no filter, no cap, no sizing)")
    print(f"Flat $50K position, as originally simulated. Note: pnl_pct is sizing-independent.")
    print(f"{'='*95}")
    buckets = [(0, 3), (3, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 1000)]
    print(f"  {'bucket':<14} {'n':>6} {'% of all':>9} {'WR':>6} "
          f"{'avg pnl_pct':>12} {'med pnl_pct':>12} "
          f"{'avg pnl($)':>12} {'total P&L':>14}")
    for lo, hi in buckets:
        sub = df[(df['entry_price'] >= lo) & (df['entry_price'] < hi)]
        if len(sub) == 0:
            continue
        wr = (sub['pnl'] > 0).mean() * 100
        pct_of_all = len(sub) / len(df) * 100
        print(f"  ${lo:>2}-${hi:<9} {len(sub):>6} {pct_of_all:>7.1f}% {wr:>5.1f}% "
              f"{sub['pnl_pct'].mean():>+11.2f}% {sub['pnl_pct'].median():>+11.2f}% "
              f"${sub['pnl'].mean():>+10,.0f} ${sub['pnl'].sum():>+12,.0f}")

    # --- Step 2: Cumulative "price >= X" view ---
    print(f"\n{'='*95}")
    print(f"CUMULATIVE: price >= X (filter out everything below threshold)")
    print(f"{'='*95}")
    print(f"  {'price >= ':<12} {'n':>6} {'% kept':>8} {'WR':>6} "
          f"{'avg pnl_pct':>12} {'total pnl ($)':>14}")
    for x in [0, 3, 5, 7, 10, 12, 15, 20, 25]:
        sub = df[df['entry_price'] >= x]
        if len(sub) == 0:
            continue
        wr = (sub['pnl'] > 0).mean() * 100
        kept = len(sub) / len(df) * 100
        print(f"  ${x:<10} {len(sub):>6} {kept:>6.1f}% {wr:>5.1f}% "
              f"{sub['pnl_pct'].mean():>+11.2f}% ${sub['pnl'].sum():>+12,.0f}")

    # --- Step 3: Run full pipeline with different price floors ---
    print(f"\n{'='*95}")
    print(f"FULL PIPELINE (N={MAX_CONCURRENT}, risk=${RISK_PER_TRADE:,.0f}, filter+cap+adaptive)")
    print(f"{'='*95}")
    print(f"  {'price >=':<10} {'Sum P&L':>13} {'Min split':>13} {'Worst DD':>12} "
          f"{'Worst day':>12} {'trades':>7} {'avg $':>10} {'WR':>6}")
    for x in [0, 3, 5, 7, 10, 12, 15, 20]:
        r = run_pipeline_filtered(df, x)
        if r is None:
            print(f"  ${x:<8} (insufficient data)")
            continue
        print(f"  ${x:<8} ${r['sum_pnl']:>+11,.0f} ${r['min_split']:>+11,.0f} "
              f"${r['worst_dd']:>+10,.0f} ${r['worst_day']:>+10,.0f} "
              f"{r['total_trades']:>7} ${r['avg_trade_pnl']:>+8,.0f} "
              f"{r['wr']:>4.1f}%")


if __name__ == '__main__':
    main()
