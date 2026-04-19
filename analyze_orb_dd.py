"""Investigate what the $-170K / $-130K drawdown actually looks like.

Questions:
  1. Is it one or two catastrophic days, or a long losing streak?
  2. When does it happen? Which dates?
  3. What does the daily-aggregated equity curve look like?
"""
from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from study_orb import SPLITS
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, ADAPTIVE_MULT_MIN, ADAPTIVE_MULT_MAX,
    fit_quintile_cutoffs, assign_quintile,
)


def build_adaptive_equity(df: pd.DataFrame, tr_s: str, tr_e: str,
                          te_s: str, te_e: str) -> pd.DataFrame:
    """Return test-period trades with sized_pnl, ordered by date, for 1 split."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    train = df[(df['date'] >= tr_s) & (df['date'] <= tr_e)]
    test = df[(df['date'] >= te_s) & (df['date'] <= te_e)]

    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    train = df[(df['date'] >= tr_s) & (df['date'] <= tr_e)]
    test = df[(df['date'] >= te_s) & (df['date'] <= te_e)]

    test_kept = test[test['_composite'] >= FILTER_THRESHOLD].copy()
    train_kept = train[train['_composite'] >= FILTER_THRESHOLD].copy()

    cutoffs = fit_quintile_cutoffs(train_kept['_composite'])
    train_kept['_quintile'] = assign_quintile(train_kept['_composite'], cutoffs)
    test_kept['_quintile'] = assign_quintile(test_kept['_composite'], cutoffs)

    train_overall_avg = float(train_kept['pnl'].mean())
    mult_by_q = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        q_sub = train_kept[train_kept['_quintile'] == q]
        if len(q_sub) == 0 or train_overall_avg <= 0:
            mult_by_q[q] = 1.0
        else:
            r = float(q_sub['pnl'].mean()) / train_overall_avg
            mult_by_q[q] = max(ADAPTIVE_MULT_MIN, min(ADAPTIVE_MULT_MAX, r))

    test_kept['_mult'] = test_kept['_quintile'].map(mult_by_q)
    test_kept['_sized_pnl'] = test_kept['pnl'] * test_kept['_mult']
    test_kept = test_kept.sort_values('date').reset_index(drop=True)
    return test_kept


def find_worst_stretch(daily: pd.DataFrame) -> dict:
    """Identify the peak-to-trough stretch that produced max DD."""
    daily = daily.sort_values('date').reset_index(drop=True)
    daily['cum'] = daily['daily_pnl'].cumsum()
    running_peak = -np.inf
    peak_idx = 0
    worst_peak_idx = 0
    worst_trough_idx = 0
    worst_dd = 0.0
    for i, cum in enumerate(daily['cum']):
        if cum > running_peak:
            running_peak = cum
            peak_idx = i
        dd = cum - running_peak
        if dd < worst_dd:
            worst_dd = dd
            worst_peak_idx = peak_idx
            worst_trough_idx = i
    return {
        'worst_dd': worst_dd,
        'peak_date': daily.iloc[worst_peak_idx]['date'],
        'peak_equity': daily.iloc[worst_peak_idx]['cum'],
        'trough_date': daily.iloc[worst_trough_idx]['date'],
        'trough_equity': daily.iloc[worst_trough_idx]['cum'],
        'days_span': worst_trough_idx - worst_peak_idx,
        'daily_during_stretch': daily.iloc[worst_peak_idx:worst_trough_idx + 1].copy(),
    }


def main():
    csv = 'analysis_results/orb_features_20260418_1715.csv'
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'date'])

    for split_name, tr_s, tr_e, te_s, te_e in SPLITS:
        print(f"\n{'='*100}")
        print(f"SPLIT: {split_name}")
        print(f"{'='*100}")
        test_kept = build_adaptive_equity(df, tr_s, tr_e, te_s, te_e)

        # Aggregate by date
        daily = test_kept.groupby('date').agg(
            n_trades=('pnl', 'count'),
            daily_pnl=('_sized_pnl', 'sum'),
            daily_pnl_unsized=('pnl', 'sum'),
        ).reset_index()

        total_pnl = daily['daily_pnl'].sum()
        print(f"Test period: {te_s} → {te_e}")
        print(f"Trading days: {len(daily)}, Total trades: {test_kept.shape[0]}, "
              f"Total P&L: ${total_pnl:+,.0f}")

        # Find worst DD stretch (daily-aggregated)
        stretch = find_worst_stretch(daily)
        print(f"\n--- WORST DD STRETCH (daily-aggregated equity) ---")
        print(f"  Max DD: ${stretch['worst_dd']:+,.0f}")
        print(f"  Peak:   {stretch['peak_date'].date()} "
              f"(equity ${stretch['peak_equity']:+,.0f})")
        print(f"  Trough: {stretch['trough_date'].date()} "
              f"(equity ${stretch['trough_equity']:+,.0f})")
        print(f"  Span:   {stretch['days_span']} trading days")

        # Per-day breakdown during the stretch
        stretch_df = stretch['daily_during_stretch']
        print(f"\n  Top 10 worst days WITHIN the stretch (daily_pnl):")
        worst = stretch_df.nsmallest(10, 'daily_pnl')[
            ['date', 'n_trades', 'daily_pnl', 'daily_pnl_unsized']]
        for _, r in worst.iterrows():
            print(f"    {r['date'].date()}  "
                  f"n={int(r['n_trades']):>3}  "
                  f"sized=${r['daily_pnl']:>+9,.0f}  "
                  f"unsized=${r['daily_pnl_unsized']:>+9,.0f}")

        # Worst single days (entire test period)
        print(f"\n  Worst 10 single days in ENTIRE test period:")
        worst_all = daily.nsmallest(10, 'daily_pnl')[
            ['date', 'n_trades', 'daily_pnl', 'daily_pnl_unsized']]
        for _, r in worst_all.iterrows():
            print(f"    {r['date'].date()}  "
                  f"n={int(r['n_trades']):>3}  "
                  f"sized=${r['daily_pnl']:>+9,.0f}  "
                  f"unsized=${r['daily_pnl_unsized']:>+9,.0f}")

        # Best 10 days for context
        print(f"\n  Best 10 single days in ENTIRE test period:")
        best_all = daily.nlargest(10, 'daily_pnl')[
            ['date', 'n_trades', 'daily_pnl']]
        for _, r in best_all.iterrows():
            print(f"    {r['date'].date()}  "
                  f"n={int(r['n_trades']):>3}  "
                  f"sized=${r['daily_pnl']:>+9,.0f}")


if __name__ == '__main__':
    main()
