"""Test: is the composite z-score a good RANKER for picking top-K trades
on high-concurrency days?

Approach:
  1. For each test-period day with N >= 3 kept trades:
     - Rank trades by composite z-score DESC (highest-z first)
     - Take top-K, bottom-K, random-K, and 'Q4-preferred' (prefer Q4 > Q5 > Q3 > ...)
     - Compute daily P&L for each selection
  2. Aggregate across test period: total P&L, worst-day damage, avg per day
  3. Special deep-dive on the 2025-11-20 catastrophic day

The key question: does top-K by composite z-score BEAT random-K?
And does 'Q4-preferred' BEAT top-by-composite (since Q4 > Q5 on avg)?
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
    FILTER_THRESHOLD, fit_quintile_cutoffs, assign_quintile,
)


def build_split_scored(df: pd.DataFrame, tr_s, tr_e, te_s, te_e) -> pd.DataFrame:
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
    test_kept['_quintile'] = assign_quintile(test_kept['_composite'], cutoffs)
    return test_kept.sort_values(['date', '_composite'], ascending=[True, False])


# Quintile-preference order: Q4 first (best), then Q5, Q3, Q2, Q1
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}


def select_top_k_by_composite(day_trades: pd.DataFrame, k: int) -> pd.DataFrame:
    return day_trades.nlargest(k, '_composite')


def select_bottom_k_by_composite(day_trades: pd.DataFrame, k: int) -> pd.DataFrame:
    return day_trades.nsmallest(k, '_composite')


def select_q4_preferred(day_trades: pd.DataFrame, k: int) -> pd.DataFrame:
    """Prefer Q4 > Q5 > Q3 > Q2 > Q1. Within same bucket, tie-break by composite DESC."""
    d = day_trades.copy()
    d['_q_rank'] = d['_quintile'].map(Q_ORDER)
    d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])
    return d.head(k)


def select_random_k(day_trades: pd.DataFrame, k: int, rng) -> pd.DataFrame:
    if len(day_trades) <= k:
        return day_trades
    return day_trades.sample(n=k, random_state=rng.integers(0, 1_000_000))


def run_selection(test_kept: pd.DataFrame, k: int, selector, name: str,
                  rng=None) -> dict:
    by_day = []
    for date, dg in test_kept.groupby('date'):
        if selector == 'random':
            sel = select_random_k(dg, k, rng)
        elif selector == 'top_composite':
            sel = select_top_k_by_composite(dg, k)
        elif selector == 'bottom_composite':
            sel = select_bottom_k_by_composite(dg, k)
        elif selector == 'q4_preferred':
            sel = select_q4_preferred(dg, k)
        else:
            raise ValueError(selector)
        by_day.append({
            'date': date,
            'n_available': len(dg),
            'n_taken': len(sel),
            'pnl': float(sel['pnl'].sum()),
            'avg': float(sel['pnl'].mean()) if len(sel) else 0.0,
            'wr': float((sel['pnl'] > 0).mean() * 100) if len(sel) else 0.0,
        })
    df = pd.DataFrame(by_day)
    # Equity-curve DD
    df = df.sort_values('date').reset_index(drop=True)
    df['cum'] = df['pnl'].cumsum()
    running_peak = -np.inf
    dd = 0.0
    for c in df['cum']:
        running_peak = max(running_peak, c)
        dd = min(dd, c - running_peak)
    total = df['pnl'].sum()
    worst_day = df['pnl'].min()
    worst_date = df.loc[df['pnl'].idxmin(), 'date']
    return {
        'name': name, 'k': k, 'selector': selector,
        'total_pnl': float(total),
        'avg_day_pnl': float(df['pnl'].mean()),
        'max_dd': float(dd),
        'worst_day_pnl': float(worst_day),
        'worst_day_date': worst_date,
        'days_traded': len(df),
        'by_day': df,
    }


def summarize(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            'strategy': r['name'], 'k': r['k'],
            'total_pnl': r['total_pnl'],
            'avg_day': r['avg_day_pnl'],
            'max_dd': r['max_dd'],
            'worst_day': r['worst_day_pnl'],
            'worst_date': r['worst_day_date'].date(),
            'days': r['days_traded'],
        })
    return pd.DataFrame(rows)


def main():
    csv = 'analysis_results/orb_features_20260418_1715.csv'
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'date'])

    rng = np.random.default_rng(42)

    # We'll run on Split C (largest test window: Oct'25→Apr'26, contains 11/20 blow-up)
    split_name, tr_s, tr_e, te_s, te_e = SPLITS[2]
    print(f"SPLIT: {split_name}")
    print(f"Test: {te_s} → {te_e}")

    test_kept = build_split_scored(df, tr_s, tr_e, te_s, te_e)
    print(f"Total test trades (after filter): {len(test_kept)}")
    print(f"Total days with trades: {test_kept['date'].nunique()}\n")

    # Distribution of concurrency
    per_day_counts = test_kept.groupby('date').size()
    print(f"Trades-per-day distribution:")
    print(f"  max: {per_day_counts.max()}")
    print(f"  p95: {per_day_counts.quantile(0.95):.0f}")
    print(f"  p75: {per_day_counts.quantile(0.75):.0f}")
    print(f"  median: {per_day_counts.median():.0f}")
    print(f"  days with >=5 trades: {(per_day_counts >= 5).sum()}/{len(per_day_counts)}")
    print(f"  days with >=10 trades: {(per_day_counts >= 10).sum()}/{len(per_day_counts)}")

    # Compare selection strategies at K = 3, 5, 10
    results = []
    for k in [3, 5, 10]:
        # top by composite
        results.append(run_selection(test_kept, k, 'top_composite',
                                      f'top-{k}-composite'))
        # Q4-preferred
        results.append(run_selection(test_kept, k, 'q4_preferred',
                                      f'top-{k}-q4pref'))
        # random (avg of 10 seeds)
        random_totals = []
        random_dds = []
        for seed in range(10):
            local_rng = np.random.default_rng(seed)
            r = run_selection(test_kept, k, 'random', f'top-{k}-rand', rng=local_rng)
            random_totals.append(r['total_pnl'])
            random_dds.append(r['max_dd'])
        # create a synthetic "avg random" result
        results.append({
            'name': f'random-{k} (avg of 10 seeds)', 'k': k, 'selector': 'random',
            'total_pnl': float(np.mean(random_totals)),
            'avg_day_pnl': float(np.mean(random_totals)) / test_kept['date'].nunique(),
            'max_dd': float(np.mean(random_dds)),
            'worst_day_pnl': 0.0,
            'worst_day_date': pd.Timestamp('1900-01-01'),
            'days_traded': 0, 'by_day': pd.DataFrame(),
        })
        # bottom by composite (control — should be worst)
        results.append(run_selection(test_kept, k, 'bottom_composite',
                                      f'bot-{k}-composite'))
        # take ALL (no cap)
        all_results = run_selection(test_kept, 9999, 'top_composite',
                                    f'all-kept (no cap)')
        all_results['name'] = f'all-kept (no cap)'
        all_results['k'] = -1
        if k == 10:  # only add once
            results.append(all_results)

    print(f"\n{'Strategy':<32} {'k':>3} {'total P&L':>12} {'avg/day':>10} "
          f"{'max DD':>12} {'worst day':>12} {'worst date':>12}")
    print('-' * 100)
    for r in results:
        print(f"{r['name']:<32} {r['k']:>3} ${r['total_pnl']:>+10,.0f} "
              f"${r['avg_day_pnl']:>+8,.0f} ${r['max_dd']:>+10,.0f} "
              f"${r['worst_day_pnl']:>+10,.0f} {r['worst_day_date'].date() if hasattr(r['worst_day_date'], 'date') else 'N/A'}")

    # --- Deep dive on 2025-11-20 ---
    print(f"\n{'='*100}")
    print(f"DEEP DIVE: 2025-11-20 catastrophic day")
    print(f"{'='*100}")
    blowup = test_kept[test_kept['date'] == pd.Timestamp('2025-11-20')].copy()
    blowup = blowup.sort_values('_composite', ascending=False).reset_index(drop=True)
    print(f"Total trades that day: {len(blowup)}")
    print(f"Total day P&L (all kept, unit sizing): ${blowup['pnl'].sum():+,.0f}")
    print(f"Win rate that day: {(blowup['pnl'] > 0).mean()*100:.1f}%")
    print(f"\nAll {len(blowup)} trades sorted by composite z (DESC):")
    print(f"{'rank':>4} {'symbol':>8} {'z-score':>8} {'Q':>3} "
          f"{'pnl':>10} {'win?':>5}")
    for i, (_, r) in enumerate(blowup.iterrows()):
        print(f"{i+1:>4} {r['symbol']:>8} {r['_composite']:>+7.3f} "
              f"{r['_quintile']:>3} ${r['pnl']:>+8,.0f} "
              f"{'W' if r['pnl']>0 else 'L':>5}")
    # If we took top-5 by composite on this day
    top5 = blowup.head(5)
    top3 = blowup.head(3)
    print(f"\nIf capped at top-5 by composite: P&L = ${top5['pnl'].sum():+,.0f} "
          f"(WR {(top5['pnl']>0).mean()*100:.0f}%)")
    print(f"If capped at top-3 by composite: P&L = ${top3['pnl'].sum():+,.0f} "
          f"(WR {(top3['pnl']>0).mean()*100:.0f}%)")
    # Q4-preferred top-5
    q4_pref = select_q4_preferred(blowup, 5)
    print(f"If capped at top-5 by Q4-preferred: P&L = ${q4_pref['pnl'].sum():+,.0f} "
          f"(WR {(q4_pref['pnl']>0).mean()*100:.0f}%)")

    # --- Rank correlation: composite vs P&L rank, within each day ---
    print(f"\n{'='*100}")
    print(f"RANK CORRELATION: composite vs P&L, within day (days with >=5 trades)")
    print(f"{'='*100}")
    per_day_spearman = []
    for date, dg in test_kept.groupby('date'):
        if len(dg) < 5:
            continue
        rho = dg['_composite'].corr(dg['pnl'], method='spearman')
        per_day_spearman.append({'date': date, 'n': len(dg), 'rho': rho})
    sp = pd.DataFrame(per_day_spearman)
    sp = sp.dropna()
    print(f"Days evaluated: {len(sp)}")
    print(f"Mean Spearman rho: {sp['rho'].mean():+.3f}")
    print(f"Median: {sp['rho'].median():+.3f}")
    print(f"% days with positive rho (composite ranks winners correctly): "
          f"{(sp['rho'] > 0).mean()*100:.1f}%")
    print(f"% days with rho > +0.2: {(sp['rho'] > 0.2).mean()*100:.1f}%")
    print(f"% days with rho < -0.2: {(sp['rho'] < -0.2).mean()*100:.1f}%")


if __name__ == '__main__':
    main()
