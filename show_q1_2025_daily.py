"""Day-by-day breakdown of Q1 2025 under the proposed ORB pipeline.

⚠️  NOT PRODUCTION-PARITY — uses fixed +2R target / -1R stop from
orb_features CSV. Shipped exit is static_lock_1R. Use
show_q1_2025_static_lock.py for production-parity results.



Shows for every trading day:
  - How many ORB signals fired (filtered)
  - Which 3 we picked (Q4-preferred)
  - Per-trade P&L
  - Daily total
  - Running equity
  - Running DD from peak

Pipeline: N=3 max concurrent, risk=$2K/trade, composite filter, Q4-pref rank,
adaptive sizing. Account assumed $100K buying power.

Note: Q1 2025 is IN TRAIN for all 3 walk-forward splits, so we fit z-params
using Split A train (2025-01-01 to 2025-06-30). This is in-sample — purpose
is to SHOW the pipeline mechanics, not claim this is a walk-forward test.
"""
from __future__ import annotations

import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, ADAPTIVE_MULT_MIN, ADAPTIVE_MULT_MAX,
    fit_quintile_cutoffs, assign_quintile,
)

ACCOUNT = 100_000
N = 3
RISK = 2000
OLD_POS = 50_000
MIN_STOP_PCT = 1.0
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}


def main():
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'pnl_pct', 'date', 'range_size_pct',
                                     'entry_price'])
    df['date'] = pd.to_datetime(df['date'])

    # Risk-parity sizing (same as study_orb_100k_account.py)
    per_pos_cap = ACCOUNT / N
    stop_pct = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    uncap = RISK / (stop_pct / 100.0)
    df['_rp_position'] = uncap.clip(upper=per_pos_cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS

    # Fit z-params on Split A train (H1 2025)
    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-30')]
    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    # Re-slice train now that _composite is on df
    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-30')]
    train_kept = train[train['_composite'] >= FILTER_THRESHOLD].copy()
    cutoffs = fit_quintile_cutoffs(train_kept['_composite'])
    train_kept['_quintile'] = assign_quintile(train_kept['_composite'], cutoffs)

    # Fit adaptive mults on train
    avg = float(train_kept['_rp_pnl'].mean())
    mults = {}
    for q in ['Q1','Q2','Q3','Q4','Q5']:
        sub = train_kept[train_kept['_quintile'] == q]
        mults[q] = max(ADAPTIVE_MULT_MIN, min(ADAPTIVE_MULT_MAX,
                                               float(sub['_rp_pnl'].mean()) / avg))

    # Apply to Q1 2025
    q1 = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-03-31')]
    q1_filtered = q1[q1['_composite'] >= FILTER_THRESHOLD].copy()
    q1_filtered['_quintile'] = assign_quintile(q1_filtered['_composite'], cutoffs)

    # Daily: select top N by Q4-preferred
    print(f"{'='*110}")
    print(f"Q1 2025 DAY-BY-DAY (N={N}, risk=${RISK:,.0f}, cap=${per_pos_cap:,.0f})")
    print(f"{'='*110}")
    print(f"Pipeline: filter (composite>=0) → Q4-pref rank → top-{N} → adaptive mult")
    print(f"Adaptive mults: " +
          "  ".join(f"{q}={mults[q]:.2f}x" for q in ['Q1','Q2','Q3','Q4','Q5']))
    print(f"\n{'date':<12} {'sig':>4} {'picked':>7} {'trades selected':<50} "
          f"{'day P&L':>11} {'equity':>11} {'DD':>10}")
    print('-' * 110)

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    peak_date = None
    trough_date = None
    worst_dd_peak = None
    worst_dd_trough = None
    daily_rows = []

    unique_dates = sorted(q1_filtered['date'].unique())
    # Also count ALL Q1 dates (even days with no signals)
    all_q1_dates = pd.date_range('2025-01-02', '2025-03-31', freq='B')  # business days

    for day in all_q1_dates:
        dg = q1_filtered[q1_filtered['date'] == day]
        n_sig = len(dg)
        if n_sig == 0:
            # No signals — no trading
            continue

        # Rank Q4-preferred
        d = dg.copy()
        d['_q_rank'] = d['_quintile'].map(Q_ORDER)
        d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])
        picked = d.head(N)
        picked = picked.copy()
        picked['_sized_pnl'] = picked.apply(
            lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)

        # Trade summary string
        trade_str_parts = []
        for _, r in picked.iterrows():
            arrow = '✓' if r['_sized_pnl'] > 0 else '✗'
            trade_str_parts.append(
                f"{r['symbol']}({r['_quintile']}) {arrow}${r['_sized_pnl']:+,.0f}"
            )
        trade_str = ', '.join(trade_str_parts)
        if len(trade_str) > 48:
            trade_str = trade_str[:45] + '…'

        day_pnl = float(picked['_sized_pnl'].sum())
        equity += day_pnl
        if equity > peak:
            peak = equity
            peak_date = day
        dd_now = equity - peak
        if dd_now < max_dd:
            max_dd = dd_now
            worst_dd_peak = peak_date
            worst_dd_trough = day

        daily_rows.append({
            'date': day, 'n_signals': n_sig,
            'n_picked': len(picked),
            'day_pnl': day_pnl, 'equity': equity, 'dd_now': dd_now,
            'trades': trade_str,
        })

        print(f"{str(day.date()):<12} {n_sig:>4} {len(picked):>7} "
              f"{trade_str:<50} "
              f"${day_pnl:>+9,.0f} ${equity:>+9,.0f} ${dd_now:>+8,.0f}")

    print(f"\n{'='*110}")
    print(f"SUMMARY for Q1 2025:")
    print(f"  Trading days with signals: {len(daily_rows)}")
    print(f"  Final equity: ${equity:+,.0f}")
    print(f"  Peak equity:  ${peak:+,.0f}  (on {peak_date.date() if peak_date else '-'})")
    print(f"  Max DD:       ${max_dd:+,.0f}  "
          f"(peak {worst_dd_peak.date() if worst_dd_peak else '-'} → "
          f"trough {worst_dd_trough.date() if worst_dd_trough else '-'})")

    # Top 5 worst days
    df_daily = pd.DataFrame(daily_rows)
    print(f"\nWORST 5 DAYS:")
    for _, r in df_daily.nsmallest(5, 'day_pnl').iterrows():
        print(f"  {r['date'].date()}  pnl=${r['day_pnl']:>+8,.0f}  "
              f"equity_after=${r['equity']:>+8,.0f}  "
              f"picks: {r['trades']}")
    print(f"\nBEST 5 DAYS:")
    for _, r in df_daily.nlargest(5, 'day_pnl').iterrows():
        print(f"  {r['date'].date()}  pnl=${r['day_pnl']:>+8,.0f}  "
              f"equity_after=${r['equity']:>+8,.0f}  "
              f"picks: {r['trades']}")

    # Save CSV
    df_daily.to_csv('analysis_results/q1_2025_daily.csv', index=False)
    print(f"\nSaved to analysis_results/q1_2025_daily.csv")


if __name__ == '__main__':
    main()
