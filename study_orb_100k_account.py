#!/usr/bin/env python3
"""ORB sizing under REAL account constraints: $100K total buying power,
max N concurrent positions, per-position cap = $100K/N.

Configuration sweep:
  - max_concurrent ∈ {3, 4, 5}  (per-position caps = $33K, $25K, $20K)
  - risk_per_trade ∈ {$250, $500, $1K, $2K, $3K}

Pipeline (walk-forward, per split):
  1. Filter: composite_z ≥ +0.00 (TRAIN-fit)
  2. Rank: Q4-preferred (Q4 > Q5 > Q3 > Q2 > Q1), tie-break composite DESC
  3. Cap to top-N per day (N = max_concurrent)
  4. Size: position = min(risk/stop_pct, per_position_cap)
  5. Apply adaptive quintile multipliers fit from TRAIN

Reports per-config: sum P&L, worst DD, worst day, worst per-trade.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from study_orb import SPLITS, OUT_DIR
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, ADAPTIVE_MULT_MIN, ADAPTIVE_MULT_MAX,
    fit_quintile_cutoffs, assign_quintile,
)

ACCOUNT_SIZE = 100_000
MIN_STOP_PCT = 1.0  # floor (prevent infinite sizing)
OLD_POSITION_USD = 50_000.0  # simulator's original position size
Q_PREF_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}


def apply_sizing(df: pd.DataFrame, risk_per_trade: float,
                 per_pos_cap: float) -> pd.DataFrame:
    """Risk-parity sizing within per-position cap."""
    df = df.copy()
    stop_pct = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    # Uncapped position based on risk
    uncapped = risk_per_trade / (stop_pct / 100.0)
    df['_rp_position'] = uncapped.clip(upper=per_pos_cap)
    df['_rp_scale'] = df['_rp_position'] / OLD_POSITION_USD
    df['_rp_pnl'] = df['pnl'] * df['_rp_scale']
    return df


def fit_adaptive_mults(train_kept: pd.DataFrame, pnl_col: str) -> Dict[str, float]:
    overall_avg = float(train_kept[pnl_col].mean()) if len(train_kept) else 1.0
    mults: Dict[str, float] = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        qsub = train_kept[train_kept['_quintile'] == q]
        if len(qsub) == 0 or overall_avg <= 0:
            mults[q] = 1.0
            continue
        r = float(qsub[pnl_col].mean()) / overall_avg
        mults[q] = max(ADAPTIVE_MULT_MIN, min(ADAPTIVE_MULT_MAX, r))
    return mults


def select_day_trades(day_df: pd.DataFrame, k: int) -> pd.DataFrame:
    d = day_df.copy()
    d['_q_rank'] = d['_quintile'].map(Q_PREF_ORDER)
    d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])
    return d.head(k)


def compute_daily(daily: pd.DataFrame, pnl_col: str) -> Dict:
    if len(daily) == 0:
        return {'total_pnl': 0, 'max_dd': 0, 'worst_day': 0, 'worst_date': None,
                'days_traded': 0, 'avg_deployed_per_active_day': 0}
    d = daily.sort_values('date').reset_index(drop=True)
    d['cum'] = d[pnl_col].cumsum()
    running_peak = -np.inf
    dd = 0.0
    for c in d['cum']:
        running_peak = max(running_peak, c)
        dd = min(dd, c - running_peak)
    worst_idx = d[pnl_col].idxmin()
    return {
        'total_pnl': float(d[pnl_col].sum()),
        'max_dd': float(dd),
        'worst_day': float(d.loc[worst_idx, pnl_col]),
        'worst_date': d.loc[worst_idx, 'date'],
        'days_traded': len(d),
    }


def run_split(df: pd.DataFrame, tr_s, tr_e, te_s, te_e,
              max_concurrent: int, risk_per_trade: float) -> Dict:
    per_pos_cap = ACCOUNT_SIZE / max_concurrent

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = apply_sizing(df, risk_per_trade, per_pos_cap)

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
    mults = fit_adaptive_mults(train_kept, '_rp_pnl')

    sel = []
    for date, dg in test_kept.groupby('date'):
        sel.append(select_day_trades(dg, max_concurrent))
    sel_df = pd.concat(sel) if sel else test_kept.iloc[:0]
    sel_df['_sized_pnl'] = sel_df.apply(
        lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)

    daily = sel_df.groupby('date').agg(
        daily_pnl=('_sized_pnl', 'sum'),
        n_trades=('_rp_pnl', 'count'),
        deployed=('_rp_position', 'sum'),
    ).reset_index()

    eq = compute_daily(daily, 'daily_pnl')
    return {
        'equity': eq,
        'worst_per_trade': float(sel_df['_sized_pnl'].min()) if len(sel_df) else 0,
        'avg_trades_per_day': float(daily['n_trades'].mean()) if len(daily) else 0,
        'avg_deployed_per_day': float(daily['deployed'].mean()) if len(daily) else 0,
        'max_deployed_in_a_day': float(daily['deployed'].max()) if len(daily) else 0,
    }


def run_config(df: pd.DataFrame, max_concurrent: int,
               risk_per_trade: float) -> Dict:
    results = []
    for split_name, tr_s, tr_e, te_s, te_e in SPLITS:
        r = run_split(df, tr_s, tr_e, te_s, te_e, max_concurrent, risk_per_trade)
        r['split_name'] = split_name
        results.append(r)
    pnls = [r['equity']['total_pnl'] for r in results]
    dds = [r['equity']['max_dd'] for r in results]
    worst_days = [r['equity']['worst_day'] for r in results]
    worst_trades = [r['worst_per_trade'] for r in results]
    avg_dep = [r['avg_deployed_per_day'] for r in results]
    max_dep = [r['max_deployed_in_a_day'] for r in results]
    return {
        'max_concurrent': max_concurrent,
        'per_pos_cap': ACCOUNT_SIZE / max_concurrent,
        'risk_per_trade': risk_per_trade,
        'sum_pnl': sum(pnls),
        'min_split_pnl': min(pnls),
        'worst_dd': min(dds),
        'worst_day': min(worst_days),
        'worst_per_trade': min(worst_trades),
        'pnl_dd_ratio': sum(pnls) / abs(min(dds)) if min(dds) < 0 else float('inf'),
        'avg_deployed': np.mean(avg_dep),
        'max_deployed': max(max_dep),
        'splits': results,
    }


def main():
    import glob
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    print(f"Loading: {csv}")
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'pnl_pct', 'date', 'range_size_pct'])
    print(f"{len(df):,} trades after dropna\n")

    print(f"{'='*110}")
    print(f"ACCOUNT=$\{ACCOUNT_SIZE:,} (buying power), MAX concurrent tested: 3/4/5")
    print(f"Per-position cap = $100K / N   →  N=3: $33K | N=4: $25K | N=5: $20K")
    print(f"{'='*110}")

    sweep = []
    for N in [3, 4, 5]:
        cap = ACCOUNT_SIZE / N
        print(f"\n--- MAX_CONCURRENT={N}   PER_POS_CAP=${cap:,.0f} ---")
        print(f"  {'Risk/tr':>8} {'Sum P&L':>13} {'Min split':>13} "
              f"{'Worst DD':>12} {'Worst day':>12} {'Worst tr':>10} "
              f"{'P&L/|DD|':>9} {'avg dep':>10} {'max dep':>10}")
        for risk in [250, 500, 1000, 2000, 3000]:
            r = run_config(df, N, risk)
            sweep.append(r)
            print(f"  ${risk:>6,.0f} ${r['sum_pnl']:>+11,.0f} ${r['min_split_pnl']:>+11,.0f} "
                  f"${r['worst_dd']:>+10,.0f} ${r['worst_day']:>+10,.0f} "
                  f"${r['worst_per_trade']:>+8,.0f} {r['pnl_dd_ratio']:>8.2f}x "
                  f"${r['avg_deployed']:>+8,.0f} ${r['max_deployed']:>+8,.0f}")

    # Best configs by P&L/DD ratio (among those with min_split > 0)
    print(f"\n{'='*110}")
    print("TOP 5 CONFIGS BY RISK-ADJUSTED (P&L / |DD|) — all must have min split > 0")
    print(f"{'='*110}")
    eligible = [s for s in sweep if s['min_split_pnl'] > 0]
    eligible.sort(key=lambda c: c['pnl_dd_ratio'], reverse=True)
    print(f"  {'N':>3} {'risk/tr':>8} {'sum P&L':>13} {'worst DD':>12} "
          f"{'worst day':>12} {'worst tr':>10} {'P&L/DD':>9}")
    for c in eligible[:5]:
        print(f"  {c['max_concurrent']:>3} ${c['risk_per_trade']:>6,.0f} "
              f"${c['sum_pnl']:>+11,.0f} ${c['worst_dd']:>+10,.0f} "
              f"${c['worst_day']:>+10,.0f} ${c['worst_per_trade']:>+8,.0f} "
              f"{c['pnl_dd_ratio']:>8.2f}x")

    print(f"\n{'='*110}")
    print("TOP 5 CONFIGS BY ABSOLUTE P&L — all must have min split > 0")
    print(f"{'='*110}")
    eligible.sort(key=lambda c: c['sum_pnl'], reverse=True)
    for c in eligible[:5]:
        print(f"  {c['max_concurrent']:>3} ${c['risk_per_trade']:>6,.0f} "
              f"${c['sum_pnl']:>+11,.0f} ${c['worst_dd']:>+10,.0f} "
              f"${c['worst_day']:>+10,.0f} ${c['worst_per_trade']:>+8,.0f} "
              f"{c['pnl_dd_ratio']:>8.2f}x")

    # Write markdown
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    md_path = f"{OUT_DIR}/orb_100k_account_{ts}.md"
    with open(md_path, 'w') as f:
        f.write(f"# ORB Sizing — $100K account constraint\n\n")
        f.write(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n\n")
        f.write(f"## Constraint\n\n")
        f.write(f"- Account buying power: **$100,000** (with leverage)\n")
        f.write(f"- Max concurrent positions: N (tested 3, 4, 5)\n")
        f.write(f"- Per-position cap: $100K / N (33K, 25K, 20K respectively)\n")
        f.write(f"- Risk-parity sizing: shares = risk / stop_distance, "
                f"capped at per-position cap\n")
        f.write(f"- Pipeline: composite_z filter → Q4-pref rank → cap-N → adaptive quintile mult\n\n")

        f.write(f"## Full sweep\n\n")
        f.write(f"| N | risk/tr | per-pos cap | Sum P&L | Min split | Worst DD | "
                f"Worst day | Worst trade | P&L/|DD| | avg deployed | max deployed |\n")
        f.write(f"|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for c in sweep:
            f.write(f"| {c['max_concurrent']} | ${c['risk_per_trade']:,.0f} | "
                    f"${c['per_pos_cap']:,.0f} | ${c['sum_pnl']:+,.0f} | "
                    f"${c['min_split_pnl']:+,.0f} | ${c['worst_dd']:+,.0f} | "
                    f"${c['worst_day']:+,.0f} | ${c['worst_per_trade']:+,.0f} | "
                    f"{c['pnl_dd_ratio']:.2f}x | ${c['avg_deployed']:,.0f} | "
                    f"${c['max_deployed']:,.0f} |\n")

        eligible.sort(key=lambda c: c['sum_pnl'], reverse=True)
        f.write(f"\n## Recommended configs (min split > 0, top by P&L)\n\n")
        f.write(f"| rank | N | risk/tr | Sum P&L | Worst DD | Worst day | Worst tr | P&L/|DD| |\n")
        f.write(f"|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for i, c in enumerate(eligible[:5], 1):
            f.write(f"| {i} | {c['max_concurrent']} | ${c['risk_per_trade']:,.0f} | "
                    f"${c['sum_pnl']:+,.0f} | ${c['worst_dd']:+,.0f} | "
                    f"${c['worst_day']:+,.0f} | ${c['worst_per_trade']:+,.0f} | "
                    f"{c['pnl_dd_ratio']:.2f}x |\n")
    print(f"\nReport: {md_path}")


if __name__ == '__main__':
    main()
