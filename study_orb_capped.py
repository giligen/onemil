#!/usr/bin/env python3
"""ORB walk-forward: filter + Q4-preferred ranking + max_concurrent cap + adaptive sizing.

Pipeline (per trading day):
  1. Filter: keep trades where composite_z >= +0.00 (TRAIN-fit params).
  2. Rank: sort by quintile-preferred order (Q4 > Q5 > Q3 > Q2 > Q1),
     tie-break by composite DESC.
  3. Cap: take top MAX_CONCURRENT trades only.
  4. Size: apply adaptive multiplier per quintile (TRAIN-fit).

Reports per-split + combined:
  - Total P&L
  - Max DD (daily-aggregated equity)
  - Worst single day
  - Days traded
  - Avg trades/day

Compares three variants:
  A. cap5_q4pref_adaptive  (the proposed shipping scheme)
  B. cap5_composite_adaptive (sanity check — rank by composite instead of Q4-pref)
  C. nocap_adaptive (upper-bound baseline — take all filtered trades)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from study_orb import SPLITS, OUT_DIR
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, ADAPTIVE_MULT_MIN, ADAPTIVE_MULT_MAX,
    fit_quintile_cutoffs, assign_quintile,
)

MAX_CONCURRENT = 5

# Q4-preferred ordering: Q4 first (best), then Q5, Q3, Q2, Q1
Q_PREF_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}


def fit_adaptive_mults(train_kept: pd.DataFrame) -> Dict[str, float]:
    """Per-quintile multiplier = clip(train_avg_Q / train_overall_avg, 0.25, 3.0)."""
    overall_avg = float(train_kept['pnl'].mean()) if len(train_kept) else 1.0
    mults: Dict[str, float] = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        qsub = train_kept[train_kept['_quintile'] == q]
        if len(qsub) == 0 or overall_avg <= 0:
            mults[q] = 1.0
            continue
        r = float(qsub['pnl'].mean()) / overall_avg
        mults[q] = max(ADAPTIVE_MULT_MIN, min(ADAPTIVE_MULT_MAX, r))
    return mults


def select_day_trades(day_df: pd.DataFrame, k: int, rank_by: str) -> pd.DataFrame:
    """Rank + cap the day's filtered trades.
    rank_by ∈ {'q4pref', 'composite'}"""
    if rank_by == 'q4pref':
        d = day_df.copy()
        d['_q_rank'] = d['_quintile'].map(Q_PREF_ORDER)
        d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])
    elif rank_by == 'composite':
        d = day_df.sort_values('_composite', ascending=False)
    else:
        raise ValueError(rank_by)
    return d.head(k)


def compute_daily_equity(sized_pnls: pd.DataFrame) -> Dict[str, float]:
    """Aggregate per-day P&L → equity curve → DD."""
    daily = sized_pnls.groupby('date').agg(
        daily_pnl=('_sized_pnl', 'sum'),
        n_trades=('pnl', 'count'),
    ).reset_index().sort_values('date').reset_index(drop=True)
    if len(daily) == 0:
        return {'total_pnl': 0.0, 'max_dd': 0.0, 'worst_day': 0.0,
                'worst_date': None, 'days_traded': 0, 'avg_trades_per_day': 0.0,
                'daily_df': daily}
    daily['cum'] = daily['daily_pnl'].cumsum()
    running_peak = -np.inf
    dd = 0.0
    for c in daily['cum']:
        running_peak = max(running_peak, c)
        dd = min(dd, c - running_peak)
    worst_idx = daily['daily_pnl'].idxmin()
    return {
        'total_pnl': float(daily['daily_pnl'].sum()),
        'max_dd': float(dd),
        'worst_day': float(daily.loc[worst_idx, 'daily_pnl']),
        'worst_date': daily.loc[worst_idx, 'date'],
        'days_traded': len(daily),
        'avg_trades_per_day': float(daily['n_trades'].mean()),
        'daily_df': daily,
    }


@dataclass
class SplitOutcome:
    split_name: str
    baseline: Dict[str, float] = field(default_factory=dict)  # no filter / no cap — for context
    variants: Dict[str, Dict[str, float]] = field(default_factory=dict)


def run_split(df: pd.DataFrame, tr_s: str, tr_e: str,
              te_s: str, te_e: str, k: int) -> SplitOutcome:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    train = df[(df['date'] >= tr_s) & (df['date'] <= tr_e)]
    test = df[(df['date'] >= te_s) & (df['date'] <= te_e)]

    # Fit z-score params on TRAIN
    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    train = df[(df['date'] >= tr_s) & (df['date'] <= tr_e)]
    test = df[(df['date'] >= te_s) & (df['date'] <= te_e)]

    # Apply filter to train + test
    test_kept = test[test['_composite'] >= FILTER_THRESHOLD].copy()
    train_kept = train[train['_composite'] >= FILTER_THRESHOLD].copy()

    # Fit quintile cutoffs on TRAIN kept
    cutoffs = fit_quintile_cutoffs(train_kept['_composite'])
    train_kept['_quintile'] = assign_quintile(train_kept['_composite'], cutoffs)
    test_kept['_quintile'] = assign_quintile(test_kept['_composite'], cutoffs)

    # Fit adaptive multipliers on TRAIN kept
    mults = fit_adaptive_mults(train_kept)

    # Baseline: raw test (no filter) — just for context
    baseline = compute_daily_equity(
        test.assign(_sized_pnl=test['pnl'])
    )

    outcome = SplitOutcome(
        split_name=f"{tr_s}/{tr_e} → {te_s}/{te_e}",
        baseline=baseline,
    )
    outcome.mults = mults  # stash for reporting

    # Variant A: filter + cap5 + q4-preferred + adaptive sizing
    a_sel = []
    for date, dg in test_kept.groupby('date'):
        sel = select_day_trades(dg, k, 'q4pref')
        a_sel.append(sel)
    a_df = pd.concat(a_sel) if a_sel else test_kept.iloc[:0]
    a_df['_sized_pnl'] = a_df.apply(lambda r: r['pnl'] * mults[r['_quintile']], axis=1)
    outcome.variants['cap5_q4pref_adaptive'] = compute_daily_equity(a_df)

    # Variant B: filter + cap5 + composite-rank + adaptive sizing
    b_sel = []
    for date, dg in test_kept.groupby('date'):
        sel = select_day_trades(dg, k, 'composite')
        b_sel.append(sel)
    b_df = pd.concat(b_sel) if b_sel else test_kept.iloc[:0]
    b_df['_sized_pnl'] = b_df.apply(lambda r: r['pnl'] * mults[r['_quintile']], axis=1)
    outcome.variants['cap5_composite_adaptive'] = compute_daily_equity(b_df)

    # Variant C: filter + NO CAP + adaptive sizing  (for comparison)
    c_df = test_kept.copy()
    c_df['_sized_pnl'] = c_df.apply(lambda r: r['pnl'] * mults[r['_quintile']], axis=1)
    outcome.variants['nocap_adaptive'] = compute_daily_equity(c_df)

    # Variant D: filter + cap5 + q4-preferred + UNIT sizing (no adaptive)
    d_sel = []
    for date, dg in test_kept.groupby('date'):
        sel = select_day_trades(dg, k, 'q4pref')
        d_sel.append(sel)
    d_df = pd.concat(d_sel) if d_sel else test_kept.iloc[:0]
    d_df['_sized_pnl'] = d_df['pnl']
    outcome.variants['cap5_q4pref_unit'] = compute_daily_equity(d_df)

    return outcome


def _fmt_date(d):
    return d.date().isoformat() if hasattr(d, 'date') else str(d)


def main():
    import glob
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    print(f"Loading: {csv}")
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'date'])
    print(f"{len(df):,} trades after dropna\n")

    outcomes: List[SplitOutcome] = []
    for split_name, tr_s, tr_e, te_s, te_e in SPLITS:
        print(f"{'='*100}")
        print(f"SPLIT: {split_name}")
        print(f"  Train {tr_s} → {tr_e}   Test {te_s} → {te_e}")
        print(f"{'='*100}")
        o = run_split(df, tr_s, tr_e, te_s, te_e, MAX_CONCURRENT)
        o.split_name = split_name
        outcomes.append(o)

        print(f"  Adaptive multipliers (TRAIN-fit): " +
              "  ".join(f"{q}={o.mults[q]:.2f}x" for q in ['Q1','Q2','Q3','Q4','Q5']))
        print(f"\n  {'variant':<28} {'P&L':>12} {'max DD':>12} "
              f"{'worst day':>12} {'on date':>12} {'days':>6} {'tr/day':>7}")
        print('  ' + '-' * 95)
        for vname, v in o.variants.items():
            print(f"  {vname:<28} ${v['total_pnl']:>+10,.0f} ${v['max_dd']:>+10,.0f} "
                  f"${v['worst_day']:>+10,.0f} {_fmt_date(v['worst_date']):>12} "
                  f"{v['days_traded']:>6} {v['avg_trades_per_day']:>6.1f}")

    # Cross-split summary
    print(f"\n{'='*100}")
    print(f"CROSS-SPLIT SUMMARY — walk-forward (test periods only), MAX_CONCURRENT={MAX_CONCURRENT}")
    print(f"{'='*100}")
    variants = list(outcomes[0].variants.keys())
    headers = ['variant'] + [o.split_name.split(':')[0] for o in outcomes] + ['Sum P&L', 'Min P&L', 'Worst DD']
    print('  ' + ' | '.join(f"{h:>15}" for h in headers))
    print('  ' + '-' * (17 * len(headers)))
    for vname in variants:
        pnls = [o.variants[vname]['total_pnl'] for o in outcomes]
        dds = [o.variants[vname]['max_dd'] for o in outcomes]
        total = sum(pnls)
        min_p = min(pnls)
        worst_dd = min(dds)
        cells = [vname] + [f"${p:+,.0f}" for p in pnls] + [
            f"${total:+,.0f}", f"${min_p:+,.0f}", f"${worst_dd:+,.0f}"]
        print('  ' + ' | '.join(f"{c:>15}" for c in cells))

    # Risk-adjusted comparison
    print(f"\n  {'variant':<28} {'Sum P&L':>14} {'Worst DD':>14} {'P&L/DD':>10}")
    for vname in variants:
        pnls = [o.variants[vname]['total_pnl'] for o in outcomes]
        dds = [o.variants[vname]['max_dd'] for o in outcomes]
        total = sum(pnls)
        worst_dd = min(dds)
        ratio = total / abs(worst_dd) if worst_dd < 0 else float('inf')
        print(f"  {vname:<28} ${total:>+12,.0f} ${worst_dd:>+12,.0f} {ratio:>9.2f}x")

    # Write markdown report
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    md_path = f"{OUT_DIR}/orb_capped_{ts}.md"
    with open(md_path, 'w') as f:
        f.write(f"# ORB cap={MAX_CONCURRENT} + Q4-preferred + adaptive sizing — walk-forward\n\n")
        f.write(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n\n")
        f.write(f"**Input**: `{csv}` ({len(df):,} trades)\n\n")
        f.write("## Pipeline (per trading day)\n\n")
        f.write(f"1. Filter: composite_z >= +0.00 (z-score params fit on TRAIN only)\n")
        f.write(f"2. Rank: Q4-preferred (Q4 > Q5 > Q3 > Q2 > Q1), tie-break composite DESC\n")
        f.write(f"3. Cap: top **{MAX_CONCURRENT}** trades per day\n")
        f.write(f"4. Size: adaptive multiplier per quintile "
                f"(mult = clip(train_avg_Q / train_overall_avg, {ADAPTIVE_MULT_MIN}, {ADAPTIVE_MULT_MAX}))\n\n")
        f.write("## Per-split results\n\n")
        for o in outcomes:
            f.write(f"### {o.split_name}\n\n")
            f.write(f"Adaptive multipliers: " +
                    "  ".join(f"{q}={o.mults[q]:.2f}x" for q in ['Q1','Q2','Q3','Q4','Q5']) + "\n\n")
            f.write("| variant | P&L | max DD | worst day | on date | days | tr/day |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for vname, v in o.variants.items():
                f.write(f"| `{vname}` | ${v['total_pnl']:+,.0f} | ${v['max_dd']:+,.0f} | "
                        f"${v['worst_day']:+,.0f} | {_fmt_date(v['worst_date'])} | "
                        f"{v['days_traded']} | {v['avg_trades_per_day']:.1f} |\n")
            f.write("\n")
        f.write("## Cross-split summary\n\n")
        f.write("| variant | A P&L | B P&L | C P&L | Sum | Min split | Worst DD | P&L/DD |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for vname in variants:
            pnls = [o.variants[vname]['total_pnl'] for o in outcomes]
            dds = [o.variants[vname]['max_dd'] for o in outcomes]
            total = sum(pnls); min_p = min(pnls); worst_dd = min(dds)
            ratio = total / abs(worst_dd) if worst_dd < 0 else float('inf')
            cells = [f"${p:+,.0f}" for p in pnls] + [
                f"${total:+,.0f}", f"${min_p:+,.0f}",
                f"${worst_dd:+,.0f}", f"{ratio:.2f}x"]
            f.write(f"| `{vname}` | " + " | ".join(cells) + " |\n")
    print(f"\nReport: {md_path}")


if __name__ == '__main__':
    main()
