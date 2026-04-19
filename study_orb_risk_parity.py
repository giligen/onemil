#!/usr/bin/env python3
"""Risk-parity sizing for ORB trades — cap per-trade loss at fixed dollar risk.

Problem with fixed-dollar position ($50K flat):
  - Stop = range_low
  - Wide-range stock (10% range) → stop 10% away → $5K risk/trade on $50K position
  - Narrow-range stock (2% range) → stop 2% away → $1K risk/trade
  - On catastrophic days, many wide-range stocks all hit stop together = $100K+ losses

Fix: compute shares so that (entry - stop) * shares = RISK_PER_TRADE (fixed).
  - Wide-range stock → fewer shares → risk capped
  - Narrow-range stock → more shares → same dollar risk, bigger % reward if works

Practical caps:
  - MIN_STOP_PCT: minimum effective stop distance (don't size up to infinity on narrow ranges)
  - MAX_POSITION_USD: never take more than this in one name

Re-run full cap5 + Q4-preferred + adaptive pipeline on risk-parity-sized P&L
to see if the shipping numbers still hold with sane per-trade risk.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
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

# Risk-parity parameters
RISK_PER_TRADE_USD = 500.0      # fixed dollar risk per trade
MIN_STOP_PCT = 1.0               # floor on stop distance (%); protects against infinite sizing
MAX_POSITION_USD = 50_000.0      # cap on notional exposure per trade
OLD_POSITION_USD = 50_000.0      # what the original simulation used (for reference)

MAX_CONCURRENT = 5
Q_PREF_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}


def recompute_risk_parity_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """Re-scale each trade's P&L assuming risk-parity sizing.

    Assumptions:
      - stop_distance_pct ≈ range_size_pct (entry at range_high, stop at range_low)
      - floored at MIN_STOP_PCT
      - position capped at MAX_POSITION_USD
    """
    df = df.copy()
    stop_pct = df['range_size_pct'].clip(lower=MIN_STOP_PCT)  # in percent
    # Uncapped position = RISK / stop_distance
    uncapped_pos = RISK_PER_TRADE_USD / (stop_pct / 100.0)
    df['_rp_position'] = uncapped_pos.clip(upper=MAX_POSITION_USD)
    # Scale factor from original $50K fixed to risk-parity position
    df['_rp_scale'] = df['_rp_position'] / OLD_POSITION_USD
    df['_rp_pnl'] = df['pnl'] * df['_rp_scale']
    return df


def summarize_per_trade(df: pd.DataFrame, pnl_col: str) -> Dict[str, float]:
    if len(df) == 0:
        return {'n': 0, 'avg': 0, 'med': 0, 'worst': 0, 'best': 0, 'total': 0,
                'loss_std': 0, 'p95_loss': 0}
    p = df[pnl_col]
    losses = p[p < 0]
    return {
        'n': len(df),
        'avg': float(p.mean()),
        'med': float(p.median()),
        'worst': float(p.min()),
        'best': float(p.max()),
        'total': float(p.sum()),
        'loss_std': float(losses.std()) if len(losses) > 0 else 0,
        'p95_loss': float(losses.quantile(0.05)) if len(losses) > 0 else 0,  # 5th pct of losses = big one
    }


def compute_daily_equity(daily: pd.DataFrame, pnl_col: str) -> Dict[str, float]:
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


def run_split_rp(df: pd.DataFrame, tr_s, tr_e, te_s, te_e, k: int,
                 pnl_col: str = '_rp_pnl') -> Dict:
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
    mults = fit_adaptive_mults(train_kept, pnl_col)

    # cap5 + q4-preferred
    sel = []
    for date, dg in test_kept.groupby('date'):
        sel.append(select_day_trades(dg, k))
    sel_df = pd.concat(sel) if sel else test_kept.iloc[:0]
    sel_df['_sized_pnl'] = sel_df.apply(
        lambda r: r[pnl_col] * mults[r['_quintile']], axis=1)

    # Aggregate daily
    daily = sel_df.groupby('date').agg(
        daily_pnl=('_sized_pnl', 'sum'),
        n_trades=(pnl_col, 'count'),
    ).reset_index()
    return {
        'mults': mults,
        'equity': compute_daily_equity(daily, 'daily_pnl'),
        'per_trade': summarize_per_trade(sel_df, '_sized_pnl'),
        'raw_per_trade': summarize_per_trade(sel_df, pnl_col),
    }


def run_full_pipeline(df: pd.DataFrame, risk_per_trade: float,
                       min_stop_pct: float = 1.0,
                       max_position: float = 50_000.0) -> Dict:
    """Full pipeline for given risk_per_trade. Returns cross-split stats."""
    # Recompute pnl with these risk-parity params
    df = df.copy()
    stop_pct = df['range_size_pct'].clip(lower=min_stop_pct)
    uncapped = risk_per_trade / (stop_pct / 100.0)
    df['_rp_position'] = uncapped.clip(upper=max_position)
    df['_rp_scale'] = df['_rp_position'] / OLD_POSITION_USD
    df['_rp_pnl'] = df['pnl'] * df['_rp_scale']

    per_trade = summarize_per_trade(df, '_rp_pnl')
    results = []
    for split_name, tr_s, tr_e, te_s, te_e in SPLITS:
        r = run_split_rp(df, tr_s, tr_e, te_s, te_e, MAX_CONCURRENT,
                         pnl_col='_rp_pnl')
        results.append(r)

    pnls = [r['equity']['total_pnl'] for r in results]
    dds = [r['equity']['max_dd'] for r in results]
    worst_days = [r['equity']['worst_day'] for r in results]
    per_trade_worsts = [r['per_trade']['worst'] for r in results]

    return {
        'risk_per_trade': risk_per_trade,
        'sum_pnl': sum(pnls),
        'min_split_pnl': min(pnls),
        'worst_dd': min(dds),
        'worst_day': min(worst_days),
        'worst_per_trade': min(per_trade_worsts),
        'pnl_dd_ratio': sum(pnls) / abs(min(dds)) if min(dds) < 0 else float('inf'),
        'per_trade_all': per_trade,
        'per_split_pnl': pnls,
        'per_split_dd': dds,
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

    # Step 0: Risk sweep
    print(f"{'='*100}")
    print(f"RISK-PER-TRADE SWEEP — cap5 + Q4-pref + adaptive, "
          f"range_floor={MIN_STOP_PCT}%, pos_cap=${MAX_POSITION_USD:,.0f}")
    print(f"{'='*100}")
    risk_levels = [250, 500, 1000, 2000, 3000, 5000]
    print(f"  {'Risk/tr':>8} {'Sum P&L':>14} {'Min split':>13} {'Worst DD':>13} "
          f"{'Worst day':>12} {'Worst tr':>10} {'P&L/|DD|':>10}")
    sweep_results = []
    for risk in risk_levels:
        r = run_full_pipeline(df, risk)
        sweep_results.append(r)
        print(f"  ${risk:>6,.0f} ${r['sum_pnl']:>+12,.0f} ${r['min_split_pnl']:>+11,.0f} "
              f"${r['worst_dd']:>+11,.0f} ${r['worst_day']:>+10,.0f} "
              f"${r['worst_per_trade']:>+8,.0f} {r['pnl_dd_ratio']:>9.2f}x")

    # Detailed run at RISK=500 (default from config)
    df = recompute_risk_parity_pnl(df)

    # Diagnostic: range_size_pct distribution
    print("=" * 90)
    print("Range-size distribution (used as stop distance proxy):")
    print("=" * 90)
    rs = df['range_size_pct']
    print(f"  min={rs.min():.2f}%  p05={rs.quantile(0.05):.2f}%  "
          f"p50={rs.median():.2f}%  p95={rs.quantile(0.95):.2f}%  max={rs.max():.2f}%")
    print(f"  trades with range < {MIN_STOP_PCT}% (floored): "
          f"{(rs < MIN_STOP_PCT).sum()} ({100*(rs < MIN_STOP_PCT).mean():.1f}%)")
    print(f"  position sizing under risk-parity:")
    print(f"    p05 position: ${df['_rp_position'].quantile(0.05):>10,.0f}")
    print(f"    p50 position: ${df['_rp_position'].median():>10,.0f}")
    print(f"    p95 position: ${df['_rp_position'].quantile(0.95):>10,.0f}")
    print(f"    position capped at ${MAX_POSITION_USD:,.0f}: "
          f"{(df['_rp_position'] >= MAX_POSITION_USD).sum()} "
          f"({100*(df['_rp_position'] >= MAX_POSITION_USD).mean():.1f}%)")

    # Per-trade loss comparison: old sizing vs risk-parity
    print(f"\n{'='*90}")
    print(f"PER-TRADE DISTRIBUTION: fixed $50K position vs risk-parity ($500 risk/trade)")
    print(f"{'='*90}")
    old_stats = summarize_per_trade(df, 'pnl')
    new_stats = summarize_per_trade(df, '_rp_pnl')
    print(f"  {'metric':<18} {'flat $50K':>14} {'risk-parity':>14}")
    for metric in ['n', 'avg', 'med', 'best', 'worst', 'p95_loss', 'total']:
        old = old_stats[metric]
        new = new_stats[metric]
        if metric == 'n':
            print(f"  {metric:<18} {old:>14,.0f} {new:>14,.0f}")
        else:
            print(f"  {metric:<18} ${old:>+13,.0f} ${new:>+13,.0f}")

    print(f"\n  Worst losing trades (flat $50K):")
    worst_flat = df.nsmallest(5, 'pnl')[['symbol', 'date', 'range_size_pct',
                                         'pnl', '_rp_pnl', '_rp_position']]
    for _, r in worst_flat.iterrows():
        print(f"    {r['symbol']:>8} {str(r['date'])[:10]}  range={r['range_size_pct']:>5.2f}%  "
              f"flat_pnl=${r['pnl']:>+9,.0f}  rp_pnl=${r['_rp_pnl']:>+9,.0f}  "
              f"rp_pos=${r['_rp_position']:>7,.0f}")

    # Step 2: run walk-forward with risk-parity P&L
    print(f"\n{'='*90}")
    print(f"WALK-FORWARD: cap{MAX_CONCURRENT} + Q4-pref + adaptive  (risk-parity sizing)")
    print(f"{'='*90}")
    results = []
    for split_name, tr_s, tr_e, te_s, te_e in SPLITS:
        print(f"\nSplit: {split_name}")
        r = run_split_rp(df, tr_s, tr_e, te_s, te_e, MAX_CONCURRENT, pnl_col='_rp_pnl')
        r['split_name'] = split_name
        results.append(r)
        print(f"  Adaptive mults: " +
              "  ".join(f"{q}={r['mults'][q]:.2f}x" for q in ['Q1','Q2','Q3','Q4','Q5']))
        eq = r['equity']; pt = r['per_trade']
        print(f"  Total P&L: ${eq['total_pnl']:+,.0f}   "
              f"Max DD: ${eq['max_dd']:+,.0f}   "
              f"Worst day: ${eq['worst_day']:+,.0f} ({eq['worst_date'].date() if hasattr(eq['worst_date'], 'date') else eq['worst_date']})")
        print(f"  Per-trade: n={pt['n']}  avg=${pt['avg']:+,.0f}  "
              f"worst=${pt['worst']:+,.0f}  best=${pt['best']:+,.0f}  "
              f"p95-loss=${pt['p95_loss']:+,.0f}")

    # Cross-split summary
    print(f"\n{'='*90}")
    print(f"CROSS-SPLIT SUMMARY (risk-parity sizing)")
    print(f"{'='*90}")
    print(f"  {'Split':<40} {'P&L':>14} {'Max DD':>12} {'Worst day':>12} {'P&L/DD':>8}")
    pnls = []; dds = []
    for r in results:
        eq = r['equity']
        pnls.append(eq['total_pnl']); dds.append(eq['max_dd'])
        ratio = eq['total_pnl'] / abs(eq['max_dd']) if eq['max_dd'] < 0 else float('inf')
        print(f"  {r['split_name']:<40} ${eq['total_pnl']:>+12,.0f} "
              f"${eq['max_dd']:>+10,.0f} ${eq['worst_day']:>+10,.0f} {ratio:>6.2f}x")
    total = sum(pnls); worst_dd = min(dds); min_p = min(pnls)
    ratio_total = total / abs(worst_dd) if worst_dd < 0 else float('inf')
    print('  ' + '-' * 90)
    print(f"  {'SUM / WORST':<40} ${total:>+12,.0f} ${worst_dd:>+10,.0f} {'':<10} {ratio_total:>6.2f}x")
    print(f"  Min split P&L: ${min_p:+,.0f}")

    # Compare to original flat-$50K numbers (from study_orb_capped.py run)
    print(f"\n{'='*90}")
    print(f"COMPARISON: flat $50K vs risk-parity ($500 risk)")
    print(f"{'='*90}")
    print(f"{'metric':<20} {'flat $50K (cap5)':>22} {'risk-parity (cap5)':>22}")
    # Original flat $50K cap5 numbers (from prior run)
    flat_pnls = [251303, 280136, 240131]
    flat_dds = [-51536, -29636, -83538]
    flat_worst_days = [-20941, -26275, -18255]
    flat_sum = sum(flat_pnls); flat_worst_dd = min(flat_dds)
    flat_ratio = flat_sum / abs(flat_worst_dd)
    print(f"{'Sum P&L':<20} ${flat_sum:>+20,.0f} ${total:>+20,.0f}")
    print(f"{'Worst DD':<20} ${flat_worst_dd:>+20,.0f} ${worst_dd:>+20,.0f}")
    print(f"{'P&L / |DD|':<20} {flat_ratio:>22.2f}x {ratio_total:>21.2f}x")
    print(f"{'Worst single day':<20} ${min(flat_worst_days):>+20,.0f} "
          f"${min(r['equity']['worst_day'] for r in results):>+20,.0f}")

    # Write markdown
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    md_path = f"{OUT_DIR}/orb_risk_parity_{ts}.md"
    with open(md_path, 'w') as f:
        f.write(f"# ORB risk-parity sizing — walk-forward\n\n")
        f.write(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n\n")
        f.write(f"## Motivation\n\n")
        f.write(f"Flat $50K position with stop at range_low yields variable per-trade risk:\n")
        f.write(f"wide-range stocks ~10% stop = $5K loss; narrow-range ~2% stop = $1K loss. "
                f"This is what drove $-3K single-trade losses in earlier runs.\n\n")
        f.write(f"**Risk-parity sizing**: fix dollar RISK per trade (${RISK_PER_TRADE_USD:.0f}), "
                f"compute shares = risk / stop_distance. Wide ranges get fewer shares, narrow "
                f"ranges get more.\n\n")
        f.write(f"Parameters:\n")
        f.write(f"- Risk per trade: **${RISK_PER_TRADE_USD:,.0f}**\n")
        f.write(f"- Min stop %: **{MIN_STOP_PCT}%** (prevents infinite sizing on tight ranges)\n")
        f.write(f"- Max position cap: **${MAX_POSITION_USD:,.0f}**\n\n")

        f.write(f"## Per-trade distribution — flat $50K vs risk-parity\n\n")
        f.write(f"| metric | flat $50K | risk-parity |\n|---|---:|---:|\n")
        for metric, label in [('n', 'n'), ('avg', 'avg'), ('med', 'median'),
                               ('best', 'best win'), ('worst', 'worst loss'),
                               ('p95_loss', '95th-pct loss'), ('total', 'total')]:
            old = old_stats[metric]; new = new_stats[metric]
            if metric == 'n':
                f.write(f"| {label} | {old:,} | {new:,} |\n")
            else:
                f.write(f"| {label} | ${old:+,.0f} | ${new:+,.0f} |\n")

        f.write(f"\n## Walk-forward: cap{MAX_CONCURRENT} + Q4-pref + adaptive (risk-parity base)\n\n")
        f.write(f"| Split | P&L | Max DD | Worst day | P&L/|DD| | per-trade worst |\n")
        f.write(f"|---|---:|---:|---:|---:|---:|\n")
        for r in results:
            eq = r['equity']; pt = r['per_trade']
            ratio = eq['total_pnl'] / abs(eq['max_dd']) if eq['max_dd'] < 0 else float('inf')
            f.write(f"| {r['split_name']} | ${eq['total_pnl']:+,.0f} | ${eq['max_dd']:+,.0f} | "
                    f"${eq['worst_day']:+,.0f} | {ratio:.2f}x | ${pt['worst']:+,.0f} |\n")
        f.write(f"\n**Sum P&L**: ${total:+,.0f}  **Min split**: ${min_p:+,.0f}  "
                f"**Worst DD**: ${worst_dd:+,.0f}  **P&L/DD**: {ratio_total:.2f}x\n\n")

        f.write(f"## vs flat $50K (same cap5 pipeline)\n\n")
        f.write(f"| metric | flat $50K | risk-parity |\n|---|---:|---:|\n")
        f.write(f"| Sum P&L | ${flat_sum:+,.0f} | ${total:+,.0f} |\n")
        f.write(f"| Worst DD | ${flat_worst_dd:+,.0f} | ${worst_dd:+,.0f} |\n")
        f.write(f"| P&L/|DD| | {flat_ratio:.2f}x | {ratio_total:.2f}x |\n")
        f.write(f"| Worst single day | ${min(flat_worst_days):+,.0f} | "
                f"${min(r['equity']['worst_day'] for r in results):+,.0f} |\n")

    print(f"\nReport: {md_path}")


if __name__ == '__main__':
    main()
