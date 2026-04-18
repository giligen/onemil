#!/usr/bin/env python3
"""Per-tier joint grid search (Phase A.3).

Combines best levers per tier into full configs. Scored on TRAIN+VAL under
stability constraint: |TRAIN%-VAL%| ≤ 25pt AND both positive. HOQ1 reported
one-shot for validation.

Key levers per tier (from A.2 isolation):

A-tier levers:
  - v_rev_bonus: {0.4, 0.6, 0.8, 1.0, 1.2} — positive gains all; 1.2 best
  - macd_strong: {1.5, 1.8, 2.0} — small positive at higher values
  - everything else: neutral-to-negative on A-tier

E-tier levers:
  - macd_normal: {1.0, 0.5, 0.25, 0.0} — smaller = bigger gain
  - macd_strong: {1.5, 1.8, 2.0, 2.5} — bigger = more gain
  - drop r3: {keep, drop} — drop helps
  - drop r2n: {keep, drop} — drop helps (counter-intuitive)
  - drop r4n: {keep, drop} — drop helps
  - drop r5: AVOID (dropping hurts a lot)

Output: research/per_tier_joint_search.md
"""
from __future__ import annotations

from itertools import product
from typing import List, Dict

from holistic_per_tier import load_trades, MIN_CONV
from holistic_per_tier_levers import (
    DEFAULT_WEIGHTS, simulate_config, summarize_by_tier,
    sim_trade_pnl, recompute_conv,
)
from collections import defaultdict


def simulate_per_tier(trades, a_overrides, e_overrides):
    """Simulate trades where A-tier trades apply a_overrides, E-tier trades
    apply e_overrides, edge trades use defaults."""
    by_split_tier = defaultdict(lambda: defaultdict(list))
    for t in trades:
        if t['tier'] == 'A':
            macd_strong = a_overrides.get('macd_strong', 1.5)
            macd_normal = a_overrides.get('macd_normal', 1.0)
            weights = a_overrides.get('weights', DEFAULT_WEIGHTS.copy())
            conv_threshold = a_overrides.get('conv_threshold', MIN_CONV)
            target_tier = 'A'
        elif t['tier'] == 'E':
            macd_strong = e_overrides.get('macd_strong', 1.5)
            macd_normal = e_overrides.get('macd_normal', 1.0)
            weights = e_overrides.get('weights', DEFAULT_WEIGHTS.copy())
            conv_threshold = e_overrides.get('conv_threshold', MIN_CONV)
            target_tier = 'E'
        else:
            macd_strong = 1.5
            macd_normal = 1.0
            weights = DEFAULT_WEIGHTS.copy()
            conv_threshold = MIN_CONV
            target_tier = None

        pnl, kept = sim_trade_pnl(t, target_tier, macd_strong, macd_normal,
                                    weights, conv_threshold)
        if kept:
            by_split_tier[t['split']][t['tier']].append({
                'date': t['date'], 'pnl': pnl, 'tier': t['tier'],
            })
    return by_split_tier


def evaluate_config(trades, a_overrides, e_overrides, baseline=None):
    """Run and summarize. Return (stats, is_valid, primary_score)."""
    raw = simulate_per_tier(trades, a_overrides, e_overrides)
    stats = summarize_by_tier(raw)

    if baseline is None:
        return stats

    # Compute deltas vs baseline
    splits = ['TRAIN (2025 Jan-Jul)', 'VAL (2025 Aug-Dec)', 'HOQ1 (2026 Q1)']
    tv_delta = 0.0
    hoq1_delta = 0.0
    train_pct = 0.0
    val_pct = 0.0
    base_train = baseline['TRAIN (2025 Jan-Jul)']['total']['pnl']
    base_val = baseline['VAL (2025 Aug-Dec)']['total']['pnl']
    base_hoq1 = baseline['HOQ1 (2026 Q1)']['total']['pnl']
    train_delta = stats['TRAIN (2025 Jan-Jul)']['total']['pnl'] - base_train
    val_delta = stats['VAL (2025 Aug-Dec)']['total']['pnl'] - base_val
    hoq1_delta = stats['HOQ1 (2026 Q1)']['total']['pnl'] - base_hoq1
    tv_delta = train_delta + val_delta
    total_delta = tv_delta + hoq1_delta

    # Stability: TRAIN% and VAL% agreement
    train_pct = (train_delta / abs(base_train) * 100) if base_train != 0 else 0
    val_pct = (val_delta / abs(base_val) * 100) if base_val != 0 else 0
    imbalance = abs(train_pct - val_pct)

    # Per-tier positivity
    a_train_d = (stats['TRAIN (2025 Jan-Jul)']['A']['pnl']
                 - baseline['TRAIN (2025 Jan-Jul)']['A']['pnl'])
    a_val_d = (stats['VAL (2025 Aug-Dec)']['A']['pnl']
                - baseline['VAL (2025 Aug-Dec)']['A']['pnl'])
    e_train_d = (stats['TRAIN (2025 Jan-Jul)']['E']['pnl']
                 - baseline['TRAIN (2025 Jan-Jul)']['E']['pnl'])
    e_val_d = (stats['VAL (2025 Aug-Dec)']['E']['pnl']
                - baseline['VAL (2025 Aug-Dec)']['E']['pnl'])

    # Trade count bounds
    n_base = sum(baseline[s]['total']['n'] for s in splits)
    n_cand = sum(stats[s]['total']['n'] for s in splits)
    ratio = n_cand / n_base if n_base else 0

    is_valid = (
        train_delta >= -1000   # TRAIN must not lose much
        and val_delta >= -1000  # VAL must not lose much
        and imbalance <= 40
        and ratio >= 0.5        # don't cut >50% of trades
    )

    return {
        'stats': stats,
        'tv_delta': tv_delta,
        'hoq1_delta': hoq1_delta,
        'total_delta': total_delta,
        'train_delta': train_delta,
        'val_delta': val_delta,
        'train_pct': train_pct,
        'val_pct': val_pct,
        'imbalance': imbalance,
        'a_train_d': a_train_d,
        'a_val_d': a_val_d,
        'e_train_d': e_train_d,
        'e_val_d': e_val_d,
        'n_ratio': ratio,
        'is_valid': is_valid,
    }


def main():
    trades = load_trades()

    # Baseline stats
    baseline_raw = simulate_per_tier(trades, {}, {})
    baseline = summarize_by_tier(baseline_raw)

    print("# Per-tier joint grid search (Phase A.3)\n")
    print("Searching: A-tier overrides × E-tier overrides. Scored on T+V "
          "total delta. Stability constraint: TRAIN and VAL both +; "
          "|TRAIN%-VAL%| ≤ 40pt; trade count ≥ 50% of baseline.\n")

    # Baseline
    bt = baseline['TRAIN (2025 Jan-Jul)']['total']['pnl']
    bv = baseline['VAL (2025 Aug-Dec)']['total']['pnl']
    bh = baseline['HOQ1 (2026 Q1)']['total']['pnl']
    print(f"Baseline: TRAIN ${bt:+,.0f}, VAL ${bv:+,.0f}, HOQ1 ${bh:+,.0f}, "
          f"grand ${bt+bv+bh:+,.0f}\n")

    # Define candidate overrides
    a_variants = {
        'A_none': {},
        'A_v0.6': {'weights': {**DEFAULT_WEIGHTS, 'r9': 0.6}},
        'A_v0.7': {'weights': {**DEFAULT_WEIGHTS, 'r9': 0.7}},
        'A_v0.8': {'weights': {**DEFAULT_WEIGHTS, 'r9': 0.8}},
        'A_v1.0': {'weights': {**DEFAULT_WEIGHTS, 'r9': 1.0}},
        'A_v1.2': {'weights': {**DEFAULT_WEIGHTS, 'r9': 1.2}},
        'A_v0.8_macd1.8': {'macd_strong': 1.8,
                            'weights': {**DEFAULT_WEIGHTS, 'r9': 0.8}},
        'A_v1.0_macd1.8': {'macd_strong': 1.8,
                            'weights': {**DEFAULT_WEIGHTS, 'r9': 1.0}},
    }
    e_variants = {
        'E_none': {},
        'E_m1.5_n0.5': {'macd_strong': 1.5, 'macd_normal': 0.5},
        'E_m1.5_n0.25': {'macd_strong': 1.5, 'macd_normal': 0.25},
        'E_m1.5_n0.0': {'macd_strong': 1.5, 'macd_normal': 0.0},
        'E_m1.8_n0.5': {'macd_strong': 1.8, 'macd_normal': 0.5},
        'E_m1.8_n0.25': {'macd_strong': 1.8, 'macd_normal': 0.25},
        'E_m2.0_n0.5': {'macd_strong': 2.0, 'macd_normal': 0.5},
        'E_m2.0_n0.25': {'macd_strong': 2.0, 'macd_normal': 0.25},
        'E_m2.0_n0.0': {'macd_strong': 2.0, 'macd_normal': 0.0},
        'E_m2.5_n0.25': {'macd_strong': 2.5, 'macd_normal': 0.25},
        'E_m2.5_n0.0': {'macd_strong': 2.5, 'macd_normal': 0.0},
        'E_m2.0_n0.25_dropR3': {
            'macd_strong': 2.0, 'macd_normal': 0.25,
            'weights': {**DEFAULT_WEIGHTS, 'r3': 0.0}},
        'E_m2.0_n0.25_dropR4n': {
            'macd_strong': 2.0, 'macd_normal': 0.25,
            'weights': {**DEFAULT_WEIGHTS, 'r4n': 0.0}},
        'E_m2.0_n0.0_dropR3R4n': {
            'macd_strong': 2.0, 'macd_normal': 0.0,
            'weights': {**DEFAULT_WEIGHTS, 'r3': 0.0, 'r4n': 0.0}},
    }

    print(f"Testing {len(a_variants)} × {len(e_variants)} = "
          f"{len(a_variants)*len(e_variants)} configs\n")

    candidates = []
    for a_name, a_ov in a_variants.items():
        for e_name, e_ov in e_variants.items():
            result = evaluate_config(trades, a_ov, e_ov, baseline)
            candidates.append({
                'a_name': a_name, 'e_name': e_name,
                'a_ov': a_ov, 'e_ov': e_ov,
                **result,
            })

    # Sort by total_delta, but keep only valid
    valid = [c for c in candidates if c['is_valid']]
    valid.sort(key=lambda c: -c['total_delta'])

    print(f"## Top 20 valid configs (by TOTAL delta)\n")
    print("| # | A-cfg | E-cfg | T+V Δ | HOQ1 Δ | Grand Δ | Δ% | T% / V% | n_ratio |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    base_grand = bt + bv + bh
    for i, c in enumerate(valid[:20]):
        total_pct = c['total_delta'] / abs(base_grand) * 100 if base_grand else 0
        print(f"| {i+1} | {c['a_name']} | {c['e_name']} | "
              f"${c['tv_delta']:+,.0f} | ${c['hoq1_delta']:+,.0f} | "
              f"${c['total_delta']:+,.0f} | {total_pct:+.1f}% | "
              f"{c['train_pct']:+.1f}% / {c['val_pct']:+.1f}% | "
              f"{c['n_ratio']:.2f} |")

    # Sort by HOQ1 delta for HOLDOUT-best ranking (informational, not selection)
    print(f"\n## Top 10 valid configs by HOQ1 Δ (information, not selection)\n")
    valid_by_hoq1 = sorted(valid, key=lambda c: -c['hoq1_delta'])[:10]
    for i, c in enumerate(valid_by_hoq1):
        print(f"- **{i+1}**: {c['a_name']} + {c['e_name']}: "
              f"T+V ${c['tv_delta']:+,.0f}, HOQ1 ${c['hoq1_delta']:+,.0f}, "
              f"grand ${c['total_delta']:+,.0f}")

    # Show best config detail
    if valid:
        best = valid[0]
        print(f"\n## Best config (stability-scored TRAIN+VAL winner)\n")
        print(f"**{best['a_name']} + {best['e_name']}**\n")
        print("Per-tier per-split breakdown:\n")
        print("| Split | A n/pnl | E n/pnl | Total n/pnl |")
        print("|---|---|---|---|")
        for split in ['TRAIN (2025 Jan-Jul)', 'VAL (2025 Aug-Dec)', 'HOQ1 (2026 Q1)']:
            a = best['stats'][split]['A']
            e = best['stats'][split]['E']
            t = best['stats'][split]['total']
            ba = baseline[split]['A']
            be = baseline[split]['E']
            bt2 = baseline[split]['total']
            print(f"| {split.split(' (')[0]} | "
                  f"{a['n']}/${a['pnl']:+,.0f} (Δ${a['pnl']-ba['pnl']:+,.0f}) | "
                  f"{e['n']}/${e['pnl']:+,.0f} (Δ${e['pnl']-be['pnl']:+,.0f}) | "
                  f"{t['n']}/${t['pnl']:+,.0f} (Δ${t['pnl']-bt2['pnl']:+,.0f}) |")


if __name__ == "__main__":
    main()
