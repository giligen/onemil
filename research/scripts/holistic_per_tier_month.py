#!/usr/bin/env python3
"""Month-by-month stability + max drawdown for S2 variants.

Confirms the S2 lift isn't concentrated in a single month (which would
signal overfit / lucky outlier dependence).
"""
from __future__ import annotations

from collections import defaultdict

from holistic_per_tier import load_trades, MIN_CONV
from holistic_per_tier_levers import DEFAULT_WEIGHTS, sim_trade_pnl


S2_VARIANTS = {
    'S2-cons': {
        'a': {},  # A-tier: no changes
        'e': {'macd_strong': 2.5, 'macd_normal': 0.0},
    },
    'S2-mid': {
        'a': {'weights': {**DEFAULT_WEIGHTS, 'r9': 0.8}},
        'e': {'macd_strong': 2.5, 'macd_normal': 0.0},
    },
    'S2-max': {
        'a': {'weights': {**DEFAULT_WEIGHTS, 'r9': 1.2}},
        'e': {'macd_strong': 2.5, 'macd_normal': 0.0},
    },
}


def run_config(trades, variant=None):
    """Simulate with variant's per-tier overrides. variant=None → baseline."""
    if variant is None:
        a_ov = {}
        e_ov = {}
    else:
        a_ov = S2_VARIANTS[variant]['a']
        e_ov = S2_VARIANTS[variant]['e']

    results = []
    for t in trades:
        if t['tier'] == 'A':
            ov = a_ov
            target_tier = 'A'
        elif t['tier'] == 'E':
            ov = e_ov
            target_tier = 'E'
        else:
            ov = {}
            target_tier = None

        weights = ov.get('weights', DEFAULT_WEIGHTS.copy())
        macd_strong = ov.get('macd_strong', 1.5)
        macd_normal = ov.get('macd_normal', 1.0)
        conv_threshold = ov.get('conv_threshold', MIN_CONV)

        pnl, kept = sim_trade_pnl(t, target_tier, macd_strong, macd_normal,
                                    weights, conv_threshold)
        if kept:
            results.append({**t, 'sim_pnl': pnl})
    return results


def max_dd(trades):
    """Compute max running-sum drawdown, sorted by date."""
    running, peak, mdd = 0.0, 0.0, 0.0
    for t in sorted(trades, key=lambda x: x['date']):
        running += t['sim_pnl']
        peak = max(peak, running)
        mdd = min(mdd, running - peak)
    return mdd


def main():
    trades = load_trades()

    print("# S2 variants — monthly stability + max drawdown\n")
    print("Per-month P&L across 2025 full + Q1 2026. Shows whether S2 lift "
          "is stable across months or concentrated in outliers.\n")

    # Monthly breakdown
    print("## Monthly P&L (all tiers, all splits)\n")
    print("| Month | Baseline | S2-cons | S2-mid | S2-max | S2-mid Δ |")
    print("|---|---:|---:|---:|---:|---:|")
    baseline_res = run_config(trades)
    baseline_by_month = defaultdict(float)
    for t in baseline_res:
        month = t['date'][:7]
        baseline_by_month[month] += t['sim_pnl']

    variant_by_month = {}
    for v in S2_VARIANTS:
        variant_by_month[v] = defaultdict(float)
        for t in run_config(trades, v):
            variant_by_month[v][t['date'][:7]] += t['sim_pnl']

    months_sorted = sorted(baseline_by_month.keys())
    for m in months_sorted:
        bp = baseline_by_month[m]
        cs = variant_by_month['S2-cons'][m]
        md = variant_by_month['S2-mid'][m]
        mx = variant_by_month['S2-max'][m]
        delta_mid = md - bp
        print(f"| {m} | ${bp:>+8,.0f} | ${cs:>+8,.0f} | ${md:>+8,.0f} | "
              f"${mx:>+8,.0f} | **${delta_mid:>+7,.0f}** |")

    # Totals
    bt_total = sum(baseline_by_month.values())
    print(f"| **TOTAL** | ${bt_total:+,.0f} | "
          f"${sum(variant_by_month['S2-cons'].values()):+,.0f} | "
          f"${sum(variant_by_month['S2-mid'].values()):+,.0f} | "
          f"${sum(variant_by_month['S2-max'].values()):+,.0f} | |")

    # Per-tier max drawdown
    print("\n## Max drawdown per tier per variant\n")
    print("| Variant | A-tier MDD | E-tier MDD | Combined MDD |")
    print("|---|---:|---:|---:|")

    for name, config in [('Baseline', None), ('S2-cons', 'S2-cons'),
                          ('S2-mid', 'S2-mid'), ('S2-max', 'S2-max')]:
        res = run_config(trades, config)
        a_mdd = max_dd([t for t in res if t['tier'] == 'A'])
        e_mdd = max_dd([t for t in res if t['tier'] == 'E'])
        all_mdd = max_dd(res)
        print(f"| {name} | ${a_mdd:+,.0f} | ${e_mdd:+,.0f} | ${all_mdd:+,.0f} |")

    # Win-rate check
    print("\n## Win rate by variant (all trades)\n")
    print("| Variant | n | WR |")
    print("|---|---:|---:|")
    for name, config in [('Baseline', None), ('S2-cons', 'S2-cons'),
                          ('S2-mid', 'S2-mid'), ('S2-max', 'S2-max')]:
        res = run_config(trades, config)
        # For E-tier macd_normal=0, trades have sim_pnl=0 (skipped-like).
        # Count only non-zero-PnL trades as "taken".
        taken = [t for t in res if abs(t['sim_pnl']) > 0.01]
        if not taken:
            continue
        wins = sum(1 for t in taken if t['sim_pnl'] > 0)
        print(f"| {name} | {len(taken)} | {wins/len(taken)*100:.1f}% |")

    # Sample size summary
    print("\n## Sample-size sanity check\n")
    print("Per-tier × MACD bucket trade counts (baseline, conv>=1.4):\n")
    print("| Tier | MACD 1.0 | MACD 1.5 | total |")
    print("|---|---:|---:|---:|")
    counts = defaultdict(lambda: defaultdict(int))
    for t in trades:
        macd = 'M1.0' if abs(t['macd_mult'] - 1.0) < 0.01 else 'M1.5'
        counts[t['tier']][macd] += 1
    for tier in ['A', 'E', 'edge']:
        m1 = counts[tier]['M1.0']
        m2 = counts[tier]['M1.5']
        print(f"| {tier} | {m1} | {m2} | {m1+m2} |")

    # Key finding: E MACD 1.5 sample size & edge strength
    print("\n### Critical sample size: E-tier MACD 1.5 bucket (the +$16K goldmine)\n")
    e_m15 = [t for t in trades if t['tier'] == 'E'
              and abs(t['macd_mult'] - 1.5) < 0.01]
    for split in ['TRAIN (2025 Jan-Jul)', 'VAL (2025 Aug-Dec)', 'HOQ1 (2026 Q1)']:
        ts = [t for t in e_m15 if t['split'] == split]
        if not ts: continue
        wins = sum(1 for t in ts if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in ts)
        mean_R = sum(t['realized_R'] for t in ts) / len(ts)
        print(f"- **{split}**: {len(ts)} trades, {wins/len(ts)*100:.1f}% WR, "
              f"mean R {mean_R:+.3f}, PnL ${total_pnl:+,.0f}")

    print("\n### Critical sample size: E-tier MACD 1.0 bucket (the -$14K landmine)\n")
    e_m10 = [t for t in trades if t['tier'] == 'E'
              and abs(t['macd_mult'] - 1.0) < 0.01]
    for split in ['TRAIN (2025 Jan-Jul)', 'VAL (2025 Aug-Dec)', 'HOQ1 (2026 Q1)']:
        ts = [t for t in e_m10 if t['split'] == split]
        if not ts: continue
        wins = sum(1 for t in ts if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in ts)
        mean_R = sum(t['realized_R'] for t in ts) / len(ts)
        print(f"- **{split}**: {len(ts)} trades, {wins/len(ts)*100:.1f}% WR, "
              f"mean R {mean_R:+.3f}, PnL ${total_pnl:+,.0f}")


if __name__ == "__main__":
    main()
