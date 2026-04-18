#!/usr/bin/env python3
"""Per-tier lever isolation (Phase A.2).

Tests individual knob changes INDEPENDENTLY per tier. Baseline = current
shipping config. For each lever, applies the change ONLY to the specified
tier and reports TRAIN/VAL/HOLDOUT deltas for that tier (and total).

Tiers: A (>=20%), E (10-19.99%), edge (<10%). edge ignored (too small).

Levers tested per tier:
  - macd_strong_mult variants: 1.0, 1.25, 1.5 (current), 1.8, 2.0, 2.5
  - macd_normal_mult variants: 0.25, 0.5, 0.75, 1.0 (current), 1.25, 1.5
  - rule weight drops: r3=0, r5=0, r1=0, r7=0 (all drop individually)
  - v_rev bonus variants: 0.4 (current), 0.5, 0.6, 0.7, 0.8, 1.0 (A-tier only)
  - Conv threshold per tier: 1.0, 1.2, 1.3, 1.4, 1.5, 1.6

Output: research/per_tier_lever_isolation.md
"""
from __future__ import annotations

import csv
from collections import defaultdict
from typing import List, Dict

from holistic_per_tier import load_trades, classify_tier, MIN_CONV

# -------------------------------------------------------------------------
# Rule weights (default V2_clean)
# -------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    'r1': 0.3,    # pole_gain
    'r2p': 0.3,   # flag tight
    'r2n': -0.3,  # flag loose
    'r3': 0.3,    # vol_ratio
    'r4p': 0.3,   # SPY good
    'r4n': -0.5,  # SPY bad
    'r5': 0.2,    # retracement
    'r7': 0.2,    # vwap_dist
    'r8': -0.3,   # gap_fading
    'r9': 0.4,    # V-reversal
}


def recompute_conv(t, weights):
    """Recompute conv_mult under alternate weights. Rule firing patterns
    inherit from cache (not recomputed)."""
    raw = 1.0
    if t['r1'] > 0: raw += weights['r1']
    if t['r2'] > 0: raw += weights['r2p']
    elif t['r2'] < 0: raw += weights['r2n']
    if t['r3'] > 0: raw += weights['r3']
    if t['r4'] > 0: raw += weights['r4p']
    elif t['r4'] < 0: raw += weights['r4n']
    if t['r5'] > 0: raw += weights['r5']
    if t['r7'] > 0: raw += weights['r7']
    if t['r8'] < 0: raw += weights['r8']
    if t['r9'] > 0:
        # r9 is stored as 0.4 firing; rescale to new weight.
        # t['r9'] > 0 means r9 fired in original cache.
        raw += weights['r9']
    return max(0.25, min(3.0, raw))


def sim_trade_pnl(t, target_tier, macd_strong, macd_normal, weights,
                   conv_threshold):
    """Simulate PnL for a single trade under the given per-tier overrides.

    target_tier: 'A', 'E', or None (apply to all). If tier matches target_tier,
    apply the overrides. If not, keep baseline sizing.

    Returns (new_pnl, kept). kept=False means conv filter rejected.
    """
    apply_overrides = (target_tier is None) or (t['tier'] == target_tier)

    # Recompute conv with either new weights (if tier matches) or default
    if apply_overrides:
        new_conv = recompute_conv(t, weights)
        active_conv_threshold = conv_threshold
    else:
        new_conv = t['conv_mult']
        active_conv_threshold = MIN_CONV

    if new_conv < active_conv_threshold:
        return (0.0, False)

    # Recompute macd zone applied
    old_macd = t['macd_mult']
    if apply_overrides:
        if abs(old_macd - 1.5) < 0.01:
            new_macd = macd_strong
        elif abs(old_macd - 1.0) < 0.01:
            new_macd = macd_normal
        else:
            new_macd = old_macd
    else:
        new_macd = old_macd

    # Scale pnl: new_pnl = pnl_at_1x * new_conv * new_macd
    # pnl_at_1x already computed in load_trades()
    new_pnl = t['pnl_at_1x'] * new_conv * new_macd
    return (new_pnl, True)


def simulate_config(trades, target_tier=None, macd_strong=1.5,
                     macd_normal=1.0, weights=None,
                     conv_threshold=MIN_CONV):
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    by_split_tier = defaultdict(lambda: defaultdict(list))
    for t in trades:
        pnl, kept = sim_trade_pnl(t, target_tier, macd_strong, macd_normal,
                                    weights, conv_threshold)
        if kept:
            by_split_tier[t['split']][t['tier']].append({
                'date': t['date'], 'pnl': pnl, 'tier': t['tier']
            })
    return by_split_tier


def summarize_by_tier(by_split_tier, tiers=('A', 'E', 'edge')):
    """Return {split: {tier: {n, pnl}}, 'total': {n, pnl}} per split."""
    out = {}
    for split in by_split_tier:
        out[split] = {}
        for tier in tiers:
            ts = by_split_tier[split][tier]
            n = len(ts)
            pnl = sum(t['pnl'] for t in ts)
            out[split][tier] = {'n': n, 'pnl': pnl}
        # Total across tiers
        all_ts = []
        for tier in tiers:
            all_ts.extend(by_split_tier[split][tier])
        out[split]['total'] = {'n': len(all_ts),
                                'pnl': sum(t['pnl'] for t in all_ts)}
    return out


def delta_table(baseline, candidate):
    """Return table rows comparing baseline vs candidate."""
    splits = ['TRAIN (2025 Jan-Jul)', 'VAL (2025 Aug-Dec)', 'HOQ1 (2026 Q1)']
    rows = []
    for split in splits:
        for tier in ['A', 'E', 'total']:
            b = baseline[split][tier]
            c = candidate[split][tier]
            rows.append({
                'split': split, 'tier': tier,
                'base_n': b['n'], 'base_pnl': b['pnl'],
                'cand_n': c['n'], 'cand_pnl': c['pnl'],
                'delta_pnl': c['pnl'] - b['pnl'],
            })
    return rows


def print_delta_summary(name, baseline, candidate):
    """Compact per-tier summary."""
    splits = ['TRAIN (2025 Jan-Jul)', 'VAL (2025 Aug-Dec)', 'HOQ1 (2026 Q1)']
    total_delta = 0.0
    train_val_delta = 0.0
    holdout_delta = 0.0
    for split in splits:
        for tier in ['A', 'E', 'total']:
            if tier != 'total':
                continue
            d = candidate[split][tier]['pnl'] - baseline[split][tier]['pnl']
            if 'HOQ1' in split:
                holdout_delta += d
            else:
                train_val_delta += d
            total_delta += d
    print(f"- **{name}**: T+V Δ ${train_val_delta:+,.0f}, "
          f"HOQ1 Δ ${holdout_delta:+,.0f}, **grand ${total_delta:+,.0f}**")
    return total_delta


def run_lever_isolation(trades):
    """Run per-tier lever isolation. Return results table."""
    baseline_raw = simulate_config(trades)
    baseline = summarize_by_tier(baseline_raw)

    print("# Per-tier lever isolation (10%-frame, conv>=1.4 baseline)\n")
    print("Each row = single knob change applied to ONE tier only. Shows "
          "Δ vs baseline on TRAIN+VAL and HOLDOUT Q1 2026 independently.\n")

    # Baseline summary
    print("## Baseline reference\n")
    for split in ['TRAIN (2025 Jan-Jul)', 'VAL (2025 Aug-Dec)', 'HOQ1 (2026 Q1)']:
        for tier in ['A', 'E', 'total']:
            b = baseline[split][tier]
            print(f"- {split} — {tier}: n={b['n']}, PnL=${b['pnl']:+,.0f}")
    print()

    # ---------------------------
    # A. MACD strong multiplier
    # ---------------------------
    print("## A. MACD strong multiplier sweep per tier\n")
    print("Current strong_mult=1.5 for macd>0.5%. What if we change it?\n")
    for target_tier in ['A', 'E']:
        print(f"\n### Target tier: {target_tier}\n")
        for new_strong in [0.5, 0.75, 1.0, 1.25, 1.5, 1.8, 2.0, 2.5]:
            cand_raw = simulate_config(trades, target_tier=target_tier,
                                          macd_strong=new_strong)
            cand = summarize_by_tier(cand_raw)
            name = f"{target_tier}: macd_strong {new_strong}"
            print_delta_summary(name, baseline, cand)

    # ---------------------------
    # B. MACD normal multiplier
    # ---------------------------
    print("\n## B. MACD normal multiplier sweep per tier\n")
    print("Current normal_mult=1.0. Drop it to downsize the neutral-zone "
          "trades, especially E-tier MACD 1.0 bucket (the $-14,734 loser).\n")
    for target_tier in ['A', 'E']:
        print(f"\n### Target tier: {target_tier}\n")
        for new_normal in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]:
            cand_raw = simulate_config(trades, target_tier=target_tier,
                                          macd_normal=new_normal)
            cand = summarize_by_tier(cand_raw)
            name = f"{target_tier}: macd_normal {new_normal}"
            print_delta_summary(name, baseline, cand)

    # ---------------------------
    # C. Rule weight drops per tier
    # ---------------------------
    print("\n## C. Rule weight drops per tier\n")
    print("Zero out each rule per tier, see if PnL improves (= rule was "
          "noise/counter-signal in that tier).\n")
    rule_drops = ['r1', 'r3', 'r5', 'r7', 'r4p', 'r4n', 'r2p', 'r2n']
    for target_tier in ['A', 'E']:
        print(f"\n### Target tier: {target_tier}\n")
        for rule in rule_drops:
            weights = DEFAULT_WEIGHTS.copy()
            weights[rule] = 0.0
            cand_raw = simulate_config(trades, target_tier=target_tier,
                                          weights=weights)
            cand = summarize_by_tier(cand_raw)
            name = f"{target_tier}: drop {rule} (0.0)"
            print_delta_summary(name, baseline, cand)

    # ---------------------------
    # D. V-reversal bonus sweep (A-tier only, since E has no r9 trades)
    # ---------------------------
    print("\n## D. V-reversal bonus sweep (A-tier only — r9 doesn't fire in E)\n")
    for new_bonus in [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]:
        weights = DEFAULT_WEIGHTS.copy()
        weights['r9'] = new_bonus
        cand_raw = simulate_config(trades, target_tier='A', weights=weights)
        cand = summarize_by_tier(cand_raw)
        name = f"A: v_rev_bonus {new_bonus}"
        print_delta_summary(name, baseline, cand)

    # ---------------------------
    # E. Conv threshold per tier
    # ---------------------------
    print("\n## E. Conv threshold per tier\n")
    print("Current threshold 1.4. Lower = more trades pass. Higher = stricter.\n")
    for target_tier in ['A', 'E']:
        print(f"\n### Target tier: {target_tier}\n")
        for new_threshold in [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]:
            cand_raw = simulate_config(trades, target_tier=target_tier,
                                          conv_threshold=new_threshold)
            cand = summarize_by_tier(cand_raw)
            name = f"{target_tier}: conv_threshold {new_threshold}"
            print_delta_summary(name, baseline, cand)


def main():
    trades = load_trades()
    run_lever_isolation(trades)


if __name__ == "__main__":
    main()
