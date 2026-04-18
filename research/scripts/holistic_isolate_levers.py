#!/usr/bin/env python3
"""Isolate individual levers to understand where the +80% gain comes from.

Runs a series of 1-change experiments starting from baseline, each toggling
ONE knob. Shows which changes contribute what, and whether they stack.
"""
from __future__ import annotations

from holistic_optimizer import (
    load_all, baseline_stats, simulate, Params,
)
from holistic_search_v2 import TIER_VARIANTS


EXPERIMENTS = [
    # Name, overrides from baseline
    ('Baseline (shipping config)', {}),

    # Single-lever experiments:
    ('L1: macd_strong 1.5→1.8', {'macd_strong': 1.8}),
    ('L2: macd_strong 1.5→2.0', {'macd_strong': 2.0}),
    ('L3: macd_normal 1.0→0.75 (shrink neutral zone)', {'macd_normal': 0.75}),
    ('L4: threshold 1.4→1.2 (let more in)', {'min_threshold': 1.2}),
    ('L5: threshold 1.4→1.0', {'min_threshold': 1.0}),
    ('L6: cap 3.0→3.5', {'cap': 3.5}),
    ('L7: cap 3.0→4.0', {'cap': 4.0}),
    ('L8: r9 bonus 0.4→0.7 (bigger V-rev)', {'w_r9': 0.7}),
    ('L9: r1 pole 0.3→0.4', {'w_r1': 0.4}),
    ('L10: r3 vol_ratio 0.3→0.0 (audit drop)', {'w_r3': 0.0}),
    ('L11: r7 vwap 0.2→0.0 (audit drop)', {'w_r7': 0.0}),
    ('L12: r3 vol_ratio 0.3→0.45 (search bump)', {'w_r3': 0.45}),
    ('L13: T3 <$5 @2.0x added', {'tiers': TIER_VARIANTS['+T3_small_2.0']}),
    ('L14: T3 <$5 @1.5x added', {'tiers': TIER_VARIANTS['+T3_small_1.5']}),

    # Stacked levers (order-independent — each builds on baseline):
    ('S1: L2 + L8 (macd2.0 + r9=0.7)', {'macd_strong': 2.0, 'w_r9': 0.7}),
    ('S2: L2 + L8 + L13 (+ T3)', {'macd_strong': 2.0, 'w_r9': 0.7,
                                    'tiers': TIER_VARIANTS['+T3_small_2.0']}),
    ('S3: S2 + threshold 1.2', {'macd_strong': 2.0, 'w_r9': 0.7, 'min_threshold': 1.2,
                                 'tiers': TIER_VARIANTS['+T3_small_2.0']}),
    ('S4: S2 + threshold 1.3 (compromise)', {'macd_strong': 2.0, 'w_r9': 0.7,
                                              'min_threshold': 1.3,
                                              'tiers': TIER_VARIANTS['+T3_small_2.0']}),

    # CONSERVATIVE SHIP CANDIDATES (keep cap=3.0, weights close to current)
    ('C1: macd1.8 + r9=0.6 + T3<5@1.5', {'macd_strong': 1.8, 'w_r9': 0.6,
                                          'tiers': TIER_VARIANTS['+T3_small_1.5']}),
    ('C2: macd2.0 + r9=0.6 + T3<5@1.5', {'macd_strong': 2.0, 'w_r9': 0.6,
                                          'tiers': TIER_VARIANTS['+T3_small_1.5']}),
    ('C3: macd2.0 + r9=0.6 + T3<5@2.0', {'macd_strong': 2.0, 'w_r9': 0.6,
                                          'tiers': TIER_VARIANTS['+T3_small_2.0']}),

    # AGGRESSIVE RECOMMENDED from v2 search
    ('A1: full Stage2 winner',
     {'w_r1': 0.4, 'w_r2p': 0.4, 'w_r2n': -0.15, 'w_r3': 0.45, 'w_r5': 0.3,
      'w_r7': 0.2, 'w_r9': 0.7, 'min_threshold': 1.2, 'cap': 4.0,
      'macd_strong': 2.0, 'tiers': TIER_VARIANTS['+T3_small_2.0']}),
]


def main():
    trades = load_all()
    base = baseline_stats(trades)
    base_all = sum(base[s]['pnl'] for s in ['TRAIN', 'VAL', 'HOQ1', 'HOAPR'])

    print("# Lever isolation — where does each gain come from\n")
    print(f"Baseline grand total PnL: ${base_all:+,.0f}\n")
    print("Each row = baseline with ONE (or combined) override applied. "
          "Shows ΔTRAIN, ΔVAL, ΔHOQ1, ΔHOAPR, ΔGrand, Δ%.\n")
    print("| Experiment | TRAIN Δ | VAL Δ | HOQ1 Δ | HOAPR Δ | HOLDOUT Δ | Grand Δ | Δ% | TRn | VLn | HOn |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for name, overrides in EXPERIMENTS:
        p = Params(**overrides)
        s = simulate(trades, p)
        tr_d = s['TRAIN']['pnl'] - base['TRAIN']['pnl']
        vl_d = s['VAL']['pnl'] - base['VAL']['pnl']
        q1_d = s['HOQ1']['pnl'] - base['HOQ1']['pnl']
        ap_d = s['HOAPR']['pnl'] - base['HOAPR']['pnl']
        ho_d = q1_d + ap_d
        total_d = tr_d + vl_d + ho_d
        pct = total_d / base_all * 100
        h_n = s['HOQ1']['n'] + s['HOAPR']['n']
        print(f"| {name} | ${tr_d:+,.0f} | ${vl_d:+,.0f} | ${q1_d:+,.0f} | "
              f"${ap_d:+,.0f} | ${ho_d:+,.0f} | ${total_d:+,.0f} | {pct:+.1f}% | "
              f"{s['TRAIN']['n']} | {s['VAL']['n']} | {h_n} |")


if __name__ == "__main__":
    main()
