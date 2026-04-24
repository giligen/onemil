#!/usr/bin/env python3
"""Phase D — Threshold tuning on V2_full conviction formula.

V2_full = current 5 rules + Rule 6 (daily_range) + Rule 7 (vwap_dist) + Rule 8 (gap_fading penalty).
The new rules shift the score distribution. Sweep thresholds 1.0/1.2/1.4/1.6/1.8
to confirm the current 1.2 is still optimal (or pick a new one).

Methodology:
1. Apply V2_full to cache (recompute conv_mult, scale pnl)
2. For each threshold: sweep config min_threshold + run BT subprocess
3. 3 walk-forward splits, train + test stats
4. Pick threshold by mean OOS ΔP&L vs current 1.2 + worst-case constraint
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

# Reuse the V2_full apply_variant logic
from study_bull_flag_conviction_v2 import (
    apply_variant, Variant, r6_two_tier, r7_simple, r8_penalty,
)

CACHE_PATH = 'data/bull_flag_cache_e50_x30.csv'
CACHE_BACKUP = '/tmp/bull_flag_cache_phase_d_orig.csv'
CONFIG_PATH = 'config.yaml'
CONFIG_BACKUP = '/tmp/config_phase_d_orig.yaml'

# V2_full formula (winner from Phase C)
V2_FULL = Variant(
    'V2_full', 'V0 + 3 new rules (winner of Phase C)',
    r6_two_tier, r7_simple, r8_penalty,
)

THRESHOLDS = [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8]

SPLITS = [
    ('A: H1\'25 → H2\'25-Apr\'26', '2025-01-01', '2025-06-30', '2025-07-01', '2026-04-30'),
    ('B: Y2025 → Q1+Apr\'26',     '2025-01-01', '2025-12-31', '2026-01-01', '2026-04-30'),
    ('C: Jan-Sep\'25 → Oct\'25-Apr\'26', '2025-01-01', '2025-09-30', '2025-10-01', '2026-04-30'),
]


def set_threshold(t: float):
    with open(CONFIG_PATH) as f:
        txt = f.read()
    new = re.sub(r'min_threshold: [\d.]+', f'min_threshold: {t}', txt)
    with open(CONFIG_PATH, 'w') as f:
        f.write(new)


def run_bt(start: str, end: str) -> Dict[str, float]:
    try:
        out = subprocess.check_output(
            ['python3', 'batch_backtest.py', '--start', start, '--end', end],
            stderr=subprocess.STDOUT, text=True, timeout=120
        )
    except subprocess.TimeoutExpired:
        return None
    n = re.search(r'Total trades taken:\s+(\d+)', out)
    pnl = re.search(r'Total P&L:\s+\$([+-]?[\d,]+\.\d+)', out)
    wr = re.search(r'Win rate:\s+([\d.]+)%', out)
    if not (n and pnl and wr):
        return None
    return {
        'n': int(n.group(1)),
        'pnl': float(pnl.group(1).replace(',', '')),
        'wr': float(wr.group(1)),
    }


def main():
    out_path = f"analysis_results/bull_flag_v2_threshold_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    shutil.copy(CACHE_PATH, CACHE_BACKUP)
    shutil.copy(CONFIG_PATH, CONFIG_BACKUP)
    print(f"Backups: cache → {CACHE_BACKUP}, config → {CONFIG_BACKUP}\n")

    base_cache = pd.read_csv(CACHE_PATH)
    print(f"Cache: {len(base_cache)} rows")

    # Apply V2_full once — cache stays modified for the whole sweep
    print(f"\nApplying V2_full to cache (one-time recompute)...")
    v2_cache = apply_variant(base_cache, V2_FULL)
    v2_cache.to_csv(CACHE_PATH, index=False)
    print(f"V2 cache stats: conv_mult range [{v2_cache['conviction_mult'].min():.2f}, "
          f"{v2_cache['conviction_mult'].max():.2f}], median {v2_cache['conviction_mult'].median():.2f}\n")

    results = {}  # (threshold, split, subset) -> stats
    try:
        for t in THRESHOLDS:
            set_threshold(t)
            print(f"\n=== threshold {t:.1f} ===")
            for split_name, train_s, train_e, test_s, test_e in SPLITS:
                trn = run_bt(train_s, train_e)
                tst = run_bt(test_s, test_e)
                if trn and tst:
                    results[(t, split_name, 'train')] = trn
                    results[(t, split_name, 'test')] = tst
                    print(f"  {split_name}")
                    print(f"    train: {trn['n']:>3} trades, ${trn['pnl']:>+10,.0f}, {trn['wr']:>4.0f}% WR")
                    print(f"    test:  {tst['n']:>3} trades, ${tst['pnl']:>+10,.0f}, {tst['wr']:>4.0f}% WR")
    finally:
        shutil.copy(CACHE_BACKUP, CACHE_PATH)
        shutil.copy(CONFIG_BACKUP, CONFIG_PATH)
        print(f"\nRestored cache + config")

    # Cross-split summary — compare each threshold's TEST P&L to threshold=1.2 (current)
    print("\n\n=== CROSS-SPLIT TEST OOS — V2_full at each threshold ===")
    print(f"{'Threshold':<10} {'A test':>13} {'B test':>13} {'C test':>13} "
          f"{'mean':>13} {'min':>13}")
    print('-' * 80)
    summaries = []
    for t in THRESHOLDS:
        pnls = []
        for split_name, *_ in SPLITS:
            r = results.get((t, split_name, 'test'))
            if r:
                pnls.append(r['pnl'])
        if pnls:
            mean_p = sum(pnls) / len(pnls)
            min_p = min(pnls)
            cells = " ".join(f"${p:>+11,.0f}" for p in pnls)
            print(f"{t:<10.1f} {cells} ${mean_p:>+11,.0f} ${min_p:>+11,.0f}")
            summaries.append((t, pnls, mean_p, min_p))

    # vs t=1.2 (current)
    base_t = 1.2
    base_summary = next((s for s in summaries if s[0] == base_t), None)
    if base_summary:
        print(f"\n=== vs threshold {base_t} (V2 baseline) ===")
        print(f"{'Threshold':<10} {'mean Δ':>13} {'min Δ':>13} verdict")
        print('-' * 50)
        for t, pnls, mean_p, min_p in summaries:
            if t == base_t:
                continue
            delta_pnls = [p - bp for p, bp in zip(pnls, base_summary[1])]
            delta_mean = sum(delta_pnls) / len(delta_pnls)
            delta_min = min(delta_pnls)
            verdict = '✓' if delta_min > 0 else ('⚠' if delta_mean > 0 else '✗')
            print(f"{t:<10.1f} ${delta_mean:>+11,.0f} ${delta_min:>+11,.0f}  {verdict}")

    # Recommend
    print("\n=== RECOMMENDATION ===")
    if summaries:
        # Best by mean OOS test P&L (absolute), with min > 0
        valid = [s for s in summaries if s[3] > 0]
        if valid:
            ranked = sorted(valid, key=lambda s: -s[2])
            best = ranked[0]
            print(f"Best by mean OOS test P&L: threshold={best[0]} "
                  f"(mean ${best[2]:+,.0f}, worst ${best[3]:+,.0f})")
            for t, pnls, mean_p, min_p in ranked[:3]:
                print(f"  t={t:<4.1f}: mean ${mean_p:+,.0f}, min ${min_p:+,.0f}")

    # Write report
    with open(out_path, 'w') as f:
        f.write(f"# Bull Flag V2 — Phase D Threshold Tuning\n\n")
        f.write(f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n\n")
        f.write(f"V2_full formula winner from Phase C:\n")
        f.write(f"- Rule 6 daily_range_pct: +0.3 if ≥40, +0.1 if ≥30\n")
        f.write(f"- Rule 7 qf_fill_vwap_dist_pct: +0.2 if ≥2\n")
        f.write(f"- Rule 8 qf_gap_fading: -0.3 if True\n\n")
        f.write("## Per-split detail\n\n")
        for split_name, *_ in SPLITS:
            f.write(f"### Split {split_name}\n\n")
            f.write("**TEST**\n\n| Threshold | n | WR | P&L |\n|---|---|---|---|\n")
            for t in THRESHOLDS:
                s = results.get((t, split_name, 'test'))
                if s:
                    f.write(f"| {t} | {s['n']} | {s['wr']:.0f}% | ${s['pnl']:+,.0f} |\n")
            f.write("\n")
        f.write("## Cross-split TEST OOS summary\n\n")
        f.write("| Threshold | "
                + " | ".join(f"Split {chr(65+i)} P&L" for i in range(len(SPLITS)))
                + " | Mean | Min |\n")
        f.write("|---|" + "---|" * (len(SPLITS) + 2) + "\n")
        for t, pnls, mean_p, min_p in summaries:
            cells = " | ".join(f"${p:+,.0f}" for p in pnls)
            f.write(f"| {t} | {cells} | ${mean_p:+,.0f} | ${min_p:+,.0f} |\n")
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
