#!/usr/bin/env python3
"""Phase C+D RE-RUN — V2 with ONLY clean (no look-ahead) new rules.

Rule 6 (daily_range_pct) was found to use the FULL day's range — known only
after market close. Removing it. V2_clean = V0 baseline + Rule 7 (vwap_dist,
PRE-fill, knowable at setup) + Rule 8 (gap_fading, knowable at setup).

Threshold sweep on V2_clean, then pick winner with mean OOS ≥ +$10K AND min > $0.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd

from study_bull_flag_conviction_v2 import apply_variant, Variant, r6_off, r7_simple, r8_penalty

CACHE_PATH = 'data/bull_flag_cache_e50_x30.csv'
CACHE_BACKUP = '/tmp/bull_flag_cache_v2clean_orig.csv'
CONFIG_PATH = 'config.yaml'
CONFIG_BACKUP = '/tmp/config_v2clean_orig.yaml'

V2_CLEAN = Variant(
    'V2_clean', 'V0 + Rule 7 (vwap_dist) + Rule 8 (gap_fading) — no Rule 6 (look-ahead)',
    r6_off, r7_simple, r8_penalty,
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
    out_path = f"analysis_results/bull_flag_v2_clean_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    shutil.copy(CACHE_PATH, CACHE_BACKUP)
    shutil.copy(CONFIG_PATH, CONFIG_BACKUP)

    base_cache = pd.read_csv(CACHE_PATH)
    print(f"Cache: {len(base_cache)} rows\n")

    # Apply V2_clean once
    print("Applying V2_clean (rules 7+8, NO rule 6)...")
    v2_cache = apply_variant(base_cache, V2_CLEAN)
    v2_cache.to_csv(CACHE_PATH, index=False)
    print(f"V2_clean conv_mult range [{v2_cache['conviction_mult'].min():.2f}, "
          f"{v2_cache['conviction_mult'].max():.2f}]\n")

    # Also need V0 baseline — original cache before any variant applied
    v0_results = {}
    v2_results = {}
    try:
        # First, capture V0 baseline (original cache, current threshold 1.2)
        print("=== V0 baseline (no V2 rules) ===")
        # Restore original cache for V0 measurement
        shutil.copy(CACHE_BACKUP, CACHE_PATH)
        set_threshold(1.2)
        for split_name, train_s, train_e, test_s, test_e in SPLITS:
            tst = run_bt(test_s, test_e)
            if tst:
                v0_results[split_name] = tst
                print(f"  {split_name} test: {tst['n']:>3} trades, ${tst['pnl']:>+10,.0f}")

        # Now apply V2_clean and sweep thresholds
        v2_cache.to_csv(CACHE_PATH, index=False)
        for t in THRESHOLDS:
            set_threshold(t)
            print(f"\n=== V2_clean @ threshold {t} ===")
            for split_name, train_s, train_e, test_s, test_e in SPLITS:
                tst = run_bt(test_s, test_e)
                if tst:
                    v2_results[(t, split_name)] = tst
                    base = v0_results.get(split_name)
                    delta = tst['pnl'] - base['pnl'] if base else 0
                    print(f"  {split_name} test: {tst['n']:>3} trades, "
                          f"${tst['pnl']:>+10,.0f} (Δ vs V0: ${delta:>+9,.0f})")
    finally:
        shutil.copy(CACHE_BACKUP, CACHE_PATH)
        shutil.copy(CONFIG_BACKUP, CONFIG_PATH)
        print(f"\nRestored cache + config")

    # Summary
    print("\n\n=== V2_clean SUMMARY — ΔP&L OOS test vs V0 (current PROD) ===")
    print(f"{'Threshold':<10} {'A Δ':>13} {'B Δ':>13} {'C Δ':>13} {'mean Δ':>13} {'min Δ':>13}  verdict")
    print('-' * 100)
    summaries = []
    for t in THRESHOLDS:
        deltas = []
        for split_name, *_ in SPLITS:
            v2 = v2_results.get((t, split_name))
            v0 = v0_results.get(split_name)
            if v2 and v0:
                deltas.append(v2['pnl'] - v0['pnl'])
        if deltas:
            mean_d = sum(deltas) / len(deltas)
            min_d = min(deltas)
            verdict = '✓ ROBUST' if min_d > 0 else ('⚠ mixed' if mean_d > 0 else '✗ losing')
            cells = " ".join(f"${d:>+11,.0f}" for d in deltas)
            print(f"{t:<10.1f} {cells} ${mean_d:>+11,.0f} ${min_d:>+11,.0f}  {verdict}")
            summaries.append((t, deltas, mean_d, min_d, verdict))

    # Decision
    print("\n=== DECISION ===")
    valid = [s for s in summaries if s[2] >= 10_000 and s[3] > 0]
    if valid:
        valid.sort(key=lambda s: -s[2])
        w = valid[0]
        print(f"WINNER: V2_clean @ threshold {w[0]} (mean +${w[2]:,.0f}, min +${w[3]:,.0f})")
    else:
        print(f"NO threshold clears the gate (mean ≥ +$10K AND min > $0).")
        print(f"Honest takeaway: V2 with clean rules doesn't justify shipping.")
        if summaries:
            best = max(summaries, key=lambda s: s[2])
            print(f"Best by mean: t={best[0]}, mean +${best[2]:,.0f}, min +${best[3]:,.0f}")

    # Write report
    with open(out_path, 'w') as f:
        f.write(f"# Bull Flag V2_clean — Phase D RE-RUN (no Rule 6)\n\n")
        f.write(f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n\n")
        f.write(f"**Critical correction:** Rule 6 (daily_range_pct) used WHOLE-day high/low, "
                f"unavailable at signal time. Dropped. V2_clean = V0 + Rule 7 + Rule 8 only.\n\n")
        f.write("## V0 baseline (current PROD)\n\n| Split | n | WR | P&L |\n|---|---|---|---|\n")
        for split_name, *_ in SPLITS:
            v0 = v0_results.get(split_name)
            if v0:
                f.write(f"| {split_name} | {v0['n']} | {v0['wr']:.0f}% | ${v0['pnl']:+,.0f} |\n")
        f.write("\n## V2_clean threshold sweep — TEST OOS Δ vs V0\n\n")
        f.write("| Threshold | A Δ | B Δ | C Δ | Mean Δ | Min Δ | Verdict |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for t, deltas, mean_d, min_d, verdict in summaries:
            cells = " | ".join(f"${d:+,.0f}" for d in deltas)
            f.write(f"| {t} | {cells} | ${mean_d:+,.0f} | ${min_d:+,.0f} | {verdict} |\n")
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
