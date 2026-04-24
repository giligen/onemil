#!/usr/bin/env python3
"""Walk-forward study: min_daily_volume threshold sweep.

Background: 3 "monster days" in Jan-Apr 2026 drove 55% of bull-flag gross
wins. 2 of those trades (MVLL, QCLS on 2026-03-06) had avg_vol 263K-328K —
uncomfortably close to the 200K prod minimum. Question: would a tighter
volume filter (400K, 500K, 1M) *improve* net P&L by dropping bad thin
trades, or *hurt* by dropping good ones?

Methodology: 3 chronological train/test splits. For each (split, threshold),
run the Stage-2 filtered BT (cache is volume-agnostic; the filter is applied
in filter_bull_flag_trades via config). A robust winner agrees between
train-best and test-best thresholds AND clears test > current prod.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple

CONFIG_PATH = 'config.yaml'
CONFIG_BACKUP = '/tmp/config_volume_study_orig.yaml'

THRESHOLDS = [200_000, 300_000, 400_000, 500_000, 1_000_000]

SPLITS = [
    ('A: H1\'25 → H2\'25-Apr\'26',  '2025-01-01', '2025-06-30', '2025-07-01', '2026-04-15'),
    ('B: Y2025 → Q1+Apr\'26',       '2025-01-01', '2025-12-31', '2026-01-01', '2026-04-15'),
    ('C: Jan-Sep\'25 → Oct\'25-Apr\'26','2025-01-01', '2025-09-30', '2025-10-01', '2026-04-15'),
]


def set_min_vol(v: int) -> None:
    with open(CONFIG_PATH) as f:
        txt = f.read()
    new = re.sub(
        r'(^|\n)(\s*)min_daily_volume:\s*[\d_]+',
        lambda m: f"{m.group(1)}{m.group(2)}min_daily_volume: {v}",
        txt, count=1,
    )
    with open(CONFIG_PATH, 'w') as f:
        f.write(new)


def run_bt(start: str, end: str) -> Dict[str, float]:
    try:
        out = subprocess.check_output(
            ['python3', 'batch_backtest.py', '--start', start, '--end', end],
            stderr=subprocess.STDOUT, text=True, timeout=120,
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


def main() -> None:
    shutil.copy(CONFIG_PATH, CONFIG_BACKUP)
    results: Dict[Tuple[str, int, str], Dict] = {}

    try:
        for v in THRESHOLDS:
            set_min_vol(v)
            print(f"\n=== min_daily_volume = {v:,} ===")
            for split_name, train_s, train_e, test_s, test_e in SPLITS:
                trn = run_bt(train_s, train_e)
                tst = run_bt(test_s, test_e)
                if trn and tst:
                    results[(split_name, v, 'train')] = trn
                    results[(split_name, v, 'test')] = tst
                    print(f"  {split_name}")
                    print(f"    train: {trn['n']:>3} trades, ${trn['pnl']:>+10,.0f}, {trn['wr']:>4.0f}% WR")
                    print(f"    test:  {tst['n']:>3} trades, ${tst['pnl']:>+10,.0f}, {tst['wr']:>4.0f}% WR")
    finally:
        shutil.copy(CONFIG_BACKUP, CONFIG_PATH)
        print(f"\nRestored config from {CONFIG_BACKUP}")

    # Per-split summary: best threshold on train vs test
    print("\n\n" + "=" * 80)
    print("SUMMARY — best threshold per split, train vs test (ΔP&L vs 200K baseline)")
    print("=" * 80)
    print(f"{'Split':<32} {'Train-best':<22} {'Test-best':<22} Agreement?")
    print('-' * 80)
    for split_name, *_ in SPLITS:
        train_rows = [(v, results[(split_name, v, 'train')]) for v in THRESHOLDS if (split_name, v, 'train') in results]
        test_rows  = [(v, results[(split_name, v, 'test')])  for v in THRESHOLDS if (split_name, v, 'test')  in results]
        if not train_rows or not test_rows:
            continue
        train_best_v, train_best_r = max(train_rows, key=lambda x: x[1]['pnl'])
        test_best_v,  test_best_r  = max(test_rows,  key=lambda x: x[1]['pnl'])
        agree = '✓ SAME' if train_best_v == test_best_v else '✗ DIFFER'
        train_str = f"{train_best_v/1000:.0f}K ${train_best_r['pnl']:+,.0f}"
        test_str  = f"{test_best_v/1000:.0f}K ${test_best_r['pnl']:+,.0f}"
        print(f"{split_name:<32} {train_str:<22} {test_str:<22} {agree}")

    # Cross-split test OOS mean delta vs baseline (200K)
    print("\n" + "=" * 80)
    print("CROSS-SPLIT TEST OOS ΔP&L vs 200K baseline")
    print("=" * 80)
    hdr_split_cells = " ".join(f"{chr(65+i)+'_Δ':>13}" for i in range(len(SPLITS)))
    print(f"{'Threshold':<12} {hdr_split_cells} {'mean':>13} {'min':>13} Verdict")
    print('-' * 90)
    for v in THRESHOLDS:
        if v == 200_000:
            continue
        deltas = []
        for split_name, *_ in SPLITS:
            test = results.get((split_name, v, 'test'))
            base = results.get((split_name, 200_000, 'test'))
            if test and base:
                deltas.append(test['pnl'] - base['pnl'])
        if deltas:
            mean_d = sum(deltas) / len(deltas)
            min_d = min(deltas)
            cells = " ".join(f"${d:>+11,.0f}" for d in deltas)
            verdict = '✓ ROBUST' if min_d > 0 else ('⚠ mixed' if mean_d > 0 else '✗ losing')
            print(f"{v:>10,}  {cells} ${mean_d:>+11,.0f} ${min_d:>+11,.0f}  {verdict}")

    # Raw sweep (all thresholds, all splits, test only)
    print("\n" + "=" * 80)
    print("RAW: test P&L per split per threshold")
    print("=" * 80)
    hdr = f"{'Threshold':<10}"
    for split_name, *_ in SPLITS:
        hdr += f" {split_name[:20]:>22}"
    print(hdr)
    print('-' * len(hdr))
    for v in THRESHOLDS:
        line = f"{v:>8,}  "
        for split_name, *_ in SPLITS:
            test = results.get((split_name, v, 'test'))
            if test:
                line += f" {test['n']:>2}t ${test['pnl']:>+10,.0f} ({test['wr']:>3.0f}%)"
            else:
                line += f" {'—':>22}"
        print(line)


if __name__ == '__main__':
    main()
