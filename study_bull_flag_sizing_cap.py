#!/usr/bin/env python3
"""Bull Flag Sizing Cap — Step 3 walk-forward study.

Tests whether capping conviction_mult at a knee point (e.g. 1.5 or 1.8)
improves OOS results vs current linear sizing. Hypothesis: conv 1.8+ trades
have lower WR than 1.5-1.8 trades but get bigger sizing — over-deploying
on apparent top setups that don't pay for the extra capital.

Methodology:
- Load raw cache (post-look-ahead-fix, post-conviction-filter-bug-fix)
- For each cap variant, RESCALE pnl by (capped_conv / orig_conv) so it
  represents pnl with capped sizing applied
- Run the SAME filter_bull_flag_trades pipeline (with prod max_concurrent,
  daily_loss_limit, regime, etc.) so comparisons are fair
- 3 chronological train/test splits (same as conviction filter study)
- Rank by mean OOS test ΔP&L vs no-cap (V0)

Reads:  data/bull_flag_cache_e50_x30.csv
Calls:  batch_backtest.filter_bull_flag_trades (with prod params via DB)
Writes: analysis_results/bull_flag_sizing_cap_<date>.md
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class Variant:
    name: str
    cap: float
    description: str


VARIANTS: List[Variant] = [
    Variant('V0_no_cap',     2.40, 'No effective cap (baseline)'),
    Variant('V1_cap_2.0',    2.00, 'Cap at 2.0'),
    Variant('V2_cap_1.8',    1.80, 'Cap at 1.8 — knee-point hypothesis'),
    Variant('V3_cap_1.7',    1.70, 'Cap at 1.7'),
    Variant('V4_cap_1.6',    1.60, 'Cap at 1.6'),
    Variant('V5_cap_1.5',    1.50, 'Cap at 1.5 — golden-bucket peak'),
]

SPLITS = [
    ('A: H1\'25 train → H2\'25-Apr\'26 test',
     '2025-01-01', '2025-06-30', '2025-07-01', '2026-04-30'),
    ('B: Y2025 train → Q1+Apr\'26 test',
     '2025-01-01', '2025-12-31', '2026-01-01', '2026-04-30'),
    ('C: Jan-Sep\'25 train → Oct\'25-Apr\'26 test',
     '2025-01-01', '2025-09-30', '2025-10-01', '2026-04-30'),
]

CACHE_PATH = 'data/bull_flag_cache_e50_x30.csv'
TMP_CACHE = '/tmp/bull_flag_cache_capped.csv'
ORIG_CACHE_BACKUP = '/tmp/bull_flag_cache_orig.csv'


def apply_cap(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Return new df with conviction_mult capped at `cap` and pnl/shares rescaled.

    pnl is linear in conv_mult (cache has pnl = base × conv × macd). Rescaling by
    (capped/orig) gives pnl as if the position was sized at the capped conviction.
    """
    out = df.copy()
    orig_conv = out['conviction_mult'].fillna(1.0).astype(float).clip(lower=0.001)
    capped_conv = orig_conv.clip(upper=cap)
    ratio = capped_conv / orig_conv
    out['pnl'] = out['pnl'].astype(float) * ratio
    if 'shares' in out.columns:
        # Keep shares aligned (informational; downstream uses pnl)
        out['shares'] = (out['shares'].fillna(0).astype(float) * ratio).clip(lower=1).astype(int)
    out['conviction_mult'] = capped_conv
    if 'partial_pnl' in out.columns:
        out['partial_pnl'] = out['partial_pnl'].fillna(0).astype(float) * ratio
    return out


def run_bt(start: str, end: str) -> Dict[str, float]:
    """Subprocess BT call. Returns {n, pnl, wr} or None on failure."""
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
    out_path = f"analysis_results/bull_flag_sizing_cap_{datetime.now().strftime('%Y%m%d_%H%M')}.md"

    # Backup original cache once
    shutil.copy(CACHE_PATH, ORIG_CACHE_BACKUP)
    print(f"Backed up cache to {ORIG_CACHE_BACKUP}")

    df = pd.read_csv(CACHE_PATH)
    print(f"Loaded {len(df)} cached trades, conv range [{df['conviction_mult'].min():.2f}, "
          f"{df['conviction_mult'].max():.2f}]\n")

    results = {}  # (variant_name, split_name, subset) -> stats dict
    try:
        for v in VARIANTS:
            # Apply cap and write to cache file the BT will read
            capped = apply_cap(df, v.cap)
            capped.to_csv(CACHE_PATH, index=False)
            print(f"\n=== {v.name} (cap={v.cap}) ===")
            for split_name, ts, te, t2s, t2e in SPLITS:
                trn = run_bt(ts, te)
                tst = run_bt(t2s, t2e)
                if trn and tst:
                    results[(v.name, split_name, 'train')] = trn
                    results[(v.name, split_name, 'test')] = tst
                    print(f"  {split_name}")
                    print(f"    train: {trn['n']:>3} trades, ${trn['pnl']:>+10,.0f}, {trn['wr']:>4.0f}% WR")
                    print(f"    test:  {tst['n']:>3} trades, ${tst['pnl']:>+10,.0f}, {tst['wr']:>4.0f}% WR")
    finally:
        # Restore original cache no matter what
        shutil.copy(ORIG_CACHE_BACKUP, CACHE_PATH)
        print(f"\nRestored original cache from {ORIG_CACHE_BACKUP}")

    # Summary
    print("\n\n=== CROSS-SPLIT TEST OOS ΔP&L vs V0_no_cap ===")
    print(f"{'Variant':<14} {'A test Δ':>13} {'B test Δ':>13} {'C test Δ':>13} "
          f"{'mean':>13} {'min':>13}")
    print('-' * 90)
    summaries = []
    for v in VARIANTS:
        if v.name == 'V0_no_cap':
            continue
        deltas = []
        for split_name, *_ in SPLITS:
            test = results.get((v.name, split_name, 'test'))
            v0 = results.get(('V0_no_cap', split_name, 'test'))
            if test and v0:
                deltas.append(test['pnl'] - v0['pnl'])
        if deltas:
            mean_d = sum(deltas) / len(deltas)
            min_d = min(deltas)
            cells = " ".join(f"${d:>+11,.0f}" for d in deltas)
            print(f"{v.name:<14} {cells} ${mean_d:>+11,.0f} ${min_d:>+11,.0f}")
            summaries.append((v.name, deltas, mean_d, min_d))

    # Write report
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(f"# Bull Flag Sizing Cap — Step 3 Walk-Forward Study\n\n")
        f.write(f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n\n")
        f.write(f"## Variants\n\n| Name | Cap | Description |\n|---|---|---|\n")
        for v in VARIANTS:
            f.write(f"| `{v.name}` | {v.cap} | {v.description} |\n")
        f.write("\n## Per-split detail\n\n")
        for split_name, *_ in SPLITS:
            f.write(f"### Split {split_name}\n\n")
            for subset in ('train', 'test'):
                f.write(f"**{subset.upper()}**\n\n")
                f.write("| Variant | n | WR | P&L | Δ vs V0 |\n|---|---|---|---|---|\n")
                base = results.get(('V0_no_cap', split_name, subset))
                for v in VARIANTS:
                    s = results.get((v.name, split_name, subset))
                    if not s:
                        continue
                    delta = s['pnl'] - base['pnl'] if base else 0
                    delta_str = f"${delta:+,.0f}" if v.name != 'V0_no_cap' else '—'
                    f.write(f"| `{v.name}` | {s['n']} | {s['wr']:.0f}% | "
                            f"${s['pnl']:+,.0f} | {delta_str} |\n")
                f.write("\n")
        f.write("## Cross-split OOS summary\n\n")
        f.write("| Variant | "
                + " | ".join(f"Split {chr(65+i)} Δ$" for i in range(len(SPLITS)))
                + " | Mean Δ | Min Δ | Verdict |\n")
        f.write("|---|" + "---|" * (len(SPLITS) + 3) + "\n")
        for name, deltas, mean_d, min_d in sorted(summaries, key=lambda r: -r[2]):
            verdict = '✓ robust' if min_d > 0 else ('⚠ mixed' if mean_d > 0 else '✗ losing')
            cells = " | ".join(f"${d:+,.0f}" for d in deltas)
            f.write(f"| `{name}` | {cells} | ${mean_d:+,.0f} | ${min_d:+,.0f} | {verdict} |\n")
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
