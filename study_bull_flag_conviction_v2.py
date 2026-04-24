#!/usr/bin/env python3
"""Phase C — Walk-forward V2 conviction variant study.

Tests V2 candidate formulas (V0 baseline + 3 new rules) against current PROD
(V0_baseline = 5 rules + 1.2 threshold) under FULL prod constraints (regime,
max_concurrent=3, daily_loss_limit, etc.).

Methodology:
- For each variant: compute v2_conv per trade in pandas using existing per-rule
  contributions from the rich cache + new rule contribs from qf_* columns
- Rescale pnl by (v2_conv / v1_conv) so it represents pnl with new sizing
- Write a temporary cache with v2_conv values replacing conviction_mult
- Run batch_backtest subprocess (uses modified cache, applies conviction filter
  + all other prod filters)
- Capture train/test results per split

Decision gate: mean OOS test ΔP&L vs V0_baseline ≥ +$10K AND no negative split.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Tuple

import pandas as pd
import numpy as np


CACHE_PATH = 'data/bull_flag_cache_e50_x30.csv'
CACHE_BACKUP = '/tmp/bull_flag_cache_phase_c_orig.csv'

CONVICTION_MIN = 0.25
CONVICTION_MAX = 3.0


@dataclass
class Variant:
    name: str
    description: str
    rule6_fn: Callable  # (daily_range_pct) -> contribution
    rule7_fn: Callable  # (qf_fill_vwap_dist_pct) -> contribution
    rule8_fn: Callable  # (qf_gap_fading) -> contribution


# Rule 6 — daily_range_pct
def r6_off(_): return 0.0
def r6_two_tier(dr):
    if pd.isna(dr): return 0.0
    if dr >= 40: return 0.3
    if dr >= 30: return 0.1
    return 0.0


# Rule 7 — qf_fill_vwap_dist_pct
def r7_off(_): return 0.0
def r7_simple(vd):
    if pd.isna(vd): return 0.0
    if vd >= 2: return 0.2
    return 0.0


# Rule 8 — qf_gap_fading
def r8_off(_): return 0.0
def r8_penalty(gf):
    if pd.isna(gf): return 0.0
    # CSV may store as "True"/"False"/bool/int
    if isinstance(gf, str):
        return -0.3 if gf.lower() in ('true', '1') else 0.0
    return -0.3 if bool(gf) else 0.0


VARIANTS: List[Variant] = [
    Variant('V0_baseline',  'Current 5-rule + 1.2 threshold (anchor)',
            r6_off, r7_off, r8_off),
    Variant('V2a_range',    '+ Rule 6 (daily_range >= 40 → +0.3, >= 30 → +0.1)',
            r6_two_tier, r7_off, r8_off),
    Variant('V2b_vwap',     '+ Rule 7 (vwap_dist >= 2 → +0.2)',
            r6_off, r7_simple, r8_off),
    Variant('V2c_gap',      '+ Rule 8 (gap_fading → -0.3)',
            r6_off, r7_off, r8_penalty),
    Variant('V2_full',      '+ All 3 new rules',
            r6_two_tier, r7_simple, r8_penalty),
]


SPLITS = [
    ('A: H1\'25 → H2\'25-Apr\'26',
     '2025-01-01', '2025-06-30', '2025-07-01', '2026-04-30'),
    ('B: Y2025 → Q1+Apr\'26',
     '2025-01-01', '2025-12-31', '2026-01-01', '2026-04-30'),
    ('C: Jan-Sep\'25 → Oct\'25-Apr\'26',
     '2025-01-01', '2025-09-30', '2025-10-01', '2026-04-30'),
]


def apply_variant(cache_df: pd.DataFrame, v: Variant) -> pd.DataFrame:
    """Recompute conviction_mult per row using v's rules + existing v1 contribs.

    pnl is rescaled by (v2 / v1) ratio since cache pnl is baked at v1 sizing.
    Defensive against NaN inputs (some pre-Phase-A rows have NaN in conv_*).
    """
    out = cache_df.copy()
    # New rule contributions (zero for rows where the input is missing).
    # Rule 7 uses qf_vwap_dist_pct (PRE-fill) — PROD-available at conviction
    # time. qf_fill_vwap_dist_pct is computed POST-fill and not available in PROD.
    r6 = out['daily_range_pct'].apply(v.rule6_fn).fillna(0)
    r7 = out['qf_vwap_dist_pct'].apply(v.rule7_fn).fillna(0)
    r8 = out['qf_gap_fading'].apply(v.rule8_fn).fillna(0)
    # v1 contribs (zero for missing rule columns)
    v1_contribs = (
        out['conv_pole_gain'].fillna(0).astype(float)
        + out['conv_flag_tightness'].fillna(0).astype(float)
        + out['conv_vol_ratio'].fillna(0).astype(float)
        + out['conv_spy_regime'].fillna(0).astype(float)
        + out['conv_retracement'].fillna(0).astype(float)
    )
    # If raw_score known, prefer 1.0+contribs; otherwise fall back to conviction_mult.
    has_breakdown = out['conv_raw_score'].notna()
    v1_raw = pd.Series(
        np.where(has_breakdown,
                 (1.0 + v1_contribs).values,
                 out['conviction_mult'].fillna(1.0).astype(float).values),
        index=out.index,
    )
    v1 = v1_raw.clip(lower=CONVICTION_MIN, upper=CONVICTION_MAX)
    v2_raw = v1_raw + r6 + r7 + r8
    v2 = v2_raw.clip(lower=CONVICTION_MIN, upper=CONVICTION_MAX)
    # Scale pnl. Avoid div-by-zero, NaN, and inf.
    ratio = (v2 / v1.clip(lower=0.001)).fillna(1.0).replace([np.inf, -np.inf], 1.0)
    out['pnl'] = out['pnl'].astype(float) * ratio
    if 'shares' in out.columns:
        scaled_shares = (out['shares'].fillna(0).astype(float) * ratio).fillna(1.0)
        out['shares'] = scaled_shares.clip(lower=1).round().astype(int)
    if 'partial_pnl' in out.columns:
        out['partial_pnl'] = out['partial_pnl'].fillna(0).astype(float) * ratio
    out['conviction_mult'] = v2
    return out


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
    out_path = f"analysis_results/bull_flag_conviction_v2_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    shutil.copy(CACHE_PATH, CACHE_BACKUP)
    print(f"Backed up cache to {CACHE_BACKUP}")

    base_cache = pd.read_csv(CACHE_PATH)
    print(f"Cache: {len(base_cache)} rows\n")

    results = {}
    try:
        for v in VARIANTS:
            print(f"\n=== {v.name} ({v.description}) ===")
            modified = apply_variant(base_cache, v)
            modified.to_csv(CACHE_PATH, index=False)
            for split_name, train_s, train_e, test_s, test_e in SPLITS:
                trn = run_bt(train_s, train_e)
                tst = run_bt(test_s, test_e)
                if trn and tst:
                    results[(v.name, split_name, 'train')] = trn
                    results[(v.name, split_name, 'test')] = tst
                    print(f"  {split_name}")
                    print(f"    train: {trn['n']:>3} trades, ${trn['pnl']:>+10,.0f}, {trn['wr']:>4.0f}% WR")
                    print(f"    test:  {tst['n']:>3} trades, ${tst['pnl']:>+10,.0f}, {tst['wr']:>4.0f}% WR")
    finally:
        shutil.copy(CACHE_BACKUP, CACHE_PATH)
        print(f"\nRestored original cache")

    # Cross-split summary
    print("\n\n=== CROSS-SPLIT TEST OOS ΔP&L vs V0_baseline ===")
    print(f"{'Variant':<14} {'A test Δ':>13} {'B test Δ':>13} {'C test Δ':>13} "
          f"{'mean':>13} {'min':>13} verdict")
    print('-' * 100)
    summaries = []
    for v in VARIANTS:
        if v.name == 'V0_baseline':
            continue
        deltas = []
        for split_name, *_ in SPLITS:
            test = results.get((v.name, split_name, 'test'))
            base = results.get(('V0_baseline', split_name, 'test'))
            if test and base:
                deltas.append(test['pnl'] - base['pnl'])
        if deltas:
            mean_d = sum(deltas) / len(deltas)
            min_d = min(deltas)
            cells = " ".join(f"${d:>+11,.0f}" for d in deltas)
            verdict = '✓ ROBUST' if min_d > 0 else (
                '⚠ mixed' if mean_d > 0 else '✗ losing')
            print(f"{v.name:<14} {cells} ${mean_d:>+11,.0f} ${min_d:>+11,.0f}  {verdict}")
            summaries.append((v.name, deltas, mean_d, min_d, verdict))

    # Decision gate
    print("\n=== DECISION GATE ===")
    winners = [s for s in summaries if s[2] >= 10_000 and s[3] > 0]
    if winners:
        winners.sort(key=lambda s: -s[2])
        w = winners[0]
        print(f"WINNER: {w[0]} (mean +${w[2]:,.0f}, worst +${w[3]:,.0f})")
        print(f"  → Proceed to Phase D (threshold tuning) with this formula.")
    else:
        print(f"NO VARIANT clears the gate (mean ≥ +$10K AND min > $0).")
        print(f"  → Report findings, don't ship.")

    # Write report
    with open(out_path, 'w') as f:
        f.write(f"# Bull Flag Conviction V2 — Phase C Walk-Forward Study\n\n")
        f.write(f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n\n")
        f.write(f"**Cache:** `{CACHE_PATH}` ({len(base_cache)} rows)\n")
        f.write(f"**Splits:** {len(SPLITS)}, **Variants:** {len(VARIANTS)}\n\n")
        f.write("## Variants\n\n| Name | Description |\n|---|---|\n")
        for v in VARIANTS:
            f.write(f"| `{v.name}` | {v.description} |\n")
        f.write("\n## Per-split detail\n\n")
        for split_name, train_s, train_e, test_s, test_e in SPLITS:
            f.write(f"### Split {split_name}\n\n")
            for subset in ('train', 'test'):
                f.write(f"**{subset.upper()}**\n\n")
                f.write("| Variant | n | WR | P&L | Δ vs V0 |\n|---|---|---|---|---|\n")
                base_subset = results.get(('V0_baseline', split_name, subset))
                for v in VARIANTS:
                    s = results.get((v.name, split_name, subset))
                    if not s:
                        continue
                    delta = s['pnl'] - base_subset['pnl'] if base_subset else 0
                    delta_str = f"${delta:+,.0f}" if v.name != 'V0_baseline' else '—'
                    f.write(f"| `{v.name}` | {s['n']} | {s['wr']:.0f}% | "
                            f"${s['pnl']:+,.0f} | {delta_str} |\n")
                f.write("\n")
        f.write("## Cross-split TEST OOS summary\n\n")
        f.write("| Variant | "
                + " | ".join(f"Split {chr(65+i)} Δ" for i in range(len(SPLITS)))
                + " | Mean Δ | Min Δ | Verdict |\n")
        f.write("|---|" + "---|" * (len(SPLITS) + 3) + "\n")
        for name, deltas, mean_d, min_d, verdict in sorted(summaries, key=lambda r: -r[2]):
            cells = " | ".join(f"${d:+,.0f}" for d in deltas)
            f.write(f"| `{name}` | {cells} | ${mean_d:+,.0f} | ${min_d:+,.0f} | {verdict} |\n")
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
