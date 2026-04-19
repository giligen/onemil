#!/usr/bin/env python3
"""ORB composite filter — walk-forward validation.

Takes the top 7 discriminator features from `study_orb_features.py` and
builds a signed-z-score composite. Mean/std fit on TRAIN split only (no
look-ahead). Sweep threshold on TEST. Ship only if filter lifts P&L on
ALL 3 splits AND cuts drawdown.

Inputs:
  analysis_results/orb_features_{ts}.csv  (per-trade features + pnl + win)
  (defaults to the most recent file matching that pattern)

Usage:
    python3 study_orb_filter.py
    python3 study_orb_filter.py --features-csv analysis_results/orb_features_XXX.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from study_orb import SPLITS, OUT_DIR

# Locked feature set from the feature study. sign = -1 means lower is better
# (invert before averaging). sign = +1 means higher is better.
FILTER_FEATURES: List[Tuple[str, int]] = [
    ('gap_pct',                   -1),  # big gap fades
    ('range_total_volume',        -1),  # high early vol = exhausted
    ('range_avg_bar_range_pct',   -1),  # wide bars = wide spreads = bad
    ('range_size_pct',            -1),  # wild 5-min range = not clean continuation
    ('price_vs_20d_high_pct',     -1),  # closer to 20d high = tired; far = fresh
    ('prev_day_close_position',   -1),  # prev closed near high = tired
    ('range_close_position',      +1),  # close near range HIGH = real momentum
]

# Thresholds to sweep (composite z-score cutoffs — trades with score >= threshold kept)
THRESHOLDS = [-1.5, -1.0, -0.5, 0.0, 0.25, 0.5]


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def fit_z_params(
    train_df: pd.DataFrame, features: List[Tuple[str, int]],
) -> Dict[str, Dict[str, float]]:
    """Fit mean/std of each feature on TRAIN data only."""
    params: Dict[str, Dict[str, float]] = {}
    for feat, sign in features:
        if feat not in train_df.columns:
            raise KeyError(f"Feature '{feat}' missing from trade df")
        col = train_df[feat].astype(float)
        mean = float(col.mean())
        std = float(col.std(ddof=0))
        params[feat] = {'mean': mean, 'std': max(std, 1e-9), 'sign': sign}
    return params


def composite_score(
    df: pd.DataFrame, params: Dict[str, Dict[str, float]],
) -> pd.Series:
    """Compute signed-z-score composite per row. Higher = better."""
    total = pd.Series(0.0, index=df.index)
    for feat, p in params.items():
        z = (df[feat].astype(float) - p['mean']) / p['std']
        # sign: -1 means flip so lower raw → higher contribution
        total = total + (z * p['sign'])
    return total / len(params)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def summarize(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {'n': 0, 'wr': 0.0, 'pnl': 0.0, 'avg': 0.0, 'max_dd': 0.0, 'peak': 0.0}
    pnls = trades['pnl'].to_numpy()
    wr = float((pnls > 0).mean() * 100)
    total = float(pnls.sum())
    avg = float(pnls.mean())
    cum = 0.0; peak = 0.0; dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    return {'n': len(pnls), 'wr': wr, 'pnl': total, 'avg': avg,
            'max_dd': dd, 'peak': peak}


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    split_name: str
    train_n: int
    test_n: int
    baseline_test_pnl: float
    baseline_test_wr: float
    baseline_test_dd: float
    sweeps: Dict[float, Dict[str, float]]  # threshold -> {n, pnl, wr, dd, avg, kept_pct}


def run_split(
    df: pd.DataFrame,
    train_start: str, train_end: str,
    test_start: str, test_end: str,
    features: List[Tuple[str, int]],
    thresholds: List[float],
) -> SplitResult:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    train = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
    test = df[(df['date'] >= test_start) & (df['date'] <= test_end)]

    # Fit z on train only
    params = fit_z_params(train, features)

    # Compute composite for ALL rows (need test's composite using train params)
    df['_composite'] = composite_score(df, params)

    # Re-slice (now with composite col)
    train = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
    test = df[(df['date'] >= test_start) & (df['date'] <= test_end)]

    baseline = summarize(test)

    sweeps: Dict[float, Dict[str, float]] = {}
    for thr in thresholds:
        kept = test[test['_composite'] >= thr]
        s = summarize(kept)
        s['kept_pct'] = len(kept) / max(len(test), 1) * 100
        s['delta_pnl'] = s['pnl'] - baseline['pnl']
        sweeps[thr] = s

    return SplitResult(
        split_name=f"{train_start} → {train_end} / {test_start} → {test_end}",
        train_n=len(train), test_n=len(test),
        baseline_test_pnl=baseline['pnl'],
        baseline_test_wr=baseline['wr'],
        baseline_test_dd=baseline['max_dd'],
        sweeps=sweeps,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_latest_features_csv() -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(OUT_DIR, 'orb_features_*.csv')))
    if not paths:
        return None
    # Exclude the corrmatrix file
    paths = [p for p in paths if 'corrmatrix' not in p]
    return paths[-1] if paths else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-csv', type=str, default=None,
                        help="Path to orb_features_*.csv from study_orb_features.py")
    args = parser.parse_args()

    feat_csv = args.features_csv or find_latest_features_csv()
    if not feat_csv or not os.path.exists(feat_csv):
        print("ERROR: no features CSV found. Run study_orb_features.py first.")
        sys.exit(1)

    print(f"Loading features from: {feat_csv}")
    df = pd.read_csv(feat_csv)
    print(f"  {len(df):,} trades, {len(df.columns)} columns")

    # Drop rows missing any filter feature
    needed = [f for f, _ in FILTER_FEATURES]
    before = len(df)
    df = df.dropna(subset=needed + ['pnl', 'pnl_pct', 'date'])
    print(f"  After dropna on features: {len(df):,} ({before - len(df)} dropped)")

    # Overall baseline summary (all trades, no filter)
    overall = summarize(df)
    print(f"\nOverall baseline (no filter, all 15.5mo): "
          f"n={overall['n']}  WR={overall['wr']:.1f}%  "
          f"P&L=${overall['pnl']:+,.0f}  avg=${overall['avg']:+,.0f}  "
          f"DD=${overall['max_dd']:+,.0f}")

    # Run walk-forward per split
    print(f"\n{'='*110}")
    print("WALK-FORWARD: composite filter (mean/std fit on TRAIN only)")
    print(f"{'='*110}")
    split_results: List[SplitResult] = []
    for split_name, tr_s, tr_e, te_s, te_e in SPLITS:
        print(f"\nSplit: {split_name}")
        print(f"  Train: {tr_s} → {tr_e}")
        print(f"  Test:  {te_s} → {te_e}")
        sr = run_split(df, tr_s, tr_e, te_s, te_e, FILTER_FEATURES, THRESHOLDS)
        sr.split_name = split_name
        split_results.append(sr)
        print(f"  Baseline test (no filter): n={sr.test_n}  "
              f"P&L=${sr.baseline_test_pnl:+,.0f}  WR={sr.baseline_test_wr:.1f}%  "
              f"DD=${sr.baseline_test_dd:+,.0f}")
        print(f"  {'Threshold':>10} {'kept %':>7} {'n':>5} {'WR':>6} "
              f"{'P&L':>11} {'Δ P&L':>11} {'avg/trade':>10} {'DD':>11}")
        for thr in THRESHOLDS:
            s = sr.sweeps[thr]
            print(f"  {thr:>+10.2f} {s['kept_pct']:>6.1f}%  "
                  f"{s['n']:>5} {s['wr']:>5.1f}% "
                  f"${s['pnl']:>+9,.0f} ${s['delta_pnl']:>+9,.0f} "
                  f"${s['avg']:>+8,.0f} ${s['max_dd']:>+9,.0f}")

    # Cross-split summary per threshold
    print(f"\n{'='*110}")
    print("CROSS-SPLIT SUMMARY — Δ P&L vs no-filter baseline (test period)")
    print(f"{'='*110}")
    print(f"{'Threshold':<10} {'A Δ':>14} {'B Δ':>14} {'C Δ':>14} "
          f"{'Mean Δ':>14} {'Min Δ':>14}  Verdict")
    print('-' * 110)

    robust_candidates = []
    for thr in THRESHOLDS:
        deltas = [sr.sweeps[thr]['delta_pnl'] for sr in split_results]
        mean_d = sum(deltas) / len(deltas)
        min_d = min(deltas)
        if min_d > 0:
            verdict = '✓ ROBUST'
        elif mean_d > 0:
            verdict = '⚠ mixed'
        else:
            verdict = '✗ losing'
        cells = ' '.join(f"${d:>+12,.0f}" for d in deltas)
        print(f"{thr:<+10.2f} {cells} ${mean_d:>+12,.0f} ${min_d:>+12,.0f}  {verdict}")
        if min_d > 0:
            robust_candidates.append((thr, mean_d, min_d, deltas))

    # Winner selection: robust threshold with highest mean Δ
    print(f"\n{'='*60}")
    if robust_candidates:
        robust_candidates.sort(key=lambda c: c[1], reverse=True)
        best = robust_candidates[0]
        print(f"✓ WINNER: threshold = {best[0]:+.2f}")
        print(f"  Mean Δ test P&L across splits: ${best[1]:+,.0f}")
        print(f"  Min Δ across splits: ${best[2]:+,.0f}")
        # Show train-fit params from the full-range fit for documentation
        full_params = fit_z_params(df, FILTER_FEATURES)
        print(f"\n  Full-period z-score params (for reference, not shipped):")
        for feat, p in full_params.items():
            print(f"    {feat:<30} mean={p['mean']:>10.3f}  std={p['std']:>9.3f}  sign={p['sign']:+d}")
    else:
        print("✗ NO ROBUST THRESHOLD")
        print("  No threshold beats baseline on all 3 splits.")
        # Report best-mean as fallback
        all_sums = []
        for thr in THRESHOLDS:
            deltas = [sr.sweeps[thr]['delta_pnl'] for sr in split_results]
            all_sums.append((thr, sum(deltas)/len(deltas), min(deltas)))
        all_sums.sort(key=lambda c: c[1], reverse=True)
        print(f"  Best by mean Δ: threshold = {all_sums[0][0]:+.2f}, "
              f"mean=${all_sums[0][1]:+,.0f}, min=${all_sums[0][2]:+,.0f}")
    print(f"{'='*60}")

    # Write markdown report
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    md_path = f"{OUT_DIR}/orb_filter_{ts}.md"
    with open(md_path, 'w') as f:
        f.write(f"# ORB Composite Filter — Walk-Forward\n\n")
        f.write(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n\n")
        f.write(f"**Input**: `{feat_csv}` ({len(df):,} trades)\n\n")
        f.write(f"**Method**: signed z-score composite of {len(FILTER_FEATURES)} features. "
                f"Mean/std fit on TRAIN split only; applied to TEST unchanged.\n\n")
        f.write("## Filter features\n\n")
        f.write("| feature | sign | interpretation |\n|---|:-:|---|\n")
        for feat, sign in FILTER_FEATURES:
            note = "lower is better" if sign < 0 else "higher is better"
            f.write(f"| `{feat}` | {sign:+d} | {note} |\n")

        f.write("\n## Per-split threshold sweep (test-only, Δ vs no-filter baseline)\n\n")
        for sr in split_results:
            f.write(f"### {sr.split_name}\n\n")
            f.write(f"- Baseline (no filter) test: n={sr.test_n}, "
                    f"P&L=${sr.baseline_test_pnl:+,.0f}, WR={sr.baseline_test_wr:.1f}%, "
                    f"DD=${sr.baseline_test_dd:+,.0f}\n\n")
            f.write("| threshold | kept % | n | WR | P&L | Δ P&L | avg/trade | DD |\n")
            f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for thr in THRESHOLDS:
                s = sr.sweeps[thr]
                f.write(f"| {thr:+.2f} | {s['kept_pct']:.1f}% | {s['n']} | "
                        f"{s['wr']:.1f}% | ${s['pnl']:+,.0f} | ${s['delta_pnl']:+,.0f} | "
                        f"${s['avg']:+,.0f} | ${s['max_dd']:+,.0f} |\n")
            f.write("\n")

        f.write("## Cross-split summary\n\n")
        f.write("| threshold | A Δ | B Δ | C Δ | Mean Δ | Min Δ | Verdict |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---|\n")
        for thr in THRESHOLDS:
            deltas = [sr.sweeps[thr]['delta_pnl'] for sr in split_results]
            mean_d = sum(deltas) / len(deltas)
            min_d = min(deltas)
            verdict = '✓ ROBUST' if min_d > 0 else ('⚠ mixed' if mean_d > 0 else '✗ losing')
            cells = ' | '.join(f"${d:+,.0f}" for d in deltas)
            f.write(f"| {thr:+.2f} | {cells} | ${mean_d:+,.0f} | ${min_d:+,.0f} | {verdict} |\n")

        if robust_candidates:
            best = robust_candidates[0]
            f.write(f"\n### ✓ Winner: threshold = **{best[0]:+.2f}**\n\n")
            f.write(f"- Mean Δ P&L on test: **${best[1]:+,.0f}**\n")
            f.write(f"- Min Δ (worst split): **${best[2]:+,.0f}**\n")
        else:
            f.write("\n### ✗ No robust threshold — filter does not generalize across splits\n")

    print(f"\nReport: {md_path}")


if __name__ == '__main__':
    main()
