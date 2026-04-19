#!/usr/bin/env python3
"""ORB conviction-based sizing study.

Builds on `study_orb_filter.py`:
  1. Apply the +0.00 composite-z filter (drop bottom ~35%).
  2. On surviving trades, bucket by z-score quintile.
  3. Examine avg/trade per bucket to learn WHICH buckets deserve up-size.
  4. Walk-forward validate two sizing schemes:
       - QUINTILE_UNIFORM: all kept trades at 1x (no sizing, just filter)
       - QUINTILE_TIERED:  Q5=1.5x, Q4=1.25x, Q3=1.0x, Q2=0.9x, Q1=0.75x (tiers
         discovered from the per-quintile avg signal, not guessed)

Quintile buckets are fit on TRAIN z-scores, cutoffs applied to TEST. No
look-ahead: z-score params and quintile cutoffs both come from TRAIN only.

Usage:
    python3 study_orb_sizing.py
    python3 study_orb_sizing.py --features-csv analysis_results/orb_features_XXX.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from study_orb import SPLITS, OUT_DIR
from study_orb_filter import (
    FILTER_FEATURES, fit_z_params, composite_score, summarize,
)

# Filter threshold (winner from filter study)
FILTER_THRESHOLD = 0.0

# Sizing schemes to test
#   key = name of scheme
#   value = list of (quintile_label, multiplier)
SIZING_SCHEMES: Dict[str, List[Tuple[str, float]]] = {
    'uniform_1x':     [('Q1', 1.0), ('Q2', 1.0), ('Q3', 1.0), ('Q4', 1.0), ('Q5', 1.0)],
    'tiered_mild':    [('Q1', 0.75), ('Q2', 0.9), ('Q3', 1.0), ('Q4', 1.25), ('Q5', 1.5)],
    'tiered_aggr':    [('Q1', 0.5),  ('Q2', 0.75),('Q3', 1.0), ('Q4', 1.5),  ('Q5', 2.0)],
    'top_heavy':      [('Q1', 1.0),  ('Q2', 1.0), ('Q3', 1.0), ('Q4', 1.25), ('Q5', 1.5)],
    'bottom_light':   [('Q1', 0.5),  ('Q2', 0.75),('Q3', 1.0), ('Q4', 1.0),  ('Q5', 1.0)],
    # Q4-peaked: Q4 is empirically the best quintile (not Q5). Extreme z-scores
    # may flag outlier setups that behave differently. Let's test peaking at Q4.
    'q4_peaked':      [('Q1', 0.5),  ('Q2', 0.75),('Q3', 1.0), ('Q4', 2.0),  ('Q5', 1.25)],
    'q4_peaked_aggr': [('Q1', 0.25), ('Q2', 0.5), ('Q3', 1.0), ('Q4', 2.5),  ('Q5', 1.25)],
}

# Adaptive scheme: multiplier per quintile fit from TRAIN per-quintile avg.
# mult_Q = clip(train_avg_Q / train_overall_avg, 0.25, 3.0).
# This is the rigorous no-hand-picking variant.
ADAPTIVE_MULT_MIN = 0.25
ADAPTIVE_MULT_MAX = 3.0


def fit_quintile_cutoffs(train_scores: pd.Series) -> List[float]:
    """Return the 4 cutoffs that split train scores into 5 equal-size buckets.
    Cutoffs ascending. Apply to test scores with pd.cut / searchsorted."""
    quantiles = train_scores.quantile([0.2, 0.4, 0.6, 0.8]).tolist()
    return quantiles


def assign_quintile(scores: pd.Series, cutoffs: List[float]) -> pd.Series:
    """Assign each score to Q1..Q5 based on fitted cutoffs."""
    # Q1 = below cutoffs[0]; Q5 = above cutoffs[3]
    def _q(x: float) -> str:
        for i, c in enumerate(cutoffs):
            if x < c:
                return f"Q{i+1}"
        return "Q5"
    return scores.apply(_q)


def apply_sizing(pnl: float, quintile: str, scheme: List[Tuple[str, float]]) -> float:
    """Scale pnl by the multiplier for this quintile."""
    for ql, mult in scheme:
        if ql == quintile:
            return pnl * mult
    return pnl  # shouldn't happen


@dataclass
class SplitSizingResult:
    split_name: str
    test_n: int
    kept_n: int
    quintile_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)  # per-Q stats on TEST (unit sizing)
    scheme_pnl: Dict[str, Dict[str, float]] = field(default_factory=dict)      # per-scheme P&L summary on kept test
    adaptive_scheme: List[Tuple[str, float]] = field(default_factory=list)     # adaptive multipliers fit on TRAIN
    train_quintile_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)  # per-Q stats on TRAIN kept (diagnostic)


def run_split(
    df: pd.DataFrame,
    train_start: str, train_end: str,
    test_start: str, test_end: str,
) -> SplitSizingResult:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    train = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
    test = df[(df['date'] >= test_start) & (df['date'] <= test_end)]

    # Fit z-score params on TRAIN
    params = fit_z_params(train, FILTER_FEATURES)

    # Compute composite score for all rows
    df['_composite'] = composite_score(df, params)
    train = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
    test = df[(df['date'] >= test_start) & (df['date'] <= test_end)]

    # Filter: drop below threshold (based on TRAIN params → applied to TEST)
    test_kept = test[test['_composite'] >= FILTER_THRESHOLD].copy()

    # Fit quintile cutoffs on TRAIN KEPT (same filter logic applied to train)
    train_kept = train[train['_composite'] >= FILTER_THRESHOLD].copy()
    cutoffs = fit_quintile_cutoffs(train_kept['_composite'])

    # Assign quintile to BOTH train kept (for adaptive fit) and test kept
    train_kept['_quintile'] = assign_quintile(train_kept['_composite'], cutoffs)
    test_kept['_quintile'] = assign_quintile(test_kept['_composite'], cutoffs)

    # Fit adaptive scheme from TRAIN per-quintile avg
    train_overall_avg = float(train_kept['pnl'].mean()) if len(train_kept) else 1.0
    if train_overall_avg <= 0:
        # Degenerate: fall back to uniform
        adaptive_scheme = [(q, 1.0) for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
    else:
        adaptive_scheme = []
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            q_sub = train_kept[train_kept['_quintile'] == q]
            if len(q_sub) == 0:
                adaptive_scheme.append((q, 1.0))
                continue
            ratio = float(q_sub['pnl'].mean()) / train_overall_avg
            ratio = max(ADAPTIVE_MULT_MIN, min(ADAPTIVE_MULT_MAX, ratio))
            adaptive_scheme.append((q, ratio))

    # Add the adaptive scheme to the schemes dict for this split
    all_schemes = dict(SIZING_SCHEMES)
    all_schemes['adaptive'] = adaptive_scheme

    sr = SplitSizingResult(
        split_name=f"{train_start} → {train_end} / {test_start} → {test_end}",
        test_n=len(test), kept_n=len(test_kept),
        adaptive_scheme=adaptive_scheme,
    )
    # Per-quintile stats on TRAIN kept (for adaptive-scheme diagnostics)
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        tsub = train_kept[train_kept['_quintile'] == q]
        sr.train_quintile_stats[q] = summarize(tsub)

    # Per-quintile stats on TEST kept (unit sizing)
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        sub = test_kept[test_kept['_quintile'] == q]
        stats = summarize(sub)
        stats['cutoff_lo'] = cutoffs[int(q[1:]) - 2] if q != 'Q1' else float('-inf')
        stats['cutoff_hi'] = cutoffs[int(q[1:]) - 1] if q != 'Q5' else float('+inf')
        sr.quintile_stats[q] = stats

    # Store the fitted adaptive scheme on result for reporting
    sr_adaptive_scheme = adaptive_scheme  # noqa
    # Per-scheme total P&L on kept test
    for scheme_name, scheme in all_schemes.items():
        sized_pnl = test_kept.apply(
            lambda r: apply_sizing(r['pnl'], r['_quintile'], scheme), axis=1,
        )
        # Reconstruct running DD
        pnls = sized_pnl.to_numpy()
        cum, peak, dd = 0.0, 0.0, 0.0
        for p in pnls:
            cum += p
            peak = max(peak, cum)
            dd = min(dd, cum - peak)
        wr = float((pnls > 0).mean() * 100) if len(pnls) else 0.0
        sr.scheme_pnl[scheme_name] = {
            'n': len(pnls), 'pnl': float(sized_pnl.sum()),
            'avg': float(sized_pnl.mean()) if len(pnls) else 0.0,
            'wr': wr, 'max_dd': dd,
        }

    return sr


def find_latest_features_csv() -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(OUT_DIR, 'orb_features_*.csv')))
    paths = [p for p in paths if 'corrmatrix' not in p]
    return paths[-1] if paths else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-csv', type=str, default=None)
    args = parser.parse_args()

    feat_csv = args.features_csv or find_latest_features_csv()
    if not feat_csv or not os.path.exists(feat_csv):
        print("ERROR: no features CSV found. Run study_orb_features.py first.")
        sys.exit(1)

    print(f"Loading features from: {feat_csv}")
    df = pd.read_csv(feat_csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'pnl_pct', 'date'])
    print(f"  {len(df):,} trades after dropna")

    # Run walk-forward
    print(f"\n{'='*110}")
    print(f"WALK-FORWARD SIZING STUDY — filter threshold +{FILTER_THRESHOLD:.2f}, "
          f"quintile buckets fit on TRAIN")
    print(f"{'='*110}")
    split_results: List[SplitSizingResult] = []
    for split_name, tr_s, tr_e, te_s, te_e in SPLITS:
        print(f"\nSplit: {split_name}")
        sr = run_split(df, tr_s, tr_e, te_s, te_e)
        sr.split_name = split_name
        split_results.append(sr)
        print(f"  Test N={sr.test_n}, Kept N={sr.kept_n} ({100*sr.kept_n/max(sr.test_n,1):.1f}%)")
        print(f"  Per-quintile (TEST kept, unit sizing):")
        print(f"    {'Q':>3} {'n':>5} {'WR':>6} {'avg/tr':>10} {'P&L':>11} {'DD':>11} {'z-range':>18}")
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            s = sr.quintile_stats[q]
            lo = s['cutoff_lo']
            hi = s['cutoff_hi']
            lo_str = f"{lo:+.2f}" if np.isfinite(lo) else "  -∞"
            hi_str = f"{hi:+.2f}" if np.isfinite(hi) else "  +∞"
            print(f"    {q:>3} {s['n']:>5} {s['wr']:>5.1f}% "
                  f"${s['avg']:>+8,.0f} ${s['pnl']:>+9,.0f} ${s['max_dd']:>+9,.0f}  "
                  f"[{lo_str}, {hi_str}]")

        print(f"  TRAIN per-quintile avg/trade (used to fit adaptive mults):")
        train_avgs = [f"{q}=${sr.train_quintile_stats[q]['avg']:+,.0f}"
                      for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
        print(f"    {'  '.join(train_avgs)}")
        print(f"  Adaptive multipliers: " + "  ".join(
            f"{q}={m:.2f}x" for q, m in sr.adaptive_scheme))
        print(f"  Sizing scheme P&L (TEST kept):")
        print(f"    {'scheme':<16} {'n':>5} {'WR':>6} {'P&L':>11} {'avg/tr':>10} {'DD':>11}")
        scheme_names = list(SIZING_SCHEMES.keys()) + ['adaptive']
        for scheme_name in scheme_names:
            s = sr.scheme_pnl[scheme_name]
            print(f"    {scheme_name:<16} {s['n']:>5} {s['wr']:>5.1f}% "
                  f"${s['pnl']:>+9,.0f} ${s['avg']:>+8,.0f} ${s['max_dd']:>+9,.0f}")

    # Cross-split summary per scheme
    print(f"\n{'='*110}")
    print("CROSS-SPLIT SUMMARY — total P&L per sizing scheme")
    print(f"{'='*110}")
    print(f"{'Scheme':<16} {'A P&L':>14} {'B P&L':>14} {'C P&L':>14} "
          f"{'Sum':>14} {'Min':>14}  Verdict")
    print('-' * 110)

    best_scheme = None
    best_sum = -1e18
    best_min = -1e18
    scheme_names = list(SIZING_SCHEMES.keys()) + ['adaptive']
    for scheme_name in scheme_names:
        pnls = [sr.scheme_pnl[scheme_name]['pnl'] for sr in split_results]
        total = sum(pnls)
        min_p = min(pnls)
        verdict = '✓ all+' if min_p > 0 else '⚠ split loss'
        cells = ' '.join(f"${p:>+12,.0f}" for p in pnls)
        print(f"{scheme_name:<16} {cells} ${total:>+12,.0f} ${min_p:>+12,.0f}  {verdict}")
        # winner = highest sum among those with min > 0
        if min_p > 0 and total > best_sum:
            best_sum = total
            best_min = min_p
            best_scheme = scheme_name

    # Split schemes into "honest walk-forward" vs "hand-picked" (designed with
    # knowledge of the test split per-quintile breakdown). The q4_peaked_* variants
    # were added AFTER seeing TEST results — they are not valid walk-forward winners.
    HAND_PICKED = {'q4_peaked', 'q4_peaked_aggr'}
    HONEST = [s for s in scheme_names if s not in HAND_PICKED]

    print()
    uniform_sum = sum(sr.scheme_pnl['uniform_1x']['pnl'] for sr in split_results)

    # Honest winner (walk-forward valid)
    honest_best = None
    honest_best_sum = -1e18
    honest_best_min = -1e18
    for scheme_name in HONEST:
        pnls = [sr.scheme_pnl[scheme_name]['pnl'] for sr in split_results]
        total = sum(pnls); min_p = min(pnls)
        if min_p > 0 and total > honest_best_sum:
            honest_best_sum = total
            honest_best_min = min_p
            honest_best = scheme_name

    if honest_best is not None:
        lift = honest_best_sum - uniform_sum
        print(f"✓ HONEST WINNER (walk-forward valid): {honest_best}")
        print(f"  Sum test P&L across splits: ${honest_best_sum:+,.0f}")
        print(f"  Min test P&L across splits: ${honest_best_min:+,.0f}")
        print(f"  Lift vs uniform_1x: ${lift:+,.0f}")
    else:
        print("✗ No honest scheme positive on all 3 splits")

    # Flag overfit "winner" for transparency
    if best_scheme is not None and best_scheme in HAND_PICKED:
        print(f"\n⚠ OVERFIT (hand-picked after seeing test): {best_scheme} "
              f"totals ${best_sum:+,.0f} but is not a valid walk-forward result.")

    # Write markdown report
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    md_path = f"{OUT_DIR}/orb_sizing_{ts}.md"
    with open(md_path, 'w') as f:
        f.write("# ORB Conviction-Based Sizing — Walk-Forward\n\n")
        f.write(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n\n")
        f.write(f"**Input**: `{feat_csv}` ({len(df):,} trades)\n\n")
        f.write(f"**Pipeline**: filter at composite z >= **+{FILTER_THRESHOLD:.2f}** "
                f"(winner from filter study) → bucket kept trades into quintiles "
                f"using TRAIN cutoffs → apply sizing scheme.\n\n")
        f.write("## Sizing schemes\n\n")
        f.write("| scheme | Q1 | Q2 | Q3 | Q4 | Q5 |\n|---|---:|---:|---:|---:|---:|\n")
        for sn, sc in SIZING_SCHEMES.items():
            mults = [m for _, m in sc]
            f.write(f"| `{sn}` | {mults[0]:.2f}x | {mults[1]:.2f}x | {mults[2]:.2f}x "
                    f"| {mults[3]:.2f}x | {mults[4]:.2f}x |\n")

        for sr in split_results:
            f.write(f"\n## Split {sr.split_name}\n\n")
            f.write(f"- Test N={sr.test_n}, Kept N={sr.kept_n} "
                    f"({100*sr.kept_n/max(sr.test_n,1):.1f}%)\n\n")
            f.write("### Per-quintile test stats (unit sizing)\n\n")
            f.write("| Q | n | WR | avg/trade | P&L | DD | z-range |\n|---|---:|---:|---:|---:|---:|---:|\n")
            for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                s = sr.quintile_stats[q]
                lo = s['cutoff_lo']; hi = s['cutoff_hi']
                lo_str = f"{lo:+.2f}" if np.isfinite(lo) else "−∞"
                hi_str = f"{hi:+.2f}" if np.isfinite(hi) else "+∞"
                f.write(f"| {q} | {s['n']} | {s['wr']:.1f}% | "
                        f"${s['avg']:+,.0f} | ${s['pnl']:+,.0f} | "
                        f"${s['max_dd']:+,.0f} | [{lo_str}, {hi_str}] |\n")
            f.write("\n### TRAIN per-quintile avg (for adaptive fit)\n\n")
            f.write("| Q | TRAIN n | TRAIN avg/trade | adaptive mult |\n|---|---:|---:|---:|\n")
            for q, m in sr.adaptive_scheme:
                ts = sr.train_quintile_stats[q]
                f.write(f"| {q} | {ts['n']} | ${ts['avg']:+,.0f} | {m:.2f}x |\n")
            f.write("\n### Per-scheme P&L (kept test)\n\n")
            f.write("| scheme | n | WR | P&L | avg/tr | DD |\n|---|---:|---:|---:|---:|---:|\n")
            for scheme_name in scheme_names:
                s = sr.scheme_pnl[scheme_name]
                f.write(f"| `{scheme_name}` | {s['n']} | {s['wr']:.1f}% | "
                        f"${s['pnl']:+,.0f} | ${s['avg']:+,.0f} | "
                        f"${s['max_dd']:+,.0f} |\n")

        f.write("\n## Cross-split summary\n\n")
        f.write("| scheme | A P&L | B P&L | C P&L | Sum | Min | Verdict |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        for scheme_name in scheme_names:
            pnls = [sr.scheme_pnl[scheme_name]['pnl'] for sr in split_results]
            total = sum(pnls); min_p = min(pnls)
            verdict = '✓ all+' if min_p > 0 else '⚠ split loss'
            cells = ' | '.join(f"${p:+,.0f}" for p in pnls)
            f.write(f"| `{scheme_name}` | {cells} | ${total:+,.0f} | ${min_p:+,.0f} | {verdict} |\n")

        if honest_best is not None:
            lift = honest_best_sum - uniform_sum
            f.write(f"\n### ✓ Honest walk-forward winner: `{honest_best}`\n\n")
            f.write(f"- Sum test P&L: **${honest_best_sum:+,.0f}**\n")
            f.write(f"- Min test P&L (worst split): **${honest_best_min:+,.0f}**\n")
            f.write(f"- Lift vs `uniform_1x` (filter-only, no sizing): **${lift:+,.0f}**\n\n")
            f.write("**Validity**: `adaptive` fits per-quintile multipliers from TRAIN per-quintile "
                    "avg P&L only (no TEST peek). The other schemes with fixed multipliers are also "
                    "valid if the multipliers were picked independently of these results.\n\n")
            f.write("⚠ The `q4_peaked` and `q4_peaked_aggr` schemes were added AFTER observing that Q4 "
                    "beats Q5 in all 3 TEST splits under unit sizing. They are reported for completeness "
                    "but are **not valid walk-forward results** — they represent an upper bound that was "
                    "designed with test knowledge.\n")
        else:
            f.write("\n### ✗ No robust sizing scheme\n")

    print(f"\nReport: {md_path}")


if __name__ == '__main__':
    main()
