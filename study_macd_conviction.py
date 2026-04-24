#!/usr/bin/env python3
"""
MACD Wave Conviction — Step 2 Walk-Forward Study.

Compares multiple conviction scoring variants across 3 chronological train/test
splits, using post-hoc scaling of pnl_dollar on the baseline BT output.

Since pnl_dollar scales linearly with shares (and shares with position_size),
we can multiply pnl_dollar by conv_mult in-memory to simulate any sizing
variant without re-running the BT. This makes variant exploration fast.

Reads: macd_wave_results_baseline.csv
Writes: analysis_results/macd_conviction_step2_<date>.md
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------- Variant definitions ----------------------------------------------

@dataclass
class Variant:
    name: str
    description: str
    scorer: Callable[[int, int], float]   # (cross_time, vol_at_cross) -> conv_mult
    filter_threshold: float = 0.0          # drop trades with conv < threshold (0 = keep all)


def _clamp(x: float, lo: float = 0.5, hi: float = 2.0) -> float:
    return max(lo, min(hi, x))


def v0_baseline(cross_time: int, vol_at_cross: int) -> float:
    return 1.0


def v1_2tier(cross_time: int, vol_at_cross: int) -> float:
    score = 1.0
    score += 0.3 if cross_time <= 3 else (0.1 if cross_time <= 5 else 0.0)
    score += 0.3 if vol_at_cross <= 27_000 else (0.1 if vol_at_cross <= 79_000 else 0.0)
    return _clamp(score)


def v3_continuous(cross_time: int, vol_at_cross: int) -> float:
    """Continuous linear contribution, no bucketing."""
    cross_contrib = 0.3 * max(0.0, 1.0 - cross_time / 7.0)
    vol_contrib = 0.3 * max(0.0, 1.0 - vol_at_cross / 150_000.0)
    return _clamp(1.0 + cross_contrib + vol_contrib)


def v4_3tier(cross_time: int, vol_at_cross: int) -> float:
    """Finer gradient — 3 tiers per rule + 0 for worst."""
    if cross_time <= 3:
        c1 = 0.4
    elif cross_time <= 5:
        c1 = 0.2
    elif cross_time <= 7:
        c1 = 0.1
    else:
        c1 = 0.0
    if vol_at_cross <= 27_000:
        c2 = 0.4
    elif vol_at_cross <= 79_000:
        c2 = 0.2
    elif vol_at_cross <= 165_000:
        c2 = 0.1
    else:
        c2 = 0.0
    return _clamp(1.0 + c1 + c2)


def v5_weighted(cross_time: int, vol_at_cross: int) -> float:
    """Rule-weighted by train ρ magnitude: vol (ρ=−0.94) > cross (ρ=−0.78)."""
    # Cross: normal weights 0.3/0.1/0
    c1 = 0.3 if cross_time <= 3 else (0.1 if cross_time <= 5 else 0.0)
    # Vol: higher weights 0.4/0.2/0
    c2 = 0.4 if vol_at_cross <= 27_000 else (0.2 if vol_at_cross <= 79_000 else 0.0)
    return _clamp(1.0 + c1 + c2)


VARIANTS: List[Variant] = [
    Variant('V0_baseline',  'No conviction (reference)',            v0_baseline),
    Variant('V1_2tier',     '2-tier 0.3/0.1/0, clamp [0.5, 2.0]',   v1_2tier),
    Variant('V2_2tier+F1.2','V1 + drop trades below conv=1.2',      v1_2tier, filter_threshold=1.2),
    Variant('V3_linear',    'Continuous linear rule contrib',        v3_continuous),
    Variant('V4_3tier',     '3-tier 0.4/0.2/0.1/0',                  v4_3tier),
    Variant('V5_weighted',  'Vol 0.4/0.2/0 + cross 0.3/0.1/0',       v5_weighted),
]


# ---------- Walk-forward splits ----------------------------------------------

def walk_forward_splits(df: pd.DataFrame) -> List[Tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Return list of (name, train_start, train_end, test_end). Test starts after train_end."""
    return [
        ('A: H1\'25 → H2\'25+Q1\'26',
         pd.Timestamp('2025-01-01'), pd.Timestamp('2025-06-30'), pd.Timestamp('2026-03-27')),
        ('B: Jan\'25-Dec\'25 → Q1\'26',
         pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31'), pd.Timestamp('2026-03-27')),
        ('C: Jan-Sep\'25 → Oct\'25-Mar\'26',
         pd.Timestamp('2025-01-01'), pd.Timestamp('2025-09-30'), pd.Timestamp('2026-03-27')),
    ]


# ---------- Per-variant stats ------------------------------------------------

@dataclass
class SplitResult:
    split_name: str
    variant_name: str
    subset: str  # 'train' or 'test'
    n: int
    wr: float
    pnl: float
    dd: float
    pf: float
    avg_conv: float
    avg_pos: float  # avg position size (for capital efficiency)


def apply_variant(df: pd.DataFrame, variant: Variant) -> pd.DataFrame:
    """Return new df with conv_mult and pnl_sized columns, filtered if threshold > 0."""
    df = df.copy()
    df['conv_mult'] = df.apply(
        lambda r: variant.scorer(int(r['cross_time_min']), int(r['vol_at_cross'])),
        axis=1
    )
    if variant.filter_threshold > 0:
        df = df[df['conv_mult'] >= variant.filter_threshold].copy()
    df['pnl_sized'] = df['pnl_dollar'] * df['conv_mult']
    return df


def summarize(df: pd.DataFrame, split_name: str, variant_name: str, subset: str,
              position_size_baseline: float = 50_000.0) -> SplitResult:
    if df.empty:
        return SplitResult(split_name, variant_name, subset, 0, 0, 0, 0, 0, 0, 0)
    n = len(df)
    # Use pnl_sized if present, else pnl_dollar
    pnl_col = 'pnl_sized' if 'pnl_sized' in df.columns else 'pnl_dollar'
    d = df.sort_values(['date', 'entry_time']).reset_index(drop=True)
    wr = (d[pnl_col] > 0).mean() * 100
    pnl = d[pnl_col].sum()
    eq = d[pnl_col].cumsum()
    peak = eq.cummax()
    dd = (eq - peak).min()
    pos = d[d[pnl_col] > 0][pnl_col].sum()
    neg = abs(d[d[pnl_col] <= 0][pnl_col].sum())
    pf = pos / neg if neg > 0 else float('inf')
    avg_conv = d['conv_mult'].mean() if 'conv_mult' in d.columns else 1.0
    avg_pos = position_size_baseline * avg_conv
    return SplitResult(split_name, variant_name, subset, n, wr, pnl, dd, pf, avg_conv, avg_pos)


# ---------- Main study -------------------------------------------------------

def run_study(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'entry_time']).reset_index(drop=True)

    splits = walk_forward_splits(df)
    all_results: List[SplitResult] = []

    print(f"\nBaseline CSV: {csv_path} ({len(df)} trades)")
    print(f"Variants: {len(VARIANTS)}, Splits: {len(splits)}")

    # Run each variant on each split
    for split_name, t_start, t_end, test_end in splits:
        train_mask = (df['date'] >= t_start) & (df['date'] <= t_end)
        test_mask = (df['date'] > t_end) & (df['date'] <= test_end)
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        for v in VARIANTS:
            train_scored = apply_variant(train_df, v)
            test_scored = apply_variant(test_df, v)
            all_results.append(summarize(train_scored, split_name, v.name, 'train'))
            all_results.append(summarize(test_scored, split_name, v.name, 'test'))

    # Write report
    write_report(out_path, csv_path, splits, all_results)
    print(f"\nReport: {out_path}")


def write_report(out_path: str, csv_path: str,
                 splits: List[Tuple], all_results: List[SplitResult]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Index results for easy lookup
    key = {(r.split_name, r.variant_name, r.subset): r for r in all_results}

    lines = []
    lines.append("# MACD Wave Conviction — Step 2 Walk-Forward Study\n")
    lines.append(f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n")
    lines.append(f"**Input:** `{csv_path}`\n")
    lines.append(f"**Splits:** {len(splits)} chronological walk-forward windows")
    lines.append(f"**Variants:** {len(VARIANTS)} scoring formulas\n")

    # --- Variant legend
    lines.append("## Variants\n")
    lines.append("| Name | Description | Filter? |")
    lines.append("|---|---|---|")
    for v in VARIANTS:
        f = f"≥{v.filter_threshold}" if v.filter_threshold > 0 else "no"
        lines.append(f"| `{v.name}` | {v.description} | {f} |")
    lines.append("")

    # --- Per-split summary tables
    for split_name, t_start, t_end, test_end in splits:
        lines.append(f"## Split {split_name}\n")
        lines.append(f"Train: {t_start.date()} → {t_end.date()}   Test: {(t_end + pd.Timedelta(days=1)).date()} → {test_end.date()}\n")

        for subset in ('train', 'test'):
            lines.append(f"### {subset.upper()}\n")
            lines.append("| Variant | n | WR | P&L | Δ vs V0 | DD | PF | avg conv |")
            lines.append("|---|---|---|---|---|---|---|---|")
            base = key.get((split_name, 'V0_baseline', subset))
            base_pnl = base.pnl if base else 0
            for v in VARIANTS:
                r = key.get((split_name, v.name, subset))
                if not r or r.n == 0:
                    continue
                delta = r.pnl - base_pnl
                delta_str = f"${delta:+,.0f}" if v.name != 'V0_baseline' else "—"
                lines.append(
                    f"| `{r.variant_name}` | {r.n} | {r.wr:.0f}% | ${r.pnl:+,.0f} | "
                    f"{delta_str} | ${r.dd:+,.0f} | {r.pf:.2f} | {r.avg_conv:.2f} |"
                )
            lines.append("")

    # --- Summary across splits (test only)
    lines.append("## Cross-split test summary (OOS)\n")
    lines.append("| Variant | "
                 + " | ".join(f"Split {chr(65+i)} Δ$" for i in range(len(splits)))
                 + " | Mean Δ | Min Δ | Range | avg conv | Cap-eff |")
    lines.append("|---|" + "---|" * (len(splits) + 5))
    variant_summaries = []
    for v in VARIANTS:
        deltas = []
        for split_name, _, _, _ in splits:
            base = key.get((split_name, 'V0_baseline', 'test'))
            r = key.get((split_name, v.name, 'test'))
            if not base or not r:
                continue
            delta = r.pnl - base.pnl
            # Capital efficiency: ΔP&L / extra-notional-deployed.
            # Baseline avg notional per trade = $50K × 1.0 × n.
            # Variant avg notional = $50K × avg_conv × n_kept.
            # Extra notional = $50K × (avg_conv × n_kept - 1.0 × n_base).
            extra_notional = 50_000.0 * (r.avg_conv * r.n - 1.0 * base.n)
            cap_eff = delta / extra_notional if extra_notional > 0 else float('nan')
            deltas.append((split_name, delta, r.avg_conv, cap_eff))
        if not deltas:
            continue
        mean_d = np.mean([d for _, d, _, _ in deltas])
        min_d = min(d for _, d, _, _ in deltas)
        max_d = max(d for _, d, _, _ in deltas)
        rng = max_d - min_d
        avg_conv = np.mean([c for _, _, c, _ in deltas])
        avg_ce = np.nanmean([ce for _, _, _, ce in deltas])
        delta_cells = " | ".join(f"${d:+,.0f}" for _, d, _, _ in deltas)
        lines.append(
            f"| `{v.name}` | {delta_cells} | ${mean_d:+,.0f} | ${min_d:+,.0f} | "
            f"${rng:,.0f} | {avg_conv:.2f} | {avg_ce:.3f} |"
        )
        variant_summaries.append((v.name, mean_d, min_d, max_d, rng, avg_conv, avg_ce))
    lines.append("")
    lines.append("_Cap-eff = ΔP&L / extra-notional-deployed. Values >0 = positive structural "
                 "lift (each extra $ of capital generates positive ΔP&L); <0 = capital being "
                 "deployed to losers OOS. Pure filter variants that REDUCE notional have "
                 "cap-eff computed only if extra_notional>0._\n")

    # --- Winner recommendation
    lines.append("## Recommendation\n")
    valid = [t for t in variant_summaries if t[0] != 'V0_baseline']
    if not valid:
        lines.append("_No variants evaluated._\n")
    else:
        # Primary sort: mean test ΔP&L
        # Tiebreak: min split delta (robustness — penalize bad worst-case)
        ranked = sorted(valid, key=lambda v: (-v[1], -v[2]))
        winner = ranked[0]
        lines.append(f"**Primary winner (by mean test ΔP&L):** `{winner[0]}` — "
                     f"mean ${winner[1]:+,.0f}, worst ${winner[2]:+,.0f}, range ${winner[4]:,.0f}\n")

        # Also rank by robustness (min test delta)
        ranked_robust = sorted(valid, key=lambda v: (-v[2], -v[1]))
        if ranked_robust[0][0] != winner[0]:
            r2 = ranked_robust[0]
            lines.append(f"**Most robust (by worst-case test ΔP&L):** `{r2[0]}` — "
                         f"worst ${r2[2]:+,.0f}, mean ${r2[1]:+,.0f}\n")

        lines.append("Full ranking (by mean ΔP&L):")
        for name, mean, mn, mx, rng, avg_conv, ce in ranked:
            safe = "✓ robust" if mn > 0 else ("⚠ mixed" if mn > -500 else "✗ fragile")
            lines.append(
                f"- `{name}`: mean ${mean:+,.0f}, range [{mn:+,.0f} .. {mx:+,.0f}], "
                f"cap-eff={ce:.3f} — {safe}"
            )
    lines.append("")

    # --- Methodology notes
    lines.append("## Methodology\n")
    lines.append(
        "- Each variant scaled pnl_dollar = original_pnl × conv_mult, where conv_mult "
        "is computed from (cross_time_min, vol_at_cross) per variant's scorer.\n"
        "- Post-hoc scaling is mathematically equivalent to re-running the BT with "
        "`shares = int(position_size × conv_mult / entry_price)` within integer-share "
        "rounding (<=0.1% delta).\n"
        "- Train quartile edges (3, 5, 7 for cross_time; 27K, 79K, 165K for vol) "
        "were derived from Split A's H1'25 train set in step 1 research and reused here.\n"
        "- V3 (continuous) uses `contrib = 0.3 × max(0, 1 − x/threshold)` with threshold "
        "= 7 (cross_time) and 150K (vol).\n"
        "- Filter variants (V2) apply the threshold BEFORE the sizing multiplier "
        "(drop below cutoff, then size survivors).\n"
    )

    with open(out_path, 'w') as f:
        f.write("\n".join(lines))


if __name__ == '__main__':
    import sys
    csv_in = sys.argv[1] if len(sys.argv) > 1 else 'macd_wave_results_baseline.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else \
        f"analysis_results/macd_conviction_step2_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    run_study(csv_in, out)
