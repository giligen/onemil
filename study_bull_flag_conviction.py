#!/usr/bin/env python3
"""Bull Flag Conviction — Step 2 Walk-Forward Study.

Compares conviction-filter variants across 3 chronological train/test splits.
Mirrors the methodology that picked V4 for MACD wave (study_macd_conviction.py).

Variants tested are FILTER-only (sizing curve held constant — defers to step 3).
Each variant calls `filter_bull_flag_trades` from batch_backtest with a custom
conviction threshold patched into Config._load_yaml_only.

Reads: data/bull_flag_cache_e50_x30.csv (raw cache, 389 trades pre-filter)
Writes: analysis_results/bull_flag_conviction_step2_<date>.md
"""
from __future__ import annotations

import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Bring in batch_backtest internals
from batch_backtest import filter_bull_flag_trades
from config import Config


# Variant: name, conviction min_threshold, description
@dataclass
class Variant:
    name: str
    threshold: float
    description: str


VARIANTS: List[Variant] = [
    Variant('V0_no_filter',     0.0, 'No conviction filter (baseline)'),
    Variant('V1_current_1.2',   1.2, 'Current PROD filter (>= 1.2)'),
    Variant('V2_threshold_1.3', 1.3, 'Tighter filter (>= 1.3)'),
    Variant('V3_threshold_1.5', 1.5, 'Aggressive filter (>= 1.5)'),
    Variant('V4_threshold_1.0', 1.0, 'Looser filter (>= 1.0)'),
    Variant('V5_threshold_1.4', 1.4, 'Mid filter (>= 1.4)'),
]


SPLITS: List[Tuple[str, str, str, str]] = [
    ('A: H1\'25 → H2\'25+Q1\'26+Apr\'26',
     '2025-01-01', '2025-06-30', '2026-04-30'),
    ('B: Y2025 → Q1\'26+Apr\'26',
     '2025-01-01', '2025-12-31', '2026-04-30'),
    ('C: Jan-Sep\'25 → Oct\'25-Apr\'26',
     '2025-01-01', '2025-09-30', '2026-04-30'),
]


def _make_cfg_with_threshold(threshold: float) -> dict:
    """Build a Config-shaped dict that mirrors prod yaml but overrides conviction."""
    base = Config._load_yaml_only()
    cfg = deepcopy(base)
    cfg.setdefault('trading', {})
    cfg['trading'].setdefault('conviction_scoring', {})
    cfg['trading']['conviction_scoring']['enabled'] = (threshold > 0)
    cfg['trading']['conviction_scoring']['min_threshold'] = threshold
    return cfg


def _load_cache(path: str = 'data/bull_flag_cache_e50_x30.csv') -> pd.DataFrame:
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def _split(df: pd.DataFrame, train_start: str, train_end: str,
           test_end: str) -> Tuple[List[Dict], List[Dict]]:
    ts = pd.Timestamp(train_start)
    te = pd.Timestamp(train_end)
    end = pd.Timestamp(test_end)
    # filter_bull_flag_trades expects 'date' as ISO string (calls date.fromisoformat)
    df_iso = df.copy()
    df_iso['date'] = df_iso['date'].dt.strftime('%Y-%m-%d')
    df_dt_mask = (df['date'] >= ts) & (df['date'] <= te)
    df_test_mask = (df['date'] > te) & (df['date'] <= end)
    train = df_iso[df_dt_mask].to_dict('records')
    test = df_iso[df_test_mask].to_dict('records')
    return train, test


def _stats(trades: List[Dict]) -> Dict[str, float]:
    if not trades:
        return {'n': 0, 'wr': 0.0, 'pnl': 0.0, 'pf': 0.0, 'dd': 0.0, 'avg': 0.0}
    df = pd.DataFrame(trades)
    # date may already be string; sort by string is fine for ISO-format
    df = df.sort_values(['date', 'entry_time_et']).reset_index(drop=True)
    n = len(df)
    pnl = df['pnl'].sum()
    wr = (df['pnl'] > 0).mean() * 100
    pos = df[df['pnl'] > 0]['pnl'].sum()
    neg = abs(df[df['pnl'] <= 0]['pnl'].sum())
    pf = pos / neg if neg > 0 else float('inf')
    eq = df['pnl'].cumsum()
    peak = eq.cummax()
    dd = (eq - peak).min()
    return {
        'n': n, 'wr': wr, 'pnl': pnl, 'pf': pf, 'dd': dd,
        'avg': pnl / n if n else 0,
    }


def _run_variant(trades_in: List[Dict], variant: Variant,
                 monkeypatch_cfg) -> List[Dict]:
    """Run filter_bull_flag_trades with the variant's threshold patched in."""
    # Save original; patch; restore. Each call reloads via Config._load_yaml_only.
    orig = Config._load_yaml_only
    Config._load_yaml_only = staticmethod(lambda: monkeypatch_cfg)
    try:
        return filter_bull_flag_trades(trades_in)
    finally:
        Config._load_yaml_only = orig


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else \
        f"analysis_results/bull_flag_conviction_step2_{datetime.now().strftime('%Y%m%d_%H%M')}.md"

    df = _load_cache()
    print(f"Loaded {len(df)} cached trades from "
          f"{df['date'].min().date()} to {df['date'].max().date()}\n")

    # Pre-compute per-variant configs
    variant_cfgs = {v.name: _make_cfg_with_threshold(v.threshold) for v in VARIANTS}

    # Run each variant on each split, capturing train/test stats
    rows = []  # (split_name, variant_name, subset, stats)
    for split_name, ts, te, tend in SPLITS:
        train_in, test_in = _split(df, ts, te, tend)
        for v in VARIANTS:
            trn_out = _run_variant(train_in, v, variant_cfgs[v.name])
            tst_out = _run_variant(test_in, v, variant_cfgs[v.name])
            rows.append((split_name, v.name, 'train', _stats(trn_out)))
            rows.append((split_name, v.name, 'test', _stats(tst_out)))

    # ---------- Write report ----------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = []
    lines.append("# Bull Flag Conviction — Step 2 Walk-Forward Study\n")
    lines.append(f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n")
    lines.append(f"**Cache:** `data/bull_flag_cache_e50_x30.csv` ({len(df)} trades)")
    lines.append(f"**Splits:** {len(SPLITS)}, **Variants:** {len(VARIANTS)}\n")

    lines.append("## Variants\n")
    lines.append("| Name | Threshold | Description |\n|---|---|---|")
    for v in VARIANTS:
        lines.append(f"| `{v.name}` | {v.threshold} | {v.description} |")
    lines.append("")

    # Per-split tables
    key = {(r[0], r[1], r[2]): r[3] for r in rows}
    for split_name, ts, te, tend in SPLITS:
        lines.append(f"## Split {split_name}\n")
        lines.append(f"Train: {ts} → {te}   Test: > {te} ≤ {tend}\n")
        for subset in ('train', 'test'):
            lines.append(f"### {subset.upper()}\n")
            lines.append("| Variant | n | WR | P&L | Δ vs V0 | DD | PF |")
            lines.append("|---|---|---|---|---|---|---|")
            base = key.get((split_name, 'V0_no_filter', subset))
            for v in VARIANTS:
                s = key.get((split_name, v.name, subset))
                if not s:
                    continue
                delta = s['pnl'] - base['pnl'] if base else 0
                delta_str = f"${delta:+,.0f}" if v.name != 'V0_no_filter' else '—'
                lines.append(
                    f"| `{v.name}` | {s['n']} | {s['wr']:.0f}% | "
                    f"${s['pnl']:+,.0f} | {delta_str} | "
                    f"${s['dd']:+,.0f} | {s['pf']:.2f} |"
                )
            lines.append("")

    # OOS summary
    lines.append("## Cross-split TEST summary (OOS)\n")
    lines.append("| Variant | "
                 + " | ".join(f"Split {chr(65+i)} Δ$" for i in range(len(SPLITS)))
                 + " | Mean Δ | Min Δ | Max Δ | Verdict |")
    lines.append("|---|" + "---|" * (len(SPLITS) + 4))
    summaries = []
    for v in VARIANTS:
        deltas = []
        for split_name, _, _, _ in SPLITS:
            base = key.get((split_name, 'V0_no_filter', 'test'))
            s = key.get((split_name, v.name, 'test'))
            if base and s:
                deltas.append(s['pnl'] - base['pnl'])
        if not deltas:
            continue
        mean_d = float(np.mean(deltas))
        min_d = float(min(deltas))
        max_d = float(max(deltas))
        verdict = '✓ robust' if min_d > 0 else (
            '⚠ mixed' if max_d > 0 else '✗ losing')
        delta_cells = " | ".join(f"${d:+,.0f}" for d in deltas)
        lines.append(f"| `{v.name}` | {delta_cells} | "
                     f"${mean_d:+,.0f} | ${min_d:+,.0f} | ${max_d:+,.0f} | {verdict} |")
        summaries.append((v.name, mean_d, min_d, max_d, verdict))
    lines.append("")

    # Recommendation
    lines.append("## Recommendation\n")
    valid = [s for s in summaries if s[0] != 'V0_no_filter']
    if valid:
        ranked = sorted(valid, key=lambda s: (-s[1], -s[2]))
        winner = ranked[0]
        lines.append(f"**Winner by mean test ΔP&L:** `{winner[0]}` — "
                     f"mean ${winner[1]:+,.0f}, worst ${winner[2]:+,.0f}\n")
        lines.append("Full ranking:")
        for n, m, mn, mx, vd in ranked:
            lines.append(f"- `{n}`: mean ${m:+,.0f}, range [${mn:+,.0f} … ${mx:+,.0f}] {vd}")
    lines.append("")
    lines.append("## Methodology notes\n")
    lines.append(
        "- Each variant patches `Config._load_yaml_only` to override "
        "`trading.conviction_scoring.min_threshold`, then calls "
        "`batch_backtest.filter_bull_flag_trades` on the train/test trade lists.\n"
        "- Sizing held constant (cache pnl already baked with conv-mult sizing).\n"
        "- Other filters (regime, volume, leveraged-ETF, daily-range, "
        "max-concurrent, daily-loss-limit) all fire identically across variants.\n"
        "- 'TEST' set excludes train period (no leakage)."
    )

    with open(out_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Report: {out_path}")

    # Console summary
    print("\nSUMMARY (cross-split TEST OOS ΔP&L vs V0):")
    print(f"{'Variant':<22} {'mean':>10} {'min':>10} {'max':>10}  verdict")
    for n, m, mn, mx, vd in sorted(summaries, key=lambda s: -s[1]):
        if n == 'V0_no_filter':
            continue
        print(f"{n:<22} ${m:>+9,.0f} ${mn:>+9,.0f} ${mx:>+9,.0f}  {vd}")


if __name__ == '__main__':
    main()
