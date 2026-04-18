#!/usr/bin/env python3
"""MACD 1.0x bucket treatment simulator (research/multiplier_audit.md finding 1).

Audit section E showed HOLDOUT MACD 1.0x bucket is -$13,609 while 1.5x is
+$34,301. Question: should we skip the 1.0x zone entirely, or downsize?

Simulates four variants post-hoc (read-only):
  - Baseline: keep 1.0x at 1.0x
  - S:        skip 1.0x (reject trade)
  - H5:       1.0x → 0.5x
  - H75:      1.0x → 0.75x

Note — cache PnL has MACD zone already baked in *only* when risk_tier_mult <= 1.0
(see backtest.py:2555 — MACD skipped when tier > 1). Since the cache is built
with risk_tiers_enabled=False (batch_backtest.py:1520), all trades have MACD
already baked in. So we divide the existing 1.0x factor out (it's identity
anyway) and multiply by the new factor — i.e. just multiply pnl by new_mult.
"""
from __future__ import annotations

import csv

CACHES = {
    'TRAIN (2025 Jan-Jul)': ('/tmp/expVH_final/cache_V_2025.csv', '2025-01-01', '2025-07-31'),
    'VAL (2025 Aug-Dec)':   ('/tmp/expVH_final/cache_V_2025.csv', '2025-08-01', '2025-12-31'),
    'HOLDOUT Q1 2026':      ('/tmp/expVH_final/cache_V_q1.csv',   '2026-01-01', '2026-03-31'),
    'HOLDOUT Apr 1-17':     ('/tmp/april_cache/cache.csv',        '2026-04-01', '2026-04-17'),
}

MIN_CONV = 1.4


def load():
    out = {}
    for label, (path, start, end) in CACHES.items():
        rows = []
        with open(path) as f:
            for t in csv.DictReader(f):
                if t['date'] < start or t['date'] > end:
                    continue
                try:
                    cm = float(t.get('conviction_mult') or 1.0)
                except ValueError:
                    cm = 1.0
                if cm < MIN_CONV:
                    continue
                try:
                    rows.append({
                        'date': t['date'],
                        'pnl': float(t['pnl']),
                        'macd_mult': float(t.get('macd_zone_mult') or 1.0),
                    })
                except (ValueError, KeyError):
                    continue
        out[label] = rows
    return out


VARIANTS = {
    'Baseline (1.0x kept)': {'action': 'keep',  'factor': 1.0},
    'S (skip 1.0x zone)':   {'action': 'skip',  'factor': 0.0},
    'H5 (1.0x → 0.5x)':     {'action': 'scale', 'factor': 0.5},
    'H75 (1.0x → 0.75x)':   {'action': 'scale', 'factor': 0.75},
}


def apply_variant(trades, spec):
    out = []
    for t in trades:
        if abs(t['macd_mult'] - 1.0) < 0.001:  # currently 1.0x bucket
            if spec['action'] == 'skip':
                continue
            if spec['action'] == 'scale':
                out.append({**t, 'pnl': t['pnl'] * spec['factor']})
                continue
        out.append(t)
    return out


def summarize(trades):
    n = len(trades)
    if n == 0:
        return {'n': 0, 'wr': 0.0, 'pnl': 0.0, 'maxdd': 0.0}
    wins = sum(1 for t in trades if t['pnl'] > 0)
    total = sum(t['pnl'] for t in trades)
    running = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: x['date']):
        running += t['pnl']
        peak = max(peak, running)
        max_dd = min(max_dd, running - peak)
    return {'n': n, 'wr': wins / n * 100, 'pnl': total, 'maxdd': max_dd}


def main():
    splits = load()

    # Sanity bucket breakdown
    print("# MACD 1.0x bucket treatment sim\n")
    print(f"Conv filter: conviction_mult >= {MIN_CONV}.\n")
    print("## Bucket baseline (no MACD change)\n")
    print("| Split | 1.0x n | 1.0x PnL | 1.0x WR | 1.5x n | 1.5x PnL | 1.5x WR |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for split, rows in splits.items():
        b1 = [t for t in rows if abs(t['macd_mult'] - 1.0) < 0.001]
        b15 = [t for t in rows if abs(t['macd_mult'] - 1.5) < 0.001]
        def pct(xs):
            return (sum(1 for t in xs if t['pnl'] > 0) / len(xs) * 100) if xs else 0.0
        print(f"| {split} | {len(b1)} | ${sum(t['pnl'] for t in b1):+,.0f} | "
              f"{pct(b1):.1f}% | {len(b15)} | "
              f"${sum(t['pnl'] for t in b15):+,.0f} | {pct(b15):.1f}% |")

    baseline_stats = {
        split: summarize(apply_variant(rows, VARIANTS['Baseline (1.0x kept)']))
        for split, rows in splits.items()
    }

    for vname, spec in VARIANTS.items():
        print(f"\n## {vname}\n")
        print("| Split | n | WR | Total PnL | MaxDD | Δ vs base |")
        print("|---|---:|---:|---:|---:|---:|")
        for split, rows in splits.items():
            scaled = apply_variant(rows, spec)
            s = summarize(scaled)
            delta = s['pnl'] - baseline_stats[split]['pnl']
            print(f"| {split} | {s['n']} | {s['wr']:.1f}% | ${s['pnl']:+,.0f} | "
                  f"${s['maxdd']:+,.0f} | ${delta:+,.0f} |")

    # HOLDOUT combined ranking
    print("\n## HOLDOUT (Q1 + April) combined\n")
    print("| Variant | n | WR | Total PnL | Δ vs base |")
    print("|---|---:|---:|---:|---:|")
    holdout_labels = ['HOLDOUT Q1 2026', 'HOLDOUT Apr 1-17']
    holdout_baseline = 0.0
    for label in holdout_labels:
        holdout_baseline += baseline_stats[label]['pnl']
    for vname, spec in VARIANTS.items():
        all_tr = []
        for label in holdout_labels:
            all_tr.extend(apply_variant(splits[label], spec))
        s = summarize(all_tr)
        delta = s['pnl'] - holdout_baseline
        print(f"| {vname} | {s['n']} | {s['wr']:.1f}% | ${s['pnl']:+,.0f} | ${delta:+,.0f} |")


if __name__ == "__main__":
    main()
