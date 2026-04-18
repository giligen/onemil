#!/usr/bin/env python3
"""Tier redesign post-hoc simulator (research/multiplier_audit.md follow-up).

Reads the three V-on backtest caches (2025 full, Q1 2026, April 2026-17),
re-applies candidate risk-tier multiplier tables, and reports per-split WR /
total P&L / running-sum min (DD proxy). Read-only; no code or config changes.

Cache PnL semantics match batch_backtest.py:339-380 — cache stores PnL with
conviction × MACD already baked in. Stage-2 divides out MACD when tier > 1.0
(matches BT planner behavior), combines conv × tier capped at 3.0x.

Reproduces the audit-time tier logic *exactly* (including the divide-out-MACD
rule) so that variants are apples-to-apples comparable to the BT Stage-2
output — not synthetic.
"""
from __future__ import annotations

import csv
from pathlib import Path

CACHES = {
    'TRAIN (2025 Jan-Jul)': ('/tmp/expVH_final/cache_V_2025.csv', '2025-01-01', '2025-07-31'),
    'VAL (2025 Aug-Dec)':   ('/tmp/expVH_final/cache_V_2025.csv', '2025-08-01', '2025-12-31'),
    'HOLDOUT Q1 2026':      ('/tmp/expVH_final/cache_V_q1.csv',   '2026-01-01', '2026-03-31'),
    'HOLDOUT Apr 1-17':     ('/tmp/april_cache/cache.csv',        '2026-04-01', '2026-04-17'),
}

# Post-filter: mirrors BT Stage-2 `min_conviction_mult: 1.4`.
MIN_CONV = 1.4


def load_trades():
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
                        'symbol': t['symbol'],
                        'date': t['date'],
                        'entry_price': float(t['entry_price']),
                        'avg_vol': float(t.get('avg_volume_20d') or 0),
                        'pnl': float(t['pnl']),
                        'conv_mult': cm,
                        'macd_mult': float(t.get('macd_zone_mult') or 1.0),
                    })
                except (ValueError, KeyError):
                    continue
        out[label] = rows
    return out


# ------------------------------ variants --------------------------------------
# Each variant is an ordered tier list: first match wins. Multiplier == None → orphan
# (i.e., trade kept at its cached pnl). A tier with multiplier 1.0 explicitly
# neutralizes a bucket (still evaluates the conv×macd divide-out logic the same
# way the orphan path does — so 1.0 is identical to "no tier" here).

VARIANTS = {
    'Baseline (current)': [
        {'name': 'T1 $10-15,500K-5M',  'min_p': 10, 'max_p': 15, 'min_v': 500_000, 'max_v': 5_000_000, 'mult': 2.0},
        {'name': 'T2 $15-23,500K-5M',  'min_p': 15, 'max_p': 23, 'min_v': 500_000, 'max_v': 5_000_000, 'mult': 1.0},
    ],
    'A: add T3 <$5 @1.5x': [
        {'name': 'T1 $10-15,500K-5M',  'min_p': 10, 'max_p': 15, 'min_v': 500_000, 'max_v': 5_000_000, 'mult': 2.0},
        {'name': 'T2 $15-23,500K-5M',  'min_p': 15, 'max_p': 23, 'min_v': 500_000, 'max_v': 5_000_000, 'mult': 1.0},
        {'name': 'T3 <$5,500K-5M @1.5','min_p': 0,  'max_p': 5,  'min_v': 500_000, 'max_v': 5_000_000, 'mult': 1.5},
    ],
    'B: demote T1, <$5 @2.0x': [
        {'name': 'T1 $10-15,500K-5M',  'min_p': 10, 'max_p': 15, 'min_v': 500_000, 'max_v': 5_000_000, 'mult': 1.0},
        {'name': 'T2 $15-23,500K-5M',  'min_p': 15, 'max_p': 23, 'min_v': 500_000, 'max_v': 5_000_000, 'mult': 1.0},
        {'name': 'T3 <$5,500K-5M @2.0','min_p': 0,  'max_p': 5,  'min_v': 500_000, 'max_v': 5_000_000, 'mult': 2.0},
    ],
    'C: broad T3 <$5 any-vol @1.5': [
        {'name': 'T1 $10-15,500K-5M',  'min_p': 10, 'max_p': 15, 'min_v': 500_000, 'max_v': 5_000_000, 'mult': 1.0},
        {'name': 'T2 $15-23,500K-5M',  'min_p': 15, 'max_p': 23, 'min_v': 500_000, 'max_v': 5_000_000, 'mult': 1.0},
        {'name': 'T3 <$5 any-vol @1.5','min_p': 0,  'max_p': 5,  'min_v': 0,       'max_v': 999_999_999, 'mult': 1.5},
    ],
    'D: B + $10-15 <500K orphan rescue': [
        {'name': 'T1a $10-15,500K-5M', 'min_p': 10, 'max_p': 15, 'min_v': 500_000, 'max_v': 5_000_000, 'mult': 1.0},
        {'name': 'T1b $10-15,<500K @1.5','min_p':10,'max_p': 15, 'min_v': 0,       'max_v': 500_000,   'mult': 1.5},
        {'name': 'T2 $15-23,500K-5M',  'min_p': 15, 'max_p': 23, 'min_v': 500_000, 'max_v': 5_000_000, 'mult': 1.0},
        {'name': 'T3 <$5,500K-5M @2.0','min_p': 0,  'max_p': 5,  'min_v': 500_000, 'max_v': 5_000_000, 'mult': 2.0},
    ],
    'E: A + rescue $10-15<500K @1.5': [
        {'name': 'T1a $10-15,500K-5M', 'min_p': 10, 'max_p': 15, 'min_v': 500_000, 'max_v': 5_000_000, 'mult': 2.0},
        {'name': 'T1b $10-15,<500K @1.5','min_p':10,'max_p': 15, 'min_v': 0,       'max_v': 500_000,   'mult': 1.5},
        {'name': 'T2 $15-23,500K-5M',  'min_p': 15, 'max_p': 23, 'min_v': 500_000, 'max_v': 5_000_000, 'mult': 1.0},
        {'name': 'T3 <$5,500K-5M @1.5','min_p': 0,  'max_p': 5,  'min_v': 500_000, 'max_v': 5_000_000, 'mult': 1.5},
    ],
}


def apply_variant(trades: list, tiers: list) -> list:
    """Return a new trade list with PnL re-scaled per this tier table."""
    out = []
    for t in trades:
        ep = t['entry_price']
        vol = t['avg_vol']
        tier_mult = 1.0
        matched = False
        for tier in tiers:
            if (tier['min_p'] <= ep < tier['max_p']
                    and tier['min_v'] <= vol <= tier['max_v']):
                tier_mult = tier['mult']
                matched = True
                break
        # Replicates batch_backtest.py:362-374 exactly.
        # Cache has conv × macd baked in. Target = min(3.0, conv * tier_mult)
        # (MACD excluded when a tier matches — matches BT planner path).
        # When NO tier matches, trade stays at cached PnL (orphan path).
        new_pnl = t['pnl']
        if matched:
            conv_mult = t['conv_mult']
            macd_mult = t['macd_mult']
            combined = min(3.0, conv_mult * tier_mult)
            denom = conv_mult * macd_mult
            if denom > 0:
                actual_scale = combined / denom
                if abs(actual_scale - 1.0) > 0.001:
                    new_pnl = t['pnl'] * actual_scale
        out.append({**t, 'pnl': new_pnl, 'tier_mult': tier_mult})
    return out


def summarize(trades: list) -> dict:
    n = len(trades)
    if n == 0:
        return {'n': 0, 'wr': 0.0, 'pnl': 0.0, 'maxdd': 0.0}
    wins = sum(1 for t in trades if t['pnl'] > 0)
    total = sum(t['pnl'] for t in trades)
    # running-sum max drawdown proxy (sorted by date)
    running = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: x['date']):
        running += t['pnl']
        peak = max(peak, running)
        max_dd = min(max_dd, running - peak)
    return {'n': n, 'wr': wins / n * 100, 'pnl': total, 'maxdd': max_dd}


def fmt_row(label, stats):
    return (f"| {label} | {stats['n']} | {stats['wr']:.1f}% | "
            f"${stats['pnl']:+,.0f} | ${stats['maxdd']:+,.0f} |")


def main():
    splits = load_trades()

    # Baseline for delta calc.
    baseline_stats = {
        split: summarize(apply_variant(rows, VARIANTS['Baseline (current)']))
        for split, rows in splits.items()
    }

    print("# Tier redesign post-hoc sim\n")
    print(f"Read-only: apply alt tier tables to cached trades (conv_mult >= {MIN_CONV}).\n")
    print("PnL scaling mirrors batch_backtest.py Stage-2 exactly (divide out MACD "
          "when tier matches; cap at 3.0x combined).\n")

    # Main table per variant
    for vname, tiers in VARIANTS.items():
        print(f"## {vname}\n")
        print("| Split | n | WR | Total PnL | MaxDD (run-sum) | Δ vs base |")
        print("|---|---:|---:|---:|---:|---:|")
        for split, rows in splits.items():
            scaled = apply_variant(rows, tiers)
            s = summarize(scaled)
            delta = s['pnl'] - baseline_stats[split]['pnl']
            print(f"| {split} | {s['n']} | {s['wr']:.1f}% | ${s['pnl']:+,.0f} | "
                  f"${s['maxdd']:+,.0f} | ${delta:+,.0f} |")
        print()

    # Per-bucket contribution (baseline) to sanity-check audit numbers
    print("## Bucket audit (current state; all splits combined)\n")
    print("| Price | Vol | n | WR | Total PnL |")
    print("|---|---|---:|---:|---:|")
    all_tr = [t for rows in splits.values() for t in rows]
    def bucket(ep, vol):
        if ep < 5: p = '<$5'
        elif ep < 10: p = '$5-10'
        elif ep < 15: p = '$10-15'
        elif ep < 23: p = '$15-23'
        else: p = '$23+'
        if vol < 500_000: v = '<500K'
        elif vol <= 5_000_000: v = '500K-5M'
        else: v = '5M+'
        return p, v
    from collections import defaultdict
    buckets = defaultdict(list)
    for t in all_tr:
        buckets[bucket(t['entry_price'], t['avg_vol'])].append(t)
    for (p, v), ts in sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n = len(ts)
        wr = sum(1 for t in ts if t['pnl'] > 0) / n * 100
        pnl = sum(t['pnl'] for t in ts)
        print(f"| {p} | {v} | {n} | {wr:.1f}% | ${pnl:+,.0f} |")

    # Winner selector
    print("\n## Ranking by HOLDOUT gain\n")
    holdout_labels = ['HOLDOUT Q1 2026', 'HOLDOUT Apr 1-17']
    holdout_baseline = sum(baseline_stats[l]['pnl'] for l in holdout_labels)
    print(f"Baseline HOLDOUT total: ${holdout_baseline:+,.0f}")
    ranked = []
    for vname, tiers in VARIANTS.items():
        if vname == 'Baseline (current)':
            continue
        v_pnl = sum(summarize(apply_variant(splits[l], tiers))['pnl']
                    for l in holdout_labels)
        ranked.append((vname, v_pnl, v_pnl - holdout_baseline))
    ranked.sort(key=lambda x: -x[2])
    print("\n| Variant | HOLDOUT total | Δ vs base |")
    print("|---|---:|---:|")
    for vname, pnl, delta in ranked:
        print(f"| {vname} | ${pnl:+,.0f} | ${delta:+,.0f} |")


if __name__ == "__main__":
    main()
