#!/usr/bin/env python3
"""Audit-fix Stage-2 comparison.

Compares baseline (V-on) caches vs audit_fix (V-on + reweighted rules 1/2/3/5/7)
caches under identical Stage-2 conditions (min_conv>=1.4, same tier config).

Inputs (expected to exist on disk):
  BASELINE: /tmp/expVH_final/cache_V_2025.csv, /tmp/expVH_final/cache_V_q1.csv,
            /tmp/april_cache/cache.csv
  AUDITFIX: /tmp/audit_fix/cache_2025.csv, /tmp/audit_fix/cache_q1.csv,
            /tmp/audit_fix/cache_apr.csv

Both sets of caches are built with the same config pre-filters; audit_fix
differs ONLY in the conviction rule weights (rules 1/2/3/5/7 downweighted).
Stage-2 applies the current config's tier scaling and min_conv filter.
"""
from __future__ import annotations

import csv
from collections import defaultdict

import yaml

SPLITS = {
    'TRAIN (2025 Jan-Jul)': ('2025-01-01', '2025-07-31', '_2025'),
    'VAL (2025 Aug-Dec)':   ('2025-08-01', '2025-12-31', '_2025'),
    'HOLDOUT Q1 2026':      ('2026-01-01', '2026-03-31', '_q1'),
    'HOLDOUT Apr 1-17':     ('2026-04-01', '2026-04-17', '_apr'),
}

BASELINE_CACHES = {
    '_2025': '/tmp/expVH_final/cache_V_2025.csv',
    '_q1':   '/tmp/expVH_final/cache_V_q1.csv',
    '_apr':  '/tmp/april_cache/cache.csv',
}

AUDITFIX_CACHES = {
    '_2025': '/tmp/audit_fix/cache_2025.csv',
    '_q1':   '/tmp/audit_fix/cache_q1.csv',
    '_apr':  '/tmp/audit_fix/cache_apr.csv',
}

MIN_CONV = 1.4


def load_trades(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_tier_cfg():
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    tier_cfg = cfg.get('trading', {}).get('risk_tiers', {})
    if not tier_cfg.get('enabled'):
        return []
    tiers = []
    for prefix in ['tier1', 'tier2', 'tier3']:
        mult = float(tier_cfg.get(f'{prefix}_multiplier', 0))
        if mult > 0:
            tiers.append({
                'min_price': float(tier_cfg.get(f'{prefix}_min_price', 0)),
                'max_price': float(tier_cfg.get(f'{prefix}_max_price', 999)),
                'min_volume': int(tier_cfg.get(f'{prefix}_min_volume', 0)),
                'max_volume': int(tier_cfg.get(f'{prefix}_max_volume', 999_999_999)),
                'multiplier': mult,
            })
    return tiers


def apply_stage2(trades, tiers):
    """Apply min_conv filter + tier scaling (mirrors batch_backtest.py:340-380)."""
    kept = []
    for t in trades:
        try:
            cm = float(t.get('conviction_mult') or 1.0)
        except ValueError:
            cm = 1.0
        if cm < MIN_CONV:
            continue
        try:
            ep = float(t['entry_price'])
            vol = float(t.get('avg_volume_20d') or 0)
            pnl = float(t['pnl'])
            macd_mult = float(t.get('macd_zone_mult') or 1.0)
        except (ValueError, KeyError):
            continue
        for tier in tiers:
            if (tier['min_price'] <= ep < tier['max_price']
                    and tier['min_volume'] <= vol <= tier['max_volume']):
                combined = min(3.0, cm * tier['multiplier'])
                denom = cm * macd_mult
                if denom > 0:
                    actual_scale = combined / denom
                    if abs(actual_scale - 1.0) > 0.001:
                        pnl *= actual_scale
                break
        kept.append({'date': t['date'], 'pnl': pnl, 'conv_mult': cm})
    return kept


def summarize(trades, label):
    n = len(trades)
    if not n:
        return {'label': label, 'n': 0, 'wr': 0.0, 'pnl': 0.0, 'maxdd': 0.0}
    wins = sum(1 for t in trades if t['pnl'] > 0)
    total = sum(t['pnl'] for t in trades)
    running = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: x['date']):
        running += t['pnl']
        peak = max(peak, running)
        max_dd = min(max_dd, running - peak)
    return {'label': label, 'n': n, 'wr': wins / n * 100,
            'pnl': total, 'maxdd': max_dd}


def run(caches, tiers):
    """Apply splits to a cache set and return {split_label: summary}."""
    loaded = {k: load_trades(p) for k, p in caches.items()}
    out = {}
    for split, (start, end, key) in SPLITS.items():
        rows = [t for t in loaded[key] if start <= t['date'] <= end]
        stage2 = apply_stage2(rows, tiers)
        out[split] = summarize(stage2, split)
    return out


def main():
    import os
    tiers = load_tier_cfg()
    print(f"# Audit-fix Stage-2 comparison\n")
    print(f"Config tiers loaded: {len(tiers)} active tiers\n")

    # Check that all audit_fix caches exist
    missing = [p for p in AUDITFIX_CACHES.values() if not os.path.exists(p)]
    if missing:
        print(f"MISSING caches (audit_fix rebuild still running?):")
        for m in missing:
            print(f"  - {m}")
        return 1

    baseline = run(BASELINE_CACHES, tiers)
    auditfix = run(AUDITFIX_CACHES, tiers)

    print("## Summary\n")
    print("| Split | Baseline n/WR/PnL/DD | Audit-fix n/WR/PnL/DD | ΔPnL | Δn | ΔWR |")
    print("|---|---|---|---:|---:|---:|")
    total_delta = 0.0
    for split in SPLITS:
        b = baseline[split]
        a = auditfix[split]
        dp = a['pnl'] - b['pnl']
        dn = a['n'] - b['n']
        dw = a['wr'] - b['wr']
        total_delta += dp
        print(f"| {split} "
              f"| {b['n']} / {b['wr']:.1f}% / ${b['pnl']:+,.0f} / ${b['maxdd']:+,.0f} "
              f"| {a['n']} / {a['wr']:.1f}% / ${a['pnl']:+,.0f} / ${a['maxdd']:+,.0f} "
              f"| ${dp:+,.0f} | {dn:+d} | {dw:+.1f}pt |")

    ho = sum(auditfix[s]['pnl'] - baseline[s]['pnl']
             for s in ['HOLDOUT Q1 2026', 'HOLDOUT Apr 1-17'])
    tr = auditfix['TRAIN (2025 Jan-Jul)']['pnl'] - baseline['TRAIN (2025 Jan-Jul)']['pnl']
    vl = auditfix['VAL (2025 Aug-Dec)']['pnl'] - baseline['VAL (2025 Aug-Dec)']['pnl']
    print(f"\n**TRAIN Δ**: ${tr:+,.0f}  |  **VAL Δ**: ${vl:+,.0f}  |  "
          f"**HOLDOUT Δ**: ${ho:+,.0f}  |  **Grand total Δ**: ${total_delta:+,.0f}")

    # Ship rule from plan: TRAIN >= +$2K OR lose < +$3K; VAL not lose > +$2K;
    # HOLDOUT >= +$3K.
    ship = (tr >= -3000) and (vl >= -2000) and (ho >= 3000)
    print(f"\n**Ship rule**: {'PASS — ship default ON' if ship else 'FAIL — hold flag OFF'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
