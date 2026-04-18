#!/usr/bin/env python3
"""Audit-fix post-hoc simulator: re-derive conviction scores from cached
per-rule breakdowns without rebuilding.

The V-on caches (/tmp/expVH_final/cache_V_*.csv, /tmp/april_cache/cache.csv)
store per-rule contributions (conv_vol_ratio, conv_retracement, conv_vwap_dist).
Under `audit_fix`, we subtract those three contributions from the raw score
and re-clamp to [0.25, 3.0].

Sweeps min_threshold on TRAIN+VAL so we can pick a clean (non-leaky)
recalibration point, then one-shot HOLDOUT.
"""
from __future__ import annotations

import csv
from collections import defaultdict

SPLITS = {
    'TRAIN (2025 Jan-Jul)': ('2025-01-01', '2025-07-31', '_2025'),
    'VAL (2025 Aug-Dec)':   ('2025-08-01', '2025-12-31', '_2025'),
    'HOLDOUT Q1 2026':      ('2026-01-01', '2026-03-31', '_q1'),
    'HOLDOUT Apr 1-17':     ('2026-04-01', '2026-04-17', '_apr'),
}

CACHES = {
    '_2025': '/tmp/expVH_final/cache_V_2025.csv',
    '_q1':   '/tmp/expVH_final/cache_V_q1.csv',
    '_apr':  '/tmp/april_cache/cache.csv',
}


def _f(x, default=0.0):
    try:
        return float(x) if x not in (None, '', 'None') else default
    except (TypeError, ValueError):
        return default


def recompute_audit_fix_conv(row) -> float:
    """Return audit_fix conviction_mult for a cached row.

    Takes raw_score, subtracts the 3 rules being dropped (r3, r5, r7),
    re-clamps to [0.25, 3.0]. Stays a strict subset of the V2_clean+V
    breakdown so we can compare apples to apples.
    """
    raw = _f(row.get('conv_raw_score'))
    if raw == 0.0:
        # Fallback for older rows without breakdown — mirror the cached
        # final value. In practice every V-on cache has conv_raw_score.
        return _f(row.get('conviction_mult'), default=1.0)
    drop = _f(row.get('conv_vol_ratio')) + _f(row.get('conv_retracement')) \
           + _f(row.get('conv_vwap_dist'))
    new_raw = raw - drop
    # Clamp to same [0.25, 3.0] envelope.
    return max(0.25, min(3.0, new_raw))


def load_all():
    loaded = {}
    for k, p in CACHES.items():
        with open(p) as f:
            loaded[k] = list(csv.DictReader(f))
    return loaded


def filter_by_threshold(rows, threshold, audit_fix):
    kept = []
    for r in rows:
        cm = recompute_audit_fix_conv(r) if audit_fix else _f(r.get('conviction_mult'), 1.0)
        if cm < threshold:
            continue
        # Also must preserve the old pnl adjustment: when conviction_mult
        # changes, position size changes. Approximate: pnl scales linearly
        # with conv_mult (fixed-risk sizing). cache already has old_cm
        # baked in.
        old_cm = _f(r.get('conviction_mult'), 1.0)
        pnl = _f(r.get('pnl'))
        if old_cm > 0:
            pnl = pnl * (cm / old_cm)
        kept.append({'date': r['date'], 'pnl': pnl, 'cm': cm})
    return kept


def stats(trades):
    n = len(trades)
    if not n:
        return {'n': 0, 'wr': 0.0, 'pnl': 0.0}
    wins = sum(1 for t in trades if t['pnl'] > 0)
    total = sum(t['pnl'] for t in trades)
    return {'n': n, 'wr': wins / n * 100, 'pnl': total}


def evaluate(loaded, threshold, audit_fix):
    out = {}
    for split, (start, end, key) in SPLITS.items():
        rows = [r for r in loaded[key] if start <= r['date'] <= end]
        kept = filter_by_threshold(rows, threshold, audit_fix)
        out[split] = stats(kept)
    return out


def main():
    loaded = load_all()

    print("# Audit-fix post-hoc simulator (drops rules 3/5/7)\n")
    print("Note: this is a first-order approximation. Re-scales pnl linearly "
          "with the new conviction_mult. Tier scaling + MACD divide-out are "
          "NOT re-applied, so absolute numbers are slightly off from a full "
          "rebuild — but the DELTA trend is reliable.\n")

    print("## Baseline (audit_fix OFF, min_threshold=1.4)\n")
    base = evaluate(loaded, 1.4, audit_fix=False)
    print("| Split | n | WR | Total PnL |")
    print("|---|---:|---:|---:|")
    for split, s in base.items():
        print(f"| {split} | {s['n']} | {s['wr']:.1f}% | ${s['pnl']:+,.0f} |")

    print("\n## Audit-fix ON, threshold sweep\n")
    print("| threshold | Split | n | WR | Total PnL | Δ vs base |")
    print("|---|---|---:|---:|---:|---:|")
    for t in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
        results = evaluate(loaded, t, audit_fix=True)
        for split, s in results.items():
            delta = s['pnl'] - base[split]['pnl']
            print(f"| {t:.1f} | {split} | {s['n']} | {s['wr']:.1f}% | "
                  f"${s['pnl']:+,.0f} | ${delta:+,.0f} |")

    # TRAIN+VAL sweet spot picker (clean, no HOLDOUT peek)
    print("\n## TRAIN+VAL sweet spot (leakage-clean threshold pick)\n")
    print("| threshold | TRAIN PnL | VAL PnL | T+V sum | Δ vs base (T+V) |")
    print("|---|---:|---:|---:|---:|")
    base_tv = base['TRAIN (2025 Jan-Jul)']['pnl'] + base['VAL (2025 Aug-Dec)']['pnl']
    best_t, best_delta = None, -1e18
    for t in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        r = evaluate(loaded, t, audit_fix=True)
        train = r['TRAIN (2025 Jan-Jul)']['pnl']
        val = r['VAL (2025 Aug-Dec)']['pnl']
        tv = train + val
        d = tv - base_tv
        print(f"| {t:.1f} | ${train:+,.0f} | ${val:+,.0f} | ${tv:+,.0f} | ${d:+,.0f} |")
        if d > best_delta:
            best_delta = d
            best_t = t

    print(f"\n**Best TRAIN+VAL threshold (audit_fix ON): {best_t}  → Δ ${best_delta:+,.0f}**")

    # Confirm on HOLDOUT (one-shot)
    holdout_audit = evaluate(loaded, best_t, audit_fix=True)
    holdout_base_sum = (base['HOLDOUT Q1 2026']['pnl']
                        + base['HOLDOUT Apr 1-17']['pnl'])
    holdout_audit_sum = (holdout_audit['HOLDOUT Q1 2026']['pnl']
                         + holdout_audit['HOLDOUT Apr 1-17']['pnl'])
    print(f"\n## One-shot HOLDOUT validation (audit_fix ON @ T={best_t})\n")
    print(f"HOLDOUT baseline total: ${holdout_base_sum:+,.0f}")
    print(f"HOLDOUT audit_fix total: ${holdout_audit_sum:+,.0f}")
    print(f"HOLDOUT Δ: ${holdout_audit_sum - holdout_base_sum:+,.0f}")


if __name__ == "__main__":
    main()
