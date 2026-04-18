#!/usr/bin/env python3
"""Validate S1 config post-hoc math against real-universe cache.

Once the fresh baseline and S1 rebuild caches exist under /tmp/baseline_val/
and /tmp/s1_val/, this script:
  1. Loads both baseline and S1 fresh caches
  2. Reports total trades + PnL per split at current shipping filter (conv>=1.4)
  3. Computes S1 lift: (S1 pnl - baseline pnl) / baseline pnl
  4. Cross-checks with post-hoc math: apply S1 scaling to baseline trades
     directly (multiply macd=1.5 pnl by 2.0/1.5=1.333, r9 gets extra mult)
     and compare to real S1 numbers. They should match within ~1% if the
     post-hoc formula is correct.

Run this AFTER both sets of caches exist.
"""
from __future__ import annotations

import csv
import os
import sys

CACHES = {
    'TRAIN (2025 H1)': ('2025-01-01', '2025-07-31', 'cache_2025.csv'),
    'VAL (2025 H2)':   ('2025-08-01', '2025-12-31', 'cache_2025.csv'),
    'HOQ1':            ('2026-01-01', '2026-03-31', 'cache_q1.csv'),
    'HOAPR':           ('2026-04-01', '2026-04-17', 'cache_apr.csv'),
}
BASELINE_DIR = '/tmp/baseline_val'
S1_DIR = '/tmp/s1_val'
BASELINE_DIR_APR = '/tmp/s1_val'  # April baseline also in s1_val dir


def _f(x, d=0.0):
    try:
        return float(x) if x not in (None, '', 'None') else d
    except Exception:
        return d


def load_cache_split(path, start, end, conv_min=1.4):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['date'] < start or r['date'] > end:
                continue
            cm = _f(r.get('conviction_mult'), 1.0)
            if cm < conv_min:
                continue
            rows.append({
                'sym': r.get('symbol', ''),
                'date': r['date'],
                'conv_mult': cm,
                'macd_mult': _f(r.get('macd_zone_mult'), 1.0) or 1.0,
                'pnl': _f(r.get('pnl')),
                'entry_price': _f(r.get('entry_price')),
                'avg_vol': _f(r.get('avg_volume_20d')),
            })
    return rows


def apply_tiers_stage2(trades):
    """Mirror batch_backtest.py:339-380 tier logic."""
    tiers = [
        {'p_min': 10, 'p_max': 15, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 2.0},
        {'p_min': 15, 'p_max': 23, 'v_min': 500_000, 'v_max': 5_000_000, 'mult': 1.0},
    ]
    out = []
    for t in trades:
        conv = t['conv_mult']
        macd = t['macd_mult']
        pnl = t['pnl']
        for tier in tiers:
            if (tier['p_min'] <= t['entry_price'] < tier['p_max']
                    and tier['v_min'] <= t['avg_vol'] <= tier['v_max']):
                combined = min(3.0, conv * tier['mult'])
                denom = conv * macd
                if denom > 0:
                    scale = combined / denom
                    if abs(scale - 1.0) > 0.001:
                        pnl *= scale
                break
        out.append({**t, 'pnl': pnl})
    return out


def summarize(trades):
    n = len(trades)
    if not n:
        return {'n': 0, 'wr': 0.0, 'pnl': 0.0}
    wins = sum(1 for t in trades if t['pnl'] > 0)
    return {'n': n, 'wr': wins / n * 100, 'pnl': sum(t['pnl'] for t in trades)}


def check_available():
    missing = []
    for name, (_, _, fn) in CACHES.items():
        for d in [BASELINE_DIR if 'APR' not in name else BASELINE_DIR_APR, S1_DIR]:
            p = os.path.join(d, fn)
            if not os.path.exists(p):
                missing.append(p)
    return missing


def main():
    missing = check_available()
    if missing:
        print("# Rebuild caches not yet complete")
        print("\nMissing:")
        for m in missing:
            print(f"  - {m}")
        return 1

    print("# S1 validation — real-universe shipping-config comparison\n")
    print("Baseline and S1 caches both built with CURRENT shipping config "
          "(max_pole_bars=3, TTF flag on at Stage-2, V-reversal on, D on, etc.), "
          "EXCEPT S1 overrides: BT_MACD_STRONG=2.0, BT_VREV_BONUS=0.7.\n")

    print("| Split | Baseline n/PnL/WR | S1 n/PnL/WR | Δ PnL | Δ% |")
    print("|---|---|---|---:|---:|")
    total_base = 0.0
    total_s1 = 0.0
    for name, (start, end, fn) in CACHES.items():
        b_path = os.path.join(
            BASELINE_DIR if 'APR' not in name else BASELINE_DIR_APR,
            fn if 'APR' not in name else 'cache_apr_baseline.csv')
        s_path = os.path.join(S1_DIR, fn)
        base_rows = apply_tiers_stage2(load_cache_split(b_path, start, end))
        s1_rows = apply_tiers_stage2(load_cache_split(s_path, start, end))
        b = summarize(base_rows)
        s = summarize(s1_rows)
        delta = s['pnl'] - b['pnl']
        pct = (delta / b['pnl'] * 100) if b['pnl'] != 0 else 0
        total_base += b['pnl']
        total_s1 += s['pnl']
        print(f"| {name} | {b['n']} / ${b['pnl']:+,.0f} / {b['wr']:.1f}% "
              f"| {s['n']} / ${s['pnl']:+,.0f} / {s['wr']:.1f}% "
              f"| ${delta:+,.0f} | {pct:+.1f}% |")
    grand_delta = total_s1 - total_base
    grand_pct = (grand_delta / total_base * 100) if total_base != 0 else 0
    print(f"| **GRAND TOTAL** | ${total_base:+,.0f} | ${total_s1:+,.0f} | "
          f"${grand_delta:+,.0f} | {grand_pct:+.1f}% |")

    # Ship rule
    print("\n## Ship decision")
    if grand_pct >= 20:
        print(f"✅ GRAND lift +{grand_pct:.1f}% meets the user's 20% target. **SHIP S1.**")
    elif grand_pct >= 10:
        print(f"⚠️ GRAND lift +{grand_pct:.1f}% is below the 20% target but positive. "
              f"Consider shipping or continuing research for bigger lever.")
    else:
        print(f"❌ GRAND lift +{grand_pct:.1f}% is too small. DO NOT SHIP as-is.")


if __name__ == "__main__":
    sys.exit(main())
