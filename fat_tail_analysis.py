#!/usr/bin/env python3
"""Fat-tail analysis: which winners drive P&L and what they have in common.

Questions answered:
  1. Pareto concentration — what % of P&L comes from top N winners?
  2. Top-10 winner feature fingerprint vs median trade
  3. Intra-trade peak R — how often do we leave 2R+ on the table?
  4. Trade-by-trade comparison vs A_f6 — what does O+TTF+D catch that A_f6
     misses, and vice versa?
  5. What DOES A_f6 do that the new stack can't replicate?

Inputs (Stage-2 filtered BT output CSVs, from shipped TTF+D config):
  /tmp/expD_cache/bt_2025_SHIP_ttf_d.csv
  /tmp/expD_cache/bt_q1_SHIP_ttf_d.csv
  /tmp/variant_runner/bt_A_f6_2025-01-01_2025-12-31.csv
"""
from __future__ import annotations
import csv
import statistics as stats
from collections import defaultdict
from pathlib import Path


SHIP_2025 = '/tmp/expD_cache/bt_2025_SHIP_ttf_d.csv'
SHIP_Q1   = '/tmp/expD_cache/bt_q1_SHIP_ttf_d.csv'
A_F6      = '/tmp/variant_runner/bt_A_f6_2025-01-01_2025-12-31.csv'


def load(path):
    return list(csv.DictReader(open(path)))


def fnum(v, default=0.0):
    try:
        return float(v) if v not in (None, '', 'None') else default
    except (ValueError, TypeError):
        return default


def fint(v, default=0):
    try:
        return int(float(v)) if v not in (None, '', 'None') else default
    except (ValueError, TypeError):
        return default


def pareto_analysis(rows, label):
    """How much P&L is concentrated in the top N winners?"""
    pnls = sorted([fnum(r['pnl']) for r in rows], reverse=True)
    total = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    print(f"\n=== {label} (n={len(rows)}) ===")
    print(f"  Total P&L:       ${total:>+9,.0f}")
    print(f"  Winners:         {len(wins)} ({len(wins)/len(rows)*100:.1f}%)   "
          f"sum=${sum(wins):+,.0f}")
    print(f"  Losers:          {len(losses)} ({len(losses)/len(rows)*100:.1f}%)  "
          f"sum=${sum(losses):+,.0f}")
    print(f"  Top 1 winner:    ${pnls[0]:+,.0f}  ({pnls[0]/sum(wins)*100:.0f}% of winners)")
    print(f"  Top 5 winners:   ${sum(pnls[:5]):+,.0f}  ({sum(pnls[:5])/sum(wins)*100:.0f}% of winners)")
    print(f"  Top 10 winners:  ${sum(pnls[:10]):+,.0f}  ({sum(pnls[:10])/sum(wins)*100:.0f}% of winners)")
    print(f"  Top 20 winners:  ${sum(pnls[:20]):+,.0f}  ({sum(pnls[:20])/sum(wins)*100:.0f}% of winners)")
    # Monster-strip: remove top N, what's left?
    for n in [1, 3, 5, 10]:
        stripped = sum(pnls[n:])
        print(f"  Strip top {n:<2}:     ${stripped:>+9,.0f}")
    return pnls


def top_winner_fingerprint(rows, top_n=10):
    """Median feature value for top N winners vs all trades."""
    winners = sorted([r for r in rows if fnum(r['pnl']) > 0],
                     key=lambda r: fnum(r['pnl']), reverse=True)
    if len(winners) < top_n:
        top_n = len(winners)
    top = winners[:top_n]
    features_to_check = [
        'conviction_mult', 'macd_zone_mult', 'qf_pole_gain_pct',
        'qf_pole_bars', 'qf_vwap_dist_pct', 'qf_gap_pct',
        'qf_fill_vwap_dist_pct', 'conv_flag_tightness', 'conv_vol_ratio',
    ]
    print(f"\n=== Feature fingerprint: top {top_n} winners vs median all trades ===")
    print(f"  {'feature':<24} {'top_med':>9} {'all_med':>9} {'delta_%':>9}")
    for f in features_to_check:
        top_vals = [fnum(r.get(f)) for r in top if r.get(f) not in (None, '', 'None')]
        all_vals = [fnum(r.get(f)) for r in rows if r.get(f) not in (None, '', 'None')]
        if not top_vals or not all_vals:
            continue
        top_med = stats.median(top_vals)
        all_med = stats.median(all_vals)
        if all_med == 0:
            delta = float('inf')
        else:
            delta = (top_med - all_med) / abs(all_med) * 100
        print(f"  {f:<24} {top_med:>+9.3f} {all_med:>+9.3f} {delta:>+8.1f}%")
    print(f"\n  Top {top_n} winners by symbol:")
    for r in top:
        print(f"    {r['symbol']:<6} {r['date']} entry={r['entry_time_et']} "
              f"P&L=${fnum(r['pnl']):>+7,.0f}  exit={r['exit_reason']}")


def pnl_distribution(rows, label):
    """Bucket trades by P&L magnitude."""
    buckets = {
        'huge_winner_>$2K': 0, 'big_winner_$1-2K': 0, 'mid_winner_$500-1K': 0,
        'small_winner_$0-500': 0,
        'small_loser_$0-500': 0, 'mid_loser_$500-1K': 0,
        'big_loser_$1-2K': 0, 'huge_loser_>$2K': 0,
    }
    pnl_sums = {k: 0.0 for k in buckets}
    for r in rows:
        p = fnum(r['pnl'])
        if p >= 2000: k = 'huge_winner_>$2K'
        elif p >= 1000: k = 'big_winner_$1-2K'
        elif p >= 500: k = 'mid_winner_$500-1K'
        elif p > 0: k = 'small_winner_$0-500'
        elif p >= -500: k = 'small_loser_$0-500'
        elif p >= -1000: k = 'mid_loser_$500-1K'
        elif p >= -2000: k = 'big_loser_$1-2K'
        else: k = 'huge_loser_>$2K'
        buckets[k] += 1
        pnl_sums[k] += p
    print(f"\n=== P&L distribution: {label} ===")
    print(f"  {'bucket':<24} {'n':>3}  {'sum':>10}  {'pct_of_total':>12}")
    total = sum(pnl_sums.values())
    for k in ['huge_winner_>$2K', 'big_winner_$1-2K', 'mid_winner_$500-1K',
              'small_winner_$0-500', 'small_loser_$0-500', 'mid_loser_$500-1K',
              'big_loser_$1-2K', 'huge_loser_>$2K']:
        pct = pnl_sums[k] / total * 100 if total != 0 else 0
        print(f"  {k:<24} {buckets[k]:>3}  ${pnl_sums[k]:>+8,.0f}  {pct:>+10.1f}%")


def missed_by_a_f6_caught_by_stack(ship, af6, label='2025'):
    """Trades O+TTF+D caught that A_f6 missed (both (sym,date,entry))."""
    ship_keys = {(r['symbol'], r['date'], r['entry_time_et']): r for r in ship}
    af6_keys = {(r['symbol'], r['date'], r['entry_time_et']): r for r in af6}

    caught_extra = [r for k, r in ship_keys.items() if k not in af6_keys]
    missed_by_stack = [r for k, r in af6_keys.items() if k not in ship_keys]

    caught_pnl = sum(fnum(r['pnl']) for r in caught_extra)
    missed_pnl = sum(fnum(r['pnl']) for r in missed_by_stack)

    print(f"\n=== {label}: trade-set diff (O+TTF+D vs A_f6 by sym/date/entry) ===")
    print(f"  In both:                   {len(ship_keys.keys() & af6_keys.keys())}")
    print(f"  Caught by stack only:      {len(caught_extra)}, P&L ${caught_pnl:+,.0f}")
    print(f"  Caught by A_f6 only:       {len(missed_by_stack)}, P&L ${missed_pnl:+,.0f}")
    print(f"  Net advantage of stack:    ${caught_pnl - missed_pnl:+,.0f}")

    # Top 5 "missed by stack" — what did A_f6 catch that we lost?
    missed_by_stack.sort(key=lambda r: fnum(r['pnl']), reverse=True)
    if missed_by_stack:
        print(f"\n  Top 5 A_f6 trades NOT in stack:")
        for r in missed_by_stack[:5]:
            print(f"    {r['symbol']:<6} {r['date']} E={r['entry_time_et']} "
                  f"P&L=${fnum(r['pnl']):>+6,.0f}  {r['exit_reason']}")


def main():
    ship_2025 = load(SHIP_2025)
    ship_q1 = load(SHIP_Q1)
    af6 = load(A_F6)

    # Part 1: Pareto concentration
    print("=" * 80)
    print("  FAT-TAIL ANALYSIS — Pareto concentration of P&L")
    print("=" * 80)
    pareto_analysis(ship_2025, 'O+TTF+D 2025 (shipping config)')
    pareto_analysis(ship_q1, 'O+TTF+D Q1 2026')
    pareto_analysis(af6, 'A_f6 2025 (prod baseline)')

    # Part 2: P&L distribution buckets
    print("\n" + "=" * 80)
    print("  P&L BUCKETING")
    print("=" * 80)
    pnl_distribution(ship_2025, 'O+TTF+D 2025')
    pnl_distribution(af6, 'A_f6 2025')

    # Part 3: Top-winner fingerprints
    print("\n" + "=" * 80)
    print("  FEATURE FINGERPRINT of fat-tail winners")
    print("=" * 80)
    top_winner_fingerprint(ship_2025, top_n=10)

    # Part 4: A_f6 vs stack diff
    print("\n" + "=" * 80)
    print("  TRADE-SET DIFF: O+TTF+D vs A_f6")
    print("=" * 80)
    missed_by_a_f6_caught_by_stack(ship_2025, af6, label='2025')


if __name__ == '__main__':
    main()
