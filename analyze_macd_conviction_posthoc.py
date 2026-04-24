#!/usr/bin/env python3
"""
Post-hoc validation of MACD wave conviction score using existing BT output.
Both input features (cross_time_min, vol_at_cross) are already in the CSV,
so we can compute conviction_mult directly and test filter thresholds without
re-running the 23-min signal generation.

Usage:
    python3 analyze_macd_conviction_posthoc.py
"""
import pandas as pd
import numpy as np
from datetime import datetime


def compute_conviction(cross_time_min: int, vol_at_cross: int) -> tuple[float, dict]:
    """
    Conviction score from the 2 OOS-validated features.

    Thresholds from H1'25 train bucket analysis:
      - cross_time_min quartile edges: 3, 5, 7 (lower is better)
      - vol_at_cross quartile edges: 27K, 79K, 165K (lower is better)

    Rule contributions (+0.3 for top-tier, +0.1 for second-tier, 0 else).
    Score clamped to [0.5, 2.0].
    """
    score = 1.0
    brk = {}
    # Rule 1: cross speed
    if cross_time_min <= 3:
        c = 0.3
    elif cross_time_min <= 5:
        c = 0.1
    else:
        c = 0.0
    score += c
    brk['cross_speed'] = c
    # Rule 2: vol at cross
    if vol_at_cross <= 27_000:
        c = 0.3
    elif vol_at_cross <= 79_000:
        c = 0.1
    else:
        c = 0.0
    score += c
    brk['vol_at_cross'] = c

    brk['raw_score'] = score
    final = max(0.5, min(2.0, score))
    brk['final_score'] = final
    return final, brk


def main():
    df = pd.read_csv('macd_wave_results.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Compute conviction per trade
    df['conv'] = df.apply(
        lambda r: compute_conviction(int(r['cross_time_min']), int(r['vol_at_cross']))[0],
        axis=1
    )

    # Train/test
    train_end = pd.Timestamp('2025-06-30')
    train = df[df['date'] <= train_end].copy()
    test = df[df['date'] > train_end].copy()

    print(f"Train {len(train)} trades  |  Test {len(test)} trades\n")

    # Distribution of conviction scores
    print("=" * 78)
    print("SCORE DISTRIBUTION (train, test)")
    print("=" * 78)
    counts_train = train['conv'].value_counts().sort_index()
    counts_test = test['conv'].value_counts().sort_index()
    print(f"{'Score':>7} {'Train n':>10} {'Test n':>10}")
    all_scores = sorted(set(counts_train.index) | set(counts_test.index))
    for s in all_scores:
        ct = counts_train.get(s, 0)
        ce = counts_test.get(s, 0)
        print(f"{s:>7.2f} {ct:>10} {ce:>10}")
    print()

    # EV by score bucket
    def bucket_stats(d, label):
        print(f"=" * 78)
        print(f"EV BY CONVICTION SCORE — {label} ({len(d)} trades)")
        print(f"=" * 78)
        print(f"{'Score':>7} {'n':>5} {'WR%':>6} {'Avg P&L $':>11} {'Total P&L':>12}")
        for s in sorted(d['conv'].unique()):
            sub = d[d['conv'] == s]
            wr = (sub['pnl_dollar'] > 0).mean() * 100
            avg = sub['pnl_dollar'].mean()
            tot = sub['pnl_dollar'].sum()
            print(f"{s:>7.2f} {len(sub):>5} {wr:>5.0f}% ${avg:>+10,.0f} ${tot:>+11,.0f}")
        print()

    bucket_stats(train, "TRAIN (H1'25)")
    bucket_stats(test, "TEST (H2'25 + Q1'26)")

    # Filter threshold sweep
    def filter_stats(d, label):
        print(f"=" * 78)
        print(f"FILTER THRESHOLD SWEEP — {label} ({len(d)} base trades)")
        print(f"=" * 78)
        base_pnl = d['pnl_dollar'].sum()
        base_wr = (d['pnl_dollar'] > 0).mean() * 100
        # Cumulative max DD under a chronological ordering
        d_sorted = d.sort_values(['date', 'entry_time']).reset_index(drop=True)
        eq = d_sorted['pnl_dollar'].cumsum()
        peak = eq.cummax()
        base_dd = (eq - peak).min()
        pos = d[d['pnl_dollar'] > 0]['pnl_dollar'].sum()
        neg = abs(d[d['pnl_dollar'] <= 0]['pnl_dollar'].sum())
        base_pf = pos / neg if neg else float('inf')
        print(f"Baseline: n={len(d)}, WR={base_wr:.1f}%, P&L=${base_pnl:+,.0f}, "
              f"DD=${base_dd:+,.0f}, PF={base_pf:.2f}")
        print()
        print(f"{'Thresh':>7} {'n_kept':>7} {'WR%':>6} {'P&L':>12} {'ΔP&L':>10} "
              f"{'DD':>10} {'PF':>5}")
        for t in [1.0, 1.1, 1.2, 1.3, 1.4, 1.6]:
            kept = d[d['conv'] >= t]
            if kept.empty:
                print(f"{t:>7.2f} {0:>7} (no trades remain)")
                continue
            kept_s = kept.sort_values(['date', 'entry_time']).reset_index(drop=True)
            eq_k = kept_s['pnl_dollar'].cumsum()
            peak_k = eq_k.cummax()
            dd_k = (eq_k - peak_k).min()
            wr_k = (kept['pnl_dollar'] > 0).mean() * 100
            pnl_k = kept['pnl_dollar'].sum()
            pos_k = kept[kept['pnl_dollar'] > 0]['pnl_dollar'].sum()
            neg_k = abs(kept[kept['pnl_dollar'] <= 0]['pnl_dollar'].sum())
            pf_k = pos_k / neg_k if neg_k else float('inf')
            print(f"{t:>7.2f} {len(kept):>7} {wr_k:>5.0f}% "
                  f"${pnl_k:>+11,.0f} ${pnl_k - base_pnl:>+9,.0f} "
                  f"${dd_k:>+9,.0f} {pf_k:>5.2f}")
        print()

    filter_stats(train, "TRAIN")
    filter_stats(test, "TEST")

    # Recommended threshold (from train): pick the one that maximizes train EV
    # while keeping reasonable n
    def rec_threshold(d):
        best = None
        for t in [1.0, 1.1, 1.2, 1.3, 1.4, 1.6]:
            kept = d[d['conv'] >= t]
            if len(kept) < 50:
                continue
            pnl = kept['pnl_dollar'].sum()
            avg = kept['pnl_dollar'].mean()
            # Maximize avg P&L (not total — we want quality over quantity)
            if best is None or avg > best[2]:
                best = (t, len(kept), avg, pnl)
        return best

    t_rec = rec_threshold(train)
    print(f"=" * 78)
    print(f"RECOMMENDED THRESHOLD (by train avg $/trade, n>=50)")
    print(f"=" * 78)
    if t_rec:
        thresh, n, avg, pnl = t_rec
        print(f"  Threshold: conv >= {thresh}")
        print(f"  Train @ threshold: n={n}, avg=${avg:+.0f}/trade, total=${pnl:+,.0f}")
        # Verify on test
        kept_test = test[test['conv'] >= thresh]
        print(f"  Test @ threshold (OOS):")
        print(f"    n={len(kept_test)}, "
              f"WR={(kept_test['pnl_dollar'] > 0).mean() * 100:.1f}%, "
              f"avg=${kept_test['pnl_dollar'].mean():+.0f}/trade, "
              f"total=${kept_test['pnl_dollar'].sum():+,.0f}")

    # Monthly breakdown: before vs after (FILTER approach)
    print()
    print(f"=" * 78)
    print(f"FILTER APPROACH — Monthly BEFORE vs AFTER (conv >= {t_rec[0] if t_rec else 'N/A'})")
    print(f"=" * 78)
    df['month'] = df['date'].dt.to_period('M')
    thresh = t_rec[0] if t_rec else 1.2
    print(f"{'Month':<10} "
          f"{'BEFORE n':>9} {'BEFORE P&L':>12} "
          f"{'AFTER n':>8} {'AFTER P&L':>11} "
          f"{'Δ P&L':>10}")
    total_before = total_after = 0
    for m in sorted(df['month'].unique()):
        g = df[df['month'] == m]
        g_f = g[g['conv'] >= thresh]
        pnl_b = g['pnl_dollar'].sum()
        pnl_a = g_f['pnl_dollar'].sum()
        total_before += pnl_b
        total_after += pnl_a
        print(f"{str(m):<10} "
              f"{len(g):>9} ${pnl_b:>+11,.0f} "
              f"{len(g_f):>8} ${pnl_a:>+10,.0f} "
              f"${pnl_a - pnl_b:>+9,.0f}")
    print(f"{'TOTAL':<10} "
          f"{len(df):>9} ${total_before:>+11,.0f} "
          f"{len(df[df['conv'] >= thresh]):>8} ${total_after:>+10,.0f} "
          f"${total_after - total_before:>+9,.0f}")

    # SIZING approach — position_size × conv_mult (shares scaled linearly)
    df['pnl_sized'] = df['pnl_dollar'] * df['conv']
    train['pnl_sized'] = train['pnl_dollar'] * train['conv']
    test['pnl_sized'] = test['pnl_dollar'] * test['conv']
    print()
    print(f"=" * 78)
    print(f"SIZING APPROACH — position_size × conv_mult (vs baseline $50K flat)")
    print(f"=" * 78)

    def sized_stats(d, label):
        d_s = d.sort_values(['date', 'entry_time']).reset_index(drop=True)
        base_pnl = d['pnl_dollar'].sum()
        sized_pnl = d['pnl_sized'].sum()
        base_wr = (d['pnl_dollar'] > 0).mean() * 100
        # DDs
        eq_b = d_s['pnl_dollar'].cumsum()
        pk_b = eq_b.cummax()
        dd_b = (eq_b - pk_b).min()
        eq_s = d_s['pnl_sized'].cumsum()
        pk_s = eq_s.cummax()
        dd_s = (eq_s - pk_s).min()
        # Profit factor
        pos_b = d[d['pnl_dollar'] > 0]['pnl_dollar'].sum()
        neg_b = abs(d[d['pnl_dollar'] <= 0]['pnl_dollar'].sum())
        pf_b = pos_b / neg_b if neg_b else float('inf')
        pos_s = d[d['pnl_sized'] > 0]['pnl_sized'].sum()
        neg_s = abs(d[d['pnl_sized'] <= 0]['pnl_sized'].sum())
        pf_s = pos_s / neg_s if neg_s else float('inf')
        avg_pos = d['conv'].mean() * 50000  # avg position size with sizing
        max_pos = d['conv'].max() * 50000   # max position size
        print(f"\n--- {label} ({len(d)} trades) ---")
        print(f"  avg position size (sized): ${avg_pos:,.0f}   max: ${max_pos:,.0f}")
        print(f"  BASELINE: n={len(d)}, WR={base_wr:.1f}%, P&L=${base_pnl:+,.0f}, DD=${dd_b:+,.0f}, PF={pf_b:.2f}")
        print(f"  SIZED:    n={len(d)}, WR={base_wr:.1f}%, P&L=${sized_pnl:+,.0f}, DD=${dd_s:+,.0f}, PF={pf_s:.2f}")
        print(f"  ΔP&L: ${sized_pnl - base_pnl:+,.0f} ({(sized_pnl - base_pnl) / abs(base_pnl) * 100:+.1f}%)")
        print(f"  Capital efficiency: ΔP&L / Δnotional = ${(sized_pnl - base_pnl):+,.0f} on "
              f"{((avg_pos / 50000) - 1) * 100:+.0f}% more avg notional")

    sized_stats(train, "TRAIN")
    sized_stats(test, "TEST")

    # Monthly sizing view
    print()
    print(f"=" * 78)
    print(f"SIZING APPROACH — Monthly BEFORE vs AFTER")
    print(f"=" * 78)
    print(f"{'Month':<10} "
          f"{'Base P&L':>12} "
          f"{'Sized P&L':>12} "
          f"{'Δ P&L':>10} "
          f"{'avg conv':>9}")
    tot_b = tot_s = 0
    for m in sorted(df['month'].unique()):
        g = df[df['month'] == m]
        pb = g['pnl_dollar'].sum()
        ps = g['pnl_sized'].sum()
        tot_b += pb
        tot_s += ps
        print(f"{str(m):<10} "
              f"${pb:>+11,.0f} "
              f"${ps:>+11,.0f} "
              f"${ps - pb:>+9,.0f} "
              f"{g['conv'].mean():>9.2f}")
    print(f"{'TOTAL':<10} "
          f"${tot_b:>+11,.0f} "
          f"${tot_s:>+11,.0f} "
          f"${tot_s - tot_b:>+9,.0f} "
          f"{df['conv'].mean():>9.2f}")


if __name__ == '__main__':
    main()
