"""Analyze time-of-day edge from bull flag backtest results."""

import csv
from datetime import datetime, time
from collections import defaultdict


def parse_time(t_str):
    """Parse HH:MM:SS string to time object."""
    h, m, s = t_str.strip().split(":")
    return time(int(h), int(m), int(s))


def get_bucket(t):
    """Return 30-minute bucket label for a given time."""
    minutes_since_open = (t.hour - 9) * 60 + t.minute - 30
    bucket_idx = minutes_since_open // 30
    bucket_start_min = 9 * 60 + 30 + bucket_idx * 30
    bucket_end_min = bucket_start_min + 30
    sh, sm = divmod(bucket_start_min, 60)
    eh, em = divmod(bucket_end_min, 60)
    return f"{sh:02d}:{sm:02d}-{eh:02d}:{em:02d}", bucket_idx


def main():
    """Load CSV, bucket trades, print analysis table."""
    csv_path = "/home/ec2-user/onemil/backtest_results_march_2026.csv"

    trades = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry_time = parse_time(row["entry_time_et"])
            pnl = float(row["pnl"])
            trades.append({"entry_time": entry_time, "pnl": pnl})

    # Group by bucket
    buckets = defaultdict(list)
    for t in trades:
        label, idx = get_bucket(t["entry_time"])
        buckets[(idx, label)].append(t["pnl"])

    sorted_buckets = sorted(buckets.items(), key=lambda x: x[0][0])

    print(f"\n{'Bucket':<20} {'Trades':>7} {'Win%':>7} {'Total P&L':>12} {'Avg/Trade':>11} {'Prof Factor':>12}")
    print("-" * 72)

    bucket_stats = []
    for (idx, label), pnls in sorted_buckets:
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / len(pnls) if pnls else 0
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        bucket_stats.append((label, len(pnls), win_rate, total_pnl, avg_pnl, pf))
        pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
        print(f"{label:<20} {len(pnls):>7} {win_rate:>6.1f}% {total_pnl:>+12,.0f} {avg_pnl:>+11,.0f} {pf_str:>12}")

    # Cumulative P&L cutoff analysis
    print(f"\n{'Cut-off at end of':>22} {'Trades':>7} {'Cumul P&L':>12} {'Excluded P&L':>14} {'Excluded Trades':>16}")
    print("-" * 75)

    total_pnl_all = sum(t["pnl"] for t in trades)
    total_trades_all = len(trades)
    cumulative_pnl = 0
    cumulative_trades = 0

    for label, count, win_rate, total_pnl, avg_pnl, pf in bucket_stats:
        cumulative_pnl += total_pnl
        cumulative_trades += count
        excluded_pnl = total_pnl_all - cumulative_pnl
        excluded_trades = total_trades_all - cumulative_trades
        print(f"{label:>22} {cumulative_trades:>7} {cumulative_pnl:>+12,.0f} {excluded_pnl:>+14,.0f} {excluded_trades:>16}")

    print(f"\nTotal trades: {total_trades_all}, Total P&L: ${total_pnl_all:,.0f}")

    # Find optimal cutoff (maximize P&L per trade while keeping meaningful sample)
    print("\n--- Optimal cutoff analysis ---")
    best_avg = None
    best_label = None
    best_cumul_pnl = None
    best_n = None

    running_pnl = 0
    running_n = 0
    for label, count, win_rate, total_pnl, avg_pnl, pf in bucket_stats:
        running_pnl += total_pnl
        running_n += count
        if running_n >= 20:  # need enough trades to be meaningful
            cur_avg = running_pnl / running_n
            if best_avg is None or cur_avg > best_avg:
                best_avg = cur_avg
                best_label = label
                best_cumul_pnl = running_pnl
                best_n = running_n

    print(f"Best avg P&L/trade cutoff: end of {best_label} -> {best_n} trades, ${best_cumul_pnl:,.0f} total, ${best_avg:,.0f}/trade avg")

    # Show where edge turns negative
    print("\n--- Buckets where avg P&L/trade is negative (edge gone) ---")
    for label, count, win_rate, total_pnl, avg_pnl, pf in bucket_stats:
        if avg_pnl < 0:
            pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
            print(f"  {label}: {count} trades, avg {avg_pnl:+,.0f}/trade, WR {win_rate:.1f}%, PF {pf_str}")


if __name__ == "__main__":
    main()
