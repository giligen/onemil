"""Monthly consistency analysis for bull flag backtest results."""
import csv
import sys
from collections import defaultdict
from statistics import mean, stdev

CSV_PATH = "/home/ec2-user/onemil/backtest_results_march_2026.csv"


def load_trades(path):
    """Load trades from CSV, return list of dicts."""
    trades = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["pnl"] = float(row["pnl"])
            trades.append(row)
    return trades


def group_by_month(trades):
    """Group trades by YYYY-MM key."""
    monthly = defaultdict(list)
    for t in trades:
        month_key = t["date"][:7]  # YYYY-MM
        monthly[month_key].append(t)
    return dict(sorted(monthly.items()))


def compute_month_stats(month_trades):
    """Compute stats for a single month's trades."""
    pnls = [t["pnl"] for t in month_trades]
    wins = [p for p in pnls if p > 0]
    total = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    max_loss = min(pnls) if pnls else 0
    best_win = max(pnls) if pnls else 0
    return {
        "count": len(pnls),
        "win_rate": win_rate,
        "total_pnl": total,
        "max_loss": max_loss,
        "best_win": best_win,
    }


def main():
    """Run monthly consistency analysis and print summary tables."""
    trades = load_trades(CSV_PATH)
    monthly = group_by_month(trades)

    print(f"\nTotal trades loaded: {len(trades)}")
    print(f"Months covered: {min(monthly.keys())} to {max(monthly.keys())}\n")

    # --- Per-month table ---
    header = f"{'Month':<10} {'Trades':>6} {'WR%':>6} {'P&L':>10} {'MaxLoss':>10} {'BestWin':>10} {'CumPnL':>12}"
    print(header)
    print("-" * len(header))

    cumulative = 0.0
    month_stats_list = []
    for month, month_trades in monthly.items():
        s = compute_month_stats(month_trades)
        cumulative += s["total_pnl"]
        s["month"] = month
        month_stats_list.append(s)
        print(
            f"{month:<10} {s['count']:>6} {s['win_rate']:>5.1f}% "
            f"{s['total_pnl']:>10,.0f} {s['max_loss']:>10,.0f} "
            f"{s['best_win']:>10,.0f} {cumulative:>12,.0f}"
        )

    # --- Summary stats ---
    pnls = [s["total_pnl"] for s in month_stats_list]
    green_months = [s for s in month_stats_list if s["total_pnl"] > 0]
    red_months = [s for s in month_stats_list if s["total_pnl"] <= 0]
    best = max(month_stats_list, key=lambda s: s["total_pnl"])
    worst = min(month_stats_list, key=lambda s: s["total_pnl"])
    avg_pnl = mean(pnls)
    std_pnl = stdev(pnls) if len(pnls) > 1 else 0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total months     : {len(month_stats_list)}")
    print(f"Green months     : {len(green_months)}  ({len(green_months)/len(month_stats_list)*100:.0f}%)")
    print(f"Red months       : {len(red_months)}  ({len(red_months)/len(month_stats_list)*100:.0f}%)")
    print(f"Best month       : {best['month']}  ${best['total_pnl']:,.0f}")
    print(f"Worst month      : {worst['month']}  ${worst['total_pnl']:,.0f}")
    print(f"Avg monthly P&L  : ${avg_pnl:,.0f}")
    print(f"Std dev monthly  : ${std_pnl:,.0f}")
    print(f"Total P&L        : ${sum(pnls):,.0f}")
    print(f"Sharpe-like ratio: {avg_pnl/std_pnl:.2f}  (monthly mean/std)")

    # --- Consecutive streaks ---
    streak_green = streak_red = 0
    cur_green = cur_red = 0
    for s in month_stats_list:
        if s["total_pnl"] > 0:
            cur_green += 1
            cur_red = 0
        else:
            cur_red += 1
            cur_green = 0
        streak_green = max(streak_green, cur_green)
        streak_red = max(streak_red, cur_red)
    print(f"\nMax consec green : {streak_green} months")
    print(f"Max consec red   : {streak_red} months")


if __name__ == "__main__":
    main()
