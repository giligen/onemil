#!/usr/bin/env python3
"""
Full threshold sweep analysis — runs filtered BT on all threshold caches,
generates comparison matrix and detailed summary.

Usage: python3 run_full_analysis.py
"""
import csv
import os
import sys
import subprocess
from collections import defaultdict
from datetime import datetime

THRESHOLDS = [5, 8, 10, 15, 20]
CACHE_DIR = "data"
RESULTS_DIR = "analysis_results"
SUMMARY_FILE = "analysis_results/threshold_sweep_summary.md"

START = "2025-01-01"
END = "2026-03-31"


def get_cache_path(threshold):
    """Get cache file path for a threshold."""
    return f"{CACHE_DIR}/bull_flag_cache_e50_x30_t{threshold}.csv"


def run_filtered_bt(threshold):
    """Run filtered backtest for a threshold, return output CSV path."""
    cache_path = get_cache_path(threshold)
    if not os.path.exists(cache_path):
        print(f"  SKIP {threshold}% — cache not found: {cache_path}")
        return None

    output_csv = f"{RESULTS_DIR}/filtered_t{threshold}.csv"
    cmd = [
        "python3", "batch_backtest.py",
        "--start", START, "--end", END,
        "--skip-missing",
        "--cache-file", cache_path,
        "--threshold", str(threshold),
        "--output", output_csv,
    ]
    print(f"  Running filtered BT for {threshold}%...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    # Extract summary line from output
    for line in result.stdout.split("\n"):
        if "Total trades" in line or "Win rate" in line or "Total P&L" in line:
            print(f"    {line.strip()}")

    # The actual results go to backtest_results_march_2026.csv, copy them
    if os.path.exists("backtest_results_march_2026.csv"):
        import shutil
        shutil.copy("backtest_results_march_2026.csv", output_csv)
        print(f"    Saved to {output_csv}")
        return output_csv
    return None


def load_trades(csv_path):
    """Load trades from CSV."""
    trades = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            trades.append(row)
    return trades


def analyze_risk_tiers(trades):
    """Analyze trades by price tier."""
    tiers = {
        "1x ($0-10)": [],
        "2x ($10-15)": [],
        "3x ($15-20)": [],
        ">$20": [],
    }
    for t in trades:
        p = float(t["entry_price"])
        pnl = float(t["pnl"])
        if p < 10:
            tiers["1x ($0-10)"].append(pnl)
        elif p < 15:
            tiers["2x ($10-15)"].append(pnl)
        elif p < 20:
            tiers["3x ($15-20)"].append(pnl)
        else:
            tiers[">$20"].append(pnl)

    results = {}
    for tier, pnls in tiers.items():
        n = len(pnls)
        if n == 0:
            continue
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n * 100
        total = sum(pnls)
        avg = total / n
        gross_w = sum(p for p in pnls if p > 0)
        gross_l = abs(sum(p for p in pnls if p <= 0))
        pf = gross_w / gross_l if gross_l > 0 else float("inf")
        results[tier] = {
            "n": n, "wr": wr, "pnl": total, "avg": avg,
            "pf": pf, "sig": n >= 30,
        }
    return results


def analyze_time_of_day(trades):
    """Analyze trades by 30-min time bucket."""
    buckets = defaultdict(list)
    for t in trades:
        h, m = t["entry_time_et"].split(":")[:2]
        h, m = int(h), int(m)
        bucket = f"{h:02d}:{0 if m < 30 else 30:02d}"
        buckets[bucket].append(float(t["pnl"]))

    results = {}
    for b in sorted(buckets):
        pnls = buckets[b]
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n * 100
        total = sum(pnls)
        avg = total / n
        gross_w = sum(p for p in pnls if p > 0)
        gross_l = abs(sum(p for p in pnls if p <= 0))
        pf = gross_w / gross_l if gross_l > 0 else float("inf")
        results[b] = {"n": n, "wr": wr, "pnl": total, "avg": avg, "pf": pf}
    return results


def analyze_monthly(trades):
    """Analyze trades by month."""
    months = defaultdict(list)
    for t in trades:
        m = t["date"][:7]
        months[m].append(float(t["pnl"]))

    results = {}
    cum = 0
    green = 0
    red = 0
    all_monthly_pnl = []
    for m in sorted(months):
        pnls = months[m]
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n * 100 if n else 0
        total = sum(pnls)
        cum += total
        if total > 0:
            green += 1
        else:
            red += 1
        all_monthly_pnl.append(total)
        results[m] = {
            "n": n, "wr": wr, "pnl": total,
            "max_loss": min(pnls), "max_win": max(pnls), "cum": cum,
        }

    import statistics
    avg_monthly = statistics.mean(all_monthly_pnl) if all_monthly_pnl else 0
    std_monthly = statistics.stdev(all_monthly_pnl) if len(all_monthly_pnl) > 1 else 0

    return results, green, red, avg_monthly, std_monthly


def analyze_losers(trades):
    """Analyze losing trades for patterns."""
    losers = [t for t in trades if float(t["pnl"]) <= 0]
    winners = [t for t in trades if float(t["pnl"]) > 0]

    # Time distribution
    loser_hours = defaultdict(int)
    winner_hours = defaultdict(int)
    for t in losers:
        h = int(t["entry_time_et"].split(":")[0])
        loser_hours[h] += 1
    for t in winners:
        h = int(t["entry_time_et"].split(":")[0])
        winner_hours[h] += 1

    # Price distribution
    loser_prices = [float(t["entry_price"]) for t in losers]
    winner_prices = [float(t["entry_price"]) for t in winners]

    # Exit reason distribution
    loser_exits = defaultdict(int)
    winner_exits = defaultdict(int)
    for t in losers:
        loser_exits[t["exit_reason"]] += 1
    for t in winners:
        winner_exits[t["exit_reason"]] += 1

    # Top 10 worst losers
    losers_sorted = sorted(losers, key=lambda t: float(t["pnl"]))[:10]

    return {
        "loser_hours": dict(loser_hours),
        "winner_hours": dict(winner_hours),
        "loser_avg_price": sum(loser_prices) / len(loser_prices) if loser_prices else 0,
        "winner_avg_price": sum(winner_prices) / len(winner_prices) if winner_prices else 0,
        "loser_exits": dict(loser_exits),
        "winner_exits": dict(winner_exits),
        "top_losers": losers_sorted,
    }


def kelly_criterion(trades):
    """Compute Kelly criterion for position sizing."""
    wins = [float(t["pnl"]) for t in trades if float(t["pnl"]) > 0]
    losses = [abs(float(t["pnl"])) for t in trades if float(t["pnl"]) <= 0]

    if not wins or not losses:
        return 0, 0, 0

    n = len(wins) + len(losses)
    wr = len(wins) / n
    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)
    wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # Kelly = W - (1-W)/R
    kelly = wr - (1 - wr) / wl_ratio if wl_ratio > 0 else 0

    return kelly, wr, wl_ratio


def write_summary(all_results):
    """Write comprehensive summary markdown."""
    lines = []
    lines.append("# Bull Flag Threshold Sweep — Complete Analysis")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    lines.append(f"Period: {START} to {END} (15 months)")
    lines.append("\nConfig: 3x tier → 1x (killed), last_entry_time = 10:30 ET")
    lines.append("Fixes applied: V-reversal running_high/low seeding, qualification gate threshold override, mover finder threshold passthrough")
    lines.append("")

    # =========================================================
    # SECTION 1: Overall comparison
    # =========================================================
    lines.append("## 1. Overall Threshold Comparison")
    lines.append("")
    lines.append(f"| Threshold | Trades | WR% | Total P&L | Avg/Trade | PF |")
    lines.append(f"|-----------|--------|-----|-----------|-----------|-----|")

    for t in THRESHOLDS:
        r = all_results.get(t)
        if not r:
            lines.append(f"| {t}% | — | — | — | — | — |")
            continue
        trades = r["trades"]
        n = len(trades)
        wins = sum(1 for tr in trades if float(tr["pnl"]) > 0)
        wr = wins / n * 100 if n else 0
        total_pnl = sum(float(tr["pnl"]) for tr in trades)
        avg = total_pnl / n if n else 0
        gross_w = sum(float(tr["pnl"]) for tr in trades if float(tr["pnl"]) > 0)
        gross_l = abs(sum(float(tr["pnl"]) for tr in trades if float(tr["pnl"]) <= 0))
        pf = gross_w / gross_l if gross_l > 0 else 0
        lines.append(f"| {t}% | {n} | {wr:.1f}% | ${total_pnl:+,.0f} | ${avg:+,.0f} | {pf:.2f} |")

    # =========================================================
    # SECTION 2: Price Tier × Threshold Matrix
    # =========================================================
    lines.append("\n## 2. Price Tier × Threshold Matrix (P&L / PF)")
    lines.append("")

    tier_names = ["1x ($0-10)", "2x ($10-15)", "3x ($15-20)", ">$20"]
    header = "| Tier |"
    divider = "|------|"
    for t in THRESHOLDS:
        header += f" {t}% |"
        divider += "-----|"
    lines.append(header)
    lines.append(divider)

    for tier in tier_names:
        row = f"| {tier} |"
        for t in THRESHOLDS:
            r = all_results.get(t)
            if not r or tier not in r["risk_tiers"]:
                row += " — |"
                continue
            rt = r["risk_tiers"][tier]
            row += f" {rt['n']}t {rt['wr']:.0f}% PF{rt['pf']:.2f} ${rt['pnl']:+,.0f} |"
        lines.append(row)

    # WR-only matrix for readability
    lines.append("\n### Win Rate Matrix")
    lines.append("")
    header = "| Tier |"
    divider = "|------|"
    for t in THRESHOLDS:
        header += f" {t}% |"
        divider += "-----|"
    lines.append(header)
    lines.append(divider)
    for tier in tier_names:
        row = f"| {tier} |"
        for t in THRESHOLDS:
            r = all_results.get(t)
            if not r or tier not in r["risk_tiers"]:
                row += " — |"
                continue
            rt = r["risk_tiers"][tier]
            marker = "✓" if rt["sig"] else "†"
            row += f" {rt['wr']:.1f}%{marker} ({rt['n']}) |"
        lines.append(row)
    lines.append("\n✓ = 30+ trades (statistically significant), † = <30 trades")

    # =========================================================
    # SECTION 3: Time of Day × Threshold
    # =========================================================
    lines.append("\n## 3. Time-of-Day × Threshold (PF)")
    lines.append("")

    all_buckets = set()
    for t in THRESHOLDS:
        r = all_results.get(t)
        if r:
            all_buckets.update(r["time_of_day"].keys())
    all_buckets = sorted(all_buckets)

    header = "| Bucket |"
    divider = "|--------|"
    for t in THRESHOLDS:
        header += f" {t}% |"
        divider += "-----|"
    lines.append(header)
    lines.append(divider)

    for bucket in all_buckets:
        row = f"| {bucket} |"
        for t in THRESHOLDS:
            r = all_results.get(t)
            if not r or bucket not in r["time_of_day"]:
                row += " — |"
                continue
            td = r["time_of_day"][bucket]
            row += f" {td['n']}t PF{td['pf']:.2f} ${td['pnl']:+,.0f} |"
        lines.append(row)

    # =========================================================
    # SECTION 4: Monthly Consistency
    # =========================================================
    lines.append("\n## 4. Monthly Consistency")
    lines.append("")

    header = "| Metric |"
    divider = "|--------|"
    for t in THRESHOLDS:
        header += f" {t}% |"
        divider += "-----|"
    lines.append(header)
    lines.append(divider)

    metrics = ["Green months", "Red months", "Avg monthly", "Std monthly", "Sharpe-like"]
    for metric in metrics:
        row = f"| {metric} |"
        for t in THRESHOLDS:
            r = all_results.get(t)
            if not r:
                row += " — |"
                continue
            ms = r["monthly_stats"]
            if metric == "Green months":
                row += f" {ms['green']} |"
            elif metric == "Red months":
                row += f" {ms['red']} |"
            elif metric == "Avg monthly":
                row += f" ${ms['avg']:+,.0f} |"
            elif metric == "Std monthly":
                row += f" ${ms['std']:,.0f} |"
            elif metric == "Sharpe-like":
                sharpe = ms['avg'] / ms['std'] if ms['std'] > 0 else 0
                row += f" {sharpe:.2f} |"
        lines.append(row)

    # Monthly breakdown for best threshold
    lines.append("\n### Monthly P&L by Threshold")
    lines.append("")
    header = "| Month |"
    divider = "|-------|"
    for t in THRESHOLDS:
        header += f" {t}% |"
        divider += "-----|"
    lines.append(header)
    lines.append(divider)

    all_months = set()
    for t in THRESHOLDS:
        r = all_results.get(t)
        if r:
            all_months.update(r["monthly"].keys())
    for month in sorted(all_months):
        row = f"| {month} |"
        for t in THRESHOLDS:
            r = all_results.get(t)
            if not r or month not in r["monthly"]:
                row += " — |"
                continue
            m = r["monthly"][month]
            row += f" ${m['pnl']:+,.0f} ({m['n']}t) |"
        lines.append(row)

    # =========================================================
    # SECTION 5: Loser Analysis (best threshold)
    # =========================================================
    lines.append("\n## 5. Loser Analysis (20% threshold)")
    lines.append("")

    r = all_results.get(20)
    if r and "loser_analysis" in r:
        la = r["loser_analysis"]
        lines.append(f"Average loser entry price: ${la['loser_avg_price']:.2f}")
        lines.append(f"Average winner entry price: ${la['winner_avg_price']:.2f}")
        lines.append("")
        lines.append("### Exit Reason Distribution")
        lines.append("| Reason | Winners | Losers |")
        lines.append("|--------|---------|--------|")
        all_reasons = set(list(la["winner_exits"].keys()) + list(la["loser_exits"].keys()))
        for reason in sorted(all_reasons):
            w = la["winner_exits"].get(reason, 0)
            l = la["loser_exits"].get(reason, 0)
            lines.append(f"| {reason} | {w} | {l} |")

        lines.append("\n### Top 10 Worst Losers")
        lines.append("| Symbol | Date | Entry | P&L | Exit Reason |")
        lines.append("|--------|------|-------|-----|-------------|")
        for t in la["top_losers"]:
            lines.append(
                f"| {t['symbol']} | {t['date']} | ${float(t['entry_price']):.2f} @ {t['entry_time_et']} "
                f"| ${float(t['pnl']):+,.0f} | {t['exit_reason']} |"
            )

    # =========================================================
    # SECTION 6: Kelly Criterion
    # =========================================================
    lines.append("\n## 6. Kelly Criterion / Position Sizing")
    lines.append("")
    lines.append("| Threshold | Kelly% | WR | W/L Ratio | Recommended |")
    lines.append("|-----------|--------|-----|-----------|-------------|")
    for t in THRESHOLDS:
        r = all_results.get(t)
        if not r:
            continue
        kelly, wr, wlr = r["kelly"]
        half_kelly = kelly / 2 * 100
        lines.append(
            f"| {t}% | {kelly*100:.1f}% | {wr*100:.1f}% | {wlr:.2f} | "
            f"Half-Kelly: {half_kelly:.1f}% |"
        )

    # =========================================================
    # SECTION 7: Recommendations
    # =========================================================
    lines.append("\n## 7. Data-Driven Recommendations")
    lines.append("")
    lines.append("Based on 15-month analysis across 5 threshold levels:")
    lines.append("")
    lines.append("### Parameter Changes (pending user review)")
    lines.append("1. **Qualification threshold**: [fill based on data]")
    lines.append("2. **Last entry time**: 10:30 ET (already applied, validated)")
    lines.append("3. **3x tier**: Killed → 1x (already applied, validated)")
    lines.append("4. **Risk tiers**: [fill based on tier × threshold matrix]")
    lines.append("5. **Position sizing**: [fill based on Kelly analysis]")
    lines.append("")
    lines.append("### Bugs Fixed During Analysis")
    lines.append("1. V-reversal running_high/low not seeded from early bars (bar 0-4 excluded)")
    lines.append("2. --threshold not passed to qualification gate in BacktestRunner")
    lines.append("3. --threshold not passed to mover finder in monthly runner")
    lines.append("4. --skip-missing flag added for fast filtered runs")
    lines.append("5. --cache-file flag added for threshold-specific cache analysis")
    lines.append("")
    lines.append("### Parked Ideas")
    lines.append("- Leveraged ETF strategy (RGTZ, SMCX, MSOX, UVIX, etc.)")
    lines.append("- Non-leveraged high-vol ETFs (ARKK, MSOS, WEED)")
    lines.append("- Price-dependent thresholds (different threshold per price tier)")
    lines.append("- News catalyst + lower threshold combo")

    return "\n".join(lines)


def main():
    """Run complete analysis."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("  FULL THRESHOLD SWEEP ANALYSIS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Check which caches exist
    print("\nChecking cache files...")
    for t in THRESHOLDS:
        path = get_cache_path(t)
        if os.path.exists(path):
            lines = sum(1 for _ in open(path)) - 1  # subtract header
            print(f"  {t}%: {path} ({lines} trades)")
        else:
            print(f"  {t}%: MISSING")

    # Run filtered BT for each threshold
    print("\n" + "=" * 70)
    print("  RUNNING FILTERED BACKTESTS")
    print("=" * 70)

    all_results = {}
    for t in THRESHOLDS:
        cache_path = get_cache_path(t)
        if not os.path.exists(cache_path):
            print(f"\n  SKIP {t}% — no cache file")
            continue

        print(f"\n--- {t}% threshold ---")
        output_csv = run_filtered_bt(t)
        if not output_csv or not os.path.exists(output_csv):
            print(f"  FAILED — no output")
            continue

        trades = load_trades(output_csv)
        print(f"  Loaded {len(trades)} filtered trades")

        risk_tiers = analyze_risk_tiers(trades)
        time_of_day = analyze_time_of_day(trades)
        monthly, green, red, avg_m, std_m = analyze_monthly(trades)
        loser_analysis = analyze_losers(trades)
        kelly, wr, wlr = kelly_criterion(trades)

        all_results[t] = {
            "trades": trades,
            "risk_tiers": risk_tiers,
            "time_of_day": time_of_day,
            "monthly": monthly,
            "monthly_stats": {"green": green, "red": red, "avg": avg_m, "std": std_m},
            "loser_analysis": loser_analysis,
            "kelly": (kelly, wr, wlr),
        }

    # Generate summary
    print("\n" + "=" * 70)
    print("  GENERATING SUMMARY")
    print("=" * 70)

    summary = write_summary(all_results)
    with open(SUMMARY_FILE, "w") as f:
        f.write(summary)
    print(f"\nSummary written to: {SUMMARY_FILE}")

    # Also print to stdout
    print("\n" + summary)

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
