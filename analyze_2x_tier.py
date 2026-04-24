"""
Analyze 2x tier ($10-15 entry price) trades from bull flag backtest results.
Compares winners vs losers on available data columns.
Note: daily_range_pct and avg_volume_20d are empty in this CSV — derived proxies used instead.
"""

import pandas as pd
import numpy as np
import sys

CSV_PATH = "/home/ec2-user/onemil/backtest_results_march_2026.csv"
PRICE_LOW = 10.0
PRICE_HIGH = 15.0


def parse_entry_hour(entry_time_str):
    """Extract decimal hour from HH:MM:SS string."""
    try:
        h, m, s = str(entry_time_str).split(":")
        return int(h) + int(m) / 60.0
    except Exception:
        return np.nan


def parse_exit_hour(exit_time_str):
    """Extract decimal hour from HH:MM:SS string."""
    return parse_entry_hour(exit_time_str)


def bucket_entry_time(hour_float):
    """Bucket entry time into named windows."""
    if np.isnan(hour_float):
        return "unknown"
    if hour_float < 9.75:
        return "pre-9:45"
    elif hour_float < 10.0:
        return "9:45-10:00"
    elif hour_float < 10.5:
        return "10:00-10:30"
    elif hour_float < 11.0:
        return "10:30-11:00"
    elif hour_float < 12.0:
        return "11:00-12:00"
    else:
        return "12:00+"


def stats(series, label):
    """Print descriptive stats for a numeric series."""
    clean = series.dropna()
    if len(clean) == 0:
        print(f"  {label}: no data")
        return
    print(f"  {label}: n={len(clean)}, mean={clean.mean():.2f}, "
          f"median={clean.median():.2f}, std={clean.std():.2f}, "
          f"min={clean.min():.2f}, max={clean.max():.2f}")


def pct_fmt(n, total):
    """Format count as 'n (x%)' string."""
    if total == 0:
        return "0 (0%)"
    return f"{n} ({100*n/total:.0f}%)"


def print_winrate_by_group(tier, col, label):
    """Print win rate and avg P&L for each value in a categorical column."""
    print(f"\n  Win rate by {label}:")
    counts = tier[col].value_counts().sort_index()
    for val in counts.index:
        grp = tier[tier[col] == val]
        wr = 100 * grp["is_winner"].mean()
        avg_pnl = grp["pnl"].mean()
        total_pnl = grp["pnl"].sum()
        print(f"    {str(val):<20} n={len(grp):>3}, WR={wr:>4.0f}%, "
              f"avg_pnl=${avg_pnl:>7,.0f}, total_pnl=${total_pnl:>9,.0f}")


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Total trades loaded: {len(df)}")

    # Filter to 2x tier
    tier = df[(df["entry_price"] >= PRICE_LOW) & (df["entry_price"] <= PRICE_HIGH)].copy()
    print(f"2x tier ($10-$15) trades: {len(tier)}")

    if tier.empty:
        print("No trades in $10-$15 range. Exiting.")
        sys.exit(1)

    # --- Derived columns ---
    tier["entry_hour"] = tier["entry_time_et"].apply(parse_entry_hour)
    tier["exit_hour"] = tier["exit_time_et"].apply(parse_exit_hour)
    tier["time_bucket"] = tier["entry_hour"].apply(bucket_entry_time)

    # Hold time in minutes
    tier["hold_minutes"] = (tier["exit_hour"] - tier["entry_hour"]) * 60.0

    # Stop distance as % of entry (risk per share %)
    tier["stop_dist_pct"] = 100.0 * (tier["entry_price"] - tier["stop_loss"]) / tier["entry_price"]

    # Target distance as % of entry
    tier["target_dist_pct"] = 100.0 * (tier["target"] - tier["entry_price"]) / tier["entry_price"]

    # Stated R:R ratio from setup
    tier["setup_rr"] = tier["target_dist_pct"] / tier["stop_dist_pct"].replace(0, np.nan)

    # Dollar risk per trade (shares * stop_dist)
    tier["dollar_risk"] = tier["shares"] * (tier["entry_price"] - tier["stop_loss"])

    # Whether a partial was taken (proxy for strength — stock ran enough to take partial)
    tier["partial_taken"] = tier["partial_taken"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False)

    tier["is_winner"] = tier["pnl"] > 0

    winners = tier[tier["is_winner"]]
    losers = tier[~tier["is_winner"]]

    print(f"\n{'='*65}")
    print(f"OVERVIEW")
    print(f"  WINNERS: {len(winners)}  |  LOSERS: {len(losers)}  |  Win rate: {100*len(winners)/len(tier):.1f}%")
    print(f"  Total P&L:          ${tier['pnl'].sum():>10,.0f}")
    print(f"  Winners total:      ${winners['pnl'].sum():>10,.0f}  avg ${winners['pnl'].mean():>8,.0f}")
    print(f"  Losers total:       ${losers['pnl'].sum():>10,.0f}  avg ${losers['pnl'].mean():>8,.0f}")
    print(f"  Profit factor:      {abs(winners['pnl'].sum()) / max(abs(losers['pnl'].sum()), 1):.2f}")

    # Compare vs full dataset
    full_wr = 100 * (df["pnl"] > 0).mean()
    full_avg = df["pnl"].mean()
    print(f"\n  Full dataset (all tiers):  WR={full_wr:.1f}%, avg_pnl=${full_avg:,.0f}")
    print(f"  $10-15 tier:               WR={100*winners.shape[0]/len(tier):.1f}%, avg_pnl=${tier['pnl'].mean():,.0f}")

    # --- 1. Entry time distribution ---
    print(f"\n{'='*65}")
    print("1. ENTRY TIME DISTRIBUTION")
    time_buckets = ["pre-9:45", "9:45-10:00", "10:00-10:30", "10:30-11:00", "11:00-12:00", "12:00+"]
    w_time = winners["time_bucket"].value_counts()
    l_time = losers["time_bucket"].value_counts()
    print(f"  {'Bucket':<18} {'Winners':>10} {'Losers':>10} {'WR%':>6} {'AvgPnL':>10}")
    for b in time_buckets:
        wc = w_time.get(b, 0)
        lc = l_time.get(b, 0)
        tot = wc + lc
        if tot == 0:
            continue
        wr = 100 * wc / tot
        grp = tier[tier["time_bucket"] == b]
        avg_pnl = grp["pnl"].mean()
        print(f"  {b:<18} {pct_fmt(wc, len(winners)):>10} {pct_fmt(lc, len(losers)):>10} "
              f"{wr:>5.0f}%  ${avg_pnl:>8,.0f}")

    # Morning edge breakout
    morning = tier[tier["entry_hour"] < 10.5]
    late = tier[tier["entry_hour"] >= 10.5]
    print(f"\n  Before 10:30: n={len(morning)}, WR={100*morning['is_winner'].mean():.0f}%, "
          f"avg_pnl=${morning['pnl'].mean():,.0f}, total=${morning['pnl'].sum():,.0f}")
    if len(late):
        print(f"  After  10:30: n={len(late)}, WR={100*late['is_winner'].mean():.0f}%, "
              f"avg_pnl=${late['pnl'].mean():,.0f}, total=${late['pnl'].sum():,.0f}")

    # --- 2. Hold time ---
    print(f"\n{'='*65}")
    print("2. HOLD TIME (minutes)")
    stats(winners["hold_minutes"], "Winners")
    stats(losers["hold_minutes"],  "Losers ")

    # Hold time buckets
    hold_cuts = [0, 5, 15, 30, 60, 10000]
    hold_labels = ["<5m", "5-15m", "15-30m", "30-60m", "60m+"]
    tier["hold_bucket"] = pd.cut(tier["hold_minutes"], bins=hold_cuts, labels=hold_labels)
    print_winrate_by_group(tier, "hold_bucket", "hold time")

    # --- 3. Stop distance (risk per trade) ---
    print(f"\n{'='*65}")
    print("3. STOP DISTANCE (risk % from entry)")
    stats(winners["stop_dist_pct"], "Winners")
    stats(losers["stop_dist_pct"],  "Losers ")

    # Stop distance buckets
    stop_cuts = [0, 2, 3, 4, 5, 100]
    stop_labels = ["<2%", "2-3%", "3-4%", "4-5%", "5%+"]
    tier["stop_bucket"] = pd.cut(tier["stop_dist_pct"], bins=stop_cuts, labels=stop_labels)
    print_winrate_by_group(tier, "stop_bucket", "stop distance")

    # --- 4. Setup R:R ratio ---
    print(f"\n{'='*65}")
    print("4. SETUP R:R RATIO (target_dist / stop_dist)")
    stats(winners["setup_rr"], "Winners")
    stats(losers["setup_rr"],  "Losers ")

    # R:R buckets
    rr_cuts = [0, 1.0, 1.5, 2.0, 3.0, 100]
    rr_labels = ["<1R", "1-1.5R", "1.5-2R", "2-3R", "3R+"]
    tier["rr_bucket"] = pd.cut(tier["setup_rr"], bins=rr_cuts, labels=rr_labels)
    print_winrate_by_group(tier, "rr_bucket", "setup R:R")

    # --- 5. Dollar risk per trade ---
    print(f"\n{'='*65}")
    print("5. DOLLAR RISK PER TRADE ($shares * stop_dist)")
    stats(winners["dollar_risk"], "Winners")
    stats(losers["dollar_risk"],  "Losers ")

    # --- 6. pnl_pct spread ---
    print(f"\n{'='*65}")
    print("6. PNL_PCT SPREAD")
    stats(winners["pnl_pct"], "Winners pnl_pct")
    stats(losers["pnl_pct"],  "Losers pnl_pct ")

    # Percentile breakdowns
    for p in [10, 25, 75, 90]:
        print(f"  P{p}: winners={np.percentile(winners['pnl_pct'], p):.1f}%  "
              f"losers={np.percentile(losers['pnl_pct'], p):.1f}%")

    # --- 7. Exit reason breakdown ---
    print(f"\n{'='*65}")
    print("7. EXIT REASON BREAKDOWN")
    all_reasons = sorted(tier["exit_reason"].unique())
    print(f"  {'Reason':<20} {'All':>5} {'Win%':>6} {'WR%':>6} {'AvgPnL':>10} {'TotalPnL':>12}")
    for r in all_reasons:
        a = tier[tier["exit_reason"] == r]
        w = winners[winners["exit_reason"] == r]
        wr = 100 * len(w) / len(a) if len(a) > 0 else 0
        avg_pnl = a["pnl"].mean()
        total_pnl = a["pnl"].sum()
        print(f"  {r:<20} {len(a):>5} {pct_fmt(len(w), len(winners)):>8}  "
              f"{wr:>5.0f}%  ${avg_pnl:>8,.0f}  ${total_pnl:>10,.0f}")

    # --- 8. Partial taken ---
    print(f"\n{'='*65}")
    print("8. PARTIAL TAKEN (proxy for stock running strongly)")
    for partial in [True, False]:
        grp = tier[tier["partial_taken"] == partial]
        if len(grp) == 0:
            continue
        wr = 100 * grp["is_winner"].mean()
        print(f"  partial_taken={partial}: n={len(grp)}, WR={wr:.0f}%, "
              f"avg_pnl=${grp['pnl'].mean():,.0f}, total_pnl=${grp['pnl'].sum():,.0f}")

    # --- 9. Entry price sub-ranges ---
    print(f"\n{'='*65}")
    print("9. ENTRY PRICE SUB-RANGES (within $10-15)")
    price_cuts = [10, 11, 12, 13, 14, 15.01]
    price_labels = ["$10-11", "$11-12", "$12-13", "$13-14", "$14-15"]
    tier["price_bucket"] = pd.cut(tier["entry_price"], bins=price_cuts, labels=price_labels)
    print_winrate_by_group(tier, "price_bucket", "entry price range")

    # --- 10. Full sorted trade list ---
    print(f"\n{'='*65}")
    print("10. ALL 2X TIER TRADES SORTED BY P&L (desc)")
    hdr = f"  {'Symbol':<7} {'Date':<12} {'Time':<10} {'Entry$':>7} {'Stop$':>7} {'Target$':>8} {'Shares':>6} {'PnL':>9} {'PnL%':>6} {'HoldMin':>8} {'Exit Reason':<18} {'Partial'}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    sorted_trades = tier.sort_values("pnl", ascending=False)
    for _, row in sorted_trades.iterrows():
        symbol = str(row["symbol"]) if pd.notna(row["symbol"]) else "N/A"
        print(f"  {symbol:<7} {str(row['date']):<12} {str(row['entry_time_et']):<10} "
              f"${row['entry_price']:>6.2f} ${row['stop_loss']:>6.2f} ${row['target']:>7.2f} "
              f"{int(row['shares']):>6}  ${row['pnl']:>8,.0f} {row['pnl_pct']:>6.1f}% "
              f"{row['hold_minutes']:>7.0f}m  {str(row['exit_reason']):<18} {str(row['partial_taken'])}")

    # --- 11. Pattern summary ---
    print(f"\n{'='*65}")
    print("11. PATTERN SUMMARY — KEY DIFFERENTIATORS")

    # Morning + tight stop combo
    for time_thresh, time_label in [(10.5, "before 10:30"), (10.0, "before 10:00")]:
        for stop_thresh in [3.0, 4.0]:
            combo = tier[(tier["entry_hour"] < time_thresh) & (tier["stop_dist_pct"] <= stop_thresh)]
            if len(combo) >= 5:
                print(f"  {time_label} AND stop<={stop_thresh}%: n={len(combo)}, "
                      f"WR={100*combo['is_winner'].mean():.0f}%, "
                      f"avg_pnl=${combo['pnl'].mean():,.0f}, "
                      f"total_pnl=${combo['pnl'].sum():,.0f}")

    # High R:R setups
    high_rr = tier[tier["setup_rr"] >= 2.0]
    if len(high_rr) >= 3:
        print(f"  Setup R:R >= 2.0: n={len(high_rr)}, WR={100*high_rr['is_winner'].mean():.0f}%, "
              f"avg_pnl=${high_rr['pnl'].mean():,.0f}")

    low_rr = tier[tier["setup_rr"] < 2.0]
    if len(low_rr) >= 3:
        print(f"  Setup R:R <  2.0: n={len(low_rr)}, WR={100*low_rr['is_winner'].mean():.0f}%, "
              f"avg_pnl=${low_rr['pnl'].mean():,.0f}")

    # Best combo: morning + high R:R
    best = tier[(tier["entry_hour"] < 10.5) & (tier["setup_rr"] >= 2.0)]
    if len(best) >= 3:
        print(f"  BEST COMBO (before 10:30 AND R:R>=2): n={len(best)}, "
              f"WR={100*best['is_winner'].mean():.0f}%, "
              f"avg_pnl=${best['pnl'].mean():,.0f}, "
              f"total_pnl=${best['pnl'].sum():,.0f}")

    worst = tier[(tier["entry_hour"] >= 10.5) | (tier["setup_rr"] < 1.5)]
    if len(worst) >= 3:
        print(f"  WORST COMBO (after 10:30 OR R:R<1.5): n={len(worst)}, "
              f"WR={100*worst['is_winner'].mean():.0f}%, "
              f"avg_pnl=${worst['pnl'].mean():,.0f}, "
              f"total_pnl=${worst['pnl'].sum():,.0f}")

    print()


if __name__ == "__main__":
    main()
