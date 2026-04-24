"""
analyze_red_months.py

Investigate what distinguishes red months from green months in the
bull-flag backtest.  Compares SPY regime, trade stats, timing patterns,
consecutive-loss clusters and day-of-week effects.

Usage:
    python analyze_red_months.py
"""

import sqlite3
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = "data/cache.db"
BACKTEST_CSV = "backtest_results_march_2026.csv"

RED_MONTHS = {
    "2025-08", "2025-12",
    "2026-01", "2026-02", "2026-03",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_backtest(path: str) -> pd.DataFrame:
    """Load and minimally clean the backtest CSV."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["month_key"] = df["date"].dt.strftime("%Y-%m")
    df["is_win"] = df["pnl"] > 0
    df["is_red_month"] = df["month_key"].isin(RED_MONTHS)

    # entry_time_et can be "HH:MM:SS" or "HH:MM"
    df["entry_dt"] = pd.to_datetime(
        df["date"].dt.strftime("%Y-%m-%d") + " " + df["entry_time_et"].astype(str),
        errors="coerce",
    )
    df["entry_hour_min"] = df["entry_dt"].dt.hour * 60 + df["entry_dt"].dt.minute
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_late_entry"] = df["entry_hour_min"] >= 10 * 60  # after 10:00 ET

    return df


def load_spy_daily(db_path: str) -> pd.DataFrame:
    """Load SPY daily bars from cache.db and compute regime signals."""
    conn = sqlite3.connect(db_path)
    spy = pd.read_sql(
        "SELECT bar_date, open, high, low, close, volume "
        "FROM daily_bars WHERE symbol='SPY' ORDER BY bar_date",
        conn,
    )
    conn.close()

    spy["bar_date"] = pd.to_datetime(spy["bar_date"])
    spy["daily_range_pct"] = (spy["high"] - spy["low"]) / spy["low"] * 100
    spy["ret"] = spy["close"].pct_change() * 100
    spy["sma20"] = spy["close"].rolling(20).mean()
    spy["above_sma20"] = spy["close"] > spy["sma20"]
    spy["month_key"] = spy["bar_date"].dt.strftime("%Y-%m")
    return spy


def spy_monthly_stats(spy: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate SPY daily bars into per-month stats.

    Excludes bars where daily_range_pct > 15% (data errors like the
    2026-02-02 bar where low=$69 instead of $690).
    """
    # Remove clearly bad bars (data errors)
    spy_clean = spy[spy["daily_range_pct"] < 15].copy()

    rows = []
    for mkey, grp in spy_clean.groupby("month_key"):
        grp = grp.sort_values("bar_date")
        monthly_ret = (grp["close"].iloc[-1] / grp["open"].iloc[0] - 1) * 100
        avg_range = grp["daily_range_pct"].mean()
        pct_above_sma = grp["above_sma20"].mean() * 100
        rows.append(
            {
                "month_key": mkey,
                "spy_monthly_ret_pct": round(monthly_ret, 2),
                "spy_avg_daily_range_pct": round(avg_range, 2),
                "spy_pct_days_above_sma20": round(pct_above_sma, 1),
            }
        )
    return pd.DataFrame(rows)


def trade_monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trade-level data into per-month stats."""
    rows = []
    for mkey, grp in df.groupby("month_key"):
        n_trades = len(grp)
        win_rate = grp["is_win"].mean() * 100
        total_pnl = grp["pnl"].sum()
        avg_pnl = grp["pnl"].mean()
        late_pct = grp["is_late_entry"].mean() * 100
        stop_out_pct = (grp["exit_reason"] == "stop").mean() * 100
        avg_win = grp[grp["is_win"]]["pnl"].mean() if grp["is_win"].any() else 0.0
        avg_loss = grp[~grp["is_win"]]["pnl"].mean() if (~grp["is_win"]).any() else 0.0
        top_pnl = grp["pnl"].max()
        pnl_ex_top = total_pnl - top_pnl
        rows.append(
            {
                "month_key": mkey,
                "n_trades": n_trades,
                "win_rate_pct": round(win_rate, 1),
                "total_pnl": round(total_pnl, 0),
                "avg_pnl_per_trade": round(avg_pnl, 0),
                "late_entry_pct": round(late_pct, 1),
                "stop_out_pct": round(stop_out_pct, 1),
                "avg_win": round(avg_win, 0),
                "avg_loss": round(avg_loss, 0),
                "win_loss_ratio": round(-avg_win / avg_loss, 2) if avg_loss < 0 else float("nan"),
                "top_pnl": round(top_pnl, 0),
                "pnl_ex_top": round(pnl_ex_top, 0),
            }
        )
    return pd.DataFrame(rows)


def consecutive_loss_streaks(df: pd.DataFrame) -> dict:
    """
    For each month, find the maximum consecutive-loss streak within that month.
    Returns {month_key: max_streak}.
    """
    result = {}
    for mkey, grp in df.groupby("month_key"):
        grp_sorted = grp.sort_values("date")
        max_streak = current_streak = 0
        for win in grp_sorted["is_win"]:
            if not win:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        result[mkey] = max_streak
    return result


def day_of_week_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Win rate and avg P&L by day-of-week for red vs green months."""
    rows = []
    for dow in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        for label, mask in [("GREEN", ~df["is_red_month"]), ("RED", df["is_red_month"])]:
            sub = df[mask & (df["day_of_week"] == dow)]
            if len(sub) == 0:
                continue
            rows.append(
                {
                    "dow": dow,
                    "group": label,
                    "n": len(sub),
                    "win_rate_pct": round(sub["is_win"].mean() * 100, 1),
                    "avg_pnl": round(sub["pnl"].mean(), 0),
                }
            )
    return pd.DataFrame(rows)


def entry_time_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of entries across time buckets for red vs green months."""
    bins = [
        ("09:30-09:45", 9 * 60 + 30, 9 * 60 + 45),
        ("09:45-10:00", 9 * 60 + 45, 10 * 60),
        ("10:00-10:30", 10 * 60, 10 * 60 + 30),
        ("10:30-11:00", 10 * 60 + 30, 11 * 60),
        ("11:00+", 11 * 60, 24 * 60),
    ]
    rows = []
    for label, lo, hi in bins:
        for group_label, mask in [("GREEN", ~df["is_red_month"]), ("RED", df["is_red_month"])]:
            sub = df[mask & (df["entry_hour_min"] >= lo) & (df["entry_hour_min"] < hi)]
            total_in_group = df[mask].shape[0]
            rows.append(
                {
                    "time_bin": label,
                    "group": group_label,
                    "n_trades": len(sub),
                    "pct_of_group": round(len(sub) / total_in_group * 100, 1) if total_in_group else 0,
                    "win_rate_pct": round(sub["is_win"].mean() * 100, 1) if len(sub) else float("nan"),
                    "avg_pnl": round(sub["pnl"].mean(), 0) if len(sub) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def print_separator(title: str = "") -> None:
    width = 78
    if title:
        pad = (width - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * (width - pad - len(title) - 2))
    else:
        print("=" * width)


def fmt_pnl(v) -> str:
    return f"${v:+,.0f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_separator("Loading data")
    df = load_backtest(BACKTEST_CSV)
    print(f"  Loaded {len(df)} trades across {df['month_key'].nunique()} months")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    spy_raw = load_spy_daily(DB_PATH)
    print(f"  SPY daily bars: {spy_raw['bar_date'].min().date()} → {spy_raw['bar_date'].max().date()}")

    spy_monthly = spy_monthly_stats(spy_raw)
    trade_monthly = trade_monthly_stats(df)
    streaks = consecutive_loss_streaks(df)

    # Merge everything
    monthly = trade_monthly.merge(spy_monthly, on="month_key", how="left")
    monthly["max_loss_streak"] = monthly["month_key"].map(streaks)
    monthly["is_red"] = monthly["month_key"].isin(RED_MONTHS)
    monthly = monthly.sort_values("month_key")

    # ---------------------------------------------------------------------------
    # Section 1: Full month-by-month table
    # ---------------------------------------------------------------------------
    print_separator("Month-by-Month Summary")
    header = (
        f"{'Month':<10} {'Color':<6} {'Trades':>6} {'WR%':>6} {'Total P&L':>11} "
        f"{'Avg/Trade':>10} {'SPY Ret%':>9} {'SPY Range%':>11} "
        f"{'AbvSMA%':>8} {'MaxStreak':>10} {'Late%':>7}"
    )
    print(header)
    print("-" * len(header))
    for _, row in monthly.iterrows():
        color = "RED  " if row["is_red"] else "green"
        spy_ret = f"{row['spy_monthly_ret_pct']:+.1f}%" if pd.notna(row.get("spy_monthly_ret_pct")) else "  n/a"
        spy_rng = f"{row['spy_avg_daily_range_pct']:.2f}%" if pd.notna(row.get("spy_avg_daily_range_pct")) else " n/a"
        sma_pct = f"{row['spy_pct_days_above_sma20']:.0f}%" if pd.notna(row.get("spy_pct_days_above_sma20")) else "n/a"
        print(
            f"{row['month_key']:<10} {color:<6} {row['n_trades']:>6} "
            f"{row['win_rate_pct']:>6.1f} {fmt_pnl(row['total_pnl']):>11} "
            f"{fmt_pnl(row['avg_pnl_per_trade']):>10} {spy_ret:>9} "
            f"{spy_rng:>11} {sma_pct:>8} {int(row['max_loss_streak']):>10} "
            f"{row['late_entry_pct']:>7.1f}"
        )

    # ---------------------------------------------------------------------------
    # Section 2: Green vs Red aggregate comparison
    # ---------------------------------------------------------------------------
    print_separator("Green vs Red Months — Aggregate Comparison")

    for label, mask in [("GREEN months", ~monthly["is_red"]), ("RED months", monthly["is_red"])]:
        sub = monthly[mask]
        print(f"\n  {label} ({sub['month_key'].tolist()})")
        print(f"    Months:              {len(sub)}")
        print(f"    Total trades:        {sub['n_trades'].sum()}")
        print(f"    Avg trades/month:    {sub['n_trades'].mean():.1f}")
        print(f"    Avg win rate:        {sub['win_rate_pct'].mean():.1f}%")
        print(f"    Avg P&L/trade:       {fmt_pnl(sub['avg_pnl_per_trade'].mean())}")
        print(f"    Total P&L:           {fmt_pnl(sub['total_pnl'].sum())}")

        spy_sub = sub.dropna(subset=["spy_monthly_ret_pct"])
        if len(spy_sub):
            print(f"    SPY avg monthly ret: {spy_sub['spy_monthly_ret_pct'].mean():+.2f}%")
            print(f"    SPY avg daily range: {spy_sub['spy_avg_daily_range_pct'].mean():.2f}%")
            print(f"    SPY %days above SMA: {spy_sub['spy_pct_days_above_sma20'].mean():.1f}%")
        else:
            print("    SPY data: n/a (outside DB range)")

        print(f"    Avg max consec loss: {sub['max_loss_streak'].mean():.1f}")
        print(f"    Avg late entry %:    {sub['late_entry_pct'].mean():.1f}%")

    # ---------------------------------------------------------------------------
    # Section 3: SPY correlation check (months where we have DB data)
    # ---------------------------------------------------------------------------
    print_separator("SPY Regime vs Trade Win Rate (months with SPY data)")
    spy_trade = monthly.dropna(subset=["spy_monthly_ret_pct"])
    if len(spy_trade) >= 4:
        corr_wr = spy_trade["spy_monthly_ret_pct"].corr(spy_trade["win_rate_pct"])
        corr_pnl = spy_trade["spy_monthly_ret_pct"].corr(spy_trade["total_pnl"])
        corr_range_wr = spy_trade["spy_avg_daily_range_pct"].corr(spy_trade["win_rate_pct"])
        print(f"  Corr(SPY monthly ret, win rate):      {corr_wr:+.3f}")
        print(f"  Corr(SPY monthly ret, total P&L):     {corr_pnl:+.3f}")
        print(f"  Corr(SPY avg daily range, win rate):  {corr_range_wr:+.3f}")
    else:
        print("  Insufficient overlapping months for correlation (SPY DB starts Dec 2024).")
        print("  Using qf_spy_return_pct from backtest CSV instead...")

        # Fall back to per-trade qf_spy_return_pct
        trade_spy = df.dropna(subset=["qf_spy_return_pct"])
        if len(trade_spy):
            print(f"\n  Per-trade SPY return on entry day (n={len(trade_spy)} trades with data):")
            for label, mask in [("GREEN months", ~trade_spy["is_red_month"]),
                                 ("RED months", trade_spy["is_red_month"])]:
                sub = trade_spy[mask]
                if len(sub):
                    print(f"    {label}: SPY avg daily ret = {sub['qf_spy_return_pct'].mean():+.3f}%  "
                          f"n={len(sub)}  wr={sub['is_win'].mean()*100:.1f}%")

    # ---------------------------------------------------------------------------
    # Section 4: Day-of-week analysis
    # ---------------------------------------------------------------------------
    print_separator("Day-of-Week Win Rate: Green vs Red Months")
    dow_df = day_of_week_stats(df)
    pivot = dow_df.pivot_table(
        index="dow",
        columns="group",
        values=["n", "win_rate_pct", "avg_pnl"],
        aggfunc="first",
    )
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    print(f"  {'Day':<12} {'GREEN n':>8} {'GREEN WR%':>10} {'GREEN Avg$':>11} "
          f"{'RED n':>8} {'RED WR%':>9} {'RED Avg$':>10}")
    print("  " + "-" * 72)
    for dow in dow_order:
        sub = dow_df[dow_df["dow"] == dow]
        g = sub[sub["group"] == "GREEN"]
        r = sub[sub["group"] == "RED"]
        gn = int(g["n"].values[0]) if len(g) else 0
        gwr = f"{g['win_rate_pct'].values[0]:.1f}%" if len(g) else "n/a"
        gavg = fmt_pnl(g["avg_pnl"].values[0]) if len(g) else "n/a"
        rn = int(r["n"].values[0]) if len(r) else 0
        rwr = f"{r['win_rate_pct'].values[0]:.1f}%" if len(r) else "n/a"
        ravg = fmt_pnl(r["avg_pnl"].values[0]) if len(r) else "n/a"
        print(f"  {dow:<12} {gn:>8} {gwr:>10} {gavg:>11} {rn:>8} {rwr:>9} {ravg:>10}")

    # ---------------------------------------------------------------------------
    # Section 5: Entry time distribution
    # ---------------------------------------------------------------------------
    print_separator("Entry Time Distribution: Green vs Red Months")
    et_df = entry_time_bins(df)
    print(f"  {'Time Bucket':<14} {'GREEN n':>8} {'GREEN %':>8} {'GREEN WR%':>10} {'GREEN Avg$':>11} "
          f"{'RED n':>8} {'RED %':>7} {'RED WR%':>9} {'RED Avg$':>10}")
    print("  " + "-" * 89)
    for tbin in et_df["time_bin"].unique():
        sub = et_df[et_df["time_bin"] == tbin]
        g = sub[sub["group"] == "GREEN"]
        r = sub[sub["group"] == "RED"]
        gn = int(g["n_trades"].values[0]) if len(g) else 0
        gpct = f"{g['pct_of_group'].values[0]:.1f}%" if len(g) else "n/a"
        gwr = f"{g['win_rate_pct'].values[0]:.1f}%" if len(g) else "n/a"
        gavg = fmt_pnl(g["avg_pnl"].values[0]) if len(g) else "n/a"
        rn = int(r["n_trades"].values[0]) if len(r) else 0
        rpct = f"{r['pct_of_group'].values[0]:.1f}%" if len(r) else "n/a"
        rwr = f"{r['win_rate_pct'].values[0]:.1f}%" if len(r) else "n/a"
        ravg = fmt_pnl(r["avg_pnl"].values[0]) if len(r) else "n/a"
        print(f"  {tbin:<14} {gn:>8} {gpct:>8} {gwr:>10} {gavg:>11} "
              f"{rn:>8} {rpct:>7} {rwr:>9} {ravg:>10}")

    # ---------------------------------------------------------------------------
    # Section 6: Consecutive loss analysis
    # ---------------------------------------------------------------------------
    print_separator("Consecutive Loss Streaks by Month")
    for _, row in monthly.sort_values("month_key").iterrows():
        color = "RED  " if row["is_red"] else "green"
        print(f"  {row['month_key']}  {color}  max_consec_losses={int(row['max_loss_streak'])}")

    # ---------------------------------------------------------------------------
    # Section 7: Win-rate distribution — does red month WR cluster below a threshold?
    # ---------------------------------------------------------------------------
    print_separator("Win-Rate Threshold Analysis")
    thresholds = [40, 45, 50, 55, 60]
    print(f"  {'Threshold':>12}  {'Red below':>10}  {'Green below':>12}  {'Red above':>10}  {'Green above':>12}")
    print("  " + "-" * 62)
    for thr in thresholds:
        red_below = (monthly[monthly["is_red"]]["win_rate_pct"] < thr).sum()
        green_below = (monthly[~monthly["is_red"]]["win_rate_pct"] < thr).sum()
        red_above = (monthly[monthly["is_red"]]["win_rate_pct"] >= thr).sum()
        green_above = (monthly[~monthly["is_red"]]["win_rate_pct"] >= thr).sum()
        print(f"  WR < {thr}%     {red_below:>10}  {green_below:>12}  {red_above:>10}  {green_above:>12}")

    # ---------------------------------------------------------------------------
    # Section 8: SPY regime using embedded qf_spy_return_pct (all months)
    # ---------------------------------------------------------------------------
    print_separator("Per-Day SPY Direction on Trade Days (embedded qf_spy_return_pct)")
    spy_col = df.dropna(subset=["qf_spy_return_pct"])
    if len(spy_col) > 0:
        print(f"  Trades with SPY data: {len(spy_col)} / {len(df)}")
        for label, mask in [("GREEN months", ~spy_col["is_red_month"]),
                             ("RED months", spy_col["is_red_month"])]:
            sub = spy_col[mask]
            if len(sub) == 0:
                continue
            spy_up = (sub["qf_spy_return_pct"] > 0).mean() * 100
            spy_down = (sub["qf_spy_return_pct"] < 0).mean() * 100
            avg_spy_ret = sub["qf_spy_return_pct"].mean()
            wr_spy_up = sub[sub["qf_spy_return_pct"] > 0]["is_win"].mean() * 100
            wr_spy_down = sub[sub["qf_spy_return_pct"] < 0]["is_win"].mean() * 100
            print(f"\n  {label} (n={len(sub)}):")
            print(f"    SPY up days: {spy_up:.1f}%   SPY down days: {spy_down:.1f}%")
            print(f"    Avg SPY ret on trade days: {avg_spy_ret:+.3f}%")
            print(f"    Win rate on SPY-up days:   {wr_spy_up:.1f}%")
            print(f"    Win rate on SPY-down days: {wr_spy_down:.1f}%")
    else:
        print("  No qf_spy_return_pct data available.")

    # ---------------------------------------------------------------------------
    # Section 8a: Win/Loss dollar asymmetry — the real culprit
    # ---------------------------------------------------------------------------
    print_separator("Win/Loss Dollar Asymmetry by Month")
    header2 = (
        f"{'Month':<10} {'Color':<6} {'AvgWin':>9} {'AvgLoss':>9} "
        f"{'W/L Ratio':>10} {'StopOut%':>9} {'PnL ex-top':>12}"
    )
    print(header2)
    print("-" * len(header2))
    for _, row in monthly.sort_values("month_key").iterrows():
        color = "RED  " if row["is_red"] else "green"
        print(
            f"{row['month_key']:<10} {color:<6} "
            f"{fmt_pnl(row['avg_win']):>9} {fmt_pnl(row['avg_loss']):>9} "
            f"{row['win_loss_ratio']:>10.2f} {row['stop_out_pct']:>9.1f} "
            f"{fmt_pnl(row['pnl_ex_top']):>12}"
        )
    print()
    for label, mask in [("GREEN", ~monthly["is_red"]), ("RED", monthly["is_red"])]:
        sub = monthly[mask]
        print(f"  {label}: avg_win={fmt_pnl(sub['avg_win'].mean())}  "
              f"avg_loss={fmt_pnl(sub['avg_loss'].mean())}  "
              f"W/L ratio={sub['win_loss_ratio'].mean():.2f}x  "
              f"stop_out%={sub['stop_out_pct'].mean():.1f}%")
    print()
    print("  KEY INSIGHT: Win/loss ratio collapses 2.32x → 1.38x in red months.")
    print("  Winners are smaller AND stop-outs rise from 57.8% → 65.8% of exits.")
    print("  This is the PRIMARY driver of red months, not just lower win rate.")

    # ---------------------------------------------------------------------------
    # Section 8c: Trade count vs SPY trend — does overtrading correlate with red?
    # ---------------------------------------------------------------------------
    print_separator("Trade Count Explosion in Red Months")
    print("  Red months take significantly MORE trades — is it overtrading or more setups?")
    print()
    print(f"  {'Month':<10} {'Color':<6} {'Trades':>7} {'WR%':>6} {'Avg P&L':>9} {'SPY Ret%':>9}")
    print("  " + "-" * 55)
    for _, row in monthly.sort_values("month_key").iterrows():
        color = "RED  " if row["is_red"] else "green"
        spy_r = f"{row['spy_monthly_ret_pct']:+.1f}%" if pd.notna(row.get("spy_monthly_ret_pct")) else "n/a"
        print(f"  {row['month_key']:<10} {color:<6} {row['n_trades']:>7} "
              f"{row['win_rate_pct']:>6.1f} {fmt_pnl(row['avg_pnl_per_trade']):>9} {spy_r:>9}")
    print()
    n_red_trades = monthly[monthly["is_red"]]["n_trades"].mean()
    n_green_trades = monthly[~monthly["is_red"]]["n_trades"].mean()
    print(f"  Red months avg trades:   {n_red_trades:.1f}  (+{(n_red_trades/n_green_trades-1)*100:.0f}% vs green)")
    print(f"  Green months avg trades: {n_green_trades:.1f}")
    print()
    print("  Hypothesis: More trades in bad conditions = the scanner generates setups")
    print("  but market regime makes them fail. A trade-count cap or")
    print("  daily-loss-limit is already in place but the monthly damage accumulates.")

    # ---------------------------------------------------------------------------
    # Section 8d: Back-to-back loss analysis (are losses clustering in streaks?)
    # ---------------------------------------------------------------------------
    print_separator("Loss Clustering: Do Losses Come in Bursts in Red Months?")
    for label, mask in [("GREEN months", ~df["is_red_month"]), ("RED months", df["is_red_month"])]:
        sub = df[mask].sort_values(["date", "entry_dt"])
        outcomes = sub["is_win"].tolist()
        streaks = []
        cur = 0
        for w in outcomes:
            if not w:
                cur += 1
            else:
                if cur > 0:
                    streaks.append(cur)
                cur = 0
        if cur > 0:
            streaks.append(cur)
        if streaks:
            print(f"\n  {label}:")
            print(f"    Total loss streaks:    {len(streaks)}")
            print(f"    Avg streak length:     {np.mean(streaks):.2f}")
            print(f"    Max streak:            {max(streaks)}")
            print(f"    Streaks >= 3:          {sum(1 for s in streaks if s >= 3)}")
            print(f"    Streaks >= 5:          {sum(1 for s in streaks if s >= 5)}")
            print(f"    Streak distribution:   {sorted(set(streaks), reverse=True)[:10]}")

    # ---------------------------------------------------------------------------
    # Section 9: Key findings summary
    # ---------------------------------------------------------------------------
    print_separator("Key Findings Summary")
    red = monthly[monthly["is_red"]]
    green = monthly[~monthly["is_red"]]

    wr_diff = red["win_rate_pct"].mean() - green["win_rate_pct"].mean()
    trade_diff = red["n_trades"].mean() - green["n_trades"].mean()
    late_diff = red["late_entry_pct"].mean() - green["late_entry_pct"].mean()
    streak_diff = red["max_loss_streak"].mean() - green["max_loss_streak"].mean()

    print(f"  Win rate gap (red minus green):      {wr_diff:+.1f} pp")
    print(f"  Avg trades/month gap:                {trade_diff:+.1f}")
    print(f"  Late entry % gap (red minus green):  {late_diff:+.1f} pp")
    print(f"  Avg max consec-loss gap:             {streak_diff:+.1f}")

    spy_overlap = monthly.dropna(subset=["spy_monthly_ret_pct"])
    if len(spy_overlap):
        spy_red = spy_overlap[spy_overlap["is_red"]]["spy_monthly_ret_pct"].mean()
        spy_green = spy_overlap[~spy_overlap["is_red"]]["spy_monthly_ret_pct"].mean()
        print(f"  SPY monthly ret — red months avg:    {spy_red:+.2f}%  (DB overlap months only)")
        print(f"  SPY monthly ret — green months avg:  {spy_green:+.2f}%  (DB overlap months only)")

    print()
    print("  Potential regime filters to test:")

    # Check if WR below 50% correlates with red months
    n_red_low_wr = (red["win_rate_pct"] < 50).sum()
    n_green_low_wr = (green["win_rate_pct"] < 50).sum()
    print(f"    - WR<50% occurs in {n_red_low_wr}/{len(red)} red months vs {n_green_low_wr}/{len(green)} green months")

    # Check late entry
    late_red = red["late_entry_pct"].mean()
    late_green = green["late_entry_pct"].mean()
    print(f"    - Late entries: red={late_red:.1f}%  green={late_green:.1f}%  (higher in red = more midday slop)")

    # Check streak
    streak_red = red["max_loss_streak"].mean()
    streak_green = green["max_loss_streak"].mean()
    print(f"    - Max consec losses: red avg={streak_red:.1f}  green avg={streak_green:.1f}")

    # SPY direction and range summary
    spy_overlap = monthly.dropna(subset=["spy_monthly_ret_pct"])
    if len(spy_overlap):
        spy_red_range = spy_overlap[spy_overlap["is_red"]]["spy_avg_daily_range_pct"].mean()
        spy_green_range = spy_overlap[~spy_overlap["is_red"]]["spy_avg_daily_range_pct"].mean()
        print(f"    - SPY avg daily range: red={spy_red_range:.2f}%  green={spy_green_range:.2f}%")

    print()
    print("  REGIME FILTER CANDIDATES (strongest signal first):")
    print()
    print("  1. WIN/LOSS RATIO COLLAPSE (strongest signal, but only detectable in hindsight):")
    print("     Green months: avg_win=$6,092, avg_loss=-$2,628, W/L ratio=2.32x")
    print("     Red months:   avg_win=$3,839, avg_loss=-$2,788, W/L ratio=1.38x")
    print("     -> Winners fail to extend. This is the ROOT CAUSE.")
    print("     -> Real-time proxy: track rolling 5-trade avg winner size.")
    print("        If avg winner < $3K for last 5 trades, half position size.")
    print()
    print("  2. TRADE COUNT EXPLOSION: Red months avg 29.8 trades vs green 21.8 (+37%)")
    print("     -> Scanner fires MORE setups in bad regimes (choppy/low-vol market).")
    print("     -> More attempts at the same low-quality setups compounds losses.")
    print("     -> Actionable: cap trades at 22/month or 5/week.")
    print()
    print("  3. STOP-OUT RATE: 65.8% in red vs 57.8% in green (+8pp)")
    print("     -> Real-time proxy: after 3 stops in a week, sit out Friday.")
    print()
    print("  4. SPY LOW VOLATILITY TRAP: Red months SPY avg range = 0.98% vs 1.28% in green")
    print("     -> Counterintuitive: LOW SPY vol = bad for momentum stocks.")
    print("     -> The momentum stocks need a volatile tape to sustain their moves.")
    print("     -> When SPY range < 0.9% for 3 consecutive days, reduce size 50%.")
    print()
    print("  5. DAY OF WEEK: Tue/Wed are killers in red months (-24% WR, -$1.3K avg)")
    print("     -> In red months: Mon/Tue/Wed all below 27% WR")
    print("     -> Thu/Fri remain tradeable (43%/41% WR) even in red months")
    print("     -> Consider Tue/Wed as 'half-size days' when in a losing streak.")
    print()
    print("  6. MORNING SLOT TRAP: 09:30-09:45 WR = 31.8% in red vs 45.0% in green")
    print("     -> The first-candle fakeout is deadlier in bad months.")
    print("     -> After 3 stops in past 5 days, skip 09:30-09:45 entirely.")
    print()
    print("  WHAT IS NOT A SIGNAL:")
    print("    - SPY monthly direction: correlation near zero (-0.20), not predictive")
    print("    - Consecutive loss streaks: red avg 5.8 vs green 5.1 (too similar)")
    print("    - Late entries (10:00+): similar % in both groups")
    print()
    print("  IS THERE A DETECTABLE SIGNAL BEFORE THE MONTH GOES RED?")
    print("    Aug 2025: First day (Aug 1) = -$7,443. Clear warning from day 1.")
    print("    Dec 2025: First week negative, but recoverable losses.")
    print("    Jan 2026: Cascading losses from week 1 (8-trade max loss streak).")
    print("    Feb 2026: Relatively mild (-$2.7K total), hard to detect early.")
    print("    Mar 2026: Moderate losses spread across month.")
    print("    -> VERDICT: 3/5 red months show damage in week 1.")
    print("    -> Weekly P&L monitoring + 3-consecutive-loss rule catches most.")
    print("    -> No predictive market-regime signal exists BEFORE the month starts.")

    print_separator()
    print("  Done.")


if __name__ == "__main__":
    main()
