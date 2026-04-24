"""
Dual Qualification Analysis for Bull Flag Backtest
===================================================

Current system: stock qualifies when intraday range hits 20% from prev_close.
Proposed dual qualification:
  Path A: Pre-market gap >= 10% from prev_close → qualify at open (bar 0)
  Path B: Intraday range >= 8% → qualify when range crosses 8%

This script:
1. Loads the 20% cache (production baseline)
2. Loads lower-threshold caches (8%, 15%)
3. For extra trades not in the 20% cache, checks whether they were gap-up stocks (gap >= 10%)
4. Reports P&L, win rate, and trade quality for extra trades
"""

import sqlite3
import sys
from datetime import date, timedelta
from typing import Dict, Optional, Set, Tuple

import pandas as pd

# --- Config ---
DB_PATH = "data/cache.db"
CACHE_T20 = "data/bull_flag_cache_e50_x30_t20.csv"
CACHE_T15 = "data/bull_flag_cache_e50_x30_t15.csv"
CACHE_T8 = "data/bull_flag_cache_e50_x30_t8.csv"
GAP_THRESHOLD = 0.10   # 10% pre-market gap threshold for Path A
INTRA_THRESHOLD = 0.08  # 8% intraday range threshold for Path B


def load_cache(path: str) -> pd.DataFrame:
    """Load a bull flag cache CSV into a DataFrame."""
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    print(f"  Loaded {len(df):,} trades from {path}")
    return df


def make_key(df: pd.DataFrame) -> Set[Tuple]:
    """Create a set of (symbol, date, entry_time_et) keys for dedup."""
    return set(zip(df["symbol"], df["date"].astype(str), df["entry_time_et"]))


def get_prev_close(conn: sqlite3.Connection, symbol: str, trade_date: date) -> Optional[float]:
    """
    Fetch the previous trading day's close price from daily_bars.

    Returns None if no prior bar exists within 5 trading days.
    """
    # Look back up to 7 calendar days for a valid prev bar
    cur = conn.cursor()
    cur.execute(
        """
        SELECT close FROM daily_bars
        WHERE symbol = ? AND bar_date < ?
        ORDER BY bar_date DESC
        LIMIT 1
        """,
        (symbol, trade_date.isoformat()),
    )
    row = cur.fetchone()
    return float(row[0]) if row else None


def compute_gap_pct(conn: sqlite3.Connection, symbol: str, trade_date: date) -> Optional[float]:
    """
    Compute the pre-market / open gap percentage from prev_close to trade_date open.

    Uses: (open_of_trade_date - prev_close) / prev_close
    Returns None if either price is unavailable or prev_close <= 0.
    """
    cur = conn.cursor()
    # Get today's open
    cur.execute(
        "SELECT open FROM daily_bars WHERE symbol = ? AND bar_date = ?",
        (symbol, trade_date.isoformat()),
    )
    row = cur.fetchone()
    if not row:
        return None
    today_open = float(row[0])

    prev_close = get_prev_close(conn, symbol, trade_date)
    if prev_close is None or prev_close <= 0:
        return None

    return (today_open - prev_close) / prev_close


def print_stats(label: str, df: pd.DataFrame) -> None:
    """Print P&L statistics for a set of trades."""
    if df.empty:
        print(f"  {label}: 0 trades — no data")
        return

    total_trades = len(df)
    winners = df[df["pnl"] > 0]
    losers = df[df["pnl"] <= 0]
    win_rate = len(winners) / total_trades * 100
    total_pnl = df["pnl"].sum()
    avg_win = winners["pnl"].mean() if len(winners) > 0 else 0
    avg_loss = losers["pnl"].mean() if len(losers) > 0 else 0
    profit_factor = (
        abs(winners["pnl"].sum() / losers["pnl"].sum())
        if losers["pnl"].sum() != 0 else float("inf")
    )

    print(f"\n  {label}")
    print(f"    Trades:        {total_trades:,}")
    print(f"    Win Rate:      {win_rate:.1f}%  ({len(winners)}W / {len(losers)}L)")
    print(f"    Total P&L:     ${total_pnl:,.0f}")
    print(f"    Avg Win:       ${avg_win:,.0f}")
    print(f"    Avg Loss:      ${avg_loss:,.0f}")
    print(f"    Profit Factor: {profit_factor:.2f}")


def analyze_extra_trades(
    extra: pd.DataFrame, conn: sqlite3.Connection, label: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For a set of extra trades (not in t20 cache), classify each as:
      - gap_up: pre-market gap >= 10% (would qualify via Path A)
      - non_gap: gap < 10% (needs intraday range to qualify, i.e. Path B only)

    Returns (gap_up_df, non_gap_df).
    """
    print(f"\n  Classifying {len(extra):,} extra trades for {label}...")

    gap_pcts: Dict[Tuple[str, str], Optional[float]] = {}

    # Batch: unique (symbol, date) pairs
    pairs = extra[["symbol", "date"]].drop_duplicates()
    for _, row in pairs.iterrows():
        sym = row["symbol"]
        dt = row["date"] if isinstance(row["date"], date) else row["date"].date()
        key = (sym, str(dt))
        gap_pcts[key] = compute_gap_pct(conn, sym, dt)

    # Classify each trade
    gap_flags = []
    gap_values = []
    for _, row in extra.iterrows():
        dt = row["date"] if isinstance(row["date"], date) else row["date"].date()
        key = (row["symbol"], str(dt))
        g = gap_pcts.get(key)
        gap_flags.append(g is not None and g >= GAP_THRESHOLD)
        gap_values.append(g)

    extra = extra.copy()
    extra["is_gap_up"] = gap_flags
    extra["gap_pct"] = gap_values

    gap_up = extra[extra["is_gap_up"]]
    non_gap = extra[~extra["is_gap_up"]]

    print(f"    Gap>=10% (Path A eligible): {len(gap_up):,}")
    print(f"    Gap<10%  (Path B only):     {len(non_gap):,}")

    return gap_up, non_gap


def main() -> None:
    print("=" * 70)
    print("DUAL QUALIFICATION ANALYSIS")
    print(f"  Gap threshold (Path A):     {GAP_THRESHOLD:.0%}")
    print(f"  Intraday threshold (Path B): {INTRA_THRESHOLD:.0%}")
    print("=" * 70)

    # --- Load caches ---
    print("\n[1] Loading caches...")
    t20 = load_cache(CACHE_T20)
    t15 = load_cache(CACHE_T15)
    t8 = load_cache(CACHE_T8)

    # Normalize dates
    for df in (t20, t15, t8):
        if not isinstance(df["date"].iloc[0], date):
            df["date"] = pd.to_datetime(df["date"]).dt.date

    # Align date range to t20 (production baseline)
    start_dt = t20["date"].min()
    end_dt = t20["date"].max()
    print(f"\n  Baseline date range: {start_dt} → {end_dt}")

    t15 = t15[(t15["date"] >= start_dt) & (t15["date"] <= end_dt)].copy()
    t8 = t8[(t8["date"] >= start_dt) & (t8["date"] <= end_dt)].copy()

    print(f"  t20 (after date filter): {len(t20):,} trades")
    print(f"  t15 (after date filter): {len(t15):,} trades")
    print(f"  t8  (after date filter): {len(t8):,} trades")

    # --- Build key sets ---
    keys_t20 = make_key(t20)
    keys_t15 = make_key(t15)
    keys_t8 = make_key(t8)

    # --- Compute extra trades per lower cache ---
    def extra_vs_t20(df: pd.DataFrame) -> pd.DataFrame:
        """Return rows in df not present in t20 (by symbol+date+entry_time key)."""
        mask = [
            (row["symbol"], str(row["date"]), row["entry_time_et"]) not in keys_t20
            for _, row in df.iterrows()
        ]
        return df[mask]

    extra_t15 = extra_vs_t20(t15)
    extra_t8 = extra_vs_t20(t8)

    print(f"\n  Extra trades in t15 vs t20: {len(extra_t15):,}")
    print(f"  Extra trades in t8  vs t20: {len(extra_t8):,}")

    # --- DB connection ---
    print("\n[2] Connecting to cache.db for prev_close / open data...")
    conn = sqlite3.connect(DB_PATH)

    # --- Baseline stats ---
    print("\n[3] Baseline: current production (20% single threshold)")
    print_stats("t20 — ALL trades (production baseline)", t20)

    # --- t15 extra trades analysis ---
    print("\n[4] Extra trades from 15% threshold (not in 20% baseline)")
    print_stats("t15 EXTRA — all extra trades", extra_t15)
    if not extra_t15.empty:
        gap_t15, nogap_t15 = analyze_extra_trades(extra_t15, conn, "t15 extra")
        print_stats("t15 EXTRA — gap>=10% subset (Path A eligible)", gap_t15)
        print_stats("t15 EXTRA — gap<10% subset (Path B only)", nogap_t15)

    # --- t8 extra trades analysis ---
    print("\n[5] Extra trades from 8% threshold (not in 20% baseline)")
    print_stats("t8 EXTRA — all extra trades", extra_t8)
    if not extra_t8.empty:
        gap_t8, nogap_t8 = analyze_extra_trades(extra_t8, conn, "t8 extra")
        print_stats("t8 EXTRA — gap>=10% subset (Path A eligible)", gap_t8)
        print_stats("t8 EXTRA — gap<10% subset (Path B only)", nogap_t8)

    # --- Dual qualification simulation ---
    # Dual qual = t20 baseline + (gap_up_only trades from t8 extra)
    # This simulates: qualify immediately at open if gap>=10%, else require 20% intraday range
    print("\n[6] Dual Qualification Simulation")
    print("     = t20 baseline + gap>=10% extra trades from t8 threshold")
    if not extra_t8.empty:
        gap_t8, nogap_t8 = analyze_extra_trades(extra_t8, conn, "t8 extra (for dual sim)")

        dual_qual_trades = pd.concat([t20, gap_t8], ignore_index=True)
        print_stats("DUAL QUAL (t20 + gap-up extras from t8)", dual_qual_trades)

        # Improvement summary
        baseline_pnl = t20["pnl"].sum()
        dual_pnl = dual_qual_trades["pnl"].sum()
        extra_gap_pnl = gap_t8["pnl"].sum()
        print(f"\n  --- Improvement Summary ---")
        print(f"    Baseline  (t20):       {len(t20):,} trades, ${baseline_pnl:,.0f}")
        print(f"    Dual Qual (t20+gap8):  {len(dual_qual_trades):,} trades, ${dual_pnl:,.0f}")
        print(f"    Delta:                 +{len(gap_t8):,} trades, ${extra_gap_pnl:+,.0f}")
        pct_change = (dual_pnl - baseline_pnl) / abs(baseline_pnl) * 100 if baseline_pnl != 0 else 0
        print(f"    P&L change:            {pct_change:+.1f}%")

    # --- Also show: what extra trades exist in t8 NOT even in t15? ---
    print("\n[7] Trades in t8 but NOT in t15 (very early qualification 8-15% range)")
    extra_t8_vs_t15 = t8[
        ~t8.apply(
            lambda r: (r["symbol"], str(r["date"]), r["entry_time_et"]) in keys_t15,
            axis=1
        )
    ].copy()
    print_stats("t8-only extra trades (8-15% range, not in t15)", extra_t8_vs_t15)
    if not extra_t8_vs_t15.empty:
        gap_8only, nogap_8only = analyze_extra_trades(extra_t8_vs_t15, conn, "t8-only extra")
        print_stats("t8-only EXTRA — gap>=10% (Path A eligible)", gap_8only)
        print_stats("t8-only EXTRA — gap<10% (Path B only)", nogap_8only)

    # --- Top extra gap-up winners ---
    if not extra_t8.empty and not gap_t8.empty:
        print("\n[8] Top 15 gap-up extra trades (by P&L)")
        top = gap_t8.sort_values("pnl", ascending=False).head(15)[
            ["symbol", "date", "entry_time_et", "entry_price", "pnl", "pnl_pct", "exit_reason", "gap_pct", "daily_range_pct"]
        ]
        top["gap_pct"] = (top["gap_pct"] * 100).round(1)
        print(top.to_string(index=False))

        print("\n[9] Bottom 10 gap-up extra trades (biggest losers)")
        bot = gap_t8.sort_values("pnl").head(10)[
            ["symbol", "date", "entry_time_et", "entry_price", "pnl", "pnl_pct", "exit_reason", "gap_pct", "daily_range_pct"]
        ]
        bot["gap_pct"] = (bot["gap_pct"] * 100).round(1)
        print(bot.to_string(index=False))

    conn.close()
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
