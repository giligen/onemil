"""
News Catalyst + Lower Threshold Analysis

Hypothesis: stocks with real catalysts (FDA, earnings, contracts) have sustained
momentum, so we can catch them earlier (8-10% range) without adding noise.

Data sources:
  - data/bull_flag_cache_e50_x30_t20.csv  -- 20% threshold cache (Stage 1 raw)
  - data/bull_flag_cache_e50_x30_t8.csv   -- 8% threshold cache (Stage 1 raw)
  - data/trades.db                         -- live-paper trades with news_catalyst
  - data/cache.db                          -- news_cache (25 symbols/day, Dec25-Mar26)
"""

import pandas as pd
import sqlite3
import sys
from pathlib import Path

DATA_DIR = Path("data")
TRADES_DB = DATA_DIR / "trades.db"
CACHE_DB = DATA_DIR / "cache.db"
T20_CSV = DATA_DIR / "bull_flag_cache_e50_x30_t20.csv"
T8_CSV = DATA_DIR / "bull_flag_cache_e50_x30_t8.csv"


# ── helpers ──────────────────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame, label: str) -> dict:
    """Compute win rate, total P&L, avg P&L, and profit factor for a trade set."""
    if df.empty:
        return {
            "label": label, "n": 0, "win_rate": 0.0,
            "total_pnl": 0.0, "avg_pnl": 0.0, "profit_factor": float("nan"),
        }
    wins = df[df["pnl"] > 0]["pnl"].sum()
    losses = df[df["pnl"] <= 0]["pnl"].sum()
    profit_factor = (wins / abs(losses)) if losses != 0 else float("inf")
    return {
        "label": label,
        "n": len(df),
        "win_rate": (df["pnl"] > 0).mean() * 100,
        "total_pnl": df["pnl"].sum(),
        "avg_pnl": df["pnl"].mean(),
        "profit_factor": profit_factor,
    }


def print_stats(stats: dict):
    """Pretty-print a stats dict."""
    pf = f"{stats['profit_factor']:.2f}" if stats["profit_factor"] != float("inf") else "∞"
    print(
        f"  {stats['label']:<35} n={stats['n']:>4}  "
        f"WR={stats['win_rate']:>5.1f}%  "
        f"PnL=${stats['total_pnl']:>10,.0f}  "
        f"AvgPnL=${stats['avg_pnl']:>7,.0f}  "
        f"PF={pf}"
    )


def load_news_cache() -> pd.DataFrame:
    """Load the news_cache table from cache.db."""
    conn = sqlite3.connect(CACHE_DB)
    df = pd.read_sql(
        "SELECT symbol, news_date, catalyst, reason, headline FROM news_cache", conn
    )
    conn.close()
    df["news_date"] = df["news_date"].astype(str)
    return df


def load_live_news() -> pd.DataFrame:
    """Load news_catalyst data from the live trades DB."""
    conn = sqlite3.connect(TRADES_DB)
    df = pd.read_sql(
        "SELECT symbol, trade_date, news_catalyst, news_headline FROM trades "
        "WHERE news_catalyst IS NOT NULL",
        conn,
    )
    conn.close()
    df["trade_date"] = df["trade_date"].astype(str)
    return df


def attach_news(trades: pd.DataFrame, news_cache: pd.DataFrame, live_news: pd.DataFrame) -> pd.DataFrame:
    """
    Attach news_catalyst to each trade row using two sources:
      1. data/cache.db news_cache  (scanner universe, Dec25-Mar26, ~17% hit rate)
      2. data/trades.db live trades (paper/live, Mar-Apr26, 13 trades)

    Unmatched trades are marked 'unknown'.
    """
    trades = trades.copy()
    trades["date_str"] = trades["date"].astype(str)
    trades["news_group"] = "unknown"

    # Build lookup dicts for O(1) access
    news_lookup = {
        (row["symbol"], row["news_date"]): row["catalyst"]
        for _, row in news_cache.iterrows()
    }
    live_lookup = {
        (row["symbol"], row["trade_date"]): row["news_catalyst"]
        for _, row in live_news.iterrows()
    }

    def classify(row):
        key = (row["symbol"], row["date_str"])
        if key in live_lookup:
            c = live_lookup[key]
            return "has_news" if c == 1 else "no_news"
        if key in news_lookup:
            c = news_lookup[key]
            return "has_news" if c == 1 else "no_news"
        return "unknown"

    trades["news_group"] = trades.apply(classify, axis=1)
    return trades


def load_universe_sectors() -> dict:
    """Load sector data from cache.db universe table."""
    conn = sqlite3.connect(CACHE_DB)
    df = pd.read_sql("SELECT symbol, sector FROM universe WHERE sector IS NOT NULL", conn)
    conn.close()
    return dict(zip(df["symbol"], df["sector"]))


# ── main analysis ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("NEWS CATALYST + LOWER THRESHOLD ANALYSIS")
    print("=" * 70)

    # ── Load data ──
    print("\n[1] Loading data...")
    df20 = pd.read_csv(T20_CSV)
    print(f"    T20 cache: {len(df20):,} trades  ({df20['date'].min()} → {df20['date'].max()})")

    df8 = None
    if T8_CSV.exists():
        df8 = pd.read_csv(T8_CSV)
        print(f"    T8  cache: {len(df8):,} trades  ({df8['date'].min()} → {df8['date'].max()})")
    else:
        print(f"    T8  cache: NOT FOUND — skipping 8% analysis")

    news_cache = load_news_cache()
    live_news = load_live_news()
    print(f"    news_cache: {len(news_cache):,} rows  "
          f"({news_cache['news_date'].min()} → {news_cache['news_date'].max()})")
    print(f"    live_news:  {len(live_news):,} rows  "
          f"({live_news['trade_date'].min()} → {live_news['trade_date'].max()})")

    # ── Attach news to T20 ──
    print("\n[2] Attaching news labels to T20 trades...")
    df20 = attach_news(df20, news_cache, live_news)

    group_counts = df20["news_group"].value_counts()
    print(f"    has_news : {group_counts.get('has_news',  0):>4}")
    print(f"    no_news  : {group_counts.get('no_news',   0):>4}")
    print(f"    unknown  : {group_counts.get('unknown',   0):>4}")

    news_window = news_cache["news_date"].min()
    news_window_end = news_cache["news_date"].max()
    total_in_window = len(df20[
        (df20["date_str"] >= news_window) & (df20["date_str"] <= news_window_end)
    ])
    known_in_window = len(df20[
        (df20["date_str"] >= news_window) &
        (df20["date_str"] <= news_window_end) &
        (df20["news_group"] != "unknown")
    ])
    coverage_pct = known_in_window / total_in_window * 100 if total_in_window else 0
    print(f"\n    News coverage within news_cache window ({news_window} → {news_window_end}):")
    print(f"    {known_in_window}/{total_in_window} trades labeled = {coverage_pct:.1f}%")
    print(f"    IMPORTANT: {100 - coverage_pct:.1f}% of in-window trades are 'unknown' because")
    print(f"    news_cache only covers the scanner's 25-stock daily universe, not all movers.")

    # ── Stats by news group (T20) ──
    print("\n" + "=" * 70)
    print("[3] T20 PERFORMANCE BY NEWS GROUP")
    print("=" * 70)
    print("    (ALL trades including unknown)")
    print()

    for group in ["has_news", "no_news", "unknown", "all"]:
        if group == "all":
            subset = df20
        else:
            subset = df20[df20["news_group"] == group]
        print_stats(compute_stats(subset, f"T20 {group}"))

    # Sub-analysis: within the news_cache window only (apples-to-apples)
    print()
    print(f"    (Only trades in news_cache window {news_window}→{news_window_end})")
    df20_window = df20[(df20["date_str"] >= news_window) & (df20["date_str"] <= news_window_end)]
    for group in ["has_news", "no_news", "unknown", "all"]:
        if group == "all":
            subset = df20_window
        else:
            subset = df20_window[df20_window["news_group"] == group]
        print_stats(compute_stats(subset, f"T20 window {group}"))

    # ── T8 analysis ──
    if df8 is not None:
        print("\n" + "=" * 70)
        print("[4] T8 vs T20: EXTRA TRADES FROM LOWERING THRESHOLD")
        print("=" * 70)

        # Identify trades in T8 but not T20 (extra trades from lower threshold)
        key20 = set(zip(df20["symbol"], df20["date_str"]))
        df8["date_str"] = df8["date"].astype(str)
        df8_extra = df8[~df8.apply(lambda r: (r["symbol"], r["date_str"]) in key20, axis=1)].copy()
        df8_shared = df8[df8.apply(lambda r: (r["symbol"], r["date_str"]) in key20, axis=1)].copy()

        print(f"\n    T8 total:  {len(df8):,} trades")
        print(f"    T20 total: {len(df20):,} trades")
        print(f"    T8 trades also in T20 (shared): {len(df8_shared):,}")
        print(f"    T8 EXTRA trades (not in T20):   {len(df8_extra):,}")

        # Attach news to extra trades
        df8_extra = attach_news(df8_extra, news_cache, live_news)
        extra_counts = df8_extra["news_group"].value_counts()
        print(f"\n    Extra trades news breakdown:")
        print(f"      has_news : {extra_counts.get('has_news',  0):>4}")
        print(f"      no_news  : {extra_counts.get('no_news',   0):>4}")
        print(f"      unknown  : {extra_counts.get('unknown',   0):>4}")

        print()
        print("    T8 EXTRA trades by news group:")
        for group in ["has_news", "no_news", "unknown", "all"]:
            if group == "all":
                subset = df8_extra
            else:
                subset = df8_extra[df8_extra["news_group"] == group]
            print_stats(compute_stats(subset, f"T8 extra {group}"))

        # ── Scenario comparison ──
        print("\n" + "=" * 70)
        print("[5] SCENARIO COMPARISON")
        print("=" * 70)
        print()

        # Scenario A: current system — T20 (all), no news filter
        stats_a = compute_stats(df20, "Scenario A: T20, no news req")
        print("  A) Current system: 20% threshold, news optional")
        print_stats(stats_a)

        # Scenario B: T8 + require news (has_news only from extra + T20 has_news)
        df8_all_news = attach_news(df8, news_cache, live_news)
        df8_with_news = df8_all_news[df8_all_news["news_group"] == "has_news"]
        stats_b = compute_stats(df8_with_news, "Scenario B: T8, has_news only")
        print()
        print("  B) New hypothesis: 8% threshold, require news catalyst")
        print_stats(stats_b)

        # Scenario C: T20 + require news
        df20_with_news = df20[df20["news_group"] == "has_news"]
        stats_c = compute_stats(df20_with_news, "Scenario C: T20, has_news only")
        print()
        print("  C) Same threshold (T20) but filter for news only")
        print_stats(stats_c)

        # Scenario D: T8 + require news in the window, compare to T20 window
        df8_all_news_window = df8_all_news[
            (df8_all_news["date_str"] >= news_window) &
            (df8_all_news["date_str"] <= news_window_end)
        ]
        df8_window_news = df8_all_news_window[df8_all_news_window["news_group"] == "has_news"]
        df20_window_all = df20[
            (df20["date_str"] >= news_window) & (df20["date_str"] <= news_window_end)
        ]
        stats_d = compute_stats(df8_window_news, "Scenario D: T8+news (window)")
        stats_e = compute_stats(df20_window_all, "Scenario E: T20 all (window)")
        print()
        print(f"  D) T8 + news ONLY, restricted to news_cache window ({news_window}→{news_window_end})")
        print_stats(stats_d)
        print()
        print(f"  E) T20 all, same window (apples-to-apples baseline)")
        print_stats(stats_e)

    # ── Sector analysis ──
    print("\n" + "=" * 70)
    print("[6] SECTOR ANALYSIS (news-positive T20 trades)")
    print("=" * 70)
    sector_map = load_universe_sectors()
    df20["sector"] = df20["symbol"].map(sector_map).fillna("Unknown")

    df20_news = df20[df20["news_group"] == "has_news"]
    if not df20_news.empty:
        by_sector = df20_news.groupby("sector").apply(
            lambda g: pd.Series({
                "n": len(g),
                "win_rate": (g["pnl"] > 0).mean() * 100,
                "total_pnl": g["pnl"].sum(),
                "avg_pnl": g["pnl"].mean(),
            })
        ).sort_values("total_pnl", ascending=False)
        print("\n  News-positive trades by sector (T20):")
        print(by_sector.to_string())
    else:
        print("  No has_news trades to segment by sector.")

    # ── Limitations ──
    print("\n" + "=" * 70)
    print("[7] DATA LIMITATIONS & CAVEATS")
    print("=" * 70)
    total_trades = len(df20)
    unknown_count = group_counts.get("unknown", 0)
    print(f"""
  1. NEWS COVERAGE IS EXTREMELY LOW: only {100 - unknown_count/total_trades*100:.1f}% of T20 trades
     have news labels ({unknown_count}/{total_trades} are 'unknown'). This makes
     any news-based segmentation statistically unreliable.

  2. NEWS CACHE SCOPE: The news_cache in cache.db covers only the daily scanner
     universe (~25 symbols/day). Most backtest movers are NOT in that set, so
     they remain 'unknown' even within the coverage window.

  3. SURVIVORSHIP BIAS IN NEWS LABELS: The 'has_news' trades we do know about
     came from the scanner — they were already selected for being big movers.
     We cannot distinguish "news caused the move" from "mover that also had news".

  4. T8 EXTRA TRADES: The T8 cache has Stage-1 (raw, unfiltered) trades.
     Comparing T8 raw P&L to T20 raw P&L is misleading without applying the
     same production filters (20% qual gate, volume, concurrent limits etc).
     See batch_backtest.py Stage 2 for proper filtered comparison.

  5. RECOMMENDATION: To properly test this hypothesis, enrich ALL backtest
     trade dates with news labels retroactively (Benzinga/Alpaca news API
     for each symbol+date in the cache). Current data cannot answer this
     question with statistical confidence.
""")


if __name__ == "__main__":
    main()
