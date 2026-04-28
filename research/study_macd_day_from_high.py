"""MACD wave: day-from-high filter parameter sweep.

Tests the hypothesis: skip MACD entries where current price has already
collapsed materially from the intraday high (i.e. we're catching a falling
knife on a stock that already had its move). Motivated by the 2026-04-28
HTCO incident — entered at $8.47 after the stock had peaked at $38.22 and
collapsed to 22% of day high, then lost $3.5K.

Methodology:
- Load 10,355 cached MACD wave signals (Jan 2025 - Apr 2026)
- For each signal, compute day_high using cached 1-min bars BEFORE entry
  time (regular hours only, no premarket — premarket prints can be exotic
  and we want the high traders are actually pricing off)
- Compute pct_of_day_high = entry_price / day_high_at_entry_time
- Slice into TRAIN / VAL / OOS:
    TRAIN: Jan-Sep 2025 (9 months — fit threshold)
    VAL:   Oct-Dec 2025 (3 months — validate threshold doesn't overfit)
    OOS:   Jan-Apr 2026 (4 months — final out-of-sample test)
- Sweep threshold ∈ {0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85}
  (skip signal if pct_of_day_high < threshold)
- Report P&L, # trades, # winners, # losers per (split, threshold)

Anti-overfit guards:
- Threshold MUST work in TRAIN, hold up in VAL, and survive OOS
- Don't pick threshold based on OOS performance — TRAIN dictates choice
- Report all 7 thresholds × 3 splits so reader can see the curve shape

Usage:
    python research/study_macd_day_from_high.py
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CACHE_DB = ROOT / "data" / "cache.db"
SIGNAL_CACHE = ROOT / "data" / "macd_signal_cache_t30_s30.csv"

# Splits — ET dates (signal cache uses YYYY-MM-DD format)
TRAIN_END = "2025-09-30"
VAL_END = "2025-12-31"
# OOS = everything after VAL_END
THRESHOLDS = [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85]

# Regular trading hours (UTC) — 9:30 ET = 14:30 UTC (winter) / 13:30 UTC (summer)
# We'll be liberal: regular session bars start at 13:30 UTC, end at 20:00 UTC.
# This includes both DST regimes.
RTH_START_UTC = time(13, 30)
RTH_END_UTC = time(20, 0)


def load_signals() -> pd.DataFrame:
    """Load signal cache and apply production-equivalent filters.

    The cache stores ALL detected MACD signals (no entry filters applied).
    Production runs with cross<3min, MACD>=0.5%, vol<300K, max_price=$30,
    matching macd_wave.yaml. Apply those here so the BT reflects what
    production actually trades.
    """
    df = pd.read_csv(SIGNAL_CACHE, parse_dates=["entry_time", "exit_time"])
    df = df[df["paper"] == False].copy()  # noqa: E712
    df = df.dropna(subset=["symbol"])
    df["symbol"] = df["symbol"].astype(str)
    print(f"Loaded {len(df):,} signals from cache (all paper=False)")

    # Production filters (from macd_wave.yaml validated config)
    before = len(df)
    df = df[df["cross_time_min"] <= 3]
    print(f"  cross<=3min: {before:,} → {len(df):,}")
    before = len(df)
    df = df[df["macd_hist_pct"] >= 0.5]
    print(f"  macd>=0.5%:  {before:,} → {len(df):,}")
    before = len(df)
    df = df[df["vol_at_cross"] <= 300_000]
    print(f"  vol<=300K:   {before:,} → {len(df):,}")
    before = len(df)
    df = df[df["entry_price"] <= 30.0]
    print(f"  price<=$30:  {before:,} → {len(df):,}")
    before = len(df)
    df = df[df["entry_price"] >= 5.0]
    print(f"  price>=$5:   {before:,} → {len(df):,}")
    return df.reset_index(drop=True)


def compute_day_high_per_signal(df: pd.DataFrame) -> pd.DataFrame:
    """For each signal, compute the regular-hours intraday HIGH up to but NOT
    including the entry minute. Adds 'day_high' and 'pct_of_day_high' columns.

    No lookahead: only bars whose timestamp < entry_time are considered.
    """
    con = sqlite3.connect(CACHE_DB)
    # Pre-fetch all relevant bars in one query (fastest)
    syms = sorted(df["symbol"].unique())
    dates = sorted(df["date"].unique())
    print(f"Loading 1-min bars for {len(syms):,} symbols × {len(dates):,} dates...")

    # Pull bars in chunks to avoid huge IN-clauses
    chunks = []
    chunk_size = 200
    for i in range(0, len(syms), chunk_size):
        sym_chunk = syms[i:i + chunk_size]
        placeholders = ",".join(["?"] * len(sym_chunk))
        q = (
            f"SELECT symbol, bar_date, timestamp, high "
            f"FROM intraday_bars_1min WHERE symbol IN ({placeholders})"
        )
        chunks.append(pd.read_sql_query(q, con, params=sym_chunk))
        if (i // chunk_size) % 10 == 0:
            print(f"  ...{i}/{len(syms)} symbols loaded")
    bars = pd.concat(chunks, ignore_index=True)
    print(f"Loaded {len(bars):,} 1-min bars total")
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)

    # Filter to regular hours (drop premarket / after-hours)
    t = bars["timestamp"].dt.time
    in_rth = (t >= RTH_START_UTC) & (t < RTH_END_UTC)
    bars = bars[in_rth].copy()
    print(f"After RTH filter: {len(bars):,} bars")

    # Group by (symbol, bar_date) and compute cumulative high per bar
    bars = bars.sort_values(["symbol", "bar_date", "timestamp"])
    bars["cum_high"] = bars.groupby(["symbol", "bar_date"])["high"].cummax()

    # For each signal, find the bar immediately BEFORE entry_time and use its
    # cum_high. Use merge_asof for efficiency.
    df = df.sort_values("entry_time").copy()
    bars = bars.sort_values("timestamp")

    # Use per-(symbol, date) merge to keep it tractable
    df_keys = df[["symbol", "date", "entry_time", "entry_price"]].copy()
    bars_keys = bars[["symbol", "bar_date", "timestamp", "cum_high"]].rename(
        columns={"bar_date": "date"}
    )

    merged = pd.merge_asof(
        df_keys.sort_values("entry_time"),
        bars_keys.sort_values("timestamp"),
        left_on="entry_time",
        right_on="timestamp",
        by=["symbol", "date"],
        direction="backward",
        allow_exact_matches=False,  # bar at exact entry_time = lookahead, exclude
    )
    merged = merged.rename(columns={"cum_high": "day_high"})
    merged["pct_of_day_high"] = merged["entry_price"] / merged["day_high"]

    # Re-attach to df
    df = df.merge(
        merged[["symbol", "date", "entry_time", "day_high", "pct_of_day_high"]],
        on=["symbol", "date", "entry_time"],
        how="left",
    )
    n_missing = df["day_high"].isna().sum()
    print(f"Signals without day_high (no prior RTH bars): {n_missing:,}")
    return df


def split(df: pd.DataFrame) -> dict:
    """Return TRAIN/VAL/OOS slices keyed by date string."""
    return {
        "TRAIN (Jan-Sep 2025)": df[df["date"] <= TRAIN_END].copy(),
        "VAL (Oct-Dec 2025)": df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].copy(),
        "OOS (Jan-Apr 2026)": df[df["date"] > VAL_END].copy(),
    }


def evaluate(slice_df: pd.DataFrame, threshold: float) -> dict:
    """Apply filter: drop signals where pct_of_day_high < threshold.

    Signals with NaN pct_of_day_high (no prior RTH bars) are KEPT — they
    represent very-early-day signals before any meaningful intraday high
    has formed, which the live filter would also pass through.

    Returns metrics for the kept set.
    """
    # baseline = all signals, filtered = signals where pct >= threshold (or NaN)
    keep = slice_df["pct_of_day_high"].isna() | (slice_df["pct_of_day_high"] >= threshold)
    kept = slice_df[keep]
    dropped = slice_df[~keep]
    return {
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "pnl_kept": kept["pnl_dollar"].sum(),
        "pnl_dropped": dropped["pnl_dollar"].sum(),
        "winners_dropped": (dropped["pnl_dollar"] > 0).sum(),
        "losers_dropped": (dropped["pnl_dollar"] < 0).sum(),
    }


def fmt(v: float) -> str:
    return f"${v:>10,.0f}"


def main():
    df = load_signals()
    df = compute_day_high_per_signal(df)

    print()
    print("=" * 100)
    print("Distribution of pct_of_day_high (the filter input):")
    print("=" * 100)
    print(df["pct_of_day_high"].describe(percentiles=[.05, .1, .25, .5, .75, .9, .95]))
    print()

    # Sanity check on HTCO if it's in cache
    htco = df[df["symbol"] == "HTCO"]
    if len(htco):
        print("HTCO signals in cache:")
        print(htco[["date", "entry_price", "day_high", "pct_of_day_high",
                    "pnl_dollar"]].to_string())
        print()

    splits = split(df)
    print("=" * 100)
    print("Split sizes:")
    for name, s in splits.items():
        n = len(s)
        pnl = s["pnl_dollar"].sum()
        wr = (s["pnl_dollar"] > 0).mean() * 100 if n else 0
        print(f"  {name:25s}: n={n:>5,}, baseline P&L={fmt(pnl)}, WR={wr:.1f}%")
    print()

    print("=" * 100)
    print("Threshold sweep (filter: skip if pct_of_day_high < threshold):")
    print("=" * 100)
    print(f"{'Split':<25} {'Threshold':>10} {'n_kept':>8} {'n_dropped':>10} "
          f"{'P&L_kept':>13} {'Δ_vs_base':>13} {'Winners_dropped':>16} "
          f"{'Losers_dropped':>15}")
    print("-" * 110)
    for name, s in splits.items():
        baseline_pnl = s["pnl_dollar"].sum()
        baseline_n = len(s)
        for thr in [0.0] + THRESHOLDS:  # 0.0 = baseline
            r = evaluate(s, thr)
            delta = r["pnl_kept"] - baseline_pnl
            print(f"{name:<25} {thr:>10.2f} {r['n_kept']:>8,} {r['n_dropped']:>10,} "
                  f"{fmt(r['pnl_kept']):>13} {fmt(delta):>13} "
                  f"{r['winners_dropped']:>16,} {r['losers_dropped']:>15,}")
        print()

    print("=" * 100)
    print("Decision matrix — best threshold by P&L on each split:")
    print("=" * 100)
    for name, s in splits.items():
        results = []
        for thr in THRESHOLDS:
            r = evaluate(s, thr)
            results.append((thr, r["pnl_kept"], r["n_dropped"]))
        results.sort(key=lambda x: -x[1])  # by pnl desc
        print(f"\n{name}:")
        for thr, pnl, dropped in results[:5]:
            print(f"  thr={thr:.2f}: P&L={fmt(pnl)}, n_dropped={dropped}")

    print()
    print("=" * 100)
    print("Anti-overfit verdict:")
    print("=" * 100)
    print("- Pick threshold from TRAIN. If it survives VAL and OOS without major")
    print("  P&L degradation vs baseline (or improves), it's robust.")
    print("- If TRAIN-best fails on OOS, the filter is overfit — abandon.")
    print("- If TRAIN-best is better than baseline on OOS, we have evidence the")
    print("  filter generalizes.")


if __name__ == "__main__":
    main()
