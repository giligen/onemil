#!/usr/bin/env python3
"""ORB stop-limit fill-model sweep — does widening the entry buffer help?

Background
==========
BT currently models 100% fill at trigger price (range_high * 1.003) for any
post-range bar with high >= range_high. Real stop-limits can MISS fills on
fast spikes through the limit price. Widening the buffer (30bps → 50/75/100)
trades slippage cost for fewer missed fills.

This sweep evaluates that tradeoff using OHLC data:
1. For each existing BT trade in `orb_static_lock_trades.csv`, reload
   1-min bars from cache and re-simulate fill with a realistic stop-limit
   model:
     - Trigger: bar.high >= range_high
     - Fill: bar.low <= limit_price (= range_high × (1 + buffer_bps/10000))
       on the trigger bar OR any subsequent bar within 60-min window.
2. If trade fills, compute P&L using the same exit (lock/stop) the BT
   recorded (we don't re-simulate the exit).
3. If trade does NOT fill, P&L = 0.
4. Compare aggregate P&L across buffer settings.

Caveats
=======
- 1-min OHLC cannot capture intra-bar tick dynamics. Model OVERESTIMATES
  fills (a bar with low <= limit might have spiked past limit too fast).
  Treat results as a CEILING.
- Existing exits (lock/stop) are taken as given even though a different
  fill price means slightly different stop level / lock arming. Small
  approximation; the dominant variable is fill rate.
- Position sizing uses the BT's `_sized_pnl` field — we scale by the
  ratio of new fill price to original entry_price.

Usage
=====
    python scripts/orb_fill_model_sweep.py
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CACHE_DB = ROOT / 'data' / 'cache.db'
TRADES_CSV = ROOT / 'analysis_results' / 'orb_static_lock_trades.csv'

CURRENT_BT_BUFFER_BPS = 30  # what's already baked into entry_price in CSV
TIME_STOP_MINUTES = 60       # ORB cancels unfilled buy-stops after 60 min

# Buffer values to sweep. Including 30 (current) as baseline so we can
# verify the model recovers ~current BT P&L when buffer matches.
SWEEP_BUFFERS_BPS = [30, 50, 75, 100, 150]


def load_bars(con: sqlite3.Connection, symbol: str, bar_date: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, volume "
        "FROM intraday_bars_1min WHERE symbol=? AND bar_date=? "
        "ORDER BY timestamp",
        con, params=(symbol, bar_date),
    )
    if df.empty:
        return df
    df['ts'] = pd.to_datetime(df['timestamp'], utc=True)
    return df


def simulate_fill(bars: pd.DataFrame, range_high: float, buffer_bps: int):
    """Simulate stop-limit fill. Returns (filled, fill_price, fill_ts).

    Logic:
    - Range close at 9:35 ET (14:35 UTC summer / 13:35 UTC winter — we use
      the bars themselves to find first bar at or after range_high).
    - Trigger: any bar where bar.high >= range_high
    - Limit fill: same bar's bar.low <= limit_price OR a subsequent bar's
      bar.low <= limit_price (within 60-min window)
    - Fill price: max(range_high, min(limit_price, bar.open))
    """
    if bars.empty:
        return False, None, None

    limit_price = range_high * (1 + buffer_bps / 10000)
    # Range starts at 9:30 ET = 13:30 or 14:30 UTC. ORB scans bars after the
    # 5-min range close — so we look for fills in bars at index >= 5 from
    # 9:30 ET. We'll just scan the whole post-9:30 ET set for safety; the
    # earliest relevant bar will be the one that actually triggers.
    # Conservative fill: limit-buy fills at LIMIT_PRICE (never below) — assumes
    # ask was at or near the limit when our order matched. Optimistic models
    # (fill at bar.open if below limit) are unrealistic because limit-buy
    # only triggers AFTER stop, and stop firing implies price is at/above
    # limit at that moment.
    triggered = False
    triggered_at = None
    for _, b in bars.iterrows():
        if not triggered:
            if b['high'] >= range_high:
                triggered = True
                triggered_at = b['ts']
                # Same-bar fill check: did the bar's range include limit?
                # If bar.low <= limit, the price retested the limit zone → fill.
                if b['low'] <= limit_price:
                    return True, float(limit_price), b['ts']
                # Else: spike through — order sits as live limit-buy
        else:
            # Time-stop check (60 min)
            if (b['ts'] - triggered_at).total_seconds() / 60 > TIME_STOP_MINUTES:
                return False, None, None
            if b['low'] <= limit_price:
                return True, float(limit_price), b['ts']
    return False, None, None


def main():
    con = sqlite3.connect(str(CACHE_DB))
    df = pd.read_csv(TRADES_CSV)
    print(f"Loaded {len(df):,} BT trades from {TRADES_CSV.name}")

    # Pre-cache bars per (symbol, date) since we'll re-use across buffer sweeps.
    print("Pre-loading bars...")
    bars_cache = {}
    pairs = list(df.groupby(['symbol', 'date']).groups.keys())
    for i, (sym, date) in enumerate(pairs):
        if i % 200 == 0:
            print(f"  {i}/{len(pairs)}")
        bars_cache[(sym, date)] = load_bars(con, sym, date)
    con.close()
    print(f"  cached {len(bars_cache):,} (symbol, date) pairs")

    # For each buffer, simulate
    print()
    print(f"{'buffer (bps)':>12} {'n_filled':>10} {'fill_rate':>10} {'gross_pnl':>14} {'sized_pnl':>14} {'Δ vs current_BT':>18}")
    print("-" * 90)

    # Current BT baseline: just sum existing pnl/sized_pnl
    current_total_pnl = df['pnl'].sum()
    current_total_sized = df['_sized_pnl'].sum()

    for buffer_bps in SWEEP_BUFFERS_BPS:
        n_filled = 0
        gross_pnl_sum = 0.0
        sized_pnl_sum = 0.0
        for _, row in df.iterrows():
            sym = row['symbol']
            date = row['date']
            entry_price_bt = float(row['entry_price'])
            # Recover range_high from BT's known formula
            range_high = entry_price_bt / (1 + CURRENT_BT_BUFFER_BPS / 10000)
            bars = bars_cache.get((sym, date))
            if bars is None or bars.empty:
                continue
            filled, fill_price, _ = simulate_fill(bars, range_high, buffer_bps)
            if not filled:
                continue
            n_filled += 1
            # Adjust BT's recorded P&L for the new fill price.
            # Gross: pnl scales with (exit_price - new_fill) / (exit_price - bt_entry)
            # Use ratio approach: BT was bt_entry, exit_price implicit.
            # Simpler: derive exit from existing pnl and entry, then recompute.
            shares_implied = row['pnl'] / (row['pnl_pct'] / 100 * entry_price_bt) \
                if row['pnl_pct'] != 0 else 0
            # Even simpler: pnl_pct is (exit - entry) / entry. So
            # exit = entry * (1 + pnl_pct/100). Then new_pnl = (exit - new_fill) * shares.
            # But we don't have shares. Use _sized_pnl scaling.
            # Cleanest: pnl_per_share = pnl_pct/100 * entry_price (positive=win)
            #           total_pnl_at_old_entry = row['pnl']
            #           shares ≈ row['pnl'] / pnl_per_share
            pnl_per_share = (row['pnl_pct'] / 100) * entry_price_bt
            if abs(pnl_per_share) > 1e-9:
                shares = row['pnl'] / pnl_per_share
                exit_price = entry_price_bt + pnl_per_share
                new_pnl = (exit_price - fill_price) * shares
                # _sized_pnl = pnl × ratio; preserve ratio
                ratio = (row['_sized_pnl'] / row['pnl']) if row['pnl'] else 1.0
                new_sized = new_pnl * ratio
            else:
                new_pnl = 0.0
                new_sized = 0.0
            gross_pnl_sum += new_pnl
            sized_pnl_sum += new_sized
        fill_rate = n_filled / len(df) * 100
        delta_vs_current = sized_pnl_sum - current_total_sized
        print(f"{buffer_bps:>12d} {n_filled:>10,} {fill_rate:>9.1f}% "
              f"{gross_pnl_sum:>+14,.0f} {sized_pnl_sum:>+14,.0f} "
              f"{delta_vs_current:>+18,.0f}")

    print()
    print(f"Reference (current BT, 100% fill at trigger):")
    print(f"  Gross: ${current_total_pnl:+,.0f}    Sized: ${current_total_sized:+,.0f}")
    print()
    print("INTERPRETATION:")
    print("  - Buffer 30bps with realistic model = baseline; should be NEAR but ≤ current BT")
    print("    (current BT assumes 100% fill; realistic model loses some)")
    print("  - Wider buffers fill more, accept higher slippage. Best buffer is the one")
    print("    that maximizes sized_pnl across the sweep.")
    print("  - This model is OPTIMISTIC vs reality (sub-bar tick dynamics not captured).")


if __name__ == '__main__':
    main()
