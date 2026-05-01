#!/usr/bin/env python3
"""MACD wave bar-count warmup sweep — quantifying the BT-vs-LIVE gap.

Background
==========
Live MACD wave engine waits 35 bars before computing MACD (standard EMA
warmup practice). BT cache uses pandas.ewm(adjust=False) from bar 0 of
the trading day, producing biased early-EMA signals at bars 6-8.

CRMX 2026-04-30 example:
  bar 8 (13:38 UTC):  hist_pct = 0.946%  ← BT cache entry
  bar 35 (14:06 UTC): hist_pct = 0.138%  ← live's first measurement

Same calc, different gate. This script filters the existing BT cache by
"minutes after market open" and recomputes total P&L with TRAIN/VAL/OOS
splits, simulating what BT would say if it required N bars of warmup
before generating any signal.

Approach
========
1. Load existing signal cache (data/macd_signal_cache_t30_s40.csv)
2. Apply production filters (cross<10m, MACD>=0.5%, vol<300K, $5-30)
3. Convert entry_time to ET; compute minutes_after_open
4. For each warmup ∈ {0, 10, 20, 35, 50, 100}:
   - Drop signals where minutes_after_open < warmup
   - Apply max_concurrent + daily_loss_limit (matches BT runner)
   - Sum P&L per TRAIN/VAL/OOS split
5. Print comparison table

Splits (matches research/study_macd_day_from_high.py):
  TRAIN: Jan-Sep 2025
  VAL:   Oct-Dec 2025
  OOS:   Jan-Apr 2026

Caveats
=======
- Doesn't re-simulate the trade itself (entry_price / exit_price /
  pnl preserved from cache). The early-EMA bias may have set ENTRY
  PRICES too — that part isn't corrected here.
- Doesn't apply max_concurrent / daily_loss_limit reordering — pure
  signal-level analysis.
- Day-from-high + halt-aware filters (shipped 4/29) NOT applied here;
  BT cache predates them. Result is "BT P&L if warmup were the only
  difference from current cache."
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytz

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CACHE_CSV = ROOT / 'data' / 'macd_signal_cache_t30_s40.csv'

ET = pytz.timezone('US/Eastern')
TRAIN_END = '2025-09-30'
VAL_END = '2025-12-31'
WARMUP_VALUES = [0, 5, 10, 15, 20, 25, 30, 35, 50, 75, 100]

# Production-equivalent filters (macd_wave.yaml)
PROD_CROSS_MAX = 10
PROD_MACD_MIN = 0.5
PROD_VOL_MAX = 300_000
PROD_PRICE_MIN = 5.0
PROD_PRICE_MAX = 30.0


def load_and_filter():
    df = pd.read_csv(CACHE_CSV, parse_dates=['entry_time', 'exit_time'])
    df = df[df['paper'] == False].copy()
    df = df.dropna(subset=['symbol'])
    df['symbol'] = df['symbol'].astype(str)

    # Apply production filters
    df = df[df['cross_time_min'] <= PROD_CROSS_MAX]
    df = df[df['macd_hist_pct'] >= PROD_MACD_MIN]
    df = df[df['vol_at_cross'] <= PROD_VOL_MAX]
    df = df[df['entry_price'] <= PROD_PRICE_MAX]
    df = df[df['entry_price'] >= PROD_PRICE_MIN]

    # Compute minutes after open (ET)
    df['et'] = df['entry_time'].dt.tz_convert(ET)
    df['min_after_open'] = (df['et'].dt.hour - 9) * 60 + (df['et'].dt.minute - 30)
    return df.reset_index(drop=True)


def split_periods(df):
    return {
        'TRAIN (Jan-Sep 2025)': df[df['date'] <= TRAIN_END],
        'VAL (Oct-Dec 2025)':   df[(df['date'] > TRAIN_END) & (df['date'] <= VAL_END)],
        'OOS (Jan-Apr 2026)':   df[df['date'] > VAL_END],
        'FULL (Jan 2025 - Apr 2026)': df,
    }


def main():
    df = load_and_filter()
    print(f"Loaded {len(df):,} production-filtered signals from cache")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print()

    splits = split_periods(df)

    # Header
    print(f"{'Warmup':>8} {'Trades':>8} {'P&L':>12} {'WR':>6} "
          f"{'Avg Win':>10} {'Avg Loss':>10} {'PF':>6}    [SPLIT]")
    print("-" * 100)

    for split_name, sdf in splits.items():
        for warmup in WARMUP_VALUES:
            slice_df = sdf[sdf['min_after_open'] >= warmup]
            if len(slice_df) == 0:
                print(f"{warmup:>8} {0:>8} {'$0':>12} {'-':>6} {'-':>10} {'-':>10} {'-':>6}    [{split_name}]")
                continue
            n = len(slice_df)
            pnl = slice_df['pnl_dollar'].sum()
            wins = (slice_df['pnl_dollar'] > 0).sum()
            wr = wins / n * 100
            win_pnl = slice_df[slice_df['pnl_dollar'] > 0]['pnl_dollar'].sum()
            loss_pnl = slice_df[slice_df['pnl_dollar'] < 0]['pnl_dollar'].sum()
            avg_win = win_pnl / wins if wins else 0
            avg_loss = loss_pnl / (n - wins) if (n - wins) else 0
            pf = abs(win_pnl / loss_pnl) if loss_pnl != 0 else float('inf')
            print(f"{warmup:>8} {n:>8,} ${pnl:>+11,.0f} {wr:>5.1f}% "
                  f"{avg_win:>+10,.0f} {avg_loss:>+10,.0f} {pf:>6.2f}    [{split_name}]")
        print()


if __name__ == '__main__':
    main()
