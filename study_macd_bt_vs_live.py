"""MACD wave BT vs LIVE side-by-side analysis.

Goal: identify the structural gap between BT projected revenue and LIVE
actual P&L for the MACD wave strategy. Categorize trades:
  A) Same symbol-date in BOTH BT and LIVE → exit-mechanism delta
  B) Symbol-date only in BT (LIVE missed) → "missed alpha"
  C) Symbol-date only in LIVE (BT skipped) → "extra noise"

Quantify each bucket's contribution to the gap. Recommend the highest-
ROI fix.

USAGE
-----
    python3 study_macd_bt_vs_live.py /tmp/macd_bt_full.csv
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).parent
LIVE_DB = ROOT / 'data' / 'trades.db'


def load_live(start_date: str, end_date: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(LIVE_DB))
    df = pd.read_sql(f"""
        SELECT trade_date as date, symbol, fill_price as live_entry,
               exit_price as live_exit, exit_reason as live_exit_reason,
               COALESCE(pnl, 0) as live_pnl,
               filled_at, exited_at
          FROM trades
         WHERE strategy='macd_wave' AND exit_price IS NOT NULL
           AND fill_price IS NOT NULL
           AND trade_date BETWEEN '{start_date}' AND '{end_date}'
    """, conn)
    conn.close()
    return df


def load_bt(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df = df.rename(columns={'entry_price': 'bt_entry', 'exit_price': 'bt_exit',
                             'exit_reason': 'bt_exit_reason', 'pnl_dollar': 'bt_pnl'})
    return df[['date', 'symbol', 'bt_entry', 'bt_exit', 'bt_exit_reason', 'bt_pnl',
               'cross_time_min', 'macd_hist_pct', 'vol_at_cross']]


def main():
    if len(sys.argv) < 2:
        print("Usage: study_macd_bt_vs_live.py <bt_csv>")
        sys.exit(1)
    bt_csv = sys.argv[1]
    live = load_live('2026-03-30', '2026-05-06')
    bt = load_bt(bt_csv)
    print(f"LIVE: {len(live)} trades, sum ${live['live_pnl'].sum():+,.0f}")
    print(f"BT:   {len(bt)} trades, sum ${bt['bt_pnl'].sum():+,.0f}")
    print(f"Gap:  ${live['live_pnl'].sum() - bt['bt_pnl'].sum():+,.0f}")

    # Aggregate live by (symbol, date) — multiple LIVE trades per symbol-day combine
    live_agg = live.groupby(['date','symbol']).agg(
        live_n=('live_pnl', 'count'),
        live_pnl=('live_pnl', 'sum'),
        live_entry=('live_entry', 'first'),
        live_exit_reason=('live_exit_reason', lambda s: '|'.join(s)),
    ).reset_index()

    bt_agg = bt.groupby(['date','symbol']).agg(
        bt_n=('bt_pnl', 'count'),
        bt_pnl=('bt_pnl', 'sum'),
        bt_entry=('bt_entry', 'first'),
        bt_exit_reason=('bt_exit_reason', lambda s: '|'.join(s)),
    ).reset_index()

    merged = live_agg.merge(bt_agg, on=['date','symbol'], how='outer', indicator=True)

    # Bucketing
    both = merged[merged['_merge']=='both']
    only_live = merged[merged['_merge']=='left_only']
    only_bt = merged[merged['_merge']=='right_only']

    print()
    print(f"{'='*100}")
    print("BUCKETING — symbol-date level")
    print(f"{'='*100}")
    print(f"  BOTH (overlap): {len(both)} pairs, "
          f"LIVE ${both['live_pnl'].sum():+,.0f}, BT ${both['bt_pnl'].sum():+,.0f}, "
          f"Δ ${both['live_pnl'].sum() - both['bt_pnl'].sum():+,.0f}")
    print(f"  LIVE-only:      {len(only_live)} pairs, LIVE ${only_live['live_pnl'].sum():+,.0f}")
    print(f"  BT-only:        {len(only_bt)} pairs, BT ${only_bt['bt_pnl'].sum():+,.0f}")

    # Detail of overlapping pairs (the exit-mechanism delta)
    print()
    print(f"{'='*100}")
    print("OVERLAP DETAIL — same symbol-date in BT and LIVE (exit-mechanism delta)")
    print(f"{'='*100}")
    print(f"{'date':>11} | {'symbol':>6} | {'BT entry':>8} | {'LIVE entry':>10} | "
          f"{'BT $':>9} | {'LIVE $':>9} | {'Δ':>9} | {'BT exit':>15} | {'LIVE exit':>20}")
    print('-' * 130)
    both_sorted = both.sort_values('date')
    for _, r in both_sorted.iterrows():
        delta = r['live_pnl'] - r['bt_pnl']
        print(f"{r['date']:>11} | {r['symbol']:>6} | ${r['bt_entry']:>7.2f} | ${r['live_entry']:>9.2f} | "
              f"${r['bt_pnl']:>+8,.0f} | ${r['live_pnl']:>+8,.0f} | ${delta:>+8,.0f} | "
              f"{r['bt_exit_reason'][:15]:>15} | {r['live_exit_reason'][:20]:>20}")

    # Break down LIVE-only and BT-only sums
    print()
    print(f"{'='*100}")
    print("LIVE-ONLY (extra noise) — top 10 worst")
    print(f"{'='*100}")
    only_live_sorted = only_live.sort_values('live_pnl').head(10)
    for _, r in only_live_sorted.iterrows():
        print(f"  {r['date']} {r['symbol']:>6}  ${r['live_pnl']:>+8,.0f}  "
              f"entry=${r['live_entry']:>6.2f}  exit_reason={r['live_exit_reason']}")

    print()
    print(f"{'='*100}")
    print("BT-ONLY (missed alpha) — top 10 best")
    print(f"{'='*100}")
    only_bt_sorted = only_bt.sort_values('bt_pnl', ascending=False).head(10)
    for _, r in only_bt_sorted.iterrows():
        print(f"  {r['date']} {r['symbol']:>6}  ${r['bt_pnl']:>+8,.0f}  "
              f"entry=${r['bt_entry']:>6.2f}  exit_reason={r['bt_exit_reason']}")

    # Decomposition table
    print()
    print(f"{'='*100}")
    print("GAP DECOMPOSITION")
    print(f"{'='*100}")
    bt_total = bt['bt_pnl'].sum()
    live_total = live['live_pnl'].sum()
    overlap_bt = both['bt_pnl'].sum()
    overlap_live = both['live_pnl'].sum()
    exit_mech_delta = overlap_live - overlap_bt    # negative = LIVE worse on same trade
    extra_noise = only_live['live_pnl'].sum()      # LIVE-only P&L
    missed_alpha = only_bt['bt_pnl'].sum()         # BT-only P&L (LIVE didn't capture)

    print(f"  BT total                     ${bt_total:>+10,.0f}")
    print(f"  LIVE total                   ${live_total:>+10,.0f}")
    print(f"  Total gap (LIVE - BT)        ${live_total - bt_total:>+10,.0f}")
    print()
    print(f"  Decomposed:")
    print(f"    1. Exit-mechanism delta    ${exit_mech_delta:>+10,.0f}  "
          f"(same trades, BT vs LIVE exit P&L difference)")
    print(f"    2. Extra noise (LIVE-only) ${extra_noise:>+10,.0f}  "
          f"(trades LIVE took that BT didn't, P&L total)")
    print(f"    3. Missed alpha (BT-only)  ${missed_alpha:>+10,.0f}  "
          f"(trades BT took that LIVE missed — would have added this if captured)")
    print()
    sanity = exit_mech_delta + extra_noise - missed_alpha
    print(f"  Sanity check: 1 + 2 - 3 = ${sanity:>+10,.0f} (should equal {live_total - bt_total:+,.0f})")

    # Save full merged for further analysis
    merged.to_csv(ROOT/'analysis_results'/'macd_bt_vs_live_merged.csv', index=False)
    print()
    print("Saved: analysis_results/macd_bt_vs_live_merged.csv")


if __name__ == '__main__':
    main()
