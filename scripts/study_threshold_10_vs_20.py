#!/usr/bin/env python3
"""Bull flag 10% vs 20% intraday-change threshold A/B — weekly breakdown.

Methodology:
  1. Read the existing 10%-cache trades (already filtered to >=10% movers
     at cache build time).
  2. Backfill `intraday_change_at_entry` per row by replaying pre-entry
     bars from intraday_bars_1min — same formula as the live scanner:
       max(gap_pct, range_pct) where
         gap_pct   = (cur - prev_close) / prev_close * 100
         range_pct = (day_high - day_low) / day_low * 100
       computed at the entry minute.
  3. Bucket into A-tier (>=20%) and Extras (10-19.99%) — the live filter
     today drops Extras; Extras-on is the proposed change.
  4. Bucket trades into ISO-week, sum P&L per bucket per tier, report
     weekly + cumulative.

This does NOT replay BT logic; it uses the already-simulated trade pnl
from the cache. The only post-hoc layer is the threshold split. So
results are exactly what the BT would output if --threshold 10 vs 20
were rerun against this cache.
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
CACHE_CSV = ROOT / 'data' / 'bull_flag_cache_e50_x30.csv'
DB = ROOT / 'data' / 'cache.db'


def parse_entry_time(date_str: str, entry_time_et: str) -> pd.Timestamp:
    """Combine `2026-04-09` + `09:36:00` ET → tz-aware UTC."""
    et = pd.Timestamp(f"{date_str} {entry_time_et}", tz='US/Eastern')
    return et.tz_convert('UTC')


def compute_intraday_change_at_entry(con, symbol: str, date_str: str,
                                     entry_ts_utc: pd.Timestamp,
                                     prev_close: float) -> float | None:
    """Replay bars up to entry, compute max(gap_pct, range_pct).

    Mirrors scanner.realtime_scanner._run_intraday_cycle logic.
    """
    df = pd.read_sql_query(
        "SELECT timestamp, high, low, close, open "
        "FROM intraday_bars_1min WHERE symbol=? AND bar_date=? "
        "ORDER BY timestamp",
        con, params=(symbol, date_str),
    )
    if df.empty or prev_close <= 0:
        return None
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Restrict to RTH start through entry minute (inclusive of entry bar)
    rth_start = pd.Timestamp(f"{date_str} 13:30", tz='UTC')
    pre_entry = df[(df['timestamp'] >= rth_start) &
                   (df['timestamp'] <= entry_ts_utc)]
    if pre_entry.empty:
        return None

    day_high = pre_entry['high'].max()
    day_low = pre_entry['low'].min()
    cur = pre_entry['close'].iloc[-1]

    gap_pct = (cur - prev_close) / prev_close * 100
    range_pct = ((day_high - day_low) / day_low * 100) if day_low > 0 else 0
    return max(gap_pct, range_pct)


def get_prev_close(con, symbol: str, date_str: str) -> float | None:
    row = con.execute(
        "SELECT close FROM daily_bars WHERE symbol=? AND bar_date<? "
        "ORDER BY bar_date DESC LIMIT 1",
        (symbol, date_str),
    ).fetchone()
    return float(row[0]) if row else None


def main():
    print(f"Reading cache: {CACHE_CSV}")
    df = pd.read_csv(CACHE_CSV)
    print(f"  {len(df)} cache rows")

    con = sqlite3.connect(str(DB))

    # Backfill intraday_change_at_entry for rows missing it
    backfilled = 0
    skipped = 0
    ic_values = []
    for i, row in df.iterrows():
        existing = row.get('intraday_change_at_entry')
        try:
            existing_f = float(existing)
            if not pd.isna(existing_f):
                ic_values.append(existing_f)
                continue
        except (ValueError, TypeError):
            pass

        symbol = row['symbol']
        date_str = row['date']
        entry_time_et = row['entry_time_et']

        if pd.isna(symbol) or pd.isna(date_str) or pd.isna(entry_time_et):
            ic_values.append(None)
            skipped += 1
            continue

        try:
            entry_ts = parse_entry_time(date_str, entry_time_et)
        except Exception:
            ic_values.append(None)
            skipped += 1
            continue

        prev_close = get_prev_close(con, symbol, date_str)
        if prev_close is None:
            ic_values.append(None)
            skipped += 1
            continue

        ic = compute_intraday_change_at_entry(
            con, symbol, date_str, entry_ts, prev_close
        )
        ic_values.append(ic)
        if ic is not None:
            backfilled += 1
        else:
            skipped += 1

    df['ic_at_entry'] = ic_values
    print(f"  backfilled: {backfilled}, skipped (no data): {skipped}, "
          f"already-set: {len(df) - backfilled - skipped}")

    # Drop rows we couldn't compute
    df = df[df['ic_at_entry'].notna()].copy()
    df['date'] = pd.to_datetime(df['date'])
    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
    df = df[df['pnl'].notna()].copy()
    print(f"  usable rows: {len(df)}")

    # Tier classification
    def tier(ic):
        if ic >= 20:
            return 'A'
        if ic >= 10:
            return 'Extras'
        return 'sub10'

    df['tier'] = df['ic_at_entry'].apply(tier)
    df['week'] = df['date'].dt.to_period('W-SUN').astype(str)

    # Overall summary
    print()
    print("=" * 78)
    print("OVERALL (Jan 2025 - Apr 2026, full cache)")
    print("=" * 78)
    print(f"{'tier':<10} {'n':>5} {'pnl':>12} {'wins':>6} {'WR':>6} {'avg':>10}")
    for t in ['A', 'Extras', 'sub10']:
        sub = df[df['tier'] == t]
        if len(sub) == 0:
            continue
        wins = (sub['pnl'] > 0).sum()
        wr = wins / len(sub) * 100 if len(sub) else 0
        avg = sub['pnl'].mean()
        print(f"  {t:<8} {len(sub):>5d} ${sub['pnl'].sum():>+10,.0f} "
              f"{wins:>5d} {wr:>5.1f}% ${avg:>+8,.0f}")

    print()
    print("THRESHOLD COMPARISON")
    print("=" * 78)
    a = df[df['tier'] == 'A']
    a_plus_extras = df[df['tier'].isin(['A', 'Extras'])]
    print(f"  20% threshold (A only):       n={len(a)}  "
          f"pnl=${a['pnl'].sum():+,.0f}  WR={(a['pnl']>0).mean()*100:.1f}%")
    print(f"  10% threshold (A+Extras):     n={len(a_plus_extras)}  "
          f"pnl=${a_plus_extras['pnl'].sum():+,.0f}  "
          f"WR={(a_plus_extras['pnl']>0).mean()*100:.1f}%")
    delta = a_plus_extras['pnl'].sum() - a['pnl'].sum()
    print(f"  Delta (Extras incremental):   "
          f"pnl_diff=${delta:+,.0f}  trades_added={len(a_plus_extras)-len(a)}")

    # Weekly breakdown
    print()
    print("WEEKLY BREAKDOWN (W-SUN buckets, * = week with ≥1 Extras trade)")
    print("=" * 78)
    print(f"{'week':<14} {'A_n':>4} {'A_pnl':>10} {'E_n':>4} "
          f"{'E_pnl':>10} {'10%_pnl':>10} {'20%_pnl':>10} {'diff':>9}")
    weekly = []
    for week in sorted(df['week'].unique()):
        wk = df[df['week'] == week]
        a_w = wk[wk['tier'] == 'A']
        e_w = wk[wk['tier'] == 'Extras']
        s_w = wk[wk['tier'] == 'sub10']
        a_pnl = a_w['pnl'].sum()
        e_pnl = e_w['pnl'].sum()
        s_pnl = s_w['pnl'].sum()
        pnl_20 = a_pnl  # 20% threshold = A only
        pnl_10 = a_pnl + e_pnl + s_pnl  # 10% threshold = all (sub10 should be 0)
        marker = '*' if len(e_w) else ' '
        weekly.append({
            'week': week, 'a_n': len(a_w), 'a_pnl': a_pnl,
            'e_n': len(e_w), 'e_pnl': e_pnl,
            'pnl_10': pnl_10, 'pnl_20': pnl_20,
            'diff': pnl_10 - pnl_20,
        })
        print(f"{marker}{week:<13} {len(a_w):>4d} ${a_pnl:>+8,.0f} "
              f"{len(e_w):>4d} ${e_pnl:>+8,.0f} "
              f"${pnl_10:>+8,.0f} ${pnl_20:>+8,.0f} ${pnl_10-pnl_20:>+7,.0f}")

    # Risk breakdown: how many "bad weeks" does Extras introduce?
    print()
    print("RISK ANALYSIS (Extras-only, weeks where Extras hurt)")
    print("=" * 78)
    extras_weekly = pd.DataFrame(weekly)
    bad_extras_wks = extras_weekly[extras_weekly['e_pnl'] < 0]
    good_extras_wks = extras_weekly[extras_weekly['e_pnl'] > 0]
    flat_wks = extras_weekly[(extras_weekly['e_n'] == 0)]
    print(f"  Weeks where Extras DESTROYED: {len(bad_extras_wks)}  "
          f"sum=${bad_extras_wks['e_pnl'].sum():+,.0f}  "
          f"worst=${bad_extras_wks['e_pnl'].min() if len(bad_extras_wks) else 0:+,.0f}")
    print(f"  Weeks where Extras HELPED:    {len(good_extras_wks)}  "
          f"sum=${good_extras_wks['e_pnl'].sum():+,.0f}  "
          f"best=${good_extras_wks['e_pnl'].max() if len(good_extras_wks) else 0:+,.0f}")
    print(f"  Weeks with no Extras trade:   {len(flat_wks)}")

    # Worst rolling 4-week Extras-only DD
    extras_weekly['e_pnl_cum'] = extras_weekly['e_pnl'].cumsum()
    extras_weekly['e_pnl_dd'] = (
        extras_weekly['e_pnl_cum']
        - extras_weekly['e_pnl_cum'].cummax()
    )
    worst_dd_idx = extras_weekly['e_pnl_dd'].idxmin()
    worst_dd_val = extras_weekly.loc[worst_dd_idx, 'e_pnl_dd']
    worst_dd_week = extras_weekly.loc[worst_dd_idx, 'week']
    print(f"  Extras-only max drawdown: ${worst_dd_val:+,.0f} "
          f"trough at {worst_dd_week}")

    # Save weekly CSV
    out = ROOT / 'study_threshold_weekly.csv'
    extras_weekly.to_csv(out, index=False)
    print()
    print(f"Wrote weekly CSV: {out}")

    con.close()


if __name__ == '__main__':
    main()
