"""Diagnostic: prove BT vs LIVE detect different crosses because of
different "open price" references.

For a sample of LIVE trades whose cross detection mismatches BT, dump:
  - bar1.open (BT's reference)
  - snapshot.open via daily_bar.open (proxy)
  - When did bar.high first cross bar1.open * 1.10? (BT detection)
  - When did bar.high first cross snapshot.open * 1.10? (LIVE detection — proxy)

Output a table that proves the hypothesis or refutes it.
"""
from datetime import time as dtime
from pathlib import Path
import sqlite3

import pandas as pd

ROOT = Path(__file__).parent
CACHE_DB = ROOT / 'data' / 'cache.db'

# Specific LIVE-vs-BT mismatch cases from earlier analysis
TARGETS = [
    ('EEIQ', '2026-04-02'),  # LIVE cross=2, BT cross=36
    ('MGRT', '2026-04-02'),  # LIVE cross=132, BT cross=8
    ('MODD', '2026-03-31'),  # LIVE cross=114, BT cross=6
    ('UGRO', '2026-03-31'),  # LIVE cross=1, BT cross=10
    ('XNDU', '2026-04-14'),  # LIVE cross=1, BT cross=19
    ('IART', '2026-05-05'),  # LIVE cross=2, BT cross=72
    ('MSAI', '2026-04-13'),  # LIVE cross=2, BT cross=62
    ('HTCO', '2026-04-28'),  # LIVE cross=2, BT cross=3 (close)
]

INTRADAY_PCT = 10.0


def get_bars(symbol: str, date_str: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(CACHE_DB))
    df = pd.read_sql(f"""
        SELECT timestamp as ts, open, high, low, close, volume
          FROM intraday_bars_1min
         WHERE symbol='{symbol}' AND date(timestamp)='{date_str}'
         ORDER BY timestamp
    """, conn)
    conn.close()
    if df.empty:
        return df
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    et = df['ts'].dt.tz_convert('America/New_York')
    # Clip to regular session 9:30-16:00 ET
    mask = (((et.dt.hour == 9) & (et.dt.minute >= 30))
            | ((et.dt.hour > 9) & (et.dt.hour < 16)))
    df = df[mask].reset_index(drop=True)
    df['et'] = et[mask].reset_index(drop=True)
    return df


def get_daily_open(symbol: str, date_str: str) -> float:
    """Daily-bar open (proxy for snapshot.open since we don't have snapshots cached)."""
    conn = sqlite3.connect(str(CACHE_DB))
    cur = conn.cursor()
    cur.execute("""
        SELECT open FROM daily_bars
         WHERE symbol=? AND bar_date=?
         LIMIT 1
    """, (symbol, date_str))
    row = cur.fetchone()
    conn.close()
    return float(row[0]) if row else 0.0


def find_first_cross(bars: pd.DataFrame, anchor_open: float) -> tuple:
    """Return (bar_idx_first_cross, et_time_first_cross, bar_high) where bar.high >= anchor*1.10.
    Returns (-1, None, 0) if never."""
    if anchor_open <= 0 or bars.empty:
        return (-1, None, 0)
    threshold = anchor_open * (1 + INTRADAY_PCT/100.0)
    for i, row in bars.iterrows():
        if row['high'] >= threshold:
            return (i, row['et'], row['high'])
    return (-1, None, 0)


def main():
    print(f"{'sym':<6} {'date':<12} | {'bar1.open':>10} {'daily.open':>10} {'gap %':>7} | "
          f"{'BT_cross_min':>12} {'LIVE_cross_min':>15} | {'cross diff':>10}")
    print('-' * 110)

    for sym, dt in TARGETS:
        bars = get_bars(sym, dt)
        if bars.empty:
            print(f"{sym:<6} {dt:<12} | NO BARS")
            continue
        bar1_open = float(bars.iloc[0]['open'])
        daily_open = get_daily_open(sym, dt)
        gap = ((daily_open - bar1_open) / bar1_open * 100) if bar1_open > 0 else 0

        bt_idx, bt_et, bt_high = find_first_cross(bars, bar1_open)
        live_idx, live_et, live_high = find_first_cross(bars, daily_open)

        bt_min = bt_idx + 1 if bt_idx >= 0 else None
        live_min = live_idx + 1 if live_idx >= 0 else None
        diff = (bt_min - live_min) if (bt_min is not None and live_min is not None) else None

        print(f"{sym:<6} {dt:<12} | ${bar1_open:>8.3f} ${daily_open:>8.3f} {gap:>+6.2f}% | "
              f"{bt_min!s:>12} {live_min!s:>15} | "
              f"{(str(diff)+' min') if diff is not None else 'n/a':>10}")

    print()
    print("Interpretation:")
    print("  bar1.open: BT's anchor (first 1-min bar open)")
    print("  daily.open: proxy for snapshot.open (LIVE's anchor, official open auction)")
    print("  gap %: how much daily.open differs from bar1.open")
    print("  cross diff: BT minutes - LIVE minutes (positive = LIVE detected earlier)")
    print()
    print("If hypothesis is right: gap > 0 → LIVE threshold higher → BT detects LATER (positive diff)")
    print("                       gap < 0 → LIVE threshold lower → LIVE detects LATER (negative diff)")


if __name__ == '__main__':
    main()
