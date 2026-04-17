#!/usr/bin/env python3
"""Exit-quality research — pick 20 mid-range Q1 winners, trace post-exit.

For each selected trade:
  1. Load 1-min bars for (symbol, date) from data/cache.db
  2. Identify the bar at exit_time_et
  3. Measure post-exit trajectory:
     - max_after_exit: highest bar high after exit (to EOD)
     - max_after_exit_60min: max high within 60 min of exit
     - min_after_exit: lowest bar low after exit (to EOD)
     - time_to_max: minutes until post-exit max is reached
     - time_to_min: minutes until post-exit min is reached
  4. Compute R-multiples:
     - held_r:        what we actually got
     - max_possible_r: what we'd have gotten at post-exit peak
     - reversal_r:    what we'd have LOST holding to post-exit min
  5. Classify trade:
     - EARLY: max_possible_r > held_r + 1.0R (>1R upside left) AND min > exit
     - GOOD:  max within 0.5R AND min down < 0.5R (caught the peak)
     - LUCKY: min_after_exit below exit - 0.5R (trailing stop saved us)
     - MIXED: rode into reversal (missed some upside AND gave back more)
  6. Pattern analysis by exit_reason + time-of-day + initial R
"""
from __future__ import annotations
import csv
import sqlite3
from datetime import datetime, timedelta
from typing import Optional


IN_PATH = '/tmp/ttf_cache/bt_q1_v3_TTF_on.csv'
DB_PATH = 'data/cache.db'


# 20 picked trades — copied from selection output
PICKED = [
    ('RRGB', '2026-02-26'),
    ('LUNL', '2026-03-20'),
    ('DBGI', '2026-02-13'),
    ('EDHL', '2026-03-13'),
    ('NBIZ', '2026-03-30'),
    ('MCRP', '2026-02-02'),
    ('WBTN', '2026-03-04'),
    ('PDYN', '2026-03-05'),
    ('CDNA', '2026-01-12'),
    ('OPTX', '2026-03-09'),
    ('LFMD', '2026-03-10'),
    ('MKDW', '2026-03-26'),
    ('GLWG', '2026-03-19'),
    ('KPTI', '2026-02-12'),
    ('QCLS', '2026-03-06'),
    ('CHNR', '2026-03-24'),
    ('HYPD', '2026-03-02'),
    ('HBIO', '2026-03-16'),
    ('MVLL', '2026-03-06'),
    ('OKLS', '2026-03-17'),
]


def et_to_utc(date_str: str, time_et: str) -> str:
    """ET -> UTC string that sorts lexically. Handles DST crudely."""
    y, m, d = map(int, date_str.split('-'))
    hh, mm = map(int, time_et.split(':')[:2])
    # Q1 2026: Jan/Feb EST (UTC-5), Mar after DST start (Mar 8 2026) EDT (UTC-4)
    dst_start = datetime(y, 3, 8)   # 2026: second Sun of March = Mar 8
    offset = 4 if datetime(y, m, d) >= dst_start else 5
    utc_h = hh + offset
    utc_d = d + (utc_h // 24)
    utc_h = utc_h % 24
    return f"{y:04d}-{m:02d}-{utc_d:02d}T{utc_h:02d}:{mm:02d}:00+00:00"


def bars_after(conn, symbol, date_str, exit_utc):
    cur = conn.execute(
        "SELECT timestamp, high, low, close, volume FROM intraday_bars_1min "
        "WHERE symbol=? AND bar_date=? AND timestamp > ? "
        "ORDER BY timestamp",
        (symbol, date_str, exit_utc),
    )
    return cur.fetchall()


def minutes_between(a_utc: str, b_utc: str) -> int:
    """Crude: works for same-day bar ISO strings."""
    fmt = "%Y-%m-%dT%H:%M:%S+00:00"
    a = datetime.strptime(a_utc, fmt)
    b = datetime.strptime(b_utc, fmt)
    return int((b - a).total_seconds() / 60)


def analyze_trade(conn, trade):
    sym = trade['symbol']; date = trade['date']
    exit_et = trade['exit_time_et']
    entry = float(trade['entry_price'])
    stop = float(trade['stop_loss'])
    exit_px = float(trade['exit_price'])
    pnl = float(trade['pnl'])
    shares = float(trade['shares'])
    risk_ps = entry - stop
    held_r = (exit_px - entry) / risk_ps if risk_ps > 0 else 0
    exit_utc = et_to_utc(date, exit_et)
    bars = bars_after(conn, sym, date, exit_utc)
    if not bars:
        return None
    # Analyze post-exit
    max_h = max(b[1] for b in bars)
    min_l = min(b[2] for b in bars if b[2] > 0)
    # Find which bar had the max
    max_bar = next(b for b in bars if b[1] == max_h)
    min_bar = next(b for b in bars if b[2] == min_l)
    # Within 60 min
    max_60_bars = bars[:60]
    max_h_60 = max(b[1] for b in max_60_bars) if max_60_bars else max_h
    mins_to_max = minutes_between(exit_utc, max_bar[0])
    mins_to_min = minutes_between(exit_utc, min_bar[0])

    # EOD price (last bar close)
    eod = bars[-1][3]
    # R multiples
    max_possible_r = (max_h - entry) / risk_ps if risk_ps > 0 else 0
    max_possible_r_60 = (max_h_60 - entry) / risk_ps if risk_ps > 0 else 0
    reversal_r = (min_l - entry) / risk_ps if risk_ps > 0 else 0
    eod_r = (eod - entry) / risk_ps if risk_ps > 0 else 0

    # Missed upside (past what we got)
    missed_r = max_possible_r - held_r

    # Classification
    # LUCKY: trailing stop saved us from a material reversal
    # EARLY: material upside left (>1R), limited downside (<0.5R reversal)
    # GOOD: close to peak, small reversal
    # REVERSAL: we exited then it kept falling
    if reversal_r < held_r - 0.8 and mins_to_min < mins_to_max:
        classification = "LUCKY (saved from reversal)"
    elif missed_r > 1.0 and reversal_r > held_r - 0.3:
        classification = "EARLY (upside left, no material give-back)"
    elif missed_r > 0.5 and reversal_r > held_r - 0.5:
        classification = "SLIGHTLY EARLY"
    elif missed_r < 0.5:
        classification = "GOOD (caught near peak)"
    else:
        classification = "MIXED"

    return {
        'sym': sym, 'date': date,
        'exit_et': exit_et,
        'entry': entry, 'stop': stop, 'exit': exit_px, 'eod': eod,
        'held_r': held_r, 'pnl': pnl, 'shares': shares, 'risk_ps': risk_ps,
        'max_h': max_h, 'max_possible_r': max_possible_r,
        'max_h_60': max_h_60, 'max_possible_r_60': max_possible_r_60,
        'mins_to_max': mins_to_max,
        'min_l': min_l, 'reversal_r': reversal_r,
        'mins_to_min': mins_to_min,
        'eod_r': eod_r,
        'missed_r': missed_r,
        'exit_reason': trade['exit_reason'],
        'class': classification,
    }


def main():
    rows = list(csv.DictReader(open(IN_PATH)))
    picked_set = set(PICKED)
    matched = [r for r in rows if (r['symbol'], r['date']) in picked_set]
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA query_only=1")
    print("=" * 140)
    print("  Exit-quality trace — 20 mid-range Q1 2026 winners")
    print("  Columns: actual exit → post-exit trajectory in R-multiples (R = entry - stop)")
    print("=" * 140)
    hdr = (f"  {'sym':<5} {'date':<11} {'x':<5} {'held':>5} "
           f"{'max_R':>6} {'@min_to_max':>12} "
           f"{'max_60_R':>8} {'revsl_R':>7} "
           f"{'@min_to_min':>12} {'eod_R':>6} "
           f"{'missed':>6} {'reason':<22} {'classification':<30}")
    print(hdr)
    print("-" * 140)
    analyses = []
    for t in matched:
        a = analyze_trade(conn, t)
        if a is None:
            continue
        analyses.append(a)
    # Sort by held_R descending for readability
    analyses.sort(key=lambda x: x['held_r'])
    for a in analyses:
        print(f"  {a['sym']:<5} {a['date']:<11} {a['exit_et']:<5} "
              f"{a['held_r']:>+4.2f}R {a['max_possible_r']:>+5.2f}R "
              f"{a['mins_to_max']:>8}m "
              f"{a['max_possible_r_60']:>+6.2f}R {a['reversal_r']:>+6.2f}R "
              f"{a['mins_to_min']:>8}m {a['eod_r']:>+5.2f}R "
              f"{a['missed_r']:>+5.2f}R {a['exit_reason']:<22} {a['class']:<30}")

    # Summary by classification
    from collections import Counter
    class_counts = Counter(a['class'] for a in analyses)
    print()
    print("=" * 80)
    print(f"  Classification distribution (n={len(analyses)})")
    print("=" * 80)
    for c, n in class_counts.most_common():
        pct = n / len(analyses) * 100
        total_missed_r = sum(a['missed_r'] for a in analyses if a['class'] == c)
        total_missed_dollar = sum(a['missed_r'] * a['risk_ps'] * a['shares'] for a in analyses if a['class'] == c)
        print(f"  {c:<36}: {n:>2} ({pct:.0f}%)  missed_upside_R={total_missed_r:+.1f}R  dollars=${total_missed_dollar:+,.0f}")

    # Summary by exit reason
    reason_counts = Counter(a['exit_reason'] for a in analyses)
    print()
    print(f"  Exit reason distribution:")
    for r, n in reason_counts.most_common():
        avg_held = sum(a['held_r'] for a in analyses if a['exit_reason']==r) / n
        avg_missed = sum(a['missed_r'] for a in analyses if a['exit_reason']==r) / n
        print(f"    {r:<22}: n={n:>2}  avg held_R={avg_held:+.2f}  avg missed_R={avg_missed:+.2f}")

    # Average missed upside
    print()
    avg_missed_r = sum(a['missed_r'] for a in analyses) / len(analyses)
    total_missed_dollar = sum(a['missed_r'] * a['risk_ps'] * a['shares'] for a in analyses)
    total_pnl = sum(a['pnl'] for a in analyses)
    print(f"  Total P&L (actual): ${total_pnl:+,.0f}")
    print(f"  Total missed upside if perfect exit at peak: ${total_missed_dollar:+,.0f}")
    print(f"  Theoretical max P&L: ${total_pnl + total_missed_dollar:+,.0f}")
    print(f"  Avg missed_R per trade: {avg_missed_r:+.2f}R")


if __name__ == '__main__':
    main()
