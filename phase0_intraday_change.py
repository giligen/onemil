#!/usr/bin/env python3
"""Phase 0: compute intraday_change_at_entry for every O_f6 trade.

Mirrors scanner/realtime_scanner.py:614 logic:
  intraday_change_pct = max(gap_pct, range_pct)
where range_pct = (day_high_so_far - day_low_so_far) / day_low_so_far * 100.

For each trade row we replay 1-min bars up to entry_time_et and track the
max qualification value. a_eligible == 1 means the stock would have passed
the 20% gate at some point before entry (A_f6 would see it).

prev_close source: daily_bars lookup; fallback to first-bar-open / (1 + gap%).

Emits augmented CSVs with:
  - max_intraday_change_pre_entry
  - a_eligible                     (>= 20.0 -> 1 else 0)

Read-only over data/cache.db.
"""
from __future__ import annotations
import csv
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


IN_2025 = Path('/tmp/variant_runner/bt_O_f6_2025-01-01_2025-12-31.csv')
IN_Q1   = Path('/tmp/variant_runner/bt_O_f6_q1.csv')
OUT_2025 = Path('/tmp/variant_runner/bt_O_f6_2025_with_intraday.csv')
OUT_Q1   = Path('/tmp/variant_runner/bt_O_f6_q1_with_intraday.csv')

DB_PATH = 'data/cache.db'


def et_to_utc(date_str: str, time_et: str) -> str:
    """Convert ET naive date+time to UTC ISO-ish string that sorts lexically
    against stored timestamps like '2025-01-02T14:30:00+00:00'.

    ET = UTC-5 (standard) or UTC-4 (DST). Use approximation: second Sunday
    of March through first Sunday of November is DST.
    """
    y, m, d = map(int, date_str.split('-'))
    hh, mm, ss = map(int, time_et.split(':'))
    # DST boundaries
    dst_start = _second_sunday(y, 3)   # March second Sunday, 02:00 local
    dst_end   = _first_sunday(y, 11)   # November first Sunday, 02:00 local
    in_dst = (m > dst_start[1] or (m == dst_start[1] and d >= dst_start[2])) and \
             (m < dst_end[1] or (m == dst_end[1] and d < dst_end[2]))
    offset = 4 if in_dst else 5
    # UTC = ET + offset hours
    utc_h = hh + offset
    extra_days = utc_h // 24
    utc_h = utc_h % 24
    # Build date
    base = datetime(y, m, d, tzinfo=timezone.utc) + timedelta(days=extra_days)
    return f"{base.strftime('%Y-%m-%d')}T{utc_h:02d}:{mm:02d}:{ss:02d}+00:00"


def _second_sunday(y: int, m: int) -> tuple[int, int, int]:
    first = datetime(y, m, 1)
    # weekday() Monday=0 Sunday=6
    days_to_sunday = (6 - first.weekday()) % 7
    first_sunday = 1 + days_to_sunday
    return (y, m, first_sunday + 7)


def _first_sunday(y: int, m: int) -> tuple[int, int, int]:
    first = datetime(y, m, 1)
    days_to_sunday = (6 - first.weekday()) % 7
    return (y, m, 1 + days_to_sunday)


def scanner_qual_pre_entry(conn, symbol: str, date_str: str, entry_utc: str,
                           prev_close_val: float | None) -> tuple[float, float, float] | None:
    """Replay bars up to entry. Return (max_qual_pct, max_high, first_open)
    or None if no bars.

    Mirrors scanner logic:
        gap_pct   = (close - prev_close) / prev_close * 100
        range_pct = (day_high - day_low) / day_low * 100
        qual      = max(gap_pct, range_pct)   # running max over pre-entry bars
    """
    cur = conn.execute(
        "SELECT open, high, low, close FROM intraday_bars_1min "
        "WHERE symbol=? AND bar_date=? AND timestamp < ? "
        "ORDER BY timestamp",
        (symbol, date_str, entry_utc)
    )
    rows = cur.fetchall()
    if not rows:
        return None
    first_open = rows[0][0]
    day_high = rows[0][1]
    day_low = rows[0][2]
    max_qual = float('-inf')
    max_h_overall = rows[0][1]
    for _o, h, l, c in rows:
        if h > day_high:
            day_high = h
        if l < day_low and l > 0:
            day_low = l
        if h > max_h_overall:
            max_h_overall = h
        range_pct = ((day_high - day_low) / day_low * 100) if day_low > 0 else 0.0
        if prev_close_val and prev_close_val > 0:
            gap_pct = (c - prev_close_val) / prev_close_val * 100
        else:
            gap_pct = float('-inf')
        qual = max(gap_pct, range_pct)
        if qual > max_qual:
            max_qual = qual
    return max_qual, max_h_overall, first_open


def prev_close(conn, symbol: str, date_str: str, first_open: float, gap_pct_str: str) -> float | None:
    """Prefer daily_bars lookup; fall back to gap_pct inverse math."""
    row = conn.execute(
        "SELECT close FROM daily_bars WHERE symbol=? AND bar_date<? ORDER BY bar_date DESC LIMIT 1",
        (symbol, date_str)
    ).fetchone()
    if row and row[0] and row[0] > 0:
        return float(row[0])
    if gap_pct_str:
        try:
            gap = float(gap_pct_str)
            return first_open / (1 + gap / 100.0)
        except (ValueError, ZeroDivisionError):
            pass
    return None


def enrich(in_path: Path, out_path: Path, conn) -> dict:
    rows = list(csv.DictReader(open(in_path)))
    out_cols = list(rows[0].keys()) + [
        'max_high_pre_entry', 'prev_close',
        'max_intraday_change_pre_entry', 'a_eligible',
    ]
    a_eligible_count = 0
    extra_count = 0
    below_10 = 0
    skipped = 0
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        for r in rows:
            symbol = r['symbol']
            date_str = r['date']
            entry_utc = et_to_utc(date_str, r['entry_time_et'])
            # Need prev_close FIRST (required for gap_pct part of qual)
            # We don't have first_open yet — do a cheap query for it.
            cur = conn.execute(
                "SELECT open FROM intraday_bars_1min WHERE symbol=? AND bar_date=? "
                "ORDER BY timestamp LIMIT 1", (symbol, date_str))
            first_row = cur.fetchone()
            first_open_val = float(first_row[0]) if first_row else None
            pc = prev_close(conn, symbol, date_str, first_open_val or 0, r.get('qf_gap_pct', '')) if first_open_val else None

            res = scanner_qual_pre_entry(conn, symbol, date_str, entry_utc, pc)
            if not res:
                r.update({'max_high_pre_entry': '', 'prev_close': '',
                          'max_intraday_change_pre_entry': '', 'a_eligible': ''})
                skipped += 1
                w.writerow(r)
                continue
            max_qual, max_h, _ = res
            if max_qual == float('-inf') or max_qual is None:
                r.update({'max_high_pre_entry': f'{max_h:.4f}', 'prev_close': f'{pc:.4f}' if pc else '',
                          'max_intraday_change_pre_entry': '', 'a_eligible': ''})
                skipped += 1
                w.writerow(r)
                continue
            a_elig = 1 if max_qual >= 20.0 else 0
            if a_elig:
                a_eligible_count += 1
            elif max_qual >= 10.0:
                extra_count += 1
            else:
                below_10 += 1
            r.update({
                'max_high_pre_entry': f'{max_h:.4f}',
                'prev_close': f'{pc:.4f}' if pc else '',
                'max_intraday_change_pre_entry': f'{max_qual:.2f}',
                'a_eligible': str(a_elig),
            })
            w.writerow(r)
    return {
        'total': len(rows),
        'a_eligible': a_eligible_count,
        'extras_10_20': extra_count,
        'below_10': below_10,
        'skipped': skipped,
    }


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--in-2025', type=str, default=str(IN_2025))
    p.add_argument('--in-q1',   type=str, default=str(IN_Q1))
    p.add_argument('--out-2025', type=str, default=str(OUT_2025))
    p.add_argument('--out-q1',   type=str, default=str(OUT_Q1))
    args = p.parse_args()
    in_2025 = Path(args.in_2025)
    in_q1 = Path(args.in_q1)
    out_2025 = Path(args.out_2025)
    out_q1 = Path(args.out_q1)
    if not in_2025.exists() or not in_q1.exists():
        print(f"missing input: {in_2025} or {in_q1}", file=sys.stderr)
        return 2
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA query_only=1")
    stats_2025 = enrich(in_2025, out_2025, conn)
    stats_q1 = enrich(in_q1, out_q1, conn)
    print("=" * 70)
    print(f"{'set':<10} {'total':>6} {'a_elig':>8} {'extras':>8} {'<10':>5} {'skip':>5}")
    print("-" * 70)
    print(f"{'2025':<10} {stats_2025['total']:>6} {stats_2025['a_eligible']:>8} "
          f"{stats_2025['extras_10_20']:>8} {stats_2025['below_10']:>5} {stats_2025['skipped']:>5}")
    print(f"{'Q1 2026':<10} {stats_q1['total']:>6} {stats_q1['a_eligible']:>8} "
          f"{stats_q1['extras_10_20']:>8} {stats_q1['below_10']:>5} {stats_q1['skipped']:>5}")
    print()
    print(f"Expected a_eligible on 2025: ~83 (A_f6 count).")
    print(f"Actual: {stats_2025['a_eligible']}")
    print(f"Delta: {stats_2025['a_eligible'] - 83:+d}")
    print()
    print(f"Output:\n  {out_2025}\n  {out_q1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
