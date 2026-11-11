"""ORB BT cache: opening-range bar-gap gate + idempotent refill (spec docs/orb_cache_bar_gaps_spec_20261111.md).

Measured 2026-11-11 (research/book_review_20261111/cache_gaps_1111.csv): 106/1159 candidate symbol-days (9 %) had < 5
one-minute bars in [09:30, 09:35) ET, but SIP returns the IDENTICAL minutes for 20/20 spot-checks -- the missing
minutes are minutes with no trade (thin names), not vendor gaps. So "< 5 bars" alone is NOT a defect; a defect is
"SIP has a minute the cache lacks". The gate therefore re-asks SIP for every gapped symbol-day and alarms on the
share that was REALLY missing.

CLI (never run against production by a task; the session runs it):
    python3 batch/orb_bar_gap_gate.py --refill-gaps --start 2026-09-29 --end 2026-11-10
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import pytz

logger = logging.getLogger(__name__)
ET = pytz.timezone('US/Eastern')
RANGE_MINUTES = 5
ALARM_SHARE = 0.10


def window_utc(day: str, minutes: int = RANGE_MINUTES) -> Tuple[datetime, datetime]:
    """[09:30, 09:30+minutes) ET of `day` as UTC datetimes (DST-correct)."""
    y, m, d = map(int, day.split('-'))
    lo = ET.localize(datetime(y, m, d, 9, 30)).astimezone(timezone.utc)
    return lo, lo + timedelta(minutes=minutes)


def gapped_pairs(conn, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str, int]]:
    """(symbol, day, n_bars) for pairs with < RANGE_MINUTES cached bars in the opening window (0 = empty)."""
    from study_orb_broad import opening_range_bar_counts_for_pairs
    counts = opening_range_bar_counts_for_pairs(conn, pairs, RANGE_MINUTES)
    return [(s, d, counts.get((s, d), 0)) for s, d in pairs if counts.get((s, d), 0) < RANGE_MINUTES]


def refill_pairs(conn, alpaca, gapped: List[Tuple[str, str, int]]) -> Dict[str, object]:
    """Re-fetch ONLY the opening window of each gapped pair from SIP; INSERT OR IGNORE the minutes the cache lacks.
    Returns {gapped, refilled_pairs, bars_added, refilled_names, failed}; idempotent (a re-run adds 0)."""
    added, refilled, failed = 0, [], 0
    for sym, day, _ in gapped:
        lo, hi = window_utc(day)
        try:
            df = alpaca.get_historical_1min_bars(sym, lo, hi - timedelta(seconds=1))
        except Exception as e:
            failed += 1
            logger.warning(f"orb bar-gap refill: SIP fetch failed {sym} {day}: {type(e).__name__}: {e}")
            continue
        if df is None or len(df) == 0:
            continue
        rows = [(sym, day, t.strftime('%Y-%m-%dT%H:%M:%S+00:00'), float(r.open), float(r.high), float(r.low),
                 float(r.close), int(r.volume))
                for r in df.itertuples() for t in [__import__('pandas').to_datetime(r.timestamp, utc=True)]]
        before = conn.total_changes
        conn.executemany("INSERT OR IGNORE INTO intraday_bars_1min (symbol, bar_date, timestamp, open, high, low, "
                         "close, volume) VALUES (?,?,?,?,?,?,?,?)", rows)
        conn.commit()
        if conn.total_changes > before:
            added += conn.total_changes - before
            refilled.append(f"{sym}@{day}")
    return {'gapped': len(gapped), 'refilled_pairs': len(refilled), 'bars_added': added,
            'refilled_names': refilled, 'failed': failed}


def run_gate(conn, alpaca, pairs: List[Tuple[str, str]], latest_day: str = None) -> Dict[str, object]:
    """Completeness gate + refill. WARNING with count and first 20 gapped names (every run with any gap); ERROR when
    > ALARM_SHARE of `latest_day`'s pairs were REAL vendor gaps (SIP had minutes the cache lacked)."""
    gapped = gapped_pairs(conn, pairs)
    res = refill_pairs(conn, alpaca, gapped) if gapped else {'gapped': 0, 'refilled_pairs': 0, 'bars_added': 0,
                                                              'refilled_names': [], 'failed': 0}
    if gapped:
        logger.warning(f"orb bar-gap gate: {len(gapped)}/{len(pairs)} symbol-days < {RANGE_MINUTES} opening bars; "
                       f"{res['refilled_pairs']} refilled from SIP ({res['bars_added']} bars), {res['failed']} fetch "
                       f"failures, rest = no-trade minutes. First 20: {[f'{s}@{d}' for s, d, _ in gapped[:20]]}")
    day = latest_day or (max(d for _, d in pairs) if pairs else None)
    n_day = sum(1 for _, d in pairs if d == day)
    real = sum(1 for x in res['refilled_names'] if x.endswith('@' + str(day)))
    if n_day and real / n_day > ALARM_SHARE:
        logger.error(f"orb bar-gap gate: VENDOR DEGRADATION {day}: {real}/{n_day} symbol-days were missing opening "
                     f"bars SIP now has (> {ALARM_SHARE:.0%}); refilled")
    res['real_gap_share_latest'] = (real / n_day) if n_day else 0.0
    return res


def main() -> None:
    """--refill-gaps --start D1 --end D2: gate+refill every candidate symbol-day of the window in data/cache.db."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--refill-gaps', action='store_true', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--db', default='data/cache.db')
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dotenv import load_dotenv
    load_dotenv('/home/ec2-user/onemil/.env')
    from data_sources.alpaca_client import AlpacaClient
    import orb_backtest as ob
    alpaca = AlpacaClient(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_API_SECRET'], paper=True)
    d0, d1 = (datetime.strptime(x, '%Y-%m-%d').date() for x in (a.start, a.end))
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1) if (d0 + timedelta(days=i)).weekday() < 5]
    by_date = ob._qualifying_pairs_for_dates(a.db, days)
    pairs = [(s, d) for d, ss in by_date.items() for s in ss]
    conn = sqlite3.connect(a.db, timeout=60)
    print(run_gate(conn, alpaca, pairs, max(by_date) if by_date else None))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
