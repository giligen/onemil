"""Backfill the 1-min intraday bar cache.

The MACD wave BT (and any analytics computing MACD on cached bars) reads
from `intraday_bars_1min`. Pre-fix the cache could contain PARTIAL coverage
for a (symbol, date) — e.g., 138 of 390 RTH minutes — because earlier
fetches were truncated. The BT then computed MACD on this truncated series,
producing different cross signals than live (which sees the real-time
stream).

This module:
  1. Audits the cache for (symbol, date) pairs with incomplete coverage.
  2. Refetches the full RTH window from Alpaca for pairs below threshold.
  3. Writes corrected bars back via INSERT OR REPLACE.

Designed to be (a) invokable manually for one-off repairs and (b) plumbed
into the nightly universe-rebuild job for routine maintenance — same shape
as the daily_bars refresh step that ships in batch/universe_builder.py
(commit ee5a0d9 / fd0446e for context).

Usage:
    # Backfill the live-traded MACD wave universe for a window
    python batch/intraday_bars_backfill.py --start 2026-05-08 --end 2026-05-22

    # Audit only — no API calls
    python batch/intraday_bars_backfill.py --start 2026-05-08 --end 2026-05-22 --dry-run

    # Broader universe (all 10%+ movers in daily_bars)
    python batch/intraday_bars_backfill.py --start 2026-05-08 --end 2026-05-22 \\
        --source all_movers
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Tuple

import pytz

logger = logging.getLogger(__name__)
ET = pytz.timezone('US/Eastern')


# Regular trading hours = 09:30-16:00 ET = 390 minutes. Actively-traded
# stocks usually have a bar in essentially every minute; thinly-traded
# stocks legitimately fewer. Use ~90% of RTH as the "complete" threshold
# above which we don't refetch. (Below: refetch and let Alpaca tell us
# what's actually available.)
DEFAULT_COMPLETENESS_THRESHOLD = 350


def audit_coverage(
    db, symbol_dates: List[Tuple[str, str]]
) -> List[Tuple[str, str, int]]:
    """Audit cache coverage for the given (symbol, date) pairs.

    Returns:
        List of (symbol, date_str, current_bar_count) sorted by ascending
        bar count. Pairs not in cache report 0.
    """
    if not symbol_dates:
        return []
    bulk: Dict[Tuple[str, str], list] = db.get_intraday_bars_bulk(symbol_dates)
    out = []
    for sd in symbol_dates:
        bars = bulk.get(sd, []) or []
        out.append((sd[0], sd[1], len(bars)))
    return sorted(out, key=lambda x: x[2])


def fetch_and_save(
    alpaca_client, db, symbol: str, date_str: str,
) -> Tuple[int, int]:
    """Fetch RTH 1-min bars for (symbol, date) from Alpaca; save to cache.

    Returns:
        (bars_fetched, bars_saved). Both 0 on parse failure or API error;
        in particular, no exception escapes.
    """
    try:
        td = date.fromisoformat(date_str)
    except (ValueError, TypeError):
        logger.warning(f"backfill: bad date_str {date_str!r}")
        return 0, 0

    # RTH 09:30-16:00 ET → UTC. ET.localize handles DST.
    mo = ET.localize(datetime(td.year, td.month, td.day, 9, 30)
                      ).astimezone(timezone.utc)
    mc = ET.localize(datetime(td.year, td.month, td.day, 16, 0)
                      ).astimezone(timezone.utc)

    try:
        bars = alpaca_client.get_historical_1min_bars(symbol, mo, mc)
    except Exception as e:
        logger.warning(f"backfill: fetch failed {symbol} {date_str}: {e}")
        return 0, 0

    if bars is None or len(bars) == 0:
        return 0, 0

    try:
        recs = bars.to_dict('records')
        saved = db.save_intraday_bars(symbol, date_str, recs)
        return len(recs), int(saved or 0)
    except Exception as e:
        logger.warning(f"backfill: save failed {symbol} {date_str}: {e}")
        return 0, 0


def backfill(
    db, alpaca_client, symbol_dates: Iterable[Tuple[str, str]],
    threshold: int = DEFAULT_COMPLETENESS_THRESHOLD,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Audit + backfill 1-min bar coverage for (symbol, date) pairs.

    For each pair with fewer than `threshold` bars cached, refetches from
    Alpaca and INSERT-OR-REPLACEs. Pairs at or above the threshold are
    skipped without an API call.

    Returns a summary dict: audited / incomplete / refetched / bars_added.

    Never raises — single-pair failures are logged and the rest proceed.
    """
    pairs = list(symbol_dates)
    audit = audit_coverage(db, pairs)
    incomplete = [(s, d, n) for (s, d, n) in audit if n < threshold]

    logger.info(
        f"backfill: audited {len(audit)} (symbol, date) pairs — "
        f"{len(incomplete)} incomplete (<{threshold} bars)"
    )

    if dry_run or not incomplete:
        return {
            'audited': len(audit),
            'incomplete': len(incomplete),
            'refetched': 0,
            'bars_added': 0,
        }

    refetched = 0
    bars_added = 0
    for s, d, current in incomplete:
        fetched, _saved = fetch_and_save(alpaca_client, db, s, d)
        if fetched > current:
            refetched += 1
            bars_added += (fetched - current)
            if refetched % 25 == 0:
                logger.info(
                    f"  ...refetched {refetched}/{len(incomplete)} pairs"
                )

    logger.info(
        f"backfill complete: refetched {refetched}/{len(incomplete)} "
        f"incomplete pairs (+{bars_added} bars added)"
    )
    return {
        'audited': len(audit),
        'incomplete': len(incomplete),
        'refetched': refetched,
        'bars_added': bars_added,
    }


def universe_for_date_range(
    db, start_date: date, end_date: date, source: str = 'macd_wave',
) -> List[Tuple[str, str]]:
    """Return (symbol, date) pairs to audit for the chosen universe source.

    source='macd_wave':  symbols live MACD wave actually traded in range.
    source='all_movers': symbols with >=10% intraday range in daily_bars.

    Other sources can be added later for other strategies' BT needs.
    """
    # `get_intraday_bars_bulk` keys its returned dict on STRING dates
    # (it stringifies the SQL date column). Coerce trade_date / bar_date
    # to ISO string here so audit lookups match exactly. Pre-fix the
    # query returned `datetime.date` and lookups silently missed.
    def _to_str(v):
        if hasattr(v, 'isoformat'):
            return v.isoformat()
        return str(v)

    if source == 'macd_wave':
        cur = db._trades_conn.execute("""
            SELECT DISTINCT symbol, trade_date FROM trades
            WHERE strategy='macd_wave'
              AND trade_date >= ? AND trade_date <= ?
              AND order_status IN ('closed','filled','stale_closed')
        """, (start_date.isoformat(), end_date.isoformat()))
        return [(r[0], _to_str(r[1])) for r in cur.fetchall()]
    elif source == 'all_movers':
        cur = db._cache_conn.execute("""
            SELECT symbol, bar_date FROM daily_bars
            WHERE bar_date >= ? AND bar_date <= ?
              AND high > 0 AND low > 0
              AND (high - low) / low * 100 >= 10.0
        """, (start_date.isoformat(), end_date.isoformat()))
        return [(r[0], _to_str(r[1])) for r in cur.fetchall()]
    else:
        raise ValueError(f"unknown source: {source!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit + backfill the intraday_bars_1min cache."
    )
    parser.add_argument("--start", type=str, required=True,
                        help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", type=str, required=True,
                        help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--source", choices=['macd_wave', 'all_movers'],
                        default='macd_wave')
    parser.add_argument("--threshold", type=int,
                        default=DEFAULT_COMPLETENESS_THRESHOLD,
                        help=f"Bars/day threshold (default {DEFAULT_COMPLETENESS_THRESHOLD})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Audit only — no API calls")
    args = parser.parse_args()

    from monitoring.logger import setup_logging
    setup_logging(verbose=True)

    from persistence.database import get_database
    from data_sources.alpaca_client import AlpacaClient
    from config import get_config

    cfg = get_config()
    db = get_database(
        db_path=cfg.db_path, cache_path=cfg.cache_db_path,
        trades_path=cfg.trades_db_path,
    )
    client = AlpacaClient(
        cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper,
    )

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    pairs = universe_for_date_range(db, start, end, source=args.source)
    logger.info(
        f"Universe ({args.source}): {len(pairs)} (symbol, date) pairs"
    )

    result = backfill(
        db, client, pairs, threshold=args.threshold, dry_run=args.dry_run,
    )
    logger.info(f"Summary: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
