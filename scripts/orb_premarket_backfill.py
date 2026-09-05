#!/usr/bin/env python3
"""ORB premarket 1-min bar backfill (2026-09-05) — bulk job #1 of the ORB
honest-reference rebuild.

WHY
  Live's ORB universe gate is "volume traded by 9:35 ET INCLUDING premarket
  from 4:00" (Alpaca snapshot). The BT gates on PREVIOUS-day volume because
  the 1-min cache holds premarket bars only sporadically (2025-03-12: 626
  premarket rows of 138,862). That gap is the PFSA-class red (BT picks a
  prev-liquid / quiet-open name live's 9:35 gate excludes). Once every ORB
  candidate (symbol, date) has its 4:00-9:35 ET bars cached, BOTH sides can
  run the same 500K@9:35 gate (study_orb_broad.py) and the entered-inclusive
  features rebuild can produce the corrected reference for the 9/12 gate.

WHAT
  For every (date, symbol) in the ORB broad universe (prev-day gate,
  `study_orb_broad.load_broad_universe`, or a `--candidates CSV` snapshot of
  it), fetch 04:00-09:35 ET 1-min bars in batches of `--batch` symbols per
  API call and store them with `Database.save_intraday_bars` (INSERT OR
  REPLACE: the 9:30-9:35 rows already cached are rewritten identically).
  Each fetched pair is recorded in `orb_premarket_backfill_done` with its
  bar count so the job is RESUMABLE and "0 premarket prints" is a real
  observation, not a missing fetch.

  Never deletes or overwrites RTH bars. Never touches daily_bars.
  A failed batch is retried `--retries` times then SKIPPED (not marked) —
  the summary counts skipped pairs and exits non-zero if any remain.

USAGE
  python scripts/orb_premarket_backfill.py --start 2025-01-01 --end 2026-09-04 \
      [--candidates data/research/orb_broad_candidates.csv] [--batch 50] \
      [--limit N] [--dry-run] [--verbose]
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Sequence, Tuple
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

logger = logging.getLogger('orb_premarket_backfill')

ET = ZoneInfo('America/New_York')
PM_START_ET = (4, 0)
PM_END_ET = (9, 35)          # inclusive of the 9:34 bar; 9:35 bar excluded
DEFAULT_BATCH = 50
DEFAULT_RETRIES = 2
DEFAULT_RETRY_SLEEP_S = 5.0


def premarket_window_utc(bar_date: date) -> Tuple[datetime, datetime]:
    """(start, end) UTC datetimes for 04:00 ET .. 09:35 ET on `bar_date`.

    Computed per date in ET so DST is right (a hardcoded UTC hour is the
    bug class behind 3fab1f9/35e9935). `end` is exclusive at the API: the
    9:35 bar (start 09:35:00) is NOT included — live's 9:35 snapshot volume
    counts prints BEFORE 9:35, matching the 09:34 bar as the last one.
    """
    s = datetime(bar_date.year, bar_date.month, bar_date.day, *PM_START_ET, tzinfo=ET)
    e = datetime(bar_date.year, bar_date.month, bar_date.day, *PM_END_ET, tzinfo=ET)
    # Alpaca's `end` is inclusive of a bar starting exactly at `end`; step
    # back one second so the 09:35 bar is excluded.
    e = e - timedelta(seconds=1)
    return s.astimezone(timezone.utc), e.astimezone(timezone.utc)


def load_candidates_csv(path: str, start: date, end: date) -> Dict[str, List[str]]:
    """{bar_date: [symbols]} from a CSV with columns bar_date, symbol."""
    out: Dict[str, List[str]] = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row['bar_date']
            if d < start.isoformat() or d > end.isoformat():
                continue
            out[d].append(row['symbol'])
    return dict(out)


def load_candidates_db(start: date, end: date) -> Dict[str, List[str]]:
    """{bar_date: [symbols]} straight from the broad-universe query."""
    import study_orb_broad as sob
    sob.DATE_START = start.isoformat()
    sob.DATE_END = end.isoformat()
    return sob.load_broad_universe()


def batches(items: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield list(items[i:i + size])


def bars_to_records(df) -> List[Dict]:
    """DataFrame → list of dicts in the Database.save_intraday_bars shape."""
    if df is None or len(df) == 0:
        return []
    return [{
        'timestamp': r['timestamp'], 'open': float(r['open']), 'high': float(r['high']),
        'low': float(r['low']), 'close': float(r['close']), 'volume': int(r['volume']),
    } for _, r in df.iterrows()]


class PremarketBackfill:
    """Fetch + store premarket bars for candidate pairs; resumable."""

    def __init__(self, alpaca, db, batch: int = DEFAULT_BATCH, retries: int = DEFAULT_RETRIES,
                 retry_sleep_s: float = DEFAULT_RETRY_SLEEP_S, dry_run: bool = False):
        self.alpaca = alpaca
        self.db = db
        self.batch = max(1, int(batch))
        self.retries = max(0, int(retries))
        self.retry_sleep_s = retry_sleep_s
        self.dry_run = dry_run
        self.stats = {'dates': 0, 'pairs': 0, 'already_done': 0, 'fetched': 0,
                      'with_prints': 0, 'bars_saved': 0, 'skipped_failed': 0,
                      'api_calls': 0}

    def run_date(self, bar_date: str, symbols: Sequence[str]) -> None:
        d = date.fromisoformat(bar_date)
        start_utc, end_utc = premarket_window_utc(d)
        done = set() if self.dry_run else self.db.get_premarket_backfilled_symbols(bar_date)
        todo = [s for s in symbols if s not in done]
        self.stats['dates'] += 1
        self.stats['pairs'] += len(symbols)
        self.stats['already_done'] += len(symbols) - len(todo)
        if not todo:
            logger.info(f"{bar_date}: {len(symbols)} candidates, all already backfilled")
            return
        for chunk in batches(todo, self.batch):
            if self.dry_run:
                logger.info(f"{bar_date}: DRY-RUN would fetch {len(chunk)} symbols "
                            f"{start_utc.isoformat()}..{end_utc.isoformat()}")
                continue
            bars_map = self._fetch_with_retries(chunk, start_utc, end_utc, bar_date)
            if bars_map is None:
                self.stats['skipped_failed'] += len(chunk)
                continue
            for sym in chunk:
                recs = bars_to_records(bars_map.get(sym))
                n = self.db.save_intraday_bars(sym, bar_date, recs) if recs else 0
                self.db.mark_premarket_backfilled(sym, bar_date, n)
                self.stats['fetched'] += 1
                self.stats['bars_saved'] += n
                if n:
                    self.stats['with_prints'] += 1
        logger.info(
            f"{bar_date}: {len(todo)} fetched ({len(symbols) - len(todo)} were done) — "
            f"cumulative fetched={self.stats['fetched']} with_prints={self.stats['with_prints']} "
            f"bars={self.stats['bars_saved']:,} failed={self.stats['skipped_failed']} "
            f"api_calls={self.stats['api_calls']}")

    def _fetch_with_retries(self, chunk, start_utc, end_utc, bar_date):
        for attempt in range(self.retries + 1):
            try:
                self.stats['api_calls'] += 1
                return self.alpaca.get_1min_bars_range_multi(chunk, start_utc, end_utc)
            except Exception as e:  # AlpacaAPIError or transport
                logger.warning(f"{bar_date}: batch of {len(chunk)} failed "
                               f"(attempt {attempt + 1}/{self.retries + 1}): {e}")
                if attempt < self.retries:
                    time.sleep(self.retry_sleep_s)
        logger.error(f"{bar_date}: batch of {len(chunk)} SKIPPED after retries — "
                     f"pairs stay unmarked; rerun to resume")
        return None

    def run(self, candidates: Dict[str, List[str]], limit: int = 0) -> Dict:
        dates = sorted(candidates)
        if limit:
            dates = dates[:limit]
        total_pairs = sum(len(candidates[d]) for d in dates)
        logger.info(f"Backfill: {len(dates)} dates, {total_pairs} pairs, batch={self.batch}, "
                    f"dry_run={self.dry_run}")
        t0 = time.time()
        for i, d in enumerate(dates, 1):
            self.run_date(d, candidates[d])
            if i % 20 == 0:
                el = time.time() - t0
                logger.info(f"progress {i}/{len(dates)} dates, {el / 60:.1f} min elapsed, "
                            f"eta {el / i * (len(dates) - i) / 60:.1f} min")
        self.stats['elapsed_s'] = round(time.time() - t0, 1)
        logger.info(f"DONE {self.stats}")
        return self.stats


def build_alpaca():
    from data_sources.alpaca_client import AlpacaClient
    key = os.getenv('ALPACA_API_KEY')
    sec = os.getenv('ALPACA_API_SECRET')
    if not key or not sec:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET missing — refusing to run")
    return AlpacaClient(key, sec, paper=os.getenv('ALPACA_PAPER', 'true').lower() == 'true')


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--start', default='2025-01-01')
    p.add_argument('--end', default=date.today().isoformat())
    p.add_argument('--candidates', default=None,
                   help='CSV (bar_date,symbol) snapshot of the broad universe; default: query cache.db')
    p.add_argument('--batch', type=int, default=DEFAULT_BATCH)
    p.add_argument('--retries', type=int, default=DEFAULT_RETRIES)
    p.add_argument('--limit', type=int, default=0, help='only the first N dates (smoke test)')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    load_dotenv()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if args.candidates:
        cands = load_candidates_csv(args.candidates, start, end)
        logger.info(f"Candidates from {args.candidates}: {len(cands)} dates")
    else:
        logger.info("Candidates from cache.db broad-universe query (slow)…")
        cands = load_candidates_db(start, end)

    from persistence.database import get_database
    db = get_database()
    db.ensure_premarket_backfill_table()
    alpaca = None if args.dry_run else build_alpaca()
    job = PremarketBackfill(alpaca, db, batch=args.batch, retries=args.retries, dry_run=args.dry_run)
    stats = job.run(cands, limit=args.limit)
    return 1 if stats['skipped_failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
