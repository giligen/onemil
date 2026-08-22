#!/usr/bin/env python3
"""P0-6 daily_bars gap backfill for the ORB winner stack (2026-08-22).

The SZ1 ATR stop-floor needs >= 15 daily bars of history strictly before a
trade date (trading/orb_winner_stack.atr14_t1). The validated B+ book left 6
trades UNFLOORED purely because cache.db daily_bars had gaps (SMST 2025-01-07
had 5 cached prior bars; SMST is a 2x-MSTR wrapper listed Aug'24 with ample
real history). Live's 40-day Alpaca fetch WOULD find the bars and floor them
— an unvalidated BT<->LIVE divergence on ~7% of trades. Fix: backfill the
cache so BT and live see the same history, then REGENERATE the reference
book (design §1b P0-6 — gate 2's target is the regenerated book).

STRICTLY INSERT-ONLY (CLAUDE.md cache rule): rows go in via
`INSERT OR IGNORE` — an existing (symbol, bar_date) row is NEVER touched.
Rowcount snapshots are printed before/after.

Scope: every (symbol, date) in the candidate universe (the features CSV the
pipeline resims — env ORB_BT_FEATURES_CSV or latest non-corrmatrix glob),
plus the current book CSV, with fewer than 15 cached daily bars strictly
before the trade date. The six known book gaps (SMST, ARQQ, BTQ, RGTZ,
FJET, PS) are always included.

Usage:
    python3 scripts/backfill_daily_bars_gaps.py [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE_DB = ROOT / 'data' / 'cache.db'
MIN_BARS = 15                     # frozen atr14_t1 availability rule
FETCH_PAD_DAYS = 60               # calendar days before earliest gap date
# The six book trades unfloored only by missing cache data (review P0-6.2).
MANDATORY_SYMBOLS = ['SMST', 'ARQQ', 'BTQ', 'RGTZ', 'FJET', 'PS']


def features_csv_path() -> str:
    """Same resolution rule as the BT pipeline (env else latest glob)."""
    csv = os.environ.get('ORB_BT_FEATURES_CSV')
    if csv:
        return csv
    paths = sorted(p for p in glob.glob(
        str(ROOT / 'analysis_results' / 'orb_features_*.csv'))
        if 'corrmatrix' not in p)
    if not paths:
        raise SystemExit("FATAL: no orb_features_*.csv found")
    return paths[-1]


def candidate_pairs() -> list:
    """(symbol, 'YYYY-MM-DD') pairs from the features CSV + current book."""
    pairs = set()
    feats = pd.read_csv(features_csv_path(), usecols=['symbol', 'date'])
    for _, r in feats.drop_duplicates().iterrows():
        pairs.add((str(r['symbol']), str(r['date'])[:10]))
    book = ROOT / 'analysis_results' / 'orb_bplus_book.csv'
    if book.exists():
        b = pd.read_csv(book, usecols=['symbol', 'date'])
        for _, r in b.iterrows():
            pairs.add((str(r['symbol']), str(r['date'])[:10]))
    return sorted(pairs)


def prior_bar_counts(con: sqlite3.Connection, pairs: list) -> dict:
    """{(symbol, date) -> count of cached daily bars strictly before date}."""
    out = {}
    by_sym = {}
    for sym, day in pairs:
        by_sym.setdefault(sym, []).append(day)
    for sym, days in by_sym.items():
        rows = con.execute(
            "SELECT bar_date FROM daily_bars WHERE symbol=? ORDER BY bar_date",
            (sym,)).fetchall()
        dates = [str(r[0]) for r in rows]
        import bisect
        for day in days:
            out[(sym, day)] = bisect.bisect_left(dates, day)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='scan + report gaps, no fetch, no insert')
    args = ap.parse_args()

    con = sqlite3.connect(CACHE_DB, timeout=30)
    total_before = con.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0]
    print(f"SNAPSHOT before: daily_bars total rows = {total_before:,}")

    pairs = candidate_pairs()
    print(f"Candidate universe: {len(pairs)} symbol-days "
          f"({features_csv_path()})")
    counts = prior_bar_counts(con, pairs)
    gaps = {k: v for k, v in counts.items() if v < MIN_BARS}
    gap_syms = sorted({s for s, _ in gaps})
    for s in MANDATORY_SYMBOLS:
        if s not in gap_syms:
            print(f"NOTE: mandatory symbol {s} no longer shows a gap "
                  f"(already backfilled?)")
    print(f"Gap scan: {len(gaps)} symbol-days across {len(gap_syms)} symbols "
          f"with < {MIN_BARS} prior cached daily bars")
    per_sym_before = {
        s: con.execute("SELECT COUNT(*) FROM daily_bars WHERE symbol=?",
                       (s,)).fetchone()[0]
        for s in sorted(set(gap_syms + MANDATORY_SYMBOLS))}
    print("SNAPSHOT before (per gap symbol): "
          + ', '.join(f"{s}={n}" for s, n in sorted(per_sym_before.items())))

    if args.dry_run:
        for (s, d), n in sorted(gaps.items()):
            print(f"  GAP {s} {d}: {n} prior bars")
        print("DRY-RUN: no fetch performed")
        con.close()
        return 0
    if not gaps:
        print("No gaps — nothing to backfill")
        con.close()
        return 0

    # Fetch window: pad before the earliest gap date through the day before
    # the latest gap date (bars ON the trade date are irrelevant to ATR-T1
    # but harmless; we still stop at date-1 to keep this strictly-history).
    gap_dates = sorted(d for _, d in gaps)
    start = (datetime.strptime(gap_dates[0], '%Y-%m-%d')
             - timedelta(days=FETCH_PAD_DAYS)).date()
    end = (datetime.strptime(gap_dates[-1], '%Y-%m-%d')
           - timedelta(days=1)).date()
    print(f"Fetching daily bars for {len(gap_syms)} symbols "
          f"{start} -> {end} from Alpaca (SIP)...")

    from dotenv import load_dotenv
    load_dotenv(str(ROOT / '.env'))
    from data_sources.alpaca_client import AlpacaClient
    client = AlpacaClient(os.environ.get('ALPACA_API_KEY', ''),
                          os.environ.get('ALPACA_API_SECRET', ''))
    fetched = client.get_daily_bars_range(gap_syms, start, end)

    inserted = 0
    now_iso = datetime.now(timezone.utc).isoformat()
    for sym in gap_syms:
        bars = fetched.get(sym) or []
        for b in bars:
            bar_date = str(b['date'])[:10]
            # INSERT-ONLY: OR IGNORE never touches an existing
            # (symbol, bar_date) row — cache content is append-only here.
            cur = con.execute(
                "INSERT OR IGNORE INTO daily_bars "
                "(symbol, bar_date, open, high, low, close, volume, fetched_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (sym, bar_date, float(b['open']), float(b['high']),
                 float(b['low']), float(b['close']), int(b['volume']),
                 now_iso))
            inserted += cur.rowcount
        if not bars:
            print(f"  {sym}: Alpaca returned 0 bars (genuine no-history — "
                  f"floor stays fail-open, correctly)")
    con.commit()

    total_after = con.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0]
    per_sym_after = {
        s: con.execute("SELECT COUNT(*) FROM daily_bars WHERE symbol=?",
                       (s,)).fetchone()[0]
        for s in sorted(set(gap_syms + MANDATORY_SYMBOLS))}
    print(f"Inserted {inserted:,} new rows (INSERT OR IGNORE — existing rows "
          f"untouched)")
    print(f"SNAPSHOT after: daily_bars total rows = {total_after:,} "
          f"(delta {total_after - total_before:+,})")
    print("SNAPSHOT after (per gap symbol): "
          + ', '.join(f"{s}={per_sym_before.get(s, 0)}->{n}"
                      for s, n in sorted(per_sym_after.items())))

    # Post-backfill re-scan: which candidate symbol-days still lack history?
    counts2 = prior_bar_counts(con, list(gaps.keys()))
    still = {k: v for k, v in counts2.items() if v < MIN_BARS}
    print(f"Post-backfill: {len(gaps) - len(still)} symbol-days now have "
          f">= {MIN_BARS} prior bars; {len(still)} remain short "
          f"(genuine new listings — ATR fail-open by the frozen rule)")
    for s in MANDATORY_SYMBOLS:
        hits = {k: v for k, v in counts2.items() if k[0] == s}
        for k, v in sorted(hits.items()):
            print(f"  mandatory {k[0]} {k[1]}: prior bars now {v} "
                  f"({'FLOORS' if v >= MIN_BARS else 'still fail-open'})")
    con.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
