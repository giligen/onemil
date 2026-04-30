#!/usr/bin/env python3
"""Reconcile ORB orphan trades — fix DB rows that lost their exits.

Background
==========
Some ORB trades were filled on Alpaca but never had their exit details
recorded in our trades DB. Causes (all addressed by commit 1a89165 going
forward):

- Service crash between BUY fill and DB write (4/28 OPRA — service died
  at 13:34/13:37 UTC, OPRA filled at 13:39 UTC during the crash window)
- Force-close didn't run for ORB at EOD (scanner only triggered bull flag
  FC, then `force_closed=True` blocked ORB's tick — OPRA carried overnight)
- Stale Alpaca-queued bracket SL fired the next morning, position closed
  without our engine seeing it

The DB rows for these positions are stuck with `pnl IS NULL` (we never
finished the exit-write path). Result: weekly P&L reports show a falsely
optimistic picture — the realized losses are real on the broker side
but invisible in our books.

This script finds those rows and fills in the missing exit details from
Alpaca's order history.

What it does
============
1. Query trades.db for rows where strategy='orb' AND fill_price IS NOT NULL
   AND exit_price IS NULL — these are the orphans.
2. For each, query the ORB Alpaca account's order history for SELL orders
   on that symbol after the entry fill timestamp.
3. Pick the matching close (typically the FIRST filled SELL of >= entry qty)
   and record exit_price, exit_reason='orphan_reconciled', pnl,
   pnl_pct, exited_at into the DB row.
4. Default DRY-RUN: prints what would change, applies nothing. Pass
   --apply to actually update the DB.

Idempotent: re-running after a successful apply finds zero orphans
because the previous run already filled in exit details.

Usage
=====
    # Preview what would change
    python scripts/reconcile_orb_orphans.py

    # Actually update the DB
    python scripts/reconcile_orb_orphans.py --apply

    # Limit to specific symbols
    python scripts/reconcile_orb_orphans.py OPRA ANNA SMCX --apply
"""
from __future__ import annotations

import argparse
import os
import sys
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from data_sources.alpaca_client import AlpacaClient


def find_orphans(con: sqlite3.Connection, symbols_filter: Optional[List[str]] = None):
    """Return list of orphan rows: (id, trade_date, symbol, fill_price, shares, filled_at)."""
    sql = """
        SELECT id, trade_date, symbol, fill_price, shares, filled_at
        FROM trades
        WHERE strategy = 'orb'
          AND fill_price IS NOT NULL
          AND exit_price IS NULL
        ORDER BY trade_date, id
    """
    rows = con.execute(sql).fetchall()
    if symbols_filter:
        keep = set(s.upper() for s in symbols_filter)
        rows = [r for r in rows if r[2] in keep]
    return rows


def find_matching_sell(alpaca: AlpacaClient, symbol: str, entry_filled_at: datetime,
                       qty: int) -> Optional[dict]:
    """Find the SELL order that closed an orphan position.

    Strategy: pull all FILLED SELL orders for `symbol` whose filled_at is
    AFTER `entry_filled_at`. Pick the earliest one whose filled_qty
    matches `qty` (or the only candidate, if there's just one).

    Returns a dict with keys: filled_avg_price, filled_qty, filled_at,
    order_id; or None if no match.
    """
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus, OrderSide
    except ImportError as e:
        print(f"  [error] alpaca-py imports failed: {e}", file=sys.stderr)
        return None

    try:
        req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            symbols=[symbol],
            side=OrderSide.SELL,
            limit=50,
        )
        orders = alpaca.trading_client.get_orders(filter=req) or []
    except Exception as e:
        print(f"  [error] {symbol}: get_orders failed: {e}", file=sys.stderr)
        return None

    candidates = []
    for o in orders:
        try:
            status = str(getattr(o, 'status', '')).lower()
            if 'filled' not in status:
                continue
            filled_at_raw = getattr(o, 'filled_at', None)
            if filled_at_raw is None:
                continue
            filled_at = filled_at_raw if isinstance(filled_at_raw, datetime) \
                else datetime.fromisoformat(str(filled_at_raw).replace('Z', '+00:00'))
            if filled_at.tzinfo is None:
                filled_at = filled_at.replace(tzinfo=timezone.utc)
            if filled_at <= entry_filled_at:
                continue
            fap = getattr(o, 'filled_avg_price', None)
            fq = getattr(o, 'filled_qty', None)
            if fap is None or fq is None:
                continue
            candidates.append({
                'filled_avg_price': float(fap),
                'filled_qty': int(fq),
                'filled_at': filled_at,
                'order_id': str(getattr(o, 'id', '')),
            })
        except Exception as e:
            print(f"  [warn] {symbol}: parse order: {e}", file=sys.stderr)
            continue

    if not candidates:
        return None
    # Prefer the earliest matching qty; fall back to earliest overall
    qty_matches = [c for c in candidates if c['filled_qty'] == qty]
    pool = qty_matches or candidates
    pool.sort(key=lambda c: c['filled_at'])
    return pool[0]


def reconcile(apply: bool, symbols_filter: Optional[List[str]] = None) -> int:
    """Run reconciliation. Returns count of rows updated (0 if dry-run)."""
    db_path = ROOT / 'data' / 'trades.db'
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    orphans = find_orphans(con, symbols_filter)
    if not orphans:
        print("No ORB orphans found (no rows with fill_price set + exit_price NULL).")
        return 0

    print(f"Found {len(orphans)} ORB orphan(s):\n")

    # Connect to ORB Alpaca account
    key = os.getenv('ALPACA_ORB_API_KEY')
    sec = os.getenv('ALPACA_ORB_API_SECRET')
    if not key or not sec:
        print("ERROR: ALPACA_ORB_API_KEY / ALPACA_ORB_API_SECRET not in env.",
              file=sys.stderr)
        return 0
    alpaca = AlpacaClient(key, sec, paper=True)

    updates_planned = 0
    updates_applied = 0
    no_match = []

    for row in orphans:
        trade_id = row['id']
        symbol = row['symbol']
        trade_date = row['trade_date']
        fill = float(row['fill_price'])
        shares = int(row['shares'])
        filled_at_raw = row['filled_at']

        # Parse filled_at (string in DB) to UTC datetime
        if filled_at_raw:
            filled_at = datetime.fromisoformat(
                str(filled_at_raw).replace('Z', '+00:00')
            )
            if filled_at.tzinfo is None:
                filled_at = filled_at.replace(tzinfo=timezone.utc)
        else:
            # No filled_at recorded — use start of trade_date as a permissive lower bound
            filled_at = datetime.fromisoformat(f"{trade_date}T00:00:00+00:00")

        print(f"  [{symbol}] trade_id={trade_id}  fill=${fill:.4f} x{shares}  "
              f"date={trade_date}  filled_at={filled_at.isoformat()}")

        match = find_matching_sell(alpaca, symbol, filled_at, shares)
        if match is None:
            print(f"    → no matching SELL on Alpaca — skipping")
            no_match.append(symbol)
            continue

        exit_price = match['filled_avg_price']
        pnl = (exit_price - fill) * shares
        pnl_pct = (exit_price - fill) / fill * 100  # in percent

        print(f"    → exit ${exit_price:.4f} on {match['filled_at'].isoformat()} "
              f"(qty={match['filled_qty']})  pnl={pnl:+,.2f} ({pnl_pct:+.2f}%)")

        updates_planned += 1
        if apply:
            con.execute(
                """
                UPDATE trades
                SET exit_price = ?, pnl = ?, pnl_pct = ?,
                    exit_reason = 'orphan_reconciled',
                    exited_at = ?, updated_at = datetime('now')
                WHERE id = ?
                """,
                (
                    exit_price, pnl, pnl_pct,
                    match['filled_at'].isoformat(), trade_id,
                ),
            )
            updates_applied += 1

    if apply:
        con.commit()
        print(f"\n✅ Applied {updates_applied} update(s) to trades.db")
    else:
        print(f"\nDry-run: {updates_planned} row(s) would be updated. "
              f"Re-run with --apply to commit.")

    if no_match:
        print(f"\nNo Alpaca SELL match for: {','.join(no_match)} "
              f"(may need manual review or the position is still open).")

    con.close()
    return updates_applied


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('symbols', nargs='*',
                        help='Specific symbols to reconcile (default: all orphans)')
    parser.add_argument('--apply', action='store_true',
                        help='Apply updates to DB. Without this flag, dry-run.')
    args = parser.parse_args()

    reconcile(apply=args.apply, symbols_filter=args.symbols or None)


if __name__ == '__main__':
    main()
