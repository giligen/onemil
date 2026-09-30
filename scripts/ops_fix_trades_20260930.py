#!/usr/bin/env python3
"""Corrects the trades ledger for the 2026-09-30 HOD-break fill-persistence defect.

Root cause (report only; this script does NOT touch the engine — the fix is in
trading/hod_break_engine.py, same change as this script):
`_on_live_fill` registered every resting-order fill (safety-net legs, Position,
StopMonitor watch, dry ledger) but its call into `_save_pending_trade` never
passed `order_status='filled'`/`fill_price`/`filled_at`, so every NEW row landed
with `fill_price`/`filled_at`/`filled_qty` NULL forever (and, since `save_trade`'s
INSERT does not carry `filled_qty`, the follow-up `update_trade` that sets it
only fires when `order_status == 'filled'`, so that stayed NULL too). `shares`,
`stop_loss_price`, `exit_price`, `exit_reason` and `pnl` were unaffected because
`_record_exit` read the in-memory `Position.fill_price` (correctly set at fill
time) — it only went wrong for a row whose Position was later rehydrated from
this same NULL-fill_price DB row (a restart), in which case `_record_exit`'s
old silent `pos.fill_price or pos.limit_price` fallback used `entry_price`
instead. That is why the owner's own numbers below are close to whatever pnl
is already stored for these rows.

Today's 11 HOD paper fills (data/trades.db, trade_date='2026-09-30',
account='paper', strategy='hod_break', rows 403-413) were all affected. Every
(symbol, filled_at, qty, avg fill price) below was given directly by the owner
from the broker's own fill record; this script only applies them, and only
backfills fill_price/filled_at/filled_qty/pnl/pnl_pct — order_status, exit_price,
exit_reason, shares and stop_loss_price were already correct and are left
untouched.
"""
import argparse
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root, so `scripts.*` imports
                                                                    # resolve whether this runs as a
                                                                    # script (sys.path[0] == scripts/) or
                                                                    # via -m / pytest.

# Reuse the tested, generic helpers from the 9/29 incident script rather than
# duplicating them (_to_iso/_now_iso/_pnl_pct/backup_db/_row are incident-agnostic).
from scripts.ops_fix_trades_20260929 import _to_iso, _now_iso, _pnl_pct, backup_db, _row  # noqa: E402

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "trades.db"

# (symbol, trades.id, filled_at time-of-day UTC 'HH:MM:SS', filled_qty, fill_price)
# Row ids are the main session's own read of data/trades.db (rows 403-413). NOT simply chronological:
# the first version of this table put SWMR at 405 / HIMZ at 406 (assumed row-id order == fill-timestamp
# order) but the real table has HIMZ at 405 / SWMR at 406 (two fills 29s apart, 13:46:24 and 13:46:53 —
# row-insertion order did not match fill-timestamp order). fix_fill_row()'s symbol check correctly
# refused to write one symbol's price onto the other's row (SKIP, not a silent wrong write) rather than
# corrupting data, but that left both rows NULL — confirmed by the owner reading the live table directly
# (independent check) after the first run applied 9/11 and skipped exactly this pair. See
# TestRowSymbolMappingIndependentOfFillsTable in tests/test_ops_fix_trades_20260930.py.
FILLS_20260930: List[Tuple[str, int, str, int, float]] = [
    ("LITZ", 403, "13:42:25", 51, 24.78),
    ("MNDY", 404, "13:44:20", 24, 80.24),
    ("HIMZ", 405, "13:46:53", 65, 28.081692),
    ("SWMR", 406, "13:46:24", 75, 17.53),
    ("RKLX", 407, "13:51:51", 107, 18.65),
    ("USDE", 408, "13:52:10", 78, 17.05),
    ("GEN", 409, "14:29:05", 90, 22.025889),
    ("CNXC", 410, "15:23:04", 74, 26.97),
    ("MLYS", 411, "15:26:57", 77, 25.744416),
    ("CBOE", 412, "15:42:24", 7, 276.965714),
    ("DOCS", 413, "16:48:07", 71, 28.111408),
]


def fix_fill_row(conn: sqlite3.Connection, dry_run: bool, changes: List[str], *,
                  row_id: int, symbol: str, filled_at_time: str, filled_qty: int, fill_price: float) -> None:
    """Backfills fill_price/filled_at/filled_qty for one 2026-09-30 row and recomputes pnl from the REAL
    fill: (exit_price - fill_price) * filled_qty (same formula trading/hod_break_engine.py:_record_exit
    uses). order_status/exit_price/exit_reason/shares are NOT touched — they were already correct."""
    row = _row(conn, row_id)
    if row is None:
        print(f"SKIP row {row_id} {symbol}: not found in this trades table")
        return
    if row["symbol"] != symbol:
        print(f"SKIP row {row_id}: expected symbol {symbol}, found {row['symbol']!r} — refusing to touch a mismatched row")
        return
    filled_at = _to_iso(f"2026-09-30T{filled_at_time}Z")
    if (row["fill_price"] is not None and abs(row["fill_price"] - fill_price) < 1e-6
            and row["filled_at"] == filled_at and row["filled_qty"] == filled_qty):
        print(f"SKIP row {row_id} {symbol}: already backfilled (fill_price={row['fill_price']})")
        return
    old_pnl = row["pnl"]
    new_pnl = round((row["exit_price"] - fill_price) * filled_qty, 4) if row["exit_price"] is not None else None
    new_pnl_pct = _pnl_pct(new_pnl, fill_price, filled_qty) if new_pnl is not None else row["pnl_pct"]
    changes.append(
        f"UPDATE trades id={row_id} {symbol}: fill_price {row['fill_price']}->{fill_price}, "
        f"filled_at {row['filled_at']}->{filled_at}, filled_qty {row['filled_qty']}->{filled_qty}, "
        f"pnl {old_pnl}->{new_pnl} (exit_price={row['exit_price']}, shares={filled_qty})"
    )
    if dry_run:
        return
    conn.execute(
        "UPDATE trades SET fill_price=?, filled_at=?, filled_qty=?, pnl=?, pnl_pct=?, updated_at=? WHERE id=?",
        (fill_price, filled_at, filled_qty, new_pnl, new_pnl_pct, _now_iso(), row_id),
    )


def apply_corrections(conn: sqlite3.Connection, dry_run: bool) -> List[str]:
    """Backfills all 11 2026-09-30 rows, idempotently (each is a no-op if already applied). Returns the
    change descriptions actually applied (or that WOULD be applied, under dry_run). Commits once at the
    end when not a dry run."""
    changes: List[str] = []
    for symbol, row_id, filled_at_time, filled_qty, fill_price in FILLS_20260930:
        fix_fill_row(conn, dry_run, changes, row_id=row_id, symbol=symbol,
                     filled_at_time=filled_at_time, filled_qty=filled_qty, fill_price=fill_price)
    if not dry_run:
        conn.commit()
    return changes


def main() -> None:
    """CLI entry point: backup (unless --dry-run), open the DB, apply every correction, print before/after."""
    parser = argparse.ArgumentParser(description="Backfill fill_price/filled_at/filled_qty for the 9/30 HOD paper rows.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to trades.db (default: data/trades.db)")
    parser.add_argument("--dry-run", action="store_true", help="Print the changes without writing anything")
    args = parser.parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: {db_path} does not exist — nothing to fix")
        raise SystemExit(1)

    if args.dry_run:
        print(f"DRY RUN — {db_path} will NOT be modified, no backup will be made")
    else:
        backup_path = backup_db(db_path)
        print(f"Backed up {db_path} -> {backup_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        changes = apply_corrections(conn, dry_run=args.dry_run)
    finally:
        conn.close()

    if changes:
        print(f"\n{'Would apply' if args.dry_run else 'Applied'} {len(changes)} change(s):")
        for c in changes:
            print(f"  - {c}")
    else:
        print("\nNo changes needed — every correction was already applied (idempotent no-op).")


if __name__ == "__main__":
    main()
