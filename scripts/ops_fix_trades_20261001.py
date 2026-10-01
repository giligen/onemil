#!/usr/bin/env python3
"""Books the exits for the 2026-10-01 HOD-break paper after-hours flatten.

Root cause (report only; this script does NOT touch the engine)
-----------------------------------------------------------------
On 2026-10-01 the onemil-trader tick loop was stalled by research load. The
HOD-break engine's 15:55 ET force-close (`force_close_all`,
trading/hod_break_engine.py) only got to submit its four flatten orders at
20:04 UTC -- after the market close -- so they could never fill (closed/
expired at the broker). The session owner cancelled those dead orders and
flattened the four open paper positions by hand with after-hours limit sell
orders on the HOD paper account (PA39QSZR60WC, client ids 'ops-fc-<SYM>-1001-
ah'). Those orders DID fill, at 20:11:30/31 UTC, per the broker's own fill
record (verified by the owner/session before this script was written -- this
script only applies the given numbers, it never fabricates a price or time).

The four `trades` rows (trade_date='2026-10-01', strategy='hod_break',
account='paper', symbols APMD/MSTU/MSTX/COHR -- ids 414/416/417/419) were
never closed: `order_status` stayed 'filled' (open) and every exit column
(exit_price/exit_reason/exited_at/pnl/pnl_pct) stayed NULL, because the
engine's own exit-booking path (`_record_exit`) was never reached -- the
orders it knows about (the dead 20:04Z ones) never filled, and it has no
visibility into the owner's manual after-hours replacement orders.

Read vs data/trades.db (2026-10-01, sqlite ?mode=ro) confirmed `shares` ==
`filled_qty` == the broker fill quantity for all four rows -- no quantity
mismatch to flag.

Field parity with the engine's own force-close exit (report only)
-----------------------------------------------------------------
trading/hod_break_engine.py `_record_exit` (~line 2543) writes, on close:
    pnl        = (exit_price - entry) * pos.shares         (entry = fill_price)
    pnl_pct    = (exit_price / entry - 1) * 100
    order_status = 'closed'; exit_price/exit_reason/exited_at set.
`_pnl_pct(pnl, entry_price, shares)` (scripts/ops_fix_trades_20260929.py,
reused here) is the same formula written differently: pnl / (entry*shares) *
100 == (exit_price-entry)/entry*100 -- verified algebraically identical, used
for exact parity with the 9/29 and 9/30 ops scripts.

For the ordinary 15:55 flatten, the engine's own `_exit_legs` names the
force-close leg's reason literally `'eod'` (Position docstring: exit_reason
in {'target','stop','eod'}); `_book_leg_fill` only appends '+partial' if more
than one leg contributed shares, which is not the case here (none of these
four had any target/stop leg fill before the flatten). `exit_reason` is a
free-string column (`VARCHAR(20)`, no CHECK constraint -- confirmed by
grepping persistence/database.py and the 9/29 script's own precedent of a
non-enum value, `exit_reason='ops_flatten'`), so this script uses the
engine's own force-close string with an ops-booking suffix,
`EXIT_REASON = 'eod_ops_ah'`, rather than overloading pattern_data for
something the column already allows verbatim.

`exit_pricing_method`/`exit_quote_bid`/`exit_quote_ask`/`exit_fill_latency_ms`/
`exit_slippage`/`exit_submitted_at` are left NULL: those are the engine's own
submit-time quote/latency telemetry for an order IT submitted
(`force_close_all` captures the quote at submit time) -- this flatten was a
manual broker order the engine never submitted or instrumented, and no
submit-time quote or order-submitted timestamp was given, so none is
fabricated. `order_id` (the entry order's id) is left untouched, exactly as
the engine's `_record_exit` never touches it either -- the exit/close order's
own id and client id go into `pattern_data`, mirroring the keys
`force_close_all` itself writes there (`close_order_id`,
`close_client_order_id`, `closed_qty`), alongside a new `ops_note` key with
the broker fill time and this root cause, merged into the EXISTING
pattern_data JSON (book/level/consol_low/tp_leg_id/sl_leg_id/... are never
overwritten).

Broker fills applied (owner-verified, all 2026-10-01, all SELL)
-----------------------------------------------------------------
    APMD  73 sh @ 22.34   20:11:30Z  1ee2a65c-48ba-447a-8032-2ab5d2cd5c66
    MSTU  34 sh @ 43.23   20:11:31Z  19d3063b-2efa-44ad-ac0c-2778b8fd32ca
    MSTX  67 sh @ 19.88   20:11:30Z  54f51069-dd58-462c-8a62-dccb185abf96
    COHR   5 sh @ 318.90  20:11:31Z  79f7532d-e9e5-4bd0-839d-99a9bb9e0a63

Usage
-----
    python scripts/ops_fix_trades_20261001.py            # dry run (default) -- read-only, prints only
    python scripts/ops_fix_trades_20261001.py --apply     # writes, inside ONE transaction, backs up first

Deliberately the OPPOSITE flag polarity of scripts/ops_fix_trades_20260930.py
(which defaults to applying and opts OUT via --dry-run): this script was
written under an explicit dry-run-by-default instruction, so the dry run
connects to the DB with the sqlite `?mode=ro` URI -- a structural guarantee
nothing is written without --apply. --apply backs up the DB first (same
`backup_db` helper as the other ops scripts), applies all four updates inside
one transaction, commits once, then runs a verification SELECT. Idempotent:
a row already showing this script's exit_reason is skipped as a no-op.
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import List, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root, so `scripts.*` imports
                                                                    # resolve whether this runs as a
                                                                    # script (sys.path[0] == scripts/) or
                                                                    # via -m / pytest.

# Reuse the tested, generic helpers already shared by the 9/29 and 9/30 ops scripts
# (_to_iso/_now_iso/_pnl_pct/backup_db/_row are incident-agnostic).
from scripts.ops_fix_trades_20260929 import _to_iso, _now_iso, _pnl_pct, backup_db, _row  # noqa: E402

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "trades.db"
TRADE_DATE = "2026-10-01"
EXIT_REASON = "eod_ops_ah"   # engine's own force-close reason 'eod' (trading/hod_break_engine.py
                              # _exit_legs/_book_leg_fill) + an ops-booking suffix; free-string column,
                              # no CHECK constraint (verified against persistence/database.py).


class Fill(NamedTuple):
    symbol: str
    trade_id: int
    broker_qty: int
    exit_price: float
    filled_at_utc: str          # 'HH:MM:SSZ' time-of-day, TRADE_DATE
    order_id: str
    client_order_id: str


# Broker's own fill record (owner-verified before this script was written).
FILLS_20261001: List[Fill] = [
    Fill("APMD", 414, 73, 22.34, "20:11:30Z", "1ee2a65c-48ba-447a-8032-2ab5d2cd5c66", "ops-fc-APMD-1001-ah"),
    Fill("MSTU", 416, 34, 43.23, "20:11:31Z", "19d3063b-2efa-44ad-ac0c-2778b8fd32ca", "ops-fc-MSTU-1001-ah"),
    Fill("MSTX", 417, 67, 19.88, "20:11:30Z", "54f51069-dd58-462c-8a62-dccb185abf96", "ops-fc-MSTX-1001-ah"),
    Fill("COHR", 419,  5, 318.90, "20:11:31Z", "79f7532d-e9e5-4bd0-839d-99a9bb9e0a63", "ops-fc-COHR-1001-ah"),
]


def _merge_pattern_data(existing_json: str, *, close_order_id: str, close_client_order_id: str,
                         closed_qty: int, ops_note: dict) -> str:
    """Merges ops-booking keys into the row's EXISTING pattern_data JSON, never dropping the keys already
    there (book/level/consol_low/entry_mode/target_r/tp_leg_id/sl_leg_id/limit/target/client_order_id/...).
    Mirrors the key names trading/hod_break_engine.py:force_close_all itself writes on a normal flatten
    (close_order_id/close_client_order_id/closed_qty) plus a new 'ops_note' key for this incident."""
    data = json.loads(existing_json) if existing_json else {}
    data["close_order_id"] = close_order_id
    data["close_client_order_id"] = close_client_order_id
    data["closed_qty"] = closed_qty
    data["ops_note"] = ops_note
    return json.dumps(data)


def book_exit(conn: sqlite3.Connection, dry_run: bool, changes: List[str], *, fill: Fill) -> None:
    """Books one broker SELL fill as this row's exit, field-for-field parity with
    trading/hod_break_engine.py:_record_exit (pnl/pnl_pct formula, order_status='closed') -- see module
    docstring for exactly which fields are NOT set and why. Refuses (SKIP, not a silent wrong write) a
    row that doesn't exist, isn't this symbol, or is already closed with this exit_reason."""
    row = _row(conn, fill.trade_id)
    if row is None:
        print(f"SKIP row {fill.trade_id} {fill.symbol}: not found in this trades table")
        return
    if row["symbol"] != fill.symbol:
        print(f"SKIP row {fill.trade_id}: expected symbol {fill.symbol}, found {row['symbol']!r} — refusing to touch a mismatched row")
        return
    if row["exit_reason"] == EXIT_REASON and row["exit_price"] is not None:
        print(f"SKIP row {fill.trade_id} {fill.symbol}: already booked (exit_price={row['exit_price']})")
        return

    stored_qty = row["filled_qty"] if row["filled_qty"] is not None else row["shares"]
    if stored_qty != fill.broker_qty:
        print(f"MISMATCH row {fill.trade_id} {fill.symbol}: ledger qty={stored_qty} != broker fill qty={fill.broker_qty} "
              f"— booking the BROKER quantity ({fill.broker_qty}) for the exit, ledger shares left untouched")

    entry = row["fill_price"] if row["fill_price"] else row["entry_price"]
    shares_for_pnl = fill.broker_qty                      # always the broker's own fill qty, never the (possibly stale) ledger qty
    pnl = round((fill.exit_price - entry) * shares_for_pnl, 4)
    pnl_pct = _pnl_pct(pnl, entry, shares_for_pnl)
    exited_at = _to_iso(f"{TRADE_DATE}T{fill.filled_at_utc}")

    ops_note = {
        "source": "ops_fix_trades_20261001",
        "reason": ("tick loop stalled by research load; 15:55 ET force-close orders were submitted at "
                   "20:04 UTC, after the close, and could not fill; session owner cancelled them and "
                   "flattened the position with an after-hours limit sell on the HOD paper account "
                   "(PA39QSZR60WC)"),
        "broker_order_id": fill.order_id,
        "client_order_id": fill.client_order_id,
        "filled_at_utc": exited_at,
        "exit_submitted_at": None,   # not given by the broker fill record — never fabricated
    }
    pattern_data = _merge_pattern_data(
        row["pattern_data"], close_order_id=fill.order_id, close_client_order_id=fill.client_order_id,
        closed_qty=shares_for_pnl, ops_note=ops_note,
    )

    changes.append(
        f"UPDATE trades id={fill.trade_id} {fill.symbol}: order_status {row['order_status']!r}->'closed', "
        f"exit_price None->{fill.exit_price}, exit_reason None->'{EXIT_REASON}', exited_at None->{exited_at}, "
        f"pnl None->{pnl}, pnl_pct None->{round(pnl_pct, 4)} (entry={entry}, shares={shares_for_pnl})"
    )
    if dry_run:
        return
    conn.execute(
        "UPDATE trades SET order_status='closed', exit_price=?, exit_reason=?, exited_at=?, pnl=?, pnl_pct=?, "
        "pattern_data=?, updated_at=? WHERE id=?",
        (fill.exit_price, EXIT_REASON, exited_at, pnl, pnl_pct, pattern_data, _now_iso(), fill.trade_id),
    )


def apply_corrections(conn: sqlite3.Connection, dry_run: bool) -> List[str]:
    """Books all four 2026-10-01 after-hours flatten exits, idempotently (a no-op if already applied).
    Returns the change descriptions actually applied (or that WOULD be applied, under dry_run). Commits
    once at the end when not a dry run -- one transaction for all four rows."""
    changes: List[str] = []
    for fill in FILLS_20261001:
        book_exit(conn, dry_run, changes, fill=fill)
    if not dry_run:
        conn.commit()
    return changes


def _print_verification(db_path: Path) -> None:
    """Read-only re-open + SELECT of the four rows post-apply, so the printed output is proof of what is
    actually on disk, not just of what this process thinks it wrote."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cols = "id,symbol,shares,filled_qty,fill_price,order_status,exit_price,exit_reason,exited_at,pnl,pnl_pct"
    rows = conn.execute(
        f"SELECT {cols} FROM trades WHERE trade_date=? AND strategy='hod_break' AND account='paper' "
        "AND symbol IN ('APMD','MSTU','MSTX','COHR') ORDER BY id",
        (TRADE_DATE,),
    ).fetchall()
    conn.close()
    print("\nVerification SELECT (fresh read-only connection):")
    for r in rows:
        print(f"  id={r['id']} {r['symbol']}: shares={r['shares']} filled_qty={r['filled_qty']} "
              f"order_status={r['order_status']!r} exit_price={r['exit_price']} exit_reason={r['exit_reason']!r} "
              f"exited_at={r['exited_at']} pnl={r['pnl']} pnl_pct={r['pnl_pct']}")
    total = sum(r["pnl"] for r in rows if r["pnl"] is not None)
    print(f"  TOTAL pnl across {len(rows)} row(s): {round(total, 4)}")


def main() -> None:
    """CLI entry point. Dry run by DEFAULT (read-only connection, prints only, no backup, no write).
    --apply backs up the DB, writes all four rows in one transaction, then re-reads and prints the
    verification SELECT."""
    parser = argparse.ArgumentParser(description="Book the 2026-10-01 HOD-break paper after-hours flatten exits.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to trades.db (default: data/trades.db)")
    parser.add_argument("--apply", action="store_true", help="Write the exits (default: dry run, no writes)")
    args = parser.parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: {db_path} does not exist — nothing to fix")
        raise SystemExit(1)

    dry_run = not args.apply
    if dry_run:
        print(f"DRY RUN — {db_path} opened READ-ONLY (?mode=ro); nothing will be modified, no backup will be made")
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        backup_path = backup_db(db_path)
        print(f"Backed up {db_path} -> {backup_path}")
        conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        changes = apply_corrections(conn, dry_run=dry_run)
    finally:
        conn.close()

    if changes:
        print(f"\n{'Would apply' if dry_run else 'Applied'} {len(changes)} change(s):")
        for c in changes:
            print(f"  - {c}")
    else:
        print("\nNo changes needed — every correction was already applied (idempotent no-op).")

    if not dry_run:
        _print_verification(db_path)


if __name__ == "__main__":
    main()
