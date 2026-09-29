#!/usr/bin/env python3
"""Corrects the trades ledger for the 2026-09-25 -> 2026-09-29 HOD-break incident.

Every number below was VERIFIED by the main session against Alpaca's broker order
history before this script was written; this script only applies them to
data/trades.db. It never fabricates a price, time or P&L that wasn't given.

Incident summary
-----------------
1. CDNA (hod_break, live, 2026-09-25): row 381 was booked with a fabricated 'eod'
   exit (63.5001 / -88.16). The real exit was StopMonitor's stop_loss fill at
   2026-09-25T18:55:05Z @ 64.1043 (pnl -53.72). Separately, the exit_qty_guard
   defect (trading/exit_qty_guard.py) re-sold the SAME 57 shares again at
   2026-09-25T15:55Z with no broker check, opening a naked short on the shared
   LIVE account that sat open until it was covered 2026-09-29T13:41Z @ 65.1152
   (pnl -92.06). Row 381 is corrected in place; the erroneous short is a second,
   new row (side='sell' + a pattern_data direction marker, since every existing
   row in this table is side='buy' and there is no other short precedent to
   follow — 'sell' is the schema's own antonym and fits the VARCHAR(4) column).
2. PRIM / ASTN / WRBY (hod_break, live, 2026-09-29): three round trips the engine
   never registered at all — the cancel/replace fill-detection gap named in
   trading/hod_break_engine.py's _adopt_unregistered_positions_on_boot docstring
   ("2026-09-29 PRIM/WRBY/ASTN: exactly this gap"). Inserted from broker-verified
   fills, exit_reason='ops_flatten'.
3. TTAN / CONI (hod_break, paper): row 393 (TTAN) was exited by StopMonitor's
   market fallback at 2026-09-29T15:13:37Z @ 63.12 but the engine never confirmed
   the fill (stuck at order_status='exit_pending_verification', exit_price NULL) —
   closed in place. Paper CONI has THREE rows for one 112-share broker position
   (account PA39QSZR60WC; order 9e39861c bought 87 sh @ 22.88 filled 15:02Z,
   client-order ...1db707a0 bought 25 sh @ 22.89 filled ~15:17Z, broker blended
   avg 22.882231): row 394 (87 sh, order_id='adopted-CONI', first boot adoption
   15:15:53Z), row 395 (25 sh, a real broker order id, the ordinary resting-fill
   registration), and row 396 (112 sh, order_id='adopted-CONI' again, a SECOND
   boot re-adoption at 17:27:06Z of the broker's single combined position). The
   engine's registry holds one position per symbol, so the ledger's end state
   must be ONE open row too: row 396 becomes the position (order_status
   'filled', shares/filled_qty 112, the broker's blended fill_price, a
   take_profit_price computed from the ORIGINAL entry 22.88 and stop 22.63 at
   config.yaml's hod_break.target_r — never the blended price, matching how the
   engine itself targets off the breakout level, not cost basis — and empty
   tp_leg_id/sl_leg_id, since the 25-sh OCO legs are cancelled at the broker by
   the main session and StopMonitor manages the stop from here). Rows 394 and
   395 are marked order_status='canceled' (exactly what scripts/eod_report.py:69
   already excludes from trade counts, alongside 'cancelled'/'expired'/
   'rejected') with exit fields NULL and a pattern_data merged_into note.

CONI triple -- code cause (report only; this script does NOT touch the engine)
--------------------------------------------------------------------------------
trading/hod_break_engine.py:_adopt_unregistered_positions_on_boot (lines 1311-1369)
found an 87-share broker CONI position not yet in `self.positions` and inserted
row 394 via `_save_pending_trade` (line 1352) — confirmed by row 394's own data:
order_id == 'adopted-CONI' and pattern_data == {"adopted_on_boot": true}, which
are exactly the literals at lines 1350-1351 (`f'adopted-{sym}'`,
`'adopted_on_boot': True`). That adoption path registers `self.positions[sym]`
(line 1354) but never writes back into `cand.live_order['trade_id']` — in fact it
never touches `cand.live_order` at all. So when the SAME resting order's
remaining quantity (25 sh) later filled through the normal path, `_on_live_fill`'s
own dedup guard at line 1693 (`lo.get('trade_id') or self._save_pending_trade(...)`)
found no `trade_id` on `lo` and inserted a SECOND row (395 — a real broker order
id, plus the ordinary resting-fill pattern_data shape with tp_leg_id/sl_leg_id/
client_order_id='hod-rest-CONI-09-29-1db707a0', matching the dict literal built at
lines 1690-1692) instead of updating row 394's shares from 87 to the true 112.
Then, at 17:27:06Z, a SECOND process restart wiped `self.positions` again (the
same in-memory-only gap — `sync_positions()`, line 2415, wired in at
main.py:829, IS this engine's DB-backed restart-safe registry rebuild, but
`_adopt_unregistered_positions_on_boot`'s own per-symbol dedup at line 1334
(`sym in self.positions`) never consults it or the trades table), so boot
adoption fired a THIRD time — this time correctly reading the broker's single,
already-merged 112-share position (row 396) confirming 87 + 25 == 112 all cover
the same physical position. Three inserts for one broker fill, same root cause
each time.

Usage
-----
    python scripts/ops_fix_trades_20260929.py [--db PATH] [--dry-run]

Backs up --db (default data/trades.db) to '<db>.bak_<UTC timestamp>' before any
write. Idempotent: every correction checks the row's CURRENT state first and is a
no-op on a second run. --dry-run prints what would change and writes nothing
(including no backup).
"""
import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "trades.db"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _to_iso(ts: str) -> str:
    """Normalize a 'Z'-suffixed UTC timestamp to this DB's own '+00:00' convention
    (see persistence/database.py created_at/updated_at, e.g.
    '...T15:15:53.012360+00:00')."""
    return ts[:-1] + "+00:00" if ts.endswith("Z") else ts


def _now_iso() -> str:
    """Current UTC time in this DB's timestamp convention."""
    return datetime.now(timezone.utc).isoformat()


def _pnl_pct(pnl: float, entry_price: float, shares: int) -> float:
    """Same formula the engine already stores (verified against the real row 381:
    -88.1619 / (65.0468 * 57) * 100 == -2.37782642651138, its own stored pnl_pct)."""
    notional = entry_price * shares
    return (pnl / notional * 100.0) if notional else 0.0


def backup_db(db_path: Path) -> Path:
    """Copies db_path to '<db_path>.bak_<UTC stamp>' before any write. Never called
    under --dry-run."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = db_path.with_name(db_path.name + f".bak_{stamp}")
    shutil.copy2(db_path, backup_path)
    return backup_path


def _row(conn: sqlite3.Connection, trade_id: int) -> Optional[dict]:
    """Fetch one trades row by id as a plain dict, or None if it doesn't exist."""
    cur = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
    row = cur.fetchone()
    return dict(row) if row else None


def _astn_entry_ts_from_broker() -> Optional[str]:
    """Best-effort, READ-ONLY lookup of ASTN's real buy-fill time from the live
    account's closed-order history. Never places, cancels or modifies any order.
    Returns None (caller falls back to the main session's verified estimate,
    2026-09-29T13:52:00Z) whenever credentials are absent or the call fails for
    any reason — logged via print(), never raised."""
    import os

    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET")
    if not (key and secret):
        print("ASTN fill time: no ALPACA_API_KEY/SECRET in this environment "
              "— using the verified fallback 2026-09-29T13:52:00Z")
        return None
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        client = TradingClient(key, secret, paper=False)
        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=["ASTN"])
        for o in client.get_orders(req):
            if str(getattr(o, "side", "")).lower().endswith("buy") and getattr(o, "filled_at", None):
                ts = o.filled_at.isoformat() if hasattr(o.filled_at, "isoformat") else str(o.filled_at)
                print(f"ASTN fill time: broker read-only lookup found {ts}")
                return ts
    except Exception as e:
        print(f"WARNING ASTN fill time: broker read-only lookup failed ({e}) "
              f"— using the verified fallback 2026-09-29T13:52:00Z")
    return None


def fix_cdna_381(conn: sqlite3.Connection, dry_run: bool, changes: List[str]) -> None:
    """Correction 1a: repair row 381's fabricated 'eod' exit with the real
    StopMonitor stop_loss fill."""
    row = _row(conn, 381)
    if row is None:
        print("SKIP row 381: not found in this trades table")
        return
    target_exit_price, target_reason = 64.1043, "stop_loss"
    target_exited_at, target_pnl = _to_iso("2026-09-25T18:55:05Z"), -53.72
    if (row["exit_reason"] == target_reason and row["exit_price"] is not None
            and abs(row["exit_price"] - target_exit_price) < 1e-6):
        print(f"SKIP row 381: already corrected (exit_reason={row['exit_reason']}, exit_price={row['exit_price']})")
        return
    pnl_pct = _pnl_pct(target_pnl, row["entry_price"], row["shares"])
    changes.append(
        f"UPDATE trades id=381 CDNA: exit_price {row['exit_price']}->{target_exit_price}, "
        f"exit_reason {row['exit_reason']!r}->{target_reason!r}, exited_at -> {target_exited_at}, "
        f"pnl {row['pnl']}->{target_pnl}"
    )
    if dry_run:
        return
    conn.execute(
        "UPDATE trades SET exit_price=?, exit_reason=?, exited_at=?, pnl=?, pnl_pct=?, "
        "order_status=?, updated_at=? WHERE id=381",
        (target_exit_price, target_reason, target_exited_at, target_pnl, pnl_pct, "closed", _now_iso()),
    )


def insert_cdna_over_exit_short(conn: sqlite3.Connection, dry_run: bool, changes: List[str]) -> None:
    """Correction 1b: record the erroneous exit_qty_guard re-sell as a new SHORT
    round trip, distinct from row 381."""
    symbol, trade_date, exit_reason = "CDNA", "2026-09-25", "over_exit_short"
    entry_price, exit_price, shares, pnl = 63.5001, 65.1152, 57, -92.06
    existing = conn.execute(
        "SELECT id FROM trades WHERE symbol=? AND trade_date=? AND exit_reason=?",
        (symbol, trade_date, exit_reason),
    ).fetchone()
    if existing:
        print(f"SKIP INSERT CDNA over_exit_short: already present as row id={existing['id']}")
        return
    entry_ts, exit_ts = _to_iso("2026-09-25T15:55:00Z"), _to_iso("2026-09-29T13:41:00Z")
    pnl_pct = _pnl_pct(pnl, entry_price, shares)
    changes.append(
        f"INSERT trades CDNA hod_break {trade_date}: SHORT sell {shares}@{entry_price} {entry_ts} "
        f"-> cover {exit_price} {exit_ts}, pnl {pnl}, exit_reason=over_exit_short, side=sell, account=NULL"
    )
    if dry_run:
        return
    now = _now_iso()
    pattern_data = json.dumps({
        "direction": "short",
        "source": "ops_fix_trades_20260929",
        "note": ("erroneous re-sell of the same 57 sh already exited by the real stop_loss (row 381); "
                 "exit_qty_guard defect, opened a naked short 2026-09-25T15:55Z, covered 2026-09-29T13:41Z"),
    })
    conn.execute(
        """INSERT INTO trades (trade_date, symbol, side, entry_price, stop_loss_price, take_profit_price,
               shares, risk_per_share, total_risk, risk_reward_ratio, order_id, order_status,
               fill_price, filled_at, exit_price, exit_reason, exited_at, pnl, pnl_pct,
               pattern_data, strategy, account, created_at, updated_at)
           VALUES (?, ?, 'sell', ?, ?, ?, ?, 0.0, 0.0, 0.0, NULL, 'closed', ?, ?, ?, ?, ?, ?, ?, ?,
                   'hod_break', NULL, ?, ?)""",
        (trade_date, symbol, entry_price, entry_price, entry_price, shares,
         entry_price, entry_ts, exit_price, exit_reason, exit_ts, pnl, pnl_pct, pattern_data, now, now),
    )


def insert_ops_flatten(conn: sqlite3.Connection, dry_run: bool, changes: List[str], *, symbol: str,
                        entry_price: float, entry_ts: str, exit_price: float, exit_ts: str,
                        shares: int, pnl: float, trade_date: str = "2026-09-29") -> None:
    """Correction 2: insert one live round trip the engine never registered at all
    (the cancel/replace fill-detection gap; see trading/hod_break_engine.py
    _adopt_unregistered_positions_on_boot, whose own docstring names this exact
    9/29 PRIM/WRBY/ASTN gap). account is stored as NULL, not the string 'live',
    per the main session's instruction — ops-inserted rows are then trivially
    selectable with `WHERE account IS NULL`."""
    existing = conn.execute(
        "SELECT id FROM trades WHERE symbol=? AND trade_date=? AND exit_reason='ops_flatten' "
        "AND ABS(entry_price-?)<0.0001 AND shares=?",
        (symbol, trade_date, entry_price, shares),
    ).fetchone()
    if existing:
        print(f"SKIP INSERT {symbol} ops_flatten: already present as row id={existing['id']}")
        return
    entry_ts_n, exit_ts_n = _to_iso(entry_ts), _to_iso(exit_ts)
    pnl_pct = _pnl_pct(pnl, entry_price, shares)
    changes.append(
        f"INSERT trades {symbol} hod_break {trade_date}: buy {shares}@{entry_price} {entry_ts_n} "
        f"-> sell {exit_price} {exit_ts_n}, pnl {pnl}, exit_reason=ops_flatten, account=NULL"
    )
    if dry_run:
        return
    now = _now_iso()
    pattern_data = json.dumps({
        "source": "ops_fix_trades_20260929",
        "note": "broker fill the engine never registered (cancel/replace fill-detection gap, 2026-09-29)",
    })
    conn.execute(
        """INSERT INTO trades (trade_date, symbol, side, entry_price, stop_loss_price, take_profit_price,
               shares, risk_per_share, total_risk, risk_reward_ratio, order_id, order_status,
               fill_price, filled_at, exit_price, exit_reason, exited_at, pnl, pnl_pct,
               pattern_data, strategy, account, created_at, updated_at)
           VALUES (?, ?, 'buy', ?, ?, ?, ?, 0.0, 0.0, 0.0, NULL, 'closed', ?, ?, ?, 'ops_flatten', ?, ?, ?, ?,
                   'hod_break', NULL, ?, ?)""",
        (trade_date, symbol, entry_price, entry_price, entry_price, shares,
         entry_price, entry_ts_n, exit_price, exit_ts_n, pnl, pnl_pct, pattern_data, now, now),
    )


def close_ttan_393(conn: sqlite3.Connection, dry_run: bool, changes: List[str]) -> None:
    """Correction 3a: close row 393 with the real StopMonitor market-fallback
    fill the engine never confirmed."""
    row = _row(conn, 393)
    if row is None:
        print("SKIP row 393: not found in this trades table")
        return
    target_exit_price, target_reason = 63.12, "stop_loss"
    target_exited_at, target_pnl = _to_iso("2026-09-29T15:13:37Z"), -16.74
    if (row["exit_price"] is not None and abs(row["exit_price"] - target_exit_price) < 1e-6
            and row["exit_reason"] == target_reason):
        print(f"SKIP row 393: already closed (exit_price={row['exit_price']})")
        return
    pnl_pct = _pnl_pct(target_pnl, row["entry_price"], row["shares"])
    changes.append(
        f"UPDATE trades id=393 TTAN: exit_price {row['exit_price']}->{target_exit_price}, "
        f"exit_reason -> {target_reason!r}, exited_at -> {target_exited_at}, pnl -> {target_pnl}"
    )
    if dry_run:
        return
    conn.execute(
        "UPDATE trades SET exit_price=?, exit_reason=?, exited_at=?, pnl=?, pnl_pct=?, "
        "order_status=?, updated_at=? WHERE id=393",
        (target_exit_price, target_reason, target_exited_at, target_pnl, pnl_pct, "closed", _now_iso()),
    )


def _hod_target_r(config_path: Path = DEFAULT_CONFIG_PATH) -> float:
    """Reads hod_break.target_r from config.yaml — never guessed, never
    hardcoded, so the take-profit this script computes always matches whatever
    the engine is actually configured to use. Fails closed (raises) if the key
    is missing: a silently wrong take-profit is worse than a crashed script."""
    import yaml
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    return float(cfg["hod_break"]["target_r"])


def merge_coni_triple(conn: sqlite3.Connection, dry_run: bool, changes: List[str],
                       target_r: Optional[float] = None) -> None:
    """Correction 4: CONI's broker position (account PA39QSZR60WC) was booked
    into three separate trades rows by three successive boot-adoption/fill
    events (394, 395, 396 — see the module docstring's 'CONI triple' section for
    the code cause) even though the engine's own registry holds one position per
    symbol. Row 396 (the latest, already carrying the broker's true combined
    112 sh) becomes the single surviving open row: order_status 'filled',
    shares/filled_qty 112, fill_price the broker's blended avg (22.882231),
    filled_at the 25-sh fill's timestamp (2026-09-29T15:17:20Z — the moment the
    position was actually complete), take_profit_price recomputed from the
    ORIGINAL entry level (22.88) and stop (22.63) at config.yaml's
    hod_break.target_r (never the blended price — the strategy targets off the
    breakout level, not cost basis), and pattern_data.merged_from=[394, 395]
    with tp_leg_id/sl_leg_id cleared (those 25-sh OCO legs are cancelled at the
    broker by the main session; StopMonitor manages the stop from here). Rows
    394 and 395 are marked order_status='canceled' — exactly what
    scripts/eod_report.py:69 already excludes from trade counts alongside
    'cancelled'/'expired'/'rejected' — with exit fields NULL and a pattern_data
    merged_into=396 note; their entry data (shares, prices, timestamps, OCO leg
    ids) is left in place as the audit trail, only order_status and exit fields
    change."""
    row394, row395, row396 = _row(conn, 394), _row(conn, 395), _row(conn, 396)
    if row396 is None:
        print("SKIP CONI triple merge: row 396 not found in this trades table")
        return
    existing_pd = json.loads(row396["pattern_data"] or "{}")
    if row396["order_status"] == "filled" and "merged_from" in existing_pd:
        print("SKIP CONI triple merge: row 396 already merged")
        return
    if target_r is None:
        target_r = _hod_target_r()
    entry_basis, stop = 22.88, 22.63
    take_profit = round(entry_basis + target_r * (entry_basis - stop), 2)
    fill_price, filled_at = 22.882231, _to_iso("2026-09-29T15:17:20Z")

    changes.append(
        f"UPDATE trades id=396 CONI: order_status {row396['order_status']!r}->'filled', "
        f"shares {row396['shares']}->112, filled_qty ->112, fill_price ->{fill_price}, "
        f"filled_at ->{filled_at}, take_profit_price {row396['take_profit_price']}->{take_profit} "
        f"(target_r={target_r} from config.yaml), pattern_data.merged_from=[394,395], "
        f"tp_leg_id/sl_leg_id -> ''"
    )
    for rid, row in ((394, row394), (395, row395)):
        if row is not None:
            changes.append(
                f"UPDATE trades id={rid} CONI: order_status {row['order_status']!r}->'canceled', "
                f"exit fields -> NULL, pattern_data.merged_into=396"
            )

    if dry_run:
        return

    now = _now_iso()
    merged_pd = dict(existing_pd)
    merged_pd.update({"merged_from": [394, 395], "tp_leg_id": "", "sl_leg_id": ""})
    conn.execute(
        "UPDATE trades SET order_status='filled', shares=112, filled_qty=112, fill_price=?, "
        "filled_at=?, take_profit_price=?, pattern_data=?, updated_at=? WHERE id=396",
        (fill_price, filled_at, take_profit, json.dumps(merged_pd), now),
    )
    note = ("merged into row 396 by ops_fix_trades_20260929.py -- CONI triple "
            "(394 + 395 + 396 -> one 112-share position)")
    for rid, row in ((394, row394), (395, row395)):
        if row is None:
            continue
        pd = json.loads(row["pattern_data"] or "{}")
        pd.update({"merged_into": 396, "note": note})
        conn.execute(
            "UPDATE trades SET order_status='canceled', exit_price=NULL, exit_reason=NULL, "
            "exited_at=NULL, pnl=NULL, pnl_pct=NULL, pattern_data=?, updated_at=? WHERE id=?",
            (json.dumps(pd), now, rid),
        )


def book_coni_partial_exit(conn: sqlite3.Connection, dry_run: bool, changes: List[str]) -> None:
    """Correction 5: the 25-share take-profit leg of the old row 395 (tp_leg_id
    starting 6438758a) FILLED on the HOD paper account before it could be
    cancelled — SELL limit 25 sh @ 23.48, 2026-09-29T17:38:15Z (limit was
    23.41). The broker now holds 87 CONI, not 112. Booked onto the merged row
    396 as a PARTIAL exit — shares/order_status stay 112/'filled', this is not
    a full close. pattern_data.closed_qty/closed_notional are what the engine's
    sync actually reads: Position.open_qty (trading/hod_break_engine.py, lines
    212-213) is `shares - closed_qty`, written the same way by a live partial
    fill (line 2071's `_update_pattern_data(pos, closed_qty=..., closed_notional=...)`)
    and read back the same way when positions are rehydrated (line 2506,
    `pd_.get('closed_qty')`) — 112 - 25 = 87 = the broker's true holding.
    partial_exit_shares/price/pnl/reason/exited_at (the dedicated columns) are
    also filled, for reporting parity with every other strategy's partial
    exits, even though this engine's own sync path only consults pattern_data."""
    row = _row(conn, 396)
    if row is None:
        print("SKIP CONI partial exit: row 396 not found in this trades table")
        return
    existing_pd = json.loads(row["pattern_data"] or "{}")
    if int(existing_pd.get("closed_qty") or 0) >= 25 or row["partial_exit_shares"] == 25:
        print("SKIP CONI partial exit: already booked (closed_qty >= 25)")
        return
    shares, exit_price, entry_basis = 25, 23.48, 22.882231
    partial_pnl = round(shares * (exit_price - entry_basis), 2)
    closed_notional = round(shares * exit_price, 2)
    partial_exited_at = _to_iso("2026-09-29T17:38:15Z")

    changes.append(
        f"UPDATE trades id=396 CONI: partial exit {shares}sh @ {exit_price} ({partial_exited_at}), "
        f"partial_exit_pnl {row['partial_exit_pnl']}->{partial_pnl}, pattern_data.closed_qty "
        f"{existing_pd.get('closed_qty')}->{shares}, closed_notional -> {closed_notional} "
        f"(shares stays 112, order_status stays 'filled'; open_qty 112-25=87 = broker holding)"
    )
    if dry_run:
        return

    now = _now_iso()
    merged_pd = dict(existing_pd)
    merged_pd.update({"closed_qty": shares, "closed_notional": closed_notional})
    conn.execute(
        "UPDATE trades SET partial_exit_shares=?, partial_exit_price=?, partial_exit_pnl=?, "
        "partial_exit_reason='target', partial_exited_at=?, pattern_data=?, updated_at=? WHERE id=396",
        (shares, exit_price, partial_pnl, partial_exited_at, json.dumps(merged_pd), now),
    )


def apply_corrections(conn: sqlite3.Connection, dry_run: bool) -> List[str]:
    """Runs every 2026-09-29 incident correction, in order, idempotently. Each
    correction checks the row's CURRENT state first and is a no-op if already
    applied. Returns the change descriptions actually applied (or that WOULD be
    applied, under dry_run). Commits once at the end when not a dry run."""
    changes: List[str] = []
    fix_cdna_381(conn, dry_run, changes)
    insert_cdna_over_exit_short(conn, dry_run, changes)
    astn_ts = _astn_entry_ts_from_broker() or "2026-09-29T13:52:00Z"
    insert_ops_flatten(conn, dry_run, changes, symbol="PRIM", entry_price=77.7386,
                        entry_ts="2026-09-29T13:47:20Z", exit_price=76.0351,
                        exit_ts="2026-09-29T14:12:30Z", shares=25, pnl=-42.59)
    insert_ops_flatten(conn, dry_run, changes, symbol="ASTN", entry_price=17.23,
                        entry_ts=astn_ts, exit_price=17.3187,
                        exit_ts="2026-09-29T14:12:35Z", shares=83, pnl=7.36)
    insert_ops_flatten(conn, dry_run, changes, symbol="WRBY", entry_price=26.1905,
                        entry_ts="2026-09-29T13:40:30Z", exit_price=26.3759,
                        exit_ts="2026-09-29T14:10:50Z", shares=76, pnl=14.09)
    close_ttan_393(conn, dry_run, changes)
    merge_coni_triple(conn, dry_run, changes)
    book_coni_partial_exit(conn, dry_run, changes)
    if not dry_run:
        conn.commit()
    return changes


def main() -> None:
    """CLI entry point: backup (unless --dry-run), open the DB, apply every
    correction, print what changed."""
    parser = argparse.ArgumentParser(description="Fix the 9/25-9/29 HOD-break trades-ledger incident.")
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
