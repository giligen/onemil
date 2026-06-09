"""Update FABC DB row to reflect the manual orphan recovery close.

Pre-state (after close_orphan_fabc.py ran):
  trade_id=417 row: order_status='exit_pending_verification',
  exit_reason='stop_loss_unconfirmed', exit_price=NULL, pnl=NULL.
  Broker now flat.

Post-state (this script writes):
  exit_price = 3.60 (actual fill avg)
  exit_reason = 'orphan_recovered_force_close'
  pnl, pnl_pct computed
  order_status = 'closed'
  partial_exited_at = manual-close timestamp (preserves original
                       exited_at NULL since the original exit was never
                       confirmed)

Mirrors scripts/reconcile_orphans_db.py — same idempotency guards.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timezone

DB_PATH = '/home/ec2-user/onemil/data/trades.db'

RECOVERY = {
    'strategy': 'orb', 'symbol': 'FABC', 'trade_date': '2026-06-09',
    'fill_qty': 10627, 'fill_price': 3.60,
    'order_id': '5c002f40-7382-40a6-953f-3577739baff3',
    'avg_entry': 4.30,
    'closed_at_utc': '2026-06-09T16:19:54+00:00',
}


def main() -> int:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        r = RECOVERY
        cur = conn.execute(
            "SELECT id, exit_price, exit_reason, pnl, partial_exited_at "
            "FROM trades WHERE strategy=? AND symbol=? AND trade_date=? "
            "ORDER BY created_at DESC LIMIT 1",
            (r['strategy'], r['symbol'], r['trade_date']),
        )
        row = cur.fetchone()
        if row is None:
            print(f"  no DB row for {r['strategy']}/{r['symbol']}/"
                  f"{r['trade_date']} — skipping")
            return 1
        if row['exit_reason'] == 'orphan_recovered_force_close':
            print(f"  already reconciled (pnl=${row['pnl']:+,.2f}) — skip")
            return 0
        new_pnl = round((r['fill_price'] - r['avg_entry']) * r['fill_qty'], 4)
        new_pnl_pct = round(
            (r['fill_price'] - r['avg_entry']) / r['avg_entry'] * 100, 4
        )
        print(
            f"  {r['symbol']} (trade_id={row['id']}): "
            f"exit_price NULL -> ${r['fill_price']:.4f}, "
            f"exit_reason '{row['exit_reason']}' -> "
            f"'orphan_recovered_force_close', "
            f"pnl NULL -> ${new_pnl:+,.2f}"
        )
        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE trades SET "
            "  exit_price = ?, "
            "  exit_reason = 'orphan_recovered_force_close', "
            "  pnl = ?, "
            "  pnl_pct = ?, "
            "  partial_exited_at = ?, "
            "  order_status = 'closed', "
            "  updated_at = ? "
            "WHERE id = ?",
            (r['fill_price'], new_pnl, new_pnl_pct,
             r['closed_at_utc'], now_iso, row['id']),
        )
        conn.commit()
        print("\nDone.")
    finally:
        conn.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
