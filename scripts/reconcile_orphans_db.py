"""Update SMU + QBTZ DB rows to reflect the actual orphan-recovery fill.

Pre-state (`scripts/close_orphans_smu_qbtz.py` already ran):
  Both trades had bogus exit_price + 'stop_loss_unconfirmed' that never
  actually filled at the broker. The positions stayed live for days
  before today's manual force-close.

Post-state (what this script writes):
  Update the existing row in-place with the REAL fill price + qty and
  a distinct exit_reason 'orphan_recovered_force_close' so analytics can
  separate this incident from normal exits. We deliberately leave the
  original `exited_at` value alone so the timeline shows when the bug
  triggered, and add a second timestamp `partial_exited_at` to mark when
  the orphan was actually flattened (re-using an existing column rather
  than adding schema).

  Also flips order_status to 'closed' explicitly (it was already 'closed'
  for these two, but be explicit for future generalization).

Safe to re-run — uses idempotent UPDATE with exit_reason guards.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = '/home/ec2-user/onemil/data/trades.db'

# Captured from scripts/close_orphans_smu_qbtz.py output 2026-06-05.
RECOVERIES = [
    {
        'strategy': 'macd_wave', 'symbol': 'SMU', 'trade_date': '2026-05-26',
        'fill_qty': 5976, 'fill_price': 11.82,
        'order_id': '7a1045c1-1cec-4317-84f9-95a2b56fc5ba',
        'avg_entry': 14.672199,
        'closed_at_utc': '2026-06-05T13:51:00+00:00',
    },
    {
        'strategy': 'orb', 'symbol': 'QBTZ', 'trade_date': '2026-06-01',
        'fill_qty': 5657, 'fill_price': 4.38,
        'order_id': 'e7286d8a-e9bb-4d93-bdd7-2b87f28059ba',
        'avg_entry': 3.75,
        'closed_at_utc': '2026-06-05T13:51:02+00:00',
    },
]


def main() -> int:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        for r in RECOVERIES:
            cur = conn.execute(
                "SELECT id, exit_price, exit_reason, pnl, partial_exited_at "
                "FROM trades WHERE strategy=? AND symbol=? AND trade_date=? "
                "ORDER BY created_at DESC LIMIT 1",
                (r['strategy'], r['symbol'], r['trade_date']),
            )
            row = cur.fetchone()
            if row is None:
                print(f"  {r['symbol']}: no DB row for "
                      f"{r['strategy']} on {r['trade_date']} — skipping")
                continue

            new_pnl = round(
                (r['fill_price'] - r['avg_entry']) * r['fill_qty'], 4
            )
            new_pnl_pct = round(
                (r['fill_price'] - r['avg_entry']) / r['avg_entry'] * 100, 4
            )

            # Idempotency: skip if already recovered.
            if row['exit_reason'] == 'orphan_recovered_force_close':
                print(f"  {r['symbol']}: already reconciled "
                      f"(pnl=${row['pnl']:+,.2f}) — skipping")
                continue

            print(
                f"  {r['symbol']} (trade_id={row['id']}): "
                f"updating exit_price ${row['exit_price']:.4f} -> "
                f"${r['fill_price']:.4f}, exit_reason "
                f"'{row['exit_reason']}' -> 'orphan_recovered_force_close', "
                f"pnl ${row['pnl']:+,.2f} -> ${new_pnl:+,.2f}"
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
                (r['fill_price'], new_pnl, new_pnl_pct, r['closed_at_utc'],
                 now_iso, row['id']),
            )
        conn.commit()
        print("\nDone.")
    finally:
        conn.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
