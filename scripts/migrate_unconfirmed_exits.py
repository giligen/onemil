"""Migrate historical 'stop_loss_unconfirmed' rows into the new
'exit_pending_verification' state so the orphan reconciler picks them up.

Without this, any pre-existing poisoned row (stop_loss_unconfirmed with a
fake exit_price) stays invisible to the new reconciler because
get_open_trades filters them out via `exit_price IS NULL`.

What this script does (idempotent, dry-run by default):

For every trades row with `exit_reason = 'stop_loss_unconfirmed'`:
  - Verify whether the broker still holds the position. We do this by
    querying the appropriate Alpaca account based on the row's `strategy`
    column (orb → ORB account, others → MAIN account).
  - If the broker still holds it AND the broker's avg_entry_price
    matches the DB's fill_price within tight tolerance:
      → reset the row: exit_price = NULL, exit_reason = 'stop_loss_unconfirmed'
        (preserved as a forensic marker), exited_at = NULL, pnl = NULL,
        pnl_pct = NULL, order_status = 'exit_pending_verification'.
      → The reconciler will then attempt to close it on the next sync.
  - If the broker does NOT hold it, leave the row alone. The position
    was likely closed by some other path (manual, broker eventual
    consistency, restart-recovery code that already handled it).

Run modes:
  python3 scripts/migrate_unconfirmed_exits.py                 # dry-run
  python3 scripts/migrate_unconfirmed_exits.py --apply         # apply changes

Idempotent — re-running with --apply on already-migrated rows is a no-op
because they no longer have exit_reason = 'stop_loss_unconfirmed' (well,
they DO, but order_status is already 'exit_pending_verification' which the
script detects and skips).

Note: we just manually fixed SMU + QBTZ via reconcile_orphans_db.py so
those are now exit_reason='orphan_recovered_force_close' and will be
skipped. This script is primarily a forward-looking safety net.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv('/home/ec2-user/onemil/.env')

from data_sources.alpaca_client import AlpacaClient
from config import get_config

DB_PATH = '/home/ec2-user/onemil/data/trades.db'
PRICE_EPS_PCT = 0.0005  # 5 bps avg-entry match tolerance


def _alpaca_for_strategy(strategy: str, cfg) -> AlpacaClient:
    """Route per the existing _strategy_uses_separate_account convention:
    'orb' uses its own keys when configured, everything else uses MAIN."""
    if strategy == 'orb' and cfg.alpaca_orb_api_key:
        return AlpacaClient(
            cfg.alpaca_orb_api_key, cfg.alpaca_orb_api_secret,
            paper=cfg.alpaca_orb_paper,
        )
    return AlpacaClient(
        cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Migrate stop_loss_unconfirmed rows to "
                    "exit_pending_verification for orphan-reconciler "
                    "pickup."
    )
    p.add_argument('--apply', action='store_true',
                   help="Apply changes (default: dry-run)")
    args = p.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cfg = get_config()

    # Pre-build broker-position snapshots per account so we don't requery.
    snapshots = {}
    for label in ('main', 'orb'):
        try:
            cli = _alpaca_for_strategy('orb' if label == 'orb' else 'macd_wave', cfg)
            snapshots[label] = {p['symbol']: p for p in cli.get_open_positions()}
            print(f"  {label}: {len(snapshots[label])} broker positions")
        except Exception as e:
            print(f"  {label}: failed to snapshot — {e}")
            snapshots[label] = {}

    cur = conn.execute("""
        SELECT id, strategy, symbol, trade_date, fill_price, filled_qty,
               exit_price, exit_reason, order_status
        FROM trades
        WHERE exit_reason = 'stop_loss_unconfirmed'
        ORDER BY trade_date
    """)
    rows = cur.fetchall()
    print(f"\nFound {len(rows)} rows with exit_reason='stop_loss_unconfirmed'")

    migrated = 0
    skipped_already = 0
    skipped_not_held = 0
    skipped_mismatch = 0
    for r in rows:
        if r['order_status'] == 'exit_pending_verification':
            skipped_already += 1
            continue
        acct = 'orb' if r['strategy'] == 'orb' else 'main'
        broker_pos = snapshots.get(acct, {}).get(r['symbol'])
        if broker_pos is None:
            skipped_not_held += 1
            continue
        db_fill = r['fill_price'] or 0.0
        if db_fill <= 0:
            skipped_mismatch += 1
            continue
        eps = max(0.005, db_fill * PRICE_EPS_PCT)
        if abs(broker_pos['avg_entry_price'] - db_fill) > eps:
            skipped_mismatch += 1
            print(f"  SKIP {r['symbol']} (id={r['id']}): broker avg "
                  f"${broker_pos['avg_entry_price']:.4f} != DB fill "
                  f"${db_fill:.4f}")
            continue
        if broker_pos['qty'] > (r['filled_qty'] or 0):
            skipped_mismatch += 1
            print(f"  SKIP {r['symbol']} (id={r['id']}): broker qty "
                  f"{broker_pos['qty']} > DB filled_qty {r['filled_qty']}")
            continue

        # All checks pass — this is OUR orphan to reconcile.
        print(f"  MIGRATE {r['symbol']} (id={r['id']}, strategy={r['strategy']}): "
              f"broker has {broker_pos['qty']}sh @ ${broker_pos['avg_entry_price']:.4f}, "
              f"matches DB ({r['filled_qty']}sh @ ${db_fill:.4f})")
        if args.apply:
            conn.execute("""
                UPDATE trades SET
                  exit_price = NULL,
                  exited_at = NULL,
                  pnl = NULL,
                  pnl_pct = NULL,
                  order_status = 'exit_pending_verification',
                  updated_at = ?
                WHERE id = ?
            """, (datetime.now(timezone.utc).isoformat(), r['id']))
            migrated += 1

    if args.apply:
        conn.commit()
    print(f"\nSummary:")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped (already migrated): {skipped_already}")
    print(f"  Skipped (broker not holding): {skipped_not_held}")
    print(f"  Skipped (mismatch — foreign): {skipped_mismatch}")
    print(f"  Mode: {'APPLIED' if args.apply else 'DRY-RUN (use --apply)'}")
    conn.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
