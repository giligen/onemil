"""One-shot orphan recovery for FABC (2026-06-09 BRANCH_LAST_RESORT).

Context: at 14:00:14 UTC today, ORB tried to close FABC via market order
after a 10s limit-sell timeout. Close failed with:
  "insufficient qty available: held_for_orders=10627, available=0"

The bracket SL leg cancel hadn't propagated. BRANCH_LAST_RESORT wrote
exit_pending_verification (L1 fix worked correctly), but no emergency
stop was placed. The position is naked on broker. ORB's reconciler runs
only at startup (L8), so the row sits until next morning unless we
intervene.

This script mirrors scripts/close_orphans_smu_qbtz.py: snapshot →
close_position → poll fill → exit summary. DB reconciliation is the
next script (scripts/reconcile_orphan_fabc_db.py) so the risky live-
order action stays isolated and reviewable.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv('/home/ec2-user/onemil/.env')

from data_sources.alpaca_client import AlpacaClient
from config import get_config


def _close_one(client: AlpacaClient, symbol: str) -> dict:
    print(f"\n=== ORB: closing {symbol} ===", flush=True)
    try:
        positions = client.get_open_positions()
    except Exception as e:
        print(f"  ERROR: get_open_positions() failed: {e}", flush=True)
        return {'symbol': symbol, 'status': 'snapshot_failed', 'error': str(e)}

    pos = next((p for p in positions if p['symbol'] == symbol), None)
    if pos is None:
        print(f"  {symbol} NOT held — nothing to close.", flush=True)
        return {'symbol': symbol, 'status': 'not_held'}

    qty_before = int(pos['qty'])
    avg_entry = float(pos['avg_entry_price'])
    upl_before = float(pos.get('unrealized_pl', 0) or 0)
    print(f"  Pre-close: {qty_before}sh @ ${avg_entry:.4f} avg "
          f"(UPL ${upl_before:+,.0f})", flush=True)

    # Alpaca's close_position will internally cancel any open orders on
    # the symbol before submitting the market sell, so we don't have to
    # cancel the (already-cancelled) bracket legs again.
    try:
        result = client.close_position(symbol)
    except Exception as e:
        print(f"  ERROR: close_position failed: {e}", flush=True)
        return {'symbol': symbol, 'status': 'close_failed', 'error': str(e),
                'qty_before': qty_before, 'avg_entry': avg_entry}

    order_id = (result or {}).get('id', '')
    print(f"  close order submitted: id={order_id}", flush=True)

    fill_price = None
    fill_qty = 0
    deadline = time.time() + 30.0
    last_status = ''
    while time.time() < deadline:
        try:
            o = client.get_order(order_id)
            status = (o.get('status') or '').lower()
            filled_qty = int(o.get('filled_qty') or 0)
            avg_fill = o.get('filled_avg_price')
            if avg_fill is not None and filled_qty > 0:
                fill_price = float(avg_fill)
                fill_qty = filled_qty
            if status != last_status:
                print(f"  status={status} filled={filled_qty}/{qty_before}",
                      flush=True)
                last_status = status
            if status == 'filled' and filled_qty >= qty_before:
                break
            if status in ('canceled', 'rejected', 'expired'):
                print(f"  Close ended terminal-non-filled: {status}",
                      flush=True)
                break
        except Exception as e:
            print(f"  poll error: {e} (retrying)", flush=True)
        time.sleep(0.5)

    realized_pnl = (
        (fill_price - avg_entry) * fill_qty if fill_price is not None else None
    )
    if fill_price is not None:
        print(f"  RESULT: filled {fill_qty}/{qty_before} sh at "
              f"${fill_price:.4f}, realized ${realized_pnl:+,.2f}", flush=True)
    else:
        print(f"  RESULT: fill not yet confirmed — check broker", flush=True)

    try:
        post = client.get_open_positions()
        still = next((p for p in post if p['symbol'] == symbol), None)
        if still is None:
            print(f"  Post-close verify: {symbol} no longer held ✓", flush=True)
        else:
            print(f"  Post-close verify: {symbol} STILL HELD "
                  f"({int(still['qty'])}sh)", flush=True)
    except Exception as e:
        print(f"  Post-close verify failed: {e}", flush=True)

    return {
        'symbol': symbol, 'status': 'closed', 'order_id': order_id,
        'qty_before': qty_before, 'avg_entry': avg_entry,
        'fill_qty': fill_qty, 'fill_price': fill_price,
        'realized_pnl': realized_pnl,
        'closed_at': datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    cfg = get_config()
    print("Connecting to ORB Alpaca account...", flush=True)
    orb = AlpacaClient(cfg.alpaca_orb_api_key, cfg.alpaca_orb_api_secret,
                       paper=cfg.alpaca_orb_paper)
    info = orb.get_account_info() or {}
    print(f"  ORB equity=${info.get('equity', 0):,.0f}, "
          f"buying_power=${info.get('buying_power', 0):,.0f} "
          f"({'paper' if orb.is_paper else 'LIVE'})", flush=True)

    result = _close_one(orb, 'FABC')
    print("\n=== Summary (JSON for DB reconcile) ===", flush=True)
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
