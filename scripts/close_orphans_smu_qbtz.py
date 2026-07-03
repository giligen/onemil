"""One-shot orphan recovery: SMU (main acct, macd_wave) + QBTZ (orb acct).

These rows are sitting OPEN on the broker despite being marked
'stop_loss_unconfirmed' / 'closed' in the trades DB:
  SMU: 5976sh @ $14.67 (orphan since 2026-05-26 — macd_wave)
  QBTZ: 5657sh @ $3.75  (orphan since 2026-06-01 — orb)

Steps per symbol:
  1. Snapshot the broker position (sanity check qty/avg).
  2. Submit a market close via `alpaca.close_position(symbol)`.
     (Alpaca's close_position internally cancels open orders for the symbol
     before the market sell — no extra cancel pass needed.)
  3. Poll briefly for fill, report avg fill price + realized P&L.
  4. Verify the position is gone from the account.

DB reconciliation is a separate pass — see
scripts/reconcile_orphans_db.py — so this script stays focused on the
one risky action (live order submission) and the DB pass is reviewable
independently.

Run from /home/ec2-user/onemil:
    python3 scripts/close_orphans_smu_qbtz.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from data_sources.alpaca_client import AlpacaClient
from config import get_config


def _close_one(client: AlpacaClient, label: str, symbol: str) -> dict:
    print(f"\n=== {label}: closing {symbol} ===", flush=True)
    # 1. Snapshot
    try:
        positions = client.get_open_positions()
    except Exception as e:
        print(f"  ERROR: get_open_positions() failed: {e}", flush=True)
        return {'symbol': symbol, 'status': 'snapshot_failed', 'error': str(e)}

    pos = next((p for p in positions if p['symbol'] == symbol), None)
    if pos is None:
        print(f"  {symbol} NOT held on this account — nothing to close.",
              flush=True)
        return {'symbol': symbol, 'status': 'not_held'}

    qty_before = int(pos['qty'])
    avg_entry = float(pos['avg_entry_price'])
    upl_before = float(pos.get('unrealized_pl', 0) or 0)
    print(f"  Pre-close: {qty_before}sh @ ${avg_entry:.4f} avg "
          f"(UPL ${upl_before:+,.0f})", flush=True)

    # 2. Market close (close_position internally cancels open orders for the
    #    symbol then submits the offsetting market order).
    try:
        result = client.close_position(symbol)
    except Exception as e:
        print(f"  ERROR: close_position failed: {e}", flush=True)
        return {'symbol': symbol, 'status': 'close_failed', 'error': str(e),
                'qty_before': qty_before, 'avg_entry': avg_entry}

    order_id = (result or {}).get('id', '')
    print(f"  close order submitted: id={order_id}", flush=True)

    # 3. Poll for fill (typically <2s for a market order)
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
            if status in ('filled',) and filled_qty >= qty_before:
                break
            if status in ('canceled', 'rejected', 'expired'):
                print(f"  Close ended in terminal-non-filled state: {status}",
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
        print(f"  RESULT: fill not yet confirmed after 30s — check broker",
              flush=True)

    # 4. Post-close verify
    try:
        post = client.get_open_positions()
        still = next((p for p in post if p['symbol'] == symbol), None)
        if still is None:
            print(f"  Post-close verify: {symbol} no longer held ✓", flush=True)
        else:
            print(f"  Post-close verify: {symbol} STILL HELD "
                  f"({int(still['qty'])}sh) — partial fill or rejected close",
                  flush=True)
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

    # MAIN account (macd_wave + bull_flag) — holds SMU
    print("Connecting to MAIN Alpaca account (macd_wave/bull_flag)...",
          flush=True)
    main_client = AlpacaClient(
        cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper,
    )
    main_info = main_client.get_account_info() or {}
    print(f"  MAIN equity=${main_info.get('equity', 0):,.0f}, "
          f"buying_power=${main_info.get('buying_power', 0):,.0f} "
          f"({'paper' if main_client.is_paper else 'LIVE'})", flush=True)

    # ORB account — holds QBTZ
    print("\nConnecting to ORB Alpaca account...", flush=True)
    orb_client = AlpacaClient(
        cfg.alpaca_orb_api_key, cfg.alpaca_orb_api_secret,
        paper=cfg.alpaca_orb_paper,
    )
    orb_info = orb_client.get_account_info() or {}
    print(f"  ORB equity=${orb_info.get('equity', 0):,.0f}, "
          f"buying_power=${orb_info.get('buying_power', 0):,.0f} "
          f"({'paper' if orb_client.is_paper else 'LIVE'})", flush=True)

    results = []
    results.append(_close_one(main_client, 'MAIN/macd_wave', 'SMU'))
    results.append(_close_one(orb_client, 'ORB', 'QBTZ'))

    print("\n=== Summary (JSON for downstream DB pass) ===", flush=True)
    print(json.dumps(results, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
