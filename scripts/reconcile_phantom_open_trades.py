"""Reconcile DB phantom-open trades against Alpaca actual state.

A "phantom-open" row is one where DB has fill_price set but exit_price NULL,
yet Alpaca shows no open position for that symbol. This happens when:
  - Force-close fired and closed at Alpaca, but the post-FC DB-update path
    (TradingStream watcher) missed the sell-fill event.
  - Or the FC sweep closed an orphan but never reconciled DB.

Surfaced 2026-05-07 after audit of ASTX 5/6: Alpaca had 0 positions but
DB showed 5 phantom-open trades back to 4/1.

Usage:
    python3 scripts/reconcile_phantom_open_trades.py            # dry-run
    python3 scripts/reconcile_phantom_open_trades.py --apply    # write to DB

Idempotent: rows with exit_price set are skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(str(ROOT / '.env'))

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus


def reconcile(apply: bool = False) -> int:
    db = Database(db_path=str(ROOT / 'data' / 'trades.db'))
    # Each strategy has its own Alpaca account. ORB has dedicated keys.
    main_client = AlpacaClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_API_SECRET'))
    orb_key = os.getenv('ALPACA_ORB_API_KEY')
    orb_secret = os.getenv('ALPACA_ORB_API_SECRET')
    orb_client = (AlpacaClient(orb_key, orb_secret) if (orb_key and orb_secret)
                  else None)
    clients_by_strategy = {
        'bull_flag': main_client,
        'macd_wave': main_client,
        'orb': orb_client or main_client,
    }

    # Get current positions per account (source of truth for "actually open")
    open_by_strategy: dict = {}
    for strat, c in clients_by_strategy.items():
        if c is None:
            open_by_strategy[strat] = set()
            continue
        try:
            open_by_strategy[strat] = {p.symbol for p in c.trading_client.get_all_positions()}
        except Exception as e:
            print(f"ERROR: positions fetch for {strat} failed: {e}")
            open_by_strategy[strat] = set()
    print(f"Alpaca open per strategy:")
    for s, syms in open_by_strategy.items():
        print(f"  {s}: {sorted(syms) if syms else 'NONE'}")

    # Find phantom-open DB rows
    import sqlite3
    conn = sqlite3.connect(str(ROOT / 'data' / 'trades.db'))
    rows = conn.execute("""
        SELECT id, strategy, symbol, trade_date, fill_price,
               COALESCE(filled_qty, shares) as qty, filled_at
          FROM trades
         WHERE fill_price IS NOT NULL AND exit_price IS NULL
         ORDER BY trade_date, id
    """).fetchall()
    conn.close()

    print(f"\nDB phantom-open candidates: {len(rows)}")
    fixed = []
    skipped = []
    for tid, strat, sym, td, fp, qty, filled_at in rows:
        client = clients_by_strategy.get(strat, main_client)
        alp_positions_this_strat = open_by_strategy.get(strat, set())
        if sym in alp_positions_this_strat:
            print(f"  [skip] {strat} {sym} ({td}) — actually open at Alpaca")
            skipped.append((sym, 'still-open'))
            continue

        # Find the most-recent filled sell order for this symbol on the
        # account that owns this strategy.
        try:
            orders = client.trading_client.get_orders(
                GetOrdersRequest(
                    status=QueryOrderStatus.CLOSED,
                    symbols=[sym], limit=20,
                )
            ) or []
        except Exception as e:
            print(f"  [error] {strat} {sym}: get_orders failed: {e}")
            skipped.append((sym, f'orders-fetch-failed: {e}'))
            continue

        sell = None
        for o in orders:
            if o.side.value != 'sell':
                continue
            if o.status.value != 'filled':
                continue
            if not o.filled_avg_price:
                continue
            # Prefer sell orders submitted/filled AFTER the buy fill
            try:
                if filled_at and o.filled_at:
                    from dateutil.parser import isoparse
                    buy_t = isoparse(filled_at) if isinstance(filled_at, str) else filled_at
                    sell_t = o.filled_at
                    if sell_t.tzinfo is None:
                        sell_t = sell_t.replace(tzinfo=timezone.utc)
                    if buy_t.tzinfo is None:
                        buy_t = buy_t.replace(tzinfo=timezone.utc)
                    if sell_t < buy_t:
                        continue
            except Exception:
                pass
            # Take the FIRST sell after buy (chronological — most recent buy may have a re-close)
            if sell is None or (o.filled_at and sell.filled_at and o.filled_at < sell.filled_at):
                sell = o

        if sell is None:
            # Aged-out order history (Alpaca paper retains ~7 days). Position
            # is genuinely closed (verified above) but we can't recover the
            # exit price. Mark as reconcile_unknown with exit=fill (zero P&L)
            # to clear the phantom-open state. The historical record loses
            # accuracy but the row stops polluting open-position queries.
            from datetime import datetime as _dt
            dt_obj = _dt.strptime(td, '%Y-%m-%d') if isinstance(td, str) else td
            age_days = (_dt.now().date() - dt_obj.date() if hasattr(dt_obj, 'date') else _dt.now().date() - dt_obj).days
            print(f"  [aged-out] {strat} {sym} ({td}, {age_days}d old) — position gone "
                  f"but no sell order in Alpaca history; would mark exit=fill (P&L=$0)")
            if apply:
                db.update_trade(tid, {
                    'exit_price': float(fp),
                    'exit_reason': 'reconcile_unknown',
                    'exited_at': datetime.now(timezone.utc),
                    'pnl': 0.0,
                    'pnl_pct': 0.0,
                })
                fixed.append((sym, 0.0))
            else:
                # count for would-fix in dry-run
                fixed.append((sym, 0.0))
            continue

        exit_price = float(sell.filled_avg_price)
        entry_price = float(fp)
        shares = int(qty or 0)
        pnl = (exit_price - entry_price) * shares
        pnl_pct = (exit_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
        exited_at = sell.filled_at

        delta = exit_price - entry_price
        sign = '+' if delta >= 0 else ''
        print(f"  [fix]  {strat} {sym} ({td}): entry=${entry_price:.2f} → "
              f"exit=${exit_price:.2f} ({sign}{delta:.2f}) "
              f"qty={shares} pnl=${pnl:+,.2f} order={str(sell.id)[:8]}")

        if apply:
            db.update_trade(tid, {
                'exit_price': exit_price,
                'exit_reason': 'force_close_reconcile',
                'exited_at': exited_at,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
            })
            fixed.append((sym, pnl))

    print(f"\nSummary: {'APPLIED' if apply else 'DRY-RUN'} — would-fix {len(fixed) if apply else len(rows)-len(skipped)} rows, skipped {len(skipped)}")
    if apply and fixed:
        total = sum(p for _, p in fixed)
        print(f"  Net P&L impact (added to historical record): ${total:+,.2f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='Write fixes to DB (otherwise dry-run)')
    args = ap.parse_args()
    sys.exit(reconcile(apply=args.apply))


if __name__ == '__main__':
    main()
