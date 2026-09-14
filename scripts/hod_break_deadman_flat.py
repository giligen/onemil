#!/usr/bin/env python3
"""HOD-break dead-man flat (2026-09-14): if the trader process is down at the close, its broker bracket legs
still guard the stop and the target, but nothing enforces the 15:55 ET flat. This cron (19:57 UTC weekdays)
sells OUR shares of every hod_break trade the DB still shows open, with a marketable limit off the bid,
and records the exit as exit_pending_verification for the next boot's reconciliation.

Never touches any other strategy's or the owner's positions (sells exactly the DB qty per open row).
Runs only when `onemil-trader` is NOT active (the engine handles the flat itself when it is running).
Usage: python3 scripts/hod_break_deadman_flat.py [--force]   (--force = act even if the service is active)
"""
import os
import subprocess
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); os.chdir(ROOT)

from config import Config                                  # noqa: E402
from data_sources.alpaca_client import AlpacaClient        # noqa: E402
from persistence.database import Database                  # noqa: E402

STRATEGY = 'hod_break'
OPEN = ('filled', 'partially_filled', 'exit_pending_verification')


def main() -> int:
    force = '--force' in sys.argv
    active = subprocess.run(['systemctl', 'is-active', 'onemil-trader'], capture_output=True, text=True).stdout.strip() == 'active'
    if active and not force:
        print('trader active — the engine owns the flat; nothing to do'); return 0
    cfg = Config(); db = Database(); today = datetime.now(timezone.utc).astimezone(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')
    rows = [r for r in db.get_open_trades(today, strategy=STRATEGY) if r.get('order_status') in OPEN and int(r.get('shares') or 0) > 0]
    if not rows:
        print(f'no open {STRATEGY} rows for {today}'); return 0
    alpaca = AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)
    held = {p.get('symbol'): int(float(p.get('qty') or 0)) for p in (alpaca.get_open_positions() or [])}
    n = 0
    for r in rows:
        sym, qty = r['symbol'], int(r['shares'])
        if held.get(sym, 0) <= 0:
            print(f'{sym}: DB open but nothing held at the broker — marking exit_pending_verification'); db.update_trade(r['id'], {'order_status': 'exit_pending_verification'}); continue
        qty = min(qty, held[sym])
        try:
            q = alpaca.get_latest_quote(sym); ref = float(q.get('bid_price') or 0) or float(r.get('fill_price') or r.get('entry_price') or 0)
            od = alpaca.submit_limit_sell_order(sym, qty, round(ref * 0.99, 2))
            db.update_trade(r['id'], {'order_status': 'exit_pending_verification', 'exit_reason': 'deadman_flat'})
            print(f'{sym}: DEAD-MAN FLAT x{qty} limit {round(ref * 0.99, 2)} order {od.get("id")}'); n += 1
        except Exception as e:
            print(f'{sym}: dead-man flat FAILED: {e}')
    try:
        subprocess.run([sys.executable, 'scripts/send_telegram_alert.py', f'[HOD DEAD-MAN] trader inactive at the close: {n} of {len(rows)} open hod_break position(s) flattened, see logs/hod_deadman.log'], check=False)
    except Exception:
        pass
    return 0 if n == len(rows) else 1


if __name__ == '__main__':
    sys.exit(main())
