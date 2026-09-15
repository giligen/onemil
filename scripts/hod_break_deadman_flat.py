#!/usr/bin/env python3
"""HOD-break dead-man flat (2026-09-14; rewritten 9/15 after review E): if the trader process is down at the close,
its broker bracket legs still guard the stop and the target, but nothing enforces the flat. This cron runs every
5 minutes from 19:30 UTC; the SCRIPT decides from Alpaca's calendar whether it is the flat window (close − 5 min,
so early closes and the November DST shift are handled), and only acts when `onemil-trader` is NOT active.

For every hod_break row the DB still shows open it reads the exit orders named in pattern_data (tp_leg_id,
sl_leg_id, close_order_id): shares those orders already sold are BOOKED (exit price/pnl written), never sold
again; the legs are canceled and re-read until terminal; only the remainder is sold, with a marketable limit
carrying our own client_order_id. It never sells more than the DB says we hold and never touches any other
strategy's or the owner's shares. Usage: python3 scripts/hod_break_deadman_flat.py [--force] [--now HH:MM-ET]
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); os.chdir(ROOT)

from config import Config                                  # noqa: E402
from data_sources.alpaca_client import AlpacaClient        # noqa: E402
from persistence.database import Database                  # noqa: E402

STRATEGY = 'hod_break'
OPEN = ('filled', 'partially_filled', 'exit_pending_verification')
TERMINAL = ('canceled', 'cancelled', 'expired', 'rejected', 'done_for_day', 'suspended', 'filled', 'replaced')
ET = ZoneInfo('America/New_York')


def flat_window(alpaca, now_et) -> tuple:
    """(in_window, close_minute): the flat window is [close − 5 min, close + 30 min) on a trading day."""
    d = now_et.date(); close_m = 960
    try:
        cal = alpaca.get_market_calendar(d, d) or []
        row = next((c for c in cal if str(c.get('date'))[:10] == d.isoformat()), None)
        if row is None: return False, None                                   # holiday
        close = row.get('close'); close_m = close.hour * 60 + close.minute if isinstance(close, datetime) else int(str(close)[:2]) * 60 + int(str(close)[3:5])
    except Exception as e:
        print(f'calendar unavailable ({e}) — regular 16:00 close assumed')
    m = now_et.hour * 60 + now_et.minute
    return close_m - 5 <= m < close_m + 30, close_m


def read_order(alpaca, oid):
    """Order dict or {} — follows a replaced order to its successor."""
    for _ in range(3):
        try: st = alpaca.get_order(oid) or {}
        except Exception as e: print(f'  get_order {oid} failed: {e}'); return {}
        if str(st.get('status', '')).lower() == 'replaced' and st.get('replaced_by'): oid = st['replaced_by']; continue
        return dict(st, id=oid)
    return {}


def main() -> int:
    force = '--force' in sys.argv
    active = subprocess.run(['systemctl', 'is-active', 'onemil-trader'], capture_output=True, text=True).stdout.strip() == 'active'
    if active and not force:
        print('trader active — the engine owns the flat; nothing to do'); return 0
    cfg = Config(); db = Database(); alpaca = AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)
    now_et = datetime.now(timezone.utc).astimezone(ET)
    if '--now' in sys.argv:
        hh, mm = sys.argv[sys.argv.index('--now') + 1].split(':'); now_et = now_et.replace(hour=int(hh), minute=int(mm))
    in_window, close_m = flat_window(alpaca, now_et)
    if not in_window and not force:
        print(f'{now_et:%H:%M} ET is outside the flat window (close {close_m}) — nothing to do'); return 0
    today = now_et.strftime('%Y-%m-%d')
    rows = [r for r in db.get_open_trades(today, strategy=STRATEGY) if r.get('order_status') in OPEN and int(r.get('shares') or 0) > 0]
    if not rows:
        print(f'no open {STRATEGY} rows for {today}'); return 0
    n_ok = 0
    for r in rows:
        sym, shares = r['symbol'], int(r['shares']); entry = float(r.get('fill_price') or r.get('entry_price') or 0)
        try: pd_ = json.loads(r.get('pattern_data') or '{}')
        except Exception: pd_ = {}
        if pd_.get('deadman_close_order_id'):                                 # a previous pass already sold — read it, never sell twice
            pd_['close_order_id'] = pd_['deadman_close_order_id']
        legs = [(k, pd_.get(k)) for k in ('close_order_id', 'tp_leg_id', 'sl_leg_id') if pd_.get(k)]
        for _, oid in legs:                                                    # cancel what is still working
            try: alpaca.cancel_order(oid)
            except Exception as e: print(f'{sym}: cancel {oid} failed ({e}) — reading it')
        sold_qty = 0; sold_notional = 0.0; reason = None; deadline = time.time() + 3.0
        while True:
            pending = False; sold_qty = 0; sold_notional = 0.0
            for key, oid in legs:
                st = read_order(alpaca, oid); fq = int(st.get('filled_qty') or 0); px = float(st.get('filled_avg_price') or 0)
                if fq > 0 and px > 0: sold_qty += fq; sold_notional += fq * px; reason = {'tp_leg_id': 'target', 'sl_leg_id': 'stop'}.get(key, 'eod')
                if st and str(st.get('status', '')).lower() not in TERMINAL: pending = True
            if not pending or time.time() >= deadline: break
            time.sleep(0.25)
        remaining = max(0, shares - sold_qty)
        if remaining == 0 and sold_qty > 0:
            px = sold_notional / sold_qty; pnl = (px - entry) * sold_qty if entry else None
            db.update_trade(r['id'], {'order_status': 'closed', 'exit_price': px, 'exit_reason': reason or 'eod', 'exited_at': datetime.now(timezone.utc).isoformat(), 'pnl': pnl,
                                      'pnl_pct': (px / entry - 1) * 100 if entry else None})
            print(f'{sym}: already sold by our exit orders ({sold_qty} @ {px:.2f}, {reason}) — booked, nothing to sell'); n_ok += 1; continue
        held = {p.get('symbol'): int(float(p.get('qty') or 0)) for p in (alpaca.get_open_positions() or [])}
        if held.get(sym, 0) <= 0:
            print(f'{sym}: {remaining} unsold per our orders but nothing held at the broker — exit_pending_verification (engine reconciles at boot)')
            db.update_trade(r['id'], {'order_status': 'exit_pending_verification'}); continue
        qty = min(remaining, held[sym])                                        # never more than ours, never more than held
        try:
            q = alpaca.get_latest_quote(sym); ref = float(q.get('bid_price') or 0) or entry
            coid = f"hod-dm-{sym}-{today[5:]}"[:48]
            od = alpaca.submit_limit_sell_order(sym, qty, round(ref * 0.99, 2), client_order_id=coid)
            pd_['deadman_close_order_id'] = od.get('id'); pd_['closed_qty'] = sold_qty; pd_['closed_notional'] = round(sold_notional, 2)
            db.update_trade(r['id'], {'order_status': 'exit_pending_verification', 'exit_reason': 'deadman_flat', 'pattern_data': json.dumps(pd_)})
            print(f'{sym}: DEAD-MAN FLAT x{qty} limit {round(ref * 0.99, 2)} order {od.get("id")} (legs sold {sold_qty} before)'); n_ok += 1
        except Exception as e:
            print(f'{sym}: dead-man flat FAILED: {e}')
    try:
        subprocess.run([sys.executable, 'scripts/send_telegram_alert.py', f'[HOD DEAD-MAN] trader inactive at the close: {n_ok} of {len(rows)} open hod_break position(s) handled, see logs/hod_deadman.log'], check=False)
    except Exception:
        pass
    return 0 if n_ok == len(rows) else 1


if __name__ == '__main__':
    sys.exit(main())
