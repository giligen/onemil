"""MANUAL emergency flattener — cancel + close all onemil positions.

NOTE (2026-07-03 post-mortem): originally written as a half-day watchdog
under the mistaken belief the trader had no early-close awareness. It does:
scanner._is_trading_day() SKIPS short trading days entirely (see
tests/test_scanner_half_day_policy.py — BT-validated policy, half-days are
-$2,656 net over 18mo). This script is retained as a MANUAL emergency tool
only (e.g., trader wedged with open positions near a close).

Ownership-aware: ORB account fully (exclusively ours); on the shared MAIN
account only symbols with open onemil trade rows today — other systems'
positions are never touched.

Run manually, then confirm the Telegram report:
  python3 scripts/half_day_flattener.py
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(str(ROOT / '.env'))

from alpaca.trading.client import TradingClient


def log(m):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}Z] {m}", flush=True)


def wait_until_utc(hh, mm, ss=0):
    while True:
        now = datetime.now(timezone.utc)
        tgt = now.replace(hour=hh, minute=mm, second=ss, microsecond=0)
        if (tgt - now).total_seconds() <= 0:
            return
        time.sleep(min((tgt - now).total_seconds(), 30))


def our_main_symbols():
    """Symbols the onemil trader owns on the shared MAIN account: any
    bull_flag/macd_wave trade row today that is not closed."""
    import sqlite3
    conn = sqlite3.connect(ROOT / 'data' / 'trades.db', timeout=15)
    cur = conn.execute(
        """SELECT DISTINCT symbol FROM trades
           WHERE trade_date = date('now')
             AND strategy IN ('bull_flag','macd_wave')
             AND (exit_price IS NULL OR exited_at IS NULL)""")
    syms = {r[0] for r in cur}
    conn.close()
    return syms


def accounts():
    for tag, k, s, p in (('MAIN', 'ALPACA_API_KEY', 'ALPACA_API_SECRET', 'ALPACA_PAPER'),
                         ('ORB', 'ALPACA_ORB_API_KEY', 'ALPACA_ORB_API_SECRET', 'ALPACA_ORB_PAPER')):
        key, sec = os.getenv(k), os.getenv(s)
        if key and sec:
            yield tag, TradingClient(key, sec, paper=os.getenv(p, 'true').lower() == 'true')


def main():
    log("half-day flattener armed — cancel 16:48Z, flatten 16:50Z (13:00 ET close)")
    wait_until_utc(16, 48, 0)
    report = []
    ours_main = our_main_symbols()
    log(f"MAIN-account ownership set (onemil open trades today): {sorted(ours_main) or 'none'}")
    for tag, tc in accounts():
        try:
            orders = tc.get_orders()
            for o in orders:
                if tag == 'MAIN' and o.symbol not in ours_main:
                    continue  # not ours — shared account, leave alone
                try:
                    tc.cancel_order_by_id(o.id)
                    log(f"[{tag}] cancelled order {o.symbol} {o.id}")
                    report.append(f"{tag}: cancelled order {o.symbol}")
                except Exception as e:
                    log(f"[{tag}] cancel {o.symbol} failed: {e}")
        except Exception as e:
            log(f"[{tag}] order query failed: {e}")
            report.append(f"{tag}: ORDER QUERY FAILED {e}")

    wait_until_utc(16, 50, 0)
    for tag, tc in accounts():
        try:
            pos = tc.get_all_positions()
            if not pos:
                log(f"[{tag}] already flat")
                continue
            for p in pos:
                if tag == 'MAIN' and p.symbol not in ours_main:
                    log(f"[{tag}] leaving {p.symbol} (not ours — shared account)")
                    continue
                try:
                    tc.close_position(p.symbol)
                    log(f"[{tag}] closing {p.symbol} x{p.qty}")
                    report.append(f"{tag}: closed {p.symbol} x{p.qty}")
                except Exception as e:
                    log(f"[{tag}] close {p.symbol} FAILED: {e}")
                    report.append(f"{tag}: CLOSE {p.symbol} FAILED {e}")
        except Exception as e:
            log(f"[{tag}] position query failed: {e}")

    time.sleep(20)
    still = []
    for tag, tc in accounts():
        try:
            for p in tc.get_all_positions():
                if tag == 'MAIN' and p.symbol not in ours_main:
                    continue  # other system's position — expected to remain
                still.append(f"{tag}:{p.symbol}x{p.qty}")
        except Exception:
            pass

    try:
        from notifications.telegram_notifier import TelegramNotifier
        n = TelegramNotifier(os.getenv('TELEGRAM_BOT_TOKEN'), os.getenv('TELEGRAM_CHAT_ID'), enabled=True)
        body = "\n".join(report) if report else "Nothing to do — all flat before early close."
        tail = f"\n⚠️ STILL OPEN: {still}" if still else "\n✅ All onemil positions flat into the long weekend."
        n.send_message_sync(
            f"<b>[HALF-DAY FLATTENER] {datetime.now(timezone.utc).strftime('%Y-%m-%d')}</b>\n"
            f"(13:00 ET early close; trader's 15:45 force-close would have missed it)\n"
            f"{body}{tail}", parse_mode='HTML')
    except Exception as e:
        log(f"telegram failed: {e}")
    log("done")


if __name__ == '__main__':
    main()
