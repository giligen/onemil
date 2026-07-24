"""Hourly open-positions P&L pulse → Telegram (only while holding).

Fills the entry→exit silence window (the TENX +200→−200 round-trip was
invisible for hours). Runs from cron every hour during market hours;
exits SILENTLY when flat, when outside 9:35-16:00 ET, or on non-trading
days — so it only ever speaks when there's a position to report.

Deliberately an EXTERNAL observer: reads broker positions (both
accounts) + the trades DB, not engine memory — it reports truth even if
the trader is wedged, and touches zero production code. MAIN account is
shared with other systems: only symbols with an open onemil trade row
today are reported (same ownership rule as half_day_flattener).

Usage: python3 scripts/holdings_pulse.py [--force] [--no-telegram]
  --force: skip the market-hours/trading-day gate (testing)
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from dotenv import load_dotenv
load_dotenv(str(ROOT / '.env'))

ET = ZoneInfo('America/New_York')


def in_market_window(now_utc: datetime) -> bool:
    et = now_utc.astimezone(ET)
    if et.weekday() >= 5:
        return False
    t = et.time()
    return t >= datetime.strptime('09:35', '%H:%M').time() and \
        t <= datetime.strptime('16:00', '%H:%M').time()


def our_main_symbols() -> set:
    """Symbols onemil owns on the shared MAIN account: open BF/MACD rows today."""
    conn = sqlite3.connect(ROOT / 'data' / 'trades.db', timeout=15)
    cur = conn.execute(
        """SELECT DISTINCT symbol FROM trades
           WHERE trade_date = date('now')
             AND strategy IN ('bull_flag', 'macd_wave')
             AND exit_price IS NULL""")
    syms = {r[0] for r in cur}
    conn.close()
    return syms


def open_entry_meta() -> dict:
    """(symbol -> {strategy, entry_price, stop_price, entered_at}) for
    OPEN onemil rows — any date, so overnight holds keep their tag."""
    conn = sqlite3.connect(ROOT / 'data' / 'trades.db', timeout=15)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT symbol, strategy, entry_price,
                  COALESCE(real_stop_loss_price, stop_loss_price) AS stop_price,
                  created_at
           FROM trades WHERE exit_price IS NULL
             AND entry_price IS NOT NULL""").fetchall()
    conn.close()
    return {r['symbol']: dict(r) for r in rows}


def fetch_positions() -> list:
    """[{account, symbol, qty, avg_entry, current, upl}] across both accounts,
    ownership-filtered on MAIN."""
    from alpaca.trading.client import TradingClient
    ours_main = our_main_symbols()
    out = []
    for tag, k, s, p in (('MAIN', 'ALPACA_API_KEY', 'ALPACA_API_SECRET', 'ALPACA_PAPER'),
                         ('ORB', 'ALPACA_ORB_API_KEY', 'ALPACA_ORB_API_SECRET',
                          'ALPACA_ORB_PAPER')):
        key, sec = os.getenv(k), os.getenv(s)
        if not (key and sec):
            continue
        try:
            tc = TradingClient(key, sec,
                               paper=os.getenv(p, 'true').lower() == 'true')
            for pos in tc.get_all_positions():
                if tag == 'MAIN' and pos.symbol not in ours_main:
                    continue  # another system's position — not ours to report
                out.append(dict(
                    account=tag, symbol=pos.symbol, qty=float(pos.qty),
                    avg_entry=float(pos.avg_entry_price),
                    current=float(pos.current_price or 0),
                    upl=float(pos.unrealized_pl or 0)))
        except Exception as e:
            out.append(dict(account=tag, symbol='(QUERY FAILED)', qty=0,
                            avg_entry=0, current=0, upl=0, error=str(e)))
    return out


def realized_today() -> float:
    conn = sqlite3.connect(ROOT / 'data' / 'trades.db', timeout=15)
    row = conn.execute(
        """SELECT COALESCE(SUM((exit_price-entry_price)*shares), 0) FROM trades
           WHERE trade_date = date('now') AND exit_price IS NOT NULL""").fetchone()
    conn.close()
    return float(row[0])


def build_message(positions: list, realized: float, meta: dict,
                  now_utc: datetime) -> str:
    et = now_utc.astimezone(ET).strftime('%H:%M ET')
    lines = [f"<b>[HOLDINGS {et}]</b>"]
    total_upl = 0.0
    for p in positions:
        if p.get('error'):
            lines.append(f"⚠ {p['account']}: position query failed — {p['error'][:80]}")
            continue
        total_upl += p['upl']
        m = meta.get(p['symbol'], {})
        stop = m.get('stop_price')
        risk = (m['entry_price'] - stop) * p['qty'] \
            if stop and m.get('entry_price') else None
        r_txt = f"  {p['upl'] / risk:+.1f}R" if risk and risk > 0 else ""
        chg = (p['current'] - p['avg_entry']) / p['avg_entry'] * 100 \
            if p['avg_entry'] else 0.0
        # a position with no onemil trade row is NOT ours — the shared
        # account also carries other services' positions (2026-07-24:
        # wulf-late-day's WULF short printed '(orb)' via the account-tag
        # fallback and read as an impossible ORB short)
        lines.append(
            f"• {p['symbol']} ({m.get('strategy') or 'external'}) "
            f"{p['qty']:.0f}sh @{p['avg_entry']:.2f} → {p['current']:.2f} "
            f"({chg:+.1f}%)  <b>${p['upl']:+,.0f}</b>{r_txt}")
    lines.append(f"unrealized ${total_upl:+,.0f} | realized today ${realized:+,.0f} "
                 f"| day ${total_upl + realized:+,.0f}")
    return '\n'.join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--no-telegram', action='store_true')
    args = ap.parse_args()
    now = datetime.now(timezone.utc)
    if not args.force and not in_market_window(now):
        return 0
    positions = fetch_positions()
    real = [p for p in positions if not p.get('error')]
    errors = [p for p in positions if p.get('error')]
    if not real and not errors:
        return 0   # flat — stay silent
    msg = build_message(positions, realized_today(), open_entry_meta(), now)
    print(msg, flush=True)
    if not args.no_telegram and not os.environ.get('PYTEST_CURRENT_TEST'):
        try:
            from notifications.telegram_notifier import TelegramNotifier
            n = TelegramNotifier(os.getenv('TELEGRAM_BOT_TOKEN'),
                                 os.getenv('TELEGRAM_CHAT_ID'), enabled=True)
            n.send_message_sync(msg, parse_mode='HTML')
        except Exception as e:
            print(f"telegram failed: {e}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
