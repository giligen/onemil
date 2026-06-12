"""Production smoke test for L9 safety net.

Validates — without waiting for a real incident — that the FABC fix's
two critical paths fire correctly when invoked manually:

  PATH A — held_qty retry on escalate market close (Fix 1)
    Inject a synthetic Alpaca client whose close_position raises
    40310000 N times then succeeds. Drive _escalate_to_market_close
    against it and assert: retry attempts logged, eventual
    BRANCH_MARKET_CLOSE, no emergency stop placed.

  PATH B — emergency stop placement when close exhausts (Fix 2 + Fix 4)
    Same harness, but close_position always fails. Assert:
    BRANCH_LAST_RESORT, emergency stop submitted via
    submit_stop_sell_order, CRITICAL log + Telegram notify_error
    captured with the FABC-pattern message.

This is the manual validation step from the live-promotion gate (5
trading days + smoke test). Run it once per major code change to L9
helpers, OR any time you want to confirm the safety net wiring still
fires correctly end-to-end.

Run from /home/ec2-user/onemil:
    python3 scripts/smoke_test_emergency_stop.py

Or with the --send-telegram flag to actually exercise the Telegram
channel (uses the .env-configured bot — sends to your real chat):
    python3 scripts/smoke_test_emergency_stop.py --send-telegram

The script reports each path's outcome in a clear PASS/FAIL summary.
Exits 0 only if BOTH paths pass.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Optional
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv('/home/ec2-user/onemil/.env')

from trading.stop_monitor import StopMonitor, WatchEntry


logger = logging.getLogger('smoke')


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

class _LogCapture(logging.Handler):
    """Capture WARNING+ log records emitted by the stop_monitor module."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def has_record_matching(self, level: int, needle: str) -> bool:
        return any(r.levelno >= level and needle in r.getMessage()
                    for r in self.records)


def _make_watch(symbol: str = 'SMOKE', shares: int = 1000,
                  stop_price: float = 9.99) -> WatchEntry:
    return WatchEntry(
        symbol=symbol, stop_price=stop_price, shares=shares,
        tp_leg_id='tp-smoke', sl_leg_id='sl-smoke',
        trade_db_id=0, entry_price=10.00,
    )


def _make_monitor(notifier=None) -> StopMonitor:
    """Construct a StopMonitor instance without going through __init__
    (which wires WebSocket + Alpaca clients). Set just enough state to
    exercise _escalate_to_market_close and _place_emergency_stop_fallback.
    """
    m = StopMonitor.__new__(StopMonitor)
    m.notifier = notifier
    # Match production timeouts but compress retry waits for the smoke run.
    m._STOP_EXIT_FILL_TIMEOUT_S = 1.0
    m._MARKET_CLOSE_FILL_TIMEOUT_S = 1.0
    m._HELD_QTY_RETRY_BACKOFFS_S = (0.05, 0.1, 0.2)  # 0.35s total — quick
    return m


def _smart_get_order(limit_oid: str, close_oid: str, sl_oid: str,
                      close_fill_px: float = 9.97):
    """Per-order_id get_order stub. Limit pending, close filled, SL canceled."""
    def _get(oid):
        oid = str(oid)
        if oid == limit_oid:
            return {'status': 'new', 'filled_qty': 0,
                    'filled_avg_price': None}
        if oid == close_oid:
            return {'status': 'filled', 'filled_qty': 1000,
                    'filled_avg_price': close_fill_px}
        if oid == sl_oid:
            return {'status': 'canceled', 'filled_qty': 0,
                    'filled_avg_price': None}
        return {'status': 'new', 'filled_qty': 0,
                'filled_avg_price': None}
    return _get


# ---------------------------------------------------------------------------
# Path A — held_qty retry succeeds on attempt N
# ---------------------------------------------------------------------------

async def _run_path_a() -> tuple:
    """Returns (passed: bool, summary: str)."""
    a = MagicMock()
    a.cancel_order = MagicMock(return_value=True)
    a.trading_client = MagicMock()
    a.trading_client.get_orders = MagicMock(return_value=[])
    attempts = {'n': 0}

    def close_position(symbol):
        attempts['n'] += 1
        if attempts['n'] < 3:
            raise Exception(
                'insufficient qty available for order (requested: 1000, '
                'available: 0); code=40310000, held_for_orders=1000'
            )
        return {'id': 'mkt-close-smoke', 'status': 'accepted'}

    a.close_position = close_position
    a.get_order = _smart_get_order(
        limit_oid='limit-smoke', close_oid='mkt-close-smoke',
        sl_oid='sl-smoke', close_fill_px=9.97,
    )

    cap = _LogCapture()
    logging.getLogger('trading.stop_monitor').addHandler(cap)
    try:
        m = _make_monitor()
        watch = _make_watch()
        price, oid, branch = await m._escalate_to_market_close(
            a, 'SMOKE', 'limit-smoke', 9.99,
            sl_leg_id='sl-smoke', watch=watch,
        )
    finally:
        logging.getLogger('trading.stop_monitor').removeHandler(cap)

    issues = []
    if attempts['n'] != 3:
        issues.append(f"expected 3 close_position attempts, got {attempts['n']}")
    if branch != StopMonitor.BRANCH_MARKET_CLOSE:
        issues.append(f"expected BRANCH_MARKET_CLOSE, got {branch}")
    if not cap.has_record_matching(logging.WARNING,
                                     'held_for_orders race'):
        issues.append("no held_for_orders retry WARNING logged")
    # Emergency stop must NOT have been placed (we recovered)
    if hasattr(a, 'submit_stop_sell_order'):
        # ensure submit_stop_sell_order was not invoked — MagicMock would
        # autogenerate it on attribute access; check if it was actually
        # called.
        try:
            a.submit_stop_sell_order.assert_not_called()
        except AssertionError:
            issues.append("emergency stop was placed but shouldn't have been")

    if issues:
        return False, " | ".join(issues)
    return True, (
        f"3 attempts → success on retry, branch={branch}, fill=${price:.2f}, "
        f"order={oid}"
    )


# ---------------------------------------------------------------------------
# Path B — exhaust retries → emergency stop fires
# ---------------------------------------------------------------------------

async def _run_path_b(notifier=None) -> tuple:
    a = MagicMock()
    a.cancel_order = MagicMock(return_value=True)
    a.trading_client = MagicMock()
    a.trading_client.get_orders = MagicMock(return_value=[])

    def close_position(symbol):
        raise Exception(
            'insufficient qty available for order; code=40310000, '
            'held_for_orders=1000'
        )

    a.close_position = close_position
    a.get_order = _smart_get_order(
        limit_oid='limit-smoke', close_oid='never-fills',
        sl_oid='sl-smoke',
    )
    a.submit_stop_sell_order = MagicMock(
        return_value={'id': 'emergency-smoke', 'status': 'accepted'}
    )

    cap = _LogCapture()
    logging.getLogger('trading.stop_monitor').addHandler(cap)
    try:
        m = _make_monitor(notifier=notifier)
        watch = _make_watch()
        price, oid, branch = await m._escalate_to_market_close(
            a, 'SMOKE', 'limit-smoke', 9.99,
            sl_leg_id='sl-smoke', watch=watch,
        )
    finally:
        logging.getLogger('trading.stop_monitor').removeHandler(cap)

    issues = []
    if branch != StopMonitor.BRANCH_LAST_RESORT:
        issues.append(f"expected BRANCH_LAST_RESORT, got {branch}")
    # Emergency stop must have been called
    try:
        a.submit_stop_sell_order.assert_called_once()
        kw = a.submit_stop_sell_order.call_args.kwargs
        if kw.get('symbol') != 'SMOKE':
            issues.append(f"emergency symbol wrong: {kw}")
        # Stop = min(trigger 9.99, watch.stop 9.99) * 0.99 = 9.89
        if not (kw.get('stop_price') and 9.85 <= kw['stop_price'] <= 9.92):
            issues.append(f"emergency stop_price wrong: {kw.get('stop_price')}")
    except AssertionError as e:
        issues.append(f"emergency stop NOT placed: {e}")
    # CRITICAL log fired
    if not cap.has_record_matching(logging.CRITICAL,
                                     'EMERGENCY stop-market'):
        issues.append("no CRITICAL 'EMERGENCY stop-market' log line")
    # Telegram alert
    if notifier is not None:
        if not notifier.notify_error.called:
            issues.append("notifier.notify_error was not called")
        else:
            msg = notifier.notify_error.call_args[0][0]
            if 'EMERGENCY STOP' not in msg:
                issues.append(f"Telegram message missing 'EMERGENCY STOP': {msg!r}")

    if issues:
        return False, " | ".join(issues)
    return True, (
        f"close failed 4× → emergency stop placed (stop_price="
        f"${a.submit_stop_sell_order.call_args.kwargs['stop_price']:.2f}, "
        f"order=emergency-smoke), CRITICAL logged"
        + (", Telegram notified" if notifier is not None else "")
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

async def _main_async(send_telegram: bool) -> int:
    print("L9 safety-net smoke test", flush=True)
    print("=" * 60, flush=True)

    # Path A
    print("\nPATH A — held_qty retry succeeds on attempt N", flush=True)
    ok_a, summary_a = await _run_path_a()
    print(f"  {'PASS' if ok_a else 'FAIL'}: {summary_a}", flush=True)

    # Path B — optionally exercise the real Telegram channel
    if send_telegram:
        from notifications.telegram_notifier import TelegramNotifier
        bot = os.getenv('TELEGRAM_BOT_TOKEN')
        chat = os.getenv('TELEGRAM_CHAT_ID')
        if not bot or not chat:
            print("  ⚠️  TELEGRAM_BOT_TOKEN/CHAT_ID not set — using mock",
                  flush=True)
            notifier = MagicMock()
        else:
            notifier = TelegramNotifier(bot_token=bot, chat_id=chat)
            print("  📡 Using REAL TelegramNotifier — message will be sent",
                  flush=True)
    else:
        notifier = MagicMock()

    print("\nPATH B — exhaust retries → emergency stop + alert", flush=True)
    ok_b, summary_b = await _run_path_b(notifier=notifier)
    print(f"  {'PASS' if ok_b else 'FAIL'}: {summary_b}", flush=True)

    print("\n" + "=" * 60, flush=True)
    if ok_a and ok_b:
        print("RESULT: ALL PATHS PASS — L9 safety net is wired correctly",
              flush=True)
        return 0
    else:
        print("RESULT: FAILURES DETECTED — investigate before live promotion",
              flush=True)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test the L9 emergency-stop + held_qty retry "
                    "paths in StopMonitor."
    )
    parser.add_argument('--send-telegram', action='store_true',
                        help="Actually invoke TelegramNotifier from Path B "
                             "(requires TELEGRAM_BOT_TOKEN/CHAT_ID in .env). "
                             "Default: mock the notifier.")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Show DEBUG/INFO log lines from stop_monitor.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
        datefmt='%H:%M:%S',
    )

    return asyncio.run(_main_async(args.send_telegram))


if __name__ == '__main__':
    sys.exit(main())
