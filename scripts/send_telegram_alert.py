#!/usr/bin/env python3
"""Minimal shell-callable Telegram alert sender.

Cron/shell scripts (e.g. scripts/nightly_bt_update.sh) run OUTSIDE the
onemil-trader service, so they don't have the root-logger → Telegram relay
that in-process ERROR logs get. This sends a one-off Telegram message via
the same TelegramNotifier the service uses.

Reads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from .env via Config. Resolves
the repo root from __file__ so it works regardless of the caller's cwd.

Usage:   python3 scripts/send_telegram_alert.py "message text"
Exit:    0 sent, 1 not-configured / send-failed, 2 usage error.

Callers should treat a failed alert as non-fatal — never let it mask the
caller's own exit code (the nightly script invokes it with `|| true`).
"""
import os
import sys

# Repo root on sys.path so `config` / `notifications` import from any cwd.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def main() -> int:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("usage: send_telegram_alert.py <message>", file=sys.stderr)
        return 2
    message = sys.argv[1]
    try:
        from config import Config
        from notifications.telegram_notifier import TelegramNotifier
        cfg = Config()
        if not cfg.telegram_bot_token or not cfg.telegram_chat_id:
            print(
                "Telegram not configured (TELEGRAM_BOT_TOKEN/CHAT_ID empty) "
                "— alert not sent",
                file=sys.stderr,
            )
            return 1
        notifier = TelegramNotifier(
            bot_token=cfg.telegram_bot_token,
            chat_id=cfg.telegram_chat_id,
        )
        ok = notifier.send_message_sync(message)
        if not ok:
            print("send_message_sync returned False", file=sys.stderr)
            return 1
        return 0
    except Exception as e:
        print(f"send_telegram_alert failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
