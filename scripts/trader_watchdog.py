"""Startup watchdog (2026-07-06 incident: silent 4h pre-start crash-loop).

Cron 12:40 UTC weekdays (10min after auto-start). Alerts via Telegram if
onemil-trader is not active, or has restarted more than twice since the
12:30 start (crash-loop signature). Silent when healthy.
"""
import os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(str(ROOT / '.env'))

def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

def main() -> int:
    """Check onemil-trader state / restart count; Telegram + print when unhealthy. Returns 0."""
    state = sh(['systemctl', 'is-active', 'onemil-trader'])
    starts = sh(['bash', '-c',
        "journalctl -u onemil-trader --since '30 minutes ago' --no-pager "
        "| grep -c 'Started onemil-trader' || true"])
    try:
        n_starts = int(starts or 0)
    except ValueError:
        n_starts = 0
    if state == 'active' and n_starts <= 2:
        return 0   # healthy — stay silent
    msg = (f"🔴 TRADER WATCHDOG: onemil-trader state='{state}', "
           f"{n_starts} starts in last 30min (crash-loop if >2). "
           f"Check: journalctl -u onemil-trader | grep ERROR")
    try:
        from notifications.telegram_notifier import TelegramNotifier
        TelegramNotifier(os.getenv('TELEGRAM_BOT_TOKEN'),
                         os.getenv('TELEGRAM_CHAT_ID'),
                         enabled=True).send_message_sync(msg)
    except Exception as e:
        import logging, urllib.parse, urllib.request
        logging.getLogger(__name__).warning('trader_watchdog: TelegramNotifier failed (%s) — raw API fallback', e)
        tok, chat = os.getenv('TELEGRAM_BOT_TOKEN'), os.getenv('TELEGRAM_CHAT_ID')
        if tok and chat:
            urllib.request.urlopen(
                f"https://api.telegram.org/bot{tok}/sendMessage?chat_id={chat}"
                f"&text={urllib.parse.quote(msg)}", timeout=10)
    print(msg)
    return 0


if __name__ == "__main__":
    from monitoring.cron_guard import run_guarded  # 11/12: no silent traceback — ERROR -> Telegram
    sys.exit(run_guarded(main, "trader_watchdog"))
