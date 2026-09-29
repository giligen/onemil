"""
Custom logging handler to send ERROR level logs to Telegram.

Sends real-time notifications for all ERROR level log messages.
NO SILENT FAILURES - every error is reported.

Transport is delegated to a `TelegramNotifier` instance (2026-09-29: this
handler used to POST directly with only a mod-10 duplicate counter, no
rate cap and no 429 handling; 144 identical ERRORs in 30s drew 68x HTTP
429 before the owner saw a wall of errors). Delegating means this handler
shares the same dedup + rate-cap + 429-retry gate as every other Telegram
send path instead of a second, weaker copy of it — see
notifications/telegram_notifier.py.
"""

import logging
import asyncio
import html
import sys
import threading
import traceback
from typing import Optional
from datetime import datetime, timezone

from notifications.telegram_notifier import TelegramNotifier


class TelegramErrorHandler(logging.Handler):
    """
    Logging handler that sends ERROR level messages to Telegram.

    Sends formatted error notifications with timestamp, logger name,
    file location, and full error message. Applies a cheap mod-10
    duplicate counter of its own (same formatted record seen back to
    back) before handing anything to the shared TelegramNotifier, which
    then applies the real flood control (60s exact-text dedup, 20/min
    rate cap, single 429 retry) shared with every other Telegram sender.
    """

    # Loggers whose ERRORs are external/expected — not actionable bugs.
    # Universe build hits Yahoo Finance for thousands of symbols; rate limits
    # and 404s for delisted/class-share tickers are normal noise.
    NOISY_LOGGERS = {
        'yfinance',
        'data_sources.float_provider',
    }

    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram error handler.

        Args:
            bot_token: Telegram bot token from BotFather
            chat_id: Telegram chat ID to send messages to
        """
        super().__init__(level=logging.ERROR)
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        # Shared transport: dedup, rate cap and 429 handling all live here.
        self._notifier = TelegramNotifier(bot_token=bot_token, chat_id=chat_id, enabled=True)

        self.last_error: Optional[str] = None
        self.last_error_count: int = 0

        self.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record by sending it to Telegram.

        Called for every ERROR level log message. Skips noisy external-API
        loggers (yfinance, float_provider). Applies a mod-10 pre-filter for
        the exact same formatted record repeating back to back; the actual
        send then goes through the shared TelegramNotifier's flood control.
        """
        # Filter out noise from external APIs — these are not application bugs
        for noisy in self.NOISY_LOGGERS:
            if record.name == noisy or record.name.startswith(noisy + '.'):
                return
        try:
            message = self._format_error_message(record)

            if message == self.last_error:
                self.last_error_count += 1
                if self.last_error_count % 10 != 0:
                    return
                message += f"\n\n(⚠️ This error repeated {self.last_error_count} times)"
            else:
                self.last_error = message
                self.last_error_count = 1

            self._send_async(message)

        except Exception as e:
            print(f"[TelegramErrorHandler] Failed to send: {e}", file=sys.stderr)

    def _format_error_message(self, record: logging.LogRecord) -> str:
        """Format error record into Telegram message with HTML formatting."""
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc)
        time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')

        lines = [
            "🚨 <b>OneMil ERROR</b> 🚨",
            "",
            f"<b>Time:</b> {time_str}",
            f"<b>Logger:</b> <code>{html.escape(record.name)}</code>",
            f"<b>File:</b> <code>{html.escape(record.filename)}:{record.lineno}</code>",
            "",
            f"<b>Message:</b>",
            f"<code>{html.escape(record.getMessage())}</code>",
        ]

        if record.exc_info:
            exc_text = ''.join(traceback.format_exception(*record.exc_info))
            if len(exc_text) > 2000:
                exc_text = exc_text[:2000] + "\n... (truncated)"
            lines.append("")
            lines.append("<b>Exception:</b>")
            lines.append(f"<pre>{html.escape(exc_text)}</pre>")

        return "\n".join(lines)

    def _send_async(self, message: str) -> None:
        """Send message to Telegram, handling both async and sync contexts."""
        try:
            loop = asyncio.get_running_loop()
            asyncio.ensure_future(self._send_to_telegram(message), loop=loop)
        except RuntimeError:
            thread = threading.Thread(
                target=self._send_sync,
                args=(message,),
                daemon=True,
            )
            thread.start()

    def _send_sync(self, message: str) -> None:
        """Synchronous wrapper to send Telegram message from a thread."""
        try:
            asyncio.run(self._send_to_telegram(message))
        except Exception as e:
            print(f"[TelegramErrorHandler] _send_sync failed: {e}", file=sys.stderr)

    async def _send_to_telegram(self, message: str) -> None:
        """
        Send message to Telegram via the shared TelegramNotifier gate.

        Delegates to TelegramNotifier.send_message so this handler's
        floods are deduped and rate-capped by the same mechanism as every
        other Telegram send path, with the same single-retry 429 handling.
        """
        try:
            await self._notifier.send_message(message, parse_mode="HTML")
        except Exception as e:
            print(f"[TelegramErrorHandler] _send_to_telegram failed: {e}", file=sys.stderr)
