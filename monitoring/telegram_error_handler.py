"""
Custom logging handler to send ERROR level logs to Telegram.

Sends real-time notifications for all ERROR level log messages.
Includes deduplication to avoid spam from repeated errors.
NO SILENT FAILURES - every error is reported.
"""

import logging
import asyncio
import aiohttp
import html
import sys
import threading
import traceback
from typing import Optional
from datetime import datetime, timezone


class TelegramErrorHandler(logging.Handler):
    """
    Logging handler that sends ERROR level messages to Telegram.

    Sends formatted error notifications with timestamp, logger name,
    file location, and full error message. Deduplicates repeated errors
    (sends every 10th duplicate).
    """

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

        self.last_error: Optional[str] = None
        self.last_error_count: int = 0

        self.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record by sending it to Telegram.

        Called for every ERROR level log message. Deduplicates repeated errors.
        """
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
        """Send message to Telegram using aiohttp."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': 'HTML',
                    'disable_web_page_preview': True,
                }
                timeout = aiohttp.ClientTimeout(total=5.0)
                async with session.post(self.api_url, json=payload, timeout=timeout) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        print(
                            f"[TelegramErrorHandler] HTTP {response.status}: {response_text}",
                            file=sys.stderr,
                        )
        except asyncio.TimeoutError:
            print("[TelegramErrorHandler] Request timed out", file=sys.stderr)
        except Exception as e:
            print(f"[TelegramErrorHandler] _send_to_telegram failed: {e}", file=sys.stderr)
