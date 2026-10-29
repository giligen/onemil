"""Crash guard shared by every cron entry point (scripts/*.py).

2026-10-26: the MOM sleeve crashed on the TXSE asset-enum bug and nobody was told — an uncaught exception
only prints a traceback to the cron log, no logger.error fires, so the Telegram error handler never sees it.
`run_guarded` turns ANY uncaught exception into one ERROR (-> Telegram) and re-raises it unchanged, and
`attach_telegram_error_handler` lets a script's own logger.error lines reach Telegram too.
"""
from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


def attach_telegram_error_handler() -> bool:
    """Attach a TelegramErrorHandler to the root logger (idempotent); never raises.

    Returns True when a handler is on the root logger afterwards, False when Telegram is not configured
    or the attachment failed (logged as WARNING; the caller's own work must not depend on it).
    """
    from config import Config
    from monitoring.telegram_error_handler import TelegramErrorHandler
    root = logging.getLogger()
    if any(isinstance(h, TelegramErrorHandler) for h in root.handlers):
        return True
    try:
        cfg = Config()
        if cfg.telegram_bot_token and cfg.telegram_chat_id:
            root.addHandler(TelegramErrorHandler(bot_token=cfg.telegram_bot_token, chat_id=cfg.telegram_chat_id))
            return True
        logger.warning("cron_guard: Telegram not configured — errors will only reach the cron log")
    except Exception as e:  # noqa: BLE001 - the script's own work / original crash must stay the one that propagates
        logger.warning("cron_guard: could not attach the Telegram error handler (%s)", e)
    return False


def _wait_for_telegram_sends(timeout: float) -> None:
    """Join in-flight Telegram error sends (daemon threads) so a crash exit does not drop the message."""
    from monitoring.telegram_error_handler import TelegramErrorHandler
    try:
        for h in list(logging.getLogger().handlers):
            if isinstance(h, TelegramErrorHandler):
                h.wait_for_sends(timeout)
    except Exception as e:  # noqa: BLE001 - never mask the original crash
        logger.warning("cron_guard: waiting for the Telegram crash message failed (%s)", e)


def run_guarded(main: Callable[[], int], name: str) -> int:
    """Call `main()`; log ANY uncaught exception as ERROR (-> Telegram), wait for the send, re-raise unchanged.

    SystemExit / KeyboardInterrupt pass through untouched. The handler is attached at crash time (after the
    script's own logging.basicConfig, which an earlier root handler would turn into a no-op).
    """
    try:
        return main()
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as e:
        attach_telegram_error_handler()
        logger.error("%s crashed (%s): %s", name, type(e).__name__, e)
        _wait_for_telegram_sends(10.0)
        raise
