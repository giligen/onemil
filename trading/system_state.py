"""
system_state — process-wide halt flag for account-level events.

When the account-state monitor detects a critical transition (margin call,
status change, account/trading blocked), the scanner sets this flag.
The strategy engines check it before submitting new entries and refuse
to open new positions while the halt is set.

Scope: account-level halts only. Strategy-specific halts (daily loss cap,
configured kill switches) live in their respective engines.

Persistence: in-memory only. On restart, the account-state monitor re-checks
account state and re-detects any persisting halt-worthy condition. There is
no auto-resume; the operator must call `clear_account_halt()` (typically via
restart-after-investigation).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class _HaltRecord:
    halted: bool = False
    reason: Optional[str] = None
    event_type: Optional[str] = None
    halted_at: Optional[datetime] = None


_state = _HaltRecord()
_lock = threading.Lock()


def is_account_halted() -> bool:
    """True if any account-level event has set the halt flag."""
    with _lock:
        return _state.halted


def set_account_halt(
    *,
    event_type: str,
    reason: str,
    occurred_at: Optional[datetime] = None,
) -> None:
    """Set the account halt flag. Idempotent — re-setting an already-halted
    state preserves the original reason but updates the timestamp.
    """
    now = occurred_at or datetime.now(timezone.utc)
    with _lock:
        if not _state.halted:
            _state.halted = True
            _state.reason = reason
            _state.event_type = event_type
            _state.halted_at = now
            logger.critical(
                f"system_state: account halt set — event={event_type} "
                f"reason={reason!r} at {now.isoformat()}"
            )
        else:
            # Already halted; record the additional event in the log.
            logger.warning(
                f"system_state: account already halted (since {_state.halted_at}); "
                f"additional event {event_type} ({reason}) noted"
            )


def clear_account_halt() -> None:
    """Manually clear the halt. Use after investigating the trigger."""
    with _lock:
        if not _state.halted:
            return
        logger.warning(
            f"system_state: account halt CLEARED (was set at {_state.halted_at}, "
            f"reason {_state.reason!r})"
        )
        _state.halted = False
        _state.reason = None
        _state.event_type = None
        _state.halted_at = None


def get_halt_details() -> dict:
    """Return current halt record as a dict (for logging/telemetry)."""
    with _lock:
        return {
            "halted": _state.halted,
            "reason": _state.reason,
            "event_type": _state.event_type,
            "halted_at": _state.halted_at.isoformat() if _state.halted_at else None,
        }
