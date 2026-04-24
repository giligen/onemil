"""Trading-hours helpers, single source of truth.

Centralizes all decisions about "is the US equity regular session over?" so
the save-side and read-side daily-bar guards can't drift. If we ever need
to support early-close days (e.g., day-after-Thanksgiving half day) or
full market holidays beyond weekends, it happens here — once.

Usage:

    from trading.trading_hours import is_regular_session_closed, today_et

    if is_regular_session_closed():
        # safe to persist today's daily bar
        ...

Rationale for the 16:15 ET threshold: regular session closes 16:00 ET, but
consolidated-tape settlement and post-market trade corrections can trickle
in for 5-15 minutes after. Using 16:15 gives us a margin while still
letting post-close BT runs complete quickly.
"""
from __future__ import annotations

from datetime import datetime, timezone, date as _date
from typing import Optional

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo('America/New_York')
except Exception:  # pragma: no cover
    ET = None


# Minutes after 16:00 ET that we consider "session finalized". Small margin
# for consolidated-tape settlement and late trade corrections.
REGULAR_SESSION_CLOSE_MINUTES = 16 * 60 + 15


def _now_et(now_et: Optional[datetime] = None) -> datetime:
    """Return `now_et` or derive from real wall clock in ET."""
    if now_et is not None:
        return now_et
    if ET is None:
        # Fallback: UTC - 4 (EDT). Better than crashing; tests shouldn't
        # exercise this branch.
        return datetime.now(timezone.utc)
    return datetime.now(timezone.utc).astimezone(ET)


def today_et(now_et: Optional[datetime] = None) -> _date:
    """Current calendar date in US/Eastern."""
    return _now_et(now_et).date()


def is_regular_session_closed(now_et: Optional[datetime] = None) -> bool:
    """True iff today's regular-session daily bar can be trusted as FINAL.

    Despite the name, this is specifically the "is a today-row in
    daily_bars safe to persist / return to callers?" predicate. It is
    NOT a "is the market currently closed?" check. Specifically:

      - Saturday/Sunday → True (no market today, no pollution risk).
      - Weekday before 16:15 ET → False (session in progress or
        pre-market; Alpaca returns a provisional intraday snapshot).
      - Weekday ≥ 16:15 ET → True (consolidated-tape has settled; the
        bar Alpaca returns is the real end-of-session close).
      - Weekday holiday (e.g. MLK Day Monday) pre-16:15 → returns
        False even though market IS closed. This is the cosmetic
        imprecision in the name. Safe in practice because Alpaca
        returns no bar on a holiday, so the save-side guard has
        nothing to drop and the read-side guard has nothing to hide.

    Notes:
      - Does NOT query Alpaca's market calendar. Relying on clock +
        weekday is sufficient for the pollution fix this function
        serves, because Alpaca simply returns nothing on closed days.
      - Does NOT handle early-close days (day after Thanksgiving,
        Christmas Eve). If we care, swap in a calendar-aware impl
        here and all callers benefit automatically.
    """
    now = _now_et(now_et)
    if ET is None:
        return True  # no tz support — permissive fallback
    if now.weekday() >= 5:
        return True
    minutes_today = now.hour * 60 + now.minute
    return minutes_today >= REGULAR_SESSION_CLOSE_MINUTES
