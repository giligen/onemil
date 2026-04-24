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
    """True if it's past 16:15 ET on a weekday, OR it's a weekend.

    Weekends are always "closed". Weekday holidays return True only post
    16:15 ET — in practice Alpaca won't return a bar for a holiday during
    regular hours so the guard is a no-op, but `True` is the safer default
    for the save-side path (more permissive — allows persisting whatever
    the API returns).

    Notes:
      - Does NOT query Alpaca's market calendar. Relying on clock + weekday
        is sufficient for the current pollution fix because Alpaca simply
        returns nothing on closed days.
      - Does NOT handle early-close days (day after Thanksgiving, Christmas
        Eve). If we care, swap in a calendar-aware impl here and all
        callers benefit automatically.
    """
    now = _now_et(now_et)
    if ET is None:
        return True  # no tz support — permissive fallback
    if now.weekday() >= 5:
        return True
    minutes_today = now.hour * 60 + now.minute
    return minutes_today >= REGULAR_SESSION_CLOSE_MINUTES
