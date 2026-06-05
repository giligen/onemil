"""
account_state_monitor — detect intraday margin / restriction transitions.

Background: FINRA retired the Pattern Day Trader rule in 2026. Alpaca replaced
it with a per-order real-time margin framework. Under the new rules:
  * `pattern_day_trader`, `daytrade_count`, `daytrading_buying_power` and the
    `last_*` variants are deprecated — `buying_power` is the canonical field.
  * Margin calls are issued in real-time when intraday exposure exceeds the
    available margin pool. Repeated unmet margin calls within 5 business days
    can lead to up to 90 days of account restriction.

There is currently NO single Account field that says "margin call issued"
(verified by probing live account 2026-06-05). The realistic signals are:
  * Transition of `status` away from `ACTIVE`
  * `trading_blocked` flips `True`
  * `account_blocked` flips `True`
  * `trade_suspended_by_user` flips `True` (e.g., user manually halts)
  * `maintenance_margin` or `initial_margin` becomes non-zero from prior zero
    (used positions consuming margin — informational, not halt-worthy)

This module is pure: input two snapshots → output a list of events. The
caller decides how to alert + how to halt. State persistence lives elsewhere
(an in-memory monitor in the scanner loop + an optional JSONL log file).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class AccountStateSeverity(str, Enum):
    """Severity tier of an account-state event."""

    INFO = "info"        # observed but not alert-worthy (margin first used, ...)
    WARN = "warn"        # operator should look (transfers blocked, ...)
    CRITICAL = "critical"  # trading must halt (account/trading blocked)


@dataclass(frozen=True)
class AccountStateSnapshot:
    """Minimum subset of Alpaca Account fields we react to.

    `status` is the string value of the Account status enum (e.g. 'ACTIVE',
    'ACCOUNT_UPDATED', 'APPROVAL_PENDING'). The blocking fields are booleans.
    Margin fields are numeric in USD. captured_at is wall-clock UTC.
    """

    status: str
    trading_blocked: bool
    account_blocked: bool
    transfers_blocked: bool
    trade_suspended_by_user: bool
    initial_margin: float
    maintenance_margin: float
    buying_power: float
    equity: float
    captured_at: datetime


@dataclass(frozen=True)
class AccountStateEvent:
    """A detected transition. `halt` indicates the system should stop opening
    new positions until manually cleared."""

    event_type: str
    severity: AccountStateSeverity
    detail: str
    occurred_at: datetime
    halt: bool


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

_OK_STATUS_VALUES = frozenset({"ACTIVE", "ACCOUNT_UPDATED"})


def _to_float(x: Any) -> float:
    if x is None:
        return 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("true", "1", "yes")


def _to_str_status(x: Any) -> str:
    """Normalize an Alpaca status (enum-or-str) to its plain string value."""
    if x is None:
        return ""
    return str(getattr(x, "value", x))


def snapshot_from_account_dict(
    info: Optional[Dict[str, Any]],
    now: Optional[datetime] = None,
) -> Optional[AccountStateSnapshot]:
    """Parse a dict returned by AlpacaClient.get_account_info() into a
    typed snapshot. Returns None if `info` is None/empty (caller treats as
    "couldn't sample this tick" and reuses the prior snapshot)."""
    if not info:
        return None
    captured_at = now or datetime.now(timezone.utc)
    return AccountStateSnapshot(
        status=_to_str_status(info.get("status", "")),
        trading_blocked=_to_bool(info.get("trading_blocked", False)),
        account_blocked=_to_bool(info.get("account_blocked", False)),
        transfers_blocked=_to_bool(info.get("transfers_blocked", False)),
        trade_suspended_by_user=_to_bool(info.get("trade_suspended_by_user", False)),
        initial_margin=_to_float(info.get("initial_margin")),
        maintenance_margin=_to_float(info.get("maintenance_margin")),
        buying_power=_to_float(info.get("buying_power")),
        equity=_to_float(info.get("equity")),
        captured_at=captured_at,
    )


# ---------------------------------------------------------------------------
# state-machine diff (pure)
# ---------------------------------------------------------------------------


def diff_state(
    prev: Optional[AccountStateSnapshot],
    curr: AccountStateSnapshot,
) -> List[AccountStateEvent]:
    """Pure: compare two snapshots, return events for any actionable transitions.

    Edge-triggered: only state CHANGES emit events. Calling with prev == curr
    yields no events. A first sample (prev=None) emits no events either — we
    only act on transitions to avoid spurious alerts at boot.

    Halt criteria (CRITICAL + halt=True):
      * status leaves ACTIVE / ACCOUNT_UPDATED
      * trading_blocked: False -> True
      * account_blocked: False -> True

    Warn (no halt):
      * trade_suspended_by_user: False -> True   (operator-triggered; surface)
      * transfers_blocked: False -> True         (operational concern)

    Info (no halt):
      * margin first used: prev had 0 margin, curr has > 0
        (just a heads-up; we may legitimately use margin at Stage 2+)
      * status recovers to ACTIVE from a non-OK state (closes a prior incident)
      * any blocked flag clears False<-True
    """
    events: List[AccountStateEvent] = []
    if prev is None:
        return events

    ts = curr.captured_at

    # status departures / recoveries
    if prev.status in _OK_STATUS_VALUES and curr.status not in _OK_STATUS_VALUES:
        events.append(AccountStateEvent(
            event_type="status_left_active",
            severity=AccountStateSeverity.CRITICAL,
            detail=f"status: {prev.status!r} -> {curr.status!r}",
            occurred_at=ts,
            halt=True,
        ))
    elif prev.status not in _OK_STATUS_VALUES and curr.status in _OK_STATUS_VALUES:
        events.append(AccountStateEvent(
            event_type="status_recovered",
            severity=AccountStateSeverity.INFO,
            detail=f"status: {prev.status!r} -> {curr.status!r}",
            occurred_at=ts,
            halt=False,
        ))

    # trading_blocked
    if not prev.trading_blocked and curr.trading_blocked:
        events.append(AccountStateEvent(
            event_type="trading_blocked_set",
            severity=AccountStateSeverity.CRITICAL,
            detail="trading_blocked: False -> True (margin call or account action)",
            occurred_at=ts,
            halt=True,
        ))
    elif prev.trading_blocked and not curr.trading_blocked:
        events.append(AccountStateEvent(
            event_type="trading_blocked_cleared",
            severity=AccountStateSeverity.INFO,
            detail="trading_blocked: True -> False",
            occurred_at=ts,
            halt=False,
        ))

    # account_blocked
    if not prev.account_blocked and curr.account_blocked:
        events.append(AccountStateEvent(
            event_type="account_blocked_set",
            severity=AccountStateSeverity.CRITICAL,
            detail="account_blocked: False -> True",
            occurred_at=ts,
            halt=True,
        ))
    elif prev.account_blocked and not curr.account_blocked:
        events.append(AccountStateEvent(
            event_type="account_blocked_cleared",
            severity=AccountStateSeverity.INFO,
            detail="account_blocked: True -> False",
            occurred_at=ts,
            halt=False,
        ))

    # user-triggered suspend
    if not prev.trade_suspended_by_user and curr.trade_suspended_by_user:
        events.append(AccountStateEvent(
            event_type="trade_suspended_by_user_set",
            severity=AccountStateSeverity.WARN,
            detail="trade_suspended_by_user: False -> True",
            occurred_at=ts,
            halt=False,
        ))

    # transfers_blocked
    if not prev.transfers_blocked and curr.transfers_blocked:
        events.append(AccountStateEvent(
            event_type="transfers_blocked_set",
            severity=AccountStateSeverity.WARN,
            detail="transfers_blocked: False -> True",
            occurred_at=ts,
            halt=False,
        ))

    # margin first used (informational; ORB Pre-0/Stage 0 should never use
    # margin since per-pos cap × max_concurrent < cash. Hitting margin in
    # production is a sign of unexpected sizing.)
    prev_used_margin = prev.initial_margin > 0 or prev.maintenance_margin > 0
    curr_used_margin = curr.initial_margin > 0 or curr.maintenance_margin > 0
    if not prev_used_margin and curr_used_margin:
        events.append(AccountStateEvent(
            event_type="margin_first_used",
            severity=AccountStateSeverity.INFO,
            detail=(
                f"initial_margin=${curr.initial_margin:,.0f} "
                f"maintenance_margin=${curr.maintenance_margin:,.0f}"
            ),
            occurred_at=ts,
            halt=False,
        ))

    return events


def is_halt_worthy(events: List[AccountStateEvent]) -> bool:
    """Convenience: True if any event has halt=True."""
    return any(e.halt for e in events)
