"""
Unit tests for trading.account_state_monitor.

Edge-triggered state machine: only transitions emit events. First snapshot
emits nothing (no prior to compare to). Steady state emits nothing.
"""

from datetime import datetime, timezone

import pytest

from trading.account_state_monitor import (
    AccountStateEvent,
    AccountStateSeverity,
    AccountStateSnapshot,
    diff_state,
    is_halt_worthy,
    snapshot_from_account_dict,
)


def _t(s: str = "2026-06-05T13:30:00+00:00") -> datetime:
    return datetime.fromisoformat(s)


def _snap(
    *,
    status="ACTIVE",
    trading_blocked=False,
    account_blocked=False,
    transfers_blocked=False,
    trade_suspended_by_user=False,
    initial_margin=0.0,
    maintenance_margin=0.0,
    buying_power=300000.0,
    equity=79000.0,
    captured_at=None,
) -> AccountStateSnapshot:
    return AccountStateSnapshot(
        status=status,
        trading_blocked=trading_blocked,
        account_blocked=account_blocked,
        transfers_blocked=transfers_blocked,
        trade_suspended_by_user=trade_suspended_by_user,
        initial_margin=initial_margin,
        maintenance_margin=maintenance_margin,
        buying_power=buying_power,
        equity=equity,
        captured_at=captured_at or _t(),
    )


# ---------------------------------------------------------------------------
# snapshot_from_account_dict
# ---------------------------------------------------------------------------


class TestSnapshotParsing:
    def test_parses_canonical_dict(self):
        info = {
            "status": "ACTIVE",
            "trading_blocked": False,
            "account_blocked": False,
            "transfers_blocked": False,
            "trade_suspended_by_user": False,
            "initial_margin": "0",
            "maintenance_margin": "0",
            "buying_power": "317957.36",
            "equity": "79489.34",
        }
        snap = snapshot_from_account_dict(info, now=_t())
        assert snap is not None
        assert snap.status == "ACTIVE"
        assert snap.trading_blocked is False
        assert snap.buying_power == 317957.36
        assert snap.equity == 79489.34
        assert snap.initial_margin == 0.0
        assert snap.captured_at == _t()

    def test_none_info_returns_none(self):
        assert snapshot_from_account_dict(None) is None
        assert snapshot_from_account_dict({}) is None

    def test_handles_enum_value_for_status(self):
        class _Enum:
            value = "ACTIVE"
        info = {"status": _Enum()}
        snap = snapshot_from_account_dict(info)
        assert snap.status == "ACTIVE"

    def test_handles_bool_strings(self):
        info = {"status": "ACTIVE", "trading_blocked": "true", "account_blocked": "false"}
        snap = snapshot_from_account_dict(info)
        assert snap.trading_blocked is True
        assert snap.account_blocked is False

    def test_handles_missing_margin_fields(self):
        snap = snapshot_from_account_dict({"status": "ACTIVE"})
        assert snap.initial_margin == 0.0
        assert snap.maintenance_margin == 0.0

    def test_handles_non_numeric_margin_safely(self):
        snap = snapshot_from_account_dict({
            "status": "ACTIVE", "initial_margin": "n/a",
        })
        assert snap.initial_margin == 0.0


# ---------------------------------------------------------------------------
# diff_state — steady-state / first-call semantics
# ---------------------------------------------------------------------------


class TestSteadyState:
    def test_first_snapshot_returns_no_events(self):
        snap = _snap()
        events = diff_state(prev=None, curr=snap)
        assert events == []

    def test_unchanged_state_returns_no_events(self):
        s1 = _snap()
        s2 = _snap()
        assert diff_state(s1, s2) == []


# ---------------------------------------------------------------------------
# diff_state — halt-worthy transitions
# ---------------------------------------------------------------------------


class TestHaltWorthyTransitions:
    """Transitions that MUST stop the bot from opening new positions."""

    def test_trading_blocked_set_is_critical_halt(self):
        prev = _snap()
        curr = _snap(trading_blocked=True)
        events = diff_state(prev, curr)
        assert len(events) == 1
        assert events[0].event_type == "trading_blocked_set"
        assert events[0].severity == AccountStateSeverity.CRITICAL
        assert events[0].halt is True
        assert is_halt_worthy(events) is True

    def test_account_blocked_set_is_critical_halt(self):
        prev = _snap()
        curr = _snap(account_blocked=True)
        events = diff_state(prev, curr)
        assert any(e.event_type == "account_blocked_set" and e.halt for e in events)

    def test_status_leaves_active_is_critical_halt(self):
        prev = _snap(status="ACTIVE")
        curr = _snap(status="ACCOUNT_RESTRICTED")
        events = diff_state(prev, curr)
        assert any(e.event_type == "status_left_active" and e.halt for e in events)
        assert is_halt_worthy(events) is True

    def test_account_updated_is_treated_as_ok(self):
        """ACCOUNT_UPDATED is a transient state Alpaca uses during config
        changes — not a halt-worthy transition."""
        prev = _snap(status="ACTIVE")
        curr = _snap(status="ACCOUNT_UPDATED")
        events = diff_state(prev, curr)
        # Neither status is the FROM/TO that triggers status_left_active
        assert not any(e.event_type == "status_left_active" for e in events)


# ---------------------------------------------------------------------------
# diff_state — warn-only transitions
# ---------------------------------------------------------------------------


class TestWarnTransitions:
    def test_trade_suspended_by_user_warn_no_halt(self):
        prev = _snap()
        curr = _snap(trade_suspended_by_user=True)
        events = diff_state(prev, curr)
        assert len(events) == 1
        assert events[0].event_type == "trade_suspended_by_user_set"
        assert events[0].severity == AccountStateSeverity.WARN
        assert events[0].halt is False
        assert is_halt_worthy(events) is False

    def test_transfers_blocked_warn_no_halt(self):
        prev = _snap()
        curr = _snap(transfers_blocked=True)
        events = diff_state(prev, curr)
        assert any(
            e.event_type == "transfers_blocked_set"
            and e.severity == AccountStateSeverity.WARN
            and e.halt is False
            for e in events
        )


# ---------------------------------------------------------------------------
# diff_state — info-only transitions (recoveries + margin used)
# ---------------------------------------------------------------------------


class TestInfoTransitions:
    def test_status_recovery_emits_info(self):
        prev = _snap(status="ACCOUNT_RESTRICTED")
        curr = _snap(status="ACTIVE")
        events = diff_state(prev, curr)
        assert any(
            e.event_type == "status_recovered"
            and e.severity == AccountStateSeverity.INFO
            and e.halt is False
            for e in events
        )

    def test_trading_blocked_clear_emits_info(self):
        prev = _snap(trading_blocked=True)
        curr = _snap(trading_blocked=False)
        events = diff_state(prev, curr)
        assert any(e.event_type == "trading_blocked_cleared" for e in events)

    def test_margin_first_used_emits_info(self):
        prev = _snap(initial_margin=0.0, maintenance_margin=0.0)
        curr = _snap(initial_margin=12_000.0, maintenance_margin=4_800.0)
        events = diff_state(prev, curr)
        assert any(
            e.event_type == "margin_first_used" and not e.halt
            for e in events
        )

    def test_margin_unchanged_emits_nothing(self):
        prev = _snap(initial_margin=10_000.0, maintenance_margin=4_000.0)
        curr = _snap(initial_margin=11_000.0, maintenance_margin=4_400.0)
        events = diff_state(prev, curr)
        # only the first-time transition emits; subsequent changes don't
        assert not any(e.event_type == "margin_first_used" for e in events)


# ---------------------------------------------------------------------------
# diff_state — multiple simultaneous transitions
# ---------------------------------------------------------------------------


class TestMultiTransition:
    def test_blocked_and_status_emit_two_events(self):
        prev = _snap()
        curr = _snap(trading_blocked=True, status="ACCOUNT_RESTRICTED")
        events = diff_state(prev, curr)
        types = {e.event_type for e in events}
        assert "trading_blocked_set" in types
        assert "status_left_active" in types
        assert is_halt_worthy(events) is True
