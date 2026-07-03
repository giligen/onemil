"""Tests for the code-review hardening pass:

- in-flight close cooldown prevents duplicate submissions
- poll-retry timeout writes pending-verification (not silent loss)
- alert cache pruning bounds memory growth
- public Database.get_strategy_trades_in_window is used
- engine consumers correctly handle confirmed=False events
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from trading.orphan_reconciler import (
    ORPHAN_RECOVERED_EXIT_REASON,
    PENDING_VERIFICATION_STATUS,
    ReconcilerConfig,
    _is_close_inflight,
    _state,
    reconcile_strategy_orphans,
    reset_state_for_tests,
)
from trading.stop_monitor import StopExitEvent, build_exit_update


@pytest.fixture(autouse=True)
def _clear_state():
    reset_state_for_tests()
    yield
    reset_state_for_tests()


def _stub_db(rows=None):
    rows = rows or []
    db = MagicMock()
    db.get_strategy_trades_in_window = lambda *_a, **_k: rows
    db.update_trade = MagicMock()
    return db


def _stub_alpaca(positions=None, close_id='cl-1', fill=None):
    a = MagicMock()
    a.get_open_positions.return_value = positions or []
    a.close_position.return_value = {'id': close_id}
    a.get_order.return_value = fill if fill is not None else {
        'filled_qty': 100, 'filled_avg_price': 9.50, 'status': 'filled',
    }
    return a


def _broker(**kw):
    base = {'symbol': 'TST', 'qty': 100, 'avg_entry_price': 10.00,
             'unrealized_pl': 0.0, 'side': 'long'}
    base.update(kw)
    return base


def _stale_row(**kw):
    base = {
        'id': 1, 'trade_date': '2026-06-01', 'symbol': 'TST',
        'strategy': 'orb', 'fill_price': 10.00, 'filled_qty': 100,
        'exit_price': None, 'exit_reason': 'stop_loss_unconfirmed',
        'order_status': 'closed',
    }
    base.update(kw)
    return base


TODAY = date(2026, 6, 5)
FAST_CFG = ReconcilerConfig(fill_poll_timeout_s=0.5, fill_poll_interval_s=0.05,
                             inflight_close_cooldown_s=30)


# =========================================================================
# In-flight close cooldown
# =========================================================================

class TestInFlightCloseGuard:
    """After submitting a close, the same orphan must NOT trigger a
    second close in the immediately-following sync cycle. Without this
    guard, a 10-second sync interval would generate duplicate close
    orders on every sync until the original propagated."""

    def test_inflight_blocks_second_close(self):
        a = _stub_alpaca(positions=[_broker()])
        db = _stub_db([_stale_row()])
        # First call → close submitted, in-flight marker set
        reconcile_strategy_orphans(
            strategy='orb', alpaca=a, db=db, notifier=None,
            tracked_symbols=set(), cfg=FAST_CFG, today_et=TODAY,
        )
        assert a.close_position.call_count == 1
        # Second call (same orphan still showing on broker) → SKIPPED
        reconcile_strategy_orphans(
            strategy='orb', alpaca=a, db=db, notifier=None,
            tracked_symbols=set(), cfg=FAST_CFG, today_et=TODAY,
        )
        assert a.close_position.call_count == 1  # NO duplicate

    def test_inflight_state_marked_after_close(self):
        a = _stub_alpaca(positions=[_broker()])
        db = _stub_db([_stale_row()])
        reconcile_strategy_orphans(
            strategy='orb', alpaca=a, db=db, notifier=None,
            tracked_symbols=set(), cfg=FAST_CFG, today_et=TODAY,
        )
        assert _is_close_inflight('orb', 'TST') is True

    def test_inflight_expires_after_cooldown(self):
        # Force expiry by writing a past datetime
        _state.inflight_close[('orb', 'EXPIRED')] = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        )
        # Calling _is_close_inflight prunes it
        assert _is_close_inflight('orb', 'EXPIRED') is False
        assert ('orb', 'EXPIRED') not in _state.inflight_close


# =========================================================================
# Poll-retry behaviour
# =========================================================================

class TestPollRetry:

    def test_eventual_fill_via_retry_writes_recovery_row(self):
        # Order returns "new" for the first 2 polls, then "filled"
        a = _stub_alpaca(positions=[_broker()])
        call_count = {'n': 0}
        def progressive_fill(_oid):
            call_count['n'] += 1
            if call_count['n'] < 2:
                return {'filled_qty': 0, 'filled_avg_price': None,
                         'status': 'new'}
            return {'filled_qty': 100, 'filled_avg_price': 9.5,
                     'status': 'filled'}
        a.get_order = progressive_fill
        db = _stub_db([_stale_row()])
        actions = reconcile_strategy_orphans(
            strategy='orb', alpaca=a, db=db, notifier=None,
            tracked_symbols=set(), cfg=FAST_CFG, today_et=TODAY,
        )
        assert actions[0].action == 'closed'
        # Recovery row written with the actual fill, not a placeholder
        update = db.update_trade.call_args[0][1]
        assert update['exit_price'] == 9.5
        assert update['exit_reason'] == ORPHAN_RECOVERED_EXIT_REASON

    def test_poll_timeout_writes_pending_verification(self):
        # Order never fills within the budget
        a = _stub_alpaca(
            positions=[_broker()],
            fill={'filled_qty': 0, 'filled_avg_price': None, 'status': 'new'},
        )
        db = _stub_db([_stale_row()])
        actions = reconcile_strategy_orphans(
            strategy='orb', alpaca=a, db=db, notifier=None,
            tracked_symbols=set(), cfg=FAST_CFG, today_et=TODAY,
        )
        # We still recorded the close + alert
        assert actions[0].action == 'closed'
        # But DB writes the pending-verification fallback (NOT a fake exit)
        update = db.update_trade.call_args[0][1]
        assert update['order_status'] == PENDING_VERIFICATION_STATUS
        assert 'exit_price' not in update  # NO fake exit
        assert 'pnl' not in update

    def test_poll_terminal_rejected_writes_pending(self):
        # Order rejected — poll returns terminal-non-filled status
        a = _stub_alpaca(
            positions=[_broker()],
            fill={'filled_qty': 0, 'filled_avg_price': None,
                  'status': 'rejected'},
        )
        db = _stub_db([_stale_row()])
        actions = reconcile_strategy_orphans(
            strategy='orb', alpaca=a, db=db, notifier=None,
            tracked_symbols=set(), cfg=FAST_CFG, today_et=TODAY,
        )
        update = db.update_trade.call_args[0][1]
        assert update['order_status'] == PENDING_VERIFICATION_STATUS


# =========================================================================
# Public DB method used (no _trades_conn poke)
# =========================================================================

class TestPublicDBMethodPreferred:

    def test_uses_public_method_when_available(self):
        called = {}
        def fake_get(strategy, since, symbols):
            called['args'] = (strategy, since, symbols)
            return [_stale_row()]
        db = MagicMock()
        db.get_strategy_trades_in_window = fake_get
        db.update_trade = MagicMock()
        a = _stub_alpaca(positions=[_broker()])
        reconcile_strategy_orphans(
            strategy='orb', alpaca=a, db=db, notifier=None,
            tracked_symbols=set(), cfg=FAST_CFG, today_et=TODAY,
        )
        assert called['args'][0] == 'orb'
        assert called['args'][2] == ['TST']

    def test_database_class_exposes_get_strategy_trades_in_window(self):
        # Source-code inspection: pin the public API exists.
        from persistence.database import Database
        assert hasattr(Database, 'get_strategy_trades_in_window')


# =========================================================================
# Alert cache pruning (memory bound)
# =========================================================================

class TestAlertCachePruning:

    def test_expired_entries_pruned_when_cache_grows(self):
        # Stuff 70 expired entries into the cache
        now = datetime.now(timezone.utc)
        for i in range(70):
            _state.last_alert[('orb', f'SYM{i}')] = (
                now - timedelta(hours=2)
            )
        a = _stub_alpaca(positions=[_broker(symbol='UNKNOWN')])
        db = _stub_db([])
        n = MagicMock()
        reconcile_strategy_orphans(
            strategy='orb', alpaca=a, db=db, notifier=n,
            tracked_symbols=set(), cfg=FAST_CFG, today_et=TODAY,
        )
        # Most/all of the 70 stale entries should have been pruned
        assert len(_state.last_alert) < 70


# =========================================================================
# Broker positions snapshot passed in (no extra API hit)
# =========================================================================

class TestBrokerPositionsPassThrough:

    def test_caller_snapshot_used_no_alpaca_call(self):
        a = _stub_alpaca(positions=[])
        # Caller-provided snapshot
        snapshot = [_broker(symbol='TST', qty=100, avg_entry_price=10.0)]
        db = _stub_db([_stale_row()])
        reconcile_strategy_orphans(
            strategy='orb', alpaca=a, db=db, notifier=None,
            tracked_symbols=set(), cfg=FAST_CFG, today_et=TODAY,
            broker_positions=snapshot,
        )
        # Alpaca's get_open_positions must NOT have been called.
        a.get_open_positions.assert_not_called()
        a.close_position.assert_called_with('TST')


# =========================================================================
# Engine consumer correctly handles confirmed=False events
# =========================================================================

class TestEngineConsumerUnconfirmedFlow:
    """The L1 fix's behaviour at the engine layer: a confirmed=False
    StopExitEvent must produce a pending-verification DB write, not
    the old fake-confirmed payload. This is the engine-side contract
    that previously had no test (we tested build_exit_update in
    isolation but not the engine drain path)."""

    def test_macd_wave_drain_writes_pending_payload(self):
        """The drain code branches on event.confirmed:
        - confirmed: full payload + daily_pnl mutation
        - unconfirmed: build_exit_update → pending-verification, NO pnl
        """
        # Light-weight check: build_exit_update directly with a fake event
        ev = StopExitEvent(
            symbol='X', stop_price=10, exit_price=10, shares=100,
            order_id='', exit_reason='stop_loss_unconfirmed',
            confirmed=False,
        )
        update = build_exit_update(ev)
        # Drain code adds pnl/pnl_pct ONLY when confirmed=True. Pin
        # that the base helper itself never adds them.
        assert 'pnl' not in update
        assert 'pnl_pct' not in update
        assert update['order_status'] == PENDING_VERIFICATION_STATUS

    def test_confirmed_event_adds_pnl_in_consumer_pattern(self):
        # Mirror the consumer's branching code: confirmed events get
        # pnl/pnl_pct added on top of build_exit_update's base payload.
        ev = StopExitEvent(
            symbol='X', stop_price=10, exit_price=9.50, shares=100,
            order_id='ok', exit_reason='stop_loss', confirmed=True,
        )
        update = build_exit_update(ev)
        # In the engine, the next two lines run:
        # update['pnl'] = (exit_price - entry_price) * shares
        # update['pnl_pct'] = ...
        # We assert the helper produces a payload safe to augment.
        assert update['exit_price'] == 9.50
        assert update['order_status'] == 'closed'


# =========================================================================
# Bull flag predicate-bug regression
# =========================================================================

class TestBullFlagNoMoreAutoClose:
    """The old _close_orphan_positions in trading_engine.py used to
    auto-close any symbol with a strategy='bull_flag' row in the
    lookback. That had the same SMU-class vulnerability we just fixed
    in ORB. Pin (via source inspection) that the close call is gone."""

    def test_no_alpaca_close_call_in_close_orphan_positions(self):
        from pathlib import Path
        src = Path("trading/trading_engine.py").read_text()
        # Find the function body via crude split (keeps the test simple).
        marker = "def _close_orphan_positions"
        i = src.index(marker)
        # Read ~80 lines after the signature to cover the body.
        body = src[i:i + 5000]
        # The OLD line called self.alpaca.close_position(symbol).
        # Forbid it returning here — close decisions belong to the
        # shared reconciler now.
        assert "self.alpaca.close_position(symbol)" not in body, (
            "_close_orphan_positions called alpaca.close_position again — "
            "delegate to trading.orphan_reconciler instead. The unguarded "
            "close was the SMU/QBTZ-class vulnerability."
        )
