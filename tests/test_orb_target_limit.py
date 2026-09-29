"""ORB exit: rest the target as a real limit at the broker.

Spec: docs/orb_target_limit_spec_20260929.md. Design: trading/orb_target_limit.py
module docstring. Covers:
  - trading/orb_target_limit.py primitives (reprice_target, cancel_resting_target,
    reconcile_orphan_targets) in isolation, MagicMock(spec=AlpacaClient) only.
  - ORBEngine._fire_touchgo_exit / _rest_touchgo_target: flag OFF parity (byte-
    identical call sequence to the pre-existing chase path), flag ON resting,
    fallback on failure, exit-fill telemetry (_handle_exit_event).
  - StopMonitor.mark_target_resting / book_target_rested_fill, and the
    cancel-before-stop race hook inside _execute_stop_exit.
  - EOD genericity: _cancel_symbol_open_orders cancels whatever is open for
    the symbol regardless of which specific leg id is live (rule 5 relies on
    this EXISTING, unchanged behaviour).
  - Boot reconciliation orphan cancellation (reconcile_orphan_targets, already
    covered under primitives; this file does not re-drive the much larger
    sync_positions function end-to-end — see the final report for that scope
    note).
  - Integration: a real ORBEngine + real StopMonitor + real Database driven
    by an order-stream fill event in the shape OrderStreamWatcher.get_status
    returns, through _check_exits_locked -> _handle_exit_event -> the DB row.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.exit_reasons import ExitBranch, ExitReason
from trading.order_stream import OrderStreamWatcher
from trading.orb_engine import ORBEngine, OpenPosition
from trading.orb_target_limit import (
    CANCEL_ACK_WAIT_S,
    CancelOutcome,
    CancelResult,
    REPLACE_RATE_LIMIT_S,
    RestOutcome,
    cancel_resting_target,
    reconcile_orphan_targets,
    reprice_target,
    target_client_order_id,
)
from trading.stop_monitor import StopMonitor, WatchEntry

REPO = Path(__file__).parent.parent


# =============================================================================
# trading/orb_target_limit.py primitives — pure unit tests
# =============================================================================

class TestTargetClientOrderId:
    def test_format(self):
        as_of = datetime(2026, 9, 29, 14, 0, tzinfo=timezone.utc)
        assert target_client_order_id('KOLD', as_of) == 'orb-tp-KOLD-20260929'

    def test_defaults_to_now(self):
        coid = target_client_order_id('PLYX')
        assert coid.startswith('orb-tp-PLYX-')
        assert len(coid) == len('orb-tp-PLYX-') + 8


class TestRepriceTarget:
    def test_rested_on_success(self):
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.replace_order_limit_price.return_value = {'id': 'new-leg-1', 'status': 'accepted'}
        result = reprice_target(alpaca, 'KOLD', 'old-leg-1', 9.50, last_replace_ts=0.0, force_first=True)
        assert result.outcome == RestOutcome.RESTED
        assert result.new_leg_id == 'new-leg-1'
        alpaca.replace_order_limit_price.assert_called_once()
        args, kwargs = alpaca.replace_order_limit_price.call_args
        assert args[0] == 'old-leg-1'
        assert args[1] == 9.50
        assert kwargs['client_order_id'].startswith('orb-tp-KOLD-')

    def test_no_leg_when_tp_leg_id_empty(self):
        alpaca = MagicMock(spec=AlpacaClient)
        result = reprice_target(alpaca, 'KOLD', '', 9.50, last_replace_ts=0.0, force_first=True)
        assert result.outcome == RestOutcome.NO_LEG
        alpaca.replace_order_limit_price.assert_not_called()

    def test_rate_limited_blocks_rapid_move(self):
        alpaca = MagicMock(spec=AlpacaClient)
        now = 1_000_000.0
        result = reprice_target(
            alpaca, 'KOLD', 'leg-1', 9.60,
            last_replace_ts=now - 2.0, now_ts=now, force_first=False,
        )
        assert result.outcome == RestOutcome.RATE_LIMITED
        alpaca.replace_order_limit_price.assert_not_called()

    def test_move_allowed_after_rate_limit_window(self):
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.replace_order_limit_price.return_value = {'id': 'new-leg-2'}
        now = 1_000_000.0
        result = reprice_target(
            alpaca, 'KOLD', 'leg-1', 9.60,
            last_replace_ts=now - REPLACE_RATE_LIMIT_S - 0.1, now_ts=now, force_first=False,
        )
        assert result.outcome == RestOutcome.RESTED

    def test_force_first_bypasses_rate_limit(self):
        """Rule 1's first rest must never be blocked by the rate limit that
        exists only to throttle rule-4 MOVES."""
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.replace_order_limit_price.return_value = {'id': 'new-leg-3'}
        now = 1_000_000.0
        result = reprice_target(
            alpaca, 'KOLD', 'leg-1', 9.60,
            last_replace_ts=now - 0.5, now_ts=now, force_first=True,
        )
        assert result.outcome == RestOutcome.RESTED

    def test_broker_exception_falls_back(self):
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.replace_order_limit_price.side_effect = RuntimeError("422 not cancelable")
        result = reprice_target(alpaca, 'KOLD', 'leg-1', 9.50, last_replace_ts=0.0, force_first=True)
        assert result.outcome == RestOutcome.FAILED

    def test_empty_id_response_treated_as_failed(self):
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.replace_order_limit_price.return_value = {'id': '', 'status': 'rejected'}
        result = reprice_target(alpaca, 'KOLD', 'leg-1', 9.50, last_replace_ts=0.0, force_first=True)
        assert result.outcome == RestOutcome.FAILED


class TestCancelRestingTarget:
    """2026-09-29 follow-up money-defect fix: cancel_order()==True does NOT
    prove zero fill (a partially-filled order cancels its remainder just as
    cleanly as an untouched one) — get_order is now ALWAYS consulted, and
    classification is driven entirely by filled_qty vs requested_qty."""

    def test_clean_cancel_zero_filled(self):
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.cancel_order.return_value = True
        alpaca.get_order.return_value = {'status': 'canceled', 'filled_qty': 0}
        result = cancel_resting_target(alpaca, 'KOLD', 'leg-1', 'stop_trigger', requested_qty=200)
        assert result.outcome == CancelOutcome.CANCELLED
        alpaca.get_order.assert_called_once()  # always verified now, never trusted blind

    def test_true_cancel_result_with_a_hidden_partial_is_not_a_clean_cancel(self):
        """The exact money defect: cancel_order() returns True (it DID
        successfully cancel the remaining open qty) but 60 shares had
        already filled before that — must classify as PARTIALLY_FILLED,
        never as a clean CANCELLED."""
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.cancel_order.return_value = True
        alpaca.get_order.return_value = {
            'status': 'canceled', 'filled_qty': 60, 'filled_avg_price': 9.55,
        }
        result = cancel_resting_target(alpaca, 'KOLD', 'leg-1', 'stop_trigger', requested_qty=200)
        assert result.outcome == CancelOutcome.PARTIALLY_FILLED
        assert result.filled_qty == 60

    def test_already_filled_race_full(self):
        """Rule 2: filled_qty >= requested_qty — the WHOLE order filled,
        position is flat, not a cancel."""
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.cancel_order.return_value = False
        alpaca.get_order.return_value = {
            'status': 'filled', 'filled_qty': 200, 'filled_avg_price': 9.55,
        }
        result = cancel_resting_target(alpaca, 'KOLD', 'leg-1', 'stop_trigger', requested_qty=200)
        assert result.outcome == CancelOutcome.ALREADY_FILLED
        assert result.fill_price == pytest.approx(9.55)
        assert result.filled_qty == 200

    def test_partially_filled_race(self):
        """Rule 6: filled_qty < requested_qty -> PARTIALLY_FILLED, reporting
        the ACTUAL filled_qty, not the full requested size."""
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.cancel_order.return_value = False
        alpaca.get_order.return_value = {
            'status': 'canceled', 'filled_qty': 60, 'filled_avg_price': 9.55,
        }
        result = cancel_resting_target(alpaca, 'KOLD', 'leg-1', 'stop_trigger', requested_qty=200)
        assert result.outcome == CancelOutcome.PARTIALLY_FILLED
        assert result.filled_qty == 60
        assert result.fill_price == pytest.approx(9.55)

    def test_no_requested_qty_treats_any_fill_as_full(self):
        """Callers that don't pass requested_qty (0 = unknown) get the
        conservative full-close classification — never silently drop a fill
        into an unhandled bucket."""
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.cancel_order.return_value = False
        alpaca.get_order.return_value = {
            'status': 'filled', 'filled_qty': 60, 'filled_avg_price': 9.55,
        }
        result = cancel_resting_target(alpaca, 'KOLD', 'leg-1', 'stop_trigger')
        assert result.outcome == CancelOutcome.ALREADY_FILLED
        assert result.filled_qty == 60

    def test_cancel_false_but_not_filled_treated_as_cancelled(self):
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.cancel_order.return_value = False
        alpaca.get_order.return_value = {'status': 'canceled', 'filled_qty': 0}
        result = cancel_resting_target(alpaca, 'KOLD', 'leg-1', 'eod', requested_qty=200)
        assert result.outcome == CancelOutcome.CANCELLED

    def test_get_order_failure_is_unresolved_error(self):
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.cancel_order.return_value = False
        alpaca.get_order.side_effect = RuntimeError("network error")
        result = cancel_resting_target(alpaca, 'KOLD', 'leg-1', 'stop_trigger', requested_qty=200)
        assert result.outcome == CancelOutcome.ERROR

    def test_no_leg(self):
        alpaca = MagicMock(spec=AlpacaClient)
        result = cancel_resting_target(alpaca, 'KOLD', '', 'eod', requested_qty=200)
        assert result.outcome == CancelOutcome.NO_LEG
        alpaca.cancel_order.assert_not_called()


class TestReconcileOrphanTargets:
    def test_orphan_cancelled(self):
        alpaca = MagicMock(spec=AlpacaClient)
        orders = [{'id': 'o1', 'symbol': 'KOLD', 'client_order_id': 'orb-tp-KOLD-20260929'}]
        result = reconcile_orphan_targets(alpaca, open_symbols_with_target=set(), all_open_orders=orders)
        assert result['orphans_cancelled'] == ['orb-tp-KOLD-20260929']
        alpaca.cancel_order.assert_called_once_with('o1')

    def test_matched_symbol_not_orphan(self):
        alpaca = MagicMock(spec=AlpacaClient)
        orders = [{'id': 'o1', 'symbol': 'KOLD', 'client_order_id': 'orb-tp-KOLD-20260929'}]
        result = reconcile_orphan_targets(alpaca, open_symbols_with_target={'KOLD'}, all_open_orders=orders)
        assert result['orphans_cancelled'] == []
        alpaca.cancel_order.assert_not_called()

    def test_non_target_orders_ignored(self):
        alpaca = MagicMock(spec=AlpacaClient)
        orders = [{'id': 'o1', 'symbol': 'PLYX', 'client_order_id': 'orb-entry-PLYX-1'}]
        result = reconcile_orphan_targets(alpaca, open_symbols_with_target=set(), all_open_orders=orders)
        assert result == {'orphans_cancelled': [], 'orphans_cancel_failed': []}
        alpaca.cancel_order.assert_not_called()

    def test_cancel_failure_recorded(self):
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.cancel_order.side_effect = RuntimeError("404")
        orders = [{'id': 'o1', 'symbol': 'KOLD', 'client_order_id': 'orb-tp-KOLD-20260929'}]
        result = reconcile_orphan_targets(alpaca, open_symbols_with_target=set(), all_open_orders=orders)
        assert result['orphans_cancel_failed'] == ['orb-tp-KOLD-20260929']


# =============================================================================
# ORBEngine wiring — _fire_touchgo_exit / _rest_touchgo_target / telemetry
# =============================================================================

@pytest.fixture
def orb_cfg():
    with open(REPO / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    cfg.setdefault('exit', {})['target_resting_limit'] = False  # explicit default; flipped per-test
    return cfg


@pytest.fixture
def mock_alpaca():
    c = MagicMock(spec=AlpacaClient)
    c.get_open_positions.return_value = []
    c.get_account_info.return_value = {'buying_power': 100_000.0}
    c.replace_order_limit_price.return_value = {'id': 'new-tp-leg', 'status': 'accepted'}
    return c


@pytest.fixture
def mock_db():
    db = MagicMock(spec=Database)
    db.save_trade.return_value = 100
    db.get_open_trades.return_value = []
    db.update_trade.return_value = True
    return db


@pytest.fixture
def mock_stop_monitor():
    sm = MagicMock(spec=StopMonitor)
    sm.drain_exit_events.return_value = []
    sm.force_exit.return_value = True
    sm.mark_target_resting.return_value = True
    return sm


@pytest.fixture
def engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor):
    return ORBEngine(
        alpaca_client=mock_alpaca, db=mock_db,
        stop_monitor=mock_stop_monitor, config=orb_cfg,
    )


def _seed_position(engine, symbol='KOLD', tp_leg_id='orig-tp-leg'):
    pos = OpenPosition(
        symbol=symbol, entry_price=9.00, stop_price=8.50, shares=200,
        trade_id=7, order_id='', entry_time=datetime.now(timezone.utc),
        range_high=9.00, range_low=8.50, lock_arm_at_r=1.75, lock_stop_r=0.5,
        composite_score=0.5, quintile='Q3', tp_leg_id=tp_leg_id, sl_leg_id='orig-sl-leg',
    )
    engine.open_positions[symbol] = pos
    return pos


class TestFireTouchgoExitParity:
    """Flag OFF must be byte-identical to the pre-existing chase-and-sell
    path: same StopMonitor.force_exit call, and replace_order_limit_price is
    NEVER touched."""

    def test_flag_off_calls_force_exit_exactly_as_before(self, engine, mock_alpaca, mock_stop_monitor):
        assert engine.target_resting_limit_enabled is False
        pos = _seed_position(engine)

        engine._fire_touchgo_exit(pos, reason=ExitReason.TAG_BB.value, exit_price=9.10, detail='bb_close_pos=0.20')

        mock_alpaca.replace_order_limit_price.assert_not_called()
        mock_stop_monitor.force_exit.assert_called_once_with(
            symbol='KOLD', reason=ExitReason.TAG_BB.value, limit_price=9.10,
        )

    def test_flag_off_pos_target_fields_untouched(self, engine, mock_stop_monitor):
        pos = _seed_position(engine)
        engine._fire_touchgo_exit(pos, reason=ExitReason.TAG_B1.value, exit_price=8.80, detail='b1_revert=0.9R')
        assert pos.target_resting is False
        assert pos.tp_leg_id == 'orig-tp-leg'  # untouched — never repriced


class TestFireTouchgoExitResting:
    def test_flag_on_rests_instead_of_chasing(self, engine, mock_alpaca, mock_stop_monitor):
        engine.target_resting_limit_enabled = True
        pos = _seed_position(engine)

        engine._fire_touchgo_exit(pos, reason=ExitReason.TAG_BB.value, exit_price=9.10, detail='bb_close_pos=0.20')

        mock_alpaca.replace_order_limit_price.assert_called_once()
        mock_stop_monitor.force_exit.assert_not_called()  # never both (spec: no two resting sells)
        assert pos.target_resting is True
        assert pos.target_price == pytest.approx(9.10)
        assert pos.tp_leg_id == 'new-tp-leg'  # repointed to the NEW leg id
        mock_stop_monitor.mark_target_resting.assert_called_once()
        call_args = mock_stop_monitor.mark_target_resting.call_args.args
        assert call_args[0] == 'KOLD'
        assert call_args[1] == 'new-tp-leg'

    def test_flag_on_falls_back_when_no_leg(self, engine, mock_alpaca, mock_stop_monitor):
        """rule 1's explicit fallback: no tp_leg_id -> chase path, unchanged."""
        engine.target_resting_limit_enabled = True
        pos = _seed_position(engine, tp_leg_id='')

        engine._fire_touchgo_exit(pos, reason=ExitReason.TAG_BB.value, exit_price=9.10, detail='bb_close_pos=0.20')

        mock_alpaca.replace_order_limit_price.assert_not_called()
        mock_stop_monitor.force_exit.assert_called_once_with(
            symbol='KOLD', reason=ExitReason.TAG_BB.value, limit_price=9.10,
        )
        assert pos.target_resting is False

    def test_flag_on_falls_back_when_broker_rejects(self, engine, mock_alpaca, mock_stop_monitor):
        engine.target_resting_limit_enabled = True
        mock_alpaca.replace_order_limit_price.side_effect = RuntimeError("reject")
        pos = _seed_position(engine)

        engine._fire_touchgo_exit(pos, reason=ExitReason.TAG_BB.value, exit_price=9.10, detail='bb_close_pos=0.20')

        mock_stop_monitor.force_exit.assert_called_once()
        assert pos.target_resting is False

    def test_second_call_is_a_rate_limited_move_not_a_first_rest(self, engine, mock_alpaca, mock_stop_monitor):
        """Once already resting, a second _fire_touchgo_exit call within the
        5s window must be rate-limited (rule 4), not treated as a fresh rest."""
        engine.target_resting_limit_enabled = True
        pos = _seed_position(engine)
        engine._fire_touchgo_exit(pos, reason=ExitReason.TAG_BB.value, exit_price=9.10, detail='first')
        mock_alpaca.replace_order_limit_price.reset_mock()

        rested = engine._rest_touchgo_target(pos, 9.20, reason='move', detail='second')

        assert rested is False
        mock_alpaca.replace_order_limit_price.assert_not_called()


class TestExitFillLatencyTelemetry:
    """_handle_exit_event's target_rested augmentation (mirrors the HOD 9/28
    resting-order telemetry addition onto the shared exit_fill_latency_ms
    column)."""

    @dataclass
    class _FakeExitEvent:
        symbol: str
        exit_price: float
        exit_reason: str
        exit_limit_price: float = 0.0
        pricing_method: str = 'target_rested_limit'
        strategy: str = 'orb'
        filled_qty: int = 0
        confirmed: bool = True

    def test_resting_seconds_recorded(self, engine, mock_db):
        pos = _seed_position(engine)
        pos.target_resting = True
        pos.target_price = 9.10
        pos.target_rested_at = datetime.now(timezone.utc) - timedelta(seconds=45)

        ev = self._FakeExitEvent(
            symbol='KOLD', exit_price=9.10, exit_reason=ExitReason.TARGET_RESTED.value,
            exit_limit_price=9.10, filled_qty=200,
        )
        engine._handle_exit_event(ev)

        merged = {}
        for call in mock_db.update_trade.call_args_list:
            args, kwargs = call
            merged.update(args[1] if len(args) >= 2 else kwargs.get('updates', {}))
        assert merged['exit_reason'] == ExitReason.TARGET_RESTED.value
        assert merged['exit_price'] == pytest.approx(9.10)
        assert merged['exit_limit_price'] == pytest.approx(9.10)
        assert merged['exit_fill_latency_ms'] == pytest.approx(45_000, rel=0.05)

    def test_missing_rested_at_logs_warning_not_crash(self, engine, mock_db, caplog):
        pos = _seed_position(engine)
        pos.target_rested_at = None
        ev = self._FakeExitEvent(symbol='KOLD', exit_price=9.10, exit_reason=ExitReason.TARGET_RESTED.value)
        engine._handle_exit_event(ev)  # must not raise
        merged = {}
        for call in mock_db.update_trade.call_args_list:
            args, kwargs = call
            merged.update(args[1] if len(args) >= 2 else kwargs.get('updates', {}))
        assert 'exit_fill_latency_ms' not in merged


class TestPollTargetFills:
    def test_fill_detected_and_booked(self, engine, mock_stop_monitor):
        pos = _seed_position(engine)
        pos.target_resting = True
        pos.target_price = 9.10
        engine.order_stream = MagicMock(spec=OrderStreamWatcher)
        engine.order_stream.get_status.return_value = {
            'status': 'filled', 'filled_qty': 200, 'filled_avg_price': 9.11,
        }
        mock_stop_monitor.book_target_rested_fill.return_value = True

        engine._poll_target_fills()

        mock_stop_monitor.book_target_rested_fill.assert_called_once_with(
            'KOLD', 200, 9.11, 9.10, pos.target_rested_at,
        )

    def test_non_target_positions_skipped(self, engine, mock_stop_monitor):
        _seed_position(engine)  # target_resting defaults False
        engine.order_stream = MagicMock(spec=OrderStreamWatcher)
        engine._poll_target_fills()
        engine.order_stream.get_status.assert_not_called()

    def test_not_yet_filled_no_op(self, engine, mock_stop_monitor):
        pos = _seed_position(engine)
        pos.target_resting = True
        engine.order_stream = MagicMock(spec=OrderStreamWatcher)
        engine.order_stream.get_status.return_value = {'status': 'new', 'filled_qty': 0}
        engine._poll_target_fills()
        mock_stop_monitor.book_target_rested_fill.assert_not_called()


class TestEodCancelIsGeneric:
    """Rule 5 (EOD cancels the resting TP first) relies on the EXISTING,
    UNCHANGED _cancel_symbol_open_orders being generic — it discovers open
    orders fresh from Alpaca and cancels every one, regardless of which
    specific leg id (safety-net or our repriced target) is currently live.
    This is a regression test for that property, not new behaviour."""

    def test_cancels_whatever_is_open_for_the_symbol(self, engine, mock_alpaca):
        fake_order = MagicMock()
        fake_order.id = 'whatever-leg-is-currently-live'
        mock_alpaca.trading_client = MagicMock()
        mock_alpaca.trading_client.get_orders.return_value = [fake_order]

        n = engine._cancel_symbol_open_orders('KOLD')

        assert n == 1
        mock_alpaca.trading_client.cancel_order_by_id.assert_called_once_with(
            'whatever-leg-is-currently-live'
        )


# =============================================================================
# StopMonitor: mark_target_resting / book_target_rested_fill / cancel-before-stop
# =============================================================================

@pytest.fixture
def sm_mock_alpaca():
    client = MagicMock(spec=AlpacaClient)
    client.cancel_order.return_value = True
    client.get_order.return_value = {
        'status': 'filled', 'filled_qty': 200, 'filled_avg_price': 9.11,
    }
    client.submit_limit_sell_order.return_value = {'id': 'sell-1', 'status': 'accepted', 'symbol': 'KOLD'}
    client.close_position.return_value = {'id': 'close-1', 'status': 'accepted', 'symbol': 'KOLD'}
    return client


@pytest.fixture
def sm_monitor(sm_mock_alpaca):
    mon = StopMonitor(
        api_key='test-key', api_secret='test-secret', alpaca_client=sm_mock_alpaca,
    )
    mon._STOP_EXIT_FILL_TIMEOUT_S = 0.2
    mon._STOP_EXIT_POLL_INTERVAL_S = 0.05
    return mon


class TestMarkTargetResting:
    def test_updates_watch_fields(self, sm_monitor):
        sm_monitor.add_watch('KOLD', 8.50, 200, 'orig-tp', 'orig-sl', trade_db_id=7)
        rested_at = datetime.now(timezone.utc)

        ok = sm_monitor.mark_target_resting('KOLD', 'new-tp-leg', 9.10, rested_at=rested_at)

        assert ok is True
        watch = sm_monitor._watches['KOLD']
        assert watch.tp_leg_id == 'new-tp-leg'
        assert watch.target_resting is True
        assert watch.target_price == pytest.approx(9.10)
        assert watch.target_rested_at == pytest.approx(rested_at.timestamp())

    def test_no_watch_warns_and_returns_false(self, sm_monitor):
        assert sm_monitor.mark_target_resting('GHOST', 'leg', 9.10) is False


class TestBookTargetRestedFill:
    def test_queues_event_and_retires_watch(self, sm_monitor):
        sm_monitor.add_watch('KOLD', 8.50, 200, 'orig-tp', 'orig-sl', trade_db_id=7)
        sm_monitor.mark_target_resting('KOLD', 'new-tp-leg', 9.10)

        ok = sm_monitor.book_target_rested_fill('KOLD', 200, 9.11, 9.10, datetime.now(timezone.utc))

        assert ok is True
        assert 'KOLD' not in sm_monitor.watched_symbols
        events = sm_monitor.drain_exit_events()
        assert len(events) == 1
        ev = events[0]
        assert ev.exit_reason == ExitReason.TARGET_RESTED.value
        assert ev.exit_branch == ExitBranch.LIMIT.value
        assert ev.exit_price == pytest.approx(9.11)
        assert ev.shares == 200
        assert ev.exit_limit_price == pytest.approx(9.10)
        assert ev.trade_db_id == 7
        assert ev.confirmed is True

    def test_no_watch_returns_false(self, sm_monitor):
        assert sm_monitor.book_target_rested_fill('GHOST', 100, 9.0, 9.0, None) is False


class TestCancelBeforeStopHook:
    """rule 2: a stop trigger on a target_resting watch cancels the resting
    TP first and resolves the already-filled race, all inside
    _execute_stop_exit — the SAME entry point every autonomous stop uses."""

    @pytest.mark.asyncio
    async def test_already_filled_race_emits_target_rested_no_stop_order(self, sm_monitor, sm_mock_alpaca):
        sm_monitor.add_watch('KOLD', 8.50, 200, 'orig-tp', 'orig-sl', trade_db_id=7)
        sm_monitor.mark_target_resting('KOLD', 'live-tp-leg', 9.10)
        sm_mock_alpaca.cancel_order.return_value = False  # not cancelable
        sm_mock_alpaca.get_order.return_value = {
            'status': 'filled', 'filled_qty': 200, 'filled_avg_price': 9.12,
        }
        watch = sm_monitor._watches['KOLD']

        await sm_monitor._execute_stop_exit('KOLD', 8.50, watch, exit_reason='stop_loss')

        sm_mock_alpaca.cancel_order.assert_called_once_with('live-tp-leg')
        # No stop-side sell was attempted — the race resolved to a fill.
        sm_mock_alpaca.submit_limit_sell_order.assert_not_called()
        sm_mock_alpaca.close_position.assert_not_called()
        events = sm_monitor.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_reason == ExitReason.TARGET_RESTED.value
        assert events[0].exit_price == pytest.approx(9.12)
        assert 'KOLD' not in sm_monitor.watched_symbols

    @pytest.mark.asyncio
    async def test_clean_cancel_falls_through_to_normal_stop_exit(self, sm_monitor, sm_mock_alpaca):
        """No race: the TP cancels cleanly, so the ordinary stop-exit flow
        (unchanged) proceeds and books a real stop exit."""
        sm_monitor.add_watch('KOLD', 8.50, 200, 'orig-tp', 'orig-sl', trade_db_id=7)
        sm_monitor.mark_target_resting('KOLD', 'live-tp-leg', 9.10)
        sm_mock_alpaca.cancel_order.return_value = True  # cancels cleanly
        # get_order is now ALWAYS consulted after a cancel (money-defect
        # fix) — for the TP leg specifically it must show 0 filled (a
        # genuinely clean cancel); every OTHER order id (the stop-exit
        # flow's own sell/SL-leg polling) keeps the shared fixture's
        # pre-wired "filled" happy path so the downstream flow still
        # resolves without hitting its poll timeout.
        default_get_order = sm_mock_alpaca.get_order.return_value
        sm_mock_alpaca.get_order.side_effect = lambda oid: (
            {'status': 'canceled', 'filled_qty': 0} if oid == 'live-tp-leg'
            else default_get_order
        )
        watch = sm_monitor._watches['KOLD']

        await sm_monitor._execute_stop_exit('KOLD', 8.40, watch, exit_reason='stop_loss')

        sm_mock_alpaca.cancel_order.assert_any_call('live-tp-leg')
        events = sm_monitor.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_reason != ExitReason.TARGET_RESTED.value

    @pytest.mark.asyncio
    async def test_non_target_watch_unaffected(self, sm_monitor, sm_mock_alpaca):
        """Parity: a watch that never rested a target (target_resting=False,
        the default) must never touch the new hook at all."""
        sm_monitor.add_watch('PLYX', 4.20, 500, 'tp-1', 'sl-1', trade_db_id=1)
        watch = sm_monitor._watches['PLYX']
        assert watch.target_resting is False

        await sm_monitor._execute_stop_exit('PLYX', 4.15, watch, exit_reason='stop_loss')

        # cancel_order was called for the ordinary bracket-leg cleanup, but
        # get_order (the race-resolution call) was never reached — proving
        # the new branch was skipped entirely.
        events = sm_monitor.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_reason != ExitReason.TARGET_RESTED.value


# =============================================================================
# Integration: real ORBEngine + real StopMonitor + real Database, driven by
# an order-stream event shaped like OrderStreamWatcher.get_status()'s output.
# =============================================================================

class TestIntegrationRealOrderStreamFill:
    @pytest.fixture
    def real_db(self, tmp_path):
        db = Database(db_path=str(tmp_path / "orb_target_limit_integration.db"))
        yield db
        db.close()

    @pytest.fixture
    def real_stop_monitor(self, sm_mock_alpaca):
        return StopMonitor(api_key='k', api_secret='s', alpaca_client=sm_mock_alpaca)

    @pytest.fixture
    def real_engine(self, orb_cfg, sm_mock_alpaca, real_db, real_stop_monitor):
        orb_cfg['exit']['target_resting_limit'] = True
        eng = ORBEngine(
            alpaca_client=sm_mock_alpaca, db=real_db,
            stop_monitor=real_stop_monitor, config=orb_cfg,
        )
        eng.order_stream = MagicMock(spec=OrderStreamWatcher)
        return eng

    def test_real_fill_event_shape_flows_to_db_row(self, real_engine, real_db, real_stop_monitor):
        # Seed a real trade row so update_trade(trade_id, ...) has something
        # to close, mirroring how a real fill confirmation would have saved it.
        trade_id = real_db.save_trade({
            'trade_date': datetime.now(timezone.utc).date(),
            'symbol': 'KOLD', 'strategy': 'orb', 'side': 'buy',
            'entry_price': 9.00, 'stop_loss_price': 8.50, 'take_profit_price': 27.00,
            'shares': 200, 'risk_per_share': 0.50, 'total_risk': 100.0,
            'risk_reward_ratio': 0.0, 'order_id': 'entry-order-1',
            'order_status': 'open', 'fill_price': 9.00, 'filled_at': datetime.now(timezone.utc),
            'exit_price': None, 'exit_reason': None, 'exited_at': None,
            'pnl': None, 'pnl_pct': None, 'pattern_data': None,
        })
        pos = OpenPosition(
            symbol='KOLD', entry_price=9.00, stop_price=8.50, shares=200,
            trade_id=trade_id, order_id='', entry_time=datetime.now(timezone.utc),
            range_high=9.00, range_low=8.50, lock_arm_at_r=1.75, lock_stop_r=0.5,
            composite_score=0.5, quintile='Q3', tp_leg_id='orig-tp-leg', sl_leg_id='orig-sl-leg',
        )
        real_engine.open_positions['KOLD'] = pos

        real_stop_monitor.add_watch('KOLD', 8.50, 200, 'orig-tp-leg', 'orig-sl-leg',
                                     trade_db_id=trade_id, strategy='orb')
        rested_at = datetime.now(timezone.utc) - timedelta(seconds=12)
        real_stop_monitor.mark_target_resting('KOLD', 'live-tp-leg', 9.10, rested_at=rested_at)
        pos.target_resting = True
        pos.target_price = 9.10
        pos.tp_leg_id = 'live-tp-leg'
        pos.target_rested_at = rested_at

        # The real shape OrderStreamWatcher.get_status() returns for a
        # filled order (trading/order_stream.py _order_to_status).
        real_engine.order_stream.get_status.return_value = {
            'id': 'live-tp-leg', 'status': 'filled', 'symbol': 'KOLD',
            'qty': 200, 'filled_qty': 200, 'filled_avg_price': 9.115,
            'side': 'sell', 'type': 'limit',
        }

        real_engine._check_exits_locked()

        import sqlite3
        conn = sqlite3.connect(str(real_db._trades_path))
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone())
        conn.close()
        assert row['exit_reason'] == ExitReason.TARGET_RESTED.value
        assert row['exit_price'] == pytest.approx(9.115)
        assert row['exit_limit_price'] == pytest.approx(9.10)
        assert row['exit_fill_latency_ms'] == pytest.approx(12_000, rel=0.2)
        assert 'KOLD' not in real_engine.open_positions
        assert 'KOLD' not in real_stop_monitor.watched_symbols


# =============================================================================
# Rule 6 money-defect fix (2026-09-29 follow-up): a partial TP fill raced
# against a stop trigger, a second partial, or an EOD flatten must NEVER
# book a full close while shares are still held.
# =============================================================================

class TestPartialFillThenStop:
    """A partial fill during the cancel-before-stop race must book only the
    filled qty (row stays open) and let the REMAINDER sell via the ordinary
    stop-exit flow — never a full close on a partial fill."""

    @pytest.mark.asyncio
    async def test_partial_then_stop_sells_only_the_remainder(self, sm_monitor, sm_mock_alpaca):
        sm_monitor.add_watch('KOLD', 8.50, 200, 'orig-tp', 'orig-sl', trade_db_id=7)
        sm_monitor.mark_target_resting('KOLD', 'live-tp-leg', 9.10)
        sm_mock_alpaca.cancel_order.return_value = True  # cancels the OPEN remainder cleanly
        sm_mock_alpaca.get_order.side_effect = lambda oid: (
            {'status': 'canceled', 'filled_qty': 60, 'filled_avg_price': 9.11}
            if oid == 'live-tp-leg'
            else {'id': 'sell-order-123', 'status': 'filled',
                  'filled_avg_price': 8.40, 'filled_qty': 140}
        )
        watch = sm_monitor._watches['KOLD']

        await sm_monitor._execute_stop_exit('KOLD', 8.40, watch, exit_reason='stop_loss')

        events = sm_monitor.drain_exit_events()
        assert len(events) == 2, "one partial event + one remainder stop event"
        partial = next(e for e in events if e.exit_reason == ExitReason.TARGET_RESTED_PARTIAL.value)
        remainder = next(e for e in events if e.exit_reason != ExitReason.TARGET_RESTED_PARTIAL.value)
        assert partial.shares == 60
        assert partial.exit_price == pytest.approx(9.11)
        # The remainder's stop-exit sold exactly what was left (200-60=140),
        # not the original 200 — proving watch.shares was reduced BEFORE the
        # stop-exit logic below the hook read it.
        assert remainder.shares == 140
        assert 'KOLD' not in sm_monitor.watched_symbols  # fully retired once the remainder clears

    @pytest.mark.asyncio
    async def test_partial_qty_never_double_counted_in_a_full_close_event(self, sm_monitor, sm_mock_alpaca):
        """Regression guard for the exact defect reported: the partial event
        must carry ONLY the filled qty, never watch.shares (the pre-fill
        full size)."""
        sm_monitor.add_watch('KOLD', 8.50, 200, 'orig-tp', 'orig-sl', trade_db_id=7)
        sm_monitor.mark_target_resting('KOLD', 'live-tp-leg', 9.10)
        sm_mock_alpaca.cancel_order.return_value = False
        sm_mock_alpaca.get_order.side_effect = lambda oid: (
            {'status': 'canceled', 'filled_qty': 60, 'filled_avg_price': 9.11}
            if oid == 'live-tp-leg'
            else {'id': 'sell-order-123', 'status': 'filled',
                  'filled_avg_price': 8.40, 'filled_qty': 140}
        )
        watch = sm_monitor._watches['KOLD']

        await sm_monitor._execute_stop_exit('KOLD', 8.40, watch, exit_reason='stop_loss')

        events = sm_monitor.drain_exit_events()
        partial = next(e for e in events if e.exit_reason == ExitReason.TARGET_RESTED_PARTIAL.value)
        assert partial.shares != 200
        assert partial.shares == 60


class TestBookTargetPartialFill:
    def test_reduces_watch_shares_and_queues_partial_event(self, sm_monitor):
        sm_monitor.add_watch('KOLD', 8.50, 200, 'orig-tp', 'orig-sl', trade_db_id=7)
        sm_monitor.mark_target_resting('KOLD', 'live-tp-leg', 9.10)

        ok = sm_monitor.book_target_partial_fill('KOLD', 60, 9.11, 9.10)

        assert ok is True
        assert sm_monitor._watches['KOLD'].shares == 140
        assert 'KOLD' in sm_monitor.watched_symbols  # NOT retired — remainder still live
        events = sm_monitor.drain_exit_events()
        assert len(events) == 1
        assert events[0].exit_reason == ExitReason.TARGET_RESTED_PARTIAL.value
        assert events[0].shares == 60
        assert events[0].confirmed is True

    def test_second_partial_reduces_further(self, sm_monitor):
        sm_monitor.add_watch('KOLD', 8.50, 200, 'orig-tp', 'orig-sl', trade_db_id=7)
        sm_monitor.mark_target_resting('KOLD', 'live-tp-leg', 9.10)
        sm_monitor.book_target_partial_fill('KOLD', 60, 9.11, 9.10)
        sm_monitor.drain_exit_events()  # drain the first partial's event

        ok = sm_monitor.book_target_partial_fill('KOLD', 30, 9.12, 9.10)

        assert ok is True
        assert sm_monitor._watches['KOLD'].shares == 110
        events = sm_monitor.drain_exit_events()
        assert len(events) == 1  # only the second call's event
        assert events[0].shares == 30

    def test_no_watch_returns_false(self, sm_monitor):
        assert sm_monitor.book_target_partial_fill('GHOST', 10, 9.0, 9.0) is False


class TestHandleTargetPartialFillEvent:
    """ORBEngine._handle_target_partial_fill_event: row stays OPEN, reuses
    the scale_qty/scale_price/scale_pnl/scaled_at representation."""

    @dataclass
    class _FakePartialEvent:
        symbol: str
        exit_price: float
        exit_reason: str = ExitReason.TARGET_RESTED_PARTIAL.value
        filled_qty: int = 0
        shares: int = 0
        trade_db_id: int = 0

    def test_partial_leaves_row_open_and_updates_scale_columns(self, engine, mock_db):
        pos = _seed_position(engine)
        pos.target_resting = True
        pos.target_price = 9.10

        ev = self._FakePartialEvent(symbol='KOLD', exit_price=9.11, filled_qty=60)
        engine._handle_target_partial_fill_event(ev)

        assert pos.shares == 140  # 200 - 60
        assert pos.scale_qty == 60
        assert pos.scale_pnl == pytest.approx((9.11 - 9.00) * 60)
        assert 'KOLD' in engine.open_positions  # NEVER popped
        merged = {}
        for call in mock_db.update_trade.call_args_list:
            args, kwargs = call
            merged.update(args[1] if len(args) >= 2 else kwargs.get('updates', {}))
        assert 'exit_price' not in merged
        assert 'exit_reason' not in merged
        assert 'pnl' not in merged
        assert merged['scale_qty'] == 60

    def test_two_partials_accumulate(self, engine, mock_db):
        pos = _seed_position(engine)
        engine._handle_target_partial_fill_event(
            self._FakePartialEvent(symbol='KOLD', exit_price=9.11, filled_qty=60))
        engine._handle_target_partial_fill_event(
            self._FakePartialEvent(symbol='KOLD', exit_price=9.13, filled_qty=30))

        assert pos.shares == 110  # 200 - 60 - 30
        assert pos.scale_qty == 90
        expected_pnl = (9.11 - 9.00) * 60 + (9.13 - 9.00) * 30
        assert pos.scale_pnl == pytest.approx(expected_pnl)

    def test_orphan_partial_writes_by_trade_db_id(self, engine, mock_db):
        """No tracked position (restart race) — best-effort DB write, never crashes."""
        ev = self._FakePartialEvent(symbol='GHOST', exit_price=9.11, filled_qty=60, trade_db_id=99)
        engine._handle_target_partial_fill_event(ev)  # must not raise
        mock_db.update_trade.assert_called_once()
        args = mock_db.update_trade.call_args.args
        assert args[0] == 99


class TestPollTargetFillsPartial:
    def test_partial_status_books_partial_not_full(self, engine, mock_stop_monitor):
        pos = _seed_position(engine)
        pos.target_resting = True
        pos.target_price = 9.10
        engine.order_stream = MagicMock(spec=OrderStreamWatcher)
        engine.order_stream.get_status.return_value = {
            'status': 'partially_filled', 'filled_qty': 60, 'filled_avg_price': 9.11,
        }
        mock_stop_monitor.book_target_partial_fill.return_value = True

        engine._poll_target_fills()

        mock_stop_monitor.book_target_partial_fill.assert_called_once_with('KOLD', 60, 9.11, 9.10)
        mock_stop_monitor.book_target_rested_fill.assert_not_called()
        assert pos.target_last_booked_qty == 60

    def test_second_partial_books_only_the_new_delta(self, engine, mock_stop_monitor):
        """rule 6: repeated polls on a still-resting, still-partially-filled
        order must book ONLY the NEW shares each time, never re-book the
        same fill twice."""
        pos = _seed_position(engine)
        pos.target_resting = True
        pos.target_price = 9.10
        pos.target_last_booked_qty = 60  # a first partial was already booked
        engine.order_stream = MagicMock(spec=OrderStreamWatcher)
        engine.order_stream.get_status.return_value = {
            'status': 'partially_filled', 'filled_qty': 90, 'filled_avg_price': 9.12,
        }
        mock_stop_monitor.book_target_partial_fill.return_value = True

        engine._poll_target_fills()

        mock_stop_monitor.book_target_partial_fill.assert_called_once_with('KOLD', 30, 9.12, 9.10)
        assert pos.target_last_booked_qty == 90

    def test_repeated_poll_with_no_new_fill_is_a_no_op(self, engine, mock_stop_monitor):
        pos = _seed_position(engine)
        pos.target_resting = True
        pos.target_last_booked_qty = 60
        engine.order_stream = MagicMock(spec=OrderStreamWatcher)
        engine.order_stream.get_status.return_value = {
            'status': 'partially_filled', 'filled_qty': 60, 'filled_avg_price': 9.11,
        }
        engine._poll_target_fills()
        mock_stop_monitor.book_target_partial_fill.assert_not_called()
        mock_stop_monitor.book_target_rested_fill.assert_not_called()

    def test_full_fill_after_a_prior_partial_closes_for_the_remainder_only(self, engine, mock_stop_monitor):
        pos = _seed_position(engine)
        pos.target_resting = True
        pos.target_price = 9.10
        pos.target_last_booked_qty = 60
        pos.shares = 140  # already reduced by the earlier partial
        engine.order_stream = MagicMock(spec=OrderStreamWatcher)
        engine.order_stream.get_status.return_value = {
            'status': 'filled', 'filled_qty': 200, 'filled_avg_price': 9.13,
        }
        mock_stop_monitor.book_target_rested_fill.return_value = True

        engine._poll_target_fills()

        mock_stop_monitor.book_target_rested_fill.assert_called_once_with(
            'KOLD', 140, 9.13, 9.10, pos.target_rested_at,
        )
        mock_stop_monitor.book_target_partial_fill.assert_not_called()


class TestPartialFillThenEodFlatten:
    """rule 6 + rule 5: EOD force-close must see any last-second fill BEFORE
    it computes what to close, or it risks overselling against pos.shares."""

    def test_force_close_polls_target_fills_first_when_flag_on(self, engine, mock_alpaca):
        engine.target_resting_limit_enabled = True
        _seed_position(engine)
        called = []
        engine._poll_target_fills = lambda: called.append(True)
        mock_alpaca.trading_client = MagicMock()
        mock_alpaca.trading_client.get_orders.return_value = []
        mock_alpaca.get_open_positions.return_value = []

        engine._force_close_all_locked()

        # _check_exits_locked (called inside the force-close VERIFY phase)
        # also polls when the flag is on — idempotent (delta-tracked), so
        # 1+ calls is the real invariant, not an exact count.
        assert len(called) >= 1

    def test_force_close_skips_poll_when_flag_off(self, engine, mock_alpaca):
        assert engine.target_resting_limit_enabled is False
        _seed_position(engine)
        called = []
        engine._poll_target_fills = lambda: called.append(True)
        mock_alpaca.trading_client = MagicMock()
        mock_alpaca.trading_client.get_orders.return_value = []
        mock_alpaca.get_open_positions.return_value = []

        engine._force_close_all_locked()

        assert called == []  # parity: untouched when the flag is off

    def test_partial_booked_before_close_leaves_correct_remaining_shares(self, engine, mock_stop_monitor):
        """A partial fill observed by the pre-close poll must reduce
        pos.shares BEFORE anything downstream reads it for the close qty."""
        pos = _seed_position(engine)
        pos.target_resting = True
        pos.target_price = 9.10
        engine.target_resting_limit_enabled = True
        engine.order_stream = MagicMock(spec=OrderStreamWatcher)
        engine.order_stream.get_status.return_value = {
            'status': 'partially_filled', 'filled_qty': 60, 'filled_avg_price': 9.11,
        }

        def _fake_book(sym, qty, price, limit):
            engine._handle_target_partial_fill_event(
                TestHandleTargetPartialFillEvent._FakePartialEvent(
                    symbol=sym, exit_price=price, filled_qty=qty))
            return True
        mock_stop_monitor.book_target_partial_fill.side_effect = _fake_book

        engine._poll_target_fills()

        assert pos.shares == 140
