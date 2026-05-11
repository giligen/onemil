"""Regression tests for 2026-04-21 fixes:

1. force_close_all must cancel bracket legs (SL/TP) BEFORE close_position,
   otherwise Alpaca refuses the close with 'insufficient qty available'
   because shares are held_for_orders. See ANNA overnight leak.

2. sync_positions must detect orphan Alpaca positions (not in DB today)
   and alert via telegram.

3. _notify_error must log + telegram.

4. Critical error paths (DB save after Alpaca accept, add_watch failure,
   _confirm_fill DB update) must telegram.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, OpenPosition
from trading.stop_monitor import StopMonitor


@pytest.fixture
def orb_cfg():
    yaml_path = Path(__file__).parent.parent / 'orb.yaml'
    with open(yaml_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def mock_alpaca():
    c = MagicMock(spec=AlpacaClient)
    c.get_open_positions.return_value = []
    c.trading_client = MagicMock()
    c.trading_client.get_orders.return_value = []
    c.close_position.return_value = {'id': 'close-order-1'}
    c.cancel_order.return_value = True
    c.get_account_info.return_value = {'buying_power': 100_000}
    return c


@pytest.fixture
def mock_db():
    db = MagicMock(spec=Database)
    db.save_trade.return_value = 1
    db.get_open_trades.return_value = []
    db.update_trade.return_value = True
    return db


@pytest.fixture
def mock_sm():
    sm = MagicMock(spec=StopMonitor)
    sm.polling_mode = False
    sm.drain_exit_events.return_value = []
    return sm


@pytest.fixture
def notifier():
    n = MagicMock()
    n.send_message = MagicMock(return_value=None)  # non-async path
    return n


@pytest.fixture
def engine(orb_cfg, mock_alpaca, mock_db, mock_sm, notifier):
    orb_cfg['strategy']['enabled'] = True
    return ORBEngine(
        alpaca_client=mock_alpaca, db=mock_db,
        stop_monitor=mock_sm, config=orb_cfg, notifier=notifier,
    )


def _pos(sym='ANNA', qty=1000, entry=10.0, stop=9.5, trade_id=42):
    return OpenPosition(
        symbol=sym, entry_price=entry, stop_price=stop, shares=qty,
        trade_id=trade_id, order_id='',
        entry_time=datetime.now(timezone.utc),
        range_high=entry, range_low=stop,
        lock_arm_at_r=1.5, lock_stop_r=1.0,
        composite_score=0.5, quintile='Q4',
    )


class TestNotifyError:
    def test_logs_and_telegrams(self, engine, notifier, caplog):
        with caplog.at_level('ERROR'):
            engine._notify_error("oh no", exc=RuntimeError("boom"))
        # Logged
        assert any('ORB CRITICAL' in r.getMessage() for r in caplog.records)
        # Telegrammed
        notifier.send_message.assert_called_once()
        msg = notifier.send_message.call_args[0][0]
        assert '❌' in msg
        assert 'oh no' in msg
        assert 'RuntimeError' in msg

    def test_no_telegram_when_notifier_none(self, orb_cfg, mock_alpaca, mock_db, mock_sm, caplog):
        orb_cfg['strategy']['enabled'] = True
        e = ORBEngine(
            alpaca_client=mock_alpaca, db=mock_db,
            stop_monitor=mock_sm, config=orb_cfg, notifier=None,
        )
        # Should still log, just no telegram — and no crash
        with caplog.at_level('ERROR'):
            e._notify_error("test")
        assert any('ORB CRITICAL' in r.getMessage() for r in caplog.records)


class TestForceCloseBracketCancel:
    def test_cancels_bracket_legs_before_close(self, engine, mock_alpaca):
        """Real bug: close_position fails if bracket legs hold shares.
        Fix: cancel all open orders for symbol, sleep briefly, then close."""
        engine.open_positions['ANNA'] = _pos()

        # Alpaca reports 2 live bracket legs for ANNA (SL + TP)
        sl_leg = MagicMock(); sl_leg.id = 'leg-sl'
        tp_leg = MagicMock(); tp_leg.id = 'leg-tp'
        mock_alpaca.trading_client.get_orders.return_value = [sl_leg, tp_leg]

        engine.force_close_all()
        # Both legs were canceled
        cancel_calls = mock_alpaca.trading_client.cancel_order_by_id.call_args_list
        canceled_ids = [c[0][0] for c in cancel_calls]
        assert 'leg-sl' in canceled_ids
        assert 'leg-tp' in canceled_ids
        # Then close_position was called
        mock_alpaca.close_position.assert_called_with('ANNA')

    def test_cancel_order_before_close_call_order(self, engine, mock_alpaca):
        """Explicit ordering: cancel must come BEFORE close_position."""
        engine.open_positions['ANNA'] = _pos()
        leg = MagicMock(); leg.id = 'leg-x'
        mock_alpaca.trading_client.get_orders.return_value = [leg]

        call_log = []
        mock_alpaca.trading_client.cancel_order_by_id.side_effect = (
            lambda oid: call_log.append(('cancel', oid))
        )
        mock_alpaca.close_position.side_effect = (
            lambda sym: call_log.append(('close', sym)) or {'id': 'c1'}
        )
        engine.force_close_all()
        # cancel → close
        assert call_log[0][0] == 'cancel'
        assert call_log[-1][0] == 'close'

    def test_close_failure_telegrams_critical(self, engine, mock_alpaca, notifier):
        """When the held_qty retry helper exhausts all 5 attempts, the
        engine still alerts (lighter-touch wording than pre-fix — FC VERIFY
        will get another chance, so the alert says 'helper exhausted retries'
        and 'only escalate if VERIFY also fails')."""
        engine.open_positions['ANNA'] = _pos()
        mock_alpaca.trading_client.get_orders.return_value = []
        # Speed: zero backoffs so the 5 attempts run instantly.
        engine._FC_HELD_QTY_BACKOFFS_S = (0.0, 0.0, 0.0, 0.0)
        mock_alpaca.close_position.side_effect = RuntimeError(
            "insufficient qty available"
        )
        engine.force_close_all()
        msgs = [c[0][0] for c in notifier.send_message.call_args_list]
        critical = [
            m for m in msgs
            if 'close_position helper exhausted retries' in m
        ]
        assert len(critical) == 1
        assert 'ANNA' in critical[0]

    def test_close_retry_succeeds_no_alert(self, engine, mock_alpaca, notifier):
        """First close fails with held_for_orders, retry succeeds → no
        critical telegram. Validates Bug 1 fix (per-FC_HELD_QTY_BACKOFFS_S
        retry helper)."""
        engine.open_positions['ANNA'] = _pos()
        mock_alpaca.trading_client.get_orders.return_value = []
        engine._FC_HELD_QTY_BACKOFFS_S = (0.0,)  # fast retry
        call_count = [0]
        def close_side(sym):
            call_count[0] += 1
            if call_count[0] == 1:
                # Must match the helper's race signature
                raise RuntimeError("40310000 insufficient qty available")
            return {'id': 'retry-close'}
        mock_alpaca.close_position.side_effect = close_side
        engine.force_close_all()
        msgs = [c[0][0] for c in notifier.send_message.call_args_list]
        assert not any(
            'close_position helper exhausted retries' in m for m in msgs
        )

    def test_non_race_exception_not_retried(self, engine, mock_alpaca, notifier):
        """Helper must re-raise non-held_for_orders exceptions immediately
        (no retry). E.g., 'invalid symbol' should NOT trigger 5 retries."""
        engine.open_positions['ANNA'] = _pos()
        mock_alpaca.trading_client.get_orders.return_value = []
        call_count = [0]
        def close_side(sym):
            call_count[0] += 1
            raise RuntimeError("invalid symbol")  # NOT a race error
        mock_alpaca.close_position.side_effect = close_side
        engine.force_close_all()
        # Only ONE call — helper re-raised on the first non-race exception
        assert call_count[0] == 1

    def test_phase1_cancel_loop_silent_when_no_db_pending(
        self, engine, mock_alpaca, mock_db, caplog
    ):
        """Bug 5 fix: candidates retain plan_submitted=True after fill+exit.
        Pre-fix this fired 'cancelling unfilled pending order for X' on
        every such candidate (SONY/APT/MLTX on 2026-05-11 FC). Post-fix:
        the loop queries DB first and skips candidates with no pending_new
        row → silent."""
        from trading.orb_engine import CandidateState
        # Three candidates that have plan_submitted=True but already
        # filled-and-exited (sym not in open_positions, no DB pending row).
        for sym in ('SONY', 'APT', 'MLTX'):
            engine.candidates[sym] = CandidateState(symbol=sym)
            engine.candidates[sym].plan_submitted = True
        mock_db.get_open_trades.return_value = []  # no pending_new rows
        import logging as _logging
        caplog.set_level(_logging.INFO, logger='trading.orb_engine')
        engine.force_close_all()
        # No "cancelling unfilled pending order for X" log lines for any
        # of the three already-exited candidates.
        for sym in ('SONY', 'APT', 'MLTX'):
            spam = [
                r for r in caplog.records
                if f"cancelling" in r.getMessage() and sym in r.getMessage()
            ]
            assert not spam, (
                f"expected silent cancel loop for filled+exited candidate "
                f"{sym}, got: {[r.getMessage() for r in spam]}"
            )

    def test_phase1_log_says_submitted_not_closed(
        self, engine, mock_alpaca, caplog
    ):
        """Bug 3 fix: log line for Phase-1 close MUST say 'submitted close
        order' not 'closed' — close_position returns submission, not fill."""
        engine.open_positions['ANNA'] = _pos()
        mock_alpaca.trading_client.get_orders.return_value = []
        mock_alpaca.close_position.return_value = {'id': 'submitted-1'}
        import logging as _logging
        caplog.set_level(_logging.INFO, logger='trading.orb_engine')
        engine.force_close_all()
        fc_logs = [r.getMessage() for r in caplog.records if 'ORB FORCE-CLOSE: ANNA' in r.getMessage()]
        assert any('submitted close order' in m for m in fc_logs), (
            f"expected 'submitted close order' wording; got: {fc_logs}"
        )
        # And NO log claims fill happened on Phase 1 submission.
        assert not any(
            'closed on retry' in m or 'closed (order=' in m
            for m in fc_logs
        ), fc_logs


class TestFCSweepRaceFix:
    """Bug 2: Phase 2 FC SWEEP previously raced Phase 1's pending close and
    fired 'CRITICAL: position WILL leak overnight' as a false alarm. Post-
    fix: SWEEP skips symbols Phase 1 already submitted a close for."""

    def test_sweep_skips_phase1_submitted_symbol(
        self, engine, mock_alpaca, notifier, caplog
    ):
        # Phase 1 submits close for ANNA (succeeds)
        engine.open_positions['ANNA'] = _pos()
        mock_alpaca.trading_client.get_orders.return_value = []
        mock_alpaca.close_position.return_value = {'id': 'phase1-close'}
        # Then Alpaca's position list still shows ANNA (close not filled yet)
        orphan = MagicMock()
        orphan.symbol = 'ANNA'
        orphan.qty = 1000
        orphan.avg_entry_price = 10.0
        orphan.unrealized_pl = 0.0
        mock_alpaca.get_open_positions.return_value = [orphan]
        # Make _verify_flat_with_grace see flat on first poll so test exits fast.
        # Patch via the alpaca mock: get_open_positions returns []
        # *after* the Phase-1 close was submitted. Simulate this by
        # tracking calls and returning [] for verify polls.
        call_count = [0]
        def positions_side():
            call_count[0] += 1
            # Phase 1 doesn't call get_open_positions; Phase 2 SWEEP does
            # (returns the orphan); FC VERIFY polls — return flat to exit.
            if call_count[0] == 1:
                return [orphan]
            return []
        mock_alpaca.get_open_positions.side_effect = positions_side

        import logging as _logging
        caplog.set_level(_logging.INFO, logger='trading.orb_engine')
        engine.force_close_all()

        # SWEEP should have logged the skip, NOT a "survived engine-state"
        # alert (which previously preceded the false-alarm CRITICAL).
        skip_logs = [
            r.getMessage() for r in caplog.records
            if 'awaiting fill via VERIFY' in r.getMessage()
        ]
        survived_logs = [
            r.getMessage() for r in caplog.records
            if 'survived engine-state' in r.getMessage()
        ]
        assert skip_logs, (
            f"expected 'awaiting fill via VERIFY' skip log; got logs: "
            f"{[r.getMessage() for r in caplog.records]}"
        )
        assert not survived_logs, (
            f"SWEEP should NOT fire 'survived engine-state' for symbol "
            f"Phase 1 already submitted close for; got: {survived_logs}"
        )
        # And no CRITICAL "leak overnight" telegram on this path
        msgs = [c[0][0] for c in notifier.send_message.call_args_list]
        assert not any('WILL leak overnight' in m for m in msgs), msgs


class TestFCVerifyPendingSellWait:
    """Bug 4: FC VERIFY previously blindly cancelled in-flight close orders
    on every attempt. Post-fix: non-final attempts wait for pending sells
    to work; final attempt forces cancel + close to break wedged state."""

    def test_verify_skips_when_pending_sell_exists_non_final(
        self, engine, mock_alpaca
    ):
        """Pending sell exists; non-final attempt → no cancel, no re-close."""
        engine.open_positions = {}  # nothing to close in Phase 1
        # Speed: zero verify backoffs so the test doesn't sleep
        engine.fc_retry_backoffs_s = [0.0, 0.0, 0.0]
        engine.fc_verify_max_wait_s = 0.1
        engine.fc_verify_poll_interval_s = 0.05
        # Position survives Phase 1 + 2; FC VERIFY sees it open every time
        pos = MagicMock(); pos.symbol = 'ANNA'; pos.qty = 1000
        pos.avg_entry_price = 10.0; pos.unrealized_pl = 0.0
        mock_alpaca.get_open_positions.return_value = [pos]
        # Pending sell ALWAYS exists in get_orders (simulates Phase-1 close
        # in flight). The non-final attempts must NOT cancel it.
        sell_order = MagicMock()
        sell_side = MagicMock(); sell_side.value = 'sell'
        sell_order.side = sell_side
        sell_order.id = 'phase1-pending-sell'
        mock_alpaca.trading_client.get_orders.return_value = [sell_order]

        engine.force_close_all()

        # On the first 2 (non-final) attempts, _cancel_symbol_open_orders
        # MUST NOT have been called for ANNA. The final attempt DOES call.
        # _cancel_symbol_open_orders calls trading_client.cancel_order_by_id
        # only once we don't skip — verify call count <= 1 (final attempt).
        cancel_calls = mock_alpaca.trading_client.cancel_order_by_id.call_args_list
        # Pre-fix this was 3 — once per VERIFY attempt × 3 attempts
        # plus 1 from FC SWEEP. Post-fix: SWEEP also skips Phase-1
        # closed, but ANNA is not in closed_symbols (no Phase 1 success),
        # so SWEEP DOES try once. Then VERIFY non-final attempts skip,
        # final attempt cancels once. Total ≤ 2 cancels.
        assert len(cancel_calls) <= 2, (
            f"expected ≤2 cancel calls (SWEEP + final VERIFY); pre-fix "
            f"would have been 4+. Got {len(cancel_calls)}: {cancel_calls}"
        )

    def test_verify_proceeds_when_no_pending_sell(
        self, engine, mock_alpaca
    ):
        """No pending sell → FC VERIFY does cancel + close normally."""
        engine.open_positions = {}
        engine.fc_retry_backoffs_s = [0.0, 0.0, 0.0]
        engine.fc_verify_max_wait_s = 0.1
        engine.fc_verify_poll_interval_s = 0.05
        engine._FC_HELD_QTY_BACKOFFS_S = (0.0,)
        # Position open, but no pending sell (helps prove the flow proceeds)
        pos = MagicMock(); pos.symbol = 'ANNA'; pos.qty = 1000
        pos.avg_entry_price = 10.0; pos.unrealized_pl = 0.0
        # First call: position is open. Subsequent calls (after close):
        # position is gone. This lets the first VERIFY attempt close
        # successfully and the next verify see flat.
        positions_call_count = [0]
        def positions_side():
            positions_call_count[0] += 1
            if positions_call_count[0] <= 2:
                return [pos]
            return []
        mock_alpaca.get_open_positions.side_effect = positions_side
        mock_alpaca.trading_client.get_orders.return_value = []  # no pending sell
        mock_alpaca.close_position.return_value = {'id': 'verify-close'}

        engine.force_close_all()
        # close_position WAS called (FC VERIFY proceeded — no skip).
        # (May be called multiple times across SWEEP + VERIFY; just verify > 0.)
        assert mock_alpaca.close_position.call_count >= 1


class TestSyncOrphanDetection:
    def test_orphan_alpaca_position_telegrams(self, engine, mock_alpaca, notifier):
        """Alpaca has ANNA open, but no DB row for today → orphan alert."""
        # Simulate Alpaca position not in DB
        orphan = MagicMock()
        orphan.symbol = 'ANNA'
        orphan.qty = 11682
        orphan.avg_entry_price = 3.93
        orphan.unrealized_pl = -2578.0
        mock_alpaca.get_open_positions.return_value = [orphan]
        mock_alpaca.trading_client.get_orders.return_value = []
        engine.db.get_open_trades.return_value = []  # empty for today

        engine.sync_positions()

        msgs = [c[0][0] for c in notifier.send_message.call_args_list]
        critical = [m for m in msgs if 'ORPHAN' in m]
        assert len(critical) == 1
        assert 'ANNA' in critical[0]
        assert '11682' in critical[0]

    def test_no_orphan_alert_when_all_positions_tracked(self, engine, mock_alpaca, notifier):
        """All Alpaca positions are in our open_positions → no alert."""
        anna = MagicMock()
        anna.symbol = 'ANNA'
        anna.qty = 100; anna.avg_entry_price = 10.0; anna.unrealized_pl = 0
        mock_alpaca.get_open_positions.return_value = [anna]
        engine.open_positions['ANNA'] = _pos(sym='ANNA')
        engine.db.get_open_trades.return_value = [{
            'id': 1, 'symbol': 'ANNA', 'strategy': 'orb',
            'order_id': '', 'order_status': 'filled',
            'entry_price': 10.0, 'fill_price': 10.0,
            'stop_loss_price': 9.5, 'shares': 100,
            'pattern_data': '{}',
        }]

        engine.sync_positions()
        msgs = [c[0][0] for c in notifier.send_message.call_args_list]
        assert not any('ORPHAN' in m for m in msgs)
