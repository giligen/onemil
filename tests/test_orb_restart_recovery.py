"""ORB sync_positions restart-safety regression tests (2026-05-08).

Two bugs surfaced by 5/7 ASTX/OKLS/CORD post-mortem:

Bug 1 (filled_at clobber): _confirm_fill set fill_at = datetime.now(),
ignoring Alpaca's actual order.filled_at. After restart, polling-based
fill confirmation wrote sync-time as filled_at, off by minutes-to-hours.

Bug 2 (time_stop bypass): sync_positions only registered the recovered
pending order on self.candidates IF the symbol already had a CandidateState.
After restart, candidates is empty → no registration → _cancel_stale_pending_orders
iterates self.candidates and never sees the recovered pending order.
CORD 5/7: filled 47 min past 60-min cancel deadline due to this.
"""
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

from trading.orb_engine import ORBEngine, OpenPosition, CandidateState


class TestFilledAtUsesAlpacaTime(unittest.TestCase):
    """Bug 1: _confirm_fill must use order.filled_at from Alpaca, not NOW."""

    def _make_engine(self):
        # Use object.__new__ to skip __init__ — only need _confirm_fill behaviour
        engine = object.__new__(ORBEngine)
        engine.daily_n_filled = 0
        engine.daily_pnl = 0.0
        engine.alpaca = MagicMock()
        engine.db = MagicMock()
        engine.stop_monitor = None
        engine.notifier = None
        engine.tg_prefix = "[ORB]"
        engine.notify_on_entry = False
        engine.STRATEGY_NAME = 'orb'
        return engine

    def test_uses_alpaca_filled_at_when_available(self):
        engine = self._make_engine()
        # Real Alpaca filled_at = 13:36:05 UTC (9:36 ET)
        real_fill_at = datetime(2026, 5, 6, 13, 36, 5, tzinfo=timezone.utc)
        order_status = SimpleNamespace(
            filled_avg_price='23.10',
            filled_qty=1988,
            filled_at=real_fill_at,
        )
        pos = OpenPosition(
            symbol='ASTX', entry_price=23.10, stop_price=21.0, shares=1988,
            trade_id=196, order_id='abc', entry_time=datetime.now(timezone.utc),
            range_high=23.16, range_low=21.0, lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
            order_submitted_at=datetime(2026, 5, 6, 13, 35, 7, tzinfo=timezone.utc),
        )

        # Pretend NOW is much later (simulating sync-time clobber scenario)
        with patch('trading.orb_engine.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 6, 19, 25, 57, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            try:
                engine._confirm_fill(pos, order_status)
            except Exception:
                pass  # downstream calls (StopMonitor) may fail; we're checking pos.entry_time

        # The fix: pos.entry_time should be the Alpaca timestamp, not NOW
        self.assertEqual(pos.entry_time, real_fill_at,
                         f"entry_time should be Alpaca's filled_at, got {pos.entry_time}")
        # And the DB update should pass the Alpaca timestamp
        update_call = engine.db.update_trade.call_args
        if update_call:
            updates = update_call[0][1] if len(update_call[0]) > 1 else update_call.kwargs
            self.assertEqual(updates.get('filled_at'), real_fill_at)

    def test_falls_back_to_now_when_alpaca_filled_at_missing(self):
        engine = self._make_engine()
        order_status = SimpleNamespace(
            filled_avg_price='23.10',
            filled_qty=1988,
            filled_at=None,  # missing
        )
        pos = OpenPosition(
            symbol='X', entry_price=23.10, stop_price=21.0, shares=100,
            trade_id=1, order_id='abc', entry_time=datetime.now(timezone.utc),
            range_high=23.16, range_low=21.0, lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
        )
        before = datetime.now(timezone.utc)
        try:
            engine._confirm_fill(pos, order_status)
        except Exception:
            pass
        after = datetime.now(timezone.utc)
        # Fallback: pos.entry_time within a 2s window around now
        self.assertGreaterEqual(pos.entry_time, before - timedelta(seconds=1))
        self.assertLessEqual(pos.entry_time, after + timedelta(seconds=1))


class TestPendingOrderRegisteredOnCandidatesAfterRestart(unittest.TestCase):
    """Bug 2: sync_positions must always register a pending order on
    self.candidates so _cancel_stale_pending_orders can apply time_stop."""

    def _make_engine_with_pending(self, candidates_pre_state=None):
        engine = object.__new__(ORBEngine)
        engine.STRATEGY_NAME = 'orb'
        engine.alpaca = MagicMock()
        engine.db = MagicMock()
        engine.stop_monitor = None
        engine.notifier = None
        engine.open_positions = {}
        engine.candidates = candidates_pre_state or {}
        engine.daily_n_placed = 0
        engine.daily_n_filled = 0
        engine.daily_n_time_stop_canceled = 0
        engine.time_stop_minutes = 60
        engine.tg_prefix = "[ORB]"

        # Alpaca returns no positions, but a pending order for CORD
        engine.alpaca.get_open_positions.return_value = []
        pending_order = SimpleNamespace(id='cord-order-id-999')
        engine.alpaca.trading_client.get_orders.return_value = [pending_order]

        # DB shows CORD as pending_new with original submit timestamp
        submit_ts = datetime(2026, 5, 7, 13, 35, 10, tzinfo=timezone.utc)
        engine.db.get_open_trades.return_value = [{
            'id': 208,
            'symbol': 'CORD',
            'order_id': 'cord-order-id-999',
            'order_status': 'pending_new',
            'order_submitted_at': submit_ts,
            'created_at': submit_ts,
            'entry_price': 4.35,
            'stop_loss_price': 4.17,
            'shares': 10594,
            'pattern_data': '{"range_high": 4.33, "range_low": 4.17, '
                            '"lock_arm_at_r": 1.5, "lock_stop_r": 1.0, '
                            '"composite_score": 0.36, "quintile": "Q4"}',
        }]
        return engine, submit_ts

    def test_recovered_pending_order_added_to_candidates(self):
        """After restart with empty candidates, sync_positions must create
        a stub CandidateState for the pending order."""
        engine, submit_ts = self._make_engine_with_pending()
        engine.sync_positions()

        # CORD should now be in self.candidates with order_id + submit_ts
        self.assertIn('CORD', engine.candidates,
                      "Recovered pending order not registered on candidates")
        cand = engine.candidates['CORD']
        self.assertEqual(cand.order_id, 'cord-order-id-999')
        self.assertEqual(cand.order_submitted_at, submit_ts)
        self.assertTrue(cand.plan_submitted,
                        "plan_submitted should be True for recovered pending")

    def test_existing_candidate_preserves_order_id_and_submit_ts(self):
        """If a candidate already exists pre-sync, sync_positions should
        update it without overwriting other fields."""
        existing = CandidateState(symbol='CORD')
        existing.composite = 0.36
        existing.quintile = 'Q4'
        engine, submit_ts = self._make_engine_with_pending(
            candidates_pre_state={'CORD': existing}
        )
        engine.sync_positions()

        cand = engine.candidates['CORD']
        # Order tracking populated
        self.assertEqual(cand.order_id, 'cord-order-id-999')
        self.assertEqual(cand.order_submitted_at, submit_ts)
        # Pre-existing fields preserved
        self.assertEqual(cand.composite, 0.36)
        self.assertEqual(cand.quintile, 'Q4')

    def test_time_stop_fires_for_recovered_pending_order(self):
        """End-to-end: after sync, _cancel_stale_pending_orders should
        cancel the recovered order if past the 60-min window."""
        # Submit was 9:35 ET = 13:35 UTC. Mock now=11:00 ET = 15:00 UTC = 85 min later.
        engine, _ = self._make_engine_with_pending()
        engine.sync_positions()
        self.assertIn('CORD', engine.candidates)

        # Post-cancel filled_qty check (2026-07-04): never filled.
        engine.alpaca.get_order.return_value = {'filled_qty': 0}
        # Simulate now() at 85 min after submit
        with patch('trading.orb_engine.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 7, 15, 0, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            engine._cancel_stale_pending_orders()

        # Cancellation submitted to Alpaca
        engine.alpaca.cancel_order.assert_called_once_with('cord-order-id-999')
        # DB updated to reflect cancellation
        engine.db.update_trade.assert_any_call(
            208, {'order_status': 'time_stop_canceled'}
        )
        # Daily counter incremented
        self.assertEqual(engine.daily_n_time_stop_canceled, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
