"""Race-A regression test.

Race A trace (pre-fix):
  1. StopMonitor.BRANCH_LAST_RESORT fires for SMU.
  2. Engine consumer writes order_status='exit_pending_verification',
     exit_price=NULL. Pops SMU from open_positions. StopMonitor watch
     removed.
  3. Next sync_positions cycle:
     - db.get_open_trades returns SMU (exit_pending_verification is in
       Database._ACTIVE_ORDER_STATUSES so the row is "active").
     - Recovery loop sees fill_price set + broker still has the symbol
       + sym not in open_positions → re-adds + creates fresh
       StopMonitor watch with hard_stop = entry × 0.98.
  4. Reconciler runs at end of sync_positions:
     - tracked_symbols = set(open_positions) — NOW includes SMU.
     - Reconciler skips SMU as already-tracked.
  5. Net: reconciler bypassed, possibly-wrong stop, tight retry loop.

Race-A fix: recovery loops skip rows where
order_status='exit_pending_verification'. This test pins that fix for
both MACD wave and ORB.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest


# =========================================================================
# Shared helpers
# =========================================================================

def _alpaca_position_obj(symbol='SMU', qty=100, avg=14.67):
    """Mimic Alpaca's Position SDK object — attribute access, not dict."""
    p = MagicMock()
    p.symbol = symbol
    p.qty = qty
    p.avg_entry_price = avg
    p.unrealized_pl = -100.0
    p.market_value = qty * avg
    return p


def _alpaca_position_dict(symbol='SMU', qty=100, avg=14.67):
    """Dict form (used by orb_engine.py::sync_positions via
    AlpacaClient.get_open_positions)."""
    return {
        'symbol': symbol, 'qty': qty, 'avg_entry_price': avg,
        'unrealized_pl': -100.0, 'market_value': qty * avg,
        'side': 'long',
    }


# =========================================================================
# MACD wave
# =========================================================================

class TestMACDWaveDoesNotRehydrateExitPending:

    def _engine(self):
        from trading.macd_wave_engine import MACDWaveEngine
        eng = MACDWaveEngine.__new__(MACDWaveEngine)
        eng.STRATEGY_NAME = 'macd_wave'
        eng.db = MagicMock()
        eng.alpaca = MagicMock()
        eng.notifier = None
        eng.stop_monitor = MagicMock()
        eng.daily_pnl = 0.0
        eng.invalidated = set()
        eng.open_positions = {}
        eng.hard_stop_pct = 0.02
        eng.trail_stop_pct = 0.003
        eng.trail_arm_pct = 0.003
        return eng

    def test_exit_pending_verification_not_rehydrated(self):
        """The contract: an exit_pending_verification row whose broker
        position is still open MUST NOT be re-added to open_positions
        or get a fresh StopMonitor watch.
        """
        eng = self._engine()
        # DB returns the poisoned row
        pending_row = {
            'id': 357, 'symbol': 'SMU', 'strategy': 'macd_wave',
            'order_status': 'exit_pending_verification',
            'fill_price': 14.6722, 'filled_qty': 100, 'shares': 100,
            'exit_price': None, 'exit_reason': 'stop_loss_unconfirmed',
            'entry_price': 14.6722,
        }
        eng.db.get_open_trades.return_value = [pending_row]
        # Broker still has the position
        eng.alpaca.trading_client.get_all_positions.return_value = [
            _alpaca_position_obj('SMU', 100, 14.6722),
        ]
        # We bypass the orphan-reconciler call at the end of
        # sync_positions to keep this test focused on the recovery loop;
        # if the reconciler tried to act it would talk to mocks that
        # don't break the contract we're pinning here.
        eng.orphan_reconciler_cfg = None

        eng.sync_positions()

        # Race-A invariants:
        assert 'SMU' not in eng.open_positions, (
            "exit_pending_verification row was rehydrated — race A "
            "regrew. Skip these rows in sync_positions's recovery loop."
        )
        # No StopMonitor watch should have been registered
        eng.stop_monitor.add_watch.assert_not_called()

    def test_pending_new_still_rehydrated_via_legitimate_orphan_path(self):
        """Sanity: the unrelated pending_new orphan recovery path
        (Phase 2 / 2026-04-16) is preserved. A real pending_new row
        whose fill landed on the broker but never reached the DB DOES
        get recovered."""
        eng = self._engine()
        # NOTE: this is a different code path (orphan loop, lines
        # 394-466), not the recovery loop we're guarding. Pending_new
        # rows still flow through the orphan path. We assert the
        # pending_new path is NOT broken by the race-A fix.
        pending_new_row = {
            'id': 999, 'symbol': 'NEWX', 'strategy': 'macd_wave',
            'order_status': 'pending_new',
            'fill_price': None, 'filled_qty': 0, 'shares': 100,
            'exit_price': None, 'exit_reason': None,
            'entry_price': 10.0,
        }
        eng.db.get_open_trades.return_value = [pending_new_row]
        eng.alpaca.trading_client.get_all_positions.return_value = [
            _alpaca_position_obj('NEWX', 100, 10.0),
        ]

        eng.sync_positions()

        # NEWX should be recovered via the pending_new orphan path
        # (this is the LEGITIMATE Phase-2 recovery behaviour).
        assert 'NEWX' in eng.open_positions, (
            "pending_new orphan recovery (separate from race-A) "
            "regressed. The race-A fix only targets exit_pending_verification."
        )


# =========================================================================
# ORB
# =========================================================================

class TestORBDoesNotRehydrateExitPending:

    def _engine(self):
        from trading.orb_engine import ORBEngine
        eng = ORBEngine.__new__(ORBEngine)
        eng.db = MagicMock()
        eng.alpaca = MagicMock()
        eng.notifier = None
        eng.stop_monitor = MagicMock()
        eng.open_positions = {}
        eng.daily_pnl = 0.0
        eng.candidates = {}
        # Reconciler config — not exercised here, but the attribute is
        # read by the call at the end of sync_positions.
        eng.orphan_reconciler_cfg = None
        return eng

    def test_exit_pending_verification_not_rehydrated(self):
        eng = self._engine()
        pending_row = {
            'id': 386, 'symbol': 'QBTZ', 'strategy': 'orb',
            'order_status': 'exit_pending_verification',
            'fill_price': 3.75, 'filled_qty': 9983, 'shares': 9983,
            'entry_price': 3.75, 'stop_loss_price': 3.65,
            'pattern_data': '{}', 'order_id': '',
            'exit_price': None, 'exit_reason': 'stop_loss_unconfirmed',
        }
        eng.db.get_open_trades.return_value = [pending_row]
        eng.alpaca.get_open_positions.return_value = [
            _alpaca_position_dict('QBTZ', 5657, 3.75),
        ]
        # Stub out trading_client.get_orders (used to find pending
        # state-B orders) to return empty.
        eng.alpaca.trading_client = MagicMock()
        eng.alpaca.trading_client.get_orders.return_value = []

        eng.sync_positions()

        assert 'QBTZ' not in eng.open_positions, (
            "ORB State A rehydrate accepted an exit_pending_verification "
            "row — race A regrew."
        )
        eng.stop_monitor.add_watch.assert_not_called()
