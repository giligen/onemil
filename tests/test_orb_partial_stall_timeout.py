"""
FABC 2026-06-09 regression: partial-fill stall timeout.

When the broker reports `partially_filled` repeatedly without ever
transitioning to terminal `filled`, the engine must NOT poll forever.
After `partial_fill_stall_seconds_max` it cancels the unfilled remainder
and accepts the observed qty.

Companion to test_orb_fixes2.py::TestPartiallyFilledHandling — that file
covers the happy multi-poll path (partial → filled). This file covers
the stuck-broker safety net.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, OpenPosition
from trading.stop_monitor import StopMonitor


@pytest.fixture
def orb_cfg():
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    # Override the stall timeout for test responsiveness.
    cfg.setdefault('fill_handling', {})['partial_fill_stall_seconds_max'] = 1
    return cfg


@pytest.fixture
def mock_alpaca():
    c = MagicMock(spec=AlpacaClient)
    c.get_open_positions.return_value = []
    c.get_account_info.return_value = {'buying_power': 100_000.0}
    c.cancel_order.return_value = True
    c.get_daily_bars.return_value = {}
    return c


@pytest.fixture
def mock_db():
    db = MagicMock(spec=Database)
    db.save_trade.return_value = 1
    db.update_trade.return_value = True
    return db


@pytest.fixture
def mock_sm():
    sm = MagicMock(spec=StopMonitor)
    sm.polling_mode = False
    sm.drain_exit_events.return_value = []
    return sm


@pytest.fixture
def engine(orb_cfg, mock_alpaca, mock_db, mock_sm):
    return ORBEngine(alpaca_client=mock_alpaca, db=mock_db,
                     stop_monitor=mock_sm, config=orb_cfg)


def _open_pos(symbol='STUCK', shares=1000) -> OpenPosition:
    """Pending OpenPosition that has been pending long enough that the
    age guard (5s) in _process_pending_fills is irrelevant — REST fallback
    will fetch order_status without needing OrderStream."""
    return OpenPosition(
        symbol=symbol, entry_price=10.03, stop_price=9.50, shares=shares,
        trade_id=1, order_id='pending-stall',
        entry_time=datetime.now(timezone.utc) - timedelta(seconds=30),
        range_high=10.0, range_low=9.5,
        lock_arm_at_r=1.5, lock_stop_r=1.0,
        composite_score=0.5, quintile='Q4',
    )


class TestPartialFillStallTimeout:

    def test_partial_within_timeout_keeps_polling(self, engine, mock_alpaca, mock_sm):
        """During the timeout window the engine just polls — no cancel, no
        _confirm_fill, no StopMonitor watch."""
        pos = _open_pos()
        engine.open_positions['STUCK'] = pos
        mock_alpaca.get_order.return_value = {
            'status': 'partially_filled',
            'filled_avg_price': 10.04,
            'filled_qty': 400,
            'qty': 1000,
        }
        engine._process_pending_fills()
        # No cancel, no watch
        mock_alpaca.cancel_order.assert_not_called()
        mock_sm.add_watch.assert_not_called()
        # Polling state intact
        assert pos.order_id == 'pending-stall'
        assert pos.first_partial_at is not None
        assert pos.last_observed_filled_qty == 400

    def test_partial_beyond_timeout_cancels_and_confirms(
        self, engine, mock_alpaca, mock_sm
    ):
        """After partial_fill_stall_seconds_max elapses without a 'filled'
        event, the engine cancels the unfilled remainder + confirms at the
        observed qty. The fixture sets timeout=1s; we pre-age first_partial_at."""
        pos = _open_pos()
        # Simulate "first observed partial happened > 1s ago"
        pos.first_partial_at = datetime.now(timezone.utc) - timedelta(seconds=5)
        pos.last_observed_filled_qty = 400
        engine.open_positions['STUCK'] = pos
        # Broker still reports same partial — nothing new happened.
        mock_alpaca.get_order.return_value = {
            'status': 'partially_filled',
            'filled_avg_price': 10.04,
            'filled_qty': 400,
            'qty': 1000,
        }
        engine._process_pending_fills()
        # Stall path triggered: remainder cancelled + StopMonitor armed
        mock_alpaca.cancel_order.assert_called_once_with('pending-stall')
        mock_sm.add_watch.assert_called_once()
        # _confirm_fill ran with observed qty
        assert pos.shares == 400
        assert pos.order_id == ''  # cleared = filled

    def test_cancel_failure_still_confirms(self, engine, mock_alpaca, mock_sm, caplog):
        """If cancel raises (broker race), still _confirm_fill at observed qty.
        Logs ERROR so the orphan-detect sweep can clean up."""
        import logging as _logging
        caplog.set_level(_logging.ERROR, logger='trading.orb_engine')

        pos = _open_pos()
        pos.first_partial_at = datetime.now(timezone.utc) - timedelta(seconds=5)
        pos.last_observed_filled_qty = 750
        engine.open_positions['STUCK'] = pos
        mock_alpaca.get_order.return_value = {
            'status': 'partially_filled',
            'filled_avg_price': 10.04,
            'filled_qty': 750,
            'qty': 1000,
        }
        mock_alpaca.cancel_order.side_effect = Exception("broker race")

        engine._process_pending_fills()

        # Still confirmed despite cancel failure
        mock_sm.add_watch.assert_called_once()
        assert pos.shares == 750
        assert pos.order_id == ''
        # Error path emits a log so operations can react
        assert any(
            'stall-cancel FAILED' in r.getMessage()
            for r in caplog.records
        )


class TestFABCIncidentReplay:
    """The exact FABC 2026-06-09 timeline. Pre-fix: shares recorded as 1438.
    Post-fix: shares recorded as 3188."""

    def test_partial_1438_then_partial_3188_then_filled_records_3188(
        self, engine, mock_alpaca, mock_sm
    ):
        pos = _open_pos(symbol='FABC', shares=3188)
        engine.open_positions['FABC'] = pos

        # Poll 1: first partial 1438/3188
        mock_alpaca.get_order.return_value = {
            'status': 'partially_filled',
            'filled_avg_price': 3.97,
            'filled_qty': 1438,
            'qty': 3188,
        }
        engine._process_pending_fills()
        assert pos.order_id == 'pending-stall'
        mock_sm.add_watch.assert_not_called()

        # Poll 2: progressed to 3188 but still 'partially_filled'
        mock_alpaca.get_order.return_value = {
            'status': 'partially_filled',
            'filled_avg_price': 3.97,
            'filled_qty': 3188,
            'qty': 3188,
        }
        engine._process_pending_fills()
        # Broker still 'partially_filled' even at 100% fill → still polling
        # (engine doesn't infer terminal from qty match — waits for 'filled')
        assert pos.order_id == 'pending-stall'
        assert pos.last_observed_filled_qty == 3188
        mock_sm.add_watch.assert_not_called()

        # Poll 3: broker reports terminal 'filled'
        mock_alpaca.get_order.return_value = {
            'status': 'filled',
            'filled_avg_price': 3.97,
            'filled_qty': 3188,
            'qty': 3188,
        }
        engine._process_pending_fills()
        # Confirmed at FULL qty (3188), not the first-seen partial (1438)
        assert pos.shares == 3188
        assert pos.order_id == ''
        mock_sm.add_watch.assert_called_once()
