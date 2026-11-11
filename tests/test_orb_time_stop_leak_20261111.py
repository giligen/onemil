"""ORB resting entry must never outlive its time stop across a restart (2026-11-11).

AXTL 9/29: trades.db id 382 showed filled_at 18:09 UTC, 4.5 h after the 60-min
time stop. Diagnosis: the order really filled ~13:41 UTC; seven restarts re-ran
_confirm_fill and the last one clobbered filled_at (fixed 9/30,
`already_confirmed`). This file pins the remaining rail: the broker's own submit
timestamp is the clock after a restart, and an adopted order past its deadline
is warned about and cancelled on the first tick.
"""
import logging
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from trading.orb_engine import ORBEngine

T0 = datetime(2026, 9, 29, 13, 35, 7, tzinfo=timezone.utc)


def _engine(order, db_submit):
    """Real ORBEngine methods on a skeleton instance, MagicMock'd broker/DB."""
    e = object.__new__(ORBEngine)
    e.STRATEGY_NAME = 'orb'
    e.alpaca = MagicMock()
    e.db = MagicMock()
    e.stop_monitor = None
    e.notifier = None
    e.open_positions = {}
    e.candidates = {}
    e.daily_n_placed = e.daily_n_filled = e.daily_n_time_stop_canceled = 0
    e.time_stop_minutes = 60
    e.tg_prefix = "[ORB]"
    e.alpaca.get_open_positions.return_value = []
    e.alpaca.trading_client.get_orders.return_value = [order]
    e.alpaca.get_order.return_value = {'filled_qty': 0}
    e.db.get_open_trades.return_value = [{
        'id': 382, 'symbol': 'AXTL', 'order_id': 'ax-1',
        'order_status': 'pending_new', 'order_submitted_at': db_submit,
        'created_at': db_submit, 'entry_price': 3.46, 'stop_loss_price': 3.21,
        'shares': 963,
        'pattern_data': '{"range_high": 3.46, "range_low": 3.21}',
    }]
    return e


def _tick(engine, now):
    """Run the cancel step of check_entries with the clock at ``now``."""
    with patch('trading.orb_engine.datetime') as dt:
        dt.now.return_value = now
        dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        engine._cancel_stale_pending_orders()


class TestAdoptedSubmitTs:
    def test_broker_datetime_is_used_when_earlier(self):
        o = SimpleNamespace(id='x', submitted_at=T0 - timedelta(hours=2))
        assert ORBEngine._adopted_order_submit_ts(o, T0) == T0 - timedelta(hours=2)

    def test_broker_iso_string_naive_is_utc(self):
        o = {'id': 'x', 'created_at': '2026-09-29T11:35:07'}
        assert ORBEngine._adopted_order_submit_ts(o, T0) == T0 - timedelta(hours=2)

    def test_missing_broker_ts_falls_back_to_db(self):
        assert ORBEngine._adopted_order_submit_ts(SimpleNamespace(id='x'), T0) == T0
        assert ORBEngine._adopted_order_submit_ts(None, T0) == T0
        assert ORBEngine._adopted_order_submit_ts(MagicMock(), T0) == T0

    def test_db_earlier_than_broker_wins(self):
        o = SimpleNamespace(submitted_at=T0 + timedelta(minutes=5))
        assert ORBEngine._adopted_order_submit_ts(o, T0) == T0


class TestRestartIntegration:
    def test_old_order_cancelled_on_first_tick_with_warning(self, caplog):
        # DB column wrongly says "just now"; broker says submitted 2 h ago.
        now = T0 + timedelta(hours=2)
        e = _engine(SimpleNamespace(id='ax-1', submitted_at=T0), db_submit=now)
        with patch('trading.orb_engine.datetime') as dt, \
                caplog.at_level(logging.WARNING, logger='trading.orb_engine'):
            dt.now.return_value = now
            dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            e.sync_positions()
        assert any('PAST its time stop' in r.getMessage() and 'ax-1' in r.getMessage()
                   and r.levelname == 'WARNING' for r in caplog.records)
        _tick(e, now)
        e.alpaca.cancel_order.assert_called_once_with('ax-1')
        e.db.update_trade.assert_any_call(382, {'order_status': 'time_stop_canceled'})
        assert e.daily_n_time_stop_canceled == 1
        assert 'AXTL' not in e.open_positions

    def test_young_order_kept_until_deadline(self, caplog):
        now = T0 + timedelta(minutes=10)
        e = _engine(SimpleNamespace(id='ax-1', submitted_at=T0), db_submit=T0)
        with patch('trading.orb_engine.datetime') as dt, \
                caplog.at_level(logging.WARNING, logger='trading.orb_engine'):
            dt.now.return_value = now
            dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            e.sync_positions()
        assert not any('PAST its time stop' in r.getMessage() for r in caplog.records)
        _tick(e, now)
        e.alpaca.cancel_order.assert_not_called()
        _tick(e, T0 + timedelta(minutes=61))
        e.alpaca.cancel_order.assert_called_once_with('ax-1')
