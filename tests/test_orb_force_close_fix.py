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
        """If close_position fails twice (first + retry), telegram fires
        with MANUAL ACTION REQUIRED."""
        engine.open_positions['ANNA'] = _pos()
        mock_alpaca.trading_client.get_orders.return_value = []
        mock_alpaca.close_position.side_effect = RuntimeError("insufficient qty available")
        engine.force_close_all()
        # _notify_error fired with manual-action phrasing
        msgs = [c[0][0] for c in notifier.send_message.call_args_list]
        critical = [m for m in msgs if 'FORCE-CLOSE FAILED' in m]
        assert len(critical) == 1
        assert 'MANUAL ACTION' in critical[0]
        assert 'ANNA' in critical[0]

    def test_close_retry_succeeds_no_alert(self, engine, mock_alpaca, notifier):
        """First close fails, retry succeeds → no critical telegram."""
        engine.open_positions['ANNA'] = _pos()
        mock_alpaca.trading_client.get_orders.return_value = []
        call_count = [0]
        def close_side(sym):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("transient")
            return {'id': 'retry-close'}
        mock_alpaca.close_position.side_effect = close_side
        engine.force_close_all()
        msgs = [c[0][0] for c in notifier.send_message.call_args_list]
        assert not any('FORCE-CLOSE FAILED' in m for m in msgs)


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
