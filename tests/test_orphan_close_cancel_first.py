"""
Unit tests for TradingEngine._cancel_open_orders_for_symbol + the
_close_orphan_positions hand-off that uses it.

Context: 2026-05-19 incident — BF cross-strategy orphan cleanup tried to
close 4 ORB positions (CORD/LMRI/PURR/WAY) on restart and failed with
`insufficient qty available, held_for_orders=qty` because ORB's bracket
SL/TP legs were holding all the shares. Fix: cancel open orders for the
symbol before close_position. Plus filter the "is this ours" check by
strategy='bull_flag' so we don't claim other strategies' positions to
begin with.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.trading_engine import TradingEngine


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _mk_engine(monkeypatch, alpaca, db):
    """Construct a TradingEngine with the minimum wiring required to call
    the orphan-cleanup path. Bypasses heavy init by patching __init__."""
    eng = TradingEngine.__new__(TradingEngine)
    eng.alpaca = alpaca
    eng.db = db
    eng._traded_symbols = set()
    return eng


def _mk_alpaca():
    a = MagicMock(spec=AlpacaClient)
    a.trading_client = MagicMock()
    return a


def _mk_db():
    db = MagicMock(spec=Database)
    return db


def _mk_order(order_id='ord-1', side='SELL', order_type='STOP', qty=100):
    o = MagicMock()
    o.id = order_id
    o.side = MagicMock(value=side)
    o.order_type = MagicMock(value=order_type)
    o.qty = qty
    return o


# ---------------------------------------------------------------------------
# _cancel_open_orders_for_symbol
# ---------------------------------------------------------------------------

class TestCancelOpenOrders:
    def test_no_open_orders_returns_zero(self, monkeypatch):
        a = _mk_alpaca()
        a.trading_client.get_orders.return_value = []
        eng = _mk_engine(monkeypatch, a, _mk_db())
        # Skip the time.sleep
        with patch('time.sleep'):
            n = eng._cancel_open_orders_for_symbol('TEST')
        assert n == 0
        a.cancel_order.assert_not_called()

    def test_two_open_orders_cancelled_then_sleep(self, monkeypatch):
        """Both bracket legs (SL + TP) cancelled before the close call."""
        a = _mk_alpaca()
        a.trading_client.get_orders.return_value = [
            _mk_order('sl-1', 'SELL', 'STOP', 947),
            _mk_order('tp-1', 'SELL', 'LIMIT', 947),
        ]
        eng = _mk_engine(monkeypatch, a, _mk_db())
        with patch('time.sleep') as sleep_mock:
            n = eng._cancel_open_orders_for_symbol('LMRI')
        assert n == 2
        assert a.cancel_order.call_count == 2
        a.cancel_order.assert_any_call('sl-1')
        a.cancel_order.assert_any_call('tp-1')
        sleep_mock.assert_called_once_with(1.0)

    def test_per_order_cancel_failure_does_not_raise(self, monkeypatch):
        a = _mk_alpaca()
        a.trading_client.get_orders.return_value = [
            _mk_order('ok-1'), _mk_order('fail-1'),
        ]
        a.cancel_order.side_effect = [None, RuntimeError("alpaca 500")]
        eng = _mk_engine(monkeypatch, a, _mk_db())
        with patch('time.sleep'):
            n = eng._cancel_open_orders_for_symbol('LMRI')
        # cancellation count includes only successful cancels
        assert n == 1
        assert a.cancel_order.call_count == 2

    def test_get_orders_failure_returns_zero(self, monkeypatch):
        a = _mk_alpaca()
        a.trading_client.get_orders.side_effect = RuntimeError("flaky")
        eng = _mk_engine(monkeypatch, a, _mk_db())
        with patch('time.sleep'):
            n = eng._cancel_open_orders_for_symbol('LMRI')
        assert n == 0
        a.cancel_order.assert_not_called()

    def test_no_sleep_when_zero_orders(self, monkeypatch):
        a = _mk_alpaca()
        a.trading_client.get_orders.return_value = []
        eng = _mk_engine(monkeypatch, a, _mk_db())
        with patch('time.sleep') as sleep_mock:
            eng._cancel_open_orders_for_symbol('TEST')
        sleep_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Strategy filtering in _close_orphan_positions
# ---------------------------------------------------------------------------

class TestStrategyFiltering:
    """BF must NOT claim ORB / MACD-wave positions as 'ours'. Today's
    incident: ORB's CORD/LMRI/PURR/WAY were tagged ours by BF's lookback,
    leading to the failed close attempts."""

    def test_orb_trade_is_not_claimed_by_bf(self, monkeypatch):
        """When the only DB record for a symbol is strategy='orb', BF's
        orphan check must NOT mark it as ours."""
        a = _mk_alpaca()
        a.get_open_positions.return_value = [{'symbol': 'CORD'}]
        db = _mk_db()
        # No get_trade_by_symbol method → falls to the 5-day lookback
        del db.get_trade_by_symbol
        db.get_trades_by_date.return_value = [
            {'symbol': 'CORD', 'strategy': 'orb'},
        ]
        eng = _mk_engine(monkeypatch, a, db)
        eng._close_orphan_positions(trades_today=[])
        # close_position should NOT have been called — not ours
        a.close_position.assert_not_called()
        # symbol is marked traded to keep this node from trading it
        assert 'CORD' in eng._traded_symbols

    def test_bf_trade_is_claimed_by_bf(self, monkeypatch):
        """A BF strategy trade IS ours — cleanup proceeds."""
        a = _mk_alpaca()
        a.get_open_positions.return_value = [{'symbol': 'XYZ'}]
        a.trading_client.get_orders.return_value = []  # no orders to cancel
        db = _mk_db()
        del db.get_trade_by_symbol
        db.get_trades_by_date.return_value = [
            {'symbol': 'XYZ', 'strategy': 'bull_flag'},
        ]
        eng = _mk_engine(monkeypatch, a, db)
        with patch('time.sleep'):
            eng._close_orphan_positions(trades_today=[])
        a.close_position.assert_called_once_with('XYZ')

    def test_legacy_trade_without_strategy_defaults_to_bf(self, monkeypatch):
        """Pre-migration-8 trades may lack a `strategy` field; default
        should be bull_flag to preserve old behavior."""
        a = _mk_alpaca()
        a.get_open_positions.return_value = [{'symbol': 'OLD'}]
        a.trading_client.get_orders.return_value = []
        db = _mk_db()
        del db.get_trade_by_symbol
        db.get_trades_by_date.return_value = [
            {'symbol': 'OLD'},   # no 'strategy' key
        ]
        eng = _mk_engine(monkeypatch, a, db)
        with patch('time.sleep'):
            eng._close_orphan_positions(trades_today=[])
        a.close_position.assert_called_once_with('OLD')


class TestCloseOrphanWithBracketLegs:
    """Integration: orphan cleanup cancels then closes (the today-incident
    replay)."""

    def test_close_after_cancel(self, monkeypatch):
        a = _mk_alpaca()
        a.get_open_positions.return_value = [{'symbol': 'XYZ'}]
        a.trading_client.get_orders.return_value = [
            _mk_order('sl-1', 'SELL', 'STOP', 100),
            _mk_order('tp-1', 'SELL', 'LIMIT', 100),
        ]
        db = _mk_db()
        del db.get_trade_by_symbol
        db.get_trades_by_date.return_value = [
            {'symbol': 'XYZ', 'strategy': 'bull_flag'}
        ]
        eng = _mk_engine(monkeypatch, a, db)
        with patch('time.sleep'):
            eng._close_orphan_positions(trades_today=[])
        # Order of operations: cancels first, then close
        assert a.cancel_order.call_count == 2
        a.close_position.assert_called_once_with('XYZ')
