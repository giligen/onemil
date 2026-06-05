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
        db.get_trades_by_date.return_value = [
            {'symbol': 'CORD', 'strategy': 'orb'},
        ]
        eng = _mk_engine(monkeypatch, a, db)
        eng._close_orphan_positions(trades_today=[])
        # close_position should NOT have been called — not ours
        a.close_position.assert_not_called()
        # symbol is marked traded to keep this node from trading it
        assert 'CORD' in eng._traded_symbols

    def test_bf_trade_registered_in_traded_symbols(self, monkeypatch):
        """A pre-existing broker position (whether ours or not) is
        registered in _traded_symbols so bull flag won't re-trade it.

        2026-06-05: _close_orphan_positions no longer calls
        close_position directly — that close was the SMU/QBTZ-class
        vulnerability (no avg-entry check, no stale-signal check, would
        flatten same-symbol unknown-strategy positions). The shared
        orphan_reconciler handles the close decision separately."""
        a = _mk_alpaca()
        a.get_open_positions.return_value = [{'symbol': 'XYZ'}]
        a.trading_client.get_orders.return_value = []
        db = _mk_db()
        db.get_trades_by_date.return_value = [
            {'symbol': 'XYZ', 'strategy': 'bull_flag'},
        ]
        eng = _mk_engine(monkeypatch, a, db)
        with patch('time.sleep'):
            eng._close_orphan_positions(trades_today=[])
        # No direct close from this method any more
        a.close_position.assert_not_called()
        # Symbol IS registered so the engine doesn't double-enter
        assert 'XYZ' in eng._traded_symbols

    def test_legacy_trade_without_strategy_also_registered(self, monkeypatch):
        """Pre-migration-8 trades lacking a `strategy` field still
        register in _traded_symbols. Same delegate-to-reconciler logic."""
        a = _mk_alpaca()
        a.get_open_positions.return_value = [{'symbol': 'OLD'}]
        a.trading_client.get_orders.return_value = []
        db = _mk_db()
        db.get_trades_by_date.return_value = [
            {'symbol': 'OLD'},   # no 'strategy' key
        ]
        eng = _mk_engine(monkeypatch, a, db)
        with patch('time.sleep'):
            eng._close_orphan_positions(trades_today=[])
        a.close_position.assert_not_called()
        assert 'OLD' in eng._traded_symbols


class TestCloseOrphanWithBracketLegs:
    """The cancel-then-close ordering used to live here. As of
    2026-06-05 the close step moved to trading.orphan_reconciler, which
    has its own bracket-cancel logic (via close_position internally) and
    a hardened ownership predicate. _close_orphan_positions only
    registers the symbol now; the cancel + close behaviour is exercised
    by tests/test_orphan_reconciler.py instead."""

    def test_close_orphan_positions_does_not_close_or_cancel(self, monkeypatch):
        a = _mk_alpaca()
        a.get_open_positions.return_value = [{'symbol': 'XYZ'}]
        a.trading_client.get_orders.return_value = [
            _mk_order('sl-1', 'SELL', 'STOP', 100),
        ]
        db = _mk_db()
        db.get_trades_by_date.return_value = [
            {'symbol': 'XYZ', 'strategy': 'bull_flag'}
        ]
        eng = _mk_engine(monkeypatch, a, db)
        with patch('time.sleep'):
            eng._close_orphan_positions(trades_today=[])
        # Neither close nor cancel is called from this method any more.
        a.close_position.assert_not_called()
        a.cancel_order.assert_not_called()
        # But the symbol is still registered so we don't double-enter.
        assert 'XYZ' in eng._traded_symbols
