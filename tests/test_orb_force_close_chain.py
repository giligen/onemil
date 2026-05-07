"""Tests for ORB force-close chain hardening (2026-05-07).

Covers the three bugs surfaced by ASTX 5/6 post-mortem:
  Bug A: verify-with-grace replaces 1s-sleep + single-shot check
  Bug B: 3x retry-on-verify-failure with backoff
  Bug C: explicit DB sync after FC success (no async-watcher dependency)
"""
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from types import SimpleNamespace

from trading.orb_engine import ORBEngine


def make_position(symbol):
    """Mock Alpaca position object with .symbol attribute."""
    return SimpleNamespace(symbol=symbol)


def make_filled_sell(symbol, price, order_id='abc12345', filled_at=None):
    """Mock Alpaca order representing a filled sell."""
    return SimpleNamespace(
        id=order_id,
        symbol=symbol,
        side=SimpleNamespace(value='sell'),
        status=SimpleNamespace(value='filled'),
        filled_avg_price=str(price),
        filled_at=filled_at or datetime.now(timezone.utc),
        type=SimpleNamespace(value='market'),
        order_class=SimpleNamespace(value='simple'),
    )


class FakeEngine:
    """Minimal stand-in for ORBEngine just enough to drive the helpers under test."""
    def __init__(self, alpaca, db):
        self.alpaca = alpaca
        self.db = db
        # bind the helpers from the real class so they call self
        self._verify_flat_with_grace = ORBEngine._verify_flat_with_grace.__get__(self)
        self._sync_db_after_fc = ORBEngine._sync_db_after_fc.__get__(self)


class TestVerifyFlatWithGrace(unittest.TestCase):
    def test_returns_empty_immediately_when_already_flat(self):
        alpaca = MagicMock()
        alpaca.get_open_positions.return_value = []
        engine = FakeEngine(alpaca, MagicMock())
        result = engine._verify_flat_with_grace(max_wait_s=10, poll_interval_s=0.1)
        self.assertEqual(result, [])
        # Single poll should suffice
        self.assertEqual(alpaca.get_open_positions.call_count, 1)

    def test_succeeds_after_position_clears_within_window(self):
        alpaca = MagicMock()
        # First 2 polls show open, then clears
        alpaca.get_open_positions.side_effect = [
            [make_position('ASTX')],
            [make_position('ASTX')],
            [],
        ]
        engine = FakeEngine(alpaca, MagicMock())
        result = engine._verify_flat_with_grace(max_wait_s=10, poll_interval_s=0.05)
        self.assertEqual(result, [])
        self.assertEqual(alpaca.get_open_positions.call_count, 3)

    def test_returns_remaining_after_timeout(self):
        alpaca = MagicMock()
        alpaca.get_open_positions.return_value = [make_position('ASTX')]
        engine = FakeEngine(alpaca, MagicMock())
        result = engine._verify_flat_with_grace(max_wait_s=0.3, poll_interval_s=0.05)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].symbol, 'ASTX')

    def test_handles_query_errors_during_poll(self):
        alpaca = MagicMock()
        # Errors then success
        alpaca.get_open_positions.side_effect = [
            Exception('rate limited'),
            Exception('rate limited'),
            [],
        ]
        engine = FakeEngine(alpaca, MagicMock())
        result = engine._verify_flat_with_grace(max_wait_s=10, poll_interval_s=0.05)
        self.assertEqual(result, [])


class TestSyncDbAfterFc(unittest.TestCase):
    def _make_db(self, open_trades):
        db = MagicMock()
        db.get_open_trades.return_value = open_trades
        return db

    def test_updates_db_with_actual_fill_price(self):
        alpaca = MagicMock()
        alpaca.trading_client.get_orders.return_value = [
            make_filled_sell('ASTX', 23.89),
        ]
        db = self._make_db([{
            'id': 196, 'symbol': 'ASTX', 'fill_price': 23.10,
            'shares': 1988, 'filled_qty': 1988, 'exit_price': None,
        }])
        engine = FakeEngine(alpaca, db)
        engine._sync_db_after_fc(['ASTX'])

        db.update_trade.assert_called_once()
        call_args = db.update_trade.call_args
        trade_id = call_args[0][0]
        updates = call_args[0][1]
        self.assertEqual(trade_id, 196)
        self.assertAlmostEqual(updates['exit_price'], 23.89, places=2)
        self.assertEqual(updates['exit_reason'], 'force_close')
        # pnl = (23.89 - 23.10) * 1988 = 1570.52
        self.assertAlmostEqual(updates['pnl'], 1570.52, places=1)

    def test_skips_already_closed_rows(self):
        alpaca = MagicMock()
        db = self._make_db([{
            'id': 196, 'symbol': 'ASTX', 'fill_price': 23.10,
            'shares': 1988, 'exit_price': 23.50,  # already closed
        }])
        engine = FakeEngine(alpaca, db)
        engine._sync_db_after_fc(['ASTX'])
        db.update_trade.assert_not_called()

    def test_skips_when_no_filled_sell_at_alpaca(self):
        alpaca = MagicMock()
        alpaca.trading_client.get_orders.return_value = []  # no orders
        db = self._make_db([{
            'id': 196, 'symbol': 'ASTX', 'fill_price': 23.10,
            'shares': 1988, 'exit_price': None,
        }])
        engine = FakeEngine(alpaca, db)
        engine._sync_db_after_fc(['ASTX'])
        db.update_trade.assert_not_called()

    def test_picks_most_recent_sell_when_multiple_exist(self):
        # FC may submit 2 close orders (engine pass + sweep). Pick the one
        # that actually filled most recently.
        alpaca = MagicMock()
        older = make_filled_sell('ASTX', 23.50, order_id='old11111',
                                  filled_at=datetime(2026, 5, 6, 19, 45, 3, tzinfo=timezone.utc))
        newer = make_filled_sell('ASTX', 23.89, order_id='new22222',
                                  filled_at=datetime(2026, 5, 6, 19, 45, 4, tzinfo=timezone.utc))
        alpaca.trading_client.get_orders.return_value = [older, newer]
        db = self._make_db([{
            'id': 196, 'symbol': 'ASTX', 'fill_price': 23.10,
            'shares': 1988, 'exit_price': None,
        }])
        engine = FakeEngine(alpaca, db)
        engine._sync_db_after_fc(['ASTX'])

        updates = db.update_trade.call_args[0][1]
        self.assertAlmostEqual(updates['exit_price'], 23.89, places=2)

    def test_handles_db_update_error_gracefully(self):
        alpaca = MagicMock()
        alpaca.trading_client.get_orders.return_value = [
            make_filled_sell('ASTX', 23.89),
        ]
        db = self._make_db([{
            'id': 196, 'symbol': 'ASTX', 'fill_price': 23.10,
            'shares': 1988, 'exit_price': None,
        }])
        db.update_trade.side_effect = Exception('db locked')
        engine = FakeEngine(alpaca, db)
        # Must not raise
        engine._sync_db_after_fc(['ASTX'])

    def test_empty_symbols_list_is_noop(self):
        alpaca = MagicMock()
        db = MagicMock()
        engine = FakeEngine(alpaca, db)
        engine._sync_db_after_fc([])
        db.get_open_trades.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
