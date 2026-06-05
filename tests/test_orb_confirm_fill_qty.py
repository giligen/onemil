"""
Regression test: ORBEngine._confirm_fill must persist the ACTUAL filled qty
to the DB, not the original order qty.

Pre-2026-06-05 bug: when a stop-limit bracket entry partially filled (e.g.,
RPGL on 2026-06-04: ordered 3,596 sh, only 1 share filled due to a
microsecond stale-ask anomaly on a thin name), the engine correctly
updated the in-memory OpenPosition.shares to 1 and registered the
StopMonitor watch with shares=1, but the DB row kept `shares=3596` (from
_save_pending_trade) and `filled_qty=NULL` (never written). Any downstream
query that joined `shares × fill_price` for notional/exposure was wrong.

Fix: include `'shares'` and `'filled_qty'` in the fill_update dict.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from trading.orb_engine import OpenPosition


def _make_pos(symbol='TEST', order_qty=100) -> OpenPosition:
    """OpenPosition representing a pending bracket entry of `order_qty` shares."""
    return OpenPosition(
        symbol=symbol,
        entry_price=10.00,           # limit_price while pending
        stop_price=9.00,
        shares=order_qty,            # the ORDERED qty (matches DB row at submit time)
        trade_id=42,
        order_id='abc-pending',
        entry_time=datetime(2026, 6, 4, 13, 35, tzinfo=timezone.utc),
        range_high=10.00,
        range_low=9.00,
        lock_arm_at_r=1.75,
        lock_stop_r=0.5,
        composite_score=0.38,
        quintile='Q4',
        bar_close_price=10.00,
        order_submitted_at=datetime(2026, 6, 4, 13, 35, tzinfo=timezone.utc),
        entry_quote_ask=10.00,
    )


def _make_engine_skel():
    """Construct an ORBEngine instance with only the attributes _confirm_fill
    actually touches. Bypasses heavy init (universe, planner, etc.)."""
    from trading.orb_engine import ORBEngine
    eng = ORBEngine.__new__(ORBEngine)
    eng.db = MagicMock()
    eng.stop_monitor = None
    eng.notifier = None
    eng.touchgo_cfg = MagicMock(breakout_bar_source='market')
    eng.daily_n_filled = 0
    eng._bar_windows = {}
    eng.open_positions = {}
    eng.candidates = {}
    eng.safety_sl_pct = 0.10
    eng._notify_error = MagicMock()
    return eng


class TestConfirmFillPersistsActualFilledQty:
    """Regression for the RPGL 6-04 incident: the DB must reflect the actual
    filled qty, not the original order qty."""

    def test_partial_fill_writes_actual_qty(self):
        """1 share filled out of 3,596 ordered → DB gets shares=1, filled_qty=1.
        Reproduces the RPGL 2026-06-04 case exactly."""
        eng = _make_engine_skel()
        pos = _make_pos(symbol='RPGL', order_qty=3596)
        # Alpaca order_status as a dict (the path _process_pending_fills uses)
        order_status = {
            'filled_avg_price': 3.84,
            'filled_qty': 1,
            'filled_at': datetime(2026, 6, 4, 13, 36, 0, tzinfo=timezone.utc),
        }

        eng._confirm_fill(pos, order_status)

        # The DB update_trade call must include shares=1 AND filled_qty=1.
        assert eng.db.update_trade.call_count == 1
        trade_id, update = eng.db.update_trade.call_args.args
        assert trade_id == 42
        assert update.get('shares') == 1, (
            f"Expected shares=1 in DB update (actual fill); got "
            f"{update.get('shares')!r}"
        )
        assert update.get('filled_qty') == 1, (
            f"Expected filled_qty=1 in DB update; got "
            f"{update.get('filled_qty')!r}"
        )
        # In-memory position also updated correctly.
        assert pos.shares == 1

    def test_full_fill_writes_full_qty(self):
        """No partial: 100 sh ordered, 100 filled → DB gets shares=100, filled_qty=100."""
        eng = _make_engine_skel()
        pos = _make_pos(symbol='FULL', order_qty=100)
        order_status = {
            'filled_avg_price': 10.05,
            'filled_qty': 100,
            'filled_at': datetime(2026, 6, 4, 13, 36, 0, tzinfo=timezone.utc),
        }
        eng._confirm_fill(pos, order_status)
        _, update = eng.db.update_trade.call_args.args
        assert update['shares'] == 100
        assert update['filled_qty'] == 100

    def test_alpaca_object_path_also_writes_qty(self):
        """When the engine sees an Alpaca SDK object (not dict), same fields land."""
        eng = _make_engine_skel()
        pos = _make_pos(symbol='OBJ', order_qty=500)

        class _AlpacaOrder:
            filled_avg_price = '20.00'
            filled_qty = '250'
            filled_at = datetime(2026, 6, 4, 13, 36, tzinfo=timezone.utc)
        eng._confirm_fill(pos, _AlpacaOrder())
        _, update = eng.db.update_trade.call_args.args
        assert update['shares'] == 250
        assert update['filled_qty'] == 250

    def test_missing_filled_qty_falls_back_to_pos_shares(self):
        """Defensive: if Alpaca doesn't return filled_qty (degenerate case),
        fall back to the position's recorded order qty. We still write
        SOMETHING — never leave the DB with stale `shares` and NULL filled_qty."""
        eng = _make_engine_skel()
        pos = _make_pos(symbol='DEGEN', order_qty=200)
        order_status = {
            'filled_avg_price': 10.0,
            # filled_qty intentionally absent
            'filled_at': datetime(2026, 6, 4, 13, 36, tzinfo=timezone.utc),
        }
        eng._confirm_fill(pos, order_status)
        _, update = eng.db.update_trade.call_args.args
        assert update['shares'] == 200       # fallback to pos.shares
        assert update['filled_qty'] == 200   # written, not NULL
