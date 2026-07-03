"""FABC 2026-06-09 regression: _escalate_to_market_close must retry on
held_for_orders (40310000) before giving up.

Pre-fix: `client.close_position()` in escalate was called once with no
retry. When the bracket cancel hadn't propagated (~typical 1-2s window),
Alpaca returned 40310000, escalate fell through to BRANCH_LAST_RESORT
with NO emergency stop, position was naked. FABC bled -$4,145 over the
planned -$3,294 stop before manual recovery.

These tests pin the retry + emergency-stop contract.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from trading.stop_monitor import StopMonitor, WatchEntry


def _stub_alpaca():
    a = MagicMock()
    # Cancel paths
    a.cancel_order = MagicMock(return_value=True)
    a.trading_client = MagicMock()
    a.trading_client.get_orders = MagicMock(return_value=[])
    return a


def _watch(symbol='FABC', shares=10627, stop=3.99):
    return WatchEntry(
        symbol=symbol, stop_price=stop, shares=shares,
        tp_leg_id='tp-1', sl_leg_id='sl-1',
        trade_db_id=1, entry_price=4.30,
    )


def _make_monitor():
    m = StopMonitor.__new__(StopMonitor)
    m.notifier = MagicMock()
    m._STOP_EXIT_FILL_TIMEOUT_S = 1.0
    m._MARKET_CLOSE_FILL_TIMEOUT_S = 1.0
    m._HELD_QTY_RETRY_BACKOFFS_S = (0.01, 0.01, 0.01)  # speed up tests
    return m


def _smart_get_order(symbol='FABC', limit_oid='limit-1', close_oid='mkt-close-1',
                       sl_oid='sl-1', close_filled_px=3.60,
                       sl_filled_px=None):
    """Per-order_id get_order stub. By default the limit order looks
    pending (we want escalate to proceed past the limit-race check), the
    close order returns filled, the SL leg returns whatever
    sl_filled_px is.
    """
    def _get(oid):
        oid = str(oid)
        if oid == limit_oid:
            return {'status': 'new', 'filled_qty': 0, 'filled_avg_price': None}
        if oid == close_oid:
            return {'status': 'filled', 'filled_qty': 10627,
                    'filled_avg_price': close_filled_px}
        if oid == sl_oid:
            if sl_filled_px is None:
                return {'status': 'canceled', 'filled_qty': 0,
                        'filled_avg_price': None}
            return {'status': 'filled', 'filled_qty': 10627,
                    'filled_avg_price': sl_filled_px}
        return {'status': 'new', 'filled_qty': 0, 'filled_avg_price': None}
    return _get


class TestEscalateRetriesHeldQty:

    def test_retry_eventually_succeeds(self):
        """Held_qty fails the first 2 attempts then succeeds — escalate
        returns BRANCH_MARKET_CLOSE."""
        a = _stub_alpaca()
        attempts = {'n': 0}

        def close_position(symbol):
            attempts['n'] += 1
            if attempts['n'] < 3:
                raise Exception(
                    'insufficient qty available, code=40310000, '
                    'held_for_orders=10627'
                )
            return {'id': 'mkt-close-1', 'status': 'accepted'}

        a.close_position = close_position
        a.get_order = _smart_get_order()

        m = _make_monitor()
        watch = _watch()
        loop = asyncio.new_event_loop()
        try:
            price, oid, branch = loop.run_until_complete(
                m._escalate_to_market_close(
                    a, 'FABC', 'limit-1', 3.97, sl_leg_id='sl-1', watch=watch,
                )
            )
        finally:
            loop.close()

        assert attempts['n'] == 3
        assert branch == StopMonitor.BRANCH_MARKET_CLOSE
        assert oid == 'mkt-close-1'
        # Returned a valid price
        assert price is not None and price > 0

    def test_retry_exhausted_places_emergency_stop(self):
        """All 4 attempts fail with held_qty → BRANCH_LAST_RESORT, AND
        emergency stop was placed at the broker."""
        a = _stub_alpaca()

        def close_position(symbol):
            raise Exception(
                'insufficient qty available, code=40310000, '
                'held_for_orders=10627'
            )

        a.close_position = close_position
        a.get_order = _smart_get_order()
        a.submit_stop_sell_order = MagicMock(return_value={'id': 'em-1'})

        m = _make_monitor()
        watch = _watch()
        loop = asyncio.new_event_loop()
        try:
            price, oid, branch = loop.run_until_complete(
                m._escalate_to_market_close(
                    a, 'FABC', 'limit-1', 3.97, sl_leg_id='sl-1', watch=watch,
                )
            )
        finally:
            loop.close()

        assert branch == StopMonitor.BRANCH_LAST_RESORT
        # Emergency stop was placed — this is the FABC fix
        a.submit_stop_sell_order.assert_called_once()
        kw = a.submit_stop_sell_order.call_args.kwargs
        assert kw['symbol'] == 'FABC'
        assert kw['qty'] == 10627
        # stop_price = min(trigger, watch.stop) * 0.99 = min(3.97, 3.99) * 0.99
        assert kw['stop_price'] == pytest.approx(round(3.97 * 0.99, 2))

    def test_non_held_qty_error_does_not_retry_but_still_emergency_stop(self):
        """Some other error (e.g., 500) — don't retry, but DO place
        emergency stop before returning LAST_RESORT."""
        a = _stub_alpaca()
        attempts = {'n': 0}

        def close_position(symbol):
            attempts['n'] += 1
            raise Exception('alpaca 500 internal server error')

        a.close_position = close_position
        a.get_order = _smart_get_order()
        a.submit_stop_sell_order = MagicMock(return_value={'id': 'em-1'})

        m = _make_monitor()
        watch = _watch()
        loop = asyncio.new_event_loop()
        try:
            _, _, branch = loop.run_until_complete(
                m._escalate_to_market_close(
                    a, 'FABC', 'limit-1', 3.97, sl_leg_id='sl-1', watch=watch,
                )
            )
        finally:
            loop.close()

        assert branch == StopMonitor.BRANCH_LAST_RESORT
        # No retry for non-held-qty errors
        assert attempts['n'] == 1
        # Emergency stop still placed
        a.submit_stop_sell_order.assert_called_once()

    def test_position_flat_race_recovers_via_sl_leg_no_emergency_stop(self):
        """If close fails with 'cannot be sold short' AND SL leg has a
        fill price, return BRANCH_SL_LEG_RACE with the recovered price.
        No emergency stop because position is genuinely flat."""
        a = _stub_alpaca()
        a.close_position = MagicMock(side_effect=Exception(
            'position not found 40410000'
        ))
        # Limit pending (so we proceed past limit-race), SL has fill
        a.get_order = _smart_get_order(sl_filled_px=3.99)
        a.submit_stop_sell_order = MagicMock(return_value={'id': 'em-1'})

        m = _make_monitor()
        watch = _watch()
        loop = asyncio.new_event_loop()
        try:
            price, oid, branch = loop.run_until_complete(
                m._escalate_to_market_close(
                    a, 'FABC', 'limit-1', 3.97, sl_leg_id='sl-1', watch=watch,
                )
            )
        finally:
            loop.close()

        assert branch == StopMonitor.BRANCH_SL_LEG_RACE
        assert price == 3.99  # actual SL leg fill price
        assert oid == 'sl-1'
        # CRITICAL: no emergency stop placed when position was already flat
        a.submit_stop_sell_order.assert_not_called()
