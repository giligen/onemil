"""Unit tests for StopMonitor._place_emergency_stop_fallback.

This helper is the single source of truth for the broker-side safety
net after our normal exit paths have failed. FABC 2026-06-09 made the
need for this helper explicit.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from trading.stop_monitor import StopMonitor, WatchEntry


def _watch(symbol='FABC', shares=10627, stop=3.99):
    return WatchEntry(
        symbol=symbol, stop_price=stop, shares=shares,
        tp_leg_id='tp-1', sl_leg_id='sl-1',
        trade_db_id=1, entry_price=4.30,
    )


def _make_monitor():
    m = StopMonitor.__new__(StopMonitor)
    m.notifier = MagicMock()
    m._HELD_QTY_RETRY_BACKOFFS_S = (0.01, 0.01, 0.01)
    return m


class TestEmergencyStopHelper:

    def test_success_returns_order_id_and_alerts_telegram(self):
        a = MagicMock()
        a.submit_stop_sell_order = MagicMock(return_value={'id': 'em-7'})
        m = _make_monitor()
        loop = asyncio.new_event_loop()
        try:
            em_id = loop.run_until_complete(
                m._place_emergency_stop_fallback(
                    loop, a, 'FABC', 3.97, watch=_watch(),
                    reason='close failed',
                )
            )
        finally:
            loop.close()
        assert em_id == 'em-7'
        # Telegram: 1 alert with EMERGENCY STOP message
        m.notifier.notify_error.assert_called_once()
        msg = m.notifier.notify_error.call_args[0][0]
        assert 'EMERGENCY STOP' in msg
        assert 'FABC' in msg

    def test_position_flat_race_no_telegram_no_alarm(self):
        """If position is already flat (bracket SL won race), the
        emergency stop submission fails with 'cannot be sold short'.
        That's expected — no Telegram noise."""
        a = MagicMock()
        a.submit_stop_sell_order = MagicMock(side_effect=Exception(
            'cannot be sold short 42210000'
        ))
        m = _make_monitor()
        loop = asyncio.new_event_loop()
        try:
            em_id = loop.run_until_complete(
                m._place_emergency_stop_fallback(
                    loop, a, 'FABC', 3.97, watch=_watch(),
                    reason='close failed',
                )
            )
        finally:
            loop.close()
        assert em_id is None
        # No telegram for the expected case
        m.notifier.notify_error.assert_not_called()

    def test_emergency_stop_failure_pages_naked(self):
        """If the emergency stop ALSO fails with a non-race error, the
        position is truly naked — page the operator."""
        a = MagicMock()
        a.submit_stop_sell_order = MagicMock(side_effect=Exception(
            'alpaca 500 internal'
        ))
        m = _make_monitor()
        loop = asyncio.new_event_loop()
        try:
            em_id = loop.run_until_complete(
                m._place_emergency_stop_fallback(
                    loop, a, 'FABC', 3.97, watch=_watch(),
                    reason='close failed',
                )
            )
        finally:
            loop.close()
        assert em_id is None
        m.notifier.notify_error.assert_called_once()
        msg = m.notifier.notify_error.call_args[0][0]
        assert 'NAKED' in msg
        assert 'FABC' in msg

    def test_zero_shares_skipped_with_alert(self):
        a = MagicMock()
        a.submit_stop_sell_order = MagicMock(return_value={'id': 'em-x'})
        m = _make_monitor()
        loop = asyncio.new_event_loop()
        try:
            em_id = loop.run_until_complete(
                m._place_emergency_stop_fallback(
                    loop, a, 'FABC', 3.97,
                    watch=WatchEntry(
                        symbol='FABC', stop_price=3.99, shares=0,
                        tp_leg_id='', sl_leg_id='',
                        trade_db_id=1, entry_price=4.30,
                    ),
                    reason='close failed',
                )
            )
        finally:
            loop.close()
        assert em_id is None
        a.submit_stop_sell_order.assert_not_called()
        m.notifier.notify_error.assert_called_once()
        assert 'NAKED' in m.notifier.notify_error.call_args[0][0]

    def test_stop_price_is_min_trigger_watch_stop_times_99pct(self):
        a = MagicMock()
        a.submit_stop_sell_order = MagicMock(return_value={'id': 'em-1'})
        m = _make_monitor()
        # trigger $3.97, watch.stop $3.99 → min × 0.99 = 3.93
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                m._place_emergency_stop_fallback(
                    loop, a, 'FABC', 3.97, watch=_watch(), reason='',
                )
            )
        finally:
            loop.close()
        kw = a.submit_stop_sell_order.call_args.kwargs
        assert kw['stop_price'] == pytest.approx(round(3.97 * 0.99, 2))

        # Now flip: watch.stop $3.50, trigger $3.97 → min × 0.99 = 3.46
        a.submit_stop_sell_order.reset_mock()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                m._place_emergency_stop_fallback(
                    loop, a, 'FABC', 3.97,
                    watch=_watch(stop=3.50), reason='',
                )
            )
        finally:
            loop.close()
        kw = a.submit_stop_sell_order.call_args.kwargs
        assert kw['stop_price'] == pytest.approx(round(3.50 * 0.99, 2))
