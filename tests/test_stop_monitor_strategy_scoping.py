"""Regression for the 2026-04-24 SMCX cross-strategy bleed bug.

Bug: `TradingEngine._check_exhaustion_exits` iterated
`stop_monitor.watched_symbols` (the full set) and tried to sell every
symbol including ORB-owned positions on bull flag's Alpaca account.
Alpaca rejected with 42210000 ("cannot be sold short" — no position on
that account), filling logs with ERROR lines every 60s. Secondary
effect: bull flag hadn't opened a new trade since ORB launched on
2026-04-20.

`TradingEngine._force_close_all` had the same pattern — it would have
removed ORB / MACD watches from the shared StopMonitor at 15:45 ET,
breaking their own exit paths.

Fix: new `StopMonitor.watched_symbols_for(strategy)` returns only
watches whose `WatchEntry.strategy` matches. Both bull-flag callers
(`_check_exhaustion_exits`, `_force_close_all`) now use it.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from data_sources.alpaca_client import AlpacaClient
from trading.stop_monitor import StopMonitor


@pytest.fixture
def monitor():
    client = MagicMock(spec=AlpacaClient)
    client.cancel_order.return_value = True
    client.submit_limit_sell_order.return_value = {'id': 'x', 'status': 'accepted'}
    client.close_position.return_value = {'id': 'y', 'status': 'accepted'}
    client.get_order.return_value = {
        'id': 'x', 'status': 'filled',
        'filled_avg_price': 1.0, 'filled_qty': 100,
    }
    return StopMonitor(
        api_key='k', api_secret='s', alpaca_client=client,
        marketable_limit_offset=0.03, marketable_limit_offset_pct=0.005,
    )


class TestWatchedSymbolsScoping:

    def test_watched_symbols_returns_all_regardless_of_strategy(self, monitor):
        """Unfiltered property keeps prior semantics — callers that want
        'everything' still get it."""
        monitor.add_watch('FLAG1', 10.0, 100, 'tp', 'sl', strategy='bull_flag')
        monitor.add_watch('ORB1',  5.0, 200, 'tp', 'sl', strategy='orb')
        monitor.add_watch('MACD1', 15.0, 50, 'tp', 'sl', strategy='macd_wave')
        assert set(monitor.watched_symbols) == {'FLAG1', 'ORB1', 'MACD1'}

    def test_watched_symbols_for_filters_by_strategy(self, monitor):
        monitor.add_watch('FLAG1', 10.0, 100, 'tp', 'sl', strategy='bull_flag')
        monitor.add_watch('FLAG2', 12.0, 100, 'tp', 'sl', strategy='bull_flag')
        monitor.add_watch('ORB1',  5.0, 200, 'tp', 'sl', strategy='orb')
        monitor.add_watch('MACD1', 15.0, 50, 'tp', 'sl', strategy='macd_wave')
        assert set(monitor.watched_symbols_for('bull_flag')) == {'FLAG1', 'FLAG2'}
        assert set(monitor.watched_symbols_for('orb')) == {'ORB1'}
        assert set(monitor.watched_symbols_for('macd_wave')) == {'MACD1'}

    def test_watched_symbols_for_unknown_strategy_empty(self, monitor):
        monitor.add_watch('FLAG1', 10.0, 100, 'tp', 'sl', strategy='bull_flag')
        assert monitor.watched_symbols_for('does-not-exist') == []

    def test_watched_symbols_for_when_empty(self, monitor):
        assert monitor.watched_symbols_for('bull_flag') == []

    def test_default_strategy_is_bull_flag(self, monitor):
        """add_watch without strategy= defaults to bull_flag — historical
        behavior preserved, documented in WatchEntry dataclass."""
        monitor.add_watch('X', 10.0, 100, 'tp', 'sl')
        assert monitor.watched_symbols_for('bull_flag') == ['X']
        assert monitor.watched_symbols_for('orb') == []


# ---------------------------------------------------------------------------
# Integration-flavor test: simulates the scenario from 2026-04-24 15:03:27
# ---------------------------------------------------------------------------


class TestSmcxBugScenario:
    """The exact shape of today's SMCX bug, reduced to a minimal test."""

    def test_bull_flag_exhaustion_loop_ignores_orb_symbol(self, monitor):
        """When bull flag's `_check_exhaustion_exits` iterates the watched
        list, it MUST NOT include ORB symbols. Previously it iterated
        the full `watched_symbols` and then tried to sell them on bull
        flag's Alpaca account, generating 42210000 errors.

        Reduced reproduction: add one of each strategy to the shared
        monitor and assert the bull-flag filter returns only its own
        symbol, never the ORB symbol.
        """
        # The monitor is shared across strategies — this is the real
        # production setup. ORB's SMCX was the symbol that blew up.
        monitor.add_watch('SMCX', 11.88, 3867, 'tp', 'sl', strategy='orb')
        monitor.add_watch('AAA', 4.00, 500, 'tp', 'sl', strategy='bull_flag')

        bull_flag_view = monitor.watched_symbols_for('bull_flag')
        assert 'SMCX' not in bull_flag_view, (
            "bull flag's scoped view must NOT include ORB's SMCX — this "
            "is the exact pollution that caused bull flag to try "
            "selling an ORB position on its own Alpaca account."
        )
        assert bull_flag_view == ['AAA']
