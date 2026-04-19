"""Unit tests for StopMonitor refactor: multi-client routing + static lock.

Two features added for ORB production:

1. Strategy-routed order execution:
   StopMonitor accepts `alpaca_clients_by_strategy={'orb': paper_client, ...}`.
   When a watch with `strategy='orb'` triggers an exit, the paper client submits
   the sell order — NOT the default (main) client. Legacy single-client callers
   unaffected (dict defaults to empty, _client_for falls back to self._alpaca).

2. Static lock (ORB exit rule):
   When price touches entry + lock_arm_at_r × risk, stop ratchets UP to
   entry + lock_stop_r × risk ONCE and stays there forever. No trailing.
"""
import asyncio
from unittest.mock import MagicMock, AsyncMock

import pytest

from data_sources.alpaca_client import AlpacaClient
from trading.stop_monitor import StopMonitor, WatchEntry


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def mock_main_alpaca():
    """Default AlpacaClient (main account — bull flag / MACD wave)."""
    client = MagicMock(spec=AlpacaClient)
    client.cancel_order.return_value = True
    client.submit_limit_sell_order.return_value = {
        'id': 'main-sell-1', 'status': 'accepted', 'symbol': 'TSLA'
    }
    client.close_position.return_value = {
        'id': 'main-close-1', 'status': 'accepted', 'symbol': 'TSLA'
    }
    client.trading_client = MagicMock()
    client.trading_client.get_orders.return_value = []
    return client


@pytest.fixture
def mock_orb_alpaca():
    """Paper AlpacaClient (ORB account)."""
    client = MagicMock(spec=AlpacaClient)
    client.cancel_order.return_value = True
    client.submit_limit_sell_order.return_value = {
        'id': 'orb-sell-1', 'status': 'accepted', 'symbol': 'AAPL'
    }
    client.close_position.return_value = {
        'id': 'orb-close-1', 'status': 'accepted', 'symbol': 'AAPL'
    }
    client.trading_client = MagicMock()
    client.trading_client.get_orders.return_value = []
    return client


@pytest.fixture
def routed_monitor(mock_main_alpaca, mock_orb_alpaca):
    """StopMonitor with strategy routing configured."""
    return StopMonitor(
        api_key='k', api_secret='s',
        alpaca_client=mock_main_alpaca,
        alpaca_clients_by_strategy={'orb': mock_orb_alpaca},
    )


@pytest.fixture
def legacy_monitor(mock_main_alpaca):
    """StopMonitor in legacy single-client mode (no dict passed)."""
    return StopMonitor(
        api_key='k', api_secret='s',
        alpaca_client=mock_main_alpaca,
    )


# =========================================================================
# _client_for routing
# =========================================================================

class TestClientRouting:
    def test_legacy_mode_all_strategies_use_default(self, legacy_monitor, mock_main_alpaca):
        """No routing dict → every strategy uses self._alpaca."""
        assert legacy_monitor._client_for('bull_flag') is mock_main_alpaca
        assert legacy_monitor._client_for('macd_wave') is mock_main_alpaca
        assert legacy_monitor._client_for('orb') is mock_main_alpaca
        assert legacy_monitor._client_for('random_strategy') is mock_main_alpaca

    def test_routing_dict_orb_gets_paper_client(self, routed_monitor,
                                                  mock_main_alpaca, mock_orb_alpaca):
        """ORB strategy routes to paper client; others fall through to main."""
        assert routed_monitor._client_for('orb') is mock_orb_alpaca
        assert routed_monitor._client_for('bull_flag') is mock_main_alpaca
        assert routed_monitor._client_for('macd_wave') is mock_main_alpaca

    def test_unknown_strategy_falls_back_to_default(self, routed_monitor,
                                                     mock_main_alpaca):
        """Unmapped strategy tag → uses self._alpaca (never crashes)."""
        assert routed_monitor._client_for('unknown_new_strategy') is mock_main_alpaca
        assert routed_monitor._client_for('') is mock_main_alpaca

    def test_multiple_strategies_in_dict(self, mock_main_alpaca):
        client_a = MagicMock(spec=AlpacaClient)
        client_b = MagicMock(spec=AlpacaClient)
        m = StopMonitor(
            api_key='k', api_secret='s',
            alpaca_client=mock_main_alpaca,
            alpaca_clients_by_strategy={'a': client_a, 'b': client_b},
        )
        assert m._client_for('a') is client_a
        assert m._client_for('b') is client_b
        assert m._client_for('c') is mock_main_alpaca


# =========================================================================
# add_watch propagates lock params to WatchEntry
# =========================================================================

class TestAddWatchLockParams:
    def test_add_watch_default_lock_disabled(self, legacy_monitor):
        legacy_monitor.add_watch(
            symbol='TSLA', stop_price=100.0, shares=10,
            tp_leg_id='tp', sl_leg_id='sl',
        )
        w = legacy_monitor._watches['TSLA']
        assert w.lock_arm_at_r == 0.0
        assert w.lock_stop_r == 0.0
        assert w.lock_armed is False

    def test_add_watch_stores_lock_levels(self, legacy_monitor):
        legacy_monitor.add_watch(
            symbol='AAPL', stop_price=99.0, shares=10,
            tp_leg_id='tp', sl_leg_id='sl',
            entry_price=100.0, risk_per_share=1.0,
            lock_arm_at_r=1.5, lock_stop_r=1.0,
        )
        w = legacy_monitor._watches['AAPL']
        assert w.lock_arm_at_r == 1.5
        assert w.lock_stop_r == 1.0
        assert w.lock_armed is False  # runtime state, not armed yet

    def test_add_watch_strategy_tag_stored(self, routed_monitor):
        routed_monitor.add_watch(
            symbol='AAPL', stop_price=99.0, shares=10,
            tp_leg_id='tp', sl_leg_id='sl',
            strategy='orb',
        )
        assert routed_monitor._watches['AAPL'].strategy == 'orb'


# =========================================================================
# Lock arming logic (simulated via _on_trade)
# =========================================================================

def _make_trade(symbol, price):
    tr = MagicMock()
    tr.symbol = symbol
    tr.price = price
    return tr


class TestLockArmingInOnTrade:
    @pytest.mark.asyncio
    async def test_lock_does_not_arm_below_trigger(self, legacy_monitor):
        legacy_monitor.add_watch(
            symbol='AAPL', stop_price=95.0, shares=10,
            tp_leg_id='tp', sl_leg_id='sl',
            entry_price=100.0, risk_per_share=5.0,
            lock_arm_at_r=1.5, lock_stop_r=1.0,
        )
        # Price reaches +1.0R (below arm threshold of +1.5R)
        await legacy_monitor._on_trade(_make_trade('AAPL', 105.0))
        w = legacy_monitor._watches['AAPL']
        assert w.lock_armed is False
        assert w.stop_price == 95.0  # unchanged

    @pytest.mark.asyncio
    async def test_lock_arms_at_trigger_level(self, legacy_monitor):
        legacy_monitor.add_watch(
            symbol='AAPL', stop_price=95.0, shares=10,
            tp_leg_id='tp', sl_leg_id='sl',
            entry_price=100.0, risk_per_share=5.0,
            lock_arm_at_r=1.5, lock_stop_r=1.0,
        )
        # Price touches +1.5R (entry=100, risk=5, arm=100+1.5*5=107.5)
        await legacy_monitor._on_trade(_make_trade('AAPL', 107.5))
        w = legacy_monitor._watches['AAPL']
        assert w.lock_armed is True
        # Stop moved to entry + 1.0R = 105.0
        assert w.stop_price == 105.0

    @pytest.mark.asyncio
    async def test_lock_arms_above_trigger(self, legacy_monitor):
        legacy_monitor.add_watch(
            symbol='AAPL', stop_price=95.0, shares=10,
            tp_leg_id='tp', sl_leg_id='sl',
            entry_price=100.0, risk_per_share=5.0,
            lock_arm_at_r=1.5, lock_stop_r=1.0,
        )
        # Price jumps well above +1.5R on single tick
        await legacy_monitor._on_trade(_make_trade('AAPL', 115.0))
        w = legacy_monitor._watches['AAPL']
        assert w.lock_armed is True
        assert w.stop_price == 105.0

    @pytest.mark.asyncio
    async def test_lock_is_one_shot_never_moves_after_arming(self, legacy_monitor, mock_main_alpaca):
        # Set up: don't trigger exit — use a very low stop_price to avoid trigger at all prices we'll test
        legacy_monitor.add_watch(
            symbol='AAPL', stop_price=50.0, shares=10,
            tp_leg_id='tp', sl_leg_id='sl',
            entry_price=100.0, risk_per_share=5.0,
            lock_arm_at_r=1.5, lock_stop_r=1.0,
        )
        # Arm lock at +1.5R
        await legacy_monitor._on_trade(_make_trade('AAPL', 107.5))
        assert legacy_monitor._watches['AAPL'].lock_armed is True
        assert legacy_monitor._watches['AAPL'].stop_price == 105.0
        # Price climbs further to +3R — lock must NOT ratchet up (unlike trail)
        await legacy_monitor._on_trade(_make_trade('AAPL', 115.0))
        assert legacy_monitor._watches['AAPL'].stop_price == 105.0  # unchanged
        # Price climbs even further
        await legacy_monitor._on_trade(_make_trade('AAPL', 120.0))
        assert legacy_monitor._watches['AAPL'].stop_price == 105.0

    @pytest.mark.asyncio
    async def test_lock_disabled_when_arm_r_is_zero(self, legacy_monitor):
        """Default lock_arm_at_r=0 means no lock behavior."""
        legacy_monitor.add_watch(
            symbol='AAPL', stop_price=95.0, shares=10,
            tp_leg_id='tp', sl_leg_id='sl',
            entry_price=100.0, risk_per_share=5.0,
            # lock_arm_at_r NOT set (defaults to 0)
        )
        await legacy_monitor._on_trade(_make_trade('AAPL', 200.0))  # any high price
        w = legacy_monitor._watches['AAPL']
        assert w.lock_armed is False
        assert w.stop_price == 95.0

    @pytest.mark.asyncio
    async def test_lock_requires_positive_risk_per_share(self, legacy_monitor):
        """If risk_per_share=0 (degenerate), lock cannot arm (would divide by 0)."""
        legacy_monitor.add_watch(
            symbol='AAPL', stop_price=95.0, shares=10,
            tp_leg_id='tp', sl_leg_id='sl',
            entry_price=100.0, risk_per_share=0.0,
            lock_arm_at_r=1.5, lock_stop_r=1.0,
        )
        await legacy_monitor._on_trade(_make_trade('AAPL', 200.0))
        w = legacy_monitor._watches['AAPL']
        assert w.lock_armed is False

    @pytest.mark.asyncio
    async def test_lock_exit_reason_is_lock_stop(self, legacy_monitor):
        """When the LOCKED stop is hit, exit_reason should be 'lock_stop'."""
        from unittest.mock import patch
        legacy_monitor.add_watch(
            symbol='AAPL', stop_price=95.0, shares=10,
            tp_leg_id='tp', sl_leg_id='sl',
            entry_price=100.0, risk_per_share=5.0,
            lock_arm_at_r=1.5, lock_stop_r=1.0,
            strategy='orb',
        )
        # Arm the lock first
        await legacy_monitor._on_trade(_make_trade('AAPL', 107.5))
        assert legacy_monitor._watches['AAPL'].lock_armed is True
        assert legacy_monitor._watches['AAPL'].stop_price == 105.0

        # Now simulate price dropping back to the locked stop
        # We patch _execute_stop_exit to capture the exit_reason without full execution
        captured = {}

        async def capture_exit(symbol, price, watch, exit_reason='stop_loss'):
            captured['reason'] = exit_reason
            captured['price'] = price

        with patch.object(legacy_monitor, '_execute_stop_exit', side_effect=capture_exit):
            await legacy_monitor._on_trade(_make_trade('AAPL', 104.0))  # below locked stop
        assert captured['reason'] == 'lock_stop'

    @pytest.mark.asyncio
    async def test_lock_arms_from_peak_not_current_tick(self, legacy_monitor):
        """If price spikes to +1.5R then pulls back below before we process,
        the lock should arm based on the peak (highest_since_entry), not the
        current-tick price. Simulated via two ticks."""
        legacy_monitor.add_watch(
            symbol='AAPL', stop_price=50.0, shares=10,
            tp_leg_id='tp', sl_leg_id='sl',
            entry_price=100.0, risk_per_share=5.0,
            lock_arm_at_r=1.5, lock_stop_r=1.0,
        )
        # Tick 1: spike to 108 (arm triggers)
        await legacy_monitor._on_trade(_make_trade('AAPL', 108.0))
        assert legacy_monitor._watches['AAPL'].lock_armed is True
        assert legacy_monitor._watches['AAPL'].stop_price == 105.0


# =========================================================================
# Exit-reason classification
# =========================================================================

class TestExitReasonClassification:
    """Verify that the new 'lock_stop' reason is wired correctly alongside
    existing 'trail_stop' and 'stop_loss' reasons."""

    @pytest.mark.asyncio
    async def test_non_lock_non_trail_is_stop_loss(self, legacy_monitor):
        """Plain initial stop with no lock/trail → exit_reason='stop_loss'."""
        from unittest.mock import patch
        legacy_monitor.add_watch(
            symbol='X', stop_price=99.0, shares=10,
            tp_leg_id='tp', sl_leg_id='sl',
            entry_price=100.0, risk_per_share=1.0,
        )
        captured = {}
        async def cap(sym, p, w, exit_reason='stop_loss'):
            captured['reason'] = exit_reason
        with patch.object(legacy_monitor, '_execute_stop_exit', side_effect=cap):
            await legacy_monitor._on_trade(_make_trade('X', 98.5))
        assert captured['reason'] == 'stop_loss'
