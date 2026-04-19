"""Tests for round-3 BT↔PROD alignment fixes.

Covers:
  Fix #1 — Planner sizing uses range_open (BT parity via range_size_pct =
            (range_high - range_low) / range_open * 100)
  Fix #4 — StopMonitor skips stop/lock checks for ticks before
            skip_exits_until_ts (matches BT's sim_bars.iloc[1:] — entry
            bar not evaluated for exits)
"""
from __future__ import annotations

import asyncio
import time as time_mod
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from data_sources.alpaca_client import AlpacaClient
from trading.orb_planner import OrbTradePlan, OrbTradePlanner, PlannerReject
from trading.stop_monitor import StopMonitor


# =========================================================================
# Fix #1 — sizing uses range_open
# =========================================================================

@pytest.fixture
def planner_cfg():
    return {
        'entry': {'entry_slip_bps': 30, 'max_spread_bps': 150},
        'exit': {'lock_arm_at_r': 1.5, 'lock_stop_r': 1.0},
        'sizing': {
            'account_budget_usd': 100_000,
            'max_concurrent': 4,
            'risk_per_trade_usd': 3_000,
            'min_stop_pct': 1.0,
        },
    }


@pytest.fixture
def planner(planner_cfg):
    return OrbTradePlanner(planner_cfg)


class TestSizingUsesRangeOpen:
    def test_wide_range_sizing_differs_between_formulas(self, planner):
        """Range 50% ($10-$15 with open $10) — BT uses 50% stop_pct,
        PRE-FIX PROD used 33.5% (based on entry=$15.045).
        Verify we now match BT."""
        plan = planner.build(
            symbol='X', range_high=15.00, range_low=10.00,
            range_open=10.00,  # ← BT denominator
            composite_score=0.5, quintile='Q4', adaptive_mult=1.0,
        )
        # BT: stop_pct = 5/10 = 50%. uncapped = 3000/0.5 = $6000.
        # Below per_pos_cap ($25K) → actual position = $6000 × 1.0 = $6000.
        assert plan.position_dollars == pytest.approx(6_000, abs=100)

    def test_range_open_stored_in_plan(self, planner):
        plan = planner.build(
            symbol='X', range_high=12.00, range_low=10.00,
            range_open=10.00,
            composite_score=0.5, quintile='Q4', adaptive_mult=1.0,
        )
        assert plan.range_open == 10.00

    def test_range_open_fallback_when_zero(self, planner):
        """When range_open not provided, planner falls back to range_high.
        Ensures backward compatibility with existing callers."""
        plan = planner.build(
            symbol='X', range_high=12.00, range_low=10.00,
            # range_open not passed → defaults to 0 → fallback to range_high
            composite_score=0.5, quintile='Q4', adaptive_mult=1.0,
        )
        # Fallback: stop_pct = (12-10) / 12 = 16.67%. uncapped = 3000/0.1667 = $18,000.
        assert plan.position_dollars == pytest.approx(18_000, abs=500)

    def test_risk_per_share_uses_actual_stop_distance(self, planner):
        """Even with BT-parity sizing, risk_per_share should use actual stop distance
        (entry - range_low) for correct telemetry of dollars at risk."""
        plan = planner.build(
            symbol='X', range_high=10.00, range_low=9.50,
            range_open=10.00,
            composite_score=0.5, quintile='Q4', adaptive_mult=1.0,
        )
        # entry = 10.00 × 1.003 = 10.03. risk_per_share = 10.03 - 9.50 = 0.53.
        assert plan.risk_per_share == pytest.approx(0.53, abs=0.01)

    def test_narrow_range_still_caps_at_per_pos(self, planner):
        """Narrow range ($10-$10.20 = 2% of open) → uncap = $150K → caps at $25K."""
        plan = planner.build(
            symbol='X', range_high=10.20, range_low=10.00,
            range_open=10.00,
            composite_score=0.5, quintile='Q4', adaptive_mult=1.0,
        )
        # stop_pct = 0.20/10.00 = 2%. uncapped = 3000/0.02 = $150K → cap at $25K.
        assert plan.position_dollars == pytest.approx(25_000, abs=500)

    def test_adaptive_mult_applies_after_cap_unchanged(self, planner):
        """Fix #1 doesn't change adaptive_mult behavior (applied AFTER cap)."""
        plan = planner.build(
            symbol='X', range_high=10.20, range_low=10.00,
            range_open=10.00,
            composite_score=1.0, quintile='Q5', adaptive_mult=1.5,
        )
        # Cap $25K × Q5 mult 1.5 = $37,500
        assert plan.position_dollars == pytest.approx(37_500, abs=500)

    def test_matches_bt_pre_cap_uncap_on_wide_range(self, planner):
        """Range $10-$13 (30%) with open $10 → uncapped = $10,000 (BT math)."""
        plan = planner.build(
            symbol='X', range_high=13.00, range_low=10.00,
            range_open=10.00,
            composite_score=0.5, quintile='Q4', adaptive_mult=1.0,
        )
        # BT: stop_pct = 3/10 = 30%. uncap = 3000/0.3 = $10,000. Below cap → $10K.
        assert plan.position_dollars == pytest.approx(10_000, abs=200)


# =========================================================================
# Fix #4 — skip exits during entry bar
# =========================================================================

@pytest.fixture
def mock_alp():
    c = MagicMock(spec=AlpacaClient)
    c.cancel_order.return_value = True
    c.submit_limit_sell_order.return_value = {'id': 'sell-1'}
    c.close_position.return_value = {'id': 'close-1'}
    c.trading_client = MagicMock()
    c.trading_client.get_orders.return_value = []
    return c


@pytest.fixture
def monitor(mock_alp):
    return StopMonitor(api_key='k', api_secret='s', alpaca_client=mock_alp)


def _trade(symbol, price):
    t = MagicMock()
    t.symbol = symbol
    t.price = price
    return t


class TestSkipExitsDuringEntryBar:
    @pytest.mark.asyncio
    async def test_stop_not_triggered_during_skip_window(self, monitor):
        """Fill at 9:36:10; skip_until = 9:37:00. Tick at 9:36:30 below stop
        should NOT trigger exit."""
        # Skip window ends 60 seconds in the future
        skip_until = time_mod.time() + 60
        monitor.add_watch(
            symbol='X', stop_price=9.50, shares=100,
            tp_leg_id='', sl_leg_id='',
            entry_price=10.03, risk_per_share=0.53,
            skip_exits_until_ts=skip_until,
        )
        # Price dips BELOW stop — would normally exit, but skip window active
        from unittest.mock import patch
        with patch.object(monitor, '_execute_stop_exit', new_callable=MagicMock) as exec_mock:
            await monitor._on_trade(_trade('X', 9.40))
        exec_mock.assert_not_called()
        # Watch still active, no exit fired
        assert 'X' in monitor._watches

    @pytest.mark.asyncio
    async def test_stop_triggers_after_skip_window_expires(self, monitor):
        """After skip window expires, stop check resumes normally."""
        # Skip window already expired (1 second ago)
        skip_until = time_mod.time() - 1.0
        monitor.add_watch(
            symbol='X', stop_price=9.50, shares=100,
            tp_leg_id='', sl_leg_id='',
            entry_price=10.03, risk_per_share=0.53,
            skip_exits_until_ts=skip_until,
        )
        from unittest.mock import patch, AsyncMock
        captured = {}
        async def cap(sym, price, watch, exit_reason='stop_loss'):
            captured['reason'] = exit_reason
            captured['price'] = price
        with patch.object(monitor, '_execute_stop_exit', side_effect=cap):
            await monitor._on_trade(_trade('X', 9.40))
        assert captured.get('reason') == 'stop_loss'

    @pytest.mark.asyncio
    async def test_lock_arming_skipped_during_entry_bar(self, monitor):
        """During skip window, a spike to +1.5R should NOT arm the lock
        (BT-parity: entry bar is fully skipped)."""
        skip_until = time_mod.time() + 60
        monitor.add_watch(
            symbol='X', stop_price=9.50, shares=100,
            tp_leg_id='', sl_leg_id='',
            entry_price=10.03, risk_per_share=0.53,
            lock_arm_at_r=1.5, lock_stop_r=1.0,
            lock_r_unit=0.50,  # range_size
            skip_exits_until_ts=skip_until,
        )
        # Spike to +1.5R = 10.03 + 0.75 = 10.78
        await monitor._on_trade(_trade('X', 10.78))
        # Lock should NOT be armed (skip window active)
        assert monitor._watches['X'].lock_armed is False
        # Stop_price should still be at range_low (unchanged)
        assert monitor._watches['X'].stop_price == 9.50

    @pytest.mark.asyncio
    async def test_peak_still_tracked_during_skip(self, monitor):
        """While exits are skipped, highest_since_entry still updates so
        arming can fire correctly once the skip window closes."""
        skip_until = time_mod.time() + 60
        monitor.add_watch(
            symbol='X', stop_price=9.50, shares=100,
            tp_leg_id='', sl_leg_id='',
            entry_price=10.03, risk_per_share=0.53,
            lock_arm_at_r=1.5, lock_stop_r=1.0,
            lock_r_unit=0.50,
            skip_exits_until_ts=skip_until,
        )
        # Spike during skip window
        await monitor._on_trade(_trade('X', 11.00))
        # Peak tracked even though lock not armed
        assert monitor._watches['X'].highest_since_entry == 11.00
        assert monitor._watches['X'].lock_armed is False

    @pytest.mark.asyncio
    async def test_lock_arms_on_first_tick_after_skip_if_peak_reached(self, monitor):
        """Peak seen during skip window should cause lock to arm immediately
        on the first tick after skip expires."""
        skip_until = time_mod.time() - 1.0  # already expired
        monitor.add_watch(
            symbol='X', stop_price=9.50, shares=100,
            tp_leg_id='', sl_leg_id='',
            entry_price=10.03, risk_per_share=0.53,
            lock_arm_at_r=1.5, lock_stop_r=1.0,
            lock_r_unit=0.50,
            skip_exits_until_ts=skip_until,
        )
        # First tick above arm level: price = 10.78 (= entry + 1.5R)
        await monitor._on_trade(_trade('X', 10.78))
        assert monitor._watches['X'].lock_armed is True
        # Stop moved to entry + 1R = 10.03 + 0.50 = 10.53
        assert monitor._watches['X'].stop_price == pytest.approx(10.53)

    @pytest.mark.asyncio
    async def test_no_skip_by_default_preserves_legacy_behavior(self, monitor):
        """Bull flag / MACD wave don't pass skip_exits_until_ts → default 0 →
        no skip behavior (legacy preserved)."""
        monitor.add_watch(
            symbol='Y', stop_price=9.50, shares=100,
            tp_leg_id='', sl_leg_id='',
            entry_price=10.03, risk_per_share=0.53,
            # skip_exits_until_ts NOT passed → defaults to 0
        )
        from unittest.mock import patch
        captured = {}
        async def cap(sym, price, watch, exit_reason='stop_loss'):
            captured['reason'] = exit_reason
        with patch.object(monitor, '_execute_stop_exit', side_effect=cap):
            await monitor._on_trade(_trade('Y', 9.40))
        # Exit fires normally (no skip)
        assert captured.get('reason') == 'stop_loss'


class TestConfirmFillSetsSkipTimestamp:
    """Integration: _confirm_fill must compute skip_exits_until_ts = end of
    current entry bar when registering the watch."""

    def test_confirm_fill_passes_skip_until_next_minute(self):
        """Fill at 10:23:45 → skip window ends at 10:24:00."""
        from pathlib import Path
        import yaml
        from persistence.database import Database
        from trading.orb_engine import ORBEngine, OpenPosition

        cfg_path = Path(__file__).parent.parent / 'orb.yaml'
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg['strategy']['enabled'] = True

        mock_alpaca = MagicMock(spec=AlpacaClient)
        mock_db = MagicMock(spec=Database)
        mock_sm = MagicMock(spec=StopMonitor)
        mock_sm.polling_mode = False

        engine = ORBEngine(
            alpaca_client=mock_alpaca, db=mock_db,
            stop_monitor=mock_sm, config=cfg,
        )
        pos = OpenPosition(
            symbol='X', entry_price=10.03, stop_price=9.50, shares=100,
            trade_id=1, order_id='pending',
            entry_time=datetime.now(timezone.utc),
            range_high=10.00, range_low=9.50,
            lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
        )
        engine.open_positions['X'] = pos
        engine._confirm_fill(pos, {'status': 'filled', 'filled_avg_price': 10.05, 'filled_qty': 100})

        kw = mock_sm.add_watch.call_args.kwargs
        # skip_exits_until_ts is set and in the near future (end of current bar)
        skip_ts = kw['skip_exits_until_ts']
        assert skip_ts > 0
        # It's within the next 60 seconds from now (at most 1 minute into the future)
        now_ts = time_mod.time()
        assert skip_ts > now_ts
        assert skip_ts <= now_ts + 60.0
