"""Tests for the CR fixes to ORBEngine.

Covers:
  * Idempotent build_universe (state preservation across ticks)
  * Bar subscription on universe add
  * _get_feature_context fetches from DB/alpaca with caching
  * _process_pending_fills transitions pending -> filled + add_watch
  * _process_pending_fills handles canceled/rejected terminal statuses
  * _cancel_stale_pending_orders fires at 10:35 ET (60 min post-range)
  * check_entries uses fetched context when feature_providers None
  * zoneinfo DST-correct force_close_time
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml
from pathlib import Path

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import (
    ORBEngine, OpenPosition, RangeData, CandidateState,
)
from trading.stop_monitor import StopMonitor


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def orb_cfg():
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    return cfg


@pytest.fixture
def mock_alpaca():
    c = MagicMock(spec=AlpacaClient)
    c.get_open_positions.return_value = []
    c.get_account_info.return_value = {'buying_power': 100_000.0}
    c.get_latest_quote.return_value = {'bid_price': 9.98, 'ask_price': 10.00}
    c.submit_stop_bracket_order.return_value = {'id': 'o-1', 'status': 'accepted'}
    c.cancel_order.return_value = True
    c.close_position.return_value = {'id': 'c-1'}
    # _get_feature_context was changed to use get_daily_bars_range (list of
    # OHLCV dicts) instead of get_daily_bars (summary dict) — keep both
    # stubbed so the full call chain works. DB-cache path is tested separately.
    c.get_daily_bars_range.return_value = {
        'TSLA': [
            {'date': f'2026-03-{(i%28)+1:02d}',
             'open': 9 + i*0.01, 'high': 10 + i*0.01, 'low': 8.5 + i*0.01,
             'close': 9.5 + i*0.01, 'volume': 1_000_000}
            for i in range(25)
        ]
    }
    c.get_daily_bars.return_value = {}  # legacy, no longer called by engine
    return c


@pytest.fixture
def mock_sm():
    sm = MagicMock(spec=StopMonitor)
    sm.polling_mode = False
    sm.drain_exit_events.return_value = []
    return sm


@pytest.fixture
def mock_db():
    db = MagicMock(spec=Database)
    db.save_trade.return_value = 1
    db.get_open_trades.return_value = []
    db.update_trade.return_value = True
    return db


@pytest.fixture
def engine(orb_cfg, mock_alpaca, mock_db, mock_sm):
    return ORBEngine(
        alpaca_client=mock_alpaca, db=mock_db,
        stop_monitor=mock_sm, config=orb_cfg,
    )


# =========================================================================
# Idempotent build_universe
# =========================================================================

class TestIdempotentUniverse:
    def test_preserves_candidate_state_on_repeat_call(self, engine):
        engine.build_universe(source_loader=lambda: ['A', 'B'])
        # Populate range data on 'A'
        engine.candidates['A'].range_data = RangeData(
            symbol='A', range_high=10, range_low=9.5, range_volume=1000,
            range_avg_bar_range_pct=1.0, range_close=9.9,
            range_start_ts=pd.Timestamp.utcnow(),
        )
        # Re-call with same universe
        engine.build_universe(source_loader=lambda: ['A', 'B'])
        # State preserved
        assert engine.candidates['A'].range_data is not None
        assert engine.candidates['A'].range_data.range_high == 10

    def test_adds_new_symbols_preserves_old(self, engine):
        engine.build_universe(source_loader=lambda: ['A'])
        engine.candidates['A'].range_data = RangeData(
            symbol='A', range_high=10, range_low=9.5, range_volume=1000,
            range_avg_bar_range_pct=1.0, range_close=9.9,
            range_start_ts=pd.Timestamp.utcnow(),
        )
        engine.build_universe(source_loader=lambda: ['A', 'B', 'C'])
        assert 'A' in engine.universe
        assert 'B' in engine.universe
        assert 'C' in engine.universe
        assert engine.candidates['A'].range_data is not None  # preserved
        assert engine.candidates['B'].range_data is None      # fresh
        assert engine.candidates['C'].range_data is None

    def test_subscribes_bars_for_new_candidates(self, engine, mock_sm):
        engine.build_universe(source_loader=lambda: ['A', 'B'])
        # subscribe_bars called for each new symbol
        assert mock_sm.subscribe_bars.call_count == 2
        mock_sm.subscribe_bars.assert_any_call('A')
        mock_sm.subscribe_bars.assert_any_call('B')
        # Re-call: no new subscriptions (symbols unchanged)
        engine.build_universe(source_loader=lambda: ['A', 'B'])
        assert mock_sm.subscribe_bars.call_count == 2  # unchanged


# =========================================================================
# Feature context auto-fetch
# =========================================================================

class TestFeatureContext:
    def test_fetches_from_alpaca_when_available(self, engine, mock_alpaca):
        ctx = engine._get_feature_context('TSLA')
        assert 'prev_day_bar' in ctx
        assert ctx['prev_day_bar']['close'] > 0
        assert 'daily_stats_20d' in ctx
        assert ctx['daily_stats_20d']['high_20d'] > 0

    def test_caches_per_symbol(self, engine, mock_alpaca):
        engine._get_feature_context('TSLA')
        engine._get_feature_context('TSLA')  # cached
        engine._get_feature_context('TSLA')
        # Only one Alpaca call per symbol (DB cache is checked first and
        # returns empty in this test fixture → falls through to alpaca)
        assert mock_alpaca.get_daily_bars_range.call_count == 1

    def test_returns_empty_on_fetch_failure(self, engine, mock_alpaca):
        mock_alpaca.get_daily_bars_range.side_effect = RuntimeError("api down")
        ctx = engine._get_feature_context('BROKEN')
        assert ctx == {}

    def test_reset_daily_clears_cache(self, engine, mock_alpaca):
        engine._get_feature_context('TSLA')
        engine.reset_daily()
        engine._get_feature_context('TSLA')
        # Fresh fetch — cache was cleared
        assert mock_alpaca.get_daily_bars_range.call_count == 2


# =========================================================================
# _process_pending_fills (the big one)
# =========================================================================

def _make_order_status(status: str, fill_price: float = None, filled_qty: int = None):
    return {
        'status': status,
        'filled_avg_price': fill_price,
        'filled_qty': filled_qty,
    }


class TestProcessPendingFills:
    def _seed_pending(self, engine, symbol='TSLA'):
        """Create a pending OpenPosition + CandidateState."""
        engine.build_universe(source_loader=lambda: [symbol])
        cand = engine.candidates[symbol]
        cand.plan_submitted = True
        cand.order_id = 'o-pending'
        cand.order_submitted_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        pos = OpenPosition(
            symbol=symbol, entry_price=10.03, stop_price=9.95, shares=100,
            trade_id=1, order_id='o-pending',
            entry_time=datetime.now(timezone.utc) - timedelta(seconds=10),
            range_high=10.00, range_low=9.95, lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
        )
        engine.open_positions[symbol] = pos
        return pos

    def test_pending_new_stays_pending(self, engine, mock_alpaca, mock_sm):
        pos = self._seed_pending(engine)
        mock_alpaca.get_order.return_value = _make_order_status('pending_new')
        engine._process_pending_fills()
        # Still pending → order_id unchanged, no add_watch
        assert engine.open_positions['TSLA'].order_id == 'o-pending'
        mock_sm.add_watch.assert_not_called()

    def test_filled_transitions_to_filled_state(self, engine, mock_alpaca, mock_sm):
        pos = self._seed_pending(engine)
        mock_alpaca.get_order.return_value = _make_order_status(
            'filled', fill_price=10.05, filled_qty=100)
        engine._process_pending_fills()
        # Position still exists, order_id cleared
        assert 'TSLA' in engine.open_positions
        assert engine.open_positions['TSLA'].order_id == ''
        assert engine.open_positions['TSLA'].entry_price == 10.05
        # StopMonitor watch added with lock params
        mock_sm.add_watch.assert_called_once()
        kwargs = mock_sm.add_watch.call_args.kwargs
        assert kwargs['symbol'] == 'TSLA'
        assert kwargs['strategy'] == 'orb'
        assert kwargs['lock_arm_at_r'] == 1.5
        assert kwargs['lock_stop_r'] == 1.0
        assert kwargs['entry_price'] == 10.05

    def test_canceled_removes_from_tracking(self, engine, mock_alpaca, mock_sm):
        pos = self._seed_pending(engine)
        mock_alpaca.get_order.return_value = _make_order_status('canceled')
        engine._process_pending_fills()
        assert 'TSLA' not in engine.open_positions
        assert engine.candidates['TSLA'].rejected_reason == 'order_canceled'
        mock_sm.add_watch.assert_not_called()

    def test_rejected_removes_from_tracking(self, engine, mock_alpaca, mock_sm):
        pos = self._seed_pending(engine)
        mock_alpaca.get_order.return_value = _make_order_status('rejected')
        engine._process_pending_fills()
        assert 'TSLA' not in engine.open_positions
        assert engine.candidates['TSLA'].rejected_reason == 'order_rejected'

    def test_fill_updates_db(self, engine, mock_alpaca, mock_db):
        pos = self._seed_pending(engine)
        mock_alpaca.get_order.return_value = _make_order_status(
            'filled', fill_price=10.05, filled_qty=100)
        engine._process_pending_fills()
        # DB update called with fill info
        mock_db.update_trade.assert_called()
        # Find the 'filled' update (there may also be log updates)
        filled_call = None
        for call in mock_db.update_trade.call_args_list:
            if call.args[1].get('order_status') == 'filled':
                filled_call = call
                break
        assert filled_call is not None
        assert filled_call.args[1]['fill_price'] == 10.05

    def test_filled_positions_not_reprocessed(self, engine, mock_alpaca):
        pos = self._seed_pending(engine)
        pos.order_id = ''  # already filled
        engine._process_pending_fills()
        # get_order should NOT be called for already-filled positions
        mock_alpaca.get_order.assert_not_called()

    def test_partial_fill_keeps_polling_no_cancel_no_confirm(
        self, engine, mock_alpaca, mock_sm
    ):
        """FABC 2026-06-09 rewrite: partial fills no longer trigger early
        _confirm_fill. The pre-fix behavior (cancel + accept first partial)
        silently under-reported multi-fill orders — FABC recorded 1438 of
        3188 shares because the first partial fired before the broker
        reached terminal 'filled'. New behavior: log + keep polling. The
        APT/MLTX orphan-growth scenario is now handled by sync_positions'
        orphan-detect (the existing backstop)."""
        pos = self._seed_pending(engine)
        mock_alpaca.get_order.return_value = {
            'status': 'partially_filled',
            'filled_avg_price': 10.05,
            'filled_qty': 44,
            'qty': 100,
        }
        engine._process_pending_fills()

        # Parent remainder NOT cancelled — we wait for terminal status
        for call in mock_alpaca.cancel_order.call_args_list:
            assert call.args[0] != 'o-pending', (
                "cancel_order should NOT fire on first partial — wait for "
                "terminal 'filled' or stall-timeout"
            )
        # Position still pending — _confirm_fill did NOT run
        assert engine.open_positions['TSLA'].order_id == 'o-pending'
        # first_partial_at recorded for stall-timeout accounting
        assert engine.open_positions['TSLA'].first_partial_at is not None
        # No StopMonitor watch yet
        mock_sm.add_watch.assert_not_called()

    def test_partial_then_filled_confirms_with_full_qty(
        self, engine, mock_alpaca, mock_sm
    ):
        """When the broker eventually transitions partial → filled, the
        engine confirms at the FINAL qty (not the first-seen partial).
        This is the path that fixes the FABC-style under-recording."""
        pos = self._seed_pending(engine)
        # Poll 1: partial 44/100
        mock_alpaca.get_order.return_value = {
            'status': 'partially_filled',
            'filled_avg_price': 10.05,
            'filled_qty': 44,
            'qty': 100,
        }
        engine._process_pending_fills()
        mock_sm.add_watch.assert_not_called()
        # Poll 2: terminal 'filled' at full 100
        mock_alpaca.get_order.return_value = {
            'status': 'filled',
            'filled_avg_price': 10.05,
            'filled_qty': 100,
            'qty': 100,
        }
        engine._process_pending_fills()
        # Confirmed at full qty
        assert engine.open_positions['TSLA'].shares == 100
        assert engine.open_positions['TSLA'].order_id == ''
        mock_sm.add_watch.assert_called_once()


# =========================================================================
# _cancel_stale_pending_orders
# =========================================================================

class TestCancelStalePending:
    def test_does_not_cancel_fresh_order(self, engine, mock_alpaca):
        engine.build_universe(source_loader=lambda: ['A'])
        engine.candidates['A'].order_id = 'o1'
        engine.candidates['A'].order_submitted_at = datetime.now(timezone.utc)
        engine.open_positions['A'] = OpenPosition(
            symbol='A', entry_price=10, stop_price=9.5, shares=100, trade_id=1,
            order_id='o1', entry_time=datetime.now(timezone.utc),
            range_high=10, range_low=9.5, lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
        )
        engine._cancel_stale_pending_orders()
        mock_alpaca.cancel_order.assert_not_called()

    def test_cancels_order_past_time_stop(self, engine, mock_alpaca, mock_db):
        engine.build_universe(source_loader=lambda: ['A'])
        old_time = datetime.now(timezone.utc) - timedelta(minutes=61)  # past 60min
        engine.candidates['A'].order_id = 'o1'
        engine.candidates['A'].order_submitted_at = old_time
        engine.open_positions['A'] = OpenPosition(
            symbol='A', entry_price=10, stop_price=9.5, shares=100, trade_id=1,
            order_id='o1', entry_time=old_time,
            range_high=10, range_low=9.5, lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
        )
        engine._cancel_stale_pending_orders()
        mock_alpaca.cancel_order.assert_called_once_with('o1')
        assert 'A' not in engine.open_positions
        assert engine.candidates['A'].rejected_reason == 'time_stop'

    def test_does_not_cancel_filled_position(self, engine, mock_alpaca):
        old_time = datetime.now(timezone.utc) - timedelta(minutes=61)
        engine.build_universe(source_loader=lambda: ['A'])
        engine.candidates['A'].order_id = None  # cleared on fill
        engine.open_positions['A'] = OpenPosition(
            symbol='A', entry_price=10.1, stop_price=9.5, shares=100, trade_id=1,
            order_id='',  # filled
            entry_time=old_time,
            range_high=10, range_low=9.5, lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
        )
        engine._cancel_stale_pending_orders()
        mock_alpaca.cancel_order.assert_not_called()


# =========================================================================
# _submit_entry contract
# =========================================================================

class TestSubmitEntry:
    def test_submit_populates_open_position_with_order_id(self, engine, mock_alpaca, mock_db):
        from trading.orb_planner import OrbTradePlan
        plan = OrbTradePlan(
            symbol='TSLA', range_high=10, range_low=9.5, range_size=0.5,
            entry_price=10.03, stop_price=9.5, shares=100, position_dollars=1003,
            lock_arm_at_r=1.5, lock_stop_r=1.0, risk_per_share=0.53, total_risk=53,
            composite_score=0.5, quintile='Q4', adaptive_mult=0.95,
        )
        mock_alpaca.submit_stop_bracket_order.return_value = {'id': 'new-order', 'status': 'accepted'}
        order_id = engine._submit_entry(plan)
        assert order_id == 'new-order'
        # OpenPosition created with pending order_id
        assert 'TSLA' in engine.open_positions
        assert engine.open_positions['TSLA'].order_id == 'new-order'

    def test_submit_uses_correct_alpaca_api(self, engine, mock_alpaca):
        from trading.orb_planner import OrbTradePlan
        plan = OrbTradePlan(
            symbol='X', range_high=10, range_low=9.5, range_size=0.5,
            entry_price=10.03, stop_price=9.5, shares=100, position_dollars=1003,
            lock_arm_at_r=1.5, lock_stop_r=1.0, risk_per_share=0.53, total_risk=53,
            composite_score=0.5, quintile='Q4', adaptive_mult=0.95,
        )
        engine._submit_entry(plan)
        call = mock_alpaca.submit_stop_bracket_order.call_args
        kw = call.kwargs
        assert kw['side'] == 'buy'
        # BT parity: stop triggers at range_high (NOT entry_price × 1.003)
        assert kw['stop_price'] == 10.00
        # limit_price = entry_price (range_high × 1.003) — single slippage buffer
        assert kw['limit_price'] == 10.03
        assert kw['limit_price'] > kw['stop_price']
        # tp_price is NOT None (Alpaca bracket requires it) — set to unreachable safety-net
        assert kw['tp_price'] > kw['stop_price'] * 2
        # sl_price is 10% below entry (safety-net)
        assert kw['sl_price'] < kw['stop_price']

    def test_sync_positions_preserves_submit_time_across_restart(
        self, engine, mock_alpaca, mock_db, mock_sm
    ):
        """Restart-safety: cancel-clock must survive restart.

        Pre-fix bug: sync_positions' pending-rehydration set
        cand.order_submitted_at = datetime.now() (at restart time), which
        restarted the 60min time-stop clock. An order submitted at 09:57
        that should have canceled at 10:57 instead survived until 11:20+
        after a 10:20 restart — ~23 min extra market exposure.

        Post-fix: rehydration reads order_submitted_at from DB so the
        original submit clock is preserved.
        """
        from datetime import datetime, timezone
        original_submit = datetime(2026, 4, 20, 13, 57, 18, tzinfo=timezone.utc)

        # sync_positions uses db.get_open_trades(date, strategy='orb')
        mock_db.get_open_trades = MagicMock(return_value=[{
            'id': 42, 'symbol': 'TSLA', 'strategy': 'orb',
            'order_id': 'ord-persisted', 'order_status': 'pending_new',
            'entry_price': 10.03, 'stop_loss_price': 9.50, 'shares': 100,
            'fill_price': None, 'exit_price': None,
            'pattern_data': '{"range_high": 10.0, "range_low": 9.5, '
                            '"lock_arm_at_r": 1.5, "lock_stop_r": 1.0, '
                            '"composite_score": 0.5, "quintile": "Q4"}',
            'order_submitted_at': original_submit,
            'bar_close_price': 10.0,
            'entry_quote_ask': 10.00,
            'created_at': original_submit,
        }])
        # Alpaca confirms the order is still live via trading_client
        fake_order = MagicMock()
        fake_order.id = 'ord-persisted'
        fake_order.symbol = 'TSLA'
        mock_alpaca.trading_client = MagicMock()
        mock_alpaca.trading_client.get_all_positions.return_value = []
        mock_alpaca.trading_client.get_orders.return_value = [fake_order]

        engine.sync_positions()

        # Position rehydrated with ORIGINAL submit time, not now()
        assert 'TSLA' in engine.open_positions
        pos = engine.open_positions['TSLA']
        assert pos.entry_time == original_submit
        assert pos.order_submitted_at == original_submit
        assert pos.bar_close_price == 10.0
        assert pos.entry_quote_ask == 10.00
        # Candidate's cancel-clock also anchored to original submit
        cand = engine.candidates.get('TSLA')
        if cand is not None:
            assert cand.order_submitted_at == original_submit

    def test_submit_persists_slippage_fields_to_db(self, engine, mock_alpaca, mock_db):
        """Restart-safety: _save_pending_trade must write order_submitted_at,
        bar_close_price, entry_quote_bid/ask/spread so a restart can rehydrate
        the OpenPosition attribution correctly."""
        from trading.orb_planner import OrbTradePlan
        plan = OrbTradePlan(
            symbol='R', range_high=10, range_low=9.5, range_size=0.5,
            entry_price=10.03, stop_price=9.5, shares=100, position_dollars=1003,
            lock_arm_at_r=1.5, lock_stop_r=1.0, risk_per_share=0.53, total_risk=53,
            composite_score=0.5, quintile='Q4', adaptive_mult=0.95,
        )
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 9.98, 'ask_price': 10.00, 'bid_size': 500, 'ask_size': 700,
        }
        engine._submit_entry(plan)
        record = mock_db.save_trade.call_args[0][0]
        # All attribution fields present at submit time
        assert record['order_submitted_at'] is not None
        assert record['bar_close_price'] == 10.0  # = range_high (BT reference)
        assert record['entry_quote_bid'] == 9.98
        assert record['entry_quote_ask'] == 10.00
        assert record['entry_quote_bid_size'] == 500
        assert record['entry_quote_ask_size'] == 700
        assert record['entry_quote_spread'] == pytest.approx(0.02, abs=1e-9)


# =========================================================================
# zoneinfo DST
# =========================================================================

class TestDailyCapsRestartSafe:
    """Regression for 2026-04-20 bugs: after mid-day restart, ORB re-entered
    BMNZ/SKYQ after stop-outs and entered brand-new QBTZ/BATL past the intended
    cutoff. Root cause: in-memory daily_n_placed + CandidateState.plan_submitted
    reset on restart → BT's 'top-K picked once per day' invariant violated.

    Fix: check_entries must use a DB-backed set of symbols-entered-today to:
      * block >max_concurrent total entries per day
      * block re-entry of any symbol already traded today
    Both gates survive restart.
    """
    def test_daily_total_cap_enforced_from_db(self, engine, mock_alpaca, mock_db, mock_sm):
        """DB has 4 ORB trades today → check_entries blocks ALL new entries."""
        from trading.orb_engine import CandidateState, RangeData
        import pandas as pd
        mock_db.get_trades_by_date = MagicMock(return_value=[
            {'symbol': 'A', 'strategy': 'orb'},
            {'symbol': 'B', 'strategy': 'orb'},
            {'symbol': 'C', 'strategy': 'orb'},
            {'symbol': 'D', 'strategy': 'orb'},
        ])
        # Universe has a fresh candidate E with full range_data + composite
        engine.build_universe(source_loader=lambda: ['E'])
        engine.candidates['E'].range_data = RangeData(
            symbol='E', range_high=10, range_low=9.5, range_volume=1000,
            range_avg_bar_range_pct=1.0, range_close=9.9,
            range_start_ts=pd.Timestamp.utcnow(), range_open=9.6,
        )
        engine.candidates['E'].plan_submitted = False  # restart simulation
        engine.enabled = True
        submitted = engine.check_entries()
        assert submitted == []
        # Alpaca submit never called — daily cap blocked before reaching plan
        mock_alpaca.submit_stop_bracket_order.assert_not_called()

    def test_per_symbol_dedup_from_db_prevents_reentry(self, engine, mock_alpaca, mock_db, mock_sm):
        """BMNZ already in DB (stopped out) → check_entries skips it even with
        clean in-memory CandidateState. Regression for the restart bug."""
        from trading.orb_engine import CandidateState, RangeData
        from unittest.mock import patch
        import pandas as pd
        # Only 1 trade today → daily_total (1) < max_concurrent (4), so cap is fine
        mock_db.get_trades_by_date = MagicMock(return_value=[
            {'symbol': 'BMNZ', 'strategy': 'orb'},
        ])
        engine.build_universe(source_loader=lambda: ['BMNZ', 'FRESH'])
        # Both symbols have full ranges + fresh in-memory state (restart-like)
        for sym in ['BMNZ', 'FRESH']:
            engine.candidates[sym].range_data = RangeData(
                symbol=sym, range_high=10, range_low=9.5, range_volume=1000,
                range_avg_bar_range_pct=1.0, range_close=9.9,
                range_start_ts=pd.Timestamp.utcnow(), range_open=9.6,
            )
            engine.candidates[sym].plan_submitted = False
        engine.enabled = True

        # Provide minimal feature ctx so composite won't be None
        providers = {
            'BMNZ': {'prev_day_bar': {'close': 9.0, 'volume': 500_000},
                     'daily_stats_20d': {'high_20d': 10, 'volume_20d': 500_000}},
            'FRESH': {'prev_day_bar': {'close': 9.0, 'volume': 500_000},
                      'daily_stats_20d': {'high_20d': 10, 'volume_20d': 500_000}},
        }
        # Bypass the time-of-day cutoff — test runs at arbitrary wall clock
        with patch.object(engine, '_past_last_entry_time', return_value=False):
            engine.check_entries(symbols=['BMNZ', 'FRESH'], feature_providers=providers)
        # BMNZ must be skipped with the new reason
        assert engine.candidates['BMNZ'].rejected_reason == 'already_entered_today'

    def test_cap_does_not_block_initial_picks(self, engine, mock_db):
        """Sanity: when DB is empty (fresh day), no cap is triggered."""
        mock_db.get_trades_by_date = MagicMock(return_value=[])
        assert engine._symbols_entered_today_db() == set()

    def test_helper_filters_by_strategy(self, engine, mock_db):
        """Bull flag + MACD wave trades must not count against ORB's daily cap."""
        mock_db.get_trades_by_date = MagicMock(return_value=[
            {'symbol': 'A', 'strategy': 'bull_flag'},
            {'symbol': 'B', 'strategy': 'macd_wave'},
            {'symbol': 'C', 'strategy': 'orb'},
        ])
        assert engine._symbols_entered_today_db() == {'C'}


class TestLastEntryCutoff:
    """Post-2026-04-20 fix: hard time-of-day cutoff on NEW entry submissions.
    BT picks top-K once at 9:35 ET; live allows a short window but must block
    late-afternoon entries (QBTZ 12:30 ET + BATL 1:11 PM ET today's bug)."""
    def _et_noon_utc(self):
        """12:00 ET in UTC. Today (2026-04-20) is EDT (UTC-4) → 16:00 UTC."""
        from datetime import datetime, timezone as _tz
        return datetime(2026, 4, 20, 16, 0, 0, tzinfo=_tz.utc)

    def _et_915_utc(self):
        """09:15 ET pre-market in UTC (EDT)."""
        from datetime import datetime, timezone as _tz
        return datetime(2026, 4, 20, 13, 15, 0, tzinfo=_tz.utc)

    def test_past_cutoff_true_after_1000_et(self, engine):
        """Default cutoff 10:00 ET → noon ET must be past."""
        assert engine._past_last_entry_time(self._et_noon_utc()) is True

    def test_past_cutoff_false_before_1000_et(self, engine):
        """Pre-market 9:15 ET must NOT trip the cutoff."""
        assert engine._past_last_entry_time(self._et_915_utc()) is False

    def test_cutoff_reads_from_config(self, orb_cfg, mock_alpaca, mock_db, mock_sm):
        """yaml::entry.last_entry_submit_time_et configures the cutoff."""
        from trading.orb_engine import ORBEngine
        cfg = dict(orb_cfg)
        cfg['entry'] = dict(cfg.get('entry', {}))
        cfg['entry']['last_entry_submit_time_et'] = "09:45"
        e = ORBEngine(alpaca_client=mock_alpaca, db=mock_db,
                      stop_monitor=mock_sm, config=cfg)
        assert e.last_entry_hour_et == 9
        assert e.last_entry_minute_et == 45

    def test_check_entries_returns_empty_past_cutoff(self, engine, mock_alpaca, mock_db, mock_sm):
        """Past cutoff, check_entries returns [] without touching Alpaca.
        Regression for today's QBTZ/BATL late-afternoon bug."""
        from unittest.mock import patch
        # Seed state so there WOULD be eligible candidates if cutoff didn't block
        from trading.orb_engine import RangeData
        import pandas as pd
        engine.build_universe(source_loader=lambda: ['LATE'])
        engine.candidates['LATE'].range_data = RangeData(
            symbol='LATE', range_high=10, range_low=9.5, range_volume=1000,
            range_avg_bar_range_pct=1.0, range_close=9.9,
            range_start_ts=pd.Timestamp.utcnow(), range_open=9.6,
        )
        engine.enabled = True

        # Mock current time to 12:00 ET (past the 10:00 cutoff)
        with patch.object(engine, '_past_last_entry_time', return_value=True):
            result = engine.check_entries()
        assert result == []
        mock_alpaca.submit_stop_bracket_order.assert_not_called()

    def test_check_entries_proceeds_before_cutoff(self, engine, mock_alpaca, mock_db, mock_sm):
        """Before cutoff, check_entries proceeds to normal evaluation (no early return)."""
        from unittest.mock import patch
        # Empty universe → returns [] BUT we want to verify the cutoff guard
        # didn't early-return. Patch _past_last_entry_time to False; verify
        # the function didn't return at the cutoff guard (would have needed
        # no DB query) — instead it proceeds, queries DB, then returns [] at
        # the empty-universe path.
        mock_db.get_trades_by_date = MagicMock(return_value=[])
        with patch.object(engine, '_past_last_entry_time', return_value=False):
            engine.enabled = True
            engine.check_entries()
        # _symbols_entered_today_db queries get_trades_by_date → proves we
        # got past the cutoff guard
        mock_db.get_trades_by_date.assert_called()

    def test_cutoff_logged_once(self, engine, caplog):
        """The 'past cutoff' log fires at most once per day (not every tick)."""
        from unittest.mock import patch
        with patch.object(engine, '_past_last_entry_time', return_value=True):
            engine.enabled = True
            with caplog.at_level('INFO'):
                engine.check_entries()
                engine.check_entries()
                engine.check_entries()
        hits = [r for r in caplog.records
                if 'past last_entry_submit_time' in r.getMessage()]
        assert len(hits) == 1


class TestFillRateTelemetry:
    """Fill-rate counters must track placed/filled/canceled per day.
    Validates PROD against BT-expected ~73% fill rate target."""

    def test_counters_init_zero(self, engine):
        assert engine.daily_n_placed == 0
        assert engine.daily_n_filled == 0
        assert engine.daily_n_time_stop_canceled == 0
        assert engine.daily_summary_sent is False

    def test_submit_entry_increments_placed(self, engine, mock_alpaca):
        from trading.orb_planner import OrbTradePlan
        plan = OrbTradePlan(
            symbol='X', range_high=10, range_low=9.5, range_size=0.5,
            entry_price=10.03, stop_price=9.5, shares=100, position_dollars=1003,
            lock_arm_at_r=1.5, lock_stop_r=1.0, risk_per_share=0.53, total_risk=53,
            composite_score=0.5, quintile='Q4', adaptive_mult=0.95,
        )
        mock_alpaca.submit_stop_bracket_order.return_value = {'id': 'o-1'}
        engine._submit_entry(plan)
        assert engine.daily_n_placed == 1
        assert engine.daily_n_filled == 0

    def test_fill_increments_filled(self, engine, mock_alpaca):
        from trading.orb_engine import OpenPosition
        from datetime import datetime, timezone
        pos = OpenPosition(
            symbol='Y', entry_price=10.03, stop_price=9.5, shares=100,
            trade_id=1, order_id='pending',
            entry_time=datetime.now(timezone.utc),
            range_high=10, range_low=9.5, lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
        )
        engine.open_positions['Y'] = pos
        engine._confirm_fill(pos, {'status': 'filled', 'filled_avg_price': 10.05, 'filled_qty': 100})
        assert engine.daily_n_filled == 1

    def test_reset_daily_zeros_counters(self, engine):
        engine.daily_n_placed = 4
        engine.daily_n_filled = 3
        engine.daily_n_time_stop_canceled = 1
        engine.daily_summary_sent = True
        engine.reset_daily()
        assert engine.daily_n_placed == 0
        assert engine.daily_n_filled == 0
        assert engine.daily_n_time_stop_canceled == 0
        assert engine.daily_summary_sent is False

    def test_send_daily_report_logs_fill_rate(self, engine, caplog):
        import logging
        engine.daily_n_placed = 4
        engine.daily_n_filled = 3
        engine.daily_n_time_stop_canceled = 1
        engine.daily_pnl = 1200.0
        with caplog.at_level(logging.INFO, logger='trading.orb_engine'):
            engine.send_daily_report()
        # Log message contains the key metrics
        msgs = [r.message for r in caplog.records]
        summary = " ".join(msgs)
        assert 'placed=4' in summary
        assert 'filled=3' in summary
        assert 'fill_rate=75%' in summary

    def test_force_close_triggers_daily_report_once(self, engine, mock_alpaca):
        engine.daily_n_placed = 2
        engine.daily_n_filled = 1
        engine.force_close_all()
        assert engine.daily_summary_sent is True
        # Second call doesn't re-trigger (flag stays True)
        engine.force_close_all()
        # No duplicate telegram (if we had a counter we'd check; suffice: flag check)


class TestDSTHandling:
    def test_dst_summer_edt(self, engine):
        # June — EDT (UTC-4). 19:45 UTC = 15:45 EDT
        t = datetime(2026, 6, 15, 19, 45, tzinfo=timezone.utc)
        assert engine.is_force_close_time(now_utc=t) is True

    def test_dst_winter_est(self, engine):
        # January — EST (UTC-5). 20:45 UTC = 15:45 EST
        t = datetime(2026, 1, 15, 20, 45, tzinfo=timezone.utc)
        assert engine.is_force_close_time(now_utc=t) is True

    def test_dst_before_winter_threshold(self, engine):
        # January 20:30 UTC = 15:30 EST — before close
        t = datetime(2026, 1, 15, 20, 30, tzinfo=timezone.utc)
        assert engine.is_force_close_time(now_utc=t) is False

    def test_dst_transition_week_spring(self, engine):
        """Second Sunday of March 2026 = March 8. Before that = EST."""
        # March 5 (EST) 20:45 UTC = 15:45 EST → should be True
        t = datetime(2026, 3, 5, 20, 45, tzinfo=timezone.utc)
        assert engine.is_force_close_time(now_utc=t) is True
        # March 5 (EST) 19:45 UTC = 14:45 EST → should be False
        t = datetime(2026, 3, 5, 19, 45, tzinfo=timezone.utc)
        assert engine.is_force_close_time(now_utc=t) is False
