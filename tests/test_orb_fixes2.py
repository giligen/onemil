"""Tests for round 2 of BT<->PROD alignment fixes.

Covers:
  - range_open plumbed through RangeData + used in feature denominators
  - gap_pct / range_size_pct / price_vs_20d_high_pct use range_open (not range_high)
  - stop_price = range_high (not entry_price); single slippage buffer
  - partially_filled handled as fill
  - shutdown_requested short-circuits check_entries
  - daily_n_placed hard cap
  - lock_r_unit explicit (BT parity: range_size as 1R)
  - sync_positions rehydrates PENDING orders
  - build_orb_universe_from_snapshots applies BT criteria
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, PropertyMock
from pathlib import Path

import pandas as pd
import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import (
    ORBEngine, OpenPosition, RangeData, CandidateState,
)
from trading.stop_monitor import StopMonitor


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
    c.trading_client = MagicMock()
    c.trading_client.get_orders.return_value = []
    c.get_daily_bars.return_value = {
        'TSLA': [{'open': 9+i*0.01, 'high': 10, 'low': 8.5, 'close': 9.5, 'volume': 1_000_000} for i in range(25)]
    }
    return c


@pytest.fixture
def mock_db():
    db = MagicMock(spec=Database)
    db.save_trade.return_value = 1
    db.get_open_trades.return_value = []
    db.update_trade.return_value = True
    return db


@pytest.fixture
def mock_sm():
    sm = MagicMock(spec=StopMonitor)
    sm.polling_mode = False
    sm.drain_exit_events.return_value = []
    return sm


@pytest.fixture
def engine(orb_cfg, mock_alpaca, mock_db, mock_sm):
    return ORBEngine(alpaca_client=mock_alpaca, db=mock_db,
                     stop_monitor=mock_sm, config=orb_cfg)


def _bars(rows):
    return pd.DataFrame([
        {'timestamp': pd.Timestamp(f'2026-04-20 {h:02d}:{m:02d}:00', tz='UTC'),
         'open': o, 'high': hi, 'low': lo, 'close': c, 'volume': v}
        for (h, m, o, hi, lo, c, v) in rows
    ])


# =========================================================================
# Fix #1 + #2: range_open and feature denominators
# =========================================================================

class TestRangeOpenAndFeatureDenominators:
    def test_range_open_captured_from_first_bar(self, engine):
        engine.build_universe(source_loader=lambda: ['TSLA'])
        bars = _bars([
            (13, 30, 10.00, 10.15, 9.95, 10.10, 1000),  # open of 9:30 bar = $10.00
            (13, 31, 10.10, 10.25, 10.05, 10.20, 1000),
            (13, 32, 10.20, 10.35, 10.15, 10.30, 1000),
            (13, 33, 10.30, 10.45, 10.25, 10.40, 1000),
            (13, 34, 10.40, 10.55, 10.35, 10.50, 1000),
        ])
        engine._ingest_bars('TSLA', bars)
        rd = engine.candidates['TSLA'].range_data
        assert rd is not None
        assert rd.range_open == 10.00
        assert rd.range_high == 10.55
        assert rd.range_low == 9.95

    def test_range_size_pct_uses_range_open_not_range_high(self, engine):
        """BT parity: (range_high - range_low) / range_open × 100."""
        engine.build_universe(source_loader=lambda: ['TSLA'])
        engine.candidates['TSLA'].range_data = RangeData(
            symbol='TSLA', range_high=12.00, range_low=10.00, range_volume=1000,
            range_avg_bar_range_pct=5.0, range_close=11.50,
            range_start_ts=pd.Timestamp.utcnow(),
            range_open=10.00,  # ← 9:30 bar open
        )
        feats = engine._compute_features(engine.candidates['TSLA'])
        # BT: (12 - 10) / 10 × 100 = 20.0%
        # PROD pre-fix was (12-10)/12 × 100 = 16.67% — WRONG
        assert feats['range_size_pct'] == pytest.approx(20.0, abs=0.001)

    def test_gap_pct_uses_range_open_not_range_high(self, engine):
        """BT parity: (range_open - prev_close) / prev_close × 100."""
        engine.build_universe(source_loader=lambda: ['X'])
        engine.candidates['X'].range_data = RangeData(
            symbol='X', range_high=12.00, range_low=10.00, range_volume=1000,
            range_avg_bar_range_pct=5.0, range_close=11.50,
            range_start_ts=pd.Timestamp.utcnow(),
            range_open=10.00,
        )
        feats = engine._compute_features(
            engine.candidates['X'],
            prev_day_bar={'open': 9.5, 'high': 10.0, 'low': 9.4, 'close': 9.50, 'volume': 500_000},
        )
        # BT: (10 - 9.5) / 9.5 × 100 = 5.26%
        # PRE-FIX: (12 - 9.5) / 9.5 × 100 = 26.3% — way off
        assert feats['gap_pct'] == pytest.approx(5.263, abs=0.01)

    def test_price_vs_20d_high_uses_range_open(self, engine):
        engine.build_universe(source_loader=lambda: ['X'])
        engine.candidates['X'].range_data = RangeData(
            symbol='X', range_high=12.00, range_low=10.00, range_volume=1000,
            range_avg_bar_range_pct=5.0, range_close=11.50,
            range_start_ts=pd.Timestamp.utcnow(),
            range_open=10.00,
        )
        feats = engine._compute_features(
            engine.candidates['X'],
            daily_stats_20d={'high_20d': 15.0, 'volume_20d': 1_000_000},
        )
        # BT: (10 - 15) / 15 × 100 = -33.33%
        # PRE-FIX: (12 - 15) / 15 × 100 = -20.0%
        assert feats['price_vs_20d_high_pct'] == pytest.approx(-33.333, abs=0.01)

    def test_range_open_fallback_to_range_high_if_missing(self, engine):
        """Defensive: old RangeData without range_open still works (degenerate)."""
        engine.build_universe(source_loader=lambda: ['X'])
        engine.candidates['X'].range_data = RangeData(
            symbol='X', range_high=12.00, range_low=10.00, range_volume=1000,
            range_avg_bar_range_pct=5.0, range_close=11.50,
            range_start_ts=pd.Timestamp.utcnow(),
            # range_open not set → 0.0 → fallback to range_high
        )
        feats = engine._compute_features(engine.candidates['X'])
        # With range_open=0, fallback uses range_high=12
        assert feats['range_size_pct'] == pytest.approx(16.666, abs=0.01)


# =========================================================================
# Fix #8: stop_price = range_high, single slippage buffer
# =========================================================================

class TestStopPriceAndSlippage:
    def test_stop_at_range_high_not_entry_price(self, engine, mock_alpaca):
        from trading.orb_planner import OrbTradePlan
        plan = OrbTradePlan(
            symbol='X', range_high=10.00, range_low=9.50, range_size=0.50,
            entry_price=10.03, stop_price=9.50, shares=100, position_dollars=1003,
            lock_arm_at_r=1.5, lock_stop_r=1.0, risk_per_share=0.53, total_risk=53,
            composite_score=0.5, quintile='Q4', adaptive_mult=0.95,
        )
        engine._submit_entry(plan)
        kw = mock_alpaca.submit_stop_bracket_order.call_args.kwargs
        # BT parity: trigger at range_high
        assert kw['stop_price'] == 10.00
        # Limit = entry_price (30bps above range_high — single buffer)
        assert kw['limit_price'] == 10.03


# =========================================================================
# Fix #10: partially_filled handling
# =========================================================================

class TestPartiallyFilledHandling:
    def test_partially_filled_treated_as_fill(self, engine, mock_alpaca, mock_sm):
        pos = OpenPosition(
            symbol='X', entry_price=10.03, stop_price=9.50, shares=100,
            trade_id=1, order_id='pending-1',
            entry_time=datetime.now(timezone.utc) - timedelta(seconds=10),
            range_high=10, range_low=9.5, lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
        )
        engine.open_positions['X'] = pos
        mock_alpaca.get_order.return_value = {
            'status': 'partially_filled',
            'filled_avg_price': 10.04,
            'filled_qty': 60,
        }
        engine._process_pending_fills()
        # Transitioned to filled state with partial shares
        assert engine.open_positions['X'].order_id == ''  # cleared
        assert engine.open_positions['X'].shares == 60
        # StopMonitor watch registered
        mock_sm.add_watch.assert_called_once()


# =========================================================================
# Fix #12: shutdown_requested
# =========================================================================

class TestShutdownRequested:
    def test_shutdown_short_circuits_check_entries(self, engine):
        engine.shutdown_requested = True
        # Even with valid candidates, should return []
        engine.build_universe(source_loader=lambda: ['X'])
        engine.candidates['X'].range_data = RangeData(
            symbol='X', range_high=10, range_low=9.5, range_volume=1000,
            range_avg_bar_range_pct=1.0, range_close=9.9,
            range_start_ts=pd.Timestamp.utcnow(), range_open=9.5,
        )
        assert engine.check_entries() == []

    def test_shutdown_requested_initialized_false(self, engine):
        assert engine.shutdown_requested is False


# =========================================================================
# Fix #6: daily_n_placed cap
# =========================================================================

class TestDailyPlacedCap:
    def test_cap_at_max_concurrent_total_per_day(self, engine, mock_alpaca):
        """Once daily_n_placed >= max_concurrent, no more entries even if slots free."""
        engine.daily_n_placed = engine.max_concurrent  # = 4
        # Add valid candidate
        engine.build_universe(source_loader=lambda: ['X'])
        engine.candidates['X'].range_data = RangeData(
            symbol='X', range_high=10, range_low=9.5, range_volume=1000,
            range_avg_bar_range_pct=1.0, range_close=9.9,
            range_start_ts=pd.Timestamp.utcnow(), range_open=9.5,
        )
        # No open_positions — slots "free" per concurrent check
        assert len(engine.open_positions) == 0
        # But daily cap blocks
        assert engine.check_entries() == []
        mock_alpaca.submit_stop_bracket_order.assert_not_called()


# =========================================================================
# Fix #5: lock_r_unit plumbed through
# =========================================================================

class TestLockRUnit:
    def test_confirm_fill_passes_range_size_as_lock_r_unit(self, engine, mock_sm):
        pos = OpenPosition(
            symbol='X', entry_price=10.03, stop_price=9.50, shares=100,
            trade_id=1, order_id='pending',
            entry_time=datetime.now(timezone.utc),
            range_high=10.00, range_low=9.50,
            lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
        )
        engine.open_positions['X'] = pos
        engine._confirm_fill(pos, {'status': 'filled', 'filled_avg_price': 10.03, 'filled_qty': 100})
        kw = mock_sm.add_watch.call_args.kwargs
        # range_size = range_high - range_low = 0.50
        assert kw['lock_r_unit'] == pytest.approx(0.50)
        # risk_per_share = entry - stop = 0.53
        assert kw['risk_per_share'] == pytest.approx(0.53, abs=0.01)

    def test_stop_monitor_uses_lock_r_unit_when_set(self):
        """StopMonitor arms based on lock_r_unit when set, NOT risk_per_share."""
        from unittest.mock import MagicMock, AsyncMock
        import asyncio

        main = MagicMock(spec=AlpacaClient)
        monitor = StopMonitor(api_key='k', api_secret='s', alpaca_client=main)

        # Add watch with lock_r_unit (range_size = 0.50) distinct from risk_per_share (0.53)
        monitor.add_watch(
            symbol='X', stop_price=9.50, shares=100,
            tp_leg_id='', sl_leg_id='',
            entry_price=10.03, risk_per_share=0.53,
            lock_arm_at_r=1.5, lock_stop_r=1.0,
            lock_r_unit=0.50,  # ← range_size, BT-parity
            strategy='orb',
        )

        # Price reaches arm level using range_size: 10.03 + 1.5 × 0.50 = 10.78
        # If we used risk_per_share (0.53), arm would be at 10.03 + 1.5 × 0.53 = 10.825 (NOT armed at 10.78)
        tr = MagicMock()
        tr.symbol = 'X'
        tr.price = 10.78  # exactly at BT's arm level
        asyncio.get_event_loop().run_until_complete(monitor._on_trade(tr))

        # Lock should be ARMED (range_size path); if using risk_per_share, it would NOT arm yet
        w = monitor._watches['X']
        assert w.lock_armed is True
        # New stop = 10.03 + 1.0 × 0.50 = 10.53
        assert w.stop_price == pytest.approx(10.53)


# =========================================================================
# Fix #4: sync_positions rehydrates pending orders
# =========================================================================

class TestSyncPendingOrders:
    def test_pending_order_on_alpaca_rehydrated(self, engine, mock_alpaca, mock_db):
        """DB has pending_new order, Alpaca still has it open → rehydrate as pending."""
        from datetime import date
        today = date.today()

        # Mock Alpaca: no open positions, but has pending order
        mock_alpaca.get_open_positions.return_value = []
        pending_order = MagicMock()
        pending_order.id = 'pending-1'
        mock_alpaca.trading_client.get_orders.return_value = [pending_order]

        # Mock DB: one pending_new trade
        import json as _json
        mock_db.get_open_trades.return_value = [{
            'id': 42,
            'symbol': 'TSLA',
            'order_id': 'pending-1',
            'order_status': 'pending_new',
            'entry_price': 10.03,
            'stop_loss_price': 9.50,
            'shares': 100,
            'strategy': 'orb',
            'pattern_data': _json.dumps({
                'range_high': 10.00, 'range_low': 9.50,
                'lock_arm_at_r': 1.5, 'lock_stop_r': 1.0,
                'quintile': 'Q4', 'composite_score': 0.5,
            }),
        }]

        engine.sync_positions()
        # Rehydrated as pending position
        assert 'TSLA' in engine.open_positions
        pos = engine.open_positions['TSLA']
        assert pos.order_id == 'pending-1'  # still pending
        assert pos.stop_price == 9.50
        assert pos.lock_arm_at_r == 1.5

    def test_pending_in_db_but_not_on_alpaca_marked_stale(self, engine, mock_alpaca, mock_db):
        """DB pending_new but order gone from Alpaca → stale."""
        from datetime import date
        mock_alpaca.get_open_positions.return_value = []
        mock_alpaca.trading_client.get_orders.return_value = []  # no pending
        mock_db.get_open_trades.return_value = [{
            'id': 42, 'symbol': 'GHOST',
            'order_id': 'o-gone', 'order_status': 'pending_new',
            'entry_price': 10, 'stop_loss_price': 9.5, 'shares': 100,
            'strategy': 'orb', 'pattern_data': '{}',
        }]
        engine.sync_positions()
        # Not rehydrated (stale)
        assert 'GHOST' not in engine.open_positions
        # DB marked stale_closed
        call = mock_db.update_trade.call_args
        assert call.args[0] == 42
        assert call.args[1]['order_status'] == 'stale_closed'


# =========================================================================
# Fix #3: build_orb_universe_from_snapshots
# =========================================================================

class TestSnapshotUniverse:
    def test_filters_by_bt_criteria(self, engine, mock_alpaca):
        """Only symbols matching gap >= 5% AND vol >= 500K AND price $3-30 are kept."""
        # Snapshots for 4 candidates; only AAA passes all criteria
        mock_alpaca.get_snapshots.return_value = {
            # AlpacaClient.get_snapshots returns a FLAT dict per symbol —
            # see alpaca_client.py::get_snapshots lines 566-585. Engine fix
            # 2271e83 made build_orb_universe_from_snapshots read this flat
            # shape directly (prior nested code was the bug).
            'AAA': {  # PASSES: gap 10%, vol 1M, price $10
                'open': 10.00, 'prev_close': 9.10, 'prev_volume': 1_000_000,
                'latest_price': 10.00,
            },
            'BBB': {  # FAILS gap (only 2%)
                'open': 10.00, 'prev_close': 9.80, 'prev_volume': 1_000_000,
                'latest_price': 10.00,
            },
            'CCC': {  # FAILS volume (100K)
                'open': 10.00, 'prev_close': 9.10, 'prev_volume': 100_000,
                'latest_price': 10.00,
            },
            'DDD': {  # FAILS price (above $30)
                'open': 50.00, 'prev_close': 45.00, 'prev_volume': 1_000_000,
                'latest_price': 50.00,
            },
        }
        kept = engine.build_orb_universe_from_snapshots(['AAA', 'BBB', 'CCC', 'DDD'])
        assert kept == ['AAA']

    def test_empty_candidates_returns_empty(self, engine, mock_alpaca):
        assert engine.build_orb_universe_from_snapshots([]) == []
        mock_alpaca.get_snapshots.assert_not_called()

    def test_snapshot_failure_returns_empty(self, engine, mock_alpaca):
        mock_alpaca.get_snapshots.side_effect = RuntimeError("API down")
        assert engine.build_orb_universe_from_snapshots(['X']) == []
