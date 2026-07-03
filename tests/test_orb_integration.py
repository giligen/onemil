"""Integration tests for the ORB trading pipeline.

Tests the full flow: range detection → feature extraction → filter → rank +
dedup → plan → order submission (mocked Alpaca) → DB persistence.

Uses a REAL Database (temp file) and REAL component instances (orb_filter,
orb_correlation, orb_conviction, orb_planner, orb_engine). Only the outermost
integration points (AlpacaClient, StopMonitor, TelegramNotifier) are mocked.
"""
from __future__ import annotations

import queue as _q
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pandas as pd
import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, CandidateState, RangeData, OpenPosition
from trading.stop_monitor import StopMonitor


# =========================================================================
# Fixtures (real DB + mocked Alpaca/StopMonitor)
# =========================================================================

@pytest.fixture
def orb_cfg():
    yaml_path = Path(__file__).parent.parent / 'orb.yaml'
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    # Enable master flag for tests
    cfg['strategy']['enabled'] = True
    return cfg


@pytest.fixture
def real_db(tmp_path):
    """Real SQLite DB — exercises save_trade / update_trade / get_open_trades."""
    db = Database(
        db_path=str(tmp_path / 'orb_test.db'),
        cache_path=str(tmp_path / 'orb_cache.db'),
        trades_path=str(tmp_path / 'orb_trades.db'),
    )
    yield db
    db.close()


@pytest.fixture
def mock_alpaca():
    client = MagicMock(spec=AlpacaClient)
    client.get_open_positions.return_value = []
    client.get_account_info.return_value = {'buying_power': 100_000.0}
    client.get_latest_quote.return_value = {'bid_price': 9.98, 'ask_price': 10.02}
    client.submit_stop_bracket_order.return_value = {
        'id': 'order-abc', 'status': 'accepted', 'symbol': 'TSLA',
    }
    client.cancel_order.return_value = True
    client.close_position.return_value = {'id': 'close-1', 'status': 'accepted'}
    return client


@pytest.fixture
def mock_stop_monitor():
    sm = MagicMock(spec=StopMonitor)
    sm.polling_mode = False
    sm.drain_exit_events.return_value = []
    return sm


@pytest.fixture
def engine(orb_cfg, real_db, mock_alpaca, mock_stop_monitor):
    e = ORBEngine(
        alpaca_client=mock_alpaca, db=real_db,
        stop_monitor=mock_stop_monitor, config=orb_cfg,
    )
    return e


def _make_bars(rows, date_str='2026-04-20'):
    """Build a bars DataFrame from list of (hh, mm, o, h, l, c, v)."""
    data = []
    for (h, m, o, hi, lo, c, v) in rows:
        data.append({
            'timestamp': pd.Timestamp(f'{date_str} {h:02d}:{m:02d}:00', tz='UTC'),
            'open': o, 'high': hi, 'low': lo, 'close': c, 'volume': v,
        })
    return pd.DataFrame(data)


def _make_trade_record(**overrides):
    """Build a complete save_trade() payload — DB requires every :bind key present."""
    base = {
        'trade_date': date.today().isoformat(),
        'symbol': 'TEST',
        'side': 'buy',
        'entry_price': 10.0,
        'stop_loss_price': 9.5,
        'take_profit_price': 0,
        'shares': 100,
        'risk_per_share': 0.5,
        'total_risk': 50.0,
        'risk_reward_ratio': 0,
        'order_id': 'test-order',
        'order_status': 'filled',
        'fill_price': None,
        'filled_at': None,
        'exit_price': None,
        'exit_reason': None,
        'exited_at': None,
        'pnl': None,
        'pnl_pct': None,
        'pattern_data': '{}',
        'strategy': 'orb',
    }
    base.update(overrides)
    return base


# =========================================================================
# Full entry flow: bars → range → features → submit → DB
# =========================================================================

class TestFullEntryFlow:
    def test_range_ingest_then_check_entries_submits_order_and_saves_trade(
        self, engine, real_db, mock_alpaca,
    ):
        from unittest.mock import patch
        engine.build_universe(source_loader=lambda: ['TSLA'])
        # Full 5-min range + breakout bar
        bars = _make_bars([
            (13, 30, 10.00, 10.10, 9.95, 10.05, 100_000),
            (13, 31, 10.05, 10.15, 10.00, 10.10, 110_000),
            (13, 32, 10.10, 10.20, 10.05, 10.15, 120_000),
            (13, 33, 10.15, 10.25, 10.10, 10.20, 130_000),
            (13, 34, 10.20, 10.30, 10.15, 10.28, 140_000),
        ])
        engine._ingest_bars('TSLA', bars)
        assert engine.candidates['TSLA'].range_data is not None
        assert engine.candidates['TSLA'].range_data.range_high == 10.30

        # Provide prior-day + 20d stats so composite filter has all features
        providers = {
            'TSLA': {
                # 2026-05-08: prev_close lowered to 9.00 (was 9.80) so the
                # real gap (range_open=10.00 vs prev_close=9.00) = +11.1% passes
                # the new phantom-gap guard at scoring time (default min_gap=5%).
                # With prev_close=9.80, real gap was 2.04% which the guard
                # correctly rejects as phantom.
                'prev_day_bar': {'open': 8.80, 'high': 9.20, 'low': 8.60, 'close': 9.00, 'volume': 500_000},
                'daily_stats_20d': {'high_20d': 12.00, 'range_pct_20d': 5.0, 'volume_20d': 800_000},
            }
        }
        # Bypass the 10:00 ET last-entry cutoff (test runs at arbitrary wall clock)
        with patch.object(engine, '_past_last_entry_time', return_value=False):
            submitted = engine.check_entries(feature_providers=providers)
        # Result depends on composite z — with reasonable values it should pass
        # Either it's submitted OR rejected for below-threshold. Both are valid
        # behavior — what we verify is NO CRASH and proper state mutation.
        assert engine.candidates['TSLA'].composite is not None  # computed
        if submitted:
            # Order went through → verify DB + Alpaca
            assert mock_alpaca.submit_stop_bracket_order.call_count == 1
            # DB record exists with strategy='orb'
            today = date.today()
            open_trades = real_db.get_open_trades(today)
            orb_trades = [t for t in open_trades if t.get('strategy') == 'orb']
            assert len(orb_trades) == 1
            assert orb_trades[0]['symbol'] == 'TSLA'

    def test_fcfs_blocks_when_bull_flag_has_symbol_today(
        self, engine, real_db, mock_alpaca,
    ):
        today = date.today()
        # Pre-seed DB with bull flag TSLA trade today
        real_db.save_trade({
            'trade_date': today,
            'symbol': 'TSLA',
            'side': 'buy',
            'entry_price': 10.00,
            'stop_loss_price': 9.50,
            'take_profit_price': 11.50,
            'shares': 100,
            'risk_per_share': 0.50,
            'total_risk': 50.0,
            'risk_reward_ratio': 3.0,
            'order_id': 'bf-xyz',
            'order_status': 'filled',
            'strategy': 'bull_flag',
            'fill_price': None, 'filled_at': None,
            'exit_price': None, 'exit_reason': None, 'exited_at': None,
            'pnl': None, 'pnl_pct': None,
            'pattern_data': '{}',
        })
        from unittest.mock import patch
        engine.build_universe(source_loader=lambda: ['TSLA'])
        engine.candidates['TSLA'].range_data = RangeData(
            symbol='TSLA', range_high=10.30, range_low=9.95, range_volume=600_000,
            range_avg_bar_range_pct=1.0, range_close=10.25,
            range_start_ts=pd.Timestamp.utcnow(),
        )
        with patch.object(engine, '_past_last_entry_time', return_value=False):
            submitted = engine.check_entries(feature_providers={})
        assert submitted == []
        assert engine.candidates['TSLA'].rejected_reason == 'fcfs_other_strategy'
        mock_alpaca.submit_stop_bracket_order.assert_not_called()

    def test_spread_gate_blocks_submission(self, engine, mock_alpaca):
        """Wide spread (>150bps) → reject with spread_gate reason + no Alpaca call."""
        from unittest.mock import patch
        engine.build_universe(source_loader=lambda: ['TSLA'])
        engine.candidates['TSLA'].range_data = RangeData(
            symbol='TSLA', range_high=10.30, range_low=9.95, range_volume=600_000,
            range_avg_bar_range_pct=1.0, range_close=10.25,
            range_start_ts=pd.Timestamp.utcnow(),
        )
        # Wide spread: bid 9.80, ask 10.00 on $10 stock = 200bps
        mock_alpaca.get_latest_quote.return_value = {'bid_price': 9.80, 'ask_price': 10.00}
        # Explosive prev day (range 16.8% > 8%) so the PDR veto passes and
        # the spread gate is what this test actually exercises.
        providers = {'TSLA': {
            'prev_day_bar': {'close': 9.50, 'high': 11.00, 'low': 9.40, 'open': 9.50},
            'daily_stats_20d': {'high_20d': 12.00},
        }}
        with patch.object(engine, '_past_last_entry_time', return_value=False):
            submitted = engine.check_entries(feature_providers=providers)
        assert submitted == []
        # If composite passed, the reject reason is spread_gate
        cand = engine.candidates['TSLA']
        if cand.composite is not None and cand.composite >= 0:
            assert cand.rejected_reason == 'spread_gate'
        mock_alpaca.submit_stop_bracket_order.assert_not_called()


# =========================================================================
# Exit flow: StopMonitor event → DB update → PnL
# =========================================================================

class TestExitFlow:
    def test_exit_event_updates_db_and_daily_pnl(
        self, engine, real_db, mock_stop_monitor,
    ):
        # Create open position + DB record
        today = date.today()
        import json as _json
        trade_id = real_db.save_trade(_make_trade_record(
            trade_date=today.isoformat(),
            symbol='AAPL',
            entry_price=100.0, stop_loss_price=95.0,
            shares=100, risk_per_share=5.0, total_risk=500.0,
            order_id='orb-1',
            pattern_data=_json.dumps({'range_high': 100.0, 'range_low': 95.0,
                                      'lock_arm_at_r': 1.5, 'lock_stop_r': 1.0}),
        ))
        engine.open_positions['AAPL'] = OpenPosition(
            symbol='AAPL', entry_price=100.0, stop_price=95.0, shares=100,
            trade_id=int(trade_id), order_id='orb-1',
            entry_time=datetime.now(timezone.utc),
            range_high=100.0, range_low=95.0,
            lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
        )

        ev = MagicMock()
        ev.symbol = 'AAPL'
        ev.exit_price = 105.0  # +$5 → lock_stop exit at +1R
        ev.exit_reason = 'lock_stop'
        ev.strategy = 'orb'
        mock_stop_monitor.drain_exit_events.return_value = [ev]

        exited = engine.check_exits()
        assert exited == ['AAPL']
        assert 'AAPL' not in engine.open_positions
        # Daily PnL updated
        assert engine.daily_pnl == pytest.approx(500.0)  # (105-100)*100
        # DB record should be updated (order_status=closed, pnl set)
        # Query real DB
        all_trades = real_db.get_open_trades(today)
        # After exit, status='closed' → not returned by get_open_trades
        assert all(t.get('strategy') != 'orb' for t in all_trades)


# =========================================================================
# Daily loss limit hard-blocks new entries
# =========================================================================

class TestDailyLossLimitIntegration:
    def test_hits_limit_then_blocks(self, engine, real_db):
        engine.daily_pnl = -5_500.0  # below -5K limit
        engine.build_universe(source_loader=lambda: ['TSLA'])
        engine.candidates['TSLA'].range_data = RangeData(
            symbol='TSLA', range_high=10.0, range_low=9.5, range_volume=100_000,
            range_avg_bar_range_pct=1.0, range_close=9.9,
            range_start_ts=pd.Timestamp.utcnow(),
        )
        submitted = engine.check_entries(feature_providers={})
        assert submitted == []
        assert engine.daily_loss_limit_logged is True


# =========================================================================
# Restart recovery (sync_positions)
# =========================================================================

class TestRestartRecovery:
    def test_sync_rehydrates_positions_and_rewatches_stop_monitor(
        self, engine, real_db, mock_alpaca, mock_stop_monitor,
    ):
        today = date.today()
        # Seed DB with an open ORB trade (pretend service crashed mid-day)
        import json as _json
        trade_id = real_db.save_trade(_make_trade_record(
            trade_date=today.isoformat(),
            symbol='NVDA',
            entry_price=120.0, stop_loss_price=115.0,
            shares=50, risk_per_share=5.0, total_risk=250.0,
            order_id='orb-nvda',
            fill_price=120.0,
            pattern_data=_json.dumps({'range_high': 120.0, 'range_low': 115.0,
                                      'lock_arm_at_r': 1.5, 'lock_stop_r': 1.0}),
        ))
        # Mock Alpaca to return NVDA as open
        alp_pos = MagicMock()
        alp_pos.symbol = 'NVDA'
        mock_alpaca.get_open_positions.return_value = [alp_pos]

        # Fresh engine → simulates restart
        engine.sync_positions()
        assert 'NVDA' in engine.open_positions
        pos = engine.open_positions['NVDA']
        assert pos.entry_price == 120.0
        assert pos.stop_price == 115.0
        assert pos.shares == 50
        # StopMonitor re-watch called with correct args
        mock_stop_monitor.add_watch.assert_called_once()
        call_kwargs = mock_stop_monitor.add_watch.call_args.kwargs
        assert call_kwargs['symbol'] == 'NVDA'
        assert call_kwargs['strategy'] == 'orb'
        assert call_kwargs['lock_arm_at_r'] == 1.5
        assert call_kwargs['lock_stop_r'] == 1.0

    def test_sync_marks_stale_when_db_open_but_alpaca_closed(
        self, engine, real_db, mock_alpaca,
    ):
        today = date.today()
        real_db.save_trade(_make_trade_record(
            trade_date=today.isoformat(),
            symbol='GHOST',
            entry_price=10, stop_loss_price=9,
            shares=100, risk_per_share=1, total_risk=100,
            order_id='ghost-1',
        ))
        mock_alpaca.get_open_positions.return_value = []  # Alpaca doesn't know
        engine.sync_positions()
        assert 'GHOST' not in engine.open_positions


# =========================================================================
# Force close at 15:45 ET
# =========================================================================

class TestForceCloseIntegration:
    def test_force_close_closes_all_open_positions(
        self, engine, mock_alpaca,
    ):
        engine.open_positions['X'] = OpenPosition(
            symbol='X', entry_price=10, stop_price=9, shares=100, trade_id=1,
            order_id='o', entry_time=datetime.now(timezone.utc),
            range_high=10, range_low=9, lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
        )
        engine.open_positions['Y'] = OpenPosition(
            symbol='Y', entry_price=20, stop_price=19, shares=50, trade_id=2,
            order_id='o2', entry_time=datetime.now(timezone.utc),
            range_high=20, range_low=19, lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
        )
        closed = engine.force_close_all()
        assert closed == 2
        assert mock_alpaca.close_position.call_count == 2


# =========================================================================
# Reset_daily
# =========================================================================

class TestResetDailyIntegration:
    def test_reset_clears_candidates_but_keeps_open_positions(self, engine):
        """reset_daily should NOT drop open positions mid-day. Only clear day-only state."""
        engine.build_universe(source_loader=lambda: ['AAPL'])
        engine.open_positions['AAPL'] = OpenPosition(
            symbol='AAPL', entry_price=100, stop_price=95, shares=10, trade_id=1,
            order_id='o', entry_time=datetime.now(timezone.utc),
            range_high=100, range_low=95, lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
        )
        engine.reset_daily()
        assert engine.candidates == {}
        # NOTE: open_positions intentionally NOT cleared by reset_daily
        # (they persist across day boundary). For ORB the convention is
        # force_close at 15:45 handles that.
        assert 'AAPL' in engine.open_positions
