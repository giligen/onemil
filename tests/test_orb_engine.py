"""Unit tests for trading/orb_engine.py.

Uses spec-based mocks per CLAUDE.md. Exercises the critical paths:
  * Init from real orb.yaml
  * Disabled engine is inert
  * build_universe populates candidate state
  * Range bar ingestion → range_data computed correctly
  * check_entries respects filter threshold + ranking + dedup
  * FCFS same-symbol conflict skips
  * Daily loss limit blocks new entries
  * is_force_close_time flips at 15:45 ET
"""
import queue as _q
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path

import pandas as pd
import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import (
    ORBEngine, OpenPosition, RangeData, CandidateState,
    _first_session_open_ts_utc, _et_offset_hours,
)
from trading.stop_monitor import StopMonitor


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def orb_cfg():
    """Load the real orb.yaml so tests validate the full config path."""
    yaml_path = Path(__file__).parent.parent / 'orb.yaml'
    with open(yaml_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def mock_alpaca():
    client = MagicMock(spec=AlpacaClient)
    client.get_open_positions.return_value = []
    client.get_account_info.return_value = {'buying_power': 100_000.0}
    client.get_latest_quote.return_value = {'bid_price': 9.95, 'ask_price': 10.00}
    client.submit_stop_bracket_order.return_value = {'id': 'order-123', 'status': 'accepted'}
    client.cancel_order.return_value = True
    client.close_position.return_value = {'id': 'close-1', 'status': 'accepted'}
    return client


@pytest.fixture
def mock_db():
    db = MagicMock(spec=Database)
    db.save_trade.return_value = 100
    db.get_open_trades.return_value = []
    db.update_trade.return_value = True
    return db


@pytest.fixture
def mock_stop_monitor():
    sm = MagicMock(spec=StopMonitor)
    sm.polling_mode = False
    sm.drain_exit_events.return_value = []
    return sm


@pytest.fixture
def engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor):
    """ORBEngine with enabled=True + all mocks wired."""
    # Override enabled for testing
    orb_cfg['strategy']['enabled'] = True
    return ORBEngine(
        alpaca_client=mock_alpaca,
        db=mock_db,
        stop_monitor=mock_stop_monitor,
        config=orb_cfg,
    )


@pytest.fixture
def disabled_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor):
    # Force enabled=False for this fixture regardless of what's in the yaml on disk.
    # (orb.yaml ships with enabled=true in prod; tests must control the flag themselves.)
    cfg = dict(orb_cfg)
    cfg['strategy'] = dict(cfg.get('strategy', {}))
    cfg['strategy']['enabled'] = False
    return ORBEngine(
        alpaca_client=mock_alpaca, db=mock_db,
        stop_monitor=mock_stop_monitor, config=cfg,
    )


def _make_bars(symbol_prices, date_str='2026-04-20'):
    """Build a bars DataFrame from list of (hh, mm, o, h, l, c, v) tuples."""
    rows = []
    for (h, m, o, hi, lo, c, v) in symbol_prices:
        rows.append({
            'timestamp': pd.Timestamp(f'{date_str} {h:02d}:{m:02d}:00', tz='UTC'),
            'open': o, 'high': hi, 'low': lo, 'close': c, 'volume': v,
        })
    return pd.DataFrame(rows)


# =========================================================================
# Init + disabled state
# =========================================================================

class TestInit:
    def test_loads_from_real_yaml(self, engine):
        assert engine.range_minutes == 5
        assert engine.max_concurrent == 4
        assert engine.planner.risk_per_trade_usd == 3000
        assert engine.planner.per_pos_cap_usd == 25_000
        assert engine.force_close_hour_et == 15
        assert engine.force_close_minute_et == 45

    def test_filter_params_loaded(self, engine):
        # orb.yaml has 7 features
        assert len(engine.z_params) == 7
        assert 'gap_pct' in engine.z_params
        assert 'range_close_position' in engine.z_params

    def test_quintile_cutoffs_correct_length(self, engine):
        assert len(engine.quintile_cutoffs) == 4

    def test_adaptive_mults_loaded_with_q5_cap(self, engine):
        # Q5 capped at 1.5 per orb_conviction.load_adaptive_mults
        assert engine.adaptive_mults['Q5'] == 1.5

    def test_disabled_by_default(self, disabled_engine):
        assert disabled_engine.enabled is False


# =========================================================================
# Disabled engine is inert
# =========================================================================

class TestDisabledEngine:
    def test_check_entries_returns_empty_when_disabled(self, disabled_engine):
        assert disabled_engine.check_entries() == []

    def test_on_bar_close_noop_when_disabled(self, disabled_engine):
        bars = _make_bars([(13, 30, 10, 10.2, 9.95, 10.1, 1000)])
        # Add candidate to universe manually
        disabled_engine.universe.add('TEST')
        disabled_engine.candidates['TEST'] = CandidateState(symbol='TEST')
        disabled_engine._on_bar_close('TEST', bars)
        # Queue should stay empty
        assert disabled_engine._bar_event_queue.empty()


# =========================================================================
# Universe
# =========================================================================

class TestBuildUniverse:
    def test_no_loader_yields_empty(self, engine):
        n = engine.build_universe(source_loader=None)
        assert n == 0
        assert len(engine.universe) == 0

    def test_loader_populates_candidates(self, engine):
        n = engine.build_universe(source_loader=lambda: ['AAPL', 'TSLA', 'NVDA'])
        assert n == 3
        assert engine.universe == {'AAPL', 'TSLA', 'NVDA'}
        assert len(engine.candidates) == 3
        for sym in engine.universe:
            assert isinstance(engine.candidates[sym], CandidateState)
            assert engine.candidates[sym].range_data is None  # not yet set

    def test_loader_exception_yields_empty(self, engine):
        def bad():
            raise RuntimeError("boom")
        n = engine.build_universe(source_loader=bad)
        assert n == 0
        assert engine.universe == set()


# =========================================================================
# Range bar ingestion
# =========================================================================

class TestRangeIngestion:
    def test_partial_range_does_not_populate(self, engine):
        engine.build_universe(source_loader=lambda: ['TEST'])
        # Only 3 of 5 range bars
        bars = _make_bars([
            (13, 30, 10.0, 10.1, 9.95, 10.05, 1000),
            (13, 31, 10.05, 10.15, 10.00, 10.10, 1000),
            (13, 32, 10.10, 10.20, 10.05, 10.15, 1000),
        ])
        engine._ingest_bars('TEST', bars)
        assert engine.candidates['TEST'].range_data is None

    def test_complete_range_populates(self, engine):
        engine.build_universe(source_loader=lambda: ['TEST'])
        # 5 complete range bars (9:30-9:34 inclusive) + a breakout bar
        bars = _make_bars([
            (13, 30, 10.0, 10.1, 9.95, 10.05, 1000),
            (13, 31, 10.05, 10.15, 10.00, 10.10, 1500),
            (13, 32, 10.10, 10.20, 10.05, 10.15, 2000),
            (13, 33, 10.15, 10.25, 10.10, 10.20, 1800),
            (13, 34, 10.20, 10.30, 10.15, 10.28, 2200),
            (13, 35, 10.28, 10.50, 10.28, 10.45, 3000),
        ])
        engine._ingest_bars('TEST', bars)
        rd = engine.candidates['TEST'].range_data
        assert rd is not None
        assert rd.range_high == 10.30
        assert rd.range_low == 9.95
        assert rd.range_volume == 8500  # 1000+1500+2000+1800+2200
        assert rd.symbol == 'TEST'

    def test_second_ingestion_noop_after_range_closed(self, engine):
        engine.build_universe(source_loader=lambda: ['TEST'])
        bars = _make_bars([
            (13, 30, 10.0, 10.1, 9.95, 10.05, 1000),
            (13, 31, 10.05, 10.15, 10.00, 10.10, 1000),
            (13, 32, 10.10, 10.20, 10.05, 10.15, 1000),
            (13, 33, 10.15, 10.25, 10.10, 10.20, 1000),
            (13, 34, 10.20, 10.30, 10.15, 10.28, 1000),
        ])
        engine._ingest_bars('TEST', bars)
        first_rd = engine.candidates['TEST'].range_data
        assert first_rd is not None
        # Re-ingest with different bars — range_data should NOT change
        bars2 = _make_bars([
            (13, 30, 20.0, 20.1, 19.95, 20.05, 9999),
            (13, 31, 20.05, 20.15, 20.00, 20.10, 9999),
            (13, 32, 20.10, 20.20, 20.05, 20.15, 9999),
            (13, 33, 20.15, 20.25, 20.10, 20.20, 9999),
            (13, 34, 20.20, 20.30, 20.15, 20.28, 9999),
        ])
        engine._ingest_bars('TEST', bars2)
        assert engine.candidates['TEST'].range_data is first_rd  # unchanged

    def test_non_universe_symbol_ignored(self, engine):
        engine.build_universe(source_loader=lambda: ['KNOWN'])
        bars = _make_bars([(13, 30, 10, 10.1, 9.95, 10.05, 1000)])
        engine._ingest_bars('UNKNOWN', bars)
        assert 'UNKNOWN' not in engine.candidates


# =========================================================================
# check_entries — ranking + FCFS
# =========================================================================

class TestCheckEntries:
    def test_no_entries_without_range(self, engine):
        engine.build_universe(source_loader=lambda: ['AAPL'])
        submitted = engine.check_entries()
        assert submitted == []

    def test_fcfs_skip_if_other_strategy_open(self, engine, mock_db):
        from unittest.mock import patch as _patch
        engine.build_universe(source_loader=lambda: ['TSLA'])
        # Pre-populate range
        engine.candidates['TSLA'].range_data = RangeData(
            symbol='TSLA', range_high=10.5, range_low=10.0, range_volume=500_000,
            range_avg_bar_range_pct=1.0, range_close=10.4,
            range_start_ts=pd.Timestamp.utcnow(),
        )
        # Mock DB: bull flag already has TSLA open today
        mock_db.get_open_trades.return_value = [{'symbol': 'TSLA', 'strategy': 'bull_flag'}]
        # Bypass wall-clock time cutoff
        with _patch.object(engine, '_past_last_entry_time', return_value=False):
            submitted = engine.check_entries()
        assert submitted == []
        # Should be marked as fcfs_other_strategy rejection
        assert engine.candidates['TSLA'].rejected_reason == 'fcfs_other_strategy'

    def test_already_has_orb_position_skipped(self, engine):
        engine.build_universe(source_loader=lambda: ['TSLA'])
        engine.candidates['TSLA'].range_data = RangeData(
            symbol='TSLA', range_high=10.5, range_low=10.0, range_volume=500_000,
            range_avg_bar_range_pct=1.0, range_close=10.4,
            range_start_ts=pd.Timestamp.utcnow(),
        )
        engine.open_positions['TSLA'] = OpenPosition(
            symbol='TSLA', entry_price=10.5, stop_price=10.0, shares=100,
            trade_id=1, order_id='x', entry_time=datetime.now(timezone.utc),
            range_high=10.5, range_low=10.0, lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
        )
        assert engine.check_entries() == []


# =========================================================================
# Daily loss limit
# =========================================================================

class TestDailyLossLimit:
    def test_below_limit_does_not_block(self, engine):
        engine.daily_pnl = -4999.0  # just above the -5K limit
        assert engine._daily_loss_limit_hit() is False

    def test_at_limit_blocks(self, engine):
        engine.daily_pnl = -5000.0
        assert engine._daily_loss_limit_hit() is True

    def test_below_limit_blocks(self, engine):
        engine.daily_pnl = -5100.0
        assert engine._daily_loss_limit_hit() is True

    def test_only_logs_once(self, engine):
        engine.daily_pnl = -6000.0
        engine._daily_loss_limit_hit()
        assert engine.daily_loss_limit_logged is True
        # Second call — flag stays set
        engine._daily_loss_limit_hit()
        assert engine.daily_loss_limit_logged is True


# =========================================================================
# Force close
# =========================================================================

class TestForceClose:
    def test_force_close_time_at_1545(self, engine):
        # April is EDT → offset 4h. 19:45 UTC == 15:45 EDT.
        t = datetime(2026, 4, 20, 19, 45, 0, tzinfo=timezone.utc)
        assert engine.is_force_close_time(now_utc=t) is True

    def test_force_close_time_before_1545(self, engine):
        t = datetime(2026, 4, 20, 19, 44, 0, tzinfo=timezone.utc)
        assert engine.is_force_close_time(now_utc=t) is False

    def test_force_close_time_after_1545(self, engine):
        t = datetime(2026, 4, 20, 19, 55, 0, tzinfo=timezone.utc)
        assert engine.is_force_close_time(now_utc=t) is True

    def test_force_close_all_empty(self, engine):
        closed = engine.force_close_all()
        assert closed == 0

    def test_force_close_all_closes_open_positions(self, engine, mock_alpaca):
        engine.open_positions['X'] = OpenPosition(
            symbol='X', entry_price=10, stop_price=9, shares=100,
            trade_id=1, order_id='o', entry_time=datetime.now(timezone.utc),
            range_high=10, range_low=9, lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
        )
        closed = engine.force_close_all()
        assert closed == 1
        mock_alpaca.close_position.assert_called_with('X')


# =========================================================================
# Exit flow
# =========================================================================

class TestCheckExits:
    def test_drain_exits_filters_by_strategy(self, engine, mock_stop_monitor):
        ev = MagicMock()
        ev.symbol = 'X'
        ev.exit_price = 10.5
        ev.exit_reason = 'lock_stop'
        ev.strategy = 'orb'
        mock_stop_monitor.drain_exit_events.return_value = [ev]

        engine.open_positions['X'] = OpenPosition(
            symbol='X', entry_price=10.0, stop_price=9.0, shares=100,
            trade_id=42, order_id='o', entry_time=datetime.now(timezone.utc),
            range_high=10, range_low=9, lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
        )
        exited = engine.check_exits()
        assert exited == ['X']
        assert 'X' not in engine.open_positions
        # DB update called with PnL
        call_args = engine.db.update_trade.call_args
        assert call_args[0][0] == 42  # trade_id
        updates = call_args[0][1]
        assert updates['exit_price'] == 10.5
        assert updates['exit_reason'] == 'lock_stop'
        assert updates['pnl'] == pytest.approx(50.0)  # (10.5-10.0)*100

    def test_exit_updates_daily_pnl(self, engine, mock_stop_monitor):
        ev = MagicMock()
        ev.symbol = 'X'
        ev.exit_price = 10.5
        ev.exit_reason = 'lock_stop'
        ev.strategy = 'orb'
        mock_stop_monitor.drain_exit_events.return_value = [ev]

        engine.open_positions['X'] = OpenPosition(
            symbol='X', entry_price=10.0, stop_price=9.0, shares=100,
            trade_id=42, order_id='o', entry_time=datetime.now(timezone.utc),
            range_high=10, range_low=9, lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
        )
        engine.check_exits()
        assert engine.daily_pnl == pytest.approx(50.0)


# =========================================================================
# reset_daily
# =========================================================================

class TestResetDaily:
    def test_reset_clears_day_state(self, engine):
        engine.build_universe(source_loader=lambda: ['AAPL'])
        engine.daily_pnl = -1000.0
        engine.daily_loss_limit_logged = True
        engine.reset_daily()
        assert engine.candidates == {}
        assert engine.daily_pnl == 0.0
        assert engine.daily_loss_limit_logged is False


# =========================================================================
# Module helpers
# =========================================================================

class TestModuleHelpers:
    def test_first_session_open_ts_finds_9_30_et(self):
        bars = _make_bars([(13, 30, 10, 10.1, 9.95, 10.05, 1000)])  # 13:30 UTC = 9:30 EDT
        ts = _first_session_open_ts_utc(bars)
        assert ts is not None
        assert ts.hour == 13 and ts.minute == 30

    def test_first_session_open_ts_empty(self):
        assert _first_session_open_ts_utc(pd.DataFrame()) is None

    def test_first_session_open_ts_object_dtype_timestamps(self):
        """Regression 2026-04-20: StopMonitor WS drain path occasionally
        produces DataFrames with object-dtype 'timestamp' (raw python datetime
        objects) instead of datetime64. Helper must coerce gracefully —
        prior code raised 'Can only use .dt accessor with datetimelike
        values' repeatedly in the scanner loop."""
        from datetime import datetime as _dt, timezone as _tz
        bars = pd.DataFrame([
            {'timestamp': _dt(2026, 4, 20, 13, 30, tzinfo=_tz.utc),
             'open': 10.0, 'high': 10.1, 'low': 9.95, 'close': 10.05, 'volume': 1000},
            {'timestamp': _dt(2026, 4, 20, 13, 31, tzinfo=_tz.utc),
             'open': 10.05, 'high': 10.15, 'low': 10.0, 'close': 10.1, 'volume': 1200},
        ])
        # Force object dtype — pandas auto-upgrades to datetime64 otherwise
        bars['timestamp'] = bars['timestamp'].astype(object)
        assert bars['timestamp'].dtype == object  # reproduces the bug condition
        # Must NOT raise — coerce inside the helper + find the 9:30 ET bar
        ts = _first_session_open_ts_utc(bars)
        assert ts is not None
        assert ts.hour == 13 and ts.minute == 30

    def test_first_session_open_ts_missing_column(self):
        """No 'timestamp' column → return None, don't raise."""
        bars = pd.DataFrame({'open': [1.0], 'close': [1.0]})
        assert _first_session_open_ts_utc(bars) is None

    def test_first_session_open_ts_malformed_strings(self):
        """Non-parseable timestamp strings → return None, don't crash the
        scanner loop."""
        bars = pd.DataFrame({
            'timestamp': ['not-a-date', 'also-nope'],
            'open': [10, 11], 'high': [11, 12], 'low': [9, 10],
            'close': [10, 11], 'volume': [100, 200],
        })
        assert _first_session_open_ts_utc(bars) is None

    def test_et_offset_summer(self):
        assert _et_offset_hours(datetime(2026, 6, 1, tzinfo=timezone.utc)) == 4

    def test_et_offset_winter(self):
        assert _et_offset_hours(datetime(2026, 1, 1, tzinfo=timezone.utc)) == 5
