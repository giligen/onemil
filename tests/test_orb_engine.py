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
# 2026-04-29 OPRA incident — force-close MUST close orphans (Alpaca sweep)
# =========================================================================
# OPRA was bought by ORB on 4/28 during a service-crash window. Fill never
# reached DB. ORB's open_positions was empty at force-close. force_close_all
# iterated engine state, closed nothing, exited normally. Position carried
# overnight.
#
# Fix: after the engine-state pass, query Alpaca for ANY remaining position
# on the account and close it (source-of-truth = Alpaca, not engine state).

class TestForceCloseOrphanSweep:

    def _orphan_position(self, symbol='OPRA', qty=2508, avg_entry=18.34, upl=-1229.0):
        """Build a mock Alpaca position object."""
        p = MagicMock()
        p.symbol = symbol
        p.qty = qty
        p.avg_entry_price = avg_entry
        p.unrealized_pl = upl
        return p

    def test_orphan_in_alpaca_but_not_engine_gets_swept(self, engine, mock_alpaca):
        """If Alpaca has a position but engine.open_positions is empty,
        force_close_all must still close the position via the sweep path.

        This is the OPRA 4/28 scenario directly.
        """
        # Engine state: empty (mirrors crash-induced state loss)
        assert engine.open_positions == {}
        # OPRA was an ORB position — ORB submitted the buy, so a DB row exists.
        engine._orb_owned_symbols = lambda *a, **k: {'OPRA'}
        # Alpaca shows the orphan on the FIRST query (sweep), then empty on
        # the SECOND query (post-FC verification).
        mock_alpaca.get_open_positions.side_effect = [
            [self._orphan_position()],
            [],
        ]

        closed = engine.force_close_all()

        assert closed == 1, "Sweep must close 1 orphan even with empty engine state"
        mock_alpaca.close_position.assert_called_with('OPRA')

    def test_sweep_runs_after_engine_state_close(self, engine, mock_alpaca):
        """When engine has a tracked position AND Alpaca has an additional
        orphan, both get closed. Tracked first, then orphan via sweep.
        """
        engine.open_positions['TRACKED'] = OpenPosition(
            symbol='TRACKED', entry_price=10, stop_price=9, shares=100,
            trade_id=1, order_id='o', entry_time=datetime.now(timezone.utc),
            range_high=10, range_low=9, lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
        )
        # First sweep query: orphan only (TRACKED already closed by engine path)
        # Second query: post-verification, all gone
        engine._orb_owned_symbols = lambda *a, **k: {'TRACKED', 'ORPHAN'}
        mock_alpaca.get_open_positions.side_effect = [
            [self._orphan_position(symbol='ORPHAN')],
            [],
        ]

        closed = engine.force_close_all()

        assert closed == 2, f"Expected TRACKED + ORPHAN both closed, got {closed}"
        # close_position called for both
        called_symbols = {
            c.args[0] for c in mock_alpaca.close_position.call_args_list
        }
        assert 'TRACKED' in called_symbols
        assert 'ORPHAN' in called_symbols

    def test_post_fc_verification_alerts_when_position_remains(
        self, engine, mock_alpaca, monkeypatch,
    ):
        """If Alpaca still shows positions AFTER sweep + 3 retries, alert fires.

        2026-05-07 hardening: previously the verify ran once with 1s sleep and
        immediately alerted on failure. Now does verify-with-grace + 3 retries
        with backoffs. Test shrinks the tunables to keep run-time small.
        Alert only fires AFTER all 3 retries fail — that's the contract.
        """
        # Always return the stuck orphan — no grace period or retry will clear it
        engine._orb_owned_symbols = lambda *a, **k: {'STUCK'}
        mock_alpaca.get_open_positions.return_value = [
            self._orphan_position(symbol='STUCK')
        ]
        # Track _notify_error calls
        engine._notify_error = MagicMock()
        # Shrink tunables for fast test execution
        engine.fc_verify_max_wait_s = 0.1
        engine.fc_verify_poll_interval_s = 0.01
        engine.fc_retry_backoffs_s = [0.0, 0.0, 0.0]
        # Patch sleep too in case the engine-pass section sleeps
        import time as _real_time
        monkeypatch.setattr(_real_time, 'sleep', lambda *a, **kw: None)

        engine.force_close_all()

        # The verification should have raised CRITICAL after 3 retries
        # 2026-05-11 fix (Bug-3 from post-code-review): per-phase alerts
        # consolidated into ONE 'FC FINAL FAILURE' alert at end-of-FC.
        # Old phrasing was 'VERIFY FAILED after N retries'; new phrasing
        # is 'FC FINAL FAILURE: ... after Phase1+SWEEP+VERIFY(N retries)'.
        verify_alerts = [
            c for c in engine._notify_error.call_args_list
            if 'FC FINAL FAILURE' in str(c)
        ]
        assert len(verify_alerts) >= 1, (
            "Post-FC verification did NOT alert when position survived sweep+retries"
        )

    def test_sweep_handles_alpaca_query_failure_gracefully(
        self, engine, mock_alpaca,
    ):
        """If Alpaca query during sweep raises, FC still completes (no crash).

        Alerts via _notify_error so operator knows verification was skipped.
        """
        engine.open_positions['X'] = OpenPosition(
            symbol='X', entry_price=10, stop_price=9, shares=100,
            trade_id=1, order_id='o', entry_time=datetime.now(timezone.utc),
            range_high=10, range_low=9, lock_arm_at_r=1.5, lock_stop_r=1.0,
            composite_score=0.5, quintile='Q4',
        )
        mock_alpaca.get_open_positions.side_effect = RuntimeError("API down")
        engine._notify_error = MagicMock()

        # Must not raise
        closed = engine.force_close_all()

        # Engine-state X still closed, sweep alerted but didn't fail
        assert closed == 1
        assert any(
            'SWEEP' in str(c) for c in engine._notify_error.call_args_list
        ), "Sweep query failure was not alerted"

    def test_non_orb_position_not_swept(self, engine, mock_alpaca):
        """Regression (2026-05-22): a position ORB does NOT own (a divergence
        CLF short on the shared account) must NOT be closed by the FC sweep.
        An ORB-owned position in the same query is still swept.
        """
        engine._orb_owned_symbols = lambda *a, **k: {'OPRA'}  # ORB owns OPRA only
        orb_pos = self._orphan_position(symbol='OPRA')
        clf = self._orphan_position(symbol='CLF', qty=-3690, avg_entry=10.84, upl=6.0)
        mock_alpaca.get_open_positions.side_effect = [
            [orb_pos, clf],          # sweep sees both
            [clf], [clf], [clf],     # verify polls: CLF lingers, correctly ignored
        ]

        engine.force_close_all()

        closed = {c.args[0] for c in mock_alpaca.close_position.call_args_list}
        assert 'OPRA' in closed, "ORB-owned OPRA must be swept"
        assert 'CLF' not in closed, "Non-ORB CLF must NOT be closed by ORB"

    def test_verify_ignores_non_orb_position(self, engine, mock_alpaca):
        """Post-FC verify must not re-close — or raise FC FINAL FAILURE for —
        a non-ORB position that legitimately remains on the shared account.
        """
        engine._orb_owned_symbols = lambda *a, **k: set()  # ORB owns nothing
        engine._notify_error = MagicMock()
        clf = self._orphan_position(symbol='CLF', qty=-3690)
        mock_alpaca.get_open_positions.return_value = [clf]  # CLF always present

        engine.force_close_all()

        assert mock_alpaca.close_position.call_count == 0, "CLF must never be closed"
        assert not any(
            'FC FINAL FAILURE' in str(c)
            for c in engine._notify_error.call_args_list
        ), "Non-ORB CLF must not trip an FC-failure alert"


# =========================================================================
# Sync_positions — orphan auto-close at off-hours startup
# =========================================================================

class TestSyncPositionsOrphanAutoClose:

    def _orphan_position(self, symbol='OPRA'):
        p = MagicMock()
        p.symbol = symbol
        p.qty = 100
        p.avg_entry_price = 10.0
        p.unrealized_pl = -50.0
        return p

    @pytest.fixture
    def patched_orphan_engine(self, engine, mock_alpaca, mock_db):
        """Engine with sync_positions inputs configured to produce an orphan.

        sync_positions queries:
        - alpaca.get_open_positions → returns the OPRA orphan
        - alpaca.trading_client.get_orders → none open (no in-flight orders)
        - db.get_open_trades(today) → empty (no DB record → classifies as orphan)

        2026-06-05: the new orphan_reconciler issues a direct SQL query via
        db._trades_conn.execute(...). We rig a stub cursor whose .fetchall()
        returns matching rows by default — individual tests override by
        setting `engine._reconciler_rows` on the engine.
        """
        from trading.orphan_reconciler import reset_state_for_tests
        reset_state_for_tests()  # don't carry alert cooldowns across tests
        mock_alpaca.get_open_positions.return_value = [{
            'symbol': 'OPRA', 'qty': 100, 'avg_entry_price': 10.0,
            'unrealized_pl': -50.0, 'side': 'long',
        }]
        mock_alpaca.close_position.return_value = {'id': 'recon-1'}
        mock_alpaca.get_order.return_value = {
            'filled_qty': 100, 'filled_avg_price': 9.50,
        }
        # _cancel_symbol_open_orders uses trading_client.get_orders; mock it
        mock_alpaca.trading_client = MagicMock()
        mock_alpaca.trading_client.get_orders.return_value = []
        mock_alpaca.trading_client.cancel_order_by_id.return_value = True
        mock_db.get_open_trades.return_value = []

        # Reconciler now prefers the public
        # Database.get_strategy_trades_in_window — implement it via a
        # closure so individual tests can override engine._reconciler_rows.
        engine._reconciler_rows = [{
            'id': 99, 'trade_date': '2026-06-01', 'symbol': 'OPRA',
            'strategy': 'orb', 'fill_price': 10.0, 'filled_qty': 100,
            'exit_price': None, 'exit_reason': 'stop_loss_unconfirmed',
            'order_status': 'closed',
        }]

        def _get_strategy_trades(strategy, since_date, symbols=None):
            out = []
            for r in engine._reconciler_rows:
                if r.get('strategy') != strategy:
                    continue
                if symbols and r['symbol'] not in symbols:
                    continue
                out.append(dict(r))
            return out

        mock_db.get_strategy_trades_in_window = _get_strategy_trades

        # Back-compat fallback if anything still pokes the private path.
        class _Cur:
            def __init__(self, rows): self._rows = rows
            def fetchall(self): return list(self._rows)

        def _execute(sql, params):
            return _Cur(engine._reconciler_rows)

        mock_db._trades_conn = MagicMock()
        mock_db._trades_conn.execute = _execute
        return engine

    def test_orphan_auto_closes_at_premarket(
        self, patched_orphan_engine, mock_alpaca, monkeypatch,
    ):
        """Premarket startup (e.g. 8:30 ET) → ORB-owned orphan auto-closed.

        The orphan_reconciler now drives this. The cross-day STALE row
        (trade_date='2026-06-01' against today=2026-06-05) plus
        avg-entry match and qty sanity satisfy the predicate.
        """
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo
        et_tz = ZoneInfo('America/New_York')
        # Force "today" to 2026-06-05 ET so the stale rows are cross-day.
        fake_now = _dt(2026, 6, 5, 8, 30, 0, tzinfo=et_tz)

        import trading.orb_engine as orb_mod
        real_dt = orb_mod.datetime
        class MockDT:
            @staticmethod
            def now(tz=None):
                if tz is not None:
                    return fake_now.astimezone(tz)
                return fake_now.replace(tzinfo=None)
            def __getattr__(self, name):
                return getattr(real_dt, name)
        monkeypatch.setattr(orb_mod, 'datetime', MockDT())

        # The reconciler also reads datetime.now via its own module.
        # Reset its rate-limit state so the test doesn't see stale cooldowns.
        from trading.orphan_reconciler import reset_state_for_tests
        reset_state_for_tests()

        patched_orphan_engine.sync_positions()

        # Auto-close should have called close_position
        mock_alpaca.close_position.assert_called_with('OPRA')

    def test_fresh_same_day_entry_not_closed_in_rth(
        self, patched_orphan_engine, mock_alpaca, monkeypatch,
    ):
        """Same-day fresh entry must NOT be auto-closed mid-RTH.

        2026-06-05: the old time-of-day gate was replaced by a predicate-
        driven safety: the STALE-signal check only matches cross-day rows,
        rows in 'exit_pending_verification' state, or rows already tagged
        with a known unconfirmed exit_reason. A same-day filled row with
        no exit is the engine's normal active position — not an orphan.

        2026-06-06: pin trade_date to TODAY's real wall-clock date.
        The reconciler reads `today_et = datetime.now(timezone.utc).date()`
        from its own module (not from orb_engine's mocked datetime),
        so we use the real date here to keep "same day" stable across
        runs.
        """
        from datetime import date as _date
        today_str = _date.today().isoformat()
        # Override the cross-day stale row from the fixture with a fresh
        # same-day filled row (no exit, no stale signal).
        patched_orphan_engine._reconciler_rows = [{
            'id': 200, 'trade_date': today_str, 'symbol': 'OPRA',
            'strategy': 'orb', 'fill_price': 10.0, 'filled_qty': 100,
            'exit_price': None, 'exit_reason': None,
            'order_status': 'filled',
        }]
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo
        et_tz = ZoneInfo('America/New_York')
        fake_now = _dt(2026, 6, 5, 10, 30, 0, tzinfo=et_tz)

        import trading.orb_engine as orb_mod
        real_dt = orb_mod.datetime
        class MockDT:
            @staticmethod
            def now(tz=None):
                if tz is not None:
                    return fake_now.astimezone(tz)
                return fake_now.replace(tzinfo=None)
            def __getattr__(self, name):
                return getattr(real_dt, name)
        monkeypatch.setattr(orb_mod, 'datetime', MockDT())

        from trading.orphan_reconciler import reset_state_for_tests
        reset_state_for_tests()

        patched_orphan_engine.sync_positions()

        # Fresh same-day entry — predicate refuses → no close.
        mock_alpaca.close_position.assert_not_called()

    def test_orphan_auto_closes_post_close(
        self, patched_orphan_engine, mock_alpaca, monkeypatch,
    ):
        """Post-close startup (17:00 ET) → ORB-owned orphan auto-closed."""
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo
        et_tz = ZoneInfo('America/New_York')
        fake_now = _dt(2026, 6, 5, 17, 0, 0, tzinfo=et_tz)  # 5 PM ET = after-hours

        import trading.orb_engine as orb_mod
        real_dt = orb_mod.datetime
        class MockDT:
            @staticmethod
            def now(tz=None):
                if tz is not None:
                    return fake_now.astimezone(tz)
                return fake_now.replace(tzinfo=None)
            def __getattr__(self, name):
                return getattr(real_dt, name)
        monkeypatch.setattr(orb_mod, 'datetime', MockDT())

        from trading.orphan_reconciler import reset_state_for_tests
        reset_state_for_tests()

        patched_orphan_engine.sync_positions()
        mock_alpaca.close_position.assert_called_with('OPRA')

    def test_non_orb_orphan_not_auto_closed(
        self, patched_orphan_engine, mock_alpaca, monkeypatch,
    ):
        """Regression (2026-05-22): an orphan ORB does NOT own (e.g., a
        divergence CLF short on the shared account) must stay alert-only —
        never auto-closed. The CRITICAL orphan alert still fires.

        2026-06-05: enforced now by the orphan_reconciler's predicate —
        no strategy='orb' DB row in the lookback → FOREIGN → no close.
        """
        # Empty rows = no orb DB row matches → FOREIGN classification.
        patched_orphan_engine._reconciler_rows = []
        patched_orphan_engine._notify_error = MagicMock()
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo
        et_tz = ZoneInfo('America/New_York')
        fake_now = _dt(2026, 6, 5, 17, 0, 0, tzinfo=et_tz)

        import trading.orb_engine as orb_mod
        real_dt = orb_mod.datetime
        class MockDT:
            @staticmethod
            def now(tz=None):
                if tz is not None:
                    return fake_now.astimezone(tz)
                return fake_now.replace(tzinfo=None)
            def __getattr__(self, name):
                return getattr(real_dt, name)
        monkeypatch.setattr(orb_mod, 'datetime', MockDT())

        from trading.orphan_reconciler import reset_state_for_tests
        reset_state_for_tests()

        patched_orphan_engine.sync_positions()

        # Not closed...
        mock_alpaca.close_position.assert_not_called()
        # ...but the reconciler still alerted (FOREIGN classification).
        # 2026-06-05: the pre-existing engine-level _notify_error
        # ("ORPHAN ALPACA POSITIONS") was removed because it had no
        # cooldown and produced alert storms on stuck orphans. The
        # reconciler's per-orphan FOREIGN alert (with 60-min cooldown)
        # replaces it.
        notifier = patched_orphan_engine.notifier
        if notifier is not None and hasattr(notifier, 'notify_error'):
            assert any(
                ('ORPHAN' in str(c)) or ('FOREIGN' in str(c))
                for c in notifier.notify_error.call_args_list
            ), "Reconciler must alert on foreign positions"


# =========================================================================
# _orb_owned_symbols — strategy-scoped position ownership
# =========================================================================

class TestOrbOwnedSymbols:
    """The helper that decides which Alpaca positions ORB may close."""

    def test_returns_open_orb_symbols(self, engine, mock_db):
        mock_db.get_open_trades.return_value = [
            {'symbol': 'AAA', 'strategy': 'orb'},
            {'symbol': 'BBB', 'strategy': 'orb'},
        ]
        assert engine._orb_owned_symbols() == {'AAA', 'BBB'}

    def test_queries_orb_strategy_only(self, engine, mock_db):
        from unittest.mock import ANY
        mock_db.get_open_trades.return_value = []
        engine._orb_owned_symbols()
        mock_db.get_open_trades.assert_called_with(ANY, strategy='orb')

    def test_lookback_scans_today_plus_prior_days(self, engine, mock_db):
        mock_db.get_open_trades.return_value = []
        engine._orb_owned_symbols(lookback_days=4)
        assert mock_db.get_open_trades.call_count == 5  # today + 4 prior

    def test_lookback_zero_is_today_only(self, engine, mock_db):
        mock_db.get_open_trades.return_value = []
        engine._orb_owned_symbols(lookback_days=0)
        assert mock_db.get_open_trades.call_count == 1

    def test_empty_set_on_db_error(self, engine, mock_db):
        mock_db.get_open_trades.side_effect = RuntimeError('db locked')
        assert engine._orb_owned_symbols() == set()


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


# =========================================================================
# Feature-context daily-bar freshness guard
#
# Regression for the 2026-05-29 prod incident: prod's daily_bars cache was
# frozen at 05-22 for ORB-only gappers (ASTN/PLTG/SOFX). The old code only
# refetched when a symbol was ENTIRELY absent, so the stale prev_close fed
# the gap/feature math → wrong quintile → wrong picks (ASTN falsely
# phantom-gap-rejected; PLTG flipped Q1→Q4). The guard must refetch from
# Alpaca when the newest cached bar predates the last trading day.
# =========================================================================
class TestFeatureContextFreshness:
    def test_stale_cache_triggers_alpaca_refetch(self, engine, mock_db, mock_alpaca):
        today = datetime.now(timezone.utc).date()
        stale_date = today - timedelta(days=10)   # > _PREV_BAR_STALENESS_MAX_DAYS
        fresh_date = today - timedelta(days=1)
        # Cache holds a STALE bar (close 5.51, like prod's frozen 05-22 ASTN).
        mock_db.get_daily_bars_cached.return_value = {
            'ASTN': [{'date': stale_date.strftime('%Y-%m-%d'),
                      'open': 6.0, 'high': 6.1, 'low': 5.2, 'close': 5.51,
                      'volume': 4_000_000}]
        }
        # Alpaca has the real recent bar (close 3.20, like dev's fresh 05-28).
        mock_alpaca.get_daily_bars_range.return_value = {
            'ASTN': [{'date': fresh_date, 'open': 3.5, 'high': 3.8, 'low': 3.1,
                      'close': 3.20, 'volume': 11_000_000}]
        }
        ctx = engine._get_feature_context('ASTN')
        mock_alpaca.get_daily_bars_range.assert_called_once()
        # prev_day_bar must come from the FRESH refetch, not the stale cache.
        assert ctx['prev_day_bar']['close'] == pytest.approx(3.20)

    def test_fresh_cache_uses_cache_without_refetch(self, engine, mock_db, mock_alpaca):
        today = datetime.now(timezone.utc).date()
        fresh_date = today - timedelta(days=1)
        mock_db.get_daily_bars_cached.return_value = {
            'BAR': [{'date': fresh_date.strftime('%Y-%m-%d'),
                     'open': 3.1, 'high': 3.3, 'low': 3.0, 'close': 3.20,
                     'volume': 9_000_000}]
        }
        ctx = engine._get_feature_context('BAR')
        mock_alpaca.get_daily_bars_range.assert_not_called()
        assert ctx['prev_day_bar']['close'] == pytest.approx(3.20)

    def test_absent_cache_falls_back_to_alpaca(self, engine, mock_db, mock_alpaca):
        today = datetime.now(timezone.utc).date()
        mock_db.get_daily_bars_cached.return_value = {}   # symbol absent
        mock_alpaca.get_daily_bars_range.return_value = {
            'NEW': [{'date': today - timedelta(days=1), 'open': 4.0, 'high': 4.2,
                     'low': 3.9, 'close': 4.10, 'volume': 2_000_000}]
        }
        ctx = engine._get_feature_context('NEW')
        mock_alpaca.get_daily_bars_range.assert_called_once()
        assert ctx['prev_day_bar']['close'] == pytest.approx(4.10)

    def test_stale_refetch_failure_falls_back_to_stale_cache(self, engine, mock_db, mock_alpaca):
        # Alpaca refetch raises on the stale path → must keep the stale cache
        # as a logged last resort, never crash the scoring loop.
        today = datetime.now(timezone.utc).date()
        stale_date = today - timedelta(days=10)
        mock_db.get_daily_bars_cached.return_value = {
            'FOO': [{'date': stale_date.strftime('%Y-%m-%d'),
                     'open': 6.0, 'high': 6.1, 'low': 5.2, 'close': 5.51,
                     'volume': 4_000_000}]
        }
        mock_alpaca.get_daily_bars_range.side_effect = RuntimeError("alpaca down")
        ctx = engine._get_feature_context('FOO')
        mock_alpaca.get_daily_bars_range.assert_called_once()
        # refetch failed → fall back to the stale cache rather than blow up
        assert ctx['prev_day_bar']['close'] == pytest.approx(5.51)


# =========================================================================
# Touchgo breakout-bar re-keying (2026-06-04 fix)
#
# Live previously keyed touchgo Rule M/D to the minute of the actual fill.
# When a stop-limit fill lagged the market breakout, that evaluated the wrong
# bar and diverged from BT (~23% of live fills, all flipping the tag_bb
# decision). The fix re-keys to the MARKET breakout bar (first high>range_high)
# captured during the pending phase, plus a late-fill staleness guard.
# =========================================================================

import dataclasses

_DATE = '2026-04-20'


def _weak_breakout_bars():
    """Range 13:30-13:34 (range_high=10.0); a WEAK breakout bar at 13:35
    (closes near its low -> Rule M should fire), then STRONG later bars so the
    fill bar (13:39) would NOT fire under the legacy fill-bar policy."""
    return _make_bars([
        (13, 30, 9.80, 10.00, 9.70, 9.90, 1000),
        (13, 31, 9.90, 9.95, 9.80, 9.90, 1000),
        (13, 32, 9.90, 9.90, 9.80, 9.85, 1000),
        (13, 33, 9.85, 9.95, 9.80, 9.90, 1000),
        (13, 34, 9.90, 10.00, 9.85, 9.95, 1000),
        (13, 35, 10.00, 10.30, 9.90, 9.95, 5000),  # breakout, WEAK close (pos=0.125)
        (13, 36, 9.95, 10.20, 9.92, 10.10, 4000),
        (13, 37, 10.10, 10.40, 10.05, 10.35, 4000),
        (13, 38, 10.35, 10.50, 10.20, 10.40, 4000),
        (13, 39, 10.40, 10.50, 10.00, 10.45, 4000),  # fill bar, STRONG close (pos=0.9)
    ], date_str=_DATE)


def _pending_pos(sym='BRK', order_id='pending'):
    return OpenPosition(
        symbol=sym, entry_price=10.03, stop_price=9.0, shares=100,
        trade_id=1, order_id=order_id,
        entry_time=pd.Timestamp(f'{_DATE} 13:35:00', tz='UTC'),
        range_high=10.0, range_low=9.0, lock_arm_at_r=1.75, lock_stop_r=0.5,
        composite_score=0.5, quintile='Q4',
    )


class TestTouchgoBreakoutBarReKey:
    def test_captures_market_breakout_bar_not_fill_minute(self, engine):
        """_ensure_breakout_bar_ts sets the first-high>range_high bar (13:35),
        independent of when the fill lands."""
        assert engine.touchgo_cfg.breakout_bar_source == 'market'
        sym = 'BRK'
        pos = _pending_pos(sym)
        engine.open_positions[sym] = pos
        bars = _weak_breakout_bars()
        engine._ingest_bars(sym, bars)
        assert pos.breakout_bar_ts == pd.Timestamp(f'{_DATE} 13:35:00', tz='UTC')

    def test_rule_m_fires_on_market_breakout_bar_despite_strong_fill_bar(
            self, engine, mock_stop_monitor):
        """Capture during pending (bb=13:35 weak), fill late at 13:39 (strong
        bar). Market mode evaluates 13:35 -> Rule M fires. This is the bug fix:
        legacy fill-bar policy would have evaluated 13:39 and NOT fired."""
        sym = 'BRK'
        pos = _pending_pos(sym)
        engine.open_positions[sym] = pos
        bars = _weak_breakout_bars()
        # Pending phase: capture the breakout bar.
        engine._ingest_bars(sym, bars)
        assert pos.breakout_bar_ts == pd.Timestamp(f'{_DATE} 13:35:00', tz='UTC')
        # Fill at 13:39 (4 min after breakout — within the 15-min guard).
        pos.order_id = ''
        pos.entry_time = pd.Timestamp(f'{_DATE} 13:39:00', tz='UTC')
        pos.rule_m_evaluated = False
        pos.rule_d_evaluated = False
        engine._evaluate_touchgo(sym, bars)
        assert mock_stop_monitor.force_exit.called
        kwargs = mock_stop_monitor.force_exit.call_args.kwargs
        assert kwargs['reason'] == 'tag_bb'

    def test_late_fill_guard_skips_touchgo(self, engine, mock_stop_monitor):
        """A fill that lags the breakout bar by more than max_breakout_age_min
        is a stale entry — touchgo must not fire a retroactive exit."""
        sym = 'BRK'
        pos = _pending_pos(sym)
        engine.open_positions[sym] = pos
        bars = _weak_breakout_bars()
        engine._ingest_bars(sym, bars)            # bb = 13:35
        pos.order_id = ''
        pos.entry_time = pd.Timestamp(f'{_DATE} 13:56:00', tz='UTC')  # 21 min late
        pos.rule_m_evaluated = False
        pos.rule_d_evaluated = False
        engine._evaluate_touchgo(sym, bars)
        assert not mock_stop_monitor.force_exit.called
        assert pos.rule_m_evaluated and pos.rule_d_evaluated  # guard marked done

    def test_no_capture_when_session_open_absent_rehydration_guard(
            self, engine, mock_stop_monitor):
        """Regression: a position rehydrated after a mid-session restart sees a
        fresh bars window with no 9:30 anchor. _ensure must DECLINE (leave
        breakout_bar_ts None) rather than match a random afternoon bar trading
        above range_high — otherwise touchgo could fire a spurious exit on a
        position that's been open since the morning."""
        sym = 'BRK'
        # Afternoon-only window (14:00-14:02 ET = 18:00 UTC): no 9:30 bar, every
        # bar trades above range_high (10.0), first bar closes weak.
        bars = _make_bars([
            (18, 0, 10.50, 10.80, 10.40, 10.45, 4000),  # high>10, weak-ish close
            (18, 1, 10.45, 10.90, 10.40, 10.85, 4000),
            (18, 2, 10.85, 11.00, 10.70, 10.95, 4000),
        ], date_str=_DATE)
        pos = _pending_pos(sym)
        pos.order_id = ''  # filled (rehydrated)
        pos.entry_time = pd.Timestamp(f'{_DATE} 13:35:00', tz='UTC')  # morning fill
        engine.open_positions[sym] = pos
        engine._ingest_bars(sym, bars)
        # No 9:30 anchor -> no capture -> touchgo stays inert.
        assert pos.breakout_bar_ts is None
        engine._evaluate_touchgo(sym, bars)
        assert not mock_stop_monitor.force_exit.called

    def test_legacy_fill_mode_no_capture_and_evaluates_fill_bar(
            self, engine, mock_stop_monitor):
        """breakout_bar_source='fill' restores legacy behaviour: _ensure does
        not capture, and Rule M evaluates the fill-minute bar (13:39, strong) so
        it does NOT fire."""
        engine.touchgo_cfg = dataclasses.replace(
            engine.touchgo_cfg, breakout_bar_source='fill')
        sym = 'BRK'
        pos = _pending_pos(sym)
        engine.open_positions[sym] = pos
        bars = _weak_breakout_bars()
        engine._ingest_bars(sym, bars)
        # _ensure is a no-op in fill mode.
        assert pos.breakout_bar_ts is None
        # Simulate the legacy fill handler keying to the fill minute (13:39).
        pos.order_id = ''
        pos.entry_time = pd.Timestamp(f'{_DATE} 13:39:00', tz='UTC')
        pos.breakout_bar_ts = pd.Timestamp(f'{_DATE} 13:39:00', tz='UTC')
        pos.rule_m_evaluated = False
        pos.rule_d_evaluated = False
        engine._evaluate_touchgo(sym, bars)
        # 13:39 bar closed strong (pos 0.9) -> no fire (the divergence the fix removes).
        assert not mock_stop_monitor.force_exit.called
