"""Unit tests for production ORB dry-run mode (docs/orb_dry_run_spec_20260925.md,
9/25): `strategy.dry_run` runs the full pipeline but logs `[ORB DRY] WOULD BUY`
+ a ledger row at the submit call instead of submitting a real order.

Style/fixture reference: tests/test_orb_addon_pools.py (same base cfg, mock
alpaca, pool-selection-chain harness) and tests/test_orb_selection_race.py
(patch('trading.orb_engine.__file__', ...) for the ledger-write path).
"""
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, RangeData, CandidateState
from trading.stop_monitor import StopMonitor


def _base_cfg():
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    return cfg


def _mock_alpaca():
    client = MagicMock(spec=AlpacaClient)
    client.get_open_positions.return_value = []
    client.get_account_info.return_value = {'buying_power': 500_000.0}
    client.get_latest_quote.return_value = {'bid_price': 9.95, 'ask_price': 10.00}
    client.submit_stop_bracket_order.return_value = {'id': 'order-1', 'status': 'accepted'}
    client.cancel_order.return_value = True
    return client


def _range(symbol, range_open=10.0, range_high=10.5, range_low=9.9):
    return RangeData(
        symbol=symbol, range_high=range_high, range_low=range_low,
        range_volume=500_000, range_avg_bar_range_pct=1.0,
        range_close=range_high - 0.02, range_start_ts=pd.Timestamp.utcnow(),
        range_open=range_open,
    )


def _disable_gates(eng):
    return patch.multiple(
        eng,
        _past_last_entry_time=MagicMock(return_value=False),
        _kill_rails_blocked=MagicMock(return_value=False),
        _pdt_would_block=MagicMock(return_value=False),
        _daily_loss_limit_hit=MagicMock(return_value=False),
    )


def _engine(strategy_dry_run):
    cfg = _base_cfg()
    cfg['strategy']['dry_run'] = strategy_dry_run
    db = MagicMock(spec=Database)
    db.get_open_trades.return_value = []
    db.get_trades_by_date.return_value = []
    a = _mock_alpaca()
    eng = ORBEngine(alpaca_client=a, db=db,
                     stop_monitor=MagicMock(spec=StopMonitor), config=cfg)
    eng.pdr_veto_enabled = False
    eng.g1_veto_enabled = False
    eng.range_size_veto_enabled = False
    eng.catalyst_veto_enabled = False
    eng.skip_q1 = False
    eng.filter_threshold = -999.0
    return eng, a, db


def _seed(eng, symbol):
    eng.universe.add(symbol)
    eng.candidates[symbol] = CandidateState(symbol=symbol)
    eng.candidates[symbol].range_data = _range(symbol)
    eng._symbol_pool[symbol] = 'production'


class TestStrategyDryRunFlag:
    def test_defaults_false(self):
        eng, _, _ = _engine(strategy_dry_run=False)
        assert eng.strategy_dry_run is False

    def test_true_when_configured(self):
        eng, _, _ = _engine(strategy_dry_run=True)
        assert eng.strategy_dry_run is True


class TestProductionDryRunSelectionChain:
    def test_dry_run_submits_nothing_and_logs_would_buy(self, caplog, tmp_path):
        eng, a, db = _engine(strategy_dry_run=True)
        _seed(eng, 'DRYSYM')
        fp = {'DRYSYM': {'prev_day_bar': {}, 'daily_stats_20d': {}}}
        with _disable_gates(eng), \
                patch('trading.orb_engine.composite_score', return_value=1.0), \
                patch('trading.orb_engine.assign_quintile', return_value='Q5'), \
                patch('trading.orb_engine.__file__', str(tmp_path / 'trading' / 'orb_engine.py')), \
                caplog.at_level(logging.INFO):
            submitted = eng.check_entries(feature_providers=fp)
        assert submitted == []
        a.submit_stop_bracket_order.assert_not_called()
        db.save_trade.assert_not_called()
        assert 'DRYSYM' not in eng.open_positions
        assert eng.candidates['DRYSYM'].rejected_reason == 'production_dry_run'
        assert '[ORB DRY] WOULD BUY DRYSYM stop $10.50 limit $' in caplog.text
        assert 'shares' in caplog.text and 'risk $' in caplog.text

    def test_dry_run_writes_ledger_row(self, tmp_path):
        eng, a, db = _engine(strategy_dry_run=True)
        _seed(eng, 'LEDGSYM')
        fp = {'LEDGSYM': {'prev_day_bar': {}, 'daily_stats_20d': {}}}
        with _disable_gates(eng), \
                patch('trading.orb_engine.composite_score', return_value=1.0), \
                patch('trading.orb_engine.assign_quintile', return_value='Q5'), \
                patch('trading.orb_engine.__file__', str(tmp_path / 'trading' / 'orb_engine.py')):
            eng.check_entries(feature_providers=fp)
        p = tmp_path / 'logs' / 'orb_dry_ledger.csv'
        assert p.exists()
        rows = list(csv.DictReader(p.open()))
        assert len(rows) == 1
        row = rows[0]
        assert row['symbol'] == 'LEDGSYM'
        assert row['pool'] == 'production'
        assert row['quintile'] == 'Q5'
        assert float(row['shares']) > 0
        assert float(row['risk_usd']) > 0

    def test_dry_run_fires_latency_tripwire(self, tmp_path):
        eng, a, db = _engine(strategy_dry_run=True)
        _seed(eng, 'LATSYM')
        fp = {'LATSYM': {'prev_day_bar': {}, 'daily_stats_20d': {}}}
        assert eng._first_submit_latency_logged is False
        with _disable_gates(eng), \
                patch('trading.orb_engine.composite_score', return_value=1.0), \
                patch('trading.orb_engine.assign_quintile', return_value='Q5'), \
                patch('trading.orb_engine.__file__', str(tmp_path / 'trading' / 'orb_engine.py')):
            eng.check_entries(feature_providers=fp)
        # The would-be submit is the measured instant in dry mode too (spec
        # item 2) — the tripwire must have run exactly as a live submit.
        assert eng._first_submit_latency_logged is True

    def test_flag_off_takes_real_submit_path(self, tmp_path):
        """Flag off must be byte-identical to pre-dry-run behavior: real
        submit, real DB row, ledger writer never invoked."""
        eng, a, db = _engine(strategy_dry_run=False)
        _seed(eng, 'REALSYM')
        fp = {'REALSYM': {'prev_day_bar': {}, 'daily_stats_20d': {}}}
        with _disable_gates(eng), \
                patch('trading.orb_engine.composite_score', return_value=1.0), \
                patch('trading.orb_engine.assign_quintile', return_value='Q5'), \
                patch.object(eng, '_append_dry_ledger_row') as mocked:
            submitted = eng.check_entries(feature_providers=fp)
        assert submitted == ['REALSYM']
        a.submit_stop_bracket_order.assert_called_once()
        assert eng.candidates['REALSYM'].rejected_reason is None
        # Flag-off path must never touch the dry ledger writer.
        mocked.assert_not_called()


class TestDryRunParityObserverParsing:
    """scripts/orb_selection_observer.py::_live_submitted_symbols — a dry
    line must parse into the same pick (symbol) as a live submit line."""

    def test_dry_and_live_lines_yield_same_pick_shape(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.orb_selection_observer import _live_submitted_symbols

        submit_line = ("ORB ENTRY SUBMITTED: REALSYM order=abc123 qty=100 "
                        "entry=$10.53 stop=$9.90 trade_id=1")
        dry_line = ("[ORB DRY] WOULD BUY DRYSYM stop $10.50 limit $10.53 "
                     "shares 100 risk $53.00 | quote 9.95/10.00 at "
                     "09:35:03.120 ET")
        addon_dry_line = ("[ORB+ DRY] WOULD BUY ADDONSYM pool=addon_gap4 "
                           "qty=50 @ stop-limit $5.10 (stop=$4.90, Q5 "
                           "comp=+1.00)")
        picks = _live_submitted_symbols([submit_line], [dry_line])
        assert picks == ['DRYSYM', 'REALSYM']
        # Addon dry lines are never passed in by main() (filtered on the
        # literal '[ORB DRY] WOULD BUY' prefix before this function runs);
        # confirm this function's own contract stays symbol-list-shaped.
        picks_only_dry = _live_submitted_symbols([], [dry_line])
        assert picks_only_dry == ['DRYSYM']
        assert 'ADDONSYM' not in picks_only_dry
        assert addon_dry_line  # sanity: fixture constructed, unused by design
