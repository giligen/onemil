"""Regression for the Q1 composite-quintile filter (shipped 2026-04-24).

Research (study_orb_q1q2_filter.py) showed Q1 candidates were TRAIN-positive
(+$6K on H1 2025) but OOS-negative on both VAL (-$5.1K) and HOQ1+ (-$3.4K).
BT lift from filtering: +$8,556 combined OOS. Filter defaults ON via
`orb.yaml::filter.skip_q1: true`.

These tests verify:
  1. Config flag loads correctly (default on, overridable)
  2. Q1 candidates are dropped + marked with rejected_reason='q1_filter'
  3. Non-Q1 candidates (Q2-Q5) are not affected by the filter
  4. Filter off → Q1 candidates are not filtered (still subject to normal ranking)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, RangeData
from trading.stop_monitor import StopMonitor


# =========================================================================
# Fixtures (mirror tests/test_orb_engine.py patterns)
# =========================================================================

@pytest.fixture
def orb_cfg():
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


def _make_engine(cfg, mock_alpaca, mock_db, mock_stop_monitor):
    cfg = yaml.safe_load(yaml.safe_dump(cfg))  # deep copy
    cfg['strategy']['enabled'] = True
    return ORBEngine(
        alpaca_client=mock_alpaca, db=mock_db,
        stop_monitor=mock_stop_monitor, config=cfg,
    )


def _seed_candidate(engine, sym: str):
    """Attach range_data to a candidate so it becomes eligible for scoring."""
    engine.candidates[sym].range_data = RangeData(
        symbol=sym, range_high=10.5, range_low=10.0, range_volume=500_000,
        range_avg_bar_range_pct=1.0, range_close=10.4,
        range_start_ts=pd.Timestamp.utcnow(),
    )


# =========================================================================
# Config loading
# =========================================================================

class TestConfigFlag:

    def test_skip_q1_default_true_in_shipped_yaml(self, orb_cfg,
                                                    mock_alpaca, mock_db, mock_stop_monitor):
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor)
        assert eng.skip_q1 is True

    def test_skip_q1_can_be_disabled(self, orb_cfg,
                                      mock_alpaca, mock_db, mock_stop_monitor):
        cfg = yaml.safe_load(yaml.safe_dump(orb_cfg))
        cfg['filter']['skip_q1'] = False
        eng = _make_engine(cfg, mock_alpaca, mock_db, mock_stop_monitor)
        assert eng.skip_q1 is False

    def test_skip_q1_defaults_true_if_key_missing(self, orb_cfg,
                                                    mock_alpaca, mock_db, mock_stop_monitor):
        """Backwards-compat: if older orb.yaml lacks skip_q1, default ON."""
        cfg = yaml.safe_load(yaml.safe_dump(orb_cfg))
        cfg['filter'].pop('skip_q1', None)
        eng = _make_engine(cfg, mock_alpaca, mock_db, mock_stop_monitor)
        assert eng.skip_q1 is True


# =========================================================================
# Filter behavior via check_entries
# =========================================================================

class TestQ1FilterBehavior:
    """We drive 5 eligible candidates through check_entries with mocked
    composite_score so each lands in a known quintile (Q1..Q5). We then
    verify the filter drops Q1 and sets rejected_reason='q1_filter'.

    The 4 real yaml cutoffs are [0.1082, 0.1959, 0.2893, 0.4081]. We pick
    scores that map cleanly to each quintile:
        0.05 -> Q1, 0.15 -> Q2, 0.25 -> Q3, 0.35 -> Q4, 0.50 -> Q5.
    """

    SCORES_BY_QUINTILE = {
        'Q1': 0.05, 'Q2': 0.15, 'Q3': 0.25, 'Q4': 0.35, 'Q5': 0.50,
    }

    def _run_check_entries(self, engine, syms_to_quintile):
        """Seed candidates + patch scoring so each sym gets its mapped quintile."""
        engine.build_universe(source_loader=lambda: list(syms_to_quintile.keys()))
        for sym in syms_to_quintile:
            _seed_candidate(engine, sym)

        # composite_score is called once per eligible candidate in insertion
        # order. Build a score queue matching that order.
        score_q = [self.SCORES_BY_QUINTILE[syms_to_quintile[s]]
                   for s in engine.candidates.keys()
                   if s in syms_to_quintile]

        from trading import orb_engine as oe
        score_iter = iter(score_q)

        def fake_score(feats, params):
            return next(score_iter)

        with patch.object(oe, 'composite_score', side_effect=fake_score):
            # Bypass wall-clock guard + spread lookup
            with patch.object(engine, '_past_last_entry_time', return_value=False):
                with patch.object(engine, '_get_spread_bps', return_value=50.0):
                    # _compute_features returns feat dict; real implementation
                    # is fine because fake_score doesn't care what it is.
                    engine.check_entries()

    def test_q1_candidate_is_filtered(self, orb_cfg,
                                        mock_alpaca, mock_db, mock_stop_monitor):
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor)
        assert eng.skip_q1 is True

        syms = {'Q1SYM': 'Q1', 'Q4SYM': 'Q4'}
        self._run_check_entries(eng, syms)

        assert eng.candidates['Q1SYM'].quintile == 'Q1'
        assert eng.candidates['Q1SYM'].rejected_reason == 'q1_filter'
        # Q4 should not be filtered by Q1 rule
        assert eng.candidates['Q4SYM'].rejected_reason != 'q1_filter'

    def test_all_quintiles_q1_rejected_others_not(self, orb_cfg,
                                                    mock_alpaca, mock_db, mock_stop_monitor):
        """Five candidates, one per quintile. Only Q1 gets 'q1_filter'."""
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor)

        syms = {f'SYM{q}': q for q in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5')}
        self._run_check_entries(eng, syms)

        assert eng.candidates['SYMQ1'].rejected_reason == 'q1_filter'
        for q in ('Q2', 'Q3', 'Q4', 'Q5'):
            reason = eng.candidates[f'SYM{q}'].rejected_reason
            assert reason != 'q1_filter', (
                f"SYM{q} was incorrectly tagged 'q1_filter' "
                f"(actual: {reason}). Only Q1 should be filtered."
            )

    def test_q1_not_filtered_when_flag_off(self, orb_cfg,
                                             mock_alpaca, mock_db, mock_stop_monitor):
        cfg = yaml.safe_load(yaml.safe_dump(orb_cfg))
        cfg['filter']['skip_q1'] = False
        eng = _make_engine(cfg, mock_alpaca, mock_db, mock_stop_monitor)
        assert eng.skip_q1 is False

        syms = {'Q1SYM': 'Q1', 'Q5SYM': 'Q5'}
        self._run_check_entries(eng, syms)

        # Q1 candidate reaches ranking (may or may not be submitted due to
        # dedup/sizing, but it is NOT filtered by our Q1 rule)
        assert eng.candidates['Q1SYM'].quintile == 'Q1'
        assert eng.candidates['Q1SYM'].rejected_reason != 'q1_filter'

    def test_no_q1_candidates_noop(self, orb_cfg,
                                     mock_alpaca, mock_db, mock_stop_monitor):
        """Filter is a no-op when no Q1 candidates are present."""
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor)

        syms = {'Q4SYM': 'Q4', 'Q5SYM': 'Q5'}
        self._run_check_entries(eng, syms)

        for sym in syms:
            reason = eng.candidates[sym].rejected_reason
            assert reason != 'q1_filter'


# =========================================================================
# BT parity — ORB_SKIP_Q1 env var on study_orb_pipeline_static_lock.py
# =========================================================================
# Covered by existing research scripts. Not a pytest fixture.
