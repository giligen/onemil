"""Unit + integration tests for the ORB add-on pool gate extension
(PREREG_LIVE_UNION.md, gate extension owner GO 2026-10-01,
trading/orb_addon_gates.py).

Style reference: tests/test_orb_addon_pools.py (engine fixtures) and
tests/test_orb_engine.py.
"""
import csv
import json
import logging

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, RangeData, CandidateState
from trading.stop_monitor import StopMonitor
from trading.orb_addon_gates import (
    PoolGateInputs, evaluate_pool_gates, warn_unknown_pool_keys, KNOWN_POOL_KEYS,
)
from tests.conftest import load_orb_yaml_pinned


# =========================================================================
# Pure unit tests — evaluate_pool_gates / warn_unknown_pool_keys. No engine,
# no I/O: known-answer on synthetic PoolGateInputs (the "synthetic bars/
# daily rows" the values would have been derived from in production).
# =========================================================================

class TestNoGatesConfigured:
    def test_byte_identical_admits_with_empty_values(self):
        """A pool dict with none of the gate keys set must admit everything
        and report no gate values — the pre-extension behaviour exactly."""
        pool_cfg = {'name': 'addon_gap4', 'min_gap_pct': 4.0, 'max_gap_pct': 5.0,
                    'min_price': 3.0, 'max_price': 30.0}
        inputs = PoolGateInputs(move_to_range_high_pct=0.001, rel_volume_0935=0.001,
                                 premarket_dollar_vol=0.0, above_vwap_0935=False,
                                 range_close_top_half=False, dist_to_52wk_high_pct=999.0,
                                 prev_day_range_atr=0.0, is_day2_gapper=False)
        admitted, gate_values = evaluate_pool_gates(pool_cfg, inputs)
        assert admitted is True
        assert gate_values == {}

    def test_all_none_inputs_still_admits_when_ungated(self):
        pool_cfg = {'name': 'p'}
        admitted, gate_values = evaluate_pool_gates(pool_cfg, PoolGateInputs())
        assert admitted is True
        assert gate_values == {}


class TestMinMoveToRangeHighPct:
    def test_pass(self):
        pool_cfg = {'name': 'p', 'min_move_to_range_high_pct': 5.0}
        admitted, gv = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(move_to_range_high_pct=10.5))
        assert admitted is True
        assert gv == {'move_to_range_high_pct': 10.5}

    def test_fail(self):
        pool_cfg = {'name': 'p', 'min_move_to_range_high_pct': 5.0}
        admitted, gv = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(move_to_range_high_pct=3.0))
        assert admitted is False
        assert gv == {'move_to_range_high_pct': 3.0}


class TestMinRelVolume0935:
    def test_pass(self):
        pool_cfg = {'name': 'p', 'min_rel_volume_0935': 1.5}
        admitted, _ = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(rel_volume_0935=2.0))
        assert admitted is True

    def test_fail(self):
        pool_cfg = {'name': 'p', 'min_rel_volume_0935': 1.5}
        admitted, _ = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(rel_volume_0935=1.0))
        assert admitted is False


class TestMinPremarketDollarVol:
    def test_pass(self):
        pool_cfg = {'name': 'p', 'min_premarket_dollar_vol': 5_000_000}
        admitted, _ = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(premarket_dollar_vol=6_000_000))
        assert admitted is True

    def test_fail(self):
        pool_cfg = {'name': 'p', 'min_premarket_dollar_vol': 5_000_000}
        admitted, _ = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(premarket_dollar_vol=1_000_000))
        assert admitted is False


class TestRequireAboveVwap0935:
    def test_pass_both_true(self):
        pool_cfg = {'name': 'p', 'require_above_vwap_0935': True}
        admitted, gv = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(above_vwap_0935=True, range_close_top_half=True))
        assert admitted is True
        assert gv == {'above_vwap_0935': True, 'range_close_top_half': True}

    def test_fail_above_vwap_false(self):
        pool_cfg = {'name': 'p', 'require_above_vwap_0935': True}
        admitted, _ = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(above_vwap_0935=False, range_close_top_half=True))
        assert admitted is False

    def test_fail_top_half_false(self):
        pool_cfg = {'name': 'p', 'require_above_vwap_0935': True}
        admitted, _ = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(above_vwap_0935=True, range_close_top_half=False))
        assert admitted is False


class TestMaxDistTo52wkHighPct:
    def test_pass_near_the_high(self):
        pool_cfg = {'name': 'p', 'max_dist_to_52wk_high_pct': 5.0}
        admitted, _ = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(dist_to_52wk_high_pct=2.0))
        assert admitted is True

    def test_fail_too_far_below_the_high(self):
        pool_cfg = {'name': 'p', 'max_dist_to_52wk_high_pct': 5.0}
        admitted, _ = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(dist_to_52wk_high_pct=20.0))
        assert admitted is False

    def test_pass_new_52wk_high_is_negative_distance(self):
        pool_cfg = {'name': 'p', 'max_dist_to_52wk_high_pct': 5.0}
        admitted, _ = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(dist_to_52wk_high_pct=-1.0))
        assert admitted is True


class TestMinPrevDayRangeAtr:
    def test_pass(self):
        pool_cfg = {'name': 'p', 'min_prev_day_range_atr': 1.0}
        admitted, _ = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(prev_day_range_atr=1.5))
        assert admitted is True

    def test_fail(self):
        pool_cfg = {'name': 'p', 'min_prev_day_range_atr': 1.0}
        admitted, _ = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(prev_day_range_atr=0.5))
        assert admitted is False


class TestRequireDay2Gapper:
    def test_pass(self):
        pool_cfg = {'name': 'p', 'require_day2_gapper': True}
        admitted, gv = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(is_day2_gapper=True))
        assert admitted is True
        assert gv == {'is_day2_gapper': True}

    def test_fail(self):
        pool_cfg = {'name': 'p', 'require_day2_gapper': True}
        admitted, _ = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(is_day2_gapper=False))
        assert admitted is False

    def test_fail_closed_on_unresolved_input(self):
        pool_cfg = {'name': 'p', 'require_day2_gapper': True}
        admitted, _ = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(is_day2_gapper=None))
        assert admitted is False


class TestFailsClosedOnMissingData:
    def test_configured_gate_with_none_input_rejects(self):
        """A gate the pool configures must FAIL CLOSED when its input could
        not be resolved — never silently admit on missing data."""
        pool_cfg = {'name': 'p', 'min_move_to_range_high_pct': 5.0}
        admitted, gv = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(move_to_range_high_pct=None))
        assert admitted is False
        assert gv == {'move_to_range_high_pct': None}


class TestMultipleGatesAllMustPass:
    def test_one_failing_gate_rejects_even_if_others_pass(self):
        pool_cfg = {'name': 'p', 'min_move_to_range_high_pct': 5.0,
                    'min_prev_day_range_atr': 1.0}
        admitted, gv = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(move_to_range_high_pct=10.0,  # passes
                                      prev_day_range_atr=0.2))      # fails
        assert admitted is False
        assert gv == {'move_to_range_high_pct': 10.0, 'prev_day_range_atr': 0.2}

    def test_all_passing_gates_admits(self):
        pool_cfg = {'name': 'p', 'min_move_to_range_high_pct': 5.0,
                    'min_prev_day_range_atr': 1.0}
        admitted, gv = evaluate_pool_gates(
            pool_cfg, PoolGateInputs(move_to_range_high_pct=10.0,
                                      prev_day_range_atr=1.2))
        assert admitted is True
        assert gv == {'move_to_range_high_pct': 10.0, 'prev_day_range_atr': 1.2}


class TestWarnUnknownPoolKeys:
    def test_typo_key_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            warn_unknown_pool_keys({'name': 'p', 'min_gap_pct': 1.0,
                                     'min_move_to_range_hi_pct': 5.0})  # typo'd
        assert 'unrecognized config key' in caplog.text
        assert 'min_move_to_range_hi_pct' in caplog.text

    def test_all_known_keys_silent(self, caplog):
        with caplog.at_level(logging.WARNING):
            warn_unknown_pool_keys({k: 1.0 for k in KNOWN_POOL_KEYS if k != 'name'})
        assert 'unrecognized' not in caplog.text


# =========================================================================
# Engine-level: byte-identical (no gates) + the 09:35-cycle integration
# test (one pool admitting one name, ledger row carrying pool + gates).
# =========================================================================

def _base_cfg():
    return load_orb_yaml_pinned()


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


def _engine_for_pool(pool_cfg, addon_dry_run=True):
    cfg = _base_cfg()
    cfg['universe']['addon_pools'] = {
        'enabled': True, 'dry_run': addon_dry_run, 'pools': [pool_cfg],
    }
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


def _seed(eng, symbol, pool):
    eng.universe.add(symbol)
    eng.candidates[symbol] = CandidateState(symbol=symbol)
    eng.candidates[symbol].range_data = _range(symbol)
    eng._symbol_pool[symbol] = pool


class TestByteIdenticalWithoutGates:
    def test_pool_without_gate_keys_admits_exactly_like_before(self):
        """A pool dict carrying only the pre-extension membership keys must
        still reach the dry-run branch (never 'addon_gate_reject') — the
        gate hook is a no-op for it."""
        pool_cfg = {'name': 'addon_gap4', 'min_gap_pct': 4.0, 'max_gap_pct': 5.0,
                    'min_price': 3.0, 'max_price': 30.0}
        eng, a, db = _engine_for_pool(pool_cfg, addon_dry_run=True)
        _seed(eng, 'ADDONSYM', 'addon_gap4')
        fp = {'ADDONSYM': {'prev_day_bar': {}, 'daily_stats_20d': {}}}
        with _disable_gates(eng), \
                patch('trading.orb_engine.composite_score', return_value=1.0), \
                patch('trading.orb_engine.assign_quintile', return_value='Q5'):
            submitted = eng.check_entries(feature_providers=fp)
        assert submitted == []
        assert eng.candidates['ADDONSYM'].rejected_reason == 'addon_dry_run'
        # Only the always-present telemetry key; no gate value leaked in.
        assert eng.candidates['ADDONSYM'].pool_gate_values == {
            'min_prev_volume_used': eng.universe_min_prev_volume}


class TestGateIntegration0935Cycle:
    def test_gate_admits_one_name_ledger_row_carries_pool_and_gates(
            self, tmp_path, monkeypatch):
        """Full check_entries cycle: one pool, one gate configured
        (min_move_to_range_high_pct), one candidate whose move clears it —
        admitted into the dry branch, and the dry ledger row records the
        pool id and the resolved gate values that admitted the name."""
        import trading.orb_engine as orb_engine_mod
        ledger_path = tmp_path / 'orb_dry_ledger.csv'
        monkeypatch.setattr(orb_engine_mod, 'DEFAULT_DRY_LEDGER_PATH', ledger_path)

        pool_cfg = {'name': 'gated_pool', 'min_gap_pct': 3.0, 'max_gap_pct': 5.0,
                    'min_price': 3.0, 'max_price': 30.0,
                    'min_move_to_range_high_pct': 5.0}
        eng, a, db = _engine_for_pool(pool_cfg, addon_dry_run=True)
        _seed(eng, 'GATEDSYM', 'gated_pool')
        # range_high=10.5 (from _range default); prev_close=9.5 ->
        # move_to_range_high_pct = (10.5-9.5)/9.5*100 = 10.526% >= 5.0 -> admits.
        fp = {'GATEDSYM': {'prev_day_bar': {'close': 9.5, 'high': 9.6, 'low': 9.4,
                                             'volume': 500_000},
                            'daily_stats_20d': {}}}
        with _disable_gates(eng), \
                patch('trading.orb_engine.composite_score', return_value=1.0), \
                patch('trading.orb_engine.assign_quintile', return_value='Q5'):
            submitted = eng.check_entries(feature_providers=fp)

        assert submitted == []  # dry pool -> no real order
        cand = eng.candidates['GATEDSYM']
        assert cand.rejected_reason == 'addon_dry_run'
        assert cand.pool_gate_values['move_to_range_high_pct'] == pytest.approx(
            10.526, abs=0.01)

        assert ledger_path.exists()
        rows = list(csv.DictReader(open(ledger_path)))
        assert len(rows) == 1
        assert rows[0]['symbol'] == 'GATEDSYM'
        assert rows[0]['pool'] == 'gated_pool'
        gate_values = json.loads(rows[0]['pool_gates'])
        assert gate_values['move_to_range_high_pct'] == pytest.approx(10.526, abs=0.01)
        assert gate_values['min_prev_volume_used'] == eng.universe_min_prev_volume

    def test_gate_rejects_candidate_that_fails_it(self, tmp_path, monkeypatch):
        import trading.orb_engine as orb_engine_mod
        ledger_path = tmp_path / 'orb_dry_ledger.csv'
        monkeypatch.setattr(orb_engine_mod, 'DEFAULT_DRY_LEDGER_PATH', ledger_path)

        pool_cfg = {'name': 'gated_pool', 'min_gap_pct': 3.0, 'max_gap_pct': 5.0,
                    'min_price': 3.0, 'max_price': 30.0,
                    'min_move_to_range_high_pct': 50.0}  # unreachable by this candidate
        eng, a, db = _engine_for_pool(pool_cfg, addon_dry_run=True)
        _seed(eng, 'GATEDSYM', 'gated_pool')
        fp = {'GATEDSYM': {'prev_day_bar': {'close': 9.5, 'high': 9.6, 'low': 9.4,
                                             'volume': 500_000},
                            'daily_stats_20d': {}}}
        with _disable_gates(eng), \
                patch('trading.orb_engine.composite_score', return_value=1.0), \
                patch('trading.orb_engine.assign_quintile', return_value='Q5'):
            submitted = eng.check_entries(feature_providers=fp)

        assert submitted == []
        assert eng.candidates['GATEDSYM'].rejected_reason == 'addon_gate_reject'
        # Rejected before the dry-ledger write -> no row for this symbol.
        if ledger_path.exists():
            rows = list(csv.DictReader(open(ledger_path)))
            assert all(r['symbol'] != 'GATEDSYM' for r in rows)
