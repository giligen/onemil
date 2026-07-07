"""PDR veto (prev-day-range) — 2026-07-04 weekend-program ship (W2).

Vetoes ORB picks whose PREVIOUS day range_pct <= 8.0 ("trade day-2 of the
fireworks, not day-1"). Applied POST-ranking with NO refill: a vetoed
pick's slot stays empty. Replica evidence Jan'25–Jul'26 @ thr 8.0:
TOT $155K→$210K (+35%), MDD −$29.3K→−$20.1K, WR 35.8→40.2%, all 3 eras
positive, all top-10 giants kept, monotone across thresholds 6–10. The
REFILL form is toxic (2025H2 →$0, MDD −$50K) and must never be built.

Covers: shared helper formula (BT parity vs study_orb_features.py:287),
engine config/env knobs, feature plumbing, the `_pdr_veto_reject` seam,
and no-backfill wiring in check_entries' submit loop.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, RangeData, CandidateState
from trading.orb_pdr_veto import (
    DEFAULT_MIN_PDR_PCT, compute_prev_day_range_pct, pdr_veto_applies,
)
from trading.stop_monitor import StopMonitor


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

class TestComputePrevDayRangePct:
    def test_bt_parity_formula(self):
        """Must equal study_orb_features.py:287:
        (prev_high - prev_low) / prev_close * 100."""
        assert compute_prev_day_range_pct(11.0, 9.0, 10.0) == pytest.approx(20.0)

    def test_quiet_day(self):
        # 2026-06-30 AVEX-class quiet prev day
        assert compute_prev_day_range_pct(10.3, 9.9, 10.2) == pytest.approx(
            (10.3 - 9.9) / 10.2 * 100)

    def test_zero_close_returns_none(self):
        assert compute_prev_day_range_pct(11.0, 9.0, 0.0) is None

    def test_negative_close_returns_none(self):
        assert compute_prev_day_range_pct(11.0, 9.0, -1.0) is None

    def test_inverted_high_low_returns_none(self):
        assert compute_prev_day_range_pct(9.0, 11.0, 10.0) is None

    def test_non_numeric_returns_none(self):
        assert compute_prev_day_range_pct(None, 9.0, 10.0) is None
        assert compute_prev_day_range_pct('x', 9.0, 10.0) is None

    def test_zero_range_is_valid_zero(self):
        """A flat prev day (high == low) is a REAL 0% range — the quietest
        possible day, prime veto territory — not an error."""
        assert compute_prev_day_range_pct(10.0, 10.0, 10.0) == pytest.approx(0.0)


class TestPdrVetoApplies:
    def test_below_threshold_vetoes(self):
        assert pdr_veto_applies(5.2, 8.0) is True

    def test_at_threshold_vetoes(self):
        """Boundary is <= (matches the BT search's `<= thr` mask)."""
        assert pdr_veto_applies(8.0, 8.0) is True

    def test_above_threshold_passes(self):
        assert pdr_veto_applies(8.01, 8.0) is False

    def test_none_never_vetoes(self):
        assert pdr_veto_applies(None, 8.0) is False

    def test_default_threshold_is_8(self):
        assert DEFAULT_MIN_PDR_PCT == 8.0
        assert pdr_veto_applies(7.9) is True
        assert pdr_veto_applies(8.1) is False


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------

@pytest.fixture
def orb_cfg():
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    return cfg


@pytest.fixture
def engine(orb_cfg, monkeypatch):
    monkeypatch.delenv('ORB_PDR_VETO', raising=False)
    monkeypatch.delenv('ORB_PDR_VETO_MIN_PCT', raising=False)
    alpaca = MagicMock(spec=AlpacaClient)
    db = MagicMock(spec=Database)
    sm = MagicMock(spec=StopMonitor)
    return ORBEngine(alpaca_client=alpaca, db=db, stop_monitor=sm, config=orb_cfg)


def _cand(sym, pdr):
    c = CandidateState(symbol=sym)
    c.features = {} if pdr == 'MISSING' else {'prev_day_range_pct': pdr}
    if pdr is None:
        c.features = {'prev_day_range_pct': None}
    return c


class TestEngineConfig:
    def test_yaml_knobs_loaded(self, engine):
        assert engine.pdr_veto_enabled is True
        assert engine.pdr_veto_min_pct == 8.0

    def test_env_master_disable(self, orb_cfg, monkeypatch):
        monkeypatch.setenv('ORB_PDR_VETO', '0')
        eng = ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient),
                        db=MagicMock(spec=Database),
                        stop_monitor=MagicMock(spec=StopMonitor), config=orb_cfg)
        assert eng.pdr_veto_enabled is False

    def test_env_threshold_override(self, orb_cfg, monkeypatch):
        monkeypatch.setenv('ORB_PDR_VETO_MIN_PCT', '10.5')
        eng = ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient),
                        db=MagicMock(spec=Database),
                        stop_monitor=MagicMock(spec=StopMonitor), config=orb_cfg)
        assert eng.pdr_veto_min_pct == 10.5

    def test_yaml_flag_off_disables(self, orb_cfg):
        orb_cfg['filter']['prev_day_range_veto']['enabled'] = False
        eng = ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient),
                        db=MagicMock(spec=Database),
                        stop_monitor=MagicMock(spec=StopMonitor), config=orb_cfg)
        assert eng.pdr_veto_enabled is False

    def test_missing_block_defaults_on(self, orb_cfg):
        """Nodes whose orb.yaml predates the ship still get the veto
        (matches skip_q1 convention: shipped default lives in code)."""
        del orb_cfg['filter']['prev_day_range_veto']
        eng = ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient),
                        db=MagicMock(spec=Database),
                        stop_monitor=MagicMock(spec=StopMonitor), config=orb_cfg)
        assert eng.pdr_veto_enabled is True
        assert eng.pdr_veto_min_pct == DEFAULT_MIN_PDR_PCT


class TestFeaturePlumbing:
    def test_compute_features_adds_pdr(self, engine):
        cand = CandidateState(symbol='AAA')
        cand.range_data = RangeData(
            symbol='AAA', range_high=10.0, range_low=9.5, range_volume=50_000,
            range_avg_bar_range_pct=1.0, range_close=9.9,
            range_start_ts=pd.Timestamp.utcnow(), range_open=9.6)
        feats = engine._compute_features(
            cand, prev_day_bar={'close': 10.0, 'high': 11.0, 'low': 9.0},
            daily_stats_20d=None)
        assert feats['prev_day_range_pct'] == pytest.approx(20.0)

    def test_compute_features_omits_pdr_on_bad_prev_bar(self, engine):
        cand = CandidateState(symbol='AAA')
        cand.range_data = RangeData(
            symbol='AAA', range_high=10.0, range_low=9.5, range_volume=50_000,
            range_avg_bar_range_pct=1.0, range_close=9.9,
            range_start_ts=pd.Timestamp.utcnow(), range_open=9.6)
        feats = engine._compute_features(
            cand, prev_day_bar={'close': 0.0, 'high': 0.0, 'low': 0.0},
            daily_stats_20d=None)
        assert 'prev_day_range_pct' not in feats


class TestVetoSeam:
    def test_quiet_prev_day_rejected(self, engine):
        c = _cand('QUIET', 5.2)
        assert engine._pdr_veto_reject(c) is True
        assert c.rejected_reason == 'pdr_veto'

    def test_explosive_prev_day_passes(self, engine):
        c = _cand('LOUD', 25.0)
        assert engine._pdr_veto_reject(c) is False
        assert c.rejected_reason is None

    def test_at_threshold_rejected(self, engine):
        assert engine._pdr_veto_reject(_cand('EDGE', 8.0)) is True

    def test_missing_feature_fails_open_with_warning(self, engine, caplog):
        import logging
        c = _cand('NOPDR', 'MISSING')
        with caplog.at_level(logging.WARNING):
            assert engine._pdr_veto_reject(c) is False
        assert any('PDR VETO' in r.message and 'fail-open' in r.message
                   for r in caplog.records)
        assert c.rejected_reason is None

    def test_disabled_flag_passes_everything(self, engine):
        engine.pdr_veto_enabled = False
        assert engine._pdr_veto_reject(_cand('QUIET', 0.5)) is False

    def test_none_features_dict_fails_open(self, engine):
        c = CandidateState(symbol='RAW')  # features never computed
        assert engine._pdr_veto_reject(c) is False


class TestNoBackfillWiring:
    """The veto must run INSIDE the submit loop over top_syms (post-ranking,
    post-dedup) so a vetoed pick's slot is never handed to the next-ranked
    candidate. Source-level pin: the refill variant scored MDD −$50K."""

    def test_veto_called_from_submit_loop_source(self):
        import inspect
        from trading import orb_engine as mod
        src = inspect.getsource(mod.ORBEngine.check_entries)
        loop_at = src.index('for sym in top_syms')
        veto_at = src.index('_pdr_veto_reject')
        assert veto_at > loop_at, (
            'veto must apply AFTER top-K selection (no-refill form)')
        # and nothing re-ranks/extends top_syms after the veto
        assert 'top_syms.extend' not in src and 'top_syms +=' not in src

    def test_vetoed_pick_not_submitted_and_not_replaced(self, engine):
        """Drive the submit loop directly: 2 picks, first vetoed. Exactly
        one submission happens, and the vetoed slot is NOT backfilled."""
        submitted = []
        engine._pdr_veto_reject = MagicMock(
            side_effect=lambda c: c.symbol == 'QUIET')
        for sym in ['QUIET', 'LOUD']:
            c = _cand(sym, 5.0 if sym == 'QUIET' else 20.0)
            if engine._pdr_veto_reject(c):
                continue
            submitted.append(sym)
        assert submitted == ['LOUD']


class TestVetoConsumesSlot:
    """2026-07-07 IREZ incident: a vetoed pick must consume its daily slot.
    Pre-fix, slot math recounted after an exit and backfilled the
    next-ranked name cross-tick — the refill form the BT proved toxic."""

    def test_veto_records_and_marks_submitted(self, engine):
        c = _cand('QUIET', 5.0)
        assert engine._pdr_veto_reject(c) is True
        assert 'QUIET' in engine._pdr_vetoed_today
        assert c.plan_submitted is True   # no per-tick rescoring/re-picking

    def test_daily_cap_counts_vetoed(self, engine):
        """entered=1 + vetoed=3 -> cap(4) hit -> no cross-tick backfill."""
        engine._symbols_entered_today_db = lambda: {'BEZ'}
        for s in ('TECS', 'SSG', 'MUD'):
            engine._pdr_veto_reject(_cand(s, 5.0))
        out = engine.check_entries(symbols=['IREZ'])
        assert out == []

    def test_reset_daily_clears_vetoed(self, engine):
        engine._pdr_vetoed_today.add('TECS')
        engine.reset_daily()
        assert engine._pdr_vetoed_today == set()
