"""Opening-range-size veto (2026-09-08): shared helper, engine knobs/seam,
BT pipeline parity, and the G1 short-history switch."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, CandidateState
from trading.orb_g1_veto import g1_reject
from trading.orb_range_size_veto import (
    DEFAULT_MIN_RANGE_SIZE_PCT, range_size_veto_applies,
)
from trading.stop_monitor import StopMonitor


class TestHelper:
    def test_threshold_is_the_scan_edge(self):
        assert DEFAULT_MIN_RANGE_SIZE_PCT == 2.221

    def test_at_or_below_vetoes_above_passes(self):
        assert range_size_veto_applies(2.221) is True
        assert range_size_veto_applies(1.28) is True
        assert range_size_veto_applies(2.222) is False
        assert range_size_veto_applies(5.73) is False

    def test_missing_never_vetoes(self):
        assert range_size_veto_applies(None) is False
        assert range_size_veto_applies(float('nan')) is False
        assert range_size_veto_applies('n/a') is False

    def test_threshold_override(self):
        assert range_size_veto_applies(3.0, min_pct=3.5) is True


class TestG1ShortHistorySwitch:
    def test_marker_fails_open_by_default(self):
        assert g1_reject(0.0, 12.0) is None

    def test_marker_vetoed_when_switch_on(self):
        r = g1_reject(0.0, 12.0, short_history_veto=True)
        assert r is not None and 'short history' in r

    def test_missing_rv_still_fails_open_with_switch(self):
        assert g1_reject(None, 12.0, short_history_veto=True) is None
        assert g1_reject(float('nan'), 12.0, short_history_veto=True) is None

    def test_real_values_unchanged_by_switch(self):
        assert g1_reject(8.0, 12.0, short_history_veto=True) is None
        assert g1_reject(5.0, 12.0, short_history_veto=True) is not None


@pytest.fixture
def orb_cfg():
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    return cfg


def _engine(cfg):
    return ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient), db=MagicMock(spec=Database),
                     stop_monitor=MagicMock(spec=StopMonitor), config=cfg)


@pytest.fixture
def engine(orb_cfg, monkeypatch):
    for k in ('ORB_RANGE_SIZE_VETO', 'ORB_RANGE_SIZE_VETO_MIN_PCT'):
        monkeypatch.delenv(k, raising=False)
    return _engine(orb_cfg)


def _cand(sym, rs):
    c = CandidateState(symbol=sym)
    c.features = {} if rs == 'MISSING' else {'range_size_pct': rs}
    return c


class TestEngineConfig:
    def test_yaml_knobs_loaded(self, engine):
        assert engine.range_size_veto_enabled is True
        assert engine.range_size_veto_min_pct == 2.221
        assert engine.g1_short_history_veto is True

    def test_env_master_disable(self, orb_cfg, monkeypatch):
        monkeypatch.setenv('ORB_RANGE_SIZE_VETO', '0')
        assert _engine(orb_cfg).range_size_veto_enabled is False

    def test_env_threshold_override(self, orb_cfg, monkeypatch):
        monkeypatch.setenv('ORB_RANGE_SIZE_VETO_MIN_PCT', '3.0')
        assert _engine(orb_cfg).range_size_veto_min_pct == 3.0

    def test_yaml_flags_off(self, orb_cfg):
        orb_cfg['filter']['range_size_veto']['enabled'] = False
        orb_cfg['filter']['g1_veto']['short_history_veto'] = False
        e = _engine(orb_cfg)
        assert e.range_size_veto_enabled is False and e.g1_short_history_veto is False

    def test_template_carries_the_knobs(self):
        t = yaml.safe_load(open(Path(__file__).parent.parent / 'orb.yaml.template'))
        assert t['filter']['range_size_veto']['min_range_size_pct'] == 2.221
        assert 'short_history_veto' in t['filter']['g1_veto']


class TestEngineSeam:
    def test_small_range_vetoed_slot_consumed(self, engine):
        c = _cand('TINY', 1.57)
        assert engine._range_size_veto_reject(c) is True
        assert c.rejected_reason == 'range_size_veto' and c.plan_submitted is True
        assert 'TINY' in engine._pdr_vetoed_today

    def test_normal_range_passes(self, engine):
        c = _cand('OK', 5.73)
        assert engine._range_size_veto_reject(c) is False
        assert c.plan_submitted is False

    def test_missing_feature_fails_open(self, engine):
        assert engine._range_size_veto_reject(_cand('NOFEAT', 'MISSING')) is False

    def test_disabled_never_vetoes(self, orb_cfg):
        orb_cfg['filter']['range_size_veto']['enabled'] = False
        assert _engine(orb_cfg)._range_size_veto_reject(_cand('TINY', 1.0)) is False

    def test_g1_seam_vetoes_short_history(self, engine):
        c = CandidateState(symbol='NEWLIST')
        c.features = {'return_volatility_20d': 0.0, 'prev_day_range_pct': 15.0}
        assert engine._g1_veto_reject(c) is True and c.rejected_reason == 'g1_veto'

    def test_call_order_after_g1_before_catalyst(self):
        import inspect
        src = inspect.getsource(ORBEngine)
        i_g1 = src.index('if self._g1_veto_reject(cand):')
        i_rs = src.index('if self._range_size_veto_reject(cand):')
        i_cat = src.index('if self._catalyst_veto_reject(cand):')
        assert i_g1 < i_rs < i_cat


class TestPipelineParity:
    """The pipeline applies the same helper to the same column (BT parity)."""

    def test_pipeline_reads_yaml_knobs_and_calls_shared_helper(self):
        src = open(Path(__file__).parent.parent / 'study_orb_pipeline_static_lock.py').read()
        assert "from trading.orb_range_size_veto import range_size_veto_applies" in src
        assert "short_history_veto=bt_cfg['g1_short_history_veto']" in src
        assert "(filt.get('range_size_veto') or {}).get('min_range_size_pct'" in src
