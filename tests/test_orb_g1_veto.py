"""G1 volatility-fingerprint veto — B+ RESTART 2026-08-15 ship.

Vetoes ORB picks whose (return_volatility_20d, prev_day_range_pct) fingerprint
fails BOTH frozen minimums (7.106 / 9.226). Applied POST-ranking with NO refill
(same invariant as PDR/catalyst). The load-bearing subtlety (review P1-2): the
fail-open branch must fire on rv20 ∈ {None, NaN, 0.0} BEFORE the AND, so
`rv20 == 0.0` (history-too-short marker) KEEPS the pick instead of inverting to
a veto; `prev_day_range_pct == 0.0` is a REAL flat day and IS vetoable.

Covers: shared helper matrix, the live rv20 computation golden-vectored against
analysis_results/orb_features_20260814_1741.csv, the `_g1_veto_reject` seam,
config/env knobs, no-refill slot accounting, and BT<->live parity by construction.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, CandidateState
from trading.orb_g1_veto import (
    DEFAULT_PDR_MIN, DEFAULT_RV20_MIN, g1_reject,
)
from trading.stop_monitor import StopMonitor


# ---------------------------------------------------------------------------
# Shared helper decision matrix (review P1-2)
# ---------------------------------------------------------------------------

class TestG1RejectMatrix:
    def test_defaults_are_frozen(self):
        assert DEFAULT_RV20_MIN == 7.106
        assert DEFAULT_PDR_MIN == 9.226

    def test_keep_when_both_above(self):
        assert g1_reject(7.2, 9.3) is None

    def test_keep_at_both_boundaries(self):
        # KEEP is `>=` on both legs.
        assert g1_reject(7.106, 9.226) is None

    def test_veto_when_rv_below(self):
        assert g1_reject(7.10, 9.30) is not None

    def test_veto_when_pdr_below(self):
        assert g1_reject(7.20, 9.20) is not None

    def test_veto_when_both_below(self):
        assert g1_reject(1.0, 1.0) is not None

    # --- fail-open markers (KEEP) ---
    def test_rv_none_fails_open(self):
        assert g1_reject(None, 5.0) is None

    def test_rv_nan_fails_open(self):
        assert g1_reject(float('nan'), 5.0) is None

    def test_rv_zero_fails_open_not_inverted(self):
        """rv20 == 0.0 is the 'history too short' marker — KEEP, never veto.
        The naive `0 >= 7.106` would VETO and invert BT (review P1-2)."""
        assert g1_reject(0.0, 5.0) is None
        assert g1_reject(0.0, 0.0) is None

    def test_pdr_none_fails_open(self):
        assert g1_reject(7.2, None) is None

    def test_pdr_nan_fails_open(self):
        assert g1_reject(7.2, float('nan')) is None

    # --- pdr == 0.0 is REAL (asymmetric with rv20) ---
    def test_pdr_zero_is_real_and_vetoable(self):
        assert g1_reject(7.2, 0.0) is not None

    def test_full_matrix(self):
        """rv ∈ {NaN, 0.0, 7.10, 7.11} × pdr ∈ {NaN, 0.0, 9.22, 9.23}."""
        nan = float('nan')
        keep = lambda r, p: g1_reject(r, p) is None
        # rv fail-open rows -> always KEEP regardless of pdr
        for p in (nan, 0.0, 9.22, 9.23):
            assert keep(nan, p)
            assert keep(0.0, p)
        # rv real, pdr NaN -> KEEP
        assert keep(7.11, nan) and keep(7.10, nan)
        # rv=7.10 (< min 7.106) real pdr -> VETO
        assert not keep(7.10, 0.0)
        assert not keep(7.10, 9.23)
        # rv=7.11 (>= min), pdr below/at
        assert not keep(7.11, 0.0)     # pdr 0 real, below -> veto
        assert not keep(7.11, 9.22)    # pdr below min -> veto
        assert keep(7.11, 9.23)        # both clear -> keep

    def test_non_numeric_inputs_fail_open(self):
        assert g1_reject('x', 9.3) is None
        assert g1_reject(7.2, 'y') is None


# ---------------------------------------------------------------------------
# Live rv20 computation — golden vectors vs the frozen features CSV
# ---------------------------------------------------------------------------

# (symbol, date, prior-20 daily closes, expected return_volatility_20d)
# Extracted from analysis_results/orb_features_20260814_1741.csv (the frozen
# B+ source) + data/cache.db daily_bars. rv20 depends only on the close
# SEQUENCE, so the fixture dates are re-labeled to recent days in the test to
# avoid the staleness-refetch path — the sequence (and thus rv20) is unchanged.
RV20_GOLDEN = [
    ("DHT",  [9.18, 9.29, 9.64, 9.35, 9.33], 2.4477256308),
    ("ENVX", [11.0, 10.87, 12.11, 12.66, 12.46], 5.2719300044),
    ("FRO",  [13.89, 14.19, 14.69, 14.25, 14.27], 2.4531419948),
    ("FUBO", [1.25, 1.26, 1.41, 1.44, 5.06], 106.7998328938),
    ("GETY", [2.1, 2.16, 2.11, 2.39, 2.57], 5.7543463757),
    ("MOB",  [3.9, 3.81, 3.47, 4.29, 4.66], 12.3214441422),
    ("MSTZ", [25.85, 28.09, 26.02, 19.21, 14.71], 13.9630559049),
    ("MUU",  [15.79, 15.31, 16.46, 17.41, 21.03], 8.5220796908),
    ("NRXP", [1.53, 2.2, 2.99, 2.69, 3.51], 20.7936721685),
    ("PBYI", [3.13, 3.05, 3.12, 2.95, 3.55], 10.0207002687),
    # < 5 prior bars -> 0.0 "history too short" marker (the G1 fail-open case)
    ("APM",  [1.355, 2.8], 0.0),
    ("ATCH", [0.1631, 0.16], 0.0),
]


@pytest.fixture
def orb_cfg():
    cfg = yaml.safe_load(open(Path(__file__).parent.parent / 'orb.yaml'))
    cfg['strategy']['enabled'] = True
    return cfg


@pytest.fixture
def engine(orb_cfg, monkeypatch):
    for v in ('ORB_G1_VETO',):
        monkeypatch.delenv(v, raising=False)
    return ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient),
                     db=MagicMock(spec=Database),
                     stop_monitor=MagicMock(spec=StopMonitor), config=orb_cfg)


def _bars_from_closes(closes):
    """Build recent-dated daily-bar dicts (only close matters for rv20)."""
    today = datetime.utcnow().date()
    bars = []
    for i, c in enumerate(closes):
        d = today - timedelta(days=len(closes) - i)  # ends yesterday, recent
        bars.append({'date': d.isoformat(), 'open': c, 'high': c,
                     'low': c, 'close': c, 'volume': 1000})
    return bars


@pytest.mark.parametrize("symbol,closes,expected", RV20_GOLDEN)
def test_live_rv20_matches_features_csv(engine, symbol, closes, expected):
    """The LIVE _get_feature_context rv20 must equal the frozen CSV value —
    exact numpy std(ddof=0)*100 parity with study_orb_features.py:308-314."""
    engine.db.get_daily_bars_cached = MagicMock(
        return_value={symbol: _bars_from_closes(closes)})
    ctx = engine._get_feature_context(symbol)
    rv = ctx['daily_stats_20d']['return_volatility_20d']
    assert rv == pytest.approx(expected, abs=1e-6)


def test_rv20_flows_into_features(engine):
    """_compute_features must surface return_volatility_20d for G1."""
    engine.db.get_daily_bars_cached = MagicMock(
        return_value={'ENVX': _bars_from_closes(
            [11.0, 10.87, 12.11, 12.66, 12.46])})
    ctx = engine._get_feature_context('ENVX')
    from trading.orb_engine import RangeData
    import pandas as pd
    cand = CandidateState(symbol='ENVX')
    cand.range_data = RangeData(
        symbol='ENVX', range_high=13.0, range_low=12.5, range_volume=50_000,
        range_avg_bar_range_pct=1.0, range_close=12.9,
        range_start_ts=pd.Timestamp.utcnow(), range_open=12.6)
    feats = engine._compute_features(
        cand, prev_day_bar=ctx.get('prev_day_bar'),
        daily_stats_20d=ctx.get('daily_stats_20d'))
    assert feats['return_volatility_20d'] == pytest.approx(5.2719300044, abs=1e-6)


# ---------------------------------------------------------------------------
# Engine config / env
# ---------------------------------------------------------------------------

class TestEngineConfig:
    def test_yaml_knobs(self, engine):
        assert engine.g1_veto_enabled is True
        assert engine.g1_rv20_min == 7.106
        assert engine.g1_pdr_min == 9.226

    def test_env_disable(self, orb_cfg, monkeypatch):
        monkeypatch.setenv('ORB_G1_VETO', '0')
        e = ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient),
                      db=MagicMock(spec=Database),
                      stop_monitor=MagicMock(spec=StopMonitor), config=orb_cfg)
        assert e.g1_veto_enabled is False

    def test_missing_block_defaults_on(self, orb_cfg):
        del orb_cfg['filter']['g1_veto']
        e = ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient),
                      db=MagicMock(spec=Database),
                      stop_monitor=MagicMock(spec=StopMonitor), config=orb_cfg)
        assert e.g1_veto_enabled is True
        assert e.g1_rv20_min == DEFAULT_RV20_MIN
        assert e.g1_pdr_min == DEFAULT_PDR_MIN


# ---------------------------------------------------------------------------
# _g1_veto_reject seam + slot accounting
# ---------------------------------------------------------------------------

def _cand(sym, rv, pdr):
    c = CandidateState(symbol=sym)
    c.features = {'return_volatility_20d': rv, 'prev_day_range_pct': pdr}
    return c


class TestVetoSeam:
    def test_low_rv_rejected(self, engine):
        c = _cand('LOWRV', 7.0, 20.0)
        assert engine._g1_veto_reject(c) is True
        assert c.rejected_reason == 'g1_veto'

    def test_clears_fingerprint_passes(self, engine):
        c = _cand('GOOD', 8.0, 12.0)
        assert engine._g1_veto_reject(c) is False
        assert c.rejected_reason is None

    def test_rv_zero_fails_open(self, engine):
        c = _cand('SHORT', 0.0, 5.0)
        assert engine._g1_veto_reject(c) is False

    def test_disabled_passes_everything(self, engine):
        engine.g1_veto_enabled = False
        assert engine._g1_veto_reject(_cand('X', 1.0, 1.0)) is False

    def test_consumes_slot_no_refill(self, engine):
        c = _cand('LOWRV', 7.0, 20.0)
        assert engine._g1_veto_reject(c) is True
        assert 'LOWRV' in engine._pdr_vetoed_today
        assert c.plan_submitted is True

    def test_missing_features_fail_open(self, engine):
        c = CandidateState(symbol='RAW')  # features never computed
        assert engine._g1_veto_reject(c) is False


class TestNoRefillWiring:
    def test_veto_runs_in_submit_loop_after_ranking(self):
        import inspect
        from trading import orb_engine as mod
        src = inspect.getsource(mod.ORBEngine.check_entries)
        loop_at = src.index('for sym in top_syms')
        g1_at = src.index('_g1_veto_reject')
        assert g1_at > loop_at
        assert 'top_syms.extend' not in src and 'top_syms +=' not in src

    def test_reset_daily_clears_slot(self, engine):
        engine._pdr_vetoed_today.add('LOWRV')
        engine.reset_daily()
        assert engine._pdr_vetoed_today == set()


# ---------------------------------------------------------------------------
# BT<->live parity by construction
# ---------------------------------------------------------------------------

class TestBTParity:
    def test_pipeline_imports_shared_helper(self):
        """The BT pipeline must call the SAME g1_reject, not a reimplementation."""
        import inspect
        import study_orb_pipeline_static_lock as pipe
        src = inspect.getsource(pipe)
        assert 'from trading.orb_g1_veto import g1_reject' in src

    def test_same_decision_both_sides(self, engine):
        """Identical inputs -> identical KEEP/VETO on the helper and the seam."""
        for rv, pdr in [(7.2, 9.3), (7.0, 9.3), (0.0, 5.0), (7.2, 0.0)]:
            helper_veto = g1_reject(rv, pdr, 7.106, 9.226) is not None
            seam_veto = engine._g1_veto_reject(_cand(f'S{rv}{pdr}', rv, pdr))
            assert helper_veto == seam_veto
