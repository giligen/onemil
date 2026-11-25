"""`_rank_production_final` must log every drop of a ranged candidate (SKDD 2026-11-24:
a preplaced order was cancelled by the reconcile with no log line)."""
import logging

import pandas as pd
import pytest

import trading.orb_engine as orb_engine_mod
from trading.orb_engine import CandidateState, RangeData
from trading.orb_filter import FeatureParam
from tests.test_orb_grace_waitlist_20261008 import engine  # noqa: F401  (real-config fixture)

TAG = 'FINAL RANK'


def _rng(sym):
    return RangeData(symbol=sym, range_high=10.0, range_low=9.5, range_volume=50_000,
                     range_avg_bar_range_pct=1.0, range_close=9.9,
                     range_start_ts=pd.Timestamp.utcnow(), range_open=9.6)


@pytest.fixture
def eng(engine, monkeypatch):  # noqa: F811
    """Engine with one feature 'f1' (z = value), gap floor 5, threshold 0.0, skip_q1 on."""
    engine.z_params = {'f1': FeatureParam(sign=1, mean=0.0, std=1.0)}
    engine.filter_threshold = 0.0
    engine.skip_q1 = True
    engine.universe_min_gap_pct = 5.0
    engine.ranking_order = ['Q4', 'Q5', 'Q3', 'Q2', 'Q1']
    engine._get_feature_context = lambda sym: {}
    engine._feats = {}
    monkeypatch.setattr(engine, '_compute_features',
                        lambda cand, prev_day_bar=None, daily_stats_20d=None: dict(engine._feats[cand.symbol]))
    monkeypatch.setattr(orb_engine_mod, 'assign_quintile',
                        lambda score, cutoffs: 'Q1' if score < 0.5 else 'Q4')
    return engine


def _add(eng, sym, feats, pool=None):
    eng.candidates[sym] = CandidateState(symbol=sym)
    eng.candidates[sym].range_data = _rng(sym)
    eng._feats[sym] = feats
    if pool:
        eng._symbol_pool[sym] = pool


def _records(caplog):
    return [r for r in caplog.records if TAG in r.getMessage()]


def test_composite_none_names_missing_feature(eng, caplog):
    _add(eng, 'AAA', {'gap_pct': 8.0, 'f1': None})
    with caplog.at_level(logging.INFO):
        assert eng._rank_production_final() == []
    recs = _records(caplog)
    assert len(recs) == 1 and 'composite None' in recs[0].getMessage() and 'f1' in recs[0].getMessage()
    assert recs[0].levelno == logging.INFO


def test_below_threshold(eng, caplog):
    eng.filter_threshold = 1.0
    _add(eng, 'AAA', {'gap_pct': 8.0, 'f1': 0.9})
    with caplog.at_level(logging.INFO):
        assert eng._rank_production_final() == []
    recs = _records(caplog)
    assert len(recs) == 1 and '< threshold' in recs[0].getMessage() and recs[0].levelno == logging.INFO


def test_q1_skipped(eng, caplog):
    _add(eng, 'AAA', {'gap_pct': 8.0, 'f1': 0.2})
    with caplog.at_level(logging.INFO):
        assert eng._rank_production_final() == []
    recs = _records(caplog)
    assert len(recs) == 1 and 'Q1' in recs[0].getMessage() and recs[0].levelno == logging.INFO


def test_phantom_gap(eng, caplog):
    _add(eng, 'AAA', {'gap_pct': 2.0, 'f1': 2.0})
    with caplog.at_level(logging.INFO):
        assert eng._rank_production_final() == []
    recs = _records(caplog)
    assert len(recs) == 1 and 'phantom gap' in recs[0].getMessage() and recs[0].levelno == logging.INFO


def test_preplaced_drop_is_warning(eng, caplog):
    _add(eng, 'SKDD', {'gap_pct': 8.0, 'f1': None})
    eng._preplace_state['SKDD'] = object()
    with caplog.at_level(logging.INFO):
        assert eng._rank_production_final() == []
    recs = _records(caplog)
    assert len(recs) == 1 and recs[0].levelno == logging.WARNING
    assert 'PREPLACED' in recs[0].getMessage() and 'SKDD' in recs[0].getMessage()


def test_normal_rank_logs_nothing(eng, caplog):
    _add(eng, 'AAA', {'gap_pct': 8.0, 'f1': 2.0})
    with caplog.at_level(logging.INFO):
        assert [c.symbol for c in eng._rank_production_final()] == ['AAA']
    assert _records(caplog) == []


def test_non_production_pool_is_quiet(eng, caplog):
    _add(eng, 'AAA', {'gap_pct': 8.0, 'f1': None}, pool='addon_p1')
    with caplog.at_level(logging.INFO):
        assert eng._rank_production_final() == []
    assert _records(caplog) == []
