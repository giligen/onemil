"""Catalyst-required veto (2026-07-18 ship, owner-approved −$36K budget).

Rule: entry needs own-ticker premarket news OR complex confirmation
(>= min_cohort same-morning candidates sharing the underlying anchor).
Newsless-and-alone -> veto, slot consumed, NO refill.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_asset_class import load_class_map, underlying_anchor
from trading.orb_catalyst_veto import (
    DEFAULT_MIN_COHORT, anchor_cohort_counts, catalyst_veto_applies,
    has_complex_confirmation,
)
from trading.orb_engine import ORBEngine, CandidateState
from trading.stop_monitor import StopMonitor


class TestUnderlyingAnchor:
    def test_stock_anchors_itself(self):
        assert underlying_anchor('MSTR', 'MicroStrategy Incorporated',
                                 {'MSTR': 'stock'}) == 'MSTR'

    def test_wrapper_anchors_underlying(self):
        cm = {'PLTR': 'stock'}
        assert underlying_anchor(
            'PLTZ', 'Tidal Trust II Defiance Daily Target 2x Short PLTR ETF',
            cm) == 'PLTR'

    def test_index_wrapper_no_anchor(self):
        assert underlying_anchor(
            'KOLD', 'ProShares UltraShort Bloomberg Natural Gas',
            {'BE': 'stock'}) is None

    def test_brand_token_not_anchor_without_map_entry(self):
        """'AI'-like fragments must not become anchors: token must be a
        known STOCK in the class map."""
        cm = {'ZZZZ': 'stock'}
        assert underlying_anchor('AIXX', 'Brand 2X Long AI Daily ETF',
                                 cm) is None

    def test_empty_name_none(self):
        assert underlying_anchor('XX', None, {}) is None

    def test_real_map_spot_checks(self):
        cm = load_class_map()
        assert underlying_anchor(
            'BEZ', 'Tradr 2X Short BE Daily ETF', cm) == 'BE'
        assert underlying_anchor(
            'MSTZ', 'T-Rex 2X Inverse MSTR Daily Target ETF', cm) == 'MSTR'


class TestVetoLogic:
    def test_news_never_vetoed(self):
        assert catalyst_veto_applies(True, None, {}) is False

    def test_unknown_news_fails_open(self):
        assert catalyst_veto_applies(None, None, {}) is False

    def test_newsless_alone_vetoed(self):
        assert catalyst_veto_applies(False, 'IREN', {'IREN': 1}) is True

    def test_newsless_complex_confirmed_kept(self):
        assert catalyst_veto_applies(False, 'IREN', {'IREN': 3}) is False

    def test_newsless_no_anchor_vetoed(self):
        assert catalyst_veto_applies(False, None, {'IREN': 5}) is True

    def test_min_cohort_boundary(self):
        assert has_complex_confirmation('A', {'A': 2}, 2) is True
        assert has_complex_confirmation('A', {'A': 1}, 2) is False

    def test_cohort_counts(self):
        assert anchor_cohort_counts(['IREN', 'IREN', 'PLTR', None, 'IREN']) \
            == {'IREN': 3, 'PLTR': 1}


@pytest.fixture
def engine(monkeypatch):
    for v in ('ORB_CATALYST_VETO', 'ORB_PDR_VETO', 'ORB_PM_MULT'):
        monkeypatch.delenv(v, raising=False)
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    return ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient),
                     db=MagicMock(spec=Database),
                     stop_monitor=MagicMock(spec=StopMonitor), config=cfg)


def _cand(sym):
    c = CandidateState(symbol=sym)
    c.features = {'prev_day_range_pct': 20.0}
    return c


class TestEngineWiring:
    def test_config_loaded(self, engine):
        assert engine.catalyst_veto_enabled is True
        assert engine.catalyst_min_cohort == 2

    def test_env_disable(self, monkeypatch):
        monkeypatch.setenv('ORB_CATALYST_VETO', '0')
        with open(Path(__file__).parent.parent / 'orb.yaml') as f:
            cfg = yaml.safe_load(f)
        cfg['strategy']['enabled'] = True
        eng = ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient),
                        db=MagicMock(spec=Database),
                        stop_monitor=MagicMock(spec=StopMonitor), config=cfg)
        assert eng.catalyst_veto_enabled is False

    def _setup(self, engine, symbols, news_map, anchors):
        engine.build_universe(source_loader=lambda: symbols)
        engine._news_flags = {s: {'n_articles': 1 if news_map.get(s) else 0,
                                  'headline': ''} for s in symbols}
        engine._news_fetch_done_day = __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc).date()
        engine._anchor_cache = dict(anchors)

    def test_newsless_alone_rejected_slot_consumed(self, engine):
        self._setup(engine, ['LONE', 'OTHR'], {}, {'LONE': 'LONE',
                                                   'OTHR': 'OTHR'})
        c = _cand('LONE')
        assert engine._catalyst_veto_reject(c) is True
        assert c.rejected_reason == 'catalyst_veto'
        assert 'LONE' in engine._pdr_vetoed_today   # slot accounting
        assert c.plan_submitted is True

    def test_newsy_kept(self, engine):
        self._setup(engine, ['NEWSY'], {'NEWSY': True}, {'NEWSY': 'NEWSY'})
        assert engine._catalyst_veto_reject(_cand('NEWSY')) is False

    def test_complex_confirmed_kept(self, engine):
        """Two IREN wrappers in the morning cohort -> both confirmed."""
        self._setup(engine, ['IREX', 'IREZ'], {},
                    {'IREX': 'IREN', 'IREZ': 'IREN'})
        assert engine._catalyst_veto_reject(_cand('IREX')) is False

    def test_unknown_news_fails_open(self, engine):
        self._setup(engine, ['NODATA'], {}, {'NODATA': 'NODATA'})
        engine._news_flags = {'NODATA': None}   # fetch failed -> poisoned
        assert engine._catalyst_veto_reject(_cand('NODATA')) is False

    def test_disabled_passes_everything(self, engine):
        engine.catalyst_veto_enabled = False
        self._setup(engine, ['LONE'], {}, {'LONE': 'LONE'})
        assert engine._catalyst_veto_reject(_cand('LONE')) is False

    def test_called_after_pdr_in_submit_loop(self):
        import inspect
        from trading import orb_engine as em
        src = inspect.getsource(em.ORBEngine.check_entries)
        assert src.index('_pdr_veto_reject') < src.index(
            '_catalyst_veto_reject')
        assert 'top_syms.extend' not in src   # still no refill anywhere


class TestBtParity:
    def test_pipeline_uses_shared_helper(self):
        src = Path(Path(__file__).parent.parent /
                   'study_orb_pipeline_static_lock.py').read_text()
        assert 'catalyst_veto_applies' in src
        assert 'underlying_anchor' in src
        assert 'ORB_CATALYST_VETO' in src
