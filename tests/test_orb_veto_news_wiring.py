"""Catalyst-veto news/anchor wiring (2026-07-19 review round 3).

Locks the round-2 P1 fixes so they can't regress:
  1. news_needed gating — the news fetch (+ 9:33 lag pass) must survive
     PM sizing being disabled, else _news_flags stays empty and the veto
     silently fail-opens into a no-op (accidental behavior; forbidden).
  2. _prewarm_anchors — anchors warm off the hot path; the submit loop
     (allow_api=False) must NEVER hit the asset-name API.
  3. Raw-vs-effective news semantics — the veto reads RAW own-ticker
     news (BT parity), NOT the class-gated effective map the sizing
     boost uses: a leveraged wrapper with news is boost-INELIGIBLE but
     still catalyst-CONFIRMED.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine
from trading.stop_monitor import StopMonitor


@pytest.fixture
def engine(monkeypatch):
    for v in ('ORB_PM_MULT', 'ORB_PM_NEWS_GATE', 'ORB_CATALYST_VETO'):
        monkeypatch.delenv(v, raising=False)
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    alpaca = MagicMock(spec=AlpacaClient)
    alpaca.get_premarket_1min_bars_multi = MagicMock(return_value={})
    alpaca.get_premarket_news_multi = MagicMock(return_value={
        'AAA': {'n_articles': 1, 'headline': 'AAA news'},
        'BBB': {'n_articles': 0, 'headline': ''}})
    alpaca.get_asset_name = MagicMock(return_value='Some Name Inc')
    alpaca.get_open_positions.return_value = []
    db = MagicMock(spec=Database)
    sm = MagicMock(spec=StopMonitor)
    eng = ORBEngine(alpaca_client=alpaca, db=db, stop_monitor=sm,
                    config=cfg)
    eng.build_universe(source_loader=lambda: ['AAA', 'BBB'])
    return eng


def _at(eng, hh, mm, fn, *a, **kw):
    with patch('trading.orb_engine.datetime') as mdt:
        mdt.now.return_value = datetime(2026, 7, 20, hh + 4, mm, 0,
                                        tzinfo=timezone.utc)  # EDT
        mdt.combine = datetime.combine
        mdt.strptime = datetime.strptime
        return fn(*a, **kw)


class TestNewsNeededGating:
    def test_pm_off_veto_on_still_fetches_news(self, engine):
        """The round-2 P1: with PM sizing disabled the veto must still
        get news flags — pre-fix everything read as news-unknown and
        the veto fail-opened into nothing."""
        engine.pm_mult_enabled = False
        assert engine.catalyst_veto_enabled is True
        _at(engine, 9, 32, engine._maybe_prefetch_pm)
        engine.alpaca.get_premarket_news_multi.assert_called_once()
        engine.alpaca.get_premarket_1min_bars_multi.assert_not_called()
        assert engine._get_has_news('AAA') is True
        assert engine._get_has_news('BBB') is False

    def test_pm_off_veto_off_fetches_nothing(self, engine):
        engine.pm_mult_enabled = False
        engine.catalyst_veto_enabled = False
        _at(engine, 9, 32, engine._maybe_prefetch_pm)
        engine.alpaca.get_premarket_news_multi.assert_not_called()
        engine.alpaca.get_premarket_1min_bars_multi.assert_not_called()

    def test_pm_on_gate_off_veto_on_fetches_news(self, engine):
        engine.pm_news_gate = False
        _at(engine, 9, 32, engine._maybe_prefetch_pm)
        engine.alpaca.get_premarket_news_multi.assert_called_once()

    def test_933_lag_pass_runs_with_pm_off(self, engine):
        """The indexing-lag re-fetch matters MORE for the veto than for
        sizing (a missed newsy flag wrongly KILLS a trade, not just
        unboosts it) — it must run when only the veto needs news."""
        engine.pm_mult_enabled = False
        _at(engine, 9, 32, engine._maybe_prefetch_pm)
        _at(engine, 9, 33, engine._maybe_prefetch_pm)
        # first call = 9:31 batch; second = 9:33 refresh of no-news syms
        assert engine.alpaca.get_premarket_news_multi.call_count == 2
        stale_arg = engine.alpaca.get_premarket_news_multi.call_args[0][0]
        assert 'BBB' in stale_arg and 'AAA' not in stale_arg

    def test_current_config_behavior_unchanged(self, engine):
        """Today's config (pm on, gate on, veto on): one news fetch +
        one pm fetch at 9:32 — byte-identical to pre-fix."""
        _at(engine, 9, 32, engine._maybe_prefetch_pm)
        engine.alpaca.get_premarket_news_multi.assert_called_once()
        engine.alpaca.get_premarket_1min_bars_multi.assert_called_once()


class TestAnchorPrewarm:
    def test_prewarm_covers_all_candidates(self, engine):
        engine._prewarm_anchors()
        assert set(engine.candidates.keys()) <= set(engine._anchor_cache)

    def test_prewarm_noop_when_veto_disabled(self, engine):
        engine.catalyst_veto_enabled = False
        engine._prewarm_anchors()
        assert engine._anchor_cache == {}

    def test_prewarm_second_call_is_pure_cache(self, engine):
        engine._prewarm_anchors()
        n = engine.alpaca.get_asset_name.call_count
        engine._prewarm_anchors()
        assert engine.alpaca.get_asset_name.call_count == n

    def test_submit_path_never_calls_asset_api(self, engine):
        """allow_api=False: an unwarmed symbol missing from the class
        map resolves to no-anchor WITHOUT any network call — the 9:35
        burst can never pay 8s name lookups."""
        a = engine._anchor_for('ZZZUNWARMED1', allow_api=False)
        assert a is None
        engine.alpaca.get_asset_name.assert_not_called()
        # memoized: repeat is a cache hit
        assert engine._anchor_for('ZZZUNWARMED1', allow_api=False) is None


class TestVetoRawNewsSemantics:
    def _cand(self, engine, sym):
        from trading.orb_engine import CandidateState
        engine.candidates.setdefault(sym, CandidateState(symbol=sym))
        return engine.candidates[sym]

    def test_wrapper_with_news_not_vetoed(self, engine):
        """BT-parity semantic: veto reads RAW own-ticker news. A
        leveraged wrapper with news gets NO sizing boost (class-gated)
        but IS catalyst-confirmed — must not be vetoed."""
        c = self._cand(engine, 'WRAP2X')
        engine._news_flags['WRAP2X'] = {'n_articles': 2, 'headline': 'x'}
        engine._anchor_cache['WRAP2X'] = None   # alone, no complex
        for s in engine.candidates:
            engine._anchor_cache.setdefault(s, None)
        assert engine._catalyst_veto_reject(c) is False
        assert c.plan_submitted is False

    def test_newsless_and_alone_vetoed(self, engine):
        c = self._cand(engine, 'LONELY')
        engine._news_flags['LONELY'] = {'n_articles': 0, 'headline': ''}
        for s in engine.candidates:
            engine._anchor_cache.setdefault(s, None)
        assert engine._catalyst_veto_reject(c) is True
        assert c.plan_submitted is True
        assert c.rejected_reason == 'catalyst_veto'
        assert 'LONELY' in engine._pdr_vetoed_today

    def test_newsless_but_complex_confirmed_survives(self, engine):
        c = self._cand(engine, 'SIBA')
        self._cand(engine, 'SIBB')
        engine._news_flags['SIBA'] = {'n_articles': 0, 'headline': ''}
        for s in engine.candidates:
            engine._anchor_cache.setdefault(s, None)
        engine._anchor_cache['SIBA'] = 'UNDER'
        engine._anchor_cache['SIBB'] = 'UNDER'
        assert engine._catalyst_veto_reject(c) is False

    def test_news_unknown_fails_open(self, engine):
        c = self._cand(engine, 'NODATA')
        assert 'NODATA' not in engine._news_flags
        for s in engine.candidates:
            engine._anchor_cache.setdefault(s, None)
        assert engine._catalyst_veto_reject(c) is False
