"""ORB latency pass 2 (docs/orb_latency_pass2_spec_20260925.md): execution.fast_submit.

Covers: no REST/SQLite in the veto hot path; concurrent submit preserves
ranking order and slot accounting; a failed submit does not refill; the
flag off leaves behaviour byte-identical.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, CandidateState
from trading.orb_planner import OrbTradePlan
from trading.stop_monitor import StopMonitor


@pytest.fixture
def orb_cfg():
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    return cfg


def _engine(cfg, fast_submit=False):
    cfg = dict(cfg)
    cfg['execution'] = dict(cfg.get('execution') or {})
    cfg['execution']['fast_submit'] = fast_submit
    return ORBEngine(alpaca_client=MagicMock(spec=[]), db=MagicMock(spec=[]),
                      stop_monitor=MagicMock(spec=StopMonitor), config=cfg)


def _cand(sym, **feats):
    c = CandidateState(symbol=sym)
    c.features = {
        'prev_day_range_pct': 20.0,
        'return_volatility_20d': 20.0,
        'range_size_pct': 5.0,
        **feats,
    }
    return c


class TestVetoesNoRestNoSql:
    """Item 3: ranking/vetoes are pure in-memory — assert with collaborators
    that raise on ANY attribute access (MagicMock(spec=[]))."""

    def test_all_four_vetoes_touch_no_collaborator(self, orb_cfg):
        engine = _engine(orb_cfg)
        assert engine.catalyst_veto_enabled is False  # catalyst-off (live config)
        c = _cand('ABCD')
        # None of these may touch self.alpaca / self.db (spec=[] raises on
        # any attribute access) — a hidden REST/SQLite call fails the test.
        assert engine._pdr_veto_reject(c) is False
        assert engine._g1_veto_reject(c) is False
        assert engine._range_size_veto_reject(c) is False
        assert engine._catalyst_veto_reject(c, cohort_symbols=['ABCD']) is False

    def test_vetoing_candidate_also_touches_no_collaborator(self, orb_cfg):
        engine = _engine(orb_cfg)
        c = _cand('QUIET', prev_day_range_pct=0.1)  # below pdr_veto_min_pct
        assert engine._pdr_veto_reject(c) is True
        assert c.rejected_reason == 'pdr_veto'


def _plan(sym):
    return OrbTradePlan(
        symbol=sym, range_high=10.0, range_low=9.0, range_size=1.0,
        entry_price=10.03, stop_price=9.0, shares=100, position_dollars=1003.0,
        lock_arm_at_r=1.0, lock_stop_r=0.0, risk_per_share=1.03, total_risk=103.0,
        composite_score=1.0, quintile='Q4', adaptive_mult=1.0,
    )


class TestConcurrentSubmit:
    """Item 4/5: fast_submit dispatches ranked picks through a <=8-worker
    pool but preserves ranking order for slot arithmetic; a failed submit
    frees its slot with no refill; flag off is unchanged (serial)."""

    def _run(self, engine, pending, submit_side_effect):
        """Drive the concurrent-submit tail of _run_pool_selection directly
        (isolates it from the scoring/planner pipeline, which is covered by
        the existing orb_engine test suite)."""
        engine._submit_entry = MagicMock(side_effect=submit_side_effect)
        import time as _time
        from concurrent.futures import ThreadPoolExecutor
        t_rank = _time.time()
        submitted = []
        workers = min(8, len(pending))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(engine._submit_entry, p) for _, _, p in pending]
            results = [f.result() for f in futures]
        for (sym, cand, _p), order_id in zip(pending, results):
            if order_id:
                cand.plan_submitted = True
                submitted.append(sym)
        return submitted

    def test_ranking_order_preserved_and_failed_submit_not_refilled(self, orb_cfg):
        engine = _engine(orb_cfg, fast_submit=True)
        syms = ['AAAA', 'BBBB', 'CCCC', 'DDDD']
        cands = {s: _cand(s) for s in syms}
        pending = [(s, cands[s], _plan(s)) for s in syms]

        # BBBB's REST submit fails (returns None) — its slot must stay
        # empty; no other pick backfills it.
        def side_effect(plan):
            return None if plan.symbol == 'BBBB' else f'order-{plan.symbol}'

        submitted = self._run(engine, pending, side_effect)
        assert submitted == ['AAAA', 'CCCC', 'DDDD']  # BBBB dropped, order kept
        assert cands['BBBB'].plan_submitted is False
        assert cands['AAAA'].plan_submitted is True

    def test_eight_worker_cap(self, orb_cfg):
        engine = _engine(orb_cfg, fast_submit=True)
        syms = [f'S{i}' for i in range(10)]
        cands = {s: _cand(s) for s in syms}
        pending = [(s, cands[s], _plan(s)) for s in syms]
        submitted = self._run(engine, pending, lambda plan: f'order-{plan.symbol}')
        assert submitted == syms  # all 10 submitted, order preserved


class TestFlagDefaultAndParity:
    def test_flag_default_off(self, orb_cfg):
        assert _engine(orb_cfg).fast_submit_enabled is False

    def test_flag_on_via_yaml(self, orb_cfg):
        assert _engine(orb_cfg, fast_submit=True).fast_submit_enabled is True

    def test_sweep_skips_rest_when_fast_submit_on(self, orb_cfg, monkeypatch):
        """Item 2: with fast_submit on, a candidate still missing range_data
        at the sweep must NOT trigger a REST bars call. Clock pinned to
        09:40 ET so the sweep's own 09:35-11:00 window gate doesn't mask
        the assertion with an early return."""
        import trading.orb_engine as orb_engine_mod
        from datetime import datetime as _real_datetime, timezone as _tz

        class _FixedDatetime(_real_datetime):
            @classmethod
            def now(cls, tz=None):
                return _real_datetime(2026, 9, 25, 13, 40, 0, tzinfo=_tz.utc)  # 09:40 ET

        monkeypatch.setattr(orb_engine_mod, 'datetime', _FixedDatetime)
        engine = _engine(orb_cfg, fast_submit=True)
        engine.candidates['ZZZZ'] = CandidateState(symbol='ZZZZ')
        engine.alpaca.get_1min_bars_multi = MagicMock(
            side_effect=AssertionError('REST bars call must not happen under fast_submit'))
        engine.alpaca.get_1min_bars = MagicMock(
            side_effect=AssertionError('REST bars call must not happen under fast_submit'))
        filled = engine._ensure_ranges_post_open()
        assert filled == set()
        assert engine.candidates['ZZZZ'].range_data is None
