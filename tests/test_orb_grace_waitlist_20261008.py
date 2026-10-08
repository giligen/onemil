"""First-rank GRACE must wait only for rankable names (docs/orb_grace_waitlist_fix_20261008.md).

10/8 and 10/6 incidents: the grace deferred ranking ~19 s because CRBP (already
PDR-vetoed at 09:34:57) and MIN (a 188-share phantom) were rangeless. Prior-day
vetoes, phantom prints and spread-gate rejects are never waited on.
"""
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
import copy

from trading.orb_engine import ORBEngine, RangeData
from trading.orb_planner import PlannerReject
from trading.stop_monitor import StopMonitor
from tests.test_orb_preplace_budget_20261006 import eng  # noqa: F401  (10/5 fixture)


@pytest.fixture
def engine():
    """Real orb.yaml engine with mocked IO (same wiring as test_orb_selection_race)."""
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    alpaca = MagicMock(spec=AlpacaClient)
    alpaca.get_open_positions.return_value = []
    alpaca.get_account_info.return_value = {'buying_power': 100_000.0}
    alpaca.get_latest_quote.return_value = {'bid_price': 9.0, 'ask_price': 10.0}
    db = MagicMock(spec=Database)
    db.get_open_trades.return_value = []
    sm = MagicMock(spec=StopMonitor)
    sm.drain_exit_events.return_value = []
    eng = ORBEngine(alpaca_client=alpaca, db=db, stop_monitor=sm, config=cfg)
    eng._symbols_entered_today_db = MagicMock(return_value=set())
    return eng


def _rng(sym):
    return RangeData(symbol=sym, range_high=10.0, range_low=9.5, range_volume=50_000,
                     range_avg_bar_range_pct=1.0, range_close=9.9,
                     range_start_ts=pd.Timestamp.utcnow(), range_open=9.6)


def _seed(engine, ranged, rangeless):
    engine.build_universe(source_loader=lambda: ranged + rangeless)
    for s in ranged:
        engine.candidates[s].range_data = _rng(s)


def _defer(engine):
    """_should_defer_first_rank at 09:35:06 ET (July -> EDT)."""
    at = datetime(2026, 7, 6, 13, 35, 6, tzinfo=timezone.utc)
    with patch('trading.orb_engine.datetime') as mdt:
        mdt.now.return_value = at
        mdt.combine = datetime.combine
        return engine._should_defer_first_rank()


class TestGraceSkipsUnwaitableNames:
    @pytest.mark.parametrize('veto', ['pdr_veto', 'g1_veto'])
    def test_prior_day_vetoed_rangeless_name_does_not_defer(self, engine, veto):
        _seed(engine, ['AAA'], ['CRBP'])
        engine._provisional_veto_reason['CRBP'] = veto
        assert _defer(engine) is False

    @pytest.mark.parametrize('veto', ['range_size_veto', 'catalyst_veto'])
    def test_range_dependent_vetoes_still_defer(self, engine, veto):
        _seed(engine, ['AAA'], ['CRBP'])
        engine._provisional_veto_reason['CRBP'] = veto
        assert _defer(engine) is True

    def test_188_share_phantom_does_not_defer(self, engine):
        _seed(engine, ['AAA'], ['MIN'])
        engine._snapshot_cache['MIN'] = {'volume': 188, 'open': 13.68}
        assert _defer(engine) is False

    def test_bar_stream_volume_overrides_a_stale_low_snapshot(self, engine):
        _seed(engine, ['AAA'], ['LATE'])
        engine._snapshot_cache['LATE'] = {'volume': 188, 'open': 5.0}
        engine._bar_windows['LATE'] = [{'volume': 40_000}, {'volume': 25_000}]
        assert _defer(engine) is True

    def test_no_volume_field_is_never_excluded(self, engine):
        _seed(engine, ['AAA'], ['NOVOL'])
        assert engine._candidate_session_volume('NOVOL') is None
        engine._snapshot_cache['NOVOL'] = {'open': 5.0}          # key absent
        assert _defer(engine) is True

    def test_wide_cached_spread_does_not_defer(self, engine):
        _seed(engine, ['AAA'], ['WIDE'])
        engine._spread_quote_cache['WIDE'] = (time.time(), engine.max_spread_bps + 100)
        assert _defer(engine) is False

    def test_stale_or_narrow_cached_spread_still_defers(self, engine):
        _seed(engine, ['AAA'], ['OLD', 'TIGHT'])
        engine._spread_quote_cache['OLD'] = (time.time() - 600, 5000.0)
        engine._spread_quote_cache['TIGHT'] = (time.time(), 20.0)
        assert _defer(engine) is True

    def test_get_spread_bps_records_the_live_quote(self, engine):
        spread = engine._get_spread_bps('ZZZ')                   # bid 9 / ask 10
        ts, cached = engine._spread_quote_cache['ZZZ']
        assert cached == pytest.approx(spread) and time.time() - ts < 5

    def test_real_rangeless_production_candidate_still_defers(self, engine):
        _seed(engine, ['AAA'], ['REAL'])
        engine._snapshot_cache['REAL'] = {'volume': 250_000, 'open': 6.0}
        assert _defer(engine) is True
        assert engine._first_rank_grace_waited_on == ['REAL']

    def test_waited_set_excludes_the_skipped_names(self, engine, caplog):
        _seed(engine, ['AAA'], ['CRBP', 'MIN', 'REAL'])
        engine._provisional_veto_reason['CRBP'] = 'pdr_veto'
        engine._snapshot_cache['MIN'] = {'volume': 188, 'open': 13.68}
        with caplog.at_level(logging.INFO):
            assert _defer(engine) is True
            _defer(engine)                                       # second tick
        assert engine._first_rank_grace_waited_on == ['REAL']
        assert 'still rangeless (REAL)' in caplog.text
        for sym in ('CRBP', 'MIN'):                              # one line per symbol per day
            assert caplog.text.count(f'GRACE skip {sym}') == 1

    def test_pool_names_never_defer(self, engine):
        _seed(engine, ['AAA'], ['POOLX'])
        engine._symbol_pool['POOLX'] = 'gap4_5'
        assert _defer(engine) is False


class TestDeferralClears:
    def test_clears_when_last_waitable_name_gets_its_range(self, engine):
        _seed(engine, ['AAA'], ['REAL', 'CRBP'])
        engine._provisional_veto_reason['CRBP'] = 'pdr_veto'
        assert _defer(engine) is True
        engine._first_rank_grace_end_utc = datetime(2099, 1, 1, tzinfo=timezone.utc)
        assert engine._first_rank_grace_elapsed() is False        # REAL still pending
        engine.candidates['REAL'].range_data = _rng('REAL')
        assert engine._first_rank_grace_elapsed() is True         # CRBP never blocks

    def test_empty_waitable_set_does_not_arm_the_deferral(self, engine):
        _seed(engine, ['AAA'], ['CRBP'])
        engine._provisional_veto_reason['CRBP'] = 'g1_veto'
        assert _defer(engine) is False
        assert engine._first_rank_defer_active is False


class TestTripwireNamesTheCause:
    def test_tripwire_line_carries_the_waited_on_set(self, engine, caplog):
        engine._first_submit_latency_logged = False
        engine.latency_warn_secs = 10.0
        engine._latency_phases = {'first_rank_grace': 19.2, 'universe_seed': 23.2}
        engine._first_rank_grace_waited_on = ['CRBP', 'MIN']
        et = datetime(2026, 10, 8, 9, 35, 26, tzinfo=timezone.utc)
        with patch.object(engine, '_et_now', return_value=et), \
                caplog.at_level(logging.WARNING):
            engine._check_first_submit_latency()
        assert 'first_rank_grace 19.2s waited on [CRBP,MIN]' in caplog.text


class TestDailyReset:
    def test_reset_clears_the_grace_state(self, engine):
        engine._provisional_veto_reason['CRBP'] = 'pdr_veto'
        engine._spread_quote_cache['X'] = (time.time(), 999.0)
        engine._grace_skip_logged.add('CRBP')
        engine._first_rank_grace_waited_on = ['CRBP']
        with engine._lock:
            engine._reset_daily_locked()
        assert engine._provisional_veto_reason == {}
        assert engine._spread_quote_cache == {}
        assert engine._grace_skip_logged == set()
        assert engine._first_rank_grace_waited_on == []


class TestPreplacePassRecordsVetoes:
    def test_provisional_rank_fills_the_reason_dict(self, eng):
        """Integration: the REAL 10/5 provisional pass (fixture of
        test_orb_preplace_budget_20261006) records each veto it decides."""
        eng._preplace_provisional_rank()
        assert eng._preplace_vetoed, "fixture must produce provisional vetoes"
        assert eng._provisional_veto_reason == eng._preplace_vetoed
        assert set(eng._provisional_veto_reason.values()) == {'pdr_veto'}
        # ...and stores the throwaway stand-in state next to each reason.
        assert set(eng._provisional_state) >= set(eng._provisional_veto_reason)
        # ...and the grace then refuses to wait on any of them.
        for sym in eng._provisional_veto_reason:
            eng.candidates[sym].range_data = None
            assert eng._grace_skip_reason(sym) is not None


# ----------------------------------------------------------------------------
# Stand-ins: a name we no longer WAIT for must still burn its top-K slot
# (BT ranks the full field; post-ranking veto / spread reject = slot burned,
# NEVER refilled - research/orb_machine_rules.md).
# ----------------------------------------------------------------------------
SCORE = {'VETO': 3.0, 'WIDE': 3.0, 'MIN': 3.0, 'A': 2.0, 'B': 1.0}
PDR = {'VETO': 5.0}          # <= 11 % -> PDR veto; everyone else passes


@pytest.fixture
def rank_engine(engine, monkeypatch):
    """Engine with a 2-slot book and a deterministic scorer; planner.build is a spy
    that rejects (so nothing is ever submitted) and records who reached it."""
    engine.max_concurrent = 2
    engine.dedup_by_family = engine.dedup_by_super_group = False
    engine.g1_veto_enabled = engine.range_size_veto_enabled = False
    engine.catalyst_veto_enabled = False
    engine.pdr_veto_enabled = True
    engine.pdr_veto_min_pct = 11.0
    engine._get_feature_context = lambda sym: {}
    engine._compute_features = lambda cand, **kw: {
        'gap_pct': 10.0, 'range_total_volume': 1e6,
        'prev_day_range_pct': PDR.get(cand.symbol, 25.0), 'score': SCORE[cand.symbol]}
    monkeypatch.setattr('trading.orb_engine.composite_score', lambda feats, zp: feats['score'])
    monkeypatch.setattr('trading.orb_engine.assign_quintile', lambda score, cuts: 'Q5')
    engine.filter_threshold = -99.0
    built = []
    engine.planner.build = lambda **kw: (built.append(kw['symbol']) or
                                         PlannerReject(kw['symbol'], 'spy', {}))
    engine._handle_reject = MagicMock()
    engine._built = built
    return engine


def _standin(engine, sym):
    """Store the 09:34:57-style throwaway state for a still-rangeless name."""
    temp = copy.copy(engine.candidates[sym])
    temp.range_data = _rng(sym)
    engine._provisional_state[sym] = temp
    return temp


def _select(engine, syms):
    return engine._run_pool_selection('production', syms, set(), None,
                                      dry_run=True, t_rank=time.time())


class TestStandInBurnsTheSlot:
    def test_vetoed_rangeless_name_ranks_and_burns_its_slot(self, rank_engine):
        e = rank_engine
        _seed(e, ['A', 'B'], ['VETO'])
        _standin(e, 'VETO')
        e._provisional_veto_reason['VETO'] = 'pdr_veto'
        _select(e, ['A', 'B'])
        # top-2 = [VETO (vetoed, slot burned), A]; B is NOT placed (no refill)
        assert e._built == ['A']
        assert 'VETO' in e._pdr_vetoed_today
        assert e.candidates['VETO'].plan_submitted is True      # real cand, late range cannot re-enter
        assert e.candidates['VETO'].rejected_reason == 'pdr_veto'
        assert e.candidates['VETO'].range_data is None          # provisional range never leaks

    def test_without_the_standin_the_next_name_would_refill(self, rank_engine):
        """Control: this is exactly the refill the fix closes."""
        e = rank_engine
        _seed(e, ['A', 'B'], ['VETO'])
        e._provisional_veto_reason['VETO'] = 'pdr_veto'          # no stored state
        _select(e, ['A', 'B'])
        assert e._built == ['A', 'B']

    def test_real_range_arriving_first_replaces_the_provisional_state(self, rank_engine):
        e = rank_engine
        _seed(e, ['A', 'B', 'VETO'], [])
        temp = _standin(e, 'VETO')
        e._provisional_veto_reason['VETO'] = 'pdr_veto'
        assert e._provisional_standins() == []
        _select(e, ['VETO', 'A', 'B'])
        assert e._built == ['A']                                  # real PDR veto, same burn
        assert e.candidates['VETO'].composite == SCORE['VETO']    # the REAL candidate was scored
        assert temp.composite is None                             # the stand-in never was

    def test_phantom_stays_out_of_the_field_entirely(self, rank_engine):
        e = rank_engine
        _seed(e, ['A', 'B'], ['MIN'])
        _standin(e, 'MIN')
        e._snapshot_cache['MIN'] = {'volume': 188, 'open': 13.68}
        assert e._provisional_standins() == []
        _select(e, ['A', 'B'])
        assert e._built == ['A', 'B']                             # MIN burned nothing

    def test_wide_spread_rangeless_name_burns_its_slot_and_is_never_placed(self, rank_engine):
        e = rank_engine
        _seed(e, ['A', 'B'], ['WIDE'])
        _standin(e, 'WIDE')
        e._spread_quote_cache['WIDE'] = (time.time(), e.max_spread_bps + 100)
        _select(e, ['A', 'B'])
        assert e._built == ['A']                                  # WIDE never reaches the planner
        assert e.candidates['WIDE'].rejected_reason == 'standin_spread_slot_burn'
        assert e.candidates['WIDE'].range_data is None

    def test_reconcile_ranking_includes_the_standin(self, rank_engine):
        e = rank_engine
        _seed(e, ['A', 'B'], ['VETO'])
        _standin(e, 'VETO')
        e._provisional_veto_reason['VETO'] = 'pdr_veto'
        assert [c.symbol for c in e._rank_production_final()][:2] == ['VETO', 'A']

    def test_daily_reset_drops_the_standin_state(self, rank_engine):
        e = rank_engine
        _seed(e, ['A'], ['VETO'])
        _standin(e, 'VETO')
        with e._lock:
            e._reset_daily_locked()
        assert e._provisional_state == {}
