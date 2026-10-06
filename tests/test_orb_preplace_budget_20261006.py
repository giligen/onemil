"""ORB preplace budget fix (docs/review_20261006/FIX_engine_preplace_budget_spec.md).

10/5/2026 defect: the provisional preplace pass vetoed six names and the veto
methods recorded them in `_pdr_vetoed_today`, so the 09:35:25 selection had
budget 8 - 6 = 2, re-vetoed SUPV/HOG and never reached DFDV (BT's only pick).
The reconcile also ranked only the preplaced names among themselves.

Fixture = the real 10/5 journal: final SCORED comps / quintiles of the top-12
(logs/orb_selection_audit.jsonl 13:35:25), prev_day_range_pct from the 10/5
features CSV (the six live-vetoed names carry the journal's PDR VETO values),
provisional top-8 = EWZS BCYC PURR SUPV HOG ALVO JAGX CRCG.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

import trading.orb_engine as orb_engine_mod
from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import CandidateState, OpenPosition, ORBEngine, RangeData
from trading.orb_planner import OrbTradePlan
from trading.stop_monitor import StopMonitor

# (final composite, quintile, prev_day_range_pct) — audit record 13:35:25 UTC.
FINAL = {
    'SUPV': (0.4208, 'Q4', 3.94), 'HOG': (0.4114, 'Q4', 2.46),
    'BNC': (0.4025, 'Q4', 8.03), 'ELPC': (0.3392, 'Q4', 4.43),
    'GGB': (0.3377, 'Q4', 4.09), 'EWZS': (0.3330, 'Q4', 3.87),
    'DFDV': (0.3216, 'Q4', 12.57), 'BCYC': (0.3116, 'Q4', 10.80),
    'PAX': (0.2968, 'Q4', 15.0), 'UGP': (0.2967, 'Q4', 15.0),
    'JAGX': (0.2964, 'Q4', 15.0), 'PURR': (0.5562, 'Q5', 9.63),
    'ALVO': (0.5324, 'Q5', 4.95), 'VIV': (0.2769, 'Q3', 15.0),
    'CRCG': (0.2410, 'Q3', 15.0), 'MSTX': (0.2119, 'Q3', 15.0),
}
PROV_TOP8 = ['EWZS', 'BCYC', 'PURR', 'SUPV', 'HOG', 'ALVO', 'JAGX', 'CRCG']
LIVE_VETOED = {'EWZS', 'BCYC', 'PURR', 'SUPV', 'HOG', 'ALVO'}   # journal 13:34:57
BT_VETOED = {'SUPV', 'HOG', 'BNC', 'ELPC', 'GGB', 'EWZS', 'BCYC'}  # final top-8 minus DFDV
PROV_RANGE_OPEN = 1.0     # marks a provisional RangeData
FINAL_RANGE_OPEN = 10.0   # marks a final RangeData


def _range(sym, range_open, range_high=10.5):
    return RangeData(
        symbol=sym, range_high=range_high, range_low=9.9, range_volume=500_000,
        range_avg_bar_range_pct=1.0, range_close=range_high - 0.02,
        range_start_ts=pd.Timestamp.utcnow(), range_open=range_open)


def _plan(sym, range_high=10.5):
    return OrbTradePlan(
        symbol=sym, range_high=range_high, range_low=9.9, range_size=range_high - 9.9,
        entry_price=range_high + 0.03, stop_price=9.9, shares=100,
        position_dollars=(range_high + 0.03) * 100, lock_arm_at_r=1.0, lock_stop_r=0.0,
        risk_per_share=range_high - 9.9 + 0.03, total_risk=(range_high - 9.9 + 0.03) * 100,
        composite_score=1.0, quintile='Q4', adaptive_mult=1.0)


def _prov_score(sym):
    """Provisional composite/quintile: the 8 provisional names Q4 (in
    PROV_TOP8 order), every other name Q3 low — so provisional top-8 = PROV_TOP8."""
    if sym in PROV_TOP8:
        return 0.9 - 0.05 * PROV_TOP8.index(sym), 'Q4'
    return 0.1, 'Q3'


@pytest.fixture
def eng(monkeypatch):
    """Live-config engine, 10/5 world: PDR veto at 11 % only, N = 8, final
    scores / PDRs from the journal, provisional scores keyed off range_open."""
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    cfg['entry'] = dict(cfg.get('entry') or {})
    cfg['entry'].update(preplace_at_close=True, preplace_rank_lead_s=3.0,
                        preplace_submit_delay_s=0.0)
    alpaca = MagicMock(spec=AlpacaClient)
    alpaca.get_account_info.return_value = {'buying_power': 5_000_000.0}
    alpaca.get_latest_quote.return_value = {'bid_price': 9.95, 'ask_price': 10.00}
    alpaca.get_snapshots.return_value = {}
    e = ORBEngine(alpaca_client=alpaca, db=MagicMock(spec=Database),
                  stop_monitor=MagicMock(spec=StopMonitor), config=cfg)
    e.max_concurrent = 8
    e.filter_threshold = 0.0
    e.skip_q1 = True
    e.dedup_by_family = False
    e.dedup_by_super_group = False
    e.ranking_order = ['Q4', 'Q5', 'Q3', 'Q2', 'Q1']
    e.pdr_veto_enabled, e.pdr_veto_min_pct = True, 11.0
    e.g1_veto_enabled = e.range_size_veto_enabled = e.catalyst_veto_enabled = False
    e.universe_min_gap_pct = 0.0
    e._get_feature_context = lambda sym: {}

    def feats(cand, prev_day_bar=None, daily_stats_20d=None):
        return {'_sym': cand.symbol, '_prov': cand.range_data.range_open == PROV_RANGE_OPEN,
                'gap_pct': 6.0, 'range_total_volume': 1e5,
                'prev_day_range_pct': FINAL[cand.symbol][2]}
    monkeypatch.setattr(e, '_compute_features', feats)

    def comp(f, z_params):
        return (_prov_score(f['_sym'])[0] if f['_prov'] else FINAL[f['_sym']][0])
    monkeypatch.setattr(orb_engine_mod, 'composite_score', comp)
    # assign_quintile(score, cutoffs): provisional scores and final scores are
    # disjoint keys; look the quintile up from whichever table owns the score.
    q_by_score = {round(v[0], 4): v[1] for v in FINAL.values()}
    q_by_score.update({round(_prov_score(s)[0], 4): 'Q4' for s in PROV_TOP8})
    q_by_score[0.1] = 'Q3'
    monkeypatch.setattr(orb_engine_mod, 'assign_quintile',
                        lambda score, cutoffs: q_by_score[round(score, 4)])
    e.planner = MagicMock()
    e.planner.build.side_effect = lambda **kw: _plan(kw['symbol'], kw['range_high'])
    e._cancel_symbol_open_orders = MagicMock(return_value=1)
    e._provisional_range_for = lambda sym, snap: _range(sym, PROV_RANGE_OPEN)
    for sym in FINAL:
        e.candidates[sym] = CandidateState(symbol=sym)
        e.candidates[sym].range_data = _range(sym, PROV_RANGE_OPEN)
    return e


def _fake_submit(eng, ok=()):
    """`_submit_entry` stand-in: names in `ok` succeed (and, like the real
    method, register an open position); all others fail (return None)."""
    def _submit(plan):
        if plan.symbol not in eng._submit_entry.ok:
            return None
        eng.open_positions[plan.symbol] = OpenPosition(
            symbol=plan.symbol, entry_price=plan.entry_price, stop_price=9.9, shares=100,
            trade_id=1, order_id=f'order-{plan.symbol}',
            entry_time=pd.Timestamp.utcnow().to_pydatetime(), range_high=plan.range_high,
            range_low=9.9, lock_arm_at_r=1.0, lock_stop_r=0.0, composite_score=1.0,
            quintile='Q4')
        return f'order-{plan.symbol}'
    eng._submit_entry = MagicMock(side_effect=_submit)
    eng._submit_entry.ok = set(ok)
    return eng._submit_entry


def _go_final(eng):
    """The 09:35 bars arrive: every candidate gets its FINAL range."""
    for sym, cand in eng.candidates.items():
        cand.range_data = _range(sym, FINAL_RANGE_OPEN)


def _final_pass(eng):
    """The normal selection pass exactly as `_check_entries_locked` calls it."""
    if not hasattr(eng._submit_entry, 'ok'):
        _fake_submit(eng)
    eng._submit_entry.ok = set(FINAL)            # the final pass submits fine
    syms = [s for s, c in eng.candidates.items()
            if not c.plan_submitted and c.range_data is not None
            and s not in eng.open_positions]
    return eng._run_pool_selection('production', syms, set(), None,
                                   dry_run=False, t_rank=time.time())


def _preplace(eng, ok=()):
    """Provisional rank + 09:35:00.0 submit; returns the submit mock."""
    eng._preplace_provisional_rank()
    submit = _fake_submit(eng, ok=ok)
    eng._preplace_submit_at_close()
    return submit


# ---------------------------------------------------------------- (a) 10/5
class TestOct5Fixture:
    def test_final_picks_are_dfdv_and_vetoed_set_is_the_bt_set(self, eng):
        _preplace(eng, ok=())                      # live: JAGX/CRCG submits failed
        assert set(eng._preplace_state) == {'JAGX', 'CRCG'}
        _go_final(eng)
        counters = eng._reconcile_preplaced()
        assert counters['n_cancelled'] == 0        # failed submits: nothing resting
        picks = _final_pass(eng)
        assert picks == ['DFDV']
        assert eng._pdr_vetoed_today == BT_VETOED
        assert len(eng._pdr_vetoed_today) == 7

    def test_success_path_cancels_outside_names_and_still_reaches_dfdv(self, eng):
        """JAGX/CRCG SUBMIT OK on Monday's boot: both are outside the final
        top-8, get cancelled, and must not shrink the final budget to 6."""
        _preplace(eng, ok=('JAGX', 'CRCG'))
        assert set(eng.open_positions) == {'JAGX', 'CRCG'}
        _go_final(eng)
        counters = eng._reconcile_preplaced()
        assert counters['n_cancelled'] == 2 and counters['n_kept'] == 0
        assert set(eng.open_positions) == set()
        assert {'JAGX', 'CRCG'} <= eng._pdr_vetoed_today      # never re-entered
        assert _final_pass(eng) == ['DFDV']
        assert BT_VETOED <= eng._pdr_vetoed_today


# ------------------------------------------------- (b) provisional vetoes
class TestProvisionalVetoesNeverTouchTheBudget:
    def test_provisional_pass_leaves_day_level_state_untouched(self, eng, caplog):
        with caplog.at_level(logging.INFO):
            eng._preplace_provisional_rank()
        assert set(eng._preplace_vetoed) == LIVE_VETOED
        assert set(eng._preplace_vetoed.values()) == {'pdr_veto'}
        assert eng._pdr_vetoed_today == set()
        assert all(not eng.candidates[s].plan_submitted for s in FINAL)
        budget = eng.max_concurrent - len(
            eng._symbols_entered_today_db() | eng._pdr_vetoed_today | set(eng.open_positions))
        assert budget == 8
        assert any('[ORB PREPLACE] provisional PDR VETO: EWZS' in r.message
                   for r in caplog.records)
        assert not any(r.message.startswith('[ORB] PDR VETO') for r in caplog.records)

    @pytest.mark.parametrize('method,setup', [
        ('_pdr_veto_reject', lambda e, c: setattr(c, 'features', {'prev_day_range_pct': 1.0})),
        ('_g1_veto_reject', lambda e, c: (setattr(e, 'g1_veto_enabled', True),
                                          setattr(c, 'features', {'return_volatility_20d': 0.1,
                                                                  'prev_day_range_pct': 0.1}))),
        ('_range_size_veto_reject', lambda e, c: (setattr(e, 'range_size_veto_enabled', True),
                                                  setattr(c, 'features', {'range_size_pct': 0.0}))),
    ])
    def test_record_false_is_pure_and_record_true_records(self, eng, method, setup):
        cand = CandidateState(symbol='ZZZZ')
        setup(eng, cand)
        assert getattr(eng, method)(cand, record=False) is True
        assert eng._pdr_vetoed_today == set() and cand.plan_submitted is False
        assert cand.rejected_reason in (None, '')
        assert getattr(eng, method)(cand) is True            # default = recorded
        assert eng._pdr_vetoed_today == {'ZZZZ'} and cand.plan_submitted is True

    def test_catalyst_record_false_is_pure(self, eng, monkeypatch):
        eng.catalyst_veto_enabled = True
        monkeypatch.setattr(orb_engine_mod, 'catalyst_veto_applies', lambda *a, **k: True)
        monkeypatch.setattr(eng, '_get_has_news', lambda s: False)
        monkeypatch.setattr(eng, '_anchor_for', lambda s, allow_api=False: None)
        cand = CandidateState(symbol='ZZZZ')
        assert eng._catalyst_veto_reject(cand, cohort_symbols=['ZZZZ'], record=False) is True
        assert eng._pdr_vetoed_today == set() and cand.plan_submitted is False
        assert eng._catalyst_veto_reject(cand, cohort_symbols=['ZZZZ']) is True
        assert eng._pdr_vetoed_today == {'ZZZZ'}


# --------------------------------------------- (c) reconcile vs full top-N
class TestReconcileAgainstFullTopN:
    def test_outside_cancelled_inside_kept_no_double_submit(self, eng):
        # DFDV (final rank 7, inside) and JAGX (final rank 11, outside) both rest.
        for sym in ('DFDV', 'JAGX'):
            eng._preplace_state[sym] = {
                'plan': _plan(sym), 'provisional_range_high': 10.5,
                'provisional_range_low': 9.9, 'submitted': False, 'order_id': None}
            eng.candidates[sym].range_data = _range(sym, PROV_RANGE_OPEN)
        submit = _fake_submit(eng, ok=('DFDV', 'JAGX'))
        eng._preplace_submit_at_close()
        assert submit.call_count == 2
        _go_final(eng)
        counters = eng._reconcile_preplaced()
        assert counters['n_kept'] == 1 and counters['n_cancelled'] == 1
        eng._cancel_symbol_open_orders.assert_called_once_with('JAGX')
        assert set(eng.open_positions) == {'DFDV'}
        assert 'JAGX' in eng._pdr_vetoed_today
        assert eng.candidates['JAGX'].rejected_reason == 'preplace_dropped_final_topk'
        picks = _final_pass(eng)
        assert picks == []                       # DFDV already rests; the rest are vetoed
        assert submit.call_count == 2            # no double submit of DFDV
        assert BT_VETOED <= eng._pdr_vetoed_today

    def test_reconcile_ranks_all_candidates_not_just_preplaced(self, eng):
        """The pre-fix form kept ANY preplaced name (top-len(preplaced) of
        themselves). JAGX is 11th overall: it must be cancelled even when it
        is the only preplaced name."""
        eng._preplace_state['JAGX'] = {
            'plan': _plan('JAGX'), 'provisional_range_high': 10.5,
            'provisional_range_low': 9.9, 'submitted': False, 'order_id': None}
        _fake_submit(eng, ok=('JAGX',))
        eng._preplace_submit_at_close()
        _go_final(eng)
        assert eng._reconcile_preplaced()['n_cancelled'] == 1


# ------------------------------------------------- (d) failed submit retry
class TestFailedSubmitReEvaluated:
    def test_failed_inside_top_n_is_submitted_by_final_pass(self, eng):
        eng.pdr_veto_enabled = False             # DFDV is inside and unvetoed
        eng._preplace_state['DFDV'] = {
            'plan': _plan('DFDV'), 'provisional_range_high': 10.5,
            'provisional_range_low': 9.9, 'submitted': False, 'order_id': None}
        submit = _fake_submit(eng, ok=())        # preplace submit FAILS
        eng._preplace_submit_at_close()
        assert eng._preplace_state['DFDV']['submitted'] is False
        assert eng.candidates['DFDV'].plan_submitted is False
        _go_final(eng)
        counters = eng._reconcile_preplaced()
        assert counters['n_cancelled'] == 0 and counters['n_kept'] == 0
        eng._cancel_symbol_open_orders.assert_not_called()
        picks = _final_pass(eng)
        assert 'DFDV' in picks

    def test_failed_outside_top_n_is_neither_cancelled_nor_submitted(self, eng):
        eng.pdr_veto_enabled = False
        eng._preplace_state['JAGX'] = {
            'plan': _plan('JAGX'), 'provisional_range_high': 10.5,
            'provisional_range_low': 9.9, 'submitted': False, 'order_id': None}
        submit = _fake_submit(eng, ok=())
        eng._preplace_submit_at_close()
        _go_final(eng)
        eng._reconcile_preplaced()
        picks = _final_pass(eng)
        assert 'JAGX' not in picks and len(picks) == 8   # BT top-8, nothing else
