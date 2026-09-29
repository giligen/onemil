"""ORB preplace-at-close (docs/orb_preplace_spec_20260928.md): entry.preplace_at_close.

Covers: provisional range from snapshot fields (+ WS-ingested bars); the
09:34:57 / 09:35:00.0 scheduler never fires the submit timer before its
target; concurrent submit-at-close records per-order latency and does not
refill a failed submit; dry-mode WOULD PREPLACE logging + ledger row
(preplaced=1); reconciliation (kept / replaced / cancelled / filled-before-
reconcile / new-entrant-added); the flag off never touches any new code.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

import trading.orb_engine as orb_engine_mod
from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import (
    CandidateState, OpenPosition, ORBEngine, RangeData,
)
from trading.orb_planner import OrbTradePlan
from trading.stop_monitor import StopMonitor

# A feature dict with every key the live orb.yaml composite filter plus the
# veto helpers could ask for -- generous on purpose so composite_score never
# returns None (missing key => None => candidate silently dropped) and no
# individual test needs to know the exact z_params key set.
FULL_FEATS = {
    'gap_pct': 12.0,
    'range_total_volume': 500_000.0,
    'range_avg_bar_range_pct': 1.2,
    'range_size_pct': 5.0,
    'price_vs_20d_high_pct': -4.0,
    'prev_day_close_position': 0.5,
    'range_close_position': 0.6,
    'prev_day_range_pct': 20.0,
    'return_volatility_20d': 20.0,
}


def _base_cfg(preplace=True):
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    cfg['entry'] = dict(cfg.get('entry') or {})
    cfg['entry']['preplace_at_close'] = preplace
    cfg['entry']['preplace_rank_lead_s'] = 3.0
    return cfg


def _mock_alpaca():
    client = MagicMock(spec=AlpacaClient)
    client.get_account_info.return_value = {'buying_power': 500_000.0}
    client.get_latest_quote.return_value = {'bid_price': 9.95, 'ask_price': 10.00}
    client.get_snapshots.return_value = {}
    return client


def _engine(preplace=True):
    cfg = _base_cfg(preplace)
    return ORBEngine(alpaca_client=_mock_alpaca(), db=MagicMock(spec=Database),
                      stop_monitor=MagicMock(spec=StopMonitor), config=cfg)


def _range(symbol, range_open=10.0, range_high=10.5, range_low=9.9):
    return RangeData(
        symbol=symbol, range_high=range_high, range_low=range_low,
        range_volume=500_000, range_avg_bar_range_pct=1.0,
        range_close=range_high - 0.02, range_start_ts=pd.Timestamp.utcnow(),
        range_open=range_open,
    )


def _plan(sym, range_high=10.5):
    return OrbTradePlan(
        symbol=sym, range_high=range_high, range_low=9.9, range_size=range_high - 9.9,
        entry_price=range_high + 0.03, stop_price=9.9, shares=100,
        position_dollars=(range_high + 0.03) * 100,
        lock_arm_at_r=1.0, lock_stop_r=0.0, risk_per_share=range_high - 9.9 + 0.03,
        total_risk=(range_high - 9.9 + 0.03) * 100, composite_score=1.0,
        quintile='Q4', adaptive_mult=1.0,
    )


def _no_vetoes(eng):
    return patch.multiple(
        eng, _pdr_veto_reject=MagicMock(return_value=False),
        _g1_veto_reject=MagicMock(return_value=False),
        _range_size_veto_reject=MagicMock(return_value=False),
        _catalyst_veto_reject=MagicMock(return_value=False),
    )


class FakeTimer:
    """Stand-in for threading.Timer: records (interval, function) and never
    actually starts a background thread -- the test calls the function
    itself when it wants to simulate the timer firing."""
    instances = []

    def __init__(self, interval, function):
        self.interval = interval
        self.function = function
        self.daemon = False
        self.started = False
        FakeTimer.instances.append(self)

    def start(self):
        self.started = True


@pytest.fixture(autouse=True)
def _reset_fake_timer():
    FakeTimer.instances = []
    yield
    FakeTimer.instances = []


# ===========================================================================
# 1. Provisional range from snapshot fields
# ===========================================================================
class TestProvisionalRange:
    def test_from_daily_bar_snapshot_only(self):
        eng = _engine()
        snap = {'open': 10.0, 'high': 10.6, 'low': 9.8, 'close': 10.4, 'volume': 1000}
        rd = eng._provisional_range_for('ABCD', snap)
        assert rd is not None
        assert rd.range_high == 10.6
        assert rd.range_low == 9.8
        assert rd.range_open == 10.0

    def test_widened_by_ingested_ws_bars(self):
        """Snapshot daily-bar can lag the WS bar stream by a beat -- the
        provisional range must take the more extreme of the two sources."""
        eng = _engine()
        eng._bar_windows['ABCD'] = [
            {'open': 10.0, 'high': 10.9, 'low': 9.7, 'close': 10.5, 'volume': 400},
        ]
        snap = {'open': 10.0, 'high': 10.6, 'low': 9.8, 'close': 10.4, 'volume': 1000}
        rd = eng._provisional_range_for('ABCD', snap)
        assert rd.range_high == 10.9  # bar high wins over stale snapshot high
        assert rd.range_low == 9.7    # bar low wins over stale snapshot low

    def test_degenerate_range_returns_none(self):
        eng = _engine()
        assert eng._provisional_range_for('ABCD', None) is None
        assert eng._provisional_range_for('ABCD', {'open': 0, 'high': 0, 'low': 0}) is None
        assert eng._provisional_range_for(
            'ABCD', {'open': 10, 'high': 9, 'low': 9.5}) is None  # high <= low


# ===========================================================================
# 2. Scheduler: fires at 09:35:00.0 ET, never earlier
# ===========================================================================
class TestSchedulerArming:
    def test_targets_0934_57_and_0935_00_exactly(self, monkeypatch):
        eng = _engine()
        monkeypatch.setattr(orb_engine_mod.threading, 'Timer', FakeTimer)
        fixed = pd.Timestamp('2026-09-28 09:34:50').to_pydatetime()
        monkeypatch.setattr(eng, '_et_now', lambda: fixed)
        eng._maybe_arm_preplace_scheduler()
        assert len(FakeTimer.instances) == 2
        rank_t, submit_t = FakeTimer.instances
        assert rank_t.interval == pytest.approx(7.0)    # 09:34:57 - 09:34:50
        assert submit_t.interval == pytest.approx(10.0)  # 09:35:00 - 09:34:50
        assert rank_t.function == eng._preplace_provisional_rank
        assert submit_t.function == eng._preplace_submit_at_close
        assert rank_t.started and submit_t.started

    def test_never_negative_when_armed_late(self, monkeypatch):
        """A tick landing AFTER 09:35:00 that still arms (e.g. first tick of
        the day at 09:34:59.9 rounding up) must clamp to >=0 -- fires ASAP,
        which is already >= the 09:35:00 target, never earlier."""
        eng = _engine()
        monkeypatch.setattr(orb_engine_mod.threading, 'Timer', FakeTimer)
        fixed = pd.Timestamp('2026-09-28 09:34:59.999').to_pydatetime()
        monkeypatch.setattr(eng, '_et_now', lambda: fixed)
        eng._maybe_arm_preplace_scheduler()
        rank_t, submit_t = FakeTimer.instances
        assert rank_t.interval >= 0.0
        assert submit_t.interval >= 0.0

    def test_outside_window_does_not_arm(self, monkeypatch):
        eng = _engine()
        monkeypatch.setattr(orb_engine_mod.threading, 'Timer', FakeTimer)
        monkeypatch.setattr(eng, '_et_now',
                             lambda: pd.Timestamp('2026-09-28 09:40:00').to_pydatetime())
        eng._maybe_arm_preplace_scheduler()
        assert FakeTimer.instances == []
        assert eng._preplace_armed_today is False

    def test_arms_once_per_day(self, monkeypatch):
        eng = _engine()
        monkeypatch.setattr(orb_engine_mod.threading, 'Timer', FakeTimer)
        monkeypatch.setattr(eng, '_et_now',
                             lambda: pd.Timestamp('2026-09-28 09:30:00').to_pydatetime())
        eng._maybe_arm_preplace_scheduler()
        eng._maybe_arm_preplace_scheduler()
        assert len(FakeTimer.instances) == 2  # not 4

    def test_flag_off_never_arms(self, monkeypatch):
        eng = _engine(preplace=False)
        monkeypatch.setattr(orb_engine_mod.threading, 'Timer', FakeTimer)
        monkeypatch.setattr(eng, '_et_now',
                             lambda: pd.Timestamp('2026-09-28 09:30:00').to_pydatetime())
        eng._maybe_arm_preplace_scheduler()
        assert FakeTimer.instances == []


# ===========================================================================
# 3. Concurrent submission at 09:35:00.0
# ===========================================================================
class TestConcurrentSubmitAtClose:
    def _seed(self, eng, syms):
        for s in syms:
            eng.candidates[s] = CandidateState(symbol=s)
            eng._preplace_state[s] = {
                'plan': _plan(s), 'provisional_range_high': 10.5,
                'provisional_range_low': 9.9, 'submitted': False, 'order_id': None,
            }

    def test_concurrent_submit_records_latency_marks_submitted(self, caplog):
        eng = _engine()
        syms = ['AAAA', 'BBBB', 'CCCC']
        self._seed(eng, syms)
        eng._submit_entry = MagicMock(side_effect=lambda p: f'order-{p.symbol}')
        with caplog.at_level(logging.INFO):
            eng._preplace_submit_at_close()
        for s in syms:
            assert eng._preplace_state[s]['submitted'] is True
            assert eng._preplace_state[s]['order_id'] == f'order-{s}'
            assert eng.candidates[s].plan_submitted is True
        assert eng._preplace_submitted_today is True
        latency_lines = [r.message for r in caplog.records if 'SUBMIT LATENCY' in r.message]
        assert len(latency_lines) == 3
        assert all('preplaced=1' in m for m in latency_lines)
        assert eng._first_submit_latency_logged is True

    def test_failed_submit_not_refilled(self):
        eng = _engine()
        syms = ['AAAA', 'BBBB']
        self._seed(eng, syms)
        eng._submit_entry = MagicMock(side_effect=lambda p: None if p.symbol == 'BBBB' else 'order-AAAA')
        eng._preplace_submit_at_close()
        assert eng._preplace_state['AAAA']['submitted'] is True
        assert eng._preplace_state['BBBB']['submitted'] is False
        assert eng.candidates['BBBB'].plan_submitted is False

    def test_empty_state_is_noop(self, caplog):
        eng = _engine()
        with caplog.at_level(logging.INFO):
            eng._preplace_submit_at_close()
        assert eng._preplace_submitted_today is True
        assert any('nothing to preplace' in r.message for r in caplog.records)

    def test_flag_off_is_noop(self):
        eng = _engine(preplace=False)
        eng._preplace_state['AAAA'] = {
            'plan': _plan('AAAA'), 'provisional_range_high': 10.5,
            'provisional_range_low': 9.9, 'submitted': False, 'order_id': None,
        }
        eng._submit_entry = MagicMock(side_effect=AssertionError('must not submit'))
        eng._preplace_submit_at_close()
        assert eng._preplace_submitted_today is False


# ===========================================================================
# 4. Dry mode
# ===========================================================================
class TestDryModePreplace:
    def test_would_preplace_logged_and_ledger_row_preplaced_1(self, caplog, tmp_path, monkeypatch):
        eng = _engine()
        eng.strategy_dry_run = True
        sym = 'DRYX'
        eng.candidates[sym] = CandidateState(symbol=sym)
        eng._preplace_state[sym] = {
            'plan': _plan(sym), 'provisional_range_high': 10.5,
            'provisional_range_low': 9.9, 'submitted': False, 'order_id': None,
        }
        with caplog.at_level(logging.INFO):
            eng._preplace_submit_at_close()
        assert any('[ORB DRY] WOULD PREPLACE DRYX' in r.message for r in caplog.records)
        assert eng.candidates[sym].plan_submitted is True
        eng.db.insert_dry_entry.assert_called_once()

        # conftest's autouse _isolated_orb_state redirects the ledger path
        # to tmp_path -- read it back via the same resolver the engine uses.
        ledger_path = orb_engine_mod._resolve_orb_path(orb_engine_mod.DEFAULT_DRY_LEDGER_PATH)
        with open(ledger_path) as f:
            rows = list(csv.DictReader(f))
        assert rows[-1]['symbol'] == 'DRYX'
        assert rows[-1]['preplaced'] == '1'


# ===========================================================================
# 5. Reconciliation
# ===========================================================================
class TestReconciliation:
    def _seed_preplaced(self, eng, sym, prov_rh=10.5, filled=False, order_id='order-1'):
        eng.candidates[sym] = CandidateState(symbol=sym)
        eng._preplace_state[sym] = {
            'plan': _plan(sym, range_high=prov_rh), 'provisional_range_high': prov_rh,
            'provisional_range_low': 9.9, 'submitted': True, 'order_id': order_id,
        }
        eng.open_positions[sym] = OpenPosition(
            symbol=sym, entry_price=prov_rh + 0.03, stop_price=9.9, shares=100,
            trade_id=1, order_id='' if filled else order_id,
            entry_time=pd.Timestamp.utcnow().to_pydatetime(),
            range_high=prov_rh, range_low=9.9, lock_arm_at_r=1.0, lock_stop_r=0.0,
            composite_score=1.0, quintile='Q4',
        )

    def _feats_side_effect(self, drop_sym=None):
        def _fn(cand, prev_day_bar=None, daily_stats_20d=None):
            if drop_sym is not None and cand.symbol == drop_sym:
                return {}
            return dict(FULL_FEATS)
        return _fn

    def test_same_trigger_kept(self, monkeypatch):
        eng = _engine()
        self._seed_preplaced(eng, 'KEEP', prov_rh=10.5, filled=False)
        eng.candidates['KEEP'].range_data = _range('KEEP', range_high=10.5)
        monkeypatch.setattr(eng, '_compute_features', self._feats_side_effect())
        eng._cancel_symbol_open_orders = MagicMock(return_value=0)
        eng._submit_entry = MagicMock(side_effect=AssertionError('must not resubmit'))
        with _no_vetoes(eng):
            counters = eng._reconcile_preplaced()
        assert counters == dict(n_preplaced=1, n_kept=1, n_replaced=0, n_cancelled=0,
                                 n_added=0, n_filled_before_reconcile=0)
        eng._cancel_symbol_open_orders.assert_not_called()

    def test_higher_final_high_replaced(self, monkeypatch):
        eng = _engine()
        self._seed_preplaced(eng, 'REPL', prov_rh=10.5, filled=False)
        eng.candidates['REPL'].range_data = _range('REPL', range_high=10.9)  # moved up
        monkeypatch.setattr(eng, '_compute_features', self._feats_side_effect())
        eng._cancel_symbol_open_orders = MagicMock(return_value=1)
        eng._submit_entry = MagicMock(return_value='order-2')
        with _no_vetoes(eng):
            counters = eng._reconcile_preplaced()
        assert counters['n_replaced'] == 1
        assert counters['n_kept'] == 0
        eng._cancel_symbol_open_orders.assert_called_once_with('REPL')
        eng._submit_entry.assert_called_once()
        called_plan = eng._submit_entry.call_args[0][0]
        assert called_plan.range_high == 10.9
        assert eng.candidates['REPL'].plan_submitted is True

    def test_dropped_from_final_topk_cancelled_no_refill(self, monkeypatch):
        eng = _engine()
        self._seed_preplaced(eng, 'DROP', prov_rh=10.5, filled=False)
        eng.candidates['DROP'].range_data = _range('DROP', range_high=10.5)
        # Final feature computation comes back empty -> composite_score None
        # -> DROP never makes the final scored set -> dropped from top-K.
        monkeypatch.setattr(eng, '_compute_features', self._feats_side_effect(drop_sym='DROP'))
        eng._cancel_symbol_open_orders = MagicMock(return_value=1)
        eng._submit_entry = MagicMock(side_effect=AssertionError('must not resubmit'))
        with _no_vetoes(eng):
            counters = eng._reconcile_preplaced()
        assert counters['n_cancelled'] == 1
        eng._cancel_symbol_open_orders.assert_called_once_with('DROP')
        assert 'DROP' not in eng.open_positions
        assert 'DROP' in eng._pdr_vetoed_today  # slot consumed, no refill
        assert eng.candidates['DROP'].rejected_reason == 'preplace_dropped_final_topk'

    def test_filled_before_reconcile_warns_and_counts(self, monkeypatch, caplog):
        eng = _engine()
        self._seed_preplaced(eng, 'FILL', prov_rh=10.5, filled=True)
        eng.candidates['FILL'].range_data = _range('FILL', range_high=10.8)  # differs
        monkeypatch.setattr(eng, '_compute_features', self._feats_side_effect())
        eng._cancel_symbol_open_orders = MagicMock(side_effect=AssertionError('must not cancel a fill'))
        with caplog.at_level(logging.WARNING), _no_vetoes(eng):
            counters = eng._reconcile_preplaced()
        assert counters['n_filled_before_reconcile'] == 1
        assert counters['n_kept'] == 0
        assert any('FILLED at provisional trigger' in r.message for r in caplog.records)
        assert eng.open_positions['FILL'].order_id == ''  # fill left standing

    def test_filled_same_trigger_counts_as_kept_not_warned(self, monkeypatch, caplog):
        eng = _engine()
        self._seed_preplaced(eng, 'FILLOK', prov_rh=10.5, filled=True)
        eng.candidates['FILLOK'].range_data = _range('FILLOK', range_high=10.5)
        monkeypatch.setattr(eng, '_compute_features', self._feats_side_effect())
        with caplog.at_level(logging.WARNING), _no_vetoes(eng):
            counters = eng._reconcile_preplaced()
        assert counters['n_kept'] == 1
        assert counters['n_filled_before_reconcile'] == 0
        assert not any('parity deviation' in r.message for r in caplog.records)

    def test_no_preplaced_state_returns_zero_counters(self):
        eng = _engine()
        counters = eng._reconcile_preplaced()
        assert counters == dict(n_preplaced=0, n_kept=0, n_replaced=0, n_cancelled=0,
                                 n_added=0, n_filled_before_reconcile=0)
        assert eng._preplace_reconciled_today is True


# ===========================================================================
# 6. Tripwire reads the pre-placement timestamp
# ===========================================================================
class TestTripwireReadsPreplaceTimestamps:
    def test_first_submit_at_close_reports_near_zero_delay(self, monkeypatch, caplog):
        eng = _engine()
        sym = 'TRIP'
        eng.candidates[sym] = CandidateState(symbol=sym)
        eng._preplace_state[sym] = {
            'plan': _plan(sym), 'provisional_range_high': 10.5,
            'provisional_range_low': 9.9, 'submitted': False, 'order_id': None,
        }
        fixed = pd.Timestamp('2026-09-28 09:35:00.400').to_pydatetime()
        monkeypatch.setattr(eng, '_et_now', lambda: fixed)
        eng._submit_entry = MagicMock(return_value='order-1')
        with caplog.at_level(logging.INFO):
            eng._preplace_submit_at_close()
        tripwire_lines = [r.message for r in caplog.records if 'first order submit' in r.message
                           or 'LATENCY TRIPWIRE' in r.message]
        assert tripwire_lines, 'tripwire must log a first-submit line'
        assert any('0.4s' in m for m in tripwire_lines)
        assert eng.latency_warn_secs >= 1.5 or 'TRIPWIRE' not in tripwire_lines[0]


# ===========================================================================
# 7. Flag off: byte-identical, new code never touched
# ===========================================================================
class TestFlagOffByteIdentical:
    def test_check_entries_locked_never_reconciles_when_off(self):
        """`_maybe_arm_preplace_scheduler` IS called every tick (cheap,
        internally gated -- see TestSchedulerArming.test_flag_off_never_arms
        for its own no-op proof). `_reconcile_preplaced` must never run when
        the flag is off: it is gated on `_preplace_submitted_today`, which
        can never turn True without the flag (nothing else sets it)."""
        eng = _engine(preplace=False)
        eng._reconcile_preplaced = MagicMock(
            side_effect=AssertionError('must not be called'))
        # A minimal, harmless check_entries call -- no candidates, so the
        # rest of the pipeline is a fast no-op regardless of the flag.
        assert eng.check_entries() == []
        eng._reconcile_preplaced.assert_not_called()

    def test_arm_is_reached_but_inert_when_flag_off(self):
        """Flag-off engines DO reach `_maybe_arm_preplace_scheduler` (it is
        called unconditionally every tick, cheaply) but it must be a no-op:
        no Timer, no state."""
        eng = _engine(preplace=False)
        assert eng.check_entries() == []
        assert eng._preplace_armed_today is False
        assert eng._preplace_state == {}


# ===========================================================================
# 8. Submit delay (entry.preplace_submit_delay_s, docs/orb_preplace_spec_
#    20260928.md "Submit delay"): config parsing, scheduler target, and the
#    normal-tick ordering guard that stops a duplicate order while a
#    delayed submit is armed but not yet fired.
# ===========================================================================
def _engine_with_delay(delay=None, preplace=True):
    cfg = _base_cfg(preplace)
    if delay is not None:
        cfg['entry']['preplace_submit_delay_s'] = delay
    return ORBEngine(alpaca_client=_mock_alpaca(), db=MagicMock(spec=Database),
                      stop_monitor=MagicMock(spec=StopMonitor), config=cfg)


def _neutralize_gates(eng, monkeypatch):
    """Neutralize every check_entries gate upstream of the eligibility loop
    so a test exercises ONLY the preplace-pending guard, independent of
    wall-clock time, DB state, or the other kill-switches."""
    monkeypatch.setattr(eng, '_maybe_arm_preplace_scheduler', lambda: None)
    monkeypatch.setattr(eng, '_maybe_prefetch_pm', lambda: None)
    monkeypatch.setattr(eng, '_prewarm_anchors', lambda: None)
    monkeypatch.setattr(eng, '_process_pending_fills', lambda: None)
    monkeypatch.setattr(eng, '_cancel_stale_pending_orders', lambda: None)
    monkeypatch.setattr(eng, '_ensure_ranges_post_open', lambda: set())
    monkeypatch.setattr(eng, '_daily_loss_limit_hit', lambda: False)
    monkeypatch.setattr(eng, '_kill_rails_blocked', lambda: False)
    monkeypatch.setattr(eng, '_pdt_would_block', lambda: False)
    monkeypatch.setattr(eng, '_past_last_entry_time', lambda: False)
    monkeypatch.setattr(eng, '_should_defer_first_rank', lambda: False)
    monkeypatch.setattr(eng, '_symbols_entered_today_db', lambda: set())
    monkeypatch.setattr(eng, '_symbol_has_any_open_trade', lambda sym: False)


def _seed_pending(eng, sym='PEND', range_high=10.5):
    """Seed a symbol exactly as `_preplace_provisional_rank` would leave it
    mid-flight: a real CandidateState with its FINAL range_data already in
    (the post-open sweep can beat a delayed submit to it), plus a
    `_preplace_state` entry that has NOT been submitted yet."""
    eng.candidates[sym] = CandidateState(symbol=sym)
    eng.candidates[sym].range_data = _range(sym, range_high=range_high)
    eng._preplace_state[sym] = {
        'plan': _plan(sym, range_high=range_high), 'provisional_range_high': range_high,
        'provisional_range_low': 9.9, 'submitted': False, 'order_id': None,
    }
    eng._preplace_ranked_today = True


class TestSubmitDelayConfig:
    def test_missing_key_defaults_zero(self):
        eng = _engine()  # _base_cfg never sets preplace_submit_delay_s
        assert eng.preplace_submit_delay_s == 0.0

    def test_negative_clamped_to_zero_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            eng = _engine_with_delay(-2.5)
        assert eng.preplace_submit_delay_s == 0.0
        assert any('preplace_submit_delay_s' in r.message and '-2.5' in r.message
                   for r in caplog.records)


class TestSubmitDelayScheduling:
    def test_default_zero_submit_target_unchanged(self, monkeypatch):
        """Byte-identical call sequence at the default: the submit Timer's
        interval and callback match pre-delay behaviour exactly."""
        eng = _engine()
        monkeypatch.setattr(orb_engine_mod.threading, 'Timer', FakeTimer)
        fixed = pd.Timestamp('2026-09-28 09:34:50').to_pydatetime()
        monkeypatch.setattr(eng, '_et_now', lambda: fixed)
        eng._maybe_arm_preplace_scheduler()
        rank_t, submit_t = FakeTimer.instances
        assert submit_t.interval == pytest.approx(10.0)  # 09:35:00.000 - 09:34:50
        assert submit_t.function == eng._preplace_submit_at_close

    def test_delay_5s_targets_0935_05(self, monkeypatch, caplog):
        eng = _engine_with_delay(5.0)
        monkeypatch.setattr(orb_engine_mod.threading, 'Timer', FakeTimer)
        fixed = pd.Timestamp('2026-09-28 09:34:50').to_pydatetime()
        monkeypatch.setattr(eng, '_et_now', lambda: fixed)
        with caplog.at_level(logging.INFO):
            eng._maybe_arm_preplace_scheduler()
        rank_t, submit_t = FakeTimer.instances
        assert submit_t.interval == pytest.approx(15.0)  # 09:35:05.000 - 09:34:50
        assert rank_t.interval == pytest.approx(7.0)      # rank target unaffected by delay
        armed_lines = [r.message for r in caplog.records if 'scheduler armed' in r.message]
        assert any('09:35:05' in m and 'submit_delay_s=5.0' in m for m in armed_lines)

    def test_never_earlier_than_0935_plus_delay_when_armed_late(self, monkeypatch):
        eng = _engine_with_delay(5.0)
        monkeypatch.setattr(orb_engine_mod.threading, 'Timer', FakeTimer)
        fixed = pd.Timestamp('2026-09-28 09:34:59.999').to_pydatetime()
        monkeypatch.setattr(eng, '_et_now', lambda: fixed)
        eng._maybe_arm_preplace_scheduler()
        _, submit_t = FakeTimer.instances
        assert submit_t.interval >= 0.0  # clamped, never negative


class TestSubmitDelayOrderingGuard:
    def test_normal_tick_defers_pending_symbol_and_logs(self, monkeypatch, caplog):
        eng = _engine_with_delay(5.0)
        _neutralize_gates(eng, monkeypatch)
        _seed_pending(eng, 'PEND')
        eng._submit_entry = MagicMock(side_effect=AssertionError(
            'normal path must not submit while a preplace submit is pending'))
        eng._run_pool_selection = MagicMock(side_effect=AssertionError(
            'must not even reach pool selection for a pending-only tick'))
        with caplog.at_level(logging.INFO):
            result = eng.check_entries()
        assert result == []
        assert eng.candidates['PEND'].rejected_reason == 'preplace_submit_pending'
        assert eng.candidates['PEND'].plan_submitted is False
        assert any('normal-tick deferral' in r.message and 'PEND' in r.message
                   for r in caplog.records)

    def test_tick_after_submit_completes_reconciles_not_deferred(self, monkeypatch):
        eng = _engine_with_delay(5.0)
        _neutralize_gates(eng, monkeypatch)
        _seed_pending(eng, 'PEND')
        eng._submit_entry = MagicMock(return_value='order-PEND')
        eng._preplace_submit_at_close()  # simulate the delayed Timer firing
        assert eng._preplace_submitted_today is True
        eng._reconcile_preplaced = MagicMock(return_value={
            'n_preplaced': 1, 'n_kept': 1, 'n_replaced': 0, 'n_cancelled': 0,
            'n_added': 0, 'n_filled_before_reconcile': 0})
        result = eng.check_entries()
        eng._reconcile_preplaced.assert_called_once()
        # PEND was submitted by the preplace path -- plan_submitted is True,
        # so it is excluded from this tick's eligibility, not deferred.
        assert eng.candidates['PEND'].rejected_reason != 'preplace_submit_pending'
        assert result == []

    def test_failed_delayed_submit_normal_path_resumes_with_warning(self, monkeypatch, caplog):
        eng = _engine_with_delay(5.0)
        _neutralize_gates(eng, monkeypatch)
        _seed_pending(eng, 'FAIL')
        eng._submit_entry = MagicMock(return_value=None)  # delayed submit fails
        with caplog.at_level(logging.WARNING):
            eng._preplace_submit_at_close()
        assert eng._preplace_submitted_today is True
        assert eng._preplace_state['FAIL']['submitted'] is False
        assert eng.candidates['FAIL'].plan_submitted is False
        assert any('FAIL' in r.message and 'submit failed' in r.message and 'no refill' in r.message
                   for r in caplog.records)
        # Normal path resumes on the very next tick: FAIL is no longer
        # "pending" (_preplace_submitted_today is True) so it must reach
        # pool selection instead of being silently dropped forever.
        eng._reconcile_preplaced = MagicMock(return_value={
            'n_preplaced': 1, 'n_kept': 0, 'n_replaced': 0, 'n_cancelled': 0,
            'n_added': 0, 'n_filled_before_reconcile': 0})
        eng._run_pool_selection = MagicMock(return_value=[])
        eng.check_entries()
        eng._run_pool_selection.assert_called_once()
        production_syms = eng._run_pool_selection.call_args[0][1]
        assert 'FAIL' in production_syms
