"""The 09:35:05 preplace submit must not wait on the post-open range sweep
(docs/orb_preplace_lock_wait_fix_20261017.md).

`check_entries` used to hold `ORBEngine._lock` across the sweep's REST bars
fetch (7.3-7.8 s measured), so the preplace timer's `with self._lock` queued
behind it. The fetch (and the straggler-retry sleep + fetch) now run OUTSIDE
the lock; ingest is re-validated under the lock.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

import trading.orb_engine as orb_engine_mod
from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import CandidateState, OpenPosition, ORBEngine, RangeData
from trading.orb_planner import OrbTradePlan
from trading.stop_monitor import StopMonitor

FULL_FEATS = {
    'gap_pct': 12.0, 'range_total_volume': 500_000.0, 'range_avg_bar_range_pct': 1.2,
    'range_size_pct': 5.0, 'price_vs_20d_high_pct': -4.0, 'prev_day_close_position': 0.5,
    'range_close_position': 0.6, 'prev_day_range_pct': 20.0, 'return_volatility_20d': 20.0,
}
_NOW = datetime(2026, 4, 21, 13, 35, 5, tzinfo=timezone.utc)  # 09:35:05 ET (EDT)


class _FakeDT(datetime):
    """datetime whose now() is pinned to 09:35:05 ET (real subclass: safe in threads)."""

    @classmethod
    def now(cls, tz=None):
        return _NOW.astimezone(tz) if tz else _NOW.replace(tzinfo=None)


@pytest.fixture(autouse=True)
def _pin_clock():
    with patch.object(orb_engine_mod, 'datetime', _FakeDT):
        yield


def _engine(retry_delay=0.0):
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    cfg['entry'] = dict(cfg.get('entry') or {})
    cfg['entry']['preplace_at_close'] = True
    cfg['entry']['preplace_submit_delay_s'] = 0.0
    cfg['entry']['sweep_retry_delay_s'] = retry_delay
    alpaca = MagicMock(spec=AlpacaClient)
    alpaca.get_account_info.return_value = {'buying_power': 500_000.0}
    eng = ORBEngine(alpaca_client=alpaca, db=MagicMock(spec=Database),
                    stop_monitor=MagicMock(spec=StopMonitor), config=cfg)
    eng.fast_submit_enabled = False  # REST sweep path under test
    eng.sweep_retry_delay_s = retry_delay
    for name in ('_maybe_arm_preplace_scheduler', '_maybe_prefetch_pm', '_prewarm_anchors',
                 '_process_pending_fills', '_cancel_stale_pending_orders'):
        setattr(eng, name, MagicMock())
    eng._daily_loss_limit_hit = MagicMock(return_value=True)  # stop right after the sweep
    return eng


def _bars(highs, open_=10.0):
    """5 one-minute bars 09:30-09:34 ET on 2026-04-21."""
    rows = []
    for i, h in enumerate(highs):
        rows.append({'timestamp': datetime(2026, 4, 21, 13, 30 + i, tzinfo=timezone.utc),
                     'open': open_, 'high': h, 'low': open_, 'close': h, 'volume': 10000})
    return pd.DataFrame(rows)


def _range(sym, high):
    return RangeData(symbol=sym, range_high=high, range_low=9.9, range_volume=500_000,
                     range_avg_bar_range_pct=1.0, range_close=high - 0.02,
                     range_start_ts=pd.Timestamp.utcnow(), range_open=10.0)


def _plan(sym, rh=10.5):
    return OrbTradePlan(
        symbol=sym, range_high=rh, range_low=9.9, range_size=rh - 9.9, entry_price=rh + 0.03,
        stop_price=9.9, shares=100, position_dollars=(rh + 0.03) * 100, lock_arm_at_r=1.0,
        lock_stop_r=0.0, risk_per_share=rh - 9.9 + 0.03, total_risk=(rh - 9.9 + 0.03) * 100,
        composite_score=1.0, quintile='Q4', adaptive_mult=1.0)


def _slow_fetch(eng, started, result, hook=None, delay=2.0):
    """Install a get_1min_bars_multi that signals `started`, runs `hook`, sleeps."""
    def _fn(symbols, lookback_minutes=60):
        started.set()
        if hook:
            hook()
        time.sleep(delay)
        return result
    eng.alpaca.get_1min_bars_multi = MagicMock(side_effect=_fn)


def _run_check_entries(eng):
    t = threading.Thread(target=eng.check_entries, daemon=True)
    t.start()
    return t


# (a) ---------------------------------------------------------------------------
def test_preplace_submit_not_blocked_by_sweep_fetch():
    """Fetch sleeps 2 s inside the sweep; the 09:35:05 submit completes < 0.5 s."""
    eng = _engine()
    eng.candidates['AAA'] = CandidateState(symbol='AAA')  # rangeless -> sweep needed
    eng.candidates['PRE'] = CandidateState(symbol='PRE')
    eng._preplace_state['PRE'] = {'plan': _plan('PRE'), 'provisional_range_high': 10.5,
                                  'provisional_range_low': 9.9, 'submitted': False,
                                  'order_id': None}
    eng._submit_entry = MagicMock(return_value='order-PRE')
    started = threading.Event()
    _slow_fetch(eng, started, {'AAA': _bars([10.1, 10.2, 10.3, 10.2, 10.4])})
    t = _run_check_entries(eng)
    assert started.wait(2.0), 'sweep fetch never started'
    t0 = time.time()
    eng._preplace_submit_at_close()
    elapsed = time.time() - t0
    t.join(10)
    assert elapsed < 0.5, f'preplace submit waited {elapsed:.2f}s behind the sweep'
    assert eng._preplace_state['PRE']['submitted'] is True
    assert eng.candidates['AAA'].range_data is not None  # sweep still applied its result
    assert eng._post_open_range_sweep_done is True


# (b) ---------------------------------------------------------------------------
def test_stream_range_during_fetch_not_overwritten_and_still_ranked(caplog):
    eng = _engine()
    eng.candidates['AAA'] = CandidateState(symbol='AAA')
    eng.candidates['BBB'] = CandidateState(symbol='BBB')
    started = threading.Event()

    def _stream_delivers():  # the WS drain finishes AAA while the REST fetch is out
        with eng._lock:
            eng.candidates['AAA'].range_data = _range('AAA', 99.0)

    _slow_fetch(eng, started, {'AAA': _bars([10.1, 10.2, 10.3, 10.2, 10.4]),
                               'BBB': _bars([20.1, 20.2, 20.3, 20.2, 20.4], open_=20.0)},
                hook=_stream_delivers, delay=0.2)
    with caplog.at_level(logging.INFO):
        filled = eng._post_open_sweep(release_lock=True)
    assert eng.candidates['AAA'].range_data.range_high == 99.0  # stream value kept
    assert eng.candidates['BBB'].range_data is not None
    assert {'AAA', 'BBB'} <= filled  # stream-filled name stays in the ranked set
    line = [r.message for r in caplog.records if 'post-open sweep fetched' in r.message]
    assert len(line) == 1
    assert 'fetched 2 ranges' in line[0] and 'outside the lock' in line[0]
    assert '(1 applied, 1 superseded by the stream)' in line[0]


# (c) ---------------------------------------------------------------------------
def test_preplaced_during_fetch_reconciled_as_today(monkeypatch):
    eng = _engine()
    eng._daily_loss_limit_hit = MagicMock(return_value=True)
    eng.candidates['AAA'] = CandidateState(symbol='AAA')
    started = threading.Event()

    def _preplace_lands():  # timer thread submitted KEEP and DROP while the fetch is out
        with eng._lock:
            for sym in ('KEEP', 'DROP'):
                eng.candidates[sym] = CandidateState(symbol=sym)
                eng.candidates[sym].range_data = _range(sym, 10.5)
                eng._preplace_state[sym] = {
                    'plan': _plan(sym), 'provisional_range_high': 10.5,
                    'provisional_range_low': 9.9, 'submitted': True, 'order_id': f'o-{sym}'}
                eng.open_positions[sym] = OpenPosition(
                    symbol=sym, entry_price=10.53, stop_price=9.9, shares=100, trade_id=1,
                    order_id=f'o-{sym}', entry_time=pd.Timestamp.utcnow().to_pydatetime(),
                    range_high=10.5, range_low=9.9, lock_arm_at_r=1.0, lock_stop_r=0.0,
                    composite_score=1.0, quintile='Q4')
            eng._preplace_submitted_today = True

    _slow_fetch(eng, started, {'AAA': _bars([10.1, 10.2, 10.3, 10.2, 10.4])},
                hook=_preplace_lands, delay=0.1)
    monkeypatch.setattr(eng, '_compute_features',
                        lambda cand, prev_day_bar=None, daily_stats_20d=None:
                        {} if cand.symbol == 'DROP' else dict(FULL_FEATS))
    eng._cancel_symbol_open_orders = MagicMock(return_value=1)
    eng._submit_entry = MagicMock(side_effect=AssertionError('reconcile must not resubmit'))
    with patch.multiple(eng, _pdr_veto_reject=MagicMock(return_value=False),
                        _g1_veto_reject=MagicMock(return_value=False),
                        _range_size_veto_reject=MagicMock(return_value=False),
                        _catalyst_veto_reject=MagicMock(return_value=False)):
        eng.check_entries()
    assert eng._preplace_reconciled_today is True
    cancelled = [c.args[0] for c in eng._cancel_symbol_open_orders.call_args_list]
    assert cancelled == ['DROP']  # DROP cancelled, KEEP kept


# (d) ---------------------------------------------------------------------------
def test_daily_reset_during_fetch_discards_result(caplog):
    eng = _engine()
    eng.candidates['AAA'] = CandidateState(symbol='AAA')
    started = threading.Event()

    def _day_rolls():
        eng.reset_daily()
        eng.candidates['AAA'] = CandidateState(symbol='AAA')  # next day's AAA, rangeless

    _slow_fetch(eng, started, {'AAA': _bars([10.1, 10.2, 10.3, 10.2, 10.4])},
                hook=_day_rolls, delay=0.1)
    with caplog.at_level(logging.WARNING):
        filled = eng._post_open_sweep(release_lock=True)
    assert filled == set()
    assert eng.candidates['AAA'].range_data is None  # no cross-day contamination
    assert eng._post_open_range_sweep_done is False
    assert any('daily reset' in r.message for r in caplog.records)


# (f) retry path ------------------------------------------------------------------
def test_straggler_retry_sleep_and_fetch_also_outside_lock():
    eng = _engine(retry_delay=1.0)
    eng.candidates['AAA'] = CandidateState(symbol='AAA')
    eng.candidates['LATE'] = CandidateState(symbol='LATE')
    eng.candidates['PRE'] = CandidateState(symbol='PRE')
    eng._preplace_state['PRE'] = {'plan': _plan('PRE'), 'provisional_range_high': 10.5,
                                  'provisional_range_low': 9.9, 'submitted': False,
                                  'order_id': None}
    eng._submit_entry = MagicMock(return_value='order-PRE')
    calls = {'n': 0}
    started = threading.Event()

    def _fn(symbols, lookback_minutes=60):
        calls['n'] += 1
        if calls['n'] == 1:
            return {'AAA': _bars([10.1, 10.2, 10.3, 10.2, 10.4])}  # LATE not yet at vendor
        started.set()
        time.sleep(0.5)
        return {'LATE': _bars([11.1, 11.2, 11.3, 11.2, 11.4], open_=11.0)}

    eng.alpaca.get_1min_bars_multi = MagicMock(side_effect=_fn)
    t = _run_check_entries(eng)
    assert started.wait(5.0), 'retry fetch never started'
    t0 = time.time()
    eng._preplace_submit_at_close()
    assert time.time() - t0 < 0.5
    t.join(10)
    assert eng.candidates['LATE'].range_data is not None
    assert calls['n'] == 2


# legacy in-lock path keeps working for direct callers ------------------------------
def test_default_call_still_sweeps_under_caller_lock():
    eng = _engine()
    eng.candidates['AAA'] = CandidateState(symbol='AAA')
    eng.alpaca.get_1min_bars_multi = MagicMock(
        return_value={'AAA': _bars([10.1, 10.2, 10.3, 10.2, 10.4])})
    assert eng._ensure_ranges_post_open() == {'AAA'}
    assert eng._post_open_range_sweep_done is True
