"""ORB reconcile: every provisional order that must go is cancelled in ONE shared pass and awaited
under ONE 15 s budget; the scanner never stacks ORB ticks (docs/orb_reconcile_cancel_batch_20261123.md,
2026-11-23: 5 x 15 s serial waits stalled the 09:35 tick 92 s and paged "Engine tick TIMEOUT")."""
from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock

import pandas as pd
import pytest

from persistence.database import Database
from scanner.realtime_scanner import RealtimeScanner
from trading.orb_engine import CandidateState, OpenPosition
from tests.test_orb_preplace_budget_20261006 import (  # noqa: F401  (fixture re-export)
    eng, _go_final)
from tests.test_orb_preplace_late_fill import _seat_replace_name, SYM  # noqa: F401

OUT = ['BMNU', 'DRAL', 'RGTI', 'AEVA', 'RDW']   # the 11/23 provisional names, none in the final set


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Count (and skip) every sleep the engine would take."""
    sleeps = []
    monkeypatch.setattr('trading.orb_engine.time.sleep', lambda s: sleeps.append(s))
    return sleeps


def _seat_outside(eng, names=OUT, first_trade_id=861):
    """Register resting preplaced orders for names with no final range (outside the final top-N)."""
    for i, sym in enumerate(names):
        eng._preplace_state[sym] = {'provisional_range_high': 10.5, 'submitted': True}
        eng.candidates[sym] = CandidateState(symbol=sym)
        eng.open_positions[sym] = OpenPosition(
            symbol=sym, entry_price=10.5, stop_price=9.9, shares=100, trade_id=first_trade_id + i,
            order_id=f'order-{sym}', entry_time=pd.Timestamp.utcnow().to_pydatetime(),
            range_high=10.5, range_low=9.9, lock_arm_at_r=1.0, lock_stop_r=0.0,
            composite_score=1.0, quintile='Q4')


def _oms_cancels_after(eng, n_polls, filled=None):
    """get_order: `pending_cancel` for the first `n_polls` calls overall (one OMS cancel queue), then
    `canceled`; `filled` maps order id -> filled_qty reported once terminal. Returns the call log."""
    calls = []
    filled = filled or {}

    def _get(oid):
        calls.append(oid)
        if len(calls) <= n_polls:
            return {'status': 'pending_cancel', 'filled_qty': 0}
        return {'status': 'canceled', 'filled_qty': filled.get(oid, 0)}
    eng.alpaca.get_order.side_effect = _get
    return calls


def _marked(eng):
    return {c.args[0]: c.args[1] for c in eng.db.update_trade.call_args_list
            if c.args[1].get('order_status') == 'preplace_canceled'}


def test_1_replay_five_outside_names_one_shared_pass(eng, caplog, _no_sleep):
    _go_final(eng)            # final ranges first: the seated names stay rangeless (outside the final set)
    _seat_outside(eng)
    order = []
    eng._cancel_symbol_open_orders.side_effect = lambda s: order.append(('cancel', s)) or 1
    calls = _oms_cancels_after(eng, 3)
    real_get = eng.alpaca.get_order.side_effect
    eng.alpaca.get_order.side_effect = lambda oid: order.append(('get', oid)) or real_get(oid)
    with caplog.at_level(logging.INFO):
        counters = eng._reconcile_preplaced()
    kinds = [k for k, _ in order]
    assert kinds[:5] == ['cancel'] * 5 and 'get' not in kinds[:5]      # all cancels BEFORE any poll
    assert eng._cancel_symbol_open_orders.call_count == 5
    assert len(_no_sleep) < 5 * 3                                       # one budget, not 5 serial waits
    marked = _marked(eng)
    assert sorted(marked) == [861, 862, 863, 864, 865]
    assert all(v == {'order_status': 'preplace_canceled', 'exit_reason': 'preplace_reconcile'}
               for v in marked.values())
    assert counters['n_cancelled'] == 5
    infos = [r for r in caplog.records if r.levelno == logging.INFO and 'cancel pass: 5 orders' in r.getMessage()]
    assert len(infos) == 1


def test_2_one_fills_during_the_wait(eng):
    _go_final(eng)            # final ranges first: the seated names stay rangeless (outside the final set)
    _seat_outside(eng)
    _oms_cancels_after(eng, 3, filled={'order-RGTI': 100})
    assert eng._cancel_preplaced_order  # method exists
    counters = eng._reconcile_preplaced()
    assert counters['n_cancelled'] == 4
    assert sorted(_marked(eng)) == [861, 862, 864, 865]     # RGTI (863) not marked
    assert 'RGTI' in eng.open_positions                      # the fill path owns it
    assert all(s not in eng.open_positions for s in ('BMNU', 'DRAL', 'AEVA', 'RDW'))


def test_3_never_terminal_still_marks_and_warns(eng, caplog):
    _go_final(eng)            # final ranges first: the seated names stay rangeless (outside the final set)
    _seat_outside(eng)
    eng.alpaca.get_order.side_effect = lambda oid: {'status': 'pending_cancel', 'filled_qty': 0}
    with caplog.at_level(logging.INFO):
        counters = eng._reconcile_preplaced()
    assert counters['n_cancelled'] == 5
    assert sorted(_marked(eng)) == [861, 862, 863, 864, 865]
    warns = [r for r in caplog.records if r.levelno == logging.WARNING
             and 'cancel pass' in r.getMessage() and 'still pending' in r.getMessage()]
    assert len(warns) == 1
    assert not [r for r in caplog.records if r.levelno == logging.INFO and 'cancel pass' in r.getMessage()]


def test_4_replace_and_outside_share_one_budget(eng, monkeypatch):
    _seat_replace_name(eng)   # also sets every final range
    _seat_outside(eng, names=['BMNU'], first_trade_id=900)
    waits = []
    clock = [1000.0]                      # fake monotonic clock: a never-terminal order burns its whole wait
    monkeypatch.setattr('trading.orb_engine.time.monotonic', lambda: clock[0])

    def _spy(oid, max_wait_s=15.0, step_s=0.5):
        waits.append((oid, max_wait_s))
        clock[0] += max_wait_s
        return 'pending_cancel', 0
    eng._await_terminal = _spy
    eng._reconcile_preplaced()
    assert sorted(o for o, _ in waits) == ['9debf1bf', 'order-BMNU']   # once per order, no second wait
    assert sum(w for _, w in waits) <= eng._RECONCILE_WAIT_S + 0.5


def test_5_other_callers_still_cancel_and_await_themselves(eng):
    _seat_outside(eng, names=['BMNU'])
    pos = eng.open_positions['BMNU']
    eng.alpaca.get_order.return_value = {'status': 'canceled', 'filled_qty': 0}
    assert eng._cancel_preplaced_order('BMNU', pos) is True
    eng._cancel_symbol_open_orders.assert_called_once_with('BMNU')
    eng.alpaca.get_order.assert_called()
    # awaited result supplied: no second cancel, no poll
    eng._cancel_symbol_open_orders.reset_mock()
    eng.alpaca.get_order.reset_mock()
    assert eng._cancel_preplaced_order('BMNU', pos, awaited=('canceled', 0)) is True
    eng._cancel_symbol_open_orders.assert_not_called()
    eng.alpaca.get_order.assert_not_called()
    assert eng._cancel_preplaced_order('BMNU', pos, awaited=('canceled', 7)) is False


def _scanner():
    """RealtimeScanner without __init__, engines mocked, only what `_orb_tick` touches."""
    s = RealtimeScanner.__new__(RealtimeScanner)
    s.orb_engine = MagicMock()
    s.orb_engine.enabled = False
    s.orb_engine.universe_build_due.return_value = False
    s.orb_engine.is_force_close_time.return_value = False
    s._orb_tick_lock = threading.Lock()
    return s


def test_6_scanner_never_stacks_orb_ticks(caplog):
    s = _scanner()
    gate, entered = threading.Event(), threading.Event()
    s.orb_engine.check_exits.side_effect = lambda: (entered.set(), gate.wait(10))
    t = threading.Thread(target=s._orb_tick)
    t.start()
    assert entered.wait(5)
    with caplog.at_level(logging.WARNING):
        s._orb_tick()                                   # returns at once
    assert s.orb_engine.check_exits.call_count == 1
    assert any('ORB tick skipped' in r.getMessage() and r.levelno == logging.WARNING
               for r in caplog.records)
    gate.set()
    t.join(5)
    s.orb_engine.check_exits.side_effect = None
    s._orb_tick()                                       # lock released: runs normally
    assert s.orb_engine.check_exits.call_count == 2
    assert s.orb_engine.check_entries.call_count == 2


def test_7_integration_real_database_rows_marked(eng, tmp_path):
    db = Database(cache_path=str(tmp_path / 'c.db'), trades_path=str(tmp_path / 't.db'))
    eng.db = db
    _go_final(eng)
    _seat_outside(eng)
    ids = {}
    for sym in OUT:
        row = dict.fromkeys(['fill_price', 'filled_at', 'exit_price', 'exited_at', 'pnl', 'pnl_pct',
                             'pattern_data'])
        row.update(symbol=sym, trade_date='2026-11-23', strategy='orb', side='buy', shares=100,
                   entry_price=10.5, stop_loss_price=9.9, take_profit_price=11.5, risk_per_share=0.6,
                   total_risk=60.0, risk_reward_ratio=2.0, order_status='pending_new', account='paper',
                   order_id=f'order-{sym}', exit_reason=None)
        ids[sym] = db.save_trade(row)
        eng.open_positions[sym].trade_id = ids[sym]
    _oms_cancels_after(eng, 3)
    counters = eng._reconcile_preplaced()
    assert counters['n_cancelled'] == 5
    for sym in OUT:
        t = db.get_trade_by_order_id(f'order-{sym}')
        assert t['order_status'] == 'preplace_canceled' and t['exit_reason'] == 'preplace_reconcile'
