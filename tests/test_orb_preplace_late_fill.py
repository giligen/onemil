"""ORB preplace: a cancelled provisional order that fills late must be KEPT and ledgered
(docs/orb_preplace_late_fill_spec_20261106.md; 2026-10-27 CRDU: 861 sh filled after the cancel
request, replacement bought 599 more, 861 sh naked)."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest

from persistence.database import Database
from trading import exit_coverage
from trading.orb_engine import OpenPosition
from tests.test_orb_preplace_budget_20261006 import (  # noqa: F401  (fixture re-export)
    eng, _go_final, _range, _plan, _fake_submit, FINAL_RANGE_OPEN)

SYM = 'DFDV'


def _seat_replace_name(eng, trade_id=618):
    """DFDV rests at provisional 10.50; the final range_high will be 10.54 -> REPLACED path."""
    eng._preplace_state[SYM] = {'provisional_range_high': 10.5, 'submitted': True}
    pos = OpenPosition(
        symbol=SYM, entry_price=10.53, stop_price=9.9, shares=906, trade_id=trade_id,
        order_id='9debf1bf', entry_time=pd.Timestamp.utcnow().to_pydatetime(), range_high=10.5,
        range_low=9.9, lock_arm_at_r=1.0, lock_stop_r=0.0, composite_score=1.0, quintile='Q4')
    eng.open_positions[SYM] = pos
    _go_final(eng)
    eng.candidates[SYM].range_data = _range(SYM, FINAL_RANGE_OPEN, range_high=10.54)
    _fake_submit(eng, ok=(SYM,))
    return pos


def _poll_sequence(eng, *orders):
    """get_order returns `orders` in turn, then the last one forever; returns the call list."""
    calls = []

    def _get(oid):
        calls.append(oid)
        return orders[min(len(calls), len(orders)) - 1]
    eng.alpaca.get_order.side_effect = _get
    return calls


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr('trading.orb_engine.time.sleep', lambda s: sleeps.append(s))
    return sleeps


def test_a_late_fill_is_kept_and_no_replacement(eng, caplog):
    pos = _seat_replace_name(eng)
    pending = {'status': 'pending_cancel', 'filled_qty': 0}
    done = {'status': 'canceled', 'filled_qty': 861, 'filled_avg_price': 5.52}
    calls = _poll_sequence(eng, pending, pending, done)
    eng._confirm_fill = MagicMock()
    with caplog.at_level(logging.WARNING):
        counters = eng._reconcile_preplaced()
    assert calls[:3] == ['9debf1bf'] * 3
    eng._submit_entry.assert_not_called()
    eng._confirm_fill.assert_called_once()
    assert eng._confirm_fill.call_args.args[0] is pos
    assert eng._confirm_fill.call_args.args[1]['filled_qty'] == 861
    assert eng.open_positions[SYM] is pos
    assert counters['n_replaced'] == 0
    assert any(r.levelno == logging.WARNING and
               f'[ORB PREPLACE] {SYM}: provisional order filled 861 sh before the cancel landed '
               f'— kept as the entry, no replacement' in r.getMessage() for r in caplog.records)


def test_b_canceled_with_nothing_filled_submits_replacement(eng):
    _seat_replace_name(eng)
    _poll_sequence(eng, {'status': 'pending_cancel', 'filled_qty': 0},
                   {'status': 'canceled', 'filled_qty': 0})
    counters = eng._reconcile_preplaced()
    eng._submit_entry.assert_called_once()
    assert counters['n_replaced'] == 1


def test_c_never_terminal_logs_error_and_no_replacement(eng, caplog, _no_sleep):
    pos = _seat_replace_name(eng)
    _poll_sequence(eng, {'status': 'pending_cancel', 'filled_qty': 0})
    with caplog.at_level(logging.WARNING):
        eng._reconcile_preplaced()
    eng._submit_entry.assert_not_called()
    assert eng.open_positions[SYM] is pos
    errs = [r for r in caplog.records if r.levelno == logging.ERROR and SYM in r.getMessage()]
    assert len(errs) == 1 and 'NO replacement' in errs[0].getMessage()


def test_d_cancel_path_late_partial_returns_false_row_unmarked(eng):
    pos = _seat_replace_name(eng)
    _poll_sequence(eng, {'status': 'pending_cancel', 'filled_qty': 0},
                   {'status': 'canceled', 'filled_qty': '37'})
    assert eng._cancel_preplaced_order(SYM, pos) is False
    eng.db.update_trade.assert_not_called()


def test_e_polls_are_bounded(eng, _no_sleep):
    calls = _poll_sequence(eng, {'status': 'pending_cancel', 'filled_qty': 0})
    status, qty = eng._await_terminal('oid', max_wait_s=15.0, step_s=0.5)
    assert (status, qty) == ('pending_cancel', 0)
    assert len(calls) == 31 and len(_no_sleep) == 30
    calls.clear()
    _no_sleep.clear()
    eng.alpaca.get_order.side_effect = None
    eng.alpaca.get_order.return_value = MagicMock(status='canceled', filled_qty=5)  # object, not dict
    assert eng._await_terminal('oid') == ('canceled', 5) and _no_sleep == []


def test_f_unledgered_excess_on_orb_symbol_errors_once_foreign_silent(caplog):
    res = exit_coverage.CoverageResult(position_qty=1460, covered_qty=599, excess_qty=861,
                                       qty_expected=599)
    warned: set = set()
    with caplog.at_level(logging.WARNING):
        assert exit_coverage.report_unledgered_excess('CRDU', res, 'orb', warned, '2026-10-27')
        assert not exit_coverage.report_unledgered_excess('CRDU', res, 'orb', warned, '2026-10-27')
        assert not exit_coverage.report_unledgered_excess('AAPL', res, 'bull_flag', warned, '2026-10-27')
    errs = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errs) == 1 and errs[0].getMessage() == (
        '[ORB] CRDU: broker holds 861 sh above the ledger on a symbol ORB traded today — unledgered fill')


def test_integration_late_fill_row_carries_filled_qty(eng, tmp_path):
    """Real Database: after the late fill the row has shares == filled_qty == 861, one live row."""
    db = Database(cache_path=str(tmp_path / 'c.db'), trades_path=str(tmp_path / 't.db'))
    row = dict.fromkeys(['order_id', 'fill_price', 'filled_at', 'exit_price', 'exit_reason',
                         'exited_at', 'pnl', 'pnl_pct', 'pattern_data'])
    row.update(symbol=SYM, trade_date='2026-10-27', strategy='orb', side='buy', shares=906,
               entry_price=5.51, stop_loss_price=5.0, take_profit_price=6.0, risk_per_share=0.5,
               total_risk=450.0, risk_reward_ratio=2.0, order_status='pending_new', account='paper')
    row['order_id'] = '9debf1bf'
    trade_id = db.save_trade(row)
    eng.db = db
    eng.stop_monitor = MagicMock()
    pos = _seat_replace_name(eng, trade_id=trade_id)
    _poll_sequence(eng, {'status': 'pending_cancel', 'filled_qty': 0},
                   {'status': 'canceled', 'filled_qty': 861, 'filled_avg_price': 5.52})
    eng._reconcile_preplaced()
    eng._submit_entry.assert_not_called()
    stored = db.get_trade_by_id(trade_id)
    assert stored['shares'] == 861 and stored['filled_qty'] == 861
    assert stored['order_status'] == 'filled' and stored['fill_price'] == pytest.approx(5.52)
    assert pos.shares == 861 and pos.order_id == ''
    rows = [t for t in db.get_trades_by_date('2026-10-27') if t['symbol'] == SYM]
    assert len(rows) == 1
