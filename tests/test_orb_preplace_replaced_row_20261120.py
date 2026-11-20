"""ORB preplace: the reconcile REPLACED branch must leave the provisional trades row
`preplace_canceled` / `preplace_replaced`, not `pending_new` (2026-11-20 SNK/SSPC/SPCQ)."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest

from persistence.database import Database
from trading.orb_engine import OpenPosition, PlannerReject
from tests.test_orb_preplace_budget_20261006 import (  # noqa: F401  (fixture re-export)
    eng, _go_final, _range, _fake_submit, _preplace, FINAL_RANGE_OPEN)
from tests.test_orb_preplace_late_fill import _poll_sequence, _no_sleep  # noqa: F401

SYM = 'DFDV'   # stands in for the 11/20 SNK replay (a final top-8 name)
OLD_ID = 843
REPLACED = {'order_status': 'preplace_canceled', 'exit_reason': 'preplace_replaced'}
CANCELED = {'status': 'canceled', 'filled_qty': 0}


def _seat(eng, ok=(SYM,), trade_id=OLD_ID, order_id='prov-snk'):
    """SNK rests at provisional 15.87; the final range_high is 16.03 -> REPLACED branch."""
    eng._preplace_state[SYM] = {'provisional_range_high': 15.87, 'submitted': True}
    eng.open_positions[SYM] = OpenPosition(
        symbol=SYM, entry_price=15.9, stop_price=15.0, shares=100, trade_id=trade_id,
        order_id=order_id, entry_time=pd.Timestamp.utcnow().to_pydatetime(), range_high=15.87,
        range_low=15.0, lock_arm_at_r=1.0, lock_stop_r=0.0, composite_score=1.0, quintile='Q4')
    _go_final(eng)
    eng.candidates[SYM].range_data = _range(SYM, FINAL_RANGE_OPEN, range_high=16.03)
    _fake_submit(eng, ok=ok)


def _marks(eng, reason='preplace_replaced'):
    return [c for c in eng.db.update_trade.call_args_list if c.args[1].get('exit_reason') == reason]


def test_1_replacement_submitted_marks_old_row(eng):
    _seat(eng)
    _poll_sequence(eng, CANCELED)
    counters = eng._reconcile_preplaced()
    assert counters['n_replaced'] == 1
    eng.db.update_trade.assert_called_once_with(OLD_ID, REPLACED)


def test_2_submit_none_marks_old_row(eng):
    _seat(eng, ok=())
    _poll_sequence(eng, CANCELED)
    counters = eng._reconcile_preplaced()
    assert counters['n_cancelled'] == 1 and counters['n_replaced'] == 0
    eng.db.update_trade.assert_called_once_with(OLD_ID, REPLACED)


def test_3_planner_reject_marks_old_row(eng):
    _seat(eng)
    eng.planner.build.side_effect = lambda **kw: PlannerReject(symbol=kw['symbol'], reason='test', details={})
    _poll_sequence(eng, CANCELED)
    counters = eng._reconcile_preplaced()
    assert counters['n_cancelled'] == 1
    eng._submit_entry.assert_not_called()
    eng.db.update_trade.assert_called_once_with(OLD_ID, REPLACED)


def test_4_late_fill_not_marked(eng):
    _seat(eng)
    eng._confirm_fill = MagicMock()
    _poll_sequence(eng, {'status': 'canceled', 'filled_qty': 861, 'filled_avg_price': 15.9})
    eng._reconcile_preplaced()
    assert _marks(eng) == []
    eng._submit_entry.assert_not_called()


def test_5_not_terminal_not_marked_error_no_replacement(eng, caplog):
    _seat(eng)
    _poll_sequence(eng, {'status': 'pending_cancel', 'filled_qty': 0})
    with caplog.at_level(logging.ERROR):
        eng._reconcile_preplaced()
    assert _marks(eng) == []
    eng._submit_entry.assert_not_called()
    assert any(r.levelno == logging.ERROR and 'NO replacement' in r.getMessage() for r in caplog.records)


def test_6_not_in_final_still_marks_preplace_reconcile(eng):
    eng.alpaca.get_order.return_value = CANCELED
    _preplace(eng, ok=('JAGX',))
    _go_final(eng)
    eng._reconcile_preplaced()
    calls = _marks(eng, 'preplace_reconcile')
    assert len(calls) == 1
    assert calls[0].args[1] == {'order_status': 'preplace_canceled', 'exit_reason': 'preplace_reconcile'}


def test_7_update_trade_raising_logs_one_error_and_still_replaces(eng, caplog):
    _seat(eng)
    eng.db.update_trade.side_effect = RuntimeError('db locked')
    _poll_sequence(eng, CANCELED)
    with caplog.at_level(logging.ERROR):
        counters = eng._reconcile_preplaced()
    assert counters['n_replaced'] == 1
    eng._submit_entry.assert_called_once()
    errs = [r for r in caplog.records if r.levelno == logging.ERROR and SYM in r.getMessage()]
    assert len(errs) == 1


def test_8_integration_real_database(eng, tmp_path):
    db = Database(cache_path=str(tmp_path / 'c.db'), trades_path=str(tmp_path / 't.db'))
    row = dict.fromkeys(['order_id', 'fill_price', 'filled_at', 'exit_price', 'exit_reason',
                         'exited_at', 'pnl', 'pnl_pct', 'pattern_data'])
    row.update(symbol=SYM, trade_date='2026-11-20', strategy='orb', side='buy', shares=100,
               entry_price=15.9, stop_loss_price=15.0, take_profit_price=17.0, risk_per_share=0.9,
               total_risk=90.0, risk_reward_ratio=2.0, order_status='pending_new', account='paper')
    row['order_id'] = 'prov-snk'
    old_id = db.save_trade(row)
    eng.db = db
    _seat(eng, trade_id=old_id)
    new_row = dict(row, order_id=f'order-{SYM}')
    eng._submit_entry.side_effect = None

    def _submit(plan):
        eng.open_positions[plan.symbol] = OpenPosition(
            symbol=plan.symbol, entry_price=plan.entry_price, stop_price=15.0, shares=100,
            trade_id=db.save_trade(new_row), order_id=f'order-{plan.symbol}',
            entry_time=pd.Timestamp.utcnow().to_pydatetime(), range_high=plan.range_high,
            range_low=15.0, lock_arm_at_r=1.0, lock_stop_r=0.0, composite_score=1.0, quintile='Q4')
        return f'order-{plan.symbol}'
    eng._submit_entry.side_effect = _submit
    _poll_sequence(eng, CANCELED)
    eng._reconcile_preplaced()
    old = db.get_trade_by_order_id('prov-snk')
    assert old['order_status'] == 'preplace_canceled' and old['exit_reason'] == 'preplace_replaced'
    new = db.get_trade_by_order_id(f'order-{SYM}')
    assert new is not None and new['id'] != old['id']
