"""ORB: a preplaced order cancelled by the 09:35 reconcile must leave its trades row
`preplace_canceled` / `preplace_reconcile`, not `pending_new` (2026-10-19 IREX)."""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from persistence.database import Database
from trading.orb_engine import OpenPosition
from trading.trading_engine import TERMINAL_PENDING_STATUSES
from tests.test_orb_preplace_budget_20261006 import (  # noqa: F401  (fixture re-export)
    eng, _preplace, _go_final)


def _seat_inside_name(eng, sym='EWZS'):
    """Register a resting preplaced order for a name that IS in the final top-8."""
    eng._preplace_state[sym] = {'provisional_range_high': 10.5, 'submitted': True}
    eng.open_positions[sym] = OpenPosition(
        symbol=sym, entry_price=10.5, stop_price=9.9, shares=100, trade_id=77,
        order_id=f'order-{sym}', entry_time=pd.Timestamp.utcnow().to_pydatetime(),
        range_high=10.5, range_low=9.9, lock_arm_at_r=1.0, lock_stop_r=0.0,
        composite_score=1.0, quintile='Q4')


def _status_calls(eng):
    return [c for c in eng.db.update_trade.call_args_list
            if c.args[1].get('order_status') == 'preplace_canceled']


def test_outside_top_n_cancels_and_marks_row(eng):
    eng.alpaca.get_order.return_value = {'filled_qty': 0}
    _preplace(eng, ok=('JAGX', 'CRCG'))
    _go_final(eng)
    counters = eng._reconcile_preplaced()
    assert counters['n_cancelled'] == 2
    assert eng._cancel_symbol_open_orders.call_count == 2
    calls = _status_calls(eng)
    assert len(calls) == 2
    assert all(c.args[1] == {'order_status': 'preplace_canceled',
                             'exit_reason': 'preplace_reconcile'} for c in calls)
    assert not eng.open_positions


def test_inside_top_n_row_untouched(eng):
    eng.alpaca.get_order.return_value = {'filled_qty': 0}
    _preplace(eng, ok=())
    _go_final(eng)
    _seat_inside_name(eng)
    counters = eng._reconcile_preplaced()
    assert counters['n_kept'] == 1
    eng.db.update_trade.assert_not_called()
    assert 'EWZS' in eng.open_positions


def test_cancel_with_partial_fill_not_marked_and_warns(eng, caplog):
    eng.alpaca.get_order.return_value = {'filled_qty': '37'}
    _preplace(eng, ok=('JAGX',))
    _go_final(eng)
    with caplog.at_level(logging.WARNING):
        counters = eng._reconcile_preplaced()
    assert counters['n_cancelled'] == 0
    assert _status_calls(eng) == []
    assert 'JAGX' in eng.open_positions            # the fill path still owns it
    assert any('37 sh' in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_reader_sites_treat_preplace_canceled_like_time_stop_canceled(tmp_path):
    # restart rebuild of _pending_orders (trading_engine terminal set)
    assert {'time_stop_canceled', 'preplace_canceled'} <= TERMINAL_PENDING_STATUSES
    # Database.get_open_trades allowlist: both are terminal, neither is "open"
    db = Database(cache_path=str(tmp_path / 'c.db'), trades_path=str(tmp_path / 't.db'))
    ids = {}
    for status in ('time_stop_canceled', 'preplace_canceled', 'pending_new'):
        row = dict.fromkeys(['order_id', 'fill_price', 'filled_at', 'exit_price',
                             'exit_reason', 'exited_at', 'pnl', 'pnl_pct', 'pattern_data'])
        row.update(symbol=status[:3].upper(), trade_date='2026-10-19', strategy='orb', side='buy',
                   shares=10, entry_price=5.0, stop_loss_price=4.9, take_profit_price=5.3,
                   risk_per_share=0.1, total_risk=1.0, risk_reward_ratio=3.0, order_status=status, account='paper')
        ids[status] = db.save_trade(row)
    open_syms = {t['order_status'] for t in db.get_open_trades('2026-10-19', 'orb')}
    assert open_syms == {'pending_new'}
