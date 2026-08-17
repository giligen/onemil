"""Unit tests for trading.orphan_reconciler.

Three layers:

1. is_owned_orphan predicate: every branch tested, with the unknown-
   strategy + manual-entry cases pinned (these must always classify
   as FOREIGN regardless of how the matching looks).

2. reconcile_strategy_orphans flow: mocked Alpaca + DB + notifier.
   Tests detection, classification, action, rate-limit, kill switch,
   and that foreign positions NEVER trigger close.

3. Rate-limit + cooldown behavior — survives across multiple calls,
   resets at process boundary.
"""
from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

from trading.orphan_reconciler import (
    ORPHAN_RECOVERED_EXIT_REASON,
    OrphanAction,
    ReconcilerConfig,
    is_owned_orphan,
    reconcile_strategy_orphans,
    reset_state_for_tests,
)


@pytest.fixture(autouse=True)
def _clear_state():
    reset_state_for_tests()
    yield
    reset_state_for_tests()


# =========================================================================
# is_owned_orphan — the predicate that decides "ours" vs "foreign"
# =========================================================================

def _broker(**kw):
    base = {'symbol': 'TST', 'qty': 100, 'avg_entry_price': 10.00,
             'unrealized_pl': 0.0, 'side': 'long'}
    base.update(kw)
    return base


def _db(**kw):
    base = {
        'id': 1, 'trade_date': '2026-06-01', 'symbol': 'TST',
        'strategy': 'orb',
        'fill_price': 10.00, 'filled_qty': 100,
        'exit_price': None, 'exit_reason': 'stop_loss_unconfirmed',
        'order_status': 'closed',
    }
    base.update(kw)
    return base


TODAY = date(2026, 6, 5)
CFG = ReconcilerConfig()


class TestIsOwnedOrphanStaleSignal:

    def test_owned_when_stop_loss_unconfirmed(self):
        assert is_owned_orphan(_broker(), _db(), TODAY, CFG)

    def test_owned_when_exit_pending_verification_status(self):
        assert is_owned_orphan(
            _broker(),
            _db(exit_reason=None,
                order_status='exit_pending_verification'),
            TODAY, CFG)

    def test_owned_when_cross_day_filled_no_exit(self):
        # Yesterday's filled row with no exit_price → cross-day orphan
        assert is_owned_orphan(
            _broker(),
            _db(trade_date='2026-06-04', order_status='filled',
                exit_reason=None, exit_price=None),
            TODAY, CFG)

    def test_foreign_when_clean_trail_stop(self):
        # Clean exit row should NEVER make us re-own the broker position
        assert not is_owned_orphan(
            _broker(),
            _db(exit_reason='trail_stop', exit_price=10.5,
                order_status='closed'),
            TODAY, CFG)

    def test_foreign_when_clean_stop_loss(self):
        assert not is_owned_orphan(
            _broker(),
            _db(exit_reason='stop_loss', exit_price=9.5,
                order_status='closed'),
            TODAY, CFG)

    def test_foreign_when_orphan_recovered(self):
        # Once we've recovered an orphan, it must not re-trigger ownership
        assert not is_owned_orphan(
            _broker(),
            _db(exit_reason=ORPHAN_RECOVERED_EXIT_REASON, exit_price=9.5,
                order_status='closed'),
            TODAY, CFG)

    def test_foreign_when_force_close(self):
        assert not is_owned_orphan(
            _broker(),
            _db(exit_reason='force_close', exit_price=10.1,
                order_status='closed'),
            TODAY, CFG)

    def test_foreign_when_filled_no_exit_same_day(self):
        # Today's filled entry not yet exited — NOT an orphan, must not be
        # closed by reconciler. Engine's normal exit flow handles it.
        assert not is_owned_orphan(
            _broker(),
            _db(trade_date=TODAY.isoformat(), order_status='filled',
                exit_reason=None, exit_price=None),
            TODAY, CFG)


class TestIsOwnedOrphanAvgEntryMatch:

    def test_owned_within_tolerance(self):
        # 5 bps on $10 = $0.005, floored at $0.005 → tolerance $0.005
        # Use 10.004 to stay safely inside even with FP imprecision.
        assert is_owned_orphan(
            _broker(avg_entry_price=10.004), _db(fill_price=10.00),
            TODAY, CFG)

    def test_foreign_above_tolerance(self):
        # $10.05 vs $10.00 = 50 bps, well above 5 bps tolerance
        assert not is_owned_orphan(
            _broker(avg_entry_price=10.05), _db(fill_price=10.00),
            TODAY, CFG)

    def test_foreign_on_zero_fill_price(self):
        assert not is_owned_orphan(
            _broker(), _db(fill_price=0.0), TODAY, CFG)

    def test_owned_on_cheap_stock_floor_dominates(self):
        # $2 stock, 5 bps = $0.001 < absolute floor $0.005 → use $0.005
        # Broker @ $2.003 ≤ $2.005, should match
        assert is_owned_orphan(
            _broker(avg_entry_price=2.003), _db(fill_price=2.00),
            TODAY, CFG)


class TestIsOwnedOrphanQtySanity:

    def test_owned_exact_qty(self):
        assert is_owned_orphan(
            _broker(qty=100), _db(filled_qty=100), TODAY, CFG)

    def test_owned_partial_fill(self):
        # Broker holds 50 (after some sold), DB recorded 100 — fine
        assert is_owned_orphan(
            _broker(qty=50), _db(filled_qty=100), TODAY, CFG)

    def test_foreign_when_broker_qty_exceeds_db(self):
        # Someone else added to this — not ours to flatten
        assert not is_owned_orphan(
            _broker(qty=200), _db(filled_qty=100), TODAY, CFG)

    def test_foreign_when_db_filled_qty_zero(self):
        # No actual fill recorded → never claim ownership
        assert not is_owned_orphan(
            _broker(), _db(filled_qty=0), TODAY, CFG)


class TestUnknownStrategySafety:
    """Pin the scenarios the user explicitly asked about: another strategy
    on the same account must NEVER be auto-closed."""

    def test_unknown_strategy_symbol_we_never_traded(self):
        # Ownership is enforced upstream by the DB query — only rows with
        # strategy=<engine> are passed to the predicate. Exercise the
        # caller layer (_select_owned_row) for that contract: empty
        # candidate list → no owned row, regardless of broker state.
        from trading.orphan_reconciler import _select_owned_row
        owned = _select_owned_row(_broker(symbol='NVDA'), [], TODAY, CFG)
        assert owned is None

    def test_manual_position_same_symbol_different_price(self):
        # User manually bought at $14 — broker shows $14, our DB has
        # a poisoned row at $10. Avg-entry mismatch → FOREIGN.
        assert not is_owned_orphan(
            _broker(avg_entry_price=14.00),
            _db(fill_price=10.00),
            TODAY, CFG)

    def test_manual_position_same_symbol_same_price_LARGER_qty(self):
        # Pathological: manual at the EXACT same price as our poisoned
        # row, BUT bigger qty. Qty sanity saves us.
        assert not is_owned_orphan(
            _broker(qty=500, avg_entry_price=10.00),
            _db(filled_qty=100, fill_price=10.00),
            TODAY, CFG)


# =========================================================================
# reconcile_strategy_orphans — end-to-end flow with mocked components
# =========================================================================

def _stub_db(rows_by_symbol=None):
    """A DB stub that satisfies the queries the reconciler issues.

    Implements the public `get_strategy_trades_in_window` API the
    reconciler uses; also keeps the legacy `_trades_conn.execute`
    fallback for back-compat with the old code path.
    """
    rows_by_symbol = rows_by_symbol or {}
    flat_rows = [r for rs in rows_by_symbol.values() for r in rs]
    db = MagicMock()

    def _get_strategy_trades(strategy, since_date, symbols=None):
        out = []
        for r in flat_rows:
            if r.get('strategy') != strategy:
                continue
            if str(r.get('trade_date') or '') < str(since_date):
                continue
            if symbols and r['symbol'] not in symbols:
                continue
            out.append(dict(r))
        return out

    db.get_strategy_trades_in_window = _get_strategy_trades

    class Cursor:
        def __init__(self, results):
            self._results = results
        def fetchall(self):
            return self._results

    def _execute(sql, params):
        symbol_filter = set(params[2:])
        out = [r for r in flat_rows if r['symbol'] in symbol_filter]
        return Cursor(out)

    db._trades_conn.execute = _execute
    db.update_trade = MagicMock()
    return db


def _stub_alpaca(positions=None, close_result=None, fill=None):
    a = MagicMock()
    a.get_open_positions.return_value = positions or []
    a.close_position.return_value = close_result or {'id': 'order-1'}
    a.get_order.return_value = fill or {'filled_qty': 100,
                                          'filled_avg_price': 10.00}
    return a


class TestReconcileEndToEnd:

    def test_no_orphans_no_actions(self):
        a = _stub_alpaca(positions=[])
        actions = reconcile_strategy_orphans(
            strategy='orb', alpaca=a, db=_stub_db(),
            notifier=None, tracked_symbols=set(),
        )
        assert actions == []
        a.close_position.assert_not_called()

    def test_foreign_position_silent_no_close(self):
        # Owner directive 2026-08-17 (BMNR incident): foreign positions are
        # the owner's own manual trades — log-only, NEVER close, NEVER
        # telegram. (Previously alert_only with a Telegram notify.)
        a = _stub_alpaca(positions=[_broker(symbol='UNKNOWN')])
        n = MagicMock()
        actions = reconcile_strategy_orphans(
            strategy='orb', alpaca=a, db=_stub_db({}),
            notifier=n, tracked_symbols=set(),
        )
        assert len(actions) == 1
        assert actions[0].classification == 'foreign'
        assert actions[0].action == 'log_only'
        a.close_position.assert_not_called()
        n.notify_error.assert_not_called()

    def test_owned_orphan_gets_closed(self):
        broker_pos = _broker(symbol='SMU', qty=5976,
                              avg_entry_price=14.6722)
        db_row = _db(symbol='SMU', strategy='macd_wave',
                     fill_price=14.6722, filled_qty=5976,
                     exit_reason='stop_loss_unconfirmed',
                     trade_date='2026-05-26')
        a = _stub_alpaca(positions=[broker_pos],
                          close_result={'id': 'closer-1'},
                          fill={'filled_qty': 5976,
                                'filled_avg_price': 11.82})
        actions = reconcile_strategy_orphans(
            strategy='macd_wave', alpaca=a,
            db=_stub_db({'SMU': [db_row]}),
            notifier=None, tracked_symbols=set(),
            today_et=date(2026, 6, 5),
        )
        assert len(actions) == 1
        assert actions[0].classification == 'owned'
        assert actions[0].action == 'closed'
        assert actions[0].close_order_id == 'closer-1'
        a.close_position.assert_called_once_with('SMU')

    def test_kill_switch_disables_close(self):
        broker_pos = _broker(symbol='SMU', qty=100,
                              avg_entry_price=10.00)
        db_row = _db(symbol='SMU', strategy='orb',
                     fill_price=10.00, filled_qty=100,
                     exit_reason='stop_loss_unconfirmed',
                     trade_date='2026-06-01')
        a = _stub_alpaca(positions=[broker_pos])
        cfg = ReconcilerConfig(auto_close_enabled=False)
        actions = reconcile_strategy_orphans(
            strategy='orb', alpaca=a, db=_stub_db({'SMU': [db_row]}),
            notifier=None, tracked_symbols=set(),
            cfg=cfg, today_et=date(2026, 6, 5),
        )
        assert len(actions) == 1
        assert actions[0].action == 'auto_close_disabled'
        a.close_position.assert_not_called()

    def test_rate_limit_caps_closes_per_hour(self):
        # Three OWNED orphans, cap=2 → first two close, third alerts
        broker = [_broker(symbol=f'S{i}', qty=10, avg_entry_price=10.0)
                  for i in range(3)]
        rows = {
            f'S{i}': [_db(symbol=f'S{i}', strategy='orb',
                          fill_price=10.0, filled_qty=10,
                          exit_reason='stop_loss_unconfirmed',
                          trade_date='2026-06-04', id=100 + i)]
            for i in range(3)
        }
        a = _stub_alpaca(positions=broker)
        cfg = ReconcilerConfig(max_closes_per_hour=2)
        actions = reconcile_strategy_orphans(
            strategy='orb', alpaca=a, db=_stub_db(rows),
            notifier=None, tracked_symbols=set(),
            cfg=cfg, today_et=date(2026, 6, 5),
        )
        assert sum(1 for x in actions if x.action == 'closed') == 2
        assert sum(1 for x in actions if x.action == 'cap_breached') == 1
        assert a.close_position.call_count == 2

    def test_tracked_symbol_not_treated_as_orphan(self):
        # Engine is tracking SMU; even though broker has it, no orphan.
        broker_pos = _broker(symbol='SMU')
        a = _stub_alpaca(positions=[broker_pos])
        actions = reconcile_strategy_orphans(
            strategy='orb', alpaca=a, db=_stub_db(),
            notifier=None, tracked_symbols={'SMU'},
        )
        assert actions == []

    def test_broker_snapshot_failure_returns_empty(self):
        a = MagicMock()
        a.get_open_positions.side_effect = RuntimeError("api down")
        actions = reconcile_strategy_orphans(
            strategy='orb', alpaca=a, db=_stub_db(),
            notifier=None, tracked_symbols=set(),
        )
        assert actions == []


class TestAlertCooldown:

    def test_foreign_orphan_never_alerts(self):
        # Owner directive 2026-08-17: foreign positions never telegram at
        # all — repeated reconcile cycles stay silent (was: one alert per
        # cooldown window, which spammed the owner about his own trades).
        broker_pos = _broker(symbol='UNKNOWN')
        a = _stub_alpaca(positions=[broker_pos])
        n = MagicMock()
        reconcile_strategy_orphans(
            strategy='orb', alpaca=a, db=_stub_db(),
            notifier=n, tracked_symbols=set(),
        )
        reconcile_strategy_orphans(
            strategy='orb', alpaca=a, db=_stub_db(),
            notifier=n, tracked_symbols=set(),
        )
        assert n.notify_error.call_count == 0


class TestDBRecoveryRowWritten:

    def test_owned_close_writes_recovery_row(self):
        broker_pos = _broker(symbol='SMU', qty=100,
                              avg_entry_price=14.0)
        db_row = _db(symbol='SMU', strategy='macd_wave',
                     fill_price=14.0, filled_qty=100,
                     exit_reason='stop_loss_unconfirmed',
                     trade_date='2026-05-26', id=357)
        a = _stub_alpaca(
            positions=[broker_pos],
            close_result={'id': 'cl-1'},
            fill={'filled_qty': 100, 'filled_avg_price': 11.82},
        )
        db = _stub_db({'SMU': [db_row]})
        reconcile_strategy_orphans(
            strategy='macd_wave', alpaca=a, db=db,
            notifier=None, tracked_symbols=set(),
            today_et=date(2026, 6, 5),
        )
        db.update_trade.assert_called_once()
        call_args = db.update_trade.call_args
        assert call_args[0][0] == 357  # trade_id
        update = call_args[0][1]
        assert update['exit_price'] == 11.82
        assert update['exit_reason'] == ORPHAN_RECOVERED_EXIT_REASON
        assert update['order_status'] == 'closed'
        # P&L: (11.82 - 14.0) * 100 = -218
        assert update['pnl'] == pytest.approx(-218.0)
