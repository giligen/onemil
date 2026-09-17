"""Pin the contract that BRANCH_LAST_RESORT no longer writes a fake-confirmed
exit. Catches the regression that produced the SMU + QBTZ orphans
(2026-05-26 / 2026-06-01 — 10 + 4 days of unmanaged broker exposure).

This file is intentionally light on integration scope: the StopMonitor's
order-flow is exercised by tests/test_stop_monitor.py. Here we pin the
narrow data contract:

  1. StopExitEvent.confirmed defaults to True.
  2. build_exit_update produces a no-exit_price/no-pnl payload when
     confirmed=False and a full payload when True.
  3. database._ACTIVE_ORDER_STATUSES contains
     'exit_pending_verification' so get_open_trades() returns rows in
     that state (the orphan reconciler depends on this).
"""
from __future__ import annotations

import pytest

from trading.stop_monitor import StopExitEvent, build_exit_update
from persistence.database import Database


# =========================================================================
# StopExitEvent
# =========================================================================

class TestStopExitEventDefault:

    def test_confirmed_defaults_true(self):
        ev = StopExitEvent(
            symbol='X', stop_price=10.0, exit_price=9.5, shares=100,
            order_id='id-1', exit_reason='stop_loss',
        )
        assert ev.confirmed is True

    def test_confirmed_can_be_false(self):
        ev = StopExitEvent(
            symbol='X', stop_price=10.0, exit_price=10.0, shares=100,
            order_id='', exit_reason='stop_loss_unconfirmed', confirmed=False,
        )
        assert ev.confirmed is False


# =========================================================================
# build_exit_update — the shared DB-write helper
# =========================================================================

class TestBuildExitUpdateConfirmed:

    def _ev(self, **kw):
        defaults = dict(
            symbol='X', stop_price=10.0, exit_price=9.50, shares=100,
            order_id='id-1', exit_reason='stop_loss',
            exit_trigger_price=9.48, exit_quote_bid=9.49, exit_quote_ask=9.51,
            exit_quote_bid_size=100, exit_quote_ask_size=200,
            exit_limit_price=9.49, pricing_method='quote_tight',
        )
        defaults.update(kw)
        return StopExitEvent(**defaults)

    def test_confirmed_writes_exit_price(self):
        out = build_exit_update(self._ev(confirmed=True))
        assert out['exit_price'] == 9.50
        assert out['order_status'] == 'closed'
        assert 'exited_at' in out

    def test_confirmed_writes_microstructure(self):
        out = build_exit_update(self._ev(confirmed=True))
        assert out['exit_reason'] == 'stop_loss'
        assert out['exit_trigger_price'] == 9.48
        assert out['exit_quote_bid'] == 9.49
        assert out['exit_quote_ask'] == 9.51
        assert out['exit_limit_price'] == 9.49
        assert out['exit_pricing_method'] == 'quote_tight'

    def test_unused_microstructure_set_to_none(self):
        out = build_exit_update(self._ev(
            confirmed=True,
            exit_trigger_price=0.0, exit_quote_bid=0.0,
            exit_quote_ask=0.0, exit_limit_price=0.0,
        ))
        assert out['exit_trigger_price'] is None
        assert out['exit_quote_bid'] is None
        assert out['exit_quote_ask'] is None
        assert out['exit_limit_price'] is None


class TestBuildExitUpdateUnconfirmed:
    """The critical contract: UNCONFIRMED events must NOT write a fake
    exit. This is the regression-prevention test for SMU/QBTZ."""

    def _unconfirmed(self):
        return StopExitEvent(
            symbol='X', stop_price=10.0, exit_price=10.0, shares=100,
            order_id='', exit_reason='stop_loss_unconfirmed',
            exit_trigger_price=10.0, confirmed=False,
        )

    def test_unconfirmed_no_exit_price(self):
        out = build_exit_update(self._unconfirmed())
        assert 'exit_price' not in out

    def test_unconfirmed_no_exited_at(self):
        out = build_exit_update(self._unconfirmed())
        assert 'exited_at' not in out

    def test_unconfirmed_no_pnl(self):
        # pnl/pnl_pct are caller-added; build_exit_update must not include
        # them, and the caller's confirmed-branch is the only place they
        # get set. This pins that the helper itself never adds them.
        out = build_exit_update(self._unconfirmed())
        assert 'pnl' not in out
        assert 'pnl_pct' not in out

    def test_unconfirmed_status_is_pending_verification(self):
        out = build_exit_update(self._unconfirmed())
        assert out['order_status'] == 'exit_pending_verification'

    def test_unconfirmed_preserves_exit_reason_for_forensics(self):
        out = build_exit_update(self._unconfirmed())
        assert out['exit_reason'] == 'stop_loss_unconfirmed'

    def test_unconfirmed_preserves_trigger_price(self):
        # The trigger price + quote context are still useful for forensics
        # even when the actual exit isn't confirmed.
        out = build_exit_update(self._unconfirmed())
        assert out['exit_trigger_price'] == 10.0


# =========================================================================
# Database active-status contract
# =========================================================================

class TestActiveStatusesContract:

    def test_exit_pending_verification_is_active(self):
        assert 'exit_pending_verification' in Database._ACTIVE_ORDER_STATUSES

    def test_active_statuses_still_includes_old_ones(self):
        # Regression: don't drop anything.
        for s in ('filled', 'partially_filled', 'pending_new', 'accepted', 'new'):
            assert s in Database._ACTIVE_ORDER_STATUSES


# =========================================================================
# get_open_trades returns exit_pending_verification rows
# =========================================================================

class TestGetOpenTradesReturnsPending(object):
    """Pins the wire-up: a row with status='exit_pending_verification'
    appears in get_open_trades — that's the signal the orphan reconciler
    relies on to retry."""

    @pytest.fixture
    def db(self, tmp_path):
        db_path = str(tmp_path / 'trades.db')
        d = Database(db_path=db_path, cache_path=str(tmp_path / 'cache.db'),
                     trades_path=db_path)
        return d

    def test_exit_pending_verification_row_returned(self, db):
        # Minimal insert via direct SQL (we don't want to depend on the
        # full trade-record-creation path here).
        db._trades_conn.execute("""
            INSERT INTO trades (
                trade_date, symbol, side, entry_price, stop_loss_price,
                take_profit_price, shares, risk_per_share, total_risk,
                risk_reward_ratio, order_status, fill_price, filled_qty,
                strategy, created_at, updated_at
            ) VALUES (?, 'TST', 'buy', 10, 9, 12, 100, 1, 100, 2,
                      'exit_pending_verification', 10.0, 100, 'orb',
                      '2026-06-05T00:00:00', '2026-06-05T00:00:00')
        """, ('2026-06-05',))
        db._trades_conn.commit()

        rows = db.get_open_trades('2026-06-05', strategy='orb')
        assert len(rows) == 1
        assert rows[0]['symbol'] == 'TST'
        assert rows[0]['order_status'] == 'exit_pending_verification'

    def test_closed_row_not_returned(self, db):
        # Sanity: clean exit still excluded.
        db._trades_conn.execute("""
            INSERT INTO trades (
                trade_date, symbol, side, entry_price, stop_loss_price,
                take_profit_price, shares, risk_per_share, total_risk,
                risk_reward_ratio, order_status, fill_price, filled_qty,
                exit_price, exit_reason, strategy,
                created_at, updated_at
            ) VALUES (?, 'TST', 'buy', 10, 9, 12, 100, 1, 100, 2,
                      'closed', 10.0, 100, 11.0, 'trail_stop', 'orb',
                      '2026-06-05T00:00:00', '2026-06-05T00:00:00')
        """, ('2026-06-05',))
        db._trades_conn.commit()
        rows = db.get_open_trades('2026-06-05', strategy='orb')
        assert rows == []


# =========================================================================
# D3 FIX 7 — `unknown_exit` must never write a price or a P&L
# =========================================================================
#
# research/fuckup_audit/D3_exec/REPORT.md §M7: four rows were booked with
# exit_price == fill_price and pnl == 0.00 EXACTLY — BDMD 2026-03-20 and
# NPT / FBYD / SVRN 2026-03-30. They were not breakeven trades; they were
# trades whose exit fill was never found, written down as flat so the
# reconcile loop would stop revisiting them. NPT alone: entry $5.49, 16:52
# close $4.96, 9,090 shares → -$4,818 booked as $0.00. Between -$1,684 and
# -$6,800 of realized loss invisible to every book that reads trades.pnl.

from trading.unknown_exit import (          # noqa: E402
    PENDING_VERIFICATION_STATUS,
    build_unknown_exit_update,
    sweep_unknown_exits,
)


class TestUnknownExitWritesNoPnl:
    """The contract, as a payload."""

    def test_unknown_exit_writes_no_pnl(self):
        out = build_unknown_exit_update()
        assert 'pnl' not in out
        assert 'pnl_pct' not in out

    def test_unknown_exit_writes_no_price(self):
        out = build_unknown_exit_update()
        assert 'exit_price' not in out

    def test_unknown_exit_writes_no_exited_at(self):
        out = build_unknown_exit_update()
        assert 'exited_at' not in out

    def test_status_is_pending_verification(self):
        out = build_unknown_exit_update()
        assert out['order_status'] == PENDING_VERIFICATION_STATUS

    def test_reason_is_unknown_exit(self):
        from trading.exit_reasons import ExitReason, needs_reconcile
        out = build_unknown_exit_update()
        assert out['exit_reason'] == ExitReason.UNKNOWN_EXIT.value
        assert needs_reconcile(out['exit_reason']) is True

    def test_trigger_price_is_forensics_only(self):
        """A last-known price may be recorded for forensics — but never
        as the exit price and never into P&L."""
        out = build_unknown_exit_update(trigger_price=4.96)
        assert out['exit_trigger_price'] == 4.96
        assert 'exit_price' not in out
        assert 'pnl' not in out

    def test_zero_and_none_trigger_price_omitted(self):
        assert 'exit_trigger_price' not in build_unknown_exit_update(0.0)
        assert 'exit_trigger_price' not in build_unknown_exit_update(None)

    def test_non_numeric_trigger_price_warns_and_is_dropped(self, caplog):
        import logging
        caplog.set_level(logging.WARNING, logger='trading.unknown_exit')
        out = build_unknown_exit_update(trigger_price='n/a')
        assert 'exit_trigger_price' not in out
        assert any('non-numeric' in r.getMessage() for r in caplog.records)


class TestNptRowIsNotBookedFlat:
    """The NPT 2026-03-30 row, end to end against a real Database."""

    @pytest.fixture
    def db(self, tmp_path):
        p = str(tmp_path / 'trades.db')
        return Database(db_path=p, cache_path=str(tmp_path / 'cache.db'),
                        trades_path=p)

    def _npt(self, db):
        db._trades_conn.execute("""
            INSERT INTO trades (
                trade_date, symbol, side, entry_price, stop_loss_price,
                take_profit_price, shares, risk_per_share, total_risk,
                risk_reward_ratio, order_status, fill_price, filled_qty,
                strategy, created_at, updated_at
            ) VALUES ('2026-03-30', 'NPT', 'buy', 5.49, 5.39, 5.69, 9090,
                      0.10, 909, 2, 'filled', 5.49, 9090, 'macd_wave',
                      '2026-03-30T00:00:00', '2026-03-30T00:00:00')
        """)
        db._trades_conn.commit()
        return db._trades_conn.execute(
            "SELECT id FROM trades WHERE symbol='NPT'").fetchone()[0]

    def test_row_left_pending_with_null_pnl(self, db):
        tid = self._npt(db)
        db.update_trade(tid, build_unknown_exit_update())
        row = db._trades_conn.execute(
            "SELECT exit_price, exited_at, pnl, pnl_pct, order_status, "
            "exit_reason FROM trades WHERE id=?", (tid,)).fetchone()
        assert row[0] is None, "exit_price must stay NULL — we never saw a fill"
        assert row[1] is None
        assert row[2] is None, "a fabricated $0 P&L is the defect"
        assert row[3] is None
        assert row[4] == PENDING_VERIFICATION_STATUS
        assert row[5] == 'unknown_exit'

    def test_row_stays_visible_to_the_reconciler(self, db):
        """Pending is not a loose end: the row keeps showing up in
        get_open_trades, which is what the orphan reconciler and the
        daily green check's HARD fail both read."""
        tid = self._npt(db)
        db.update_trade(tid, build_unknown_exit_update())
        rows = db.get_open_trades('2026-03-30', strategy='macd_wave')
        assert [r['symbol'] for r in rows] == ['NPT']

    def test_the_old_fabricated_flat_is_what_we_no_longer_write(self, db):
        """Documents the defect this test file exists to prevent: the old
        payload produced a row that reads as a clean breakeven trade."""
        tid = self._npt(db)
        db.update_trade(tid, {
            'exit_price': 5.49, 'exit_reason': 'unknown_exit',
            'pnl': 0.0, 'pnl_pct': 0.0, 'order_status': 'closed',
        })
        row = db._trades_conn.execute(
            "SELECT exit_price, fill_price, pnl FROM trades WHERE id=?",
            (tid,)).fetchone()
        # exit == fill and pnl == 0.00 exactly: the D3 §M7 signature.
        assert row[0] == row[1] and row[2] == 0.0
        # And it disappears from the reconciler's view — the reason nobody
        # noticed -$4,818.
        assert db.get_open_trades('2026-03-30', strategy='macd_wave') == []


class TestSweepStub:
    """The daily sweep is deliberately NOT built (D3 FIX 7 ships the half
    that stops the bleeding). The stub must be honest about that."""

    def test_sweep_resolves_nothing_and_says_so(self, caplog):
        import logging
        caplog.set_level(logging.WARNING, logger='trading.unknown_exit')
        assert sweep_unknown_exits(db=None) == []
        msgs = [r.getMessage() for r in caplog.records]
        assert any('NOT IMPLEMENTED' in m for m in msgs), msgs

    def test_sweep_never_writes(self):
        """A stub that quietly wrote P&L would be worse than none."""
        from unittest.mock import MagicMock
        db = MagicMock(spec=Database)
        sweep_unknown_exits(db=db, older_than_session=True)
        db.update_trade.assert_not_called()
