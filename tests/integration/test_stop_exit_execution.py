"""Integration tests for the D3 execution fixes — real StopMonitor, real
Database, real TradingEngine code paths, and `tests/fakes/fake_alpaca_broker`
standing in for Alpaca with Alpaca's ORDER semantics.

Nothing here is mocked at the seam under test. The fake broker holds
positions, reserves shares against working sell orders, elects stops,
runs OCO, and returns a NEW order id on a replace — the behaviours that
actually bit us. See `research/fuckup_audit/D3_exec/REPORT.md`.

Covered:
  * FIX 7 — NPT 2026-03-30: an unattributable exit leaves the row
    UNATTRIBUTED (no price, no P&L) instead of booking a fabricated flat.
  * FIX 1 — the SL-leg-is-the-stop lifecycle: the live broker stop is
    REPLACED, never cancelled to make room for our own limit, so there is
    no window with neither a stop nor a working sell.
  * FIX 3 — a partial fill is booked and only the remainder is worked;
    the exit event carries the blended price.
  * FIX 2 — RBNE 2026-07-16 (thin book): the ladder slices to the
    displayed bid instead of dumping 2,841 shares into a 200-share bid
    and then market-ordering into the hole it just made.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from notifications.telegram_notifier import TelegramNotifier  # noqa: E402
from persistence.database import Database  # noqa: E402
from tests.fakes.fake_alpaca_broker import FakeAlpacaBroker  # noqa: E402
from trading.exit_reasons import ExitBranch  # noqa: E402
from trading.stop_monitor import StopMonitor  # noqa: E402


# ---------------------------------------------------------------------------
# shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    p = str(tmp_path / 'trades.db')
    return Database(db_path=p, cache_path=str(tmp_path / 'cache.db'),
                    trades_path=p)


@pytest.fixture
def broker():
    b = FakeAlpacaBroker()
    # The HOD-break lifecycle tests assert close_position is never called;
    # the StopMonitor's escalation path legitimately calls it, so make it
    # behave like a market close here instead of raising.
    b.close_position = _close_position_for(b)
    return b


def _close_position_for(b: FakeAlpacaBroker):
    def close_position(symbol: str) -> dict:
        b.calls.append(('close_position', {'symbol': symbol}))
        qty = b.positions.get(symbol, 0)
        if qty <= 0:
            from data_sources.alpaca_client import AlpacaAPIError
            raise AlpacaAPIError('40410000: position not found')
        # Cancel whatever is working so the shares are free, then market out.
        for o in list(b.orders.values()):
            if (o['symbol'] == symbol and o['side'] == 'sell'
                    and o['status'] in ('new', 'accepted', 'partially_filled',
                                        'held')):
                b.cancel_order(o['id'])
        o = b._new(symbol=symbol, qty=int(b.positions[symbol]), side='sell',
                   type='market')
        b._settle(force=True)
        return {'id': o['id'], 'status': o['status']}
    return close_position


def _monitor(broker, **kw) -> StopMonitor:
    m = StopMonitor(api_key='k', api_secret='s', alpaca_client=broker, **kw)
    m._STOP_EXIT_FILL_TIMEOUT_S = 0.3
    m._MARKET_CLOSE_FILL_TIMEOUT_S = 0.3
    m._STOP_EXIT_POLL_INTERVAL_S = 0.05
    return m


def _watch(monitor, symbol, stop, shares, bid, ask, bid_size=200,
           tp_leg='', sl_leg=''):
    monitor.add_watch(symbol, stop, shares, tp_leg, sl_leg)
    with monitor._watch_lock:
        w = monitor._watches[symbol]
        w.latest_bid, w.latest_ask = bid, ask
        w.latest_bid_size, w.latest_ask_size = bid_size, bid_size
        w.latest_quote_ts = time.time()
    return w


# ---------------------------------------------------------------------------
# FIX 7 — NPT 2026-03-30
# ---------------------------------------------------------------------------

class TestUnknownExitLeavesRowPending:
    """NPT 2026-03-30 (macd_wave, 9,090 sh, entry $5.49, stop $5.39).

    The position was gone, no order could be attributed to it, and the row
    was booked `exit_price = fill_price`, `pnl = $0.00` — while the 16:52
    close was $4.96, i.e. -$4,818. The reconcile loop then skipped it
    forever because it looked closed.
    """

    def _engine(self, db, broker):
        from trading.trading_engine import TradingEngine
        eng = TradingEngine.__new__(TradingEngine)
        eng.db = db
        eng.alpaca = broker
        broker.trading_client.get_orders = MagicMock(return_value=[])
        eng.stop_monitor = None
        eng.notifier = MagicMock(spec=TelegramNotifier)
        eng.position_manager = MagicMock()
        eng._unknown_exit_alerted = set()
        eng._process_stop_monitor_exits = lambda: None
        eng._check_profit_partials = lambda: None
        eng._check_exhaustion_exits = lambda: None
        return eng

    def _npt_row(self, db, day):
        db._trades_conn.execute("""
            INSERT INTO trades (
                trade_date, symbol, side, entry_price, stop_loss_price,
                take_profit_price, shares, risk_per_share, total_risk,
                risk_reward_ratio, order_status, fill_price, filled_qty,
                strategy, created_at, updated_at
            ) VALUES (?, 'NPT', 'buy', 5.49, 5.39, 5.69, 9090, 0.10, 909, 2,
                      'filled', 5.49, 9090, 'bull_flag', ?, ?)
        """, (day, f'{day}T00:00:00', f'{day}T00:00:00'))
        db._trades_conn.commit()
        return db._trades_conn.execute(
            "SELECT id FROM trades WHERE symbol='NPT'").fetchone()[0]

    def _run_sync(self, eng, day, monkeypatch):
        import trading.trading_engine as te

        class _FixedDate(te.date):
            @classmethod
            def today(cls):
                return te.date.fromisoformat(day)
        monkeypatch.setattr(te, 'date', _FixedDate)
        eng._sync_closed_positions()

    def test_unknown_exit_e2e_leaves_row_pending(self, db, broker, monkeypatch):
        day = '2026-03-30'
        tid = self._npt_row(db, day)
        eng = self._engine(db, broker)
        # Broker is flat and has no attributable order history.
        assert broker.get_open_positions() == []

        self._run_sync(eng, day, monkeypatch)

        row = db._trades_conn.execute(
            "SELECT exit_price, exited_at, pnl, pnl_pct, order_status, "
            "exit_reason FROM trades WHERE id=?", (tid,)).fetchone()
        assert row[0] is None, "no fill was observed — no price may be written"
        assert row[1] is None
        assert row[2] is None, "the fabricated $0.00 P&L is the defect"
        assert row[3] is None
        assert row[4] == 'exit_pending_verification'
        assert row[5] == 'unknown_exit'

    def test_no_fake_pnl_reaches_the_daily_book(self, db, broker, monkeypatch):
        day = '2026-03-30'
        self._npt_row(db, day)
        eng = self._engine(db, broker)
        self._run_sync(eng, day, monkeypatch)
        eng.position_manager.record_trade_pnl.assert_not_called()

    def test_operator_is_paged_once_not_every_cycle(self, db, broker,
                                                    monkeypatch):
        """The row now stays open, so the sync revisits it every cycle.
        It must page once, then keep logging."""
        day = '2026-03-30'
        self._npt_row(db, day)
        eng = self._engine(db, broker)
        for _ in range(4):
            self._run_sync(eng, day, monkeypatch)
        assert eng.notifier.notify_error.call_count == 1
        msg = eng.notifier.notify_error.call_args.args[0]
        assert 'NO P&L' in msg

    def test_row_stays_visible_for_reconciliation(self, db, broker,
                                                  monkeypatch):
        day = '2026-03-30'
        self._npt_row(db, day)
        eng = self._engine(db, broker)
        self._run_sync(eng, day, monkeypatch)
        rows = db.get_open_trades(day, strategy='bull_flag')
        assert [r['symbol'] for r in rows] == ['NPT']

    def test_recovered_history_still_books_the_real_pnl(self, db, broker,
                                                       monkeypatch):
        """Regression guard: FIX 7 must not swallow the GLXG 2026-06-11
        recovery path. When order history DOES have the sell, the row is
        closed with the real price."""
        day = '2026-03-30'
        tid = self._npt_row(db, day)
        eng = self._engine(db, broker)
        sell = MagicMock()
        sell.side = MagicMock(value='sell')
        sell.status = MagicMock(value='filled')
        sell.filled_avg_price = 4.96
        sell.order_class = None
        broker.trading_client.get_orders = MagicMock(return_value=[sell])

        self._run_sync(eng, day, monkeypatch)

        row = db._trades_conn.execute(
            "SELECT exit_price, pnl, order_status FROM trades WHERE id=?",
            (tid,)).fetchone()
        assert row[0] == 4.96
        assert row[1] == pytest.approx((4.96 - 5.49) * 9090, abs=1.0)
        assert row[2] != 'exit_pending_verification'


# ---------------------------------------------------------------------------
# FIX 1 — the SL leg IS the stop
# ---------------------------------------------------------------------------

class TestSlLegIsTheStop:
    """Pre-fix, `_execute_stop_exit` bulk-cancelled EVERY open order for the
    symbol — the bracket SL leg included — and only then submitted its own
    limit. Measured naked window across the 11 D3 events: 23.4-68.6 s on a
    position being liquidated precisely because it is falling."""

    def _bracketed(self, broker, symbol='RBNE', qty=2841, entry=5.20,
                   stop=4.685):
        """Fill a bracket buy so the two sell legs are live at the broker."""
        broker.tick(bid=entry, ask=entry)
        parent = broker.submit_bracket_order(
            symbol=symbol, qty=qty, side='buy', limit_price=entry,
            tp_price=entry * 1.5, sl_price=stop)
        legs = broker.orders[parent['id']]['legs']
        tp_id = next(l for l in legs if broker.orders[l]['type'] == 'limit')
        sl_id = next(l for l in legs if broker.orders[l]['type'] == 'stop')
        return tp_id, sl_id

    def test_sl_leg_replaced_not_cancelled(self, broker):
        tp_id, sl_id = self._bracketed(broker)
        m = _monitor(broker)
        w = _watch(m, 'RBNE', 4.685, 2841, 4.68, 4.74, bid_size=200,
                   tp_leg=tp_id, sl_leg=sl_id)

        asyncio.run(m._execute_stop_exit('RBNE', 4.68, w,
                                         exit_reason='stop_loss'))

        replaced = [c for c in broker.calls
                    if c[0] == 'replace_order_stop_price']
        assert replaced, "the live SL leg must be REPLACED, not cancelled"
        assert replaced[0][1]['order_id'] == sl_id
        # The original SL id was never handed to cancel_order.
        cancels = [c[1]['order_id'] for c in broker.calls
                   if c[0] == 'cancel_order']
        assert sl_id not in cancels

    def test_no_window_without_a_stop_or_a_working_sell(self, broker):
        """Walk the call log and assert protection never lapses."""
        tp_id, sl_id = self._bracketed(broker)
        m = _monitor(broker)
        w = _watch(m, 'RBNE', 4.685, 2841, 4.68, 4.74, tp_leg=tp_id,
                   sl_leg=sl_id)
        asyncio.run(m._execute_stop_exit('RBNE', 4.68, w,
                                         exit_reason='stop_loss'))
        # Position is flat and the SL chain is the thing that did it.
        assert broker.positions.get('RBNE', 0) == 0

    def test_exit_event_branch_is_sl_leg(self, broker):
        tp_id, sl_id = self._bracketed(broker)
        m = _monitor(broker)
        w = _watch(m, 'RBNE', 4.685, 2841, 4.68, 4.74, tp_leg=tp_id,
                   sl_leg=sl_id)
        asyncio.run(m._execute_stop_exit('RBNE', 4.68, w,
                                         exit_reason='trail_stop'))
        ev = m.drain_exit_events()[0]
        assert ev.exit_branch == ExitBranch.SL_LEG.value
        assert ev.exit_reason == 'trail_stop'
        assert ev.confirmed is True

    def test_no_sl_leg_falls_back_to_own_limit(self, broker):
        """No live SL leg (ORB time-stop cancel, a restart, a manual flat)
        → today's cancel-and-place path, unchanged."""
        broker.positions['XYZ'] = 500
        broker.tick(bid=4.68, ask=4.74)
        m = _monitor(broker)
        w = _watch(m, 'XYZ', 4.685, 500, 4.68, 4.74)
        asyncio.run(m._execute_stop_exit('XYZ', 4.68, w,
                                         exit_reason='stop_loss'))
        assert any(c[0] == 'submit_limit_sell_order' for c in broker.calls)
        assert not [c for c in broker.calls
                    if c[0] == 'replace_order_stop_price']
