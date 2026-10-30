"""ORB daily Telegram report must state ledger truth (2026-10-30).

Defect 10/29: ``_sync_db_after_fc`` booked the force-close pnl to the ledger but
never to ``self.daily_pnl``; ``send_daily_report`` printed the in-memory number
(-$228) while the ledger closed the day at -$163.  Spec:
docs/orb_daily_report_ledger_truth_spec_20261030.md
"""
import logging
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from persistence.database import Database
from trading.orb_engine import ORBEngine


TODAY = datetime.now(timezone.utc).date().isoformat()


class FakeEngine:
    """Minimal stand-in: binds the two real methods under test to plain attrs."""

    def __init__(self, db, alpaca=None, daily_pnl=0.0, notifier=True):
        self.db = db
        self.alpaca = alpaca or MagicMock()
        self.daily_pnl = daily_pnl
        self.tg_prefix = '[ORB-LIVE]'
        self.daily_n_placed = 5
        self.daily_n_filled = 3
        self.daily_n_time_stop_canceled = 0
        self.notifier = notifier
        self.sent = []
        self._notify = self.sent.append
        self._et_now = lambda: datetime.fromisoformat(f"{TODAY}T15:50:00+00:00")
        self.check_exits = MagicMock()
        self.send_daily_report = ORBEngine.send_daily_report.__get__(self)
        self._sync_db_after_fc = ORBEngine._sync_db_after_fc.__get__(self)


def _row(sym, pnl, strategy='orb', trade_date=None):
    return {'symbol': sym, 'pnl': pnl, 'strategy': strategy,
            'trade_date': trade_date or TODAY}


def _db_with_rows(rows):
    db = MagicMock(spec=Database)
    db.get_trades_by_date.return_value = rows
    return db


def _sell(price, qty):
    return SimpleNamespace(
        id='abcdef123456', symbol='AXTQ', side=SimpleNamespace(value='sell'),
        status=SimpleNamespace(value='filled'), filled_avg_price=str(price),
        filled_qty=str(qty), filled_at=datetime.now(timezone.utc))


# ----------------------------------------------------------------- report unit
class TestDailyReportLedger:
    def test_reports_ledger_pnl_and_exit_count(self, caplog):
        db = _db_with_rows([_row('A', -237.0), _row('B', 9.0), _row('AXTQ', 65.0)])
        eng = FakeEngine(db, daily_pnl=-228.0)
        with caplog.at_level(logging.WARNING):
            eng.send_daily_report()
        msg = eng.sent[0]
        assert 'P&L=$-163' in msg
        assert '3 exits' in msg
        assert 'in-memory $-228' in msg
        assert any('ledger/in-memory P&L drift' in r.message and r.levelno == logging.WARNING
                   for r in caplog.records)
        db.get_trades_by_date.assert_called_once_with(TODAY)

    def test_no_drift_suffix_when_in_sync(self, caplog):
        db = _db_with_rows([_row('A', -237.0), _row('B', 9.0), _row('AXTQ', 65.0)])
        eng = FakeEngine(db, daily_pnl=-163.0)
        with caplog.at_level(logging.WARNING):
            eng.send_daily_report()
        assert 'in-memory' not in eng.sent[0]
        assert not any('drift' in r.message for r in caplog.records)

    def test_ignores_other_strategies_open_rows_and_other_days(self):
        db = _db_with_rows([
            _row('A', -100.0), _row('BF', 500.0, strategy='bull_flag'),
            _row('OPEN', None), _row('OLD', 999.0, trade_date='2026-01-02')])
        eng = FakeEngine(db, daily_pnl=-100.0)
        eng.send_daily_report()
        assert 'P&L=$-100' in eng.sent[0] and '1 exits' in eng.sent[0]

    def test_db_error_falls_back_to_in_memory_and_still_sends(self, caplog):
        db = MagicMock(spec=Database)
        db.get_trades_by_date.side_effect = sqlite3.OperationalError('locked')
        eng = FakeEngine(db, daily_pnl=-228.0)
        with caplog.at_level(logging.WARNING):
            eng.send_daily_report()
        assert len(eng.sent) == 1
        assert 'P&L=$-228' in eng.sent[0]
        assert any(r.levelno == logging.WARNING and 'ledger' in r.message.lower()
                   for r in caplog.records)


# -------------------------------------------------------------- sync unit tests
def _open_row(**kw):
    r = {'id': 7, 'symbol': 'AXTQ', 'fill_price': 10.0, 'shares': 100,
         'filled_qty': 100, 'exit_price': None, 'pnl': None}
    r.update(kw)
    return r


class TestSyncBooksDailyPnl:
    def _engine(self, rows, sell_px=10.65):
        db = MagicMock(spec=Database)
        db.get_open_trades.return_value = rows
        alpaca = MagicMock()
        alpaca.trading_client.get_orders.return_value = [_sell(sell_px, 100)]
        return FakeEngine(db, alpaca=alpaca, daily_pnl=-228.0), db

    def test_books_pnl_once_into_daily_pnl(self, caplog):
        eng, db = self._engine([_open_row()])
        with caplog.at_level(logging.INFO):
            eng._sync_db_after_fc(['AXTQ'])
        assert eng.daily_pnl == pytest.approx(-228.0 + 65.0)
        assert any('ORB FC SYNC: AXTQ pnl=$' in r.message and 'daily_pnl=$' in r.message
                   for r in caplog.records)
        assert db.update_trade.call_count == 1

    def test_second_call_on_pnl_set_row_adds_zero(self):
        # the DB now returns the row with pnl/exit set (e.g. open-query quirk)
        eng, db = self._engine([_open_row(pnl=65.0)])
        eng._sync_db_after_fc(['AXTQ'])
        assert eng.daily_pnl == pytest.approx(-228.0)
        db.update_trade.assert_not_called()

    def test_failed_db_update_does_not_book(self):
        eng, db = self._engine([_open_row()])
        db.update_trade.side_effect = sqlite3.OperationalError('locked')
        eng._sync_db_after_fc(['AXTQ'])
        assert eng.daily_pnl == pytest.approx(-228.0)


# ------------------------------------------------------------------ integration
class TestLedgerIntegration:
    @pytest.fixture
    def real_db(self, tmp_path):
        db = Database(cache_path=str(tmp_path / 'cache.db'),
                      trades_path=str(tmp_path / 'trades.db'))
        yield db

    def _save(self, db, sym, pnl=None, exit_price=None, order_status='filled'):
        return db.save_trade({
            'trade_date': TODAY, 'symbol': sym, 'side': 'buy',
            'entry_price': 10.0, 'stop_loss_price': 9.0, 'take_profit_price': 12.0,
            'shares': 100, 'risk_per_share': 1.0, 'total_risk': 100.0,
            'risk_reward_ratio': 2.0, 'order_id': f'o-{sym}',
            'order_status': order_status, 'fill_price': 10.0, 'filled_at': None,
            'exit_price': exit_price, 'exit_reason': None, 'exited_at': None,
            'pnl': pnl, 'pnl_pct': None, 'pattern_data': None,
            'strategy': 'orb', 'account': 'live'})

    def test_report_after_force_close_sync_matches_ledger(self, real_db):
        self._save(real_db, 'NEBX', pnl=-237.0, exit_price=7.63, order_status='closed')
        self._save(real_db, 'CORD', pnl=9.0, exit_price=10.09, order_status='closed')
        self._save(real_db, 'AXTQ')  # filled, no exit yet -> force-closed below
        alpaca = MagicMock()
        alpaca.trading_client.get_orders.return_value = [_sell(10.65, 100)]
        # in-memory saw only the two StopMonitor exits
        eng = FakeEngine(real_db, alpaca=alpaca, daily_pnl=-228.0)

        eng._sync_db_after_fc(['AXTQ'])
        assert eng.daily_pnl == pytest.approx(-163.0)      # booked exactly once
        eng._sync_db_after_fc(['AXTQ'])                    # row now closed
        assert eng.daily_pnl == pytest.approx(-163.0)

        eng.send_daily_report()
        assert 'P&L=$-163' in eng.sent[0]
        assert '3 exits' in eng.sent[0]
        assert 'in-memory' not in eng.sent[0]

    def test_report_flags_drift_against_real_ledger(self, real_db):
        self._save(real_db, 'NEBX', pnl=-237.0, exit_price=7.63, order_status='closed')
        self._save(real_db, 'CORD', pnl=9.0, exit_price=10.09, order_status='closed')
        self._save(real_db, 'AXTQ', pnl=65.0, exit_price=10.65, order_status='closed')
        eng = FakeEngine(real_db, daily_pnl=-228.0)
        eng.send_daily_report()
        assert 'P&L=$-163 (3 exits) | in-memory $-228' in eng.sent[0]
