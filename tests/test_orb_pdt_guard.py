"""ORB PDT (pattern-day-trader) guard — B+ RESTART 2026-08-15 (design §P1-10).

Every ORB trade is a same-day round trip = 1 day trade. On a MARGIN account
with equity < $25K, block the entry that would be the 4th day-trade in a
rolling 5-business-day window (DB-derived count, restart-safe). Cash accounts:
counter dormant (PDT N/A), account type logged. Reads account.multiplier +
equity + daytrade_count from get_account_info. Uses a REAL Database.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import timedelta
from unittest.mock import MagicMock

import pytest
import yaml
from pathlib import Path

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine
from trading.stop_monitor import StopMonitor


@pytest.fixture
def orb_cfg():
    cfg = yaml.safe_load(open(Path(__file__).parent.parent / 'orb.yaml'))
    cfg['strategy']['enabled'] = True
    return cfg


@pytest.fixture
def tmp_db():
    d = tempfile.mkdtemp()
    return Database(trades_path=os.path.join(d, 'trades.db'),
                    cache_path=os.path.join(d, 'cache.db'))


def _mk(orb_cfg, tmp_db, account, monkeypatch=None):
    al = MagicMock(spec=AlpacaClient)
    al.get_account_info.return_value = account
    e = ORBEngine(alpaca_client=al, db=tmp_db,
                  stop_monitor=MagicMock(spec=StopMonitor), config=orb_cfg)
    return e


def _fill(db, trade_date, symbol):
    conn = sqlite3.connect(db._trades_path)
    conn.execute(
        """INSERT INTO trades
        (trade_date,symbol,strategy,side,entry_price,stop_loss_price,
         take_profit_price,shares,risk_per_share,total_risk,risk_reward_ratio,
         created_at,updated_at,fill_price)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (trade_date, symbol, 'orb', 'buy', 10.0, 9.0, 12.0, 100, 1.0, 100.0,
         2.0, 'x', 'x', 10.0))
    conn.commit()
    conn.close()


MARGIN_LOW = {'equity': 10000.0, 'multiplier': 4.0, 'daytrade_count': 0,
              'buying_power': 40000.0}
MARGIN_HIGH = {'equity': 30000.0, 'multiplier': 4.0, 'daytrade_count': 0,
               'buying_power': 120000.0}
CASH = {'equity': 10000.0, 'multiplier': 1.0, 'daytrade_count': 0,
        'buying_power': 10000.0}


class TestConfig:
    def test_yaml_knobs(self, orb_cfg, tmp_db):
        e = _mk(orb_cfg, tmp_db, MARGIN_LOW)
        assert e.pdt_guard_enabled is True
        assert e.pdt_max_daytrades_5d == 3
        assert e.pdt_equity_threshold_usd == 25000

    def test_env_disable(self, orb_cfg, tmp_db, monkeypatch):
        monkeypatch.setenv('ORB_PDT_GUARD', '0')
        e = _mk(orb_cfg, tmp_db, MARGIN_LOW)
        e.init_account_guards()
        assert e.pdt_guard_enabled is False
        assert e._pdt_active is False
        assert e._pdt_would_block() is False


class TestAccountType:
    def test_margin_low_equity_active(self, orb_cfg, tmp_db):
        e = _mk(orb_cfg, tmp_db, MARGIN_LOW)
        e.init_account_guards()
        assert e._pdt_account_type == 'margin'
        assert e._pdt_active is True

    def test_margin_high_equity_dormant(self, orb_cfg, tmp_db):
        e = _mk(orb_cfg, tmp_db, MARGIN_HIGH)
        e.init_account_guards()
        assert e._pdt_active is False

    def test_cash_dormant(self, orb_cfg, tmp_db):
        e = _mk(orb_cfg, tmp_db, CASH)
        e.init_account_guards()
        assert e._pdt_account_type == 'cash'
        assert e._pdt_active is False

    def test_account_read_failure_dormant(self, orb_cfg, tmp_db):
        al = MagicMock(spec=AlpacaClient)
        al.get_account_info.side_effect = RuntimeError('api down')
        e = ORBEngine(alpaca_client=al, db=tmp_db,
                      stop_monitor=MagicMock(spec=StopMonitor), config=orb_cfg)
        e.init_account_guards()
        assert e._pdt_active is False
        assert e._pdt_account_type == 'unknown'


class TestCounter:
    def test_count_filled_in_window(self, orb_cfg, tmp_db):
        e = _mk(orb_cfg, tmp_db, MARGIN_LOW)
        today = e._et_now().strftime('%Y-%m-%d')
        _fill(tmp_db, today, 'A')
        _fill(tmp_db, today, 'B')
        assert e._count_orb_daytrades_5d() == 2

    def test_old_trades_outside_window_excluded(self, orb_cfg, tmp_db):
        e = _mk(orb_cfg, tmp_db, MARGIN_LOW)
        old = (e._et_now() - timedelta(days=20)).strftime('%Y-%m-%d')
        _fill(tmp_db, old, 'OLD')
        assert e._count_orb_daytrades_5d() == 0

    def test_block_at_cap(self, orb_cfg, tmp_db):
        e = _mk(orb_cfg, tmp_db, MARGIN_LOW)
        e.init_account_guards()
        today = e._et_now().strftime('%Y-%m-%d')
        assert e._pdt_would_block() is False
        _fill(tmp_db, today, 'A')
        _fill(tmp_db, today, 'B')
        _fill(tmp_db, today, 'C')      # 3 -> the 4th would flag
        assert e._pdt_would_block() is True

    def test_cash_never_blocks(self, orb_cfg, tmp_db):
        e = _mk(orb_cfg, tmp_db, CASH)
        e.init_account_guards()
        today = e._et_now().strftime('%Y-%m-%d')
        for s in ('A', 'B', 'C', 'D'):
            _fill(tmp_db, today, s)
        assert e._pdt_would_block() is False

    def test_count_failure_fails_closed(self, orb_cfg, tmp_db, monkeypatch):
        e = _mk(orb_cfg, tmp_db, MARGIN_LOW)
        e.init_account_guards()
        monkeypatch.setattr(e, '_count_orb_daytrades_5d',
                            lambda: e.pdt_max_daytrades_5d)
        assert e._pdt_would_block() is True

    def test_gates_check_entries(self, orb_cfg, tmp_db):
        e = _mk(orb_cfg, tmp_db, MARGIN_LOW)
        today = e._et_now().strftime('%Y-%m-%d')
        for s in ('A', 'B', 'C'):
            _fill(tmp_db, today, s)
        assert e.check_entries(symbols=['ZZZ']) == []
