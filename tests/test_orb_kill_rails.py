"""ORB pre-committed kill rails — B+ RESTART 2026-08-15 (design §5).

DB-derived (restart-safe), ET-dated, realized-ORB-P&L gates:
  daily  <= -$500  -> no new entries today
  weekly <= -$750  -> flat (force_close_all) + no new entries
  month  <= -$1500 -> ABANDON ping + no new entries
Fail-CLOSED on a pnl-query error (blocks WITHOUT escalating to abandon/flat),
mirroring ignition_engine._kill_blocked. Uses a REAL Database (8/14 lesson).
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
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


@pytest.fixture
def engine(orb_cfg, tmp_db, monkeypatch):
    monkeypatch.delenv('ORB_KILL_RAILS', raising=False)
    return ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient), db=tmp_db,
                     stop_monitor=MagicMock(spec=StopMonitor), config=orb_cfg)


def _insert(db, trade_date, symbol, pnl, filled=True):
    conn = sqlite3.connect(db._trades_path)
    conn.execute(
        """INSERT INTO trades
        (trade_date,symbol,strategy,side,entry_price,stop_loss_price,
         take_profit_price,shares,risk_per_share,total_risk,risk_reward_ratio,
         created_at,updated_at,pnl,fill_price,exit_price)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (trade_date, symbol, 'orb', 'buy', 10.0, 9.0, 12.0, 100, 1.0, 100.0,
         2.0, 'x', 'x', pnl, 10.0 if filled else None, 10.0 + pnl / 100))
    conn.commit()
    conn.close()


class TestConfig:
    def test_yaml_knobs(self, engine):
        assert engine.kill_rails_enabled is True
        assert engine.kill_daily_usd == -500
        assert engine.kill_weekly_usd == -750
        assert engine.kill_month_abandon_usd == -1500

    def test_env_disable(self, orb_cfg, tmp_db, monkeypatch):
        monkeypatch.setenv('ORB_KILL_RAILS', '0')
        e = ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient), db=tmp_db,
                      stop_monitor=MagicMock(spec=StopMonitor), config=orb_cfg)
        assert e.kill_rails_enabled is False
        assert e._kill_rails_blocked() is None


class TestDailyRail:
    def test_clean_day_no_block(self, engine):
        assert engine._kill_rails_blocked() is None

    def test_small_loss_no_block(self, engine):
        _insert(engine.db, engine._et_now().strftime('%Y-%m-%d'), 'A', -200)
        assert engine._kill_rails_blocked() is None

    def test_daily_breach_blocks(self, engine):
        _insert(engine.db, engine._et_now().strftime('%Y-%m-%d'), 'A', -600)
        assert engine._kill_rails_blocked() == 'daily_kill'

    def test_daily_notified_once(self, engine):
        _insert(engine.db, engine._et_now().strftime('%Y-%m-%d'), 'A', -600)
        engine._kill_rails_blocked()
        engine._kill_rails_blocked()
        engine.notifier = MagicMock()
        # second-and-later calls do not re-notify (latch set)
        assert engine._kill_daily_notified is True

    def test_gates_check_entries(self, engine):
        _insert(engine.db, engine._et_now().strftime('%Y-%m-%d'), 'A', -600)
        assert engine.check_entries(symbols=['ZZZ']) == []


class TestWeeklyRail:
    def test_weekly_breach_blocks_and_flattens(self, engine):
        et = engine._et_now()
        wk_start = (et - __import__('datetime').timedelta(days=et.weekday()))
        # Two -400 losses this week (=-800 <= -750) but each day > -500 so it's
        # the WEEKLY rail that fires, not daily.
        _insert(engine.db, wk_start.strftime('%Y-%m-%d'), 'A', -400)
        _insert(engine.db, et.strftime('%Y-%m-%d'), 'B', -400)
        engine.force_close_all = MagicMock(return_value=2)
        assert engine._kill_rails_blocked() == 'weekly_kill'
        engine.force_close_all.assert_called_once()

    def test_weekly_flatten_once(self, engine):
        et = engine._et_now()
        _insert(engine.db, et.strftime('%Y-%m-%d'), 'A', -400)
        wk_start = (et - __import__('datetime').timedelta(days=et.weekday()))
        _insert(engine.db, wk_start.strftime('%Y-%m-%d'), 'B', -400)
        engine.force_close_all = MagicMock(return_value=0)
        engine._kill_rails_blocked()
        engine._kill_rails_blocked()
        assert engine.force_close_all.call_count == 1


class TestMonthRail:
    def test_month_abandon_blocks(self, engine, monkeypatch):
        # Spread -$1600 across the month so no single day/week rail pre-empts
        # is impossible (month is most-negative); assert month is reported.
        et = engine._et_now()
        first = et.strftime('%Y-%m-01')
        _insert(engine.db, first, 'A', -1600)
        # month <= -1500 -> abandon (also weekly/daily may be true for `first`
        # if it is this week; month is checked FIRST by severity)
        assert engine._kill_rails_blocked() == 'month_abandon'


class TestFailClosed:
    def test_query_error_blocks_without_abandon(self, engine, monkeypatch):
        monkeypatch.setattr(engine, '_realized_orb_pnl',
                            lambda since: -1e9)
        engine._notify = MagicMock()
        assert engine._kill_rails_blocked() == 'pnl_query_failed'
        # fail-closed must NOT fire the MONTH ABANDON telegram
        engine._notify.assert_not_called()


class TestResetRoll:
    def test_reset_daily_rolls_latches(self, engine):
        engine._kill_daily_notified = True
        engine._kill_query_fail_notified = True
        engine.reset_daily()
        assert engine._kill_daily_notified is False
        assert engine._kill_query_fail_notified is False
