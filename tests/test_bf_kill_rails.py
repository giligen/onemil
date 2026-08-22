"""BF pre-committed kill rails — Discipline Program Phase 1 (2026-08-22).

Mirrors tests/test_orb_kill_rails.py (the template). DB-derived
(restart-safe), ET-dated, realized-bull_flag-P&L gates:
  daily  <= -$800  -> no new entries rest of day
  weekly <= -$1200 -> flatten BF (_force_close_all) + no entries rest of week
  month  <= -$2500 -> PAUSE latch + data/bf_month_pause.flag (honored at
                      boot, cleared only by owner removing the file) +
                      [BF] ABANDON-GATE telegram with the month trade list
Fail-CLOSED on a pnl-query error (blocks WITHOUT escalating to
pause/flatten). Uses a REAL Database (8/14 lesson).
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from notifications.telegram_notifier import TelegramNotifier
from persistence.database import Database
from trading.order_executor import OrderExecutor
from trading.pattern_detector import BullFlagDetector
from trading.position_manager import PositionManager
from trading.trade_planner import TradePlanner
from trading.trading_engine import TradingEngine

ROOT = Path(__file__).parent.parent


@pytest.fixture
def tmp_db():
    d = tempfile.mkdtemp()
    return Database(trades_path=os.path.join(d, 'trades.db'),
                    cache_path=os.path.join(d, 'cache.db'))


def _build_engine(tmp_db, notifier=None):
    return TradingEngine(
        alpaca_client=MagicMock(spec=AlpacaClient),
        db=tmp_db,
        detector=MagicMock(spec=BullFlagDetector),
        planner=MagicMock(spec=TradePlanner),
        executor=MagicMock(spec=OrderExecutor),
        position_manager=MagicMock(spec=PositionManager),
        enabled=True,
        notifier=notifier,
    )


@pytest.fixture
def engine(tmp_db, tmp_path, monkeypatch):
    monkeypatch.delenv('BF_KILL_RAILS', raising=False)
    monkeypatch.setenv('BF_MONTH_PAUSE_FLAG',
                       str(tmp_path / 'bf_month_pause.flag'))
    eng = _build_engine(
        tmp_db, notifier=MagicMock(spec=TelegramNotifier))
    return eng


def _insert(db, trade_date, symbol, pnl, strategy='bull_flag'):
    """Insert one CLOSED trade row (pnl set) into the real trades DB."""
    conn = sqlite3.connect(db._trades_path)
    conn.execute(
        """INSERT INTO trades
        (trade_date,symbol,strategy,side,entry_price,stop_loss_price,
         take_profit_price,shares,risk_per_share,total_risk,risk_reward_ratio,
         created_at,updated_at,pnl,fill_price,exit_price)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (trade_date, symbol, strategy, 'buy', 10.0, 9.0, 12.0, 100, 1.0,
         100.0, 2.0, 'x', 'x', pnl, 10.0, 10.0 + pnl / 100))
    conn.commit()
    conn.close()


def _today(engine):
    return engine._et_now().strftime('%Y-%m-%d')


def _fix_et(monkeypatch, engine, iso_dt):
    """Pin engine._et_now to a fixed ET wall-clock datetime."""
    monkeypatch.setattr(engine, '_et_now',
                        lambda: datetime.fromisoformat(iso_dt))


def _quiet_reset(monkeypatch, engine):
    """Neutralize reset_daily's data-hungry sync steps for unit tests."""
    monkeypatch.setattr(engine, '_refresh_spy_data', lambda: None)
    monkeypatch.setattr(engine, '_sync_startup_state', lambda: None)


# ===========================================================================
# Config
# ===========================================================================

class TestConfig:
    def test_yaml_knobs(self, engine):
        """Live config.yaml ships the Phase-1 thresholds, enabled ON."""
        assert engine.kill_rails_enabled is True
        assert engine.kill_daily_usd == -800
        assert engine.kill_weekly_usd == -1200
        assert engine.kill_month_pause_usd == -2500

    def test_template_matches_live(self):
        """config.yaml.template carries the same kill_rails block."""
        tmpl = yaml.safe_load(open(ROOT / 'config.yaml.template'))
        kr = tmpl['trading']['bull_flag']['kill_rails']
        assert kr == {'enabled': True, 'daily_usd': -800,
                      'weekly_usd': -1200, 'month_pause_usd': -2500}

    def test_env_disable(self, tmp_db, tmp_path, monkeypatch):
        monkeypatch.setenv('BF_KILL_RAILS', '0')
        monkeypatch.setenv('BF_MONTH_PAUSE_FLAG',
                           str(tmp_path / 'bf_month_pause.flag'))
        e = _build_engine(tmp_db)
        assert e.kill_rails_enabled is False
        _insert(e.db, _today(e), 'A', -9000)
        assert e._kill_rails_blocked() is None


# ===========================================================================
# Daily rail
# ===========================================================================

class TestDailyRail:
    def test_clean_day_no_block(self, engine):
        assert engine._kill_rails_blocked() is None

    def test_small_loss_no_block(self, engine):
        _insert(engine.db, _today(engine), 'A', -200)
        assert engine._kill_rails_blocked() is None

    def test_daily_breach_blocks(self, engine):
        _insert(engine.db, _today(engine), 'A', -900)
        assert engine._kill_rails_blocked() == 'daily_kill'

    def test_other_strategy_pnl_ignored(self, engine):
        """ORB/MACD-wave losses never trip the BF rails."""
        _insert(engine.db, _today(engine), 'A', -9000, strategy='orb')
        _insert(engine.db, _today(engine), 'B', -9000, strategy='macd_wave')
        assert engine._kill_rails_blocked() is None

    def test_open_rows_ignored(self, engine):
        """pnl IS NULL (open trade) does not count toward realized sums."""
        conn = sqlite3.connect(engine.db._trades_path)
        conn.execute(
            """INSERT INTO trades (trade_date,symbol,strategy,side,
            entry_price,stop_loss_price,take_profit_price,shares,
            risk_per_share,total_risk,risk_reward_ratio,created_at,
            updated_at,pnl) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)""",
            (_today(engine), 'OPEN', 'bull_flag', 'buy', 10.0, 9.0, 12.0,
             100, 1.0, 100.0, 2.0, 'x', 'x'))
        conn.commit()
        conn.close()
        assert engine._kill_rails_blocked() is None

    def test_daily_notified_once(self, engine):
        _insert(engine.db, _today(engine), 'A', -900)
        engine._kill_rails_blocked()
        engine._kill_rails_blocked()
        assert engine._kill_daily_notified is True
        sends = engine.notifier.send_message_sync.call_args_list
        assert len(sends) == 1
        assert '[BF] DAILY KILL' in sends[0].args[0]

    def test_gates_run_pattern_check(self, engine, monkeypatch):
        """Daily breach: run_pattern_check never reaches _check_symbol."""
        _insert(engine.db, _today(engine), 'A', -900)
        monkeypatch.setattr(engine, '_sync_closed_positions', lambda: None)
        monkeypatch.setattr(engine, '_manage_pending_orders', lambda: None)
        check = MagicMock()
        monkeypatch.setattr(engine, '_check_symbol', check)
        engine._qualified_symbols = {'AAA'}
        assert engine.run_pattern_check() is None
        check.assert_not_called()

    def test_gates_rt_bar_path(self, engine, monkeypatch):
        """Daily breach: RT bar events are flushed, not traded."""
        _insert(engine.db, _today(engine), 'A', -900)
        check = MagicMock()
        monkeypatch.setattr(engine, '_check_symbol', check)
        engine._qualified_symbols = {'AAA'}
        engine._bar_event_queue.put_nowait(('AAA', MagicMock()))
        assert engine._drain_bar_events() is None
        check.assert_not_called()
        assert engine._bar_event_queue.empty()


# ===========================================================================
# Weekly rail
# ===========================================================================

class TestWeeklyRail:
    def _breach_week(self, engine):
        et = engine._et_now()
        wk_start = et - timedelta(days=et.weekday())
        # Two -700 losses this ISO week (=-1400 <= -1200) but each day
        # > -800 so it's the WEEKLY rail that fires, not daily (unless
        # today IS Monday, in which case weekly still pre-empts daily).
        _insert(engine.db, wk_start.strftime('%Y-%m-%d'), 'A', -700)
        _insert(engine.db, et.strftime('%Y-%m-%d'), 'B', -700)

    def test_weekly_breach_blocks_and_flattens(self, engine):
        self._breach_week(engine)
        engine._force_close_all = MagicMock()
        assert engine._kill_rails_blocked() == 'weekly_kill'
        engine._force_close_all.assert_called_once()
        msg = engine.notifier.send_message_sync.call_args.args[0]
        assert '[BF] WEEKLY KILL' in msg

    def test_weekly_flatten_once(self, engine):
        self._breach_week(engine)
        engine._force_close_all = MagicMock()
        engine._kill_rails_blocked()
        engine._kill_rails_blocked()
        assert engine._force_close_all.call_count == 1
        assert engine.notifier.send_message_sync.call_count == 1

    def test_flatten_failure_still_blocks(self, engine):
        """_force_close_all raising must not unblock entries (logged)."""
        self._breach_week(engine)
        engine._force_close_all = MagicMock(side_effect=RuntimeError('boom'))
        assert engine._kill_rails_blocked() == 'weekly_kill'
        assert engine._kill_rails_blocked() == 'weekly_kill'

    def test_prev_week_losses_do_not_count(self, engine, monkeypatch):
        _fix_et(monkeypatch, engine, '2026-08-19T10:00:00')  # Wed
        _insert(engine.db, '2026-08-14', 'A', -900)          # prev week Fri
        assert engine._kill_rails_blocked() is None
        _insert(engine.db, '2026-08-17', 'B', -1250)         # this week Mon
        engine._force_close_all = MagicMock()
        assert engine._kill_rails_blocked() == 'weekly_kill'


# ===========================================================================
# Month rail — the PAUSE latch
# ===========================================================================

class TestMonthPause:
    def test_month_breach_pauses(self, engine):
        first = engine._et_now().strftime('%Y-%m-01')
        _insert(engine.db, first, 'A', -2600)
        assert engine._kill_rails_blocked() == 'month_pause'
        assert engine._bf_month_paused is True
        assert engine.month_pause_flag_path.exists()

    def test_abandon_gate_telegram_has_trade_list(self, engine):
        first = engine._et_now().strftime('%Y-%m-01')
        _insert(engine.db, first, 'AAA', -1400)
        _insert(engine.db, first, 'BBB', -1200)
        engine._kill_rails_blocked()
        msg = engine.notifier.send_message_sync.call_args.args[0]
        assert '[BF] ABANDON-GATE' in msg
        assert 'AAA' in msg and 'BBB' in msg
        assert 'Month trades (2)' in msg
        assert '<' not in msg   # HTML-safe

    def test_pause_notifies_once(self, engine):
        first = engine._et_now().strftime('%Y-%m-01')
        _insert(engine.db, first, 'A', -2600)
        engine._kill_rails_blocked()
        assert engine._kill_rails_blocked() == 'month_pause'
        assert engine.notifier.send_message_sync.call_count == 1

    def test_flag_file_honored_at_boot(self, engine, tmp_db, tmp_path,
                                       monkeypatch):
        """A pre-existing flag pauses a fresh engine with an EMPTY DB."""
        flag = tmp_path / 'bf_month_pause.flag'
        flag.write_text('paused_at=drill\n')
        monkeypatch.setenv('BF_MONTH_PAUSE_FLAG', str(flag))
        e = _build_engine(tmp_db)
        assert e._bf_month_paused is True
        assert e._kill_rails_blocked() == 'month_pause'

    def test_flag_absent_is_normal(self, engine):
        assert engine._bf_month_paused is False
        assert engine._kill_rails_blocked() is None

    def test_manual_clear_via_reset_daily(self, engine, monkeypatch):
        first = engine._et_now().strftime('%Y-%m-01')
        _insert(engine.db, first, 'A', -2600)
        engine._kill_rails_blocked()
        assert engine.month_pause_flag_path.exists()
        # Owner clears the month's losing trades context + deletes the flag.
        conn = sqlite3.connect(engine.db._trades_path)
        conn.execute("DELETE FROM trades")
        conn.commit()
        conn.close()
        engine.month_pause_flag_path.unlink()
        _quiet_reset(monkeypatch, engine)
        engine.reset_daily()
        assert engine._bf_month_paused is False
        assert engine._kill_rails_blocked() is None

    def test_flag_survives_reset_daily(self, engine, monkeypatch):
        """reset_daily does NOT clear the pause while the flag exists —
        even if realized P&L rolled (new month)."""
        first = engine._et_now().strftime('%Y-%m-01')
        _insert(engine.db, first, 'A', -2600)
        engine._kill_rails_blocked()
        _quiet_reset(monkeypatch, engine)
        engine.reset_daily()
        assert engine._bf_month_paused is True
        assert engine._kill_rails_blocked() == 'month_pause'


# ===========================================================================
# Fail-closed
# ===========================================================================

class TestFailClosed:
    def test_query_error_blocks_without_escalation(self, engine, monkeypatch):
        monkeypatch.setattr(engine, '_realized_bf_pnl', lambda since: -1e9)
        engine._force_close_all = MagicMock()
        assert engine._kill_rails_blocked() == 'pnl_query_failed'
        # fail-closed must NOT pause, flatten, or telegram
        engine.notifier.send_message_sync.assert_not_called()
        engine._force_close_all.assert_not_called()
        assert engine._bf_month_paused is False
        assert not engine.month_pause_flag_path.exists()

    def test_real_db_error_returns_sentinel(self, engine):
        """A corrupt trades DB path yields the -1e9 sentinel."""
        engine.db._trades_path = Path('/nonexistent/dir/trades.db')
        assert engine._realized_bf_pnl('2026-01-01') == -1e9

    def test_mock_db_is_inert(self, tmp_path, monkeypatch, tmp_db):
        """A db without _trades_path stays inert (fail-open, logged)."""
        monkeypatch.setenv('BF_MONTH_PAUSE_FLAG',
                           str(tmp_path / 'f.flag'))
        monkeypatch.delenv('BF_KILL_RAILS', raising=False)
        e = _build_engine(tmp_db)
        e.db = MagicMock(spec=[])   # no _trades_path attribute
        assert e._realized_bf_pnl('2026-01-01') == 0.0
        assert e._kill_rails_blocked() is None


class TestErrorBranches:
    """Fallback paths — every one must log (CLAUDE.md fallback rule) and
    never unlatch/escalate a rail."""

    def test_month_trades_no_trades_path(self, engine):
        engine.db = MagicMock(spec=[])
        assert engine._bf_month_trades('2026-08-01') == []

    def test_month_trades_query_error(self, engine):
        engine.db._trades_path = Path('/nonexistent/dir/trades.db')
        assert engine._bf_month_trades('2026-08-01') == []

    def test_notify_without_notifier_is_noop(self, engine):
        engine.notifier = None
        engine._bf_notify('x')   # must not raise

    def test_notify_send_failure_swallowed(self, engine):
        engine.notifier.send_message_sync.side_effect = RuntimeError('tg down')
        engine._bf_notify('x')   # must not raise

    def test_flag_write_failure_still_latches(self, engine):
        """Unwritable flag path: pause latch still set + telegram sent."""
        engine.month_pause_flag_path = Path('/proc/nonexistent/no.flag')
        first = engine._et_now().strftime('%Y-%m-01')
        _insert(engine.db, first, 'A', -2600)
        assert engine._kill_rails_blocked() == 'month_pause'
        assert engine._bf_month_paused is True
        assert '[BF] ABANDON-GATE' in \
            engine.notifier.send_message_sync.call_args.args[0]

    def test_abandon_trade_list_truncated_at_40(self, engine):
        first = engine._et_now().strftime('%Y-%m-01')
        for i in range(45):
            _insert(engine.db, first, f'S{i:02d}', -60)
        engine._kill_rails_blocked()
        msg = engine.notifier.send_message_sync.call_args.args[0]
        assert 'Month trades (45)' in msg
        assert '...and 5 more' in msg


# ===========================================================================
# ET keying + latch rolls
# ===========================================================================

class TestResetRoll:
    def test_reset_daily_rolls_daily_latches(self, engine, monkeypatch):
        engine._kill_daily_notified = True
        engine._kill_query_fail_notified = True
        engine._kill_pause_logged = True
        _quiet_reset(monkeypatch, engine)
        engine.reset_daily()
        assert engine._kill_daily_notified is False
        assert engine._kill_query_fail_notified is False
        assert engine._kill_pause_logged is False

    def test_week_roll_resets_weekly_latches(self, engine, monkeypatch):
        engine._kill_week_key = '1999-01-04'   # stale key -> roll
        engine._kill_weekly_notified = True
        engine._kill_weekly_flattened = True
        _quiet_reset(monkeypatch, engine)
        engine.reset_daily()
        assert engine._kill_weekly_notified is False
        assert engine._kill_weekly_flattened is False

    def test_same_week_keeps_weekly_latches(self, engine, monkeypatch):
        et = engine._et_now()
        engine._kill_week_key = (
            et - timedelta(days=et.weekday())).strftime('%Y-%m-%d')
        engine._kill_weekly_notified = True
        engine._kill_weekly_flattened = True
        _quiet_reset(monkeypatch, engine)
        engine.reset_daily()
        assert engine._kill_weekly_notified is True
        assert engine._kill_weekly_flattened is True

    def test_month_roll_excludes_prev_month(self, engine, monkeypatch):
        _fix_et(monkeypatch, engine, '2026-09-02T10:00:00')  # Wed, new month
        _insert(engine.db, '2026-08-30', 'A', -2600)  # prev month + prev week
        assert engine._kill_rails_blocked() is None
        _insert(engine.db, '2026-09-01', 'B', -2600)  # this month
        assert engine._kill_rails_blocked() == 'month_pause'

    def test_daily_keys_on_et_date(self, engine, monkeypatch):
        _fix_et(monkeypatch, engine, '2026-08-19T10:00:00')
        _insert(engine.db, '2026-08-18', 'A', -900)   # yesterday
        assert engine._kill_rails_blocked() is None   # week -900 > -1200
        _insert(engine.db, '2026-08-19', 'B', -900)   # ET-today
        engine._force_close_all = MagicMock()
        # -1800 week <= -1200 pre-empts daily by severity
        assert engine._kill_rails_blocked() == 'weekly_kill'


# ===========================================================================
# Rails never place orders
# ===========================================================================

class TestRailsNeverPlaceOrders:
    def test_no_executor_or_submit_calls_across_all_rails(self, engine,
                                                          monkeypatch):
        et = engine._et_now()
        wk_start = et - timedelta(days=et.weekday())
        _insert(engine.db, et.strftime('%Y-%m-01'), 'A', -2600)
        _insert(engine.db, wk_start.strftime('%Y-%m-%d'), 'B', -700)
        _insert(engine.db, et.strftime('%Y-%m-%d'), 'C', -900)
        engine._force_close_all = MagicMock()
        assert engine._kill_rails_blocked() == 'month_pause'
        assert engine.executor.mock_calls == []
        submits = [c for c in engine.alpaca.mock_calls if 'submit' in str(c)]
        assert submits == []


# ===========================================================================
# Integration — mock month: daily gate -> weekly flatten -> monthly pause
# ===========================================================================

class TestMockMonthIntegration:
    def test_full_escalation_sequence(self, engine, monkeypatch):
        _quiet_reset(monkeypatch, engine)
        engine._force_close_all = MagicMock()
        tg = engine.notifier.send_message_sync

        # Day 1 (Mon 9/7): -900 -> DAILY gate
        _fix_et(monkeypatch, engine, '2026-09-07T11:00:00')
        _insert(engine.db, '2026-09-07', 'AAA', -900)
        assert engine._kill_rails_blocked() == 'daily_kill'
        assert '[BF] DAILY KILL' in tg.call_args.args[0]

        # Day 2 (Tue 9/8): -400 more -> week -1300 -> WEEKLY flatten
        _fix_et(monkeypatch, engine, '2026-09-08T11:00:00')
        engine.reset_daily()
        _insert(engine.db, '2026-09-08', 'BBB', -400)
        assert engine._kill_rails_blocked() == 'weekly_kill'
        engine._force_close_all.assert_called_once()
        assert '[BF] WEEKLY KILL' in tg.call_args.args[0]

        # Day 3 (Mon 9/14, new ISO week): -1300 -> month -2600 -> PAUSE
        _fix_et(monkeypatch, engine, '2026-09-14T11:00:00')
        engine.reset_daily()
        _insert(engine.db, '2026-09-14', 'CCC', -1300)
        assert engine._kill_rails_blocked() == 'month_pause'
        assert engine.month_pause_flag_path.exists()
        last = tg.call_args.args[0]
        assert '[BF] ABANDON-GATE' in last
        assert 'AAA' in last and 'BBB' in last and 'CCC' in last

        # Telegram sequence: exactly daily -> weekly -> abandon, in order
        msgs = [c.args[0] for c in tg.call_args_list]
        assert len(msgs) == 3
        assert 'DAILY KILL' in msgs[0]
        assert 'WEEKLY KILL' in msgs[1]
        assert 'ABANDON-GATE' in msgs[2]

        # Day 4: pause persists via flag, no re-notify, entries stay gated
        _fix_et(monkeypatch, engine, '2026-09-15T11:00:00')
        engine.reset_daily()
        assert engine._kill_rails_blocked() == 'month_pause'
        assert tg.call_count == 3


# ===========================================================================
# report_common.bf_rails_status — EoD-dive visibility
# ===========================================================================

class TestBfRailsStatus:
    @pytest.fixture
    def rc_env(self, tmp_path, monkeypatch):
        import scripts.report_common as rc
        (tmp_path / 'data').mkdir()
        conn = sqlite3.connect(tmp_path / 'data' / 'trades.db')
        conn.execute("CREATE TABLE trades (trade_date TEXT, symbol TEXT, "
                     "strategy TEXT, pnl REAL)")
        conn.commit()
        conn.close()
        monkeypatch.setattr(rc, 'ROOT', tmp_path)
        monkeypatch.setattr(rc, 'BF_MONTH_PAUSE_FLAG',
                            tmp_path / 'data' / 'bf_month_pause.flag')
        return rc, tmp_path

    def _add(self, tmp_path, trade_date, pnl, strategy='bull_flag'):
        conn = sqlite3.connect(tmp_path / 'data' / 'trades.db')
        conn.execute("INSERT INTO trades VALUES (?,?,?,?)",
                     (trade_date, 'X', strategy, pnl))
        conn.commit()
        conn.close()

    def test_sums_and_windows(self, rc_env):
        rc, tmp_path = rc_env
        # day = Wed 2026-08-19; week Mon 8/17; month 8/01
        self._add(tmp_path, '2026-08-19', -100)   # day+week+month
        self._add(tmp_path, '2026-08-17', -200)   # week+month
        self._add(tmp_path, '2026-08-05', -300)   # month only
        self._add(tmp_path, '2026-07-30', -999)   # excluded
        self._add(tmp_path, '2026-08-19', -5000, strategy='orb')  # excluded
        st = rc.bf_rails_status('2026-08-19')
        assert st['daily'] == -100
        assert st['weekly'] == -300
        assert st['monthly'] == -600
        assert not (st['daily_breached'] or st['weekly_breached']
                    or st['month_breached'] or st['month_paused'])
        line = rc.bf_rails_line(st)
        assert line.startswith('BF RAILS')
        assert 'BREACH' not in line and 'pause=no' in line

    def test_breach_flags_and_line(self, rc_env):
        rc, tmp_path = rc_env
        self._add(tmp_path, '2026-08-19', -900)
        st = rc.bf_rails_status('2026-08-19')
        assert st['daily_breached'] is True
        assert st['weekly_breached'] is False
        assert 'BREACH' in rc.bf_rails_line(st)

    def test_flag_file_reports_paused(self, rc_env):
        rc, tmp_path = rc_env
        (tmp_path / 'data' / 'bf_month_pause.flag').write_text('x')
        st = rc.bf_rails_status('2026-08-19')
        assert st['month_pause_flag'] is True
        assert st['month_paused'] is True
        assert 'pause=YES' in rc.bf_rails_line(st)

    def test_db_error_fails_closed(self, tmp_path, monkeypatch):
        import scripts.report_common as rc
        monkeypatch.setattr(rc, 'ROOT', tmp_path)   # no data/trades.db
        monkeypatch.setattr(rc, 'BF_MONTH_PAUSE_FLAG',
                            tmp_path / 'nope.flag')
        st = rc.bf_rails_status('2026-08-19')
        assert st['query_failed'] is True
        assert st['daily_breached'] and st['weekly_breached'] \
            and st['month_breached']
        assert 'QUERY FAILED' in rc.bf_rails_line(st)

    def test_thresholds_from_live_config(self):
        import scripts.report_common as rc
        kr = rc._bf_rails_cfg()
        assert kr['daily_usd'] == -800
        assert kr['weekly_usd'] == -1200
        assert kr['month_pause_usd'] == -2500
