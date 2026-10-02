"""
Unit tests for the 2026-10-01 20:04-20:10 UTC shutdown-hygiene incident
(docs/orb_shutdown_hygiene_20261001.md).

The service's tick loop was stalled by research load; by the time it
unwound, the process's main thread had already returned and Python's
interpreter-shutdown sequence had begun. ORB's force-close verify poll /
sweep, and the Telegram send path, treated the resulting
"cannot schedule new futures after interpreter shutdown" RuntimeError as a
retryable fault: 24 ERRORs + a Telegram rate-cap hit chasing 4 HOD
positions that could never fill after the regular session close.

Covers:
- ORB (a): _verify_flat_with_grace and the FC SWEEP stop on the
  interpreter-shutdown RuntimeError with ONE warning, never an ERROR/Telegram
  send, and set engine.shutdown_requested.
- ORB (3): the sizing.rvol_tilt boot INFO log line (enabled and disabled).
- HOD (c): force_close_all short-circuits with ONE warning once the regular
  session is closed, submits nothing, and never changes in-session mechanics.
- Telegram (b): the send path logs the same RuntimeError once at WARNING
  ("interpreter shutting down, message dropped") instead of ERROR, while an
  unrelated RuntimeError still logs ERROR (regression guard).
"""
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.stop_monitor import StopMonitor
from trading.orb_engine import ORBEngine
from trading.hod_break_engine import HodBreakEngine, Position
from tests.test_hod_break_engine import cfg as _hod_cfg
from notifications.telegram_notifier import TelegramNotifier

SHUTDOWN_RUNTIME_ERROR = RuntimeError(
    "cannot schedule new futures after interpreter shutdown"
)


# ---------------------------------------------------------------------------
# ORB helpers (mirrors tests/test_orb_engine.py's fixtures, kept local so
# this file has no cross-module fixture-name coupling)
# ---------------------------------------------------------------------------

def _orb_engine():
    yaml_path = Path(__file__).parent.parent / 'orb.yaml'
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    alpaca = MagicMock(spec=AlpacaClient)
    alpaca.get_open_positions.return_value = []
    db = MagicMock(spec=Database)
    sm = MagicMock(spec=StopMonitor)
    sm.polling_mode = False
    sm.drain_exit_events.return_value = []
    return ORBEngine(alpaca_client=alpaca, db=db, stop_monitor=sm, config=cfg), alpaca, cfg


class TestOrbShutdownStopsVerifyPoll:
    """Fix (a): _verify_flat_with_grace."""

    def test_interpreter_shutdown_stops_the_poll_with_one_warning(self, caplog):
        engine, alpaca, _ = _orb_engine()
        alpaca.get_open_positions.side_effect = SHUTDOWN_RUNTIME_ERROR

        with caplog.at_level(logging.WARNING, logger='trading.orb_engine'):
            result = engine._verify_flat_with_grace(
                max_wait_s=10, poll_interval_s=0.01, orb_owned=None,
            )

        assert alpaca.get_open_positions.call_count == 1, (
            "must stop after the FIRST failure, not poll for the full "
            "max_wait_s window"
        )
        assert engine.shutdown_requested is True
        assert result == []
        warnings = [r for r in caplog.records if r.levelname == 'WARNING'
                    and 'FC VERIFY poll' in r.message]
        assert len(warnings) == 1
        assert 'shutdown' in warnings[0].message.lower()
        assert not any(r.levelname == 'ERROR' for r in caplog.records)

    def test_ordinary_transient_error_still_retries(self, caplog):
        """Regression guard: a real Alpaca hiccup must keep its old
        retry-for-max_wait_s behaviour — only the specific interpreter-
        shutdown RuntimeError short-circuits."""
        engine, alpaca, _ = _orb_engine()
        alpaca.get_open_positions.side_effect = ConnectionError("timeout")

        with caplog.at_level(logging.WARNING, logger='trading.orb_engine'):
            engine._verify_flat_with_grace(
                max_wait_s=0.05, poll_interval_s=0.01, orb_owned=None,
            )

        assert alpaca.get_open_positions.call_count > 1
        assert engine.shutdown_requested is False


class TestOrbShutdownStopsForceCloseSweep:
    """Fix (a): the FC SWEEP phase inside force_close_all()."""

    def test_interpreter_shutdown_skips_notify_error_and_the_retry_loop(
        self, caplog,
    ):
        engine, alpaca, _ = _orb_engine()
        engine.open_positions = {}
        engine._orb_owned_symbols = lambda *a, **k: set()
        alpaca.get_open_positions.side_effect = SHUTDOWN_RUNTIME_ERROR
        engine._notify_error = MagicMock()
        engine._notify = MagicMock()

        with caplog.at_level(logging.WARNING, logger='trading.orb_engine'):
            closed = engine.force_close_all()

        assert closed == 0
        engine._notify_error.assert_not_called()
        assert engine.shutdown_requested is True
        # One get_open_positions call for the SWEEP phase; the retry loop's
        # own top-of-loop shutdown check must stop it from calling
        # _verify_flat_with_grace at all once shutdown is known.
        assert alpaca.get_open_positions.call_count == 1
        sweep_warnings = [
            r for r in caplog.records if r.levelname == 'WARNING'
            and 'FC SWEEP' in r.message and 'shutdown' in r.message.lower()
        ]
        assert len(sweep_warnings) == 1
        assert not any(r.levelname == 'ERROR' for r in caplog.records)


class TestOrbRvolTiltBootLog:
    """Task 3: ORB RVOL tilt boot INFO line, logged enabled or disabled."""

    @pytest.mark.parametrize('enabled', [True, False])
    def test_boot_line_logged_with_config(self, caplog, enabled):
        yaml_path = Path(__file__).parent.parent / 'orb.yaml'
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        cfg['strategy']['enabled'] = True
        cfg['sizing']['rvol_tilt'] = {
            'enabled': enabled, 'edges': [1.5, 3.0], 'mults': [0.5, 1.0, 1.5],
            'applies_to': ['production'],
        }
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.get_open_positions.return_value = []
        db = MagicMock(spec=Database)
        sm = MagicMock(spec=StopMonitor)
        sm.polling_mode = False
        sm.drain_exit_events.return_value = []

        with caplog.at_level(logging.INFO, logger='trading.orb_engine'):
            ORBEngine(alpaca_client=alpaca, db=db, stop_monitor=sm, config=cfg)

        lines = [r.message for r in caplog.records if 'ORB RVOL tilt:' in r.message]
        assert len(lines) == 1
        assert f'enabled={enabled}' in lines[0]
        assert 'edges=1.5/3.0' in lines[0]
        assert 'mults=0.5/1.0/1.5' in lines[0]
        assert "applies_to=['production']" in lines[0]


# ---------------------------------------------------------------------------
# HOD helper
# ---------------------------------------------------------------------------

def _hod_engine():
    alpaca = MagicMock(spec=AlpacaClient)
    db = MagicMock(spec=Database)
    sm = MagicMock(spec=StopMonitor)
    sm.polling_mode = False
    sm.drain_exit_events.return_value = []
    e = HodBreakEngine(alpaca, db, sm, notifier=None, cfg=_hod_cfg())
    e._roll_session()
    return e, alpaca


class TestHodForceCloseAfterRegularClose:
    """Fix (c): force_close_all short-circuits once minute_of_day >=
    close_minute (16:00 ET) instead of re-running the cancel/resubmit/leg
    -poll machinery against a closed session every tick."""

    def test_logs_one_warning_and_submits_nothing(self, caplog):
        engine, alpaca = _hod_engine()
        engine._minute_of_day = lambda: engine.close_minute + 5
        pos = MagicMock(spec=Position)
        pos.status = 'open'
        engine.positions = {'ABCD': pos}

        caplog.clear()  # drop construction/_roll_session noise (unrelated ERRORs on a minimal mock universe)
        with caplog.at_level(logging.WARNING, logger='trading.hod_break_engine'):
            n = engine.force_close_all()

        assert n == 0
        alpaca.submit_limit_sell_order.assert_not_called()
        alpaca.submit_market_sell_order.assert_not_called()
        alpaca.submit_moc_sell_order.assert_not_called()
        alpaca.cancel_order.assert_not_called()
        skipped = [r for r in caplog.records if r.levelname == 'WARNING'
                   and 'FORCE CLOSE skipped' in r.message]
        assert len(skipped) == 1
        assert 'ABCD' in skipped[0].message
        assert not any(r.levelname == 'ERROR' for r in caplog.records)

    def test_second_call_does_not_repeat_the_warning(self, caplog):
        engine, _alpaca = _hod_engine()
        engine._minute_of_day = lambda: engine.close_minute + 5
        pos = MagicMock(spec=Position)
        pos.status = 'open'
        engine.positions = {'ABCD': pos}

        with caplog.at_level(logging.WARNING, logger='trading.hod_break_engine'):
            engine.force_close_all()
            caplog.clear()
            engine.force_close_all()

        assert not any('FORCE CLOSE skipped' in r.message for r in caplog.records)

    # In-session force_close_all mechanics (submit/cancel/retry) are
    # exercised end-to-end by tests/test_hod_break_engine.py::TestForceClose,
    # now pinned to minute_of_day=955 so this fix's new >= close_minute
    # branch can never shadow them (see that file's patch.object calls).


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

class TestTelegramInterpreterShutdown:
    """Fix (b): the send path."""

    @pytest.mark.asyncio
    async def test_interpreter_shutdown_logs_warning_not_error(self, caplog):
        notifier = TelegramNotifier(bot_token="t", chat_id="c", enabled=True)

        with caplog.at_level(logging.WARNING, logger='notifications.telegram_notifier'):
            with patch('notifications.telegram_notifier.aiohttp.ClientSession',
                       side_effect=SHUTDOWN_RUNTIME_ERROR):
                result = await notifier.send_message("hello")

        assert result is False
        warnings = [r for r in caplog.records if r.levelname == 'WARNING'
                    and 'interpreter shutting down, message dropped' in r.message]
        assert len(warnings) == 1
        assert not any(r.levelname == 'ERROR' for r in caplog.records)

    @pytest.mark.asyncio
    async def test_unrelated_runtime_error_still_logs_error(self, caplog):
        """Regression guard: only the specific shutdown RuntimeError is
        downgraded — any other RuntimeError keeps the old ERROR behaviour."""
        notifier = TelegramNotifier(bot_token="t", chat_id="c", enabled=True)

        with caplog.at_level(logging.WARNING, logger='notifications.telegram_notifier'):
            with patch('notifications.telegram_notifier.aiohttp.ClientSession',
                       side_effect=RuntimeError("boom")):
                result = await notifier.send_message("hello")

        assert result is False
        assert any(r.levelname == 'ERROR' and 'Unexpected error' in r.message
                    for r in caplog.records)


class TestWrappedShutdownError:
    """10/2 close: AlpacaClient wraps the RuntimeError in AlpacaAPIError — the detector must still match."""

    def test_detector_matches_the_wrapped_alpaca_error(self):
        from trading.orb_engine import ORBEngine
        from data_sources.alpaca_client import AlpacaAPIError
        wrapped = AlpacaAPIError("Failed to get open positions: cannot schedule new futures after interpreter shutdown")
        assert ORBEngine._is_interpreter_shutdown_error(wrapped) is True
        assert ORBEngine._is_interpreter_shutdown_error(AlpacaAPIError("Failed to get open positions: 500")) is False
