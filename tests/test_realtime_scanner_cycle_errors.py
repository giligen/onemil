"""Scanner cycle wrapper: consecutive-failure counter (spec 2026-11-05).

First AlpacaAPIError -> WARNING, no notifier. Second in a row -> ERROR + notifier.
Success resets. A non-Alpaca exception (code bug) -> ERROR immediately.
"""
import json
import logging
from http.client import RemoteDisconnected
from unittest.mock import MagicMock, patch

import requests

from data_sources.alpaca_client import AlpacaAPIError, AlpacaClient
from scanner.realtime_scanner import RealtimeScanner
from notifications.telegram_notifier import TelegramNotifier


def _scanner(cycle_side_effect, notifier=True):
    """RealtimeScanner without __init__; _run_intraday_cycle replaced by the given behaviour."""
    s = RealtimeScanner.__new__(RealtimeScanner)
    s._consecutive_cycle_failures = 0
    s.notifier = MagicMock(spec=TelegramNotifier) if notifier else None
    s._run_intraday_cycle = MagicMock(side_effect=cycle_side_effect)
    return s


def _levels(caplog, level):
    return [r for r in caplog.records if r.levelno == level]


class TestCycleErrorEscalation:
    def test_first_alpaca_error_is_warning_no_notifier(self, caplog):
        s = _scanner([AlpacaAPIError("drop")])
        with caplog.at_level(logging.DEBUG):
            s._run_guarded_cycle("10:30")
        assert len(_levels(caplog, logging.WARNING)) == 1
        assert not _levels(caplog, logging.ERROR)
        assert _levels(caplog, logging.WARNING)[0].exc_info
        s.notifier.send_message_sync.assert_not_called()

    def test_second_consecutive_is_error_and_notifies_once(self, caplog):
        s = _scanner([AlpacaAPIError("a"), AlpacaAPIError("b")])
        with caplog.at_level(logging.DEBUG):
            s._run_guarded_cycle("10:30")
            s._run_guarded_cycle("10:31")
        errs = _levels(caplog, logging.ERROR)
        assert len(errs) == 1 and "2 consecutive cycles failed" in errs[0].getMessage()
        s.notifier.send_message_sync.assert_called_once()
        assert "2 consecutive" in s.notifier.send_message_sync.call_args[0][0]

    def test_success_resets_counter(self, caplog):
        s = _scanner([AlpacaAPIError("a"), None, AlpacaAPIError("b")])
        with caplog.at_level(logging.DEBUG):
            for b in ("10:30", "10:31", "10:32"):
                s._run_guarded_cycle(b)
        assert not _levels(caplog, logging.ERROR)
        s.notifier.send_message_sync.assert_not_called()

    def test_runtime_error_is_error_immediately(self, caplog):
        s = _scanner([RuntimeError("bug")])
        with caplog.at_level(logging.DEBUG):
            s._run_guarded_cycle("10:30")
        assert len(_levels(caplog, logging.ERROR)) == 1
        s.notifier.send_message_sync.assert_called_once()

    def test_no_notifier_does_not_crash(self, caplog):
        s = _scanner([RuntimeError("bug")], notifier=False)
        s._run_guarded_cycle("10:30")


class TestCycleWithRealClientAndDroppedConnection:
    """Integration: real AlpacaClient + real alpaca-py data client, HTTP layer drops the first call."""

    def test_cycle_completes_on_retry_with_zero_errors(self, caplog):
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.text = json.dumps({"trades": {"AAPL": {
            "t": "2026-11-05T15:30:00Z", "x": "V", "p": 12.5, "s": 100, "c": [], "i": 1, "z": "C"}}})
        resp.json.return_value = json.loads(resp.text)
        calls = {"n": 0}

        def flaky_request(self_, *a, **kw):
            """Drop the first connection like the 11/5 15:30:20 RemoteDisconnected, then answer."""
            calls["n"] += 1
            if calls["n"] == 1:
                raise requests.exceptions.ConnectionError(
                    "('Connection aborted.', RemoteDisconnected('Remote end closed connection'))")
            return resp

        client = AlpacaClient(api_key="k", api_secret="s")
        seen = {}
        s = _scanner(None)
        s._run_intraday_cycle = lambda: seen.update(client.get_latest_trades(["AAPL"]))
        with patch.object(requests.Session, "request", flaky_request), \
             patch("data_sources.alpaca_client.time_mod.sleep"), \
             caplog.at_level(logging.DEBUG):
            s._run_guarded_cycle("10:30")
        assert calls["n"] == 2
        assert seen["AAPL"]["price"] == 12.5
        assert not _levels(caplog, logging.ERROR)
        s.notifier.send_message_sync.assert_not_called()
