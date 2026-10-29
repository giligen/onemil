"""Shared cron crash guard (10/26 silent MOM crash): ERROR -> Telegram, wait for the send, re-raise unchanged."""
import logging
from unittest.mock import MagicMock

import pytest

import monitoring.telegram_error_handler as teh
from monitoring import cron_guard


class FakeHandler(logging.Handler):
    """Stand-in TelegramErrorHandler recording wait_for_sends calls (no network)."""
    calls: list = []

    def emit(self, record):
        pass

    def wait_for_sends(self, timeout=10.0):
        FakeHandler.calls.append(timeout)


def _boom():
    raise RuntimeError('assets down')


def test_run_guarded_logs_once_attaches_and_reraises(monkeypatch, caplog):
    """Uncaught exception -> one ERROR '<name> crashed (Type): msg', handler attached, exception re-raised."""
    attached = []
    monkeypatch.setattr(cron_guard, 'attach_telegram_error_handler', lambda: attached.append(1) or True)
    with caplog.at_level('ERROR'):
        with pytest.raises(RuntimeError, match='assets down'):
            cron_guard.run_guarded(_boom, 'myjob')
    errs = [r for r in caplog.records if r.levelname == 'ERROR']
    assert len(errs) == 1 and 'myjob crashed (RuntimeError): assets down' in errs[0].getMessage()
    assert attached == [1]


def test_run_guarded_passes_systemexit_keyboardinterrupt_and_return(monkeypatch, caplog):
    """SystemExit/KeyboardInterrupt pass untouched (no ERROR, no attach); a return value is passed through."""
    monkeypatch.setattr(cron_guard, 'attach_telegram_error_handler', lambda: pytest.fail('must not attach'))
    for exc in (SystemExit(2), KeyboardInterrupt()):
        def raiser(e=exc):
            raise e
        with caplog.at_level('ERROR'):
            with pytest.raises(type(exc)):
                cron_guard.run_guarded(raiser, 'myjob')
    assert cron_guard.run_guarded(lambda: 7, 'myjob') == 7
    assert not [r for r in caplog.records if r.levelname == 'ERROR']


def test_run_guarded_waits_for_telegram_sends_before_reraise(monkeypatch):
    """After logging the crash, wait_for_sends(10.0) is called on the root TelegramErrorHandler."""
    FakeHandler.calls = []
    fake = FakeHandler()
    monkeypatch.setattr(teh, 'TelegramErrorHandler', FakeHandler)
    monkeypatch.setattr(cron_guard, 'attach_telegram_error_handler', lambda: True)
    root = logging.getLogger()
    root.addHandler(fake)
    try:
        with pytest.raises(RuntimeError):
            cron_guard.run_guarded(_boom, 'myjob')
    finally:
        root.removeHandler(fake)
    assert FakeHandler.calls == [10.0]


def test_wait_failure_never_masks_the_crash(monkeypatch):
    """A raising wait_for_sends is swallowed (WARNING); the original exception still propagates."""
    class Bad(FakeHandler):
        def wait_for_sends(self, timeout=10.0):
            raise OSError('join failed')
    bad = Bad()
    monkeypatch.setattr(teh, 'TelegramErrorHandler', Bad)
    monkeypatch.setattr(cron_guard, 'attach_telegram_error_handler', lambda: True)
    root = logging.getLogger()
    root.addHandler(bad)
    try:
        with pytest.raises(RuntimeError, match='assets down'):
            cron_guard.run_guarded(_boom, 'myjob')
    finally:
        root.removeHandler(bad)


def test_attach_is_idempotent_and_reports_state(monkeypatch):
    """Configured -> True and exactly one root TelegramErrorHandler after repeated calls; unconfigured -> False."""
    root = logging.getLogger()
    before = list(root.handlers)
    cfg = MagicMock()
    cfg.telegram_bot_token, cfg.telegram_chat_id = 'tok', '123'
    import config
    try:
        monkeypatch.setattr(config, 'Config', lambda: cfg)
        assert cron_guard.attach_telegram_error_handler() is True
        assert cron_guard.attach_telegram_error_handler() is True
        assert sum(isinstance(h, teh.TelegramErrorHandler) for h in root.handlers) == 1
        root.handlers[:] = before
        cfg.telegram_bot_token = ''
        assert cron_guard.attach_telegram_error_handler() is False
        assert not any(isinstance(h, teh.TelegramErrorHandler) for h in root.handlers)
    finally:
        root.handlers[:] = before


def test_attach_never_raises_when_config_fails(monkeypatch):
    """A Config() failure is a WARNING and False, not an exception."""
    import config
    monkeypatch.setattr(config, 'Config', lambda: (_ for _ in ()).throw(RuntimeError('no env')))
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        assert cron_guard.attach_telegram_error_handler() is False
    finally:
        root.handlers[:] = before
