"""
Unit tests for monitoring/telegram_error_handler.py - TelegramErrorHandler.

Covers:
- Initialization
- Message formatting
- Deduplication logic
- Async/sync send dispatch
"""

import logging
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from monitoring.telegram_error_handler import TelegramErrorHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def handler():
    """Create a TelegramErrorHandler."""
    return TelegramErrorHandler(
        bot_token="test-token",
        chat_id="12345",
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for TelegramErrorHandler initialization."""

    def test_level_is_error(self, handler):
        """Handler is set to ERROR level."""
        assert handler.level == logging.ERROR

    def test_api_url_formed(self, handler):
        """API URL includes bot token."""
        assert "test-token" in handler.api_url

    def test_dedup_state_initialized(self, handler):
        """Deduplication state is clean."""
        assert handler.last_error is None
        assert handler.last_error_count == 0


# ---------------------------------------------------------------------------
# Message Formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    """Tests for error message formatting."""

    def test_basic_error_format(self, handler):
        """Basic error record formats with HTML."""
        record = logging.LogRecord(
            name="test.module", level=logging.ERROR,
            pathname="test.py", lineno=42,
            msg="Something went wrong", args=(), exc_info=None,
        )
        message = handler._format_error_message(record)
        assert "OneMil ERROR" in message
        assert "test.module" in message
        assert "test.py:42" in message
        assert "Something went wrong" in message

    def test_error_with_exception(self, handler):
        """Error with exception info includes traceback."""
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test", level=logging.ERROR,
            pathname="test.py", lineno=1,
            msg="Error occurred", args=(), exc_info=exc_info,
        )
        message = handler._format_error_message(record)
        assert "Exception:" in message
        assert "ValueError" in message


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    """Tests for error deduplication logic."""

    def test_first_error_sends(self, handler):
        """First error always sends."""
        record = logging.LogRecord(
            name="test", level=logging.ERROR,
            pathname="test.py", lineno=1,
            msg="Error 1", args=(), exc_info=None,
        )
        with patch.object(handler, '_send_async') as mock_send:
            handler.emit(record)
            mock_send.assert_called_once()

    def test_duplicate_error_suppressed(self, handler):
        """Duplicate error is suppressed (not sent again)."""
        record = logging.LogRecord(
            name="test", level=logging.ERROR,
            pathname="test.py", lineno=1,
            msg="Same error", args=(), exc_info=None,
        )
        with patch.object(handler, '_send_async') as mock_send:
            handler.emit(record)  # First: sends
            handler.emit(record)  # Second: suppressed
            assert mock_send.call_count == 1

    def test_different_error_sends(self, handler):
        """Different error message sends immediately."""
        record1 = logging.LogRecord(
            name="test", level=logging.ERROR,
            pathname="test.py", lineno=1,
            msg="Error A", args=(), exc_info=None,
        )
        record2 = logging.LogRecord(
            name="test", level=logging.ERROR,
            pathname="test.py", lineno=2,
            msg="Error B", args=(), exc_info=None,
        )
        with patch.object(handler, '_send_async') as mock_send:
            handler.emit(record1)
            handler.emit(record2)
            assert mock_send.call_count == 2

    def test_tenth_duplicate_sends(self, handler):
        """Every 10th duplicate error sends."""
        record = logging.LogRecord(
            name="test", level=logging.ERROR,
            pathname="test.py", lineno=1,
            msg="Repeated error", args=(), exc_info=None,
        )
        with patch.object(handler, '_send_async') as mock_send:
            for i in range(10):
                handler.emit(record)
            # First (count=1) + 10th (count=10) = 2 sends
            assert mock_send.call_count == 2


def test_wait_for_sends_blocks_until_thread_finished_and_prunes():
    """wait_for_sends returns only after the (patched, 0.2 s) sender thread has finished."""
    import time
    from unittest.mock import patch
    from monitoring.telegram_error_handler import TelegramErrorHandler
    with patch("monitoring.telegram_error_handler.TelegramNotifier"):
        h = TelegramErrorHandler(bot_token="t", chat_id="1")
    done = []
    h._send_sync = lambda m: (time.sleep(0.2), done.append(m))
    h._send_async("boom")
    assert len(h._send_threads) == 1 and not done
    h.wait_for_sends(5.0)
    assert done == ["boom"] and h._send_threads == []


def test_wait_for_sends_respects_timeout():
    """A stuck sender does not block past the timeout."""
    import time
    import threading
    from unittest.mock import patch
    from monitoring.telegram_error_handler import TelegramErrorHandler
    with patch("monitoring.telegram_error_handler.TelegramNotifier"):
        h = TelegramErrorHandler(bot_token="t", chat_id="1")
    stop = threading.Event()
    h._send_sync = lambda m: stop.wait(5)
    h._send_async("stuck")
    t0 = time.monotonic()
    h.wait_for_sends(0.2)
    assert time.monotonic() - t0 < 1.5 and len(h._send_threads) == 1
    stop.set()
    h.wait_for_sends(2.0)
