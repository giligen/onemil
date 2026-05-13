"""Tests for AlpacaClient._log_order_op_failure log-level routing.

Background: yesterday (2026-05-12 FC) VG + BMNZ both closed cleanly via
the ORB FC retry helper, but four ERROR-level log lines from
alpaca_client.py spammed the operator inbox — the API layer logged ERROR
on every failed close_position call, including the well-known
held_for_orders (40310000) race that the retry helper handles by design.

This test suite validates the level-routing helper:
  - WARNING for known-transient errors (caller may retry via backoff)
  - ERROR for everything else (real operator-attention failures)

If a future change reverts to blanket-ERROR logging, these tests fail.
"""
from __future__ import annotations

import logging
import pytest

from data_sources.alpaca_client import AlpacaClient


class TestLogOrderOpFailure:
    """The helper picks WARNING vs ERROR based on the exception text."""

    def test_held_for_orders_40310000_is_warning(self, caplog):
        e = RuntimeError(
            '{"available":"0","code":40310000,"existing_qty":"1431",'
            '"held_for_orders":"1431","message":"insufficient qty available '
            'for order","symbol":"GLWG"}'
        )
        with caplog.at_level(logging.DEBUG, logger='data_sources.alpaca_client'):
            AlpacaClient._log_order_op_failure("close position", "GLWG", e)
        # Exactly one record, at WARNING level (not ERROR)
        recs = [r for r in caplog.records if 'GLWG' in r.getMessage()]
        assert len(recs) == 1
        assert recs[0].levelno == logging.WARNING
        assert 'transient error' in recs[0].getMessage()
        assert 'caller may retry' in recs[0].getMessage()

    def test_insufficient_qty_text_is_warning(self, caplog):
        """Same as 40310000 but matched by message text instead of code —
        some API responses include the message without the code."""
        e = RuntimeError("insufficient qty available for order (requested: 500)")
        with caplog.at_level(logging.DEBUG, logger='data_sources.alpaca_client'):
            AlpacaClient._log_order_op_failure("submit limit sell order", "ABSI", e)
        recs = [r for r in caplog.records if 'ABSI' in r.getMessage()]
        assert len(recs) == 1
        assert recs[0].levelno == logging.WARNING

    def test_rate_limit_is_warning(self, caplog):
        e = RuntimeError("Rate limit exceeded: too many requests")
        with caplog.at_level(logging.DEBUG, logger='data_sources.alpaca_client'):
            AlpacaClient._log_order_op_failure("submit bracket order", "AAPL", e)
        recs = [r for r in caplog.records if 'AAPL' in r.getMessage()]
        assert len(recs) == 1
        assert recs[0].levelno == logging.WARNING

    def test_too_many_requests_is_warning(self, caplog):
        e = RuntimeError("HTTP 429 Too Many Requests")
        with caplog.at_level(logging.DEBUG, logger='data_sources.alpaca_client'):
            AlpacaClient._log_order_op_failure("submit limit buy order", "MSFT", e)
        recs = [r for r in caplog.records if 'MSFT' in r.getMessage()]
        assert len(recs) == 1
        assert recs[0].levelno == logging.WARNING

    @pytest.mark.parametrize("phrase", [
        "Internal server error",
        "Bad Gateway",
        "Service Unavailable",
        "Gateway Timeout",
    ])
    def test_5xx_phrases_are_warning(self, phrase, caplog):
        e = RuntimeError(f"Alpaca returned: {phrase}")
        with caplog.at_level(logging.DEBUG, logger='data_sources.alpaca_client'):
            AlpacaClient._log_order_op_failure("close position", "NVDA", e)
        recs = [r for r in caplog.records if 'NVDA' in r.getMessage()]
        assert len(recs) == 1
        assert recs[0].levelno == logging.WARNING, (
            f"5xx phrase {phrase!r} should be WARNING"
        )

    def test_auth_failure_is_error(self, caplog):
        """Auth failure is a real operator-attention issue — stays at ERROR."""
        e = RuntimeError("Unauthorized: invalid API key")
        with caplog.at_level(logging.DEBUG, logger='data_sources.alpaca_client'):
            AlpacaClient._log_order_op_failure("submit bracket order", "AAPL", e)
        recs = [r for r in caplog.records if 'AAPL' in r.getMessage()]
        assert len(recs) == 1
        assert recs[0].levelno == logging.ERROR

    def test_invalid_order_is_error(self, caplog):
        """Invalid order params is a real bug, not a transient race."""
        e = RuntimeError("invalid limit_price: cannot be negative")
        with caplog.at_level(logging.DEBUG, logger='data_sources.alpaca_client'):
            AlpacaClient._log_order_op_failure("submit limit sell order", "TSLA", e)
        recs = [r for r in caplog.records if 'TSLA' in r.getMessage()]
        assert len(recs) == 1
        assert recs[0].levelno == logging.ERROR

    def test_network_failure_is_error(self, caplog):
        """Connection failures are operator-relevant — not auto-recoverable."""
        e = RuntimeError("Connection refused")
        with caplog.at_level(logging.DEBUG, logger='data_sources.alpaca_client'):
            AlpacaClient._log_order_op_failure("close position", "GOOG", e)
        recs = [r for r in caplog.records if 'GOOG' in r.getMessage()]
        assert len(recs) == 1
        assert recs[0].levelno == logging.ERROR

    def test_no_false_positive_on_500_in_price(self, caplog):
        """Numeric-coincidence guard: a price/qty containing '500' must NOT
        match the 5xx detector. Regression for the design note in
        _log_order_op_failure — uses phrase-based 5xx detection
        (e.g. 'internal server error'), not raw substring '500'."""
        # Error mentions qty 5153 — contains '500' lexically? Actually '5153'
        # doesn't contain '500' but qty 50000 does. Use a case where the
        # number contains 500 to validate.
        e = RuntimeError("Insufficient buying power for 500 shares")
        with caplog.at_level(logging.DEBUG, logger='data_sources.alpaca_client'):
            AlpacaClient._log_order_op_failure("submit bracket order", "AMZN", e)
        recs = [r for r in caplog.records if 'AMZN' in r.getMessage()]
        assert len(recs) == 1
        assert recs[0].levelno == logging.ERROR, (
            "'500 shares' in an error message must NOT be classified as 5xx"
        )
