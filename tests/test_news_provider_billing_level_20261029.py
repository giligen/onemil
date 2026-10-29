"""Billing/auth 4xx from the LLM classifier: one WARNING per hour, False returned, never cached."""
import logging
from unittest.mock import MagicMock

from data_sources.news_provider import LLMNewsAnalyzer

ARTICLE = {'headline': 'Real catalyst', 'summary': 'foo'}


def _billing_error():
    exc = Exception("Error code: 400 - {'message': 'Your credit balance is too low to access the API'}")
    exc.status_code = 400
    return exc


def test_billing_400_warns_once_returns_false_not_cached(caplog):
    """Four symbols in a row: one WARNING, zero ERRORs, False each time, every call re-attempted."""
    client = MagicMock()
    client.messages.create.side_effect = _billing_error()
    analyzer = LLMNewsAnalyzer(client, model="test-model")
    with caplog.at_level(logging.WARNING, logger="data_sources.news_provider"):
        results = [analyzer.is_interesting(ARTICLE, symbol=s) for s in ("AAA", "BBB", "CCC", "DDD")]
    assert results == [False] * 4
    assert client.messages.create.call_count == 4  # 1 attempt each (non-retryable), NOT cached
    unavailable = [r for r in caplog.records if "LLM classifier unavailable" in r.getMessage()]
    assert len(unavailable) == 1 and unavailable[0].levelno == logging.WARNING
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_other_non_retryable_error_stays_error(caplog):
    """A non-billing non-retryable error keeps today's ERROR level."""
    client = MagicMock()
    client.messages.create.side_effect = Exception("API timeout")
    analyzer = LLMNewsAnalyzer(client)
    with caplog.at_level(logging.WARNING, logger="data_sources.news_provider"):
        assert analyzer.is_interesting(ARTICLE, symbol="EARN") is False
    assert any(r.levelno == logging.ERROR and "non-retryable" in r.getMessage() for r in caplog.records)
