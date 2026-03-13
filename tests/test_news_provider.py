"""
Tests for data_sources/news_provider.py - NewsAnalyzer and NewsProvider.

Covers:
- NewsAnalyzer.is_interesting (V1 always True)
- NewsProvider.get_recent_news delegates to alpaca_client
- has_interesting_news: with articles, without articles, with custom analyzer
"""

import pytest
from unittest.mock import MagicMock

from data_sources.alpaca_client import AlpacaClient
from data_sources.news_provider import NewsAnalyzer, NewsProvider


# =============================================================================
# NewsAnalyzer
# =============================================================================

class TestNewsAnalyzer:
    """Tests for NewsAnalyzer (V1 stub)."""

    def test_is_interesting_returns_true(self):
        """V1 analyzer always returns True for any article."""
        analyzer = NewsAnalyzer()
        article = {'headline': 'Earnings beat', 'summary': 'Big win'}
        assert analyzer.is_interesting(article) is True

    def test_is_interesting_empty_article(self):
        """V1 analyzer returns True even for empty article dict."""
        analyzer = NewsAnalyzer()
        assert analyzer.is_interesting({}) is True

    def test_is_interesting_none_values(self):
        """V1 analyzer returns True regardless of article content."""
        analyzer = NewsAnalyzer()
        article = {'headline': None, 'summary': None}
        assert analyzer.is_interesting(article) is True


# =============================================================================
# NewsProvider.get_recent_news
# =============================================================================

class TestGetRecentNews:
    """Tests for NewsProvider.get_recent_news delegation."""

    def test_delegates_to_alpaca_client(self):
        """get_recent_news passes through to alpaca_client.get_news."""
        mock_client = MagicMock(spec=AlpacaClient)
        mock_client.get_news.return_value = [
            {'headline': 'News 1'},
            {'headline': 'News 2'},
        ]
        provider = NewsProvider(mock_client)

        result = provider.get_recent_news("AAPL", limit=3)

        mock_client.get_news.assert_called_once_with("AAPL", limit=3)
        assert len(result) == 2

    def test_delegates_default_limit(self):
        """get_recent_news uses default limit of 5."""
        mock_client = MagicMock(spec=AlpacaClient)
        mock_client.get_news.return_value = []
        provider = NewsProvider(mock_client)

        provider.get_recent_news("XYZ")

        mock_client.get_news.assert_called_once_with("XYZ", limit=5)


# =============================================================================
# NewsProvider.has_interesting_news
# =============================================================================

class TestHasInterestingNews:
    """Tests for NewsProvider.has_interesting_news."""

    def test_with_articles_returns_true_and_headline(self):
        """has_interesting_news returns (True, headline) when articles exist and are interesting."""
        mock_client = MagicMock(spec=AlpacaClient)
        mock_client.get_news.return_value = [
            {'headline': 'FDA Approval Announced'},
            {'headline': 'Stock surges'},
        ]
        provider = NewsProvider(mock_client)

        has_news, headline = provider.has_interesting_news("DRUG")

        assert has_news is True
        assert headline == 'FDA Approval Announced'

    def test_without_articles_returns_false(self):
        """has_interesting_news returns (False, None) when no articles found."""
        mock_client = MagicMock(spec=AlpacaClient)
        mock_client.get_news.return_value = []
        provider = NewsProvider(mock_client)

        has_news, headline = provider.has_interesting_news("QUIET")

        assert has_news is False
        assert headline is None

    def test_custom_analyzer_returns_false(self):
        """has_interesting_news returns (False, None) when analyzer rejects all articles."""
        mock_client = MagicMock(spec=AlpacaClient)
        mock_client.get_news.return_value = [
            {'headline': 'Boring update'},
            {'headline': 'Another boring one'},
        ]

        # Custom analyzer that always says no
        strict_analyzer = MagicMock(spec=NewsAnalyzer)
        strict_analyzer.is_interesting.return_value = False

        provider = NewsProvider(mock_client, analyzer=strict_analyzer)

        has_news, headline = provider.has_interesting_news("BORE")

        assert has_news is False
        assert headline is None
        assert strict_analyzer.is_interesting.call_count == 2

    def test_missing_headline_key_uses_fallback(self):
        """has_interesting_news uses 'No headline' when headline key is missing."""
        mock_client = MagicMock(spec=AlpacaClient)
        mock_client.get_news.return_value = [
            {'summary': 'Some summary but no headline key'},
        ]
        provider = NewsProvider(mock_client)

        has_news, headline = provider.has_interesting_news("NOHDR")

        assert has_news is True
        assert headline == 'No headline'

    def test_custom_limit(self):
        """has_interesting_news passes limit to get_recent_news."""
        mock_client = MagicMock(spec=AlpacaClient)
        mock_client.get_news.return_value = []
        provider = NewsProvider(mock_client)

        provider.has_interesting_news("XYZ", limit=10)

        mock_client.get_news.assert_called_once_with("XYZ", limit=10)
