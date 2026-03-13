"""
Tests for data_sources/news_provider.py - NewsAnalyzer, LLMNewsAnalyzer, NewsProvider.

Covers:
- NewsAnalyzer.is_interesting (V1 always True)
- LLMNewsAnalyzer.is_interesting (mocked Anthropic client)
- NewsProvider.get_recent_news delegates to alpaca_client
- has_interesting_news: with articles, without articles, with custom analyzer
"""

import pytest
from unittest.mock import MagicMock, call

from data_sources.alpaca_client import AlpacaClient
from data_sources.news_provider import NewsAnalyzer, LLMNewsAnalyzer, NewsProvider


# =============================================================================
# NewsAnalyzer (V1 stub)
# =============================================================================

class TestNewsAnalyzer:
    """Tests for NewsAnalyzer (V1 stub)."""

    def test_is_interesting_returns_true(self):
        """V1 analyzer always returns True for any article."""
        analyzer = NewsAnalyzer()
        article = {'headline': 'Earnings beat', 'summary': 'Big win'}
        assert analyzer.is_interesting(article) is True

    def test_is_interesting_with_symbol(self):
        """V1 analyzer accepts optional symbol kwarg."""
        analyzer = NewsAnalyzer()
        article = {'headline': 'FDA Approval', 'summary': 'Drug approved'}
        assert analyzer.is_interesting(article, symbol='DRUG') is True

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
# LLMNewsAnalyzer (mocked Anthropic client)
# =============================================================================

class TestLLMNewsAnalyzer:
    """Tests for LLMNewsAnalyzer with mocked Anthropic API."""

    def _make_mock_client(self, response_text='{"catalyst": true, "reason": "test"}'):
        """Create a mock Anthropic client that returns the given text."""
        client = MagicMock()
        mock_content = MagicMock()
        mock_content.text = response_text
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        client.messages.create.return_value = mock_response
        return client

    def test_real_catalyst_returns_true(self):
        """LLM returning catalyst=true JSON => is_interesting returns True."""
        client = self._make_mock_client('{"catalyst": true, "reason": "FDA approval"}')
        analyzer = LLMNewsAnalyzer(client, model="test-model")

        result = analyzer.is_interesting(
            {'headline': 'FDA Approves Drug', 'summary': 'Big pharma win'},
            symbol='DRUG',
        )

        assert result is True
        client.messages.create.assert_called_once()

    def test_noise_returns_false(self):
        """LLM returning catalyst=false JSON => is_interesting returns False."""
        client = self._make_mock_client('{"catalyst": false, "reason": "generic listicle"}')
        analyzer = LLMNewsAnalyzer(client, model="test-model")

        result = analyzer.is_interesting(
            {'headline': '12 Stocks Moving Thursday', 'summary': 'Market recap'},
            symbol='XYZ',
        )

        assert result is False

    def test_api_error_returns_false(self):
        """API exception => is_interesting returns False (safe default)."""
        client = MagicMock()
        client.messages.create.side_effect = Exception("API timeout")
        analyzer = LLMNewsAnalyzer(client)

        result = analyzer.is_interesting(
            {'headline': 'Earnings Beat', 'summary': 'Revenue surged'},
            symbol='EARN',
        )

        assert result is False

    def test_cache_prevents_duplicate_calls(self):
        """Second call with same (symbol, headline) uses cache, no API call."""
        client = self._make_mock_client('{"catalyst": true, "reason": "FDA"}')
        analyzer = LLMNewsAnalyzer(client)

        article = {'headline': 'FDA Approval', 'summary': 'Approved'}

        result1 = analyzer.is_interesting(article, symbol='DRUG')
        result2 = analyzer.is_interesting(article, symbol='DRUG')

        assert result1 is True
        assert result2 is True
        assert client.messages.create.call_count == 1

    def test_empty_article_returns_false(self):
        """Empty headline + summary => False without API call."""
        client = self._make_mock_client('{"catalyst": true, "reason": "test"}')
        analyzer = LLMNewsAnalyzer(client)

        result = analyzer.is_interesting(
            {'headline': '', 'summary': ''},
            symbol='EMPTY',
        )

        assert result is False
        client.messages.create.assert_not_called()

    def test_none_headline_summary_returns_false(self):
        """None headline + summary => False without API call."""
        client = self._make_mock_client('{"catalyst": true, "reason": "test"}')
        analyzer = LLMNewsAnalyzer(client)

        result = analyzer.is_interesting(
            {'headline': None, 'summary': None},
            symbol='NONE',
        )

        assert result is False
        client.messages.create.assert_not_called()

    def test_symbol_in_prompt(self):
        """Verify symbol appears in the user message sent to API."""
        client = self._make_mock_client('{"catalyst": false, "reason": "noise"}')
        analyzer = LLMNewsAnalyzer(client)

        analyzer.is_interesting(
            {'headline': 'Some headline', 'summary': 'Some summary'},
            symbol='AAPL',
        )

        call_kwargs = client.messages.create.call_args
        user_msg = call_kwargs.kwargs['messages'][0]['content']
        assert 'AAPL' in user_msg

    def test_summary_truncated(self):
        """Long summary is truncated to 200 chars in the prompt."""
        client = self._make_mock_client('{"catalyst": true, "reason": "test"}')
        analyzer = LLMNewsAnalyzer(client)

        long_summary = "A" * 500
        analyzer.is_interesting(
            {'headline': 'Headline', 'summary': long_summary},
            symbol='LONG',
        )

        call_kwargs = client.messages.create.call_args
        user_msg = call_kwargs.kwargs['messages'][0]['content']
        # Summary line should contain at most 200 A's
        summary_line = [l for l in user_msg.split('\n') if l.startswith('Summary:')][0]
        summary_content = summary_line.split('Summary: ')[1]
        assert len(summary_content) == 200

    def test_unexpected_response_returns_false(self):
        """Non-JSON, non-TRUE response (e.g. 'MAYBE') => False (safe default)."""
        client = self._make_mock_client("MAYBE")
        analyzer = LLMNewsAnalyzer(client)

        result = analyzer.is_interesting(
            {'headline': 'Ambiguous news', 'summary': 'Unclear impact'},
            symbol='AMB',
        )

        assert result is False

    def test_plain_text_true_fallback(self):
        """Plain 'TRUE' text (non-JSON) falls back to text parsing and returns True."""
        client = self._make_mock_client("TRUE")
        analyzer = LLMNewsAnalyzer(client)

        result = analyzer.is_interesting(
            {'headline': 'FDA Approval', 'summary': 'Drug approved'},
            symbol='DRUG',
        )

        assert result is True

    def test_plain_text_false_fallback(self):
        """Plain 'FALSE' text (non-JSON) falls back to text parsing and returns False."""
        client = self._make_mock_client("FALSE")
        analyzer = LLMNewsAnalyzer(client)

        result = analyzer.is_interesting(
            {'headline': 'Market recap', 'summary': 'Stocks moved'},
            symbol='XYZ',
        )

        assert result is False

    def test_json_with_extra_whitespace(self):
        """JSON response with whitespace still parses correctly."""
        client = self._make_mock_client('  {"catalyst": true, "reason": "earnings beat"}  ')
        analyzer = LLMNewsAnalyzer(client)

        result = analyzer.is_interesting(
            {'headline': 'Earnings Beat', 'summary': 'Revenue up'},
            symbol='EARN',
        )

        assert result is True

    def test_different_symbols_not_cached_together(self):
        """Same headline for different symbols should make separate API calls."""
        client = self._make_mock_client('{"catalyst": true, "reason": "test"}')
        analyzer = LLMNewsAnalyzer(client)

        article = {'headline': 'Market Update', 'summary': 'Details'}

        analyzer.is_interesting(article, symbol='AAA')
        analyzer.is_interesting(article, symbol='BBB')

        assert client.messages.create.call_count == 2

    def test_parse_response_valid_json_true(self):
        """_parse_response correctly parses catalyst=true JSON."""
        result, reason = LLMNewsAnalyzer._parse_response(
            '{"catalyst": true, "reason": "FDA approval for new drug"}'
        )
        assert result is True
        assert "FDA" in reason

    def test_parse_response_valid_json_false(self):
        """_parse_response correctly parses catalyst=false JSON."""
        result, reason = LLMNewsAnalyzer._parse_response(
            '{"catalyst": false, "reason": "generic market recap"}'
        )
        assert result is False
        assert "recap" in reason

    def test_parse_response_text_fallback(self):
        """_parse_response falls back to text parsing for non-JSON."""
        result, reason = LLMNewsAnalyzer._parse_response("TRUE\nSome explanation")
        assert result is True
        assert "text-fallback" in reason

    def test_parse_response_code_fence_json(self):
        """_parse_response strips markdown code fences around JSON."""
        raw = '```json\n{"catalyst": true, "reason": "FDA approval"}\n```'
        result, reason = LLMNewsAnalyzer._parse_response(raw)
        assert result is True
        assert "FDA" in reason

    def test_parse_response_code_fence_multiline(self):
        """_parse_response handles multiline JSON inside code fences."""
        raw = '```json\n{\n  "catalyst": false,\n  "reason": "generic recap"\n}\n```'
        result, reason = LLMNewsAnalyzer._parse_response(raw)
        assert result is False
        assert "recap" in reason


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

    def test_passes_symbol_to_analyzer(self):
        """has_interesting_news passes symbol to analyzer.is_interesting."""
        mock_client = MagicMock(spec=AlpacaClient)
        mock_client.get_news.return_value = [
            {'headline': 'Some news'},
        ]

        mock_analyzer = MagicMock(spec=NewsAnalyzer)
        mock_analyzer.is_interesting.return_value = True

        provider = NewsProvider(mock_client, analyzer=mock_analyzer)
        provider.has_interesting_news("AAPL")

        mock_analyzer.is_interesting.assert_called_once_with(
            {'headline': 'Some news'}, symbol='AAPL'
        )

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

    def test_api_error_returns_false_and_logs(self):
        """has_interesting_news returns (False, None) on API error, logs ERROR."""
        from data_sources.alpaca_client import AlpacaAPIError
        mock_client = MagicMock(spec=AlpacaClient)
        mock_client.get_news.side_effect = AlpacaAPIError("News API down")
        provider = NewsProvider(mock_client)

        has_news, headline = provider.has_interesting_news("FAIL")

        assert has_news is False
        assert headline is None
