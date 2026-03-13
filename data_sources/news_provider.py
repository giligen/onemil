"""
News provider with pluggable analysis hook.

Uses Alpaca News API for fetching articles and NewsAnalyzer
for determining if articles represent meaningful catalysts.

V1 NewsAnalyzer is a stub that returns True for any article.
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class NewsAnalyzer:
    """
    Pluggable news analysis hook. V1: always returns True.

    Future versions: keyword scoring, sentiment analysis, LLM classification.
    """

    def is_interesting(self, article: Dict) -> bool:
        """
        Analyze a news article. Return True if it's a meaningful catalyst.

        V1: Returns True for any article (stub).
        Future: keyword scoring, sentiment analysis, LLM classification.

        Args:
            article: Dict with headline, summary, source, created_at, url

        Returns:
            True if article is a meaningful catalyst
        """
        return True


class NewsProvider:
    """
    Provides news data and analysis for scanner candidates.

    Wraps Alpaca News API (via AlpacaClient) and passes articles
    through NewsAnalyzer to determine relevance.
    """

    def __init__(self, alpaca_client, analyzer: Optional[NewsAnalyzer] = None):
        """
        Initialize NewsProvider.

        Args:
            alpaca_client: AlpacaClient instance for news fetching
            analyzer: NewsAnalyzer instance (defaults to V1 stub)
        """
        self.alpaca_client = alpaca_client
        self.analyzer = analyzer or NewsAnalyzer()
        logger.info("NewsProvider initialized")

    def get_recent_news(self, symbol: str, limit: int = 5) -> List[Dict]:
        """
        Fetch recent news articles for a symbol.

        Args:
            symbol: Stock symbol
            limit: Maximum number of articles

        Returns:
            List of article dicts
        """
        return self.alpaca_client.get_news(symbol, limit=limit)

    def has_interesting_news(self, symbol: str, limit: int = 5) -> tuple:
        """
        Check if a symbol has at least one interesting news article.

        Args:
            symbol: Stock symbol
            limit: Maximum articles to check

        Returns:
            Tuple of (has_news: bool, headline: str or None)
            headline is the first interesting article's headline
        """
        articles = self.get_recent_news(symbol, limit=limit)

        if not articles:
            logger.debug(f"{symbol}: no news articles found")
            return False, None

        for article in articles:
            if self.analyzer.is_interesting(article):
                headline = article.get('headline', 'No headline')
                logger.debug(f"{symbol}: interesting news found - {headline}")
                return True, headline

        logger.debug(f"{symbol}: {len(articles)} articles found, none interesting")
        return False, None
