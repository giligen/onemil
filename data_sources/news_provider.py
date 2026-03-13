"""
News provider with pluggable analysis hook.

Uses Alpaca News API for fetching articles and NewsAnalyzer
for determining if articles represent meaningful catalysts.

V1 NewsAnalyzer is a stub that returns True for any article.
LLMNewsAnalyzer uses Claude Haiku 4.5 to classify articles.
"""

import json
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You classify stock news. Reply with ONLY a JSON object, no other text.\n"
    "Format: {\"catalyst\": true/false, \"reason\": \"<max 20 words>\"}\n\n"
    "catalyst=true: real catalyst (FDA approval, earnings beat/miss, contract win, "
    "merger/acquisition, offering, partnership, analyst upgrade/downgrade, "
    "significant legal/regulatory action).\n"
    "catalyst=false: noise (listicle, generic market recap, tangential mention, "
    "penny stock promo, technical analysis, \"stocks moving\" roundup)."
)


class NewsAnalyzer:
    """
    Pluggable news analysis hook. V1: always returns True.

    Future versions: keyword scoring, sentiment analysis, LLM classification.
    """

    def is_interesting(self, article: Dict, symbol: str = None) -> bool:
        """
        Analyze a news article. Return True if it's a meaningful catalyst.

        V1: Returns True for any article (stub).

        Args:
            article: Dict with headline, summary, source, created_at, url
            symbol: Stock symbol (unused in V1, used by LLMNewsAnalyzer)

        Returns:
            True if article is a meaningful catalyst
        """
        return True


class LLMNewsAnalyzer(NewsAnalyzer):
    """
    News analyzer using Claude Haiku 4.5 to classify articles as
    real catalysts (True) or noise (False).

    Uses an in-memory cache keyed on (symbol, headline) to avoid
    re-classifying the same article across 60s poll cycles.
    """

    def __init__(self, anthropic_client, model: str = "claude-haiku-4-5-20251001"):
        """
        Initialize LLMNewsAnalyzer.

        Args:
            anthropic_client: anthropic.Anthropic client instance
            model: Model ID to use for classification
        """
        self._client = anthropic_client
        self._model = model
        self._cache: Dict[Tuple[str, str], bool] = {}
        logger.info(f"LLMNewsAnalyzer initialized with model={model}")

    @staticmethod
    def _parse_response(raw: str) -> Tuple[bool, str]:
        """
        Parse LLM JSON response into (catalyst_bool, reason_string).

        Handles JSON (with or without markdown code fences) and plain TRUE/FALSE fallback.

        Args:
            raw: Raw response text from LLM

        Returns:
            Tuple of (is_catalyst: bool, reason: str)
        """
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove opening ```json or ``` line
            lines = cleaned.split("\n")
            lines = lines[1:]  # Drop ```json line
            # Remove closing ``` if present
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
            catalyst = bool(data.get("catalyst", False))
            reason = str(data.get("reason", ""))[:100]
            return catalyst, reason
        except (json.JSONDecodeError, AttributeError):
            # Fallback: parse as plain TRUE/FALSE
            first_word = raw.split()[0].upper() if raw.split() else ""
            logger.warning(
                f"LLM returned non-JSON response, falling back to text parsing: "
                f"'{raw[:80]}'"
            )
            return first_word == "TRUE", f"text-fallback: {raw[:60]}"

    def is_interesting(self, article: Dict, symbol: str = None) -> bool:
        """
        Classify a news article using Claude Haiku 4.5.

        Args:
            article: Dict with headline, summary, source, created_at, url
            symbol: Stock symbol for context in the prompt

        Returns:
            True if article is a real catalyst, False if noise or on error
        """
        headline = (article.get('headline') or '').strip()
        summary = (article.get('summary') or '').strip()

        if not headline and not summary:
            logger.warning(
                f"{symbol or '???'}: empty headline+summary, skipping LLM call"
            )
            return False

        # Check cache
        cache_key = (symbol or '', headline)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            logger.debug(f"{symbol}: cache hit for '{headline[:60]}' -> {cached}")
            return cached

        # Build user prompt
        truncated_summary = summary[:200]
        user_msg = (
            f"Symbol: {symbol or 'UNKNOWN'}\n"
            f"Headline: {headline}\n"
            f"Summary: {truncated_summary}"
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=100,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            result, reason = self._parse_response(raw)

            logger.info(
                f"{symbol}: LLM classified '{headline[:60]}' -> {result} "
                f"(reason='{reason}')"
            )
        except Exception as e:
            logger.error(
                f"{symbol}: LLM classification failed: {e} — defaulting to False"
            )
            result = False

        self._cache[cache_key] = result
        return result


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
        try:
            articles = self.get_recent_news(symbol, limit=limit)
        except Exception as e:
            logger.error(f"{symbol}: news fetch failed: {e}")
            return False, None

        if not articles:
            logger.debug(f"{symbol}: no news articles found")
            return False, None

        for article in articles:
            if self.analyzer.is_interesting(article, symbol=symbol):
                headline = article.get('headline', 'No headline')
                logger.debug(f"{symbol}: interesting news found - {headline}")
                return True, headline

        logger.debug(f"{symbol}: {len(articles)} articles found, none interesting")
        return False, None
