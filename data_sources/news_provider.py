"""
News provider with pluggable analysis hook.

Uses Alpaca News API for fetching articles and NewsAnalyzer
for determining if articles represent meaningful catalysts.

V1 NewsAnalyzer is a stub that returns True for any article.
LLMNewsAnalyzer uses Claude Haiku 4.5 to classify articles.
"""

import json
import logging
import time
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# HTTP status codes worth retrying — server-side capacity / transient.
# 429 rate_limit, 502 bad_gateway, 503 service_unavailable, 504 gateway_timeout,
# 529 anthropic-specific overloaded_error.
_RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504, 529})

# Class names for SDK exceptions without an HTTP status (network/timeout).
_RETRYABLE_EXC_NAMES = frozenset({
    'APIConnectionError', 'APITimeoutError', 'APIConnectionTimeoutError',
})

SYSTEM_PROMPT = (
    "You classify stock news. Reply with ONLY a JSON object, no other text.\n"
    "Format: {\"catalyst\": true/false, \"category\": \"<CATEGORY>\", \"reason\": \"<max 20 words>\"}\n\n"
    "Categories (catalyst=true):\n"
    "  FDA_CLINICAL — FDA approval, clinical trial results, drug/therapy news\n"
    "  EARNINGS — quarterly results, revenue, EPS, guidance, beat/miss\n"
    "  CONTRACT_DEAL — contract win, partnership, licensing deal, collaboration\n"
    "  MA — merger, acquisition, buyout, tender offer\n"
    "  ANALYST — upgrade, downgrade, price target, initiate coverage\n"
    "  PRODUCT_LAUNCH — new product, expansion, patent, initiative\n"
    "  MANAGEMENT — CEO/CFO hire/resign, board changes\n"
    "  SEC_FILING — offering, IPO, shelf registration\n"
    "\n"
    "Categories (catalyst=false):\n"
    "  GARBAGE_RECAP — listicle, \"why is X moving\", \"stocks moving in\", market roundup\n"
    "  OTHER — tangential mention, penny stock promo, technical analysis, unrelated\n"
    "  NO_NEWS — no meaningful content\n"
    "\n"
    "CRITICAL RULES:\n"
    "1. 'Why is X stock moving' or 'N stocks moving in session' = ALWAYS GARBAGE_RECAP\n"
    "2. If stock is >80% below 52-week high AND has float <500K: treat ANY 'contract', "
    "'agreement', 'partnership' with undisclosed/unnamed counterparty as GARBAGE_RECAP "
    "(penny stock promotional PR, not a real catalyst)\n"
    "3. If no dollar amounts in the deal AND counterparty is 'undisclosed' = suspicious\n"
    "4. Multiple similar PRs from same company in short period = pump cadence = GARBAGE_RECAP"
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
    def _parse_response(raw: str) -> Tuple[bool, str, str]:
        """
        Parse LLM JSON response into (catalyst_bool, category, reason_string).

        Handles JSON (with or without markdown code fences) and plain TRUE/FALSE fallback.

        Returns:
            Tuple of (is_catalyst: bool, category: str, reason: str)
        """
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
            catalyst = bool(data.get("catalyst", False))
            category = str(data.get("category", "OTHER"))[:30]
            reason = str(data.get("reason", ""))[:100]
            return catalyst, category, reason
        except (json.JSONDecodeError, AttributeError):
            first_word = raw.split()[0].upper() if raw.split() else ""
            logger.warning(
                f"LLM returned non-JSON response, falling back to text parsing: "
                f"'{raw[:80]}'"
            )
            return first_word == "TRUE", "OTHER", f"text-fallback: {raw[:60]}"

    def classify(self, article: Dict, symbol: str = None,
                 stock_context: Optional[Dict] = None) -> Tuple[bool, str, str]:
        """
        Classify a news article using Claude Haiku 4.5.

        Returns classification, category, AND reason for persistence.

        Args:
            article: Dict with headline, summary, source, created_at, url
            symbol: Stock symbol for context in the prompt
            stock_context: Optional dict with float_shares, price, pct_from_high
                          for penny stock promo detection

        Returns:
            Tuple of (is_catalyst: bool, category: str, reason: str)
        """
        headline = (article.get('headline') or '').strip()
        summary = (article.get('summary') or '').strip()

        if not headline and not summary:
            logger.warning(
                f"{symbol or '???'}: empty headline+summary, skipping LLM call"
            )
            return False, 'NO_NEWS', 'empty_content'

        # Check cache
        cache_key = (symbol or '', headline)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            logger.debug(f"{symbol}: cache hit for '{headline[:60]}' -> {cached}")
            return cached, 'cached', 'cached'

        # Build user prompt with stock context
        truncated_summary = summary[:200]
        context_lines = ""
        if stock_context:
            fl = stock_context.get('float_shares', 0)
            pct = stock_context.get('pct_from_high', 0)
            price = stock_context.get('price', 0)
            if fl:
                context_lines += f"Float: {fl:,.0f} shares\n"
            if pct:
                context_lines += f"Stock is {abs(pct):.0f}% below 52-week high\n"
            if price:
                context_lines += f"Current price: ${price:.2f}\n"

        user_msg = (
            f"Symbol: {symbol or 'UNKNOWN'}\n"
            f"{context_lines}"
            f"Headline: {headline}\n"
            f"Summary: {truncated_summary}"
        )

        # Up to 3 attempts with exponential backoff for transient capacity errors
        # (HTTP 529 overloaded, 503 unavailable, etc.). Backoff sleeps before
        # attempts 2 and 3 only — first attempt is immediate.
        # Total worst-case sleep on full failure: 1s + 2s = 3s. The retry loop
        # runs in the NewsWorker background thread, so it does NOT block the
        # trading hot path (which reads a pre-populated dict via get_result).
        MAX_ATTEMPTS = 3
        raw = None
        last_error: Optional[Exception] = None
        for attempt in range(MAX_ATTEMPTS):
            if attempt > 0:
                # Exponential: 1s, 2s, 4s, ... — derived from attempt index so
                # bumping MAX_ATTEMPTS doesn't require touching a backoff tuple.
                time.sleep(2 ** (attempt - 1))
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=150,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = response.content[0].text.strip()
                break  # success — exit retry loop
            except Exception as e:
                last_error = e
                if not self._is_retryable_error(e):
                    # Permanent (e.g., 4xx auth, 400 bad request) — don't retry.
                    logger.error(
                        f"{symbol}: LLM classification failed (non-retryable): {e} "
                        f"— defaulting to False, NOT cached"
                    )
                    break
                logger.warning(
                    f"{symbol}: LLM classification attempt "
                    f"{attempt + 1}/{MAX_ATTEMPTS} failed (retryable): {e}"
                )

        if raw is None:
            # All attempts failed (or non-retryable broke out early).
            # Do NOT cache the failure — next scanner cycle gets a clean retry.
            if last_error and self._is_retryable_error(last_error):
                logger.error(
                    f"{symbol}: LLM classification exhausted {MAX_ATTEMPTS} "
                    f"retries — last error: {last_error}. Returning False "
                    f"(NOT cached, will retry on next scanner cycle)"
                )
            return False, 'OTHER', f'error: {str(last_error)[:50]}'

        # Success — parse, log, cache, return.
        result, category, reason = self._parse_response(raw)
        logger.info(
            f"{symbol}: LLM classified '{headline[:60]}' -> {result} "
            f"[{category}] (reason='{reason}')"
        )
        self._cache[cache_key] = result
        return result, category, reason

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        """True if the exception is a transient capacity/server error worth retrying.

        Checks HTTP status_code first (Anthropic SDK sets this on APIStatusError
        subclasses). Falls back to exception class name for network errors that
        have no HTTP response (APIConnectionError, APITimeoutError).
        """
        status = getattr(exc, 'status_code', None)
        if status in _RETRYABLE_STATUS_CODES:
            return True
        if type(exc).__name__ in _RETRYABLE_EXC_NAMES:
            return True
        return False

    def is_interesting(self, article: Dict, symbol: str = None) -> bool:
        """
        Classify a news article. Returns True if real catalyst.

        Delegates to classify() and discards category/reason.
        """
        result, _cat, _reason = self.classify(article, symbol=symbol)
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

    def classify_news(self, symbol: str, limit: int = 5,
                      stock_context: Optional[Dict] = None) -> Dict:
        """
        Classify news for a symbol with full LLM output for persistence.

        Returns the classification result including the LLM's reason,
        so it can be stored with the trade record for future analysis.

        Args:
            symbol: Stock symbol
            limit: Maximum articles to check

        Returns:
            Dict with keys: has_news, catalyst, category, headline, reason
            - has_news: bool — any articles found
            - catalyst: bool or None — LLM classification (None if no articles)
            - category: str — news category (FDA_CLINICAL, EARNINGS, etc.)
            - headline: str — top article headline
            - reason: str — LLM's reason for classification
        """
        result = {'has_news': False, 'catalyst': None, 'category': 'NO_NEWS',
                  'headline': '', 'reason': '', 'news_headline': ''}

        try:
            articles = self.get_recent_news(symbol, limit=limit)
        except Exception as e:
            logger.error(f"{symbol}: news fetch failed: {e}")
            return result

        if not articles:
            return result

        result['has_news'] = True
        result['headline'] = (articles[0].get('headline') or '')[:200]
        result['news_headline'] = result['headline']

        # Classify articles — find the BEST (highest quality) catalyst
        best_catalyst = False
        best_category = 'OTHER'
        best_reason = ''
        best_headline = result['headline']

        # Priority: real catalysts first
        catalyst_priority = ['FDA_CLINICAL', 'MA', 'EARNINGS', 'CONTRACT_DEAL',
                            'ANALYST', 'PRODUCT_LAUNCH', 'MANAGEMENT', 'SEC_FILING']

        for article in articles:
            if hasattr(self.analyzer, 'classify'):
                catalyst, category, reason = self.analyzer.classify(
                    article, symbol=symbol, stock_context=stock_context)
            else:
                catalyst = self.analyzer.is_interesting(article, symbol=symbol)
                category = 'OTHER'
                reason = 'stub_v1'

            if catalyst and category in catalyst_priority:
                # Found a real catalyst — use it
                result['catalyst'] = True
                result['category'] = category
                result['headline'] = (article.get('headline') or '')[:200]
                result['news_headline'] = result['headline']
                result['reason'] = reason
                return result

            # Track best non-catalyst classification
            if not best_catalyst:
                best_category = category
                best_reason = reason
                best_headline = (article.get('headline') or '')[:200]

        # No real catalyst found
        result['catalyst'] = False
        result['category'] = best_category
        result['reason'] = best_reason
        result['headline'] = best_headline
        result['news_headline'] = best_headline

        return result


class NewsWorker:
    """
    Async news classification worker.

    Decouples news API + LLM calls from the scanner loop. Scanner enqueues
    symbols, worker classifies in background. Results cached for trading
    engine to read at trade time.

    Usage:
        worker = NewsWorker(news_provider)
        worker.start()
        worker.enqueue('AAPL', stock_context={...})
        # Later:
        result = worker.get_result('AAPL')  # None if not yet classified
    """

    def __init__(self, news_provider: 'NewsProvider'):
        """
        Initialize NewsWorker.

        Args:
            news_provider: NewsProvider instance for classification
        """
        import queue as _q
        import threading as _t
        self._news = news_provider
        self._queue: _q.Queue = _q.Queue()
        self._results: Dict[str, Dict] = {}  # symbol -> classification result
        self._pending: set = set()  # symbols queued but not yet classified
        self._lock = _t.Lock()
        self._thread = _t.Thread(target=self._run, daemon=True, name='news-worker')
        self._running = False

    def start(self):
        """Start the background worker thread."""
        self._running = True
        self._thread.start()
        logger.info("NewsWorker started (async news classification)")

    def stop(self):
        """Signal the worker to stop."""
        self._running = False

    def enqueue(self, symbol: str, stock_context: Optional[Dict] = None):
        """
        Queue a symbol for news classification.

        Skips if already classified or already pending.

        Args:
            symbol: Stock symbol
            stock_context: Optional dict with float_shares, price for LLM context
        """
        with self._lock:
            if symbol in self._results or symbol in self._pending:
                return  # Already classified or queued
            self._pending.add(symbol)
        self._queue.put((symbol, stock_context))

    def get_result(self, symbol: str) -> Optional[Dict]:
        """
        Get cached classification result for a symbol.

        Returns:
            Classification dict if available, None if pending/not requested
        """
        with self._lock:
            return self._results.get(symbol)

    def _run(self):
        """Background worker loop — processes news queue."""
        while self._running:
            try:
                symbol, stock_context = self._queue.get(timeout=1.0)
            except Exception:
                continue  # Timeout, check _running flag

            try:
                result = self._news.classify_news(symbol, stock_context=stock_context)
                with self._lock:
                    self._results[symbol] = result
                    self._pending.discard(symbol)
                logger.debug(
                    f"{symbol}: news classified async — "
                    f"catalyst={result.get('catalyst')}, "
                    f"category={result.get('category', 'N/A')}"
                )
            except Exception as e:
                logger.warning(f"{symbol}: async news classification failed: {e}")
                with self._lock:
                    self._pending.discard(symbol)
                    # Don't cache errors — allow re-enqueue on next scanner cycle
                    # Transient LLM failures shouldn't permanently classify a stock
