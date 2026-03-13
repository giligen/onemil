"""
Float data provider using Yahoo Finance (yfinance).

Fetches shares float for stock universe filtering.
Sequential fetching with progress logging (10-20 min for full batch).
Results cached in DB with weekly refresh.
Includes exponential backoff retry for Yahoo Finance rate limits.
"""

import logging
import time
from typing import Optional, Dict

import yfinance as yf

logger = logging.getLogger(__name__)

# Retry config for Yahoo Finance rate limits
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0
# Pause between requests to avoid triggering rate limits
REQUEST_DELAY_SECONDS = 0.5


class FloatProvider:
    """
    Provides shares float data from Yahoo Finance.

    Uses yfinance .info property to get floatShares.
    Sequential processing with progress logging.
    Exponential backoff on rate limit / HTTP errors.
    """

    def __init__(self, max_retries: int = MAX_RETRIES,
                 initial_backoff: float = INITIAL_BACKOFF_SECONDS,
                 request_delay: float = REQUEST_DELAY_SECONDS):
        """
        Initialize FloatProvider with retry configuration.

        Args:
            max_retries: Max retry attempts per symbol on rate limit
            initial_backoff: Initial backoff in seconds (doubles each retry)
            request_delay: Pause between sequential requests (rate limit avoidance)
        """
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.request_delay = request_delay

    def _fetch_with_retry(self, symbol: str, fetch_fn) -> Optional[dict]:
        """
        Execute a yfinance fetch with exponential backoff retry.

        Args:
            symbol: Stock symbol (for logging)
            fetch_fn: Callable that returns the result dict

        Returns:
            Result from fetch_fn, or None on exhausted retries
        """
        backoff = self.initial_backoff

        for attempt in range(self.max_retries + 1):
            try:
                return fetch_fn()
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = any(hint in error_str for hint in [
                    '429', 'rate limit', 'too many requests',
                    'connection', 'timeout', 'read timed out'
                ])

                if is_rate_limit and attempt < self.max_retries:
                    logger.warning(
                        f"{symbol}: rate limited/connection error (attempt {attempt + 1}/{self.max_retries + 1}), "
                        f"retrying in {backoff:.1f}s... Error: {e}"
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    logger.error(f"Failed to fetch data for {symbol}: {e}")
                    return None

        return None

    def get_float(self, symbol: str) -> Optional[int]:
        """
        Get shares float for a single symbol with retry.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Float shares count, or None if unavailable
        """
        def _fetch():
            ticker = yf.Ticker(symbol)
            info = ticker.info
            float_shares = info.get('floatShares')

            if float_shares is not None:
                float_shares = int(float_shares)
                logger.debug(f"{symbol}: float = {float_shares:,}")
                return {'float_shares': float_shares}
            else:
                logger.warning(f"{symbol}: floatShares not available from Yahoo Finance")
                return {'float_shares': None}

        result = self._fetch_with_retry(symbol, _fetch)
        if result is None:
            return None
        return result.get('float_shares')

    def get_float_batch(self, symbols: list, progress_interval: int = 50) -> Dict[str, Optional[int]]:
        """
        Get float for multiple symbols with progress logging and rate limit handling.

        Args:
            symbols: List of stock symbols
            progress_interval: Log progress every N symbols

        Returns:
            Dict mapping symbol -> float_shares (None if unavailable)
        """
        results = {}
        total = len(symbols)
        success_count = 0
        fail_count = 0

        logger.info(f"Fetching float data for {total} symbols...")

        for i, symbol in enumerate(symbols):
            float_shares = self.get_float(symbol)
            results[symbol] = float_shares

            if float_shares is not None:
                success_count += 1
            else:
                fail_count += 1

            if (i + 1) % progress_interval == 0 or (i + 1) == total:
                logger.info(
                    f"Fetching float: {i + 1}/{total} complete "
                    f"(success: {success_count}, failed: {fail_count})"
                )

            # Throttle requests to avoid rate limits
            if i < total - 1 and self.request_delay > 0:
                time.sleep(self.request_delay)

        logger.info(
            f"Float fetch complete: {success_count}/{total} successful, "
            f"{fail_count} failed/unavailable"
        )
        return results

    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get extended stock info (sector, country, float) from Yahoo Finance with retry.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with sector, country, float_shares (values may be None)
        """
        def _fetch():
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'sector': info.get('sector'),
                'country': info.get('country'),
                'float_shares': int(info['floatShares']) if info.get('floatShares') else None,
            }

        result = self._fetch_with_retry(symbol, _fetch)
        if result is None:
            return {'sector': None, 'country': None, 'float_shares': None}
        return result
