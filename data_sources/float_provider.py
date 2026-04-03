"""
Float data provider with multi-source fallback.

Sources (in order):
1. Yahoo Finance (yfinance) — primary, covers ~85% of stocks
2. FMP marketCap/price estimate — fallback for BATS-listed micro-caps
3. Volume-based estimation — last resort for stocks with no data anywhere

Results cached in DB with weekly refresh.
"""

import json
import logging
import os
import time
import urllib.request
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
        info = self.get_stock_info(symbol)
        return info.get('float_shares')

    def get_stock_info_batch(
        self, symbols: list, progress_interval: int = 50,
        volume_map: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Dict]:
        """
        Get float + sector + country for multiple symbols in a single pass.

        Makes ONE yfinance call per symbol (not two separate passes).
        Falls back to FMP + volume estimation for stocks Yahoo misses.

        Args:
            symbols: List of stock symbols
            progress_interval: Log progress every N symbols
            volume_map: Optional dict mapping symbol -> avg_daily_volume
                        (used for volume-based float estimation fallback)

        Returns:
            Dict mapping symbol -> {float_shares, sector, country}
        """
        volume_map = volume_map or {}
        results = {}
        total = len(symbols)
        success_count = 0
        fail_count = 0

        logger.info(f"Fetching stock info (float/sector/country) for {total} symbols...")

        for i, symbol in enumerate(symbols):
            avg_vol = volume_map.get(symbol, 0)
            info = self.get_stock_info(symbol, avg_daily_volume=avg_vol)
            results[symbol] = info

            if info.get('float_shares') is not None:
                success_count += 1
            else:
                fail_count += 1

            if (i + 1) % progress_interval == 0 or (i + 1) == total:
                logger.info(
                    f"Fetching stock info: {i + 1}/{total} complete "
                    f"(float success: {success_count}, failed: {fail_count})"
                )

            # Throttle requests to avoid rate limits
            if i < total - 1 and self.request_delay > 0:
                time.sleep(self.request_delay)

        logger.info(
            f"Stock info fetch complete: {success_count}/{total} with float, "
            f"{fail_count} without float"
        )
        return results

    def get_float_batch(self, symbols: list, progress_interval: int = 50) -> Dict[str, Optional[int]]:
        """
        Get float for multiple symbols with progress logging and rate limit handling.

        Args:
            symbols: List of stock symbols
            progress_interval: Log progress every N symbols

        Returns:
            Dict mapping symbol -> float_shares (None if unavailable)
        """
        batch = self.get_stock_info_batch(symbols, progress_interval)
        return {sym: info.get('float_shares') for sym, info in batch.items()}

    def get_stock_info(self, symbol: str, avg_daily_volume: int = 0) -> Dict:
        """
        Get extended stock info (sector, country, float) with multi-source fallback.

        Sources tried in order:
        1. Yahoo Finance — primary
        2. FMP marketCap/price — for BATS-listed stocks Yahoo misses
        3. Volume-based estimate — if avg_daily_volume provided, estimate float as 5x volume

        Args:
            symbol: Stock symbol

        Returns:
            Dict with sector, country, float_shares (values may be None)
        """
        # Source 1: Yahoo Finance
        def _fetch_yahoo():
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'sector': info.get('sector'),
                'country': info.get('country'),
                'float_shares': int(info['floatShares']) if info.get('floatShares') else None,
            }

        result = self._fetch_with_retry(symbol, _fetch_yahoo)
        if result is None:
            result = {'sector': None, 'country': None, 'float_shares': None}

        # If Yahoo got float, we're done
        if result.get('float_shares') is not None:
            return result

        # Source 2: FMP marketCap / price estimate
        fmp_float = self._get_float_from_fmp(symbol)
        if fmp_float is not None:
            result['float_shares'] = fmp_float
            logger.debug(f"{symbol}: float from FMP estimate = {fmp_float:,}")
            return result

        # Source 3: Volume-based estimate
        # Float is typically 3-10x avg daily volume for momentum small-caps.
        # Use 5x as conservative estimate. Only if volume is meaningful (>100K shares).
        if avg_daily_volume > 100_000:
            est_float = avg_daily_volume * 5
            result['float_shares'] = est_float
            logger.debug(f"{symbol}: float estimated from volume ({avg_daily_volume:,} × 5 = {est_float:,})")
            return result

        logger.debug(f"{symbol}: float unavailable from all sources")
        return result

    def _get_float_from_fmp(self, symbol: str) -> Optional[int]:
        """
        Estimate shares outstanding from FMP marketCap / price.

        Uses the /stable/profile endpoint (works on free tier).
        Returns estimated total shares (conservative proxy for float).

        Args:
            symbol: Stock symbol

        Returns:
            Estimated shares count, or None if unavailable
        """
        api_key = os.getenv('FMP_API_KEY', '')
        if not api_key:
            return None

        try:
            url = (
                f"https://financialmodelingprep.com/stable/profile"
                f"?symbol={symbol}&apikey={api_key}"
            )
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read())

            if isinstance(data, list) and data:
                d = data[0]
            elif isinstance(data, dict):
                d = data
            else:
                return None

            market_cap = d.get('marketCap', 0)
            price = d.get('price', 0)

            if price > 0 and market_cap > 0:
                shares_est = int(market_cap / price)
                return shares_est

        except Exception as e:
            logger.debug(f"{symbol}: FMP float estimate failed: {e}")

        return None
