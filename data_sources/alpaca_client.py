"""
Alpaca API client for market data and asset information.

Provides:
- All tradeable US assets
- Daily bars (previous close)
- 15-min intraday bars (volume profiles)
- Latest trade (SIP pre-market/real-time)
- News articles

All methods raise AlpacaAPIError on failure (no silent fallbacks).
Rate limit retry with exponential backoff built in.
"""

import logging
import time as time_mod
from datetime import datetime, timezone, timedelta, date
from typing import Dict, Optional, List, Callable, TypeVar
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import (
    StockLatestTradeRequest,
    StockLatestQuoteRequest,
    StockBarsRequest,
    StockLatestBarRequest,
    StockSnapshotRequest,
    NewsRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetAssetsRequest,
    GetCalendarRequest,
    GetOrderByIdRequest,
    MarketOrderRequest,
    LimitOrderRequest,
)
from alpaca.trading.enums import AssetClass, AssetStatus, OrderSide, TimeInForce, OrderClass, OrderType
from alpaca.common.exceptions import APIError
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

T = TypeVar('T')

DEFAULT_API_TIMEOUT = 90       # was 60; bumped after 2026-04-27 observed
                                # 30-90s Alpaca REST latency at 9:30-10:00 ET
                                # market-open congestion. 60s was inside the
                                # natural response distribution → spurious
                                # timeouts. 90s is at p98 of observed latency.
NEWS_API_TIMEOUT = 8           # premarket news fetch: best-effort call in
                                # the 9:31-9:35 tick path — fail-open callers,
                                # 9:33 lag pass is the retry. Normal response
                                # <1s (stupid-money telemetry); a hanging
                                # gateway must never stall the open.
MAX_RATE_LIMIT_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.0
MAX_TIMEOUT_RETRIES = 1        # 2026-04-27: retry once on FuturesTimeoutError —
                                # most timeouts during congestion are transient
                                # (same call 5s later succeeds). Asymmetry with
                                # 429 retries before this was unintentional.
                                # Why 1 and not 5 (like rate-limit): at 90s
                                # timeout, 5× retries × ~95s would block the
                                # cycle for ~8 minutes. 1 retry caps worst-case
                                # at ~185s — bad but recoverable next cycle.
TIMEOUT_RETRY_BACKOFF_SECONDS = 5.0


class AlpacaAPIError(Exception):
    """Exception raised for Alpaca API errors."""
    pass


class AlpacaAPITimeoutError(AlpacaAPIError):
    """Exception raised when Alpaca API call times out."""
    pass


class AlpacaClient:
    """
    Client for Alpaca market data and trading API using alpaca-py SDK.

    Provides market data access for universe building and real-time scanning.
    All methods include timeout protection and rate limit retry.
    """

    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        """
        Initialize AlpacaClient.

        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            paper: If True, use paper trading endpoint. Default True for safety.

        Raises:
            AlpacaAPIError: If API credentials are missing
        """
        if not api_key:
            raise AlpacaAPIError("ALPACA_API_KEY required")
        if not api_secret:
            raise AlpacaAPIError("ALPACA_API_SECRET required")

        self.data_client = StockHistoricalDataClient(api_key, api_secret)
        self._paper = paper
        self.trading_client = TradingClient(api_key, api_secret, paper=self._paper)
        self.news_client = NewsClient(api_key, api_secret)
        self._api_timeout = DEFAULT_API_TIMEOUT

        logger.info(f"AlpacaClient initialized (paper={self._paper})")

    @property
    def is_paper(self) -> bool:
        """Whether this client is connected to Alpaca paper trading."""
        return self._paper

    def _call_with_timeout(self, func: Callable[[], T], operation: str,
                           timeout: Optional[float] = None,
                           timeout_retries: Optional[int] = None,
                           rate_limit_retries: Optional[int] = None) -> T:
        """
        Execute API call with timeout and rate limit retry.

        Args:
            func: Callable to execute
            operation: Description for logging
            timeout: Per-call override of self._api_timeout (seconds).
                Use a SHORT value for best-effort calls in latency-critical
                windows (e.g. the premarket news fetch: a hanging news
                gateway must cost seconds, not the 90s x retries default,
                because callers fail-open and trades must never wait).
            timeout_retries: Per-call override of MAX_TIMEOUT_RETRIES
                (0 = raise on first timeout).
            rate_limit_retries: Per-call override of MAX_RATE_LIMIT_RETRIES
                (0 = raise on first 429 — no backoff sleeps; the default
                ladder sleeps up to ~31s, unacceptable for best-effort
                calls in the entry window).

        Returns:
            Result of the function call

        Raises:
            AlpacaAPITimeoutError: If call times out
            AlpacaAPIError: If call fails or rate limit exhausted
        """
        backoff = INITIAL_BACKOFF_SECONDS
        last_exception = None
        timeout_attempts = 0
        eff_timeout = timeout if timeout is not None else self._api_timeout
        eff_retries = (timeout_retries if timeout_retries is not None
                       else MAX_TIMEOUT_RETRIES)
        eff_rl_retries = (rate_limit_retries if rate_limit_retries is not None
                          else MAX_RATE_LIMIT_RETRIES)

        for attempt in range(eff_rl_retries + 1):
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func)
                    return future.result(timeout=eff_timeout)
            except FuturesTimeoutError:
                # 2026-04-27: retry once on timeout. Alpaca REST occasionally
                # has 60-90s response latency during 9:30-10:00 ET congestion.
                # A second attempt 5s later usually succeeds. Without this,
                # transient slowness becomes a hard error and the scanner
                # cycle skips an entire minute of work.
                if timeout_attempts < eff_retries:
                    timeout_attempts += 1
                    logger.warning(
                        f"API call timed out after {eff_timeout}s on {operation}, "
                        f"retrying in {TIMEOUT_RETRY_BACKOFF_SECONDS}s "
                        f"(attempt {timeout_attempts}/{eff_retries})"
                    )
                    time_mod.sleep(TIMEOUT_RETRY_BACKOFF_SECONDS)
                    continue
                logger.error(
                    f"API call timed out after {eff_timeout}s + "
                    f"{eff_retries} retry: {operation}"
                )
                raise AlpacaAPITimeoutError(
                    f"API call timed out ({eff_timeout}s × "
                    f"{1 + eff_retries} attempts): {operation}"
                )
            except Exception as e:
                error_str = str(e).lower()
                if '429' in str(e) or 'rate limit' in error_str or 'too many requests' in error_str:
                    last_exception = e
                    if attempt < eff_rl_retries:
                        logger.warning(
                            f"Rate limited on {operation} (attempt {attempt + 1}/{eff_rl_retries + 1}), "
                            f"retrying in {backoff:.1f}s..."
                        )
                        time_mod.sleep(backoff)
                        backoff *= 2
                        continue
                    else:
                        logger.error(f"Rate limit exhausted after {eff_rl_retries + 1} attempts: {operation}")
                        raise AlpacaAPIError(f"Rate limit exhausted: {operation}") from e
                raise

        if last_exception:
            raise AlpacaAPIError(f"API call failed: {operation}") from last_exception
        raise AlpacaAPIError(f"API call failed unexpectedly: {operation}")

    @staticmethod
    def _to_dict(response) -> dict:
        """
        Normalize alpaca-py SDK response to a plain dict.

        The SDK's BarSet/TradeSet/QuoteSet objects have a broken __contains__
        that doesn't match __getitem__. Using .data gives us a real dict.
        """
        if hasattr(response, 'data'):
            return response.data
        if isinstance(response, dict):
            return response
        logger.warning(f"Unexpected response type: {type(response)}, returning as-is")
        return response

    @staticmethod
    def _log_order_op_failure(op_name: str, symbol: str, e: Exception) -> None:
        """Log an order-operation failure at the right level.

        Background (added 2026-05-13 after 2 days of FC noise complaints):
        Order-op failures (submit_*, close_position) were ALL logged at
        ERROR. But retry-wrapped callers (ORB FC `_close_position_with_held_qty_retry`,
        StopMonitor `_submit_with_held_qty_retry`) handle the well-known
        transient races gracefully via backoff. Each retry attempt
        produced an ERROR line — operator inbox spammed even on clean
        recovery (GLWG 5/11, VG+BMNZ 5/12: 4 ERROR lines, zero actual
        leaks).

        This helper splits by error content:
          - WARNING: known-transient (40310000 / held_for_orders,
            429 rate-limit, 5xx) — caller may retry; the API-layer log
            is informational
          - ERROR:   everything else (auth, network, invalid order, etc.)
            — real operator-attention failures

        Net: ERROR channel becomes a real-failures-only signal.

        Note: this only changes the log level. The exception still
        propagates (caller wraps and re-raises as AlpacaAPIError) — error
        handling behavior is unchanged.
        """
        msg = str(e)
        msg_l = msg.lower()
        is_transient = (
            '40310000' in msg
            or 'insufficient qty available' in msg_l
            or 'rate limit' in msg_l
            or 'too many requests' in msg_l
            or any(p in msg_l for p in (
                'internal server error', 'bad gateway',
                'service unavailable', 'gateway timeout',
            ))
        )
        if is_transient:
            logger.warning(
                f"{op_name} {symbol} transient error "
                f"(caller may retry via backoff): {e}"
            )
        else:
            logger.error(f"Failed to {op_name} for {symbol}: {e}")

    # =========================================================================
    # Assets
    # =========================================================================

    # Non-common-stock keywords in asset names
    _EXCLUDED_NAME_KEYWORDS = [
        'Warrant', 'Rights', 'Preferred',
    ]

    # Non-common-stock patterns in symbols
    _EXCLUDED_SYMBOL_PATTERNS = ['.PR']

    # Leveraged/inverse ETF keywords in asset names — synthetic instruments,
    # not real stocks. Bull flags on these are pattern artifacts (31% WR, -$5K/15mo).
    _LEVERAGED_ETF_NAME_KEYWORDS = [
        '2X', '3X', '-2X', '-3X',
        'LEVERAGED', 'INVERSE', 'ULTRA',
        'DAILY BULL', 'DAILY BEAR',
        'DIREXION', 'PROSHARES', 'GRANITESHARES',
    ]

    # Known leveraged/inverse ETF symbols — manually maintained for tickers
    # whose names don't always contain the keywords above.
    _LEVERAGED_ETF_SYMBOLS = {
        # Single-stock leveraged
        'MSTU', 'MSTX', 'AMZU', 'AMZZ', 'AMDL', 'NVDX', 'AAPU', 'AAPB', 'AAPX',
        'TSLT', 'BABX', 'NVDU', 'NVDD', 'GOOX', 'GOOGL', 'MSFU', 'MSFD',
        'TSLL', 'TSLZ', 'CONL', 'CONY', 'NFLX', 'APTV',
        # Index leveraged
        'TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'UDOW', 'SDOW', 'QLD', 'QID',
        'SSO', 'SDS', 'SPXS', 'SPXL',
        # Sector leveraged
        'TNA', 'TZA', 'LABU', 'LABD', 'SOXL', 'SOXS',
        'FNGU', 'FNGD', 'BULZ', 'BERZ', 'NAIL', 'CURE', 'DFEN',
        'FAS', 'FAZ', 'ERX', 'ERY', 'NUGT', 'DUST', 'JNUG', 'JDST',
        'DRN', 'DRV', 'TECL', 'TECS', 'DPST', 'PILL', 'RETL',
        # Leveraged commodity/crypto
        'BITX', 'BITU', 'MARA', 'RIOT',
        # Leveraged with "CR" suffix (credit-based)
        'LFCR',
    }

    @classmethod
    def _is_common_stock(cls, symbol: str, name: str) -> bool:
        """
        Filter out warrants, units, preferred shares, rights, SPACs,
        and leveraged/inverse ETFs.

        Only keeps common stocks suitable for momentum day trading.
        """
        name_upper = (name or '').upper()

        # Preferred shares: symbol contains .PR (e.g., BAC.PRE)
        if '.PR' in symbol:
            return False

        # Check name for non-stock keywords
        for keyword in cls._EXCLUDED_NAME_KEYWORDS:
            if keyword.upper() in name_upper:
                return False

        # Units: symbol ends with 'U' AND name ends with 'Unit' or 'Units'
        if symbol.endswith('U') and (name_upper.endswith('UNIT') or name_upper.endswith('UNITS')):
            return False

        # Leveraged/inverse ETFs: explicit symbol list
        if symbol in cls._LEVERAGED_ETF_SYMBOLS:
            return False

        # Leveraged/inverse ETFs: name keywords
        for keyword in cls._LEVERAGED_ETF_NAME_KEYWORDS:
            if keyword in name_upper:
                return False

        return True

    def get_all_tradeable_assets(self) -> List[Dict]:
        """
        Get all tradeable US common stock assets.

        Filters out warrants, units, preferred shares, and rights.

        Returns:
            List of dicts with symbol, name, exchange info

        Raises:
            AlpacaAPIError: If API call fails
        """
        try:
            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE
            )
            assets = self._call_with_timeout(
                lambda: self.trading_client.get_all_assets(request),
                "get_all_tradeable_assets"
            )

            total_tradeable = 0
            tradeable = []
            excluded_count = 0

            for asset in assets:
                if not asset.tradable:
                    continue
                total_tradeable += 1

                if not self._is_common_stock(asset.symbol, asset.name or ''):
                    excluded_count += 1
                    continue

                tradeable.append({
                    'symbol': asset.symbol,
                    'company_name': asset.name or '',
                    'exchange': asset.exchange.value if asset.exchange else '',
                    'marginable': bool(getattr(asset, 'marginable', False)),
                })

            logger.info(
                f"Fetched {len(tradeable)} common stocks "
                f"(excluded {excluded_count} warrants/preferred/units/rights "
                f"from {total_tradeable} tradeable)"
            )
            return tradeable
        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get tradeable assets: {e}")
            raise AlpacaAPIError(f"Failed to get tradeable assets: {e}")

    # =========================================================================
    # Daily Bars
    # =========================================================================

    def get_daily_bars(self, symbols: List[str], days: int = 20) -> Dict[str, Dict]:
        """
        Get recent daily bars for multiple symbols.

        Args:
            symbols: List of stock symbols
            days: Number of trading days to fetch

        Returns:
            Dict mapping symbol -> latest bar dict {close, volume, timestamp}

        Raises:
            AlpacaAPIError: If API call fails
        """
        if not symbols:
            return {}

        try:
            start = datetime.now(timezone.utc) - timedelta(days=days * 2)  # Extra buffer for weekends
            results = {}

            # Process in chunks to avoid API limits
            chunk_size = 200
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i:i + chunk_size]
                request = StockBarsRequest(
                    symbol_or_symbols=chunk,
                    timeframe=TimeFrame.Day,
                    start=start,
                    feed=DataFeed.SIP
                )
                bars_raw = self._call_with_timeout(
                    lambda req=request: self.data_client.get_stock_bars(req),
                    f"get_daily_bars(chunk {i // chunk_size + 1})"
                )
                bars = self._to_dict(bars_raw)

                for symbol in chunk:
                    if symbol in bars and len(bars[symbol]) > 0:
                        latest = bars[symbol][-1]
                        avg_vol = sum(int(b.volume) for b in bars[symbol]) / len(bars[symbol])
                        results[symbol] = {
                            'close': float(latest.close),
                            'volume': int(avg_vol),
                            'timestamp': latest.timestamp
                        }

                logger.info(f"Daily bars progress: {min(i + chunk_size, len(symbols))}/{len(symbols)}")

            logger.info(f"Fetched daily bars for {len(results)}/{len(symbols)} symbols")
            return results

        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get daily bars: {e}")
            raise AlpacaAPIError(f"Failed to get daily bars: {e}")

    def get_daily_bars_range(
        self, symbols: List[str], start: date, end: date
    ) -> Dict[str, List[Dict]]:
        """
        Get daily bars for multiple symbols over a date range.

        Fetches daily OHLCV bars chunked by 200 symbols (same pattern
        as get_daily_bars). Used for batch scanning of intraday movers.

        Args:
            symbols: List of stock symbols
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            Dict mapping symbol -> list of bar dicts
            [{date, open, high, low, close, volume}, ...]

        Raises:
            AlpacaAPIError: If API call fails
        """
        if not symbols:
            return {}

        try:
            start_dt = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
            end_dt = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc)
            results: Dict[str, List[Dict]] = {}

            chunk_size = 200
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i:i + chunk_size]
                request = StockBarsRequest(
                    symbol_or_symbols=chunk,
                    timeframe=TimeFrame.Day,
                    start=start_dt,
                    end=end_dt,
                    feed=DataFeed.SIP,
                )
                bars_raw = self._call_with_timeout(
                    lambda req=request: self.data_client.get_stock_bars(req),
                    f"get_daily_bars_range(chunk {i // chunk_size + 1})"
                )
                bars = self._to_dict(bars_raw)

                for symbol in chunk:
                    if symbol in bars and len(bars[symbol]) > 0:
                        symbol_bars = []
                        for bar in bars[symbol]:
                            symbol_bars.append({
                                'date': bar.timestamp.date() if hasattr(bar.timestamp, 'date') else bar.timestamp,
                                'open': float(bar.open),
                                'high': float(bar.high),
                                'low': float(bar.low),
                                'close': float(bar.close),
                                'volume': int(bar.volume),
                            })
                        results[symbol] = symbol_bars

                logger.info(
                    f"Daily bars range progress: "
                    f"{min(i + chunk_size, len(symbols))}/{len(symbols)} symbols"
                )

            logger.info(
                f"Fetched daily bars range for {len(results)}/{len(symbols)} symbols "
                f"({start} to {end})"
            )
            return results

        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get daily bars range: {e}")
            raise AlpacaAPIError(f"Failed to get daily bars range: {e}")

    # =========================================================================
    # Intraday Bars (15-min for volume profiles)
    # =========================================================================

    def get_intraday_bars(self, symbol: str, days: int = 50) -> pd.DataFrame:
        """
        Get 15-minute intraday bars for volume profile calculation.

        Args:
            symbol: Stock symbol
            days: Number of calendar days to look back

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            AlpacaAPIError: If API call fails
        """
        try:
            start = datetime.now(timezone.utc) - timedelta(days=days * 2)  # Buffer for weekends
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(15, TimeFrameUnit.Minute),
                start=start,
                feed=DataFeed.SIP
            )
            bars_raw = self._call_with_timeout(
                lambda: self.data_client.get_stock_bars(request),
                f"get_intraday_bars({symbol})"
            )
            bars = self._to_dict(bars_raw)

            if symbol not in bars or len(bars[symbol]) == 0:
                logger.warning(f"No intraday bars returned for {symbol}")
                return pd.DataFrame()

            records = []
            for bar in bars[symbol]:
                records.append({
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume)
                })

            return pd.DataFrame(records)

        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get intraday bars for {symbol}: {e}")
            raise AlpacaAPIError(f"Failed to get intraday bars for {symbol}: {e}")

    # =========================================================================
    # Real-Time Data
    # =========================================================================

    def get_latest_trades(self, symbols: List[str], feed: DataFeed = DataFeed.SIP) -> Dict[str, Dict]:
        """
        Get latest trade for multiple symbols (SIP for pre-market).

        Args:
            symbols: List of stock symbols
            feed: Data feed (SIP for pre-market data)

        Returns:
            Dict mapping symbol -> {price, size, timestamp}

        Raises:
            AlpacaAPIError: If API call fails
        """
        if not symbols:
            return {}

        try:
            results = {}
            chunk_size = 200
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i:i + chunk_size]
                request = StockLatestTradeRequest(symbol_or_symbols=chunk, feed=feed)
                trades_raw = self._call_with_timeout(
                    lambda req=request: self.data_client.get_stock_latest_trade(req),
                    f"get_latest_trades(chunk {i // chunk_size + 1})"
                )
                trades = self._to_dict(trades_raw)
                for symbol in chunk:
                    if symbol in trades:
                        trade = trades[symbol]
                        results[symbol] = {
                            'price': float(trade.price) if trade.price else 0,
                            'size': int(trade.size) if trade.size else 0,
                            'timestamp': trade.timestamp.isoformat() if trade.timestamp else None
                        }

            logger.debug(f"Fetched latest trades for {len(results)}/{len(symbols)} symbols")
            return results

        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get latest trades: {e}")
            raise AlpacaAPIError(f"Failed to get latest trades: {e}")

    def get_snapshots(self, symbols: List[str], feed: DataFeed = DataFeed.SIP) -> Dict[str, Dict]:
        """
        Get stock snapshots (today's daily bar open/high/low/close + latest trade).

        Args:
            symbols: List of stock symbols
            feed: Data feed (SIP for consolidated data)

        Returns:
            Dict mapping symbol -> {open, high, low, close, volume, latest_price}
        """
        if not symbols:
            return {}

        try:
            results = {}
            chunk_size = 200
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i:i + chunk_size]
                request = StockSnapshotRequest(symbol_or_symbols=chunk, feed=feed)
                snapshots_raw = self._call_with_timeout(
                    lambda req=request: self.data_client.get_stock_snapshot(req),
                    f"get_snapshots(chunk {i // chunk_size + 1})"
                )
                snapshots = self._to_dict(snapshots_raw)
                for symbol in chunk:
                    if symbol in snapshots:
                        snap = snapshots[symbol]
                        bar = snap.daily_bar
                        result = {}
                        if bar:
                            result['open'] = float(bar.open) if bar.open else 0
                            result['high'] = float(bar.high) if bar.high else 0
                            result['low'] = float(bar.low) if bar.low else 0
                            result['close'] = float(bar.close) if bar.close else 0
                            result['volume'] = int(bar.volume) if bar.volume else 0
                        prev_bar = snap.previous_daily_bar
                        if prev_bar:
                            result['prev_close'] = float(prev_bar.close) if prev_bar.close else 0
                            result['prev_volume'] = int(prev_bar.volume) if prev_bar.volume else 0
                        if snap.latest_trade:
                            result['latest_price'] = float(snap.latest_trade.price) if snap.latest_trade.price else 0
                        if snap.latest_quote:
                            q = snap.latest_quote
                            result['bid_price'] = float(q.bid_price) if q.bid_price else 0
                            result['ask_price'] = float(q.ask_price) if q.ask_price else 0
                            result['bid_size'] = int(q.bid_size) if q.bid_size else 0
                            result['ask_size'] = int(q.ask_size) if q.ask_size else 0
                        if result:
                            results[symbol] = result

            logger.debug(f"Fetched snapshots for {len(results)}/{len(symbols)} symbols")
            return results

        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get snapshots: {e}")
            raise AlpacaAPIError(f"Failed to get snapshots: {e}")

    def get_latest_quote(self, symbol: str, feed: DataFeed = DataFeed.SIP) -> Dict:
        """
        Get latest NBBO quote (bid/ask) for a single symbol.

        Used by StopMonitor for spread-based exit pricing — sets the limit
        sell price relative to the current bid/ask instead of a fixed offset.

        Args:
            symbol: Stock symbol
            feed: Data feed (SIP for consolidated NBBO)

        Returns:
            Dict with bid_price, ask_price, bid_size, ask_size, timestamp

        Raises:
            AlpacaAPIError: If API call fails or no quote available
        """
        try:
            request = StockLatestQuoteRequest(
                symbol_or_symbols=symbol, feed=feed
            )
            quote_raw = self._call_with_timeout(
                lambda: self.data_client.get_stock_latest_quote(request),
                f"get_latest_quote({symbol})"
            )
            quotes = self._to_dict(quote_raw)
            if symbol not in quotes:
                raise AlpacaAPIError(f"No quote returned for {symbol}")

            quote = quotes[symbol]
            result = {
                'bid_price': float(quote.bid_price) if quote.bid_price else 0.0,
                'ask_price': float(quote.ask_price) if quote.ask_price else 0.0,
                'bid_size': int(quote.bid_size) if quote.bid_size else 0,
                'ask_size': int(quote.ask_size) if quote.ask_size else 0,
                'timestamp': quote.timestamp.isoformat() if quote.timestamp else None,
            }
            logger.debug(
                f"{symbol}: quote bid=${result['bid_price']:.2f} "
                f"ask=${result['ask_price']:.2f} "
                f"spread=${result['ask_price'] - result['bid_price']:.3f}"
            )
            return result

        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get latest quote for {symbol}: {e}")
            raise AlpacaAPIError(f"Failed to get latest quote for {symbol}: {e}")

    def get_current_bars(self, symbols: List[str], feed: DataFeed = DataFeed.SIP) -> Dict[str, Dict]:
        """
        Get the day's RUNNING high/low/open/close/volume per symbol via
        1-min RTH bars from today's market open (09:30 ET) to now.

        2026-05-08: switched from latest-15-min-bar to day-running-1-min
        aggregation so live's qualification matches BASELINE BT exactly.
        BT iterates 1-min bars from 09:30 ET tracking running max/min;
        without this change, live's 15-min-bar polling lagged BT by
        10-15 minutes (live couldn't qualify a stock until the first
        09:30-09:45 RTH 15-min bar completed at 09:46 ET, while BT can
        qualify as early as 09:35 ET after a 5-bar seed). That window
        is where many of the day's strongest setups fire, so missing it
        was a structural drift.

        Premarket bars are excluded by the start=today_open_utc query —
        Alpaca's 1-min bar feed includes extended hours, but we use
        only RTH so the running stats match baseline BT.

        Args:
            symbols: List of stock symbols
            feed: Data feed (SIP for live trading)

        Returns:
            Dict mapping symbol -> {open, high, low, close, volume, timestamp}.
            Schema is unchanged from prior 15-min behavior — fields now
            mean: open=first 1-min bar's open at 09:30, high=day's running
            high since 09:30, low=day's running low since 09:30, close=
            latest minute's close, volume=cumulative volume since 09:30,
            timestamp=latest minute bar's timestamp. Symbol absent from
            result means no RTH bars yet today.

        Raises:
            AlpacaAPIError: If API call fails
        """
        if not symbols:
            return {}

        # Compute today's 09:30 ET in UTC
        ET = pytz.timezone('US/Eastern')
        now_et = datetime.now(ET)
        today_open_et = ET.localize(datetime(
            now_et.year, now_et.month, now_et.day, 9, 30, 0
        ))
        today_open_utc = today_open_et.astimezone(timezone.utc)
        # Pre-market guard: if the scanner happens to call this before
        # market open, fall back to the prior 15-min behavior so we
        # don't return empty for every symbol.
        if datetime.now(timezone.utc) < today_open_utc:
            today_open_utc = datetime.now(timezone.utc) - timedelta(minutes=45)

        try:
            results = {}
            chunk_size = 200
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i:i + chunk_size]
                request = StockBarsRequest(
                    symbol_or_symbols=chunk,
                    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    start=today_open_utc,
                    feed=feed,
                )
                bars_raw = self._call_with_timeout(
                    lambda req=request: self.data_client.get_stock_bars(req),
                    f"get_current_bars(chunk {i // chunk_size + 1})"
                )
                bars = self._to_dict(bars_raw)
                for symbol in chunk:
                    syms_bars = bars.get(symbol) or []
                    if not syms_bars:
                        continue
                    # Aggregate 1-min bars to day-running stats
                    day_high = max(float(b.high) for b in syms_bars)
                    day_low = min(float(b.low) for b in syms_bars)
                    day_volume = sum(int(b.volume) for b in syms_bars)
                    first_open = float(syms_bars[0].open)
                    latest = syms_bars[-1]
                    results[symbol] = {
                        'open': first_open,
                        'high': day_high,
                        'low': day_low,
                        'close': float(latest.close),
                        'volume': day_volume,
                        'timestamp': latest.timestamp,
                    }

            logger.debug(
                f"Fetched day-running 1-min bars (since 09:30 ET) for "
                f"{len(results)}/{len(symbols)} symbols"
            )
            return results

        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get current bars: {e}")
            raise AlpacaAPIError(f"Failed to get current bars: {e}")

    def get_premarket_extremes(
        self, symbols: List[str], trade_date: date,
        feed: DataFeed = DataFeed.SIP,
    ) -> Dict[str, tuple]:
        """Batch-fetch premarket high/low (04:00-09:30 ET) for many symbols.

        Returns {symbol: (pm_high, pm_low)} for symbols that traded in
        premarket. Symbols with no premarket activity are absent from the
        return dict (caller should default to None).

        Used by live scanner at startup to seed _day_highs/_day_lows so
        intraday range_pct qualification matches BT-with-premarket. Closes
        the BT-LIVE drift documented in 5/5 INTT (gapped down 13% in PM,
        ranged 17.7% during RTH; live caught it via premarket-derived
        range_pct, BT missed without premarket).
        """
        if not symbols:
            return {}
        ET = pytz.timezone('US/Eastern')
        pm_start_et = ET.localize(datetime(
            trade_date.year, trade_date.month, trade_date.day, 4, 0, 0))
        pm_end_et = ET.localize(datetime(
            trade_date.year, trade_date.month, trade_date.day, 9, 30, 0))
        pm_start = pm_start_et.astimezone(timezone.utc)
        pm_end = pm_end_et.astimezone(timezone.utc)
        results: Dict[str, tuple] = {}
        chunk_size = 200
        try:
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i:i + chunk_size]
                request = StockBarsRequest(
                    symbol_or_symbols=chunk,
                    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    start=pm_start,
                    end=pm_end,
                    feed=feed,
                )
                bars_raw = self._call_with_timeout(
                    lambda req=request: self.data_client.get_stock_bars(req),
                    f"get_premarket_extremes(chunk {i // chunk_size + 1})"
                )
                bars = self._to_dict(bars_raw)
                for symbol in chunk:
                    syms_bars = bars.get(symbol) or []
                    if not syms_bars:
                        continue
                    pm_high = max(float(b.high) for b in syms_bars)
                    pm_low = min(float(b.low) for b in syms_bars)
                    results[symbol] = (pm_high, pm_low)
        except Exception as e:
            logger.error(
                f"get_premarket_extremes failed (chunk i={i}): {e}"
            )
            return results  # partial results are still useful
        logger.info(
            f"Premarket extremes fetched for {len(results)}/{len(symbols)} "
            f"symbols on {trade_date}"
        )
        return results

    # =========================================================================
    # News
    # =========================================================================

    def get_news(self, symbol: str, limit: int = 5) -> List[Dict]:
        """
        Get recent news articles for a symbol via Alpaca News API.

        Args:
            symbol: Stock symbol
            limit: Maximum number of articles to fetch

        Returns:
            List of dicts with headline, summary, source, created_at, url

        Raises:
            AlpacaAPIError: If API call fails
        """
        try:
            request = NewsRequest(symbols=symbol, limit=limit, sort='desc')
            news_set = self._call_with_timeout(
                lambda: self.news_client.get_news(request),
                f"get_news({symbol})"
            )
        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"News API call failed for {symbol}: {e}")
            raise AlpacaAPIError(f"News API call failed for {symbol}: {e}")

        try:
            articles = []
            # NewsSet.data contains the list of news articles keyed by 'news'
            news_data = news_set.data if hasattr(news_set, 'data') else {}
            news_list = news_data.get('news', []) if isinstance(news_data, dict) else []

            for article in news_list:
                articles.append({
                    'headline': getattr(article, 'headline', ''),
                    'summary': getattr(article, 'summary', ''),
                    'source': getattr(article, 'source', ''),
                    'created_at': str(getattr(article, 'created_at', '')),
                    'url': getattr(article, 'url', ''),
                })

            logger.debug(f"Fetched {len(articles)} news articles for {symbol}")
            return articles

        except Exception as e:
            logger.warning(f"Failed to parse news response for {symbol}: {e}")
            return []

    def get_premarket_news_multi(self, symbols: List[str]) -> Dict[str, Dict]:
        """Batched pre-market news for the news-gated PM sizing mult
        (2026-07-10 ship, research/orb_news_catalyst_jul2026.md).

        Window: previous CALENDAR day 15:00 ET → now. Matches the BT
        backfill window (research/scripts/orb_news_backfill.py) so live
        has_news and BT has_news are the same measurement by construction.
        Weekend/holiday gaps are intentional: the BT evidence was built
        with the same prev-calendar-day window, and a Monday candidate's
        Fri/Sat/Sun news lands in the [Sun 15:00 ET, Mon 09:3x] window
        only if fresh — stale Friday news correctly doesn't count in
        either system.

        Returns dict symbol -> {'n_articles': int, 'headline': str}.
        Raises on API failure (callers implement fail-open + poisoning,
        mirroring the PM dollar-volume fetch).
        """
        if not symbols:
            return {}
        import pytz
        et_tz = pytz.timezone('US/Eastern')
        now_utc = datetime.now(timezone.utc)
        now_et = now_utc.astimezone(et_tz)
        start_et = (now_et - timedelta(days=1)).replace(
            hour=15, minute=0, second=0, microsecond=0)
        result: Dict[str, Dict] = {
            s: {'n_articles': 0, 'headline': ''} for s in symbols}
        page_token = None
        for _page in range(6):  # 6 pages × 50 = 300 articles, ample
            request = NewsRequest(
                symbols=','.join(symbols),
                start=start_et.astimezone(timezone.utc),
                end=now_utc,
                limit=50, sort='desc', page_token=page_token)
            # Short timeout, NO retries: the news gateway is a separate,
            # observably-flakier backend (stupid-money May'26 incident) and
            # this call sits in the 9:31-9:35 tick path. Callers fail-open
            # (no boost) and the 9:33 lag pass IS the retry — a hanging
            # gateway must cost ~8s once, never 90s x retries.
            news_set = self._call_with_timeout(
                lambda: self.news_client.get_news(request),
                f"get_premarket_news_multi({len(symbols)} symbols)",
                timeout=NEWS_API_TIMEOUT, timeout_retries=0,
                rate_limit_retries=0)
            news_data = news_set.data if hasattr(news_set, 'data') else {}
            news_list = news_data.get('news', []) \
                if isinstance(news_data, dict) else []
            for article in news_list:
                for sym in (getattr(article, 'symbols', None) or []):
                    if sym in result:
                        result[sym]['n_articles'] += 1
                        if not result[sym]['headline']:
                            result[sym]['headline'] = str(
                                getattr(article, 'headline', '') or '')[:200]
            page_token = getattr(news_set, 'next_page_token', None)
            if not page_token:
                break
        n_newsy = sum(1 for v in result.values() if v['n_articles'] > 0)
        logger.debug(
            f"Premarket news: {n_newsy}/{len(symbols)} symbols with articles")
        return result

    # =========================================================================
    # 1-Minute Bars (for pattern detection)
    # =========================================================================

    def get_1min_bars(self, symbol: str, lookback_minutes: int = 30) -> pd.DataFrame:
        """
        Get 1-minute bars for pattern detection.

        Returns all bars from the API, including the current in-progress bar.
        The caller (BullFlagDetector) is responsible for dropping the last
        (in-progress) bar via bars.iloc[:-1].

        Args:
            symbol: Stock symbol
            lookback_minutes: Number of minutes to look back

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            AlpacaAPIError: If API call fails
        """
        try:
            start = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes + 5)

            # Clamp to today's market open (09:30 ET) to exclude premarket bars.
            # Premarket candles have unreliable volume and price levels that
            # contaminate pattern detection, especially in the first 10 minutes.
            import pytz
            et_tz = pytz.timezone('US/Eastern')
            now_utc = datetime.now(timezone.utc)
            market_open_et = now_utc.astimezone(et_tz).replace(
                hour=9, minute=30, second=0, microsecond=0)
            market_open_utc = market_open_et.astimezone(timezone.utc)
            if start < market_open_utc:
                start = market_open_utc
                logger.debug(
                    f"Clamped 1-min bar start to market open {market_open_utc.isoformat()}"
                )

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=start,
                feed=DataFeed.SIP,
            )
            bars_raw = self._call_with_timeout(
                lambda: self.data_client.get_stock_bars(request),
                f"get_1min_bars({symbol})"
            )
            bars = self._to_dict(bars_raw)

            if symbol not in bars or len(bars[symbol]) == 0:
                logger.warning(f"No 1-min bars returned for {symbol}")
                return pd.DataFrame()

            records = []
            for bar in bars[symbol]:
                records.append({
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume),
                })

            logger.debug(f"Fetched {len(records)} 1-min bars for {symbol}")
            return pd.DataFrame(records)

        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get 1-min bars for {symbol}: {e}")
            raise AlpacaAPIError(f"Failed to get 1-min bars for {symbol}: {e}")

    def get_1min_bars_multi(self, symbols: list, lookback_minutes: int = 30) -> dict:
        """
        Fetch 1-minute bars for multiple symbols in a single API call.

        Returns dict mapping symbol -> DataFrame. Missing symbols have empty DataFrame.
        Single API call vs N sequential calls = N× faster.
        """
        if not symbols:
            return {}
        try:
            start = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes + 5)
            import pytz
            et_tz = pytz.timezone('US/Eastern')
            now_utc = datetime.now(timezone.utc)
            market_open_et = now_utc.astimezone(et_tz).replace(
                hour=9, minute=30, second=0, microsecond=0)
            market_open_utc = market_open_et.astimezone(timezone.utc)
            if start < market_open_utc:
                start = market_open_utc

            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=start,
                feed=DataFeed.SIP,
            )
            bars_raw = self._call_with_timeout(
                lambda: self.data_client.get_stock_bars(request),
                f"get_1min_bars_multi({len(symbols)} symbols)"
            )
            bars = self._to_dict(bars_raw)

            result = {}
            for sym in symbols:
                if sym not in bars or len(bars[sym]) == 0:
                    result[sym] = pd.DataFrame()
                    continue
                records = []
                for bar in bars[sym]:
                    records.append({
                        'timestamp': bar.timestamp,
                        'open': float(bar.open),
                        'high': float(bar.high),
                        'low': float(bar.low),
                        'close': float(bar.close),
                        'volume': int(bar.volume),
                    })
                result[sym] = pd.DataFrame(records)
            logger.debug(f"Fetched 1-min bars for {len(result)} symbols in single call")
            return result
        except Exception as e:
            logger.error(f"Failed to get multi-symbol 1-min bars: {e}")
            return {}

    def get_premarket_1min_bars_multi(self, symbols: list) -> dict:
        """Fetch TODAY's premarket (4:00-9:29 ET) 1-min bars, batched.

        2026-07-06: get_1min_bars_multi deliberately clamps start to the
        9:30 session open (its callers compute opening ranges and must not
        see premarket bars) — which silently starved the PM dollar-volume
        sizing mult (pm_dollar_vol=None -> fail-open 1.0 on every trade).
        This method is the premarket-specific twin: same batching, no clamp.

        Returns dict symbol -> DataFrame (empty for symbols with no
        premarket prints).
        """
        if not symbols:
            return {}
        try:
            import pytz
            et_tz = pytz.timezone('US/Eastern')
            now_et = datetime.now(timezone.utc).astimezone(et_tz)
            pm_start = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
            pm_end = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=pm_start.astimezone(timezone.utc),
                end=pm_end.astimezone(timezone.utc),
                feed=DataFeed.SIP,
            )
            bars_raw = self._call_with_timeout(
                lambda: self.data_client.get_stock_bars(request),
                f"get_premarket_1min_bars_multi({len(symbols)} symbols)"
            )
            bars = self._to_dict(bars_raw)
            result = {}
            for sym in symbols:
                if sym not in bars or len(bars[sym]) == 0:
                    result[sym] = pd.DataFrame()
                    continue
                records = [{
                    'timestamp': bar.timestamp,
                    'open': float(bar.open), 'high': float(bar.high),
                    'low': float(bar.low), 'close': float(bar.close),
                    'volume': int(bar.volume),
                } for bar in bars[sym]]
                result[sym] = pd.DataFrame(records)
            logger.debug(
                f"Fetched premarket bars for {len(result)} symbols in single call")
            return result
        except Exception as e:
            logger.error(f"Failed to get premarket 1-min bars: {e}")
            return {}

    def get_historical_1min_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Get historical 1-minute bars for a specific time range.

        Used by the backtesting engine to fetch a full day's worth of bars.

        Args:
            symbol: Stock symbol
            start: UTC start datetime (inclusive)
            end: UTC end datetime (inclusive)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            AlpacaAPIError: If API call fails
        """
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=start,
                end=end,
                feed=DataFeed.SIP,
            )
            bars_raw = self._call_with_timeout(
                lambda: self.data_client.get_stock_bars(request),
                f"get_historical_1min_bars({symbol})"
            )
            bars = self._to_dict(bars_raw)

            if symbol not in bars or len(bars[symbol]) == 0:
                logger.warning(f"No historical 1-min bars returned for {symbol} ({start} to {end})")
                return pd.DataFrame()

            records = []
            for bar in bars[symbol]:
                records.append({
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume),
                })

            logger.info(f"Fetched {len(records)} historical 1-min bars for {symbol} ({start} to {end})")
            return pd.DataFrame(records)

        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get historical 1-min bars for {symbol}: {e}")
            raise AlpacaAPIError(f"Failed to get historical 1-min bars for {symbol}: {e}")

    # =========================================================================
    # Trading Operations
    # =========================================================================

    def submit_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        limit_price: float,
        tp_price: float,
        sl_price: float,
    ) -> Dict:
        """
        Submit a bracket order (entry + stop loss + take profit).

        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            limit_price: Entry limit price
            tp_price: Take profit price
            sl_price: Stop loss price

        Returns:
            Dict with order details (id, status, etc.)

        Raises:
            AlpacaAPIError: If order submission fails
        """
        try:
            order_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL

            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                type=OrderType.LIMIT,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),
                order_class=OrderClass.BRACKET,
                take_profit={'limit_price': round(tp_price, 2)},
                stop_loss={'stop_price': round(sl_price, 2)},
            )

            order = self._call_with_timeout(
                lambda: self.trading_client.submit_order(request),
                f"submit_bracket_order({symbol})"
            )

            # 2026-06-09 (FABC fix): include `legs` in the return dict so
            # callers (ORB engine, bull flag engine) can populate
            # tp_leg_id / sl_leg_id on the StopMonitor watch. Previously
            # those IDs were always empty strings, which silently disabled
            # the BRANCH_SL_LEG_RACE recovery path in stop_monitor.py.
            # Mirrors the leg-extraction shape used by get_order:1358-1369.
            result = {
                'id': str(order.id) if hasattr(order, 'id') else '',
                'status': str(order.status.value) if hasattr(order, 'status') else 'unknown',
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'limit_price': limit_price,
                'legs': [
                    {
                        'id': str(leg.id),
                        'side': (str(leg.side.value) if hasattr(leg, 'side')
                                  and leg.side else ''),
                        'type': (str(leg.type.value) if hasattr(leg, 'type')
                                  and leg.type else ''),
                        'stop_price': (float(leg.stop_price)
                                        if leg.stop_price else None),
                        'limit_price': (float(leg.limit_price)
                                         if leg.limit_price else None),
                    }
                    for leg in (getattr(order, 'legs', None) or [])
                ],
            }

            logger.info(
                f"Bracket order submitted: {symbol} {side} {qty} "
                f"@ ${limit_price:.2f}, TP ${tp_price:.2f}, SL ${sl_price:.2f} "
                f"— ID: {result['id']}, status: {result['status']}, "
                f"legs={len(result['legs'])}"
            )
            return result

        except AlpacaAPIError:
            raise
        except Exception as e:
            self._log_order_op_failure("submit bracket order", symbol, e)
            raise AlpacaAPIError(f"Failed to submit bracket order for {symbol}: {e}")

    def get_open_positions(self) -> List[Dict]:
        """
        Get all current open positions from Alpaca.

        Returns:
            List of position dicts with symbol, qty, avg_entry, market_value, pnl

        Raises:
            AlpacaAPIError: If API call fails
        """
        try:
            positions = self._call_with_timeout(
                lambda: self.trading_client.get_all_positions(),
                "get_open_positions"
            )

            result = []
            for pos in positions:
                result.append({
                    'symbol': pos.symbol,
                    'qty': int(pos.qty),
                    'side': pos.side,
                    'avg_entry_price': float(pos.avg_entry_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                })

            logger.debug(f"Open positions: {len(result)}")
            return result

        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            raise AlpacaAPIError(f"Failed to get open positions: {e}")

    def get_account_info(self) -> Dict:
        """
        Get account information (equity, buying power, day trades).

        Returns:
            Dict with account details

        Raises:
            AlpacaAPIError: If API call fails
        """
        try:
            account = self._call_with_timeout(
                lambda: self.trading_client.get_account(),
                "get_account_info"
            )

            # 2026-07-06 incident: Alpaca returned daytrade_count=None on an
            # otherwise-ACTIVE account (post-holiday API quirk) and the bare
            # int() crashed pre-start validation -> SILENT 4h crash-loop on
            # the first live day after the weekend ships. Parse defensively;
            # None -> 0 with a WARNING (CLAUDE.md fallback rule).
            def _num(name, cast, default):
                v = getattr(account, name, None)
                if v is None:
                    logger.warning(
                        "get_account_info: %s is None from Alpaca — "
                        "defaulting to %s (account status: %s)",
                        name, default, account.status)
                    return default
                return cast(v)
            return {
                'equity': _num('equity', float, 0.0),
                'buying_power': _num('buying_power', float, 0.0),
                'cash': _num('cash', float, 0.0),
                'daytrade_count': _num('daytrade_count', int, 0),
                'pattern_day_trader': account.pattern_day_trader,
            }

        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise AlpacaAPIError(f"Failed to get account info: {e}")

    def get_buying_power(self) -> float:
        """Get current account buying power."""
        try:
            return self.get_account_info()['buying_power']
        except Exception as e:
            logger.error(f"Failed to get buying power: {e}")
            return 0.0

    def is_marginable(self, symbol: str) -> bool:
        """
        Check if a stock can be bought on margin.

        Args:
            symbol: Stock ticker

        Returns:
            True if marginable, False otherwise
        """
        try:
            asset = self._call_with_timeout(
                lambda: self.trading_client.get_asset(symbol),
                f"is_marginable({symbol})"
            )
            return bool(getattr(asset, 'marginable', False))
        except Exception as e:
            logger.warning(f"{symbol}: Failed to check marginable: {e}")
            return False  # Assume not marginable on error

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Handles 404 (already cancelled/not found) and 422 (already filled)
        gracefully by returning False instead of raising.

        Args:
            order_id: Alpaca order ID

        Returns:
            True if cancelled successfully, False if already cancelled/filled

        Raises:
            AlpacaAPIError: If cancellation fails for unexpected reasons
        """
        try:
            self._call_with_timeout(
                lambda: self.trading_client.cancel_order_by_id(order_id),
                f"cancel_order({order_id})"
            )
            logger.info(f"Order cancelled: {order_id}")
            return True
        except (AlpacaAPIError, APIError, Exception) as e:
            error_str = str(e).lower()
            status_code = getattr(e, 'status_code', None)
            # 404: order not found (already cancelled or never existed)
            if status_code == 404 or '404' in str(e) or 'not found' in error_str:
                logger.warning(f"Order {order_id} not found (already cancelled)")
                return False
            # 422: not cancelable (already filled)
            if status_code == 422 or '422' in str(e) or 'not cancelable' in error_str:
                logger.warning(f"Order {order_id} not cancelable (may be filled)")
                return False
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise AlpacaAPIError(f"Failed to cancel order {order_id}: {e}")

    def get_order(self, order_id: str) -> Dict:
        """
        Get order status by ID.

        Args:
            order_id: Alpaca order ID

        Returns:
            Dict with order details

        Raises:
            AlpacaAPIError: If query fails
        """
        try:
            order = self._call_with_timeout(
                lambda: self.trading_client.get_order_by_id(
                    order_id,
                    filter=GetOrderByIdRequest(nested=True)
                ),
                f"get_order({order_id})"
            )

            return {
                'id': str(order.id),
                'status': str(order.status.value) if hasattr(order, 'status') else 'unknown',
                'symbol': order.symbol,
                'qty': int(order.qty) if order.qty else 0,
                'filled_qty': int(order.filled_qty) if order.filled_qty else 0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'side': str(order.side.value) if hasattr(order, 'side') else '',
                'type': str(order.type.value) if hasattr(order, 'type') else '',
                'legs': [
                    {
                        'id': str(leg.id),
                        'side': str(leg.side.value) if hasattr(leg, 'side') and leg.side else '',
                        'type': str(leg.type.value) if hasattr(leg, 'type') and leg.type else '',
                        'stop_price': float(leg.stop_price) if leg.stop_price else None,
                        'limit_price': float(leg.limit_price) if leg.limit_price else None,
                        'filled_avg_price': float(leg.filled_avg_price) if leg.filled_avg_price else None,
                        'status': str(leg.status.value) if hasattr(leg, 'status') and leg.status else 'unknown',
                    }
                    for leg in (order.legs or [])
                ],
            }

        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            raise AlpacaAPIError(f"Failed to get order {order_id}: {e}")

    def replace_order_stop_price(self, order_id: str, new_stop_price: float) -> Dict:
        """
        Replace a child stop-loss order's stop price (for gap-fill adjustment).

        Args:
            order_id: Alpaca order ID of the SL leg
            new_stop_price: New stop price

        Returns:
            Dict with order id and status

        Raises:
            AlpacaAPIError: If replacement fails
        """
        try:
            from alpaca.trading.requests import ReplaceOrderRequest
            request = ReplaceOrderRequest(stop_price=round(new_stop_price, 2))
            order = self._call_with_timeout(
                lambda: self.trading_client.replace_order_by_id(order_id, request),
                f"replace_order_stop_price({order_id}, ${new_stop_price:.2f})"
            )
            logger.info(f"Order {order_id} stop replaced to ${new_stop_price:.2f}")
            return {'id': str(order.id), 'status': str(order.status.value)}
        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to replace order stop price {order_id}: {e}")
            raise AlpacaAPIError(f"Failed to replace order stop price {order_id}: {e}")

    def replace_order_limit_price(self, order_id: str, new_limit_price: float) -> Dict:
        """
        Replace a child take-profit order's limit price (for gap-fill adjustment).

        Args:
            order_id: Alpaca order ID of the TP leg
            new_limit_price: New limit price

        Returns:
            Dict with order id and status

        Raises:
            AlpacaAPIError: If replacement fails
        """
        try:
            from alpaca.trading.requests import ReplaceOrderRequest
            request = ReplaceOrderRequest(limit_price=round(new_limit_price, 2))
            order = self._call_with_timeout(
                lambda: self.trading_client.replace_order_by_id(order_id, request),
                f"replace_order_limit_price({order_id}, ${new_limit_price:.2f})"
            )
            logger.info(f"Order {order_id} limit replaced to ${new_limit_price:.2f}")
            return {'id': str(order.id), 'status': str(order.status.value)}
        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to replace order limit price {order_id}: {e}")
            raise AlpacaAPIError(f"Failed to replace order limit price {order_id}: {e}")

    def replace_order_qty(self, order_id: str, new_qty: int) -> Dict:
        """
        Replace an order's quantity (e.g., update safety-net SL after partial sell).

        Args:
            order_id: Alpaca order ID
            new_qty: New quantity

        Returns:
            Dict with order id and status

        Raises:
            AlpacaAPIError: If replacement fails
        """
        try:
            from alpaca.trading.requests import ReplaceOrderRequest
            request = ReplaceOrderRequest(qty=new_qty)
            order = self._call_with_timeout(
                lambda: self.trading_client.replace_order_by_id(order_id, request),
                f"replace_order_qty({order_id}, {new_qty})"
            )
            logger.info(f"Order {order_id} qty replaced to {new_qty}")
            return {'id': str(order.id), 'status': str(order.status.value)}
        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to replace order qty {order_id}: {e}")
            raise AlpacaAPIError(f"Failed to replace order qty {order_id}: {e}")

    def submit_stop_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        stop_price: float,
        limit_price: float,
        tp_price: float,
        sl_price: float,
    ) -> Dict:
        """
        Submit a stop-limit bracket order (buy-stop entry + stop loss + take profit).

        The order triggers when price hits stop_price, then fills at limit_price.
        Used for pre-placing buy-stop orders at breakout levels.

        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            stop_price: Trigger price (breakout_level)
            limit_price: Max fill price (breakout_level + slippage buffer)
            tp_price: Take profit price
            sl_price: Stop loss price

        Returns:
            Dict with order details (id, status, etc.)

        Raises:
            AlpacaAPIError: If order submission fails
        """
        try:
            from alpaca.trading.requests import StopLimitOrderRequest

            order_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL

            request = StopLimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                type=OrderType.STOP_LIMIT,
                time_in_force=TimeInForce.DAY,
                stop_price=round(stop_price, 2),
                limit_price=round(limit_price, 2),
                order_class=OrderClass.BRACKET,
                take_profit={'limit_price': round(tp_price, 2)},
                stop_loss={'stop_price': round(sl_price, 2)},
            )

            order = self._call_with_timeout(
                lambda: self.trading_client.submit_order(request),
                f"submit_stop_bracket_order({symbol})"
            )

            # 2026-06-12 (FABC fix follow-up): mirror submit_bracket_order
            # (commit 1797cc0) — include 'legs' in the response so callers
            # can populate tp_leg_id / sl_leg_id on the StopMonitor watch.
            # Pre-fix: ORB's REBUMP_STOP and SUBMIT_AS_IS entry paths went
            # through this method and silently lost the leg IDs, leaving
            # BRANCH_SL_LEG_RACE recovery dead-code for those entries.
            # OSCR 2026-06-10 was the canary case that exposed the gap.
            # Bull flag's stop-limit entries via order_executor:322 also
            # benefit from this.
            result = {
                'id': str(order.id) if hasattr(order, 'id') else '',
                'status': str(order.status.value) if hasattr(order, 'status') else 'unknown',
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'stop_price': stop_price,
                'limit_price': limit_price,
                'legs': [
                    {
                        'id': str(leg.id),
                        'side': (str(leg.side.value) if hasattr(leg, 'side')
                                  and leg.side else ''),
                        'type': (str(leg.type.value) if hasattr(leg, 'type')
                                  and leg.type else ''),
                        'stop_price': (float(leg.stop_price)
                                        if leg.stop_price else None),
                        'limit_price': (float(leg.limit_price)
                                         if leg.limit_price else None),
                    }
                    for leg in (getattr(order, 'legs', None) or [])
                ],
            }

            logger.info(
                f"Stop-bracket order submitted: {symbol} {side} {qty} "
                f"stop @ ${stop_price:.2f}, limit ${limit_price:.2f}, "
                f"TP ${tp_price:.2f}, SL ${sl_price:.2f} "
                f"— ID: {result['id']}, status: {result['status']}, "
                f"legs={len(result['legs'])}"
            )
            return result

        except AlpacaAPIError:
            raise
        except Exception as e:
            self._log_order_op_failure("submit stop-bracket order", symbol, e)
            raise AlpacaAPIError(f"Failed to submit stop-bracket order for {symbol}: {e}")

    def submit_stop_limit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        stop_price: float,
        limit_price: float,
    ) -> Dict:
        """
        Submit a simple stop-limit order (no bracket legs).

        Uses less margin than bracket orders — no TP/SL legs reserved.
        Used when StopMonitor handles real-time stop management.

        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            stop_price: Trigger price
            limit_price: Max fill price after trigger

        Returns:
            Dict with order details (id, status, symbol, qty)

        Raises:
            AlpacaAPIError: If order submission fails
        """
        try:
            from alpaca.trading.requests import StopLimitOrderRequest

            order_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL

            request = StopLimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                type=OrderType.STOP_LIMIT,
                time_in_force=TimeInForce.DAY,
                stop_price=round(stop_price, 2),
                limit_price=round(limit_price, 2),
                order_class=OrderClass.SIMPLE,
            )

            order = self._call_with_timeout(
                lambda: self.trading_client.submit_order(request),
                f"submit_stop_limit_order({symbol})"
            )

            result = {
                'id': str(order.id) if hasattr(order, 'id') else '',
                'status': str(order.status.value) if hasattr(order, 'status') else 'unknown',
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'stop_price': stop_price,
                'limit_price': limit_price,
            }

            logger.info(
                f"Stop-limit order submitted: {symbol} {side} {qty} "
                f"stop @ ${stop_price:.2f}, limit ${limit_price:.2f} "
                f"— ID: {result['id']}, status: {result['status']}"
            )
            return result

        except AlpacaAPIError:
            raise
        except Exception as e:
            self._log_order_op_failure("submit stop-limit order", symbol, e)
            raise AlpacaAPIError(f"Failed to submit stop-limit order for {symbol}: {e}")

    def submit_stop_sell_order(
        self,
        symbol: str,
        qty: int,
        stop_price: float,
    ) -> Dict:
        """
        Submit a standalone stop-market sell order (safety-net SL).

        Used as crash protection after fill — if service dies, this stop
        remains on Alpaca to limit loss.

        Args:
            symbol: Stock symbol
            qty: Number of shares to sell
            stop_price: Trigger price for stop sell

        Returns:
            Dict with order details (id, status, symbol)

        Raises:
            AlpacaAPIError: If order submission fails
        """
        try:
            from alpaca.trading.requests import StopOrderRequest

            request = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                type=OrderType.STOP,
                time_in_force=TimeInForce.DAY,
                stop_price=round(stop_price, 2),
            )

            order = self._call_with_timeout(
                lambda: self.trading_client.submit_order(request),
                f"submit_stop_sell_order({symbol})"
            )

            result = {
                'id': str(order.id) if hasattr(order, 'id') else '',
                'status': str(order.status.value) if hasattr(order, 'status') else 'unknown',
                'symbol': symbol,
                'qty': qty,
                'stop_price': stop_price,
            }

            logger.info(
                f"Stop sell order submitted: {symbol} SELL {qty} "
                f"stop @ ${stop_price:.2f} — ID: {result['id']}, "
                f"status: {result['status']}"
            )
            return result

        except AlpacaAPIError:
            raise
        except Exception as e:
            self._log_order_op_failure("submit stop sell order", symbol, e)
            raise AlpacaAPIError(f"Failed to submit stop sell order for {symbol}: {e}")

    def submit_limit_buy_order(
        self, symbol: str, qty: int, limit_price: float
    ) -> Dict:
        """
        Submit a plain limit buy order (no bracket).

        Used by MACD wave engine for entries — limit price set at/above ask
        for immediate fill while capping worst-case slippage.

        Args:
            symbol: Stock symbol
            qty: Number of shares to buy
            limit_price: Limit price

        Returns:
            Dict with order details (id, status, symbol)

        Raises:
            AlpacaAPIError: If order submission fails
        """
        try:
            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),
                order_class=OrderClass.SIMPLE,
            )

            order = self._call_with_timeout(
                lambda: self.trading_client.submit_order(request),
                f"submit_limit_buy_order({symbol})"
            )

            result = {
                'id': str(order.id) if hasattr(order, 'id') else '',
                'status': str(order.status.value) if hasattr(order, 'status') else 'unknown',
                'symbol': symbol,
                'qty': qty,
                'limit_price': limit_price,
            }

            logger.info(
                f"Limit buy order submitted: {symbol} BUY {qty} "
                f"@ ${limit_price:.2f} — ID: {result['id']}, "
                f"status: {result['status']}"
            )
            return result

        except AlpacaAPIError:
            raise
        except Exception as e:
            self._log_order_op_failure("submit limit buy order", symbol, e)
            raise AlpacaAPIError(f"Failed to submit limit buy order for {symbol}: {e}")

    def submit_limit_sell_order(
        self, symbol: str, qty: int, limit_price: float
    ) -> Dict:
        """
        Submit a plain limit sell order (no bracket).

        Used by StopMonitor for marketable limit exits — the limit price
        is set just below the current market price to fill immediately
        while capping worst-case slippage.

        Args:
            symbol: Stock symbol
            qty: Number of shares to sell
            limit_price: Limit price (marketable = just below current price)

        Returns:
            Dict with order details (id, status, symbol)

        Raises:
            AlpacaAPIError: If order submission fails
        """
        try:
            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                type=OrderType.LIMIT,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),
                order_class=OrderClass.SIMPLE,
            )

            order = self._call_with_timeout(
                lambda: self.trading_client.submit_order(request),
                f"submit_limit_sell_order({symbol})"
            )

            result = {
                'id': str(order.id) if hasattr(order, 'id') else '',
                'status': str(order.status.value) if hasattr(order, 'status') else 'unknown',
                'symbol': symbol,
                'qty': qty,
                'limit_price': limit_price,
            }

            logger.info(
                f"Limit sell order submitted: {symbol} SELL {qty} "
                f"@ ${limit_price:.2f} — ID: {result['id']}, "
                f"status: {result['status']}"
            )
            return result

        except AlpacaAPIError:
            raise
        except Exception as e:
            self._log_order_op_failure("submit limit sell order", symbol, e)
            raise AlpacaAPIError(f"Failed to submit limit sell order for {symbol}: {e}")

    def close_position(self, symbol: str) -> Dict:
        """
        Close a position by submitting a market sell order.

        Used for force-close at end of day or setup invalidation.

        Args:
            symbol: Stock symbol to close

        Returns:
            Dict with order details

        Raises:
            AlpacaAPIError: If close fails
        """
        try:
            order = self._call_with_timeout(
                lambda: self.trading_client.close_position(symbol),
                f"close_position({symbol})"
            )

            result = {
                'id': str(order.id) if hasattr(order, 'id') else '',
                'status': str(order.status.value) if hasattr(order, 'status') else 'unknown',
                'symbol': symbol,
            }

            logger.info(f"Position closed: {symbol} — ID: {result['id']}")
            return result

        except AlpacaAPIError:
            raise
        except Exception as e:
            self._log_order_op_failure("close position", symbol, e)
            raise AlpacaAPIError(f"Failed to close position for {symbol}: {e}")

    # =========================================================================
    # Connection Test
    # =========================================================================

    # ------------------------------------------------------------------
    # Market Calendar
    # ------------------------------------------------------------------

    def get_market_calendar(
        self, start_date: date, end_date: date
    ) -> List[Dict]:
        """
        Get market calendar for a date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of trading day dicts with 'date', 'open', 'close' keys

        Raises:
            AlpacaAPIError: If API call fails
        """
        logger.debug(f"Getting market calendar: {start_date} to {end_date}")

        try:
            request = GetCalendarRequest(
                start=start_date,
                end=end_date,
            )
            calendar = self._call_with_timeout(
                lambda: self.trading_client.get_calendar(request),
                "get_market_calendar"
            )
            result = [
                {
                    'date': day.date,
                    'open': day.open,
                    'close': day.close,
                }
                for day in calendar
            ]
            logger.debug(f"Found {len(result)} trading days")
            return result
        except Exception as e:
            logger.error(f"Failed to get market calendar: {e}")
            raise AlpacaAPIError(f"Failed to get market calendar: {e}")

    def is_trading_day(self, check_date: Optional[date] = None) -> bool:
        """
        Check if a date is a trading day (not weekend/holiday).

        Args:
            check_date: Date to check (defaults to today)

        Returns:
            True if it's a trading day
        """
        if check_date is None:
            check_date = date.today()
        calendar = self.get_market_calendar(check_date, check_date)
        return len(calendar) > 0

    def is_short_trading_day(self, check_date: Optional[date] = None) -> bool:
        """
        Check if a date is a short trading day (closes before 16:00 ET).

        Short days: day before Independence Day, Black Friday,
        Christmas Eve, etc.

        Args:
            check_date: Date to check (defaults to today)

        Returns:
            True if it's a short trading day
        """
        if check_date is None:
            check_date = date.today()

        calendar = self.get_market_calendar(check_date, check_date)
        if not calendar:
            return False

        close_time = calendar[0]['close']
        close_hour = (
            close_time.hour
            if hasattr(close_time, 'hour')
            else int(str(close_time).split(':')[0])
        )
        is_short = close_hour < 16

        if is_short:
            logger.info(
                f"Short trading day: {check_date} closes at {close_time}"
            )
        return is_short

    def test_connection(self) -> bool:
        """
        Test API connection by fetching account info.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            account = self._call_with_timeout(
                lambda: self.trading_client.get_account(),
                "test_connection"
            )
            logger.info(f"Alpaca API connected - Account: {account.account_number}")
            return True
        except Exception as e:
            logger.error(f"Alpaca API connection failed: {e}")
            return False
