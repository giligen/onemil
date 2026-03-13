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
    StockBarsRequest,
    StockLatestBarRequest,
    NewsRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetAssetsRequest,
    MarketOrderRequest,
    LimitOrderRequest,
)
from alpaca.trading.enums import AssetClass, AssetStatus, OrderSide, TimeInForce, OrderClass, OrderType
from alpaca.common.exceptions import APIError
import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar('T')

DEFAULT_API_TIMEOUT = 60
MAX_RATE_LIMIT_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.0


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

    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize AlpacaClient.

        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret

        Raises:
            AlpacaAPIError: If API credentials are missing
        """
        if not api_key:
            raise AlpacaAPIError("ALPACA_API_KEY required")
        if not api_secret:
            raise AlpacaAPIError("ALPACA_API_SECRET required")

        self.data_client = StockHistoricalDataClient(api_key, api_secret)
        self.trading_client = TradingClient(api_key, api_secret, paper=True)
        self.news_client = NewsClient(api_key, api_secret)
        self._api_timeout = DEFAULT_API_TIMEOUT

        logger.info("AlpacaClient initialized")

    def _call_with_timeout(self, func: Callable[[], T], operation: str) -> T:
        """
        Execute API call with timeout and rate limit retry.

        Args:
            func: Callable to execute
            operation: Description for logging

        Returns:
            Result of the function call

        Raises:
            AlpacaAPITimeoutError: If call times out
            AlpacaAPIError: If call fails or rate limit exhausted
        """
        backoff = INITIAL_BACKOFF_SECONDS
        last_exception = None

        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func)
                    return future.result(timeout=self._api_timeout)
            except FuturesTimeoutError:
                logger.error(f"API call timed out after {self._api_timeout}s: {operation}")
                raise AlpacaAPITimeoutError(f"API call timed out ({self._api_timeout}s): {operation}")
            except Exception as e:
                error_str = str(e).lower()
                if '429' in str(e) or 'rate limit' in error_str or 'too many requests' in error_str:
                    last_exception = e
                    if attempt < MAX_RATE_LIMIT_RETRIES:
                        logger.warning(
                            f"Rate limited on {operation} (attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES + 1}), "
                            f"retrying in {backoff:.1f}s..."
                        )
                        time_mod.sleep(backoff)
                        backoff *= 2
                        continue
                    else:
                        logger.error(f"Rate limit exhausted after {MAX_RATE_LIMIT_RETRIES + 1} attempts: {operation}")
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

    # =========================================================================
    # Assets
    # =========================================================================

    # Non-common-stock keywords in asset names
    _EXCLUDED_NAME_KEYWORDS = [
        'Warrant', 'Rights', 'Preferred',
    ]

    # Non-common-stock patterns in symbols
    _EXCLUDED_SYMBOL_PATTERNS = ['.PR']

    @classmethod
    def _is_common_stock(cls, symbol: str, name: str) -> bool:
        """
        Filter out warrants, units, preferred shares, rights, and SPACs.

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

    def get_daily_bars(self, symbols: List[str], days: int = 5) -> Dict[str, Dict]:
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
                        results[symbol] = {
                            'close': float(latest.close),
                            'volume': int(latest.volume),
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

    def get_current_bars(self, symbols: List[str], feed: DataFeed = DataFeed.SIP) -> Dict[str, Dict]:
        """
        Get current 15-min bar for multiple symbols.

        Fetches the most recent 15-min bar (aligned to bucket boundaries).
        This is the actual 15-min bucket volume needed for relative volume calculation.

        Args:
            symbols: List of stock symbols
            feed: Data feed

        Returns:
            Dict mapping symbol -> {open, high, low, close, volume, timestamp}

        Raises:
            AlpacaAPIError: If API call fails
        """
        if not symbols:
            return {}

        try:
            # Fetch last 30 min of 15-min bars to ensure we get the current bucket
            start = datetime.now(timezone.utc) - timedelta(minutes=30)
            results = {}
            chunk_size = 200
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i:i + chunk_size]
                request = StockBarsRequest(
                    symbol_or_symbols=chunk,
                    timeframe=TimeFrame(15, TimeFrameUnit.Minute),
                    start=start,
                    feed=feed,
                )
                bars_raw = self._call_with_timeout(
                    lambda req=request: self.data_client.get_stock_bars(req),
                    f"get_current_bars(chunk {i // chunk_size + 1})"
                )
                bars = self._to_dict(bars_raw)
                for symbol in chunk:
                    if symbol in bars and len(bars[symbol]) > 0:
                        bar = bars[symbol][-1]  # Most recent 15-min bar
                        results[symbol] = {
                            'open': float(bar.open),
                            'high': float(bar.high),
                            'low': float(bar.low),
                            'close': float(bar.close),
                            'volume': int(bar.volume),
                            'timestamp': bar.timestamp
                        }

            logger.debug(f"Fetched current 15-min bars for {len(results)}/{len(symbols)} symbols")
            return results

        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get current bars: {e}")
            raise AlpacaAPIError(f"Failed to get current bars: {e}")

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

    # =========================================================================
    # 1-Minute Bars (for pattern detection)
    # =========================================================================

    def get_1min_bars(self, symbol: str, lookback_minutes: int = 30) -> pd.DataFrame:
        """
        Get 1-minute bars for pattern detection.

        Returns completed bars only — the caller (BullFlagDetector) handles
        dropping the current in-progress bar.

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

            result = {
                'id': str(order.id) if hasattr(order, 'id') else '',
                'status': str(order.status.value) if hasattr(order, 'status') else 'unknown',
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'limit_price': limit_price,
            }

            logger.info(
                f"Bracket order submitted: {symbol} {side} {qty} "
                f"@ ${limit_price:.2f}, TP ${tp_price:.2f}, SL ${sl_price:.2f} "
                f"— ID: {result['id']}, status: {result['status']}"
            )
            return result

        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to submit bracket order for {symbol}: {e}")
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

            return {
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'daytrade_count': int(account.daytrade_count),
                'pattern_day_trader': account.pattern_day_trader,
            }

        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise AlpacaAPIError(f"Failed to get account info: {e}")

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Alpaca order ID

        Returns:
            True if cancelled successfully

        Raises:
            AlpacaAPIError: If cancellation fails
        """
        try:
            self._call_with_timeout(
                lambda: self.trading_client.cancel_order_by_id(order_id),
                f"cancel_order({order_id})"
            )
            logger.info(f"Order cancelled: {order_id}")
            return True
        except AlpacaAPIError:
            raise
        except Exception as e:
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
                lambda: self.trading_client.get_order_by_id(order_id),
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
            }

        except AlpacaAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            raise AlpacaAPIError(f"Failed to get order {order_id}: {e}")

    # =========================================================================
    # Connection Test
    # =========================================================================

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
