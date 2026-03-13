"""
Nightly batch universe builder.

Runs after market close to build the tradeable stock universe:
1. Fetch all tradeable US assets from Alpaca
2. Get previous close prices via daily bars
3. Filter by price ($2-$20)
4. Get float from Yahoo Finance (cached, weekly refresh)
5. Filter by float (<= 10M shares)
6. Cache 15-min volume profiles (50-day averages)

Verbose progress logging throughout.
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional

import pandas as pd

from data_sources.alpaca_client import AlpacaClient
from data_sources.float_provider import FloatProvider
from persistence.database import Database

logger = logging.getLogger(__name__)

# Time buckets for volume profiles (09:30 to 15:45, 15-min intervals)
TIME_BUCKETS = [
    f"{h:02d}:{m:02d}"
    for h in range(9, 16)
    for m in (0, 15, 30, 45)
    if (h > 9 or m >= 30) and (h < 16 or m == 0)
]
# Remove 16:00, keep 09:30 through 15:45
TIME_BUCKETS = [b for b in TIME_BUCKETS if b <= '15:45' and b >= '09:30']


class UniverseBuilder:
    """
    Builds and maintains the stock universe for the scanner.

    Orchestrates: Alpaca assets -> price filter -> float filter -> volume profiles.
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        float_provider: FloatProvider,
        db: Database,
        price_min: float = 2.0,
        price_max: float = 20.0,
        float_max: int = 10_000_000,
        volume_profile_days: int = 50,
        float_cache_refresh_days: int = 7,
    ):
        """
        Initialize UniverseBuilder.

        Args:
            alpaca_client: Alpaca API client
            float_provider: Yahoo Finance float provider
            db: Database instance
            price_min: Minimum price filter
            price_max: Maximum price filter
            float_max: Maximum float shares filter
            volume_profile_days: Days for volume profile calculation
            float_cache_refresh_days: Days before float data is stale
        """
        self.alpaca = alpaca_client
        self.float_provider = float_provider
        self.db = db
        self.price_min = price_min
        self.price_max = price_max
        self.float_max = float_max
        self.volume_profile_days = volume_profile_days
        self.float_cache_refresh_days = float_cache_refresh_days

    def build(self) -> Dict:
        """
        Run the full universe build pipeline.

        Pipeline (optimized to minimize expensive API calls):
        1. Fetch all tradeable common stock assets
        2. Get previous close prices (daily bars)
        3. Price filter $2-$20 (cuts ~11K to ~3K)
        4. Fetch float for price-filtered only (~3K, not 11K)
        5. Float filter <=10M (cuts to ~hundreds)
        6. Fetch 50-day 15-min volume profiles (only final universe)

        Returns:
            Summary dict with counts and changes
        """
        logger.info("=" * 60)
        logger.info("UNIVERSE BUILD STARTED")
        logger.info("=" * 60)

        # Capture previous universe for comparison
        previous_symbols = set(
            s['symbol'] for s in self.db.get_active_universe()
        )

        # Step 1: Fetch all tradeable common stock assets
        assets = self.alpaca.get_all_tradeable_assets()
        asset_map = {a['symbol']: a for a in assets}
        logger.info(f"Step 1: Fetched {len(assets)} tradeable US common stocks")

        # Step 2: Get previous close prices
        symbols = [a['symbol'] for a in assets]
        daily_bars = self.alpaca.get_daily_bars(symbols)
        logger.info(f"Step 2: Got daily bars for {len(daily_bars)}/{len(symbols)} stocks")

        # Step 3: Filter by price ($2-$20)
        price_filtered = self._filter_by_price(assets, daily_bars)
        price_filtered_syms = set(s['symbol'] for s in price_filtered)
        logger.info(
            f"Step 3: Price filter ${self.price_min}-${self.price_max}: "
            f"{len(price_filtered)} passed out of {len(daily_bars)}"
        )

        # Step 4: Persist price-filtered to DB, then fetch float
        now = datetime.now(timezone.utc)
        for stock in price_filtered:
            symbol = stock['symbol']
            bar = daily_bars.get(symbol, {})
            self.db.upsert_universe_stock({
                'symbol': symbol,
                'company_name': stock.get('company_name', ''),
                'exchange': stock.get('exchange', ''),
                'sector': None,
                'country': None,
                'price_close': bar.get('close', 0),
                'float_shares': None,
                'float_updated_at': None,
                'avg_volume_daily': bar.get('volume', 0),
                'last_updated': now,
                'active': 1,
            })
        logger.info(f"Step 4: Stored {len(price_filtered)} price-filtered stocks in DB")

        # Step 5: Fetch float + sector/country (only for price-filtered, skip fresh cache)
        symbols_need_float = self.db.get_symbols_needing_float_update(
            max_age_days=self.float_cache_refresh_days
        )
        symbols_need_float = [s for s in symbols_need_float if s in price_filtered_syms]
        logger.info(
            f"Step 5: {len(symbols_need_float)} need float update "
            f"(of {len(price_filtered)} price-filtered)"
        )

        if symbols_need_float:
            # Single pass: fetch float + sector + country together
            stock_info_batch = self.float_provider.get_stock_info_batch(symbols_need_float)
            for symbol, info in stock_info_batch.items():
                float_shares = info.get('float_shares')
                # Always update float_updated_at (even for None) to prevent re-fetching
                self.db.update_float(symbol, float_shares)

                if info.get('sector') or info.get('country'):
                    stock = self.db.get_universe_stock(symbol)
                    if stock:
                        stock['sector'] = info.get('sector') or stock.get('sector')
                        stock['country'] = info.get('country') or stock.get('country')
                        stock['last_updated'] = now
                        self.db.upsert_universe_stock(stock)

        # Step 6: Filter by float
        float_passed = self._filter_by_float()
        current_symbols = set(s['symbol'] for s in float_passed)
        logger.info(
            f"Step 6: Float filter <= {self.float_max / 1_000_000:.0f}M: "
            f"{len(float_passed)} passed"
        )

        # Deactivate stocks that no longer qualify
        to_deactivate = (previous_symbols | price_filtered_syms) - current_symbols
        if to_deactivate:
            self.db.deactivate_stocks(list(to_deactivate))
            logger.info(f"Deactivated {len(to_deactivate)} stocks no longer qualifying")

        # Step 7: Cache volume profiles (only for final universe - smallest set)
        self._cache_volume_profiles(float_passed)

        # Summary
        added = current_symbols - previous_symbols
        removed = previous_symbols - current_symbols

        logger.info("=" * 60)
        logger.info(
            f"UNIVERSE BUILD COMPLETE: {len(current_symbols)} stocks, "
            f"{self.db.get_volume_profile_count()} volume profiles cached"
        )
        logger.info(f"  Added: {len(added)} | Removed: {len(removed)}")
        if added:
            logger.info(f"  New: {sorted(added)[:20]}{'...' if len(added) > 20 else ''}")
        if removed:
            logger.info(f"  Removed: {sorted(removed)[:20]}{'...' if len(removed) > 20 else ''}")
        logger.info("=" * 60)

        return {
            'total_stocks': len(current_symbols),
            'volume_profiles': self.db.get_volume_profile_count(),
            'added': sorted(added),
            'removed': sorted(removed),
        }

    def _filter_by_price(
        self, assets: List[Dict], daily_bars: Dict[str, Dict]
    ) -> List[Dict]:
        """Filter assets by price range using previous close."""
        filtered = []
        for asset in assets:
            symbol = asset['symbol']
            bar = daily_bars.get(symbol)
            if not bar:
                continue
            close = bar['close']
            if self.price_min <= close <= self.price_max:
                filtered.append(asset)
        return filtered

    def _filter_by_float(self) -> List[Dict]:
        """
        Get active stocks that pass the float filter.

        Stocks with no float data are excluded (we can't verify they qualify).
        """
        all_active = self.db.get_active_universe()
        passed = []
        for stock in all_active:
            float_shares = stock.get('float_shares')
            if float_shares is not None and float_shares <= self.float_max:
                passed.append(stock)
            elif float_shares is None:
                logger.debug(f"{stock['symbol']}: excluded (float data unavailable)")
        return passed

    def _cache_volume_profiles(self, stocks: List[Dict]) -> None:
        """
        Fetch and cache 15-min volume profiles for all universe stocks.

        Args:
            stocks: List of stock dicts from universe
        """
        total = len(stocks)
        logger.info(f"Caching volume profiles for {total} stocks...")

        progress_interval = max(1, total // 10)  # Log every ~10%
        cached_count = 0

        for i, stock in enumerate(stocks):
            symbol = stock['symbol']
            try:
                profiles = self._calculate_volume_profile(symbol)
                if profiles:
                    self.db.upsert_volume_profiles(profiles)
                    cached_count += 1
            except Exception as e:
                logger.error(f"Failed to cache volume profile for {symbol}: {e}")

            if (i + 1) % progress_interval == 0 or (i + 1) == total:
                logger.info(f"Caching volume profiles: {i + 1}/{total} complete...")

        logger.info(f"Volume profiles cached: {cached_count}/{total} successful")

    def _calculate_volume_profile(self, symbol: str) -> List[Dict]:
        """
        Calculate average volume per 15-min bucket from historical intraday data.

        Args:
            symbol: Stock symbol

        Returns:
            List of profile dicts ready for DB insertion
        """
        df = self.alpaca.get_intraday_bars(symbol, days=self.volume_profile_days)
        if df.empty:
            logger.warning(f"{symbol}: no intraday data for volume profile")
            return []

        # Extract time bucket from timestamp
        df['time_bucket'] = df['timestamp'].apply(
            lambda ts: f"{ts.hour:02d}:{(ts.minute // 15) * 15:02d}"
        )

        # Filter to market hours only
        df = df[df['time_bucket'].isin(TIME_BUCKETS)]

        if df.empty:
            logger.warning(f"{symbol}: no market-hours data for volume profile")
            return []

        # Average volume per bucket across all days
        avg_by_bucket = df.groupby('time_bucket')['volume'].mean()

        now = datetime.now(timezone.utc)
        profiles = []
        for bucket, avg_vol in avg_by_bucket.items():
            profiles.append({
                'symbol': symbol,
                'time_bucket': bucket,
                'avg_volume': int(avg_vol),
                'last_updated': now,
            })

        logger.debug(f"{symbol}: calculated {len(profiles)} volume profile buckets")
        return profiles
