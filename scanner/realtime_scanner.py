"""
Real-time stock scanner for momentum day trading.

Two phases:
- Pre-market (4:00 AM - 9:30 AM ET): Gap-up detection with news check
- Intraday (9:30 AM - 4:00 PM ET): Volume + move qualification every 15 min

Uses Alpaca SIP feed for real-time data.
"""

import logging
import time as time_mod
from datetime import datetime, timezone, date
from typing import List, Dict, Optional, Set

import pytz

from data_sources.alpaca_client import AlpacaClient
from data_sources.news_provider import NewsProvider
from persistence.database import Database
from scanner.criteria import ScannerCriteria, ScanCandidate

logger = logging.getLogger(__name__)

ET = pytz.timezone('US/Eastern')


class RealtimeScanner:
    """
    Real-time momentum stock scanner.

    Pre-market: polls SIP every ~60s for gap-ups, checks news.
    Intraday: every 15 min, checks volume + price move + news.
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        news_provider: NewsProvider,
        db: Database,
        criteria: ScannerCriteria,
        poll_interval: int = 60,
        verbose: bool = False,
    ):
        """
        Initialize RealtimeScanner.

        Args:
            alpaca_client: Alpaca API client
            news_provider: News provider with analyzer
            db: Database instance
            criteria: Scanner criteria engine
            poll_interval: Pre-market polling interval in seconds
            verbose: Enable verbose output
        """
        self.alpaca = alpaca_client
        self.news = news_provider
        self.db = db
        self.criteria = criteria
        self.poll_interval = poll_interval
        self.verbose = verbose

        self._universe: List[Dict] = []
        self._volume_profiles: Dict[str, Dict[str, int]] = {}
        self._premarket_gap_symbols: Set[str] = set()
        self._today = date.today().isoformat()

    def run(self) -> None:
        """
        Run the scanner (main loop).

        Determines current phase based on Eastern Time
        and runs the appropriate scanning loop.
        """
        self._load_universe()

        if not self._universe:
            logger.error("No stocks in universe. Run --batch first.")
            return

        logger.info(f"Scanner starting with {len(self._universe)} universe stocks")

        while True:
            now_et = datetime.now(ET)
            current_time = now_et.strftime("%H:%M")

            if current_time < "04:00":
                logger.info("Before pre-market (04:00 ET). Waiting...")
                self._sleep_until("04:00")
            elif current_time < "09:30":
                self._run_premarket_cycle()
                time_mod.sleep(self.poll_interval)
            elif current_time < "16:00":
                self._run_intraday_cycle()
                self._sleep_until_next_bucket()
            else:
                logger.info("Market closed (after 16:00 ET). Scanner complete.")
                break

    def run_test_cycle(self) -> Dict:
        """
        Run a single premarket + intraday cycle for testing.

        Runs against real API data, no time-gating or sleep.
        Returns a summary dict for verification.
        """
        self._load_universe()

        if not self._universe:
            logger.error("No stocks in universe. Run --batch first.")
            return {'error': 'empty universe'}

        symbols = [s['symbol'] for s in self._universe]
        logger.info(f"TEST CYCLE: {len(self._universe)} stocks, "
                     f"{len(self._volume_profiles)} with volume profiles")

        # --- Pre-market gap scan (uses latest trade vs prev close) ---
        logger.info("=" * 60)
        logger.info("TEST: Running pre-market gap scan...")
        self._run_premarket_cycle()

        premarket_results = self.db.get_scan_results(self._today, phase='premarket')
        logger.info(f"Pre-market: {len(premarket_results)} gap-up candidates found")

        # --- Intraday scan ---
        logger.info("=" * 60)
        logger.info("TEST: Running intraday scan cycle...")
        self._run_intraday_cycle()

        intraday_results = self.db.get_scan_results(self._today, phase='intraday')
        logger.info(f"Intraday: {len(intraday_results)} qualified stocks found")

        # --- Summary stats ---
        logger.info("=" * 60)
        logger.info("TEST CYCLE SUMMARY")
        logger.info(f"  Universe: {len(self._universe)} stocks")
        logger.info(f"  Volume profiles loaded: {len(self._volume_profiles)}")
        logger.info(f"  Pre-market gap-ups (>=2%): {len(premarket_results)}")
        for r in premarket_results:
            logger.info(f"    {r['symbol']}: gap {r['gap_pct']:.1f}%, "
                        f"price ${r['current_price']:.2f}, news: {r['news_headline']}")
        logger.info(f"  Intraday qualified: {len(intraday_results)}")
        for r in intraday_results:
            logger.info(f"    {r['symbol']}: {r['intraday_change_pct']:+.1f}%, "
                        f"relVol {r['relative_volume']:.1f}x, "
                        f"price ${r['current_price']:.2f}, "
                        f"bucket {r['time_bucket']}")
        logger.info("=" * 60)

        return {
            'universe_size': len(self._universe),
            'volume_profiles': len(self._volume_profiles),
            'premarket_candidates': len(premarket_results),
            'intraday_qualified': len(intraday_results),
            'premarket_results': premarket_results,
            'intraday_results': intraday_results,
        }

    def _load_universe(self) -> None:
        """Load universe and volume profiles from DB."""
        self._universe = self.db.get_active_universe()
        self._volume_profiles = self.db.get_all_volume_profiles()
        logger.info(
            f"Loaded universe: {len(self._universe)} stocks, "
            f"{len(self._volume_profiles)} with volume profiles"
        )

    # =========================================================================
    # Pre-Market Phase
    # =========================================================================

    def _run_premarket_cycle(self) -> None:
        """Run one pre-market scan cycle."""
        symbols = [s['symbol'] for s in self._universe]

        # Get latest trades (SIP for pre-market data)
        trades = self.alpaca.get_latest_trades(symbols)

        candidates = []
        for stock in self._universe:
            symbol = stock['symbol']
            trade = trades.get(symbol)
            if not trade or trade['price'] <= 0:
                continue

            prev_close = stock['price_close']
            if prev_close <= 0:
                continue

            current_price = trade['price']
            gap_pct = ((current_price - prev_close) / prev_close) * 100

            if gap_pct < self.criteria.gap_pct_min:
                continue

            # Check news for gap candidates
            has_news, headline = self.news.has_interesting_news(symbol)

            candidate = ScanCandidate(
                symbol=symbol,
                company_name=stock.get('company_name', ''),
                prev_close=prev_close,
                current_price=current_price,
                float_shares=stock.get('float_shares', 0),
                gap_pct=gap_pct,
                has_news=has_news,
                news_headline=headline,
            )

            qualified = self.criteria.evaluate_premarket(candidate)
            if qualified:
                candidates.append(candidate)
                self._premarket_gap_symbols.add(symbol)

                # Save to DB
                self.db.save_scan_result({
                    'scan_date': self._today,
                    'symbol': symbol,
                    'detected_at': datetime.now(timezone.utc),
                    'phase': 'premarket',
                    'prev_close': prev_close,
                    'current_price': current_price,
                    'gap_pct': gap_pct,
                    'intraday_change_pct': gap_pct,
                    'relative_volume': None,
                    'current_volume': None,
                    'time_bucket': None,
                    'float_shares': stock.get('float_shares', 0),
                    'has_news': 1 if has_news else 0,
                    'news_headline': headline,
                    'qualified': 1,
                })

        # Output
        now_et = datetime.now(ET).strftime("%H:%M")
        if candidates:
            print(f"\n{'=' * 70}")
            print(f"PRE-MARKET GAP-UPS ({now_et} ET) - {len(candidates)} candidates")
            print(f"{'=' * 70}")
            for c in sorted(candidates, key=lambda x: x.gap_pct, reverse=True):
                print(self.criteria.format_candidate(c, 'premarket'))
        elif self.verbose:
            print(f"Pre-market scan {now_et} ET | Scanned: {len(symbols)} | Gap-ups: 0")

    # =========================================================================
    # Intraday Phase
    # =========================================================================

    def _run_intraday_cycle(self) -> None:
        """Run one intraday scan cycle (every 15 min)."""
        now_et = datetime.now(ET)
        bucket = f"{now_et.hour:02d}:{(now_et.minute // 15) * 15:02d}"

        symbols = [s['symbol'] for s in self._universe]
        universe_map = {s['symbol']: s for s in self._universe}

        # Get current bars for volume check
        bars = self.alpaca.get_current_bars(symbols)

        # Get latest trades for current price
        trades = self.alpaca.get_latest_trades(symbols)

        qualified = []
        close_calls = []
        hot_stocks = []  # 5x vol + 10% move (pre-news filter)
        vol_5x_count = 0
        move_10pct_count = 0
        news_count = 0

        for stock in self._universe:
            symbol = stock['symbol']
            bar = bars.get(symbol)
            trade = trades.get(symbol)
            if not bar or not trade:
                continue

            prev_close = stock['price_close']
            if prev_close <= 0:
                continue

            current_price = trade['price']
            current_volume = bar['volume']

            # Calculate metrics
            gap_pct = ((current_price - prev_close) / prev_close) * 100
            intraday_change_pct = gap_pct  # Same calc: current vs prev close

            # Relative volume
            profile = self._volume_profiles.get(symbol, {})
            avg_vol = profile.get(bucket, 0)
            relative_volume = current_volume / avg_vol if avg_vol > 0 else 0.0

            if relative_volume >= self.criteria.relative_volume_min:
                vol_5x_count += 1
            if intraday_change_pct >= self.criteria.intraday_change_pct_min:
                move_10pct_count += 1

            # Only check news for stocks that pass volume + price criteria
            has_news = False
            headline = None
            if (relative_volume >= self.criteria.relative_volume_min and
                    intraday_change_pct >= self.criteria.intraday_change_pct_min):
                has_news, headline = self.news.has_interesting_news(symbol)
                if has_news:
                    news_count += 1

                # Track hot stocks for verbose output (regardless of news)
                hot_stocks.append({
                    'symbol': symbol,
                    'company_name': stock.get('company_name', ''),
                    'prev_close': prev_close,
                    'current_price': current_price,
                    'change_pct': intraday_change_pct,
                    'relative_volume': relative_volume,
                    'current_volume': current_volume,
                    'avg_volume': avg_vol,
                    'float_shares': stock.get('float_shares', 0),
                    'has_news': has_news,
                    'headline': headline,
                })

            candidate = ScanCandidate(
                symbol=symbol,
                company_name=stock.get('company_name', ''),
                prev_close=prev_close,
                current_price=current_price,
                float_shares=stock.get('float_shares', 0),
                gap_pct=gap_pct,
                intraday_change_pct=intraday_change_pct,
                relative_volume=relative_volume,
                current_volume=current_volume,
                time_bucket=bucket,
                has_news=has_news,
                news_headline=headline,
            )

            is_qualified = self.criteria.evaluate_intraday(candidate)

            if is_qualified:
                qualified.append(candidate)

                # Save qualified result to DB
                self.db.save_scan_result({
                    'scan_date': self._today,
                    'symbol': symbol,
                    'detected_at': datetime.now(timezone.utc),
                    'phase': 'intraday',
                    'prev_close': prev_close,
                    'current_price': current_price,
                    'gap_pct': gap_pct,
                    'intraday_change_pct': intraday_change_pct,
                    'relative_volume': relative_volume,
                    'current_volume': current_volume,
                    'time_bucket': bucket,
                    'float_shares': stock.get('float_shares', 0),
                    'has_news': 1 if has_news else 0,
                    'news_headline': headline,
                    'qualified': 1,
                })
            elif candidate.criteria_met_count >= candidate.total_criteria - 1:
                close_calls.append(candidate)

        # Output
        self._print_intraday_output(bucket, symbols, vol_5x_count, move_10pct_count,
                                     news_count, qualified, close_calls, hot_stocks)

    def _print_intraday_output(
        self,
        bucket: str,
        symbols: list,
        vol_5x: int,
        move_10pct: int,
        news: int,
        qualified: List[ScanCandidate],
        close_calls: List[ScanCandidate],
        hot_stocks: Optional[List[Dict]] = None,
    ) -> None:
        """Print intraday scan results to console."""
        summary_line = (
            f"Scan {bucket} ET | Universe: {len(symbols)} | "
            f"5x Vol: {vol_5x} | 10%+ Move: {move_10pct} | "
            f"News: {news} | QUALIFIED: {len(qualified)}"
        )

        if qualified:
            print(f"\n{'=' * 70}")
            print(summary_line)
            print(f"{'=' * 70}")
            for c in sorted(qualified, key=lambda x: x.intraday_change_pct, reverse=True):
                print(self.criteria.format_candidate(c, 'intraday'))
        elif self.verbose:
            print(summary_line)

        if self.verbose:
            # Show hot stocks: passed 5x vol + 10% move (before news filter)
            if hot_stocks:
                print(f"  Hot stocks (5x vol + 10%+ move): {len(hot_stocks)}")
                for h in sorted(hot_stocks, key=lambda x: x['change_pct'], reverse=True):
                    news_status = f'"{h["headline"]}"' if h['has_news'] else 'NO NEWS'
                    print(
                        f"    {h['symbol']:<6} "
                        f"${h['prev_close']:.2f} -> ${h['current_price']:.2f} "
                        f"({h['change_pct']:+.1f}%)  "
                        f"RelVol: {h['relative_volume']:.1f}x "
                        f"(vol: {h['current_volume']:,} / avg: {h['avg_volume']:,})  "
                        f"Float: {h['float_shares'] / 1_000_000:.1f}M  "
                        f"News: {news_status}"
                    )

            if close_calls:
                print(f"  Close calls ({len(close_calls)}):")
                for c in close_calls:
                    missing = [k for k, v in c.criteria_met.items() if not v]
                    print(
                        f"    {c.symbol:<6} "
                        f"${c.current_price:.2f} ({c.intraday_change_pct:+.1f}%) "
                        f"RelVol: {c.relative_volume:.1f}x "
                        f"Missing: {missing}"
                    )

    # =========================================================================
    # Timing Helpers
    # =========================================================================

    def _sleep_until(self, target_time: str) -> None:
        """Sleep until a target Eastern Time (HH:MM)."""
        now_et = datetime.now(ET)
        target_h, target_m = map(int, target_time.split(':'))
        target = now_et.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
        if target <= now_et:
            return
        sleep_secs = (target - now_et).total_seconds()
        logger.info(f"Sleeping {sleep_secs:.0f}s until {target_time} ET")
        time_mod.sleep(sleep_secs)

    def _sleep_until_next_bucket(self) -> None:
        """Sleep until the next 15-min bucket boundary."""
        now_et = datetime.now(ET)
        # Next bucket: round up to next 15-min mark
        minutes_past = now_et.minute % 15
        if minutes_past == 0:
            # We're at a boundary, sleep until the next one
            sleep_mins = 15
        else:
            sleep_mins = 15 - minutes_past
        sleep_secs = sleep_mins * 60 - now_et.second
        if sleep_secs > 0:
            logger.debug(f"Sleeping {sleep_secs}s until next 15-min bucket")
            time_mod.sleep(sleep_secs)
