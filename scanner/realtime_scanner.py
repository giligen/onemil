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
        trading_engine=None,
        notifier=None,
        shutdown_event=None,
        macd_engine=None,
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
            trading_engine: Optional TradingEngine for bull flag strategy
            notifier: Optional TelegramNotifier for alerts
            shutdown_event: Optional threading.Event for graceful shutdown
            macd_engine: Optional MACDWaveEngine for MACD wave strategy
        """
        self.alpaca = alpaca_client
        self.news = news_provider
        self.db = db
        self.criteria = criteria
        self.poll_interval = poll_interval
        self.verbose = verbose
        self.trading_engine = trading_engine
        self.notifier = notifier
        self.shutdown_event = shutdown_event
        self.macd_engine = macd_engine
        # Throttle repeated bar-drain errors: the 1s sleep chunk loop can
        # otherwise log the same exception 60× per minute if something stays
        # broken. Track last-seen error signature + timestamp.
        self._macd_bar_drain_last_err: Optional[str] = None
        self._macd_bar_drain_last_err_ts: float = 0.0
        self._macd_bar_drain_err_count: int = 0

        # Load notification preferences from config
        from config import Config
        _cfg = Config._load_yaml_only()
        notif_cfg = _cfg.get("notifications", {}).get("telegram", {})
        self._send_on_qualified = bool(notif_cfg.get("send_on_qualified", True))

        self._universe: List[Dict] = []
        self._volume_profiles: Dict[str, Dict[str, int]] = {}
        self._premarket_gap_symbols: Set[str] = set()
        self._premarket_gap_data: List[Dict] = []
        self._qualified_stock_data: List[Dict] = []
        self._day_highs: Dict[str, float] = {}  # Track intraday highs for V-reversal detection
        self._day_lows: Dict[str, float] = {}   # Track intraday lows for V-reversal detection

        # Async news worker — non-blocking LLM classification
        # Enabled by default; tests can pass async_news=False to disable
        self._news_worker = None

    def enable_async_news(self):
        """Start async news worker for non-blocking LLM classification."""
        from data_sources.news_provider import NewsWorker
        self._news_worker = NewsWorker(self.news)
        self._news_worker.start()

    @property
    def _today(self) -> str:
        """Current date as ISO string. Always fresh, never stale."""
        return date.today().isoformat()

    def run(self) -> None:
        """
        Run the scanner (main loop).

        0. Check if today is a trading day (skip weekends/holidays/short days)
        1. Wait until market open (09:30 ET)
        2. Run ONE gap-up scan (current price vs previous close)
        3. Run intraday cycles every 15 min until market close (16:00 ET)
        """
        # Check if today is a trading day
        if not self._is_trading_day():
            return

        self._load_universe()

        if not self._universe and self.macd_engine is None:
            logger.error("No stocks in universe. Run --batch first.")
            return

        # Build MACD wave universe (pre-market, independent from bull flag)
        if self.macd_engine is not None:
            self.macd_engine.build_universe()
            logger.info(f"MACD Wave universe built: {len(self.macd_engine.universe)} stocks")

        if self.trading_engine is not None:
            self.trading_engine.reset_daily()
            logger.info("Trading engine daily state reset at scanner startup")

        logger.info(f"Scanner starting with {len(self._universe)} universe stocks")

        now_et = datetime.now(ET)
        current_time = now_et.strftime("%H:%M")

        if current_time >= "16:00":
            logger.info("Market closed (after 16:00 ET). Nothing to do.")
            return

        # Pre-market wait (interruptible)
        if current_time < "09:30":
            logger.info("Waiting for market open (09:30 ET)...")
            if self._sleep_until("09:30"):
                logger.warning("Shutdown during pre-market wait — exiting")
                return

        # Gap scan once at open
        if not self._premarket_gap_symbols:
            logger.info("Running opening gap scan...")
            self._run_premarket_cycle()

        # Intraday loop: scanner cycles every 15 min, engine ticks every 60s
        force_closed = False
        last_bucket = None
        _scanner_start = time_mod.time()

        while True:
            # Check shutdown
            if self.shutdown_event and self.shutdown_event.is_set():
                logger.warning("Shutdown signal received in scanner loop")
                break

            now_et = datetime.now(ET)
            current_time = now_et.strftime("%H:%M")

            if current_time >= "16:00":
                logger.info("Market closed (16:00 ET). Scanner complete.")
                break

            engine = self.trading_engine

            # Force close check (Bug #1 fix)
            if engine is not None and engine.enabled and not force_closed:
                if engine._is_past_force_close_time():
                    logger.info("Force close time reached — closing all positions")
                    engine._force_close_all()
                    force_closed = True

            # Scanner intraday cycle every minute (matches backtest bar-by-bar qualification)
            current_bucket = f"{now_et.hour:02d}:{now_et.minute:02d}"
            if current_bucket != last_bucket:
                if engine is not None:
                    engine.clear_qualified_symbols()   # Bug #4 fix
                self._run_intraday_cycle()
                last_bucket = current_bucket

            # Circuit breaker: StopMonitor dead → close all → exit
            # Skip first 3 minutes (grace period for WebSocket to connect)
            sm = (getattr(engine, 'stop_monitor', None) if engine
                  else getattr(self.macd_engine, 'stop_monitor', None) if self.macd_engine
                  else None)
            if (sm is not None
                    and time_mod.time() - _scanner_start > 180
                    and not sm.is_healthy()):
                msg = (
                    f"CRITICAL: StopMonitor DEAD — "
                    f"running={sm._running}, "
                    f"thread={sm._thread.is_alive() if sm._thread else False}"
                )
                logger.error(msg)
                if engine:
                    engine._force_close_all()
                if self.macd_engine:
                    self.macd_engine.force_close_all()
                if self.notifier:
                    try:
                        self.notifier.send_message_sync(
                            msg + "\nEmergency closed all positions."
                        )
                    except Exception:
                        pass
                sm.stop()
                logger.error("Exiting — StopMonitor infrastructure failure")
                import sys
                sys.exit(1)

            # Engine pattern check every tick ~60s (Bug #2 fix)
            if engine is not None and engine.enabled and not force_closed:
                engine.run_pattern_check()

            # MACD wave tick — scan, entries, exits every 60s
            if self.macd_engine is not None and not force_closed:
                try:
                    self.macd_engine.scan_for_movers()
                    self.macd_engine.check_entries()
                    self.macd_engine.check_exits()
                    if self.macd_engine.is_force_close_time():
                        self.macd_engine.force_close_all()
                except Exception as e:
                    logger.error(f"MACD wave cycle error: {e}", exc_info=True)

            # Sleep in 1s chunks, draining RT bar events between (Bug #3 fix + latency fix)
            # Bar events arrive via WebSocket on bar close. Without frequent
            # draining, they queue up for the full 60s sleep → 60s+ latency.
            # With 1s chunks: worst case 1s from bar close to pattern check.
            _sleep_remaining = 60
            while _sleep_remaining > 0:
                _chunk = min(1, _sleep_remaining)
                if self._interruptible_sleep(_chunk):
                    logger.warning("Shutdown signal received during sleep")
                    break
                _sleep_remaining -= _chunk
                # Drain RT bar events from WebSocket (sub-1s latency)
                if engine is not None and engine.enabled and not force_closed:
                    rt = engine._drain_bar_events()
                    if rt:
                        logger.info(f"RT bar event processed during sleep: {rt.get('symbol', '?')}")
                # T1.1: drain MACD wave bar events and run targeted check_entries
                # for any crossed_stocks that just got a new bar close.
                if self.macd_engine is not None and not force_closed:
                    try:
                        mw_syms = self.macd_engine.drain_bar_events()
                        reeval = mw_syms & set(self.macd_engine.crossed_stocks.keys())
                        if reeval:
                            self.macd_engine.check_entries(symbols=reeval)
                        # Reset error tracking on success
                        if self._macd_bar_drain_err_count:
                            logger.info(
                                f"MACD wave bar-drain recovered after "
                                f"{self._macd_bar_drain_err_count} errors"
                            )
                            self._macd_bar_drain_last_err = None
                            self._macd_bar_drain_err_count = 0
                    except Exception as e:
                        self._log_throttled_bar_drain_error(e)
            else:
                continue  # Normal loop continuation
            break  # Shutdown signal

        # Post-loop safety net: force close (Bug #1 fix)
        if self.trading_engine is not None and self.trading_engine.enabled:
            if not force_closed:
                logger.info("End-of-day safety net — force-closing all positions")
                self.trading_engine._force_close_all()
        if self.macd_engine is not None:
            self.macd_engine.force_close_all()

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

        if self.trading_engine is not None:
            self.trading_engine.reset_daily()
            logger.info("Trading engine daily state reset at scanner startup")

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
        """Load universe, volume profiles, and fresh prev closes from Alpaca."""
        self._universe = self.db.get_active_universe()
        self._volume_profiles = self.db.get_all_volume_profiles()

        # Fetch yesterday's actual close for all symbols from Alpaca.
        # The universe price_close can be weeks stale — using it for gap
        # calculation causes stocks with recent price changes to be missed
        # or incorrectly qualified.
        self._refresh_prev_closes()

        logger.info(
            f"Loaded universe: {len(self._universe)} stocks, "
            f"{len(self._volume_profiles)} with volume profiles"
        )

    def _refresh_prev_closes(self) -> None:
        """Fetch yesterday's close from Alpaca and update universe records in memory."""
        from datetime import timedelta
        symbols = [s['symbol'] for s in self._universe]
        if not symbols:
            return

        today = date.today()
        # Fetch last 7 calendar days to find the most recent trading day
        start = today - timedelta(days=7)
        try:
            daily_bars = self.alpaca.get_daily_bars_range(symbols, start, today)
        except Exception as e:
            logger.error(f"Failed to fetch prev closes from Alpaca: {e}")
            return

        updated = 0
        for stock in self._universe:
            sym = stock['symbol']
            bars = daily_bars.get(sym, [])
            if not bars:
                continue
            # Use the most recent bar's close as prev_close
            # Sort by date descending, take the first that's before today
            sorted_bars = sorted(bars, key=lambda b: str(b['date']), reverse=True)
            for bar in sorted_bars:
                bar_date = bar['date'] if isinstance(bar['date'], date) else date.fromisoformat(str(bar['date']))
                if bar_date < today:
                    stock['price_close'] = bar['close']
                    updated += 1
                    break

        logger.info(f"Refreshed prev closes from Alpaca: {updated}/{len(symbols)} symbols")

    # =========================================================================
    # Pre-Market Phase
    # =========================================================================

    def _run_premarket_cycle(self) -> None:
        """Run one pre-market gap scan. Pure quantitative — no LLM calls.

        Finds stocks gapping >= threshold from previous close.
        News/LLM classification happens later in intraday, as the LAST step.
        """
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

            candidate = ScanCandidate(
                symbol=symbol,
                company_name=stock.get('company_name', ''),
                prev_close=prev_close,
                current_price=current_price,
                float_shares=stock.get('float_shares', 0),
                gap_pct=gap_pct,
            )
            candidates.append(candidate)
            self._premarket_gap_symbols.add(symbol)
            self._premarket_gap_data.append({
                'symbol': symbol,
                'prev_close': prev_close,
                'current_price': current_price,
                'gap_pct': gap_pct,
            })

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
                'has_news': 0,
                'news_headline': None,
                'qualified': 1,
            })

        candidates.sort(key=lambda x: x.gap_pct, reverse=True)
        logger.info(
            f"Pre-market gap scan: {len(candidates)} stocks >={self.criteria.gap_pct_min}% gap "
            f"(out of {len(symbols)} universe)"
        )

        # Telegram notification for gap-ups
        if candidates and self.notifier:
            self.notifier.notify_premarket_gaps(self._premarket_gap_data)

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
        symbols = [s['symbol'] for s in self._universe]
        universe_map = {s['symbol']: s for s in self._universe}

        # Default bucket from wall clock (overridden per-stock from bar timestamp)
        now_et = datetime.now(ET)
        bucket = f"{now_et.hour:02d}:{(now_et.minute // 15) * 15:02d}"

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

            # Compute volume bucket from the bar's timestamp (not wall clock)
            # so it matches the completed bar returned by get_current_bars()
            bar_ts = bar.get('timestamp')
            if bar_ts is not None:
                if isinstance(bar_ts, str):
                    bar_ts = datetime.fromisoformat(bar_ts.replace('Z', '+00:00'))
                if bar_ts.tzinfo is None:
                    bar_ts = bar_ts.replace(tzinfo=pytz.utc)
                bar_et = bar_ts.astimezone(ET)
            else:
                # Fallback: use wall clock if bar has no timestamp
                bar_et = datetime.now(ET)
                logger.warning(f"{symbol}: Bar has no timestamp, using wall clock for bucket")

            bucket = f"{bar_et.hour:02d}:{(bar_et.minute // 15) * 15:02d}"

            # Track intraday extremes for V-reversal detection
            bar_high = bar.get('high', current_price)
            bar_low = bar.get('low', current_price)
            self._day_highs[symbol] = max(self._day_highs.get(symbol, 0), bar_high)
            self._day_lows[symbol] = min(self._day_lows.get(symbol, float('inf')), bar_low)

            # Calculate metrics — qualify on gap-up OR V-reversal (intraday range)
            gap_pct = ((current_price - prev_close) / prev_close) * 100
            day_low = self._day_lows.get(symbol, current_price)
            range_pct = ((self._day_highs.get(symbol, current_price) - day_low) / day_low * 100) if day_low > 0 else 0
            intraday_change_pct = max(gap_pct, range_pct)  # Qualify on whichever is higher

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
            news_catalyst = None
            news_reason = None
            news_category = None
            if (relative_volume >= self.criteria.relative_volume_min and
                    intraday_change_pct >= self.criteria.intraday_change_pct_min):
                _stock_ctx = {
                    'float_shares': stock.get('float_shares', 0),
                    'price': current_price,
                }
                # Async news: check cache first, enqueue if not yet classified
                if self._news_worker:
                    news_info = self._news_worker.get_result(symbol)
                    if news_info is None:
                        # Not yet classified — enqueue for background processing
                        self._news_worker.enqueue(symbol, stock_context=_stock_ctx)
                        # Use empty result for now; next cycle will have it
                        news_info = {'has_news': False, 'catalyst': None,
                                     'category': 'PENDING', 'headline': '',
                                     'reason': 'async pending', 'news_headline': ''}
                else:
                    # Fallback: blocking classification (no async worker)
                    news_info = self.news.classify_news(symbol, stock_context=_stock_ctx)
                has_news = news_info.get('has_news', False)
                headline = news_info.get('headline')
                news_catalyst = news_info.get('catalyst')
                news_reason = news_info.get('reason', '')
                news_category = news_info.get('category', 'OTHER')
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

                # Track for daily report
                self._qualified_stock_data.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'intraday_change_pct': intraday_change_pct,
                    'relative_volume': relative_volume,
                    'news_headline': headline,
                    'time_bucket': bucket,
                })

                # Notify via Telegram (if send_on_qualified enabled)
                if self.notifier and self._send_on_qualified:
                    self.notifier.notify_stock_qualified(
                        symbol=symbol,
                        price=current_price,
                        change_pct=intraday_change_pct,
                        relative_volume=relative_volume,
                        headline=headline,
                    )

                # Hand off to trading engine if available
                if self.trading_engine is not None:
                    self.trading_engine.on_stock_qualified(
                        symbol,
                        news_catalyst=news_catalyst,
                        news_headline=headline,
                        news_reason=news_reason,
                        news_category=news_category,
                    )

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
                    float_str = f"{h['float_shares'] / 1_000_000:.1f}M" if h.get('float_shares') else "N/A"
                    print(
                        f"    {h['symbol']:<6} "
                        f"${h['prev_close']:.2f} -> ${h['current_price']:.2f} "
                        f"({h['change_pct']:+.1f}%)  "
                        f"RelVol: {h['relative_volume']:.1f}x "
                        f"(vol: {h['current_volume']:,} / avg: {h['avg_volume']:,})  "
                        f"Float: {float_str}  "
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

    def _log_throttled_bar_drain_error(self, exc: Exception) -> None:
        """
        Rate-limit MACD bar-drain error logs.

        First occurrence of a given error signature logs at ERROR with stack;
        repeats of the SAME signature within 60s are counted silently; every
        60s we emit a throttle summary (N more of the same error).
        """
        sig = f"{type(exc).__name__}: {exc}"
        now = time_mod.time()
        if sig != self._macd_bar_drain_last_err:
            # New error — log full detail once.
            logger.error(f"MACD wave bar-drain error: {exc}", exc_info=True)
            self._macd_bar_drain_last_err = sig
            self._macd_bar_drain_last_err_ts = now
            self._macd_bar_drain_err_count = 1
            return
        self._macd_bar_drain_err_count += 1
        if now - self._macd_bar_drain_last_err_ts >= 60.0:
            logger.error(
                f"MACD wave bar-drain error (repeating, {self._macd_bar_drain_err_count} "
                f"occurrences in last ~60s): {sig}"
            )
            self._macd_bar_drain_last_err_ts = now
            self._macd_bar_drain_err_count = 0

    def _interruptible_sleep(self, seconds: float) -> bool:
        """Sleep for up to `seconds`, returning True if shutdown requested."""
        if self.shutdown_event is not None:
            triggered = self.shutdown_event.wait(timeout=seconds)
            if triggered:
                logger.info("Sleep interrupted by shutdown signal")
            return triggered
        else:
            time_mod.sleep(seconds)
            return False

    def _sleep_until(self, target_time: str) -> bool:
        """Sleep until a target Eastern Time (HH:MM).

        Returns:
            True if shutdown was requested during sleep, False otherwise.
        """
        now_et = datetime.now(ET)
        target_h, target_m = map(int, target_time.split(':'))
        target = now_et.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
        if target <= now_et:
            return False
        sleep_secs = (target - now_et).total_seconds()
        logger.info(f"Sleeping {sleep_secs:.0f}s until {target_time} ET")
        return self._interruptible_sleep(sleep_secs)

    def _is_trading_day(self) -> bool:
        """
        Check if today is a full trading day via Alpaca calendar API.

        Skips:
        - Weekends and market holidays (not a trading day)
        - Short trading days (closes before 16:00 ET, e.g. Black Friday)

        Returns:
            True if today is a normal full trading day
        """
        try:
            if not self.alpaca.is_trading_day():
                now_et = datetime.now(ET)
                logger.info(
                    f"Not a trading day ({now_et.strftime('%A %Y-%m-%d')}). "
                    f"Skipping scanner."
                )
                return False

            if self.alpaca.is_short_trading_day():
                logger.warning(
                    "Short trading day detected — insufficient hours for "
                    "momentum strategy. Skipping."
                )
                return False

            return True
        except Exception as e:
            logger.error(
                f"Failed to check trading calendar: {e}. "
                f"Proceeding with caution."
            )
            # If we can't check, let the scanner run —
            # it will just get no data on non-trading days
            return True
