"""
OneMil Day Trading Scanner - Entry Point.

Usage:
    python main.py --batch              Run nightly universe builder
    python main.py --scan               Run real-time scanner
    python main.py --scan --verbose     Run scanner with verbose output

Requires .env with ALPACA_API_KEY and ALPACA_API_SECRET.
"""

import argparse
import signal
import sys
import logging
import threading
from typing import Optional

from config import get_config
from monitoring.logger import setup_logging
from persistence.database import get_database
from data_sources.alpaca_client import AlpacaClient
from data_sources.float_provider import FloatProvider
from data_sources.news_provider import NewsProvider, NewsAnalyzer, LLMNewsAnalyzer
from batch.universe_builder import UniverseBuilder
from scanner.criteria import ScannerCriteria
from scanner.realtime_scanner import RealtimeScanner
from trading.pattern_detector import BullFlagDetector
from trading.trade_planner import TradePlanner
from trading.order_executor import OrderExecutor
from trading.position_manager import PositionManager
from trading.trading_engine import TradingEngine
from notifications.telegram_notifier import TelegramNotifier
from monitoring.telegram_error_handler import TelegramErrorHandler

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OneMil Day Trading Scanner - Momentum stock scanner"
    )
    parser.add_argument(
        '--batch', action='store_true',
        help='Run nightly universe builder (fetch assets, filter, cache volumes)'
    )
    parser.add_argument(
        '--scan', action='store_true',
        help='Run real-time scanner (pre-market gaps + intraday volume/move)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose output (debug logging + detailed scan output)'
    )
    parser.add_argument(
        '--trade', action='store_true',
        help='Enable automated paper trading (requires --scan or --test-cycle)'
    )
    parser.add_argument(
        '--test-cycle', action='store_true',
        help='Run one premarket + one intraday cycle with real data, then exit'
    )
    return parser.parse_args()


def _create_news_analyzer(config) -> NewsAnalyzer:
    """
    Create the appropriate NewsAnalyzer based on available API keys.

    Returns LLMNewsAnalyzer if ANTHROPIC_API_KEY is set, else V1 stub.
    """
    if config.anthropic_api_key:
        import anthropic
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        model = config.news_analyzer_model
        logger.info(f"News analysis: LLM (model={model})")
        return LLMNewsAnalyzer(client, model=model)
    else:
        logger.warning(
            "ANTHROPIC_API_KEY not set — using V1 stub (all news = True)"
        )
        return NewsAnalyzer()


def run_batch(config) -> None:
    """Run the nightly universe builder."""
    logger.info("Starting batch universe builder...")

    alpaca = AlpacaClient(config.alpaca_api_key, config.alpaca_api_secret)
    if not alpaca.test_connection():
        logger.error("Alpaca API connection failed. Aborting batch.")
        sys.exit(1)

    float_provider = FloatProvider()
    db = get_database(config.db_path)

    builder = UniverseBuilder(
        alpaca_client=alpaca,
        float_provider=float_provider,
        db=db,
        price_min=config.price_min,
        price_max=config.price_max,
        float_max=config.float_max,
        volume_profile_days=config.volume_profile_days,
        float_cache_refresh_days=config.float_cache_refresh_days,
    )

    summary = builder.build()
    logger.info(f"Batch complete: {summary}")


def _create_notifier(config) -> Optional[TelegramNotifier]:
    """Create Telegram notifier if configured."""
    if not config.telegram_enabled:
        logger.info("Telegram notifications disabled")
        return None

    if not config.telegram_bot_token or not config.telegram_chat_id:
        logger.warning("Telegram enabled but token/chat_id missing in .env")
        return None

    notifier = TelegramNotifier(
        bot_token=config.telegram_bot_token,
        chat_id=config.telegram_chat_id,
        enabled=True,
    )
    logger.info("Telegram notifier created")
    return notifier


def _setup_telegram_error_handler(config) -> None:
    """Add Telegram error handler to root logger if configured."""
    if not config.telegram_enabled or not config.telegram_bot_token:
        return

    handler = TelegramErrorHandler(
        bot_token=config.telegram_bot_token,
        chat_id=config.telegram_chat_id,
    )
    logging.getLogger().addHandler(handler)
    logger.info("Telegram error handler attached to root logger")


def _create_trading_engine(config, alpaca, db, notifier=None) -> TradingEngine:
    """Create the trading engine with all components wired up."""
    from trading.market_regime import MarketRegimeFilter

    market_regime = MarketRegimeFilter(
        enabled=config.market_regime_enabled,
        spy_5d_return_min=config.market_regime_spy_5d_return_min,
    )

    detector = BullFlagDetector(
        min_pole_candles=config.min_pole_candles,
        min_pole_gain_pct=config.min_pole_gain_pct,
        max_retracement_pct=config.max_retracement_pct,
        max_pullback_candles=config.max_pullback_candles,
        min_breakout_volume_ratio=config.min_breakout_volume_ratio,
        require_macd_positive=config.require_macd_positive,
    )
    planner = TradePlanner(
        position_size_dollars=config.position_size_dollars,
        max_shares=config.max_shares,
        max_risk_per_share=config.max_risk_per_share,
        min_risk_per_share=config.min_risk_per_share,
        min_risk_reward=config.min_risk_reward,
        sizing_mode=config.sizing_mode,
        risk_per_trade=config.risk_per_trade,
        min_risk_pct=config.min_risk_pct,
        max_risk_pct=config.max_risk_pct,
    )
    position_manager = PositionManager(
        alpaca_client=alpaca,
        db=db,
        max_positions=config.max_positions,
        daily_loss_limit=config.daily_loss_limit,
        stop_trading_before_close_min=config.stop_trading_before_close_min,
        circuit_breaker_dd=config.circuit_breaker_dd,
        circuit_breaker_pause=config.circuit_breaker_pause,
    )
    executor = OrderExecutor(alpaca_client=alpaca, db=db)

    engine = TradingEngine(
        alpaca_client=alpaca,
        db=db,
        detector=detector,
        planner=planner,
        executor=executor,
        position_manager=position_manager,
        pattern_poll_interval=config.pattern_poll_interval,
        enabled=config.trading_enabled,
        notifier=notifier,
        setup_expiry_seconds=config.setup_expiry_bars * config.pattern_poll_interval,
        market_regime=market_regime,
    )

    # Load SPY data immediately so regime is ready if service starts mid-day
    engine._refresh_spy_data()

    logger.info(
        f"Trading engine created — enabled: {config.trading_enabled}, "
        f"position_size: ${config.position_size_dollars}, "
        f"max_positions: {config.max_positions}, "
        f"regime_filter: {config.market_regime_enabled}"
    )
    return engine


def run_scan(config, verbose: bool = False, trade: bool = False) -> None:
    """Run the real-time scanner."""
    logger.info("Starting real-time scanner...")

    alpaca = AlpacaClient(config.alpaca_api_key, config.alpaca_api_secret)
    if not alpaca.test_connection():
        logger.error("Alpaca API connection failed. Aborting scan.")
        sys.exit(1)

    # Fix 10: Pre-start validation
    if trade:
        try:
            account = alpaca.get_account_info()
            # Verify paper mode — refuse to start on live account
            if not account.get('paper', True):
                logger.error("REFUSING TO START: Alpaca account is LIVE, not paper!")
                sys.exit(1)
            # Warn on low buying power
            buying_power = float(account.get('buying_power', 0))
            if buying_power < config.position_size_dollars:
                logger.warning(
                    f"Low buying power: ${buying_power:,.0f} < "
                    f"position size ${config.position_size_dollars:,.0f}"
                )
            logger.info(
                f"Account validated — paper mode, "
                f"buying power: ${buying_power:,.0f}"
            )
        except Exception as e:
            logger.error(f"Pre-start account validation failed: {e}")
            sys.exit(1)

    analyzer = _create_news_analyzer(config)
    news_provider = NewsProvider(alpaca, analyzer)
    db = get_database(config.db_path)
    notifier = _create_notifier(config)

    criteria = ScannerCriteria(
        price_min=config.price_min,
        price_max=config.price_max,
        float_max=config.float_max,
        gap_pct_min=config.gap_pct_min,
        intraday_change_pct_min=config.intraday_change_pct_min,
        relative_volume_min=config.relative_volume_min,
    )

    trading_engine = None
    if trade:
        trading_engine = _create_trading_engine(config, alpaca, db, notifier=notifier)
        trading_engine.enabled = True
        logger.info("Trading mode ACTIVE — paper trading enabled")

    # Fix 4: Graceful shutdown via SIGTERM/SIGINT
    shutdown_event = threading.Event()

    def handle_shutdown(signum, frame):
        """Handle shutdown signals for graceful position close."""
        sig_name = signal.Signals(signum).name
        logger.warning(f"Received {sig_name}, initiating graceful shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    if trading_engine:
        trading_engine.shutdown_event = shutdown_event

    scanner = RealtimeScanner(
        alpaca_client=alpaca,
        news_provider=news_provider,
        db=db,
        criteria=criteria,
        poll_interval=config.premarket_poll_interval,
        verbose=verbose,
        trading_engine=trading_engine,
        notifier=notifier,
    )

    # Notify startup
    if notifier:
        notifier.notify_scanner_started(
            universe_size=len(scanner._universe) if scanner._universe else 0,
            trading_enabled=trade,
        )

    scanner.run()

    # End-of-day report + summary
    if trading_engine:
        trading_engine.send_daily_report(
            premarket_gaps=scanner._premarket_gap_data,
            qualified_stocks=scanner._qualified_stock_data,
            universe_size=len(scanner._universe),
        )
        trading_engine.save_daily_summary()


def run_test_cycle(config, trade: bool = False) -> None:
    """Run a single test cycle (premarket + intraday) against real API."""
    logger.info("Starting test cycle...")

    alpaca = AlpacaClient(config.alpaca_api_key, config.alpaca_api_secret)
    if not alpaca.test_connection():
        logger.error("Alpaca API connection failed. Aborting test.")
        sys.exit(1)

    analyzer = _create_news_analyzer(config)
    news_provider = NewsProvider(alpaca, analyzer)
    db = get_database(config.db_path)
    notifier = _create_notifier(config)

    criteria = ScannerCriteria(
        price_min=config.price_min,
        price_max=config.price_max,
        float_max=config.float_max,
        gap_pct_min=config.gap_pct_min,
        intraday_change_pct_min=config.intraday_change_pct_min,
        relative_volume_min=config.relative_volume_min,
    )

    trading_engine = None
    if trade:
        trading_engine = _create_trading_engine(config, alpaca, db, notifier=notifier)
        trading_engine.enabled = True
        logger.info("Trading mode ACTIVE for test cycle — paper trading enabled")

    scanner = RealtimeScanner(
        alpaca_client=alpaca,
        news_provider=news_provider,
        db=db,
        criteria=criteria,
        poll_interval=60,
        verbose=True,
        trading_engine=trading_engine,
        notifier=notifier,
    )

    summary = scanner.run_test_cycle()
    logger.info(f"Test cycle complete: {summary}")

    if trading_engine:
        trading_engine.run_pattern_check()
        stats = trading_engine.get_daily_stats()
        logger.info(f"Trading stats: {stats}")
        trading_engine.send_daily_report(
            premarket_gaps=scanner._premarket_gap_data,
            qualified_stocks=scanner._qualified_stock_data,
            universe_size=len(scanner._universe),
        )
        trading_engine.save_daily_summary()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if not args.batch and not args.scan and not args.test_cycle:
        print("Error: specify --batch, --scan, or --test-cycle")
        print("Run 'python main.py --help' for usage")
        sys.exit(1)

    config = get_config()
    setup_logging(
        log_level=config.log_level,
        verbose=args.verbose,
    )
    _setup_telegram_error_handler(config)

    logger.info("OneMil Scanner starting...")

    if args.batch:
        run_batch(config)

    if args.scan:
        run_scan(config, verbose=args.verbose, trade=args.trade)

    if args.test_cycle:
        run_test_cycle(config, trade=args.trade)


if __name__ == "__main__":
    main()
