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
        '--batch', '--rebuild-universe', action='store_true', dest='batch',
        help='Rebuild universe (fetch assets, filter by price/float/volume, cache profiles)'
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
    parser.add_argument(
        '--flag', action='store_true',
        help='Enable bull flag strategy (default: both strategies enabled)'
    )
    parser.add_argument(
        '--macd', action='store_true',
        help='Enable MACD wave strategy (default: both strategies enabled)'
    )
    parser.add_argument(
        '--orb', action='store_true',
        help='Enable ORB (Opening Range Breakout) strategy. Reads orb.yaml.'
             ' OFF by default — requires ALPACA_ORB_API_KEY/SECRET in .env and'
             ' strategy.enabled=true in orb.yaml to actually trade.'
    )
    args = parser.parse_args()
    # If neither --flag nor --macd specified, enable both (legacy default).
    # --orb stays opt-in even when unspecified.
    if not args.flag and not args.macd:
        args.flag = True
        args.macd = True
    return args


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

    alpaca = AlpacaClient(config.alpaca_api_key, config.alpaca_api_secret, paper=config.alpaca_paper)
    if not alpaca.test_connection():
        logger.error("Alpaca API connection failed. Aborting batch.")
        sys.exit(1)

    float_provider = FloatProvider()
    db = get_database(db_path=config.db_path, cache_path=config.cache_db_path, trades_path=config.trades_db_path)

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


def _create_stop_monitor(config, alpaca, notifier=None, alpaca_clients_by_strategy=None):
    """Create and start the shared StopMonitor (one per process).

    Args:
        alpaca_clients_by_strategy: optional dict mapping strategy tag → AlpacaClient.
            When set, exit orders for watches with matching `strategy` are submitted
            via the mapped client (enables multi-account operation). Legacy callers
            leave this None → all exits use the default `alpaca` client.
    """
    if not config.self_managed_stops_enabled:
        return None
    from trading.stop_monitor import StopMonitor
    # Paper accounts use REST polling (no WebSocket) to avoid conflict
    # with live account's WebSocket on shared Alpaca account
    # Paper CAN use WebSocket SIP — only force polling if live+paper share the same WS slot
    # For single-node paper trading, WebSocket gives real-time stops + instant bar detection
    use_polling = False
    stop_monitor = StopMonitor(
        api_key=config.alpaca_api_key,
        api_secret=config.alpaca_api_secret,
        alpaca_client=alpaca,
        marketable_limit_offset=config.marketable_limit_offset,
        marketable_limit_offset_pct=config.marketable_limit_offset_pct,
        notifier=notifier,
        polling_mode=use_polling,
        polling_interval=2.0,
        alpaca_clients_by_strategy=alpaca_clients_by_strategy,
    )
    stop_monitor.start()
    mode = "REST polling (paper)" if use_polling else "WebSocket (live)"
    logger.info(
        f"StopMonitor STARTED ({mode}) — safety_net={config.safety_net_sl_pct:.0%}, "
        f"offset=${config.marketable_limit_offset}, "
        f"offset_pct={config.marketable_limit_offset_pct:.1%}"
    )
    return stop_monitor


def _create_trading_engine(config, alpaca, db, notifier=None, stop_monitor=None, order_stream=None) -> TradingEngine:
    """Create the trading engine with all components wired up."""
    from trading.market_regime import MarketRegimeFilter

    _regime_cfg = config._load_yaml_only().get("trading", {}).get("market_regime", {})
    market_regime = MarketRegimeFilter(
        enabled=config.market_regime_enabled,
        vol_threshold=config.market_regime_vol_threshold,
        sma_period=config.market_regime_sma_period,
        max_trades_per_day=config.max_trades_per_day,
        min_spy_volume_ratio=config.market_regime_min_spy_volume_ratio,
        thin_liquidity_breakout_vol_ratio=config.market_regime_thin_liquidity_breakout_vol_ratio,
        sma_slope_filter=bool(_regime_cfg.get("sma_slope_filter", False)),
        sma_slope_threshold=float(_regime_cfg.get("sma_slope_threshold", -0.5)),
        euphoria_filter=bool(_regime_cfg.get("euphoria_filter", False)),
        euphoria_ud_threshold=float(_regime_cfg.get("euphoria_ud_threshold", 1.2)),
        euphoria_rsi_threshold=float(_regime_cfg.get("euphoria_rsi_threshold", 60.0)),
    )

    detector = BullFlagDetector(
        min_pole_candles=config.min_pole_candles,
        min_pole_gain_pct=config.min_pole_gain_pct,
        max_retracement_pct=config.max_retracement_pct,
        max_pullback_candles=config.max_pullback_candles,
        min_breakout_volume_ratio=config.min_breakout_volume_ratio,
        require_macd_positive=config.require_macd_positive,
        max_green_in_flag=config.max_green_in_flag,
        max_pole_bars=config.max_pole_bars,
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
        max_consecutive_losses=config.max_consecutive_losses,
    )
    executor = OrderExecutor(alpaca_client=alpaca, db=db, order_stream=order_stream)

    # Load time controls from config (backtest loads these too — must match)
    _trading_cfg = config._load_yaml_only().get("trading", {})
    _last_entry = _trading_cfg.get("last_entry_time", "15:00")
    _force_close = _trading_cfg.get("force_close_time", "15:45")

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
        last_entry_time_et=_last_entry,
        force_close_time_et=_force_close,
        setup_expiry_seconds=config.setup_expiry_bars * 60,  # bars are 1-min, not poll intervals
        market_regime=market_regime,
        stop_monitor=stop_monitor,
        safety_net_sl_pct=config.safety_net_sl_pct,
        order_stream=order_stream,
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


def run_scan(config, verbose: bool = False, trade: bool = False,
             enable_flag: bool = True, enable_macd: bool = True,
             enable_orb: bool = False) -> None:
    """Run the real-time scanner with one or more strategies."""
    logger.info("Starting real-time scanner...")

    alpaca = AlpacaClient(config.alpaca_api_key, config.alpaca_api_secret, paper=config.alpaca_paper)
    if not alpaca.test_connection():
        logger.error("Alpaca API connection failed. Aborting scan.")
        sys.exit(1)

    # Fix 10: Pre-start validation
    if trade:
        try:
            account = alpaca.get_account_info()
            # Verify paper/live mode matches config
            if not alpaca.is_paper:
                if config.alpaca_paper:
                    logger.error("REFUSING TO START: Alpaca account is LIVE but ALPACA_PAPER=true! Fix .env")
                    sys.exit(1)
                logger.warning("LIVE TRADING MODE — real money at risk")
            # Warn on low buying power
            buying_power = float(account.get('buying_power', 0))
            if buying_power < config.position_size_dollars:
                logger.warning(
                    f"Low buying power: ${buying_power:,.0f} < "
                    f"position size ${config.position_size_dollars:,.0f}"
                )
            mode_str = "paper" if alpaca.is_paper else "LIVE"
            logger.info(
                f"Account validated — {mode_str} mode, "
                f"buying power: ${buying_power:,.0f}"
            )
        except Exception as e:
            logger.error(f"Pre-start account validation failed: {e}")
            sys.exit(1)

    analyzer = _create_news_analyzer(config)
    news_provider = NewsProvider(alpaca, analyzer)
    db = get_database(db_path=config.db_path, cache_path=config.cache_db_path, trades_path=config.trades_db_path)
    notifier = _create_notifier(config)

    criteria = ScannerCriteria(
        price_min=config.price_min,
        price_max=config.price_max,
        float_max=config.float_max,
        gap_pct_min=config.gap_pct_min,
        intraday_change_pct_min=config.intraday_change_pct_min,
        relative_volume_min=config.relative_volume_min,
        require_news=config.require_news,
        min_dollar_volume=config.min_dollar_volume,
    )

    # --- ORB paper AlpacaClient (Phase 1) — created BEFORE StopMonitor so we can
    # route ORB exit-order execution to this separate paper account via the
    # StopMonitor's alpaca_clients_by_strategy dict. When we eventually migrate
    # ORB to the main account, this block goes away (dict becomes empty, legacy
    # path takes over). ---
    orb_alpaca = None
    if trade and enable_orb:
        if not config.alpaca_orb_api_key or not config.alpaca_orb_api_secret:
            logger.warning(
                "--orb requested but ALPACA_ORB_API_KEY/SECRET are empty. "
                "ORB will be DISABLED for this session. Set creds in .env first."
            )
            enable_orb = False
        else:
            try:
                orb_alpaca = AlpacaClient(
                    config.alpaca_orb_api_key,
                    config.alpaca_orb_api_secret,
                    paper=config.alpaca_orb_paper,
                )
                if not orb_alpaca.test_connection():
                    raise RuntimeError("ORB Alpaca connection test failed")
                orb_account = orb_alpaca.get_account_info()
                orb_mode = "paper" if orb_alpaca.is_paper else "LIVE"
                logger.info(
                    f"ORB Alpaca client connected — {orb_mode} mode, "
                    f"buying power: ${float(orb_account.get('buying_power', 0)):,.0f}"
                )
            except Exception as e:
                logger.error(f"ORB Alpaca client init failed: {e} — disabling ORB")
                orb_alpaca = None
                enable_orb = False

    # Create ONE shared StopMonitor for all strategies. If ORB has its own
    # AlpacaClient (paper account), pass the routing dict so ORB exit orders
    # go to the paper account — bull flag + MACD wave continue using `alpaca`
    # (main account) by default.
    stop_monitor = None
    if trade:
        orb_clients_dict = {'orb': orb_alpaca} if orb_alpaca is not None else None
        stop_monitor = _create_stop_monitor(
            config, alpaca, notifier,
            alpaca_clients_by_strategy=orb_clients_dict,
        )

    # T3.1 / S1: shared OrderStreamWatcher — one TradingStream for both strategies.
    # MUST be created BEFORE trading_engine so the bull flag engine can consume
    # push-delivered order status via _get_order_hybrid (replaces REST polling
    # at 8 hot-path fill-detection sites).
    order_stream = None
    if trade and stop_monitor is not None:
        try:
            from trading.order_stream import OrderStreamWatcher
            order_stream = OrderStreamWatcher(
                api_key=config.alpaca_api_key,
                api_secret=config.alpaca_api_secret,
                paper=alpaca.is_paper,
                alpaca_client=alpaca,
            )
            order_stream.start()
            logger.info("OrderStreamWatcher STARTED — fill detection via TradingStream")
        except Exception as e:
            logger.warning(f"OrderStreamWatcher failed to start (fallback to REST polling): {e}")
            order_stream = None

    # Bull flag engine (optional)
    trading_engine = None
    if trade and enable_flag:
        trading_engine = _create_trading_engine(config, alpaca, db, notifier=notifier,
                                                 stop_monitor=stop_monitor,
                                                 order_stream=order_stream)
        trading_engine.enabled = True
        trading_engine.news_provider = news_provider  # For news re-check at trade time
        # Register real-time bar handler (multi-consumer since Step 1) for instant pattern detection
        if stop_monitor and not stop_monitor.polling_mode:
            stop_monitor.register_bar_handler('bull_flag', trading_engine._on_bar_close)
            logger.info("Real-time bar stream → instant pattern detection ENABLED")
        logger.info(f"Bull Flag strategy ENABLED")

    # MACD wave engine (optional)
    macd_engine = None
    if trade and enable_macd:
        import yaml
        from trading.macd_wave_engine import MACDWaveEngine
        macd_cfg = yaml.safe_load(open('macd_wave.yaml'))
        macd_engine = MACDWaveEngine(
            alpaca_client=alpaca, db=db, notifier=notifier,
            config=macd_cfg, stop_monitor=stop_monitor,
            order_stream=order_stream,
        )
        # T1.1: register bar handler on the shared StopMonitor so bar closes
        # for crossed_stocks flow into MACD's event queue (drained by scanner's
        # 1s sleep chunks for targeted check_entries).
        if stop_monitor is not None and not stop_monitor.polling_mode:
            macd_engine.register_on_stop_monitor()
            logger.info("MACD Wave: bar handler registered on shared StopMonitor")
        # Restart-safe: reconcile positions (DB ↔ Alpaca) then rebuild intraday
        # state. `_sync_intraday_state` is a no-op before 9:30 or after 15:45 ET
        # but on mid-day restart it repopulates `universe_opens`, rebuilds
        # `crossed_stocks` from historical 1-min bars, and replays closed
        # trades from DB into `daily_pnl` / `trades_today`. Without this a
        # restart after 9:40 ET permanently loses the day's cross events.
        try:
            macd_engine.sync_positions()
        except Exception as e:
            logger.error(f"MACD Wave: sync_positions failed at startup: {e}")
        try:
            macd_engine._sync_intraday_state()
        except Exception as e:
            logger.error(f"MACD Wave: _sync_intraday_state failed at startup: {e}")
        logger.info(f"MACD Wave strategy ENABLED")

    # ORB (Opening Range Breakout) engine — runs on separate paper Alpaca
    # account in Phase 1. Fires at 9:35 ET. See orb.yaml for tuning.
    orb_engine = None
    if trade and enable_orb and orb_alpaca is not None:
        import yaml
        from trading.orb_engine import ORBEngine
        from trading.order_stream import OrderStreamWatcher
        orb_cfg = yaml.safe_load(open('orb.yaml'))
        # ORB has its own OrderStreamWatcher on the paper account — order events
        # are account-specific (unlike market data). Must be started separately.
        orb_order_stream = None
        try:
            orb_order_stream = OrderStreamWatcher(
                api_key=config.alpaca_orb_api_key,
                api_secret=config.alpaca_orb_api_secret,
                paper=orb_alpaca.is_paper,
                alpaca_client=orb_alpaca,
            )
            orb_order_stream.start()
            logger.info("ORB OrderStreamWatcher STARTED — paper account")
        except Exception as e:
            logger.warning(f"ORB OrderStreamWatcher failed to start: {e}")
            orb_order_stream = None
        orb_engine = ORBEngine(
            alpaca_client=orb_alpaca, db=db, notifier=notifier,
            config=orb_cfg, stop_monitor=stop_monitor,
            order_stream=orb_order_stream,
        )
        if stop_monitor is not None and not stop_monitor.polling_mode:
            orb_engine.register_on_stop_monitor()
            logger.info("ORB: bar handler registered on shared StopMonitor")
        # Restart-safe: rehydrate any open ORB positions on the paper account.
        orb_engine.sync_positions()
        logger.info(
            f"ORB strategy ENABLED — "
            f"master_flag={orb_engine.enabled}, dry_run={orb_engine.dry_run}"
        )

    mode_label = "paper" if alpaca.is_paper else "LIVE"
    strategies = []
    if enable_flag:
        strategies.append("Bull Flag")
    if enable_macd:
        strategies.append("MACD Wave")
    if enable_orb and orb_engine is not None:
        strategies.append("ORB (paper)")
    logger.info(f"Trading mode ACTIVE — {mode_label}, strategies: {', '.join(strategies)}")

    # Fix 4: Graceful shutdown via SIGTERM/SIGINT
    shutdown_event = threading.Event()

    def handle_shutdown(signum, frame):
        """Handle shutdown signals for graceful position close."""
        sig_name = signal.Signals(signum).name
        logger.warning(f"Received {sig_name}, initiating graceful shutdown...")
        shutdown_event.set()
        if macd_engine:
            macd_engine.shutdown_requested = True
        if orb_engine:
            orb_engine.shutdown_requested = True

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
        shutdown_event=shutdown_event,
        macd_engine=macd_engine,
        orb_engine=orb_engine,
    )

    # Enable async news classification (non-blocking LLM calls)
    scanner.enable_async_news()

    # Notify startup
    if notifier:
        notifier.notify_scanner_started(
            universe_size=len(scanner._universe) if scanner._universe else 0,
            trading_enabled=trade,
            mode=mode_label,
        )

    scanner.run()

    # End-of-day reports
    if trading_engine:
        trading_engine.send_daily_report(
            premarket_gaps=scanner._premarket_gap_data,
            qualified_stocks=scanner._qualified_stock_data,
            universe_size=len(scanner._universe),
        )
        trading_engine.save_daily_summary()
    if macd_engine:
        macd_engine.send_daily_report()

    # Close shared WebSockets
    if order_stream:
        try:
            order_stream.stop()
        except Exception as e:
            logger.warning(f"order_stream.stop() error: {e}")
    if stop_monitor:
        stop_monitor.stop()


def run_test_cycle(config, trade: bool = False) -> None:
    """Run a single test cycle (premarket + intraday) against real API."""
    logger.info("Starting test cycle...")

    alpaca = AlpacaClient(config.alpaca_api_key, config.alpaca_api_secret, paper=config.alpaca_paper)
    if not alpaca.test_connection():
        logger.error("Alpaca API connection failed. Aborting test.")
        sys.exit(1)

    analyzer = _create_news_analyzer(config)
    news_provider = NewsProvider(alpaca, analyzer)
    db = get_database(db_path=config.db_path, cache_path=config.cache_db_path, trades_path=config.trades_db_path)
    notifier = _create_notifier(config)

    criteria = ScannerCriteria(
        price_min=config.price_min,
        price_max=config.price_max,
        float_max=config.float_max,
        gap_pct_min=config.gap_pct_min,
        intraday_change_pct_min=config.intraday_change_pct_min,
        relative_volume_min=config.relative_volume_min,
        require_news=config.require_news,
        min_dollar_volume=config.min_dollar_volume,
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
        print("Error: specify --rebuild-universe (or --batch), --scan, or --test-cycle")
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
        run_scan(config, verbose=args.verbose, trade=args.trade,
                 enable_flag=args.flag, enable_macd=args.macd,
                 enable_orb=args.orb)

    if args.test_cycle:
        run_test_cycle(config, trade=args.trade)


if __name__ == "__main__":
    main()
