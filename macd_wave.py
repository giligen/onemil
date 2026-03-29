"""
MACD Wave Trading Service.

Standalone service for MACD wave momentum strategy. Detects $15-30 stocks
that cross +10% intraday within 3 minutes of open, enters on MACD histogram
confirmation, exits on histogram flip or 2% hard stop.

Shares Alpaca paper account and DB with the bull flag service.
Does NOT interfere with main.py / trading_engine.py.

Usage:
    python macd_wave.py                  # Live trading
    python macd_wave.py --dry-run        # Monitor only, no orders
    python macd_wave.py --verbose        # Debug logging
"""

import argparse
import logging
import os
import signal
import sys
import time as time_mod
from datetime import datetime
from pathlib import Path

import pytz
import yaml
from dotenv import load_dotenv

load_dotenv()

ET = pytz.timezone('US/Eastern')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'macd_wave.yaml')

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for MACD wave service."""
    level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = []

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    root.addHandler(console)

    # File
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_dir / "macd_wave.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8',
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    root.addHandler(file_handler)

    # Suppress noisy loggers
    for name in ['websockets', 'httpcore', 'httpx', 'anthropic', 'urllib3']:
        logging.getLogger(name).setLevel(logging.WARNING)


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load MACD wave config."""
    with open(path) as f:
        return yaml.safe_load(f)


def create_notifier(config: dict):
    """Create Telegram notifier if configured."""
    try:
        from notifications.telegram_notifier import TelegramNotifier
        from config import get_config
        cfg = get_config()  # Reuse bull flag .env for Telegram keys
        if cfg.telegram_enabled and cfg.telegram_bot_token and cfg.telegram_chat_id:
            notifier = TelegramNotifier(
                bot_token=cfg.telegram_bot_token,
                chat_id=cfg.telegram_chat_id,
                enabled=True,
            )
            logger.info("Telegram notifier created")
            return notifier
    except Exception as e:
        logger.warning(f"Telegram notifier creation failed: {e}")
    return None


def wait_for_premarket() -> None:
    """Wait until 8:30 AM ET (1h before market open)."""
    while True:
        now = datetime.now(ET)
        target = now.replace(hour=8, minute=30, second=0, microsecond=0)
        if now >= target:
            break
        wait_seconds = (target - now).total_seconds()
        if wait_seconds > 300:
            logger.info(f"Waiting for pre-market (8:30 ET)... {wait_seconds/60:.0f}min remaining")
            time_mod.sleep(min(300, wait_seconds))
        else:
            time_mod.sleep(min(30, wait_seconds))


def main():
    parser = argparse.ArgumentParser(description="MACD Wave Trading Service")
    parser.add_argument("--dry-run", action="store_true", help="Monitor only, no orders")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="Config file path")
    parser.add_argument("--skip-wait", action="store_true", help="Skip pre-market wait (for testing)")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    config = load_config(args.config)

    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')
    if not api_key or not api_secret:
        logger.error("Missing ALPACA_API_KEY or ALPACA_API_SECRET")
        sys.exit(1)

    from data_sources.alpaca_client import AlpacaClient
    from persistence.database import get_database
    from trading.macd_wave_engine import MACDWaveEngine

    alpaca = AlpacaClient(api_key, api_secret)
    if not alpaca.test_connection():
        logger.error("Alpaca connection failed")
        sys.exit(1)

    # Verify paper mode
    if not alpaca.is_paper:
        logger.error("REFUSING TO START: Alpaca account is LIVE, not paper!")
        sys.exit(1)

    db = get_database()
    notifier = create_notifier(config)

    # StopMonitor for real-time trail stop + hard stop via SIP WebSocket
    # Separate instance from bull flag's StopMonitor (different process)
    from trading.stop_monitor import StopMonitor
    stop_monitor = StopMonitor(
        api_key=api_key,
        api_secret=api_secret,
        alpaca_client=alpaca,
        marketable_limit_offset=0.03,
        marketable_limit_offset_pct=0.005,
        notifier=notifier,
    )
    stop_monitor.start()
    logger.info(
        f"StopMonitor STARTED for MACD wave — "
        f"offset=${0.03}, offset_pct={0.005:.1%}"
    )

    engine = MACDWaveEngine(
        alpaca_client=alpaca,
        db=db,
        notifier=notifier,
        config=config,
        dry_run=args.dry_run,
        stop_monitor=stop_monitor,
    )

    # Graceful shutdown
    def handle_shutdown(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.warning(f"Received {sig_name}, shutting down...")
        engine.shutdown_requested = True

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    mode = "DRY RUN" if args.dry_run else "LIVE"
    logger.info(f"MACD Wave Service starting ({mode})")

    # Wait for pre-market
    if not args.skip_wait:
        wait_for_premarket()

    # Build universe
    engine.build_universe()

    # Wait for market open
    while not engine.is_market_open() and not engine.shutdown_requested:
        now_et = datetime.now(ET)
        logger.debug(f"Waiting for market open... {now_et.strftime('%H:%M ET')}")
        time_mod.sleep(30)

    if engine.shutdown_requested:
        logger.info("Shutdown before market open")
        return

    logger.info("Market open — starting intraday loop")

    # Intraday loop
    cycle = 0
    while engine.is_market_open() and not engine.shutdown_requested:
        cycle += 1
        t0 = time_mod.time()

        try:
            # Scan for new movers
            new_crosses = engine.scan_for_movers()

            # Check entry signals
            entries = engine.check_entries()

            # Check exit signals
            exits = engine.check_exits()

            # Force close check
            if engine.is_force_close_time():
                engine.force_close_all()
                break

            elapsed = time_mod.time() - t0
            if cycle % 10 == 0:
                logger.info(
                    f"Cycle {cycle}: {len(engine.crossed_stocks)} monitoring, "
                    f"{len(engine.open_positions)} open, "
                    f"daily P&L ${engine.daily_pnl:+,.0f}, "
                    f"{elapsed:.1f}s"
                )

        except Exception as e:
            logger.error(f"Cycle {cycle} error: {e}", exc_info=True)

        # Sleep until next cycle (target 60s intervals)
        elapsed = time_mod.time() - t0
        sleep_time = max(1, 60 - elapsed)
        time_mod.sleep(sleep_time)

    # End of day
    engine.force_close_all()  # In case we exited loop early
    engine.send_daily_report()
    logger.info("MACD Wave Service stopped")


if __name__ == "__main__":
    main()
