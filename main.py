"""
OneMil Day Trading Scanner - Entry Point.

Usage:
    python main.py --batch              Run nightly universe builder
    python main.py --scan               Run real-time scanner
    python main.py --scan --verbose     Run scanner with verbose output

Requires .env with ALPACA_API_KEY and ALPACA_API_SECRET.
"""

import argparse
import sys
import logging

from config import get_config
from monitoring.logger import setup_logging
from persistence.database import get_database
from data_sources.alpaca_client import AlpacaClient
from data_sources.float_provider import FloatProvider
from data_sources.news_provider import NewsProvider, NewsAnalyzer
from batch.universe_builder import UniverseBuilder
from scanner.criteria import ScannerCriteria
from scanner.realtime_scanner import RealtimeScanner

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
    return parser.parse_args()


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


def run_scan(config, verbose: bool = False) -> None:
    """Run the real-time scanner."""
    logger.info("Starting real-time scanner...")

    alpaca = AlpacaClient(config.alpaca_api_key, config.alpaca_api_secret)
    if not alpaca.test_connection():
        logger.error("Alpaca API connection failed. Aborting scan.")
        sys.exit(1)

    news_provider = NewsProvider(alpaca, NewsAnalyzer())
    db = get_database(config.db_path)

    criteria = ScannerCriteria(
        price_min=config.price_min,
        price_max=config.price_max,
        float_max=config.float_max,
        gap_pct_min=config.gap_pct_min,
        intraday_change_pct_min=config.intraday_change_pct_min,
        relative_volume_min=config.relative_volume_min,
    )

    scanner = RealtimeScanner(
        alpaca_client=alpaca,
        news_provider=news_provider,
        db=db,
        criteria=criteria,
        poll_interval=config.premarket_poll_interval,
        verbose=verbose,
    )

    scanner.run()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if not args.batch and not args.scan:
        print("Error: specify --batch or --scan (or both)")
        print("Run 'python main.py --help' for usage")
        sys.exit(1)

    config = get_config()
    setup_logging(
        log_level=config.log_level,
        verbose=args.verbose,
    )

    logger.info("OneMil Scanner starting...")

    if args.batch:
        run_batch(config)

    if args.scan:
        run_scan(config, verbose=args.verbose)


if __name__ == "__main__":
    main()
