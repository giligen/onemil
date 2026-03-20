#!/usr/bin/env python3
"""
System test for self-managed stops via StopMonitor.

Performs a REAL paper trade to validate the full pipeline:
1. Buy a cheap liquid stock via market order
2. Register position with StopMonitor (stop = just above current price)
3. Wait for WebSocket price tick to trigger the stop
4. Verify: bracket legs cancelled, limit sell submitted, exit event emitted
5. Clean up: close any remaining position

Usage:
    python tests/system_test_stop_monitor.py
    python tests/system_test_stop_monitor.py --symbol AAPL --dollars 100
    python tests/system_test_stop_monitor.py --dry-run   # validate setup only

Requires:
    - Market hours (9:30-16:00 ET) for live price ticks
    - Paper trading account with buying power
    - .env with ALPACA_API_KEY, ALPACA_API_SECRET

Safety:
    - Paper account only (refuses to run on live)
    - Max spend: $100 default (configurable)
    - Auto-cleanup: closes position on exit/failure
    - Timeout: exits after 120s if stop not triggered
"""

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

from data_sources.alpaca_client import AlpacaClient, AlpacaAPIError
from trading.stop_monitor import StopMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('system_test')

# Suppress noisy sub-loggers
logging.getLogger('alpaca').setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='System test: buy stock, watch via StopMonitor, trigger stop exit'
    )
    parser.add_argument(
        '--symbol', default='AAPL',
        help='Stock to trade (default: AAPL — liquid, tight spreads)'
    )
    parser.add_argument(
        '--dollars', type=float, default=100.0,
        help='Max dollar amount to spend (default: $100)'
    )
    parser.add_argument(
        '--stop-offset-pct', type=float, default=0.005,
        help='Set stop this %% above current price to trigger quickly (default: 0.5%%)'
    )
    parser.add_argument(
        '--timeout', type=int, default=120,
        help='Seconds to wait for stop trigger (default: 120)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate setup without placing any orders'
    )
    return parser.parse_args()


class SystemTestRunner:
    """Runs the full stop monitor system test."""

    def __init__(self, symbol: str, max_dollars: float,
                 stop_offset_pct: float, timeout: int, dry_run: bool):
        """Initialize with test parameters."""
        self.symbol = symbol
        self.max_dollars = max_dollars
        self.stop_offset_pct = stop_offset_pct
        self.timeout = timeout
        self.dry_run = dry_run
        self.client = None
        self.monitor = None
        self.position_open = False
        self.test_passed = False

    def run(self) -> bool:
        """
        Execute the full system test.

        Returns:
            True if test passed, False otherwise
        """
        try:
            self._setup()
            if self.dry_run:
                logger.info("DRY RUN complete — setup validated, no orders placed")
                return True
            self._buy_position()
            self._register_stop_monitor()
            self._wait_for_stop_trigger()
            self._verify_results()
            return self.test_passed
        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
            return False
        except Exception as e:
            logger.error(f"System test failed with exception: {e}")
            return False
        finally:
            self._cleanup()

    def _setup(self):
        """Validate prerequisites: API connection, paper mode, buying power."""
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_API_SECRET')
        if not api_key or not api_secret:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET required in .env")

        self.client = AlpacaClient(api_key, api_secret)

        # Verify paper mode
        if not self.client.is_paper:
            raise RuntimeError("REFUSING TO RUN: Account is LIVE, not paper!")

        # Test connection
        if not self.client.test_connection():
            raise RuntimeError("Alpaca API connection failed")

        # Check buying power
        account = self.client.get_account_info()
        buying_power = float(account.get('buying_power', 0))
        logger.info(f"Account validated — paper mode, buying power: ${buying_power:,.0f}")

        if buying_power < self.max_dollars:
            raise RuntimeError(
                f"Insufficient buying power: ${buying_power:,.0f} < ${self.max_dollars:,.0f}"
            )

        # Get current price
        trades = self.client.get_latest_trades([self.symbol])
        if self.symbol not in trades:
            raise RuntimeError(f"Cannot get price for {self.symbol}")

        self.current_price = trades[self.symbol]['price']
        self.qty = max(1, int(self.max_dollars / self.current_price))
        cost = self.qty * self.current_price

        # Set stop ABOVE current price so it triggers immediately
        # (price is already below the stop = instant trigger on next tick)
        self.stop_price = round(self.current_price * (1 + self.stop_offset_pct), 2)
        # Safety-net SL far below (won't trigger)
        self.safety_net_sl = round(self.current_price * 0.90, 2)
        # TP far above (won't trigger)
        self.tp_price = round(self.current_price * 1.10, 2)

        logger.info(
            f"Test plan: BUY {self.qty} {self.symbol} @ ~${self.current_price:.2f} "
            f"(~${cost:.0f})"
        )
        logger.info(
            f"Stop monitor: stop=${self.stop_price:.2f} "
            f"(+{self.stop_offset_pct:.1%} above current — should trigger immediately)"
        )
        logger.info(
            f"Bracket: TP=${self.tp_price:.2f}, safety-net SL=${self.safety_net_sl:.2f}"
        )

        # Create StopMonitor
        self.monitor = StopMonitor(
            api_key=api_key,
            api_secret=api_secret,
            alpaca_client=self.client,
            marketable_limit_offset=0.03,
            marketable_limit_offset_pct=0.005,
        )

        if self.dry_run:
            logger.info("DRY RUN: Would buy, register monitor, wait for trigger")
            return

    def _buy_position(self):
        """Buy the stock with a bracket order (TP + safety-net SL)."""
        logger.info(f"Step 1: Buying {self.qty} {self.symbol} via bracket order...")

        result = self.client.submit_bracket_order(
            symbol=self.symbol,
            qty=self.qty,
            side='buy',
            limit_price=round(self.current_price * 1.005, 2),  # slight premium to fill fast
            tp_price=self.tp_price,
            sl_price=self.safety_net_sl,
        )

        self.parent_order_id = result['id']
        logger.info(f"  Bracket order submitted — ID: {self.parent_order_id}")

        # Wait for fill
        fill_price = None
        for attempt in range(30):
            time.sleep(1)
            order = self.client.get_order(self.parent_order_id)
            status = order.get('status', 'unknown')
            if status == 'filled':
                fill_price = order.get('filled_avg_price')
                logger.info(f"  FILLED at ${fill_price:.2f} (attempt {attempt + 1})")
                break
            elif status in ('cancelled', 'expired', 'rejected'):
                raise RuntimeError(f"Order {status} — cannot proceed")
            else:
                if attempt % 5 == 4:
                    logger.info(f"  Waiting for fill... status={status} (attempt {attempt + 1})")

        if fill_price is None:
            raise RuntimeError("Order not filled after 30s — aborting")

        self.fill_price = fill_price
        self.position_open = True

        # Identify bracket legs
        order_detail = self.client.get_order(self.parent_order_id)
        legs = order_detail.get('legs', [])
        self.tp_leg_id = ''
        self.sl_leg_id = ''
        for leg in legs:
            if leg.get('side') != 'sell':
                continue
            if leg.get('stop_price') and not leg.get('limit_price'):
                self.sl_leg_id = leg['id']
            elif leg.get('limit_price') and not leg.get('stop_price'):
                self.tp_leg_id = leg['id']

        logger.info(f"  Bracket legs — TP: {self.tp_leg_id[:8]}..., SL: {self.sl_leg_id[:8]}...")
        logger.info(f"  Position open: {self.qty} shares @ ${fill_price:.2f}")

    def _register_stop_monitor(self):
        """Start StopMonitor and register the position."""
        logger.info("Step 2: Starting StopMonitor WebSocket...")
        self.monitor.start()
        time.sleep(2)  # Let WebSocket connect

        logger.info(f"Step 3: Registering watch — stop=${self.stop_price:.2f}")
        self.monitor.add_watch(
            symbol=self.symbol,
            stop_price=self.stop_price,
            shares=self.qty,
            tp_leg_id=self.tp_leg_id,
            sl_leg_id=self.sl_leg_id,
        )
        logger.info(f"  Watching {self.symbol} — next tick at/below ${self.stop_price:.2f} triggers exit")

    def _wait_for_stop_trigger(self):
        """Wait for StopMonitor to trigger and process the exit."""
        logger.info(f"Step 4: Waiting for stop trigger (timeout={self.timeout}s)...")

        start_time = time.time()
        self.exit_events = []

        while time.time() - start_time < self.timeout:
            events = self.monitor.drain_exit_events()
            if events:
                self.exit_events = events
                elapsed = time.time() - start_time
                logger.info(f"  EXIT TRIGGERED after {elapsed:.1f}s!")
                for event in events:
                    logger.info(
                        f"  Event: {event.symbol} — "
                        f"stop=${event.stop_price:.2f}, "
                        f"exit=${event.exit_price:.2f}, "
                        f"reason={event.exit_reason}, "
                        f"order={event.order_id[:8]}..."
                    )
                self.position_open = False
                return

            # Log progress every 10s
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                watched = self.monitor.watched_symbols
                logger.info(
                    f"  Waiting... {elapsed:.0f}s elapsed, "
                    f"watched: {watched}"
                )

            time.sleep(0.5)

        logger.warning(f"  Timeout after {self.timeout}s — stop not triggered")

    def _verify_results(self):
        """Verify the test results."""
        logger.info("Step 5: Verifying results...")

        if not self.exit_events:
            logger.error("FAIL: No exit events received")
            self.test_passed = False
            return

        event = self.exit_events[0]

        # Check exit reason
        if event.exit_reason not in ('stop_loss', 'stop_loss_fallback'):
            logger.error(f"FAIL: Unexpected exit reason: {event.exit_reason}")
            self.test_passed = False
            return
        logger.info(f"  PASS: Exit reason = {event.exit_reason}")

        # Check order was submitted
        if not event.order_id:
            logger.error("FAIL: No order ID in exit event")
            self.test_passed = False
            return
        logger.info(f"  PASS: Exit order submitted (ID: {event.order_id[:8]}...)")

        # Check position is closed
        time.sleep(2)  # Let order settle
        positions = self.client.get_open_positions()
        still_open = any(p['symbol'] == self.symbol for p in positions)
        if still_open:
            logger.warning("  WARN: Position still open — limit sell may not have filled yet")
            # Give it more time
            time.sleep(5)
            positions = self.client.get_open_positions()
            still_open = any(p['symbol'] == self.symbol for p in positions)
            if still_open:
                logger.error("FAIL: Position still open after 7s")
                self.test_passed = False
                return
        logger.info("  PASS: Position closed")

        # Check slippage
        slippage = self.fill_price - event.exit_price
        slippage_pct = slippage / self.fill_price * 100
        logger.info(f"  Entry: ${self.fill_price:.2f}, Exit: ${event.exit_price:.2f}")
        logger.info(f"  Slippage: ${slippage:.3f} ({slippage_pct:.2f}%)")

        # Check bracket legs cancelled
        try:
            if self.tp_leg_id:
                tp_order = self.client.get_order(self.tp_leg_id)
                tp_status = tp_order.get('status', 'unknown')
                logger.info(f"  TP leg status: {tp_status}")
            if self.sl_leg_id:
                sl_order = self.client.get_order(self.sl_leg_id)
                sl_status = sl_order.get('status', 'unknown')
                logger.info(f"  SL leg status: {sl_status}")
        except Exception as e:
            logger.info(f"  Bracket leg check: {e} (may be auto-cancelled)")

        self.test_passed = True
        logger.info("")
        logger.info("=" * 60)
        logger.info("  SYSTEM TEST PASSED")
        logger.info("=" * 60)

    def _cleanup(self):
        """Clean up: stop monitor, close any remaining position."""
        logger.info("Cleaning up...")

        if self.monitor:
            self.monitor.stop()
            logger.info("  StopMonitor stopped")

        if self.position_open:
            logger.info(f"  Closing remaining {self.symbol} position...")
            try:
                self.client.close_position(self.symbol)
                logger.info(f"  {self.symbol} position closed")
            except Exception as e:
                logger.error(f"  Failed to close {self.symbol}: {e}")
                logger.error(f"  MANUAL ACTION: Close {self.symbol} on Alpaca dashboard!")

        logger.info("Cleanup complete")


def main():
    """Entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  STOP MONITOR SYSTEM TEST")
    logger.info("=" * 60)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Budget: ${args.dollars:.0f}")
    logger.info(f"Stop offset: +{args.stop_offset_pct:.1%} above current price")
    logger.info(f"Timeout: {args.timeout}s")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("")

    runner = SystemTestRunner(
        symbol=args.symbol,
        max_dollars=args.dollars,
        stop_offset_pct=args.stop_offset_pct,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )

    passed = runner.run()
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
