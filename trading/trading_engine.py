"""
Trading engine — orchestrator for the automated trading pipeline.

Flow:
1. Scanner qualifies a stock → on_stock_qualified(symbol)
2. Fetch 1-min bars for qualified symbols
3. Run bull flag detection
4. Create trade plan if pattern detected
5. Check position manager limits
6. Submit bracket order
7. Track positions
"""

import logging
import time as time_mod
from datetime import date, datetime
from typing import Set, Optional, Dict, Any, List

import pytz

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.pattern_detector import BullFlagDetector
from trading.trade_planner import TradePlanner
from trading.order_executor import OrderExecutor
from trading.position_manager import PositionManager
from notifications.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)

ET = pytz.timezone('US/Eastern')


class TradingEngine:
    """
    Orchestrates the automated trading pipeline.

    Receives qualified stocks from the scanner, detects patterns,
    creates trade plans, and executes bracket orders.
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        db: Database,
        detector: BullFlagDetector,
        planner: TradePlanner,
        executor: OrderExecutor,
        position_manager: PositionManager,
        pattern_poll_interval: int = 60,
        enabled: bool = False,
        notifier: Optional['TelegramNotifier'] = None,
        last_entry_time_et: str = "15:00",
        force_close_time_et: str = "15:45",
    ):
        """
        Initialize TradingEngine.

        Args:
            alpaca_client: Alpaca API client
            db: Database instance
            detector: Bull flag pattern detector
            planner: Trade planner
            executor: Order executor
            position_manager: Position manager
            pattern_poll_interval: Seconds between pattern checks
            enabled: Master kill switch
            notifier: Optional Telegram notifier for trading alerts
            last_entry_time_et: No new entries after this ET time (HH:MM)
            force_close_time_et: Force close all positions at this ET time (HH:MM)
        """
        self.alpaca = alpaca_client
        self.db = db
        self.detector = detector
        self.planner = planner
        self.executor = executor
        self.position_manager = position_manager
        self.pattern_poll_interval = pattern_poll_interval
        self.enabled = enabled
        self.notifier = notifier

        # Time controls
        last_h, last_m = last_entry_time_et.split(':')
        self.last_entry_hour = int(last_h)
        self.last_entry_minute = int(last_m)
        fc_h, fc_m = force_close_time_et.split(':')
        self.force_close_hour = int(fc_h)
        self.force_close_minute = int(fc_m)

        self._qualified_symbols: Set[str] = set()
        self._traded_symbols: Set[str] = set()
        self._patterns_detected: int = 0
        self._patterns_traded: int = 0
        self._pattern_details: list = []
        self._pending_orders: Dict[str, Dict] = {}  # symbol -> {order_id, plan, setup}

    def on_stock_qualified(self, symbol: str) -> None:
        """
        Handle a stock qualified by the scanner.

        Adds to the qualified symbols set for pattern monitoring.

        Args:
            symbol: Qualified stock symbol
        """
        if not self.enabled:
            logger.debug(f"{symbol}: Trading engine disabled, ignoring qualified stock")
            return

        if symbol in self._traded_symbols:
            logger.debug(f"{symbol}: Already traded today, skipping")
            return

        if symbol not in self._qualified_symbols:
            self._qualified_symbols.add(symbol)
            logger.info(f"{symbol}: Added to qualified symbols for pattern monitoring")

    def _is_past_last_entry_time(self) -> bool:
        """Check if current ET time is past last_entry_time."""
        now_et = datetime.now(ET)
        return (now_et.hour > self.last_entry_hour or
                (now_et.hour == self.last_entry_hour and now_et.minute >= self.last_entry_minute))

    def _is_past_force_close_time(self) -> bool:
        """Check if current ET time is past force_close_time."""
        now_et = datetime.now(ET)
        return (now_et.hour > self.force_close_hour or
                (now_et.hour == self.force_close_hour and now_et.minute >= self.force_close_minute))

    def _manage_pending_orders(self) -> Optional[Dict[str, Any]]:
        """
        Check status of all pending buy-stop orders.

        For each pending order:
        - If filled → mark traded, send notification
        - If price dropped below flag_low → cancel order (setup invalidated)
        - If cancelled/expired → remove from tracking

        Returns:
            Dict with fill details if an order was filled, None otherwise
        """
        if not self._pending_orders:
            return None

        symbols_to_remove = []

        for symbol, pending in list(self._pending_orders.items()):
            order_id = pending['order_id']

            try:
                order_status = self.alpaca.get_order(order_id)
            except Exception as e:
                logger.error(f"{symbol}: Failed to get order status: {e}")
                continue

            status = order_status.get('status', 'unknown')

            if status == 'filled':
                fill_price = order_status.get('filled_avg_price')
                logger.info(
                    f"{symbol}: Buy-stop order FILLED at ${fill_price} — "
                    f"ID: {order_id}"
                )
                self._traded_symbols.add(symbol)
                self._patterns_traded += 1
                symbols_to_remove.append(symbol)

                if self.notifier:
                    plan = pending['plan']
                    self.notifier.notify_order_submitted(
                        symbol=symbol,
                        order_id=order_id,
                        shares=plan.shares,
                        entry=fill_price or plan.entry_price,
                    )

                return {
                    'order_id': order_id,
                    'status': 'filled',
                    'symbol': symbol,
                    'fill_price': fill_price,
                }

            elif status in ('cancelled', 'expired', 'rejected'):
                logger.info(f"{symbol}: Pending order {status} — ID: {order_id}")
                symbols_to_remove.append(symbol)

            else:
                # Still pending — check if setup invalidated
                setup = pending.get('setup')
                if setup:
                    try:
                        bars = self.alpaca.get_1min_bars(symbol, lookback_minutes=5)
                        if bars is not None and not bars.empty:
                            latest_low = bars.iloc[-1]['low']
                            if latest_low < setup.flag_low:
                                logger.info(
                                    f"{symbol}: Setup INVALIDATED — "
                                    f"low ${latest_low:.2f} < flag_low ${setup.flag_low:.2f}, "
                                    f"cancelling order {order_id}"
                                )
                                self.alpaca.cancel_order(order_id)
                                symbols_to_remove.append(symbol)
                    except Exception as e:
                        logger.error(f"{symbol}: Failed to check invalidation: {e}")

        for symbol in symbols_to_remove:
            self._pending_orders.pop(symbol, None)

        return None

    def run_pattern_check(self) -> Optional[Dict[str, Any]]:
        """
        Run one pattern detection cycle on all qualified symbols.

        Flow:
        1. Manage pending buy-stop orders (check fills, invalidations)
        2. For each qualified symbol without a pending/filled order:
           a. Fetch 1-min bars
           b. Run bull flag setup detection
           c. If setup found, create plan and submit buy-stop bracket order
        3. If past last_entry_time, skip new order placement

        Returns:
            Dict with order details if a trade was executed, None otherwise
        """
        if not self.enabled:
            return None

        # Step 1: Manage existing pending orders
        fill_result = self._manage_pending_orders()
        if fill_result is not None:
            return fill_result

        if not self._qualified_symbols:
            logger.debug("No qualified symbols to check")
            return None

        # Skip new orders after last_entry_time
        if self._is_past_last_entry_time():
            logger.debug("Past last entry time, not placing new orders")
            return None

        symbols_to_check = (
            self._qualified_symbols - self._traded_symbols
            - set(self._pending_orders.keys())
        )
        if not symbols_to_check:
            logger.debug("All qualified symbols already traded or have pending orders")
            return None

        logger.info(f"Pattern check: {len(symbols_to_check)} symbols — {sorted(symbols_to_check)}")

        for symbol in sorted(symbols_to_check):
            result = self._check_symbol(symbol)
            if result is not None:
                return result

        return None

    def _check_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Check a single symbol for bull flag setup and place buy-stop order.

        Uses detect_setup() instead of detect() to find setups BEFORE breakout,
        then submits a buy-stop bracket order at breakout_level.

        Args:
            symbol: Stock symbol to check

        Returns:
            Dict with order details if buy-stop placed, None otherwise
        """
        # Fetch 1-min bars
        try:
            bars = self.alpaca.get_1min_bars(symbol, lookback_minutes=30)
        except Exception as e:
            logger.error(f"{symbol}: Failed to fetch 1-min bars: {e}")
            return None

        if bars is None or bars.empty:
            logger.debug(f"{symbol}: No 1-min bars available")
            return None

        # Detect setup (before breakout)
        setup = self.detector.detect_setup(symbol, bars)
        if setup is None:
            return None

        self._patterns_detected += 1
        self._pattern_details.append({
            'symbol': symbol,
            'pole_gain_pct': setup.pole_gain_pct,
            'retracement_pct': setup.retracement_pct,
            'breakout_level': setup.breakout_level,
        })

        # Notify pattern detected
        if self.notifier:
            self.notifier.notify_pattern_detected(
                symbol=symbol,
                pole_gain_pct=setup.pole_gain_pct,
                retracement_pct=setup.retracement_pct,
                breakout_level=setup.breakout_level,
            )

        # Create trade plan
        plan = self.planner.create_plan(setup)
        if plan is None:
            return None

        # Notify trade planned
        if self.notifier:
            self.notifier.notify_trade_planned(
                symbol=symbol,
                entry=plan.entry_price,
                stop=plan.stop_loss_price,
                target=plan.take_profit_price,
                shares=plan.shares,
                risk_reward=plan.risk_reward_ratio,
            )

        # Check position limits
        if not self.position_manager.can_open_position(symbol):
            return None

        # Submit buy-stop bracket order
        result = self.executor.submit_buy_stop_bracket_order(plan)
        if result is not None:
            self.position_manager.mark_traded(symbol)
            self._pending_orders[symbol] = {
                'order_id': result['order_id'],
                'plan': plan,
                'setup': setup,
            }
            logger.info(f"{symbol}: BUY-STOP ORDER PLACED — {result}")

            # Notify order submitted
            if self.notifier:
                self.notifier.notify_order_submitted(
                    symbol=symbol,
                    order_id=result.get('order_id', ''),
                    shares=plan.shares,
                    entry=plan.entry_price,
                )

        return result

    def _force_close_all(self) -> None:
        """
        Cancel all pending orders and close all open positions.

        Called at force_close_time to ensure we're flat before market close.
        """
        # Cancel pending orders
        for symbol, pending in list(self._pending_orders.items()):
            try:
                self.alpaca.cancel_order(pending['order_id'])
                logger.info(f"{symbol}: Force-close — cancelled pending order {pending['order_id']}")
            except Exception as e:
                logger.error(f"{symbol}: Failed to cancel pending order: {e}")
        self._pending_orders.clear()

        # Close open positions
        try:
            positions = self.alpaca.get_open_positions()
            for pos in positions:
                symbol = pos['symbol']
                try:
                    self.alpaca.close_position(symbol)
                    logger.info(f"{symbol}: Force-close — position closed")

                    if self.notifier:
                        self.notifier.notify_position_closed(
                            symbol=symbol,
                            exit_reason='force_close',
                        )
                except Exception as e:
                    logger.error(f"{symbol}: Failed to close position: {e}")
        except Exception as e:
            logger.error(f"Failed to get open positions for force-close: {e}")

    def run_monitoring_loop(self) -> None:
        """
        Run the pattern monitoring loop.

        Polls qualified symbols every pattern_poll_interval seconds.
        Stops placing new orders after last_entry_time.
        Force-closes all positions at force_close_time.
        Stops at market close (16:00 ET).
        """
        if not self.enabled:
            logger.info("Trading engine disabled, skipping monitoring loop")
            return

        logger.info(
            f"Trading engine monitoring loop started — "
            f"interval: {self.pattern_poll_interval}s, "
            f"symbols: {len(self._qualified_symbols)}, "
            f"last entry: {self.last_entry_hour}:{self.last_entry_minute:02d} ET, "
            f"force close: {self.force_close_hour}:{self.force_close_minute:02d} ET"
        )

        force_closed = False

        while True:
            now_et = datetime.now(ET)
            if now_et.hour >= 16:
                logger.info("Market closed, stopping monitoring loop")
                break

            # Force close check
            if not force_closed and self._is_past_force_close_time():
                logger.info("Force close time reached — closing all positions")
                self._force_close_all()
                force_closed = True

            if not force_closed:
                self.run_pattern_check()

            time_mod.sleep(self.pattern_poll_interval)

    def get_daily_stats(self) -> Dict[str, Any]:
        """Get daily trading statistics."""
        today = date.today().isoformat()
        trades = self.db.get_trades_by_date(today)
        daily_pnl = self.db.get_daily_pnl(today)
        open_trades = self.db.get_open_trades(today)

        winning = sum(1 for t in trades if t.get('pnl') and t['pnl'] > 0)
        losing = sum(1 for t in trades if t.get('pnl') and t['pnl'] < 0)

        return {
            'trade_date': today,
            'total_trades': len(trades),
            'winning_trades': winning,
            'losing_trades': losing,
            'gross_pnl': daily_pnl,
            'open_positions': len(open_trades),
            'patterns_detected': self._patterns_detected,
            'patterns_traded': self._patterns_traded,
            'qualified_symbols': len(self._qualified_symbols),
            'patterns_detected_details': list(self._pattern_details),
            'trades': [dict(t) for t in trades] if trades else [],
        }

    def generate_daily_report(self, premarket_gaps: list = None,
                               qualified_stocks: list = None,
                               universe_size: int = 0) -> Dict[str, Any]:
        """
        Generate the full daily report data for Telegram.

        Args:
            premarket_gaps: List of pre-market gap dicts from scanner
            qualified_stocks: List of qualified stock dicts from scanner
            universe_size: Size of the stock universe

        Returns:
            Complete report dict for TelegramNotifier.send_daily_report()
        """
        stats = self.get_daily_stats()
        return {
            'trade_date': stats['trade_date'],
            'universe_size': universe_size,
            'premarket_gaps': premarket_gaps or [],
            'qualified_stocks': qualified_stocks or [],
            'patterns_detected': stats['patterns_detected'],
            'patterns_detected_details': stats['patterns_detected_details'],
            'trades': stats['trades'],
            'total_trades': stats['total_trades'],
            'winning_trades': stats['winning_trades'],
            'losing_trades': stats['losing_trades'],
            'gross_pnl': stats['gross_pnl'],
            'open_positions': stats['open_positions'],
        }

    def send_daily_report(self, premarket_gaps: list = None,
                           qualified_stocks: list = None,
                           universe_size: int = 0) -> None:
        """Generate and send the end-of-day Telegram report."""
        if not self.notifier:
            logger.debug("No notifier configured, skipping daily report")
            return

        report = self.generate_daily_report(
            premarket_gaps=premarket_gaps,
            qualified_stocks=qualified_stocks,
            universe_size=universe_size,
        )
        self.notifier.send_daily_report(report)
        logger.info("End-of-day Telegram report sent")

    def save_daily_summary(self) -> None:
        """Save daily trading summary to database."""
        stats = self.get_daily_stats()
        self.db.save_daily_summary({
            'trade_date': stats['trade_date'],
            'total_trades': stats['total_trades'],
            'winning_trades': stats['winning_trades'],
            'losing_trades': stats['losing_trades'],
            'gross_pnl': stats['gross_pnl'],
            'patterns_detected': stats['patterns_detected'],
            'patterns_traded': stats['patterns_traded'],
        })
        logger.info(f"Daily summary saved: {stats}")

    def reset_daily(self) -> None:
        """Reset daily state for a new trading day."""
        self._qualified_symbols.clear()
        self._traded_symbols.clear()
        self._patterns_detected = 0
        self._patterns_traded = 0
        self._pattern_details.clear()
        self._pending_orders.clear()
        self.position_manager.reset_daily()
        logger.info("Trading engine: daily state reset")
