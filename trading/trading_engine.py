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
from datetime import date, datetime, timedelta, timezone
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
        setup_expiry_seconds: int = 600,
        market_regime: Optional['MarketRegimeFilter'] = None,
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
            setup_expiry_seconds: Cancel pending buy-stop after this many seconds
            market_regime: Optional MarketRegimeFilter for SPY regime check
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

        self.setup_expiry_seconds = setup_expiry_seconds

        self.market_regime = market_regime

        self._qualified_symbols: Set[str] = set()
        self._traded_symbols: Set[str] = set()
        self._patterns_detected: int = 0
        self._patterns_traded: int = 0
        self._pattern_details: list = []
        self._pending_orders: Dict[str, Dict] = {}  # symbol -> {order_id, plan, setup, placed_at}
        self.shutdown_event = None  # Set by caller for graceful shutdown

    def _refresh_spy_data(self) -> None:
        """Fetch recent SPY daily bars for regime filter."""
        if not self.market_regime:
            return
        try:
            end = date.today()
            start = end - timedelta(days=14)  # 14 calendar days -> ~10 trading days
            bars = self.alpaca.get_daily_bars_range(['SPY'], start, end)
            spy_bars = bars.get('SPY', [])
            self.market_regime.load_spy_bars(spy_bars)
            ret = self.market_regime.get_spy_5d_return(end)
            if ret is not None:
                logger.info(f"SPY regime refreshed: {len(spy_bars)} bars, 5d return: {ret:.1f}%")
            else:
                logger.info(f"SPY regime refreshed: {len(spy_bars)} bars, insufficient data")
        except Exception as e:
            logger.error(f"Failed to refresh SPY regime data: {e}")

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

    def _identify_bracket_legs(
        self, legs: List[Dict], expected_sl: float = None, expected_tp: float = None
    ) -> tuple:
        """
        Identify stop-loss and take-profit legs from bracket order legs.

        Args:
            legs: List of leg dicts from Alpaca order
            expected_sl: Expected stop loss price (for disambiguation)
            expected_tp: Expected take profit price (for disambiguation)

        Returns:
            Tuple of (sl_leg, tp_leg) — either may be None if not found
        """
        sl_leg = None
        tp_leg = None
        for leg in legs:
            if leg.get('side') != 'sell':
                continue
            has_stop = leg.get('stop_price') is not None
            has_limit = leg.get('limit_price') is not None
            if has_stop and not has_limit:
                sl_leg = leg
            elif has_limit and not has_stop:
                tp_leg = leg
            elif has_stop and has_limit:
                # Both present — match by proximity to expected prices
                if expected_sl and abs(leg['stop_price'] - expected_sl) < abs(leg['limit_price'] - expected_sl):
                    sl_leg = leg
                else:
                    tp_leg = leg
        return sl_leg, tp_leg

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
                filled_qty = order_status.get('filled_qty', 0)

                # Fix 1: Retry if fill data missing (Alpaca can lag on fill price)
                if fill_price is None:
                    for attempt in range(5):
                        time_mod.sleep(0.5)
                        try:
                            refreshed = self.alpaca.get_order(order_id)
                            fill_price = refreshed.get('filled_avg_price')
                            filled_qty = refreshed.get('filled_qty', filled_qty)
                            if fill_price is not None:
                                logger.info(f"{symbol}: Fill price resolved on retry {attempt + 1}")
                                break
                        except Exception:
                            pass

                    # Position fallback
                    if fill_price is None:
                        try:
                            positions = self.alpaca.get_open_positions()
                            for pos in positions:
                                if pos['symbol'] == symbol:
                                    fill_price = float(pos['avg_entry_price'])
                                    filled_qty = int(pos['qty'])
                                    logger.warning(f"{symbol}: Using position fallback — ${fill_price}")
                                    break
                        except Exception as e:
                            logger.error(f"{symbol}: Position fallback failed: {e}")

                    if fill_price is None:
                        logger.error(f"{symbol}: Fill price unavailable after retries — skipping")
                        continue

                # Fix 2: Partial fill detection
                plan = pending['plan']
                requested_qty = plan.shares
                if filled_qty and filled_qty < requested_qty:
                    logger.warning(
                        f"{symbol}: PARTIAL FILL — {filled_qty}/{requested_qty} shares @ ${fill_price}"
                    )
                actual_qty = filled_qty if filled_qty and filled_qty > 0 else requested_qty

                logger.info(
                    f"{symbol}: Buy-stop order FILLED at ${fill_price} — "
                    f"{actual_qty} shares, ID: {order_id}"
                )
                self._traded_symbols.add(symbol)
                self._patterns_traded += 1
                symbols_to_remove.append(symbol)

                # Phase 2: Update trade record with fill data
                trade_record = self.db.get_trade_by_order_id(order_id)
                if trade_record:
                    self.db.update_trade(trade_record['id'], {
                        'order_status': 'filled',
                        'fill_price': fill_price,
                        'filled_qty': actual_qty,
                        'filled_at': datetime.now(timezone.utc),
                    })
                    logger.info(f"{symbol}: Trade DB updated — fill ${fill_price}, qty {actual_qty}")
                else:
                    logger.error(f"{symbol}: No trade record for order {order_id}")

                # Phase 3: Gap-fill stop + target adjustment
                setup = pending.get('setup')
                if fill_price and setup and fill_price > setup.breakout_level:
                    entry_gap = fill_price - setup.breakout_level
                    adjusted_stop = round(fill_price - plan.risk_per_share, 2)
                    adjusted_target = round(fill_price + plan.risk_per_share * plan.risk_reward_ratio, 2)
                    logger.info(
                        f"{symbol}: Gap fill +${entry_gap:.2f} — "
                        f"stop ${plan.stop_loss_price:.2f} → ${adjusted_stop:.2f}, "
                        f"target ${plan.take_profit_price:.2f} → ${adjusted_target:.2f}"
                    )
                    try:
                        order_detail = self.alpaca.get_order(order_id)
                        sl_leg, tp_leg = self._identify_bracket_legs(
                            order_detail.get('legs', []),
                            expected_sl=plan.stop_loss_price,
                            expected_tp=plan.take_profit_price,
                        )

                        if sl_leg:
                            self.alpaca.replace_order_stop_price(sl_leg['id'], adjusted_stop)
                            logger.info(f"{symbol}: Stop adjusted to ${adjusted_stop:.2f}")
                        else:
                            logger.error(f"{symbol}: No SL leg found in bracket order")

                        if tp_leg:
                            self.alpaca.replace_order_limit_price(tp_leg['id'], adjusted_target)
                            logger.info(f"{symbol}: Target adjusted to ${adjusted_target:.2f}")
                        else:
                            logger.error(f"{symbol}: No TP leg found in bracket order")

                        if trade_record:
                            self.db.update_trade(trade_record['id'], {
                                'stop_loss_price': adjusted_stop,
                                'take_profit_price': adjusted_target,
                            })
                    except Exception as e:
                        logger.error(f"{symbol}: Failed to adjust orders after gap fill: {e}")

                if self.notifier:
                    self.notifier.notify_order_submitted(
                        symbol=symbol,
                        order_id=order_id,
                        shares=actual_qty,
                        entry=fill_price or plan.entry_price,
                    )

                return {
                    'order_id': order_id,
                    'status': 'filled',
                    'symbol': symbol,
                    'fill_price': fill_price,
                    'filled_qty': actual_qty,
                }

            elif status in ('cancelled', 'expired', 'rejected'):
                logger.info(f"{symbol}: Pending order {status} — ID: {order_id}")
                symbols_to_remove.append(symbol)

            else:
                # Phase 5: Setup expiry — cancel stale buy-stops
                placed_at = pending.get('placed_at')
                if placed_at:
                    age = (datetime.now(timezone.utc) - placed_at).total_seconds()
                    if age > self.setup_expiry_seconds:
                        logger.info(f"{symbol}: Buy-stop EXPIRED after {age:.0f}s, cancelling")
                        # Fix 7: Refresh status before cancel — order may have filled
                        try:
                            refreshed = self.alpaca.get_order(order_id)
                            if refreshed.get('status') == 'filled':
                                logger.info(f"{symbol}: Order filled while checking expiry — handling next cycle")
                                continue
                            elif refreshed.get('status') in ('cancelled', 'expired'):
                                logger.info(f"{symbol}: Order already {refreshed['status']}")
                                symbols_to_remove.append(symbol)
                                continue
                        except Exception:
                            pass  # proceed with cancel attempt
                        try:
                            self.alpaca.cancel_order(order_id)
                        except Exception as e:
                            logger.error(f"{symbol}: Failed to cancel expired order: {e}")
                        symbols_to_remove.append(symbol)
                        continue

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
                                # Fix 7: Refresh status before cancel
                                try:
                                    refreshed = self.alpaca.get_order(order_id)
                                    if refreshed.get('status') == 'filled':
                                        logger.info(f"{symbol}: Order filled while checking invalidation — handling next cycle")
                                        continue
                                    elif refreshed.get('status') in ('cancelled', 'expired'):
                                        logger.info(f"{symbol}: Order already {refreshed['status']}")
                                        symbols_to_remove.append(symbol)
                                        continue
                                except Exception:
                                    pass  # proceed with cancel attempt
                                self.alpaca.cancel_order(order_id)
                                symbols_to_remove.append(symbol)
                    except Exception as e:
                        logger.error(f"{symbol}: Failed to check invalidation: {e}")

        for symbol in symbols_to_remove:
            self._pending_orders.pop(symbol, None)

        return None

    def _sync_closed_positions(self) -> None:
        """Detect bracket exits (SL/TP hit) and update DB + circuit breaker."""
        today = date.today().isoformat()
        open_trades = self.db.get_open_trades(today)
        if not open_trades:
            return

        try:
            alpaca_positions = {p['symbol'] for p in self.alpaca.get_open_positions()}
        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")
            return

        for trade in open_trades:
            symbol = trade['symbol']
            if symbol not in alpaca_positions and trade.get('fill_price'):
                try:
                    order_id = trade.get('order_id')
                    exit_price = None
                    exit_reason = None
                    if order_id:
                        order_detail = self.alpaca.get_order(order_id)
                        sl_leg, tp_leg = self._identify_bracket_legs(
                            order_detail.get('legs', []),
                            expected_sl=trade.get('stop_loss_price'),
                            expected_tp=trade.get('take_profit_price'),
                        )
                        # Check SL leg
                        if sl_leg and sl_leg.get('status') == 'filled':
                            fill = sl_leg.get('filled_avg_price')
                            exit_price = fill or sl_leg['stop_price']
                            exit_reason = 'stop_loss'
                        # Check TP leg
                        elif tp_leg and tp_leg.get('status') == 'filled':
                            fill = tp_leg.get('filled_avg_price')
                            exit_price = fill or tp_leg['limit_price']
                            exit_reason = 'take_profit'

                    if exit_price:
                        # Use filled_qty if available, fall back to shares
                        qty_for_pnl = trade.get('filled_qty') or trade['shares']
                        pnl = (exit_price - trade['fill_price']) * qty_for_pnl
                        pnl_pct = (exit_price / trade['fill_price'] - 1) * 100
                        self.db.update_trade(trade['id'], {
                            'exit_price': exit_price,
                            'exit_reason': exit_reason,
                            'exited_at': datetime.now(timezone.utc),
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                        })
                        self.position_manager.record_trade_pnl(pnl)
                        logger.info(
                            f"{symbol}: {exit_reason} — exit ${exit_price:.2f}, "
                            f"P&L ${pnl:+,.2f} ({pnl_pct:+.1f}%)"
                        )
                    else:
                        logger.warning(f"{symbol}: Position closed but exit price unknown")
                except Exception as e:
                    logger.error(f"{symbol}: Failed to process closed position: {e}")

    def run_pattern_check(self) -> Optional[Dict[str, Any]]:
        """
        Run one pattern detection cycle on all qualified symbols.

        Flow:
        1. Sync closed positions (detect bracket exits)
        2. Manage pending buy-stop orders (check fills, invalidations)
        3. For each qualified symbol without a pending/filled order:
           a. Fetch 1-min bars
           b. Run bull flag setup detection
           c. If setup found, create plan and submit buy-stop bracket order
        4. If past last_entry_time, skip new order placement

        Returns:
            Dict with order details if a trade was executed, None otherwise
        """
        if not self.enabled:
            return None

        # Market regime filter
        if self.market_regime and not self.market_regime.is_regime_ok(date.today()):
            ret = self.market_regime.get_spy_5d_return(date.today())
            ret_str = f"{ret:.1f}%" if ret is not None else "N/A"
            logger.warning(
                f"REGIME FILTER: SPY 5d return {ret_str} "
                f"< {self.market_regime.spy_5d_return_min}% — skipping all trades"
            )
            return None

        # Step 0: Sync closed positions (detect bracket exits for circuit breaker)
        self._sync_closed_positions()

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
                'placed_at': datetime.now(timezone.utc),
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

        # Close open positions and update DB
        try:
            positions = self.alpaca.get_open_positions()
            today = date.today().isoformat()
            open_trades = self.db.get_open_trades(today)
            # Index open trades by symbol for fast lookup
            trades_by_symbol = {}
            for t in open_trades:
                trades_by_symbol[t['symbol']] = t

            FORCE_CLOSE_RETRIES = 3
            FORCE_CLOSE_BACKOFF = [2, 5, 10]

            for pos in positions:
                symbol = pos['symbol']
                close_succeeded = False

                for attempt in range(FORCE_CLOSE_RETRIES):
                    try:
                        self.alpaca.close_position(symbol)
                        close_succeeded = True
                        break
                    except Exception as e:
                        if attempt < FORCE_CLOSE_RETRIES - 1:
                            wait = FORCE_CLOSE_BACKOFF[attempt]
                            logger.warning(
                                f"{symbol}: Force close attempt {attempt + 1} failed: {e}, "
                                f"retry in {wait}s"
                            )
                            time_mod.sleep(wait)
                        else:
                            logger.error(f"{symbol}: ALL force close attempts failed: {e}")
                            if self.notifier:
                                self.notifier.notify_error(
                                    f"MANUAL INTERVENTION: {symbol} force close failed "
                                    f"after {FORCE_CLOSE_RETRIES} attempts"
                                )

                if not close_succeeded:
                    continue

                exit_price = pos.get('avg_entry_price', 0)
                # Use current market value to compute actual exit price
                qty = pos.get('qty', 0)
                if qty > 0 and pos.get('market_value'):
                    exit_price = float(pos['market_value']) / qty

                logger.info(f"{symbol}: Force-close — position closed at ~${exit_price:.2f}")

                # Update DB trade record with exit details
                trade = trades_by_symbol.get(symbol)
                if trade and trade.get('fill_price'):
                    qty_for_pnl = trade.get('filled_qty') or trade['shares']
                    pnl = (exit_price - trade['fill_price']) * qty_for_pnl
                    pnl_pct = (exit_price / trade['fill_price'] - 1) * 100
                    self.db.update_trade(trade['id'], {
                        'exit_price': exit_price,
                        'exit_reason': 'force_close',
                        'exited_at': datetime.now(timezone.utc),
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                    })
                    self.position_manager.record_trade_pnl(pnl)
                    logger.info(
                        f"{symbol}: Force-close DB updated — "
                        f"P&L ${pnl:+,.2f} ({pnl_pct:+.1f}%)"
                    )
                elif trade:
                    logger.warning(
                        f"{symbol}: Force-close — trade has no fill_price, "
                        f"cannot compute P&L"
                    )

                if self.notifier:
                    self.notifier.notify_position_closed(
                        symbol=symbol,
                        exit_reason='force_close',
                    )
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

        while not (self.shutdown_event and self.shutdown_event.is_set()):
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

            # Use shutdown_event.wait() instead of time.sleep() for interruptible sleep
            if self.shutdown_event:
                self.shutdown_event.wait(self.pattern_poll_interval)
            else:
                time_mod.sleep(self.pattern_poll_interval)

        # Graceful shutdown: force-close all positions
        if self.shutdown_event and self.shutdown_event.is_set():
            logger.info("Shutdown signal received — force-closing all positions...")
            self._force_close_all()
            self.save_daily_summary()
            logger.info("Graceful shutdown complete")

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
        self._refresh_spy_data()
        logger.info("Trading engine: daily state reset")
