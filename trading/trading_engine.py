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

import pandas as pd
import pytz

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.pattern_detector import BullFlagDetector
from trading.trade_planner import TradePlanner, TradePlan
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
        stop_monitor: Optional[Any] = None,
        safety_net_sl_pct: float = 0.05,
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
            stop_monitor: Optional StopMonitor for self-managed stops
            safety_net_sl_pct: Safety-net SL percentage for bracket when
                using self-managed stops (default 5%)
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

        # Self-managed stops
        self.stop_monitor = stop_monitor
        self.safety_net_sl_pct = safety_net_sl_pct

        # Load trailing stop + skip_fridays from config
        from config import Config
        _cfg = Config._load_yaml_only()
        self.skip_fridays = bool(_cfg.get("trading", {}).get("skip_fridays", False))
        trail_cfg = _cfg.get("trading", {}).get("trailing_stop", {})
        self.trailing_stop_enabled = bool(trail_cfg.get("enabled", False))
        self.trailing_stop_r = float(trail_cfg.get("trail_r", 1.0))
        self.trailing_activate_at_r = float(trail_cfg.get("activate_at_r", 2.0))

        self._qualified_symbols: Set[str] = set()
        self._traded_symbols: Set[str] = set()
        self._patterns_detected: int = 0
        self._patterns_traded: int = 0
        self._pattern_details: list = []
        self._pending_orders: Dict[str, Dict] = {}  # symbol -> {order_id, plan, setup, placed_at}
        self._daily_trade_count: int = 0
        self._notified_setups: Dict[str, float] = {}  # symbol -> breakout_level (dedup Telegram)
        self._macd_warmup_cache: Dict[str, Optional[pd.Series]] = {}  # symbol -> prev-day closes
        self.shutdown_event = None  # Set by caller for graceful shutdown

    def _fetch_macd_warmup(self, symbol: str) -> None:
        """
        Fetch previous trading day's 1-min bars for MACD warm-up.

        Caches the result per symbol so we only fetch once per day.
        Uses the last 60 bars (1 hour) of the previous session.

        Args:
            symbol: Stock symbol to fetch warm-up data for
        """
        import pytz as _pytz
        _et = _pytz.timezone('US/Eastern')
        today = datetime.now(_et).date()

        # Find previous trading day (skip weekends)
        prev_date = today - timedelta(days=1)
        while prev_date.weekday() >= 5:
            prev_date -= timedelta(days=1)

        try:
            # Previous day's market hours in UTC
            prev_open = _et.localize(
                datetime(prev_date.year, prev_date.month, prev_date.day, 9, 30)
            ).astimezone(timezone.utc)
            prev_close = _et.localize(
                datetime(prev_date.year, prev_date.month, prev_date.day, 16, 0)
            ).astimezone(timezone.utc)

            prev_bars = self.alpaca.get_historical_1min_bars(symbol, prev_open, prev_close)
            if prev_bars is not None and not prev_bars.empty:
                warmup_closes = prev_bars['close'].tail(60).reset_index(drop=True)
                self._macd_warmup_cache[symbol] = warmup_closes
                logger.debug(
                    f"{symbol}: MACD warm-up loaded ({len(warmup_closes)} bars "
                    f"from {prev_date})"
                )
            else:
                self._macd_warmup_cache[symbol] = None
                logger.debug(f"{symbol}: No prev-day bars for MACD warm-up")
        except Exception as e:
            self._macd_warmup_cache[symbol] = None
            logger.warning(f"{symbol}: Failed to fetch MACD warm-up: {e}")

    def _refresh_spy_data(self) -> None:
        """Fetch recent SPY daily bars for regime filter."""
        if not self.market_regime:
            return
        try:
            end = date.today()
            # Need enough history for SMA period + buffer
            sma_period = getattr(self.market_regime, 'sma_period', 50)
            lookback_days = int(sma_period * 1.5) + 14  # trading days -> calendar days
            start = end - timedelta(days=lookback_days)
            bars = self.alpaca.get_daily_bars_range(['SPY'], start, end)
            spy_bars = bars.get('SPY', [])
            self.market_regime.load_spy_bars(spy_bars)
            info = self.market_regime.get_regime_info(end)
            vol_str = f"{info['vol_5d']:.2f}%" if info['vol_5d'] is not None else "N/A"
            sma_str = f"{info['sma']:.2f}" if info['sma'] is not None else "N/A"
            below_str = info['is_below_sma']
            logger.info(
                f"SPY regime refreshed: {len(spy_bars)} bars, "
                f"vol_5d={vol_str}, SMA={sma_str}, below_SMA={below_str}, "
                f"regime_ok={info['is_ok']}"
            )
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

    def clear_qualified_symbols(self) -> None:
        """Clear qualified symbols for fresh scanner cycle.

        Called by scanner before each 15-min intraday cycle so stale
        symbols don't accumulate. Pending orders and traded symbols
        are tracked separately and unaffected.
        """
        count = len(self._qualified_symbols)
        self._qualified_symbols.clear()
        if count > 0:
            logger.debug(f"Cleared {count} qualified symbols before fresh scan")

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

        Processes ALL pending orders each cycle (does not return early on
        first fill). This ensures no fills are missed when multiple orders
        fill simultaneously.

        For each pending order:
        - If filled → mark traded, send notification
        - If price dropped below flag_low → cancel order (setup invalidated)
        - If cancelled/expired → remove from tracking

        Returns:
            Dict with last fill details if any order was filled, None otherwise
        """
        if not self._pending_orders:
            return None

        symbols_to_remove = []
        last_fill_result = None

        for symbol, pending in list(self._pending_orders.items()):
            order_id = pending['order_id']

            try:
                order_status = self.alpaca.get_order(order_id)
            except Exception as e:
                error_msg = f"{symbol}: Failed to get order status: {e}"
                logger.error(error_msg)
                if self.notifier:
                    self.notifier.notify_error(error_msg, component="OrderTracking")
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
                            error_msg = f"{symbol}: Position fallback failed: {e}"
                            logger.error(error_msg)
                            if self.notifier:
                                self.notifier.notify_error(error_msg, component="FillTracking")

                    if fill_price is None:
                        error_msg = (
                            f"{symbol}: Fill price unavailable after retries — "
                            f"UNTRACKED FILLED POSITION (order {order_id})"
                        )
                        logger.error(error_msg)
                        if self.notifier:
                            self.notifier.notify_error(error_msg, component="FillTracking")
                        continue

                # Fix 2: Partial fill detection
                plan = pending['plan']
                requested_qty = plan.shares if plan else 0
                if filled_qty and requested_qty and filled_qty < requested_qty:
                    logger.warning(
                        f"{symbol}: PARTIAL FILL — {filled_qty}/{requested_qty} shares @ ${fill_price}"
                    )
                actual_qty = filled_qty if filled_qty and filled_qty > 0 else requested_qty

                logger.info(
                    f"{symbol}: Buy-stop order FILLED at ${fill_price} — "
                    f"{actual_qty} shares, ID: {order_id}"
                )
                self._traded_symbols.add(symbol)
                self.position_manager.mark_traded(symbol)
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
                    error_msg = f"{symbol}: No trade record for order {order_id} — DB integrity issue"
                    logger.error(error_msg)
                    if self.notifier:
                        self.notifier.notify_error(error_msg, component="DBIntegrity")

                # H5 OR: check breakout volume on thin-liquidity days
                if pending.get('thin_liquidity', False):
                    if not self._check_breakout_volume(symbol, pending):
                        self._emergency_close_position(
                            symbol, order_id, fill_price, actual_qty, trade_record
                        )
                        last_fill_result = {
                            'status': 'thin_liquidity_rejected',
                            'symbol': symbol,
                            'fill_price': fill_price,
                            'reason': 'weak_breakout_volume',
                        }
                        continue

                # Phase 3: Gap-fill TARGET adjustment only.
                # Stop stays at the original technical level (flag low region).
                # Moving stop above the technical level puts it in no-man's land
                # where normal price noise triggers it. Dollar risk increases
                # but the stop is at a price that has structural meaning.
                # When trailing stop is enabled, skip TP adjustment — trail handles exits.
                setup = pending.get('setup')
                trail_active = self.trailing_stop_enabled and self.stop_monitor
                if fill_price and plan and setup and fill_price > setup.breakout_level:
                    entry_gap = fill_price - setup.breakout_level
                    actual_risk = round(fill_price - plan.stop_loss_price, 2)
                    adjusted_target = round(fill_price + plan.risk_per_share * plan.risk_reward_ratio, 2)
                    logger.info(
                        f"{symbol}: Gap fill +${entry_gap:.2f} — "
                        f"stop KEPT at ${plan.stop_loss_price:.2f} (technical level), "
                        f"risk ${plan.risk_per_share:.2f} → ${actual_risk:.2f}/sh"
                        f"{' (trail handles TP)' if trail_active else f', target ${plan.take_profit_price:.2f} → ${adjusted_target:.2f}'}"
                    )

                    # When trailing stop is active, skip TP adjustment — trail handles exits.
                    # The TP leg will be cancelled after StopMonitor registration below.
                    if not trail_active:
                        gap_adjust_failed = False
                        try:
                            order_detail = self.alpaca.get_order(order_id)
                            sl_leg, tp_leg = self._identify_bracket_legs(
                                order_detail.get('legs', []),
                                expected_sl=plan.stop_loss_price,
                                expected_tp=plan.take_profit_price,
                            )

                            # Stop stays at original — no replacement needed
                            # Only adjust target upward
                            if tp_leg:
                                self.alpaca.replace_order_limit_price(tp_leg['id'], adjusted_target)
                                logger.info(f"{symbol}: Target adjusted to ${adjusted_target:.2f}")
                            else:
                                logger.error(f"{symbol}: No TP leg found — cannot adjust target")
                                gap_adjust_failed = True

                            if not gap_adjust_failed and trade_record:
                                self.db.update_trade(trade_record['id'], {
                                    'take_profit_price': adjusted_target,
                                })
                        except Exception as e:
                            logger.error(f"{symbol}: Failed to adjust target after gap fill: {e}")
                            gap_adjust_failed = True

                        if gap_adjust_failed:
                            error_msg = (
                                f"{symbol}: GAP FILL TARGET ADJUSTMENT FAILED — "
                                f"entry gap +${entry_gap:.2f}, actual risk "
                                f"${actual_risk:.2f}/sh. Target not updated."
                            )
                            logger.error(error_msg)
                            if self.notifier:
                                self.notifier.notify_error(error_msg, component="GapFill")
                            self._emergency_close_position(
                                symbol, order_id, fill_price, actual_qty, trade_record,
                                exit_reason='gap_adjust_failed',
                            )
                            last_fill_result = {
                                'status': 'gap_adjust_failed',
                                'symbol': symbol,
                                'fill_price': fill_price,
                                'reason': 'leg_replacement_failed',
                            }
                            continue

                # Register with StopMonitor for real-time stop watching
                if self.stop_monitor and pending.get('real_stop_level'):
                    real_stop = pending['real_stop_level']
                    try:
                        order_detail = self.alpaca.get_order(order_id)
                        sl_leg, tp_leg = self._identify_bracket_legs(
                            order_detail.get('legs', []),
                            expected_sl=plan.entry_price * (1 - self.safety_net_sl_pct) if plan else None,
                            expected_tp=plan.take_profit_price if plan else None,
                        )
                        tp_leg_id = tp_leg['id'] if tp_leg else ''
                        sl_leg_id = sl_leg['id'] if sl_leg else ''

                        # Save real_stop_loss_price to DB
                        if trade_record:
                            self.db.update_trade(trade_record['id'], {
                                'real_stop_loss_price': real_stop,
                            })

                        # Trailing stop params (0 = disabled)
                        trail_r = self.trailing_stop_r if self.trailing_stop_enabled else 0.0
                        activate_r = self.trailing_activate_at_r if self.trailing_stop_enabled else 0.0

                        self.stop_monitor.add_watch(
                            symbol=symbol,
                            stop_price=real_stop,
                            shares=actual_qty,
                            tp_leg_id=tp_leg_id,
                            sl_leg_id=sl_leg_id,
                            trade_db_id=trade_record['id'] if trade_record else None,
                            entry_price=fill_price,
                            risk_per_share=fill_price - real_stop,
                            trail_r=trail_r,
                            activate_at_r=activate_r,
                        )

                        # Cancel TP leg when trailing stop is active — trail replaces fixed TP
                        if trail_r > 0 and tp_leg_id:
                            try:
                                self.alpaca.cancel_order(tp_leg_id)
                                # Clear TP leg in watch so _execute_stop_exit
                                # doesn't try to cancel it again
                                with self.stop_monitor._watch_lock:
                                    w = self.stop_monitor._watches.get(symbol)
                                    if w:
                                        w.tp_leg_id = ''
                                logger.info(
                                    f"{symbol}: Cancelled TP leg — "
                                    f"trailing stop ({trail_r:.1f}R, +{activate_r:.1f}R) "
                                    f"replaces fixed TP"
                                )
                            except Exception as e:
                                # TP leg still active — if it fills before trail,
                                # _sync_closed_positions will detect it. Not fatal.
                                logger.warning(
                                    f"{symbol}: TP leg cancel failed: {e} — "
                                    f"TP may still fill before trail activates"
                                )

                        logger.info(
                            f"{symbol}: StopMonitor watching — "
                            f"real stop ${real_stop:.2f}, "
                            f"TP leg {tp_leg_id}, SL leg {sl_leg_id}"
                            f"{f', trail={trail_r:.1f}R' if trail_r > 0 else ''}"
                        )
                    except Exception as e:
                        error_msg = (
                            f"{symbol}: Failed to register with StopMonitor: {e} — "
                            f"safety-net SL on Alpaca is active"
                        )
                        logger.error(error_msg)
                        if self.notifier:
                            self.notifier.notify_error(error_msg, component="StopMonitor")

                self._daily_trade_count += 1
                self._patterns_traded += 1

                if self.notifier:
                    self.notifier.notify_order_submitted(
                        symbol=symbol,
                        order_id=order_id,
                        shares=actual_qty,
                        entry=fill_price or (plan.entry_price if plan else 0),
                    )

                last_fill_result = {
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
                # Cancel pending orders in midday (11:30-14:00 ET).
                # Matches position_manager.can_open_position() and backtest.
                now_et = datetime.now(ET)
                current_minutes = now_et.hour * 60 + now_et.minute
                if 11 * 60 + 30 <= current_minutes < 14 * 60:
                    logger.info(f"{symbol}: Cancelling pending buy-stop — midday dead zone")
                    try:
                        self.alpaca.cancel_order(order_id)
                    except Exception as e:
                        logger.error(f"{symbol}: Failed to cancel midday order: {e}")
                    symbols_to_remove.append(symbol)
                    continue

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
            pending = self._pending_orders.pop(symbol, None)
            # Update DB record so cancelled orders don't show as "open positions"
            if pending:
                order_id = pending.get('order_id')
                if order_id:
                    trade_record = self.db.get_trade_by_order_id(order_id)
                    if trade_record and trade_record.get('fill_price') is None:
                        self.db.update_trade(trade_record['id'], {
                            'order_status': 'cancelled',
                        })
                        logger.debug(f"{symbol}: DB trade record marked cancelled")

        return last_fill_result

    def _process_stop_monitor_exits(self) -> None:
        """Drain and process exit events from StopMonitor."""
        if not self.stop_monitor:
            return

        events = self.stop_monitor.drain_exit_events()
        for event in events:
            logger.info(
                f"{event.symbol}: StopMonitor exit — "
                f"stop=${event.stop_price:.2f}, exit=${event.exit_price:.2f}, "
                f"reason={event.exit_reason}, order={event.order_id}"
            )

            # Poll exit order for ACTUAL fill price (event.exit_price is the
            # limit price, but the fill is typically better on a marketable limit)
            actual_exit_price = event.exit_price
            if event.order_id:
                for poll_attempt in range(5):
                    time_mod.sleep(0.5)
                    try:
                        exit_order = self.alpaca.get_order(event.order_id)
                        if exit_order.get('status') == 'filled':
                            fill = exit_order.get('filled_avg_price')
                            if fill is not None:
                                actual_exit_price = fill
                                logger.info(
                                    f"{event.symbol}: StopMonitor exit filled at "
                                    f"${fill:.2f} (limit was ${event.exit_price:.2f})"
                                )
                                break
                    except Exception:
                        pass
                else:
                    logger.warning(
                        f"{event.symbol}: StopMonitor exit order fill price unavailable "
                        f"after 5 polls — using limit ${event.exit_price:.2f}"
                    )

            # Update DB trade record
            if event.trade_db_id:
                try:
                    # Look up trade by ID in today's open trades
                    trades_today = self.db.get_open_trades(date.today().isoformat())
                    trade_record = None
                    for t in trades_today:
                        if t['id'] == event.trade_db_id:
                            trade_record = t
                            break

                    if trade_record and trade_record.get('fill_price'):
                        qty_for_pnl = trade_record.get('filled_qty') or trade_record['shares']
                        pnl = (actual_exit_price - trade_record['fill_price']) * qty_for_pnl
                        pnl_pct = (actual_exit_price / trade_record['fill_price'] - 1) * 100
                        self.db.update_trade(event.trade_db_id, {
                            'exit_price': actual_exit_price,
                            'exit_reason': event.exit_reason,
                            'exited_at': datetime.now(timezone.utc),
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                        })
                        self.position_manager.record_trade_pnl(pnl)
                        logger.info(
                            f"{event.symbol}: StopMonitor exit DB updated — "
                            f"P&L ${pnl:+,.2f} ({pnl_pct:+.1f}%)"
                        )

                        if self.notifier:
                            self.notifier.notify_position_closed(
                                symbol=event.symbol,
                                entry_price=trade_record['fill_price'],
                                exit_price=actual_exit_price,
                                shares=qty_for_pnl,
                                pnl=pnl,
                                exit_reason=event.exit_reason,
                            )
                    else:
                        logger.warning(
                            f"{event.symbol}: StopMonitor exit — "
                            f"no matching open trade for DB id {event.trade_db_id}"
                        )
                except Exception as e:
                    logger.error(
                        f"{event.symbol}: Failed to process StopMonitor exit: {e}"
                    )

    def _sync_closed_positions(self) -> None:
        """Detect bracket exits (SL/TP hit) and update DB + circuit breaker."""
        # Process StopMonitor exits first — updates DB with exit_price.
        # Must happen BEFORE we fetch open_trades, otherwise trades just
        # closed by StopMonitor still appear as "open" and get double-processed.
        self._process_stop_monitor_exits()

        today = date.today().isoformat()
        open_trades = self.db.get_open_trades(today)
        if not open_trades:
            return

        try:
            alpaca_positions = {p['symbol'] for p in self.alpaca.get_open_positions()}
        except Exception as e:
            error_msg = f"Failed to sync positions: {e}"
            logger.error(error_msg)
            if self.notifier:
                self.notifier.notify_error(error_msg, component="PositionSync")
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
                        # Remove StopMonitor watch — position is gone (TP or
                        # safety-net SL filled on Alpaca side)
                        if self.stop_monitor:
                            self.stop_monitor.remove_watch(symbol)
                        logger.info(
                            f"{symbol}: {exit_reason} — exit ${exit_price:.2f}, "
                            f"P&L ${pnl:+,.2f} ({pnl_pct:+.1f}%)"
                        )
                    else:
                        # Use fill_price as fallback exit to prevent infinite re-check
                        # (exit_price IS NULL keeps this trade in get_open_trades forever)
                        fallback_exit = trade['fill_price']
                        pnl_est = 0.0  # Assume breakeven if unknown
                        error_msg = (
                            f"{symbol}: Position closed but exit price unknown — "
                            f"using fill_price ${fallback_exit:.2f} as estimate"
                        )
                        logger.warning(error_msg)
                        if self.notifier:
                            self.notifier.notify_error(error_msg, component="PositionSync")
                        self.db.update_trade(trade['id'], {
                            'exit_price': fallback_exit,
                            'exit_reason': 'unknown_exit',
                            'exited_at': datetime.now(timezone.utc),
                            'pnl': pnl_est,
                            'pnl_pct': 0.0,
                        })
                except Exception as e:
                    error_msg = f"{symbol}: Failed to process closed position: {e}"
                    logger.error(error_msg)
                    if self.notifier:
                        self.notifier.notify_error(error_msg, component="PositionSync")

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

        # ALWAYS sync positions and manage pending orders — these must run
        # regardless of regime filter or max trades. Skipping them means
        # SL/TP exits go unrecorded, PnL is wrong, and circuit breaker is deaf.
        self._sync_closed_positions()
        fill_result = self._manage_pending_orders()

        # Friday filter — blocks NEW order placement only
        if self.skip_fridays and date.today().weekday() == 4:
            logger.info("FRIDAY FILTER: skipping new trades (30% WR on Fridays)")
            return fill_result

        # Market regime filter — blocks NEW order placement only
        if self.market_regime and not self.market_regime.is_regime_ok(date.today()):
            info = self.market_regime.get_regime_info(date.today())
            vol_str = f"{info['vol_5d']:.2f}%" if info['vol_5d'] is not None else "N/A"
            logger.warning(
                f"REGIME FILTER: vol_5d={vol_str} > {self.market_regime.vol_threshold}% "
                f"AND below SMA{self.market_regime.sma_period} — skipping new trades"
            )
            return fill_result

        # Thin liquidity: log warning for awareness (H5 OR filter)
        # Production enforcement: buy-stop submitted on all days; post-fill volume check on thin days
        if self.market_regime and self.market_regime.is_thin_liquidity(date.today()):
            info = self.market_regime.get_regime_info(date.today())
            svr = info.get('spy_volume_ratio')
            svr_str = f"{svr:.2f}" if svr is not None else "N/A"
            logger.warning(
                f"THIN LIQUIDITY: SPY vol ratio {svr_str} "
                f"< {self.market_regime.min_spy_volume_ratio} — "
                f"breakout vol threshold raised to "
                f"{self.market_regime.thin_liquidity_breakout_vol_ratio:.1f}x"
            )

        # Max trades per day — blocks NEW order placement only
        if self.market_regime and self._daily_trade_count >= self.market_regime.max_trades_per_day:
            logger.warning(
                f"MAX TRADES PER DAY reached ({self._daily_trade_count}) — skipping new trades"
            )
            return fill_result

        if not self._qualified_symbols:
            logger.debug("No qualified symbols to check")
            return fill_result

        # Skip new orders after last_entry_time
        if self._is_past_last_entry_time():
            logger.debug("Past last entry time, not placing new orders")
            return fill_result

        symbols_to_check = (
            self._qualified_symbols - self._traded_symbols
            - set(self._pending_orders.keys())
        )
        if not symbols_to_check:
            logger.debug("All qualified symbols already traded or have pending orders")
            return fill_result

        logger.info(f"Pattern check: {len(symbols_to_check)} symbols — {sorted(symbols_to_check)}")

        last_order_result = None
        for symbol in sorted(symbols_to_check):
            result = self._check_symbol(symbol)
            if result is not None:
                last_order_result = result

        return fill_result or last_order_result

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
        # Fetch 1-min bars from market open (not a fixed window).
        # Fixed 90-min window misses setups that formed earlier in the day —
        # the backtest sees all bars from open, so live must too.
        import pytz as _pytz
        _et = _pytz.timezone('US/Eastern')
        _now_et = datetime.now(_et)
        _market_open = _now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        _minutes_since_open = max(int((_now_et - _market_open).total_seconds() / 60), 30)
        try:
            bars = self.alpaca.get_1min_bars(symbol, lookback_minutes=_minutes_since_open)
        except Exception as e:
            logger.error(f"{symbol}: Failed to fetch 1-min bars: {e}")
            return None

        if bars is None or bars.empty:
            logger.debug(f"{symbol}: No 1-min bars available")
            return None

        # MACD warm-up: fetch previous trading day's bars (once per symbol per day)
        if getattr(self.detector, 'require_macd_positive', False):
            if symbol not in self._macd_warmup_cache:
                self._fetch_macd_warmup(symbol)
            warmup = self._macd_warmup_cache.get(symbol)
            self.detector.set_macd_warmup(warmup)

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

        # Deduplicate notifications — don't spam Telegram with the same
        # setup every 60s when position manager blocks (e.g., midday).
        # Only notify if breakout_level changed (new setup) or first time.
        already_notified = (
            self._notified_setups.get(symbol) == setup.breakout_level
        )

        if not already_notified:
            self._notified_setups[symbol] = setup.breakout_level

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

        # Notify trade planned (only if new setup)
        if not already_notified and self.notifier:
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

        # Production enforcement: buy-stop submitted on all days; post-fill volume check on thin days
        is_thin = self.market_regime and self.market_regime.is_thin_liquidity(date.today())

        # Self-managed stops: widen bracket SL to safety-net level,
        # real stop is monitored by StopMonitor via WebSocket.
        # Pass safety-net SL as override — DON'T mutate plan, so DB records
        # correct risk_per_share and stop_loss_price.
        real_stop_level = plan.stop_loss_price
        sl_override = None
        if self.stop_monitor:
            sl_override = round(plan.entry_price * (1 - self.safety_net_sl_pct), 2)
            logger.info(
                f"{symbol}: Self-managed stops — real stop ${real_stop_level:.2f}, "
                f"safety-net SL ${sl_override:.2f} ({self.safety_net_sl_pct:.0%})"
            )

        result = self.executor.submit_buy_stop_bracket_order(
            plan, sl_override=sl_override
        )

        if result is not None:
            # NOTE: _daily_trade_count and mark_traded are deferred to fill
            # time (_manage_pending_orders status=='filled'). This allows
            # re-entry after cancel/expire and accurate trade counting.
            pending = {
                'order_id': result['order_id'],
                'plan': plan,
                'setup': setup,
                'placed_at': datetime.now(timezone.utc),
            }
            # Store real stop for StopMonitor registration on fill
            if self.stop_monitor:
                pending['real_stop_level'] = real_stop_level
            if is_thin:
                pending['thin_liquidity'] = True
                pending['min_breakout_vol_ratio'] = self.market_regime.get_min_breakout_volume_ratio(date.today())
                logger.info(
                    f"{symbol}: BUY-STOP ORDER PLACED (thin liquidity) — "
                    f"min BVR {pending['min_breakout_vol_ratio']:.1f}x, {result}"
                )
            else:
                logger.info(f"{symbol}: BUY-STOP ORDER PLACED — {result}")
            self._pending_orders[symbol] = pending

            # Notify order submitted
            if self.notifier:
                self.notifier.notify_order_submitted(
                    symbol=symbol,
                    order_id=result.get('order_id', ''),
                    shares=plan.shares,
                    entry=plan.entry_price,
                )

        return result

    def _check_breakout_volume(self, symbol: str, pending: Dict) -> bool:
        """
        Check if the breakout bar had sufficient volume on a thin-liquidity day.

        Lookback window is computed from order placement time to now (the buy-stop
        could have filled anytime in that window), ensuring the breakout bar is
        captured even if fill detection is delayed by the poll interval.

        Finds the first bar where high >= breakout_level and computes
        BVR (breakout volume ratio) = bar_volume / avg_flag_volume.

        Fails open: returns True if bars unavailable or no breakout bar found.
        Fails safe: returns False if avg_flag_volume <= 0.

        Args:
            symbol: Stock symbol
            pending: Pending order dict with setup and min_breakout_vol_ratio

        Returns:
            True if volume is adequate (keep trade), False if weak (reject trade)
        """
        setup = pending['setup']
        min_bvr = pending.get('min_breakout_vol_ratio', 2.0)

        # Lookback must cover from order placement to now (fill could happen anytime)
        # Add 2-min buffer for bar completion lag and poll delay
        placed_at = pending.get('placed_at')
        if placed_at:
            elapsed_minutes = (datetime.now(timezone.utc) - placed_at).total_seconds() / 60.0
            lookback = int(elapsed_minutes) + 2
        else:
            lookback = 15  # fallback: conservative wide window
        lookback = max(lookback, 5)  # minimum 5 minutes
        lookback = min(lookback, 30)  # cap at 30 minutes (same as detection window)

        # Fetch recent 1-min bars
        try:
            bars = self.alpaca.get_1min_bars(symbol, lookback_minutes=lookback)
        except Exception as e:
            logger.warning(
                f"{symbol}: Failed to fetch bars for breakout volume check: {e} — "
                f"failing open (keeping trade)"
            )
            return True

        if bars is None or bars.empty:
            logger.warning(
                f"{symbol}: No bars available for breakout volume check — "
                f"failing open (keeping trade)"
            )
            return True

        # Find first bar where high >= breakout_level
        breakout_bar = None
        for _, bar in bars.iterrows():
            if bar['high'] >= setup.breakout_level:
                breakout_bar = bar
                break

        if breakout_bar is None:
            logger.warning(
                f"{symbol}: No breakout bar found in recent bars — "
                f"failing open (keeping trade)"
            )
            return True

        # Compute BVR
        avg_flag_vol = setup.avg_flag_volume
        if avg_flag_vol <= 0:
            logger.warning(
                f"{symbol}: avg_flag_volume <= 0 — failing safe (rejecting trade)"
            )
            return False

        bar_volume = breakout_bar['volume']
        bvr = bar_volume / avg_flag_vol

        if bvr >= min_bvr:
            logger.info(
                f"{symbol}: Breakout volume CONFIRMED — "
                f"BVR {bvr:.1f}x >= {min_bvr:.1f}x (keeping trade)"
            )
            return True
        else:
            logger.info(
                f"{symbol}: Breakout volume WEAK — "
                f"BVR {bvr:.1f}x < {min_bvr:.1f}x (rejecting trade)"
            )
            return False

    def _emergency_close_position(
        self, symbol: str, order_id: str, fill_price: float,
        actual_qty: int, trade_record: dict,
        exit_reason: str = 'thin_liquidity_reject'
    ) -> None:
        """
        Close a position immediately after fill.

        Used when a post-fill check fails: weak breakout volume on thin days,
        gap-fill leg replacement failure, etc.

        Handles the full lifecycle: close position, poll for exit price,
        compute PnL, update DB, record in circuit breaker, notify via Telegram.

        Note: _sync_closed_positions() filters by exit_price IS NULL,
        so once we set exit_price here, it won't double-process.

        Args:
            symbol: Stock symbol
            order_id: Original buy-stop order ID
            fill_price: Entry fill price
            actual_qty: Number of shares filled
            trade_record: DB trade record dict (may be None)
            exit_reason: Reason string for DB (e.g. 'thin_liquidity_reject',
                'gap_adjust_failed')
        """
        logger.info(
            f"{symbol}: EMERGENCY CLOSE ({exit_reason}) — closing position immediately"
        )

        # Close the position
        close_order_id = None
        try:
            close_result = self.alpaca.close_position(symbol)
            close_order_id = close_result.get('id', '')
            logger.info(f"{symbol}: Close order submitted — ID: {close_order_id}")
        except Exception as e:
            error_msg = f"{symbol}: Failed to close position ({exit_reason}): {e}"
            logger.error(error_msg)
            if self.notifier:
                self.notifier.notify_error(error_msg, component="EmergencyClose")
            return

        # Poll for exit price (reuse fill-price retry pattern)
        exit_price = None
        if close_order_id:
            for attempt in range(5):
                time_mod.sleep(0.5)
                try:
                    close_order = self.alpaca.get_order(close_order_id)
                    if close_order.get('status') == 'filled':
                        exit_price = close_order.get('filled_avg_price')
                        if exit_price is not None:
                            logger.info(
                                f"{symbol}: Close filled at ${exit_price:.2f} "
                                f"(attempt {attempt + 1})"
                            )
                            break
                except Exception:
                    pass

        if exit_price is None:
            logger.warning(
                f"{symbol}: Could not get exit price for {exit_reason} — "
                f"using fill_price as estimate"
            )
            exit_price = fill_price

        # Compute PnL
        pnl = (exit_price - fill_price) * actual_qty
        pnl_pct = (exit_price / fill_price - 1) * 100 if fill_price > 0 else 0.0

        # Update DB
        if trade_record:
            self.db.update_trade(trade_record['id'], {
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'exited_at': datetime.now(timezone.utc),
                'pnl': pnl,
                'pnl_pct': pnl_pct,
            })
            logger.info(
                f"{symbol}: {exit_reason} DB updated — "
                f"exit ${exit_price:.2f}, P&L ${pnl:+,.2f} ({pnl_pct:+.1f}%)"
            )
        else:
            logger.error(
                f"{symbol}: No trade record to update for {exit_reason}"
            )

        # Circuit breaker
        self.position_manager.record_trade_pnl(pnl)

        # Notify
        if self.notifier:
            self.notifier.notify_position_closed(
                symbol=symbol,
                entry_price=fill_price,
                exit_price=exit_price,
                shares=actual_qty,
                pnl=pnl,
                exit_reason=exit_reason,
            )

    def _force_close_all(self) -> None:
        """
        Cancel all pending orders and close all open positions.

        Called at force_close_time to ensure we're flat before market close.
        Syncs closed positions first so any SL/TP exits that already happened
        are recorded before we attempt to close remaining positions.
        """
        # Stop StopMonitor before force-closing — prevents race conditions
        # where monitor tries to exit while we're also closing
        if self.stop_monitor:
            for symbol in list(self.stop_monitor.watched_symbols):
                self.stop_monitor.remove_watch(symbol)
            self._process_stop_monitor_exits()

        # Sync first — record any SL/TP exits that happened before force close
        self._sync_closed_positions()
        # Process any pending order fills (e.g., late fills just before force close)
        self._manage_pending_orders()

        # Cancel pending orders
        for symbol, pending in list(self._pending_orders.items()):
            try:
                self.alpaca.cancel_order(pending['order_id'])
                logger.info(f"{symbol}: Force-close — cancelled pending order {pending['order_id']}")
            except Exception as e:
                error_msg = f"{symbol}: Failed to cancel pending order during force-close: {e}"
                logger.error(error_msg)
                if self.notifier:
                    self.notifier.notify_error(error_msg, component="ForceClose")
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

                # Cancel any open sell orders (TP/SL legs) holding shares
                # before attempting close_position — otherwise Alpaca rejects
                # with "insufficient qty available" (shares held by orders).
                try:
                    from alpaca.trading.requests import GetOrdersRequest
                    from alpaca.trading.enums import QueryOrderStatus
                    open_orders = self.alpaca.trading_client.get_orders(
                        GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
                    )
                    for oo in open_orders:
                        try:
                            self.alpaca.cancel_order(str(oo.id))
                        except Exception:
                            pass
                    if open_orders:
                        time_mod.sleep(1)  # Let cancels settle
                except Exception as e:
                    logger.warning(f"{symbol}: Failed to cancel open orders before force-close: {e}")

                close_order_id = None
                for attempt in range(FORCE_CLOSE_RETRIES):
                    try:
                        close_result = self.alpaca.close_position(symbol)
                        close_order_id = close_result.get('id', '') if close_result else ''
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
                                    f"after {FORCE_CLOSE_RETRIES} attempts",
                                    component="ForceClose",
                                )

                if not close_succeeded:
                    continue

                # Poll for actual fill price (don't use stale position snapshot)
                exit_price = None
                if close_order_id:
                    for poll in range(5):
                        time_mod.sleep(0.5)
                        try:
                            close_order = self.alpaca.get_order(close_order_id)
                            if close_order.get('status') == 'filled':
                                exit_price = close_order.get('filled_avg_price')
                                if exit_price is not None:
                                    break
                        except Exception:
                            pass

                # Fallback to position snapshot if poll fails
                if exit_price is None:
                    qty = pos.get('qty', 0)
                    if qty > 0 and pos.get('market_value'):
                        exit_price = float(pos['market_value']) / qty
                    else:
                        exit_price = pos.get('avg_entry_price', 0)
                    logger.warning(
                        f"{symbol}: Force-close fill price unavailable, "
                        f"using snapshot ${exit_price:.2f}"
                    )

                logger.info(f"{symbol}: Force-close — position closed at ${exit_price:.2f}")

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
                    entry = trade['fill_price'] if trade and trade.get('fill_price') else 0
                    fc_shares = trade.get('filled_qty') or trade.get('shares', 0) if trade else 0
                    fc_pnl = pnl if trade and trade.get('fill_price') else 0
                    self.notifier.notify_position_closed(
                        symbol=symbol,
                        entry_price=entry,
                        exit_price=exit_price,
                        shares=fc_shares,
                        pnl=fc_pnl,
                        exit_reason='force_close',
                    )
        except Exception as e:
            error_msg = f"Failed to get open positions for force-close: {e}"
            logger.error(error_msg)
            if self.notifier:
                self.notifier.notify_error(error_msg, component="ForceClose")

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

        # Start StopMonitor WebSocket thread if configured
        if self.stop_monitor:
            self.stop_monitor.start()
            logger.info("StopMonitor started for self-managed stops")

        logger.info(
            f"Trading engine monitoring loop started — "
            f"interval: {self.pattern_poll_interval}s, "
            f"symbols: {len(self._qualified_symbols)}, "
            f"last entry: {self.last_entry_hour}:{self.last_entry_minute:02d} ET, "
            f"force close: {self.force_close_hour}:{self.force_close_minute:02d} ET, "
            f"self_managed_stops: {self.stop_monitor is not None}"
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

        # Stop StopMonitor regardless of exit reason (market close or SIGTERM)
        if self.stop_monitor:
            self.stop_monitor.stop()

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
        """Reset daily state for a new trading day, then sync from DB/Alpaca.

        After clearing in-memory state, rebuilds _traded_symbols,
        _pending_orders, and _daily_trade_count from today's DB trades
        and Alpaca open orders. This ensures crash recovery doesn't
        orphan live positions or allow double-entry.
        """
        self._qualified_symbols.clear()
        self._traded_symbols.clear()
        self._patterns_detected = 0
        self._patterns_traded = 0
        self._pattern_details.clear()
        self._pending_orders.clear()
        self._daily_trade_count = 0
        self._notified_setups.clear()
        self.position_manager.reset_daily()
        self._refresh_spy_data()
        self._sync_startup_state()
        logger.info("Trading engine: daily state reset")

    def _sync_startup_state(self) -> None:
        """Rebuild in-memory state from DB trades and Alpaca for today.

        Prevents crash recovery from:
        - Allowing double-entry on symbols already traded today
        - Losing track of pending buy-stop orders still live on Alpaca
        - Miscounting daily trades
        - Leaving orphan positions from prior days open
        """
        today = date.today().isoformat()
        try:
            trades_today = self.db.get_trades_by_date(today)
        except Exception as e:
            logger.error(f"Startup sync: failed to load today's trades: {e}")
            return

        # Rebuild _traded_symbols and _daily_trade_count from DB
        filled_count = 0
        for trade in trades_today:
            symbol = trade['symbol']
            self._traded_symbols.add(symbol)
            self.position_manager.mark_traded(symbol)
            if trade.get('fill_price') is not None:
                filled_count += 1

        self._daily_trade_count = filled_count

        # Rebuild _pending_orders from DB trades that have order_id but no fill
        for trade in trades_today:
            symbol = trade['symbol']
            order_id = trade.get('order_id')
            if order_id and trade.get('fill_price') is None and trade.get('exit_price') is None:
                plan = self._reconstruct_plan(trade)
                setup = self._reconstruct_setup(trade)
                self._pending_orders[symbol] = {
                    'order_id': order_id,
                    'plan': plan,
                    'setup': setup,
                    'placed_at': trade.get('created_at', datetime.now(timezone.utc)),
                }
                logger.info(f"{symbol}: Recovered pending order {order_id} from DB (plan={'yes' if plan else 'no'})")

        # Re-register filled-but-open positions with StopMonitor
        # (crash recovery: service restarted with live positions)
        if self.stop_monitor:
            for trade in trades_today:
                symbol = trade['symbol']
                if (trade.get('fill_price') is not None
                        and trade.get('exit_price') is None
                        and trade.get('real_stop_loss_price') is not None):
                    order_id = trade.get('order_id')
                    try:
                        order_detail = self.alpaca.get_order(order_id)
                        sl_leg, tp_leg = self._identify_bracket_legs(
                            order_detail.get('legs', []),
                            expected_sl=trade.get('entry_price', 0) * (1 - self.safety_net_sl_pct),
                            expected_tp=trade.get('take_profit_price'),
                        )
                        tp_leg_id = tp_leg['id'] if tp_leg else ''
                        sl_leg_id = sl_leg['id'] if sl_leg else ''
                        trail_r = self.trailing_stop_r if self.trailing_stop_enabled else 0.0
                        activate_r = self.trailing_activate_at_r if self.trailing_stop_enabled else 0.0
                        fill = trade['fill_price']
                        real_sl = trade['real_stop_loss_price']
                        self.stop_monitor.add_watch(
                            symbol=symbol,
                            stop_price=real_sl,
                            shares=trade.get('filled_qty') or trade['shares'],
                            tp_leg_id=tp_leg_id,
                            sl_leg_id=sl_leg_id,
                            trade_db_id=trade['id'],
                            entry_price=fill,
                            risk_per_share=fill - real_sl,
                            trail_r=trail_r,
                            activate_at_r=activate_r,
                        )
                        logger.info(
                            f"{symbol}: Crash recovery — re-registered StopMonitor watch "
                            f"stop=${real_sl:.2f}"
                            f"{f', trail={trail_r:.1f}R' if trail_r > 0 else ''}"
                        )
                    except Exception as e:
                        logger.error(
                            f"{symbol}: Crash recovery — failed to re-register "
                            f"StopMonitor watch: {e} (safety-net SL active)"
                        )

        # Detect orphan positions from prior days
        self._close_orphan_positions(trades_today)

        logger.info(
            f"Startup sync: {len(self._traded_symbols)} traded symbols, "
            f"{self._daily_trade_count} filled trades, "
            f"{len(self._pending_orders)} pending orders recovered"
        )

    def _reconstruct_plan(self, trade: Dict[str, Any]) -> Optional['TradePlan']:
        """Reconstruct a TradePlan from DB trade fields.

        Args:
            trade: Trade dict from database

        Returns:
            TradePlan if enough data exists, None otherwise
        """
        try:
            entry = trade.get('entry_price')
            sl = trade.get('stop_loss_price')
            tp = trade.get('take_profit_price')
            shares = trade.get('shares')
            if not all([entry, sl, tp, shares]):
                return None

            risk = trade.get('risk_per_share', entry - sl)
            reward = tp - entry
            rr = trade.get('risk_reward_ratio', reward / risk if risk > 0 else 0)

            return TradePlan(
                symbol=trade['symbol'],
                entry_price=entry,
                stop_loss_price=sl,
                take_profit_price=tp,
                risk_per_share=risk,
                reward_per_share=reward,
                risk_reward_ratio=rr,
                shares=shares,
                total_risk=trade.get('total_risk', risk * shares),
                pattern=None,
            )
        except Exception as e:
            logger.warning(f"{trade.get('symbol')}: Failed to reconstruct plan: {e}")
            return None

    def _reconstruct_setup(self, trade: Dict[str, Any]) -> Optional[Any]:
        """Reconstruct a BullFlagSetup from trade's pattern_data JSON.

        Args:
            trade: Trade dict from database (with pattern_data JSON field)

        Returns:
            BullFlagSetup if pattern_data is parseable, None otherwise
        """
        import json
        from trading.pattern_detector import BullFlagSetup

        pattern_data = trade.get('pattern_data')
        if not pattern_data:
            return None

        try:
            data = json.loads(pattern_data) if isinstance(pattern_data, str) else pattern_data
            if not data or not isinstance(data, dict):
                return None

            breakout = data.get('breakout_level')
            if breakout is None:
                return None

            return BullFlagSetup(
                symbol=trade['symbol'],
                pole_start_idx=data.get('pole_start_idx', 0),
                pole_end_idx=data.get('pole_end_idx', 0),
                flag_start_idx=data.get('flag_start_idx', 0),
                flag_end_idx=data.get('flag_end_idx', 0),
                pole_low=data.get('pole_low', 0),
                pole_high=data.get('pole_high', 0),
                pole_height=data.get('pole_height', 0),
                pole_gain_pct=data.get('pole_gain_pct', 0),
                flag_low=data.get('flag_low', 0),
                flag_high=data.get('flag_high', 0),
                retracement_pct=data.get('retracement_pct', 0),
                pullback_candle_count=data.get('pullback_candle_count', 0),
                avg_pole_volume=data.get('avg_pole_volume', 0),
                avg_flag_volume=data.get('avg_flag_volume', 0),
                breakout_level=breakout,
            )
        except Exception as e:
            logger.warning(f"{trade.get('symbol')}: Failed to reconstruct setup: {e}")
            return None

    def _close_orphan_positions(self, trades_today: List[Dict]) -> None:
        """Detect and close positions from prior days still open on Alpaca.

        An orphan is an Alpaca position with no matching trade in today's DB.
        This handles the case where the service crashed after market close
        without running force_close.

        Args:
            trades_today: Today's trades from DB (already fetched)
        """
        try:
            positions = self.alpaca.get_open_positions()
        except Exception as e:
            logger.error(f"Startup sync: failed to get Alpaca positions: {e}")
            return

        if not positions:
            return

        today_symbols = {t['symbol'] for t in trades_today}

        for pos in positions:
            symbol = pos['symbol']
            if symbol not in today_symbols:
                logger.warning(f"{symbol}: Orphan position from prior day — closing")
                try:
                    self.alpaca.close_position(symbol)
                    self._traded_symbols.add(symbol)
                    logger.info(f"{symbol}: Orphan position closed")
                except Exception as e:
                    logger.error(f"{symbol}: Failed to close orphan position: {e}")
