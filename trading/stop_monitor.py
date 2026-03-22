"""
Self-managed stop monitor via WebSocket price streaming.

Replaces bracket stop-market orders with real-time price monitoring
and marketable limit sell exits. Caps exit slippage at a known offset
instead of relying on stop-market fills that can slip 0-2%+ on
low-float stocks.

Architecture:
    - Daemon thread runs its own asyncio event loop with StockDataStream
    - Main thread calls add_watch/remove_watch/drain_exit_events
    - Thread-safe queue.Queue bridges exit events to main thread
    - Safety-net SL (5%) remains on Alpaca for crash/disconnect protection

Usage:
    monitor = StopMonitor(api_key, api_secret, alpaca_client)
    monitor.start()
    monitor.add_watch("PLYX", stop=4.29, shares=500,
                       tp_leg_id="tp-123", sl_leg_id="sl-456")
    # Main loop:
    for event in monitor.drain_exit_events():
        # process exit ...
    monitor.stop()
"""

import asyncio
import logging
import queue
import threading
import time as time_mod
from dataclasses import dataclass
from typing import Dict, List, Optional

from data_sources.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

RECONNECT_DELAY_SECONDS = 5.0
MAX_RECONNECT_ATTEMPTS = 50


@dataclass
class WatchEntry:
    """Configuration for a symbol being monitored for stop exit."""

    symbol: str
    stop_price: float
    shares: int
    tp_leg_id: str
    sl_leg_id: str
    trade_db_id: Optional[int] = None
    # Trailing stop fields (all default to 0/False = disabled)
    entry_price: float = 0.0
    risk_per_share: float = 0.0
    trail_r: float = 0.0
    activate_at_r: float = 0.0
    highest_since_entry: float = 0.0
    trailing_active: bool = False
    # Exhaustion exit: set True after partial sell into strength
    exhaustion_partial_taken: bool = False
    # Quote cache (updated by WebSocket quote stream)
    latest_bid: float = 0.0
    latest_ask: float = 0.0
    latest_bid_size: int = 0
    latest_ask_size: int = 0
    latest_quote_ts: float = 0.0
    # OFI (Order Flow Imbalance) — rolling sum over last 20 quote ticks
    # Negative = selling pressure, positive = buying pressure
    ofi_cumulative: float = 0.0
    _prev_bid: float = 0.0
    _prev_ask: float = 0.0
    _prev_bid_size: int = 0
    _prev_ask_size: int = 0


@dataclass
class StopExitEvent:
    """Event emitted when a stop exit is triggered and executed."""

    symbol: str
    stop_price: float
    exit_price: float
    shares: int
    order_id: str
    exit_reason: str  # 'stop_loss' or 'stop_loss_fallback'
    trade_db_id: Optional[int] = None
    submitted_at: float = 0.0  # time.time() when exit order was submitted
    pricing_method: str = 'fixed_offset'  # quote_tight, quote_medium, quote_wide, fixed_offset
    filled_qty: int = 0  # actual shares filled (may differ from shares on partial fill)
    # Exit microstructure (for execution analysis)
    exit_trigger_price: float = 0.0  # trade price that breached stop
    exit_quote_bid: float = 0.0
    exit_quote_ask: float = 0.0
    exit_quote_bid_size: int = 0
    exit_quote_ask_size: int = 0
    exit_limit_price: float = 0.0  # limit price we submitted
    exit_ofi: float = 0.0  # OFI at exit time (negative = selling pressure)


class StopMonitor:
    """
    Monitors real-time trade prices via WebSocket and triggers marketable
    limit sell exits when price breaches stop levels.

    Thread-safe: add_watch/remove_watch/drain_exit_events are called from
    the main thread. WebSocket callbacks run in a daemon thread.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        alpaca_client: AlpacaClient,
        marketable_limit_offset: float = 0.03,
        marketable_limit_offset_pct: float = 0.005,
        notifier=None,
    ):
        """
        Initialize StopMonitor.

        Args:
            api_key: Alpaca API key (for StockDataStream)
            api_secret: Alpaca API secret
            alpaca_client: AlpacaClient instance for order submission
            marketable_limit_offset: Minimum dollar offset below price for
                marketable limit sell (default $0.03)
            marketable_limit_offset_pct: Percentage offset below price
                (default 0.5%). Whichever is larger is used.
            notifier: Optional TelegramNotifier for alerts
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._alpaca = alpaca_client
        self._marketable_limit_offset = marketable_limit_offset
        self._marketable_limit_offset_pct = marketable_limit_offset_pct
        self._notifier = notifier

        self._watches: Dict[str, WatchEntry] = {}
        self._watch_lock = threading.Lock()

        self._exit_events: queue.Queue = queue.Queue()
        self._exit_in_progress: Dict[str, bool] = {}
        self._exit_lock = threading.Lock()

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stream = None
        self._running = False
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Launch the WebSocket daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("StopMonitor already running")
            return

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_stream_loop,
            name="StopMonitor-WS",
            daemon=True,
        )
        self._thread.start()
        logger.info("StopMonitor started (daemon thread)")

    def stop(self) -> None:
        """Gracefully shut down the WebSocket thread."""
        self._running = False
        self._stop_event.set()

        if self._loop and self._stream:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._close_stream(), self._loop
                )
            except Exception as e:
                logger.warning(f"StopMonitor: error during stream close: {e}")

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)

        logger.info("StopMonitor stopped")

    def add_watch(
        self,
        symbol: str,
        stop_price: float,
        shares: int,
        tp_leg_id: str,
        sl_leg_id: str,
        trade_db_id: Optional[int] = None,
        entry_price: float = 0.0,
        risk_per_share: float = 0.0,
        trail_r: float = 0.0,
        activate_at_r: float = 0.0,
    ) -> None:
        """
        Register a symbol for stop-price monitoring.

        Called from main thread when a buy-stop order fills.

        Args:
            symbol: Stock symbol
            stop_price: Real stop level (flag_low region)
            shares: Position size
            tp_leg_id: Alpaca order ID of the take-profit leg
            sl_leg_id: Alpaca order ID of the safety-net stop-loss leg
            trade_db_id: Database trade record ID
            entry_price: Fill price (needed for trailing stop R calculation)
            risk_per_share: entry_price - original_stop (= 1R, for trail distance)
            trail_r: Trail distance in R units below highest high (0 = disabled)
            activate_at_r: Activate trail after price reaches +NR from entry
        """
        entry = WatchEntry(
            symbol=symbol,
            stop_price=stop_price,
            shares=shares,
            tp_leg_id=tp_leg_id,
            sl_leg_id=sl_leg_id,
            trade_db_id=trade_db_id,
            entry_price=entry_price,
            risk_per_share=risk_per_share,
            trail_r=trail_r,
            activate_at_r=activate_at_r,
            highest_since_entry=entry_price,
        )
        with self._watch_lock:
            self._watches[symbol] = entry

        # Subscribe to trades for this symbol on the WebSocket
        if self._loop and self._stream and self._running:
            asyncio.run_coroutine_threadsafe(
                self._subscribe_symbol(symbol), self._loop
            )

        trail_msg = ""
        if trail_r > 0:
            trail_msg = f", trail={trail_r:.1f}R (activates +{activate_at_r:.1f}R)"
        logger.info(
            f"StopMonitor: watching {symbol} — "
            f"stop=${stop_price:.2f}, shares={shares}{trail_msg}"
        )

    def remove_watch(self, symbol: str) -> None:
        """
        Stop monitoring a symbol.

        Called from main thread on exit/force-close.

        Args:
            symbol: Stock symbol to stop watching
        """
        with self._watch_lock:
            removed = self._watches.pop(symbol, None)

        with self._exit_lock:
            self._exit_in_progress.pop(symbol, None)

        if removed:
            # Unsubscribe from WebSocket
            if self._loop and self._stream and self._running:
                asyncio.run_coroutine_threadsafe(
                    self._unsubscribe_symbol(symbol), self._loop
                )
            logger.info(f"StopMonitor: removed watch for {symbol}")

    def drain_exit_events(self) -> List[StopExitEvent]:
        """
        Drain all pending exit events from the queue.

        Called from main thread each monitoring cycle.

        Returns:
            List of StopExitEvent objects to process
        """
        events = []
        while True:
            try:
                event = self._exit_events.get_nowait()
                events.append(event)
            except queue.Empty:
                break
        return events

    def update_stop(self, symbol: str, new_stop_price: float) -> bool:
        """
        Update the stop price for a watched symbol. Only moves stop UP.

        Args:
            symbol: Stock symbol
            new_stop_price: New stop price (ignored if lower than current)

        Returns:
            True if stop was updated, False if not found or not higher
        """
        with self._watch_lock:
            watch = self._watches.get(symbol)
            if watch and new_stop_price > watch.stop_price:
                old = watch.stop_price
                watch.stop_price = new_stop_price
                logger.info(
                    f"StopMonitor: {symbol} stop updated "
                    f"${old:.2f} → ${new_stop_price:.2f}"
                )
                return True
        return False

    @property
    def watched_symbols(self) -> List[str]:
        """Return list of currently watched symbols."""
        with self._watch_lock:
            return list(self._watches.keys())

    def get_watch_snapshot(self, symbol: str) -> Optional[Dict]:
        """
        Return a thread-safe dict copy of a WatchEntry's fields.

        Called from main thread (TradingEngine) to check R-gain and
        whether exhaustion partial was already taken.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with watch fields, or None if not watched
        """
        with self._watch_lock:
            w = self._watches.get(symbol)
            if not w:
                return None
            return {
                'entry_price': w.entry_price,
                'risk_per_share': w.risk_per_share,
                'highest_since_entry': w.highest_since_entry,
                'shares': w.shares,
                'exhaustion_partial_taken': w.exhaustion_partial_taken,
                'trade_db_id': w.trade_db_id,
                'stop_price': w.stop_price,
                'trail_r': w.trail_r,
                'trailing_active': w.trailing_active,
                'latest_bid': w.latest_bid,
                'latest_ask': w.latest_ask,
                'latest_bid_size': w.latest_bid_size,
                'latest_ask_size': w.latest_ask_size,
                'latest_quote_ts': w.latest_quote_ts,
                'ofi_cumulative': w.ofi_cumulative,
            }

    def execute_partial_exit(
        self,
        symbol: str,
        fraction: float,
        tighter_trail_r: float,
    ) -> Optional['StopExitEvent']:
        """
        Execute a partial exhaustion exit: sell fraction of shares, tighten trail.

        Thread-safe: reads watch under lock, releases for API call,
        re-acquires to update state.

        Called from main thread (TradingEngine._check_exhaustion_exits).

        Args:
            symbol: Stock symbol
            fraction: Fraction of shares to sell (e.g. 0.5)
            tighter_trail_r: New trail distance in R units for remainder

        Returns:
            StopExitEvent for the partial sell, or None on failure
        """
        # Step 1: Read watch state under lock + block concurrent stop exits.
        # CRITICAL: Set _exit_in_progress to prevent _execute_stop_exit from
        # firing on the WS thread while we hold an open partial sell order.
        # Without this, a simultaneous trail stop + partial sell = double sell = SHORT.
        with self._exit_lock:
            if self._exit_in_progress.get(symbol, False):
                logger.debug(
                    f"StopMonitor: {symbol} exit already in progress, skipping partial"
                )
                return None
            self._exit_in_progress[symbol] = True

        with self._watch_lock:
            watch = self._watches.get(symbol)
            if not watch:
                logger.warning(
                    f"StopMonitor: execute_partial_exit({symbol}) — not watched"
                )
                with self._exit_lock:
                    self._exit_in_progress[symbol] = False
                return None
            if watch.exhaustion_partial_taken:
                logger.debug(
                    f"StopMonitor: {symbol} exhaustion partial already taken"
                )
                with self._exit_lock:
                    self._exit_in_progress[symbol] = False
                return None
            sell_shares = int(watch.shares * fraction)
            if sell_shares < 1:
                logger.warning(
                    f"StopMonitor: {symbol} partial shares < 1 "
                    f"({watch.shares} × {fraction})"
                )
                with self._exit_lock:
                    self._exit_in_progress[symbol] = False
                return None
            entry_price = watch.entry_price
            risk_per_share = watch.risk_per_share
            highest = watch.highest_since_entry
            trade_db_id = watch.trade_db_id

        # NOTE: We intentionally keep the safety-net SL ALIVE during partial sell.
        # The SL is at 5% below entry (~$4.75 on $5) while exhaustion fires at +3R
        # (~$6.50). Zero overlap risk in normal conditions.
        #
        # In a flash crash during the 30s wait:
        # - Partial sell limit (near $6.50) won't fill → we cancel it → no state change
        # - SL fires for full qty → closes everything → position safe
        # - If partial DID fill first (500), SL fires for remaining (Alpaca caps at position)
        #
        # After confirmed fill: replace SL qty to match remaining shares.
        # This eliminates the 30s naked window and SL-restore-on-cancelled-ID bugs.

        # Step 2: API call outside lock — get quote and submit sell
        exit_price = 0.0
        order_id = ""
        try:
            # Quote-based pricing: try NBBO, fall back to fixed offset
            quote = self._fetch_quote_for_exit(symbol)
            if quote:
                bid = quote.get('bid_price', 0.0)
                ask = quote.get('ask_price', 0.0)
                limit_price, pricing_method = self.compute_limit_price_from_quote(bid, ask)
                if limit_price <= 0:
                    limit_price = self.compute_limit_price(highest)
                    pricing_method = 'fixed_offset'
                else:
                    logger.info(
                        f"StopMonitor: {symbol} partial quote pricing — "
                        f"bid=${bid:.2f} ask=${ask:.2f} → ${limit_price:.2f} ({pricing_method})"
                    )
            else:
                limit_price = self.compute_limit_price(highest)
                pricing_method = 'fixed_offset'

            result = self._alpaca.submit_limit_sell_order(
                symbol=symbol,
                qty=sell_shares,
                limit_price=limit_price,
            )
            order_id = result.get("id", "")
            exit_price = limit_price
            logger.info(
                f"StopMonitor: {symbol} EXHAUSTION partial sell — "
                f"{sell_shares}sh @ limit ${limit_price:.2f} ({pricing_method}), "
                f"order={order_id}"
            )
        except Exception as e:
            logger.error(
                f"StopMonitor: {symbol} exhaustion partial sell FAILED: {e}"
            )
            with self._exit_lock:
                self._exit_in_progress[symbol] = False
            return None

        # Step 3: Wait for fill confirmation before updating state.
        # Without fill confirmation, StopMonitor would track reduced shares
        # while the full position still exists → orphaned shares on trail exit.
        actual_fill = None
        deadline = time_mod.time() + 30
        while time_mod.time() < deadline:
            time_mod.sleep(1)
            try:
                order_status = self._alpaca.get_order(order_id)
                status = order_status.get('status', '')
                if status == 'filled':
                    actual_fill = order_status.get('filled_avg_price') or exit_price
                    logger.info(
                        f"StopMonitor: {symbol} partial sell FILLED at "
                        f"${actual_fill:.2f}"
                    )
                    break
                elif status == 'partially_filled':
                    # Some shares filled, cancel remainder and handle partial
                    logger.warning(
                        f"StopMonitor: {symbol} partial sell PARTIALLY filled — "
                        f"cancelling remainder"
                    )
                    try:
                        self._alpaca.cancel_order(order_id)
                    except Exception:
                        pass
                    actual_fill = order_status.get('filled_avg_price') or exit_price
                    break
                elif status in ('cancelled', 'expired', 'rejected'):
                    logger.warning(
                        f"StopMonitor: {symbol} partial sell {status} "
                        f"— aborting exhaustion exit"
                    )
                    with self._exit_lock:
                        self._exit_in_progress[symbol] = False
                    return None
            except Exception:
                pass

        if actual_fill is None:
            # Timeout — cancel and abort (don't update state)
            logger.warning(
                f"StopMonitor: {symbol} partial sell UNFILLED after 30s — "
                f"cancelling, keeping full position"
            )
            try:
                self._alpaca.cancel_order(order_id)
                # Check one more time (race with fill)
                order_status = self._alpaca.get_order(order_id)
                if order_status.get('status') == 'filled':
                    actual_fill = order_status.get('filled_avg_price') or exit_price
                    logger.info(
                        f"StopMonitor: {symbol} partial filled during cancel at "
                        f"${actual_fill:.2f}"
                    )
                else:
                    with self._exit_lock:
                        self._exit_in_progress[symbol] = False
                    return None
            except Exception:
                with self._exit_lock:
                    self._exit_in_progress[symbol] = False
                return None

        # Step 3b: Verify filled_qty matches sell_shares — handle partial-of-partial
        # close_shortfall_only=True: only sell the shortfall, keep remainder for trail
        actual_filled_qty, verified_fill_price = self._verify_fill_qty(
            symbol, order_id, sell_shares, close_shortfall_only=True
        )
        if actual_filled_qty <= 0:
            logger.error(
                f"StopMonitor: {symbol} partial sell fill verification failed"
            )
            with self._exit_lock:
                self._exit_in_progress[symbol] = False
            return None

        if actual_filled_qty != sell_shares:
            logger.warning(
                f"StopMonitor: {symbol} partial-of-partial: "
                f"wanted {sell_shares}, got {actual_filled_qty} "
                f"(emergency close handled remaining)"
            )
            sell_shares = actual_filled_qty

        exit_price = verified_fill_price if verified_fill_price > 0 else actual_fill

        # Step 4: Fill confirmed — NOW update watch state
        remaining = 0
        sl_leg_id = ""
        with self._watch_lock:
            watch = self._watches.get(symbol)
            if not watch:
                logger.warning(
                    f"StopMonitor: {symbol} removed during partial exit"
                )
                with self._exit_lock:
                    self._exit_in_progress[symbol] = False
                return None

            remaining = watch.shares - sell_shares
            watch.shares = remaining
            watch.exhaustion_partial_taken = True
            watch.trail_r = tighter_trail_r
            sl_leg_id = watch.sl_leg_id

            # Ratchet stop with tighter trail
            if risk_per_share > 0:
                new_stop = highest - risk_per_share * tighter_trail_r
                if new_stop > watch.stop_price:
                    old_stop = watch.stop_price
                    watch.stop_price = new_stop
                    logger.info(
                        f"StopMonitor: {symbol} trail tightened "
                        f"{tighter_trail_r}R — stop ${old_stop:.2f} → "
                        f"${new_stop:.2f}, remaining {remaining}sh"
                    )

        # Step 5: Update safety-net SL leg qty on Alpaca to match remaining shares.
        # Without this, a crash/gap-down would trigger the SL for the original full
        # qty, potentially shorting us or leaving a rejected order.
        if sl_leg_id and remaining > 0:
            try:
                self._alpaca.replace_order_qty(sl_leg_id, remaining)
                logger.info(
                    f"StopMonitor: {symbol} safety-net SL qty updated "
                    f"to {remaining} shares"
                )
            except Exception as e:
                logger.warning(
                    f"StopMonitor: {symbol} failed to update SL qty: {e} — "
                    f"SL leg may still be for original qty"
                )

        # Step 6: Clear exit-in-progress and emit event
        with self._exit_lock:
            self._exit_in_progress[symbol] = False

        event = StopExitEvent(
            symbol=symbol,
            stop_price=0.0,  # Not a stop exit — exhaustion partial
            exit_price=exit_price,
            shares=sell_shares,
            order_id=order_id,
            exit_reason='exhaustion_partial',
            trade_db_id=trade_db_id,
            submitted_at=time_mod.time(),
            pricing_method=pricing_method,
            filled_qty=sell_shares,  # verified by _verify_fill_qty
        )
        self._exit_events.put(event)
        return event

    def compute_limit_price(self, current_price: float) -> float:
        """
        Compute the marketable limit sell price.

        Uses the larger of:
        - Fixed offset: current_price - marketable_limit_offset
        - Percentage offset: current_price * (1 - marketable_limit_offset_pct)

        This ensures the limit is aggressive enough to fill immediately
        while capping worst-case slippage.

        Args:
            current_price: Current trade price

        Returns:
            Limit price for the sell order
        """
        fixed_offset = self._marketable_limit_offset
        pct_offset = current_price * self._marketable_limit_offset_pct
        offset = max(fixed_offset, pct_offset)
        limit_price = round(current_price - offset, 2)
        return max(limit_price, 0.01)  # floor at $0.01

    @staticmethod
    def compute_limit_price_from_quote(bid: float, ask: float) -> tuple:
        """
        Compute limit sell price from current NBBO quote using spread tiers.

        Adapts pricing to market microstructure:
        - Tight spread (<$0.05): sell at midpoint — liquid, saves vs fixed offset
        - Medium spread ($0.05-$0.15): sell at bid + $0.01 — fast fill, minimal give
        - Wide spread (>$0.15): sell at bid — take what's available on illiquid

        Args:
            bid: Current best bid price
            ask: Current best ask price

        Returns:
            Tuple of (limit_price, pricing_method) where pricing_method is
            'quote_tight', 'quote_medium', or 'quote_wide'.
            Returns (0.0, 'invalid') if quote data is invalid.
        """
        if bid <= 0 or ask <= 0 or ask < bid:
            return (0.0, 'invalid')

        spread = ask - bid

        if spread < 0.05:
            limit = round((bid + ask) / 2, 2)
            method = 'quote_tight'
        elif spread <= 0.15:
            limit = round(bid + 0.01, 2)
            method = 'quote_medium'
        else:
            limit = round(bid, 2)
            method = 'quote_wide'

        return (max(limit, 0.01), method)

    def _verify_fill_qty(
        self, symbol: str, order_id: str, expected_qty: int,
        close_shortfall_only: bool = False,
    ) -> tuple:
        """
        Verify an order's filled_qty matches expected. Handle partial fills.

        If partial fill detected, sells the shortfall shares to complete
        the intended quantity.

        Args:
            symbol: Stock symbol
            order_id: Order to check
            expected_qty: How many shares we expected to fill
            close_shortfall_only: If True, only sell the shortfall qty
                (not the entire position). Used for exhaustion partials
                where we want to keep the remainder for trailing stop.

        Returns:
            Tuple of (total_filled_qty, avg_fill_price) after handling
            any partial fill remainder. (0, 0.0) if unable to determine.
        """
        try:
            order = self._alpaca.get_order(order_id)
            filled_qty = int(order.get('filled_qty', 0) or 0)
            fill_price = float(order.get('filled_avg_price', 0) or 0)

            if filled_qty <= 0:
                logger.warning(
                    f"StopMonitor: {symbol} filled_qty=0 — "
                    f"using expected_qty={expected_qty} as fallback"
                )
                return (expected_qty, fill_price)

            if filled_qty == expected_qty:
                return (filled_qty, fill_price)

            # PARTIAL FILL — close remaining shares immediately
            remaining = expected_qty - filled_qty
            logger.error(
                f"StopMonitor: {symbol} PARTIAL FILL — "
                f"filled {filled_qty}/{expected_qty}, "
                f"{remaining} shares UNPROTECTED — emergency closing"
            )

            # Verify actual position to use broker as source of truth
            try:
                positions = self._alpaca.get_open_positions()
                broker_qty = 0
                for pos in positions:
                    if pos.get('symbol') == symbol:
                        broker_qty = abs(int(float(pos.get('qty', 0))))
                        break

                if broker_qty > 0:
                    # For partial exits, only sell the shortfall — keep remainder
                    # for trailing stop. For full exits, close entire position.
                    sell_qty = remaining if close_shortfall_only else broker_qty
                    if close_shortfall_only:
                        close_result = self._alpaca.submit_limit_sell_order(
                            symbol=symbol,
                            qty=min(sell_qty, broker_qty),
                            limit_price=round(fill_price * 0.95, 2),  # aggressive limit
                        )
                    else:
                        close_result = self._alpaca.close_position(symbol)
                    close_id = close_result.get('id', '')
                    # Poll for market fill
                    for _ in range(10):
                        time_mod.sleep(0.5)
                        try:
                            close_order = self._alpaca.get_order(close_id)
                            if close_order.get('status') == 'filled':
                                close_fill = float(
                                    close_order.get('filled_avg_price', 0) or 0
                                )
                                close_qty = int(
                                    close_order.get('filled_qty', 0) or 0
                                )
                                # Blended fill price
                                total_qty = filled_qty + close_qty
                                if total_qty > 0 and fill_price > 0 and close_fill > 0:
                                    blended = (
                                        fill_price * filled_qty
                                        + close_fill * close_qty
                                    ) / total_qty
                                else:
                                    blended = fill_price
                                logger.info(
                                    f"StopMonitor: {symbol} partial fill resolved — "
                                    f"{filled_qty}@${fill_price:.2f} + "
                                    f"{close_qty}@${close_fill:.2f} = "
                                    f"{total_qty}@${blended:.2f}"
                                )
                                return (total_qty, blended)
                        except Exception:
                            pass

                    logger.error(
                        f"StopMonitor: {symbol} emergency close fill unknown"
                    )
                    return (filled_qty, fill_price)
                else:
                    logger.info(
                        f"StopMonitor: {symbol} position already 0 — "
                        f"partial fill was closed externally"
                    )
                    return (filled_qty, fill_price)

            except Exception as e:
                logger.error(
                    f"StopMonitor: {symbol} emergency close failed: {e} — "
                    f"only {filled_qty}/{expected_qty} filled"
                )
                return (filled_qty, fill_price)

        except Exception as e:
            logger.error(
                f"StopMonitor: {symbol} fill verification failed: {e}"
            )
            return (0, 0.0)

    def _fetch_quote_for_exit(self, symbol: str) -> Optional[Dict]:
        """
        Fetch latest NBBO quote for exit pricing. Returns None on failure.

        Logs a warning on failure — caller falls back to fixed-offset pricing.

        Args:
            symbol: Stock symbol

        Returns:
            Quote dict with bid_price, ask_price, or None
        """
        try:
            return self._alpaca.get_latest_quote(symbol)
        except Exception as e:
            logger.warning(
                f"StopMonitor: {symbol} quote fetch failed: {e} — "
                f"falling back to fixed-offset pricing"
            )
            return None

    # =========================================================================
    # Internal — WebSocket thread
    # =========================================================================

    def _run_stream_loop(self) -> None:
        """Entry point for the daemon thread. Runs asyncio event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._stream_with_reconnect())
        except Exception as e:
            if self._running:
                logger.error(f"StopMonitor: stream loop crashed: {e}")
        finally:
            self._loop.close()
            self._loop = None

    async def _stream_with_reconnect(self) -> None:
        """Run the WebSocket stream with automatic reconnection."""
        from alpaca.data.live import StockDataStream

        reconnect_count = 0

        while self._running and reconnect_count < MAX_RECONNECT_ATTEMPTS:
            try:
                self._stream = StockDataStream(
                    self._api_key, self._api_secret
                )
                # Subscribe to all currently watched symbols.
                # If empty, subscribe_trades with no symbols is a no-op;
                # add_watch will subscribe dynamically via _subscribe_symbol.
                watched = self._get_watched_symbols()
                if watched:
                    self._stream.subscribe_trades(
                        self._on_trade, *watched
                    )
                    self._stream.subscribe_quotes(
                        self._on_quote, *watched
                    )
                    logger.info(f"StopMonitor: WebSocket connecting with {len(watched)} symbols (trades+quotes)...")
                else:
                    logger.info("StopMonitor: WebSocket connecting (no symbols yet)...")
                await self._stream._run_forever()

            except Exception as e:
                if not self._running:
                    break
                reconnect_count += 1
                error_msg = (
                    f"StopMonitor: WebSocket disconnected ({e}) — "
                    f"reconnecting in {RECONNECT_DELAY_SECONDS}s "
                    f"(attempt {reconnect_count}/{MAX_RECONNECT_ATTEMPTS})"
                )
                logger.error(error_msg)
                if self._notifier and reconnect_count <= 3:
                    try:
                        self._notifier.notify_error(
                            error_msg, component="StopMonitor"
                        )
                    except Exception:
                        pass

                await asyncio.sleep(RECONNECT_DELAY_SECONDS)

        if reconnect_count >= MAX_RECONNECT_ATTEMPTS:
            error_msg = (
                f"StopMonitor: exhausted {MAX_RECONNECT_ATTEMPTS} reconnect "
                f"attempts — safety-net SL on Alpaca is active"
            )
            logger.error(error_msg)
            if self._notifier:
                try:
                    self._notifier.notify_error(
                        error_msg, component="StopMonitor"
                    )
                except Exception:
                    pass

    async def _on_quote(self, quote) -> None:
        """
        WebSocket quote callback — update latest NBBO + compute OFI.

        OFI (Order Flow Imbalance) measures net buying/selling pressure
        from quote changes. Negative cumulative OFI = selling pressure
        building (bid deteriorating faster than ask improving).

        Lightweight: no logging, no lock held during field writes.
        """
        symbol = quote.symbol
        with self._watch_lock:
            watch = self._watches.get(symbol)
        if watch is None:
            return

        bid = float(quote.bid_price)
        ask = float(quote.ask_price)
        bid_size = int(quote.bid_size)
        ask_size = int(quote.ask_size)

        # Compute OFI tick from quote change
        if watch._prev_bid > 0:
            # Bid side: price up = buying pressure, down = selling
            if bid > watch._prev_bid:
                delta_bid = bid_size
            elif bid < watch._prev_bid:
                delta_bid = -bid_size
            else:
                delta_bid = bid_size - watch._prev_bid_size

            # Ask side: price down = buying pressure, up = selling
            if ask < watch._prev_ask:
                delta_ask = ask_size
            elif ask > watch._prev_ask:
                delta_ask = -ask_size
            else:
                delta_ask = ask_size - watch._prev_ask_size

            ofi_tick = delta_bid - delta_ask
            # Exponential moving average (decay=0.95 ≈ 20-tick window)
            watch.ofi_cumulative = watch.ofi_cumulative * 0.95 + ofi_tick

        # Update quote cache
        watch._prev_bid = watch.latest_bid
        watch._prev_ask = watch.latest_ask
        watch._prev_bid_size = watch.latest_bid_size
        watch._prev_ask_size = watch.latest_ask_size
        watch.latest_bid = bid
        watch.latest_ask = ask
        watch.latest_bid_size = bid_size
        watch.latest_ask_size = ask_size
        watch.latest_quote_ts = time_mod.time()

    async def _on_trade(self, trade) -> None:
        """
        WebSocket trade callback — check price against stop level,
        with optional trailing stop that ratchets up as price climbs.

        Trailing stop logic (when trail_r > 0):
        1. Track highest high since entry on every tick
        2. When gain reaches activate_at_r, trailing becomes active
        3. Ratchet stop up to (highest - risk * trail_r), never down
        4. Exit reason becomes 'trail_stop' when trailing is active

        Args:
            trade: Alpaca trade object with .symbol and .price
        """
        symbol = trade.symbol
        price = float(trade.price)

        with self._watch_lock:
            watch = self._watches.get(symbol)

        if watch is None:
            return

        # Trailing stop: update highest high and ratchet stop
        if watch.trail_r > 0 and watch.risk_per_share > 0:
            if price > watch.highest_since_entry:
                watch.highest_since_entry = price

            if not watch.trailing_active:
                r_gain = (watch.highest_since_entry - watch.entry_price) / watch.risk_per_share
                if r_gain >= watch.activate_at_r:
                    watch.trailing_active = True
                    logger.info(
                        f"StopMonitor: {symbol} trailing stop ACTIVATED — "
                        f"high=${watch.highest_since_entry:.2f}, "
                        f"+{r_gain:.1f}R from entry ${watch.entry_price:.2f}"
                    )

            if watch.trailing_active:
                new_stop = watch.highest_since_entry - watch.risk_per_share * watch.trail_r
                if new_stop > watch.stop_price:
                    old_stop = watch.stop_price
                    watch.stop_price = new_stop
                    logger.debug(
                        f"StopMonitor: {symbol} trail ratchet "
                        f"${old_stop:.2f} → ${new_stop:.2f} "
                        f"(high=${watch.highest_since_entry:.2f})"
                    )

        # Check stop level (works for both fixed and trailing stops)
        if price <= watch.stop_price:
            exit_reason = 'trail_stop' if (watch.trail_r > 0 and watch.trailing_active) else 'stop_loss'
            logger.info(
                f"StopMonitor: {symbol} price ${price:.2f} "
                f"<= stop ${watch.stop_price:.2f} — triggering {exit_reason}"
            )
            await self._execute_stop_exit(symbol, price, watch, exit_reason=exit_reason)

    async def _execute_stop_exit(
        self, symbol: str, trigger_price: float, watch: WatchEntry,
        exit_reason: str = "stop_loss",
    ) -> None:
        """
        Execute a stop exit: cancel bracket legs, submit marketable limit sell.

        Uses _exit_in_progress flag to prevent double-fire from rapid ticks.
        All synchronous API calls are dispatched via run_in_executor to avoid
        blocking the WebSocket event loop (which would delay processing of
        other symbols' price ticks).

        Args:
            symbol: Stock symbol
            trigger_price: Price that triggered the stop
            watch: WatchEntry with position details
            exit_reason: 'stop_loss', 'trail_stop', or 'stop_loss_fallback'
        """
        with self._exit_lock:
            if self._exit_in_progress.get(symbol, False):
                logger.debug(
                    f"StopMonitor: {symbol} exit already in progress, skipping"
                )
                return
            self._exit_in_progress[symbol] = True

        loop = asyncio.get_event_loop()
        exit_price = 0.0
        order_id = ""

        try:
            # Cancel the bracket legs (TP and safety-net SL)
            # Run in executor to avoid blocking the event loop
            for leg_id, leg_name in [
                (watch.tp_leg_id, "TP"),
                (watch.sl_leg_id, "SL"),
            ]:
                if leg_id:
                    try:
                        await loop.run_in_executor(
                            None, self._alpaca.cancel_order, leg_id
                        )
                        logger.info(
                            f"StopMonitor: {symbol} cancelled {leg_name} leg {leg_id}"
                        )
                    except Exception as e:
                        # 422 = already filled/cancelled — expected race condition
                        logger.warning(
                            f"StopMonitor: {symbol} {leg_name} cancel failed "
                            f"(may be filled): {e}"
                        )

            # Compute limit price: use cached quote from stream (sub-50ms),
            # fall back to REST if stale, then fixed offset as last resort
            bid, ask, bid_size, ask_size = 0.0, 0.0, 0, 0
            pricing_method = 'fixed_offset'
            quote_age_ms = (time_mod.time() - watch.latest_quote_ts) * 1000 if watch.latest_quote_ts > 0 else float('inf')

            if watch.latest_bid > 0 and watch.latest_ask > 0 and quote_age_ms < 5000:
                # Use cached quote from WebSocket stream
                bid, ask = watch.latest_bid, watch.latest_ask
                bid_size, ask_size = watch.latest_bid_size, watch.latest_ask_size
                limit_price, pricing_method = self.compute_limit_price_from_quote(bid, ask)
                if limit_price > 0:
                    logger.info(
                        f"StopMonitor: {symbol} cached-quote pricing — "
                        f"bid=${bid:.2f} ask=${ask:.2f} spread=${ask-bid:.3f} "
                        f"age={quote_age_ms:.0f}ms → limit=${limit_price:.2f} ({pricing_method})"
                    )
                else:
                    limit_price = self.compute_limit_price(trigger_price)
                    pricing_method = 'fixed_offset'
            else:
                # Fallback: REST call if quote stream data is stale/missing
                try:
                    quote = await loop.run_in_executor(
                        None, self._alpaca.get_latest_quote, symbol
                    )
                    bid = quote.get('bid_price', 0.0)
                    ask = quote.get('ask_price', 0.0)
                    bid_size = quote.get('bid_size', 0)
                    ask_size = quote.get('ask_size', 0)
                    limit_price, pricing_method = self.compute_limit_price_from_quote(bid, ask)
                    if limit_price <= 0:
                        raise ValueError(f"invalid quote: bid={bid}, ask={ask}")
                    logger.info(
                        f"StopMonitor: {symbol} REST-quote pricing — "
                        f"bid=${bid:.2f} ask=${ask:.2f} spread=${ask-bid:.3f} "
                        f"→ limit=${limit_price:.2f} ({pricing_method})"
                    )
                except Exception as e:
                    limit_price = self.compute_limit_price(trigger_price)
                    pricing_method = 'fixed_offset'
                    logger.warning(
                        f"StopMonitor: {symbol} quote failed ({e}) — "
                        f"fixed-offset limit=${limit_price:.2f}"
                    )

            # Submit marketable limit sell (in executor)
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: self._alpaca.submit_limit_sell_order(
                        symbol=symbol,
                        qty=watch.shares,
                        limit_price=limit_price,
                    ),
                )
                order_id = result.get("id", "")
                exit_price = limit_price
                logger.info(
                    f"StopMonitor: {symbol} limit sell submitted — "
                    f"qty={watch.shares}, limit=${limit_price:.2f} ({pricing_method}), "
                    f"order={order_id}"
                )
            except Exception as e:
                logger.error(
                    f"StopMonitor: {symbol} limit sell failed: {e} — "
                    f"falling back to close_position()"
                )
                # Fallback: market close (in executor)
                try:
                    fallback = await loop.run_in_executor(
                        None, self._alpaca.close_position, symbol
                    )
                    order_id = fallback.get("id", "")
                    exit_price = trigger_price
                    exit_reason = "stop_loss_fallback"
                    logger.info(
                        f"StopMonitor: {symbol} fallback close_position — "
                        f"order={order_id}"
                    )
                except Exception as e2:
                    logger.error(
                        f"StopMonitor: {symbol} fallback close also failed: {e2} — "
                        f"safety-net SL is the last line of defense"
                    )
                    # Remove watch to prevent infinite retry on every tick
                    # (likely TP already filled — no position to sell)
                    with self._watch_lock:
                        self._watches.pop(symbol, None)
                    with self._exit_lock:
                        self._exit_in_progress[symbol] = False
                    return

            # Emit exit event for main thread
            event = StopExitEvent(
                symbol=symbol,
                stop_price=watch.stop_price,
                exit_price=exit_price,
                shares=watch.shares,
                order_id=order_id,
                exit_reason=exit_reason,
                trade_db_id=watch.trade_db_id,
                submitted_at=time_mod.time(),
                pricing_method=pricing_method,
                exit_trigger_price=trigger_price,
                exit_quote_bid=bid,
                exit_quote_ask=ask,
                exit_quote_bid_size=bid_size,
                exit_quote_ask_size=ask_size,
                exit_limit_price=limit_price,
                exit_ofi=watch.ofi_cumulative,
            )
            self._exit_events.put(event)

            # Remove from watch list
            with self._watch_lock:
                self._watches.pop(symbol, None)

        except Exception as e:
            logger.error(f"StopMonitor: {symbol} exit execution error: {e}")
            with self._exit_lock:
                self._exit_in_progress[symbol] = False

    async def _subscribe_symbol(self, symbol: str) -> None:
        """Subscribe to trade and quote updates for a symbol."""
        if self._stream:
            try:
                self._stream.subscribe_trades(self._on_trade, symbol)
                self._stream.subscribe_quotes(self._on_quote, symbol)
                logger.debug(f"StopMonitor: subscribed to {symbol} (trades+quotes)")
            except Exception as e:
                logger.error(
                    f"StopMonitor: failed to subscribe {symbol}: {e}"
                )

    async def _unsubscribe_symbol(self, symbol: str) -> None:
        """Unsubscribe from trade and quote updates for a symbol."""
        if self._stream:
            try:
                self._stream.unsubscribe_trades(symbol)
                self._stream.unsubscribe_quotes(symbol)
                logger.debug(f"StopMonitor: unsubscribed from {symbol}")
            except Exception as e:
                logger.warning(
                    f"StopMonitor: failed to unsubscribe {symbol}: {e}"
                )

    async def _close_stream(self) -> None:
        """Close the WebSocket stream."""
        if self._stream:
            try:
                await self._stream.close()
            except Exception as e:
                logger.warning(f"StopMonitor: error closing stream: {e}")

    def _get_watched_symbols(self) -> list:
        """Get current watch list symbols (thread-safe)."""
        with self._watch_lock:
            return list(self._watches.keys())
