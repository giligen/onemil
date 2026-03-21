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
        # Step 1: Read watch state under lock
        with self._watch_lock:
            watch = self._watches.get(symbol)
            if not watch:
                logger.warning(
                    f"StopMonitor: execute_partial_exit({symbol}) — not watched"
                )
                return None
            if watch.exhaustion_partial_taken:
                logger.debug(
                    f"StopMonitor: {symbol} exhaustion partial already taken"
                )
                return None
            sell_shares = int(watch.shares * fraction)
            if sell_shares < 1:
                logger.warning(
                    f"StopMonitor: {symbol} partial shares < 1 "
                    f"({watch.shares} × {fraction})"
                )
                return None
            entry_price = watch.entry_price
            risk_per_share = watch.risk_per_share
            highest = watch.highest_since_entry
            trade_db_id = watch.trade_db_id

        # Step 2: API call outside lock — get latest price and submit sell
        exit_price = 0.0
        order_id = ""
        try:
            # Use latest snapshot price for limit
            bars = self._alpaca.get_1min_bars(symbol, lookback_minutes=2)
            if bars is not None and not bars.empty:
                latest_price = float(bars.iloc[-1]['close'])
            else:
                logger.warning(
                    f"StopMonitor: {symbol} no bars for partial exit price, "
                    f"using highest_since_entry"
                )
                latest_price = highest

            limit_price = self.compute_limit_price(latest_price)
            result = self._alpaca.submit_limit_sell_order(
                symbol=symbol,
                qty=sell_shares,
                limit_price=limit_price,
            )
            order_id = result.get("id", "")
            exit_price = limit_price
            logger.info(
                f"StopMonitor: {symbol} EXHAUSTION partial sell — "
                f"{sell_shares}sh @ limit ${limit_price:.2f}, "
                f"order={order_id}"
            )
        except Exception as e:
            logger.error(
                f"StopMonitor: {symbol} exhaustion partial sell FAILED: {e}"
            )
            return None

        # Step 3: Re-acquire lock and update watch state
        with self._watch_lock:
            watch = self._watches.get(symbol)
            if not watch:
                logger.warning(
                    f"StopMonitor: {symbol} removed during partial exit API call"
                )
                return None

            remaining = watch.shares - sell_shares
            watch.shares = remaining
            watch.exhaustion_partial_taken = True
            watch.trail_r = tighter_trail_r

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

        # Step 4: Emit exit event
        event = StopExitEvent(
            symbol=symbol,
            stop_price=0.0,  # Not a stop exit — exhaustion partial
            exit_price=exit_price,
            shares=sell_shares,
            order_id=order_id,
            exit_reason='exhaustion_partial',
            trade_db_id=trade_db_id,
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
                    logger.info(f"StopMonitor: WebSocket connecting with {len(watched)} symbols...")
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

            # Submit marketable limit sell (in executor)
            limit_price = self.compute_limit_price(trigger_price)
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
                    f"qty={watch.shares}, limit=${limit_price:.2f}, "
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
        """Subscribe to trade updates for a symbol."""
        if self._stream:
            try:
                self._stream.subscribe_trades(self._on_trade, symbol)
                logger.debug(f"StopMonitor: subscribed to {symbol}")
            except Exception as e:
                logger.error(
                    f"StopMonitor: failed to subscribe {symbol}: {e}"
                )

    async def _unsubscribe_symbol(self, symbol: str) -> None:
        """Unsubscribe from trade updates for a symbol."""
        if self._stream:
            try:
                self._stream.unsubscribe_trades(symbol)
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
