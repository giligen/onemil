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
from typing import Callable, Dict, List, Optional

from data_sources.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

RECONNECT_DELAY_SECONDS = 60.0  # 60s between retries — stop triggering Alpaca's rate limiter
MAX_RECONNECT_ATTEMPTS = 60     # 60 × 60s = 1 hour of retrying before giving up


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
    # Percentage-based trail (for MACD wave): trail = highest * (1 - trail_pct)
    # When > 0, overrides R-based trail computation
    trail_pct: float = 0.0
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
    # Strategy tag (routes exit events back to the originating engine under the
    # unified trader service). Defaults preserve legacy callers.
    strategy: str = 'bull_flag'
    # Experiment D — volume-confirmed trail exit (2026-04-17).
    # When `vol_confirmed_trail_enabled` is True AND trailing_active is True
    # AND the current price crosses the trail stop, we first check whether the
    # LAST CLOSED 1-min bar's volume >= flag_avg_volume × min_ratio. If not,
    # skip the exit (treat as low-vol drift). Falls back to naive exit when any
    # required field is missing. See trading/trail_vol_guard.py.
    avg_flag_volume: float = 0.0
    vol_confirmed_trail_enabled: bool = False
    vol_confirmed_trail_min_ratio: float = 1.0
    # Last fully-closed bar's volume (updated on bar-close events). Used only
    # by the vol-confirmed guard above.
    last_bar_volume: int = 0
    # Static-lock exit (ORB strategy, 2026-04-19). When lock_arm_at_r > 0, the
    # stop ratchets UP to entry + lock_stop_r × lock_r_unit the first time
    # price prints at or above entry + lock_arm_at_r × lock_r_unit. Unlike
    # trailing stops, the lock NEVER moves after arming — letting runners go
    # freely while protecting the locked-in minimum profit. Backtest: +34%
    # Calmar vs fixed targets on ORB (see study_orb_lock_variants.py).
    #
    # lock_r_unit defines what "1R" means for the lock. Defaults to
    # risk_per_share when 0.0. ORB passes range_size (range_high - range_low)
    # explicitly to match BT's R convention (BT: `range_size * LOCK_TRIGGER_R`).
    lock_arm_at_r: float = 0.0     # if > 0: trigger level (R-multiple above entry)
    lock_stop_r: float = 0.0       # stop-level after arming (R-multiple above entry)
    lock_armed: bool = False       # runtime state (internal; do not set at add_watch)
    lock_r_unit: float = 0.0       # explicit 1R for lock math; 0 = use risk_per_share
    # Entry-bar skip (ORB BT parity, 2026-04-19): in BT simulate_orb_trade,
    # `for row in sim_bars.iloc[1:]` skips the entry bar entirely for exit
    # checks. PROD matches by skipping all stop/lock checks for ticks before
    # this timestamp. Defaults to 0.0 (no skip) so bull flag / MACD wave are
    # unaffected. ORB sets to end-of-entry-bar (fill_time ceiled to next minute).
    skip_exits_until_ts: float = 0.0   # Unix timestamp; no checks before this


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
    strategy: str = 'bull_flag'  # copied from the originating WatchEntry


@dataclass
class QuoteWatch:
    """Passive quote monitoring for pending buy-stop orders.

    Tracks NBBO + OFI from WebSocket while order is pending.
    Used for entry microstructure analysis — no trading actions taken.
    """

    symbol: str
    submitted_at: float = 0.0  # time.time() when order placed
    # Snapshot at submission (filled by caller from REST quote)
    submit_bid: float = 0.0
    submit_ask: float = 0.0
    submit_bid_size: int = 0
    submit_ask_size: int = 0
    # Live cache (updated by _on_quote callback)
    latest_bid: float = 0.0
    latest_ask: float = 0.0
    latest_bid_size: int = 0
    latest_ask_size: int = 0
    latest_quote_ts: float = 0.0
    ofi_cumulative: float = 0.0
    _prev_bid: float = 0.0
    _prev_ask: float = 0.0
    _prev_bid_size: int = 0
    _prev_ask_size: int = 0


class StopMonitor:
    """
    Monitors real-time trade prices via WebSocket and triggers marketable
    limit sell exits when price breaches stop levels.

    Thread-safe: add_watch/remove_watch/drain_exit_events are called from
    the main thread. WebSocket callbacks run in a daemon thread.
    """

    @staticmethod
    def _is_race_condition_error(e: Exception) -> bool:
        """Detect errors from bracket SL winning the race to exit.

        When the broker-side bracket SL fills first, our subsequent sell
        attempt lands on an already-flat position. Alpaca returns:
          - 42210000 / "cannot be sold short" — position already flat
          - 40410000 / "position not found" — close_position on empty
        These are expected noise after a successful bracket exit,
        NOT actionable failures — don't page on them.
        """
        msg = str(e).lower()
        return (
            '42210000' in msg or 'cannot be sold short' in msg or
            '40410000' in msg or 'position not found' in msg or
            'no position' in msg
        )

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        alpaca_client: AlpacaClient,
        marketable_limit_offset: float = 0.03,
        marketable_limit_offset_pct: float = 0.005,
        notifier=None,
        polling_mode: bool = False,
        polling_interval: float = 2.0,
        alpaca_clients_by_strategy: Optional[Dict[str, AlpacaClient]] = None,
    ):
        """
        Initialize StopMonitor.

        Args:
            api_key: Alpaca API key (for StockDataStream market-data feed)
            api_secret: Alpaca API secret
            alpaca_client: Default AlpacaClient for order submission (fallback
                when a watch's strategy has no entry in alpaca_clients_by_strategy).
            marketable_limit_offset: Minimum dollar offset below price for
                marketable limit sell (default $0.03)
            marketable_limit_offset_pct: Percentage offset below price
                (default 0.5%). Whichever is larger is used.
            notifier: Optional TelegramNotifier for alerts
            polling_mode: Use REST API polling instead of WebSocket.
                For paper nodes that share Alpaca account with live (1 WS slot).
            polling_interval: Seconds between REST polls (default 2.0)
            alpaca_clients_by_strategy: Optional dict of {strategy: AlpacaClient}.
                When set, exit orders for a watch with matching `strategy` tag
                are submitted via the mapped client. Strategies not in the dict
                fall through to `alpaca_client` (the default). This enables
                multi-account operation (e.g., ORB paper + bull flag live)
                without requiring a separate StopMonitor per account. The
                market-data WebSocket is account-agnostic so one feed is fine.
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._alpaca = alpaca_client
        # Per-strategy order-execution routing. Empty dict = legacy single-client behavior.
        self._clients_by_strategy: Dict[str, AlpacaClient] = dict(alpaca_clients_by_strategy or {})
        self._marketable_limit_offset = marketable_limit_offset
        self._marketable_limit_offset_pct = marketable_limit_offset_pct
        self._notifier = notifier

        self._watches: Dict[str, WatchEntry] = {}
        self._quote_watches: Dict[str, QuoteWatch] = {}  # passive entry monitoring
        self._watch_lock = threading.Lock()  # protects both _watches and _quote_watches

        self._exit_events: queue.Queue = queue.Queue()
        self._exit_in_progress: Dict[str, bool] = {}
        self._exit_lock = threading.Lock()

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stream = None
        self._running = False
        self._stop_event = threading.Event()
        self._last_data_ts: float = 0.0  # time.time() of last data received
        self._ws_connected: bool = False  # True only when WebSocket is actively connected
        self._polling_mode = polling_mode
        self._polling_interval = polling_interval

        # Bar streaming for pattern detection: handlers fire on each 1-min bar close.
        # Multi-consumer: multiple strategies may register; each gets (symbol, bars_df).
        # Keyed by handler_id so consumers can update/unregister cleanly.
        self._bar_handlers: Dict[str, Callable] = {}
        self._bar_handler_lock = threading.Lock()
        self._bar_symbols: set = set()  # symbols subscribed to bar stream
        self._bar_windows: Dict[str, list] = {}  # rolling bar window per symbol

    @property
    def polling_mode(self) -> bool:
        """True if this monitor is in REST polling mode (no data WebSocket)."""
        return self._polling_mode

    def _client_for(self, strategy: str) -> AlpacaClient:
        """Return the AlpacaClient that should execute orders for this strategy.

        If `alpaca_clients_by_strategy` was provided at init AND contains an
        entry for the given strategy, that client is returned. Otherwise the
        default (self._alpaca) is used — preserves legacy single-client
        behavior for strategies not in the routing dict.
        """
        return self._clients_by_strategy.get(strategy, self._alpaca)

    def start(self) -> None:
        """Launch the stop monitoring daemon thread (WebSocket or REST polling)."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("StopMonitor already running")
            return

        self._stop_event.clear()
        self._running = True

        if self._polling_mode:
            self._thread = threading.Thread(
                target=self._run_polling_loop,
                name="StopMonitor-REST",
                daemon=True,
            )
        else:
            self._thread = threading.Thread(
                target=self._run_stream_loop,
                name="StopMonitor-WS",
                daemon=True,
            )
        self._thread.start()
        mode = "REST polling" if self._polling_mode else "WebSocket"
        logger.info(f"StopMonitor started ({mode}, daemon thread)")

    def stop(self) -> None:
        """Gracefully shut down the WebSocket thread."""
        self._running = False
        self._ws_connected = False
        self._stop_event.set()

        # Close WebSocket and WAIT for completion — prevents zombie connections
        if self._loop and self._stream:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._close_stream(), self._loop
                )
                future.result(timeout=5)  # block until close completes or 5s timeout
                logger.info("StopMonitor: WebSocket stream closed")
            except Exception as e:
                logger.warning(f"StopMonitor: error during stream close: {e}")
                # Force-close the underlying websocket if graceful close failed
                try:
                    if self._stream and hasattr(self._stream, '_ws') and self._stream._ws:
                        asyncio.run_coroutine_threadsafe(
                            self._stream._ws.close(), self._loop
                        ).result(timeout=3)
                except Exception:
                    pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)

        self._stream = None
        logger.info("StopMonitor stopped")

    def register_bar_handler(self, handler_id: str, callback: Callable) -> None:
        """
        Register a bar-close handler: callback(symbol, bars_df).

        Multiple strategies may register under different handler_id keys.
        Registering with an existing id OVERWRITES the previous handler for that id
        (supports clean re-registration after config reloads).

        Args:
            handler_id: stable identifier, e.g. 'bull_flag', 'macd_wave'
            callback: invoked for every bar event on every subscribed symbol
        """
        if self._polling_mode:
            # Handler is still stored so that a mode-switch later would wire it
            # up — but in polling mode the WS isn't active, no bar events fire.
            # Surface this loudly so misconfigured deployments get noticed.
            logger.warning(
                f"StopMonitor: register_bar_handler('{handler_id}') called but "
                f"monitor is in REST polling mode — bar events will NOT be delivered."
            )
        with self._bar_handler_lock:
            existed = handler_id in self._bar_handlers
            self._bar_handlers[handler_id] = callback
        logger.info(
            f"StopMonitor: bar handler {'updated' if existed else 'registered'} "
            f"(id={handler_id}, total={len(self._bar_handlers)})"
        )

    def unregister_bar_handler(self, handler_id: str) -> None:
        """Remove a bar-close handler by id. No-op if not registered."""
        with self._bar_handler_lock:
            existed = self._bar_handlers.pop(handler_id, None) is not None
        if existed:
            logger.info(f"StopMonitor: bar handler unregistered (id={handler_id})")

    def set_bar_callback(self, callback: Callable) -> None:
        """
        Deprecated — use register_bar_handler(id, callback) instead.

        Kept as a back-compat shim that registers under id='default'. Schedule
        for removal once all call sites have been migrated (see project audit
        notes). Any future caller hitting this path will see a DeprecationWarning
        via the stdlib warnings module and a logged warning, making the
        remaining uses visible in production logs.
        """
        import warnings
        warnings.warn(
            "StopMonitor.set_bar_callback is deprecated; "
            "use register_bar_handler(handler_id, callback) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning(
            "StopMonitor.set_bar_callback is deprecated — "
            "registering under handler_id='default'. "
            "Migrate callers to register_bar_handler(id, cb)."
        )
        self.register_bar_handler('default', callback)

    def subscribe_bars(self, symbol: str) -> None:
        """Subscribe to 1-min bar stream for a symbol (for pattern detection).

        Thread-safe: can be called from main thread.
        """
        if symbol in self._bar_symbols:
            return
        self._bar_symbols.add(symbol)
        self._bar_windows.setdefault(symbol, [])

        if self._loop and self._stream and self._ws_connected:
            asyncio.run_coroutine_threadsafe(
                self._subscribe_bars_async(symbol), self._loop
            )

    async def _subscribe_bars_async(self, symbol: str) -> None:
        """Subscribe to bar stream on the WebSocket (async, runs in WS thread)."""
        if self._stream:
            try:
                self._stream._handlers["bars"][symbol] = self._on_bar
                if self._stream._ws:
                    await self._stream._send_subscribe_msg()
                    logger.info(f"StopMonitor: subscribed to {symbol} bars")
            except Exception as e:
                logger.error(f"StopMonitor: failed to subscribe {symbol} bars: {e}")

    async def _on_bar(self, bar) -> None:
        """Handle 1-min bar close from WebSocket. Fans out to all registered handlers."""
        try:
            symbol = bar.symbol
            if symbol not in self._bar_symbols:
                return

            # Append to rolling window
            bar_dict = {
                'timestamp': bar.timestamp,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume),
            }
            self._bar_windows.setdefault(symbol, []).append(bar_dict)
            self._last_data_ts = time_mod.time()

            # Experiment D: stash last-closed-bar volume on the active watch so
            # the tick-path vol-confirmed trail check can read it. Poll path
            # uses the bar dict directly; this covers the tick-path.
            with self._watch_lock:
                watch = self._watches.get(symbol)
                if watch is not None:
                    watch.last_bar_volume = bar_dict['volume']

            # Snapshot handlers under lock, then fire outside lock so a slow handler
            # doesn't block registration/unregistration on another thread.
            with self._bar_handler_lock:
                handlers = list(self._bar_handlers.items())
            if not handlers:
                return
            import pandas as pd
            bars_df = pd.DataFrame(self._bar_windows[symbol])
            # Defensive: each handler gets its own copy so in-place mutation
            # by one consumer cannot corrupt state for the others.
            for handler_id, cb in handlers:
                try:
                    cb(symbol, bars_df.copy())
                except Exception as e:
                    logger.error(
                        f"StopMonitor: bar handler '{handler_id}' raised for {symbol}: {e}"
                    )

        except Exception as e:
            logger.error(f"StopMonitor: _on_bar error: {e}")

    def is_healthy(self, max_stale_seconds: float = 30.0) -> bool:
        """
        Check if StopMonitor WebSocket is operational.

        Returns True only if:
        1. _running is True
        2. Daemon thread is alive
        3. Stream object exists
        4. Data received within max_stale_seconds (only when watches exist)

        Called from service main loops each cycle as circuit breaker.
        """
        if not self._running:
            return False
        if self._thread is None or not self._thread.is_alive():
            return False
        if not self._polling_mode and self._stream is None:
            return False
        # Only check WebSocket connection + data freshness when watches exist
        # No watches = WebSocket waiting for first symbol = normal startup
        with self._watch_lock:
            has_watches = len(self._watches) > 0
        if has_watches:
            if not self._ws_connected:
                return False
            if self._last_data_ts > 0:
                age = time_mod.time() - self._last_data_ts
                if age > max_stale_seconds:
                    return False
        return True

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
        trail_pct: float = 0.0,
        strategy: str = 'bull_flag',
        avg_flag_volume: float = 0.0,
        vol_confirmed_trail_enabled: bool = False,
        vol_confirmed_trail_min_ratio: float = 1.0,
        lock_arm_at_r: float = 0.0,
        lock_stop_r: float = 0.0,
        lock_r_unit: float = 0.0,
        skip_exits_until_ts: float = 0.0,
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
            avg_flag_volume: Avg 1-min bar volume during the setup's flag
                consolidation — baseline for vol-confirmed trail exit (0 disables).
            vol_confirmed_trail_enabled: When True, trail exits require the
                latest closed bar's volume >= avg_flag_volume × min_ratio.
            vol_confirmed_trail_min_ratio: Ratio threshold (default 1.0).
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
            trail_pct=trail_pct,
            trailing_active=trail_pct > 0,  # %-based trail activates immediately
            strategy=strategy,
            avg_flag_volume=avg_flag_volume,
            vol_confirmed_trail_enabled=vol_confirmed_trail_enabled,
            vol_confirmed_trail_min_ratio=vol_confirmed_trail_min_ratio,
            lock_arm_at_r=lock_arm_at_r,
            lock_stop_r=lock_stop_r,
            lock_r_unit=lock_r_unit,
            skip_exits_until_ts=skip_exits_until_ts,
        )
        with self._watch_lock:
            self._watches[symbol] = entry
            already_subscribed = symbol in self._quote_watches
        # Clear any stale exit-in-progress flag from previous trade of same symbol
        with self._exit_lock:
            self._exit_in_progress[symbol] = False

        # Subscribe to trades for this symbol on the WebSocket
        # Skip if quote_watch already has this symbol subscribed
        if not already_subscribed and self._loop and self._stream and self._running:
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
        Sets _exit_in_progress=True BEFORE removing watch so any in-flight
        _on_trade callback is blocked from double-selling.

        Args:
            symbol: Stock symbol to stop watching
        """
        # Block any in-flight _on_trade from starting a new exit
        with self._exit_lock:
            self._exit_in_progress[symbol] = True

        with self._watch_lock:
            removed = self._watches.pop(symbol, None)
            has_quote_watch = symbol in self._quote_watches

        if removed:
            # Only unsubscribe if no quote_watch also needs this symbol
            if not has_quote_watch and self._loop and self._stream and self._running:
                asyncio.run_coroutine_threadsafe(
                    self._unsubscribe_symbol(symbol), self._loop
                )
            logger.info(f"StopMonitor: removed watch for {symbol}")

    # ------------------------------------------------------------------
    # Passive quote monitoring for pending buy-stop orders
    # ------------------------------------------------------------------

    def add_quote_watch(
        self,
        symbol: str,
        submit_bid: float = 0.0,
        submit_ask: float = 0.0,
        submit_bid_size: int = 0,
        submit_ask_size: int = 0,
    ) -> None:
        """
        Start passively monitoring quotes for a pending buy-stop order.

        Called from main thread when a buy-stop is submitted. Subscribes to
        quotes via the SIP WebSocket to track NBBO + OFI while the order is
        pending. No trading actions taken — data collection only.

        Args:
            symbol: Stock symbol
            submit_bid: NBBO bid at submission time
            submit_ask: NBBO ask at submission time
            submit_bid_size: Bid depth at submission time
            submit_ask_size: Ask depth at submission time
        """
        entry = QuoteWatch(
            symbol=symbol,
            submitted_at=time_mod.time(),
            submit_bid=submit_bid,
            submit_ask=submit_ask,
            submit_bid_size=submit_bid_size,
            submit_ask_size=submit_ask_size,
            latest_bid=submit_bid,
            latest_ask=submit_ask,
            latest_bid_size=submit_bid_size,
            latest_ask_size=submit_ask_size,
        )
        with self._watch_lock:
            self._quote_watches[symbol] = entry
            already_subscribed = symbol in self._watches

        # Subscribe only if StopMonitor isn't already watching this symbol
        if not already_subscribed and self._loop and self._stream and self._running:
            asyncio.run_coroutine_threadsafe(
                self._subscribe_symbol(symbol), self._loop
            )

        logger.info(
            f"StopMonitor: quote-watch {symbol} — "
            f"bid=${submit_bid:.2f} ask=${submit_ask:.2f} "
            f"depth={submit_bid_size}×{submit_ask_size}"
        )

    def remove_quote_watch(self, symbol: str) -> None:
        """
        Stop passively monitoring quotes for a symbol.

        Called from main thread when buy-stop fills, is cancelled, or expires.
        Only unsubscribes from WebSocket if StopMonitor isn't also watching.
        """
        with self._watch_lock:
            removed = self._quote_watches.pop(symbol, None)
            has_stop_watch = symbol in self._watches

        if removed and not has_stop_watch:
            if self._loop and self._stream and self._running:
                asyncio.run_coroutine_threadsafe(
                    self._unsubscribe_symbol(symbol), self._loop
                )
        if removed:
            logger.debug(f"StopMonitor: removed quote-watch for {symbol}")

    def get_quote_watch_snapshot(self, symbol: str) -> Optional[Dict]:
        """
        Return a thread-safe snapshot of a QuoteWatch's fields.

        Called from main thread at fill time to capture entry microstructure.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with quote watch fields, or None if not watched
        """
        with self._watch_lock:
            qw = self._quote_watches.get(symbol)
            if not qw:
                return None
            return {
                'submit_bid': qw.submit_bid,
                'submit_ask': qw.submit_ask,
                'submit_bid_size': qw.submit_bid_size,
                'submit_ask_size': qw.submit_ask_size,
                'latest_bid': qw.latest_bid,
                'latest_ask': qw.latest_ask,
                'latest_bid_size': qw.latest_bid_size,
                'latest_ask_size': qw.latest_ask_size,
                'ofi_cumulative': qw.ofi_cumulative,
                'submitted_at': qw.submitted_at,
            }

    def drain_exit_events(self, strategy: Optional[str] = None) -> List[StopExitEvent]:
        """
        Drain pending exit events from the queue.

        Called from main thread each monitoring cycle. In the unified trader
        process, BOTH strategies drain the same queue, so each must pass its
        own `strategy` tag. Events for OTHER strategies are preserved (requeued)
        so the owning engine still sees them on its next drain.

        Args:
            strategy: optional strategy name ('bull_flag' | 'macd_wave'). If None,
                      drain everything regardless of tag (legacy behavior, tests).

        Returns:
            List of StopExitEvent objects matching `strategy` (or all, if None).
        """
        mine: List[StopExitEvent] = []
        others: List[StopExitEvent] = []
        while True:
            try:
                event = self._exit_events.get_nowait()
            except queue.Empty:
                break
            ev_strategy = getattr(event, 'strategy', 'bull_flag') or 'bull_flag'
            if strategy is None or ev_strategy == strategy:
                mine.append(event)
            else:
                others.append(event)
        # Re-queue events that don't belong to this caller's strategy so the
        # owning engine picks them up on its own drain. Ordering across
        # strategies is not meaningful (each engine processes its own chain).
        for ev in others:
            self._exit_events.put(ev)
        return mine

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
            watch_strategy = watch.strategy

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
            strategy=watch_strategy,
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
    def compute_limit_price_from_quote(
        bid: float, ask: float,
        ofi: float = 0.0,
        shares: int = 0,
        bid_size: int = 0,
    ) -> tuple:
        """
        Compute limit sell price from current NBBO quote using spread tiers,
        with OFI and size-awareness for urgent exits.

        Spread tiers (base pricing):
        - Tight spread (<$0.05): sell at midpoint — liquid, saves vs fixed offset
        - Medium spread ($0.05-$0.15): sell at bid + $0.01 — fast fill, minimal give
        - Wide spread (>$0.15): sell at bid — take what's available on illiquid

        OFI override (selling pressure detection):
        - OFI < -1000: use bid (skip midpoint — sellers dominating, bid dropping)
        - OFI < -3000: use bid - $0.01 (urgent exit, accept walking the book)

        Size override (thin book detection):
        - shares > 5× bid_size: use bid (we'll walk the book regardless)

        Args:
            bid: Current best bid price
            ask: Current best ask price
            ofi: Order Flow Imbalance (negative = selling pressure)
            shares: Number of shares to sell
            bid_size: Depth at top of bid book

        Returns:
            Tuple of (limit_price, pricing_method) where pricing_method is
            'quote_tight', 'quote_medium', 'quote_wide', 'ofi_urgent',
            'ofi_aggressive', or 'size_aggressive'.
            Returns (0.0, 'invalid') if quote data is invalid.
        """
        if bid <= 0 or ask <= 0 or ask < bid:
            return (0.0, 'invalid')

        spread = ask - bid

        # OFI override: heavy selling pressure → hit bid or lower
        if ofi < -3000:
            limit = round(bid - 0.01, 2)
            method = 'ofi_urgent'
        elif ofi < -1000:
            limit = round(bid, 2)
            method = 'ofi_aggressive'
        # Size override: we'd walk the book anyway
        elif bid_size > 0 and shares > 5 * bid_size:
            limit = round(bid, 2)
            method = 'size_aggressive'
        # Normal spread tiers
        elif spread < 0.05:
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
    # Internal — REST polling mode (paper nodes)
    # =========================================================================

    def _run_polling_loop(self) -> None:
        """REST polling via single get_snapshots() call per cycle. Paper nodes only."""
        import time as time_mod
        logger.info(
            f"StopMonitor: REST polling mode — interval={self._polling_interval}s "
            f"(1 snapshot call per cycle, no WebSocket)"
        )
        self._ws_connected = True  # pretend connected for is_healthy()
        self._last_data_ts = time_mod.time()

        while self._running:
            try:
                with self._watch_lock:
                    watch_symbols = list(self._watches.keys())
                    quote_watch_symbols = list(self._quote_watches.keys())
                all_symbols = list(set(watch_symbols + quote_watch_symbols))

                if not all_symbols:
                    time_mod.sleep(self._polling_interval)
                    self._last_data_ts = time_mod.time()
                    continue

                # Single API call: snapshot returns trade price + bid/ask/sizes
                try:
                    snapshots = self._alpaca.get_snapshots(all_symbols)
                except Exception as e:
                    logger.debug(f"StopMonitor poll: snapshot failed: {e}")
                    time_mod.sleep(self._polling_interval)
                    continue

                self._last_data_ts = time_mod.time()

                # Update quote watches (passive entry monitoring)
                for sym in quote_watch_symbols:
                    snap = snapshots.get(sym)
                    if not snap:
                        continue
                    with self._watch_lock:
                        qw = self._quote_watches.get(sym)
                    if qw:
                        qw.latest_bid = snap.get('bid_price', 0)
                        qw.latest_ask = snap.get('ask_price', 0)
                        qw.latest_bid_size = snap.get('bid_size', 0)
                        qw.latest_ask_size = snap.get('ask_size', 0)

                # Process stop/trail watches
                for sym in watch_symbols:
                    snap = snapshots.get(sym)
                    if not snap:
                        continue
                    with self._watch_lock:
                        watch = self._watches.get(sym)
                    if not watch:
                        continue

                    price = snap.get('latest_price', 0)
                    if price <= 0:
                        continue

                    watch.latest_bid = snap.get('bid_price', 0)
                    watch.latest_ask = snap.get('ask_price', 0)

                    if price > watch.highest_since_entry:
                        watch.highest_since_entry = price

                    # Trail activation
                    if not watch.trailing_active and watch.entry_price > 0:
                        risk = watch.risk_per_share if watch.risk_per_share > 0 else watch.entry_price * 0.02
                        if risk > 0:
                            r_gain = (watch.highest_since_entry - watch.entry_price) / risk
                            if watch.trail_pct > 0:
                                watch.trailing_active = True
                            elif r_gain >= watch.activate_at_r:
                                watch.trailing_active = True
                                logger.info(f"StopMonitor poll: {sym} trail ACTIVATED +{r_gain:.1f}R")

                    # Ratchet trail
                    if watch.trailing_active:
                        if watch.trail_pct > 0:
                            new_stop = watch.highest_since_entry * (1 - watch.trail_pct)
                        else:
                            new_stop = watch.highest_since_entry - watch.risk_per_share * watch.trail_r
                        if new_stop > watch.stop_price:
                            old = watch.stop_price
                            watch.stop_price = new_stop
                            logger.debug(f"StopMonitor poll: {sym} trail ${old:.2f}→${new_stop:.2f}")

                    # Check stop
                    if price <= watch.stop_price:
                        reason = 'trail_stop' if watch.trailing_active else 'stop_loss'
                        # Experiment D: vol-confirmed trail gate. Only applies
                        # to trail exits with the flag enabled on this watch.
                        if reason == 'trail_stop' and watch.vol_confirmed_trail_enabled:
                            from trading.trail_vol_guard import should_skip_trail_exit_on_low_vol
                            if should_skip_trail_exit_on_low_vol(
                                bar_volume=watch.last_bar_volume,
                                flag_avg_volume=watch.avg_flag_volume,
                                min_vol_ratio=watch.vol_confirmed_trail_min_ratio,
                            ):
                                logger.info(
                                    f"StopMonitor poll: {sym} TRAIL VOL-CONF SKIP "
                                    f"— bar_vol={watch.last_bar_volume:,} < "
                                    f"{watch.vol_confirmed_trail_min_ratio}×"
                                    f"flag_avg={watch.avg_flag_volume:,.0f}"
                                )
                                continue
                        logger.info(f"StopMonitor poll: {sym} {reason} @ ${price:.2f} (stop=${watch.stop_price:.2f})")
                        self._exit_events.put(StopExitEvent(
                            symbol=sym, stop_price=watch.stop_price,
                            exit_price=price, shares=watch.shares,
                            order_id=getattr(watch, 'order_id', ''),
                            exit_reason=reason, trade_db_id=watch.trade_db_id,
                            submitted_at=time_mod.time(), exit_trigger_price=price,
                            exit_quote_bid=watch.latest_bid, exit_quote_ask=watch.latest_ask,
                            exit_quote_bid_size=snap.get('bid_size', 0),
                            exit_quote_ask_size=snap.get('ask_size', 0),
                            pricing_method='poll_snapshot',
                            exit_limit_price=watch.latest_bid,
                            strategy=watch.strategy,
                        ))

                time_mod.sleep(self._polling_interval)
            except Exception as e:
                logger.error(f"StopMonitor poll error: {e}")
                time_mod.sleep(self._polling_interval)

        logger.info("StopMonitor polling loop stopped")

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
                # Close previous stream to release TCP connection
                if self._stream is not None:
                    try:
                        await self._close_stream()
                    except Exception:
                        pass
                    self._stream = None

                from alpaca.data.enums import DataFeed
                self._stream = StockDataStream(
                    self._api_key, self._api_secret,
                    feed=DataFeed.SIP,
                )
                # Connect ONCE at startup with SPY as keepalive.
                # Dynamic symbol subscriptions happen via _subscribe_symbol().
                # No waiting for watches — the connection stays alive permanently.
                watched = self._get_watched_symbols()
                # Always include SPY as keepalive — prevents SDK busy-spin on empty handlers
                all_symbols = list(set(watched + ['SPY']))

                self._stream.subscribe_trades(
                    self._on_trade, *all_symbols
                )
                self._stream.subscribe_quotes(
                    self._on_quote, *all_symbols
                )
                # Subscribe to bars for pattern detection symbols
                if self._bar_symbols:
                    bar_syms = list(self._bar_symbols)
                    self._stream.subscribe_bars(
                        self._on_bar, *bar_syms
                    )
                    logger.info(f"StopMonitor: subscribing to bars for {len(bar_syms)} pattern detection symbols")
                logger.info(f"StopMonitor: WebSocket connecting with {len(all_symbols)} symbols ({len(watched)} watched + SPY keepalive)...")
                # Single connection attempt — don't use _run_forever (has uncontrollable internal retry)
                await self._stream._start_ws()
                self._ws_connected = True
                self._last_data_ts = time_mod.time()
                logger.info("StopMonitor: WebSocket CONNECTED")
                await self._stream._consume()

            except Exception as e:
                self._ws_connected = False
                if not self._running:
                    break

                # Force-close any lingering TCP socket from failed connection
                if self._stream:
                    try:
                        if hasattr(self._stream, '_ws') and self._stream._ws:
                            await self._stream._ws.close()
                        await self._close_stream()
                    except Exception:
                        pass
                    self._stream = None

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
            self._running = False
            self._stream = None
            error_msg = (
                f"StopMonitor: exhausted {MAX_RECONNECT_ATTEMPTS} reconnect "
                f"attempts — DEAD, emergency exit required"
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
            qwatch = self._quote_watches.get(symbol)
        if watch is None and qwatch is None:
            return

        bid = float(quote.bid_price)
        ask = float(quote.ask_price)
        bid_size = int(quote.bid_size)
        ask_size = int(quote.ask_size)
        now = time_mod.time()
        self._last_data_ts = now  # heartbeat — quotes flow even when no trades

        # Update all matching watches (stop watch and/or quote watch)
        for w in [watch, qwatch]:
            if w is None:
                continue

            # Compute OFI tick from quote change
            if w._prev_bid > 0:
                if bid > w._prev_bid:
                    delta_bid = bid_size
                elif bid < w._prev_bid:
                    delta_bid = -bid_size
                else:
                    delta_bid = bid_size - w._prev_bid_size

                if ask < w._prev_ask:
                    delta_ask = ask_size
                elif ask > w._prev_ask:
                    delta_ask = -ask_size
                else:
                    delta_ask = ask_size - w._prev_ask_size

                ofi_tick = delta_bid - delta_ask
                w.ofi_cumulative = w.ofi_cumulative * 0.95 + ofi_tick

            # Update quote cache
            w._prev_bid = w.latest_bid
            w._prev_ask = w.latest_ask
            w._prev_bid_size = w.latest_bid_size
            w._prev_ask_size = w.latest_ask_size
            w.latest_bid = bid
            w.latest_ask = ask
            w.latest_bid_size = bid_size
            w.latest_ask_size = ask_size
            w.latest_quote_ts = now

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
        self._last_data_ts = time_mod.time()

        with self._watch_lock:
            watch = self._watches.get(symbol)

        if watch is None:
            return

        # Entry-bar skip (ORB BT parity): tick is within the entry bar — don't
        # evaluate arm/stop yet, but DO keep highest_since_entry updated so
        # when the skip window closes, arming can fire on the correct peak.
        if watch.skip_exits_until_ts > 0 and self._last_data_ts < watch.skip_exits_until_ts:
            if price > watch.highest_since_entry:
                watch.highest_since_entry = price
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
                # Percentage-based trail (MACD wave) or R-based trail (bull flag)
                if watch.trail_pct > 0:
                    new_stop = watch.highest_since_entry * (1 - watch.trail_pct)
                else:
                    new_stop = watch.highest_since_entry - watch.risk_per_share * watch.trail_r
                if new_stop > watch.stop_price:
                    old_stop = watch.stop_price
                    watch.stop_price = new_stop
                    logger.debug(
                        f"StopMonitor: {symbol} trail ratchet "
                        f"${old_stop:.2f} → ${new_stop:.2f} "
                        f"(high=${watch.highest_since_entry:.2f})"
                    )

        # Static-lock arming (ORB strategy). When price first reaches
        # entry + lock_arm_at_r × R, move stop UP to entry + lock_stop_r × R
        # and NEVER lower it (one-shot, not trailing).
        # R unit precedence: watch.lock_r_unit (explicit) > risk_per_share (fallback).
        if watch.lock_arm_at_r > 0 and not watch.lock_armed:
            r_unit = watch.lock_r_unit if watch.lock_r_unit > 0 else watch.risk_per_share
            if r_unit > 0:
                # Track running high so arming uses the peak (not current tick)
                if watch.trail_r == 0 and price > watch.highest_since_entry:
                    watch.highest_since_entry = price
                arm_level = watch.entry_price + watch.lock_arm_at_r * r_unit
                if watch.highest_since_entry >= arm_level:
                    new_stop = watch.entry_price + watch.lock_stop_r * r_unit
                    if new_stop > watch.stop_price:
                        old_stop = watch.stop_price
                        watch.stop_price = new_stop
                        watch.lock_armed = True
                        logger.info(
                            f"StopMonitor: {symbol} LOCK ARMED at "
                            f"+{watch.lock_arm_at_r:.1f}R (R=${r_unit:.2f}) — stop moved "
                            f"${old_stop:.2f} → ${new_stop:.2f} "
                            f"(high=${watch.highest_since_entry:.2f}, "
                            f"entry=${watch.entry_price:.2f})"
                        )

        # Check stop level (works for fixed, trailing, and static-lock stops)
        if price <= watch.stop_price:
            if watch.trail_r > 0 and watch.trailing_active:
                exit_reason = 'trail_stop'
            elif watch.lock_armed:
                exit_reason = 'lock_stop'
            else:
                exit_reason = 'stop_loss'
            # Experiment D: for trail exits (not initial hard stop), require
            # the last closed bar's volume to confirm active selling. Skip if
            # the bar looks like low-volume drift. Only applies when
            # trailing_active is True AND vol_confirmed_trail_enabled.
            if exit_reason == 'trail_stop' and watch.vol_confirmed_trail_enabled:
                from trading.trail_vol_guard import should_skip_trail_exit_on_low_vol
                if should_skip_trail_exit_on_low_vol(
                    bar_volume=watch.last_bar_volume,
                    flag_avg_volume=watch.avg_flag_volume,
                    min_vol_ratio=watch.vol_confirmed_trail_min_ratio,
                ):
                    logger.info(
                        f"StopMonitor: {symbol} TRAIL VOL-CONF SKIP — "
                        f"bar_vol={watch.last_bar_volume:,} < "
                        f"{watch.vol_confirmed_trail_min_ratio}×flag_avg={watch.avg_flag_volume:,.0f}, "
                        f"price=${price:.2f} stop=${watch.stop_price:.2f} (holding)"
                    )
                    return
            logger.info(
                f"StopMonitor: {symbol} price ${price:.2f} "
                f"<= stop ${watch.stop_price:.2f} — triggering {exit_reason}"
            )
            await self._execute_stop_exit(symbol, price, watch, exit_reason=exit_reason)

    # Shared constants for fill-confirmation polling. Short and aggressive —
    # on a stop we prefer getting flat fast over holding out for price.
    _STOP_EXIT_FILL_TIMEOUT_S = 10
    _STOP_EXIT_POLL_INTERVAL_S = 1.0

    async def _poll_order_fill(
        self, client, order_id: str, fallback_price: float,
        timeout_s: Optional[float] = None,
    ) -> Optional[float]:
        """Poll an Alpaca order until it fills, is cancelled, or timeout.

        Returns the `filled_avg_price` on a confirmed fill, `None` if the poll
        timed out or the order ended in a non-fill terminal state
        (cancelled/rejected/expired). Caller decides how to recover (escalate
        to market close, give up, etc.).

        Only `status == 'filled'` counts as success — a brief `partially_filled`
        window won't short-circuit us; we wait for either full fill, a terminal
        state, or timeout. `fallback_price` is returned in the exotic case
        where Alpaca reports filled but with no `filled_avg_price`.
        """
        loop = asyncio.get_event_loop()
        deadline = time_mod.time() + (
            timeout_s if timeout_s is not None else self._STOP_EXIT_FILL_TIMEOUT_S
        )
        while time_mod.time() < deadline:
            await asyncio.sleep(self._STOP_EXIT_POLL_INTERVAL_S)
            try:
                status_info = await loop.run_in_executor(
                    None, client.get_order, order_id
                )
            except Exception as e:
                logger.debug(
                    f"StopMonitor: get_order({order_id[:8]}...) failed: {e}"
                )
                continue
            st = str(status_info.get('status', '')).lower()
            if st == 'filled':
                return float(status_info.get('filled_avg_price') or fallback_price)
            if st in ('canceled', 'cancelled', 'rejected', 'expired'):
                logger.warning(
                    f"StopMonitor: order {order_id[:8]}... ended in state "
                    f"'{st}' before fill"
                )
                return None
        return None

    async def _sl_leg_fill_price(
        self, client, sl_leg_id: Optional[str],
    ) -> Optional[float]:
        """If the broker-side bracket SL leg won the exit race, its order record
        carries the real fill price. Look that up so a race-condition
        reconciliation doesn't fall back to a made-up price.

        Returns the SL leg's `filled_avg_price` or None if we can't recover it
        (missing id, API failure, or the leg isn't filled).
        """
        if not sl_leg_id:
            return None
        loop = asyncio.get_event_loop()
        try:
            status_info = await loop.run_in_executor(
                None, client.get_order, sl_leg_id
            )
        except Exception as e:
            logger.debug(
                f"StopMonitor: SL leg {sl_leg_id[:8]} lookup failed: {e}"
            )
            return None
        if str(status_info.get('status', '')).lower() != 'filled':
            return None
        price = status_info.get('filled_avg_price')
        return float(price) if price else None

    async def _escalate_to_market_close(
        self, client, symbol: str, stale_limit_order_id: str,
        trigger_price: float, sl_leg_id: Optional[str] = None,
    ) -> tuple:
        """Cancel a stale limit, submit a market close, poll for its fill.

        Returns (fill_price, market_order_id). fill_price is the real
        filled_avg_price when we can recover one (limit filled during cancel
        race, market close confirmed, or broker-side bracket SL race resolved
        via `_sl_leg_fill_price`). Falls back to `trigger_price` with an ERROR
        log and a "verify manually" flag only when every recovery path fails.

        `sl_leg_id` is queried when `close_position` raises the race-condition
        error (Alpaca says "position not found"): the bracket SL leg probably
        won, and its order record has the real fill.

        Note: the bulk-cancel upstream in `_execute_stop_exit` already freed
        our shares by cancelling all open orders for the symbol before the
        limit submit — so by the time we reach this escalation, the SL leg
        is usually ALREADY cancelled and can't protect anything. The SL
        recovery below only pays off when the bulk-cancel itself errored
        and the SL survived long enough to fill.
        """
        loop = asyncio.get_event_loop()
        logger.warning(
            f"StopMonitor: {symbol} limit sell UNFILLED after "
            f"{self._STOP_EXIT_FILL_TIMEOUT_S}s — cancel + market close "
            f"(bid moved through limit)"
        )
        try:
            await loop.run_in_executor(None, client.cancel_order, stale_limit_order_id)
        except Exception as e:
            logger.debug(
                f"StopMonitor: {symbol} cancel after fill-timeout: {e}"
            )
        # Race: limit may have filled during cancel — check one more time.
        try:
            status_info = await loop.run_in_executor(
                None, client.get_order, stale_limit_order_id
            )
            if str(status_info.get('status', '')).lower() == 'filled':
                price = float(status_info.get('filled_avg_price') or trigger_price)
                logger.info(
                    f"StopMonitor: {symbol} limit filled during cancel race "
                    f"at ${price:.4f}"
                )
                return price, stale_limit_order_id
        except Exception:
            pass

        # Submit market close
        try:
            close_result = await loop.run_in_executor(
                None, client.close_position, symbol
            )
            mkt_order_id = close_result.get('id', '') or ''
            logger.info(
                f"StopMonitor: {symbol} market close submitted order={mkt_order_id} "
                f"— awaiting fill confirmation"
            )
        except Exception as e:
            if self._is_race_condition_error(e):
                logger.info(
                    f"StopMonitor: {symbol} market close race — position "
                    f"already flat; recovering fill from SL leg"
                )
                sl_price = await self._sl_leg_fill_price(client, sl_leg_id)
                if sl_price is not None:
                    logger.info(
                        f"StopMonitor: {symbol} bracket SL filled at "
                        f"${sl_price:.4f} — using as exit price"
                    )
                    return sl_price, ''
                logger.error(
                    f"StopMonitor: {symbol} POSITION FLAT BUT FILL PRICE UNKNOWN — "
                    f"DB will record trigger_price=${trigger_price:.2f}. "
                    f"VERIFY POSITION MANUALLY on Alpaca."
                )
            else:
                logger.error(
                    f"StopMonitor: {symbol} market close failed: {e}"
                )
            return trigger_price, ''

        if not mkt_order_id:
            logger.error(
                f"StopMonitor: {symbol} market close returned no order id"
            )
            return trigger_price, ''

        price = await self._poll_order_fill(
            client, mkt_order_id, fallback_price=trigger_price,
        )
        if price is None:
            logger.error(
                f"StopMonitor: {symbol} MARKET CLOSE DID NOT CONFIRM FILL within "
                f"{self._STOP_EXIT_FILL_TIMEOUT_S}s — DB will record trigger "
                f"price as last-resort. VERIFY POSITION MANUALLY on Alpaca."
            )
            return trigger_price, mkt_order_id
        return price, mkt_order_id

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
        # Route order-submission calls to the correct Alpaca account when
        # alpaca_clients_by_strategy was configured. Market-data calls below
        # (get_latest_quote) stay on the default client — quotes are public.
        client = self._client_for(watch.strategy)
        order_id = ""

        try:
            # Cancel TP leg first (safe — doesn't protect downside)
            if watch.tp_leg_id:
                try:
                    await loop.run_in_executor(
                        None, client.cancel_order, watch.tp_leg_id
                    )
                    logger.info(
                        f"StopMonitor: {symbol} cancelled TP leg {watch.tp_leg_id}"
                    )
                except Exception as e:
                    logger.warning(
                        f"StopMonitor: {symbol} TP cancel failed (may be filled): {e}"
                    )

            # NOTE on naked-window: Alpaca holds shares inside active bracket
            # legs, blocking a separate limit sell. So the bulk-cancel below
            # (before submit) must kill the SL leg too, which means there's
            # NO broker-side stop active during the fill-poll window (up to
            # ~10s limit poll + ~10s market-close fallback = 20s max). This
            # is an accepted trade-off: we're *actively* exiting via our own
            # limit + market-close fallback, and the max window is short vs
            # the 10% bracket buffer. The trailing `watch.sl_leg_id` cancel
            # after fill confirmation is a best-effort backstop in case the
            # bulk-cancel's get_orders call errored and left the SL alive.

            # Compute limit price: use cached quote from stream (sub-50ms),
            # fall back to REST if stale, then fixed offset as last resort
            bid, ask, bid_size, ask_size = 0.0, 0.0, 0, 0
            pricing_method = 'fixed_offset'
            quote_age_ms = (time_mod.time() - watch.latest_quote_ts) * 1000 if watch.latest_quote_ts > 0 else float('inf')

            if watch.latest_bid > 0 and watch.latest_ask > 0 and quote_age_ms < 5000:
                # Use cached quote from WebSocket stream
                bid, ask = watch.latest_bid, watch.latest_ask
                bid_size, ask_size = watch.latest_bid_size, watch.latest_ask_size
                # Stop exits: always price BELOW bid (selling into weakness on a
                # falling stock). Pricing AT bid was the 2026-04-23 BMNZ bug —
                # limit at $14.03 never filled because bid dropped to $13.86
                # within seconds of submission. compute_limit_price applies a
                # max(3¢, 50bps) buffer below, keeping the limit marketable
                # even if bid slips during submission latency.
                limit_price = self.compute_limit_price(bid)
                pricing_method = 'stop_bid'
                if limit_price > 0:
                    logger.info(
                        f"StopMonitor: {symbol} stop-exit pricing — "
                        f"bid=${bid:.2f} ask=${ask:.2f} spread=${ask-bid:.3f} "
                        f"age={quote_age_ms:.0f}ms ofi={watch.ofi_cumulative:.0f} "
                        f"depth={bid_size}×{ask_size} → limit=${limit_price:.2f} ({pricing_method})"
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
                    if bid <= 0:
                        raise ValueError(f"invalid quote: bid={bid}, ask={ask}")
                    # Same buffer rationale as the WS-cache path above.
                    limit_price = self.compute_limit_price(bid)
                    pricing_method = 'stop_bid_rest'
                    logger.info(
                        f"StopMonitor: {symbol} REST stop-exit pricing — "
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

            # Cancel bracket legs BEFORE sell (they hold shares, blocking limit sell)
            try:
                from alpaca.trading.requests import GetOrdersRequest
                from alpaca.trading.enums import QueryOrderStatus
                open_orders = await loop.run_in_executor(
                    None,
                    lambda: client.trading_client.get_orders(
                        GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
                    ),
                )
                for oo in open_orders:
                    try:
                        await loop.run_in_executor(
                            None, client.cancel_order, str(oo.id)
                        )
                        logger.info(f"StopMonitor: {symbol} cancelled bracket leg {str(oo.id)[:8]}")
                    except Exception:
                        pass
            except Exception as cancel_err:
                logger.warning(f"StopMonitor: {symbol} cancel bracket legs: {cancel_err}")

            # Submit marketable limit sell (in executor)
            # After submit, we POLL for fill with a timeout. If the limit
            # hasn't filled within STOP_EXIT_FILL_TIMEOUT_S (bid moved
            # through our limit, thin book, etc.), we cancel and escalate
            # to a market close. Only after a confirmed fill do we write
            # the exit event — so DB never records a phantom close that
            # Alpaca didn't actually execute.
            #
            # Previously (pre-2026-04-23): we set `exit_price = limit_price`
            # on SUBMIT, emitted the event, and wrote the DB as closed. When
            # the limit sat unfilled (as with BMNZ on 2026-04-23), DB said
            # closed while Alpaca was still long — a silent desync that cost
            # us $288 extra loss on BMNZ before we noticed it manually.
            exit_price: Optional[float] = None
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: client.submit_limit_sell_order(
                        symbol=symbol,
                        qty=watch.shares,
                        limit_price=limit_price,
                    ),
                )
                order_id = result.get("id", "")
                logger.info(
                    f"StopMonitor: {symbol} limit sell submitted — "
                    f"qty={watch.shares}, limit=${limit_price:.2f} ({pricing_method}), "
                    f"order={order_id} — awaiting fill confirmation"
                )
                exit_price = await self._poll_order_fill(
                    client, order_id, fallback_price=limit_price,
                )
                if exit_price is None:
                    # Limit unfilled within timeout — escalate to market close.
                    # Pass sl_leg_id so we can recover the real price if the
                    # broker-side bracket SL won the race (rare but possible
                    # when the bulk-cancel above errored).
                    exit_price, mkt_order_id = await self._escalate_to_market_close(
                        client, symbol, order_id, trigger_price,
                        sl_leg_id=watch.sl_leg_id,
                    )
                    if mkt_order_id:
                        order_id = mkt_order_id
                        exit_reason = 'stop_loss_market_fallback'
            except Exception as e:
                # Bracket SL may have won the race — position already flat.
                # Don't page on this; just clean up and exit.
                if self._is_race_condition_error(e):
                    logger.info(
                        f"StopMonitor: {symbol} position already closed "
                        f"(bracket SL filled first) — skipping limit sell"
                    )
                    with self._watch_lock:
                        self._watches.pop(symbol, None)
                    with self._exit_lock:
                        self._exit_in_progress[symbol] = False
                    return

                logger.error(
                    f"StopMonitor: {symbol} limit sell failed: {e} — "
                    f"falling back to close_position()"
                )
                # Fallback: market close (in executor)
                try:
                    fallback = await loop.run_in_executor(
                        None, client.close_position, symbol
                    )
                    order_id = fallback.get("id", "")
                    exit_reason = "stop_loss_fallback"
                    logger.info(
                        f"StopMonitor: {symbol} fallback close_position — "
                        f"order={order_id} — awaiting fill confirmation"
                    )
                    exit_price = await self._poll_order_fill(
                        client, order_id, fallback_price=trigger_price,
                    )
                    if exit_price is None:
                        # Market close order never confirmed filled. Log loudly;
                        # use trigger_price as last-resort marker but flag it.
                        logger.error(
                            f"StopMonitor: {symbol} MARKET CLOSE UNCONFIRMED — "
                            f"DB will use trigger_price=${trigger_price:.2f}. "
                            f"VERIFY POSITION MANUALLY on Alpaca."
                        )
                        exit_price = trigger_price
                except Exception as e2:
                    # Race condition: bracket closed the position before our
                    # fallback ran. Log as INFO, not ERROR — Telegram stays quiet.
                    if self._is_race_condition_error(e2):
                        logger.info(
                            f"StopMonitor: {symbol} bracket SL already closed "
                            f"position — fallback close_position returned "
                            f"{e2.__class__.__name__}"
                        )
                    else:
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

            # Position confirmed flat (or we've exhausted retries with a
            # known-bad price). NOW safe to cancel any surviving SL leg and
            # use the actual fill price in the exit event. Every exit path
            # above either set exit_price to a real fill or to trigger_price
            # as a last-resort; `None` is impossible here, but defensive.
            if exit_price is None:
                exit_price = trigger_price
            if watch.sl_leg_id:
                try:
                    await loop.run_in_executor(
                        None, client.cancel_order, watch.sl_leg_id
                    )
                    logger.info(
                        f"StopMonitor: {symbol} cancelled SL leg {watch.sl_leg_id}"
                    )
                except Exception as e:
                    logger.warning(
                        f"StopMonitor: {symbol} SL cancel failed (may be filled): {e}"
                    )

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
                strategy=watch.strategy,
            )
            self._exit_events.put(event)

            # Remove from watch list and clear exit-in-progress
            with self._watch_lock:
                self._watches.pop(symbol, None)
            with self._exit_lock:
                self._exit_in_progress[symbol] = False

        except Exception as e:
            logger.error(f"StopMonitor: {symbol} exit execution error: {e}")
            with self._exit_lock:
                self._exit_in_progress[symbol] = False

    async def _subscribe_symbol(self, symbol: str) -> None:
        """Subscribe to trade and quote updates for a symbol.

        IMPORTANT: Cannot call self._stream.subscribe_trades() here because
        that method uses asyncio.run_coroutine_threadsafe().result() which
        deadlocks when called from the same event loop. Instead, we directly
        register handlers and send the subscribe message asynchronously.
        """
        if self._stream:
            try:
                # Register handlers directly (bypass sync .subscribe_trades)
                self._stream._handlers["trades"][symbol] = self._on_trade
                self._stream._handlers["quotes"][symbol] = self._on_quote
                # Send subscribe message asynchronously (no .result() deadlock)
                # Guard: _ws may be None if stream hasn't connected yet —
                # _run_forever will pick up handlers on next reconnect via
                # _send_subscribe_msg() after _start_ws().
                if self._stream._ws:
                    await self._stream._send_subscribe_msg()
                    logger.debug(f"StopMonitor: subscribed to {symbol} (trades+quotes)")
                else:
                    logger.info(
                        f"StopMonitor: {symbol} handlers registered, "
                        f"WS not connected yet — will subscribe on connect"
                    )
            except Exception as e:
                logger.error(
                    f"StopMonitor: failed to subscribe {symbol}: {e}"
                )

    async def _unsubscribe_symbol(self, symbol: str) -> None:
        """Unsubscribe from trade and quote updates for a symbol.

        Uses SDK's _send_unsubscribe_msg (action=unsubscribe) instead of
        re-sending subscribe with the symbol removed — avoids Alpaca 400
        'invalid syntax' error when unsubscribing the last symbol.
        """
        if self._stream:
            try:
                self._stream._handlers["trades"].pop(symbol, None)
                self._stream._handlers["quotes"].pop(symbol, None)
                if self._stream._ws:
                    await self._stream._send_unsubscribe_msg("trades", [symbol])
                    await self._stream._send_unsubscribe_msg("quotes", [symbol])
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
        """Get all symbols needing WebSocket subscription (stop + quote watches)."""
        with self._watch_lock:
            return list(set(self._watches.keys()) | set(self._quote_watches.keys()))
