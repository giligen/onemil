"""
OrderStreamWatcher — Alpaca TradingStream subscriber for real-time order updates.

Replaces polling `get_order(order_id)` for fill detection. Fill-detection
latency drops from ~30 s (60 s polling bounded) to <1 s (push).

The watcher runs one WebSocket connection on a daemon asyncio thread. A
trade-update callback maintains an in-memory `{order_id: status_dict}` map
whose fields mirror what `AlpacaClient.get_order()` returns, so call sites
can swap with minimal changes and fall back to REST if a status is missing.
"""

import asyncio
import contextlib
import logging
import threading
import time as time_mod
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)

# Terminal order statuses — once an order hits one of these, it won't receive
# further updates. Such entries can be pruned aggressively to bound memory.
_TERMINAL_STATUSES = frozenset({'filled', 'canceled', 'cancelled', 'expired',
                                 'rejected', 'done_for_day', 'replaced'})
_MAX_TERMINAL_ENTRIES = 2000  # retain most-recent 2000 terminal orders


class OrderStreamWatcher:
    """
    Owns one `alpaca.trading.stream.TradingStream` connection. Thread-safe.

    Public API:
      start()                       — launch the daemon thread (connects + subscribes)
      stop()                        — graceful shutdown
      is_healthy(max_stale_s=30)    — last event received within max_stale_s
      get_status(order_id) -> dict  — latest-known status or None

    Status dict shape (matches `AlpacaClient.get_order()`):
      {
        'id': str,
        'status': str,                 # 'filled', 'partial_fill', 'canceled', ...
        'filled_avg_price': float|None,
        'filled_qty': int,
        'submitted_at': datetime,
        'filled_at': datetime|None,
        'updated_at': datetime,        # time we observed the update
        'event': str,                  # raw Alpaca event name
      }
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
        alpaca_client=None,  # optional, for reconnect re-sync via REST
    ):
        self._api_key = api_key
        self._api_secret = api_secret
        self._paper = paper
        self._alpaca = alpaca_client

        self._stream = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()

        # OrderedDict preserves insertion order so we can evict the oldest
        # terminal entries once _MAX_TERMINAL_ENTRIES is exceeded. Open orders
        # (non-terminal) are never evicted — they're still actionable.
        self._statuses: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self._last_event_ts: float = 0.0

    # -------- public API --------

    def start(self) -> None:
        """Launch the daemon thread. Non-blocking."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("OrderStreamWatcher already running")
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_stream_loop,
            name="OrderStreamWatcher-WS",
            daemon=True,
        )
        self._thread.start()
        logger.info("OrderStreamWatcher started (paper=%s)", self._paper)

    def stop(self) -> None:
        """Graceful shutdown — closes the stream and joins the thread."""
        self._running = False
        self._stop_event.set()
        if self._loop and self._stream:
            try:
                fut = asyncio.run_coroutine_threadsafe(self._close_stream(), self._loop)
                fut.result(timeout=5)
            except Exception as e:
                logger.warning("OrderStreamWatcher: close error: %s", e)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._stream = None
        logger.info("OrderStreamWatcher stopped")

    def is_healthy(self, max_stale_s: float = 30.0) -> bool:
        """True if the thread is alive and we've received an event within max_stale_s.

        Before any event arrives (empty account), healthy = thread alive.
        """
        if not self._running or not self._thread or not self._thread.is_alive():
            return False
        if self._last_event_ts == 0.0:
            return True  # no orders yet; not a health problem
        return (time_mod.time() - self._last_event_ts) <= max_stale_s

    def get_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Thread-safe read. Returns None if never observed."""
        with self._lock:
            return self._statuses.get(order_id)

    def snapshot_by_client_prefix(self, prefix: str) -> Dict[str, Dict[str, Any]]:
        """Latest-known statuses for orders whose client_order_id starts
        with `prefix`, keyed by client_order_id.

        Ignition prestage (2026-08-22, P0-4): staged stop-limit fills are
        consumed from THIS stream by the `ign-stage-` client_order_id
        prefix instead of per-order REST polling — at 100-310 staged
        orders/day, per-order polling would burn the shared 200/min
        trading-API budget instantly. Returns copies (thread-safe reads).
        """
        out: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for s in self._statuses.values():
                coid = s.get('client_order_id') or ''
                if coid.startswith(prefix):
                    out[coid] = dict(s)
        return out

    def get_open_order_symbols(self) -> Set[str]:
        """
        Set of symbols currently holding a non-terminal order, per the stream
        cache. Used by MACDWaveEngine as a fast local wash-trade check instead
        of a REST round-trip on the entry hot path.

        Correctness depends on the cache being fresh — we populate from
        _resync_from_rest on (re)connect, then keep it live via push events.
        Callers should gate on `is_healthy()` and fall back to REST when
        the stream is unavailable or stale.
        """
        with self._lock:
            return {
                s.get('symbol') for s in self._statuses.values()
                if s.get('symbol')
                and (s.get('status') or '').lower() not in _TERMINAL_STATUSES
            }

    # -------- internals --------

    def _run_stream_loop(self) -> None:
        """Daemon thread entry — owns the asyncio loop for the stream."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._stream_with_reconnect())
        except Exception as e:
            logger.error("OrderStreamWatcher: thread crashed: %s", e, exc_info=True)
        finally:
            try:
                self._loop.close()
            except Exception:
                pass
            self._loop = None

    async def _stream_with_reconnect(self) -> None:
        """Reconnect loop with exponential backoff up to 30 s."""
        backoff = 1.0
        while self._running:
            try:
                from alpaca.trading.stream import TradingStream
                self._stream = TradingStream(
                    self._api_key, self._api_secret, paper=self._paper
                )
                self._stream.subscribe_trade_updates(self._on_trade_update)

                logger.info("OrderStreamWatcher: connecting TradingStream...")
                # Run the stream as a task so the REST resync can happen
                # AFTER the WS is actually connected. Previous ordering did
                # resync before connect, which created a window where fills
                # between the REST snapshot and WS-online were missed. Now:
                #   1. subscribe handler (stream will deliver events on connect)
                #   2. start `_run_forever` task — WS handshake begins
                #   3. sleep briefly for the handshake to complete
                #   4. REST resync — reconciles any state we couldn't see yet
                #   5. await the stream task (blocks until WS exits)
                #
                # NOTE: alpaca-py's public `run()` creates its own event loop,
                # incompatible with our daemon-thread loop. Call the private
                # async entry point `_run_forever()` — same pattern used by
                # StopMonitor. Pinned via requirements.txt upper bound.
                stream_task = asyncio.create_task(self._stream._run_forever())
                try:
                    # Give the WS handshake ~1.5s to complete before firing
                    # the REST resync. Empirically enough for Alpaca's TLS +
                    # subscribe acknowledgment on healthy connections.
                    await asyncio.sleep(1.5)
                    await self._resync_from_rest()
                    await stream_task
                except asyncio.CancelledError:
                    stream_task.cancel()
                    with contextlib.suppress(Exception):
                        await stream_task
                    raise
                backoff = 1.0  # reset on clean exit
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not self._running:
                    break
                logger.warning(
                    "OrderStreamWatcher: stream error, reconnecting in %.0fs: %s",
                    backoff, e
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _resync_from_rest(self) -> None:
        """Best-effort: fetch open orders via REST to seed the status map.

        The Alpaca TradingClient is synchronous, so we run the call in a worker
        thread via asyncio.to_thread to avoid blocking the stream event loop.
        """
        if not self._alpaca:
            return
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            existing = await asyncio.to_thread(
                self._alpaca.trading_client.get_orders, filter=req
            )
            if not existing:
                return
            with self._lock:
                for o in existing:
                    oid = getattr(o, 'id', None) or getattr(o, 'client_order_id', None)
                    if not oid:
                        continue
                    self._statuses[str(oid)] = _order_to_status(o, event='resync')
            logger.info(
                "OrderStreamWatcher: resynced %d open orders from REST", len(existing)
            )
        except Exception as e:
            logger.warning("OrderStreamWatcher: REST resync failed: %s", e)

    async def _on_trade_update(self, data) -> None:
        """Alpaca TradeUpdate callback — writes to the status map."""
        try:
            order = getattr(data, 'order', None)
            event = getattr(data, 'event', None)
            event_str = event.value if hasattr(event, 'value') else str(event) if event else ''
            if order is None:
                return
            oid = getattr(order, 'id', None)
            if not oid:
                return
            status_dict = _order_to_status(order, event=event_str)
            with self._lock:
                self._statuses[str(oid)] = status_dict
                self._prune_terminal_locked()
            self._last_event_ts = time_mod.time()
        except Exception as e:
            logger.error("OrderStreamWatcher: _on_trade_update error: %s", e)

    def _prune_terminal_locked(self) -> None:
        """
        Evict oldest terminal entries when over cap. Caller holds _lock.

        Note: eviction order is by *first-insertion* into the OrderedDict, not
        by when the order reached a terminal status. This is fine for memory
        bounding (our goal) but means an order submitted early in the day that
        filled recently may be evicted before one submitted later that filled
        earlier. We don't guarantee "most-recently-terminal first".
        """
        # Count terminal entries; if over limit, evict from the front of the
        # OrderedDict (oldest first) while preserving open orders.
        terminal_ids = [
            oid for oid, s in self._statuses.items()
            if (s.get('status') or '').lower() in _TERMINAL_STATUSES
        ]
        overflow = len(terminal_ids) - _MAX_TERMINAL_ENTRIES
        if overflow <= 0:
            return
        for oid in terminal_ids[:overflow]:
            self._statuses.pop(oid, None)

    async def _close_stream(self) -> None:
        if self._stream:
            try:
                await self._stream.stop_ws()
            except Exception:
                pass


def _order_to_status(order, event: str = '') -> Dict[str, Any]:
    """Normalize an Alpaca Order object into the dict shape call sites expect."""
    def _f(x):
        try:
            return float(x) if x is not None else None
        except (TypeError, ValueError):
            return None

    def _i(x):
        try:
            return int(x) if x is not None else 0
        except (TypeError, ValueError):
            return 0

    def _s(x):
        if x is None:
            return ''
        return x.value if hasattr(x, 'value') else str(x)

    # reject_reason: Alpaca sometimes attaches a reason field on the Order
    # (alpaca-py vs raw REST differ; defensive over multiple known names).
    # If none of these are populated, leave as None — DB column allows NULL.
    # Captured here so post-mortems can SQL-query rejections directly instead
    # of grepping journalctl + querying REST. See db Migration 13.
    _reject_reason = (
        getattr(order, 'reject_reason', None)
        or getattr(order, 'cancel_reason', None)
        or getattr(order, 'reason', None)
    )
    return {
        'id': str(getattr(order, 'id', '') or ''),
        # client_order_id: prestage fill routing keys on the ign-stage-
        # prefix (see snapshot_by_client_prefix); '' when absent.
        'client_order_id': str(getattr(order, 'client_order_id', '') or ''),
        'symbol': str(getattr(order, 'symbol', '') or ''),
        'status': _s(getattr(order, 'status', '')),
        'filled_avg_price': _f(getattr(order, 'filled_avg_price', None)),
        'filled_qty': _i(getattr(order, 'filled_qty', None)),
        'submitted_at': getattr(order, 'submitted_at', None),
        'filled_at': getattr(order, 'filled_at', None),
        'updated_at': datetime.now(timezone.utc),
        'event': event,
        'reject_reason': str(_reject_reason)[:100] if _reject_reason else None,
    }
