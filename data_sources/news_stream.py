"""
Real-time news stream listener for the Alpaca news WebSocket.

Connects to wss://stream.data.alpaca.markets/v1beta1/news, authenticates
with the standard ALPACA_API_KEY / ALPACA_API_SECRET pair, subscribes to
["*"] and fans parsed NewsEvent objects out to registered consumers.

Design goals (owner directive 2026-08: "realtime news feed is a must"):
  1. Always-on session component: own daemon thread, auto-reconnect with
     capped exponential backoff, staleness tracking, clean shutdown.
  2. Multi-symbol aware event model: ONE article -> ONE NewsEvent carrying
     the FULL symbol list (this is the complex-group solver — consumers
     see every ticker an article tags, so wrapper/underlying groups can be
     resolved downstream).
  3. Consumer bus with fault isolation: one raising consumer never kills
     the stream loop.
  4. Latency telemetry: per-event '[NEWS-STREAM] ... recv_latency=...s'
     log line + rolling stats (count, median, p90, staleness gaps) — the
     wire->us lag number the prestage Tier-1 design needs.
  5. Standalone shadow runner:  python3 -m data_sources.news_stream --shadow
     Zero coupling to trading — Monday's measurement mode.

NOT yet wired into main.py / the trader (config.yaml.template ships
news_stream.enabled: false); integration is a later task.
"""

import argparse
import asyncio
import json
import logging
import math
import os
import statistics
import threading
import time
from collections import deque
import dataclasses
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Set

import pytz

logger = logging.getLogger(__name__)

ET_TZ = pytz.timezone('US/Eastern')

DEFAULT_NEWS_STREAM_URL = 'wss://stream.data.alpaca.markets/v1beta1/news'

# Reconnect backoff: base * 2**(attempt-1), capped.
DEFAULT_BACKOFF_BASE_S = 1.0
DEFAULT_BACKOFF_MAX_S = 60.0

# Staleness: no inbound frame for this long -> is_stale() True by default.
DEFAULT_STALE_AFTER_S = 60.0

# recv() poll timeout — short so stop() is honoured promptly and the loop
# can notice wall-clock staleness even on a silent socket.
_RECV_POLL_TIMEOUT_S = 5.0

# Rolling telemetry window (events). ~a full day of Benzinga wire volume.
_LATENCY_WINDOW = 4096


def _utcnow() -> datetime:
    """Timezone-aware UTC now (single seam for tests)."""
    return datetime.now(timezone.utc)


def _today_et() -> date:
    """Current calendar date in US/Eastern (index day-roll boundary)."""
    return datetime.now(ET_TZ).date()


def parse_iso8601(value: str) -> Optional[datetime]:
    """Parse an Alpaca ISO-8601 timestamp ('...Z' or offset form) to UTC.

    Returns None (with a WARNING) on unparseable input — a bad vendor
    timestamp must not kill the stream loop.
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        logger.warning("[NEWS-STREAM] unparseable timestamp %r — dropping "
                       "latency measurement for this event", value)
        return None
    if dt.tzinfo is None:
        # Alpaca always sends offsets; a naive value means the vendor
        # changed format — assume UTC but say so.
        logger.warning("[NEWS-STREAM] naive timestamp %r — assuming UTC", value)
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class NewsEvent:
    """One news article as delivered on the wire.

    MULTI-SYMBOL AWARE: `symbols` is the article's complete ticker list —
    one article produces exactly one event, never one-per-symbol.
    """
    article_id: str
    headline: str
    symbols: tuple  # tuple[str, ...] — full set, uppercased
    source: str
    created_at: Optional[datetime]   # UTC
    updated_at: Optional[datetime]   # UTC
    received_at: datetime            # UTC, when WE got the frame
    is_update: bool = False          # True if this id was seen before

    @property
    def recv_latency_s(self) -> Optional[float]:
        """Wire->us lag in seconds (received_at - created_at); None if
        created_at was unparseable."""
        if self.created_at is None:
            return None
        return (self.received_at - self.created_at).total_seconds()


def parse_news_message(msg: dict, received_at: Optional[datetime] = None,
                       is_update: bool = False) -> Optional[NewsEvent]:
    """Parse one raw Alpaca news dict ({"T":"n", ...}) into a NewsEvent.

    Returns None (with a WARNING) for malformed messages — parsing
    failures are logged, never raised, so one bad frame can't kill the
    consume loop.
    """
    if not isinstance(msg, dict) or msg.get('T') != 'n':
        return None
    article_id = msg.get('id')
    if article_id is None:
        logger.warning("[NEWS-STREAM] news message without id — dropped: %r",
                       str(msg)[:200])
        return None
    raw_symbols = msg.get('symbols') or []
    symbols = tuple(sorted({s.strip().upper() for s in raw_symbols
                            if isinstance(s, str) and s.strip()}))
    return NewsEvent(
        article_id=str(article_id),
        headline=(msg.get('headline') or '').strip(),
        symbols=symbols,
        source=(msg.get('source') or '').strip(),
        created_at=parse_iso8601(msg.get('created_at') or ''),
        updated_at=parse_iso8601(msg.get('updated_at') or ''),
        received_at=received_at or _utcnow(),
        is_update=is_update,
    )


def nearest_rank_percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile over an already-sorted sequence.

    Deterministic (no interpolation): value at rank ceil(pct/100 * n).
    """
    if not sorted_values:
        raise ValueError("nearest_rank_percentile of empty sequence")
    n = len(sorted_values)
    rank = max(1, math.ceil((pct / 100.0) * n))
    return sorted_values[min(rank, n) - 1]


class LatencyStats:
    """Thread-safe rolling recv-latency + inter-arrival gap telemetry.

    Feeds the EOD dive: count, median/p90 wire->us latency, and staleness
    gaps (inter-event wall-clock gaps above `gap_threshold_s`).
    """

    def __init__(self, window: int = _LATENCY_WINDOW,
                 gap_threshold_s: float = DEFAULT_STALE_AFTER_S,
                 time_fn: Callable[[], float] = time.monotonic):
        self._lock = threading.Lock()
        self._latencies: deque = deque(maxlen=window)
        self._gap_threshold_s = gap_threshold_s
        self._time_fn = time_fn
        self._last_event_mono: Optional[float] = None
        self._max_gap_s = 0.0
        self._gaps_over_threshold = 0
        self._events_total = 0

    def record_event(self, recv_latency_s: Optional[float]) -> None:
        """Record one delivered event (latency may be None if the vendor
        timestamp was unparseable — the gap tracking still counts it)."""
        now = self._time_fn()
        with self._lock:
            self._events_total += 1
            if self._last_event_mono is not None:
                gap = now - self._last_event_mono
                if gap > self._max_gap_s:
                    self._max_gap_s = gap
                if gap > self._gap_threshold_s:
                    self._gaps_over_threshold += 1
            self._last_event_mono = now
            if recv_latency_s is not None:
                self._latencies.append(recv_latency_s)

    def snapshot(self) -> dict:
        """Point-in-time stats dict for logging / the shadow runner."""
        with self._lock:
            lat = sorted(self._latencies)
            out = {
                'events_total': self._events_total,
                'latency_count': len(lat),
                'latency_median_s': None,
                'latency_p90_s': None,
                'latency_min_s': None,
                'latency_max_s': None,
                'max_gap_s': round(self._max_gap_s, 1),
                'gaps_over_%ds' % int(self._gap_threshold_s):
                    self._gaps_over_threshold,
            }
            if lat:
                out['latency_median_s'] = round(statistics.median(lat), 3)
                out['latency_p90_s'] = round(
                    nearest_rank_percentile(lat, 90.0), 3)
                out['latency_min_s'] = round(lat[0], 3)
                out['latency_max_s'] = round(lat[-1], 3)
            return out


class TodayNewsIndex:
    """Thread-safe per-symbol index of today's articles, ET-date rolled.

    symbols_with_news_today() / articles_for(sym) / first_article_ts(sym).
    The index self-clears the first time it is touched on a new ET date.
    """

    def __init__(self, date_provider: Callable[[], date] = _today_et):
        self._lock = threading.Lock()
        self._date_provider = date_provider
        self._day: date = date_provider()
        self._by_symbol: Dict[str, List[NewsEvent]] = {}

    def _roll_if_needed_locked(self) -> None:
        today = self._date_provider()
        if today != self._day:
            logger.info("[NEWS-STREAM] ET day roll %s -> %s — clearing "
                        "today-index (%d symbols)", self._day, today,
                        len(self._by_symbol))
            self._day = today
            self._by_symbol = {}

    def add(self, event: NewsEvent) -> None:
        """Index one event under each of its symbols (multi-symbol aware)."""
        with self._lock:
            self._roll_if_needed_locked()
            for sym in event.symbols:
                self._by_symbol.setdefault(sym, []).append(event)

    def symbols_with_news_today(self) -> Set[str]:
        """Set of symbols with at least one article today (ET)."""
        with self._lock:
            self._roll_if_needed_locked()
            return set(self._by_symbol.keys())

    def articles_for(self, symbol: str) -> List[NewsEvent]:
        """Today's events for `symbol`, in arrival order (copy)."""
        with self._lock:
            self._roll_if_needed_locked()
            return list(self._by_symbol.get(symbol.upper(), []))

    def first_article_ts(self, symbol: str) -> Optional[datetime]:
        """UTC created_at (fallback received_at) of the symbol's first
        article today, or None."""
        with self._lock:
            self._roll_if_needed_locked()
            events = self._by_symbol.get(symbol.upper())
            if not events:
                return None
            first = events[0]
            return first.created_at or first.received_at


class NewsStreamListener:
    """Always-on Alpaca news WebSocket listener (daemon thread).

    Lifecycle:  start() -> [auto-reconnect forever] -> stop().
    Consumers:  register_callback(fn: NewsEvent -> None).
    Health:     last_event_ts / is_stale(threshold).
    Telemetry:  latency_stats().
    Index:      symbols_with_news_today() / articles_for() /
                first_article_ts().
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 url: str = DEFAULT_NEWS_STREAM_URL,
                 subscribe_symbols: Sequence[str] = ('*',),
                 stale_after_s: float = DEFAULT_STALE_AFTER_S,
                 backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
                 backoff_max_s: float = DEFAULT_BACKOFF_MAX_S,
                 connect_factory: Optional[Callable] = None,
                 date_provider: Callable[[], date] = _today_et,
                 time_fn: Callable[[], float] = time.monotonic):
        self._api_key = api_key or os.getenv('ALPACA_API_KEY', '')
        self._api_secret = api_secret or os.getenv('ALPACA_API_SECRET', '')
        if not self._api_key or not self._api_secret:
            raise ValueError(
                "NewsStreamListener requires ALPACA_API_KEY and "
                "ALPACA_API_SECRET (env or constructor args)")
        self._url = url
        self._subscribe_symbols = list(subscribe_symbols)
        self._stale_after_s = stale_after_s
        self._backoff_base_s = backoff_base_s
        self._backoff_max_s = backoff_max_s
        self._connect_factory = connect_factory  # test seam; None -> websockets
        self._time_fn = time_fn

        self._callbacks: List[Callable[[NewsEvent], None]] = []
        self._callbacks_lock = threading.Lock()

        self._stats = LatencyStats(gap_threshold_s=stale_after_s,
                                   time_fn=time_fn)
        self._index = TodayNewsIndex(date_provider=date_provider)

        # Dedup state: article id -> last updated_at seen (UTC or None).
        self._seen_lock = threading.Lock()
        self._seen_updated_at: Dict[str, Optional[datetime]] = {}
        self._duplicates_dropped = 0
        self._updates_seen = 0

        self._stop_requested = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._started_mono: Optional[float] = None
        self._last_frame_mono: Optional[float] = None
        self._last_event_ts: Optional[datetime] = None  # UTC, news events only
        self._reconnects = 0

    # ------------------------------------------------------------------ #
    # Consumer bus
    # ------------------------------------------------------------------ #
    def register_callback(self, fn: Callable[[NewsEvent], None]) -> None:
        """Register a consumer. Exceptions in a consumer are logged as
        ERROR and never propagate to the stream loop or other consumers."""
        with self._callbacks_lock:
            self._callbacks.append(fn)

    def _dispatch(self, event: NewsEvent) -> None:
        with self._callbacks_lock:
            callbacks = list(self._callbacks)
        for fn in callbacks:
            try:
                fn(event)
            except Exception:
                logger.exception(
                    "[NEWS-STREAM] consumer %s raised on article %s — "
                    "continuing (consumer bus is fault-isolated)",
                    getattr(fn, '__name__', repr(fn)), event.article_id)

    # ------------------------------------------------------------------ #
    # Health / telemetry accessors
    # ------------------------------------------------------------------ #
    @property
    def last_event_ts(self) -> Optional[datetime]:
        """UTC received_at of the last news event (None before the first)."""
        return self._last_event_ts

    def is_stale(self, threshold_s: Optional[float] = None) -> bool:
        """True if no inbound frame (news OR control/handshake) within
        `threshold_s` (default: constructor stale_after_s). True before
        start() — an unstarted listener has no fresh data by definition."""
        threshold = threshold_s if threshold_s is not None else self._stale_after_s
        anchor = self._last_frame_mono if self._last_frame_mono is not None \
            else self._started_mono
        if anchor is None:
            return True
        return (self._time_fn() - anchor) > threshold

    def latency_stats(self) -> dict:
        """Rolling latency/gap stats + dedup counters (EOD-dive input)."""
        snap = self._stats.snapshot()
        with self._seen_lock:
            snap['duplicates_dropped'] = self._duplicates_dropped
            snap['updates_seen'] = self._updates_seen
            snap['unique_articles'] = len(self._seen_updated_at)
        snap['reconnects'] = self._reconnects
        return snap

    def symbols_with_news_today(self) -> Set[str]:
        """Symbols with at least one article today (ET-date rolled)."""
        return self._index.symbols_with_news_today()

    def articles_for(self, symbol: str) -> List[NewsEvent]:
        """Today's articles for one symbol (ET-date rolled)."""
        return self._index.articles_for(symbol)

    def first_article_ts(self, symbol: str) -> Optional[datetime]:
        """UTC timestamp of the symbol's first article today, or None."""
        return self._index.first_article_ts(symbol)

    # ------------------------------------------------------------------ #
    # Message handling (sync core — unit-testable without a socket)
    # ------------------------------------------------------------------ #
    def _classify_dedup(self, article_id: str,
                        updated_at: Optional[datetime]) -> Optional[bool]:
        """Dedup verdict for an article id.

        Returns False (new article), True (tracked update: same id, changed
        updated_at), or None (exact duplicate — drop)."""
        with self._seen_lock:
            if article_id not in self._seen_updated_at:
                self._seen_updated_at[article_id] = updated_at
                return False
            if self._seen_updated_at[article_id] != updated_at:
                self._seen_updated_at[article_id] = updated_at
                self._updates_seen += 1
                return True
            self._duplicates_dropped += 1
            return None

    def _handle_raw(self, raw: str,
                    received_at: Optional[datetime] = None) -> List[NewsEvent]:
        """Parse one raw frame (JSON array of messages) and process every
        news message: dedup, telemetry, index, per-event latency log line,
        consumer fan-out. Returns the delivered events (for tests)."""
        received_at = received_at or _utcnow()
        try:
            messages = json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("[NEWS-STREAM] undecodable frame dropped: %r",
                           str(raw)[:200])
            return []
        if isinstance(messages, dict):
            messages = [messages]
        if not isinstance(messages, list):
            logger.warning("[NEWS-STREAM] unexpected frame shape %s dropped",
                           type(messages).__name__)
            return []

        delivered: List[NewsEvent] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            mtype = msg.get('T')
            if mtype == 'n':
                event = self._process_news_message(msg, received_at)
                if event is not None:
                    delivered.append(event)
            elif mtype == 'error':
                # Surface protocol errors loudly; the async loop decides
                # whether to reconnect.
                logger.error("[NEWS-STREAM] server error frame: %s", msg)
            elif mtype in ('success', 'subscription'):
                logger.info("[NEWS-STREAM] control: %s", msg)
            else:
                logger.warning("[NEWS-STREAM] unknown message type %r dropped",
                               mtype)
        return delivered

    def _process_news_message(self, msg: dict,
                              received_at: datetime) -> Optional[NewsEvent]:
        event = parse_news_message(msg, received_at=received_at)
        if event is None:
            return None
        verdict = self._classify_dedup(event.article_id, event.updated_at)
        if verdict is None:
            logger.debug("[NEWS-STREAM] duplicate article %s dropped",
                         event.article_id)
            return None
        if verdict:
            event = dataclasses.replace(event, is_update=True)

        self._last_event_ts = event.received_at
        latency = event.recv_latency_s
        self._stats.record_event(latency)
        self._index.add(event)

        syms = ','.join(event.symbols) or '-'
        latency_str = ('%.2f' % latency) if latency is not None else 'n/a'
        logger.info("[NEWS-STREAM] %s recv_latency=%ss%s src=%s id=%s | %s",
                    syms, latency_str,
                    ' UPDATE' if event.is_update else '',
                    event.source, event.article_id, event.headline[:120])
        self._dispatch(event)
        return event

    # ------------------------------------------------------------------ #
    # Async connection loop
    # ------------------------------------------------------------------ #
    def _next_backoff_s(self, attempt: int) -> float:
        """Capped exponential backoff for reconnect `attempt` (1-based)."""
        return min(self._backoff_max_s,
                   self._backoff_base_s * (2 ** max(0, attempt - 1)))

    def _get_connect_factory(self) -> Callable:
        if self._connect_factory is not None:
            return self._connect_factory
        import websockets  # deferred: tests never need the real lib
        return websockets.connect

    async def _consume_connection(self, ws) -> None:
        """Auth + subscribe + recv-loop on one live connection.

        Raises on socket errors — the caller owns reconnect policy."""
        await ws.send(json.dumps({'action': 'auth',
                                  'key': self._api_key,
                                  'secret': self._api_secret}))
        await ws.send(json.dumps({'action': 'subscribe',
                                  'news': self._subscribe_symbols}))
        logger.info("[NEWS-STREAM] connected to %s, subscribed news=%s",
                    self._url, self._subscribe_symbols)
        while not self._stop_requested.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(),
                                             timeout=_RECV_POLL_TIMEOUT_S)
            except asyncio.TimeoutError:
                continue  # poll tick: re-check stop flag
            self._last_frame_mono = self._time_fn()
            self._handle_raw(raw)

    async def _run_forever(self) -> None:
        """Reconnect loop: connect/consume until stop; on any error, log a
        WARNING and back off exponentially (capped)."""
        attempt = 0
        connect = self._get_connect_factory()
        while not self._stop_requested.is_set():
            try:
                async with await self._ensure_awaitable(connect(self._url)) as ws:
                    self._last_frame_mono = self._time_fn()
                    attempt = 0  # successful connection resets the ladder
                    await self._consume_connection(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stop_requested.is_set():
                    break
                attempt += 1
                self._reconnects += 1
                delay = self._next_backoff_s(attempt)
                logger.warning(
                    "[NEWS-STREAM] connection lost (%s: %s) — reconnect #%d "
                    "in %.1fs", type(exc).__name__, exc, self._reconnects,
                    delay)
                await self._interruptible_sleep(delay)
        logger.info("[NEWS-STREAM] loop stopped cleanly")

    @staticmethod
    async def _ensure_awaitable(obj):
        """Accept both awaitable connect() (websockets>=11 returns an
        awaitable/async-CM hybrid) and plain async-CM test fakes."""
        if asyncio.iscoroutine(obj) or asyncio.isfuture(obj):
            return await obj
        return obj

    async def _interruptible_sleep(self, delay: float) -> None:
        """Backoff sleep in short slices so stop() is honoured promptly."""
        end = self._time_fn() + delay
        while not self._stop_requested.is_set() and self._time_fn() < end:
            await asyncio.sleep(min(0.5, max(0.0, end - self._time_fn())))

    # ------------------------------------------------------------------ #
    # Thread lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Spawn the daemon stream thread. Idempotent-hostile by design:
        raises if already started (call stop() first)."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("NewsStreamListener already started")
        self._stop_requested.clear()
        self._started_mono = self._time_fn()
        self._thread = threading.Thread(target=self._thread_main,
                                        name='news-stream', daemon=True)
        self._thread.start()
        logger.info("[NEWS-STREAM] listener thread started (url=%s)",
                    self._url)

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_forever())
        except Exception:
            logger.exception("[NEWS-STREAM] stream thread died unexpectedly — "
                             "no further news until restart")
        finally:
            self._loop.close()
            self._loop = None

    def stop(self, timeout_s: float = 15.0) -> None:
        """Clean shutdown: signal the loop, join the thread."""
        self._stop_requested.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout_s)
            if thread.is_alive():
                logger.error("[NEWS-STREAM] stream thread failed to stop "
                             "within %.1fs (daemon thread — will die with "
                             "process)", timeout_s)
        self._thread = None
        logger.info("[NEWS-STREAM] listener stopped")


# ---------------------------------------------------------------------- #
# Shadow runner — Monday's measurement mode (zero trading coupling)
# ---------------------------------------------------------------------- #
def _shadow_print_event(event: NewsEvent) -> None:
    """Stdout consumer for --shadow mode."""
    lat = event.recv_latency_s
    print("%s  %-24s  lat=%-8s  %s%s" % (
        event.received_at.strftime('%H:%M:%S'),
        ','.join(event.symbols)[:24] or '-',
        ('%.2fs' % lat) if lat is not None else 'n/a',
        'UPDATE ' if event.is_update else '',
        event.headline[:90]), flush=True)


def run_shadow(stats_interval_s: float = 60.0,
               listener: Optional[NewsStreamListener] = None,
               max_seconds: Optional[float] = None) -> None:
    """Run the listener solo, printing events live and a stats block every
    `stats_interval_s`. Ctrl-C (or max_seconds, for tests) exits cleanly."""
    own_listener = listener is None
    if own_listener:
        listener = NewsStreamListener()
    listener.register_callback(_shadow_print_event)
    listener.start()
    started = time.monotonic()
    print("[NEWS-STREAM] shadow mode — Ctrl-C to stop. Stats every %.0fs."
          % stats_interval_s, flush=True)
    try:
        while True:
            slept = 0.0
            while slept < stats_interval_s:
                step = min(1.0, stats_interval_s - slept)
                time.sleep(step)
                slept += step
                if max_seconds is not None and \
                        time.monotonic() - started >= max_seconds:
                    return
            stats = listener.latency_stats()
            stats['stale'] = listener.is_stale()
            print("[NEWS-STREAM] stats %s | %s" % (
                datetime.now(ET_TZ).strftime('%H:%M:%S ET'),
                json.dumps(stats, default=str)), flush=True)
    except KeyboardInterrupt:
        print("[NEWS-STREAM] shadow mode interrupted — shutting down",
              flush=True)
    finally:
        if own_listener:
            listener.stop()
        final = listener.latency_stats()
        print("[NEWS-STREAM] final stats: %s" % json.dumps(final, default=str),
              flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry: python3 -m data_sources.news_stream --shadow"""
    parser = argparse.ArgumentParser(
        description="Alpaca real-time news stream listener (standalone)")
    parser.add_argument('--shadow', action='store_true',
                        help='run the listener solo, printing events + '
                             'latency stats (Monday measurement mode)')
    parser.add_argument('--stats-interval', type=float, default=60.0,
                        help='seconds between stats blocks (default 60)')
    parser.add_argument('--max-seconds', type=float, default=None,
                        help='auto-exit after N seconds (smoke runs)')
    args = parser.parse_args(argv)
    if not args.shadow:
        parser.error("only --shadow mode is implemented (trader wiring is a "
                     "later task)")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.warning("python-dotenv not installed — relying on process env "
                       "for ALPACA_API_KEY/SECRET")
    run_shadow(stats_interval_s=args.stats_interval,
               max_seconds=args.max_seconds)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
