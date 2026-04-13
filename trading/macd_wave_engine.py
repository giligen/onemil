"""
MACD Wave Trading Engine.

Detects +10% intraday movers on $15-30 stocks, enters on MACD histogram
positive confirmation (3 bars), exits on histogram flip or 2% hard stop.
Wave 1 only — no W2/W3.

Runs as standalone service, shares Alpaca account and DB with bull flag engine.
"""

import logging
import time as time_mod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pytz
from dateutil.parser import isoparse

logger = logging.getLogger(__name__)

ET = pytz.timezone('US/Eastern')


@dataclass
class OpenPosition:
    """Tracked open position."""
    symbol: str
    entry_price: float
    shares: int
    hard_stop: float
    trade_id: int
    order_id: str
    entry_time: datetime
    macd_hist_at_entry: float = 0.0
    highest_since_entry: float = 0.0  # For trailing stop
    # Slippage attribution (set at entry submit, used at fill)
    bar_close_price: Optional[float] = None
    order_submitted_at: Optional[datetime] = None
    entry_quote_ask: Optional[float] = None


@dataclass
class CrossedStock:
    """Stock that crossed the +threshold% trigger."""
    symbol: str
    open_price: float
    cross_time_min: int
    vol_at_cross: int
    crossed_at: datetime
    bars_cache: Optional[pd.DataFrame] = None
    pos_count: int = 0  # Consecutive positive MACD bars


class MACDWaveEngine:
    """
    MACD Wave trading engine.

    Lifecycle:
    1. Pre-market: build_universe() — fetch $15-30 stocks
    2. Every 60s: scan_for_movers() → check_entries() → check_exits()
    3. 15:45 ET: force_close_all()
    4. EOD: send_daily_report()
    """

    STRATEGY_NAME = 'macd_wave'

    def __init__(
        self,
        alpaca_client,
        db,
        notifier=None,
        config: Optional[dict] = None,
        dry_run: bool = False,
        stop_monitor=None,
        order_stream=None,
    ):
        cfg = config or {}
        self.alpaca = alpaca_client
        self.db = db
        self.notifier = notifier
        self.stop_monitor = stop_monitor
        self.order_stream = order_stream  # T3.1: TradingStream watcher (optional)
        self.dry_run = dry_run
        # T1.1: bar events pushed here by StopMonitor's dispatched bar handler.
        # Main loop drains via drain_bar_events() for targeted check_entries.
        # Bounded to detect stuck main loops — at steady state this holds only
        # ~1-2 events (one per sec at most from the scanner's 1s tick), so any
        # actual backlog signals something is wrong.
        import queue as _q
        self._bar_event_queue: "_q.Queue" = _q.Queue(maxsize=1000)
        self._bar_handler_registered = False
        self._bar_queue_full_logged = False

        # Universe filters
        uni = cfg.get('universe', {})
        self.min_price = float(uni.get('min_price', 15.0))
        self.max_price = float(uni.get('max_price', 30.0))
        self.min_daily_volume = int(uni.get('min_daily_volume', 1_000_000))
        self.min_intraday_pct = float(uni.get('min_intraday_pct', 10.0))

        # Entry filters
        entry = cfg.get('entry', {})
        self.cross_time_max_min = int(entry.get('cross_time_max_min', 3))
        self.min_vol_at_cross = int(entry.get('min_vol_at_cross', 0))
        self.max_vol_at_cross = int(entry.get('max_vol_at_cross', 300_000))
        self.min_macd_hist_pct = float(entry.get('min_macd_hist_pct', 0.5))
        self.max_price_at_entry = float(entry.get('max_price_at_entry', 0))

        # Smart entry: L1-informed pricing + early entry on strong book
        self.smart_entry_enabled = bool(entry.get('smart_entry_enabled', False))
        self.early_entry_bars = int(entry.get('early_entry_bars', 1))
        self.early_bid_ask_ratio = float(entry.get('early_bid_ask_ratio', 2.0))
        self.early_max_spread_pct = float(entry.get('early_max_spread_pct', 0.003))
        self.normal_bid_ask_ratio = float(entry.get('normal_bid_ask_ratio', 1.5))
        self.normal_max_spread_pct = float(entry.get('normal_max_spread_pct', 0.005))

        # MACD params
        macd = cfg.get('macd', {})
        self.macd_fast = int(macd.get('fast_period', 12))
        self.macd_slow = int(macd.get('slow_period', 26))
        self.macd_signal = int(macd.get('signal_period', 9))
        self.confirm_bars = int(macd.get('confirm_bars', 3))

        # Sizing
        sizing = cfg.get('sizing', {})
        self.position_size = float(sizing.get('position_size', 50_000))
        self.max_concurrent = int(sizing.get('max_concurrent', 5))
        adv_pct = sizing.get('max_adv_participation_pct')
        self.max_adv_participation_pct = float(adv_pct) if adv_pct else None

        # Risk
        risk = cfg.get('risk', {})
        self.hard_stop_pct = float(risk.get('hard_stop_pct', 0.02))
        self.trail_stop_pct = float(risk.get('trail_stop_pct', 0.003))  # 0.3% trail below highest
        self.daily_loss_limit = float(risk.get('daily_loss_limit', -5000))
        self.safety_net_sl_pct = float(risk.get('safety_net_sl_pct', 0.05))  # 5% floor on Alpaca

        # Slippage (for logging comparison, not applied to orders)
        slip = cfg.get('slippage', {})
        self.entry_slippage_pct = float(slip.get('entry_pct', 0.001))
        self.exit_slippage_pct = float(slip.get('exit_pct', 0.001))

        # Force close time
        self.force_close_hour = 15
        self.force_close_minute = 45

        # State (reset daily)
        self.universe: List[str] = []
        self.universe_opens: Dict[str, float] = {}  # symbol → today's open price
        self.universe_volumes: Dict[str, int] = {}  # symbol → prev day volume (ADV proxy)
        self.crossed_stocks: Dict[str, CrossedStock] = {}
        self.open_positions: Dict[str, OpenPosition] = {}
        self.invalidated: Set[str] = set()
        self.daily_pnl: float = 0.0
        self.trades_today: int = 0
        self.shutdown_requested: bool = False

    # ------------------------------------------------------------------
    # Startup sync: recover state from DB + Alpaca
    # ------------------------------------------------------------------

    def sync_positions(self) -> None:
        """
        Reconcile in-memory state with DB and Alpaca.

        Called on startup AND every cycle. Handles:
        1. Startup recovery: repopulate open_positions from DB
        2. Closed position detection: if Alpaca no longer has a position
           that we think is open, it was closed externally (bracket SL/TP).
           Update DB with actual exit price from Alpaca order history.
        """
        today = date.today().isoformat()

        # Get open MACD wave trades from DB (filled, no exit)
        db_open = self.db.get_open_trades(today, strategy=self.STRATEGY_NAME)

        # Get actual Alpaca positions
        try:
            alpaca_positions = {
                p.symbol: p for p in self.alpaca.trading_client.get_all_positions()
            }
        except Exception as e:
            logger.error(f"[{self.STRATEGY_NAME}] sync_positions: failed to get Alpaca positions: {e}")
            return

        for trade in db_open:
            sym = trade['symbol']
            fill_price = trade.get('fill_price')
            if not fill_price:
                continue

            if sym in alpaca_positions:
                # Position still open on Alpaca — ensure we're tracking it
                if sym not in self.open_positions:
                    shares = trade.get('filled_qty') or trade['shares']
                    hard_stop = round(fill_price * (1 - self.hard_stop_pct), 2)
                    self.open_positions[sym] = OpenPosition(
                        symbol=sym,
                        entry_price=fill_price,
                        shares=shares,
                        hard_stop=hard_stop,
                        trade_id=trade['id'],
                        order_id='',
                        entry_time=datetime.now(timezone.utc),
                        macd_hist_at_entry=0,
                        highest_since_entry=fill_price,
                    )
                    # Re-register with StopMonitor
                    if self.stop_monitor:
                        self.stop_monitor.add_watch(
                            symbol=sym, stop_price=hard_stop,
                            shares=shares, tp_leg_id='', sl_leg_id='',
                            trade_db_id=trade['id'],
                            entry_price=fill_price,
                            risk_per_share=0,
                            trail_r=0, activate_at_r=0.0,
                            trail_pct=self.trail_stop_pct,
                            strategy=self.STRATEGY_NAME,
                        )
                    logger.info(
                        f"[{self.STRATEGY_NAME}] sync: recovered {sym} "
                        f"({shares}sh @ ${fill_price:.2f})"
                    )
            else:
                # Position GONE from Alpaca — closed externally (bracket SL/TP)
                # Find the actual exit from Alpaca order history
                exit_price = fill_price  # fallback
                exit_reason = 'bracket_exit'
                try:
                    from alpaca.trading.requests import GetOrdersRequest
                    from alpaca.trading.enums import QueryOrderStatus
                    orders = self.alpaca.trading_client.get_orders(
                        GetOrdersRequest(
                            status=QueryOrderStatus.CLOSED,
                            symbols=[sym], limit=5,
                        )
                    )
                    for o in orders:
                        if (o.side.value == 'sell' and o.status.value == 'filled'
                                and o.filled_avg_price):
                            exit_price = float(o.filled_avg_price)
                            if o.order_class and o.order_class.value == 'bracket':
                                exit_reason = 'bracket_sl_tp'
                            break
                except Exception:
                    pass

                pnl = (exit_price - fill_price) * (trade.get('filled_qty') or trade['shares'])
                pnl_pct = (exit_price - fill_price) / fill_price * 100 if fill_price > 0 else 0
                self.daily_pnl += pnl

                self.db.update_trade(trade['id'], {
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'exited_at': datetime.now(timezone.utc),
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                })

                # Clean up in-memory if present
                if sym in self.open_positions:
                    del self.open_positions[sym]
                self.invalidated.add(sym)

                logger.info(
                    f"[{self.STRATEGY_NAME}] sync: {sym} closed externally — "
                    f"exit ${exit_price:.2f}, P&L ${pnl:+,.0f} ({exit_reason})"
                )

    # ------------------------------------------------------------------
    # Pre-market: build universe
    # ------------------------------------------------------------------

    def build_universe(self) -> int:
        """
        Build daily universe of $15-30 stocks from Alpaca.

        Called ~1h before market open. Fetches snapshots to get latest prices.
        Returns number of stocks in universe.
        """
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass, AssetStatus

        logger.info(f"[{self.STRATEGY_NAME}] Building universe (${self.min_price}-${self.max_price})...")

        assets = self.alpaca.trading_client.get_all_assets(
            GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        )
        # Filter tradeable stocks in price range
        candidates = [
            a.symbol for a in assets
            if a.tradable
            and a.exchange in ('NYSE', 'NASDAQ', 'AMEX', 'ARCA', 'BATS')
        ]

        # Get snapshots to check price range + previous day volume
        self.universe = []
        vol_filtered = 0
        chunk_size = 200
        for i in range(0, len(candidates), chunk_size):
            chunk = candidates[i:i + chunk_size]
            try:
                snapshots = self.alpaca.get_snapshots(chunk)
                for sym, snap in snapshots.items():
                    price = snap.get('latest_price', 0) or snap.get('close', 0)
                    if not (self.min_price <= price <= self.max_price):
                        continue
                    prev_vol = snap.get('prev_volume', 0)
                    if self.min_daily_volume > 0 and prev_vol < self.min_daily_volume:
                        vol_filtered += 1
                        continue
                    self.universe.append(sym)
                    if prev_vol > 0:
                        self.universe_volumes[sym] = prev_vol
            except Exception as e:
                logger.warning(f"[{self.STRATEGY_NAME}] Snapshot chunk failed: {e}")

        logger.info(
            f"[{self.STRATEGY_NAME}] Universe built: {len(self.universe)} stocks "
            f"(${self.min_price}-${self.max_price}, vol>={self.min_daily_volume:,}, "
            f"{vol_filtered} filtered by volume)"
        )

        if self.notifier:
            self.notifier.send_message_sync(
                f"[MACD Wave] Service started — universe: {len(self.universe)} stocks "
                f"(${self.min_price:.0f}-${self.max_price:.0f})"
            )

        return len(self.universe)

    # ------------------------------------------------------------------
    # T1.1: WebSocket-bar-event integration (unified main loop drives this)
    # ------------------------------------------------------------------

    def register_on_stop_monitor(self) -> None:
        """
        Register a bar handler with the shared StopMonitor so 1-min bar closes
        for our crossed_stocks flow into self._bar_event_queue. Called once by
        the unified service after StopMonitor.start().
        """
        if self.stop_monitor is None:
            return
        if self._bar_handler_registered:
            return

        import queue as _q_mod

        def _on_bar(symbol: str, bars_df) -> None:
            # Cheap enqueue; main loop does the actual MACD re-eval.
            try:
                self._bar_event_queue.put_nowait(symbol)
                if self._bar_queue_full_logged:
                    # Recovered from a previous Full — log recovery and reset.
                    logger.info(
                        f"[{self.STRATEGY_NAME}] bar event queue recovered "
                        f"(size={self._bar_event_queue.qsize()})"
                    )
                    self._bar_queue_full_logged = False
            except _q_mod.Full:
                # Main loop is stuck or too slow — drop the event rather than
                # blocking the WS thread. Log once per blockage; recovery logs
                # again when space opens up.
                if not self._bar_queue_full_logged:
                    logger.error(
                        f"[{self.STRATEGY_NAME}] bar event queue FULL "
                        f"(maxsize=1000) — main loop likely stuck; dropping {symbol}"
                    )
                    self._bar_queue_full_logged = True
            except Exception as e:
                logger.error(f"[{self.STRATEGY_NAME}] bar event enqueue error: {e}")

        self.stop_monitor.register_bar_handler(self.STRATEGY_NAME, _on_bar)
        self._bar_handler_registered = True
        logger.info(f"[{self.STRATEGY_NAME}] bar handler registered on StopMonitor")

    def drain_bar_events(self) -> Set[str]:
        """Pop all queued bar events and return the unique set of symbols."""
        symbols: Set[str] = set()
        while True:
            try:
                sym = self._bar_event_queue.get_nowait()
            except Exception:
                break
            symbols.add(sym)
        return symbols

    # ------------------------------------------------------------------
    # Intraday: scan for +10% movers
    # ------------------------------------------------------------------

    def scan_for_movers(self) -> List[str]:
        """
        Scan universe for stocks crossing +threshold% from today's open.

        Called every 60s. Detects new crosses, applies filters.
        Returns list of newly crossed symbols.
        """
        if not self.universe:
            return []

        now_et = datetime.now(ET)
        market_open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        minutes_since_open = max(0, int((now_et - market_open_et).total_seconds() / 60))

        new_crosses = []

        # First cycle: fetch real open prices from Alpaca snapshots
        if not self.universe_opens:
            symbols_needing_opens = [s for s in self.universe
                                     if s not in self.crossed_stocks and s not in self.invalidated]
            logger.info(
                f"[{self.STRATEGY_NAME}] Recording open prices for {len(symbols_needing_opens)} symbols via snapshots..."
            )
            chunk_size = 200
            for i in range(0, len(symbols_needing_opens), chunk_size):
                chunk = symbols_needing_opens[i:i + chunk_size]
                try:
                    snapshots = self.alpaca.get_snapshots(chunk)
                    for sym, snap in snapshots.items():
                        open_price = snap.get('open', 0)
                        prev_close = snap.get('prev_close', 0)
                        # Reverse-split detection: >100% jump overnight = corporate
                        # action (2:1 split or bigger). % move calcs would be meaningless.
                        if prev_close > 0 and open_price > 0:
                            jump_ratio = abs(open_price - prev_close) / prev_close
                            if jump_ratio > 1.0:
                                self.invalidated.add(sym)
                                logger.info(
                                    f"[{self.STRATEGY_NAME}] {sym}: split detected "
                                    f"(prev_close ${prev_close:.2f} → open ${open_price:.2f}, "
                                    f"{jump_ratio * 100:.0f}% jump) — invalidating"
                                )
                                continue
                        if open_price > 0:
                            self.universe_opens[sym] = open_price
                except Exception as e:
                    logger.warning(f"[{self.STRATEGY_NAME}] Snapshot chunk failed: {e}")
            logger.info(
                f"[{self.STRATEGY_NAME}] Open prices recorded: {len(self.universe_opens)} stocks"
            )
            return new_crosses  # First cycle just records opens — scan on next cycle

        # Subsequent cycles: fetch latest trades and compare to recorded opens
        chunk_size = 200
        for i in range(0, len(self.universe), chunk_size):
            chunk = self.universe[i:i + chunk_size]
            try:
                trades = self.alpaca.get_latest_trades(chunk)
                for sym, trade_data in trades.items():
                    if sym in self.crossed_stocks or sym in self.invalidated:
                        continue
                    if sym in self.open_positions:
                        continue  # Already have position or pending order for this symbol

                    price = trade_data.get('price', 0)

                    # Skip stale trades from before today's open
                    trade_ts = trade_data.get('timestamp')
                    if trade_ts:
                        trade_dt = isoparse(trade_ts)
                        if trade_dt.astimezone(ET) < market_open_et:
                            continue

                    open_price = self.universe_opens.get(sym)
                    if open_price is None or open_price <= 0:
                        continue

                    if open_price <= 0:
                        continue

                    # Check if crossed threshold
                    pct_change = (price - open_price) / open_price * 100
                    if pct_change < self.min_intraday_pct:
                        continue

                    # Apply cross time filter
                    if self.cross_time_max_min > 0 and minutes_since_open > self.cross_time_max_min:
                        continue

                    # Get volume at cross (approximate from bars)
                    vol_at_cross = 0
                    try:
                        bars = self.alpaca.get_1min_bars(sym, lookback_minutes=minutes_since_open + 1)
                        if bars is not None and not bars.empty:
                            vol_at_cross = int(bars['volume'].sum())
                    except Exception:
                        pass

                    # Apply volume filter
                    if self.max_vol_at_cross > 0 and vol_at_cross > self.max_vol_at_cross:
                        logger.debug(f"[{self.STRATEGY_NAME}] {sym}: vol {vol_at_cross:,} > {self.max_vol_at_cross:,}, skipping")
                        self.invalidated.add(sym)
                        continue
                    if self.min_vol_at_cross > 0 and vol_at_cross < self.min_vol_at_cross:
                        continue

                    # Passed all filters — start monitoring
                    self.crossed_stocks[sym] = CrossedStock(
                        symbol=sym,
                        open_price=open_price,
                        cross_time_min=minutes_since_open,
                        vol_at_cross=vol_at_cross,
                        crossed_at=datetime.now(timezone.utc),
                    )
                    new_crosses.append(sym)
                    # T1.1: subscribe to 1-min bars so the unified main loop
                    # can react within seconds of each bar close instead of
                    # waiting for the 60s polling sweep.
                    if self.stop_monitor is not None:
                        try:
                            self.stop_monitor.subscribe_bars(sym)
                        except Exception as e:
                            logger.debug(
                                f"[{self.STRATEGY_NAME}] {sym}: subscribe_bars failed: {e}"
                            )

                    logger.info(
                        f"[{self.STRATEGY_NAME}] {sym}: crossed +{pct_change:.1f}% "
                        f"in {minutes_since_open}min, vol {vol_at_cross:,} — monitoring MACD"
                    )
                    if self.notifier:
                        self.notifier.send_message_sync(
                            f"[MACD Wave] 🔍 {sym} crossed +{pct_change:.1f}% "
                            f"in {minutes_since_open}min, vol {vol_at_cross:,} — monitoring MACD"
                        )

            except Exception as e:
                logger.warning(f"[{self.STRATEGY_NAME}] Mover scan chunk failed: {e}")

        return new_crosses

    # ------------------------------------------------------------------
    # Intraday: check entries (MACD confirmation)
    # ------------------------------------------------------------------

    def _has_conflicting_alpaca_orders(self, symbol: str) -> bool:
        """
        True if Alpaca shows an existing OPEN order for this symbol (any strategy).

        Prevents wash trades in the unified process where both strategies run
        under one Alpaca account. Two code paths:

        FAST: if OrderStreamWatcher is attached AND healthy, read from its
              push-updated in-memory set. O(1). No network.
        SLOW: REST `get_orders(filter=req)`. 200-400ms. Used as fallback when
              the stream is unavailable or unhealthy.

        Returns False on any failure (fail-open — Alpaca still rejects on
        submit, we just lose the pre-emptive log).
        """
        # Fast path via OrderStreamWatcher's push-updated cache.
        if self.order_stream is not None and self.order_stream.is_healthy():
            try:
                open_symbols = self.order_stream.get_open_order_symbols()
                if symbol in open_symbols:
                    logger.warning(
                        f"[{self.STRATEGY_NAME}] {symbol}: conflicting open order "
                        f"(detected via order-stream cache)"
                    )
                    return True
                return False
            except Exception as e:
                logger.warning(
                    f"[{self.STRATEGY_NAME}] {symbol}: order-stream check failed, "
                    f"falling back to REST: {e}"
                )
                # fall through to REST

        # Slow path: REST (stream absent or unhealthy).
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            existing = self.alpaca.trading_client.get_orders(filter=req)
            if existing:
                sides = [getattr(o.side, 'value', str(o.side)) for o in existing]
                logger.warning(
                    f"[{self.STRATEGY_NAME}] {symbol}: "
                    f"{len(existing)} conflicting open order(s) already "
                    f"({', '.join(sides)})"
                )
                return True
        except Exception as e:
            logger.warning(
                f"[{self.STRATEGY_NAME}] {symbol}: conflict check failed: {e}"
            )
        return False

    def _gc_stale_pending(self) -> None:
        """
        Remove open_positions whose buy order has been pending >2 minutes
        (likely rejected by Alpaca, or the submit result was lost). Mutates
        self.open_positions and self.invalidated. Safe to call multiple times.
        """
        now = datetime.now(timezone.utc)
        stale = [
            psym for psym, pos in self.open_positions.items()
            if pos.order_id and (now - pos.entry_time).total_seconds() >= 120
        ]
        for psym in stale:
            logger.warning(
                f"[{self.STRATEGY_NAME}] {psym}: stale pending order (>2min), removing"
            )
            self.open_positions.pop(psym, None)
            self.invalidated.add(psym)

    def _has_entry_capacity(self) -> bool:
        """
        Pure predicate: True if we can still submit a new entry (under
        max_concurrent AND above daily_loss_limit). No side effects — call
        _gc_stale_pending() separately if you want stale-pending cleanup.
        """
        if self.daily_loss_limit < 0 and self.daily_pnl <= self.daily_loss_limit:
            return False
        now = datetime.now(timezone.utc)
        active = sum(
            1 for pos in self.open_positions.values()
            if not pos.order_id or (now - pos.entry_time).total_seconds() < 120
        )
        return active < self.max_concurrent

    def check_entries(self, symbols: Optional[Iterable[str]] = None) -> List[str]:
        """
        Check crossed stocks for MACD entry signal.

        For each crossed stock, compute MACD histogram on 1-min bars.
        Enter when confirm_bars consecutive positive bars with hist >= threshold.

        Args:
            symbols: optional subset of symbols to evaluate (used by bar-event
                     triggered targeted re-eval). If None, evaluate all
                     crossed_stocks (legacy behavior).
        """
        entries: List[str] = []

        # Gate 1: capacity + daily-loss limit — check ONCE before any I/O so
        # we don't burn a batch REST call when we have no slots to use.
        # GC stale pending orders first (frees up slots that have gone dead).
        self._gc_stale_pending()
        if not self._has_entry_capacity():
            return entries

        # Gate 2: build the effective watchlist (exclude positioned/invalidated)
        # and use it directly as the iteration target — no double-filtering.
        if symbols is not None:
            candidate_iter = list(symbols)
        else:
            candidate_iter = list(self.crossed_stocks.keys())
        watchlist = [
            s for s in candidate_iter
            if s in self.crossed_stocks
            and s not in self.open_positions
            and s not in self.invalidated
        ]
        if not watchlist:
            return entries

        # T1.2: batch-fetch bars for the watchlist in one REST call instead of
        # N serial calls. `loop_processed_at` captured per-symbol below now
        # includes only the thin loop overhead (not per-symbol RTT) for
        # batched symbols — expect bar_close_to_loop_ms medians to shrink by
        # ~(N-1) * 300ms after this lands. For symbols missed by the batch
        # (empty response), a per-symbol get_1min_bars RTT is still billed.
        now_et_outer = datetime.now(ET)
        mins_outer = max(30, int((now_et_outer - now_et_outer.replace(
            hour=9, minute=30, second=0)).total_seconds() / 60))
        bars_by_sym: Dict[str, pd.DataFrame] = {}
        try:
            bars_by_sym = self.alpaca.get_1min_bars_multi(
                watchlist, lookback_minutes=mins_outer
            )
        except Exception as e:
            logger.warning(
                f"[{self.STRATEGY_NAME}] batch bar fetch failed "
                f"({len(watchlist)} syms), falling back per-symbol: {e}"
            )

        for sym in watchlist:
            crossed = self.crossed_stocks.get(sym)
            if crossed is None:  # defensive: races with scan_for_movers
                continue
            # Re-check capacity inside the loop — we may enter symbols as we go,
            # and subsequent iterations must honor the updated count.
            if not self._has_entry_capacity():
                break

            try:
                # T1.2: look up from the batched fetch; fall back per-symbol on miss.
                bars = bars_by_sym.get(sym)
                if bars is None or bars.empty:
                    bars = self.alpaca.get_1min_bars(sym, lookback_minutes=mins_outer)
                # Slippage instrumentation: timestamp right after bars are in hand.
                # Captures polling interval + bar-fetch RTT (the "wait" component of drift).
                loop_processed_at = datetime.now(timezone.utc)
                if bars is None or len(bars) < self.macd_slow + self.macd_signal:
                    logger.info(
                        f"[{self.STRATEGY_NAME}] {sym}: insufficient bars "
                        f"({len(bars) if bars is not None else 0}, need {self.macd_slow + self.macd_signal})"
                    )
                    continue

                # Compute MACD
                close = bars['close']
                ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
                ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
                histogram = macd_line - signal_line

                # Count consecutive positive histogram bars from the end
                latest_price = close.iloc[-1]
                latest_hist = histogram.iloc[-1]
                hist_pct = latest_hist / latest_price * 100 if latest_price > 0 else 0

                # Reference price/time for slippage attribution: last 1-min bar
                # seen at signal time (the BT's ideal entry reference).
                try:
                    bar_close_at_raw = bars['timestamp'].iloc[-1]
                    if isinstance(bar_close_at_raw, pd.Timestamp):
                        bar_close_at = bar_close_at_raw.to_pydatetime()
                    elif isinstance(bar_close_at_raw, datetime):
                        bar_close_at = bar_close_at_raw
                    else:
                        bar_close_at = pd.to_datetime(bar_close_at_raw, utc=True).to_pydatetime()
                    if bar_close_at.tzinfo is None:
                        bar_close_at = bar_close_at.replace(tzinfo=timezone.utc)
                    bar_close_price = float(latest_price)
                except Exception:
                    bar_close_at = None
                    bar_close_price = None

                consecutive_pos = 0
                for h in reversed(histogram.values):
                    if h > 0:
                        consecutive_pos += 1
                    else:
                        break
                crossed.pos_count = consecutive_pos

                logger.info(
                    f"[{self.STRATEGY_NAME}] {sym}: MACD check — "
                    f"pos_count={consecutive_pos}, hist_pct={hist_pct:.2f}%, "
                    f"price=${latest_price:.2f}, bars={len(bars)}"
                )

                # Entry filters (applied at all entry levels)
                if self.min_macd_hist_pct > 0 and hist_pct < self.min_macd_hist_pct:
                    if crossed.pos_count >= self.confirm_bars:
                        logger.info(
                            f"[{self.STRATEGY_NAME}] {sym}: MACD hist too weak "
                            f"({hist_pct:.2f}% < {self.min_macd_hist_pct}%), resetting"
                        )
                        crossed.pos_count = 0
                    continue

                if self.max_price_at_entry > 0 and latest_price > self.max_price_at_entry:
                    continue

                # Smart early entry: check L1 book at 1st and 2nd bar
                entered = False
                # Smart entry: get quote once, reuse for both early entry check and submission
                smart_quote = None
                if self.smart_entry_enabled:
                    try:
                        smart_quote = self._get_smart_limit_price(sym)
                    except Exception as e:
                        logger.debug(f"[{self.STRATEGY_NAME}] {sym}: smart pricing failed: {e}")

                pipeline_timing = {
                    'loop_processed_at': loop_processed_at,
                    'bar_close_at': bar_close_at,
                    'bar_close_price': bar_close_price,
                }

                if self.smart_entry_enabled and smart_quote and crossed.pos_count < self.confirm_bars:
                    _, info = smart_quote
                    ba_ratio = info.get('ba_ratio', 0)
                    spread_pct = info.get('spread_pct', 1)

                    if crossed.pos_count >= self.early_entry_bars:
                        # Early entry: strong book required
                        if ba_ratio >= self.early_bid_ask_ratio and spread_pct <= self.early_max_spread_pct:
                            logger.info(
                                f"[{self.STRATEGY_NAME}] {sym}: EARLY ENTRY (bar {crossed.pos_count}) — "
                                f"ba_ratio={ba_ratio:.1f} spread={spread_pct:.2%}"
                            )
                            result = self._submit_entry(sym, latest_price, hist_pct, crossed,
                                                        smart_quote=smart_quote,
                                                        pipeline_timing=pipeline_timing)
                            if result:
                                entries.append(sym)
                                entered = True
                        elif crossed.pos_count >= 2 and ba_ratio >= self.normal_bid_ask_ratio and spread_pct <= self.normal_max_spread_pct:
                            logger.info(
                                f"[{self.STRATEGY_NAME}] {sym}: ENTRY (bar 2, moderate book) — "
                                f"ba_ratio={ba_ratio:.1f} spread={spread_pct:.2%}"
                            )
                            result = self._submit_entry(sym, latest_price, hist_pct, crossed,
                                                        smart_quote=smart_quote,
                                                        pipeline_timing=pipeline_timing)
                            if result:
                                entries.append(sym)
                                entered = True

                # Standard entry: 3rd bar (always, with smart pricing if enabled)
                if not entered and crossed.pos_count >= self.confirm_bars:
                    logger.info(
                        f"[{self.STRATEGY_NAME}] {sym}: ENTRY SIGNAL (bar {consecutive_pos}) — "
                        f"hist_pct={hist_pct:.2f}%, price=${latest_price:.2f}"
                    )
                    result = self._submit_entry(sym, latest_price, hist_pct, crossed,
                                                smart_quote=smart_quote,
                                                pipeline_timing=pipeline_timing)
                    if result:
                        entries.append(sym)

            except Exception as e:
                logger.error(f"[{self.STRATEGY_NAME}] {sym}: entry check failed: {e}")

        return entries

    def _submit_entry(
        self, symbol: str, price: float, macd_hist_pct: float, crossed: CrossedStock,
        smart_quote: tuple = None,
        pipeline_timing: dict = None,
    ) -> bool:
        """Submit a buy order for MACD wave entry.

        Args:
            smart_quote: Optional (limit_price, quote_info) from _get_smart_limit_price,
                         passed to avoid double API call.
            pipeline_timing: dict with loop_processed_at, bar_close_at, bar_close_price
                             for slippage attribution. Optional.
        """
        try:
            # Get limit price: smart (L1-informed) or dumb (ask + 0.1%)
            quote_fetched_at = None
            if smart_quote:
                limit_price, quote_info = smart_quote
                bid = quote_info.get('bid', 0)
                ask = quote_info.get('ask', 0)
                quote_fetched_at = quote_info.get('quote_fetched_at')
            elif self.smart_entry_enabled:
                limit_price, quote_info = self._get_smart_limit_price(symbol)
                bid = quote_info.get('bid', 0)
                ask = quote_info.get('ask', 0)
                quote_fetched_at = quote_info.get('quote_fetched_at')
            else:
                quote = self.alpaca.get_latest_quote(symbol)
                quote_fetched_at = datetime.now(timezone.utc)
                ask = quote.get('ask_price', 0)
                bid = quote.get('bid_price', 0)
                if ask <= 0:
                    ask = price * 1.001
                limit_price = round(ask * 1.001, 2)  # ask + 0.1%

            if limit_price <= 0:
                logger.warning(f"[{self.STRATEGY_NAME}] {symbol}: limit_price=0, skipping")
                return False
            shares = int(self.position_size / limit_price)
            if shares <= 0:
                return False

            hard_stop = round(limit_price * (1 - self.hard_stop_pct), 2)

            if self.dry_run:
                logger.info(
                    f"[{self.STRATEGY_NAME}] DRY RUN: would BUY {symbol} "
                    f"{shares}sh @ ${limit_price:.2f}, stop ${hard_stop:.2f}, "
                    f"MACD hist {macd_hist_pct:.2f}%"
                )
                self.invalidated.add(symbol)
                return False

            # Wash-trade pre-check: if bull flag (or any other strategy) already
            # has an open order for this symbol on Alpaca, submitting ours would
            # be rejected as a wash trade. Mirrors OrderExecutor._has_conflicting_orders
            # (trading/order_executor.py:44). Cheap REST call; only runs when we
            # actually have a signal to submit.
            if self._has_conflicting_alpaca_orders(symbol):
                logger.warning(
                    f"[{self.STRATEGY_NAME}] {symbol}: skipping entry — "
                    f"another strategy has an open order (would be wash trade)"
                )
                self.invalidated.add(symbol)
                return False

            # Submit bracket buy with safety-net SL on Alpaca
            # If everything dies (StopMonitor + service), Alpaca still has a 5% floor
            safety_sl = round(limit_price * (1 - self.safety_net_sl_pct), 2)
            safety_tp = round(limit_price * 1.50, 2)  # effectively infinite
            order = self.alpaca.submit_bracket_order(
                symbol=symbol, qty=shares, side='buy',
                limit_price=limit_price,
                tp_price=safety_tp, sl_price=safety_sl,
            )
            order_submitted_at = datetime.now(timezone.utc)
            order_id = order.get('id', '') if order else ''

            # Save to DB
            trade_id = self.db.save_trade({
                'strategy': self.STRATEGY_NAME,
                'trade_date': date.today().isoformat(),
                'symbol': symbol,
                'side': 'buy',
                'entry_price': limit_price,
                'stop_loss_price': hard_stop,
                'take_profit_price': 0,  # No fixed TP — MACD exit
                'shares': shares,
                'risk_per_share': limit_price * self.hard_stop_pct,
                'total_risk': limit_price * self.hard_stop_pct * shares,
                'risk_reward_ratio': 0,  # N/A for MACD
                'order_id': order_id,
                'order_status': 'pending_new',
                'fill_price': None,
                'filled_at': None,
                'exit_price': None,
                'exit_reason': None,
                'exited_at': None,
                'pnl': None,
                'pnl_pct': None,
                'pattern_data': f'{{"cross_time_min": {crossed.cross_time_min}, '
                                f'"vol_at_cross": {crossed.vol_at_cross}, '
                                f'"macd_hist_pct": {macd_hist_pct:.4f}}}',
            })

            # Entry microstructure + slippage pipeline timing (Migration 10).
            # Decomposes the "bar close → order at Alpaca" latency so we can
            # see where drift is eaten: wait, MACD compute, quote RTT, submit RTT.
            micro = {
                'entry_quote_bid': bid,
                'entry_quote_ask': ask,
                'entry_quote_spread': ask - bid if ask > 0 and bid > 0 else None,
            }
            pt = pipeline_timing or {}
            bar_close_at = pt.get('bar_close_at')
            loop_processed_at = pt.get('loop_processed_at')
            bar_close_price = pt.get('bar_close_price')

            def _ms(a, b):
                if a is None or b is None:
                    return None
                return int((b - a).total_seconds() * 1000)

            if bar_close_price:
                micro['bar_close_price'] = bar_close_price
            if bar_close_at:
                micro['bar_close_at'] = bar_close_at
            if loop_processed_at:
                micro['loop_processed_at'] = loop_processed_at
            if quote_fetched_at:
                micro['quote_fetched_at'] = quote_fetched_at
            micro['order_submitted_at'] = order_submitted_at

            micro['bar_close_to_loop_ms'] = _ms(bar_close_at, loop_processed_at)
            micro['loop_to_quote_ms'] = _ms(loop_processed_at, quote_fetched_at)
            micro['quote_to_submit_ms'] = _ms(quote_fetched_at, order_submitted_at)

            if bar_close_price and bar_close_price > 0 and ask > 0:
                micro['drift_bar_to_ask_bps'] = (ask - bar_close_price) / bar_close_price * 10000

            self.db.update_trade(trade_id, micro)

            self.open_positions[symbol] = OpenPosition(
                symbol=symbol,
                entry_price=limit_price,
                shares=shares,
                hard_stop=hard_stop,
                trade_id=trade_id,
                order_id=order_id,
                entry_time=datetime.now(timezone.utc),
                macd_hist_at_entry=macd_hist_pct,
                highest_since_entry=limit_price,
                bar_close_price=bar_close_price,
                order_submitted_at=order_submitted_at,
                entry_quote_ask=ask if ask > 0 else None,
            )
            self.trades_today += 1

            logger.info(
                f"[{self.STRATEGY_NAME}] BUY {symbol} {shares}sh @ ${limit_price:.2f} "
                f"— MACD hist {macd_hist_pct:.2f}%, stop ${hard_stop:.2f}, "
                f"cross {crossed.cross_time_min}min, vol {crossed.vol_at_cross:,}"
            )
            if self.notifier:
                self.notifier.send_message_sync(
                    f"[MACD Wave] 📈 BUY {symbol} ${limit_price:.2f} × {shares}sh "
                    f"(${self.position_size:,.0f}) — MACD hist {macd_hist_pct:.2f}%, "
                    f"hard stop ${hard_stop:.2f}, trail {self.trail_stop_pct*100:.1f}%"
                )

            # NOTE: StopMonitor watch added AFTER fill confirmation in check_exits(),
            # NOT here. Adding before fill would let StopMonitor sell shares we
            # don't own yet → naked short position.

            return True

        except Exception as e:
            logger.error(f"[{self.STRATEGY_NAME}] {symbol}: entry submission failed: {e}")
            return False

    def _get_smart_limit_price(self, symbol: str) -> tuple:
        """
        Get L1-informed limit price using bid/ask spread.

        Returns (limit_price, quote_info_dict). info dict carries
        quote_fetched_at (UTC datetime) for slippage-pipeline timing.
        """
        quote = self.alpaca.get_latest_quote(symbol)
        quote_fetched_at = datetime.now(timezone.utc)
        bid = quote.get('bid_price', 0)
        ask = quote.get('ask_price', 0)
        bid_sz = quote.get('bid_size', 0)
        ask_sz = quote.get('ask_size', 0)

        if ask <= 0 or bid <= 0:
            fallback = round(ask * 1.001, 2) if ask > 0 else 0
            return fallback, {
                'bid': bid, 'ask': ask, 'pricing': 'fallback_no_quote',
                'quote_fetched_at': quote_fetched_at,
            }

        if bid >= ask:
            # Crossed/inverted market — stale quotes, use ask as-is
            logger.debug(f"[{self.STRATEGY_NAME}] {symbol}: crossed market bid=${bid} >= ask=${ask}, using ask")
            return round(ask, 2), {
                'bid': bid, 'ask': ask, 'pricing': 'fallback_crossed',
                'quote_fetched_at': quote_fetched_at,
            }

        spread = ask - bid
        spread_pct = spread / ask if ask > 0 else 0
        ba_ratio = bid_sz / ask_sz if ask_sz > 0 else 0

        # Spread-aware pricing: tighter spread → more aggressive limit
        if spread_pct < 0.0015:        # < 0.15% — very tight
            limit = ask                 # buy at ask, no premium needed
        elif spread_pct < 0.005:       # < 0.5% — moderate
            limit = bid + spread * 0.75 # 75% into the spread
        else:                           # wide spread
            limit = bid + spread * 0.60 # 60% into the spread

        limit = min(round(limit, 2), ask)

        info = {
            'bid': bid, 'ask': ask, 'bid_sz': bid_sz, 'ask_sz': ask_sz,
            'spread': spread, 'spread_pct': spread_pct, 'ba_ratio': ba_ratio,
            'pricing': 'smart',
            'quote_fetched_at': quote_fetched_at,
        }
        logger.info(
            f"[{self.STRATEGY_NAME}] {symbol}: smart pricing — "
            f"bid=${bid:.2f}×{bid_sz} ask=${ask:.2f}×{ask_sz} "
            f"spread={spread_pct:.2%} ba_ratio={ba_ratio:.1f} → limit=${limit:.2f}"
        )
        return limit, info

    # ------------------------------------------------------------------
    # Intraday: check exits
    # ------------------------------------------------------------------

    def check_exits(self) -> List[str]:
        """
        Check open positions for exit signals.

        Two exit paths:
        1. StopMonitor (real-time via SIP WebSocket): trail stop + hard stop
        2. Polling (60s via 1-min bars): MACD histogram flip

        StopMonitor events are drained EVERY cycle, even if no open positions
        (events could arrive between cycles).
        """
        exits = []

        # 0. Sync positions with Alpaca (detect external closes, recover after restart)
        self.sync_positions()

        # 1. Drain StopMonitor exit events (real-time trail/hard stop exits)
        if self.stop_monitor:
            for event in self.stop_monitor.drain_exit_events(strategy=self.STRATEGY_NAME):
                sym = event.symbol
                pos = self.open_positions.get(sym)
                if not pos:
                    logger.warning(f"[{self.STRATEGY_NAME}] {sym}: StopMonitor exit but no position")
                    continue

                exit_price = event.exit_price
                pnl = (exit_price - pos.entry_price) * pos.shares
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0
                self.daily_pnl += pnl

                self.db.update_trade(pos.trade_id, {
                    'exit_price': exit_price,
                    'exit_reason': event.exit_reason,
                    'exited_at': datetime.now(timezone.utc),
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_trigger_price': event.exit_trigger_price,
                    'exit_quote_bid': event.exit_quote_bid,
                    'exit_quote_ask': event.exit_quote_ask,
                    'exit_limit_price': event.exit_limit_price,
                    'exit_pricing_method': event.pricing_method,
                })

                emoji = '✅' if pnl > 0 else '❌'
                logger.info(
                    f"[{self.STRATEGY_NAME}] {emoji} EXIT {sym} ${exit_price:.2f} "
                    f"{pnl_pct:+.1f}% (${pnl:+,.0f}) — {event.exit_reason} (StopMonitor)"
                )
                if self.notifier:
                    self.notifier.send_message_sync(
                        f"[MACD Wave] {emoji} SELL {sym} ${exit_price:.2f} "
                        f"{pnl_pct:+.1f}% (${pnl:+,.0f}) — {event.exit_reason}"
                    )

                # Log exit L2 async
                try:
                    from data_sources.l2_depth import log_l2_async
                    trigger_dt = (datetime.fromtimestamp(event.submitted_at, tz=timezone.utc)
                                  if hasattr(event, 'submitted_at') and event.submitted_at > 0
                                  else datetime.now(timezone.utc))
                    log_l2_async(sym, trigger_dt, pos.trade_id,
                                 self.db.update_trade, column='exit_l2_depth')
                except Exception as e:
                    logger.debug(f"[{self.STRATEGY_NAME}] {sym}: Exit L2 async failed: {e}")

                del self.open_positions[sym]
                self.invalidated.add(sym)
                exits.append(sym)

        # 2. Check pending order fills + MACD flip on remaining positions
        for sym, pos in list(self.open_positions.items()):
            if sym in exits:
                continue  # Already exited by StopMonitor this cycle

            try:
                # Check if entry order filled
                if pos.order_id:
                    try:
                        # T3.1: prefer push-delivered status from TradingStream;
                        # fall back to REST if stream is silent and the order has aged.
                        order_status = None
                        if self.order_stream is not None:
                            order_status = self.order_stream.get_status(pos.order_id)
                        if order_status is None:
                            age_s = 0.0
                            if pos.order_submitted_at:
                                age_s = (datetime.now(timezone.utc) - pos.order_submitted_at).total_seconds()
                            # Always fall back to REST if the stream has nothing yet
                            # past the first few seconds after submit. First fill is
                            # often 100-300 ms, so don't wait too long.
                            if age_s >= 5.0 or self.order_stream is None:
                                order_status = self.alpaca.get_order(pos.order_id)
                        if order_status is None:
                            continue  # stream empty, too-fresh-for-REST — try next tick
                        status = order_status.get('status', '')
                        if status == 'filled':
                            fill_price = order_status.get('filled_avg_price')
                            filled_qty = order_status.get('filled_qty', 0)
                            if fill_price:
                                pos.entry_price = float(fill_price)
                                pos.hard_stop = round(pos.entry_price * (1 - self.hard_stop_pct), 2)
                                pos.highest_since_entry = pos.entry_price
                                if filled_qty:
                                    pos.shares = int(filled_qty)
                                fill_at = datetime.now(timezone.utc)
                                fill_update = {
                                    'order_status': 'filled',
                                    'fill_price': pos.entry_price,
                                    'filled_qty': pos.shares,
                                    'filled_at': fill_at,
                                    'order_filled_at': fill_at,
                                }
                                # Slippage attribution uses values captured at submit-time
                                # on the in-memory OpenPosition (no DB roundtrip needed).
                                if pos.bar_close_price and pos.bar_close_price > 0:
                                    fill_update['drift_bar_to_fill_bps'] = (
                                        (pos.entry_price - pos.bar_close_price) / pos.bar_close_price * 10000
                                    )
                                if pos.entry_quote_ask and pos.entry_quote_ask > 0:
                                    fill_update['drift_ask_to_fill_bps'] = (
                                        (pos.entry_price - pos.entry_quote_ask) / pos.entry_quote_ask * 10000
                                    )
                                if pos.order_submitted_at:
                                    fill_update['submit_to_fill_ms'] = int(
                                        (fill_at - pos.order_submitted_at).total_seconds() * 1000
                                    )
                                self.db.update_trade(pos.trade_id, fill_update)
                                pos.order_id = ''
                                logger.info(
                                    f"[{self.STRATEGY_NAME}] {sym}: filled @ ${pos.entry_price:.2f} "
                                    f"({pos.shares}sh)"
                                )
                                # NOW add StopMonitor watch (only after fill confirmed)
                                if self.stop_monitor:
                                    self.stop_monitor.add_watch(
                                        symbol=sym,
                                        stop_price=pos.hard_stop,
                                        shares=pos.shares,
                                        tp_leg_id='', sl_leg_id='',
                                        trade_db_id=pos.trade_id,
                                        entry_price=pos.entry_price,
                                        risk_per_share=0,  # not used — trail_pct overrides
                                        trail_r=0, activate_at_r=0.0,
                                        trail_pct=self.trail_stop_pct,  # percentage-based trail
                                        strategy=self.STRATEGY_NAME,
                                    )

                                # Log entry L2 async
                                try:
                                    from data_sources.l2_depth import log_l2_async
                                    log_l2_async(sym, datetime.now(timezone.utc), pos.trade_id,
                                                 self.db.update_trade, column='entry_l2_depth')
                                except Exception as e:
                                    logger.debug(f"[{self.STRATEGY_NAME}] {sym}: L2 async failed: {e}")

                        elif status in ('cancelled', 'expired', 'rejected'):
                            logger.warning(f"[{self.STRATEGY_NAME}] {sym}: order {status}")
                            del self.open_positions[sym]
                            self.invalidated.add(sym)
                            self.db.update_trade(pos.trade_id, {'order_status': status})
                            continue
                        else:
                            continue  # Still pending
                    except Exception as e:
                        logger.warning(f"[{self.STRATEGY_NAME}] {sym}: fill check failed: {e}")
                        continue

                # MACD flip check (polling — only signal that needs bar computation)
                now_et = datetime.now(ET)
                mins = max(30, int((now_et - now_et.replace(hour=9, minute=30, second=0)).total_seconds() / 60))
                bars = self.alpaca.get_1min_bars(sym, lookback_minutes=mins)
                if bars is None or len(bars) < self.macd_slow + self.macd_signal:
                    continue

                close = bars['close']
                ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
                ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
                histogram = (ema_fast - ema_slow) - (ema_fast - ema_slow).ewm(span=self.macd_signal, adjust=False).mean()

                if histogram.iloc[-1] <= 0:
                    # MACD flipped — remove from StopMonitor FIRST, then exit
                    if self.stop_monitor:
                        self.stop_monitor.remove_watch(sym)
                    self._submit_exit(sym, 'macd_flip')
                    exits.append(sym)

            except Exception as e:
                logger.error(f"[{self.STRATEGY_NAME}] {sym}: exit check failed: {e}")

        return exits

    def _submit_exit(self, symbol: str, reason: str) -> bool:
        """Submit a sell order."""
        pos = self.open_positions.get(symbol)
        if not pos:
            return False

        try:
            exit_price = 0.0
            order_id = ''

            if self.dry_run:
                logger.info(f"[{self.STRATEGY_NAME}] DRY RUN: would SELL {symbol} — {reason}")
                del self.open_positions[symbol]
                self.invalidated.add(symbol)
                return True

            # close_position() auto-cancels bracket legs (safety-net SL/TP)
            # Works for both force_close and macd_flip
            quote = self.alpaca.get_latest_quote(symbol)
            bid = quote.get('bid_price', 0)
            exit_price = bid if bid > 0 else pos.entry_price * 0.99

            # Cancel bracket legs first (SL/TP hold shares, blocking close_position)
            try:
                from alpaca.trading.requests import GetOrdersRequest
                from alpaca.trading.enums import QueryOrderStatus
                open_orders = self.alpaca.trading_client.get_orders(
                    GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
                )
                for oo in open_orders:
                    try:
                        self.alpaca.cancel_order(str(oo.id))
                        logger.info(f"[{self.STRATEGY_NAME}] {symbol}: cancelled bracket leg {str(oo.id)[:8]}")
                    except Exception:
                        pass
            except Exception as cancel_err:
                logger.warning(f"[{self.STRATEGY_NAME}] {symbol}: cancel orders before close: {cancel_err}")

            result = self.alpaca.close_position(symbol)
            order_id = result.get('id', '') if result else ''

            # Log exit microstructure
            if bid > 0:
                self.db.update_trade(pos.trade_id, {
                    'exit_quote_bid': bid,
                    'exit_quote_ask': quote.get('ask_price', 0),
                    'exit_limit_price': bid,
                    'exit_pricing_method': f'{reason}_close',
                })

            # Compute P&L
            pnl = (exit_price - pos.entry_price) * pos.shares
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0
            self.daily_pnl += pnl

            # Update DB
            self.db.update_trade(pos.trade_id, {
                'exit_price': exit_price,
                'exit_reason': reason,
                'exited_at': datetime.now(timezone.utc),
                'pnl': pnl,
                'pnl_pct': pnl_pct,
            })

            # Notify
            emoji = '✅' if pnl > 0 else '❌'
            logger.info(
                f"[{self.STRATEGY_NAME}] {emoji} SELL {symbol} ${exit_price:.2f} "
                f"{pnl_pct:+.1f}% (${pnl:+,.0f}) — {reason}"
            )
            if self.notifier:
                self.notifier.send_message_sync(
                    f"[MACD Wave] {emoji} SELL {symbol} ${exit_price:.2f} "
                    f"{pnl_pct:+.1f}% (${pnl:+,.0f}) — {reason}"
                )

            # Log L2 depth at exit time (non-blocking)
            try:
                from data_sources.l2_depth import log_l2_async
                log_l2_async(symbol, datetime.now(timezone.utc), pos.trade_id,
                             self.db.update_trade, column='exit_l2_depth')
            except Exception as e_l2:
                logger.debug(f"[{self.STRATEGY_NAME}] {symbol}: Exit L2 async failed: {e_l2}")

            del self.open_positions[symbol]
            self.invalidated.add(symbol)
            return True

        except Exception as e:
            logger.error(f"[{self.STRATEGY_NAME}] {symbol}: exit submission failed: {e}")
            # If we can't sell (e.g., "cannot be sold short" = already closed),
            # remove from open_positions to stop infinite retry loop
            err_str = str(e).lower()
            if 'sold short' in err_str or 'no position' in err_str or 'not found' in err_str:
                logger.warning(f"[{self.STRATEGY_NAME}] {symbol}: position gone — removing from tracking")
                del self.open_positions[symbol]
                self.invalidated.add(symbol)
            return False

    # ------------------------------------------------------------------
    # Force close
    # ------------------------------------------------------------------

    def force_close_all(self) -> int:
        """Force close all open positions at market. Returns count closed."""
        # Remove all watches from StopMonitor FIRST (prevents race with _on_trade)
        if self.stop_monitor:
            for sym in list(self.open_positions.keys()):
                self.stop_monitor.remove_watch(sym)

            # Drain any pending StopMonitor events (DON'T call check_exits —
            # that would trigger MACD flip sells and double-sell positions)
            for event in self.stop_monitor.drain_exit_events(strategy=self.STRATEGY_NAME):
                sym = event.symbol
                if sym in self.open_positions:
                    pos = self.open_positions[sym]
                    pnl = (event.exit_price - pos.entry_price) * pos.shares
                    self.db.update_trade(pos.trade_id, {
                        'exit_price': event.exit_price,
                        'exit_reason': event.exit_reason,
                        'exited_at': datetime.now(timezone.utc),
                        'pnl': pnl,
                    })
                    del self.open_positions[sym]
                    self.invalidated.add(sym)
                    logger.info(f"[{self.STRATEGY_NAME}] {sym}: drained StopMonitor exit before force close")

        closed = 0
        for sym in list(self.open_positions.keys()):
            if self._submit_exit(sym, 'force_close'):
                closed += 1

        if closed:
            logger.info(f"[{self.STRATEGY_NAME}] Force-closed {closed} positions")
            if self.notifier:
                self.notifier.send_message_sync(
                    f"[MACD Wave] ⏰ Force-closed {closed} positions — end of day"
                )

        return closed

    # ------------------------------------------------------------------
    # Daily report
    # ------------------------------------------------------------------

    def send_daily_report(self) -> None:
        """Send end-of-day summary via Telegram."""
        today = date.today().isoformat()

        # Get today's MACD wave trades from DB
        try:
            cursor = self.db.conn.execute(
                "SELECT * FROM trades WHERE trade_date = ? AND strategy = ? AND fill_price IS NOT NULL",
                (today, self.STRATEGY_NAME)
            )
            trades = [dict(row) for row in cursor.fetchall()]
        except Exception:
            trades = []

        if not trades:
            msg = f"[MACD Wave] 📊 Daily: 0 trades"
        else:
            wins = len([t for t in trades if (t.get('pnl') or 0) > 0])
            losses = len(trades) - wins
            total_pnl = sum(t.get('pnl', 0) or 0 for t in trades)
            msg = (
                f"[MACD Wave] 📊 Daily: {len(trades)} trades, "
                f"{wins}W {losses}L, P&L ${total_pnl:+,.0f}"
            )

        logger.info(msg)
        if self.notifier:
            self.notifier.send_message_sync(msg)

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def reset_daily(self) -> None:
        """Reset all daily state."""
        self.universe_opens.clear()
        self.universe_volumes.clear()
        self.crossed_stocks.clear()
        self.open_positions.clear()
        self.invalidated.clear()
        self.daily_pnl = 0.0
        self.trades_today = 0
        # Drain any stale bar events leftover from yesterday.
        self.drain_bar_events()
        self._bar_queue_full_logged = False

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_market_open() -> bool:
        """Check if US market is currently open (9:30-16:00 ET)."""
        now = datetime.now(ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close and now.weekday() < 5

    def is_force_close_time(self) -> bool:
        """Check if it's time to force close all positions."""
        now = datetime.now(ET)
        return (now.hour > self.force_close_hour or
                (now.hour == self.force_close_hour and now.minute >= self.force_close_minute))
