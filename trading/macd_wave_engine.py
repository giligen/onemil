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
from typing import Any, Dict, List, Optional, Set, Tuple

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
    ):
        cfg = config or {}
        self.alpaca = alpaca_client
        self.db = db
        self.notifier = notifier
        self.stop_monitor = stop_monitor
        self.dry_run = dry_run

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
        self.crossed_stocks: Dict[str, CrossedStock] = {}
        self.open_positions: Dict[str, OpenPosition] = {}
        self.invalidated: Set[str] = set()
        self.daily_pnl: float = 0.0
        self.trades_today: int = 0
        self.shutdown_requested: bool = False

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
                        continue

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

    def check_entries(self) -> List[str]:
        """
        Check crossed stocks for MACD entry signal.

        For each crossed stock, compute MACD histogram on 1-min bars.
        Enter when confirm_bars consecutive positive bars with hist >= threshold.
        """
        entries = []

        for sym, crossed in list(self.crossed_stocks.items()):
            if sym in self.open_positions:
                continue
            if sym in self.invalidated:
                continue

            # Capacity check
            if len(self.open_positions) >= self.max_concurrent:
                break

            # Daily loss limit
            if self.daily_loss_limit < 0 and self.daily_pnl <= self.daily_loss_limit:
                logger.warning(
                    f"[{self.STRATEGY_NAME}] Daily loss limit hit (${self.daily_pnl:,.0f}), "
                    f"skipping {sym}"
                )
                break

            try:
                # Fetch 1-min bars
                now_et = datetime.now(ET)
                mins = max(30, int((now_et - now_et.replace(hour=9, minute=30, second=0)).total_seconds() / 60))
                bars = self.alpaca.get_1min_bars(sym, lookback_minutes=mins)
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

                # Entry signal: N consecutive positive bars + histogram strength
                if crossed.pos_count >= self.confirm_bars:
                    if self.min_macd_hist_pct > 0 and hist_pct < self.min_macd_hist_pct:
                        logger.info(
                            f"[{self.STRATEGY_NAME}] {sym}: MACD hist too weak "
                            f"({hist_pct:.2f}% < {self.min_macd_hist_pct}%), resetting"
                        )
                        crossed.pos_count = 0  # Reset, wait for stronger signal
                        continue

                    if self.max_price_at_entry > 0 and latest_price > self.max_price_at_entry:
                        logger.debug(
                            f"[{self.STRATEGY_NAME}] {sym}: price ${latest_price:.2f} > "
                            f"max ${self.max_price_at_entry}, skipping"
                        )
                        continue

                    # ENTRY SIGNAL
                    logger.info(
                        f"[{self.STRATEGY_NAME}] {sym}: ENTRY SIGNAL — "
                        f"pos_count={consecutive_pos}, hist_pct={hist_pct:.2f}%, "
                        f"price=${latest_price:.2f}"
                    )
                    result = self._submit_entry(sym, latest_price, hist_pct, crossed)
                    if result:
                        entries.append(sym)

            except Exception as e:
                logger.error(f"[{self.STRATEGY_NAME}] {sym}: entry check failed: {e}")

        return entries

    def _submit_entry(
        self, symbol: str, price: float, macd_hist_pct: float, crossed: CrossedStock,
    ) -> bool:
        """Submit a buy order for MACD wave entry."""
        try:
            # Get quote for pricing
            quote = self.alpaca.get_latest_quote(symbol)
            ask = quote.get('ask_price', 0)
            bid = quote.get('bid_price', 0)
            if ask <= 0:
                ask = price * 1.001

            limit_price = round(ask * 1.001, 2)  # ask + 0.1%
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

            # Submit bracket buy with safety-net SL on Alpaca
            # If everything dies (StopMonitor + service), Alpaca still has a 5% floor
            safety_sl = round(limit_price * (1 - self.safety_net_sl_pct), 2)
            safety_tp = round(limit_price * 1.50, 2)  # effectively infinite
            order = self.alpaca.submit_bracket_order(
                symbol=symbol, qty=shares, side='buy',
                limit_price=limit_price,
                tp_price=safety_tp, sl_price=safety_sl,
            )
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

            # Entry microstructure
            self.db.update_trade(trade_id, {
                'entry_quote_bid': bid,
                'entry_quote_ask': ask,
                'entry_quote_spread': ask - bid if ask > 0 and bid > 0 else None,
            })

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

        # 1. Drain StopMonitor exit events (real-time trail/hard stop exits)
        if self.stop_monitor:
            for event in self.stop_monitor.drain_exit_events():
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
                        order_status = self.alpaca.get_order(pos.order_id)
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
                                self.db.update_trade(pos.trade_id, {
                                    'order_status': 'filled',
                                    'fill_price': pos.entry_price,
                                    'filled_qty': pos.shares,
                                    'filled_at': datetime.now(timezone.utc),
                                })
                                pos.order_id = ''
                                logger.info(
                                    f"[{self.STRATEGY_NAME}] {sym}: filled @ ${pos.entry_price:.2f} "
                                    f"({pos.shares}sh)"
                                )
                                # NOW add StopMonitor watch (only after fill confirmed)
                                if self.stop_monitor:
                                    risk_for_trail = pos.entry_price * self.trail_stop_pct
                                    self.stop_monitor.add_watch(
                                        symbol=sym,
                                        stop_price=pos.hard_stop,
                                        shares=pos.shares,
                                        tp_leg_id='', sl_leg_id='',
                                        trade_db_id=pos.trade_id,
                                        entry_price=pos.entry_price,
                                        risk_per_share=risk_for_trail,
                                        trail_r=1.0, activate_at_r=0.0,
                                    )
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
            for event in self.stop_monitor.drain_exit_events():
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
        self.crossed_stocks.clear()
        self.open_positions.clear()
        self.invalidated.clear()
        self.daily_pnl = 0.0
        self.trades_today = 0

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
