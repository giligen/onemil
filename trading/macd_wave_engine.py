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
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pytz
from dateutil.parser import isoparse

from trading.macd_conviction import compute_conviction_score
from trading.orphan_reconciler import (
    ReconcilerConfig, reconcile_strategy_orphans,
)
from trading.stop_monitor import build_exit_update
from trading.macd_cross_detector import (
    compute_macd_histogram,
    count_consecutive_positive_ending_at,
    find_wave_onset,
)

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
        news_provider=None,
        news_worker=None,
    ):
        cfg = config or {}
        self.alpaca = alpaca_client
        self.db = db
        self.notifier = notifier
        self.stop_monitor = stop_monitor
        self.order_stream = order_stream  # T3.1: TradingStream watcher (optional)
        # 2026-04-28: news_provider drives the halt-aware entry filter
        # (sub-ms SQLite read against news_cache.halt). news_worker is the
        # bull flag scanner's async classifier — when present, MACD wave
        # enqueues newly-monitored symbols so halt detection runs in the
        # background well before any entry signal could fire (≥35 bars / 30+
        # min lead). Both optional — if None, halt filter no-ops.
        self.news_provider = news_provider
        self.news_worker = news_worker
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
        # One-shot log flag for the entry time-of-day cutoff (reset daily).
        self._entry_cutoff_logged = False

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
        # Per-symbol trade cap per day — used by intraday recovery to decide
        # whether to re-add a previously-traded symbol to crossed_stocks. The
        # live path relies on crossed_stocks membership to prevent re-entry;
        # on mid-day restart that membership is lost, so we DB-check instead.
        self.max_waves = int(cfg.get('waves', {}).get('max_waves', 1))
        self.min_macd_hist_pct = float(entry.get('min_macd_hist_pct', 0.5))
        self.max_price_at_entry = float(entry.get('max_price_at_entry', 0))
        # Entry time-of-day gate (added 2026-05-22). cross_time_max_min only
        # bounds when the +10% threshold was first hit — NOT when the entry
        # fires. Forensic of 110 live trades (4 wks): entries <=09:45 ET =
        # 31% WR / -$17K; entries >09:45 ET = 11% WR / -$37K. Late-day MACD
        # crosses on momentum stocks are noise. This gate rejects any entry
        # whose actual decision time is > N minutes after 09:30 ET.
        # 0 = disabled (legacy behaviour). Default 15 (= 09:45 cliff edge).
        self.last_entry_minutes_after_open = int(
            entry.get('last_entry_minutes_after_open', 15))

        # Spread gate — skip entries when bid-ask spread is too wide.
        # 0 = disabled. 100 = skip if spread > 100bps.
        self.max_entry_spread_bps = float(entry.get('max_entry_spread_bps', 0))

        # 2026-04-28 incident-driven entry filters (HTCO bought at 22% of day
        # high after a 360% pre-mkt pump retraced; ONEG ran +12.9% then
        # collapsed to hard stop). Both are STRUCTURAL guards, not fitted
        # parameters — calibrated to be inert in the 16-month BT (min observed
        # pct_of_day_high in production-filtered signals: 0.727).
        filt = cfg.get('filter', {})
        # Day-from-high floor: skip when entry_price / today's RTH high
        # < threshold. 0 = disabled. 0.70 = catches HTCO archetype while
        # provably inert in BT (research/study_macd_day_from_high.py).
        self.day_from_high_min_pct = float(filt.get('day_from_high_min_pct', 0.0))
        # Halt-aware filter: skip when recent news headline mentions a
        # halt or circuit breaker. Requires news_provider to be wired in.
        self.halt_aware = bool(filt.get('halt_aware', False))

        # Smart entry: L1-informed pricing + early entry on strong book
        self.smart_entry_enabled = bool(entry.get('smart_entry_enabled', False))
        # Limit-price buffer above the ask. Set >0 to survive sub-second ask ticks
        # during submit latency. Matches ORB/bull flag's 30bps stop-limit buffer.
        # 0 = use current ask as-is (legacy behavior — vulnerable to fast movers).
        self.smart_limit_ask_buffer_bps = float(entry.get('smart_limit_ask_buffer_bps', 30.0))
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

        # V4 conviction sizing (step 1+2 validated: +$54K / +49.5% on 15mo BT).
        # Applied as a position-size multiplier at entry time; never as a filter.
        # Formula lives in trading/macd_conviction.py (shared with BT).
        conv_cfg = sizing.get('conviction_sizing', {})
        self.conviction_sizing_enabled = bool(conv_cfg.get('enabled', False))
        self.max_position_size_usd = float(conv_cfg.get('max_position_size_usd', 90_000))
        # Sanity checks — invalid caps silently break trading. Surface at startup.
        if self.max_position_size_usd <= 0:
            logger.warning(
                f"[{self.STRATEGY_NAME}] conviction_sizing.max_position_size_usd="
                f"{self.max_position_size_usd} is <=0 — all conviction-sized trades "
                f"will be rejected (shares=0). Set to a positive value or disable sizing."
            )
        elif (self.conviction_sizing_enabled
              and self.max_position_size_usd < self.position_size):
            logger.warning(
                f"[{self.STRATEGY_NAME}] conviction_sizing.max_position_size_usd="
                f"${self.max_position_size_usd:,.0f} is BELOW baseline position_size="
                f"${self.position_size:,.0f} — baseline (conv=1.0) trades will be "
                f"capped below flat sizing. Raise the cap or lower position_size."
            )
        if self.conviction_sizing_enabled:
            logger.info(
                f"[{self.STRATEGY_NAME}] Conviction sizing: ENABLED "
                f"(V4 3-tier, max position ${self.max_position_size_usd:,.0f})"
            )

        # Risk
        risk = cfg.get('risk', {})
        self.hard_stop_pct = float(risk.get('hard_stop_pct', 0.02))
        self.trail_stop_pct = float(risk.get('trail_stop_pct', 0.003))  # 0.3% trail below highest
        # CORD 5/8 fix: trail arms only after high crosses entry × (1+arm_pct).
        # Default = trail_stop_pct so trail can never trigger at a loss
        # (premature exit on first post-fill bid dip — see tests).
        self.trail_arm_pct = float(risk.get('trail_arm_pct', self.trail_stop_pct))
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
        # Per-entry telemetry for EOD telegram (conviction sizing trace)
        self._eod_traded: List[Dict[str, Any]] = []

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
            # Race-A fix (2026-06-06): an exit_pending_verification row was
            # emitted by StopMonitor.BRANCH_LAST_RESORT — the orphan_reconciler
            # owns it from this point. The engine must NOT re-rehydrate it
            # into open_positions / StopMonitor; doing so would spawn a fresh
            # watch with a stale hard_stop AND bypass the reconciler (which
            # would then skip any symbol back in open_positions). See plan
            # mellow-sniffing-abelson + L1/L2 in the orphan-fix series.
            if trade.get('order_status') == 'exit_pending_verification':
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
                            trail_arm_pct=self.trail_arm_pct,
                            strategy=self.STRATEGY_NAME,
                        )
                    logger.info(
                        f"[{self.STRATEGY_NAME}] sync: recovered {sym} "
                        f"({shares}sh @ ${fill_price:.2f})"
                    )
            else:
                # Position GONE from Alpaca — closed externally. The exit could
                # be: StopMonitor's market sell at hard_stop/trail_stop, the
                # Alpaca bracket SL/TP leg firing, or a manual cancel.
                # Inspect order history to label correctly (2026-05-06: was
                # always tagging as bracket_exit which inflated apparent
                # bracket damage in post-mortems).
                exit_price = fill_price  # fallback
                exit_reason = 'bracket_exit'  # default if we can't classify
                try:
                    from alpaca.trading.requests import GetOrdersRequest
                    from alpaca.trading.enums import QueryOrderStatus
                    orders = self.alpaca.trading_client.get_orders(
                        GetOrdersRequest(
                            status=QueryOrderStatus.CLOSED,
                            symbols=[sym], limit=5,
                        )
                    )
                    classified_sell = None
                    for o in orders:
                        if (o.side.value == 'sell' and o.status.value == 'filled'
                                and o.filled_avg_price):
                            classified_sell = o
                            break
                    if classified_sell is not None:
                        o = classified_sell
                        exit_price = float(o.filled_avg_price)
                        oc = o.order_class.value if o.order_class else None
                        # 2026-05-07 fix (regression from yesterday's D2 fix):
                        # StopMonitor submits LIMIT sells with quote-aware
                        # pricing for ALL exits — trail_stop AND hard_stop.
                        # The previous "ot == 'limit' → limit_exit" branch
                        # mislabeled today's trail_stops as limit_exit. The
                        # discriminator is the price magnitude, not the order
                        # type. Bracket legs are tagged by oc explicitly.
                        if oc == 'bracket':
                            exit_reason = 'bracket_sl_tp'
                        elif oc in ('oto', 'oco'):
                            exit_reason = 'bracket_sl_tp'
                        else:
                            # Solo sell from StopMonitor — disambiguate by drop %.
                            # MACD wave hard_stop is at -2% (config). Anything
                            # close to -2% is hard_stop; smaller drop is trail.
                            try:
                                drop_pct = ((exit_price - fill_price)
                                             / fill_price) if fill_price > 0 else 0
                                if drop_pct <= -0.018:  # within 0.2% of -2% hard_stop
                                    exit_reason = 'hard_stop'
                                else:
                                    exit_reason = 'trail_stop'  # incl small wins
                            except Exception:
                                exit_reason = 'stopmonitor_exit'
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

        # Phase 2 (added 2026-04-16): orphan-position recovery.
        # Catch the reverse of the above: Alpaca has a position we have no
        # filled DB record for. Happens when the fill-update path fails
        # (e.g., the old _gc_stale_pending purged a still-live order before
        # the fill landed; TradingStream reconnected and missed a push; or
        # a crash dropped the in-flight update). Without this, the position
        # silently exits via bracket SL/TP and we never even log it.
        # db_open already contains pending_new rows (they have no exit_price).
        # Filter to pending_new only — those are the candidates for orphan recovery.
        db_pending_by_sym = {
            t['symbol']: t for t in db_open
            if (t.get('order_status') == 'pending_new'
                and t.get('fill_price') in (None, 0, 0.0))
        }
        db_open_syms = {
            t['symbol'] for t in db_open
            if t.get('fill_price') not in (None, 0, 0.0)
        }

        for sym, alpaca_pos in alpaca_positions.items():
            if sym in self.open_positions or sym in db_open_syms:
                continue  # already tracked
            # Alpaca has it, we don't — orphan.
            trade = db_pending_by_sym.get(sym)
            try:
                avg_price = float(alpaca_pos.avg_entry_price)
                qty = int(alpaca_pos.qty)
            except Exception:
                continue
            if trade is None:
                logger.warning(
                    f"[{self.STRATEGY_NAME}] sync: orphan Alpaca position {sym} "
                    f"({qty}sh @ ${avg_price:.2f}) — no DB record, skipping"
                )
                continue
            # Update the stuck pending_new trade with actual fill info.
            hard_stop = round(avg_price * (1 - self.hard_stop_pct), 2)
            fill_update = {
                'order_status': 'filled',
                'fill_price': avg_price,
                'filled_qty': qty,
                'filled_at': datetime.now(timezone.utc),
                'order_filled_at': datetime.now(timezone.utc),
            }
            try:
                self.db.update_trade(trade['id'], fill_update)
            except Exception as e:
                logger.warning(
                    f"[{self.STRATEGY_NAME}] sync: DB update failed for {sym}: {e}"
                )
                continue
            # Recover into open_positions so MACD flip + trailing-stop monitoring resumes.
            self.open_positions[sym] = OpenPosition(
                symbol=sym, entry_price=avg_price, shares=qty,
                hard_stop=hard_stop, trade_id=trade['id'], order_id='',
                entry_time=datetime.now(timezone.utc),
                macd_hist_at_entry=0, highest_since_entry=avg_price,
            )
            if self.stop_monitor:
                self.stop_monitor.add_watch(
                    symbol=sym, stop_price=hard_stop, shares=qty,
                    tp_leg_id='', sl_leg_id='', trade_db_id=trade['id'],
                    entry_price=avg_price, risk_per_share=0,
                    trail_r=0, activate_at_r=0.0,
                    trail_pct=self.trail_stop_pct,
                    trail_arm_pct=self.trail_arm_pct,
                    strategy=self.STRATEGY_NAME,
                )
            logger.warning(
                f"[{self.STRATEGY_NAME}] sync: RECOVERED orphan {sym} "
                f"({qty}sh @ ${avg_price:.2f}) — DB back-filled, "
                f"MACD/trail monitoring resumed"
            )

        # Phase 3 (added 2026-06-05): cross-strategy orphan reconciliation.
        # Handles the case the pending_new recovery above CAN'T handle:
        # broker still holds a position despite our DB marking it as
        # exited via 'stop_loss_unconfirmed' (the SMU 10-day orphan was
        # this exact pattern). Strictly more conservative than the old
        # per-engine "warning + skip" branch — closes only when the
        # hardened predicate proves ownership.
        try:
            cfg = getattr(self, 'orphan_reconciler_cfg', None) or ReconcilerConfig()
            # alpaca_positions above maps symbol → SDK Position object;
            # convert to the dict shape the reconciler expects so it
            # doesn't have to re-hit Alpaca.
            broker_snapshot = [
                {
                    'symbol': p.symbol,
                    'qty': int(p.qty),
                    'avg_entry_price': float(p.avg_entry_price),
                    'unrealized_pl': float(getattr(p, 'unrealized_pl', 0) or 0),
                }
                for p in alpaca_positions.values()
            ]
            reconcile_strategy_orphans(
                strategy=self.STRATEGY_NAME, alpaca=self.alpaca, db=self.db,
                notifier=self.notifier,
                tracked_symbols=set(self.open_positions.keys()),
                cfg=cfg,
                broker_positions=broker_snapshot,
            )
        except Exception as e:
            logger.error(
                f"[{self.STRATEGY_NAME}] orphan reconciler raised: {e} — "
                f"sync_positions continues"
            )

    # ------------------------------------------------------------------
    # Mid-day restart recovery
    # ------------------------------------------------------------------

    def _find_cross_minute(
        self,
        bars_df: pd.DataFrame,
        open_price: float,
        market_open_et: datetime,
    ) -> Tuple[Optional[int], int]:
        """Scan historical bars for the first minute the stock hit the
        `min_intraday_pct` threshold. Returns (cross_minute, vol_at_cross) or
        (None, 0) if never crossed in the bars provided.

        Mirrors the cross logic in scan_for_movers: vol_at_cross = cumulative
        volume from 9:30 through the bar that breaches the threshold.
        """
        if open_price <= 0 or bars_df is None or bars_df.empty:
            return None, 0
        threshold = open_price * (1.0 + self.min_intraday_pct / 100.0)
        cum_vol = 0
        for _, bar in bars_df.iterrows():
            try:
                cum_vol += int(bar.get('volume', 0) or 0)
            except (TypeError, ValueError):
                pass
            try:
                bar_high = float(bar.get('high', 0) or 0)
            except (TypeError, ValueError):
                continue
            if bar_high >= threshold:
                bar_ts = bar.get('timestamp')
                try:
                    cross_minute = int(
                        (bar_ts.astimezone(ET) - market_open_et).total_seconds() / 60
                    )
                except Exception:
                    return None, 0
                return cross_minute, cum_vol
        return None, 0

    def _already_traded_and_capped_today(self, sym: str) -> bool:
        """True if symbol has hit `max_waves` trades today (DB-backed).
        Prevents re-adding a symbol to `crossed_stocks` if it already filled
        + exited today and the per-symbol wave cap blocks another entry."""
        if self.max_waves <= 0:
            return False
        today = date.today().isoformat()
        try:
            trades = self.db.get_trades_by_date(today)
        except Exception as e:
            logger.debug(f"[{self.STRATEGY_NAME}] _already_traded: DB query failed: {e}")
            return False
        count = sum(
            1 for t in (trades or [])
            if t.get('symbol') == sym and t.get('strategy') == self.STRATEGY_NAME
        )
        return count >= self.max_waves

    def _sync_intraday_state(self) -> None:
        """Rebuild intraday state on mid-day restart.

        Repopulates `universe_opens`, `crossed_stocks`, `invalidated` from
        snapshots + historical 1-min bars, and replays closed-trade P&L from
        DB. No-op outside 9:30-15:45 ET window (nothing to recover) and when
        universe is empty (pre-build restart — next scanner cycle will handle).

        Called from main.py after `sync_positions()`. Idempotent.
        """
        now_et = datetime.now(ET)
        from datetime import time as dtime
        market_open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

        # Time window guard
        if now_et < market_open_et:
            logger.debug(f"[{self.STRATEGY_NAME}] intraday sync: pre-market, skipping")
            return
        if now_et.time() >= dtime(15, 45):
            logger.debug(f"[{self.STRATEGY_NAME}] intraday sync: post-close, skipping")
            return

        # Ensure universe exists. If pre-market build missed us (booted after
        # 9:30) try to build now — idempotent, cheap.
        if not self.universe:
            try:
                self.build_universe()
            except Exception as e:
                logger.warning(f"[{self.STRATEGY_NAME}] intraday sync: build_universe failed: {e}")
        if not self.universe:
            logger.warning(f"[{self.STRATEGY_NAME}] intraday sync: universe empty, skipping")
            return

        minutes_since_open = max(0, int((now_et - market_open_et).total_seconds() / 60))
        logger.info(
            f"[{self.STRATEGY_NAME}] intraday sync: recovering state "
            f"({minutes_since_open}min since open, universe={len(self.universe)})"
        )

        # ---- Step 1: snapshots → universe_opens + split detection ----
        snapshots: Dict[str, Dict] = {}
        chunk_size = 200
        for i in range(0, len(self.universe), chunk_size):
            chunk = self.universe[i:i + chunk_size]
            try:
                chunk_snaps = self.alpaca.get_snapshots(chunk)
                snapshots.update(chunk_snaps or {})
            except Exception as e:
                logger.warning(f"[{self.STRATEGY_NAME}] intraday sync: snapshot chunk failed: {e}")

        for sym, snap in snapshots.items():
            if not isinstance(snap, dict):
                continue
            open_price = float(snap.get('open', 0) or 0)
            prev_close = float(snap.get('prev_close', 0) or 0)
            # Reverse-split detection (mirrors scan_for_movers lines 535-546)
            if prev_close > 0 and open_price > 0:
                jump_ratio = abs(open_price - prev_close) / prev_close
                if jump_ratio > 1.0:
                    self.invalidated.add(sym)
                    logger.info(
                        f"[{self.STRATEGY_NAME}] {sym}: split detected on sync "
                        f"(prev_close ${prev_close:.2f} → open ${open_price:.2f}) — invalidating"
                    )
                    continue
            if open_price > 0:
                self.universe_opens[sym] = open_price

        # ---- Step 2: prefilter candidates via snapshot high/open ----
        threshold_ratio = 1.0 + self.min_intraday_pct / 100.0
        candidates: List[str] = []
        for sym, snap in snapshots.items():
            if sym in self.invalidated:
                continue
            if sym in self.open_positions:
                continue  # sync_positions already handled live positions
            if sym in self.crossed_stocks:
                continue  # already tracked (shouldn't happen at startup but safe)
            if not isinstance(snap, dict):
                continue
            open_price = float(snap.get('open', 0) or 0)
            bar_high = float(snap.get('high', 0) or 0)
            if open_price <= 0 or bar_high <= 0:
                continue
            if bar_high / open_price < threshold_ratio:
                continue
            if self._already_traded_and_capped_today(sym):
                continue
            candidates.append(sym)

        # ---- Step 3: fetch historical bars + reconstruct crossed_stocks ----
        recovered: List[str] = []
        if candidates:
            try:
                bars_by_sym = self.alpaca.get_1min_bars_multi(
                    candidates, lookback_minutes=minutes_since_open + 5
                )
            except Exception as e:
                logger.warning(f"[{self.STRATEGY_NAME}] intraday sync: bar fetch failed: {e}")
                bars_by_sym = {}

            for sym in candidates:
                bars = bars_by_sym.get(sym)
                if bars is None or (hasattr(bars, 'empty') and bars.empty):
                    continue
                open_price = self.universe_opens.get(sym, 0)
                cross_minute, vol_at_cross = self._find_cross_minute(
                    bars, open_price, market_open_et
                )
                if cross_minute is None:
                    continue
                if self.cross_time_max_min > 0 and cross_minute > self.cross_time_max_min:
                    continue  # crossed outside the entry window
                # Apply volume filters (same gates scan_for_movers uses)
                if self.max_vol_at_cross > 0 and vol_at_cross > self.max_vol_at_cross:
                    self.invalidated.add(sym)
                    continue
                if self.min_vol_at_cross > 0 and vol_at_cross < self.min_vol_at_cross:
                    continue

                # Derive crossed_at timestamp from the actual cross bar. Best
                # approximation: the minute that breached threshold.
                crossed_at_utc = market_open_et.astimezone(timezone.utc) + timedelta(
                    minutes=cross_minute
                )
                self.crossed_stocks[sym] = CrossedStock(
                    symbol=sym,
                    open_price=open_price,
                    cross_time_min=cross_minute,
                    vol_at_cross=vol_at_cross,
                    crossed_at=crossed_at_utc,
                )
                if self.stop_monitor is not None:
                    try:
                        self.stop_monitor.subscribe_bars(sym)
                    except Exception as e:
                        logger.debug(
                            f"[{self.STRATEGY_NAME}] {sym}: subscribe_bars failed: {e}"
                        )
                recovered.append(sym)

        # ---- Step 4: replay today's closed trades into daily_pnl counters ----
        today = date.today().isoformat()
        replayed = 0
        try:
            all_today = self.db.get_trades_by_date(today) or []
        except Exception as e:
            logger.warning(f"[{self.STRATEGY_NAME}] intraday sync: DB trades fetch failed: {e}")
            all_today = []
        closed_today = [
            t for t in all_today
            if t.get('strategy') == self.STRATEGY_NAME
            and t.get('exit_price') is not None
        ]
        closed_today.sort(key=lambda t: t.get('exited_at') or t.get('created_at') or '')
        for t in closed_today:
            try:
                pnl = float(t.get('pnl') or 0)
            except (TypeError, ValueError):
                pnl = 0.0
            self.daily_pnl += pnl
            self.trades_today += 1
            replayed += 1

        # ---- Telemetry ----
        sample = ', '.join(recovered[:8]) + ('...' if len(recovered) > 8 else '')
        logger.info(
            f"[{self.STRATEGY_NAME}] intraday state synced: "
            f"{len(self.universe_opens)} opens, {len(recovered)} crossed recovered "
            f"[{sample}], {replayed} closed trades replayed, "
            f"daily_pnl=${self.daily_pnl:+,.2f}, trades_today={self.trades_today}"
        )
        if self.notifier and (recovered or replayed):
            try:
                self.notifier.send_message_sync(
                    f"[MACD Wave] 🔄 Restart recovery: {len(recovered)} crossed symbols "
                    f"reconstructed, {replayed} trades replayed "
                    f"(day P&L ${self.daily_pnl:+,.0f})"
                )
            except Exception:
                pass

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

                    # Tick-level pre-filter (cheap): is the latest trade above threshold?
                    # If not, skip — no need to fetch bars.
                    pct_change = (price - open_price) / open_price * 100
                    if pct_change < self.min_intraday_pct:
                        continue

                    # BAR-CONFIRMED CROSS DETECTION (2026-05-06: BT-LIVE parity fix)
                    # The previous tick-only path detected phantom crosses where a
                    # single off-exchange print exceeded threshold but no 1-min bar
                    # high actually crossed. Those phantom signals drove ~30 of 69
                    # losing live trades through 5/6 (BT cache had no signal for
                    # them). Fix: require bar.high >= threshold, matching BT's
                    # `bars.iloc[i]['high'] >= open*1.10` logic exactly.
                    #
                    # cross_time_min = bar INDEX (1-based) within the regular-
                    # session bars, matching BT's `cross_time_min = si + 1` in
                    # macd_wave_backtest.py:438. Index-based (not elapsed-minutes)
                    # is the BT semantic and is what the cross_time_max_min
                    # filter is calibrated against in the validated parameters.
                    vol_at_cross = 0
                    bar_cross_minute: Optional[int] = None
                    bar_cross_ts = None
                    threshold_price = open_price * (1 + self.min_intraday_pct / 100.0)
                    try:
                        bars = self.alpaca.get_1min_bars(sym, lookback_minutes=minutes_since_open + 1)
                        if bars is not None and not bars.empty:
                            # Clip to regular session (09:30 ET to 16:00 ET) — matches
                            # BT generate_signals (macd_wave_backtest.py:412-422).
                            if 'timestamp' in bars.columns:
                                bar_ts_series = pd.to_datetime(bars['timestamp'], utc=True)
                                et_dt = bar_ts_series.dt.tz_convert(ET)
                                regular = (
                                    ((et_dt.dt.hour == 9) & (et_dt.dt.minute >= 30))
                                    | ((et_dt.dt.hour > 9) & (et_dt.dt.hour < 16))
                                )
                                bars_clipped = bars[regular].reset_index(drop=True)
                            else:
                                bars_clipped = bars.reset_index(drop=True)
                            vol_at_cross = int(bars_clipped['volume'].sum()) if not bars_clipped.empty else 0
                            for idx, bar in bars_clipped.iterrows():
                                try:
                                    bar_high = float(bar.get('high', 0) or 0)
                                except (TypeError, ValueError):
                                    continue
                                if bar_high >= threshold_price:
                                    bar_cross_minute = int(idx) + 1  # BT semantic: si + 1
                                    bar_cross_ts = bar.get('timestamp')
                                    # Recompute vol_at_cross to be cumulative volume
                                    # through the cross bar (matches BT).
                                    try:
                                        vol_at_cross = int(
                                            bars_clipped.iloc[:int(idx) + 1]['volume'].sum()
                                        )
                                    except Exception:
                                        pass
                                    break
                    except Exception:
                        pass

                    # Phantom-tick guard: tick said crossed but no bar.high crossed → SKIP
                    if bar_cross_minute is None:
                        logger.debug(
                            f"[{self.STRATEGY_NAME}] {sym}: tick at +{pct_change:.1f}% but "
                            f"no bar.high >= threshold (${threshold_price:.2f}) — phantom, skipping"
                        )
                        continue

                    # Apply cross time filter on BAR-CONFIRMED minute (matches BT)
                    if self.cross_time_max_min > 0 and bar_cross_minute > self.cross_time_max_min:
                        continue

                    # Apply volume filter (existing behaviour preserved)
                    if self.max_vol_at_cross > 0 and vol_at_cross > self.max_vol_at_cross:
                        logger.debug(f"[{self.STRATEGY_NAME}] {sym}: vol {vol_at_cross:,} > {self.max_vol_at_cross:,}, skipping")
                        self.invalidated.add(sym)
                        continue
                    if self.min_vol_at_cross > 0 and vol_at_cross < self.min_vol_at_cross:
                        continue

                    # Passed all filters — start monitoring
                    # crossed_at: prefer the actual bar timestamp; fallback to
                    # market_open + bar_cross_minute (rough proxy when ts missing).
                    if bar_cross_ts is not None:
                        try:
                            crossed_at_utc = pd.to_datetime(bar_cross_ts, utc=True).to_pydatetime()
                        except Exception:
                            crossed_at_utc = market_open_et.astimezone(timezone.utc) + timedelta(minutes=bar_cross_minute)
                    else:
                        crossed_at_utc = market_open_et.astimezone(timezone.utc) + timedelta(minutes=bar_cross_minute)
                    self.crossed_stocks[sym] = CrossedStock(
                        symbol=sym,
                        open_price=open_price,
                        cross_time_min=bar_cross_minute,
                        vol_at_cross=vol_at_cross,
                        crossed_at=crossed_at_utc,
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

                    # 2026-04-28: pre-warm halt detection. NewsWorker fetches
                    # + classifies in background; classify_news side-effects
                    # the halt flag into news_cache. By the time MACD's entry
                    # signal fires (≥35 bars later), the halt-aware filter's
                    # is_halted_today() read is sub-ms with the answer in
                    # place. Idempotent: NewsWorker.enqueue de-dups symbols
                    # already classified or pending.
                    if self.news_worker is not None:
                        try:
                            self.news_worker.enqueue(sym)
                        except Exception as e:
                            logger.debug(
                                f"[{self.STRATEGY_NAME}] {sym}: news enqueue failed: {e}"
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

    # Order statuses that mean the order is really gone and safe to purge.
    _DEAD_ORDER_STATUSES = frozenset({'rejected', 'canceled', 'cancelled', 'expired'})
    # Order statuses that mean the order is still working — keep tracking.
    _LIVE_ORDER_STATUSES = frozenset({
        'pending_new', 'new', 'accepted', 'accepted_for_bidding',
        'partially_filled', 'pending_replace', 'replaced',
    })

    def _gc_stale_pending(self) -> None:
        """Purge pending orders, but ONLY after confirming they're really dead.

        Prior behavior (pre-2026-04-16): any order >2min in open_positions was
        purged unconditionally on the assumption "pending >2min = rejected or
        lost". That's wrong for limit orders on thin/volatile stocks, which
        legitimately take 3-20 minutes to fill. The old GC orphaned those
        orders — Alpaca still had them, but our code stopped tracking, so the
        subsequent fill event never updated the DB, the MACD-flip exit loop
        never ran, and the position could only exit via bracket SL/TP.
        Evidence: 2026-04-16 CDNA (filled 19 min after submit, -$0 realized,
        orphaned) + BBGI (filled 6:49 after submit, later bracket-stopped for
        -$4,603 without ever being in our tracking).

        New behavior: for each order >2min old, query Alpaca for actual status.
        - Dead (rejected/canceled/expired) → purge
        - Filled → leave in place, fill-check loop will claim it next tick
        - Live (pending_new/accepted/new/partially_filled) → keep tracking
        - Very stale (>30min) AND still live → actively cancel + purge

        Network errors keep the position tracked (retry next cycle). No silent
        purges that strand capital.
        """
        now = datetime.now(timezone.utc)
        HARD_CANCEL_SECONDS = 30 * 60  # 30 min

        for psym in list(self.open_positions.keys()):
            pos = self.open_positions.get(psym)
            if pos is None or not pos.order_id:
                continue  # already filled or no order_id — nothing to GC
            age_s = (now - pos.entry_time).total_seconds()
            if age_s < 120:
                continue  # still within the "let it settle" window

            try:
                order_status = self.alpaca.get_order(pos.order_id)
            except Exception as e:
                logger.warning(
                    f"[{self.STRATEGY_NAME}] {psym}: stale-GC skipped — "
                    f"Alpaca get_order failed: {e}. Will retry next cycle."
                )
                continue  # transient network — keep tracking, retry next tick

            status = (order_status or {}).get('status', '') if order_status else ''

            if status in self._DEAD_ORDER_STATUSES:
                logger.warning(
                    f"[{self.STRATEGY_NAME}] {psym}: order {status} on Alpaca "
                    f"(age {age_s:.0f}s) — purging"
                )
                self.open_positions.pop(psym, None)
                self.invalidated.add(psym)
            elif status == 'filled':
                # Missed the fill push; next fill-check tick will claim it.
                logger.info(
                    f"[{self.STRATEGY_NAME}] {psym}: order already filled on "
                    f"Alpaca (age {age_s:.0f}s) — keeping for fill-check"
                )
            elif age_s >= HARD_CANCEL_SECONDS:
                # Truly stale — actively cancel to free capital, then purge.
                try:
                    self.alpaca.cancel_order(pos.order_id)
                    logger.warning(
                        f"[{self.STRATEGY_NAME}] {psym}: order still '{status}' "
                        f"after {age_s/60:.0f}min — cancelling + purging"
                    )
                except Exception as e:
                    logger.warning(
                        f"[{self.STRATEGY_NAME}] {psym}: cancel failed: {e} — "
                        f"purging anyway (will resolve via sync_positions)"
                    )
                self.open_positions.pop(psym, None)
                self.invalidated.add(psym)
            elif status in self._LIVE_ORDER_STATUSES or not status:
                # Live or unknown-but-not-dead — keep tracking silently.
                # Emit DEBUG so we can grep for "still pending" if needed.
                status_label = status or 'unknown'
                logger.debug(
                    f"[{self.STRATEGY_NAME}] {psym}: still '{status_label}' "
                    f"on Alpaca (age {age_s:.0f}s) — keeping"
                )
            else:
                # Unknown status — log and keep tracking; reconcile next cycle.
                logger.warning(
                    f"[{self.STRATEGY_NAME}] {psym}: unrecognized Alpaca status "
                    f"'{status}' (age {age_s:.0f}s) — keeping, will retry"
                )

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
        # True (un-floored) minutes since 09:30 ET — for the entry time-of-day
        # gate. mins_outer is floored at 30 for bar lookback and cannot be
        # reused here.
        true_mins_since_open = self._minutes_since_open(now_et_outer)
        past_entry_cutoff = self._is_past_entry_cutoff(now_et_outer)
        if past_entry_cutoff and not self._entry_cutoff_logged:
            logger.info(
                f"[{self.STRATEGY_NAME}] entry window closed — "
                f"{true_mins_since_open}min since open > "
                f"{self.last_entry_minutes_after_open}min cutoff; "
                f"no new entries (monitoring/exits unaffected)"
            )
            self._entry_cutoff_logged = True
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
                # 2026-05-01: warmup gate dropped from 35 (= macd_slow + macd_signal)
                # to confirm_bars (=3) to match BT's signal-cache generation —
                # `pandas.ewm(adjust=False)` produces values from bar 0 with
                # `adjust=False` initialization. Production-filtered BT analysis
                # showed only 4 trades over 16mo at warmup=35 (effectively zero
                # edge); at warmup=0 the cache shows 595 trades / +$92K cum P&L
                # with TRAIN/VAL/OOS all positive. The "MACD" at early bars is
                # mathematically biased (EMAs initialized at close[0], haven't
                # converged) — closer to a momentum-since-open measurement than
                # a true MACD. Cross-validation supports edge but the
                # methodology is non-standard. Monitor day-1 metrics:
                #   - signal volume (expect 5-30/day, was ~0)
                #   - halt filter coverage (NewsWorker queue depth at entry)
                #   - spread gate hit rate
                # Revert: change to len(bars) < self.macd_slow + self.macd_signal.
                min_bars_required = max(self.confirm_bars, 3)
                if bars is None or len(bars) < min_bars_required:
                    logger.debug(
                        f"[{self.STRATEGY_NAME}] {sym}: insufficient bars "
                        f"({len(bars) if bars is not None else 0}, need {min_bars_required})"
                    )
                    continue

                # Compute MACD via shared detector module — same logic as
                # macd_wave_backtest.py uses; parity by construction.
                close = bars['close']
                histogram = compute_macd_histogram(
                    close, self.macd_fast, self.macd_slow, self.macd_signal,
                )

                # Count consecutive positive histogram bars ending at the
                # latest bar (live's "is now confirmed?" semantic). Replaces
                # the inline reverse-iteration loop with the shared helper.
                latest_price = close.iloc[-1]
                latest_hist = histogram.iloc[-1]
                hist_pct = latest_hist / latest_price * 100 if latest_price > 0 else 0

                # Reference price/time for slippage attribution: last 1-min bar
                # seen at signal time (the BT's ideal entry reference).
                # NOTE: Alpaca 1-min bar `timestamp` is the bar START (the minute
                # the bar represents). The bar CLOSES 60s later — see
                # _bar_start_to_close() for the conversion (was off by 60s before
                # 2026-04-15, inflating bar_close_to_loop_ms by exactly 60_000).
                try:
                    bar_close_at = self._bar_start_to_close(bars['timestamp'].iloc[-1])
                    bar_close_price = float(latest_price)
                except Exception:
                    bar_close_at = None
                    bar_close_price = None

                consecutive_pos = count_consecutive_positive_ending_at(
                    histogram, len(histogram) - 1,
                )
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

                # Entry time-of-day gate. cross_time_max_min bounds only when
                # the +10% threshold was first hit, not when the entry fires —
                # so without this, MACD-cross entries can submit hours after
                # the open. Forensic: entries >09:45 ET ran 11% WR / -$37K.
                if past_entry_cutoff:
                    logger.debug(
                        f"[{self.STRATEGY_NAME}] {sym}: entry skipped — "
                        f"{true_mins_since_open}min since open > "
                        f"{self.last_entry_minutes_after_open}min cutoff"
                    )
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

    # ---------------------------------------------------------------------
    # Entry filters (2026-04-28 incident-driven)
    # ---------------------------------------------------------------------

    def _check_day_from_high(
        self, bars_df, price: float,
    ) -> Tuple[bool, str]:
        """Day-from-high floor: skip when entry price has retraced too far
        from today's RTH high. Returns (passes, reason_if_skip).

        Computes today's high from the bar cache (RTH only — premarket bars
        excluded since they often print exotic). If no prior RTH bars are
        available (e.g. 9:30 ET first-bar entry), we pass (no information).

        DST-safe: converts UTC bar timestamps to ET via module-level `ET`
        timezone, then checks 9:30 <= time < 16:00 in ET directly. A naive
        UTC range (13:30-20:00) only matches EDT — in EST it would include
        8:30-9:30 ET premarket (exactly the gap-up regime we want to
        exclude) and miss 15:00-16:00 ET RTH.

        Calibration: in 16-mo BT on production-filtered signals, no signal
        had pct_of_day_high < 0.727. Threshold 0.70 is provably inert.
        """
        if self.day_from_high_min_pct <= 0:
            return True, ""
        if bars_df is None or len(bars_df) == 0:
            return True, ""

        df = bars_df
        try:
            if 'timestamp' in df.columns:
                ts_utc = pd.to_datetime(df['timestamp'], utc=True)
            else:
                ts_utc = pd.to_datetime(df.index, utc=True)
            ts_et = ts_utc.dt.tz_convert(ET)
            today_et = datetime.now(ET).date()
            mask = (
                (ts_et.dt.date == today_et)
                & (ts_et.dt.time >= time(9, 30))
                & (ts_et.dt.time < time(16, 0))
            )
            today_bars = df[mask.values]
        except Exception:
            return True, ""  # bar cache shape unexpected — fail open

        if len(today_bars) == 0 or 'high' not in today_bars.columns:
            return True, ""
        day_high = float(today_bars['high'].max())
        if day_high <= 0:
            return True, ""
        pct = price / day_high
        if pct < self.day_from_high_min_pct:
            return False, (
                f"price ${price:.2f} = {pct:.0%} of day_high ${day_high:.2f} "
                f"< floor {self.day_from_high_min_pct:.0%}"
            )
        return True, ""

    def _check_halt_news(self, symbol: str) -> Tuple[bool, str]:
        """Halt-aware filter: skip if news_cache flags a halt today.

        Reads from `news_provider.is_halted_today()` — a sub-millisecond
        SQLite point-query against the (symbol, news_date, halt) index.
        The persisted halt flag is set by `NewsProvider.classify_news()` as
        a side effect of the bull flag scanner's NewsWorker enqueue at
        qualification time, so the cache is warm well before MACD's entry
        signal fires (≥35 bars / ~30 min lead).

        Fail-open if filter disabled, news_provider not wired, or DB error
        — never block legitimate trades on infrastructure issues.

        Returns (passes, reason_if_skip).
        """
        if not self.halt_aware:
            return True, ""
        if self.news_provider is None:
            return True, ""
        try:
            halted, headline = self.news_provider.is_halted_today(symbol)
        except Exception as e:
            logger.warning(
                f"[{self.STRATEGY_NAME}] {symbol}: halt check failed (fail-open): {e}"
            )
            return True, ""
        if halted:
            return False, f"halt detected — '{(headline or '')[:80]}'"
        return True, ""

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

            # Spread gate: skip if bid-ask spread exceeds threshold.
            # Live data (36 trades): <100bps → +$920/trade avg; >=100bps → -$1,621/trade.
            if self.max_entry_spread_bps > 0 and bid > 0 and ask > 0:
                spread_bps = (ask - bid) / limit_price * 10000
                if spread_bps >= self.max_entry_spread_bps:
                    logger.info(
                        f"[{self.STRATEGY_NAME}] {symbol}: SPREAD SKIP — "
                        f"{spread_bps:.0f}bps >= {self.max_entry_spread_bps:.0f}bps "
                        f"(bid=${bid:.2f} ask=${ask:.2f})"
                    )
                    if self.notifier:
                        self.notifier.send_message_sync(
                            f"[MACD Wave] {symbol}: SPREAD SKIP — "
                            f"{spread_bps:.0f}bps >= {self.max_entry_spread_bps:.0f}bps "
                            f"(bid=${bid:.2f} ask=${ask:.2f}, hist={macd_hist_pct:.2f}%)"
                        )
                    self.invalidated.add(symbol)
                    return False

            # 2026-04-28: day-from-high floor. Skip if we're entering far
            # below today's RTH intraday high (a falling-knife archetype).
            # See research/study_macd_day_from_high.py — at 0.70 this is
            # provably inert across 16mo of production signals (min observed
            # 0.727) yet would have caught the 4/28 HTCO entry at 0.22.
            ok, day_high_reason = self._check_day_from_high(crossed.bars_cache, price)
            if not ok:
                logger.info(
                    f"[{self.STRATEGY_NAME}] {symbol}: DAY-HIGH SKIP — {day_high_reason}"
                )
                if self.notifier:
                    self.notifier.send_message_sync(
                        f"[MACD Wave] {symbol}: DAY-HIGH SKIP — {day_high_reason} "
                        f"(hist={macd_hist_pct:.2f}%)"
                    )
                self.invalidated.add(symbol)
                return False

            # 2026-04-28: halt-aware filter. Skip if Alpaca news headlines for
            # this symbol mention a circuit-breaker / volatility halt today —
            # post-halt rebound trades carry asymmetric tail risk (HTCO 4/28).
            ok, halt_reason = self._check_halt_news(symbol)
            if not ok:
                logger.info(
                    f"[{self.STRATEGY_NAME}] {symbol}: HALT SKIP — {halt_reason}"
                )
                if self.notifier:
                    self.notifier.send_message_sync(
                        f"[MACD Wave] {symbol}: HALT SKIP — {halt_reason} "
                        f"(hist={macd_hist_pct:.2f}%)"
                    )
                self.invalidated.add(symbol)
                return False

            # V4 conviction — always compute for logging/DB; scale shares only if enabled.
            conv_mult, conv_brkdn = compute_conviction_score(
                crossed.cross_time_min, crossed.vol_at_cross
            )
            if self.conviction_sizing_enabled:
                effective_position = min(
                    self.position_size * conv_mult,
                    self.max_position_size_usd,
                )
            else:
                effective_position = self.position_size

            shares = int(effective_position / limit_price)
            if shares <= 0:
                return False

            hard_stop = round(limit_price * (1 - self.hard_stop_pct), 2)

            if self.dry_run:
                logger.info(
                    f"[{self.STRATEGY_NAME}] DRY RUN: would BUY {symbol} "
                    f"{shares}sh @ ${limit_price:.2f} (${effective_position:,.0f}), "
                    f"stop ${hard_stop:.2f}, MACD hist {macd_hist_pct:.2f}%, "
                    f"CONVICTION {conv_mult:.2f} "
                    f"(cross=+{conv_brkdn['cross_speed']:.1f} "
                    f"vol=+{conv_brkdn['vol_at_cross']:.1f})"
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
                'pattern_data': (
                    f'{{"cross_time_min": {crossed.cross_time_min}, '
                    f'"vol_at_cross": {crossed.vol_at_cross}, '
                    f'"macd_hist_pct": {macd_hist_pct:.4f}, '
                    f'"conviction_mult": {conv_mult:.3f}, '
                    f'"conv_cross_speed": {conv_brkdn["cross_speed"]:.1f}, '
                    f'"conv_vol_at_cross": {conv_brkdn["vol_at_cross"]:.1f}}}'
                ),
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

            # Anomaly signal: warn when q2s exceeds the threshold so we get
            # an early heads-up on Alpaca/cloud-provider degradation
            # (see 2026-04-15 incident — q2s spiked from 220-450ms to 3.3s).
            # Threshold matches OrderExecutor._SUBMIT_LATENCY_WARN_MS for parity.
            _q2s = micro['quote_to_submit_ms']
            if _q2s is not None and _q2s > 1000:
                logger.warning(
                    f"[{self.STRATEGY_NAME}] {symbol}: SLOW SUBMIT — "
                    f"quote→submit {_q2s}ms > 1000ms threshold. "
                    f"Likely Alpaca/cloud-provider degradation."
                )

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
            self._eod_traded.append({
                'symbol': symbol,
                'shares': shares,
                'effective_position': effective_position,
                'conv_mult': conv_mult,
                'conv_cross_speed': conv_brkdn['cross_speed'],
                'conv_vol_at_cross': conv_brkdn['vol_at_cross'],
            })

            logger.info(
                f"[{self.STRATEGY_NAME}] BUY {symbol} {shares}sh @ ${limit_price:.2f} "
                f"— MACD hist {macd_hist_pct:.2f}%, stop ${hard_stop:.2f}, "
                f"cross {crossed.cross_time_min}min, vol {crossed.vol_at_cross:,}, "
                f"CONVICTION {conv_mult:.2f} "
                f"(cross=+{conv_brkdn['cross_speed']:.1f} vol=+{conv_brkdn['vol_at_cross']:.1f}; "
                f"pos=${effective_position:,.0f})"
            )
            if self.notifier:
                # Telegram reflects the ACTUAL position deployed (scaled by conv_mult
                # when sizing is enabled), not the flat baseline. Otherwise users
                # see flat $ in telegram while DB/broker show scaled.
                conv_note = (
                    f" · conv {conv_mult:.2f}" if self.conviction_sizing_enabled
                    else ""
                )
                self.notifier.send_message_sync(
                    f"[MACD Wave] 📈 BUY {symbol} ${limit_price:.2f} × {shares}sh "
                    f"(${effective_position:,.0f}{conv_note}) — "
                    f"MACD hist {macd_hist_pct:.2f}%, "
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

        # Ask + buffer: buy at ask shifted up by `smart_limit_ask_buffer_bps`.
        # Rationale (from 2026-04-20 post-mortem): on fast momentum entries, the
        # old spread-aware limit (midpoint / bid + 0.75×spread) went stale in
        # the ~1s quote→submit latency because the ask ticked up. USGG 2026-04-20
        # 15:21 UTC missed a +3.6R move because limit was $15.56 and ask ran
        # from $15.58 to $15.60+ during submit. A small buffer above the ask
        # survives 1-2 tick upward moves and matches the 30bps slip budget
        # ORB/bull flag already use.
        limit = ask * (1.0 + self.smart_limit_ask_buffer_bps / 10000.0)
        limit = round(limit, 2)

        info = {
            'bid': bid, 'ask': ask, 'bid_sz': bid_sz, 'ask_sz': ask_sz,
            'spread': spread, 'spread_pct': spread_pct, 'ba_ratio': ba_ratio,
            'pricing': 'ask_plus_buffer',
            'ask_buffer_bps': self.smart_limit_ask_buffer_bps,
            'quote_fetched_at': quote_fetched_at,
        }
        logger.info(
            f"[{self.STRATEGY_NAME}] {symbol}: smart pricing — "
            f"bid=${bid:.2f}×{bid_sz} ask=${ask:.2f}×{ask_sz} "
            f"spread={spread_pct:.2%} ba_ratio={ba_ratio:.1f} "
            f"→ limit=${limit:.2f} (ask+{self.smart_limit_ask_buffer_bps:.0f}bps)"
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
                # Unconfirmed-exit branch: do NOT mutate daily_pnl or write
                # confirmed exit columns. Orphan reconciler picks it up.
                update = build_exit_update(event)
                if event.confirmed:
                    self.daily_pnl += pnl
                    update['pnl'] = pnl
                    update['pnl_pct'] = pnl_pct
                else:
                    logger.error(
                        f"[{self.STRATEGY_NAME}] {sym}: UNCONFIRMED EXIT — "
                        f"order_status=exit_pending_verification, "
                        f"exit_reason={event.exit_reason}. Orphan reconciler "
                        f"will retry. Position remains OPEN on broker."
                    )
                self.db.update_trade(pos.trade_id, update)

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
                                        trail_arm_pct=self.trail_arm_pct,  # CORD 5/8 fix
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
                # MACD via shared detector — same logic the BT uses.
                histogram = compute_macd_histogram(
                    close, self.macd_fast, self.macd_slow, self.macd_signal,
                )

                if histogram.iloc[-1] <= 0:
                    # MACD flipped — remove from StopMonitor FIRST, then exit.
                    # trigger_price = the bar close that caused the flip (telemetry).
                    if self.stop_monitor:
                        self.stop_monitor.remove_watch(sym)
                    self._submit_exit(sym, 'macd_flip',
                                      trigger_price=float(close.iloc[-1]))
                    exits.append(sym)

            except Exception as e:
                logger.error(f"[{self.STRATEGY_NAME}] {sym}: exit check failed: {e}")

        return exits

    def _submit_exit(self, symbol: str, reason: str,
                     trigger_price: Optional[float] = None) -> bool:
        """Submit a sell order.

        Args:
            symbol: Stock to close.
            reason: 'macd_flip' or 'force_close' (used for exit_reason +
                exit_pricing_method).
            trigger_price: Bar close price that triggered the exit decision
                (e.g., the bar whose MACD histogram flipped). Stored for
                post-hoc analysis — distinguishes "what we saw" from "what
                we filled at" (which can drift over the 10s submit→fill).
                None for force_close (no specific price triggered it).
        """
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
            ask = quote.get('ask_price', 0)
            bid_size = quote.get('bid_size', 0)
            ask_size = quote.get('ask_size', 0)
            exit_spread = (ask - bid) if (ask > 0 and bid > 0) else None
            exit_limit_price = bid if bid > 0 else None
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

            # Capture submit timestamp BEFORE the close_position call so we can
            # measure submit→fill latency. Mirrors the bull-flag exit pattern.
            submitted_at_ts = time_mod.time()
            submitted_at_dt = datetime.fromtimestamp(submitted_at_ts, tz=timezone.utc)

            result = self.alpaca.close_position(symbol)
            order_id = result.get('id', '') if result else ''

            fill_latency_ms = (time_mod.time() - submitted_at_ts) * 1000
            # Exit slippage: how far did the actual fill drift from our intended
            # limit (the bid we saw at submit). Positive = we got LESS than
            # planned (sold below our target). Bull-flag uses the same sign
            # convention (exit_limit_price - actual_exit_price).
            exit_slippage = (
                exit_limit_price - exit_price
                if exit_limit_price is not None else None
            )

            # Compute P&L
            pnl = (exit_price - pos.entry_price) * pos.shares
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0
            self.daily_pnl += pnl

            # Update DB — single write for all exit telemetry (atomic per row).
            # Mirrors trading_engine.py:1877-1896 field set so post-hoc analysis
            # can join across both strategies on the same column names.
            self.db.update_trade(pos.trade_id, {
                'exit_price': exit_price,
                'exit_reason': reason,
                'exited_at': datetime.now(timezone.utc),
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                # Exit microstructure (parity with bull_flag stop/TP exits)
                'exit_trigger_price': trigger_price,
                'exit_quote_bid': bid if bid > 0 else None,
                'exit_quote_ask': ask if ask > 0 else None,
                'exit_quote_bid_size': bid_size if bid_size > 0 else None,
                'exit_quote_ask_size': ask_size if ask_size > 0 else None,
                'exit_quote_spread': exit_spread,
                'exit_limit_price': exit_limit_price,
                'exit_pricing_method': f'{reason}_close',
                'exit_submitted_at': submitted_at_dt,
                'exit_fill_latency_ms': fill_latency_ms,
                'exit_slippage': exit_slippage,
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
                    update = build_exit_update(event)
                    if event.confirmed:
                        update['pnl'] = (event.exit_price - pos.entry_price) * pos.shares
                    else:
                        logger.error(
                            f"[{self.STRATEGY_NAME}] {sym}: UNCONFIRMED EXIT "
                            f"during force-close drain — reconciler will retry"
                        )
                    self.db.update_trade(pos.trade_id, update)
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
            # Append conviction stats for sized runs (quiet block when all conv=1.0)
            if self._eod_traded:
                convs = [t['conv_mult'] for t in self._eod_traded]
                positions = [t['effective_position'] for t in self._eod_traded]
                avg_conv = sum(convs) / len(convs)
                max_conv = max(convs)
                min_conv = min(convs)
                total_notional = sum(positions)
                flat_notional = self.position_size * len(self._eod_traded)
                status = "ENABLED (V4)" if self.conviction_sizing_enabled else "disabled"
                msg += (
                    f"\nConviction stats ({status}):"
                    f"\n  avg conv: {avg_conv:.2f}  max: {max_conv:.2f}  min: {min_conv:.2f}"
                    f"\n  notional: ${total_notional:,.0f}"
                    f"  (vs flat ${flat_notional:,.0f})"
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
        self._eod_traded.clear()
        # Drain any stale bar events leftover from yesterday.
        self.drain_bar_events()
        self._bar_queue_full_logged = False
        self._entry_cutoff_logged = False

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _minutes_since_open(now_et: datetime) -> int:
        """Whole minutes elapsed since 09:30 ET on now_et's date.

        Negative before the open. Pure function — no I/O.
        """
        open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        return int((now_et - open_et).total_seconds() / 60)

    def _is_past_entry_cutoff(self, now_et: datetime) -> bool:
        """True when no new MACD-wave entry should be opened — the time-of-day
        gate. Bounds entries to the opening window where the strategy has edge.

        last_entry_minutes_after_open == 0 disables the gate (legacy behaviour).
        """
        if self.last_entry_minutes_after_open <= 0:
            return False
        return (self._minutes_since_open(now_et)
                > self.last_entry_minutes_after_open)

    @staticmethod
    def is_market_open() -> bool:
        """Check if US market is currently open (9:30-16:00 ET)."""
        now = datetime.now(ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close and now.weekday() < 5

    @staticmethod
    def _bar_start_to_close(bar_timestamp_raw) -> datetime:
        """Convert an Alpaca 1-min bar timestamp (bar START) to actual close time.

        Alpaca convention: a bar with `timestamp = 14:04:00` represents trades
        from 14:04:00 to 14:05:00, closing at 14:05:00. Add 60s to get the
        actual close. Accepts pd.Timestamp, datetime, or string-like.
        Always returns a UTC-aware datetime.

        Pre-2026-04-15 the engine stored the bar START in `bar_close_at` (off
        by 60s, inflating bar_close_to_loop_ms by 60_000). This helper is the
        single source of truth for the conversion.
        """
        if isinstance(bar_timestamp_raw, pd.Timestamp):
            bar_start = bar_timestamp_raw.to_pydatetime()
        elif isinstance(bar_timestamp_raw, datetime):
            bar_start = bar_timestamp_raw
        else:
            bar_start = pd.to_datetime(bar_timestamp_raw, utc=True).to_pydatetime()
        if bar_start.tzinfo is None:
            bar_start = bar_start.replace(tzinfo=timezone.utc)
        return bar_start + timedelta(seconds=60)

    def is_force_close_time(self) -> bool:
        """Check if it's time to force close all positions."""
        now = datetime.now(ET)
        return (now.hour > self.force_close_hour or
                (now.hour == self.force_close_hour and now.minute >= self.force_close_minute))
