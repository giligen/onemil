"""HOD-break live engine (2026-09-13) — the executable side of `trading/hod_break.py`.

The scanner feeds it two things: (1) `on_mover(...)` for every symbol the per-minute broad scan
finds >= min_dist_open_pct above its 09:30 open (enqueue-only, never blocks the scan), and
(2) closed 1-minute bars from the StopMonitor websocket stream (`register_on_stop_monitor`,
handler id 'hod_break'). Every closed bar is run through `hod_break.detect` on the day's bars;
a break on the LAST closed bar is actionable and becomes a capped limit BUY with a broker
bracket: stop-loss at the consolidation low, take-profit at limit + target_r × R. The bracket
legs ARE the exits — the engine polls them, records fills, and flattens at flat_minute. Nothing
here depends on a StopMonitor watch (which rejects symbols another strategy already holds).

Gates, in order, at signal time: enabled → kill rails (DB realized P&L, fail-closed) → per-day
cap (DB-derived + pending) → concurrency → once per symbol per day → quote (spread, ask at or
under the cap = no chase) → size. dry_run runs the whole pipeline and logs `[HOD DRY] WOULD BUY`.

Every fallback logs WARNING/ERROR. Restart-safe: `sync_positions()` rebuilds open/pending
state from the trades DB (leg ids live in pattern_data).
"""
from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np

from trading.hod_break import HodBreakParams, detect, shares_for, OPEN_MINUTE

logger = logging.getLogger(__name__)
STRATEGY_NAME = 'hod_break'
ET = ZoneInfo('America/New_York')
_TERMINAL = ('canceled', 'cancelled', 'expired', 'rejected', 'done_for_day', 'suspended')
_OPEN_STATUSES = ('filled', 'partially_filled', 'exit_pending_verification')


@dataclass
class Candidate:
    symbol: str
    day_open: float
    adv20: float
    bars: List[dict] = field(default_factory=list)      # closed RTH 1-min bars (dicts with timestamp/o/h/l/c/v)
    subscribed: bool = False
    backfill_ok: bool = False                             # True once the bar set starts at the 09:30 bar
    backfill_tries: int = 0
    next_idx: int = 0                                     # first bar index detect() has not scanned yet
    rejected_reason: Optional[str] = None
    dry_logged: bool = False


@dataclass
class Position:
    symbol: str
    trade_id: Optional[int]
    order_id: str
    shares: int
    limit_price: float
    stop: float
    target: float
    level: float
    submitted_at: datetime
    tp_leg_id: Optional[str] = None
    sl_leg_id: Optional[str] = None
    fill_price: Optional[float] = None
    filled_at: Optional[datetime] = None
    status: str = 'pending'                               # pending | open
    close_order_id: Optional[str] = None
    close_submitted_at: Optional[datetime] = None


class HodBreakEngine:
    """See module docstring. Construct with `Config().hod_break_cfg`."""
    STRATEGY_NAME = STRATEGY_NAME

    def __init__(self, alpaca_client, db, stop_monitor=None, notifier=None, cfg: Optional[dict] = None, order_stream=None):
        cfg = cfg or {}
        self.alpaca = alpaca_client; self.db = db; self.stop_monitor = stop_monitor; self.notifier = notifier; self.order_stream = order_stream
        self.enabled = bool(cfg.get('enabled', False)); self.dry_run = bool(cfg.get('dry_run', True))
        self.risk_usd = float(cfg.get('risk_usd', 100.0)); self.daily_kill_usd = float(cfg.get('daily_kill_usd', -600.0))
        self.weekly_kill_usd = float(cfg.get('weekly_kill_usd', -1500.0)); self.max_notional_usd = float(cfg.get('max_notional_usd', 5000.0))
        self.min_price = float(cfg.get('min_price', 1.0)); self.min_adv20 = float(cfg.get('min_adv20', 100_000.0))
        self.max_spread_bps = float(cfg.get('max_spread_bps', 100.0)); self.order_timeout_s = float(cfg.get('order_timeout_s', 75.0))
        self.max_spread_frac_r = float(cfg.get('max_spread_frac_r', 0.0))   # 0 = off; e.g. 0.15 = skip when the spread is > 15% of R (9/14: 57% of signals)
        self.params = HodBreakParams(**(cfg.get('params') or {}))
        self.candidates: Dict[str, Candidate] = {}; self.positions: Dict[str, Position] = {}
        self.entered_today: set = set(); self.daily_pnl = 0.0; self.session_date: Optional[str] = None
        self._mover_queue: queue.Queue = queue.Queue(maxsize=5000); self._bar_queue: queue.Queue = queue.Queue(maxsize=5000)
        self._adv_map: Dict[str, float] = {}; self._kill_notified: set = set(); self._flattened = False
        self.shutdown_requested = False; self._lock = threading.RLock()   # tick (engine pool) and drains (main thread) must not interleave
        self.seen_today: set = set()                                      # once-per-symbol (orders incl. no-fills); entered_today = the day-cap set (fills/working orders)
        logger.info(f"[HOD] engine gates: enabled={self.enabled} dry_run={self.dry_run} risk=${self.risk_usd:.0f} "
                    f"kills={self.daily_kill_usd}/{self.weekly_kill_usd} cap={self.params.cap:.2%} target={self.params.target_r}R "
                    f"per_day={self.params.max_per_day} concurrent={self.params.max_concurrent} flat={self.params.flat_minute}")

    # ------------------------------------------------------------------ clock / session
    def _et_now(self) -> datetime:
        return datetime.now(timezone.utc).astimezone(ET)

    def _minute_of_day(self) -> int:
        n = self._et_now(); return n.hour * 60 + n.minute

    def _roll_session(self) -> None:
        today = self._et_now().strftime('%Y-%m-%d')
        if self.session_date != today:
            self.session_date = today; self.candidates.clear(); self.entered_today.clear(); self.daily_pnl = 0.0
            self._kill_notified.clear(); self._flattened = False
            self._adv_map = self._load_adv_map()
            logger.info(f"[HOD] session {today}: adv map {len(self._adv_map)} symbols")

    def _load_adv_map(self) -> Dict[str, float]:
        try:
            rows = self.db.get_active_universe()
            return {r['symbol']: float(r.get('avg_volume_daily') or 0.0) for r in rows}
        except Exception as e:
            logger.error(f"[HOD] universe ADV map unavailable ({e}) — every mover will fail the ADV gate today")
            return {}

    # ------------------------------------------------------------------ scanner hooks (enqueue-only)
    def on_mover(self, symbol: str, *, price: float, day_open: float, cum_volume: float, above_open_pct: float, ts=None) -> None:
        """Called from the per-minute broad scan for symbols >= min_dist_open_pct above the open. Never blocks."""
        if not self.enabled:
            return
        try:
            self._mover_queue.put_nowait((symbol, float(price), float(day_open), float(cum_volume), float(above_open_pct)))
        except queue.Full:
            logger.warning(f"[HOD] mover queue full — dropped {symbol}")

    def register_on_stop_monitor(self) -> bool:
        if self.stop_monitor is None or getattr(self.stop_monitor, 'polling_mode', False):
            logger.warning("[HOD] no websocket StopMonitor — bar stream unavailable, engine cannot detect breaks")
            return False
        self.stop_monitor.register_bar_handler(STRATEGY_NAME, self._on_bar_close)
        return True

    def _on_bar_close(self, symbol: str, bars_df) -> None:
        """WS thread: zero work, enqueue only."""
        if not self.enabled or symbol not in self.candidates:
            return
        try:
            self._bar_queue.put_nowait((symbol, bars_df))
        except queue.Full:
            logger.error(f"[HOD] bar queue full — dropped a bar for {symbol}")

    # ------------------------------------------------------------------ main-thread work
    def process_tick(self) -> None:
        """Called every scanner cycle (unconditionally — the shared force-close latch must not stop us)."""
        if not self.enabled:
            return
        with self._lock:
            try:
                self._roll_session()
                self._admit_movers()
                self.drain_bar_events()
                self._process_pending_fills()
                self.check_exits()
                if self.is_force_close_time() and (not self._flattened or any(p.status == 'open' for p in self.positions.values())):
                    self.force_close_all()
            except Exception as e:
                logger.error(f"[HOD] process_tick failed: {e}", exc_info=True)

    def _admit_movers(self) -> None:
        n = 0; new: List[Candidate] = []
        while not self._mover_queue.empty() and n < 500:
            symbol, price, day_open, cum_vol, above = self._mover_queue.get_nowait(); n += 1
            if symbol in self.candidates or symbol in self.seen_today or symbol in self.positions:
                continue
            if not self.candidates and not self.seen_today:            # first admission of the session: pull the DB set once
                db_syms = self._db_symbols_today(); self.seen_today |= db_syms; self.entered_today |= db_syms
                if symbol in self.seen_today: continue
            adv = self._adv_map.get(symbol, 0.0)
            if price < self.min_price or adv < self.min_adv20 or day_open <= 0:
                continue
            cand = Candidate(symbol=symbol, day_open=day_open, adv20=adv)
            self.candidates[symbol] = cand; new.append(cand)
            logger.info(f"[HOD] candidate {symbol} admitted: {price:.2f} +{above:.1f}% from open {day_open:.2f}, adv20 {adv:,.0f} (#{len(self.candidates)} today)")
            self._subscribe(cand)
        retry = [c for c in self.candidates.values() if c.subscribed and not c.backfill_ok and c.rejected_reason is None]
        self._backfill([c for c in new if c.subscribed] + [c for c in retry if c not in new])

    def _subscribe(self, cand: Candidate) -> None:
        try:
            if self.stop_monitor is not None and hasattr(self.stop_monitor, 'subscribe_bars'):
                self.stop_monitor.subscribe_bars(cand.symbol); cand.subscribed = True
        except Exception as e:
            logger.error(f"[HOD] {cand.symbol}: bar subscribe failed ({e}) — candidate dropped")
            self.candidates.pop(cand.symbol, None)

    def _backfill(self, cands: List[Candidate]) -> None:
        """ONE batched REST call for every candidate that still lacks the 09:30 open; a symbol missing from the
        result is an ERROR and is retried next tick (never evaluated on stream-only bars — the DBI-09:50 class)."""
        cands = [c for c in cands if not c.backfill_ok]
        if not cands: return
        try:
            got = self.alpaca.get_1min_bars_multi([c.symbol for c in cands], lookback_minutes=max(30, self._minute_of_day() - OPEN_MINUTE + 5))
        except Exception as e:
            logger.error(f"[HOD] backfill call failed for {len(cands)} candidates ({e}) — retry next tick"); return
        for c in cands:
            df = (got or {}).get(c.symbol)
            if df is None or not len(df):
                c.backfill_tries += 1
                if c.backfill_tries in (1, 5): logger.error(f"[HOD] {c.symbol}: backfill returned no bars (try {c.backfill_tries}) — not evaluated until the 09:30 open is present")
                continue
            self._ingest_bars(c.symbol, df)

    def drain_bar_events(self) -> List[str]:
        touched = []
        with self._lock:
            while not self._bar_queue.empty():
                symbol, df = self._bar_queue.get_nowait()
                if symbol in self.candidates:
                    self._ingest_bars(symbol, df); touched.append(symbol)
        return touched

    @staticmethod
    def _rth_arrays(bars: List[dict]):
        rows = []
        for b in bars:
            ts = b.get('timestamp')
            t = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
            if t.tzinfo is None: t = t.replace(tzinfo=timezone.utc)
            et = t.astimezone(ET); m = et.hour * 60 + et.minute
            if OPEN_MINUTE <= m < 960:
                rows.append((m, float(b['open']), float(b['high']), float(b['low']), float(b['close']), float(b.get('volume') or 0)))
        rows.sort(key=lambda r: r[0])
        if not rows:
            return None
        a = np.array(rows, dtype=float)
        return a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 0].astype(int)

    @staticmethod
    def _bar_key(b: dict):
        ts = b.get('timestamp')
        t = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t.astimezone(timezone.utc).replace(second=0, microsecond=0)

    def _ingest_bars(self, symbol: str, bars_df) -> None:
        """MERGE by minute — never replace. Found 2026-09-14 09:50 (DBI): the stream's DataFrame holds
        only bars since SUBSCRIPTION, so replacing the backfilled day once the stream list grew longer
        dropped the 09:30 open, the early high-of-day and the early volume → wrong open, wrong HOD,
        wrong relative volume. The union keeps every bar seen from either source."""
        cand = self.candidates.get(symbol)
        if cand is None:
            return
        try:
            recs = bars_df.to_dict('records') if hasattr(bars_df, 'to_dict') else list(bars_df)
            merged = {self._bar_key(b): b for b in cand.bars}
            for b in recs:
                merged[self._bar_key(b)] = b
            cand.bars = [merged[k] for k in sorted(merged)]
        except Exception as e:
            logger.error(f"[HOD] {symbol}: bad bar payload ({e})"); return
        self._evaluate(cand)

    def _evaluate(self, cand: Candidate) -> None:
        if cand.rejected_reason or cand.symbol in self.positions or cand.symbol in self.entered_today:
            return
        arr = self._rth_arrays(cand.bars)
        if arr is None:
            return
        o, h, l, c, v, m = arr; n = len(o)
        if int(m[0]) != OPEN_MINUTE and abs(float(o[0]) - cand.day_open) > 0.011:
            return                                         # the day's first bar is missing: wrong open/HOD/rv — wait for the backfill
        cand.backfill_ok = True
        start = cand.next_idx
        while True:
            sig = detect(o, h, l, v, m, cand.adv20, self.params, start_idx=start)
            if sig is None:
                cand.next_idx = n; return
            if sig.bar_idx < n - 1:
                # a break that already passed (backfill / late bar): its next-open fill is gone — skip, keep scanning
                start = sig.bar_idx + 1; continue
            cand.next_idx = n
            self._try_enter(cand, sig, o[0]); return

    # ------------------------------------------------------------------ entry
    def _try_enter(self, cand: Candidate, sig, day_open: float) -> None:
        p = self.params; sym = cand.symbol
        blocked = self._kill_rails_blocked()
        if blocked:
            self._notify_once(blocked, f"[HOD] {blocked}: no new entries"); return
        if self._entered_today_count() >= p.max_per_day:
            cand.rejected_reason = 'day_cap'; logger.info(f"[HOD] {sym}: per-day cap {p.max_per_day} reached — skip"); return
        if len(self.positions) >= p.max_concurrent:
            logger.info(f"[HOD] {sym}: concurrency cap {p.max_concurrent} — skip (signal not re-armed)"); cand.rejected_reason = 'concurrency'; return
        limit = round(sig.level * (1.0 + p.cap), 2); stop = round(sig.stop, 2)
        if stop >= limit:
            cand.rejected_reason = 'r_min'; return
        q = self._quote(sym)
        if q is None:
            cand.rejected_reason = 'no_quote'; logger.warning(f"[HOD] {sym}: no quote — fail closed, no order"); return
        bid, ask = q
        spread_bps = (ask - bid) / ask * 1e4 if ask > 0 else 1e9
        if spread_bps > self.max_spread_bps:
            cand.rejected_reason = 'spread'; logger.info(f"[HOD] {sym}: spread {spread_bps:.0f} bps > {self.max_spread_bps:.0f} — skip"); return
        if ask > limit:
            cand.rejected_reason = 'no_chase'; logger.info(f"[HOD] {sym}: ask {ask:.2f} above cap {limit:.2f} — NO CHASE, skip"); return
        # R, the r_min gate and the size use the EXPECTED FILL = the ask (the spec uses the next bar's open, the price
        # actually paid); measuring against the limit over-states R and let tight stops through (9/14 EOD parity: 11 of 35).
        entry_est = round(ask, 2)
        r = entry_est - stop
        if r <= 0 or r / entry_est * 100.0 < p.min_r_pct:
            cand.rejected_reason = 'r_min'; logger.info(f"[HOD] {sym}: stop {stop:.2f} within {p.min_r_pct}% of the ask {entry_est:.2f} — skip"); return
        if self.max_spread_frac_r > 0 and (ask - bid) / r > self.max_spread_frac_r:
            cand.rejected_reason = 'spread_r'; logger.info(f"[HOD] {sym}: spread {spread_bps:.0f} bps = {(ask - bid) / r:.0%} of R {r:.2f} > {self.max_spread_frac_r:.0%} — skip"); return
        target = round(entry_est + p.target_r * r, 2)      # target from the EXPECTED fill (the spec's basis); a fill at the limit makes the real target slightly < 2R
        shares = shares_for(self.risk_usd, entry_est, stop)
        shares = min(shares, int(self.max_notional_usd // limit))
        if shares < 1:
            cand.rejected_reason = 'size'; return
        msg = (f"{sym} level {sig.level:.2f} limit {limit:.2f} stop {stop:.2f} target {target:.2f} R {r:.2f} ({r / entry_est * 100:.1f}%) "
               f"x{shares} | +{sig.dist_open_pct:.1f}% from open, rv {sig.rv_profile:.1f}, spread {spread_bps:.0f} bps, ask {entry_est:.2f}")
        if self.dry_run:
            if not cand.dry_logged:
                cand.dry_logged = True; cand.rejected_reason = 'dry_run'
                logger.info(f"[HOD DRY] WOULD BUY {msg}"); self._notify(f"[HOD DRY] WOULD BUY {msg}")
            return
        try:
            od = self.alpaca.submit_bracket_order(symbol=sym, qty=shares, side='buy', limit_price=limit, tp_price=target, sl_price=stop)
        except Exception as e:
            od = self._adopt_open_buy(sym)                     # a client-side timeout may have left an accepted order at the broker
            if od is None:
                cand.rejected_reason = 'submit_failed'; logger.error(f"[HOD] {sym}: submit failed: {e}"); self._notify(f"[HOD] ERROR submit {sym}: {e}"); return
            logger.warning(f"[HOD] {sym}: submit raised ({e}) but an open BUY order exists at the broker — adopted {od.get('id')}")
        order_id = str(od.get('id', '')); tp_id = sl_id = None
        for leg in od.get('legs') or []:
            if leg.get('limit_price') is not None and leg.get('stop_price') is None: tp_id = str(leg.get('id'))
            elif leg.get('stop_price') is not None: sl_id = str(leg.get('id'))
        if not order_id:
            logger.error(f"[HOD] {sym}: submit returned no order id — untracked order possible"); self._notify(f"[HOD] ERROR {sym}: order id missing")
        now = datetime.now(timezone.utc)
        trade_id = self._save_pending_trade(sym, shares, limit, stop, target, order_id, tp_id, sl_id, sig, bid, ask)
        self.positions[sym] = Position(symbol=sym, trade_id=trade_id, order_id=order_id, shares=shares, limit_price=limit, stop=stop, target=target,
                                       level=sig.level, submitted_at=now, tp_leg_id=tp_id, sl_leg_id=sl_id)
        self.entered_today.add(sym); self.seen_today.add(sym); cand.rejected_reason = 'ordered'
        logger.info(f"[HOD] ENTRY SUBMITTED {msg} order {order_id}"); self._notify(f"[HOD] BUY {msg}")

    def _adopt_open_buy(self, symbol: str) -> Optional[dict]:
        try:
            for od in self.alpaca.get_open_orders() or []:
                if od.get('symbol') == symbol and str(od.get('side', '')).lower() == 'buy':
                    return od
        except Exception as e:
            logger.warning(f"[HOD] {symbol}: open-orders reconcile failed ({e})")
        return None

    def _quote(self, symbol: str, tries: int = 3):
        for k in range(tries):
            try:
                q = self.alpaca.get_latest_quote(symbol)
                bid = float(q.get('bid_price') or 0); ask = float(q.get('ask_price') or 0)
                if bid > 0 and ask > 0: return (bid, ask)
            except Exception as e:
                logger.warning(f"[HOD] {symbol}: quote failed ({e}) try {k + 1}/{tries}")
            if k + 1 < tries: time.sleep(0.5)
        return None

    def _save_pending_trade(self, sym, shares, limit, stop, target, order_id, tp_id, sl_id, sig, bid, ask) -> Optional[int]:
        rec = {
            'trade_date': self.session_date or self._et_now().strftime('%Y-%m-%d'), 'symbol': sym, 'side': 'buy', 'entry_price': limit,
            'stop_loss_price': stop, 'take_profit_price': target, 'shares': shares, 'risk_per_share': limit - stop, 'total_risk': (limit - stop) * shares,
            'risk_reward_ratio': self.params.target_r, 'order_id': order_id, 'order_status': 'pending_new', 'fill_price': None, 'filled_at': None,
            'exit_price': None, 'exit_reason': None, 'exited_at': None, 'pnl': None, 'pnl_pct': None, 'strategy': STRATEGY_NAME,
            'pattern_data': json.dumps({'level': sig.level, 'consol_low': sig.stop, 'dist_open_pct': sig.dist_open_pct, 'rv_profile': sig.rv_profile,
                                        'cap': self.params.cap, 'target_r': self.params.target_r, 'tp_leg_id': tp_id, 'sl_leg_id': sl_id,
                                        'quote_bid': bid, 'quote_ask': ask, 'limit': limit, 'target': target}),
        }
        try:
            return int(self.db.save_trade(rec))
        except Exception as e:
            logger.error(f"[HOD] {sym}: save_trade failed ({e}) — order {order_id} is NOT in the DB"); self._notify(f"[HOD] ERROR DB {sym}: {e}"); return None

    # ------------------------------------------------------------------ fills / exits
    def _order_status(self, order_id: str, rest: bool = False) -> Optional[dict]:
        """Order status: the stream cache only while the stream is healthy (a stalled stream must not mask a
        fill), REST otherwise or when `rest=True` (always after a cancel)."""
        st = None
        if not rest and self.order_stream is not None:
            try:
                healthy = self.order_stream.is_healthy() if hasattr(self.order_stream, 'is_healthy') else True
                st = self.order_stream.get_status(order_id) if healthy else None
            except Exception: st = None
        if st is None:
            try: st = self.alpaca.get_order(order_id)
            except Exception as e:
                logger.warning(f"[HOD] get_order {order_id} failed ({e})"); return None
        return st

    def _process_pending_fills(self) -> None:
        for sym, pos in list(self.positions.items()):
            if pos.status != 'pending':
                continue
            st = self._order_status(pos.order_id)
            status = str((st or {}).get('status', '')).lower(); fq = int((st or {}).get('filled_qty') or 0)
            age = (datetime.now(timezone.utc) - pos.submitted_at).total_seconds()
            if status == 'filled':
                self._confirm_fill(pos, st); continue
            if status in _TERMINAL:
                if fq > 0: self._confirm_fill(pos, st)
                else: self._drop_pending(pos, status)
                continue
            if age >= self.order_timeout_s:
                try: self.alpaca.cancel_order(pos.order_id)
                except Exception as e: logger.warning(f"[HOD] {sym}: cancel failed ({e})")
                st2 = self._order_status(pos.order_id, rest=True) or {}
                s2 = str(st2.get('status', '')).lower()
                if int(st2.get('filled_qty') or 0) > 0: self._confirm_fill(pos, st2)
                elif s2 in _TERMINAL: self._drop_pending(pos, 'time_stop_canceled'); logger.info(f"[HOD] {sym}: unfilled after {age:.0f}s — canceled (no chase)")
                else: logger.warning(f"[HOD] {sym}: cancel not confirmed (status {s2 or 'unknown'}) — order kept as pending, retry next tick")

    def _drop_pending(self, pos: Position, status: str) -> None:
        self.positions.pop(pos.symbol, None); self.entered_today.discard(pos.symbol)   # a no-fill does not consume a day slot (once-per-symbol stays)
        if pos.trade_id is not None:
            try: self.db.update_trade(pos.trade_id, {'order_status': status})
            except Exception as e: logger.error(f"[HOD] {pos.symbol}: DB update ({status}) failed: {e}")

    def _confirm_fill(self, pos: Position, st: dict) -> None:
        fq = int(st.get('filled_qty') or pos.shares) or pos.shares
        px = float(st.get('filled_avg_price') or pos.limit_price)
        pos.fill_price = px; pos.filled_at = datetime.now(timezone.utc); pos.shares = fq; pos.status = 'open'
        if pos.trade_id is not None:
            try: self.db.update_trade(pos.trade_id, {'order_status': 'filled', 'fill_price': px, 'filled_at': pos.filled_at.isoformat(), 'shares': fq})
            except Exception as e: logger.error(f"[HOD] {pos.symbol}: DB fill update failed: {e}")
        logger.info(f"[HOD] FILLED {pos.symbol} x{fq} @ {px:.2f} (limit {pos.limit_price:.2f}, slip {(px / pos.level - 1) * 1e4:.0f} bps vs level)")
        self._notify(f"[HOD] FILLED {pos.symbol} x{fq} @ {px:.2f} stop {pos.stop:.2f} target {pos.target:.2f}")

    def check_exits(self) -> List[str]:
        """Poll the bracket legs (and the force-close order) of open positions; record exits."""
        done = []
        for sym, pos in list(self.positions.items()):
            if pos.status != 'open':
                continue
            for leg_id, reason in ((pos.close_order_id, 'eod'), (pos.tp_leg_id, 'target'), (pos.sl_leg_id, 'stop')):
                if not leg_id:
                    continue
                st = self._order_status(leg_id)
                if st and str(st.get('status', '')).lower() == 'filled' and int(st.get('filled_qty') or 0) > 0:
                    self._record_exit(pos, float(st.get('filled_avg_price') or 0.0), reason); done.append(sym)
                    for other in (pos.tp_leg_id, pos.sl_leg_id):          # belt and braces: the OCO sibling must be dead
                        if other and other != leg_id:
                            try: self.alpaca.cancel_order(other)
                            except Exception: pass
                    break
        return done

    def _record_exit(self, pos: Position, exit_price: float, reason: str) -> None:
        self.positions.pop(pos.symbol, None)
        entry = pos.fill_price or pos.limit_price
        pnl = (exit_price - entry) * pos.shares if exit_price > 0 else None
        upd = {'order_status': 'closed', 'exit_price': exit_price, 'exit_reason': reason, 'exited_at': datetime.now(timezone.utc).isoformat()}
        if pnl is not None:
            upd['pnl'] = pnl; upd['pnl_pct'] = (exit_price / entry - 1) * 100; self.daily_pnl += pnl
        else:
            upd['order_status'] = 'exit_pending_verification'; logger.warning(f"[HOD] {pos.symbol}: exit price unknown — pending verification")
        if pos.trade_id is not None:
            try: self.db.update_trade(pos.trade_id, upd)
            except Exception as e: logger.error(f"[HOD] {pos.symbol}: DB exit update failed: {e}")
        rr = (exit_price - entry) / (entry - pos.stop) if entry > pos.stop and exit_price > 0 else float('nan')
        logger.info(f"[HOD] EXIT {pos.symbol} {reason} @ {exit_price:.2f} pnl {pnl if pnl is None else round(pnl, 2)} ({rr:+.2f}R) day {self.daily_pnl:+.0f}")
        self._notify(f"[HOD] EXIT {pos.symbol} {reason} @ {exit_price:.2f} pnl {'?' if pnl is None else f'{pnl:+.0f}'} ({rr:+.2f}R) | day {self.daily_pnl:+.0f}")

    # ------------------------------------------------------------------ force close
    def is_force_close_time(self) -> bool:
        return self._minute_of_day() >= self.params.flat_minute

    FC_RESUBMIT_S = 60.0

    def force_close_all(self) -> int:
        """Cancel pending entries; cancel legs; sell OUR shares (never `close_position`, which would liquidate a
        position the owner or another strategy holds in the same symbol) with a marketable limit; re-check every
        tick and re-submit until every position is gone. `_flattened` only when nothing is left."""
        n = 0; now = datetime.now(timezone.utc)
        for sym, pos in list(self.positions.items()):
            if pos.status == 'pending':
                try: self.alpaca.cancel_order(pos.order_id)
                except Exception as e: logger.warning(f"[HOD] FC cancel {sym} failed: {e}")
                st = self._order_status(pos.order_id) or {}
                if int(st.get('filled_qty') or 0) > 0: self._confirm_fill(pos, st)
                else: self._drop_pending(pos, 'time_stop_canceled'); continue
            if pos.close_order_id:
                st = self._order_status(pos.close_order_id) or {}
                status = str(st.get('status', '')).lower(); age = (now - (pos.close_submitted_at or now)).total_seconds()
                if status in ('filled',) or (status in _TERMINAL and int(st.get('filled_qty') or 0) >= pos.shares):
                    continue                                       # check_exits records it
                if status not in _TERMINAL and age < self.FC_RESUBMIT_S:
                    continue                                       # still working
                try: self.alpaca.cancel_order(pos.close_order_id)
                except Exception: pass
                logger.warning(f"[HOD] FORCE CLOSE {sym}: close order {pos.close_order_id} {status or 'unknown'} after {age:.0f}s — re-submitting")
            for leg in (pos.tp_leg_id, pos.sl_leg_id):
                if leg:
                    try: self.alpaca.cancel_order(leg)
                    except Exception as e: logger.warning(f"[HOD] FC leg cancel {sym} failed: {e}")
            time.sleep(0.5)
            q = self._quote(sym); ref = q[0] if q else (pos.fill_price or pos.limit_price)
            limit = round(ref * 0.99, 2)
            try:
                od = self.alpaca.submit_limit_sell_order(sym, pos.shares, limit)
                pos.close_order_id = str(od.get('id') or '') or None; pos.close_submitted_at = now; n += 1
                logger.info(f"[HOD] FORCE CLOSE {sym} x{pos.shares} limit {limit:.2f} submitted ({pos.close_order_id})")
            except Exception as e:
                logger.error(f"[HOD] FORCE CLOSE {sym} FAILED: {e}"); self._notify(f"[HOD] ERROR force close {sym}: {e}")
        remaining = [s_ for s_, p_ in self.positions.items() if p_.status == 'open']
        self._flattened = not remaining
        if n: self._notify(f"[HOD] flat at {self.params.flat_minute // 60:02d}:{self.params.flat_minute % 60:02d} ET — {n} close order(s) submitted, {len(remaining)} still open")
        return n

    # ------------------------------------------------------------------ rails / caps
    def _realized_pnl(self, since: str) -> float:
        path = getattr(self.db, '_trades_path', None)
        if not path:
            logger.warning("[HOD] realized-pnl: db has no _trades_path — FAIL CLOSED"); return -1e9
        try:
            conn = sqlite3.connect(str(path), timeout=10)
            try:
                v = conn.execute("SELECT COALESCE(SUM(pnl),0) FROM trades WHERE strategy=? AND trade_date>=?", (STRATEGY_NAME, since)).fetchone()[0]
            finally: conn.close()
            return float(v or 0.0)
        except Exception as e:
            logger.error(f"[HOD] realized-pnl query failed ({e}) — FAIL CLOSED"); return -1e9

    def _kill_rails_blocked(self) -> Optional[str]:
        now = self._et_now(); today = now.strftime('%Y-%m-%d'); week = (now - timedelta(days=now.weekday())).strftime('%Y-%m-%d')
        if self._realized_pnl(week) <= self.weekly_kill_usd: return 'weekly_kill'
        if self._realized_pnl(today) <= self.daily_kill_usd: return 'daily_kill'
        return None

    _DEAD = ('canceled', 'cancelled', 'expired', 'rejected', 'time_stop_canceled')

    def _db_symbols_today(self) -> set:
        """Symbols with ANY non-dead hod_break row today (open, closed, pending) — the restart-safe source of
        truth for the per-day cap and once-per-symbol. Returns an empty set (memory only) on DB failure."""
        today = self.session_date or self._et_now().strftime('%Y-%m-%d'); path = getattr(self.db, '_trades_path', None)
        if not path: return set()
        try:
            conn = sqlite3.connect(str(path), timeout=10)
            try:
                rows = conn.execute("SELECT DISTINCT symbol FROM trades WHERE strategy=? AND trade_date=? AND COALESCE(order_status,'') NOT IN (%s)"
                                    % ','.join('?' * len(self._DEAD)), (STRATEGY_NAME, today, *self._DEAD)).fetchall()
            finally: conn.close()
            return {r[0] for r in rows}
        except Exception as e:
            logger.warning(f"[HOD] symbols-today DB query failed ({e}) — using memory only"); return set()

    def _entered_today_count(self) -> int:
        db_syms = self._db_symbols_today(); self.entered_today |= db_syms
        return len(self.entered_today)

    # ------------------------------------------------------------------ restart
    def sync_positions(self) -> int:
        """Rebuild pending/open positions from the trades DB for today (restart-safe)."""
        self._roll_session(); today = self.session_date; n = 0
        try: rows = self.db.get_open_trades(today, strategy=STRATEGY_NAME)
        except Exception as e:
            logger.error(f"[HOD] sync_positions: DB read failed ({e})"); return 0
        broker: Dict[str, int] = {}
        try: broker = {p.get('symbol'): int(float(p.get('qty') or 0)) for p in (self.alpaca.get_open_positions() or [])}
        except Exception as e: logger.warning(f"[HOD] sync_positions: broker positions unavailable ({e})")
        for r in rows:
            sym = r['symbol']; pd_ = {}
            try: pd_ = json.loads(r.get('pattern_data') or '{}')
            except Exception: pass
            pos = Position(symbol=sym, trade_id=r.get('id'), order_id=str(r.get('order_id') or ''), shares=int(r.get('shares') or 0), limit_price=float(r.get('entry_price') or 0),
                           stop=float(r.get('stop_loss_price') or 0), target=float(r.get('take_profit_price') or 0), level=float(pd_.get('level') or 0),
                           submitted_at=datetime.now(timezone.utc) - timedelta(seconds=self.order_timeout_s),   # a restored pending order is past its window: the next poll cancels-or-confirms it
                           tp_leg_id=pd_.get('tp_leg_id'), sl_leg_id=pd_.get('sl_leg_id'), fill_price=r.get('fill_price'))
            status = r.get('order_status')
            if status == 'pending_new':
                pos.status = 'pending'
            elif status in _OPEN_STATUSES:
                pos.status = 'open'
                if broker and broker.get(sym, 0) < int(r.get('shares') or 0):
                    logger.warning(f"[HOD] sync: {sym} open in DB ({r.get('shares')} sh) but broker holds {broker.get(sym, 0)} — exit pending verification")
                    try: self.db.update_trade(r['id'], {'order_status': 'exit_pending_verification'})
                    except Exception: pass
                    continue
            else:
                continue
            self.positions[sym] = pos; self.entered_today.add(sym); self.seen_today.add(sym); n += 1
        logger.info(f"[HOD] sync_positions: {n} position(s) rehydrated for {today}")
        return n

    # ------------------------------------------------------------------ notify
    def _notify(self, msg: str) -> None:
        if not self.notifier: return
        try:
            send = getattr(self.notifier, 'send_message', None)
            if send is None: return
            import asyncio
            res = send(msg)
            if asyncio.iscoroutine(res):
                try: loop = asyncio.get_event_loop(); loop.run_until_complete(res)
                except RuntimeError: asyncio.run(res)
        except Exception as e:
            logger.debug(f"[HOD] notifier failed (non-critical): {e}")

    def _notify_once(self, key: str, msg: str) -> None:
        if key in self._kill_notified: return
        self._kill_notified.add(key); logger.warning(msg); self._notify(msg)
