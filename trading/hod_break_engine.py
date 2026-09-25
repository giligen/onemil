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
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np

from trading.hod_break import HodBreakParams, arm_state, detect, resting_entry_fill, resting_order_qty, shares_for, OPEN_MINUTE
from trading.red_to_green import RedToGreenParams, detect as r2g_detect, prior_day_range_pct

logger = logging.getLogger(__name__)
STRATEGY_NAME = 'hod_break'
ET = ZoneInfo('America/New_York')
_TERMINAL = ('canceled', 'cancelled', 'expired', 'rejected', 'done_for_day', 'suspended')
_OPEN_STATUSES = ('filled', 'partially_filled', 'exit_pending_verification')


STALE_DAYS = 7               # a symbol whose last daily bar is older than this is dead/halted/delisted: not streamed, not admitted


def load_adv20_from_daily_bars(cache_path, min_rows: int = 5, stale_days: int = STALE_DAYS):
    """ADV20 = mean volume of the latest 20 daily_bars rows within 45 days (>= min_rows rows), plus each symbol's last
    close. Symbols whose LAST bar is older than `stale_days` (no trades for a week: acquired, delisted, halted) are
    dropped from both maps — they would only produce empty backfills. ONE definition for the engine's universe/ADV gate
    and the miss audit (the spec's ADV20 is the same rolling mean)."""
    adv: Dict[str, float] = {}; last: Dict[str, float] = {}
    conn = sqlite3.connect(f'file:{cache_path}?mode=ro', uri=True, timeout=30)
    try:
        q = ("with d as (select symbol, bar_date, volume, close, row_number() over (partition by symbol order by bar_date desc) rn "
             "from daily_bars where bar_date >= date('now', '-45 days') and bar_date < ?) "        # T-1..T-20 only: never today's (provisional) row
             "select symbol, avg(volume), count(*), max(case when rn = 1 then close end), max(bar_date) from d where rn <= 20 group by symbol")
        today = datetime.now(timezone.utc).astimezone(ET).strftime('%Y-%m-%d')
        for sym, a, cnt, lc, last_date in conn.execute(q, (today,)):
            if not last_date or (datetime.now(timezone.utc).date() - datetime.strptime(str(last_date)[:10], '%Y-%m-%d').date()).days > stale_days:
                continue
            if cnt and cnt >= min_rows and a: adv[sym] = float(a)
            if lc: last[sym] = float(lc)
    finally:
        conn.close()
    return adv, last


def load_prev_day_from_daily_bars(cache_path, stale_days: int = STALE_DAYS) -> Dict[str, tuple]:
    """Each symbol's PRIOR session (close, high, low) from daily_bars — the red-to-green book's prior close and prior-day
    range (ORB's PDR helper computes the same (high-low)/low). Never today's provisional row; stale names dropped."""
    out: Dict[str, tuple] = {}
    conn = sqlite3.connect(f'file:{cache_path}?mode=ro', uri=True, timeout=30)
    try:
        today = datetime.now(timezone.utc).astimezone(ET).strftime('%Y-%m-%d')
        q = ("with d as (select symbol, bar_date, close, high, low, row_number() over (partition by symbol order by bar_date desc) rn "
             "from daily_bars where bar_date >= date('now', '-20 days') and bar_date < ?) select symbol, close, high, low, bar_date from d where rn = 1")
        for sym, c, h, l, last_date in conn.execute(q, (today,)):
            if not last_date or (datetime.now(timezone.utc).date() - datetime.strptime(str(last_date)[:10], '%Y-%m-%d').date()).days > stale_days:
                continue
            if c and h and l: out[sym] = (float(c), float(h), float(l))
    finally:
        conn.close()
    return out


RTH_MINUTES = 960 - OPEN_MINUTE      # 390 one-minute slots, 09:30 .. 15:59 ET


@dataclass
class Candidate:
    """One symbol-day. Bars live in a fixed (390 × 5) array indexed by minute-of-session — O(1) merge per bar, no
    per-bar dict/DataFrame churn (3,500 streamed names × 390 minutes must fit in ~70 MB, not a gigabyte)."""
    symbol: str
    day_open: float
    adv20: float
    subscribed: bool = False
    prior_close: float = 0.0                              # red-to-green book: the prior session close (level base)
    pdr_pct: Optional[float] = None                       # red-to-green book: prior-day (high-low)/low in %
    backfill_ok: bool = False                             # True once the bar set starts at the 09:30 bar
    needs_refill: bool = False                            # set by an outage / a dropped bar; cleared ONLY by a non-empty REST backfill
    reconciled: bool = False                              # the day's streamed bars were merged with REST once (review A: a silently dropped early bar)
    backfill_tries: int = 0
    next_idx: int = 0                                     # first bar index detect() has not scanned yet
    rejected_reason: Optional[str] = None
    dry_logged: bool = False
    ohlcv: np.ndarray = field(default_factory=lambda: np.full((RTH_MINUTES, 5), np.nan))
    have: np.ndarray = field(default_factory=lambda: np.zeros(RTH_MINUTES, dtype=bool))
    resting_arm: Optional[dict] = None                     # current arm dict(level, trigger, limit, stop, idx) or None
    resting_scanned_idx: int = -1                          # last bar index whose cross was already judged (no double fill/log)
    resting_filled: bool = False                            # one fill per symbol-day (entry_mode='resting_stop_limit') — TAPE side only, never set by a broker fill
    resting_tape_cross_idx: Optional[int] = None            # arm['idx'] already resolved via a live trade print — skip the bar-level fallback for it
    live_order: Optional[dict] = None                       # the REAL resting order at the broker (dry_run=false only): order_id, coid, level/trigger/limit/stop/qty/booked_qty/tp_leg_id/sl_leg_id/trade_id
    live_filled: bool = False                                # broker CONFIRMED fill (distinct from resting_filled, the tape's prediction) — stops further arming for the day
    tape_cross: Optional[dict] = None                         # the CURRENT arm's print-watch cross resolution: {ts, print, ask, nbbo_ok} — set only by _on_trade_print (a real
                                                               # tape print), None for a bar-level fallback cross or no cross yet; read by _append_live_parity_row (item 9)

    def set_bar(self, minute: int, o: float, h: float, l: float, c: float, v: float) -> bool:
        i = minute - OPEN_MINUTE
        if not (0 <= i < RTH_MINUTES):
            return False
        changed = (not self.have[i]) or bool(np.any(self.ohlcv[i, :4] != (o, h, l, c))) or self.ohlcv[i, 4] != v
        self.ohlcv[i, 0] = o; self.ohlcv[i, 1] = h; self.ohlcv[i, 2] = l; self.ohlcv[i, 3] = c; self.ohlcv[i, 4] = v
        if changed:
            # a bar that lands BEFORE bars already scanned (late/out-of-order delivery) or an UPDATED bar (a late print that
            # moved a high or the volume — the cache's final bar) changes the HOD, the cumulative volume and the consolidation
            # of everything after it: detect must rescan from its compacted index
            self.have[i] = True
            self.next_idx = min(self.next_idx, int(self.have[:i].sum()))
        return True

    flags: set = field(default_factory=set)

    def pattern_flag(self, name: str) -> bool:
        """True if the flag was already set; sets it (once-only logging)."""
        if name in self.flags: return True
        self.flags.add(name); return False

    @property
    def n_bars(self) -> int:
        return int(self.have.sum())

    @property
    def bars(self) -> List[dict]:
        """The closed RTH bars as dicts (diagnostics/tests; the hot path never builds this)."""
        return [{'minute': int(i) + OPEN_MINUTE, 'open': float(r[0]), 'high': float(r[1]), 'low': float(r[2]), 'close': float(r[3]), 'volume': float(r[4])}
                for i, r in zip(np.flatnonzero(self.have), self.ohlcv[self.have])]


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
    client_order_id: Optional[str] = None                 # our own id: the ONLY key an order is ever re-identified by
    pattern_data: dict = field(default_factory=dict)      # the DB pattern_data JSON (leg ids live here)
    fill_at_estimate_r: Optional[float] = None
    leg_booked: Dict[str, int] = field(default_factory=dict)   # order id -> shares already booked as sold from it
    closed_qty: int = 0                                   # shares sold so far (TP / SL / close order, partials included)
    closed_notional: float = 0.0
    last_close_reason: Optional[str] = None
    fc_attempts: int = 0

    @property
    def open_qty(self) -> int:
        return max(0, self.shares - self.closed_qty)


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
        self.max_spread_bps = float(cfg.get('max_spread_bps', 100.0))
        # The spec fills at the NEXT bar's open or never (no chase). A resting limit that fills a minute later on a
        # pullback is a trade the backtest never took — so the order lives only long enough to cover submit latency.
        self.order_timeout_s = float(cfg.get('order_timeout_s', 10.0))
        self.max_quote_age_s = float(cfg.get('max_quote_age_s', 5.0))       # a quote older than this (halt, stale feed) = no order
        self._last_pending_check = 0.0
        self.max_spread_frac_r = float(cfg.get('max_spread_frac_r', 0.0))   # 0 = off; e.g. 0.15 = skip when the spread is > 15% of R (9/14: 57% of signals)
        # ADMISSION threshold (9/15 CRWL miss): the scanner must start streaming a stock's bars BEFORE its break, so
        # candidates are admitted at a lower distance from the open than the spec's floor; the floor itself is
        # enforced at the break inside hod_break.detect (min_dist_open_pct). Default 1.5 pct-points below the floor.
        # THE BOOK: 'hod_break' (the HOD consolidation break, trading/hod_break.py) or 'red_to_green' (F6-PDR,
        # trading/red_to_green.py). Same engine, same fill/bracket/flat machinery; the signal function, the universe
        # screen, the DB strategy tag, the handler id and the log tag differ. (research/fuckup_audit/H/F6/ENGINE_SPEC.md)
        self.book = str(cfg.get('book', 'hod_break'))
        if self.book == 'red_to_green':
            self.params = RedToGreenParams(**(cfg.get('params') or {}))
            self.STRATEGY_NAME = 'red_to_green'; self.tag = '[R2G]'; self.dry_tag = '[R2G DRY]'; self.coid_prefix = 'r2g'
        elif self.book == 'hod_break':
            self.params = HodBreakParams(**(cfg.get('params') or {}))
            self.tag = '[HOD]'; self.dry_tag = '[HOD DRY]'; self.coid_prefix = 'hod'
        else:
            raise ValueError(f"unknown book {self.book!r} — expected 'hod_break' or 'red_to_green'")
        # entry_mode (docs/hod_resting_entry_spec_20260925.md, cell 1,438): 'next_open' (default) is today's
        # behaviour, byte-identical. 'resting_stop_limit' arms a buy-stop-limit at the close of every bar; the tape
        # side (_evaluate_resting/_on_trade_print) always runs and never submits an order. When dry_run is ALSO
        # False, a REAL stop-limit order is placed/cancelled/replaced alongside the tape
        # (docs/hod_live_resting_orders_spec_20260925.md) — see _arm_live_order/_poll_live_fills.
        self.entry_mode = str(cfg.get('entry_mode', 'next_open'))
        if self.entry_mode not in ('next_open', 'resting_stop_limit'):
            logger.warning(f"{self.tag} unknown entry_mode {self.entry_mode!r} — falling back to next_open")
            self.entry_mode = 'next_open'
        logger.info(f"{self.tag} entry_mode={self.entry_mode} (resting stop-limit: trigger level+0.01, "
                    f"limit {float(getattr(self.params, 'entry_limit_pct', 0.0015)):.4%})")
        self.dry_ledger_path = str(cfg.get('dry_ledger_path', 'logs/hod_dry_entry_ledger.csv'))
        # LIVE resting-order state (docs/hod_live_resting_orders_spec_20260925.md): _live_cap_slots counts a
        # RESTING order as a slot (never just a fill) so a burst of fills can never exceed max_per_day/max_concurrent.
        # live_orders_state_path persists {symbol: live_order dict} so a crash/restart reconciles by TRACKED ORDER
        # ID (never a client_order_id-prefix guess — replace_order_by_id returns a NEW id/coid, so this engine never
        # calls it; a level change is always cancel + a brand-new order, keeping our prefix on every live order).
        self.live_parity_ledger_path = str(cfg.get('live_parity_ledger_path', 'logs/hod_live_parity_ledger.csv'))
        self.live_orders_state_path = str(cfg.get('live_orders_state_path', 'logs/hod_live_resting_orders_state.json'))
        self._live_cap_slots: set = set()
        self._cap_logged: set = set()      # symbols already logged at 'LIVE cap reached' this session — dedup (9/25: was a WARNING every bar)
        self._bp_cache_value: Optional[float] = None   # cached account buying power for the resting-notional guard (_buying_power_cached)
        self._bp_cache_ts: float = 0.0                 # refreshed at most once/minute — read once per minute via the alpaca client
        self._live_cancel_swept_entry = False
        self._live_cancel_swept_flat = False
        self._live_reconciled = False
        self.live_since: Optional[datetime] = None            # ET timestamp of the FIRST bar this engine received live over the websocket
                                                                # (never a backfill/catch-up bar); reset each session in _roll_session
        self._prev_day: Dict[str, tuple] = {}
        self.admit_above_open_pct = float(cfg.get('admit_above_open_pct', max(0.0, getattr(self.params, 'min_dist_open_pct', 5.0) - 1.5)))
        # STREAM THE UNIVERSE (9/15 core fix for the CRWL class): every tradable name's bars flow from 09:30 through the
        # websocket, exactly the spec's world — no snapshot admission, no threshold to cross, no backfill on a normal day.
        self.stream_universe = bool(cfg.get('stream_universe', True))
        self.universe_min_prev_close = float(cfg.get('universe_min_prev_close', self.min_price * 0.85))
        self._last_close: Dict[str, float] = {}
        self.stream_list_dir = str(cfg.get('stream_list_dir', 'logs'))    # where the day's streamed-symbol list is written (tests point it elsewhere)
        self.candidates: Dict[str, Candidate] = {}; self.positions: Dict[str, Position] = {}
        self.entered_today: set = set(); self.daily_pnl = 0.0; self.session_date: Optional[str] = None
        self._mover_queue: queue.Queue = queue.Queue(maxsize=5000); self._bar_queue: queue.Queue = queue.Queue(maxsize=100_000)   # ~300 B per bar; 3,600 names × a few minutes must never drop
        self._last_bar_ingest = 0.0; self._silence_alerted = False; self.calendar_ok = True
        self._adv_map: Dict[str, float] = {}; self._kill_notified: set = set(); self._flattened = False
        self.shutdown_requested = False; self._lock = threading.RLock()   # tick (engine pool) and drains (main thread) must not interleave
        self.seen_today: set = set()                                      # once-per-symbol (orders incl. no-fills); entered_today = the day-cap set (fills/working orders)
        self._ws_gen: Optional[int] = None                                # StopMonitor connect generation last seen (outage → re-backfill)
        self._drain_thread: Optional[threading.Thread] = None
        logger.info(f"{self.tag} engine gates: book={self.book} enabled={self.enabled} dry_run={self.dry_run} risk=${self.risk_usd:.0f} "
                    f"kills={self.daily_kill_usd}/{self.weekly_kill_usd} cap={self.params.cap:.2%} target={self.params.target_r}R "
                    f"per_day={self.params.max_per_day} concurrent={self.params.max_concurrent} flat={self.params.flat_minute} admit>={self.admit_above_open_pct:.1f}% stream_universe={self.stream_universe}")

    # ------------------------------------------------------------------ clock / session
    def _et_now(self) -> datetime:
        return datetime.now(timezone.utc).astimezone(ET)

    def _minute_of_day(self) -> int:
        n = self._et_now(); return n.hour * 60 + n.minute

    def _bar_close_et(self, minute: int) -> datetime:
        """Wall-clock ET close time of the bar that STARTS at `minute` (minutes since midnight) in the
        current session — a bar labeled minute m covers [m, m+1) and closes at m+1."""
        midnight = datetime.strptime(self.session_date, '%Y-%m-%d').replace(tzinfo=ET)
        return midnight + timedelta(minutes=int(minute) + 1)

    def _roll_session(self) -> None:
        today = self._et_now().strftime('%Y-%m-%d')
        if self.session_date != today:
            self.session_date = today; self.candidates.clear(); self.entered_today.clear(); self.daily_pnl = 0.0
            self.live_since = None
            self._kill_notified.clear(); self._flattened = False
            self._live_cap_slots.clear(); self._cap_logged.clear(); self._bp_cache_value = None; self._bp_cache_ts = 0.0
            self._live_cancel_swept_entry = False; self._live_cancel_swept_flat = False
            self._apply_session_calendar()
            self._adv_map = self._load_adv_map()
            logger.info(f"{self.tag} session {today}: adv map {len(self._adv_map)} symbols")
            if self.stream_universe:
                self._stream_the_universe()

    def _apply_session_calendar(self) -> None:
        """Early closes (13:00 ET: the day after Thanksgiving, Christmas Eve): the spec's day ends at the last bar, live
        must be flat 5 minutes before the close and stop entering an hour before. DAY bracket legs die at the close —
        a flat at 15:55 on a 13:00 day would be an overnight position. Failure → the regular 15:55/14:00 with a WARNING."""
        self.flat_minute = int(self.params.flat_minute); self.last_entry_minute = int(self.params.last_entry_minute); self.close_minute = 960
        try:
            d = self._et_now().date()
            cal = self.alpaca.get_market_calendar(d, d) or []
            row = next((c for c in cal if str(c.get('date'))[:10] == d.isoformat()), None)
            if row is None:
                logger.warning(f"{self.tag} no market-calendar row for {d} (holiday?) — regular session assumed"); self.calendar_ok = True; return
            close = row.get('close'); ct = close if isinstance(close, datetime) else None
            if ct is None:
                hh, mm = str(close)[:5].split(':'); cm = int(hh) * 60 + int(mm)
            else:
                cm = ct.hour * 60 + ct.minute
            if 0 < cm < 960:
                self.close_minute = cm; self.flat_minute = cm - 5; self.last_entry_minute = min(self.last_entry_minute, cm - 65)
                logger.warning(f"{self.tag} EARLY CLOSE {d}: close {cm // 60:02d}:{cm % 60:02d} ET — flat at {self.flat_minute // 60:02d}:{self.flat_minute % 60:02d}, last entry {self.last_entry_minute // 60:02d}:{self.last_entry_minute % 60:02d}")
                self._notify(f"{self.tag} early close today: flat {self.flat_minute // 60:02d}:{self.flat_minute % 60:02d} ET")
            self.calendar_ok = True
            logger.info(f"{self.tag} session calendar {d}: close {self.close_minute // 60:02d}:{self.close_minute % 60:02d} ET, flat {self.flat_minute // 60:02d}:{self.flat_minute % 60:02d}, last entry {self.last_entry_minute // 60:02d}:{self.last_entry_minute % 60:02d}")
        except Exception as e:
            self.calendar_ok = False
            logger.error(f"{self.tag} market calendar unavailable ({e}) — NO ENTRIES until it answers (the session close is unknown); retried every tick")
            self._notify_once('calendar', f"{self.tag} ERROR: market calendar unavailable — no entries until it answers")

    def _load_adv_map(self) -> Dict[str, float]:
        """True 20-session ADV from daily_bars (the study's definition), universe field as the fallback.
        Also records each symbol's last close for the streamed-universe screen."""
        adv: Dict[str, float] = {}
        path = getattr(self.db, '_cache_path', None)
        if path:
            try:
                adv20, last = load_adv20_from_daily_bars(path)
                adv.update(adv20); self._last_close.update(last)
                if self.book == 'red_to_green':
                    self._prev_day = load_prev_day_from_daily_bars(path)
                    logger.info(f"{self.tag} prior-day close/high/low from daily_bars for {len(self._prev_day)} symbols")
                logger.info(f"{self.tag} ADV20 from daily_bars for {len(adv20)} symbols")
                return adv
            except Exception as e:
                logger.error(f"{self.tag} daily_bars ADV20 unavailable ({e}) — falling back to the universe field (a single day's volume, NOT the spec's ADV20)")
        else:
            logger.warning(f"{self.tag} db has no _cache_path — daily_bars ADV20 unavailable, using the universe field (tests only)")
        try:
            rows = self.db.get_active_universe()
            adv = {r['symbol']: float(r.get('avg_volume_daily') or 0.0) for r in rows}
        except Exception as e:
            logger.error(f"{self.tag} universe ADV map unavailable ({e}) — every mover will fail the ADV gate today")
        return adv

    def _stream_the_universe(self) -> None:
        """Subscribe every tradable name at session start; each becomes a candidate whose bars stream from 09:30."""
        syms = sorted(s for s, a in self._adv_map.items() if a >= self.min_adv20 and self._last_close.get(s, 0.0) >= self.universe_min_prev_close)
        if self.book == 'red_to_green':                        # the F6-PDR precondition known at 09:30: prior-day range >= pdr_min_pct
            keep = []
            for s in syms:
                pc, ph, pl = self._prev_day.get(s, (0.0, None, None))
                pdr = prior_day_range_pct(ph, pl)
                if pc > 0 and pdr is not None and pdr >= self.params.pdr_min_pct: keep.append(s)
            logger.info(f"{self.tag} universe screen: {len(keep)} of {len(syms)} names have prior-day range >= {self.params.pdr_min_pct:.0f}% (the gap-down test is per bar at 09:30)")
            syms = keep
        if not syms:
            logger.error(f"{self.tag} streamed universe is EMPTY (no last closes / ADV) — falling back to scan admission only"); return
        late = self._minute_of_day() >= OPEN_MINUTE             # any roll at/after 09:30 may have missed the opening bar (emitted ~09:31:00.2): backfill
        for s in syms:
            if s in self.candidates: continue
            pc, ph, pl = self._prev_day.get(s, (0.0, None, None))
            self.candidates[s] = Candidate(symbol=s, day_open=0.0, adv20=self._adv_map[s], subscribed=True, backfill_ok=not late,
                                           prior_close=float(pc or 0.0), pdr_pct=prior_day_range_pct(ph, pl))
        try:
            if self.stop_monitor is not None and hasattr(self.stop_monitor, 'subscribe_bars_many'):
                self.stop_monitor.subscribe_bars_many(syms)
            elif self.stop_monitor is not None and hasattr(self.stop_monitor, 'subscribe_bars'):
                for s in syms: self.stop_monitor.subscribe_bars(s)
        except Exception as e:
            logger.error(f"{self.tag} universe bar subscription failed ({e}) — scan admission remains as the fallback")
        logger.info(f"{self.tag} streaming {len(syms)} universe symbols (prev close >= {self.universe_min_prev_close:.2f}, ADV20 >= {self.min_adv20:,.0f}){' — restart after the open: backfilling' if late else ''}")
        try:                                                   # the miss audit reads this to tell a streamed symbol from a scan-admitted one
            if not self.session_date: raise ValueError('no session date')
            import os; os.makedirs(self.stream_list_dir, exist_ok=True)
            with open(os.path.join(self.stream_list_dir, f'hod_stream_universe_{self.session_date}.txt'), 'w') as f: f.write('\n'.join(syms) + '\n')
        except Exception as e:
            logger.warning(f"{self.tag} could not write the streamed-universe list ({e}) — the miss audit will treat every symbol as scan-admitted")
        if late:
            self._backfill([self.candidates[s] for s in syms])

    # ------------------------------------------------------------------ scanner hooks (enqueue-only)
    def on_mover(self, symbol: str, *, price: float, day_open: float, cum_volume: float, above_open_pct: float, ts=None) -> None:
        """Called from the per-minute broad scan for symbols >= min_dist_open_pct above the open. Never blocks."""
        if not self.enabled:
            return
        try:
            self._mover_queue.put_nowait((symbol, float(price), float(day_open), float(cum_volume), float(above_open_pct)))
        except queue.Full:
            logger.warning(f"{self.tag} mover queue full — dropped {symbol}")

    def register_on_stop_monitor(self) -> bool:
        if self.stop_monitor is None or getattr(self.stop_monitor, 'polling_mode', False):
            logger.warning(f"{self.tag} no websocket StopMonitor — bar stream unavailable, engine cannot detect breaks")
            return False
        try:
            self.stop_monitor.register_bar_handler(self.STRATEGY_NAME, self._on_bar_close, window=False)   # ONE bar dict per event, no DataFrame
        except TypeError:
            logger.warning(f"{self.tag} StopMonitor has no light bar handlers — receiving full windows (slower)")
            self.stop_monitor.register_bar_handler(self.STRATEGY_NAME, self._on_bar_close)
        if hasattr(self.stop_monitor, 'register_trade_print_handler'):
            self.stop_monitor.register_trade_print_handler(self.STRATEGY_NAME, self._on_trade_print)   # tape-accurate resting fills, entry_mode='resting_stop_limit' only
        return True

    def _on_bar_close(self, symbol: str, bar) -> None:
        """WS thread: zero work, enqueue only. `bar` is one bar dict (light handler) or a DataFrame/list of bars."""
        if not self.enabled or symbol not in self.candidates:
            return
        if self.live_since is None:
            self.live_since = self._et_now()                  # first bar this engine ever saw over the websocket this session
            logger.info(f"{self.tag} live_since={self.live_since.isoformat()} (first streamed bar) — earlier/backfilled bars will not arm")
        try:
            self._bar_queue.put_nowait((symbol, bar))
        except queue.Full:
            logger.error(f"{self.tag} bar queue full — dropped a bar for {symbol}; the day must be re-backfilled before any evaluation")
            self.candidates[symbol].needs_refill = True

    # ------------------------------------------------------------------ main-thread work
    def process_tick(self) -> None:
        """Called every scanner cycle (unconditionally — the shared force-close latch must not stop us)."""
        if not self.enabled:
            return
        with self._lock:
            try:
                if not self._live_reconciled and self.book == 'hod_break' and self.entry_mode == 'resting_stop_limit' and not self.dry_run:
                    self._reconcile_live_orders_on_boot()
                self._roll_session()
                if not self.calendar_ok: self._apply_session_calendar()
                self._check_stream_outage()
                self._check_stream_silence()
                if self._drain_thread is not None and not self._drain_thread.is_alive() and not self.shutdown_requested:
                    logger.error(f"{self.tag} bar drain thread is DEAD — restarting it"); self._drain_thread = None; self.start_drain_thread()
                self._admit_movers()
                self.drain_bar_events()
                self._process_pending_fills()
                self.check_exits()
                if self.book == 'hod_break' and self.entry_mode == 'resting_stop_limit' and not self.dry_run:
                    self._poll_live_fills()
                    self._sweep_live_cutoffs()
                if self.is_force_close_time() and (not self._flattened or any(p.status == 'open' for p in self.positions.values())):
                    self.force_close_all()
            except Exception as e:
                logger.error(f"{self.tag} process_tick failed: {e}", exc_info=True)
        try:
            self._reconcile_stream_chunk()
        except Exception as e:
            logger.error(f"{self.tag} stream reconcile failed: {e}", exc_info=True)

    RECONCILE_FROM = OPEN_MINUTE + 6          # 09:36: the early bars are what a subscribe lag or a dropped frame would have cost
    RECONCILE_CHUNK = 200

    def _reconcile_stream_chunk(self) -> None:
        """Once per day per streamed symbol, MERGE the REST truth into the streamed day (review A: nothing verified that
        the 09:30 bar or any early bar actually arrived on the websocket — a dropped frame meant a silently wrong open,
        HOD and volume). One chunk of symbols per tick, the REST call OUTSIDE the engine lock so evaluation never
        waits; `set_bar` rewinds `next_idx` when a missing earlier bar is filled, so the merge is self-correcting."""
        if self._minute_of_day() < self.RECONCILE_FROM or self._minute_of_day() >= 960:
            return
        with self._lock:
            todo = [c for c in self.candidates.values() if c.subscribed and not c.reconciled][:self.RECONCILE_CHUNK]
        if not todo: return
        try:
            got = self.alpaca.get_1min_bars_multi([c.symbol for c in todo], lookback_minutes=max(30, self._minute_of_day() - OPEN_MINUTE + 5))
        except Exception as e:
            logger.warning(f"{self.tag} stream reconcile: REST failed for {len(todo)} symbols ({e}) — retried next tick"); return
        fixed = 0
        with self._lock:
            for c in todo:
                c.reconciled = True
                df = (got or {}).get(c.symbol)
                if df is None or not len(df): continue
                before = c.n_bars; nxt = c.next_idx
                cand = self._set_bars(c.symbol, df)
                if cand is not None and (cand.n_bars != before or cand.next_idx < nxt):
                    fixed += 1
                    if cand.rejected_reason:
                        logger.error(f"{self.tag} {c.symbol}: streamed day was missing {cand.n_bars - before} bar(s) and the symbol was already judged ({cand.rejected_reason}) — a decision on an incomplete day (parity incident)")
                    else:
                        logger.warning(f"{self.tag} {c.symbol}: streamed day was missing {cand.n_bars - before} bar(s) — merged from REST, rescanning from bar {cand.next_idx}")
                        if cand.backfill_ok and not cand.needs_refill: self._evaluate(cand)
        left = sum(1 for c in self.candidates.values() if c.subscribed and not c.reconciled)
        (logger.warning if fixed else logger.info)(f"{self.tag} stream reconcile: {len(todo)} symbols checked against REST, {fixed} had missing bars, {left} to go")

    def _check_stream_outage(self) -> None:
        """A WebSocket reconnect after the open means bars were missed: every live candidate's day (HOD, cumulative
        volume) is suspect until re-backfilled from REST. Pre-open reconnects lose nothing."""
        gen = getattr(self.stop_monitor, 'ws_generation', None)
        if gen is None or gen == self._ws_gen:
            return
        first = self._ws_gen is None; self._ws_gen = gen
        if first or self._minute_of_day() <= OPEN_MINUTE:
            return
        n = 0
        for c in self.candidates.values():
            if c.subscribed and c.rejected_reason is None:
                c.needs_refill = True; n += 1
        logger.warning(f"{self.tag} bar stream reconnected (generation {gen}) after the open — {n} candidates re-backfilled before any evaluation")
        self._notify(f"{self.tag} bar stream reconnected after the open — re-backfilling {n} names")

    def _check_stream_silence(self) -> None:
        """Liveness rail: during RTH a streamed universe of thousands of names produces bars every minute. Two minutes
        without a single ingested bar means the subscription is gone (silent reconnect, auth drop) — re-send it,
        re-backfill everything, and say so. Fail loud, never blind."""
        if not self.candidates or not (OPEN_MINUTE + 3 <= self._minute_of_day() < 960):
            return
        since = time.time() - (self._last_bar_ingest or 0)
        if self._last_bar_ingest and since < 120:
            self._silence_alerted = False; return
        if self._last_bar_ingest == 0 and self._minute_of_day() < OPEN_MINUTE + 5:
            return
        if not self._silence_alerted:
            self._silence_alerted = True
            logger.error(f"{self.tag} NO BARS INGESTED for {since:.0f}s during RTH — re-subscribing and re-backfilling the universe")
            self._notify(f"{self.tag} ERROR: no bars for {since:.0f}s during RTH — re-subscribing")
            try:
                sm = self.stop_monitor
                if sm is not None and hasattr(sm, 'subscribe_bars_many'):
                    syms = [c.symbol for c in self.candidates.values() if c.subscribed]
                    sm._bar_symbols.difference_update(syms) if hasattr(sm, '_bar_symbols') else None
                    sm.subscribe_bars_many(syms)
            except Exception as e:
                logger.error(f"{self.tag} re-subscribe failed: {e}")
            for c in self.candidates.values():
                if c.subscribed and c.rejected_reason is None: c.needs_refill = True

    def _admit_movers(self) -> None:
        n = 0; new: List[Candidate] = []
        while not self._mover_queue.empty() and n < 500:
            symbol, price, day_open, cum_vol, above = self._mover_queue.get_nowait(); n += 1
            if symbol in self.candidates or symbol in self.seen_today or symbol in self.positions:
                continue
            if not self.seen_today:                                    # first admission of the session: pull the DB set once
                db_syms = self._db_symbols_today(); self.seen_today |= db_syms | self._db_symbols_today(include_dead=True); self.entered_today |= db_syms
                if symbol in self.seen_today: continue
            adv = self._adv_map.get(symbol, 0.0)
            if price < self.min_price or adv < self.min_adv20 or day_open <= 0:
                continue
            cand = Candidate(symbol=symbol, day_open=day_open, adv20=adv)
            self.candidates[symbol] = cand; new.append(cand)
            logger.info(f"{self.tag} candidate {symbol} admitted: {price:.2f} +{above:.1f}% from open {day_open:.2f}, adv20 {adv:,.0f} (#{len(self.candidates)} today)")
            self._subscribe(cand)
        retry = [c for c in self.candidates.values() if c.subscribed and (not c.backfill_ok or c.needs_refill) and c.rejected_reason is None]
        self._backfill([c for c in new if c.subscribed] + [c for c in retry if c not in new])

    def _subscribe(self, cand: Candidate) -> None:
        try:
            if self.stop_monitor is not None and hasattr(self.stop_monitor, 'subscribe_bars'):
                self.stop_monitor.subscribe_bars(cand.symbol); cand.subscribed = True
        except Exception as e:
            logger.error(f"{self.tag} {cand.symbol}: bar subscribe failed ({e}) — candidate dropped")
            self.candidates.pop(cand.symbol, None)

    BACKFILL_CHUNK = 200

    def _backfill(self, cands: List[Candidate]) -> None:
        """Batched REST calls (chunks of BACKFILL_CHUNK) for every candidate whose day is incomplete (scan-admitted after the
        open, a restart, a stream outage). The REST window starts at 09:30, so a non-empty result IS the complete day and
        marks the candidate evaluable; a symbol missing from the result is an ERROR and is retried next tick (never
        evaluated on stream-only bars — the DBI-09:50 class). Before the open there is nothing to fetch."""
        cands = [c for c in cands if not c.backfill_ok or c.needs_refill]
        if not cands or self._minute_of_day() <= OPEN_MINUTE: return
        lookback = max(30, self._minute_of_day() - OPEN_MINUTE + 5)
        for i in range(0, len(cands), self.BACKFILL_CHUNK):
            chunk = cands[i:i + self.BACKFILL_CHUNK]
            try:
                got = self.alpaca.get_1min_bars_multi([c.symbol for c in chunk], lookback_minutes=lookback)
            except Exception as e:
                logger.error(f"{self.tag} backfill call failed for {len(chunk)} candidates ({e}) — retry next tick"); continue
            empty = []
            for c in chunk:
                df = (got or {}).get(c.symbol)
                if df is None or not len(df):
                    c.backfill_tries += 1
                    if c.backfill_tries in (1, 5): empty.append(c.symbol)
                    continue
                c.backfill_ok = True; c.needs_refill = False; c.next_idx = 0   # a (re)filled day is re-scanned from its first bar
                self._ingest_bars(c.symbol, df)
            if empty:                                          # a halted/dead name returns nothing all day (WARNING); a whole empty chunk is a REST failure (ERROR → Telegram)
                (logger.error if len(empty) == len(chunk) else logger.warning)(
                    f"{self.tag} backfill returned no bars for {len(empty)} of {len(chunk)} symbols — not evaluated until the 09:30 open is present: {empty[:20]}{'…' if len(empty) > 20 else ''}")

    def start_drain_thread(self) -> None:
        """Evaluate bars the moment they close. The scanner's cycle can spend 10-30 s in its own work between drains;
        the spec acts at the bar close, so a dedicated thread blocks on the bar queue instead."""
        if self._drain_thread is not None and self._drain_thread.is_alive():
            return
        self._drain_thread = threading.Thread(target=self._drain_loop, name='hod-break-drain', daemon=True); self._drain_thread.start()
        logger.info(f"{self.tag} bar drain thread started")

    SETTLE_S = 0.4                                          # a minute's bars arrive within a few hundred ms; wait for the batch

    def _drain_loop(self) -> None:
        """Bars of one minute are processed as a BATCH: collect for SETTLE_S after the first arrival, store them all,
        resolve pending orders, then evaluate the touched candidates in SYMBOL order — the spec's tie-break
        (`run_book`), instead of websocket arrival order."""
        while not self.shutdown_requested:
            items = []
            try:
                items.append(self._bar_queue.get(timeout=1.0))
            except queue.Empty:
                pass
            if items:
                t0 = time.time()
                while time.time() - t0 < self.SETTLE_S:
                    try: items.append(self._bar_queue.get(timeout=0.05))
                    except queue.Empty: pass
            try:
                with self._lock:
                    if items: self._ingest_batch(items)
                    if any(p.status == 'pending' for p in self.positions.values()) and time.time() - self._last_pending_check >= 1.0:
                        self._process_pending_fills()
            except Exception as e:
                logger.error(f"{self.tag} drain loop failed: {e}", exc_info=True)

    def _ingest_batch(self, items) -> List[str]:
        """Store every bar of the batch, close the fill window of symbols whose next bar arrived, then evaluate the
        touched candidates in symbol order. Used by the drain thread and by `drain_bar_events`."""
        touched: Dict[str, Candidate] = {}
        self._last_bar_ingest = time.time()
        self._check_stream_outage()                                # the drain thread sees a reconnect before the next scanner tick
        for symbol, payload in items:
            cand = self._set_bars(symbol, payload)
            if cand is not None: touched[symbol] = cand
        for symbol in sorted(touched):
            pos = self.positions.get(symbol)
            if pos is not None and pos.status == 'pending':
                self._process_pending_fills(force_timeout=symbol)   # the next bar closed: the spec's fill window is over
        for symbol in sorted(touched):
            self._evaluate(touched[symbol])
        return sorted(touched)

    def drain_bar_events(self) -> List[str]:
        """Scanner-cycle drain — a no-op while the drain thread is alive (ONE consumer keeps bars in order; two
        consumers of the same FIFO could ingest a symbol's later bar before its earlier one)."""
        if self._drain_thread is not None and self._drain_thread.is_alive():
            return []
        with self._lock:
            items = []
            while not self._bar_queue.empty():
                items.append(self._bar_queue.get_nowait())
            return self._ingest_batch(items) if items else []

    @staticmethod
    def _rth_arrays(cand: Candidate):
        idx = np.flatnonzero(cand.have)
        if not len(idx):
            return None
        a = cand.ohlcv[idx]
        return a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], (idx + OPEN_MINUTE).astype(int)

    @staticmethod
    def _bar_minute(b: dict) -> int:
        """ET minute-of-day of a bar's timestamp (aware/naive datetime or ISO string)."""
        ts = b.get('timestamp')
        t = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        et = t.astimezone(ET)
        return et.hour * 60 + et.minute

    def _set_bars(self, symbol: str, payload) -> Optional[Candidate]:
        """MERGE by minute into the candidate's fixed array — never replace. Found 2026-09-14 09:50 (DBI): the stream
        carries only bars since SUBSCRIPTION; replacing the backfilled day with it dropped the 09:30 open, the early
        high-of-day and the early volume. `payload` = one bar dict (light stream handler), a DataFrame (backfill) or a
        list of dicts. Bars outside 09:30-15:59 ET are ignored. Returns the candidate (or None)."""
        cand = self.candidates.get(symbol)
        if cand is None:
            return None
        try:
            if isinstance(payload, dict): recs = [payload]
            elif hasattr(payload, 'to_dict'): recs = payload.to_dict('records')
            else: recs = list(payload)
            for b in recs:
                cand.set_bar(self._bar_minute(b), float(b['open']), float(b['high']), float(b['low']), float(b['close']), float(b.get('volume') or 0))
        except Exception as e:
            logger.error(f"{self.tag} {symbol}: bad bar payload ({e})"); return None
        return cand

    def _ingest_bars(self, symbol: str, payload) -> None:
        """Store one symbol's bars and evaluate it (single-symbol path: backfill and tests)."""
        self._ingest_batch([(symbol, payload)])

    def _evaluate(self, cand: Candidate) -> None:
        if cand.rejected_reason or cand.symbol in self.positions or cand.symbol in self.entered_today:
            return
        arr = self._rth_arrays(cand)
        if arr is None:
            return
        o, h, l, c, v, m = arr; n = len(o)
        if cand.needs_refill:
            return                                         # a hole in the day (outage, dropped bar): nothing is judged until REST refilled it
        if not cand.backfill_ok:
            if int(m[0]) != OPEN_MINUTE:
                return                                     # stream-only bars (no 09:30 open yet): wrong open/HOD/rv — wait for the backfill
            cand.backfill_ok = True                        # the set starts at the opening bar: complete from the open whatever the source
        if cand.day_open > 0 and abs(float(o[0]) - cand.day_open) > 0.011 and not cand.pattern_flag('open_mismatch_logged'):
            logger.warning(f"{self.tag} {cand.symbol}: first 1-min bar open {float(o[0]):.2f} != snapshot day open {cand.day_open:.2f} — the spec's o0 is the first bar; using it")
        cand.day_open = float(o[0])                        # the spec's o0 = the first RTH bar's open, always
        if not self.calendar_ok:
            return                                         # no calendar = no session close known = no entries (fail closed)
        params = self.params if getattr(self, 'last_entry_minute', self.params.last_entry_minute) == self.params.last_entry_minute \
            else type(self.params)(**{**self.params.__dict__, 'last_entry_minute': self.last_entry_minute})
        if self.book == 'hod_break' and self.entry_mode == 'resting_stop_limit':
            self._evaluate_resting(cand, o, h, l, v, m, params)
            return
        if self.book == 'red_to_green':
            sig = r2g_detect(o, h, l, v, m, cand.adv20, cand.prior_close, cand.pdr_pct, params, start_idx=cand.next_idx)
        else:
            sig = detect(o, h, l, v, m, cand.adv20, params, start_idx=cand.next_idx)
        if sig is None:
            cand.next_idx = n; return
        cand.next_idx = n
        if sig.bar_idx < n - 1:
            # The spec trades a symbol's FIRST break only. This one already passed (late admission, restart, outage):
            # its next-open fill is gone and any later break is a trade the backtest never took — the symbol is done.
            cand.rejected_reason = 'stale_break'
            logger.warning(f"{self.tag} {cand.symbol}: MISSED the spec's break at bar {sig.bar_idx} (level {sig.level:.2f}, now {n - 1 - sig.bar_idx} bars old) — no later break is taken")
            return
        self._try_enter(cand, sig, o[0])

    # ------------------------------------------------------------------ resting entry (dry only, entry_mode='resting_stop_limit')
    def _on_trade_print(self, symbol: str, price: float, size: int, ts: float) -> None:
        """WS thread: resolve an armed resting order against a REAL trade print — the tape-accurate path
        (docs/hod_resting_entry_spec_20260925.md). Same fill rule as the bar-level fallback
        (`trading.hod_break.resting_entry_fill`), just fed a real print + the quote cached at that instant
        instead of a closed bar's high. Only the FIRST print >= trigger resolves the arm; on NO FILL the
        arm is left in place (`resting_tape_cross_idx` marks it resolved) so `_evaluate_resting` skips its
        own bar-level fallback for this cross and, at the bar's close, re-evaluates from scratch."""
        if not self.enabled or self.book != 'hod_break' or self.entry_mode != 'resting_stop_limit':
            return
        with self._lock:
            cand = self.candidates.get(symbol)
            if cand is None or cand.resting_filled or cand.live_filled or cand.resting_arm is None:
                return
            arm = cand.resting_arm
            if cand.resting_tape_cross_idx == arm['idx'] or price < arm['trigger'] - 1e-9:
                return                                       # already resolved this arm, or print below trigger — do nothing
            q = self.stop_monitor.get_print_quote(symbol) if self.stop_monitor is not None else None
            if q is None:
                return                                        # no quote cached yet — the bar-level fallback resolves it at the bar's close
            bid, ask, qts = q
            fill_px = resting_entry_fill(ask, arm)
            filled = fill_px is not None
            cross_ts = self._et_now()
            # A real print resolved this cross — the parity ledger's tape_* columns (item 9) are populated ONLY
            # here, never from the bar-level fallback (tape_accurate=False there): nbbo_ok=True because Alpaca's
            # own stop trigger is itself NBBO-filtered consolidated-tape, and we only reach this branch on a print.
            cand.tape_cross = {'ts': cross_ts, 'print': price, 'ask': ask, 'nbbo_ok': True}
            logger.info(f"{self.dry_tag} CROSS {symbol} (tape) at {cross_ts:%H:%M:%S}.{cross_ts.microsecond // 1000:03d} "
                        f"print {price:.2f} size {size} ask {ask:.2f} -> " + (f"FILL {fill_px:.2f}" if filled else "NO FILL (ask > limit)"))
            if filled:
                cand.resting_filled = True
                cand.resting_arm = None
                self._notify(f"{self.dry_tag} FILL {symbol} {fill_px:.2f} (level {arm['level']:.2f} stop {arm['stop']:.2f})")
            else:
                cand.resting_tape_cross_idx = arm['idx']       # arm stays in place — bar close re-evaluates it
            self._append_dry_ledger(cand, arm, cross_ts, ask=ask, filled=filled, fill_px=fill_px, tape_accurate=True)
        if filled and self.stop_monitor is not None:
            try:
                self.stop_monitor.unsubscribe([symbol])
            except Exception as e:
                logger.error(f"{self.tag} {symbol}: failed to unsubscribe print-watch after fill: {e}")

    def _evaluate_resting(self, cand: Candidate, o, h, l, v, m, p: HodBreakParams) -> None:
        """docs/hod_resting_entry_spec_20260925.md, cell 1,438: at the close of every bar, arm/re-arm a resting
        buy-stop-limit for the NEXT bar via `trading.hod_break.arm_state` (the same rule research/hod_entry's
        causal_arming.py walks offline — PARITY enforced by tests/test_hod_resting_entry.py), and resolve any
        cross of a previously armed trigger. DRY ONLY: never submits an order regardless of `self.dry_run`. While
        armed, `_on_trade_print` (WS thread) resolves crosses tape-accurately against the live print stream
        (`StopMonitor.subscribe_trades_quotes`/`.unsubscribe`, capped and shared with the eventual live order
        path via `trading.hod_break.resting_entry_fill`). This walk only falls back to the bar's high + the
        current quote (logged WARNING, `tape_accurate=0`) when no print resolved the cross by the bar's close —
        i.e. no live print stream reached this symbol for that bar."""
        n = len(o); sym = cand.symbol; skipped = 0
        # Bar-by-bar walk (not just the newest bar): a backfill/reconcile can hand several closed bars to one
        # _evaluate call, and each must arm/resolve in order — exactly the causal_arming.py walk offline.
        for j in range(cand.resting_scanned_idx + 1, n):
            if cand.resting_filled or cand.live_filled:
                break
            if self.live_since is None or self._bar_close_et(int(m[j])) < self.live_since:
                skipped += 1                                    # backfill/catch-up bar (or pre-live): never arms, never resolves, never ledgered
                continue
            if cand.resting_arm is not None and cand.resting_arm['idx'] + 1 == j:
                arm = cand.resting_arm
                if cand.resting_tape_cross_idx == arm['idx']:
                    cand.resting_tape_cross_idx = None            # already resolved tape-accurately — no fallback double-resolution
                elif h[j] >= arm['trigger'] - 1e-9:
                    logger.warning(f"{self.tag} {sym}: no live print stream — resolving the cross of {arm['trigger']:.2f} "
                                    f"with the bar high and the current quote (not tape-accurate)")
                    cross_ts = self._et_now()
                    q = self._quote(sym)                        # _quote() already refuses a quote older than max_quote_age_s (default 5 s) — fail closed
                    if q is None:
                        logger.warning(f"{self.tag} {sym}: CROSS {arm['trigger']:.2f} but no fresh quote — NO FILL (fail closed)")
                        self._append_dry_ledger(cand, arm, cross_ts, ask=float('nan'), filled=False, fill_px=None, tape_accurate=False)
                    else:
                        _, ask = q
                        eff_ask = max(ask, arm['trigger'])      # a triggered buy-stop-limit never fills BELOW its own trigger
                        fill_px = resting_entry_fill(eff_ask, arm)
                        filled = fill_px is not None
                        logger.info(f"{self.dry_tag} CROSS {sym} at {cross_ts:%H:%M:%S}.{cross_ts.microsecond // 1000:03d} "
                                    f"print {h[j]:.2f} ask {ask:.2f} -> " + (f"FILL {fill_px:.2f}" if filled else "NO FILL (ask > limit)"))
                        if filled:
                            cand.resting_filled = True
                            self._notify(f"{self.dry_tag} FILL {sym} {fill_px:.2f} (level {arm['level']:.2f} stop {arm['stop']:.2f})")
                        self._append_dry_ledger(cand, arm, cross_ts, ask=ask, filled=filled, fill_px=fill_px, tape_accurate=False)
                cand.resting_arm = None
            if cand.resting_filled or cand.live_filled:
                break
            new_arm = arm_state(o, h, l, v, m, j, cand.adv20, p)
            if new_arm is not None:
                # bar_volume = the arming bar's own volume (v[j]) — resting_order_qty's 5%-of-prior-bar-volume
                # cap on the LIVE order size (docs/hod_live_resting_orders_spec_20260925.md item 1); unused by
                # the tape/dry path, only read by _arm_live_order via resting_order_qty.
                new_arm = dict(new_arm, idx=j, arm_ts=self._et_now().isoformat(), bar_volume=float(v[j]))
                cand.tape_cross = None   # a fresh arm — any print-watch cross belongs to the arm just resolved above
                logger.info(f"{self.dry_tag} ARMED {sym} level {new_arm['level']:.2f} trigger {new_arm['trigger']:.2f} "
                            f"limit {new_arm['limit']:.2f} stop {new_arm['stop']:.2f}")
            if not self.dry_run and self.entry_mode == 'resting_stop_limit':
                if new_arm is not None:
                    self._arm_live_order(cand, new_arm)
                elif cand.live_order is not None:
                    self._cancel_live_order(cand, 'arm_lost')
            cand.resting_arm = new_arm
        if skipped:
            logger.info(f"{self.tag} {sym}: skipped {skipped} backfill bars for arming")
        cand.resting_scanned_idx = max(cand.resting_scanned_idx, n - 1)
        if self.stop_monitor is not None:                          # tape subscription tracks the FINAL state: armed now = live edge, streamed
            try:
                if cand.resting_arm is not None and not cand.resting_filled:
                    self.stop_monitor.subscribe_trades_quotes([sym])
                else:
                    self.stop_monitor.unsubscribe([sym])
            except Exception as e:
                logger.error(f"{self.tag} {sym}: print-watch subscribe/unsubscribe failed: {e}")

    def _append_dry_ledger(self, cand: Candidate, arm: dict, cross_ts, ask: float, filled: bool,
                            fill_px: Optional[float], tape_accurate: bool, live: bool = True) -> None:
        """Append one row to `self.dry_ledger_path` (docs/hod_resting_entry_spec_20260925.md); never raises —
        a logging failure must never affect trading. ERROR on write failure. `live` MUST be True: only bars
        that closed after this session's `live_since` ever reach an arm/resolve, so every row is a live row —
        the column exists so a stray backfill row is instantly visible as a defect, never silently trusted."""
        import csv, os
        stop = arm['stop']
        target = fill_px + self.params.target_r * (fill_px - stop) if filled and fill_px is not None else None
        row = [self.session_date or '', cand.symbol, arm.get('arm_ts', ''), cross_ts.isoformat(),
               f"{arm['level']:.4f}", f"{arm['trigger']:.4f}", f"{arm['limit']:.4f}",
               '' if ask != ask else f"{ask:.4f}", int(bool(filled)), '' if fill_px is None else f"{fill_px:.4f}",
               f"{stop:.4f}", '' if target is None else f"{target:.4f}", int(bool(tape_accurate)), int(bool(live))]
        try:
            path = self.dry_ledger_path
            is_new = not os.path.exists(path)
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(path, 'a', newline='') as fh:
                w = csv.writer(fh)
                if is_new:
                    w.writerow(['date', 'symbol', 'arm_ts', 'cross_ts', 'level', 'trigger', 'limit', 'ask',
                                'filled', 'fill_px', 'stop', 'target', 'tape_accurate', 'live'])
                w.writerow(row)
        except Exception as e:
            logger.error(f"{self.tag} {cand.symbol}: failed to append dry entry ledger row to {self.dry_ledger_path}: {e}")

    # ------------------------------------------------------------------ LIVE resting order (real orders; entry_mode='resting_stop_limit', dry_run=false)
    # docs/hod_live_resting_orders_spec_20260925.md. The tape above (_evaluate_resting/_on_trade_print) is
    # UNCHANGED and always runs — it is the parity ledger's "expected" side. This block is the "actual" side.
    # DECISION (docs/alpaca_stop_limit_probe_20260925.md): a level change is CANCEL + a brand-new order, never
    # replace_order_by_id — the probe found replace returns a NEW order id AND a new random client_order_id (our
    # prefix is lost), so every live order this engine ever places keeps its own client_order_id AND its id is
    # tracked in `cand.live_order` and persisted to `live_orders_state_path` — restart reconciliation adopts by
    # TRACKED ID, never by prefix-guessing, and never cancels an order that is not in that tracked set (the
    # owner's manual orders on the shared account are never touched).
    SAFETY_NET_PCT = 0.05   # broker stop-sell 5 % under the real stop (StopMonitor owns the real stop)
    LIVE_COID_PREFIX = 'hod-rest'

    def _persist_live_orders(self) -> None:
        """Write {symbol: live_order} for every currently-resting or partially-filled live order. Never raises —
        a logging/persist failure must never affect trading; restart reconciliation degrades to 'nothing adopted'
        (logged WARNING there), never to touching an unknown order."""
        import json, os
        try:
            path = self.live_orders_state_path
            d = os.path.dirname(path)
            if d: os.makedirs(d, exist_ok=True)
            snap = {sym: c.live_order for sym, c in self.candidates.items() if c.live_order}
            with open(path, 'w') as fh:
                json.dump(snap, fh, default=str)
        except Exception as e:
            logger.error(f"{self.tag} failed to persist live order state to {self.live_orders_state_path}: {e}")

    def _reconcile_live_orders_on_boot(self) -> None:
        """Runs ONCE per process start (first process_tick). Adopts a persisted order iff its id is STILL OPEN at
        the broker; a persisted id no longer open (filled/expired/cancelled while the process was down) is
        dropped with a WARNING — never cancelled, since we cannot know if it already became a real position. An
        open order at the broker that is NOT in the persisted set is left completely alone (may be the owner's
        manual order): this reconciliation only ever ADOPTS, never cancels."""
        self._live_reconciled = True
        import json
        try:
            with open(self.live_orders_state_path) as fh:
                persisted = json.load(fh)
        except FileNotFoundError:
            return
        except Exception as e:
            logger.error(f"{self.tag} live order state file unreadable ({e}) — starting with no adopted orders"); return
        if not persisted:
            return
        try:
            open_orders = {str(o.get('id')): o for o in (self.alpaca.get_open_orders() or [])}
        except Exception as e:
            logger.error(f"{self.tag} could not list open orders for live-order reconciliation ({e}) — nothing adopted this boot"); return
        for sym, lo in persisted.items():
            oid = str((lo or {}).get('order_id') or '')
            if oid and oid in open_orders:
                cand = self.candidates.get(sym)
                if cand is None:
                    cand = self.candidates[sym] = Candidate(symbol=sym, day_open=0.0, adv20=self._adv_map.get(sym, 0.0))
                cand.live_order = lo; self._live_cap_slots.add(sym)
                logger.warning(f"{self.tag} {sym}: ADOPTED resting order {oid} on restart (tracked id, still open at the broker)")
            else:
                logger.warning(f"{self.tag} {sym}: persisted order {oid or '(none)'} is no longer open — dropped untouched, not re-armed")
        self._persist_live_orders()

    def _arm_live_order(self, cand: Candidate, arm: dict) -> None:
        """Place, or cancel + replace, the REAL resting buy-stop-limit for `cand` at `arm` — called once per bar
        close from `_evaluate_resting`, never from the tape. 9/25 fix: a RESTING order counts only against
        `max_resting` — with ~30 armed names only the first 2 to arm got a real order when a resting slot also
        spent a max_concurrent slot. `max_per_day`/`max_concurrent` count FILLED positions only, exactly the
        spec's slot rule, via `_entered_today_count`/`_open_position_count` (both DB-backed so a fill or an exit
        made elsewhere is visible on the very next bar, not a stale poll)."""
        sym = cand.symbol
        prev = cand.live_order
        if prev is not None and abs(prev['trigger'] - arm['trigger']) < 1e-9 and abs(prev['limit'] - arm['limit']) < 1e-9:
            return                                            # unchanged level — the resting order already covers it
        if prev is not None:
            self._cancel_live_order(cand, 'replace')
        blocked = self._kill_rails_blocked()
        if blocked:
            logger.warning(f"{self.tag} {sym}: LIVE order blocked by kill rail ({blocked}) — arm stays tape-only"); return
        if self._entered_today_count() >= self.params.max_per_day:
            self._log_cap_once(sym, f"day cap {self.params.max_per_day} reached"); return
        if self._open_position_count() >= self.params.max_concurrent:
            self._log_cap_once(sym, f"concurrency cap {self.params.max_concurrent} reached"); return
        live_today = len(self._live_cap_slots)
        if live_today >= self.params.max_resting:
            self._log_cap_once(sym, f"resting cap {self.params.max_resting} reached (resting {live_today})"); return
        qty = resting_order_qty(self.risk_usd, arm)
        if qty < 1:
            logger.warning(f"{self.tag} {sym}: LIVE qty < 1 share at risk ${self.risk_usd:.0f} (trigger {arm['trigger']:.2f} stop {arm['stop']:.2f}) — no real order")
            return
        # Owner 9/25: the backtest never capped resting-order COUNT, only fills — max_resting above is a safety
        # ceiling, not a selection rule. The real limiter is NOTIONAL: skip a new resting order if the sum of
        # limit_price x qty over every resting order we hold (this one included) would exceed 25% of buying power.
        bp = self._buying_power_cached()
        if bp is None:
            self._log_cap_once(sym, "buying power unavailable — fail closed, no LIVE order"); return
        existing_notional = sum(c2.live_order['limit'] * c2.live_order['qty']
                                for s2, c2 in self.candidates.items() if s2 != sym and c2.live_order is not None)
        new_notional = existing_notional + arm['limit'] * qty
        bp_cap = 0.25 * bp
        if new_notional > bp_cap:
            self._log_cap_once(sym, f"buying-power guard: resting notional ${new_notional:,.0f} would exceed 25% of "
                                    f"BP (${bp_cap:,.0f} of ${bp:,.0f})")
            return
        coid = f"{self.LIVE_COID_PREFIX}-{sym}-{(self.session_date or '')[5:]}-{uuid.uuid4().hex[:8]}"[:48]
        try:
            od = self.alpaca.submit_stop_limit_order(symbol=sym, qty=qty, side='buy', stop_price=arm['trigger'],
                                                       limit_price=arm['limit'], client_order_id=coid)
        except Exception as e:
            logger.error(f"{self.tag} {sym}: LIVE stop-limit submit failed: {e}"); self._notify(f"{self.tag} ERROR live submit {sym}: {e}"); return
        order_id = str((od or {}).get('id') or '')
        if not order_id:
            logger.error(f"{self.tag} {sym}: LIVE stop-limit submit returned no order id — nothing tracked"); return
        cand.live_order = dict(order_id=order_id, coid=coid, level=arm['level'], trigger=arm['trigger'], limit=arm['limit'],
                               stop=arm['stop'], qty=qty, booked_qty=0, arm_ts=arm.get('arm_ts', ''), tp_leg_id='', sl_leg_id='', trade_id=None)
        self._live_cap_slots.add(sym)
        logger.info(f"{self.tag} {sym}: LIVE ARMED stop {arm['trigger']:.2f} limit {arm['limit']:.2f} qty {qty} order {order_id}")
        self._notify(f"{self.tag} LIVE ARM {sym} stop {arm['trigger']:.2f} limit {arm['limit']:.2f} x{qty}")
        self._persist_live_orders()

    def _cancel_live_order(self, cand: Candidate, reason: str) -> None:
        """Cancel `cand`'s resting entry order (never called once it has ANY booked_qty — see _poll_live_fills).
        Never raises. Writes the arm's parity-ledger row (NO_FILL/cancelled — the arm never got a broker fill)."""
        lo = cand.live_order
        if lo is None:
            return
        try:
            self.alpaca.cancel_order(lo['order_id']); status = 'cancelled'
            logger.info(f"{self.tag} {cand.symbol}: LIVE order {lo['order_id']} cancelled ({reason})")
        except Exception as e:
            status = 'cancel_failed'
            logger.error(f"{self.tag} {cand.symbol}: LIVE cancel failed for {lo['order_id']} ({reason}): {e}")
        self._append_live_parity_row(cand, lo, broker_status=status, reason=reason)
        cand.live_order = None
        self._live_cap_slots.discard(cand.symbol)
        self._persist_live_orders()

    def _log_cap_once(self, sym: str, msg: str) -> None:
        """Dedup the 'LIVE cap reached' line to once per symbol per session at INFO (was a WARNING every bar
        while a name stayed armed-but-blocked — never paged, just flooded the log; _cap_logged cleared in
        _roll_session)."""
        if sym in self._cap_logged:
            return
        self._cap_logged.add(sym)
        logger.info(f"{self.tag} {sym}: LIVE cap reached ({msg}) — arm stays tape-only")

    def _buying_power_cached(self) -> Optional[float]:
        """Account buying power for the resting-notional guard, refreshed at most once a minute via
        `AlpacaClient.get_buying_power` (owner 9/25). None means unknown — a fetch failure with no prior
        value; callers fail CLOSED on None (never risk an unbounded resting book on a monitoring hiccup)."""
        now = self._et_now().timestamp()
        if self._bp_cache_value is not None and (now - self._bp_cache_ts) < 60.0:
            return self._bp_cache_value
        try:
            bp = float(self.alpaca.get_buying_power())
            self._bp_cache_value = bp; self._bp_cache_ts = now
            return bp
        except Exception as e:
            logger.error(f"{self.tag} buying-power fetch failed ({e}) — "
                        f"{'using the last cached value' if self._bp_cache_value is not None else 'NONE cached, fail closed'}")
            return self._bp_cache_value

    def _open_position_count(self) -> int:
        """Currently OPEN filled positions for this book. The resting-order path's real exit is owned by
        StopMonitor off this engine (`_on_live_fill` never populates `self.positions`), so the DB is ground
        truth here — re-queried on every cap check so a close is visible immediately, not on a stale poll."""
        today = self.session_date or self._et_now().strftime('%Y-%m-%d')
        try:
            rows = self.db.get_open_trades(today, strategy=STRATEGY_NAME)
            db_open = {r['symbol'] for r in rows}
        except Exception as e:
            logger.error(f"{self.tag} open-position DB query failed ({e}) — falling back to in-memory count only"); db_open = set()
        return len(set(self.positions) | db_open)

    def _cancel_all_resting(self, reason: str) -> None:
        """Cancel every resting entry order with NOTHING booked yet (never one already carrying a partial fill —
        _cancel_live_order's invariant) — called when a fill takes the open-position count to max_concurrent, or
        entered_today to max_per_day. Re-arming is blocked naturally: the next `_arm_live_order` cap check re-reads
        the DB and stays tape-only until a position actually closes."""
        syms = [s for s, c in self.candidates.items()
                if c.live_order is not None and int(c.live_order.get('booked_qty') or 0) == 0]
        for s in syms:
            self._cancel_live_order(self.candidates[s], reason)
        if syms:
            logger.info(f"{self.tag} {reason} — cancelled {len(syms)} resting orders")

    def _sweep_live_cutoffs(self) -> None:
        """Cancel ALL resting real entry orders at last_entry_minute and at flat_minute (15:55 ET) — Alpaca
        auto-cancels an untriggered DAY order at 16:00 ET on its own, but the spec's window is tighter and this
        must not depend on the broker's own cutoff. Each boundary sweeps exactly once per session."""
        mnow = self._minute_of_day()
        if not self._live_cancel_swept_entry and mnow >= getattr(self, 'last_entry_minute', self.params.last_entry_minute):
            self._live_cancel_swept_entry = True
            for cand in self.candidates.values():
                if cand.live_order is not None:
                    self._cancel_live_order(cand, 'last_entry_minute')
        if not self._live_cancel_swept_flat and mnow >= getattr(self, 'flat_minute', self.params.flat_minute):
            self._live_cancel_swept_flat = True
            for cand in self.candidates.values():
                if cand.live_order is not None:
                    self._cancel_live_order(cand, 'flat_minute')

    def _poll_live_fills(self) -> None:
        """Main-thread poll of OrderStreamWatcher for our resting entry orders — the ignition-prestage pattern
        (snapshot_by_client_prefix), not per-order REST calls (API budget: stagger, never > ~100 calls/min from
        this engine). Alpaca's stop triggers on a CONSOLIDATED-TAPE print at/through the stop, NBBO-filtered;
        partial fills arrive as discrete trade_updates events with a CUMULATIVE filled_qty per order id, which is
        exactly what `get_status`'s latest-known snapshot already carries."""
        if self.order_stream is None:
            logger.warning(f"{self.tag} no OrderStreamWatcher attached — LIVE fills cannot be detected this tick"); return
        try:
            statuses = self.order_stream.snapshot_by_client_prefix(self.LIVE_COID_PREFIX)
        except Exception as e:
            logger.error(f"{self.tag} live fill poll failed: {e}"); return
        for cand in self.candidates.values():
            lo = cand.live_order
            if lo is None:
                continue
            st = statuses.get(lo['coid'])
            if st is None:
                continue
            status = str(st.get('status') or '').lower()
            if status in ('filled', 'partially_filled'):
                self._on_live_fill(cand, lo, st, status)
            elif status in _TERMINAL:
                logger.warning(f"{self.tag} {cand.symbol}: LIVE order {lo['order_id']} went {status} at the broker (not our cancel)")
                self._append_live_parity_row(cand, lo, broker_status=status, reason='broker_terminal')
                cand.live_order = None; self._live_cap_slots.discard(cand.symbol); self._persist_live_orders()

    def _on_live_fill(self, cand: Candidate, lo: dict, st: dict, status: str) -> None:
        """A real fill (full or partial) on `cand`'s resting entry order: submit safety-net TP/SL sized to the
        FULL cumulative filled qty (cancelling the previous pair first — a partial top-up must never leave a
        stale, undersized safety-net order resting) and register the position with StopMonitor exactly as the
        self-managed-stops live path elsewhere in this codebase does, tagged `strategy=self.STRATEGY_NAME` so
        scripts/hod_dry_ledger.py / EOD attribution see it as hod_break. On a PARTIAL fill the entry order is left
        resting for the remainder (Alpaca does this automatically); it is cancelled by the next bar-close replace
        or the cutoff sweep like any other resting order."""
        sym = cand.symbol
        filled_qty = int(st.get('filled_qty') or 0)
        if filled_qty <= int(lo.get('booked_qty') or 0):
            return                                            # no NEW shares since the last poll of this same status
        fill_px = float(st.get('filled_avg_price') or lo['trigger'])
        stop = lo['stop']; target = round(fill_px + self.params.target_r * (fill_px - stop), 2)
        logger.info(f"{self.tag} {sym}: LIVE {status.upper()} {fill_px:.2f} cum {filled_qty}/{lo['qty']} (stop {stop:.2f} target {target:.2f})")
        self._notify(f"{self.tag} LIVE {status.upper()} {sym} {fill_px:.2f} x{filled_qty}")
        for old_leg in (lo.get('tp_leg_id'), lo.get('sl_leg_id')):
            if old_leg:
                try: self.alpaca.cancel_order(old_leg)
                except Exception as e: logger.error(f"{self.tag} {sym}: could not cancel the stale safety-net leg {old_leg}: {e}")
        tp_id = sl_id = ''
        try:
            tp = self.alpaca.submit_limit_sell_order(symbol=sym, qty=filled_qty, limit_price=target); tp_id = str((tp or {}).get('id') or '')
        except Exception as e:
            logger.error(f"{self.tag} {sym}: safety-net TP submit failed after a LIVE fill: {e}"); self._notify(f"{self.tag} ERROR TP {sym}: {e}")
        try:
            # Safety-net SL sits SAFETY_NET_PCT below the real stop (the bull-flag live pattern): StopMonitor sells at the
            # real stop; a broker stop at the same price could fire on the same print and leave us SHORT.
            sl_px = round(stop * (1.0 - self.SAFETY_NET_PCT), 2)
            sl = self.alpaca.submit_stop_sell_order(symbol=sym, qty=filled_qty, stop_price=sl_px); sl_id = str((sl or {}).get('id') or '')
        except Exception as e:
            logger.error(f"{self.tag} {sym}: safety-net SL submit failed after a LIVE fill: {e}"); self._notify(f"{self.tag} ERROR SL {sym} — UNPROTECTED POSITION: {e}")
        pattern_data = {'book': self.book, 'level': lo['level'], 'consol_low': stop, 'entry_mode': 'resting_stop_limit',
                        'target_r': self.params.target_r, 'tp_leg_id': tp_id, 'sl_leg_id': sl_id, 'limit': lo['limit'],
                        'target': target, 'client_order_id': lo['coid']}
        trade_id = lo.get('trade_id') or self._save_pending_trade(sym, filled_qty, fill_px, stop, target, lo['order_id'], pattern_data)
        lo['booked_qty'] = filled_qty; lo['tp_leg_id'] = tp_id; lo['sl_leg_id'] = sl_id; lo['trade_id'] = trade_id
        if self.stop_monitor is not None:
            try:
                self.stop_monitor.add_watch(symbol=sym, stop_price=stop, shares=filled_qty, tp_leg_id=tp_id, sl_leg_id=sl_id,
                                            trade_db_id=trade_id, entry_price=fill_px, risk_per_share=fill_px - stop, strategy=self.STRATEGY_NAME)
            except Exception as e:
                logger.error(f"{self.tag} {sym}: StopMonitor.add_watch failed after a LIVE fill — position is UNMANAGED: {e}")
                self._notify(f"{self.tag} ERROR add_watch {sym} — UNMANAGED POSITION: {e}")
        else:
            logger.error(f"{self.tag} {sym}: no StopMonitor attached — LIVE fill has no exit management")
        self.entered_today.add(sym); self.seen_today.add(sym)
        if status == 'filled':
            cand.live_filled = True; cand.live_order = None; self._live_cap_slots.discard(sym)
            self._append_live_parity_row(cand, lo, broker_status='filled', broker_fill_ts=self._et_now(), broker_fill_px=fill_px, broker_fill_qty=filled_qty)
            try:
                if self.stop_monitor is not None: self.stop_monitor.unsubscribe([sym])
            except Exception as e:
                logger.error(f"{self.tag} {sym}: failed to unsubscribe print-watch after a LIVE fill: {e}")
            if self._entered_today_count() >= self.params.max_per_day:
                self._cancel_all_resting('day cap reached')
            elif self._open_position_count() >= self.params.max_concurrent:
                self._cancel_all_resting('fill cap reached')
        else:
            logger.warning(f"{self.tag} {sym}: PARTIAL fill {filled_qty}/{lo['qty']} — remainder stays resting, safety-net legs cover the filled qty only")
        self._persist_live_orders()

    def _append_live_parity_row(self, cand: Candidate, lo: dict, broker_status: str, broker_fill_ts=None,
                                broker_fill_px: Optional[float] = None, broker_fill_qty: Optional[int] = None, reason: str = '') -> None:
        """One row per armed signal per day in `self.live_parity_ledger_path` — EXPECTED (tape) vs ACTUAL (broker),
        docs/hod_live_resting_orders_spec_20260925.md item 3. tape_cross_ts/tape_print/tape_ask/trigger_print_nbbo_ok
        are threaded through from `cand.tape_cross` — set ONLY by `_on_trade_print` on a real tape print, so they
        stay blank for a bar-level-fallback cross or a NO_CROSS arm (there was nothing for the print watch to see).
        slippage_vs_tape_bps is computed only when both a broker fill and a tape ask are known. tape_expected is
        read from the SAME flags the dry ledger already sets on `cand`. Never raises."""
        import csv, os
        tape_expected = 'FILL' if cand.resting_filled else ('NO_FILL' if cand.resting_tape_cross_idx is not None else 'NO_CROSS')
        tc = cand.tape_cross or {}
        tape_ask = tc.get('ask')
        slippage_bps = '' if broker_fill_px is None or not tape_ask else f"{(broker_fill_px - tape_ask) / tape_ask * 1e4:.2f}"
        row = [self.session_date or '', cand.symbol, lo.get('arm_ts', ''), f"{lo['level']:.4f}", f"{lo['trigger']:.4f}", f"{lo['limit']:.4f}",
              lo.get('qty', ''), tape_expected,
              tc['ts'].isoformat() if tc.get('ts') else '', '' if tc.get('print') is None else f"{tc['print']:.4f}",
              '' if tape_ask is None else f"{tape_ask:.4f}", '' if tc.get('nbbo_ok') is None else int(bool(tc['nbbo_ok'])),
              lo['order_id'], broker_status,
              broker_fill_ts.isoformat() if broker_fill_ts else '', '' if broker_fill_px is None else f"{broker_fill_px:.4f}",
              '' if broker_fill_qty is None else broker_fill_qty, slippage_bps, reason]
        try:
            path = self.live_parity_ledger_path
            is_new = not os.path.exists(path)
            d = os.path.dirname(path)
            if d: os.makedirs(d, exist_ok=True)
            with open(path, 'a', newline='') as fh:
                w = csv.writer(fh)
                if is_new:
                    w.writerow(['date', 'symbol', 'arm_ts', 'level', 'trigger', 'limit', 'qty', 'tape_expected',
                               'tape_cross_ts', 'tape_print', 'tape_ask', 'trigger_print_nbbo_ok', 'broker_order_id',
                               'broker_status', 'broker_fill_ts', 'broker_fill_px', 'broker_fill_qty', 'slippage_vs_tape_bps', 'reason'])
                w.writerow(row)
        except Exception as e:
            logger.error(f"{self.tag} {cand.symbol}: failed to append live parity ledger row to {self.live_parity_ledger_path}: {e}")

    # ------------------------------------------------------------------ entry
    def _try_enter(self, cand: Candidate, sig, day_open: float) -> None:
        p = self.params; sym = cand.symbol
        limit = round(sig.level * (1.0 + p.cap), 2); stop = round(sig.stop, 2)
        if sig.level < self.min_price:                        # level-based rules first: the spec applies them before the book
            cand.rejected_reason = 'price'; logger.info(f"{self.tag} {sym}: level {sig.level:.2f} below the ${self.min_price:.0f} floor — skip"); return
        if stop >= limit:
            cand.rejected_reason = 'r_min'; return
        blocked = self._kill_rails_blocked()
        if blocked:
            cand.rejected_reason = blocked; self._notify_once(blocked, f"{self.tag} {blocked}: no new entries"); return
        if self._entered_today_count() >= p.max_per_day or len(self.positions) >= p.max_concurrent:
            # Resolve the book BEFORE judging the caps: an exit or a no-fill that already happened at the broker frees its
            # slot now (the spec's causal rule: an exit on a bar strictly before this entry bar), not on the next 60 s tick.
            self._process_pending_fills(); self.check_exits()
        if self._entered_today_count() >= p.max_per_day:
            cand.rejected_reason = 'day_cap'; logger.info(f"{self.tag} {sym}: per-day cap {p.max_per_day} reached — skip"); return
        if len(self.positions) >= p.max_concurrent:
            logger.info(f"{self.tag} {sym}: concurrency cap {p.max_concurrent} — skip (signal not re-armed)"); cand.rejected_reason = 'concurrency'; return
        q = self._quote(sym)
        if q is None:
            cand.rejected_reason = 'no_quote'; logger.warning(f"{self.tag} {sym}: no quote — fail closed, no order"); return
        bid, ask = q
        spread_bps = (ask - bid) / ask * 1e4 if ask > 0 else 1e9
        if spread_bps > self.max_spread_bps:
            cand.rejected_reason = 'spread'; logger.info(f"{self.tag} {sym}: spread {spread_bps:.0f} bps > {self.max_spread_bps:.0f} — skip"); return
        if ask > limit:
            cand.rejected_reason = 'no_chase'; logger.info(f"{self.tag} {sym}: ask {ask:.2f} above cap {limit:.2f} — NO CHASE, skip"); return
        # R, the r_min gate and the size use the EXPECTED FILL = the ask (the spec uses the next bar's open, the price
        # actually paid); measuring against the limit over-states R and let tight stops through (9/14 EOD parity: 11 of 35).
        entry_est = round(ask, 2)
        r = entry_est - stop
        if r <= 0 or r / entry_est * 100.0 < p.min_r_pct:
            cand.rejected_reason = 'r_min'; logger.info(f"{self.tag} {sym}: stop {stop:.2f} within {p.min_r_pct}% of the ask {entry_est:.2f} — skip"); return
        if self.max_spread_frac_r > 0 and (ask - bid) / r > self.max_spread_frac_r:
            cand.rejected_reason = 'spread_r'; logger.info(f"{self.tag} {sym}: spread {spread_bps:.0f} bps = {(ask - bid) / r:.0%} of R {r:.2f} > {self.max_spread_frac_r:.0%} — skip"); return
        target = round(entry_est + p.target_r * r, 2)      # target from the EXPECTED fill (the spec's basis); a fill at the limit makes the real target slightly < 2R
        shares = shares_for(self.risk_usd, entry_est, stop)
        cap_shares = int(self.max_notional_usd // limit)
        if shares > cap_shares:
            logger.warning(f"{self.tag} {sym}: notional cap ${self.max_notional_usd:,.0f} binds — {shares} → {cap_shares} shares (risk ${cap_shares * r:.0f} of ${self.risk_usd:.0f}; the backtest sized by risk alone)")
            shares = cap_shares
        if shares < 1:
            cand.rejected_reason = 'size'; return
        msg = (f"{sym} level {sig.level:.2f} limit {limit:.2f} stop {stop:.2f} target {target:.2f} R {r:.2f} ({r / entry_est * 100:.1f}%) "
               f"x{shares} | +{sig.dist_open_pct:.1f}% from open, rv {sig.rv_profile:.1f}, spread {spread_bps:.0f} bps, ask {entry_est:.2f}"
               + (f", prior close {cand.prior_close:.2f}, pdr {cand.pdr_pct:.1f}%" if self.book == 'red_to_green' and cand.pdr_pct is not None else ""))
        if self.dry_run:
            if not cand.dry_logged:
                cand.dry_logged = True; cand.rejected_reason = 'dry_run'
                logger.info(f"{self.dry_tag} WOULD BUY {msg}"); self._notify(f"{self.dry_tag} WOULD BUY {msg}")
            return
        coid = f"{self.coid_prefix}-{sym}-{(self.session_date or '')[5:]}-{uuid.uuid4().hex[:8]}"[:48]   # OUR id: the only key we ever adopt by
        try:
            od = self.alpaca.submit_bracket_order(symbol=sym, qty=shares, side='buy', limit_price=limit, tp_price=target, sl_price=stop, client_order_id=coid)
        except Exception as e:
            od = self._adopt_open_buy(sym, coid)               # a client-side timeout may have left OUR accepted order at the broker
            if od is None:
                cand.rejected_reason = 'submit_failed'; logger.error(f"{self.tag} {sym}: submit failed: {e}"); self._notify(f"{self.tag} ERROR submit {sym}: {e}"); return
            logger.warning(f"{self.tag} {sym}: submit raised ({e}) but our order {coid} exists at the broker — adopted {od.get('id')}")
        order_id = str(od.get('id') or ''); tp_id = sl_id = None
        for leg in od.get('legs') or []:
            if leg.get('limit_price') is not None and leg.get('stop_price') is None: tp_id = str(leg.get('id'))
            elif leg.get('stop_price') is not None: sl_id = str(leg.get('id'))
        if not order_id:
            od2 = self._adopt_open_buy(sym, coid)              # the broker may have it under our client id
            order_id = str((od2 or {}).get('id') or '')
            if not order_id:
                cand.rejected_reason = 'submit_failed'
                logger.error(f"{self.tag} {sym}: submit returned no order id and {coid} is not at the broker — nothing tracked, no order assumed"); self._notify(f"{self.tag} ERROR {sym}: order id missing"); return
        now = datetime.now(timezone.utc)
        pd_ = self._pattern_data(sig, tp_id, sl_id, bid, ask, limit, target, coid)
        trade_id = self._save_pending_trade(sym, shares, limit, stop, target, order_id, pd_)
        self.positions[sym] = Position(symbol=sym, trade_id=trade_id, order_id=order_id, shares=shares, limit_price=limit, stop=stop, target=target,
                                       level=sig.level, submitted_at=now, tp_leg_id=tp_id, sl_leg_id=sl_id, client_order_id=coid, pattern_data=pd_,
                                       fill_at_estimate_r=r)
        self.entered_today.add(sym); self.seen_today.add(sym); cand.rejected_reason = 'ordered'
        logger.info(f"{self.tag} ENTRY SUBMITTED {msg} order {order_id} ({coid})"); self._notify(f"{self.tag} BUY {msg}")

    def _adopt_open_buy(self, symbol: str, client_order_id: str) -> Optional[dict]:
        """Find OUR order by client_order_id — never by symbol/side (the owner trades manually on the same account)."""
        try:
            for od in self.alpaca.get_open_orders() or []:
                if str(od.get('client_order_id') or '') == client_order_id:
                    return od
        except Exception as e:
            logger.warning(f"{self.tag} {symbol}: open-orders reconcile failed ({e})")
        return None

    def _quote(self, symbol: str, tries: int = 3):
        """Latest NBBO (bid, ask) or None. A quote older than max_quote_age_s (halted name, stalled feed) is refused:
        the spec's fill is the next print, and a pre-halt NBBO would queue a limit for the reopen."""
        for k in range(tries):
            try:
                q = self.alpaca.get_latest_quote(symbol)
                bid = float(q.get('bid_price') or 0); ask = float(q.get('ask_price') or 0)
                ts = q.get('timestamp')
                if ts and self.max_quote_age_s > 0:
                    t = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
                    if t.tzinfo is None: t = t.replace(tzinfo=timezone.utc)
                    age = (datetime.now(timezone.utc) - t).total_seconds()
                    if age > self.max_quote_age_s:
                        logger.warning(f"{self.tag} {symbol}: quote is {age:.0f}s old (halted or stalled feed) — refused"); return None
                if bid > 0 and ask > 0: return (bid, ask)
            except Exception as e:
                logger.warning(f"{self.tag} {symbol}: quote failed ({e}) try {k + 1}/{tries}")
            if k + 1 < tries: time.sleep(0.5)
        return None

    def _pattern_data(self, sig, tp_id, sl_id, bid, ask, limit, target, coid) -> dict:
        return {'book': self.book, 'level': sig.level, 'consol_low': sig.stop, 'dist_open_pct': sig.dist_open_pct, 'rv_profile': sig.rv_profile,
                'cap': self.params.cap, 'target_r': self.params.target_r, 'tp_leg_id': tp_id, 'sl_leg_id': sl_id,
                'quote_bid': bid, 'quote_ask': ask, 'limit': limit, 'target': target, 'client_order_id': coid}

    def _save_pending_trade(self, sym, shares, limit, stop, target, order_id, pattern_data: dict) -> Optional[int]:
        rec = {
            'trade_date': self.session_date or self._et_now().strftime('%Y-%m-%d'), 'symbol': sym, 'side': 'buy', 'entry_price': limit,
            'stop_loss_price': stop, 'take_profit_price': target, 'shares': shares, 'risk_per_share': limit - stop, 'total_risk': (limit - stop) * shares,
            'risk_reward_ratio': self.params.target_r, 'order_id': order_id, 'order_status': 'pending_new', 'fill_price': None, 'filled_at': None,
            'exit_price': None, 'exit_reason': None, 'exited_at': None, 'pnl': None, 'pnl_pct': None, 'strategy': self.STRATEGY_NAME,
            'pattern_data': json.dumps(pattern_data),
        }
        try:
            return int(self.db.save_trade(rec))
        except Exception as e:
            logger.error(f"{self.tag} {sym}: save_trade failed ({e}) — order {order_id} is NOT in the DB"); self._notify(f"{self.tag} ERROR DB {sym}: {e}"); return None

    def _update_pattern_data(self, pos: Position, **kv) -> None:
        pos.pattern_data.update(kv)
        if pos.trade_id is not None:
            try: self.db.update_trade(pos.trade_id, {'pattern_data': json.dumps(pos.pattern_data)})
            except Exception as e: logger.error(f"{self.tag} {pos.symbol}: pattern_data update failed: {e}")

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
                logger.warning(f"{self.tag} get_order {order_id} failed ({e})"); return None
        return st

    def _process_pending_fills(self, force_timeout: Optional[str] = None) -> None:
        """Poll our working entry orders. Runs from the drain loop every ~1 s (not only from the scanner's 60 s cycle —
        the order life must really be `order_timeout_s`). `force_timeout=symbol`: that symbol's next bar closed, so the
        spec's fill window (the next bar's open) is over whatever the wall clock says."""
        self._last_pending_check = time.time()
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
            if age >= self.order_timeout_s or (force_timeout == sym and age >= 3.0):
                try: self.alpaca.cancel_order(pos.order_id)
                except Exception as e: logger.warning(f"{self.tag} {sym}: cancel failed ({e})")
                st2 = self._order_status(pos.order_id, rest=True) or {}
                s2 = str(st2.get('status', '')).lower()
                if int(st2.get('filled_qty') or 0) > 0: self._confirm_fill(pos, st2)
                elif s2 in _TERMINAL: self._drop_pending(pos, 'time_stop_canceled'); logger.info(f"{self.tag} {sym}: unfilled after {age:.0f}s — canceled (no chase)")
                else: logger.warning(f"{self.tag} {sym}: cancel not confirmed (status {s2 or 'unknown'}) — order kept as pending, retry next tick")

    def _drop_pending(self, pos: Position, status: str) -> None:
        self.positions.pop(pos.symbol, None); self.entered_today.discard(pos.symbol)   # a no-fill does not consume a day slot (once-per-symbol stays)
        if pos.trade_id is not None:
            try: self.db.update_trade(pos.trade_id, {'order_status': status})
            except Exception as e: logger.error(f"{self.tag} {pos.symbol}: DB update ({status}) failed: {e}")

    def _confirm_fill(self, pos: Position, st: dict) -> None:
        fq = int(st.get('filled_qty') or 0)
        if fq <= 0:
            logger.warning(f"{self.tag} {pos.symbol}: fill confirmed without filled_qty — assuming the full {pos.shares}"); fq = pos.shares
        px = float(st.get('filled_avg_price') or 0)
        if px <= 0:
            logger.warning(f"{self.tag} {pos.symbol}: fill confirmed without filled_avg_price — booking the limit {pos.limit_price:.2f} (worst case)"); px = pos.limit_price
        pos.fill_price = px; pos.filled_at = datetime.now(timezone.utc); pos.shares = fq; pos.status = 'open'
        self._anchor_target_to_fill(pos)
        r_fill = px - pos.stop; fill_delay = (pos.filled_at - pos.submitted_at).total_seconds()
        spec_no_trade = r_fill <= 0 or r_fill / px * 100.0 < self.params.min_r_pct   # the spec would have skipped this fill (r_min on the fill)
        self._update_pattern_data(pos, fill_delay_s=round(fill_delay, 1), risk_at_fill=round(r_fill * fq, 2), spec_no_trade=bool(spec_no_trade))
        if pos.trade_id is not None:
            try: self.db.update_trade(pos.trade_id, {'order_status': 'filled', 'fill_price': px, 'filled_at': pos.filled_at.isoformat(), 'shares': fq, 'filled_qty': fq,
                                                     'take_profit_price': pos.target, 'risk_per_share': r_fill, 'total_risk': r_fill * fq})
            except Exception as e: logger.error(f"{self.tag} {pos.symbol}: DB fill update failed: {e}")
        logger.info(f"{self.tag} FILLED {pos.symbol} x{fq} @ {px:.2f} after {fill_delay:.1f}s (limit {pos.limit_price:.2f}, slip {(px / pos.level - 1) * 1e4:.0f} bps vs level, "
                    f"risk ${r_fill * fq:.0f} vs ${self.risk_usd:.0f} planned{', SPEC WOULD SKIP: r_min on the fill' if spec_no_trade else ''})")
        self._notify(f"{self.tag} FILLED {pos.symbol} x{fq} @ {px:.2f} stop {pos.stop:.2f} target {pos.target:.2f}")

    def _anchor_target_to_fill(self, pos: Position) -> None:
        """The spec's target is entry + target_r × (entry − stop) on the ACTUAL fill. The bracket was submitted with the
        target from the expected fill (the ask); once the real fill is known, move the take-profit leg to the spec's price."""
        if not pos.fill_price or not pos.tp_leg_id or pos.fill_price <= pos.stop:
            return
        t2 = round(pos.fill_price + self.params.target_r * (pos.fill_price - pos.stop), 2)
        if abs(t2 - pos.target) < 0.01:
            return
        old_id = pos.tp_leg_id
        try:
            res = self.alpaca.replace_order_limit_price(old_id, t2) or {}
            new_id = str(res.get('id') or '')
            # Alpaca's replace creates a NEW order (the old one becomes 'replaced'): track the new id or every later
            # target fill is invisible to check_exits and the 15:55 flat would sell shares we no longer hold.
            st = self._order_status(new_id, rest=True) if new_id else None
            status = str((st or {}).get('status', '')).lower()
            if not new_id or status in ('rejected', 'canceled', 'cancelled', 'expired'):
                logger.error(f"{self.tag} {pos.symbol}: take-profit replace to {t2:.2f} not effective (new id {new_id or 'none'}, status {status or 'unknown'}) — leg {old_id} stays at {pos.target:.2f}")
                return
            pos.tp_leg_id = new_id; pos.target = t2
            self._update_pattern_data(pos, tp_leg_id=new_id, tp_leg_replaced=old_id, target=t2)
            logger.info(f"{self.tag} {pos.symbol}: target re-anchored to the fill: {pos.target:.2f} (fill {pos.fill_price:.2f}, stop {pos.stop:.2f}); TP leg {old_id} → {new_id}")
        except Exception as e:
            logger.error(f"{self.tag} {pos.symbol}: take-profit re-anchor to {t2:.2f} FAILED ({e}) — leg {old_id} stays at {pos.target:.2f} (target {'below' if t2 > pos.target else 'above'} the spec's)")

    def _exit_legs(self, pos: Position):
        return ((pos.close_order_id, 'eod'), (pos.tp_leg_id, 'target'), (pos.sl_leg_id, 'stop'))

    def _leg_status(self, pos: Position, leg_id: str, reason: str, rest: bool = False):
        """Status of an exit leg, following a replaced order to its successor (Alpaca's replace creates a new id)."""
        st = self._order_status(leg_id, rest=rest)
        if st and str(st.get('status', '')).lower() == 'replaced' and st.get('replaced_by'):
            new_id = str(st['replaced_by']); logger.warning(f"{self.tag} {pos.symbol}: {reason} leg {leg_id} was replaced by {new_id} — following it")
            if reason == 'target': pos.tp_leg_id = new_id; self._update_pattern_data(pos, tp_leg_id=new_id)
            elif reason == 'stop': pos.sl_leg_id = new_id; self._update_pattern_data(pos, sl_leg_id=new_id)
            else: pos.close_order_id = new_id; self._update_pattern_data(pos, close_order_id=new_id)
            st = self._order_status(new_id, rest=rest); leg_id = new_id
        return leg_id, st

    def _book_leg_fill(self, pos: Position, leg_id: str, st: dict, reason: str) -> int:
        """Book the shares an exit order has sold so far (partials included) — qty-based, so a partial take-profit
        followed by the stop, or a partially filled close order, is priced exactly as it happened. Returns the new shares
        booked. When every share is sold the position is recorded as exited (blended price, reason = the last seller)."""
        fq = int(st.get('filled_qty') or 0); px = float(st.get('filled_avg_price') or 0.0)
        new = fq - pos.leg_booked.get(leg_id, 0)
        if new <= 0 or px <= 0:
            return 0
        pos.leg_booked[leg_id] = fq; pos.closed_qty += new; pos.closed_notional += new * px; pos.last_close_reason = reason
        if pos.closed_qty < pos.shares:
            logger.info(f"{self.tag} {pos.symbol}: {reason} sold {new} @ {px:.2f} — {pos.open_qty} still held (partial)")
            self._update_pattern_data(pos, closed_qty=pos.closed_qty, closed_notional=round(pos.closed_notional, 2), partial_exit=reason)
        else:
            self._record_exit(pos, pos.closed_notional / pos.closed_qty, reason if len(set(pos.leg_booked)) == 1 else f'{reason}+partial')
        return new

    def check_exits(self, rest: bool = False) -> List[str]:
        """Poll the bracket legs (and the force-close order) of open positions; book fills; record completed exits."""
        done = []
        for sym, pos in list(self.positions.items()):
            if pos.status != 'open':
                continue
            for leg_id, reason in self._exit_legs(pos):
                if not leg_id:
                    continue
                leg_id, st = self._leg_status(pos, leg_id, reason, rest=rest)
                if st and int(st.get('filled_qty') or 0) > 0:
                    self._book_leg_fill(pos, leg_id, st, reason)
                    if sym not in self.positions:
                        done.append(sym)
                        for other in (pos.tp_leg_id, pos.sl_leg_id, pos.close_order_id):   # belt and braces: nothing may keep selling
                            if other and other != leg_id:
                                try: self.alpaca.cancel_order(other)
                                except Exception as e: logger.warning(f"{self.tag} {sym}: sibling cancel {other} failed ({e})")
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
            upd['order_status'] = 'exit_pending_verification'; logger.warning(f"{self.tag} {pos.symbol}: exit price unknown — pending verification")
        if pos.trade_id is not None:
            try: self.db.update_trade(pos.trade_id, upd)
            except Exception as e: logger.error(f"{self.tag} {pos.symbol}: DB exit update failed: {e}")
        rr = (exit_price - entry) / (entry - pos.stop) if entry > pos.stop and exit_price > 0 else float('nan')
        logger.info(f"{self.tag} EXIT {pos.symbol} {reason} @ {exit_price:.2f} pnl {pnl if pnl is None else round(pnl, 2)} ({rr:+.2f}R) day {self.daily_pnl:+.0f}")
        self._notify(f"{self.tag} EXIT {pos.symbol} {reason} @ {exit_price:.2f} pnl {'?' if pnl is None else f'{pnl:+.0f}'} ({rr:+.2f}R) | day {self.daily_pnl:+.0f}")

    # ------------------------------------------------------------------ force close
    def is_force_close_time(self) -> bool:
        return self._minute_of_day() >= getattr(self, 'flat_minute', self.params.flat_minute)

    FC_RESUBMIT_S = 60.0
    FC_LEG_POLL_S = 3.0

    def _cancel_pending_and_confirm(self, pos: Position, why: str) -> Optional[str]:
        """Cancel a working entry and read its REST truth: 'filled' (partial or full — the position is now open),
        'dropped' (nothing filled, terminal) or None (cancel not yet confirmed: keep it pending, never assume)."""
        try: self.alpaca.cancel_order(pos.order_id)
        except Exception as e: logger.warning(f"{self.tag} {pos.symbol}: cancel failed ({e})")
        st = self._order_status(pos.order_id, rest=True) or {}
        status = str(st.get('status', '')).lower()
        if int(st.get('filled_qty') or 0) > 0:
            self._confirm_fill(pos, st); return 'filled'
        if status in _TERMINAL:
            self._drop_pending(pos, why); return 'dropped'
        logger.warning(f"{self.tag} {pos.symbol}: cancel not confirmed (status {status or 'unknown'}) — order kept as pending, retry next pass")
        return None

    def _settle_exit_legs(self, pos: Position) -> None:
        """Cancel the bracket legs and READ them until terminal (≤ FC_LEG_POLL_S): a leg that filled in the meantime is
        booked, never sold twice. Nothing is sold on assumptions."""
        for leg, reason in ((pos.tp_leg_id, 'target'), (pos.sl_leg_id, 'stop')):
            if not leg: continue
            try: self.alpaca.cancel_order(leg)
            except Exception as e: logger.warning(f"{self.tag} FC leg cancel {pos.symbol} failed ({e}) — reading the leg")
        deadline = time.time() + self.FC_LEG_POLL_S
        while True:
            pending = False
            for leg, reason in ((pos.tp_leg_id, 'target'), (pos.sl_leg_id, 'stop')):
                if not leg or pos.symbol not in self.positions: continue
                leg, st = self._leg_status(pos, leg, reason, rest=True)
                st = st or {}
                if int(st.get('filled_qty') or 0) > 0: self._book_leg_fill(pos, leg, st, reason)
                if str(st.get('status', '')).lower() not in _TERMINAL + ('filled', 'replaced'): pending = True
            if not pending or pos.symbol not in self.positions or time.time() >= deadline: break
            time.sleep(0.25)
        if pending and pos.symbol in self.positions:
            logger.warning(f"{self.tag} FORCE CLOSE {pos.symbol}: a leg is still not terminal after {self.FC_LEG_POLL_S:.0f}s — selling only the {pos.open_qty} shares no leg has sold")

    def _close_reference_price(self, pos: Position) -> float:
        q = self._quote(pos.symbol)
        if q: return q[0]
        cand = self.candidates.get(pos.symbol)
        if cand is not None and cand.n_bars:
            last = cand.ohlcv[np.flatnonzero(cand.have)[-1]][3]
            logger.warning(f"{self.tag} FORCE CLOSE {pos.symbol}: no quote — using the last bar close {last:.2f}"); return float(last)
        logger.warning(f"{self.tag} FORCE CLOSE {pos.symbol}: no quote and no bars — using the fill {pos.fill_price or pos.limit_price:.2f}")
        return float(pos.fill_price or pos.limit_price)

    def force_close_all(self) -> int:
        """Flatten OUR shares at flat_minute: cancel working entries (REST-confirmed), cancel and READ the legs (a fill
        in the race is booked, not sold again), then sell exactly the shares no exit has sold, with a marketable limit
        whose id is persisted (a restart must not sell twice). Never `close_position` (the owner trades the same
        account). Re-checked every pass; re-submitted after FC_RESUBMIT_S; a third attempt goes 3% through the bid."""
        n = 0; now = datetime.now(timezone.utc)
        with self._lock:
            for sym, pos in list(self.positions.items()):
                if pos.status == 'pending':
                    if self._cancel_pending_and_confirm(pos, 'time_stop_canceled') != 'filled':
                        continue
                if pos.close_order_id:
                    cid, st = self._leg_status(pos, pos.close_order_id, 'eod', rest=True); st = st or {}
                    status = str(st.get('status', '')).lower(); age = (now - (pos.close_submitted_at or now)).total_seconds()
                    if int(st.get('filled_qty') or 0) > 0: self._book_leg_fill(pos, cid, st, 'eod')
                    if sym not in self.positions: continue
                    if status not in _TERMINAL + ('filled',) and age < self.FC_RESUBMIT_S:
                        continue                                       # still working
                    if status not in _TERMINAL + ('filled',):
                        try: self.alpaca.cancel_order(cid)
                        except Exception: pass
                        st2 = self._order_status(cid, rest=True) or {}
                        if int(st2.get('filled_qty') or 0) > 0: self._book_leg_fill(pos, cid, st2, 'eod')
                        if sym not in self.positions: continue
                    logger.warning(f"{self.tag} FORCE CLOSE {sym}: close order {cid} {status or 'unknown'} after {age:.0f}s — re-submitting the remaining {pos.open_qty}")
                self._settle_exit_legs(pos)
                if sym not in self.positions: continue
                qty = pos.open_qty
                if qty <= 0:
                    logger.warning(f"{self.tag} FORCE CLOSE {sym}: nothing left to sell after the legs were read"); continue
                pos.fc_attempts += 1
                ref = self._close_reference_price(pos); limit = round(ref * (0.97 if pos.fc_attempts >= 3 else 0.99), 2)
                coid = f"hod-fc-{sym}-{(self.session_date or '')[5:]}-{uuid.uuid4().hex[:6]}"[:48]
                try:
                    od = self.alpaca.submit_limit_sell_order(sym, qty, limit, **({'client_order_id': coid} if self._client_supports_coid('submit_limit_sell_order') else {}))
                    pos.close_order_id = str(od.get('id') or '') or None; pos.close_submitted_at = now; n += 1
                    self._update_pattern_data(pos, close_order_id=pos.close_order_id, close_client_order_id=coid, close_submitted_at=now.isoformat(), closed_qty=pos.closed_qty)
                    logger.info(f"{self.tag} FORCE CLOSE {sym} x{qty} limit {limit:.2f} submitted ({pos.close_order_id}, attempt {pos.fc_attempts})")
                except Exception as e:
                    logger.error(f"{self.tag} FORCE CLOSE {sym} FAILED: {e}"); self._notify(f"{self.tag} ERROR force close {sym}: {e}")
            remaining = [s_ for s_, p_ in self.positions.items() if p_.status == 'open']
            self._flattened = not remaining
        if n: self._notify(f"{self.tag} flat at {getattr(self, 'flat_minute', self.params.flat_minute) // 60:02d}:{getattr(self, 'flat_minute', self.params.flat_minute) % 60:02d} ET — {n} close order(s) submitted, {len(remaining)} still open")
        return n

    def _client_supports_coid(self, method: str) -> bool:
        import inspect
        try: return 'client_order_id' in inspect.signature(getattr(self.alpaca, method)).parameters
        except (TypeError, ValueError): return False

    def reconcile_pending_exits(self, days: int = 7) -> int:
        """Rows left `exit_pending_verification` (exit price unknown, broker held fewer shares, dead-man flat): read the
        exit orders named in pattern_data from REST and write the truth — closed with P&L when the legs/close sold every
        share, back to open (re-adopted) when the broker still holds them. Without this the kill rails never see those
        losses."""
        path = getattr(self.db, '_trades_path', None)
        if not path: return 0
        try:
            conn = sqlite3.connect(str(path), timeout=10)
            try:
                rows = conn.execute("SELECT id, symbol, shares, fill_price, entry_price, pattern_data, trade_date FROM trades WHERE strategy=? AND order_status='exit_pending_verification' "
                                    "AND trade_date >= date('now', ?)", (self.STRATEGY_NAME, f'-{days} days')).fetchall()
            finally: conn.close()
        except Exception as e:
            logger.error(f"{self.tag} reconcile: DB read failed ({e})"); return 0
        n = 0
        for tid, sym, shares, fill, entry, pdj, tdate in rows:
            try: pd_ = json.loads(pdj or '{}')
            except Exception: pd_ = {}
            pos = Position(symbol=sym, trade_id=tid, order_id='', shares=int(shares or 0), limit_price=float(entry or 0), stop=0.0, target=0.0, level=0.0,
                           submitted_at=datetime.now(timezone.utc), tp_leg_id=pd_.get('tp_leg_id'), sl_leg_id=pd_.get('sl_leg_id'), close_order_id=pd_.get('close_order_id'),
                           fill_price=fill, pattern_data=dict(pd_), status='open')
            self.positions[sym] = pos                              # _book_leg_fill records through the normal path
            for leg, reason in self._exit_legs(pos):
                if not leg or sym not in self.positions: continue
                leg, st = self._leg_status(pos, leg, reason, rest=True)
                if st and int(st.get('filled_qty') or 0) > 0: self._book_leg_fill(pos, leg, st, reason)
            if sym in self.positions:
                self.positions.pop(sym)
                logger.error(f"{self.tag} reconcile {sym} ({tdate}): {pos.closed_qty}/{pos.shares} shares accounted for by its exit orders — still exit_pending_verification, needs a human look")
                self._notify(f"{self.tag} UNRECONCILED {sym} {tdate}: {pos.closed_qty}/{pos.shares} shares sold by our orders")
            else:
                n += 1
        if rows: logger.info(f"{self.tag} reconcile: {n} of {len(rows)} exit_pending_verification rows resolved")
        return n

    # ------------------------------------------------------------------ rails / caps
    def _realized_pnl(self, since: str) -> float:
        path = getattr(self.db, '_trades_path', None)
        if not path:
            logger.warning(f"{self.tag} realized-pnl: db has no _trades_path — FAIL CLOSED"); return -1e9
        try:
            conn = sqlite3.connect(str(path), timeout=10)
            try:
                v = conn.execute("SELECT COALESCE(SUM(pnl),0) FROM trades WHERE strategy=? AND trade_date>=?", (self.STRATEGY_NAME, since)).fetchone()[0]
            finally: conn.close()
            return float(v or 0.0)
        except Exception as e:
            logger.error(f"{self.tag} realized-pnl query failed ({e}) — FAIL CLOSED"); return -1e9

    def _unverified_exits(self, days: int = 7) -> int:
        path = getattr(self.db, '_trades_path', None)
        if not path: return 1
        try:
            conn = sqlite3.connect(str(path), timeout=10)
            try:
                return int(conn.execute("SELECT COUNT(*) FROM trades WHERE strategy=? AND order_status='exit_pending_verification' AND trade_date >= date('now', ?)",
                                        (self.STRATEGY_NAME, f'-{days} days')).fetchone()[0])
            finally: conn.close()
        except Exception as e:
            logger.error(f"{self.tag} unverified-exit query failed ({e}) — FAIL CLOSED"); return 1

    def _kill_rails_blocked(self) -> Optional[str]:
        now = self._et_now(); today = now.strftime('%Y-%m-%d'); week = (now - timedelta(days=now.weekday())).strftime('%Y-%m-%d')
        if self._unverified_exits() > 0: return 'unverified_exit'      # a loss the rails cannot see = no new risk until it is written
        if self._realized_pnl(week) <= self.weekly_kill_usd: return 'weekly_kill'
        if self._realized_pnl(today) <= self.daily_kill_usd: return 'daily_kill'
        return None

    _DEAD = _TERMINAL + ('time_stop_canceled',)            # every no-fill status: never a day slot

    def _db_symbols_today(self, include_dead: bool = False) -> set:
        """Symbols with a hod_break row today: non-dead rows (open, closed, pending) feed the per-day cap; with
        `include_dead` every row (no-fills too) feeds once-per-symbol. Restart-safe. Empty set (memory only) on DB failure."""
        today = self.session_date or self._et_now().strftime('%Y-%m-%d'); path = getattr(self.db, '_trades_path', None)
        if not path: return set()
        try:
            conn = sqlite3.connect(str(path), timeout=10)
            try:
                if include_dead:
                    rows = conn.execute("SELECT DISTINCT symbol FROM trades WHERE strategy=? AND trade_date=?", (self.STRATEGY_NAME, today)).fetchall()
                else:
                    rows = conn.execute("SELECT DISTINCT symbol FROM trades WHERE strategy=? AND trade_date=? AND COALESCE(order_status,'') NOT IN (%s)"
                                        % ','.join('?' * len(self._DEAD)), (self.STRATEGY_NAME, today, *self._DEAD)).fetchall()
            finally: conn.close()
            return {r[0] for r in rows}
        except Exception as e:
            logger.warning(f"{self.tag} symbols-today DB query failed ({e}) — using memory only"); return set()

    def _entered_today_count(self) -> int:
        db_syms = self._db_symbols_today(); self.entered_today |= db_syms
        return len(self.entered_today)

    # ------------------------------------------------------------------ restart
    def sync_positions(self) -> int:
        """Rebuild pending/open positions from the trades DB for today (restart-safe)."""
        self._roll_session(); today = self.session_date; n = 0
        try: rows = self.db.get_open_trades(today, strategy=STRATEGY_NAME)
        except Exception as e:
            logger.error(f"{self.tag} sync_positions: DB read failed ({e})"); return 0
        broker: Optional[Dict[str, int]] = None
        try: broker = {p.get('symbol'): int(float(p.get('qty') or 0)) for p in (self.alpaca.get_open_positions() or [])}
        except Exception as e: logger.warning(f"{self.tag} sync_positions: broker positions unavailable ({e}) — DB rows trusted as-is")
        for r in rows:
            sym = r['symbol']; pd_ = {}
            try: pd_ = json.loads(r.get('pattern_data') or '{}')
            except Exception: pass
            pos = Position(symbol=sym, trade_id=r.get('id'), order_id=str(r.get('order_id') or ''), shares=int(r.get('shares') or 0), limit_price=float(r.get('entry_price') or 0),
                           stop=float(r.get('stop_loss_price') or 0), target=float(r.get('take_profit_price') or 0), level=float(pd_.get('level') or 0),
                           submitted_at=datetime.now(timezone.utc) - timedelta(seconds=self.order_timeout_s),   # a restored pending order is past its window: the next poll cancels-or-confirms it
                           tp_leg_id=pd_.get('tp_leg_id'), sl_leg_id=pd_.get('sl_leg_id'), fill_price=r.get('fill_price'),
                           client_order_id=pd_.get('client_order_id'), pattern_data=dict(pd_), close_order_id=pd_.get('close_order_id'),
                           closed_qty=int(pd_.get('closed_qty') or 0), closed_notional=float(pd_.get('closed_notional') or 0.0))
            if pos.close_order_id and pd_.get('close_submitted_at'):
                try: pos.close_submitted_at = datetime.fromisoformat(pd_['close_submitted_at'])
                except Exception: pos.close_submitted_at = None
            status = r.get('order_status')
            if status == 'pending_new':
                pos.status = 'pending'
            elif status in _OPEN_STATUSES:
                pos.status = 'open'
                if broker is not None and broker.get(sym, 0) < pos.open_qty:
                    logger.warning(f"{self.tag} sync: {sym} open in DB ({r.get('shares')} sh) but broker holds {broker.get(sym, 0)} — exit pending verification")
                    try: self.db.update_trade(r['id'], {'order_status': 'exit_pending_verification'})
                    except Exception: pass
                    continue
            else:
                continue
            self.positions[sym] = pos; self.entered_today.add(sym); self.seen_today.add(sym); n += 1
        self.seen_today |= self._db_symbols_today(include_dead=True); self.entered_today |= self._db_symbols_today()
        logger.info(f"{self.tag} sync_positions: {n} position(s) rehydrated for {today}")
        try: self.reconcile_pending_exits()
        except Exception as e: logger.error(f"{self.tag} reconcile_pending_exits failed: {e}")
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
            logger.warning(f"{self.tag} Telegram notify FAILED ({e}): {msg[:80]}")

    def _notify_once(self, key: str, msg: str) -> None:
        if key in self._kill_notified: return
        self._kill_notified.add(key); logger.warning(msg); self._notify(msg)
