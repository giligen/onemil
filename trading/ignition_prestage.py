"""Ignition pre-staged entry subsystem ("prestage") — 2026-08-22 build.

WHY (docs/ignition_prestage_design_aug2026.md): live ignition chases the
ask 30-60s after the level cross and pays ~180bps median. The resting
model (research/ignition_capcheck/RESTING_MODEL.md) validates a
stop-limit RESTING at the level before the cross: stop = level*1.003,
limit = level*(1+300bps), filling on the way up. Placement is
everything; detection speed is worthless.

WHAT THIS MODULE OWNS
  - PrestageManager: candidate intake (shadow sighting flow), proximity
    ranking with dual-threshold hysteresis, a staging scheduler under a
    rate budget (ops/min) + buying-power watermark budget, the
    per-(symbol, day) stage state machine (P0-3), fill adoption with
    AT-FILL structure validation (P0-1/P0-4/P0-5), and every sweep path
    (boot §A1, kill §C10, feed watchdog §D12, 13:00, shutdown §A2).
  - SHADOW MODE (config `ignition_live.prestage.shadow: true`, the
    default): every decision runs — heap, scheduler, would-stage /
    would-cancel / would-fill (fill inferred from the tape crossing the
    stop while 'staged') — logged as [PRESTAGE SHADOW] + telemetry,
    ZERO orders. Real chase behavior is never gated in shadow mode.

INVARIANTS (each has a named test in tests/test_ignition_prestage.py)
  - Take-all coverage: staging changes PRICE, never COVERAGE — any
    shadow trigger with no staged fill falls to the chase path; the
    nightly set-equality parity (check_set_parity) is a HARD gate.
  - One stage per (symbol, day) EVER — client_order_id
    `ign-stage-{YYYYMMDD}-{sym}` is broker-enforced idempotency (§A5).
  - Never stage before 9:35 ET; never while last >= level (P0-6 —
    already-crossed names route to the chase path with an explicit
    parity reason). Cancel-all at 13:00 ET.
  - Cancel-reject != cancelled: poll the order; if filled, ADOPT (§A3).
  - Chase entry for a staged symbol only after broker-confirmed
    terminal-and-unfilled disposition (P0-3, resolve_for_chase).
  - A fill that raced the kill sweep is adopted, managed, and counted —
    never dropped, never retro-vetoed (P1-6).
  - DB trade rows are created at FILL time ONLY (P1-4); stage lifecycle
    lives in the event log + state file.

Kill switches: config `ignition_live.prestage.enabled` (default false),
env `IGNITION_PRESTAGE=0`, and `shadow: true` (zero orders).
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import logging

from trading import ignition_rules as _rules
from trading.exit_reasons import ExitReason

logger = logging.getLogger(__name__)

STRATEGY_NAME = 'ignition'
PRESTAGE_ID_PREFIX = 'ign-stage-'
# Alpaca caps client_order_id at 128 chars; we self-cap far lower so the
# scheme survives any broker tightening (§F18). Charset: the scheme only
# emits [A-Za-z0-9.-] (symbols can carry '.' class suffixes).
MAX_CLIENT_ORDER_ID_LEN = 48
_CLIENT_ORDER_ID_RE = re.compile(r'^[A-Za-z0-9.\-]+$')

_ET = ZoneInfo('America/New_York')

# ---------------------------------------------------------------------------
# Stage state machine (P0-3): one record per (symbol, day)
# ---------------------------------------------------------------------------
STATE_STAGED = 'STAGED'
STATE_CANCEL_PENDING = 'CANCEL_PENDING'
STATE_CANCEL_CONFIRMED = 'CANCEL_CONFIRMED'
STATE_FILLED = 'FILLED'
STATE_REJECTED = 'REJECTED'

# Legal transitions. Absence of a record == never staged == chase-eligible.
_VALID_TRANSITIONS = {
    STATE_STAGED: {STATE_FILLED, STATE_CANCEL_PENDING, STATE_REJECTED,
                   STATE_CANCEL_CONFIRMED},
    STATE_CANCEL_PENDING: {STATE_CANCEL_CONFIRMED, STATE_FILLED},
    STATE_CANCEL_CONFIRMED: set(),
    STATE_FILLED: set(),
    STATE_REJECTED: set(),
}

# States from which the chase path may proceed (terminal-and-unfilled).
_CHASE_ELIGIBLE_STATES = {STATE_CANCEL_CONFIRMED, STATE_REJECTED}


def round_price_for_tick(price: float) -> float:
    """SEC tick-rule rounding (§F17): penny at >= $1, 4 decimals < $1.

    The universe is low-priced — an unrounded limit REJECTS at the
    broker day one. Central helper used for BOTH stop and limit prices.
    """
    if price >= 1.0:
        return round(price, 2)
    return round(price, 4)


def stage_client_order_id(day_yyyymmdd: str, symbol: str) -> str:
    """`ign-stage-{YYYYMMDD}-{sym}` — idempotency + attribution (§A5/F18).

    Raises ValueError on length/charset violations so a bad id fails in
    tests, never as a broker reject at 9:35.
    """
    coid = f"{PRESTAGE_ID_PREFIX}{day_yyyymmdd}-{symbol}"
    if len(coid) > MAX_CLIENT_ORDER_ID_LEN:
        raise ValueError(
            f"client_order_id too long ({len(coid)} > "
            f"{MAX_CLIENT_ORDER_ID_LEN}): {coid}")
    if not _CLIENT_ORDER_ID_RE.match(coid):
        raise ValueError(f"client_order_id has invalid chars: {coid}")
    return coid


def check_set_parity(shadow_triggers: Set[str], staged_fills: Set[str],
                     chase_entries: Set[str],
                     explicit_reasons: Dict[str, str]) -> dict:
    """The nightly set-equality HARD gate (§B8, P0-1).

    Every shadow trigger must be covered by a staged fill, a chase
    entry, or an explicit reason; every staged fill must correspond to a
    shadow trigger or carry an explicit reason (fill-without-trigger,
    structure reject, ...). Returns {'ok', 'missing', 'unexplained_fills'}.
    """
    covered = staged_fills | chase_entries | set(explicit_reasons)
    missing = sorted(shadow_triggers - covered)
    unexplained = sorted(staged_fills - shadow_triggers
                         - set(explicit_reasons))
    return {'ok': not missing and not unexplained,
            'missing': missing, 'unexplained_fills': unexplained}


@dataclass
class PrestageCandidate:
    """A stageable universe name tracked by the proximity scheduler."""

    symbol: str
    day_open: float
    level: float
    last: float
    has_news: Optional[bool] = None
    anchor: Optional[str] = None
    stop_est: float = 0.0          # staging-time stop ESTIMATE for qty
    sibling_eligible: bool = False  # P0-2 reactive sibling staging
    promote_streak: int = 0
    last_update_ts: float = 0.0
    rank: int = 10 ** 6

    @property
    def distance_pct(self) -> float:
        """(level - last) / level in percent; negative = crossed."""
        if self.level <= 0:
            return float('inf')
        return (self.level - self.last) / self.level * 100.0

    @property
    def news_eligible(self) -> bool:
        return self.has_news is True


@dataclass
class PrestageTelemetry:
    """§G counters, surfaced by telemetry_snapshot() for the EOD dive."""

    stage_ops: int = 0
    cancel_ops: int = 0
    fills_staged: int = 0
    fills_shadow_inferred: int = 0
    would_stage: int = 0
    would_cancel: int = 0
    would_fill: int = 0
    bp_high_watermark_usd: float = 0.0
    bp_budget_usd: float = 0.0
    gap_through_count: int = 0
    scratch_count: int = 0
    scratch_cost_usd: float = 0.0
    churn_limiter_activations: int = 0
    fallback_activations: int = 0
    feed_stale_events: int = 0
    fills_without_trigger: int = 0
    rank_at_trigger: List[int] = field(default_factory=list)


class PrestageManager:
    """Pre-staged stop-limit entries for the ignition strategy.

    Thread-safety: intake callbacks (on_candidate/on_price/
    notify_trigger) arrive from the shadow worker + engine worker;
    process_tick and sweeps arrive from the scanner thread. One RLock
    covers all check-then-act sequences (ignition engine pattern).
    """

    def __init__(self, alpaca_client, db=None, stop_monitor=None,
                 order_stream=None, notifier=None,
                 cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get('enabled', False))
        env = os.environ.get('IGNITION_PRESTAGE')
        if env is not None:
            self.enabled = env.strip().lower() not in (
                '0', 'false', 'no', 'off', '')
        self.shadow = bool(cfg.get('shadow', True))
        self.risk_usd = float(cfg.get('risk_usd', 50.0))
        self.cap_bps = float(cfg.get('cap_bps', 300.0))
        self.stop_offset_bps = float(cfg.get('stop_offset_bps', 15.0))
        self.heap_k = int(cfg.get('heap_k', 400))
        self.promote_rank_slack = int(cfg.get('promote_rank_slack', 2))
        self.demote_rank_slack = int(cfg.get('demote_rank_slack', 3))
        self.promote_distance_pct = float(
            cfg.get('promote_distance_pct', 20.0))
        self.demote_distance_pct = float(
            cfg.get('demote_distance_pct', 25.0))
        self.promote_consecutive = int(cfg.get('promote_consecutive', 2))
        self.ops_per_min = int(cfg.get('ops_per_min', 100))
        self.bp_frac = float(cfg.get('bp_frac', 0.25))
        self.bp_abs_usd = float(cfg.get('bp_abs_usd', 30000.0))
        self.stage_start_min = int(cfg.get('stage_start_min', 575))
        self.cancel_all_min = int(cfg.get('cancel_all_min', 780))
        self.gap_through_cancel_min = int(
            cfg.get('gap_through_cancel_min', 60))
        self.watchdog_stale_s = float(cfg.get('watchdog_stale_s', 60.0))
        self.max_staged_fills = int(cfg.get('max_staged_fills', 10))
        self.pdt_equity_min = float(cfg.get('pdt_equity_min', 25000.0))

        self.alpaca = alpaca_client
        self.db = db
        self.stop_monitor = stop_monitor
        self.order_stream = order_stream
        self.notifier = notifier

        self._lock = threading.RLock()
        self._day: Optional[str] = None
        self._candidates: Dict[str, PrestageCandidate] = {}
        # per-(symbol, day) stage records: state machine + order ids
        self._stages: Dict[str, dict] = {}
        self._op_times: deque = deque()   # rate budget window (60s)
        self._last_feed_ts: float = 0.0
        self._chase_only_mode = False
        self._kill_active = False
        self._midday_swept = False
        self._dtbp: Optional[float] = None
        self._equity: Optional[float] = None
        self._pdt_halved = False
        self.telemetry = PrestageTelemetry()
        # parity ledger (§B8): sets keyed by symbol for today
        self._parity_triggers: Set[str] = set()
        self._parity_staged_fills: Set[str] = set()
        self._parity_chase_entries: Set[str] = set()
        self._parity_explicit: Dict[str, str] = {}
        self._log_dir = cfg.get('log_dir') or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'logs')
        if self.enabled:
            logger.info(
                f"PrestageManager ACTIVE: shadow={self.shadow}, "
                f"risk=${self.risk_usd:.0f}, cap={self.cap_bps:.0f}bps, "
                f"stop_offset={self.stop_offset_bps:.0f}bps, "
                f"K={self.heap_k}, ops/min={self.ops_per_min}, "
                f"bp_budget=min({self.bp_frac:.0%}xDTBP, "
                f"${self.bp_abs_usd:,.0f}), window "
                f"{self.stage_start_min}->{self.cancel_all_min} ET min")

    # ------------------------------------------------------------------
    # time helpers — ET minutes everywhere, never UTC math (§F20)
    # ------------------------------------------------------------------
    @staticmethod
    def _now_et() -> datetime:
        return datetime.now(timezone.utc).astimezone(_ET)

    @staticmethod
    def _et_minute(dt_et: datetime) -> int:
        return dt_et.hour * 60 + dt_et.minute

    def _roll_day(self, day: str) -> None:
        if day == self._day:
            return
        self._day = day
        self._candidates = {}
        self._stages = {}
        self._op_times.clear()
        self._chase_only_mode = False
        self._kill_active = False
        self._midday_swept = False
        self._dtbp = None
        self._equity = None
        self._pdt_halved = False
        self.telemetry = PrestageTelemetry()
        self._parity_triggers = set()
        self._parity_staged_fills = set()
        self._parity_chase_entries = set()
        self._parity_explicit = {}

    # ------------------------------------------------------------------
    # §G lifecycle event log + state file (kill −9 replay source)
    # ------------------------------------------------------------------
    def _event(self, symbol: str, event: str, **kw) -> None:
        tag = '[PRESTAGE SHADOW]' if self.shadow else '[PRESTAGE]'
        detail = ' '.join(f"{k}={v}" for k, v in kw.items())
        logger.info(f"{tag} {symbol} {event} {detail}".rstrip())
        try:
            path = os.path.join(self._log_dir,
                                f"prestage_events_{self._day}.jsonl")
            with open(path, 'a') as fh:
                fh.write(json.dumps(
                    {'ts_utc': datetime.now(timezone.utc).isoformat(),
                     'symbol': symbol, 'event': event,
                     'shadow': self.shadow, **kw}, default=str) + '\n')
        except Exception as e:
            logger.warning(f"[PRESTAGE] event journal write failed: {e}")

    def _state_file(self, day: Optional[str] = None) -> str:
        return os.path.join(self._log_dir,
                            f"prestage_state_{day or self._day}.json")

    def _persist_state(self) -> None:
        """Write the stage records after every transition — the boot
        reconciliation (kill −9 drill) replays from this file."""
        try:
            with open(self._state_file(), 'w') as fh:
                json.dump({'day': self._day, 'stages': self._stages},
                          fh, default=str)
        except Exception as e:
            logger.warning(f"[PRESTAGE] state persist failed: {e} — "
                           f"boot replay will fall back to broker scan")

    # ------------------------------------------------------------------
    # state machine (P0-3)
    # ------------------------------------------------------------------
    def _transition(self, symbol: str, new_state: str,
                    reason: str = '') -> bool:
        """Move a (symbol, day) stage record to `new_state`.

        Returns False (with ERROR log) on an illegal transition —
        callers must treat that as 'do not proceed'.
        """
        rec = self._stages.get(symbol)
        if rec is None:
            logger.error(f"[PRESTAGE] {symbol} transition to {new_state} "
                         f"with NO stage record — bug")
            return False
        cur = rec['state']
        if new_state not in _VALID_TRANSITIONS.get(cur, set()):
            logger.error(
                f"[PRESTAGE] {symbol} ILLEGAL transition {cur} -> "
                f"{new_state} ({reason}) — ignored")
            return False
        rec['state'] = new_state
        rec['state_reason'] = reason
        rec['state_ts'] = time.time()
        self._event(symbol, f"state:{cur}->{new_state}", reason=reason)
        self._persist_state()
        return True

    def chase_allowed(self, symbol: str) -> bool:
        """P0-3 fast local check: may the chase path enter this symbol?

        True when no stage was ever placed today or the stage reached a
        terminal-and-unfilled state. False while a stage is live,
        cancel-pending, or filled. Shadow mode NEVER gates real trading.
        """
        if not self.enabled or self.shadow:
            return True
        with self._lock:
            rec = self._stages.get(symbol)
            if rec is None:
                return True
            return rec['state'] in _CHASE_ELIGIBLE_STATES

    def resolve_for_chase(self, symbol: str) -> str:
        """Broker-proof stage disposition before a chase entry (P0-3).

        Returns:
            'chase_ok' — no stage today, or terminal-and-unfilled proven.
            'adopted'  — the stage turned out FILLED (position adopted
                         here); the chase MUST NOT enter.
            'blocked'  — disposition could not be proven (broker error);
                         the chase MUST NOT enter (a double position is
                         strictly worse than a missed chase).
        """
        if not self.enabled or self.shadow:
            return 'chase_ok'
        with self._lock:
            rec = self._stages.get(symbol)
            if rec is None:
                return 'chase_ok'
            state = rec['state']
            if state in _CHASE_ELIGIBLE_STATES:
                return 'chase_ok'
            if state == STATE_FILLED:
                return 'adopted'
        # STAGED / CANCEL_PENDING: prove it at the broker.
        return self._cancel_stage(symbol, reason='chase_fallback')

    # ------------------------------------------------------------------
    # intake (shadow worker thread)
    # ------------------------------------------------------------------
    def on_candidate(self, rec: dict) -> None:
        """Candidate intake from the shadow's sighting flow.

        Called (guarded, never raises into the shadow) once the shadow
        has computed day_open for a sighting. Universe gates are the
        shadow's; we only track names with a computable level (§obs:
        no day_open = not stageable, chase covers — logged)."""
        if not self.enabled:
            return
        try:
            symbol = rec.get('symbol')
            day = rec.get('day')
            day_open = rec.get('day_open')
            if not symbol or not day:
                return
            with self._lock:
                self._roll_day(day)
                if day_open is None or float(day_open) <= 0:
                    self._event(symbol, 'stage_skip_no_day_open')
                    self._parity_explicit.setdefault(
                        symbol, 'stage_skip_no_day_open')
                    return
                day_open = float(day_open)
                lvl = _rules.level(day_open)
                c = self._candidates.get(symbol)
                if c is None:
                    c = PrestageCandidate(
                        symbol=symbol, day_open=day_open, level=lvl,
                        last=float(rec.get('price') or 0.0),
                        has_news=rec.get('has_news'),
                        anchor=rec.get('anchor'),
                        stop_est=float(rec.get('_stop') or 0.0))
                    self._candidates[symbol] = c
                    self._event(symbol, 'candidate',
                                level=round(lvl, 4),
                                news=c.has_news, anchor=c.anchor)
                else:
                    c.day_open, c.level = day_open, lvl
                    if rec.get('has_news') is not None:
                        c.has_news = rec.get('has_news')
                    if rec.get('anchor'):
                        c.anchor = rec.get('anchor')
                c.last = float(rec.get('price') or c.last)
                c.last_update_ts = time.time()
                self._last_feed_ts = time.time()
        except Exception as e:
            logger.error(f"[PRESTAGE] on_candidate failed: {e}")

    def on_price(self, symbol: str, price: float,
                 minute_et: Optional[int] = None) -> None:
        """Per-cycle price update from the scanner sighting flow.

        Feeds the proximity ranks, the feed watchdog (§D12), and — in
        shadow mode — the would-fill inference (tape crossing the stop
        while 'staged'). Never raises into the caller."""
        if not self.enabled:
            return
        try:
            with self._lock:
                self._last_feed_ts = time.time()
                if self._chase_only_mode:
                    # feed recovered — resume staging (§D12 recovery)
                    self._chase_only_mode = False
                    logger.warning(
                        "[PRESTAGE] feed recovered — leaving chase-only "
                        "mode, staging resumes")
                c = self._candidates.get(symbol)
                if c is None:
                    return
                c.last = float(price)
                c.last_update_ts = time.time()
                rec = self._stages.get(symbol)
                if rec is not None and rec['state'] == STATE_STAGED:
                    stop_px = rec['stop_px']
                    cap_px = rec['limit_px']
                    if price > cap_px and not rec.get('gap_through_ts'):
                        # gap-through-band observed (§B6): the order can
                        # only fill LATER on a fade (adverse class)
                        rec['gap_through_ts'] = time.time()
                        self.telemetry.gap_through_count += 1
                        self._event(symbol, 'gap_through',
                                    last=price, cap_px=cap_px)
                    if self.shadow and price >= stop_px:
                        self._shadow_infer_fill(symbol, rec, price)
        except Exception as e:
            logger.error(f"[PRESTAGE] on_price failed: {e}")

    def notify_trigger(self, rec: dict) -> None:
        """A shadow trigger fired (staged fill or chase-eligible).

        Records the parity-ledger trigger + rank-at-trigger telemetry,
        and performs the P0-2 REACTIVE sibling staging: the first
        trigger of an anchor makes every same-anchor sibling still below
        level stage-eligible regardless of news."""
        if not self.enabled:
            return
        try:
            symbol = rec.get('symbol')
            anchor = rec.get('anchor')
            with self._lock:
                if rec.get('day'):
                    self._roll_day(rec['day'])
                if symbol:
                    self._parity_triggers.add(symbol)
                    c = self._candidates.get(symbol)
                    if c is not None:
                        self.telemetry.rank_at_trigger.append(c.rank)
                if anchor:
                    self._activate_siblings(anchor, first=symbol or '')
        except Exception as e:
            logger.error(f"[PRESTAGE] notify_trigger failed: {e}")

    def notify_chase_entry(self, symbol: str) -> None:
        """A chase-path entry was submitted — parity ledger + telemetry."""
        if not self.enabled:
            return
        with self._lock:
            self._parity_chase_entries.add(symbol)
            self.telemetry.fallback_activations += 1

    def _activate_siblings(self, anchor: str, first: str) -> None:
        """P0-2: mark all same-anchor candidates sibling-eligible so the
        scheduler stages them (still below level) on the next tick."""
        n = 0
        for c in self._candidates.values():
            if c.anchor == anchor and c.symbol != first \
                    and not c.sibling_eligible:
                c.sibling_eligible = True
                n += 1
        if n:
            self._event(anchor, 'sibling_activation',
                        first_trigger=first, siblings=n)

    # ------------------------------------------------------------------
    # kill rails (§C10) — same call path as the kill itself
    # ------------------------------------------------------------------
    def notify_kill(self, kind: str) -> None:
        """Kill rail fired: immediate stage-sweep + no staging until the
        day rolls (weekly kill re-fires on each _kill_blocked call)."""
        if not self.enabled:
            return
        with self._lock:
            already = self._kill_active
            self._kill_active = True
        if not already:
            logger.warning(f"[PRESTAGE] KILL ({kind}) — sweeping all "
                           f"staged orders, no staging until day roll")
        self._sweep_all(reason=f'kill_{kind}')

    # ------------------------------------------------------------------
    # scheduler tick (scanner thread)
    # ------------------------------------------------------------------
    def process_tick(self, now_et: Optional[datetime] = None) -> None:
        """One scheduler cycle: watchdog, window sweeps, fill
        consumption, hysteresis promote/demote, staging under budgets.
        Bounded work, never raises."""
        if not self.enabled:
            return
        try:
            now = now_et or self._now_et()
            day = now.strftime('%Y-%m-%d')
            minute = self._et_minute(now)
            with self._lock:
                self._roll_day(day)
            self._watchdog_check()
            self._consume_fills()
            if minute >= self.cancel_all_min:
                if not self._midday_swept:
                    self._midday_swept = True
                    self._sweep_all(reason='cancel_all_1300')
                return
            self._gap_through_cancels()
            if minute < self.stage_start_min:
                return   # P0-6/§1: never stage before 9:35 ET
            if self._kill_active or self._chase_only_mode:
                return
            self._rank_candidates()
            self._demote_pass()
            self._promote_pass(minute)
        except Exception as e:
            logger.error(f"[PRESTAGE] process_tick failed: {e}",
                         exc_info=True)

    def _watchdog_check(self) -> None:
        """§D12 feed watchdog: no heap update for watchdog_stale_s while
        stages are live => sweep + chase-only mode + alert."""
        with self._lock:
            if self._chase_only_mode or not self._candidates:
                return
            if self._last_feed_ts <= 0:
                return
            stale = time.time() - self._last_feed_ts
            if stale <= self.watchdog_stale_s:
                return
            any_staged = any(r['state'] == STATE_STAGED
                             for r in self._stages.values())
            self._chase_only_mode = True
            self.telemetry.feed_stale_events += 1
        logger.error(
            f"[PRESTAGE] FEED STALE {stale:.0f}s > "
            f"{self.watchdog_stale_s:.0f}s — sweeping stages, "
            f"chase-only mode until feed recovers")
        self._notify(f"[PRESTAGE] ⚠ feed stale {stale:.0f}s — staged "
                     f"orders swept, chase-only mode")
        if any_staged:
            self._sweep_all(reason='feed_watchdog')

    def _rank_candidates(self) -> None:
        """Proximity rank = distance-to-level ascending (K-study order:
        nearest-to-level staged first — front-loads monster coverage)."""
        with self._lock:
            below = sorted(self._candidates.values(),
                           key=lambda c: c.distance_pct)
            for i, c in enumerate(below):
                c.rank = i + 1

    def _effective_k(self) -> int:
        """Heap depth, halved when equity < PDT line (§C11)."""
        k = self.heap_k
        if self._equity is not None and self._equity < self.pdt_equity_min:
            if not self._pdt_halved:
                self._pdt_halved = True
                logger.warning(
                    f"[PRESTAGE] equity ${self._equity:,.0f} < PDT line "
                    f"${self.pdt_equity_min:,.0f} — staging depth halved "
                    f"to {k // 2}")
            k //= 2
        return k

    def _promote_pass(self, minute: int) -> None:
        """Stage-eligible candidates, nearest-to-level first, under the
        rate + BP budgets, with dual-threshold hysteresis (§E15)."""
        with self._lock:
            # account snapshot BEFORE the depth calc — the PDT halving
            # (§C11) must apply to this pass, not the next one
            self._refresh_account()
            k = self._effective_k()
            cands = sorted(self._candidates.values(),
                           key=lambda c: c.rank)
        for c in cands:
            with self._lock:
                if c.symbol in self._stages:
                    continue   # one stage per (symbol, day) EVER (§A5)
                if not (c.news_eligible or c.sibling_eligible):
                    continue   # P0-2: news leg at 9:35; siblings reactive
                if c.last >= c.level:
                    # P0-6: already crossed — buy-stop would reject;
                    # route to chase with an explicit parity reason
                    if self._parity_explicit.get(c.symbol) is None:
                        self._parity_explicit[c.symbol] = \
                            'stage_skip_already_crossed'
                        self._event(c.symbol,
                                    'stage_skip_already_crossed',
                                    last=c.last, level=round(c.level, 4))
                    continue
                # dual-threshold hysteresis: promote only inside D_in
                # AND rank inside K - slack, for N consecutive updates
                if (c.distance_pct < self.promote_distance_pct
                        and c.rank <= max(1, k - self.promote_rank_slack)):
                    c.promote_streak += 1
                else:
                    c.promote_streak = 0
                    continue
                if c.promote_streak < self.promote_consecutive:
                    continue
                if not self._op_allowed():
                    self.telemetry.churn_limiter_activations += 1
                    self._event(c.symbol, 'stage_deferred_rate_budget')
                    return   # deferred is safe — chase path covers
                if not self._bp_allows(c):
                    # budgeter engaged: chase covers; visible in parity
                    self._parity_explicit.setdefault(
                        c.symbol, 'stage_skip_bp_budget')
                    self._event(c.symbol, 'stage_skip_bp_budget')
                    continue
            self._stage(c, minute)

    def _demote_pass(self) -> None:
        """Cancel stages whose candidate faded out (rank > K + slack OR
        distance > D_out) — hysteresis prevents place/cancel storms."""
        with self._lock:
            k = self._effective_k()
            to_demote = []
            for sym, rec in self._stages.items():
                if rec['state'] != STATE_STAGED:
                    continue
                c = self._candidates.get(sym)
                if c is None:
                    continue
                if (c.rank > k + self.demote_rank_slack
                        or c.distance_pct > self.demote_distance_pct):
                    to_demote.append(sym)
        for sym in to_demote:
            if not self._op_allowed():
                self.telemetry.churn_limiter_activations += 1
                return
            self._cancel_stage(sym, reason='demoted')

    # ------------------------------------------------------------------
    # budgets
    # ------------------------------------------------------------------
    def _op_allowed(self) -> bool:
        """Hard churn limiter: max ops_per_min order ops (§E15)."""
        now = time.time()
        while self._op_times and now - self._op_times[0] > 60.0:
            self._op_times.popleft()
        if len(self._op_times) >= self.ops_per_min:
            return False
        self._op_times.append(now)
        return True

    def _refresh_account(self) -> None:
        """Fetch DTBP + equity once per day (fail => abs cap, WARNING)."""
        if self._dtbp is not None:
            return
        try:
            info = self.alpaca.get_account_info() or {}
            self._dtbp = float(info.get('buying_power') or 0.0)
            self._equity = float(info.get('equity') or 0.0)
        except Exception as e:
            logger.warning(f"[PRESTAGE] account fetch failed ({e}) — "
                           f"using abs BP cap ${self.bp_abs_usd:,.0f} "
                           f"only")
            self._dtbp = 0.0
            self._equity = None

    def _bp_budget(self) -> float:
        self._refresh_account()
        if self._dtbp and self._dtbp > 0:
            budget = min(self.bp_frac * self._dtbp, self.bp_abs_usd)
        else:
            budget = self.bp_abs_usd
        self.telemetry.bp_budget_usd = budget
        return budget

    def _bp_reserved(self) -> float:
        """Reserved (resting stop-limit notional) + filled notional."""
        total = 0.0
        for rec in self._stages.values():
            if rec['state'] == STATE_STAGED:
                total += rec['qty'] * rec['limit_px']
            elif rec['state'] == STATE_FILLED:
                total += (rec.get('filled_qty') or rec['qty']) \
                    * (rec.get('fill_price') or rec['limit_px'])
        return total

    def _bp_allows(self, c: PrestageCandidate) -> bool:
        """§C9 watermark budget with 80% alert + 100% hard stop."""
        budget = self._bp_budget()
        qty = self._stage_qty(c)
        if qty < 1:
            # zero-qty is not a budget problem — let _stage label it
            # with its own explicit reason (stage_skip_zero_qty)
            return True
        add = qty * self._cap_px(c)
        reserved = self._bp_reserved()
        hw = reserved + add
        if hw > self.telemetry.bp_high_watermark_usd and hw <= budget:
            self.telemetry.bp_high_watermark_usd = hw
        # 80% alert BEFORE the hard stop: a single stage can jump the
        # watermark from below-80% straight past 100% — the alert must
        # still fire on the attempt (§C9: alert at 80, hard stop at 100)
        if reserved + add > 0.8 * budget and not getattr(
                self, '_bp_alerted', False):
            self._bp_alerted = True
            self._notify(f"[PRESTAGE] ⚠ BP watermark "
                         f"${reserved + add:,.0f} > 80% of budget "
                         f"${budget:,.0f}")
        if reserved + add > budget:
            if self.telemetry.bp_high_watermark_usd < reserved:
                self.telemetry.bp_high_watermark_usd = reserved
            return False
        return True

    # ------------------------------------------------------------------
    # staging
    # ------------------------------------------------------------------
    def _stop_px(self, c: PrestageCandidate) -> float:
        return round_price_for_tick(
            c.level * (1 + self.stop_offset_bps / 1e4))

    def _cap_px(self, c: PrestageCandidate) -> float:
        return round_price_for_tick(c.level * (1 + self.cap_bps / 1e4))

    def _stage_qty(self, c: PrestageCandidate) -> int:
        """qty = floor($risk / (entry_est − stop_est)) at staging time.

        entry_est = stop trigger price. stop_est falls back to the R-min
        floor (entry × (1 − 5%)) when the sighting carried no stop
        estimate — actual stop/R are recomputed AT FILL (P0-5); the
        realized-risk-vs-$risk ratio is telemetried there."""
        entry_est = self._stop_px(c)
        stop_est = c.stop_est if 0 < c.stop_est < entry_est else \
            entry_est * (1 - _rules.R_MIN_PCT / 100.0)
        rps = entry_est - stop_est
        if rps <= 0:
            return 0
        return max(0, math.floor(self.risk_usd / rps))

    def _stage(self, c: PrestageCandidate, minute: int) -> None:
        """Place (or shadow-log) one staged stop-limit BUY."""
        stop_px = self._stop_px(c)
        limit_px = self._cap_px(c)
        qty = self._stage_qty(c)
        if qty < 1:
            self._event(c.symbol, 'stage_skip_zero_qty')
            self._parity_explicit.setdefault(c.symbol,
                                             'stage_skip_zero_qty')
            return
        day_compact = self._day.replace('-', '') if self._day else \
            self._now_et().strftime('%Y%m%d')
        try:
            coid = stage_client_order_id(day_compact, c.symbol)
        except ValueError as e:
            logger.error(f"[PRESTAGE] {c.symbol} bad client_order_id: "
                         f"{e} — not staged (chase covers)")
            self._parity_explicit.setdefault(c.symbol,
                                             'stage_skip_bad_coid')
            return
        with self._lock:
            self._stages[c.symbol] = {
                'state': STATE_STAGED, 'symbol': c.symbol,
                'client_order_id': coid, 'order_id': '',
                'stop_px': stop_px, 'limit_px': limit_px, 'qty': qty,
                'level': c.level, 'rank_at_stage': c.rank,
                'stop_est': c.stop_est, 'anchor': c.anchor,
                'has_news': c.has_news, 'staged_minute': minute,
                'stage_ts': time.time(), 'gap_through_ts': None,
                'filled_qty': 0, 'fill_price': None,
            }
        if self.shadow:
            self.telemetry.would_stage += 1
            self._event(c.symbol, 'would_stage', stop=stop_px,
                        limit=limit_px, qty=qty, rank=c.rank,
                        news=c.has_news, sibling=c.sibling_eligible)
            self._persist_state()
            return
        try:
            result = self.alpaca.submit_stop_limit_order(
                symbol=c.symbol, qty=qty, side='buy',
                stop_price=stop_px, limit_price=limit_px,
                client_order_id=coid, tick_rounding=True)
            with self._lock:
                self._stages[c.symbol]['order_id'] = \
                    (result or {}).get('id', '')
            self.telemetry.stage_ops += 1
            self._event(c.symbol, 'staged', stop=stop_px,
                        limit=limit_px, qty=qty, rank=c.rank,
                        order_id=(result or {}).get('id', ''))
            self._persist_state()
        except Exception as e:
            logger.error(f"[PRESTAGE] {c.symbol} stage submit FAILED: "
                         f"{e} — REJECTED (chase covers)")
            with self._lock:
                self._transition(c.symbol, STATE_REJECTED,
                                 reason=f'submit_error:{str(e)[:60]}')
                self._parity_explicit.setdefault(
                    c.symbol, 'stage_submit_rejected')

    # ------------------------------------------------------------------
    # cancels + sweeps (all route through the state machine — P0-3)
    # ------------------------------------------------------------------
    def _cancel_stage(self, symbol: str, reason: str) -> str:
        """Cancel one staged order. Cancel-reject != cancelled (§A3):
        on reject/False, poll the order; if filled, ADOPT.

        Returns the disposition: 'chase_ok' | 'adopted' | 'blocked'.
        """
        with self._lock:
            rec = self._stages.get(symbol)
            if rec is None:
                return 'chase_ok'
            if rec['state'] == STATE_FILLED:
                return 'adopted'
            if rec['state'] in _CHASE_ELIGIBLE_STATES:
                return 'chase_ok'
            if rec['state'] == STATE_STAGED:
                self._transition(symbol, STATE_CANCEL_PENDING,
                                 reason=reason)
        if self.shadow:
            self.telemetry.would_cancel += 1
            self._event(symbol, 'would_cancel', reason=reason)
            with self._lock:
                self._transition(symbol, STATE_CANCEL_CONFIRMED,
                                 reason=reason)
            return 'chase_ok'
        rec = self._stages.get(symbol)
        order_id = rec.get('order_id') or rec.get('client_order_id')
        cancelled = False
        try:
            cancelled = bool(self.alpaca.cancel_order(order_id))
            self.telemetry.cancel_ops += 1
        except Exception as e:
            logger.warning(f"[PRESTAGE] {symbol} cancel errored: {e} — "
                           f"polling order for disposition")
        # Proof at the broker: poll the order regardless of the cancel's
        # apparent success — the fill may have raced the cancel.
        try:
            od = self.alpaca.get_order(order_id) or {}
        except Exception as e:
            logger.error(f"[PRESTAGE] {symbol} disposition poll FAILED "
                         f"({e}) — BLOCKED (no chase until proven)")
            return 'blocked'
        status = (od.get('status') or '').lower()
        filled_qty = int(float(od.get('filled_qty') or 0))
        if status == 'filled' or filled_qty > 0:
            logger.warning(f"[PRESTAGE] {symbol} cancel({reason}) raced "
                           f"a FILL — adopting")
            self._adopt_fill(symbol, od)
            return 'adopted'
        if status in ('canceled', 'cancelled', 'expired', 'rejected',
                      'done_for_day'):
            with self._lock:
                self._transition(symbol, STATE_CANCEL_CONFIRMED,
                                 reason=reason)
            self._event(symbol, 'cancel_confirmed', reason=reason)
            return 'chase_ok'
        if cancelled:
            # cancel acked but status not yet terminal — treat as
            # pending until the next poll proves it
            logger.info(f"[PRESTAGE] {symbol} cancel acked, status="
                        f"{status} — awaiting terminal confirm")
        return 'blocked'

    def _sweep_all(self, reason: str) -> None:
        """Cancel every live stage, nearest-to-level FIRST (shrinks the
        fill-during-sweep window). Used by kill/watchdog/13:00/shutdown."""
        with self._lock:
            live = [s for s, r in self._stages.items()
                    if r['state'] in (STATE_STAGED, STATE_CANCEL_PENDING)]
            live.sort(key=lambda s: (
                self._candidates[s].distance_pct
                if s in self._candidates else float('inf')))
        if live:
            logger.info(f"[PRESTAGE] sweep({reason}): cancelling "
                        f"{len(live)} staged order(s)")
        for sym in live:
            self._cancel_stage(sym, reason=reason)

    def shutdown_sweep(self) -> None:
        """§A2: cancel-sweep all stages in the shutdown sequence (before
        finalize_eod). DAY TIF is belt-and-braces only."""
        if not self.enabled:
            return
        self._sweep_all(reason='shutdown')

    def _gap_through_cancels(self) -> None:
        """§B6/P1-5: gap-through names cancel after N=60min (FROZEN —
        tighter N needs its own harness pass; the study measured late
        limit fills net-POSITIVE at cap<=300, and the 60-min cancel
        inert (<$3K) — a hygiene rail, not a selection rule)."""
        with self._lock:
            expired = [
                s for s, r in self._stages.items()
                if r['state'] == STATE_STAGED and r.get('gap_through_ts')
                and time.time() - r['gap_through_ts']
                > self.gap_through_cancel_min * 60.0]
        for sym in expired:
            self._parity_explicit.setdefault(sym, 'gap_through_expired')
            self._cancel_stage(sym, reason='gap_through_expired')

    # ------------------------------------------------------------------
    # fills (P0-4: stream-first, batched REST as degraded slow path)
    # ------------------------------------------------------------------
    def _consume_fills(self) -> None:
        if self.shadow:
            return   # shadow fills are inferred in on_price
        statuses: Dict[str, dict] = {}
        if self.order_stream is not None:
            try:
                statuses = self.order_stream.snapshot_by_client_prefix(
                    PRESTAGE_ID_PREFIX)
            except Exception as e:
                logger.warning(f"[PRESTAGE] order-stream snapshot "
                               f"failed: {e} — REST slow path")
        with self._lock:
            live = {r['client_order_id']: s for s, r in
                    self._stages.items()
                    if r['state'] in (STATE_STAGED, STATE_CANCEL_PENDING)}
        for coid, sym in live.items():
            od = statuses.get(coid)
            if od is None:
                continue
            status = (od.get('status') or '').lower()
            filled_qty = int(float(od.get('filled_qty') or 0))
            if status == 'filled' or (status == 'partially_filled'
                                      and filled_qty > 0) \
                    or (status in ('canceled', 'cancelled', 'expired')
                        and filled_qty > 0):
                self._adopt_fill(sym, od)
            elif status == 'rejected':
                with self._lock:
                    self._transition(sym, STATE_REJECTED,
                                     reason='broker_rejected')
                    self._parity_explicit.setdefault(
                        sym, 'stage_submit_rejected')

    def _adopt_fill(self, symbol: str, od: dict) -> None:
        """P0-1/P0-4/P0-5 adoption: dead-man SL first, then AT-FILL
        structure validation, then watch + DB row (created NOW — P1-4).

        Kill-race semantics (P1-6): adoption NEVER checks kills — a fill
        that raced the kill sweep is adopted, managed, and counted."""
        with self._lock:
            rec = self._stages.get(symbol)
            if rec is None:
                logger.error(f"[PRESTAGE] {symbol} fill with no stage "
                             f"record — orphan adoption (boot path?)")
                rec = {'state': STATE_STAGED, 'symbol': symbol,
                       'client_order_id': '', 'order_id': '',
                       'stop_px': 0.0, 'limit_px': 0.0, 'qty': 0,
                       'level': 0.0, 'rank_at_stage': -1,
                       'stop_est': 0.0, 'anchor': None,
                       'has_news': None, 'staged_minute': 0,
                       'stage_ts': time.time(), 'gap_through_ts': None,
                       'filled_qty': 0, 'fill_price': None}
                self._stages[symbol] = rec
            if rec['state'] == STATE_FILLED:
                return   # idempotent — double adoption is a no-op
            fill_price = float(od.get('filled_avg_price')
                               or od.get('fill_price')
                               or rec['stop_px'] or 0.0)
            filled_qty = int(float(od.get('filled_qty') or rec['qty']))
            rec['fill_price'] = fill_price
            rec['filled_qty'] = filled_qty
            self._transition(symbol, STATE_FILLED, reason='fill')
        # partial fill (§A4): cancel the resting remainder immediately
        status = (od.get('status') or '').lower()
        if status not in ('filled', 'canceled', 'cancelled', 'expired') \
                and filled_qty < rec['qty']:
            try:
                self.alpaca.cancel_order(rec.get('order_id')
                                         or rec['client_order_id'])
                self._event(symbol, 'partial_fill_remainder_cancelled',
                            filled=filled_qty, staged=rec['qty'])
            except Exception as e:
                logger.error(f"[PRESTAGE] {symbol} partial-fill "
                             f"remainder cancel failed: {e}")
        # dead-man safety SL BEFORE anything slow (P0-4 naked window)
        prelim_stop = rec['stop_est'] if 0 < rec['stop_est'] < fill_price \
            else fill_price * 0.99
        sl_leg_id = ''
        try:
            res = self.alpaca.submit_stop_sell_order(
                symbol, filled_qty,
                round_price_for_tick(prelim_stop * 0.90))
            sl_leg_id = (res or {}).get('id', '')
        except Exception as e:
            logger.error(f"[PRESTAGE] {symbol} dead-man SL submit "
                         f"FAILED: {e} — position naked until watch")
        # AT-FILL structure validation via the shared helper (P0-1)
        gates = self._run_structure_gates(symbol, fill_price)
        fill_class = self._classify_fill(rec, fill_price)
        self.telemetry.fills_staged += 1
        if symbol not in self._parity_triggers:
            self.telemetry.fills_without_trigger += 1
            self._event(symbol, 'fill_without_trigger_flag',
                        fill=fill_price)
        if gates.get('ok'):
            stop = float(gates['stop'])
            r_pct = float(gates['r_pct'])
            self._register_watch_and_db(
                rec, fill_price, filled_qty, stop, r_pct, sl_leg_id,
                fill_class, parity_reason='staged_book', od=od)
            self._parity_staged_fills.add(symbol)
            self._event(symbol, 'adopted', fill=fill_price,
                        qty=filled_qty, stop=round(stop, 4),
                        r_pct=round(r_pct, 2), fill_class=fill_class)
        else:
            reason = gates.get('reject', 'no_bars')
            stop = prelim_stop
            self._register_watch_and_db(
                rec, fill_price, filled_qty, stop, 0.0, sl_leg_id,
                fill_class,
                parity_reason=ExitReason.STAGE_REJECT_STRUCTURE.value,
                structure_reject=reason, od=od)
            self._parity_explicit[symbol] = \
                ExitReason.STAGE_REJECT_STRUCTURE.value
            self.telemetry.scratch_count += 1
            self._event(symbol, 'stage_reject_structure',
                        reject=reason, fill=fill_price)
            if self.stop_monitor is not None:
                try:
                    self.stop_monitor.force_exit(
                        symbol,
                        reason=ExitReason.STAGE_REJECT_STRUCTURE.value)
                except Exception as e:
                    logger.error(f"[PRESTAGE] {symbol} disposition "
                                 f"force_exit FAILED: {e} — dead-man SL "
                                 f"is the backstop")
        # P1-6: max staged-fill positions — on breach, sweep the rest
        with self._lock:
            n_filled = sum(1 for r in self._stages.values()
                           if r['state'] == STATE_FILLED)
        if n_filled >= self.max_staged_fills:
            logger.warning(f"[PRESTAGE] staged-fill count {n_filled} >= "
                           f"max {self.max_staged_fills} — sweeping "
                           f"remaining stages")
            self._sweep_all(reason='max_staged_fills')

    def _run_structure_gates(self, symbol: str,
                             fill_price: float) -> dict:
        """Fetch bars up to now and run the shared at-fill gates."""
        try:
            now = self._now_et()
            minute_now = self._et_minute(now)
            lookback = max(minute_now - 570 + 5, 40)
            bars = self.alpaca.get_1min_bars(
                symbol, lookback_minutes=int(lookback))
            if bars is None or len(bars) < 2:
                return {'reject': 'no_bars'}
            import pandas as pd
            ts = pd.to_datetime(bars['timestamp'], utc=True)
            et = ts.dt.tz_convert('America/New_York')
            g = bars.copy()
            g['m'] = et.dt.hour * 60 + et.dt.minute
            g = g[(g['m'] >= 570) & (g['m'] < 960)] \
                .sort_values('m').reset_index(drop=True)
            return _rules.structure_gates_at_fill(g, fill_price)
        except Exception as e:
            logger.error(f"[PRESTAGE] {symbol} at-fill bars/gates "
                         f"failed: {e} — treating as structure reject")
            return {'reject': 'no_bars'}

    def _classify_fill(self, rec: dict, fill_price: float) -> str:
        """Resting-model fill classes: clean / gap_into / adverse."""
        stop_px = rec.get('stop_px') or 0.0
        cap_px = rec.get('limit_px') or 0.0
        if stop_px and fill_price <= stop_px * 1.0001:
            return 'clean'
        if rec.get('gap_through_ts'):
            return 'adverse'
        if cap_px and fill_price <= cap_px:
            return 'gap_into'
        return 'unknown'

    def _register_watch_and_db(self, rec: dict, fill_price: float,
                               filled_qty: int, stop: float,
                               r_pct: float, sl_leg_id: str,
                               fill_class: str, parity_reason: str,
                               structure_reject: str = '',
                               od: Optional[dict] = None) -> None:
        """StopMonitor watch (fill-keyed — P0-5) + DB row (P1-4: created
        at FILL only) + the path=staged FILL QUALITY line (P1-7)."""
        symbol = rec['symbol']
        risk_ps = max(fill_price - stop, 0.01)
        # skip_exits_until_ts keyed to the FILL minute (broker time when
        # available — P0-5; wall clock only as a logged fallback)
        fill_ts = None
        if od is not None and od.get('filled_at'):
            try:
                fa = od['filled_at']
                if isinstance(fa, str):
                    fa = datetime.fromisoformat(fa.replace('Z', '+00:00'))
                fill_ts = fa.timestamp()
            except Exception as e:
                logger.warning(f"[PRESTAGE] {symbol} filled_at parse "
                               f"failed ({e}) — wall-clock skip window")
        if fill_ts is None:
            fill_ts = time.time()
        skip_until = (int(fill_ts // 60) + 1) * 60
        if self.stop_monitor is not None:
            try:
                ok = self.stop_monitor.add_watch(
                    symbol=symbol, stop_price=stop, shares=filled_qty,
                    tp_leg_id='', sl_leg_id=sl_leg_id,
                    trade_db_id=None, entry_price=fill_price,
                    risk_per_share=risk_ps, strategy=STRATEGY_NAME,
                    lock_arm_at_r=_rules.ARM_R,
                    lock_stop_r=_rules.LOCK_R,
                    lock_r_unit=risk_ps,
                    skip_exits_until_ts=float(skip_until))
                if ok is False:
                    logger.error(f"[PRESTAGE] {symbol} add_watch "
                                 f"REJECTED (cross-strategy collision) "
                                 f"— dead-man SL is the only stop")
                    self._notify(f"[PRESTAGE] ⚠ {symbol} watch collision"
                                 f" — dead-man SL only, manual check")
            except Exception as e:
                logger.error(f"[PRESTAGE] {symbol} add_watch failed: "
                             f"{e} — dead-man SL is the only stop")
        trade_id = None
        if self.db is not None:
            record = {
                'trade_date': self._day
                or self._now_et().strftime('%Y-%m-%d'),
                'symbol': symbol, 'side': 'buy',
                'strategy': STRATEGY_NAME,
                'shares': filled_qty, 'entry_price': fill_price,
                'stop_loss_price': round_price_for_tick(stop * 0.90),
                'take_profit_price': round(fill_price * 3.0, 2),
                'risk_per_share': risk_ps,
                'total_risk': filled_qty * risk_ps,
                'risk_reward_ratio': 0.0,
                'order_id': rec.get('order_id') or '',
                'order_status': 'filled',
                'fill_price': fill_price,
                'filled_at': datetime.now(timezone.utc).isoformat(),
                'exit_price': None, 'exit_reason': None,
                'exited_at': None, 'pnl': None, 'pnl_pct': None,
                'pattern_data': json.dumps({
                    'path': 'staged',
                    'level': rec.get('level'),
                    'rank_at_stage': rec.get('rank_at_stage'),
                    'fill_class': fill_class,
                    'parity_reason': parity_reason,
                    'structure_reject': structure_reject or None,
                    'stop_px': rec.get('stop_px'),
                    'limit_px': rec.get('limit_px'),
                    'r_pct': round(r_pct, 2) if r_pct else None,
                    'anchor': rec.get('anchor'),
                    'has_news': rec.get('has_news'),
                    'lock_arm_at_r': _rules.ARM_R,
                    'lock_stop_r': _rules.LOCK_R,
                }),
            }
            try:
                trade_id = self.db.save_trade(record)
                self.db.update_trade(trade_id, {
                    'real_stop_loss_price': stop,
                    'filled_qty': filled_qty,
                })
                rec['trade_db_id'] = trade_id
                # late-bind the watch's DB id (watch was added first to
                # minimize the naked window)
                if self.stop_monitor is not None:
                    try:
                        w = self.stop_monitor._watches.get(symbol)
                        if w is not None and w.strategy == STRATEGY_NAME:
                            w.trade_db_id = trade_id
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"[PRESTAGE] {symbol} DB save failed: {e}")
        # FILL QUALITY (P1-7): staged path metric = fill vs LEVEL
        level = rec.get('level') or 0.0
        vs_level_bps = ((fill_price - level) / level * 1e4) \
            if level > 0 else float('nan')
        realized_risk = risk_ps * filled_qty
        logger.info(
            f"[IGNITION] FILL QUALITY {symbol}: path=staged "
            f"fill_vs_level={vs_level_bps:+.0f}bps "
            f"fill_class={fill_class} qty={filled_qty} "
            f"risk_realized=${realized_risk:.0f} "
            f"(planned ${self.risk_usd:.0f})")
        self._notify(f"[PRESTAGE] STAGED FILL {symbol} {filled_qty}sh @ "
                     f"{fill_price:.3f} ({fill_class}, "
                     f"vs_level {vs_level_bps:+.0f}bps) "
                     f"disposition={parity_reason}")

    def _shadow_infer_fill(self, symbol: str, rec: dict,
                           price: float) -> None:
        """Shadow-mode would-fill: the tape crossed the stop while
        'staged'. Telemetry + parity only — ZERO orders, ZERO DB."""
        if not self._transition(symbol, STATE_FILLED,
                                reason='shadow_inferred'):
            return
        rec['fill_price'] = rec['stop_px']
        rec['filled_qty'] = rec['qty']
        self.telemetry.would_fill += 1
        self.telemetry.fills_shadow_inferred += 1
        self._parity_staged_fills.add(symbol)
        level = rec.get('level') or 0.0
        vs_level_bps = ((rec['stop_px'] - level) / level * 1e4) \
            if level > 0 else float('nan')
        self._event(symbol, 'would_fill', last=price,
                    stop=rec['stop_px'],
                    vs_level_bps=round(vs_level_bps, 1),
                    fill_class=self._classify_fill(rec, rec['stop_px']))

    # ------------------------------------------------------------------
    # boot reconciliation (§A1 — BMNR/DFNS class)
    # ------------------------------------------------------------------
    def boot_reconcile(self) -> None:
        """Adopt-or-cancel every `ign-stage-*` at the broker + replay the
        state file. Runs in sync_positions BEFORE anything trades. Live
        mode only (shadow never has broker state)."""
        if not self.enabled:
            return
        now = self._now_et()
        day = now.strftime('%Y-%m-%d')
        with self._lock:
            self._roll_day(day)
        self._load_state(day)
        if self.shadow:
            return
        # 1) broker scan: open orders by prefix
        try:
            open_orders = self.alpaca.get_open_orders() or []
        except Exception as e:
            logger.error(f"[PRESTAGE] boot: open-orders scan FAILED "
                         f"({e}) — stale stages may be live; state-file "
                         f"replay continues")
            open_orders = []
        day_compact = day.replace('-', '')
        for od in open_orders:
            coid = od.get('client_order_id') or ''
            if not coid.startswith(PRESTAGE_ID_PREFIX):
                continue
            sym = od.get('symbol') or ''
            stale = day_compact not in coid
            with self._lock:
                if sym not in self._stages:
                    # order exists at broker, no local record (state
                    # file lost) — rebuild a minimal STAGED record
                    self._stages[sym] = {
                        'state': STATE_STAGED, 'symbol': sym,
                        'client_order_id': coid,
                        'order_id': od.get('id', ''),
                        'stop_px': float(od.get('stop_price') or 0.0),
                        'limit_px': float(od.get('limit_price') or 0.0),
                        'qty': int(float(od.get('qty') or 0)),
                        'level': 0.0, 'rank_at_stage': -1,
                        'stop_est': 0.0, 'anchor': None,
                        'has_news': None, 'staged_minute': 0,
                        'stage_ts': time.time(), 'gap_through_ts': None,
                        'filled_qty': 0, 'fill_price': None}
            self._event(sym, 'boot_stale_stage' if stale
                        else 'boot_open_stage', coid=coid)
            # boot policy: cancel EVERY resting stage (today's get
            # re-staged by the scheduler if still eligible); adopt on
            # cancel-reject via the shared path
            self._cancel_stage(sym, reason='boot_reconcile')
        # 2) state-file replay: records the broker no longer lists open
        with self._lock:
            pending = [s for s, r in self._stages.items()
                       if r['state'] in (STATE_STAGED,
                                         STATE_CANCEL_PENDING)]
        for sym in pending:
            rec = self._stages[sym]
            oid = rec.get('order_id') or rec.get('client_order_id')
            if not oid:
                with self._lock:
                    self._transition(sym, STATE_CANCEL_CONFIRMED,
                                     reason='boot_no_order_id')
                continue
            try:
                od = self.alpaca.get_order(oid) or {}
            except Exception as e:
                logger.error(f"[PRESTAGE] boot: {sym} order poll failed "
                             f"({e}) — left unresolved (chase blocked "
                             f"until proven)")
                continue
            status = (od.get('status') or '').lower()
            filled_qty = int(float(od.get('filled_qty') or 0))
            if status == 'filled' or filled_qty > 0:
                logger.warning(f"[PRESTAGE] boot: {sym} FILLED while "
                               f"down — adopting (orphan-fill path)")
                self._adopt_fill(sym, od)
            elif status in ('canceled', 'cancelled', 'expired',
                            'rejected', 'done_for_day'):
                with self._lock:
                    self._transition(sym, STATE_CANCEL_CONFIRMED,
                                     reason=f'boot_{status}')
            else:
                self._cancel_stage(sym, reason='boot_reconcile')
        logger.info(f"[PRESTAGE] boot_reconcile complete: "
                    f"{len(self._stages)} stage record(s) resolved")

    def _load_state(self, day: str) -> None:
        """Replay the persisted stage records for `day` (kill −9 drill).
        Missing/corrupt file logs a WARNING and falls back to the
        broker scan only."""
        path = self._state_file(day)
        if not os.path.exists(path):
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
            if data.get('day') != day:
                logger.warning(f"[PRESTAGE] state file day mismatch "
                               f"({data.get('day')} != {day}) — ignored")
                return
            with self._lock:
                for sym, rec in (data.get('stages') or {}).items():
                    if sym not in self._stages:
                        self._stages[sym] = rec
            logger.info(f"[PRESTAGE] state replay: "
                        f"{len(data.get('stages') or {})} record(s) "
                        f"loaded from {path}")
        except Exception as e:
            logger.warning(f"[PRESTAGE] state replay failed ({e}) — "
                           f"broker scan only")

    # ------------------------------------------------------------------
    # telemetry + parity surfaces
    # ------------------------------------------------------------------
    def telemetry_snapshot(self) -> dict:
        """§G counters for the EOD dive / green check."""
        t = self.telemetry
        with self._lock:
            staged_now = sum(1 for r in self._stages.values()
                             if r['state'] == STATE_STAGED)
            ranks = list(t.rank_at_trigger)
        return {
            'day': self._day, 'shadow': self.shadow,
            'stage_ops': t.stage_ops, 'cancel_ops': t.cancel_ops,
            'staged_open_now': staged_now,
            'fills_staged': t.fills_staged,
            'fills_shadow_inferred': t.fills_shadow_inferred,
            'would_stage': t.would_stage,
            'would_cancel': t.would_cancel,
            'would_fill': t.would_fill,
            'bp_high_watermark_usd': t.bp_high_watermark_usd,
            'bp_budget_usd': t.bp_budget_usd,
            'gap_through_count': t.gap_through_count,
            'scratch_count': t.scratch_count,
            'scratch_cost_usd': t.scratch_cost_usd,
            'churn_limiter_activations': t.churn_limiter_activations,
            'fallback_activations': t.fallback_activations,
            'feed_stale_events': t.feed_stale_events,
            'fills_without_trigger': t.fills_without_trigger,
            'rank_at_trigger': ranks,
        }

    def parity_ledger(self) -> dict:
        """Inputs for check_set_parity (nightly HARD gate)."""
        with self._lock:
            return {
                'shadow_triggers': set(self._parity_triggers),
                'staged_fills': set(self._parity_staged_fills),
                'chase_entries': set(self._parity_chase_entries),
                'explicit_reasons': dict(self._parity_explicit),
            }

    def check_parity(self) -> dict:
        led = self.parity_ledger()
        return check_set_parity(led['shadow_triggers'],
                                led['staged_fills'],
                                led['chase_entries'],
                                led['explicit_reasons'])

    # ------------------------------------------------------------------
    def _notify(self, msg: str) -> None:
        logger.info(msg)
        if self.notifier:
            try:
                self.notifier.send_message_sync(msg)
            except Exception as e:
                logger.warning(f"[PRESTAGE] telegram failed: {e}")
