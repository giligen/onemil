"""Ignition S3 live execution engine (2026-08-14, owner-approved plan).

A THIN EXECUTION CONSUMER of shadow triggers: the shadow
(trading/ignition_shadow.py) remains the single detection brain — every
rule lives in trading/ignition_rules.py — and this engine only executes
what the shadow journals as SHADOW_TRIGGER. Selection parity with the
shadow/BT is therefore by construction (the owner's explicit S3
requirement: any truncation or re-selection would make live results
incomparable to the validated book).

S3 spec (amended 2026-08-14): take ALL catalyst-confirmed triggers,
$50 risk/trade, kills -$300/day (new entries only) and -$750/week,
15:45 ET force-flat, judged on FILL QUALITY vs shadow-journaled quotes
— not on P&L.

Order flow mirrors ORB (the proven path):
  submit_bracket_order (marketable limit, broker-side safety legs)
  -> poll get_order -> StopMonitor.add_watch(static lock 1.75R/0.5R)
  -> drain_exit_events('ignition') -> build_exit_update + pnl -> DB.

Kill switches: config ignition_live.enabled, env IGNITION_LIVE=0,
dry_run mode (full pipeline, zero orders, '[IGNITION DRY]' telegrams).
"""
from __future__ import annotations

import json
import logging
import math
import os
import queue
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
from zoneinfo import ZoneInfo

from trading import ignition_rules as _rules
from trading.stop_monitor import build_exit_update

logger = logging.getLogger(__name__)

STRATEGY_NAME = 'ignition'
_ET = ZoneInfo('America/New_York')


def _et_now() -> datetime:
    """Market-clock now. Kill windows / trade_date / day rolls key on the
    ET date (8/14 audit P3): the UTC date rolls at 20:00 ET, so a UTC-
    dated 'today' silently splits the post-8pm evening into tomorrow's
    kill window and mislabels late-boot syncs."""
    return datetime.now(timezone.utc).astimezone(_ET)
LOCK_ARM_AT_R = _rules.ARM_R      # 1.75
LOCK_STOP_R = _rules.LOCK_R       # 0.5


class _Pending:
    """A submitted-not-yet-confirmed entry order."""

    def __init__(self, symbol: str, order_id: str, trade_id: int,
                 entry: float, stop: float, shares: int,
                 tp_leg_id: str, sl_leg_id: str, submitted_ts: float):
        self.symbol = symbol
        self.order_id = order_id
        self.trade_id = trade_id
        self.entry = entry
        self.stop = stop
        self.shares = shares
        self.tp_leg_id = tp_leg_id
        self.sl_leg_id = sl_leg_id
        self.submitted_ts = submitted_ts


class _Open:
    """A filled position under StopMonitor management."""

    def __init__(self, symbol: str, trade_id: int, entry_price: float,
                 stop: float, shares: int,
                 tp_leg_id: str = '', sl_leg_id: str = ''):
        self.symbol = symbol
        self.trade_id = trade_id
        self.entry_price = entry_price
        self.stop = stop
        self.shares = shares
        self.tp_leg_id = tp_leg_id
        self.sl_leg_id = sl_leg_id
        self.eod_sell_order_id = ''


class IgnitionEngine:
    """Executes shadow triggers at micro size. Zero detection logic."""

    def __init__(self, alpaca_client, db, stop_monitor, notifier=None,
                 cfg: Optional[dict] = None, prestage=None):
        """`prestage`: optional trading.ignition_prestage.PrestageManager.
        Every touchpoint is guarded — a None/disabled prestage leaves the
        engine byte-identical to the pre-prestage build (2026-08-22)."""
        cfg = cfg or {}
        self.prestage = prestage
        self.enabled = bool(cfg.get('enabled', False))
        env = os.environ.get('IGNITION_LIVE')
        if env is not None:
            self.enabled = env.strip().lower() not in (
                '0', 'false', 'no', 'off', '')
        self.dry_run = bool(cfg.get('dry_run', True))
        self.risk_usd = float(cfg.get('risk_usd', 50.0))
        self.daily_kill_usd = float(cfg.get('daily_kill_usd', -300.0))
        self.weekly_kill_usd = float(cfg.get('weekly_kill_usd', -750.0))
        self.max_concurrent = int(cfg.get('max_concurrent', 15))
        self.max_notional_usd = float(cfg.get('max_notional_usd', 1500.0))
        self.entry_buffer_bps = float(cfg.get('entry_buffer_bps', 20.0))
        self.entry_timeout_s = float(cfg.get('entry_timeout_s', 90.0))

        self.alpaca = alpaca_client
        self.db = db
        self.stop_monitor = stop_monitor
        self.notifier = notifier

        self.pending: Dict[str, _Pending] = {}
        self.open_positions: Dict[str, _Open] = {}
        # state is touched from the engine worker (trigger handling) AND
        # the scanner thread (process_tick/check_exits/force_close) —
        # one lock covers all check-then-act sequences
        self._lock = threading.RLock()
        self._eod_closing: Dict[str, _Open] = {}
        self._entered_today: set = set()
        self._day: Optional[str] = None
        self._daily_kill_notified = False
        self._weekly_kill_notified = False

        # own worker so order I/O never delays shadow measurement
        self._queue: 'queue.Queue' = queue.Queue(maxsize=100)
        self._worker = threading.Thread(target=self._worker_loop,
                                        daemon=True, name='ignition-engine')
        if self.enabled:
            self._worker.start()
        logger.info(
            f"IgnitionEngine gates: enabled={self.enabled}, "
            f"dry_run={self.dry_run}, risk=${self.risk_usd:.0f}, "
            f"kills={self.daily_kill_usd:.0f}/day "
            f"{self.weekly_kill_usd:.0f}/week, "
            f"max_concurrent={self.max_concurrent}, "
            f"notional_cap=${self.max_notional_usd:.0f}")

    # ------------------------------------------------------------------
    # trigger intake (called from the SHADOW worker thread)
    # ------------------------------------------------------------------
    def enqueue_trigger(self, rec: dict) -> None:
        """Callback target for IgnitionShadow.on_trigger. NEVER raises."""
        # prestage parity ledger + P0-2 reactive sibling staging fires
        # for EVERY shadow trigger, before the engine's own enabled gate
        # (the prestage shadow must see triggers even in engine dry-run)
        if self.prestage is not None:
            try:
                self.prestage.notify_trigger(rec)
            except Exception as e:
                logger.error(f"[IGNITION] prestage notify_trigger "
                             f"failed: {e}")
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(dict(rec))
        except Exception:
            logger.error(
                f"[IGNITION] trigger queue full — DROPPED "
                f"{rec.get('symbol')} (engine wedged?)")

    def _worker_loop(self) -> None:
        while True:
            try:
                rec = self._queue.get(timeout=2.0)
            except Exception:
                continue
            try:
                self._handle_trigger(rec)
            except Exception as e:
                logger.error(f"[IGNITION] trigger handling failed for "
                             f"{rec.get('symbol')}: {e}")
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------
    # gates + sizing + submission
    # ------------------------------------------------------------------
    def _roll_day(self) -> None:
        day = _et_now().strftime('%Y-%m-%d')
        if day != self._day:
            self._day = day
            self._entered_today = set()
            self._daily_kill_notified = False

    def _realized_pnl(self, since_date: str) -> float:
        """DB-derived (survives restarts, zero extra state)."""
        try:
            import sqlite3
            conn = sqlite3.connect(str(self.db._trades_path), timeout=10)
            cur = conn.execute(
                "SELECT COALESCE(SUM(pnl),0) FROM trades WHERE "
                "strategy=? AND trade_date>=? AND pnl IS NOT NULL",
                (STRATEGY_NAME, since_date))
            v = float(cur.fetchone()[0] or 0.0)
            conn.close()
            return v
        except Exception as e:
            logger.error(f"[IGNITION] realized-pnl query failed ({e}) — "
                         f"FAIL-SAFE: treating as kill-breached")
            return -1e9   # fail-closed: no new entries if we can't know

    def _kill_blocked(self) -> Optional[str]:
        now_et = _et_now()
        today = now_et.strftime('%Y-%m-%d')
        week_start = (now_et - timedelta(days=now_et.weekday())
                      ).strftime('%Y-%m-%d')
        wk = self._realized_pnl(week_start)
        if wk <= self.weekly_kill_usd:
            if not self._weekly_kill_notified:
                self._weekly_kill_notified = True
                self._notify(f"[IGNITION] WEEKLY KILL — realized "
                             f"${wk:+,.0f} ≤ ${self.weekly_kill_usd:,.0f}"
                             f"; no new entries this week")
            self._prestage_kill_sweep('weekly')
            return 'weekly_kill'
        dy = self._realized_pnl(today)
        if dy <= self.daily_kill_usd:
            if not self._daily_kill_notified:
                self._daily_kill_notified = True
                self._notify(f"[IGNITION] DAILY KILL — realized "
                             f"${dy:+,.0f} ≤ ${self.daily_kill_usd:,.0f};"
                             f" no new entries today (open positions "
                             f"keep their stops)")
            self._prestage_kill_sweep('daily')
            return 'daily_kill'
        return None

    def _prestage_kill_sweep(self, kind: str) -> None:
        """§C10: kills only block NEW entries, but staged stop-limits
        ARE entries already armed — sweep them in the SAME call path
        that raises the kill (idempotent inside PrestageManager)."""
        if self.prestage is None:
            return
        try:
            self.prestage.notify_kill(kind)
        except Exception as e:
            logger.error(f"[IGNITION] prestage kill sweep failed: {e}")

    def _handle_trigger(self, rec: dict) -> None:
        self._roll_day()
        symbol = rec['symbol']
        with self._lock:
            if symbol in self._entered_today or symbol in self.pending \
                    or symbol in self.open_positions:
                logger.info(f"[IGNITION] {symbol} dedup — already "
                            f"traded/pending today")
                return
            if len(self.pending) + len(self.open_positions) \
                    >= self.max_concurrent:
                logger.warning(f"[IGNITION] {symbol} skipped — "
                               f"max_concurrent {self.max_concurrent}")
                return
            # reserve the slot before the (slow) submit path
            self._entered_today.add(symbol)
        blocked = self._kill_blocked()
        if blocked:
            logger.info(f"[IGNITION] {symbol} blocked by {blocked}")
            return
        # P0-3: chase entry for a staged symbol ONLY after broker-
        # confirmed terminal-and-unfilled disposition. 'adopted' means
        # the stage filled (position exists) — the chase MUST NOT enter;
        # 'blocked' means disposition unprovable — a double position is
        # strictly worse than a missed chase.
        if self.prestage is not None:
            try:
                disp = self.prestage.resolve_for_chase(symbol)
            except Exception as e:
                logger.error(f"[IGNITION] {symbol} prestage disposition "
                             f"check FAILED ({e}) — chase blocked "
                             f"(fail-closed, no double entry)")
                disp = 'blocked'
            if disp != 'chase_ok':
                logger.warning(f"[IGNITION] {symbol} chase skipped — "
                               f"prestage disposition={disp}")
                return
        entry = float(rec.get('_entry') or rec.get('ask')
                      or rec['price'])
        stop = float(rec.get('_stop')
                     or rec['price'] * (1 - rec['r_pct'] / 100.0))
        if not entry > stop > 0:
            logger.error(f"[IGNITION] {symbol} bad levels entry={entry} "
                         f"stop={stop} — skipped")
            return
        shares = math.floor(self.risk_usd / (entry - stop))
        if shares < 1:
            logger.info(f"[IGNITION] {symbol} <1 share at "
                        f"${self.risk_usd:.0f} risk — skipped")
            return
        if shares * entry > self.max_notional_usd:
            shares = math.floor(self.max_notional_usd / entry)
            if shares < 1:
                return
        ask = float(rec.get('ask') or entry)
        limit = round(ask * (1 + self.entry_buffer_bps / 1e4), 4)
        if self.dry_run:
            self._notify(
                f"[IGNITION DRY] WOULD BUY {shares} {symbol} "
                f"@≤{limit} (entry~{entry:.2f} stop {stop:.2f} "
                f"risk ${shares * (entry - stop):.0f}) "
                f"cat={rec.get('catalyst')}")
            logger.info(f"[IGNITION DRY] {symbol} {shares}sh "
                        f"limit={limit} stop={stop}")
            return
        self._submit(rec, symbol, shares, limit, entry, stop)

    def _submit(self, rec: dict, symbol: str, shares: int,
                limit: float, entry: float, stop: float) -> None:
        try:
            result = self.alpaca.submit_bracket_order(
                symbol=symbol, qty=shares, side='buy',
                limit_price=limit,
                tp_price=round(entry * 3.0, 2),      # unreachable safety
                sl_price=round(stop * 0.90, 2))      # broker dead-man
        except Exception as e:
            logger.error(f"[IGNITION] {symbol} submit failed: {e}")
            self._notify(f"[IGNITION] ⚠ {symbol} submit FAILED: "
                         f"{str(e)[:80]}")
            return
        order_id = result.get('id', '')
        tp_leg = sl_leg = ''
        for leg in (result.get('legs') or []):
            if leg.get('type') == 'limit' or leg.get('order_type') == 'limit':
                tp_leg = leg.get('id', '')
            else:
                sl_leg = leg.get('id', '')
        q = {}
        try:
            q = self.alpaca.get_latest_quote(symbol) or {}
        except Exception as e:
            logger.warning(f"[IGNITION] {symbol} entry quote fetch "
                           f"failed: {e}")
        # save_trade's INSERT binds a FIXED named-parameter list (verified
        # empirically 8/14): every listed key MUST be present (None ok)
        # and any EXTRA key is SILENTLY DROPPED — telemetry therefore
        # goes through update_trade immediately after the insert.
        record = {
            'trade_date': _et_now().strftime('%Y-%m-%d'),
            'symbol': symbol, 'side': 'buy', 'strategy': STRATEGY_NAME,
            'shares': shares, 'entry_price': entry,
            'stop_loss_price': round(stop * 0.90, 2),
            'take_profit_price': round(entry * 3.0, 2),
            'risk_per_share': entry - stop,
            'total_risk': shares * (entry - stop),
            'risk_reward_ratio': 0.0,
            'order_id': order_id, 'order_status': 'pending_new',
            'fill_price': None, 'filled_at': None,
            'exit_price': None, 'exit_reason': None, 'exited_at': None,
            'pnl': None, 'pnl_pct': None,
            'pattern_data': json.dumps({
                'catalyst': rec.get('catalyst'),
                'r_pct': rec.get('r_pct'),
                'spread_bps': rec.get('spread_bps'),
                'anchor': rec.get('anchor'),
                'anchor_cohort': rec.get('anchor_cohort'),
                'chg_from_open': rec.get('chg_from_open'),
                'trigger_m': rec.get('trigger_m'),
                'minute_et': rec.get('minute_et'),
                'latency_s': rec.get('latency_s'),
                'lock_arm_at_r': LOCK_ARM_AT_R,
                'lock_stop_r': LOCK_STOP_R,
                'hypo_entry': rec.get('hypo_entry'),
                'hypo_stop': rec.get('hypo_stop'),
                'shadow_day': rec.get('day'),
            }),
        }
        trade_id = None
        try:
            trade_id = self.db.save_trade(record)
            self.db.update_trade(trade_id, {
                'real_stop_loss_price': stop,
                'entry_quote_bid': q.get('bid_price'),
                'entry_quote_ask': q.get('ask_price'),
                'entry_quote_bid_size': q.get('bid_size'),
                'entry_quote_ask_size': q.get('ask_size'),
                'entry_quote_spread': (
                    (q['ask_price'] - q['bid_price'])
                    if q.get('ask_price') and q.get('bid_price')
                    else None),
            })
        except Exception as e:
            logger.error(f"[IGNITION] {symbol} DB save failed: {e}")
        with self._lock:
            self.pending[symbol] = _Pending(
                symbol, order_id, trade_id, entry, stop, shares,
                tp_leg, sl_leg, time.time())
        if self.prestage is not None:
            try:
                self.prestage.notify_chase_entry(symbol)
            except Exception as e:
                logger.warning(f"[IGNITION] prestage chase-entry ledger "
                               f"failed: {e}")
        self._notify(f"[IGNITION] ENTRY SUBMITTED {shares} {symbol} "
                     f"@≤{limit} stop {stop:.2f} "
                     f"(risk ${shares * (entry - stop):.0f}, "
                     f"cat={rec.get('catalyst')})")
        logger.info(f"[IGNITION] ENTRY SUBMITTED: {symbol} {shares}sh "
                    f"limit={limit} order={order_id}")

    # ------------------------------------------------------------------
    # tick processing (called from the scanner loop, like ORB)
    # ------------------------------------------------------------------
    def process_tick(self) -> None:
        # prestage scheduler runs regardless of the engine's dry_run —
        # its own gates (enabled/shadow/IGNITION_PRESTAGE) govern it
        if self.prestage is not None:
            try:
                self.prestage.process_tick()
            except Exception as e:
                logger.error(f"[IGNITION] prestage tick failed: {e}")
        if not self.enabled or self.dry_run:
            return
        self.process_eod_fills()
        for symbol in list(self.pending.keys()):
            p = self.pending[symbol]
            try:
                od = self.alpaca.get_order(p.order_id) or {}
            except Exception as e:
                logger.warning(f"[IGNITION] {symbol} order poll failed: "
                               f"{e}")
                continue
            status = od.get('status', '')
            if status == 'filled':
                self._confirm_fill(p, od)
            elif status in ('canceled', 'cancelled', 'expired',
                            'rejected'):
                logger.warning(f"[IGNITION] {symbol} order {status}")
                self._mark_cancelled(p, status)
            elif time.time() - p.submitted_ts > self.entry_timeout_s:
                logger.info(f"[IGNITION] {symbol} unfilled "
                            f">{self.entry_timeout_s:.0f}s — cancel "
                            f"(momentum gone)")
                try:
                    self.alpaca.cancel_order(p.order_id)
                except Exception as e:
                    logger.warning(f"[IGNITION] cancel failed: {e}")
                self._mark_cancelled(p, 'entry_timeout_canceled')

    def _confirm_fill(self, p: _Pending, od: dict) -> None:
        fill_price = float(od.get('filled_avg_price')
                           or od.get('fill_price') or p.entry)
        filled_qty = int(float(od.get('filled_qty') or p.shares))
        q = {}
        try:
            q = self.alpaca.get_latest_quote(p.symbol) or {}
        except Exception:
            pass
        if p.trade_id:
            try:
                self.db.update_trade(p.trade_id, {
                    'order_status': 'filled',
                    'fill_price': fill_price,
                    'filled_at': datetime.now(timezone.utc).isoformat(),
                    'filled_qty': filled_qty,
                    'entry_fill_quote_bid': q.get('bid_price'),
                    'entry_fill_quote_ask': q.get('ask_price'),
                })
            except Exception as e:
                logger.error(f"[IGNITION] fill DB update failed: {e}")
        risk_ps = fill_price - p.stop
        now_min_end = (int(time.time() // 60) + 1) * 60
        self.stop_monitor.add_watch(
            symbol=p.symbol, stop_price=p.stop, shares=filled_qty,
            tp_leg_id=p.tp_leg_id, sl_leg_id=p.sl_leg_id,
            trade_db_id=p.trade_id, entry_price=fill_price,
            risk_per_share=risk_ps if risk_ps > 0 else (p.entry - p.stop),
            strategy=STRATEGY_NAME,
            lock_arm_at_r=LOCK_ARM_AT_R, lock_stop_r=LOCK_STOP_R,
            lock_r_unit=p.entry - p.stop,
            skip_exits_until_ts=float(now_min_end))
        with self._lock:
            self.open_positions[p.symbol] = _Open(
                p.symbol, p.trade_id, fill_price, p.stop, filled_qty,
                tp_leg_id=p.tp_leg_id, sl_leg_id=p.sl_leg_id)
            self.pending.pop(p.symbol, None)
        self._notify(f"[IGNITION] FILLED {filled_qty} {p.symbol} "
                     f"@ {fill_price:.3f} (planned {p.entry:.3f}, "
                     f"slip {(fill_price - p.entry) / p.entry * 1e4:+.0f}"
                     f"bps) stop {p.stop:.2f}")
        logger.info(f"[IGNITION] FILLED {p.symbol} {filled_qty}sh "
                    f"@{fill_price} — watch added (lock "
                    f"{LOCK_ARM_AT_R}R/{LOCK_STOP_R}R)")
        # FILL QUALITY line (8/21, owner ask): the S3 verdict metric,
        # derived once here so EOD/weekly audits grep it instead of
        # recomputing from quote telemetry. chase = fill vs the shadow
        # trigger level (the BT book's assumed entry); vs_ask = fill vs
        # the ask at submit (execution quality proper).
        try:
            ask_at_submit = float(q.get('ask_price') or 0.0)
            chase_bps = (fill_price - p.entry) / p.entry * 1e4
            vs_ask_bps = ((fill_price - ask_at_submit) / ask_at_submit
                          * 1e4) if ask_at_submit > 0 else float('nan')
            latency_s = time.time() - p.submitted_ts
            risk_realized = (fill_price - p.stop) * filled_qty
            # path=chase tag (P1-7, 2026-08-22): staged fills log their
            # own FILL QUALITY line (path=staged, metric fill-vs-level)
            # from PrestageManager — aggregation MUST split by path or
            # the S3 fill ledger pools two different metrics.
            logger.info(
                f"[IGNITION] FILL QUALITY {p.symbol}: path=chase "
                f"chase={chase_bps:+.0f}bps "
                f"vs_ask={vs_ask_bps:+.0f}bps latency={latency_s:.1f}s "
                f"risk_realized=${risk_realized:.0f} "
                f"(planned ${self.risk_usd:.0f})")
        except Exception as e:
            logger.warning(f"[IGNITION] fill-quality line failed: {e}")

    def _mark_cancelled(self, p: _Pending, reason: str) -> None:
        if p.trade_id:
            try:
                self.db.update_trade(p.trade_id, {
                    'order_status': 'cancelled', 'exit_reason': reason})
            except Exception as e:
                logger.error(f"[IGNITION] cancel DB update failed: {e}")
        self.pending.pop(p.symbol, None)

    # ------------------------------------------------------------------
    # exits
    # ------------------------------------------------------------------
    def check_exits(self) -> None:
        if not self.enabled or self.dry_run:
            return
        try:
            events = self.stop_monitor.drain_exit_events(
                strategy=STRATEGY_NAME)
        except Exception as e:
            logger.error(f"[IGNITION] drain_exit_events failed: {e}")
            return
        for ev in events:
            self._handle_exit_event(ev)

    def _handle_exit_event(self, ev) -> None:
        pos = self.open_positions.pop(ev.symbol, None)
        if pos is None:
            # staged fills live in the PRESTAGE book, not ours — without
            # this delegation their exits were dropped as orphans and the
            # DB row stayed open, pnl invisible to the kills (2026-08-28)
            if self.prestage is not None:
                try:
                    if self.prestage.handle_exit_event(ev):
                        return
                except Exception as e:
                    logger.error(f"[IGNITION] prestage exit delegation "
                                 f"failed for {ev.symbol}: {e}")
            logger.warning(f"[IGNITION] exit event for {ev.symbol} but "
                           f"no tracked position — orphan?")
            return
        confirmed = getattr(ev, 'confirmed', True)
        exit_price = float(ev.exit_price)
        pnl = (exit_price - pos.entry_price) * pos.shares
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100.0
        exit_update = build_exit_update(ev)
        if confirmed:
            exit_update['pnl'] = pnl
            exit_update['pnl_pct'] = pnl_pct
            elp = exit_update.get('exit_limit_price')
            exit_update['exit_slippage'] = (
                elp - exit_price if isinstance(elp, (int, float))
                else None)
        else:
            logger.error(f"[IGNITION] {ev.symbol} UNCONFIRMED EXIT — "
                         f"exit_pending_verification; reconciler retries")
        try:
            self.db.update_trade(pos.trade_id, exit_update)
        except Exception as e:
            logger.error(f"[IGNITION] {ev.symbol} exit DB update "
                         f"failed: {e}")
        self._notify(f"[IGNITION] EXIT {ev.symbol} @ {exit_price:.3f} "
                     f"({ev.exit_reason}) PnL ${pnl:+,.0f} "
                     f"({pnl_pct:+.1f}%)")
        logger.info(f"[IGNITION] EXIT {ev.symbol} {ev.exit_reason} "
                    f"@{exit_price} pnl=${pnl:+,.2f}")

    # ------------------------------------------------------------------
    # 15:45 force-flat + boot resume
    # ------------------------------------------------------------------
    def force_close_all(self) -> int:
        """EOD flat. IDEMPOTENT (8/14 independent audit P1: the scanner
        calls this EVERY tick after 15:45 — the first version re-sold
        the same shares each cycle and never recorded the exit, blinding
        the realized-P&L kills to the most common exit path). Positions
        move to _eod_closing; process_eod_fills() confirms the sell and
        writes the exit row with pnl."""
        if not self.enabled or self.dry_run:
            return 0
        n = 0
        with self._lock:
            pend = list(self.pending.values())
            opens = list(self.open_positions.values())
        for p in pend:
            try:
                self.alpaca.cancel_order(p.order_id)
            except Exception as e:
                logger.warning(f"[IGNITION] force-close cancel failed "
                               f"{p.symbol}: {e}")
            self._mark_cancelled(p, 'eod_canceled')
            n += 1
        for pos in opens:
            symbol = pos.symbol
            with self._lock:
                # move OUT of open_positions FIRST -> re-entrant calls
                # see nothing to close (idempotency)
                if symbol not in self.open_positions:
                    continue
                del self.open_positions[symbol]
                self._eod_closing[symbol] = pos
            try:
                self.stop_monitor.remove_watch(symbol)
            except Exception as e:
                logger.warning(f"[IGNITION] EOD remove_watch failed "
                               f"{symbol}: {e}")
            # cancel bracket legs FIRST — they hold the shares
            for leg_id in (pos.tp_leg_id, pos.sl_leg_id):
                if leg_id:
                    try:
                        self.alpaca.cancel_order(leg_id)
                    except Exception as e:
                        logger.warning(f"[IGNITION] EOD leg cancel "
                                       f"failed {symbol}/{leg_id}: {e}")
            try:
                q = self.alpaca.get_latest_quote(symbol) or {}
                bid = float(q.get('bid_price') or 0)
                limit = round(bid * 0.995, 4) if bid \
                    else round(pos.entry_price * 0.90, 2)
                res = self.alpaca.submit_limit_sell_order(
                    symbol, pos.shares, limit)
                pos.eod_sell_order_id = (res or {}).get('id', '')
                logger.info(f"[IGNITION] EOD flat: selling {pos.shares} "
                            f"{symbol} @>={limit} "
                            f"order={pos.eod_sell_order_id}")
                n += 1
            except Exception as e:
                logger.error(f"[IGNITION] EOD close failed {symbol}: "
                             f"{e}")
                self._mark_exit_unverified(pos, f'eod submit: {e}')
        return n

    def process_eod_fills(self) -> None:
        """Confirm EOD sells and RECORD the exits (pnl included) — the
        realized-P&L kills depend on these rows. Called from
        process_tick each cycle."""
        for symbol in list(self._eod_closing.keys()):
            pos = self._eod_closing[symbol]
            oid = getattr(pos, 'eod_sell_order_id', '')
            if not oid:
                self._mark_exit_unverified(pos, 'eod sell has no order id')
                self._eod_closing.pop(symbol, None)
                continue
            try:
                od = self.alpaca.get_order(oid) or {}
            except Exception as e:
                logger.warning(f"[IGNITION] EOD fill poll failed "
                               f"{symbol}: {e}")
                continue
            status = od.get('status', '')
            if status == 'filled':
                exit_price = float(od.get('filled_avg_price')
                                   or pos.entry_price)
                pnl = (exit_price - pos.entry_price) * pos.shares
                try:
                    self.db.update_trade(pos.trade_id, {
                        'exit_price': exit_price,
                        'exit_reason': 'eod_flat',
                        'exited_at': datetime.now(
                            timezone.utc).isoformat(),
                        'order_status': 'closed',
                        'pnl': pnl,
                        'pnl_pct': (exit_price - pos.entry_price)
                        / pos.entry_price * 100.0,
                    })
                except Exception as e:
                    logger.error(f"[IGNITION] EOD exit DB update "
                                 f"failed {symbol}: {e}")
                self._eod_closing.pop(symbol, None)
                self._notify(f"[IGNITION] EOD FLAT {symbol} @ "
                             f"{exit_price:.3f} PnL ${pnl:+,.0f}")
            elif status in ('canceled', 'cancelled', 'expired',
                            'rejected'):
                self._mark_exit_unverified(
                    pos, f'eod sell {status}')
                self._eod_closing.pop(symbol, None)

    def finalize_eod(self, timeout_s: float = 45.0,
                     poll_interval_s: float = 3.0) -> None:
        """Final grace-poll for pending EOD sell confirmations at shutdown.

        2026-08-21 incident (DFNS id=349): the service's clean 20:00 UTC
        exit killed process_eod_fills mid-poll — the broker fill landed but
        the DB row stayed 'filled' (open) with NO unverified mark, making
        the loss invisible to the realized-P&L kills until a manual
        reconcile. Called from the scanner shutdown sequence right after
        force_close_all(): poll until confirmations land or timeout, then
        mark anything still pending as exit_pending_verification so the
        green check and the next boot's reconciler see it.
        """
        # stage sweep FIRST (§A2 shutdown-race class): resting
        # stop-limits must be cancelled before the process exits
        if self.prestage is not None:
            try:
                self.prestage.shutdown_sweep()
            except Exception as e:
                logger.error(f"[IGNITION] prestage shutdown sweep "
                             f"failed: {e}")
        deadline = time.monotonic() + timeout_s
        while self._eod_closing and time.monotonic() < deadline:
            self.process_eod_fills()
            if not self._eod_closing:
                break
            time.sleep(poll_interval_s)
        for symbol in list(self._eod_closing.keys()):
            pos = self._eod_closing.pop(symbol)
            self._mark_exit_unverified(
                pos, f'shutdown before fill confirm ({timeout_s:.0f}s grace)')

    def _mark_exit_unverified(self, pos, why: str) -> None:
        logger.error(f"[IGNITION] {pos.symbol} exit UNVERIFIED ({why}) "
                     f"— order_status=exit_pending_verification")
        try:
            self.db.update_trade(pos.trade_id, {
                'order_status': 'exit_pending_verification'})
        except Exception as e:
            logger.error(f"[IGNITION] unverified-mark failed: {e}")
        self._notify(f"[IGNITION] ⚠ {pos.symbol} EOD exit UNVERIFIED "
                     f"({why}) — MANUAL CHECK; kills may under-count "
                     f"until reconciled")

    def sync_positions(self) -> None:
        """Boot/mid-session-restart rehydration (mirrors ORB)."""
        # prestage boot reconciliation runs BEFORE anything trades and
        # regardless of the engine's dry_run — a previous non-shadow run
        # may have left resting ign-stage-* orders at the broker (§A1)
        if self.prestage is not None:
            try:
                self.prestage.boot_reconcile()
            except Exception as e:
                logger.error(f"[IGNITION] prestage boot reconcile "
                             f"failed: {e}")
        if not self.enabled or self.dry_run:
            return
        # Look back 5 calendar days, not just today: a row stuck 'filled'
        # by a shutdown race (2026-08-21 DFNS incident — EOD fill landed
        # at the broker but the poll died with the service) is a PRIOR-day
        # row by the next boot, and a today-only sync never sees it, so
        # the loss stays invisible to the realized-P&L kills.
        now_et = _et_now()
        rows = []
        for back in range(5):
            d = (now_et - timedelta(days=back)).strftime('%Y-%m-%d')
            try:
                rows.extend(
                    self.db.get_open_trades(d, strategy=STRATEGY_NAME))
            except Exception as e:
                logger.error(f"[IGNITION] sync: DB read failed ({d}): {e}")
        # NOTE: no early return on empty rows — the orphan reconciler
        # below must run at every boot regardless (8/14 audit P2).
        try:
            broker = {p['symbol']: p for p in
                      (self.alpaca.get_open_positions() or [])}
        except Exception as e:
            logger.error(f"[IGNITION] sync: broker read failed: {e}")
            broker = {}
        n_watch = n_pend = 0
        for row in rows:
            sym = row['symbol']
            status = row.get('order_status')
            if status == 'filled':
                if sym not in broker:
                    logger.error(
                        f"[IGNITION] sync: {sym} filled in DB but GONE "
                        f"at broker — exit happened while down; "
                        f"orphan-marking for reconciler")
                    try:
                        self.db.update_trade(row['id'], {
                            'order_status': 'exit_pending_verification'})
                    except Exception:
                        pass
                    self._notify(f"[IGNITION] ⚠ {sym} exited while "
                                 f"engine was down — reconcile manually")
                    continue
                pd_json = {}
                try:
                    pd_json = json.loads(row.get('pattern_data') or '{}')
                except Exception:
                    pass
                stop = float(row.get('real_stop_loss_price')
                             or row.get('stop_loss_price'))
                entry = float(row.get('fill_price')
                              or row.get('entry_price'))
                # recover bracket leg IDs from the broker so exits (both
                # StopMonitor's and EOD flat) can cancel them — empty
                # leg IDs after a restart would strand held shares
                tp_leg = sl_leg = ''
                try:
                    od = self.alpaca.get_order(row.get('order_id')) or {}
                    for leg in (od.get('legs') or []):
                        if leg.get('type') == 'limit':
                            tp_leg = leg.get('id', '')
                        else:
                            sl_leg = leg.get('id', '')
                except Exception as e:
                    logger.warning(f"[IGNITION] sync: leg recovery "
                                   f"failed {sym}: {e}")
                self.stop_monitor.add_watch(
                    symbol=sym, stop_price=stop,
                    shares=int(row.get('filled_qty') or row['shares']),
                    tp_leg_id=tp_leg, sl_leg_id=sl_leg,
                    trade_db_id=row['id'], entry_price=entry,
                    risk_per_share=max(entry - stop, 0.01),
                    strategy=STRATEGY_NAME,
                    lock_arm_at_r=float(pd_json.get('lock_arm_at_r',
                                                    LOCK_ARM_AT_R)),
                    lock_stop_r=float(pd_json.get('lock_stop_r',
                                                  LOCK_STOP_R)),
                    lock_r_unit=max(entry - stop, 0.01))
                self.open_positions[sym] = _Open(
                    sym, row['id'], entry, stop,
                    int(row.get('filled_qty') or row['shares']),
                    tp_leg_id=tp_leg, sl_leg_id=sl_leg)
                n_watch += 1
            elif status == 'pending_new' and row.get('order_id'):
                self.pending[sym] = _Pending(
                    sym, row['order_id'], row['id'],
                    float(row['entry_price']),
                    float(row.get('real_stop_loss_price')
                          or row['stop_loss_price']),
                    int(row['shares']), '', '', time.time())
                n_pend += 1
            self._entered_today.add(sym)
        logger.info(f"[IGNITION] sync_positions: {n_watch} watches "
                    f"re-added, {n_pend} pending resumed")
        self._run_orphan_reconciler(broker)

    def _run_orphan_reconciler(self, broker: dict) -> None:
        """Boot-time orphan pass (8/14 audit P2 — same machinery as ORB/
        bull-flag). SHARED-ACCOUNT NUANCE: ignition trades the MAIN
        account alongside bull flag, so sibling strategies' open
        positions must count as tracked — otherwise every bull-flag
        position gets a spurious 'foreign orphan' alert on each boot
        (and vice versa). Ownership stays safe regardless: the OWNED
        predicate requires a matching ignition DB row."""
        try:
            from trading.orphan_reconciler import (
                ReconcilerConfig, reconcile_strategy_orphans)
            sibling_open = set()
            try:
                import sqlite3
                since = (_et_now() - timedelta(days=14)
                         ).strftime('%Y-%m-%d')
                conn = sqlite3.connect(str(self.db._trades_path),
                                       timeout=10)
                sibling_open = {r[0] for r in conn.execute(
                    "SELECT DISTINCT symbol FROM trades WHERE "
                    "strategy != ? AND trade_date >= ? AND "
                    "fill_price IS NOT NULL AND exit_price IS NULL",
                    (STRATEGY_NAME, since))}
                conn.close()
            except Exception as e:
                logger.warning(f"[IGNITION] reconciler sibling query "
                               f"failed ({e}) — proceeding without "
                               f"sibling exclusion (alert-only risk)")
            snapshot = []
            for p in broker.values():
                try:
                    snapshot.append({
                        'symbol': p.get('symbol'),
                        'qty': int(float(p.get('qty') or 0)),
                        'avg_entry_price': float(
                            p.get('avg_entry_price') or 0),
                        'unrealized_pl': float(
                            p.get('unrealized_pl') or 0)})
                except Exception:
                    continue
            tracked = (set(self.open_positions) | set(self.pending)
                       | sibling_open)
            reconcile_strategy_orphans(
                strategy=STRATEGY_NAME, alpaca=self.alpaca, db=self.db,
                notifier=self.notifier, tracked_symbols=tracked,
                cfg=ReconcilerConfig(), broker_positions=snapshot)
        except Exception as e:
            logger.error(f"[IGNITION] orphan reconciler raised: {e} — "
                         f"sync continues")

    # ------------------------------------------------------------------
    def _notify(self, msg: str) -> None:
        logger.info(msg)
        if self.notifier:
            try:
                self.notifier.send_message_sync(msg)
            except Exception as e:
                logger.warning(f"[IGNITION] telegram failed: {e}")
