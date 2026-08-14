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

from trading import ignition_rules as _rules
from trading.stop_monitor import build_exit_update

logger = logging.getLogger(__name__)

STRATEGY_NAME = 'ignition'
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
                 stop: float, shares: int):
        self.symbol = symbol
        self.trade_id = trade_id
        self.entry_price = entry_price
        self.stop = stop
        self.shares = shares


class IgnitionEngine:
    """Executes shadow triggers at micro size. Zero detection logic."""

    def __init__(self, alpaca_client, db, stop_monitor, notifier=None,
                 cfg: Optional[dict] = None):
        cfg = cfg or {}
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
        # UTC date is fine: the worker only sees triggers during RTH
        day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
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
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        week_start = (datetime.now(timezone.utc)
                      - timedelta(days=datetime.now(timezone.utc)
                                  .weekday())).strftime('%Y-%m-%d')
        wk = self._realized_pnl(week_start)
        if wk <= self.weekly_kill_usd:
            if not self._weekly_kill_notified:
                self._weekly_kill_notified = True
                self._notify(f"[IGNITION] WEEKLY KILL — realized "
                             f"${wk:+,.0f} ≤ ${self.weekly_kill_usd:,.0f}"
                             f"; no new entries this week")
            return 'weekly_kill'
        dy = self._realized_pnl(today)
        if dy <= self.daily_kill_usd:
            if not self._daily_kill_notified:
                self._daily_kill_notified = True
                self._notify(f"[IGNITION] DAILY KILL — realized "
                             f"${dy:+,.0f} ≤ ${self.daily_kill_usd:,.0f};"
                             f" no new entries today (open positions "
                             f"keep their stops)")
            return 'daily_kill'
        return None

    def _handle_trigger(self, rec: dict) -> None:
        self._roll_day()
        symbol = rec['symbol']
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
        blocked = self._kill_blocked()
        if blocked:
            logger.info(f"[IGNITION] {symbol} blocked by {blocked}")
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
            self._entered_today.add(symbol)
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
        record = {
            'trade_date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            'symbol': symbol, 'side': 'buy', 'strategy': STRATEGY_NAME,
            'shares': shares, 'entry_price': entry,
            'stop_loss_price': round(stop * 0.90, 2),
            'real_stop_loss_price': stop,
            'take_profit_price': round(entry * 3.0, 2),
            'risk_per_share': entry - stop,
            'total_risk': shares * (entry - stop),
            'risk_reward_ratio': 0.0,
            'order_id': order_id, 'order_status': 'pending_new',
            'entry_quote_bid': q.get('bid_price'),
            'entry_quote_ask': q.get('ask_price'),
            'entry_quote_bid_size': q.get('bid_size'),
            'entry_quote_ask_size': q.get('ask_size'),
            'entry_quote_spread': (
                (q['ask_price'] - q['bid_price'])
                if q.get('ask_price') and q.get('bid_price') else None),
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
        try:
            trade_id = self.db.save_trade(record)
        except Exception as e:
            logger.error(f"[IGNITION] {symbol} DB save failed: {e}")
            trade_id = None
        self.pending[symbol] = _Pending(
            symbol, order_id, trade_id, entry, stop, shares,
            tp_leg, sl_leg, time.time())
        self._entered_today.add(symbol)
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
        if not self.enabled or self.dry_run:
            return
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
        self.open_positions[p.symbol] = _Open(
            p.symbol, p.trade_id, fill_price, p.stop, filled_qty)
        del self.pending[p.symbol]
        self._notify(f"[IGNITION] FILLED {filled_qty} {p.symbol} "
                     f"@ {fill_price:.3f} (planned {p.entry:.3f}, "
                     f"slip {(fill_price - p.entry) / p.entry * 1e4:+.0f}"
                     f"bps) stop {p.stop:.2f}")
        logger.info(f"[IGNITION] FILLED {p.symbol} {filled_qty}sh "
                    f"@{fill_price} — watch added (lock "
                    f"{LOCK_ARM_AT_R}R/{LOCK_STOP_R}R)")

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
        if not self.enabled or self.dry_run:
            return 0
        n = 0
        for symbol in list(self.pending.keys()):
            p = self.pending[symbol]
            try:
                self.alpaca.cancel_order(p.order_id)
            except Exception as e:
                logger.warning(f"[IGNITION] force-close cancel failed "
                               f"{symbol}: {e}")
            self._mark_cancelled(p, 'eod_canceled')
            n += 1
        for symbol in list(self.open_positions.keys()):
            pos = self.open_positions[symbol]
            try:
                self.stop_monitor.remove_watch(symbol)
            except Exception as e:
                logger.warning(f"[IGNITION] EOD remove_watch failed "
                               f"{symbol}: {e}")
            try:
                q = self.alpaca.get_latest_quote(symbol) or {}
                bid = float(q.get('bid_price') or 0)
                # marketable limit slightly below bid; deep fallback if
                # the quote is missing (must not fail to flatten)
                limit = round(bid * 0.995, 4) if bid \
                    else round(pos.entry_price * 0.90, 2)
                self.alpaca.submit_limit_sell_order(
                    symbol, pos.shares, limit)
                logger.info(f"[IGNITION] EOD flat: selling {pos.shares} "
                            f"{symbol} @>={limit}")
                n += 1
            except Exception as e:
                logger.error(f"[IGNITION] EOD close failed {symbol}: "
                             f"{e}")
                self._notify(f"[IGNITION] ⚠ EOD close FAILED {symbol}: "
                             f"{str(e)[:80]} — MANUAL CHECK")
        return n

    def sync_positions(self) -> None:
        """Boot/mid-session-restart rehydration (mirrors ORB)."""
        if not self.enabled or self.dry_run:
            return
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        try:
            rows = self.db.get_open_trades(today, strategy=STRATEGY_NAME)
        except Exception as e:
            logger.error(f"[IGNITION] sync: DB read failed: {e}")
            return
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
                self.stop_monitor.add_watch(
                    symbol=sym, stop_price=stop,
                    shares=int(row.get('filled_qty') or row['shares']),
                    tp_leg_id='', sl_leg_id='',
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
                    int(row.get('filled_qty') or row['shares']))
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

    # ------------------------------------------------------------------
    def _notify(self, msg: str) -> None:
        logger.info(msg)
        if self.notifier:
            try:
                self.notifier.send_message_sync(msg)
            except Exception as e:
                logger.warning(f"[IGNITION] telegram failed: {e}")
