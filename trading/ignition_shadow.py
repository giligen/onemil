"""Ignition S1 signal shadow — ZERO ORDERS by construction (2026-07-19).

Passive consumer of the scanner's qualified/hot stream. On each first
sighting of a +10% intraday mover (gap<5%, before the trigger cutoff) it
applies the researched Ignition filters (R>=5% consolidation, catalyst =
news OR complex-confirmation), captures the LIVE NBBO at that moment,
and journals a hypothetical trade to logs/ignition_shadow_<day>.jsonl.

Purpose (S1 of the shadow plan, research/ignition_program_design.md):
measure the two things the harness could not — real detection latency
and real spread/depth at trigger — before any order is ever placed.
Exits are NOT tracked live; the nightly report resims them from bars
(parity with the harness by construction).

This module contains NO order-submission code. Escalation to paper (S2)
and micro-live (S3) happens in a separate engine, later, deliberately.

Evidence for the strategy: research/ignition_program_design.md —
latency-honest book $245.8K/19mo, all eras, 12/19 monster months, under
participation-scaled fills. Constants below are the researched values,
pinned here + config.yaml (live-parity doctrine: one source of truth).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

# Researched constants (pinned; config.yaml may override, same values)
TRIGGER_PCT = 10.0          # +10% from open = ignition
MAX_TRIGGER_MIN_ET = 630    # 10:30 ET
MIN_R_PCT = 5.0             # consolidation stop distance floor
GAP_MAX_PCT = 5.0           # disjoint from ORB (gap>=5 is ORB's)
MIN_COHORT = 2              # complex confirmation
MODEL_RISK_USD = 3000.0
POS_CAP_USD = 25000.0


class IgnitionShadow:
    """Journal-only shadow. One evaluation per symbol per day."""

    def __init__(self, alpaca_client, config: Optional[dict] = None,
                 log_dir: Optional[str] = None):
        cfg = ((config or {}).get('ignition_shadow') or {})
        self.enabled = bool(cfg.get('enabled', True))
        env = os.environ.get('IGNITION_SHADOW')
        if env is not None:
            self.enabled = env.strip().lower() not in (
                '0', 'false', 'no', 'off', '')
        self.trigger_pct = float(cfg.get('trigger_pct', TRIGGER_PCT))
        self.max_trigger_min = int(cfg.get('max_trigger_min_et',
                                           MAX_TRIGGER_MIN_ET))
        self.min_r_pct = float(cfg.get('min_r_pct', MIN_R_PCT))
        self.gap_max = float(cfg.get('gap_max_pct', GAP_MAX_PCT))
        self.min_cohort = int(cfg.get('min_cohort', MIN_COHORT))
        self.alpaca = alpaca_client
        self._seen_today: Set[str] = set()
        self._anchors: Dict[str, Optional[str]] = {}
        self._day_anchor_counts: Dict[str, int] = {}
        self._day: Optional[str] = None
        self._log_dir = log_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'logs')
        self._class_map = None
        self._class_names: Dict[str, str] = {}
        # safety rails: bounded API work inside the scanner cycle
        self.max_evals_per_day = int(cfg.get('max_evals_per_day', 25))
        self._evals_today = 0
        # no-catalyst skips wait for late complex confirmation
        self._await_confirm: Dict[str, dict] = {}
        # WORKER-THREAD DESIGN (2026-07-19 pre-deploy review): all API
        # work (bars/quote, up to ~12s bounded per eval) happens on this
        # daemon worker — the scanner's on_mover() only ENQUEUES
        # (microseconds). The worker owns ALL mutable state; the scanner
        # thread never touches it. A wedged worker loses shadow data,
        # never scanner time.
        import queue as _q
        import threading as _t
        self._queue: '_q.Queue' = _q.Queue(maxsize=500)
        self._worker = _t.Thread(target=self._worker_loop, daemon=True,
                                 name='ignition-shadow')
        if self.enabled:
            self._worker.start()
        if self.enabled:
            logger.info(
                f"IgnitionShadow ACTIVE (journal-only, zero orders): "
                f"trigger>={self.trigger_pct}%, gap<{self.gap_max}%, "
                f"R>={self.min_r_pct}%, cutoff "
                f"{self.max_trigger_min // 60}:{self.max_trigger_min % 60:02d} ET")

    # ------------------------------------------------------------------
    def _roll_day(self, day: str) -> None:
        if day != self._day:
            self._day = day
            self._seen_today = set()
            self._day_anchor_counts = {}
            self._evals_today = 0
            self._await_confirm = {}

    def _anchor(self, symbol: str) -> Optional[str]:
        if symbol in self._anchors:
            return self._anchors[symbol]
        try:
            from trading.orb_asset_class import (
                DEFAULT_CLASS_MAP, load_class_map, underlying_anchor)
            if self._class_map is None:
                self._class_map = load_class_map()
                try:
                    import csv as _csv
                    with open(DEFAULT_CLASS_MAP, newline='') as fh:
                        for row in _csv.DictReader(fh):
                            self._class_names[row['symbol']] = \
                                row.get('name', '')
                except Exception as e:
                    logger.warning(f"ignition-shadow: class names "
                                   f"unavailable ({e})")
            a = underlying_anchor(symbol,
                                  self._class_names.get(symbol),
                                  self._class_map)
        except Exception as e:
            logger.warning(f"ignition-shadow: anchor({symbol}) failed: {e}")
            a = None
        self._anchors[symbol] = a
        return a

    # ------------------------------------------------------------------
    def on_mover(self, symbol: str, *, intraday_change_pct: float,
                 gap_pct: float, price: float,
                 has_news: Optional[bool],
                 bar_ts_utc: Optional[datetime] = None) -> None:
        """Scanner-thread entry point: ENQUEUE ONLY (microseconds).
        All evaluation/API/journal work happens on the shadow worker.
        NEVER raises; drops the sighting if the queue is full."""
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(
                (symbol, intraday_change_pct, gap_pct, price, has_news,
                 bar_ts_utc, datetime.now(timezone.utc)))
        except Exception:
            pass   # full queue -> drop; shadow data loss only

    def _worker_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=2.0)
            except Exception:
                continue
            try:
                sym, chg, gap, price, hn, bts, seen_at = item
                self._eval(sym, chg, gap, price, hn, bts, seen_at)
            except Exception as e:
                logger.warning(f"ignition-shadow: worker eval failed: {e}")
            finally:
                self._queue.task_done()

    def drain(self, timeout_s: float = 10.0) -> bool:
        """Block until every enqueued sighting has been processed.
        Tests/ops only — never called from the scanner thread."""
        import time as _t
        deadline = _t.monotonic() + timeout_s
        while _t.monotonic() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            _t.sleep(0.02)
        return False

    def _eval(self, symbol, chg, gap, price, has_news, bar_ts_utc,
              seen_at=None):
        now = seen_at or datetime.now(timezone.utc)
        try:
            from zoneinfo import ZoneInfo
            et = now.astimezone(ZoneInfo('America/New_York'))
        except Exception:
            return
        day = et.strftime('%Y-%m-%d')
        self._roll_day(day)
        minute = et.hour * 60 + et.minute
        if chg < self.trigger_pct or minute > self.max_trigger_min \
                or minute < 575:
            return
        # cohort updates on EVERY sighting (siblings confirm each other)
        a = self._anchor(symbol)
        first_sight = symbol not in self._seen_today
        if first_sight:
            self._seen_today.add(symbol)
            if a:
                self._day_anchor_counts[a] = \
                    self._day_anchor_counts.get(a, 0) + 1
        # late complex confirmation: a prior no-catalyst skip whose anchor
        # cohort just reached threshold re-fires (the 7/06 IREN-trio shape)
        for wsym, wrec in list(self._await_confirm.items()):
            wa = wrec.get('anchor')
            if wa and self._day_anchor_counts.get(wa, 0) >= self.min_cohort:
                del self._await_confirm[wsym]
                self._finalize(wrec, catalyst_kind=f'complex_late({wa})')
        if not first_sight:
            return
        rec = {'ts_utc': now.isoformat(), 'symbol': symbol, 'day': day,
               'minute_et': minute, 'intraday_change_pct': round(chg, 2),
               'gap_pct': round(gap, 2), 'price': price,
               'has_news': has_news, 'anchor': a,
               'anchor_cohort': self._day_anchor_counts.get(a or '', 0)}
        # detection latency: scanner sighting time vs the bar that made
        # the move — the S1 pass bar (p90 <= 90s) reads this field
        if bar_ts_utc is not None:
            try:
                rec['latency_s'] = round(
                    (now - bar_ts_utc).total_seconds(), 1)
            except Exception as e:
                logger.warning(
                    f"ignition-shadow: latency calc failed ({e})")
        # NOTE: no gap gate here — the scanner's gap_pct is current-vs-
        # prev-close (>=10 for every mover by construction). The TRUE
        # open-gap gate runs in _finalize from the day's own bars.
        if self._evals_today >= self.max_evals_per_day:
            rec['verdict'] = 'skip_eval_cap'
            return self._journal(rec)
        self._evals_today += 1
        catalyst = (has_news is True) or (
            a is not None
            and self._day_anchor_counts.get(a, 0) >= self.min_cohort)
        if not catalyst:
            # quote-only journal (S1 spread data) + park for late confirm
            self._capture_quote(rec)
            rec['verdict'] = 'skip_no_catalyst'
            self._await_confirm[symbol] = rec
            return self._journal(rec)
        self._finalize(rec, catalyst_kind='news' if has_news
                       else f'complex({a})')

    def _bounded(self, fn, timeout_s: float = 6.0):
        """Run an API call with a hard 6s bound (thread-join): a hanging
        data endpoint must cost the scanner cycle seconds, never the
        90s×2 client default (the news-fetch lesson, 2026-07-13)."""
        import threading
        out = {}
        def _run():
            try:
                out['v'] = fn()
            except Exception as e:
                out['e'] = e
        th = threading.Thread(target=_run, daemon=True)
        th.start(); th.join(timeout_s)
        if 'e' in out:
            raise out['e']
        if not th.is_alive():
            return out.get('v')
        raise TimeoutError(f'bounded call exceeded {timeout_s}s')

    def _capture_quote(self, rec: dict) -> None:
        try:
            q = self._bounded(
                lambda: self.alpaca.get_latest_quote(rec['symbol'])) or {}
            rec['bid'] = q.get('bid_price'); rec['ask'] = q.get('ask_price')
            rec['bid_size'] = q.get('bid_size')
            rec['ask_size'] = q.get('ask_size')
            if q.get('bid_price') and q.get('ask_price'):
                rec['spread_bps'] = round(
                    (q['ask_price'] - q['bid_price'])
                    / q['ask_price'] * 1e4, 1)
        except Exception as e:
            rec['quote_error'] = str(e)[:80]

    def _finalize(self, rec: dict, catalyst_kind: str) -> None:
        """Full evaluation of a catalyst-confirmed trigger: bars for R,
        fresh NBBO, hypothetical trade. Journal-only."""
        symbol = rec['symbol']
        now = datetime.now(timezone.utc)
        rec = dict(rec)
        rec['ts_final_utc'] = now.isoformat()
        rec['catalyst'] = catalyst_kind
        # full-day bars: recompute the TRUE harness gates (chg-from-OPEN,
        # OPEN gap) — the scanner's hot-loop values are approximations
        # (its intraday_change is max(vs-prev-close, range), not vs-open)
        r_pct = None
        gates_ok = True
        try:
            import time as _t
            _t0 = _t.monotonic()
            mins_since_open = max(rec['minute_et'] - 570 + 2, 35)
            bars = self._bounded(lambda: self.alpaca.get_1min_bars(
                symbol, lookback_minutes=int(mins_since_open)))
            rec['bars_fetch_s'] = round(_t.monotonic() - _t0, 2)
            if bars is not None and len(bars) >= 10:
                day_open = float(bars.iloc[0]['open'])
                rec['day_open'] = day_open
                rec['chg_from_open'] = round(
                    (rec['price'] / day_open - 1) * 100, 2)
                # derive prev_close from the scanner's gap semantics to
                # compute the TRUE open gap (ORB-disjointness gate)
                prev_close = rec['price'] / (1 + rec['gap_pct'] / 100.0)
                rec['open_gap_pct'] = round(
                    (day_open / prev_close - 1) * 100, 2)
                if rec['open_gap_pct'] >= self.gap_max:
                    rec['verdict'] = 'skip_gap_orb_territory'
                    gates_ok = False
                elif rec['chg_from_open'] < self.trigger_pct:
                    rec['verdict'] = 'skip_true_chg_below_trigger'
                    gates_ok = False
                pre = bars.tail(31)
                stop = float(pre['low'].min())
                r_pct = max((rec['price'] - stop) / rec['price'] * 100.0,
                            1.0)
        except Exception as e:
            rec['bars_error'] = str(e)[:80]
        rec['r_pct'] = round(r_pct, 2) if r_pct else None
        self._capture_quote(rec)
        if not gates_ok:
            return self._journal(rec)
        if r_pct is None:
            rec['verdict'] = 'no_bars'
        elif r_pct < self.min_r_pct:
            rec['verdict'] = 'skip_r_too_small'
        else:
            pos = min(MODEL_RISK_USD / (r_pct / 100.0), POS_CAP_USD)
            rec['verdict'] = 'SHADOW_TRIGGER'
            rec['hypo_entry'] = rec.get('ask') or rec['price']
            rec['hypo_stop'] = round(
                rec['price'] * (1 - r_pct / 100.0), 4)
            rec['hypo_position_usd'] = round(pos, 0)
            logger.info(
                f"[IGNITION-SHADOW] TRIGGER {symbol} "
                f"+{rec['intraday_change_pct']:.1f}% R={r_pct:.1f}% "
                f"catalyst={catalyst_kind} "
                f"spread={rec.get('spread_bps', '?')}bps "
                f"pos=${rec.get('hypo_position_usd', 0):,.0f} "
                f"(journal-only, no order)")
        self._journal(rec)

    def _journal(self, rec: dict) -> None:
        try:
            path = os.path.join(self._log_dir,
                                f"ignition_shadow_{rec['day']}.jsonl")
            with open(path, 'a') as fh:
                fh.write(json.dumps(rec, default=str) + '\n')
        except Exception as e:
            logger.warning(f"ignition-shadow: journal write failed: {e}")
