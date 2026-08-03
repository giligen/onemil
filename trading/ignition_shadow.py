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

import pandas as pd

from trading import ignition_rules as _rules

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
        self._news_cache: Dict[str, Optional[bool]] = {}
        self._dropped = 0
        self._day: Optional[str] = None
        self._log_dir = log_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'logs')
        self._class_map = None
        self._class_names: Dict[str, str] = {}
        # eval cap 150 (60 was exhausted on the hot 7/30 tape: 131
        # sightings, 71 capped). Post gate-reorder evals are cheap (bars
        # fetch only; news runs solely for structure-passers), so the
        # cap is a runaway backstop, not a budget.
        # (history: 25 starved on 7/20, 60 on 7/30)
        self.max_evals_per_day = int(cfg.get('max_evals_per_day', 150))
        self._evals_today = 0
        # no-catalyst skips wait for late complex confirmation
        self._await_confirm: Dict[str, dict] = {}
        # level-not-crossed-yet parks awaiting re-evaluation (8/3 fix)
        self._await_level: Dict[str, dict] = {}
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
            self._news_cache = {}
            self._evals_today = 0
            self._await_confirm = {}
            self._await_level = {}

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
                 price_ts_utc: Optional[datetime] = None) -> None:
        """Scanner-thread entry point: ENQUEUE ONLY (microseconds).
        All evaluation/API/journal work happens on the shadow worker.
        NEVER raises; drops the sighting if the queue is full."""
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(
                (symbol, intraday_change_pct, gap_pct, price, has_news,
                 price_ts_utc, datetime.now(timezone.utc)))
        except Exception:
            # full queue -> drop (shadow data loss only, never scanner
            # time); throttled so a wedged worker can't spam the log
            self._dropped += 1
            if self._dropped == 1 or self._dropped % 100 == 0:
                logger.warning(
                    f"ignition-shadow: queue full — {self._dropped} "
                    f"sighting(s) dropped (worker slow or wedged)")

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

    def _eval(self, symbol, chg, gap, price, has_news, price_ts_utc,
              seen_at=None):
        now = seen_at or datetime.now(timezone.utc)
        try:
            from zoneinfo import ZoneInfo
            et = now.astimezone(ZoneInfo('America/New_York'))
        except Exception as e:
            logger.warning(f"ignition-shadow: ET conversion failed ({e}) "
                           f"— sighting dropped")
            return
        day = et.strftime('%Y-%m-%d')
        self._roll_day(day)
        minute = et.hour * 60 + et.minute
        if chg < self.trigger_pct or minute > self.max_trigger_min \
                or minute < 575:
            return
        # rough pre-floor on sighting price (cheap junk cut; the BT-
        # parity floor on DAY OPEN runs in _finalize via ignition_rules)
        if price < _rules.PRICE_FLOOR:
            return
        a = self._anchor(symbol)
        first_sight = symbol not in self._seen_today
        if first_sight:
            self._seen_today.add(symbol)
        if not first_sight:
            # RE-EVALUATION of level-parked symbols (8/3 finding: the BT
            # trigger window is CONTINUOUS 9:35-10:30 but the shadow
            # sampled each symbol ONCE at first sighting — NBIG/NBIL/
            # IREX were sighted before their +10% level cross, parked
            # skip_level_not_crossed, and never looked at again: $9K of
            # BT profit invisible to the shadow. The scanner re-sights
            # every cycle while a mover holds; a level-parked symbol
            # re-finalizes on each sighting until cross or cutoff.
            if (symbol in self._await_level
                    and minute <= self.max_trigger_min
                    and self._evals_today < self.max_evals_per_day):
                wrec = dict(self._await_level.pop(symbol))
                wrec['minute_et'] = minute
                wrec['price'] = price
                wrec['intraday_change_pct'] = round(chg, 2)
                self._evals_today += 1
                self._finalize(wrec, catalyst_kind=None)
            return
        rec = {'ts_utc': now.isoformat(), 'symbol': symbol, 'day': day,
               'minute_et': minute, 'intraday_change_pct': round(chg, 2),
               'gap_pct': round(gap, 2), 'price': price,
               'has_news': has_news, 'anchor': a,
               'anchor_cohort': self._day_anchor_counts.get(a or '', 0)}
        # detection latency: scanner sighting time vs the latest-trade
        # timestamp that showed the move (NOT a bar-window start — the
        # scanner's bars are 15-min) — the S1 pass bar (p90<=90s) reads
        # this field
        if price_ts_utc is not None:
            try:
                rec['latency_s'] = round(
                    (now - price_ts_utc).total_seconds(), 1)
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
        # GATE ORDER (2026-07-24 parity refactor): STRUCTURE first
        # (bars — floor on day open, open gap, level-crossed, R, chase,
        # participation), news/catalyst LAST inside _finalize. Pre-24
        # the news fetch ran before any bar work, so the 9:35 junk
        # flood burned the eval budget resolving news for symbols
        # structure would reject — the starvation that cost ONDL/ONDG
        # (7/21) and NBIG (7/23) vs the BT replay.
        self._finalize(rec, catalyst_kind=None)

    def _resolve_news(self, symbol: str,
                      rec: dict) -> Optional[bool]:
        """Own-ticker news, worker-thread only. Same helper + window as
        the ORB news gate (prev-day 15:00 ET -> fetch time, 8s bound,
        no retries); at an Ignition trigger this additionally counts
        same-morning breaking news — for an igniter that IS the
        catalyst. Tri-state: True/False known; None = fetch failed,
        which degrades THIS symbol to complex-confirmation only
        (fail-open on the catalyst it can still prove)."""
        if symbol in self._news_cache:
            return self._news_cache[symbol]
        try:
            res = self._bounded(
                lambda: self.alpaca.get_premarket_news_multi([symbol]),
                8.0) or {}
            info = res.get(symbol) or {}
            hn: Optional[bool] = bool(info.get('n_articles', 0))
            if hn:
                rec['news_headline'] = str(info.get('headline', ''))[:120]
        except Exception as e:
            logger.warning(f"ignition-shadow: news({symbol}) failed: {e} "
                           f"— complex-confirmation only for this symbol")
            rec['news_error'] = str(e)[:80]
            hn = None
        self._news_cache[symbol] = hn
        return hn

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

    def _finalize(self, rec: dict, catalyst_kind: Optional[str]) -> None:
        """Full evaluation via the SHARED rules (trading/ignition_rules —
        2026-07-24 parity refactor): structure gates first from bars,
        catalyst LAST. catalyst_kind is None on the first pass (resolved
        here); a string only for complex_late re-fires of parked recs
        (structure already passed for those). Journal-only."""
        symbol = rec['symbol']
        now = datetime.now(timezone.utc)
        rec = dict(rec)
        rec['ts_final_utc'] = now.isoformat()
        rec['catalyst'] = catalyst_kind
        # actionable minute (== sighting minute except for complex_late
        # confirmations) — the resim must key exits to THIS, and the
        # bars lookback must be computed from NOW, not the stale
        # sighting minute (else a late-confirm fetch window starts
        # mid-morning and iloc[0] is not the 9:30 open)
        try:
            from zoneinfo import ZoneInfo
            _et_now = now.astimezone(ZoneInfo('America/New_York'))
            minute_now = _et_now.hour * 60 + _et_now.minute
        except Exception:
            minute_now = rec['minute_et']
        rec['minute_final_et'] = minute_now
        # full-day bars: recompute the TRUE harness gates (chg-from-OPEN,
        # OPEN gap) — the scanner's hot-loop values are approximations
        # (its intraday_change is max(vs-prev-close, range), not vs-open)
        r_pct = None
        gates_ok = True
        try:
            import time as _t
            _t0 = _t.monotonic()
            # over-asking is free: the client clamps the window start to
            # the 9:30 open, so this guarantees iloc[0] IS the open bar
            mins_since_open = max(minute_now - 570 + 5, 40)
            bars = self._bounded(lambda: self.alpaca.get_1min_bars(
                symbol, lookback_minutes=int(mins_since_open)))
            rec['bars_fetch_s'] = round(_t.monotonic() - _t0, 2)
            # >=6 bars matches the harness minimum (5-bar pre-window +
            # trigger bar); 10 was over-strict and turned every 9:35-39
            # early trigger into no_bars before the gap gate could even
            # classify it (live 7/20 finding: 15 such records)
            if bars is not None and len(bars) >= 6:
                day_open = float(bars.iloc[0]['open'])
                rec['day_open'] = day_open
                rec['chg_from_open'] = round(
                    (rec['price'] / day_open - 1) * 100, 2)
                # derive prev_close from the scanner's gap semantics to
                # compute the TRUE open gap (ORB-disjointness gate)
                prev_close = rec['price'] / (1 + rec['gap_pct'] / 100.0)
                rec['open_gap_pct'] = round(
                    (day_open / prev_close - 1) * 100, 2)
                urej = _rules.universe_reject(day_open, prev_close)
                if urej:
                    rec['verdict'] = urej
                    gates_ok = False
                else:
                    # BOOK trigger semantics: the +10% LEVEL was crossed
                    # by a bar HIGH in the window — NOT price-at-sighting
                    # (a pullback after the cross still counts; the 7/21
                    # BIYA parity miss)
                    try:
                        _ts = pd.to_datetime(bars['timestamp'], utc=True)
                        _m = (_ts.dt.tz_convert('America/New_York')
                              .dt.hour * 60
                              + _ts.dt.tz_convert('America/New_York')
                              .dt.minute)
                        win = bars[( _m >= _rules.TRIGGER_MIN_START)
                                   & (_m <= _rules.TRIGGER_MIN_END)]
                        crossed = _rules.level_crossed(
                            win['high'].tolist(), day_open)
                    except Exception as e:
                        logger.warning(f"ignition-shadow: level check "
                                       f"failed for {symbol}: {e}")
                        crossed = rec['chg_from_open'] >= self.trigger_pct
                    if not crossed:
                        rec['verdict'] = 'skip_level_not_crossed'
                        gates_ok = False
                        # park for re-evaluation on later sightings —
                        # the cross may come any time before 10:30
                        self._await_level[symbol] = rec
                pre = bars.tail(31)
                stop = _rules.stop_from_pre_lows(
                    float(pre['low'].min()), rec['price'])
                r_pct = max(_rules.r_pct_from_stop(rec['price'], stop),
                            1.0)
        except Exception as e:
            rec['bars_error'] = str(e)[:80]
        rec['r_pct'] = round(r_pct, 2) if r_pct else None
        self._capture_quote(rec)
        if not gates_ok:
            return self._journal(rec)
        # BT chase guard (2026-07-24 audit: never ported from the
        # latency-honest harness — 4/16 week-1 triggers were entries the
        # BT refuses): entry must not exceed 5% past the +10% level,
        # i.e. ask <= day_open * 1.10 * 1.05.
        _ask = rec.get('ask') or rec['price']
        _open = rec.get('day_open')
        if _open and _ask > _open * 1.10 * 1.05:
            rec['verdict'] = 'skip_chase_guard'
            rec['chase_ratio'] = round(_ask / _open, 4)
            return self._journal(rec)
        if r_pct is None:
            rec['verdict'] = 'no_bars'
            return self._journal(rec)
        if r_pct < self.min_r_pct:
            rec['verdict'] = 'skip_r_too_small'
            return self._journal(rec)
        # sizing via shared rules (participation-capped; the book's live
        # proxy for its EOD day-dollar universe gate)
        try:
            lb = bars.iloc[-1]
            bar_dollar = float(lb['volume']) * float(lb['close'])
        except Exception as e:
            logger.warning(f"ignition-shadow: bar dollar failed for "
                           f"{symbol}: {e}")
            bar_dollar = 0.0
        pos = _rules.position_usd(r_pct, bar_dollar)
        rec['participation_cap_usd'] = round(
            _rules.PARTICIPATION * bar_dollar, 0)
        if _rules.position_reject(pos):
            rec['verdict'] = 'skip_illiquid'
            rec['hypo_position_usd'] = round(pos, 0)
            return self._journal(rec)
        # STRUCTURE PASSED — this symbol counts toward the day's
        # TRIGGER cohort (book `uc` semantics: cohort over candidate
        # triggers, NOT raw sightings — 7/24 parity audit: sighting-
        # cohort wrongly confirmed APLX/CIFU)
        a = rec.get('anchor')
        if catalyst_kind is None and a:
            self._day_anchor_counts[a] = \
                self._day_anchor_counts.get(a, 0) + 1
        rec['anchor_cohort'] = self._day_anchor_counts.get(a or '', 0)
        # late complex confirmation: parked structure-passed recs whose
        # anchor cohort just reached threshold re-fire
        for wsym, wrec in list(self._await_confirm.items()):
            wa = wrec.get('anchor')
            if wa and wsym != symbol and \
                    self._day_anchor_counts.get(wa, 0) >= self.min_cohort:
                del self._await_confirm[wsym]
                self._trigger(wrec, f'complex_late({wa})',
                              wrec['r_pct'], wrec['hypo_position_usd'])
        # catalyst LAST (news fetch only for structure-passed symbols)
        if catalyst_kind is None:
            has_news = rec.get('has_news')
            if has_news is None:
                has_news = self._resolve_news(symbol, rec)
                rec['has_news'] = has_news
            if not _rules.catalyst_confirmed(
                    has_news, a, self._day_anchor_counts.get(a or '', 0)):
                rec['verdict'] = 'skip_no_catalyst'
                rec['r_pct'] = round(r_pct, 2)
                rec['hypo_position_usd'] = round(pos, 0)
                self._await_confirm[symbol] = rec
                return self._journal(rec)
            catalyst_kind = 'news' if has_news is True else f'complex({a})'
        self._trigger(rec, catalyst_kind, r_pct, pos)

    def _trigger(self, rec: dict, catalyst_kind: str, r_pct: float,
                 pos: float) -> None:
        rec = dict(rec)
        rec['catalyst'] = catalyst_kind
        rec['verdict'] = 'SHADOW_TRIGGER'
        rec['hypo_entry'] = rec.get('ask') or rec['price']
        rec['hypo_stop'] = round(rec['price'] * (1 - r_pct / 100.0), 4)
        rec['hypo_position_usd'] = round(pos, 0)
        logger.info(
            f"[IGNITION-SHADOW] TRIGGER {rec['symbol']} "
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
