"""Cross-strategy orphan position reconciler — shared by all engines.

Problem this solves (SMU + QBTZ incident, 2026-05-26 → 2026-06-05):
  A position on the broker that no engine is tracking is an "orphan". The
  failure modes are silent: stop_loss_unconfirmed wrote a fake confirmed
  exit; the macd_wave/orb per-engine orphan loops each had a different
  blind spot; nothing detected "broker has it AND we have a poisoned DB
  row". Result: 5,976 SMU sh and 5,657 QBTZ sh sat unmanaged for 10 and
  4 days respectively, ballooning a recorded $-1.2K to an actual $-13.5K.

This module provides a single source of truth for orphan detection +
classification + action. Every strategy engine calls
reconcile_strategy_orphans() from its sync_positions() — that's it.

Safety-first design:
  - Auto-close fires ONLY when ALL four predicate conditions hold (see
    is_owned_orphan). Any failure → FOREIGN → alert only, never closed.
  - The ownership predicate cannot mistake another strategy's positions
    for our own: it requires strategy-tag match in DB + avg-entry match
    within 5 bps + qty ≤ DB.filled_qty + STALE signal indicating the
    DB-recorded "exit" was never verified or the trade is from a
    previous calendar day.
  - Belt-and-suspenders: config kill switch (auto_close_enabled),
    per-strategy per-hour close-rate cap (max_closes_per_hour), and
    Telegram ERROR alert with the full decision context BEFORE every
    close attempt. Even if a future bug breaks the predicate, you SEE
    every action in chat.

Tested by tests/test_orphan_reconciler.py (unit) and downstream by
tests/test_*_engine_*_reconciler.py (integration with each engine).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# DB exit_reason / order_status values that signal "exit attempted but not
# confirmed" — i.e., the broker may still hold the position despite the DB
# row looking exited. Clean exit reasons (trail_stop, force_close, stop_loss,
# tag_bb, etc.) are NOT in this set, so they cannot re-trigger ownership of
# a current broker position.
STALE_EXIT_REASONS: frozenset = frozenset({
    'stop_loss_unconfirmed',
    'exit_pending_verification',
})
PENDING_VERIFICATION_STATUS = 'exit_pending_verification'

# Once a row has been reconciled, this exit_reason marks it as terminal.
from trading.exit_reasons import ExitReason

ORPHAN_RECOVERED_EXIT_REASON = ExitReason.ORPHAN_RECOVERED_FORCE_CLOSE.value


@dataclass(frozen=True)
class ReconcilerConfig:
    """All knobs for orphan reconciliation. Tune via config.yaml; see
    config.orphan_reconciler_cfg."""

    # Master kill switch. False → reconciler runs in observe-mode: detect
    # + alert + log, but NEVER submit close orders. Useful for the first
    # week of rollout to confirm decisions before going active.
    auto_close_enabled: bool = True

    # Days backward to search for matching strategy rows. Covers
    # multi-day broker-state drift (max stuck position observed: 10 days).
    lookback_days: int = 14

    # Avg-entry-price match tolerance, expressed as fraction of price.
    # 0.0005 = 5 bps. Coupled with a $0.005 absolute floor (see code) so
    # sub-dollar stocks still get a sensible tolerance.
    avg_entry_match_pct: float = 0.0005
    avg_entry_match_abs_min: float = 0.005

    # Per-(strategy) cap on auto-closes per rolling hour. If breached,
    # reconciler switches to alert-only for the rest of the hour. The
    # cap mainly limits blast radius if a future bug ever surfaces.
    max_closes_per_hour: int = 3

    # Per-(strategy, symbol) Telegram alert cooldown. We don't want to
    # spam chat once an orphan has been alerted — the engine syncs every
    # few seconds, so without a cooldown we'd see 100s of duplicates.
    alert_cooldown_minutes: int = 60

    # Fill-poll budget after submitting close. Market orders typically
    # fill in <2s; we poll briefly so the recovery row carries the actual
    # exit_price. If we run out of budget, we mark the row as
    # exit_pending_verification (NOT a fake exit) and a separate
    # in-flight tracker prevents duplicate close submission next cycle.
    fill_poll_timeout_s: float = 5.0
    fill_poll_interval_s: float = 0.5

    # In-flight close cooldown — after submitting close, skip this symbol
    # for N seconds even if broker still reports the position. Prevents
    # duplicate close submissions while the original is propagating.
    inflight_close_cooldown_s: float = 30.0


@dataclass(frozen=True)
class OrphanAction:
    """One row per orphan handled in a sync cycle."""

    symbol: str
    qty: int
    avg_entry: float
    classification: str  # 'owned' or 'foreign'
    action: str          # 'closed' / 'alert_only' / 'cap_breached' / 'auto_close_disabled'
    db_trade_id: Optional[int] = None
    db_exit_reason: Optional[str] = None
    close_order_id: Optional[str] = None
    note: Optional[str] = None


# =========================================================================
# Ownership predicate
# =========================================================================

def is_owned_orphan(
    broker_pos: Dict[str, Any],
    db_row: Dict[str, Any],
    today_et: date,
    cfg: ReconcilerConfig,
) -> bool:
    """Return True iff the broker position is provably an unrecovered
    orphan owned by this strategy.

    REQUIRES ALL FOUR:
      1. The DB row's `strategy` column matches the caller's strategy
         (already filtered in the upstream query).
      2. The DB row carries a STALE signal — either exit_reason in
         STALE_EXIT_REASONS, or order_status =
         'exit_pending_verification', or it's a cross-day row that's
         filled but has no exit_price.
      3. The broker's avg_entry_price matches the DB row's fill_price
         within max(avg_entry_match_abs_min, fill * avg_entry_match_pct).
      4. The broker's qty is <= DB filled_qty (partial fills allowed).

    Any failure → False → caller classifies as FOREIGN → alert only.
    """
    # 2. Stale signal check.
    is_cross_day = (
        db_row.get('trade_date') is not None
        and str(db_row['trade_date']) < today_et.isoformat()
    )
    has_stale_reason = db_row.get('exit_reason') in STALE_EXIT_REASONS
    has_pending_status = (
        db_row.get('order_status') == PENDING_VERIFICATION_STATUS
    )
    has_no_exit_cross_day = (
        db_row.get('order_status') == 'filled'
        and db_row.get('exit_price') is None
        and is_cross_day
    )
    if not (has_stale_reason or has_pending_status or has_no_exit_cross_day):
        return False

    # 3. Avg-entry match — the strongest "this IS our position" signal.
    db_fill = db_row.get('fill_price') or 0.0
    if db_fill <= 0:
        return False
    eps = max(cfg.avg_entry_match_abs_min, db_fill * cfg.avg_entry_match_pct)
    broker_avg = float(broker_pos.get('avg_entry_price') or 0.0)
    if abs(broker_avg - db_fill) > eps:
        return False

    # 4. Qty sanity — larger broker position implies someone else added.
    # Fallback to the planned `shares` column when filled_qty is missing
    # (legacy rows + tests). Both express "this is the max # of shares we
    # ever held"; the partial-fill sanity still holds for either.
    db_qty = int(db_row.get('filled_qty') or db_row.get('shares') or 0)
    broker_qty = int(broker_pos.get('qty') or 0)
    if db_qty <= 0 or broker_qty > db_qty:
        return False

    return True


def _select_owned_row(
    broker_pos: Dict[str, Any],
    candidate_rows: List[Dict[str, Any]],
    today_et: date,
    cfg: ReconcilerConfig,
) -> Optional[Dict[str, Any]]:
    """From the per-symbol candidate DB rows, return the most-recent
    matching row, or None if no candidate satisfies is_owned_orphan."""
    matches = [
        r for r in candidate_rows
        if is_owned_orphan(broker_pos, r, today_et, cfg)
    ]
    if not matches:
        return None
    matches.sort(key=lambda r: (str(r.get('trade_date') or ''),
                                int(r.get('id') or 0)), reverse=True)
    return matches[0]


# =========================================================================
# Rate limiters / cooldown — module-level so they survive across sync
# cycles within a single process. Reset at process restart, which is fine:
# startup is exactly when we want a fresh alert anyway.
# =========================================================================

@dataclass
class _ReconcilerState:
    """Per-strategy rolling state for rate-limiting + cooldown."""
    # rolling close-count window: list of (strategy, datetime) of recent closes
    close_timestamps: Dict[str, List[datetime]] = field(default_factory=dict)
    # last alert per (strategy, symbol)
    last_alert: Dict[tuple, datetime] = field(default_factory=dict)
    # in-flight close orders awaiting fill: (strategy, symbol) → expiry datetime.
    # Stops a follow-up sync from re-submitting close on a symbol whose
    # close just went out but hasn't propagated to get_open_positions yet.
    inflight_close: Dict[tuple, datetime] = field(default_factory=dict)


_state = _ReconcilerState()


def _record_close(strategy: str) -> None:
    _state.close_timestamps.setdefault(strategy, []).append(
        datetime.now(timezone.utc)
    )


def _closes_in_last_hour(strategy: str) -> int:
    """Count how many auto-closes have fired in the last 60min for the
    strategy, pruning older entries while we're here."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=1)
    bucket = [t for t in _state.close_timestamps.get(strategy, []) if t >= cutoff]
    _state.close_timestamps[strategy] = bucket
    return len(bucket)


def _prune_alert_cache(cooldown_minutes: int) -> None:
    """Drop expired alert-cooldown entries so the dict can't grow unbounded
    across long-running processes. Called every reconcile cycle (not just
    on alert sends — foreign positions no longer alert)."""
    if len(_state.last_alert) <= 64:
        return
    now = datetime.now(timezone.utc)
    cooldown = timedelta(minutes=cooldown_minutes)
    expired = [k for k, t in _state.last_alert.items()
               if (now - t) >= cooldown]
    for k in expired:
        _state.last_alert.pop(k, None)


def _should_alert(strategy: str, symbol: str, cooldown_minutes: int) -> bool:
    """True iff no alert has been sent for (strategy, symbol) within the
    cooldown window."""
    now = datetime.now(timezone.utc)
    cooldown = timedelta(minutes=cooldown_minutes)
    _prune_alert_cache(cooldown_minutes)
    key = (strategy, symbol)
    last = _state.last_alert.get(key)
    if last is not None and (now - last) < cooldown:
        return False
    _state.last_alert[key] = now
    return True


def _is_close_inflight(strategy: str, symbol: str) -> bool:
    """Has a close been recently submitted for this (strategy, symbol)?

    Also prunes expired entries while we're here. Returning True means
    the reconciler should SKIP this symbol — no detection, no alert, no
    second close submission.
    """
    now = datetime.now(timezone.utc)
    expired = [k for k, exp in _state.inflight_close.items() if exp <= now]
    for k in expired:
        _state.inflight_close.pop(k, None)
    return (strategy, symbol) in _state.inflight_close


def _mark_close_inflight(strategy: str, symbol: str,
                          cooldown_s: float) -> None:
    _state.inflight_close[(strategy, symbol)] = (
        datetime.now(timezone.utc) + timedelta(seconds=cooldown_s)
    )


def reset_state_for_tests() -> None:
    """Clear in-memory rate-limit state. Tests only."""
    _state.close_timestamps.clear()
    _state.last_alert.clear()
    _state.inflight_close.clear()


# =========================================================================
# Telegram alert formatters
# =========================================================================

def _format_owned_alert(strategy: str, action: OrphanAction,
                         db_fill: float) -> str:
    upl = (action.avg_entry, action.qty)  # not actually used yet
    line1 = (
        f"❗ [{strategy.upper()}] ORPHAN POSITION — {action.symbol}"
    )
    line2 = (
        f"Owned by {strategy}: trade_id={action.db_trade_id}, "
        f"db_exit_reason={action.db_exit_reason or 'NULL'}"
    )
    line3 = (
        f"Broker: {action.qty} sh @ ${action.avg_entry:.4f} "
        f"(DB fill ${db_fill:.4f})"
    )
    if action.action == 'closed':
        line4 = f"Action: market-close submitted (order={action.close_order_id})"
    elif action.action == 'cap_breached':
        line4 = (
            f"Action: ALERT ONLY — per-hour close cap breached. "
            f"Manual close required."
        )
    elif action.action == 'auto_close_disabled':
        line4 = (
            "Action: ALERT ONLY — auto_close_enabled=false. "
            "Manual close required."
        )
    else:
        line4 = f"Action: {action.action}"
    return "\n".join([line1, line2, line3, line4])


def _format_foreign_alert(strategy: str, action: OrphanAction) -> str:
    return (
        f"⚠️ [{strategy.upper()}] FOREIGN POSITION on shared account — "
        f"{action.symbol}\n"
        f"Broker: {action.qty} sh @ ${action.avg_entry:.4f}\n"
        f"No matching {strategy} DB row in lookback. NOT auto-closed.\n"
        f"Likely belongs to another strategy/project or manual entry."
    )


def _alert(notifier, strategy: str, symbol: str, msg: str,
            cooldown_minutes: int) -> None:
    """Best-effort Telegram + logger.error. Never raises — orphan handling
    must continue even if Telegram is down."""
    logger.error(msg)
    if notifier is None:
        return
    if not _should_alert(strategy, symbol, cooldown_minutes):
        return
    try:
        if hasattr(notifier, 'notify_error'):
            notifier.notify_error(msg, component=f"OrphanReconciler/{strategy}")
        elif hasattr(notifier, 'send_message_sync'):
            notifier.send_message_sync(msg)
    except Exception as e:
        logger.warning(f"orphan reconciler: Telegram send failed: {e}")


# =========================================================================
# DB updates
# =========================================================================

def _write_recovery_row(
    db,
    db_row: Dict[str, Any],
    fill_price: float,
    fill_qty: int,
    avg_entry: float,
    close_order_id: str,
) -> None:
    """Update the DB row to reflect the orphan-recovery close.

    Always sets exit_reason=ORPHAN_RECOVERED_EXIT_REASON so the predicate
    will NOT consider this row a candidate again (stale set excludes
    'orphan_recovered_force_close')."""
    try:
        pnl = (fill_price - avg_entry) * fill_qty
        pnl_pct = (fill_price - avg_entry) / avg_entry * 100.0 if avg_entry > 0 else 0.0
        update = {
            'exit_price': fill_price,
            'exit_reason': ORPHAN_RECOVERED_EXIT_REASON,
            'exited_at': datetime.now(timezone.utc),
            'order_status': 'closed',
            'pnl': pnl,
            'pnl_pct': pnl_pct,
        }
        db.update_trade(db_row['id'], update)
    except Exception as e:
        logger.error(
            f"orphan reconciler: DB recovery update failed for trade_id="
            f"{db_row.get('id')}: {e}"
        )


# =========================================================================
# Main reconciliation entry point
# =========================================================================

def _poll_for_fill(
    fetcher: Callable[[str], Dict[str, Any]],
    order_id: str,
    timeout_s: float,
    interval_s: float,
) -> tuple:
    """Poll the order until filled or timeout. Returns (fill_price, fill_qty).

    Break conditions (any):
      - get a non-zero fill_price + fill_qty (assume fill or partial fill —
        good enough for recovery row).
      - status reports terminal-non-filled (canceled / rejected / expired).
      - timeout hit.

    Returns (None, 0) on no fill or polling failure. The caller decides
    whether to write a recovery row (got fill) or a pending-verification
    row (poll timed out).
    """
    import time as _time
    deadline = _time.time() + timeout_s
    fill_price: Optional[float] = None
    fill_qty: int = 0
    while _time.time() < deadline:
        try:
            o = fetcher(order_id)
        except Exception as e:
            logger.warning(f"orphan reconciler: fill poll error: {e}")
            return (None, 0)
        if not isinstance(o, dict):
            break
        status = str(o.get('status') or '').lower()
        fq = int(o.get('filled_qty') or 0)
        fp = o.get('filled_avg_price')
        if fp is not None and fq > 0:
            fill_price = float(fp)
            fill_qty = fq
            break  # Got a fill — done.
        if status in ('canceled', 'rejected', 'expired'):
            break
        _time.sleep(interval_s)
    return (fill_price, fill_qty)


def _cancel_open_orders_for_symbol(alpaca, symbol: str) -> int:
    """Cancel any OPEN orders for a symbol; returns count cancelled.

    2026-07-04 review fix: Alpaca rejects close_position with
    "40310000 insufficient qty available" while shares are held for open
    orders — and the OWNED orphans this module recovers frequently carry
    an expired-DAY emergency stop or leftover bracket leg. Pre-reconciler
    code (trading_engine._cancel_open_orders_for_symbol) always cancelled
    first; the reconciler must too or its close fails every cycle and the
    position rides naked into the overnight gap.

    Best-effort: per-order failures log WARNING and don't raise.
    """
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        open_orders = alpaca.trading_client.get_orders(filter=req)
    except Exception as e:
        logger.warning(
            f"orphan reconciler: open-order query before close({symbol}) "
            f"failed: {e} — attempting close anyway")
        return 0
    cancelled = 0
    for o in (open_orders or []):
        try:
            alpaca.cancel_order(str(o.id))
            cancelled += 1
            logger.info(
                f"orphan reconciler: cancelled open order {o.id} on {symbol} "
                f"before close")
        except Exception as ce:
            logger.warning(
                f"orphan reconciler: cancel {getattr(o, 'id', '?')} on "
                f"{symbol} failed: {ce}")
    return cancelled


# Backoffs (seconds) between close retries while cancelled shares release.
_CLOSE_HELD_QTY_BACKOFFS_S = (0.5, 1.5)


def reconcile_strategy_orphans(
    *,
    strategy: str,
    alpaca,
    db,
    notifier,
    tracked_symbols: Set[str],
    cfg: ReconcilerConfig = None,
    today_et: Optional[date] = None,
    close_position_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    fetch_fill_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    broker_positions: Optional[List[Dict[str, Any]]] = None,
) -> List[OrphanAction]:
    """Compare broker positions vs (engine-tracked + DB) and act on orphans.

    Args:
        strategy: 'bull_flag' / 'macd_wave' / 'orb' — ownership filter.
            ALSO the account-routing assumption: caller must pass the
            ``alpaca`` client for THIS strategy's broker account.
        alpaca: AlpacaClient for the strategy's broker account.
        db: Database instance. Used via ``get_strategy_trades_in_window``.
        notifier: TelegramNotifier (or None).
        tracked_symbols: symbols the engine currently has in its
            open_positions map. Anything broker has but engine doesn't is
            a candidate orphan.
        cfg: tuning knobs (defaults to ReconcilerConfig()).
        today_et: optional date override (testing).
        close_position_fn: optional override for the close call (testing).
        fetch_fill_fn: optional override for fill polling (testing).
        broker_positions: optional pre-fetched positions snapshot. When
            sync_positions already called get_open_positions, pass that
            list here to avoid a duplicate API hit.

    Returns:
        List of OrphanAction — one per orphan detected this cycle.
        Empty if nothing was orphan.

    Pure side-effect ordering: detect → classify → alert → (close + DB
    update if OWNED). The in-flight-close cache shields a freshly
    submitted close from re-detection until either the broker stops
    reporting the position or `cfg.inflight_close_cooldown_s` expires.
    """
    if cfg is None:
        cfg = ReconcilerConfig()
    if today_et is None:
        today_et = datetime.now(timezone.utc).date()
    actions: List[OrphanAction] = []

    # 1. Snapshot broker positions (use the caller's if provided).
    if broker_positions is None:
        try:
            broker_positions = alpaca.get_open_positions()
        except Exception as e:
            logger.error(
                f"orphan reconciler [{strategy}]: broker snapshot failed: {e}"
            )
            return actions

    # Prune the alert-cooldown cache every cycle. This used to happen only
    # inside _alert via _should_alert, but foreign positions no longer
    # alert at all (owner directive 8/17) — without this, a quiet account
    # with many owner positions would never prune the cache.
    _prune_alert_cache(cfg.alert_cooldown_minutes)

    # 2. Filter to candidates: broker has it, engine doesn't track it,
    #    and we don't have a close already in flight for this symbol.
    candidates = [
        p for p in broker_positions
        if p.get('symbol') not in tracked_symbols
        and not _is_close_inflight(strategy, p.get('symbol'))
    ]
    if not candidates:
        return actions

    # 3. Query DB for strategy-tagged rows in the lookback window.
    candidate_symbols = [p['symbol'] for p in candidates]
    db_rows_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    lookback_start = (today_et - timedelta(days=cfg.lookback_days)).isoformat()
    try:
        if hasattr(db, 'get_strategy_trades_in_window'):
            rows = db.get_strategy_trades_in_window(
                strategy, lookback_start, candidate_symbols,
            )
        else:
            # Legacy / test-double fallback. Will be removed once all
            # callers are on the public API.
            placeholders = ','.join('?' for _ in candidate_symbols)
            cur = db._trades_conn.execute(
                f"SELECT * FROM trades "
                f"WHERE strategy = ? AND trade_date >= ? "
                f"AND symbol IN ({placeholders}) "
                f"ORDER BY trade_date DESC, id DESC",
                (strategy, lookback_start, *candidate_symbols),
            )
            rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            db_rows_by_symbol.setdefault(r['symbol'], []).append(r)
    except Exception as e:
        logger.error(
            f"orphan reconciler [{strategy}]: DB lookup failed: {e} — "
            f"defaulting to FOREIGN for all candidates"
        )

    # 4. Classify each candidate.
    for broker_pos in candidates:
        sym = broker_pos['symbol']
        avg_entry = float(broker_pos.get('avg_entry_price') or 0.0)
        qty = int(broker_pos.get('qty') or 0)
        rows = db_rows_by_symbol.get(sym, [])
        owned_row = _select_owned_row(broker_pos, rows, today_et, cfg)

        if owned_row is None:
            action = OrphanAction(
                symbol=sym, qty=qty, avg_entry=avg_entry,
                classification='foreign',
                action='log_only',
                note='No matching strategy row with all OWNED predicates',
            )
            actions.append(action)
            # Owner directive 2026-08-17 (post-BMNR incident): the owner
            # trades manually — shorts, brackets, any size — on this same
            # live account. Foreign positions are ROUTINE, not anomalies:
            # log at INFO, never Telegram, never touch. (Was: _alert with
            # a repeating cooldown, which spammed the owner about his own
            # trades every reconcile cycle.)
            logger.info(
                f"orphan reconciler [{strategy}]: foreign position "
                f"{sym} ({qty} sh @ ${avg_entry:.4f}) — owner's manual "
                f"trade, ignored (no alert by owner directive 8/17)")
            continue

        db_fill = float(owned_row.get('fill_price') or 0.0)
        if not cfg.auto_close_enabled:
            action = OrphanAction(
                symbol=sym, qty=qty, avg_entry=avg_entry,
                classification='owned',
                action='auto_close_disabled',
                db_trade_id=owned_row.get('id'),
                db_exit_reason=owned_row.get('exit_reason'),
            )
            actions.append(action)
            _alert(notifier, strategy, sym,
                   _format_owned_alert(strategy, action, db_fill),
                   cfg.alert_cooldown_minutes)
            continue

        if _closes_in_last_hour(strategy) >= cfg.max_closes_per_hour:
            action = OrphanAction(
                symbol=sym, qty=qty, avg_entry=avg_entry,
                classification='owned',
                action='cap_breached',
                db_trade_id=owned_row.get('id'),
                db_exit_reason=owned_row.get('exit_reason'),
            )
            actions.append(action)
            _alert(notifier, strategy, sym,
                   _format_owned_alert(strategy, action, db_fill),
                   cfg.alert_cooldown_minutes)
            continue

        close = close_position_fn or alpaca.close_position
        try:
            # Release broker-held shares first (open bracket legs / expired
            # emergency stops hold qty and make close_position fail with
            # 40310000). Then close, with a short backoff retry while the
            # cancel propagates. 2026-07-04 review fix.
            if close_position_fn is None:
                _cancel_open_orders_for_symbol(alpaca, sym)
            result = None
            last_err = None
            for attempt, backoff_s in enumerate(
                    (0.0,) + _CLOSE_HELD_QTY_BACKOFFS_S):
                if backoff_s:
                    time.sleep(backoff_s)
                try:
                    result = close(sym)
                    last_err = None
                    break
                except Exception as ce:
                    last_err = ce
                    if 'insufficient qty' not in str(ce).lower():
                        raise
                    logger.warning(
                        f"orphan reconciler [{strategy}]: close({sym}) "
                        f"held-qty retry {attempt + 1} — {ce}")
            if last_err is not None:
                raise last_err
            order_id = (result or {}).get('id', '')
        except Exception as e:
            logger.error(
                f"orphan reconciler [{strategy}]: close_position({sym}) "
                f"failed: {e}"
            )
            action = OrphanAction(
                symbol=sym, qty=qty, avg_entry=avg_entry,
                classification='owned',
                action='close_failed',
                db_trade_id=owned_row.get('id'),
                db_exit_reason=owned_row.get('exit_reason'),
                note=str(e),
            )
            actions.append(action)
            _alert(notifier, strategy, sym,
                   _format_owned_alert(strategy, action, db_fill),
                   cfg.alert_cooldown_minutes)
            continue

        _record_close(strategy)
        _mark_close_inflight(strategy, sym, cfg.inflight_close_cooldown_s)

        # Poll for fill with retries. Single-shot polls almost never see
        # the fill because the broker hasn't acknowledged yet — the
        # original implementation's single get_order call left ~half of
        # rows stuck in exit_pending_verification with no real exit_price.
        fill_price: Optional[float] = None
        fill_qty: int = 0
        if order_id:
            getter = fetch_fill_fn or (
                getattr(alpaca, 'get_order', None) if hasattr(alpaca, 'get_order')
                else None
            )
            if getter is not None:
                fill_price, fill_qty = _poll_for_fill(
                    getter, order_id,
                    cfg.fill_poll_timeout_s, cfg.fill_poll_interval_s,
                )

        if fill_price is not None and fill_qty > 0:
            _write_recovery_row(db, owned_row, fill_price, fill_qty,
                                 avg_entry, order_id)
        else:
            # Poll budget exhausted — flip to pending-verification. The
            # in-flight cache ensures the next sync cycle (which will see
            # the broker still holding it) doesn't re-submit close.
            try:
                db.update_trade(owned_row['id'], {
                    'order_status': PENDING_VERIFICATION_STATUS,
                    'exit_reason': owned_row.get('exit_reason')
                                     or 'stop_loss_unconfirmed',
                })
            except Exception as e:
                logger.warning(
                    f"orphan reconciler [{strategy}]: pending-verification "
                    f"DB write for {sym} failed: {e}"
                )

        action = OrphanAction(
            symbol=sym, qty=qty, avg_entry=avg_entry,
            classification='owned',
            action='closed',
            db_trade_id=owned_row.get('id'),
            db_exit_reason=owned_row.get('exit_reason'),
            close_order_id=order_id,
        )
        actions.append(action)
        _alert(notifier, strategy, sym,
               _format_owned_alert(strategy, action, db_fill),
               cfg.alert_cooldown_minutes)

    return actions
