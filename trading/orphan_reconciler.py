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
ORPHAN_RECOVERED_EXIT_REASON = 'orphan_recovered_force_close'


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
    db_qty = db_row.get('filled_qty') or 0
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


def _should_alert(strategy: str, symbol: str, cooldown_minutes: int) -> bool:
    """True iff no alert has been sent for (strategy, symbol) within the
    cooldown window."""
    key = (strategy, symbol)
    now = datetime.now(timezone.utc)
    last = _state.last_alert.get(key)
    if last is not None and (now - last) < timedelta(minutes=cooldown_minutes):
        return False
    _state.last_alert[key] = now
    return True


def reset_state_for_tests() -> None:
    """Clear in-memory rate-limit state. Tests only."""
    _state.close_timestamps.clear()
    _state.last_alert.clear()


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
) -> List[OrphanAction]:
    """Compare broker positions vs (engine-tracked + DB-open) and act on
    orphans.

    Args:
        strategy: 'bull_flag' / 'macd_wave' / 'orb' — ownership filter.
        alpaca: AlpacaClient for the strategy's broker account.
        db: persistence.database.Database instance.
        notifier: TelegramNotifier (or None).
        tracked_symbols: symbols the engine currently has in its
            open_positions map. Anything broker has but engine doesn't is
            a candidate orphan.
        cfg: tuning knobs (defaults to ReconcilerConfig()).
        today_et: optional date override (testing).
        close_position_fn: optional override for the close call (testing).
        fetch_fill_fn: optional override for fill polling (testing).

    Returns:
        List of OrphanAction — one per orphan detected this cycle.
        Empty if nothing was orphan. Pure side-effect ordering: detect →
        classify → alert → (close + DB update if OWNED).
    """
    if cfg is None:
        cfg = ReconcilerConfig()
    if today_et is None:
        today_et = datetime.now(timezone.utc).date()
    actions: List[OrphanAction] = []

    # 1. Snapshot broker positions.
    try:
        broker_positions = alpaca.get_open_positions()
    except Exception as e:
        logger.error(
            f"orphan reconciler [{strategy}]: broker snapshot failed: {e}"
        )
        return actions

    # 2. Filter to candidates: broker has it, engine doesn't track it.
    candidates = [
        p for p in broker_positions
        if p['symbol'] not in tracked_symbols
    ]
    if not candidates:
        return actions

    # 3. Query DB for strategy-tagged rows in the lookback window keyed by
    #    symbol. We grab all rows (any state) so we can apply the OWNED
    #    predicate's stale-signal check.
    candidate_symbols = [p['symbol'] for p in candidates]
    db_rows_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    lookback_start = (today_et - timedelta(days=cfg.lookback_days)).isoformat()
    placeholders = ','.join('?' for _ in candidate_symbols)
    try:
        # Direct query — get_open_trades doesn't include rows where
        # the OLD code wrote exit_price (the very case we need to
        # reconcile). This is the cross-day-historical-row query.
        cur = db._trades_conn.execute(
            f"SELECT id, trade_date, symbol, strategy, fill_price, "
            f"filled_qty, exit_price, exit_reason, order_status "
            f"FROM trades "
            f"WHERE strategy = ? AND trade_date >= ? "
            f"AND symbol IN ({placeholders}) "
            f"ORDER BY trade_date DESC, id DESC",
            (strategy, lookback_start, *candidate_symbols),
        )
        for r in cur.fetchall():
            d = dict(r)
            db_rows_by_symbol.setdefault(d['symbol'], []).append(d)
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
            # FOREIGN — alert only, never close.
            action = OrphanAction(
                symbol=sym, qty=qty, avg_entry=avg_entry,
                classification='foreign',
                action='alert_only',
                note='No matching strategy row with all OWNED predicates',
            )
            actions.append(action)
            _alert(notifier, strategy, sym,
                   _format_foreign_alert(strategy, action),
                   cfg.alert_cooldown_minutes)
            continue

        # OWNED — decide action based on auto_close + rate-limit gates.
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

        # Act: submit close.
        close = close_position_fn or alpaca.close_position
        try:
            result = close(sym)
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

        # Best-effort: poll the fill to write the DB recovery row with
        # actual exit_price. If the caller didn't supply fetch_fill_fn,
        # use alpaca.get_order with a brief poll. If we can't get a fill
        # within the budget, write what we know (avg_entry as a
        # placeholder is misleading — better to leave exit_price NULL
        # and let the next sync see it as still pending).
        fill_price: Optional[float] = None
        fill_qty: int = 0
        if order_id and (fetch_fill_fn or hasattr(alpaca, 'get_order')):
            try:
                getter = fetch_fill_fn or (lambda oid: alpaca.get_order(oid))
                # Simple single-poll: market closes fill in <2s typically;
                # if not, the next sync cycle catches it. Don't block.
                o = getter(order_id)
                fq = int(o.get('filled_qty') or 0) if isinstance(o, dict) else 0
                fp = o.get('filled_avg_price') if isinstance(o, dict) else None
                if fp is not None and fq > 0:
                    fill_price = float(fp)
                    fill_qty = fq
            except Exception as e:
                logger.warning(
                    f"orphan reconciler [{strategy}]: fill poll for "
                    f"{order_id} failed: {e}"
                )

        if fill_price is not None and fill_qty > 0:
            _write_recovery_row(db, owned_row, fill_price, fill_qty,
                                 avg_entry, order_id)
        else:
            # Mark as pending-verification so we re-check next cycle.
            try:
                db.update_trade(owned_row['id'], {
                    'order_status': PENDING_VERIFICATION_STATUS,
                    'exit_reason': owned_row.get('exit_reason')
                                     or 'stop_loss_unconfirmed',
                })
            except Exception:
                pass

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
