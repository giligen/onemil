"""ORB exit: rest the target as a real limit at the broker.

Spec: docs/orb_target_limit_spec_20260929.md (main session, 2026-09-29).
Rulebook entry: research/orb_machine_rules.md.

## Why

Today, when ORB's touchgo filter (Rule M / Rule D — trading/orb_touchgo_filter.py)
decides an open position should exit, `ORBEngine._fire_touchgo_exit` calls
`StopMonitor.force_exit(limit_price=exit_price)`, which submits a MARKETABLE
limit sell — crossing the spread every time. Measured live: 68.8 bps mean /
47.6 bps median / p90 140 bps worse than the target price, on 39 fills, no
fill ever beat it (research/exec_quality/REPORT_20260928.md §4). The backtest
fills the target at the touch; resting a real limit at the broker is the live
mechanism that matches it (obtainability: the tape trades at or through the
limit while our order has queue priority).

## Design decision (stated per the spec's explicit ask)

The ORB entry bracket ALREADY carries a take-profit leg (`trading/orb_engine.py`
`submit_entry`, `safety_tp = entry_price * 3.0` — an intentionally UNREACHABLE
safety net, "ORB has no fixed target; legally must set"). Rather than create a
second, freestanding resting sell (which would leave TWO TP-side orders live,
or require an extra cancel-then-submit step with its own race window), this
module REPRICES that existing leg from the unreachable safety price to the
real touchgo target via `AlpacaClient.replace_order_limit_price` — the same
primitive `trading/stop_monitor.py`'s D3-FIX-1 stop path already uses to
reprice the SL leg in place instead of cancel-and-resubmit. This preserves the
broker-side OCO relationship with the SL leg (a fill on one leg auto-cancels
the other — no orphan safety-net order), and reuses an existing, already-
shipped code path instead of inventing a new order type.

Alpaca's replace endpoint mints a NEW order id for the replaced leg (already
documented in this codebase — trading/hod_break_engine.py:1163's HOD resting-
entry lesson: "replace returns a NEW order id AND a new random client_order_id
— our prefix is lost"). To keep the `orb-tp-<sym>-<yyyymmdd>` naming rule 7's
boot reconciliation depends on, every replace call in this module explicitly
passes `client_order_id` (ReplaceOrderRequest supports it) rather than
letting Alpaca mint a random one. Every caller of `reprice_target` MUST persist
the returned `new_leg_id` as the position's current TP-leg id (both
`OpenPosition.tp_leg_id` and `StopMonitor`'s `WatchEntry.tp_leg_id` via
`StopMonitor.mark_target_resting`) — a stale id makes the next cancel a no-op
against a dead order.

## What lives here vs. the callers

Every function below is synchronous and takes an already-authenticated Alpaca
client (or `MagicMock(spec=AlpacaClient)` in tests) plus plain data — no
asyncio, no ORBEngine/StopMonitor coupling — because `orb_engine.py` (sync
tick loop) and `stop_monitor.py` (asyncio WS thread, D3-FIX-1 style) each call
these from different execution contexts and must not duplicate the broker-
facing logic (parity by construction, CLAUDE.md "ONE spec ... shared by BT and
live through ONE helper module").
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

# Rule 4: at most one replace (target move) per symbol per this many seconds.
REPLACE_RATE_LIMIT_S = 5.0

# Rule 2: cancel-before-stop ack-wait budget (enforced by the asyncio caller
# via asyncio.wait_for — this constant is the shared source of truth so the
# stop_monitor.py hook and any test asserting the budget can't drift apart).
CANCEL_ACK_WAIT_S = 1.0

# Rule 7: the ONE client_order_id prefix every resting-target order carries,
# and the ONE thing boot/sync reconciliation greps for.
TARGET_COID_PREFIX = "orb-tp-"


def target_client_order_id(symbol: str, as_of: Optional[datetime] = None) -> str:
    """Build the `orb-tp-<sym>-<yyyymmdd>` client_order_id (spec mechanism).

    `as_of` is injectable so tests get a deterministic string; defaults to
    now (UTC).
    """
    day = (as_of or datetime.now(timezone.utc)).strftime('%Y%m%d')
    return f"{TARGET_COID_PREFIX}{symbol}-{day}"


class RestOutcome(str, Enum):
    """Result of an attempt to (re)rest the target limit (rules 1 + 4)."""

    RESTED = "rested"                  # replace succeeded; new leg is live
    RATE_LIMITED = "rate_limited"      # < REPLACE_RATE_LIMIT_S since the last replace; old TP kept
    NO_LEG = "no_leg"                  # tp_leg_id empty — nothing to reprice
    FAILED = "failed"                  # broker call raised or returned no id; old TP kept


@dataclass
class RestResult:
    """Outcome of `reprice_target`. `new_leg_id` is only meaningful when
    `outcome == RestOutcome.RESTED` — every other outcome means the PRIOR
    leg (whatever it was) is still the live one."""

    outcome: RestOutcome
    new_leg_id: str = ''
    detail: str = ''


def reprice_target(
    alpaca: Any,
    symbol: str,
    tp_leg_id: str,
    target_price: float,
    last_replace_ts: float,
    now_ts: Optional[float] = None,
    force_first: bool = False,
) -> RestResult:
    """Reprice the bracket TP leg to `target_price`.

    Rule 1 (first rest, from the unreachable safety price to the real
    touchgo target) and rule 4 (later moves when a trail/lock rule updates
    the target) are the SAME broker operation — a limit-price replace on the
    existing leg — so this one function serves both; `force_first` bypasses
    the rate limit for rule 1 (the rate limit exists only to throttle
    repeated MOVES, never the initial rest).

    Args:
        alpaca: AlpacaClient (or MagicMock(spec=AlpacaClient)). Must expose
            `replace_order_limit_price(order_id, new_limit_price,
            client_order_id=...)`.
        symbol: for logging + the client_order_id.
        tp_leg_id: current Alpaca order id of the TP-side leg. Empty means
            the entry bracket's leg extraction failed at submit time
            (trading/orb_engine.py logs this at WARNING already) — nothing
            to reprice.
        target_price: the touchgo-computed exit price to rest at.
        last_replace_ts: time.time() of the previous replace (0.0 if never
            replaced yet — i.e. still the original safety-net leg).
        now_ts: injected clock for tests; defaults to time.time().
        force_first: True on the very first rest for this position — skips
            the rate-limit check entirely.

    Returns:
        RestResult. Every non-RESTED outcome is logged at WARNING here with
        the reason (all fallback paths log per CLAUDE.md); the caller's job
        is only to decide what happens next (fall back to the chase-and-sell
        path on rule 1's first call, or silently keep the old TP on a rule-4
        move — the spec's own wording for rule 4).
    """
    now = now_ts if now_ts is not None else time.time()
    if not tp_leg_id:
        logger.warning(
            f"[ORB TP] {symbol} no tp_leg_id to reprice — entry bracket leg "
            f"extraction must have failed at submit; nothing to rest"
        )
        return RestResult(RestOutcome.NO_LEG, detail="no tp_leg_id")
    if not force_first and last_replace_ts > 0 and (now - last_replace_ts) < REPLACE_RATE_LIMIT_S:
        logger.warning(
            f"[ORB TP] {symbol} replace rate-limited "
            f"({now - last_replace_ts:.1f}s < {REPLACE_RATE_LIMIT_S:.0f}s "
            f"since the last move) — keeping the old resting TP"
        )
        return RestResult(RestOutcome.RATE_LIMITED, detail="rate limited")
    coid = target_client_order_id(symbol, datetime.fromtimestamp(now, tz=timezone.utc))
    try:
        result = alpaca.replace_order_limit_price(
            tp_leg_id, target_price, client_order_id=coid,
        )
        new_id = str((result or {}).get('id') or '')
        if not new_id:
            logger.warning(
                f"[ORB TP] {symbol} replace_order_limit_price returned no "
                f"id ({result!r}) — treating as failed, keeping the old TP"
            )
            return RestResult(RestOutcome.FAILED, detail="empty id in response")
        logger.info(
            f"[ORB TP] rested {symbol} @ ${target_price:.2f} "
            f"(leg {tp_leg_id[:8]} -> {new_id[:8]}, coid={coid})"
        )
        return RestResult(RestOutcome.RESTED, new_leg_id=new_id, detail=coid)
    except Exception as e:
        logger.warning(
            f"[ORB TP] {symbol} replace_order_limit_price({tp_leg_id[:8]}, "
            f"${target_price:.2f}) failed: {e} — keeping the old TP"
        )
        return RestResult(RestOutcome.FAILED, detail=str(e))


class CancelOutcome(str, Enum):
    """Result of `cancel_resting_target` (rules 2 + 5)."""

    CANCELLED = "cancelled"                # 0 filled; order is dead, caller may proceed
    ALREADY_FILLED = "already_filled"      # rule 2's race: the WHOLE order filled, position flat
    PARTIALLY_FILLED = "partially_filled"  # rule 6: SOME shares filled, remainder is dead/dying
    NO_LEG = "no_leg"
    ERROR = "error"                        # unresolved — caller must NOT assume flat


@dataclass
class CancelResult:
    outcome: CancelOutcome
    fill_price: float = 0.0
    filled_qty: int = 0
    detail: str = ''


def cancel_resting_target(
    alpaca: Any, symbol: str, tp_leg_id: str, reason: str, requested_qty: int = 0,
) -> CancelResult:
    """Cancel the resting TP leg before any other exit path submits its own
    sell (rule 2: stop trigger; rule 5: EOD and every other exit path).

    Synchronous — `stop_monitor.py`'s asyncio loop dispatches this via
    `run_in_executor` under an `asyncio.wait_for(..., CANCEL_ACK_WAIT_S)`
    budget; `orb_engine.py`'s sync EOD/reconciliation paths call it directly.

    MONEY-CORRECTNESS NOTE (rule 6, fixed 2026-09-29 follow-up): a PARTIALLY
    filled order still returns `cancel_order() == True` (Alpaca successfully
    cancels the remaining OPEN quantity of a partial fill — this is NOT the
    same as "nothing filled"). A cancel's boolean result alone can therefore
    NEVER prove zero fill. This function always makes a `get_order` call
    after the cancel attempt — regardless of what the cancel itself returned
    — and classifies on `filled_qty` vs `requested_qty`, the only reliable
    source of truth:
        filled_qty == 0                      -> CANCELLED
        0 < filled_qty < requested_qty        -> PARTIALLY_FILLED
        filled_qty >= requested_qty (or 'filled') -> ALREADY_FILLED
    `requested_qty` should be the position's CURRENT remaining shares at the
    moment of the call (watch.shares / pos.shares) — the caller must reduce
    its own bookkeeping by `filled_qty` on every non-CANCELLED outcome
    before treating the rest of the position as still fully sized.
    """
    if not tp_leg_id:
        return CancelResult(CancelOutcome.NO_LEG, detail="no tp_leg_id")
    try:
        cancel_acked = alpaca.cancel_order(tp_leg_id)
    except Exception as e:
        logger.error(
            f"[ORB TP] {symbol} cancel_order({tp_leg_id[:8]}) raised "
            f"({reason}): {e} — checking order status to resolve the race"
        )
        cancel_acked = False
    # ALWAYS verify via get_order — a True cancel_order result only means
    # "no OPEN quantity remains"; it does NOT mean zero shares filled (a
    # partially-filled order cancels its remainder just as cleanly as an
    # untouched one). Skipping this check on a bare `True` was the 2026-09-29
    # money defect: a partial race would have been silently booked as a
    # same-size full close.
    try:
        order = alpaca.get_order(tp_leg_id)
    except Exception as e:
        logger.error(
            f"[ORB TP] {symbol} get_order({tp_leg_id[:8]}) failed while "
            f"resolving cancel_acked={cancel_acked} ({reason}): {e} — "
            f"treating as an UNRESOLVED error; caller must not assume the "
            f"position is flat OR that the full requested qty is still held"
        )
        return CancelResult(CancelOutcome.ERROR, detail=str(e))
    status = str((order or {}).get('status') or '')
    filled_qty = int((order or {}).get('filled_qty') or 0)
    fill_price = float((order or {}).get('filled_avg_price') or 0.0)
    if filled_qty <= 0:
        logger.info(
            f"[ORB TP] {symbol} cancelled resting TP {tp_leg_id[:8]} ({reason}, "
            f"cancel_acked={cancel_acked}, status={status!r}, 0 filled)"
        )
        return CancelResult(CancelOutcome.CANCELLED, detail=f"0 filled, status={status}")
    if requested_qty > 0 and filled_qty < requested_qty:
        logger.warning(
            f"[ORB TP] {symbol} cancel RACE — target PARTIALLY filled "
            f"{filled_qty}/{requested_qty}sh @ ${fill_price:.2f} before the "
            f"cancel landed ({reason}); booking the partial, remainder "
            f"({requested_qty - filled_qty}sh) continues under its stop"
        )
        return CancelResult(CancelOutcome.PARTIALLY_FILLED, fill_price=fill_price,
                             filled_qty=filled_qty, detail=status)
    logger.warning(
        f"[ORB TP] {symbol} cancel RACE — target already FILLED "
        f"{filled_qty}sh @ ${fill_price:.2f} before the cancel landed "
        f"({reason}); recording target_rested, no stop order"
    )
    return CancelResult(CancelOutcome.ALREADY_FILLED, fill_price=fill_price,
                         filled_qty=filled_qty, detail=status)


def reconcile_orphan_targets(
    alpaca: Any,
    open_symbols_with_target: Iterable[str],
    all_open_orders: Iterable[Dict[str, Any]],
) -> Dict[str, list]:
    """Rule 7, orphan half: any `orb-tp-*` order with no matching open
    position is an orphan — cancel it and WARN.

    Args:
        alpaca: AlpacaClient-like; needs `cancel_order(order_id)`.
        open_symbols_with_target: symbols this engine currently tracks as
            target_resting=True AFTER rehydration (sync_positions).
        all_open_orders: the account's full open-orders list, each dict-like
            with at least 'client_order_id', 'id', 'symbol' (AlpacaClient's
            own order-dict shape — see `get_order`'s return, or a `list_orders`
            equivalent the caller builds).

    Returns:
        {'orphans_cancelled': [coid, ...], 'orphans_cancel_failed': [coid, ...]}

    Does NOT re-rest a missing target for a position that should have one —
    that decision needs per-position touchgo state (whether a target was
    ever computed for this position) that only ORBEngine's own DB/trade
    record carries; see ORBEngine.sync_positions.
    """
    cancelled, failed = [], []
    symbols = set(open_symbols_with_target)
    for order in all_open_orders:
        coid = str((order or {}).get('client_order_id') or '')
        if not coid.startswith(TARGET_COID_PREFIX):
            continue
        sym = str((order or {}).get('symbol') or '')
        if sym in symbols:
            continue  # matches a live target_resting position; not an orphan
        oid = str((order or {}).get('id') or '')
        logger.warning(
            f"[ORB TP] orphan resting TP {coid} (order {oid[:8] if oid else '?'}, "
            f"symbol {sym or '?'}) — no matching open position; cancelling"
        )
        try:
            alpaca.cancel_order(oid)
            cancelled.append(coid)
        except Exception as e:
            logger.error(f"[ORB TP] orphan cancel {coid} failed: {e}")
            failed.append(coid)
    return {'orphans_cancelled': cancelled, 'orphans_cancel_failed': failed}
