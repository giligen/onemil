"""Exit-coverage invariant (docs/exit_coverage_invariant_spec_20261029.md).

Invariant: for every position the system watches, the broker holds the shares
we think it holds AND the open SELL orders on that symbol cover at least that
many shares. A position with no (or too little) resting exit is NAKED.

LABX 10/27: a partial-fill stall cancelled the bracket legs, 423 shares sat
with no broker exit for six hours and nothing noticed. This module is the one
engine-agnostic check that catches that whole class; `StopMonitor` runs it
every 60 s (`_exit_coverage_loop`).

Pure REST reads, existing `AlpacaClient` methods only:
  * positions  -> `AlpacaClient.get_open_positions()`
  * open sells -> `AlpacaClient.get_open_orders()` (status OPEN, flat list so
    bracket / OCO legs appear as separate rows)
  * repair     -> `AlpacaClient.submit_stop_sell_order()` (same call the
    StopMonitor EMERGENCY stop-market path uses)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Broker statuses that mean "this order can still sell shares".
LIVE_ORDER_STATUSES = frozenset({
    "new", "accepted", "held", "partially_filled", "pending_new",
    "pending_replace", "pending_cancel", "accepted_for_bidding",
})
# Terminal statuses: a leg id with one of these is dead.
TERMINAL_ORDER_STATUSES = frozenset({
    "canceled", "cancelled", "expired", "filled", "rejected", "replaced",
    "done_for_day", "stopped", "suspended",
})
# Multi-leg classes: sibling legs are alternatives (one fills, the other is
# cancelled), so together they cover ONE quantity, not the sum.
_LINKED_CLASSES = frozenset({"oco", "bracket", "oto"})


@dataclass
class CoverageResult:
    """Outcome of one coverage check for one symbol."""

    position_qty: float
    covered_qty: float
    open_sell_orders: List[Dict[str, Any]] = field(default_factory=list)
    naked_qty: float = 0.0
    qty_expected: float = 0.0
    excess_qty: float = 0.0  # broker shares above the watch qty: NOT ours

    @property
    def broker_short_of_watch(self) -> bool:
        """True when the broker holds fewer shares than the watch believes."""
        return self.position_qty < self.qty_expected


@dataclass
class BrokerSnapshot:
    """One batched read of an account: {symbol: long qty} and open orders."""

    positions: Dict[str, float]
    orders: List[Dict[str, Any]]


def _norm(value: Any) -> str:
    """Lower-case enum-ish value ('OrderStatus.NEW' / 'new') to a bare word."""
    text = str(getattr(value, "value", value) or "").lower()
    return text.rsplit(".", 1)[-1]


def snapshot(client) -> BrokerSnapshot:
    """Read positions + open orders from `client` (two REST calls).

    Raises whatever the client raises; the caller decides how to report it.
    Short positions are not ours (watches are long) and are ignored.
    """
    positions: Dict[str, float] = {}
    for pos in client.get_open_positions() or []:
        qty = float(pos.get("qty") or 0)
        side = _norm(pos.get("side", "long"))
        if side == "short" or qty <= 0:
            continue
        positions[str(pos.get("symbol", ""))] = qty
    orders = list(client.get_open_orders() or [])
    return BrokerSnapshot(positions=positions, orders=orders)


def _remaining(order: Dict[str, Any]) -> float:
    """Shares an open order can still sell (qty - filled_qty, floored at 0)."""
    qty = float(order.get("qty") or 0)
    filled = float(order.get("filled_qty") or 0)
    return max(0.0, qty - filled)


def covered_qty(open_sells: List[Dict[str, Any]]) -> float:
    """Shares protected by `open_sells` (live SELL orders on one symbol).

    Independent orders add up. Linked legs (OCO / bracket / OTO) are
    alternatives and count once, as the largest leg.
    """
    standalone = 0.0
    linked_max = 0.0
    for order in open_sells:
        remaining = _remaining(order)
        if _norm(order.get("order_class", "")) in _LINKED_CLASSES:
            linked_max = max(linked_max, remaining)
        else:
            standalone += remaining
    return standalone + linked_max


def check(client, symbol: str, qty_expected: float,
          snap: Optional[BrokerSnapshot] = None) -> CoverageResult:
    """Compare the broker position and open sells on `symbol`.

    Args:
        client: AlpacaClient of the account that owns the watch.
        symbol: ticker.
        qty_expected: shares the watch believes it holds.
        snap: a pre-fetched `BrokerSnapshot` (the monitor reads each account
            once per cycle); fetched here when omitted.

    Returns:
        CoverageResult. `naked_qty` = broker qty - covered qty, but 0 when the
        broker holds FEWER shares than the watch believes (orphan / over-exit
        paths own that case) or none at all. Repair size is capped at the
        watch qty: broker shares above it (`excess_qty`, e.g. the owner's
        manual shares on the shared account) are never ours to cover.
    """
    if snap is None:
        snap = snapshot(client)
    position_qty = float(snap.positions.get(symbol, 0.0))
    open_sells = [
        o for o in snap.orders
        if o.get("symbol") == symbol
        and _norm(o.get("side")) == "sell"
        and _norm(o.get("status")) in LIVE_ORDER_STATUSES
    ]
    covered = covered_qty(open_sells)
    if position_qty <= 0 or position_qty < float(qty_expected):
        naked = 0.0
    else:
        naked = max(0.0, min(float(qty_expected), position_qty) - covered)
    excess = max(0.0, position_qty - float(qty_expected))
    return CoverageResult(
        position_qty=position_qty,
        covered_qty=covered,
        open_sell_orders=open_sells,
        naked_qty=naked,
        qty_expected=float(qty_expected),
        excess_qty=excess,
    )


def _leg_is_dead(client, leg_id: str, symbol: str) -> bool:
    """True when `leg_id` is empty or the broker reports it terminal."""
    if not leg_id:
        return True
    try:
        status = _norm(client.get_order(leg_id).get("status", ""))
    except Exception as e:  # unknown state -> keep the old id, say so
        logger.warning(
            f"EXIT COVERAGE: {symbol} could not read old sl leg {leg_id[:8]}: "
            f"{e} — leaving watch.sl_leg_id unchanged")
        return False
    return status in TERMINAL_ORDER_STATUSES


def repair(client, watch, result: CoverageResult,
           place_stop_fn: Callable[[str, int, float], Dict[str, Any]]) -> str:
    """Place ONE plain stop-sell for the naked shares.

    Cancels nothing. Uses the watch's current `stop_price` (the lock stop once
    armed — StopMonitor writes the locked level into `stop_price`).

    Args:
        client: AlpacaClient of the watch's account (reads the old sl leg).
        watch: the StopMonitor WatchEntry (symbol, strategy, stop_price,
            sl_leg_id).
        result: the confirming `check` result with `naked_qty > 0`.
        place_stop_fn: `(symbol, qty, stop_price) -> {'id': ...}`.

    Returns:
        The new order id, or '' when nothing was placed (no naked shares, no
        stop price, or the placement failed — logged CRITICAL).
    """
    naked = int(result.naked_qty)
    symbol = watch.symbol
    if naked <= 0:
        return ""
    stop = float(getattr(watch, "stop_price", 0) or 0)
    if stop <= 0:
        logger.critical(
            f"EXIT COVERAGE: {watch.strategy} {symbol} NAKED {naked}/"
            f"{int(result.position_qty)} sh and the repair failed: watch has "
            f"no stop price — VERIFY ON ALPACA")
        return ""
    try:
        placed = place_stop_fn(symbol, naked, stop) or {}
    except Exception as e:
        logger.critical(
            f"EXIT COVERAGE: {watch.strategy} {symbol} NAKED {naked}/"
            f"{int(result.position_qty)} sh and the repair failed: {e} — "
            f"VERIFY ON ALPACA")
        return ""
    order_id = str(placed.get("id", "") or "")
    if _leg_is_dead(client, getattr(watch, "sl_leg_id", ""), symbol) and order_id:
        watch.sl_leg_id = order_id
    logger.error(
        f"EXIT COVERAGE: {watch.strategy} {symbol} NAKED {naked}/"
        f"{int(result.position_qty)} sh — placed stop-sell @ {stop:.2f} "
        f"(order {order_id[:8]})")
    return order_id
