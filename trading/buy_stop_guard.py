"""
buy_stop_guard — shared pre-flight check for buy stop-limit orders.

Alpaca LIVE rejects a buy stop-limit whenever `stop_price <= current ASK`
(immediately-marketable stop, not a real stop). Paper Alpaca does not enforce
this. The IREZ+TTGT post-mortem (2026-05-08), KPTI/TRT (2026-05-14), and
today's ORB BTCZ/YSS/BMNZ rejections (2026-05-18) are all the same rule.

This module unifies the rejection-avoidance logic across:
  - bull flag entries via OrderExecutor.submit_buy_stop_order
  - ORB entries via ORBEngine._submit_entry

Both callers fetch a fresh NBBO quote, pass (bid, ask, stop, limit, buffer)
to `evaluate_buy_stop()`, and dispatch the returned BuyStopDecision via the
appropriate per-strategy submit method (BF: simple stop-limit / limit;
ORB: stop-limit-bracket / limit-bracket). The decision logic itself is pure
and stateless — parity by construction.

Decision tree (with default buffer = $0.02):

  Quote unavailable (bid <= 0 or ask <= 0)
    → SUBMIT_AS_IS (defensive — same as today's behavior, may reject)

  bid >= stop
    → MARKETABLE_LIMIT (breakout fully confirmed; enter at limit_price now)

  stop <= ask AND ask + buffer <= limit
    → REBUMP_STOP (straddle; bump stop to ask + buffer, keep limit)

  stop <= ask AND ask + buffer > limit
    → SKIP (breakout ran past max fill price; don't chase)

  ask < stop
    → SUBMIT_AS_IS (normal pre-breakout; stop above market)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BuyStopAction(str, Enum):
    """Action a caller should take after pre-flight quote inspection."""

    SUBMIT_AS_IS = "submit_as_is"
    MARKETABLE_LIMIT = "marketable_limit"
    REBUMP_STOP = "rebump_stop"
    SKIP = "skip"


@dataclass(frozen=True)
class BuyStopDecision:
    """Result of `evaluate_buy_stop()`.

    Attributes:
        action: One of BuyStopAction.
        new_stop_price: When action == REBUMP_STOP, the bumped stop price
            (rounded to 2 decimals, guaranteed >= ask and <= limit_price).
            Otherwise None.
        reason: Short human-readable explanation, suitable for logs.
    """

    action: BuyStopAction
    new_stop_price: Optional[float] = None
    reason: str = ""


_DEFAULT_REBUMP_BUFFER = 0.02


def evaluate_buy_stop(
    bid: float,
    ask: float,
    stop_price: float,
    limit_price: float,
    rebump_buffer: float = _DEFAULT_REBUMP_BUFFER,
) -> BuyStopDecision:
    """Pre-flight decision for a buy stop-limit order.

    Pure function — no I/O, no side effects, no logging. The caller fetches
    the quote and submits the chosen action.

    Args:
        bid: Current best bid. Pass 0 or negative when the quote is missing.
        ask: Current best ask. Pass 0 or negative when the quote is missing.
        stop_price: Intended stop trigger (typically breakout_level for
            bull flag, range_high for ORB).
        limit_price: Maximum acceptable fill price (typically stop_price ×
            1.02 — the 2% slippage cap shared by BF + ORB BT).
        rebump_buffer: Cents to add above the ask when rebumping the stop.
            Min enforced at 0.02 (rounding to 2dp guarantees stop > ask).

    Returns:
        BuyStopDecision describing how to submit (or skip).
    """
    # Floor the buffer so the rebumped stop is strictly > ask after rounding.
    if rebump_buffer < 0.02:
        rebump_buffer = 0.02

    # Bid branch — whole spread above the breakout level → breakout fully
    # confirmed by trades. A native stop here is immediately marketable →
    # Alpaca rejects. Enter NOW at limit_price; the slippage cap is preserved.
    # Only requires the bid side to be valid (callers may have a partial
    # quote with only one side populated).
    if bid > 0 and bid >= stop_price:
        return BuyStopDecision(
            action=BuyStopAction.MARKETABLE_LIMIT,
            reason=(
                f"breakout confirmed (bid ${bid:.2f} >= stop ${stop_price:.2f})"
            ),
        )

    # Ask branch — spread straddles the breakout level (bid < stop <= ask) or
    # ask is the only valid side. A native stop is immediately marketable →
    # Alpaca rejects. Re-bump the stop just above the ask so it stays a real
    # stop that only fires on a genuine upward print. Skip if the bumped stop
    # would exceed limit_price — breakout ran past our max fill price; don't
    # chase. Only requires the ask side to be valid.
    if ask > 0 and ask >= stop_price:
        new_stop = round(ask + rebump_buffer, 2)
        if new_stop > limit_price:
            return BuyStopDecision(
                action=BuyStopAction.SKIP,
                reason=(
                    f"breakout extended past limit "
                    f"(ask ${ask:.2f} + buf ${rebump_buffer:.2f} = "
                    f"${new_stop:.2f} > limit ${limit_price:.2f})"
                ),
            )
        return BuyStopDecision(
            action=BuyStopAction.REBUMP_STOP,
            new_stop_price=new_stop,
            reason=(
                f"spread straddles stop "
                f"(bid ${bid:.2f} < stop ${stop_price:.2f} <= ask ${ask:.2f}); "
                f"rebumping stop to ${new_stop:.2f}"
            ),
        )

    # Either ask < stop (normal pre-breakout) OR both quotes are degenerate
    # (defensive fallback — submit at the intended levels, may reject but no
    # worse than today's behavior).
    return BuyStopDecision(
        action=BuyStopAction.SUBMIT_AS_IS,
        reason=(
            f"normal pre-breakout (ask ${ask:.2f} < stop ${stop_price:.2f})"
            if ask > 0
            else "no quote available — defensive fallback"
        ),
    )
