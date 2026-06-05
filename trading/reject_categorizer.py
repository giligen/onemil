"""
reject_categorizer — coarse classification of Alpaca order rejection reasons.

Alpaca's rejection metadata is notoriously sparse: `reject_reason` is often
None for synchronous server-side rejections (the `stop ≤ ask` 4ms rejects
that triggered the buy_stop_guard ship had `reject_reason=None`). Sometimes
it's a freeform string; sometimes it's `code=NNNN, message=...`.

This module gives us a small set of stable categories so:
  * journalctl + DB grep can quickly count by class (margin vs wash vs stop)
  * Telegram alerts can use a category-aware tone
  * The system_state monitor can be triggered when a rejection looks
    account-level (margin deficit, suspended account, …)

Status: scaffolding. We've seen `stop_below_ask` (handled upstream by
buy_stop_guard) and `insufficient_qty` (handled by orphan-cleanup). We have
NOT yet observed a margin-deficit rejection on this account under the new
intraday-margin framework — the categorizer therefore matches against
documented Alpaca error codes and known FINRA terminology, but the catalog
will need updating once a real margin-deficit reject lands in our logs.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class RejectCategory(str, Enum):
    """Coarse category for an Alpaca order rejection."""

    MARGIN_DEFICIT = "margin_deficit"        # account-level — triggers halt scrutiny
    WASH_TRADE = "wash_trade"                # SOR / regulatory
    STOP_PRICE_INVALID = "stop_price_invalid"  # the 4ms `stop ≤ ask` class
    INSUFFICIENT_QTY = "insufficient_qty"    # bracket leg held — pre-cancel needed
    SYMBOL_HALTED = "symbol_halted"          # trading halt on the name
    NOT_TRADABLE = "not_tradable"            # asset_status, delisting, restricted-list
    ACCOUNT_BLOCKED = "account_blocked"      # account-level — triggers halt scrutiny
    UNKNOWN = "unknown"                      # surface for investigation
    NONE = "none"                            # no reject_reason / no rejection


# Mapping from substring → category. Lowercased substring match.
# Add entries as we observe new reject_reason strings.
_PATTERNS: list[tuple[str, RejectCategory]] = [
    # Known stop-price classes (handled by buy_stop_guard but may still appear)
    ("stop price must be", RejectCategory.STOP_PRICE_INVALID),
    ("immediately marketable", RejectCategory.STOP_PRICE_INVALID),
    # Quantity / position
    ("insufficient qty", RejectCategory.INSUFFICIENT_QTY),
    ("held_for_orders", RejectCategory.INSUFFICIENT_QTY),
    # Margin (likely strings — confirm when observed)
    ("insufficient buying power", RejectCategory.MARGIN_DEFICIT),
    ("insufficient margin", RejectCategory.MARGIN_DEFICIT),
    ("margin call", RejectCategory.MARGIN_DEFICIT),
    ("intraday margin", RejectCategory.MARGIN_DEFICIT),
    ("buying power deficit", RejectCategory.MARGIN_DEFICIT),
    ("dtbp", RejectCategory.MARGIN_DEFICIT),
    # Wash trade
    ("wash trade", RejectCategory.WASH_TRADE),
    ("wash sale", RejectCategory.WASH_TRADE),
    # Halt
    ("symbol is halted", RejectCategory.SYMBOL_HALTED),
    ("trading halt", RejectCategory.SYMBOL_HALTED),
    # Asset
    ("not tradable", RejectCategory.NOT_TRADABLE),
    ("delisted", RejectCategory.NOT_TRADABLE),
    ("restricted list", RejectCategory.NOT_TRADABLE),
    # Account
    ("account blocked", RejectCategory.ACCOUNT_BLOCKED),
    ("trading blocked", RejectCategory.ACCOUNT_BLOCKED),
]


def categorize_reject(
    reject_reason: Optional[str],
    status: Optional[Any] = None,
) -> RejectCategory:
    """Classify a rejection. Pure function — no I/O, no logging.

    Args:
        reject_reason: the `reject_reason` field as Alpaca returned it
            (may be None for some sync rejections).
        status: the order status (string or enum). When provided, used as a
            short-circuit — non-rejected orders return NONE.

    Returns:
        A RejectCategory. UNKNOWN means we got a reject we don't recognize
        yet — surface it for catalog updating.
    """
    if status is not None:
        s = str(getattr(status, "value", status)).lower()
        if s != "rejected" and s != "rejected_by_broker":
            return RejectCategory.NONE
    if reject_reason is None:
        return RejectCategory.UNKNOWN
    rr = str(reject_reason).lower()
    if not rr.strip():
        return RejectCategory.UNKNOWN
    for needle, category in _PATTERNS:
        if needle in rr:
            return category
    return RejectCategory.UNKNOWN


def is_account_level(category: RejectCategory) -> bool:
    """Whether a category implies the whole account is unable to trade —
    callers should re-check `account_state_monitor` to decide whether to
    set the system halt flag.
    """
    return category in (
        RejectCategory.MARGIN_DEFICIT,
        RejectCategory.ACCOUNT_BLOCKED,
    )
