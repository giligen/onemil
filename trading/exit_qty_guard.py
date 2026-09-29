"""Broker-truth sell-qty guard, shared by every live exit path (HOD stop/target/force-close/EOD/
dead-man; mirrored anywhere else a registry qty is sold). 2026-09-25 CDNA incident: HOD's registry
still showed 57 sh open after StopMonitor's own stop-loss had already flattened the broker position
(the drain that marks the registry closed raced a restart and lost the event); force_close_all sold
the registry's 57 sh again at 15:55 with no broker check, taking the shared live account SHORT.

ONE helper, never re-implemented per call site: every exit submission must clamp to what the broker
actually shows long for the symbol, and must NEVER submit a sell that can create or extend a short.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_broker_capped_sell_qty(
    symbol: str,
    registry_qty: int,
    broker_qty_signed: int,
    tag: str,
    notify_fn=None,
) -> Optional[int]:
    """Return the qty a live exit path may actually sell for `symbol`, or None to skip entirely.

    Args:
        symbol: ticker.
        registry_qty: shares our own position registry believes are open (must be > 0; callers
            should not invoke this for an already-flat registry entry).
        broker_qty_signed: the broker's ACTUAL position qty for this symbol on OUR account, signed
            (negative = broker already short). Callers must pass the real signed value, never abs().
        tag: log/Telegram prefix, e.g. "[HOD]".
        notify_fn: optional callable(str) -> None for a Telegram line; called only when this guard
            changes or blocks the sell (never on the silent-match common case).

    Returns:
        None -- broker shows no long position (flat or already short): the caller MUST submit
            nothing. Logged at WARNING (never a silent skip).
        int -- broker_qty_signed, always. Selling exactly what the broker shows long can only ever
            reach flat, never short, regardless of whether that is MORE or LESS than the registry
            believed: less (a stale-high registry, e.g. the 9/25 CDNA drain race) must clamp DOWN
            to avoid a short; more (a stale-low registry, e.g. a partial-fill race the registry
            missed — the APT/MLTX 2026-05-11 class) must clamp UP so the sell doesn't strand an
            orphan residual. Both directions are logged at WARNING; the silent common case (exact
            match) never logs or notifies.
    """
    if broker_qty_signed <= 0:
        short_note = f"a SHORT of {-broker_qty_signed} sh" if broker_qty_signed < 0 else "no position"
        msg = (f"{tag} {symbol}: broker shows {short_note} but the registry wanted to sell "
               f"{registry_qty} sh — SKIPPING, no order submitted (an exit must never create or "
               f"extend a short on the shared live account)")
        logger.warning(msg)
        if notify_fn:
            try: notify_fn(msg)
            except Exception as e: logger.error(f"{tag} {symbol}: exit-guard notify failed: {e}")
        return None
    if broker_qty_signed != registry_qty:
        direction = "only" if broker_qty_signed < registry_qty else "more than expected —"
        msg = (f"{tag} {symbol}: registry wanted to sell {registry_qty} sh but the broker holds "
               f"{direction} {broker_qty_signed} long — selling the broker's qty, not the registry's")
        logger.warning(msg)
        if notify_fn:
            try: notify_fn(msg)
            except Exception as e: logger.error(f"{tag} {symbol}: exit-guard notify failed: {e}")
    return broker_qty_signed


def get_signed_broker_qty(alpaca_client, symbol: str) -> int:
    """Signed live position qty for `symbol` on `alpaca_client`'s account (negative = short, 0 =
    flat/not found). Never raises -- an error is logged and treated as unknown-but-callers must fail
    CLOSED (0), never assume long, on a lookup failure."""
    try:
        for p in alpaca_client.get_open_positions():
            if p.get('symbol') == symbol:
                return int(float(p.get('qty', 0) or 0))
        return 0
    except Exception as e:
        logger.error(f"{symbol}: broker position lookup failed for the exit guard ({e}) — treating as flat, fail closed")
        return 0
