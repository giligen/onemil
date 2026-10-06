"""Broker-truth sell-qty guard, shared by every live exit path (HOD stop/target/force-close/EOD/
dead-man; mirrored anywhere else a registry qty is sold). 2026-09-25 CDNA incident: HOD's registry
still showed 57 sh open after StopMonitor's own stop-loss had already flattened the broker position
(the drain that marks the registry closed raced a restart and lost the event); force_close_all sold
the registry's 57 sh again at 15:55 with no broker check, taking the shared live account SHORT.

ONE helper, never re-implemented per call site: every exit submission must clamp to what the broker
actually shows long for the symbol, and must NEVER submit a sell that can create or extend a short.
"""
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)
_LOOKUP_WARN_TS: Dict[str, float] = {}      # symbol -> last 'positions lookup failed' WARNING time (throttle)


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


def get_signed_broker_qty(alpaca_client, symbol: str) -> Optional[int]:
    """Signed live position qty for `symbol` on `alpaca_client`'s account (negative = short, 0 = the
    symbol is genuinely ABSENT from the positions list). Never raises.

    Returns None when the lookup itself failed (429 / timeout / API error): the qty is UNKNOWN, which is
    NOT the same as flat. 2026-10-02 VIRT / review B1: the old 'fail closed to 0' made force_close_all pop
    the position and mark it exit_pending_verification with no sell. Every caller must treat None as
    'unknown -- keep the position registered and retry', never as flat and never as long."""
    try:
        for p in alpaca_client.get_open_positions():
            if p.get('symbol') == symbol:
                return int(float(p.get('qty', 0) or 0))
        return 0
    except Exception as e:
        now = time.time()
        if now - _LOOKUP_WARN_TS.get(symbol, -1e9) >= 60.0:       # one WARNING per symbol per minute (a flatten retries every tick)
            _LOOKUP_WARN_TS[symbol] = now
            logger.warning(f"{symbol}: broker position lookup failed for the exit guard ({e}) — qty UNKNOWN (None), "
                           f"callers must retry and keep the position registered, never assume flat")
        return None


def get_our_buy_fill_qty(alpaca_client, symbol: str, coid_prefix: str, since_utc) -> Optional[int]:
    """Shares of `symbol` BOUGHT by this book since `since_utc`: the sum of filled_qty over BUY orders whose
    client_order_id starts with `coid_prefix` (the orders API, status ALL). Review B2: the broker's position
    in a symbol is the account total, which on a shared account includes the owner's manual shares -- this is
    OUR side of it. Sells are not subtracted here (OCO / replaced legs lose our prefix); callers net their
    own registry `closed_qty`. Returns None on any error (logged WARNING) -- callers fall back to the registry
    quantity, never to the broker total."""
    try:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=[symbol], after=since_utc, limit=500)
        orders = alpaca_client.trading_client.get_orders(filter=req)
    except Exception as e:
        logger.warning(f"{symbol}: our-fills lookup (orders API) failed ({e}) — caller falls back to the registry qty")
        return None
    total = 0
    for o in orders or []:
        get = (lambda k, o=o: o.get(k)) if isinstance(o, dict) else (lambda k, o=o: getattr(o, k, None))
        side = get('side'); side = str(getattr(side, 'value', side) or '').lower()
        if side != 'buy' or not str(get('client_order_id') or '').startswith(coid_prefix):
            continue
        try: total += int(float(get('filled_qty') or 0))
        except (TypeError, ValueError): continue
    return total
