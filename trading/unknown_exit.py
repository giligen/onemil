"""The ONE spec for an unattributed exit — D3 FIX 7.

`research/fuckup_audit/D3_exec/REPORT.md` §M7: four production rows
(BDMD bull_flag 2026-03-20; NPT / FBYD / SVRN macd_wave 2026-03-30) were
booked with ``exit_price == fill_price`` and ``pnl == 0.00`` **exactly**.
They were not breakeven trades. They were trades whose exit fill we never
found, written down as flat so the reconcile loop would stop revisiting
them. The tape says otherwise:

    row                 booked   tape at the exit minute            hole
    BDMD 03-20 15:23    $0.00    bar 2.1408-2.2000                 -$491
    NPT  03-30 16:53    $0.00    16:52 close 4.96 (entry 5.49)   -$4,818
    FBYD 03-30 16:37    $0.00    bar 11.34-11.48                   -$877
    SVRN 03-30 16:53    $0.00    last print 16:46 7.23             -$614

Between -$1,684 (floor) and -$6,800 of realized loss, invisible to every
book that reads ``trades.pnl``.

The contract for "the position is gone but we cannot attribute the fill"
already existed — ``StopExitEvent(confirmed=False)`` →
``order_status='exit_pending_verification'``, no ``exit_price``, no
``exited_at``, no ``pnl`` (``trading/stop_monitor.py::build_exit_update``,
``tests/test_stop_exit_unconfirmed.py``). ``unknown_exit`` simply did not
use it. This module makes that contract callable by any writer, so the
rule cannot be re-implemented differently in a fifth place.

A row left pending is NOT a loose end: `trading/orphan_reconciler.py`
owns it (``PENDING_VERIFICATION_STATUS`` is in its ownership test), it
stays in ``Database.get_open_trades`` (``_ACTIVE_ORDER_STATUSES``), and
the daily green check HARD-fails on it
(``scripts/report_common.py`` — "no rows stuck in
exit_pending_verification"). Loud beats invisible.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from trading.exit_reasons import ExitReason

logger = logging.getLogger(__name__)

PENDING_VERIFICATION_STATUS = 'exit_pending_verification'


def build_unknown_exit_update(
    trigger_price: Optional[float] = None,
) -> Dict[str, Any]:
    """The DB payload for an exit we cannot attribute.

    Deliberately NOT a full exit payload. What is absent is the point:

      * no ``exit_price``  — we never observed a fill,
      * no ``exited_at``   — we do not know when it left,
      * no ``pnl`` / ``pnl_pct`` — a fabricated $0 is the defect.

    ``order_status`` goes to ``exit_pending_verification`` so the orphan
    reconciler picks the row up and the daily green check fails loudly
    until a human or the sweep resolves it.

    Args:
        trigger_price: Optional last-known price, recorded as
            ``exit_trigger_price`` for forensics ONLY. It is never the
            exit price and never feeds P&L.

    Returns:
        A dict for ``Database.update_trade``.
    """
    upd: Dict[str, Any] = {
        'exit_reason': ExitReason.UNKNOWN_EXIT.value,
        'order_status': PENDING_VERIFICATION_STATUS,
    }
    if trigger_price is not None:
        try:
            tp = float(trigger_price)
        except (TypeError, ValueError):
            logger.warning(
                "build_unknown_exit_update: non-numeric trigger_price "
                f"{trigger_price!r} — recording no trigger price"
            )
            return upd
        if tp > 0:
            upd['exit_trigger_price'] = tp
    return upd


def sweep_unknown_exits(
    db: Any,
    older_than_session: bool = True,
    resolver: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> List[Dict[str, Any]]:
    """STUB — the daily resolver for rows left in pending-verification.

    **Not built.** D3 FIX 7 ships the half that stops the bleeding: an
    unattributable exit no longer fabricates a price and a P&L. Resolving
    those rows from Alpaca order history the next morning is a separate,
    scheduled job and is deliberately out of this change's blast radius
    (it submits nothing, but it WRITES P&L, so it needs its own
    evidence and its own review).

    The pieces it will compose already exist and must be reused rather
    than rewritten:

      * ``TradingEngine._recover_exit_from_order_history`` — the lookup
        and the reason classification (GLXG 2026-06-11 fix),
      * ``trading/orphan_reconciler.py`` — ownership of a row whose
        broker state disagrees with ours,
      * ``Database.get_open_trades`` — returns pending rows already.

    Until then the operator-facing signal is the green check's HARD fail
    on stuck ``exit_pending_verification`` rows, which is exactly the
    alarm the fabricated $0 used to silence.

    Args:
        db: Database handle (unused by the stub).
        older_than_session: Whether to restrict to rows older than one
            session (unused by the stub).
        resolver: Injection point for the order-history lookup (unused
            by the stub).

    Returns:
        Always ``[]`` — "resolved nothing", which is the truth.
    """
    logger.warning(
        "sweep_unknown_exits: NOT IMPLEMENTED (D3 FIX 7 stub) — rows in "
        f"'{PENDING_VERIFICATION_STATUS}' are resolved by the orphan "
        "reconciler and surfaced by the daily green check's HARD fail, "
        "not by this sweep. Nothing was resolved."
    )
    return []
