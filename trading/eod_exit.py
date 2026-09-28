"""
trading/eod_exit.py

Shared end-of-day exit-pricing logic for the ORB and HOD engines. Config key
`eod_exit_mode` (orb.yaml `exit.eod_exit_mode`, hod_break cfg `eod_exit_mode`),
default 'market' — pure functions only (no Alpaca client, no engine state), so
both engines get IDENTICAL mode semantics by construction and this module is
unit-testable without either engine (ONE spec, CLAUDE.md "no accidental
behaviour").

Three modes:
  market            (default) — today's behaviour: ORB close_position(), HOD
                     marketable limit at ref*0.99/0.97. resolve_mode() returns
                     'market' and callers keep their EXISTING order call
                     untouched — eod_exit_mode absent from config (it is not
                     added to orb.yaml/hod cfg by this change) == byte-identical.
  limit_then_market — rest a limit sell at the NBBO mid captured at the exit
                     instant; if unfilled after `eod_limit_timeout_s` (default
                     20s) cancel and re-submit as a plain market order.
  moc               — submit a market-on-close order (TimeInForce.CLS). Per the
                     installed alpaca-py's OWN TimeInForce docstring
                     (alpaca.trading.enums.TimeInForce, verified 2026-09-28):
                     "CLS orders submitted after 3:50pm but before 7:00pm ET
                     will be rejected." So MOC_CUTOFF_ET = 15:50 ET. Past the
                     cutoff, resolve_mode() downgrades to 'limit_then_market'
                     and returns a WARNING string — the caller MUST log it,
                     never submit CLS past 15:50 ET.

PARITY CAVEAT: `moc` fills at the 16:00 ET closing auction. The ORB backtest
force-closes at 15:45 ET (orb.yaml exit.force_close_time_et) and the HOD
backtest flattens at 15:55 ET (flat_minute) — neither backtest models a 16:00
exit. A `moc` fill is NOT comparable to either book's backtested P&L until the
backtest itself is re-verified with a close-price exit (see
docs/eod_exit_modes_20260928.md). This module is a live/dry MEASUREMENT
option — cost vs the bid at the exit instant, cost vs the official close —
never a backtest-parity rule.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time as dtime
from typing import Optional

logger = logging.getLogger(__name__)

MARKET = 'market'
LIMIT_THEN_MARKET = 'limit_then_market'
MOC = 'moc'
EOD_EXIT_MODES = (MARKET, LIMIT_THEN_MARKET, MOC)

DEFAULT_EOD_EXIT_MODE = MARKET
DEFAULT_LIMIT_TIMEOUT_S = 20.0

# Alpaca's own TimeInForce.CLS docstring (alpaca.trading.enums, verified
# locally against the installed alpaca-py 2026-09-28): "CLS orders submitted
# after 3:50pm but before 7:00pm ET will be rejected." 10 minutes before the
# 4pm close.
MOC_CUTOFF_ET = dtime(15, 50)

# Pricing-method tags written to trades.exit_pricing_method (existing column,
# shared with StopMonitor's 'stop_loss'/'quote_tight'/... tags — these are
# EOD-specific and namespaced eod_* so the two families never collide in a
# report).
PM_EOD_MARKET = 'eod_market'
PM_EOD_LIMIT = 'eod_limit_mid'
PM_EOD_LIMIT_MKT_FALLBACK = 'eod_limit_mkt_fb'
PM_EOD_MOC = 'eod_moc'
PM_EOD_MOC_CUTOFF_LIMIT = 'eod_moc_cut_limit'
PM_EOD_MOC_CUTOFF_MKT_FB = 'eod_moc_cut_mkt_fb'


@dataclass(frozen=True)
class ResolvedEodExit:
    """Result of resolve_mode(): the mode to actually submit plus any
    downgrade reason the caller must log."""
    mode: str                       # one of EOD_EXIT_MODES — always actionable
    configured_mode: str            # what the config asked for, pre-downgrade
    moc_cutoff_fallback: bool = False
    warning: Optional[str] = None   # non-None => caller MUST log at WARNING


def resolve_mode(configured_mode: str, now_et: datetime) -> ResolvedEodExit:
    """Resolve the configured eod_exit_mode against the clock.

    Unknown mode strings fall back to 'market' with a WARNING (fail safe —
    never silently pick an untested order type). 'moc' at/after MOC_CUTOFF_ET
    falls back to 'limit_then_market' with a WARNING (never submitted past
    Alpaca's documented CLS cutoff).
    """
    raw = configured_mode or DEFAULT_EOD_EXIT_MODE
    mode = raw.strip().lower()
    if mode not in EOD_EXIT_MODES:
        warning = (
            f"eod_exit.resolve_mode: unknown eod_exit_mode={configured_mode!r} "
            f"(expected one of {EOD_EXIT_MODES}) — falling back to {DEFAULT_EOD_EXIT_MODE!r}"
        )
        logger.warning(warning)
        return ResolvedEodExit(mode=DEFAULT_EOD_EXIT_MODE, configured_mode=configured_mode, warning=warning)

    if mode == MOC and now_et.time() >= MOC_CUTOFF_ET:
        warning = (
            f"eod_exit.resolve_mode: moc requested at {now_et.time().isoformat(timespec='seconds')} ET is past "
            f"Alpaca's CLS cutoff ({MOC_CUTOFF_ET.isoformat(timespec='minutes')} ET) — falling back to {LIMIT_THEN_MARKET!r}"
        )
        logger.warning(warning)
        return ResolvedEodExit(mode=LIMIT_THEN_MARKET, configured_mode=configured_mode,
                                moc_cutoff_fallback=True, warning=warning)

    return ResolvedEodExit(mode=mode, configured_mode=configured_mode)


def quote_mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    """NBBO mid at the exit instant. None if either side is missing/crossed
    (caller must fall back — never submits a limit at a fabricated price)."""
    if not bid or not ask or bid <= 0 or ask <= 0 or ask < bid:
        return None
    return round((bid + ask) / 2.0, 4)


def pricing_method(resolved: ResolvedEodExit, phase: str) -> str:
    """Map (resolved mode, phase) to the trades.exit_pricing_method tag.

    phase: 'primary' (the order actually submitted at the exit instant) or
    'fallback' (limit_then_market's post-timeout market re-submit).
    """
    if resolved.mode == MARKET:
        return PM_EOD_MARKET
    if resolved.mode == MOC:
        return PM_EOD_MOC
    # limit_then_market — either configured directly or a moc cutoff downgrade
    if resolved.moc_cutoff_fallback:
        return PM_EOD_MOC_CUTOFF_MKT_FB if phase == 'fallback' else PM_EOD_MOC_CUTOFF_LIMIT
    return PM_EOD_LIMIT_MKT_FALLBACK if phase == 'fallback' else PM_EOD_LIMIT


def build_eod_exit_telemetry(
    *,
    method: str,
    quote_bid: float = 0.0,
    quote_ask: float = 0.0,
    exit_price: Optional[float] = None,
    submitted_at_epoch: Optional[float] = None,
    filled_at_epoch: Optional[float] = None,
    reference_price: Optional[float] = None,
) -> dict:
    """Build the trades-table telemetry dict for one EOD exit fill — same
    column names the ORB StopMonitor path already writes (see
    trading/trading_engine.py ~L2758-2767): exit_pricing_method,
    exit_quote_bid, exit_quote_ask, exit_price, exit_fill_latency_ms,
    exit_slippage.

    `exit_slippage` = reference_price - exit_price (positive = sold below the
    reference = cost). reference_price is the NBBO mid for limit_then_market;
    for market/moc callers should pass the pre-submit bid (Alpaca prices a
    moc fill at the closing auction — there is no limit to compare against,
    so the bid at the exit instant is the honest reference, per RESULT_1443's
    10-12bps convention). The EOD report fills in the vs-official-close
    column separately (fetched after the close) — not this function's job.
    """
    out = {
        'exit_pricing_method': method,
        'exit_quote_bid': quote_bid or None,
        'exit_quote_ask': quote_ask or None,
    }
    if exit_price is not None:
        out['exit_price'] = exit_price
    if submitted_at_epoch is not None and filled_at_epoch is not None:
        out['exit_fill_latency_ms'] = max(0.0, (filled_at_epoch - submitted_at_epoch) * 1000.0)
    if reference_price is not None and exit_price is not None:
        out['exit_slippage'] = round(reference_price - exit_price, 4)
    return out
