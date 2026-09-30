"""HOD-break failure-short: pure order-mechanics logic (paper trading, behind
`config.yaml hod_break.failure_short.enabled` — default OFF/telemetry_only, docs/hod_failure_short_spec_20260930.md).

Rule: a HOD-break long we filled ourselves may be a failed breakout. If a signal model says the
breakout is likely to fail (P >= tau) at the close of the bar AFTER our fill, and every rail
passes, flip the position — sell 2x the long's shares (closes the long, opens a short of the
same size) — with a protective stop-buy at the day's high-so-far + 1c and a target buy-limit
back at the long's own stop level.

This module owns the MECHANISM only: rails, sizing, prices, caps, the day kill and the ledger
row shape, as plain functions with no I/O — CLAUDE.md "ONE spec ... mechanism + evidence +
explicit code" applied to this overlay. The signal itself (the probability) is computed by a
separate model module (trading/hod_failure_features.py, owned elsewhere) and handed in to the
engine as a `signal_fn(symbol, bars, arm_context) -> float | None` callable; this module never
imports or evaluates a model, and never touches the broker or the DB — trading/hod_break_engine.py
is the only caller and owns all I/O.
"""
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# SSR proxy (spec-fixed, not a config knob): never short a name already down >= 10% from the
# prior close intraday — that is exactly Reg SHO short-sale-restriction territory and this book
# has no locate/SSR feed of its own.
SSR_FLOOR_PCT = 0.90
# Protective stop sits 1c above the day's high so far (ratchets up with every new high).
STOP_TICK = 0.01


@dataclass
class FailureShortConfig:
    """hod_break.failure_short.* (config.yaml), parsed once by config.hod_break_cfg. Every field
    has a FAIL-SAFE default (disabled + telemetry_only) so a missing key can never silently arm
    an order path — the 2026-09-25 whitelist lesson (a dropped key left the engine in the wrong
    mode with no error)."""
    enabled: bool = False
    telemetry_only: bool = True
    tau: float = 0.6
    risk_usd: float = 150.0
    max_concurrent: int = 3
    max_per_day: int = 8
    day_kill_r: float = -5.0
    require_etb: bool = True
    entry_bar_offset: int = 2
    allow_fresh_short: bool = False
    ledger_path: str = 'logs/hod_failure_short_ledger.csv'


def rails_reason(*, shortable: bool, easy_to_borrow: bool, require_etb: bool, price: float,
                  prior_close: float, open_count: int, submitted_today: int, day_realized_r: float,
                  cfg: FailureShortConfig) -> Optional[str]:
    """First rail that blocks this short, or None if every rail passes. Order matches the spec:
    borrow -> SSR -> concurrency cap -> day cap -> day kill. `prior_close <= 0` (unknown) fails
    CLOSED as SSR-blocked — an unpriceable SSR check is never treated as a pass."""
    if not shortable:
        return 'not_shortable'
    if require_etb and not easy_to_borrow:
        return 'not_etb'
    if prior_close <= 0:
        return 'ssr_unknown_prior_close'
    if price < SSR_FLOOR_PCT * prior_close:
        return 'ssr_blocked'
    if open_count >= cfg.max_concurrent:
        return 'max_concurrent'
    if submitted_today >= cfg.max_per_day:
        return 'max_per_day'
    if day_realized_r <= cfg.day_kill_r:
        return 'day_kill'
    return None


def reversal_sell_qty(long_shares: int) -> int:
    """Shares for the marketable SELL that flips the long into a short: 2x the long's own shares
    (closes the long, opens a short of the SAME size as the long)."""
    return 2 * int(long_shares)


def resulting_short_qty(long_shares: int) -> int:
    """Shares left short once the reversal sell above fills — always equal to the long's shares."""
    return int(long_shares)


def fresh_short_qty(risk_usd: float, entry_price: float, stop_price: float) -> int:
    """Risk-based size for the `allow_fresh_short` path only: the long was already closed before
    the signal bar (stopped out inside minute 1), so there is nothing to reverse — this would be
    a brand-new short. Off by default (`allow_fresh_short: false`); first version trades only
    reversals of our own fills."""
    risk_per_share = stop_price - entry_price
    if risk_per_share <= 0:
        logger.warning(f"fresh_short_qty: non-positive risk_per_share ({risk_per_share}) for "
                        f"entry={entry_price} stop={stop_price} — returning 0 shares")
        return 0
    return max(0, int(risk_usd // risk_per_share))


def protective_stop_price(day_high_so_far: float) -> float:
    """Stop-buy trigger for the short: the day's high so far + 1c."""
    return round(day_high_so_far + STOP_TICK, 2)


def target_buy_price(long_stop_price: float) -> float:
    """Target buy-limit for the short: back at the long's OWN stop level — the level that would
    have stopped the long out of the trade we just reversed."""
    return round(long_stop_price, 2)


def should_evaluate_signal(current_minute: int, fill_minute: int) -> bool:
    """True once the bar immediately after the fill bar (fill+1) has CLOSED. `>=` (not `==`) so a
    caller that polls rather than receiving an exact per-minute event still fires exactly once,
    driven by the engine's own state transition rather than by this predicate being edge-triggered."""
    return current_minute >= fill_minute + 1


def should_submit(current_minute: int, fill_minute: int, entry_bar_offset: int) -> bool:
    """True once the submit bar (fill + entry_bar_offset, default fill+2) has closed."""
    return current_minute >= fill_minute + entry_bar_offset


def ledger_row(*, ts: str, date: str, symbol: str, fill_bar: int, p: Optional[float],
                cfg: FailureShortConfig, rails: Optional[str], would_be_qty: int,
                would_be_stop: float, would_be_target: float) -> dict:
    """One row for logs/hod_failure_short_ledger.csv. Written on every evaluation regardless of
    telemetry_only (a superset of the spec's telemetry_only requirement — always safe, always
    useful forward measurement, matches the 'verbose progress' rule)."""
    passed = bool(p is not None and p >= cfg.tau and rails is None)
    return {
        'date': date, 'ts': ts, 'symbol': symbol, 'fill_bar': fill_bar,
        'p': '' if p is None else round(p, 4), 'tau': cfg.tau, 'rails_reason': rails or '',
        'passed': passed, 'would_be_qty': would_be_qty, 'would_be_stop': would_be_stop,
        'would_be_target': would_be_target, 'telemetry_only': cfg.telemetry_only,
    }


def short_pattern_data(*, long_trade_id: Optional[int], p: float, tp_leg_id: str, sl_leg_id: str,
                        target: float, stop: float) -> dict:
    """pattern_data JSON for the NEW short's trades row — `mechanism` is the field the spec names
    explicitly, so any downstream query can isolate failure-short rows from ordinary HOD longs."""
    return {
        'book': 'hod_break', 'mechanism': 'failure_short', 'reversed_long_trade_id': long_trade_id,
        'signal_p': p, 'tp_leg_id': tp_leg_id, 'sl_leg_id': sl_leg_id, 'target': target, 'stop': stop,
    }
