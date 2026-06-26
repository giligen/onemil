"""
Central catalog of `trades.exit_reason` values.

The DB column `persistence/database.py::trades.exit_reason` stores one of
the strings below on every closed/exited trade. Until 2026-06-12 these
were inline string literals scattered across `orb_engine.py`,
`trading_engine.py`, `macd_wave_engine.py`, `stop_monitor.py`, and
`backtest.py`. The audit (see DB query: 12 distinct values across 268
exit rows) found inconsistent naming across strategies (`tag_bb` vs
`stop_loss_market_fallback`), two ghost values with no current writer
(`sync_reconcile` x6 on 2026-04-02, `stop_loss_timeout` x1 on
2026-03-26), and a known-leak path (`unknown_exit`) that fired again
2026-06-11 with no Telegram alert.

This module is the single source of truth. New strategies + new code
paths MUST import from here. The string values are LOAD-BEARING:
production DB rows reference them and analytics SQL groups on them —
never rename a member without a backfill migration.

Grouping:
  Shared (any strategy):
    STOP_LOSS, TAKE_PROFIT, TRAIL_STOP, LOCK_STOP, FORCE_CLOSE,
    UNKNOWN_EXIT, POST_FILL_EXIT

  StopMonitor recovery branches (used by BF + ORB):
    STOP_LOSS_MARKET_FALLBACK, STOP_LOSS_BRACKET_SL_RACE,
    STOP_LOSS_UNCONFIRMED, STOP_LOSS_FALLBACK, EXHAUSTION_PARTIAL

  Bull flag specific:
    GAP_OVER_REJECTION, GAP_ADJUST_FAILED, THIN_LIQUIDITY_REJECT

  ORB specific (touchgo Rules M + D):
    TAG_BB, TAG_B1

  MACD wave specific:
    MACD_FLIP, BRACKET_EXIT, BRACKET_SL_TP, HARD_STOP, STOPMONITOR_EXIT

  Historical (no current writer; kept for DB-query compat):
    SYNC_RECONCILE, STOP_LOSS_TIMEOUT
"""
from __future__ import annotations

from enum import Enum
from typing import Optional


class ExitReason(str, Enum):
    """Stable string contract for the `trades.exit_reason` column.

    Inherits from `str` so members are usable wherever a string is
    expected (`f"{ExitReason.STOP_LOSS}"`, `{'exit_reason': ExitReason.X}`),
    while still being a typed enum for grep + IDE refactoring. Prefer
    `ExitReason.X.value` at DB-write sites for explicit clarity.
    """

    # ---- shared across strategies -------------------------------------
    STOP_LOSS = "stop_loss"
    """The watch's stop_price was hit and the position was sold via
    StopMonitor's marketable limit (or its escalation chain). The
    canonical exit for a losing trade."""

    TAKE_PROFIT = "take_profit"
    """The watch's take_profit_price was hit. Pre-trail bull flag mode;
    rare under current config since trailing_stop is shipped on."""

    TRAIL_STOP = "trail_stop"
    """Trailing stop tightened past the trade's last high, then triggered.
    The winning-trade exit path under the trailing_stop feature."""

    LOCK_STOP = "lock_stop"
    """ORB static-lock variant: after price touched +1.5R, the stop locked
    at +1R and was later hit. See docs/orb_rollout_plan.md and CLAUDE.md
    ORB section."""

    FORCE_CLOSE = "force_close"
    """End-of-day force close (15:45 ET for ORB; configurable per strategy).
    Catches positions that didn't hit stop / target / trail before the
    forced-flat deadline."""

    UNKNOWN_EXIT = "unknown_exit"
    """The trade was reconciled to a closed state via sync_positions /
    orphan-detect but no specific exit branch could attribute the price.
    Written by `trading_engine.py:_handle_unknown_exit_*` +
    `stop_monitor.py:2786` after every recovery path failed.
    **Operational signal**: a non-zero daily count of these rows is a
    bug. Yesterday's GLXG (2026-06-11) was the 5th historical occurrence
    and surfaced silently — see issue tracker for the alert proposal."""

    POST_FILL_EXIT = "post_fill_exit"
    """Exit triggered by a post-fill filter (BT: gap-fill check; BF live:
    thin-liquidity reject after fill). Used by both BT and live BF."""

    # ---- StopMonitor recovery branches (any StopMonitor-driven exit) --
    STOP_LOSS_MARKET_FALLBACK = "stop_loss_market_fallback"
    """The marketable-limit sell didn't fill within timeout → escalated
    to `close_position` market order, which filled. Distinct from plain
    STOP_LOSS so analytics can separate "clean stop" from "limit got
    stranded but recovery worked"."""

    STOP_LOSS_BRACKET_SL_RACE = "stop_loss_bracket_sl_race"
    """`close_position` raced with Alpaca's bracket SL leg and reported
    `40410000 position not found`. The bracket SL leg won the exit. Real
    fill price recovered from the SL leg's order status."""

    STOP_LOSS_UNCONFIRMED = "stop_loss_unconfirmed"
    """Last-resort path: limit poll AND market-close poll both timed out.
    Records `trigger_price` as a placeholder + WARNING log. Differs from
    UNKNOWN_EXIT in that we know it was a stop-loss attempt; only the
    fill price is uncertain. sync_positions reconciles afterwards."""

    STOP_LOSS_FALLBACK = "stop_loss_fallback"
    """Generic stop-loss recovery (older synonym; treated as a superset of
    the market_fallback / bracket_sl_race / unconfirmed branches for
    BF-only paths that haven't been split out yet)."""

    EXHAUSTION_PARTIAL = "exhaustion_partial"
    """Partial-position sell driven by exhaustion candle detection (large
    reversal candle after a big run). Not a full close — the remaining
    shares stay watched. See partial_exit_* DB columns."""

    # ---- Bull flag specific -------------------------------------------
    GAP_OVER_REJECTION = "gap_over_rejection"
    """BF rejected at fill time because the gap up was too large per the
    gap-cap filter in `trading_engine.py::_evaluate_entry_filters`."""

    GAP_ADJUST_FAILED = "gap_adjust_failed"
    """BF post-fill: the SL adjust call for a gap-up open failed to
    submit; the position was force-exited as a safety measure."""

    THIN_LIQUIDITY_REJECT = "thin_liquidity_reject"
    """BF post-fill: depth/spread degraded below the live-trading floor
    after fill → force-exit at market. `trading_engine.py:3632`."""

    # ---- ORB specific (touchgo filter rules) --------------------------
    TAG_BB = "tag_bb"
    """ORB touchgo Rule M: the breakout bar closed in the bottom half of
    its range (bb_close_pos < 0.5) → exit at next bar open. Catches
    failed breakouts within the first minute of trade life. See
    `trading/orb_touchgo_filter.py` + CLAUDE.md "Touchgo filter"."""

    TAG_B1 = "tag_b1"
    """ORB touchgo Rule D: the bar after entry reverted ≥0.75R below
    entry → exit at entry - 0.5R. Catches fast reversals."""

    # ---- MACD wave specific -------------------------------------------
    MACD_FLIP = "macd_flip"
    """MACD histogram flipped sign — momentum reversal exit signal."""

    BRACKET_EXIT = "bracket_exit"
    """MACD wave: Alpaca bracket (SL/TP) handled the exit — generic
    fallback when we can't classify which leg fired (no SL/TP child
    order status available)."""

    BRACKET_SL_TP = "bracket_sl_tp"
    """MACD wave: bracket-attributed exit, classified to either SL or TP
    branch but not split out further for this column."""

    HARD_STOP = "hard_stop"
    """MACD wave: hard stop-loss leg fired (distinct from a trail-stop)."""

    STOPMONITOR_EXIT = "stopmonitor_exit"
    """MACD wave: StopMonitor (not the bracket) closed the position."""

    # ---- Historical (no current writer; kept for query compat) --------
    SYNC_RECONCILE = "sync_reconcile"
    """**Historical only**. Written on 2026-04-02 (6 macd_wave rows)
    during the orphan-cleanup incident. No current writer in the
    codebase. New reconciliation flows should emit UNKNOWN_EXIT
    instead. Kept here so daily-summary SQL doesn't error on the 6
    archived rows."""

    STOP_LOSS_TIMEOUT = "stop_loss_timeout"
    """**Historical only**. Single row 2026-03-26 EEIQ BF. Superseded by
    STOP_LOSS_UNCONFIRMED + the test in `test_stop_exit_limit_buffer.py
    ::TestExitReasonPerBranch`. No current writer."""


# Categorization helpers ----------------------------------------------------

_ATTRIBUTED_EXITS = frozenset({
    ExitReason.STOP_LOSS.value,
    ExitReason.TAKE_PROFIT.value,
    ExitReason.TRAIL_STOP.value,
    ExitReason.LOCK_STOP.value,
    ExitReason.FORCE_CLOSE.value,
    ExitReason.STOP_LOSS_MARKET_FALLBACK.value,
    ExitReason.STOP_LOSS_BRACKET_SL_RACE.value,
    ExitReason.STOP_LOSS_FALLBACK.value,
    ExitReason.EXHAUSTION_PARTIAL.value,
    ExitReason.GAP_OVER_REJECTION.value,
    ExitReason.GAP_ADJUST_FAILED.value,
    ExitReason.THIN_LIQUIDITY_REJECT.value,
    ExitReason.TAG_BB.value,
    ExitReason.TAG_B1.value,
    ExitReason.MACD_FLIP.value,
    ExitReason.BRACKET_EXIT.value,
    ExitReason.BRACKET_SL_TP.value,
    ExitReason.HARD_STOP.value,
    ExitReason.STOPMONITOR_EXIT.value,
    ExitReason.POST_FILL_EXIT.value,
})

_NEEDS_RECONCILE = frozenset({
    ExitReason.UNKNOWN_EXIT.value,
    ExitReason.STOP_LOSS_UNCONFIRMED.value,
    ExitReason.SYNC_RECONCILE.value,       # historical
    ExitReason.STOP_LOSS_TIMEOUT.value,    # historical
})

_HISTORICAL = frozenset({
    ExitReason.SYNC_RECONCILE.value,
    ExitReason.STOP_LOSS_TIMEOUT.value,
})


def is_known(value: Optional[str]) -> bool:
    """True iff `value` matches a defined ExitReason member string.

    Use at every DB read / Telegram event where an unknown reason should
    be loudly flagged for catalog update (it usually means a new code
    path is emitting an undocumented string)."""
    if value is None:
        return False
    return value in {er.value for er in ExitReason}


def is_attributed(value: Optional[str]) -> bool:
    """True iff this exit has a clearly attributed trigger (we know WHY
    the position closed). False for UNKNOWN_EXIT, *_unconfirmed, and the
    historical reconcile values."""
    return value in _ATTRIBUTED_EXITS


def needs_reconcile(value: Optional[str]) -> bool:
    """True iff this exit was a leak / fallback that requires operator
    follow-up. The daily-summary monitor should surface these counts to
    Telegram so silent leaks (yesterday's GLXG) get caught next time."""
    return value in _NEEDS_RECONCILE


def is_historical_only(value: Optional[str]) -> bool:
    """True iff this value has NO current writer in the codebase.
    Useful for the AST-scan drift test (these strings must NOT appear as
    new inline literals — anyone reproducing the underlying behavior
    should emit the modern replacement instead)."""
    return value in _HISTORICAL
