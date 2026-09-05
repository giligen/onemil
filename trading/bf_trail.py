"""Shared bull-flag R-trail exit math — single source of truth for BT + LIVE.

Imported by BOTH:
  - backtest.py  TradeSimulator.simulate()       (1-min bar loop)
  - trading/stop_monitor.py  _maybe_ratchet_from_bar_high()  (closed-bar path)

Why this module exists (2026-09-05 BF trail unification):
  The live StopMonitor and the BT TradeSimulator carried two independent
  copies of the R-trail (arm at +activate_at_r, ratchet highest - trail_r×R).
  They drifted on THREE money-relevant semantics and the reference book
  ($198K/19mo) was built on the wrong side of each:

    1. R basis   — live used plan-R (planned_entry - planned_stop; README
                   Bug 5, BT-validated 2026-05-08) while the cache builder
                   simulated fill-R (`use_planned_r` was never wired from
                   config). CWVX 2026-08-03: plan-R R=$0.21 → live trail
                   14.75 hit 9:57 (+$313); fill-R R=$0.46 → BT rode to
                   13:32 (+$2,381 at BT sizing). Same trade, two specs.
    2. Ratchet source — live ratcheted the R-trail on every TICK (ratchet-
                   then-check inside one tick); BT ratchets on CLOSED-BAR
                   highs and checks the NEXT bar's low against the stop
                   (check-then-ratchet). Pct trails were already bar-only
                   (BOBS 5/8 fix); the R-trail never got the same fix.
    3. Vol-guard bar — BT tested the TRIGGERING bar's own volume (known
                   only after the bar closes = lookahead); live tests the
                   PREVIOUS closed bar. Causal side wins: previous bar.

  The contract now (both sides, by construction):
    * R baseline/unit come from `r_baseline_and_unit()` with ONE config
      knob: `trading.trailing_stop.r_basis` = 'plan' (default) | 'fill'.
    * Trail state advances ONLY on closed 1-min bars via `arm_and_ratchet()`.
      A bar's high can never stop the trade out inside that same bar —
      the stop it produces is live from the NEXT bar on.
    * The entry bar contributes nothing (BT loop starts at entry+1; live
      skips bar events whose start < end of the fill minute).
    * Trail-stop volume confirmation reads the PREVIOUS closed bar.

  Parity is enforced by tests/test_bf_trail.py (same bar tape → identical
  stop path in TradeSimulator and StopMonitor, plus the CWVX golden day).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

R_BASIS_PLAN = 'plan'
R_BASIS_FILL = 'fill'
VALID_R_BASES = (R_BASIS_PLAN, R_BASIS_FILL)
DEFAULT_R_BASIS = R_BASIS_PLAN


def normalize_r_basis(value) -> str:
    """Return a validated r_basis string ('plan' | 'fill').

    Raises ValueError on anything else — a misspelt config key must break
    loudly, not silently fall back to one side of the parity contract.
    """
    v = str(value or DEFAULT_R_BASIS).strip().lower()
    if v not in VALID_R_BASES:
        raise ValueError(
            f"trading.trailing_stop.r_basis must be one of {VALID_R_BASES}, got {value!r}"
        )
    return v


def r_baseline_and_unit(
    planned_entry: float,
    planned_stop: float,
    fill_price: float,
    fill_stop: float,
    r_basis: str = DEFAULT_R_BASIS,
) -> Tuple[float, float]:
    """Return (r_baseline, r_unit) the R-trail measures gains from.

    'plan': baseline = planned breakout level, unit = planned_entry - planned_stop.
            Slippage-immune: the +2R arm gate and the 1R trail width are the
            SETUP's, not the fill's (README Bug 5).
    'fill': baseline = fill price, unit = fill - stop (legacy BT default).

    Falls back to fill-R when the planned numbers are unusable (<= 0 or a
    non-positive unit) — the caller is expected to log that fallback.
    """
    basis = normalize_r_basis(r_basis)
    if basis == R_BASIS_PLAN:
        unit = planned_entry - planned_stop
        if planned_entry > 0 and unit > 0:
            return float(planned_entry), float(unit)
    return float(fill_price), float(fill_price - fill_stop)


@dataclass
class TrailStep:
    """Result of feeding one closed bar to `arm_and_ratchet`."""
    highest: float
    stop: float
    trailing_active: bool
    armed_now: bool = False
    ratcheted: bool = False
    r_gain: float = 0.0


def arm_and_ratchet(
    bar_high: float,
    highest_since_entry: float,
    current_stop: float,
    trailing_active: bool,
    r_baseline: float,
    r_unit: float,
    activate_at_r: float,
    trail_r: float,
) -> TrailStep:
    """Advance R-trail state with ONE closed bar's high.

    Pure function; monotone (highest and stop only move up); idempotent
    (re-feeding the same bar changes nothing). Order of operations is the
    BT's: raise highest → arm if +activate_at_r reached → ratchet stop to
    highest − trail_r×R when armed. The returned stop applies to bars AFTER
    this one — callers must have evaluated this bar's low against the
    PREVIOUS stop before calling.

    trail_r <= 0 or r_unit <= 0 means "no R-trail": state passes through.
    """
    highest = max(float(highest_since_entry), float(bar_high))
    stop = float(current_stop)
    active = bool(trailing_active)
    armed_now = False
    ratcheted = False
    r_gain = 0.0

    if trail_r <= 0 or r_unit <= 0:
        return TrailStep(highest, stop, active, armed_now, ratcheted, r_gain)

    r_gain = (highest - r_baseline) / r_unit
    if not active and r_gain >= activate_at_r:
        active = True
        armed_now = True
    if active:
        new_stop = highest - r_unit * trail_r
        if new_stop > stop:
            stop = new_stop
            ratcheted = True
    return TrailStep(highest, stop, active, armed_now, ratcheted, r_gain)


def entry_bar_excluded(bar_start_ts: float, skip_exits_until_ts: float) -> bool:
    """True when a closed bar belongs to the entry minute and must not
    advance trail state (BT parity: the simulate loop starts at entry+1).

    `skip_exits_until_ts` is the Unix time at the END of the fill minute
    (0 = no exclusion). A bar whose START is before that boundary is the
    entry bar (or earlier) → excluded.
    """
    if skip_exits_until_ts <= 0:
        return False
    return bar_start_ts < skip_exits_until_ts
