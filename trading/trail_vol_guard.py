"""Shared volume-confirmed trail exit guard (Experiment D, 2026-04-17).

Single source of truth imported by both:
  - backtest.py TradeSimulator.simulate() — exit-bar evaluation
  - trading/stop_monitor.py — live tick + poll paths

Rationale: when a trailing stop triggers on a bar whose volume is below
flag-average × min_ratio, the price crossing is most likely passive drift
(pullback without active selling). Skip the exit on that bar; the trail
stop stays put, next bar re-evaluates. The initial hard stop (pre-trailing)
is never skipped — this only filters trail exits.

BT empirical (2025 + Q1 2026 holdout, TTF on, r=1.0):
    2025: +$3,764 (+5.8%), same trade count, same DD, same avg loss
    Q1 2026: +$1,836 (+18%)

Semantics are intentionally conservative: missing baseline → never skip.
"""
from __future__ import annotations
from typing import Optional


def should_skip_trail_exit_on_low_vol(
    bar_volume: Optional[int],
    flag_avg_volume: Optional[float],
    min_vol_ratio: float,
) -> bool:
    """Return True if the stop-triggering bar is low-volume drift.

    Callers pass this bar's volume + the setup's flag-average volume (baseline
    for "normal" activity during the pattern) + the configured min ratio.

    Return value semantics:
        True  → skip the exit on this bar (treat as noise; hold position)
        False → fire the exit normally

    Safe defaults:
        - flag_avg_volume None, NaN, or <= 0: return False (no baseline known;
          never skip — fall back to naive trail behavior)
        - bar_volume None: treat as 0 (skip if min_vol_ratio > 0, matching a
          "no trading on this bar" scenario)
        - min_vol_ratio <= 0: returns False (ratio effectively disables the
          check; any volume >= 0 passes)
    """
    if flag_avg_volume is None:
        return False
    try:
        baseline = float(flag_avg_volume)
    except (TypeError, ValueError):
        return False
    if baseline <= 0:
        return False
    try:
        ratio = float(min_vol_ratio)
    except (TypeError, ValueError):
        return False
    if ratio <= 0:
        return False
    vol = bar_volume if bar_volume is not None else 0
    try:
        vol = int(vol)
    except (TypeError, ValueError):
        vol = 0
    return vol < baseline * ratio
