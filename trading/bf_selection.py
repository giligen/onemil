"""Shared Bull Flag candidate screen — BT and LIVE import THIS.

2026-08-29 (owner: "we want a money machine aligned with BT"): the drift
quantification (research/bf_drift_quantification_aug2026.md) found only
~50% selection overlap between the live scanner and the BT cache build.
The per-symbol DECISION gates were already shared modules; the drift
lived in the candidate SCREEN, which was implemented twice with
different semantics:

- BT (batch_backtest.find_big_movers): full-day daily-bar range >= thr,
  plus a direction gate `gap_up OR (range AND close > open)` and a price
  band on the CLOSING price — both EOD facts a live scanner can never
  evaluate mid-session (lookahead). Result: spike-then-close-red days
  were BT-invisible while live traded them (IREZ 7/31, LUNL 8/13,
  PFSA 8/19 class).
- LIVE (scanner criteria): streaming max(gap%, range%) >= thr with no
  close-color condition.

This module is the single CAUSAL predicate both sides call. Alignment
decisions (deliberate-rules ledger):
1. `close > open` REMOVED from the screen. It was belt-and-braces
   against wide-range crash days — but the shared BullFlagDetector is
   the real filter (a crash prints no pole); the screen term only
   manufactured live-vs-BT divergence. BT-side effect: red-close spike
   days now enter the cache (screen widens; the detector + Stage-2
   pre-entry threshold still gate entries causally).
2. Direction still requires SOME upside evidence: gap-up (high vs
   prev_close) OR the range test — both monotone within the day, so the
   EOD evaluation in the cache build is a superset of what live sees
   streaming; no live-only class remains at the screen level.
3. The price band's reference price stays caller-supplied (BT: daily
   close as tradeable-range proxy; LIVE: current price). Residual
   documented divergence — the 1-30 band is wide and this is a proxy
   choice, not a semantics fork.

Cache regen note: cache rows built before this change were screened
WITH the direction gate — a clean regen (owner-gated; NEVER overwrite
data/bull_flag_cache_e50_x30.csv in place) is required before absolute
Stage-2 numbers on red-close days are meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class MoverVerdict:
    qualifies: bool
    reason: str          # 'gap_up' | 'range' | rejection reason


def mover_day_qualifies(
    *,
    high: float,
    low: float,
    prev_close: Optional[float],
    price_ref: float,
    volume: float,
    threshold_pct: float,
    price_min: float = 0.0,
    price_max: float = 0.0,
    min_dollar_volume: float = 0.0,
) -> MoverVerdict:
    """THE candidate screen. All inputs are causal at evaluation time
    for the live caller (running high/low/price) and EOD aggregates for
    the BT cache build (supersets of any intraday snapshot — monotone).

    threshold_pct is a FRACTION-style percent (e.g. 0.10 for 10%) to
    match the historical find_big_movers call shape.
    """
    if low is None or low <= 0 or high is None:
        return MoverVerdict(False, 'no_bar')
    range_move = (high - low) / low
    gap_up = False
    if prev_close and prev_close > 0:
        gap_up = (high - prev_close) / prev_close >= threshold_pct
    # Threshold + direction in one test: gap-up (upside vs prev close)
    # OR in-day range — the old `close > open` direction term is
    # deliberately ABSENT (EOD lookahead — see module docstring).
    if range_move < threshold_pct and not gap_up:
        return MoverVerdict(False, 'below_threshold')
    if price_min > 0 and price_ref < price_min:
        return MoverVerdict(False, 'price_below_min')
    if price_max > 0 and price_ref > price_max:
        return MoverVerdict(False, 'price_above_max')
    if min_dollar_volume > 0 and price_ref * (volume or 0) \
            < min_dollar_volume:
        return MoverVerdict(False, 'dollar_volume')
    kind = 'gap_up' if gap_up else 'range'
    return MoverVerdict(True, kind)


def intraday_qualifies(
    *,
    gap_pct: float,
    range_pct: float,
    threshold_pct_points: float,
) -> bool:
    """LIVE streaming form: max(gap%, range%) >= threshold, in PERCENT
    POINTS (scanner convention, e.g. 15.0). Kept as a thin shim so both
    conventions route through one file and the equivalence is testable:
    intraday_qualifies(g, r, T) == the screen's threshold test with
    fraction T/100 given high/low/prev_close producing the same g, r."""
    return max(gap_pct or 0.0, range_pct or 0.0) >= threshold_pct_points
