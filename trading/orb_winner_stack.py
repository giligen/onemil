"""ORB "winner stack" shared physics — ATR14 stop-floor (SZ1) + scale-out.

Shipped 2026-08-22 behind TWO independent default-OFF flags
(orb.yaml::exit.atr_stop_floor / exit.scale_out; env kills ORB_ATR_FLOOR=0 /
ORB_SCALE_OUT=0). Parity by construction: this module is imported by BOTH the
live engine (trading/orb_engine.py + trading/stop_monitor.py) and the BT
pipeline (study_orb_pipeline_static_lock.py), so the frozen semantics live in
exactly one place.

FROZEN SEMANTICS (docs/orb_winner_stack_design_aug2026.md §1 + §1b — the
validated harness is research/stability/phaseB_regime_atr.py::atr14_lookup /
sz1_exit and research/stability/resim_exit.py::variant_scale; deviations from
those are bugs):

* ATR14: TR = max(H−L, |H−prevC|, |L−prevC|) on DAILY bars; ATR14 = simple
  rolling(14).mean() of the TR series; the value used on trading day T is the
  ATR ending T−1 (shift(1) — no lookahead). Fewer than 15 daily bars before T
  (whatever makes pandas' rolling window contain the NaN first-TR) → ATR
  unavailable → floor NOT applied (fail-open to range_low; caller must log
  WARNING). This freezes design §1.1's ≥15/fail-open rule — NOT the
  phaseB_frontier 14-bar/13-TR variant (P0-6.1; 0/81 book trades sit on the
  boundary, golden-tested at exactly 14/15 bars).

* Floor applies to the PROTECTIVE STOP ONLY: stop = max(range_low,
  entry − k×ATR14). Sizing and the lock machinery (R = range_high−range_low)
  are untouched by this module.

* Degenerate-ATR clamp (review P1-3): a floored stop above entry×(1−ε)
  (ε = 0.5%) is rejected → fall back to range_low (caller logs WARNING).
  Study data: min bound-floor distance below entry across the book = 1.57%,
  so the clamp costs nothing on the validated book; it only guards the
  ATR≈0 halt-shell tape where floor == entry would guarantee an instant
  stop-out.

* Scale-out: level = entry + level_r×R (range-R); qty = floor(frac×shares)
  (integer); qty < 1 → no scale (all-runner). Fill assumption in BT = limit
  at level with 10bps slip; same-bar ordering per the CORRECTED P0-1 reading:
  the frozen harness FILLS THE SCALE on a bar hitting both the stop and the
  scale level (its stop check is gated `low<=stop AND high<scale_px`) — the
  NCNA 2025-08-21 golden. Live tick ordering (stop checked first) can only be
  conservative vs this.
"""
from __future__ import annotations

import logging
import math
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Frozen defaults (owner order 8/22; research/stability/phaseB_frontier.py
# C-point = SCALE 40%@3.0R + SZ1 k=0.25).
DEFAULT_ATR_K = 0.25
DEFAULT_SCALE_FRAC = 0.40
DEFAULT_SCALE_LEVEL_R = 3.0
# P1-3 degenerate clamp: floored stop must sit at least this far below entry.
DEGENERATE_STOP_EPS = 0.005

# floored_stop status strings (stable contract — logged + tested)
FLOOR_BOUND = 'bound'          # floor > range_low: the ATR floor is active
FLOOR_UNBOUND = 'unbound'      # ATR available but range_low is already tighter
FLOOR_NO_ATR = 'no_atr'        # <15 daily bars / NaN — fail-open to range_low
FLOOR_DEGENERATE = 'degenerate'  # P1-3 clamp fired — fail-open to range_low


def _bars_to_df(daily_bars) -> Optional[pd.DataFrame]:
    """Normalize a daily-bars input (DataFrame | iterable of dicts/objects)
    to a DataFrame with float high/low/close columns, original order kept.

    Returns None when the input is empty/unusable (caller fail-opens)."""
    if daily_bars is None:
        return None
    if isinstance(daily_bars, pd.DataFrame):
        if daily_bars.empty:
            return None
        df = daily_bars
    else:
        rows = list(daily_bars)
        if not rows:
            return None
        if isinstance(rows[0], dict):
            df = pd.DataFrame(rows)
        else:  # objects with attributes
            df = pd.DataFrame([{k: getattr(r, k, None)
                                for k in ('high', 'low', 'close')}
                               for r in rows])
    if not {'high', 'low', 'close'} <= set(df.columns):
        return None
    return df


def atr14_t1(daily_bars) -> Optional[float]:
    """ATR14 available on trading day T from the daily bars STRICTLY BEFORE T.

    `daily_bars`: bars ascending by date, ending at T−1 (the live feature
    fetch and the BT cache query both produce exactly this shape). Any extra
    history earlier than the last 15 bars is harmless (rolling window only
    sees the trailing 14 TRs).

    Exact harness formula (phaseB_regime_atr.atr14_lookup): TR series →
    rolling(14).mean() → the day-T value is that series shifted by one
    session. Because the input already ends at T−1, the last unshifted
    rolling value IS the shift(1) value for day T — algebraically identical,
    golden-tested against the harness.

    Returns None when unavailable (fewer than 15 bars → the rolling window
    still contains the NaN first-TR → NaN → None). Callers MUST treat None
    as "floor not applied" and log a WARNING (fail-open to range_low).
    """
    df = _bars_to_df(daily_bars)
    if df is None:
        return None
    try:
        h = df['high'].astype(float).reset_index(drop=True)
        l = df['low'].astype(float).reset_index(drop=True)
        c = df['close'].astype(float).reset_index(drop=True)
        pc = c.shift(1)
        tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
        atr = tr.rolling(14).mean()
        v = atr.iloc[-1]
        if pd.isna(v):
            return None
        return float(v)
    except Exception as e:
        # Fail-open path MUST be loud (CLAUDE.md fallback rule).
        logger.warning(f"orb_winner_stack.atr14_t1: computation failed ({e}) "
                       f"— ATR unavailable, floor will not apply")
        return None


def floored_stop(range_low: float, entry: float, atr14: Optional[float],
                 k: float = DEFAULT_ATR_K,
                 eps: float = DEGENERATE_STOP_EPS) -> Tuple[float, str]:
    """The SZ1 protective-stop floor: max(range_low, entry − k×ATR14).

    Args:
        range_low: legacy initial stop (5-min opening-range low).
        entry: the price anchoring the floor. BT anchors on the harness
            entry price; LIVE anchors on the ACTUAL FILL price — a
            deliberate, documented BT deviation (review P1-3, same class
            as the touchgo fill-vs-market-bar decision).
        atr14: ATR14 ending T−1 (atr14_t1), or None when unavailable.
        k: floor multiplier (frozen 0.25).
        eps: P1-3 degenerate clamp — the floored stop must stay at or
            below entry×(1−eps) or we fall back to range_low.

    Returns:
        (stop_price, status) with status one of FLOOR_BOUND / FLOOR_UNBOUND /
        FLOOR_NO_ATR / FLOOR_DEGENERATE. The stop is range_low on the two
        fail-open statuses. Callers must WARNING-log no_atr and degenerate.
    """
    if atr14 is None or not np.isfinite(atr14):
        return float(range_low), FLOOR_NO_ATR
    floor = max(float(range_low), float(entry) - float(k) * float(atr14))
    if floor > float(entry) * (1.0 - eps):
        # P1-3: ATR≈0 (halt-shell tape) would put the stop at/near entry —
        # a guaranteed instant stop-out. Reject the floor entirely.
        return float(range_low), FLOOR_DEGENERATE
    if floor > float(range_low) + 1e-9:
        return floor, FLOOR_BOUND
    return float(range_low), FLOOR_UNBOUND


def scale_params(entry: float, range_size: float,
                 frac: float = DEFAULT_SCALE_FRAC,
                 level_r: float = DEFAULT_SCALE_LEVEL_R,
                 shares: int = 0) -> Tuple[float, int]:
    """Scale-out level + integer quantity for a filled position.

    Args:
        entry: entry price anchoring the +level_r×R level (BT: harness entry;
            LIVE: actual fill — same deliberate anchor rule as floored_stop).
        range_size: range_high − range_low (the frozen R unit).
        frac: fraction of the filled shares to sell (frozen 0.40).
        level_r: level in R above entry (frozen 3.0).
        shares: filled share count.

    Returns:
        (scale_px, scale_qty). scale_qty = floor(frac×shares); when that is
        < 1 the trade is all-runner → scale_qty = 0 and callers must NOT arm
        a scale (frozen tiny-qty rule).
    """
    scale_px = float(entry) + float(level_r) * float(range_size)
    qty = int(math.floor(float(frac) * int(shares)))
    if qty < 1:
        return scale_px, 0
    return scale_px, qty
