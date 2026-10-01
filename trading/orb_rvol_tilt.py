"""RVOL-at-09:35 risk tilt — shared sizing math for BT and live (sizing.rvol_tilt).

Owner 2026-10-01 ("find the money"): cell 1,694 Part B (research/orb_freq/
RESULT_1694.md, PREREG research/orb_freq/PREREG_1694.md) is the only
BOTH-DIRECTION money read on production ORB sizing found in that study —
tilting risk by the RVOL tercile at 09:35 lifts EV per unit of risk +20.1%
(dirA: select on TRAIN2025, test VAL2026) and +12.7% (dirB: select on
VAL2026, test TRAIN2025). It fails ONLY the study's own 3-bin Spearman
ordering-agreement robustness clause (-0.250 vs the >=+0.60 bar) — that
clause is mis-specified for a monotonic low/mid/high tilt (it tests
agreement across ALL 3 bins' rank order, not direction of the tilt), per
the owner's own read of cell 1,694. Shipped disabled; the owner can flip
`sizing.rvol_tilt.enabled` once satisfied.

RVOL definition (RESULT_1694.md Part B "B1_rvol_0935"; research/orb_freq/
1693_pool_exits.py lines ~513-537): the 5-minute opening-range volume
divided by an ADV20-scaled expectation. This is the SAME quantity
trading/orb_addon_gates.py already computes as `rel_volume_0935` (used by
ORBEngine._build_pool_gate_inputs for the add-on-pool min_rel_volume_0935
gate) — ONE formula, reused here (`resolve_rel_volume_0935` below), never
duplicated with a different scale.

Scale note (load-bearing — read before changing DEFAULT_EDGES): the
research feature divides by the RAW full-day ADV20
(`range_total_volume / avg_daily_volume_20d`); `rel_volume_0935` divides by
that ADV20 scaled DOWN to a 5-minute expectation
(`adv20 * RANGE_MINUTES / SESSION_MINUTES`), i.e. it is the research ratio
x (SESSION_MINUTES / RANGE_MINUTES) = x78. DEFAULT_EDGES below are the
TRAIN2025 tercile edges in the RESEARCH ratio's own units
(0.03716, 0.07649 — 1/3 and 2/3 quantiles, n=204, research/orb_freq/
1693_pool_exits.py::tercile_edges, cross-checked against the bin
boundaries actually assigned in research/orb_freq/1694_runners.csv),
rescaled x78 so they compare correctly against `rel_volume_0935`. A config
override (`sizing.rvol_tilt.edges`) must be given in `rel_volume_0935`
units too — NOT the raw research ratio.

Mults: low tercile 1.5x / mid 1.0x / high 0.5x (owner 2026-10-01). The
TOTAL_MULT_CAP (1.5) reuses the Q5 anti-overfit ceiling
(trading/orb_conviction.py Q5_MAX_MULT) as the ceiling on the TOTAL
sizing multiplier actually applied by trading/orb_planner.py
(adaptive_mult x pm_mult x this hook's tilt) — the never-rule: this hook's
own tilt factor is clamped down (never adaptive_mult or pm_mult) so the
full stack never exceeds 1.5x. A clamp only ever REDUCES this hook's own
contribution (tilt_mult == 1.0 is always a byte-identical no-op — see
clamp_rvol_tilt_to_cap), so adaptive_mults is never touched, per CLAUDE.md
("never drop the Q5 1.5 cap", "never touching adaptive_mults").

Fail-open, like orb_pm_mult.py: unknown/missing RVOL -> mult 1.0, tercile
None (never blocks a trade, never tilts blind).
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Regular session length (09:30-16:00 ET) and opening-range length — the
# SAME constants trading/orb_addon_gates.py uses to scale ADV20 down to a
# 5-minute expectation. Imported (not redefined) so the rescale factor
# below can never silently drift from the live rel_volume_0935 formula.
from trading.orb_addon_gates import RANGE_MINUTES, SESSION_MINUTES

# TRAIN2025 tercile edges of the RESEARCH ratio (range_total_volume /
# raw full-day avg_daily_volume_20d) — see module docstring "Scale note".
_RESEARCH_EDGES_RAW_RATIO: Tuple[float, float] = (0.03716, 0.07649)
_RESCALE = SESSION_MINUTES / RANGE_MINUTES  # 78.0

# Edges in rel_volume_0935 units (what this module's callers pass in).
DEFAULT_EDGES: Tuple[float, float] = (
    _RESEARCH_EDGES_RAW_RATIO[0] * _RESCALE,
    _RESEARCH_EDGES_RAW_RATIO[1] * _RESCALE,
)  # ~= (2.898, 5.966)
DEFAULT_MULTS: Tuple[float, float, float] = (1.5, 1.0, 0.5)  # low, mid, high
TOTAL_MULT_CAP = 1.5  # Q5 anti-overfit ceiling, reused as the total-stack cap


def resolve_rel_volume_0935(
    range_total_volume: Optional[float],
    adv20: Optional[float],
    range_minutes: float = RANGE_MINUTES,
    session_minutes: float = SESSION_MINUTES,
) -> Optional[float]:
    """range_total_volume / (adv20-scaled 5-min expectation).

    Byte-identical twin of the formula inlined in
    ORBEngine._build_pool_gate_inputs (trading/orb_engine.py) — kept here
    so the sizing hook and the pool-gate system can never silently diverge
    (CLAUDE.md "ONE spec ... through ONE helper module"). Returns None on
    missing/non-positive inputs (fail-open for the caller).
    """
    try:
        adv20 = float(adv20) if adv20 is not None else 0.0
        if adv20 <= 0 or range_total_volume is None:
            return None
        expected = adv20 * (range_minutes / session_minutes)
        if expected <= 0:
            return None
        return float(range_total_volume) / expected
    except (TypeError, ValueError):
        return None


def tercile_for_rvol(
    rvol: Optional[float], edges: Tuple[float, float] = DEFAULT_EDGES,
) -> Optional[str]:
    """'low' / 'mid' / 'high' tercile label for rvol against fixed edges.

    None/NaN/non-numeric -> None (fail-open; caller treats as inactive).
    """
    if rvol is None:
        return None
    try:
        rvol = float(rvol)
    except (TypeError, ValueError):
        return None
    if math.isnan(rvol):
        return None
    e1, e2 = edges
    if rvol < e1:
        return 'low'
    if rvol < e2:
        return 'mid'
    return 'high'


def resolve_rvol_tilt_mult(
    rvol: Optional[float],
    edges: Tuple[float, float] = DEFAULT_EDGES,
    mults: Tuple[float, float, float] = DEFAULT_MULTS,
) -> Tuple[float, Optional[str]]:
    """(mult, tercile) for one rvol_0935 reading. rvol None/NaN -> (1.0, None)."""
    tercile = tercile_for_rvol(rvol, edges)
    if tercile is None:
        return 1.0, None
    mult_by_tercile = {'low': float(mults[0]), 'mid': float(mults[1]), 'high': float(mults[2])}
    return mult_by_tercile[tercile], tercile


def clamp_rvol_tilt_to_cap(
    stacked_mult_before_tilt: float,
    tilt_mult: float,
    cap: float = TOTAL_MULT_CAP,
) -> Tuple[float, bool]:
    """Clamp this hook's OWN tilt factor so the total stack never exceeds `cap`.

    Returns (effective_tilt_mult, clamped).

    tilt_mult == 1.0 (disabled, or the neutral mid tercile) is ALWAYS a
    byte-identical no-op — this is what makes `sizing.rvol_tilt.enabled:
    false` reproduce pre-hook sizing exactly, and it is also why a
    reducing tilt (tilt_mult < 1.0) can never be pushed back UP by this
    function. Only tilt_mult > 1.0 (the low-RVOL upsize) can trigger a
    clamp, and the clamp reduces ONLY the tilt factor — adaptive_mult and
    pm_mult (folded into stacked_mult_before_tilt by the caller) are never
    adjusted here or anywhere else by this module.
    """
    if tilt_mult == 1.0 or stacked_mult_before_tilt <= 0:
        return float(tilt_mult), False
    raw_total = stacked_mult_before_tilt * tilt_mult
    if raw_total <= cap + 1e-9:
        return float(tilt_mult), False
    effective = cap / stacked_mult_before_tilt
    logger.warning(
        "[ORB] RVOL_TILT cap clamp: stacked_mult_before_tilt=%.3f x tilt=%.3f "
        "= %.3f > cap %.3f -> effective_tilt=%.3f (adaptive_mult/pm_mult untouched)",
        stacked_mult_before_tilt, tilt_mult, raw_total, cap, effective,
    )
    return effective, True
