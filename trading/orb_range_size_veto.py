"""Opening-range-size veto — shared by the BT pipeline and the live ORB engine.

Shipped 2026-09-08 (V1 veto study, research/orb_veto_study/REPORT.md). Vetoes
a SELECTED pick whose 5-minute opening range is tiny relative to price: with
`range_size_pct <= 2.221` the +30 bps stop-limit above range_high is a noise
trigger, not a breakout. The bucket is the worst raw quintile in BOTH years
(2025 −0.43R, 2026 −0.30R on 7,116 entered candidates); on the honest B+ book
it removes 10 picks — 3 fills, all losers, 7 slot-burning no-fills — for
$6,085 → $6,256, MDD −$685 → −$620, worst month −$236 → −$185.

CRITICAL — NO-REFILL FORM ONLY (same invariant as PDR/G1/catalyst): applied
AFTER ranking/top-K, the vetoed pick's slot stays EMPTY.

Feature definition MUST match study_orb_features.py / orb_engine.py:1800:
    range_size_pct = (range_high - range_low) / open * 100
"""
from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MIN_RANGE_SIZE_PCT = 2.221   # bottom-quintile edge of the raw scan (fixed, not tuned)


def range_size_veto_applies(range_size_pct, min_pct: float = DEFAULT_MIN_RANGE_SIZE_PCT) -> bool:
    """True iff the pick must be vetoed (opening range too small).

    None / NaN / non-numeric NEVER vetoes — fail-open; the caller logs a
    WARNING. `<=` matches the pipeline's quintile edge (bucket (0.123, 2.221]).
    """
    try:
        v = float(range_size_pct)
    except (TypeError, ValueError):
        return False
    if math.isnan(v):
        return False
    return v <= float(min_pct)
