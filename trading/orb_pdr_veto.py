"""Prev-day-range (PDR) veto — shared by BT pipeline and live ORB engine.

Shipped 2026-07-04 (weekend money-machine program, W2). Vetoes ORB picks
whose PREVIOUS day's range was quiet: ORB monetizes CONTINUATION of an
already-explosive move ("day-2 of the fireworks"); a quiet-prev-day name
gapping up is a day-1 fresh pop that mean-reverts.

CRITICAL — NO-REFILL FORM ONLY. The veto applies AFTER ranking/top-K
selection: a vetoed pick's slot stays EMPTY. Backfilling slots with
next-ranked candidates was tested and is TOXIC (2025H2 P&L collapses to
~$0, MDD balloons −$29K→−$50K — same failure mode as the refuted ETF
exclusion: refills pull in below-cutline junk).

Evidence (defended-pipeline replica, Jan'25–Jul'26, 1,193 trades):
  threshold 8.0: TOT $154,892 → $209,734 (+35%), MDD −$29,297 → −$20,129,
  WR 35.8% → 40.2%, all three eras positive (25H1 +$53K / 25H2 +$51K /
  2026 +$106K), ALL top-10 giants kept, monotone across thresholds 6–10.
  Search artifacts: /tmp/orb_veto_search.csv; state file
  research/weekend_state_jul2026.md.

Feature definition MUST match study_orb_features.py:287:
    prev_day_range_pct = (prev_high - prev_low) / prev_close * 100
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MIN_PDR_PCT = 8.0


def compute_prev_day_range_pct(prev_high: float, prev_low: float,
                               prev_close: float) -> Optional[float]:
    """Previous day's high-low range as a percent of its close.

    Exact BT-parity formula (study_orb_features.py:287). Returns None on
    unusable inputs (non-positive close, inverted/absent high-low) so the
    caller can fail-open with a WARNING rather than veto on garbage.
    """
    try:
        prev_high = float(prev_high)
        prev_low = float(prev_low)
        prev_close = float(prev_close)
    except (TypeError, ValueError):
        return None
    if prev_close <= 0 or prev_high < prev_low or prev_high <= 0:
        return None
    return (prev_high - prev_low) / prev_close * 100.0


def pdr_veto_applies(prev_day_range_pct: Optional[float],
                     min_pdr_pct: float = DEFAULT_MIN_PDR_PCT) -> bool:
    """True iff the pick must be vetoed (quiet prev day).

    None (feature unavailable) NEVER vetoes — fail-open; the caller logs
    WARNING. Missing prev-day data in BT drops the candidate at the
    feature stage, so live fail-open cannot create a BT<->live divergence
    on any candidate BT actually traded.
    """
    if prev_day_range_pct is None:
        return False
    return float(prev_day_range_pct) <= float(min_pdr_pct)
