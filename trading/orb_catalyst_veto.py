"""Catalyst-required veto — shared by BT and live (2026-07-18 ship).

Owner-approved rule (−$36K/18mo budget accepted): every ORB entry must
have a CATALYST —
  (a) premarket news on its own ticker (the news-gate cell), OR
  (b) COMPLEX CONFIRMATION: ≥ min_cohort same-morning candidates sharing
      its underlying anchor (a stock + its wrappers, or sibling wrappers
      of one underlying, gapping together = the underlying's move is
      real).
Newsless-and-alone picks are VETOED post-ranking; the slot is CONSUMED
(no refill — the refill form re-tested toxic on 2026-07-18: MDD +42%).

Evidence (live-parity book, full selection resim, 2026-07-18):
  V1 no-refill: $257,310 vs base $293,568 (−$36.3K, owner-approved),
  MDD −$16.3K→−$14.0K, worst month −$10.3K→−$7.8K, July-26 bleed −62%,
  trades 677→221. Cost era-consistent (−15/−7/−14K). Rejected variants:
  wrapper-only veto (era-lopsided, kills 2026 winners), refill forms
  (MDD +42%), further sub-cuts inside the kept book (every candidate
  cut era-flips — see ledger). Newsless-alone universe cohort is
  negative in ALL eras; its selected slice was +$34K/464tr of churn.

Fail-open doctrine: missing NEWS data (fetch failed, has_news=None) →
do NOT veto (PDR precedent: absence of data is not evidence). A
computed cohort of 1 is DATA (the day's candidate list is complete at
ranking time), so newsless+alone with known-no-news IS vetoed.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_MIN_COHORT = 2


def anchor_cohort_counts(anchors: Iterable[Optional[str]]) -> Dict[str, int]:
    """Count same-morning candidates per underlying anchor (None skipped)."""
    counts: Dict[str, int] = {}
    for a in anchors:
        if a:
            counts[a] = counts.get(a, 0) + 1
    return counts


def has_complex_confirmation(anchor: Optional[str],
                             cohort_counts: Dict[str, int],
                             min_cohort: int = DEFAULT_MIN_COHORT) -> bool:
    """True when the candidate's anchor complex has >= min_cohort members
    this morning (anchor None -> never confirmed)."""
    if not anchor:
        return False
    return cohort_counts.get(anchor, 0) >= min_cohort


def catalyst_veto_applies(has_news: Optional[bool],
                          anchor: Optional[str],
                          cohort_counts: Dict[str, int],
                          min_cohort: int = DEFAULT_MIN_COHORT) -> bool:
    """True -> VETO (no catalyst). has_news semantics:
      True  -> catalyst present, never veto
      None  -> news UNKNOWN (fetch failed) -> fail-open, never veto
      False -> known-newsless -> veto unless complex-confirmed."""
    if has_news is True or has_news is None:
        return False
    return not has_complex_confirmation(anchor, cohort_counts, min_cohort)
