"""Anchor dedup — at most ONE ORB pick per underlying anchor per day.

ONE spec for the backtest pipeline (`study_orb_pipeline_static_lock.py`) and
the live engine (`trading/orb_engine.py`): both import this module, so the
rule is identical by construction.

Motivation (2026-09-18 live): CIFG and CIFU — both 2X long wrappers on CIFR
per `data/research/orb_asset_class_map_20260711.csv` — were both selected,
both filled and both stopped in the same second. `orb.yaml dedup:
by_family + by_super_group` resolves families from the hand-kept table in
`study_orb_correlation_filter.py` (14 families / 91 symbols) which predates
those tickers, while the engine ALREADY computes `underlying_anchor` for the
catalyst veto. One underlying, two slots, twice the size.

The rule (pre-registered in research/orb_anchor_dedup/PREREG.md):

  * rank order decides the survivor (the day's ranking: quintile order asc,
    composite desc — the order the submit loop already walks);
  * every LATER pick sharing that anchor is rejected;
  * NO REFILL — the rejected pick's slot stays empty, the invariant every ORB
    veto obeys (a refilling form was toxic for PDR: MDD −$29K → −$50K);
  * an anchor is marked seen at its rank position whether or not that pick is
    later vetoed by PDR/G1/range-size/catalyst, so the survivor depends on the
    ranking alone and never on veto ordering;
  * an unresolvable anchor (None/empty) NEVER dedups — unknown fails OPEN,
    like every other consumer of `underlying_anchor`.
"""
from __future__ import annotations

from typing import Iterable, List, Optional


class AnchorDedup:
    """Per-day seen-anchor state. Walk picks in RANK order, one call each.

    Usage is deliberately identical on both sides — the live submit loop calls
    `reject()` per pick, the pipeline calls `reject_mask()` over the day's
    selected rows in rank order, and `reject_mask` is implemented with
    `reject()` so there is exactly one behaviour.
    """

    def __init__(self) -> None:
        self._seen: set = set()

    def reject(self, anchor: Optional[str]) -> bool:
        """True iff this pick duplicates an anchor already taken today.

        Marks the anchor as taken on the first (highest-ranked) sighting.
        """
        if not anchor:
            return False          # unknown anchor fails OPEN — never dedups
        if anchor in self._seen:
            return True
        self._seen.add(anchor)
        return False

    @property
    def seen(self) -> set:
        """Anchors already taken today (diagnostics/tests only)."""
        return set(self._seen)


def reject_mask(anchors: Iterable[Optional[str]]) -> List[bool]:
    """Reject flags for one day's picks given their anchors IN RANK ORDER."""
    state = AnchorDedup()
    return [state.reject(a) for a in anchors]
