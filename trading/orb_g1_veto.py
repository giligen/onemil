"""G1 volatility-fingerprint veto — shared by BT pipeline and live ORB engine.

Shipped 2026-08-15 (B+ RESTART, research/orb_bplus_frozen_params_aug2026.yaml
§3). Vetoes ORB picks whose volatility fingerprint says "not the day-2
continuation this strategy monetizes": the pick is KEPT only when BOTH its
prior-20-day return volatility AND its previous-day range clear their frozen
minimums.

CRITICAL — NO-REFILL FORM ONLY. Like the PDR and catalyst vetoes, G1 applies
AFTER ranking/top-K selection: a vetoed pick's slot stays EMPTY. Backfilling
is toxic (proven for PDR: 2025H2 collapses to ~$0, MDD −$29K→−$50K).

FAIL-OPEN ASYMMETRY (the load-bearing subtlety, review P1-2): a naive
`keep = rv >= RV_MIN and pdr >= PDR_MIN` would VETO on `rv == 0.0`
(0 >= 7.106 is False) and INVERT the BT decision, killing every legit but
short-history name. `return_volatility_20d == 0.0` is the "history too short
to certify" MARKER that study_orb_features.py:299,314 writes when a symbol has
< 5 prior daily bars — it is NOT a real 0% volatility. So rv ∈ {None, NaN, 0.0}
=> KEEP (fail-open), branched BEFORE the AND. Missing/NaN `prev_day_range_pct`
also fails open. But `prev_day_range_pct == 0.0` is a REAL flat prior day
(the quietest possible = prime veto territory) and IS vetoable — asymmetric
with rv20 by design.

Feature definitions MUST match study_orb_features.py:
    return_volatility_20d = std(diff(closes)/closes[:-1], ddof=0) * 100  (:308-314)
    prev_day_range_pct    = (prev_high - prev_low) / prev_close * 100     (:287)
"""
from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_RV20_MIN = 7.106       # frozen yaml §3 return_volatility_20d_min
DEFAULT_PDR_MIN = 9.226        # frozen yaml §3 prev_day_range_pct_min


def _to_float(x) -> Optional[float]:
    """Coerce to float; return None on unparseable input (never raises)."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _is_missing(x: Optional[float]) -> bool:
    """True iff the value is absent (None) or NaN — a fail-open marker."""
    return x is None or (isinstance(x, float) and math.isnan(x))


def g1_reject(return_volatility_20d,
              prev_day_range_pct,
              rv20_min: float = DEFAULT_RV20_MIN,
              pdr_min: float = DEFAULT_PDR_MIN) -> Optional[str]:
    """Decide the G1 veto for one SELECTED pick.

    Returns a short reason string (the pick is VETOED, drop it, slot stays
    empty) or None (KEEP the pick). Parity by construction: both the live
    engine and the BT pipeline call this exact function.

    KEEP (return None) when:
      * return_volatility_20d is None / NaN / == 0.0  (fail-open; short history)
      * prev_day_range_pct is None / NaN               (fail-open; missing)
      * BOTH rv20 >= rv20_min AND pdr >= pdr_min        (passes the fingerprint)

    VETO (return reason) only when both features are REAL values and at least
    one is below its minimum. `prev_day_range_pct == 0.0` is a real flat day
    and IS vetoable.
    """
    rv = _to_float(return_volatility_20d)
    pdr = _to_float(prev_day_range_pct)

    # Fail-open markers — evaluated BEFORE the AND so rv==0.0 never inverts.
    if _is_missing(rv) or rv == 0.0:
        return None
    if _is_missing(pdr):
        return None

    if rv >= rv20_min and pdr >= pdr_min:
        return None

    return (f"rv20={rv:.3f} (min {rv20_min}) "
            f"pdr={pdr:.3f} (min {pdr_min})")
