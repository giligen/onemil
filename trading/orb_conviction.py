"""ORB adaptive quintile multipliers — production module.

Loads per-quintile multipliers from orb.yaml's `adaptive_mults` block.
The multipliers are TRAIN-fit ratios of (quintile avg P&L) / (overall avg P&L),
clipped to [min_mult, max_mult_per_quintile]. Q5 is capped tighter (1.5x)
than other quintiles (3.0x) to prevent over-sizing on extreme z-scores that
empirically overfit on Split A of our walk-forward.

Background: on Split A (H1 2025 TRAIN), raw Q5 ratio = 2.462 — Q5 trades had
the highest avg P&L in that training window. But across Splits B/C and full
out-of-sample, Q4 consistently outperformed Q5. Capping Q5 at 1.5x prevents
Split A's anomaly from propagating into live sizing decisions.

Pure computation module — no side effects, no DB/Alpaca calls.
"""
from __future__ import annotations

from typing import Dict


# Global bounds applied during sanity-check of loaded mults.
# Matches the validated research code (study_orb_sizing.ADAPTIVE_MULT_MIN /
# study_orb_correlation_filter.PER_QUINTILE_MAX_MULT).
MIN_MULT = 0.25           # floor — even the worst quintile gets at least this
DEFAULT_MAX_MULT = 3.0    # ceiling for Q1-Q4
Q5_MAX_MULT = 1.5         # tighter ceiling for Q5 (anti-overfit)

_ALL_QUINTILES = ('Q1', 'Q2', 'Q3', 'Q4', 'Q5')


def load_adaptive_mults(orb_yaml_mults: Dict[str, float]) -> Dict[str, float]:
    """Parse the `adaptive_mults` section of orb.yaml + validate bounds.

    Args:
        orb_yaml_mults: dict like {'Q1': 0.83, ..., 'Q5': 1.5}.

    Returns:
        dict {'Q1': mult, ..., 'Q5': mult} with all 5 keys present and all
        values validated against their caps.

    Raises:
        ValueError if:
          - any quintile Q1..Q5 is missing
          - any value is < MIN_MULT
          - Q1-Q4 exceeds DEFAULT_MAX_MULT
          - Q5 exceeds Q5_MAX_MULT (baked-in anti-overfit check)
    """
    if not isinstance(orb_yaml_mults, dict):
        raise ValueError(f"adaptive_mults must be a dict, got {type(orb_yaml_mults).__name__}")
    out: Dict[str, float] = {}
    for q in _ALL_QUINTILES:
        if q not in orb_yaml_mults:
            raise ValueError(f"adaptive_mults missing quintile '{q}' (must have Q1..Q5)")
        val = float(orb_yaml_mults[q])
        if val < MIN_MULT:
            raise ValueError(
                f"adaptive_mults[{q}]={val:.3f} < min {MIN_MULT}. "
                f"Refit or clip in orb.yaml."
            )
        cap = Q5_MAX_MULT if q == 'Q5' else DEFAULT_MAX_MULT
        if val > cap:
            raise ValueError(
                f"adaptive_mults[{q}]={val:.3f} > cap {cap}. "
                f"{'Q5 is capped at 1.5x (anti-overfit).' if q == 'Q5' else ''}"
            )
        out[q] = val
    return out


def apply_adaptive_mult(quintile: str, mults: Dict[str, float]) -> float:
    """Look up the multiplier for a given quintile bucket.

    Args:
        quintile: one of 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'.
        mults: as returned by load_adaptive_mults.

    Returns:
        multiplier (float), defaulting to 1.0 if quintile not found (safety
        fallback with WARNING — should never happen in production since
        assign_quintile always returns one of Q1..Q5).

    Raises:
        ValueError if quintile not a valid bucket label.
    """
    if quintile not in _ALL_QUINTILES:
        raise ValueError(f"Invalid quintile '{quintile}' (must be Q1..Q5)")
    return mults.get(quintile, 1.0)


def compute_adaptive_mults_from_averages(
    quintile_avgs: Dict[str, float],
    overall_avg: float,
) -> Dict[str, float]:
    """Compute adaptive mults from raw per-quintile P&L averages.

    This is the refit formula used by `study_orb_refit.py`:
      mult_Q = clip(quintile_avg / overall_avg, MIN_MULT, MAX_Q_CAP)

    Where MAX_Q_CAP is 1.5 for Q5 and 3.0 for others.

    Args:
        quintile_avgs: dict {'Q1': avg_pnl, ..., 'Q5': avg_pnl}
        overall_avg: mean P&L across all trades.

    Returns:
        dict of validated mults ready to write back to orb.yaml.

    Raises:
        ValueError on missing quintile or overall_avg <= 0.
    """
    if overall_avg <= 0:
        raise ValueError(
            f"overall_avg must be > 0 for ratio calc, got {overall_avg}. "
            f"Strategy P&L is negative — do not refit."
        )
    out: Dict[str, float] = {}
    for q in _ALL_QUINTILES:
        if q not in quintile_avgs:
            raise ValueError(f"quintile_avgs missing '{q}'")
        raw = float(quintile_avgs[q]) / overall_avg
        cap = Q5_MAX_MULT if q == 'Q5' else DEFAULT_MAX_MULT
        out[q] = max(MIN_MULT, min(cap, raw))
    return out
