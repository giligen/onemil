"""Shared MACD wave conviction scoring (V4 — 3-tier gradient).

Single source of truth for the conviction formula. Both the backtest
(`macd_wave_backtest.py`) and PROD engine (`trading/macd_wave_engine.py`)
import from here to eliminate BT/PROD drift risk — parallel implementations
were the root cause of bull flag's Friday-review followup #1.

Used as a POSITION SIZING multiplier only. Never as a hard filter:
the step 2 walk-forward study showed filter variants have mean OOS
ΔP&L of -$153 with a -$5.9K worst case (fragile).

## Research
- Step 1 (analyze_macd_conviction.py): validated 2 OOS-persistent axes —
    cross_time_min (train ρ=-0.78, test ρ=-0.88) and
    vol_at_cross   (train ρ=-0.94, test ρ=-0.55).
- Step 2 (study_macd_conviction.py): compared 5 scoring formulas across
    3 chronological walk-forward splits. V4 (this formula) won by
    mean OOS ΔP&L +$24.6K with worst split +$10.3K (robust).
- Canonical 15mo BT lift: +$54K / +49.5% (551 trades, DD -$9.5K → -$13K).
"""
from __future__ import annotations

from typing import Dict, Tuple


# Cross-time tiers (minutes from open to +10% cross; lower = better momentum)
_CROSS_TIME_TIER1_MAX = 3   # top-tier: fast cross
_CROSS_TIME_TIER2_MAX = 5
_CROSS_TIME_TIER3_MAX = 7

# Volume-at-cross tiers (cumulative shares up to cross bar; lower = less crowded)
_VOL_AT_CROSS_TIER1_MAX = 27_000
_VOL_AT_CROSS_TIER2_MAX = 79_000
_VOL_AT_CROSS_TIER3_MAX = 165_000

# V4 per-tier contribution: 0.4 / 0.2 / 0.1 / 0
_TIER1_CONTRIB = 0.4
_TIER2_CONTRIB = 0.2
_TIER3_CONTRIB = 0.1

# Score clamp range — safety bound even if rules are edited wrongly.
_CONVICTION_MIN = 0.5
_CONVICTION_MAX = 2.0


def compute_conviction_score(
    cross_time_min: int, vol_at_cross: int
) -> Tuple[float, Dict[str, float]]:
    """Compute V4 conviction score for a MACD wave setup.

    Args:
        cross_time_min: minutes from market open to the +10% cross bar (1-indexed).
        vol_at_cross:   cumulative share volume through the cross bar.

    Returns:
        (final_score, breakdown_dict) where:
          final_score: clamped to [0.5, 2.0]. Range on current rules: [1.0, 1.8].
          breakdown_dict: {'cross_speed', 'vol_at_cross', 'raw_score', 'final_score'}.

    Used as a sizing multiplier:
        shares = int(position_size * final_score / entry_price)
    """
    score = 1.0
    breakdown: Dict[str, float] = {}

    # Rule 1: cross speed (earlier cross = stronger momentum)
    if cross_time_min <= _CROSS_TIME_TIER1_MAX:
        c1 = _TIER1_CONTRIB
    elif cross_time_min <= _CROSS_TIME_TIER2_MAX:
        c1 = _TIER2_CONTRIB
    elif cross_time_min <= _CROSS_TIME_TIER3_MAX:
        c1 = _TIER3_CONTRIB
    else:
        c1 = 0.0
    score += c1
    breakdown['cross_speed'] = c1

    # Rule 2: vol at cross (less crowded = less competition for the fill)
    if vol_at_cross <= _VOL_AT_CROSS_TIER1_MAX:
        c2 = _TIER1_CONTRIB
    elif vol_at_cross <= _VOL_AT_CROSS_TIER2_MAX:
        c2 = _TIER2_CONTRIB
    elif vol_at_cross <= _VOL_AT_CROSS_TIER3_MAX:
        c2 = _TIER3_CONTRIB
    else:
        c2 = 0.0
    score += c2
    breakdown['vol_at_cross'] = c2

    # Snap accumulated FP noise (0.4+0.4 = 0.79999...) — contributions are tenths,
    # so 3 decimals is lossless and guarantees downstream cleanliness.
    score = round(score, 3)
    breakdown['raw_score'] = score
    final = max(_CONVICTION_MIN, min(_CONVICTION_MAX, score))
    breakdown['final_score'] = final
    return final, breakdown
