"""Pure helpers for tier-aware MACD zone multiplier selection.

Shared between `backtest.py` and `trading/trading_engine.py` to guarantee
byte-identical tier routing behavior. Per-tier S2-max ship, 2026-04-18.

Design:
  - Both BT and PROD call `select_tier_multipliers` with the same 7-arg
    signature (intraday change, 3 A-tier mults, 3 Extras mults).
  - Returns the (strong_pos, strong_neg, normal) tuple for the classified tier.
  - Keeps `_get_macd_zone_multiplier` implementations minimal: classifier
    + zone-boundary logic remains in each file, but multiplier selection is
    delegated here.

Unit-testable without instantiating TradingEngine or BacktestRunner.
"""
from __future__ import annotations

from typing import Tuple

from trading.two_tier_filter import classify_tier, TIER_EXTRAS


def select_tier_multipliers(
    intraday_change_pct: float,
    a_strong_pos: float,
    a_strong_neg: float,
    a_normal: float,
    extras_strong_pos: float,
    extras_strong_neg: float,
    extras_normal: float,
) -> Tuple[float, float, float, str]:
    """Pick (strong_pos, strong_neg, normal, tier) based on intraday-change tier.

    Args:
        intraday_change_pct: max intraday %-change-from-prev-close at entry.
            0.0 (or < 10%) → edge tier → falls back to A-tier multipliers.
            [10%, 20%)     → Extras tier → uses extras_* multipliers.
            >= 20%         → A-tier → uses a_* multipliers.
        a_strong_pos, a_strong_neg, a_normal: A-tier (default) multipliers.
        extras_strong_pos, extras_strong_neg, extras_normal: Extras overrides.

    Returns:
        (strong_pos_mult, strong_neg_mult, normal_mult, tier_label) — the
        three multipliers to apply for each MACD zone classification, plus
        the tier label (for logging).
    """
    tier = classify_tier(intraday_change_pct)
    if tier == TIER_EXTRAS:
        return extras_strong_pos, extras_strong_neg, extras_normal, tier
    # A-tier or edge → A-tier defaults
    return a_strong_pos, a_strong_neg, a_normal, tier
