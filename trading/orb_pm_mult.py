"""Premarket dollar-volume sizing multiplier — shared by BT and live.

Shipped 2026-07-04 (assumption-audit follow-through). Trades whose
premarket (4:00–9:29 ET) dollar volume exceeds a TRAIN-frozen cut get
their position UPSIZED. Upsize-only by design: the PM$ gradient's value
is at the top (+$792/trade high tercile vs −$17 low), monsters can only
be boosted never cut, and the downside of a boosted loser is capped by
its stop while the boosted winner is uncapped.

Evidence (defended book, walk-forward, TRAIN-fit cut):
  high-tier ×1.5: ΔTRAIN +$19.6K / Δ25H2 +$14.7K / Δ2026 +$42.5K
  (= +$76.8K/18mo), 13/19 months ≥0 with worst month −$2.6K,
  0 giants downsized, corr(PM$, composite) = 0.05 (orthogonal channel).
  Study: research/scripts/orb_premarket_volume_study.py; data:
  data/research/orb_premarket_dollar_vol_20260704.csv.

The cut is FROZEN from the H1-2025 TRAIN tercile (same doctrine as the
z-fit — do not refit without a walk-forward harness). The 1.5 cap
mirrors the Q5 anti-overfit cap.

Fail-open: unknown/missing PM data → mult 1.0 (never blocks a trade,
never boosts blind).
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# H1-2025 TRAIN upper-tercile cut of premarket dollar volume (vwap-weighted).
DEFAULT_HIGH_CUT_USD = 5_816_688.0
DEFAULT_HIGH_MULT = 1.5


def compute_pm_dollar_vol(bars_df: Optional[pd.DataFrame]) -> Optional[float]:
    """Premarket dollar volume from 1-min bars (rows before 9:30 ET).

    Accepts a frame covering any window that includes premarket; regular-
    session rows are excluded by the 9:30 ET cutoff. Uses vwap when the
    column exists (BT-study parity), else close — per-minute difference
    is immaterial vs a $5.8M cut. Returns None when no premarket rows.
    """
    if bars_df is None or bars_df.empty or 'timestamp' not in bars_df:
        return None
    try:
        et = pd.to_datetime(bars_df['timestamp'], utc=True) \
            .dt.tz_convert('America/New_York').dt.time
        pm = bars_df[et < pd.Timestamp('09:30').time()]
        if pm.empty:
            return None
        px = pm['vwap'] if 'vwap' in pm.columns and pm['vwap'].notna().any() \
            else pm['close']
        return float((pm['volume'] * px).sum())
    except Exception as e:
        logger.warning(f"compute_pm_dollar_vol failed: {e} — fail-open (None)")
        return None


def pm_size_multiplier(pm_dollar_vol: Optional[float],
                       high_cut_usd: float = DEFAULT_HIGH_CUT_USD,
                       high_mult: float = DEFAULT_HIGH_MULT) -> float:
    """Sizing multiplier from premarket dollar volume. Upsize-only.

    None / non-numeric / below-cut → 1.0. Above cut → high_mult.
    """
    try:
        if pm_dollar_vol is None:
            return 1.0
        return float(high_mult) if float(pm_dollar_vol) > float(high_cut_usd) \
            else 1.0
    except (TypeError, ValueError):
        return 1.0
