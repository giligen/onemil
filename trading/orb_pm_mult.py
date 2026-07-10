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

NEWS GATE (shipped 2026-07-10, owner-approved A2 variant): the PM boost is
gated on pre-market NEWS presence. The 2026-07-10 catalyst study
(research/orb_news_catalyst_jul2026.md) showed the PM$ bucket WITHOUT news
is ~flat per-trade in all 3 eras (+$145/−$101/+$36) while the news×PM$
interaction cell is the strongest era-consistent separator ever found
(+$1,580/+$1,569/+$935 per trade; monster rate 28/15/13% vs 6-8% rest).
Pipeline-integrated: TOT $250,276 → $301,518 (+$51K, all eras positive),
MDD −$18,815 → −$18,174 (improves). Raw has_news is the signal — catalyst-
quality classification REFUTED for longs (recap-only articles hold
AMCI +$23K / BNAI +$13.6K; do not port stupid-money's LLM classifier).

Fail-open (both channels): unknown/missing PM data → mult 1.0; above-cut
but news UNKNOWN (fetch failed) → high_mult, i.e. no news boost (never
blocks a trade, never boosts blind).
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# H1-2025 TRAIN upper-tercile cut of premarket dollar volume (vwap-weighted).
DEFAULT_HIGH_CUT_USD = 5_816_688.0
# 2026-07-10 news gate: pm-hi WITHOUT news de-boosted 1.5 → 1.0 (flat bucket);
# pm-hi WITH news boosted to 2.0 (= 1.5 PM × 1.33 news, two stacked mults
# each within the 1.5 single-mult cap doctrine). Legacy (gate off): 1.5.
DEFAULT_HIGH_MULT = 1.0
DEFAULT_HIGH_MULT_NEWS = 2.0
LEGACY_HIGH_MULT = 1.5  # pre-news-gate behavior (rollback reference)


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
                       high_mult: float = DEFAULT_HIGH_MULT,
                       has_news: Optional[bool] = None,
                       high_mult_news: float = DEFAULT_HIGH_MULT_NEWS,
                       news_gate: bool = True) -> float:
    """Sizing multiplier from premarket dollar volume + news presence.

    Upsize-only. None / non-numeric / below-cut PM$ → 1.0 always.
    Above the cut:
      news_gate=False        → high_mult (legacy pre-2026-07-10 behavior;
                               callers wanting byte-identical legacy pass
                               high_mult=LEGACY_HIGH_MULT)
      news_gate=True:
        has_news is True     → high_mult_news (the news×PM$ combo cell)
        has_news is False    → high_mult (flat bucket, de-boosted default 1.0)
        has_news is None     → high_mult (news UNKNOWN — fetch failed;
                               fail-open = no news boost, never boost blind)

    Shared by BT (study_orb_pipeline_static_lock.py) and live
    (trading/orb_engine.py) — parity by construction.
    """
    try:
        if pm_dollar_vol is None:
            return 1.0
        if float(pm_dollar_vol) <= float(high_cut_usd):
            return 1.0
        if news_gate and has_news is True:
            return float(high_mult_news)
        return float(high_mult)
    except (TypeError, ValueError):
        return 1.0
