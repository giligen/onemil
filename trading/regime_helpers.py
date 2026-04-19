"""Regime classification + sizing multiplier (Phase 1.4b ship, 2026-04-18).

Shared between `backtest.py` and `trading/trading_engine.py` to guarantee
byte-identical regime assignment. Both callers import `classify_regime`
and `compute_regime_features` from this module.

Four regimes — classified from SPY features available BEFORE market open
on day T (features computed from T-1 close):

    A  : Clean Bull             — above SMA_50, vol_20_ann <  22%
    B  : Volatile               —                vol_20_ann >= 22%
    C1 : True Defensive         — below SMA_50, slope_10d <= +0.15%
    C2 : Shallow-dip-in-uptrend — below SMA_50, slope_10d >  +0.15%

Ship multipliers (Grid winner, post-hoc capped at 6× BT hard cap,
validated across TRAIN/VAL/HOQ1 splits, +$27,321 total lift):

    A=1.25, B=1.00, C1=1.50, C2=0.00 (skip)

Safe defaults:
  - NaN/None on vol or above_sma → 'unknown' → multiplier 1.0 (no trade
    effect: no boost, no skip). Callers should NEVER silently over-boost.
  - NaN/None on slope within the below-SMA low-vol branch → 'C1' (the
    safer of the two defensive sub-regimes).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


# Defaults — mirror the yaml ship values. If callers don't pass thresholds,
# these are what's used, which matches the shipped config.
DEFAULT_VOL_THRESHOLD_PCT = 22.0
DEFAULT_SLOPE_THRESHOLD_PCT = 0.15

# Feature window sizes — match research/regime_classifier.py exactly.
SMA_WINDOW = 50
SLOPE_WINDOW = 10
VOL_WINDOW = 20


def classify_regime(
    vol_20_ann: Optional[float],
    above_sma_50: Optional[bool],
    sma_slope_10d: Optional[float],
    vol_threshold_pct: float = DEFAULT_VOL_THRESHOLD_PCT,
    slope_threshold_pct: float = DEFAULT_SLOPE_THRESHOLD_PCT,
) -> str:
    """Classify a single day's regime from pre-open SPY features.

    Args:
        vol_20_ann: 20-day annualized realized vol of SPY returns, percent.
            None/NaN → 'unknown'.
        above_sma_50: whether SPY close is strictly above its 50-day SMA.
            None → 'unknown'.
        sma_slope_10d: 10-day percent change of the SMA_50 itself.
            None/NaN → safer-default 'C1' when applicable.
        vol_threshold_pct: vol cutoff for B regime (inclusive >=).
        slope_threshold_pct: slope cutoff for C1 vs C2 (C2 if strictly >).

    Returns: 'A' | 'B' | 'C1' | 'C2' | 'unknown'.
    """
    if vol_20_ann is None:
        return 'unknown'
    try:
        if np.isnan(vol_20_ann):
            return 'unknown'
    except (TypeError, ValueError):
        return 'unknown'

    if above_sma_50 is None:
        return 'unknown'

    if vol_20_ann >= vol_threshold_pct:
        return 'B'
    if above_sma_50:
        return 'A'

    # below SMA, low vol — split by slope
    if sma_slope_10d is None:
        return 'C1'
    try:
        if np.isnan(sma_slope_10d):
            return 'C1'
    except (TypeError, ValueError):
        return 'C1'
    return 'C2' if sma_slope_10d > slope_threshold_pct else 'C1'


def get_regime_multiplier(regime: str, multipliers: Optional[Dict]) -> float:
    """Look up regime multiplier from a config dict.

    Defaults to 1.0 on missing key / missing dict / bad value. Never raises.
    A 1.0 default means "no boost, no skip" — the safest fallback.
    """
    if not multipliers:
        return 1.0
    raw = multipliers.get(regime)
    if raw is None:
        return 1.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1.0


def compute_regime_features(spy_daily_bars: pd.DataFrame) -> pd.DataFrame:
    """Append regime features to a SPY daily-bars DataFrame.

    Input: DataFrame with at minimum `bar_date` and `close` columns.
           Input need not be pre-sorted (sort is applied).
    Output: same DataFrame + 4 feature columns (NaN for rows lacking
            sufficient history for the rolling window):

      - sma_50              : 50-day simple moving average of `close`
      - sma_50_slope_10d    : percent change of sma_50 over 10 bars
      - vol_20_ann          : 20-day realized vol of pct_change, annualized %
      - above_sma_50        : bool, True if close > sma_50

    No look-ahead WITHIN the feature computation (row T uses bars ≤ T).
    The CALLER is responsible for shifting by one day when classifying
    day T — see `build_regime_lookup`.
    """
    df = spy_daily_bars.copy().sort_values('bar_date').reset_index(drop=True)
    df['_ret'] = df['close'].pct_change()
    df['sma_50'] = df['close'].rolling(SMA_WINDOW).mean()
    df['sma_50_slope_10d'] = (
        (df['sma_50'] - df['sma_50'].shift(SLOPE_WINDOW))
        / df['sma_50'].shift(SLOPE_WINDOW) * 100.0
    )
    df['vol_20_ann'] = df['_ret'].rolling(VOL_WINDOW).std() * np.sqrt(252) * 100.0
    df['above_sma_50'] = df['close'] > df['sma_50']
    return df.drop(columns=['_ret'])


def build_regime_lookup(
    spy_daily_bars: pd.DataFrame,
    vol_threshold_pct: float = DEFAULT_VOL_THRESHOLD_PCT,
    slope_threshold_pct: float = DEFAULT_SLOPE_THRESHOLD_PCT,
) -> Dict[str, str]:
    """Build {trading_date_str → regime_str} from SPY daily bars.

    Look-ahead safe: the regime assigned to day T is classified from
    the features computed on row T-1 (yesterday's close). Output keys
    are ISO date strings `YYYY-MM-DD` — the TRADING day the regime
    applies to, not the row the features came from.

    Used by BT. PROD uses `compute_regime_features` + `classify_regime`
    directly on the last row of a bars-through-yesterday DataFrame.
    """
    feats = compute_regime_features(spy_daily_bars)
    out: Dict[str, str] = {}
    for i in range(1, len(feats)):
        prev = feats.iloc[i - 1]
        today_raw = feats.iloc[i]['bar_date']
        if hasattr(today_raw, 'strftime'):
            today_str = today_raw.strftime('%Y-%m-%d')
        else:
            today_str = str(today_raw)[:10]

        above = prev['above_sma_50']
        above_bool = None if pd.isna(above) else bool(above)

        vol = prev['vol_20_ann']
        vol_val = None if pd.isna(vol) else float(vol)

        slope = prev['sma_50_slope_10d']
        slope_val = None if pd.isna(slope) else float(slope)

        out[today_str] = classify_regime(
            vol_val, above_bool, slope_val,
            vol_threshold_pct, slope_threshold_pct,
        )
    return out
