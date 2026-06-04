"""ORB touch-and-go filter — shared between backtest and live ORB engine.

Mirrors trading.two_tier_filter / trading.trail_vol_guard / trading.regime_helpers
patterns: pure stateless functions, config dict passed by caller, safe defaults
on None/invalid input.

Rules:
    Rule M (entry-bar weakness): At close of the breakout bar (the 1-min bar
        that triggered the stop-limit BUY), if its close sat in the bottom half
        of its high-low range (bb_close_pos < threshold), exit at next bar
        open. Catches "touch and go" failed breakouts.

    Rule D (bar-1 pullback): After the first post-entry bar closes, if the
        bar's low went >= 0.75R below entry (R = range_high - range_low),
        exit at entry - 0.5R. Catches fast reversal patterns.

    Breakout-bar attribution (2026-06-04): both rules evaluate the MARKET
        breakout bar (first 1-min bar with high > range_high) via
        find_breakout_bar_ts — identical in BT and live. Live previously keyed
        Rule M to the minute of the actual fill, which diverged from BT when a
        stop-limit fill lagged the breakout (measured ~23% of live fills, all
        flipping the tag_bb decision). breakout_bar_source='fill' restores the
        legacy behaviour for rollback.

Walk-forward validated: 8/11 OOS months helped, +$27K OOS lift on 924 trades,
+$44K full-timeline (Jan 2025 - May 2026). Threshold stable at 0.5-0.6 across
all rolling 6-month training windows.

Both BT (study_orb_pipeline_static_lock.py) and LIVE (trading/orb_engine.py)
import from this module — guaranteed identical accept/reject decisions by
construction.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class TouchgoConfig:
    """Touchgo filter configuration.

    Defaults are the validated values (Jan 2025 - May 2026 walk-forward).
    """
    rule_m_enabled: bool = True
    rule_m_threshold: float = 0.5      # bb_close_pos < this -> Rule M fires
    rule_d_enabled: bool = True
    rule_d_revert_R: float = 0.75      # b1_revert >= this R -> Rule D fires
    rule_d_exit_R: float = -0.5        # exit price = entry + this*range_size
    master_enabled: bool = True        # master kill switch (overrides both)
    # Which bar Rule M / Rule D evaluate.
    #   'market' (default, BT-parity): the MARKET breakout bar — the first
    #       1-min bar whose high > range_high (what BT validates). Robust to
    #       fill latency; live and BT agree by construction.
    #   'fill' (legacy rollback): the minute of the actual fill. Pre-2026-06
    #       live behaviour — diverges from BT when fills lag the breakout.
    breakout_bar_source: str = 'market'
    # Late-fill guard: if the actual fill landed more than this many minutes
    # after the market breakout bar, the entry is no longer a clean opening-
    # range breakout — skip touchgo (don't fire a retroactive exit on a bar
    # that closed long before we held the position). BT never sees this gap
    # (it assumes an instant fill at the breakout bar), so this only guards
    # live's pathological late stop-limit fills (e.g. a 34-min-late fill).
    max_breakout_age_min: float = 15.0


def evaluate_rule_m(
    bb_open: float,
    bb_high: float,
    bb_low: float,
    bb_close: float,
    cfg: TouchgoConfig,
) -> Tuple[bool, Optional[float]]:
    """Evaluate Rule M at close of the breakout bar.

    Args:
        bb_open/high/low/close: OHLC of the breakout bar (the bar whose
            high triggered our stop-limit BUY).
        cfg: TouchgoConfig — controls enabled flag + threshold.

    Returns:
        (should_exit, exit_price_or_None). exit_price = bb_close (we exit
        at the breakout bar close ≈ next bar open). Caller applies slippage
        on top.

    Safe-default: returns (False, None) when disabled or on degenerate bars
    (high <= low). Never raises.
    """
    if not cfg.master_enabled or not cfg.rule_m_enabled:
        return (False, None)
    bar_range = bb_high - bb_low
    if bar_range <= 0:
        # Degenerate bar (single-tick or invalid). Don't fire — fail open.
        return (False, None)
    close_pos = (bb_close - bb_low) / bar_range
    if close_pos < cfg.rule_m_threshold:
        return (True, float(bb_close))
    return (False, None)


def evaluate_rule_d(
    entry_price: float,
    b1_low: float,
    range_size: float,
    cfg: TouchgoConfig,
) -> Tuple[bool, Optional[float]]:
    """Evaluate Rule D at close of the first post-entry bar.

    Args:
        entry_price: actual fill price.
        b1_low: low of the first post-entry bar (bar after the breakout bar).
        range_size: 1R unit = range_high - range_low from the 5-min opening
            range.
        cfg: TouchgoConfig — controls enabled flag + revert_R / exit_R.

    Returns:
        (should_exit, exit_price_or_None). exit_price = entry_price +
        cfg.rule_d_exit_R * range_size. Caller applies slippage on top.

    Safe-default: returns (False, None) when disabled or range_size <= 0.
    Never raises.
    """
    if not cfg.master_enabled or not cfg.rule_d_enabled:
        return (False, None)
    if range_size <= 0:
        # Degenerate range. Fail open.
        return (False, None)
    revert_R = (entry_price - b1_low) / range_size
    if revert_R >= cfg.rule_d_revert_R:
        exit_price = entry_price + cfg.rule_d_exit_R * range_size
        return (True, float(exit_price))
    return (False, None)


def find_breakout_bar_ts(
    bars_df: "pd.DataFrame",
    range_high: float,
    range_end_ts=None,
):
    """Timestamp of the MARKET breakout bar = first bar whose high > range_high.

    This is the bar BT keys touchgo Rule M / Rule D to (study_orb_pipeline_
    static_lock.py finds `first bar with high > range_high`), and the bar live
    re-keys to under breakout_bar_source='market'. Sharing this single function
    keeps BT and live identical by construction.

    Args:
        bars_df: DataFrame with 'timestamp' and 'high' columns, in ascending
            timestamp order.
        range_high: the 5-min opening-range high. Strict ``>`` — opening-range
            bars (whose max high IS range_high) never match, so they are
            excluded automatically.
        range_end_ts: optional lower bound (tz-aware). When provided, only bars
            at/after it are considered — excludes any pre-market spike above
            range_high. Callers that pre-window their bars (BT) may pass None.

    Returns:
        UTC-aware pd.Timestamp of the first qualifying bar, or None if no bar
        breaks the range high. Pure / no side effects; never raises.
    """
    if bars_df is None or len(bars_df) == 0:
        return None
    if 'timestamp' not in bars_df.columns or 'high' not in bars_df.columns:
        return None
    try:
        ts = bars_df['timestamp']
        if not pd.api.types.is_datetime64_any_dtype(ts):
            ts = pd.to_datetime(ts, utc=True, errors='coerce')
        elif ts.dt.tz is None:
            ts = ts.dt.tz_localize('UTC')
        highs = pd.to_numeric(bars_df['high'], errors='coerce')
        mask = highs > range_high
        if range_end_ts is not None:
            mask = mask & (ts >= range_end_ts)
        if not mask.any():
            return None
        return ts[mask].iloc[0]
    except (AttributeError, TypeError, ValueError, KeyError):
        # Degenerate / malformed bars — fail closed (no breakout bar found).
        return None


def load_touchgo_config(cfg_dict: Optional[dict]) -> TouchgoConfig:
    """Build TouchgoConfig from raw orb.yaml::filter.touchgo dict.

    Conservative defaults: validated thresholds baked in, so callers that pass
    {} or None still get the shipped policy. Env-var overrides take precedence
    for BT research runs:

        ORB_TOUCHGO_ENABLED=0|false|no|off  -> master disable
        ORB_TOUCHGO_RULE_M_ENABLED=0        -> disable Rule M only
        ORB_TOUCHGO_RULE_M_THRESH=0.4       -> override threshold
        ORB_TOUCHGO_RULE_D_ENABLED=0        -> disable Rule D only
        ORB_TOUCHGO_RULE_D_R=0.75           -> override revert_R trigger
        ORB_TOUCHGO_RULE_D_EXIT_R=-0.5      -> override exit_R level

    Args:
        cfg_dict: raw dict from orb.yaml::filter.touchgo, or None.

    Returns:
        TouchgoConfig with merged values (defaults <- YAML <- env vars).
    """
    if cfg_dict is None:
        cfg_dict = {}
    rule_m_dict = cfg_dict.get('rule_m', {}) if isinstance(cfg_dict, dict) else {}
    rule_d_dict = cfg_dict.get('rule_d', {}) if isinstance(cfg_dict, dict) else {}
    if not isinstance(rule_m_dict, dict):
        rule_m_dict = {}
    if not isinstance(rule_d_dict, dict):
        rule_d_dict = {}

    # Start with defaults, layer YAML
    master_enabled = bool(cfg_dict.get('enabled', True)) if isinstance(cfg_dict, dict) else True
    rule_m_enabled = bool(rule_m_dict.get('enabled', True))
    rule_m_threshold = float(rule_m_dict.get('threshold', 0.5))
    rule_d_enabled = bool(rule_d_dict.get('enabled', True))
    rule_d_revert_R = float(rule_d_dict.get('revert_R', 0.75))
    rule_d_exit_R = float(rule_d_dict.get('exit_R', -0.5))
    breakout_bar_source = str(
        cfg_dict.get('breakout_bar_source', 'market')
    ).strip().lower() if isinstance(cfg_dict, dict) else 'market'
    if breakout_bar_source not in ('market', 'fill'):
        breakout_bar_source = 'market'
    max_breakout_age_min = float(cfg_dict.get('max_breakout_age_min', 15.0)) \
        if isinstance(cfg_dict, dict) else 15.0

    # Env-var overrides
    def _truthy(v: str) -> bool:
        return v.strip().lower() not in ('0', 'false', 'no', 'off', '')

    env_master = os.environ.get('ORB_TOUCHGO_ENABLED')
    if env_master is not None:
        master_enabled = _truthy(env_master)
    env_m_enabled = os.environ.get('ORB_TOUCHGO_RULE_M_ENABLED')
    if env_m_enabled is not None:
        rule_m_enabled = _truthy(env_m_enabled)
    env_m_thresh = os.environ.get('ORB_TOUCHGO_RULE_M_THRESH')
    if env_m_thresh is not None:
        try:
            rule_m_threshold = float(env_m_thresh)
        except ValueError:
            pass
    env_d_enabled = os.environ.get('ORB_TOUCHGO_RULE_D_ENABLED')
    if env_d_enabled is not None:
        rule_d_enabled = _truthy(env_d_enabled)
    env_d_r = os.environ.get('ORB_TOUCHGO_RULE_D_R')
    if env_d_r is not None:
        try:
            rule_d_revert_R = float(env_d_r)
        except ValueError:
            pass
    env_d_exit = os.environ.get('ORB_TOUCHGO_RULE_D_EXIT_R')
    if env_d_exit is not None:
        try:
            rule_d_exit_R = float(env_d_exit)
        except ValueError:
            pass
    env_src = os.environ.get('ORB_TOUCHGO_BREAKOUT_BAR_SOURCE')
    if env_src is not None and env_src.strip().lower() in ('market', 'fill'):
        breakout_bar_source = env_src.strip().lower()
    env_age = os.environ.get('ORB_TOUCHGO_MAX_BREAKOUT_AGE_MIN')
    if env_age is not None:
        try:
            max_breakout_age_min = float(env_age)
        except ValueError:
            pass

    return TouchgoConfig(
        master_enabled=master_enabled,
        rule_m_enabled=rule_m_enabled,
        rule_m_threshold=rule_m_threshold,
        rule_d_enabled=rule_d_enabled,
        rule_d_revert_R=rule_d_revert_R,
        rule_d_exit_R=rule_d_exit_R,
        breakout_bar_source=breakout_bar_source,
        max_breakout_age_min=max_breakout_age_min,
    )
