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

    return TouchgoConfig(
        master_enabled=master_enabled,
        rule_m_enabled=rule_m_enabled,
        rule_m_threshold=rule_m_threshold,
        rule_d_enabled=rule_d_enabled,
        rule_d_revert_R=rule_d_revert_R,
        rule_d_exit_R=rule_d_exit_R,
    )
