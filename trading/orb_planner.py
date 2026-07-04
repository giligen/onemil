"""ORB trade planner — converts a ranked candidate + range data into a TradePlan.

Pure computation module. No DB/Alpaca side effects. Called from ORBEngine.check_entries
per candidate at 9:35+ ET after the range window closes.

Key responsibilities:
  * Validate spread gate (max_spread_bps)
  * Compute entry_price (range_high + slip buffer)
  * Size via risk-parity: shares = risk_per_trade / stop_distance, capped at per-pos
  * Apply adaptive quintile multiplier (AFTER cap — matches BT behavior)
  * Package everything into OrbTradePlan

Research parity: produces a plan that, when executed at BT-level slippage assumptions,
reproduces the per-trade P&L from our validated features CSV within ~$1 rounding.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional


logger = logging.getLogger(__name__)


@dataclass
class OrbTradePlan:
    """Complete ORB trade plan ready for OrderExecutor + StopMonitor.add_watch."""

    symbol: str
    # Range data
    range_high: float
    range_low: float
    range_size: float       # = range_high - range_low (= 1R)
    # Order params
    entry_price: float      # range_high × (1 + slip_bps/10000); used as stop-limit limit
    stop_price: float       # initial stop = range_low
    shares: int             # final share count (post-cap + post-adaptive-mult)
    position_dollars: float # effective notional at entry
    # Exit spec
    lock_arm_at_r: float    # from config; passed to StopMonitor.add_watch
    lock_stop_r: float
    # Risk metadata
    risk_per_share: float   # entry - stop
    total_risk: float       # risk_per_share × shares — intended $ risk
    # Conviction metadata (for telemetry + DB)
    composite_score: float
    quintile: str           # 'Q1'..'Q5'
    adaptive_mult: float    # multiplier applied to position_dollars
    # BT-parity metadata (optional — defaults to 0 when not set by planner)
    range_open: float = 0.0  # 9:30 bar open — denominator for stop_pct (BT parity)


# Skip reasons (returned in place of a plan when gate fails; used for telegram/logs)
SKIP_SPREAD_GATE = 'spread_gate'
SKIP_ZERO_RANGE = 'zero_range'
SKIP_TOO_SMALL = 'shares_below_one'
SKIP_INSUFFICIENT_BUYING_POWER = 'insufficient_buying_power'


@dataclass
class PlannerReject:
    """Returned when a plan cannot be built; carries reason + metrics for logs/telegram."""
    symbol: str
    reason: str
    details: Dict[str, float]


class OrbTradePlanner:
    """Builds ORB trade plans from ranked candidates.

    Stateless — all configuration passed at init. Reusable across days.
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: dict parsed from orb.yaml. Must contain 'entry', 'exit', 'sizing' sections.
        """
        entry_cfg = cfg.get('entry', {})
        exit_cfg = cfg.get('exit', {})
        sizing_cfg = cfg.get('sizing', {})

        self.entry_slip_bps = float(entry_cfg.get('entry_slip_bps', 30))
        self.max_spread_bps = float(entry_cfg.get('max_spread_bps', 300))

        self.lock_arm_at_r = float(exit_cfg.get('lock_arm_at_r', 1.5))
        self.lock_stop_r = float(exit_cfg.get('lock_stop_r', 1.0))

        self.account_budget_usd = float(sizing_cfg.get('account_budget_usd', 100_000))
        self.max_concurrent = int(sizing_cfg.get('max_concurrent', 4))
        self.risk_per_trade_usd = float(sizing_cfg.get('risk_per_trade_usd', 3_000))
        self.min_stop_pct = float(sizing_cfg.get('min_stop_pct', 1.0))

        if self.max_concurrent <= 0:
            raise ValueError(f"max_concurrent must be > 0, got {self.max_concurrent}")
        if self.risk_per_trade_usd <= 0:
            raise ValueError(f"risk_per_trade_usd must be > 0, got {self.risk_per_trade_usd}")

        self.per_pos_cap_usd = self.account_budget_usd / self.max_concurrent

    def build(
        self,
        symbol: str,
        range_high: float,
        range_low: float,
        composite_score: float,
        quintile: str,
        adaptive_mult: float,
        range_open: float = 0.0,
        spread_bps: Optional[float] = None,
    ):
        """Build a plan or return PlannerReject.

        Args:
            symbol: ticker
            range_high: 5-min opening range high (9:30-9:35 ET)
            range_low: 5-min opening range low
            composite_score: filter composite z-score
            quintile: Q1..Q5 bucket
            adaptive_mult: per-quintile multiplier (Q5 already capped at 1.5 by loader)
            range_open: price at open of 9:30 bar (BT parity: used as denominator
                in stop_pct = (range_high - range_low) / range_open × 100).
                Defaults to range_high if 0 (pre-fix fallback).
            spread_bps: current bid-ask spread in bps (None = spread gate not applied)

        Returns:
            OrbTradePlan on success, PlannerReject on gate fail.
        """
        details = {
            'range_high': range_high, 'range_low': range_low,
            'composite': composite_score, 'spread_bps': spread_bps or 0.0,
        }

        # Gate 1: spread
        if spread_bps is not None and spread_bps > self.max_spread_bps:
            return PlannerReject(symbol=symbol, reason=SKIP_SPREAD_GATE, details=details)

        # Gate 2: range sanity
        range_size = range_high - range_low
        if range_size <= 0 or range_high <= 0:
            return PlannerReject(symbol=symbol, reason=SKIP_ZERO_RANGE, details=details)

        # Entry = range_high × (1 + slip_bps/10000)
        entry_price = range_high * (1.0 + self.entry_slip_bps / 10000.0)

        # BT-parity sizing: stop_pct relative to range_open (NOT entry_price).
        # Source: study_orb_100k_defended.py::apply_sizing uses
        #   stop_pct = range_size_pct = (range_high - range_low) / range_open × 100
        # This is what BT validated on (Calmar 18.90x under static_lock_1R). PROD's earlier
        # (entry - range_low) / entry formula is mathematically more correct
        # for risk parity, but it produces different sizing on wide-range
        # trades, violating BT parity.
        ref_open = range_open if range_open > 0 else range_high
        range_size_pct = range_size / ref_open * 100.0
        sizing_stop_pct = max(range_size_pct, self.min_stop_pct)

        # Risk-parity uncapped position: spend enough to put $risk_per_trade at the stop
        uncapped_position = self.risk_per_trade_usd / (sizing_stop_pct / 100.0)

        # Cap at per-position limit BEFORE adaptive mult (matches BT study_orb_100k_defended.py)
        position_before_mult = min(uncapped_position, self.per_pos_cap_usd)

        # Apply adaptive multiplier (may push above per-pos cap — BT behavior)
        position_dollars = position_before_mult * adaptive_mult

        shares = int(math.floor(position_dollars / entry_price))
        if shares < 1:
            details['position_dollars'] = position_dollars
            details['entry_price'] = entry_price
            return PlannerReject(symbol=symbol, reason=SKIP_TOO_SMALL, details=details)

        # risk_per_share uses ACTUAL stop distance for telemetry correctness.
        # Sizing was BT-parity via range_size_pct above; this field is the real
        # $ at stake per share (entry - stop).
        risk_per_share = entry_price - range_low
        total_risk = risk_per_share * shares

        return OrbTradePlan(
            symbol=symbol,
            range_high=range_high,
            range_low=range_low,
            range_size=range_size,
            range_open=ref_open,
            entry_price=round(entry_price, 2),
            stop_price=round(range_low, 2),
            shares=shares,
            position_dollars=position_dollars,
            lock_arm_at_r=self.lock_arm_at_r,
            lock_stop_r=self.lock_stop_r,
            risk_per_share=risk_per_share,
            total_risk=total_risk,
            composite_score=composite_score,
            quintile=quintile,
            adaptive_mult=adaptive_mult,
        )
