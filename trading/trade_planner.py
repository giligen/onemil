"""
Trade planner for bull flag patterns.

Converts a detected BullFlagPattern into a TradePlan with:
- Entry at breakout level
- Stop loss below flag low (capped by max_risk_per_share or max_risk_pct)
- Target at configurable R:R ratio
- Position sizing: fixed_investment (dollars/price) or fixed_risk (risk_budget/risk_per_share)
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

from trading.pattern_detector import BullFlagPattern

logger = logging.getLogger(__name__)


@dataclass
class TradePlan:
    """Complete trade plan derived from a bull flag pattern."""

    symbol: str
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    risk_per_share: float
    reward_per_share: float
    risk_reward_ratio: float
    shares: int
    total_risk: float
    pattern: BullFlagPattern


class TradePlanner:
    """
    Creates trade plans from detected bull flag patterns.

    Supports two sizing modes:
    - fixed_investment: shares = floor(dollars / entry_price) [original behavior]
    - fixed_risk: shares = floor(risk_per_trade / risk_per_share) [normalized risk]

    Stop thresholds can be flat dollar amounts or percentage of entry price.
    Target is configurable via min_risk_reward ratio.
    """

    def __init__(
        self,
        position_size_dollars: float = 10000,
        max_shares: int = 10000,
        max_risk_per_share: float = 0.20,
        min_risk_per_share: float = 0.02,
        min_risk_reward: float = 2.0,
        sizing_mode: str = "fixed_investment",
        risk_per_trade: float = 500.0,
        min_risk_pct: Optional[float] = 0.005,
        max_risk_pct: Optional[float] = None,
    ):
        """
        Initialize TradePlanner.

        Args:
            position_size_dollars: Dollar amount per position (fixed_investment mode)
            max_shares: Maximum shares per position
            max_risk_per_share: Max risk per share flat dollar (Ross's 20-cent rule)
            min_risk_per_share: Min risk per share flat dollar — rejects noise stops
            min_risk_reward: Minimum acceptable risk/reward ratio (also used for target calc)
            sizing_mode: "fixed_investment" or "fixed_risk"
            risk_per_trade: Dollar risk budget per trade (fixed_risk mode)
            min_risk_pct: Min risk as fraction of entry price (e.g., 0.01 = 1%); overrides min_risk_per_share
            max_risk_pct: Max risk as fraction of entry price (e.g., 0.05 = 5%); overrides max_risk_per_share
        """
        if sizing_mode not in ("fixed_investment", "fixed_risk"):
            raise ValueError(f"sizing_mode must be 'fixed_investment' or 'fixed_risk', got '{sizing_mode}'")

        self.position_size_dollars = position_size_dollars
        self.max_shares = max_shares
        self.max_risk_per_share = max_risk_per_share
        self.min_risk_per_share = min_risk_per_share
        self.min_risk_reward = min_risk_reward
        self.sizing_mode = sizing_mode
        self.risk_per_trade = risk_per_trade
        self.min_risk_pct = min_risk_pct
        self.max_risk_pct = max_risk_pct

    @classmethod
    def from_config(cls) -> 'TradePlanner':
        """Create a TradePlanner with parameters from config.yaml."""
        try:
            from config import Config
            cfg = Config._load_yaml_only()
            trading = cfg.get("trading", {})
            min_risk_pct = trading.get("min_risk_pct")
            max_risk_pct = trading.get("max_risk_pct")
            return cls(
                position_size_dollars=float(trading.get("position_size_dollars", 10000)),
                max_shares=int(trading.get("max_shares", 10000)),
                max_risk_per_share=float(trading.get("max_risk_per_share", 0.20)),
                min_risk_per_share=float(trading.get("min_risk_per_share", 0.02)),
                min_risk_reward=float(trading.get("min_risk_reward", 2.0)),
                sizing_mode=str(trading.get("sizing_mode", "fixed_investment")),
                risk_per_trade=float(trading.get("risk_per_trade", 500.0)),
                min_risk_pct=float(min_risk_pct) if min_risk_pct is not None else None,
                max_risk_pct=float(max_risk_pct) if max_risk_pct is not None else None,
            )
        except Exception as e:
            logger.warning(f"Failed to load config.yaml, using defaults: {e}")
            return cls()

    def create_plan(self, pattern: BullFlagPattern) -> Optional[TradePlan]:
        """
        Create a trade plan from a detected bull flag pattern.

        Args:
            pattern: Detected BullFlagPattern

        Returns:
            TradePlan if valid, None if rejected
        """
        entry_price = pattern.breakout_level

        if entry_price <= 0:
            logger.warning(f"{pattern.symbol}: Invalid entry price {entry_price}")
            return None

        # Stop loss = below flag low by 1 penny
        natural_stop = pattern.flag_low - 0.01
        natural_risk = entry_price - natural_stop

        if natural_risk <= 0:
            logger.debug(f"{pattern.symbol}: Stop ({natural_stop:.2f}) >= entry ({entry_price:.2f}), rejecting")
            return None

        # Compute effective min risk: max(absolute_floor, pct_of_price)
        # This scales with price — tight stops are ok on cheap stocks,
        # but noise on expensive ones.
        if self.min_risk_pct is not None:
            effective_min_risk = max(self.min_risk_per_share, entry_price * self.min_risk_pct)
        else:
            effective_min_risk = self.min_risk_per_share
        effective_max_risk = (
            entry_price * self.max_risk_pct
            if self.max_risk_pct is not None
            else self.max_risk_per_share
        )

        # Hard rejection threshold: 2.5x max risk (replaces hardcoded $0.50)
        hard_reject = entry_price * self.max_risk_pct * 2.5 if self.max_risk_pct is not None else 0.50
        if natural_risk > hard_reject:
            logger.debug(
                f"{pattern.symbol}: Natural risk ${natural_risk:.2f} > "
                f"hard reject ${hard_reject:.2f}, pattern too volatile, rejecting"
            )
            return None

        # Reject if stop is too tight — will get stopped out on noise
        if natural_risk < effective_min_risk:
            logger.debug(
                f"{pattern.symbol}: Natural risk ${natural_risk:.2f} < "
                f"min ${effective_min_risk:.2f}, noise stop, rejecting"
            )
            return None

        # Cap risk at effective max
        if natural_risk > effective_max_risk:
            stop_loss_price = entry_price - effective_max_risk
            risk_per_share = effective_max_risk
            logger.debug(
                f"{pattern.symbol}: Capped stop from {natural_stop:.2f} "
                f"to {stop_loss_price:.2f} (max risk ${effective_max_risk:.2f})"
            )
        else:
            stop_loss_price = natural_stop
            risk_per_share = natural_risk

        # Target = R:R * risk (uses min_risk_reward as target multiplier)
        take_profit_price = entry_price + (self.min_risk_reward * risk_per_share)

        reward_per_share = take_profit_price - entry_price
        risk_reward_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0

        if risk_reward_ratio < self.min_risk_reward:
            logger.debug(
                f"{pattern.symbol}: R:R {risk_reward_ratio:.1f} < {self.min_risk_reward}, rejecting"
            )
            return None

        # Position sizing
        if self.sizing_mode == "fixed_risk":
            shares = math.floor(self.risk_per_trade / risk_per_share)
            if shares > self.max_shares:
                logger.warning(
                    f"{pattern.symbol}: fixed_risk shares {shares} exceeds "
                    f"max_shares {self.max_shares}, capping (risk budget distorted)"
                )
        else:
            shares = math.floor(self.position_size_dollars / entry_price)
        shares = min(shares, self.max_shares)

        if shares <= 0:
            logger.debug(f"{pattern.symbol}: Zero shares at ${entry_price:.2f}, rejecting")
            return None

        total_risk = risk_per_share * shares

        plan = TradePlan(
            symbol=pattern.symbol,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            risk_per_share=risk_per_share,
            reward_per_share=reward_per_share,
            risk_reward_ratio=risk_reward_ratio,
            shares=shares,
            total_risk=total_risk,
            pattern=pattern,
        )

        logger.info(
            f"{pattern.symbol}: TRADE PLAN — "
            f"Entry ${entry_price:.2f}, Stop ${stop_loss_price:.2f}, "
            f"Target ${take_profit_price:.2f}, "
            f"R:R {risk_reward_ratio:.1f}, {shares} shares, "
            f"Risk ${total_risk:.2f}"
        )
        return plan
