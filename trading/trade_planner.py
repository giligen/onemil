"""
Trade planner for bull flag patterns.

Converts a detected BullFlagPattern into a TradePlan with:
- Entry at breakout level
- Stop loss below flag low (max 20 cents from entry per Ross's rule)
- Target at 2:1 R:R or pole height projection (whichever is larger)
- Position sizing based on dollar amount and share cap
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

    Applies Ross Cameron's risk management rules:
    - Stop below flag low, capped at 20 cents from entry
    - Target = max(2:1 R:R, pole height projection)
    - Position size = floor(dollars / entry), capped at max shares
    - Rejects if natural stop > 50 cents (pattern too volatile)
    - Rejects if R:R < 2.0
    """

    def __init__(
        self,
        position_size_dollars: float = 500,
        max_shares: int = 1000,
        max_risk_per_share: float = 0.20,
        min_risk_reward: float = 2.0,
    ):
        """
        Initialize TradePlanner.

        Args:
            position_size_dollars: Dollar amount per position
            max_shares: Maximum shares per position
            max_risk_per_share: Max risk per share (Ross's 20-cent rule)
            min_risk_reward: Minimum acceptable risk/reward ratio
        """
        self.position_size_dollars = position_size_dollars
        self.max_shares = max_shares
        self.max_risk_per_share = max_risk_per_share
        self.min_risk_reward = min_risk_reward

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

        # Reject if natural stop is too far away (pattern too volatile for safe trading)
        if natural_risk > 0.50:
            logger.debug(
                f"{pattern.symbol}: Natural risk ${natural_risk:.2f} > $0.50, "
                f"pattern too volatile, rejecting"
            )
            return None

        # Cap risk at max_risk_per_share (Ross's 20-cent rule)
        if natural_risk > self.max_risk_per_share:
            stop_loss_price = entry_price - self.max_risk_per_share
            risk_per_share = self.max_risk_per_share
            logger.debug(
                f"{pattern.symbol}: Capped stop from {natural_stop:.2f} "
                f"to {stop_loss_price:.2f} (20¢ rule)"
            )
        else:
            stop_loss_price = natural_stop
            risk_per_share = natural_risk

        # Target = max(2:1 R:R, pole height projection)
        target_2_to_1 = entry_price + (2 * risk_per_share)
        target_pole = entry_price + pattern.pole_height
        take_profit_price = max(target_2_to_1, target_pole)

        reward_per_share = take_profit_price - entry_price
        risk_reward_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0

        if risk_reward_ratio < self.min_risk_reward:
            logger.debug(
                f"{pattern.symbol}: R:R {risk_reward_ratio:.1f} < {self.min_risk_reward}, rejecting"
            )
            return None

        # Position sizing
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
