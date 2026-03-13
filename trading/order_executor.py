"""
Order executor for submitting bracket orders via Alpaca.

Handles:
- Bracket order submission (entry + stop loss + take profit)
- Order status tracking
- Trade record creation in database
"""

import json
import logging
from datetime import date, datetime, timezone
from typing import Optional, Dict, Any

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.trade_planner import TradePlan

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Executes trade plans by submitting bracket orders to Alpaca.

    Each bracket order consists of:
    - Entry: limit buy at entry_price
    - Stop loss: sell stop at stop_loss_price
    - Take profit: sell limit at take_profit_price
    All with TimeInForce.DAY (expire at close).
    """

    def __init__(self, alpaca_client: AlpacaClient, db: Database):
        """
        Initialize OrderExecutor.

        Args:
            alpaca_client: Alpaca API client for order submission
            db: Database for trade record persistence
        """
        self.alpaca = alpaca_client
        self.db = db

    def submit_bracket_order(self, plan: TradePlan) -> Optional[Dict[str, Any]]:
        """
        Submit a bracket order for a trade plan.

        Args:
            plan: TradePlan with entry, stop, target, and sizing

        Returns:
            Dict with order details if successful, None on failure
        """
        logger.info(
            f"{plan.symbol}: Submitting bracket order — "
            f"BUY {plan.shares} @ ${plan.entry_price:.2f}, "
            f"SL ${plan.stop_loss_price:.2f}, TP ${plan.take_profit_price:.2f}"
        )

        try:
            order = self.alpaca.submit_bracket_order(
                symbol=plan.symbol,
                qty=plan.shares,
                side='buy',
                limit_price=plan.entry_price,
                tp_price=plan.take_profit_price,
                sl_price=plan.stop_loss_price,
            )
        except Exception as e:
            logger.error(f"{plan.symbol}: Bracket order submission failed: {e}")
            return None

        if order is None:
            logger.error(f"{plan.symbol}: Bracket order returned None")
            return None

        order_id = order.get('id', '')
        order_status = order.get('status', 'unknown')

        logger.info(
            f"{plan.symbol}: Bracket order submitted — "
            f"ID: {order_id}, status: {order_status}"
        )

        # Save trade record to database
        pattern_data = json.dumps({
            'pole_start_idx': plan.pattern.pole_start_idx,
            'pole_end_idx': plan.pattern.pole_end_idx,
            'flag_start_idx': plan.pattern.flag_start_idx,
            'flag_end_idx': plan.pattern.flag_end_idx,
            'pole_low': plan.pattern.pole_low,
            'pole_high': plan.pattern.pole_high,
            'pole_height': plan.pattern.pole_height,
            'pole_gain_pct': plan.pattern.pole_gain_pct,
            'flag_low': plan.pattern.flag_low,
            'flag_high': plan.pattern.flag_high,
            'retracement_pct': plan.pattern.retracement_pct,
            'pullback_candle_count': plan.pattern.pullback_candle_count,
            'avg_pole_volume': plan.pattern.avg_pole_volume,
            'avg_flag_volume': plan.pattern.avg_flag_volume,
            'breakout_level': plan.pattern.breakout_level,
        })

        now = datetime.now(timezone.utc)
        trade_record = {
            'trade_date': date.today().isoformat(),
            'symbol': plan.symbol,
            'side': 'buy',
            'entry_price': plan.entry_price,
            'stop_loss_price': plan.stop_loss_price,
            'take_profit_price': plan.take_profit_price,
            'shares': plan.shares,
            'risk_per_share': plan.risk_per_share,
            'total_risk': plan.total_risk,
            'risk_reward_ratio': plan.risk_reward_ratio,
            'order_id': order_id,
            'order_status': order_status,
            'fill_price': None,
            'filled_at': None,
            'exit_price': None,
            'exit_reason': None,
            'exited_at': None,
            'pnl': None,
            'pnl_pct': None,
            'pattern_data': pattern_data,
            'created_at': now,
            'updated_at': now,
        }

        try:
            trade_id = self.db.save_trade(trade_record)
            logger.info(f"{plan.symbol}: Trade record saved (id={trade_id})")
        except Exception as e:
            logger.error(f"{plan.symbol}: Failed to save trade record: {e}")

        return {
            'order_id': order_id,
            'status': order_status,
            'symbol': plan.symbol,
            'shares': plan.shares,
            'entry_price': plan.entry_price,
            'stop_loss_price': plan.stop_loss_price,
            'take_profit_price': plan.take_profit_price,
        }
