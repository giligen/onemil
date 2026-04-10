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

    def _has_conflicting_orders(self, symbol: str) -> bool:
        """Check if symbol has existing open orders on Alpaca (any strategy).

        Prevents wash trades when two strategies target the same symbol.
        """
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            existing = self.alpaca.trading_client.get_orders(filter=req)
            if existing:
                sides = [o.side.value for o in existing]
                logger.warning(
                    f"{symbol}: BLOCKED — {len(existing)} existing order(s) "
                    f"({', '.join(sides)}) would cause wash trade"
                )
                return True
        except Exception as e:
            logger.warning(f"{symbol}: Failed to check existing orders: {e}")
        return False

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
            'strategy': 'bull_flag',
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

    def submit_buy_stop_bracket_order(
        self, plan: TradePlan, slippage_pct: float = 0.02,
        sl_override: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Submit a buy-stop bracket order for a trade plan.

        Places a stop-limit order that triggers at breakout_level and fills
        at breakout_level * (1 + slippage_pct) maximum.

        Slippage limit set to 2% based on 15-month backtest analysis (636 trades,
        Jan 2025 — Mar 2026). Gap-over trades where entry > breakout_level:
          - 0-0.5% gap: 109 trades, 42% win, +$475 avg → profitable
          - 0.5-1% gap:  82 trades, 39% win, +$318 avg → profitable
          - 1-2% gap:    98 trades, 41% win, +$314 avg → profitable
          - 2-5% gap:    43 trades, 23% win, -$134 avg → NET LOSERS
          - >5% gap:      7 trades, 14% win, -$281 avg → garbage
        At 2% cap: $254K total PnL (peak). At 0.5% cap: $198K (-$56K left on table).
        Gap-fill adjustment maintains planned dollar risk regardless of entry gap.

        Args:
            plan: TradePlan with entry (breakout_level), stop, target, sizing
            slippage_pct: Maximum slippage above stop_price (default 2.0%)
            sl_override: If set, use this SL price on the bracket instead of
                plan.stop_loss_price. Used by self-managed stops to place a
                wide safety-net SL while keeping plan.stop_loss_price (and
                risk_per_share/total_risk) at the real stop level.

        Returns:
            Dict with order details if successful, None on failure
        """
        if self._has_conflicting_orders(plan.symbol):
            return None

        stop_price = plan.entry_price
        limit_price = round(stop_price * (1 + slippage_pct), 2)
        bracket_sl = sl_override if sl_override is not None else plan.stop_loss_price

        logger.info(
            f"{plan.symbol}: Submitting buy-stop bracket order — "
            f"BUY {plan.shares} stop @ ${stop_price:.2f}, "
            f"limit ${limit_price:.2f}, "
            f"SL ${bracket_sl:.2f}, TP ${plan.take_profit_price:.2f}"
            f"{' (safety-net SL)' if sl_override else ''}"
        )

        try:
            order = self.alpaca.submit_stop_bracket_order(
                symbol=plan.symbol,
                qty=plan.shares,
                side='buy',
                stop_price=stop_price,
                limit_price=limit_price,
                tp_price=plan.take_profit_price,
                sl_price=bracket_sl,
            )
        except Exception as e:
            logger.error(f"{plan.symbol}: Buy-stop order submission failed: {e}")
            return None

        if order is None:
            logger.error(f"{plan.symbol}: Buy-stop order returned None")
            return None

        order_id = order.get('id', '')
        order_status = order.get('status', 'unknown')

        logger.info(
            f"{plan.symbol}: Buy-stop bracket order submitted — "
            f"ID: {order_id}, status: {order_status}"
        )

        # Save trade record to database (same as bracket order)
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
            'strategy': 'bull_flag',
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
            'stop_price': stop_price,
            'limit_price': limit_price,
            'stop_loss_price': plan.stop_loss_price,
            'take_profit_price': plan.take_profit_price,
            'order_type': 'stop_bracket',
        }

    def submit_buy_stop_order(
        self, plan: TradePlan, slippage_pct: float = 0.02,
    ) -> Optional[Dict[str, Any]]:
        """
        Submit a simple stop-limit buy order (no bracket legs).

        Uses less margin than bracket orders — no TP/SL legs reserved on Alpaca.
        Safety-net SL is submitted separately after fill detection.
        Used when self_managed_stops is enabled (StopMonitor handles real stop).

        Args:
            plan: TradePlan with entry (breakout_level), stop, target, sizing
            slippage_pct: Maximum slippage above stop_price (default 2.0%)

        Returns:
            Dict with order details if successful, None on failure
        """
        if self._has_conflicting_orders(plan.symbol):
            return None

        stop_price = plan.entry_price
        limit_price = round(stop_price * (1 + slippage_pct), 2)

        logger.info(
            f"{plan.symbol}: Submitting buy-stop order (simple) — "
            f"BUY {plan.shares} stop @ ${stop_price:.2f}, "
            f"limit ${limit_price:.2f}"
        )

        try:
            order = self.alpaca.submit_stop_limit_order(
                symbol=plan.symbol,
                qty=plan.shares,
                side='buy',
                stop_price=stop_price,
                limit_price=limit_price,
            )
        except Exception as e:
            logger.error(f"{plan.symbol}: Buy-stop order submission failed: {e}")
            return None

        if order is None:
            logger.error(f"{plan.symbol}: Buy-stop order returned None")
            return None

        order_id = order.get('id', '')
        order_status = order.get('status', 'unknown')

        logger.info(
            f"{plan.symbol}: Buy-stop order submitted (simple) — "
            f"ID: {order_id}, status: {order_status}"
        )

        # Save trade record to database (same fields as bracket order)
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
            'strategy': 'bull_flag',
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
            'stop_price': stop_price,
            'limit_price': limit_price,
            'stop_loss_price': plan.stop_loss_price,
            'take_profit_price': plan.take_profit_price,
            'order_type': 'stop_simple',
        }

