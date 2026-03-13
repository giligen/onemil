"""
Position manager for tracking open positions and enforcing risk limits.

Enforces:
- Max 3 concurrent positions
- Daily loss limit (-$100)
- No duplicate symbols
- No new positions within 15 min of market close
- Syncs with Alpaca actual positions
"""

import logging
from datetime import datetime, date
from typing import Set, List, Dict, Any, Optional

import pytz

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database

logger = logging.getLogger(__name__)

ET = pytz.timezone('US/Eastern')


class PositionManager:
    """
    Manages trading positions and enforces risk limits.

    Checks before each trade:
    1. Max concurrent positions not exceeded
    2. Daily loss limit not breached
    3. No duplicate symbol positions
    4. Not too close to market close
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        db: Database,
        max_positions: int = 3,
        daily_loss_limit: float = -100.0,
        stop_trading_before_close_min: int = 15,
    ):
        """
        Initialize PositionManager.

        Args:
            alpaca_client: Alpaca API client for position sync
            db: Database for trade records
            max_positions: Maximum concurrent open positions
            daily_loss_limit: Stop trading if daily P&L hits this (negative $)
            stop_trading_before_close_min: Minutes before close to stop new positions
        """
        self.alpaca = alpaca_client
        self.db = db
        self.max_positions = max_positions
        self.daily_loss_limit = daily_loss_limit
        self.stop_trading_before_close_min = stop_trading_before_close_min
        self._traded_symbols: Set[str] = set()

    def can_open_position(self, symbol: str) -> bool:
        """
        Check if a new position can be opened for the given symbol.

        Validates all risk limits:
        1. Max positions not exceeded
        2. Daily loss limit not breached
        3. Symbol not already traded today
        4. Not too close to market close

        Args:
            symbol: Stock symbol to check

        Returns:
            True if position can be opened, False otherwise
        """
        # Check close proximity
        if self._is_near_close():
            logger.warning(
                f"{symbol}: Cannot open position — "
                f"within {self.stop_trading_before_close_min} min of close"
            )
            return False

        # Check duplicate symbol
        if symbol in self._traded_symbols:
            logger.debug(f"{symbol}: Already traded today, skipping")
            return False

        # Check open positions from DB
        today = date.today().isoformat()
        open_trades = self.db.get_open_trades(today)
        open_symbols = {t['symbol'] for t in open_trades}

        if symbol in open_symbols:
            logger.debug(f"{symbol}: Already has open position, skipping")
            return False

        if len(open_trades) >= self.max_positions:
            logger.warning(
                f"{symbol}: Max positions ({self.max_positions}) reached, "
                f"open: {[t['symbol'] for t in open_trades]}"
            )
            return False

        # Check daily loss limit
        daily_pnl = self.db.get_daily_pnl(today)
        if daily_pnl <= self.daily_loss_limit:
            logger.warning(
                f"{symbol}: Daily loss limit breached — "
                f"P&L ${daily_pnl:.2f} <= ${self.daily_loss_limit:.2f}"
            )
            return False

        return True

    def mark_traded(self, symbol: str) -> None:
        """
        Mark a symbol as traded today (prevents re-entry).

        Args:
            symbol: Stock symbol that was traded
        """
        self._traded_symbols.add(symbol)
        logger.debug(f"{symbol}: Marked as traded today")

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get current open positions from Alpaca.

        Returns:
            List of position dicts from Alpaca
        """
        try:
            positions = self.alpaca.get_open_positions()
            logger.debug(f"Open positions: {len(positions)}")
            return positions
        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return []

    def get_open_position_count(self) -> int:
        """Get count of open trades for today from the database."""
        today = date.today().isoformat()
        return len(self.db.get_open_trades(today))

    def reset_daily(self) -> None:
        """Reset daily state (called at start of each trading day)."""
        self._traded_symbols.clear()
        logger.info("Position manager: daily state reset")

    def _is_near_close(self) -> bool:
        """Check if current time is within stop_trading_before_close_min of market close."""
        now_et = datetime.now(ET)
        close_hour = 16
        close_minute = 0
        minutes_to_close = (close_hour * 60 + close_minute) - (now_et.hour * 60 + now_et.minute)
        return minutes_to_close <= self.stop_trading_before_close_min
