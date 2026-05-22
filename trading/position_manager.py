"""
Position manager for tracking open positions and enforcing risk limits.

Enforces:
- Max concurrent positions (per-strategy when a strategy tag is set)
- Daily loss limit (-$100)
- No duplicate symbols
- No new positions within 15 min of market close
- No new positions during midday dead zone (11:30-14:00 ET)
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

# Strategy tags that may own rows in the DB `trades.strategy` column. Used to
# validate the `strategy` constructor arg: a typo there would make the
# per-strategy position cap silently count zero open trades — i.e. never fire —
# so an unrecognized value is a hard error, not a quiet fallback.
KNOWN_STRATEGIES = ('bull_flag', 'macd_wave', 'orb')


class PositionManager:
    """
    Manages trading positions and enforces risk limits.

    Checks before each trade:
    1. Not in midday dead zone (11:30-14:00 ET)
    2. Max concurrent positions not exceeded
    3. Daily loss limit not breached
    4. No duplicate symbol positions
    5. Not too close to market close
    """

    # Midday dead zone: 11:30-14:00 ET has 37.5% WR vs 62.5% outside
    MIDDAY_START_MINUTES = 11 * 60 + 30   # 11:30 ET
    MIDDAY_END_MINUTES = 14 * 60          # 14:00 ET

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        db: Database,
        max_positions: int = 3,
        daily_loss_limit: float = -100.0,
        stop_trading_before_close_min: int = 15,
        skip_midday: bool = True,
        max_consecutive_losses: int = 2,
        strategy: Optional[str] = None,
    ):
        """
        Initialize PositionManager.

        Args:
            alpaca_client: Alpaca API client for position sync
            db: Database for trade records
            max_positions: Maximum concurrent open positions
            daily_loss_limit: Stop trading if daily P&L hits this (negative $)
            stop_trading_before_close_min: Minutes before close to stop new positions
            skip_midday: Skip 11:30-14:00 ET entries (backtest-proven dead zone)
            max_consecutive_losses: Stop trading for the day after N consecutive
                losses. Reduces worst-day drawdown by 40% at 8% P&L cost.
            strategy: Strategy tag this manager governs (e.g. 'bull_flag').
                When set, the max_positions cap counts ONLY this strategy's
                open trades, so strategies sharing one Alpaca account don't
                consume each other's slots. None counts all strategies (legacy).

        Raises:
            ValueError: if ``strategy`` is not None and not a recognized tag.
        """
        if strategy is not None and strategy not in KNOWN_STRATEGIES:
            raise ValueError(
                f"PositionManager: unrecognized strategy {strategy!r}. The "
                f"per-strategy position cap would silently never fire. "
                f"Expected one of {KNOWN_STRATEGIES} or None."
            )
        self.alpaca = alpaca_client
        self.db = db
        self.max_positions = max_positions
        self.daily_loss_limit = daily_loss_limit
        self.stop_trading_before_close_min = stop_trading_before_close_min
        self.skip_midday = skip_midday
        self.max_consecutive_losses = max_consecutive_losses
        self.strategy = strategy
        self._traded_symbols: Set[str] = set()
        self._consecutive_losses: int = 0
        self._stopped_for_day: bool = False
        self._current_et_date: Optional[date] = None

    def _check_day_rollover(self) -> None:
        """Auto-reset daily state when the ET calendar date changes.

        Why: long-running services that don't restart at the day boundary
        leak state from yesterday into today — _stopped_for_day,
        _traded_symbols, and _consecutive_losses all need to clear.
        Idempotent on the first call (no false reset on init).
        """
        today_et = datetime.now(ET).date()
        if self._current_et_date is None:
            self._current_et_date = today_et
            return
        if self._current_et_date != today_et:
            logger.warning(
                f"Position manager: ET date rolled "
                f"{self._current_et_date} → {today_et} — auto-resetting daily state "
                f"(was: stopped_for_day={self._stopped_for_day}, "
                f"consecutive_losses={self._consecutive_losses}, "
                f"traded={len(self._traded_symbols)})"
            )
            self.reset_daily()
            self._current_et_date = today_et

    def record_trade_pnl(self, pnl: float) -> None:
        """Record completed trade P&L for consecutive-loss tracking."""
        self._check_day_rollover()
        if pnl > 0:
            self._consecutive_losses = 0
            return
        self._consecutive_losses += 1
        # max_consecutive_losses <= 0 means the circuit breaker is disabled.
        # Without this guard, 0 silently triggers on the first loss
        # (1 >= 0 is True) — that's a footgun, not a sane default.
        if self.max_consecutive_losses <= 0:
            return
        if self._consecutive_losses >= self.max_consecutive_losses:
            self._stopped_for_day = True
            logger.warning(
                f"CONSECUTIVE LOSS LIMIT: {self._consecutive_losses} losses in a row "
                f"— done for the day"
            )

    def can_open_position(self, symbol: str) -> bool:
        """
        Check if a new position can be opened for the given symbol.

        Validates all risk limits:
        0. Circuit breaker not active
        1. Not in midday dead zone (11:30-14:00 ET)
        2. Max positions not exceeded
        3. Daily loss limit not breached
        4. Symbol not already traded today
        5. Not too close to market close

        Args:
            symbol: Stock symbol to check

        Returns:
            True if position can be opened, False otherwise
        """
        self._check_day_rollover()

        # Check consecutive loss limit
        if self._stopped_for_day:
            logger.warning(
                f"{symbol}: Stopped for the day — "
                f"{self._consecutive_losses} consecutive losses"
            )
            return False

        # Check midday dead zone
        if self.skip_midday and self._is_midday():
            logger.info(
                f"{symbol}: Skipping — midday dead zone (11:30-14:00 ET), "
                f"backtest shows 37.5% WR in this window"
            )
            return False

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

        # Check open positions from DB.
        today = date.today().isoformat()

        # Duplicate-symbol guard — CROSS-STRATEGY on purpose: two engines must
        # never both hold the same symbol on one Alpaca account. The broker
        # nets same-symbol orders into a single position, and the strategies'
        # StopMonitors would then fight over that one position.
        all_open_trades = self.db.get_open_trades(today)
        if symbol in {t['symbol'] for t in all_open_trades}:
            logger.debug(f"{symbol}: Already has an open position (any strategy), skipping")
            return False

        # Max-concurrent cap — PER-STRATEGY: max_positions governs only this
        # manager's own strategy. On a shared Alpaca account, counting other
        # strategies' positions here would let one strategy (e.g. ORB filling
        # its slots at the open) lock this one out for the whole day.
        strategy_open_trades = self.db.get_open_trades(today, strategy=self.strategy)
        if len(strategy_open_trades) >= self.max_positions:
            logger.warning(
                f"{symbol}: Max positions ({self.max_positions}) reached for "
                f"strategy={self.strategy or 'all'}, "
                f"open: {[t['symbol'] for t in strategy_open_trades]}"
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
        """Get count of this strategy's open trades for today from the database.

        Scoped to this manager's ``strategy`` when one is set; counts all
        strategies when ``strategy`` is None (legacy behavior).
        """
        today = date.today().isoformat()
        return len(self.db.get_open_trades(today, strategy=self.strategy))

    def reset_daily(self) -> None:
        """Reset daily state (called at start of each trading day)."""
        self._traded_symbols.clear()
        self._consecutive_losses = 0
        self._stopped_for_day = False
        logger.info("Position manager: daily state reset")

    def _is_midday(self) -> bool:
        """Check if current time is in the 11:30-14:00 ET dead zone."""
        now_et = datetime.now(ET)
        current_minutes = now_et.hour * 60 + now_et.minute
        return self.MIDDAY_START_MINUTES <= current_minutes < self.MIDDAY_END_MINUTES

    def _is_near_close(self) -> bool:
        """Check if current time is within stop_trading_before_close_min of market close."""
        now_et = datetime.now(ET)
        close_hour = 16
        close_minute = 0
        minutes_to_close = (close_hour * 60 + close_minute) - (now_et.hour * 60 + now_et.minute)
        return minutes_to_close <= self.stop_trading_before_close_min
