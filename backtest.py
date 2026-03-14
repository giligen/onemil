"""
Backtesting engine for bull flag momentum strategy.

Given a stock symbol and date, fetches the day's 1-minute bars, runs a
sliding-window simulation minute-by-minute, detects patterns, simulates
trades, and reports P&L.

Usage:
    python backtest.py PLYX 2026-03-13
    python backtest.py SVCO 2026-03-13 --verbose
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv

from data_sources.alpaca_client import AlpacaClient
from trading.pattern_detector import BullFlagDetector, BullFlagPattern, BullFlagSetup
from trading.trade_planner import TradePlanner, TradePlan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SimulatedTrade:
    """Result of simulating a trade through historical bars."""

    symbol: str
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    shares: int
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'target', 'stop', 'eod', 'force_close'
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0
    plan: Optional[TradePlan] = None
    # Entry bar OHLCV — the actual 1-min bar where entry was triggered
    entry_bar_open: Optional[float] = None
    entry_bar_high: Optional[float] = None
    entry_bar_low: Optional[float] = None
    entry_bar_close: Optional[float] = None
    entry_bar_volume: Optional[int] = None
    # Realistic entry tracking (buy-stop mode)
    planned_entry: Optional[float] = None  # the breakout_level we targeted
    entry_gap: float = 0.0  # realistic_entry - breakout_level (slippage)
    # Partial profit tracking (Ross Cameron exit strategy)
    partial_exit_taken: bool = False
    partial_exit_time: Optional[datetime] = None
    partial_exit_price: Optional[float] = None
    partial_shares: int = 0
    partial_pnl: float = 0.0
    remaining_shares: int = 0
    breakeven_stop_active: bool = False


@dataclass
class PatternDetection:
    """Record of a pattern detected during backtest scanning."""

    bar_index: int
    timestamp: datetime
    pattern: BullFlagPattern


@dataclass
class PendingBuyStop:
    """A pending buy-stop order waiting for breakout to trigger fill."""

    setup: BullFlagSetup
    plan: TradePlan
    placed_at_bar_idx: int
    breakout_level: float


@dataclass
class BacktestResult:
    """Complete result from a backtest run."""

    symbol: str
    trade_date: str
    total_bars: int
    patterns_detected: int
    trades_simulated: List[SimulatedTrade] = field(default_factory=list)
    pattern_details: List[PatternDetection] = field(default_factory=list)

    @property
    def summary_pnl(self) -> float:
        """Total P&L across all simulated trades."""
        return sum(t.pnl for t in self.trades_simulated)


# ---------------------------------------------------------------------------
# Trade Simulator
# ---------------------------------------------------------------------------


class TradeSimulator:
    """
    Simulates a trade by walking forward through bars from entry.

    Rules:
    - Entry at plan.entry_price (or override price for realistic buy-stop fills)
    - Each bar: check force_close, then stop, then target (conservative on ambiguity)
    - End of day: exit at last bar's close
    """

    def __init__(
        self,
        force_close_time_utc: Optional[float] = None,
        partial_profit_enabled: bool = False,
        partial_profit_r_multiple: float = 1.0,
        partial_profit_fraction: float = 0.5,
    ):
        """
        Initialize TradeSimulator.

        Args:
            force_close_time_utc: UTC hour (e.g. 19.75 = 15:45 ET) to force
                close all positions. None disables force-close.
            partial_profit_enabled: Enable partial profit exit at +NR, then
                move stop to breakeven on remaining shares.
            partial_profit_r_multiple: Take partial profit at this R multiple
                (default 1.0 = +1R).
            partial_profit_fraction: Fraction of shares to sell at partial
                target (default 0.5 = half).
        """
        self.force_close_time_utc = force_close_time_utc
        self.partial_profit_enabled = partial_profit_enabled
        self.partial_profit_r_multiple = partial_profit_r_multiple
        self.partial_profit_fraction = partial_profit_fraction

    def simulate(
        self,
        plan: TradePlan,
        bars: pd.DataFrame,
        entry_bar_idx: int,
        entry_price_override: Optional[float] = None,
    ) -> SimulatedTrade:
        """
        Simulate a trade from entry_bar_idx through remaining bars.

        Args:
            plan: The trade plan with entry/stop/target
            bars: Full day's bars DataFrame
            entry_bar_idx: Index of the bar where entry occurs
            entry_price_override: If set, use this as actual entry price instead
                of plan.entry_price (for realistic buy-stop fills at
                max(bar_open, breakout_level))

        Returns:
            SimulatedTrade with fill details and P&L
        """
        entry_bar = bars.iloc[entry_bar_idx]
        actual_entry = entry_price_override if entry_price_override is not None else plan.entry_price

        trade = SimulatedTrade(
            symbol=plan.symbol,
            entry_time=entry_bar['timestamp'],
            entry_price=actual_entry,
            stop_loss=plan.stop_loss_price,
            take_profit=plan.take_profit_price,
            shares=plan.shares,
            plan=plan,
            entry_bar_open=float(entry_bar['open']),
            entry_bar_high=float(entry_bar['high']),
            entry_bar_low=float(entry_bar['low']),
            entry_bar_close=float(entry_bar['close']),
            entry_bar_volume=int(entry_bar['volume']),
            planned_entry=plan.entry_price,
            entry_gap=actual_entry - plan.entry_price,
        )

        last_bar_idx = len(bars) - 1

        if self.partial_profit_enabled:
            return self._simulate_with_partial(trade, plan, bars, entry_bar_idx, actual_entry)

        for i in range(entry_bar_idx + 1, len(bars)):
            bar = bars.iloc[i]

            # Force close check
            if self.force_close_time_utc is not None:
                bar_ts = bar['timestamp']
                bar_hour = bar_ts.hour if hasattr(bar_ts, 'hour') else 0
                bar_minute = bar_ts.minute if hasattr(bar_ts, 'minute') else 0
                bar_time_utc = bar_hour + bar_minute / 60.0
                if bar_time_utc >= self.force_close_time_utc:
                    self._exit_trade(trade, bar, 'force_close', bar['open'])
                    logger.debug(f"  Bar {i}: force close at ${bar['open']:.2f}")
                    return trade

            bar_low = bar['low']
            bar_high = bar['high']

            hit_stop = bar_low <= trade.stop_loss
            hit_target = bar_high >= trade.take_profit

            if hit_stop and hit_target:
                # Same-bar ambiguity: conservative = assume stop hit first
                self._exit_trade(trade, bar, 'stop', trade.stop_loss)
                logger.debug(
                    f"  Bar {i}: ambiguous (stop & target) → stopped out "
                    f"at ${trade.stop_loss:.2f}"
                )
                return trade

            if hit_stop:
                self._exit_trade(trade, bar, 'stop', trade.stop_loss)
                logger.debug(f"  Bar {i}: stopped out at ${trade.stop_loss:.2f}")
                return trade

            if hit_target:
                self._exit_trade(trade, bar, 'target', trade.take_profit)
                logger.debug(f"  Bar {i}: target hit at ${trade.take_profit:.2f}")
                return trade

        # End of day — exit at last bar's close
        last_bar = bars.iloc[last_bar_idx]
        self._exit_trade(trade, last_bar, 'eod', last_bar['close'])
        logger.debug(f"  EOD exit at ${last_bar['close']:.2f}")
        return trade

    def _simulate_with_partial(
        self,
        trade: SimulatedTrade,
        plan: TradePlan,
        bars: pd.DataFrame,
        entry_bar_idx: int,
        actual_entry: float,
    ) -> SimulatedTrade:
        """
        Simulate a trade with breakeven stop / partial profit exit strategy.

        When fraction > 0: sells a fraction of shares at +NR, moves stop to
        breakeven on the remainder, then trails for the full target.
        When fraction == 0: just moves stop to breakeven at +NR (no partial sell).

        Args:
            trade: The SimulatedTrade being filled
            plan: The trade plan
            bars: Full day's bars DataFrame
            entry_bar_idx: Index of entry bar
            actual_entry: Actual entry price used

        Returns:
            SimulatedTrade with partial profit fields populated
        """
        last_bar_idx = len(bars) - 1
        partial_taken = False
        current_stop = trade.stop_loss
        active_shares = trade.shares
        # Use actual risk from fill price to stop, not plan risk (which uses
        # breakout_level, not realistic fill price — slippage makes plan risk
        # too small, triggering breakeven too early)
        actual_risk = actual_entry - plan.stop_loss_price
        partial_target = actual_entry + actual_risk * self.partial_profit_r_multiple

        for i in range(entry_bar_idx + 1, len(bars)):
            bar = bars.iloc[i]

            # 1. Force close check
            if self.force_close_time_utc is not None:
                bar_ts = bar['timestamp']
                bar_hour = bar_ts.hour if hasattr(bar_ts, 'hour') else 0
                bar_minute = bar_ts.minute if hasattr(bar_ts, 'minute') else 0
                bar_time_utc = bar_hour + bar_minute / 60.0
                if bar_time_utc >= self.force_close_time_utc:
                    reason = 'partial+force_close' if partial_taken else 'force_close'
                    self._exit_trade(
                        trade, bar, reason, bar['open'], active_shares=active_shares
                    )
                    logger.debug(f"  Bar {i}: force close at ${bar['open']:.2f} ({reason})")
                    return trade

            bar_low = bar['low']
            bar_high = bar['high']

            hit_stop = bar_low <= current_stop
            hit_partial = not partial_taken and bar_high >= partial_target
            hit_target = bar_high >= trade.take_profit

            # 2. Stop hit check (uses current_stop which may be breakeven)
            if hit_stop and not hit_partial:
                if partial_taken:
                    # Stop is at breakeven after partial
                    if abs(current_stop - trade.entry_price) < 0.001:
                        reason = 'partial+breakeven'
                    else:
                        reason = 'partial+stop'
                else:
                    reason = 'stop'
                self._exit_trade(
                    trade, bar, reason, current_stop, active_shares=active_shares
                )
                logger.debug(f"  Bar {i}: {reason} at ${current_stop:.2f}")
                return trade

            # On same-bar stop+partial ambiguity: conservative = stop wins
            if hit_stop and hit_partial:
                self._exit_trade(
                    trade, bar, 'stop', current_stop, active_shares=active_shares
                )
                logger.debug(
                    f"  Bar {i}: ambiguous (stop & partial) → stopped out "
                    f"at ${current_stop:.2f}"
                )
                return trade

            # 3. Partial target hit
            if hit_partial:
                partial_shares = int(active_shares * self.partial_profit_fraction)
                remaining = active_shares - partial_shares

                trade.partial_exit_taken = True
                trade.partial_exit_time = bar['timestamp']
                trade.partial_exit_price = partial_target
                trade.partial_shares = partial_shares
                trade.partial_pnl = (partial_target - actual_entry) * partial_shares
                trade.remaining_shares = remaining
                trade.breakeven_stop_active = True

                active_shares = remaining
                current_stop = trade.entry_price  # Move to breakeven
                partial_taken = True

                logger.debug(
                    f"  Bar {i}: partial exit {partial_shares} shares at "
                    f"${partial_target:.2f}, P&L ${trade.partial_pnl:.2f}, "
                    f"remaining {active_shares} shares, stop → breakeven"
                )

                # Check if target also hit on same bar
                if hit_target:
                    self._exit_trade(
                        trade, bar, 'partial+target', trade.take_profit,
                        active_shares=active_shares,
                    )
                    logger.debug(
                        f"  Bar {i}: target also hit at ${trade.take_profit:.2f}"
                    )
                    return trade

                continue

            # 4. Final target hit
            if hit_target:
                reason = 'partial+target' if partial_taken else 'target'
                self._exit_trade(
                    trade, bar, reason, trade.take_profit,
                    active_shares=active_shares,
                )
                logger.debug(f"  Bar {i}: {reason} at ${trade.take_profit:.2f}")
                return trade

        # 5. End of day — exit remaining at last bar's close
        last_bar = bars.iloc[last_bar_idx]
        reason = 'partial+eod' if partial_taken else 'eod'
        self._exit_trade(
            trade, last_bar, reason, last_bar['close'],
            active_shares=active_shares,
        )
        logger.debug(f"  {reason} exit at ${last_bar['close']:.2f}")
        return trade

    def _exit_trade(
        self,
        trade: SimulatedTrade,
        bar: pd.Series,
        reason: str,
        price: float,
        active_shares: Optional[int] = None,
    ) -> None:
        """
        Fill in exit details on the trade.

        Args:
            trade: SimulatedTrade to update
            bar: The bar where exit occurs
            reason: Exit reason string
            price: Exit price
            active_shares: Number of shares being exited (None = trade.shares,
                used when partial profit has reduced the position)
        """
        trade.exit_time = bar['timestamp']
        trade.exit_price = price
        trade.exit_reason = reason

        if active_shares is not None and self.partial_profit_enabled:
            # Combined P&L: partial profit + remaining shares exit
            final_pnl = (price - trade.entry_price) * active_shares
            trade.pnl = trade.partial_pnl + final_pnl
            total_position = trade.entry_price * trade.shares
            trade.pnl_pct = (
                (trade.pnl / total_position * 100) if total_position > 0 else 0.0
            )
        else:
            trade.pnl = (price - trade.entry_price) * trade.shares
            trade.pnl_pct = (
                (price - trade.entry_price) / trade.entry_price * 100
                if trade.entry_price > 0
                else 0.0
            )

        trade.bars_held = (
            bar.name - trade.plan.pattern.flag_end_idx
            if hasattr(bar, 'name')
            else 0
        )


# ---------------------------------------------------------------------------
# Backtest Runner
# ---------------------------------------------------------------------------


class BacktestRunner:
    """
    Runs a backtest over a day's 1-minute bars.

    Two modes:
    - Fantasy (realistic=False, default): Uses detect() — enters at breakout_level
      after breakout candle closes. Original behavior, backward compatible.
    - Realistic (realistic=True): Uses detect_setup() — detects pole+flag before
      breakout, places a pending buy-stop at flag_high, fills at
      max(bar_open, breakout_level) when breakout happens.
    """

    MIN_BARS_FOR_DETECTION = 7  # 3 pole + 2 pullback + 1 breakout + 1 dropped
    MIN_BARS_FOR_SETUP = 6      # 3 pole + 2 pullback + 1 dropped

    DEFAULT_MIN_PRICE = 2.0
    DEFAULT_SKIP_MIDDAY = True
    MIDDAY_START_UTC = 15  # 11:30 ET = 15:30 UTC (hour boundary)
    MIDDAY_END_UTC = 18    # 14:00 ET = 18:00 UTC

    def __init__(
        self,
        detector: Optional[BullFlagDetector] = None,
        planner: Optional[TradePlanner] = None,
        simulator: Optional[TradeSimulator] = None,
        min_price: Optional[float] = None,
        skip_midday: Optional[bool] = None,
        early_exit_after_trade: bool = True,
        realistic: bool = True,
        last_entry_time_utc: float = 19.0,
        force_close_time_utc: Optional[float] = None,
        setup_expiry_bars: int = 10,
        partial_profit_enabled: bool = False,
        partial_profit_r_multiple: float = 1.0,
        partial_profit_fraction: float = 0.5,
    ):
        """
        Initialize BacktestRunner.

        Args:
            detector: BullFlagDetector instance (uses defaults if None)
            planner: TradePlanner instance (uses defaults if None)
            simulator: TradeSimulator instance (uses defaults if None)
            min_price: Minimum entry price filter (default 2.0)
            skip_midday: Skip 11:30-14:00 ET entries (default True)
            early_exit_after_trade: Stop scanning after first trade (default True)
            realistic: Use detect_setup() + pending buy-stop simulation
            last_entry_time_utc: No new entries after this UTC hour (default 19.0 = 15:00 ET)
            force_close_time_utc: Force close at this UTC hour (default None;
                in realistic mode defaults to 19.75 = 15:45 ET)
            setup_expiry_bars: Cancel pending buy-stop after N bars (default 10)
            partial_profit_enabled: Enable partial profit exit strategy
            partial_profit_r_multiple: Take partial at this R multiple (default 1.0)
            partial_profit_fraction: Fraction of shares for partial exit (default 0.5)
        """
        self.detector = detector or BullFlagDetector.from_config()
        self.planner = planner or TradePlanner.from_config()
        self.min_price = min_price if min_price is not None else self.DEFAULT_MIN_PRICE
        self.skip_midday = skip_midday if skip_midday is not None else self.DEFAULT_SKIP_MIDDAY
        self.early_exit_after_trade = early_exit_after_trade
        self.realistic = realistic
        self.last_entry_time_utc = last_entry_time_utc
        self.setup_expiry_bars = setup_expiry_bars

        # In realistic mode, default force_close to 15:45 ET = 19.75 UTC
        if force_close_time_utc is not None:
            self.force_close_time_utc = force_close_time_utc
        elif realistic:
            self.force_close_time_utc = 19.75
        else:
            self.force_close_time_utc = None

        # Wire force_close and partial profit into the simulator
        if simulator is not None:
            self.simulator = simulator
        else:
            self.simulator = TradeSimulator(
                force_close_time_utc=self.force_close_time_utc,
                partial_profit_enabled=partial_profit_enabled,
                partial_profit_r_multiple=partial_profit_r_multiple,
                partial_profit_fraction=partial_profit_fraction,
            )

    def _get_bar_time_utc(self, bar_ts) -> float:
        """Extract UTC fractional hour from a bar timestamp."""
        bar_hour = bar_ts.hour if hasattr(bar_ts, 'hour') else 0
        bar_minute = bar_ts.minute if hasattr(bar_ts, 'minute') else 0
        return bar_hour + bar_minute / 60.0

    def _is_midday(self, bar_time_utc: float) -> bool:
        """Check if bar time falls in midday dead zone (11:30-14:00 ET)."""
        return self.MIDDAY_START_UTC + 0.5 <= bar_time_utc < self.MIDDAY_END_UTC

    def run(self, symbol: str, bars: pd.DataFrame, trade_date: str) -> BacktestResult:
        """
        Run backtest for a symbol over a day's bars.

        Delegates to _run_fantasy() or _run_realistic() based on self.realistic.

        Args:
            symbol: Stock ticker symbol
            bars: DataFrame with 1-min bars (timestamp, OHLCV)
            trade_date: Date string for reporting (e.g., '2026-03-13')

        Returns:
            BacktestResult with trades, patterns, and P&L
        """
        if self.realistic:
            return self._run_realistic(symbol, bars, trade_date)
        return self._run_fantasy(symbol, bars, trade_date)

    def _run_fantasy(self, symbol: str, bars: pd.DataFrame, trade_date: str) -> BacktestResult:
        """
        Original backtest mode: detect() fires after breakout candle, enters at breakout_level.

        Kept for backward compatibility and as a baseline comparison.
        """
        result = BacktestResult(
            symbol=symbol,
            trade_date=trade_date,
            total_bars=len(bars),
            patterns_detected=0,
        )

        if len(bars) < self.MIN_BARS_FOR_DETECTION:
            logger.warning(
                f"{symbol}: Only {len(bars)} bars, need at least "
                f"{self.MIN_BARS_FOR_DETECTION} for detection"
            )
            return result

        trade_taken = False
        last_end = len(bars) - 1

        logger.info(f"{symbol}: Scanning {len(bars)} bars for patterns (fantasy mode)...")

        for i in range(self.MIN_BARS_FOR_DETECTION - 1, last_end):
            pattern = self.detector.detect(symbol, bars, end_idx=i)

            if pattern is None:
                continue

            detection = PatternDetection(
                bar_index=i,
                timestamp=bars.iloc[i]['timestamp'],
                pattern=pattern,
            )
            result.pattern_details.append(detection)
            result.patterns_detected += 1

            logger.info(
                f"  Pattern #{result.patterns_detected} at bar {i} "
                f"({bars.iloc[i]['timestamp']}): "
                f"pole {pattern.pole_gain_pct:.1f}% gain, "
                f"retracement {pattern.retracement_pct:.1f}%"
            )

            if trade_taken:
                logger.debug(f"  Skipping — already in a trade")
                continue

            plan = self.planner.create_plan(pattern)
            if plan is None:
                logger.debug(f"  Plan rejected at bar {i}")
                continue

            if self.min_price > 0 and plan.entry_price < self.min_price:
                logger.debug(
                    f"  Skipping — entry ${plan.entry_price:.2f} below "
                    f"min price ${self.min_price:.2f}"
                )
                continue

            if self.skip_midday:
                bar_time_utc = self._get_bar_time_utc(bars.iloc[i]['timestamp'])
                if self._is_midday(bar_time_utc):
                    logger.debug(
                        f"  Skipping — midday entry at {bars.iloc[i]['timestamp']} "
                        f"(11:30-14:00 ET filter)"
                    )
                    continue

            logger.info(
                f"  TRADE ENTRY at bar {i}: "
                f"${plan.entry_price:.2f} entry, "
                f"${plan.stop_loss_price:.2f} stop, "
                f"${plan.take_profit_price:.2f} target, "
                f"{plan.shares} shares"
            )

            trade = self.simulator.simulate(plan, bars, i)
            result.trades_simulated.append(trade)
            trade_taken = True

            logger.info(
                f"  TRADE EXIT ({trade.exit_reason}): "
                f"${trade.exit_price:.2f}, "
                f"P&L ${trade.pnl:.2f} ({trade.pnl_pct:+.1f}%)"
            )

            if self.early_exit_after_trade:
                logger.debug("  Early exit — skipping remaining bars after trade")
                break

        logger.info(
            f"{symbol}: Scan complete — "
            f"{result.patterns_detected} patterns, "
            f"{len(result.trades_simulated)} trades, "
            f"P&L ${result.summary_pnl:.2f}"
        )

        return result

    def _run_realistic(self, symbol: str, bars: pd.DataFrame, trade_date: str) -> BacktestResult:
        """
        Realistic backtest: detect_setup() fires before breakout, places pending
        buy-stop at flag_high, fills at max(bar_open, breakout_level).

        Loop:
        1. Check pending buy-stop against current bar
        2. If no trade and no pending order, scan for new setup
        3. Apply filters (min_price, midday, last_entry_time)
        """
        result = BacktestResult(
            symbol=symbol,
            trade_date=trade_date,
            total_bars=len(bars),
            patterns_detected=0,
        )

        if len(bars) < self.MIN_BARS_FOR_SETUP:
            logger.warning(
                f"{symbol}: Only {len(bars)} bars, need at least "
                f"{self.MIN_BARS_FOR_SETUP} for setup detection"
            )
            return result

        trade_taken = False
        pending_order: Optional[PendingBuyStop] = None
        last_end = len(bars) - 1

        logger.info(f"{symbol}: Scanning {len(bars)} bars for setups (realistic mode)...")

        for i in range(self.MIN_BARS_FOR_SETUP - 1, last_end):
            bar = bars.iloc[i]
            bar_time_utc = self._get_bar_time_utc(bar['timestamp'])

            # --- Step 1: Check pending buy-stop ---
            if pending_order is not None and not trade_taken:
                bar_high = bar['high']
                bar_low = bar['low']
                bar_open = bar['open']

                # Check if setup invalidated (price dropped below flag_low)
                if bar_low < pending_order.setup.flag_low:
                    logger.debug(
                        f"  Bar {i}: buy-stop INVALIDATED — "
                        f"low ${bar_low:.2f} < flag_low ${pending_order.setup.flag_low:.2f}"
                    )
                    pending_order = None

                # Check expiry
                elif i - pending_order.placed_at_bar_idx > self.setup_expiry_bars:
                    logger.debug(
                        f"  Bar {i}: buy-stop EXPIRED after {self.setup_expiry_bars} bars"
                    )
                    pending_order = None

                # Check if triggered
                elif bar_high >= pending_order.breakout_level:
                    # Fill at max(bar_open, breakout_level) — realistic fill price
                    fill_price = max(bar_open, pending_order.breakout_level)
                    plan = pending_order.plan
                    entry_gap = fill_price - pending_order.breakout_level

                    # Adjust stop loss for entry gap: maintain planned risk_per_share
                    # so dollar risk stays at the budgeted amount.
                    # Without this, a $0.78 gap on a $0.34 planned risk turns
                    # a $2K risk trade into a $6.6K risk trade.
                    if entry_gap > 0:
                        adjusted_stop = fill_price - plan.risk_per_share
                        adjusted_target = fill_price + plan.risk_per_share * plan.risk_reward_ratio
                        logger.info(
                            f"  Entry gap +${entry_gap:.2f}: "
                            f"stop ${plan.stop_loss_price:.2f} → ${adjusted_stop:.2f}, "
                            f"target ${plan.take_profit_price:.2f} → ${adjusted_target:.2f} "
                            f"(maintain ${plan.risk_per_share:.2f}/sh risk)"
                        )
                        plan = TradePlan(
                            symbol=plan.symbol,
                            entry_price=plan.entry_price,
                            stop_loss_price=adjusted_stop,
                            take_profit_price=adjusted_target,
                            risk_per_share=plan.risk_per_share,
                            reward_per_share=plan.reward_per_share,
                            risk_reward_ratio=plan.risk_reward_ratio,
                            shares=plan.shares,
                            total_risk=plan.total_risk,
                            pattern=plan.pattern,
                        )

                    logger.info(
                        f"  BUY-STOP TRIGGERED at bar {i}: "
                        f"planned ${pending_order.breakout_level:.2f}, "
                        f"fill ${fill_price:.2f} (gap +${entry_gap:.2f}), "
                        f"{plan.shares} shares"
                    )

                    trade = self.simulator.simulate(
                        plan, bars, i, entry_price_override=fill_price
                    )
                    result.trades_simulated.append(trade)
                    trade_taken = True
                    pending_order = None

                    logger.info(
                        f"  TRADE EXIT ({trade.exit_reason}): "
                        f"${trade.exit_price:.2f}, "
                        f"P&L ${trade.pnl:.2f} ({trade.pnl_pct:+.1f}%)"
                    )

                    if self.early_exit_after_trade:
                        break

            # --- Step 2: Scan for new setup ---
            if not trade_taken and pending_order is None:
                # Last entry time check
                if bar_time_utc >= self.last_entry_time_utc:
                    continue

                setup = self.detector.detect_setup(symbol, bars, end_idx=i)

                if setup is None:
                    continue

                result.patterns_detected += 1
                detection = PatternDetection(
                    bar_index=i,
                    timestamp=bar['timestamp'],
                    pattern=setup,
                )
                result.pattern_details.append(detection)

                logger.info(
                    f"  Setup #{result.patterns_detected} at bar {i} "
                    f"({bar['timestamp']}): "
                    f"pole {setup.pole_gain_pct:.1f}% gain, "
                    f"retracement {setup.retracement_pct:.1f}%, "
                    f"buy-stop @ ${setup.breakout_level:.2f}"
                )

                plan = self.planner.create_plan(setup)
                if plan is None:
                    logger.debug(f"  Plan rejected at bar {i}")
                    continue

                # Price filter
                if self.min_price > 0 and plan.entry_price < self.min_price:
                    logger.debug(
                        f"  Skipping — entry ${plan.entry_price:.2f} below "
                        f"min price ${self.min_price:.2f}"
                    )
                    continue

                # Midday filter
                if self.skip_midday and self._is_midday(bar_time_utc):
                    logger.debug(
                        f"  Skipping — midday setup at {bar['timestamp']} "
                        f"(11:30-14:00 ET filter)"
                    )
                    continue

                pending_order = PendingBuyStop(
                    setup=setup,
                    plan=plan,
                    placed_at_bar_idx=i,
                    breakout_level=setup.breakout_level,
                )
                logger.info(
                    f"  PENDING BUY-STOP placed at bar {i}: "
                    f"${setup.breakout_level:.2f}, "
                    f"expires in {self.setup_expiry_bars} bars"
                )

        logger.info(
            f"{symbol}: Scan complete — "
            f"{result.patterns_detected} setups, "
            f"{len(result.trades_simulated)} trades, "
            f"P&L ${result.summary_pnl:.2f}"
        )

        return result


# ---------------------------------------------------------------------------
# Report Printer
# ---------------------------------------------------------------------------


def print_report(result: BacktestResult) -> None:
    """Print a formatted backtest report to console."""
    print("\n" + "=" * 70)
    print(f"  BACKTEST REPORT: {result.symbol} on {result.trade_date}")
    print("=" * 70)
    print(f"  Total bars scanned:    {result.total_bars}")
    print(f"  Patterns detected:     {result.patterns_detected}")
    print(f"  Trades simulated:      {len(result.trades_simulated)}")
    print(f"  Summary P&L:           ${result.summary_pnl:.2f}")
    print("-" * 70)

    if result.pattern_details:
        print("\n  Pattern Detections:")
        for det in result.pattern_details:
            p = det.pattern
            print(
                f"    Bar {det.bar_index:>4d} | {det.timestamp} | "
                f"Pole +{p.pole_gain_pct:.1f}% | "
                f"Retrace {p.retracement_pct:.1f}% | "
                f"Breakout ${p.breakout_level:.2f}"
            )

    if result.trades_simulated:
        print("\n  Trade Details:")
        for t in result.trades_simulated:
            pnl_sign = "+" if t.pnl >= 0 else ""
            print(f"    Symbol:      {t.symbol}")
            print(f"    Entry:       ${t.entry_price:.2f} at {t.entry_time}")
            print(f"    Stop Loss:   ${t.stop_loss:.2f}")
            print(f"    Target:      ${t.take_profit:.2f}")
            print(f"    Shares:      {t.shares}")
            print(f"    Exit:        ${t.exit_price:.2f} at {t.exit_time}")
            print(f"    Exit Reason: {t.exit_reason}")
            print(f"    P&L:         {pnl_sign}${t.pnl:.2f} ({t.pnl_pct:+.1f}%)")
            print()
    else:
        print("\n  No trades taken.\n")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for backtesting."""
    parser = argparse.ArgumentParser(
        description="Backtest bull flag momentum strategy on historical data"
    )
    parser.add_argument("symbol", type=str, help="Stock symbol (e.g., PLYX)")
    parser.add_argument("date", type=str, help="Trade date YYYY-MM-DD (e.g., 2026-03-13)")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose/debug logging"
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load environment
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")

    if not api_key or not api_secret:
        logger.error("Missing ALPACA_API_KEY or ALPACA_API_SECRET in environment")
        sys.exit(1)

    # Parse date and build UTC time range for market hours (9:30 - 16:00 ET)
    trade_date = datetime.strptime(args.date, "%Y-%m-%d")
    market_open = trade_date.replace(hour=13, minute=30, second=0, tzinfo=timezone.utc)
    market_close = trade_date.replace(hour=20, minute=0, second=0, tzinfo=timezone.utc)

    symbol = args.symbol.upper()

    logger.info(f"Backtesting {symbol} on {args.date}")
    logger.info(f"Market hours (UTC): {market_open} to {market_close}")

    # Fetch historical bars
    client = AlpacaClient(api_key=api_key, api_secret=api_secret)
    bars = client.get_historical_1min_bars(symbol, market_open, market_close)

    if bars.empty:
        logger.error(f"No bars returned for {symbol} on {args.date}")
        sys.exit(1)

    logger.info(f"Fetched {len(bars)} bars for {symbol}")

    # Run backtest
    runner = BacktestRunner()
    result = runner.run(symbol, bars, args.date)

    # Print report
    print_report(result)


if __name__ == "__main__":
    main()
