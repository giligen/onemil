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
from typing import Dict, List, Optional

import pandas as pd
import pytz
from dotenv import load_dotenv

from data_sources.alpaca_client import AlpacaClient
from trading.pattern_detector import BullFlagDetector, BullFlagPattern, BullFlagSetup
from trading.trade_planner import TradePlanner, TradePlan
from trading.exhaustion_signals import (
    check_exhaustion,
    sig_volume_divergence,
    sig_climax_candle,
    sig_shrinking_bodies,
    sig_shooting_star,
)

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

    _ET = pytz.timezone('US/Eastern')

    @classmethod
    def _get_bar_time_et(cls, bar_ts) -> tuple:
        """Convert bar timestamp to ET (hour, minute), handling DST."""
        if hasattr(bar_ts, 'tzinfo') and bar_ts.tzinfo is not None:
            et_time = bar_ts.astimezone(cls._ET)
        elif hasattr(bar_ts, 'hour'):
            from datetime import timezone as tz
            et_time = bar_ts.replace(tzinfo=tz.utc).astimezone(cls._ET)
        else:
            return (0, 0)
        return (et_time.hour, et_time.minute)

    def __init__(
        self,
        force_close_time_et: Optional[tuple] = None,
        partial_profit_enabled: bool = False,
        partial_profit_r_multiple: float = 1.0,
        partial_profit_fraction: float = 0.5,
        exit_slippage_pct: float = 0.0,
        marketable_limit_offset: float = 0.0,
        marketable_limit_offset_pct: float = 0.0,
        trailing_stop_r: float = 0.0,
        trailing_activate_at_r: float = 0.0,
        breakeven_at_r: float = 0.0,
        breakeven_profit_r: float = 0.0,
        exhaustion_exit_enabled: bool = False,
        exhaustion_partial_fraction: float = 0.5,
        exhaustion_tighter_trail_r: float = 0.5,
        exhaustion_min_profit_r: float = 3.0,
        exhaustion_signals: Optional[Dict[str, bool]] = None,
        no_pop_exit_bars: int = 0,
        no_pop_exit_min_pct: float = 0.005,
        trail_exit_slippage_pct: float = None,
        trail_tighten_at_r: float = 0.0,
        trail_tightened_r: float = 0.5,
    ):
        """
        Initialize TradeSimulator.

        Args:
            force_close_time_et: ET time tuple (hour, minute) to force close.
            partial_profit_enabled: Enable partial profit exit at +NR, then
                move stop to breakeven on remaining shares.
            partial_profit_r_multiple: Take partial profit at this R multiple.
            partial_profit_fraction: Fraction of shares to sell at partial target.
            exit_slippage_pct: Slippage as fraction of stop price, subtracted
                from stop-loss fills. Not applied to exhaustion exits (selling
                into strength = ~0 slippage).
            marketable_limit_offset: Cap stop exit slippage at this dollar offset.
            marketable_limit_offset_pct: Cap stop exit slippage at this pct.
            trailing_stop_r: Trail N×R below highest high (0 = disabled).
            trailing_activate_at_r: Activate trail after +NR from entry.
            exhaustion_exit_enabled: Enable exhaustion signal partial exits.
                When a signal fires while profitable, sell partial_fraction at
                bar close (0 slippage) and tighten trail on remainder.
            exhaustion_partial_fraction: Fraction to sell on exhaustion signal.
            exhaustion_tighter_trail_r: Trail distance for remainder after
                exhaustion partial (in R units, e.g. 0.5 = tighter than 1.0).
            exhaustion_min_profit_r: Only check signals when trade is this
                many R in profit (default 1.0).
            exhaustion_signals: Dict of signal_name -> enabled. Valid keys:
                'volume_divergence', 'climax_candle', 'shrinking_bodies',
                'shooting_star'. None = all enabled when exhaustion_exit_enabled.
        """
        self.force_close_time_et = force_close_time_et
        self.partial_profit_enabled = partial_profit_enabled
        self.partial_profit_r_multiple = partial_profit_r_multiple
        self.partial_profit_fraction = partial_profit_fraction
        self.exit_slippage_pct = exit_slippage_pct
        self.marketable_limit_offset = marketable_limit_offset
        self.marketable_limit_offset_pct = marketable_limit_offset_pct
        self.trailing_stop_r = trailing_stop_r
        self.trailing_activate_at_r = trailing_activate_at_r
        # Exhaustion exit
        self.exhaustion_exit_enabled = exhaustion_exit_enabled
        self.exhaustion_partial_fraction = exhaustion_partial_fraction
        self.exhaustion_tighter_trail_r = exhaustion_tighter_trail_r
        self.exhaustion_min_profit_r = exhaustion_min_profit_r
        self.exhaustion_signals = exhaustion_signals or {
            'volume_divergence': False,
            'climax_candle': True,
            'shrinking_bodies': False,
            'shooting_star': True,
        }
        # No-pop exit: force close if trade doesn't gain min_pct within N bars
        self.no_pop_exit_bars = no_pop_exit_bars  # 0 = disabled
        self.no_pop_exit_min_pct = no_pop_exit_min_pct  # 0.005 = 0.5%
        # Trail exit slippage: separate from stop loss slippage (None = use same)
        self.trail_exit_slippage_pct = trail_exit_slippage_pct
        # Breakeven stop: move stop to entry + breakeven_profit_r * risk after +breakeven_at_r
        # Acts as stage 1 protection before full trail activates at activate_at_r
        self.breakeven_at_r = breakeven_at_r  # 0 = disabled
        self.breakeven_profit_r = breakeven_profit_r  # 0 = clean breakeven, 0.4 = lock 0.4R
        # Trail tightening: once profit reaches tighten_at_r, reduce trail from trail_r to tightened_r
        # Locks in more profit on big runners without capping them (replaces fixed TP)
        self.trail_tighten_at_r = trail_tighten_at_r  # 0 = disabled
        self.trail_tightened_r = trail_tightened_r  # 0.5 = tighter trail after threshold

    # ------------------------------------------------------------------
    # Exhaustion signal detectors (delegated to shared module)
    # ------------------------------------------------------------------

    def _check_exhaustion(self, bars: pd.DataFrame, idx: int) -> bool:
        """Check if any enabled exhaustion signal fires at bar idx."""
        return check_exhaustion(bars, idx, self.exhaustion_signals)

    def _compute_stop_fill(self, stop_price: float, slippage_override: float = None) -> float:
        """
        Compute stop-loss fill price with slippage, optionally capped by
        marketable limit offset (models self-managed stops).

        Without cap: stop_fill = stop_price * (1 - exit_slippage_pct)
        With cap:    slippage is min(raw_slippage, max(fixed_offset, pct_offset))

        Args:
            stop_price: The stop-loss trigger price
            slippage_override: If set, use this slippage pct instead of default

        Returns:
            Simulated fill price after slippage
        """
        slip_pct = slippage_override if slippage_override is not None else self.exit_slippage_pct
        raw_slip = stop_price * slip_pct

        if self.marketable_limit_offset > 0 or self.marketable_limit_offset_pct > 0:
            fixed_cap = self.marketable_limit_offset
            pct_cap = stop_price * self.marketable_limit_offset_pct
            cap = max(fixed_cap, pct_cap)
            raw_slip = min(raw_slip, cap)

        return stop_price - raw_slip

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

        # Trailing stop state
        use_trail = self.trailing_stop_r > 0
        trailing_active = False
        highest_since_entry = actual_entry
        current_stop = trade.stop_loss
        risk = actual_entry - plan.stop_loss_price

        # Exhaustion exit state
        exhaust_partial_taken = False
        active_shares = trade.shares
        effective_trail_r = self.trailing_stop_r

        for i in range(entry_bar_idx + 1, len(bars)):
            bar = bars.iloc[i]

            # Force close check
            if self.force_close_time_et is not None:
                bar_et = self._get_bar_time_et(bar['timestamp'])
                if bar_et >= self.force_close_time_et:
                    reason = 'exhaust+force_close' if exhaust_partial_taken else 'force_close'
                    self._exit_trade(trade, bar, reason, bar['open'],
                                     active_shares=active_shares if exhaust_partial_taken else None)
                    logger.debug(f"  Bar {i}: {reason} at ${bar['open']:.2f}")
                    return trade

            bar_low = bar['low']
            bar_high = bar['high']

            # Update highest high for trailing stop
            if use_trail and bar_high > highest_since_entry:
                highest_since_entry = bar_high

            # No-pop exit: if trade hasn't gained min_pct within N bars, force close
            if self.no_pop_exit_bars > 0 and (i - entry_bar_idx) == self.no_pop_exit_bars:
                max_move_pct = (highest_since_entry - actual_entry) / actual_entry
                if max_move_pct < self.no_pop_exit_min_pct:
                    reason = 'no_pop'
                    if exhaust_partial_taken:
                        reason = 'exhaust+no_pop'
                    self._exit_trade(trade, bar, reason, bar['close'],
                                     active_shares=active_shares if exhaust_partial_taken else None)
                    logger.debug(
                        f"  Bar {i}: NO-POP exit after {self.no_pop_exit_bars} bars — "
                        f"max +{max_move_pct:.2%} < {self.no_pop_exit_min_pct:.1%}"
                    )
                    return trade

            hit_stop = bar_low <= current_stop

            if use_trail:
                # With trailing stop: no fixed TP, trail replaces it
                if hit_stop:
                    # Use trail-specific slippage if set (reversals may have worse fills)
                    slip = self.trail_exit_slippage_pct if (trailing_active and self.trail_exit_slippage_pct is not None) else None
                    stop_fill = self._compute_stop_fill(current_stop, slippage_override=slip)
                    reason = 'trail_stop' if trailing_active else 'stop'
                    if exhaust_partial_taken:
                        reason = 'exhaust+' + reason
                    self._exit_trade(trade, bar, reason, stop_fill,
                                     active_shares=active_shares if exhaust_partial_taken else None)
                    logger.debug(f"  Bar {i}: {reason} at ${stop_fill:.2f}")
                    return trade

                # Stage 1: Move stop to breakeven (+ optional profit) after +breakeven_at_r
                if self.breakeven_at_r > 0 and risk > 0 and not trailing_active:
                    r_gain = (highest_since_entry - actual_entry) / risk
                    breakeven_target = actual_entry + self.breakeven_profit_r * risk
                    if r_gain >= self.breakeven_at_r and current_stop < breakeven_target:
                        current_stop = breakeven_target
                        logger.debug(
                            f"  Bar {i}: BREAKEVEN stop at +{r_gain:.1f}R → "
                            f"stop ${breakeven_target:.2f} (+{self.breakeven_profit_r}R)"
                        )

                # Stage 2: Activate trailing after +NR
                if risk > 0:
                    r_gain = (highest_since_entry - actual_entry) / risk
                    if r_gain >= self.trailing_activate_at_r:
                        trailing_active = True

                # Stage 2.5: Tighten trail after passing threshold (e.g., 2.5R)
                # Locks in more profit on runners without capping them
                if (trailing_active and not exhaust_partial_taken
                        and self.trail_tighten_at_r > 0 and risk > 0):
                    if r_gain >= self.trail_tighten_at_r and effective_trail_r > self.trail_tightened_r:
                        effective_trail_r = self.trail_tightened_r
                        logger.debug(
                            f"  Bar {i}: trail TIGHTENED at +{r_gain:.1f}R → "
                            f"{effective_trail_r}R trail"
                        )

                # Ratchet stop up (uses tighter trail after exhaustion partial)
                if trailing_active and risk > 0:
                    new_stop = highest_since_entry - risk * effective_trail_r
                    if new_stop > current_stop:
                        current_stop = new_stop

                # Exhaustion exit: check signals when profitable, sell partial
                if (self.exhaustion_exit_enabled and not exhaust_partial_taken
                        and risk > 0):
                    current_r = (bar['close'] - actual_entry) / risk
                    if current_r >= self.exhaustion_min_profit_r:
                        if self._check_exhaustion(bars, i):
                            # Partial exit at bar close (0 slippage — selling
                            # into strength while buyers are aggressive)
                            partial_shares = int(
                                active_shares * self.exhaustion_partial_fraction
                            )
                            remaining = active_shares - partial_shares

                            trade.partial_exit_taken = True
                            trade.partial_exit_time = bar['timestamp']
                            trade.partial_exit_price = bar['close']
                            trade.partial_shares = partial_shares
                            trade.partial_pnl = (
                                (bar['close'] - actual_entry) * partial_shares
                            )
                            trade.remaining_shares = remaining

                            active_shares = remaining
                            exhaust_partial_taken = True
                            effective_trail_r = self.exhaustion_tighter_trail_r

                            # Immediately ratchet with tighter trail
                            new_stop = (highest_since_entry
                                        - risk * effective_trail_r)
                            if new_stop > current_stop:
                                current_stop = new_stop

                            logger.debug(
                                f"  Bar {i}: EXHAUSTION partial — "
                                f"sold {partial_shares}sh @ ${bar['close']:.2f}, "
                                f"P&L ${trade.partial_pnl:.0f}, "
                                f"trail tightened to {effective_trail_r}R"
                            )
            else:
                # Original fixed TP logic (unchanged)
                hit_target = bar_high >= trade.take_profit

                if hit_stop and hit_target:
                    stop_fill = self._compute_stop_fill(trade.stop_loss)
                    self._exit_trade(trade, bar, 'stop', stop_fill)
                    logger.debug(
                        f"  Bar {i}: ambiguous (stop & target) → stopped out "
                        f"at ${stop_fill:.2f}"
                    )
                    return trade

                if hit_stop:
                    stop_fill = self._compute_stop_fill(trade.stop_loss)
                    self._exit_trade(trade, bar, 'stop', stop_fill)
                    logger.debug(f"  Bar {i}: stopped out at ${stop_fill:.2f}")
                    return trade

                if hit_target:
                    self._exit_trade(trade, bar, 'target', trade.take_profit)
                    logger.debug(f"  Bar {i}: target hit at ${trade.take_profit:.2f}")
                    return trade

        # End of day — exit at last bar's close
        last_bar = bars.iloc[last_bar_idx]
        reason = 'exhaust+eod' if exhaust_partial_taken else 'eod'
        self._exit_trade(trade, last_bar, reason, last_bar['close'],
                         active_shares=active_shares if exhaust_partial_taken else None)
        logger.debug(f"  {reason} exit at ${last_bar['close']:.2f}")
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
            if self.force_close_time_et is not None:
                bar_et = self._get_bar_time_et(bar['timestamp'])
                if bar_et >= self.force_close_time_et:
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
                stop_fill = self._compute_stop_fill(current_stop)
                self._exit_trade(
                    trade, bar, reason, stop_fill, active_shares=active_shares
                )
                logger.debug(f"  Bar {i}: {reason} at ${stop_fill:.2f}")
                return trade

            # On same-bar stop+partial ambiguity: conservative = stop wins
            if hit_stop and hit_partial:
                stop_fill = self._compute_stop_fill(current_stop)
                self._exit_trade(
                    trade, bar, 'stop', stop_fill, active_shares=active_shares
                )
                logger.debug(
                    f"  Bar {i}: ambiguous (stop & partial) → stopped out "
                    f"at ${stop_fill:.2f}"
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

        if active_shares is not None and trade.partial_exit_taken:
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
    MIDDAY_START_ET = (11, 30)  # 11:30 ET
    MIDDAY_END_ET = (14, 0)    # 14:00 ET

    def __init__(
        self,
        detector: Optional[BullFlagDetector] = None,
        planner: Optional[TradePlanner] = None,
        simulator: Optional[TradeSimulator] = None,
        min_price: Optional[float] = None,
        skip_midday: Optional[bool] = None,
        early_exit_after_trade: bool = True,
        realistic: bool = True,
        last_entry_time_et: tuple = (15, 0),
        force_close_time_et: Optional[tuple] = None,
        setup_expiry_bars: int = 10,
        partial_profit_enabled: bool = False,
        partial_profit_r_multiple: float = 1.0,
        partial_profit_fraction: float = 0.5,
        rvol_mode: str = 'cumulative',
        entry_slippage: Optional[float] = None,
        exit_slippage: Optional[float] = None,
        trailing_stop_r: float = 0.0,
        trailing_activate_at_r: float = 0.0,
        trailing_breakeven_at_r: float = 0.0,
        trailing_breakeven_profit_r: float = 0.0,
        trail_exit_slippage_pct: float = None,
        min_stop_distance: float = 0.0,
        vol_dead_zone_enabled: bool = False,
        vol_dead_zone_min: float = 2.0,
        vol_dead_zone_max: float = 5.0,
        no_pop_exit_bars: int = 0,
        no_pop_exit_min_pct: float = 0.005,
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
            last_entry_time_et: No new entries after this ET time (default (15, 0) = 15:00 ET)
            force_close_time_et: Force close at this ET time (default None;
                in realistic mode defaults to (15, 45) = 15:45 ET)
            setup_expiry_bars: Cancel pending buy-stop after N bars (default 10)
            partial_profit_enabled: Enable partial profit exit strategy
            partial_profit_r_multiple: Take partial at this R multiple (default 1.0)
            partial_profit_fraction: Fraction of shares for partial exit (default 0.5)
            rvol_mode: 'cumulative' (Ross's definition: total vol vs expected by time)
                or 'bucket' (scanner's impl: 15-min bucket vol vs profile avg)
            entry_slippage: $/share added to buy-stop fill price (default from config)
            exit_slippage: $/share subtracted from stop-loss fill (default from config)
            trailing_stop_r: Replace fixed TP with trailing stop N×R below high
                (0 = disabled, use fixed TP). E.g., 1.0 = trail 1R below high.
            trailing_activate_at_r: Activate trail after price reaches +NR
                from entry. E.g., 2.0 = trail starts after +2R.
        """
        self.detector = detector or BullFlagDetector.from_config()
        self.planner = planner or TradePlanner.from_config()
        self.min_price = min_price if min_price is not None else self.DEFAULT_MIN_PRICE
        self.skip_midday = skip_midday if skip_midday is not None else self.DEFAULT_SKIP_MIDDAY
        self.min_stop_distance = min_stop_distance
        self.vol_dead_zone_enabled = vol_dead_zone_enabled
        self.vol_dead_zone_min = vol_dead_zone_min
        self.vol_dead_zone_max = vol_dead_zone_max
        # UD risk scaling: set per-date by batch_backtest (0 = disabled)
        self._ud_risk_scale = 0.0  # 0 = don't scale, >0 = multiply shares by this

        # Cumulative rvol check at entry time (Ross's 5x filter)
        # Loaded from config.yaml scanner.relative_volume_min
        from config import Config
        cfg = Config._load_yaml_only()
        self.relative_volume_min = float(
            cfg.get("scanner", {}).get("relative_volume_min", 5.0)
        )
        # Total trading minutes per day (9:30-16:00 ET = 390 min)
        self.TRADING_MINUTES = 390
        self.rvol_mode = rvol_mode
        self._min_breakout_vol_override = 0  # 0 = disabled; set per-date by batch_backtest
        self._spy_macd_cutoff = None  # SpyMacdCutoff instance; set per-date by batch_backtest

        # JIT liquidity cap: size based on fill bar volume (not static ADV)
        # On breakout days, volume is 3-10x ADV — static ADV% is too conservative.
        # 15% of fill bar volume = realistic participation at the breakout moment.
        self.max_bar_participation_pct = float(
            cfg.get("trading", {}).get("max_bar_participation_pct", 0)
        )

        self.early_exit_after_trade = early_exit_after_trade
        self.realistic = realistic
        self.setup_expiry_bars = setup_expiry_bars

        # Slippage model: percentage-based (of stock price).
        # Entry: buy-stop fills at price * (1 + pct) — lifting the ask.
        # Exit: stop-market fills at price * (1 - pct) — selling into bid.
        # Percentage scales naturally with stock price ($0.005 on $5, $0.02 on $20).
        trading_cfg = cfg.get("trading", {})

        # Load last_entry_time from config (override default if set)
        if last_entry_time_et == (15, 0):  # default — check config
            let_str = trading_cfg.get("last_entry_time", "15:00")
            _h, _m = let_str.split(':')
            self.last_entry_time_et = (int(_h), int(_m))
        else:
            self.last_entry_time_et = last_entry_time_et
        self.entry_slippage_pct = entry_slippage if entry_slippage is not None else float(
            trading_cfg.get("entry_slippage_pct", 0.0)
        )
        resolved_exit_slippage_pct = exit_slippage if exit_slippage is not None else float(
            trading_cfg.get("exit_slippage_pct", 0.0)
        )
        self.exit_slippage_pct = resolved_exit_slippage_pct

        # Marketable limit offset: caps stop exit slippage when self-managed
        # stops are configured. Models the real-time WebSocket + limit sell
        # strategy instead of uncapped stop-market fills.
        sms_cfg = trading_cfg.get("self_managed_stops", {})
        self.marketable_limit_offset = float(sms_cfg.get("marketable_limit_offset", 0.0))
        self.marketable_limit_offset_pct = float(sms_cfg.get("marketable_limit_offset_pct", 0.0))
        sms_enabled = bool(sms_cfg.get("enabled", False))

        if self.entry_slippage_pct > 0 or resolved_exit_slippage_pct > 0:
            slip_msg = (
                f"Slippage model: entry +{self.entry_slippage_pct:.2%}, "
                f"exit -{resolved_exit_slippage_pct:.2%} (stop only)"
            )
            if sms_enabled and (self.marketable_limit_offset > 0 or self.marketable_limit_offset_pct > 0):
                slip_msg += (
                    f", capped by marketable limit offset "
                    f"${self.marketable_limit_offset}/"
                    f"{self.marketable_limit_offset_pct:.1%}"
                )
            logger.info(slip_msg)

        # In realistic mode, default force_close to 15:45 ET
        if force_close_time_et is not None:
            self.force_close_time_et = force_close_time_et
        elif realistic:
            self.force_close_time_et = (15, 45)
        else:
            self.force_close_time_et = None

        # Trailing stop: load from config if not explicitly passed
        trail_cfg = trading_cfg.get("trailing_stop", {})
        trail_enabled = bool(trail_cfg.get("enabled", False))
        resolved_trail_r = trailing_stop_r
        resolved_trail_activate = trailing_activate_at_r
        if resolved_trail_r == 0.0 and trail_enabled:
            resolved_trail_r = float(trail_cfg.get("trail_r", 1.0))
            resolved_trail_activate = float(trail_cfg.get("activate_at_r", 2.0))

        # Breakeven stop: from param or config
        resolved_breakeven = trailing_breakeven_at_r
        if resolved_breakeven == 0.0:
            resolved_breakeven = float(trail_cfg.get("breakeven_at_r", 0.0))
        resolved_breakeven_profit = trailing_breakeven_profit_r
        if resolved_breakeven_profit == 0.0:
            resolved_breakeven_profit = float(trail_cfg.get("breakeven_profit_r", 0.0))

        # Exhaustion exit config
        exhaust_cfg = trading_cfg.get("exhaustion_exit", {})
        self.exhaustion_exit_enabled = bool(exhaust_cfg.get("enabled", False))
        exhaust_signals = exhaust_cfg.get("signals", {})
        self.exhaustion_signals = {
            'volume_divergence': bool(exhaust_signals.get('volume_divergence', False)),
            'climax_candle': bool(exhaust_signals.get('climax_candle', True)),
            'shrinking_bodies': bool(exhaust_signals.get('shrinking_bodies', False)),
            'shooting_star': bool(exhaust_signals.get('shooting_star', True)),
        }

        # Wire force_close, partial profit, exit slippage, marketable
        # limit offset, trailing stop, and exhaustion exit into the simulator
        if simulator is not None:
            self.simulator = simulator
        else:
            ml_offset = self.marketable_limit_offset if sms_enabled else 0.0
            ml_offset_pct = self.marketable_limit_offset_pct if sms_enabled else 0.0
            self.simulator = TradeSimulator(
                force_close_time_et=self.force_close_time_et,
                partial_profit_enabled=partial_profit_enabled,
                partial_profit_r_multiple=partial_profit_r_multiple,
                partial_profit_fraction=partial_profit_fraction,
                exit_slippage_pct=resolved_exit_slippage_pct,
                marketable_limit_offset=ml_offset,
                marketable_limit_offset_pct=ml_offset_pct,
                trailing_stop_r=resolved_trail_r,
                trailing_activate_at_r=resolved_trail_activate,
                trail_tighten_at_r=float(trail_cfg.get("tighten_at_r", 0)),
                trail_tightened_r=float(trail_cfg.get("tightened_trail_r", 0.5)),
                breakeven_at_r=resolved_breakeven,
                breakeven_profit_r=resolved_breakeven_profit,
                exhaustion_exit_enabled=self.exhaustion_exit_enabled,
                exhaustion_partial_fraction=float(exhaust_cfg.get("partial_fraction", 0.5)),
                exhaustion_tighter_trail_r=float(exhaust_cfg.get("tighter_trail_r", 0.5)),
                exhaustion_min_profit_r=float(exhaust_cfg.get("min_profit_r", 3.0)),
                exhaustion_signals=self.exhaustion_signals,
                no_pop_exit_bars=no_pop_exit_bars,
                no_pop_exit_min_pct=no_pop_exit_min_pct,
                trail_exit_slippage_pct=trail_exit_slippage_pct,
            )
            if resolved_trail_r > 0:
                logger.info(
                    f"Trailing stop: {resolved_trail_r:.1f}R below high, "
                    f"activates at +{resolved_trail_activate:.1f}R (replaces fixed TP)"
                )
            if self.exhaustion_exit_enabled:
                active = [k for k, v in self.exhaustion_signals.items() if v]
                logger.info(
                    f"Exhaustion exit: {', '.join(active)}, "
                    f"partial {float(exhaust_cfg.get('partial_fraction', 0.5)):.0%}, "
                    f"tighter trail {float(exhaust_cfg.get('tighter_trail_r', 0.5)):.1f}R"
                )

        # MACD zone filter: risk scaling based on MACD histogram strength at entry
        macd_zones_cfg = trading_cfg.get("macd_zones", {})
        self.macd_zones_enabled = bool(macd_zones_cfg.get("enabled", False))
        self.macd_dead_zone_min = float(macd_zones_cfg.get("dead_zone_min_pct", -0.2))
        self.macd_dead_zone_max = float(macd_zones_cfg.get("dead_zone_max_pct", 0.1))
        self.macd_strong_neg_threshold = float(macd_zones_cfg.get("strong_neg_threshold_pct", -0.5))
        self.macd_strong_neg_multiplier = float(macd_zones_cfg.get("strong_neg_multiplier", 1.5))
        self.macd_strong_pos_threshold = float(macd_zones_cfg.get("strong_pos_threshold_pct", 0.5))
        self.macd_strong_pos_multiplier = float(macd_zones_cfg.get("strong_pos_multiplier", 1.5))
        self.macd_normal_multiplier = float(macd_zones_cfg.get("normal_multiplier", 1.0))
        # Sweet spot zone: MACD% in [sweet_min, sweet_max] gets its own multiplier
        self.macd_sweet_spot_min = float(macd_zones_cfg.get("sweet_spot_min_pct", 0.0))  # 0 = disabled
        self.macd_sweet_spot_max = float(macd_zones_cfg.get("sweet_spot_max_pct", 0.0))
        self.macd_sweet_spot_multiplier = float(macd_zones_cfg.get("sweet_spot_multiplier", 2.0))
        self._prev_day_bars: Optional[pd.DataFrame] = None

        if self.macd_zones_enabled:
            logger.info(
                f"MACD zones: dead [{self.macd_dead_zone_min}%, {self.macd_dead_zone_max}%], "
                f"strong neg <{self.macd_strong_neg_threshold}% → {self.macd_strong_neg_multiplier}x, "
                f"strong pos >{self.macd_strong_pos_threshold}% → {self.macd_strong_pos_multiplier}x, "
                f"normal → {self.macd_normal_multiplier}x"
            )

    _ET = pytz.timezone('US/Eastern')

    def _get_macd_zone_multiplier(
        self, symbol: str, bars: pd.DataFrame, entry_bar_idx: int, entry_price: float,
    ) -> float:
        """
        Compute MACD zone risk multiplier at entry time.

        Uses warmed-up MACD histogram (prev-day bars prepended) to determine
        which zone the trade falls in, and returns the appropriate risk multiplier.

        Args:
            symbol: Stock symbol
            bars: Full day's 1-min bars
            entry_bar_idx: Index of entry bar
            entry_price: Fill price at entry

        Returns:
            Risk multiplier: 0.0 = skip (dead zone), 1.0 = normal,
            1.25/1.5 = boosted (strong zones)
        """
        from trading.indicators import macd_histogram

        # Build closes up to entry bar
        closes = bars['close'].iloc[:entry_bar_idx + 1]

        # Prepend warm-up closes from previous day
        if self._prev_day_bars is not None and not self._prev_day_bars.empty:
            warmup = self._prev_day_bars['close'].tail(60).reset_index(drop=True)
            closes = pd.concat([warmup, closes], ignore_index=True)

        # Need enough bars for MACD
        min_bars = 35  # slow(26) + signal(9)
        if len(closes) < min_bars:
            logger.debug(f"{symbol}: MACD zone — insufficient bars ({len(closes)}/{min_bars}), using 1.0x")
            return 1.0

        hist = macd_histogram(closes)
        hist_val = float(hist.iloc[-1])
        macd_pct = (hist_val / entry_price) * 100

        # Determine zone
        if self.macd_dead_zone_min <= macd_pct <= self.macd_dead_zone_max:
            logger.info(
                f"  MACD ZONE SKIP: histogram {hist_val:.4f} = {macd_pct:.2f}% "
                f"(dead zone [{self.macd_dead_zone_min}%, {self.macd_dead_zone_max}%])"
            )
            return 0.0
        elif macd_pct < self.macd_strong_neg_threshold:
            mult = self.macd_strong_neg_multiplier
            logger.debug(
                f"  MACD zone: {macd_pct:.2f}% → strong neg → {mult}x risk"
            )
            return mult
        elif macd_pct > self.macd_strong_pos_threshold:
            # Check sweet spot sub-zone first (higher priority)
            if (self.macd_sweet_spot_min > 0 and
                    self.macd_sweet_spot_min <= macd_pct <= self.macd_sweet_spot_max):
                mult = self.macd_sweet_spot_multiplier
                logger.debug(
                    f"  MACD zone: {macd_pct:.2f}% → sweet spot → {mult}x risk"
                )
                return mult
            mult = self.macd_strong_pos_multiplier
            logger.debug(
                f"  MACD zone: {macd_pct:.2f}% → strong pos → {mult}x risk"
            )
            return mult
        else:
            mult = self.macd_normal_multiplier
            logger.debug(f"  MACD zone: {macd_pct:.2f}% → normal → {mult}x risk")
            return mult

    def set_spy_macd_cutoff(self, cutoff) -> None:
        """Set SPY MACD cutoff filter for the current trading day.

        Called by batch_backtest per date with a SpyMacdCutoff instance.
        Pass None to disable.
        """
        self._spy_macd_cutoff = cutoff

    def _get_bar_time_et(self, bar_ts) -> tuple:
        """Convert bar timestamp to ET (hour, minute), handling DST correctly."""
        if hasattr(bar_ts, 'tzinfo') and bar_ts.tzinfo is not None:
            et_time = bar_ts.astimezone(self._ET)
        elif hasattr(bar_ts, 'hour'):
            # Assume UTC if no timezone info
            et_time = bar_ts.replace(tzinfo=timezone.utc).astimezone(self._ET)
        else:
            return (0, 0)
        return (et_time.hour, et_time.minute)

    def _is_midday(self, bar_time_et: tuple) -> bool:
        """Check if bar time falls in midday dead zone (11:30-14:00 ET)."""
        return self.MIDDAY_START_ET <= bar_time_et < self.MIDDAY_END_ET

    def run(
        self, symbol: str, bars: pd.DataFrame, trade_date: str,
        avg_daily_volume: Optional[int] = None,
        volume_profile: Optional[Dict[str, int]] = None,
        prev_close: Optional[float] = None,
        prev_day_bars: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Run backtest for a symbol over a day's bars.

        Delegates to _run_fantasy() or _run_realistic() based on self.realistic.

        Args:
            symbol: Stock ticker symbol
            bars: DataFrame with 1-min bars (timestamp, OHLCV)
            trade_date: Date string for reporting (e.g., '2026-03-13')
            avg_daily_volume: Average daily volume for cumulative rvol check
                (from universe table; None = skip rvol filter)
            volume_profile: Dict mapping time_bucket ('09:30', etc.) to avg 15-min
                volume (from DB volume_profiles table; None = skip bucket rvol)
            prev_close: Previous day's close price. When provided, simulates
                real-time qualification: pattern scanning only starts after the
                bar where (bar_high - prev_close) / prev_close >= qualification_pct.
                This eliminates look-ahead bias where the backtest scans from bar 0
                knowing the stock WILL move 10%+ (from the daily bar pre-filter).
            prev_day_bars: Previous trading day's 1-min bars for MACD warm-up.
                Only used when require_macd_positive=True. Last ~60 bars are
                sufficient. None = no warm-up (cold-start, backward compatible).

        Returns:
            BacktestResult with trades, patterns, and P&L
        """
        # Store prev-day bars for MACD zone filter
        self._prev_day_bars = prev_day_bars

        # Set MACD warm-up on detector — always clear to prevent stale state
        # from previous symbol leaking into current one
        if hasattr(self.detector, 'set_macd_warmup'):
            if prev_day_bars is not None and not prev_day_bars.empty:
                warmup_closes = prev_day_bars['close'].tail(60).reset_index(drop=True)
                self.detector.set_macd_warmup(warmup_closes)
            else:
                self.detector.set_macd_warmup(None)

        self._current_avg_daily_volume = avg_daily_volume

        if self.realistic:
            return self._run_realistic(symbol, bars, trade_date,
                                       avg_daily_volume, volume_profile,
                                       prev_close=prev_close)
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

            plan = self.planner.create_plan(
                pattern, avg_daily_volume=self._current_avg_daily_volume)
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
                bar_time_et = self._get_bar_time_et(bars.iloc[i]['timestamp'])
                if self._is_midday(bar_time_et):
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

    def _get_bucket_rvol(
        self, bars: pd.DataFrame, bar_idx: int,
        volume_profile: Dict[str, int],
    ) -> float:
        """
        Compute bucket-based rvol matching the live scanner's approach.

        Sums 1-min bar volumes within the current 15-min ET bucket,
        divides by the volume profile's avg volume for that bucket.

        Args:
            bars: Full day's 1-min bars
            bar_idx: Current bar index
            volume_profile: {bucket_key: avg_volume} from DB

        Returns:
            Bucket-based relative volume ratio
        """
        bar_ts = bars.iloc[bar_idx]['timestamp']
        # Volume profiles are stored with ET hour keys (matching the live scanner).
        # Convert UTC timestamps to ET before computing bucket keys.
        ET = pytz.timezone('US/Eastern')
        if bar_ts.tzinfo is not None:
            bar_et = bar_ts.astimezone(ET)
        else:
            bar_et = pytz.utc.localize(bar_ts).astimezone(ET)

        bucket_minute = (bar_et.minute // 15) * 15
        bucket_key = f"{bar_et.hour:02d}:{bucket_minute:02d}"

        # Sum volume in current 15-min bucket up to and including bar_idx
        bucket_vol = 0
        bars_in_bucket = 0
        for j in range(max(0, bar_idx - 14), bar_idx + 1):
            j_ts = bars.iloc[j]['timestamp']
            if j_ts.tzinfo is not None:
                j_et = j_ts.astimezone(ET)
            else:
                j_et = pytz.utc.localize(j_ts).astimezone(ET)

            j_bucket_min = (j_et.minute // 15) * 15
            j_key = f"{j_et.hour:02d}:{j_bucket_min:02d}"
            if j_key == bucket_key:
                bucket_vol += bars.iloc[j]['volume']
                bars_in_bucket += 1

        avg_vol = volume_profile.get(bucket_key, 0)
        if avg_vol <= 0:
            return 0.0

        # Scale expected volume by fraction of bucket elapsed (1-15 bars)
        # to avoid penalizing partial buckets — matches scanner behavior
        # where partial 15-min bars have proportionally less volume
        if bars_in_bucket < 15:
            scaled_avg = avg_vol * (bars_in_bucket / 15.0)
        else:
            scaled_avg = avg_vol

        return bucket_vol / scaled_avg if scaled_avg > 0 else 0.0

    def _run_realistic(
        self, symbol: str, bars: pd.DataFrame, trade_date: str,
        avg_daily_volume: Optional[int] = None,
        volume_profile: Optional[Dict[str, int]] = None,
        prev_close: Optional[float] = None,
    ) -> BacktestResult:
        """
        Realistic backtest: detect_setup() fires before breakout, places pending
        buy-stop at flag_high, fills at max(bar_open, breakout_level).

        When prev_close is provided, simulates real-time qualification:
        pattern scanning only starts after the bar where the stock's high
        exceeds prev_close * (1 + intraday_change_pct_min). This matches
        production where the scanner must qualify a stock before the trading
        engine sees it.

        Loop:
        1. Check if stock has qualified (real-time simulation)
        2. Check pending buy-stop against current bar
        3. If no trade and no pending order, scan for new setup
        4. Apply filters (min_price, midday, last_entry_time)
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
        resume_after_bar = 0  # For multi-trade: skip bars until trade exits

        # Real-time qualification gate: simulate the scanner's qualification step.
        # Without prev_close, all bars are scanned (backward compatible).
        # With prev_close, scanning starts only after the stock hits the threshold.
        from config import Config
        _cfg = Config._load_yaml_only()
        qualification_pct = float(
            _cfg.get("scanner", {}).get("intraday_change_pct_min", 10.0)
        ) / 100.0  # Convert 10.0 → 0.10
        qualified = prev_close is None or prev_close <= 0
        qualification_bar = 0

        logger.info(
            f"{symbol}: Scanning {len(bars)} bars for setups (realistic mode)"
            f"{f', qualification at +{qualification_pct:.0%} from ${prev_close:.2f}' if not qualified else ''}..."
        )

        for i in range(self.MIN_BARS_FOR_SETUP - 1, last_end):
            # Real-time qualification check: stock must cross threshold before scanning.
            # In live, stocks with big premarket gaps qualify immediately at open.
            if not qualified:
                bar_high = bars.iloc[i]['high']
                move = (bar_high - prev_close) / prev_close
                if move >= qualification_pct:
                    qualified = True
                    qualification_bar = i
                    logger.info(
                        f"  Bar {i}: QUALIFIED at +{move:.1%} "
                        f"(high=${bar_high:.2f} vs prev_close=${prev_close:.2f})"
                    )
                else:
                    continue  # Not qualified yet — skip all scanning
            # Multi-trade: skip bars while previous trade is still active
            if i < resume_after_bar:
                continue

            bar = bars.iloc[i]
            bar_time_et = self._get_bar_time_et(bar['timestamp'])

            # --- Step 1: Check pending buy-stop ---
            if pending_order is not None and not trade_taken:
                bar_high = bar['high']
                bar_low = bar['low']
                bar_open = bar['open']

                # Cancel pending orders in midday (11:30-14:00 ET).
                # Matches production position_manager.can_open_position() check.
                if self.skip_midday and self._is_midday(bar_time_et):
                    logger.debug(
                        f"  Bar {i}: buy-stop CANCELLED — entered deep midday (12-14 ET)"
                    )
                    pending_order = None

                # Check if setup invalidated (price dropped below flag_low)
                elif bar_low < pending_order.setup.flag_low:
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
                    # Breakout volume check on thin liquidity days (H5 OR filter)
                    # Only applied when min_breakout_vol_override is set (> base 1.5)
                    if self._min_breakout_vol_override > 0:
                        breakout_bar_volume = bar['volume']
                        avg_flag_vol = pending_order.setup.avg_flag_volume
                        if avg_flag_vol > 0:
                            bvr = breakout_bar_volume / avg_flag_vol
                            if bvr < self._min_breakout_vol_override:
                                logger.info(
                                    f"  Bar {i}: breakout REJECTED (thin liquidity) — "
                                    f"volume ratio {bvr:.1f}x "
                                    f"< {self._min_breakout_vol_override:.1f}x min"
                                )
                                pending_order = None
                                continue

                    # Vol dead zone filter: reject breakouts with volume ratio in 2-5x range
                    if self.vol_dead_zone_enabled:
                        breakout_vol = bar['volume']
                        avg_flag_vol = pending_order.setup.avg_flag_volume
                        if avg_flag_vol > 0:
                            bvr_check = breakout_vol / avg_flag_vol
                            if self.vol_dead_zone_min <= bvr_check <= self.vol_dead_zone_max:
                                logger.debug(
                                    f"  Bar {i}: breakout REJECTED (vol dead zone) — "
                                    f"volume ratio {bvr_check:.1f}x in "
                                    f"{self.vol_dead_zone_min:.0f}-{self.vol_dead_zone_max:.0f}x range"
                                )
                                pending_order = None
                                continue

                    # Fill at max(bar_open, breakout_level) * (1 + entry_slippage_pct)
                    raw_fill = max(bar_open, pending_order.breakout_level)
                    fill_price = raw_fill * (1 + self.entry_slippage_pct)
                    plan = pending_order.plan
                    entry_gap = fill_price - pending_order.breakout_level

                    # Reject gap-overs beyond 2% — matches live buy-stop limit.
                    # 15-month data: 0-2% gap trades are profitable (+$314/avg),
                    # >2% gap trades are net losers (23% win rate, -$134/avg).
                    max_gap_pct = 0.02
                    if pending_order.breakout_level > 0:
                        gap_pct = entry_gap / pending_order.breakout_level
                        if gap_pct > max_gap_pct:
                            logger.info(
                                f"  Bar {i}: breakout REJECTED (gap-over) — "
                                f"fill ${fill_price:.2f} is {gap_pct:.1%} above "
                                f"breakout ${pending_order.breakout_level:.2f} "
                                f"(max {max_gap_pct:.0%})"
                            )
                            pending_order = None
                            continue

                    # Gap-fill adjustment: keep stop at TECHNICAL level (flag low),
                    # only shift target up by entry gap. The original stop is where
                    # the pattern actually fails — moving it above that level puts
                    # the stop in no-man's land where normal price noise triggers it.
                    # Dollar risk increases but the stop is meaningful.
                    if entry_gap > 0:
                        adjusted_target = fill_price + plan.risk_per_share * plan.risk_reward_ratio
                        actual_risk = fill_price - plan.stop_loss_price
                        logger.info(
                            f"  Entry gap +${entry_gap:.2f}: "
                            f"stop KEPT at ${plan.stop_loss_price:.2f} (technical level), "
                            f"target ${plan.take_profit_price:.2f} → ${adjusted_target:.2f}, "
                            f"risk ${plan.risk_per_share:.2f} → ${actual_risk:.2f}/sh"
                        )
                        plan = TradePlan(
                            symbol=plan.symbol,
                            entry_price=plan.entry_price,
                            stop_loss_price=plan.stop_loss_price,  # UNCHANGED — technical level
                            take_profit_price=adjusted_target,
                            risk_per_share=actual_risk,
                            reward_per_share=adjusted_target - fill_price,
                            risk_reward_ratio=plan.risk_reward_ratio,
                            shares=plan.shares,
                            total_risk=actual_risk * plan.shares,
                            pattern=plan.pattern,
                        )

                    # JIT liquidity cap: limit shares to X% of fill bar volume.
                    # On big-mover days, bar volume reflects ACTUAL liquidity at
                    # the breakout moment — far more honest than static ADV%.
                    jit_shares = plan.shares
                    bar_volume = bar.get('volume', 0) if hasattr(bar, 'get') else bar['volume']
                    if self.max_bar_participation_pct > 0 and bar_volume > 0:
                        bar_cap = int(bar_volume * self.max_bar_participation_pct)
                        if bar_cap > 0 and jit_shares > bar_cap:
                            logger.info(
                                f"  JIT cap: {jit_shares} → {bar_cap} shares "
                                f"({self.max_bar_participation_pct:.0%} of {bar_volume:,} bar vol)"
                            )
                            jit_shares = bar_cap
                            # Rebuild plan with capped shares
                            plan = TradePlan(
                                symbol=plan.symbol,
                                entry_price=plan.entry_price,
                                stop_loss_price=plan.stop_loss_price,
                                take_profit_price=plan.take_profit_price,
                                risk_per_share=plan.risk_per_share,
                                reward_per_share=plan.reward_per_share,
                                risk_reward_ratio=plan.risk_reward_ratio,
                                shares=jit_shares,
                                total_risk=plan.risk_per_share * jit_shares,
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
                    pending_order = None

                    logger.info(
                        f"  TRADE EXIT ({trade.exit_reason}): "
                        f"${trade.exit_price:.2f}, "
                        f"P&L ${trade.pnl:.2f} ({trade.pnl_pct:+.1f}%)"
                    )

                    if self.early_exit_after_trade:
                        trade_taken = True
                        break
                    else:
                        # Multi-trade: mark the bar where scanning can resume.
                        # Can't modify loop variable i, so use resume_after gate.
                        trade_taken = False
                        pending_order = None
                        if trade.exit_time is not None:
                            for skip_idx in range(i + 1, len(bars)):
                                if bars.iloc[skip_idx]['timestamp'] >= trade.exit_time:
                                    resume_after_bar = skip_idx
                                    break

            # --- Step 2: Scan for new setup ---
            if not trade_taken and pending_order is None:
                # Last entry time check
                if bar_time_et >= self.last_entry_time_et:
                    continue

                # SPY MACD afternoon cutoff: block when SPY MACD > 0 after cutoff time
                if self._spy_macd_cutoff is not None and self._spy_macd_cutoff.is_blocked(bar_time_et):
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

                plan = self.planner.create_plan(
                    setup, avg_daily_volume=self._current_avg_daily_volume)
                if plan is None:
                    logger.debug(f"  Plan rejected at bar {i}")
                    continue

                # Min stop distance filter (reject tick-noise setups)
                stop_dist = plan.entry_price - plan.stop_loss_price
                if self.min_stop_distance > 0 and stop_dist < self.min_stop_distance:
                    logger.debug(
                        f"  Skipping — stop dist ${stop_dist:.2f} "
                        f"< min ${self.min_stop_distance:.2f}"
                    )
                    continue

                # Price filter
                if self.min_price > 0 and plan.entry_price < self.min_price:
                    logger.debug(
                        f"  Skipping — entry ${plan.entry_price:.2f} below "
                        f"min price ${self.min_price:.2f}"
                    )
                    continue

                # Midday filter
                if self.skip_midday and self._is_midday(bar_time_et):
                    logger.debug(
                        f"  Skipping — midday setup at {bar['timestamp']} "
                        f"(11:30-14:00 ET filter)"
                    )
                    continue

                # Relative volume check at entry time
                if self.relative_volume_min > 0:
                    rvol = 0.0
                    rvol_label = ""

                    if self.rvol_mode == 'bucket' and volume_profile:
                        # Bucket mode: 15-min bucket vol vs profile avg
                        # Matches the live scanner's actual implementation
                        rvol = self._get_bucket_rvol(bars, i, volume_profile)
                        rvol_label = "bucket"
                    elif self.rvol_mode == 'cumulative' and avg_daily_volume and avg_daily_volume > 0:
                        # Cumulative mode: total vol so far vs expected by time
                        # Ross Cameron's stated definition of relative volume
                        cumulative_vol = bars.iloc[:i + 1]['volume'].sum()
                        minutes_elapsed = max(1, (i + 1))
                        expected_vol = avg_daily_volume * (minutes_elapsed / self.TRADING_MINUTES)
                        rvol = cumulative_vol / expected_vol if expected_vol > 0 else 0
                        rvol_label = "cumulative"
                    else:
                        # No volume data available — skip filter
                        rvol = float('inf')

                    if rvol < self.relative_volume_min:
                        logger.debug(
                            f"  Skipping — {rvol_label} rvol {rvol:.1f}x "
                            f"< {self.relative_volume_min:.1f}x at bar {i}"
                        )
                        continue

                # MACD zone filter: skip dead zone, scale risk on strong zones
                # Checked at setup time (matches production — decide before placing order)
                if self.macd_zones_enabled:
                    zone_mult = self._get_macd_zone_multiplier(
                        symbol, bars, i, plan.entry_price
                    )
                    if zone_mult == 0.0:
                        continue  # dead zone — don't place order
                    elif zone_mult != 1.0:
                        scaled_shares = min(self.planner.max_shares, max(1, int(plan.shares * zone_mult)))
                        plan = TradePlan(
                            symbol=plan.symbol,
                            entry_price=plan.entry_price,
                            stop_loss_price=plan.stop_loss_price,
                            take_profit_price=plan.take_profit_price,
                            risk_per_share=plan.risk_per_share,
                            reward_per_share=plan.reward_per_share,
                            risk_reward_ratio=plan.risk_reward_ratio,
                            shares=scaled_shares,
                            total_risk=plan.risk_per_share * scaled_shares,
                            pattern=plan.pattern,
                        )

                # UD risk scaling: reduce size on euphoric SPY days
                if self._ud_risk_scale > 0 and self._ud_risk_scale != 1.0:
                    ud_shares = min(self.planner.max_shares, max(1, int(plan.shares * self._ud_risk_scale)))
                    if ud_shares != plan.shares:
                        logger.debug(
                            f"  UD scaling {self._ud_risk_scale}x → "
                            f"shares {plan.shares} → {ud_shares}"
                        )
                        plan = TradePlan(
                            symbol=plan.symbol,
                            entry_price=plan.entry_price,
                            stop_loss_price=plan.stop_loss_price,
                            take_profit_price=plan.take_profit_price,
                            risk_per_share=plan.risk_per_share,
                            reward_per_share=plan.reward_per_share,
                            risk_reward_ratio=plan.risk_reward_ratio,
                            shares=ud_shares,
                            total_risk=plan.risk_per_share * ud_shares,
                            pattern=plan.pattern,
                        )

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
    # Uses pytz for DST-safe conversion (EDT: -4h, EST: -5h)
    trade_date = datetime.strptime(args.date, "%Y-%m-%d")
    _ET = pytz.timezone('US/Eastern')
    market_open_et = _ET.localize(trade_date.replace(hour=9, minute=30, second=0))
    market_close_et = _ET.localize(trade_date.replace(hour=16, minute=0, second=0))
    market_open = market_open_et.astimezone(timezone.utc)
    market_close = market_close_et.astimezone(timezone.utc)

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

    # Fetch previous trading day bars for MACD warm-up
    prev_day_bars = None
    try:
        from datetime import timedelta
        prev_date = trade_date - timedelta(days=1)
        # Skip weekends
        while prev_date.weekday() >= 5:
            prev_date -= timedelta(days=1)
        prev_open_et = _ET.localize(prev_date.replace(hour=9, minute=30, second=0))
        prev_close_et = _ET.localize(prev_date.replace(hour=16, minute=0, second=0))
        prev_open = prev_open_et.astimezone(timezone.utc)
        prev_close = prev_close_et.astimezone(timezone.utc)
        prev_day_bars = client.get_historical_1min_bars(symbol, prev_open, prev_close)
        if prev_day_bars is not None and not prev_day_bars.empty:
            logger.info(f"MACD warm-up: {len(prev_day_bars)} bars from {prev_date.strftime('%Y-%m-%d')}")
        else:
            logger.warning(f"No previous-day bars for MACD warm-up")
            prev_day_bars = None
    except Exception as e:
        logger.warning(f"Failed to fetch prev-day bars for MACD warm-up: {e}")

    # Run backtest
    runner = BacktestRunner()
    result = runner.run(symbol, bars, args.date, prev_day_bars=prev_day_bars)

    # Print report
    print_report(result)


if __name__ == "__main__":
    main()
