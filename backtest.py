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
from trading.exit_reasons import ExitReason
from trading.pattern_detector import BullFlagDetector, BullFlagPattern, BullFlagSetup
from trading.trade_planner import TradePlanner, TradePlan
from trading.news_kill_guard import news_kill_decision
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
    # Conviction multiplier at setup time (for cache → batch filter cap alignment)
    conviction_mult: float = 1.0
    # MACD zone multiplier (for cache → batch filter tier interaction)
    macd_zone_mult: float = 1.0
    # Per-rule conviction contributions + raw inputs (Phase A — V2 research).
    # Cached so post-hoc analyses can recompute conviction with new rules
    # without rebuilding the whole BT. Each conv_* is the rule's signed
    # contribution; spy_3d_range is rule 4's input (computed live, not in qf_*).
    conv_pole_gain: float = 0.0
    conv_flag_tightness: float = 0.0
    conv_vol_ratio: float = 0.0
    conv_spy_regime: float = 0.0
    conv_retracement: float = 0.0
    # V2_clean rules (added 2026-04-15)
    conv_vwap_dist: float = 0.0
    conv_gap_fading: float = 0.0
    conv_raw_score: float = 1.0
    spy_3d_range: float = 0.0


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
        vol_confirmed_trail_enabled: bool = False,
        vol_confirmed_trail_min_ratio: float = 1.0,
        # Static lock (ORB-style): touch +static_lock_arm_r → stop ratchets to
        # entry + static_lock_at_r * R ONCE and NEVER moves further. Disables
        # trailing after lock arms. Both fields > 0 to enable.
        static_lock_arm_r: float = 0.0,
        static_lock_at_r: float = 0.0,
        # 2026-09-06 profit partial (trading/bf_profit_partial.py — ONE spec
        # with live StopMonitor): sell `fraction` at +r_multiple R (plan-R
        # baseline, closed-bar high), stop to the fill (breakeven), the
        # remainder keeps trailing. None/disabled = byte-identical legacy.
        profit_partial: Optional['ProfitPartialConfig'] = None,
        # Planned-R variant: when True, R-based math (activation, breakeven,
        # static lock, trail ratchet, r_gain) uses (plan.entry_price -
        # plan.stop_loss_price) — the SETUP's structural R — instead of
        # (actual_fill - plan.stop_loss_price). De-couples trail behavior
        # from entry slippage so a fast breakout that fills 1-2% above the
        # planned breakout level doesn't push the activation gate further
        # away. Hard stop stays at plan.stop_loss_price either way.
        use_planned_r: bool = False,
        # 2026-09-05 BF trail unification: ONE R-basis knob shared with live
        # (`trading.trailing_stop.r_basis`, default 'plan'). `use_planned_r`
        # is kept for the two legacy studies; True forces 'plan'. When
        # neither is given the simulator uses the shared default — the
        # basis live has traded since 2026-05-08 (README Bug 5).
        r_basis: Optional[str] = None,
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
        from trading.bf_profit_partial import DISABLED as _PP_DISABLED
        self.profit_partial = profit_partial or _PP_DISABLED
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
        # Volume-confirmed trail exit (Experiment D): when trailing stop triggers,
        # require the stop-crossing bar to have volume >= min_ratio × flag avg.
        # Low-volume drift-downs are skipped (treated as noise, not active selling).
        # Prevents exit on passive pullbacks within a slow-burn rally.
        self.vol_confirmed_trail_enabled = vol_confirmed_trail_enabled
        self.vol_confirmed_trail_min_ratio = vol_confirmed_trail_min_ratio
        # Static lock (ORB-style)
        self.static_lock_arm_r = static_lock_arm_r
        self.static_lock_at_r = static_lock_at_r
        # R basis — shared contract with live (trading/bf_trail.py).
        from trading.bf_trail import normalize_r_basis, R_BASIS_PLAN
        if r_basis is None:
            r_basis = R_BASIS_PLAN if use_planned_r else None
        self.r_basis = normalize_r_basis(r_basis)
        self.use_planned_r = (self.r_basis == R_BASIS_PLAN)

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
        use_trail = self.trailing_stop_r > 0 or self.static_lock_arm_r > 0
        trailing_active = False
        static_locked = False
        highest_since_entry = actual_entry
        current_stop = trade.stop_loss
        # R basis via the SHARED helper (trading/bf_trail.py) — same call
        # live makes in StopMonitor._r_baseline_and_unit. 'plan': R is the
        # SETUP's structural risk (planned_entry - planned_stop) measured
        # from planned_entry; 'fill': legacy fill-based. Hard stop is
        # unchanged either way (trade.stop_loss = plan.stop_loss_price).
        from trading.bf_trail import r_baseline_and_unit, arm_and_ratchet
        r_baseline, risk = r_baseline_and_unit(
            planned_entry=plan.entry_price,
            planned_stop=plan.stop_loss_price,
            fill_price=actual_entry,
            fill_stop=plan.stop_loss_price,
            r_basis=self.r_basis,
        )

        # Exhaustion exit state
        exhaust_partial_taken = False
        active_shares = trade.shares
        effective_trail_r = self.trailing_stop_r
        # Profit partial state (shared spec; level anchored on the SAME
        # r_baseline/risk the trail uses)
        from trading.bf_profit_partial import (
            breakeven_stop as _pp_breakeven, partial_level as _pp_level,
            partial_shares as _pp_shares, profit_partial_fires as _pp_fires)
        pp_taken = False
        pp_level = (_pp_level(r_baseline, risk, self.profit_partial.r_multiple)
                    if (self.profit_partial.enabled and risk > 0) else 0.0)

        for i in range(entry_bar_idx + 1, len(bars)):
            bar = bars.iloc[i]

            # Force close check
            if self.force_close_time_et is not None:
                bar_et = self._get_bar_time_et(bar['timestamp'])
                if bar_et >= self.force_close_time_et:
                    reason = ('exhaust+force_close' if exhaust_partial_taken
                              else ('pp+force_close' if pp_taken else 'force_close'))
                    self._exit_trade(trade, bar, reason, bar['open'],
                                     active_shares=active_shares if (exhaust_partial_taken or pp_taken) else None)
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
                    elif pp_taken:
                        reason = 'pp+no_pop'
                    self._exit_trade(trade, bar, reason, bar['close'],
                                     active_shares=active_shares if (exhaust_partial_taken or pp_taken) else None)
                    logger.debug(
                        f"  Bar {i}: NO-POP exit after {self.no_pop_exit_bars} bars — "
                        f"max +{max_move_pct:.2%} < {self.no_pop_exit_min_pct:.1%}"
                    )
                    return trade

            hit_stop = bar_low <= current_stop

            # Experiment D: volume-confirmed trail exit. Shared helper gates the
            # trail-stop trigger; skipped bars leave the stop in place and the
            # next bar re-evaluates. Does NOT apply to initial hard stop.
            if (hit_stop and use_trail and trailing_active
                    and self.vol_confirmed_trail_enabled):
                from trading.trail_vol_guard import should_skip_trail_exit_on_low_vol
                flag_vol = plan.pattern.avg_flag_volume if plan.pattern else None
                # 2026-09-05 parity fix: confirm on the PREVIOUS closed bar,
                # exactly what live's `watch.last_bar_volume` holds when the
                # tick crosses the stop. Using this bar's own volume was
                # lookahead — the volume of the minute the stop trips is
                # not known until the minute ends.
                cur_vol = bars.iloc[i - 1]['volume']
                if should_skip_trail_exit_on_low_vol(
                    bar_volume=cur_vol,
                    flag_avg_volume=flag_vol,
                    min_vol_ratio=self.vol_confirmed_trail_min_ratio,
                ):
                    hit_stop = False
                    logger.info(
                        f"  Bar {i}: TRAIL VOL-CONF SKIP — bar_vol={cur_vol:,} < "
                        f"{self.vol_confirmed_trail_min_ratio}×flag_avg={flag_vol:,.0f}"
                    )

            if use_trail:
                # With trailing stop: no fixed TP, trail replaces it
                if hit_stop:
                    # Use trail-specific slippage if set (reversals may have worse fills)
                    slip = self.trail_exit_slippage_pct if (trailing_active and self.trail_exit_slippage_pct is not None) else None
                    stop_fill = self._compute_stop_fill(current_stop, slippage_override=slip)
                    reason = 'trail_stop' if trailing_active else 'stop'
                    if exhaust_partial_taken:
                        reason = 'exhaust+' + reason
                    elif pp_taken:
                        reason = 'pp+' + reason
                    self._exit_trade(trade, bar, reason, stop_fill,
                                     active_shares=active_shares if (exhaust_partial_taken or pp_taken) else None)
                    logger.debug(f"  Bar {i}: {reason} at ${stop_fill:.2f}")
                    return trade

                # Profit partial (2026-09-06, shared spec): closed bar's high
                # reached +r_multiple R → sell `fraction` at this bar's close
                # (market after the close; stop-fill model), stop → fill price
                # (true breakeven) on the remainder. Same-bar stop+partial:
                # the stop check above already ran — stop wins (conservative).
                if (pp_level > 0 and not pp_taken and not exhaust_partial_taken
                        and _pp_fires(bar_high, pp_level)):
                    _psh = _pp_shares(active_shares, self.profit_partial.fraction)
                    if _psh > 0:
                        _pfill = self._compute_stop_fill(float(bar['close']))
                        trade.partial_exit_taken = True
                        trade.partial_exit_time = bar['timestamp']
                        trade.partial_exit_price = _pfill
                        trade.partial_shares = _psh
                        trade.partial_pnl = (_pfill - actual_entry) * _psh
                        trade.remaining_shares = active_shares - _psh
                        active_shares -= _psh
                        pp_taken = True
                        if self.profit_partial.move_to_breakeven:
                            current_stop = _pp_breakeven(current_stop, actual_entry)
                            trade.breakeven_stop_active = True
                        logger.debug(
                            f"  Bar {i}: PROFIT PARTIAL {_psh}sh at ${_pfill:.2f} "
                            f"(level ${pp_level:.2f} = +{self.profit_partial.r_multiple}R), "
                            f"P&L ${trade.partial_pnl:.2f}, remaining {active_shares}sh, "
                            f"stop ${current_stop:.2f}"
                        )

                # Stage 1: Move stop to breakeven (+ optional profit) after +breakeven_at_r.
                # Levels are anchored at r_baseline (= planned_entry under planned-R,
                # fill price under fill-R) so slippage doesn't shift the gates.
                if self.breakeven_at_r > 0 and risk > 0 and not trailing_active:
                    r_gain = (highest_since_entry - r_baseline) / risk
                    breakeven_target = r_baseline + self.breakeven_profit_r * risk
                    if r_gain >= self.breakeven_at_r and current_stop < breakeven_target:
                        current_stop = breakeven_target
                        logger.debug(
                            f"  Bar {i}: BREAKEVEN stop at +{r_gain:.1f}R → "
                            f"stop ${breakeven_target:.2f} (+{self.breakeven_profit_r}R)"
                        )

                # Compute r_gain once (used by Stage 2, 2.5, ratchet, static lock)
                r_gain = (highest_since_entry - r_baseline) / risk if risk > 0 else 0.0

                # Static lock (ORB-style): one-shot stop move at +arm_r touch.
                # Stop ratchets to r_baseline + lock_r × risk and NEVER moves again.
                # Disables trailing-style ratcheting below.
                if (self.static_lock_arm_r > 0 and not static_locked
                        and risk > 0):
                    if r_gain >= self.static_lock_arm_r:
                        lock_target = r_baseline + self.static_lock_at_r * risk
                        if lock_target > current_stop:
                            current_stop = lock_target
                        static_locked = True
                        trailing_active = True  # so exit reports as 'trail_stop'
                        logger.debug(
                            f"  Bar {i}: STATIC LOCK at +{r_gain:.1f}R → "
                            f"stop ${lock_target:.2f} (+{self.static_lock_at_r}R, NO TRAIL)"
                        )

                # Stage 2: arm + ratchet via the SHARED state machine
                # (trading/bf_trail.py::arm_and_ratchet — live's closed-bar
                # path calls the identical function). Runs AFTER this bar's
                # stop check, so the stop it produces is live from the next
                # bar on. Skipped when static-locked (one-shot lock is final).
                if not static_locked and risk > 0 and self.trailing_stop_r > 0:
                    step = arm_and_ratchet(
                        bar_high=bar_high,
                        highest_since_entry=highest_since_entry,
                        current_stop=current_stop,
                        trailing_active=trailing_active,
                        r_baseline=r_baseline,
                        r_unit=risk,
                        activate_at_r=self.trailing_activate_at_r,
                        trail_r=effective_trail_r,
                    )
                    highest_since_entry = step.highest
                    trailing_active = step.trailing_active
                    if step.ratcheted:
                        current_stop = step.stop

                # Stage 2.5: Tighten trail after passing threshold (e.g., 2.5R)
                # Locks in more profit on runners without capping them
                if (trailing_active and not static_locked and not exhaust_partial_taken
                        and self.trail_tighten_at_r > 0 and risk > 0):
                    if r_gain >= self.trail_tighten_at_r and effective_trail_r > self.trail_tightened_r:
                        effective_trail_r = self.trail_tightened_r
                        logger.debug(
                            f"  Bar {i}: trail TIGHTENED at +{r_gain:.1f}R → "
                            f"{effective_trail_r}R trail"
                        )

                # Re-ratchet after a Stage-2.5 tighten on this same bar
                # (effective_trail_r may have just shrunk). Same shared
                # function; a no-op when nothing changed.
                if trailing_active and not static_locked and risk > 0 and self.trailing_stop_r > 0:
                    step = arm_and_ratchet(
                        bar_high=bar_high,
                        highest_since_entry=highest_since_entry,
                        current_stop=current_stop,
                        trailing_active=True,
                        r_baseline=r_baseline,
                        r_unit=risk,
                        activate_at_r=self.trailing_activate_at_r,
                        trail_r=effective_trail_r,
                    )
                    if step.ratcheted:
                        current_stop = step.stop

                # Exhaustion exit: check signals when profitable, sell partial
                if (self.exhaustion_exit_enabled and not exhaust_partial_taken
                        and not pp_taken and risk > 0):
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
        reason = 'exhaust+eod' if exhaust_partial_taken else ('pp+eod' if pp_taken else 'eod')
        self._exit_trade(trade, last_bar, reason, last_bar['close'],
                         active_shares=active_shares if (exhaust_partial_taken or pp_taken) else None)
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
        min_cum_dollar_vol: float = 0,
        min_cum_shares: int = 0,
        min_relative_vol_rate: float = 0,
        avg_daily_volume: int = 0,
        db: Optional["Database"] = None,
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
        # Research env overrides (2026-07-11 session-structure study —
        # mirrors the BF_* pattern in pattern_detector.py: default OFF,
        # no behavior change without explicit env). Used by twin cache
        # rebuilds to /tmp; NEVER set in production.
        _le_env = os.environ.get('BF_LAST_ENTRY')
        if _le_env:
            _h, _m = _le_env.split(':')
            self.last_entry_time_et = (int(_h), int(_m))
            logger.warning(f"BF_LAST_ENTRY override active: {_le_env} "
                           f"(research twin — not production behavior)")
        _sm_env = os.environ.get('BF_SKIP_MIDDAY')
        if _sm_env is not None:
            self.skip_midday = _sm_env.strip().lower() not in (
                '0', 'false', 'no', 'off', '')
            logger.warning(f"BF_SKIP_MIDDAY override active: "
                           f"{self.skip_midday} (research twin)")
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

        # Partial profit (early scale-out): load from config if not explicitly passed.
        # When enabled, take partial_profit_fraction of shares at +r_multiple*R,
        # remainder runs with trail/exhaustion. Improves daily consistency by
        # locking in a small win before trail activation.
        pp_cfg = trading_cfg.get("partial_profit", {})
        pp_enabled_cfg = bool(pp_cfg.get("enabled", False))
        if not partial_profit_enabled and pp_enabled_cfg:
            partial_profit_enabled = True
            partial_profit_r_multiple = float(pp_cfg.get("r_multiple", partial_profit_r_multiple))
            partial_profit_fraction = float(pp_cfg.get("fraction", partial_profit_fraction))

        # No-pop exit (scratch-trade): load from config if not explicitly passed.
        # Exits at entry price if price hasn't moved up min_pct within N bars.
        # Cuts slow-bleeding losers before they hit stop.
        npe_cfg = trading_cfg.get("no_pop_exit", {})
        if no_pop_exit_bars == 0 and bool(npe_cfg.get("enabled", False)):
            no_pop_exit_bars = int(npe_cfg.get("bars", 5))
            no_pop_exit_min_pct = float(npe_cfg.get("min_pct", 0.005))

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

        # Profit partial (2026-09-06): trading.profit_partial — the shared
        # BT/live spec; default off (byte-identical books until flipped).
        from trading.bf_profit_partial import load_profit_partial_config
        self.profit_partial = load_profit_partial_config(trading_cfg)
        if self.profit_partial.enabled:
            logger.info(
                f"Profit partial: {self.profit_partial.fraction:.0%} at "
                f"+{self.profit_partial.r_multiple}R, breakeven="
                f"{self.profit_partial.move_to_breakeven} (shared spec, trading/bf_profit_partial.py)"
            )

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
                profit_partial=self.profit_partial,
                no_pop_exit_bars=no_pop_exit_bars,
                no_pop_exit_min_pct=no_pop_exit_min_pct,
                trail_exit_slippage_pct=trail_exit_slippage_pct,
                vol_confirmed_trail_enabled=bool(trail_cfg.get("vol_confirmed_exit", {}).get("enabled", False)),
                vol_confirmed_trail_min_ratio=float(trail_cfg.get("vol_confirmed_exit", {}).get("min_vol_ratio", 1.0)),
                # 2026-09-05: ONE R-basis knob read by BT and live alike.
                r_basis=trail_cfg.get("r_basis"),
            )
            logger.info(
                f"Trail R-basis: {self.simulator.r_basis} "
                f"(trading.trailing_stop.r_basis; shared with live via trading/bf_trail.py)"
            )
        # Volume qualification gates (for live/backtest alignment testing)
        self.min_cum_dollar_vol = min_cum_dollar_vol
        self.min_cum_shares = min_cum_shares
        self.min_relative_vol_rate = min_relative_vol_rate
        self.avg_daily_volume = avg_daily_volume
        # BT-LIVE marginability parity (2026-05-01): when provided, BT reads
        # universe.is_marginable to mirror LIVE's risk-tier downgrade on
        # non-marginable symbols. None = fail open (legacy behavior).
        self.db = db

        # Risk tiers: same logic as trading_engine._get_risk_tier()
        tier_cfg = trading_cfg.get("risk_tiers", {})
        self.risk_tiers_enabled = bool(tier_cfg.get("enabled", False))
        self.risk_tiers = []
        if self.risk_tiers_enabled:
            for prefix in ['tier1', 'tier2', 'tier3']:
                mult = float(tier_cfg.get(f"{prefix}_multiplier", 0))
                if mult > 0:
                    self.risk_tiers.append({
                        'min_price': float(tier_cfg.get(f"{prefix}_min_price", 0)),
                        'max_price': float(tier_cfg.get(f"{prefix}_max_price", 999)),
                        'min_volume': int(tier_cfg.get(f"{prefix}_min_volume", 0)),
                        'max_volume': int(tier_cfg.get(f"{prefix}_max_volume", 999999999)),
                        'multiplier': mult,
                    })
            if self.risk_tiers:
                logger.info(f"Risk tiers: {len(self.risk_tiers)} tiers loaded")

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
        # Per-tier MACD zone multipliers for Extras tier (10% ≤ intraday < 20%).
        # Falls back to A-tier defaults if extras_tier block absent.
        _extras_cfg = macd_zones_cfg.get("extras_tier", {}) or {}
        self.macd_extras_strong_pos_multiplier = float(
            _extras_cfg.get("strong_pos_multiplier", self.macd_strong_pos_multiplier))
        self.macd_extras_strong_neg_multiplier = float(
            _extras_cfg.get("strong_neg_multiplier", self.macd_strong_neg_multiplier))
        self.macd_extras_normal_multiplier = float(
            _extras_cfg.get("normal_multiplier", self.macd_normal_multiplier))
        self._prev_day_bars: Optional[pd.DataFrame] = None

        if self.macd_zones_enabled:
            logger.info(
                f"MACD zones — A-tier (≥20%): dead [{self.macd_dead_zone_min}%, "
                f"{self.macd_dead_zone_max}%], "
                f"strong neg <{self.macd_strong_neg_threshold}% → {self.macd_strong_neg_multiplier}x, "
                f"strong pos >{self.macd_strong_pos_threshold}% → {self.macd_strong_pos_multiplier}x, "
                f"normal → {self.macd_normal_multiplier}x | "
                f"Extras-tier (10-19.99%): strong → "
                f"{self.macd_extras_strong_pos_multiplier}x, "
                f"normal → {self.macd_extras_normal_multiplier}x"
            )

        # Quality filter: skip low-probability setups based on pre-entry features
        qf_cfg = trading_cfg.get("quality_filter", {})
        self.quality_filter_enabled = bool(qf_cfg.get("enabled", False))
        self.qf_max_vwap_dist = float(qf_cfg.get("max_vwap_distance_pct", 4.0))
        self.qf_gap_fade_threshold = float(qf_cfg.get("gap_fade_threshold_pct", 15.0))
        self.qf_min_spy_return = float(qf_cfg.get("min_spy_return_pct", -0.3))
        self.qf_slow_pole_max_bars = int(qf_cfg.get("slow_pole_max_bars", 15))
        self.qf_slow_pole_min_gain = float(qf_cfg.get("slow_pole_min_gain_pct", 5.0))
        self._spy_bars_cache: Dict[str, pd.DataFrame] = {}
        self._db_path: Optional[str] = None

        if self.quality_filter_enabled:
            logger.info(
                f"Quality filter: VWAP>{self.qf_max_vwap_dist}%, "
                f"gap_fade>{self.qf_gap_fade_threshold}%, "
                f"SPY<{self.qf_min_spy_return}%, "
                f"slow_pole>{self.qf_slow_pole_max_bars}bars/<{self.qf_slow_pole_min_gain}%"
            )

        # Regime-aware sizing (Phase 1.4b ship, 2026-04-18). Classifies each
        # trading day A/B/C1/C2 from SPY T-1 features; applies per-regime
        # multiplier ON TOP of macd_zone * conviction sizing. C2 regime (shallow
        # dip in uptrend) → multiplier 0 → skip trade. Shared classifier with
        # PROD via trading/regime_helpers.py (parity by construction).
        _regime_cfg_dict = trading_cfg.get("regime_sizing", {}) or {}
        _regime_mults_raw = _regime_cfg_dict.get("multipliers", {}) or {}
        self.regime_sizing_enabled = bool(_regime_cfg_dict.get("enabled", False))
        self.regime_vol_threshold = float(_regime_cfg_dict.get("vol_threshold_pct", 22.0))
        self.regime_slope_threshold = float(_regime_cfg_dict.get("slope_threshold_pct", 0.15))
        self.regime_multipliers: Dict[str, float] = {
            "A":  float(_regime_mults_raw.get("A",  1.0)),
            "B":  float(_regime_mults_raw.get("B",  1.0)),
            "C1": float(_regime_mults_raw.get("C1", 1.0)),
            "C2": float(_regime_mults_raw.get("C2", 1.0)),
        }
        self._regime_by_date: Dict[str, str] = {}   # lazily built on first use
        self._regime_lookup_built = False
        if self.regime_sizing_enabled:
            logger.info(
                f"Regime sizing: vol>={self.regime_vol_threshold}%=B, "
                f"slope>{self.regime_slope_threshold}%→C2, "
                f"mults A={self.regime_multipliers['A']} "
                f"B={self.regime_multipliers['B']} "
                f"C1={self.regime_multipliers['C1']} "
                f"C2={self.regime_multipliers['C2']} (0=skip)"
            )

        # Conviction scoring: scale position size based on setup quality
        conv_cfg = trading_cfg.get("conviction_scoring", {})
        self.conviction_enabled = bool(conv_cfg.get("enabled", False))

        # Post-fill gate thresholds (IREZ post-mortem 2026-05-08).
        # Defaults 0.5 / 0.5 — see docs/post_fill_gate_variant_analysis.md.
        # Override via `trading.post_fill_gate.{spy_3d_threshold,bk_ratio_threshold}`
        # in config.yaml. `enabled=false` short-circuits the entire kill switch.
        # Read directly from yaml to keep parity with trading_engine.py without
        # introducing a Config singleton dependency in the BT entry point.
        _pfg = trading_cfg.get("post_fill_gate", {}) or {}
        self.post_fill_gate_enabled = bool(_pfg.get("enabled", True))
        self.post_fill_gate_spy_threshold = float(_pfg.get("spy_3d_threshold", 0.5))
        self.post_fill_gate_bk_threshold = float(_pfg.get("bk_ratio_threshold", 0.5))
        # Conviction filter (skip trades below threshold). 0.0 = disabled.
        # Mirrors trading_engine.py for BT/PROD parity.
        # Env var override for sweep / research runs that need to capture all
        # candidates in cache (so post-hoc sweeps over conviction params are honest).
        import os as _os
        _conv_env = _os.environ.get('BT_CONV_THRESHOLD_OVERRIDE')
        if _conv_env is not None:
            self.conviction_min_threshold = float(_conv_env)
        else:
            self.conviction_min_threshold = float(conv_cfg.get("min_threshold", 0.0))
        # Sanity-check threshold at startup (parity with trading_engine.py)
        if self.conviction_min_threshold > 3.0:
            logger.warning(
                f"conviction_min_threshold={self.conviction_min_threshold:.2f} > 3.0 "
                f"(max possible conviction score). ALL trades will be blocked. "
                f"Did you mean {self.conviction_min_threshold/10:.2f}?"
            )
        elif self.conviction_min_threshold < 0:
            logger.warning(
                f"conviction_min_threshold={self.conviction_min_threshold:.2f} < 0 "
                f"— filter is INACTIVE (threshold must be > 0)."
            )
        if not self.conviction_enabled and self.conviction_min_threshold > 0:
            logger.warning(
                f"conviction_min_threshold={self.conviction_min_threshold:.2f} set but "
                f"conviction_scoring.enabled=false — filter is INACTIVE."
            )
        if self.conviction_enabled:
            msg = "Conviction scoring: ENABLED"
            if self.conviction_min_threshold > 0:
                msg += f" — filter trades with conv < {self.conviction_min_threshold:.2f}"
            logger.info(msg)
        # Marginal-conviction defensive scaling (Experiment H, 2026-04-17).
        # Feature-flagged under trading.conviction_scoring.marginal_scaling.
        # When enabled, trades with conviction in [min_threshold, upper_bound)
        # have SIZING scaled by scale_factor. The cached conviction_mult stays
        # as the raw quality value so Stage-2 filters see it unchanged.
        _marg_cfg = conv_cfg.get("marginal_scaling", {}) or {}
        _marg_enabled = bool(_marg_cfg.get("enabled", False))
        self.conviction_marginal_scale_factor = (
            float(_marg_cfg.get("scale_factor", 0.5)) if _marg_enabled else 1.0
        )
        self.conviction_marginal_upper = float(_marg_cfg.get("upper_bound", 1.7))
        # V-reversal bonus (Experiment V, 2026-04-17). Feature-flagged under
        # trading.conviction_scoring.v_reversal_bonus. When disabled, Rule 9
        # never fires regardless of setup. When enabled, adds `bonus` to raw
        # conviction score for gap-down V-reversal setups.
        _vrev_cfg = conv_cfg.get("v_reversal_bonus", {}) or {}
        self.v_reversal_enabled = bool(_vrev_cfg.get("enabled", False))
        self.v_reversal_bonus = float(_vrev_cfg.get("bonus", 0.4))
        self.v_reversal_gap_pct_max = float(_vrev_cfg.get("gap_pct_max", 0.0))
        self.v_reversal_intraday_range_min = float(
            _vrev_cfg.get("intraday_range_min", 20.0))
        self.v_reversal_pole_gain_min = float(
            _vrev_cfg.get("pole_gain_min", 5.0))

        # News kill rules: block no-news trades in specific loser segments
        nkr_cfg = trading_cfg.get("news_kill_rules", {})
        self.news_kill_enabled = bool(nkr_cfg.get("enabled", False))
        self.nkr_max_avg_vol = float(nkr_cfg.get("max_avg_vol_no_news", 3_000_000))
        self.nkr_min_price = float(nkr_cfg.get("min_price_no_news", 3.0))
        self.nkr_max_float = float(nkr_cfg.get("max_float_no_news", 30_000_000))
        # Catalyst exemption — default OFF (2026-05 A/B found the exemption is
        # value-destroying; the segment rules now apply to every trade). See
        # trading/news_kill_guard.py.
        self.news_kill_catalyst_exemption = bool(
            nkr_cfg.get("catalyst_exemption", False))
        self._news_cache: Dict[str, bool] = {}  # (symbol, date) → has_real_catalyst

        # News-classifier A/B (research). BT_NEWS_CLASSIFIER ∈ regex|haiku|
        # haiku_revised reads precomputed verdicts from data/news_ab.db so the
        # arms differ ONLY by classifier. Unset → production behavior (regex via
        # news_history) is byte-identical. BT_NEWS_KILL=0/1 force the news-kill
        # gate off/on — the "no news filter at all" A/B arm uses BT_NEWS_KILL=0.
        import os as _os
        self._news_ab_mode = _os.environ.get('BT_NEWS_CLASSIFIER') or None
        self._news_ab_store = None
        if self._news_ab_mode and self._news_ab_mode not in (
                'regex', 'haiku', 'haiku_revised'):
            raise ValueError(
                f"BT_NEWS_CLASSIFIER='{self._news_ab_mode}' invalid — "
                f"expected one of: regex, haiku, haiku_revised")
        if self._news_ab_mode:
            logger.info(
                f"News A/B ENABLED: classifier='{self._news_ab_mode}' "
                f"(precomputed data/news_ab.db)")
        _nk_env = _os.environ.get('BT_NEWS_KILL')
        if _nk_env in ('0', '1'):
            self.news_kill_enabled = (_nk_env == '1')
            logger.info(
                f"News-kill gate forced "
                f"{'ON' if self.news_kill_enabled else 'OFF'} "
                f"via BT_NEWS_KILL={_nk_env}")

        if self.news_kill_enabled:
            logger.info(
                f"News kill rules: vol>={self.nkr_max_avg_vol/1e6:.0f}M, "
                f"price<${self.nkr_min_price:.0f}, float>={self.nkr_max_float/1e6:.0f}M "
                f"(catalyst_exemption={self.news_kill_catalyst_exemption})")

    _ET = pytz.timezone('US/Eastern')

    def _get_macd_zone_multiplier(
        self, symbol: str, bars: pd.DataFrame, entry_bar_idx: int,
        entry_price: float, intraday_change_pct: float = 0.0,
    ) -> float:
        """
        Compute MACD zone risk multiplier at entry time (tier-aware).

        Uses warmed-up MACD histogram (prev-day bars prepended) to determine
        which zone the trade falls in, then selects the multiplier from the
        tier bucket based on intraday_change_pct (A-tier ≥20%, Extras 10-20%).

        Args:
            symbol: Stock symbol
            bars: Full day's 1-min bars
            entry_bar_idx: Index of entry bar
            entry_price: Fill price at entry
            intraday_change_pct: max intraday % gain at entry time (0.0 means
                "unknown" → defaults to A-tier multipliers for back-compat)

        Returns:
            Risk multiplier: 0.0 = skip (dead zone OR Extras-tier normal), else
            tier-specific multiplier for the identified zone.
        """
        from trading.indicators import macd_histogram
        from trading.macd_tier_helpers import select_tier_multipliers

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

        # Tier-aware multiplier selection — single source of truth shared
        # with trading/trading_engine.py via trading.macd_tier_helpers.
        strong_pos_mult, strong_neg_mult, normal_mult, tier = \
            select_tier_multipliers(
                intraday_change_pct,
                self.macd_strong_pos_multiplier,
                self.macd_strong_neg_multiplier,
                self.macd_normal_multiplier,
                self.macd_extras_strong_pos_multiplier,
                self.macd_extras_strong_neg_multiplier,
                self.macd_extras_normal_multiplier,
            )

        # Determine zone (dead / strong neg / strong pos / normal)
        if self.macd_dead_zone_min <= macd_pct <= self.macd_dead_zone_max:
            logger.info(
                f"  MACD ZONE SKIP (dead): histogram {hist_val:.4f} = {macd_pct:.2f}% "
                f"(dead zone [{self.macd_dead_zone_min}%, {self.macd_dead_zone_max}%])"
            )
            return 0.0
        elif macd_pct < self.macd_strong_neg_threshold:
            logger.info(
                f"  MACD zone strong neg ({macd_pct:.2f}%) tier={tier} → {strong_neg_mult}x"
            )
            return strong_neg_mult
        elif macd_pct > self.macd_strong_pos_threshold:
            logger.info(
                f"  MACD zone strong pos ({macd_pct:.2f}%) tier={tier} → {strong_pos_mult}x"
            )
            return strong_pos_mult
        else:
            # Normal zone. For Extras tier under S2-max, normal_mult=0.0 →
            # trade is skipped (caller checks `zone_mult == 0.0`). Log at
            # INFO so live monitor catches it (matches dead-zone visibility).
            if normal_mult == 0.0:
                logger.info(
                    f"  MACD ZONE SKIP (Extras-tier normal): {macd_pct:.2f}% → 0.0x "
                    f"(Extras-tier MACD-neutral filtered per S2-max)"
                )
            else:
                logger.info(
                    f"  MACD zone normal ({macd_pct:.2f}%) tier={tier} → {normal_mult}x"
                )
            return normal_mult

    def set_db_path(self, db_path: str) -> None:
        """Set DB path for SPY bar loading (quality filter)."""
        self._db_path = db_path

    def _lookup_marginability(self, symbol: str) -> Optional[bool]:
        """BT-LIVE parity: read marginability from universe table.

        Returns True (marginable), False (not marginable, downgrade
        risk_tier), or None (unknown — fail open). Memoized per
        BacktestRunner instance + lazy-opens the cache DB once.

        Can be overridden by setting `self.db` to a Database instance
        (preferred) — when set, uses Database.get_marginability instead
        of opening a raw sqlite3 connection.
        """
        if not hasattr(self, '_marginability_cache'):
            self._marginability_cache = {}
        if symbol in self._marginability_cache:
            return self._marginability_cache[symbol]

        result: Optional[bool] = None
        # Prefer the Database wrapper when available (handles errors etc).
        if getattr(self, 'db', None) is not None:
            try:
                result = self.db.get_marginability(symbol)
            except Exception as e:
                logger.warning(
                    f"BT marginability lookup via Database failed for "
                    f"{symbol}: {e} — failing open"
                )
                result = None
        else:
            # Fallback: raw sqlite3 read off _db_path. Some research call
            # sites construct a BacktestRunner without a Database — keep
            # them working with a direct query.
            db_path = getattr(self, '_db_path', None)
            if db_path:
                try:
                    import sqlite3 as _sql
                    con = _sql.connect(db_path)
                    row = con.execute(
                        "SELECT is_marginable FROM universe WHERE symbol = ?",
                        (symbol,),
                    ).fetchone()
                    con.close()
                    if row is not None and row[0] is not None:
                        result = bool(row[0])
                except Exception as e:
                    logger.warning(
                        f"BT marginability raw lookup failed for "
                        f"{symbol}: {e} — failing open"
                    )
        self._marginability_cache[symbol] = result
        return result

    def _compute_vwap(self, bars: pd.DataFrame, up_to_idx: int) -> Optional[float]:
        """Compute VWAP from bars[0:up_to_idx+1]. All bars are complete — no look-ahead."""
        if up_to_idx < 0:
            return None
        slice_end = min(up_to_idx + 1, len(bars))
        highs = bars['high'].iloc[:slice_end].values
        lows = bars['low'].iloc[:slice_end].values
        closes = bars['close'].iloc[:slice_end].values
        volumes = bars['volume'].iloc[:slice_end].values
        cum_vol = volumes.sum()
        if cum_vol <= 0:
            return None
        typical_prices = (highs + lows + closes) / 3.0
        return float((typical_prices * volumes).sum() / cum_vol)

    def _get_regime_for_date(self, date_str: str) -> str:
        """Return the Phase 1.4b regime label ('A'/'B'/'C1'/'C2'/'unknown')
        for a trading date, using SPY features from the PREVIOUS session.

        Lazily builds the full {date → regime} lookup on first call from
        SPY daily bars in the SQLite cache. Returns 'disabled' when the
        feature flag is off (caller treats as multiplier=1.0, no skip).
        """
        if not self.regime_sizing_enabled:
            return 'disabled'
        if not self._regime_lookup_built:
            import sqlite3
            from trading.regime_helpers import build_regime_lookup

            db_path = self._db_path or "data/cache.db"
            try:
                conn = sqlite3.connect(db_path)
                spy = pd.read_sql_query(
                    "SELECT bar_date, close FROM daily_bars "
                    "WHERE symbol='SPY' ORDER BY bar_date",
                    conn,
                )
                conn.close()
            except Exception as exc:
                logger.error(
                    f"Regime sizing: SPY bar load from {db_path} failed "
                    f"({exc!r}) — disabling regime sizing for this run."
                )
                self.regime_sizing_enabled = False
                self._regime_lookup_built = True
                return 'disabled'

            if spy is None or spy.empty:
                logger.warning(
                    "Regime sizing enabled but daily_bars has no SPY rows — "
                    "disabling regime sizing for this run."
                )
                self.regime_sizing_enabled = False
                self._regime_lookup_built = True
                return 'disabled'

            spy['bar_date'] = pd.to_datetime(spy['bar_date'])
            self._regime_by_date = build_regime_lookup(
                spy,
                vol_threshold_pct=self.regime_vol_threshold,
                slope_threshold_pct=self.regime_slope_threshold,
            )
            self._regime_lookup_built = True
            logger.info(
                f"Regime sizing: classified {len(self._regime_by_date)} "
                f"SPY trading days (lookup built from {len(spy)} bars)."
            )
        return self._regime_by_date.get(date_str, 'unknown')

    def _load_spy_bars(self, trade_date: str) -> Optional[pd.DataFrame]:
        """Load SPY 1-min bars for a date, with caching."""
        if trade_date in self._spy_bars_cache:
            return self._spy_bars_cache[trade_date]

        db_path = self._db_path
        if not db_path:
            # Try default path
            db_path = "data/cache.db"

        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(
                "SELECT timestamp, open, high, low, close, volume FROM intraday_bars_1min "
                "WHERE symbol='SPY' AND DATE(timestamp)=? ORDER BY timestamp",
                conn, params=(trade_date,)
            )
            conn.close()
            if df.empty:
                self._spy_bars_cache[trade_date] = None
                return None
            self._spy_bars_cache[trade_date] = df
            return df
        except Exception as e:
            logger.debug(f"Failed to load SPY bars for {trade_date}: {e}")
            self._spy_bars_cache[trade_date] = None
            return None

    def _compute_qf_features(
        self, symbol: str, bars: pd.DataFrame, setup, bar_idx: int,
        plan, prev_close: Optional[float],
    ) -> dict:
        """Compute quality filter features at setup detection time.

        Always runs (even when QF disabled) so features are stored in the cache.
        All values are known at setup detection — no look-ahead.
        """
        breakout_level = setup.breakout_level
        features = {}

        # 1. VWAP distance
        vwap = self._compute_vwap(bars, bar_idx)
        if vwap and vwap > 0:
            features['qf_vwap_dist_pct'] = round((breakout_level - vwap) / vwap * 100, 2)
        else:
            features['qf_vwap_dist_pct'] = None

        # 2. Gap info
        if prev_close and prev_close > 0:
            open_price = float(bars.iloc[0]['open'])
            features['qf_gap_pct'] = round((open_price - prev_close) / prev_close * 100, 2)
            features['qf_gap_fading'] = (
                features['qf_gap_pct'] >= self.qf_gap_fade_threshold
                and breakout_level < open_price
            )
        else:
            features['qf_gap_pct'] = None
            features['qf_gap_fading'] = False

        # 3. SPY return at setup time
        features['qf_spy_return_pct'] = None
        trade_date = None
        try:
            ts = bars.iloc[0].get('timestamp') if hasattr(bars.iloc[0], 'get') else bars.iloc[0].name
            trade_date = str(ts)[:10]
        except Exception:
            pass

        if trade_date:
            spy_bars = self._load_spy_bars(trade_date)
            if spy_bars is not None and len(spy_bars) > 1:
                spy_open = float(spy_bars.iloc[0]['open'])
                try:
                    stock_ts = bars.iloc[bar_idx].get('timestamp', bars.iloc[bar_idx].name)
                    stock_ts_str = str(stock_ts)[:19]
                    spy_ts_strs = spy_bars['timestamp'].astype(str).str[:19]
                    spy_mask = spy_ts_strs <= stock_ts_str
                    if spy_mask.any():
                        spy_close = float(spy_bars.loc[spy_mask, 'close'].iloc[-1])
                    else:
                        spy_close = float(spy_bars.iloc[0]['close'])
                except Exception:
                    spy_idx = min(max(0, bar_idx - 1), len(spy_bars) - 1)
                    spy_close = float(spy_bars.iloc[spy_idx]['close'])
                if spy_open > 0:
                    features['qf_spy_return_pct'] = round(
                        (spy_close - spy_open) / spy_open * 100, 3)

        # 4. Pole quality
        features['qf_pole_bars'] = setup.pole_end_idx - setup.pole_start_idx
        features['qf_pole_gain_pct'] = round(setup.pole_gain_pct, 2)

        return features

    def _evaluate_qf(self, features: dict) -> tuple:
        """Evaluate quality filter thresholds against pre-computed features.

        Returns (pass: bool, reason: str). If pass=False, skip this setup.
        """
        # 1. VWAP overextension
        vwap_dist = features.get('qf_vwap_dist_pct')
        if vwap_dist is not None and vwap_dist > self.qf_max_vwap_dist:
            return (False, f"VWAP +{vwap_dist:.1f}% > {self.qf_max_vwap_dist}% (overextended)")

        # 2. Gap fading
        if features.get('qf_gap_fading'):
            gap = features.get('qf_gap_pct', 0)
            return (False, f"gap_fade: gap +{gap:.1f}%")

        # 3. SPY down
        spy_ret = features.get('qf_spy_return_pct')
        if spy_ret is not None and spy_ret < self.qf_min_spy_return:
            return (False, f"SPY {spy_ret:+.2f}% < {self.qf_min_spy_return}% (risk-off)")

        # 4. Slow weak pole
        pole_bars = features.get('qf_pole_bars', 0)
        pole_gain = features.get('qf_pole_gain_pct', 99)
        if pole_bars > self.qf_slow_pole_max_bars and pole_gain < self.qf_slow_pole_min_gain:
            return (False, f"slow_pole: {pole_bars} bars, {pole_gain:.1f}% gain")

        return (True, "")

    def _compute_conviction_score_setup(
        self, setup, spy_3d_range: Optional[float], *,
        vwap_dist_pct: float = 0.0,
        gap_fading: bool = False,
        gap_pct: float = 0.0,
        intraday_range_pct: float = 0.0,
        v_reversal_enabled: bool = False,
        v_reversal_bonus: float = 0.4,
        v_reversal_gap_pct_max: float = 0.0,
        v_reversal_intraday_range_min: float = 20.0,
        v_reversal_pole_gain_min: float = 5.0,
        return_breakdown: bool = False,
    ):
        """Compute conviction score at setup detection time.

        Returns a multiplier (0.25 to 3.0) that scales position size.
        Uses only features known at setup detection — no look-ahead.

        Args:
            setup: BullFlagSetup object
            spy_3d_range: SPY 3-day avg daily range %, or None when SPY data
                is missing/stale (per `trading.spy_regime`). None is mapped
                to the worst-case -0.5 penalty in rule 4 — same as low-vol
                regime — so degraded data fails closed rather than silently
                neutralizing the rule.
            vwap_dist_pct: (breakout_level - vwap)/vwap * 100, computed up to
                setup bar. Defaults to 0.0 (rule 7 silent) for back-compat.
                NEW in V2_clean (shipped 2026-04-15).
            gap_fading: True if today gapped up >= qf_gap_fade_threshold%
                from prev_close AND breakout_level is below today's open.
                Defaults to False (rule 8 silent) for back-compat.
                NEW in V2_clean.
            gap_pct: (today_open - prev_close) / prev_close * 100. Used by
                Rule 9 (V-reversal) — fires only on gap-down setups.
            intraday_range_pct: (day_high - day_low) / day_low * 100 up to
                setup bar. Used by Rule 9 — V-reversal requires >= 20% range.
            return_breakdown: If True, return (final_score, breakdown_dict).

        Returns:
            float (when return_breakdown=False) — the position multiplier
            tuple (float, dict) — when return_breakdown=True

        V2_clean rules 7+8 added 2026-04-15 from walk-forward research:
        canonical 16mo +$52K (+15.5%), mean OOS test +$28K, robust on
        all 3 splits. Rule 6 (daily_range_pct) was rejected — look-ahead.

        Rule 9 (V-reversal bonus) added 2026-04-17 from fat-tail analysis:
        top-10 2025 winners have median gap_pct=-1.09% and cleared 20%
        intraday range — oversold V-reversals, not gap-ups. Bonus captures
        this distinctive pattern within the existing conviction envelope
        (clamp stays [0.25, 3.0]).
        """
        score = 1.0
        breakdown = {}

        # 1. Pole gain sweet spot (4.5-9%)
        pg = setup.pole_gain_pct
        pg_contrib = 0.3 if 4.5 <= pg <= 9.0 else 0.0
        score += pg_contrib
        breakdown['pole_gain'] = pg_contrib

        # 2. Flag tightness (tight < 30% = good, loose > 50% = bad)
        ft_contrib = 0.0
        pole_height = setup.pole_high - setup.pole_low
        if pole_height > 0:
            flag_range = setup.flag_high - setup.flag_low
            tightness = flag_range / pole_height * 100
            if tightness < 30:
                ft_contrib = 0.3
            elif tightness > 50:
                ft_contrib = -0.3
        score += ft_contrib
        breakdown['flag_tightness'] = ft_contrib

        # 3. Volume ratio pole/flag (>1.7x = buying conviction)
        vr_contrib = 0.0
        if setup.avg_flag_volume > 0:
            vol_ratio = setup.avg_pole_volume / setup.avg_flag_volume
            if vol_ratio > 1.7:
                vr_contrib = 0.3
        score += vr_contrib
        breakdown['vol_ratio'] = vr_contrib

        # 4. SPY 3d range regime.
        # None = data missing/stale per spy_regime helper. Treat as worst case
        # (same penalty as low-vol regime) — matches live exactly and degrades
        # gracefully when SPY refresh fails. Was a 1.0 sentinel pre-2026-05-02
        # which silently inflated conviction by +0.5 (EAF post-mortem).
        #
        # Ablation hook: BT_SPY_FEATURE_OFF=1 zeros this rule's contribution.
        # Used by study_spy_filter_ablation.py to measure the SPY regime
        # feature's standalone value. Does NOT change production behavior;
        # default-off (no env var → original logic). Live code (trading_engine.
        # py:_compute_conviction_score_setup) is intentionally NOT toggled.
        if os.getenv("BT_SPY_FEATURE_OFF") == "1":
            sr_contrib = 0.0
        elif spy_3d_range is None:
            sr_contrib = -0.5
        elif spy_3d_range > 1.2:
            sr_contrib = 0.3
        elif spy_3d_range < 0.8:
            sr_contrib = -0.5
        else:
            sr_contrib = 0.0
        score += sr_contrib
        breakdown['spy_regime'] = sr_contrib

        # 5. Shallow retracement (< 30%)
        rt_contrib = 0.2 if setup.retracement_pct < 30 else 0.0
        score += rt_contrib
        breakdown['retracement'] = rt_contrib

        # 6. (Rule 6 reserved — daily_range_pct was rejected as look-ahead.)

        # 7. VWAP distance — extension above VWAP signals momentum quality.
        # Walk-forward bucket EV: vwap_dist >= 2 → mean +$1.5K/trade vs <0 → -$1K/tr.
        vw_contrib = 0.2 if vwap_dist_pct >= 2.0 else 0.0
        score += vw_contrib
        breakdown['vwap_dist'] = vw_contrib

        # 8. Gap fading penalty — gap-up that broke down before entry is bearish.
        # Walk-forward: gap_fading=True → -$612/trade test, =False → +$535/tr.
        gf_contrib = -0.3 if gap_fading else 0.0
        score += gf_contrib
        breakdown['gap_fading'] = gf_contrib

        # 9. V-reversal bonus — oversold reversal plays (gap-down + high intraday
        # range + meaningful pole). Feature-flagged; default OFF. When enabled,
        # adds `v_reversal_bonus` to the raw score. Thresholds are configurable.
        # Fat-tail analysis showed top-10 2025 winners have median gap=-1.1%,
        # cleared 20% range — this rule explicitly rewards that shape.
        vr_contrib = 0.0
        if v_reversal_enabled and (
            gap_pct < v_reversal_gap_pct_max
            and intraday_range_pct >= v_reversal_intraday_range_min
            and setup.pole_gain_pct >= v_reversal_pole_gain_min
        ):
            vr_contrib = v_reversal_bonus
        score += vr_contrib
        breakdown['v_reversal'] = vr_contrib

        final = max(0.25, min(3.0, score))
        if return_breakdown:
            breakdown['raw_score'] = score
            breakdown['final_score'] = final
            return final, breakdown
        return final

    def _compute_conviction_score_fill(self, entry_bar_volume: float,
                                        avg_flag_volume: float,
                                        setup_score: float) -> float:
        """Adjust conviction score at fill time based on breakout bar volume.

        Called AFTER buy-stop triggers. Can exit immediately if score drops too low.
        """
        score = setup_score

        # 6. Breakout bar volume vs flag average
        if avg_flag_volume > 0:
            bk_ratio = entry_bar_volume / avg_flag_volume
            if bk_ratio > 1.5:
                score += 0.4  # strong breakout volume
            elif bk_ratio < 0.5:
                score -= 0.5  # weak breakout volume

        return max(0.25, min(3.0, score))

    def _get_spy_3d_range(self, trade_date: str) -> Optional[float]:
        """SPY 3-day avg daily range for `trade_date` from cache.db, or None.

        Wraps `trading.spy_regime.compute_spy_3d_range` (shared with live
        trader so BT and live compute identically). Returns `None` if SPY
        bars are missing or stale; the conviction rule treats `None` as
        the worst-case -0.5 penalty.

        Pre-2026-05-02 this returned a `1.0` sentinel on missing data, which
        the rule maps to `sr_contrib = 0.0` — silently inflating conviction
        by +0.5 vs real data. See post-mortem on EAF 2026-05-01.
        """
        from trading.spy_regime import (
            compute_spy_3d_range,
            is_spy_data_stale,
        )
        from datetime import date as _date

        db_path = self._db_path or "data/cache.db"
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            rows = conn.execute(
                "SELECT bar_date, high, low FROM daily_bars "
                "WHERE symbol='SPY' AND bar_date<? "
                "ORDER BY bar_date DESC LIMIT 3", (trade_date,)
            ).fetchall()
            conn.close()
        except Exception as e:
            logger.error(
                "_get_spy_3d_range: DB query failed for trade_date=%s: %s "
                "— returning None", trade_date, e,
            )
            return None

        if not rows:
            logger.warning(
                "_get_spy_3d_range: no SPY bars in cache.db before %s "
                "— returning None (rebuild SPY cache or check universe cron)",
                trade_date,
            )
            return None

        # Staleness guard: rows[0] is the most-recent bar (DESC order).
        try:
            ref = _date.fromisoformat(trade_date)
        except (ValueError, TypeError) as e:
            logger.error(
                "_get_spy_3d_range: invalid trade_date=%r (%s) — returning None",
                trade_date, e,
            )
            return None
        latest = rows[0][0]
        if isinstance(latest, str):
            try:
                latest = _date.fromisoformat(latest)
            except (ValueError, TypeError):
                latest = None
        if is_spy_data_stale(latest, ref):
            return None  # is_spy_data_stale already logged ERROR

        # DESC -> ASC so compute_spy_3d_range's "last 3" semantics match.
        bars = [{'high': r[1], 'low': r[2]} for r in reversed(rows)]
        return compute_spy_3d_range(bars)

    _REAL_CATS = {'FDA_CLINICAL', 'EARNINGS', 'CONTRACT_DEAL', 'CONTRACT',
                  'MA', 'ANALYST', 'PRODUCT', 'PRODUCT_LAUNCH',
                  'MGMT', 'MANAGEMENT', 'SEC_FILING', 'CRYPTO'}

    @staticmethod
    def _classify_headline(h: str) -> str:
        """Classify a news headline into category using regex (fast, no LLM)."""
        import re
        h = h.lower()
        if re.search(r'fda|phase [123]|clinical|trial|drug|therapy|approv|orphan|ind |nda|biologics', h): return 'FDA_CLINICAL'
        if re.search(r'earn|revenue|quarter|q[1-4]|eps|guidance|beat|miss|fiscal', h): return 'EARNINGS'
        if re.search(r'contract|deal|agreement|partner|collaborat|licens|amend|award', h): return 'CONTRACT_DEAL'
        if re.search(r'acqui|merge|buyout|takeover', h): return 'MA'
        if re.search(r'analyst|upgrade|downgrade|price target|initiat.*coverage', h): return 'ANALYST'
        if re.search(r'launch|new product|expansion|patent|initiative', h): return 'PRODUCT'
        if re.search(r'insider|ceo|cfo|director|appoint|resign|hire', h): return 'MGMT'
        if re.search(r'offering|ipo|shelf|registration|prospectus', h): return 'SEC_FILING'
        if re.search(r'why is|why are|stocks? moving|here are \d+|top \d+ stocks', h): return 'GARBAGE_RECAP'
        if re.search(r'bitcoin|crypto|blockchain|mining', h): return 'CRYPTO'
        return 'OTHER'

    def _ensure_news_table(self, conn) -> None:
        """Create news_history table if it doesn't exist."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_history (
                symbol VARCHAR(10) NOT NULL,
                trade_date DATE NOT NULL,
                article_time VARCHAR(30),
                headline TEXT,
                source VARCHAR(50),
                category VARCHAR(30),
                is_catalyst INTEGER,
                UNIQUE(symbol, trade_date, headline)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_history_sym_date
                ON news_history(symbol, trade_date)
        """)

    def _fetch_news_for_date(self, symbol: str, trade_date: str, conn) -> bool:
        """Fetch news from Alpaca API and store in news_history. Returns has_catalyst."""
        import os
        from config import Config
        cfg = Config._load_yaml_only()

        api_key = os.environ.get('APCA_API_KEY_ID', cfg.get('alpaca', {}).get('api_key', ''))
        api_secret = os.environ.get('APCA_API_SECRET_KEY', cfg.get('alpaca', {}).get('api_secret', ''))
        if not api_key or not api_secret:
            # Try Config object
            try:
                c = Config()
                api_key = api_key or c.alpaca_api_key
                api_secret = api_secret or c.alpaca_api_secret
            except Exception:
                pass
        if not api_key:
            return False

        headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': api_secret,
        }

        # Time window: prev day 4PM ET (21:00 UTC) to trade_date 3PM ET (20:00 UTC)
        from datetime import timedelta
        prev_date = (datetime.strptime(trade_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        start = f"{prev_date}T21:00:00Z"
        end = f"{trade_date}T20:00:00Z"

        try:
            import requests
            url = (f"https://data.alpaca.markets/v1beta1/news?"
                   f"symbols={symbol}&start={start}&end={end}&limit=10&sort=desc")
            resp = requests.get(url, headers=headers, timeout=10)
            articles = resp.json().get('news', [])
        except Exception as e:
            logger.debug(f"{symbol} {trade_date}: news fetch failed: {e}")
            return False

        self._ensure_news_table(conn)
        has_catalyst = False

        if not articles:
            conn.execute(
                "INSERT OR IGNORE INTO news_history (symbol, trade_date, headline, category, is_catalyst) "
                "VALUES (?, ?, '', 'NO_NEWS', 0)",
                (symbol, trade_date)
            )
            conn.commit()
            return False

        for a in articles:
            headline = (a.get('headline') or '')[:500]
            article_time = (a.get('created_at') or '')[:30]
            source = (a.get('source') or '')[:50]
            cat = self._classify_headline(headline)
            is_cat = 1 if cat in self._REAL_CATS else 0
            if is_cat:
                has_catalyst = True
            conn.execute(
                "INSERT OR IGNORE INTO news_history "
                "(symbol, trade_date, article_time, headline, source, category, is_catalyst) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (symbol, trade_date, article_time, headline, source, cat, is_cat)
            )

        conn.commit()
        return has_catalyst

    def _has_real_catalyst(self, symbol: str, trade_date: str) -> bool:
        """Check for a real news catalyst on this symbol/date.

        Default (production): regex classification via news_history, auto-
        fetching from the Alpaca News API on a cache miss. When
        BT_NEWS_CLASSIFIER is set, reads the precomputed verdict from
        data/news_ab.db instead — the 3-arm news-classifier A/B (see
        scripts/precompute_news_ab.py).
        """
        cache_key = f"{symbol}_{trade_date}"
        if cache_key in self._news_cache:
            return self._news_cache[cache_key]

        if self._news_ab_mode:
            result = self._news_ab_catalyst(symbol, trade_date)
        else:
            result = self._has_real_catalyst_regex(symbol, trade_date)

        self._news_cache[cache_key] = result
        return result

    def _news_ab_catalyst(self, symbol: str, trade_date: str) -> bool:
        """Read the precomputed catalyst verdict for the active A/B classifier.

        Falls back to the regex path (with a WARNING) if the (symbol, date) was
        never precomputed — that signals scripts/precompute_news_ab.py must be
        re-run to cover the row; it must NOT be silently treated as 'no news'.
        """
        if self._news_ab_store is None:
            from trading.news_ab import NewsABStore
            self._news_ab_store = NewsABStore()
        verdict = self._news_ab_store.get_verdict(
            symbol, trade_date, self._news_ab_mode)
        if verdict is None:
            logger.warning(
                f"{symbol} {trade_date}: no precomputed news_ab verdict for "
                f"classifier='{self._news_ab_mode}' — falling back to regex. "
                f"Re-run scripts/precompute_news_ab.py to cover this row.")
            return self._has_real_catalyst_regex(symbol, trade_date)
        return verdict

    def _has_real_catalyst_regex(self, symbol: str, trade_date: str) -> bool:
        """Production catalyst check: regex classification via the news_history
        table, auto-fetching from the Alpaca News API on a cache miss.
        """
        result = False
        db_path = self._db_path or "data/cache.db"
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            self._ensure_news_table(conn)
            rows = conn.execute(
                "SELECT category FROM news_history WHERE symbol=? AND trade_date=?",
                (symbol, trade_date)).fetchall()

            if rows:
                # Already cached — check categories
                for r in rows:
                    if r[0] in self._REAL_CATS:
                        result = True
                        break
            else:
                # Not cached — fetch from API and classify
                result = self._fetch_news_for_date(symbol, trade_date, conn)

            conn.close()
        except Exception as e:
            logger.warning(f"{symbol} {trade_date}: news lookup failed: {e}")
        return result

    def _check_news_kill(self, symbol: str, bars: pd.DataFrame, setup,
                          plan, avg_daily_volume: int,
                          float_shares: int = 0) -> tuple:
        """Check if a trade should be killed by the news-kill segment gate.

        Delegates to the shared trading.news_kill_guard.news_kill_decision so
        BT and the live engine cannot drift. Returns (should_trade, reason).
        """
        if not self.news_kill_enabled:
            return (True, "")

        # Get trade date
        trade_date = None
        try:
            trade_date = str(bars.iloc[0].get('timestamp', bars.iloc[0].name))[:10]
        except Exception:
            pass

        if not trade_date:
            return (True, "")

        # Catalyst status is only consulted when the exemption is enabled —
        # skip the (regex-classifier) news lookup entirely when it is off.
        has_cat = (self.news_kill_catalyst_exemption
                   and self._has_real_catalyst(symbol, trade_date))
        return news_kill_decision(
            has_catalyst=has_cat,
            catalyst_exemption=self.news_kill_catalyst_exemption,
            avg_vol=avg_daily_volume or 0,
            entry_price=plan.entry_price,
            float_shares=float_shares or 0,
            pole_gain=setup.pole_gain_pct,
            max_avg_vol=self.nkr_max_avg_vol,
            min_price=self.nkr_min_price,
            max_float=self.nkr_max_float,
        )

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
        qualification_pct_override: Optional[float] = None,
        premarket_extremes: Optional[tuple] = None,
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
                                       prev_close=prev_close,
                                       qualification_pct_override=qualification_pct_override,
                                       premarket_extremes=premarket_extremes)
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
        qualification_pct_override: Optional[float] = None,
        premarket_extremes: Optional[tuple] = None,
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
        if qualification_pct_override is not None:
            qualification_pct = qualification_pct_override / 100.0
        else:
            from config import Config
            _cfg = Config._load_yaml_only()
            qualification_pct = float(
                _cfg.get("scanner", {}).get("intraday_change_pct_min", 20.0)
            ) / 100.0  # Convert 20.0 → 0.20
        qualified = prev_close is None or prev_close <= 0
        price_qualified = qualified  # Track price and volume independently
        volume_qualified = qualified or (
            self.min_cum_dollar_vol <= 0 and self.min_cum_shares <= 0
            and self.min_relative_vol_rate <= 0)  # No volume gate = auto-qualified
        qualification_bar = 0

        logger.info(
            f"{symbol}: Scanning {len(bars)} bars for setups (realistic mode)"
            f"{f', qualification at +{qualification_pct:.0%} from ${prev_close:.2f}' if not qualified else ''}..."
        )

        running_high = 0.0
        running_low = float('inf')
        # Premarket-extremes seed (parity with live's get_current_bars
        # behavior: when called early in the session, the latest completed
        # 15-min bar is a premarket bar, so live's _day_highs/_day_lows
        # carry premarket high/low forward into RTH qualification. BT
        # without this seed only sees RTH bars and qualifies later — or
        # not at all — for stocks that gapped/extended in premarket.
        # See research notes: 5/5 INTT and 5/7 PN both qualified live
        # via premarket-derived range_pct.
        if premarket_extremes is not None:
            pm_high, pm_low = premarket_extremes
            if pm_high is not None and pm_high > 0:
                running_high = max(running_high, float(pm_high))
            if pm_low is not None and pm_low > 0 and pm_low != float('inf'):
                running_low = min(running_low, float(pm_low))
            logger.info(
                f"{symbol}: Premarket seed — pm_high=${pm_high}, "
                f"pm_low=${pm_low} (running_high=${running_high:.2f}, "
                f"running_low=${running_low if running_low != float('inf') else 'inf'})"
            )

        # Seed running extremes from early bars (before scan loop starts)
        # so V-reversal qualification doesn't miss the opening range
        for j in range(min(self.MIN_BARS_FOR_SETUP - 1, len(bars))):
            running_high = max(running_high, bars.iloc[j]['high'])
            running_low = min(running_low, bars.iloc[j]['low'])

        for i in range(self.MIN_BARS_FOR_SETUP - 1, last_end):
            # Track intraday extremes for V-reversal detection
            if not qualified:
                running_high = max(running_high, bars.iloc[i]['high'])
                running_low = min(running_low, bars.iloc[i]['low'])

            # Real-time qualification check: price and volume tracked INDEPENDENTLY.
            # Price qualifies once (stays qualified). Volume qualifies once (stays qualified).
            # Scanning starts when BOTH have been met (at any point, not same bar).
            if not qualified:
                # Price check
                if not price_qualified:
                    bar_high = bars.iloc[i]['high']
                    move = (bar_high - prev_close) / prev_close
                    range_move = (running_high - running_low) / running_low if running_low > 0 else 0
                    if move >= qualification_pct or range_move >= qualification_pct:
                        price_qualified = True

                # Volume check (independent of price)
                if not volume_qualified:
                    cum_vol = int(bars.iloc[:i+1]['volume'].sum())
                    minutes = i + 1
                    vol_ok = True
                    if self.min_cum_dollar_vol > 0:
                        cum_dv = float((bars.iloc[:i+1]['close'] * bars.iloc[:i+1]['volume']).sum())
                        if cum_dv < self.min_cum_dollar_vol:
                            vol_ok = False
                    if self.min_cum_shares > 0 and cum_vol < self.min_cum_shares:
                        vol_ok = False
                    if self.min_relative_vol_rate > 0 and self.avg_daily_volume > 0:
                        rate = cum_vol / minutes
                        expected = self.avg_daily_volume / 390
                        if expected > 0 and rate / expected < self.min_relative_vol_rate:
                            vol_ok = False
                    if vol_ok:
                        volume_qualified = True

                # Both met?
                if price_qualified and volume_qualified:
                    qualified = True
                    qualification_bar = i
                    bar_high = bars.iloc[i]['high']
                    move = (bar_high - prev_close) / prev_close
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

                    # Post-fill exit: in calm markets (SPY 3d range < 0.8%),
                    # exit immediately if breakout volume is weak (< 1.0x flag avg).
                    # Cost: ~0.8% slippage. Saves: ~$2K avg stop loss.
                    # Experiment harness (2026-05-08 IREZ post-mortem):
                    #   BT_POST_FILL_GATE_DISABLE=1 -> skip the kill but still
                    #     record bk_ratio_at_fill / spy_3d_at_fill on the trade
                    #     so post-hoc analysis can replay alternative gate logic.
                    _gate_disabled = (
                        bool(os.environ.get('BT_POST_FILL_GATE_DISABLE'))
                        or not self.post_fill_gate_enabled
                    )
                    _spy_thresh = self.post_fill_gate_spy_threshold
                    _bk_thresh = self.post_fill_gate_bk_threshold
                    _bk_vol_post = 0.0
                    _afv_post = 0.0
                    _bk_ratio_at_fill = None
                    _spy_3d_at_fill = None
                    if self.conviction_enabled:
                        _bk_vol_post = float(bar.get('volume', 0) if hasattr(bar, 'get') else bar['volume'])
                        _afv_post = float(pending_order.setup.avg_flag_volume) if hasattr(pending_order.setup, 'avg_flag_volume') else 0
                        _bk_ratio_at_fill = _bk_vol_post / _afv_post if _afv_post > 0 else 99
                        _trade_date = None
                        try:
                            _trade_date = str(bars.iloc[0].get('timestamp', bars.iloc[0].name))[:10]
                        except Exception:
                            pass
                        _spy_3d_at_fill = self._get_spy_3d_range(_trade_date) if _trade_date else None
                        # None (missing/stale SPY) treated same as low-vol regime —
                        # consistent with the conviction rule's worst-case mapping.
                        _spy_hostile = _spy_3d_at_fill is None or _spy_3d_at_fill < _spy_thresh
                        # Backwards-compat aliases for the kill branch below
                        _bk_vol = _bk_vol_post
                        _afv = _afv_post
                        _bk_ratio = _bk_ratio_at_fill
                        _spy_3d = _spy_3d_at_fill
                        if not _gate_disabled and _spy_hostile and _bk_ratio < _bk_thresh:
                            # Immediate exit — record slippage loss
                            exit_price = fill_price * (1 - self.exit_slippage_pct)
                            slippage_pnl = (exit_price - fill_price) * plan.shares
                            logger.info(
                                f"  POST-FILL EXIT: SPY 3d {_spy_3d:.2f}% + bk_vol {_bk_ratio:.1f}x "
                                f"→ immediate exit, P&L ${slippage_pnl:.0f}")
                            # Create a minimal trade record for the cache
                            trade = SimulatedTrade(
                                symbol=plan.symbol,
                                entry_time=bars.iloc[i].get('timestamp', bars.iloc[i].name),
                                entry_price=fill_price,
                                stop_loss=plan.stop_loss_price,
                                take_profit=plan.take_profit_price,
                                shares=plan.shares,
                                exit_time=bars.iloc[min(i+1, len(bars)-1)].get('timestamp', bars.iloc[min(i+1, len(bars)-1)].name),
                                exit_price=exit_price,
                                exit_reason=ExitReason.POST_FILL_EXIT.value,
                                pnl=slippage_pnl,
                                pnl_pct=(exit_price - fill_price) / fill_price * 100,
                                bars_held=1,
                                plan=plan,
                            )
                            if hasattr(pending_order, '_qf_features'):
                                trade._qf_features = pending_order._qf_features
                            trade.conviction_mult = getattr(pending_order, '_conviction_mult', 1.0)
                            trade.macd_zone_mult = getattr(pending_order, '_macd_zone_mult', 1.0)
                            trade.intraday_change_at_entry = getattr(pending_order, '_intraday_change_at_entry', None)
                            _bd = getattr(pending_order, '_conv_breakdown', None)
                            if _bd:
                                trade.conv_pole_gain = _bd.get('pole_gain', 0.0)
                                trade.conv_flag_tightness = _bd.get('flag_tightness', 0.0)
                                trade.conv_vol_ratio = _bd.get('vol_ratio', 0.0)
                                trade.conv_spy_regime = _bd.get('spy_regime', 0.0)
                                trade.conv_retracement = _bd.get('retracement', 0.0)
                                trade.conv_vwap_dist = _bd.get('vwap_dist', 0.0)
                                trade.conv_gap_fading = _bd.get('gap_fading', 0.0)
                                trade.conv_raw_score = _bd.get('raw_score', 1.0)
                            # `or 0.0` collapses None (missing/stale SPY data) into 0.0 —
                            # the dataclass field is `float`, and downstream CSV writers
                            # format with `{:.3f}` which crashes on None. The conviction
                            # breakdown's `conv_spy_regime` field already records the
                            # missing-data state via -0.5 contribution.
                            trade.spy_3d_range = getattr(pending_order, '_spy_3d_range', 0.0) or 0.0
                            # Experiment columns (always populated when conviction_enabled)
                            trade.bk_ratio_at_fill = _bk_ratio_at_fill
                            trade.spy_3d_at_fill = _spy_3d_at_fill
                            result.trades_simulated.append(trade)
                            pending_order = None
                            continue

                    trade = self.simulator.simulate(
                        plan, bars, i, entry_price_override=fill_price
                    )
                    # Experiment columns: gate inputs at fill time, populated
                    # for both kill and natural-exit branches so post-hoc gate
                    # variants can replay decisions.
                    trade.bk_ratio_at_fill = _bk_ratio_at_fill
                    trade.spy_3d_at_fill = _spy_3d_at_fill
                    # Propagate QF features and conviction from pending order to trade
                    if hasattr(pending_order, '_qf_features'):
                        trade._qf_features = pending_order._qf_features
                    trade.conviction_mult = getattr(pending_order, '_conviction_mult', 1.0)
                    trade.macd_zone_mult = getattr(pending_order, '_macd_zone_mult', 1.0)
                    trade.intraday_change_at_entry = getattr(pending_order, '_intraday_change_at_entry', None)
                    _bd = getattr(pending_order, '_conv_breakdown', None)
                    if _bd:
                        trade.conv_pole_gain = _bd.get('pole_gain', 0.0)
                        trade.conv_flag_tightness = _bd.get('flag_tightness', 0.0)
                        trade.conv_vol_ratio = _bd.get('vol_ratio', 0.0)
                        trade.conv_spy_regime = _bd.get('spy_regime', 0.0)
                        trade.conv_retracement = _bd.get('retracement', 0.0)
                        trade.conv_vwap_dist = _bd.get('vwap_dist', 0.0)
                        trade.conv_gap_fading = _bd.get('gap_fading', 0.0)
                        trade.conv_raw_score = _bd.get('raw_score', 1.0)
                    # `or 0.0` collapses None (missing/stale SPY data) into 0.0 —
                    # see comment on the parallel assignment above for rationale.
                    trade.spy_3d_range = getattr(pending_order, '_spy_3d_range', 0.0) or 0.0
                    # Add post-fill VWAP (at fill bar) — for post-filter analysis
                    fill_vwap = self._compute_vwap(bars, i)
                    if fill_vwap and fill_vwap > 0:
                        trade._qf_features = trade._qf_features.copy() if hasattr(trade, '_qf_features') else {}
                        trade._qf_features['qf_fill_vwap_dist_pct'] = round(
                            (fill_price - fill_vwap) / fill_vwap * 100, 2)
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

                # News kill rules: block no-news trades in loser segments
                if self.news_kill_enabled:
                    _float_shares = 0
                    try:
                        import sqlite3 as _sql
                        _nconn = _sql.connect(self._db_path or "data/cache.db")
                        _fr = _nconn.execute(
                            "SELECT float_shares FROM universe WHERE symbol=? LIMIT 1",
                            (symbol,)).fetchone()
                        _nconn.close()
                        _float_shares = int(_fr[0]) if _fr else 0
                    except Exception:
                        pass
                    nk_pass, nk_reason = self._check_news_kill(
                        symbol, bars, setup, plan,
                        avg_daily_volume=self._current_avg_daily_volume or 0,
                        float_shares=_float_shares)
                    if not nk_pass:
                        logger.info(f"  NEWS KILL: {nk_reason}")
                        continue

                # Filter ordering matches PROD trading_engine.py:
                #   news_kill (above) → conviction → risk_tier → combined_mult
                # Conviction filter runs before risk_tier for parity, even
                # though BT has no API call to save (PROD does).

                # Conviction scoring: compute breakdown once; use on skip-log too.
                conviction_mult = 1.0
                # Defaults for cache emission when conviction disabled (Phase A).
                _conv_brkdn = {
                    'pole_gain': 0.0, 'flag_tightness': 0.0, 'vol_ratio': 0.0,
                    'spy_regime': 0.0, 'retracement': 0.0,
                    'vwap_dist': 0.0, 'gap_fading': 0.0,
                    'raw_score': 1.0, 'final_score': 1.0,
                }
                _spy_3d: Optional[float] = None
                if self.conviction_enabled:
                    _trade_date = None
                    try:
                        _trade_date = str(bars.iloc[0].get('timestamp', bars.iloc[0].name))[:10]
                    except Exception:
                        pass
                    # _get_spy_3d_range returns None on missing/stale data —
                    # the conviction rule maps None to the -0.5 worst-case
                    # penalty (parity with live).
                    _spy_3d = self._get_spy_3d_range(_trade_date) if _trade_date else None
                    # V2_clean rule 7+8 inputs — computed inline at the SAME
                    # bar boundary PROD uses (setup.flag_end_idx) so BT and PROD
                    # produce identical conviction scores. (qf_features below
                    # uses bar_idx=i = flag_end_idx+1 for legacy reasons; we
                    # don't reuse it here because that would diverge from PROD.)
                    _v7_dist = 0.0
                    _g8_fade = False
                    _gap_pct_for_v9 = 0.0
                    _vwap_at_setup = self._compute_vwap(bars, setup.flag_end_idx)
                    if _vwap_at_setup and _vwap_at_setup > 0:
                        _v7_dist = (setup.breakout_level - _vwap_at_setup) / _vwap_at_setup * 100
                    if prev_close and prev_close > 0:
                        _today_open = float(bars.iloc[0]['open'])
                        _gap_pct = (_today_open - prev_close) / prev_close * 100
                        _gap_pct_for_v9 = _gap_pct
                        _g8_fade = bool(
                            _gap_pct >= self.qf_gap_fade_threshold
                            and setup.breakout_level < _today_open
                        )
                    # Rule 9 intraday range (V-reversal). Uses bars up to setup.
                    _day_high = float(bars.iloc[:setup.flag_end_idx + 1]['high'].max())
                    _day_low = float(bars.iloc[:setup.flag_end_idx + 1]['low'].min())
                    _intraday_range_pct = (
                        (_day_high - _day_low) / _day_low * 100
                        if _day_low > 0 else 0.0
                    )
                    conviction_mult, _conv_brkdn = self._compute_conviction_score_setup(
                        setup, _spy_3d,
                        vwap_dist_pct=_v7_dist,
                        gap_fading=_g8_fade,
                        gap_pct=_gap_pct_for_v9,
                        intraday_range_pct=_intraday_range_pct,
                        v_reversal_enabled=self.v_reversal_enabled,
                        v_reversal_bonus=self.v_reversal_bonus,
                        v_reversal_gap_pct_max=self.v_reversal_gap_pct_max,
                        v_reversal_intraday_range_min=self.v_reversal_intraday_range_min,
                        v_reversal_pole_gain_min=self.v_reversal_pole_gain_min,
                        return_breakdown=True)
                    if abs(conviction_mult - 1.0) > 0.05:
                        logger.debug(f"  Conviction score: {conviction_mult:.2f}x")

                    # Conviction filter: skip below threshold (mirrors trading_engine).
                    if (self.conviction_min_threshold > 0
                            and conviction_mult < self.conviction_min_threshold):
                        logger.debug(
                            f"  CONVICTION SKIP: {conviction_mult:.2f} < "
                            f"{self.conviction_min_threshold:.2f} "
                            f"(pole={_conv_brkdn['pole_gain']:+.1f} "
                            f"flag={_conv_brkdn['flag_tightness']:+.1f} "
                            f"vol={_conv_brkdn['vol_ratio']:+.1f} "
                            f"spy={_conv_brkdn['spy_regime']:+.1f} "
                            f"retr={_conv_brkdn['retracement']:+.1f} "
                            f"vwap={_conv_brkdn['vwap_dist']:+.1f} "
                            f"gap={_conv_brkdn['gap_fading']:+.1f})"
                        )
                        continue

                # Marginal-conviction defensive scaling: compute the sizing
                # multiplier separately. conviction_mult stays as-is so that
                # the Stage-2 cache-filter sees the true quality signal; only
                # the applied sizing is scaled.
                sizing_conviction = conviction_mult
                if (self.conviction_marginal_scale_factor < 1.0
                        and conviction_mult < self.conviction_marginal_upper):
                    sizing_conviction = conviction_mult * self.conviction_marginal_scale_factor
                    logger.debug(
                        f"  CONVICTION MARGINAL SCALE: {conviction_mult:.2f} → "
                        f"sizing {sizing_conviction:.2f} "
                        f"(below {self.conviction_marginal_upper}, "
                        f"factor {self.conviction_marginal_scale_factor})"
                    )

                # Risk tier: scale risk on high-conviction setups (same as trading_engine)
                risk_tier_mult = 1.0
                if self.risk_tiers_enabled:
                    ep = plan.entry_price
                    av = self.avg_daily_volume or 0
                    for tier in self.risk_tiers:
                        if (tier['min_price'] <= ep < tier['max_price'] and
                                tier['min_volume'] <= av <= tier['max_volume']):
                            risk_tier_mult = tier['multiplier']
                            break

                    # BT-LIVE parity: mirror LIVE's marginability downgrade.
                    # LIVE (trading_engine.py:2812) calls alpaca.is_marginable
                    # before applying risk_tier_mult > 1.0; if False, downgrades
                    # to 1.0x. BT can't call Alpaca during a backtest, so we
                    # read the persisted value from the universe table — LIVE
                    # populates it on first observation per symbol.
                    # NULL = unknown → fail open (full risk_tier, current
                    # behavior). False = downgrade to 1.0x. True = use full
                    # risk_tier. Without this BT systematically inflates P&L
                    # on non-marginable micro-caps (OPTX 4/13: BT 2.0x → 10,303
                    # sh vs LIVE 1.0x → 5,799 sh; same setup, 2x P&L gap).
                    if risk_tier_mult > 1.0:
                        persisted = self._lookup_marginability(pattern.symbol)
                        if persisted is False:
                            logger.debug(
                                f"  {pattern.symbol}: BT marginability "
                                f"downgrade — universe.is_marginable=False "
                                f"(LIVE-observed); risk_tier "
                                f"{risk_tier_mult:.1f}x → 1.0x"
                            )
                            risk_tier_mult = 1.0

                # Combine risk tier + (sizing) conviction, cap at 3x (max leverage on $50K base).
                # sizing_conviction reflects the marginal defensive scaling; the cached
                # conviction_mult stays as the raw value so Stage-2 filters see it.
                combined_mult = min(3.0, risk_tier_mult * sizing_conviction)
                if combined_mult != 1.0:
                    plan = self.planner.create_plan(setup, avg_daily_volume=self._current_avg_daily_volume, risk_multiplier=combined_mult)
                    if plan is None:
                        continue
                    logger.debug(
                        f"  Risk scaling {combined_mult:.2f}x (tier={risk_tier_mult:.1f} × conv_size={sizing_conviction:.2f}) — "
                        f"{plan.shares} shares, ${plan.total_risk:.0f} risk"
                    )

                # Compute max intraday change at entry (for tier classification).
                # Reused: (a) MACD zone multiplier lookup (tier-aware), and
                # (b) stashed on pending_order for Stage-2 TTF filter.
                try:
                    from trading.two_tier_filter import max_intraday_change_pre_entry as _max_ic
                    _pre_bars = [
                        (str(bars.iloc[j].get('timestamp', '')),
                         bars.iloc[j].get('open'),
                         bars.iloc[j].get('high'),
                         bars.iloc[j].get('low'),
                         bars.iloc[j].get('close'))
                        for j in range(i + 1)  # inclusive of setup bar i
                    ]
                    _intraday_change_at_entry = _max_ic(
                        _pre_bars, prev_close, "\uffff",
                        premarket_extremes=premarket_extremes,
                    )
                except Exception as _ttf_exc:
                    logger.warning(
                        f"{symbol}: intraday_change_at_entry compute failed "
                        f"({_ttf_exc!r}); defaulting to 0.0 → A-tier MACD path."
                    )
                    _intraday_change_at_entry = None

                # MACD zone filter: skip dead zone, scale risk on strong zones
                # Skip scaling if risk tier already applied (don't compound)
                _applied_macd_zone = 1.0
                if self.macd_zones_enabled:
                    zone_mult = self._get_macd_zone_multiplier(
                        symbol, bars, i, plan.entry_price,
                        intraday_change_pct=float(_intraday_change_at_entry or 0.0)
                    )
                    if zone_mult == 0.0:
                        continue  # dead zone — don't place order
                    elif zone_mult != 1.0 and risk_tier_mult <= 1.0:
                        _applied_macd_zone = zone_mult
                        effective_max = int(self.planner.max_shares * zone_mult)
                        scaled_shares = min(effective_max, max(1, int(plan.shares * zone_mult)))
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

                # Regime-aware sizing (Phase 1.4b): C2 regime skips; A/C1 boost.
                # Stacks multiplicatively on top of MACD zone scaling (above).
                if self.regime_sizing_enabled:
                    from trading.regime_helpers import get_regime_multiplier
                    # Match the idiom used elsewhere in BT (e.g., line 2311):
                    # timestamp may be a column OR the DataFrame index.
                    _trade_date = str(
                        bars.iloc[i].get('timestamp', bars.iloc[i].name)
                    )[:10]
                    _regime = self._get_regime_for_date(_trade_date)
                    _regime_mult = get_regime_multiplier(_regime, self.regime_multipliers)
                    if _regime_mult == 0.0:
                        logger.info(
                            f"  REGIME {_regime} skip ({_trade_date}) — no trade"
                        )
                        continue
                    if _regime_mult != 1.0:
                        _reg_max = int(
                            self.planner.max_shares * _applied_macd_zone * _regime_mult
                        )
                        _reg_shares = min(_reg_max, max(1, int(plan.shares * _regime_mult)))
                        logger.info(
                            f"  REGIME {_regime} mult={_regime_mult:.2f} "
                            f"({_trade_date}) → shares {plan.shares} → {_reg_shares}"
                        )
                        plan = TradePlan(
                            symbol=plan.symbol,
                            entry_price=plan.entry_price,
                            stop_loss_price=plan.stop_loss_price,
                            take_profit_price=plan.take_profit_price,
                            risk_per_share=plan.risk_per_share,
                            reward_per_share=plan.reward_per_share,
                            risk_reward_ratio=plan.risk_reward_ratio,
                            shares=_reg_shares,
                            total_risk=plan.risk_per_share * _reg_shares,
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

                # Quality filter: compute features for EVERY setup, store on pending order.
                # Filter only when enabled (production + final BT validation).
                # During --build-cache: disabled → all trades generated with features logged.
                qf_features = self._compute_qf_features(
                    symbol, bars, setup, i, plan, prev_close)
                if self.quality_filter_enabled:
                    qf_pass, qf_reason = self._evaluate_qf(qf_features)
                    if not qf_pass:
                        logger.info(f"  QUALITY FILTER SKIP: {qf_reason}")
                        continue

                pending_order = PendingBuyStop(
                    setup=setup,
                    plan=plan,
                    placed_at_bar_idx=i,
                    breakout_level=setup.breakout_level,
                )
                pending_order._qf_features = qf_features
                pending_order._conviction_mult = conviction_mult
                pending_order._macd_zone_mult = _applied_macd_zone
                # Two-tier filter feature: max intraday change (max(gap%, range%))
                # over bars [0..i] INCLUSIVE of setup bar. Mirrors live parity:
                # scanner updates _day_max_intraday_change BEFORE the engine
                # runs setup detection on the same bar, so live includes it too.
                # Reuse value computed above for tier-aware MACD zone multiplier.
                pending_order._intraday_change_at_entry = _intraday_change_at_entry
                # Phase A — V2 research: stash per-rule contributions + spy_3d input
                # so they flow into the cache CSV via batch_backtest._trade_to_cache_row.
                pending_order._conv_breakdown = _conv_brkdn
                pending_order._spy_3d_range = _spy_3d
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
    parser.add_argument(
        "--include-premarket", action=argparse.BooleanOptionalAction, default=False,
        help="Fetch premarket 1-min bars (04:00-09:30 ET) and seed the "
             "qualification gate's running_high/low with their extremes. "
             "Default: False. 16-month validation showed PM-added trades "
             "have 50%% higher volatility (CoV 1.36→2.13) with outlier-"
             "driven edge — keep opt-in for BT-LIVE drift research."
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

    # Get previous day's closing price for qualification gate
    _prev_close_price = None
    if prev_day_bars is not None and not prev_day_bars.empty:
        _prev_close_price = float(prev_day_bars['close'].iloc[-1])
        logger.info(f"Qualification gate: prev_close=${_prev_close_price:.2f}")

    # Optional: fetch premarket bars and compute extremes
    pm_extremes = None
    if args.include_premarket:
        try:
            pm_start_et = _ET.localize(trade_date.replace(hour=4, minute=0, second=0))
            pm_end_et = _ET.localize(trade_date.replace(hour=9, minute=30, second=0))
            pm_start = pm_start_et.astimezone(timezone.utc)
            pm_end = pm_end_et.astimezone(timezone.utc)
            pm_bars = client.get_historical_1min_bars(symbol, pm_start, pm_end)
            if pm_bars is not None and not pm_bars.empty:
                pm_high = float(pm_bars['high'].max())
                pm_low = float(pm_bars['low'].min())
                pm_extremes = (pm_high, pm_low)
                logger.info(
                    f"Premarket: {len(pm_bars)} bars, "
                    f"high=${pm_high:.2f}, low=${pm_low:.2f}"
                )
            else:
                logger.info("Premarket: no bars returned (light/no premarket trading)")
        except Exception as e:
            logger.warning(f"Premarket fetch failed: {e}")

    # Run backtest
    runner = BacktestRunner()
    result = runner.run(symbol, bars, args.date, prev_day_bars=prev_day_bars,
                        prev_close=_prev_close_price,
                        premarket_extremes=pm_extremes)

    # Print report
    print_report(result)


if __name__ == "__main__":
    main()
