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
from trading.pattern_detector import BullFlagDetector, BullFlagPattern
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
    exit_reason: Optional[str] = None  # 'target', 'stop', 'eod'
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


@dataclass
class PatternDetection:
    """Record of a pattern detected during backtest scanning."""

    bar_index: int
    timestamp: datetime
    pattern: BullFlagPattern


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
    - Entry at plan.entry_price (breakout level)
    - Each bar: check stop then target (conservative on ambiguity)
    - End of day: exit at last bar's close
    """

    def simulate(
        self, plan: TradePlan, bars: pd.DataFrame, entry_bar_idx: int
    ) -> SimulatedTrade:
        """
        Simulate a trade from entry_bar_idx through remaining bars.

        Args:
            plan: The trade plan with entry/stop/target
            bars: Full day's bars DataFrame
            entry_bar_idx: Index of the bar where entry occurs

        Returns:
            SimulatedTrade with fill details and P&L
        """
        entry_bar = bars.iloc[entry_bar_idx]
        trade = SimulatedTrade(
            symbol=plan.symbol,
            entry_time=entry_bar['timestamp'],
            entry_price=plan.entry_price,
            stop_loss=plan.stop_loss_price,
            take_profit=plan.take_profit_price,
            shares=plan.shares,
            plan=plan,
            entry_bar_open=float(entry_bar['open']),
            entry_bar_high=float(entry_bar['high']),
            entry_bar_low=float(entry_bar['low']),
            entry_bar_close=float(entry_bar['close']),
            entry_bar_volume=int(entry_bar['volume']),
        )

        last_bar_idx = len(bars) - 1

        for i in range(entry_bar_idx + 1, len(bars)):
            bar = bars.iloc[i]
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

    def _exit_trade(
        self, trade: SimulatedTrade, bar: pd.Series, reason: str, price: float
    ) -> None:
        """Fill in exit details on the trade."""
        trade.exit_time = bar['timestamp']
        trade.exit_price = price
        trade.exit_reason = reason
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

    Sliding window from bar 7 to N-2 (minimum bars for detection).
    Takes only the first valid trade (one trade per symbol per day).
    Tracks all pattern detections for analysis.
    """

    MIN_BARS_FOR_DETECTION = 7  # 3 pole + 2 pullback + 1 breakout + 1 dropped

    # Default filters based on Feb+Mar 2026 backtest analysis (48 trades):
    # - Midday skip alone: 32 trades, 62.5% WR, $4,241 PnL (best PnL retention)
    # - No price filter: sub-$5 trades have big avg wins ($502) despite lower WR
    DEFAULT_MIN_PRICE = 0.0
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
    ):
        """
        Initialize BacktestRunner.

        Args:
            detector: BullFlagDetector instance (uses defaults if None)
            planner: TradePlanner instance (uses defaults if None)
            simulator: TradeSimulator instance (uses defaults if None)
            min_price: Minimum entry price filter (default 5.0)
            skip_midday: Skip 11:30-14:00 ET entries (default True)
            early_exit_after_trade: Stop scanning after first trade (default True).
                Set to False to count all pattern detections for analysis.
        """
        self.detector = detector or BullFlagDetector()
        self.planner = planner or TradePlanner()
        self.simulator = simulator or TradeSimulator()
        self.min_price = min_price if min_price is not None else self.DEFAULT_MIN_PRICE
        self.skip_midday = skip_midday if skip_midday is not None else self.DEFAULT_SKIP_MIDDAY
        self.early_exit_after_trade = early_exit_after_trade

    def run(self, symbol: str, bars: pd.DataFrame, trade_date: str) -> BacktestResult:
        """
        Run backtest for a symbol over a day's bars.

        Args:
            symbol: Stock ticker symbol
            bars: DataFrame with 1-min bars (timestamp, OHLCV)
            trade_date: Date string for reporting (e.g., '2026-03-13')

        Returns:
            BacktestResult with trades, patterns, and P&L
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
        last_end = len(bars) - 1  # Leave room — detector drops last bar

        logger.info(f"{symbol}: Scanning {len(bars)} bars for patterns...")

        for i in range(self.MIN_BARS_FOR_DETECTION - 1, last_end):
            # Pass full DataFrame + end_idx to avoid O(N^2) DataFrame copies.
            # end_idx=i means completed = bars[:i], which matches the old behavior
            # of passing bars[:i+1] and then having detect() drop the last bar.
            pattern = self.detector.detect(symbol, bars, end_idx=i)

            if pattern is None:
                continue

            # Record detection
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

            # Price filter: skip sub-min_price stocks
            if self.min_price > 0 and plan.entry_price < self.min_price:
                logger.debug(
                    f"  Skipping — entry ${plan.entry_price:.2f} below "
                    f"min price ${self.min_price:.2f}"
                )
                continue

            # Midday filter: skip 11:30-14:00 ET (15:30-18:00 UTC)
            if self.skip_midday:
                bar_ts = bars.iloc[i]['timestamp']
                bar_hour = bar_ts.hour if hasattr(bar_ts, 'hour') else 0
                bar_minute = bar_ts.minute if hasattr(bar_ts, 'minute') else 0
                bar_time_utc = bar_hour + bar_minute / 60.0
                if self.MIDDAY_START_UTC + 0.5 <= bar_time_utc < self.MIDDAY_END_UTC:
                    logger.debug(
                        f"  Skipping — midday entry at {bar_ts} "
                        f"(11:30-14:00 ET filter)"
                    )
                    continue

            # Simulate the trade
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
