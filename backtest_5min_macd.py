"""
5-Minute MACD Exit Signal Backtest.

Tests whether exiting when the 5-min MACD histogram crosses below zero
(while in a profitable trade) improves over the baseline trailing stop.

For each trade from the baseline backtest, re-simulates with a MACD exit
overlay: if the 5-min MACD histogram crosses below zero while the trade
is profitable AND before the baseline exit, exit at the next bar's open.

MACD is warmed up using the previous trading day's last ~60 1-min bars
to avoid cold-start artifacts in the early-morning 5-min candles.

Usage:
    python backtest_5min_macd.py --start 2026-01-01 --end 2026-03-20 --verbose
"""

import argparse
import logging
import os
import sys
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz
from dotenv import load_dotenv

from backtest import BacktestRunner, BacktestResult, SimulatedTrade
from batch_backtest import (
    fetch_daily_bars_cached,
    find_big_movers,
    get_1min_bars_cached,
    run_batch_backtest,
    _market_hours_utc,
)
from data_sources.alpaca_client import AlpacaClient, AlpacaAPIError
from persistence.database import get_database
from trading.indicators import macd_histogram

logger = logging.getLogger(__name__)

ET = pytz.timezone('US/Eastern')


# ---------------------------------------------------------------------------
# 5-min bar resampling with warmup
# ---------------------------------------------------------------------------


def resample_1min_to_5min(bars_1min: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-minute bars to 5-minute bars.

    Groups by 5-minute intervals (09:30, 09:35, ...) using the bar's
    timestamp as the period start.

    Args:
        bars_1min: DataFrame with columns [timestamp, open, high, low, close, volume]

    Returns:
        DataFrame with 5-min OHLCV bars, indexed by period start timestamp
    """
    if bars_1min.empty:
        return pd.DataFrame()

    df = bars_1min.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.set_index('timestamp').sort_index()

    resampled = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['open'])

    resampled = resampled.reset_index()
    return resampled


def get_previous_trading_day(trade_date: date) -> date:
    """
    Get the previous trading day (skip weekends).

    Args:
        trade_date: The current trading date

    Returns:
        Previous trading day (Mon->Fri, Tue-Fri->prev weekday)
    """
    prev = trade_date - timedelta(days=1)
    # Skip weekends
    while prev.weekday() >= 5:  # 5=Sat, 6=Sun
        prev -= timedelta(days=1)
    return prev


def build_5min_bars_with_warmup(
    symbol: str,
    trade_date: date,
    bars_1min: pd.DataFrame,
    client: AlpacaClient,
    db,
) -> pd.DataFrame:
    """
    Build 5-min bars for a trade date with MACD warmup from previous day.

    Fetches the previous trading day's last ~60 1-min bars, prepends them
    to the current day's bars, resamples to 5-min, and computes MACD on
    the full series. Returns only the current day's 5-min bars (with MACD
    values warmed up by the previous day's data).

    Args:
        symbol: Stock symbol
        trade_date: Current trading date
        bars_1min: Current day's 1-min bars
        client: AlpacaClient for fetching previous day data
        db: Database for caching

    Returns:
        DataFrame with 5-min bars and 'macd_hist' column, filtered to trade_date only
    """
    prev_date = get_previous_trading_day(trade_date)

    # Fetch previous day's 1-min bars for warmup
    try:
        prev_bars = get_1min_bars_cached(symbol, prev_date, client, db)
    except Exception as e:
        logger.warning(
            f"{symbol}: Could not fetch prev day bars for MACD warmup: {e}"
        )
        prev_bars = pd.DataFrame()

    # Take last ~60 bars from previous day for warmup
    if not prev_bars.empty:
        warmup_bars = prev_bars.tail(60).copy()
        combined = pd.concat([warmup_bars, bars_1min], ignore_index=True)
        logger.debug(
            f"{symbol}: MACD warmup with {len(warmup_bars)} prev-day bars + "
            f"{len(bars_1min)} trade-day bars"
        )
    else:
        combined = bars_1min.copy()
        logger.debug(f"{symbol}: No warmup bars, using trade-day only")

    # Resample to 5-min
    bars_5min = resample_1min_to_5min(combined)
    if bars_5min.empty:
        return bars_5min

    # Compute MACD histogram on full (warmed-up) series
    bars_5min['macd_hist'] = macd_histogram(bars_5min['close']).values

    # Filter to trade_date only
    if not pd.api.types.is_datetime64_any_dtype(bars_5min['timestamp']):
        bars_5min['timestamp'] = pd.to_datetime(bars_5min['timestamp'])

    # Make timestamps tz-aware for comparison
    trade_day_start, trade_day_end = _market_hours_utc(trade_date)
    if bars_5min['timestamp'].dt.tz is None:
        bars_5min['timestamp'] = bars_5min['timestamp'].dt.tz_localize('UTC')

    trade_day_bars = bars_5min[
        (bars_5min['timestamp'] >= trade_day_start) &
        (bars_5min['timestamp'] <= trade_day_end)
    ].copy().reset_index(drop=True)

    return trade_day_bars


# ---------------------------------------------------------------------------
# MACD exit overlay simulation
# ---------------------------------------------------------------------------


def find_macd_exit(
    trade: SimulatedTrade,
    bars_5min: pd.DataFrame,
    bars_1min: pd.DataFrame,
) -> Optional[Tuple[datetime, float, str]]:
    """
    Check if a MACD histogram cross below zero would exit this trade earlier.

    Scans 5-min bars from trade entry to trade exit. If the MACD histogram
    crosses below zero while the trade is profitable, returns the exit
    signal (time and price from the next 1-min bar's open after the 5-min
    bar closes).

    Args:
        trade: The baseline SimulatedTrade
        bars_5min: 5-min bars with 'macd_hist' column (trade date only)
        bars_1min: 1-min bars for the trade date

    Returns:
        Tuple of (exit_time, exit_price, 'macd_cross') if MACD exit triggers,
        None if baseline exit happens first
    """
    if bars_5min.empty or len(bars_5min) < 2:
        return None

    entry_time = trade.entry_time
    baseline_exit_time = trade.exit_time
    entry_price = trade.entry_price

    # Make timestamps comparable
    def _ensure_tz(ts):
        if ts is None:
            return None
        if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
            return ts
        return ts.replace(tzinfo=timezone.utc)

    entry_time = _ensure_tz(entry_time)
    baseline_exit_time = _ensure_tz(baseline_exit_time)

    prev_hist = None

    for i in range(len(bars_5min)):
        bar_5m = bars_5min.iloc[i]
        bar_ts = _ensure_tz(bar_5m['timestamp'])

        # Only look at bars after entry
        if bar_ts <= entry_time:
            prev_hist = bar_5m['macd_hist']
            continue

        # Don't look past baseline exit
        if baseline_exit_time and bar_ts > baseline_exit_time:
            break

        current_hist = bar_5m['macd_hist']

        # Detect cross below zero: previous was >= 0, current < 0
        if prev_hist is not None and prev_hist >= 0 and current_hist < 0:
            # Check if trade is profitable at this 5-min bar's close
            bar_close = bar_5m['close']
            unrealized_pnl = (bar_close - entry_price) * trade.shares

            if unrealized_pnl > 0:
                # Find the next 1-min bar after this 5-min bar closes
                # The 5-min bar timestamp is the period START, so the bar
                # covers [ts, ts+5min). The exit is at the open of the next
                # 1-min bar after ts+5min.
                five_min_end = bar_ts + timedelta(minutes=5)

                exit_bar = None
                for j in range(len(bars_1min)):
                    bar_1m_ts = _ensure_tz(bars_1min.iloc[j]['timestamp'])
                    if bar_1m_ts >= five_min_end:
                        exit_bar = bars_1min.iloc[j]
                        break

                if exit_bar is not None:
                    exit_ts = _ensure_tz(exit_bar['timestamp'])
                    # Only trigger if this exit is before the baseline exit
                    if baseline_exit_time is None or exit_ts < baseline_exit_time:
                        logger.debug(
                            f"  MACD exit signal at {bar_ts}: hist {prev_hist:.4f} -> "
                            f"{current_hist:.4f}, exit at {exit_bar['open']:.2f}"
                        )
                        return (exit_ts, float(exit_bar['open']), 'macd_cross')

        prev_hist = current_hist

    return None


def apply_macd_exit(trade: SimulatedTrade, macd_exit: Tuple) -> SimulatedTrade:
    """
    Create a modified copy of a trade with MACD exit applied.

    Args:
        trade: Original baseline trade
        macd_exit: Tuple of (exit_time, exit_price, exit_reason)

    Returns:
        New SimulatedTrade with MACD exit details
    """
    exit_time, exit_price, exit_reason = macd_exit

    modified = SimulatedTrade(
        symbol=trade.symbol,
        entry_time=trade.entry_time,
        entry_price=trade.entry_price,
        stop_loss=trade.stop_loss,
        take_profit=trade.take_profit,
        shares=trade.shares,
        exit_time=exit_time,
        exit_price=exit_price,
        exit_reason=exit_reason,
        plan=trade.plan,
        entry_bar_open=trade.entry_bar_open,
        entry_bar_high=trade.entry_bar_high,
        entry_bar_low=trade.entry_bar_low,
        entry_bar_close=trade.entry_bar_close,
        entry_bar_volume=trade.entry_bar_volume,
        planned_entry=trade.planned_entry,
        entry_gap=trade.entry_gap,
    )

    # Compute P&L
    modified.pnl = (exit_price - trade.entry_price) * trade.shares
    modified.pnl_pct = (
        (exit_price - trade.entry_price) / trade.entry_price * 100
        if trade.entry_price > 0 else 0.0
    )

    return modified


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def compute_stats(trades: List[SimulatedTrade]) -> Dict:
    """
    Compute summary statistics for a list of trades.

    Args:
        trades: List of SimulatedTrade objects

    Returns:
        Dict with count, wins, losses, win_rate, total_pnl, avg_win, avg_loss,
        profit_factor
    """
    if not trades:
        return {
            'count': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0,
            'total_pnl': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
            'profit_factor': 0.0,
        }

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in trades)
    win_rate = len(wins) / len(trades) * 100 if trades else 0.0

    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0

    gross_wins = sum(t.pnl for t in wins) if wins else 0.0
    gross_losses = abs(sum(t.pnl for t in losses)) if losses else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    return {
        'count': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
    }


def print_comparison(baseline_stats: Dict, macd_stats: Dict, start: str, end: str) -> None:
    """
    Print side-by-side comparison of baseline vs MACD exit strategies.

    Args:
        baseline_stats: Stats dict from compute_stats for baseline
        macd_stats: Stats dict from compute_stats for MACD exit
        start: Start date string
        end: End date string
    """
    b = baseline_stats
    m = macd_stats

    print()
    print("=" * 65)
    print(f"  5-Min MACD Exit Signal Test ({start} to {end})")
    print("=" * 65)
    print()
    print(f"  {'':20s} {'Baseline (Trail)':>20s}    {'MACD Exit':>20s}")
    print(f"  {'-'*20} {'-'*20}    {'-'*20}")
    print(f"  {'Trades':20s} {b['count']:>20d}    {m['count']:>20d}")
    print(f"  {'Wins':20s} {b['wins']:>20d}    {m['wins']:>20d}")
    print(f"  {'Losses':20s} {b['losses']:>20d}    {m['losses']:>20d}")
    print(f"  {'Win Rate':20s} {b['win_rate']:>19.1f}%    {m['win_rate']:>19.1f}%")
    print(f"  {'Total P&L':20s} {'${:>,.2f}'.format(b['total_pnl']):>20s}    {'${:>,.2f}'.format(m['total_pnl']):>20s}")
    print(f"  {'Avg Win':20s} {'${:>,.2f}'.format(b['avg_win']):>20s}    {'${:>,.2f}'.format(m['avg_win']):>20s}")
    print(f"  {'Avg Loss':20s} {'${:>,.2f}'.format(b['avg_loss']):>20s}    {'${:>,.2f}'.format(m['avg_loss']):>20s}")

    pf_b = f"{b['profit_factor']:.2f}" if b['profit_factor'] != float('inf') else "inf"
    pf_m = f"{m['profit_factor']:.2f}" if m['profit_factor'] != float('inf') else "inf"
    print(f"  {'Profit Factor':20s} {pf_b:>20s}    {pf_m:>20s}")
    print("=" * 65)

    # Delta summary
    pnl_delta = m['total_pnl'] - b['total_pnl']
    wr_delta = m['win_rate'] - b['win_rate']
    print()
    print(f"  Delta P&L:    ${pnl_delta:>+,.2f}")
    print(f"  Delta WR:     {wr_delta:>+.1f}%")
    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_macd_backtest(
    start_date: date,
    end_date: date,
    verbose: bool = False,
) -> None:
    """
    Run the full 5-min MACD exit signal comparison backtest.

    Steps:
    1. Load universe, find movers, run baseline backtest
    2. For each trade, build 5-min MACD bars and check for earlier exit
    3. Print side-by-side comparison

    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        verbose: Enable debug logging
    """
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        logger.error("Missing ALPACA_API_KEY or ALPACA_API_SECRET in environment")
        sys.exit(1)

    client = AlpacaClient(api_key=api_key, api_secret=api_secret)
    db = get_database()

    # --- Step 1: Load universe and find movers ---
    universe = db.get_active_universe()
    symbols = [s['symbol'] for s in universe]
    logger.info(f"Loaded {len(symbols)} symbols from active universe")

    if not symbols:
        logger.error("No active symbols in universe")
        sys.exit(1)

    logger.info("Fetching daily bars for mover detection...")
    daily_bars = fetch_daily_bars_cached(
        symbols, start_date - timedelta(days=7), end_date, client, db
    )
    universe_dict = {s['symbol']: s for s in universe}

    from config import Config
    from trading.market_regime import MarketRegimeFilter

    cfg = Config._load_yaml_only()
    scanner_cfg = cfg.get("scanner", {})
    trading_cfg = cfg.get("trading", {})

    movers = find_big_movers(
        daily_bars,
        universe_dict=universe_dict,
        price_min=float(scanner_cfg.get("price_min", 2.0)),
        price_max=float(scanner_cfg.get("price_max", 20.0)),
        float_max=int(scanner_cfg.get("float_max", 10_000_000)),
        start_date=start_date,
        end_date=end_date,
    )

    if not movers:
        logger.warning("No movers found in date range")
        return

    logger.info(f"Found {len(movers)} mover pairs to backtest")

    # --- Build market regime filter ---
    regime_cfg = trading_cfg.get("market_regime", {})
    sma_period = int(regime_cfg.get("sma_period", 50))
    spy_lookback_days = int(sma_period * 1.5) + 14
    spy_start = start_date - timedelta(days=spy_lookback_days)
    spy_bars_raw = fetch_daily_bars_cached(['SPY'], spy_start, end_date, client, db)
    spy_bars = spy_bars_raw.get('SPY', [])
    max_trades_per_day = int(trading_cfg.get("max_trades_per_day", 5))

    market_regime = MarketRegimeFilter(
        enabled=bool(regime_cfg.get("enabled", True)),
        vol_threshold=float(regime_cfg.get("vol_threshold", 1.5)),
        sma_period=sma_period,
        max_trades_per_day=max_trades_per_day,
        min_spy_volume_ratio=float(regime_cfg.get("min_spy_volume_ratio", 0.70)),
        thin_liquidity_breakout_vol_ratio=float(
            regime_cfg.get("thin_liquidity_breakout_vol_ratio", 2.0)
        ),
    )
    market_regime.load_spy_bars(spy_bars)
    max_consec = int(trading_cfg.get("max_consecutive_losses", 2))

    # --- Step 2: Run baseline backtest ---
    logger.info("Running baseline backtest...")
    runner = BacktestRunner()
    results = run_batch_backtest(
        movers, client, runner, db=db, universe_dict=universe_dict,
        market_regime=market_regime,
        max_consecutive_losses=max_consec,
    )

    baseline_trades = [t for r in results for t in r.trades_simulated]
    logger.info(f"Baseline: {len(baseline_trades)} trades")

    if not baseline_trades:
        logger.warning("No baseline trades — nothing to compare")
        return

    # --- Step 3: Apply MACD exit overlay ---
    logger.info("Applying 5-min MACD exit overlay...")

    # Cache 5-min bars per (symbol, date) to avoid recomputing
    bars_5min_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    bars_1min_cache: Dict[Tuple[str, str], pd.DataFrame] = {}

    macd_trades = []
    macd_modified_count = 0

    for trade in baseline_trades:
        symbol = trade.symbol
        # Extract trade date from entry_time
        entry_ts = trade.entry_time
        if hasattr(entry_ts, 'date'):
            trade_date = entry_ts.date() if entry_ts.tzinfo is None else (
                entry_ts.astimezone(ET).date()
            )
        else:
            trade_date = start_date  # fallback

        cache_key = (symbol, trade_date.isoformat())

        # Get or build 1-min bars
        if cache_key not in bars_1min_cache:
            try:
                bars_1min_cache[cache_key] = get_1min_bars_cached(
                    symbol, trade_date, client, db
                )
            except Exception as e:
                logger.warning(f"{symbol} {trade_date}: Failed to get 1-min bars: {e}")
                bars_1min_cache[cache_key] = pd.DataFrame()

        bars_1min = bars_1min_cache[cache_key]

        # Get or build 5-min bars with MACD
        if cache_key not in bars_5min_cache:
            if not bars_1min.empty:
                bars_5min_cache[cache_key] = build_5min_bars_with_warmup(
                    symbol, trade_date, bars_1min, client, db
                )
            else:
                bars_5min_cache[cache_key] = pd.DataFrame()

        bars_5min = bars_5min_cache[cache_key]

        # Check for MACD exit
        macd_exit = find_macd_exit(trade, bars_5min, bars_1min)

        if macd_exit is not None:
            modified_trade = apply_macd_exit(trade, macd_exit)
            macd_trades.append(modified_trade)
            macd_modified_count += 1
            if verbose:
                logger.info(
                    f"  {symbol} {trade_date}: MACD exit at ${macd_exit[1]:.2f} "
                    f"(was {trade.exit_reason} at ${trade.exit_price:.2f}, "
                    f"PnL ${trade.pnl:.2f} -> ${modified_trade.pnl:.2f})"
                )
        else:
            # Keep baseline trade unchanged
            macd_trades.append(trade)

    logger.info(
        f"MACD overlay: {macd_modified_count}/{len(baseline_trades)} trades modified"
    )

    # --- Step 4: Print comparison ---
    baseline_stats = compute_stats(baseline_trades)
    macd_stats = compute_stats(macd_trades)
    print_comparison(
        baseline_stats, macd_stats,
        start_date.isoformat(), end_date.isoformat(),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for 5-min MACD exit signal backtest."""
    parser = argparse.ArgumentParser(
        description="Test 5-min MACD histogram exit signal vs baseline trailing stop"
    )
    parser.add_argument(
        "--start", type=str, default="2026-01-01",
        help="Start date YYYY-MM-DD (default: 2026-01-01)"
    )
    parser.add_argument(
        "--end", type=str, default="2026-03-20",
        help="End date YYYY-MM-DD (default: 2026-03-20)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose/debug logging"
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    s = datetime.strptime(args.start, "%Y-%m-%d").date()
    e = datetime.strptime(args.end, "%Y-%m-%d").date()

    logger.info(f"5-Min MACD Exit Signal Test: {s} to {e}")
    run_macd_backtest(s, e, verbose=args.verbose)


if __name__ == "__main__":
    main()
