"""
Filter comparison tool for momentum day trading backtest.

Runs 4 scenarios with different filter configurations on the same movers data
to evaluate the impact of market regime filter and consecutive loss limit.

Scenarios:
1. All filters OFF (baseline)
2. Market regime filter ON only
3. Consecutive loss limit ON only (max_consecutive_losses=2)
4. All filters ON (regime + consecutive loss limit)

Usage:
    python3 backtest_filter_test.py --start 2025-01-01 --end 2026-03-20 --verbose
"""

import argparse
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

from backtest import BacktestRunner, BacktestResult
from batch_backtest import (
    fetch_daily_bars_cached,
    find_big_movers,
    run_batch_backtest,
)
from data_sources.alpaca_client import AlpacaClient
from persistence.database import get_database
from trading.market_regime import MarketRegimeFilter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "name": "All OFF",
        "regime_enabled": False,
        "max_consecutive_losses": 0,
        "max_trades_per_day": 0,
    },
    {
        "name": "Regime ON",
        "regime_enabled": True,
        "max_consecutive_losses": 0,
        "max_trades_per_day": 0,
    },
    {
        "name": "ConsecLoss",
        "regime_enabled": False,
        "max_consecutive_losses": 2,
        "max_trades_per_day": 0,
    },
    {
        "name": "All ON",
        "regime_enabled": True,
        "max_consecutive_losses": 2,
        "max_trades_per_day": 0,
    },
]


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(results: List[BacktestResult]) -> Dict:
    """
    Compute summary metrics from backtest results.

    Args:
        results: List of BacktestResult objects from a scenario run.

    Returns:
        Dict with trades, wins, losses, win_rate, total_pnl, avg_win,
        avg_loss, max_drawdown, profit_factor, sharpe.
    """
    all_trades = [t for r in results for t in r.trades_simulated]
    wins = [t for t in all_trades if t.pnl > 0]
    losses = [t for t in all_trades if t.pnl <= 0]

    total_pnl = sum(t.pnl for t in all_trades)
    win_rate = len(wins) / len(all_trades) * 100 if all_trades else 0.0
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0

    # Max drawdown (peak-to-trough on cumulative equity curve)
    max_drawdown = 0.0
    if all_trades:
        cum_pnl = 0.0
        peak = 0.0
        for t in all_trades:
            cum_pnl += t.pnl
            if cum_pnl > peak:
                peak = cum_pnl
            dd = peak - cum_pnl
            if dd > max_drawdown:
                max_drawdown = dd

    # Profit factor = gross wins / abs(gross losses)
    gross_wins = sum(t.pnl for t in wins) if wins else 0.0
    gross_losses = abs(sum(t.pnl for t in losses)) if losses else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Approximate Sharpe ratio (daily P&L based)
    sharpe = 0.0
    if all_trades:
        # Group trades by date to get daily P&L
        daily_pnl: Dict[str, float] = defaultdict(float)
        for r in results:
            for t in r.trades_simulated:
                daily_pnl[r.trade_date] += t.pnl

        pnl_series = list(daily_pnl.values())
        if len(pnl_series) > 1:
            mean_daily = np.mean(pnl_series)
            std_daily = np.std(pnl_series, ddof=1)
            if std_daily > 0:
                sharpe = (mean_daily / std_daily) * np.sqrt(252)

    return {
        "trades": len(all_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
    }


# ---------------------------------------------------------------------------
# Comparison table output
# ---------------------------------------------------------------------------


def print_comparison(
    scenario_metrics: List[Tuple[str, Dict]],
    start_date: date,
    end_date: date,
) -> None:
    """
    Print formatted comparison table across scenarios.

    Args:
        scenario_metrics: List of (scenario_name, metrics_dict) tuples.
        start_date: Backtest start date.
        end_date: Backtest end date.
    """
    months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    header = f"=== Filter Comparison: {months}-Month Honest Backtest ({start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}) ==="

    print("\n" + header)
    print()

    # Column widths
    label_w = 20
    col_w = 14
    names = [name for name, _ in scenario_metrics]

    # Header row
    print(f"{'':>{label_w}}", end="")
    for name in names:
        print(f"{name:>{col_w}}", end="")
    print()
    print("-" * (label_w + col_w * len(names)))

    # Data rows
    def _row_int(label: str, key: str):
        """Print a row with integer values."""
        print(f"{label:>{label_w}}", end="")
        for _, m in scenario_metrics:
            print(f"{m[key]:>{col_w},}", end="")
        print()

    def _row_pct(label: str, key: str):
        """Print a row with percentage values."""
        print(f"{label:>{label_w}}", end="")
        for _, m in scenario_metrics:
            print(f"{m[key]:>{col_w}.1f}%", end="")
        print()

    def _row_dollar(label: str, key: str):
        """Print a row with dollar values."""
        print(f"{label:>{label_w}}", end="")
        for _, m in scenario_metrics:
            val = m[key]
            formatted = f"${val:+,.0f}" if abs(val) >= 1 else f"${val:+,.2f}"
            print(f"{formatted:>{col_w}}", end="")
        print()

    def _row_float(label: str, key: str, fmt: str = ".2f"):
        """Print a row with float values."""
        print(f"{label:>{label_w}}", end="")
        for _, m in scenario_metrics:
            val = m[key]
            if val == float('inf'):
                print(f"{'inf':>{col_w}}", end="")
            else:
                print(f"{val:>{col_w}{fmt}}", end="")
        print()

    _row_int("Trades", "trades")
    _row_int("Wins", "wins")
    _row_int("Losses", "losses")
    _row_pct("Win Rate", "win_rate")
    _row_dollar("Total P&L", "total_pnl")
    _row_dollar("Avg Win", "avg_win")
    _row_dollar("Avg Loss", "avg_loss")
    _row_dollar("Max Drawdown", "max_drawdown")
    _row_float("Profit Factor", "profit_factor")
    _row_float("Sharpe (approx)", "sharpe")

    print()

    # Recommendation
    best_name, best_metrics = max(scenario_metrics, key=lambda x: x[1]["sharpe"])
    baseline_pnl = scenario_metrics[0][1]["total_pnl"]
    best_pnl = best_metrics["total_pnl"]
    pnl_diff = best_pnl - baseline_pnl

    print(f"RECOMMENDATION: '{best_name}' has best Sharpe ({best_metrics['sharpe']:.2f}). ", end="")
    if pnl_diff >= 0:
        print(f"P&L is ${pnl_diff:+,.0f} vs baseline.")
    else:
        print(f"P&L trades ${pnl_diff:,.0f} vs baseline for better risk-adjusted returns.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for filter comparison backtest."""
    parser = argparse.ArgumentParser(
        description="Compare filter configurations across 4 backtest scenarios"
    )
    parser.add_argument(
        "--start", type=str, default="2025-01-01",
        help="Start date YYYY-MM-DD (default: 2025-01-01)"
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

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    logger.info(f"Filter comparison backtest: {start_date} to {end_date}")

    # -----------------------------------------------------------------------
    # Step 1: Load universe and fetch data ONCE (shared across all scenarios)
    # -----------------------------------------------------------------------

    db = get_database()
    client = AlpacaClient(api_key=api_key, api_secret=api_secret)

    universe = db.get_active_universe()
    symbols = [s['symbol'] for s in universe]
    universe_dict = {s['symbol']: s for s in universe}
    logger.info(f"Loaded {len(symbols)} symbols from active universe")

    if not symbols:
        logger.error("No active symbols in universe -- run universe builder first")
        sys.exit(1)

    # Fetch daily bars (cached) -- shared by all scenarios
    from config import Config
    cfg = Config._load_yaml_only()
    scanner_cfg = cfg.get("scanner", {})

    logger.info("Fetching daily bars for date range (cache-first)...")
    daily_bars = fetch_daily_bars_cached(
        symbols, start_date - timedelta(days=7), end_date, client, db
    )

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
        logger.warning("No movers found -- nothing to backtest")
        sys.exit(0)

    logger.info(f"Found {len(movers)} mover/date pairs to test across 4 scenarios")

    # Fetch SPY bars for regime filter (shared, fetched once)
    regime_cfg = cfg.get("trading", {}).get("market_regime", {})
    sma_period = int(regime_cfg.get("sma_period", 50))
    spy_lookback_days = int(sma_period * 1.5) + 14
    spy_start = start_date - timedelta(days=spy_lookback_days)
    spy_bars_raw = fetch_daily_bars_cached(['SPY'], spy_start, end_date, client, db)
    spy_bars = spy_bars_raw.get('SPY', [])
    logger.info(f"Loaded {len(spy_bars)} SPY daily bars for regime filter")

    # -----------------------------------------------------------------------
    # Step 2: Run each scenario
    # -----------------------------------------------------------------------

    scenario_metrics = []

    for i, scenario in enumerate(SCENARIOS, 1):
        scenario_name = scenario["name"]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"SCENARIO {i}/4: {scenario_name}")
        logger.info(f"  regime_enabled={scenario['regime_enabled']}, "
                     f"max_consecutive_losses={scenario['max_consecutive_losses']}, "
                     f"max_trades_per_day={scenario['max_trades_per_day']}")
        logger.info(f"{'=' * 60}")

        # Build regime filter for this scenario
        regime_filter = MarketRegimeFilter(
            enabled=scenario["regime_enabled"],
            vol_threshold=float(regime_cfg.get("vol_threshold", 1.5)),
            sma_period=sma_period,
            max_trades_per_day=scenario["max_trades_per_day"],
            min_spy_volume_ratio=float(regime_cfg.get("min_spy_volume_ratio", 0.70)),
            thin_liquidity_breakout_vol_ratio=float(
                regime_cfg.get("thin_liquidity_breakout_vol_ratio", 2.0)
            ),
        )
        regime_filter.load_spy_bars(spy_bars)

        # Build a fresh BacktestRunner per scenario (uses from_config defaults)
        runner = BacktestRunner()

        results = run_batch_backtest(
            movers,
            client,
            runner,
            db=db,
            universe_dict=universe_dict,
            market_regime=regime_filter,
            max_consecutive_losses=scenario["max_consecutive_losses"],
            max_trades_per_day=scenario["max_trades_per_day"],
        )

        metrics = compute_metrics(results)
        scenario_metrics.append((scenario_name, metrics))

        logger.info(
            f"Scenario '{scenario_name}' complete: "
            f"{metrics['trades']} trades, "
            f"P&L ${metrics['total_pnl']:+,.0f}, "
            f"WR {metrics['win_rate']:.1f}%, "
            f"Sharpe {metrics['sharpe']:.2f}"
        )

    # -----------------------------------------------------------------------
    # Step 3: Print comparison table
    # -----------------------------------------------------------------------

    print_comparison(scenario_metrics, start_date, end_date)


if __name__ == "__main__":
    main()
