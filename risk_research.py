"""
Risk model research: test position sizing, stop thresholds, and R:R optimization.

Runs 10 hypotheses against historical backtest data to find the optimal
combination that maximizes P&L while maintaining 60%+ WR and true 2:1 R:R.

Usage:
    python risk_research.py                                       # run all 10
    python risk_research.py --hypothesis H0,H1,H4                 # run subset
    python risk_research.py --hypothesis H1 --start 2026-02-01    # single with date range
"""

import argparse
import csv
import logging
import math
import os
import sys
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from backtest import BacktestRunner, BacktestResult, SimulatedTrade
from batch_backtest import (
    fetch_daily_bars_cached,
    find_big_movers,
    run_batch_backtest,
)
from data_sources.alpaca_client import AlpacaClient
from persistence.database import get_database
from trading.pattern_detector import BullFlagDetector
from trading.trade_planner import TradePlanner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hypothesis Registry
# ---------------------------------------------------------------------------

HYPOTHESES: Dict[str, Dict] = {
    "H0": {
        "description": "Baseline (current)",
        "position_size_dollars": 50000,
        "sizing_mode": "fixed_investment",
        "risk_per_trade": 500,
        "min_risk_per_share": 0.05,
        "max_risk_per_share": 0.20,
        "min_risk_pct": None,
        "max_risk_pct": None,
        "min_risk_reward": 2.0,
    },
    "H1": {
        "description": "Fixed risk sizing only",
        "position_size_dollars": 50000,
        "sizing_mode": "fixed_risk",
        "risk_per_trade": 500,
        "min_risk_per_share": 0.05,
        "max_risk_per_share": 0.20,
        "min_risk_pct": None,
        "max_risk_pct": None,
        "min_risk_reward": 2.0,
    },
    "H2": {
        "description": "Conservative risk",
        "position_size_dollars": 50000,
        "sizing_mode": "fixed_risk",
        "risk_per_trade": 250,
        "min_risk_per_share": 0.05,
        "max_risk_per_share": 0.20,
        "min_risk_pct": None,
        "max_risk_pct": None,
        "min_risk_reward": 2.0,
    },
    "H3": {
        "description": "Aggressive risk",
        "position_size_dollars": 50000,
        "sizing_mode": "fixed_risk",
        "risk_per_trade": 1000,
        "min_risk_per_share": 0.05,
        "max_risk_per_share": 0.20,
        "min_risk_pct": None,
        "max_risk_pct": None,
        "min_risk_reward": 2.0,
    },
    "H4": {
        "description": "Pct stops, moderate",
        "position_size_dollars": 50000,
        "sizing_mode": "fixed_risk",
        "risk_per_trade": 500,
        "min_risk_per_share": 0.05,
        "max_risk_per_share": 0.20,
        "min_risk_pct": 0.01,
        "max_risk_pct": 0.05,
        "min_risk_reward": 2.0,
    },
    "H5": {
        "description": "Pct stops, tight",
        "position_size_dollars": 50000,
        "sizing_mode": "fixed_risk",
        "risk_per_trade": 500,
        "min_risk_per_share": 0.05,
        "max_risk_per_share": 0.20,
        "min_risk_pct": 0.005,
        "max_risk_pct": 0.03,
        "min_risk_reward": 2.0,
    },
    "H6": {
        "description": "Pct stops, wide",
        "position_size_dollars": 50000,
        "sizing_mode": "fixed_risk",
        "risk_per_trade": 500,
        "min_risk_per_share": 0.05,
        "max_risk_per_share": 0.20,
        "min_risk_pct": 0.005,
        "max_risk_pct": 0.08,
        "min_risk_reward": 2.0,
    },
    "H7": {
        "description": "Lower R:R, more wins",
        "position_size_dollars": 50000,
        "sizing_mode": "fixed_risk",
        "risk_per_trade": 500,
        "min_risk_per_share": 0.05,
        "max_risk_per_share": 0.20,
        "min_risk_pct": 0.01,
        "max_risk_pct": 0.05,
        "min_risk_reward": 1.5,
    },
    "H8": {
        "description": "Higher R:R, bigger wins",
        "position_size_dollars": 50000,
        "sizing_mode": "fixed_risk",
        "risk_per_trade": 500,
        "min_risk_per_share": 0.05,
        "max_risk_per_share": 0.20,
        "min_risk_pct": 0.01,
        "max_risk_pct": 0.05,
        "min_risk_reward": 3.0,
    },
    "H9": {
        "description": "Mid R:R sweet spot",
        "position_size_dollars": 50000,
        "sizing_mode": "fixed_risk",
        "risk_per_trade": 500,
        "min_risk_per_share": 0.05,
        "max_risk_per_share": 0.20,
        "min_risk_pct": 0.01,
        "max_risk_pct": 0.05,
        "min_risk_reward": 2.5,
    },
    "H9a": {
        "description": "H9 + $750 risk budget (1.5%)",
        "position_size_dollars": 50000,
        "sizing_mode": "fixed_risk",
        "risk_per_trade": 750,
        "min_risk_per_share": 0.05,
        "max_risk_per_share": 0.20,
        "min_risk_pct": 0.01,
        "max_risk_pct": 0.05,
        "min_risk_reward": 2.5,
    },
    "H9b": {
        "description": "H9 + $1000 risk budget (2%)",
        "position_size_dollars": 50000,
        "sizing_mode": "fixed_risk",
        "risk_per_trade": 1000,
        "min_risk_per_share": 0.05,
        "max_risk_per_share": 0.20,
        "min_risk_pct": 0.01,
        "max_risk_pct": 0.05,
        "min_risk_reward": 2.5,
    },
    "H10": {
        "description": "H9b + MACD positive filter",
        "position_size_dollars": 50000,
        "sizing_mode": "fixed_risk",
        "risk_per_trade": 1000,
        "min_risk_per_share": 0.05,
        "max_risk_per_share": 0.20,
        "min_risk_pct": 0.01,
        "max_risk_pct": 0.05,
        "min_risk_reward": 2.5,
        "require_macd_positive": True,
    },
    "H10a": {
        "description": "H10 + $2000 risk budget",
        "position_size_dollars": 50000,
        "sizing_mode": "fixed_risk",
        "risk_per_trade": 2000,
        "min_risk_per_share": 0.05,
        "max_risk_per_share": 0.20,
        "min_risk_pct": 0.01,
        "max_risk_pct": 0.05,
        "min_risk_reward": 2.5,
        "require_macd_positive": True,
    },
    "H11": {
        "description": "H10a + 1.5R target",
        "position_size_dollars": 50000,
        "sizing_mode": "fixed_risk",
        "risk_per_trade": 2000,
        "min_risk_per_share": 0.05,
        "max_risk_per_share": 0.20,
        "min_risk_pct": 0.01,
        "max_risk_pct": 0.05,
        "min_risk_reward": 1.5,
        "require_macd_positive": True,
    },
    "H11a": {
        "description": "H10a + 2.0R target",
        "position_size_dollars": 50000,
        "sizing_mode": "fixed_risk",
        "risk_per_trade": 2000,
        "min_risk_per_share": 0.05,
        "max_risk_per_share": 0.20,
        "min_risk_pct": 0.01,
        "max_risk_pct": 0.05,
        "min_risk_reward": 2.0,
        "require_macd_positive": True,
    },
    "H12": {
        "description": "H10a no MACD + relaxed risk + circuit breaker",
        "position_size_dollars": 50000,
        "sizing_mode": "fixed_risk",
        "risk_per_trade": 2000,
        "min_risk_per_share": 0.02,
        "max_risk_per_share": 0.20,
        "min_risk_pct": 0.005,
        "max_risk_pct": 0.05,
        "min_risk_reward": 2.5,
        "circuit_breaker_dd": 3000,
        "circuit_breaker_pause": 2,
    },
}


def build_planner(hypothesis_id: str) -> TradePlanner:
    """
    Build a TradePlanner configured for a given hypothesis.

    Args:
        hypothesis_id: Key into HYPOTHESES registry (e.g., "H0")

    Returns:
        TradePlanner configured with hypothesis parameters

    Raises:
        KeyError: If hypothesis_id not in registry
    """
    if hypothesis_id not in HYPOTHESES:
        raise KeyError(f"Unknown hypothesis '{hypothesis_id}'. Valid: {list(HYPOTHESES.keys())}")

    params = HYPOTHESES[hypothesis_id]
    return TradePlanner(
        position_size_dollars=params["position_size_dollars"],
        max_shares=10000,
        max_risk_per_share=params["max_risk_per_share"],
        min_risk_per_share=params["min_risk_per_share"],
        min_risk_reward=params["min_risk_reward"],
        sizing_mode=params["sizing_mode"],
        risk_per_trade=params["risk_per_trade"],
        min_risk_pct=params["min_risk_pct"],
        max_risk_pct=params["max_risk_pct"],
    )


# ---------------------------------------------------------------------------
# Circuit Breaker (Drawdown Management)
# ---------------------------------------------------------------------------


def apply_circuit_breaker(
    trades: List[SimulatedTrade],
    dd_threshold: float = 3000.0,
    pause_trades: int = 2,
) -> List[SimulatedTrade]:
    """
    Apply trailing-drawdown circuit breaker to a chronological list of trades.

    When cumulative P&L drops more than dd_threshold from its peak, the next
    pause_trades trades are skipped. This reduces max drawdown by sitting out
    during losing streaks.

    Args:
        trades: List of SimulatedTrade objects (will be sorted by entry_time)
        dd_threshold: Dollar drawdown from peak that triggers the pause
        pause_trades: Number of trades to skip after trigger

    Returns:
        Filtered list of trades (skipped trades removed)
    """
    if not trades:
        return []

    sorted_trades = sorted(trades, key=lambda t: t.entry_time)

    kept = []
    cumulative = 0.0
    peak = 0.0
    skip_remaining = 0

    for t in sorted_trades:
        if skip_remaining > 0:
            skip_remaining -= 1
            logger.debug(
                f"  Circuit breaker: skipping {t.symbol} "
                f"({skip_remaining} skips remaining)"
            )
            continue

        kept.append(t)
        cumulative += t.pnl
        if cumulative > peak:
            peak = cumulative

        dd = peak - cumulative
        if dd >= dd_threshold:
            skip_remaining = pause_trades
            logger.info(
                f"  Circuit breaker TRIGGERED: DD ${dd:,.0f} >= ${dd_threshold:,.0f} "
                f"after {t.symbol} — skipping next {pause_trades} trade(s)"
            )

    skipped = len(sorted_trades) - len(kept)
    if skipped > 0:
        logger.info(
            f"  Circuit breaker: {len(kept)} trades kept, {skipped} skipped"
        )

    return kept


# ---------------------------------------------------------------------------
# Hypothesis Runner
# ---------------------------------------------------------------------------


def compute_metrics(trades: List[SimulatedTrade]) -> Dict:
    """
    Compute aggregate metrics from a list of simulated trades.

    Args:
        trades: List of SimulatedTrade objects

    Returns:
        Dict with trade_count, win_rate, total_pnl, avg_win, avg_loss,
        actual_rr, max_drawdown, profit_factor
    """
    if not trades:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "actual_rr": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
        }

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    total_pnl = sum(t.pnl for t in trades)
    win_rate = len(wins) / len(trades) * 100

    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0

    actual_rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    # Max drawdown: largest peak-to-trough in cumulative P&L
    # Sort by entry time so drawdown reflects the real chronological sequence,
    # not the alphabetical symbol order from batch processing.
    sorted_trades = sorted(trades, key=lambda t: t.entry_time)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in sorted_trades:
        cumulative += t.pnl
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    # Profit factor = gross wins / gross losses
    gross_wins = sum(t.pnl for t in wins)
    gross_losses = abs(sum(t.pnl for t in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    return {
        "trade_count": len(trades),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "actual_rr": actual_rr,
        "max_drawdown": -max_dd,
        "profit_factor": profit_factor,
    }


def run_hypothesis(
    hypothesis_id: str,
    movers: List[Tuple[str, date]],
    client: AlpacaClient,
    db,
    realistic: bool = False,
) -> Tuple[Dict, List[SimulatedTrade]]:
    """
    Run a single hypothesis through the backtest engine.

    Args:
        hypothesis_id: Key into HYPOTHESES registry
        movers: List of (symbol, date) pairs to backtest
        client: AlpacaClient for fetching bars
        db: Database for caching
        realistic: Use realistic buy-stop entries instead of fantasy breakout entries

    Returns:
        Tuple of (metrics_dict, list_of_trades)
    """
    params = HYPOTHESES[hypothesis_id]
    planner = build_planner(hypothesis_id)

    # Build detector with optional MACD filter
    detector_kwargs = {}
    if params.get("require_macd_positive"):
        detector_kwargs["require_macd_positive"] = True
    detector = BullFlagDetector(**detector_kwargs) if detector_kwargs else None

    # Partial profit config from hypothesis
    partial_profit_enabled = params.get("partial_profit_enabled", False)
    partial_profit_r_multiple = params.get("partial_profit_r_multiple", 1.0)
    partial_profit_fraction = params.get("partial_profit_fraction", 0.5)

    mode_label = "realistic" if realistic else "fantasy"
    runner = BacktestRunner(
        planner=planner,
        detector=detector,
        realistic=realistic,
        min_price=2.0 if realistic else 0.0,
        partial_profit_enabled=partial_profit_enabled,
        partial_profit_r_multiple=partial_profit_r_multiple,
        partial_profit_fraction=partial_profit_fraction,
    )

    logger.info(
        f"--- Running {hypothesis_id}: {params['description']} [{mode_label}] "
        f"(sizing={params['sizing_mode']}, risk=${params['risk_per_trade']}, "
        f"R:R={params['min_risk_reward']}) ---"
    )

    results = run_batch_backtest(movers, client, runner, db=db)

    all_trades = [t for r in results for t in r.trades_simulated]

    # Apply circuit breaker if configured
    cb_dd = params.get("circuit_breaker_dd")
    cb_pause = params.get("circuit_breaker_pause")
    if cb_dd is not None and cb_pause is not None:
        pre_count = len(all_trades)
        all_trades = apply_circuit_breaker(
            all_trades, dd_threshold=cb_dd, pause_trades=cb_pause
        )
        logger.info(
            f"{hypothesis_id}: circuit breaker DD>${cb_dd:,} skip {cb_pause} — "
            f"{pre_count} → {len(all_trades)} trades"
        )

    metrics = compute_metrics(all_trades)
    metrics["hypothesis"] = hypothesis_id
    metrics["description"] = params["description"]

    logger.info(
        f"{hypothesis_id} [{mode_label}] complete: {metrics['trade_count']} trades, "
        f"{metrics['win_rate']:.1f}% WR, ${metrics['total_pnl']:+.2f} P&L, "
        f"R:R {metrics['actual_rr']:.2f}:1"
    )

    return metrics, all_trades


# ---------------------------------------------------------------------------
# Price Bucket Analysis
# ---------------------------------------------------------------------------

PRICE_BUCKETS = [
    ("$2-5", 2.0, 5.0),
    ("$5-10", 5.0, 10.0),
    ("$10-20", 10.0, 20.0),
    ("$20+", 20.0, float('inf')),
]


def compute_price_bucket_metrics(trades: List[SimulatedTrade]) -> List[Dict]:
    """
    Break down trade metrics by entry price bucket.

    Args:
        trades: List of SimulatedTrade objects

    Returns:
        List of dicts with bucket_name, trade_count, win_rate, actual_rr
    """
    buckets = []
    for name, low, high in PRICE_BUCKETS:
        bucket_trades = [t for t in trades if low <= t.entry_price < high]
        if not bucket_trades:
            continue
        metrics = compute_metrics(bucket_trades)
        buckets.append({
            "bucket": name,
            "trade_count": len(bucket_trades),
            "win_rate": metrics["win_rate"],
            "actual_rr": metrics["actual_rr"],
            "total_pnl": metrics["total_pnl"],
        })
    return buckets


def print_price_bucket_analysis(hypothesis_id: str, trades: List[SimulatedTrade]) -> None:
    """
    Print price bucket breakdown for a hypothesis.

    Args:
        hypothesis_id: Hypothesis identifier
        trades: List of SimulatedTrade objects
    """
    buckets = compute_price_bucket_metrics(trades)
    if not buckets:
        print(f"{hypothesis_id} by price: no trades")
        return

    parts = []
    for b in buckets:
        parts.append(
            f"{b['bucket']} ({b['trade_count']} trades, "
            f"{b['win_rate']:.0f}% WR, {b['actual_rr']:.1f}:1 RR)"
        )
    print(f"{hypothesis_id} by price: {' | '.join(parts)}")


# ---------------------------------------------------------------------------
# Output: Comparison Table + CSVs
# ---------------------------------------------------------------------------


def print_comparison_table(all_results: Dict[str, Dict]) -> None:
    """
    Print formatted comparison table of all hypothesis results.

    Args:
        all_results: Dict mapping hypothesis_id -> metrics dict
    """
    header = (
        f"{'ID':<4} {'Trades':>6} {'WinRate':>8} {'PnL':>10} "
        f"{'AvgWin':>8} {'AvgLoss':>9} {'ActualRR':>9} "
        f"{'MaxDD':>9} {'ProfitF':>8}"
    )
    print("\n" + "=" * 80)
    print("  HYPOTHESIS COMPARISON")
    print("=" * 80)
    print(header)
    print("-" * 80)

    for h_id in sorted(all_results.keys()):
        m = all_results[h_id]
        rr_str = f"{m['actual_rr']:.2f}:1" if m['actual_rr'] != float('inf') else "inf"
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] != float('inf') else "inf"
        print(
            f"{h_id:<4} {m['trade_count']:>6} {m['win_rate']:>7.1f}% "
            f"${m['total_pnl']:>+9,.0f} ${m['avg_win']:>+7,.0f} "
            f"${m['avg_loss']:>+8,.0f} {rr_str:>9} "
            f"${m['max_drawdown']:>+8,.0f} {pf_str:>8}"
        )

    print("=" * 80 + "\n")


def write_comparison_csv(all_results: Dict[str, Dict], output_path: str) -> None:
    """
    Write hypothesis comparison to CSV.

    Args:
        all_results: Dict mapping hypothesis_id -> metrics dict
        output_path: Path for the CSV file
    """
    headers = [
        "hypothesis", "description", "trade_count", "win_rate",
        "total_pnl", "avg_win", "avg_loss", "actual_rr",
        "max_drawdown", "profit_factor",
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        for h_id in sorted(all_results.keys()):
            row = dict(all_results[h_id])
            row["hypothesis"] = h_id
            writer.writerow(row)

    logger.info(f"Comparison CSV written to {output_path}")


def write_trades_csv(
    hypothesis_id: str,
    trades: List[SimulatedTrade],
    params: Dict,
    output_dir: str,
) -> str:
    """
    Write per-hypothesis trade-level CSV.

    Args:
        hypothesis_id: Hypothesis identifier
        trades: List of SimulatedTrade objects
        params: Hypothesis parameters dict
        output_dir: Directory for output files

    Returns:
        Path to written CSV
    """
    output_path = os.path.join(output_dir, f"{hypothesis_id}_trades.csv")
    headers = [
        "hypothesis", "symbol", "entry_time", "entry_price",
        "planned_entry", "entry_gap",
        "stop_loss", "target", "shares", "exit_time", "exit_price",
        "exit_reason", "pnl", "pnl_pct",
        "sizing_mode", "risk_budget", "risk_per_share",
        "dollar_risk_per_trade", "position_value",
        "partial_taken", "partial_price", "partial_shares", "partial_pnl",
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for t in trades:
            risk_per_share = t.entry_price - t.stop_loss if t.stop_loss else 0
            position_value = t.entry_price * t.shares
            planned = t.planned_entry if t.planned_entry is not None else t.entry_price
            writer.writerow([
                hypothesis_id,
                t.symbol,
                t.entry_time,
                f"{t.entry_price:.2f}",
                f"{planned:.2f}",
                f"{t.entry_gap:.4f}",
                f"{t.stop_loss:.2f}",
                f"{t.take_profit:.2f}",
                t.shares,
                t.exit_time,
                f"{t.exit_price:.2f}" if t.exit_price else "",
                t.exit_reason or "",
                f"{t.pnl:.2f}",
                f"{t.pnl_pct:.2f}",
                params["sizing_mode"],
                f"{params['risk_per_trade']:.0f}",
                f"{risk_per_share:.4f}",
                f"{risk_per_share * t.shares:.2f}",
                f"{position_value:.2f}",
                t.partial_exit_taken,
                f"{t.partial_exit_price:.2f}" if t.partial_exit_price else "",
                t.partial_shares,
                f"{t.partial_pnl:.2f}",
            ])

    logger.info(f"{hypothesis_id}: {len(trades)} trades written to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for risk model research."""
    parser = argparse.ArgumentParser(
        description="Risk model research: test position sizing, stop thresholds, and R:R optimization"
    )
    parser.add_argument(
        "--hypothesis", type=str, default=None,
        help="Comma-separated hypothesis IDs to run (default: all)"
    )
    parser.add_argument(
        "--start", type=str, default="2026-02-01",
        help="Start date YYYY-MM-DD (default: 2026-02-01)"
    )
    parser.add_argument(
        "--end", type=str, default="2026-03-13",
        help="End date YYYY-MM-DD (default: 2026-03-13)"
    )
    parser.add_argument(
        "--capital", type=float, default=50000,
        help="Capital for position sizing (default: 50000)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for CSVs (default: results)"
    )
    parser.add_argument(
        "--realistic", action="store_true",
        help="Use realistic buy-stop entries (detect_setup + pending order simulation)"
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

    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    # Determine which hypotheses to run
    if args.hypothesis:
        hypothesis_ids = [h.strip() for h in args.hypothesis.split(",")]
        for h_id in hypothesis_ids:
            if h_id not in HYPOTHESES:
                logger.error(f"Unknown hypothesis '{h_id}'. Valid: {list(HYPOTHESES.keys())}")
                sys.exit(1)
    else:
        hypothesis_ids = sorted(HYPOTHESES.keys())

    # Override capital if specified
    if args.capital != 50000:
        for h_id in hypothesis_ids:
            HYPOTHESES[h_id]["position_size_dollars"] = args.capital

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Risk Research: {start_date} to {end_date}")
    logger.info(f"Hypotheses to test: {hypothesis_ids}")

    # Step 1: Load universe + find movers (shared across all hypotheses)
    db = get_database()
    universe = db.get_active_universe()
    symbols = [s['symbol'] for s in universe]
    logger.info(f"Loaded {len(symbols)} symbols from active universe")

    if not symbols:
        logger.error("No active symbols in universe — run universe builder first")
        sys.exit(1)

    client = AlpacaClient(api_key=api_key, api_secret=api_secret)
    logger.info("Fetching daily bars for date range (cache-first)...")
    daily_bars = fetch_daily_bars_cached(symbols, start_date, end_date, client, db)
    movers = find_big_movers(daily_bars)

    if not movers:
        logger.warning("No symbols with 10%+ intraday move found — nothing to test")
        sys.exit(0)

    logger.info(f"Found {len(movers)} mover/date pairs to backtest per hypothesis")

    # Step 2: Run each hypothesis
    all_results = {}
    all_trades = {}

    if args.realistic:
        logger.info("REALISTIC MODE: Using detect_setup() + pending buy-stop simulation")

    for h_id in hypothesis_ids:
        metrics, trades = run_hypothesis(
            h_id, movers, client, db, realistic=args.realistic
        )
        all_results[h_id] = metrics
        all_trades[h_id] = trades

        # Write per-hypothesis trade CSV
        write_trades_csv(h_id, trades, HYPOTHESES[h_id], args.output_dir)

    # Step 3: Output
    print_comparison_table(all_results)

    print("\nPRICE BUCKET ANALYSIS")
    print("-" * 80)
    for h_id in hypothesis_ids:
        print_price_bucket_analysis(h_id, all_trades[h_id])

    # Write comparison CSV
    comparison_path = os.path.join(args.output_dir, "hypothesis_comparison.csv")
    write_comparison_csv(all_results, comparison_path)

    logger.info(f"Research complete. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
