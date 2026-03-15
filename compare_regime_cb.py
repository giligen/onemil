"""
Compare backtest results: baseline vs regime filter + circuit breaker.

Runs two backtests (baseline already cached from prior run) and produces
a month-by-month comparison of trades, win rate, P&L, and max drawdown.

Usage:
    python3 compare_regime_cb.py --start 2025-01-01 --end 2026-03-14
"""

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone, date, timedelta
from typing import List, Tuple, Dict

import pandas as pd
from dotenv import load_dotenv

from backtest import BacktestRunner, BacktestResult
from batch_backtest import (
    fetch_daily_bars_cached, find_big_movers, run_batch_backtest,
    get_1min_bars_cached, write_csv_report,
)
from data_sources.alpaca_client import AlpacaClient
from persistence.database import get_database
from trading.market_regime import MarketRegimeFilter

logger = logging.getLogger(__name__)


def collect_monthly_stats(results: List[BacktestResult]) -> Dict[str, Dict]:
    """
    Aggregate backtest results into monthly buckets.

    Returns dict keyed by 'YYYY-MM' with:
        trades, wins, losses, win_rate, pnl, max_drawdown, avg_win, avg_loss
    """
    monthly = defaultdict(lambda: {
        'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0,
        'win_pnl': 0.0, 'loss_pnl': 0.0,
        'cumulative': [],  # list of trade P&Ls for drawdown calc
    })

    for result in results:
        for trade in result.trades_simulated:
            # Use trade_date from result (YYYY-MM-DD string)
            month_key = result.trade_date[:7]  # 'YYYY-MM'
            m = monthly[month_key]
            m['trades'] += 1
            m['pnl'] += trade.pnl
            m['cumulative'].append(trade.pnl)
            if trade.pnl > 0:
                m['wins'] += 1
                m['win_pnl'] += trade.pnl
            else:
                m['losses'] += 1
                m['loss_pnl'] += trade.pnl

    # Calculate derived stats
    for month, m in monthly.items():
        m['win_rate'] = m['wins'] / m['trades'] * 100 if m['trades'] > 0 else 0
        m['avg_win'] = m['win_pnl'] / m['wins'] if m['wins'] > 0 else 0
        m['avg_loss'] = m['loss_pnl'] / m['losses'] if m['losses'] > 0 else 0

        # Max drawdown from cumulative P&L within the month
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in m['cumulative']:
            cum += pnl
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd
        m['max_drawdown'] = max_dd

    return dict(monthly)


def print_comparison(baseline: Dict[str, Dict], filtered: Dict[str, Dict]) -> None:
    """Print side-by-side month-by-month comparison."""
    all_months = sorted(set(list(baseline.keys()) + list(filtered.keys())))

    empty = {
        'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
        'pnl': 0, 'max_drawdown': 0, 'avg_win': 0, 'avg_loss': 0,
    }

    print("\n" + "=" * 120)
    print("  MONTH-BY-MONTH COMPARISON: BASELINE vs REGIME + CIRCUIT BREAKER")
    print("=" * 120)
    print(f"  {'Month':<10} | {'--- BASELINE ---':^42} | {'--- REGIME + CB ---':^42} | {'Delta':^12}")
    print(f"  {'':10} | {'Trades':>6} {'WR%':>6} {'P&L':>12} {'MaxDD':>10} | {'Trades':>6} {'WR%':>6} {'P&L':>12} {'MaxDD':>10} | {'P&L':>12}")
    print("-" * 120)

    tot_b = {'trades': 0, 'pnl': 0.0, 'wins': 0, 'losses': 0, 'max_dd': 0.0}
    tot_f = {'trades': 0, 'pnl': 0.0, 'wins': 0, 'losses': 0, 'max_dd': 0.0}

    for month in all_months:
        b = baseline.get(month, empty)
        f = filtered.get(month, empty)
        delta_pnl = f['pnl'] - b['pnl']

        print(
            f"  {month:<10} | "
            f"{b['trades']:>6} {b['win_rate']:>5.1f}% ${b['pnl']:>+10,.0f} ${b['max_drawdown']:>9,.0f} | "
            f"{f['trades']:>6} {f['win_rate']:>5.1f}% ${f['pnl']:>+10,.0f} ${f['max_drawdown']:>9,.0f} | "
            f"${delta_pnl:>+10,.0f}"
        )

        tot_b['trades'] += b['trades']
        tot_b['pnl'] += b['pnl']
        tot_b['wins'] += b.get('wins', 0)
        tot_b['losses'] += b.get('losses', 0)
        tot_b['max_dd'] = max(tot_b['max_dd'], b['max_drawdown'])

        tot_f['trades'] += f['trades']
        tot_f['pnl'] += f['pnl']
        tot_f['wins'] += f.get('wins', 0)
        tot_f['losses'] += f.get('losses', 0)
        tot_f['max_dd'] = max(tot_f['max_dd'], f['max_drawdown'])

    print("-" * 120)
    b_wr = tot_b['wins'] / tot_b['trades'] * 100 if tot_b['trades'] > 0 else 0
    f_wr = tot_f['wins'] / tot_f['trades'] * 100 if tot_f['trades'] > 0 else 0
    delta_total = tot_f['pnl'] - tot_b['pnl']
    print(
        f"  {'TOTAL':<10} | "
        f"{tot_b['trades']:>6} {b_wr:>5.1f}% ${tot_b['pnl']:>+10,.0f} ${tot_b['max_dd']:>9,.0f} | "
        f"{tot_f['trades']:>6} {f_wr:>5.1f}% ${tot_f['pnl']:>+10,.0f} ${tot_f['max_dd']:>9,.0f} | "
        f"${delta_total:>+10,.0f}"
    )
    print("=" * 120)

    # Compute overall max drawdown across entire period
    for label, results_monthly in [("BASELINE", baseline), ("REGIME+CB", filtered)]:
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for month in all_months:
            m = results_monthly.get(month, empty)
            for pnl in m.get('cumulative', []):
                cum += pnl
                if cum > peak:
                    peak = cum
                dd = peak - cum
                if dd > max_dd:
                    max_dd = dd
        print(f"  {label} overall max drawdown: ${max_dd:,.0f}")

    print()


def main():
    """Run comparison backtest."""
    parser = argparse.ArgumentParser(description="Compare baseline vs regime+CB backtest")
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--end", type=str, default="2026-03-14")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        logger.error("Missing ALPACA_API_KEY or ALPACA_API_SECRET")
        sys.exit(1)

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    logger.info(f"Comparison backtest: {start_date} to {end_date}")

    # Load data (shared between both runs)
    db = get_database()
    universe = db.get_active_universe()
    symbols = [s['symbol'] for s in universe]
    logger.info(f"Loaded {len(symbols)} symbols")

    client = AlpacaClient(api_key=api_key, api_secret=api_secret)
    daily_bars = fetch_daily_bars_cached(symbols, start_date, end_date, client, db)
    universe_dict = {s['symbol']: s for s in universe}

    from config import Config
    cfg = Config._load_yaml_only()
    scanner_cfg = cfg.get("scanner", {})
    trading_cfg = cfg.get("trading", {})

    movers = find_big_movers(
        daily_bars,
        universe_dict=universe_dict,
        price_min=float(scanner_cfg.get("price_min", 2.0)),
        price_max=float(scanner_cfg.get("price_max", 20.0)),
        float_max=int(scanner_cfg.get("float_max", 10_000_000)),
    )

    if not movers:
        logger.error("No movers found")
        return

    runner = BacktestRunner()

    # ---- Run 1: BASELINE (no regime, no CB) ----
    logger.info("=" * 60)
    logger.info("RUN 1: BASELINE (no regime filter, no circuit breaker)")
    logger.info("=" * 60)
    baseline_results = run_batch_backtest(
        movers, client, runner, db=db, universe_dict=universe_dict,
    )

    # ---- Run 2: REGIME + CIRCUIT BREAKER ----
    logger.info("=" * 60)
    logger.info("RUN 2: REGIME FILTER + CIRCUIT BREAKER")
    logger.info("=" * 60)

    # Load SPY bars for regime filter
    spy_bars_raw = fetch_daily_bars_cached(['SPY'], start_date - timedelta(days=20), end_date, client, db)
    spy_bars = spy_bars_raw.get('SPY', [])

    regime_cfg = trading_cfg.get("market_regime", {})
    regime = MarketRegimeFilter(
        enabled=regime_cfg.get("enabled", True),
        spy_5d_return_min=float(regime_cfg.get("spy_5d_return_min", -2.0)),
    )
    regime.load_spy_bars(spy_bars)

    cb_dd = float(trading_cfg.get("circuit_breaker_dd", 1500.0))
    cb_pause = int(trading_cfg.get("circuit_breaker_pause", 1))
    logger.info(f"Regime: SPY 5d min = {regime.spy_5d_return_min}%, CB: dd=${cb_dd}, pause={cb_pause}")

    filtered_results = run_batch_backtest(
        movers, client, runner, db=db, universe_dict=universe_dict,
        market_regime=regime,
        circuit_breaker_dd=cb_dd,
        circuit_breaker_pause=cb_pause,
    )

    # ---- Compare ----
    baseline_monthly = collect_monthly_stats(baseline_results)
    filtered_monthly = collect_monthly_stats(filtered_results)

    print_comparison(baseline_monthly, filtered_monthly)

    # Write CSVs
    write_csv_report(baseline_results, "backtest_baseline.csv")
    write_csv_report(filtered_results, "backtest_regime_cb.csv")
    logger.info("CSVs written: backtest_baseline.csv, backtest_regime_cb.csv")


if __name__ == "__main__":
    main()
