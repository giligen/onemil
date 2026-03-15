"""
Compare backtest results at 3 slippage levels: 0/0, 0.03/0.02, 0.05/0.05.

Usage:
    python3 compare_slippage.py --start 2025-01-01 --end 2026-03-14
"""

import argparse
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, date, timedelta
from typing import List, Dict

from dotenv import load_dotenv

from backtest import BacktestRunner, BacktestResult
from batch_backtest import (
    fetch_daily_bars_cached, find_big_movers, run_batch_backtest,
)
from data_sources.alpaca_client import AlpacaClient
from persistence.database import get_database

logger = logging.getLogger(__name__)


def monthly_stats(results: List[BacktestResult]) -> Dict[str, Dict]:
    """Aggregate results by month."""
    monthly = defaultdict(lambda: {
        'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0,
        'win_pnl': 0.0, 'loss_pnl': 0.0, 'trade_pnls': [],
    })

    for r in results:
        for t in r.trades_simulated:
            m = monthly[r.trade_date[:7]]
            m['trades'] += 1
            m['pnl'] += t.pnl
            m['trade_pnls'].append(t.pnl)
            if t.pnl > 0:
                m['wins'] += 1
                m['win_pnl'] += t.pnl
            else:
                m['losses'] += 1
                m['loss_pnl'] += t.pnl

    for m in monthly.values():
        m['wr'] = m['wins'] / m['trades'] * 100 if m['trades'] > 0 else 0
        m['avg_win'] = m['win_pnl'] / m['wins'] if m['wins'] > 0 else 0
        m['avg_loss'] = m['loss_pnl'] / m['losses'] if m['losses'] > 0 else 0
        # Max drawdown
        cum = peak = 0.0
        max_dd = 0.0
        for pnl in m['trade_pnls']:
            cum += pnl
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)
        m['max_dd'] = max_dd

    return dict(monthly)


def overall_max_dd(results: List[BacktestResult]) -> float:
    """Compute overall max drawdown across entire period."""
    cum = peak = 0.0
    max_dd = 0.0
    for r in sorted(results, key=lambda x: x.trade_date):
        for t in r.trades_simulated:
            cum += t.pnl
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)
    return max_dd


def main():
    parser = argparse.ArgumentParser(description="Compare slippage levels")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-03-14")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    # Load shared data
    db = get_database()
    universe = db.get_active_universe()
    symbols = [s['symbol'] for s in universe]
    client = AlpacaClient(api_key=api_key, api_secret=api_secret)
    daily_bars = fetch_daily_bars_cached(symbols, start_date, end_date, client, db)
    universe_dict = {s['symbol']: s for s in universe}

    from config import Config
    cfg = Config._load_yaml_only()
    scanner_cfg = cfg.get("scanner", {})

    movers = find_big_movers(
        daily_bars, universe_dict=universe_dict,
        price_min=float(scanner_cfg.get("price_min", 2.0)),
        price_max=float(scanner_cfg.get("price_max", 20.0)),
        float_max=int(scanner_cfg.get("float_max", 10_000_000)),
    )

    # 3 slippage scenarios
    scenarios = [
        ("0/0 (none)", 0.0, 0.0),
        ("0.03/0.02 (base)", 0.03, 0.02),
        ("0.05/0.05 (stress)", 0.05, 0.05),
    ]

    all_results = {}
    for label, entry_slip, exit_slip in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"SCENARIO: {label} — entry=${entry_slip}, exit=${exit_slip}")
        logger.info(f"{'='*60}")
        runner = BacktestRunner(entry_slippage=entry_slip, exit_slippage=exit_slip)
        results = run_batch_backtest(movers, client, runner, db=db, universe_dict=universe_dict)
        all_results[label] = results

    # Print comparison
    all_monthly = {label: monthly_stats(r) for label, r in all_results.items()}
    all_months = sorted(set(m for stats in all_monthly.values() for m in stats))

    print("\n" + "=" * 140)
    print("  SLIPPAGE IMPACT COMPARISON")
    print("=" * 140)
    print(f"  {'Month':<10}", end="")
    for label, _, _ in scenarios:
        print(f" | {'Tr':>4} {'WR%':>5} {'P&L':>11} {'MaxDD':>9}", end="")
    print()
    print("-" * 140)

    totals = {label: {'trades': 0, 'wins': 0, 'pnl': 0.0} for label, _, _ in scenarios}
    empty = {'trades': 0, 'wr': 0, 'pnl': 0, 'max_dd': 0}

    for month in all_months:
        print(f"  {month:<10}", end="")
        for label, _, _ in scenarios:
            m = all_monthly[label].get(month, empty)
            print(f" | {m['trades']:>4} {m['wr']:>4.1f}% ${m['pnl']:>+9,.0f} ${m['max_dd']:>8,.0f}", end="")
            totals[label]['trades'] += m['trades']
            totals[label]['wins'] += m.get('wins', 0)
            totals[label]['pnl'] += m['pnl']
        print()

    print("-" * 140)
    print(f"  {'TOTAL':<10}", end="")
    for label, _, _ in scenarios:
        t = totals[label]
        wr = t['wins'] / t['trades'] * 100 if t['trades'] > 0 else 0
        print(f" | {t['trades']:>4} {wr:>4.1f}% ${t['pnl']:>+9,.0f} {'':>9}", end="")
    print()
    print("=" * 140)

    # Overall stats
    print()
    for label, _, _ in scenarios:
        results = all_results[label]
        trades = [t for r in results for t in r.trades_simulated]
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in trades)
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
        max_dd = overall_max_dd(results)

        print(f"  {label}:")
        print(f"    Trades: {len(trades)}, WR: {len(wins)/len(trades)*100:.1f}%, "
              f"P&L: ${total_pnl:+,.0f}")
        print(f"    Avg win: ${avg_win:+,.0f}, Avg loss: ${avg_loss:+,.0f}, "
              f"W:L ratio: {abs(avg_win/avg_loss):.2f}:1" if avg_loss != 0 else "")
        print(f"    Max drawdown: ${max_dd:,.0f}")
        print()


if __name__ == "__main__":
    main()
