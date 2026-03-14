"""
Compare rvol filtering approaches across the 15-month backtest.

Runs three variants:
1. no_rvol: No relative volume filter (what production scanner actually does
   due to timezone mismatch bug in bucket rvol)
2. cumulative: Cumulative rvol at entry time (Ross Cameron's definition)
3. bucket: Fixed bucket rvol using correct UTC-keyed volume profiles

Usage:
    python compare_rvol_modes.py
"""

import logging
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from backtest import BacktestRunner
from batch_backtest import (
    fetch_daily_bars_cached,
    find_big_movers,
    get_1min_bars_cached,
    run_batch_backtest,
)
from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database, get_database
from config import Config

logger = logging.getLogger(__name__)

# Month ranges for the 15-month comparison
MONTHS = [
    (date(2025, 1, 1), date(2025, 1, 31)),
    (date(2025, 2, 1), date(2025, 2, 28)),
    (date(2025, 3, 1), date(2025, 3, 31)),
    (date(2025, 4, 1), date(2025, 4, 30)),
    (date(2025, 5, 1), date(2025, 5, 31)),
    (date(2025, 6, 1), date(2025, 6, 30)),
    (date(2025, 7, 1), date(2025, 7, 31)),
    (date(2025, 8, 1), date(2025, 8, 31)),
    (date(2025, 9, 1), date(2025, 9, 30)),
    (date(2025, 10, 1), date(2025, 10, 31)),
    (date(2025, 11, 1), date(2025, 11, 30)),
    (date(2025, 12, 1), date(2025, 12, 31)),
    (date(2026, 1, 1), date(2026, 1, 31)),
    (date(2026, 2, 1), date(2026, 2, 28)),
    (date(2026, 3, 1), date(2026, 3, 14)),
]


def run_month_with_mode(
    month_start, month_end, client, db, universe_dict, volume_profiles,
    rvol_mode, rvol_min,
):
    """Run backtest for one month with a specific rvol mode."""
    cfg = Config._load_yaml_only()
    scanner_cfg = cfg.get("scanner", {})

    symbols = list(universe_dict.keys())
    daily_bars = fetch_daily_bars_cached(symbols, month_start, month_end, client, db)

    movers = find_big_movers(
        daily_bars,
        universe_dict=universe_dict,
        price_min=float(scanner_cfg.get("price_min", 2.0)),
        price_max=float(scanner_cfg.get("price_max", 20.0)),
        float_max=int(scanner_cfg.get("float_max", 10_000_000)),
    )

    if not movers:
        return [], 0

    # Create runner with specific rvol mode
    runner = BacktestRunner(rvol_mode=rvol_mode)
    # Override rvol min if needed (0 = disabled)
    runner.relative_volume_min = rvol_min

    results = run_batch_backtest(
        movers, client, runner, db=db,
        universe_dict=universe_dict,
        volume_profiles=volume_profiles,
    )

    trades = []
    for r in results:
        for t in r.trades_simulated:
            trades.append({
                'symbol': t.symbol,
                'date': r.trade_date,
                'pnl': t.pnl,
                'exit_reason': t.exit_reason,
            })

    total_pnl = sum(t['pnl'] for t in trades)
    return trades, total_pnl


def main():
    """Run the three-way rvol comparison."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    client = AlpacaClient(api_key=api_key, api_secret=api_secret)
    db = Database()

    # Load universe and volume profiles once
    universe = db.get_active_universe()
    universe_dict = {s['symbol']: s for s in universe}
    volume_profiles = db.get_all_volume_profiles()
    print(f"Universe: {len(universe_dict)} stocks, Volume profiles: {len(volume_profiles)} stocks")

    modes = [
        ('no_rvol', 'cumulative', 0.0),      # No filter (production reality)
        ('cumulative', 'cumulative', 5.0),     # Ross's definition
        ('bucket_fixed', 'bucket', 5.0),       # Correct bucket rvol
    ]

    # Collect results per mode per month
    all_results = {name: [] for name, _, _ in modes}

    for month_start, month_end in MONTHS:
        label = f"{month_start.year}-{month_start.month:02d}"
        month_pnls = {}

        for mode_name, rvol_mode, rvol_min in modes:
            t0 = time.time()
            trades, total_pnl = run_month_with_mode(
                month_start, month_end, client, db,
                universe_dict, volume_profiles,
                rvol_mode, rvol_min,
            )
            elapsed = time.time() - t0

            all_results[mode_name].append({
                'month': label,
                'trades': len(trades),
                'pnl': total_pnl,
                'wins': sum(1 for t in trades if t['pnl'] > 0),
                'losses': sum(1 for t in trades if t['pnl'] <= 0),
            })
            month_pnls[mode_name] = (len(trades), total_pnl)

        # Print month results
        print(f"\n{label}:", end="")
        for mode_name, _, _ in modes:
            n, pnl = month_pnls[mode_name]
            print(f"  {mode_name}: {n} trades ${pnl:+,.0f}", end="")
        print()

    # Print summary
    print("\n" + "=" * 90)
    print(f"{'RVOL MODE COMPARISON — 15 MONTHS':^90}")
    print("=" * 90)

    header = f"{'Month':<10}"
    for name, _, _ in modes:
        header += f"  {'Trades':>6} {'P&L':>10}"
    print(header)
    print("-" * 90)

    totals = {name: {'trades': 0, 'pnl': 0.0, 'wins': 0, 'losses': 0} for name, _, _ in modes}

    for i, (month_start, _) in enumerate(MONTHS):
        label = f"{month_start.year}-{month_start.month:02d}"
        row = f"{label:<10}"
        for name, _, _ in modes:
            r = all_results[name][i]
            row += f"  {r['trades']:>6} ${r['pnl']:>+9,.0f}"
            totals[name]['trades'] += r['trades']
            totals[name]['pnl'] += r['pnl']
            totals[name]['wins'] += r['wins']
            totals[name]['losses'] += r['losses']
        print(row)

    print("-" * 90)
    row = f"{'TOTAL':<10}"
    for name, _, _ in modes:
        t = totals[name]
        row += f"  {t['trades']:>6} ${t['pnl']:>+9,.0f}"
    print(row)

    # Compute Sharpe for each mode
    print("\n" + "=" * 90)
    print(f"{'RISK METRICS':^90}")
    print("=" * 90)
    print(f"{'Metric':<30}", end="")
    for name, _, _ in modes:
        print(f"  {name:>16}", end="")
    print()
    print("-" * 90)

    for name, _, _ in modes:
        t = totals[name]
        monthly_pnls = [r['pnl'] for r in all_results[name]]
        monthly_arr = np.array(monthly_pnls)
        sharpe = (monthly_arr.mean() / monthly_arr.std() * np.sqrt(12)
                  if monthly_arr.std() > 0 else 0)
        win_months = sum(1 for p in monthly_pnls if p > 0)
        max_dd_monthly = 0
        peak = 0
        cumsum = 0
        for p in monthly_pnls:
            cumsum += p
            if cumsum > peak:
                peak = cumsum
            dd = peak - cumsum
            if dd > max_dd_monthly:
                max_dd_monthly = dd

        totals[name]['sharpe'] = sharpe
        totals[name]['win_months'] = win_months
        totals[name]['max_dd'] = max_dd_monthly
        totals[name]['win_rate'] = t['wins'] / t['trades'] * 100 if t['trades'] > 0 else 0
        totals[name]['pf'] = (
            sum(r['pnl'] for r in all_results[name] if r['pnl'] > 0) /
            abs(sum(r['pnl'] for r in all_results[name] if r['pnl'] < 0))
            if any(r['pnl'] < 0 for r in all_results[name]) else float('inf')
        )

    metrics = [
        ('Trades', lambda n: f"{totals[n]['trades']}"),
        ('Total P&L', lambda n: f"${totals[n]['pnl']:,.0f}"),
        ('P&L/trade', lambda n: f"${totals[n]['pnl']/max(totals[n]['trades'],1):,.0f}"),
        ('Win Rate', lambda n: f"{totals[n]['win_rate']:.1f}%"),
        ('Profit Factor', lambda n: f"{totals[n]['pf']:.2f}"),
        ('Sharpe (monthly ann.)', lambda n: f"{totals[n]['sharpe']:.2f}"),
        ('Winning Months', lambda n: f"{totals[n]['win_months']}/15"),
        ('Max Drawdown', lambda n: f"${totals[n]['max_dd']:,.0f}"),
    ]

    for metric_name, fn in metrics:
        row = f"{metric_name:<30}"
        for name, _, _ in modes:
            row += f"  {fn(name):>16}"
        print(row)

    print("\n" + "=" * 90)
    print("RECOMMENDATION:")
    best = max(modes, key=lambda m: totals[m[0]]['sharpe'])
    print(f"  Best risk-adjusted: {best[0]} (Sharpe {totals[best[0]]['sharpe']:.2f})")
    print(f"  Production scanner bug: rvol bucket keys are UTC but scanner uses ET hours")
    print(f"  Fix the scanner OR switch to cumulative rvol — whichever backtests better")
    print("=" * 90)


if __name__ == "__main__":
    main()
