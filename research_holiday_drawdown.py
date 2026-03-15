"""
Holiday Drawdown Research — analyze thin-liquidity filters to reduce holiday-driven MaxDD.

The regime filter reduced MaxDD from ~$29K to $19K, but the remaining MaxDD
($19,080, Dec 17 2025 -> Jan 8 2026) is entirely holiday-driven: 41 trades
at 22% WR (vs 40.6% overall). These are structurally different trades —
thin liquidity causes immediate stop-outs (many exit within 1-3 bars).

Goal: find generalizable, structurally-justified filters that reduce holiday
drawdown WITHOUT overfitting to specific dates. No hardcoded date ranges —
filters must be based on observable market signals applicable to ALL days.

Hypotheses tested:
  H1: SPY volume ratio (market-wide liquidity proxy)
  H2: Breakout bar volume ratio (entry_bar_volume / avg_flag_volume)
  H3: Absolute pole volume
  H4: Quick stop-out diagnostic (bars_held <= 3 on stops)
  H5: Combined H1 + H2

Usage:
    python research_holiday_drawdown.py [--verbose]
"""

import argparse
import os
import sqlite3
import sys
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Holiday period for diagnostic comparison (NOT used in filters)
HOLIDAY_START = date(2025, 12, 17)
HOLIDAY_END = date(2026, 1, 8)

TRADES_CSV = 'backtest_results/backtest_full_2025_01_to_2026_03.csv'
DB_PATH = 'data/onemil.db'
OUTPUT_DIR = 'research_results'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'holiday_filter_analysis.csv')


def load_trades(csv_path: str = TRADES_CSV) -> pd.DataFrame:
    """Load backtest trades from rich CSV export."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['pnl'] = df['pnl'].astype(float)
    df['entry_time'] = pd.to_datetime(
        df['date'].astype(str) + ' ' + df['entry_time_et']
    )
    df = df.sort_values(['date', 'entry_time']).reset_index(drop=True)
    print(f"Loaded {len(df)} trades from {df['date'].min()} to {df['date'].max()}")
    return df


def load_spy_volume(db_path: str = DB_PATH) -> pd.DataFrame:
    """Query SQLite for SPY daily bars (date, volume)."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT bar_date AS date, volume
        FROM daily_bars
        WHERE symbol = 'SPY'
        ORDER BY bar_date
    """, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    print(f"Loaded {len(df)} SPY daily bars ({df['date'].min()} to {df['date'].max()})")
    return df


def enrich_trades(trades_df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich each trade with SPY volume ratio on T-1 (day before trade).

    SPY volume ratio = spy_volume(T-1) / SMA20(spy_volume ending at T-1).
    Uses T-1 to avoid lookahead bias.
    """
    spy = spy_df.sort_values('date').reset_index(drop=True)
    spy['volume_sma20'] = spy['volume'].rolling(20).mean()

    # Build lookup: trade_date -> T-1 SPY volume ratio
    spy_dates = spy['date'].tolist()
    spy_lookup = {}
    for i, row in spy.iterrows():
        d = row['date']
        vol = row['volume']
        sma20 = row['volume_sma20']
        if pd.notna(sma20) and sma20 > 0:
            spy_lookup[d] = {
                'spy_volume': vol,
                'spy_volume_sma20': sma20,
                'spy_volume_ratio': vol / sma20,
            }

    # For each trade, find T-1 (the trading day before trade_date)
    enriched = trades_df.copy()
    spy_vol_ratios = []
    spy_volumes = []
    spy_vol_sma20s = []

    for _, trade in enriched.iterrows():
        trade_date = trade['date']
        # Find the last SPY date strictly before trade_date
        prior_dates = [d for d in spy_dates if d < trade_date]
        if prior_dates:
            t_minus_1 = prior_dates[-1]
            info = spy_lookup.get(t_minus_1)
            if info:
                spy_vol_ratios.append(info['spy_volume_ratio'])
                spy_volumes.append(info['spy_volume'])
                spy_vol_sma20s.append(info['spy_volume_sma20'])
                continue
        spy_vol_ratios.append(np.nan)
        spy_volumes.append(np.nan)
        spy_vol_sma20s.append(np.nan)

    enriched['spy_volume_t1'] = spy_volumes
    enriched['spy_volume_sma20'] = spy_vol_sma20s
    enriched['spy_volume_ratio'] = spy_vol_ratios

    valid = enriched['spy_volume_ratio'].notna().sum()
    print(f"Enriched {valid}/{len(enriched)} trades with SPY volume ratio")
    return enriched


def compute_max_drawdown(pnl_series: pd.Series) -> float:
    """
    Compute max drawdown from a sequence of trade PnLs.

    Returns the maximum peak-to-trough decline in cumulative equity.
    """
    if pnl_series.empty:
        return 0.0
    cum = pnl_series.cumsum()
    peak = cum.cummax()
    drawdown = peak - cum
    return drawdown.max()


def compute_sharpe(trades_df: pd.DataFrame) -> float:
    """
    Compute annualized Sharpe ratio from monthly PnL.

    Groups trades by month, computes monthly PnL std, returns
    annualized Sharpe = (mean_monthly * 12) / (std_monthly * sqrt(12)).
    """
    if trades_df.empty:
        return 0.0
    df = trades_df.copy()
    df['month'] = df['date'].apply(lambda d: d.replace(day=1))
    monthly = df.groupby('month')['pnl'].sum()
    if len(monthly) < 2:
        return 0.0
    mean_m = monthly.mean()
    std_m = monthly.std()
    if std_m == 0:
        return 0.0
    return (mean_m * 12) / (std_m * np.sqrt(12))


def is_holiday_period(d: date) -> bool:
    """Check if date falls in the known holiday drawdown period (diagnostic only)."""
    return HOLIDAY_START <= d <= HOLIDAY_END


def test_filter(
    trades_df: pd.DataFrame,
    mask_keep: pd.Series,
    filter_name: str,
    verbose: bool = False,
) -> Dict:
    """
    Evaluate one filter: PnL, MaxDD, Sharpe, selectivity ratio.

    Args:
        trades_df: Full enriched trades DataFrame.
        mask_keep: Boolean mask — True = keep trade, False = filter out.
        filter_name: Name of the filter for logging.
        verbose: Print detailed breakdown.

    Returns:
        Dict with filter metrics.
    """
    total = len(trades_df)
    kept = trades_df[mask_keep]
    removed = trades_df[~mask_keep]

    # Holiday breakdown (diagnostic)
    holiday_mask = trades_df['date'].apply(is_holiday_period)
    holiday_total = holiday_mask.sum()
    holiday_removed = (~mask_keep & holiday_mask).sum()
    nonholiday_total = total - holiday_total
    nonholiday_removed = (~mask_keep & ~holiday_mask).sum()

    # Core metrics on kept trades
    pnl_kept = kept['pnl'].sum()
    pnl_removed = removed['pnl'].sum()
    maxdd_kept = compute_max_drawdown(kept['pnl'])
    sharpe_kept = compute_sharpe(kept)

    # Win rate
    wr_kept = (kept['pnl'] > 0).mean() * 100 if len(kept) > 0 else 0.0

    # Selectivity ratio: % removed from holiday / % removed overall
    pct_removed_overall = len(removed) / total * 100 if total > 0 else 0
    pct_removed_holiday = holiday_removed / holiday_total * 100 if holiday_total > 0 else 0
    selectivity = (pct_removed_holiday / pct_removed_overall
                   if pct_removed_overall > 0 else 0)

    result = {
        'filter': filter_name,
        'trades_kept': len(kept),
        'trades_removed': len(removed),
        'pct_removed': pct_removed_overall,
        'holiday_removed': holiday_removed,
        'holiday_total': holiday_total,
        'nonholiday_removed': nonholiday_removed,
        'nonholiday_total': nonholiday_total,
        'pnl_kept': pnl_kept,
        'pnl_removed': pnl_removed,
        'maxdd_kept': maxdd_kept,
        'sharpe_kept': sharpe_kept,
        'wr_kept': wr_kept,
        'selectivity': selectivity,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Filter: {filter_name}")
        print(f"  Kept: {len(kept)}/{total} trades "
              f"({len(removed)} removed, {pct_removed_overall:.1f}%)")
        print(f"  Holiday removed: {holiday_removed}/{holiday_total} "
              f"({pct_removed_holiday:.1f}%)")
        print(f"  Non-holiday removed: {nonholiday_removed}/{nonholiday_total} "
              f"({nonholiday_removed/nonholiday_total*100:.1f}%)" if nonholiday_total > 0 else "")
        print(f"  Selectivity ratio: {selectivity:.2f}x")
        print(f"  PnL kept: ${pnl_kept:,.0f}  |  PnL removed: ${pnl_removed:,.0f}")
        print(f"  MaxDD kept: ${maxdd_kept:,.0f}")
        print(f"  Sharpe kept: {sharpe_kept:.2f}")
        print(f"  WR kept: {wr_kept:.1f}%")

    return result


def run_sweep(trades_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Test all hypotheses with parameter sweeps.

    Returns DataFrame with all filter results.
    """
    results = []

    # Baseline (no filter)
    baseline_mask = pd.Series([True] * len(trades_df), index=trades_df.index)
    results.append(test_filter(trades_df, baseline_mask, "BASELINE", verbose))

    # --- H1: SPY volume ratio ---
    for threshold in [0.65, 0.70, 0.75, 0.80, 0.85]:
        mask = trades_df['spy_volume_ratio'] >= threshold
        # Keep trades where SPY volume ratio is ABOVE threshold (healthy liquidity)
        # NaN = missing data -> keep (safe default)
        mask = mask | trades_df['spy_volume_ratio'].isna()
        name = f"H1: SPY_vol_ratio >= {threshold:.2f}"
        results.append(test_filter(trades_df, mask, name, verbose))

    # --- H2: Breakout bar volume ratio ---
    if 'entry_bar_volume' in trades_df.columns and 'avg_flag_volume' in trades_df.columns:
        trades_df['breakout_vol_ratio'] = (
            trades_df['entry_bar_volume'] / trades_df['avg_flag_volume']
        )
        for threshold in [1.5, 2.0, 2.5, 3.0, 3.5]:
            mask = trades_df['breakout_vol_ratio'] >= threshold
            mask = mask | trades_df['breakout_vol_ratio'].isna()
            name = f"H2: breakout_vol_ratio >= {threshold:.1f}"
            results.append(test_filter(trades_df, mask, name, verbose))

    # --- H3: Absolute pole volume ---
    if 'avg_pole_volume' in trades_df.columns:
        for threshold in [500, 1000, 2000, 3000, 5000]:
            mask = trades_df['avg_pole_volume'] >= threshold
            mask = mask | trades_df['avg_pole_volume'].isna()
            name = f"H3: avg_pole_volume >= {threshold}"
            results.append(test_filter(trades_df, mask, name, verbose))

    # --- H4: Quick stop-out diagnostic ---
    # Not a filter — just measure concentration of quick stops in holiday vs non-holiday
    if 'bars_held' in trades_df.columns and 'exit_reason' in trades_df.columns:
        quick_stops = (
            (trades_df['bars_held'] <= 3) &
            (trades_df['exit_reason'] == 'stop')
        )
        holiday_mask = trades_df['date'].apply(is_holiday_period)
        qs_holiday = (quick_stops & holiday_mask).sum()
        qs_nonholiday = (quick_stops & ~holiday_mask).sum()
        qs_total = quick_stops.sum()
        total_holiday = holiday_mask.sum()
        total_nonholiday = (~holiday_mask).sum()
        print(f"\n{'='*60}")
        print(f"H4 DIAGNOSTIC: Quick stop-outs (bars_held <= 3 AND exit=stop)")
        print(f"  Holiday:     {qs_holiday}/{total_holiday} trades "
              f"({qs_holiday/total_holiday*100:.1f}%)" if total_holiday > 0 else "")
        print(f"  Non-holiday: {qs_nonholiday}/{total_nonholiday} trades "
              f"({qs_nonholiday/total_nonholiday*100:.1f}%)" if total_nonholiday > 0 else "")
        print(f"  Total:       {qs_total}/{len(trades_df)} trades "
              f"({qs_total/len(trades_df)*100:.1f}%)")
        print(f"  Holiday concentration: "
              f"{qs_holiday/qs_total*100:.1f}% of all quick stops are in holiday"
              if qs_total > 0 else "  No quick stops found")

    # --- H5: Combined H1 + H2 ---
    if 'breakout_vol_ratio' in trades_df.columns:
        for spy_thresh in [0.70, 0.75, 0.80]:
            for bv_thresh in [2.0, 2.5, 3.0]:
                # Keep if EITHER spy vol is healthy OR breakout vol is strong
                # Block only when BOTH are weak
                spy_ok = (trades_df['spy_volume_ratio'] >= spy_thresh) | trades_df['spy_volume_ratio'].isna()
                bv_ok = (trades_df['breakout_vol_ratio'] >= bv_thresh) | trades_df['breakout_vol_ratio'].isna()
                # Keep = at least one signal is OK
                mask = spy_ok | bv_ok
                name = f"H5: SPY>={spy_thresh:.2f} OR bv>={bv_thresh:.1f}"
                results.append(test_filter(trades_df, mask, name, verbose))

        # Also test AND logic: block if EITHER is weak
        for spy_thresh in [0.70, 0.75, 0.80]:
            for bv_thresh in [2.0, 2.5, 3.0]:
                spy_ok = (trades_df['spy_volume_ratio'] >= spy_thresh) | trades_df['spy_volume_ratio'].isna()
                bv_ok = (trades_df['breakout_vol_ratio'] >= bv_thresh) | trades_df['breakout_vol_ratio'].isna()
                # Keep = BOTH signals are OK
                mask = spy_ok & bv_ok
                name = f"H5: SPY>={spy_thresh:.2f} AND bv>={bv_thresh:.1f}"
                results.append(test_filter(trades_df, mask, name, verbose))

    return pd.DataFrame(results)


def print_summary_table(results_df: pd.DataFrame) -> None:
    """Print formatted recommendation table sorted by MaxDD reduction vs PnL impact."""
    print(f"\n{'='*100}")
    print("SUMMARY TABLE — sorted by MaxDD (ascending)")
    print(f"{'='*100}")

    baseline = results_df[results_df['filter'] == 'BASELINE'].iloc[0]
    baseline_pnl = baseline['pnl_kept']
    baseline_dd = baseline['maxdd_kept']

    df = results_df.copy()
    df['dd_reduction'] = baseline_dd - df['maxdd_kept']
    df['pnl_delta'] = df['pnl_kept'] - baseline_pnl
    df['pnl_delta_pct'] = df['pnl_delta'] / abs(baseline_pnl) * 100 if baseline_pnl != 0 else 0

    df = df.sort_values('maxdd_kept')

    header = (
        f"{'Filter':<40} {'Kept':>5} {'Rm%':>5} "
        f"{'PnL':>10} {'dPnL':>8} {'MaxDD':>8} {'dDD':>7} "
        f"{'Sharpe':>6} {'WR%':>5} {'Sel':>5}"
    )
    print(header)
    print('-' * len(header))

    for _, row in df.iterrows():
        line = (
            f"{row['filter']:<40} "
            f"{row['trades_kept']:>5.0f} "
            f"{row['pct_removed']:>4.1f}% "
            f"${row['pnl_kept']:>9,.0f} "
            f"${row['pnl_delta']:>7,.0f} "
            f"${row['maxdd_kept']:>7,.0f} "
            f"${row['dd_reduction']:>6,.0f} "
            f"{row['sharpe_kept']:>6.2f} "
            f"{row['wr_kept']:>4.1f}% "
            f"{row['selectivity']:>5.2f}"
        )
        print(line)


def print_recommendation(results_df: pd.DataFrame) -> None:
    """Print top recommendation based on best DD reduction with acceptable PnL cost."""
    baseline = results_df[results_df['filter'] == 'BASELINE'].iloc[0]
    baseline_pnl = baseline['pnl_kept']
    baseline_dd = baseline['maxdd_kept']

    candidates = results_df[results_df['filter'] != 'BASELINE'].copy()
    candidates['dd_reduction'] = baseline_dd - candidates['maxdd_kept']
    candidates['pnl_delta'] = candidates['pnl_kept'] - baseline_pnl

    # Filter: must reduce DD, and PnL cost < 10% of baseline
    good = candidates[
        (candidates['dd_reduction'] > 0) &
        (candidates['pnl_delta'] > -abs(baseline_pnl) * 0.10)
    ]

    if good.empty:
        print("\nNo filter improves MaxDD without excessive PnL cost.")
        return

    # Rank by DD reduction (higher = better), break ties by PnL preserved
    best = good.sort_values(['dd_reduction', 'pnl_kept'], ascending=[False, False]).iloc[0]

    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")
    print(f"  Filter:       {best['filter']}")
    print(f"  Trades kept:  {best['trades_kept']:.0f} "
          f"({best['trades_removed']:.0f} removed, {best['pct_removed']:.1f}%)")
    print(f"  PnL:          ${best['pnl_kept']:,.0f} "
          f"(delta: ${best['pnl_delta']:,.0f})")
    print(f"  MaxDD:        ${best['maxdd_kept']:,.0f} "
          f"(reduced by ${best['dd_reduction']:,.0f})")
    print(f"  Sharpe:       {best['sharpe_kept']:.2f}")
    print(f"  Selectivity:  {best['selectivity']:.2f}x "
          f"(higher = more targeted at holiday trades)")


def main():
    """Orchestrate holiday drawdown research."""
    parser = argparse.ArgumentParser(description='Holiday drawdown filter research')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed per-filter breakdown')
    args = parser.parse_args()

    print("=" * 60)
    print("Holiday Drawdown Research")
    print("=" * 60)

    # Load data
    trades = load_trades()
    spy = load_spy_volume()
    trades = enrich_trades(trades, spy)

    # Quick holiday stats
    holiday_mask = trades['date'].apply(is_holiday_period)
    h_trades = trades[holiday_mask]
    nh_trades = trades[~holiday_mask]
    print(f"\nHoliday period ({HOLIDAY_START} to {HOLIDAY_END}):")
    print(f"  Trades: {len(h_trades)} ({len(h_trades)/len(trades)*100:.1f}%)")
    print(f"  PnL: ${h_trades['pnl'].sum():,.0f}")
    print(f"  WR: {(h_trades['pnl'] > 0).mean()*100:.1f}%")
    print(f"  MaxDD: ${compute_max_drawdown(h_trades['pnl']):,.0f}")
    print(f"\nNon-holiday:")
    print(f"  Trades: {len(nh_trades)}")
    print(f"  PnL: ${nh_trades['pnl'].sum():,.0f}")
    print(f"  WR: {(nh_trades['pnl'] > 0).mean()*100:.1f}%")

    # SPY volume ratio distribution for holiday vs non-holiday
    h_spy = trades[holiday_mask]['spy_volume_ratio'].dropna()
    nh_spy = trades[~holiday_mask]['spy_volume_ratio'].dropna()
    print(f"\nSPY volume ratio distribution:")
    print(f"  Holiday:     mean={h_spy.mean():.3f}, "
          f"min={h_spy.min():.3f}, max={h_spy.max():.3f}")
    print(f"  Non-holiday: mean={nh_spy.mean():.3f}, "
          f"min={nh_spy.min():.3f}, max={nh_spy.max():.3f}")

    # Run all hypothesis sweeps
    results = run_sweep(trades, verbose=args.verbose)

    # Summary table
    print_summary_table(results)

    # Recommendation
    print_recommendation(results)

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
