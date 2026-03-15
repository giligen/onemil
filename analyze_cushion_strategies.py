"""
Cushion-based progressive sizing strategy analysis.

Tests 10+ hypotheses inspired by Ross Cameron's approach:
- Start small, build a cushion, then scale up
- Scale down or stop on bad days
- No binary regime filter — continuous throttle

Uses the 15-month backtest CSV (670 trades) to simulate each strategy.
Trades are processed in chronological order within each day.
Position sizing is expressed as a multiplier on the original trade's shares/PnL.

Goal: Kill the Mar-May 2025 drawdown without overfitting.
Metric: Max drawdown, Sharpe-like ratio, total PnL, win rate.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import sys


def load_trades(csv_path: str) -> pd.DataFrame:
    """Load and prepare trade data."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['entry_time'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['entry_time_et'])
    df['pnl'] = df['pnl'].astype(float)
    df['shares'] = df['shares'].astype(float)
    df['entry_price'] = df['entry_price'].astype(float)
    df['exit_price'] = df['exit_price'].astype(float)
    # Sort by date then entry time
    df = df.sort_values(['date', 'entry_time']).reset_index(drop=True)
    return df


def compute_metrics(daily_pnls: list, trade_pnls: list, trade_count: int,
                    original_trade_count: int) -> dict:
    """Compute strategy metrics from daily and trade-level PnLs."""
    cum = np.cumsum(daily_pnls)
    total_pnl = sum(daily_pnls)
    peak = np.maximum.accumulate(cum)
    drawdowns = cum - peak
    max_dd = abs(min(drawdowns)) if len(drawdowns) > 0 else 0

    wins = sum(1 for p in trade_pnls if p > 0)
    losses = sum(1 for p in trade_pnls if p <= 0)
    wr = wins / len(trade_pnls) * 100 if trade_pnls else 0

    avg_win = np.mean([p for p in trade_pnls if p > 0]) if wins > 0 else 0
    avg_loss = np.mean([p for p in trade_pnls if p <= 0]) if losses > 0 else 0

    # Profit factor
    gross_profit = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Daily Sharpe (annualized)
    if len(daily_pnls) > 1 and np.std(daily_pnls) > 0:
        sharpe = (np.mean(daily_pnls) / np.std(daily_pnls)) * np.sqrt(252)
    else:
        sharpe = 0

    # Mar-May 2025 PnL (the problem period)
    # We'll compute this separately in the caller

    return {
        'total_pnl': total_pnl,
        'max_dd': max_dd,
        'win_rate': wr,
        'trades': trade_count,
        'trades_taken_pct': trade_count / original_trade_count * 100,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': pf,
        'sharpe': sharpe,
        'pnl_per_trade': total_pnl / trade_count if trade_count > 0 else 0,
        'calmar': total_pnl / max_dd if max_dd > 0 else float('inf'),
    }


def compute_mar_may_pnl(daily_pnls_by_date: dict) -> float:
    """Sum PnL for Mar-May 2025."""
    from datetime import date
    total = 0
    for d, pnl in daily_pnls_by_date.items():
        if date(2025, 3, 1) <= d <= date(2025, 5, 31):
            total += pnl
    return total


def compute_max_dd_mar_may(daily_pnls_by_date: dict) -> float:
    """Max drawdown specifically within Mar-May 2025."""
    from datetime import date
    pnls = [pnl for d, pnl in sorted(daily_pnls_by_date.items())
            if date(2025, 3, 1) <= d <= date(2025, 5, 31)]
    if not pnls:
        return 0
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    drawdowns = cum - peak
    return abs(min(drawdowns)) if len(drawdowns) > 0 else 0


# =============================================================================
# HYPOTHESIS STRATEGIES
# =============================================================================

def H0_baseline(df: pd.DataFrame) -> tuple:
    """H0: Baseline — no cushion, no scaling. All trades at 100%."""
    trade_pnls = []
    daily_pnls_by_date = defaultdict(float)

    for _, row in df.iterrows():
        pnl = row['pnl']
        trade_pnls.append(pnl)
        daily_pnls_by_date[row['date']] += pnl

    daily_pnls = [daily_pnls_by_date[d] for d in sorted(daily_pnls_by_date)]
    return trade_pnls, daily_pnls, daily_pnls_by_date


def H1_start_half_scale_after_win(df: pd.DataFrame) -> tuple:
    """H1: Start at 50%, scale to 100% after first winning trade of the day."""
    trade_pnls = []
    daily_pnls_by_date = defaultdict(float)

    for trade_date, day_trades in df.groupby('date'):
        day_trades = day_trades.sort_values('entry_time')
        has_cushion = False

        for _, row in day_trades.iterrows():
            scale = 1.0 if has_cushion else 0.5
            pnl = row['pnl'] * scale
            trade_pnls.append(pnl)
            daily_pnls_by_date[trade_date] += pnl

            if row['pnl'] > 0:
                has_cushion = True

    daily_pnls = [daily_pnls_by_date[d] for d in sorted(daily_pnls_by_date)]
    return trade_pnls, daily_pnls, daily_pnls_by_date


def H2_start_half_scale_after_cushion_500(df: pd.DataFrame) -> tuple:
    """H2: Start at 50%, scale to 100% after cumulative daily PnL > $500."""
    trade_pnls = []
    daily_pnls_by_date = defaultdict(float)

    for trade_date, day_trades in df.groupby('date'):
        day_trades = day_trades.sort_values('entry_time')
        cum_pnl = 0.0

        for _, row in day_trades.iterrows():
            scale = 1.0 if cum_pnl >= 500 else 0.5
            pnl = row['pnl'] * scale
            trade_pnls.append(pnl)
            daily_pnls_by_date[trade_date] += pnl
            cum_pnl += pnl

    daily_pnls = [daily_pnls_by_date[d] for d in sorted(daily_pnls_by_date)]
    return trade_pnls, daily_pnls, daily_pnls_by_date


def H3_start_half_stop_after_2_losses(df: pd.DataFrame) -> tuple:
    """H3: Start at 50%, scale to 100% after first win. Stop after 2 consecutive losses."""
    trade_pnls = []
    daily_pnls_by_date = defaultdict(float)

    for trade_date, day_trades in df.groupby('date'):
        day_trades = day_trades.sort_values('entry_time')
        has_cushion = False
        consecutive_losses = 0
        stopped = False

        for _, row in day_trades.iterrows():
            if stopped:
                continue  # skip rest of day

            scale = 1.0 if has_cushion else 0.5
            pnl = row['pnl'] * scale
            trade_pnls.append(pnl)
            daily_pnls_by_date[trade_date] += pnl

            if row['pnl'] > 0:
                has_cushion = True
                consecutive_losses = 0
            else:
                consecutive_losses += 1
                if consecutive_losses >= 2:
                    stopped = True

    daily_pnls = [daily_pnls_by_date[d] for d in sorted(daily_pnls_by_date)]
    return trade_pnls, daily_pnls, daily_pnls_by_date


def H4_three_tier_scaling(df: pd.DataFrame) -> tuple:
    """H4: Three tiers — 50% start, 100% after cushion > 0, 150% after cushion > $1000."""
    trade_pnls = []
    daily_pnls_by_date = defaultdict(float)

    for trade_date, day_trades in df.groupby('date'):
        day_trades = day_trades.sort_values('entry_time')
        cum_pnl = 0.0

        for _, row in day_trades.iterrows():
            if cum_pnl >= 1000:
                scale = 1.5
            elif cum_pnl > 0:
                scale = 1.0
            else:
                scale = 0.5
            pnl = row['pnl'] * scale
            trade_pnls.append(pnl)
            daily_pnls_by_date[trade_date] += pnl
            cum_pnl += pnl

    daily_pnls = [daily_pnls_by_date[d] for d in sorted(daily_pnls_by_date)]
    return trade_pnls, daily_pnls, daily_pnls_by_date


def H5_cb_1500_skip_rest_of_day(df: pd.DataFrame) -> tuple:
    """H5: No scaling, but stop trading for the day after $1500 drawdown."""
    trade_pnls = []
    daily_pnls_by_date = defaultdict(float)

    for trade_date, day_trades in df.groupby('date'):
        day_trades = day_trades.sort_values('entry_time')
        cum_pnl = 0.0
        peak_pnl = 0.0
        stopped = False

        for _, row in day_trades.iterrows():
            if stopped:
                continue

            pnl = row['pnl']
            trade_pnls.append(pnl)
            daily_pnls_by_date[trade_date] += pnl
            cum_pnl += pnl
            peak_pnl = max(peak_pnl, cum_pnl)

            if peak_pnl - cum_pnl >= 1500:
                stopped = True

    daily_pnls = [daily_pnls_by_date[d] for d in sorted(daily_pnls_by_date)]
    return trade_pnls, daily_pnls, daily_pnls_by_date


def H6_cushion_plus_cb(df: pd.DataFrame) -> tuple:
    """H6: Start at 50%, scale to 100% after first win, CB $1500 stops the day."""
    trade_pnls = []
    daily_pnls_by_date = defaultdict(float)

    for trade_date, day_trades in df.groupby('date'):
        day_trades = day_trades.sort_values('entry_time')
        has_cushion = False
        cum_pnl = 0.0
        peak_pnl = 0.0
        stopped = False

        for _, row in day_trades.iterrows():
            if stopped:
                continue

            scale = 1.0 if has_cushion else 0.5
            pnl = row['pnl'] * scale
            trade_pnls.append(pnl)
            daily_pnls_by_date[trade_date] += pnl
            cum_pnl += pnl
            peak_pnl = max(peak_pnl, cum_pnl)

            if row['pnl'] > 0:
                has_cushion = True

            if peak_pnl - cum_pnl >= 1500:
                stopped = True

    daily_pnls = [daily_pnls_by_date[d] for d in sorted(daily_pnls_by_date)]
    return trade_pnls, daily_pnls, daily_pnls_by_date


def H7_scale_down_after_loss(df: pd.DataFrame) -> tuple:
    """H7: Start at 100%. After each loss, reduce by 25% (floor 25%). After each win, restore to 100%."""
    trade_pnls = []
    daily_pnls_by_date = defaultdict(float)

    for trade_date, day_trades in df.groupby('date'):
        day_trades = day_trades.sort_values('entry_time')
        scale = 1.0

        for _, row in day_trades.iterrows():
            pnl = row['pnl'] * scale
            trade_pnls.append(pnl)
            daily_pnls_by_date[trade_date] += pnl

            if row['pnl'] > 0:
                scale = 1.0  # reset on win
            else:
                scale = max(0.25, scale - 0.25)  # reduce on loss

    daily_pnls = [daily_pnls_by_date[d] for d in sorted(daily_pnls_by_date)]
    return trade_pnls, daily_pnls, daily_pnls_by_date


def H8_stop_after_daily_loss_1000(df: pd.DataFrame) -> tuple:
    """H8: Stop trading for the day once cumulative daily PnL < -$1000."""
    trade_pnls = []
    daily_pnls_by_date = defaultdict(float)

    for trade_date, day_trades in df.groupby('date'):
        day_trades = day_trades.sort_values('entry_time')
        cum_pnl = 0.0
        stopped = False

        for _, row in day_trades.iterrows():
            if stopped:
                continue

            pnl = row['pnl']
            trade_pnls.append(pnl)
            daily_pnls_by_date[trade_date] += pnl
            cum_pnl += pnl

            if cum_pnl <= -1000:
                stopped = True

    daily_pnls = [daily_pnls_by_date[d] for d in sorted(daily_pnls_by_date)]
    return trade_pnls, daily_pnls, daily_pnls_by_date


def H9_previous_day_momentum(df: pd.DataFrame) -> tuple:
    """H9: Scale based on previous day's result.
    After a losing day: start at 50%. After a winning day: start at 100%.
    Scale up to 100%/150% after first win of current day."""
    trade_pnls = []
    daily_pnls_by_date = defaultdict(float)

    dates = sorted(df['date'].unique())
    prev_day_pnl = 0  # neutral for first day

    for trade_date in dates:
        day_trades = df[df['date'] == trade_date].sort_values('entry_time')

        # Previous day determines base scale
        if prev_day_pnl < 0:
            base_scale = 0.5
            scaled_scale = 1.0
        else:
            base_scale = 1.0
            scaled_scale = 1.0  # no extra scaling after winning day

        has_win = False
        cum_pnl = 0.0

        for _, row in day_trades.iterrows():
            scale = scaled_scale if has_win else base_scale
            pnl = row['pnl'] * scale
            trade_pnls.append(pnl)
            daily_pnls_by_date[trade_date] += pnl
            cum_pnl += pnl

            if row['pnl'] > 0:
                has_win = True

        prev_day_pnl = cum_pnl

    daily_pnls = [daily_pnls_by_date[d] for d in sorted(daily_pnls_by_date)]
    return trade_pnls, daily_pnls, daily_pnls_by_date


def H10_rolling_3day_momentum(df: pd.DataFrame) -> tuple:
    """H10: Scale based on rolling 3-day PnL.
    If rolling 3-day PnL < -$1000: trade at 50%.
    If rolling 3-day PnL > $2000: trade at 100%.
    Otherwise: 75%."""
    trade_pnls = []
    daily_pnls_by_date = defaultdict(float)

    dates = sorted(df['date'].unique())
    daily_results = {}

    for trade_date in dates:
        day_trades = df[df['date'] == trade_date].sort_values('entry_time')

        # Compute rolling 3-day PnL from PREVIOUS days
        past_dates = [d for d in sorted(daily_results.keys()) if d < trade_date]
        recent_3 = past_dates[-3:] if len(past_dates) >= 3 else past_dates
        rolling_pnl = sum(daily_results[d] for d in recent_3)

        if rolling_pnl < -1000:
            scale = 0.5
        elif rolling_pnl > 2000:
            scale = 1.0
        else:
            scale = 0.75

        cum_pnl = 0.0
        for _, row in day_trades.iterrows():
            pnl = row['pnl'] * scale
            trade_pnls.append(pnl)
            daily_pnls_by_date[trade_date] += pnl
            cum_pnl += pnl

        daily_results[trade_date] = cum_pnl

    daily_pnls = [daily_pnls_by_date[d] for d in sorted(daily_pnls_by_date)]
    return trade_pnls, daily_pnls, daily_pnls_by_date


def H11_cushion_with_max_loss_cap(df: pd.DataFrame) -> tuple:
    """H11: Start at 50%. Scale to 100% after first win.
    Hard stop at -$2000 daily. Scale back to 50% after any loss."""
    trade_pnls = []
    daily_pnls_by_date = defaultdict(float)

    for trade_date, day_trades in df.groupby('date'):
        day_trades = day_trades.sort_values('entry_time')
        scale = 0.5
        cum_pnl = 0.0
        stopped = False

        for _, row in day_trades.iterrows():
            if stopped:
                continue

            pnl = row['pnl'] * scale
            trade_pnls.append(pnl)
            daily_pnls_by_date[trade_date] += pnl
            cum_pnl += pnl

            if row['pnl'] > 0:
                scale = 1.0
            else:
                scale = 0.5  # drop back after any loss

            if cum_pnl <= -2000:
                stopped = True

    daily_pnls = [daily_pnls_by_date[d] for d in sorted(daily_pnls_by_date)]
    return trade_pnls, daily_pnls, daily_pnls_by_date


def H12_adaptive_streak(df: pd.DataFrame) -> tuple:
    """H12: Scale = 0.5 + 0.25 * (consecutive wins, max 2). Reset to 0.5 on loss.
    So: 50% → 75% after 1 win → 100% after 2 wins. Any loss resets to 50%."""
    trade_pnls = []
    daily_pnls_by_date = defaultdict(float)

    for trade_date, day_trades in df.groupby('date'):
        day_trades = day_trades.sort_values('entry_time')
        streak = 0  # consecutive wins

        for _, row in day_trades.iterrows():
            scale = min(1.0, 0.5 + 0.25 * streak)
            pnl = row['pnl'] * scale
            trade_pnls.append(pnl)
            daily_pnls_by_date[trade_date] += pnl

            if row['pnl'] > 0:
                streak = min(streak + 1, 2)
            else:
                streak = 0

    daily_pnls = [daily_pnls_by_date[d] for d in sorted(daily_pnls_by_date)]
    return trade_pnls, daily_pnls, daily_pnls_by_date


def H13_first_trade_probe(df: pd.DataFrame) -> tuple:
    """H13: First trade at 33%. If win, next at 66%, then 100%.
    If first trade loses, second at 33% again. 2 losses in a row = stop."""
    trade_pnls = []
    daily_pnls_by_date = defaultdict(float)

    for trade_date, day_trades in df.groupby('date'):
        day_trades = day_trades.sort_values('entry_time')
        tier = 0  # 0=33%, 1=66%, 2=100%
        consecutive_losses = 0
        stopped = False

        for _, row in day_trades.iterrows():
            if stopped:
                continue

            scales = [0.33, 0.66, 1.0]
            scale = scales[min(tier, 2)]
            pnl = row['pnl'] * scale
            trade_pnls.append(pnl)
            daily_pnls_by_date[trade_date] += pnl

            if row['pnl'] > 0:
                tier = min(tier + 1, 2)
                consecutive_losses = 0
            else:
                consecutive_losses += 1
                if consecutive_losses >= 2:
                    stopped = True

    daily_pnls = [daily_pnls_by_date[d] for d in sorted(daily_pnls_by_date)]
    return trade_pnls, daily_pnls, daily_pnls_by_date


def H14_half_size_all_day(df: pd.DataFrame) -> tuple:
    """H14: Simple control — all trades at 50%. Tests pure size reduction."""
    trade_pnls = []
    daily_pnls_by_date = defaultdict(float)

    for _, row in df.iterrows():
        pnl = row['pnl'] * 0.5
        trade_pnls.append(pnl)
        daily_pnls_by_date[row['date']] += pnl

    daily_pnls = [daily_pnls_by_date[d] for d in sorted(daily_pnls_by_date)]
    return trade_pnls, daily_pnls, daily_pnls_by_date


def H15_cushion_scale_cb_combo(df: pd.DataFrame) -> tuple:
    """H15: COMBO — Start at 50%, scale to 100% after $500 cushion,
    CB stops at -$1500 from peak, 2 consecutive losses at 50% = stop.
    This combines H2+H6+H3 elements."""
    trade_pnls = []
    daily_pnls_by_date = defaultdict(float)

    for trade_date, day_trades in df.groupby('date'):
        day_trades = day_trades.sort_values('entry_time')
        cum_pnl = 0.0
        peak_pnl = 0.0
        consecutive_losses_at_base = 0
        stopped = False

        for _, row in day_trades.iterrows():
            if stopped:
                continue

            scale = 1.0 if cum_pnl >= 500 else 0.5
            pnl = row['pnl'] * scale
            trade_pnls.append(pnl)
            daily_pnls_by_date[trade_date] += pnl
            cum_pnl += pnl
            peak_pnl = max(peak_pnl, cum_pnl)

            # CB check
            if peak_pnl - cum_pnl >= 1500:
                stopped = True

            # 2 losses at base size = stop
            if scale <= 0.5:
                if row['pnl'] <= 0:
                    consecutive_losses_at_base += 1
                    if consecutive_losses_at_base >= 2:
                        stopped = True
                else:
                    consecutive_losses_at_base = 0

    daily_pnls = [daily_pnls_by_date[d] for d in sorted(daily_pnls_by_date)]
    return trade_pnls, daily_pnls, daily_pnls_by_date


# =============================================================================
# MAIN
# =============================================================================

def main():
    csv_path = 'C:/Work/onemil/full_15mo.csv'
    df = load_trades(csv_path)
    print(f"Loaded {len(df)} trades across {df['date'].nunique()} trading days")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print()

    hypotheses = [
        ("H0:  Baseline (no scaling)", H0_baseline),
        ("H1:  50%→100% after first win", H1_start_half_scale_after_win),
        ("H2:  50%→100% after $500 cushion", H2_start_half_scale_after_cushion_500),
        ("H3:  50%→100% on win, stop after 2 losses", H3_start_half_stop_after_2_losses),
        ("H4:  50%→100%→150% three tiers", H4_three_tier_scaling),
        ("H5:  CB $1500 stop day (no scaling)", H5_cb_1500_skip_rest_of_day),
        ("H6:  50%→100% on win + CB $1500", H6_cushion_plus_cb),
        ("H7:  100%, scale down 25% per loss", H7_scale_down_after_loss),
        ("H8:  Stop after daily PnL < -$1000", H8_stop_after_daily_loss_1000),
        ("H9:  Scale 50% after losing day", H9_previous_day_momentum),
        ("H10: Rolling 3-day momentum scaling", H10_rolling_3day_momentum),
        ("H11: 50%→100%, back to 50% on loss, stop -$2K", H11_cushion_with_max_loss_cap),
        ("H12: Streak-based: 50%→75%→100%", H12_adaptive_streak),
        ("H13: Probe 33%→66%→100%, stop 2 losses", H13_first_trade_probe),
        ("H14: Control — all trades at 50%", H14_half_size_all_day),
        ("H15: Combo: 50%+$500 cushion+CB+2L stop", H15_cushion_scale_cb_combo),
    ]

    results = []
    for name, func in hypotheses:
        trade_pnls, daily_pnls, daily_pnls_by_date = func(df)
        metrics = compute_metrics(daily_pnls, trade_pnls, len(trade_pnls), len(df))
        metrics['name'] = name
        metrics['mar_may_pnl'] = compute_mar_may_pnl(daily_pnls_by_date)
        metrics['mar_may_dd'] = compute_max_dd_mar_may(daily_pnls_by_date)
        results.append(metrics)

    # Print results table
    print(f"{'Hypothesis':<48} {'Total PnL':>10} {'MaxDD':>8} {'MM PnL':>10} {'MM DD':>8} {'WR%':>6} {'Trades':>7} {'PF':>6} {'Sharpe':>7} {'Calmar':>7}")
    print("=" * 148)

    for r in results:
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 100 else "inf"
        print(f"{r['name']:<48} ${r['total_pnl']:>9,.0f} ${r['max_dd']:>6,.0f} ${r['mar_may_pnl']:>9,.0f} ${r['mar_may_dd']:>6,.0f} {r['win_rate']:>5.1f} {r['trades']:>7} {pf_str:>6} {r['sharpe']:>7.2f} {r['calmar']:>7.1f}")

    print()
    print("Legend: MM PnL = Mar-May 2025 PnL, MM DD = Mar-May 2025 Max Drawdown")
    print("        PF = Profit Factor, Calmar = Total PnL / Max DD")
    print()

    # Rank by Calmar ratio (PnL/DD tradeoff)
    print("=== RANKED BY CALMAR (PnL/DD efficiency) ===")
    ranked = sorted(results, key=lambda r: r['calmar'], reverse=True)
    for i, r in enumerate(ranked[:5], 1):
        print(f"  #{i}: {r['name']} — Calmar={r['calmar']:.1f}, PnL=${r['total_pnl']:,.0f}, MaxDD=${r['max_dd']:,.0f}, MM_DD=${r['mar_may_dd']:,.0f}")

    print()
    print("=== RANKED BY MAR-MAY DRAWDOWN REDUCTION ===")
    baseline_mm_dd = results[0]['mar_may_dd']
    ranked_mm = sorted(results, key=lambda r: r['mar_may_dd'])
    for i, r in enumerate(ranked_mm[:5], 1):
        reduction = (1 - r['mar_may_dd'] / baseline_mm_dd) * 100 if baseline_mm_dd > 0 else 0
        print(f"  #{i}: {r['name']} — MM_DD=${r['mar_may_dd']:,.0f} ({reduction:.0f}% reduction), Total PnL=${r['total_pnl']:,.0f}")

    print()
    print("=== RANKED BY TOTAL PnL (must still be profitable) ===")
    ranked_pnl = sorted(results, key=lambda r: r['total_pnl'], reverse=True)
    for i, r in enumerate(ranked_pnl[:5], 1):
        print(f"  #{i}: {r['name']} — PnL=${r['total_pnl']:,.0f}, MaxDD=${r['max_dd']:,.0f}, Calmar={r['calmar']:.1f}")


if __name__ == '__main__':
    main()
