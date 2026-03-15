"""
Cushion strategy V2 — deeper exploration of top performers from V1.

V1 Key Findings:
- H11 (50%→100% on win, back to 50% on loss, stop at -$2K) best Calmar=12.5
- H2  (50%→100% after $500 cushion) good balance: $158K PnL, $13.7K DD
- H15 (combo) strong but complex
- H7  (scale down 25% per loss) highest PnL ($221K) while cutting DD 21%

This script:
1. Parameterizes the top strategies to find optimal settings
2. Tests sensitivity to thresholds (is $500 magical or robust?)
3. Tests combinations of the best ideas
4. Shows monthly breakdown for top strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from collections import defaultdict


def load_trades(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['entry_time'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['entry_time_et'])
    df['pnl'] = df['pnl'].astype(float)
    df = df.sort_values(['date', 'entry_time']).reset_index(drop=True)
    return df


def compute_metrics(daily_pnls_by_date: dict, trade_pnls: list, total_trades: int) -> dict:
    daily_pnls = [daily_pnls_by_date[d] for d in sorted(daily_pnls_by_date)]
    cum = np.cumsum(daily_pnls)
    total_pnl = sum(daily_pnls)
    peak = np.maximum.accumulate(cum)
    drawdowns = cum - peak
    max_dd = abs(min(drawdowns)) if len(drawdowns) > 0 else 0

    wins = sum(1 for p in trade_pnls if p > 0)
    wr = wins / len(trade_pnls) * 100 if trade_pnls else 0

    gross_profit = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    if len(daily_pnls) > 1 and np.std(daily_pnls) > 0:
        sharpe = (np.mean(daily_pnls) / np.std(daily_pnls)) * np.sqrt(252)
    else:
        sharpe = 0

    # Mar-May PnL and DD
    mm_pnls = [pnl for d, pnl in sorted(daily_pnls_by_date.items())
                if date(2025, 3, 1) <= d <= date(2025, 5, 31)]
    mm_pnl = sum(mm_pnls)
    if mm_pnls:
        mm_cum = np.cumsum(mm_pnls)
        mm_peak = np.maximum.accumulate(mm_cum)
        mm_dd = abs(min(mm_cum - mm_peak))
    else:
        mm_dd = 0

    calmar = total_pnl / max_dd if max_dd > 0 else float('inf')

    return {
        'total_pnl': total_pnl,
        'max_dd': max_dd,
        'mm_pnl': mm_pnl,
        'mm_dd': mm_dd,
        'win_rate': wr,
        'trades': len(trade_pnls),
        'pf': pf,
        'sharpe': sharpe,
        'calmar': calmar,
    }


def run_strategy(df, start_scale, scale_up_on, scale_down_on, max_scale,
                 daily_stop_loss, consec_loss_stop, cushion_threshold,
                 dd_cb=0):
    """
    Generalized cushion strategy.

    Args:
        start_scale: Initial scale for first trade of day (0.0-2.0)
        scale_up_on: 'win' (first win), 'cushion' (cum PnL > threshold), 'streak' (each win adds)
        scale_down_on: 'loss' (any loss drops scale), 'none', 'consec' (2 consec losses)
        max_scale: Maximum scaling factor
        daily_stop_loss: Stop trading if cum daily PnL < this (0 = disabled)
        consec_loss_stop: Stop after N consecutive losses (0 = disabled)
        cushion_threshold: PnL threshold for 'cushion' scale-up mode
    """
    trade_pnls = []
    daily_pnls_by_date = defaultdict(float)

    for trade_date, day_trades in df.groupby('date'):
        day_trades = day_trades.sort_values('entry_time')
        scale = start_scale
        cum_pnl = 0.0
        peak_pnl = 0.0
        consec_losses = 0
        stopped = False
        win_streak = 0

        for _, row in day_trades.iterrows():
            if stopped:
                continue

            pnl = row['pnl'] * scale
            trade_pnls.append(pnl)
            daily_pnls_by_date[trade_date] += pnl
            cum_pnl += pnl
            peak_pnl = max(peak_pnl, cum_pnl)

            if row['pnl'] > 0:
                consec_losses = 0
                win_streak += 1

                # Scale up logic
                if scale_up_on == 'win':
                    scale = min(max_scale, 1.0)
                elif scale_up_on == 'cushion' and cum_pnl >= cushion_threshold:
                    scale = min(max_scale, 1.0)
                elif scale_up_on == 'streak':
                    scale = min(max_scale, start_scale + 0.25 * win_streak)
            else:
                win_streak = 0
                consec_losses += 1

                # Scale down logic
                if scale_down_on == 'loss':
                    scale = start_scale
                elif scale_down_on == 'consec' and consec_losses >= 2:
                    scale = start_scale

                # Consecutive loss stop
                if consec_loss_stop > 0 and consec_losses >= consec_loss_stop:
                    stopped = True

            # Daily stop loss check
            if daily_stop_loss < 0 and cum_pnl <= daily_stop_loss:
                stopped = True

            # DD-based CB (0 = disabled)
            if dd_cb > 0 and peak_pnl - cum_pnl >= dd_cb:
                stopped = True

    daily_pnls = [daily_pnls_by_date[d] for d in sorted(daily_pnls_by_date)]
    return trade_pnls, daily_pnls, daily_pnls_by_date


def main():
    csv_path = 'C:/Work/onemil/full_15mo.csv'
    df = load_trades(csv_path)
    total_trades = len(df)
    print(f"Loaded {len(df)} trades across {df['date'].nunique()} days")
    print()

    # =========================================================================
    # PART 1: Parameter sweeps on top strategies
    # =========================================================================

    print("=" * 130)
    print("PART 1: Start scale sweep (with scale-up on first win, scale-down on loss)")
    print("=" * 130)
    print(f"{'Start%':>7} {'Total PnL':>10} {'MaxDD':>8} {'MM PnL':>10} {'MM DD':>8} {'WR%':>6} {'Trades':>7} {'Calmar':>7} {'Sharpe':>7}")
    print("-" * 100)

    for start in [0.25, 0.33, 0.40, 0.50, 0.60, 0.75]:
        tp, dp, dpd = run_strategy(df, start_scale=start, scale_up_on='win',
                                    scale_down_on='loss', max_scale=1.0,
                                    daily_stop_loss=0, consec_loss_stop=0,
                                    cushion_threshold=0)
        m = compute_metrics(dpd, tp, total_trades)
        print(f"  {start*100:>4.0f}%  ${m['total_pnl']:>9,.0f} ${m['max_dd']:>6,.0f} ${m['mm_pnl']:>9,.0f} ${m['mm_dd']:>6,.0f} {m['win_rate']:>5.1f} {m['trades']:>7} {m['calmar']:>7.1f} {m['sharpe']:>7.2f}")

    print()
    print("=" * 130)
    print("PART 2: Cushion threshold sweep (start at 50%, scale to 100% after $X cushion)")
    print("=" * 130)
    print(f"{'Cushion':>7} {'Total PnL':>10} {'MaxDD':>8} {'MM PnL':>10} {'MM DD':>8} {'WR%':>6} {'Trades':>7} {'Calmar':>7} {'Sharpe':>7}")
    print("-" * 100)

    for cushion in [0, 100, 250, 500, 750, 1000, 1500, 2000]:
        tp, dp, dpd = run_strategy(df, start_scale=0.5, scale_up_on='cushion',
                                    scale_down_on='none', max_scale=1.0,
                                    daily_stop_loss=0, consec_loss_stop=0,
                                    cushion_threshold=cushion)
        m = compute_metrics(dpd, tp, total_trades)
        label = f"${cushion}" if cushion > 0 else "win"
        print(f"  {label:>5}  ${m['total_pnl']:>9,.0f} ${m['max_dd']:>6,.0f} ${m['mm_pnl']:>9,.0f} ${m['mm_dd']:>6,.0f} {m['win_rate']:>5.1f} {m['trades']:>7} {m['calmar']:>7.1f} {m['sharpe']:>7.2f}")

    print()
    print("=" * 130)
    print("PART 3: Scale-down on loss with different start scales + max scale variants")
    print("=" * 130)
    print(f"{'Config':>35} {'Total PnL':>10} {'MaxDD':>8} {'MM PnL':>10} {'MM DD':>8} {'WR%':>6} {'Trades':>7} {'Calmar':>7}")
    print("-" * 120)

    configs = [
        ("50% start, up on win, no down", 0.5, 'win', 'none', 1.0),
        ("50% start, up on win, down on loss", 0.5, 'win', 'loss', 1.0),
        ("50% start, up on win, down 2consec", 0.5, 'win', 'consec', 1.0),
        ("33% start, up streak, no down", 0.33, 'streak', 'none', 1.0),
        ("33% start, up streak, down on loss", 0.33, 'streak', 'loss', 1.0),
        ("50% start, up streak, no down", 0.5, 'streak', 'none', 1.0),
        ("50% start, up streak, down on loss", 0.5, 'streak', 'loss', 1.0),
        ("75% start, up on win, down on loss", 0.75, 'win', 'loss', 1.0),
        ("50% start, up on win, down, max 1.5x", 0.5, 'win', 'loss', 1.5),
        ("50% start, up streak, down, max 1.5x", 0.5, 'streak', 'loss', 1.5),
    ]

    for label, start, up, down, mx in configs:
        tp, dp, dpd = run_strategy(df, start_scale=start, scale_up_on=up,
                                    scale_down_on=down, max_scale=mx,
                                    daily_stop_loss=0, consec_loss_stop=0,
                                    cushion_threshold=0)
        m = compute_metrics(dpd, tp, total_trades)
        print(f"  {label:>33}  ${m['total_pnl']:>9,.0f} ${m['max_dd']:>6,.0f} ${m['mm_pnl']:>9,.0f} ${m['mm_dd']:>6,.0f} {m['win_rate']:>5.1f} {m['trades']:>7} {m['calmar']:>7.1f}")

    print()
    print("=" * 130)
    print("PART 4: Daily stop-loss sweep (no scaling, just stop-loss)")
    print("=" * 130)
    print(f"{'DailyStop':>10} {'Total PnL':>10} {'MaxDD':>8} {'MM PnL':>10} {'MM DD':>8} {'WR%':>6} {'Trades':>7} {'Calmar':>7}")
    print("-" * 100)

    for stop in [0, -500, -750, -1000, -1500, -2000, -2500, -3000]:
        tp, dp, dpd = run_strategy(df, start_scale=1.0, scale_up_on='none',
                                    scale_down_on='none', max_scale=1.0,
                                    daily_stop_loss=stop, consec_loss_stop=0,
                                    cushion_threshold=0)
        m = compute_metrics(dpd, tp, total_trades)
        label = f"${stop}" if stop < 0 else "none"
        print(f"  {label:>8}  ${m['total_pnl']:>9,.0f} ${m['max_dd']:>6,.0f} ${m['mm_pnl']:>9,.0f} ${m['mm_dd']:>6,.0f} {m['win_rate']:>5.1f} {m['trades']:>7} {m['calmar']:>7.1f}")

    print()
    print("=" * 130)
    print("PART 5: Consecutive loss stop sweep (no scaling)")
    print("=" * 130)
    print(f"{'ConsecStop':>10} {'Total PnL':>10} {'MaxDD':>8} {'MM PnL':>10} {'MM DD':>8} {'WR%':>6} {'Trades':>7} {'Calmar':>7}")
    print("-" * 100)

    for stop in [0, 2, 3, 4, 5]:
        tp, dp, dpd = run_strategy(df, start_scale=1.0, scale_up_on='none',
                                    scale_down_on='none', max_scale=1.0,
                                    daily_stop_loss=0, consec_loss_stop=stop,
                                    cushion_threshold=0)
        m = compute_metrics(dpd, tp, total_trades)
        label = f"{stop}" if stop > 0 else "none"
        print(f"  {label:>8}  ${m['total_pnl']:>9,.0f} ${m['max_dd']:>6,.0f} ${m['mm_pnl']:>9,.0f} ${m['mm_dd']:>6,.0f} {m['win_rate']:>5.1f} {m['trades']:>7} {m['calmar']:>7.1f}")

    print()
    print("=" * 130)
    print("PART 6: BEST COMBOS — combining cushion + CB + stop-loss")
    print("=" * 130)
    print(f"{'Config':>50} {'Total PnL':>10} {'MaxDD':>8} {'MM PnL':>10} {'MM DD':>8} {'WR%':>6} {'Trades':>7} {'Calmar':>7}")
    print("-" * 140)

    combos = [
        # (label, start, up, down, max, daily_stop, consec_stop, cushion, dd_cb)
        ("Baseline (no rules)", 1.0, 'none', 'none', 1.0, 0, 0, 0, 0),
        ("CB only $1500", 1.0, 'none', 'none', 1.0, 0, 0, 0, 1500),
        ("CB only $2000", 1.0, 'none', 'none', 1.0, 0, 0, 0, 2000),
        ("50% start, win up, loss down", 0.5, 'win', 'loss', 1.0, 0, 0, 0, 0),
        ("50% start, win up, loss down + CB $1500", 0.5, 'win', 'loss', 1.0, 0, 0, 0, 1500),
        ("50% start, win up, loss down, 3 consec stop", 0.5, 'win', 'loss', 1.0, 0, 3, 0, 0),
        ("40% start, win up, loss down", 0.40, 'win', 'loss', 1.0, 0, 0, 0, 0),
        ("40% start, win up, loss down + CB $1500", 0.40, 'win', 'loss', 1.0, 0, 0, 0, 1500),
        ("40% start, win up, loss down, 3 consec", 0.40, 'win', 'loss', 1.0, 0, 3, 0, 0),
        ("33% start, win up, loss down", 0.33, 'win', 'loss', 1.0, 0, 0, 0, 0),
        ("33% start, win up, loss down + CB $1500", 0.33, 'win', 'loss', 1.0, 0, 0, 0, 1500),
        ("50% start, streak up, loss down", 0.5, 'streak', 'loss', 1.0, 0, 0, 0, 0),
        ("50% start, streak up, loss down + CB $1500", 0.5, 'streak', 'loss', 1.0, 0, 0, 0, 1500),
        ("75% start, win up, loss down", 0.75, 'win', 'loss', 1.0, 0, 0, 0, 0),
        ("60% start, win up, loss down", 0.60, 'win', 'loss', 1.0, 0, 0, 0, 0),
        ("50% start, $750 cushion, no down", 0.5, 'cushion', 'none', 1.0, 0, 0, 750, 0),
        ("50% start, $750 cushion + CB $1500", 0.5, 'cushion', 'none', 1.0, 0, 0, 750, 1500),
    ]

    for label, start, up, down, mx, ds, cs, ct, cb in combos:
        tp, dp, dpd = run_strategy(df, start_scale=start, scale_up_on=up,
                                    scale_down_on=down, max_scale=mx,
                                    daily_stop_loss=ds, consec_loss_stop=cs,
                                    cushion_threshold=ct, dd_cb=cb)
        m = compute_metrics(dpd, tp, total_trades)
        print(f"  {label:>48}  ${m['total_pnl']:>9,.0f} ${m['max_dd']:>6,.0f} ${m['mm_pnl']:>9,.0f} ${m['mm_dd']:>6,.0f} {m['win_rate']:>5.1f} {m['trades']:>7} {m['calmar']:>7.1f}")

    # =========================================================================
    # PART 7: Monthly breakdown for top strategies
    # =========================================================================
    print()
    print("=" * 130)
    print("PART 7: Monthly breakdown — Baseline vs Top 3 strategies")
    print("=" * 130)

    strategies = {
        'Base': (1.0, 'none', 'none', 1.0, 0, 0, 0, 0),
        '40%WL': (0.40, 'win', 'loss', 1.0, 0, 0, 0, 0),      # 40% start, win up, loss down
        '50%WL': (0.50, 'win', 'loss', 1.0, 0, 0, 0, 0),       # 50% start, win up, loss down
        '50%WL+CB': (0.50, 'win', 'loss', 1.0, 0, 0, 0, 1500), # with CB
    }

    strat_data = {}
    for name, params in strategies.items():
        tp, dp, dpd = run_strategy(df, *params)
        strat_data[name] = dpd

    months = sorted(set((d.year, d.month) for d in strat_data['Base'].keys()))
    header = f"{'Month':>10}"
    for name in strategies:
        header += f" {name:>10}"
    print(header)
    print("-" * (10 + 11 * len(strategies)))

    for yr, mo in months:
        line = f"  {yr}-{mo:02d}"
        for name in strategies:
            pnl = sum(v for d, v in strat_data[name].items() if d.year == yr and d.month == mo)
            line += f" ${pnl:>9,.0f}"
        print(line)

    print()
    line = "  TOTAL  "
    for name in strategies:
        total = sum(strat_data[name].values())
        line += f" ${total:>9,.0f}"
    print(line)

    # Cumulative DD per strategy
    print()
    print("  Max DD:")
    for name in strategies:
        daily_pnls = [strat_data[name][d] for d in sorted(strat_data[name])]
        cum = np.cumsum(daily_pnls)
        peak = np.maximum.accumulate(cum)
        dd = abs(min(cum - peak))
        print(f"    {name}: ${dd:,.0f}")

    # Mar-May specifically
    print()
    print("  Mar-May 2025:")
    for name in strategies:
        mm_pnl = sum(v for d, v in strat_data[name].items()
                     if date(2025, 3, 1) <= d <= date(2025, 5, 31))
        mm_pnls = [v for d, v in sorted(strat_data[name].items())
                    if date(2025, 3, 1) <= d <= date(2025, 5, 31)]
        mm_cum = np.cumsum(mm_pnls) if mm_pnls else [0]
        mm_peak = np.maximum.accumulate(mm_cum)
        mm_dd = abs(min(mm_cum - mm_peak)) if len(mm_pnls) > 0 else 0
        print(f"    {name}: PnL=${mm_pnl:,.0f}, DD=${mm_dd:,.0f}")


if __name__ == '__main__':
    main()
