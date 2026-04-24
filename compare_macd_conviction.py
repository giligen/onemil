#!/usr/bin/env python3
"""
Compare MACD wave BT: baseline vs conviction-sizing.

Reads two CSV outputs and produces a monthly before/after table + summary stats.
"""
import pandas as pd
import sys


def stats(df, label):
    """Compute summary stats for a trade CSV."""
    n = len(df)
    wr = (df['pnl_dollar'] > 0).mean() * 100
    pnl = df['pnl_dollar'].sum()
    pos = df[df['pnl_dollar'] > 0]['pnl_dollar'].sum()
    neg = abs(df[df['pnl_dollar'] <= 0]['pnl_dollar'].sum())
    pf = pos / neg if neg else float('inf')
    # Chronological DD
    d = df.sort_values(['date', 'entry_time']).reset_index(drop=True)
    eq = d['pnl_dollar'].cumsum()
    peak = eq.cummax()
    dd = (eq - peak).min()
    # Sharpe-ish
    mean_pct = df['pnl_pct'].mean()
    std_pct = df['pnl_pct'].std()
    sharpe = mean_pct / std_pct if std_pct > 0 else 0
    return {
        'label': label, 'n': n, 'wr': wr, 'pnl': pnl, 'pf': pf,
        'dd': dd, 'sharpe': sharpe,
        'avg_pnl': pnl / n if n else 0,
    }


def monthly_compare(base_df, sized_df):
    base_df['date'] = pd.to_datetime(base_df['date'])
    sized_df['date'] = pd.to_datetime(sized_df['date'])
    base_df['month'] = base_df['date'].dt.to_period('M')
    sized_df['month'] = sized_df['date'].dt.to_period('M')

    months = sorted(set(base_df['month']) | set(sized_df['month']))

    print()
    print("=" * 100)
    print("MONTHLY BREAKDOWN — BASELINE vs CONVICTION SIZING")
    print("=" * 100)
    header = (f"{'Month':<10} {'n':>5} {'WR%':>5} "
              f"{'Baseline P&L':>14} {'Sized P&L':>14} "
              f"{'Δ $':>10} {'Δ %':>7} {'avg conv':>9}")
    print(header)
    print('-' * 100)

    tot_b = tot_s = 0
    for m in months:
        gb = base_df[base_df['month'] == m]
        gs = sized_df[sized_df['month'] == m]
        pb = gb['pnl_dollar'].sum()
        ps = gs['pnl_dollar'].sum()
        tot_b += pb
        tot_s += ps
        pct = (ps - pb) / abs(pb) * 100 if pb else 0
        avg_conv = gs['conv_mult'].mean() if 'conv_mult' in gs.columns else 0
        wr_b = (gb['pnl_dollar'] > 0).mean() * 100 if len(gb) else 0
        print(f"{str(m):<10} {len(gb):>5} {wr_b:>4.0f}% "
              f"${pb:>+13,.0f} ${ps:>+13,.0f} "
              f"${ps - pb:>+9,.0f} {pct:>+6.1f}% {avg_conv:>9.2f}")

    print('-' * 100)
    pct_tot = (tot_s - tot_b) / abs(tot_b) * 100 if tot_b else 0
    print(f"{'TOTAL':<10} {len(base_df):>5}      "
          f"${tot_b:>+13,.0f} ${tot_s:>+13,.0f} "
          f"${tot_s - tot_b:>+9,.0f} {pct_tot:>+6.1f}%")


def main():
    base_path = sys.argv[1] if len(sys.argv) > 1 else 'macd_wave_results_baseline.csv'
    sized_path = sys.argv[2] if len(sys.argv) > 2 else 'macd_wave_results_sized.csv'

    base = pd.read_csv(base_path)
    sized = pd.read_csv(sized_path)

    print("=" * 78)
    print("MACD WAVE — BASELINE vs CONVICTION SIZING (15.5mo)")
    print("=" * 78)
    print(f"Baseline CSV: {base_path} ({len(base)} trades)")
    print(f"Sized CSV:    {sized_path} ({len(sized)} trades)")

    b = stats(base, 'BASELINE')
    s = stats(sized, 'SIZED')

    print()
    print(f"{'Metric':<18} {'Baseline':>15} {'Sized':>15} {'Δ':>15}")
    print('-' * 65)
    print(f"{'Trades':<18} {b['n']:>15} {s['n']:>15} {s['n'] - b['n']:>+15}")
    print(f"{'Win rate':<18} {b['wr']:>14.1f}% {s['wr']:>14.1f}% {s['wr'] - b['wr']:>+14.1f}%")
    print(f"{'Total P&L':<18} ${b['pnl']:>+14,.0f} ${s['pnl']:>+14,.0f} ${s['pnl'] - b['pnl']:>+14,.0f}")
    print(f"{'Avg P&L/trade':<18} ${b['avg_pnl']:>+14,.0f} ${s['avg_pnl']:>+14,.0f} ${s['avg_pnl'] - b['avg_pnl']:>+14,.0f}")
    print(f"{'Profit factor':<18} {b['pf']:>15.2f} {s['pf']:>15.2f} {s['pf'] - b['pf']:>+15.2f}")
    print(f"{'Max drawdown':<18} ${b['dd']:>+14,.0f} ${s['dd']:>+14,.0f} ${s['dd'] - b['dd']:>+14,.0f}")
    pct_improve = (s['pnl'] - b['pnl']) / abs(b['pnl']) * 100 if b['pnl'] else 0
    print(f"\nP&L improvement: {pct_improve:+.1f}% on {(sized['conv_mult'].mean() / 1.0 - 1) * 100:+.1f}% "
          f"more avg notional (avg conv_mult = {sized['conv_mult'].mean():.2f}x)")
    print(f"Capital efficiency: ΔP&L / Δnotional = "
          f"${s['pnl'] - b['pnl']:+,.0f} per "
          f"${(sized['conv_mult'].mean() - 1.0) * 50000 * len(sized):.0f} extra capital deployed")

    monthly_compare(base, sized)


if __name__ == '__main__':
    main()
