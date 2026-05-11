"""SPY conviction-filter ablation study (2025 + 2026 YTD).

Compares production BT (SPY filter ON, V4 conviction rule 4 active) vs
the same BT with the SPY rule's contribution zeroed out via the
`BT_SPY_FEATURE_OFF=1` env var added to backtest.py:1640.

The SPY rule (V4 conviction rule 4) adds:
  - +0.3 when SPY 3d avg daily range > 1.2%  (volatile)
  -  0.0 when 0.8% <= SPY 3d range <= 1.2%  (neutral)
  - -0.5 when SPY 3d range < 0.8%           (low-vol / range-bound)
  - -0.5 when SPY data is missing/stale
to the raw conviction score (then clamped to [0.25, 3.0]).
The score multiplies position size and gates entry (< 1.40 = skip).

Two CSVs read:
  - backtest_results/spy_filter_ON_2025_2026.csv  (production)
  - backtest_results/spy_filter_OFF_2025_2026.csv (env-toggled ablation)

Outputs:
  - Monthly breakdown table (year-month rows, side-by-side)
  - Yearly aggregate (2025 / 2026 YTD)
  - Decision-matrix summary (per the framework discussed: filter helps /
    is neutral / hurts based on combination of WR / DD / P&L deltas)
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ON_CSV  = Path("backtest_results/spy_filter_ON_2025_2026.csv")
OFF_CSV = Path("backtest_results/spy_filter_OFF_2025_2026.csv")


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"missing: {path} — run batch_backtest first")
    df = pd.read_csv(path)
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')
    df['year'] = df['date'].dt.year
    return df


def metrics(df: pd.DataFrame) -> dict:
    """Compute aggregate trade statistics + DD curve."""
    if df.empty:
        return {
            'n': 0, 'pnl': 0.0, 'wr': 0.0,
            'avg_win': 0.0, 'avg_loss': 0.0,
            'expectancy': 0.0, 'max_dd': 0.0,
        }
    pnl = df['pnl'].sum()
    wins = df[df['pnl'] > 0]['pnl']
    losses = df[df['pnl'] <= 0]['pnl']
    n = len(df)
    wr = len(wins) / n * 100 if n else 0.0
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    expectancy = (
        (wr / 100) * avg_win + (1 - wr / 100) * avg_loss
    )
    # Max DD on the cumulative equity curve, trades sorted by date+entry_time.
    df_sorted = df.sort_values(['date', 'entry_time_et']).copy()
    df_sorted['cum_pnl'] = df_sorted['pnl'].cumsum()
    df_sorted['peak'] = df_sorted['cum_pnl'].cummax()
    df_sorted['dd'] = df_sorted['cum_pnl'] - df_sorted['peak']
    max_dd = df_sorted['dd'].min() if len(df_sorted) else 0.0
    return {
        'n': n, 'pnl': pnl, 'wr': wr,
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'expectancy': expectancy, 'max_dd': max_dd,
    }


def fmt_row(label: str, m: dict, width: int = 28) -> str:
    return (
        f"{label:<{width}} "
        f"n={m['n']:>4}  "
        f"pnl=${m['pnl']:>+10,.0f}  "
        f"WR={m['wr']:>5.1f}%  "
        f"avgW=${m['avg_win']:>+7,.0f}  "
        f"avgL=${m['avg_loss']:>+7,.0f}  "
        f"E=${m['expectancy']:>+6,.0f}  "
        f"MDD=${m['max_dd']:>+8,.0f}"
    )


def diff_metrics(on: dict, off: dict) -> dict:
    return {
        'n':          off['n']          - on['n'],
        'pnl':        off['pnl']        - on['pnl'],
        'wr':         off['wr']         - on['wr'],
        'avg_win':    off['avg_win']    - on['avg_win'],
        'avg_loss':   off['avg_loss']   - on['avg_loss'],
        'expectancy': off['expectancy'] - on['expectancy'],
        'max_dd':     off['max_dd']     - on['max_dd'],
    }


def fmt_diff_row(label: str, d: dict, width: int = 28) -> str:
    """Δ row: positive = removing filter helped that metric."""
    return (
        f"{label:<{width}} "
        f"Δn={d['n']:>+4}  "
        f"Δpnl=${d['pnl']:>+10,.0f}  "
        f"ΔWR={d['wr']:>+5.1f}%  "
        f"ΔavgW=${d['avg_win']:>+6,.0f}  "
        f"ΔavgL=${d['avg_loss']:>+6,.0f}  "
        f"ΔE=${d['expectancy']:>+5,.0f}  "
        f"ΔMDD=${d['max_dd']:>+8,.0f}"
    )


def main():
    on  = load(ON_CSV)
    off = load(OFF_CSV)

    print("=" * 138)
    print("SPY conviction-filter ablation — 2025-01 to 2026-05")
    print("=" * 138)
    print(f"ON file:  {ON_CSV}  ({len(on)} trades)")
    print(f"OFF file: {OFF_CSV} ({len(off)} trades)")
    print()

    # ---------------- aggregate ----------------
    on_m  = metrics(on)
    off_m = metrics(off)
    print("AGGREGATE 2025-01 to 2026-05")
    print("-" * 138)
    print(fmt_row("SPY filter ON (production)", on_m))
    print(fmt_row("SPY filter OFF (ablation)", off_m))
    print(fmt_diff_row("Δ (OFF - ON)", diff_metrics(on_m, off_m)))
    print()

    # ---------------- yearly ----------------
    print("BY YEAR")
    print("-" * 138)
    for year in sorted(set(on['year'].unique()) | set(off['year'].unique())):
        y_on  = metrics(on[on['year'] == year])
        y_off = metrics(off[off['year'] == year])
        print(fmt_row(f"{year} ON ", y_on))
        print(fmt_row(f"{year} OFF", y_off))
        print(fmt_diff_row(f"{year} Δ ", diff_metrics(y_on, y_off)))
        print()

    # ---------------- monthly ----------------
    print("BY MONTH (Δ shows OFF minus ON)")
    print("-" * 138)
    print(f"{'month':<10}  "
          f"{'ON pnl':>10}  {'OFF pnl':>10}  {'Δpnl':>10}  "
          f"{'ON n':>5}  {'OFF n':>5}  {'Δn':>5}  "
          f"{'ON WR':>7}  {'OFF WR':>7}  {'ΔWR':>7}  "
          f"{'Δexp':>7}  {'ΔMDD':>9}")
    print("-" * 138)
    months = sorted(set(on['year_month'].unique()) | set(off['year_month'].unique()))
    diff_pos_count = 0
    diff_neg_count = 0
    diff_pos_sum = 0.0
    diff_neg_sum = 0.0
    for ym in months:
        on_ym  = metrics(on[on['year_month'] == ym])
        off_ym = metrics(off[off['year_month'] == ym])
        d = diff_metrics(on_ym, off_ym)
        if d['pnl'] > 0:
            diff_pos_count += 1
            diff_pos_sum += d['pnl']
        elif d['pnl'] < 0:
            diff_neg_count += 1
            diff_neg_sum += d['pnl']
        print(f"{str(ym):<10}  "
              f"${on_ym['pnl']:>+8,.0f}  ${off_ym['pnl']:>+8,.0f}  ${d['pnl']:>+8,.0f}  "
              f"{on_ym['n']:>5}  {off_ym['n']:>5}  {d['n']:>+5}  "
              f"{on_ym['wr']:>6.1f}%  {off_ym['wr']:>6.1f}%  {d['wr']:>+6.1f}%  "
              f"${d['expectancy']:>+5,.0f}  ${d['max_dd']:>+7,.0f}")
    print()

    print("MONTHLY SUMMARY")
    print("-" * 60)
    print(f"  months where removing filter HELPS (Δpnl > 0): {diff_pos_count:>3}  "
          f"sum +${diff_pos_sum:,.0f}")
    print(f"  months where removing filter HURTS (Δpnl < 0): {diff_neg_count:>3}  "
          f"sum  ${diff_neg_sum:,.0f}")
    print()

    # ---------------- decision matrix ----------------
    d_total = diff_metrics(on_m, off_m)
    print("DECISION MATRIX (Δ = OFF - ON, so positive Δ means removing filter HELPS)")
    print("-" * 138)
    pnl_dir = "↑ HELPS" if d_total['pnl'] > 0 else ("↓ HURTS" if d_total['pnl'] < 0 else "≈ FLAT")
    wr_dir  = "↓"      if d_total['wr']  < 0 else ("↑"      if d_total['wr']  > 0 else "≈")
    dd_dir  = "↑ worse" if d_total['max_dd'] < 0 else ("↓ better" if d_total['max_dd'] > 0 else "≈")
    print(f"  ΔP&L:        ${d_total['pnl']:>+10,.0f}  ({pnl_dir})")
    print(f"  ΔWR:         {d_total['wr']:>+10.1f}%  ({wr_dir})")
    print(f"  Δexpectancy: ${d_total['expectancy']:>+10,.0f}/trade")
    print(f"  Δtrade-count:{d_total['n']:>+10}")
    print(f"  ΔMDD:        ${d_total['max_dd']:>+10,.0f}  ({dd_dir})")
    print()
    # Verdict per the matrix discussed
    if d_total['pnl'] < -1000 and d_total['wr'] < 0 and d_total['max_dd'] < 0:
        verdict = "filter is GENUINELY DEFENSIVE — keep (kills losers, accepts modest cost)"
    elif abs(d_total['pnl']) < 2000 and d_total['wr'] < 0 and d_total['max_dd'] < 0:
        verdict = "filter is Sharpe/Calmar-positive but alpha-neutral — keep for RISK MGMT"
    elif d_total['pnl'] > 1000 and d_total['wr'] >= 0 and d_total['max_dd'] >= -2000:
        verdict = "filter is SUPPRESSING EDGE in this period — REMOVE or recalibrate"
    elif d_total['pnl'] > 0 and d_total['wr'] < 0 and d_total['max_dd'] < -3000:
        verdict = "filter blocks BOTH winners AND losers (asymmetric on winners) — RECALIBRATE"
    elif abs(d_total['pnl']) < 500 and abs(d_total['max_dd']) < 1000:
        verdict = "filter is essentially NOISE — REMOVE for simplicity"
    else:
        verdict = "mixed signal — see monthly + yearly breakdown above"
    print(f"  Verdict (heuristic): {verdict}")


if __name__ == '__main__':
    main()
