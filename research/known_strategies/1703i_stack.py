#!/usr/bin/env python3
"""
CELL 1703i: Stack weight optimization — BTC trend + guarded GREF sleeve.
Frozen PREREG 2026-10-03
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import os

def get_btc_data():
    """Fetch yfinance BTC-USD and splice with Alpaca OHLC."""
    print("Fetching yfinance BTC-USD 2014-09 to 2026-10...")
    yf_data = yf.download('BTC-USD', start='2014-09-01', end='2026-10-01',
                          auto_adjust=True, progress=False)
    # Flatten MultiIndex columns if present
    if isinstance(yf_data.columns, pd.MultiIndex):
        yf_data.columns = yf_data.columns.droplevel(1)
    yf_data = yf_data[['Close']].rename(columns={'Close': 'close'})
    yf_data.index.name = 'date'

    print(f"yfinance data: {len(yf_data)} rows, {yf_data.index.min()} to {yf_data.index.max()}")

    # Load Alpaca OHLC
    print("Loading Alpaca OHLC from 1703g_btc_ohlc.parquet...")
    alpaca_data = pd.read_parquet('research/known_strategies/1703g_btc_ohlc.parquet')
    alpaca_data.index.name = 'date'
    alpaca_data = alpaca_data[['close']]

    print(f"Alpaca data: {len(alpaca_data)} rows, {alpaca_data.index.min()} to {alpaca_data.index.max()}")

    # Price-scale check on overlap
    overlap_start = max(yf_data.index.min(), alpaca_data.index.min())
    overlap_end = min(yf_data.index.max(), alpaca_data.index.max())
    print(f"Overlap window: {overlap_start} to {overlap_end}")

    yf_overlap = yf_data.loc[overlap_start:overlap_end, 'close']
    alp_overlap = alpaca_data.loc[overlap_start:overlap_end, 'close']

    ratio = alp_overlap / yf_overlap
    ratio_median = ratio.median()
    ratio_pct_diff = abs(ratio_median - 1.0) * 100
    print(f"Price-scale check: median Alpaca/yfinance ratio = {ratio_median:.6f} ({ratio_pct_diff:.4f}% diff)")

    if ratio_pct_diff > 0.5:
        print(f"ERROR: Price-scale check failed! Ratio diff {ratio_pct_diff:.4f}% exceeds 0.5%")
        sys.exit(1)

    # Splice: use Alpaca where it exists, yfinance before
    btc_close = pd.concat([yf_data.loc[:overlap_start], alpaca_data.loc[overlap_start:]])
    # Remove duplicates at boundary
    btc_close = btc_close[~btc_close.index.duplicated(keep='last')]
    btc_close = btc_close.sort_index()
    btc_close = btc_close.iloc[:, 0]  # Ensure Series

    print(f"Final spliced data: {len(btc_close)} rows, {btc_close.index.min()} to {btc_close.index.max()}")
    print()
    return btc_close

def load_sleeve():
    """Load sleeve equity from 1700u_curves_daily.csv."""
    print("Loading sleeve from 1700u_curves_daily.csv...")
    sleeve = pd.read_csv('research/momentum_weekly/1700u_curves_daily.csv', index_col=0)
    sleeve.index = pd.to_datetime(sleeve.index)
    sleeve = sleeve['guard']  # Use 'guard' column
    print(f"Sleeve data: {len(sleeve)} rows, {sleeve.index.min()} to {sleeve.index.max()}")
    print()
    return sleeve

def momentum_signal(close):
    """Momentum 20: 1 if close[t-1] > close[t-21], 0 else."""
    mom = close > close.shift(20)
    return mom

def sma_signal(close):
    """SMA 100: 1 if close[t-1] > SMA100[t-1], 0 else."""
    sma = close.rolling(100).mean()
    return close > sma

def build_portfolio(btc_close, component_signal, sleeve_eq, weight, start_date, end_date):
    """
    Build portfolio with rebalancing each Monday.
    weight: component weight (e.g., 0.10 for 10%)
    sleeve_eq: Series of daily sleeve equity
    Returns: equity Series, daily returns Series
    """
    # Align all series to common date range
    window_close = btc_close.loc[start_date:end_date].copy()
    window_signal = component_signal.loc[start_date:end_date].copy()
    window_sleeve = sleeve_eq.loc[start_date:end_date].copy()

    # Forward-fill sleeve on non-trading days (calendar days)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    window_sleeve = window_sleeve.reindex(all_dates, method='ffill')

    # Align component signal to trading days only
    window_signal = window_signal.reindex(window_close.index)

    # Shift signal: apply at t-1 for day t
    position = window_signal.shift(1).fillna(False).astype(int)  # 0 or 1

    # Component returns
    comp_ret = window_close.pct_change()

    # Sleeve returns (using daily equity)
    sleeve_ret = window_sleeve.pct_change()

    # Initialize portfolio
    dates = window_close.index
    portfolio_equity = pd.Series(1.0, index=dates)
    portfolio_returns = pd.Series(0.0, index=dates)
    rebalance_cost = pd.Series(0.0, index=dates)

    # Monday detection
    is_monday = window_close.index.dayofweek == 0

    # Initial portfolio state
    prev_target_alloc = 0  # Component allocation at start
    current_alloc = 0

    for i in range(len(dates)):
        date = dates[i]

        # Determine target allocation for today
        target_comp_alloc = weight if position.iloc[i] == 1 else 0

        # Check if rebalance should happen (Monday + different allocation)
        should_rebalance = (is_monday[i] and abs(target_comp_alloc - current_alloc) > 1e-9)

        # Rebalance cost (1 bp on absolute flow)
        if should_rebalance and i > 0:
            flow = abs(target_comp_alloc - current_alloc)
            rebalance_cost.iloc[i] = flow * 0.0001
            current_alloc = target_comp_alloc

        # Daily returns
        if i > 0:
            c_ret = comp_ret.iloc[i] if not pd.isna(comp_ret.iloc[i]) else 0
            s_ret = sleeve_ret.iloc[i] if not pd.isna(sleeve_ret.iloc[i]) else 0

            # Blended return
            blended_ret = current_alloc * c_ret + (1 - current_alloc) * s_ret - rebalance_cost.iloc[i]
            portfolio_returns.iloc[i] = blended_ret

            # Update equity
            portfolio_equity.iloc[i] = portfolio_equity.iloc[i - 1] * (1 + blended_ret)

    return portfolio_equity, portfolio_returns

def compute_metrics(equity, dates, start_date, end_date):
    """Compute performance metrics."""
    window_eq = equity.loc[start_date:end_date].copy()
    window_dates = pd.DatetimeIndex(dates).intersection(window_eq.index)
    window_eq = window_eq.loc[window_dates]

    if len(window_eq) < 2:
        return None

    total_ret = window_eq.iloc[-1] - 1
    years = len(window_eq) / 365.25
    cagr = (window_eq.iloc[-1] ** (1 / years)) - 1 if years > 0 else 0

    # Max drawdown
    cummax = window_eq.cummax()
    dd = (window_eq - cummax) / cummax
    maxdd = dd.min()

    ratio = cagr / abs(maxdd) if maxdd != 0 else 0

    # Weekly stats (Wed-Fri resampled, use last value)
    # Actually PREREG says W-FRI, which is Friday of each week
    weekly_ret = window_eq.resample('W-FRI').last().pct_change()
    worst_week = weekly_ret.min() if len(weekly_ret) > 1 else 0
    weekly_p10 = weekly_ret.quantile(0.10) if len(weekly_ret) > 1 else 0
    green_weeks = (weekly_ret > 0).sum() if len(weekly_ret) > 1 else 0
    green_week_pct = green_weeks / (len(weekly_ret) - 1) * 100 if len(weekly_ret) > 1 else 0

    # Monthly stats
    monthly_ret = window_eq.resample('M').last().pct_change()
    worst_month = monthly_ret.min() if len(monthly_ret) > 1 else 0

    # Half-window ratios
    mid_idx = len(window_eq) // 2
    h1_eq = window_eq.iloc[:mid_idx]
    h2_eq = window_eq.iloc[mid_idx:]

    def calc_ratio(eq):
        if len(eq) < 2:
            return 0
        ret = eq.iloc[-1] - 1
        years = len(eq) / 365.25
        c = (eq.iloc[-1] ** (1 / years)) - 1 if years > 0 else 0
        cummax = eq.cummax()
        d = ((eq - cummax) / cummax).min()
        return c / abs(d) if d != 0 else 0

    h1_ratio = calc_ratio(h1_eq)
    h2_ratio = calc_ratio(h2_eq)

    return {
        'cagr': cagr,
        'maxdd': maxdd,
        'ratio': ratio,
        'worst_week': worst_week,
        'worst_month': worst_month,
        'weekly_p10': weekly_p10,
        'green_week_pct': green_week_pct,
        'h1_ratio': h1_ratio,
        'h2_ratio': h2_ratio,
        'equity': window_eq,
    }

def compute_shared_tail(comp_returns, sleeve_returns, sleeve_equity, start_date, end_date):
    """
    Find the 10 worst weeks of sleeve and compute component return in those weeks.
    Returns DataFrame with date, sleeve_ret, comp_ret.
    """
    window_comp = comp_returns.loc[start_date:end_date].copy()
    window_sleeve_eq = sleeve_equity.loc[start_date:end_date].copy()

    # Align to common dates
    common_dates = window_comp.index.intersection(window_sleeve_eq.index)
    window_comp = window_comp.loc[common_dates]
    window_sleeve_eq = window_sleeve_eq.loc[common_dates]

    # Sleeve returns
    sleeve_ret = window_sleeve_eq.pct_change()

    # Weekly aggregation (W-FRI)
    weekly_comp = window_comp.resample('W-FRI').sum()
    weekly_sleeve = sleeve_ret.resample('W-FRI').sum()

    # Find 10 worst weeks
    weekly_df = pd.DataFrame({
        'date': weekly_sleeve.index,
        'sleeve_ret': weekly_sleeve.values,
        'comp_ret': weekly_comp.values,
    })
    weekly_df = weekly_df.sort_values('sleeve_ret').head(10)

    return weekly_df[['date', 'sleeve_ret', 'comp_ret']]

def main():
    os.chdir('/home/ec2-user/onemil')

    btc_close = get_btc_data()
    sleeve_eq = load_sleeve()

    # Full window
    start_date = '2017-01-01'
    end_date = '2026-09-30'

    # Define signals
    print("Computing component signals...")
    mom_20_signal = momentum_signal(btc_close)
    sma_100_signal = sma_signal(btc_close)

    print("Building reference (weight 0 / GREF alone)...")
    ref_eq, ref_ret = build_portfolio(btc_close, pd.Series(False, index=btc_close.index),
                                      sleeve_eq, 0.0, start_date, end_date)

    ref_metrics = compute_metrics(ref_eq, ref_eq.index, start_date, end_date)
    print(f"GREF ratio: {ref_metrics['ratio']:.4f}")
    print()

    # Run cells
    forms = {
        'mom_20': mom_20_signal,
        'sma_100': sma_100_signal,
    }
    weights = [0.10, 0.20, 0.30]

    cells_results = {}
    cells_rows = []

    for form_name, form_signal in forms.items():
        for w in weights:
            cell_name = f'1703i-{form_name}-w{int(w*100)}'
            print(f"Running {cell_name}...")

            eq, ret = build_portfolio(btc_close, form_signal, sleeve_eq, w, start_date, end_date)
            metrics = compute_metrics(eq, eq.index, start_date, end_date)

            if metrics:
                # Shared tail
                tail_df = compute_shared_tail(ret, sleeve_eq.pct_change(), sleeve_eq, start_date, end_date)
                comp_mean_in_worst10 = tail_df['comp_ret'].mean()

                cells_results[cell_name] = {
                    'form': form_name,
                    'weight': w,
                    'metrics': metrics,
                    'comp_mean_worst10': comp_mean_in_worst10,
                }

                row = {
                    'form': form_name,
                    'w': int(w * 100),
                    'cagr': metrics['cagr'],
                    'maxdd': metrics['maxdd'],
                    'ratio': metrics['ratio'],
                    'worst_week': metrics['worst_week'],
                    'worst_month': metrics['worst_month'],
                    'weekly_p10': metrics['weekly_p10'],
                    'green_week_share': metrics['green_week_pct'] / 100,
                    'h1_ratio': metrics['h1_ratio'],
                    'h2_ratio': metrics['h2_ratio'],
                    'comp_mean_in_sleeve_worst10': comp_mean_in_worst10,
                    'd_cagr': metrics['cagr'] - ref_metrics['cagr'],
                    'd_maxdd': metrics['maxdd'] - ref_metrics['maxdd'],
                    'd_ratio': metrics['ratio'] - ref_metrics['ratio'],
                }
                cells_rows.append(row)

                print(f"  CAGR {metrics['cagr']:.4f}, DD {metrics['maxdd']:.4f}, R {metrics['ratio']:.4f}")
                print(f"  vs GREF: ΔR {row['d_ratio']:+.4f}, ΔDD {row['d_maxdd']:+.4f}")
                print()

    # Write cells CSV
    cells_df = pd.DataFrame(cells_rows)
    cells_df.to_csv('research/known_strategies/1703i_cells.csv', index=False)
    print(f"Wrote 1703i_cells.csv with {len(cells_df)} rows")
    print()

    # Write worst weeks CSV
    # Use mom_20 w10 as reference for sleeve's worst weeks
    mom_20_signal_full = momentum_signal(btc_close)
    eq_ref, ret_ref = build_portfolio(btc_close, mom_20_signal_full, sleeve_eq, 0.10, start_date, end_date)
    worst_weeks_df = compute_shared_tail(ret_ref, sleeve_eq.pct_change(), sleeve_eq, start_date, end_date)
    worst_weeks_df.to_csv('research/known_strategies/1703i_worst_weeks.csv', index=False)
    print(f"Wrote 1703i_worst_weeks.csv with {len(worst_weeks_df)} rows")
    print()

    # Write verdict
    gref_ratio = ref_metrics['ratio']
    gref_maxdd = ref_metrics['maxdd']
    gref_h1_ratio = compute_metrics(ref_eq, ref_eq.index, '2017-01-01', '2021-12-31')['ratio']
    gref_h2_ratio = compute_metrics(ref_eq, ref_eq.index, '2022-01-01', '2026-09-30')['ratio']

    passing_cells = []
    for cell_name, cell_data in cells_results.items():
        m = cell_data['metrics']
        w = cell_data['weight']
        comp_mean = cell_data['comp_mean_worst10']

        passes = (
            m['ratio'] >= gref_ratio + 0.05 and
            m['maxdd'] >= gref_maxdd - 0.01 and
            m['h1_ratio'] >= gref_h1_ratio and
            m['h2_ratio'] >= gref_h2_ratio and
            comp_mean >= 0
        )

        if passes:
            passing_cells.append((cell_name, w, m['ratio'], m['maxdd']))

    report_lines = [
        "# RESULT 1703i: Stack weight optimization",
        "",
        f"## GREF reference (w=0)",
        f"GREF ratio: {gref_ratio:.4f} (expected ≈ 0.75)",
        f"GREF max DD: {gref_maxdd:.4f}",
        f"H1 ratio (2017-21): {gref_h1_ratio:.4f}",
        f"H2 ratio (2022-26): {gref_h2_ratio:.4f}",
        "",
        "## Component cells (w > 0)",
    ]

    for _, row in cells_df.iterrows():
        form = row['form']
        w = row['w']
        ratio = row['ratio']
        maxdd = row['maxdd']
        comp_mean = row['comp_mean_in_sleeve_worst10']
        d_ratio = row['d_ratio']

        rule_check = (
            ratio >= gref_ratio + 0.05 and
            maxdd >= gref_maxdd - 0.01 and
            comp_mean >= 0
        )
        verdict = "PASS" if rule_check else "FAIL"

        report_lines.append(f"{form} w={w}%: ratio {ratio:.4f} ({d_ratio:+.4f}), DD {maxdd:.4f}, comp_worst10 {comp_mean:+.4f} — {verdict}")

    report_lines.extend([
        "",
        "## Shared tail (10 worst weeks)",
        "See 1703i_worst_weeks.csv",
        "",
    ])

    if passing_cells:
        report_lines.append("## Verdict: CELLS PASS")
        for cell_name, w, ratio, maxdd in passing_cells:
            report_lines.append(f"- {cell_name}: recommended for paper stack")
    else:
        report_lines.append("## Verdict: NO CELLS PASS")
        report_lines.append("BTC trend is a personal-holding matter; close the stack question.")

    report_text = "\n".join(report_lines)
    with open('research/known_strategies/RESULT_1703i.md', 'w') as f:
        f.write(report_text)
    print(f"Wrote RESULT_1703i.md")
    print()
    print(report_text)

if __name__ == '__main__':
    main()
