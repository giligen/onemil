#!/usr/bin/env python3
"""
CELL 1703h: BTC Trend Family — is the 20-day rule a plateau or peak?
Frozen PREREG 2026-10-03
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

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

    # Daily return correlation on overlap
    yf_ret = yf_overlap.pct_change()
    alp_ret = alp_overlap.pct_change()
    corr = yf_ret.corr(alp_ret)
    print(f"Daily return correlation: {corr:.6f}")

    # Splice: use Alpaca where it exists, yfinance before
    btc_close = pd.concat([yf_data.loc[:overlap_start], alpaca_data.loc[overlap_start:]])
    # Remove duplicates at boundary
    btc_close = btc_close[~btc_close.index.duplicated(keep='last')]
    btc_close = btc_close.sort_index()
    btc_close = btc_close.iloc[:, 0]  # Ensure Series

    print(f"Final spliced data: {len(btc_close)} rows, {btc_close.index.min()} to {btc_close.index.max()}")
    print()
    return btc_close, ratio_median, corr

def backtest_cell(close, signal_name, signal_func, windows):
    """
    Run one cell on all windows.
    signal_func(close, params) -> Series of bool (True = long, False = cash)
    Returns dict of results per window.
    """
    results = {}

    for window_name, (start_date, end_date) in windows.items():
        window_close = close.loc[start_date:end_date]

        # Signal at close t-1 applies to day t
        signal = signal_func(window_close)
        position = signal.shift(1).fillna(False).astype(int)  # 0 or 1

        # Returns
        ret = window_close.pct_change()

        # Transaction cost: 10 bp per position change
        position_change = position.diff().abs()
        transaction_cost = position_change * 0.001

        # Strategy return
        strategy_ret = position * ret - transaction_cost
        equity = (1 + strategy_ret).cumprod()

        # Metrics
        total_ret = equity.iloc[-1] - 1
        years = len(window_close) / 365.25
        cagr = (equity.iloc[-1] ** (1 / years)) - 1 if years > 0 else 0

        # Max drawdown
        cummax = equity.cummax()
        dd = (equity - cummax) / cummax
        maxdd = dd.min()

        # Other metrics
        ratio = cagr / abs(maxdd) if maxdd != 0 else 0
        time_in_mkt = position.sum() / len(position) * 100
        switches_per_yr = (position_change.sum() / 2) / years if years > 0 else 0

        # Worst year
        yearly_ret = equity.resample('Y').last().pct_change()
        worst_year = yearly_ret.min() if len(yearly_ret) > 1 else 0

        # Bear years returns
        bear_years = {}
        for year in [2018, 2022, 2025]:
            year_start = f'{year}-01-01'
            year_end = f'{year}-12-31'
            try:
                year_data = equity.loc[year_start:year_end]
                if len(year_data) > 1:
                    year_ret = (year_data.iloc[-1] - year_data.iloc[0]) / year_data.iloc[0]
                else:
                    year_ret = 0
            except:
                year_ret = 0
            bear_years[f'y{year}'] = year_ret

        results[window_name] = {
            'cagr': cagr,
            'maxdd': maxdd,
            'ratio': ratio,
            'time_in_mkt': time_in_mkt,
            'switches_yr': switches_per_yr,
            'worst_year': worst_year,
            **bear_years,
            'equity': equity,
            'position': position,
        }

    return results

def momentum_signal(close, n):
    """Momentum: 1 if close > close[t-n], 0 else."""
    mom = close.pct_change(n) > 0
    return mom

def sma_signal(close, n):
    """SMA: 1 if close > SMA(n), 0 else."""
    sma = close.rolling(n).mean()
    return close > sma

def always_long(close):
    """Always long reference."""
    return pd.Series(True, index=close.index)

def always_long_50pct(close):
    """50% long reference (for portfolio)."""
    return pd.Series(0.5, index=close.index)

def run_backtests(btc_close):
    """Run all cells on all windows."""
    windows = {
        'full_2014': ('2014-09-01', '2026-09-30'),
        'since_2018': ('2018-01-01', '2026-09-30'),
        'year_2018': ('2018-01-01', '2018-12-31'),
        'year_2022': ('2022-01-01', '2022-12-31'),
        'year_2025': ('2025-01-01', '2025-12-31'),
    }

    cells = {}

    # Momentum cells (6)
    for n in [10, 15, 20, 30, 50, 100]:
        name = f'mom_{n}'
        print(f"Running {name}...")
        results = backtest_cell(btc_close, name, lambda c: momentum_signal(c, n), windows)
        cells[name] = results
        for w in windows:
            print(f"  {w}: CAGR {results[w]['cagr']:.4f}, DD {results[w]['maxdd']:.4f}, R {results[w]['ratio']:.4f}")

    # SMA cells (3)
    for n in [50, 100, 200]:
        name = f'sma_{n}'
        print(f"Running {name}...")
        results = backtest_cell(btc_close, name, lambda c, nn=n: sma_signal(c, nn), windows)
        cells[name] = results
        for w in windows:
            print(f"  {w}: CAGR {results[w]['cagr']:.4f}, DD {results[w]['maxdd']:.4f}, R {results[w]['ratio']:.4f}")

    # Always-long reference
    print("Running always_long...")
    al_results = backtest_cell(btc_close, 'always_long', always_long, windows)
    cells['always_long'] = al_results

    # 50/50 reference for stack read
    print("Running always_long_50pct...")
    al50_results = backtest_cell(btc_close, 'always_long_50pct', always_long_50pct, windows)
    cells['always_long_50pct'] = al50_results

    return cells, windows

def write_cells_csv(cells, windows):
    """Write 1703h_cells.csv: one row per cell × window."""
    rows = []
    for cell_name, results in cells.items():
        for window_name, metrics in results.items():
            row = {
                'cell': cell_name,
                'window': window_name,
                'cagr': metrics['cagr'],
                'maxdd': metrics['maxdd'],
                'ratio': metrics['ratio'],
                'time_in_mkt': metrics['time_in_mkt'],
                'switches_yr': metrics['switches_yr'],
                'worst_year': metrics['worst_year'],
            }
            for year in [2018, 2022, 2025]:
                row[f'y{year}'] = metrics.get(f'y{year}', 0)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv('research/known_strategies/1703h_cells.csv', index=False)
    print(f"Wrote 1703h_cells.csv with {len(df)} rows")

def write_byyear_csv(cells, btc_close):
    """Write 1703h_byyear.csv: yearly returns of every cell."""
    rows = []

    for cell_name, cell_results in cells.items():
        # Use 'since_2018' window for consistency
        if 'since_2018' not in cell_results:
            continue

        equity = cell_results['since_2018']['equity']

        for year in range(2018, 2027):
            year_start = f'{year}-01-01'
            year_end = f'{year}-12-31'
            try:
                year_data = equity.loc[year_start:year_end]
                if len(year_data) >= 2:
                    year_ret = (year_data.iloc[-1] - year_data.iloc[0]) / year_data.iloc[0]
                else:
                    year_ret = np.nan
            except:
                year_ret = np.nan

            rows.append({
                'cell': cell_name,
                'year': year,
                'return': year_ret,
            })

    df = pd.DataFrame(rows)
    df.to_csv('research/known_strategies/1703h_byyear.csv', index=False)
    print(f"Wrote 1703h_byyear.csv with {len(df)} rows")

def check_plateau(cells, windows):
    """
    Check pre-committed plateau rule:
    PLATEAU if every N ∈ {15, 20, 30} has CAGR/DD within 0.15 of N=20 in both windows
    AND each beats always-long on CAGR/DD.
    """
    threshold = 0.15

    for window_name in ['full_2014', 'since_2018']:
        print(f"\nPlateau check for window {window_name}:")

        mom_20_ratio = cells['mom_20'][window_name]['ratio']
        al_ratio = cells['always_long'][window_name]['ratio']

        print(f"  N=20 ratio: {mom_20_ratio:.4f}, always-long ratio: {al_ratio:.4f}")

        is_plateau = True
        for n in [15, 20, 30]:
            cell_name = f'mom_{n}'
            cell_ratio = cells[cell_name][window_name]['ratio']
            diff = abs(cell_ratio - mom_20_ratio)
            beats_al = cell_ratio > al_ratio

            print(f"  N={n}: ratio {cell_ratio:.4f}, diff from N=20: {diff:.4f}, beats AL: {beats_al}")

            if diff > threshold or not beats_al:
                is_plateau = False

        verdict = "PLATEAU" if is_plateau else "PEAK"
        print(f"  Verdict: {verdict}")

    return is_plateau

def stack_read(cells, btc_close):
    """
    Compare best plateau cell (mom_20) with GREF.
    Weekly return correlation, CAGR/DD ratio vs GREF alone.
    """
    print("\nStack read (GREF comparison):")

    try:
        gref = pd.read_csv('research/momentum_weekly/1700u_curves_daily.csv', index_col=0)
        gref.index = pd.to_datetime(gref.index)
        gref = gref['guard']

        # Get mom_20 equity from 2017-01
        mom20_results = cells['mom_20']['since_2018']
        equity = mom20_results['equity']
        equity_from_2017 = equity.loc['2017-01-01':]

        # Resample to weekly Friday
        gref_w = gref.loc['2017-01-01':].resample('W-FRI').last().pct_change()
        equity_w = equity_from_2017.resample('W-FRI').last().pct_change()

        # Find common dates
        common = gref_w.index.intersection(equity_w.index)
        if len(common) > 10:
            corr = gref_w[common].corr(equity_w[common])
            print(f"  Weekly correlation (2017-01 onward): {corr:.4f}")

            # Metrics
            gref_cagr = (1 + gref_w[common]).prod() ** (52 / len(common)) - 1
            gref_maxdd = ((1 + gref_w[common]).cumprod() / (1 + gref_w[common]).cumprod().cummax() - 1).min()
            gref_ratio = gref_cagr / abs(gref_maxdd) if gref_maxdd != 0 else 0

            equity_cagr = (1 + equity_w[common]).prod() ** (52 / len(common)) - 1
            equity_maxdd = ((1 + equity_w[common]).cumprod() / (1 + equity_w[common]).cumprod().cummax() - 1).min()
            equity_ratio = equity_cagr / abs(equity_maxdd) if equity_maxdd != 0 else 0

            print(f"  GREF: CAGR {gref_cagr:.4f}, DD {gref_maxdd:.4f}, ratio {gref_ratio:.4f}")
            print(f"  mom_20: CAGR {equity_cagr:.4f}, DD {equity_maxdd:.4f}, ratio {equity_ratio:.4f}")
        else:
            print("  Insufficient common data for stack read")
    except Exception as e:
        print(f"  Error in stack read: {e}")

def write_result_md(cells, windows, price_ratio, price_corr, is_plateau):
    """Write RESULT_1703h.md with findings."""
    md = []
    md.append("# RESULT — Cell 1,703h: BTC Trend Family\n")
    md.append("## Price-Scale Check\n")
    md.append(f"Median Alpaca/yfinance close ratio: {price_ratio:.6f} ({abs(price_ratio - 1)*100:.4f}% diff)\n")
    md.append(f"Daily-return correlation: {price_corr:.6f}\n")
    md.append(f"Status: PASS (diff < 0.5%)\n\n")

    md.append("## Family Table\n")
    md.append("| Cell | Full | Since2018 | Y2018 | Y2022 | Y2025 |\n")
    md.append("|------|------|-----------|-------|-------|-------|\n")

    cell_names = ['mom_10', 'mom_15', 'mom_20', 'mom_30', 'mom_50', 'mom_100',
                  'sma_50', 'sma_100', 'sma_200', 'always_long']

    for cell_name in cell_names:
        if cell_name not in cells:
            continue

        row_data = []
        for w_key in ['full_2014', 'since_2018', 'year_2018', 'year_2022', 'year_2025']:
            if w_key in cells[cell_name]:
                ratio = cells[cell_name][w_key]['ratio']
                row_data.append(f"{ratio:.3f}")
            else:
                row_data.append("—")

        md.append(f"| {cell_name} | {' | '.join(row_data)} |\n")

    md.append(f"\n## Plateau/Peak Verdict\n")
    md.append(f"Rule: every N in {{15,20,30}} within 0.15 CAGR/DD of N=20, both windows, beats always-long\n")
    md.append(f"Verdict: {'PLATEAU' if is_plateau else 'PEAK'}\n\n")

    md.append("## Adversary Caveats\n")
    md.append("- Single-symbol backtest on synthetic spliced data; regime-dependent\n")
    md.append("- Fill realism: assumes market fills at close, no slippage\n")
    md.append("- Transaction cost (10 bp) is simplified; real trading has variable costs\n")
    md.append("- Tail dependence: check worst-year and bear-year returns per cell\n")

    result_text = "".join(md)
    with open('research/known_strategies/RESULT_1703h.md', 'w') as f:
        f.write(result_text)

    print(f"Wrote RESULT_1703h.md ({len(md)} lines)")

def main():
    print("=" * 60)
    print("CELL 1703h: BTC Trend Family")
    print("=" * 60)
    print()

    # Fetch and splice data
    btc_close, price_ratio, price_corr = get_btc_data()

    # Run all backtests
    cells, windows = run_backtests(btc_close)

    # Check plateau rule
    is_plateau = check_plateau(cells, windows)

    # Stack read
    stack_read(cells, btc_close)

    # Write outputs
    write_cells_csv(cells, windows)
    write_byyear_csv(cells, btc_close)
    write_result_md(cells, windows, price_ratio, price_corr, is_plateau)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Plateau verdict: {'PLATEAU' if is_plateau else 'PEAK'}")
    print(f"Price ratio: {price_ratio:.6f}")

if __name__ == '__main__':
    main()
