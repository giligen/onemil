"""
Deep regime analysis — what makes Mar-May 2025 structurally different?

Goal: Find observable market signals that PREDICT high-risk periods
for our strategy, so we can apply cushion rules selectively.

Approach:
1. Load SPY daily bars from DB → compute market indicators
2. Load our trade data → compute daily strategy performance
3. Join and correlate: which indicators predict bad days/weeks?
4. Test regime-conditional strategies

Candidate signals (all computed with NO lookahead):
- SPY realized volatility (trailing ATR, daily range)
- SPY 5d return (already tested, weak)
- SPY trend (above/below moving averages)
- Number of movers per day (from our trade data)
- Rolling strategy win rate (adaptive)
- Combination indicators
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import date, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def load_spy_bars(db_path='data/onemil.db'):
    """Load SPY daily bars from the database."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT bar_date as date, open, high, low, close, volume
        FROM daily_bars
        WHERE symbol = 'SPY'
        ORDER BY bar_date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date']).dt.date
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_trades(csv_path='C:/Work/onemil/full_15mo.csv'):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['entry_time'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['entry_time_et'])
    df['pnl'] = df['pnl'].astype(float)
    df = df.sort_values(['date', 'entry_time']).reset_index(drop=True)
    return df


def compute_spy_indicators(spy_df):
    """Compute market regime indicators from SPY bars.
    All indicators use ONLY data available BEFORE market open on the given day."""

    spy = spy_df.copy().sort_values('date').reset_index(drop=True)

    # Daily range as % of close
    spy['daily_range_pct'] = (spy['high'] - spy['low']) / spy['close'] * 100

    # True Range
    spy['prev_close'] = spy['close'].shift(1)
    spy['true_range'] = np.maximum(
        spy['high'] - spy['low'],
        np.maximum(abs(spy['high'] - spy['prev_close']),
                   abs(spy['low'] - spy['prev_close']))
    )
    spy['tr_pct'] = spy['true_range'] / spy['close'] * 100

    # Daily return
    spy['daily_return'] = (spy['close'] / spy['prev_close'] - 1) * 100

    # Rolling indicators (all shifted by 1 = using T-1 and earlier)
    for window in [3, 5, 10, 20]:
        # Realized volatility: avg daily range over N days
        spy[f'vol_{window}d'] = spy['daily_range_pct'].rolling(window).mean().shift(1)

        # ATR
        spy[f'atr_{window}d'] = spy['tr_pct'].rolling(window).mean().shift(1)

        # Return over N days
        spy[f'ret_{window}d'] = ((spy['close'] / spy['close'].shift(window)) - 1).shift(1) * 100

        # Max daily range in N days (peak volatility)
        spy[f'max_range_{window}d'] = spy['daily_range_pct'].rolling(window).max().shift(1)

        # Std dev of daily returns (another vol measure)
        spy[f'ret_std_{window}d'] = spy['daily_return'].rolling(window).std().shift(1)

    # Gap: open vs prev close (market gap at open)
    spy['spy_gap_pct'] = ((spy['open'] / spy['prev_close']) - 1) * 100

    # Distance from 20-day high and low
    spy['high_20d'] = spy['high'].rolling(20).max().shift(1)
    spy['low_20d'] = spy['low'].rolling(20).min().shift(1)
    spy['dist_from_20d_high'] = ((spy['close'].shift(1) / spy['high_20d']) - 1) * 100
    spy['dist_from_20d_low'] = ((spy['close'].shift(1) / spy['low_20d']) - 1) * 100

    # SMA positions (is SPY above/below its MAs?)
    for sma in [10, 20, 50]:
        spy[f'sma_{sma}'] = spy['close'].rolling(sma).mean().shift(1)
        spy[f'above_sma_{sma}'] = (spy['close'].shift(1) > spy[f'sma_{sma}']).astype(int)

    # Down days ratio in last N days
    for window in [5, 10]:
        spy[f'down_days_{window}d'] = (spy['daily_return'] < 0).rolling(window).sum().shift(1)

    # Consecutive down days
    is_down = (spy['daily_return'] < 0).astype(int)
    consec = []
    cnt = 0
    for v in is_down:
        if v == 1: cnt += 1
        else: cnt = 0
        consec.append(cnt)
    spy['consec_down_days'] = pd.Series(consec, index=spy.index).shift(1)

    return spy


def main():
    print("Loading data...")
    spy_df = load_spy_bars()
    trades_df = load_trades()
    spy = compute_spy_indicators(spy_df)

    print(f"SPY bars: {len(spy_df)} ({spy_df['date'].min()} to {spy_df['date'].max()})")
    print(f"Trades: {len(trades_df)} across {trades_df['date'].nunique()} days")
    print()

    # Aggregate trades by day
    daily_trades = trades_df.groupby('date').agg(
        n_trades=('pnl', 'count'),
        pnl=('pnl', 'sum'),
        wins=('pnl', lambda x: (x > 0).sum()),
        losses=('pnl', lambda x: (x <= 0).sum()),
        avg_pnl=('pnl', 'mean'),
    ).reset_index()
    daily_trades['wr'] = daily_trades['wins'] / daily_trades['n_trades'] * 100

    # Merge with SPY indicators
    merged = daily_trades.merge(spy, on='date', how='left')

    # =========================================================================
    print("=" * 150)
    print("SECTION 1: SPY indicators during Mar-May 2025 vs Rest")
    print("=" * 150)

    mm = merged[(merged['date'] >= date(2025, 3, 1)) & (merged['date'] <= date(2025, 5, 31))]
    rest = merged[~((merged['date'] >= date(2025, 3, 1)) & (merged['date'] <= date(2025, 5, 31)))]

    indicators = [
        'vol_5d', 'vol_10d', 'vol_20d',
        'atr_5d', 'atr_10d', 'atr_20d',
        'ret_5d', 'ret_10d', 'ret_20d',
        'max_range_5d', 'max_range_10d',
        'ret_std_5d', 'ret_std_10d',
        'above_sma_10', 'above_sma_20', 'above_sma_50',
        'dist_from_20d_high', 'dist_from_20d_low',
        'down_days_5d', 'down_days_10d',
        'consec_down_days',
        'n_trades',
    ]

    print(f"\n  {'Indicator':<25} {'Mar-May Mean':>12} {'Rest Mean':>12} {'Ratio':>8} {'Direction':>10}")
    print("  " + "-" * 75)

    for ind in indicators:
        if ind not in merged.columns: continue
        mm_mean = mm[ind].mean()
        rest_mean = rest[ind].mean()
        if rest_mean != 0:
            ratio = mm_mean / rest_mean
        else:
            ratio = float('inf')
        direction = "HIGHER ⚠" if mm_mean > rest_mean * 1.2 else "LOWER ⚠" if mm_mean < rest_mean * 0.8 else "~same"
        print(f"  {ind:<25} {mm_mean:>12.3f} {rest_mean:>12.3f} {ratio:>8.2f} {direction:>10}")

    # =========================================================================
    print()
    print("=" * 150)
    print("SECTION 2: Correlation of each indicator with DAILY P&L")
    print("=" * 150)

    print(f"\n  {'Indicator':<25} {'Corr w/ PnL':>12} {'Corr w/ WR':>12} {'Strength':>10}")
    print("  " + "-" * 65)

    correlations = []
    for ind in indicators:
        if ind not in merged.columns: continue
        valid = merged[[ind, 'pnl', 'wr']].dropna()
        if len(valid) < 10: continue
        corr_pnl = valid[ind].corr(valid['pnl'])
        corr_wr = valid[ind].corr(valid['wr'])
        strength = "STRONG" if abs(corr_pnl) > 0.15 else "moderate" if abs(corr_pnl) > 0.08 else "weak"
        print(f"  {ind:<25} {corr_pnl:>12.4f} {corr_wr:>12.4f} {strength:>10}")
        correlations.append((ind, corr_pnl, corr_wr))

    # =========================================================================
    print()
    print("=" * 150)
    print("SECTION 3: Bucketed analysis — strategy perf by indicator buckets")
    print("=" * 150)

    # Test the most promising indicators
    for ind in ['vol_5d', 'vol_10d', 'atr_5d', 'atr_10d',
                'ret_5d', 'ret_10d', 'above_sma_20', 'above_sma_50',
                'dist_from_20d_high', 'max_range_10d', 'ret_std_5d',
                'down_days_5d']:
        if ind not in merged.columns: continue
        valid = merged[[ind, 'pnl', 'wr', 'n_trades']].dropna()
        if len(valid) < 10: continue

        if ind in ['above_sma_20', 'above_sma_50', 'above_sma_10']:
            # Binary indicator
            buckets = [(0, 0.5, "Below"), (0.5, 1.5, "Above")]
        elif ind == 'n_trades':
            buckets = [(0.5, 1.5, "1"), (1.5, 2.5, "2"), (2.5, 3.5, "3"), (3.5, 5.5, "4-5"), (5.5, 20.5, "6+")]
        elif ind == 'down_days_5d':
            buckets = [(0, 1, "0"), (1, 2, "1"), (2, 3, "2"), (3, 4, "3"), (4, 6, "4-5")]
        else:
            # Quantile-based buckets
            try:
                quantiles = valid[ind].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).values
                # Deduplicate quantiles
                unique_q = sorted(set(quantiles))
                if len(unique_q) < 3:
                    continue
                buckets = []
                for i in range(len(unique_q) - 1):
                    lo, hi = unique_q[i], unique_q[i+1]
                    if i == len(unique_q) - 2:
                        hi += 0.001
                    buckets.append((lo, hi, f"{lo:.2f}-{hi:.2f}"))
            except:
                continue

        print(f"\n  {ind}:")
        print(f"  {'Bucket':<20} {'Days':>6} {'Trades':>7} {'PnL':>10} {'Avg/Day':>10} {'WR%':>6} {'$/Trade':>10}")
        print("  " + "-" * 80)

        for lo, hi, label in buckets:
            bucket = valid[(valid[ind] >= lo) & (valid[ind] < hi)]
            if len(bucket) == 0: continue
            total_pnl = bucket['pnl'].sum()
            avg_day = bucket['pnl'].mean()
            avg_wr = bucket['wr'].mean()
            total_trades = bucket['n_trades'].sum()
            pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
            print(f"  {label:<20} {len(bucket):>6} {total_trades:>7} ${total_pnl:>9,.0f} ${avg_day:>9,.0f} {avg_wr:>5.1f} ${pnl_per_trade:>9,.0f}")

    # =========================================================================
    print()
    print("=" * 150)
    print("SECTION 4: WEEKLY analysis — rolling 5-day metrics")
    print("=" * 150)

    # Compute weekly aggregates
    merged_sorted = merged.sort_values('date')
    merged_sorted['week'] = pd.to_datetime(merged_sorted['date'].astype(str)).dt.isocalendar().week
    merged_sorted['year'] = pd.to_datetime(merged_sorted['date'].astype(str)).dt.year

    weekly = merged_sorted.groupby(['year', 'week']).agg(
        start_date=('date', 'first'),
        end_date=('date', 'last'),
        n_days=('date', 'count'),
        total_pnl=('pnl', 'sum'),
        total_trades=('n_trades', 'sum'),
        avg_vol_5d=('vol_5d', 'mean'),
        avg_atr_5d=('atr_5d', 'mean'),
        avg_ret_5d=('ret_5d', 'mean'),
        avg_above_sma20=('above_sma_20', 'mean'),
        avg_dist_high=('dist_from_20d_high', 'mean'),
    ).reset_index()

    weekly['wr_proxy'] = weekly['total_pnl'] / weekly['total_trades']

    # Show all weeks with worst performance
    print(f"\n  15 worst weeks:")
    print(f"  {'Week':>15} {'PnL':>10} {'Trades':>7} {'$/Trade':>8} {'Vol5d':>7} {'ATR5d':>7} {'Ret5d':>7} {'SMA20':>6} {'DistHi':>7}")
    print("  " + "-" * 95)
    worst = weekly.nsmallest(15, 'total_pnl')
    for _, w in worst.iterrows():
        ppt = w['total_pnl'] / w['total_trades'] if w['total_trades'] > 0 else 0
        print(f"  {str(w['start_date']):>15} ${w['total_pnl']:>9,.0f} {w['total_trades']:>7} ${ppt:>7,.0f} {w['avg_vol_5d']:>6.2f}% {w['avg_atr_5d']:>6.2f}% {w['avg_ret_5d']:>+6.2f}% {w['avg_above_sma20']:>5.1f} {w['avg_dist_high']:>6.2f}%")

    print(f"\n  15 best weeks:")
    print(f"  {'Week':>15} {'PnL':>10} {'Trades':>7} {'$/Trade':>8} {'Vol5d':>7} {'ATR5d':>7} {'Ret5d':>7} {'SMA20':>6} {'DistHi':>7}")
    print("  " + "-" * 95)
    best = weekly.nlargest(15, 'total_pnl')
    for _, w in best.iterrows():
        ppt = w['total_pnl'] / w['total_trades'] if w['total_trades'] > 0 else 0
        print(f"  {str(w['start_date']):>15} ${w['total_pnl']:>9,.0f} {w['total_trades']:>7} ${ppt:>7,.0f} {w['avg_vol_5d']:>6.2f}% {w['avg_atr_5d']:>6.2f}% {w['avg_ret_5d']:>+6.2f}% {w['avg_above_sma20']:>5.1f} {w['avg_dist_high']:>6.2f}%")

    # =========================================================================
    print()
    print("=" * 150)
    print("SECTION 5: REGIME SIGNAL TESTING — which thresholds best separate good from bad?")
    print("=" * 150)

    # For each candidate signal, find the threshold that maximizes separation
    # between "high risk" and "normal" periods

    signals = [
        ('vol_5d', 'above', [0.8, 1.0, 1.2, 1.4, 1.5, 1.7, 2.0]),
        ('vol_10d', 'above', [0.8, 1.0, 1.2, 1.4, 1.5, 1.7, 2.0]),
        ('atr_5d', 'above', [0.8, 1.0, 1.2, 1.5, 1.7, 2.0]),
        ('atr_10d', 'above', [0.8, 1.0, 1.2, 1.5, 1.7, 2.0]),
        ('ret_5d', 'below', [-4, -3, -2, -1, 0]),
        ('ret_10d', 'below', [-5, -4, -3, -2, -1, 0]),
        ('above_sma_20', 'below', [0.5]),
        ('above_sma_50', 'below', [0.5]),
        ('dist_from_20d_high', 'below', [-1, -2, -3, -4, -5, -7, -10]),
        ('max_range_10d', 'above', [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
        ('ret_std_5d', 'above', [0.5, 0.7, 1.0, 1.2, 1.5]),
        ('down_days_5d', 'above', [2.5, 3.5, 4.5]),
    ]

    print(f"\n  {'Signal':<20} {'Dir':>5} {'Thresh':>8} {'HiRisk Days':>11} {'HiRisk PnL':>11} {'HiRisk $/D':>10} {'Normal PnL':>11} {'Normal $/D':>10} {'Blocked$':>10}")
    print("  " + "-" * 120)

    for signal, direction, thresholds in signals:
        if signal not in merged.columns: continue
        valid = merged[[signal, 'pnl']].dropna()

        for thresh in thresholds:
            if direction == 'above':
                high_risk = valid[valid[signal] >= thresh]
                normal = valid[valid[signal] < thresh]
            else:
                high_risk = valid[valid[signal] <= thresh]
                normal = valid[valid[signal] > thresh]

            if len(high_risk) < 5 or len(normal) < 5: continue

            hr_pnl = high_risk['pnl'].sum()
            nr_pnl = normal['pnl'].sum()
            hr_avg = high_risk['pnl'].mean()
            nr_avg = normal['pnl'].mean()

            print(f"  {signal:<20} {direction:>5} {thresh:>8.1f} {len(high_risk):>11} ${hr_pnl:>10,.0f} ${hr_avg:>9,.0f} ${nr_pnl:>10,.0f} ${nr_avg:>9,.0f} ${-hr_pnl:>9,.0f}")

    # =========================================================================
    print()
    print("=" * 150)
    print("SECTION 6: COMPOSITE SIGNALS — combining multiple indicators")
    print("=" * 150)

    # Test combinations
    composites = [
        ("vol_5d > 1.5 AND below SMA20", lambda r: r['vol_5d'] > 1.5 and r['above_sma_20'] == 0),
        ("vol_5d > 1.2 AND below SMA20", lambda r: r['vol_5d'] > 1.2 and r['above_sma_20'] == 0),
        ("vol_5d > 1.0 AND below SMA20", lambda r: r['vol_5d'] > 1.0 and r['above_sma_20'] == 0),
        ("vol_5d > 1.5 AND below SMA50", lambda r: r['vol_5d'] > 1.5 and r['above_sma_50'] == 0),
        ("vol_5d > 1.2 AND below SMA50", lambda r: r['vol_5d'] > 1.2 and r['above_sma_50'] == 0),
        ("atr_5d > 1.5 AND below SMA20", lambda r: r['atr_5d'] > 1.5 and r['above_sma_20'] == 0),
        ("atr_5d > 1.2 AND below SMA20", lambda r: r['atr_5d'] > 1.2 and r['above_sma_20'] == 0),
        ("dist_20d_high < -5% AND vol_5d > 1.0", lambda r: r['dist_from_20d_high'] < -5 and r['vol_5d'] > 1.0),
        ("dist_20d_high < -3% AND vol_5d > 1.2", lambda r: r['dist_from_20d_high'] < -3 and r['vol_5d'] > 1.2),
        ("dist_20d_high < -3% AND vol_5d > 1.0", lambda r: r['dist_from_20d_high'] < -3 and r['vol_5d'] > 1.0),
        ("max_range_10d > 3.0 AND below SMA20", lambda r: r['max_range_10d'] > 3.0 and r['above_sma_20'] == 0),
        ("max_range_10d > 2.5 AND below SMA20", lambda r: r['max_range_10d'] > 2.5 and r['above_sma_20'] == 0),
        ("ret_std_5d > 1.0 AND below SMA20", lambda r: r['ret_std_5d'] > 1.0 and r['above_sma_20'] == 0),
        ("ret_std_5d > 1.0 AND ret_5d < -1", lambda r: r['ret_std_5d'] > 1.0 and r['ret_5d'] < -1),
        ("vol_5d > 1.2 AND ret_5d < 0", lambda r: r['vol_5d'] > 1.2 and r['ret_5d'] < 0),
        ("vol_5d > 1.0 AND ret_5d < -1", lambda r: r['vol_5d'] > 1.0 and r['ret_5d'] < -1),
        ("3+ down days in 5 AND vol_5d > 1.0", lambda r: r['down_days_5d'] >= 3 and r['vol_5d'] > 1.0),
        ("below SMA20 AND below SMA50", lambda r: r['above_sma_20'] == 0 and r['above_sma_50'] == 0),
    ]

    print(f"\n  {'Composite Signal':<45} {'HiRisk':>7} {'Normal':>7} {'HR PnL':>10} {'HR $/D':>8} {'NR PnL':>10} {'NR $/D':>8} {'HR WR':>6} {'NR WR':>6}")
    print("  " + "-" * 120)

    valid = merged.dropna(subset=['vol_5d', 'above_sma_20', 'above_sma_50',
                                    'dist_from_20d_high', 'ret_5d', 'atr_5d',
                                    'max_range_10d', 'ret_std_5d', 'down_days_5d'])

    for label, func in composites:
        try:
            high_risk = valid[valid.apply(func, axis=1)]
            normal = valid[~valid.apply(func, axis=1)]
        except:
            continue

        if len(high_risk) < 3: continue

        hr_pnl = high_risk['pnl'].sum()
        nr_pnl = normal['pnl'].sum()
        hr_avg = high_risk['pnl'].mean()
        nr_avg = normal['pnl'].mean()
        hr_wr = high_risk['wr'].mean()
        nr_wr = normal['wr'].mean()

        print(f"  {label:<45} {len(high_risk):>7} {len(normal):>7} ${hr_pnl:>9,.0f} ${hr_avg:>7,.0f} ${nr_pnl:>9,.0f} ${nr_avg:>7,.0f} {hr_wr:>5.1f} {nr_wr:>5.1f}")

    # =========================================================================
    print()
    print("=" * 150)
    print("SECTION 7: Apply best regime signal with cushion strategies")
    print("=" * 150)

    # For each trade, determine if it's a "high risk" day
    # Then apply cushion rules only on high-risk days, trade normally otherwise

    # Merge trades with SPY indicators
    trades_merged = trades_df.merge(spy[['date', 'vol_5d', 'vol_10d', 'atr_5d', 'atr_10d',
                                          'ret_5d', 'above_sma_20', 'above_sma_50',
                                          'dist_from_20d_high', 'max_range_10d',
                                          'ret_std_5d', 'down_days_5d']],
                                     on='date', how='left')

    # Define regime signals to test
    regime_defs = [
        ("vol_5d > 1.5 AND below SMA20",
         lambda r: r.get('vol_5d', 0) > 1.5 and r.get('above_sma_20', 1) == 0),
        ("vol_5d > 1.2 AND below SMA20",
         lambda r: r.get('vol_5d', 0) > 1.2 and r.get('above_sma_20', 1) == 0),
        ("vol_5d > 1.0 AND below SMA20",
         lambda r: r.get('vol_5d', 0) > 1.0 and r.get('above_sma_20', 1) == 0),
        ("atr_5d > 1.5 AND below SMA20",
         lambda r: r.get('atr_5d', 0) > 1.5 and r.get('above_sma_20', 1) == 0),
        ("dist_20d_high < -5% AND vol_5d > 1.0",
         lambda r: r.get('dist_from_20d_high', 0) < -5 and r.get('vol_5d', 0) > 1.0),
        ("dist_20d_high < -3% AND vol_5d > 1.2",
         lambda r: r.get('dist_from_20d_high', 0) < -3 and r.get('vol_5d', 0) > 1.2),
        ("max_range_10d > 3.0 AND below SMA20",
         lambda r: r.get('max_range_10d', 0) > 3.0 and r.get('above_sma_20', 1) == 0),
        ("below SMA20 AND below SMA50",
         lambda r: r.get('above_sma_20', 1) == 0 and r.get('above_sma_50', 1) == 0),
    ]

    # For each regime signal, test: normal days at 100%, high-risk days at various scales
    print(f"\n  Testing: 100% on normal days, X% on high-risk days + max 5 trades + 3 consec stop on HR days")
    print()

    BASE_PNL = 247088
    BASE_DD = 28781

    print(f"  {'Regime Signal':<45} {'HR%':>4} {'PnL':>10} {'Δ%':>7} {'MaxDD':>8} {'Δ%':>7} {'MM PnL':>9} {'MM DD':>8} {'Calmr':>6} {'HR days':>7}")
    print("  " + "-" * 130)

    # Also print baseline
    print(f"  {'BASELINE (no regime)':<45} {'100':>4} ${247088:>9,.0f} {'+0.0%':>7} ${28781:>7,.0f} {'+0.0%':>7} ${-18888:>8,.0f} ${28057:>7,.0f} {'8.6':>6} {'0':>7}")

    for label, regime_func in regime_defs:
        # Pre-compute which dates are high-risk
        hr_dates = set()
        for d in trades_merged['date'].unique():
            day_data = trades_merged[trades_merged['date'] == d].iloc[0]
            if regime_func(day_data):
                hr_dates.add(d)

        for hr_scale in [0.0, 0.25, 0.50, 0.75]:
            tp, dpd = [], defaultdict(float)
            for td, dt in trades_merged.groupby('date'):
                dt = dt.sort_values('entry_time')
                is_hr = td in hr_dates
                scale = hr_scale if is_hr else 1.0
                cum, cl, stopped, count = 0.0, 0, False, 0

                for _, r in dt.iterrows():
                    if stopped: continue
                    if is_hr and count >= 5: continue  # max 5 on HR days

                    pnl = r['pnl'] * scale
                    tp.append(pnl); dpd[td] += pnl
                    cum += pnl; count += 1

                    if r['pnl'] <= 0:
                        cl += 1
                        if is_hr and cl >= 3: stopped = True  # 3 consec stop on HR
                    else:
                        cl = 0

            daily = [dpd[d] for d in sorted(dpd)]
            total = sum(daily)
            c = np.cumsum(daily)
            pk = np.maximum.accumulate(c)
            dd = abs(min(c - pk))

            mm_pnls = [v for d, v in sorted(dpd.items()) if date(2025, 3, 1) <= d <= date(2025, 5, 31)]
            mm_pnl = sum(mm_pnls)
            mc = np.cumsum(mm_pnls) if mm_pnls else [0]
            mp = np.maximum.accumulate(mc)
            mm_dd = abs(min(mc - mp)) if len(mm_pnls) > 0 else 0

            calmar = total / dd if dd > 0 else 0
            pd_ = (total / BASE_PNL - 1) * 100
            dd_ = (dd / BASE_DD - 1) * 100

            hr_pct_label = f"{hr_scale*100:.0f}" if hr_scale > 0 else "SKIP"
            print(f"  {label:<45} {hr_pct_label:>4} ${total:>9,.0f} ({pd_:>+5.1f}%) ${dd:>7,.0f} ({dd_:>+5.1f}%) ${mm_pnl:>8,.0f} ${mm_dd:>7,.0f} {calmar:>6.1f} {len(hr_dates):>7}")


if __name__ == '__main__':
    main()
