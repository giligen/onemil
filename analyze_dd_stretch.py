"""Find the worst DD stretch in each walk-forward split (config: baseline A).

Pipeline: N=3, risk=$2K, filter+cap3+Q4-pref+adaptive (no correlation fixes).
For each split, identify peak→trough dates and show EVERY day in between.
"""
from __future__ import annotations

import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from study_orb import SPLITS
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, ADAPTIVE_MULT_MIN, ADAPTIVE_MULT_MAX,
    fit_quintile_cutoffs, assign_quintile,
)

ACCOUNT = 100_000
N = 3
RISK = 2000
OLD_POS = 50_000
MIN_STOP_PCT = 1.0
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}


def apply_rp(df):
    df = df.copy()
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    cap = ACCOUNT / N
    uncap = RISK / (stop / 100.0)
    df['_rp_position'] = uncap.clip(upper=cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS
    return df


def fit_mults(tk):
    avg = float(tk['_rp_pnl'].mean())
    out = {}
    for q in ['Q1','Q2','Q3','Q4','Q5']:
        sub = tk[tk['_quintile'] == q]
        out[q] = max(ADAPTIVE_MULT_MIN, min(ADAPTIVE_MULT_MAX,
                                             float(sub['_rp_pnl'].mean()) / avg))
    return out


def pick_top3(dg):
    d = dg.copy()
    d['_q_rank'] = d['_quintile'].map(Q_ORDER)
    return d.sort_values(['_q_rank', '_composite'], ascending=[True, False]).head(N)


def run_split(df, tr_s, tr_e, te_s, te_e):
    df = apply_rp(df)
    df['date'] = pd.to_datetime(df['date'])
    train = df[(df['date'] >= tr_s) & (df['date'] <= tr_e)]
    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    train = df[(df['date'] >= tr_s) & (df['date'] <= tr_e)]
    test = df[(df['date'] >= te_s) & (df['date'] <= te_e)]
    test_k = test[test['_composite'] >= FILTER_THRESHOLD].copy()
    train_k = train[train['_composite'] >= FILTER_THRESHOLD].copy()
    cutoffs = fit_quintile_cutoffs(train_k['_composite'])
    train_k['_quintile'] = assign_quintile(train_k['_composite'], cutoffs)
    test_k['_quintile'] = assign_quintile(test_k['_composite'], cutoffs)
    mults = fit_mults(train_k)

    sel = pd.concat([pick_top3(dg) for _, dg in test_k.groupby('date')])
    sel['_sized_pnl'] = sel.apply(
        lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)

    daily = sel.groupby('date').agg(
        daily_pnl=('_sized_pnl', 'sum'),
        n_picks=('_rp_pnl', 'count'),
    ).reset_index().sort_values('date').reset_index(drop=True)
    daily['cum'] = daily['daily_pnl'].cumsum()
    # Running peak and DD
    running_peak = -np.inf
    peak_track = []
    dd_track = []
    for c in daily['cum']:
        running_peak = max(running_peak, c)
        peak_track.append(running_peak)
        dd_track.append(c - running_peak)
    daily['peak'] = peak_track
    daily['dd'] = dd_track

    return sel, daily, mults


def find_worst_stretch(daily):
    peak_idx = 0
    worst_dd = 0.0
    worst_peak = 0
    worst_trough = 0
    running_peak = -np.inf
    for i, c in enumerate(daily['cum']):
        if c > running_peak:
            running_peak = c
            peak_idx = i
        d = c - running_peak
        if d < worst_dd:
            worst_dd = d
            worst_peak = peak_idx
            worst_trough = i
    return worst_peak, worst_trough, worst_dd


def main():
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl','date','range_size_pct'])
    print(f"Loaded {len(df):,} trades\n")

    for split_name, tr_s, tr_e, te_s, te_e in SPLITS:
        print(f"{'='*120}")
        print(f"SPLIT: {split_name}")
        print(f"Train {tr_s} → {tr_e}   Test {te_s} → {te_e}")
        print(f"{'='*120}")
        sel, daily, mults = run_split(df, tr_s, tr_e, te_s, te_e)
        print(f"Adaptive mults: " +
              " ".join(f"{q}={mults[q]:.2f}x" for q in ['Q1','Q2','Q3','Q4','Q5']))
        print(f"Total test P&L: ${daily['daily_pnl'].sum():+,.0f}")
        p, t, dd = find_worst_stretch(daily)
        pd_peak = daily.iloc[p]
        pd_trough = daily.iloc[t]
        print(f"\nMax DD: ${dd:+,.0f}")
        print(f"Peak:   {pd_peak['date'].date()} (equity ${pd_peak['cum']:+,.0f})")
        print(f"Trough: {pd_trough['date'].date()} (equity ${pd_trough['cum']:+,.0f})")
        print(f"Span:   {t - p} trading days ({(pd_trough['date'] - pd_peak['date']).days} calendar days)")

        # Show EVERY day in the stretch
        print(f"\n{'date':<12} {'wd':<4} {'day P&L':>10} {'equity':>11} "
              f"{'DD':>10} {'n':>3} {'picks'}")
        print('-' * 120)

        stretch = daily.iloc[p:t+1].reset_index(drop=True)
        for _, row in stretch.iterrows():
            # Get the actual picks for this date
            day_sel = sel[sel['date'] == row['date']].sort_values('_composite', ascending=False)
            parts = []
            for _, r in day_sel.iterrows():
                arrow = '✓' if r['_sized_pnl'] > 0 else '✗'
                parts.append(f"{r['symbol']}({r['_quintile']}){arrow}${r['_sized_pnl']:+,.0f}")
            picks_str = ', '.join(parts)
            if len(picks_str) > 70:
                picks_str = picks_str[:68] + '…'

            wd = row['date'].strftime('%a')
            print(f"{row['date'].date().isoformat():<12} {wd:<4} "
                  f"${row['daily_pnl']:>+8,.0f} ${row['cum']:>+9,.0f} "
                  f"${row['dd']:>+8,.0f} {int(row['n_picks']):>3} {picks_str}")

        # Summary stats of the stretch
        print(f"\n  Stretch stats:")
        print(f"    Days: {len(stretch)}")
        print(f"    Winning days: {(stretch['daily_pnl'] > 0).sum()}")
        print(f"    Losing days: {(stretch['daily_pnl'] < 0).sum()}")
        print(f"    Total stretch P&L: ${stretch['daily_pnl'].sum():+,.0f}")
        print(f"    Worst single day: ${stretch['daily_pnl'].min():+,.0f} "
              f"on {stretch.loc[stretch['daily_pnl'].idxmin(), 'date'].date()}")
        print(f"    Best single day:  ${stretch['daily_pnl'].max():+,.0f} "
              f"on {stretch.loc[stretch['daily_pnl'].idxmax(), 'date'].date()}")
        print()


if __name__ == '__main__':
    main()
