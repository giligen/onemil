#!/usr/bin/env python3
"""Pass 2 step 3: stats for REPORT.md — signal vs control by split, t-stats
(day-clustered via day-mean two-sample t), F-quintile monotonicity, MDE,
entries/week, $/week at $66K book (1% risk/trade = $660, R defined as 2%
of price). Writes p2_trades_TRAIN.csv / p2_trades_VAL.csv for cadence_bar.py
and prints a stats block to paste into REPORT.md.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = '/home/ec2-user/onemil'
OUT = os.path.join(ROOT, 'research', 'lev_rebalance')


def day_means(df, date_col='date', val_col='pnl_R_moc'):
    return df.groupby(date_col)[val_col].mean()


def two_sample_day_clustered_t(sig, ctrl, date_col='date', val_col='pnl_R_moc'):
    sm = day_means(sig, date_col, val_col)
    cm = day_means(ctrl, date_col, val_col)
    t, p = stats.ttest_ind(sm.values, cm.values, equal_var=False)
    diff = sm.mean() - cm.mean()
    return diff, t, p, len(sm), len(cm)


def one_sample_day_clustered_t(df, date_col='date', val_col='pnl_R_moc'):
    dm = day_means(df, date_col, val_col)
    if len(dm) < 2:
        return dm.mean() if len(dm) else np.nan, np.nan, len(dm)
    t, p = stats.ttest_1samp(dm.values, 0.0)
    return dm.mean(), t, len(dm)


def mde(df, date_col='date', val_col='pnl_R_moc', power=0.8, alpha=0.05):
    dm = day_means(df, date_col, val_col)
    n = len(dm)
    if n < 2:
        return np.nan
    sd = dm.std(ddof=1)
    from scipy.stats import norm
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    return (z_a + z_b) * sd / np.sqrt(n)


def main():
    sig = pd.read_csv(os.path.join(OUT, 'p2_signal_trades.csv'), parse_dates=['date'])
    ctrl = pd.read_csv(os.path.join(OUT, 'p2_control_trades.csv'), parse_dates=['date'])
    qual = pd.read_csv(os.path.join(OUT, 'p2_signal_days_qualified.csv'), parse_dates=['date'])

    out = {}
    for split in ['TRAIN', 'VAL']:
        s = sig[sig['split'] == split]
        c = ctrl[ctrl['split'] == split]
        s_mean, s_t, s_ndays = one_sample_day_clustered_t(s)
        c_mean, c_t, c_ndays = one_sample_day_clustered_t(c)
        diff, dt, dp, sn, cn = two_sample_day_clustered_t(s, c)
        weeks = max((s['date'].max() - s['date'].min()).days / 7.0, 1.0) if len(s) else 1.0
        entries_per_week = len(s) / weeks
        m = mde(s)
        out[split] = dict(
            n_signal=len(s), n_signal_days=int(s_ndays), signal_mean_R=float(s_mean), signal_t=float(s_t),
            n_control=len(c), n_control_days=int(c_ndays), control_mean_R=float(c_mean), control_t=float(c_t),
            diff_R=float(diff), diff_t=float(dt), diff_p=float(dp),
            entries_per_week=float(entries_per_week), mde_R=float(m),
        )
        # % of price
        out[split]['signal_mean_pct_price'] = out[split]['signal_mean_R'] * 0.02 * 100
        out[split]['diff_pct_price'] = out[split]['diff_R'] * 0.02 * 100

    # stop diagnostic (cell 1294)
    for split in ['TRAIN', 'VAL']:
        s = sig[sig['split'] == split].copy()
        s['pnl_R_moc_tmp'] = s['pnl_R_stop']
        sm, st, snd = one_sample_day_clustered_t(s, val_col='pnl_R_moc_tmp')
        out[split]['stop_mean_R'] = float(sm)

    # long/short cells
    for side, name in [(1, 'LONG'), (-1, 'SHORT')]:
        s = sig[sig['side'] == side]
        for split in ['TRAIN', 'VAL']:
            ss = s[s['split'] == split]
            m, t, nd = one_sample_day_clustered_t(ss)
            out.setdefault(f'{name}_{split}', {})['n'] = len(ss)
            out[f'{name}_{split}']['mean_R'] = float(m)
            out[f'{name}_{split}']['t'] = float(t) if not np.isnan(t) else None
            out[f'{name}_{split}']['n_days'] = int(nd)

    # F-quintile monotonicity on qualified (pre-threshold) days, TRAIN+VAL, has moc_close
    q = qual.dropna(subset=['F']).copy()
    q = q.merge(sig[['symbol', 'date', 'pnl_R_moc']], on=['symbol', 'date'], how='left')
    # need pnl for ALL qualified days, not just selected ones -> recompute pnl using qualified rows directly
    # qual has eff_entry/side/moc_close already (from p2_signal_days_qualified.csv)
    q['pnl_frac'] = (q['moc_close'] / q['eff_entry'] - 1.0) * q['side']
    q['pnl_R'] = q['pnl_frac'] / 0.02
    q = q[q['sho_gate'] == True]
    try:
        q['F_quintile'] = pd.qcut(q['F'], 5, labels=False, duplicates='drop')
        fq = q.groupby('F_quintile')['pnl_R'].mean()
        f_corr = q['F'].corr(q['pnl_R'])
    except Exception as e:
        fq = pd.Series(dtype=float)
        f_corr = np.nan
    out['F_quintile_means'] = {str(k): float(v) for k, v in fq.items()}
    out['F_corr'] = float(f_corr) if not np.isnan(f_corr) else None
    out['F_n_qualified_days'] = int(len(q))

    # $/week at $66K book: 1% risk/trade = $660, combined TRAIN+VAL
    R_DOLLAR = 0.01 * 66000.0
    combined = sig.copy()
    weeks_c = max((combined['date'].max() - combined['date'].min()).days / 7.0, 1.0)
    epw_combined = len(combined) / weeks_c
    dpw = epw_combined * combined['pnl_R_moc'].mean() * R_DOLLAR
    out['book'] = dict(entries_per_week_combined=float(epw_combined),
                        dollars_per_week_at_66k=float(dpw), r_dollar=R_DOLLAR)

    # write cadence CSVs (combined book, both splits)
    for split in ['TRAIN', 'VAL']:
        s = sig[sig['split'] == split][['date', 'pnl_R_moc', 'symbol']].rename(columns={'pnl_R_moc': 'pnl_R'})
        s['date'] = s['date'].dt.strftime('%Y-%m-%d')
        s.to_csv(os.path.join(OUT, f'p2_trades_{split}.csv'), index=False)

    with open(os.path.join(OUT, 'p2_stats.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))


if __name__ == '__main__':
    main()
