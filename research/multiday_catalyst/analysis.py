#!/usr/bin/env python3
"""Final stats for multiday_catalyst cells 1,285/1,286. Reads trades_{signal,control}_raw.csv.
TEST (>=2026-06-01) is SEALED: counted only, never scored. 2024H2 has no news coverage -> not runnable
for either cell (corrected from the PREREG addendum's control-only claim; recorded as a mid-run change)."""
import json, math, os
import numpy as np
import pandas as pd

OUT = '/home/ec2-user/onemil/research/multiday_catalyst'
NEWS_START = pd.Timestamp('2025-01-02')
TRAIN = (pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31'))
VAL = (pd.Timestamp('2026-01-01'), pd.Timestamp('2026-05-31'))
TEST_START = pd.Timestamp('2026-06-01')
R_DOLLARS_66K = 660.0  # 1% of $66K


def load(name):
    df = pd.read_csv(os.path.join(OUT, name), parse_dates=['date', 'exit_day'])
    return df


def clustered_t(x, cluster):
    """Day-clustered (Huber/White) t-stat for the mean of x, clusters = date."""
    df = pd.DataFrame({'x': x, 'c': cluster})
    n = len(df)
    grand_mean = df['x'].mean()
    g = df.groupby('c')['x'].agg(['mean', 'count'])
    # cluster-robust variance of the mean (Liang-Zeger style, simplified: sum of cluster sums of residuals)
    resid = df['x'] - grand_mean
    df['resid'] = resid
    cluster_sums = df.groupby('c')['resid'].sum()
    V = (cluster_sums ** 2).sum() / (n ** 2)
    se = math.sqrt(V) if V > 0 else float('nan')
    t = grand_mean / se if se and se > 0 else float('nan')
    return grand_mean, se, t, df['c'].nunique()


def iid_t(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return x.mean() if n else float('nan'), float('nan'), float('nan')
    se = x.std(ddof=1) / math.sqrt(n)
    t = x.mean() / se if se > 0 else float('nan')
    return x.mean(), se, t


def mdd_R(pnls_ordered):
    cum = np.cumsum(pnls_ordered)
    peak = np.maximum.accumulate(np.concatenate([[0], cum]))[1:]
    dd = cum - peak
    return dd.min() if len(dd) else 0.0


def mde(se, n, alpha=0.05, power=0.8):
    """Minimum detectable effect, two-sided, normal approx: (z_a/2+z_b)*se."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return z * se


def summarize(df, label):
    if len(df) == 0:
        return dict(label=label, n=0)
    mean_c, se_c, t_c, nclust = clustered_t(df['pnl_R'], df['date'])
    mean_i, se_i, t_i = iid_t(df['pnl_R'])
    gross_mean = df['pnl_R_gross'].mean()
    cost_R = gross_mean - mean_c
    wr = (df['pnl_R'] > 0).mean()
    wins = df.loc[df['pnl_R'] > 0, 'pnl_R']
    losses = df.loc[df['pnl_R'] <= 0, 'pnl_R']
    srt = df.sort_values('pnl_R')
    n = len(srt)
    k1 = max(1, int(round(n * 0.01)))
    k5 = max(1, int(round(n * 0.05)))
    ex1 = srt['pnl_R'].iloc[:-k1].mean() if n > k1 else float('nan')
    ex5 = srt['pnl_R'].iloc[:-k5].mean() if n > k5 else float('nan')
    top5_share = srt['pnl_R'].iloc[-5:].sum() / srt['pnl_R'].sum() if srt['pnl_R'].sum() != 0 and n >= 5 else float('nan')
    ordered = df.sort_values('date')['pnl_R'].values
    mdd_r = mdd_R(ordered)
    mdd_dollar = mdd_r * R_DOLLARS_66K  # approx: 1 unit R ~ R_dollars per trade at book risk sizing
    weeks = df['date'].dt.to_period('W').nunique()
    entries_per_week = n / weeks if weeks else float('nan')
    wrap = df[df['asset_class'] == 'wrapper']['pnl_R']
    stock = df[df['asset_class'] == 'stock']['pnl_R']
    hold_profile = {}
    for i in (1, 2, 3):
        col = f'R_d{i}'
        if col in df.columns:
            v = df[col].dropna()
            hold_profile[f'D+{i}'] = float(v.mean()) if len(v) else None
    try:
        m_mde = mde(se_c, nclust)
    except Exception:
        m_mde = float('nan')
    return dict(
        label=label, n=n, weeks=weeks, entries_per_week=entries_per_week,
        net_R_clustered=mean_c, se_clustered=se_c, t_clustered=t_c, n_clusters=nclust,
        net_R_iid=mean_i, se_iid=se_i, t_iid=t_i,
        gross_R=gross_mean, cost_R=cost_R, WR=wr,
        avg_win_R=wins.mean() if len(wins) else float('nan'),
        avg_loss_R=losses.mean() if len(losses) else float('nan'),
        ex_top1pct_R=ex1, ex_top5pct_R=ex5, top5_share=top5_share,
        MDD_R=mdd_r, MDD_dollar_approx=mdd_dollar,
        wrapper_n=len(wrap), wrapper_mean_R=float(wrap.mean()) if len(wrap) else None,
        stock_n=len(stock), stock_mean_R=float(stock.mean()) if len(stock) else None,
        holding_profile=hold_profile, MDE_R_clustered=m_mde,
    )


def diff_test(sig, ctl, dates_union):
    """Signal-minus-control, day-clustered on the union of trade dates (paired by day: mean(sig|day) -
    mean(ctl|day)) where both exist that day; simpler unpaired clustered diff-of-means with a
    Welch-style combined clustered SE (conservative: sum of variances)."""
    if len(sig) == 0 or len(ctl) == 0:
        return dict(n_sig=len(sig), n_ctl=len(ctl))
    ms, ses, ts, _ = clustered_t(sig['pnl_R'], sig['date'])
    mc, sec, tc, _ = clustered_t(ctl['pnl_R'], ctl['date'])
    diff = ms - mc
    se_diff = math.sqrt(ses ** 2 + sec ** 2) if not (math.isnan(ses) or math.isnan(sec)) else float('nan')
    t_diff = diff / se_diff if se_diff and se_diff > 0 else float('nan')
    return dict(signal_R=ms, control_R=mc, diff_R=diff, se_diff=se_diff, t_diff=t_diff)


def book_sim(df, slots=10, equity=66000.0, risk_pct=0.01):
    """10-concurrent, notional-capped book. date = entry day D, exit_day already computed.
    Returns trades actually admitted + per-day position counts."""
    if len(df) == 0:
        return df.iloc[0:0], pd.Series(dtype=float)
    df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
    open_positions = []  # list of (exit_day, notional)
    admitted_idx = []
    per_day_count = {}
    open_symbols = {}
    for idx, row in df.iterrows():
        d = row['date']
        # free positions that exited before d
        open_positions = [(ed, nt) for (ed, nt) in open_positions if ed >= d]
        open_symbols = {s: ed for s, ed in open_symbols.items() if ed >= d}
        if row['symbol'] in open_symbols:
            continue
        if len(open_positions) >= slots:
            continue
        risk_dollars = risk_pct * equity
        shares = risk_dollars / row['R_raw'] if row['R_raw'] > 0 else 0
        notional = shares * row['entry_paid']
        cur_notional = sum(nt for _, nt in open_positions)
        if cur_notional + notional > equity:
            continue
        open_positions.append((row['exit_day'], notional))
        open_symbols[row['symbol']] = row['exit_day']
        admitted_idx.append(idx)
        per_day_count[d] = per_day_count.get(d, 0) + 1
    return df.loc[admitted_idx], pd.Series(per_day_count)


def main():
    sig = load('trades_signal_raw.csv')
    ctl_all = load('trades_control_raw.csv')
    ctl = ctl_all[ctl_all['date'] >= NEWS_START].copy()  # news-covered era only: absence of a news
    # record before 2025-01-02 is missing data, not confirmed "no news" -- correction vs PREREG addendum.
    n_ctl_dropped_precoverage = len(ctl_all) - len(ctl)

    results = {'n_ctl_dropped_precoverage_2024H2': int(n_ctl_dropped_precoverage)}

    for split_name, (lo, hi) in (('TRAIN', TRAIN), ('VAL', VAL)):
        s = sig[(sig['date'] >= lo) & (sig['date'] <= hi)]
        c = ctl[(ctl['date'] >= lo) & (ctl['date'] <= hi)]
        results[f'{split_name}_signal'] = summarize(s, f'{split_name} signal')
        results[f'{split_name}_control'] = summarize(c, f'{split_name} control')
        results[f'{split_name}_diff'] = diff_test(s, c, None)
        bsig, _ = book_sim(s)
        bctl, _ = book_sim(c)
        results[f'{split_name}_signal_book10'] = summarize(bsig, f'{split_name} signal (10-slot book)')
        results[f'{split_name}_positions_per_day_signal'] = float(bsig.groupby('date').size().mean()) if len(bsig) else None

    # TRAIN halves same-signed check
    h1 = sig[(sig['date'] >= '2025-01-01') & (sig['date'] <= '2025-06-30')]
    h2 = sig[(sig['date'] >= '2025-07-01') & (sig['date'] <= '2025-12-31')]
    results['TRAIN_H1_signal_meanR'] = float(h1['pnl_R'].mean()) if len(h1) else None
    results['TRAIN_H2_signal_meanR'] = float(h2['pnl_R'].mean()) if len(h2) else None
    results['TRAIN_H1_n'] = len(h1)
    results['TRAIN_H2_n'] = len(h2)

    # TEST sealed -- count only, never score
    test_sig_n = int((sig['date'] >= TEST_START).sum())
    test_ctl_n = int((ctl['date'] >= TEST_START).sum())
    results['TEST_sealed_signal_n'] = test_sig_n
    results['TEST_sealed_control_n'] = test_ctl_n

    with open(os.path.join(OUT, 'final_stats.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # write cadence-bar CSVs (entry date D, pnl_R, symbol) for VAL and TRAIN, unlimited-slot version
    for split_name, (lo, hi) in (('TRAIN', TRAIN), ('VAL', VAL)):
        s = sig[(sig['date'] >= lo) & (sig['date'] <= hi)][['date', 'pnl_R', 'symbol']]
        s.to_csv(os.path.join(OUT, f'cadence_signal_{split_name}.csv'), index=False)

    print(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()
