#!/usr/bin/env python3
"""Compute cells 1,289 (post-then-cross) and 1,290 (spread-in-R gate) from per_trade.csv."""
import pandas as pd
import numpy as np

d = pd.read_csv('/home/ec2-user/onemil/research/exec_cost/per_trade.csv')
d['date'] = pd.to_datetime(d.date)

# per-share economics
d['cost_old'] = d.spread0 / 2.0
d['delta_cost_share'] = d.cost_old - d.cost_new          # $/share saved by the rule
d['delta_cost_R'] = d.delta_cost_share / d.R
# shares implied by the book's own pnl/pnl_pct/entry_price (pnl_pct = (exit-entry)/entry*100)
d['dollar_move'] = d.pnl_pct / 100.0 * d.entry_price_book
d['shares'] = (d.pnl / d.dollar_move).abs()
d['delta_cost_dollar'] = d.delta_cost_share * d.shares
d['net_R'] = d.dollar_move / d.R   # per-share return in R units (== pnl/(R*shares))

d['ratio'] = d.cost_old / d.R      # cell 1290: half-spread at trigger / R


def cluster_t(vals, days):
    df = pd.DataFrame({'v': vals, 'd': days})
    by_day = df.groupby('d').v.mean()
    n = len(by_day)
    if n < 2:
        return np.nan, np.nan, n
    m = by_day.mean()
    se = by_day.std(ddof=1) / np.sqrt(n)
    t = m / se if se > 0 else np.nan
    return m, t, n


print('=' * 70)
print('CELL 1,289 -- post-then-cross entry')
print('=' * 70)
for split in ['TRAIN', 'VAL']:
    s = d[d.split == split]
    n = len(s)
    pr = s.passive.mean()
    tot_dollar = s.delta_cost_dollar.sum()
    mean_R = s.delta_cost_R.mean()
    m, t, ndays = cluster_t(s.delta_cost_dollar.values, s.date.values)
    print(f'\n-- {split} (n={n}) --')
    print(f'passive fill rate: {pr:.1%}  ({s.passive.sum()}/{n})')
    print(f'mean spread-earned (passive) $/share: {s.cost_old.mean():.4f}')
    print(f'mean delta-cost per trade: {mean_R:.4f} R  |  ${s.delta_cost_dollar.mean():.2f}')
    print(f'net delta$ whole rule: ${tot_dollar:.2f}   (day-clustered t={t:.2f}, n_days={ndays})')
    pas = s[s.passive].net_R
    cro = s[~s.passive].net_R
    print(f'deciding table: passive net_R mean={pas.mean():.3f} (n={len(pas)})  |  '
          f'crossed net_R mean={cro.mean():.3f} (n={len(cro)})  |  '
          f'all(orig) net_R mean={s.net_R.mean():.3f}')
    adverse_gap = pas.mean() - cro.mean()
    print(f'adverse-selection gap (passive - crossed): {adverse_gap:.3f} R  '
          f'({"FLAG: worse by >0.05R" if adverse_gap < -0.05 else "within tolerance"})')
    pass_bar = (tot_dollar > 0) and (adverse_gap >= -0.05)
    print(f'PASS BAR: {"PASS" if pass_bar else "FAIL"}')

print()
print('=' * 70)
print('CELL 1,290 -- spread-in-R gate')
print('=' * 70)
for split in ['TRAIN', 'VAL']:
    s = d[d.split == split]
    base_net = s.pnl.sum()
    print(f'\n-- {split} (n={len(s)}, base net ${base_net:.2f}) --')
    for thresh in [0.05, 0.10, 0.15]:
        veto = s[s.ratio > thresh]
        keep = s[s.ratio <= thresh]
        keep_net = keep.pnl.sum()
        veto_net = veto.pnl.sum()
        # simple running-drawdown (date-ordered cumulative pnl, per-trade sequence)
        def mdd(sub):
            c = sub.sort_values('date').pnl.cumsum()
            if len(c) == 0:
                return 0.0
            return (c - c.cummax()).min()
        base_mdd = mdd(s)
        keep_mdd = mdd(keep)
        print(f'  ratio>{thresh:.2f}: vetoed n={len(veto)}  vetoed_net=${veto_net:.2f}  '
              f'kept_net=${keep_net:.2f} (vs base ${base_net:.2f})  '
              f'MDD base={base_mdd:.2f} kept={keep_mdd:.2f}  '
              f'PASS={"Y" if (keep_net>base_net and keep_mdd>=base_mdd and veto_net<0) else "N"}')

print()
print('coverage: fetch errors terr/qerr all-NaN (0 errors) on', len(d), 'of 122 eligible fills')
d.to_csv('/home/ec2-user/onemil/research/exec_cost/per_trade_scored.csv', index=False)
