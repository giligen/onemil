#!/usr/bin/env python3
"""a1 — Q2: is the out-of-sample strength a market regime?

Builds per-day regime variables (SPY realized vol, cross-sectional dispersion of daily stock returns,
qualifying-candidate count), aggregates the book to weeks, and regresses weekly R on the regime
variables with and without split dummies.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
A = 'research/bf_zero2/audit_stats'
OUT = []
def P(*a):
    s = ' '.join(str(x) for x in a); OUT.append(s); print(s, flush=True)

book = pd.read_csv('research/bf_zero2/f6_2r_book.csv')
pool = pd.read_parquet(f'{A}/pool_f6.parquet')

# ---------- regime variables from the daily panel ----------
import pyarrow.parquet as pq
pf = pq.ParquetFile('research/lit_review_2026/daily_panel.parquet')
spy, disp = [], []
for i in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(i, columns=['symbol', 'bar_date', 'ret_cc', 'ret_id', 'dvol20', 'close']).to_pandas()
    spy.append(t[t.symbol == 'SPY'])
    liq = t[(t.dvol20 >= 2e6) & (t.close >= 5) & t.ret_cc.notna() & np.isfinite(t.ret_cc)].copy()
    liq['ret_cc'] = liq.ret_cc.clip(-0.5, 0.5)
    disp.append(liq.groupby('bar_date').ret_cc.agg(['std', 'count', lambda s: s.abs().mean()]).rename(columns={'<lambda_0>': 'mad'}))
    del t
spy = pd.concat(spy).sort_values('bar_date')
disp = pd.concat(disp).groupby(level=0).apply(
    lambda g: pd.Series({'xs_std': np.sqrt((g['std'] ** 2 * g['count']).sum() / g['count'].sum()),
                         'xs_mad': (g['mad'] * g['count']).sum() / g['count'].sum(),
                         'n_liq': g['count'].sum()}))
disp.index.name = 'day'
P(f'SPY days {len(spy)}  dispersion days {len(disp)}')

spy = spy[['bar_date', 'ret_cc', 'ret_id']].rename(columns={'bar_date': 'day'}).set_index('day')
spy['spy_absret'] = spy.ret_cc.abs()
spy['spy_rv20'] = spy.ret_cc.rolling(20).std() * np.sqrt(252)

day = spy.join(disp, how='inner')
cand = pool.groupby('day', observed=True).size().rename('n_cand')
day = day.join(cand, how='left')
day['n_cand'] = day.n_cand.fillna(0)
day = day.loc[(day.index >= '2025-01-01') & (day.index <= '2026-09-30')]
day['spy_rv20'] = day.spy_rv20.fillna(day.spy_rv20.median())
day['xs_std'] = day.xs_std.fillna(day.xs_std.median()); day['xs_mad'] = day.xs_mad.fillna(day.xs_mad.median())
day['split'] = np.where(day.index < '2026-01-01', 'TRAIN', np.where(day.index < '2026-06-01', 'VAL', 'TEST'))

P('\n## Q2a — regime level by split (trading days in the book period)')
P(f"{'split':6} {'days':>5} {'SPY rv20 ann':>13} {'SPY |ret| bps':>14} {'xs disp bps':>12} {'xs MAD bps':>11} {'F6 cand/day':>12}")
for s in ('TRAIN', 'VAL', 'TEST'):
    d = day[day.split == s]
    P(f'{s:6} {len(d):5d} {d.spy_rv20.mean()*100:13.2f} {d.spy_absret.mean()*1e4:14.1f} '
      f'{d.xs_std.mean()*1e4:12.1f} {d.xs_mad.mean()*1e4:11.1f} {d.n_cand.mean():12.1f}')

# ---------- book weekly series ----------
book['day'] = book.day.astype(str)
wk_book = book.groupby(['split', 'wk'], observed=True).net.agg(['sum', 'count']).reset_index()
alldays = pool[['day', 'wk', 'split']].drop_duplicates('day')
wk_all = alldays.groupby(['split', 'wk'], observed=True).size().rename('n_days').reset_index()
W = wk_all.merge(wk_book, on=['split', 'wk'], how='left').fillna({'sum': 0, 'count': 0})
W = W.rename(columns={'sum': 'R', 'count': 'n_tr'})

dayw = day.copy(); dayw['wk'] = pd.to_datetime(dayw.index).to_period('W-FRI').astype(str)
wkreg = dayw.groupby('wk').agg(spy_rv20=('spy_rv20', 'mean'), spy_absret=('spy_absret', 'mean'),
                               xs_std=('xs_std', 'mean'), xs_mad=('xs_mad', 'mean'),
                               n_cand=('n_cand', 'mean')).reset_index()
W = W.merge(wkreg, on='wk', how='left').dropna()
P(f'\nweeks in regression: {len(W)}  ({W.groupby("split").size().to_dict()})')


def ols(y, X, names):
    X = np.column_stack([np.ones(len(y))] + list(X))
    names = ['const'] + list(names)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    n, k = X.shape
    s2 = r @ r / (n - k)
    XtXi = np.linalg.pinv(X.T @ X)
    # HC1 robust
    V = XtXi @ (X.T @ np.diag(r ** 2) @ X) @ XtXi * n / (n - k)
    se = np.sqrt(np.diag(V))
    return pd.DataFrame({'coef': b, 'se': se, 't': b / se}, index=names), 1 - (r @ r) / ((y - y.mean()) ** 2).sum()


y = W.R.values
dVAL = (W.split == 'VAL').values.astype(float)
dTEST = (W.split == 'TEST').values.astype(float)
z = lambda c: (W[c].values - W[c].values.mean()) / W[c].values.std()

P('\n## Q2b — weekly book R on split dummies ONLY (TRAIN is the base)')
t1, r2 = ols(y, [dVAL, dTEST], ['VAL', 'TEST']); P(t1.round(3).to_string()); P(f'R2 {r2:.3f}')

P('\n## Q2c — weekly book R on regime variables ONLY')
t2, r2 = ols(y, [z('spy_rv20'), z('xs_std'), z('n_cand')], ['spy_rv20', 'xs_disp', 'n_cand']); P(t2.round(3).to_string()); P(f'R2 {r2:.3f}')

P('\n## Q2d — weekly book R on regime variables AND split dummies (do the split effects survive?)')
t3, r2 = ols(y, [z('spy_rv20'), z('xs_std'), z('n_cand'), dVAL, dTEST],
             ['spy_rv20', 'xs_disp', 'n_cand', 'VAL', 'TEST']); P(t3.round(3).to_string()); P(f'R2 {r2:.3f}')

P('\n## Q2e — same, per-TRADE mean R as the dependent (weeks weighted by trades)')
m = W.n_tr > 0
yt = (W.R / W.n_tr.replace(0, np.nan))[m].values
t4, r2 = ols(yt, [z('spy_rv20')[m.values], z('xs_std')[m.values], z('n_cand')[m.values], dVAL[m.values], dTEST[m.values]],
             ['spy_rv20', 'xs_disp', 'n_cand', 'VAL', 'TEST']); P(t4.round(3).to_string()); P(f'R2 {r2:.3f}')

# ---------- the population control: does the WHOLE candidate pool move the same way? ----------
P('\n## Q2f — the control: mean R of the FULL F6 candidate population (no selection) by split')
for s in ('TRAIN', 'VAL', 'TEST'):
    x = pool[pool.split == s]
    P(f'  {s:6} n {len(x):6d}  pop meanR {x.net_e1c.mean():+.4f}  t {x.net_e1c.mean()/(x.net_e1c.std()/np.sqrt(len(x))):+.2f}'
      f'   book meanR {book[book.split==s].net.mean():+.4f}')

P('\n## Q2g — book R and pool R per month, side by side')
mo = pool.groupby('mo', observed=True).net_e1c.agg(['mean', 'size']).rename(columns={'mean': 'pool_meanR', 'size': 'pool_n'})
mb = book.groupby('mo').net.agg(['mean', 'sum', 'size']).rename(columns={'mean': 'book_meanR', 'sum': 'book_R', 'size': 'book_n'})
P(mo.join(mb).round(3).to_string())

# ---------- regime vars vs the cross-sectional selection effect ----------
P('\n## Q2h — weekly regression of the book-minus-pool EXCESS on regime (is the selection edge regime-driven?)')
pw = pool.groupby('wk', observed=True).net_e1c.mean().rename('pool_meanR').reset_index()
W2 = W.merge(pw, on='wk', how='left').dropna()
W2 = W2[W2.n_tr > 0]
exc = (W2.R / W2.n_tr - W2.pool_meanR).values
zz = lambda c: (W2[c].values - W2[c].values.mean()) / W2[c].values.std()
t5, r2 = ols(exc, [zz('spy_rv20'), zz('xs_std'), zz('n_cand'), (W2.split == 'VAL').values.astype(float),
                   (W2.split == 'TEST').values.astype(float)],
             ['spy_rv20', 'xs_disp', 'n_cand', 'VAL', 'TEST'])
P(t5.round(3).to_string()); P(f'R2 {r2:.3f}')
P(f'  mean excess by split: ' + ' '.join(f'{s} {exc[(W2.split==s).values].mean():+.3f}' for s in ('TRAIN', 'VAL', 'TEST')))

W.to_csv(f'{A}/weekly_regime.csv', index=False)
day.to_csv(f'{A}/daily_regime.csv')
open(f'{A}/q2_regime.md', 'w').write('\n'.join(OUT))
print('DONE', flush=True)
