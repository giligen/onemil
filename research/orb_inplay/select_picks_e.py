#!/usr/bin/env python3
"""orb_inplay Cell E — liquid sub-universe re-rank (see PREREG.md ## Cell E).

SAME picks pipeline as select_picks.py, but restricted BEFORE ranking to
price >= $20 (prev_close) AND ADV20 >= 5,000,000 shares. Top-20 by RVOL
re-ranked WITHIN that sub-universe each day (RVOL >= 1.0 floor unchanged).
"""
import numpy as np, pandas as pd
D = '/home/ec2-user/onemil/research/orb_inplay'
u = pd.read_parquet(f'{D}/universe.parquet')
u_sub = u[(u.prev_close >= 20.0) & (u.adv20 >= 5_000_000)].copy()
o = pd.read_parquet(f'{D}/open5.parquet')
o = o[(o.m >= 570) & (o.m <= 574)]
g = o.sort_values(['symbol', 'bar_date', 'm']).groupby(['symbol', 'bar_date'], sort=False)
a = g.agg(v5=('volume', 'sum'), nbar=('m', 'size'), o5=('open', 'first'), c5=('close', 'last')).reset_index()
a = a.sort_values(['symbol', 'bar_date'])
gg = a.groupby('symbol', sort=False).v5
a['base'] = gg.transform(lambda s: s.shift(1).rolling(14, min_periods=10).mean())
a['nbase'] = gg.transform(lambda s: s.shift(1).rolling(14, min_periods=1).count())
a['rvol'] = a.v5 / a.base
m = u_sub.merge(a, on=['symbol', 'bar_date'], how='left')
m['has5'] = m.nbar.notna()
rank = m[(m.rvol >= 1.0) & m.base.notna() & (m.base > 0)].copy()
rank['rk'] = rank.groupby('bar_date').rvol.rank(ascending=False, method='first')
pk = rank[rank.rk <= 20].copy()
pk['side'] = np.where(pk.c5 > pk.o5, 'long', np.where(pk.c5 < pk.o5, 'short', 'doji'))
pk.to_parquet(f'{D}/picks_e.parquet', index=False)
cov = m.groupby('bar_date').has5.mean()
print(f'sub-universe days {u_sub.bar_date.nunique()} | sub-universe/day median {int(u_sub.groupby("bar_date").size().median())}')
print(f'open-tape coverage of sub-universe: mean {cov.mean():.1%} min {cov.min():.1%}')
print(f'rankable/day median {int(m[m.rvol.notna()].groupby("bar_date").size().median())}')
print(f'picks {len(pk)} over {pk.bar_date.nunique()} days = {len(pk)/pk.bar_date.nunique():.1f}/day')
print(pk.side.value_counts().to_dict())
print('rvol of picks: median %.2f p10 %.2f max %.1f' % (pk.rvol.median(), pk.rvol.quantile(.1), pk.rvol.max()))
print('prev_close of picks: median %.2f p10 %.2f' % (pk.prev_close.median(), pk.prev_close.quantile(.1)))
