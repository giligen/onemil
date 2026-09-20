#!/usr/bin/env python3
"""orb_inplay step 2 — RVOL ranking -> top-20 stocks in play per day (PREREG 1.2-1.4)."""
import os,sys
import numpy as np, pandas as pd
D='/home/ec2-user/onemil/research/orb_inplay'
u=pd.read_parquet(f'{D}/universe.parquet')
o=pd.read_parquet(f'{D}/open5.parquet')
o=o[(o.m>=570)&(o.m<=574)]
g=o.sort_values(['symbol','bar_date','m']).groupby(['symbol','bar_date'],sort=False)
a=g.agg(v5=('volume','sum'),nbar=('m','size'),o5=('open','first'),c5=('close','last')).reset_index()
a=a.sort_values(['symbol','bar_date'])
# prior-14-session mean of the same 5 minutes (strictly causal, >=10 present)
gg=a.groupby('symbol',sort=False).v5
a['base']=gg.transform(lambda s:s.shift(1).rolling(14,min_periods=10).mean())
a['nbase']=gg.transform(lambda s:s.shift(1).rolling(14,min_periods=1).count())
a['rvol']=a.v5/a.base
m=u.merge(a,on=['symbol','bar_date'],how='left')
m['has5']=m.nbar.notna()
rank=m[(m.rvol>=1.0)&m.base.notna()&(m.base>0)].copy()
rank['rk']=rank.groupby('bar_date').rvol.rank(ascending=False,method='first')
pk=rank[rank.rk<=20].copy()
pk['side']=np.where(pk.c5>pk.o5,'long',np.where(pk.c5<pk.o5,'short','doji'))
pk.to_parquet(f'{D}/picks.parquet',index=False)
cov=m.groupby('bar_date').has5.mean()
print(f'days {m.bar_date.nunique()} | univ/day median {int(m.groupby("bar_date").size().median())} | '
      f'open-tape coverage of universe: mean {cov.mean():.1%} min {cov.min():.1%}')
print(f'rankable/day median {int(m[m.rvol.notna()].groupby("bar_date").size().median())}')
print(f'picks {len(pk)} over {pk.bar_date.nunique()} days = {len(pk)/pk.bar_date.nunique():.1f}/day')
print(pk.side.value_counts().to_dict())
print('rvol of picks: median %.2f p10 %.2f max %.1f'%(pk.rvol.median(),pk.rvol.quantile(.1),pk.rvol.max()))
