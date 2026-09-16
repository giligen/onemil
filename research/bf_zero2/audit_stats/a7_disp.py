#!/usr/bin/env python3
"""a7 — a robust cross-sectional dispersion series (the a1 version was contaminated by
non-finite daily returns in the panel). Median absolute close-to-close return across liquid
names each day, plus the 90th percentile, and the split means."""
import os, sys, numpy as np, pandas as pd, pyarrow.parquet as pq
ROOT='/home/ec2-user/onemil'; os.chdir(ROOT)
A='research/bf_zero2/audit_stats'
pf=pq.ParquetFile('research/lit_review_2026/daily_panel.parquet')
acc=[]
for i in range(pf.metadata.num_row_groups):
    t=pf.read_row_group(i,columns=['symbol','bar_date','ret_cc','dvol20','close']).to_pandas()
    t=t[(t.dvol20>=2e6)&(t.close>=5)&np.isfinite(t.ret_cc)&t.ret_cc.between(-0.9,3.0)]
    acc.append(t[['bar_date','ret_cc']])
    del t
d=pd.concat(acc); del acc
g=d.groupby('bar_date').ret_cc.agg(n='size', med_abs=lambda s: s.abs().median(),
                                   p90_abs=lambda s: s.abs().quantile(0.90), sd='std')
g=g.loc[(g.index>='2025-01-01')&(g.index<='2026-09-30')]
g['split']=np.where(g.index<'2026-01-01','TRAIN',np.where(g.index<'2026-06-01','VAL','TEST'))
print(g.groupby('split').agg(days=('n','size'), names=('n','mean'),
      med_abs_bps=('med_abs',lambda s: s.mean()*1e4), p90_abs_bps=('p90_abs',lambda s: s.mean()*1e4),
      sd_bps=('sd',lambda s: s.mean()*1e4)).round(1).to_string(), flush=True)
g.to_csv(f'{A}/xs_dispersion.csv')
print('DONE', flush=True)
