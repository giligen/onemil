#!/usr/bin/env python3
"""Refine the measured cost curve: the sample was drawn from ALL lvl1003_f5 signals, but the shipped book
only trades the subset that clears level >= $5, adv20 >= 100K and pdr >= 8.  Build the curve on THAT subset
where a stratum has n >= 10 quotes, falling back to the whole sample otherwise, and report both."""
import os, sys
import numpy as np, pandas as pd
ROOT='/home/ec2-user/onemil'; os.chdir(ROOT); OUT='research/mature_method/red_to_green'
RD=lambda p,**k: pd.read_csv(p,dtype={'symbol':str,'day':str},keep_default_na=False,na_values=[''],**k)
log=lambda *a:(print(*a),sys.stdout.flush())

q=RD(f'{OUT}/nbbo_ok.csv')
d=RD(f'{OUT}/cands.csv',usecols=['day','symbol','entry_m','variants','pdr','level','adv20','r_pct','sig_m','over_cap_bps'])
d=d[d.variants.str.contains('lvl1003_f5',regex=False)].drop_duplicates(['day','symbol','entry_m'])
j=q.merge(d,on=['day','symbol','entry_m'],how='left',suffixes=('','_c'))
j['b0']=(j.pdr>=8)&(j.level>=5)&(j.adv20.fillna(0)>=1e5)&(j.r_pct_c>=1)&(j.sig_m<=840)
log('quotes %d  of which B0-eligible %d'%(len(j),int(j.b0.sum())))
for lab,x in (('ALL sampled signals',j),('B0-eligible only',j[j.b0])):
    log('%-22s median %.1f  mean %.1f  p90 %.1f  band-median %.1f  >300bps %.1f%%'
        %(lab,x.full_bps.median(),x.full_bps.mean(),x.full_bps.quantile(.9),x.band_bps.median(),
          100*(x.full_bps>300).mean()))
a=j.groupby(['pb','hb']).full_bps.agg(['median','count']).rename(columns={'median':'all_med','count':'all_n'})
b=j[j.b0].groupby(['pb','hb']).full_bps.agg(['median','count']).rename(columns={'median':'b0_med','count':'b0_n'})
cur=a.join(b,how='outer').reset_index()
cur['bps_median']=np.where(cur.b0_n.fillna(0)>=10,cur.b0_med,cur.all_med)
cur['source']=np.where(cur.b0_n.fillna(0)>=10,'B0','all')
cur['bps_mean']=cur.bps_median
cur['n']=np.where(cur.b0_n.fillna(0)>=10,cur.b0_n,cur.all_n)
cur.to_csv(f'{OUT}/cost_curve_measured.csv',index=False)
log(cur.to_string(index=False))
