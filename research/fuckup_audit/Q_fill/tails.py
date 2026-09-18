#!/usr/bin/env python3
"""Stage Q — tail dependence (PLAN §1 item 5) for each fill model."""
import os, sys
import numpy as np, pandas as pd
ROOT='/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0,ROOT)
sys.path.insert(0,f'{ROOT}/research/fuckup_audit/Q_fill')
from rescore_q import ARMS, range_lookup, Q          # noqa
from trading.orb_csv import read_orb_csv             # noqa
rows=[]
for arm in ARMS:
    for n in (8,3):
        p=f'{Q}/book_{arm}_n{n}.csv'
        if not os.path.exists(p): continue
        b=read_orb_csv(p); b['date']=pd.to_datetime(b['date']).dt.strftime('%Y-%m-%d')
        rl=range_lookup()
        rng=np.array([rl.get((s,d),np.nan) for s,d in zip(b.symbol,b.date)])
        sh=b['_rp_position']/b['entry_price']
        R=np.where(rng>0, b['_sized_pnl']/(sh*rng), 0.0)
        R=pd.Series(R)
        srt=R.sort_values()
        k1=max(1,int(round(0.01*len(R)))); k5=max(1,int(round(0.05*len(R))))
        rows.append(dict(arm=arm, slots=n, n=len(R), mean_R=round(R.mean(),3),
                         ex_top1=round(srt.iloc[:-k1].mean(),3),
                         ex_top5=round(srt.iloc[:-k5].mean(),3),
                         cap3R=round(np.minimum(R,3.0).mean(),3),
                         t=round(R.mean()/(R.std(ddof=1)/np.sqrt(len(R))),2)))
t=pd.DataFrame(rows); t.to_csv(f'{Q}/tails.csv',index=False); print(t.to_string(index=False))
