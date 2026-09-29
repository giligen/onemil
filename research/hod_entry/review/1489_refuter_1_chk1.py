import pandas as pd, numpy as np, pickle, sqlite3, os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
ET=ZoneInfo("America/New_York"); UTC=ZoneInfo("UTC")
p=pd.read_csv('rebuild_1481_fills.csv'); p=p[p.status=='fill'].copy()
p=p[np.floor(p.fill_min)==p.retest_minute].sample(300,random_state=1)
con=sqlite3.connect('file:bars_fills_1478.db?mode=ro',uri=True)
res=[]
for r in p.itertuples():
    M=int(r.retest_minute)
    f=f'sip_cache_1481/{r.symbol}_{r.day}_{M}.pkl'
    if not os.path.exists(f): continue
    tr,q=pickle.load(open(f,'rb'))
    mid=datetime(*map(int,r.day.split('-')),tzinfo=ET)
    s=int((mid+timedelta(minutes=M)).timestamp()*1e9); e=s+60_000_000_000
    tstr=(mid+timedelta(minutes=M)).astimezone(UTC).strftime('%Y-%m-%dT%H:%M')
    b=pd.read_sql("select t,h,l,c,v from bars where symbol=? and day=? and t like ?",con,params=(r.symbol,r.day,tstr+'%'))
    if b.empty: continue
    inm=tr[(tr.ts>=s)&(tr.ts<e)]
    pre=tr[(tr.ts>=s)&(tr.ts<=int(r.retest_ts))]
    post=tr[(tr.ts>int(r.retest_ts))&(tr.ts<e)]
    res.append(dict(sym=r.symbol,day=r.day,tmin=tr.ts.min()<=s, tmax=tr.ts.max()>=e-1e9,
       bar_v=b.v[0], tape_v_min=inm['size'].sum(), bar_h=b.h[0], bar_l=b.l[0],bar_c=b.c[0],
       pre_h=pre.price.max(), pre_l=pre.price.min(), post_h=post.price.max() if len(post) else np.nan,
       post_l=post.price.min() if len(post) else np.nan, n_post=len(post), level=r.level, net=r.net_R_prime))
d=pd.DataFrame(res)
print(len(d)); print('tape window covers whole minute:', (d.tmin&d.tmax).mean())
w=d[d.tmin&d.tmax]
print('bar_v/tape_v median', (w.bar_v/w.tape_v_min).median(), 'IQR', (w.bar_v/w.tape_v_min).quantile([.25,.75]).tolist())
print('rows with post-tr trades in minute:', (d.n_post>0).mean())
print('bar low < min pre-t_r print (bar low set AFTER t_r):', (d.bar_l < d.pre_l-1e-9).mean())
print('bar high > max pre-t_r print (bar high set AFTER t_r):', (d.bar_h > d.pre_h+1e-9).mean())
print(d.head(8).to_string())
