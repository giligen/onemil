import pandas as pd, numpy as np, pickle, sqlite3, os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
ET=ZoneInfo("America/New_York"); UTC=ZoneInfo("UTC")
F=pd.read_csv('features_1489.csv',usecols=['day','symbol','retest_ts','ctx_spy_ret_fill_to_tr','brk_minutes_fill_to_tr','dip_bar_vol_rel_break_bar','Y'])
p=pd.read_csv('rebuild_1481_fills.csv'); p=p[p.status=='fill']
d=F.merge(p[['day','symbol','retest_ts','fill_min','retest_minute','level']],on=['day','symbol','retest_ts'])
same=np.floor(d.fill_min)==d.retest_minute
nn=d.ctx_spy_ret_fill_to_tr.notna()
print('ctx_spy nonzero | same-minute:',(d.ctx_spy_ret_fill_to_tr[same&nn]!=0).mean(),' | later-minute:',(d.ctx_spy_ret_fill_to_tr[~same&nn]!=0).mean())
print('dip_bar_vol_rel==1 | same-minute:',(abs(d.dip_bar_vol_rel_break_bar[same]-1)<1e-9).mean())
mid_s=[]
for r in d.itertuples():
    mid=datetime(*map(int,r.day.split('-')),tzinfo=ET)
    mid_s.append((r.retest_ts-int((mid+timedelta(minutes=int(r.retest_minute))).timestamp()*1e9))/1e9)
d['tr_sec']=mid_s
print('t_r seconds into its minute: min',d.tr_sec.min(),'median',d.tr_sec.median(),'max',d.tr_sec.max(),'share<59.999s',(d.tr_sec<59.999).mean())
# post-t_r prints in minute M setting bar M high/low (tape available after t_r to minute end)
con=sqlite3.connect('file:bars_fills_1478.db?mode=ro',uri=True)
s=d[same].sample(600,random_state=3); out=[]
for r in s.itertuples():
    M=int(r.retest_minute); f=f'sip_cache_1481/{r.symbol}_{r.day}_{M}.pkl'
    if not os.path.exists(f): continue
    tr,q=pickle.load(open(f,'rb'))
    mid=datetime(*map(int,r.day.split('-')),tzinfo=ET); st=int((mid+timedelta(minutes=M)).timestamp()*1e9); e=st+60_000_000_000
    tstr=(mid+timedelta(minutes=M)).astimezone(UTC).strftime('%Y-%m-%dT%H:%M')
    b=pd.read_sql("select h,l,c,v from bars where symbol=? and day=? and t like ?",con,params=(r.symbol,r.day,tstr+'%'))
    if b.empty: continue
    post=tr[(tr.ts>r.retest_ts)&(tr.ts<e)]
    out.append(dict(npost=len(post), vpost=post['size'].sum(), bar_v=b.v[0],
        low_post=len(post)>0 and post.price.min()<=b.l[0]+1e-9, high_post=len(post)>0 and post.price.max()>=b.h[0]-1e-9, Y=r.Y))
o=pd.DataFrame(out); print('sample',len(o))
print('share with post-t_r prints inside bar M:',(o.npost>0).mean(),' median post-t_r share of bar volume:',(o.vpost/o.bar_v).median(), 'mean', (o.vpost/o.bar_v).mean())
print('bar M LOW touched by a post-t_r print:',o.low_post.mean(),' bar M HIGH touched post-t_r:',o.high_post.mean())
print('Y rate | low set post:',o[o.low_post].Y.mean(),' | not:',o[~o.low_post].Y.mean())
