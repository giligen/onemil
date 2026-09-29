import pandas as pd, numpy as np, pickle, sqlite3, os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
ET=ZoneInfo("America/New_York"); UTC=ZoneInfo("UTC")
F=pd.read_csv('features_1489.csv',usecols=['day','symbol','retest_ts','arm_close_j','arm_range_to_j_pct','arm_arm_m','brk_high_pct_of_level','dip_speed_min_high_to_tr','dip_bar_vol_rel_break_bar','Y','split'])
p=pd.read_csv('rebuild_1481_fills.csv'); p=p[p.status=='fill']
d=F.merge(p[['day','symbol','retest_ts','fill_min','retest_minute','level']],on=['day','symbol','retest_ts'])
print('rows',len(d),'arm_m == floor(fill_min):',(d.arm_arm_m==np.floor(d.fill_min)).mean())
print('arm_m == retest_minute:',(d.arm_arm_m==d.retest_minute).mean())
s=d[d.arm_arm_m==d.retest_minute].sample(400,random_state=2)
con=sqlite3.connect('file:bars_fills_1478.db?mode=ro',uri=True)
out=[]
for r in s.itertuples():
    M=int(r.retest_minute)
    f=f'sip_cache_1481/{r.symbol}_{r.day}_{M}.pkl'
    if not os.path.exists(f): continue
    tr,q=pickle.load(open(f,'rb'))
    mid=datetime(*map(int,r.day.split('-')),tzinfo=ET)
    st=int((mid+timedelta(minutes=M)).timestamp()*1e9); e=st+60_000_000_000
    inm=tr[(tr.ts>=st)&(tr.ts<e)].sort_values('ts')
    if inm.empty: continue
    last=inm.iloc[-1]
    out.append(dict(close_eq_last=abs(last.price-r.arm_close_j)<1e-6, last_after_tr=last.ts>r.retest_ts,
                    win_start=(tr.ts.min()-st)/1e9, win_end=(tr.ts.max()-st)/1e9, tr_sec=(r.retest_ts-st)/1e9))
o=pd.DataFrame(out); print(len(o))
print('arm_close_j == last tape print of minute M:',o.close_eq_last.mean())
print('  ... and that print is AFTER t_r:', (o.close_eq_last&o.last_after_tr).mean())
print('window start/end sec rel minute start (median):',o.win_start.median(),o.win_end.median(),'t_r sec median',o.tr_sec.median())
# recompute range excluding bar M
def rng(sym,day,M,upto_incl):
    b=pd.read_sql("select t,h,l from bars where symbol=? and day=? order by t",con,params=(sym,day))
    t=pd.to_datetime(b.t,utc=True).dt.tz_convert(ET); mid=datetime(*map(int,day.split('-')),tzinfo=ET)
    b['m']=((t-mid).dt.total_seconds()/60).astype(int)
    b=b[(b.m>=570)&(b.m<=959)]
    bb=b[b.m<=M] if upto_incl else b[b.m<M]
    return (bb.h.max()-bb.l.min())/bb.l.min()*100 if len(bb) else np.nan
chg=[]
for r in s.head(150).itertuples():
    a=rng(r.symbol,r.day,int(r.retest_minute),True); b=rng(r.symbol,r.day,int(r.retest_minute),False)
    chg.append(dict(inc=a,exc=b,feat=r.arm_range_to_j_pct))
c=pd.DataFrame(chg)
print('recomputed incl bar M == feature:',(abs(c.inc-c.feat)<1e-6).mean(),' differs when bar M excluded:',(abs(c.inc-c.exc)>1e-9).mean())
