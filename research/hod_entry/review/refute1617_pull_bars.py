import sqlite3, pandas as pd, numpy as np
c=sqlite3.connect('file:research/bf_zero/bars_sip.db?mode=ro',uri=True)
d=pd.read_csv('research/hod_entry/cell_1617_nights.csv')
d=d[d.cell.astype(str).isin(['1617','1618_failed'])].copy()
base=pd.read_csv('research/hod_entry/causal_arming_causal.csv',low_memory=False)
base=base[base.status=='fill'][['day','symbol','level']]
d=d.merge(base,left_on=['date','symbol'],right_on=['day','symbol'],how='left')
days=sorted(set(pd.read_parquet('research/overnight_high/panel_2024_2026.parquet',columns=['bar_date']).bar_date.astype(str)))
nxt={a:b for a,b in zip(days[:-1],days[1:])}
out=[]
for i,r in enumerate(d.itertuples()):
    b=pd.read_sql("select t,o,c from bars where symbol=? and day=? order by t",c,params=[r.symbol,r.date])
    nb=pd.read_sql("select t,o,c from bars where symbol=? and day=? order by t",c,params=[r.symbol,nxt.get(r.date,'x')])
    rec=dict(idx=i)
    if len(b):
        et=pd.to_datetime(b.t).dt.tz_convert('America/New_York').dt.strftime('%H:%M')
        b['et']=et.values
        pre=b[b.et<='15:49']; rth=b[b.et<='15:59']
        rec['p1549']=pre.c.iloc[-1] if len(pre) else np.nan
        rec['p1559']=rth.c.iloc[-1] if len(rth) else np.nan
        rec['last1559_et']=rth.et.iloc[-1] if len(rth) else None
    if len(nb):
        nb['et']=pd.to_datetime(nb.t).dt.tz_convert('America/New_York').dt.strftime('%H:%M').values
        o=nb[nb.et>='09:30']
        rec['o0930']=o.o.iloc[0] if len(o) else np.nan
        rec['o_et']=o.et.iloc[0] if len(o) else None
    out.append(rec)
    if i%2000==0: print(i,flush=True)
o=pd.DataFrame(out)
d=pd.concat([d.reset_index(drop=True),o.drop(columns='idx')],axis=1)
d.to_csv('/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/causal_rows.csv',index=False)
print('done',len(d))
