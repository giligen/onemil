"""Latency-honest variant: entry at NEXT bar open after the trigger bar
(models the 60s scan cadence + order placement)."""
import sqlite3, sys
import pandas as pd, numpy as np
sys.stdout.reconfigure(line_buffering=True)
fb=pd.read_csv('/tmp/ignition_final_book.csv')   # the candidate book's trades
conn=sqlite3.connect('file:data/cache.db?mode=ro',uri=True,timeout=30)
gap=pd.read_csv('/tmp/ignition_gap_bars.csv')
gts=pd.to_datetime(gap['t'],utc=True).dt.tz_convert('America/New_York')
gap['m']=gts.dt.hour*60+gts.dt.minute
gap=gap.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
ARM,LOCK=1.75,0.5
out=[]
for i,r in fb.iterrows():
    day=r['day']; sym=r['symbol']
    b=pd.read_sql("select timestamp,open,high,low,close,volume from intraday_bars_1min where symbol=? and bar_date=?",conn,params=(sym,day))
    if b.empty:
        g=gap[(gap['symbol']==sym)&(gap['day']==day)]
        if g.empty: continue
        b=g[['open','high','low','close','volume','m']].copy()
    else:
        ts=pd.to_datetime(b['timestamp'],utc=True).dt.tz_convert('America/New_York')
        b['m']=ts.dt.hour*60+ts.dt.minute
    g=b[(b['m']>=570)&(b['m']<960)].sort_values('m').reset_index(drop=True)
    if len(g)<20: continue
    o=g.iloc[0]['open']; lvl=o*1.10
    trig=g[(g['high']>=lvl)&(g['m']>=575)&(g['m']<=780)]
    if trig.empty: continue
    ti=trig.index[0]
    nxt=g[g.index>ti]
    if nxt.empty: continue
    nb=nxt.iloc[0]
    entry=nb['open']*1.003
    if entry>lvl*1.05: continue          # latency chase guard: skip if next bar opened >5% past level
    pre=g[(g['m']>=g.loc[ti,'m']-30)&(g['m']<g.loc[ti,'m'])]
    if len(pre)<5: continue
    stop=min(pre['low'].min(),entry*0.99)
    R=entry-stop
    if not R>0: continue
    post=g[g.index>nb.name]
    cur=stop; armed=False; ex=None
    for _,r_ in post.iterrows():
        if r_['m']>=945: ex=r_['open']; break
        if r_['low']<=cur: ex=cur*0.999; break
        if not armed and r_['high']>=entry+ARM*R: armed=True; cur=entry+LOCK*R
    if ex is None: ex=post.iloc[-1]['close'] if len(post) else entry
    rr=(ex-entry)/R; rp=R/entry*100
    pos=min(min(3000.0/(rp/100.0),25000.0),0.15*nb['volume']*entry)
    if pos<2000: continue
    part=pos/max(nb['volume']*entry,1)
    pnl=pos*(rr*rp/100.0)-pos*0.0012*min(part/0.15,1.0)
    out.append({'day':day,'era3':r['era3'],'pnl':pnl})
t=pd.DataFrame(out)
print(f"LATENCY-HONEST BOOK (entry next-bar-open, chase guard 5%): n={len(t)} of {len(fb)}")
print(f"TOT ${t['pnl'].sum():+,.0f}  mean ${t['pnl'].mean():+,.0f}")
print(t.groupby('era3')['pnl'].agg(['size','mean','sum']).round(0).to_string())
t['date']=pd.to_datetime(t['day'])
ym=t.groupby(t['date'].dt.strftime('%Y-%m'))['pnl'].sum()
print(f"negMo {(ym<0).sum()}/19  worstMo ${ym.min():+,.0f}  monster months(>=10K): {(ym>=10000).sum()}/19")
