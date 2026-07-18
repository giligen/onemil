"""Complete-river rerun: cache bars + gap-fetch CSV bars."""
import sqlite3, sys
import pandas as pd, numpy as np
sys.stdout.reconfigure(line_buffering=True)
T=pd.read_csv('/tmp/ignition_universe.csv')
gap=pd.read_csv('/tmp/ignition_gap_bars.csv')
gts=pd.to_datetime(gap['t'],utc=True).dt.tz_convert('America/New_York')
gap['m']=gts.dt.hour*60+gts.dt.minute
gap=gap.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
conn=sqlite3.connect('file:data/cache.db?mode=ro',uri=True,timeout=30)
ARM,LOCK=1.75,0.5
def sim(g):
    g=g.sort_values('m').reset_index(drop=True)
    g=g[(g['m']>=570)&(g['m']<960)]
    if len(g)<20: return None
    o=g.iloc[0]['open']
    if not o>0: return None
    lvl=o*1.10
    if not g[(g['high']>=lvl)&(g['m']<575)].empty: return None
    trig=g[(g['high']>=lvl)&(g['m']>=575)&(g['m']<=780)]
    if trig.empty: return None
    ti=trig.index[0]; tb=g.loc[ti]
    entry=max(lvl,tb['open'])*1.003
    if entry>lvl*1.03: return None
    pre=g[(g['m']>=tb['m']-30)&(g['m']<tb['m'])]
    if len(pre)<5: return None
    stop=min(pre['low'].min(),entry*0.99)
    R=entry-stop
    if not R>0: return None
    post=g[g.index>ti]
    cur=stop; armed=False; ex=None; reason=None
    for _,r_ in post.iterrows():
        if r_['m']>=945: ex=r_['open']; reason='eod'; break
        if r_['low']<=cur: ex=cur*0.999; reason='lock' if armed else 'stop'; break
        if not armed and r_['high']>=entry+ARM*R: armed=True; cur=entry+LOCK*R
    if ex is None:
        ex=post.iloc[-1]['close'] if len(post) else entry; reason='eod'
    rr=(ex-entry)/R; rp=R/entry*100
    pos=min(3000.0/(rp/100.0),25000.0)
    return dict(trig_min=int(tb['m']),R_pct=rp,final_r=rr,pnl=rr*pos*rp/100.0,
                reason=reason,vol_trig=tb['volume'],pre_vol=pre['volume'].mean())
trades=[]
for di,day in enumerate(sorted(T['bar_date'].unique())):
    sub=T[T['bar_date']==day]
    cs=sub[sub['cached']]['symbol'].tolist()
    if cs:
        q=("select symbol,timestamp,open,high,low,close,volume from intraday_bars_1min "
           f"where bar_date=? and symbol in ({','.join('?'*len(cs))})")
        b=pd.read_sql(q,conn,params=[day]+cs)
        if not b.empty:
            ts=pd.to_datetime(b['timestamp'],utc=True).dt.tz_convert('America/New_York')
            b['m']=ts.dt.hour*60+ts.dt.minute
            for sym,g in b.groupby('symbol'):
                r=sim(g)
                if r: trades.append({'day':day,'symbol':sym,**r})
    gs=gap[gap['day']==day]
    for sym,g in gs.groupby('symbol'):
        r=sim(g)
        if r: trades.append({'day':day,'symbol':sym,**r})
    if (di+1)%80==0: print(f"  {di+1} days, {len(trades)} trades",flush=True)
t=pd.DataFrame(trades)
t.to_csv('/tmp/ignition_trades_full.csv',index=False)
t['era3']=np.where(t['day']<'2025-07-01','25H1',np.where(t['day']<'2026-01-01','25H2','2026'))
print(f"\nFULL RIVER TRADES: {len(t)}  mean ${t['pnl'].mean():+,.0f}  WR {(t['pnl']>0).mean()*100:.0f}%")
print(t.groupby('era3')['pnl'].agg(['size','mean']).round(0).to_string())
