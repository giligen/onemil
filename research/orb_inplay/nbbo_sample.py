#!/usr/bin/env python3
"""orb_inplay step 3b — measured NBBO (Alpaca SIP) on a stratified sample of legs.

PREREG 2: >=600 legs, strata = price band x leg clock, from TRAIN+VAL. Fit median
half-spread as % of price per cell; every unmeasured leg is imputed from it.
Method identical to frames16/nbbo.py & frames17/nbbo17.py (mean(ask-bid) over the minute).
"""
import os,sys,time,json
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
ROOT='/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0,ROOT)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest
from alpaca.data.enums import DataFeed
from config import Config
ET=ZoneInfo('America/New_York'); D=f'{ROOT}/research/orb_inplay'; OUT=f'{D}/nbbo.csv'
PB=[0,10,20,50,1e9]; PBL=['a','b','c','d']
def band(p): return pd.cut([p],PB,labels=PBL)[0]
def clk(m): return 'entry' if m==575 else ('mid' if m<900 else 'late')

t=pd.read_csv(f'{D}/trades_raw.csv')
t=t[t.status=='ok'].copy()
legs=pd.concat([t.assign(px=t.entry,m=575),t.assign(px=t['exit'],m=t.exit_m)],ignore_index=True)
legs['pb']=[band(p) for p in legs.px]; legs['cb']=[clk(m) for m in legs.m]
rng=np.random.default_rng(11)
samp=legs.groupby(['pb','cb'],observed=True).apply(
    lambda g:g.sample(min(len(g),70),random_state=11)).reset_index(drop=True)
print(f'legs total {len(legs):,} | sampled {len(samp):,} over {samp.groupby(["pb","cb"],observed=True).ngroups} cells',flush=True)
done=set()
if os.path.exists(OUT):
    d0=pd.read_csv(OUT,dtype={'day':str,'symbol':str}); done=set(zip(d0.day,d0.symbol,d0.m))
cl=StockHistoricalDataClient(Config().alpaca_api_key,Config().alpaca_api_secret)
rows=[];t0=time.time()
todo=[r for r in samp.itertuples() if (r.day,r.symbol,int(r.m)) not in done]
for i,r in enumerate(todo):
    s=datetime.strptime(r.day,'%Y-%m-%d').replace(hour=int(r.m)//60,minute=int(r.m)%60,tzinfo=ET)
    try:
        q=cl.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=r.symbol,start=s,
            end=s+timedelta(minutes=1),feed=DataFeed.SIP,limit=6000)).data.get(r.symbol,[])
        sp=np.array([float(x.ask_price)-float(x.bid_price) for x in q
                     if x.ask_price and x.bid_price and x.ask_price>x.bid_price])
        rows.append(dict(day=r.day,symbol=r.symbol,m=int(r.m),px=r.px,pb=str(r.pb),cb=r.cb,
                         sp_mean=float(sp.mean()) if len(sp) else np.nan,n_q=len(sp)))
    except Exception as e:
        rows.append(dict(day=r.day,symbol=r.symbol,m=int(r.m),px=r.px,pb=str(r.pb),cb=r.cb,
                         sp_mean=np.nan,n_q=0))
    if len(rows)>=150:
        pd.DataFrame(rows).to_csv(OUT,mode='a',header=not os.path.exists(OUT),index=False);rows=[]
        print(f'  {i+1}/{len(todo)} {time.time()-t0:.0f}s',flush=True)
if rows: pd.DataFrame(rows).to_csv(OUT,mode='a',header=not os.path.exists(OUT),index=False)
q=pd.read_csv(OUT)
q=q[(q.n_q>0)&q.sp_mean.notna()]
q['hs_pct']=q.sp_mean/2.0/q.px*100
tab={};rowsout=[]
glob=float(q.hs_pct.median())
for (pb,cb),g in q.groupby(['pb','cb']):
    if len(g)>=20:
        tab[f'{pb}|{cb}']=float(g.hs_pct.median())
        rowsout.append((pb,cb,len(g),float(g.hs_pct.median())))
tab['glob|glob']=glob
json.dump(dict(table=tab,n_measured=int(len(q)),n_legs=int(len(legs)),
               measured_share=len(q)/len(legs),glob=glob),open(f'{D}/hs_table.json','w'),indent=1)
print(f'measured {len(q)} legs, global median half-spread {glob:.4f}% of price')
for r in sorted(rowsout): print('  band %s %-5s n=%3d  half-spread %.4f%%'%r)
