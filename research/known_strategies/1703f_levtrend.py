"""Cell 1,703f: leveraged index trend (PREREG_1703_sweep.md Amendment 2). Open-to-open daily returns;
state at open d = signal at close d-1 (QQQ/SPY close > SMA200 through d-1); 5 bp per switch."""
import numpy as np, pandas as pd, pyarrow.parquet as pq, yfinance as yf
P='research/momentum_weekly/panel_2016_2026.parquet'
SYMS=['TQQQ','UPRO','QQQ','SPY']
pn=pq.read_table(P,filters=[('symbol','in',SYMS)],columns=['symbol','bar_date','open','close']).to_pandas()
pn['bar_date']=pd.to_datetime(pn.bar_date)
def panel(s): return pn[pn.symbol==s].set_index('bar_date').sort_index()[['open','close']]
bars={}
for s in SYMS:
    p=panel(s)
    y=yf.download(s,start='2009-06-01',end='2026-10-01',auto_adjust=True,progress=False)
    y.columns=[c[0] if isinstance(c,tuple) else c for c in y.columns]
    y=y[['Open','Close']].rename(columns={'Open':'open','Close':'close'}); y.index=pd.to_datetime(y.index).tz_localize(None)
    ov=p.index.intersection(y.index)
    cc=np.corrcoef(p.loc[ov,'close'].pct_change().dropna(),y.loc[ov,'close'].pct_change().dropna())[0,1]
    oc=np.corrcoef(p.loc[ov,'open'].pct_change().dropna(),y.loc[ov,'open'].pct_change().dropna())[0,1]
    print(s,'overlap',len(ov),'close-ret corr %.5f open-ret corr %.5f'%(cc,oc),flush=True)
    m=ov[(ov<'2020-03-01')|(ov>'2020-03-31')]
    cx=np.corrcoef(p.loc[m,'close'].pct_change().dropna(),y.loc[m,'close'].pct_change().dropna())[0,1]
    print(s,'close corr excl 2020-03: %.5f'%cx,flush=True)
    assert cx>=0.999 and (oc>=0.999 or s in ('SPY','QQQ')),s  # signal ETFs use closes only; SPY open corr 0.9985 unused
    pre=y[y.index<p.index[0]]
    # splice by returns: scale yfinance pre-2016 levels so they join the panel level at the first panel date
    k=p.iloc[0]['close']/y.loc[p.index[0],'close']; bars[s]=pd.concat([pre*k,p])
def sma_signal(s):
    c=bars[s].close; return (c>c.rolling(200).mean()).shift(1)  # known at close d-1, applies at open d
def oo(s): return bars[s].open.shift(-1)/bars[s].open-1   # open d -> open d+1
def strat(lev,idx,trend):
    r=oo(lev); st=sma_signal(idx).astype(float) if trend else pd.Series(1.0,index=r.index)
    d=pd.concat([r,st],axis=1,keys=['r','s']).dropna()
    sw=d.s.diff().abs().fillna(d.s.abs()); sw.iloc[0]=0
    return (d.s*d.r-sw*0.0005), sw
rows=[];ser={}
cfg=[('TQQQ-trend','TQQQ','QQQ',True),('UPRO-trend','UPRO','SPY',True),('TQQQ-hold','TQQQ',None,False),('UPRO-hold','UPRO',None,False)]
def stats(r,sw):
    n=len(r); eq=(1+r).cumprod(); yrs=n/252; dd=(eq/eq.cummax()-1).min()
    return dict(cagr=eq.iloc[-1]**(1/yrs)-1,maxdd=dd,end50k=50000*eq.iloc[-1],sharpe=r.mean()/r.std()*252**.5,switches_per_yr=sw.sum()/yrs)
W={'2017-2026':('2017-01-01','2026-09-30'),'2011-2026':('2011-01-01','2026-09-30')}
for nm,l,i,t in cfg:
    r,sw=strat(l,i,t); ser[nm]=r
    for w,(a,b) in W.items():
        rr=r[a:b]; ss=sw[a:b]; rows.append(dict(cell=nm,window=w,**stats(rr,ss)))
        if w=='2017-2026':
            h1=(1+rr[:'2021']).prod()-1; h2=(1+rr['2022':]).prod()-1; rows[-1].update(h1_2017_21=h1,h2_2022_26=h2)
# weekly series (Monday open to next Monday open) matching RECON weeks
g=pd.read_csv('research/momentum_weekly/RECON_1700tu_weeks.csv',parse_dates=['date']).set_index('date')
gref=g['net_A']
gc=(1+gref).prod()**(52/len(gref))-1; print('GREF net_A CAGR %.4f'%gc,flush=True)
def weekly(r):
    wk=pd.Series(r.index.map(lambda d:d-pd.Timedelta(days=d.weekday())),index=r.index)
    return (1+r).groupby(wk.values).prod()-1
def pstats(w):
    eq=(1+w).cumprod(); yrs=len(w)/52; cg=eq.iloc[-1]**(1/yrs)-1; dd=(eq/eq.cummax()-1).min(); return cg,dd,cg/abs(dd)
# GREF three deepest drawdown episodes
geq=(1+gref).cumprod(); peak=geq.cummax(); ddv=geq/peak-1; ep=(ddv==0).cumsum()
eps=sorted([(ddv[ep==k].min(),ddv[ep==k].index[0],ddv[ep==k].index[-1]) for k in ep.unique() if (ep==k).sum()>1])[:3]
print('GREF episodes',eps,flush=True)
gcg,gdd,gra=pstats(gref); print('GREF',gcg,gdd,gra)
for nm in ser:
    w=weekly(ser[nm]).reindex(gref.index).dropna(); gg=gref.reindex(w.index)
    corr=w.corr(gg); epc=[w[a:b].corr(gg[a:b]) for _,a,b in eps]
    c5,d5,r5=pstats(0.5*w+0.5*gg); cs,ds,rs=pstats(gg+0.5*w)
    s=[x for x in rows if x['cell']==nm and x['window']=='2017-2026'][0]
    s.update(corr_gref=corr,corr_ep1=epc[0],corr_ep2=epc[1],corr_ep3=epc[2],p5050_cagr=c5,p5050_dd=d5,p5050_ratio=r5,stack_cagr=cs,stack_dd=ds,stack_ratio=rs)
    print(nm,'corr %.3f eps %s 50/50 %.3f %.3f %.2f stack %.3f %.3f %.2f'%(corr,np.round(epc,2),c5,d5,r5,cs,ds,rs),flush=True)
pd.DataFrame(rows).to_csv('research/known_strategies/1703f_cells.csv',index=False)
# by-year TQQQ-trend vs hold, switches, shift test
r,sw=strat('TQQQ','QQQ',True); rh=ser['TQQQ-hold']
yr=pd.DataFrame({'trend':r.groupby(r.index.year).apply(lambda x:(1+x).prod()-1),'hold':rh.groupby(rh.index.year).apply(lambda x:(1+x).prod()-1),'switches':sw.groupby(sw.index.year).sum()})
yr['trend_dd']=r.groupby(r.index.year).apply(lambda x:((1+x).cumprod()/(1+x).cumprod().cummax()-1).min())
print(yr.round(3).to_string(),flush=True); yr.to_csv('research/known_strategies/1703f_byyear.csv')
d0=pd.Timestamp('2022-03-15'); c=bars['QQQ'].close; i=c.index.get_loc(d0)
print('shift test',d0.date(),'close',c.iloc[i],'sma200 thru day',c.iloc[i-199:i+1].mean(),'signal at open next day',sma_signal('QQQ').iloc[i+1],'(computed from close_{d}>sma_{d})',c.iloc[i]>c.iloc[i-199:i+1].mean(),flush=True)
print(pd.DataFrame(rows).round(4).to_string(),flush=True)
