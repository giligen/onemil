import sqlite3, pandas as pd, numpy as np
from scipy.stats import spearmanr
db=sqlite3.connect('file:/home/ec2-user/onemil/data/trades.db?mode=ro',uri=True)
t=pd.read_sql("""select trade_date as date,symbol,order_id,shares,filled_qty,fill_price,stop_loss_price,real_stop_loss_price,
entry_quote_bid bid,entry_quote_ask ask,pnl,exit_reason,account from trades where strategy='orb' and pnl is not null
and ((trade_date between '2026-05-19' and '2026-09-23') or (trade_date between '2026-09-28' and '2026-10-22'))""",db)
b=pd.read_csv('broker_ts.csv')
print(len(t), len(b), t.order_id.duplicated().sum(), b.order_id.duplicated().sum())
print(b.brk_status.value_counts().to_dict(), b.brk_filled.isna().sum())
t=t.merge(b[['order_id','brk_created','brk_filled','brk_status']],on='order_id',how='left')
print('unmatched',t.brk_created.isna().sum(),'no filled_at',t.brk_filled.isna().sum())
t['book']=np.where(t.date<='2026-09-23','LIVE','PAPER')
q=t.filled_qty.fillna(t.shares)
t['qty']=q
t['R']=t.pnl/(q*(t.fill_price-t.stop_loss_price))
t['drift_bps']=(t.fill_price-t.ask)/t.ask*1e4
t['spread_bps']=(t.ask-t.bid)/((t.ask+t.bid)/2)*1e4
t['L_brk']=(pd.to_datetime(t.brk_filled,utc=True,format='ISO8601')-pd.to_datetime(t.brk_created,utc=True,format='ISO8601')).dt.total_seconds()
t.to_csv('rebuild_1700.csv',index=False)
print(t[['R','drift_bps','spread_bps','L_brk']].describe().T)
print('R inf/odd',t[~np.isfinite(t.R)|(t.R.abs()>10)][['date','symbol','R','fill_price','stop_loss_price','pnl']])
print('stop vs real differ',(t.real_stop_loss_price.notna()&(t.real_stop_loss_price!=t.stop_loss_price)).sum())
rng=np.random.default_rng(1)
def rep(d,label):
    print('=====',label,'n',len(d),'sumR',d.R.sum().round(2),'meanR',d.R.mean().round(3),'sum$',d.pnl.sum().round(0))
    x=d.dropna(subset=['L_brk','R'])
    r=spearmanr(x.L_brk,x.R); print('spearman(L,R) n',len(x),round(r[0],3),'p',round(r[1],3))
    r2=spearmanr(x.L_brk,x.drift_bps,nan_policy='omit'); print('spearman(L,drift)',round(r2[0],3))
    days=x.date.unique(); g={k:v for k,v in x.groupby('date')}
    bs=[]
    for _ in range(2000):
        s=pd.concat([g[k] for k in rng.choice(days,len(days))])
        bs.append(spearmanr(s.L_brk,s.R)[0])
    print('day-boot CI',np.nanpercentile(bs,[2.5,97.5]).round(3),'days',len(days))
    x=x.copy(); x['terc']=pd.qcut(x.L_brk,3,labels=['lo','mid','hi'])
    print(x.groupby('terc',observed=True).agg(n=('R','size'),Lmin=('L_brk','min'),Lmax=('L_brk','max'),meanR=('R','mean'),sumR=('R','sum'),usd=('pnl','sum')).round(2))
    y=d.dropna(subset=['drift_bps','R']); y=y.assign(dg=np.where(y.drift_bps<=30,'<=30','>30'))
    print(y.groupby('dg').agg(n=('R','size'),meanR=('R','mean'),sumR=('R','sum'),usd=('pnl','sum')).round(3))
    z=d.dropna(subset=['spread_bps','R']); z=z.assign(sb=pd.cut(z.spread_bps,[-1e9,50,150,300,1e9],right=False,labels=['<50','50-150','150-300','>=300']))
    print(z.groupby('sb',observed=True).agg(n=('R','size'),meanR=('R','mean'),drift=('drift_bps','mean'),usd=('pnl','sum')).round(2))
rep(t[t.book=='LIVE'],'LIVE')
rep(t,'LIVE+PAPER')
