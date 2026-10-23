import sqlite3, json, numpy as np, pandas as pd
OUT={}
con=sqlite3.connect('file:data/trades.db?mode=ro',uri=True)
o=pd.read_sql("select * from trades where strategy='orb' and pnl is not null",con)
o['fd']=o.filled_at.str[:10]
o['acct']=np.where(o.account=='paper','paper','live')
o['risk']=o.shares*(o.entry_price-o.stop_loss_price)
o['R']=o.pnl/o.risk
o['R_fill']=o.pnl/(o.shares*(o.fill_price-o.stop_loss_price))
o['R_tr']=o.pnl/o.total_risk
o['isstop']=o.exit_reason.str.startswith('stop_loss')
o['hold']=(pd.to_datetime(o.exited_at,utc=True,format='ISO8601')-pd.to_datetime(o.filled_at,utc=True,format='ISO8601')).dt.total_seconds()
live=o[(o.acct=='live')&(o.fd>='2026-05-19')&(o.fd<='2026-09-23')]
pap=o[(o.acct=='paper')&(o.fd>='2026-09-28')&(o.fd<='2026-10-22')]
print(len(live),len(pap), o.acct.value_counts().to_dict())
def summ(d):
    g=d.groupby('exit_reason').agg(n=('pnl','size'),sum=('pnl','sum'),R=('R','mean'),Rfill=('R_fill','mean'),Rtr=('R_tr','mean'))
    return g
for nm,d in [('live',live),('paper',pap),('all',pd.concat([live,pap]))]:
    g=summ(d); print(nm,'\n',g.round(3))
    st=d[d.isstop]
    OUT['1702_'+nm]=dict(n=len(d),sum=round(d.pnl.sum(),2),meanR=round(d.R.mean(),4),meanR_fill=round(d.R_fill.mean(),4),meanR_totalrisk=round(d.R_tr.mean(),4),
        stop_n=len(st),stop_share=round(len(st)/len(d),4),stop_le5min=int((st.hold<=300).sum()),stop_le5min_share_of_stops=round(float((st.hold<=300).mean()),4),
        stop_le5min_share_of_fills=round(float((st.hold<=300).sum()/len(d)),4),stop_le1min=int((st.hold<=60).sum()),
        by_exit={k:dict(n=int(v.n),sum=round(v['sum'],2),meanR=round(v.R,4),meanR_fill=round(v.Rfill,4),meanR_totalrisk=round(v.Rtr,4)) for k,v in g.iterrows()},
        hold_median_stop=float(st.hold.median()))
    print(nm,OUT['1702_'+nm]['stop_share'],OUT['1702_'+nm]['stop_le5min'],st.hold.describe().round(0).to_dict())
b=pd.read_csv('analysis_results/orb_bplus_book.csv',usecols=['date','entered','exit_reason','pnl'])
b=b[(b.date>='2026-05-19')&(b.date<='2026-09-23')]
print(b.entered.value_counts(dropna=False).to_dict()); print(b.exit_reason.value_counts().to_dict())
e=b[b.entered==1]
OUT['1702_bt']=dict(rows=len(b),entered=len(e),exits=e.exit_reason.value_counts().to_dict(),stop_share_of_entered=round(float((e.exit_reason=='stop').mean()),4),
   stop_share_of_all_rows=round(float((b.exit_reason=='stop').mean()),4))
print(OUT['1702_bt'])
# 1704
h=pd.read_sql("select * from trades where strategy='hod_break' and pnl is not null",con)
h['fd']=h.filled_at.str[:10]
h['td']=h.trade_date
print(h.td.min(),h.td.max(),h.filled_at.isna().sum(),h.account.value_counts(dropna=False).to_dict())
h=h[(h.td>='2026-09-25')&(h.td<='2026-10-22')].copy()
h['r']=h.pnl/50.0
def clus(d,col):
    day=d.groupby('td')[col]
    m=d[col].mean(); sums=day.sum(); cnt=day.size()
    # ratio estimator cluster SE
    res=sums-m*cnt; se=np.sqrt((res**2).sum()*len(cnt)/(len(cnt)-1))/len(d)
    return m,se,d[col].std(ddof=1)/np.sqrt(len(d)),len(cnt)
m,se,seiid,nd=clus(h,'r')
k=int(np.ceil(0.05*len(h))); srt=h.sort_values('pnl',ascending=False)
ex5=srt.iloc[k:]
h['wk']=pd.to_datetime(h.td).dt.to_period('W-SUN').astype(str)
wk=h.groupby('wk').pnl.agg(['size','sum']).round(2)
print(wk)
OUT['1704_book']=dict(n=len(h),sum=round(h.pnl.sum(),2),mean_r=round(m,4),se_cluster=round(se,4),se_iid=round(seiid,4),ndays=nd,ci95=[round(m-1.96*se,3),round(m+1.96*se,3)],
  top5_k=k,ex_top5_sum=round(ex5.pnl.sum(),2),ex_top5_mean_r=round(ex5.pnl.mean()/50,4),top5_sum=round(srt.iloc[:k].pnl.sum(),2),
  weekly={i:dict(n=int(r['size']),sum=r['sum']) for i,r in wk.iterrows()},exit_mix=h.groupby('exit_reason').pnl.agg(['size','sum']).round(2).to_dict('index'),
  last2wk=round(h[h.td>='2026-10-09'].pnl.sum(),2), n_last2wk=int((h.td>='2026-10-09').sum()))
print(OUT['1704_book'])
d=pd.read_sql("select * from dry_trades where strategy='hod_break' and trade_date>='2026-09-25' and trade_date<='2026-10-22' and source not in ('backfill_journal','replay_capped')",con)
print(d.source.value_counts().to_dict(), d.r_multiple.isna().sum())
d=d.dropna(subset=['r_multiple']).rename(columns={'trade_date':'td'})
pm,pse,pseiid,pnd=clus(d,'r_multiple')
OUT['1704_pop']=dict(n=len(d),mean_r=round(pm,4),se_cluster=round(pse,4),se_iid=round(pseiid,4),ndays=pnd,ci95=[round(pm-1.96*pse,3),round(pm+1.96*pse,3)],by_source=d.groupby('source').r_multiple.agg(['size','mean']).round(4).to_dict('index'))
print(OUT['1704_pop'])
# same days diff: day-level cluster bootstrap on difference of per-day means? use pooled means with cluster SE jointly
days=sorted(set(h.td)&set(d.td))
hs=h[h.td.isin(days)]; ds=d[d.td.isin(days)]
rng=np.random.default_rng(1); diffs=[]
hg={k:v.r.values for k,v in hs.groupby('td')}; dg={k:v.r_multiple.values for k,v in ds.groupby('td')}
for _ in range(5000):
    s=rng.choice(days,len(days)); a=np.concatenate([hg.get(x,np.array([])) for x in s]); c=np.concatenate([dg.get(x,np.array([])) for x in s])
    diffs.append(a.mean()-c.mean())
OUT['1704_diff']=dict(days_common=len(days),book_mean=round(hs.r.mean(),4),book_n=len(hs),pop_mean=round(ds.r_multiple.mean(),4),pop_n=len(ds),diff=round(hs.r.mean()-ds.r_multiple.mean(),4),boot_ci=[round(float(np.percentile(diffs,2.5)),3),round(float(np.percentile(diffs,97.5)),3)],
   book_only_days=sorted(set(h.td)-set(d.td)),pop_only_days=sorted(set(d.td)-set(h.td)))
print(OUT['1704_diff'])
json.dump(OUT,open('/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/part_a.json','w'),indent=1,default=str)
