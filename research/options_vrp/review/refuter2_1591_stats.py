"""Refuter 2 (statistics & risk) for cells 1,591-1,598: recompute every number from the cycles CSV."""
import math, pandas as pd, numpy as np
from scipy.stats import spearmanr
D='/home/ec2-user/onemil/research/options_vrp/'
B=6500.0
c=pd.read_csv(D+'cell_1591_cycles.csv',parse_dates=['entry_date','exit_date','expiry'])
m=pd.read_csv(D+'cell_1591_monthly.csv')
SPL={'TRAIN':('2024-02-05','2025-06-30'),'VAL':('2025-07-07','2026-08-17')}
def split(df,s):
    a,b=SPL[s]; return df[(df.entry_date>=a)&(df.entry_date<=b)]
def monthly(df,s,col='pnl_usd'):
    a,b=SPL[s]; months=pd.period_range(pd.Period(a,'M'),pd.Period(b,'M'),freq='M')
    g=df.groupby(df.exit_date.dt.to_period('M'))[col].sum()
    return g.reindex(months,fill_value=0.0)
def monthly_full(df,col='pnl_usd'):
    """All exits, no truncation at split end."""
    g=df.groupby(df.exit_date.dt.to_period('M'))[col].sum()
    months=pd.period_range(g.index.min(),g.index.max(),freq='M'); return g.reindex(months,fill_value=0.0)
rows=[]
for cell,g in c.groupby('cell'):
    for s in SPL:
        d=split(g,s); mo=monthly(d,s); r=mo/B
        trunc=d[d.exit_date.dt.to_period('M')>pd.Period(SPL[s][1],'M')].pnl_usd.sum()
        cum=mo.cumsum(); dd=-(cum-cum.cummax()).min()
        srt=d.pnl_usd.sort_values(ascending=False); k=max(1,math.ceil(0.05*len(srt)))
        rows.append(dict(cell=cell,split=s,n=len(d),months=len(mo),mean_ret=r.mean(),sharpe=r.mean()/r.std(ddof=1)*math.sqrt(12),
            green=(mo>0).mean(),worst=mo.min(),maxdd=dd,sum=d.pnl_usd.sum(),exits_dropped_pnl=trunc,
            top5_usd=srt.iloc[:k].sum(),ex_top5=srt.iloc[k:].sum(),top_month_share=mo.max()/mo.sum() if mo.sum() else np.nan,
            top2_month_share=mo.nlargest(2).sum()/mo.sum() if mo.sum() else np.nan,
            aug24=mo.get(pd.Period('2024-08'),np.nan),apr25=mo.get(pd.Period('2025-04'),np.nan),
            ret005=monthly(d,s,'pnl_usd_slip005').mean()/B,ret010=monthly(d,s,'pnl_usd_slip010').mean()/B,
            naked=d.naked_pnl_usd.sum(),naked_worst_cycle=d.naked_pnl_usd.min(),spread_worst_cycle=d.pnl_usd.min(),
            mean_worstcase=d.worst_case_usd.mean()))
R=pd.DataFrame(rows); pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
print(R.round(4).to_string())
t=R[R.split=='TRAIN'].set_index('cell'); v=R[R.split=='VAL'].set_index('cell')
print('\nSpearman TRAIN->VAL sharpe', spearmanr(t.sharpe,v.sharpe)); print('Spearman mean_ret',spearmanr(t.mean_ret,v.mean_ret))
print('TRAIN rank of each cell by sharpe:',t.sharpe.rank(ascending=False).to_dict()); print('VAL rank:',v.sharpe.rank(ascending=False).to_dict())
# budget assertion: sum of worst cases of positions open at each entry
for cell,g in c.groupby('cell'):
    g=g.sort_values('entry_date'); mx=0; mxn=0
    for i,rw in g.iterrows():
        op=g[(g.entry_date<=rw.entry_date)&(g.exit_date>rw.entry_date)]
        mx=max(mx,op.worst_case_usd.sum()); mxn=max(mxn,len(op))
    print(f'cell {cell}: max concurrent worst-case ${mx:,.0f} (B={B:,.0f}) max open {mxn}, mean contracts {g.contracts.mean()}')
# 1597 detail
g=c[c.cell==1597].sort_values('entry_date')
print(g[['entry_date','expiry','short_strike','long_strike','credit','pnl_usd','pnl_usd_slip010','naked_pnl_usd','worst_case_usd']].to_string())
print('1597 full monthly (no truncation):'); print(monthly_full(g).to_string())
R.to_csv(D+'review/refuter2_1591_cellstats.csv',index=False)
