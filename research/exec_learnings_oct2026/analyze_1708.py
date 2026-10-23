"""Cell 1,708 analysis: old (results.csv) vs new (fills_1708.csv) by period and delay, with day-clustered CI."""
import numpy as np, pandas as pd
R=375.0
old=pd.read_csv('research/orb_latency_bt/results.csv'); new=pd.read_csv('research/exec_learnings_oct2026/fills_1708.csv')
pop=pd.read_csv('research/orb_latency_bt/population.csv')
pop['shares']=pop['_rp_position']/pop['entry_price']
def prep(df):
    df=df[df.status!='missing_tick_data'].copy(); df['d']=pd.to_datetime(df.date); return df
old,new=map(prep,(old,new))
def stats(x,dates):
    x=x.fillna(0.0)/R; n=len(x); m=x.mean(); cs=(x-m).groupby(dates.values).sum()
    se=np.sqrt((cs**2).sum())/n; k=int(np.ceil(.05*n)); ex=x.sort_values(ascending=False).iloc[k:].mean()
    return m,se,ex
per={'OOS(2023-24+25H2-26-09)':lambda d:d.period.isin(['2023-24','2025H2-2026-09']),
 'May-Sep26':lambda d:(d.d>='2026-05-01')&(d.d<='2026-09-30'),'2023-24':lambda d:d.period=='2023-24',
 '2025H1':lambda d:d.period=='2025H1','2025H2-26-09':lambda d:d.period=='2025H2-2026-09','ALL':lambda d:d.period.notna()}
rows=[]
for pn,f in per.items():
    for dl in sorted(new.delay_s.unique()):
        o=old[(old.delay_s==dl)&f(old)]; n=new[(new.delay_s==dl)&f(new)]
        mo,so,eo=stats(o.pnl_replay,o.date); mn,sn,en=stats(n.pnl_replay,n.date)
        rows.append(dict(period=pn,delay=dl,n=len(n),n_filled=int((n.status=='filled').sum()),old_R=mo,old_usd=o.pnl_replay.sum(),old_exTop5=eo,
          new_R=mn,new_se=sn,new_lo=mn-1.96*sn,new_hi=mn+1.96*sn,new_usd=n.pnl_replay.sum(),new_exTop5=en,new_t=mn/sn if sn>0 else np.nan))
t=pd.DataFrame(rows); t.to_csv('research/exec_learnings_oct2026/table_1708.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
print(t.round(3).to_string())
# 30bps match + price effect
d0=new[(new.delay_s==0)&(new.status=='filled')].merge(pop[['symbol','date','entry_price','shares']],on=['symbol','date'])
d0['diff']=d0.fill_price-d0.entry_price
print('d0 filled',len(d0),'match<=1c',(d0['diff'].abs()<=0.01).mean(),'mean diff $',d0['diff'].mean(),'mean diff bps',(d0['diff']/d0.entry_price*1e4).mean(),'median bps',(d0['diff']/d0.entry_price*1e4).median())
print('mean price term $ at d0 (shares*(entry-fill))', (-(d0['diff'])*d0.shares).mean())
# sanity: old-formula reproduction check = old pnl_replay vs _sized_pnl for same fills
chk=new.merge(old[['symbol','date','delay_s','pnl_replay']],on=['symbol','date','delay_s'],suffixes=('','_old'))
chk=chk[chk.status=='filled']; print('mean |new-old| per filled row $',(chk.pnl_replay-chk.pnl_replay_old).abs().mean(), 'mean new-old',(chk.pnl_replay-chk.pnl_replay_old).mean())
# max single-trade ratios
print('max |pnl_new| ', new.pnl_replay.abs().max(), ' min',new.pnl_replay.min(),' max',new.pnl_replay.max())
# fills at d=0 hurt: share with fill > entry
print('share fill>BT entry', (d0['diff']>0.01).mean(),'share fill<BT', (d0['diff']<-0.01).mean())
