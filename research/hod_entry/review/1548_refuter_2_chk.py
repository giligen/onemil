"""Refuter 2 (statistics/artifacts) for cells 1,548-1,549: recomputes from cell_1548_fills.csv."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
H='/home/ec2-user/onemil/research/hod_entry/'
d=pd.read_csv(H+'cell_1548_fills.csv')
f=d[d.status=='filled'].copy()

def dct(x,day):
    """Day-clustered t of the mean."""
    g=pd.DataFrame({'x':x.values,'d':day.values})
    n=len(g); m=g.x.mean(); r=(g.x-m).groupby(g.d).sum()
    se=np.sqrt((r**2).sum())/n; return m/se if se>0 else np.nan

def cap_book(s):
    """Live cap: per day, in entry order, <=12 entries and <=4 concurrent."""
    keep=[]
    for day,g in s.sort_values(['day','entry_m']).groupby('day'):
        open_exits=[];k=0
        for i,r in g.iterrows():
            open_exits=[e for e in open_exits if e>r.entry_m]
            if k<12 and len(open_exits)<4:
                keep.append(i);k+=1;open_exits.append(r.exit_m)
    return s.loc[keep]

def stats(x,day):
    """mean, t, ex-top5, ex-top1, drop best 2 days, cap3, top-5% share of sum."""
    x=x.reset_index(drop=True);day=day.reset_index(drop=True)
    q5=x.quantile(.95);q1=x.quantile(.99)
    dm=x.groupby(day).sum().sort_values(ascending=False)
    drop2=x[~day.isin(dm.index[:2])]
    return dict(n=len(x),mean=round(x.mean(),4),t=round(dct(x,day),2),
        ex5=round(x[x<=q5].mean(),4),ex1=round(x[x<=q1].mean(),4),
        drop2d=round(drop2.mean(),4),cap3=round(x.clip(upper=3).mean(),4),
        med=round(x.median(),4),top2day_share=round(dm.iloc[:2].sum()/x.sum(),2) if x.sum()!=0 else None)

rows=[]
for (c,sp,k),g in f.groupby(['cell','split','kept']):
    for coh,gg in (('all',g),('realSIP',g[g.is_real_sip]),('cacheOnly',g[~g.is_real_sip])):
        s=stats(gg.net_R,gg.day); s.update(cell=c,split=sp,kept=k,cohort=coh); rows.append(s)
    cb=cap_book(g); s=stats(cb.net_R,cb.day); s.update(cell=c,split=sp,kept=k,cohort='CAPPED')
    wk=pd.to_datetime(g.day).dt.to_period('W').nunique(); s['wks']=wk; s['fills_wk_cap']=round(len(cb)/wk,1)
    rows.append(s)
R=pd.DataFrame(rows)[['cell','split','kept','cohort','n','mean','t','ex5','ex1','drop2d','cap3','med','top2day_share']+['fills_wk_cap']]
pd.set_option('display.width',250); print(R.to_string(index=False))

# kept - dropped difference, day-clustered (difference of day means via paired days)
print('\nkept-dropped diff (day-block bootstrap 2000, seed 1)')
rng=np.random.default_rng(1)
for (c,sp),g in f.groupby(['cell','split']):
    for coh,gg in (('all',g),('realSIP',g[g.is_real_sip]),('cacheOnly',g[~g.is_real_sip])):
        days=gg.day.unique(); by={dy:x for dy,x in gg.groupby('day')}
        obs=gg[gg.kept].net_R.mean()-gg[~gg.kept].net_R.mean()
        bs=[]
        for _ in range(2000):
            s=pd.concat([by[dy] for dy in rng.choice(days,len(days))])
            bs.append(s[s.kept].net_R.mean()-s[~s.kept].net_R.mean())
        bs=np.array(bs); print(c,sp,coh,'diff',round(obs,4),'CI95',np.round(np.percentile(bs,[2.5,97.5]),3),'P(<=0)',round((bs<=0).mean(),3))

# base outcome kept vs dropped (the model's own lift under the base trade)
b=d[d.cell==1549]
p=pd.read_csv(H+'model_1478_L3_v2_predictions.csv')
for sp in ('TRAIN','VAL'):
    q=p[p.split==sp].dropna(subset=['L3'])
    a=roc_auc_score(q.L3,q.hgb_prob_L3); ar=roc_auc_score(q[q.store_served_1438==0].L3,q[q.store_served_1438==0].hgb_prob_L3)
    ac=roc_auc_score(q[q.store_served_1438==1].L3,q[q.store_served_1438==1].hgb_prob_L3)
    print(sp,'AUC all',round(a,3),'realSIP',round(ar,3),'cacheOnly',round(ac,3),'base rate',round(q.L3.mean(),3),
          'kept prec',round(q[q.hgb_kept_L3].L3.mean(),3),'realSIP kept prec',round(q[(q.hgb_kept_L3)&(q.store_served_1438==0)].L3.mean(),3),
          'cache kept prec',round(q[(q.hgb_kept_L3)&(q.store_served_1438==1)].L3.mean(),3))
    print('  base outcome kept',round(q[q.hgb_kept_L3].outcome_R.mean(),3),'dropped',round(q[~q.hgb_kept_L3].outcome_R.mean(),3))
# sweep lottery arithmetic
s=f[(f.cell==1548)]
print('\n1548 target share by split/kept'); print(s.groupby(['split','kept']).why.value_counts(normalize=True).unstack().round(3))
# monthly VAL/TRAIN
f['mo']=f.day.str[:7]
print(f[f.kept].groupby(['cell','mo']).net_R.agg(['count','mean']).round(3).unstack(0))
