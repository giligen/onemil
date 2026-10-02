import pandas as pd, numpy as np, re
A=pd.read_csv('A_holdings.csv',parse_dates=['rebalance_date']); Aw=pd.read_csv('A_weekly.csv',parse_dates=['entry_date']).set_index('entry_date')
def load(t):
    h=pd.read_csv(f'B_{t}_holdings.csv',parse_dates=['rebalance_date']); w=pd.read_csv(f'B_{t}_weekly.csv',parse_dates=['week_start']).set_index('week_start'); return h,w
out=[]
def cmp(t,top=15):
    h,w=load(t)
    sa=A.groupby('rebalance_date').symbol.apply(set); sb=h.groupby('rebalance_date').symbol.apply(set)
    d=sa.index.intersection(sb.index)
    jac=pd.Series({x:len(sa[x]&sb[x])/len(sa[x]|sb[x]) for x in d})
    ra=Aw.net_ret.reindex(d); rb=w.book_return.reindex(d)
    df=pd.DataFrame({'jaccard':jac,'A_net':ra,'B_net':rb}); df['diff']=df.B_net-df.A_net
    df.to_csv(f'weekly_cmp_{t}.csv')
    print(t,'dates A',len(sa),'B',len(sb),'common',len(d),'first A',sa.index.min().date(),'B',sb.index.min().date(),'mean Jaccard',round(jac.mean(),3),'median',round(jac.median(),3),'share J=1',round((jac==1).mean(),3),'sd diff',round(df['diff'].std(),4), 'sum diff',round(df['diff'].sum(),3))
    print(' Jaccard by year', {y:round(g.mean(),2) for y,g in jac.groupby(jac.index.year)})
    if t=='base':
        ha=A.set_index(['rebalance_date','symbol']).wk_ret; hb=h.set_index(['rebalance_date','symbol']).wk_ret
        rows=[]
        for x in df['diff'].abs().sort_values(ascending=False).index[:top]:
            oa=sa[x]-sb[x]; ob=sb[x]-sa[x]
            rows.append(dict(week=x.date(),A_net=df.A_net[x],B_net=df.B_net[x],jac=df.jaccard[x],
              onlyA=' '.join(f'{s}:{ha[(x,s)]*100:.0f}%' for s in sorted(oa)),onlyB=' '.join(f'{s}:{hb[(x,s)]*100:.0f}%' for s in sorted(ob))))
        pd.DataFrame(rows).to_csv('top15_weeks.csv',index=False); print(pd.DataFrame(rows).to_string())
        # drawdown episodes
        for n,s in (('A',df.A_net),('B',df.B_net)):
            c=(1+s).cumprod(); dd=c/c.cummax()-1; tr=dd.idxmin(); pk=c[:tr].idxmax(); print(n,'maxDD',round(dd.min(),3),'peak',pk.date(),'trough',tr.date())
        # weekly return diff by year
        print((df['diff'].groupby(df.index.year).sum()).round(3).to_dict())
        # exclusions
        asn=pd.read_csv('../1700c_assets.csv',dtype=str).fillna(''); 
        ARE=re.compile(r'\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|\bPREFERRED\b|\bRIGHTS?\b',re.I)
        BRE=re.compile(r'ETF|ETN|Fund|Trust|Index|Warrant|Unit|Preferred|Depositary|Right|Notes|Bond|Portfolio',re.I)
        print('A regex',ARE.pattern);print('B regex',BRE.pattern)
        ever=set(A.symbol)|set(h.symbol); nm=dict(zip(asn.symbol,asn.name))
        allsym=set(asn.symbol)
        exA=set(asn.symbol[asn.name.str.contains(ARE)]); exB=set(asn.symbol[asn.name.str.contains(BRE)])|{s for s in allsym if re.search(r'[./]',s)}
        onlyB_ex=sorted(exB-exA); onlyA_ex=sorted(exA-exB)
        print('assets excl by A',len(exA),'by B(+dotted)',len(exB),'B-only',len(onlyB_ex),'A-only',len(onlyA_ex))
        held_A_not_B=sorted((set(A.symbol)-set(h.symbol)));held_B_not_A=sorted(set(h.symbol)-set(A.symbol))
        r=[]
        for s in sorted(ever):
            if s in exB and s not in exA: r.append((s,nm.get(s,''),'excluded by B only',int((A.symbol==s).sum()),'A weeks'))
            if s in exA and s not in exB: r.append((s,nm.get(s,''),'excluded by A only',int((h.symbol==s).sum()),'B weeks'))
        pd.DataFrame(r,columns=['symbol','name','what','weeks','who']).to_csv('exclusion_diff_held.csv',index=False)
        print('held-ever symbols differing in exclusion:',len(r)); print(pd.DataFrame(r).sort_values(3,ascending=False).head(25).to_string())
        print('names held only by A:',len(held_A_not_B),'only by B:',len(held_B_not_A))
        # attribute A-held-not-excluded-by-B weeks: loss of non-B names; fraction of diff
        h['exB']=h.symbol.isin(exB); A['exB']=A.symbol.isin(exB)
        print('A holdings (weeks) that B excludes:',int(A.exB.sum()),'of',len(A))
        # weekly-return contribution of names A-only vs B-only across all weeks
        print('mean wk_ret A-only-held weeks',A[~A.set_index(['rebalance_date','symbol']).index.isin(h.set_index(['rebalance_date','symbol']).index)].wk_ret.mean(),
              'B-only',h[~h.set_index(['rebalance_date','symbol']).index.isin(A.set_index(['rebalance_date','symbol']).index)].wk_ret.mean())
for t in ('base','all'): cmp(t)
# 2021 only, base
d=pd.read_csv('weekly_cmp_base.csv',parse_dates=['Unnamed: 0']).set_index('Unnamed: 0'); y=d[d.index.year==2021]
print('2021 A',((1+y.A_net).prod()-1),'B',((1+y.B_net).prod()-1),'jac',y.jaccard.mean())
