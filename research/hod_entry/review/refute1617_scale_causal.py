import pandas as pd, numpy as np
d=pd.read_csv('/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/causal_rows.csv')
print('level missing',d.level.isna().sum(),'bars coverage p1559',d.p1559.notna().mean().round(3),'o0930',d.o0930.notna().mean().round(3))
print('last bar ET dist', d.last1559_et.value_counts().head(4).to_dict(), 'open ET', d.o_et.value_counts().head(3).to_dict())
r=(d.close/d.p1559-1)*1e4; ro=(d.next_open/d.o0930-1)*1e4
print('panel close vs 15:59 bar close bps: median|.| %.1f p90 %.1f share>100bps %.3f share>2000 %.4f'%(r.abs().median(),r.abs().quantile(.9),(r.abs()>100).mean(),(r.abs()>2000).mean()))
print('panel next_open vs 09:30 bar open bps: median|.| %.1f p90 %.1f share>100bps %.3f share>2000 %.4f'%(ro.abs().median(),ro.abs().quantile(.9),(ro.abs()>100).mean(),(ro.abs()>2000).mean()))
lr=d.close/d.level; print('close/level ratio: min %.3f max %.3f share outside [0.7,1.5] %.4f'%(lr.min(),lr.max(),((lr<0.7)|(lr>1.5)).mean()))
d['cell']=d.cell.astype(str)
print('held by panel close vs level recompute mismatch', ((d.close>=d.level)!=(d.cell=='1617')).sum())
ok=~d['flags'].str.contains('excluded')
d['net']=d.ret_bps-10
d['held_causal']=d.p1549>=d.level
print(pd.crosstab(d.cell,d.held_causal,dropna=False))
def dct(s):
    y=s.net; g=(y-y.mean()).groupby(s.date).sum(); se=np.sqrt((g**2).sum())/len(y)*np.sqrt(len(g)/(len(g)-1)); return y.mean()/se
def rep(lbl,s):
    y=s.net.values; q=np.percentile(y,95)
    print(f'{lbl:34s} n {len(y):5d} mean {y.mean():7.2f} t {dct(s):5.2f} ex-top5 {y[y<=q].mean():7.2f} cap+10% {np.minimum(y,1000).mean():7.2f}')
for sp in ['TRAIN','VAL']:
    s=d[ok&(d.split==sp)]
    rep(f'{sp} builder held (close>=lvl)',s[s.cell=='1617'])
    rep(f'{sp} CAUSAL held (15:49>=lvl)',s[s.held_causal==True])
    rep(f'{sp} CAUSAL failed',s[s.held_causal==False])
    # bars-based return 15:59 close -> 09:30 open
    t=s[s.cell=='1617'].copy(); t['net']=(t.o0930/t.p1559-1)*1e4-10; t=t[t.net.abs()<3000].dropna(subset=['net'])
    rep(f'{sp} held, bar-price return',t)
# month table VAL
v=d[ok&(d.split=='VAL')&(d.cell=='1617')]
print(v.groupby(v.date.str[:7]).net.agg(['count','mean']).round(1).T)
