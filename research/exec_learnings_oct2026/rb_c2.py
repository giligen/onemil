import pandas as pd, numpy as np, json
S='/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/'
d=pd.read_csv(S+'rb_c_rows.csv'); d=d[d.status!='missing_tick_data'].copy()
d['marketable']=d.marketable.astype(bool)
out={}
for nm,m in [('in',d.date>='2025-01-01'),('out',d.date<'2025-01-01')]:
    for c,mm in [('MKT',d.marketable),('REST',~d.marketable)]:
        g=d[m&mm]; out[f'{nm}_{c}_BT_R_all_covered']=dict(n=len(g),meanR_bt=round(g.R_bt.mean(),4))
    f=d[m&(d.status=='filled')&~d.marketable]; k=int(np.ceil(.05*len(f)))
    out[f'{nm}_REST_filled_ex5']=round(f.R.sort_values(ascending=False).iloc[k:].mean(),4)
f=d[(d.status=='filled')&~d.marketable&(d.date>='2025-01-01')]
out['no_print_le5']=int(d.no_print_le5.sum()); out['n_covered']=len(d)
out['wk_in_filled_rest']=round(len(f)/ (pd.to_datetime(d[d.date>='2025-01-01'].date).max()-pd.to_datetime('2025-01-01')).days*7,2)
print(out)
h=pd.read_csv('/dev/null') if False else None
a=json.load(open('research/exec_learnings_oct2026/rebuild_1702_1704_1707.json')); a['1707_extra']=out
json.dump(a,open('research/exec_learnings_oct2026/rebuild_1702_1704_1707.json','w'),indent=1,default=str)
