import sys, json, os, numpy as np, pandas as pd
sys.path.insert(0,'/home/ec2-user/onemil/research/orb_latency_bt')
import replay as rp
SUBMIT=5.0
pop=pd.read_csv(rp.POP_CSV)
rows=[]
for r in pop.to_dict('records'):
    path=f"{rp.RAW_DIR}/{r['date']}__{r['symbol']}.parquet"
    base=dict(symbol=r['symbol'],date=r['date'],period=r['period'],R_bt=r['_sized_pnl']/rp.R_DENOM,bt_exit=r['exit_reason'],trigger=r['trigger'])
    if not os.path.exists(path):
        rows.append(dict(base,status='missing_tick_data')); continue
    raw=pd.read_parquet(path)
    tr=raw[raw.schema=='trades'].copy(); mb=raw[raw.schema=='mbp-1'].copy()
    tr['t_sec']=rp.to_sec_since_0935(tr.ts_event.values,r['date']); mb['t_sec']=rp.to_sec_since_0935(mb.ts_event.values,r['date'])
    tr=tr.sort_values('t_sec'); mb=mb.sort_values('t_sec')
    prior=tr[tr.t_sec<=SUBMIT]
    last=float(prior.price.iloc[-1]) if len(prior) else np.nan
    res=rp.replay_one(tr,mb,r['trigger'],r['limit'],SUBMIT)
    shares=0.0 if pd.isna(r['shares']) else r['shares']
    if res['status']=='filled':
        pnl=r['_sized_pnl']+shares*(r['entry_price']-res['fill_price']); pnlfix=r['_sized_pnl']+(r['_rp_position']/r['entry_price'])*(r['entry_price']-res['fill_price']); tf=max(res['t_star'],SUBMIT)-SUBMIT
    else: pnl=0.0; pnlfix=0.0; tf=np.nan
    rows.append(dict(base,status=res['status'],t_star=res['t_star'],last_print=last,no_print_le5=bool(np.isnan(last)),
        marketable=bool(last>=r['trigger']) if not np.isnan(last) else False,
        marketable_p10=bool(last>=r['trigger']*1.001) if not np.isnan(last) else False,
        marketable_p30=bool(last>=r['trigger']*1.003) if not np.isnan(last) else False,
        R=pnl/rp.R_DENOM,R_fix=pnlfix/rp.R_DENOM,t_fill=tf))
d=pd.DataFrame(rows)
d.to_csv('/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/rb_c_rows.csv',index=False)
print(d.status.value_counts().to_dict())
d=d[d.status!='missing_tick_data'].copy()
for c in ['marketable','marketable_p10','marketable_p30','no_print_le5']: d[c]=d[c].astype(bool)
def regime(x):
    if x<'2024-07-01': return 'oos_2023_2024H1'
    if x<'2025-01-01': return 'oos_2024H2'
    return 'in_2025_2026'
d['regime']=d.date.map(regime)
def stats(g):
    f=g[g.status=='filled']
    out=dict(n_all=len(g),n_filled=len(f),mean_R_all=round(g.R.mean(),4) if len(g) else None,mean_R_filled=round(f.R.mean(),4) if len(f) else None,
      sum_R_all=round(g.R.sum(),2),stop_share_filled=round(float((f.bt_exit=='stop').mean()),4) if len(f) else None,mean_Rbt_filled=round(f.R_bt.mean(),4) if len(f) else None,mean_Rfix_all=round(g.R_fix.mean(),4),mean_Rfix_filled=round(f.R_fix.mean(),4) if len(f) else None)
    if len(g)>1:
        m=g.R.mean(); day=g.groupby('date').R; res=day.sum()-m*day.size(); nd=day.ngroups
        se=np.sqrt((res**2).sum()*nd/max(nd-1,1))/len(g); out['se_day']=round(se,4); out['ci95']=[round(m-1.96*se,3),round(m+1.96*se,3)]
        k=int(np.ceil(.05*len(g))); out['ex_top5_mean_R_all']=round(g.R.sort_values(ascending=False).iloc[k:].mean(),4); out['ex_top5_mean_Rfix_all']=round(g.R_fix.sort_values(ascending=False).iloc[k:].mean(),4)
    return out
OUT={}
def block(name,sub):
    OUT[name]={}
    for coh,col in [('all',None),('MARKETABLE','marketable'),('RESTING','~marketable')]:
        g=sub if col is None else (sub[sub.marketable] if col=='marketable' else sub[~sub.marketable])
        OUT[name][coh]=stats(g)
    for sens in ['marketable_p10','marketable_p30']:
        OUT[name][sens+'_MKT']=stats(sub[sub[sens]]); OUT[name][sens+'_REST']=stats(sub[~sub[sens]])
    f=sub[sub.status=='filled'].copy()
    f['bucket']=pd.cut(f.t_fill,[-1,5,34,240,1e9],labels=['<=5s','5-34s','34s-4min','>4min'])
    OUT[name]['buckets']={str(b):stats(g) for b,g in f.groupby('bucket',observed=True)}
    OUT[name]['no_print_le5']=int(sub.no_print_le5.sum())
for nm,sub in [('in_2025_2026',d[d.regime=='in_2025_2026']),('oos_2023_2024H1',d[d.regime=='oos_2023_2024H1']),('oos_2024H2',d[d.regime=='oos_2024H2']),('oos_pre2025',d[d.regime!='in_2025_2026']),('all',d)]:
    block(nm,sub)
json.dump(OUT,open('/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/part_c.json','w'),indent=1)
for nm in OUT:
    print('==',nm)
    for k in ['all','MARKETABLE','RESTING']: print('FIX',k,OUT[nm][k]['n_all'],OUT[nm][k]['mean_Rfix_all'],OUT[nm][k]['mean_Rfix_filled'],OUT[nm][k].get('ex_top5_mean_Rfix_all'))
    for k,v in OUT[nm]['buckets'].items(): print('FIXB',k,v['n_all'],v['mean_Rfix_all'])
    continue
    for k in ['all','MARKETABLE','RESTING']: print(k,OUT[nm][k])
    for k,v in OUT[nm]['buckets'].items(): print(k,v)
    print('p10',OUT[nm]['marketable_p10_MKT']['n_all'],OUT[nm]['marketable_p10_MKT']['mean_R_all'],OUT[nm]['marketable_p10_REST']['mean_R_all'],'p30',OUT[nm]['marketable_p30_MKT']['n_all'],OUT[nm]['marketable_p30_MKT']['mean_R_all'],OUT[nm]['marketable_p30_REST']['mean_R_all'])
