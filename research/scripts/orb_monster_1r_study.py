"""v2: correct path ordering (stop-before-1R = never reached; tag-exited =
never alive) + simulate BE-at-1R conditional on impulse quality."""
import sqlite3, sys
import pandas as pd, numpy as np
sys.stdout.reconfigure(line_buffering=True)

sel=pd.read_csv('/tmp/univ_windows.csv')
sel=sel[sel['_slot_rank']<4].copy()
sel['stop_pct']=sel['range_size_pct'].clip(lower=1.0)
sel['risk_usd']=sel['_rp_position']*sel['stop_pct']/100.0
sel['final_r']=sel['_rp_pnl']/sel['risk_usd']
sel['news']=(sel['own_B']+sel['own_C'])>0
sel['pm_hi']=sel['pm_dollar_vol'].fillna(0)>=5816688
sel['flag2x']=sel['pm_hi']&sel['news']&(sel['cls']=='STOCK')
QM={'Q2':1.5,'Q3':1.4,'Q4':0.5,'Q5':0.5,'Q1':0.502}
sel['qmult']=sel['_quintile'].map(QM)
conn=sqlite3.connect('data/cache.db')
ARM=1.75

def bars_for(sym,day):
    df=pd.read_sql("select timestamp,open,high,low,close,volume from intraday_bars_1min "
                   "where symbol=? and bar_date=? order by timestamp",conn,params=(sym,day))
    if df.empty: return None
    ts=pd.to_datetime(df['timestamp'],utc=True).dt.tz_convert('America/New_York')
    df=df.copy(); df['min_of_day']=ts.dt.hour*60+ts.dt.minute
    return df

rows=[]
for _,t in sel.iterrows():
    if t['exit_reason'] in ('tag_bb','tag_b1'):      # exited in first ~2min — never alive at 1R
        rows.append({'symbol':t['symbol'],'day':t['day'],'state':'tagged'}); continue
    b=bars_for(t['symbol'],t['day'])
    if b is None or len(b)<10:
        rows.append({'symbol':t['symbol'],'day':t['day'],'state':'nobars'}); continue
    rng=b[(b['min_of_day']>=570)&(b['min_of_day']<575)]
    if rng.empty: rows.append({'symbol':t['symbol'],'day':t['day'],'state':'nobars'}); continue
    rh,rl=rng['high'].max(),rng['low'].min()
    entry=rh*1.003; R=entry-rl
    if R<=0: rows.append({'symbol':t['symbol'],'day':t['day'],'state':'nobars'}); continue
    win=b[(b['min_of_day']>=575)&(b['min_of_day']<635)]
    bo=win[win['high']>rh]
    if bo.empty: rows.append({'symbol':t['symbol'],'day':t['day'],'state':'nofill'}); continue
    e_idx=bo.index[0]; e_min=b.loc[e_idx,'min_of_day']
    post=b[(b.index>=e_idx)&(b['min_of_day']<945)]
    tgt=entry+R
    # ordering: first event = stop or 1R? walk
    state='held_no1R'; h_idx=None
    for i,r_ in post.iterrows():
        if r_['low']<=rl and r_['high']>=tgt:
            # both in one bar: ambiguous — assume worst (stop first) unless bar closed high
            state='reached' if r_['close']>=tgt else 'stopped_first'; h_idx=i; break
        if r_['low']<=rl: state='stopped_first'; break
        if r_['high']>=tgt: state='reached'; h_idx=i; break
    row={'symbol':t['symbol'],'day':t['day'],'state':state}
    if state=='reached':
        seg=post.loc[e_idx:h_idx]
        row['t_1R']=int(b.loc[h_idx,'min_of_day']-e_min)
        row['pullback_R']=(entry-seg['low'].min())/R
        row['pct_green']=(seg['close']>seg['open']).mean()
        # post-1R path: does it touch BE (entry) before touching ARM?
        after=post[post.index>h_idx]
        be_first=None
        for i,r_ in after.iterrows():
            if r_['high']>=entry+ARM*R: be_first=False; break
            if r_['low']<=entry: be_first=True; break
        row['be_before_arm']=be_first   # None = neither (rode to eod between 1R and arm)
    rows.append(row)
paths=pd.DataFrame(rows)
d=sel.merge(paths,on=['symbol','day'],how='left')
d.to_csv('/tmp/monster_1r_v2.csv',index=False)
rc=d[d['state']=='reached'].copy()
print(f"clean 1R-reachers (alive, 1R before stop): {len(rc)} of {len(d)} "
      f"(tagged {int((d['state']=='tagged').sum())}, stopped-first {int((d['state']=='stopped_first').sum())})")
rc['label']=np.where(rc['final_r']>=3,'MONSTER',np.where(rc['final_r']>=1,'kept','GAVEBACK'))
rc['era3']=np.where(rc['day']<'2025-07-01','25H1',np.where(rc['day']<'2026-01-01','25H2','2026'))
print("\n=== v2 era check: monster vs gaveback medians ===")
for f in ['t_1R','pullback_R','pct_green']:
    line=f"  {f:12s}"
    for era in ['25H1','25H2','2026']:
        e=rc[rc['era3']==era]
        m=e[e['label']=='MONSTER'][f].median(); g=e[e['label']=='GAVEBACK'][f].median()
        line+=f"  {era}: {m:5.2f} vs {g:5.2f}"
    print(line)

# IMPULSE flag (round numbers, no fit): fast + shallow
rc['impulse']=(rc['t_1R']<=15)&(rc['pullback_R']<=0.5)
print(f"\nimpulse@1R: {rc['impulse'].sum()}/{len(rc)}  "
      f"P(MONSTER|impulse) {rc[rc['impulse']]['label'].eq('MONSTER').mean()*100:.0f}%  "
      f"P(MONSTER|grind) {rc[~rc['impulse']]['label'].eq('MONSTER').mean()*100:.0f}%")
for era,e in rc.groupby('era3'):
    print(f"  {era}: P(M|impulse) {e[e['impulse']]['label'].eq('MONSTER').mean()*100:3.0f}% (n={e['impulse'].sum()})  "
          f"P(M|grind) {e[~e['impulse']]['label'].eq('MONSTER').mean()*100:3.0f}% (n={(~e['impulse']).sum()})")

# ===== BE-at-1R counterfactual for GRINDERS =====
# grinder that touches BE before arm -> exit 0R (at entry); else unchanged.
def sized(pnl_r):
    return pnl_r*d_all['risk_usd']*d_all['qmult']*np.where(d_all['flag2x'],2.0,1.0)
d_all=d.copy()
d_all['era3']=np.where(d_all['day']<'2025-07-01','25H1',np.where(d_all['day']<'2026-01-01','25H2','2026'))
d_all['impulse']=(d_all['t_1R']<=15)&(d_all['pullback_R']<=0.5)
base_r=d_all['final_r'].copy()
cf=base_r.copy()
mask=(d_all['state']=='reached')&(~d_all['impulse'].fillna(False))&(d_all['be_before_arm']==True)
cf[mask]=0.0     # grinders that round-tripped to BE -> flat exit
d_all['date']=pd.to_datetime(d_all['day'])
for nm,r_ in [('BASELINE',base_r),('BE@1R for grinders',cf)]:
    s=sized(r_)
    ym=pd.DataFrame({'d':d_all['date'],'p':s}).groupby(d_all['date'].dt.strftime('%Y-%m'))['p'].sum()
    daily=pd.DataFrame({'d':d_all['date'],'p':s}).groupby('d')['p'].sum().sort_index(); cum=daily.cumsum()
    per_era=' '.join(f"{e} ${s[d_all['era3']==e].sum():+,.0f}" for e in ['25H1','25H2','2026'])
    print(f"\n{nm}: TOT ${s.sum():+,.0f}  {per_era}")
    print(f"   negMo {(ym<0).sum()}/19  moσ ${ym.std():,.0f}  worstMo ${ym.min():+,.0f}  MDD ${(cum-cum.cummax()).min():+,.0f}")
n_saved=int(mask.sum()); saved_val=(0-base_r[mask]).mul(d_all['risk_usd'][mask]).sum()
print(f"\ngrinder round-trips converted to flat: {n_saved}, raw loss avoided ${saved_val:,.0f} model")
# what monsters would BE-stop kill? grinder monsters that touched BE before arm:
killed=d_all[mask&(d_all['final_r']>=3)]
print(f"monsters killed by the rule: {len(killed)}", killed[['symbol','day','final_r']].round(1).to_dict('records') if len(killed) else '')
