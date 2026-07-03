import pandas as pd, numpy as np, sys, json
sys.path.insert(0,'/home/ec2-user/onemil')
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import FILTER_THRESHOLD, fit_quintile_cutoffs, assign_quintile, ADAPTIVE_MULT_MIN
from trading.orb_correlation import symbol_family, symbol_super_group
ACCOUNT=100_000.0; N=4; RISK=3000.0; MIN_STOP_PCT=1.0; OLD_POS=50_000.0
Q_CAPS={'Q1':3.0,'Q2':3.0,'Q3':3.0,'Q4':3.0,'Q5':1.5}
Q_ORDER={'Q4':0,'Q5':1,'Q3':2,'Q2':3,'Q1':4}
def load_candidates():
    c=pd.read_csv('/tmp/orb_candidates_resim.csv'); c['date']=pd.to_datetime(c['date']); return c
def run_pipeline(df, train_start='2025-01-01', train_end='2025-06-30', exclude=None, skip_q1=True, n_slots=N):
    df=df.copy()
    if exclude is not None: df=df[~df['symbol'].isin(exclude)]
    per_pos_cap=ACCOUNT/n_slots
    stop=df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    df['_rp_position']=(RISK/(stop/100.0)).clip(upper=per_pos_cap)
    df['_rp_pnl']=df['pnl']*df['_rp_position']/OLD_POS
    train=df[(df['date']>=train_start)&(df['date']<=train_end)]
    params=fit_z_params(train,FILTER_FEATURES)
    df['_composite']=composite_score(df,params)
    train=df[(df['date']>=train_start)&(df['date']<=train_end)]
    train_k=train[train['_composite']>=FILTER_THRESHOLD].copy()
    cutoffs=fit_quintile_cutoffs(train_k['_composite'])
    train_k['_quintile']=assign_quintile(train_k['_composite'],cutoffs)
    avg=float(train_k['_rp_pnl'].mean())
    mults={q:max(ADAPTIVE_MULT_MIN,min(Q_CAPS[q],float(train_k[train_k['_quintile']==q]['_rp_pnl'].mean())/avg)) for q in Q_CAPS}
    kept=df[df['_composite']>=FILTER_THRESHOLD].copy()
    kept['_quintile']=assign_quintile(kept['_composite'],cutoffs)
    if skip_q1: kept=kept[kept['_quintile']!='Q1']
    rows=[]
    for day,dg in kept.groupby('date'):
        d=dg.copy(); d['_q_rank']=d['_quintile'].map(Q_ORDER)
        d=d.sort_values(['_q_rank','_composite'],ascending=[True,False])
        sf=set(); ss=set(); today=[]
        for _,r in d.iterrows():
            f=symbol_family(r['symbol']); s=symbol_super_group(r['symbol'])
            if f and f in sf: continue
            if s and s in ss: continue
            if f: sf.add(f)
            if s: ss.add(s)
            today.append(r)
            if len(today)>=n_slots: break
        rows.extend(today)
    sel=pd.DataFrame(rows)
    sel['_sized_pnl']=sel.apply(lambda r:r['_rp_pnl']*mults[r['_quintile']],axis=1)
    return sel
def summarize(sel,label):
    sel=sel.copy(); sel['ym']=sel['date'].dt.strftime('%Y-%m')
    tot=sel['_sized_pnl'].sum()
    y26=sel[sel['date']>='2026-01-01']['_sized_pnl'].sum()
    recent=sel[sel['date']>='2026-05-01']['_sized_pnl'].sum()
    daily=sel.groupby('date')['_sized_pnl'].sum().sort_index()
    cum=daily.cumsum(); mdd=(cum-cum.cummax()).min()
    negm=(sel.groupby('ym')['_sized_pnl'].sum()<0).sum()
    print(f"{label:<28} n={len(sel):>4}  TOT ${tot:>+9,.0f}  2026 ${y26:>+8,.0f}  May-Jul26 ${recent:>+8,.0f}  MDD ${mdd:>+8,.0f}  negMo {negm}")
    return dict(label=label,n=len(sel),tot=tot,y26=y26,recent=recent,mdd=mdd)
