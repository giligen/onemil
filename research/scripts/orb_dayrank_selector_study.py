"""Parameterless selector study (2026-07-17 owner mandate).

Candidate D: per-day cross-sectional percentile ranking — no TRAIN
constants at all. Each morning, each candidate's 7 features are ranked
WITHIN the day's cohort (signed percentiles, averaged). Selection =
top-4 by day-rank after the same hard gates (PDR veto, family dedup).
Sizing = flat base × news-gated PM mult (unchanged). Exits identical.

vs Baseline A (current live): frozen z-composite, threshold 0, Q1 skip,
Q4-first ordering, flat mults.

The whole 18mo is OOS for D by construction (nothing is fitted).
"""
import os, sys
import pandas as pd, numpy as np
sys.path.insert(0, '/home/ec2-user/onemil')
from trading.orb_correlation import symbol_family, symbol_super_group
from trading.orb_pdr_veto import pdr_veto_applies
from trading.orb_pm_mult import pm_size_multiplier
from trading.orb_asset_class import effective_has_news, load_class_map
import glob, yaml

FEATS = [('gap_pct',-1),('range_total_volume',-1),('range_avg_bar_range_pct',-1),
         ('range_size_pct',-1),('price_vs_20d_high_pct',-1),
         ('prev_day_close_position',-1),('range_close_position',1)]
N_SLOTS=4

df=pd.read_csv('/tmp/orb_cands_liveparity.csv')
df['date']=pd.to_datetime(df['date'])
df['day']=df['date'].dt.strftime('%Y-%m-%d')

# frozen composite (baseline A path) from orb.yaml
cfg=yaml.safe_load(open('orb.yaml'))
fp=cfg['filter']['features']; cuts=cfg['quintile_cutoffs']; thr=float(cfg['filter'].get('threshold',0.0))
comp=pd.Series(0.0,index=df.index)
for f,sgn in FEATS:
    comp=comp+((df[f]-fp[f]['mean'])/fp[f]['std'])*fp[f]['sign']
df['_comp_frozen']=comp/len(FEATS)
def quintile(c):
    for i,cu in enumerate(cuts):
        if c<cu: return f"Q{i+1}"
    return "Q5"
df['_q']=df['_comp_frozen'].map(quintile)

# day-rank composite (parameterless)
def dayrank(g):
    s=pd.Series(0.0,index=g.index)
    n=len(g)
    for f,sgn in FEATS:
        r=g[f].rank(pct=True, method='average')
        s=s+(r if sgn>0 else (1.0-r))
    return s/len(FEATS)
df['_dayrank']=df.groupby('day',group_keys=False).apply(lambda g: dayrank(g))

# news / pm inputs (shared with pipeline logic)
pm=pd.concat([pd.read_csv(x) for x in glob.glob('data/research/orb_premarket_dollar_vol_*.csv')]).dropna(subset=['pm_dollar_vol']).drop_duplicates(['symbol','day'],keep='last')
nw=pd.concat([pd.read_csv(x) for x in glob.glob('data/research/orb_news_catalyst_*.csv')]).drop_duplicates(['symbol','day'],keep='last')
cmap=load_class_map()
pmm={(r['symbol'],r['day']):r['pm_dollar_vol'] for _,r in pm.iterrows()}
nwm={(r['symbol'],r['day']):(r['n_articles'] or 0)>0 for _,r in nw.iterrows()}

def pm_mult_for(sym,day):
    hn=nwm.get((sym,day))
    hn=effective_has_news(hn, cmap.get(sym,'unknown')) if hn is not None else None
    return pm_size_multiplier(pmm.get((sym,day)), has_news=hn)

def select(day_df, mode):
    d=day_df.copy()
    if mode=='A':
        d=d[d['_comp_frozen']>=thr]
        d=d[d['_q']!='Q1']
        qorder={'Q4':0,'Q5':1,'Q3':2,'Q2':3}
        d['_qr']=d['_q'].map(qorder)
        d=d.sort_values(['_qr','_comp_frozen'],ascending=[True,False])
    elif mode=='D0':
        d=d.sort_values('_dayrank',ascending=False)
    elif mode=='D1':
        d=d[d['_dayrank']>=0.5].sort_values('_dayrank',ascending=False)
    picks=[]; fams=set(); sgs=set(); slots=0
    for _,r in d.iterrows():
        if slots>=N_SLOTS: break
        f=symbol_family(r['symbol']); s=symbol_super_group(r['symbol'])
        if f and f in fams: continue
        if s and s in sgs: continue
        # PDR veto consumes the slot (no refill), exactly like live
        slots+=1
        if f: fams.add(f)
        if s: sgs.add(s)
        v=r.get('prev_day_range_pct')
        if pdr_veto_applies(None if pd.isna(v) else float(v)):
            continue
        picks.append(r)
    return picks

def run(mode):
    rows=[]
    for day,g in df.groupby('day'):
        for r in select(g,mode):
            rows.append({'day':day,'symbol':r['symbol'],
                         'pnl':r['_rp_pnl']*pm_mult_for(r['symbol'],day)})
    b=pd.DataFrame(rows); b['date']=pd.to_datetime(b['day'])
    return b

def stats(b,label):
    s=b['pnl']
    eras='  '.join(f"{e} ${s[(b['date']>=lo)&(b['date']<hi)].sum():+,.0f}"
        for e,lo,hi in [('25H1','2025-01-01','2025-07-01'),
                        ('25H2','2025-07-01','2026-01-01'),
                        ('2026','2026-01-01','2027-01-01')])
    daily=b.groupby('date')['pnl'].sum().sort_index(); cum=daily.cumsum()
    ym=b.groupby(b['date'].dt.strftime('%Y-%m'))['pnl'].sum()
    print(f"{label:14s} TOT ${s.sum():+,.0f}  {eras}  MDD ${(cum-cum.cummax()).min():+,.0f}  "
          f"negMo {(ym<0).sum()}/19  n={len(b)}")
    return b

A=stats(run('A'),'A frozen(live)')
D0=stats(run('D0'),'D0 dayrank')
D1=stats(run('D1'),'D1 dayrank>=.5')

# rigor: leave-out on D-vs-A delta
for name,B in [('D0',D0),('D1',D1)]:
    a_by=A.groupby('day')['pnl'].sum(); b_by=B.groupby('day')['pnl'].sum()
    delta=(b_by.reindex(sorted(set(a_by.index)|set(b_by.index)),fill_value=0)
           -a_by.reindex(sorted(set(a_by.index)|set(b_by.index)),fill_value=0))
    ds=delta.sort_values(ascending=False); tot=delta.sum()
    lo=[f"top{k}:{tot-ds.head(k).sum():+,.0f}" for k in [1,3,5,10]]
    print(f"{name}-vs-A day-delta leave-out: full {tot:+,.0f} | {' | '.join(lo)}")
