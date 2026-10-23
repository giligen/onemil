"""Cell 1,703: engine ORB SCORED set vs nightly features CSV, per session. Read-only."""
import glob, re, subprocess, sys, json, sqlite3
sys.path.insert(0, '/home/ec2-user/onemil')
import yaml, pandas as pd
from trading.orb_csv import read_orb_csv
from trading.orb_filter import composite_score, load_feature_params, assign_quintile
from trading.orb_correlation import dedup_candidates
ROOT='/home/ec2-user/onemil'
cfg=yaml.safe_load(open(ROOT+'/orb.yaml'))
params=load_feature_params(cfg['filter'])
thr=float(cfg['filter']['threshold']); cuts=cfg['quintile_cutoffs']
order=cfg['ranking']['order']; slots=int(cfg['sizing']['max_concurrent'])
skipq1=bool(cfg['filter'].get('skip_q1'))
ansi=re.compile(r'\x1b\[[0-9;]*m')
days=['2026-10-06','2026-10-07','2026-10-08','2026-10-09','2026-10-12','2026-10-13','2026-10-14','2026-10-15','2026-10-16','2026-10-19','2026-10-20','2026-10-21','2026-10-22']
def engine(day):
    out={}
    for ln in open(f'{ROOT}/logs/session_archive/{day}.log',errors='replace'):
        if 'ORB SCORED' not in ln and 'below filter threshold' not in ln: continue
        ln=ansi.sub('',ln)
        m=re.search(r'ORB SCORED: (\S+) comp=(-?[\d.]+) (\S+) \| gap=(\S+) rtv=(\S+) rabr=\S+ rs=\S+ p20h=\S+ pdcp=(\S+) rcp=\S+ \| prev_close=(\S+) range_open=(\S+) pool=(\S+)',ln)
        if m:
            s=m.group(1)
            if s not in out:
                out[s]=dict(comp=float(m.group(2)),q=m.group(3),gap=float(m.group(4)),rtv=float(m.group(5)),prev_close=float(m.group(7)),ropen=float(m.group(8)),pool=m.group(9))
    return out
def rank(rows):
    """rows: sym->(comp, gap). returns top-slots list after thr/skip_q1/Q-order/dedup"""
    sc=[]
    for s,(c,g) in rows.items():
        if c is None or c<thr: continue
        if g is not None and g<float(cfg['universe']['min_gap_pct']): continue
        q=assign_quintile(c,cuts)
        if skipq1 and q=='Q1': continue
        sc.append((order.index(q) if q in order else 99,-c,s))
    sc.sort()
    ranked=[s for _,_,s in sc]
    return ranked, dedup_candidates(ranked,max_keep=slots,by_family=True,by_super_group=True)
res={}
for day in days:
    eng=engine(day)
    paths=sorted(glob.glob(f'{ROOT}/analysis_results/orb_features_{day.replace("-","")}_*.csv'))
    paths=[p for p in paths if 'corrmatrix' not in p]
    f=read_orb_csv(paths[-1]); f=f[f['date'].astype(str).str.startswith(day)].copy()
    bt={}
    for _,r in f.iterrows():
        c=composite_score({k:r[k] for k in params if k in f.columns},params)
        bt[r['symbol']]=dict(comp=c,gap=r['gap_pct'],rtv=r['range_total_volume'],entered=int(r['entered']),pdr=r['prev_day_range_pct'],price=r['entry_price'])
    prod={s:v for s,v in eng.items() if v['pool']=='production'}
    er,etop=rank({s:(v['comp'],v['gap']) for s,v in prod.items()})
    br,btop=rank({s:(v['comp'],v['gap']) for s,v in bt.items()})
    res[day]=dict(csv=paths[-1].split('/')[-1],n_eng_all=len(eng),n_eng_prod=len(prod),n_bt=len(bt),
      eng_only=sorted(set(prod)-set(bt)),bt_only=sorted(set(bt)-set(prod)),
      addon=sorted(s for s,v in eng.items() if v['pool']!='production'),
      eng_rank=er,bt_rank=br,eng_top=etop,bt_top=btop,
      bt_entered=sorted(s for s,v in bt.items() if v['entered']),
      eng={s:v for s,v in eng.items()},bt={s:{k:(None if v[k]!=v[k] else v[k]) for k in v} for s,v in bt.items()})
json.dump(res,open(ROOT+'/research/exec_learnings_oct2026/cell1703_raw.json','w'),default=str)
for d,r in res.items():
    print(d,r['csv'],'eng_prod',r['n_eng_prod'],'addon',len(r['addon']),'bt',r['n_bt'],'eng_only',len(r['eng_only']),'bt_only',len(r['bt_only']),'bt_entered',len(r['bt_entered']))
    print('   etop',r['eng_top']); print('   btop',r['bt_top']); print('   entered',r['bt_entered'])
