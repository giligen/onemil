"""Cell 1,703 part 2: classify engine-only / BT-only names. Read-only."""
import json, re, sqlite3, sys
sys.path.insert(0,'/home/ec2-user/onemil')
import yaml
from trading.bf_universe_filter import is_bf_eligible, load_names
from trading.orb_filter import assign_quintile
from trading.orb_correlation import dedup_candidates
ROOT='/home/ec2-user/onemil'
cfg=yaml.safe_load(open(ROOT+'/orb.yaml')); thr=float(cfg['filter']['threshold']); cuts=cfg['quintile_cutoffs']; order=cfg['ranking']['order']
raw=json.load(open(ROOT+'/research/exec_learnings_oct2026/cell1703_raw.json'))
db=sqlite3.connect('file:'+ROOT+'/data/cache.db?mode=ro',uri=True)
ansi=re.compile(r'\x1b\[[0-9;]*m')
names=load_names()
def wrapper(s): return not is_bf_eligible(s)
def daily(s,day):
    n=db.execute("select count(*),min(bar_date) from daily_bars where symbol=? and bar_date<?",(s,day)).fetchone()
    prev=db.execute("select close,volume,bar_date from daily_bars where symbol=? and bar_date<? order by bar_date desc limit 1",(s,day)).fetchone()
    t=db.execute("select open from daily_bars where symbol=? and bar_date=?",(s,day)).fetchone()
    return n,prev,t
def reconcile(day):
    top=pre=None; nrank=None; subm=[]
    for ln in open(f'{ROOT}/logs/session_archive/{day}.log',errors='replace'):
        ln=ansi.sub('',ln)
        m=re.search(r'reconcile final top-(\d+) of (\d+) ranked: (\[.*?\]) \| preplaced: (\[.*?\])',ln)
        if m: nrank=int(m.group(2)); top=eval(m.group(3)); pre=eval(m.group(4))
        m=re.search(r'ORB ENTRY SUBMITTED: (\S+)',ln)
        if m: subm.append(m.group(1))
    return top,nrank,pre,subm
out=[]
for day,r in raw.items():
    top,nrank,pre,subm=reconcile(day)
    eng=r['eng']; bt=r['bt']
    prod={s for s,v in eng.items() if v['pool']=='production'}
    extra=set((pre or []))|set(subm)
    Eset=prod|{s for s in extra if eng.get(s,{}).get('pool','production')=='production'}
    notscored=sorted(Eset-prod)
    Bset={s for s,v in bt.items() if v['comp'] is not None and v['comp']>=thr and (v['gap'] or 0)>=5}
    eo=sorted(Eset-Bset); bo=sorted(Bset-Eset)
    print(f"\n== {day} E={len(Eset)} (scored {len(prod)} + preplaced/submitted-not-scored {notscored}) B(comp>=thr)={len(Bset)} reconcile_nranked={nrank} eng_top={top}")
    for tag,L in (('ENG-ONLY',eo),('BT-ONLY',bo)):
        for s in L:
            e=eng.get(s); b=bt.get(s)
            n,prev,t=daily(s,day)
            q=assign_quintile(b['comp'],cuts) if b and b['comp'] is not None else None
            print(f"  {tag} {s:6} eng={None if not e else (e['pool'],round(e['comp'],3),e['q'],int(e['rtv']),round(e['gap'],1))} bt={None if not b else (round(b['comp'],3) if b['comp'] is not None else None,q,int(b['rtv']),round(b['gap'],1),round(b['price'],2) if b['price'] else None,'ent' if b['entered'] else '-','pdr%.1f'%b['pdr'])} wrap={wrapper(s)} age_days={n[0]} prev={None if not prev else (prev[0],prev[1])} dayopen={None if not t else t[0]}")
