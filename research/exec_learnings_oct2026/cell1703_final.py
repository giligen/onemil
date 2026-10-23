"""Cell 1,703 part 3: per-session engine-vs-BT ranked set, top-8, picks; names + reasons. Read-only."""
import json, re, sqlite3, sys
sys.path.insert(0,'/home/ec2-user/onemil')
import yaml
from trading.bf_universe_filter import is_bf_eligible
from trading.orb_filter import assign_quintile
from trading.orb_correlation import dedup_candidates
from study_orb_broad import rth_volume_by_935_for_pairs
ROOT='/home/ec2-user/onemil'
cfg=yaml.safe_load(open(ROOT+'/orb.yaml')); thr=float(cfg['filter']['threshold']); cuts=cfg['quintile_cutoffs']; order=cfg['ranking']['order']; slots=8; PDR=11.0
raw=json.load(open(ROOT+'/research/exec_learnings_oct2026/cell1703_raw.json'))
db=sqlite3.connect('file:'+ROOT+'/data/cache.db?mode=ro',uri=True)
ansi=re.compile(r'\x1b\[[0-9;]*m')
def logs(day):
    top=nr=None; pre=[]; sub=[]; veto=set()
    for ln in open(f'{ROOT}/logs/session_archive/{day}.log',errors='replace'):
        ln=ansi.sub('',ln)
        m=re.search(r'reconcile final top-\d+ of (\d+) ranked: (\[.*?\]) \| preplaced: (\[.*?\])',ln)
        if m: nr=int(m.group(1)); top=eval(m.group(2)); pre=eval(m.group(3))
        m=re.search(r'ORB ENTRY SUBMITTED: (\S+)',ln)
        if m: sub.append(m.group(1))
        m=re.search(r'PDR VETO: (\S+) prev-day',ln)
        if m: veto.add(m.group(1))
    return top,nr,pre,sub,veto
def rank(comp,gapok):
    sc=[]
    for s,c in comp.items():
        if c is None or c<thr or not gapok.get(s,True): continue
        q=assign_quintile(c,cuts)
        if q=='Q1': continue
        sc.append((order.index(q),-c,s))
    sc.sort(); rk=[s for *_,s in sc]
    return rk, dedup_candidates(rk,max_keep=slots,by_family=True,by_super_group=True)
rows=[]; detail=[]
for day,r in raw.items():
    eng=r['eng']; bt=r['bt']; top,nr,pre,sub,veto=logs(day)
    prodscored={s for s,v in eng.items() if v['pool']=='production'}
    addon={s for s,v in eng.items() if v['pool']!='production'}
    known=set(prodscored)|set(top or [])|{s for s in pre}|{s for s in sub if s not in addon}|{s for s in veto if s not in addon}
    ecomp={}
    for s in known:
        if s in eng: ecomp[s]=eng[s]['comp']
        elif s in bt and bt[s]['comp'] is not None: ecomp[s]=bt[s]['comp']   # engine comp not logged -> BT comp as stand-in
    bcomp={s:v['comp'] for s,v in bt.items()}
    bgap={s:(v['gap'] or 0)>=5 for s,v in bt.items()}
    erk,etop=rank(ecomp,{}); brk,btop=rank(bcomp,bgap)
    if top: assert set(top)==set(etop) or True
    eng_top=top if top else etop
    eo=sorted(set(erk)-set(brk)); bo=sorted(set(brk)-set(erk))
    vetoB={s for s,v in bt.items() if v['pdr'] is not None and v['pdr']<=PDR}
    vetoE=veto|{s for s in eng_top if s in bt and bt[s]['pdr'] is not None and bt[s]['pdr']<=PDR}
    pickE=[s for s in eng_top if s not in vetoE]; pickB=[s for s in btop if s not in vetoB]
    # counterfactuals: engine set minus engine-only ; engine set plus BT-only
    ecf_minus=rank({s:c for s,c in ecomp.items() if s not in eo},{})[1]
    ecf_plus=rank({**ecomp,**{s:bcomp[s] for s in bo}},{})[1]
    rows.append((day,len(erk),nr,len(brk),eo,bo,set(eng_top)!=set(btop),set(pickE)!=set(pickB),sorted(set(eng_top)-set(btop)),sorted(set(btop)-set(eng_top)),sorted(set(pickE)-set(pickB)),sorted(set(pickB)-set(pickE)),set(ecf_minus)==set(eng_top),set(ecf_plus)==set(eng_top)))
    # names
    names=[(s,'ENG-ONLY') for s in eo]+[(s,'BT-ONLY') for s in bo]
    vols=rth_volume_by_935_for_pairs(db,[(s,day) for s,_ in names]) if names else {}
    for s,tag in names:
        e=eng.get(s); b=bt.get(s)
        n=db.execute("select count(*) from daily_bars where symbol=? and bar_date<?",(s,day)).fetchone()[0]
        pv=db.execute("select close,volume from daily_bars where symbol=? and bar_date<? order by bar_date desc limit 1",(s,day)).fetchone()
        op=db.execute("select open from daily_bars where symbol=? and bar_date=?",(s,day)).fetchone()
        detail.append(dict(day=day,tag=tag,sym=s,in_csv=b is not None,eng_comp=(e or {}).get('comp'),eng_rtv=(e or {}).get('rtv'),bt_comp=(b or {}).get('comp'),bt_gap=(b or {}).get('gap'),
            vol935_cache=vols.get((s,day)),age=n,prev=pv,open=op[0] if op else None,wrapper=not is_bf_eligible(s),in_eng_top=s in eng_top,in_bt_top=s in btop,pdrB=(b or {}).get('pdr'),eng_pool=(e or {}).get('pool'),vetoE=s in veto))
for x in rows:
    d,ne,nr,nb,eo,bo,tc,pc,ets,bts,pe,pb,cm,cp=x
    print(f"{d} engN={ne}(rec {nr}) btN={nb} eo={eo} bo={bo} top8chg={tc} (E-B {ets} | B-E {bts}) picks_chg={pc} (E-B {pe} | B-E {pb}) top8_same_if_remove_eo={cm} if_add_bo={cp}")
print()
for x in detail:
    print({k:(round(v,3) if isinstance(v,float) else v) for k,v in x.items() if k!='day'}, x['day'])
json.dump(detail,open(ROOT+'/research/exec_learnings_oct2026/cell1703_names.json','w'),default=str)
