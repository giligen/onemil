#!/usr/bin/env python3
"""Owner 2026-09-13: "coin flip; risk:reward should be 1:2 — move risk to 0.5R?"
Same entries/bars as run.py; the initial stop is tightened to STOP_MULT x the
structural R (0.5 / 0.75), the +1R partial target stays at the OLD +1R (=> +2R
/ +1.33R in new units) with stop->breakeven, remainder on the lock rule
(re-expressed in new R). Position re-sized for the new stop with capsim's
position_usd (same $ risk, same participation cap). Metrics in DOLLARS at
equal risk so the R rescaling cannot flatter anything; "tail" = trades with
pnl >= 2x the trade's $ risk."""
import os, sys, sqlite3
import pandas as pd
ROOT='/home/ec2-user/onemil'; sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, f'{ROOT}/research/ignition_exit_study')
import trading.ignition_rules as R
from run import bars, walk, TRADES, PART, OUT, D
t=pd.read_csv(TRADES); t=t[t.rr.notna()]
STOPS=[1.0, 0.75, 0.5]
rows=[]; days=sorted(t.day.unique()); print(f'[{PART}] {len(t)} trades / {len(days)} days', flush=True)
for i,day in enumerate(days):
    sub=t[t.day==day]; B=bars(day, sub.symbol.tolist())
    for r in sub.itertuples():
        g=B.get(r.symbol)
        if g is None or len(g)<20: continue
        R_old=r.entry-r.stop; row={'day':day,'symbol':r.symbol}
        for sm in STOPS:
            stop=r.entry-sm*R_old; rp=R.r_pct_from_stop(r.entry, stop); pos=R.position_usd(rp, r.bar_dollar)
            fric=pos*R.FRICTION_BPS*min((pos/max(r.bar_dollar,1))/R.PARTICIPATION,1.0)
            risk_usd=pos*rp/100.0
            for nm,kw in (('hold', dict()), ('p1old', dict(partial_at=1.0/sm)), ('p1new', dict(partial_at=1.0))):
                rr,why=walk(g, r.entry, stop, int(r.trig_m), **kw)
                tag=f's{int(sm*100)}_{nm}'; row[tag+'_pnl']=pos*(rr*rp/100.0)-fric; row[tag+'_risk']=risk_usd; row[tag+'_rr']=rr; row[tag+'_why']=why
        rows.append(row)
    if i%25==0: print(f'[{PART}] {i+1}/{len(days)} days', flush=True)
df=pd.DataFrame(rows); df.to_csv(f'{OUT}/stop_trades_{PART}.csv', index=False)
ann=pd.read_csv(f'{D}/trades_NODOLLAR2026_annotated.csv' if PART=='2026' else f'{D}/trades_all_annotated.csv', low_memory=False)[['day','symbol','complex_conf']]
df=df.merge(ann,on=['day','symbol'],how='left'); df['cc']=df.complex_conf.astype(str).eq('True'); df['mo']=df.day.str[:7]
def summ(d, label):
    out=[]
    for sm in STOPS:
        for nm in ('hold','p1old','p1new'):
            tag=f's{int(sm*100)}_{nm}'; pnl=d[tag+'_pnl']; risk=d[tag+'_risk']; tail=pnl>=2*risk; m=d.groupby('mo')[tag+'_pnl'].sum()
            out.append(dict(set=label, stop=sm, exit=nm, n=len(d), WR=round((pnl>0).mean()*100,1), pnl=round(pnl.sum()), pnl_per_risk=round((pnl/risk).mean(),3),
                            pnl_ex_tail=round(pnl[~tail].sum()), per_risk_ex_tail=round((pnl[~tail]/risk[~tail]).mean(),3), tail_n=int(tail.sum()),
                            stops=int((d[tag+'_why']=='stop').sum()), green=f"{int((m>0).sum())}/{len(m)}", worst=round(m.min())))
    return pd.DataFrame(out)
S=pd.concat([summ(df[df.cc],'CC'), summ(df,'ALL')]); S.to_csv(f'{OUT}/stop_summary_{PART}.csv', index=False)
pd.set_option('display.width',250); print(S.to_string(index=False), flush=True)
