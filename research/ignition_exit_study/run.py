#!/usr/bin/env python3
"""Ignition EXIT-DESIGN study (owner 2026-09-13: "relying on 3 monsters is not a
strategy"). The validated BT has meanR ~0, WR ~50%, and ALL its expectancy in
the top 3% of trades (78% of trades hold to the close for +0.21R, 19% stop at
-1R). Question: is there an exit design with POSITIVE expectancy EXCLUDING
R>=2 trades? Same entries, same bars, same friction as capsim; only the exit
walk changes. V0 must reproduce capsim's rr/reason before any row counts.

Pre-committed bar (per variant): meanR excluding R>=2 >= +0.05 AND months
green >= 5/8 (2026) AND total P&L > V0. Passing variants are then re-run on
2025H2 (env PART=25H2) — must be positive there too.
"""
import os, sys, sqlite3, json
import numpy as np, pandas as pd
ROOT='/home/ec2-user/onemil'; sys.path.insert(0, ROOT); os.chdir(ROOT)
import trading.ignition_rules as R
D='research/ignition_capcheck'; OUT='research/ignition_exit_study'
PART=os.environ.get('PART','2026')
TRADES={'2026': f'{D}/trades_NODOLLAR2026.csv', '25H2': f'{D}/trades_25H2.csv', '25H1': f'{D}/trades_25H1.csv', 'LIVEWIN': f'{D}/trades_LIVEWIN.csv'}[PART]
cache=sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
topup=sqlite3.connect(f'file:{D}/topup.db?mode=ro', uri=True, timeout=120)

def bars(day, syms):
    q=("select symbol, timestamp as t, open, high, low, close, volume from intraday_bars_1min "
       f"where bar_date=? and symbol in ({','.join('?'*len(syms))})")
    b=pd.read_sql(q, cache, params=[day]+syms); out={s:g for s,g in b.groupby('symbol')}
    left=[s for s in syms if s not in out or len(out[s])<20]
    if left:
        t=pd.read_sql("select symbol, t, o as open, h as high, l as low, c as close, v as volume from bars where day=?", topup, params=[day])
        for s,g in t[t.symbol.isin(left)].groupby('symbol'): out[s]=g
    res={}
    for s,g in out.items():
        ts=pd.to_datetime(g['t'], utc=True).dt.tz_convert('America/New_York')
        g=g.assign(m=ts.dt.hour*60+ts.dt.minute); res[s]=g[(g.m>=570)&(g.m<960)].sort_values('m').reset_index(drop=True)
    return res

def walk(post, entry, stop, entry_min, *, arm=R.ARM_R, lock=R.LOCK_R, partial_at=None, partial_frac=0.5,
         be_after_partial=True, trail_after_partial=None, cut_at=None, hold_max=None):
    """Return (rr on full position, reason). Partial: sells `partial_frac` at
    entry+partial_at*R (touch fill at the level); remainder: stop->entry if
    be_after_partial, then trail (R below high) if trail_after_partial else the
    lock rule. cut_at: at the first bar with m>=entry_min+cut_at, if close<entry
    exit at that close. hold_max: exit at the close of the bar m>=entry_min+hold_max."""
    Rd=entry-stop; cur=stop; armed=False; taken=False; rr_part=0.0; frac=1.0; hi=entry
    post=post[post.m>entry_min]
    for _,r in post.iterrows():
        if r.m>=R.EOD_FLAT_MIN: return rr_part+frac*(r.open-entry)/Rd, 'eod'
        if r.low<=cur:
            fill=min(cur,r.open); return rr_part+frac*(fill*0.999-entry)/Rd, ('lock' if armed else ('be' if taken else 'stop'))
        if partial_at is not None and not taken and r.high>=entry+partial_at*Rd:
            taken=True; rr_part=partial_frac*partial_at; frac=1-partial_frac
            if be_after_partial: cur=max(cur, entry)
        hi=max(hi, r.high)
        if taken and trail_after_partial is not None:
            cur=max(cur, hi-trail_after_partial*Rd)
        elif not armed and r.high>=entry+arm*Rd:
            armed=True; cur=entry+lock*Rd
        if cut_at is not None and r.m>=entry_min+cut_at and r.close<entry and not taken:
            return rr_part+frac*(r.close-entry)/Rd, 'cut'
        if hold_max is not None and r.m>=entry_min+hold_max:
            return rr_part+frac*(r.close-entry)/Rd, 'hold'
    if len(post): return rr_part+frac*(post.iloc[-1].close-entry)/Rd, 'eod'
    return 0.0,'none'

VARIANTS={
 'V0_baseline': dict(),
 'V1_lock1.0_0.3': dict(arm=1.0, lock=0.3),
 'V2_partial1R_BE': dict(partial_at=1.0),
 'V3_partial1R_trail1R': dict(partial_at=1.0, trail_after_partial=1.0),
 'V4_cut30': dict(cut_at=30),
 'V5_cut60': dict(cut_at=60),
 'V6_hold60': dict(hold_max=60),
 'V7_partial1R_BE_cut30': dict(partial_at=1.0, cut_at=30),
 'V8_partial0.75R_trail0.75': dict(partial_at=0.75, trail_after_partial=0.75),
}
t=pd.read_csv(TRADES); t=t[t.rr.notna()]
rows=[]; days=sorted(t.day.unique()); print(f'[{PART}] {len(t)} trades over {len(days)} days', flush=True)
for i,day in enumerate(days):
    sub=t[t.day==day]; B=bars(day, sub.symbol.tolist())
    for r in sub.itertuples():
        g=B.get(r.symbol)
        if g is None or len(g)<20: continue
        Rd=r.entry-r.stop; rp=r.r_pct; pos=r.pos
        fric=pos*R.FRICTION_BPS*min((pos/max(r.bar_dollar,1))/R.PARTICIPATION,1.0) if 'bar_dollar' in t.columns else 0.0
        row={'day':day,'symbol':r.symbol,'rr_bt':r.rr,'reason_bt':r.reason}
        for nm,kw in VARIANTS.items():
            rr,why=walk(g, r.entry, r.stop, int(r.trig_m), **kw)
            row[nm+'_rr']=rr; row[nm+'_pnl']=pos*(rr*rp/100.0)-fric; row[nm+'_why']=why
        rows.append(row)
    if i%20==0: print(f'[{PART}] {i+1}/{len(days)} days, {len(rows)} trades', flush=True)
df=pd.DataFrame(rows); df.to_csv(f'{OUT}/trades_{PART}.csv', index=False)
base_ok=(df.V0_baseline_rr-df.rr_bt).abs().max()
print(f'[{PART}] V0 reproduces capsim: max |rr diff| = {base_ok:.4f} (reason mismatches {(df.V0_baseline_why!=df.reason_bt).sum()})', flush=True)
df['mo']=df.day.str[:7]; S=[]
for nm in VARIANTS:
    rr=df[nm+'_rr']; pnl=df[nm+'_pnl']; nm_=~(rr>=2); m=df.groupby('mo')[nm+'_pnl'].sum()
    S.append(dict(variant=nm, n=len(df), meanR=round(rr.mean(),3), WR=round((rr>0).mean()*100,1), pnl=round(pnl.sum()),
                  meanR_exMonster=round(rr[nm_].mean(),3), pnl_exMonster=round(pnl[nm_].sum()), monsters=int((rr>=2).sum()),
                  months_green=f"{int((m>0).sum())}/{len(m)}", worst_mo=round(m.min()), exits=dict(df[nm+'_why'].value_counts())))
S=pd.DataFrame(S); base=S.iloc[0]
S['verdict']=['' if i==0 else ('PASS' if (s.meanR_exMonster>=0.05 and int(s.months_green.split('/')[0])>=5 and s.pnl>base.pnl) else 'fail') for i,s in S.iterrows()]
S.to_csv(f'{OUT}/summary_{PART}.csv', index=False); pd.set_option('display.width',250); print(S.drop(columns=['exits']).to_string(index=False), flush=True)
print('\nexit mix:', {s.variant: s.exits for _,s in S.iterrows()}, flush=True)
