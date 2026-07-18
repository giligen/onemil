"""Exit-ladder counterfactual on the FULL selected book (bar-accurate)."""
import pandas as pd, numpy as np, sqlite3, sys
sys.stdout.reconfigure(line_buffering=True)
t=pd.read_csv('analysis_results/orb_static_lock_trades.csv')
t['date']=pd.to_datetime(t['date']); t['day']=t['date'].dt.strftime('%Y-%m-%d')
t['mult']=t['_sized_pnl']/t['_rp_pnl'].replace(0,np.nan)   # flat qmult x pm (1.0 or 2.0)
t['mult']=t['mult'].fillna(1.0)
conn=sqlite3.connect('data/cache.db',timeout=15)

def walk(sym,day):
    b=pd.read_sql("select timestamp,open,high,low,close from intraday_bars_1min where symbol=? and bar_date=? order by timestamp",conn,params=(sym,day))
    if b.empty: return None
    ts=pd.to_datetime(b['timestamp'],utc=True).dt.tz_convert('America/New_York')
    b['m']=ts.dt.hour*60+ts.dt.minute
    rng=b[(b['m']>=570)&(b['m']<575)]
    rh,rl=rng['high'].max(),rng['low'].min()
    entry=rh*1.003; R=entry-rl
    if not R>0: return None
    win=b[(b['m']>=575)&(b['m']<635)]; bo=win[win['high']>rh]
    if bo.empty: return None
    return entry,R,b[(b.index>=bo.index[0])&(b['m']<945)]

def ladder_exit(entry,R,post,tgt_r,be_at=None):
    """Sequential: stop -1R (or BE after be_at), target tgt_r. Same-bar -> stop first."""
    stop=entry-R
    for _,r_ in post.iterrows():
        if r_['low']<=stop: return (stop-entry)/R
        if be_at is not None and stop<entry and r_['high']>=entry+be_at*R:
            stop=entry
        if r_['high']>=entry+tgt_r*R: return tgt_r
    return (post.iloc[-1]['close']-entry)/R   # eod

paths={}
for _,r in t.iterrows():
    k=(r['symbol'],r['day'])
    if k not in paths: paths[k]=walk(*k)
print(f"paths: {sum(1 for v in paths.values() if v)} of {len(paths)}")

LADDERS=[('tgt_0.75R',dict(tgt_r=0.75)),('tgt_1R',dict(tgt_r=1.0)),
         ('tgt_1.25R',dict(tgt_r=1.25)),('tgt_1R_BE@0.5',dict(tgt_r=1.0,be_at=0.5))]
out={}
for name,kw in LADDERS:
    pnls=[]
    for _,r in t.iterrows():
        p=paths.get((r['symbol'],r['day']))
        if p is None:
            pnls.append(r['_sized_pnl']); continue   # keep book value if no bars
        entry,R,post=p
        rr=ladder_exit(entry,R,post,**kw)
        risk=r['_rp_position']*max(r['range_size_pct'],1.0)/100.0
        pnls.append(rr*risk*r['mult'])
    out[name]=pd.Series(pnls,index=t.index)
out['CURRENT lock']=t['_sized_pnl']

print(f"\n{'ladder':16s} {'JULY':>9s} {'TOT 18mo':>11s} {'25H1':>9s} {'25H2':>9s} {'2026':>10s} {'MDD':>9s} {'negMo':>6s}")
for name,s in out.items():
    jul=s[t['date']>='2026-07-01'].sum()
    daily=s.groupby(t['date']).sum().sort_index(); cum=daily.cumsum()
    ym=s.groupby(t['date'].dt.strftime('%Y-%m')).sum()
    eras=[s[(t['date']>=lo)&(t['date']<hi)].sum() for lo,hi in
          [('2025-01-01','2025-07-01'),('2025-07-01','2026-01-01'),('2026-01-01','2027-01-01')]]
    print(f"{name:16s} {jul:>+9,.0f} {s.sum():>+11,.0f} {eras[0]:>+9,.0f} {eras[1]:>+9,.0f} {eras[2]:>+10,.0f} {(cum-cum.cummax()).min():>+9,.0f} {(ym<0).sum():>3d}/19")
# monster damage for tgt_1R
s=out['tgt_1R']; dmg=t['_sized_pnl']-s
big=t[t['_sized_pnl']>=3000].copy(); big['after']=s[big.index]
print("\nmonsters (>= $3K sized) under tgt_1R:")
print(big[['symbol','day','_sized_pnl']].assign(after=big['after'].round(0)).round(0).to_string(index=False))
