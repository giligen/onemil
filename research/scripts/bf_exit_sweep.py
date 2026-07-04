"""BF exit-parameter sweep on the Stage-2 selected set (never swept).

Replays 1-min bars from each trade's entry with parameterized exits:
  hard stop = stop_loss (cache), trail activates at A x R, trails at T x R
  (R = entry - stop). EOD close at 15:45 if nothing hit.
VALIDATION GATE: baseline (A=1.5, T=1.0) must reproduce cache exit prices
within tolerance on most trades, else STOP (harness untrusted).
Vol-confirmed trail + exhaustion partials NOT replayed (both validated
separately; noted as approximation).
Walk-forward: choose on 2025, confirm on 2026.
"""
import sys
import pandas as pd
from datetime import timedelta, time as dtime
sys.path.insert(0,'/home/ec2-user/onemil')
from persistence.database import Database

sel=pd.read_csv('/tmp/bf_stage2_selected.csv')
sel=sel.dropna(subset=['entry_time_et','entry_price','stop_loss','date'])
print(f"Stage-2 selected trades: {len(sel)}")
db=Database(db_path='/home/ec2-user/onemil/data/cache.db')
pairs=[(r['symbol'], str(r['date'])[:10]) for _,r in sel.iterrows()]
raw=db.get_intraday_bars_bulk(pairs)

ENTRY_SLIP=0.005; EXIT_SLIP=0.003

def replay(sym, day, entry_et, entry_px, stop_px, act_r, trail_r):
    bars=raw.get((sym,day))
    if not bars: return None
    df=pd.DataFrame(bars); df['timestamp']=pd.to_datetime(df['timestamp'],utc=True)
    df=df.sort_values('timestamp').reset_index(drop=True)
    et=df['timestamp'].dt.tz_convert('America/New_York')
    df['et_t']=et.dt.time
    try:
        h,m,s2=(entry_et.split(':')+['0'])[:3]
        et_entry=dtime(int(h),int(m))
    except Exception:
        return None
    post=df[df['et_t']>=et_entry]
    post=post[post['et_t']<=dtime(15,45)]
    if post.empty: return None
    R=entry_px-stop_px
    if R<=0: return None
    stop=stop_px; hi=entry_px; trail_on=False
    for _,b in post.iterrows():
        lo_,hi_=float(b['low']),float(b['high'])
        if lo_<=stop:
            return stop*(1-EXIT_SLIP)
        hi=max(hi,hi_)
        if not trail_on and hi>=entry_px+act_r*R:
            trail_on=True
        if trail_on:
            stop=max(stop, hi-trail_r*R)
    return float(post.iloc[-1]['close'])*(1-EXIT_SLIP)

def run(act_r, trail_r):
    outs=[]
    for _,r in sel.iterrows():
        px=replay(r['symbol'],str(r['date'])[:10],str(r['entry_time_et']),
                  float(r['entry_price']),float(r['stop_loss']),act_r,trail_r)
        if px is None: outs.append(None); continue
        shares=r['shares'] if 'shares' in r and pd.notna(r.get('shares')) else None
        pnl=(px-r['entry_price'])* (shares if shares else 1000/r['entry_price']*5)  # fallback sizing
        outs.append((px,pnl))
    return outs

# VALIDATION: baseline vs cache exit prices
base=run(1.5,1.0)
ok=tot=0; diffs=[]
for o,(_,r) in zip(base, sel.iterrows()):
    if o is None: continue
    tot+=1
    d=abs(o[0]-r['exit_price'])/r['exit_price']
    diffs.append(d)
    if d<0.02: ok+=1
print(f"VALIDATION: {ok}/{tot} trades within 2% of cache exit price "
      f"(median diff {pd.Series(diffs).median()*100:.2f}%)")
if tot==0 or ok/tot<0.7:
    print("HARNESS UNTRUSTED — stopping (approximations too large)")
    sys.exit(1)

sel['date']=pd.to_datetime(sel['date'])
is25=sel['date']<'2026-01-01'
def score(res, mask):
    s=0
    for o,(_,r),m in zip(res, sel.iterrows(), mask):
        if o is None or not m: continue
        s+=(o[0]-r['entry_price'])/r['entry_price']
    return s*100  # sum of returns in %
print(f"\n{'A/T':>10} {'2025 sum%':>10} {'2026 sum%':>10}")
grid=[(1.0,0.5),(1.0,1.0),(1.5,0.5),(1.5,1.0),(1.5,1.5),(2.0,1.0),(2.0,1.5),(1.0,1.5),(2.5,1.0)]
for a,t in grid:
    res=run(a,t)
    print(f"{a}/{t:>5} {score(res,is25):>+10.1f} {score(res,~is25):>+10.1f}", flush=True)
print("SWEEP DONE")
