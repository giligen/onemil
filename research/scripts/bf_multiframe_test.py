"""Multi-timeframe BF detection test (owner: 'literature is full of BF
setups — 1-2/day'). Resample cached 1-min bars to 2/5-min, run the SAME
detector, enter next 1-min bar open after the higher-TF breakout close,
exit via the validated trail replay (stop=flag_low, trail A=2.0/T=1.0,
15:45 EOD). Months: 2026-06 (starved regime) + 2025-03 (healthy)."""
import sys
import pandas as pd, sqlite3
from datetime import time as dtime
sys.path.insert(0,'/home/ec2-user/onemil')
from persistence.database import Database
from trading.pattern_detector import BullFlagDetector

conn=sqlite3.connect('/home/ec2-user/onemil/data/cache.db')
uni=pd.read_sql("SELECT symbol, float_shares FROM universe", conn)
fmap=dict(zip(uni['symbol'],uni['float_shares']))
def eligible(month):
    q=f"""
    WITH d AS (SELECT symbol, bar_date, close, high, low,
        LAG(close) OVER (PARTITION BY symbol ORDER BY bar_date) pc
      FROM daily_bars WHERE bar_date >= date('{month}-01','-5 days') AND bar_date <= date('{month}-01','+35 days'))
    SELECT symbol, bar_date, close FROM d
    WHERE bar_date LIKE '{month}%' AND low>0 AND (high-low)/low>=0.10 AND pc>0
      AND (high>=pc*1.10 OR close>pc) AND close BETWEEN 2 AND 20"""
    m=pd.read_sql(q,conn)
    m['float']=m['symbol'].map(fmap)
    return m[(m['float'].isna())|(m['float']<=10_000_000)]

db=Database(db_path='/home/ec2-user/onemil/data/cache.db')
EXIT_SLIP=0.003; ENTRY_SLIP=0.005

def resample(df, minutes):
    d=df.set_index('timestamp')
    o=d.resample(f'{minutes}min').agg(open=('open','first'),high=('high','max'),
        low=('low','min'),close=('close','last'),volume=('volume','sum')).dropna()
    return o.reset_index()

def trail_replay(df1, entry_ts, entry_px, stop_px):
    t=df1['timestamp'].dt.tz_convert('America/New_York').dt.time
    post=df1[(df1['timestamp']>entry_ts)&(t<=dtime(15,45))]
    if post.empty: return None
    R=entry_px-stop_px
    if R<=0: return None
    stop=stop_px; hi=entry_px; on=False
    for _,b in post.iterrows():
        if float(b['low'])<=stop: return stop*(1-EXIT_SLIP)
        hi=max(hi,float(b['high']))
        if not on and hi>=entry_px+2.0*R: on=True
        if on: stop=max(stop,hi-1.0*R)
    return float(post.iloc[-1]['close'])*(1-EXIT_SLIP)

def run_month(month, tf_minutes):
    el=eligible(month)
    pairs=list({(r['symbol'],r['bar_date']) for _,r in el.iterrows()})
    raw=db.get_intraday_bars_bulk(pairs)
    det=BullFlagDetector()
    trades=[]
    for (sym,day) in pairs:
        bars=raw.get((sym,day))
        if not bars: continue
        df1=pd.DataFrame(bars); df1['timestamp']=pd.to_datetime(df1['timestamp'],utc=True)
        df1=df1.sort_values('timestamp').reset_index(drop=True)
        dfr=df1 if tf_minutes==1 else resample(df1, tf_minutes)
        if len(dfr)<7: continue
        for end in range(6, len(dfr)+1):
            try: pat=det.detect(sym, dfr, end_idx=end)
            except Exception: pat=None
            if pat is None: continue
            bko_ts=dfr.iloc[end-1]['timestamp']  # breakout bar close on TF
            nxt=df1[df1['timestamp']>bko_ts + pd.Timedelta(minutes=tf_minutes-1)]
            if nxt.empty: break
            entry_px=float(nxt.iloc[0]['open'])*(1+ENTRY_SLIP)
            stop=float(pat.flag_low) if hasattr(pat,'flag_low') else None
            if not stop or stop>=entry_px: break
            xp=trail_replay(df1, nxt.iloc[0]['timestamp'], entry_px, stop)
            if xp is None: break
            trades.append(dict(symbol=sym,day=day,tf=tf_minutes,
                entry=entry_px,exit=xp,ret=(xp-entry_px)/entry_px*100,
                risk_pct=(entry_px-stop)/entry_px*100))
            break   # one-shot per symbol-day (live parity)
    return trades

for month in ('2026-06','2025-03'):
    print(f"\n================ {month} ================")
    for tf in (1,2,5):
        tr=run_month(month, tf)
        t=pd.DataFrame(tr)
        if t.empty:
            print(f"TF {tf}min: 0 setups"); continue
        days=t['day'].nunique()
        # R-multiple economics: ret / risk
        t['r_mult']=t['ret']/t['risk_pct']
        print(f"TF {tf}min: {len(t)} setups ({len(t)/21:.1f}/day)  "
              f"WR {(t['ret']>0).mean()*100:.0f}%  sum-ret {t['ret'].sum():+.1f}%  "
              f"avg R {t['r_mult'].mean():+.2f}  med risk {t['risk_pct'].median():.1f}%")
conn.close()
print("\nMULTIFRAME TEST DONE")
