"""Detection kill-table: run PatternDetector over sampled eligible movers,
capture debug rejection reasons, compare 2025 vs 2026 mix under SAME code."""
import io, logging, random, re, sys
import pandas as pd, sqlite3
sys.path.insert(0,'/home/ec2-user/onemil')
from persistence.database import Database
from trading.pattern_detector import BullFlagDetector

random.seed(42)
conn=sqlite3.connect('/home/ec2-user/onemil/data/cache.db')
uni=pd.read_sql("SELECT symbol, float_shares FROM universe", conn)
fmap=dict(zip(uni['symbol'],uni['float_shares']))
q="""
WITH d AS (
  SELECT symbol, bar_date, close, high, low,
         LAG(close) OVER (PARTITION BY symbol ORDER BY bar_date) pc
  FROM daily_bars WHERE bar_date >= '2025-01-01'
)
SELECT symbol, bar_date, close FROM d
WHERE low>0 AND (high-low)/low>=0.10 AND pc>0 AND (high>=pc*1.10 OR close>pc)
  AND close BETWEEN 2 AND 20
"""
m=pd.read_sql(q,conn); conn.close()
m['float']=m['symbol'].map(fmap)
m=m[(m['float'].isna())|(m['float']<=10_000_000)]
s25=m[m['bar_date'].str.startswith('2025')].sample(250, random_state=1)
s26=m[m['bar_date'].str.startswith('2026')].sample(250, random_state=2)
db=Database(db_path='/home/ec2-user/onemil/data/cache.db')

log=logging.getLogger('trading.pattern_detector')
def scan(sample):
    reasons={}
    setups=0; nodata=0
    pairs=[(r['symbol'], r['bar_date']) for _,r in sample.iterrows()]
    raw=db.get_intraday_bars_bulk(pairs)
    for (sym,day) in pairs:
        bars=raw.get((sym,day))
        if not bars: nodata+=1; continue
        df=pd.DataFrame(bars)
        df['timestamp']=pd.to_datetime(df['timestamp'],utc=True)
        df=df.sort_values('timestamp').reset_index(drop=True)
        det=BullFlagDetector()
        buf=io.StringIO(); h=logging.StreamHandler(buf); h.setLevel(logging.DEBUG)
        log.addHandler(h); old=log.level; log.setLevel(logging.DEBUG)
        found=False
        # slide through the day like live: check at each bar index
        for end in range(6, len(df)+1, 3):   # every 3 bars for speed
            try:
                res=det.detect(sym, df, end_idx=end)
            except Exception:
                res=None
            if res: found=True; break
        log.removeHandler(h); log.setLevel(old)
        if found: setups+=1; continue
        # last rejection reason for this symbol-day (the terminal blocker)
        lines=[l for l in buf.getvalue().splitlines() if l.startswith(f"{sym}:")]
        if lines:
            reason=re.sub(r'[0-9.]+','N',lines[-1].split(': ',1)[1])[:60]
            reasons[reason]=reasons.get(reason,0)+1
        else:
            reasons['(no debug line — pre-gate)']=reasons.get('(no debug line — pre-gate)',0)+1
    return setups, nodata, reasons

for label, s in (('2025',s25),('2026',s26)):
    setups,nodata,reasons=scan(s)
    print(f"\n=== {label}: {len(s)} eligible movers sampled ===")
    print(f"setups detected: {setups} ({setups/(len(s)-nodata)*100:.1f}% of {len(s)-nodata} with bars; {nodata} no-bars)")
    for k,v in sorted(reasons.items(), key=lambda x:-x[1])[:10]:
        print(f"  {v:>4}  {k}")
